//! Direct, bounded execution for explicitly supported CPU-local native work.
//!
//! This module accepts CPU-local Nagi/Laya decisions, standard Breeze speech,
//! and embeddings through the existing language adapter. It does not start
//! services, install runtimes, select a fallback, or retry an attempted
//! inference. Native input and output stay out of its errors.

use std::error::Error;
use std::fmt;
use std::path::Path;
use std::time::Duration;

use super::catalog::{Catalog, Device, Lane};
use super::decision::{
    execute_accepted_cpu_decision, DecisionExecutionError, NativeDecisionResult,
};
use super::language::{DispatchDisposition, LanguageAdapter, LanguageErrorKind, LanguageFailure};
use super::media_execution::{
    execute_breeze_speech_on_cpu, MediaExecutionErrorKind, MediaExecutionFailure,
    MediaExecutionResult,
};
use super::native_commands::CPU_BREEZE_PROFILE_ID;
use super::process::{ProcessFailureDisposition, ProcessFailureKind};
use super::requests::{
    EmbeddingPayload, InferenceRequest, InferenceResult, MediaInput, SpeechMode,
};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::{OperationContext, RecoveryPolicy};

const PROCESS_RECOVERY_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of one direct native operation. The payloads intentionally do not
/// implement `Debug`: native answers and filesystem paths are not log data.
pub enum DirectRunResult {
    Decision(NativeDecisionResult),
    Speech(MediaExecutionResult),
    Embeddings(EmbeddingPayload),
}

/// Fixed failure categories with no request body, path, runner output, or OS
/// error retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectRunErrorKind {
    InvalidRequest,
    CatalogUnavailable,
    UnsupportedRoute,
    Cancelled,
    DeadlineExceeded,
    ExecutionFailed,
    InvalidOutput,
    OwnershipUncertain,
    RemoteCompletionUncertain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectRunDisposition {
    Process(ProcessFailureDisposition),
    Remote(DispatchDisposition),
}

/// A redacted failure plus whether local ownership or remote completion is uncertain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectRunError {
    kind: DirectRunErrorKind,
    disposition: DirectRunDisposition,
}

impl DirectRunError {
    fn new(kind: DirectRunErrorKind, disposition: ProcessFailureDisposition) -> Self {
        Self {
            kind,
            disposition: DirectRunDisposition::Process(disposition),
        }
    }

    fn remote(kind: DirectRunErrorKind, disposition: DispatchDisposition) -> Self {
        let kind = if disposition == DispatchDisposition::Uncertain {
            DirectRunErrorKind::RemoteCompletionUncertain
        } else {
            kind
        };
        Self {
            kind,
            disposition: DirectRunDisposition::Remote(disposition),
        }
    }

    #[cfg(test)]
    pub fn kind(self) -> DirectRunErrorKind {
        self.kind
    }

    #[cfg(test)]
    fn disposition(self) -> DirectRunDisposition {
        self.disposition
    }

    /// True only when cancellation is terminal: no child was spawned, the
    /// owned child group was reaped, or a remote request was never dispatched.
    pub fn cancellation_confirmed(self) -> bool {
        self.kind == DirectRunErrorKind::Cancelled
            && matches!(
                self.disposition,
                DirectRunDisposition::Process(
                    ProcessFailureDisposition::NoChildSpawned
                        | ProcessFailureDisposition::ChildGroupReaped,
                ) | DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
            )
    }

    /// Callers must keep execution serialized when a child group or remote
    /// request may still be active.
    pub fn blocks_following_work(self) -> bool {
        matches!(
            self.disposition,
            DirectRunDisposition::Process(ProcessFailureDisposition::OwnershipUncertain)
                | DirectRunDisposition::Remote(DispatchDisposition::Uncertain)
        )
    }
}

impl fmt::Display for DirectRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.disposition == DirectRunDisposition::Remote(DispatchDisposition::Uncertain) {
            return formatter.write_str("remote completion is unconfirmed; stop before continuing");
        }
        formatter.write_str(match self.kind {
            DirectRunErrorKind::InvalidRequest => "direct native request is invalid",
            DirectRunErrorKind::CatalogUnavailable => "native catalog is unavailable",
            DirectRunErrorKind::UnsupportedRoute => "direct native route is unsupported",
            DirectRunErrorKind::Cancelled => "native operation was cancelled",
            DirectRunErrorKind::DeadlineExceeded => "native operation timed out",
            DirectRunErrorKind::ExecutionFailed => "native execution failed",
            DirectRunErrorKind::InvalidOutput => "native output is invalid",
            DirectRunErrorKind::OwnershipUncertain => {
                "native process cleanup is uncertain; stop before continuing"
            }
            DirectRunErrorKind::RemoteCompletionUncertain => {
                "remote completion is unconfirmed; stop before continuing"
            }
        })
    }
}

impl Error for DirectRunError {}

/// Execute one already-built native request on a supported direct route.
///
/// The current catalog is refreshed before dispatch, then the accepted route
/// is rechecked by the relevant adapter. CPU-local decisions and standard
/// speech use the bounded process runner; embeddings use the caller's existing
/// language adapter and configured endpoint without managing llama-swap.
/// Speech requires a caller-owned fresh output directory; other lanes reject
/// one. Process cleanup is bounded by a fixed five-second recovery budget.
/// Uncertain process ownership or remote completion blocks following work.
pub fn run_native(
    catalog: &Catalog,
    request: &InferenceRequest,
    speech_output_directory: Option<&Path>,
    language: Option<&LanguageAdapter>,
    context: &OperationContext<'_>,
) -> Result<DirectRunResult, DirectRunError> {
    let is_decision = matches!(request, InferenceRequest::Decision { .. });
    let is_embeddings = matches!(request, InferenceRequest::Embeddings { .. });
    let is_standard_speech = matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } if input.mode == SpeechMode::Standard
    );
    let valid_lane_arguments = match request {
        InferenceRequest::Decision { .. } => {
            speech_output_directory.is_none() && language.is_none()
        }
        InferenceRequest::Media { .. } if is_standard_speech => {
            speech_output_directory.is_some() && language.is_none()
        }
        InferenceRequest::Embeddings { .. } => {
            speech_output_directory.is_none() && language.is_some()
        }
        _ => false,
    };
    if !valid_lane_arguments {
        return Err(unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_embeddings,
        ));
    }
    if context.check_active("direct native operation").is_err() {
        return Err(if is_embeddings {
            inactive_remote_error(context)
        } else {
            inactive_error(context)
        });
    }

    let original_profile = catalog
        .inspect(request.profile())
        .ok_or_else(|| unstarted_error(DirectRunErrorKind::InvalidRequest, is_embeddings))?;
    let catalog = catalog
        .refresh()
        .map_err(|_| unstarted_error(DirectRunErrorKind::CatalogUnavailable, is_embeddings))?;
    if catalog.inspect(request.profile()) != Some(original_profile) {
        return Err(unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_embeddings,
        ));
    }
    request
        .validate(&catalog)
        .map_err(|_| unstarted_error(DirectRunErrorKind::InvalidRequest, is_embeddings))?;
    let route = resolve_profile_route(&catalog, request.profile())
        .map_err(|_| unstarted_error(DirectRunErrorKind::InvalidRequest, is_embeddings))?;
    let recovery = RecoveryPolicy::new(PROCESS_RECOVERY_TIMEOUT)
        .expect("fixed process recovery timeout is valid");
    if is_decision {
        if !is_cpu_local_nagi_or_laya(request, &route) {
            return Err(DirectRunError::new(
                DirectRunErrorKind::UnsupportedRoute,
                ProcessFailureDisposition::NoChildSpawned,
            ));
        }
        execute_accepted_cpu_decision(&catalog, request, &route, context, recovery)
            .map(DirectRunResult::Decision)
            .map_err(|error| execution_error(error, context))
    } else if is_standard_speech {
        if !is_cpu_breeze_standard_route(request, &route) {
            return Err(DirectRunError::new(
                DirectRunErrorKind::UnsupportedRoute,
                ProcessFailureDisposition::NoChildSpawned,
            ));
        }
        let output_directory = speech_output_directory.expect("validated speech output directory");
        execute_breeze_speech_on_cpu(
            &catalog,
            request,
            &route,
            output_directory,
            context,
            recovery,
        )
        .map(DirectRunResult::Speech)
        .map_err(|error| media_execution_error(error, context))
    } else if is_embeddings {
        if !is_language_embeddings_route(request, &route) {
            return Err(DirectRunError::remote(
                DirectRunErrorKind::UnsupportedRoute,
                DispatchDisposition::NoDispatch,
            ));
        }
        let adapter = language.expect("validated embeddings adapter");
        let result = adapter
            .execute_embeddings_for_route(&catalog, request, &route, context)
            .map_err(remote_language_error)?;
        match result {
            InferenceResult::Embeddings { vectors, usage } => {
                Ok(DirectRunResult::Embeddings(EmbeddingPayload {
                    vectors,
                    usage,
                }))
            }
            _ => Err(DirectRunError::remote(
                DirectRunErrorKind::InvalidOutput,
                DispatchDisposition::Uncertain,
            )),
        }
    } else {
        Err(unstarted_error(DirectRunErrorKind::InvalidRequest, false))
    }
}

fn is_cpu_local_nagi_or_laya(request: &InferenceRequest, route: &ResolvedProfileRoute) -> bool {
    let InferenceRequest::Decision { profile, .. } = request else {
        return false;
    };
    let NativeRoute::DecisionRunner { profile_type, .. } = &route.native_route else {
        return false;
    };
    profile == &route.profile
        && route.profile.lane == Lane::Decisions
        && route.runtime.lane == Lane::Decisions
        && matches!(route.runtime.as_str(), "nagi" | "laya")
        && route.device.as_ref() == Some(&Device::Cpu)
        && profile_type == "local"
}

fn is_language_embeddings_route(request: &InferenceRequest, route: &ResolvedProfileRoute) -> bool {
    let InferenceRequest::Embeddings { profile, .. } = request else {
        return false;
    };
    let NativeRoute::LlamaSwapModel { model_id, .. } = &route.native_route else {
        return false;
    };
    profile == &route.profile
        && route.profile.lane == Lane::Language
        && route.runtime.lane == Lane::Language
        && model_id == &route.profile.profile_id
        && route.tasks.iter().any(|task| task.as_str() == "embeddings")
}

fn is_cpu_breeze_standard_route(request: &InferenceRequest, route: &ResolvedProfileRoute) -> bool {
    let InferenceRequest::Media {
        profile,
        input: MediaInput::Speech(input),
    } = request
    else {
        return false;
    };
    matches!(
        &route.native_route,
        NativeRoute::MediaScript { path }
            if path.file_name().and_then(|name| name.to_str()) == Some("generate-breeze-tts-2.sh")
    ) && profile == &route.profile
        && profile.lane == Lane::Media
        && profile.profile_id.as_str() == CPU_BREEZE_PROFILE_ID
        && route.runtime.lane == Lane::Media
        && route.runtime.as_str() == "audio.cpp"
        && route.device.as_ref() == Some(&Device::Cpu)
        && input.mode == SpeechMode::Standard
        && input.instruction.is_none()
}

fn inactive_error(context: &OperationContext<'_>) -> DirectRunError {
    let kind = if context.is_cancelled() {
        DirectRunErrorKind::Cancelled
    } else {
        DirectRunErrorKind::DeadlineExceeded
    };
    DirectRunError::new(kind, ProcessFailureDisposition::NoChildSpawned)
}

fn inactive_remote_error(context: &OperationContext<'_>) -> DirectRunError {
    let kind = if context.is_cancelled() {
        DirectRunErrorKind::Cancelled
    } else {
        DirectRunErrorKind::DeadlineExceeded
    };
    DirectRunError::remote(kind, DispatchDisposition::NoDispatch)
}

fn unstarted_error(kind: DirectRunErrorKind, remote: bool) -> DirectRunError {
    if remote {
        DirectRunError::remote(kind, DispatchDisposition::NoDispatch)
    } else {
        DirectRunError::new(kind, ProcessFailureDisposition::NoChildSpawned)
    }
}

fn remote_language_error(error: LanguageFailure) -> DirectRunError {
    let kind = match error.kind {
        LanguageErrorKind::Configuration | LanguageErrorKind::InvalidRequest => {
            DirectRunErrorKind::InvalidRequest
        }
        LanguageErrorKind::Cancelled => DirectRunErrorKind::Cancelled,
        LanguageErrorKind::Deadline => DirectRunErrorKind::DeadlineExceeded,
        LanguageErrorKind::Protocol | LanguageErrorKind::TooLarge => {
            DirectRunErrorKind::InvalidOutput
        }
        LanguageErrorKind::Transport
        | LanguageErrorKind::Rejected
        | LanguageErrorKind::ResultPublication => DirectRunErrorKind::ExecutionFailed,
    };
    DirectRunError::remote(kind, error.disposition)
}

fn execution_error(
    error: DecisionExecutionError,
    context: &OperationContext<'_>,
) -> DirectRunError {
    match error {
        DecisionExecutionError::NotDispatched => {
            if context.check_active("direct native operation").is_err() {
                inactive_error(context)
            } else {
                DirectRunError::new(
                    DirectRunErrorKind::ExecutionFailed,
                    ProcessFailureDisposition::NoChildSpawned,
                )
            }
        }
        DecisionExecutionError::InvalidResult(_) => DirectRunError::new(
            DirectRunErrorKind::InvalidOutput,
            ProcessFailureDisposition::ChildGroupReaped,
        ),
        DecisionExecutionError::Uncertain { process } => {
            let kind = match process {
                ProcessFailureDisposition::OwnershipUncertain => {
                    DirectRunErrorKind::OwnershipUncertain
                }
                ProcessFailureDisposition::ChildGroupReaped if context.is_cancelled() => {
                    DirectRunErrorKind::Cancelled
                }
                ProcessFailureDisposition::ChildGroupReaped if context.remaining().is_zero() => {
                    DirectRunErrorKind::DeadlineExceeded
                }
                _ => DirectRunErrorKind::ExecutionFailed,
            };
            DirectRunError::new(kind, process)
        }
    }
}

fn media_execution_error(
    error: MediaExecutionFailure,
    context: &OperationContext<'_>,
) -> DirectRunError {
    let disposition = error.disposition();
    if disposition == ProcessFailureDisposition::NoChildSpawned
        && context.check_active("direct native operation").is_err()
    {
        return inactive_error(context);
    }
    let kind = if disposition == ProcessFailureDisposition::OwnershipUncertain {
        DirectRunErrorKind::OwnershipUncertain
    } else {
        match error.kind() {
            MediaExecutionErrorKind::InvalidArtifact
            | MediaExecutionErrorKind::ArtifactTooLarge
            | MediaExecutionErrorKind::TooManyArtifacts => DirectRunErrorKind::InvalidOutput,
            MediaExecutionErrorKind::InvalidRequest => DirectRunErrorKind::InvalidRequest,
            MediaExecutionErrorKind::Process(ProcessFailureKind::Cancelled) => {
                DirectRunErrorKind::Cancelled
            }
            MediaExecutionErrorKind::Process(ProcessFailureKind::DeadlineExceeded) => {
                DirectRunErrorKind::DeadlineExceeded
            }
            MediaExecutionErrorKind::Process(_) => {
                if context.is_cancelled()
                    && disposition != ProcessFailureDisposition::NoChildSpawned
                {
                    DirectRunErrorKind::Cancelled
                } else if context.remaining().is_zero()
                    && disposition != ProcessFailureDisposition::NoChildSpawned
                {
                    DirectRunErrorKind::DeadlineExceeded
                } else {
                    DirectRunErrorKind::ExecutionFailed
                }
            }
            MediaExecutionErrorKind::InvalidOutputDirectory
            | MediaExecutionErrorKind::OutputAlreadyExists => DirectRunErrorKind::InvalidRequest,
        }
    };
    DirectRunError::new(kind, disposition)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    use serde_json::{json, Map};
    use tempfile::TempDir;

    use super::super::catalog::ProfileKey;
    use super::super::requests::InferenceResult;
    use super::super::runtime::NeverCancel;

    struct AlwaysCancel;

    impl super::super::runtime::Cancellation for AlwaysCancel {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    struct SharedCancel(Arc<AtomicBool>);

    impl super::super::runtime::Cancellation for SharedCancel {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    fn fixture(
        engine: &str,
        profile_type: &str,
        device: &str,
        response: &str,
    ) -> (TempDir, Catalog, InferenceRequest, std::path::PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("toolkits/l2m2");
        let decisions = root.path().join("toolkits/l3ds");
        fs::create_dir_all(&media).unwrap();
        fs::create_dir_all(decisions.join("decision-models")).unwrap();
        fs::create_dir_all(decisions.join(".venv/bin")).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-chat:\n    cmd: server\n",
        )
        .unwrap();
        fs::write(
            media.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        let manifest = json!({
            "version": "1.0",
            "profiles": [{
                "id": "fixture-decision",
                "name": "Fixture decision",
                "engine": engine,
                "type": profile_type,
                "device": device,
                "model": "fixture-model",
                "entrypoint": "decision-models/runner.py"
            }]
        });
        fs::write(
            decisions.join("decision-runtimes.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let interpreter = decisions.join(".venv/bin/python");
        let fixture_guard = super::super::process::executable_fixture_guard();
        fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(
            decisions.join("decision-models/runner.py"),
            format!(
                "#!/bin/sh\nprintf x >> \"$0.calls\"\n/bin/cat >/dev/null\nprintf '%s\\n' '{}'\n",
                response.replace('\'', "'\\''")
            ),
        )
        .unwrap();
        drop(fixture_guard);

        let catalog = Catalog::load(root.path()).unwrap();
        let profile =
            ProfileKey::new(Lane::Decisions, "fixture-decision").expect("valid fixture profile");
        let mut questions = Map::new();
        questions.insert(
            "route".into(),
            json!({"type": "choice", "criteria": {"yes": "Proceed", "no": "Stop"}}),
        );
        let request = InferenceRequest::Decision {
            profile,
            state: json!({"ready": true}),
            questions,
        };
        let launch_marker = decisions
            .join("decision-models/runner.py")
            .with_extension("py.calls");
        (root, catalog, request, launch_marker)
    }

    fn context<'a>(
        cancellation: &'a dyn super::super::runtime::Cancellation,
    ) -> OperationContext<'a> {
        OperationContext::new(Instant::now() + Duration::from_secs(5), cancellation)
    }

    fn embedding_fixture() -> (TempDir, Catalog, InferenceRequest) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("toolkits/l2m2")).unwrap();
        fs::create_dir_all(root.path().join("toolkits/l3ds")).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-embeddings:\n    cmd: |\n      llama-server --embedding\n",
        )
        .unwrap();
        fs::write(
            root.path().join("toolkits/l2m2/media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        fs::write(
            root.path().join("toolkits/l3ds/decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        let request = InferenceRequest::Embeddings {
            profile: ProfileKey::new(Lane::Language, "fixture-embeddings").unwrap(),
            input: vec!["alpha".into(), "beta".into()],
        };
        (root, catalog, request)
    }

    fn accept_bounded(listener: &TcpListener) -> TcpStream {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "embedding request not received");
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("fixture listener failed: {error}"),
            }
        }
    }

    fn read_embedding_request(stream: &mut TcpStream) -> serde_json::Value {
        let mut header = Vec::new();
        let mut byte = [0];
        while !header.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            header.push(byte[0]);
            assert!(header.len() < 16_384, "request headers are bounded");
        }
        let header = String::from_utf8(header).unwrap();
        assert!(header.contains("POST /v1/embeddings HTTP/1.1"));
        let length = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        assert!(length <= 1024 * 1024 + 4096);
        let mut body = vec![0; length];
        stream.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn write_http_response(stream: &mut TcpStream, status: &str, body: &str) {
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
    }

    fn assert_no_connection(listener: &TcpListener) {
        match listener.accept() {
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Ok(_) => panic!("direct embeddings unexpectedly dispatched HTTP"),
            Err(error) => panic!("fixture listener failed: {error}"),
        }
    }

    fn native_response(engine: &str) -> String {
        serde_json::to_string(&json!({
            "engine": engine,
            "answers": {"route": {"choice": "yes"}},
            "latency_ms": 17.25,
            "usage": {"fixture_count": 1}
        }))
        .unwrap()
    }

    #[test]
    fn direct_runner_returns_native_output_once_without_replay() {
        let (_root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let DirectRunResult::Decision(result) =
            run_native(&catalog, &request, None, None, &context(&NeverCancel)).unwrap()
        else {
            panic!("direct decision returns native decision result")
        };

        let InferenceResult::Decision { answers } = result.result else {
            panic!("decision runner returns native answers")
        };
        assert_eq!(answers["route"]["choice"], "yes");
        assert_eq!(
            result.native_latency,
            Some(Duration::from_nanos(17_250_000))
        );
        assert_eq!(result.native_usage.unwrap()["fixture_count"], 1);
        assert_eq!(fs::read(launch_marker).unwrap(), b"x");
    }

    #[test]
    fn direct_runner_rejects_cuda_and_hosted_routes_before_launch() {
        for (engine, profile_type, device) in [("nagi", "local", "cuda"), ("jev", "api", "remote")]
        {
            let (_root, catalog, request, launch_marker) = fixture(
                engine,
                profile_type,
                device,
                &native_response("nagi-native"),
            );
            let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
                .err()
                .expect("unsupported route is rejected");
            assert_eq!(error.kind(), DirectRunErrorKind::UnsupportedRoute);
            assert_eq!(
                error.disposition(),
                DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
            );
            assert!(!error.blocks_following_work());
            assert!(!launch_marker.exists());
        }
    }

    #[test]
    fn direct_runner_rejects_catalog_drift_before_launch() {
        let (root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let manifest_path = root.path().join("toolkits/l3ds/decision-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = json!("cuda");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
            .err()
            .expect("catalog drift is rejected before dispatch");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!launch_marker.exists());
    }

    #[test]
    fn embedding_request_without_a_connection_does_not_launch_a_local_engine() {
        let (_root, catalog, _request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let request = InferenceRequest::Embeddings {
            profile: ProfileKey::new(Lane::Language, "fixture-chat").unwrap(),
            input: vec!["private input".into()],
        };
        let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
            .err()
            .expect("embedding request without a connection is rejected");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
        );
        assert!(!error.blocks_following_work());
        assert!(!launch_marker.exists());
    }

    #[test]
    fn invalid_native_output_is_not_replayed() {
        let (_root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", "{\"engine\":\"wrong\"}");
        let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
            .err()
            .expect("invalid native output is rejected");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidOutput);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::ChildGroupReaped)
        );
        assert!(!error.blocks_following_work());
        assert_eq!(fs::read(launch_marker).unwrap(), b"x");
        assert!(!error.to_string().contains("wrong"));
    }

    #[test]
    fn direct_runner_executes_typed_embeddings_through_existing_adapter() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_embedding_request(&mut stream);
            assert_eq!(payload["model"], "fixture-embeddings");
            assert_eq!(payload["input"], json!(["alpha", "beta"]));
            assert_eq!(payload["encoding_format"], "float");
            let body = r#"{"object":"list","model":"fixture-embeddings","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]},{"object":"embedding","index":1,"embedding":[0.3,0.4]}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#;
            write_http_response(&mut stream, "200 OK", body);
        });
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let result = run_native(
            &catalog,
            &request,
            None,
            Some(&adapter),
            &context(&NeverCancel),
        )
        .unwrap();
        let DirectRunResult::Embeddings(result) = result else {
            panic!("language embeddings return their typed payload")
        };
        assert_eq!(result.vectors, [vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert_eq!(result.usage.prompt_tokens, 2);
        assert_eq!(result.usage.total_tokens, 2);
        server.join().unwrap();
    }

    #[test]
    fn direct_runner_confirms_embedding_cancellation_only_before_dispatch() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(&adapter),
            &context(&AlwaysCancel),
        )
        .err()
        .expect("cancelled embedding request is not dispatched");
        assert_eq!(error.kind(), DirectRunErrorKind::Cancelled);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
        );
        assert!(error.cancellation_confirmed());
        assert!(!error.blocks_following_work());
        assert_no_connection(&listener);
    }

    #[test]
    fn dispatched_embedding_cancellation_is_remote_uncertainty() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let cancelled = Arc::new(AtomicBool::new(false));
        let server_cancelled = cancelled.clone();
        let server = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_embedding_request(&mut stream);
            assert_eq!(payload["input"], json!(["alpha", "beta"]));
            server_cancelled.store(true, Ordering::Release);
            std::thread::sleep(Duration::from_millis(100));
        });
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let cancellation = SharedCancel(cancelled);
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &cancellation);
        let error = run_native(&catalog, &request, None, Some(&adapter), &context)
            .err()
            .expect("dispatched cancellation remains uncertain");
        assert_eq!(error.kind(), DirectRunErrorKind::RemoteCompletionUncertain);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::Uncertain)
        );
        assert!(!error.cancellation_confirmed());
        assert!(error.blocks_following_work());
        assert_eq!(
            error.to_string(),
            "remote completion is unconfirmed; stop before continuing"
        );
        assert!(!error.to_string().contains("process cleanup"));
        server.join().unwrap();
    }

    #[test]
    fn uncertain_embedding_response_is_not_retried() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let no_retry_probe = listener.try_clone().unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_embedding_request(&mut stream);
            assert_eq!(payload["model"], "fixture-embeddings");
            write_http_response(&mut stream, "503 Service Unavailable", "private body");
        });
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(&adapter),
            &context(&NeverCancel),
        )
        .err()
        .expect("server rejection after dispatch is uncertain");
        assert_eq!(error.kind(), DirectRunErrorKind::RemoteCompletionUncertain);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::Uncertain)
        );
        assert!(error.blocks_following_work());
        assert!(!error.cancellation_confirmed());
        assert_eq!(
            error.to_string(),
            "remote completion is unconfirmed; stop before continuing"
        );
        server.join().unwrap();
        assert_no_connection(&no_retry_probe);
    }

    #[test]
    fn direct_runner_rejects_embedding_catalog_drift_before_http_dispatch() {
        let (root, catalog, request) = embedding_fixture();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-embeddings:\n    cmd: llama-server --port 8080\n",
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(&adapter),
            &context(&NeverCancel),
        )
        .err()
        .expect("catalog drift is rejected before HTTP dispatch");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
        );
        assert!(!error.blocks_following_work());
        assert_no_connection(&listener);
    }

    #[test]
    fn uncertain_cleanup_blocks_following_work() {
        let error = DirectRunError::new(
            DirectRunErrorKind::OwnershipUncertain,
            ProcessFailureDisposition::OwnershipUncertain,
        );
        assert!(error.blocks_following_work());
        assert!(!error.cancellation_confirmed());
        assert_eq!(
            error.to_string(),
            "native process cleanup is uncertain; stop before continuing"
        );
    }

    #[test]
    fn undispatched_errors_classify_cancellation_deadline_and_setup_failure() {
        let cancelled = execution_error(
            DecisionExecutionError::NotDispatched,
            &context(&AlwaysCancel),
        );
        assert_eq!(cancelled.kind(), DirectRunErrorKind::Cancelled);
        assert_eq!(
            cancelled.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(cancelled.cancellation_confirmed());

        let expired_context = OperationContext::new(Instant::now(), &NeverCancel);
        let expired = execution_error(DecisionExecutionError::NotDispatched, &expired_context);
        assert_eq!(expired.kind(), DirectRunErrorKind::DeadlineExceeded);
        assert_eq!(
            expired.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!expired.cancellation_confirmed());

        let setup_failure = execution_error(
            DecisionExecutionError::NotDispatched,
            &context(&NeverCancel),
        );
        assert_eq!(setup_failure.kind(), DirectRunErrorKind::ExecutionFailed);
        assert_eq!(
            setup_failure.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!setup_failure.cancellation_confirmed());
    }
}
