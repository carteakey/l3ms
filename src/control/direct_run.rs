//! Direct, bounded execution for explicitly supported native work.
//!
//! This module accepts CPU-local Nagi/Laya decisions, standard and voice-design
//! Breeze speech, embeddings and Comfy workflows through existing adapters. It does not start
//! services, install runtimes, select a fallback, or retry an attempted
//! inference. Native input and output stay out of its errors.

use std::error::Error;
use std::fmt;
use std::path::Path;
use std::time::Duration;

use super::catalog::{Catalog, Device, Lane};
use super::comfy::{
    ComfyAdapter, ComfyErrorKind, ComfyFailure, ComfyJobState, ComfyOutputRoot, NativeComfyResult,
};
use super::decision::{
    execute_accepted_cpu_decision, DecisionExecutionError, NativeDecisionResult,
};
use super::language::{DispatchDisposition, LanguageAdapter, LanguageErrorKind, LanguageFailure};
use super::media_execution::{
    execute_breeze_speech_on_cpu, is_supported_cpu_breeze_route, MediaExecutionErrorKind,
    MediaExecutionFailure, MediaExecutionResult,
};
use super::process::{ProcessFailureDisposition, ProcessFailureKind};
use super::requests::{
    EmbeddingPayload, InferenceRequest, InferenceResult, MediaInput, SpeechMode,
};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::{NeverCancel, OperationContext, RecoveryPolicy};

const PROCESS_RECOVERY_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of one direct native operation. The payloads intentionally do not
/// implement `Debug`: native answers and filesystem paths are not log data.
pub enum DirectRunResult {
    Decision(NativeDecisionResult),
    Speech(MediaExecutionResult),
    Embeddings(EmbeddingPayload),
    Comfy(NativeComfyResult),
}

/// Explicit endpoint connection for one direct lane-native request.
pub enum DirectConnection<'a> {
    Language(&'a LanguageAdapter),
    Comfy {
        adapter: &'a ComfyAdapter,
        native_id: &'a str,
    },
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
    RemoteTerminal { cancellation_confirmed: bool },
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

    fn remote_terminal(kind: DirectRunErrorKind, cancellation_confirmed: bool) -> Self {
        Self {
            kind,
            disposition: DirectRunDisposition::RemoteTerminal {
                cancellation_confirmed,
            },
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

    /// True only when no child was spawned, the owned child group was reaped,
    /// a remote request was never dispatched, or Comfy reported Cancelled.
    pub fn cancellation_confirmed(self) -> bool {
        self.kind == DirectRunErrorKind::Cancelled
            && matches!(
                self.disposition,
                DirectRunDisposition::Process(
                    ProcessFailureDisposition::NoChildSpawned
                        | ProcessFailureDisposition::ChildGroupReaped,
                ) | DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
                    | DirectRunDisposition::RemoteTerminal {
                        cancellation_confirmed: true,
                    }
            )
    }

    pub fn is_remote(self) -> bool {
        matches!(
            self.disposition,
            DirectRunDisposition::Remote(_) | DirectRunDisposition::RemoteTerminal { .. }
        )
    }

    /// True when a client cancellation request was followed by a
    /// confirmed terminal Comfy state. This does not say the backend honored
    /// cancellation: `cancellation_confirmed` is true only for native Cancelled.
    pub fn remote_terminal_confirmed(self) -> bool {
        self.kind == DirectRunErrorKind::Cancelled
            && matches!(
                self.disposition,
                DirectRunDisposition::RemoteTerminal { .. }
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

/// Execute one already-built native request on an explicitly selected route.
/// Comfy workflow IDs must be generated and persisted by the caller before this
/// function; this runner never chooses a fallback, starts a runtime, or retries
/// a dispatched request. Comfy cancellation uses one bounded scoped recovery
/// cancel and terminal status is not GPU-release proof.
pub fn run_native(
    catalog: &Catalog,
    request: &InferenceRequest,
    speech_output_directory: Option<&Path>,
    connection: Option<DirectConnection<'_>>,
    context: &OperationContext<'_>,
) -> Result<DirectRunResult, DirectRunError> {
    let is_decision = matches!(request, InferenceRequest::Decision { .. });
    let is_embeddings = matches!(request, InferenceRequest::Embeddings { .. });
    let is_comfy_workflow = matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Workflow { .. },
            ..
        }
    );
    let is_cpu_breeze_speech = matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } if matches!(
            input.mode,
            SpeechMode::Standard
                | SpeechMode::VoiceDesign
                | SpeechMode::VoiceClone
                | SpeechMode::Streaming
        )
    );
    let (language, comfy) = match connection {
        Some(DirectConnection::Language(adapter)) => (Some(adapter), None),
        Some(DirectConnection::Comfy { adapter, native_id }) => (None, Some((adapter, native_id))),
        None => (None, None),
    };
    let valid_lane_arguments = match request {
        InferenceRequest::Decision { .. } => {
            speech_output_directory.is_none() && language.is_none() && comfy.is_none()
        }
        InferenceRequest::Media { .. } if is_cpu_breeze_speech => {
            speech_output_directory.is_some() && language.is_none() && comfy.is_none()
        }
        InferenceRequest::Media { .. } if is_comfy_workflow => {
            speech_output_directory.is_some() && language.is_none() && comfy.is_some()
        }
        InferenceRequest::Embeddings { .. } => {
            speech_output_directory.is_none() && language.is_some() && comfy.is_none()
        }
        _ => false,
    };
    if !valid_lane_arguments {
        return Err(unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_embeddings || is_comfy_workflow,
        ));
    }
    if context.check_active("direct native operation").is_err() {
        return Err(if is_embeddings || is_comfy_workflow {
            inactive_remote_error(context)
        } else {
            inactive_error(context)
        });
    }

    let original_profile = catalog.inspect(request.profile()).ok_or_else(|| {
        unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_embeddings || is_comfy_workflow,
        )
    })?;
    let catalog = catalog.refresh().map_err(|_| {
        unstarted_error(
            DirectRunErrorKind::CatalogUnavailable,
            is_embeddings || is_comfy_workflow,
        )
    })?;
    if catalog.inspect(request.profile()) != Some(original_profile) {
        return Err(unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_embeddings || is_comfy_workflow,
        ));
    }
    request.validate(&catalog).map_err(|_| {
        unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_embeddings || is_comfy_workflow,
        )
    })?;
    let route = resolve_profile_route(&catalog, request.profile()).map_err(|_| {
        unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_embeddings || is_comfy_workflow,
        )
    })?;
    let recovery = RecoveryPolicy::new(PROCESS_RECOVERY_TIMEOUT)
        .expect("fixed process recovery timeout is valid");
    if is_comfy_workflow {
        if !is_comfy_workflow_route(request, &route) {
            return Err(DirectRunError::remote(
                DirectRunErrorKind::UnsupportedRoute,
                DispatchDisposition::NoDispatch,
            ));
        }
        let (adapter, native_id) = comfy.expect("validated Comfy connection");
        let output_directory = speech_output_directory.expect("validated Comfy output directory");
        execute_comfy_workflow(
            &catalog,
            request,
            &route,
            output_directory,
            adapter,
            native_id,
            context,
        )
        .map(DirectRunResult::Comfy)
    } else if is_decision {
        if !is_cpu_local_nagi_or_laya(request, &route) {
            return Err(DirectRunError::new(
                DirectRunErrorKind::UnsupportedRoute,
                ProcessFailureDisposition::NoChildSpawned,
            ));
        }
        execute_accepted_cpu_decision(&catalog, request, &route, context, recovery)
            .map(DirectRunResult::Decision)
            .map_err(|error| execution_error(error, context))
    } else if is_cpu_breeze_speech {
        if !is_supported_cpu_breeze_route(request, &route) {
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

fn is_comfy_workflow_route(request: &InferenceRequest, route: &ResolvedProfileRoute) -> bool {
    let InferenceRequest::Media {
        profile,
        input: MediaInput::Workflow { task, graph },
    } = request
    else {
        return false;
    };
    profile == &route.profile
        && route.profile.lane == Lane::Media
        && route.runtime.lane == Lane::Media
        && route.runtime.as_str() == "ComfyUI"
        && route.tasks.iter().any(|declared| declared == task)
        && matches!(&route.native_route, NativeRoute::MediaScript { .. })
        && graph.as_object().is_some_and(|object| !object.is_empty())
}

fn execute_comfy_workflow(
    catalog: &Catalog,
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
    output_directory: &Path,
    adapter: &ComfyAdapter,
    native_id: &str,
    context: &OperationContext<'_>,
) -> Result<NativeComfyResult, DirectRunError> {
    let output_root = ComfyOutputRoot::open_fresh(output_directory).map_err(comfy_failure_error)?;
    let mut job = adapter
        .prepare_for_route(catalog, request, route, native_id)
        .map_err(comfy_failure_error)?;
    if let Err(error) = adapter.submit_for_route(&mut job, route, context) {
        if job.submission_attempted() && context_inactive(context) {
            return recover_comfy_cancellation(adapter, &job, context);
        }
        return Err(comfy_failure_error(error));
    }

    loop {
        if context_inactive(context) {
            return recover_comfy_cancellation(adapter, &job, context);
        }
        match adapter.status(&job, context) {
            Ok(snapshot) => {
                if let Some(result) = finish_comfy_snapshot(
                    adapter,
                    snapshot,
                    &output_root,
                    output_directory,
                    native_id,
                    context,
                )? {
                    return Ok(result);
                }
            }
            Err(_) if context_inactive(context) => {
                return recover_comfy_cancellation(adapter, &job, context)
            }
            Err(error) => return Err(comfy_failure_error(error)),
        }
        wait_for_poll(context);
    }
}

fn recover_comfy_cancellation(
    adapter: &ComfyAdapter,
    job: &super::comfy::PreparedComfyJob,
    caller_context: &OperationContext<'_>,
) -> Result<NativeComfyResult, DirectRunError> {
    let deadline = std::time::Instant::now()
        .checked_add(PROCESS_RECOVERY_TIMEOUT)
        .ok_or_else(remote_completion_uncertain)?;
    let recovery = OperationContext::new(deadline, &NeverCancel);

    // A successful cancel response is only an acknowledgement. The exact
    // native ID is polled until terminal; this never proves engine/GPU release.
    let _ = adapter.cancel(job, &recovery);
    loop {
        if recovery.remaining().is_zero() {
            return Err(remote_completion_uncertain());
        }
        match adapter.status(job, &recovery) {
            Ok(snapshot) => match snapshot.state {
                ComfyJobState::Completed | ComfyJobState::Failed => {
                    return Err(comfy_recovery_terminal_error(caller_context))
                }
                ComfyJobState::Cancelled => {
                    return Err(DirectRunError::remote_terminal(
                        DirectRunErrorKind::Cancelled,
                        true,
                    ))
                }
                ComfyJobState::Unknown | ComfyJobState::Pending | ComfyJobState::Running => {}
            },
            Err(_) => return Err(remote_completion_uncertain()),
        }
        wait_for_poll(&recovery);
    }
}

fn comfy_recovery_terminal_error(context: &OperationContext<'_>) -> DirectRunError {
    let kind = if context.is_cancelled() {
        DirectRunErrorKind::Cancelled
    } else {
        DirectRunErrorKind::DeadlineExceeded
    };
    DirectRunError::remote_terminal(kind, false)
}

fn finish_comfy_snapshot(
    adapter: &ComfyAdapter,
    snapshot: super::comfy::ComfySnapshot,
    output_root: &ComfyOutputRoot,
    output_directory: &Path,
    native_id: &str,
    context: &OperationContext<'_>,
) -> Result<Option<NativeComfyResult>, DirectRunError> {
    match snapshot.state {
        ComfyJobState::Completed => {
            let artifacts = adapter
                .collect_terminal_outputs(output_root, &snapshot, context)
                .map_err(|error| {
                    DirectRunError::remote_terminal(comfy_direct_error_kind(error.kind), false)
                })?;
            Ok(Some(NativeComfyResult {
                native_id: native_id.to_owned(),
                output_root: output_directory.to_path_buf(),
                artifacts,
            }))
        }
        ComfyJobState::Failed => Err(DirectRunError::remote_terminal(
            DirectRunErrorKind::ExecutionFailed,
            false,
        )),
        ComfyJobState::Cancelled => Err(DirectRunError::remote_terminal(
            DirectRunErrorKind::Cancelled,
            true,
        )),
        ComfyJobState::Unknown | ComfyJobState::Pending | ComfyJobState::Running => Ok(None),
    }
}

fn wait_for_poll(context: &OperationContext<'_>) {
    let remaining = context.remaining();
    if !remaining.is_zero() {
        std::thread::sleep(remaining.min(Duration::from_millis(25)));
    }
}

fn context_inactive(context: &OperationContext<'_>) -> bool {
    context.is_cancelled() || context.remaining().is_zero()
}

fn comfy_failure_error(error: ComfyFailure) -> DirectRunError {
    DirectRunError::remote(comfy_direct_error_kind(error.kind), error.disposition)
}

fn comfy_direct_error_kind(kind: ComfyErrorKind) -> DirectRunErrorKind {
    match kind {
        ComfyErrorKind::Cancelled => DirectRunErrorKind::Cancelled,
        ComfyErrorKind::Deadline => DirectRunErrorKind::DeadlineExceeded,
        ComfyErrorKind::Configuration
        | ComfyErrorKind::InvalidRequest
        | ComfyErrorKind::InvalidOwnership => DirectRunErrorKind::InvalidRequest,
        ComfyErrorKind::Protocol | ComfyErrorKind::TooLarge => DirectRunErrorKind::InvalidOutput,
        ComfyErrorKind::AlreadyAttempted | ComfyErrorKind::Transport | ComfyErrorKind::Rejected => {
            DirectRunErrorKind::ExecutionFailed
        }
    }
}

fn remote_completion_uncertain() -> DirectRunError {
    DirectRunError::remote(
        DirectRunErrorKind::RemoteCompletionUncertain,
        DispatchDisposition::Uncertain,
    )
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
    use super::super::native_commands::CPU_BREEZE_PROFILE_ID;
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

    fn cpu_breeze_fixture() -> (TempDir, Catalog) {
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("toolkits/l2m2");
        let decisions = root.path().join("toolkits/l3ds");
        fs::create_dir_all(media.join("media-models")).unwrap();
        fs::create_dir_all(&decisions).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-chat:\n    cmd: server\n",
        )
        .unwrap();
        fs::write(
            media.join("media-runtimes.json"),
            json!({
                "schema_version": 1,
                "profiles": [{
                    "id": CPU_BREEZE_PROFILE_ID,
                    "name": "CPU Breeze fixture",
                    "runtime": "audio.cpp",
                    "device": "cpu",
                    "tasks": ["tts", "clon", "design"],
                    "script": "media-models/generate-breeze-tts-2.sh"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            media.join("media-models/generate-breeze-tts-2.sh"),
            r#"#!/usr/bin/env bash
set -euo pipefail
output=""
out_dir=""
backend=""
task=""
instruction=""
mode="offline"
voice_ref=""
reference_text=""
while (($#)); do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --instruction) instruction="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --voice-ref) voice_ref="$2"; shift 2 ;;
    --reference-text) reference_text="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ "${CUDA_VISIBLE_DEVICES-unset}" == "" ]]
[[ "$backend" == "cpu" ]]
[[ "$output" == "$out_dir/result.wav" ]]
if [[ "$task" == "design" ]]; then
  [[ "$instruction" == "fixture voice direction" ]]
elif [[ "$task" == "clon" ]]; then
  [[ -n "$voice_ref" && -f "$voice_ref" && "$reference_text" == "fixture transcript" ]]
else
  [[ "$task" == "tts" && -z "$instruction" ]]
fi
printf 'RIFF\046\000\000\000WAVEfmt \020\000\000\000\001\000\001\000\300\135\000\000\200\273\000\000\002\000\020\000data\002\000\000\000\000\000' > "$output"
if [[ "$mode" == "streaming" ]]; then
  mkdir "$output.chunks"
  cp "$output" "$output.chunks/chunk_0001.wav"
fi
"#,
        )
        .unwrap();
        fs::set_permissions(
            media.join("media-models/generate-breeze-tts-2.sh"),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        fs::write(
            decisions.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        (root, catalog)
    }

    fn cpu_breeze_request(mode: SpeechMode, root: &std::path::Path) -> InferenceRequest {
        let voice_reference = if mode == SpeechMode::VoiceClone {
            let path = root.join("speaker.wav");
            fs::write(&path, b"fixture reference").unwrap();
            Some(path)
        } else {
            None
        };
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, CPU_BREEZE_PROFILE_ID).unwrap(),
            input: super::super::requests::MediaInput::Speech(
                super::super::requests::SpeechInput {
                    text: "fixture speech".into(),
                    mode,
                    instruction: (mode == SpeechMode::VoiceDesign)
                        .then(|| "fixture voice direction".into()),
                    voice_reference,
                    reference_text: (mode == SpeechMode::VoiceClone)
                        .then(|| "fixture transcript".into()),
                },
            ),
        }
    }

    fn cpu_breeze_output(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let output = root.join(name);
        fs::create_dir(&output).unwrap();
        fs::set_permissions(&output, fs::Permissions::from_mode(0o700)).unwrap();
        output
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
    fn direct_runner_accepts_supported_cpu_breeze_modes() {
        let (root, catalog) = cpu_breeze_fixture();
        for (mode, name) in [
            (SpeechMode::Standard, "cpu-standard"),
            (SpeechMode::VoiceDesign, "cpu-design"),
            (SpeechMode::VoiceClone, "cpu-clone"),
            (SpeechMode::Streaming, "cpu-streaming"),
        ] {
            let request = cpu_breeze_request(mode, root.path());
            let output = cpu_breeze_output(root.path(), name);
            let DirectRunResult::Speech(result) = run_native(
                &catalog,
                &request,
                Some(&output),
                None,
                &context(&NeverCancel),
            )
            .unwrap() else {
                panic!("CPU Breeze direct mode returns validated speech artifacts")
            };
            assert_eq!(result.mode(), mode);
            assert_eq!(
                result.artifacts().len(),
                if mode == SpeechMode::Streaming { 2 } else { 1 }
            );
            assert!(output.join("result.wav").is_file());
            if mode == SpeechMode::Streaming {
                assert_eq!(
                    result.artifacts()[1].relative_path(),
                    std::path::Path::new("result.wav.chunks/chunk_0001.wav")
                );
            }
        }
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
            Some(DirectConnection::Language(&adapter)),
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
            Some(DirectConnection::Language(&adapter)),
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
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Language(&adapter)),
            &context,
        )
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
            Some(DirectConnection::Language(&adapter)),
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
            Some(DirectConnection::Language(&adapter)),
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
