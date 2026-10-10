//! Direct, in-process execution for the first CPU-local decision slice.
//!
//! This module accepts only an explicitly declared CPU Nagi/Laya route. It
//! does not start services, install runtimes, select a fallback, or retry an
//! attempted inference. Native input and output stay out of its errors.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use super::catalog::{Catalog, Device, Lane};
use super::decision::{
    execute_accepted_cpu_decision, DecisionExecutionError, NativeDecisionResult,
};
use super::process::ProcessFailureDisposition;
use super::requests::InferenceRequest;
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::{OperationContext, RecoveryPolicy};

const PROCESS_RECOVERY_TIMEOUT: Duration = Duration::from_secs(5);

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
}

/// A redacted direct-run failure plus the strongest child-cleanup conclusion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectRunError {
    kind: DirectRunErrorKind,
    disposition: ProcessFailureDisposition,
}

impl DirectRunError {
    fn new(kind: DirectRunErrorKind, disposition: ProcessFailureDisposition) -> Self {
        Self { kind, disposition }
    }

    #[cfg(test)]
    pub fn kind(self) -> DirectRunErrorKind {
        self.kind
    }

    #[cfg(test)]
    pub fn disposition(self) -> ProcessFailureDisposition {
        self.disposition
    }

    /// True only when the caller cancelled and either no child was spawned
    /// or the owned child group was confirmed reaped.
    pub fn cancellation_confirmed(self) -> bool {
        self.kind == DirectRunErrorKind::Cancelled
            && matches!(
                self.disposition,
                ProcessFailureDisposition::NoChildSpawned
                    | ProcessFailureDisposition::ChildGroupReaped
            )
    }

    /// Callers must keep execution serialized and surface recovery when the
    /// child group could not be confirmed absent.
    pub fn blocks_following_work(self) -> bool {
        self.disposition == ProcessFailureDisposition::OwnershipUncertain
    }
}

impl fmt::Display for DirectRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            DirectRunErrorKind::InvalidRequest => "CPU decision request is invalid",
            DirectRunErrorKind::CatalogUnavailable => "decision catalog is unavailable",
            DirectRunErrorKind::UnsupportedRoute => "CPU decision route is unsupported",
            DirectRunErrorKind::Cancelled => "CPU decision was cancelled",
            DirectRunErrorKind::DeadlineExceeded => "CPU decision timed out",
            DirectRunErrorKind::ExecutionFailed => "CPU decision execution failed",
            DirectRunErrorKind::InvalidOutput => "CPU decision returned invalid output",
            DirectRunErrorKind::OwnershipUncertain => {
                "CPU decision process cleanup is uncertain; stop before continuing"
            }
        })
    }
}

impl Error for DirectRunError {}

/// Execute one already-built decision request on its declared CPU-local Nagi
/// or Laya route.
///
/// The current catalog is refreshed before dispatch, then the accepted route
/// is rechecked by the native command adapter. Cancellation and the caller's
/// deadline are honored by the bounded process runner; its separate cleanup
/// budget is fixed at five seconds. If cleanup is uncertain, callers must not
/// clear their busy state or start another operation.
pub fn run_cpu_decision(
    catalog: &Catalog,
    request: &InferenceRequest,
    context: &OperationContext<'_>,
) -> Result<NativeDecisionResult, DirectRunError> {
    if !matches!(request, InferenceRequest::Decision { .. }) {
        return Err(DirectRunError::new(
            DirectRunErrorKind::InvalidRequest,
            ProcessFailureDisposition::NoChildSpawned,
        ));
    }
    if context.check_active("direct CPU decision").is_err() {
        return Err(inactive_error(context));
    }

    let catalog = catalog.refresh().map_err(|_| {
        DirectRunError::new(
            DirectRunErrorKind::CatalogUnavailable,
            ProcessFailureDisposition::NoChildSpawned,
        )
    })?;
    request.validate(&catalog).map_err(|_| {
        DirectRunError::new(
            DirectRunErrorKind::InvalidRequest,
            ProcessFailureDisposition::NoChildSpawned,
        )
    })?;
    let route = resolve_profile_route(&catalog, request.profile()).map_err(|_| {
        DirectRunError::new(
            DirectRunErrorKind::InvalidRequest,
            ProcessFailureDisposition::NoChildSpawned,
        )
    })?;
    if !is_cpu_local_nagi_or_laya(request, &route) {
        return Err(DirectRunError::new(
            DirectRunErrorKind::UnsupportedRoute,
            ProcessFailureDisposition::NoChildSpawned,
        ));
    }

    let recovery = RecoveryPolicy::new(PROCESS_RECOVERY_TIMEOUT)
        .expect("fixed process recovery timeout is valid");
    execute_accepted_cpu_decision(&catalog, request, &route, context, recovery)
        .map_err(|error| execution_error(error, context))
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

fn inactive_error(context: &OperationContext<'_>) -> DirectRunError {
    let kind = if context.is_cancelled() {
        DirectRunErrorKind::Cancelled
    } else {
        DirectRunErrorKind::DeadlineExceeded
    };
    DirectRunError::new(kind, ProcessFailureDisposition::NoChildSpawned)
}

fn execution_error(
    error: DecisionExecutionError,
    context: &OperationContext<'_>,
) -> DirectRunError {
    match error {
        DecisionExecutionError::NotDispatched => {
            if context.check_active("direct CPU decision").is_err() {
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::os::unix::fs::PermissionsExt;
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
        let result = run_cpu_decision(&catalog, &request, &context(&NeverCancel)).unwrap();

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
            let error = run_cpu_decision(&catalog, &request, &context(&NeverCancel))
                .err()
                .expect("unsupported route is rejected");
            assert_eq!(error.kind(), DirectRunErrorKind::UnsupportedRoute);
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::NoChildSpawned
            );
            assert!(!error.blocks_following_work());
            assert!(!launch_marker.exists());
        }
    }

    #[test]
    fn direct_runner_rejects_non_decision_request_without_launch() {
        let (_root, catalog, _request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let request = InferenceRequest::Embeddings {
            profile: ProfileKey::new(Lane::Language, "fixture-chat").unwrap(),
            input: vec!["private input".into()],
        };
        let error = run_cpu_decision(&catalog, &request, &context(&NeverCancel))
            .err()
            .expect("non-decision request is rejected");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert!(!launch_marker.exists());
    }

    #[test]
    fn invalid_native_output_is_not_replayed() {
        let (_root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", "{\"engine\":\"wrong\"}");
        let error = run_cpu_decision(&catalog, &request, &context(&NeverCancel))
            .err()
            .expect("invalid native output is rejected");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidOutput);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert!(!error.blocks_following_work());
        assert_eq!(fs::read(launch_marker).unwrap(), b"x");
        assert!(!error.to_string().contains("wrong"));
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
            "CPU decision process cleanup is uncertain; stop before continuing"
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
            ProcessFailureDisposition::NoChildSpawned
        );
        assert!(cancelled.cancellation_confirmed());

        let expired_context = OperationContext::new(Instant::now(), &NeverCancel);
        let expired = execution_error(DecisionExecutionError::NotDispatched, &expired_context);
        assert_eq!(expired.kind(), DirectRunErrorKind::DeadlineExceeded);
        assert_eq!(
            expired.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert!(!expired.cancellation_confirmed());

        let setup_failure = execution_error(
            DecisionExecutionError::NotDispatched,
            &context(&NeverCancel),
        );
        assert_eq!(setup_failure.kind(), DirectRunErrorKind::ExecutionFailed);
        assert_eq!(
            setup_failure.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert!(!setup_failure.cancellation_confirmed());
    }
}
