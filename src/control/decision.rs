//! Strict native decision result decoding. Never substitutes demo decisions.
//!
//! This only validates captured runner stdout; it does not execute a model or
//! establish live acceptance. The supervised executor must separately require
//! successful process exit and acknowledge cleanup before releasing its lease.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use super::catalog::{Catalog, Device, Lane};
use super::native_commands::{
    prepare_native_command, prepare_native_command_for_route, NativeCommandRecipe,
    NativeOutputExpectation,
};
use super::process::{self, ProcessFailureDisposition, ProcessLimits};
use super::requests::{InferenceRequest, InferenceResult};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::{OperationContext, RecoveryPolicy};

const MAX_RESULT_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionDecodeError {
    InvalidRequest,
    TooLarge,
    Malformed,
    UnexpectedEngine,
    InvalidResult,
}

impl fmt::Display for DecisionDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "native decision result rejected: {self:?}")
    }
}
impl Error for DecisionDecodeError {}

/// Native answers and optional forward-pass/API latency remain separate from
/// coordinator queue/cold-start/execution timings. No Debug payload dump.
pub struct NativeDecisionResult {
    pub result: InferenceResult,
    pub native_latency: Option<Duration>,
    /// Provider-specific usage fields stay native, not mapped to chat tokens.
    pub native_usage: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionExecutionError {
    /// No runner was started; request/configuration can be corrected explicitly.
    NotDispatched,
    /// A started native/hosted request has no verified result; do not replay.
    Uncertain { process: ProcessFailureDisposition },
    /// Local process finished, but its stdout is not a valid native result.
    InvalidResult(DecisionDecodeError),
}

impl fmt::Display for DecisionExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "native decision execution failed: {self:?}")
    }
}
impl Error for DecisionExecutionError {}

/// Concrete child-owned local decision execution. The caller must hold/reconcile its
/// resource lease and admission first and must verify GPU release afterward.
/// This blocking operation never creates an environment, retries a request, or
/// accepts synthetic results. Engines run with Hugging Face/Transformers
/// offline flags; model acquisition is separate explicit preparation work.
/// Hosted execution remains gated until its native SDK retry/auth contract is
/// verified; the standalone hosted CLI and result decoder are unaffected.
pub fn execute_native_decision(
    catalog: &Catalog,
    request: &InferenceRequest,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<NativeDecisionResult, DecisionExecutionError> {
    if !matches!(request, InferenceRequest::Decision { .. }) {
        return Err(DecisionExecutionError::NotDispatched);
    }
    let recipe = prepare_native_command(catalog, request)
        .map_err(|_| DecisionExecutionError::NotDispatched)?;
    let NativeOutputExpectation::DecisionJsonStdout {
        engine,
        profile_type,
        ..
    } = &recipe.expectation
    else {
        return Err(DecisionExecutionError::NotDispatched);
    };
    if !matches!(engine.as_str(), "nagi" | "laya") || profile_type != "local" {
        return Err(DecisionExecutionError::NotDispatched);
    }
    execute_prepared_recipe(catalog, request, &recipe, None, false, context, recovery)
}

/// Execute only a route already accepted by the coordinator's immediately
/// preceding dispatch gate. This is the narrow CPU-only worker seam: the
/// checked recipe helper must re-resolve the full accepted route, including
/// device, before returning the prepared command. Device selection is derived
/// from that route by `decision_recipe`; it is never inferred from fallback
/// metadata or arbitrary argv parsing.
pub(crate) fn execute_accepted_cpu_decision(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<NativeDecisionResult, DecisionExecutionError> {
    if !accepted_cpu_decision_route(request, accepted_route) {
        return Err(DecisionExecutionError::NotDispatched);
    }
    let recipe = prepare_native_command_for_route(catalog, request, accepted_route)
        .map_err(|_| DecisionExecutionError::NotDispatched)?;
    if !recipe_matches_route(&recipe, accepted_route) {
        return Err(DecisionExecutionError::NotDispatched);
    }
    execute_prepared_recipe(
        catalog,
        request,
        &recipe,
        Some(accepted_route),
        true,
        context,
        recovery,
    )
}

fn execute_prepared_recipe(
    catalog: &Catalog,
    request: &InferenceRequest,
    recipe: &NativeCommandRecipe,
    accepted_route: Option<&ResolvedProfileRoute>,
    cpu_only: bool,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<NativeDecisionResult, DecisionExecutionError> {
    let NativeOutputExpectation::DecisionJsonStdout {
        engine,
        profile_type,
        ..
    } = &recipe.expectation
    else {
        return Err(DecisionExecutionError::NotDispatched);
    };
    if !matches!(engine.as_str(), "nagi" | "laya") || profile_type != "local" {
        return Err(DecisionExecutionError::NotDispatched);
    }
    if let Some(route) = accepted_route {
        if !accepted_cpu_decision_route(request, route) || !recipe_matches_route(recipe, route) {
            return Err(DecisionExecutionError::NotDispatched);
        }
    }
    let mut command = std::process::Command::new(&recipe.executable);
    command.args(&recipe.argv).current_dir(&recipe.cwd);
    command
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1");
    if cpu_only {
        // The accepted route is explicitly CPU. Do not let CUDA be discovered
        // by a local library that ignores the runner's --device argument.
        command.env("CUDA_VISIBLE_DEVICES", "");
    }
    let limits = ProcessLimits::new(1024 * 1024, 2 * 1024 * 1024)
        .map_err(|_| DecisionExecutionError::NotDispatched)?;
    let output = process::execute(
        command,
        recipe
            .stdin_payload
            .as_ref()
            .map(|payload| payload.as_bytes()),
        limits,
        context,
        recovery,
    )
    .map_err(|error| match error.disposition() {
        ProcessFailureDisposition::NoChildSpawned => DecisionExecutionError::NotDispatched,
        disposition => DecisionExecutionError::Uncertain {
            process: disposition,
        },
    })?;
    let decoded = match accepted_route {
        Some(route) => decode_decision_result_for_route(request, route, output.stdout()),
        None => decode_decision_result(catalog, request, output.stdout()),
    };
    decoded.map_err(DecisionExecutionError::InvalidResult)
}

pub fn decode_decision_result(
    catalog: &Catalog,
    request: &InferenceRequest,
    stdout: &[u8],
) -> Result<NativeDecisionResult, DecisionDecodeError> {
    if stdout.len() > MAX_RESULT_BYTES {
        return Err(DecisionDecodeError::TooLarge);
    }
    if !matches!(request, InferenceRequest::Decision { .. }) {
        return Err(DecisionDecodeError::InvalidRequest);
    }
    let catalog = catalog
        .refresh()
        .map_err(|_| DecisionDecodeError::InvalidRequest)?;
    request
        .validate(&catalog)
        .map_err(|_| DecisionDecodeError::InvalidRequest)?;
    let route = resolve_profile_route(&catalog, request.profile())
        .map_err(|_| DecisionDecodeError::InvalidRequest)?;
    decode_decision_result_for_route(request, &route, stdout)
}

fn decode_decision_result_for_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
    stdout: &[u8],
) -> Result<NativeDecisionResult, DecisionDecodeError> {
    if stdout.len() > MAX_RESULT_BYTES {
        return Err(DecisionDecodeError::TooLarge);
    }
    if !matches!(request, InferenceRequest::Decision { .. }) || request.profile() != &route.profile
    {
        return Err(DecisionDecodeError::InvalidRequest);
    }
    if !matches!(route.native_route, NativeRoute::DecisionRunner { .. }) {
        return Err(DecisionDecodeError::InvalidRequest);
    }
    let expected_engine = match route.runtime.as_str() {
        "nagi" => "nagi-native",
        "laya" => "laya-native",
        "jev" => "systemone-official-api",
        // SemIf intentionally has no implemented native forward pass yet.
        _ => return Err(DecisionDecodeError::UnexpectedEngine),
    };
    let payload: serde_json::Value =
        serde_json::from_slice(stdout).map_err(|_| DecisionDecodeError::Malformed)?;
    if payload.get("engine").and_then(serde_json::Value::as_str) != Some(expected_engine) {
        return Err(DecisionDecodeError::UnexpectedEngine);
    }
    let answers = payload
        .get("answers")
        .and_then(serde_json::Value::as_object)
        .ok_or(DecisionDecodeError::InvalidResult)?
        .clone();
    let result = InferenceResult::Decision { answers };
    result
        .validate_for(request)
        .map_err(|_| DecisionDecodeError::InvalidResult)?;
    let native_latency = payload
        .get("latency_ms")
        .map(|value| {
            let milliseconds = value
                .as_f64()
                .filter(|value| {
                    value.is_finite() && *value >= 0.0 && *value <= 24.0 * 60.0 * 60.0 * 1000.0
                })
                .ok_or(DecisionDecodeError::InvalidResult)?;
            Duration::try_from_secs_f64(milliseconds / 1000.0)
                .map_err(|_| DecisionDecodeError::InvalidResult)
        })
        .transpose()?;
    let native_usage = payload
        .get("usage")
        .map(|value| {
            value
                .as_object()
                .cloned()
                .ok_or(DecisionDecodeError::InvalidResult)
        })
        .transpose()?;
    Ok(NativeDecisionResult {
        result,
        native_latency,
        native_usage,
    })
}

fn accepted_cpu_decision_route(request: &InferenceRequest, route: &ResolvedProfileRoute) -> bool {
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
        && route.device == Some(Device::Cpu)
        && profile_type == "local"
}

fn recipe_matches_route(recipe: &NativeCommandRecipe, route: &ResolvedProfileRoute) -> bool {
    let (
        NativeRoute::DecisionRunner {
            model,
            profile_type,
            tier,
            ..
        },
        NativeOutputExpectation::DecisionJsonStdout {
            engine,
            model: recipe_model,
            profile_type: recipe_type,
            tier: recipe_tier,
        },
    ) = (&route.native_route, &recipe.expectation)
    else {
        return false;
    };
    engine == route.runtime.as_str()
        && model == recipe_model
        && profile_type == recipe_type
        && tier == recipe_tier
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{Lane, ProfileKey};
    use super::*;

    fn request(profile: &str) -> InferenceRequest {
        InferenceRequest::Decision {
            profile: ProfileKey::new(Lane::Decisions,profile).unwrap(),
            state:serde_json::json!({"alert":"fixture"}),
            questions:serde_json::json!({"route":{"type":"choice","criteria":{"yes":"Proceed","no":"Stop"}}}).as_object().unwrap().clone(),
        }
    }
    fn catalog() -> Catalog {
        Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap()
    }

    #[test]
    fn native_answers_and_latency_are_preserved_without_claiming_live_inference() {
        for (profile, engine) in [
            ("nagi-smol", "nagi-native"),
            ("laya-en", "laya-native"),
            ("jev-api", "systemone-official-api"),
        ] {
            let answers = serde_json::json!({"route":{"choice":"yes","confidence":0.8,"probabilities":{"yes":0.8,"no":0.2}}});
            let bytes = serde_json::to_vec(
                &serde_json::json!({"engine":engine,"answers":answers,"latency_ms":17.25}),
            )
            .unwrap();
            let decoded = decode_decision_result(&catalog(), &request(profile), &bytes).unwrap();
            assert_eq!(decoded.native_latency, Some(Duration::from_micros(17250)));
            let InferenceResult::Decision { answers: actual } = decoded.result else {
                panic!("wrong lane")
            };
            assert_eq!(serde_json::Value::Object(actual), answers);
        }
    }

    #[test]
    fn synthetic_unknown_and_wrong_backend_labels_are_never_accepted() {
        for engine in [
            "nagi-simulated (fixture)",
            "jev-api (simulated-wan)",
            "semif-simulated",
            "laya-native",
            "",
        ] {
            let bytes = serde_json::to_vec(
                &serde_json::json!({"engine":engine,"answers":{"route":{"choice":"yes"}}}),
            )
            .unwrap();
            assert_eq!(
                decode_decision_result(&catalog(), &request("nagi-smol"), &bytes)
                    .err()
                    .unwrap(),
                DecisionDecodeError::UnexpectedEngine
            );
        }
        assert_eq!(
            decode_decision_result(&catalog(), &request("semif-openjev"), b"{}")
                .err()
                .unwrap(),
            DecisionDecodeError::UnexpectedEngine
        );
    }

    #[test]
    fn malformed_oversized_mismatched_and_invalid_results_fail_without_echoing_payload() {
        for bytes in [
            b"sk-secret-invalid-json".to_vec(),
            b"x".repeat(MAX_RESULT_BYTES + 1),
            br#"{"engine":"nagi-native","answers":{"route":{"choice":"undeclared"}}}"#.to_vec(),
            br#"{"engine":"nagi-native","answers":{"other":{"choice":"yes"}}}"#.to_vec(),
            br#"{"engine":"nagi-native","answers":{"route":{"choice":"yes"}},"latency_ms":-1}"#
                .to_vec(),
        ] {
            let error = decode_decision_result(&catalog(), &request("nagi-smol"), &bytes)
                .err()
                .unwrap();
            assert!(!error.to_string().contains("secret"));
            assert!(!error.to_string().contains("undeclared"));
        }
    }

    #[cfg(target_os = "linux")]
    fn executable_fixture(engine_label: &str) -> (tempfile::TempDir, Catalog, InferenceRequest) {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("toolkits/l2m2");
        let decisions = root.path().join("toolkits/l3ds");
        std::fs::create_dir_all(&media).unwrap();
        std::fs::create_dir_all(decisions.join("decision-models")).unwrap();
        std::fs::create_dir_all(decisions.join(".venv/bin")).unwrap();
        std::fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture:\n    cmd: fixture\n",
        )
        .unwrap();
        std::fs::write(
            media.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        std::fs::write(decisions.join("decision-runtimes.json"), r#"{"version":"1.0","profiles":[{"id":"fixture","name":"Fixture","engine":"nagi","type":"local","device":"cpu","model":"fixture","entrypoint":"decision-models/runner.py"}]}"#).unwrap();
        // A harmless shell stand-in, not a model or Python environment.
        let interpreter = decisions.join(".venv/bin/python");
        let fixture_guard = super::super::process::executable_fixture_guard();
        std::fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o700)).unwrap();
        drop(fixture_guard);
        let response = serde_json::to_string(&serde_json::json!({"engine":engine_label,"answers":{"route":{"choice":"yes"}},"latency_ms":17.25,"usage":{"fixture_count":1}})).unwrap();
        let script = format!("[ \"$HF_HUB_OFFLINE\" = 1 ] && [ \"$TRANSFORMERS_OFFLINE\" = 1 ] || exit 3\ninput=$(/bin/cat)\ncase \"$input\" in *'\"state\"'*'\"questions\"'*) ;; *) exit 4;; esac\nprintf '%s\\n' 'fixture log on stderr' >&2\nprintf '%s\\n' '{response}'\n");
        std::fs::write(decisions.join("decision-models/runner.py"), script).unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        (root, catalog, request("fixture"))
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn supervised_local_adapter_keeps_stderr_out_of_json_and_supplies_offline_stdin() {
        let (_root, catalog, request) = executable_fixture("nagi-native");
        let result = execute_native_decision(
            &catalog,
            &request,
            &OperationContext::new(
                std::time::Instant::now() + Duration::from_secs(3),
                &super::super::runtime::NeverCancel,
            ),
            RecoveryPolicy::new(Duration::from_secs(1)).unwrap(),
        )
        .unwrap();
        assert_eq!(result.native_latency, Some(Duration::from_micros(17250)));
        assert_eq!(result.native_usage.unwrap()["fixture_count"], 1);
        result.result.validate_for(&request).unwrap();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn supervised_adapter_rejects_demo_results_and_missing_interpreters() {
        let (root, catalog, request) = executable_fixture("nagi-simulated");
        let context = OperationContext::new(
            std::time::Instant::now() + Duration::from_secs(3),
            &super::super::runtime::NeverCancel,
        );
        let recovery = RecoveryPolicy::new(Duration::from_secs(1)).unwrap();
        assert_eq!(
            execute_native_decision(&catalog, &request, &context, recovery)
                .err()
                .unwrap(),
            DecisionExecutionError::InvalidResult(DecisionDecodeError::UnexpectedEngine)
        );
        std::fs::remove_file(root.path().join("toolkits/l3ds/.venv/bin/python")).unwrap();
        assert_eq!(
            execute_native_decision(&catalog, &request, &context, recovery)
                .err()
                .unwrap(),
            DecisionExecutionError::NotDispatched
        );
    }
}
