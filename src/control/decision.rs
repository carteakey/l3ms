//! Strict native decision execution and result decoding for local, hosted, and
//! llama.cpp SystemOne HTTP operations. Never substitutes demo decisions.
//!
//! Decoding validates native outputs; it does not establish live acceptance.
//! The supervised executor must separately require successful completion and
//! acknowledge cleanup before releasing its lease.

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

/// Decode the native llama.cpp SystemOne HTTP envelope. Unlike the existing
/// runner stdout format, this response carries no trusted engine label and no
/// adapter-measured latency, so neither is invented or inferred here.
pub(crate) fn decode_systemone_response(
    request: &InferenceRequest,
    response: &[u8],
) -> Result<NativeDecisionResult, DecisionDecodeError> {
    if response.len() > MAX_RESULT_BYTES {
        return Err(DecisionDecodeError::TooLarge);
    }
    if !matches!(request, InferenceRequest::Decision { .. }) {
        return Err(DecisionDecodeError::InvalidRequest);
    }
    let payload: serde_json::Value =
        serde_json::from_slice(response).map_err(|_| DecisionDecodeError::Malformed)?;
    if payload.get("error").is_some()
        || ["choices", "scores", "nouls"]
            .iter()
            .any(|field| payload.get(field).is_some_and(|value| !value.is_object()))
    {
        return Err(DecisionDecodeError::InvalidResult);
    }
    let decoded = decode_decision_payload(request, &payload, false, true)?;
    let InferenceResult::Decision { answers } = &decoded.result else {
        return Err(DecisionDecodeError::InvalidRequest);
    };
    validate_systemone_answer_types(request, answers)?;
    Ok(decoded)
}

fn validate_systemone_answer_types(
    request: &InferenceRequest,
    answers: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), DecisionDecodeError> {
    let InferenceRequest::Decision { questions, .. } = request else {
        return Err(DecisionDecodeError::InvalidRequest);
    };
    let unit_interval = |value: &serde_json::Value| {
        value
            .as_f64()
            .is_some_and(|number| number.is_finite() && (0.0..=1.0).contains(&number))
    };
    for (id, question) in questions {
        let answer = answers.get(id).ok_or(DecisionDecodeError::InvalidResult)?;
        match question.get("type").and_then(serde_json::Value::as_str) {
            Some("score")
                if !answer
                    .as_object()
                    .and_then(|fields| fields.get("score"))
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(f64::is_finite) =>
            {
                return Err(DecisionDecodeError::InvalidResult);
            }
            Some("noul")
                if !answer
                    .as_object()
                    .and_then(|fields| fields.get("noul"))
                    .is_some_and(unit_interval) =>
            {
                return Err(DecisionDecodeError::InvalidResult);
            }
            _ => {}
        }
        if answer.get("probabilities").is_some_and(|values| {
            !values
                .as_object()
                .is_some_and(|values| values.values().all(unit_interval))
        }) {
            return Err(DecisionDecodeError::InvalidResult);
        }
    }
    Ok(())
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
    execute_prepared_recipe(
        catalog,
        request,
        &recipe,
        DecisionBinding {
            route: None,
            device: DecisionDevice::Inherited,
        },
        context,
        recovery,
    )
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
        DecisionBinding {
            route: Some(accepted_route),
            device: DecisionDevice::Cpu,
        },
        context,
        recovery,
    )
}

/// Execute one accepted Jev API decision with the toolkit's hosted Python
/// environment. The caller owns admission and must not replay an uncertain
/// attempt; the process result establishes no provider-side cancellation or
/// resource-release claim.
pub(crate) fn execute_accepted_hosted_decision(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<NativeDecisionResult, DecisionExecutionError> {
    if !accepted_hosted_decision_route(request, accepted_route) {
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
        DecisionBinding {
            route: Some(accepted_route),
            device: DecisionDevice::Hosted,
        },
        context,
        recovery,
    )
}

enum DecisionDevice {
    Inherited,
    Cpu,
    Hosted,
}

struct DecisionBinding<'a> {
    route: Option<&'a ResolvedProfileRoute>,
    device: DecisionDevice,
}

fn execute_prepared_recipe(
    catalog: &Catalog,
    request: &InferenceRequest,
    recipe: &NativeCommandRecipe,
    binding: DecisionBinding<'_>,
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
    let local_runner = matches!(engine.as_str(), "nagi" | "laya") && profile_type == "local";
    let hosted_runner = engine == "jev" && profile_type == "api";
    let binding_matches = match &binding.device {
        DecisionDevice::Hosted => hosted_runner,
        DecisionDevice::Inherited | DecisionDevice::Cpu => local_runner,
    };
    if !binding_matches {
        return Err(DecisionExecutionError::NotDispatched);
    }
    if let Some(route) = binding.route {
        let route_matches = match &binding.device {
            DecisionDevice::Cpu => accepted_local_decision_route(request, route, Device::Cpu),
            DecisionDevice::Hosted => accepted_hosted_decision_route(request, route),
            DecisionDevice::Inherited => false,
        };
        if !route_matches || !recipe_matches_route(recipe, route) {
            return Err(DecisionExecutionError::NotDispatched);
        }
    }
    let mut command = std::process::Command::new(&recipe.executable);
    command.args(&recipe.argv).current_dir(&recipe.cwd);
    if hosted_runner {
        command
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env_remove("TYPESAFE_LOG_LEVEL");
    } else {
        command
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("HF_DATASETS_OFFLINE", "1");
    }
    match &binding.device {
        DecisionDevice::Cpu => {
            // Do not let a library that ignores --device discover CUDA.
            command.env("CUDA_VISIBLE_DEVICES", "");
        }
        DecisionDevice::Inherited | DecisionDevice::Hosted => {}
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
    let decoded = match binding.route {
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
    decode_decision_payload(request, &payload, true, false)
}

fn decode_decision_payload(
    request: &InferenceRequest,
    payload: &serde_json::Value,
    parse_native_latency: bool,
    accept_systemone_fields: bool,
) -> Result<NativeDecisionResult, DecisionDecodeError> {
    let answers = match payload.get("answers") {
        Some(serde_json::Value::Object(answers)) if !answers.is_empty() => answers.clone(),
        Some(_) => return Err(DecisionDecodeError::InvalidResult),
        None if accept_systemone_fields => {
            let mut answers = serde_json::Map::new();
            for field in ["choices", "scores", "nouls"] {
                if let Some(value) = payload.get(field) {
                    let Some(values) = value.as_object() else {
                        return Err(DecisionDecodeError::InvalidResult);
                    };
                    answers.extend(values.clone());
                }
            }
            if answers.is_empty() {
                return Err(DecisionDecodeError::InvalidResult);
            }
            answers
        }
        None => return Err(DecisionDecodeError::InvalidResult),
    };
    let result = InferenceResult::Decision { answers };
    result
        .validate_for(request)
        .map_err(|_| DecisionDecodeError::InvalidResult)?;
    let native_latency = if parse_native_latency {
        payload
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
            .transpose()?
    } else {
        None
    };
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
    accepted_local_decision_route(request, route, Device::Cpu)
}

fn accepted_hosted_decision_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
) -> bool {
    let InferenceRequest::Decision { profile, .. } = request else {
        return false;
    };
    let NativeRoute::DecisionRunner { profile_type, .. } = &route.native_route else {
        return false;
    };
    profile == &route.profile
        && route.profile.lane == Lane::Decisions
        && route.runtime.lane == Lane::Decisions
        && route.runtime.as_str() == "jev"
        && route.device.as_ref() == Some(&Device::Remote)
        && profile_type == "api"
}

fn accepted_local_decision_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
    device: Device,
) -> bool {
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
        && route.device.as_ref() == Some(&device)
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
            images: Vec::new(),
        }
    }
    fn typed_request(question_type: &str) -> InferenceRequest {
        InferenceRequest::Decision {
            profile: ProfileKey::new(Lane::Decisions, "nagi-smol").unwrap(),
            state: serde_json::json!({}),
            questions: serde_json::json!({"value":{"type":question_type,"range":[0,100]}})
                .as_object()
                .unwrap()
                .clone(),
            images: Vec::new(),
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
    fn systemone_envelope_keeps_usage_and_does_not_infer_engine_or_latency() {
        let request = request("nagi-smol");
        let payload = serde_json::json!({
            "answers": {"route": {"choice":"yes", "confidence":0.8}},
            "usage": {"total_tokens": 4},
            "latency_ms": 19.5,
            "engine": "untrusted-wrapper-label"
        });
        let decoded =
            decode_systemone_response(&request, &serde_json::to_vec(&payload).unwrap()).unwrap();
        assert_eq!(decoded.native_latency, None);
        assert_eq!(
            decoded.native_usage,
            Some(
                serde_json::json!({"total_tokens":4})
                    .as_object()
                    .unwrap()
                    .clone()
            )
        );
        let InferenceResult::Decision { answers } = decoded.result else {
            panic!("wrong result lane")
        };
        assert_eq!(answers["route"]["choice"], "yes");

        let fallback = serde_json::json!({
            "choices": {"route": {"choice":"yes"}},
            "usage": {"total_tokens": 0}
        });
        assert!(
            decode_systemone_response(&request, &serde_json::to_vec(&fallback).unwrap()).is_ok()
        );
        let invalid = serde_json::json!({
            "answers": {"route": {"choice":"undeclared"}}
        });
        assert_eq!(
            decode_systemone_response(&request, &serde_json::to_vec(&invalid).unwrap())
                .err()
                .unwrap(),
            DecisionDecodeError::InvalidResult
        );
        for invalid in [
            serde_json::json!({
                "answers": null,
                "choices": {"route": {"choice":"yes"}}
            }),
            serde_json::json!({
                "answers": {"route": {"choice":"yes"}},
                "error": {"message":"native failure"}
            }),
            serde_json::json!({
                "choices": "malformed",
                "answers": {"route": {"choice":"yes"}}
            }),
        ] {
            assert_eq!(
                decode_systemone_response(&request, &serde_json::to_vec(&invalid).unwrap())
                    .err()
                    .unwrap(),
                DecisionDecodeError::InvalidResult
            );
        }
        assert_eq!(
            decode_systemone_response(&request, &vec![b'x'; MAX_RESULT_BYTES + 1])
                .err()
                .unwrap(),
            DecisionDecodeError::TooLarge
        );
    }

    #[test]
    fn systemone_score_noul_and_probabilities_have_native_numeric_types() {
        for (question_type, answer, valid) in [
            ("score", serde_json::json!({"score":75}), true),
            ("score", serde_json::json!({"score":"75"}), false),
            ("noul", serde_json::json!({"noul":0.75}), true),
            ("noul", serde_json::json!({"noul":-0.01}), false),
            ("noul", serde_json::json!({"noul":1.01}), false),
            ("noul", serde_json::json!({"noul":"0.75"}), false),
            (
                "score",
                serde_json::json!({"score":75,"probabilities":{"low":0.1,"high":1.0}}),
                true,
            ),
            (
                "score",
                serde_json::json!({"score":75,"probabilities":{"high":1.01}}),
                false,
            ),
            (
                "noul",
                serde_json::json!({"noul":0.5,"probabilities":[]}),
                false,
            ),
        ] {
            let request = typed_request(question_type);
            let payload = serde_json::json!({"answers":{"value":answer}});
            assert_eq!(
                decode_systemone_response(&request, &serde_json::to_vec(&payload).unwrap()).is_ok(),
                valid,
                "question type {question_type} with answer {answer}"
            );
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
        executable_device_fixture(engine_label, "cpu")
    }

    #[cfg(target_os = "linux")]
    fn executable_device_fixture(
        engine_label: &str,
        device: &str,
    ) -> (tempfile::TempDir, Catalog, InferenceRequest) {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let media = root.path().to_path_buf();
        let decisions = root.path().to_path_buf();
        std::fs::create_dir_all(&media).unwrap();
        std::fs::create_dir_all(decisions.join("decision-models")).unwrap();
        std::fs::create_dir_all(decisions.join(".venv-decisions/bin")).unwrap();
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
        let manifest = serde_json::json!({"version":"1.0","profiles":[{"id":"fixture","name":"Fixture","engine":"nagi","type":"local","device":device,"model":"fixture","entrypoint":"decision-models/runner.py"}]});
        std::fs::write(
            decisions.join("decision-runtimes.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        // A harmless shell stand-in, not a model or Python environment.
        let interpreter = decisions.join(".venv-decisions/bin/python");
        let fixture_guard = super::super::process::executable_fixture_guard();
        std::fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o700)).unwrap();
        drop(fixture_guard);
        let response = serde_json::to_string(&serde_json::json!({"engine":engine_label,"answers":{"route":{"choice":"yes"}},"latency_ms":17.25,"usage":{"fixture_count":1}})).unwrap();
        let script = format!(
            "[ \"$HF_HUB_OFFLINE\" = 1 ] && [ \"$TRANSFORMERS_OFFLINE\" = 1 ] || exit 3\ninput=$(/bin/cat)\ncase \"$input\" in *'\"state\"'*'\"questions\"'*) ;; *) exit 4;; esac\nprintf '%s\\n' 'fixture log on stderr' >&2\nprintf '%s\\n' '{response}'\n"
        );
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
        std::fs::remove_file(root.path().join(".venv-decisions/bin/python")).unwrap();
        assert_eq!(
            execute_native_decision(&catalog, &request, &context, recovery)
                .err()
                .unwrap(),
            DecisionExecutionError::NotDispatched
        );
    }
}
