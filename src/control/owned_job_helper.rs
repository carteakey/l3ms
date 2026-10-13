//! Internal one-shot native job wrapper. A result packet is not completion:
//! the wrapper remains alive until exact owned-service Stop so its owner can
//! prove cgroup cleanup, GPU release and fresh artifact validity separately.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use super::catalog::Catalog;
use super::decision::{execute_accepted_cpu_decision, execute_accepted_cuda_decision};
use super::engine_transport::{CoordinatorIdentityDescriptor, CoordinatorLiveness};
use super::foreground_shutdown::ShutdownSignals;
use super::media_execution::{
    execute_breeze_speech_on_device, revalidate_breeze_artifacts, MediaArtifactKind,
};
use super::owned_job_store::{
    BreezeArtifactKind, DeviceSelection, NativeJobFailure, NativeJobResult, OwnedJobStore,
    ValidatedOwnedJob,
};
use super::requests::InferenceRequest;
use super::runtime::{Cancellation, OperationContext};

const MARKER: &str = "--l3ms-owned-local-job-v1";

/// Repeatable fresh artifact check for the post-stop publisher. The helper
/// also uses it before writing a candidate; neither call establishes cleanup.
pub(crate) fn revalidate_candidate(
    result: &NativeJobResult,
    directory: &std::path::Path,
    context: &OperationContext<'_>,
) -> Result<(), OwnedJobHelperError> {
    let NativeJobResult::Breeze(receipt) = result else {
        return Ok(());
    };
    let fresh = revalidate_breeze_artifacts(directory, receipt.mode(), context)
        .map_err(|_| OwnedJobHelperError::InvalidJob)?;
    if fresh.artifacts().len() != receipt.artifacts().len()
        || !fresh
            .artifacts()
            .iter()
            .zip(receipt.artifacts())
            .all(|(actual, expected)| {
                let kind = match actual.kind() {
                    MediaArtifactKind::FinalWave => BreezeArtifactKind::FinalWave,
                    MediaArtifactKind::StreamingChunk => BreezeArtifactKind::StreamingChunk,
                    MediaArtifactKind::RawRgb24Video | MediaArtifactKind::Video => return false,
                };
                kind == expected.kind()
                    && actual.relative_path().to_str() == Some(expected.relative_path())
            })
    {
        return Err(OwnedJobHelperError::InvalidJob);
    }
    Ok(())
}

pub(crate) fn helper_arguments(
    binding: &super::managed_service::OwnedLocalJobBinding,
) -> Vec<OsString> {
    let parent = binding.coordinator_identity();
    vec![
        MARKER.into(),
        binding.job_directory().as_os_str().to_owned(),
        binding.job_id().to_string().into(),
        binding.request_id().into(),
        binding.directory_device().to_string().into(),
        binding.directory_inode().to_string().into(),
        parent.pid().to_string().into(),
        parent.uid().to_string().into(),
        parent.start_time_ticks().to_string().into(),
        parent.network_namespace_device().to_string().into(),
        parent.network_namespace_inode().to_string().into(),
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnedJobHelperError {
    InvalidArguments,
    InvalidJob,
    Unavailable,
    ResultUncertain,
}

impl fmt::Display for OwnedJobHelperError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("owned native job wrapper failed; ownership requires reconciliation")
    }
}
impl std::error::Error for OwnedJobHelperError {}

pub(crate) fn is_helper_entrypoint(argument: &OsStr) -> bool {
    argument == MARKER
}

struct HelperInput {
    directory: PathBuf,
    job_id: u64,
    request_id: String,
    directory_device: u64,
    directory_inode: u64,
    coordinator: CoordinatorIdentityDescriptor,
}

fn parse_arguments(arguments: Vec<OsString>) -> Result<HelperInput, OwnedJobHelperError> {
    if arguments.len() != 11 || !is_helper_entrypoint(&arguments[0]) {
        return Err(OwnedJobHelperError::InvalidArguments);
    }
    let directory = PathBuf::from(&arguments[1]);
    if !directory.is_absolute() {
        return Err(OwnedJobHelperError::InvalidArguments);
    }
    let number = |index: usize| -> Result<u64, OwnedJobHelperError> {
        let text = arguments[index]
            .to_str()
            .ok_or(OwnedJobHelperError::InvalidArguments)?;
        let value = text
            .parse::<u64>()
            .map_err(|_| OwnedJobHelperError::InvalidArguments)?;
        if value.to_string() != text {
            return Err(OwnedJobHelperError::InvalidArguments);
        }
        Ok(value)
    };
    let job_id = number(2)?;
    if job_id == 0 {
        return Err(OwnedJobHelperError::InvalidArguments);
    }
    let request_id = arguments[3]
        .to_str()
        .ok_or(OwnedJobHelperError::InvalidArguments)?
        .to_owned();
    super::ipc::validate_request_id(&request_id)
        .map_err(|_| OwnedJobHelperError::InvalidArguments)?;
    let coordinator = CoordinatorIdentityDescriptor::from_parts(
        u32::try_from(number(6)?).map_err(|_| OwnedJobHelperError::InvalidArguments)?,
        u32::try_from(number(7)?).map_err(|_| OwnedJobHelperError::InvalidArguments)?,
        number(8)?,
        number(9)?,
        number(10)?,
    )
    .map_err(|_| OwnedJobHelperError::InvalidArguments)?;
    let directory_device = number(4)?;
    let directory_inode = number(5)?;
    if directory_inode == 0 {
        return Err(OwnedJobHelperError::InvalidArguments);
    }
    Ok(HelperInput {
        directory,
        job_id,
        request_id,
        directory_device,
        directory_inode,
        coordinator,
    })
}

struct HelperCancellation<'a> {
    signals: &'a ShutdownSignals,
    parent: &'a CoordinatorLiveness,
}
impl Cancellation for HelperCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        self.signals.is_cancelled() || !self.parent.is_live()
    }
}

fn execute_once(
    job: &ValidatedOwnedJob,
    directory: &std::path::Path,
    context: &OperationContext<'_>,
) -> NativeJobResult {
    let Ok(catalog) = Catalog::load(job.repository_root()) else {
        return NativeJobResult::Failed(NativeJobFailure::NotDispatched);
    };
    match (job.request(), job.device()) {
        (InferenceRequest::Decision { .. }, DeviceSelection::Cpu) => execute_accepted_cpu_decision(
            &catalog,
            job.request(),
            job.route(),
            context,
            job.bounds().recovery(),
        )
        .map(NativeJobResult::Decision)
        .unwrap_or(NativeJobResult::Failed(NativeJobFailure::Uncertain)),
        (InferenceRequest::Decision { .. }, DeviceSelection::Nvidia { device, .. }) => {
            execute_accepted_cuda_decision(
                &catalog,
                job.request(),
                job.route(),
                device,
                context,
                job.bounds().recovery(),
            )
            .map(NativeJobResult::Decision)
            .unwrap_or(NativeJobResult::Failed(NativeJobFailure::Uncertain))
        }
        (InferenceRequest::Media { .. }, DeviceSelection::Nvidia { device, .. }) => {
            execute_breeze_speech_on_device(
                &catalog,
                job.request(),
                job.route(),
                device,
                directory,
                context,
                job.bounds().recovery(),
            )
            .ok()
            .and_then(|result| NativeJobResult::from_breeze(&result).ok())
            .unwrap_or(NativeJobResult::Failed(NativeJobFailure::Uncertain))
        }
        _ => NativeJobResult::Failed(NativeJobFailure::NotDispatched),
    }
}

pub(crate) fn run_helper(arguments: Vec<OsString>) -> Result<(), OwnedJobHelperError> {
    let input = parse_arguments(arguments)?;
    let signals = ShutdownSignals::install().map_err(|_| OwnedJobHelperError::Unavailable)?;
    let store =
        OwnedJobStore::open(&input.directory).map_err(|_| OwnedJobHelperError::InvalidJob)?;
    if store.directory_device() != input.directory_device
        || store.directory_inode() != input.directory_inode
    {
        return Err(OwnedJobHelperError::InvalidJob);
    }
    let job = store
        .read_request()
        .map_err(|_| OwnedJobHelperError::InvalidJob)?;
    if job.job_id() != input.job_id || job.request_id() != input.request_id {
        return Err(OwnedJobHelperError::InvalidJob);
    }
    let parent = CoordinatorLiveness::from_expected(input.coordinator)
        .map_err(|_| OwnedJobHelperError::Unavailable)?;
    // Durable one-shot claim: a restarted or duplicate helper must never infer
    // permission to rerun from the presence of a readable request packet.
    store
        .claim_execution(&job)
        .map_err(|_| OwnedJobHelperError::InvalidJob)?;
    let cancellation = HelperCancellation {
        signals: &signals,
        parent: &parent,
    };
    let deadline = Instant::now()
        .checked_add(job.bounds().timeout())
        .ok_or(OwnedJobHelperError::InvalidJob)?;
    let context = OperationContext::new(deadline, &cancellation);
    let mut result = execute_once(&job, &input.directory, &context);
    if revalidate_candidate(&result, &input.directory, &context).is_err() {
        result = NativeJobResult::Failed(NativeJobFailure::InvalidOutput);
    }
    let publication = store
        .write_result(&job, &result)
        .map_err(|_| OwnedJobHelperError::ResultUncertain);
    // Parent loss cancels native execution, never self-retires an owned runtime.
    // Even uncertain publication remains parked for an exact-generation Stop.
    while !signals.is_cancelled() {
        thread::sleep(Duration::from_millis(25));
    }
    publication
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{Lane, ProfileKey};
    use super::super::owned_job_store::{JobExecutionBounds, OwnedJobRequestSpec};
    use super::super::routes::resolve_profile_route;
    use super::super::runtime::RecoveryPolicy;
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::process::{Command, Stdio};

    fn arguments() -> Vec<OsString> {
        let parent = CoordinatorIdentityDescriptor::capture_current().unwrap();
        [
            MARKER.to_owned(),
            "/tmp/l3ms-private-job".to_owned(),
            "1".to_owned(),
            "fixture-request".to_owned(),
            "0".to_owned(),
            "1".to_owned(),
            parent.pid().to_string(),
            parent.uid().to_string(),
            parent.start_time_ticks().to_string(),
            parent.network_namespace_device().to_string(),
            parent.network_namespace_inode().to_string(),
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    #[test]
    fn strict_arguments_require_canonical_job_and_parent_identity() {
        let valid = arguments();
        assert!(parse_arguments(valid.clone()).is_ok());
        for (index, replacement) in [
            (1, "relative"),
            (2, "0"),
            (2, "01"),
            (3, ""),
            (5, "0"),
            (6, "-1"),
            (8, "0"),
        ] {
            let mut invalid = valid.clone();
            invalid[index] = replacement.into();
            assert!(parse_arguments(invalid).is_err());
        }
        let mut extra = valid;
        extra.push("--replay".into());
        assert!(parse_arguments(extra).is_err());
        assert!(!OwnedJobHelperError::InvalidJob.to_string().contains("/tmp"));
    }

    #[test]
    fn parent_observation_is_birth_verified_and_not_pid_only() {
        let parent = CoordinatorIdentityDescriptor::capture_current().unwrap();
        assert!(CoordinatorLiveness::from_expected(parent)
            .unwrap()
            .is_live());
        let wrong_birth = CoordinatorIdentityDescriptor::from_parts(
            parent.pid(),
            parent.uid(),
            parent.start_time_ticks() + 1,
            parent.network_namespace_device(),
            parent.network_namespace_inode(),
        )
        .unwrap();
        assert!(CoordinatorLiveness::from_expected(wrong_birth).is_err());
    }

    #[test]
    fn helper_subprocess_entrypoint() {
        let Some(encoded) = std::env::var_os("L3MS_OWNED_JOB_TEST_ARGS") else {
            return;
        };
        let arguments: Vec<String> = serde_json::from_str(encoded.to_str().unwrap()).unwrap();
        run_helper(arguments.into_iter().map(OsString::from).collect()).unwrap();
    }

    #[test]
    fn native_result_does_not_exit_helper_before_exact_stop() {
        let root = tempfile::tempdir().unwrap();
        let decisions = root.path().join("toolkits/l3ds");
        let media = root.path().join("toolkits/l2m2");
        std::fs::create_dir_all(decisions.join("decision-models")).unwrap();
        std::fs::create_dir_all(decisions.join(".venv/bin")).unwrap();
        std::fs::create_dir_all(&media).unwrap();
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
        let guard = super::super::process::executable_fixture_guard();
        let interpreter = decisions.join(".venv/bin/python");
        std::fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o700)).unwrap();
        drop(guard);
        std::fs::write(decisions.join("decision-models/runner.py"), "[ -z \"$CUDA_VISIBLE_DEVICES\" ] || exit 8\n/bin/cat >/dev/null\nprintf '%s' '{\"engine\":\"nagi-native\",\"answers\":{\"route\":{\"choice\":\"yes\"}}}'\n").unwrap();
        let directory = root.path().join("job");
        std::fs::create_dir(&directory).unwrap();
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        let request = InferenceRequest::Decision {
            profile: ProfileKey::new(Lane::Decisions, "fixture").unwrap(),
            state: serde_json::json!({"fixture":true}),
            questions: serde_json::json!({"route":{"type":"choice","criteria":{"yes":"Proceed","no":"Stop"}}}).as_object().unwrap().clone(),
        };
        let route = resolve_profile_route(&catalog, request.profile()).unwrap();
        let store = OwnedJobStore::open(&directory).unwrap();
        let job = store
            .write_request(OwnedJobRequestSpec {
                job_id: 1,
                request_id: "fixture-request",
                repository_root: root.path(),
                request: &request,
                accepted_route: &route,
                device: DeviceSelection::Cpu,
                bounds: JobExecutionBounds::new(
                    Duration::from_secs(3),
                    RecoveryPolicy::new(Duration::from_secs(1)).unwrap(),
                )
                .unwrap(),
            })
            .unwrap();
        let mut args = arguments();
        args[1] = directory.clone().into_os_string();
        args[4] = store.directory_device().to_string().into();
        args[5] = store.directory_inode().to_string().into();
        let encoded = serde_json::to_string(
            &args
                .iter()
                .map(|arg| arg.to_str().unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let executable = std::env::current_exe().unwrap();
        let listing = Command::new(&executable).arg("--list").output().unwrap();
        let listing = String::from_utf8(listing.stdout).unwrap();
        let name = listing
            .lines()
            .find_map(|line| {
                line.strip_suffix(": test")
                    .filter(|name| name.ends_with("::helper_subprocess_entrypoint"))
            })
            .unwrap();
        let mut child = Command::new(executable)
            .args(["--exact", name])
            .env("L3MS_OWNED_JOB_TEST_ARGS", encoded)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let result = loop {
            if let Ok(result) = store.read_result(&job) {
                break Some(result);
            }
            if child.try_wait().unwrap().is_some() || Instant::now() >= deadline {
                break None;
            }
            thread::sleep(Duration::from_millis(10));
        };
        let parked = child.try_wait().unwrap().is_none();
        if parked {
            // Owned unreaped Child: its PID cannot be reused before wait.
            assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
        }
        let stop_deadline = Instant::now() + Duration::from_secs(3);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if Instant::now() >= stop_deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                break None;
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert!(
            matches!(result, Some(NativeJobResult::Decision(_))),
            "no real-model inference claimed by this fixture"
        );
        assert!(parked, "native output must not exit the owned wrapper");
        assert!(status.is_some_and(|status| status.success()));
    }
}
