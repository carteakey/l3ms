//! Concrete managed foreground language/local-job/Comfy configuration and executor.
//!
//! This first resident-engine path is deliberately specific: one llama-swap
//! v247 router, llama-server-compatible model routes, one configured NVIDIA
//! GPU, one private isolated supervisor, and the peer-authenticated Unix relay.
//! Explicitly allowlisted Breeze and local Nagi/Laya jobs use separate owned
//! one-shot services. CUDA jobs release the resident router first; CPU jobs
//! do not acquire its GPU. Publication follows exact Stop and release proof.
//! Explicitly configured image workflows use fresh per-job isolated Comfy
//! engines with main-thread durable permits and backend acknowledgments.
//! It does not adopt existing Comfy services or manage arbitrary HTTP engines,
//! ambient credentials, or direct clients. Model activation is a separate
//! one-shot operation from read-only readiness and inference dispatch.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

use super::catalog::{Catalog, Device, Lane, ProfileKey, RuntimeIdentity};
use super::cgroup_release::CgroupReleaseProbe;
use super::cuda_device::CudaDevice;
use super::engine_health::{
    activate_llama_swap_model_once, probe_llama_swap_router_model_readiness, ActivationDisposition,
    NativeEngineKind, RouterModelReadinessObservation,
};
use super::engine_transport::BackendPort;
use super::gpu_probe::{GpuProbeConfig, NvidiaSmiGpuProbe};
use super::inference::{Completion, DispatchTicket, RequestCancellation};
use super::language::{
    ChatEventCallbackError, ChatStreamEvent, DispatchDisposition, LanguageAdapter,
    LanguageErrorKind,
};
use super::managed_service::{
    LaunchRecipe, ManagedServiceController, ReconcileResult, ServiceLaunchRole,
};
use super::requests::{InferenceRequest, InferenceResult};
use super::result_buffer::{ResultEvent, ResultFailureCode, ResultTerminal};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::{Cancellation, OperationContext, RecoveryPolicy};
use super::service_intents::ServiceIntentStore;
use super::service_owner::StoppedServiceObservation;
use super::service_owner::{IsolatedEngineLaunch, ManagedRuntimeOwner};

const WORKER_POLL: Duration = Duration::from_millis(25);

const MAX_ROUTER_CONFIG_BYTES: u64 = 1024 * 1024;

/// Fixed backend contract. The router must be v247 and its model processes
/// must implement the llama-server `/health` contract used by the one-shot
/// upstream activation route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BackendContract {
    LlamaSwapV247LlamaServer,
}

/// Inputs for the concrete managed language session. These are already
/// validated against the current native catalog and the private router config;
/// no shell string, secret, or ambient endpoint is carried into execution.
pub(crate) struct ManagedSessionConfig {
    pub(crate) repository_root: PathBuf,
    pub(crate) data_directory: PathBuf,
    pub(crate) standalone_data_directory: PathBuf,
    pub(crate) service_intent_directory: PathBuf,
    pub(crate) transport_directory: PathBuf,
    pub(crate) coordinator_id: String,
    pub(crate) runtime: RuntimeIdentity,
    pub(crate) startup_profile: ProfileKey,
    pub(crate) private_router_config: PathBuf,
    private_router_config_identity: (u64, u64),
    private_router_config_bytes: Vec<u8>,
    pub(crate) supported_model_tasks: BTreeMap<String, BTreeSet<String>>,
    pub(crate) gpu: GpuProbeConfig,
    pub(crate) local_job_profiles: BTreeSet<ProfileKey>,
    pub(crate) comfy: Option<super::managed_comfy_config::ManagedComfyConfig>,
    job_directory_root: Option<PathBuf>,
    job_directory_identity: Option<(u64, u64)>,
    launch_plan: RouterLaunchPlan,
    pub(crate) activation_timeout: Duration,
    pub(crate) request_timeout: Duration,
    pub(crate) recovery: RecoveryPolicy,
}

/// Immutable declared launch inputs, not a prepared manager operation or an
/// owned handle. A proven transition may build a new launch; an uncertain
/// start/stop must never use this to retry. Coordinator identity is captured
/// afresh by the owner for every separately authorized launch.
struct RouterLaunchPlan {
    executable: PathBuf,
    working_directory: PathBuf,
    router_config: PathBuf,
    repository_root: PathBuf,
    helper_executable: PathBuf,
    port: u16,
    stop_timeout: Duration,
    device: CudaDevice,
}

impl RouterLaunchPlan {
    fn build(
        &self,
        runtime: &RuntimeIdentity,
        transport: &Path,
    ) -> Result<IsolatedEngineLaunch, ManagedConfigError> {
        let port = BackendPort::new(self.port).map_err(|_| ManagedConfigError::InvalidLaunch)?;
        let arguments = [
            OsString::from("-config"),
            self.router_config.clone().into_os_string(),
            OsString::from("-listen"),
            OsString::from(format!("127.0.0.1:{}", port.get())),
        ];
        let recipe = LaunchRecipe::new(&self.executable, &self.working_directory, arguments)
            .map_err(|_| ManagedConfigError::InvalidLaunch)?;
        IsolatedEngineLaunch::new(
            runtime.clone(),
            recipe,
            &self.helper_executable,
            transport,
            self.port,
            self.stop_timeout,
        )
        .and_then(|launch| launch.with_stack_root(&self.repository_root))
        .map(|launch| launch.with_cuda_device(self.device.clone()))
        .map_err(|_| ManagedConfigError::InvalidLaunch)
    }
}

/// Named construction seam keeps the validated contract explicit without a
/// long positional call whose arguments are easy to transpose.
pub(crate) struct ManagedSessionConfigInput {
    pub(crate) contract: BackendContract,
    pub(crate) repository_root: PathBuf,
    pub(crate) data_directory: PathBuf,
    pub(crate) standalone_data_directory: PathBuf,
    pub(crate) service_intent_directory: PathBuf,
    pub(crate) transport_directory: PathBuf,
    pub(crate) coordinator_id: String,
    pub(crate) startup_profile: String,
    pub(crate) supported_model_tasks: BTreeMap<String, BTreeSet<String>>,
    pub(crate) local_job_profiles: Vec<String>,
    pub(crate) comfy: Option<super::managed_comfy_config::ManagedComfyLaunchInput>,
    pub(crate) job_directory_root: Option<PathBuf>,
    pub(crate) gpu_uuid: String,
    pub(crate) gpu_index: u32,
    pub(crate) gpu_idle_mib: u64,
    pub(crate) router_executable: PathBuf,
    pub(crate) router_working_directory: PathBuf,
    pub(crate) private_router_config: PathBuf,
    pub(crate) router_port: u16,
    pub(crate) activation_timeout: Duration,
    pub(crate) request_timeout: Duration,
    pub(crate) stop_timeout: Duration,
    pub(crate) recovery: RecoveryPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ManagedConfigError {
    InvalidBackendContract,
    InvalidRuntime,
    InvalidStartupProfile,
    InvalidRouterConfig,
    RouterConfigUnavailable,
    RouterConfigHasApiKeys,
    RouterConfigUsesAmbientEnvironment,
    StartupModelMissing,
    InvalidLaunch,
    InvalidGpu,
    InvalidLocalJobs,
    InvalidComfy,
}

fn cuda_device_for_gpu_config(gpu: &GpuProbeConfig) -> Result<CudaDevice, ManagedConfigError> {
    CudaDevice::new(gpu.gpu_uuid()).map_err(|_| ManagedConfigError::InvalidGpu)
}

impl ManagedSessionConfig {
    pub(crate) fn validate(
        input: ManagedSessionConfigInput,
        catalog: &Catalog,
    ) -> Result<Self, ManagedConfigError> {
        if input.contract != BackendContract::LlamaSwapV247LlamaServer {
            return Err(ManagedConfigError::InvalidBackendContract);
        }
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap")
            .map_err(|_| ManagedConfigError::InvalidRuntime)?;
        let startup_profile = input
            .startup_profile
            .parse::<ProfileKey>()
            .map_err(|_| ManagedConfigError::InvalidStartupProfile)?;
        if startup_profile.lane != Lane::Language {
            return Err(ManagedConfigError::InvalidStartupProfile);
        }
        let router_config = validate_private_router_config(
            &input.private_router_config,
            input.repository_root.join("llama-swap.yaml").as_path(),
        )?;
        let route = super::routes::resolve_profile_route(catalog, &startup_profile)
            .map_err(|_| ManagedConfigError::InvalidStartupProfile)?;
        if route.runtime != runtime {
            return Err(ManagedConfigError::InvalidStartupProfile);
        }
        let super::routes::NativeRoute::LlamaSwapModel { model_id, .. } = route.native_route else {
            return Err(ManagedConfigError::InvalidStartupProfile);
        };
        let router_snapshot = read_router_config_snapshot(&router_config)?;
        if input.supported_model_tasks.is_empty()
            || input.supported_model_tasks.len() > 256
            || input.supported_model_tasks.iter().any(|(id, tasks)| {
                super::catalog::ProfileId::new(id.clone()).is_err()
                    || tasks.is_empty()
                    || tasks
                        .iter()
                        .any(|task| !matches!(task.as_str(), "chat" | "embeddings"))
                    || !router_snapshot.model_ids.contains(id)
            })
            || !input
                .supported_model_tasks
                .get(model_id.as_str())
                .is_some_and(|tasks| route.tasks.iter().all(|task| tasks.contains(task.as_str())))
        {
            return Err(ManagedConfigError::StartupModelMissing);
        }
        let gpu = GpuProbeConfig::new(input.gpu_uuid, input.gpu_index, input.gpu_idle_mib)
            .map_err(|_| ManagedConfigError::InvalidGpu)?;
        let cuda_device = cuda_device_for_gpu_config(&gpu)?;
        let mut local_job_profiles = BTreeSet::new();
        if input.local_job_profiles.len() > 128 {
            return Err(ManagedConfigError::InvalidLocalJobs);
        }
        for profile in &input.local_job_profiles {
            let key: ProfileKey = profile
                .parse()
                .map_err(|_| ManagedConfigError::InvalidLocalJobs)?;
            if !local_job_profiles.insert(key.clone())
                || !supported_local_job_route(
                    &resolve_profile_route(catalog, &key)
                        .map_err(|_| ManagedConfigError::InvalidLocalJobs)?,
                )
            {
                return Err(ManagedConfigError::InvalidLocalJobs);
            }
        }
        if local_job_profiles.is_empty() != input.job_directory_root.is_none() {
            return Err(ManagedConfigError::InvalidLocalJobs);
        }
        let job_directory_identity = if let Some(path) = &input.job_directory_root {
            let pinned = super::owned_job_store::OwnedJobStore::open(path)
                .map_err(|_| ManagedConfigError::InvalidLocalJobs)?;
            for reserved in [
                &input.data_directory,
                &input.standalone_data_directory,
                &input.service_intent_directory,
                &input.transport_directory,
            ] {
                let reserved = reserved
                    .canonicalize()
                    .map_err(|_| ManagedConfigError::InvalidLocalJobs)?;
                if path.starts_with(&reserved) || reserved.starts_with(path) {
                    return Err(ManagedConfigError::InvalidLocalJobs);
                }
            }
            Some((pinned.directory_device(), pinned.directory_inode()))
        } else {
            None
        };
        let helper_executable = std::env::current_exe()
            .map_err(|_| ManagedConfigError::InvalidLaunch)?
            .canonicalize()
            .map_err(|_| ManagedConfigError::InvalidLaunch)?;
        let launch_plan = RouterLaunchPlan {
            executable: input.router_executable,
            working_directory: input.router_working_directory,
            router_config: router_config.clone(),
            repository_root: input.repository_root.clone(),
            helper_executable,
            port: input.router_port,
            stop_timeout: input.stop_timeout,
            device: cuda_device,
        };
        // Validate the complete declaration now, without manager dispatch.
        launch_plan.build(&runtime, &input.transport_directory)?;

        let mut comfy = input
            .comfy
            .map(|declaration| {
                super::managed_comfy_config::ManagedComfyConfig::validate(
                    declaration,
                    catalog,
                    &gpu,
                    launch_plan.helper_executable.clone(),
                    input.transport_directory.clone(),
                    input.stop_timeout,
                )
            })
            .transpose()
            .map_err(|_| ManagedConfigError::InvalidComfy)?;
        if let Some(comfy) = comfy.as_mut() {
            let mut protected = vec![
                input.data_directory.clone(),
                input.standalone_data_directory.clone(),
                input.service_intent_directory.clone(),
                input.transport_directory.clone(),
            ];
            if let Some(root) = input.job_directory_root.as_ref() {
                protected.push(root.clone());
            }
            comfy
                .validate_protected_directories(&protected)
                .map_err(|_| ManagedConfigError::InvalidComfy)?;
        }

        Ok(Self {
            repository_root: input.repository_root,
            data_directory: input.data_directory,
            standalone_data_directory: input.standalone_data_directory,
            service_intent_directory: input.service_intent_directory,
            transport_directory: input.transport_directory,
            coordinator_id: input.coordinator_id,
            runtime,
            startup_profile,
            private_router_config: router_config,
            private_router_config_identity: router_snapshot.identity,
            private_router_config_bytes: router_snapshot.bytes,
            supported_model_tasks: input.supported_model_tasks,
            gpu,
            local_job_profiles,
            comfy,
            job_directory_root: input.job_directory_root,
            job_directory_identity,
            launch_plan,
            activation_timeout: input.activation_timeout,
            request_timeout: input.request_timeout,
            recovery: input.recovery,
        })
    }

    fn prepare_router_launch(&self) -> Result<IsolatedEngineLaunch, ManagedConfigError> {
        self.verify_private_router_config_unchanged()?;
        self.launch_plan
            .build(&self.runtime, &self.transport_directory)
    }

    fn verify_private_router_config_unchanged(&self) -> Result<(), ManagedConfigError> {
        let current = read_router_config_snapshot(&self.private_router_config)?;
        if current.identity != self.private_router_config_identity
            || current.bytes != self.private_router_config_bytes
        {
            return Err(ManagedConfigError::InvalidRouterConfig);
        }
        Ok(())
    }
}

fn supported_local_job_route(route: &ResolvedProfileRoute) -> bool {
    match (&route.native_route, &route.device) {
        (NativeRoute::DecisionRunner { profile_type, .. }, Some(Device::Cpu | Device::Cuda)) => {
            route.profile.lane == Lane::Decisions
                && route.runtime.lane == Lane::Decisions
                && matches!(route.runtime.as_str(), "nagi" | "laya")
                && profile_type == "local"
        }
        (NativeRoute::MediaScript { .. }, Some(Device::Cuda)) => {
            route.profile.to_string() == "media:breeze-tts-2"
                && route.runtime.lane == Lane::Media
                && route.runtime.as_str() == "audio.cpp"
        }
        _ => false,
    }
}

/// Cancellation bridge shared by the listener loop and the bounded native
/// worker. The listener's signal object itself is not sent across threads.
#[derive(Clone)]
pub(crate) struct ManagedWorkerCancellation(Arc<AtomicBool>);

impl ManagedWorkerCancellation {
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl Cancellation for ManagedWorkerCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

pub(crate) enum ManagedWorkerCommand {
    Dispatch(Box<DispatchTicket>),
    Shutdown,
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct ManagedJobKey {
    pub(crate) job_id: u64,
    pub(crate) request_id: String,
}

pub(crate) struct ManagedJobCompletion {
    pub(crate) key: ManagedJobKey,
    pub(crate) completion: Completion,
    pub(crate) result_terminal: ResultTerminal,
    /// True only after the owned service generation, cgroup, GPU idle state,
    /// and retained intent have all been reconciled.
    pub(crate) cleanup_proven: bool,
    /// A model/config/transport failure may end an otherwise cleaned session.
    pub(crate) stop_session: bool,
    pub(crate) artifacts: Vec<String>,
}

pub(crate) enum ManagedWorkerEvent {
    Ready(RouterModelReadinessObservation),
    StartupFailed,
    Append {
        key: ManagedJobKey,
        event: ResultEvent,
        reply: SyncSender<bool>,
    },
    PublishArtifacts {
        key: ManagedJobKey,
        files: Vec<super::artifact_store::PublishedArtifact>,
        reply: SyncSender<bool>,
    },
    /// No native POST may precede this main-thread durable handoff.
    ComfyCandidate {
        ticket: Box<DispatchTicket>,
        candidate: String,
        generation: Box<super::managed_service::OwnedServiceGenerationIdentity>,
        reply: SyncSender<Option<super::inference::ManagedComfyDispatchPermit>>,
    },
    ComfyBackendAcknowledged {
        acknowledgement: Box<super::comfy::ManagedComfyBackendAcknowledgement>,
        reply: SyncSender<bool>,
    },
    JobFinished(ManagedJobCompletion),
    Stopped {
        cleanup_proven: bool,
    },
}

/// Concrete worker for the first resident language path. All owner, health,
/// activation, router config, and native HTTP calls happen here while the
/// exact listener-owned `CoordinatorLock` remains borrowed. The main thread
/// owns history, reservations, queue scheduling and result-buffer mutation.
pub(crate) fn run_worker(
    config: ManagedSessionConfig,
    catalog: Catalog,
    host_guard: &super::ownership::CoordinatorLock,
    session_deadline: std::time::Instant,
    cancellation: ManagedWorkerCancellation,
    commands: Receiver<ManagedWorkerCommand>,
    events: SyncSender<ManagedWorkerEvent>,
) {
    let controller = match ManagedServiceController::production(&config.coordinator_id) {
        Ok(controller) => controller,
        Err(_) => {
            send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
            return;
        }
    };
    let store = match ServiceIntentStore::open(&config.service_intent_directory) {
        Ok(store) => store,
        Err(_) => {
            send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
            return;
        }
    };
    let mut owner = ManagedRuntimeOwner::new(controller, store, host_guard);
    let cgroups = match CgroupReleaseProbe::production() {
        Ok(probe) => probe,
        Err(_) => {
            send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
            return;
        }
    };
    let gpu = NvidiaSmiGpuProbe::new(config.gpu.clone());
    if reconcile_retained_services(&mut owner, &cgroups, &gpu, &config).is_err()
        || cancellation.is_cancelled()
        || std::time::Instant::now() >= session_deadline
    {
        send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
        return;
    }

    let startup_deadline = std::time::Instant::now()
        .checked_add(config.activation_timeout)
        .unwrap_or(session_deadline)
        .min(session_deadline);
    let startup_context = OperationContext::new(startup_deadline, &cancellation);
    let startup_route = match resolve_startup_route(&catalog, &config) {
        Some(route) => route,
        None => {
            send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
            return;
        }
    };
    if config.verify_private_router_config_unchanged().is_err() {
        send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
        return;
    }
    let launch = match config.prepare_router_launch() {
        Ok(launch) => launch,
        Err(_) => {
            send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
            return;
        }
    };
    let service = match owner.start_isolated_engine(launch, &startup_context) {
        Ok(service) => service,
        Err(_) => {
            send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
            return;
        }
    };
    let adapter = match LanguageAdapter::new_private_engine(&config.transport_directory, None) {
        Ok(adapter) => adapter,
        Err(_) => {
            let _ = stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
            send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
            return;
        }
    };
    if config.verify_private_router_config_unchanged().is_err() {
        let _ = stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
        send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
        return;
    }
    // Re-read the native catalog after the service start and before the
    // activation request. A profile that drifted while systemd was starting
    // must not trigger a model switch under the old accepted configuration.
    if resolve_startup_route(&catalog, &config).as_ref() != Some(&startup_route) {
        let _ = stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
        send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
        return;
    }
    let NativeRoute::LlamaSwapModel { model_id, .. } = &startup_route.native_route else {
        let _ = stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
        send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
        return;
    };
    let startup_model = model_id.as_str();
    let activation = activate_llama_swap_model_once(
        &owner,
        &service,
        &config.transport_directory,
        NativeEngineKind::LlamaServer,
        startup_model,
        &startup_context,
    );
    let startup_ready = activation
        .is_ok_and(|ack| {
            ack.matches_model(startup_model)
                && config.verify_private_router_config_unchanged().is_ok()
        })
        .then(|| {
            probe_llama_swap_router_model_readiness(
                &owner,
                &service,
                &config.transport_directory,
                startup_model,
                &startup_context,
            )
            .ok()
        })
        .flatten();
    let Some(readiness) = startup_ready else {
        let _ = stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
        send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
        return;
    };
    if !send_worker_event_bounded(
        &events,
        ManagedWorkerEvent::Ready(readiness),
        config.recovery,
    ) {
        let _ = stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
        return;
    }

    let mut resident = Some(service);
    loop {
        match commands.recv_timeout(WORKER_POLL) {
            Ok(ManagedWorkerCommand::Dispatch(ticket)) => {
                let local_ticket = config.local_job_profiles.contains(ticket.request.profile());
                let comfy_ticket = config
                    .comfy
                    .as_ref()
                    .is_some_and(|comfy| comfy.profiles().contains(ticket.request.profile()));
                let completion = if comfy_ticket {
                    ManagedComfyTicketExecutor {
                        config: &config,
                        catalog: &catalog,
                        owner: &mut owner,
                        resident: &mut resident,
                        cgroups: &cgroups,
                        gpu: &gpu,
                        session_cancellation: &cancellation,
                        session_deadline,
                        events: &events,
                    }
                    .execute(*ticket)
                } else if local_ticket {
                    ManagedLocalTicketExecutor {
                        config: &config,
                        owner: &mut owner,
                        resident: &mut resident,
                        cgroups: &cgroups,
                        gpu: &gpu,
                        session_cancellation: &cancellation,
                        session_deadline,
                        events: &events,
                    }
                    .execute(*ticket)
                } else {
                    if resident.is_none() {
                        let launched = config.prepare_router_launch().ok().and_then(|launch| {
                            let context = OperationContext::new(
                                ticket.deadline.min(session_deadline),
                                &cancellation,
                            );
                            observe_gpu_idle_retry(&gpu, &context, config.recovery).ok()?;
                            owner.start_isolated_engine(launch, &context).ok()
                        });
                        let Some(launched) = launched else {
                            send_worker_event_bounded(
                                &events,
                                ManagedWorkerEvent::JobFinished(uncertain_managed_completion(
                                    &ticket,
                                )),
                                config.recovery,
                            );
                            return;
                        };
                        resident = Some(launched);
                    }
                    ManagedTicketExecutor {
                        config: &config,
                        catalog: &catalog,
                        adapter: &adapter,
                        owner: &mut owner,
                        service: resident.as_ref().expect("language launch established"),
                        session_cancellation: &cancellation,
                        session_deadline,
                        events: &events,
                    }
                    .execute(*ticket)
                };
                let must_stop = completion.stop_session
                    || completion.completion == Completion::Uncertain
                    || cancellation.is_cancelled();
                let completion = if must_stop {
                    let resident_clean =
                        stop_resident(&mut resident, &mut owner, &cgroups, &gpu, config.recovery);
                    let clean = combined_cleanup_proven(
                        resident_clean,
                        local_ticket || comfy_ticket,
                        completion.cleanup_proven,
                    );
                    let ManagedJobCompletion {
                        key,
                        completion: prior_completion,
                        result_terminal: prior_terminal,
                        artifacts,
                        ..
                    } = completion;
                    ManagedJobCompletion {
                        key,
                        completion: if prior_completion == Completion::Uncertain && clean {
                            Completion::Uncertain
                        } else if clean {
                            prior_completion
                        } else {
                            Completion::Uncertain
                        },
                        result_terminal: if prior_completion == Completion::Uncertain {
                            ResultTerminal::Interrupted
                        } else if clean {
                            prior_terminal
                        } else {
                            ResultTerminal::Interrupted
                        },
                        cleanup_proven: clean,
                        stop_session: true,
                        artifacts,
                    }
                } else {
                    completion
                };
                let should_return =
                    deliver_job_finished(&events, completion, config.recovery, || {
                        let _ = stop_resident(
                            &mut resident,
                            &mut owner,
                            &cgroups,
                            &gpu,
                            config.recovery,
                        );
                    });
                if should_return {
                    return;
                }
            }
            Ok(ManagedWorkerCommand::Shutdown) => {
                let cleanup_proven =
                    stop_resident(&mut resident, &mut owner, &cgroups, &gpu, config.recovery);
                send_worker_event_bounded(
                    &events,
                    ManagedWorkerEvent::Stopped { cleanup_proven },
                    config.recovery,
                );
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                if cancellation.is_cancelled() || std::time::Instant::now() >= session_deadline {
                    let cleanup_proven =
                        stop_resident(&mut resident, &mut owner, &cgroups, &gpu, config.recovery);
                    send_worker_event_bounded(
                        &events,
                        ManagedWorkerEvent::Stopped { cleanup_proven },
                        config.recovery,
                    );
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = stop_resident(&mut resident, &mut owner, &cgroups, &gpu, config.recovery);
                return;
            }
        }
    }
}

fn send_worker_event_bounded(
    sender: &SyncSender<ManagedWorkerEvent>,
    event: ManagedWorkerEvent,
    recovery: RecoveryPolicy,
) -> bool {
    let context = recovery_context(recovery);
    let mut event = event;
    loop {
        if context.remaining().is_zero() {
            return false;
        }
        match sender.try_send(event) {
            Ok(()) => return true,
            Err(mpsc::TrySendError::Disconnected(_)) => return false,
            Err(mpsc::TrySendError::Full(returned)) => {
                event = returned;
                thread::sleep(WORKER_POLL.min(context.remaining()));
            }
        }
    }
}

fn deliver_job_finished(
    sender: &SyncSender<ManagedWorkerEvent>,
    completion: ManagedJobCompletion,
    recovery: RecoveryPolicy,
    cleanup_active_service: impl FnOnce(),
) -> bool {
    let terminate = completion.stop_session;
    if send_worker_event_bounded(
        sender,
        ManagedWorkerEvent::JobFinished(completion),
        recovery,
    ) {
        terminate
    } else {
        if !terminate {
            cleanup_active_service();
        }
        true
    }
}

struct NeverCancel;

impl Cancellation for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

static NEVER_CANCEL: NeverCancel = NeverCancel;

fn recovery_context(policy: RecoveryPolicy) -> OperationContext<'static> {
    let deadline = std::time::Instant::now()
        .checked_add(policy.timeout())
        .unwrap_or_else(std::time::Instant::now);
    OperationContext::new(deadline, &NEVER_CANCEL)
}

fn uncertain_managed_completion(ticket: &DispatchTicket) -> ManagedJobCompletion {
    ManagedJobCompletion {
        key: ManagedJobKey {
            job_id: ticket.job_id,
            request_id: ticket.request_id.clone(),
        },
        completion: Completion::Uncertain,
        result_terminal: ResultTerminal::Interrupted,
        cleanup_proven: false,
        stop_session: true,
        artifacts: Vec::new(),
    }
}

fn combined_cleanup_proven(resident_clean: bool, local_ticket: bool, local_clean: bool) -> bool {
    // A missing resident router says nothing about the local service. This
    // separate proof must survive uncertain completions and shutdown.
    resident_clean && (!local_ticket || local_clean)
}

fn stop_resident(
    resident: &mut Option<super::managed_service::OwnedService>,
    owner: &mut ManagedRuntimeOwner<'_>,
    cgroups: &CgroupReleaseProbe,
    gpu: &NvidiaSmiGpuProbe,
    recovery: RecoveryPolicy,
) -> bool {
    resident
        .take()
        .is_none_or(|service| stop_owned_service(owner, &service, cgroups, gpu, recovery))
}

struct ManagedComfyTicketExecutor<'executor, 'host> {
    config: &'executor ManagedSessionConfig,
    catalog: &'executor Catalog,
    owner: &'executor mut ManagedRuntimeOwner<'host>,
    resident: &'executor mut Option<super::managed_service::OwnedService>,
    cgroups: &'executor CgroupReleaseProbe,
    gpu: &'executor NvidiaSmiGpuProbe,
    session_cancellation: &'executor ManagedWorkerCancellation,
    session_deadline: std::time::Instant,
    events: &'executor SyncSender<ManagedWorkerEvent>,
}

impl ManagedComfyTicketExecutor<'_, '_> {
    fn execute(&mut self, ticket: DispatchTicket) -> ManagedJobCompletion {
        use super::comfy::{ComfyAdapter, ComfyContract, ComfyJobState, OwnedComfySession};
        let mut outcome = uncertain_managed_completion(&ticket);
        let Some(comfy) = self.config.comfy.as_ref() else {
            return outcome;
        };
        let cancellation = RequestAndSessionCancellation {
            request: &ticket.cancellation,
            session: self.session_cancellation,
        };
        let deadline = std::time::Instant::now()
            .checked_add(self.config.request_timeout)
            .unwrap_or(ticket.deadline)
            .min(ticket.deadline)
            .min(self.session_deadline);
        let context = OperationContext::new(deadline, &cancellation);
        let fresh = self.catalog.refresh().ok();
        if !comfy.profiles().contains(ticket.request.profile())
            || fresh
                .as_ref()
                .and_then(|catalog| resolve_profile_route(catalog, ticket.request.profile()).ok())
                .as_ref()
                != Some(&ticket.route)
            || !matches!(
                ticket.request,
                InferenceRequest::Media {
                    input: super::requests::MediaInput::Workflow { .. },
                    ..
                }
            )
        {
            return outcome;
        }
        if !stop_resident(
            self.resident,
            self.owner,
            self.cgroups,
            self.gpu,
            self.config.recovery,
        ) {
            return outcome;
        }
        // Until a Comfy start is attempted, the language release proves this
        // dispatch owns no engine. A Start uncertainty reverses this below.
        outcome.cleanup_proven = true;
        if context.is_cancelled() || context.remaining().is_zero() {
            set_comfy_context_terminal(&mut outcome, &context);
            return outcome;
        }
        let Ok((launch, output_directory)) = comfy.allocate_job_launch(ticket.job_id) else {
            return outcome;
        };
        let Some(prefix) = output_directory.file_name().and_then(|name| name.to_str()) else {
            return outcome;
        };
        let Ok(artifacts_root) =
            super::comfy_artifacts::ComfyArtifactRoot::open(&output_directory, prefix)
        else {
            return outcome;
        };
        if observe_gpu_idle_retry(self.gpu, &context, self.config.recovery).is_err() {
            return outcome;
        }
        outcome.cleanup_proven = false;
        let Ok(service) = self.owner.start_isolated_engine(launch, &context) else {
            return outcome;
        };
        let mut post_may_have_been_sent = false;
        let native = (|| {
            // Startup health is read-only and bounded. It cannot submit work.
            loop {
                if super::engine_health::probe(
                    self.owner,
                    &service,
                    &self.config.transport_directory,
                    NativeEngineKind::Comfy,
                    &context,
                )
                .is_ok()
                {
                    break;
                }
                if context.is_cancelled() || context.remaining().is_zero() {
                    return Err(());
                }
                thread::sleep(WORKER_POLL.min(context.remaining()));
            }
            let adapter = ComfyAdapter::new_private_engine(
                &self.config.transport_directory,
                None,
                ComfyContract::CanonicalIdsAndAtomicScopedCancellation,
            )
            .map_err(|_| ())?;
            let session = OwnedComfySession::new(
                &adapter,
                self.owner,
                &service,
                &self.config.transport_directory,
                &context,
            )
            .map_err(|_| ())?;
            let catalog = fresh.as_ref().ok_or(())?;
            let (mut job, permit) = session
                .prepare_dispatch_using(catalog, &ticket, &context, |candidate, generation| {
                    request_comfy_permit(self.events, &ticket, candidate, generation, &context)
                        .map_err(|_| super::comfy::ComfyFailure {
                            disposition: DispatchDisposition::NoDispatch,
                            kind: super::comfy::ComfyErrorKind::InvalidOwnership,
                        })
                })
                .map_err(|_| ())?;
            let mut acknowledged = false;
            post_may_have_been_sent = true;
            match session.submit_with_acknowledgement(&mut job, permit, &context) {
                Ok(acknowledgement) => {
                    publish_comfy_acknowledgement(self.events, acknowledgement, &context)?;
                    acknowledged = true;
                }
                // An uncertain POST permits only read-only observation of the
                // same candidate; never a second submission or fresh ID.
                Err(error) if error.disposition == DispatchDisposition::Uncertain => {}
                Err(_) => {
                    post_may_have_been_sent = false;
                    return Err(());
                }
            }
            loop {
                if context.is_cancelled() || context.remaining().is_zero() {
                    let recovery = recovery_context(self.config.recovery);
                    let _ = session.cancel(&mut job, &recovery);
                    // Cancel acknowledgment is not completion/release proof.
                    return Err(());
                }
                let (snapshot, evidence) = session
                    .status_with_acknowledgement(&job, &ticket, &context)
                    .map_err(|_| ())?;
                if !acknowledged {
                    if let Some(acknowledgement) = evidence {
                        publish_comfy_acknowledgement(self.events, acknowledgement, &context)?;
                        acknowledged = true;
                    }
                }
                match snapshot.state {
                    ComfyJobState::Completed | ComfyJobState::Failed | ComfyJobState::Cancelled => {
                        return Ok(snapshot);
                    }
                    ComfyJobState::Unknown | ComfyJobState::Pending | ComfyJobState::Running => {}
                }
                thread::sleep(WORKER_POLL.min(context.remaining()));
            }
        })();
        // Independent cleanup survives request cancellation/deadline. No
        // native terminal, output descriptor or cancel reply substitutes for
        // exact Stop, cgroup emptiness, fresh GPU idle and WAL retirement.
        outcome.cleanup_proven = stop_owned_service(
            self.owner,
            &service,
            self.cgroups,
            self.gpu,
            self.config.recovery,
        );
        if !outcome.cleanup_proven {
            return outcome;
        }
        let Ok(snapshot) = native else {
            if !post_may_have_been_sent {
                // Preparation failure or a positively NoDispatch submit
                // cannot be mistaken for an unknown native execution.
                if context.is_cancelled() || context.remaining().is_zero() {
                    set_comfy_context_terminal(&mut outcome, &context);
                } else {
                    outcome.completion = Completion::Failed;
                    outcome.result_terminal =
                        ResultTerminal::Failed(ResultFailureCode::Unavailable);
                }
            }
            return outcome;
        };
        if context.is_cancelled() || context.remaining().is_zero() {
            set_comfy_context_terminal(&mut outcome, &context);
            return outcome;
        }
        match snapshot.state {
            ComfyJobState::Completed => {
                // These opt-in profiles declare image output, not successful
                // empty workflows or unrelated audio/video descriptors.
                if snapshot.artifacts.is_empty()
                    || snapshot
                        .artifacts
                        .iter()
                        .any(|artifact| artifact.media_type != "images")
                {
                    outcome.completion = Completion::Failed;
                    outcome.result_terminal =
                        ResultTerminal::Failed(ResultFailureCode::InvalidOutput);
                    return outcome;
                }
                let Ok(artifacts) = artifacts_root.validate(&snapshot.artifacts, 512 * 1024 * 1024)
                else {
                    outcome.completion = Completion::Failed;
                    outcome.result_terminal =
                        ResultTerminal::Failed(ResultFailureCode::InvalidOutput);
                    return outcome;
                };
                let paths = artifacts
                    .iter()
                    .map(|artifact| artifact.relative_path.clone())
                    .collect();
                let mut files = Vec::with_capacity(artifacts.len());
                for (reference, descriptor) in artifacts.into_iter().zip(&snapshot.artifacts) {
                    let native_path = if descriptor.subfolder.is_empty() {
                        descriptor.filename.clone()
                    } else {
                        format!("{}/{}", descriptor.subfolder, descriptor.filename)
                    };
                    let Ok(file) = super::artifact_store::PublishedArtifact::capture(
                        &output_directory,
                        reference,
                        &native_path,
                        &context,
                    ) else {
                        return outcome;
                    };
                    files.push(file);
                }
                // Keep the original pre-launch root association through the
                // byte-handle handoff as well as descriptor validation.
                if artifacts_root
                    .validate(&snapshot.artifacts, 512 * 1024 * 1024)
                    .is_err()
                    || publish_artifacts(self.events, &outcome.key, files, &context).is_err()
                {
                    return outcome;
                }
                outcome.artifacts = paths;
                outcome.completion = Completion::Succeeded;
                outcome.result_terminal = ResultTerminal::Succeeded;
                outcome.stop_session = false;
            }
            ComfyJobState::Failed => {
                outcome.completion = Completion::Failed;
                outcome.result_terminal = ResultTerminal::Failed(ResultFailureCode::Internal);
                outcome.stop_session = false;
            }
            ComfyJobState::Cancelled => {
                outcome.completion = Completion::Cancelled;
                outcome.result_terminal = ResultTerminal::Cancelled;
                outcome.stop_session = false;
            }
            _ => {}
        }
        outcome
    }
}

fn set_comfy_context_terminal(outcome: &mut ManagedJobCompletion, context: &OperationContext<'_>) {
    if context.is_cancelled() {
        outcome.completion = Completion::Cancelled;
        outcome.result_terminal = ResultTerminal::Cancelled;
    } else {
        outcome.completion = Completion::Failed;
        outcome.result_terminal = ResultTerminal::Failed(ResultFailureCode::TimedOut);
    }
}

pub(crate) fn request_comfy_permit(
    sender: &SyncSender<ManagedWorkerEvent>,
    ticket: &DispatchTicket,
    candidate: String,
    generation: super::managed_service::OwnedServiceGenerationIdentity,
    context: &OperationContext<'_>,
) -> Result<super::inference::ManagedComfyDispatchPermit, ()> {
    let (reply, result) = mpsc::sync_channel(1);
    let ticket = Box::new(DispatchTicket {
        job_id: ticket.job_id,
        request_id: ticket.request_id.clone(),
        request: ticket.request.clone(),
        route: ticket.route.clone(),
        resource: ticket.resource,
        deadline: ticket.deadline,
        cancellation: ticket.cancellation.clone(),
        queue_wait: ticket.queue_wait,
    });
    worker_exchange(
        sender,
        ManagedWorkerEvent::ComfyCandidate {
            ticket,
            candidate,
            generation: Box::new(generation),
            reply,
        },
        result,
        context,
    )?
    .ok_or(())
}

fn publish_comfy_acknowledgement(
    sender: &SyncSender<ManagedWorkerEvent>,
    acknowledgement: super::comfy::ManagedComfyBackendAcknowledgement,
    context: &OperationContext<'_>,
) -> Result<(), ()> {
    let (reply, result) = mpsc::sync_channel(1);
    if worker_exchange(
        sender,
        ManagedWorkerEvent::ComfyBackendAcknowledged {
            acknowledgement: Box::new(acknowledgement),
            reply,
        },
        result,
        context,
    )? {
        Ok(())
    } else {
        Err(())
    }
}

struct ManagedLocalTicketExecutor<'executor, 'host> {
    config: &'executor ManagedSessionConfig,
    owner: &'executor mut ManagedRuntimeOwner<'host>,
    resident: &'executor mut Option<super::managed_service::OwnedService>,
    cgroups: &'executor CgroupReleaseProbe,
    gpu: &'executor NvidiaSmiGpuProbe,
    session_cancellation: &'executor ManagedWorkerCancellation,
    session_deadline: std::time::Instant,
    events: &'executor SyncSender<ManagedWorkerEvent>,
}

impl ManagedLocalTicketExecutor<'_, '_> {
    fn execute(&mut self, ticket: DispatchTicket) -> ManagedJobCompletion {
        use super::service_owner::{
            DeviceSelection, JobExecutionBounds, OwnedJobFinalizeErrorKind, OwnedJobRequestSpec,
            ReleasedNativeOutcome,
        };
        let mut outcome = uncertain_managed_completion(&ticket);
        if !self
            .config
            .local_job_profiles
            .contains(ticket.request.profile())
            || !supported_local_job_route(&ticket.route)
        {
            return outcome;
        }
        // One worker owns native ingress and the daemon has one active slot.
        // CPU decisions need no GPU transition; CUDA jobs must release the
        // resident language engine before creating a native child.
        if ticket.route.device == Some(Device::Cuda)
            && !stop_resident(
                self.resident,
                self.owner,
                self.cgroups,
                self.gpu,
                self.config.recovery,
            )
        {
            return outcome;
        }
        let cancellation = RequestAndSessionCancellation {
            request: &ticket.cancellation,
            session: self.session_cancellation,
        };
        let deadline = std::time::Instant::now()
            .checked_add(self.config.request_timeout)
            .unwrap_or(ticket.deadline)
            .min(ticket.deadline)
            .min(self.session_deadline);
        let context = OperationContext::new(deadline, &cancellation);
        if context.is_cancelled() || context.remaining().is_zero() {
            outcome.completion = if context.is_cancelled() {
                Completion::Cancelled
            } else {
                Completion::Failed
            };
            outcome.result_terminal = if context.is_cancelled() {
                ResultTerminal::Cancelled
            } else {
                ResultTerminal::Failed(ResultFailureCode::TimedOut)
            };
            outcome.cleanup_proven = true;
            outcome.stop_session = self.session_cancellation.is_cancelled();
            return outcome;
        }
        if ticket.route.device == Some(Device::Cuda)
            && observe_gpu_idle_retry(self.gpu, &context, self.config.recovery).is_err()
        {
            // A previous job's idle snapshot is never activation authority.
            // Do not stop an unrelated claimant or launch into its allocation.
            return outcome;
        }
        let Some(directory) = allocate_local_job_directory(self.config, ticket.job_id) else {
            return outcome;
        };
        let device = match ticket.route.device {
            Some(Device::Cpu) => DeviceSelection::Cpu,
            Some(Device::Cuda) => match DeviceSelection::nvidia(
                self.config.gpu.gpu_uuid(),
                self.config.gpu.expected_index(),
                self.config.gpu.max_idle_mib(),
            ) {
                Ok(device) => device,
                Err(_) => return outcome,
            },
            _ => return outcome,
        };
        let Ok(bounds) = JobExecutionBounds::new(context.remaining(), self.config.recovery) else {
            return outcome;
        };
        let job = match self.owner.start_owned_local_job_request(
            &directory,
            OwnedJobRequestSpec {
                job_id: ticket.job_id,
                request_id: &ticket.request_id,
                repository_root: &self.config.repository_root,
                request: &ticket.request,
                accepted_route: &ticket.route,
                device,
                bounds,
            },
            self.config.launch_plan.stop_timeout,
            &context,
        ) {
            Ok(job) => job,
            Err(error) => {
                if error.intent_record().is_none() {
                    outcome.completion = Completion::Failed;
                    outcome.result_terminal = ResultTerminal::Failed(ResultFailureCode::Internal);
                    outcome.cleanup_proven = true;
                    outcome.stop_session = false;
                }
                return outcome;
            }
        };
        while !context.is_cancelled() && !context.remaining().is_zero() {
            if job.observe_native_result().is_ok() {
                break;
            }
            thread::sleep(WORKER_POLL.min(context.remaining()));
        }
        let cancelled = context.is_cancelled();
        let timed_out = context.remaining().is_zero();
        let Ok(cgroups) = CgroupReleaseProbe::production() else {
            return outcome;
        };
        let mut finalizer = self.owner.begin_owned_job_finalization(job, cgroups);
        let cleanup = recovery_context(self.config.recovery);
        let released = loop {
            match finalizer.finalize(&cleanup) {
                Ok(result) => break Some(result),
                Err(error)
                    if matches!(
                        error.kind(),
                        OwnedJobFinalizeErrorKind::AlreadyReleased
                            | OwnedJobFinalizeErrorKind::InvalidAssociation
                            | OwnedJobFinalizeErrorKind::StopUncertain
                            | OwnedJobFinalizeErrorKind::Stop(_)
                            | OwnedJobFinalizeErrorKind::RetirementUncertain
                    ) =>
                {
                    break None;
                }
                Err(_) => {
                    if cleanup.remaining().is_zero() {
                        break None;
                    }
                    thread::sleep(WORKER_POLL.min(cleanup.remaining()));
                }
            }
        };
        let Some(released) = released else {
            return outcome;
        };
        outcome.cleanup_proven = true;
        outcome.stop_session = self.session_cancellation.is_cancelled();
        if cancelled || timed_out {
            outcome.completion = if cancelled {
                Completion::Cancelled
            } else {
                Completion::Failed
            };
            outcome.result_terminal = if cancelled {
                ResultTerminal::Cancelled
            } else {
                ResultTerminal::Failed(ResultFailureCode::TimedOut)
            };
            return outcome;
        }
        let mut published_files = None;
        let event = match released {
            ReleasedNativeOutcome::Decision(native) => match native.result {
                InferenceResult::Decision { answers } => ResultEvent::DecisionResult {
                    answers,
                    native_latency: native.native_latency,
                    native_usage: native.native_usage,
                },
                _ => return outcome,
            },
            ReleasedNativeOutcome::Breeze { mode: _, artifacts } => {
                let Some(prefix) = directory.file_name().and_then(|name| name.to_str()) else {
                    return outcome;
                };
                let mut located = Vec::with_capacity(artifacts.len());
                let mut files = Vec::with_capacity(artifacts.len());
                for artifact in artifacts {
                    let path = format!("{prefix}/{}", artifact.relative_path);
                    let Ok(reference) =
                        super::result_buffer::MediaArtifact::new(artifact.kind, path.clone())
                    else {
                        return outcome;
                    };
                    let Ok(file) = super::artifact_store::PublishedArtifact::capture(
                        &directory,
                        reference.clone(),
                        &artifact.relative_path,
                        &cleanup,
                    ) else {
                        return outcome;
                    };
                    files.push(file);
                    outcome.artifacts.push(path);
                    located.push(reference);
                }
                published_files = Some(files);
                ResultEvent::MediaArtifacts(located)
            }
            ReleasedNativeOutcome::Failed(failure) => {
                (outcome.completion, outcome.result_terminal) = released_native_failure(failure);
                return outcome;
            }
        };
        // Publication uses an independent bound after the job is released,
        // never a request deadline that expired during cleanup.
        let published = match published_files {
            Some(files) => publish_artifacts(self.events, &outcome.key, files, &cleanup),
            None => publish_result_event(self.events, &outcome.key, event, &cleanup),
        };
        if published.is_err() {
            outcome.stop_session = true;
            return outcome;
        }
        outcome.completion = Completion::Succeeded;
        outcome.result_terminal = ResultTerminal::Succeeded;
        outcome
    }
}

/// Release proof permits resource accounting, not rewriting the native
/// attempt's outcome. In particular an uncertain attempt stays interrupted.
fn released_native_failure(
    failure: super::owned_job_store::NativeJobFailure,
) -> (Completion, ResultTerminal) {
    use super::owned_job_store::NativeJobFailure;
    match failure {
        NativeJobFailure::Uncertain => (Completion::Uncertain, ResultTerminal::Interrupted),
        NativeJobFailure::Cancelled => (Completion::Cancelled, ResultTerminal::Cancelled),
        failure => (
            Completion::Failed,
            ResultTerminal::Failed(match failure {
                NativeJobFailure::NotDispatched | NativeJobFailure::Unavailable => {
                    ResultFailureCode::Unavailable
                }
                NativeJobFailure::TimedOut => ResultFailureCode::TimedOut,
                NativeJobFailure::InvalidOutput => ResultFailureCode::InvalidOutput,
                NativeJobFailure::Internal => ResultFailureCode::Internal,
                NativeJobFailure::Cancelled | NativeJobFailure::Uncertain => unreachable!(),
            }),
        ),
    }
}

fn allocate_local_job_directory(config: &ManagedSessionConfig, job_id: u64) -> Option<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    static NEXT_DIRECTORY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let root = config.job_directory_root.as_ref()?;
    let pinned = super::owned_job_store::OwnedJobStore::open(root).ok()?;
    if Some((pinned.directory_device(), pinned.directory_inode())) != config.job_directory_identity
    {
        return None;
    }
    let nonce = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let path = root.join(format!("job-{job_id}-{}-{nonce}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&path).ok()?;
    let rechecked = super::owned_job_store::OwnedJobStore::open(root).ok()?;
    if Some((rechecked.directory_device(), rechecked.directory_inode()))
        != config.job_directory_identity
    {
        return None;
    }
    Some(path)
}

fn reconcile_retained_services(
    owner: &mut ManagedRuntimeOwner<'_>,
    cgroups: &CgroupReleaseProbe,
    gpu: &NvidiaSmiGpuProbe,
    config: &ManagedSessionConfig,
) -> Result<(), ()> {
    let context = recovery_context(config.recovery);
    let retained = owner.reconcile(&context).map_err(|_| ())?;
    // Validate the complete retained set before the first name-targeted Stop.
    // A later unsupported/unknown intent must not cause partial cleanup of an
    // earlier service. Mixed-runtime admission remains closed until durable
    // stop-stage and job/history association are implemented.
    preflight_retained_runtime_intents(
        &config.runtime,
        retained.iter().map(|observation| &observation.intent),
    )?;
    if retained.iter().any(|observation| {
        matches!(
            observation.observation,
            ReconcileResult::AbsentNow | ReconcileResult::Unknown
        )
    }) {
        return Err(());
    }
    let mut stopped = Vec::with_capacity(retained.len());
    for observation in retained {
        match observation.observation {
            ReconcileResult::Owned(service) => {
                let receipt = owner.stop(&service, &context).map_err(|_| ())?;
                stopped.push(observe_stopped_retry(owner, receipt, cgroups, &context)?);
            }
            ReconcileResult::StopOnly(service) => {
                let receipt = owner.stop_stop_only(&service, &context).map_err(|_| ())?;
                stopped.push(observe_stopped_retry(owner, receipt, cgroups, &context)?);
            }
            ReconcileResult::AbsentNow | ReconcileResult::Unknown => return Err(()),
        }
    }
    observe_gpu_idle_retry(gpu, &context, config.recovery)?;
    for observation in &stopped {
        retire_stopped_retry(owner, observation, cgroups, &context)?;
    }
    Ok(())
}

fn preflight_retained_runtime_intents<'a>(
    runtime: &RuntimeIdentity,
    intents: impl IntoIterator<Item = &'a super::managed_service::ServiceIntentRecord>,
) -> Result<(), ()> {
    for intent in intents {
        if intent.runtime_identity() != Some(runtime)
            || !matches!(
                intent.launch_role(),
                ServiceLaunchRole::IsolatedEngineSupervisor(_)
            )
        {
            return Err(());
        }
    }
    Ok(())
}

fn stop_owned_service(
    owner: &mut ManagedRuntimeOwner<'_>,
    service: &super::managed_service::OwnedService,
    cgroups: &CgroupReleaseProbe,
    gpu: &NvidiaSmiGpuProbe,
    recovery: RecoveryPolicy,
) -> bool {
    let context = recovery_context(recovery);
    let Ok(receipt) = owner.stop(service, &context) else {
        return false;
    };
    let Ok(stopped) = observe_stopped_retry(owner, receipt, cgroups, &context) else {
        return false;
    };
    if observe_gpu_idle_retry(gpu, &context, recovery).is_err() {
        return false;
    }
    retire_stopped_retry(owner, &stopped, cgroups, &context).is_ok()
}

fn observe_stopped_retry(
    owner: &mut ManagedRuntimeOwner<'_>,
    mut receipt: super::managed_service::StoppedService,
    cgroups: &CgroupReleaseProbe,
    context: &OperationContext<'_>,
) -> Result<StoppedServiceObservation, ()> {
    loop {
        match owner.observe_stopped(receipt, cgroups, context) {
            Ok(stopped) => return Ok(stopped),
            Err(error) => {
                let Some(retained_receipt) = error.into_receipt() else {
                    return Err(());
                };
                receipt = retained_receipt;
                if context.is_cancelled() || context.remaining().is_zero() {
                    return Err(());
                }
                thread::sleep(WORKER_POLL.min(context.remaining()));
            }
        }
    }
}

fn observe_gpu_idle_retry(
    gpu: &NvidiaSmiGpuProbe,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<(), ()> {
    loop {
        if gpu.observe_idle(context, recovery).is_ok() {
            return Ok(());
        }
        if context.is_cancelled() || context.remaining().is_zero() {
            return Err(());
        }
        thread::sleep(WORKER_POLL.min(context.remaining()));
    }
}

fn retire_stopped_retry(
    owner: &mut ManagedRuntimeOwner<'_>,
    stopped: &StoppedServiceObservation,
    cgroups: &CgroupReleaseProbe,
    context: &OperationContext<'_>,
) -> Result<(), ()> {
    loop {
        match owner.retire_stopped(stopped, cgroups, context) {
            Ok(()) => return Ok(()),
            Err(error)
                if error.disposition()
                    == super::service_owner::IntentRetirementDisposition::NotRetired =>
            {
                if context.is_cancelled() || context.remaining().is_zero() {
                    return Err(());
                }
                thread::sleep(WORKER_POLL.min(context.remaining()));
            }
            Err(_) => return Err(()),
        }
    }
}

fn resolve_startup_route(
    catalog: &Catalog,
    config: &ManagedSessionConfig,
) -> Option<ResolvedProfileRoute> {
    let refreshed = catalog.refresh().ok()?;
    let route = resolve_profile_route(&refreshed, &config.startup_profile).ok()?;
    let NativeRoute::LlamaSwapModel { model_id, .. } = &route.native_route else {
        return None;
    };
    if route.runtime != config.runtime
        || !config
            .supported_model_tasks
            .get(model_id.as_str())
            .is_some_and(|tasks| route.tasks.iter().all(|task| tasks.contains(task.as_str())))
    {
        return None;
    }
    Some(route)
}

struct RequestAndSessionCancellation<'a> {
    request: &'a RequestCancellation,
    session: &'a ManagedWorkerCancellation,
}

impl Cancellation for RequestAndSessionCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        self.request.is_cancelled() || self.session.is_cancelled()
    }
}

struct ManagedTicketExecutor<'executor, 'host> {
    config: &'executor ManagedSessionConfig,
    catalog: &'executor Catalog,
    adapter: &'executor LanguageAdapter,
    owner: &'executor mut ManagedRuntimeOwner<'host>,
    service: &'executor super::managed_service::OwnedService,
    session_cancellation: &'executor ManagedWorkerCancellation,
    session_deadline: std::time::Instant,
    events: &'executor SyncSender<ManagedWorkerEvent>,
}

impl ManagedTicketExecutor<'_, '_> {
    fn execute(&mut self, ticket: DispatchTicket) -> ManagedJobCompletion {
        let config = self.config;
        let catalog = self.catalog;
        let adapter = self.adapter;
        let owner = &mut *self.owner;
        let service = self.service;
        let session_cancellation = self.session_cancellation;
        let session_deadline = self.session_deadline;
        let events = self.events;
        let key = ManagedJobKey {
            job_id: ticket.job_id,
            request_id: ticket.request_id.clone(),
        };
        let mut outcome = ManagedJobCompletion {
            key: key.clone(),
            completion: Completion::Failed,
            result_terminal: ResultTerminal::Failed(ResultFailureCode::Internal),
            cleanup_proven: false,
            stop_session: false,
            artifacts: Vec::new(),
        };
        let NativeRoute::LlamaSwapModel { model_id, .. } = &ticket.route.native_route else {
            outcome.stop_session = true;
            return outcome;
        };
        let refreshed_route = catalog
            .refresh()
            .ok()
            .and_then(|fresh| resolve_profile_route(&fresh, ticket.request.profile()).ok());
        if refreshed_route.as_ref() != Some(&ticket.route) {
            outcome.stop_session = true;
            return outcome;
        }
        if ticket.route.runtime != config.runtime
            || !config
                .supported_model_tasks
                .get(model_id.as_str())
                .is_some_and(|tasks| tasks.contains(ticket.request.task()))
            || !matches!(
                ticket.request,
                InferenceRequest::Chat { .. } | InferenceRequest::Embeddings { .. }
            )
        {
            outcome.stop_session = true;
            return outcome;
        }
        let request_deadline = std::time::Instant::now()
            .checked_add(config.request_timeout)
            .unwrap_or(ticket.deadline)
            .min(ticket.deadline)
            .min(session_deadline);
        let cancellation = RequestAndSessionCancellation {
            request: &ticket.cancellation,
            session: session_cancellation,
        };
        let context = OperationContext::new(request_deadline, &cancellation);
        if context.is_cancelled() {
            outcome.completion = Completion::Cancelled;
            outcome.result_terminal = ResultTerminal::Cancelled;
            outcome.stop_session = session_cancellation.is_cancelled();
            return outcome;
        }

        let model_id = model_id.as_str();
        if config.verify_private_router_config_unchanged().is_err() {
            outcome.stop_session = true;
            return outcome;
        }
        let activation = activate_llama_swap_model_once(
            owner,
            service,
            &config.transport_directory,
            NativeEngineKind::LlamaServer,
            model_id,
            &context,
        );
        match activation {
            Ok(ack) if ack.matches_model(model_id) => {}
            Ok(_) => {
                outcome.stop_session = true;
                return outcome;
            }
            Err(error) => {
                outcome.result_terminal =
                    if error.disposition() == ActivationDisposition::Unconfirmed {
                        ResultTerminal::Interrupted
                    } else {
                        ResultTerminal::Failed(ResultFailureCode::Unavailable)
                    };
                outcome.completion = if error.disposition() == ActivationDisposition::Unconfirmed {
                    Completion::Uncertain
                } else {
                    Completion::Failed
                };
                outcome.stop_session = true;
                return outcome;
            }
        }

        if config.verify_private_router_config_unchanged().is_err() {
            outcome.stop_session = true;
            return outcome;
        }
        if probe_llama_swap_router_model_readiness(
            owner,
            service,
            &config.transport_directory,
            model_id,
            &context,
        )
        .is_err()
        {
            outcome.completion = Completion::Uncertain;
            outcome.result_terminal = ResultTerminal::Interrupted;
            outcome.stop_session = true;
            return outcome;
        }
        // Activation can take long enough for manifests to change. Re-resolve
        // the complete accepted route after readiness and before creating the
        // inference request; the adapter performs its own final route check too.
        let route_still_current = catalog
            .refresh()
            .ok()
            .and_then(|fresh| resolve_profile_route(&fresh, ticket.request.profile()).ok())
            .is_some_and(|current| current == ticket.route);
        if !route_still_current {
            outcome.stop_session = true;
            return outcome;
        }
        if config.verify_private_router_config_unchanged().is_err() {
            outcome.stop_session = true;
            return outcome;
        }

        let result = match &ticket.request {
            InferenceRequest::Chat { .. } => adapter.execute_with_events_for_route(
                catalog,
                &ticket.request,
                &ticket.route,
                &context,
                |event| publish_chat_event(events, &key, event, &context),
            ),
            InferenceRequest::Embeddings { .. } => adapter.execute_embeddings_for_route(
                catalog,
                &ticket.request,
                &ticket.route,
                &context,
            ),
            InferenceRequest::Media { .. } | InferenceRequest::Decision { .. } => unreachable!(),
        };
        match result {
            Ok(result) if result.validate_for(&ticket.request).is_ok() => {
                if let InferenceResult::Embeddings { vectors, usage } = result {
                    if publish_result_event(
                        events,
                        &key,
                        ResultEvent::Embeddings { vectors, usage },
                        &context,
                    )
                    .is_err()
                    {
                        outcome.completion = Completion::Uncertain;
                        outcome.result_terminal = ResultTerminal::Interrupted;
                        outcome.stop_session = true;
                        return outcome;
                    }
                }
                outcome.completion = Completion::Succeeded;
                outcome.result_terminal = ResultTerminal::Succeeded;
            }
            Ok(_) => {
                outcome.completion = Completion::Uncertain;
                outcome.result_terminal = ResultTerminal::Interrupted;
                outcome.stop_session = true;
            }
            Err(error) if error.disposition == DispatchDisposition::Uncertain => {
                outcome.completion = Completion::Uncertain;
                outcome.result_terminal = ResultTerminal::Interrupted;
                outcome.stop_session = true;
            }
            Err(error) => {
                match error.kind {
                    LanguageErrorKind::Cancelled => {
                        outcome.completion = Completion::Cancelled;
                        outcome.result_terminal = ResultTerminal::Cancelled;
                    }
                    LanguageErrorKind::Deadline => {
                        outcome.completion = Completion::Failed;
                        outcome.result_terminal =
                            ResultTerminal::Failed(ResultFailureCode::TimedOut);
                    }
                    _ => {
                        outcome.completion = Completion::Failed;
                        outcome.result_terminal =
                            ResultTerminal::Failed(ResultFailureCode::InvalidOutput);
                        outcome.stop_session = true;
                    }
                }
                if session_cancellation.is_cancelled() {
                    outcome.stop_session = true;
                }
            }
        }
        outcome
    }
}

fn publish_chat_event(
    sender: &SyncSender<ManagedWorkerEvent>,
    key: &ManagedJobKey,
    event: ChatStreamEvent<'_>,
    context: &OperationContext<'_>,
) -> Result<(), ChatEventCallbackError> {
    let event = match event {
        ChatStreamEvent::Delta(super::language::ChatDelta::Content(text)) => {
            ResultEvent::ChatContent(text.to_owned())
        }
        ChatStreamEvent::Delta(super::language::ChatDelta::Reasoning(text)) => {
            ResultEvent::ChatReasoning(text.to_owned())
        }
        ChatStreamEvent::Usage { completion_tokens } => {
            ResultEvent::ChatUsage { completion_tokens }
        }
        ChatStreamEvent::StreamComplete => ResultEvent::ChatStreamComplete,
    };
    publish_result_event(sender, key, event, context).map_err(|_| {
        if context.is_cancelled() {
            ChatEventCallbackError::Cancelled
        } else {
            ChatEventCallbackError::Publication
        }
    })
}

fn publish_artifacts(
    sender: &SyncSender<ManagedWorkerEvent>,
    key: &ManagedJobKey,
    files: Vec<super::artifact_store::PublishedArtifact>,
    context: &OperationContext<'_>,
) -> Result<(), ()> {
    let (reply, result) = mpsc::sync_channel(1);
    if worker_exchange(
        sender,
        ManagedWorkerEvent::PublishArtifacts {
            key: key.clone(),
            files,
            reply,
        },
        result,
        context,
    )? {
        Ok(())
    } else {
        Err(())
    }
}

fn publish_result_event(
    sender: &SyncSender<ManagedWorkerEvent>,
    key: &ManagedJobKey,
    event: ResultEvent,
    context: &OperationContext<'_>,
) -> Result<(), ()> {
    let (reply, result) = mpsc::sync_channel(1);
    let message = ManagedWorkerEvent::Append {
        key: key.clone(),
        event,
        reply,
    };
    match worker_exchange(sender, message, result, context)? {
        true => Ok(()),
        false => Err(()),
    }
}

/// Bounded, single-send exchange. Losing the reply never retries the message:
/// the main thread may already have committed its durable effect.
fn worker_exchange<T>(
    sender: &SyncSender<ManagedWorkerEvent>,
    mut message: ManagedWorkerEvent,
    result: Receiver<T>,
    context: &OperationContext<'_>,
) -> Result<T, ()> {
    if context.is_cancelled() || context.remaining().is_zero() {
        return Err(());
    }
    loop {
        if context.is_cancelled() || context.remaining().is_zero() {
            return Err(());
        }
        match sender.try_send(message) {
            Ok(()) => break,
            Err(mpsc::TrySendError::Disconnected(_)) => return Err(()),
            Err(mpsc::TrySendError::Full(returned)) => {
                message = returned;
                thread::sleep(WORKER_POLL.min(context.remaining()));
            }
        }
    }
    loop {
        if context.is_cancelled() || context.remaining().is_zero() {
            return Err(());
        }
        match result.recv_timeout(WORKER_POLL.min(context.remaining())) {
            Ok(reply) => return Ok(reply),
            Err(RecvTimeoutError::Disconnected) => return Err(()),
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

fn validate_private_router_config(
    path: &Path,
    native_config: &Path,
) -> Result<PathBuf, ManagedConfigError> {
    if !path.is_absolute() {
        return Err(ManagedConfigError::InvalidRouterConfig);
    }
    let entry =
        fs::symlink_metadata(path).map_err(|_| ManagedConfigError::RouterConfigUnavailable)?;
    if entry.file_type().is_symlink() {
        return Err(ManagedConfigError::InvalidRouterConfig);
    }
    let path = path
        .canonicalize()
        .map_err(|_| ManagedConfigError::RouterConfigUnavailable)?;
    let native_config = native_config
        .canonicalize()
        .map_err(|_| ManagedConfigError::InvalidRouterConfig)?;
    if path == native_config {
        return Err(ManagedConfigError::InvalidRouterConfig);
    }
    let _ = read_router_config_snapshot(&path)?;
    Ok(path)
}

struct RouterConfigSnapshot {
    identity: (u64, u64),
    bytes: Vec<u8>,
    model_ids: BTreeSet<String>,
}

fn read_router_config_snapshot(path: &Path) -> Result<RouterConfigSnapshot, ManagedConfigError> {
    let entry =
        fs::symlink_metadata(path).map_err(|_| ManagedConfigError::RouterConfigUnavailable)?;
    let uid = unsafe { libc::geteuid() };
    if !entry.is_file()
        || entry.uid() != uid
        || entry.mode() & 0o7777 != 0o600
        || entry.nlink() != 1
        || entry.len() > MAX_ROUTER_CONFIG_BYTES
    {
        return Err(ManagedConfigError::InvalidRouterConfig);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| ManagedConfigError::RouterConfigUnavailable)?;
    let opened = file
        .metadata()
        .map_err(|_| ManagedConfigError::RouterConfigUnavailable)?;
    if !opened.is_file()
        || opened.uid() != uid
        || opened.mode() & 0o7777 != 0o600
        || opened.nlink() != 1
        || (opened.dev(), opened.ino()) != (entry.dev(), entry.ino())
    {
        return Err(ManagedConfigError::InvalidRouterConfig);
    }
    let mut bytes = Vec::with_capacity(entry.len() as usize);
    file.take(MAX_ROUTER_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| ManagedConfigError::RouterConfigUnavailable)?;
    if bytes.len() as u64 > MAX_ROUTER_CONFIG_BYTES {
        return Err(ManagedConfigError::InvalidRouterConfig);
    }
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    let UniqueJson(value) = UniqueJson::deserialize(&mut deserializer)
        .map_err(|_| ManagedConfigError::InvalidRouterConfig)?;
    deserializer
        .end()
        .map_err(|_| ManagedConfigError::InvalidRouterConfig)?;
    if contains_key_recursive(&value, "apiKeys") {
        return Err(ManagedConfigError::RouterConfigHasApiKeys);
    }
    if contains_non_root_environment_macro(&value) {
        return Err(ManagedConfigError::RouterConfigUsesAmbientEnvironment);
    }
    let models = value
        .as_object()
        .and_then(|document| document.get("models"))
        .and_then(Value::as_object)
        .ok_or(ManagedConfigError::InvalidRouterConfig)?;
    let model_ids = models.keys().cloned().collect();
    Ok(RouterConfigSnapshot {
        identity: (opened.dev(), opened.ino()),
        bytes,
        model_ids,
    })
}

fn contains_key_recursive(value: &Value, expected: &str) -> bool {
    match value {
        Value::Object(object) => object
            .iter()
            .any(|(key, nested)| key == expected || contains_key_recursive(nested, expected)),
        Value::Array(items) => items
            .iter()
            .any(|nested| contains_key_recursive(nested, expected)),
        _ => false,
    }
}

fn contains_non_root_environment_macro(value: &Value) -> bool {
    match value {
        Value::String(text) => {
            let mut rest = text.as_str();
            while let Some((_, after)) = rest.split_once("${env.") {
                let Some((name, _)) = after.split_once('}') else {
                    return true;
                };
                if name != "L3MS_ROOT" {
                    return true;
                }
                rest = after;
            }
            false
        }
        Value::Object(object) => object.iter().any(|(key, nested)| {
            contains_non_root_environment_macro(&Value::String(key.clone()))
                || contains_non_root_environment_macro(nested)
        }),
        Value::Array(items) => items.iter().any(contains_non_root_environment_macro),
        _ => false,
    }
}

struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueVisitor;

        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueJson;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::from(value)))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::from(value)))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(Value::Number)
                    .map(UniqueJson)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value)))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items = Vec::new();
                while let Some(UniqueJson(item)) = sequence.next_element()? {
                    items.push(item);
                }
                Ok(UniqueJson(Value::Array(items)))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut object = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    let UniqueJson(value) = map.next_value()?;
                    if object.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate JSON key"));
                    }
                    object.insert(key, value);
                }
                Ok(UniqueJson(Value::Object(object)))
            }
        }

        deserializer.deserialize_any(UniqueVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn retained_runtime_preflight_rejects_whole_mixed_or_legacy_batch() {
        use super::super::managed_service::ServiceIntentRecord;
        let language = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let media = RuntimeIdentity::new(Lane::Media, "comfyui").unwrap();
        let coordinator =
            super::super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                .unwrap();
        let intent = |suffix: u128, runtime: Option<RuntimeIdentity>, role: ServiceLaunchRole| {
            ServiceIntentRecord::from_persisted_parts_with_role(
                "coordinator".into(),
                format!("l3ms-coordinator-{suffix:032x}.service"),
                "b".repeat(64),
                30_000,
                runtime,
                role,
            )
            .unwrap()
        };
        let supported = intent(
            1,
            Some(language.clone()),
            ServiceLaunchRole::IsolatedEngineSupervisor(coordinator),
        );
        let unsupported = intent(
            2,
            Some(media),
            ServiceLaunchRole::IsolatedEngineSupervisor(coordinator),
        );
        let legacy = intent(3, None, ServiceLaunchRole::Ordinary);
        let ordinary_language = intent(4, Some(language.clone()), ServiceLaunchRole::Ordinary);
        assert!(preflight_retained_runtime_intents(&language, []).is_ok());
        assert!(preflight_retained_runtime_intents(&language, [&supported]).is_ok());
        for denied in [&unsupported, &legacy, &ordinary_language] {
            assert!(preflight_retained_runtime_intents(&language, [&supported, denied]).is_err());
            assert!(preflight_retained_runtime_intents(&language, [denied, &supported]).is_err());
        }
    }

    #[test]
    fn resident_cleanup_cannot_release_an_unproven_local_job() {
        assert!(!combined_cleanup_proven(true, true, false));
        assert!(!combined_cleanup_proven(false, true, true));
        assert!(!combined_cleanup_proven(false, false, true));
        assert!(combined_cleanup_proven(true, true, true));
        assert!(combined_cleanup_proven(true, false, false));
    }

    #[test]
    fn worker_exchange_delivers_once_and_returns_the_main_thread_reply() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (reply, result) = mpsc::sync_channel(1);
        let context = OperationContext::new(
            std::time::Instant::now() + Duration::from_secs(1),
            &NEVER_CANCEL,
        );
        thread::scope(|scope| {
            let main = scope.spawn(move || {
                match receiver.recv().unwrap() {
                    ManagedWorkerEvent::Append { reply, .. } => reply.send(false).unwrap(),
                    _ => panic!("wrong exchange"),
                }
                receiver
            });
            assert!(!worker_exchange(
                &sender,
                ManagedWorkerEvent::Append {
                    key: ManagedJobKey {
                        job_id: 1,
                        request_id: "exchange".into()
                    },
                    event: ResultEvent::ChatContent("private".into()),
                    reply,
                },
                result,
                &context
            )
            .unwrap());
            assert!(matches!(
                main.join().unwrap().try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
        });
    }

    #[test]
    fn worker_exchange_lost_reply_never_resends_a_committed_message() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (reply, result) = mpsc::sync_channel(1);
        let context = OperationContext::new(
            std::time::Instant::now() + Duration::from_millis(50),
            &NEVER_CANCEL,
        );
        thread::scope(|scope| {
            let main = scope.spawn(move || {
                let message = receiver.recv().unwrap();
                drop(message); // Main effect could already be durable.
                receiver
            });
            assert!(worker_exchange(
                &sender,
                ManagedWorkerEvent::Append {
                    key: ManagedJobKey {
                        job_id: 1,
                        request_id: "exchange".into()
                    },
                    event: ResultEvent::ChatContent("private".into()),
                    reply,
                },
                result,
                &context
            )
            .is_err());
            assert!(matches!(
                main.join().unwrap().try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
        });
    }

    #[test]
    fn worker_exchange_deadline_does_not_dispatch_into_an_available_channel() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (reply, result) = mpsc::sync_channel(1);
        let context = OperationContext::new(std::time::Instant::now(), &NEVER_CANCEL);
        assert!(worker_exchange(
            &sender,
            ManagedWorkerEvent::Append {
                key: ManagedJobKey {
                    job_id: 1,
                    request_id: "exchange".into()
                },
                event: ResultEvent::ChatContent("private".into()),
                reply,
            },
            result,
            &context
        )
        .is_err());
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn worker_exchange_reply_timeout_does_not_enqueue_another_effect() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let (reply, result) = mpsc::sync_channel(1);
        let context = OperationContext::new(
            std::time::Instant::now() + Duration::from_millis(20),
            &NEVER_CANCEL,
        );
        let started = std::time::Instant::now();
        assert!(worker_exchange(
            &sender,
            ManagedWorkerEvent::Append {
                key: ManagedJobKey {
                    job_id: 1,
                    request_id: "timeout".into()
                },
                event: ResultEvent::ChatContent("private".into()),
                reply,
            },
            result,
            &context,
        )
        .is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(matches!(
            receiver.try_recv(),
            Ok(ManagedWorkerEvent::Append { .. })
        ));
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn proven_release_does_not_reclassify_native_failure_or_uncertainty() {
        use super::super::owned_job_store::NativeJobFailure;
        for (native, completion, terminal) in [
            (
                NativeJobFailure::Uncertain,
                Completion::Uncertain,
                ResultTerminal::Interrupted,
            ),
            (
                NativeJobFailure::Cancelled,
                Completion::Cancelled,
                ResultTerminal::Cancelled,
            ),
            (
                NativeJobFailure::TimedOut,
                Completion::Failed,
                ResultTerminal::Failed(ResultFailureCode::TimedOut),
            ),
            (
                NativeJobFailure::Unavailable,
                Completion::Failed,
                ResultTerminal::Failed(ResultFailureCode::Unavailable),
            ),
            (
                NativeJobFailure::NotDispatched,
                Completion::Failed,
                ResultTerminal::Failed(ResultFailureCode::Unavailable),
            ),
            (
                NativeJobFailure::InvalidOutput,
                Completion::Failed,
                ResultTerminal::Failed(ResultFailureCode::InvalidOutput),
            ),
            (
                NativeJobFailure::Internal,
                Completion::Failed,
                ResultTerminal::Failed(ResultFailureCode::Internal),
            ),
        ] {
            assert_eq!(released_native_failure(native), (completion, terminal));
        }
    }

    fn private_file(directory: &Path, name: &str, body: &str) -> PathBuf {
        let path = directory.join(name);
        fs::write(&path, body).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        path
    }

    #[test]
    fn dedicated_router_config_is_strict_json_and_rejects_auth_ambient_secrets_and_duplicates() {
        let root = tempfile::tempdir().unwrap();
        let native = private_file(root.path(), "native.yaml", "{\"models\":{}}\n");
        let dedicated = private_file(
            root.path(),
            "private.yaml",
            "{\"models\":{\"local-model\":{\"cmd\":\"llama-server\"}}}\n",
        );
        assert!(validate_private_router_config(&dedicated, &native).is_ok());

        let keyed = private_file(
            root.path(),
            "keyed.yaml",
            "{\"apiKeys\":[\"${env.LLAMA_SWAP_API_KEY}\"],\"models\":{}}\n",
        );
        assert_eq!(
            validate_private_router_config(&keyed, &native),
            Err(ManagedConfigError::RouterConfigHasApiKeys)
        );

        let ambient = private_file(
            root.path(),
            "ambient.yaml",
            "{\"models\":{\"local-model\":{\"cmd\":\"${env.HOME}/llama-server\"}}}\n",
        );
        assert_eq!(
            validate_private_router_config(&ambient, &native),
            Err(ManagedConfigError::RouterConfigUsesAmbientEnvironment)
        );

        let disguised = private_file(
            root.path(),
            "disguised.yaml",
            r#"{"nested":{"apiKeys":[]},"models":{}}"#,
        );
        assert_eq!(
            validate_private_router_config(&disguised, &native),
            Err(ManagedConfigError::RouterConfigHasApiKeys)
        );

        let duplicate = private_file(
            root.path(),
            "duplicate.yaml",
            r#"{"apiKeys":[],"apiKeys":null,"models":{}}"#,
        );
        assert_eq!(
            validate_private_router_config(&duplicate, &native),
            Err(ManagedConfigError::InvalidRouterConfig)
        );
    }

    #[test]
    fn router_config_snapshot_reads_exact_model_map_and_detects_config_drift() {
        let root = tempfile::tempdir().unwrap();
        let native = private_file(root.path(), "native.yaml", "{}\n");
        let dedicated = private_file(
            root.path(),
            "private.yaml",
            r#"{"models":{"qwen-4":{"cmd":"llama-server"},"other":{"cmd":"server"}}}"#,
        );
        let canonical = validate_private_router_config(&dedicated, &native).unwrap();
        let snapshot = read_router_config_snapshot(&canonical).unwrap();
        assert!(snapshot.model_ids.contains("qwen-4"));
        assert!(!snapshot.model_ids.contains("qwen"));

        let original = snapshot.bytes.clone();
        fs::write(
            &canonical,
            r#"{"models":{"qwen-4":{"cmd":"changed"},"other":{"cmd":"server"}}}"#,
        )
        .unwrap();
        assert_ne!(fs::read(&canonical).unwrap(), original);
    }

    #[test]
    fn managed_cuda_launch_selection_comes_from_the_validated_probe_identity() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let gpu = GpuProbeConfig::new(UUID, 0, 128).unwrap();
        let selection = cuda_device_for_gpu_config(&gpu).unwrap();
        assert_eq!(selection.gpu_uuid(), gpu.gpu_uuid());

        // The probe accepts a syntactically valid case-insensitive UUID, but
        // the child-process handoff requires one canonical spelling.
        let noncanonical =
            GpuProbeConfig::new("GPU-01234567-89AB-cdef-0123-456789abcdef", 0, 128).unwrap();
        assert_eq!(
            cuda_device_for_gpu_config(&noncanonical),
            Err(ManagedConfigError::InvalidGpu)
        );
    }

    #[test]
    fn worker_event_delivery_is_bounded_and_handles_full_or_disconnected_channels() {
        let recovery = RecoveryPolicy::new(Duration::from_millis(20)).unwrap();
        let (sender, _receiver) = mpsc::sync_channel(1);
        assert!(sender.try_send(ManagedWorkerEvent::StartupFailed).is_ok());
        let started = std::time::Instant::now();
        assert!(!send_worker_event_bounded(
            &sender,
            ManagedWorkerEvent::StartupFailed,
            recovery,
        ));
        assert!(started.elapsed() < Duration::from_secs(1));

        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        assert!(!send_worker_event_bounded(
            &sender,
            ManagedWorkerEvent::StartupFailed,
            recovery,
        ));

        let (sender, receiver) = mpsc::sync_channel(1);
        assert!(send_worker_event_bounded(
            &sender,
            ManagedWorkerEvent::StartupFailed,
            recovery,
        ));
        assert!(matches!(
            receiver.try_recv(),
            Ok(ManagedWorkerEvent::StartupFailed)
        ));
    }

    #[test]
    fn lost_normal_completion_delivery_stops_service_but_terminal_delivery_does_not_stop_twice() {
        let recovery = RecoveryPolicy::new(Duration::from_millis(20)).unwrap();
        let (sender, _receiver) = mpsc::sync_channel(1);
        assert!(sender.try_send(ManagedWorkerEvent::StartupFailed).is_ok());
        let mut cleanup_calls = 0;
        let completion = ManagedJobCompletion {
            key: ManagedJobKey {
                job_id: 1,
                request_id: "request-1".into(),
            },
            completion: Completion::Succeeded,
            result_terminal: ResultTerminal::Succeeded,
            cleanup_proven: false,
            stop_session: false,
            artifacts: Vec::new(),
        };
        assert!(deliver_job_finished(&sender, completion, recovery, || {
            cleanup_calls += 1
        },));
        assert_eq!(cleanup_calls, 1);

        let (sender, _receiver) = mpsc::sync_channel(1);
        assert!(sender.try_send(ManagedWorkerEvent::StartupFailed).is_ok());
        let completion = ManagedJobCompletion {
            key: ManagedJobKey {
                job_id: 2,
                request_id: "request-2".into(),
            },
            completion: Completion::Uncertain,
            result_terminal: ResultTerminal::Interrupted,
            cleanup_proven: false,
            stop_session: true,
            artifacts: Vec::new(),
        };
        assert!(deliver_job_finished(&sender, completion, recovery, || {
            cleanup_calls += 1
        },));
        assert_eq!(cleanup_calls, 1);
    }
}
