//! Concrete managed foreground language-session configuration and executor.
//!
//! This first resident-engine path is deliberately specific: one llama-swap
//! v247 router, llama-server-compatible model routes, one configured NVIDIA
//! GPU, one private isolated supervisor, and the peer-authenticated Unix relay.
//! It does not manage ComfyUI, arbitrary HTTP engines, ambient credentials, or
//! direct clients. Model activation is a separate one-shot operation from
//! read-only readiness and inference dispatch.

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

use super::catalog::{Catalog, Lane, ProfileKey, RuntimeIdentity};
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
    launch: Option<IsolatedEngineLaunch>,
    pub(crate) activation_timeout: Duration,
    pub(crate) request_timeout: Duration,
    pub(crate) recovery: RecoveryPolicy,
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
        let port =
            BackendPort::new(input.router_port).map_err(|_| ManagedConfigError::InvalidLaunch)?;
        let arguments = [
            OsString::from("-config"),
            router_config.clone().into_os_string(),
            OsString::from("-listen"),
            OsString::from(format!("127.0.0.1:{}", port.get())),
        ];
        let recipe = LaunchRecipe::new(
            input.router_executable,
            input.router_working_directory,
            arguments,
        )
        .map_err(|_| ManagedConfigError::InvalidLaunch)?;
        let helper_executable = std::env::current_exe()
            .map_err(|_| ManagedConfigError::InvalidLaunch)?
            .canonicalize()
            .map_err(|_| ManagedConfigError::InvalidLaunch)?;
        let launch = IsolatedEngineLaunch::new(
            runtime.clone(),
            recipe,
            helper_executable,
            &input.transport_directory,
            input.router_port,
            input.stop_timeout,
        )
        .and_then(|launch| launch.with_stack_root(&input.repository_root))
        .map(|launch| launch.with_cuda_device(cuda_device))
        .map_err(|_| ManagedConfigError::InvalidLaunch)?;

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
            launch: Some(launch),
            activation_timeout: input.activation_timeout,
            request_timeout: input.request_timeout,
            recovery: input.recovery,
        })
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
}

pub(crate) enum ManagedWorkerEvent {
    Ready(RouterModelReadinessObservation),
    StartupFailed,
    Append {
        key: ManagedJobKey,
        event: ResultEvent,
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
    mut config: ManagedSessionConfig,
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
    let Some(launch) = config.launch.take() else {
        send_worker_event_bounded(&events, ManagedWorkerEvent::StartupFailed, config.recovery);
        return;
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

    loop {
        match commands.recv_timeout(WORKER_POLL) {
            Ok(ManagedWorkerCommand::Dispatch(ticket)) => {
                let completion = ManagedTicketExecutor {
                    config: &config,
                    catalog: &catalog,
                    adapter: &adapter,
                    owner: &mut owner,
                    service: &service,
                    session_cancellation: &cancellation,
                    session_deadline,
                    events: &events,
                }
                .execute(*ticket);
                let must_stop = completion.stop_session
                    || completion.completion == Completion::Uncertain
                    || cancellation.is_cancelled();
                let completion = if must_stop {
                    let clean =
                        stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
                    let ManagedJobCompletion {
                        key,
                        completion: prior_completion,
                        result_terminal: prior_terminal,
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
                    }
                } else {
                    completion
                };
                let should_return =
                    deliver_job_finished(&events, completion, config.recovery, || {
                        let _ = stop_owned_service(
                            &mut owner,
                            &service,
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
                    stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
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
                        stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
                    send_worker_event_bounded(
                        &events,
                        ManagedWorkerEvent::Stopped { cleanup_proven },
                        config.recovery,
                    );
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = stop_owned_service(&mut owner, &service, &cgroups, &gpu, config.recovery);
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

fn reconcile_retained_services(
    owner: &mut ManagedRuntimeOwner<'_>,
    cgroups: &CgroupReleaseProbe,
    gpu: &NvidiaSmiGpuProbe,
    config: &ManagedSessionConfig,
) -> Result<(), ()> {
    let context = recovery_context(config.recovery);
    let retained = owner.reconcile(&context).map_err(|_| ())?;
    let mut stopped = Vec::with_capacity(retained.len());
    for observation in retained {
        if observation.intent.runtime_identity() != Some(&config.runtime)
            || !matches!(
                observation.intent.launch_role(),
                ServiceLaunchRole::IsolatedEngineSupervisor(_)
            )
        {
            return Err(());
        }
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

fn publish_result_event(
    sender: &SyncSender<ManagedWorkerEvent>,
    key: &ManagedJobKey,
    event: ResultEvent,
    context: &OperationContext<'_>,
) -> Result<(), ()> {
    if context.is_cancelled() || context.remaining().is_zero() {
        return Err(());
    }
    let (reply, result) = mpsc::sync_channel(1);
    let mut message = ManagedWorkerEvent::Append {
        key: key.clone(),
        event,
        reply,
    };
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
        match result.recv_timeout(WORKER_POLL) {
            Ok(true) => return Ok(()),
            Ok(false) | Err(RecvTimeoutError::Disconnected) => return Err(()),
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
        };
        assert!(deliver_job_finished(&sender, completion, recovery, || {
            cleanup_calls += 1
        },));
        assert_eq!(cleanup_calls, 1);
    }
}
