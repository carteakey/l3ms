//! Persist-before-dispatch ownership of coordinator-created services.
//!
//! This composes the real intent store and controller under a retained host
//! guard. It is not an admission barrier or generic lifecycle adapter. Restored intents
//! are observed, never replayed; neither stop acknowledgment nor an absent
//! unit retires its record or proves resource release.
//! A lifecycle executor must reconcile retained intents and separately own
//! admission/resource reservations before calling start. Host exclusion alone
//! does not authorize engine activation or resolve GPU conflicts.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::catalog::RuntimeIdentity;
use super::cgroup_release::{
    CgroupObservation, CgroupObservationKind, CgroupReleaseErrorKind, CgroupReleaseProbe,
};
use super::cuda_device::CudaDevice;
use super::decision::NativeDecisionResult;
use super::engine_helper::{supervisor_arguments, BackendProcessSpec, StackRoot};
use super::engine_transport::{BackendPort, CoordinatorIdentityDescriptor};
#[cfg(test)]
use super::gpu_probe::GpuProbeConfig;
use super::gpu_probe::{GpuProbeFailureKind, NvidiaSmiGpuProbe};
use super::managed_service::{
    LaunchRecipe, ManagedServiceController, ManagedServiceErrorKind, ManagedServiceState,
    OwnedLocalJobBinding, OwnedService, OwnedServiceGenerationIdentity, ReconcileResult,
    ServiceIntentRecord, StartDisposition, StartFailure, StopOnlyService, StoppedService,
};
use super::owned_job_store::{
    BreezeArtifactKind, NativeJobFailure, NativeJobResult, OwnedJobStore, ValidatedOwnedJob,
};
pub use super::owned_job_store::{
    DeviceSelection, JobExecutionBounds, OwnedJobRequestSpec, OwnedJobStoreError,
};
use super::ownership::CoordinatorLock;
use super::requests::SpeechMode;
use super::result_buffer::{ArtifactKind, MediaArtifact};
use super::runtime::OperationContext;
use super::service_intents::{
    ServiceIntentRetireDisposition, ServiceIntentStore, ServiceIntentStoreError,
};

pub enum OwnerStartError {
    Preparation(ManagedServiceErrorKind),
    Persistence {
        intent: Box<ServiceIntentRecord>,
        kind: ServiceIntentStoreError,
    },
    NotDispatched {
        intent: Box<ServiceIntentRecord>,
        kind: ManagedServiceErrorKind,
    },
    Dispatch(StartFailure),
}

impl OwnerStartError {
    pub fn intent_record(&self) -> Option<&ServiceIntentRecord> {
        match self {
            Self::Preparation(_) => None,
            Self::Persistence { intent, .. } | Self::NotDispatched { intent, .. } => {
                Some(intent.as_ref())
            }
            Self::Dispatch(failure) => Some(failure.intent_record()),
        }
    }

    pub fn disposition(&self) -> StartDisposition {
        match self {
            Self::Dispatch(failure) => failure.disposition(),
            _ => StartDisposition::NotDispatched,
        }
    }
}

impl fmt::Debug for OwnerStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnerStartError")
            .field("disposition", &self.disposition())
            .field("intent", &"<retained privately>")
            .finish()
    }
}
impl fmt::Display for OwnerStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Preparation(_) => "service preparation failed",
            Self::Persistence { .. } => "service intent persistence failed; dispatch refused",
            Self::NotDispatched { .. } => "service operation ended before dispatch",
            Self::Dispatch(_) => "service dispatch failed; intent retained",
        })
    }
}
impl std::error::Error for OwnerStartError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnerObservationError {
    StoreUnavailable,
    ManagerUnavailable,
    Cancelled,
    DeadlineExceeded,
}
impl fmt::Display for OwnerObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("retained service observation failed")
    }
}
impl std::error::Error for OwnerObservationError {}

pub struct RetainedServiceObservation {
    pub intent: ServiceIntentRecord,
    pub observation: ReconcileResult,
}

/// A live owned wrapper plus its pinned private packet association. A native
/// result candidate never drops this handle or authorizes reservation release.
pub struct OwnedNativeJob {
    service: OwnedService,
    store: OwnedJobStore,
    job: ValidatedOwnedJob,
}

impl OwnedNativeJob {
    pub fn service(&self) -> &OwnedService {
        &self.service
    }
    pub fn job_id(&self) -> u64 {
        self.job.job_id()
    }
    pub fn request_id(&self) -> &str {
        self.job.request_id()
    }
    pub fn observe_native_result(&self) -> Result<NativeJobCandidate, OwnedJobStoreError> {
        self.store
            .read_result(&self.job)
            .map(|result| NativeJobCandidate { result })
    }
}

impl fmt::Debug for OwnedNativeJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("OwnedNativeJob(<retained private association>)")
    }
}

/// Private typed output that is deliberately not a published inference result.
pub struct NativeJobCandidate {
    result: NativeJobResult,
}
impl NativeJobCandidate {
    pub fn is_success(&self) -> bool {
        !matches!(self.result, NativeJobResult::Failed(_))
    }
}
impl fmt::Debug for NativeJobCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NativeJobCandidate(<unpublished>)")
    }
}

/// Native output that is publishable only after this owner has stopped the
/// exact job generation, observed its cgroup empty or removed, checked the
/// configured physical GPU when applicable, and durably retired the intent.
/// Breeze paths are relative LocalJob references; no output root is included.
pub(crate) enum ReleasedNativeOutcome {
    Decision(NativeDecisionResult),
    Breeze {
        mode: SpeechMode,
        artifacts: Vec<MediaArtifact>,
    },
    Failed(NativeJobFailure),
}

impl fmt::Debug for ReleasedNativeOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decision(_) => formatter.write_str("ReleasedNativeOutcome::Decision(<redacted>)"),
            Self::Breeze { mode, artifacts } => formatter
                .debug_struct("ReleasedNativeOutcome::Breeze")
                .field("mode", mode)
                .field("artifact_count", &artifacts.len())
                .field("artifacts", &"<redacted>")
                .finish(),
            Self::Failed(failure) => formatter
                .debug_tuple("ReleasedNativeOutcome::Failed")
                .field(failure)
                .finish(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnedJobFinalizeErrorKind {
    AlreadyReleased,
    InvalidAssociation,
    StopUncertain,
    Context(ManagedServiceErrorKind),
    IntentStore(ServiceIntentStoreError),
    Stop(ManagedServiceErrorKind),
    Observation(StopAndObserveErrorKind),
    Gpu(GpuProbeFailureKind),
    ArtifactCancelled,
    ArtifactDeadlineExceeded,
    Retirement(IntentRetirementErrorKind),
    RetirementUncertain,
}

/// Fixed redacted finalization failure. The stateful finalizer retains the
/// job and any stop receipt so callers can resume without issuing another
/// Stop or publishing native output early.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct OwnedJobFinalizeError {
    kind: OwnedJobFinalizeErrorKind,
}

impl OwnedJobFinalizeError {
    fn new(kind: OwnedJobFinalizeErrorKind) -> Self {
        Self { kind }
    }

    pub(crate) fn kind(&self) -> OwnedJobFinalizeErrorKind {
        self.kind
    }
}

impl fmt::Debug for OwnedJobFinalizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedJobFinalizeError")
            .field("kind", &self.kind)
            .field("job", &"<retained privately>")
            .field("receipt", &"<retained privately>")
            .finish()
    }
}

impl fmt::Display for OwnedJobFinalizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("owned native job finalization did not complete")
    }
}

impl std::error::Error for OwnedJobFinalizeError {}

#[cfg(test)]
type TestGpuIdleObserver = Box<dyn FnMut(&GpuProbeConfig) -> Result<String, GpuProbeFailureKind>>;

/// Stateful Stop → cgroup/GPU proof → result revalidation → WAL retirement.
/// The owner remains mutably borrowed and the job remains owned until a
/// `ReleasedNativeOutcome` is returned exactly once. The caller must keep its
/// reservation and exclusive ingress held for the finalizer's full lifetime:
/// the NVIDIA observation is point-in-time, not an admission fence.
pub(crate) struct OwnedJobFinalizer<'owner, 'host> {
    owner: &'owner mut ManagedRuntimeOwner<'host>,
    job: Option<OwnedNativeJob>,
    cgroup_probe: CgroupReleaseProbe,
    stop_attempted: bool,
    stop_uncertain: bool,
    stopped_receipt: Option<StoppedService>,
    stopped_observation: Option<StoppedServiceObservation>,
    pending_outcome: Option<ReleasedNativeOutcome>,
    retirement_uncertain: bool,
    released: bool,
    #[cfg(test)]
    gpu_idle_observer: Option<TestGpuIdleObserver>,
}

impl fmt::Debug for OwnedJobFinalizer<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedJobFinalizer")
            .field("job", &"<retained privately>")
            .field("stop_attempted", &self.stop_attempted)
            .field("stop_uncertain", &self.stop_uncertain)
            .field("has_stop_receipt", &self.stopped_receipt.is_some())
            .field(
                "has_cgroup_observation",
                &self.stopped_observation.is_some(),
            )
            .field("has_pending_outcome", &self.pending_outcome.is_some())
            .field("retirement_uncertain", &self.retirement_uncertain)
            .field("released", &self.released)
            .finish()
    }
}

/// Fixed failure phase and category for the stop-plus-observation operation.
/// The contained kinds are themselves fixed and redacted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopAndObserveErrorKind {
    Stop(ManagedServiceErrorKind),
    Observation(CgroupReleaseErrorKind),
}

/// The generation-bound cgroup observation returned together with the exact
/// stop receipt it inspected. Neither item proves GPU release or admission
/// safety.
#[derive(Eq, PartialEq)]
pub struct StoppedServiceObservation {
    receipt: StoppedService,
    observation: CgroupObservation,
}

impl StoppedServiceObservation {
    pub fn receipt(&self) -> &StoppedService {
        &self.receipt
    }

    pub fn observation(&self) -> &CgroupObservation {
        &self.observation
    }
}

impl fmt::Debug for StoppedServiceObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoppedServiceObservation")
            .field("receipt", &"<redacted>")
            .field("observation", &self.observation)
            .finish()
    }
}

/// Stop errors carry no receipt. If stop succeeded but cgroup observation
/// failed, the receipt is retained so callers can re-observe without issuing
/// another stop command.
pub struct StopAndObserveError {
    kind: StopAndObserveErrorKind,
    receipt: Option<Box<StoppedService>>,
}

impl StopAndObserveError {
    pub fn kind(&self) -> StopAndObserveErrorKind {
        self.kind
    }

    pub fn receipt(&self) -> Option<&StoppedService> {
        self.receipt.as_deref()
    }

    pub fn into_receipt(self) -> Option<StoppedService> {
        self.receipt.map(|receipt| *receipt)
    }
}

impl fmt::Debug for StopAndObserveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StopAndObserveError")
            .field("kind", &self.kind)
            .field("receipt", &"<retained privately>")
            .finish()
    }
}

impl fmt::Display for StopAndObserveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            StopAndObserveErrorKind::Stop(_) => "service stop failed",
            StopAndObserveErrorKind::Observation(_) => "stopped service observation failed",
        })
    }
}

impl std::error::Error for StopAndObserveError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntentRetirementDisposition {
    /// No unlink was acknowledged; the retained write-ahead record remains.
    NotRetired,
    /// Unlink succeeded but parent-directory durability could not be confirmed.
    DurabilityUncertain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntentRetirementErrorKind {
    InvalidProof,
    Cancelled,
    DeadlineExceeded,
    Cgroup(CgroupReleaseErrorKind),
    Manager(ManagedServiceErrorKind),
    Store(ServiceIntentStoreError),
}

/// Fixed, redacted retirement failure. The original safe intent metadata is
/// retained even when unlink may have occurred, so a caller can reconcile the
/// exact record without retrying by unit name.
pub struct IntentRetirementError {
    kind: IntentRetirementErrorKind,
    disposition: IntentRetirementDisposition,
    intent: Box<ServiceIntentRecord>,
}

impl IntentRetirementError {
    pub fn kind(&self) -> IntentRetirementErrorKind {
        self.kind
    }

    pub fn disposition(&self) -> IntentRetirementDisposition {
        self.disposition
    }

    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }

    pub fn into_intent_record(self) -> ServiceIntentRecord {
        *self.intent
    }
}

impl fmt::Debug for IntentRetirementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntentRetirementError")
            .field("kind", &self.kind)
            .field("disposition", &self.disposition)
            .field("intent", &"<retained privately>")
            .finish()
    }
}

impl fmt::Display for IntentRetirementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.disposition {
            IntentRetirementDisposition::NotRetired => "stopped service intent was not retired",
            IntentRetirementDisposition::DurabilityUncertain => {
                "stopped service intent retirement durability is uncertain"
            }
        })
    }
}

impl std::error::Error for IntentRetirementError {}

fn observe_stopped_receipt(
    receipt: StoppedService,
    probe: &CgroupReleaseProbe,
    context: &OperationContext<'_>,
) -> Result<StoppedServiceObservation, StopAndObserveError> {
    let observation = match probe.observe(&receipt, context) {
        Ok(observation) => observation,
        Err(kind) => {
            return Err(StopAndObserveError {
                kind: StopAndObserveErrorKind::Observation(kind),
                receipt: Some(Box::new(receipt)),
            });
        }
    };
    if !observation.matches(&receipt) {
        return Err(StopAndObserveError {
            kind: StopAndObserveErrorKind::Observation(CgroupReleaseErrorKind::InvalidReceipt),
            receipt: Some(Box::new(receipt)),
        });
    }
    Ok(StoppedServiceObservation {
        receipt,
        observation,
    })
}

/// Validated inputs for one isolated engine launch. The manager still captures
/// coordinator identity immediately before preparation so this value cannot
/// carry stale process identity across a coordinator restart.
pub struct IsolatedEngineLaunch {
    runtime_identity: RuntimeIdentity,
    backend_recipe: LaunchRecipe,
    helper_executable: PathBuf,
    transport_directory: PathBuf,
    backend_port: BackendPort,
    stack_root: Option<StackRoot>,
    cuda_device: Option<CudaDevice>,
    stop_timeout: Duration,
}

impl IsolatedEngineLaunch {
    pub fn new(
        runtime_identity: RuntimeIdentity,
        backend_recipe: LaunchRecipe,
        helper_executable: impl AsRef<Path>,
        transport_directory: impl AsRef<Path>,
        backend_port: u16,
        stop_timeout: Duration,
    ) -> Result<Self, ManagedServiceErrorKind> {
        if stop_timeout.is_zero() || stop_timeout > Duration::from_secs(300) {
            return Err(ManagedServiceErrorKind::InvalidStopTimeout);
        }
        let backend_port = BackendPort::new(backend_port)
            .map_err(|_| ManagedServiceErrorKind::InvalidBackendPort)?;
        Ok(Self {
            runtime_identity,
            backend_recipe,
            helper_executable: helper_executable.as_ref().to_path_buf(),
            transport_directory: transport_directory.as_ref().to_path_buf(),
            backend_port,
            stack_root: None,
            cuda_device: None,
            stop_timeout,
        })
    }

    /// Explicitly provide the backend's `L3MS_ROOT`. The directory is
    /// canonicalized and validated here, then carried through the private
    /// helper argv; no ambient systemd variable is consulted.
    pub fn with_stack_root(
        mut self,
        path: impl AsRef<Path>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        self.stack_root =
            Some(StackRoot::new(path).map_err(|_| ManagedServiceErrorKind::InvalidRecipe)?);
        Ok(self)
    }

    /// Explicitly bind this launch to one validated physical CUDA device.
    pub fn with_cuda_device(mut self, cuda_device: CudaDevice) -> Self {
        self.cuda_device = Some(cuda_device);
        self
    }
}

impl fmt::Debug for IsolatedEngineLaunch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("IsolatedEngineLaunch([redacted])")
    }
}

/// Borrowing the guard prevents normal callers from dropping host exclusion
/// while this owner is in use. The caller must use the same configured host
/// lock directory as all other coordinated entry points; advisory locks do not
/// exclude standalone clients or concurrent same-user manager changes.
/// Create one owner per dedicated store/controller under that guard. Mutable
/// operation methods serialize use of this instance; constructing other owners
/// or direct controller/store users does not bypass the caller precondition.
pub struct ManagedRuntimeOwner<'host> {
    controller: ManagedServiceController,
    store: ServiceIntentStore,
    _host: &'host CoordinatorLock,
}

impl<'host> ManagedRuntimeOwner<'host> {
    pub fn new(
        controller: ManagedServiceController,
        store: ServiceIntentStore,
        host: &'host CoordinatorLock,
    ) -> Self {
        Self {
            controller,
            store,
            _host: host,
        }
    }

    /// No manager command is attempted unless the write-ahead record has been
    /// acknowledged durable. Persistence errors can leave a partial record;
    /// retain it and quarantine rather than delete/retry. Context is checked
    /// before preparation and again after synchronous non-preemptible fsync.
    /// The caller must already have reconciled runtime ownership and obtained
    /// admission/resource authority; this method does not check GPU availability.
    pub fn start(
        &mut self,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        self.start_inner(None, recipe, stop_timeout, context)
    }

    /// Persist and dispatch a start intent bound to the selected catalog
    /// runtime. This shares the exact write-ahead path with generic starts.
    pub fn start_for_runtime(
        &mut self,
        runtime_identity: RuntimeIdentity,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        self.start_inner(Some(runtime_identity), recipe, stop_timeout, context)
    }

    /// Start the supervisor and backend together in one manager-created
    /// private network namespace and one systemd-owned cgroup. The write-ahead
    /// intent binds the runtime and coordinator process identity before the
    /// manager sees a start request. Manager READY proves transport setup
    /// only; native backend health remains a separate admission check.
    pub fn start_isolated_engine(
        &mut self,
        launch: IsolatedEngineLaunch,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        check_context(context).map_err(OwnerStartError::Preparation)?;
        let coordinator = CoordinatorIdentityDescriptor::capture_current().map_err(|_| {
            OwnerStartError::Preparation(ManagedServiceErrorKind::CoordinatorIdentityUnavailable)
        })?;
        let current_executable = std::env::current_exe()
            .map_err(|_| {
                OwnerStartError::Preparation(ManagedServiceErrorKind::ExecutableUnavailable)
            })?
            .canonicalize()
            .map_err(|_| {
                OwnerStartError::Preparation(ManagedServiceErrorKind::ExecutableUnavailable)
            })?;
        if launch.helper_executable.canonicalize().ok().as_deref()
            != Some(current_executable.as_path())
        {
            return Err(OwnerStartError::Preparation(
                ManagedServiceErrorKind::InvalidRecipe,
            ));
        }
        let mut backend = match launch.backend_recipe.verified_python() {
            Some(interpreter) => BackendProcessSpec::new_with_verified_python(
                interpreter,
                launch.backend_recipe.working_directory(),
                launch.backend_recipe.arguments().iter().cloned(),
            ),
            None => BackendProcessSpec::new(
                launch.backend_recipe.executable(),
                launch.backend_recipe.working_directory(),
                launch.backend_recipe.arguments().iter().cloned(),
            ),
        }
        .map_err(|_| OwnerStartError::Preparation(ManagedServiceErrorKind::InvalidRecipe))?;
        if let Some(stack_root) = launch.stack_root {
            backend = backend.with_stack_root(stack_root);
        }
        if let Some(cuda_device) = launch.cuda_device {
            backend = backend.with_cuda_device(cuda_device);
        }
        let helper_arguments = supervisor_arguments(
            &launch.transport_directory,
            launch.backend_port,
            coordinator,
            &backend,
        )
        .map_err(|_| OwnerStartError::Preparation(ManagedServiceErrorKind::InvalidRecipe))?;
        let supervisor_recipe = LaunchRecipe::new(
            &current_executable,
            launch.backend_recipe.working_directory(),
            helper_arguments,
        )
        .map_err(OwnerStartError::Preparation)?;
        let prepared = self
            .controller
            .prepare_isolated_engine_supervisor(
                launch.runtime_identity,
                supervisor_recipe,
                launch.stop_timeout,
                coordinator,
            )
            .map_err(OwnerStartError::Preparation)?;
        self.persist_and_dispatch(prepared, context)
    }

    /// Start a one-shot local job wrapper with durable job/runtime ownership.
    /// The caller must first publish its validated private request packet and
    /// retain admission/reservations. Native result readiness is not release;
    /// this uses the ordinary exact-service Stop/cgroup observation path.
    pub fn start_owned_local_job(
        &mut self,
        binding: OwnedLocalJobBinding,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        check_context(context).map_err(OwnerStartError::Preparation)?;
        let executable = std::env::current_exe()
            .and_then(|path| path.canonicalize())
            .map_err(|_| {
                OwnerStartError::Preparation(ManagedServiceErrorKind::ExecutableUnavailable)
            })?;
        let recipe = LaunchRecipe::new(
            &executable,
            binding.job_directory(),
            super::owned_job_helper::helper_arguments(&binding),
        )
        .map_err(OwnerStartError::Preparation)?;
        let prepared = self
            .controller
            .prepare_owned_local_job(binding, recipe, stop_timeout)
            .map_err(OwnerStartError::Preparation)?;
        self.persist_and_dispatch(prepared, context)
    }

    /// Publish a fresh validated request and start its one-shot owned wrapper.
    /// Caller admission and resource checks remain prerequisites. Failed packet
    /// or WAL publication is retained, never overwritten or replayed.
    pub fn start_owned_local_job_request(
        &mut self,
        job_directory: &Path,
        spec: OwnedJobRequestSpec<'_>,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedNativeJob, OwnerStartError> {
        check_context(context).map_err(OwnerStartError::Preparation)?;
        let store = OwnedJobStore::open(job_directory).map_err(|_| {
            OwnerStartError::Preparation(ManagedServiceErrorKind::InvalidOwnedJobBinding)
        })?;
        let job = store.write_request(spec).map_err(|_| {
            OwnerStartError::Preparation(ManagedServiceErrorKind::InvalidOwnedJobBinding)
        })?;
        let coordinator = CoordinatorIdentityDescriptor::capture_current().map_err(|_| {
            OwnerStartError::Preparation(ManagedServiceErrorKind::CoordinatorIdentityUnavailable)
        })?;
        let binding = OwnedLocalJobBinding::capture(
            job.route().runtime.clone(),
            job.job_id(),
            job.request_id(),
            job_directory,
            coordinator,
        )
        .map_err(OwnerStartError::Preparation)?;
        if binding.directory_device() != store.directory_device()
            || binding.directory_inode() != store.directory_inode()
        {
            return Err(OwnerStartError::Preparation(
                ManagedServiceErrorKind::InvalidOwnedJobBinding,
            ));
        }
        let service = self.start_owned_local_job(binding, stop_timeout, context)?;
        Ok(OwnedNativeJob {
            service,
            store,
            job,
        })
    }

    /// Begin one stateful release protocol for a running native job. The
    /// finalizer owns the job association and retains its stop receipt across
    /// read-only retries. It never trusts result readiness as cleanup proof.
    pub(crate) fn begin_owned_job_finalization<'owner>(
        &'owner mut self,
        job: OwnedNativeJob,
        cgroup_probe: CgroupReleaseProbe,
    ) -> OwnedJobFinalizer<'owner, 'host> {
        OwnedJobFinalizer {
            owner: self,
            job: Some(job),
            cgroup_probe,
            stop_attempted: false,
            stop_uncertain: false,
            stopped_receipt: None,
            stopped_observation: None,
            pending_outcome: None,
            retirement_uncertain: false,
            released: false,
            #[cfg(test)]
            gpu_idle_observer: None,
        }
    }

    fn start_inner(
        &mut self,
        runtime_identity: Option<RuntimeIdentity>,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        check_context(context).map_err(OwnerStartError::Preparation)?;
        let prepared = match runtime_identity {
            Some(identity) => {
                self.controller
                    .prepare_start_for_runtime(identity, recipe, stop_timeout)
            }
            None => self.controller.prepare_start(recipe, stop_timeout),
        }
        .map_err(OwnerStartError::Preparation)?;
        self.persist_and_dispatch(prepared, context)
    }

    fn persist_and_dispatch(
        &mut self,
        prepared: super::managed_service::PreparedServiceStart,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        let intent = prepared.intent_record().clone();
        self.store
            .record(&intent)
            .map_err(|kind| OwnerStartError::Persistence {
                intent: Box::new(intent.clone()),
                kind,
            })?;
        check_context(context).map_err(|kind| OwnerStartError::NotDispatched {
            intent: Box::new(intent),
            kind,
        })?;
        self.controller
            .dispatch(prepared, context)
            .map_err(OwnerStartError::Dispatch)
    }

    /// Observe only a handle whose exact durable intent is still retained.
    /// The controller then checks the active manager generation, cgroup,
    /// MainPID, and (for an isolated supervisor) network namespace. A
    /// `StopOnlyService` intentionally has no corresponding health/state
    /// operation.
    pub fn state(
        &self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<ManagedServiceState, ManagedServiceErrorKind> {
        check_context(context)?;
        let records = self
            .store
            .list()
            .map_err(|_| ManagedServiceErrorKind::UnitStateUnverifiable)?;
        if !records
            .iter()
            .any(|record| record == service.intent_record())
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        check_context(context)?;
        self.controller.state(service, context)
    }

    /// Return opaque equality evidence for a generation only after this owner
    /// confirms its exact durable intent and current manager state. This value
    /// is not an operation capability and the observation is point-in-time;
    /// callers must revalidate the owner immediately before native dispatch.
    /// The `OwnedService` argument deliberately excludes restored
    /// `StopOnlyService` handles from this path.
    pub(crate) fn running_generation_identity(
        &self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<OwnedServiceGenerationIdentity, ManagedServiceErrorKind> {
        match self.state(service, context)? {
            ManagedServiceState::Running => Ok(service.generation_identity()),
            ManagedServiceState::Stopped => Err(ManagedServiceErrorKind::ServiceNotActive),
            ManagedServiceState::Unknown => Err(ManagedServiceErrorKind::UnitStateUnverifiable),
        }
    }

    /// Observe every retained intent; never recreate a recipe or replay start.
    /// Any partial/failed observation is an error, not startup admission.
    pub fn reconcile(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<Vec<RetainedServiceObservation>, OwnerObservationError> {
        observation_context(context)?;
        let records = self
            .store
            .list()
            .map_err(|_| OwnerObservationError::StoreUnavailable)?;
        let mut observations = Vec::with_capacity(records.len());
        for intent in records {
            observation_context(context)?;
            let observation = self
                .controller
                .reconcile_restored_intent(&intent, context)
                .map_err(|kind| match kind {
                    ManagedServiceErrorKind::Cancelled => OwnerObservationError::Cancelled,
                    ManagedServiceErrorKind::DeadlineExceeded => {
                        OwnerObservationError::DeadlineExceeded
                    }
                    _ => OwnerObservationError::ManagerUnavailable,
                })?;
            observations.push(RetainedServiceObservation {
                intent,
                observation,
            });
        }
        observation_context(context)?;
        Ok(observations)
    }

    /// Retains the write-ahead record on every outcome. Success acknowledges
    /// the manager operation only; a separate release/admission protocol is
    /// needed before conflicting inference can be accepted.
    pub fn stop(
        &mut self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<StoppedService, ManagedServiceErrorKind> {
        check_context(context)?;
        let records = self
            .store
            .list()
            .map_err(|_| ManagedServiceErrorKind::UnitStateUnverifiable)?;
        if !records
            .iter()
            .any(|record| record == service.intent_record())
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        check_context(context)?;
        self.controller.stop(service, context)
    }

    /// Stop one exact isolated-supervisor generation recovered from the
    /// secure retained-intent store. The borrowed token survives every
    /// failure, and the durable WAL record is never retired here.
    pub fn stop_stop_only(
        &mut self,
        service: &StopOnlyService,
        context: &OperationContext<'_>,
    ) -> Result<StoppedService, ManagedServiceErrorKind> {
        check_context(context)?;
        let records = self
            .store
            .list()
            .map_err(|_| ManagedServiceErrorKind::UnitStateUnverifiable)?;
        if !records
            .iter()
            .any(|record| record == service.intent_record())
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        check_context(context)?;
        self.controller.stop_stop_only(service, context)
    }

    /// Stop the exact owned generation, then inspect only the cgroup bound to
    /// the returned stop receipt. The durable intent remains retained on all
    /// outcomes. If observation fails after stop acknowledgment, the error
    /// carries the receipt so the caller can retry only the read-only probe.
    /// This does not release GPU ownership or open runtime admission.
    pub fn stop_and_observe(
        &mut self,
        service: &OwnedService,
        probe: &CgroupReleaseProbe,
        context: &OperationContext<'_>,
    ) -> Result<StoppedServiceObservation, StopAndObserveError> {
        let receipt = self
            .stop(service, context)
            .map_err(|kind| StopAndObserveError {
                kind: StopAndObserveErrorKind::Stop(kind),
                receipt: None,
            })?;
        observe_stopped_receipt(receipt, probe, context)
    }

    /// Revalidate an existing stop receipt and retry only the read-only
    /// cgroup observation. This never issues another stop. The original WAL
    /// intent must still be retained and the manager must still show the same
    /// inactive generation (or the unit must remain absent).
    pub fn observe_stopped(
        &mut self,
        receipt: StoppedService,
        probe: &CgroupReleaseProbe,
        context: &OperationContext<'_>,
    ) -> Result<StoppedServiceObservation, StopAndObserveError> {
        let fail = |kind, receipt| StopAndObserveError {
            kind,
            receipt: Some(Box::new(receipt)),
        };
        if let Err(kind) = check_context(context) {
            return Err(fail(StopAndObserveErrorKind::Stop(kind), receipt));
        }
        let records = match self.store.list() {
            Ok(records) => records,
            Err(_) => {
                return Err(fail(
                    StopAndObserveErrorKind::Stop(ManagedServiceErrorKind::UnitStateUnverifiable),
                    receipt,
                ));
            }
        };
        if !records
            .iter()
            .any(|record| record == receipt.intent_record())
        {
            return Err(fail(
                StopAndObserveErrorKind::Stop(ManagedServiceErrorKind::OwnershipMismatch),
                receipt,
            ));
        }
        if let Err(kind) = check_context(context) {
            return Err(fail(StopAndObserveErrorKind::Stop(kind), receipt));
        }
        if let Err(kind) = self.controller.verify_stopped_receipt(&receipt, context) {
            return Err(fail(StopAndObserveErrorKind::Stop(kind), receipt));
        }
        if let Err(kind) = check_context(context) {
            return Err(fail(StopAndObserveErrorKind::Stop(kind), receipt));
        }
        observe_stopped_receipt(receipt, probe, context)
    }

    /// Stop a recovered supervisor generation and continue through the same
    /// generation-bound cgroup observation path used by ordinary ownership.
    pub fn stop_stop_only_and_observe(
        &mut self,
        service: &StopOnlyService,
        probe: &CgroupReleaseProbe,
        context: &OperationContext<'_>,
    ) -> Result<StoppedServiceObservation, StopAndObserveError> {
        let receipt =
            self.stop_stop_only(service, context)
                .map_err(|kind| StopAndObserveError {
                    kind: StopAndObserveErrorKind::Stop(kind),
                    receipt: None,
                })?;
        observe_stopped_receipt(receipt, probe, context)
    }

    /// Retire a retained intent only after rechecking its opaque stop receipt,
    /// a fresh cgroup observation, and the current manager generation. The
    /// exact record is then descriptor-relatively validated and unlinked, with
    /// its parent directory synced. The caller must retain the host guard and
    /// trust that no same-UID manager client replaces/restarts the unit and
    /// that the launched executable stays within its cgroup. Neither success
    /// nor the cgroup observation proves GPU release, admission closure, or a
    /// released resource claim.
    pub fn retire_stopped(
        &mut self,
        stopped: &StoppedServiceObservation,
        probe: &CgroupReleaseProbe,
        context: &OperationContext<'_>,
    ) -> Result<(), IntentRetirementError> {
        let receipt = stopped.receipt();
        let intent = receipt.intent_record().clone();
        let fail = |kind, disposition| IntentRetirementError {
            kind,
            disposition,
            intent: Box::new(intent.clone()),
        };

        retirement_context(context)
            .map_err(|kind| fail(kind, IntentRetirementDisposition::NotRetired))?;
        if !stopped.observation().matches(receipt)
            || !matches!(
                stopped.observation().kind(),
                CgroupObservationKind::Empty | CgroupObservationKind::Removed
            )
        {
            return Err(fail(
                IntentRetirementErrorKind::InvalidProof,
                IntentRetirementDisposition::NotRetired,
            ));
        }

        let fresh = probe.observe(receipt, context).map_err(|kind| {
            let kind = match kind {
                CgroupReleaseErrorKind::Cancelled => IntentRetirementErrorKind::Cancelled,
                CgroupReleaseErrorKind::DeadlineExceeded => {
                    IntentRetirementErrorKind::DeadlineExceeded
                }
                other => IntentRetirementErrorKind::Cgroup(other),
            };
            fail(kind, IntentRetirementDisposition::NotRetired)
        })?;
        if !fresh.matches(receipt)
            || !matches!(
                fresh.kind(),
                CgroupObservationKind::Empty | CgroupObservationKind::Removed
            )
        {
            return Err(fail(
                IntentRetirementErrorKind::InvalidProof,
                IntentRetirementDisposition::NotRetired,
            ));
        }
        retirement_context(context)
            .map_err(|kind| fail(kind, IntentRetirementDisposition::NotRetired))?;

        self.controller
            .verify_stopped_receipt(receipt, context)
            .map_err(|kind| {
                let kind = match kind {
                    ManagedServiceErrorKind::Cancelled => IntentRetirementErrorKind::Cancelled,
                    ManagedServiceErrorKind::DeadlineExceeded => {
                        IntentRetirementErrorKind::DeadlineExceeded
                    }
                    other => IntentRetirementErrorKind::Manager(other),
                };
                fail(kind, IntentRetirementDisposition::NotRetired)
            })?;
        retirement_context(context)
            .map_err(|kind| fail(kind, IntentRetirementDisposition::NotRetired))?;

        self.store
            .retire_record(&intent, context)
            .map_err(|failure| {
                let disposition = match failure.disposition() {
                    ServiceIntentRetireDisposition::NotRetired => {
                        IntentRetirementDisposition::NotRetired
                    }
                    ServiceIntentRetireDisposition::DurabilityUncertain => {
                        IntentRetirementDisposition::DurabilityUncertain
                    }
                };
                let kind = match failure.kind() {
                    ServiceIntentStoreError::Cancelled => IntentRetirementErrorKind::Cancelled,
                    ServiceIntentStoreError::DeadlineExceeded => {
                        IntentRetirementErrorKind::DeadlineExceeded
                    }
                    other => IntentRetirementErrorKind::Store(other),
                };
                fail(kind, disposition)
            })
    }
}

impl OwnedJobFinalizer<'_, '_> {
    /// Finish native-job cleanup and publish its typed result exactly once.
    /// Every failure leaves the job association in this finalizer. Once Stop
    /// succeeds, retries carry the same opaque receipt and perform only
    /// read-only cgroup/GPU/result checks plus proof-bound WAL retirement.
    pub(crate) fn finalize(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<ReleasedNativeOutcome, OwnedJobFinalizeError> {
        if self.released {
            return Err(OwnedJobFinalizeError::new(
                OwnedJobFinalizeErrorKind::AlreadyReleased,
            ));
        }
        if self.retirement_uncertain {
            return Err(OwnedJobFinalizeError::new(
                OwnedJobFinalizeErrorKind::RetirementUncertain,
            ));
        }
        check_context(context)
            .map_err(|kind| OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Context(kind)))?;

        if self.stopped_observation.is_none() && self.stopped_receipt.is_none() {
            if self.stop_uncertain || self.stop_attempted {
                return Err(OwnedJobFinalizeError::new(
                    OwnedJobFinalizeErrorKind::StopUncertain,
                ));
            }
            self.validate_job_association(context)?;
            check_context(context).map_err(|kind| {
                OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Context(kind))
            })?;

            // Mark before invoking the synchronous manager adapter. A local
            // error cannot establish that a Stop request never reached the
            // manager, so this finalizer never repeats the command.
            self.stop_attempted = true;
            let job = self.job.as_ref().expect("unfinished finalizer retains job");
            match self.owner.stop(&job.service, context) {
                Ok(receipt) => self.stopped_receipt = Some(receipt),
                Err(kind) => {
                    self.stop_uncertain = true;
                    return Err(OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Stop(
                        kind,
                    )));
                }
            }
        }

        // A retained observation from an earlier GPU/retirement failure is
        // not fresh authority for reading artifacts on this attempt. Convert
        // it back to the same exact Stop receipt and re-observe first.
        if let Some(previous) = self.stopped_observation.take() {
            self.stopped_receipt = Some(previous.receipt);
        }
        if self.stopped_observation.is_none() {
            let receipt = self
                .stopped_receipt
                .take()
                .expect("a successful Stop retains its generation receipt");
            match self
                .owner
                .observe_stopped(receipt, &self.cgroup_probe, context)
            {
                Ok(observation) => {
                    self.stopped_observation = Some(observation);
                }
                Err(error) => {
                    let kind = error.kind();
                    self.stopped_receipt = error.into_receipt();
                    if self.stopped_receipt.is_none() {
                        self.stop_uncertain = true;
                    }
                    return Err(OwnedJobFinalizeError::new(
                        OwnedJobFinalizeErrorKind::Observation(kind),
                    ));
                }
            }
        }

        // Re-observe the configured device and re-read/revalidate native data
        // on every retirement attempt. A prior WAL failure must not make a
        // point-in-time GPU observation or an old artifact receipt appear
        // fresh on a later retry.
        self.observe_configured_gpu(context)?;
        check_context(context)
            .map_err(|kind| OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Context(kind)))?;
        let outcome = self.read_released_native_outcome(context)?;
        self.pending_outcome = Some(outcome);

        let stopped = self
            .stopped_observation
            .as_ref()
            .expect("pending publication follows a generation-bound cgroup observation");
        if let Err(failure) = self
            .owner
            .retire_stopped(stopped, &self.cgroup_probe, context)
        {
            let kind = failure.kind();
            if failure.disposition() == IntentRetirementDisposition::DurabilityUncertain {
                self.retirement_uncertain = true;
            }
            return Err(OwnedJobFinalizeError::new(
                OwnedJobFinalizeErrorKind::Retirement(kind),
            ));
        }

        self.stopped_observation.take();
        self.job.take();
        self.released = true;
        Ok(self
            .pending_outcome
            .take()
            .expect("successful retirement has a validated native outcome"))
    }

    fn validate_job_association(
        &self,
        context: &OperationContext<'_>,
    ) -> Result<(), OwnedJobFinalizeError> {
        let job = self.job.as_ref().expect("unfinished finalizer retains job");
        let intent = job.service.intent_record();
        let Some(binding) = intent.owned_local_job_binding() else {
            return Err(OwnedJobFinalizeError::new(
                OwnedJobFinalizeErrorKind::InvalidAssociation,
            ));
        };
        if binding.job_id() != job.job.job_id()
            || binding.request_id() != job.job.request_id()
            || binding.runtime_identity() != &job.job.route().runtime
            || binding.directory_device() != job.store.directory_device()
            || binding.directory_inode() != job.store.directory_inode()
        {
            return Err(OwnedJobFinalizeError::new(
                OwnedJobFinalizeErrorKind::InvalidAssociation,
            ));
        }
        check_context(context)
            .map_err(|kind| OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Context(kind)))?;
        let records = self.owner.store.list().map_err(|kind| {
            OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::IntentStore(kind))
        })?;
        if !records.iter().any(|record| record == intent) {
            return Err(OwnedJobFinalizeError::new(
                OwnedJobFinalizeErrorKind::InvalidAssociation,
            ));
        }
        Ok(())
    }

    fn observe_configured_gpu(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<(), OwnedJobFinalizeError> {
        let job = self.job.as_ref().expect("unfinished finalizer retains job");
        let config = job.job.device().gpu_probe_config().map_err(|_| {
            OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::InvalidAssociation)
        })?;
        let Some(config) = config else {
            return Ok(());
        };

        let observed_uuid = {
            #[cfg(test)]
            if let Some(observer) = self.gpu_idle_observer.as_mut() {
                observer(&config).map_err(|kind| {
                    OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Gpu(kind))
                })?
            } else {
                NvidiaSmiGpuProbe::new(config.clone())
                    .observe_idle(context, job.job.bounds().recovery())
                    .map_err(|failure| {
                        OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Gpu(failure.kind()))
                    })?
                    .gpu_uuid()
                    .to_owned()
            }
            #[cfg(not(test))]
            {
                NvidiaSmiGpuProbe::new(config.clone())
                    .observe_idle(context, job.job.bounds().recovery())
                    .map_err(|failure| {
                        OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Gpu(failure.kind()))
                    })?
                    .gpu_uuid()
                    .to_owned()
            }
        };
        if observed_uuid != config.gpu_uuid() {
            return Err(OwnedJobFinalizeError::new(OwnedJobFinalizeErrorKind::Gpu(
                GpuProbeFailureKind::DeviceMismatch,
            )));
        }
        Ok(())
    }

    fn read_released_native_outcome(
        &self,
        context: &OperationContext<'_>,
    ) -> Result<ReleasedNativeOutcome, OwnedJobFinalizeError> {
        let job = self.job.as_ref().expect("unfinished finalizer retains job");
        let result = match job.store.read_result(&job.job) {
            Ok(result) => result,
            Err(_) => {
                return Ok(ReleasedNativeOutcome::Failed(
                    NativeJobFailure::InvalidOutput,
                ));
            }
        };
        match result {
            NativeJobResult::Decision(result) => Ok(ReleasedNativeOutcome::Decision(result)),
            NativeJobResult::Failed(failure) => Ok(ReleasedNativeOutcome::Failed(failure)),
            NativeJobResult::Breeze(receipt) => {
                let Some(binding) = job.service.intent_record().owned_local_job_binding() else {
                    return Ok(ReleasedNativeOutcome::Failed(
                        NativeJobFailure::InvalidOutput,
                    ));
                };
                let fresh = match super::media_execution::revalidate_breeze_artifacts(
                    binding.job_directory(),
                    receipt.mode(),
                    context,
                ) {
                    Ok(fresh) => fresh,
                    Err(failure) => {
                        use super::media_execution::MediaExecutionErrorKind as MediaError;
                        use super::process::ProcessFailureKind;
                        return match failure.kind() {
                            MediaError::Process(ProcessFailureKind::Cancelled) => {
                                Err(OwnedJobFinalizeError::new(
                                    OwnedJobFinalizeErrorKind::ArtifactCancelled,
                                ))
                            }
                            MediaError::Process(ProcessFailureKind::DeadlineExceeded) => {
                                Err(OwnedJobFinalizeError::new(
                                    OwnedJobFinalizeErrorKind::ArtifactDeadlineExceeded,
                                ))
                            }
                            _ => Ok(ReleasedNativeOutcome::Failed(
                                NativeJobFailure::InvalidOutput,
                            )),
                        };
                    }
                };
                if !breeze_receipt_matches(&receipt, &fresh) {
                    return Ok(ReleasedNativeOutcome::Failed(
                        NativeJobFailure::InvalidOutput,
                    ));
                }
                let still_current = matches!(
                    job.store.read_result(&job.job),
                    Ok(NativeJobResult::Breeze(current))
                        if breeze_receipts_equal(&receipt, &current)
                );
                if !still_current {
                    return Ok(ReleasedNativeOutcome::Failed(
                        NativeJobFailure::InvalidOutput,
                    ));
                }
                let artifacts = fresh
                    .artifacts()
                    .iter()
                    .map(|artifact| {
                        artifact
                            .relative_path()
                            .to_str()
                            .ok_or(())
                            .and_then(|path| {
                                MediaArtifact::new(ArtifactKind::LocalJob, path.to_owned())
                                    .map_err(|_| ())
                            })
                    })
                    .collect::<Result<Vec<_>, _>>();
                match artifacts {
                    Ok(artifacts) => Ok(ReleasedNativeOutcome::Breeze {
                        mode: fresh.mode(),
                        artifacts,
                    }),
                    Err(()) => Ok(ReleasedNativeOutcome::Failed(
                        NativeJobFailure::InvalidOutput,
                    )),
                }
            }
        }
    }
}

fn breeze_receipt_matches(
    receipt: &super::owned_job_store::BreezeArtifactReceipt,
    fresh: &super::media_execution::MediaExecutionResult,
) -> bool {
    receipt.mode() == fresh.mode()
        && receipt.artifacts().len() == fresh.artifacts().len()
        && receipt
            .artifacts()
            .iter()
            .zip(fresh.artifacts())
            .all(|(candidate, artifact)| {
                let fresh_kind = match artifact.kind() {
                    super::media_execution::MediaArtifactKind::FinalWave => {
                        BreezeArtifactKind::FinalWave
                    }
                    super::media_execution::MediaArtifactKind::StreamingChunk => {
                        BreezeArtifactKind::StreamingChunk
                    }
                    super::media_execution::MediaArtifactKind::RawRgb24Video
                    | super::media_execution::MediaArtifactKind::Video
                    | super::media_execution::MediaArtifactKind::Image => return false,
                };
                candidate.kind() == fresh_kind
                    && candidate.relative_path() == artifact.relative_path().to_str().unwrap_or("")
            })
}

fn breeze_receipts_equal(
    left: &super::owned_job_store::BreezeArtifactReceipt,
    right: &super::owned_job_store::BreezeArtifactReceipt,
) -> bool {
    left.mode() == right.mode()
        && left.artifacts().len() == right.artifacts().len()
        && left
            .artifacts()
            .iter()
            .zip(right.artifacts())
            .all(|(left, right)| {
                left.kind() == right.kind() && left.relative_path() == right.relative_path()
            })
}

fn check_context(context: &OperationContext<'_>) -> Result<(), ManagedServiceErrorKind> {
    if context.is_cancelled() {
        Err(ManagedServiceErrorKind::Cancelled)
    } else if context.remaining().is_zero() {
        Err(ManagedServiceErrorKind::DeadlineExceeded)
    } else {
        Ok(())
    }
}
fn observation_context(context: &OperationContext<'_>) -> Result<(), OwnerObservationError> {
    check_context(context).map_err(|kind| match kind {
        ManagedServiceErrorKind::Cancelled => OwnerObservationError::Cancelled,
        _ => OwnerObservationError::DeadlineExceeded,
    })
}
fn retirement_context(context: &OperationContext<'_>) -> Result<(), IntentRetirementErrorKind> {
    if context.is_cancelled() {
        Err(IntentRetirementErrorKind::Cancelled)
    } else if context.remaining().is_zero() {
        Err(IntentRetirementErrorKind::DeadlineExceeded)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{Catalog, Lane, ProfileKey, RuntimeIdentity};
    use super::super::cgroup_release::CgroupObservationKind;
    use super::super::decision::NativeDecisionResult;
    use super::super::requests::{InferenceRequest, InferenceResult, MediaInput, SpeechInput};
    use super::super::routes::resolve_profile_route;
    use super::super::runtime::{Cancellation, NeverCancel, RecoveryPolicy};
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::str::FromStr;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    struct Fixture {
        root: tempfile::TempDir,
        intents: tempfile::TempDir,
        locks: tempfile::TempDir,
        cgroups: tempfile::TempDir,
        repo: PathBuf,
        job_directory: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let mut result = Self {
                root: tempfile::tempdir().unwrap(),
                intents: tempfile::tempdir().unwrap(),
                locks: tempfile::tempdir().unwrap(),
                cgroups: tempfile::tempdir().unwrap(),
                repo: PathBuf::new(),
                job_directory: PathBuf::new(),
            };
            result.repo = result.root.path().join("repo");
            result.job_directory = result.root.path().join("owned-job");
            for directory in [
                result.root.path(),
                result.intents.path(),
                result.locks.path(),
                result.cgroups.path(),
            ] {
                fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
            }
            result.create_native_catalog();
            let root_json = serde_json::to_string(result.root.path().to_str().unwrap()).unwrap();
            let intents_json =
                serde_json::to_string(result.intents.path().to_str().unwrap()).unwrap();
            let script = format!(
                r#"#!/usr/bin/python3
import json, os, pathlib, sys
root = pathlib.Path({root_json})
intents = pathlib.Path({intents_json})
state = root / 'manager-state.json'
args = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == 'fake-run':
    unit = next(a.split('=', 1)[1] for a in args if a.startswith('--unit='))
    description = next(a.split('=', 1)[1] for a in args if a.startswith('--description='))
    record = json.loads((intents / (unit + '.json')).read_text())
    assert record['unit_name'] == unit
    if (root / 'require-owned-job').exists():
        assert record['schema'] == 4
        assert record['launch_role'] == 'owned_local_job'
        assert record['runtime_name'] in ('nagi', 'audio.cpp')
        assert record['owned_local_job']['job_id'] == 17
        assert record['owned_local_job']['request_id'] == 'fixture-owned-job'
        assert '--l3ms-owned-local-job-v1' in args
        assert '--service-type=exec' in args
        assert '--property=PrivateNetwork=yes' not in args
        assert not any('OpenFile=' in a for a in args)
        assert record['coordinator_identity']['pid'] == os.getppid()
    if (root / 'require-supervisor').exists():
        identity = record['coordinator_identity']
        assert record['schema'] == 3
        assert record['launch_role'] == 'isolated_engine_supervisor'
        assert record['runtime_lane'] == 'media'
        assert record['runtime_name'] == 'comfyui'
        assert identity['pid'] == os.getppid()
        assert identity['uid'] == os.geteuid()
        assert '--service-type=notify' in args
        assert '--property=PrivateNetwork=yes' in args
        assert any(a.startswith('--property=OpenFile=/proc/' + str(identity['pid'])
                   + '/ns/net:l3ms-coordinator-netns:read-only') for a in args)
        assert not any('JoinsNamespaceOf=' in a for a in args)
    if (root / 'require-runtime').exists():
        assert record['schema'] == 2
        assert record['runtime_lane'] == 'media'
        assert record['runtime_name'] == 'comfyui'
    with (root / 'starts').open('a') as log:
        log.write('start\n')
    state.write_text(json.dumps(dict(unit=unit, description=description,
                                     active='active', invocation='a'*32)))
    if (root / 'require-supervisor').exists() or (root / 'isolated-snapshot').exists():
        data = json.loads(state.read_text())
        data.update(type='notify', private_network='yes', joins_namespace_of='')
        if (root / 'require-supervisor').exists():
            data['main_pid'] = os.getppid()
        state.write_text(json.dumps(data))
    sys.exit(9 if (root / 'fail-launcher').exists() else 0)
if 'list-units' in args or 'list-unit-files' in args:
    sys.exit(0)
if 'stop' in args:
    with (root / 'stops').open('a') as log:
        log.write('stop\n')
    if (root / 'fail-stop').exists():
        sys.exit(23)
    state.unlink()
    sys.exit(0)
if 'show' in args:
    if (root / 'fail-show').exists():
        sys.exit(23)
    if not state.exists():
        print('LoadState=not-found')
        sys.exit(0)
    data = json.loads(state.read_text())
    if args[-1] != data['unit']:
        print('LoadState=not-found')
        sys.exit(0)
    active = data.get('active', 'active')
    main_pid = data.get('main_pid', '1234' if active == 'active' else '0')
    print('LoadState=loaded\nActiveState=' + active + '\nMainPID=' + main_pid + '\nTransient=yes')
    print('InvocationID=' + data.get('invocation', 'a' * 32))
    print('ControlGroup=' + data.get('cgroup', '/user.slice/' + data['unit']))
    print('FragmentPath=/run/user/1000/systemd/transient/' + data['unit'])
    print('Description=' + data['description'])
    print('Restart=no\nKillMode=control-group')
    print('Type=' + data.get('type', 'exec'))
    print('PrivateNetwork=' + data.get('private_network', 'no'))
    print('JoinsNamespaceOf=' + data.get('joins_namespace_of', ''))
    sys.exit(0)
sys.exit(23)
"#
            );
            for name in ["fake-run", "fake-ctl"] {
                let _guard = super::super::process::executable_fixture_guard();
                let path = result.root.path().join(name);
                fs::write(&path, &script).unwrap();
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
            }
            result
        }

        fn create_native_catalog(&self) {
            let media_root = self.repo.join("toolkits/l2m2");
            let decisions_root = self.repo.join("toolkits/l3ds");
            fs::create_dir_all(media_root.join("media-models")).unwrap();
            fs::create_dir_all(decisions_root.join("decision-models")).unwrap();
            fs::create_dir_all(decisions_root.join(".venv/bin")).unwrap();
            fs::write(
                self.repo.join("llama-swap.yaml"),
                "models:\n  \"fixture-language\":\n    cmd: |\n      llama-server -m /not-loaded.gguf\n",
            )
            .unwrap();
            fs::write(
                media_root.join("media-runtimes.json"),
                serde_json::to_vec(&serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "breeze-tts-2",
                        "name": "Fixture Breeze",
                        "runtime": "audio.cpp",
                        "device": "cuda",
                        "tasks": ["tts", "clon", "design"],
                        "script": "media-models/generate-breeze-tts-2.sh"
                    }]
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                media_root.join("media-models/generate-breeze-tts-2.sh"),
                "#!/bin/bash\nexit 0\n",
            )
            .unwrap();
            fs::write(
                decisions_root.join("decision-runtimes.json"),
                serde_json::to_vec(&serde_json::json!({
                    "version": "test",
                    "profiles": [{
                        "id": "nagi-fixture",
                        "name": "Fixture Nagi",
                        "engine": "nagi",
                        "tier": "test",
                        "type": "local",
                        "model": "fixture-model",
                        "device": "cpu",
                        "entrypoint": "decision-models/runner.py"
                    }]
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                decisions_root.join("decision-models/runner.py"),
                "# fixture only\n",
            )
            .unwrap();
            let python = decisions_root.join(".venv/bin/python");
            fs::write(&python, "#!/bin/sh\nexit 0\n").unwrap();
            fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
            fs::create_dir(&self.job_directory).unwrap();
            fs::set_permissions(&self.job_directory, fs::Permissions::from_mode(0o700)).unwrap();
        }

        fn controller(&self) -> ManagedServiceController {
            ManagedServiceController::with_test_executables(
                "fixture",
                self.root.path().join("fake-run"),
                self.root.path().join("fake-ctl"),
            )
            .unwrap()
        }
        fn owner<'a>(&self, guard: &'a CoordinatorLock) -> ManagedRuntimeOwner<'a> {
            ManagedRuntimeOwner::new(
                self.controller(),
                ServiceIntentStore::open(self.intents.path()).unwrap(),
                guard,
            )
        }
        fn recipe(&self) -> LaunchRecipe {
            LaunchRecipe::new(Path::new("/usr/bin/true"), self.root.path(), Vec::new()).unwrap()
        }
        fn context(&self) -> OperationContext<'static> {
            OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel)
        }

        fn start_decision_job(&self, owner: &mut ManagedRuntimeOwner<'_>) -> OwnedNativeJob {
            self.start_native_job(owner, self.decision_request(), DeviceSelection::Cpu)
        }

        fn start_breeze_job(&self, owner: &mut ManagedRuntimeOwner<'_>) -> OwnedNativeJob {
            self.start_native_job(
                owner,
                self.breeze_request(),
                DeviceSelection::nvidia("GPU-01234567-89ab-cdef-0123-456789abcdef", 0, 256)
                    .unwrap(),
            )
        }

        fn start_native_job(
            &self,
            owner: &mut ManagedRuntimeOwner<'_>,
            request: InferenceRequest,
            device: DeviceSelection,
        ) -> OwnedNativeJob {
            fs::write(self.root.path().join("require-owned-job"), "1").unwrap();
            let catalog = Catalog::load(&self.repo).unwrap();
            let route = resolve_profile_route(&catalog, request.profile()).unwrap();
            owner
                .start_owned_local_job_request(
                    &self.job_directory,
                    OwnedJobRequestSpec {
                        job_id: 17,
                        request_id: "fixture-owned-job",
                        repository_root: &self.repo,
                        request: &request,
                        accepted_route: &route,
                        device,
                        bounds: JobExecutionBounds::new(
                            Duration::from_secs(30),
                            RecoveryPolicy::new(Duration::from_secs(2)).unwrap(),
                        )
                        .unwrap(),
                    },
                    Duration::from_secs(1),
                    &self.context(),
                )
                .unwrap()
        }

        fn decision_request(&self) -> InferenceRequest {
            let mut questions = serde_json::Map::new();
            questions.insert(
                "choice".to_owned(),
                serde_json::json!({"type":"choice", "options":["yes", "no"]}),
            );
            InferenceRequest::Decision {
                profile: ProfileKey::from_str("decisions:nagi-fixture").unwrap(),
                state: serde_json::json!({"state":"fixture"}),
                questions,
            }
        }

        fn breeze_request(&self) -> InferenceRequest {
            InferenceRequest::Media {
                profile: ProfileKey::from_str("media:breeze-tts-2").unwrap(),
                input: MediaInput::Speech(SpeechInput {
                    text: "fixture speech".to_owned(),
                    mode: SpeechMode::Standard,
                    instruction: None,
                    voice_reference: None,
                    reference_text: None,
                }),
            }
        }

        fn cgroup_probe(&self, service: &OwnedService, events: Option<&str>) -> CgroupReleaseProbe {
            self.cgroup_probe_for_unit(service.unit_name(), events)
        }

        fn cgroup_probe_for_unit(
            &self,
            unit_name: &str,
            events: Option<&str>,
        ) -> CgroupReleaseProbe {
            let leaf = self.cgroups.path().join("user.slice").join(unit_name);
            fs::create_dir_all(&leaf).unwrap();
            if let Some(events) = events {
                fs::write(leaf.join("cgroup.events"), events).unwrap();
            }
            CgroupReleaseProbe::with_test_root(self.cgroups.path()).unwrap()
        }

        fn cgroup_events_path(&self, service: &OwnedService) -> PathBuf {
            self.cgroups
                .path()
                .join("user.slice")
                .join(service.unit_name())
                .join("cgroup.events")
        }

        fn claim_and_write_result(&self, job: &OwnedNativeJob, result: &NativeJobResult) {
            job.store.claim_execution(&job.job).unwrap();
            job.store.write_result(&job.job, result).unwrap();
        }

        fn empty_cgroup_probe(&self, job: &OwnedNativeJob) -> CgroupReleaseProbe {
            self.cgroup_probe(job.service(), Some("populated 0\nfrozen 0\n"))
        }

        fn minimal_wave() -> Vec<u8> {
            let mut bytes = Vec::with_capacity(46);
            bytes.extend_from_slice(b"RIFF");
            bytes.extend_from_slice(&38u32.to_le_bytes());
            bytes.extend_from_slice(b"WAVEfmt ");
            bytes.extend_from_slice(&16u32.to_le_bytes());
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&8_000u32.to_le_bytes());
            bytes.extend_from_slice(&16_000u32.to_le_bytes());
            bytes.extend_from_slice(&2u16.to_le_bytes());
            bytes.extend_from_slice(&16u16.to_le_bytes());
            bytes.extend_from_slice(b"data");
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&[0, 0]);
            bytes
        }

        fn install_stale_supervisor_intent(
            &self,
            owner: &mut ManagedRuntimeOwner<'_>,
        ) -> ServiceIntentRecord {
            fs::write(self.root.path().join("isolated-snapshot"), "").unwrap();
            let current = CoordinatorIdentityDescriptor::capture_current().unwrap();
            let stale_pid = if current.pid() == 2 { 3 } else { 2 };
            let stale = CoordinatorIdentityDescriptor::from_parts(
                stale_pid,
                current.uid(),
                current.start_time_ticks(),
                current.network_namespace_device(),
                current.network_namespace_inode(),
            )
            .unwrap();
            let prepared = owner
                .controller
                .prepare_isolated_engine_supervisor(
                    RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
                    self.recipe(),
                    Duration::from_secs(1),
                    stale,
                )
                .unwrap();
            let intent = prepared.intent_record().clone();
            owner.store.record(&intent).unwrap();
            let failure = owner
                .controller
                .dispatch(prepared, &self.context())
                .unwrap_err();
            assert_eq!(failure.disposition(), StartDisposition::OwnershipUncertain);
            intent
        }

        fn cgroup_leaf(&self, service: &OwnedService) -> std::path::PathBuf {
            self.cgroups
                .path()
                .join("user.slice")
                .join(service.unit_name())
        }

        fn start(&self, owner: &mut ManagedRuntimeOwner<'_>) -> OwnedService {
            owner
                .start(self.recipe(), Duration::from_secs(1), &self.context())
                .unwrap()
        }

        fn stop_with_observation(
            &self,
            owner: &mut ManagedRuntimeOwner<'_>,
            service: &OwnedService,
            events: Option<&str>,
        ) -> (CgroupReleaseProbe, StoppedServiceObservation) {
            let probe = self.cgroup_probe(service, events);
            let stopped = owner
                .stop_and_observe(service, &probe, &self.context())
                .unwrap();
            (probe, stopped)
        }
    }

    #[test]
    fn owned_job_wal_precedes_manager_dispatch_and_restores_stop_only() {
        let fixture = Fixture::new();
        fs::write(fixture.root.path().join("require-owned-job"), "1").unwrap();
        let directory = fixture.root.path().join("private-job");
        fs::create_dir(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let binding = OwnedLocalJobBinding::capture(
            RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap(),
            17,
            "fixture-owned-job",
            &directory,
            CoordinatorIdentityDescriptor::capture_current().unwrap(),
        )
        .unwrap();
        let service = owner
            .start_owned_local_job(binding, Duration::from_secs(1), &fixture.context())
            .unwrap();
        let restored = owner.reconcile(&fixture.context()).unwrap();
        assert_eq!(restored.len(), 1);
        assert!(matches!(
            restored[0].observation,
            ReconcileResult::StopOnly(_)
        ));
        let (probe, stopped) =
            fixture.stop_with_observation(&mut owner, &service, Some("populated 0\n"));
        owner
            .retire_stopped(&stopped, &probe, &fixture.context())
            .unwrap();
        assert!(owner.reconcile(&fixture.context()).unwrap().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn durable_record_precedes_dispatch_and_restart_observes_without_replay() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        assert_eq!(
            owner.state(&handle, &fixture.context()),
            Ok(ManagedServiceState::Running)
        );
        assert_eq!(
            ServiceIntentStore::open(fixture.intents.path())
                .unwrap()
                .list()
                .unwrap()
                .len(),
            1
        );
        drop(owner);
        let mut owner = fixture.owner(&guard);
        let observations = owner.reconcile(&fixture.context()).unwrap();
        assert_eq!(observations.len(), 1);
        assert!(matches!(
            &observations[0].observation,
            ReconcileResult::Owned(_)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
        owner.stop(&handle, &fixture.context()).unwrap();
        assert_eq!(
            owner.state(&handle, &fixture.context()),
            Ok(ManagedServiceState::Unknown)
        );
        let observations = owner.reconcile(&fixture.context()).unwrap();
        assert!(matches!(
            &observations[0].observation,
            ReconcileResult::AbsentNow
        ));
        assert_eq!(
            ServiceIntentStore::open(fixture.intents.path())
                .unwrap()
                .list()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn runtime_bound_start_is_persisted_before_dispatch_and_restored_without_replay() {
        let fixture = Fixture::new();
        fs::write(fixture.root.path().join("require-runtime"), "").unwrap();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "comfyui").unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start_for_runtime(
                runtime.clone(),
                fixture.recipe(),
                Duration::from_secs(1),
                &fixture.context(),
            )
            .unwrap();

        let stored = ServiceIntentStore::open(fixture.intents.path())
            .unwrap()
            .list()
            .unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].runtime_identity(), Some(&runtime));
        assert_eq!(handle.intent_record().runtime_identity(), Some(&runtime));

        drop(owner);
        let mut restarted_owner = fixture.owner(&guard);
        let observations = restarted_owner.reconcile(&fixture.context()).unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].intent.runtime_identity(), Some(&runtime));
        assert!(matches!(
            &observations[0].observation,
            ReconcileResult::Owned(_)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
    }

    #[test]
    fn isolated_engine_stack_root_is_explicit_validated_and_redacted() {
        let fixture = Fixture::new();
        let stack_root = fixture.root.path().join("stack-root");
        fs::create_dir(&stack_root).unwrap();
        let launch = IsolatedEngineLaunch::new(
            RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
            fixture.recipe(),
            std::env::current_exe().unwrap(),
            fixture.root.path(),
            8188,
            Duration::from_secs(1),
        )
        .unwrap()
        .with_stack_root(&stack_root)
        .unwrap();
        let debug = format!("{launch:?}");
        assert_eq!(debug, "IsolatedEngineLaunch([redacted])");
        assert!(!debug.contains(stack_root.to_str().unwrap()));

        let invalid = IsolatedEngineLaunch::new(
            RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
            fixture.recipe(),
            std::env::current_exe().unwrap(),
            fixture.root.path(),
            8188,
            Duration::from_secs(1),
        )
        .unwrap()
        .with_stack_root("relative");
        assert_eq!(
            invalid.map(|_| ()),
            Err(ManagedServiceErrorKind::InvalidRecipe)
        );

        let file = fixture.root.path().join("not-a-directory");
        fs::write(&file, b"fixture").unwrap();
        let invalid = IsolatedEngineLaunch::new(
            RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
            fixture.recipe(),
            std::env::current_exe().unwrap(),
            fixture.root.path(),
            8188,
            Duration::from_secs(1),
        )
        .unwrap()
        .with_stack_root(file);
        assert_eq!(
            invalid.map(|_| ()),
            Err(ManagedServiceErrorKind::InvalidRecipe)
        );
    }

    #[test]
    fn isolated_engine_launch_accepts_only_typed_cuda_selection_and_redacts_it() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let launch = IsolatedEngineLaunch::new(
            RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
            fixture.recipe(),
            std::env::current_exe().unwrap(),
            fixture.root.path(),
            8188,
            Duration::from_secs(1),
        )
        .unwrap()
        .with_cuda_device(CudaDevice::new(UUID).unwrap());
        assert_eq!(
            launch.cuda_device.as_ref().map(CudaDevice::gpu_uuid),
            Some(UUID)
        );
        assert_eq!(format!("{launch:?}"), "IsolatedEngineLaunch([redacted])");
    }

    #[test]
    fn isolated_engine_wal_persists_one_supervisor_unit_before_manager_dispatch() {
        let fixture = Fixture::new();
        fs::write(fixture.root.path().join("require-supervisor"), "").unwrap();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let launch = IsolatedEngineLaunch::new(
            RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
            fixture.recipe(),
            std::env::current_exe().unwrap(),
            fixture.root.path(),
            8188,
            Duration::from_secs(1),
        )
        .unwrap();
        let error = owner
            .start_isolated_engine(launch, &fixture.context())
            .unwrap_err();

        let error_category = match &error {
            OwnerStartError::Preparation(kind) => format!("preparation:{kind:?}"),
            OwnerStartError::Persistence { kind, .. } => format!("persistence:{kind:?}"),
            OwnerStartError::NotDispatched { kind, .. } => format!("not-dispatched:{kind:?}"),
            OwnerStartError::Dispatch(failure) => format!("dispatch:{:?}", failure.kind()),
        };

        // The fake manager reports the expected generated supervisor shape,
        // but the MainPID shares the coordinator's host namespace. The role
        // must fail closed after dispatch while preserving the exact schema-3
        // WAL record for reconciliation.
        assert_eq!(
            error.disposition(),
            StartDisposition::OwnershipUncertain,
            "unexpected error category: {error_category}"
        );
        let records = ServiceIntentStore::open(fixture.intents.path())
            .unwrap()
            .list()
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].runtime_identity(),
            Some(&RuntimeIdentity::new(Lane::Media, "comfyui").unwrap())
        );
        let stored: serde_json::Value = serde_json::from_slice(
            &fs::read(
                fixture
                    .intents
                    .path()
                    .join(format!("{}.json", records[0].unit_name())),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(stored["schema"], 3);
        assert_eq!(stored["launch_role"], "isolated_engine_supervisor");
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
    }

    #[test]
    fn running_generation_identity_requires_exact_wal_and_manager_generation() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let service = fixture.start(&mut owner);

        let identity = owner
            .running_generation_identity(&service, &fixture.context())
            .unwrap();
        assert_eq!(
            identity,
            owner
                .running_generation_identity(&service, &fixture.context())
                .unwrap()
        );
        assert_eq!(
            format!("{identity:?}"),
            "OwnedServiceGenerationIdentity([redacted])"
        );

        let intent_path = fixture
            .intents
            .path()
            .join(format!("{}.json", service.unit_name()));
        fs::remove_file(&intent_path).unwrap();
        assert_eq!(
            owner.running_generation_identity(&service, &fixture.context()),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        );

        owner.store.record(service.intent_record()).unwrap();
        let state_path = fixture.root.path().join("manager-state.json");
        let mut state: serde_json::Value =
            serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
        state["invocation"] = serde_json::Value::String("b".repeat(32));
        fs::write(&state_path, serde_json::to_vec(&state).unwrap()).unwrap();
        assert_eq!(
            owner.running_generation_identity(&service, &fixture.context()),
            Err(ManagedServiceErrorKind::UnitStateUnverifiable)
        );
    }

    #[test]
    fn restored_supervisor_from_different_coordinator_can_stop_only_exact_generation() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let intent = fixture.install_stale_supervisor_intent(&mut owner);

        assert!(matches!(
            owner.controller.reconcile(&intent, &fixture.context()),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        ));

        let observations = owner.reconcile(&fixture.context()).unwrap();
        let stop_only = match &observations[0].observation {
            ReconcileResult::StopOnly(stop_only) => stop_only,
            other => panic!("expected stop-only recovery, got {other:?}"),
        };
        assert_eq!(stop_only.intent_record(), &intent);

        let probe =
            fixture.cgroup_probe_for_unit(intent.unit_name(), Some("populated 0\nfrozen 0\n"));
        let stopped = owner
            .stop_stop_only_and_observe(stop_only, &probe, &fixture.context())
            .unwrap();
        assert_eq!(stopped.receipt().intent_record(), &intent);
        owner
            .retire_stopped(&stopped, &probe, &fixture.context())
            .unwrap();

        assert!(owner.store.list().unwrap().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn manager_observation_failure_never_mints_stop_only_token() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        fixture.install_stale_supervisor_intent(&mut owner);

        let state_path = fixture.root.path().join("manager-state.json");
        let mut state: serde_json::Value =
            serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
        state["private_network"] = serde_json::Value::String("no".to_owned());
        fs::write(&state_path, serde_json::to_vec(&state).unwrap()).unwrap();
        let wrong_role = owner.reconcile(&fixture.context()).unwrap();
        assert!(matches!(
            &wrong_role[0].observation,
            ReconcileResult::Unknown
        ));

        state["private_network"] = serde_json::Value::String("yes".to_owned());
        fs::write(&state_path, serde_json::to_vec(&state).unwrap()).unwrap();
        fs::write(fixture.root.path().join("fail-show"), "").unwrap();

        assert!(matches!(
            owner.reconcile(&fixture.context()),
            Err(OwnerObservationError::ManagerUnavailable)
        ));
        assert!(!fixture.root.path().join("stops").exists());
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn stop_only_refuses_replaced_invocation_and_main_pid_without_stopping() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let intent = fixture.install_stale_supervisor_intent(&mut owner);
        let observations = owner.reconcile(&fixture.context()).unwrap();
        let ReconcileResult::StopOnly(stop_only) = &observations[0].observation else {
            panic!("expected stop-only recovery");
        };
        let state_path = fixture.root.path().join("manager-state.json");
        let mut state: serde_json::Value =
            serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();

        state["invocation"] = serde_json::Value::String("b".repeat(32));
        fs::write(&state_path, serde_json::to_vec(&state).unwrap()).unwrap();
        assert_eq!(
            owner.stop_stop_only(stop_only, &fixture.context()),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        );

        state["invocation"] = serde_json::Value::String("a".repeat(32));
        state["main_pid"] = serde_json::Value::String("4321".to_owned());
        fs::write(&state_path, serde_json::to_vec(&state).unwrap()).unwrap();
        assert_eq!(
            owner.stop_stop_only(stop_only, &fixture.context()),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        );

        assert!(owner
            .store
            .list()
            .unwrap()
            .iter()
            .any(|record| record == &intent));
        assert!(!fixture.root.path().join("stops").exists());
    }

    #[test]
    fn corrupt_wal_refuses_stop_only_without_removing_intent() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let intent = fixture.install_stale_supervisor_intent(&mut owner);
        let observations = owner.reconcile(&fixture.context()).unwrap();
        let ReconcileResult::StopOnly(stop_only) = &observations[0].observation else {
            panic!("expected stop-only recovery");
        };
        fs::write(fixture.intents.path().join("unexpected"), "preserve me").unwrap();

        assert_eq!(
            owner.stop_stop_only(stop_only, &fixture.context()),
            Err(ManagedServiceErrorKind::UnitStateUnverifiable)
        );
        assert!(!fixture.root.path().join("stops").exists());
        assert!(fixture
            .intents
            .path()
            .join(format!("{}.json", intent.unit_name()))
            .exists());
    }

    #[test]
    fn uncertain_stop_retains_wal_and_never_replays_start() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let intent = fixture.install_stale_supervisor_intent(&mut owner);
        let observations = owner.reconcile(&fixture.context()).unwrap();
        let ReconcileResult::StopOnly(stop_only) = &observations[0].observation else {
            panic!("expected stop-only recovery");
        };
        fs::write(fixture.root.path().join("fail-stop"), "").unwrap();

        assert_eq!(
            owner.stop_stop_only(stop_only, &fixture.context()),
            Err(ManagedServiceErrorKind::ManagerCommandFailed)
        );
        assert!(owner
            .store
            .list()
            .unwrap()
            .iter()
            .any(|record| record == &intent));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );

        let retry_observation = owner.reconcile(&fixture.context()).unwrap();
        assert!(matches!(
            &retry_observation[0].observation,
            ReconcileResult::StopOnly(_)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn ordinary_and_external_units_never_get_stop_only_capability() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();

        let observations = owner.reconcile(&fixture.context()).unwrap();
        assert!(matches!(
            &observations[0].observation,
            ReconcileResult::Owned(_)
        ));
        assert!(ServiceIntentRecord::from_persisted_parts(
            "fixture".to_owned(),
            "llama-swap.service".to_owned(),
            "b".repeat(64),
            30_000,
        )
        .is_err());
        assert!(!fixture.root.path().join("stops").exists());
    }

    #[test]
    fn uncertain_launcher_retains_durable_intent_for_observation() {
        let fixture = Fixture::new();
        fs::write(fixture.root.path().join("fail-launcher"), "").unwrap();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap_err();
        assert_eq!(error.disposition(), StartDisposition::OwnershipUncertain);
        let records = ServiceIntentStore::open(fixture.intents.path())
            .unwrap()
            .list()
            .unwrap();
        assert_eq!(records.len(), 1);
        assert!(error.intent_record() == Some(&records[0]));
        assert!(matches!(
            &owner.reconcile(&fixture.context()).unwrap()[0].observation,
            ReconcileResult::Owned(_)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
    }

    #[test]
    fn persistence_failure_refuses_manager_dispatch_and_preserves_files() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        fs::write(fixture.intents.path().join("unexpected"), "retain me").unwrap();
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap_err();
        assert!(matches!(error, OwnerStartError::Persistence { .. }));
        assert_eq!(error.disposition(), StartDisposition::NotDispatched);
        assert!(!fixture.root.path().join("starts").exists());
        assert_eq!(
            fs::read_to_string(fixture.intents.path().join("unexpected")).unwrap(),
            "retain me"
        );
    }

    #[test]
    fn expired_context_does_not_create_intent_or_launch() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let context = OperationContext::new(Instant::now(), &NeverCancel);
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &context)
            .unwrap_err();
        assert!(error.intent_record().is_none());
        assert!(owner.store.list().unwrap().is_empty());
        assert!(!fixture.root.path().join("starts").exists());
    }

    #[test]
    fn corrupt_store_refuses_stop_without_removing_ownership_record() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        fs::write(fixture.intents.path().join("unexpected"), "preserve me").unwrap();
        assert_eq!(
            owner.stop(&handle, &fixture.context()),
            Err(ManagedServiceErrorKind::UnitStateUnverifiable)
        );
        assert!(!fixture.root.path().join("stops").exists());
        assert!(fixture
            .intents
            .path()
            .join(format!("{}.json", handle.unit_name()))
            .exists());
        assert!(fixture.root.path().join("manager-state.json").exists());
    }

    #[test]
    fn cancellation_after_persistence_retains_record_without_dispatch() {
        struct CancelWhenRecorded<'a>(&'a Path);
        impl super::super::runtime::Cancellation for CancelWhenRecorded<'_> {
            fn is_cancelled(&self) -> bool {
                fs::read_dir(self.0).unwrap().next().is_some()
            }
        }
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let cancellation = CancelWhenRecorded(fixture.intents.path());
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &cancellation);
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &context)
            .unwrap_err();
        assert!(matches!(
            error,
            OwnerStartError::NotDispatched {
                kind: ManagedServiceErrorKind::Cancelled,
                ..
            }
        ));
        let records = owner.store.list().unwrap();
        assert_eq!(records.len(), 1);
        assert!(error.intent_record() == Some(&records[0]));
        assert!(!fixture.root.path().join("starts").exists());
        assert!(!fixture.root.path().join("manager-state.json").exists());
    }

    #[test]
    fn valid_nonmember_handle_is_refused_before_manager_stop() {
        let fixture = Fixture::new();
        let foreign = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let foreign_guard = CoordinatorLock::acquire(foreign.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let mut foreign_owner = foreign.owner(&foreign_guard);
        let handle = foreign_owner
            .start(foreign.recipe(), Duration::from_secs(1), &foreign.context())
            .unwrap();
        assert_eq!(
            owner.stop(&handle, &fixture.context()),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        );
        assert!(!fixture.root.path().join("stops").exists());
        assert!(!foreign.root.path().join("stops").exists());
        assert!(foreign.root.path().join("manager-state.json").exists());
    }

    #[test]
    fn stop_and_observe_returns_receipt_bound_empty_observation_and_retains_ledger() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, Some("populated 0\nfrozen 0\n"));

        let result = owner
            .stop_and_observe(&handle, &probe, &fixture.context())
            .unwrap();
        assert_eq!(result.observation().kind(), CgroupObservationKind::Empty);
        assert!(result.observation().matches(result.receipt()));
        assert!(fixture.root.path().join("stops").exists());
        assert_eq!(owner.store.list().unwrap().len(), 1);
        assert!(!format!("{result:?}").contains(handle.unit_name()));
    }

    #[test]
    fn stopped_intent_retires_only_after_fresh_empty_or_removed_observation() {
        for removed in [false, true] {
            let fixture = Fixture::new();
            let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
            let mut owner = fixture.owner(&guard);
            let handle = fixture.start(&mut owner);
            let probe = if removed {
                CgroupReleaseProbe::with_test_root(fixture.cgroups.path()).unwrap()
            } else {
                fixture.cgroup_probe(&handle, Some("populated 0\nfrozen 0\n"))
            };
            let stopped = owner
                .stop_and_observe(&handle, &probe, &fixture.context())
                .unwrap();
            assert_eq!(
                stopped.observation().kind(),
                if removed {
                    CgroupObservationKind::Removed
                } else {
                    CgroupObservationKind::Empty
                }
            );
            let record_path = fixture
                .intents
                .path()
                .join(format!("{}.json", handle.unit_name()));
            assert!(record_path.exists());

            owner
                .retire_stopped(&stopped, &probe, &fixture.context())
                .unwrap();
            assert!(owner.store.list().unwrap().is_empty());
            assert!(!record_path.exists());
            assert_eq!(
                fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
                "stop\n"
            );
        }
    }

    #[test]
    fn retirement_reobserves_and_refuses_occupied_or_missing_events() {
        for replacement in [Some("populated 1\nfrozen 0\n"), None] {
            let fixture = Fixture::new();
            let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
            let mut owner = fixture.owner(&guard);
            let handle = fixture.start(&mut owner);
            let (probe, stopped) =
                fixture.stop_with_observation(&mut owner, &handle, Some("populated 0\nfrozen 0\n"));
            let events_path = fixture.cgroup_leaf(&handle).join("cgroup.events");
            if let Some(contents) = replacement {
                fs::write(&events_path, contents).unwrap();
            } else {
                fs::remove_file(&events_path).unwrap();
            }

            let error = owner
                .retire_stopped(&stopped, &probe, &fixture.context())
                .unwrap_err();
            assert_eq!(error.disposition(), IntentRetirementDisposition::NotRetired);
            assert!(matches!(
                error.kind(),
                IntentRetirementErrorKind::Cgroup(
                    CgroupReleaseErrorKind::CgroupNotEmpty | CgroupReleaseErrorKind::MissingEvents
                )
            ));
            assert_eq!(error.intent_record(), handle.intent_record());
            assert_eq!(owner.store.list().unwrap().len(), 1);
        }
    }

    #[test]
    fn retirement_refuses_active_replaced_or_unknown_manager_state() {
        let modes = [
            ("active", 'a', ManagedServiceErrorKind::ServiceNotActive),
            ("inactive", 'b', ManagedServiceErrorKind::OwnershipMismatch),
            (
                "activating",
                'a',
                ManagedServiceErrorKind::UnitStateUnverifiable,
            ),
        ];
        for (active, invocation, expected) in modes {
            let fixture = Fixture::new();
            let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
            let mut owner = fixture.owner(&guard);
            let handle = fixture.start(&mut owner);
            let mut replacement: serde_json::Value = serde_json::from_slice(
                &fs::read(fixture.root.path().join("manager-state.json")).unwrap(),
            )
            .unwrap();
            let (probe, stopped) =
                fixture.stop_with_observation(&mut owner, &handle, Some("populated 0\nfrozen 0\n"));
            replacement["active"] = serde_json::Value::String(active.to_owned());
            replacement["invocation"] =
                serde_json::Value::String(invocation.to_string().repeat(32));
            fs::write(
                fixture.root.path().join("manager-state.json"),
                serde_json::to_vec(&replacement).unwrap(),
            )
            .unwrap();

            let error = owner
                .retire_stopped(&stopped, &probe, &fixture.context())
                .unwrap_err();
            assert_eq!(error.disposition(), IntentRetirementDisposition::NotRetired);
            assert_eq!(error.kind(), IntentRetirementErrorKind::Manager(expected));
            assert_eq!(owner.store.list().unwrap().len(), 1);
        }

        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = fixture.start(&mut owner);
        let (probe, stopped) =
            fixture.stop_with_observation(&mut owner, &handle, Some("populated 0\nfrozen 0\n"));
        fs::write(fixture.root.path().join("fail-show"), "").unwrap();
        let error = owner
            .retire_stopped(&stopped, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(error.disposition(), IntentRetirementDisposition::NotRetired);
        assert!(matches!(
            error.kind(),
            IntentRetirementErrorKind::Manager(ManagedServiceErrorKind::ManagerCommandFailed)
        ));
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn mismatched_or_unsafe_retained_record_is_never_unlinked() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start_for_runtime(
                RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
                fixture.recipe(),
                Duration::from_secs(1),
                &fixture.context(),
            )
            .unwrap();
        let (probe, stopped) =
            fixture.stop_with_observation(&mut owner, &handle, Some("populated 0\nfrozen 0\n"));
        let record_path = fixture
            .intents
            .path()
            .join(format!("{}.json", handle.unit_name()));
        let mut stored: serde_json::Value =
            serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
        stored["runtime_name"] = serde_json::Value::String("llama-swap".to_owned());
        fs::write(&record_path, serde_json::to_vec(&stored).unwrap()).unwrap();

        let error = owner
            .retire_stopped(&stopped, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(error.disposition(), IntentRetirementDisposition::NotRetired);
        assert_eq!(
            error.kind(),
            IntentRetirementErrorKind::Store(ServiceIntentStoreError::IntentMismatch)
        );
        assert_eq!(error.intent_record(), handle.intent_record());
        assert!(record_path.exists());

        for unsafe_kind in ["corrupt", "mode", "symlink", "other-corrupt"] {
            let fixture = Fixture::new();
            let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
            let mut owner = fixture.owner(&guard);
            let handle = fixture.start(&mut owner);
            let (probe, stopped) =
                fixture.stop_with_observation(&mut owner, &handle, Some("populated 0\nfrozen 0\n"));
            let record_path = fixture
                .intents
                .path()
                .join(format!("{}.json", handle.unit_name()));
            match unsafe_kind {
                "corrupt" => fs::write(&record_path, b"{not-json").unwrap(),
                "mode" => {
                    fs::set_permissions(&record_path, fs::Permissions::from_mode(0o644)).unwrap()
                }
                "symlink" => {
                    let target = fixture.root.path().join("outside-record");
                    fs::write(&target, b"preserve-target").unwrap();
                    fs::remove_file(&record_path).unwrap();
                    std::os::unix::fs::symlink(&target, &record_path).unwrap();
                }
                "other-corrupt" => {
                    fs::write(fixture.intents.path().join("unexpected"), b"retain me").unwrap();
                }
                _ => unreachable!(),
            }

            let error = owner
                .retire_stopped(&stopped, &probe, &fixture.context())
                .unwrap_err();
            assert_eq!(error.disposition(), IntentRetirementDisposition::NotRetired);
            assert!(matches!(
                error.kind(),
                IntentRetirementErrorKind::Store(ServiceIntentStoreError::CorruptStore)
            ));
            assert!(record_path.exists());
            if unsafe_kind == "symlink" {
                assert_eq!(
                    fs::read(fixture.root.path().join("outside-record")).unwrap(),
                    b"preserve-target"
                );
            }
        }
    }

    #[test]
    fn retirement_reclaims_capacity_without_replaying_or_rewriting_other_records() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = fixture.start(&mut owner);
        let (probe, stopped) =
            fixture.stop_with_observation(&mut owner, &handle, Some("populated 0\nfrozen 0\n"));
        let mut stored_count = owner.store.list().unwrap().len();
        let mut suffix = 0_u128;
        while stored_count < 128 {
            let record = ServiceIntentRecord::from_persisted_parts(
                "fixture".to_owned(),
                format!("l3ms-fixture-{suffix:032x}.service"),
                "b".repeat(64),
                30_000,
            )
            .unwrap();
            if record.unit_name() != handle.unit_name() {
                owner.store.record(&record).unwrap();
                stored_count += 1;
            }
            suffix += 1;
        }
        assert_eq!(owner.store.list().unwrap().len(), 128);
        let extra = ServiceIntentRecord::from_persisted_parts(
            "fixture".to_owned(),
            format!("l3ms-fixture-{:032x}.service", 9999_u128),
            "d".repeat(64),
            30_000,
        )
        .unwrap();
        assert_eq!(
            owner.store.record(&extra),
            Err(ServiceIntentStoreError::CapacityExceeded)
        );

        owner
            .retire_stopped(&stopped, &probe, &fixture.context())
            .unwrap();
        assert_eq!(owner.store.list().unwrap().len(), 127);
        owner.store.record(&extra).unwrap();
        assert_eq!(owner.store.list().unwrap().len(), 128);
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
    }

    #[test]
    fn cancellation_or_deadline_before_retirement_preserves_intent() {
        struct Cancel;
        impl Cancellation for Cancel {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        for cancelled in [true, false] {
            let fixture = Fixture::new();
            let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
            let mut owner = fixture.owner(&guard);
            let handle = fixture.start(&mut owner);
            let (probe, stopped) =
                fixture.stop_with_observation(&mut owner, &handle, Some("populated 0\nfrozen 0\n"));
            let context = if cancelled {
                OperationContext::new(Instant::now() + Duration::from_secs(2), &Cancel)
            } else {
                OperationContext::new(Instant::now(), &NeverCancel)
            };
            let error = owner
                .retire_stopped(&stopped, &probe, &context)
                .unwrap_err();
            assert_eq!(error.disposition(), IntentRetirementDisposition::NotRetired);
            assert_eq!(
                error.kind(),
                if cancelled {
                    IntentRetirementErrorKind::Cancelled
                } else {
                    IntentRetirementErrorKind::DeadlineExceeded
                }
            );
            assert_eq!(error.intent_record(), handle.intent_record());
            assert!(fixture
                .intents
                .path()
                .join(format!("{}.json", handle.unit_name()))
                .exists());
        }
    }

    #[test]
    fn observation_failure_retains_transferable_receipt_without_repeating_stop() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, Some("populated 1\nfrozen 0\n"));

        let error = owner
            .stop_and_observe(&handle, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Observation(CgroupReleaseErrorKind::CgroupNotEmpty)
        );
        assert!(error.receipt().is_some());
        assert!(!format!("{error:?}").contains(handle.unit_name()));
        let receipt = error.into_receipt().unwrap();
        let retry = owner
            .observe_stopped(receipt, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(
            retry.kind(),
            StopAndObserveErrorKind::Observation(CgroupReleaseErrorKind::CgroupNotEmpty)
        );
        let receipt = retry.into_receipt().unwrap();
        fs::write(
            fixture.cgroup_leaf(&handle).join("cgroup.events"),
            "populated 0\nfrozen 0\n",
        )
        .unwrap();
        let observed = owner
            .observe_stopped(receipt, &probe, &fixture.context())
            .unwrap();
        assert_eq!(observed.receipt().intent_record(), handle.intent_record());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn cancelled_observation_retry_retains_receipt_without_repeating_stop() {
        struct Cancel;
        impl Cancellation for Cancel {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = fixture.start(&mut owner);
        let probe = fixture.cgroup_probe(&handle, Some("populated 0\nfrozen 0\n"));
        let receipt = owner.stop(&handle, &fixture.context()).unwrap();
        let cancelled = OperationContext::new(Instant::now() + Duration::from_secs(3), &Cancel);
        let error = owner
            .observe_stopped(receipt, &probe, &cancelled)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Stop(ManagedServiceErrorKind::Cancelled)
        );
        let receipt = error.into_receipt().unwrap();
        assert_eq!(receipt.intent_record(), handle.intent_record());
        owner
            .observe_stopped(receipt, &probe, &fixture.context())
            .unwrap();
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn observation_retry_rejects_unretained_receipt_without_another_stop() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = fixture.start(&mut owner);
        let probe = fixture.cgroup_probe(&handle, Some("populated 0\nfrozen 0\n"));
        let receipt = owner.stop(&handle, &fixture.context()).unwrap();
        let other = Fixture::new();
        let mut wrong_owner = other.owner(&guard);
        let error = wrong_owner
            .observe_stopped(receipt, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Stop(ManagedServiceErrorKind::OwnershipMismatch)
        );
        let receipt = error.into_receipt().unwrap();
        owner
            .observe_stopped(receipt, &probe, &fixture.context())
            .unwrap();
        assert!(!other.root.path().join("stops").exists());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn missing_events_is_observation_failure_with_receipt_but_not_removal() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, None);

        let error = owner
            .stop_and_observe(&handle, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Observation(CgroupReleaseErrorKind::MissingEvents)
        );
        assert!(error.receipt().is_some());
        assert_eq!(owner.store.list().unwrap().len(), 1);
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn cancelled_stop_has_no_receipt_and_does_not_dispatch_manager_stop() {
        struct Cancel;
        impl Cancellation for Cancel {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, Some("populated 0\n"));
        let cancelled = OperationContext::new(Instant::now() + Duration::from_secs(2), &Cancel);

        let error = owner
            .stop_and_observe(&handle, &probe, &cancelled)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Stop(ManagedServiceErrorKind::Cancelled)
        );
        assert!(error.receipt().is_none());
        assert!(!fixture.root.path().join("stops").exists());
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    struct FlagCancellation(Arc<AtomicBool>);

    impl Cancellation for FlagCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn valid_decision_result() -> NativeJobResult {
        let mut answers = serde_json::Map::new();
        answers.insert(
            "choice".to_owned(),
            serde_json::json!({"choice":"yes", "confidence":0.8}),
        );
        let mut usage = serde_json::Map::new();
        usage.insert("input_tokens".to_owned(), serde_json::Value::from(7));
        NativeJobResult::Decision(NativeDecisionResult {
            result: InferenceResult::Decision { answers },
            native_latency: Some(Duration::from_micros(23)),
            native_usage: Some(usage),
        })
    }

    #[test]
    fn finalizer_returns_decision_only_after_release_and_preserves_native_metadata() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_decision_job(&mut owner);
        fixture.claim_and_write_result(&job, &valid_decision_result());
        let probe = fixture.empty_cgroup_probe(&job);
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);
        finalizer.gpu_idle_observer = Some(Box::new(|_| panic!("CPU job must skip GPU probe")));

        let outcome = finalizer.finalize(&fixture.context()).unwrap();
        let ReleasedNativeOutcome::Decision(native) = outcome else {
            panic!("decision outcome expected")
        };
        assert_eq!(native.native_latency, Some(Duration::from_micros(23)));
        assert_eq!(
            native.native_usage.as_ref().unwrap().get("input_tokens"),
            Some(&serde_json::Value::from(7))
        );
        assert!(matches!(
            native.result,
            InferenceResult::Decision { answers }
                if answers.get("choice").and_then(|value| value.get("choice"))
                    == Some(&serde_json::Value::String("yes".to_owned()))
        ));
        assert_eq!(
            finalizer.finalize(&fixture.context()).unwrap_err().kind(),
            OwnedJobFinalizeErrorKind::AlreadyReleased
        );
        drop(finalizer);
        assert!(owner.store.list().unwrap().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn finalizer_retains_one_stop_receipt_across_cgroup_and_cancellation_retries() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_decision_job(&mut owner);
        fixture.claim_and_write_result(
            &job,
            &NativeJobResult::Failed(NativeJobFailure::Unavailable),
        );
        let events = fixture.cgroup_events_path(job.service());
        let probe = fixture.cgroup_probe(job.service(), Some("populated 1\nfrozen 0\n"));
        let flag = Arc::new(AtomicBool::new(false));
        let cancellation = FlagCancellation(flag.clone());
        let cancelled =
            OperationContext::new(Instant::now() + Duration::from_secs(3), &cancellation);
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);

        let first = finalizer.finalize(&fixture.context()).unwrap_err();
        assert_eq!(
            first.kind(),
            OwnedJobFinalizeErrorKind::Observation(StopAndObserveErrorKind::Observation(
                CgroupReleaseErrorKind::CgroupNotEmpty
            ))
        );
        assert!(finalizer.stopped_receipt.is_some());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );

        fs::write(&events, "populated 0\nfrozen 0\n").unwrap();
        flag.store(true, Ordering::SeqCst);
        assert_eq!(
            finalizer.finalize(&cancelled).unwrap_err().kind(),
            OwnedJobFinalizeErrorKind::Context(ManagedServiceErrorKind::Cancelled)
        );
        assert!(finalizer.stopped_receipt.is_some());
        flag.store(false, Ordering::SeqCst);
        assert!(matches!(
            finalizer.finalize(&fixture.context()).unwrap(),
            ReleasedNativeOutcome::Failed(NativeJobFailure::Unavailable)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn uncertain_stop_is_never_issued_twice_by_finalizer() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_decision_job(&mut owner);
        fixture.claim_and_write_result(
            &job,
            &NativeJobResult::Failed(NativeJobFailure::Unavailable),
        );
        let probe = fixture.empty_cgroup_probe(&job);
        fs::write(fixture.root.path().join("fail-stop"), "1").unwrap();
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);

        assert_eq!(
            finalizer.finalize(&fixture.context()).unwrap_err().kind(),
            OwnedJobFinalizeErrorKind::Stop(ManagedServiceErrorKind::ManagerCommandFailed)
        );
        fs::remove_file(fixture.root.path().join("fail-stop")).unwrap();
        assert_eq!(
            finalizer.finalize(&fixture.context()).unwrap_err().kind(),
            OwnedJobFinalizeErrorKind::StopUncertain
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn finalizer_rejects_foreign_wal_without_stopping_foreign_job() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_decision_job(&mut owner);
        fixture.claim_and_write_result(
            &job,
            &NativeJobResult::Failed(NativeJobFailure::Unavailable),
        );

        let foreign_path = fixture.root.path().join("foreign-intents");
        fs::create_dir(&foreign_path).unwrap();
        fs::set_permissions(&foreign_path, fs::Permissions::from_mode(0o700)).unwrap();
        let mut foreign_owner = ManagedRuntimeOwner::new(
            fixture.controller(),
            ServiceIntentStore::open(&foreign_path).unwrap(),
            &guard,
        );
        let foreign_probe = fixture.empty_cgroup_probe(&job);
        let mut foreign_finalizer = foreign_owner.begin_owned_job_finalization(job, foreign_probe);
        assert_eq!(
            foreign_finalizer
                .finalize(&fixture.context())
                .unwrap_err()
                .kind(),
            OwnedJobFinalizeErrorKind::InvalidAssociation
        );
        assert!(!fixture.root.path().join("stops").exists());
        drop(foreign_finalizer);
        drop(foreign_owner);
        assert_eq!(owner.store.list().unwrap().len(), 1);
        assert!(!fixture.root.path().join("stops").exists());
    }

    #[test]
    fn finalizer_rejects_job_packet_bound_to_another_service() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let started = fixture.start_decision_job(&mut owner);
        let first_service = started.service;

        let second_directory = fixture.root.path().join("second-owned-job");
        fs::create_dir(&second_directory).unwrap();
        fs::set_permissions(&second_directory, fs::Permissions::from_mode(0o700)).unwrap();
        let second_store = OwnedJobStore::open(&second_directory).unwrap();
        let request = fixture.decision_request();
        let catalog = Catalog::load(&fixture.repo).unwrap();
        let route = resolve_profile_route(&catalog, request.profile()).unwrap();
        let second_job = second_store
            .write_request(OwnedJobRequestSpec {
                job_id: 18,
                request_id: "fixture-other-job",
                repository_root: &fixture.repo,
                request: &request,
                accepted_route: &route,
                device: DeviceSelection::Cpu,
                bounds: JobExecutionBounds::new(
                    Duration::from_secs(30),
                    RecoveryPolicy::new(Duration::from_secs(2)).unwrap(),
                )
                .unwrap(),
            })
            .unwrap();
        let wrong_association = OwnedNativeJob {
            service: first_service,
            store: second_store,
            job: second_job,
        };
        let probe = fixture.cgroup_probe(wrong_association.service(), Some("populated 0\n"));
        let mut finalizer = owner.begin_owned_job_finalization(wrong_association, probe);

        assert_eq!(
            finalizer.finalize(&fixture.context()).unwrap_err().kind(),
            OwnedJobFinalizeErrorKind::InvalidAssociation
        );
        assert!(!fixture.root.path().join("stops").exists());
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn finalizer_rejects_stale_generation_without_repeating_stop() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_decision_job(&mut owner);
        fixture.claim_and_write_result(
            &job,
            &NativeJobResult::Failed(NativeJobFailure::Unavailable),
        );
        let original_state_path = fixture.root.path().join("manager-state.json");
        let original_state = fs::read(&original_state_path).unwrap();
        let events = fixture.cgroup_events_path(job.service());
        let probe = fixture.cgroup_probe(job.service(), Some("populated 1\nfrozen 0\n"));
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);
        assert!(finalizer.finalize(&fixture.context()).is_err());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );

        fs::write(&events, "populated 0\nfrozen 0\n").unwrap();
        let mut replacement: serde_json::Value = serde_json::from_slice(&original_state).unwrap();
        replacement["invocation"] = serde_json::Value::String("b".repeat(32));
        replacement["active"] = serde_json::Value::String("active".to_owned());
        fs::write(
            &original_state_path,
            serde_json::to_vec(&replacement).unwrap(),
        )
        .unwrap();
        let stale = finalizer.finalize(&fixture.context()).unwrap_err();
        assert!(matches!(
            stale.kind(),
            OwnedJobFinalizeErrorKind::Observation(StopAndObserveErrorKind::Stop(_))
        ));
        assert!(finalizer.stopped_receipt.is_some());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn invalid_or_mutated_native_packet_becomes_failure_only_after_release() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_decision_job(&mut owner);
        fixture.claim_and_write_result(&job, &valid_decision_result());
        fs::write(fixture.job_directory.join("l3ms-job-result.json"), b"{").unwrap();
        let probe = fixture.empty_cgroup_probe(&job);
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);

        assert!(matches!(
            finalizer.finalize(&fixture.context()).unwrap(),
            ReleasedNativeOutcome::Failed(NativeJobFailure::InvalidOutput)
        ));
        drop(finalizer);
        assert!(owner.store.list().unwrap().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn finalizer_uses_same_physical_gpu_and_retries_probe_without_repeating_stop() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_breeze_job(&mut owner);
        fixture.claim_and_write_result(
            &job,
            &NativeJobResult::Failed(NativeJobFailure::Unavailable),
        );
        let events = fixture.cgroup_events_path(job.service());
        let probe = fixture.empty_cgroup_probe(&job);
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls_for_probe = calls.clone();
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);
        finalizer.gpu_idle_observer = Some(Box::new(move |config| {
            calls_for_probe.lock().unwrap().push((
                config.gpu_uuid().to_owned(),
                config.expected_index(),
                config.max_idle_mib(),
            ));
            if calls_for_probe.lock().unwrap().len() == 1 {
                Err(GpuProbeFailureKind::MemoryBusy)
            } else {
                Ok(config.gpu_uuid().to_owned())
            }
        }));

        assert_eq!(
            finalizer.finalize(&fixture.context()).unwrap_err().kind(),
            OwnedJobFinalizeErrorKind::Gpu(GpuProbeFailureKind::MemoryBusy)
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
        // An earlier empty observation is not release authority on retry.
        // Recheck the same stopped generation before querying the GPU again.
        fs::write(&events, "populated 1\nfrozen 0\n").unwrap();
        assert!(matches!(
            finalizer.finalize(&fixture.context()).unwrap_err().kind(),
            OwnedJobFinalizeErrorKind::Observation(_)
        ));
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
        fs::write(&events, "populated 0\nfrozen 0\n").unwrap();
        assert!(matches!(
            finalizer.finalize(&fixture.context()).unwrap(),
            ReleasedNativeOutcome::Failed(NativeJobFailure::Unavailable)
        ));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                (
                    "GPU-01234567-89ab-cdef-0123-456789abcdef".to_owned(),
                    0,
                    256,
                ),
                (
                    "GPU-01234567-89ab-cdef-0123-456789abcdef".to_owned(),
                    0,
                    256,
                ),
            ]
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn finalizer_revalidates_breeze_artifacts_after_stop_and_returns_relative_refs() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_breeze_job(&mut owner);
        job.store.claim_execution(&job.job).unwrap();
        fs::write(
            fixture.job_directory.join("result.wav"),
            Fixture::minimal_wave(),
        )
        .unwrap();
        let candidate = super::super::media_execution::revalidate_breeze_artifacts(
            &fixture.job_directory,
            SpeechMode::Standard,
            &fixture.context(),
        )
        .unwrap();
        let native = NativeJobResult::from_breeze(&candidate).unwrap();
        job.store.write_result(&job.job, &native).unwrap();
        let probe = fixture.empty_cgroup_probe(&job);
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);
        finalizer.gpu_idle_observer = Some(Box::new(|config| Ok(config.gpu_uuid().to_owned())));

        let ReleasedNativeOutcome::Breeze { mode, artifacts } =
            finalizer.finalize(&fixture.context()).unwrap()
        else {
            panic!("validated Breeze artifact outcome expected")
        };
        assert_eq!(mode, SpeechMode::Standard);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, ArtifactKind::LocalJob);
        assert_eq!(artifacts[0].relative_path, "result.wav");
        assert!(!format!("{:?}", artifacts[0]).contains("owned-job"));
    }

    #[test]
    fn artifact_mutation_returns_fixed_failure_after_cgroup_gpu_and_wal_cleanup() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_breeze_job(&mut owner);
        job.store.claim_execution(&job.job).unwrap();
        fs::write(
            fixture.job_directory.join("result.wav"),
            Fixture::minimal_wave(),
        )
        .unwrap();
        let candidate = super::super::media_execution::revalidate_breeze_artifacts(
            &fixture.job_directory,
            SpeechMode::Standard,
            &fixture.context(),
        )
        .unwrap();
        let native = NativeJobResult::from_breeze(&candidate).unwrap();
        job.store.write_result(&job.job, &native).unwrap();
        fs::write(
            fixture.job_directory.join("result.wav"),
            b"invalid but regular artifact bytes",
        )
        .unwrap();
        let probe = fixture.empty_cgroup_probe(&job);
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);
        finalizer.gpu_idle_observer = Some(Box::new(|config| Ok(config.gpu_uuid().to_owned())));

        assert!(matches!(
            finalizer.finalize(&fixture.context()).unwrap(),
            ReleasedNativeOutcome::Failed(NativeJobFailure::InvalidOutput)
        ));
        drop(finalizer);
        assert!(owner.store.list().unwrap().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn retirement_failure_retains_pending_outcome_receipt_and_job_for_readonly_retry() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let job = fixture.start_breeze_job(&mut owner);
        fixture.claim_and_write_result(
            &job,
            &NativeJobResult::Failed(NativeJobFailure::Unavailable),
        );
        let intent_path = fixture
            .intents
            .path()
            .join(format!("{}.json", job.service().unit_name()));
        let probe = fixture.empty_cgroup_probe(&job);
        let mut finalizer = owner.begin_owned_job_finalization(job, probe);
        let mut first_probe = true;
        let probe_intent_path = intent_path.clone();
        finalizer.gpu_idle_observer = Some(Box::new(move |config| {
            if first_probe {
                fs::set_permissions(&probe_intent_path, fs::Permissions::from_mode(0o400)).unwrap();
                first_probe = false;
            }
            Ok(config.gpu_uuid().to_owned())
        }));

        let error = finalizer.finalize(&fixture.context()).unwrap_err();
        assert!(matches!(
            error.kind(),
            OwnedJobFinalizeErrorKind::Retirement(_)
        ));
        assert!(finalizer.stopped_observation.is_some());
        assert!(finalizer.pending_outcome.is_some());
        assert!(!finalizer.released);
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );

        fs::set_permissions(&intent_path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(matches!(
            finalizer.finalize(&fixture.context()).unwrap(),
            ReleasedNativeOutcome::Failed(NativeJobFailure::Unavailable)
        ));
        drop(finalizer);
        assert!(owner.store.list().unwrap().is_empty());
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }
}
