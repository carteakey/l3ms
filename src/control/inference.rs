//! Queue/admission core for the future host-local coordinator.
//!
//! This module has no engine or IPC implementation. A dispatch ticket reserves
//! in-flight work; the host executor must separately prove runtime readiness,
//! admission, and GPU residency before execution. Completion releases only the
//! request reservation, never proves that a resident model released VRAM.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Weak,
};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};

#[cfg(target_os = "linux")]
use super::catalog::Task;
use super::catalog::{Catalog, Device, Lane, ProfileKey, RuntimeIdentity};
use super::ipc::{
    self, AcceptedOperation, ClientRequest, RequestStatus, ResponseEnvelope, ResponseErrorCode,
    ResponseMessage,
};
use super::requests::{InferenceRequest, MediaInput};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::{Cancellation, ResourceClaim};
use crate::job_history::{CrossToolkitJobStart, CrossToolkitJobUpdate, JobHistory, JobRecord};

#[cfg(target_os = "linux")]
use super::managed_service::OwnedServiceGenerationIdentity;
use crate::state_store::ComfyServiceGenerationReference;

#[cfg(target_os = "linux")]
use super::requests::{SpeechInput, SpeechMode};

const MAX_CAPACITY: usize = 128;
const MAX_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// Fixed submission failures; no request contents, native paths or OS errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionError {
    Unavailable,
    RuntimeAdmissionClosed,
    QueueFull,
    DuplicateRequest,
    InvalidRequest,
    Internal,
}
impl std::fmt::Display for SubmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "coordinator submission: {self:?}")
    }
}
impl std::error::Error for SubmissionError {}

#[derive(Clone)]
pub struct RequestCancellation(Arc<AtomicBool>);

impl RequestCancellation {
    fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn same_token_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Cancellation for RequestCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionState {
    ReconciliationRequired,
    Ready,
}

/// Fixed failures from per-runtime admission closure operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeAdmissionError {
    UnboundRuntime,
    GenerationExhausted,
    StaleCloseToken,
}

impl std::fmt::Display for RuntimeAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::UnboundRuntime => "runtime admission: runtime is not bound",
            Self::GenerationExhausted => "runtime admission: close generation exhausted",
            Self::StaleCloseToken => "runtime admission: close token is stale",
        })
    }
}

impl std::error::Error for RuntimeAdmissionError {}

/// Opaque proof of one runtime-specific admission close generation. The
/// token is deliberately non-Clone; a later close for the same runtime makes
/// this generation stale.
pub struct RuntimeAdmissionClose {
    runtime: RuntimeIdentity,
    generation: u64,
    owner_identity: Arc<()>,
}

impl std::fmt::Debug for RuntimeAdmissionClose {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RuntimeAdmissionClose([redacted])")
    }
}

/// Accepted local reservations and unresolved retained history for one runtime.
/// A zero local pending/active count is not drained while retained ownership is
/// unknown or interrupted; even a fully drained report does not prove backend
/// cleanup or resource release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeDrainState {
    pending_jobs: usize,
    active_jobs: usize,
    unresolved_retained_jobs: usize,
    admission_closed: bool,
}

impl RuntimeDrainState {
    pub fn pending_jobs(self) -> usize {
        self.pending_jobs
    }

    pub fn active_jobs(self) -> usize {
        self.active_jobs
    }

    pub fn unresolved_retained_jobs(self) -> usize {
        self.unresolved_retained_jobs
    }

    pub fn admission_closed(self) -> bool {
        self.admission_closed
    }

    pub fn is_drained(self) -> bool {
        self.admission_closed
            && self.pending_jobs == 0
            && self.active_jobs == 0
            && self.unresolved_retained_jobs == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Completion {
    Succeeded,
    Failed,
    Cancelled,
    /// Backend outcome or cleanup cannot be proved; no replay is permitted.
    Uncertain,
}

impl Completion {
    fn history_state(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Uncertain => "interrupted",
        }
    }
}

struct Pending {
    job_id: u64,
    request_id: String,
    request: InferenceRequest,
    route: ResolvedProfileRoute,
    resource: ResourceClaim,
    submitted: Instant,
    deadline: Instant,
    cancellation: RequestCancellation,
}

struct Active {
    permit_reservation_identity: Arc<()>,
    request_id: String,
    request: InferenceRequest,
    profile: ProfileKey,
    task: String,
    route: ResolvedProfileRoute,
    resource: ResourceClaim,
    cancellation: RequestCancellation,
    cleanup_unresolved: bool,
    dispatch_validated: bool,
    deadline: Instant,
    reserved: Instant,
    started: Option<Instant>,
    queue_wait: Duration,
}

/// Caller-owned work payload. Dropping this ticket does not release its
/// reservation: the executor must report completion or uncertain cleanup.
pub struct DispatchTicket {
    pub job_id: u64,
    pub request_id: String,
    pub request: InferenceRequest,
    pub route: ResolvedProfileRoute,
    pub resource: ResourceClaim,
    pub deadline: Instant,
    pub cancellation: RequestCancellation,
    pub queue_wait: Duration,
}

/// One-shot capability to submit exactly the ComfyUI candidate persisted for
/// one accepted dispatch. It retains the request transiently for exact handoff
/// matching; it is never persisted or exposed in `Debug`.
pub struct ComfyDispatchPermit {
    owner_identity: Weak<()>,
    reservation_identity: Weak<()>,
    cancellation: RequestCancellation,
    deadline: Instant,
    job_id: u64,
    request_id: String,
    request: InferenceRequest,
    candidate_id: String,
    route: ResolvedProfileRoute,
}

impl ComfyDispatchPermit {
    pub fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn route(&self) -> &ResolvedProfileRoute {
        &self.route
    }

    pub(crate) fn matches_candidate(
        &self,
        candidate_id: &str,
        route: &ResolvedProfileRoute,
    ) -> bool {
        self.is_live() && self.candidate_id == candidate_id && self.route == *route
    }

    pub(crate) fn matches_request(
        &self,
        request_id: &str,
        request: &InferenceRequest,
        route: &ResolvedProfileRoute,
    ) -> bool {
        self.is_live()
            && self.request_id == request_id
            && same_request(&self.request, request)
            && self.route == *route
    }

    pub(crate) fn belongs_to(&self, owner_identity: &Arc<()>) -> bool {
        self.owner_identity
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, owner_identity))
    }

    fn is_live(&self) -> bool {
        self.owner_identity.upgrade().is_some()
            && self.reservation_identity.upgrade().is_some()
            && !self.cancellation.is_cancelled()
            && Instant::now() < self.deadline
    }
}

/// One-shot managed Comfy handoff bound to the exact owner-issued supervisor
/// generation whose safe comparison reference was persisted with its UUID.
/// The identity and reference are transient evidence only: this type has no
/// persistence or restoration constructor and does not prove native acceptance.
#[cfg(target_os = "linux")]
pub struct ManagedComfyDispatchPermit {
    dispatch: ComfyDispatchPermit,
    generation_identity: OwnedServiceGenerationIdentity,
    generation_reference: ComfyServiceGenerationReference,
}

#[cfg(target_os = "linux")]
impl ManagedComfyDispatchPermit {
    pub fn job_id(&self) -> u64 {
        self.dispatch.job_id()
    }

    pub fn request_id(&self) -> &str {
        self.dispatch.request_id()
    }

    pub fn candidate_id(&self) -> &str {
        self.dispatch.candidate_id()
    }

    pub fn route(&self) -> &ResolvedProfileRoute {
        self.dispatch.route()
    }

    pub(crate) fn matches_candidate(
        &self,
        candidate_id: &str,
        route: &ResolvedProfileRoute,
    ) -> bool {
        self.dispatch.matches_candidate(candidate_id, route)
    }

    pub(crate) fn matches_request(
        &self,
        request_id: &str,
        request: &InferenceRequest,
        route: &ResolvedProfileRoute,
    ) -> bool {
        self.dispatch.matches_request(request_id, request, route)
    }

    pub(crate) fn matches_generation(&self, generation: &OwnedServiceGenerationIdentity) -> bool {
        self.generation_identity == *generation
            && generation
                .comfy_service_generation_reference()
                .is_ok_and(|reference| reference == self.generation_reference)
    }
}

#[cfg(target_os = "linux")]
impl std::fmt::Debug for ManagedComfyDispatchPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ManagedComfyDispatchPermit([redacted])")
    }
}

impl std::fmt::Debug for ComfyDispatchPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ComfyDispatchPermit([redacted])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestTiming {
    pub queue_wait: Duration,
    pub cold_start: Duration,
    pub execution: Duration,
}

/// Fixed, redacted refusal categories for the final pre-dispatch gate.
///
/// These errors deliberately contain no manifest paths, request payloads, or
/// underlying filesystem/catalog diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchBeginError {
    AdmissionUnavailable,
    ReservationUnavailable,
    TicketMismatch,
    AlreadyStarted,
    Cancelled,
    DeadlineExceeded,
    CatalogUnavailable,
    InvalidRequest,
    RouteChanged,
    BindingChanged,
    JobAssociationMismatch,
    PersistenceFailed,
}

impl std::fmt::Display for DispatchBeginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::AdmissionUnavailable => "coordinator admission is unavailable",
            Self::ReservationUnavailable => "dispatch reservation is unavailable",
            Self::TicketMismatch => "dispatch ticket does not match its reservation",
            Self::AlreadyStarted => "dispatch ticket was already started",
            Self::Cancelled => "dispatch ticket was cancelled",
            Self::DeadlineExceeded => "dispatch ticket deadline expired",
            Self::CatalogUnavailable => "current runtime catalog is unavailable",
            Self::InvalidRequest => "dispatch request is invalid for the current catalog",
            Self::RouteChanged => "dispatch route changed after acceptance",
            Self::BindingChanged => "dispatch resource binding changed after acceptance",
            Self::JobAssociationMismatch => "dispatch history association is invalid",
            Self::PersistenceFailed => "dispatch start state could not be persisted",
        })
    }
}

impl std::error::Error for DispatchBeginError {}

/// Fixed refusal categories for pre-POST Comfy candidate persistence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComfyCandidateError {
    AdmissionUnavailable,
    ReservationUnavailable,
    FinalDispatchGateMissing,
    TicketMismatch,
    Cancelled,
    DeadlineExceeded,
    CatalogUnavailable,
    UnsupportedRoute,
    RouteChanged,
    BindingChanged,
    JobAssociationMismatch,
    ServiceGenerationMismatch,
    CandidateRejected,
    PersistenceFailed,
}

impl std::fmt::Display for ComfyCandidateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::AdmissionUnavailable => "Comfy candidate: admission unavailable",
            Self::ReservationUnavailable => "Comfy candidate: reservation unavailable",
            Self::FinalDispatchGateMissing => "Comfy candidate: final dispatch gate missing",
            Self::TicketMismatch => "Comfy candidate: ticket mismatch",
            Self::Cancelled => "Comfy candidate: request cancelled",
            Self::DeadlineExceeded => "Comfy candidate: deadline exceeded",
            Self::CatalogUnavailable => "Comfy candidate: catalog unavailable",
            Self::UnsupportedRoute => "Comfy candidate: unsupported route",
            Self::RouteChanged => "Comfy candidate: route changed",
            Self::BindingChanged => "Comfy candidate: runtime binding changed",
            Self::JobAssociationMismatch => "Comfy candidate: job association mismatch",
            Self::ServiceGenerationMismatch => "Comfy candidate: service generation mismatch",
            Self::CandidateRejected => "Comfy candidate: candidate rejected",
            Self::PersistenceFailed => "Comfy candidate: persistence failed",
        })
    }
}

impl std::error::Error for ComfyCandidateError {}

/// Independent opt-in profile sets; Comfy workflows never inherit the local
/// script/decision allowance or vice versa.
#[cfg(target_os = "linux")]
pub(crate) struct ManagedStackProfiles {
    pub(crate) local_jobs: BTreeSet<ProfileKey>,
    pub(crate) comfy: BTreeSet<ProfileKey>,
}

/// One coordinator's bounded queue. Only the daemon may own this core
/// and write its history; multiple CLI/TUI processes must use IPC rather than
/// create competing instances against the same store.
pub struct CoordinatorCore {
    catalog: Catalog,
    bindings: BTreeMap<RuntimeIdentity, ResourceClaim>,
    history: JobHistory,
    data_root: PathBuf,
    pending: VecDeque<Pending>,
    active: BTreeMap<u64, Active>,
    capacity: usize,
    admission: AdmissionState,
    runtime_admission_closures: BTreeMap<RuntimeIdentity, u64>,
    next_runtime_close_generation: u64,
    admission_owner_identity: Arc<()>,
    dispatch_permit_owner_identity: Arc<()>,
    cpu_decisions_only: bool,
    managed_language_model_tasks: Option<BTreeMap<String, BTreeSet<String>>>,
    managed_local_job_profiles: Option<BTreeSet<ProfileKey>>,
    managed_comfy_profiles: Option<BTreeSet<ProfileKey>>,
    managed_stack_gpu_index: Option<u32>,
}

impl CoordinatorCore {
    /// Loads existing history, which reconciles queued/running rows as
    /// interrupted without replay. Runtime reconciliation is still required.
    pub fn load(
        catalog: Catalog,
        bindings: BTreeMap<RuntimeIdentity, ResourceClaim>,
        data_root: impl AsRef<Path>,
        repository_root: &Path,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_CAPACITY).contains(&capacity),
            "queue capacity must be 1–{MAX_CAPACITY}"
        );
        let (history, notice) = JobHistory::load_in_strict(data_root.as_ref(), repository_root)?;
        ensure!(
            notice.normalization_error.is_none(),
            "could not persist restart reconciliation"
        );
        Ok(Self {
            catalog,
            bindings,
            history,
            data_root: data_root.as_ref().to_path_buf(),
            pending: VecDeque::new(),
            active: BTreeMap::new(),
            capacity,
            admission: AdmissionState::ReconciliationRequired,
            runtime_admission_closures: BTreeMap::new(),
            next_runtime_close_generation: 1,
            admission_owner_identity: Arc::new(()),
            dispatch_permit_owner_identity: Arc::new(()),
            cpu_decisions_only: false,
            managed_language_model_tasks: None,
            managed_local_job_profiles: None,
            managed_comfy_profiles: None,
            managed_stack_gpu_index: None,
        })
    }

    pub fn admission(&self) -> AdmissionState {
        self.admission
    }

    /// Close only new submissions routed to one bound runtime and return the
    /// current close-generation token. Calling this again for a closed runtime
    /// rotates the generation, making any earlier token stale. The close is
    /// local to this coordinator instance and does not change accepted work,
    /// history, queue capacity, or global reconciliation state.
    pub fn close_runtime_admission(
        &mut self,
        runtime: &RuntimeIdentity,
    ) -> std::result::Result<RuntimeAdmissionClose, RuntimeAdmissionError> {
        if !self.bindings.contains_key(runtime) {
            return Err(RuntimeAdmissionError::UnboundRuntime);
        }
        let generation = self.next_runtime_close_generation;
        let next_generation = generation
            .checked_add(1)
            .ok_or(RuntimeAdmissionError::GenerationExhausted)?;
        self.next_runtime_close_generation = next_generation;
        self.runtime_admission_closures
            .insert(runtime.clone(), generation);
        Ok(RuntimeAdmissionClose {
            runtime: runtime.clone(),
            generation,
            owner_identity: Arc::clone(&self.admission_owner_identity),
        })
    }

    /// Reopen submissions for a runtime only with its latest close token. The
    /// caller must first establish that this exact runtime is healthy; token
    /// matching proves only that this close generation is current.
    pub fn reopen_runtime_admission(
        &mut self,
        close: RuntimeAdmissionClose,
    ) -> std::result::Result<(), RuntimeAdmissionError> {
        if !self.is_current_runtime_close(&close) {
            return Err(RuntimeAdmissionError::StaleCloseToken);
        }
        self.runtime_admission_closures.remove(&close.runtime);
        Ok(())
    }

    fn is_current_runtime_close(&self, close: &RuntimeAdmissionClose) -> bool {
        Arc::ptr_eq(&self.admission_owner_identity, &close.owner_identity)
            && self.runtime_admission_closures.get(&close.runtime) == Some(&close.generation)
    }

    /// Report accepted queued/active work for one bound runtime, plus all
    /// unresolved retained history (its original runtime is not persisted).
    /// Active includes reservations awaiting cleanup
    /// acknowledgment. Unknown or interrupted retained jobs prevent a drained
    /// result. This is not a native-backend or resource-release probe, and it
    /// does not itself reserve work. Global `reserve_next` may select other-
    /// runtime FIFO work first; use [`Self::reserve_next_for_runtime`] or a
    /// higher-level admission policy when this runtime needs a targeted drain.
    pub fn runtime_drain_state(
        &self,
        runtime: &RuntimeIdentity,
    ) -> std::result::Result<RuntimeDrainState, RuntimeAdmissionError> {
        if !self.bindings.contains_key(runtime) {
            return Err(RuntimeAdmissionError::UnboundRuntime);
        }
        let tracked_jobs = self
            .pending
            .iter()
            .map(|pending| pending.job_id)
            .chain(self.active.keys().copied())
            .collect::<std::collections::BTreeSet<_>>();
        Ok(RuntimeDrainState {
            pending_jobs: self
                .pending
                .iter()
                .filter(|pending| pending.route.runtime == *runtime)
                .count(),
            active_jobs: self
                .active
                .values()
                .filter(|active| active.route.runtime == *runtime)
                .count(),
            unresolved_retained_jobs: self
                .history
                .records()
                .iter()
                .filter(|job| !tracked_jobs.contains(&job.id))
                .filter(|job| retained_job_may_be_unresolved(job))
                .count(),
            admission_closed: self.runtime_admission_closures.contains_key(runtime),
        })
    }

    pub fn queued_count(&self) -> usize {
        self.pending.len()
    }
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
    pub fn history(&self) -> &JobHistory {
        &self.history
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    fn managed_request_allowed(
        &self,
        catalog: &Catalog,
        route: &ResolvedProfileRoute,
        request: &InferenceRequest,
        resource: ResourceClaim,
    ) -> bool {
        let no_comfy_profiles = BTreeSet::new();
        if let Some(allowed_local_profiles) = self.managed_local_job_profiles.as_ref() {
            self.managed_stack_gpu_index
                .zip(self.managed_language_model_tasks.as_ref())
                .is_some_and(|(gpu, supported)| {
                    validate_managed_stack_request(
                        catalog,
                        route,
                        request,
                        supported,
                        allowed_local_profiles,
                        self.managed_comfy_profiles
                            .as_ref()
                            .unwrap_or(&no_comfy_profiles),
                        gpu,
                    )
                    .is_ok_and(|expected| expected == resource)
                })
        } else {
            !self
                .managed_language_model_tasks
                .as_ref()
                .is_some_and(|supported| {
                    let NativeRoute::LlamaSwapModel { model_id, .. } = &route.native_route else {
                        return true;
                    };
                    !supported
                        .get(model_id.as_str())
                        .is_some_and(|tasks| tasks.contains(request.task()))
                })
        }
    }

    /// Validate the explicit startup path before its managed worker touches
    /// retained native ownership. Generic `confirm_reconciled` stays a
    /// compatibility boundary; callers that opt into a concrete executor use
    /// this stricter gate and later provide that executor's exact readiness
    /// observation.
    #[cfg(target_os = "linux")]
    pub(crate) fn require_managed_language_startup(
        &self,
        runtime: &RuntimeIdentity,
        expected_gpu: u32,
    ) -> Result<()> {
        ensure!(
            self.admission == AdmissionState::ReconciliationRequired,
            "managed language startup requires unresolved startup admission"
        );
        ensure!(
            !self.cpu_decisions_only && self.active.is_empty() && self.pending.is_empty(),
            "managed language startup requires empty reservations"
        );
        ensure!(
            runtime.lane == Lane::Language
                && runtime.as_str() == "llama-swap"
                && self.bindings.len() == 1
                && matches!(
                    self.bindings.get(runtime),
                    Some(ResourceClaim::ExclusiveGpu(gpu)) if *gpu == expected_gpu
                ),
            "managed language startup must bind its configured exclusive GPU"
        );
        Ok(())
    }

    /// Opt into child-owned CPU local decisions only. The daemon must retain
    /// host ownership and use the bounded CPU worker, which disables CUDA and
    /// never downloads models. This is not reconciliation for resident engines.
    /// Any interrupted or foreign retained history refuses this startup path.
    pub fn confirm_cpu_decisions_ready(&mut self) -> Result<()> {
        ensure!(
            self.active.is_empty() && self.pending.is_empty(),
            "reservations require reconciliation"
        );
        ensure!(
            !self.bindings.is_empty(),
            "CPU decision bindings are absent"
        );
        let refreshed = self.catalog.refresh()?;
        for (runtime, resource) in &self.bindings {
            ensure!(
                runtime.lane == Lane::Decisions
                    && matches!(runtime.as_str(), "nagi" | "laya")
                    && *resource == ResourceClaim::None,
                "CPU decision bindings must exclude GPU and hosted engines"
            );
            ensure!(
                refreshed.list().iter().any(|profile| {
                    profile.runtime == *runtime
                        && !profile.disabled
                        && resolve_profile_route(&refreshed, &profile.key)
                            .is_ok_and(|route| cpu_decision_route(&route))
                }),
                "bound runtime has no explicit CPU local decision profile"
            );
        }
        ensure!(
            self.history.records().iter().all(|job| {
                job.kind == "cross-toolkit"
                    && job.metadata.toolkit.as_deref() == Some("l3ds")
                    && matches!(
                        job.metadata.result_state.as_deref(),
                        Some("succeeded" | "failed" | "cancelled")
                    )
            }),
            "retained history requires explicit native cleanup reconciliation"
        );
        self.catalog = refreshed;
        self.cpu_decisions_only = true;
        self.admission = AdmissionState::Ready;
        Ok(())
    }

    /// Complete first startup reconciliation for the concrete managed
    /// llama-swap language session. The caller must hold the listener's host
    /// guard and obtain `readiness` from the one-shot activation followed by
    /// read-only owner/router reconciliation; this method validates that the
    /// observation names a currently declared language model before opening
    /// global admission. It deliberately does not make CPU-only readiness a
    /// substitute for resident-engine ownership.
    #[cfg(target_os = "linux")]
    pub(crate) fn confirm_managed_language_ready(
        &mut self,
        runtime: &RuntimeIdentity,
        expected_gpu: u32,
        startup_profile: &ProfileKey,
        supported_model_tasks: BTreeMap<String, BTreeSet<String>>,
        readiness: &super::engine_health::RouterModelReadinessObservation,
    ) -> Result<()> {
        ensure!(
            self.admission == AdmissionState::ReconciliationRequired,
            "coordinator readiness was already resolved"
        );
        ensure!(
            !self.cpu_decisions_only && self.active.is_empty() && self.pending.is_empty(),
            "managed language readiness requires empty reconciled reservations"
        );
        ensure!(
            runtime.lane == Lane::Language
                && runtime.as_str() == "llama-swap"
                && self.bindings.len() == 1
                && matches!(
                    self.bindings.get(runtime),
                    Some(ResourceClaim::ExclusiveGpu(gpu)) if *gpu == expected_gpu
                ),
            "managed language session must bind one exclusive llama-swap runtime"
        );

        let refreshed = self.catalog.refresh()?;
        let route = resolve_profile_route(&refreshed, startup_profile)?;
        ensure!(
            !supported_model_tasks.is_empty(),
            "managed model list is empty"
        );
        for (model, tasks) in &supported_model_tasks {
            ensure!(
                super::catalog::ProfileId::new(model.clone()).is_ok()
                    && !tasks.is_empty()
                    && tasks
                        .iter()
                        .all(|task| matches!(task.as_str(), "chat" | "embeddings")),
                "managed model/task declaration is invalid"
            );
        }
        ensure!(
            startup_profile.lane == Lane::Language
                && route.runtime == *runtime
                && matches!(
                    &route.native_route,
                    NativeRoute::LlamaSwapModel { model_id, .. }
                        if readiness.matches_model(model_id.as_str())
                            && supported_model_tasks.get(model_id.as_str()).is_some_and(|tasks|
                                route.tasks.iter().all(|task| tasks.contains(task.as_str())))
                ),
            "native readiness does not match the configured language profile"
        );
        ensure!(
            refreshed
                .list()
                .iter()
                .filter(|profile| profile.runtime == *runtime)
                .all(|profile| {
                    resolve_profile_route(&refreshed, &profile.key).is_ok_and(|candidate| {
                        matches!(candidate.native_route, NativeRoute::LlamaSwapModel { .. })
                    })
                }),
            "bound language runtime contains an unsupported native route"
        );
        ensure!(
            self.history
                .records()
                .iter()
                .all(|job| !retained_job_may_be_unresolved(job)),
            "retained history still requires native cleanup reconciliation"
        );

        self.catalog = refreshed;
        self.managed_language_model_tasks = Some(supported_model_tasks);
        self.admission = AdmissionState::Ready;
        Ok(())
    }

    /// Validate the configured full managed stack before a worker begins
    /// reconciliation. Unlike the language-only gate above, this accepts an
    /// exact set of local Breeze and CPU/CUDA Nagi/Laya routes, with each
    /// runtime bound to the resource declared by its profile. Entrypoints are
    /// checked through the inert native-command preparer; no script is run.
    #[cfg(target_os = "linux")]
    pub(crate) fn require_managed_stack_startup(
        &self,
        runtime: &RuntimeIdentity,
        expected_gpu: u32,
        startup_profile: &ProfileKey,
        supported_model_tasks: &BTreeMap<String, BTreeSet<String>>,
        allowed_local_profiles: &BTreeSet<ProfileKey>,
        allowed_comfy_profiles: &BTreeSet<ProfileKey>,
    ) -> Result<()> {
        ensure!(
            self.admission == AdmissionState::ReconciliationRequired,
            "managed stack startup requires unresolved startup admission"
        );
        ensure!(
            !self.cpu_decisions_only && self.active.is_empty() && self.pending.is_empty(),
            "managed stack startup requires empty reservations"
        );
        let (_, expected_bindings) = validate_managed_stack_configuration(
            &self.catalog,
            runtime,
            expected_gpu,
            startup_profile,
            supported_model_tasks,
            allowed_local_profiles,
            allowed_comfy_profiles,
        )?;
        ensure!(
            self.bindings == expected_bindings,
            "managed stack runtime bindings do not match the configured routes"
        );
        Ok(())
    }

    /// Open admission only after the configured stack has been freshly
    /// validated, the startup language model has a real router observation,
    /// and retained history contains no unresolved ownership. This remains
    /// separate from generic reconciliation and from language-only readiness.
    #[cfg(target_os = "linux")]
    pub(crate) fn confirm_managed_stack_ready(
        &mut self,
        runtime: &RuntimeIdentity,
        expected_gpu: u32,
        startup_profile: &ProfileKey,
        supported_model_tasks: BTreeMap<String, BTreeSet<String>>,
        profiles: ManagedStackProfiles,
        readiness: &super::engine_health::RouterModelReadinessObservation,
    ) -> Result<()> {
        let ManagedStackProfiles {
            local_jobs: allowed_local_profiles,
            comfy: allowed_comfy_profiles,
        } = profiles;
        ensure!(
            self.admission == AdmissionState::ReconciliationRequired,
            "coordinator readiness was already resolved"
        );
        ensure!(
            !self.cpu_decisions_only && self.active.is_empty() && self.pending.is_empty(),
            "managed stack readiness requires empty reconciled reservations"
        );
        let (refreshed, expected_bindings) = validate_managed_stack_configuration(
            &self.catalog,
            runtime,
            expected_gpu,
            startup_profile,
            &supported_model_tasks,
            &allowed_local_profiles,
            &allowed_comfy_profiles,
        )?;
        ensure!(
            self.bindings == expected_bindings,
            "managed stack runtime bindings do not match the configured routes"
        );
        let startup_route = resolve_profile_route(&refreshed, startup_profile)?;
        ensure!(
            matches!(
                &startup_route.native_route,
                NativeRoute::LlamaSwapModel { model_id, .. }
                    if readiness.matches_model(model_id.as_str())
            ),
            "router readiness does not match the configured startup model"
        );
        ensure!(
            self.history
                .records()
                .iter()
                .all(|job| !retained_job_may_be_unresolved(job)),
            "retained history still requires native cleanup reconciliation"
        );

        self.catalog = refreshed;
        self.managed_language_model_tasks = Some(supported_model_tasks);
        self.managed_local_job_profiles = Some(allowed_local_profiles);
        self.managed_comfy_profiles = Some(allowed_comfy_profiles);
        self.managed_stack_gpu_index = Some(expected_gpu);
        self.admission = AdmissionState::Ready;
        Ok(())
    }

    /// Reconnect lookup uses the caller's retained request ID, never a numeric
    /// history ID (which can be rehydrated differently after restart).
    pub fn request_state(&self, request_id: &str) -> Option<&str> {
        self.history
            .records()
            .iter()
            .find(|job| job.metadata.request_id.as_deref() == Some(request_id))
            .and_then(|job| job.metadata.result_state.as_deref())
    }

    /// Typed authoritative status from retained durable history. Result-buffer
    /// availability is deliberately not consulted: an empty or restarted
    /// cache cannot turn a retained request into `not_found`.
    pub(crate) fn authoritative_request_status(&self, request_id: &str) -> Option<RequestStatus> {
        self.wire_state(request_id)
    }

    /// Test-only durable fixture; it does not open admission or create an
    /// executor reservation. Used to exercise read-only IPC against retained
    /// history states that a real executor would persist.
    #[cfg(test)]
    pub(crate) fn seed_test_request_status(&mut self, request_id: &str, state: &str) -> Result<()> {
        ensure!(
            matches!(state, "queued" | "succeeded"),
            "test history state is unsupported"
        );
        let mut candidate = self.history.clone();
        let job_id = candidate.begin_cross_toolkit(CrossToolkitJobStart {
            toolkit: "l3ms".into(),
            profile: Some("test-profile".into()),
            task: Some("test-task".into()),
            request_id: Some(request_id.into()),
        })?;
        if state != "queued" {
            candidate.update_cross_toolkit(
                job_id,
                CrossToolkitJobUpdate {
                    result_state: Some(state.into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )?;
        }
        candidate.persist_in(&self.data_root)?;
        self.history = candidate;
        Ok(())
    }

    /// Cancels only retained, nonterminal work. A terminal row is not rewritten
    /// as cancelled, and an unknown request cannot target another numeric job.
    pub fn cancel_request(&mut self, request_id: &str) -> Result<()> {
        let job = self
            .history
            .records()
            .iter()
            .find(|job| job.metadata.request_id.as_deref() == Some(request_id))
            .context("unknown retained request")?;
        ensure!(
            matches!(
                job.metadata.result_state.as_deref(),
                Some("queued" | "running")
            ),
            "request is already terminal"
        );
        self.cancel(job.id)
    }

    /// Called only after the host executor has reconciled all retained engine
    /// jobs, runtime health, admission and resource ownership. This does not
    /// perform those probes. Per-runtime close state survives quarantine in
    /// this core but is not restored after process restart; restart begins
    /// globally unavailable until this reconciliation is completed.
    pub fn confirm_reconciled(&mut self) -> Result<()> {
        ensure!(
            self.active.is_empty(),
            "active reservations still require reconciliation"
        );
        self.dispatch_permit_owner_identity = Arc::new(());
        self.admission = AdmissionState::Ready;
        Ok(())
    }

    pub fn quarantine(&mut self) {
        // Outstanding pre-POST capabilities must not survive quarantine even
        // if the caller still holds its original request or route.
        self.dispatch_permit_owner_identity = Arc::new(());
        self.admission = AdmissionState::ReconciliationRequired;
    }

    /// IPC control operations only: never reserves work or starts an engine.
    /// Submit acceptance follows durable persistence, not merely validation.
    pub fn handle_client_request(
        &mut self,
        request: ClientRequest,
        now: Instant,
    ) -> ResponseEnvelope {
        if ipc::validate_request_id(request.request_id()).is_err() {
            return ResponseEnvelope::v1(ResponseMessage::Error {
                request_id: None,
                code: ResponseErrorCode::InvalidRequest,
            });
        }
        let message = match request {
            ClientRequest::Submit {
                request_id,
                request,
                timeout,
            } => match self.submit_checked(request_id.clone(), request, timeout, now) {
                Ok(_) => ResponseMessage::Accepted {
                    request_id,
                    operation: AcceptedOperation::Submit,
                    state: RequestStatus::Queued,
                },
                Err(error) => wire_error(
                    request_id,
                    match error {
                        SubmissionError::Unavailable | SubmissionError::RuntimeAdmissionClosed => {
                            ResponseErrorCode::Unavailable
                        }
                        SubmissionError::QueueFull => ResponseErrorCode::QueueFull,
                        SubmissionError::DuplicateRequest => ResponseErrorCode::DuplicateRequest,
                        SubmissionError::InvalidRequest => ResponseErrorCode::InvalidRequest,
                        SubmissionError::Internal => ResponseErrorCode::Internal,
                    },
                ),
            },
            ClientRequest::Status { request_id } => match self.wire_state(&request_id) {
                Some(state) => ResponseMessage::Status { request_id, state },
                None => wire_error(request_id, ResponseErrorCode::Internal),
            },
            ClientRequest::Cancel { request_id } => match self.wire_state(&request_id) {
                Some(RequestStatus::Queued | RequestStatus::Running) => {
                    if self.cancel_request(&request_id).is_err() {
                        self.quarantine();
                        wire_error(request_id, ResponseErrorCode::Internal)
                    } else {
                        match self.wire_state(&request_id) {
                            Some(state) => ResponseMessage::Accepted {
                                request_id,
                                operation: AcceptedOperation::Cancel,
                                state,
                            },
                            None => wire_error(request_id, ResponseErrorCode::Internal),
                        }
                    }
                }
                Some(_) => wire_error(request_id, ResponseErrorCode::InvalidRequest),
                None => wire_error(request_id, ResponseErrorCode::Internal),
            },
        };
        ResponseEnvelope::v1(message)
    }

    fn wire_state(&self, request_id: &str) -> Option<RequestStatus> {
        let Some(job) = self
            .history
            .records()
            .iter()
            .find(|job| job.metadata.request_id.as_deref() == Some(request_id))
        else {
            return Some(RequestStatus::NotFound);
        };
        Some(match job.metadata.result_state.as_deref() {
            Some("queued") => RequestStatus::Queued,
            Some("running") => RequestStatus::Running,
            Some("succeeded") => RequestStatus::Succeeded,
            Some("failed") => RequestStatus::Failed,
            Some("cancelled") => RequestStatus::Cancelled,
            Some("interrupted") => RequestStatus::Interrupted,
            None => return None,
            Some(_) => return None,
        })
    }

    /// One authenticated socket exchange for a future bounded daemon loop.
    /// Handler persistence is synchronous trusted-local-filesystem work; it
    /// cannot be preempted. If acknowledgment IO fails after acceptance, keep
    /// accepted work and use retained request-ID status to reconcile, not replay.
    #[cfg(target_os = "linux")]
    pub fn serve_connection(
        &mut self,
        connection: &mut super::local_ipc::LocalConnection,
        context: &super::runtime::OperationContext<'_>,
    ) -> Result<(), super::local_ipc::LocalIpcError> {
        use super::local_ipc::LocalIpcError;
        let request = connection.read_request(context)?;
        if context.is_cancelled() {
            return Err(LocalIpcError::Cancelled);
        }
        if context.remaining().is_zero() {
            return Err(LocalIpcError::Deadline);
        }
        let response = self.handle_client_request(request, Instant::now());
        connection.write_response(&response, context)
    }

    pub fn submit(
        &mut self,
        request_id: String,
        request: InferenceRequest,
        timeout: Duration,
        now: Instant,
    ) -> Result<u64> {
        self.submit_checked(request_id, request, timeout, now)
            .map_err(anyhow::Error::new)
    }

    pub fn submit_checked(
        &mut self,
        request_id: String,
        request: InferenceRequest,
        timeout: Duration,
        now: Instant,
    ) -> std::result::Result<u64, SubmissionError> {
        if self.admission != AdmissionState::Ready {
            return Err(SubmissionError::Unavailable);
        }
        if timeout.is_zero()
            || timeout > MAX_TIMEOUT
            || ipc::validate_request_id(&request_id).is_err()
        {
            return Err(SubmissionError::InvalidRequest);
        }
        if self
            .history
            .records()
            .iter()
            .any(|job| job.metadata.request_id.as_ref() == Some(&request_id))
        {
            return Err(SubmissionError::DuplicateRequest);
        }
        let profile = request
            .validate(&self.catalog)
            .map_err(|_| SubmissionError::InvalidRequest)?;
        let route = resolve_profile_route(&self.catalog, &profile.key)
            .map_err(|_| SubmissionError::InvalidRequest)?;
        let resource = *self
            .bindings
            .get(&route.runtime)
            .ok_or(SubmissionError::Unavailable)?;
        if !self.managed_request_allowed(&self.catalog, &route, &request, resource) {
            return Err(SubmissionError::Unavailable);
        }
        if self.runtime_admission_closures.contains_key(&route.runtime) {
            return Err(SubmissionError::RuntimeAdmissionClosed);
        }
        if self.pending.len() + self.active.len() >= self.capacity {
            return Err(SubmissionError::QueueFull);
        }
        if self.cpu_decisions_only
            && (!cpu_decision_route(&route) || resource != ResourceClaim::None)
        {
            return Err(SubmissionError::Unavailable);
        }
        let deadline = now
            .checked_add(timeout)
            .ok_or(SubmissionError::InvalidRequest)?;
        let mut candidate = self.history.clone();
        let job_id = candidate
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: toolkit_for_lane(profile.key.lane).into(),
                profile: Some(profile.key.profile_id.to_string()),
                task: Some(request.task().into()),
                request_id: Some(request_id.clone()),
            })
            .map_err(|_| SubmissionError::InvalidRequest)?;
        if !self
            .pending
            .iter()
            .map(|p| p.job_id)
            .chain(self.active.keys().copied())
            .all(|id| candidate.records().iter().any(|job| job.id == id))
        {
            return Err(SubmissionError::QueueFull);
        }
        if candidate.persist_in(&self.data_root).is_err() {
            self.quarantine();
            return Err(SubmissionError::Internal);
        }
        self.history = candidate;
        self.pending.push_back(Pending {
            job_id,
            request_id,
            request,
            route,
            resource,
            submitted: now,
            deadline,
            cancellation: RequestCancellation::new(),
        });
        Ok(job_id)
    }

    /// Cancelled queued work is durably terminal before it leaves the queue.
    /// Active cancellation signals the executor but retains the reservation
    /// until cleanup/completion is acknowledged.
    pub fn cancel(&mut self, job_id: u64) -> Result<()> {
        if let Some(index) = self.pending.iter().position(|p| p.job_id == job_id) {
            if let Err(error) = self.persist_update(job_id, "cancelled", None, None) {
                self.quarantine();
                return Err(error);
            }
            self.pending.remove(index);
            return Ok(());
        }
        let active = self
            .active
            .get(&job_id)
            .context("unknown active or queued job")?;
        active.cancellation.cancel();
        Ok(())
    }

    /// FIFO among conflicting resources; CPU/hosted work may bypass a GPU
    /// request blocked by active work. Reserved streams hold their place until
    /// the executor acknowledges actual completion.
    pub fn reserve_next(&mut self, now: Instant) -> Result<Option<DispatchTicket>> {
        self.reserve_matching_runtime(None, now)
    }

    /// Reserve accepted work for exactly the runtime represented by a current
    /// close token. The borrowed token remains current for additional drain
    /// reservations until admission is reopened or a newer close replaces it.
    /// This still only reserves coordinator work; it does not establish engine
    /// health, dispatch permission, or resource release.
    pub fn reserve_next_for_runtime(
        &mut self,
        close: &RuntimeAdmissionClose,
        now: Instant,
    ) -> Result<Option<DispatchTicket>> {
        ensure!(
            self.is_current_runtime_close(close),
            "runtime admission close token is stale"
        );
        self.reserve_matching_runtime(Some(&close.runtime), now)
    }

    fn reserve_matching_runtime(
        &mut self,
        runtime: Option<&RuntimeIdentity>,
        now: Instant,
    ) -> Result<Option<DispatchTicket>> {
        ensure!(
            self.admission == AdmissionState::Ready,
            "runtime reconciliation is required"
        );
        let mut index = 0;
        while index < self.pending.len() {
            let pending = &self.pending[index];
            let selected = runtime.is_none_or(|runtime| pending.route.runtime == *runtime);
            if selected && (now >= pending.deadline || pending.cancellation.is_cancelled()) {
                let job_id = pending.job_id;
                let state = if pending.cancellation.is_cancelled() {
                    "cancelled"
                } else {
                    "failed"
                };
                if let Err(error) = self.persist_update(job_id, state, None, None) {
                    self.quarantine();
                    return Err(error);
                }
                self.pending.remove(index);
            } else {
                index += 1;
            }
        }
        let Some(index) = self.pending.iter().position(|pending| {
            runtime.is_none_or(|runtime| pending.route.runtime == *runtime)
                && !self
                    .active
                    .values()
                    .any(|active| resources_conflict(pending.resource, active.resource))
        }) else {
            return Ok(None);
        };
        let pending = self
            .pending
            .remove(index)
            .expect("selected queue index exists");
        let Pending {
            job_id,
            request_id,
            request,
            route,
            resource,
            submitted,
            deadline,
            cancellation,
        } = pending;
        let queue_wait = now.saturating_duration_since(submitted);
        let accepted_request = request.clone();
        let profile = request.profile().clone();
        let task = request.task().to_owned();
        self.active.insert(
            job_id,
            Active {
                permit_reservation_identity: Arc::new(()),
                request_id: request_id.clone(),
                request: accepted_request,
                profile,
                task,
                route: route.clone(),
                resource,
                cancellation: cancellation.clone(),
                deadline,
                reserved: now,
                cleanup_unresolved: false,
                dispatch_validated: false,
                started: None,
                queue_wait,
            },
        );
        Ok(Some(DispatchTicket {
            job_id,
            request_id,
            request,
            route,
            resource,
            deadline,
            cancellation,
            queue_wait,
        }))
    }

    /// Revalidate the accepted request and native route immediately before
    /// recording the durable running transition. Any stale or inconsistent
    /// association closes admission and leaves the active resource reservation
    /// in place for reconciliation; this method never dispatches native work.
    pub fn begin_dispatch(
        &mut self,
        ticket: &DispatchTicket,
        now: Instant,
    ) -> std::result::Result<(), DispatchBeginError> {
        if self.admission != AdmissionState::Ready {
            return Err(DispatchBeginError::AdmissionUnavailable);
        }

        let Some(active) = self.active.get(&ticket.job_id) else {
            self.quarantine();
            return Err(DispatchBeginError::ReservationUnavailable);
        };
        if active.started.is_some() {
            return Err(DispatchBeginError::AlreadyStarted);
        }
        if active.cancellation.is_cancelled() {
            return Err(DispatchBeginError::Cancelled);
        }
        if now >= active.deadline {
            return Err(DispatchBeginError::DeadlineExceeded);
        }

        let ticket_matches_reservation = ticket.request_id == active.request_id
            && same_request(&ticket.request, &active.request)
            && ticket.request.profile() == &active.profile
            && ticket.request.task() == active.task
            && ticket.route == active.route
            && ticket.resource == active.resource
            && ticket.deadline == active.deadline
            && ticket.queue_wait == active.queue_wait
            && ticket.cancellation.same_token_as(&active.cancellation);
        if !ticket_matches_reservation {
            self.quarantine();
            return Err(DispatchBeginError::TicketMismatch);
        }

        let refreshed = match self.catalog.refresh() {
            Ok(catalog) => catalog,
            Err(_) => {
                self.quarantine();
                return Err(DispatchBeginError::CatalogUnavailable);
            }
        };
        // Retain the latest valid catalog even when this ticket is stale, so
        // later submissions cannot continue using an older manifest snapshot.
        self.catalog = refreshed;

        if ticket.request.validate(&self.catalog).is_err() {
            self.quarantine();
            return Err(DispatchBeginError::InvalidRequest);
        }
        let current_route = match resolve_profile_route(&self.catalog, ticket.request.profile()) {
            Ok(route) => route,
            Err(_) => {
                self.quarantine();
                return Err(DispatchBeginError::RouteChanged);
            }
        };
        if !self.managed_request_allowed(
            &self.catalog,
            &current_route,
            &ticket.request,
            ticket.resource,
        ) {
            self.quarantine();
            return Err(DispatchBeginError::RouteChanged);
        }
        if self.cpu_decisions_only
            && (!cpu_decision_route(&current_route) || ticket.resource != ResourceClaim::None)
        {
            self.quarantine();
            return Err(DispatchBeginError::RouteChanged);
        }
        let accepted_route = self
            .active
            .get(&ticket.job_id)
            .map(|active| active.route.clone())
            .ok_or_else(|| {
                self.quarantine();
                DispatchBeginError::ReservationUnavailable
            })?;
        if current_route != accepted_route || current_route != ticket.route {
            self.quarantine();
            return Err(DispatchBeginError::RouteChanged);
        }

        let binding_matches = self
            .bindings
            .get(&current_route.runtime)
            .is_some_and(|resource| *resource == ticket.resource);
        let active_resource_matches = self
            .active
            .get(&ticket.job_id)
            .is_some_and(|active| active.resource == ticket.resource);
        if !binding_matches || !active_resource_matches {
            self.quarantine();
            return Err(DispatchBeginError::BindingChanged);
        }

        let expected_toolkit = toolkit_for_lane(ticket.request.lane());
        let profile_id = ticket.request.profile().profile_id.as_str();
        let request_id = ticket.request_id.as_str();
        let task = ticket.request.task();
        let association_matches = self
            .history
            .records()
            .iter()
            .find(|job| job.id == ticket.job_id)
            .is_some_and(|job| {
                job.kind == "cross-toolkit"
                    && job.status == "queued"
                    && job.metadata.toolkit.as_deref() == Some(expected_toolkit)
                    && job.metadata.profile.as_deref() == Some(profile_id)
                    && job.metadata.task.as_deref() == Some(task)
                    && job.metadata.request_id.as_deref() == Some(request_id)
                    && job.metadata.result_state.as_deref() == Some("queued")
            });
        if !association_matches {
            self.quarantine();
            return Err(DispatchBeginError::JobAssociationMismatch);
        }

        // Keep the caller-supplied timestamp for deterministic admission checks,
        // but never let catalog IO make that timestamp stale at the commit.
        let commit_time = now.max(Instant::now());
        match self.mark_running_checked(ticket.job_id, commit_time) {
            Ok(()) => {
                self.active
                    .get_mut(&ticket.job_id)
                    .expect("validated active reservation")
                    .dispatch_validated = true;
                Ok(())
            }
            Err(DispatchBeginError::PersistenceFailed) => {
                self.quarantine();
                Err(DispatchBeginError::PersistenceFailed)
            }
            Err(error) => Err(error),
        }
    }

    /// Durably allocate the one-shot native Comfy prompt ID after the final
    /// dispatch gate and immediately before the caller's single POST attempt.
    /// This records intent only; it does not acknowledge native acceptance.
    pub fn record_comfy_dispatch_candidate(
        &mut self,
        ticket: &DispatchTicket,
        candidate_id: String,
        now: Instant,
    ) -> std::result::Result<ComfyDispatchPermit, ComfyCandidateError> {
        self.record_comfy_dispatch_candidate_inner(ticket, candidate_id, None, now)
    }

    /// Managed Comfy dispatch requires an exact owner-issued generation. Its
    /// payload-free comparison reference is committed atomically with the
    /// candidate before this non-restorable permit is returned.
    #[cfg(target_os = "linux")]
    pub(crate) fn record_managed_comfy_dispatch_candidate(
        &mut self,
        ticket: &DispatchTicket,
        candidate_id: String,
        generation_identity: OwnedServiceGenerationIdentity,
        now: Instant,
    ) -> std::result::Result<ManagedComfyDispatchPermit, ComfyCandidateError> {
        let generation_reference = match generation_identity.comfy_service_generation_reference() {
            Ok(reference) => reference,
            Err(_) => {
                self.quarantine();
                return Err(ComfyCandidateError::ServiceGenerationMismatch);
            }
        };
        let dispatch = self.record_comfy_dispatch_candidate_inner(
            ticket,
            candidate_id,
            Some(generation_reference.clone()),
            now,
        )?;
        let permit = ManagedComfyDispatchPermit {
            dispatch,
            generation_identity,
            generation_reference,
        };
        if !self.accepts_managed_comfy_dispatch_permit(&permit) {
            self.quarantine();
            return Err(ComfyCandidateError::JobAssociationMismatch);
        }
        Ok(permit)
    }

    fn record_comfy_dispatch_candidate_inner(
        &mut self,
        ticket: &DispatchTicket,
        candidate_id: String,
        generation_reference: Option<ComfyServiceGenerationReference>,
        now: Instant,
    ) -> std::result::Result<ComfyDispatchPermit, ComfyCandidateError> {
        if self.admission != AdmissionState::Ready {
            self.quarantine();
            return Err(ComfyCandidateError::AdmissionUnavailable);
        }
        let Some(active) = self.active.get(&ticket.job_id) else {
            self.quarantine();
            return Err(ComfyCandidateError::ReservationUnavailable);
        };
        if !active.dispatch_validated {
            self.quarantine();
            return Err(ComfyCandidateError::FinalDispatchGateMissing);
        }
        if active.started.is_none() {
            self.quarantine();
            return Err(ComfyCandidateError::FinalDispatchGateMissing);
        }
        if active.cancellation.is_cancelled() || ticket.cancellation.is_cancelled() {
            self.quarantine();
            return Err(ComfyCandidateError::Cancelled);
        }
        if now >= active.deadline {
            self.quarantine();
            return Err(ComfyCandidateError::DeadlineExceeded);
        }

        let ticket_matches_reservation = ticket.request_id == active.request_id
            && same_request(&ticket.request, &active.request)
            && ticket.request.profile() == &active.profile
            && ticket.request.task() == active.task
            && ticket.route == active.route
            && ticket.resource == active.resource
            && ticket.deadline == active.deadline
            && ticket.queue_wait == active.queue_wait
            && ticket.cancellation.same_token_as(&active.cancellation);
        if !ticket_matches_reservation {
            self.quarantine();
            return Err(ComfyCandidateError::TicketMismatch);
        }

        let request = ticket.request.clone();
        let request_id = ticket.request_id.clone();
        let accepted_route = active.route.clone();
        let resource = active.resource;
        let managed_generation = generation_reference.is_some();
        let is_comfy_workflow = matches!(
            &request,
            InferenceRequest::Media {
                input: MediaInput::Workflow { graph, .. },
                ..
            } if graph.as_object().is_some_and(|object| !object.is_empty())
        );
        if !is_comfy_workflow
            || accepted_route.profile.lane != Lane::Media
            || accepted_route.runtime.lane != Lane::Media
            || accepted_route.runtime.as_str() != "ComfyUI"
            || !matches!(
                &accepted_route.native_route,
                NativeRoute::MediaScript { .. }
            )
            || (managed_generation
                && (accepted_route.device != Some(Device::Cuda)
                    || !self
                        .managed_comfy_profiles
                        .as_ref()
                        .is_some_and(|profiles| profiles.contains(&accepted_route.profile))))
        {
            self.quarantine();
            return Err(ComfyCandidateError::UnsupportedRoute);
        }

        let refreshed = match self.catalog.refresh() {
            Ok(catalog) => catalog,
            Err(_) => {
                self.quarantine();
                return Err(ComfyCandidateError::CatalogUnavailable);
            }
        };
        self.catalog = refreshed;
        if request.validate(&self.catalog).is_err() {
            self.quarantine();
            return Err(ComfyCandidateError::RouteChanged);
        }
        let current_route = match resolve_profile_route(&self.catalog, request.profile()) {
            Ok(route) => route,
            Err(_) => {
                self.quarantine();
                return Err(ComfyCandidateError::RouteChanged);
            }
        };
        if current_route != accepted_route || current_route != ticket.route {
            self.quarantine();
            return Err(ComfyCandidateError::RouteChanged);
        }
        if !self
            .bindings
            .get(&current_route.runtime)
            .is_some_and(|binding| *binding == resource)
        {
            self.quarantine();
            return Err(ComfyCandidateError::BindingChanged);
        }

        let profile_id = request.profile().profile_id.as_str();
        let task = request.task();
        let association_matches = self
            .history
            .records()
            .iter()
            .find(|job| job.id == ticket.job_id)
            .is_some_and(|job| {
                job.kind == "cross-toolkit"
                    && job.status == "running"
                    && job.metadata.toolkit.as_deref() == Some("l2m2")
                    && job.metadata.profile.as_deref() == Some(profile_id)
                    && job.metadata.task.as_deref() == Some(task)
                    && job.metadata.request_id.as_deref() == Some(request_id.as_str())
                    && job.metadata.result_state.as_deref() == Some("running")
                    && job.metadata.backend_id.is_none()
                    && job.metadata.comfy_dispatch_candidate.is_none()
                    && job.metadata.comfy_service_generation.is_none()
            });
        if !association_matches {
            self.quarantine();
            return Err(ComfyCandidateError::JobAssociationMismatch);
        }
        let commit_time = now.max(Instant::now());
        if commit_time >= ticket.deadline {
            self.quarantine();
            return Err(ComfyCandidateError::DeadlineExceeded);
        }
        if ticket.cancellation.is_cancelled() {
            self.quarantine();
            return Err(ComfyCandidateError::Cancelled);
        }

        let mut candidate_history = self.history.clone();
        let history_result = match generation_reference.as_ref() {
            Some(generation) => candidate_history.record_managed_comfy_dispatch_candidate(
                ticket.job_id,
                candidate_id.clone(),
                generation.clone(),
            ),
            None => candidate_history
                .record_comfy_dispatch_candidate(ticket.job_id, candidate_id.clone()),
        };
        if history_result.is_err() {
            self.quarantine();
            return Err(ComfyCandidateError::CandidateRejected);
        }
        if candidate_history.persist_in(&self.data_root).is_err() {
            self.quarantine();
            return Err(ComfyCandidateError::PersistenceFailed);
        }
        self.history = candidate_history;

        let permit = ComfyDispatchPermit {
            owner_identity: Arc::downgrade(&self.dispatch_permit_owner_identity),
            reservation_identity: Arc::downgrade(
                &self
                    .active
                    .get(&ticket.job_id)
                    .expect("validated active reservation")
                    .permit_reservation_identity,
            ),
            cancellation: ticket.cancellation.clone(),
            deadline: ticket.deadline,
            job_id: ticket.job_id,
            request_id,
            request,
            candidate_id,
            route: accepted_route,
        };
        let permit_matches = match generation_reference.as_ref() {
            None => self.accepts_comfy_dispatch_permit(&permit),
            Some(reference) => {
                self.accepts_comfy_dispatch_permit_with_generation(&permit, Some(reference))
            }
        };
        if !permit_matches {
            self.quarantine();
            return Err(ComfyCandidateError::JobAssociationMismatch);
        }
        Ok(permit)
    }

    /// Check that a pre-POST permit still belongs to this live coordinator and
    /// to its unresolved reservation. Daemon handoffs call this before moving
    /// the permit into the Comfy adapter.
    pub(crate) fn accepts_comfy_dispatch_permit(&self, permit: &ComfyDispatchPermit) -> bool {
        self.accepts_comfy_dispatch_permit_with_generation(permit, None)
    }

    /// Managed equivalent: also requires the row's durable generation
    /// reference to match the transient owner identity carried by the permit.
    #[cfg(target_os = "linux")]
    pub(crate) fn accepts_managed_comfy_dispatch_permit(
        &self,
        permit: &ManagedComfyDispatchPermit,
    ) -> bool {
        permit.matches_generation(&permit.generation_identity)
            && self.accepts_comfy_dispatch_permit_with_generation(
                &permit.dispatch,
                Some(&permit.generation_reference),
            )
    }

    fn accepts_comfy_dispatch_permit_with_generation(
        &self,
        permit: &ComfyDispatchPermit,
        generation_reference: Option<&ComfyServiceGenerationReference>,
    ) -> bool {
        if self.admission != AdmissionState::Ready
            || !permit.is_live()
            || !permit.belongs_to(&self.dispatch_permit_owner_identity)
        {
            return false;
        }
        let now = Instant::now();
        let active_matches = self.active.get(&permit.job_id).is_some_and(|active| {
            active.dispatch_validated
                && active.started.is_some()
                && !active.cleanup_unresolved
                && !active.cancellation.is_cancelled()
                && now < active.deadline
                && active.request_id == permit.request_id
                && same_request(&active.request, &permit.request)
                && active.route == permit.route
        });
        let history_matches = self
            .history
            .records()
            .iter()
            .find(|job| job.id == permit.job_id)
            .is_some_and(|job| {
                job.metadata.request_id.as_deref() == Some(permit.request_id.as_str())
                    && job.metadata.comfy_dispatch_candidate.as_deref()
                        == Some(permit.candidate_id.as_str())
                    && job.metadata.backend_id.is_none()
                    && job.metadata.comfy_service_generation.as_ref() == generation_reference
            });
        active_matches && history_matches
    }

    /// Persist running state immediately before native inference begins.
    /// Kept for compatibility with existing core fixtures; native executors
    /// must use [`Self::begin_dispatch`] for current-route validation.
    pub fn mark_running(&mut self, job_id: u64, now: Instant) -> Result<()> {
        self.mark_running_checked(job_id, now)
            .map_err(|error| anyhow::anyhow!(error))
    }

    fn mark_running_checked(
        &mut self,
        job_id: u64,
        now: Instant,
    ) -> std::result::Result<(), DispatchBeginError> {
        if self.admission != AdmissionState::Ready {
            return Err(DispatchBeginError::AdmissionUnavailable);
        }
        let Some(active) = self.active.get(&job_id) else {
            return Err(DispatchBeginError::ReservationUnavailable);
        };
        if active.started.is_some() {
            return Err(DispatchBeginError::AlreadyStarted);
        }
        if active.cancellation.is_cancelled() {
            return Err(DispatchBeginError::Cancelled);
        }
        if now >= active.deadline {
            return Err(DispatchBeginError::DeadlineExceeded);
        }
        if self.persist_update(job_id, "running", None, None).is_err() {
            return Err(DispatchBeginError::PersistenceFailed);
        }
        self.active
            .get_mut(&job_id)
            .expect("active reservation exists")
            .started = Some(now);
        Ok(())
    }

    /// Native asynchronous jobs may attach their backend ID without releasing
    /// the reservation. Payloads, argv and credentials never enter history.
    pub fn attach_backend(&mut self, job_id: u64, backend_id: String) -> Result<()> {
        ensure!(
            self.active
                .get(&job_id)
                .is_some_and(|active| active.started.is_some()),
            "request has not started"
        );
        let candidate = self
            .history
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .and_then(|job| job.metadata.comfy_dispatch_candidate.as_deref());
        if candidate.is_some_and(|candidate| candidate != backend_id) {
            self.quarantine();
            return Err(anyhow::anyhow!(
                "verified backend ID does not match the persisted Comfy candidate; reconcile before more work"
            ));
        }
        if let Err(error) = self.persist_update(job_id, "running", Some(backend_id), None) {
            self.quarantine();
            return Err(error.context("backend ID persistence failed; reconcile before more work"));
        }
        Ok(())
    }

    /// Commit native Comfy acceptance evidence only when it still names the
    /// exact live managed workflow reservation and its durably associated
    /// owned-service generation. Cancellation and deadline expiry do not
    /// invalidate an already-observed native acknowledgement; the active
    /// reservation remains held until normal completion or cleanup.
    #[cfg(target_os = "linux")]
    pub(crate) fn accept_managed_comfy_backend_acknowledgement(
        &mut self,
        acknowledgement: super::comfy::ManagedComfyBackendAcknowledgement,
    ) -> Result<()> {
        let job_id = acknowledgement.job_id();
        let Some(active) = self.active.get(&job_id) else {
            self.quarantine();
            anyhow::bail!("managed Comfy acknowledgement has no active reservation")
        };
        let active_matches = active.started.is_some()
            && acknowledgement.request_id() == active.request_id
            && acknowledgement.matches_active(
                job_id,
                &active.request_id,
                &active.request,
                &active.route,
            )
            && matches!(
                &active.request,
                InferenceRequest::Media {
                    input: MediaInput::Workflow { graph, .. },
                    ..
                } if graph.as_object().is_some_and(|object| !object.is_empty())
            )
            && active.route.profile.lane == Lane::Media
            && active.route.runtime.lane == Lane::Media
            && active.route.runtime.as_str() == "ComfyUI"
            && active.route.device == Some(Device::Cuda)
            && self
                .managed_comfy_profiles
                .as_ref()
                .is_some_and(|profiles| profiles.contains(&active.route.profile));
        let active_request_id = active.request_id.clone();
        let active_profile_id = active.route.profile.profile_id.as_str().to_owned();
        let active_task = active.task.clone();
        if !active_matches {
            self.quarantine();
            anyhow::bail!("managed Comfy acknowledgement does not match its active reservation")
        }

        let candidate_id = acknowledgement.candidate_id();
        let backend_id = acknowledgement.backend_id();
        let generation = acknowledgement.generation_reference();
        let history_matches = self
            .history
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .is_some_and(|job| {
                job.kind == "cross-toolkit"
                    && job.status == "running"
                    && job.metadata.toolkit.as_deref() == Some("l2m2")
                    && job.metadata.profile.as_deref() == Some(active_profile_id.as_str())
                    && job.metadata.task.as_deref() == Some(active_task.as_str())
                    && job.metadata.request_id.as_deref() == Some(active_request_id.as_str())
                    && job.metadata.result_state.as_deref() == Some("running")
                    && job.metadata.comfy_dispatch_candidate.as_deref() == Some(candidate_id)
                    && candidate_id == backend_id
                    && generation.as_ref().is_some_and(|reference| {
                        reference.is_valid()
                            && job.metadata.comfy_service_generation.as_ref() == Some(reference)
                    })
            });
        if !history_matches {
            self.quarantine();
            anyhow::bail!("managed Comfy acknowledgement does not match durable candidate evidence")
        }

        let existing_backend_id = self
            .history
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .and_then(|job| job.metadata.backend_id.clone());
        match existing_backend_id {
            Some(existing) if existing == backend_id => Ok(()),
            Some(_) => {
                self.quarantine();
                anyhow::bail!("conflicting managed Comfy backend acknowledgement")
            }
            None => self.attach_backend(job_id, backend_id.to_owned()),
        }
    }

    pub fn complete(
        &mut self,
        job_id: u64,
        outcome: Completion,
        artifacts: Vec<String>,
        now: Instant,
    ) -> Result<RequestTiming> {
        let active = self
            .active
            .get(&job_id)
            .context("unknown dispatch reservation")?;
        ensure!(
            !active.cleanup_unresolved,
            "uncertain outcome requires cleanup acknowledgment, not reclassification"
        );
        ensure!(
            outcome != Completion::Succeeded || active.started.is_some(),
            "undispatched request cannot succeed"
        );
        let timing = RequestTiming {
            queue_wait: active.queue_wait,
            cold_start: active
                .started
                .unwrap_or(now)
                .saturating_duration_since(active.reserved),
            execution: active.started.map_or(Duration::ZERO, |started| {
                now.saturating_duration_since(started)
            }),
        };
        if let Err(error) =
            self.persist_update(job_id, outcome.history_state(), None, Some(artifacts))
        {
            self.quarantine();
            return Err(error
                .context("completion persistence failed; reconcile ownership before more work"));
        }
        if outcome == Completion::Uncertain {
            // A terminal outcome is not a cleanup acknowledgment. Preserve
            // the reservation until the executor separately proves cleanup.
            self.active
                .get_mut(&job_id)
                .expect("validated active reservation")
                .cleanup_unresolved = true;
            self.quarantine();
        } else {
            self.active.remove(&job_id);
        }
        Ok(timing)
    }

    /// Release an uncertain reservation only after the host executor has
    /// independently verified native cleanup and resource ownership. This
    /// method performs no probes, does not rewrite the interrupted outcome,
    /// and never opens admission; full reconciliation is still required.
    pub fn acknowledge_reconciled_cleanup(&mut self, job_id: u64) -> Result<()> {
        let active = self
            .active
            .get(&job_id)
            .context("unknown dispatch reservation")?;
        ensure!(
            active.cleanup_unresolved,
            "reservation has no unresolved cleanup"
        );
        ensure!(
            self.admission == AdmissionState::ReconciliationRequired,
            "cleanup acknowledgment requires quarantine"
        );
        self.active.remove(&job_id);
        Ok(())
    }

    fn persist_update(
        &mut self,
        id: u64,
        state: &str,
        backend_id: Option<String>,
        artifacts: Option<Vec<String>>,
    ) -> Result<()> {
        let mut candidate = self.history.clone();
        candidate.update_cross_toolkit(
            id,
            CrossToolkitJobUpdate {
                backend_id,
                result_state: Some(state.into()),
                artifact_refs: artifacts,
            },
        )?;
        candidate.persist_in(&self.data_root)?;
        self.history = candidate;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn validate_managed_stack_configuration(
    catalog: &Catalog,
    runtime: &RuntimeIdentity,
    expected_gpu: u32,
    startup_profile: &ProfileKey,
    supported_model_tasks: &BTreeMap<String, BTreeSet<String>>,
    allowed_local_profiles: &BTreeSet<ProfileKey>,
    allowed_comfy_profiles: &BTreeSet<ProfileKey>,
) -> Result<(Catalog, BTreeMap<RuntimeIdentity, ResourceClaim>)> {
    ensure!(
        runtime.lane == Lane::Language && runtime.as_str() == "llama-swap",
        "managed stack requires the configured llama-swap language runtime"
    );
    ensure!(
        startup_profile.lane == Lane::Language,
        "managed stack startup profile must be a language profile"
    );
    ensure!(
        !supported_model_tasks.is_empty(),
        "managed model list is empty"
    );
    ensure!(
        !allowed_local_profiles.is_empty() || !allowed_comfy_profiles.is_empty(),
        "managed local and Comfy profile allowlists are empty"
    );

    let refreshed = catalog.refresh()?;
    ensure!(
        refreshed
            .list()
            .iter()
            .filter(|profile| profile.runtime == *runtime)
            .all(|profile| {
                resolve_profile_route(&refreshed, &profile.key).is_ok_and(|route| {
                    matches!(route.native_route, NativeRoute::LlamaSwapModel { .. })
                })
            }),
        "configured language runtime contains an unsupported route"
    );

    for (model, tasks) in supported_model_tasks {
        ensure!(
            super::catalog::ProfileId::new(model.clone()).is_ok()
                && !tasks.is_empty()
                && tasks
                    .iter()
                    .all(|task| matches!(task.as_str(), "chat" | "embeddings")),
            "managed model/task declaration is invalid"
        );
        let profile = ProfileKey::new(Lane::Language, model.clone())?;
        let route = resolve_profile_route(&refreshed, &profile)?;
        ensure!(
            route.runtime == *runtime
                && matches!(
                    &route.native_route,
                    NativeRoute::LlamaSwapModel { model_id, .. }
                        if model_id.as_str() == model
                )
                && route.tasks.iter().all(|task| tasks.contains(task.as_str())),
            "managed model/task declaration does not match its native route"
        );
    }

    let startup_route = resolve_profile_route(&refreshed, startup_profile)?;
    ensure!(
        startup_route.runtime == *runtime
            && matches!(
                &startup_route.native_route,
                NativeRoute::LlamaSwapModel { model_id, .. }
                    if model_id.as_str() == startup_profile.profile_id.as_str()
                        && supported_model_tasks.get(model_id.as_str()).is_some_and(|tasks|
                            startup_route.tasks.iter().all(|task| tasks.contains(task.as_str())))
            ),
        "managed startup profile is not in the supported language model set"
    );

    let mut expected_bindings =
        BTreeMap::from([(runtime.clone(), ResourceClaim::ExclusiveGpu(expected_gpu))]);
    for profile in allowed_local_profiles {
        let route = resolve_profile_route(&refreshed, profile)?;
        let request = representative_managed_local_request(profile)?;
        let claim = validate_managed_local_route(&refreshed, &route, &request, expected_gpu)?;
        if let Some(existing) = expected_bindings.get(&route.runtime) {
            ensure!(
                *existing == claim,
                "profiles on one managed runtime require conflicting resource claims"
            );
        } else {
            expected_bindings.insert(route.runtime, claim);
        }
    }
    for profile in allowed_comfy_profiles {
        let route = resolve_profile_route(&refreshed, profile)?;
        let request = representative_managed_comfy_request(profile)?;
        let claim = validate_managed_comfy_route(
            &refreshed,
            &route,
            &request,
            allowed_comfy_profiles,
            expected_gpu,
        )?;
        if let Some(existing) = expected_bindings.get(&route.runtime) {
            ensure!(
                *existing == claim,
                "profiles on one managed runtime require conflicting resource claims"
            );
        } else {
            expected_bindings.insert(route.runtime, claim);
        }
    }

    Ok((refreshed, expected_bindings))
}

#[cfg(target_os = "linux")]
fn representative_managed_local_request(profile: &ProfileKey) -> Result<InferenceRequest> {
    match profile.lane {
        Lane::Media if profile.profile_id.as_str() == "breeze-tts-2" => {
            Ok(InferenceRequest::Media {
                profile: profile.clone(),
                input: MediaInput::Speech(SpeechInput {
                    text: "managed-stack native route validation".into(),
                    mode: SpeechMode::Standard,
                    instruction: None,
                    voice_reference: None,
                    reference_text: None,
                }),
            })
        }
        Lane::Decisions => Ok(InferenceRequest::Decision {
            profile: profile.clone(),
            state: serde_json::json!({}),
            questions: serde_json::json!({
                "route": {"type": "choice", "options": ["continue", "stop"]}
            })
            .as_object()
            .cloned()
            .context("build local decision route validation request")?,
        }),
        _ => anyhow::bail!("managed local profile is outside the explicit allowlist kinds"),
    }
}

#[cfg(target_os = "linux")]
fn representative_managed_comfy_request(profile: &ProfileKey) -> Result<InferenceRequest> {
    Ok(InferenceRequest::Media {
        profile: profile.clone(),
        input: MediaInput::Workflow {
            task: Task::new("image")?,
            graph: serde_json::json!({
                "1": { "class_type": "LoadImage", "inputs": { "image": "fixture.png" } }
            }),
        },
    })
}

fn validate_managed_stack_request(
    catalog: &Catalog,
    route: &ResolvedProfileRoute,
    request: &InferenceRequest,
    supported_model_tasks: &BTreeMap<String, BTreeSet<String>>,
    allowed_local_profiles: &BTreeSet<ProfileKey>,
    allowed_comfy_profiles: &BTreeSet<ProfileKey>,
    expected_gpu: u32,
) -> Result<ResourceClaim> {
    if let NativeRoute::LlamaSwapModel { model_id, .. } = &route.native_route {
        ensure!(
            route.profile.lane == Lane::Language
                && route.runtime.lane == Lane::Language
                && route.runtime.as_str() == "llama-swap"
                && request.lane() == Lane::Language
                && supported_model_tasks
                    .get(model_id.as_str())
                    .is_some_and(|tasks| tasks.contains(request.task())),
            "language request is outside the configured managed model/task set"
        );
        return Ok(ResourceClaim::ExclusiveGpu(expected_gpu));
    }
    if allowed_comfy_profiles.contains(&route.profile) {
        #[cfg(target_os = "linux")]
        {
            return validate_managed_comfy_route(
                catalog,
                route,
                request,
                allowed_comfy_profiles,
                expected_gpu,
            );
        }
        #[cfg(not(target_os = "linux"))]
        {
            anyhow::bail!("managed Comfy is available only on Linux")
        }
    }
    ensure!(
        allowed_local_profiles.contains(&route.profile),
        "local profile is outside the configured managed allowlist"
    );
    validate_managed_local_route(catalog, route, request, expected_gpu)
}

#[cfg(target_os = "linux")]
fn validate_managed_comfy_route(
    catalog: &Catalog,
    route: &ResolvedProfileRoute,
    request: &InferenceRequest,
    allowed_comfy_profiles: &BTreeSet<ProfileKey>,
    expected_gpu: u32,
) -> Result<ResourceClaim> {
    ensure!(
        allowed_comfy_profiles.contains(&route.profile)
            && request.profile() == &route.profile
            && request.validate(catalog).is_ok(),
        "managed Comfy request is outside its explicit profile allowlist"
    );
    let InferenceRequest::Media {
        input: MediaInput::Workflow { graph, .. },
        ..
    } = request
    else {
        anyhow::bail!("managed Comfy accepts only explicit native workflow requests")
    };
    let (native_route, _) =
        super::managed_comfy_config::validate_comfy_profile(catalog, &route.profile)?;
    ensure!(
        &native_route == route
            && route.profile.lane == Lane::Media
            && route.runtime.lane == Lane::Media
            && route.runtime.as_str() == "ComfyUI"
            && route.device == Some(Device::Cuda)
            && route.tasks.len() == 1
            && route.tasks[0].as_str() == "image"
            && request.task() == "image",
        "managed Comfy native profile route is not explicit CUDA image execution"
    );
    super::managed_comfy_config::validate_api_workflow_graph(graph)?;
    Ok(ResourceClaim::ExclusiveGpu(expected_gpu))
}

fn validate_managed_local_route(
    catalog: &Catalog,
    route: &ResolvedProfileRoute,
    request: &InferenceRequest,
    expected_gpu: u32,
) -> Result<ResourceClaim> {
    ensure!(
        request.profile() == &route.profile && request.validate(catalog).is_ok(),
        "managed local request does not match a valid current route"
    );
    match (
        &route.profile.lane,
        route.profile.profile_id.as_str(),
        &route.native_route,
    ) {
        (Lane::Media, "breeze-tts-2", NativeRoute::MediaScript { .. }) => {
            ensure!(
                route.runtime.lane == Lane::Media
                    && route.runtime.as_str() == "audio.cpp"
                    && route.device == Some(Device::Cuda)
                    && matches!(
                        request,
                        InferenceRequest::Media {
                            input: MediaInput::Speech(_),
                            ..
                        }
                    ),
                "managed Breeze route must be CUDA audio.cpp speech"
            );
            let recipe =
                super::native_commands::prepare_native_command_for_route(catalog, request, route)?;
            ensure!(
                matches!(
                    recipe.expectation,
                    super::native_commands::NativeOutputExpectation::BreezeSpeech { .. }
                ),
                "managed Breeze profile did not prepare a speech command"
            );
            Ok(ResourceClaim::ExclusiveGpu(expected_gpu))
        }
        (Lane::Decisions, _, NativeRoute::DecisionRunner { profile_type, .. }) => {
            ensure!(
                route.runtime.lane == Lane::Decisions
                    && matches!(route.runtime.as_str(), "nagi" | "laya")
                    && profile_type == "local"
                    && matches!(request, InferenceRequest::Decision { .. }),
                "managed decision route must be a local Nagi/Laya request"
            );
            let claim = match route.device.as_ref() {
                Some(Device::Cpu) => ResourceClaim::None,
                Some(Device::Cuda) => ResourceClaim::ExclusiveGpu(expected_gpu),
                _ => anyhow::bail!("managed local decision route has no explicit CPU/CUDA device"),
            };
            let recipe =
                super::native_commands::prepare_native_command_for_route(catalog, request, route)?;
            ensure!(
                matches!(
                    recipe.expectation,
                    super::native_commands::NativeOutputExpectation::DecisionJsonStdout {
                        engine,
                        profile_type: prepared_type,
                        ..
                    } if engine == route.runtime.as_str() && prepared_type == "local"
                ),
                "managed decision route did not prepare its local native runner"
            );
            Ok(claim)
        }
        _ => anyhow::bail!("managed local route is not an admitted Breeze or local decision"),
    }
}

fn wire_error(request_id: String, code: ResponseErrorCode) -> ResponseMessage {
    ResponseMessage::Error {
        request_id: Some(request_id),
        code,
    }
}

fn cpu_decision_route(route: &ResolvedProfileRoute) -> bool {
    route.profile.lane == Lane::Decisions
        && route.runtime.lane == Lane::Decisions
        && matches!(route.runtime.as_str(), "nagi" | "laya")
        && route.device == Some(Device::Cpu)
        && matches!(&route.native_route, super::routes::NativeRoute::DecisionRunner { profile_type, .. } if profile_type == "local")
}

fn toolkit_for_lane(lane: Lane) -> &'static str {
    match lane {
        Lane::Language => "l3ms",
        Lane::Media => "l2m2",
        Lane::Decisions => "l3ds",
    }
}

fn retained_job_may_be_unresolved(job: &JobRecord) -> bool {
    match job.metadata.result_state.as_deref() {
        Some("succeeded") => job.status != "done",
        Some("failed") => job.status != "failed",
        Some("cancelled") => job.status != "cancelled",
        Some(_) => true,
        None => !matches!(job.status.as_str(), "done" | "failed" | "cancelled"),
    }
}

fn same_request(left: &InferenceRequest, right: &InferenceRequest) -> bool {
    use InferenceRequest::{Chat, Decision, Embeddings, Media};

    match (left, right) {
        (
            Chat {
                profile: left_profile,
                turns: left_turns,
                max_tokens: left_max_tokens,
            },
            Chat {
                profile: right_profile,
                turns: right_turns,
                max_tokens: right_max_tokens,
            },
        ) => {
            left_profile == right_profile
                && left_max_tokens == right_max_tokens
                && left_turns.len() == right_turns.len()
                && left_turns
                    .iter()
                    .zip(right_turns)
                    .all(|(left, right)| left.role == right.role && left.content == right.content)
        }
        (
            Embeddings {
                profile: left_profile,
                input: left_input,
            },
            Embeddings {
                profile: right_profile,
                input: right_input,
            },
        ) => left_profile == right_profile && left_input == right_input,
        (
            Media {
                profile: left_profile,
                input: left_input,
            },
            Media {
                profile: right_profile,
                input: right_input,
            },
        ) => {
            left_profile == right_profile
                && match (left_input, right_input) {
                    (
                        MediaInput::Workflow {
                            task: left_task,
                            graph: left_graph,
                        },
                        MediaInput::Workflow {
                            task: right_task,
                            graph: right_graph,
                        },
                    ) => left_task == right_task && left_graph == right_graph,
                    (
                        MediaInput::Script {
                            task: left_task,
                            args: left_args,
                        },
                        MediaInput::Script {
                            task: right_task,
                            args: right_args,
                        },
                    ) => left_task == right_task && left_args == right_args,
                    (MediaInput::Speech(left), MediaInput::Speech(right)) => {
                        left.text == right.text
                            && left.mode == right.mode
                            && left.instruction == right.instruction
                            && left.voice_reference == right.voice_reference
                            && left.reference_text == right.reference_text
                    }
                    _ => false,
                }
        }
        (
            Decision {
                profile: left_profile,
                state: left_state,
                questions: left_questions,
            },
            Decision {
                profile: right_profile,
                state: right_state,
                questions: right_questions,
            },
        ) => {
            left_profile == right_profile
                && left_state == right_state
                && left_questions == right_questions
        }
        _ => false,
    }
}

fn resources_conflict(left: ResourceClaim, right: ResourceClaim) -> bool {
    matches!((left, right), (ResourceClaim::ExclusiveGpu(a), ResourceClaim::ExclusiveGpu(b)) if a == b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::catalog::{ProfileKey, Task};
    use crate::control::requests::{ChatRole, ChatTurn, MediaInput};
    use tempfile::TempDir;

    fn core(data: &Path, capacity: usize) -> CoordinatorCore {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let bindings = BTreeMap::from([
            (
                RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap(),
                ResourceClaim::ExclusiveGpu(0),
            ),
            (
                RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap(),
                ResourceClaim::ExclusiveGpu(0),
            ),
            (
                RuntimeIdentity::new(Lane::Decisions, "jev").unwrap(),
                ResourceClaim::None,
            ),
        ]);
        CoordinatorCore::load(Catalog::load(root).unwrap(), bindings, data, root, capacity).unwrap()
    }

    fn chat() -> InferenceRequest {
        InferenceRequest::Chat {
            profile: ProfileKey::new(Lane::Language, "gemma-4-26b-qat-mtp").unwrap(),
            turns: vec![ChatTurn {
                role: ChatRole::User,
                content: "REQUEST_PAYLOAD_MUST_NOT_BE_PERSISTED".into(),
            }],
            max_tokens: 8,
        }
    }

    fn media() -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap(),
            input: MediaInput::Script {
                task: Task::new("tts").unwrap(),
                args: vec!["--text".into(), "hello".into()],
            },
        }
    }

    fn hosted() -> InferenceRequest {
        InferenceRequest::Decision {
            profile: ProfileKey::new(Lane::Decisions, "jev-api").unwrap(), state: serde_json::json!({}),
            questions: serde_json::json!({"route":{"type":"choice","criteria":{"yes":"Accept","no":"Reject"}}}).as_object().unwrap().clone(),
        }
    }

    struct DispatchFixture {
        root: TempDir,
        media_manifest: PathBuf,
    }

    impl DispatchFixture {
        fn new() -> Self {
            let root = TempDir::new().unwrap();
            let language_root = root.path().join("language");
            let media_root = root.path().join("media");
            let decisions_root = root.path().join("decisions");
            std::fs::create_dir_all(&language_root).unwrap();
            std::fs::create_dir_all(&media_root).unwrap();
            std::fs::create_dir_all(&decisions_root).unwrap();
            std::fs::write(
                language_root.join("llama-swap.yaml"),
                "models:\n  fixture-model:\n    cmd: model\n",
            )
            .unwrap();
            let media_manifest = media_root.join("media-runtimes.json");
            std::fs::write(
                &media_manifest,
                serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "fixture-profile",
                        "name": "Fixture media profile",
                        "runtime": "fixture-runtime",
                        "tasks": ["tts"],
                        "script": "scripts/runner.py"
                    }]
                })
                .to_string(),
            )
            .unwrap();
            std::fs::write(
                decisions_root.join("decision-runtimes.json"),
                r#"{"version":"1.0","profiles":[]}"#,
            )
            .unwrap();
            Self {
                root,
                media_manifest,
            }
        }

        fn coordinator(&self) -> CoordinatorCore {
            let language_root = self.root.path().join("language");
            let media_root = self.root.path().join("media");
            let decisions_root = self.root.path().join("decisions");
            let catalog =
                Catalog::load_from_roots(language_root, media_root, decisions_root).unwrap();
            let bindings = BTreeMap::from([(
                RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap(),
                ResourceClaim::ExclusiveGpu(0),
            )]);
            CoordinatorCore::load(catalog, bindings, self.root.path(), self.root.path(), 8).unwrap()
        }

        fn new_comfy() -> Self {
            let fixture = Self::new();
            fixture.write_comfy_manifest("ComfyUI", "workflows/fixture.json");
            fixture
        }

        fn comfy_coordinator(&self) -> CoordinatorCore {
            let language_root = self.root.path().join("language");
            let media_root = self.root.path().join("media");
            let decisions_root = self.root.path().join("decisions");
            let catalog =
                Catalog::load_from_roots(language_root, media_root, decisions_root).unwrap();
            let bindings = BTreeMap::from([(
                RuntimeIdentity::new(Lane::Media, "ComfyUI").unwrap(),
                ResourceClaim::ExclusiveGpu(0),
            )]);
            CoordinatorCore::load(catalog, bindings, self.root.path(), self.root.path(), 8).unwrap()
        }

        fn write_comfy_manifest(&self, runtime: &str, script: &str) {
            std::fs::write(
                &self.media_manifest,
                serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "fixture-profile",
                        "name": "Fixture Comfy profile",
                        "runtime": runtime,
                        "device": "cuda",
                        "tasks": ["image"],
                        "script": script
                    }]
                })
                .to_string(),
            )
            .unwrap();
        }

        fn rewrite_media(&self, runtime: &str, script: &str) {
            std::fs::write(
                &self.media_manifest,
                serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "fixture-profile",
                        "name": "Fixture media profile",
                        "runtime": runtime,
                        "tasks": ["tts"],
                        "script": script
                    }]
                })
                .to_string(),
            )
            .unwrap();
        }

        fn request() -> InferenceRequest {
            InferenceRequest::Media {
                profile: ProfileKey::new(Lane::Media, "fixture-profile").unwrap(),
                input: MediaInput::Script {
                    task: Task::new("tts").unwrap(),
                    args: vec!["--text".into(), "fixture input".into()],
                },
            }
        }

        fn comfy_request() -> InferenceRequest {
            InferenceRequest::Media {
                profile: ProfileKey::new(Lane::Media, "fixture-profile").unwrap(),
                input: MediaInput::Workflow {
                    task: Task::new("image").unwrap(),
                    graph: serde_json::json!({"1": {"class_type": "FixtureNode", "inputs": {}}}),
                },
            }
        }

        fn reserve(
            &self,
            coordinator: &mut CoordinatorCore,
            now: Instant,
        ) -> (u64, DispatchTicket) {
            coordinator.confirm_reconciled().unwrap();
            let id = coordinator
                .submit(
                    "dispatch-fixture".into(),
                    Self::request(),
                    Duration::from_secs(30),
                    now,
                )
                .unwrap();
            let ticket = coordinator.reserve_next(now).unwrap().unwrap();
            (id, ticket)
        }

        fn reserve_comfy(
            &self,
            coordinator: &mut CoordinatorCore,
            now: Instant,
        ) -> (u64, DispatchTicket) {
            coordinator.confirm_reconciled().unwrap();
            let id = coordinator
                .submit(
                    "comfy-dispatch-fixture".into(),
                    Self::comfy_request(),
                    Duration::from_secs(30),
                    now,
                )
                .unwrap();
            let ticket = coordinator.reserve_next(now).unwrap().unwrap();
            (id, ticket)
        }
    }

    fn started_comfy_dispatch() -> (
        DispatchFixture,
        CoordinatorCore,
        u64,
        DispatchTicket,
        Instant,
    ) {
        let fixture = DispatchFixture::new_comfy();
        let mut coordinator = fixture.comfy_coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve_comfy(&mut coordinator, now);
        coordinator
            .begin_dispatch(&ticket, now + Duration::from_millis(1))
            .unwrap();
        (fixture, coordinator, job_id, ticket, now)
    }

    fn persisted_state(data_root: &Path, request_id: &str) -> Option<String> {
        let jobs: serde_json::Value =
            serde_json::from_slice(&std::fs::read(data_root.join("jobs.json")).unwrap()).unwrap();
        jobs.as_array()?
            .iter()
            .find(|job| job["request_id"] == request_id)
            .and_then(|job| job["result_state"].as_str())
            .map(str::to_owned)
    }

    #[cfg(target_os = "linux")]
    fn managed_language_core(
        root: &Path,
        data: &Path,
        bindings: BTreeMap<RuntimeIdentity, ResourceClaim>,
    ) -> CoordinatorCore {
        let language = root.join("language");
        let media = root.join("media");
        let decisions = root.join("decisions");
        std::fs::create_dir_all(&language).unwrap();
        std::fs::create_dir_all(&media).unwrap();
        std::fs::create_dir_all(&decisions).unwrap();
        std::fs::write(
            language.join("llama-swap.yaml"),
            "models:\n  fixture-model:\n    cmd: llama-server\n",
        )
        .unwrap();
        std::fs::write(
            media.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        std::fs::write(
            decisions.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let catalog = Catalog::load_from_roots(&language, &media, &decisions).unwrap();
        CoordinatorCore::load(catalog, bindings, data, root, 8).unwrap()
    }

    #[cfg(target_os = "linux")]
    fn managed_stack_core(
        root: &Path,
        data: &Path,
        bindings: BTreeMap<RuntimeIdentity, ResourceClaim>,
    ) -> CoordinatorCore {
        use std::os::unix::fs::PermissionsExt;

        let language = root.join("language");
        let media = root.join("media");
        let decisions = root.join("decisions");
        std::fs::create_dir_all(&language).unwrap();
        std::fs::create_dir_all(media.join("media-models")).unwrap();
        std::fs::create_dir_all(decisions.join("decision-models")).unwrap();
        std::fs::create_dir_all(decisions.join(".venv/bin")).unwrap();
        std::fs::write(
            language.join("llama-swap.yaml"),
            "models:\n  fixture-model:\n    cmd: llama-server\n",
        )
        .unwrap();
        std::fs::write(
            media.join("media-runtimes.json"),
            serde_json::json!({
                "schema_version": 1,
                "profiles": [
                    {
                        "id": "breeze-tts-2",
                        "name": "Breeze fixture",
                        "runtime": "audio.cpp",
                        "device": "cuda",
                        "tasks": ["tts", "design", "clon"],
                        "script": "media-models/generate-breeze-tts-2.sh"
                    },
                    {
                        "id": "other-audio",
                        "name": "Other fixture script",
                        "runtime": "audio.cpp",
                        "device": "cuda",
                        "tasks": ["tts"],
                        "script": "media-models/other.sh"
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            decisions.join("decision-runtimes.json"),
            serde_json::json!({
                "version": "1.0",
                "profiles": [
                    {
                        "id": "nagi-cpu",
                        "name": "Nagi CPU fixture",
                        "engine": "nagi",
                        "type": "local",
                        "device": "cpu",
                        "model": "fixture/nagi",
                        "entrypoint": "decision-models/runner.py"
                    },
                    {
                        "id": "nagi-cpu-alt",
                        "name": "Nagi CPU alternate fixture",
                        "engine": "nagi",
                        "type": "local",
                        "device": "cpu",
                        "model": "fixture/nagi-alt",
                        "entrypoint": "decision-models/runner.py"
                    },
                    {
                        "id": "nagi-cuda",
                        "name": "Nagi CUDA fixture",
                        "engine": "nagi",
                        "type": "local",
                        "device": "cuda",
                        "model": "fixture/nagi-cuda",
                        "entrypoint": "decision-models/runner.py"
                    },
                    {
                        "id": "laya-cuda",
                        "name": "Laya CUDA fixture",
                        "engine": "laya",
                        "type": "local",
                        "device": "cuda",
                        "model": "fixture/laya",
                        "entrypoint": "decision-models/runner.py"
                    },
                    {
                        "id": "jev-api",
                        "name": "Hosted fixture",
                        "engine": "jev",
                        "type": "api",
                        "device": "remote",
                        "model": "fixture/hosted",
                        "entrypoint": "decision-models/runner.py"
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            media.join("media-models/generate-breeze-tts-2.sh"),
            "#!/bin/bash\ntouch managed-media-script-executed\nexit 0\n",
        )
        .unwrap();
        std::fs::write(media.join("media-models/other.sh"), "#!/bin/bash\nexit 0\n").unwrap();
        std::fs::write(
            decisions.join("decision-models/runner.py"),
            "# inert fixture; startup validation must not execute this file\n",
        )
        .unwrap();
        let python = decisions.join(".venv/bin/python");
        std::fs::write(
            &python,
            "#!/bin/sh\ntouch managed-decision-runner-executed\nexit 0\n",
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o700)).unwrap();

        let catalog = Catalog::load_from_roots(&language, &media, &decisions).unwrap();
        CoordinatorCore::load(catalog, bindings, data, root, 8).unwrap()
    }

    #[cfg(target_os = "linux")]
    fn managed_comfy_stack_core(root: &Path, data: &Path, device: &str) -> CoordinatorCore {
        use std::os::unix::fs::PermissionsExt;

        let language = root.join("language");
        let media = root.join("media");
        let decisions = root.join("decisions");
        std::fs::create_dir_all(&language).unwrap();
        std::fs::create_dir_all(media.join("media-models")).unwrap();
        std::fs::create_dir_all(media.join("workflows/qwen-image-2.1")).unwrap();
        std::fs::create_dir_all(&decisions).unwrap();
        std::fs::write(
            language.join("llama-swap.yaml"),
            "models:\n  fixture-model:\n    cmd: llama-server\n",
        )
        .unwrap();
        std::fs::write(
            media.join("media-runtimes.json"),
            serde_json::json!({
                "schema_version": 1,
                "profiles": [{
                    "id": "qwen-image-2.1",
                    "name": "Comfy fixture",
                    "runtime": "ComfyUI",
                    "device": device,
                    "tasks": ["image"],
                    "script": "media-models/generate-qwen-image-2.1.sh"
                }]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            decisions.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        // Inert scripts and native workflow templates are identity/shape
        // evidence only. Admission never invokes a media script.
        let script = media.join("media-models/generate-qwen-image-2.1.sh");
        std::fs::write(&script, "#!/bin/sh\ntouch media-script-was-executed\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        for name in [
            "image_qwen_image_2_1_t2i.json",
            "image_qwen_image_2_1_image_edit.json",
        ] {
            std::fs::write(
                media.join("workflows/qwen-image-2.1").join(name),
                r#"{"nodes":[{"id":1,"type":"LoadImage"}],"links":[]}"#,
            )
            .unwrap();
        }
        let catalog = Catalog::load_from_roots(&language, &media, &decisions).unwrap();
        CoordinatorCore::load(catalog, managed_comfy_stack_bindings(), data, root, 8).unwrap()
    }

    #[cfg(target_os = "linux")]
    fn managed_comfy_stack_bindings() -> BTreeMap<RuntimeIdentity, ResourceClaim> {
        BTreeMap::from([
            (
                RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
            (
                RuntimeIdentity::new(Lane::Media, "ComfyUI").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
        ])
    }

    #[cfg(target_os = "linux")]
    fn managed_tasks() -> BTreeMap<String, BTreeSet<String>> {
        BTreeMap::from([("fixture-model".into(), BTreeSet::from(["chat".into()]))])
    }

    #[cfg(target_os = "linux")]
    fn managed_stack_bindings() -> BTreeMap<RuntimeIdentity, ResourceClaim> {
        BTreeMap::from([
            (
                RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
            (
                RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
            (
                RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap(),
                ResourceClaim::None,
            ),
            (
                RuntimeIdentity::new(Lane::Decisions, "laya").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
        ])
    }

    #[cfg(target_os = "linux")]
    fn managed_local_profiles() -> BTreeSet<ProfileKey> {
        BTreeSet::from([
            ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap(),
            ProfileKey::new(Lane::Decisions, "nagi-cpu").unwrap(),
            ProfileKey::new(Lane::Decisions, "nagi-cpu-alt").unwrap(),
            ProfileKey::new(Lane::Decisions, "laya-cuda").unwrap(),
        ])
    }

    #[cfg(target_os = "linux")]
    fn managed_comfy_profiles() -> BTreeSet<ProfileKey> {
        BTreeSet::from([ProfileKey::new(Lane::Media, "qwen-image-2.1").unwrap()])
    }

    #[cfg(target_os = "linux")]
    fn managed_stack_speech() -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap(),
            input: MediaInput::Speech(SpeechInput {
                text: "small model-free admission test".into(),
                mode: SpeechMode::Standard,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            }),
        }
    }

    #[cfg(target_os = "linux")]
    fn managed_local_decision(profile_id: &str) -> InferenceRequest {
        InferenceRequest::Decision {
            profile: ProfileKey::new(Lane::Decisions, profile_id).unwrap(),
            state: serde_json::json!({"fixture": true}),
            questions: serde_json::json!({
                "route": {"type": "choice", "options": ["continue", "stop"]}
            })
            .as_object()
            .unwrap()
            .clone(),
        }
    }

    #[cfg(target_os = "linux")]
    fn confirm_managed_stack(core: &mut CoordinatorCore) -> Result<()> {
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap")?;
        let startup = ProfileKey::new(Lane::Language, "fixture-model")?;
        core.confirm_managed_stack_ready(
            &runtime,
            2,
            &startup,
            managed_tasks(),
            ManagedStackProfiles {
                local_jobs: managed_local_profiles(),
                comfy: BTreeSet::new(),
            },
            &super::super::engine_health::RouterModelReadinessObservation::test_for_model(
                "fixture-model",
            ),
        )
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_stack_startup_requires_exact_supported_routes_and_resource_claims() {
        let root = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let startup = ProfileKey::new(Lane::Language, "fixture-model").unwrap();
        let allowed = managed_local_profiles();
        let core = managed_stack_core(root.path(), data.path(), managed_stack_bindings());

        core.require_managed_stack_startup(
            &runtime,
            2,
            &startup,
            &managed_tasks(),
            &allowed,
            &BTreeSet::new(),
        )
        .unwrap();
        assert!(!root
            .path()
            .join("media/managed-media-script-executed")
            .exists());
        assert!(!root
            .path()
            .join("decisions/managed-decision-runner-executed")
            .exists());
        assert!(core
            .require_managed_stack_startup(
                &runtime,
                1,
                &startup,
                &managed_tasks(),
                &allowed,
                &BTreeSet::new(),
            )
            .is_err());

        let wrong_cpu_data = TempDir::new().unwrap();
        let wrong_cpu_fallback = BTreeMap::from([
            (runtime.clone(), ResourceClaim::ExclusiveGpu(2)),
            (
                RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
            (
                RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
            (
                RuntimeIdentity::new(Lane::Decisions, "laya").unwrap(),
                ResourceClaim::ExclusiveGpu(2),
            ),
        ]);
        let cpu_claim_mismatch =
            managed_stack_core(root.path(), wrong_cpu_data.path(), wrong_cpu_fallback);
        assert!(cpu_claim_mismatch
            .require_managed_stack_startup(
                &runtime,
                2,
                &startup,
                &managed_tasks(),
                &allowed,
                &BTreeSet::new(),
            )
            .is_err());

        let mut conflicting = allowed.clone();
        conflicting.insert(ProfileKey::new(Lane::Decisions, "nagi-cuda").unwrap());
        assert!(core
            .require_managed_stack_startup(
                &runtime,
                2,
                &startup,
                &managed_tasks(),
                &conflicting,
                &BTreeSet::new(),
            )
            .is_err());

        let missing_runtime_data = TempDir::new().unwrap();
        let missing_runtime = managed_stack_core(
            root.path(),
            missing_runtime_data.path(),
            BTreeMap::from([
                (runtime.clone(), ResourceClaim::ExclusiveGpu(2)),
                (
                    RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap(),
                    ResourceClaim::ExclusiveGpu(2),
                ),
                (
                    RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap(),
                    ResourceClaim::None,
                ),
            ]),
        );
        assert!(missing_runtime
            .require_managed_stack_startup(
                &runtime,
                2,
                &startup,
                &managed_tasks(),
                &allowed,
                &BTreeSet::new(),
            )
            .is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_comfy_opt_in_requires_native_cuda_workflow_route_and_never_runs_script() {
        let root = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let startup = ProfileKey::new(Lane::Language, "fixture-model").unwrap();
        let comfy_profiles = managed_comfy_profiles();
        let mut core = managed_comfy_stack_core(root.path(), data.path(), "cuda");

        assert!(core
            .require_managed_stack_startup(
                &runtime,
                2,
                &startup,
                &managed_tasks(),
                &BTreeSet::new(),
                &BTreeSet::new(),
            )
            .is_err());
        core.require_managed_stack_startup(
            &runtime,
            2,
            &startup,
            &managed_tasks(),
            &BTreeSet::new(),
            &comfy_profiles,
        )
        .unwrap();
        let cpu_root = TempDir::new().unwrap();
        let cpu_data = TempDir::new().unwrap();
        let cpu_core = managed_comfy_stack_core(cpu_root.path(), cpu_data.path(), "cpu");
        assert!(cpu_core
            .require_managed_stack_startup(
                &runtime,
                2,
                &startup,
                &managed_tasks(),
                &BTreeSet::new(),
                &comfy_profiles,
            )
            .is_err());

        core.confirm_managed_stack_ready(
            &runtime,
            2,
            &startup,
            managed_tasks(),
            ManagedStackProfiles {
                local_jobs: BTreeSet::new(),
                comfy: comfy_profiles.clone(),
            },
            &super::super::engine_health::RouterModelReadinessObservation::test_for_model(
                "fixture-model",
            ),
        )
        .unwrap();
        let now = Instant::now();
        let profile = ProfileKey::new(Lane::Media, "qwen-image-2.1").unwrap();
        let scripted = InferenceRequest::Media {
            profile: profile.clone(),
            input: MediaInput::Script {
                task: Task::new("image").unwrap(),
                args: Vec::new(),
            },
        };
        assert_eq!(
            core.submit_checked(
                "managed-comfy-script-fallback".into(),
                scripted,
                Duration::from_secs(30),
                now,
            ),
            Err(SubmissionError::Unavailable)
        );
        let invalid_graph = InferenceRequest::Media {
            profile: profile.clone(),
            input: MediaInput::Workflow {
                task: Task::new("image").unwrap(),
                graph: serde_json::json!({
                    "1": { "class_type": "LoadImage", "inputs": {}, "raw": true }
                }),
            },
        };
        assert_eq!(
            core.submit_checked(
                "managed-comfy-invalid-graph".into(),
                invalid_graph,
                Duration::from_secs(30),
                now,
            ),
            Err(SubmissionError::Unavailable)
        );

        let request = InferenceRequest::Media {
            profile,
            input: MediaInput::Workflow {
                task: Task::new("image").unwrap(),
                graph: serde_json::json!({
                    "1": { "class_type": "LoadImage", "inputs": { "image": "fixture.png" } }
                }),
            },
        };
        let job_id = core
            .submit_checked(
                "managed-comfy-workflow".into(),
                request,
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let ticket = core.reserve_next(now).unwrap().unwrap();
        assert_eq!(ticket.resource, ResourceClaim::ExclusiveGpu(2));
        assert_eq!(ticket.route.runtime.as_str(), "ComfyUI");
        core.begin_dispatch(&ticket, now + Duration::from_millis(1))
            .unwrap();
        core.complete(
            job_id,
            Completion::Failed,
            Vec::new(),
            now + Duration::from_millis(2),
        )
        .unwrap();
        assert!(!root.path().join("media-script-was-executed").exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_stack_readiness_opens_only_for_observed_language_and_local_allowlist() {
        let root = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let startup = ProfileKey::new(Lane::Language, "fixture-model").unwrap();
        let mut core = managed_stack_core(root.path(), data.path(), managed_stack_bindings());
        core.require_managed_stack_startup(
            &runtime,
            2,
            &startup,
            &managed_tasks(),
            &managed_local_profiles(),
            &BTreeSet::new(),
        )
        .unwrap();
        assert!(core
            .confirm_managed_stack_ready(
                &runtime,
                2,
                &startup,
                managed_tasks(),
                ManagedStackProfiles {
                    local_jobs: managed_local_profiles(),
                    comfy: BTreeSet::new()
                },
                &super::super::engine_health::RouterModelReadinessObservation::test_for_model(
                    "other-model",
                ),
            )
            .is_err());
        confirm_managed_stack(&mut core).unwrap();
        assert_eq!(core.admission(), AdmissionState::Ready);

        let now = Instant::now();
        let language_id = core
            .submit_checked(
                "stack-language".into(),
                InferenceRequest::Chat {
                    profile: ProfileKey::new(Lane::Language, "fixture-model").unwrap(),
                    turns: vec![ChatTurn {
                        role: ChatRole::User,
                        content: "model-free admission request".into(),
                    }],
                    max_tokens: 8,
                },
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let language_ticket = core.reserve_next(now).unwrap().unwrap();
        assert_eq!(language_ticket.resource, ResourceClaim::ExclusiveGpu(2));
        core.begin_dispatch(&language_ticket, now + Duration::from_millis(1))
            .unwrap();
        core.complete(
            language_id,
            Completion::Succeeded,
            Vec::new(),
            now + Duration::from_millis(2),
        )
        .unwrap();

        let speech_id = core
            .submit_checked(
                "stack-speech".into(),
                managed_stack_speech(),
                Duration::from_secs(30),
                now + Duration::from_millis(3),
            )
            .unwrap();
        let speech_ticket = core
            .reserve_next(now + Duration::from_millis(3))
            .unwrap()
            .unwrap();
        core.begin_dispatch(&speech_ticket, now + Duration::from_millis(4))
            .unwrap();
        core.complete(
            speech_id,
            Completion::Succeeded,
            Vec::new(),
            now + Duration::from_millis(5),
        )
        .unwrap();

        let decision_id = core
            .submit_checked(
                "stack-cpu-decision".into(),
                managed_local_decision("nagi-cpu"),
                Duration::from_secs(30),
                now + Duration::from_millis(6),
            )
            .unwrap();
        let decision_ticket = core
            .reserve_next(now + Duration::from_millis(6))
            .unwrap()
            .unwrap();
        assert_eq!(decision_ticket.resource, ResourceClaim::None);
        core.begin_dispatch(&decision_ticket, now + Duration::from_millis(7))
            .unwrap();
        core.complete(
            decision_id,
            Completion::Succeeded,
            Vec::new(),
            now + Duration::from_millis(8),
        )
        .unwrap();

        let cuda_decision_id = core
            .submit_checked(
                "stack-cuda-decision".into(),
                managed_local_decision("laya-cuda"),
                Duration::from_secs(30),
                now + Duration::from_millis(9),
            )
            .unwrap();
        let cuda_decision_ticket = core
            .reserve_next(now + Duration::from_millis(9))
            .unwrap()
            .unwrap();
        assert_eq!(
            cuda_decision_ticket.resource,
            ResourceClaim::ExclusiveGpu(2)
        );
        core.begin_dispatch(&cuda_decision_ticket, now + Duration::from_millis(10))
            .unwrap();
        core.complete(
            cuda_decision_id,
            Completion::Succeeded,
            Vec::new(),
            now + Duration::from_millis(11),
        )
        .unwrap();

        let mismatch = InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap(),
            input: MediaInput::Script {
                task: Task::new("tts").unwrap(),
                args: Vec::new(),
            },
        };
        assert_eq!(
            core.submit_checked(
                "stack-wrong-media-shape".into(),
                mismatch,
                Duration::from_secs(30),
                now,
            ),
            Err(SubmissionError::Unavailable)
        );
        let unjoined_comfy = InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap(),
            input: MediaInput::Workflow {
                task: Task::new("tts").unwrap(),
                graph: serde_json::json!({"fixture": true}),
            },
        };
        assert_eq!(
            core.submit_checked(
                "stack-unjoined-comfy".into(),
                unjoined_comfy,
                Duration::from_secs(30),
                now,
            ),
            Err(SubmissionError::Unavailable)
        );
        let unlisted = InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "other-audio").unwrap(),
            input: MediaInput::Script {
                task: Task::new("tts").unwrap(),
                args: Vec::new(),
            },
        };
        assert_eq!(
            core.submit_checked(
                "stack-unlisted-media".into(),
                unlisted,
                Duration::from_secs(30),
                now,
            ),
            Err(SubmissionError::Unavailable)
        );
        assert_eq!(
            core.submit_checked(
                "stack-hosted-decision".into(),
                hosted(),
                Duration::from_secs(30),
                now,
            ),
            Err(SubmissionError::Unavailable)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_stack_revalidates_route_before_dispatch_and_refuses_unresolved_history() {
        let root = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let startup = ProfileKey::new(Lane::Language, "fixture-model").unwrap();
        let mut core = managed_stack_core(root.path(), data.path(), managed_stack_bindings());
        core.history
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: "l3ds".into(),
                profile: Some("nagi-cpu".into()),
                task: Some("decision".into()),
                request_id: Some("retained-unresolved".into()),
            })
            .unwrap();
        assert!(core
            .confirm_managed_stack_ready(
                &runtime,
                2,
                &startup,
                managed_tasks(),
                ManagedStackProfiles {
                    local_jobs: managed_local_profiles(),
                    comfy: BTreeSet::new()
                },
                &super::super::engine_health::RouterModelReadinessObservation::test_for_model(
                    "fixture-model",
                ),
            )
            .is_err());

        let clean_root = TempDir::new().unwrap();
        let clean_data = TempDir::new().unwrap();
        let mut clean = managed_stack_core(
            clean_root.path(),
            clean_data.path(),
            managed_stack_bindings(),
        );
        confirm_managed_stack(&mut clean).unwrap();
        let now = Instant::now();
        clean
            .submit_checked(
                "stack-route-drift".into(),
                managed_stack_speech(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let ticket = clean.reserve_next(now).unwrap().unwrap();
        std::fs::write(
            clean_root.path().join("media/media-models/replacement.sh"),
            "#!/bin/bash\nexit 0\n",
        )
        .unwrap();
        std::fs::write(
            clean_root.path().join("media/media-runtimes.json"),
            serde_json::json!({
                "schema_version": 1,
                "profiles": [{
                    "id": "breeze-tts-2",
                    "name": "Breeze fixture",
                    "runtime": "audio.cpp",
                    "device": "cuda",
                    "tasks": ["tts"],
                    "script": "media-models/replacement.sh"
                }]
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(
            clean.begin_dispatch(&ticket, now + Duration::from_millis(1)),
            Err(DispatchBeginError::RouteChanged)
        );
        assert_eq!(clean.admission(), AdmissionState::ReconciliationRequired);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_startup_gate_requires_the_exact_single_language_gpu_binding() {
        let root = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let core = managed_language_core(
            root.path(),
            data.path(),
            BTreeMap::from([(runtime.clone(), ResourceClaim::ExclusiveGpu(2))]),
        );

        core.require_managed_language_startup(&runtime, 2).unwrap();
        assert!(core.require_managed_language_startup(&runtime, 1).is_err());
        assert!(core
            .require_managed_language_startup(
                &RuntimeIdentity::new(Lane::Language, "other-runtime").unwrap(),
                2,
            )
            .is_err());
        assert!(core
            .require_managed_language_startup(
                &RuntimeIdentity::new(Lane::Media, "llama-swap").unwrap(),
                2,
            )
            .is_err());

        let extra_runtime = RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap();
        let multiple_data = TempDir::new().unwrap();
        let multiple = managed_language_core(
            root.path(),
            multiple_data.path(),
            BTreeMap::from([
                (runtime.clone(), ResourceClaim::ExclusiveGpu(2)),
                (extra_runtime, ResourceClaim::None),
            ]),
        );
        assert!(multiple
            .require_managed_language_startup(&runtime, 2)
            .is_err());
        assert_eq!(core.admission(), AdmissionState::ReconciliationRequired);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_readiness_requires_exact_route_clean_history_and_fresh_source() {
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let profile = ProfileKey::new(Lane::Language, "fixture-model").unwrap();
        let readiness =
            super::super::engine_health::RouterModelReadinessObservation::test_for_model(
                "fixture-model",
            );

        let root = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        let mut core = managed_language_core(
            root.path(),
            data.path(),
            BTreeMap::from([(runtime.clone(), ResourceClaim::ExclusiveGpu(2))]),
        );
        core.require_managed_language_startup(&runtime, 2).unwrap();
        assert!(core
            .confirm_managed_language_ready(
                &runtime,
                2,
                &profile,
                managed_tasks(),
                &super::super::engine_health::RouterModelReadinessObservation::test_for_model(
                    "other-model",
                ),
            )
            .is_err());
        core.confirm_managed_language_ready(&runtime, 2, &profile, managed_tasks(), &readiness)
            .unwrap();
        assert_eq!(core.admission(), AdmissionState::Ready);

        let drift_root = TempDir::new().unwrap();
        let drift_data = TempDir::new().unwrap();
        let mut drifted = managed_language_core(
            drift_root.path(),
            drift_data.path(),
            BTreeMap::from([(runtime.clone(), ResourceClaim::ExclusiveGpu(2))]),
        );
        std::fs::write(
            drift_root.path().join("language/llama-swap.yaml"),
            "models:\n  replacement-model:\n    cmd: llama-server\n",
        )
        .unwrap();
        assert!(drifted
            .confirm_managed_language_ready(&runtime, 2, &profile, managed_tasks(), &readiness,)
            .is_err());
        assert_eq!(drifted.admission(), AdmissionState::ReconciliationRequired);

        let retained_root = TempDir::new().unwrap();
        let retained_data = TempDir::new().unwrap();
        let mut retained = managed_language_core(
            retained_root.path(),
            retained_data.path(),
            BTreeMap::from([(runtime.clone(), ResourceClaim::ExclusiveGpu(2))]),
        );
        retained
            .history
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: "l3ms".into(),
                profile: Some("fixture-model".into()),
                task: Some("chat".into()),
                request_id: Some("retained-queued".into()),
            })
            .unwrap();
        assert!(retained
            .confirm_managed_language_ready(&runtime, 2, &profile, managed_tasks(), &readiness,)
            .is_err());
        assert_eq!(retained.admission(), AdmissionState::ReconciliationRequired);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn generic_reconciliation_remains_compatible_for_one_llama_swap_binding() {
        let root = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let mut core = managed_language_core(
            root.path(),
            data.path(),
            BTreeMap::from([(runtime, ResourceClaim::ExclusiveGpu(2))]),
        );
        core.confirm_reconciled().unwrap();
        assert_eq!(core.admission(), AdmissionState::Ready);
    }

    #[test]
    fn runtime_close_refuses_only_selected_submissions_without_persisting_them() {
        let data = TempDir::new().unwrap();
        let mut coordinator = core(data.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let media_runtime = RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap();
        let _close = coordinator.close_runtime_admission(&media_runtime).unwrap();

        assert_eq!(
            coordinator.submit_checked(
                "closed-media".into(),
                media(),
                Duration::from_secs(5),
                Instant::now(),
            ),
            Err(SubmissionError::RuntimeAdmissionClosed)
        );
        assert_eq!(coordinator.request_state("closed-media"), None);
        assert!(coordinator.history().records().is_empty());
        assert_eq!(coordinator.queued_count(), 0);
        assert!(!data.path().join("jobs.json").exists());

        coordinator
            .submit(
                "open-language".into(),
                chat(),
                Duration::from_secs(5),
                Instant::now(),
            )
            .unwrap();
        assert_eq!(coordinator.queued_count(), 1);
        assert_eq!(coordinator.request_state("open-language"), Some("queued"));
        let media_state = coordinator.runtime_drain_state(&media_runtime).unwrap();
        assert_eq!(media_state.pending_jobs(), 0);
        assert!(media_state.admission_closed());
        let language_runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let language_state = coordinator.runtime_drain_state(&language_runtime).unwrap();
        assert_eq!(language_state.pending_jobs(), 1);
        assert!(!language_state.admission_closed());
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(data.path().join("jobs.json")).unwrap()
            )
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn closed_runtime_continues_scheduling_accepted_work_through_cancellation() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap();
        let now = Instant::now();
        let first_id = coordinator
            .submit(
                "first-accepted".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let second_id = coordinator
            .submit(
                "second-accepted".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let _close = coordinator.close_runtime_admission(&runtime).unwrap();

        let initial = coordinator.runtime_drain_state(&runtime).unwrap();
        assert_eq!(initial.pending_jobs(), 2);
        assert_eq!(initial.active_jobs(), 0);
        assert!(!initial.is_drained());

        let first = coordinator.reserve_next(now).unwrap().unwrap();
        assert_eq!(first.job_id, first_id);
        coordinator.cancel(first_id).unwrap();
        assert!(first.cancellation.is_cancelled());
        let cancelling = coordinator.runtime_drain_state(&runtime).unwrap();
        assert_eq!(cancelling.pending_jobs(), 1);
        assert_eq!(cancelling.active_jobs(), 1);
        coordinator
            .complete(
                first_id,
                Completion::Cancelled,
                Vec::new(),
                now + Duration::from_millis(1),
            )
            .unwrap();

        let second = coordinator
            .reserve_next(now + Duration::from_millis(2))
            .unwrap()
            .unwrap();
        assert_eq!(second.job_id, second_id);
        coordinator.cancel(second_id).unwrap();
        coordinator
            .complete(
                second_id,
                Completion::Cancelled,
                Vec::new(),
                now + Duration::from_millis(3),
            )
            .unwrap();
        assert!(coordinator
            .reserve_next(now + Duration::from_millis(4))
            .unwrap()
            .is_none());

        let drained = coordinator.runtime_drain_state(&runtime).unwrap();
        assert_eq!(drained.pending_jobs(), 0);
        assert_eq!(drained.active_jobs(), 0);
        assert_eq!(drained.unresolved_retained_jobs(), 0);
        assert!(drained.admission_closed());
        assert!(drained.is_drained());
        assert_eq!(
            coordinator.submit_checked(
                "too-late".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            ),
            Err(SubmissionError::RuntimeAdmissionClosed)
        );
    }

    #[test]
    fn scoped_reservation_skips_other_runtime_fifo_work_and_drains_only_closed_runtime() {
        let data = TempDir::new().unwrap();
        let mut coordinator = core(data.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap();
        let now = Instant::now();
        coordinator
            .submit(
                "other-runtime-first".into(),
                chat(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let first_target = coordinator
            .submit("target-first".into(), media(), Duration::from_secs(30), now)
            .unwrap();
        let second_target = coordinator
            .submit(
                "target-second".into(),
                media(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let close = coordinator.close_runtime_admission(&runtime).unwrap();

        let first = coordinator
            .reserve_next_for_runtime(&close, now)
            .unwrap()
            .unwrap();
        assert_eq!(first.job_id, first_target);
        assert_eq!(first.request_id, "target-first");
        assert_eq!(first.route.runtime, runtime);
        assert_eq!(coordinator.queued_count(), 2);
        assert_eq!(
            coordinator
                .runtime_drain_state(&RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap())
                .unwrap()
                .pending_jobs(),
            1
        );

        coordinator
            .begin_dispatch(&first, now + Duration::from_millis(1))
            .unwrap();
        coordinator
            .complete(
                first_target,
                Completion::Failed,
                Vec::new(),
                now + Duration::from_millis(2),
            )
            .unwrap();

        let second = coordinator
            .reserve_next_for_runtime(&close, now + Duration::from_millis(3))
            .unwrap()
            .unwrap();
        assert_eq!(second.job_id, second_target);
        assert_eq!(second.route.runtime, runtime);
        coordinator.cancel(second_target).unwrap();
        coordinator
            .complete(
                second_target,
                Completion::Cancelled,
                Vec::new(),
                now + Duration::from_millis(4),
            )
            .unwrap();
        assert!(coordinator
            .reserve_next_for_runtime(&close, now + Duration::from_millis(5))
            .unwrap()
            .is_none());

        let target_state = coordinator.runtime_drain_state(&runtime).unwrap();
        assert_eq!(target_state.pending_jobs(), 0);
        assert_eq!(target_state.active_jobs(), 0);
        assert!(target_state.is_drained());
        assert_eq!(
            coordinator.request_state("other-runtime-first"),
            Some("queued")
        );
        let other = coordinator
            .reserve_next(now + Duration::from_millis(6))
            .unwrap()
            .unwrap();
        assert_eq!(other.request_id, "other-runtime-first");
    }

    #[test]
    fn scoped_reservation_respects_conflicting_active_resources() {
        let data = TempDir::new().unwrap();
        let mut coordinator = core(data.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap();
        let now = Instant::now();
        let other_id = coordinator
            .submit(
                "active-other-runtime".into(),
                chat(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let target_id = coordinator
            .submit(
                "blocked-target-runtime".into(),
                media(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let close = coordinator.close_runtime_admission(&runtime).unwrap();

        let active_other = coordinator.reserve_next(now).unwrap().unwrap();
        assert_eq!(active_other.job_id, other_id);
        assert!(coordinator
            .reserve_next_for_runtime(&close, now + Duration::from_millis(1))
            .unwrap()
            .is_none());
        assert_eq!(coordinator.queued_count(), 1);
        assert_eq!(coordinator.active_count(), 1);

        coordinator.cancel(other_id).unwrap();
        coordinator
            .complete(
                other_id,
                Completion::Cancelled,
                Vec::new(),
                now + Duration::from_millis(2),
            )
            .unwrap();
        let target = coordinator
            .reserve_next_for_runtime(&close, now + Duration::from_millis(3))
            .unwrap()
            .unwrap();
        assert_eq!(target.job_id, target_id);
        coordinator.cancel(target_id).unwrap();
        coordinator
            .complete(
                target_id,
                Completion::Cancelled,
                Vec::new(),
                now + Duration::from_millis(4),
            )
            .unwrap();
    }

    #[test]
    fn scoped_reservation_expires_only_selected_runtime_work() {
        let data = TempDir::new().unwrap();
        let mut coordinator = core(data.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "audio.cpp").unwrap();
        let now = Instant::now();
        coordinator
            .submit(
                "expired-target".into(),
                media(),
                Duration::from_secs(1),
                now,
            )
            .unwrap();
        coordinator
            .submit("other-runtime".into(), chat(), Duration::from_secs(30), now)
            .unwrap();
        let close = coordinator.close_runtime_admission(&runtime).unwrap();

        assert!(coordinator
            .reserve_next_for_runtime(&close, now + Duration::from_secs(2))
            .unwrap()
            .is_none());
        assert_eq!(coordinator.request_state("expired-target"), Some("failed"));
        assert_eq!(coordinator.request_state("other-runtime"), Some("queued"));
        assert_eq!(coordinator.queued_count(), 1);
        let other = coordinator
            .reserve_next(now + Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(other.request_id, "other-runtime");
    }

    #[test]
    fn scoped_reservation_quarantines_and_retains_expired_work_if_persistence_fails() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap();
        let now = Instant::now();
        coordinator
            .submit(
                "expired-but-retained".into(),
                DispatchFixture::request(),
                Duration::from_secs(1),
                now,
            )
            .unwrap();
        let close = coordinator.close_runtime_admission(&runtime).unwrap();

        let jobs_path = fixture.root.path().join("jobs.json");
        std::fs::remove_file(&jobs_path).unwrap();
        std::fs::create_dir(&jobs_path).unwrap();
        assert!(coordinator
            .reserve_next_for_runtime(&close, now + Duration::from_secs(2))
            .is_err());

        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.queued_count(), 1);
        assert_eq!(coordinator.active_count(), 0);
        assert_eq!(
            coordinator.request_state("expired-but-retained"),
            Some("queued")
        );
        let drain = coordinator.runtime_drain_state(&runtime).unwrap();
        assert_eq!(drain.pending_jobs(), 1);
        assert!(drain.admission_closed());
        assert!(!drain.is_drained());
    }

    #[test]
    fn stale_runtime_close_token_cannot_reopen_a_newer_close_generation() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap();
        let now = Instant::now();
        let job_id = coordinator
            .submit(
                "accepted-before-stale-close".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let stale = coordinator.close_runtime_admission(&runtime).unwrap();
        let current = coordinator.close_runtime_admission(&runtime).unwrap();

        assert!(coordinator.reserve_next_for_runtime(&stale, now).is_err());
        assert_eq!(coordinator.queued_count(), 1);
        assert_eq!(
            coordinator.reopen_runtime_admission(stale),
            Err(RuntimeAdmissionError::StaleCloseToken)
        );
        assert!(coordinator
            .runtime_drain_state(&runtime)
            .unwrap()
            .admission_closed());
        assert_eq!(
            coordinator.submit_checked(
                "still-closed".into(),
                DispatchFixture::request(),
                Duration::from_secs(5),
                Instant::now(),
            ),
            Err(SubmissionError::RuntimeAdmissionClosed)
        );

        let ticket = coordinator
            .reserve_next_for_runtime(&current, now)
            .unwrap()
            .unwrap();
        assert_eq!(ticket.job_id, job_id);
        coordinator
            .complete(job_id, Completion::Cancelled, Vec::new(), now)
            .unwrap();
        coordinator.reopen_runtime_admission(current).unwrap();
        assert!(!coordinator
            .runtime_drain_state(&runtime)
            .unwrap()
            .admission_closed());
        assert!(coordinator
            .submit(
                "reopened".into(),
                DispatchFixture::request(),
                Duration::from_secs(5),
                Instant::now(),
            )
            .is_ok());
    }

    #[test]
    fn scoped_reservation_rejects_close_token_from_another_core_with_same_generation() {
        let first_fixture = DispatchFixture::new();
        let second_fixture = DispatchFixture::new();
        let mut first_core = first_fixture.coordinator();
        let mut second_core = second_fixture.coordinator();
        first_core.confirm_reconciled().unwrap();
        second_core.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap();
        let now = Instant::now();
        first_core
            .submit(
                "first-core-pending".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        second_core
            .submit(
                "second-core-pending".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let first_close = first_core.close_runtime_admission(&runtime).unwrap();
        let second_close = second_core.close_runtime_admission(&runtime).unwrap();

        assert!(second_core
            .reserve_next_for_runtime(&first_close, now)
            .is_err());
        assert_eq!(second_core.queued_count(), 1);
        let own_ticket = second_core
            .reserve_next_for_runtime(&second_close, now)
            .unwrap()
            .unwrap();
        assert_eq!(own_ticket.request_id, "second-core-pending");
        assert_eq!(first_core.queued_count(), 1);
    }

    #[test]
    fn same_core_quarantine_preserves_runtime_close_and_pending_work_can_resume() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap();
        let now = Instant::now();
        let job_id = coordinator
            .submit(
                "accepted-before-close".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let close = coordinator.close_runtime_admission(&runtime).unwrap();
        coordinator.quarantine();

        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert!(coordinator.reserve_next(now).is_err());
        assert!(coordinator.reserve_next_for_runtime(&close, now).is_err());
        assert!(coordinator
            .runtime_drain_state(&runtime)
            .unwrap()
            .admission_closed());

        coordinator.confirm_reconciled().unwrap();
        assert!(coordinator
            .runtime_drain_state(&runtime)
            .unwrap()
            .admission_closed());
        let ticket = coordinator
            .reserve_next_for_runtime(&close, now)
            .unwrap()
            .unwrap();
        assert_eq!(ticket.job_id, job_id);
        coordinator
            .complete(job_id, Completion::Cancelled, Vec::new(), now)
            .unwrap();
        assert!(coordinator
            .runtime_drain_state(&runtime)
            .unwrap()
            .is_drained());
    }

    #[test]
    fn restart_restores_no_close_token_and_interrupted_history_is_not_drained() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        coordinator.confirm_reconciled().unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap();
        let now = Instant::now();
        coordinator
            .submit(
                "interrupted-on-restart".into(),
                DispatchFixture::request(),
                Duration::from_secs(30),
                now,
            )
            .unwrap();
        let _close = coordinator.close_runtime_admission(&runtime).unwrap();
        drop(coordinator);

        // The durable row stores a profile, not its original runtime binding.
        // A later manifest change must not hide potentially live old work.
        fixture.rewrite_media("moved-runtime", "scripts/runner.py");

        let mut restarted = fixture.coordinator();
        assert_eq!(
            restarted.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(restarted.queued_count(), 0);
        assert_eq!(
            restarted.request_state("interrupted-on-restart"),
            Some("interrupted")
        );
        let drain = restarted.runtime_drain_state(&runtime).unwrap();
        assert!(!drain.admission_closed());
        assert_eq!(drain.unresolved_retained_jobs(), 1);
        assert!(!drain.is_drained());
        assert!(restarted.reserve_next(now).is_err());
        assert_eq!(
            restarted.submit_checked(
                "no-replay-before-reconcile".into(),
                DispatchFixture::request(),
                Duration::from_secs(5),
                now,
            ),
            Err(SubmissionError::Unavailable)
        );
    }

    fn cpu_fixture_core(fixture: &DispatchFixture, resource: ResourceClaim) -> CoordinatorCore {
        let root = fixture.root.path();
        std::fs::write(root.join("decisions/decision-runtimes.json"), serde_json::json!({
            "version": "1.0", "profiles": [
                {"id":"cpu", "name":"CPU", "engine":"nagi", "type":"local", "model":"fixture", "device":"cpu", "entrypoint":"runner.py"},
                {"id":"gpu", "name":"GPU", "engine":"nagi", "type":"local", "model":"fixture", "device":"cuda", "entrypoint":"runner.py"}
            ]
        }).to_string()).unwrap();
        let catalog = Catalog::load_from_roots(
            root.join("language"),
            root.join("media"),
            root.join("decisions"),
        )
        .unwrap();
        CoordinatorCore::load(
            catalog,
            BTreeMap::from([(
                RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap(),
                resource,
            )]),
            root,
            root,
            8,
        )
        .unwrap()
    }

    fn cpu_request(profile: &str) -> InferenceRequest {
        let mut request = hosted();
        if let InferenceRequest::Decision { profile: key, .. } = &mut request {
            *key = ProfileKey::new(Lane::Decisions, profile).unwrap();
        }
        request
    }

    #[test]
    fn close_token_cannot_reopen_a_different_core_with_the_same_generation() {
        let first_data = TempDir::new().unwrap();
        let second_data = TempDir::new().unwrap();
        let mut first = core(first_data.path(), 8);
        let mut second = core(second_data.path(), 8);
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        assert!(!first.runtime_drain_state(&runtime).unwrap().is_drained());
        let first_close = first.close_runtime_admission(&runtime).unwrap();
        let second_close = second.close_runtime_admission(&runtime).unwrap();
        assert_eq!(
            second.reopen_runtime_admission(first_close),
            Err(RuntimeAdmissionError::StaleCloseToken)
        );
        assert!(second
            .runtime_drain_state(&runtime)
            .unwrap()
            .admission_closed());
        second.reopen_runtime_admission(second_close).unwrap();
        assert!(!second.runtime_drain_state(&runtime).unwrap().is_drained());
    }

    #[test]
    fn cpu_admission_refuses_gpu_profiles_even_when_the_runtime_binding_is_shared() {
        let fixture = DispatchFixture::new();
        let mut core = cpu_fixture_core(&fixture, ResourceClaim::None);
        core.confirm_cpu_decisions_ready().unwrap();
        assert!(core
            .submit(
                "cpu".into(),
                cpu_request("cpu"),
                Duration::from_secs(5),
                Instant::now()
            )
            .is_ok());
        assert_eq!(
            core.submit_checked(
                "gpu".into(),
                cpu_request("gpu"),
                Duration::from_secs(5),
                Instant::now()
            ),
            Err(SubmissionError::Unavailable)
        );
        assert_eq!(core.queued_count(), 1);
        assert_eq!(core.request_state("gpu"), None);
    }

    #[test]
    fn cpu_startup_rejects_gpu_bindings_and_unresolved_restart_history() {
        let fixture = DispatchFixture::new();
        let mut wrong = cpu_fixture_core(&fixture, ResourceClaim::ExclusiveGpu(0));
        assert!(wrong.confirm_cpu_decisions_ready().is_err());
        assert_eq!(wrong.admission(), AdmissionState::ReconciliationRequired);
        drop(wrong);
        let mut core = cpu_fixture_core(&fixture, ResourceClaim::None);
        core.confirm_cpu_decisions_ready().unwrap();
        core.submit(
            "unresolved".into(),
            cpu_request("cpu"),
            Duration::from_secs(5),
            Instant::now(),
        )
        .unwrap();
        drop(core);
        let mut restarted = cpu_fixture_core(&fixture, ResourceClaim::None);
        assert_eq!(restarted.request_state("unresolved"), Some("interrupted"));
        assert!(restarted.confirm_cpu_decisions_ready().is_err());
        assert_eq!(
            restarted.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(restarted.queued_count(), 0);
    }

    #[test]
    fn begin_dispatch_revalidates_and_durably_marks_the_stable_route_running() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve(&mut coordinator, now);

        coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap();

        assert_eq!(coordinator.admission(), AdmissionState::Ready);
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            persisted_state(fixture.root.path(), "dispatch-fixture").as_deref(),
            Some("running")
        );
        assert!(
            coordinator
                .history()
                .records()
                .iter()
                .any(|job| job.id == job_id
                    && job.metadata.result_state.as_deref() == Some("running"))
        );
    }

    #[test]
    fn comfy_candidate_is_persisted_after_the_final_gate_before_a_permit_is_returned() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (fixture, coordinator, job_id, ticket, now) = started_comfy_dispatch();
        let mut coordinator = coordinator;
        let permit = coordinator
            .record_comfy_dispatch_candidate(&ticket, CANDIDATE.into(), now)
            .unwrap();

        assert_eq!(permit.job_id(), job_id);
        assert_eq!(permit.request_id(), "comfy-dispatch-fixture");
        assert_eq!(permit.candidate_id(), CANDIDATE);
        assert_eq!(permit.route(), &ticket.route);
        assert!(coordinator.accepts_comfy_dispatch_permit(&permit));
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(coordinator.admission(), AdmissionState::Ready);

        let rows: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.root.path().join("jobs.json")).unwrap())
                .unwrap();
        let row = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["request_id"] == "comfy-dispatch-fixture")
            .unwrap();
        assert_eq!(row["comfy_dispatch_candidate"], CANDIDATE);
        assert!(row.get("backend_id").is_none());
        assert!(row.get("comfy_service_generation").is_none());
        assert!(!rows.to_string().contains("FixtureNode"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_comfy_candidate_commits_exact_owner_generation_before_permit() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (fixture, mut coordinator, job_id, ticket, now) = started_comfy_dispatch();
        coordinator.managed_comfy_profiles =
            Some(BTreeSet::from([ticket.request.profile().clone()]));
        let generation = OwnedServiceGenerationIdentity::for_test_comfy_supervisor().unwrap();
        let permit = coordinator
            .record_managed_comfy_dispatch_candidate(
                &ticket,
                CANDIDATE.into(),
                generation.clone(),
                now,
            )
            .unwrap();

        assert_eq!(permit.job_id(), job_id);
        assert_eq!(permit.request_id(), "comfy-dispatch-fixture");
        assert_eq!(permit.candidate_id(), CANDIDATE);
        assert_eq!(permit.route(), &ticket.route);
        assert!(permit.matches_candidate(CANDIDATE, &ticket.route));
        assert!(permit.matches_request("comfy-dispatch-fixture", &ticket.request, &ticket.route));
        assert!(permit.matches_generation(&generation));
        let different_generation =
            OwnedServiceGenerationIdentity::for_test_comfy_supervisor_with_main_pid(9876).unwrap();
        assert!(!permit.matches_generation(&different_generation));
        assert!(coordinator.accepts_managed_comfy_dispatch_permit(&permit));
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(coordinator.admission(), AdmissionState::Ready);
        assert_eq!(
            format!("{permit:?}"),
            "ManagedComfyDispatchPermit([redacted])"
        );

        let rows: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.root.path().join("jobs.json")).unwrap())
                .unwrap();
        let row = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["request_id"] == "comfy-dispatch-fixture")
            .unwrap();
        assert_eq!(row["comfy_dispatch_candidate"], CANDIDATE);
        assert_eq!(
            row["comfy_service_generation"]["schema_version"],
            serde_json::json!(1)
        );
        assert_eq!(
            row["comfy_service_generation"]["intent"]["runtime_name"],
            "ComfyUI"
        );
        assert_eq!(
            row["comfy_service_generation"]["generation"]["main_pid"],
            serde_json::json!(2345)
        );
        assert!(row.get("backend_id").is_none());
        let persisted = rows.to_string();
        assert!(!persisted.contains("FixtureNode"));
        assert!(!persisted.contains("workflow-payload-secret"));
        assert!(!persisted.contains("argv"));
        assert!(!persisted.contains("environment"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_comfy_worker_handoff_persists_before_one_reply_and_late_cancel_revokes_permit() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (fixture, mut coordinator, job_id, ticket, _now) = started_comfy_dispatch();
        coordinator.managed_comfy_profiles =
            Some(BTreeSet::from([ticket.request.profile().clone()]));
        let generation = OwnedServiceGenerationIdentity::for_test_comfy_supervisor().unwrap();
        let cancellation = RequestCancellation::new();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);

        let permit = std::thread::scope(|scope| {
            let worker_cancellation = cancellation.clone();
            let worker_generation = generation.clone();
            let worker_ticket = &ticket;
            let worker = scope.spawn(move || {
                let context = super::super::runtime::OperationContext::new(
                    Instant::now() + Duration::from_secs(1),
                    &worker_cancellation,
                );
                super::super::managed_session::request_comfy_permit(
                    &sender,
                    worker_ticket,
                    CANDIDATE.into(),
                    worker_generation,
                    &context,
                )
            });
            match receiver.recv_timeout(Duration::from_secs(1)).unwrap() {
                super::super::managed_session::ManagedWorkerEvent::ComfyCandidate {
                    ticket: event_ticket,
                    candidate,
                    generation: event_generation,
                    reply,
                } => {
                    assert_eq!(event_ticket.job_id, job_id);
                    assert_eq!(candidate, CANDIDATE);
                    assert_eq!(*event_generation, generation);
                    let permit = coordinator
                        .record_managed_comfy_dispatch_candidate(
                            &event_ticket,
                            candidate,
                            *event_generation,
                            Instant::now(),
                        )
                        .unwrap();
                    let rows: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(fixture.root.path().join("jobs.json")).unwrap(),
                    )
                    .unwrap();
                    let row = rows
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|row| row["request_id"].as_str() == Some(ticket.request_id.as_str()))
                        .unwrap();
                    assert_eq!(row["comfy_dispatch_candidate"], CANDIDATE);
                    assert!(row.get("backend_id").is_none());
                    assert!(row.get("comfy_service_generation").is_some());
                    reply.send(Some(permit)).unwrap();
                    worker.join().unwrap().unwrap()
                }
                _ => panic!("expected one Comfy candidate handoff"),
            }
        });

        assert!(coordinator.accepts_managed_comfy_dispatch_permit(&permit));
        let active = coordinator.active.get_mut(&job_id).unwrap();
        let deadline = active.deadline;
        active.deadline = Instant::now() - Duration::from_millis(1);
        assert!(!coordinator.accepts_managed_comfy_dispatch_permit(&permit));
        coordinator.active.get_mut(&job_id).unwrap().deadline = deadline;
        assert!(coordinator.accepts_managed_comfy_dispatch_permit(&permit));
        ticket.cancellation.cancel();
        assert!(!coordinator.accepts_managed_comfy_dispatch_permit(&permit));
        assert_eq!(coordinator.active_count(), 1);
        assert!(matches!(
            receiver.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty | std::sync::mpsc::TryRecvError::Disconnected)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_comfy_lost_permit_reply_keeps_durable_candidate_without_replay() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (fixture, mut coordinator, job_id, ticket, _now) = started_comfy_dispatch();
        coordinator.managed_comfy_profiles =
            Some(BTreeSet::from([ticket.request.profile().clone()]));
        let generation = OwnedServiceGenerationIdentity::for_test_comfy_supervisor().unwrap();
        let cancellation = RequestCancellation::new();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);

        let worker_result = std::thread::scope(|scope| {
            let worker_cancellation = cancellation.clone();
            let worker_generation = generation.clone();
            let worker_ticket = &ticket;
            let worker = scope.spawn(move || {
                let context = super::super::runtime::OperationContext::new(
                    Instant::now() + Duration::from_secs(1),
                    &worker_cancellation,
                );
                super::super::managed_session::request_comfy_permit(
                    &sender,
                    worker_ticket,
                    CANDIDATE.into(),
                    worker_generation,
                    &context,
                )
            });
            match receiver.recv_timeout(Duration::from_secs(1)).unwrap() {
                super::super::managed_session::ManagedWorkerEvent::ComfyCandidate {
                    ticket: event_ticket,
                    candidate,
                    generation: event_generation,
                    reply,
                } => {
                    coordinator
                        .record_managed_comfy_dispatch_candidate(
                            &event_ticket,
                            candidate,
                            *event_generation,
                            Instant::now(),
                        )
                        .unwrap();
                    let rows: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(fixture.root.path().join("jobs.json")).unwrap(),
                    )
                    .unwrap();
                    let row = rows
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|row| row["request_id"].as_str() == Some(ticket.request_id.as_str()))
                        .unwrap();
                    assert_eq!(row["comfy_dispatch_candidate"], CANDIDATE);
                    assert!(row.get("backend_id").is_none());
                    drop(reply); // Worker observes loss; this event is never resent.
                    worker.join().unwrap()
                }
                _ => panic!("expected one Comfy candidate handoff"),
            }
        });

        assert!(worker_result.is_err());
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            coordinator
                .history()
                .records()
                .iter()
                .find(|job| job.id == job_id)
                .unwrap()
                .metadata
                .comfy_dispatch_candidate
                .as_deref(),
            Some(CANDIDATE)
        );
        assert!(matches!(
            receiver.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty | std::sync::mpsc::TryRecvError::Disconnected)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_comfy_ack_commits_after_late_cancel_and_exact_duplicate_is_read_only() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (fixture, mut coordinator, _job_id, ticket, now) = started_comfy_dispatch();
        coordinator.managed_comfy_profiles =
            Some(BTreeSet::from([ticket.request.profile().clone()]));
        let generation = OwnedServiceGenerationIdentity::for_test_comfy_supervisor().unwrap();
        coordinator
            .record_managed_comfy_dispatch_candidate(
                &ticket,
                CANDIDATE.into(),
                generation.clone(),
                now,
            )
            .unwrap();

        let acknowledgement = super::super::comfy::test_managed_comfy_acknowledgement(
            &ticket,
            generation.clone(),
            CANDIDATE,
        )
        .expect("fixture receipt should bind the active request and owned generation");
        ticket.cancellation.cancel();
        coordinator
            .accept_managed_comfy_backend_acknowledgement(acknowledgement)
            .unwrap();
        let path = fixture.root.path().join("jobs.json");
        let committed = std::fs::read(&path).unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&committed).unwrap();
        let row = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["request_id"] == "comfy-dispatch-fixture")
            .unwrap();
        assert_eq!(row["comfy_dispatch_candidate"], CANDIDATE);
        assert_eq!(row["backend_id"], CANDIDATE);
        assert!(row.get("comfy_service_generation").is_some());

        let duplicate = super::super::comfy::test_managed_comfy_acknowledgement(
            &ticket,
            generation.clone(),
            CANDIDATE,
        )
        .unwrap();
        coordinator
            .accept_managed_comfy_backend_acknowledgement(duplicate)
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), committed);
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(coordinator.admission(), AdmissionState::Ready);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_comfy_ack_rejects_a_different_owned_generation() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (_fixture, mut coordinator, job_id, ticket, now) = started_comfy_dispatch();
        coordinator.managed_comfy_profiles =
            Some(BTreeSet::from([ticket.request.profile().clone()]));
        let generation = OwnedServiceGenerationIdentity::for_test_comfy_supervisor().unwrap();
        coordinator
            .record_managed_comfy_dispatch_candidate(
                &ticket,
                CANDIDATE.into(),
                generation.clone(),
                now,
            )
            .unwrap();
        let stale_generation =
            OwnedServiceGenerationIdentity::for_test_comfy_supervisor_with_main_pid(9876).unwrap();
        let acknowledgement = super::super::comfy::test_managed_comfy_acknowledgement(
            &ticket,
            stale_generation,
            CANDIDATE,
        )
        .unwrap();

        assert!(coordinator
            .accept_managed_comfy_backend_acknowledgement(acknowledgement)
            .is_err());
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        let row = coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .unwrap();
        assert_eq!(
            row.metadata.comfy_dispatch_candidate.as_deref(),
            Some(CANDIDATE)
        );
        assert!(row.metadata.backend_id.is_none());
        assert!(row.metadata.comfy_service_generation.is_some());
    }

    #[test]
    fn comfy_candidate_requires_begin_dispatch_and_rejects_wrong_route_job_or_uuid() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";

        let fixture = DispatchFixture::new_comfy();
        let mut coordinator = fixture.comfy_coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve_comfy(&mut coordinator, now);
        assert!(matches!(
            coordinator.record_comfy_dispatch_candidate(&ticket, CANDIDATE.into(), now),
            Err(ComfyCandidateError::FinalDispatchGateMissing)
        ));
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .unwrap()
            .metadata
            .comfy_dispatch_candidate
            .is_none());

        for scenario in ["wrong-route", "wrong-job", "invalid-uuid"] {
            let (_fixture, mut coordinator, job_id, mut ticket, now) = started_comfy_dispatch();
            let (candidate, expected) = match scenario {
                "wrong-route" => {
                    ticket.route.native_route = NativeRoute::MediaScript {
                        path: PathBuf::from("workflows/other.json"),
                    };
                    (CANDIDATE, ComfyCandidateError::TicketMismatch)
                }
                "wrong-job" => {
                    ticket.job_id = ticket.job_id.saturating_add(1);
                    (CANDIDATE, ComfyCandidateError::ReservationUnavailable)
                }
                _ => (
                    "123E4567-E89B-12D3-A456-426614174000",
                    ComfyCandidateError::CandidateRejected,
                ),
            };
            assert!(
                matches!(
                    coordinator.record_comfy_dispatch_candidate(
                        &ticket,
                        candidate.into(),
                        now
                    ),
                    Err(error) if error == expected
                ),
                "scenario {scenario}"
            );
            assert_eq!(
                coordinator.admission(),
                AdmissionState::ReconciliationRequired
            );
            assert_eq!(coordinator.active_count(), 1);
            assert!(coordinator
                .history()
                .records()
                .iter()
                .find(|job| job.id == job_id)
                .unwrap()
                .metadata
                .comfy_dispatch_candidate
                .is_none());
        }
    }

    #[test]
    fn comfy_candidate_refreshes_exact_route_and_rejects_non_comfy_media() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";

        let (fixture, mut coordinator, _job_id, ticket, now) = started_comfy_dispatch();
        fixture.write_comfy_manifest("comfyui", "workflows/fixture.json");
        assert!(matches!(
            coordinator.record_comfy_dispatch_candidate(&ticket, CANDIDATE.into(), now),
            Err(ComfyCandidateError::RouteChanged)
        ));
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );

        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve(&mut coordinator, now);
        coordinator.begin_dispatch(&ticket, now).unwrap();
        assert!(matches!(
            coordinator.record_comfy_dispatch_candidate(&ticket, CANDIDATE.into(), now),
            Err(ComfyCandidateError::UnsupportedRoute)
        ));
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .unwrap()
            .metadata
            .comfy_dispatch_candidate
            .is_none());
    }

    #[test]
    fn comfy_candidate_persistence_failure_quarantines_without_releasing_reservation() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (fixture, mut coordinator, job_id, ticket, now) = started_comfy_dispatch();
        let jobs_path = fixture.root.path().join("jobs.json");
        std::fs::remove_file(&jobs_path).unwrap();
        std::fs::create_dir(&jobs_path).unwrap();

        assert!(matches!(
            coordinator.record_comfy_dispatch_candidate(&ticket, CANDIDATE.into(), now),
            Err(ComfyCandidateError::PersistenceFailed)
        ));
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .unwrap()
            .metadata
            .comfy_dispatch_candidate
            .is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_comfy_persistence_failure_returns_no_permit_or_partial_association() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let (fixture, mut coordinator, job_id, ticket, now) = started_comfy_dispatch();
        coordinator.managed_comfy_profiles =
            Some(BTreeSet::from([ticket.request.profile().clone()]));
        let jobs_path = fixture.root.path().join("jobs.json");
        std::fs::remove_file(&jobs_path).unwrap();
        std::fs::create_dir(&jobs_path).unwrap();
        let generation = OwnedServiceGenerationIdentity::for_test_comfy_supervisor().unwrap();

        assert!(matches!(
            coordinator.record_managed_comfy_dispatch_candidate(
                &ticket,
                CANDIDATE.into(),
                generation,
                now
            ),
            Err(ComfyCandidateError::PersistenceFailed)
        ));
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        let row = coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .unwrap();
        assert!(row.metadata.comfy_dispatch_candidate.is_none());
        assert!(row.metadata.comfy_service_generation.is_none());
    }

    #[test]
    fn verified_comfy_backend_id_is_explicit_and_must_match_the_candidate() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";

        let (_fixture, mut coordinator, job_id, ticket, now) = started_comfy_dispatch();
        coordinator
            .record_comfy_dispatch_candidate(&ticket, CANDIDATE.into(), now)
            .unwrap();
        assert!(coordinator
            .attach_backend(job_id, "another-id".into())
            .is_err());
        let record = coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .unwrap();
        assert_eq!(
            record.metadata.comfy_dispatch_candidate.as_deref(),
            Some(CANDIDATE)
        );
        assert!(record.metadata.backend_id.is_none());
        assert_eq!(coordinator.active_count(), 1);

        let (_fixture, mut coordinator, job_id, ticket, now) = started_comfy_dispatch();
        coordinator
            .record_comfy_dispatch_candidate(&ticket, CANDIDATE.into(), now)
            .unwrap();
        coordinator
            .attach_backend(job_id, CANDIDATE.into())
            .unwrap();
        let record = coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == job_id)
            .unwrap();
        assert_eq!(record.metadata.backend_id.as_deref(), Some(CANDIDATE));
        assert_eq!(
            record.metadata.comfy_dispatch_candidate.as_deref(),
            Some(CANDIDATE)
        );
    }

    #[test]
    fn coordinator_strict_load_rejects_malformed_duplicate_or_truncated_history_unchanged() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let fixture = DispatchFixture::new_comfy();
        let catalog = Catalog::load_from_roots(
            fixture.root.path().join("language"),
            fixture.root.path().join("media"),
            fixture.root.path().join("decisions"),
        )
        .unwrap();
        let bindings = BTreeMap::from([(
            RuntimeIdentity::new(Lane::Media, "ComfyUI").unwrap(),
            ResourceClaim::ExclusiveGpu(0),
        )]);
        let malformed = serde_json::json!([{
            "name": "malformed",
            "started": "12:34:56",
            "elapsed": "-",
            "exit": "…",
            "mode": "cross-toolkit",
            "script_path": "",
            "toolkit": "l2m2",
            "profile": "fixture-profile",
            "task": "image",
            "request_id": "req-malformed",
            "comfy_dispatch_candidate": "123E4567-E89B-12D3-A456-426614174000"
        }]);
        let duplicate = serde_json::json!([
            {
                "name": "first",
                "started": "12:34:56",
                "elapsed": "-",
                "exit": "…",
                "mode": "cross-toolkit",
                "script_path": "",
                "toolkit": "l2m2",
                "profile": "fixture-profile",
                "task": "image",
                "request_id": "req-first",
                "result_state": "running",
                "comfy_dispatch_candidate": CANDIDATE
            },
            {
                "name": "second",
                "started": "12:34:56",
                "elapsed": "-",
                "exit": "…",
                "mode": "cross-toolkit",
                "script_path": "",
                "toolkit": "l2m2",
                "profile": "fixture-profile",
                "task": "image",
                "request_id": "req-second",
                "result_state": "running",
                "comfy_dispatch_candidate": CANDIDATE
            }
        ]);
        let truncated = (0..=crate::state_store::JOBS_MAX)
            .map(|index| {
                serde_json::json!({
                    "name": format!("legacy-{index}"),
                    "started": "12:34:56",
                    "elapsed": "0s",
                    "exit": "0"
                })
            })
            .collect::<Vec<_>>();
        let duplicate_candidate_key = br#"[{"name":"duplicate-candidate-key","started":"12:34:56","elapsed":"-","exit":"0","toolkit":"l2m2","profile":"fixture-profile","task":"image","request_id":"req-a","comfy_dispatch_candidate":"123e4567-e89b-12d3-a456-426614174000","comfy_dispatch_candidate":"123e4567-e89b-12d3-a456-426614174000"}]"#.to_vec();
        let duplicate_request_key = br#"[{"name":"duplicate-request-key","started":"12:34:56","elapsed":"-","exit":"0","toolkit":"l2m2","profile":"fixture-profile","task":"image","request_id":"req-a","request_id":"req-b","comfy_dispatch_candidate":"123e4567-e89b-12d3-a456-426614174000"}]"#.to_vec();
        let cases = vec![
            serde_json::to_vec(&malformed).unwrap(),
            serde_json::to_vec(&duplicate).unwrap(),
            serde_json::to_vec(&serde_json::Value::Array(truncated)).unwrap(),
            duplicate_candidate_key,
            duplicate_request_key,
        ];

        for bytes in cases {
            let data = TempDir::new().unwrap();
            std::fs::write(data.path().join("jobs.json"), &bytes).unwrap();
            assert!(CoordinatorCore::load(
                catalog.clone(),
                bindings.clone(),
                data.path(),
                fixture.root.path(),
                8,
            )
            .is_err());
            assert_eq!(std::fs::read(data.path().join("jobs.json")).unwrap(), bytes);
        }
    }

    #[test]
    fn device_only_manifest_change_quarantines_before_dispatch() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (_, ticket) = fixture.reserve(&mut coordinator, now);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.media_manifest).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = serde_json::json!("cuda");
        std::fs::write(&fixture.media_manifest, manifest.to_string()).unwrap();
        assert_eq!(
            coordinator.begin_dispatch(&ticket, now),
            Err(DispatchBeginError::RouteChanged)
        );
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            persisted_state(fixture.root.path(), "dispatch-fixture").as_deref(),
            Some("queued")
        );
    }

    #[test]
    fn changed_manifest_runtime_or_native_path_quarantines_and_retains_ticket() {
        for (runtime, script) in [
            ("changed-runtime", "scripts/runner.py"),
            ("fixture-runtime", "scripts/changed-private-route.py"),
        ] {
            let fixture = DispatchFixture::new();
            let mut coordinator = fixture.coordinator();
            let now = Instant::now();
            let (job_id, ticket) = fixture.reserve(&mut coordinator, now);
            fixture.rewrite_media(runtime, script);

            let error = coordinator
                .begin_dispatch(&ticket, now + Duration::from_secs(1))
                .unwrap_err();

            assert_eq!(error, DispatchBeginError::RouteChanged);
            assert_eq!(
                coordinator.admission(),
                AdmissionState::ReconciliationRequired
            );
            assert_eq!(coordinator.active_count(), 1);
            assert!(coordinator
                .history()
                .records()
                .iter()
                .any(|job| job.id == job_id
                    && job.metadata.result_state.as_deref() == Some("queued")));
            assert_eq!(
                persisted_state(fixture.root.path(), "dispatch-fixture").as_deref(),
                Some("queued")
            );
            assert!(!error.to_string().contains("changed-private-route.py"));
            assert!(!format!("{error:?}").contains("changed-private-route.py"));
        }
    }

    #[test]
    fn catalog_refresh_failure_is_redacted_quarantines_and_retains_ticket() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve(&mut coordinator, now);
        std::fs::write(&fixture.media_manifest, "malformed-private-manifest").unwrap();

        let error = coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap_err();

        assert_eq!(error, DispatchBeginError::CatalogUnavailable);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            coordinator
                .history()
                .records()
                .iter()
                .find(|job| job.id == job_id)
                .unwrap()
                .metadata
                .result_state
                .as_deref(),
            Some("queued")
        );
        assert!(!error.to_string().contains("malformed-private-manifest"));
        assert!(!format!("{error:?}").contains("malformed-private-manifest"));
    }

    #[test]
    fn changed_runtime_binding_quarantines_and_keeps_the_reservation() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve(&mut coordinator, now);
        coordinator.bindings.insert(
            RuntimeIdentity::new(Lane::Media, "fixture-runtime").unwrap(),
            ResourceClaim::None,
        );

        let error = coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap_err();

        assert_eq!(error, DispatchBeginError::BindingChanged);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .any(|job| job.id == job_id && job.metadata.result_state.as_deref() == Some("queued")));
    }

    #[test]
    fn invalid_ticket_request_quarantines_without_starting_or_releasing_work() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, mut ticket) = fixture.reserve(&mut coordinator, now);
        if let InferenceRequest::Media {
            input: MediaInput::Script { args, .. },
            ..
        } = &mut ticket.request
        {
            *args = vec!["oversized".into(); 129];
        }

        let error = coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap_err();

        assert_eq!(error, DispatchBeginError::TicketMismatch);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .any(|job| job.id == job_id && job.metadata.result_state.as_deref() == Some("queued")));
    }

    #[test]
    fn valid_payload_mutation_is_rejected_as_a_ticket_mismatch() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, mut ticket) = fixture.reserve(&mut coordinator, now);
        if let InferenceRequest::Media {
            input: MediaInput::Script { args, .. },
            ..
        } = &mut ticket.request
        {
            args[1] = "different but still valid input".into();
        }

        let error = coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap_err();

        assert_eq!(error, DispatchBeginError::TicketMismatch);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .any(|job| job.id == job_id && job.metadata.result_state.as_deref() == Some("queued")));
        assert!(!error
            .to_string()
            .contains("different but still valid input"));
        assert!(!format!("{error:?}").contains("different but still valid input"));
    }

    #[test]
    fn mismatched_history_association_quarantines_without_releasing_work() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve(&mut coordinator, now);
        coordinator
            .history
            .update_cross_toolkit(
                job_id,
                CrossToolkitJobUpdate {
                    result_state: Some("failed".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();

        let error = coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap_err();

        assert_eq!(error, DispatchBeginError::JobAssociationMismatch);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
    }

    #[test]
    fn running_persistence_failure_quarantines_and_retains_queued_reservation() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (job_id, ticket) = fixture.reserve(&mut coordinator, now);
        let blocked = fixture.root.path().join("not-a-directory");
        std::fs::write(&blocked, "private persistence failure").unwrap();
        coordinator.data_root = blocked;

        let error = coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap_err();

        assert_eq!(error, DispatchBeginError::PersistenceFailed);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .any(|job| job.id == job_id && job.metadata.result_state.as_deref() == Some("queued")));
    }

    #[test]
    fn begin_dispatch_refuses_cancelled_and_expired_reservations() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (cancelled_id, cancelled_ticket) = fixture.reserve(&mut coordinator, now);
        coordinator.cancel(cancelled_id).unwrap();
        assert_eq!(
            coordinator.begin_dispatch(&cancelled_ticket, now + Duration::from_secs(1)),
            Err(DispatchBeginError::Cancelled)
        );
        assert_eq!(coordinator.admission(), AdmissionState::Ready);
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            persisted_state(fixture.root.path(), "dispatch-fixture").as_deref(),
            Some("queued")
        );

        let expired_fixture = DispatchFixture::new();
        let mut expired = expired_fixture.coordinator();
        let now = Instant::now();
        let (expired_id, expired_ticket) = expired_fixture.reserve(&mut expired, now);
        assert_eq!(
            expired.begin_dispatch(&expired_ticket, expired_ticket.deadline),
            Err(DispatchBeginError::DeadlineExceeded)
        );
        assert_eq!(expired.admission(), AdmissionState::Ready);
        assert_eq!(expired.active_count(), 1);
        assert_eq!(
            persisted_state(expired_fixture.root.path(), "dispatch-fixture").as_deref(),
            Some("queued")
        );
        assert!(expired
            .history()
            .records()
            .iter()
            .any(|job| job.id == expired_id
                && job.metadata.result_state.as_deref() == Some("queued")));
    }

    #[test]
    fn begin_dispatch_rechecks_real_deadline_after_catalog_io() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        coordinator.confirm_reconciled().unwrap();
        let submitted_at = Instant::now();
        let job_id = coordinator
            .submit(
                "short-deadline".into(),
                DispatchFixture::request(),
                Duration::from_millis(80),
                submitted_at,
            )
            .unwrap();
        let ticket = coordinator.reserve_next(submitted_at).unwrap().unwrap();
        std::thread::sleep(Duration::from_millis(100));

        assert_eq!(
            coordinator.begin_dispatch(&ticket, submitted_at),
            Err(DispatchBeginError::DeadlineExceeded)
        );
        assert_eq!(coordinator.admission(), AdmissionState::Ready);
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator
            .history()
            .records()
            .iter()
            .any(|job| job.id == job_id && job.metadata.result_state.as_deref() == Some("queued")));
    }

    #[test]
    fn repeated_begin_dispatch_is_rejected_without_releasing_the_reservation() {
        let fixture = DispatchFixture::new();
        let mut coordinator = fixture.coordinator();
        let now = Instant::now();
        let (_job_id, ticket) = fixture.reserve(&mut coordinator, now);

        coordinator
            .begin_dispatch(&ticket, now + Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            coordinator.begin_dispatch(&ticket, now + Duration::from_secs(2)),
            Err(DispatchBeginError::AlreadyStarted)
        );
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            persisted_state(fixture.root.path(), "dispatch-fixture").as_deref(),
            Some("running")
        );
    }

    #[test]
    fn rpc_submit_is_durable_and_classifies_rejections_without_payloads() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 1);
        let now = Instant::now();
        let submit = |id: &str| ClientRequest::Submit {
            request_id: id.into(),
            request: chat(),
            timeout: Duration::from_secs(10),
        };
        assert_eq!(
            coordinator
                .handle_client_request(submit("one"), now)
                .message,
            wire_error("one".into(), ResponseErrorCode::Unavailable)
        );
        assert_eq!(coordinator.queued_count(), 0);
        coordinator.confirm_reconciled().unwrap();
        assert_eq!(
            coordinator
                .handle_client_request(submit("one"), now)
                .message,
            ResponseMessage::Accepted {
                request_id: "one".into(),
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued
            }
        );
        let persisted = std::fs::read_to_string(temp.path().join("jobs.json")).unwrap();
        assert!(persisted.contains("one"));
        assert!(!persisted.contains("REQUEST_PAYLOAD_MUST_NOT_BE_PERSISTED"));
        assert_eq!(
            coordinator
                .handle_client_request(submit("one"), now)
                .message,
            wire_error("one".into(), ResponseErrorCode::DuplicateRequest)
        );
        assert_eq!(
            coordinator
                .handle_client_request(submit("two"), now)
                .message,
            wire_error("two".into(), ResponseErrorCode::QueueFull)
        );
        assert_eq!(coordinator.queued_count(), 1);
    }

    #[test]
    fn rpc_cancel_preserves_active_reservation_and_never_relabels_terminal_work() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        let queued = coordinator
            .submit("queued".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Cancel {
                        request_id: "queued".into()
                    },
                    now
                )
                .message,
            ResponseMessage::Accepted {
                request_id: "queued".into(),
                operation: AcceptedOperation::Cancel,
                state: RequestStatus::Cancelled
            }
        );
        assert!(!coordinator
            .history()
            .records()
            .iter()
            .find(|job| job.id == queued)
            .unwrap()
            .is_running());
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Cancel {
                        request_id: "queued".into()
                    },
                    now
                )
                .message,
            wire_error("queued".into(), ResponseErrorCode::InvalidRequest)
        );
        let active = coordinator
            .submit("active".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        let ticket = coordinator.reserve_next(now).unwrap().unwrap();
        coordinator.mark_running(active, now).unwrap();
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Cancel {
                        request_id: "active".into()
                    },
                    now
                )
                .message,
            ResponseMessage::Accepted {
                request_id: "active".into(),
                operation: AcceptedOperation::Cancel,
                state: RequestStatus::Running
            }
        );
        assert!(ticket.cancellation.is_cancelled());
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator.confirm_reconciled().is_err());
        coordinator
            .complete(active, Completion::Cancelled, vec![], now)
            .unwrap();
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Status {
                        request_id: "active".into()
                    },
                    now
                )
                .message,
            ResponseMessage::Status {
                request_id: "active".into(),
                state: RequestStatus::Cancelled
            }
        );
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Cancel {
                        request_id: "missing".into()
                    },
                    now
                )
                .message,
            wire_error("missing".into(), ResponseErrorCode::InvalidRequest)
        );
    }

    #[test]
    fn rpc_persistence_failures_never_acknowledge_acceptance_or_cancellation() {
        let temp = TempDir::new().unwrap();
        let blocked = temp.path().join("blocked");
        let mut coordinator = core(&blocked, 8);
        coordinator.confirm_reconciled().unwrap();
        std::fs::write(&blocked, "fixture").unwrap();
        let now = Instant::now();
        let result = coordinator.handle_client_request(
            ClientRequest::Submit {
                request_id: "failed".into(),
                request: chat(),
                timeout: Duration::from_secs(10),
            },
            now,
        );
        assert_eq!(
            result.message,
            wire_error("failed".into(), ResponseErrorCode::Internal)
        );
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(coordinator.queued_count(), 0);
        assert!(coordinator.history().records().is_empty());
        assert!(!format!("{:?}", result.message).contains("fixture"));
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        coordinator
            .submit("queued".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        coordinator.data_root = blocked;
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Cancel {
                        request_id: "queued".into()
                    },
                    now
                )
                .message,
            wire_error("queued".into(), ResponseErrorCode::Internal)
        );
        assert_eq!(coordinator.queued_count(), 1);
        assert_eq!(coordinator.request_state("queued"), Some("queued"));
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
    }

    #[test]
    fn rpc_invalid_inputs_and_unbound_runtimes_have_fixed_errors() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Status {
                        request_id: "bad\nid".into()
                    },
                    now
                )
                .message,
            ResponseMessage::Error {
                request_id: None,
                code: ResponseErrorCode::InvalidRequest
            }
        );
        let mut invalid = chat();
        if let InferenceRequest::Chat { turns, .. } = &mut invalid {
            turns.clear();
        }
        let rejected = coordinator.handle_client_request(
            ClientRequest::Submit {
                request_id: "invalid".into(),
                request: invalid,
                timeout: Duration::from_secs(1),
            },
            now,
        );
        assert_eq!(
            rejected.message,
            wire_error("invalid".into(), ResponseErrorCode::InvalidRequest)
        );
        coordinator.bindings.clear();
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Submit {
                        request_id: "unbound".into(),
                        request: chat(),
                        timeout: Duration::from_secs(1)
                    },
                    now
                )
                .message,
            wire_error("unbound".into(), ResponseErrorCode::Unavailable)
        );
        assert_eq!(coordinator.queued_count(), 0);
        assert_eq!(
            coordinator
                .handle_client_request(
                    ClientRequest::Status {
                        request_id: "missing".into()
                    },
                    now
                )
                .message,
            ResponseMessage::Status {
                request_id: "missing".into(),
                state: RequestStatus::NotFound
            }
        );
    }

    #[test]
    fn rpc_restart_returns_interrupted_and_refuses_duplicate_replay() {
        let temp = TempDir::new().unwrap();
        let now = Instant::now();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        coordinator
            .submit("retained".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        drop(coordinator);
        let mut restarted = core(temp.path(), 8);
        assert_eq!(
            restarted
                .handle_client_request(
                    ClientRequest::Status {
                        request_id: "retained".into()
                    },
                    now
                )
                .message,
            ResponseMessage::Status {
                request_id: "retained".into(),
                state: RequestStatus::Interrupted
            }
        );
        assert_eq!(restarted.queued_count(), 0);
        restarted.confirm_reconciled().unwrap();
        assert_eq!(
            restarted
                .handle_client_request(
                    ClientRequest::Submit {
                        request_id: "retained".into(),
                        request: chat(),
                        timeout: Duration::from_secs(10)
                    },
                    now
                )
                .message,
            wire_error("retained".into(), ResponseErrorCode::DuplicateRequest)
        );
    }

    #[test]
    fn retained_id_without_state_is_not_misreported_as_not_found() {
        let temp = TempDir::new().unwrap();
        let now = Instant::now();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let id = coordinator
            .submit("incomplete".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        coordinator.cancel(id).unwrap();
        drop(coordinator);
        let path = temp.path().join("jobs.json");
        let mut jobs: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        jobs[0].as_object_mut().unwrap().remove("result_state");
        jobs[0]["exit"] = serde_json::json!("0");
        std::fs::write(path, jobs.to_string()).unwrap();
        let mut restarted = core(temp.path(), 8);
        assert_eq!(
            restarted
                .handle_client_request(
                    ClientRequest::Status {
                        request_id: "incomplete".into()
                    },
                    now
                )
                .message,
            wire_error("incomplete".into(), ResponseErrorCode::Internal)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn lost_socket_ack_keeps_durable_work_for_reconnect_without_dispatch() {
        use crate::control::{
            ipc::{
                RequestEnvelope, WireChatRole, WireChatTurn, WireInferenceRequest,
                WireRequestMessage,
            },
            local_ipc::{LocalConnection, LocalListener},
            runtime::{NeverCancel, OperationContext},
        };
        use std::net::Shutdown;
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixStream;
        let data = TempDir::new().unwrap();
        let socket = TempDir::new().unwrap();
        let locks = TempDir::new().unwrap();
        for directory in [socket.path(), locks.path()] {
            std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let listener = LocalListener::bind(socket.path(), locks.path(), [0]).unwrap();
        let mut coordinator = core(data.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let mut peer = UnixStream::connect(socket.path().join("coordinator.sock")).unwrap();
        ipc::write_request(
            &mut peer,
            &RequestEnvelope::v1(WireRequestMessage::Submit {
                request_id: "lost-ack".into(),
                timeout_ms: 10000,
                request: WireInferenceRequest::Language {
                    profile_id: "gemma-4-26b-qat-mtp".into(),
                    turns: vec![WireChatTurn {
                        role: WireChatRole::User,
                        content: "SENSITIVE_SOCKET_PAYLOAD".into(),
                    }],
                    max_tokens: 8,
                },
            }),
        )
        .unwrap();
        peer.shutdown(Shutdown::Both).unwrap();
        drop(peer);
        let context = OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel);
        let mut connection = listener.accept(&context).unwrap();
        assert!(coordinator
            .serve_connection(&mut connection, &context)
            .is_err());
        assert_eq!(coordinator.queued_count(), 1);
        assert_eq!(coordinator.active_count(), 0);
        assert_eq!(coordinator.request_state("lost-ack"), Some("queued"));
        assert!(!std::fs::read_to_string(data.path().join("jobs.json"))
            .unwrap()
            .contains("SENSITIVE_SOCKET_PAYLOAD"));
        let directory = socket.path().to_owned();
        let client = std::thread::spawn(move || {
            let mut responses = Vec::new();
            for operation in [
                WireRequestMessage::Status {
                    request_id: "lost-ack".into(),
                },
                WireRequestMessage::Cancel {
                    request_id: "lost-ack".into(),
                },
                WireRequestMessage::Status {
                    request_id: "lost-ack".into(),
                },
            ] {
                let ctx =
                    OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel);
                let mut connection = LocalConnection::connect(&directory, &ctx).unwrap();
                connection
                    .write_request(&RequestEnvelope::v1(operation), &ctx)
                    .unwrap();
                responses.push(connection.read_response(&ctx).unwrap().message);
            }
            responses
        });
        for _ in 0..3 {
            let ctx = OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel);
            let mut reconnected = listener.accept(&ctx).unwrap();
            coordinator
                .serve_connection(&mut reconnected, &ctx)
                .unwrap();
        }
        assert_eq!(
            client.join().unwrap(),
            vec![
                ResponseMessage::Status {
                    request_id: "lost-ack".into(),
                    state: RequestStatus::Queued
                },
                ResponseMessage::Accepted {
                    request_id: "lost-ack".into(),
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Cancelled
                },
                ResponseMessage::Status {
                    request_id: "lost-ack".into(),
                    state: RequestStatus::Cancelled
                }
            ]
        );
        assert_eq!(coordinator.queued_count(), 0);
        assert_eq!(coordinator.active_count(), 0);
    }

    #[test]
    fn acceptance_is_persisted_bounded_and_never_records_payloads() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 1);
        let now = Instant::now();
        assert!(coordinator
            .submit("first".into(), chat(), Duration::from_secs(10), now)
            .is_err());
        coordinator.confirm_reconciled().unwrap();
        let id = coordinator
            .submit("first".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        let history = std::fs::read_to_string(temp.path().join("jobs.json")).unwrap();
        assert!(history.contains("first"));
        assert!(!history.contains("REQUEST_PAYLOAD_MUST_NOT_BE_PERSISTED"));
        assert!(coordinator
            .submit("second".into(), chat(), Duration::from_secs(10), now)
            .is_err());
        coordinator.cancel(id).unwrap();
        assert_eq!(coordinator.queued_count(), 0);
        assert_eq!(coordinator.request_state("first"), Some("cancelled"));
        assert!(coordinator.cancel_request("first").is_err());
        assert!(coordinator.cancel_request("unknown").is_err());
        assert!(coordinator
            .submit("first".into(), chat(), Duration::from_secs(10), now)
            .is_err());
        assert!(coordinator.reserve_next(now).unwrap().is_none());
    }

    #[test]
    fn request_id_lookup_and_cancel_survive_restart_without_replay() {
        let temp = TempDir::new().unwrap();
        let now = Instant::now();
        let mut coordinator = core(temp.path(), 2);
        coordinator.confirm_reconciled().unwrap();
        coordinator
            .submit("stable-id".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        assert_eq!(coordinator.request_state("stable-id"), Some("queued"));
        drop(coordinator);
        let mut restarted = core(temp.path(), 2);
        assert_eq!(restarted.request_state("stable-id"), Some("interrupted"));
        assert!(restarted.cancel_request("stable-id").is_err());
        restarted.confirm_reconciled().unwrap();
        restarted
            .submit("new-id".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        restarted.cancel_request("new-id").unwrap();
        assert_eq!(restarted.request_state("new-id"), Some("cancelled"));
        assert_eq!(restarted.queued_count(), 0);
    }

    #[test]
    fn quarantine_blocks_already_reserved_dispatch_without_releasing_it() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 2);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        let id = coordinator
            .submit("reserved".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        let _ticket = coordinator.reserve_next(now).unwrap().unwrap();
        coordinator.quarantine();
        assert!(coordinator.mark_running(id, now).is_err());
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator.confirm_reconciled().is_err());
        coordinator
            .complete(id, Completion::Cancelled, vec![], now)
            .unwrap();
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        coordinator.confirm_reconciled().unwrap();
    }

    #[test]
    fn gpu_fifo_survives_active_cancel_while_hosted_work_can_proceed() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        let language = coordinator
            .submit("chat".into(), chat(), Duration::from_secs(30), now)
            .unwrap();
        let speech = coordinator
            .submit("speech".into(), media(), Duration::from_secs(30), now)
            .unwrap();
        let remote = coordinator
            .submit("remote".into(), hosted(), Duration::from_secs(30), now)
            .unwrap();
        let ticket = coordinator
            .reserve_next(now + Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert_eq!(ticket.job_id, language);
        coordinator
            .mark_running(language, now + Duration::from_secs(2))
            .unwrap();
        assert_eq!(
            coordinator
                .reserve_next(now + Duration::from_secs(2))
                .unwrap()
                .unwrap()
                .job_id,
            remote
        );
        coordinator
            .mark_running(remote, now + Duration::from_secs(2))
            .unwrap();
        coordinator.cancel(language).unwrap();
        assert!(ticket.cancellation.is_cancelled());
        assert!(coordinator
            .reserve_next(now + Duration::from_secs(3))
            .unwrap()
            .is_none());
        let timing = coordinator
            .complete(
                language,
                Completion::Cancelled,
                vec![],
                now + Duration::from_secs(5),
            )
            .unwrap();
        assert_eq!(
            timing,
            RequestTiming {
                queue_wait: Duration::from_secs(1),
                cold_start: Duration::from_secs(1),
                execution: Duration::from_secs(3)
            }
        );
        assert_eq!(
            coordinator
                .reserve_next(now + Duration::from_secs(5))
                .unwrap()
                .unwrap()
                .job_id,
            speech
        );
        assert_eq!(coordinator.active_count(), 2);
    }

    #[test]
    fn expired_work_never_dispatches_and_restart_never_replays() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        let expired = coordinator
            .submit("expired".into(), chat(), Duration::from_secs(1), now)
            .unwrap();
        assert!(coordinator
            .reserve_next(now + Duration::from_secs(2))
            .unwrap()
            .is_none());
        assert_eq!(
            coordinator
                .history()
                .records()
                .iter()
                .find(|job| job.id == expired)
                .unwrap()
                .metadata
                .result_state
                .as_deref(),
            Some("failed")
        );
        coordinator
            .submit("accepted".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        let ticket = coordinator.reserve_next(now).unwrap().unwrap();
        coordinator.mark_running(ticket.job_id, now).unwrap();
        drop(coordinator);
        let restarted = core(temp.path(), 8);
        assert_eq!(
            restarted.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert_eq!(restarted.queued_count(), 0);
        assert_eq!(restarted.active_count(), 0);
        assert!(restarted
            .history()
            .records()
            .iter()
            .any(|job| job.metadata.request_id.as_deref() == Some("accepted")
                && job.metadata.result_state.as_deref() == Some("interrupted")));
    }

    #[test]
    fn persistence_failure_does_not_accept_work_or_release_active_ownership() {
        let temp = TempDir::new().unwrap();
        let blocked = temp.path().join("blocked");
        let mut coordinator = core(&blocked, 8);
        coordinator.confirm_reconciled().unwrap();
        std::fs::write(&blocked, "file blocks history directory").unwrap();
        assert!(coordinator
            .submit(
                "first".into(),
                chat(),
                Duration::from_secs(10),
                Instant::now()
            )
            .is_err());
        assert_eq!(coordinator.queued_count(), 0);
        assert!(coordinator.history().records().is_empty());

        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        let id = coordinator
            .submit("active".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        coordinator.reserve_next(now).unwrap().unwrap();
        coordinator.mark_running(id, now).unwrap();
        assert!(coordinator
            .complete(
                id,
                Completion::Succeeded,
                vec!["https://user:secret@example.com".into()],
                now
            )
            .is_err());
        assert_eq!(coordinator.active_count(), 1);
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
    }

    #[test]
    fn backend_identity_can_be_persisted_before_dispatch_and_survives_restart() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        let id = coordinator
            .submit("owned-comfy".into(), media(), Duration::from_secs(10), now)
            .unwrap();
        coordinator.reserve_next(now).unwrap().unwrap();
        coordinator.mark_running(id, now).unwrap();
        let backend = "12345678-1234-1234-1234-123456789abc";
        coordinator.attach_backend(id, backend.into()).unwrap();
        let persisted = std::fs::read_to_string(temp.path().join("jobs.json")).unwrap();
        assert!(persisted.contains(backend));
        assert!(!persisted.contains("--text"));
        drop(coordinator);

        let mut restarted = core(temp.path(), 8);
        assert_eq!(restarted.request_state("owned-comfy"), Some("interrupted"));
        assert_eq!(
            restarted.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert!(restarted.reserve_next(now).is_err());
        let recovered = std::fs::read_to_string(temp.path().join("jobs.json")).unwrap();
        assert!(recovered.contains(backend));
    }

    #[test]
    fn uncertain_completion_and_pre_dispatch_cancellation_require_acknowledgement() {
        let temp = TempDir::new().unwrap();
        let mut coordinator = core(temp.path(), 8);
        coordinator.confirm_reconciled().unwrap();
        let now = Instant::now();
        let id = coordinator
            .submit("cancelled".into(), chat(), Duration::from_secs(10), now)
            .unwrap();
        let ticket = coordinator.reserve_next(now).unwrap().unwrap();
        assert!(coordinator.acknowledge_reconciled_cleanup(id).is_err());
        assert!(coordinator.attach_backend(id, "backend".into()).is_err());
        coordinator.cancel(id).unwrap();
        assert!(ticket.cancellation.is_cancelled());
        assert!(coordinator.mark_running(id, now).is_err());
        assert!(coordinator.confirm_reconciled().is_err());
        coordinator
            .complete(id, Completion::Uncertain, vec![], now)
            .unwrap();
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        assert!(coordinator
            .submit("new".into(), chat(), Duration::from_secs(10), now)
            .is_err());
        assert_eq!(coordinator.active_count(), 1);
        assert!(coordinator.confirm_reconciled().is_err());
        assert_eq!(coordinator.request_state("cancelled"), Some("interrupted"));
        assert!(coordinator
            .complete(id, Completion::Cancelled, vec![], now)
            .is_err());
        coordinator.acknowledge_reconciled_cleanup(id).unwrap();
        assert_eq!(coordinator.active_count(), 0);
        assert_eq!(coordinator.request_state("cancelled"), Some("interrupted"));
        assert_eq!(
            coordinator.admission(),
            AdmissionState::ReconciliationRequired
        );
        coordinator.confirm_reconciled().unwrap();
        assert!(coordinator
            .submit("cancelled".into(), chat(), Duration::from_secs(10), now)
            .is_err());
    }
}
