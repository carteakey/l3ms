//! Queue/admission core for the future host-local coordinator.
//!
//! This module has no engine or IPC implementation. A dispatch ticket reserves
//! in-flight work; the host executor must separately prove runtime readiness,
//! admission, and GPU residency before execution. Completion releases only the
//! request reservation, never proves that a resident model released VRAM.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};

use super::catalog::{Catalog, Lane, RuntimeIdentity};
use super::ipc::{
    self, AcceptedOperation, ClientRequest, RequestStatus, ResponseEnvelope, ResponseErrorCode,
    ResponseMessage,
};
use super::requests::InferenceRequest;
use super::routes::{resolve_profile_route, ResolvedProfileRoute};
use super::runtime::{Cancellation, ResourceClaim};
use crate::job_history::{CrossToolkitJobStart, CrossToolkitJobUpdate, JobHistory};

const MAX_CAPACITY: usize = 128;
const MAX_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// Fixed submission failures; no request contents, native paths or OS errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionError {
    Unavailable,
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
    resource: ResourceClaim,
    cancellation: RequestCancellation,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestTiming {
    pub queue_wait: Duration,
    pub cold_start: Duration,
    pub execution: Duration,
}

/// One coordinator's bounded queue. Only the future daemon may own this core
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
        let (history, notice) = JobHistory::load_in(data_root.as_ref(), repository_root)?;
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
        })
    }

    pub fn admission(&self) -> AdmissionState {
        self.admission
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

    /// Reconnect lookup uses the caller's retained request ID, never a numeric
    /// history ID (which can be rehydrated differently after restart).
    pub fn request_state(&self, request_id: &str) -> Option<&str> {
        self.history
            .records()
            .iter()
            .find(|job| job.metadata.request_id.as_deref() == Some(request_id))
            .and_then(|job| job.metadata.result_state.as_deref())
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

    /// Called only after the host executor has reconciled actual backend jobs,
    /// admission and resource ownership. This does not perform those probes.
    pub fn confirm_reconciled(&mut self) -> Result<()> {
        ensure!(
            self.active.is_empty(),
            "active reservations still require reconciliation"
        );
        self.admission = AdmissionState::Ready;
        Ok(())
    }

    pub fn quarantine(&mut self) {
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
                        SubmissionError::Unavailable => ResponseErrorCode::Unavailable,
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
        if self.pending.len() + self.active.len() >= self.capacity {
            return Err(SubmissionError::QueueFull);
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
        let deadline = now
            .checked_add(timeout)
            .ok_or(SubmissionError::InvalidRequest)?;
        let mut candidate = self.history.clone();
        let job_id = candidate
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: match profile.key.lane {
                    Lane::Language => "l3ms",
                    Lane::Media => "l2m2",
                    Lane::Decisions => "l3ds",
                }
                .into(),
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
        ensure!(
            self.admission == AdmissionState::Ready,
            "runtime reconciliation is required"
        );
        let mut index = 0;
        while index < self.pending.len() {
            let pending = &self.pending[index];
            if now >= pending.deadline || pending.cancellation.is_cancelled() {
                let state = if pending.cancellation.is_cancelled() {
                    "cancelled"
                } else {
                    "failed"
                };
                self.persist_update(pending.job_id, state, None, None)?;
                self.pending.remove(index);
            } else {
                index += 1;
            }
        }
        let Some(index) = self.pending.iter().position(|pending| {
            !self
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
        let queue_wait = now.saturating_duration_since(pending.submitted);
        self.active.insert(
            pending.job_id,
            Active {
                resource: pending.resource,
                cancellation: pending.cancellation.clone(),
                deadline: pending.deadline,
                reserved: now,
                started: None,
                queue_wait,
            },
        );
        Ok(Some(DispatchTicket {
            job_id: pending.job_id,
            request_id: pending.request_id,
            request: pending.request,
            route: pending.route,
            resource: pending.resource,
            deadline: pending.deadline,
            cancellation: pending.cancellation,
            queue_wait,
        }))
    }

    /// Persist running state immediately before native inference begins.
    pub fn mark_running(&mut self, job_id: u64, now: Instant) -> Result<()> {
        ensure!(
            self.admission == AdmissionState::Ready,
            "runtime reconciliation is required"
        );
        let active = self
            .active
            .get(&job_id)
            .context("unknown dispatch reservation")?;
        ensure!(active.started.is_none(), "request was already dispatched");
        ensure!(
            now < active.deadline && !active.cancellation.is_cancelled(),
            "request cancelled or deadline expired before dispatch"
        );
        self.persist_update(job_id, "running", None, None)?;
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
        if let Err(error) = self.persist_update(job_id, "running", Some(backend_id), None) {
            self.quarantine();
            return Err(error.context("backend ID persistence failed; reconcile before more work"));
        }
        Ok(())
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
        self.active.remove(&job_id);
        if outcome == Completion::Uncertain {
            self.quarantine();
        }
        Ok(timing)
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

fn wire_error(request_id: String, code: ResponseErrorCode) -> ResponseMessage {
    ResponseMessage::Error {
        request_id: Some(request_id),
        code,
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
        coordinator.confirm_reconciled().unwrap();
        assert!(coordinator
            .submit("cancelled".into(), chat(), Duration::from_secs(10), now)
            .is_err());
    }
}
