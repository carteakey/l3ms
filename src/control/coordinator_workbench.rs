//! One-shot, private-socket actions for the Stack Catalog workbench.
//!
//! Submit preparation is synchronous and bounded by the shared strict reader;
//! its opaque envelope can only be consumed once by the dedicated worker. The
//! worker does no service discovery, startup, retry, history write, or payload
//! logging. Every caller supplies the explicit coordinator socket directory.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::catalog::{Catalog, Lane, ProfileKey};
use super::client::{self, ClientError};
use super::ipc::{
    self, AcceptedOperation, ClientRequest, RequestEnvelope, RequestStatus, ResponseErrorCode,
    ResponseMessage, WireRequestMessage,
};
use super::result_consumer::NativeOutput;
use super::runtime::{Cancellation, OperationContext};
use super::submission_file::{self, SubmissionFileError};
use super::task_form::{self, TaskFormDraft, TaskFormError};

const ACTION_TIMEOUT: Duration = Duration::from_secs(3);
const ACTION_QUEUE_CAPACITY: usize = 1;
const EVENT_QUEUE_CAPACITY: usize = 4;

/// A fixed failure while preparing a caller-owned request file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparationError {
    SubmitFile(SubmissionFileError),
    TaskForm(TaskFormError),
    SelectedProfileMismatch,
}

impl std::fmt::Display for PreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubmitFile(error) => error.fmt(f),
            Self::TaskForm(error) => error.fmt(f),
            Self::SelectedProfileMismatch => {
                f.write_str("submit profile does not match the selected Stack Catalog profile")
            }
        }
    }
}

impl std::error::Error for PreparationError {}

/// An explicit caller-supplied ID and lane. Its Debug intentionally hides the
/// ID, and it is never generated or persisted by this module.
#[derive(Clone, Eq, PartialEq)]
pub struct RetainedRequest {
    request_id: String,
    lane: Lane,
}

impl RetainedRequest {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn lane(&self) -> Lane {
        self.lane
    }
}

impl std::fmt::Debug for RetainedRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedRequest")
            .field("request_id", &"<caller-supplied>")
            .field("lane", &self.lane)
            .finish()
    }
}

/// A validated v1 Submit plus the exact caller-owned identity. This is
/// intentionally non-Clone so UI code cannot accidentally replay it.
pub struct PreparedSubmission {
    envelope: RequestEnvelope,
    request: RetainedRequest,
}

impl PreparedSubmission {
    pub fn request_id(&self) -> &str {
        self.request.request_id()
    }

    pub fn lane(&self) -> Lane {
        self.request.lane()
    }
}

impl std::fmt::Debug for PreparedSubmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedSubmission")
            .field("request", &self.request)
            .field("envelope", &"<redacted native request>")
            .finish()
    }
}

/// Read once through the shared strict file contract and require exact lane
/// and profile identity with the currently selected Stack Catalog row.
pub fn prepare_submission(
    path: &Path,
    catalog: &Catalog,
    selected_profile: &ProfileKey,
) -> Result<PreparedSubmission, PreparationError> {
    let envelope = submission_file::read(path, catalog).map_err(PreparationError::SubmitFile)?;
    prepare_validated_envelope(envelope, selected_profile)
}

/// Construct an opaque one-shot submission from the in-memory native task
/// form. `build_submit` performs the bounded envelope and refreshed-catalog
/// validation before this function accepts the envelope for enqueue.
pub(crate) fn prepare_task_form_submission(
    catalog: &Catalog,
    draft: TaskFormDraft,
    selected_profile: &ProfileKey,
) -> Result<PreparedSubmission, PreparationError> {
    let envelope = task_form::build_submit(catalog, draft).map_err(PreparationError::TaskForm)?;
    prepare_validated_envelope(envelope, selected_profile)
}

fn prepare_validated_envelope(
    envelope: RequestEnvelope,
    selected_profile: &ProfileKey,
) -> Result<PreparedSubmission, PreparationError> {
    ipc::validate_request_envelope(&envelope)
        .map_err(|_| PreparationError::SubmitFile(SubmissionFileError::Invalid))?;
    let ClientRequest::Submit {
        request_id,
        request,
        ..
    } = envelope
        .clone()
        .into_client_request()
        .map_err(|_| PreparationError::SubmitFile(SubmissionFileError::Invalid))?
    else {
        return Err(PreparationError::SubmitFile(
            SubmissionFileError::WrongOperation,
        ));
    };
    if request.profile() != selected_profile {
        return Err(PreparationError::SelectedProfileMismatch);
    }
    Ok(PreparedSubmission {
        envelope,
        request: RetainedRequest {
            request_id,
            lane: request.lane(),
        },
    })
}

/// Fixed refusal to enqueue more work on the single bounded coordinator
/// worker. No action has been sent when this is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnqueueError {
    Busy,
    WorkerUnavailable,
    InvalidRequestId,
}

impl std::fmt::Display for EnqueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Busy => "a coordinator action is already in progress",
            Self::WorkerUnavailable => "coordinator workbench worker is unavailable",
            Self::InvalidRequestId => "coordinator request ID is invalid",
        })
    }
}

/// A one-shot Submit result. `Unconfirmed` deliberately preserves uncertainty:
/// callers must observe the retained ID and must not automatically resubmit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmitOutcome {
    Accepted,
    Rejected(ResponseErrorCode),
    Unconfirmed(ClientError),
}

/// Bounded status and native-output snapshot from an explicit read-only query.
pub struct Observation {
    pub status: RequestStatus,
    pub output: Option<NativeOutput>,
    pub decision_metadata: Option<super::result_consumer::NativeDecisionMetadata>,
    pub output_complete: bool,
    pub results_available: bool,
    pub results_state: ResultReadState,
}

impl std::fmt::Debug for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Observation")
            .field("status", &self.status)
            .field(
                "output",
                &self.output.as_ref().map(|_| "<redacted native output>"),
            )
            .field("output_complete", &self.output_complete)
            .field("results_available", &self.results_available)
            .field("results_state", &self.results_state)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultReadState {
    Available,
    Unknown,
    Expired,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObserveFailure {
    Exchange(ClientError),
    Rejected(ResponseErrorCode),
    InvalidResponse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelOutcome {
    Accepted(RequestStatus),
    Rejected(ResponseErrorCode),
    Unconfirmed(ClientError),
}

/// Completed background action. Request IDs are available to correlate with
/// the UI's retained caller identity but are hidden in Debug.
pub enum WorkbenchEvent {
    Submitted {
        request_id: String,
        outcome: SubmitOutcome,
    },
    Observed {
        request_id: String,
        outcome: Result<Observation, ObserveFailure>,
    },
    Cancelled {
        request_id: String,
        outcome: CancelOutcome,
    },
    WorkerFailed {
        request_id: String,
    },
}

impl WorkbenchEvent {
    pub fn request_id(&self) -> &str {
        match self {
            Self::Submitted { request_id, .. }
            | Self::Observed { request_id, .. }
            | Self::Cancelled { request_id, .. }
            | Self::WorkerFailed { request_id } => request_id,
        }
    }
}

impl std::fmt::Debug for WorkbenchEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (name, outcome) = match self {
            Self::Submitted { outcome, .. } => ("Submitted", format!("{outcome:?}")),
            Self::Observed { outcome, .. } => ("Observed", format!("{outcome:?}")),
            Self::Cancelled { outcome, .. } => ("Cancelled", format!("{outcome:?}")),
            Self::WorkerFailed { .. } => ("WorkerFailed", "<fixed>".to_owned()),
        };
        f.debug_struct("WorkbenchEvent")
            .field("kind", &name)
            .field("request_id", &"<caller-supplied>")
            .field("outcome", &outcome)
            .finish()
    }
}

enum Action {
    Submit {
        socket_directory: PathBuf,
        prepared: PreparedSubmission,
    },
    Observe {
        socket_directory: PathBuf,
        request: RetainedRequest,
    },
    Cancel {
        socket_directory: PathBuf,
        request_id: String,
    },
}

impl Action {
    fn request_id(&self) -> &str {
        match self {
            Self::Submit { prepared, .. } => prepared.request_id(),
            Self::Observe { request, .. } => request.request_id(),
            Self::Cancel { request_id, .. } => request_id,
        }
    }
}

/// Dedicated single-thread, one-slot worker for explicit local coordinator
/// exchanges. The caller prepares Submit files before enqueue and retains the
/// returned ID before any network exchange can happen.
pub struct CoordinatorWorkbench {
    actions: Option<SyncSender<Action>>,
    events: Option<Receiver<WorkbenchEvent>>,
    in_flight: bool,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl CoordinatorWorkbench {
    pub fn start() -> Result<Self, EnqueueError> {
        let (action_sender, action_receiver) = mpsc::sync_channel(ACTION_QUEUE_CAPACITY);
        let (event_sender, event_receiver) = mpsc::sync_channel(EVENT_QUEUE_CAPACITY);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("l3ms-coordinator-workbench".into())
            .spawn(move || work_loop(action_receiver, event_sender, worker_stop))
            .map_err(|_| EnqueueError::WorkerUnavailable)?;
        Ok(Self {
            actions: Some(action_sender),
            events: Some(event_receiver),
            in_flight: false,
            stop,
            worker: Some(worker),
        })
    }

    /// Enqueue the opaque one-shot Submit. The caller must copy and retain
    /// `prepared.request_id()` and `prepared.lane()` before calling this:
    /// once enqueued, the worker may exchange IPC immediately.
    pub fn submit(
        &mut self,
        socket_directory: PathBuf,
        prepared: PreparedSubmission,
    ) -> Result<RetainedRequest, EnqueueError> {
        let retained = prepared.request.clone();
        self.enqueue(Action::Submit {
            socket_directory,
            prepared,
        })?;
        Ok(retained)
    }

    pub fn observe(
        &mut self,
        socket_directory: PathBuf,
        request_id: String,
        lane: Lane,
    ) -> Result<(), EnqueueError> {
        ipc::validate_request_id(&request_id).map_err(|_| EnqueueError::InvalidRequestId)?;
        self.enqueue(Action::Observe {
            socket_directory,
            request: RetainedRequest { request_id, lane },
        })
    }

    pub fn cancel(
        &mut self,
        socket_directory: PathBuf,
        request_id: String,
    ) -> Result<(), EnqueueError> {
        ipc::validate_request_id(&request_id).map_err(|_| EnqueueError::InvalidRequestId)?;
        self.enqueue(Action::Cancel {
            socket_directory,
            request_id,
        })
    }

    pub fn try_next_event(&mut self) -> Option<WorkbenchEvent> {
        let event = self.events.as_ref()?.try_recv();
        match event {
            Ok(event) => {
                self.in_flight = false;
                Some(event)
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    fn enqueue(&mut self, action: Action) -> Result<(), EnqueueError> {
        if self.in_flight {
            return Err(EnqueueError::Busy);
        }
        let Some(sender) = &self.actions else {
            return Err(EnqueueError::WorkerUnavailable);
        };
        match sender.try_send(action) {
            Ok(()) => {
                self.in_flight = true;
                Ok(())
            }
            Err(TrySendError::Full(_)) => Err(EnqueueError::Busy),
            Err(TrySendError::Disconnected(_)) => Err(EnqueueError::WorkerUnavailable),
        }
    }
}

impl Drop for CoordinatorWorkbench {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.actions.take();
        // Closing the event receiver wakes a worker blocked in send, so Drop
        // cannot wait forever for an event consumer that is no longer alive.
        self.events.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct WorkerStop(Arc<AtomicBool>);
impl Cancellation for WorkerStop {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

fn work_loop(actions: Receiver<Action>, events: SyncSender<WorkbenchEvent>, stop: Arc<AtomicBool>) {
    let mut output_cache = None;
    while let Ok(action) = actions.recv() {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let request_id = action.request_id().to_owned();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            execute(action, Arc::clone(&stop), &mut output_cache)
        }))
        .unwrap_or(WorkbenchEvent::WorkerFailed { request_id });
        if events.send(outcome).is_err() {
            break;
        }
    }
}

// Exactly one bounded native consumer is retained, in memory only. Its cursor
// is valid only for this caller ID/lane/endpoint; another observation drops it.
struct OutputCache {
    socket_directory: PathBuf,
    request: RetainedRequest,
    consumer: super::result_consumer::ResultConsumer,
}

fn execute(
    action: Action,
    stop: Arc<AtomicBool>,
    output_cache: &mut Option<OutputCache>,
) -> WorkbenchEvent {
    let cancellation = WorkerStop(stop);
    let context = OperationContext::new(Instant::now() + ACTION_TIMEOUT, &cancellation);
    match action {
        Action::Submit {
            socket_directory,
            prepared,
        } => {
            *output_cache = None;
            let request_id = prepared.request.request_id.clone();
            let outcome = match client::exchange(&socket_directory, prepared.envelope, &context) {
                Ok(ResponseMessage::Accepted {
                    operation: AcceptedOperation::Submit,
                    ..
                }) => SubmitOutcome::Accepted,
                Ok(ResponseMessage::Error { code, .. }) => SubmitOutcome::Rejected(code),
                Ok(_) => SubmitOutcome::Unconfirmed(ClientError::InvalidResponse),
                Err(error) => SubmitOutcome::Unconfirmed(error),
            };
            WorkbenchEvent::Submitted {
                request_id,
                outcome,
            }
        }
        Action::Observe {
            socket_directory,
            request,
        } => {
            if !output_cache.as_ref().is_some_and(|cached| {
                cached.socket_directory == socket_directory && cached.request == request
            }) {
                *output_cache = None;
            }
            let RetainedRequest { request_id, lane } = request;
            let status_envelope = RequestEnvelope::v1(WireRequestMessage::Status {
                request_id: request_id.clone(),
            });
            let outcome = match client::exchange(&socket_directory, status_envelope, &context) {
                Err(error) => Err(ObserveFailure::Exchange(error)),
                Ok(ResponseMessage::Error { code, .. }) => Err(ObserveFailure::Rejected(code)),
                Ok(ResponseMessage::Status { state, .. }) if state == RequestStatus::NotFound => {
                    *output_cache = None;
                    Ok(Observation {
                        status: state,
                        output: None,
                        decision_metadata: None,
                        output_complete: false,
                        results_available: false,
                        results_state: ResultReadState::Unavailable,
                    })
                }
                Ok(ResponseMessage::Status { state, .. }) => {
                    let read = match output_cache.as_mut() {
                        Some(cached) => client::read_native_results_into(
                            &socket_directory,
                            &mut cached.consumer,
                            &context,
                        )
                        .map(|()| cached.consumer.clone()),
                        None => client::read_native_results(
                            &socket_directory,
                            &request_id,
                            lane,
                            &context,
                        ),
                    };
                    match read {
                        Ok(consumer) => {
                            let observation = Observation {
                                status: state,
                                output: Some(consumer.output().clone()),
                                decision_metadata: consumer.decision_metadata().cloned(),
                                output_complete: consumer.is_complete(),
                                results_available: true,
                                results_state: ResultReadState::Available,
                            };
                            *output_cache = Some(OutputCache {
                                socket_directory,
                                request: RetainedRequest {
                                    request_id: request_id.clone(),
                                    lane,
                                },
                                consumer,
                            });
                            Ok(observation)
                        }
                        Err(client::NativeReadError::Assembly(
                            super::result_consumer::ConsumeError::Unknown,
                        )) => {
                            *output_cache = None;
                            Ok(Observation {
                                status: state,
                                output: None,
                                decision_metadata: None,
                                output_complete: false,
                                results_available: false,
                                results_state: ResultReadState::Unknown,
                            })
                        }
                        Err(client::NativeReadError::Assembly(
                            super::result_consumer::ConsumeError::Expired,
                        )) => {
                            *output_cache = None;
                            Ok(Observation {
                                status: state,
                                output: None,
                                decision_metadata: None,
                                output_complete: false,
                                results_available: false,
                                results_state: ResultReadState::Expired,
                            })
                        }
                        Err(_) => Ok(Observation {
                            status: state,
                            output: None,
                            decision_metadata: None,
                            output_complete: false,
                            results_available: false,
                            results_state: ResultReadState::Unavailable,
                        }),
                    }
                }
                Ok(_) => Err(ObserveFailure::InvalidResponse),
            };
            WorkbenchEvent::Observed {
                request_id,
                outcome,
            }
        }
        Action::Cancel {
            socket_directory,
            request_id,
        } => {
            let envelope = RequestEnvelope::v1(WireRequestMessage::Cancel {
                request_id: request_id.clone(),
            });
            let outcome = match client::exchange(&socket_directory, envelope, &context) {
                Ok(ResponseMessage::Accepted {
                    operation: AcceptedOperation::Cancel,
                    state,
                    ..
                }) => CancelOutcome::Accepted(state),
                Ok(ResponseMessage::Error { code, .. }) => CancelOutcome::Rejected(code),
                Ok(_) => CancelOutcome::Unconfirmed(ClientError::InvalidResponse),
                Err(error) => CancelOutcome::Unconfirmed(error),
            };
            WorkbenchEvent::Cancelled {
                request_id,
                outcome,
            }
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;
    use std::thread;

    use crate::control::ipc::{self, ResponseEnvelope};
    use crate::control::result_buffer::{
        ResultBuffer, ResultEvent, ResultTerminal, MAX_PAGE_BYTES,
    };

    fn private_socket_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }

    fn fixture_catalog() -> Catalog {
        Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap()
    }

    fn language_submit(id: &str) -> RequestEnvelope {
        RequestEnvelope::v1(WireRequestMessage::Submit {
            request_id: id.into(),
            timeout_ms: 5_000,
            request: ipc::WireInferenceRequest::Language {
                profile_id: "gemma-4-26b-qat-mtp".into(),
                turns: vec![ipc::WireChatTurn {
                    role: ipc::WireChatRole::User,
                    content: "private-submit-payload".into(),
                }],
                max_tokens: 64,
            },
        })
    }

    fn write_submit(path: &Path, id: &str) {
        std::fs::write(path, serde_json::to_vec(&language_submit(id)).unwrap()).unwrap();
    }

    fn receive_event(worker: &mut CoordinatorWorkbench) -> WorkbenchEvent {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(event) = worker.try_next_event() {
                return event;
            }
            assert!(
                Instant::now() < deadline,
                "coordinator workbench event timed out"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn preparation_enforces_selected_profile_and_hides_request_body() {
        let catalog = fixture_catalog();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("submit.json");
        write_submit(&path, "caller:workbench-id");
        let selected = ProfileKey::new(Lane::Language, "gemma-4-26b-qat-mtp").unwrap();
        let prepared = prepare_submission(&path, &catalog, &selected).unwrap();
        assert_eq!(prepared.request_id(), "caller:workbench-id");
        assert_eq!(prepared.lane(), Lane::Language);
        assert!(!format!("{prepared:?}").contains("private-submit-payload"));
        let wrong = ProfileKey::new(Lane::Decisions, "nagi-small").unwrap();
        assert_eq!(
            prepare_submission(&path, &catalog, &wrong).unwrap_err(),
            PreparationError::SelectedProfileMismatch
        );
    }

    #[test]
    fn repeated_explicit_observation_resumes_cursor_without_resubmitting() {
        let directory = private_socket_directory();
        let socket = directory.path().to_owned();
        let listener = UnixListener::bind(socket.join("coordinator.sock")).unwrap();
        std::fs::set_permissions(
            socket.join("coordinator.sock"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let server = thread::spawn(move || {
            let mut buffer = ResultBuffer::new();
            buffer.begin("observe-stream", Lane::Language).unwrap();
            for (cursor, text) in [(0, "first"), (1, "second")] {
                buffer
                    .append("observe-stream", ResultEvent::ChatContent(text.into()))
                    .unwrap();
                let mut stream = listener.accept().unwrap().0;
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let ipc::IncomingRequest::Control(ClientRequest::Status { request_id }) =
                    ipc::read_incoming_request(&mut stream).unwrap()
                else {
                    panic!("observe must not submit")
                };
                ipc::write_response(
                    &mut stream,
                    &ResponseEnvelope::v1(ResponseMessage::Status {
                        request_id,
                        state: RequestStatus::Running,
                    }),
                )
                .unwrap();
                let mut stream = listener.accept().unwrap().0;
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let ipc::IncomingRequest::Results(query) =
                    ipc::read_incoming_request(&mut stream).unwrap()
                else {
                    panic!("observe must query results")
                };
                assert_eq!(query.after_cursor, cursor);
                let page = buffer
                    .page("observe-stream", cursor, MAX_PAGE_BYTES)
                    .unwrap();
                ipc::write_response(
                    &mut stream,
                    &ResponseEnvelope::v2_results(ResponseMessage::Results {
                        request_id: query.request_id,
                        state: RequestStatus::Running,
                        after_cursor: query.after_cursor,
                        byte_budget: query.byte_budget,
                        page,
                    }),
                )
                .unwrap();
            }
        });
        let mut worker = CoordinatorWorkbench::start().unwrap();
        for expected in ["first", "firstsecond"] {
            worker
                .observe(socket.clone(), "observe-stream".into(), Lane::Language)
                .unwrap();
            let WorkbenchEvent::Observed {
                outcome: Ok(observation),
                ..
            } = receive_event(&mut worker)
            else {
                panic!("expected successful incremental observation")
            };
            assert!(!observation.output_complete);
            let Some(NativeOutput::Language { content, .. }) = observation.output else {
                panic!("expected native chat output")
            };
            assert_eq!(content, expected);
        }
        server.join().unwrap();
    }

    #[test]
    fn submit_uncertainty_is_retained_and_a_followup_observe_is_read_only() {
        let directory = private_socket_directory();
        let socket = directory.path().to_owned();
        let listener = UnixListener::bind(socket.join("coordinator.sock")).unwrap();
        std::fs::set_permissions(
            socket.join("coordinator.sock"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let server = thread::spawn(move || {
            let mut stream = listener.accept().unwrap().0;
            let ipc::IncomingRequest::Control(ClientRequest::Submit { request_id, .. }) =
                ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("first workbench operation should be submit")
            };
            assert_eq!(request_id, "caller:workbench-id");
            drop(stream); // Simulate a lost acknowledgment; client must not retry.

            let mut status_stream = listener.accept().unwrap().0;
            let ipc::IncomingRequest::Control(ClientRequest::Status {
                request_id: status_id,
            }) = ipc::read_incoming_request(&mut status_stream).unwrap()
            else {
                panic!("follow-up should be read-only status")
            };
            assert_eq!(status_id, "caller:workbench-id");
            ipc::write_response(
                &mut status_stream,
                &ResponseEnvelope::v1(ResponseMessage::Status {
                    request_id: status_id,
                    state: RequestStatus::Succeeded,
                }),
            )
            .unwrap();

            let mut results_stream = listener.accept().unwrap().0;
            let ipc::IncomingRequest::Results(query) =
                ipc::read_incoming_request(&mut results_stream).unwrap()
            else {
                panic!("observation should query native results")
            };
            assert_eq!(query.request_id, "caller:workbench-id");
            let mut buffer = ResultBuffer::new();
            buffer.begin(&query.request_id, Lane::Language).unwrap();
            buffer
                .append(
                    &query.request_id,
                    ResultEvent::ChatContent("secret native result".into()),
                )
                .unwrap();
            buffer
                .terminal(&query.request_id, ResultTerminal::Succeeded)
                .unwrap();
            let page = buffer.page(&query.request_id, 0, MAX_PAGE_BYTES).unwrap();
            ipc::write_response(
                &mut results_stream,
                &ResponseEnvelope::v2_results(ResponseMessage::Results {
                    request_id: query.request_id,
                    state: RequestStatus::Succeeded,
                    after_cursor: query.after_cursor,
                    byte_budget: query.byte_budget,
                    page,
                }),
            )
            .unwrap();
        });

        let catalog = fixture_catalog();
        let directory_request = tempfile::tempdir().unwrap();
        let submit_file = directory_request.path().join("request.json");
        write_submit(&submit_file, "caller:workbench-id");
        let selected = ProfileKey::new(Lane::Language, "gemma-4-26b-qat-mtp").unwrap();
        let prepared = prepare_submission(&submit_file, &catalog, &selected).unwrap();
        let mut worker = CoordinatorWorkbench::start().unwrap();
        let retained = worker.submit(socket.clone(), prepared).unwrap();
        assert_eq!(retained.request_id(), "caller:workbench-id");
        let submitted = receive_event(&mut worker);
        assert!(matches!(
            submitted,
            WorkbenchEvent::Submitted {
                ref request_id,
                outcome: SubmitOutcome::Unconfirmed(_),
            } if request_id == "caller:workbench-id"
        ));
        worker
            .observe(
                socket.clone(),
                retained.request_id().into(),
                retained.lane(),
            )
            .unwrap();
        let observed = receive_event(&mut worker);
        match &observed {
            WorkbenchEvent::Observed {
                request_id,
                outcome: Ok(observation),
            } => {
                assert_eq!(request_id, "caller:workbench-id");
                assert_eq!(observation.status, RequestStatus::Succeeded);
                assert!(observation.output_complete);
                let Some(NativeOutput::Language { content, .. }) = &observation.output else {
                    panic!("expected language-native output")
                };
                assert_eq!(content, "secret native result");
                assert!(!format!("{observed:?}").contains("secret native result"));
            }
            _ => panic!("expected native result observation"),
        }
        server.join().unwrap();
    }

    #[test]
    fn cancel_is_a_single_explicit_exchange_and_uses_the_retained_id() {
        let directory = private_socket_directory();
        let socket = directory.path().to_owned();
        let listener = UnixListener::bind(socket.join("coordinator.sock")).unwrap();
        std::fs::set_permissions(
            socket.join("coordinator.sock"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let ipc::IncomingRequest::Control(ClientRequest::Cancel { request_id }) =
                ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("expected one explicit cancellation")
            };
            assert_eq!(request_id, "caller:cancel-id");
            ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v1(ResponseMessage::Accepted {
                    request_id,
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Running,
                }),
            )
            .unwrap();
        });
        let mut worker = CoordinatorWorkbench::start().unwrap();
        worker.cancel(socket, "caller:cancel-id".into()).unwrap();
        assert!(matches!(
            receive_event(&mut worker),
            WorkbenchEvent::Cancelled {
                ref request_id,
                outcome: CancelOutcome::Accepted(RequestStatus::Running),
            } if request_id == "caller:cancel-id"
        ));
        server.join().unwrap();
    }

    #[test]
    fn only_one_action_remains_in_flight_until_its_event_is_consumed() {
        let directory = private_socket_directory();
        let socket = directory.path().to_owned();
        let listener = UnixListener::bind(socket.join("coordinator.sock")).unwrap();
        std::fs::set_permissions(
            socket.join("coordinator.sock"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let server = thread::spawn(move || {
            let mut stream = listener.accept().unwrap().0;
            let ipc::IncomingRequest::Control(ClientRequest::Cancel { request_id }) =
                ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("expected first explicit cancel")
            };
            ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v1(ResponseMessage::Accepted {
                    request_id,
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Running,
                }),
            )
            .unwrap();
        });

        let mut worker = CoordinatorWorkbench::start().unwrap();
        worker
            .cancel(socket.clone(), "caller:first-action".into())
            .unwrap();
        server.join().unwrap();
        assert_eq!(
            worker.cancel(socket, "caller:must-not-queue".into()),
            Err(EnqueueError::Busy)
        );
        assert!(matches!(
            receive_event(&mut worker),
            WorkbenchEvent::Cancelled {
                outcome: CancelOutcome::Accepted(RequestStatus::Running),
                ..
            }
        ));
        assert!(!worker.in_flight);
    }

    #[test]
    fn dropping_with_an_undrained_event_closes_receiver_before_join() {
        let directory = private_socket_directory();
        let socket = directory.path().to_owned();
        let listener = UnixListener::bind(socket.join("coordinator.sock")).unwrap();
        std::fs::set_permissions(
            socket.join("coordinator.sock"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let server = thread::spawn(move || {
            let mut stream = listener.accept().unwrap().0;
            let ipc::IncomingRequest::Control(ClientRequest::Cancel { request_id }) =
                ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("expected explicit cancel")
            };
            ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v1(ResponseMessage::Accepted {
                    request_id,
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Cancelled,
                }),
            )
            .unwrap();
        });
        let mut worker = CoordinatorWorkbench::start().unwrap();
        worker
            .cancel(socket, "caller:undrained-event".into())
            .unwrap();
        server.join().unwrap();
        thread::sleep(Duration::from_millis(25));
        let started = Instant::now();
        drop(worker);
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
