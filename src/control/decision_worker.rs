//! One-at-a-time CPU-local decision execution for a future daemon executor.
//!
//! The caller must durably accept and begin the ticket before handing it here,
//! and must retain its reservation until it consumes a completion and acts on
//! the typed cleanup disposition. This worker owns no coordinator core,
//! history, result buffer, service, or admission state. It runs only declared
//! CPU local Nagi/Laya routes with no GPU claim; it does not choose a fallback,
//! install dependencies, fetch models, retry, or perform hosted execution.
//!
//! The child process inherits the daemon environment except for the explicit
//! offline and CUDA-visibility variables. This is not a sandbox: the native
//! toolkit and dependencies remain trusted code, and process-group containment
//! does not include descendants that deliberately escape their process group.
//!
//! The work channel holds at most one ticket and the completion channel holds
//! one result. The in-flight slot is released only when the caller consumes the
//! completion. Dropping the worker requests cancellation and detaches; it does
//! not establish cleanup or release the caller's reservation.

use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use super::catalog::{Device, Lane};
use super::decision::{
    execute_accepted_cpu_decision, DecisionDecodeError, DecisionExecutionError,
    NativeDecisionResult,
};
use super::inference::DispatchTicket;
use super::process::ProcessFailureDisposition;
use super::routes::NativeRoute;
use super::runtime::{Cancellation, OperationContext, RecoveryPolicy, ResourceClaim};

/// Fixed worker setup failures. Thread names, OS errors, and request data are
/// intentionally not retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionWorkerStartError {
    ThreadUnavailable,
}

impl fmt::Display for DecisionWorkerStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CPU decision worker could not be started")
    }
}

impl std::error::Error for DecisionWorkerStartError {}

/// Fixed reasons a handoff can be refused. The associated [`DecisionWork`]
/// remains available to the caller so a begun core reservation is not lost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionDispatchRefusalKind {
    Busy,
    Stopped,
    WorkerUnavailable,
}

pub struct DecisionDispatchRefusal {
    kind: DecisionDispatchRefusalKind,
    work: Box<DecisionWork>,
}

impl DecisionDispatchRefusal {
    pub fn kind(&self) -> DecisionDispatchRefusalKind {
        self.kind
    }

    pub fn work(&self) -> &DecisionWork {
        &self.work
    }

    pub fn into_work(self) -> DecisionWork {
        *self.work
    }
}

impl fmt::Debug for DecisionDispatchRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecisionDispatchRefusal")
            .field("kind", &self.kind)
            .field("work", &"<retained>")
            .finish()
    }
}

/// A ticket and the immutable native catalog snapshot needed by its worker.
/// The caller constructs this only after `begin_dispatch` succeeds.
pub struct DecisionWork {
    ticket: DispatchTicket,
    catalog: super::catalog::Catalog,
    recovery: RecoveryPolicy,
}

impl DecisionWork {
    pub fn new(
        ticket: DispatchTicket,
        catalog: super::catalog::Catalog,
        recovery: RecoveryPolicy,
    ) -> Self {
        Self {
            ticket,
            catalog,
            recovery,
        }
    }

    pub fn job_id(&self) -> u64 {
        self.ticket.job_id
    }

    pub fn into_ticket(self) -> DispatchTicket {
        self.ticket
    }
}

/// Fixed execution categories. This type contains no prompt, path, argv, child
/// output, source error, or credential.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionWorkerFailureKind {
    UnsupportedRoute,
    Unavailable,
    Cancelled,
    DeadlineExceeded,
    ProcessFailed,
    InvalidOutput(DecisionDecodeError),
    Internal,
}

/// Failure category plus the strongest process cleanup conclusion. In
/// particular, `OwnershipUncertain` must not be treated as a release receipt.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct DecisionWorkerFailure {
    kind: DecisionWorkerFailureKind,
    cleanup: ProcessFailureDisposition,
}

impl DecisionWorkerFailure {
    pub fn kind(self) -> DecisionWorkerFailureKind {
        self.kind
    }

    pub fn cleanup(self) -> ProcessFailureDisposition {
        self.cleanup
    }
}

impl fmt::Debug for DecisionWorkerFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecisionWorkerFailure")
            .field("kind", &self.kind)
            .field("cleanup", &self.cleanup)
            .finish()
    }
}

/// One final typed result or fixed failure. Native result bodies intentionally
/// have no `Debug` rendering through this wrapper.
pub enum DecisionWorkerOutcome {
    Succeeded(NativeDecisionResult),
    Failed(DecisionWorkerFailure),
}

impl DecisionWorkerOutcome {
    pub fn cleanup(&self) -> ProcessFailureDisposition {
        match self {
            // A successful process result is returned only after the process
            // runner confirmed the owned process group was gone.
            Self::Succeeded(_) => ProcessFailureDisposition::ChildGroupReaped,
            Self::Failed(failure) => failure.cleanup(),
        }
    }

    pub fn into_result(self) -> Result<NativeDecisionResult, DecisionWorkerFailure> {
        match self {
            Self::Succeeded(result) => Ok(result),
            Self::Failed(failure) => Err(failure),
        }
    }
}

impl fmt::Debug for DecisionWorkerOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Succeeded(_) => formatter.write_str("Succeeded(<redacted native result>)"),
            Self::Failed(failure) => formatter.debug_tuple("Failed").field(failure).finish(),
        }
    }
}

/// Completion for the caller to apply to core history and ResultBuffer.
/// Request IDs are hidden in Debug because callers control their contents.
pub struct DecisionWorkerCompletion {
    job_id: u64,
    request_id: String,
    outcome: DecisionWorkerOutcome,
}

impl DecisionWorkerCompletion {
    pub fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn outcome(&self) -> &DecisionWorkerOutcome {
        &self.outcome
    }

    pub fn into_parts(self) -> (u64, String, DecisionWorkerOutcome) {
        (self.job_id, self.request_id, self.outcome)
    }
}

impl fmt::Debug for DecisionWorkerCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecisionWorkerCompletion")
            .field("job_id", &self.job_id)
            .field("request_id", &"<redacted>")
            .field("outcome", &self.outcome)
            .finish()
    }
}

/// A one-job-at-a-time CPU decision executor. It does not mutate coordinator
/// state: the owner receives [`DecisionWorkerCompletion`] and decides whether
/// results can be committed and whether cleanup permits reservation release.
pub struct CpuDecisionWorker {
    work_sender: Option<SyncSender<DecisionWork>>,
    completion_receiver: Receiver<DecisionWorkerCompletion>,
    stop: Arc<AtomicBool>,
    in_flight: bool,
    thread: Option<JoinHandle<()>>,
}

impl CpuDecisionWorker {
    /// Start the single dedicated worker thread. Both channels are bounded to
    /// one item; `in_flight` prevents accepting another ticket until the prior
    /// completion is consumed.
    pub fn start() -> Result<Self, DecisionWorkerStartError> {
        let (work_sender, work_receiver) = mpsc::sync_channel::<DecisionWork>(1);
        let (completion_sender, completion_receiver) =
            mpsc::sync_channel::<DecisionWorkerCompletion>(1);
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("l3ms-cpu-decision".into())
            .spawn(move || {
                while let Ok(work) = work_receiver.recv() {
                    let fallback = DecisionWorkerCompletion {
                        job_id: work.ticket.job_id,
                        request_id: work.ticket.request_id.clone(),
                        outcome: DecisionWorkerOutcome::Failed(DecisionWorkerFailure {
                            kind: DecisionWorkerFailureKind::Internal,
                            cleanup: ProcessFailureDisposition::OwnershipUncertain,
                        }),
                    };
                    let completion =
                        catch_unwind(AssertUnwindSafe(|| execute_work(work, &thread_stop)))
                            .unwrap_or(fallback);
                    match completion_sender.try_send(completion) {
                        Ok(()) => {}
                        // This is unreachable while callers respect the
                        // in-flight gate. Losing a completion leaves the
                        // caller's durable reservation unresolved.
                        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => break,
                    }
                    if thread_stop.load(Ordering::Acquire) {
                        break;
                    }
                }
            })
            .map_err(|_| DecisionWorkerStartError::ThreadUnavailable)?;

        Ok(Self {
            work_sender: Some(work_sender),
            completion_receiver,
            stop,
            in_flight: false,
            thread: Some(thread),
        })
    }

    /// Try to hand off one begun ticket without blocking the daemon event loop.
    /// A refusal returns the whole work item, retaining the active ticket.
    pub fn try_dispatch(&mut self, work: DecisionWork) -> Result<(), DecisionDispatchRefusal> {
        if self.in_flight {
            return Err(DecisionDispatchRefusal {
                kind: DecisionDispatchRefusalKind::Busy,
                work: Box::new(work),
            });
        }
        let Some(sender) = &self.work_sender else {
            return Err(DecisionDispatchRefusal {
                kind: DecisionDispatchRefusalKind::Stopped,
                work: Box::new(work),
            });
        };
        match sender.try_send(work) {
            Ok(()) => {
                self.in_flight = true;
                Ok(())
            }
            Err(TrySendError::Full(work)) => Err(DecisionDispatchRefusal {
                kind: DecisionDispatchRefusalKind::Busy,
                work: Box::new(work),
            }),
            Err(TrySendError::Disconnected(work)) => Err(DecisionDispatchRefusal {
                kind: DecisionDispatchRefusalKind::WorkerUnavailable,
                work: Box::new(work),
            }),
        }
    }

    /// Nonblocking poll. Only consuming the result frees the worker's one-job
    /// slot. Channel closure without a completion is not cleanup proof.
    pub fn try_receive(
        &mut self,
    ) -> Result<Option<DecisionWorkerCompletion>, DecisionWorkerReceiveError> {
        match self.completion_receiver.try_recv() {
            Ok(completion) => {
                self.in_flight = false;
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(DecisionWorkerReceiveError::Closed),
        }
    }

    /// Wait only until the caller's supplied monotonic deadline. This does not
    /// extend the request deadline or claim process cleanup if it times out.
    pub fn receive_until(
        &mut self,
        deadline: Instant,
    ) -> Result<Option<DecisionWorkerCompletion>, DecisionWorkerReceiveError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match self.completion_receiver.recv_timeout(remaining) {
            Ok(completion) => {
                self.in_flight = false;
                Ok(Some(completion))
            }
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(DecisionWorkerReceiveError::Closed),
        }
    }

    pub fn is_in_flight(&self) -> bool {
        self.in_flight
    }

    /// Request cooperative cancellation and stop accepting tickets. Any active
    /// process still requires a completion/cleanup decision from the caller.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.work_sender.take();
    }

    /// Join only when the worker has already finished; this poll never waits.
    pub fn try_join_finished(&mut self) -> Result<bool, DecisionWorkerJoinError> {
        let finished = self.thread.as_ref().is_some_and(JoinHandle::is_finished);
        if !finished {
            return Ok(false);
        }
        let thread = self.thread.take().expect("finished thread handle exists");
        thread
            .join()
            .map_err(|_| DecisionWorkerJoinError::Panicked)?;
        Ok(true)
    }
}

impl Drop for CpuDecisionWorker {
    fn drop(&mut self) {
        self.shutdown();
        // Do not block the daemon's Drop path on native cleanup. Dropping the
        // handle detaches the thread; the owner must retain/reconcile the core
        // reservation rather than infer cleanup from this destructor.
        self.thread.take();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionWorkerReceiveError {
    Closed,
}

impl fmt::Display for DecisionWorkerReceiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CPU decision worker stopped before a completion was received")
    }
}

impl std::error::Error for DecisionWorkerReceiveError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionWorkerJoinError {
    Panicked,
}

impl fmt::Display for DecisionWorkerJoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CPU decision worker thread failed")
    }
}

impl std::error::Error for DecisionWorkerJoinError {}

fn execute_work(work: DecisionWork, stop: &AtomicBool) -> DecisionWorkerCompletion {
    let DecisionWork {
        ticket,
        catalog,
        recovery,
    } = work;
    let job_id = ticket.job_id;
    let request_id = ticket.request_id.clone();
    if !accepted_cpu_ticket(&ticket) {
        return failure_completion(
            job_id,
            request_id,
            DecisionWorkerFailureKind::UnsupportedRoute,
            ProcessFailureDisposition::NoChildSpawned,
        );
    }

    let cancellation = CombinedCancellation {
        request: ticket.cancellation.clone(),
        stop,
    };
    let context = OperationContext::new(ticket.deadline, &cancellation);
    if context
        .check_active("before CPU decision preparation")
        .is_err()
    {
        return failure_completion(
            job_id,
            request_id,
            inactive_failure_kind(&context),
            ProcessFailureDisposition::NoChildSpawned,
        );
    }

    match execute_accepted_cpu_decision(
        &catalog,
        &ticket.request,
        &ticket.route,
        &context,
        recovery,
    ) {
        Ok(result) => DecisionWorkerCompletion {
            job_id,
            request_id,
            outcome: DecisionWorkerOutcome::Succeeded(result),
        },
        Err(error) => {
            let (kind, cleanup) = decision_failure(&error, &context);
            failure_completion(job_id, request_id, kind, cleanup)
        }
    }
}

fn accepted_cpu_ticket(ticket: &DispatchTicket) -> bool {
    if ticket.resource != ResourceClaim::None
        || ticket.request.profile() != &ticket.route.profile
        || ticket.request.lane() != Lane::Decisions
        || ticket.route.profile.lane != Lane::Decisions
        || ticket.route.runtime.lane != Lane::Decisions
        || !matches!(ticket.route.runtime.as_str(), "nagi" | "laya")
        || ticket.route.device != Some(Device::Cpu)
        || !ticket
            .route
            .tasks
            .iter()
            .any(|task| task.as_str() == "decision")
    {
        return false;
    }
    let NativeRoute::DecisionRunner { profile_type, .. } = &ticket.route.native_route else {
        return false;
    };
    if profile_type != "local" {
        return false;
    }
    matches!(
        &ticket.request,
        super::requests::InferenceRequest::Decision { .. }
    )
}

fn decision_failure(
    error: &DecisionExecutionError,
    context: &OperationContext<'_>,
) -> (DecisionWorkerFailureKind, ProcessFailureDisposition) {
    match error {
        DecisionExecutionError::NotDispatched => (
            inactive_failure_kind(context),
            ProcessFailureDisposition::NoChildSpawned,
        ),
        DecisionExecutionError::Uncertain { process } => {
            let kind = inactive_failure_kind(context);
            let kind = if matches!(kind, DecisionWorkerFailureKind::Unavailable) {
                DecisionWorkerFailureKind::ProcessFailed
            } else {
                kind
            };
            (kind, *process)
        }
        DecisionExecutionError::InvalidResult(error) => (
            DecisionWorkerFailureKind::InvalidOutput(*error),
            // A malformed native result is returned only after successful
            // child exit and process-group cleanup.
            ProcessFailureDisposition::ChildGroupReaped,
        ),
    }
}

fn inactive_failure_kind(context: &OperationContext<'_>) -> DecisionWorkerFailureKind {
    if context.is_cancelled() {
        DecisionWorkerFailureKind::Cancelled
    } else if context.remaining().is_zero() {
        DecisionWorkerFailureKind::DeadlineExceeded
    } else {
        DecisionWorkerFailureKind::Unavailable
    }
}

fn failure_completion(
    job_id: u64,
    request_id: String,
    kind: DecisionWorkerFailureKind,
    cleanup: ProcessFailureDisposition,
) -> DecisionWorkerCompletion {
    DecisionWorkerCompletion {
        job_id,
        request_id,
        outcome: DecisionWorkerOutcome::Failed(DecisionWorkerFailure { kind, cleanup }),
    }
}

struct CombinedCancellation<'a> {
    request: super::inference::RequestCancellation,
    stop: &'a AtomicBool,
}

impl Cancellation for CombinedCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        self.request.is_cancelled() || self.stop.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::time::Duration;

    use tempfile::TempDir;

    use super::super::catalog::{Catalog, ProfileKey};
    use super::super::inference::CoordinatorCore;
    use super::super::requests::InferenceRequest;

    struct Fixture {
        _root: TempDir,
        catalog: Catalog,
        core: CoordinatorCore,
    }

    impl Fixture {
        fn new(script_body: &str) -> Self {
            let root = tempfile::tempdir().unwrap();
            let media = root.path().join("toolkits/l2m2");
            let decisions = root.path().join("toolkits/l3ds");
            fs::create_dir_all(&media).unwrap();
            fs::create_dir_all(decisions.join("decision-models")).unwrap();
            fs::create_dir_all(decisions.join(".venv/bin")).unwrap();
            fs::write(
                root.path().join("llama-swap.yaml"),
                "models:\n  fixture:\n    cmd: fixture\n",
            )
            .unwrap();
            fs::write(
                media.join("media-runtimes.json"),
                r#"{"schema_version":1,"profiles":[]}"#,
            )
            .unwrap();
            fs::write(
                decisions.join("decision-runtimes.json"),
                r#"{"version":"1.0","profiles":[{"id":"fixture","name":"Fixture CPU decision","engine":"nagi","type":"local","device":"cpu","model":"fixture-model","entrypoint":"decision-models/runner.py"}]}"#,
            )
            .unwrap();
            let _guard = super::super::process::executable_fixture_guard();
            let interpreter = decisions.join(".venv/bin/python");
            fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
            fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
            fs::write(decisions.join("decision-models/runner.py"), script_body).unwrap();
            fs::set_permissions(
                decisions.join("decision-models/runner.py"),
                fs::Permissions::from_mode(0o700),
            )
            .unwrap();
            drop(_guard);

            let catalog = Catalog::load(root.path()).unwrap();
            let identity =
                super::super::catalog::RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap();
            let data_root = root.path().join("state");
            fs::create_dir(&data_root).unwrap();
            let mut core = CoordinatorCore::load(
                catalog.clone(),
                BTreeMap::from([(identity, ResourceClaim::None)]),
                &data_root,
                root.path(),
                4,
            )
            .unwrap();
            core.confirm_reconciled().unwrap();
            Self {
                _root: root,
                catalog,
                core,
            }
        }

        fn ticket(&mut self, request_id: &str, timeout: Duration) -> DispatchTicket {
            let request = InferenceRequest::Decision {
                profile: ProfileKey::new(Lane::Decisions, "fixture").unwrap(),
                state: serde_json::json!({"case":"fixture"}),
                questions: serde_json::json!({
                    "route": {
                        "type":"choice",
                        "criteria":{"secret-answer":"Proceed","no":"Stop"}
                    }
                })
                .as_object()
                .unwrap()
                .clone(),
            };
            let now = Instant::now();
            self.core
                .submit(request_id.to_owned(), request, timeout, now)
                .unwrap();
            let ticket = self.core.reserve_next(now).unwrap().unwrap();
            self.core.begin_dispatch(&ticket, now).unwrap();
            ticket
        }
    }

    fn work(fixture: &mut Fixture, id: &str, timeout: Duration) -> DecisionWork {
        DecisionWork::new(
            fixture.ticket(id, timeout),
            fixture.catalog.clone(),
            RecoveryPolicy::new(Duration::from_secs(1)).unwrap(),
        )
    }

    fn success_script(marker: &Path) -> String {
        format!(
            "#!/bin/sh\n[ \"$HF_HUB_OFFLINE\" = 1 ] || exit 3\n[ \"$TRANSFORMERS_OFFLINE\" = 1 ] || exit 4\n[ \"$HF_DATASETS_OFFLINE\" = 1 ] || exit 5\n[ -z \"$CUDA_VISIBLE_DEVICES\" ] || exit 6\ncase \" $* \" in *\" --device cpu \"*) ;; *) exit 7 ;; esac\nprintf x >> {}\ncat >/dev/null\nprintf '%s\\n' '{{\"engine\":\"nagi-native\",\"answers\":{{\"route\":{{\"choice\":\"secret-answer\"}}}},\"latency_ms\":12.5}}'\n",
            shell_quote(marker)
        )
    }

    fn sleeping_script(marker: &Path) -> String {
        format!(
            "#!/bin/sh\ntouch {}\nexec /bin/sleep 30\n",
            shell_quote(marker)
        )
    }

    fn shell_quote(path: &Path) -> String {
        format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
    }

    fn receive(worker: &mut CpuDecisionWorker) -> DecisionWorkerCompletion {
        worker
            .receive_until(Instant::now() + Duration::from_secs(4))
            .expect("worker channel remains available")
            .expect("completion arrives within bounded recovery")
    }

    fn wait_for(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if path.exists() {
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("fixture child did not reach its marker");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn one_bounded_worker_returns_native_result_without_debugging_payloads_or_retrying() {
        let root = tempfile::tempdir().unwrap();
        let marker = root.path().join("calls");
        let mut fixture = Fixture::new(&success_script(&marker));
        let work = work(&mut fixture, "secret-request-id", Duration::from_secs(3));
        let mut worker = CpuDecisionWorker::start().unwrap();
        worker.try_dispatch(work).unwrap();
        assert!(worker.is_in_flight());
        let completion = receive(&mut worker);
        assert_eq!(completion.job_id(), 1);
        assert!(!worker.is_in_flight());
        assert_eq!(fs::read_to_string(marker).unwrap(), "x");
        let debug = format!("{completion:?}");
        assert!(!debug.contains("secret-request-id"));
        assert!(!debug.contains("secret-answer"));
        assert!(
            matches!(completion.outcome(), DecisionWorkerOutcome::Succeeded(_)),
            "unexpected worker outcome: {:?}",
            completion.outcome()
        );
        let (_, _, outcome) = completion.into_parts();
        let DecisionWorkerOutcome::Succeeded(result) = outcome else {
            panic!("native CPU result expected")
        };
        let super::super::requests::InferenceResult::Decision { answers } = result.result else {
            panic!("decision result expected")
        };
        assert_eq!(answers["route"]["choice"], "secret-answer");
        assert_eq!(result.native_latency, Some(Duration::from_micros(12_500)));
    }

    #[test]
    fn cpu_worker_refuses_wrong_device_binding_runtime_and_profile_type() {
        let mut fixture = Fixture::new("#!/bin/sh\nexit 99\n");
        let mut ticket = fixture.ticket("wrong-route", Duration::from_secs(2));
        assert!(accepted_cpu_ticket(&ticket));

        ticket.resource = ResourceClaim::ExclusiveGpu(0);
        assert!(!accepted_cpu_ticket(&ticket));
        ticket.resource = ResourceClaim::None;

        ticket.route.device = Some(Device::Cuda);
        assert!(!accepted_cpu_ticket(&ticket));
        ticket.route.device = Some(Device::Cpu);

        ticket.route.runtime =
            super::super::catalog::RuntimeIdentity::new(Lane::Decisions, "jev").unwrap();
        assert!(!accepted_cpu_ticket(&ticket));
        ticket.route.runtime =
            super::super::catalog::RuntimeIdentity::new(Lane::Decisions, "nagi").unwrap();

        let NativeRoute::DecisionRunner { profile_type, .. } = &mut ticket.route.native_route
        else {
            panic!("fixture route must be a decision runner")
        };
        *profile_type = "api".into();
        assert!(!accepted_cpu_ticket(&ticket));

        let work = DecisionWork::new(
            ticket,
            fixture.catalog.clone(),
            RecoveryPolicy::new(Duration::from_secs(1)).unwrap(),
        );
        let mut worker = CpuDecisionWorker::start().unwrap();
        worker.try_dispatch(work).unwrap();
        let completion = receive(&mut worker);
        let DecisionWorkerOutcome::Failed(failure) = completion.outcome() else {
            panic!("invalid route must not execute")
        };
        assert_eq!(failure.kind(), DecisionWorkerFailureKind::UnsupportedRoute);
        assert_eq!(failure.cleanup(), ProcessFailureDisposition::NoChildSpawned);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn cancellation_reaps_child_group_and_busy_refusal_returns_its_ticket() {
        let root = tempfile::tempdir().unwrap();
        let marker = root.path().join("started");
        let mut fixture = Fixture::new(&sleeping_script(&marker));
        let first = work(&mut fixture, "request-one", Duration::from_secs(5));
        let second = work(&mut fixture, "request-two", Duration::from_secs(5));
        let cancellation = first.ticket.cancellation.clone();
        let mut worker = CpuDecisionWorker::start().unwrap();
        worker.try_dispatch(first).unwrap();
        let refused = worker.try_dispatch(second).unwrap_err();
        assert_eq!(refused.kind(), DecisionDispatchRefusalKind::Busy);
        let second_work = refused.into_work();
        assert_eq!(second_work.job_id(), 2);
        wait_for(&marker);
        cancellation.cancel();
        let completion = receive(&mut worker);
        let DecisionWorkerOutcome::Failed(failure) = completion.outcome() else {
            panic!("cancelled work cannot succeed")
        };
        assert_eq!(failure.kind(), DecisionWorkerFailureKind::Cancelled);
        assert_eq!(
            failure.cleanup(),
            ProcessFailureDisposition::ChildGroupReaped
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn deadline_reaps_started_child_group() {
        let root = tempfile::tempdir().unwrap();
        let marker = root.path().join("started");
        let mut fixture = Fixture::new(&sleeping_script(&marker));
        let work = work(&mut fixture, "deadline-request", Duration::from_secs(2));
        let mut worker = CpuDecisionWorker::start().unwrap();
        worker.try_dispatch(work).unwrap();
        wait_for(&marker);
        // The ticket deadline is carried into the child operation context. For
        // this fixture, the manifest/interpreter is already prepared by the
        // time the marker appears; cancellation below is not needed.
        let completion = receive(&mut worker);
        let DecisionWorkerOutcome::Failed(failure) = completion.outcome() else {
            panic!("long-running work cannot succeed before deadline")
        };
        assert_eq!(failure.kind(), DecisionWorkerFailureKind::DeadlineExceeded);
        assert_eq!(
            failure.cleanup(),
            ProcessFailureDisposition::ChildGroupReaped
        );
    }
}
