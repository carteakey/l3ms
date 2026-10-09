//! Linux control-loop owner with explicit opt-in CPU-local dispatch.
//!
//! Binding acquires listener/host/device ownership before history is loaded or
//! reconciled. The default loop keeps admission closed; only `run_cpu_decisions`
//! dispatches restricted CPU-local work. General managed-engine reconciliation
//! remains gated, and neither loop exposes its core for mutation. Each invocation consumes the owner
//! and releases its endpoint and locks on every return path.
//!
//! One authenticated peer is served at a time, one exchange per connection.
//! Peer time and accepted/denied connection attempts are bounded. Same-user
//! denial of service is still possible; UID authentication is not app isolation.
//! Local filesystem operations are synchronous and cannot be preempted. Caller
//! supplied private local directories and their ancestors must remain trusted.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, TryRecvError, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use super::catalog::{Catalog, RuntimeIdentity};
use super::decision_worker::{
    CpuDecisionWorker, DecisionDispatchRefusalKind, DecisionWork, DecisionWorkerCompletion,
    DecisionWorkerFailureKind, DecisionWorkerOutcome, DecisionWorkerReceiveError,
};
use super::inference::{Completion, CoordinatorCore, DispatchBeginError};
use super::ipc::{
    AcceptedOperation, ClientRequest, IncomingRequest, RequestStatus, ResponseEnvelope,
    ResponseErrorCode, ResponseMessage, ResultsQuery,
};
use super::local_ipc::{private_directory, LocalIpcError, LocalListener};
#[cfg(target_os = "linux")]
use super::managed_session::{ManagedJobKey, ManagedWorkerCommand};
use super::process::ProcessFailureDisposition;
use super::result_buffer::{
    ResultBuffer, ResultBufferError, ResultEvent, ResultFailureCode, ResultTerminal,
};
#[cfg(target_os = "linux")]
use super::runtime::Cancellation;
use super::runtime::{OperationContext, RecoveryPolicy, ResourceClaim};

const CPU_ACCEPT_POLL: Duration = Duration::from_millis(25);
const CPU_JOIN_POLL: Duration = Duration::from_millis(5);

/// Explicit paths: no directory creation, default live store, or stale reclaim.
pub struct DaemonConfig {
    pub socket_directory: PathBuf,
    pub lock_directory: PathBuf,
    pub data_directory: PathBuf,
    /// Legacy UI/bench store to exclude. May not exist yet; never created here.
    pub standalone_data_directory: PathBuf,
    pub gpus: Vec<u32>,
    pub queue_capacity: usize,
    pub peer_timeout: Duration,
    pub max_connections: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DaemonError {
    InvalidConfig,
    Transport(LocalIpcError),
    HistoryUnavailable,
    HistoryConflict,
    CpuDecisionsUnavailable,
    DecisionWorkerUnavailable,
    ManagedSessionUnavailable,
    ResultBufferInvariant,
}
impl std::fmt::Display for DaemonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "coordinator loop: {self:?}")
    }
}
impl std::error::Error for DaemonError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    Cancelled,
    Deadline,
    ConnectionLimit,
    WorkerUncertain,
}

/// Counts only; never includes peer data, request IDs, paths or errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunReport {
    /// Full request/reply exchanges, including business-error responses.
    pub completed_exchanges: usize,
    /// Denied or failed connections, NOT rejected inference requests. A failed
    /// reply can follow a durable mutation: reconnect by ID, never replay.
    pub failed_connections: usize,
    pub stopped: StopReason,
}

pub struct ControlDaemon {
    // Drop history/payload ownership before removing the socket and host guard.
    core: CoordinatorCore,
    // Result bodies are intentionally process-local; durable history in core
    // remains authoritative after buffer loss or daemon restart.
    results: ResultBuffer,
    listener: LocalListener,
    peer_timeout: Duration,
    max_connections: usize,
}

impl ControlDaemon {
    pub fn bind(
        config: DaemonConfig,
        catalog: Catalog,
        bindings: BTreeMap<RuntimeIdentity, ResourceClaim>,
        repository_root: &Path,
    ) -> Result<Self, DaemonError> {
        if !(1..=128).contains(&config.queue_capacity)
            || !(1..=4096).contains(&config.max_connections)
            || config.peer_timeout.is_zero()
            || config.peer_timeout > Duration::from_secs(30)
            || bindings.values().any(|claim| {
                matches!(claim, ResourceClaim::ExclusiveGpu(gpu) if !config.gpus.contains(gpu))
            })
        {
            return Err(DaemonError::InvalidConfig);
        }
        let data = private_directory(&config.data_directory).map_err(DaemonError::Transport)?;
        verify_history_isolation(&data, &config.standalone_data_directory)?;
        let listener = LocalListener::bind(
            &config.socket_directory,
            &config.lock_directory,
            config.gpus,
        )
        .map_err(DaemonError::Transport)?;
        let core = CoordinatorCore::load(
            catalog,
            bindings,
            data,
            repository_root,
            config.queue_capacity,
        )
        .map_err(|_| DaemonError::HistoryUnavailable)?;
        Ok(Self {
            core,
            results: ResultBuffer::new(),
            listener,
            peer_timeout: config.peer_timeout,
            max_connections: config.max_connections,
        })
    }

    /// Consuming, bounded session. A failed exchange is never retried. The
    /// overall deadline also limits a peer arriving just before shutdown.
    /// Cancellation/deadline stops intake; no backend was started to clean up.
    pub fn run(mut self, context: &OperationContext<'_>) -> Result<RunReport, DaemonError> {
        let mut report = RunReport {
            completed_exchanges: 0,
            failed_connections: 0,
            stopped: StopReason::ConnectionLimit,
        };
        for _ in 0..self.max_connections {
            if let Some(reason) = stop_reason(context) {
                report.stopped = reason;
                return Ok(report);
            }
            let mut peer = match self.listener.accept(context) {
                Ok(peer) => peer,
                Err(LocalIpcError::Cancelled) => {
                    report.stopped = StopReason::Cancelled;
                    return Ok(report);
                }
                Err(LocalIpcError::Deadline) => {
                    report.stopped = StopReason::Deadline;
                    return Ok(report);
                }
                Err(LocalIpcError::PeerDenied) => {
                    report.failed_connections += 1;
                    continue;
                }
                Err(error) => return Err(DaemonError::Transport(error)),
            };
            let deadline = (Instant::now() + self.peer_timeout).min(context.deadline());
            let cancellation = ForwardCancellation(context);
            let peer_context = OperationContext::new(deadline, &cancellation);
            if self.serve_connection(&mut peer, &peer_context).is_ok() {
                report.completed_exchanges += 1;
            } else {
                report.failed_connections += 1;
            }
            if let Some(reason) = stop_reason(context) {
                report.stopped = reason;
                return Ok(report);
            }
        }
        Ok(report)
    }

    /// Opt in to the concrete, single-supervisor llama-swap language path.
    /// The main thread retains history, durable reservations, results and IPC;
    /// one scoped worker borrows this listener's exact host guard for the
    /// complete owner/activation/inference/cleanup lifetime.
    #[cfg(target_os = "linux")]
    pub(crate) fn run_managed_language(
        mut self,
        config: super::managed_session::ManagedSessionConfig,
        context: &OperationContext<'_>,
    ) -> Result<RunReport, DaemonError> {
        let mut report = RunReport {
            completed_exchanges: 0,
            failed_connections: 0,
            stopped: StopReason::ConnectionLimit,
        };
        if let Some(reason) = stop_reason(context) {
            report.stopped = reason;
            return Ok(report);
        }
        let expected_gpu = config.gpu.expected_index();
        let comfy_profiles = config
            .comfy
            .as_ref()
            .map(|comfy| comfy.profiles().clone())
            .unwrap_or_default();
        if config.local_job_profiles.is_empty() && comfy_profiles.is_empty() {
            self.core
                .require_managed_language_startup(&config.runtime, expected_gpu)
        } else {
            self.core.require_managed_stack_startup(
                &config.runtime,
                expected_gpu,
                &config.startup_profile,
                &config.supported_model_tasks,
                &config.local_job_profiles,
                &comfy_profiles,
            )
        }
        .map_err(|_| DaemonError::ManagedSessionUnavailable)?;

        let runtime = config.runtime.clone();
        let startup_profile = config.startup_profile.clone();
        let supported_model_tasks = config.supported_model_tasks.clone();
        let local_job_profiles = config.local_job_profiles.clone();
        let session_cancellation = super::managed_session::ManagedWorkerCancellation::new();
        let worker_cancel = session_cancellation.clone();
        let (command_sender, command_receiver) = mpsc::sync_channel(1);
        let (event_sender, event_receiver) = mpsc::sync_channel(1);
        let session_deadline = context.deadline();
        let mut fatal_error = None;

        std::thread::scope(|scope| {
            let ControlDaemon {
                core,
                results,
                listener,
                peer_timeout,
                max_connections,
            } = &mut self;
            let host_guard = listener.ownership_guard();
            let catalog = core.catalog().clone();
            let worker = scope.spawn(move || {
                super::managed_session::run_worker(
                    config,
                    catalog,
                    host_guard,
                    session_deadline,
                    worker_cancel,
                    command_receiver,
                    event_sender,
                )
            });

            let mut worker_ready = false;
            let mut worker_finished = false;
            let mut worker_shutdown_sent = false;
            let mut stop_cleanup_started = false;
            let mut stopping = false;
            let mut handled_connections = 0usize;
            let mut accepted_result_ids = BTreeSet::new();
            let mut active_slot: Option<ManagedJobKey> = None;
            let mut cleanup_ack_pending = false;

            loop {
                if !stopping {
                    if let Some(reason) = stop_reason(context) {
                        report.stopped = reason;
                        stopping = true;
                    } else if handled_connections >= *max_connections {
                        stopping = true;
                    }
                }

                match event_receiver.try_recv() {
                    Ok(super::managed_session::ManagedWorkerEvent::Ready(readiness)) => {
                        if !managed_ready_may_open_admission(
                            stopping,
                            session_cancellation.is_cancelled(),
                        ) {
                            // A shutdown may race with the final startup
                            // observation. Do not reopen admission after the
                            // listener has already committed to stop.
                        } else if worker_ready
                            || (if local_job_profiles.is_empty() && comfy_profiles.is_empty() {
                                core.confirm_managed_language_ready(
                                    &runtime,
                                    expected_gpu,
                                    &startup_profile,
                                    supported_model_tasks.clone(),
                                    &readiness,
                                )
                            } else {
                                core.confirm_managed_stack_ready(
                                    &runtime,
                                    expected_gpu,
                                    &startup_profile,
                                    supported_model_tasks.clone(),
                                    super::inference::ManagedStackProfiles {
                                        local_jobs: local_job_profiles.clone(),
                                        comfy: comfy_profiles.clone(),
                                    },
                                    &readiness,
                                )
                            })
                            .is_err()
                        {
                            core.quarantine();
                            report.stopped = StopReason::WorkerUncertain;
                            stopping = true;
                        } else {
                            worker_ready = true;
                        }
                    }
                    Ok(super::managed_session::ManagedWorkerEvent::StartupFailed) => {
                        core.quarantine();
                        report.stopped = StopReason::WorkerUncertain;
                        stopping = true;
                        worker_finished = true;
                    }
                    Ok(super::managed_session::ManagedWorkerEvent::Append {
                        key,
                        event,
                        reply,
                    }) => {
                        let appended = if active_slot.as_ref() == Some(&key) {
                            results.append(&key.request_id, event).is_ok()
                        } else {
                            false
                        };
                        if !appended {
                            core.quarantine();
                            report.stopped = StopReason::WorkerUncertain;
                            stopping = true;
                        }
                        let _ = reply.send(appended);
                    }
                    Ok(super::managed_session::ManagedWorkerEvent::ComfyCandidate {
                        ticket,
                        candidate,
                        generation,
                        reply,
                    }) => {
                        let permit = if !stopping
                            && active_slot.as_ref().is_some_and(|key| {
                                key.job_id == ticket.job_id && key.request_id == ticket.request_id
                            }) {
                            core.record_managed_comfy_dispatch_candidate(
                                &ticket,
                                candidate,
                                *generation,
                                Instant::now(),
                            )
                            .ok()
                            .filter(|permit| core.accepts_managed_comfy_dispatch_permit(permit))
                        } else {
                            None
                        };
                        let accepted = permit.is_some();
                        // A persisted candidate whose reply was lost is not a
                        // license to mint another ID or repeat submission.
                        if reply.send(permit).is_err() || !accepted {
                            core.quarantine();
                            report.stopped = StopReason::WorkerUncertain;
                            stopping = true;
                        }
                    }
                    Ok(super::managed_session::ManagedWorkerEvent::ComfyBackendAcknowledged {
                        acknowledgement,
                        reply,
                    }) => {
                        let matches_slot = active_slot.as_ref().is_some_and(|key| {
                            key.job_id == acknowledgement.job_id()
                                && key.request_id == acknowledgement.request_id()
                        });
                        let accepted = matches_slot
                            && core
                                .accept_managed_comfy_backend_acknowledgement(*acknowledgement)
                                .is_ok();
                        if reply.send(accepted).is_err() || !accepted {
                            core.quarantine();
                            report.stopped = StopReason::WorkerUncertain;
                            stopping = true;
                        }
                    }
                    Ok(super::managed_session::ManagedWorkerEvent::JobFinished(completion)) => {
                        let stop_session = completion.stop_session;
                        match consume_managed_completion(
                            core,
                            results,
                            completion,
                            &mut active_slot,
                            &mut accepted_result_ids,
                            &mut cleanup_ack_pending,
                        ) {
                            Ok(()) => {}
                            Err(error) => {
                                fatal_error = Some(error);
                                report.stopped = StopReason::WorkerUncertain;
                                stopping = true;
                            }
                        }
                        if cleanup_ack_pending {
                            stopping = true;
                        }
                        if stop_session {
                            report.stopped = StopReason::WorkerUncertain;
                            stopping = true;
                        }
                    }
                    Ok(super::managed_session::ManagedWorkerEvent::Stopped { cleanup_proven }) => {
                        worker_finished = true;
                        stopping = true;
                        if !cleanup_proven {
                            core.quarantine();
                            report.stopped = StopReason::WorkerUncertain;
                        }
                        if cleanup_ack_pending && cleanup_proven {
                            if let Some(slot) = active_slot.clone() {
                                if core.acknowledge_reconciled_cleanup(slot.job_id).is_ok() {
                                    active_slot = None;
                                    if results
                                        .terminal(&slot.request_id, ResultTerminal::Interrupted)
                                        .is_ok()
                                    {
                                        accepted_result_ids.remove(&slot.request_id);
                                        cleanup_ack_pending = false;
                                    } else {
                                        core.quarantine();
                                        report.stopped = StopReason::WorkerUncertain;
                                    }
                                } else {
                                    core.quarantine();
                                    report.stopped = StopReason::WorkerUncertain;
                                }
                            }
                        }
                    }
                    Err(TryRecvError::Disconnected) if worker.is_finished() => {
                        worker_finished = true;
                        core.quarantine();
                        report.stopped = StopReason::WorkerUncertain;
                        if active_slot.is_some() {
                            if let Some(slot) = active_slot.as_ref() {
                                let _ = core.complete(
                                    slot.job_id,
                                    Completion::Uncertain,
                                    Vec::new(),
                                    Instant::now(),
                                );
                            }
                        }
                        stopping = true;
                    }
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => {}
                }

                if let Err(error) = settle_managed_terminal_result_slots(
                    core,
                    results,
                    &mut accepted_result_ids,
                    active_slot.as_ref(),
                ) {
                    fatal_error = Some(error);
                    report.stopped = StopReason::WorkerUncertain;
                    stopping = true;
                }

                // Any transition to stopping (including worker/protocol
                // failures discovered above) closes intake, cancels queued
                // durable submissions, and interrupts the one active ticket.
                // This is deliberately centralized so no failure branch can
                // skip cancellation merely because `stopping` was already
                // set in the same loop iteration.
                if stopping && !stop_cleanup_started {
                    stop_cleanup_started = true;
                    session_cancellation.cancel();
                    let mut cleanup_uncertain = false;
                    if let Some(slot) = active_slot.as_ref() {
                        if core.cancel_request(&slot.request_id).is_err() {
                            cleanup_uncertain = true;
                        }
                    }
                    if cancel_managed_queued(
                        core,
                        results,
                        &mut accepted_result_ids,
                        active_slot.as_ref(),
                    )
                    .is_err()
                    {
                        cleanup_uncertain = true;
                    }
                    if cleanup_uncertain {
                        core.quarantine();
                        report.stopped = StopReason::WorkerUncertain;
                    }
                }

                if !stopping && worker_ready && active_slot.is_none() {
                    match core.reserve_next(Instant::now()) {
                        Ok(Some(ticket)) => {
                            let key = ManagedJobKey {
                                job_id: ticket.job_id,
                                request_id: ticket.request_id.clone(),
                            };
                            match core.begin_dispatch(&ticket, Instant::now()) {
                                Ok(()) => {
                                    active_slot = Some(key);
                                    if command_sender
                                        .try_send(ManagedWorkerCommand::Dispatch(Box::new(ticket)))
                                        .is_err()
                                    {
                                        core.quarantine();
                                        report.stopped = StopReason::WorkerUncertain;
                                        stopping = true;
                                    }
                                }
                                Err(error) => {
                                    let (completion, terminal) = dispatch_error_result(error);
                                    if core
                                        .complete(
                                            ticket.job_id,
                                            completion,
                                            Vec::new(),
                                            Instant::now(),
                                        )
                                        .is_err()
                                        || results.terminal(&ticket.request_id, terminal).is_err()
                                    {
                                        core.quarantine();
                                        report.stopped = StopReason::WorkerUncertain;
                                        stopping = true;
                                    } else {
                                        accepted_result_ids.remove(&ticket.request_id);
                                    }
                                    if core.admission() != super::inference::AdmissionState::Ready {
                                        report.stopped = StopReason::WorkerUncertain;
                                        stopping = true;
                                    }
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(_) => {
                            core.quarantine();
                            report.stopped = StopReason::WorkerUncertain;
                            stopping = true;
                        }
                    }
                }

                if stopping && active_slot.is_none() && !worker_shutdown_sent && !worker_finished {
                    worker_shutdown_sent = true;
                    match command_sender.try_send(ManagedWorkerCommand::Shutdown) {
                        Ok(()) => {}
                        Err(TrySendError::Disconnected(_)) if worker.is_finished() => {
                            worker_finished = true;
                        }
                        Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                            core.quarantine();
                            report.stopped = StopReason::WorkerUncertain;
                        }
                    }
                }
                if stopping && worker_finished {
                    break;
                }

                if stopping {
                    thread::sleep(CPU_JOIN_POLL);
                    continue;
                }

                let accept_deadline = (Instant::now() + CPU_ACCEPT_POLL).min(context.deadline());
                let accept_cancel = ForwardCancellation(context);
                let accept_context = OperationContext::new(accept_deadline, &accept_cancel);
                let mut peer = match listener.accept(&accept_context) {
                    Ok(peer) => {
                        handled_connections += 1;
                        peer
                    }
                    Err(LocalIpcError::PeerDenied) => {
                        handled_connections += 1;
                        report.failed_connections += 1;
                        continue;
                    }
                    Err(LocalIpcError::Deadline) => continue,
                    Err(LocalIpcError::Cancelled) => continue,
                    Err(error) => {
                        fatal_error = Some(DaemonError::Transport(error));
                        report.stopped = StopReason::WorkerUncertain;
                        stopping = true;
                        continue;
                    }
                };
                let peer_deadline = (Instant::now() + *peer_timeout).min(context.deadline());
                let peer_cancel = ForwardCancellation(context);
                let peer_context = OperationContext::new(peer_deadline, &peer_cancel);
                if serve_managed_connection(
                    core,
                    results,
                    &mut peer,
                    &peer_context,
                    &mut accepted_result_ids,
                )
                .is_ok()
                {
                    report.completed_exchanges += 1;
                } else {
                    report.failed_connections += 1;
                }
            }

            if worker.join().is_err() {
                core.quarantine();
                report.stopped = StopReason::WorkerUncertain;
                fatal_error = Some(DaemonError::ManagedSessionUnavailable);
            }
            if let Some(error) = fatal_error {
                Err(error)
            } else {
                Ok(report)
            }
        })
    }

    /// Opt-in, bounded execution for declared CPU-local Nagi/Laya decision
    /// routes only. The default [`Self::run`] remains control-only. Startup
    /// refuses GPU/hosted bindings and retained work needing reconciliation.
    /// Every accepted submit reserves its full result allowance before the
    /// acceptance response can be written; execution is serial and never
    /// replayed. Shutdown stops intake, cancels accepted work, and waits under
    /// an independent recovery bound while this daemon still owns its listener
    /// and host guard. Missing completion or uncertain child cleanup leaves an
    /// interrupted durable row and retained core reservation.
    pub fn run_cpu_decisions(
        mut self,
        context: &OperationContext<'_>,
        recovery: RecoveryPolicy,
    ) -> Result<RunReport, DaemonError> {
        let mut report = RunReport {
            completed_exchanges: 0,
            failed_connections: 0,
            stopped: StopReason::ConnectionLimit,
        };
        if let Some(reason) = stop_reason(context) {
            report.stopped = reason;
            return Ok(report);
        }
        self.core
            .confirm_cpu_decisions_ready()
            .map_err(|_| DaemonError::CpuDecisionsUnavailable)?;
        let mut worker =
            CpuDecisionWorker::start().map_err(|_| DaemonError::DecisionWorkerUnavailable)?;

        let mut accepted_result_ids = BTreeSet::new();
        let mut active_ticket: Option<CpuDecisionSlot> = None;
        let mut handled_connections = 0usize;
        let mut fatal_error = None;

        while handled_connections < self.max_connections {
            if let Some(reason) = stop_reason(context) {
                report.stopped = reason;
                break;
            }

            match worker.try_receive() {
                Ok(Some(completion)) => match self.consume_cpu_completion(
                    completion,
                    &mut active_ticket,
                    &mut accepted_result_ids,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        report.stopped = StopReason::WorkerUncertain;
                        break;
                    }
                    Err(error) => {
                        fatal_error = Some(error);
                        report.stopped = StopReason::WorkerUncertain;
                        break;
                    }
                },
                Ok(None) => {}
                Err(DecisionWorkerReceiveError::Closed) => {
                    if let Some(slot) = active_ticket.take() {
                        self.mark_cpu_slot_uncertain(&slot, &mut accepted_result_ids);
                    } else {
                        self.core.quarantine();
                    }
                    report.stopped = StopReason::WorkerUncertain;
                    break;
                }
            }

            if let Err(error) = self.settle_terminal_result_slots(&mut accepted_result_ids) {
                fatal_error = Some(error);
                report.stopped = StopReason::WorkerUncertain;
                break;
            }

            if active_ticket.is_none() && !worker.is_in_flight() {
                match self.core.reserve_next(Instant::now()) {
                    Ok(Some(ticket)) => {
                        let job_id = ticket.job_id;
                        let request_id = ticket.request_id.clone();
                        match self.core.begin_dispatch(&ticket, Instant::now()) {
                            Ok(()) => {
                                let slot = CpuDecisionSlot { job_id, request_id };
                                let work = DecisionWork::new(
                                    ticket,
                                    self.core.catalog().clone(),
                                    recovery,
                                );
                                match worker.try_dispatch(work) {
                                    Ok(()) => active_ticket = Some(slot),
                                    Err(refusal) => {
                                        let kind = refusal.kind();
                                        let ticket = refusal.into_work().into_ticket();
                                        let refused_slot = CpuDecisionSlot {
                                            job_id: ticket.job_id,
                                            request_id: ticket.request_id,
                                        };
                                        let code = if kind == DecisionDispatchRefusalKind::Busy {
                                            ResultFailureCode::Internal
                                        } else {
                                            ResultFailureCode::Unavailable
                                        };
                                        if self
                                            .complete_cpu_without_child(
                                                &refused_slot,
                                                Completion::Failed,
                                                ResultTerminal::Failed(code),
                                                &mut accepted_result_ids,
                                            )
                                            .is_err()
                                        {
                                            fatal_error = Some(DaemonError::HistoryUnavailable);
                                            report.stopped = StopReason::WorkerUncertain;
                                            break;
                                        }
                                        self.core.quarantine();
                                        report.stopped = StopReason::WorkerUncertain;
                                        break;
                                    }
                                }
                            }
                            Err(error) => {
                                let (outcome, terminal) = dispatch_error_result(error);
                                if self
                                    .complete_cpu_without_child(
                                        &CpuDecisionSlot { job_id, request_id },
                                        outcome,
                                        terminal,
                                        &mut accepted_result_ids,
                                    )
                                    .is_err()
                                {
                                    fatal_error = Some(DaemonError::HistoryUnavailable);
                                    report.stopped = StopReason::WorkerUncertain;
                                    break;
                                }
                                if self.core.admission() != super::inference::AdmissionState::Ready
                                {
                                    report.stopped = StopReason::WorkerUncertain;
                                    break;
                                }
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {
                        self.core.quarantine();
                        fatal_error = Some(DaemonError::HistoryUnavailable);
                        report.stopped = StopReason::WorkerUncertain;
                        break;
                    }
                }
            }

            if let Err(error) = self.settle_terminal_result_slots(&mut accepted_result_ids) {
                fatal_error = Some(error);
                report.stopped = StopReason::WorkerUncertain;
                break;
            }

            let accept_deadline = (Instant::now() + CPU_ACCEPT_POLL).min(context.deadline());
            let accept_cancel = ForwardCancellation(context);
            let accept_context = OperationContext::new(accept_deadline, &accept_cancel);
            let mut peer = match self.listener.accept(&accept_context) {
                Ok(peer) => {
                    handled_connections += 1;
                    peer
                }
                Err(LocalIpcError::PeerDenied) => {
                    handled_connections += 1;
                    report.failed_connections += 1;
                    continue;
                }
                Err(LocalIpcError::Deadline) => {
                    if let Some(reason) = stop_reason(context) {
                        report.stopped = reason;
                        break;
                    }
                    // A 25 ms idle poll is not a connection attempt.
                    continue;
                }
                Err(LocalIpcError::Cancelled) => {
                    report.stopped = stop_reason(context).unwrap_or(StopReason::Cancelled);
                    break;
                }
                Err(error) => {
                    fatal_error = Some(DaemonError::Transport(error));
                    break;
                }
            };

            let peer_deadline = (Instant::now() + self.peer_timeout).min(context.deadline());
            let peer_cancel = ForwardCancellation(context);
            let peer_context = OperationContext::new(peer_deadline, &peer_cancel);
            if self
                .serve_cpu_connection(&mut peer, &peer_context, &mut accepted_result_ids)
                .is_ok()
            {
                report.completed_exchanges += 1;
            } else {
                report.failed_connections += 1;
            }
            if stop_reason(context).is_some() {
                report.stopped = stop_reason(context).expect("checked stop reason");
                break;
            }
        }

        // On every session end, stop accepting new work and persist queued
        // cancellations before asking the worker to clean up its active child.
        let cancellation_uncertain = self.cancel_accepted_cpu_requests(&mut accepted_result_ids);
        let worker_uncertain = self.shutdown_cpu_worker(
            &mut worker,
            &mut active_ticket,
            &mut accepted_result_ids,
            recovery,
        );
        if worker_uncertain || cancellation_uncertain {
            report.stopped = StopReason::WorkerUncertain;
        }
        if fatal_error.is_none() {
            if let Err(error) = self.settle_terminal_result_slots(&mut accepted_result_ids) {
                fatal_error = Some(error);
            }
        }
        if let Some(error) = fatal_error {
            return Err(error);
        }
        Ok(report)
    }

    fn serve_connection(
        &mut self,
        connection: &mut super::local_ipc::LocalConnection,
        context: &OperationContext<'_>,
    ) -> Result<(), LocalIpcError> {
        let request = connection.read_incoming_request(context)?;
        if context.is_cancelled() {
            return Err(LocalIpcError::Cancelled);
        }
        if context.remaining().is_zero() {
            return Err(LocalIpcError::Deadline);
        }
        let response = match request {
            IncomingRequest::Control(request) => {
                self.core.handle_client_request(request, Instant::now())
            }
            IncomingRequest::Results(query) => self.handle_results_query(query),
        };
        connection.write_response(&response, context)
    }

    fn serve_cpu_connection(
        &mut self,
        connection: &mut super::local_ipc::LocalConnection,
        context: &OperationContext<'_>,
        accepted_result_ids: &mut BTreeSet<String>,
    ) -> Result<(), LocalIpcError> {
        let request = connection.read_incoming_request(context)?;
        if context.is_cancelled() {
            return Err(LocalIpcError::Cancelled);
        }
        if context.remaining().is_zero() {
            return Err(LocalIpcError::Deadline);
        }

        if let IncomingRequest::Control(ClientRequest::Submit { request_id, .. }) = &request {
            if let Err(error) = self.results.can_begin_reserved(request_id) {
                let code = match error {
                    ResultBufferError::InvalidRequestId => ResponseErrorCode::InvalidRequest,
                    ResultBufferError::DuplicateRequest => ResponseErrorCode::DuplicateRequest,
                    ResultBufferError::CapacityExceeded => ResponseErrorCode::QueueFull,
                    _ => ResponseErrorCode::Internal,
                };
                return connection.write_response(
                    &ResponseEnvelope::v1(ResponseMessage::Error {
                        request_id: Some(request_id.clone()),
                        code,
                    }),
                    context,
                );
            }
        }

        let response = match request {
            IncomingRequest::Control(request) => {
                self.core.handle_client_request(request, Instant::now())
            }
            IncomingRequest::Results(query) => self.handle_results_query(query),
        };
        if let ResponseMessage::Accepted {
            request_id,
            operation: AcceptedOperation::Submit,
            state: RequestStatus::Queued,
        } = &response.message
        {
            if self
                .results
                .begin_reserved(request_id, super::catalog::Lane::Decisions)
                .is_err()
            {
                // The single daemon thread has no intervening buffer mutation
                // between preflight and this commit. Treat any disagreement as
                // an internal invariant failure, quarantine, and cancel queued
                // work before it can be dispatched.
                self.core.quarantine();
                let _ = self.core.cancel_request(request_id);
                return connection.write_response(
                    &ResponseEnvelope::v1(ResponseMessage::Error {
                        request_id: Some(request_id.clone()),
                        code: ResponseErrorCode::Internal,
                    }),
                    context,
                );
            }
            accepted_result_ids.insert(request_id.clone());
        }
        connection.write_response(&response, context)
    }

    fn settle_terminal_result_slots(
        &mut self,
        accepted_result_ids: &mut BTreeSet<String>,
    ) -> Result<(), DaemonError> {
        let mut terminal = Vec::new();
        for request_id in accepted_result_ids.iter() {
            let state = self.core.authoritative_request_status(request_id);
            let result = match state {
                Some(RequestStatus::Queued | RequestStatus::Running) => continue,
                Some(RequestStatus::Succeeded) => ResultTerminal::Succeeded,
                Some(RequestStatus::Failed) => ResultTerminal::Failed(ResultFailureCode::TimedOut),
                Some(RequestStatus::Cancelled) => ResultTerminal::Cancelled,
                Some(RequestStatus::Interrupted) | Some(RequestStatus::NotFound) | None => {
                    self.core.quarantine();
                    ResultTerminal::Interrupted
                }
            };
            terminal.push((request_id.clone(), result));
        }
        for (request_id, result) in terminal {
            self.results
                .terminal(&request_id, result)
                .map_err(|_| DaemonError::ResultBufferInvariant)?;
            accepted_result_ids.remove(&request_id);
        }
        Ok(())
    }

    fn complete_cpu_without_child(
        &mut self,
        slot: &CpuDecisionSlot,
        completion: Completion,
        result: ResultTerminal,
        accepted_result_ids: &mut BTreeSet<String>,
    ) -> Result<(), DaemonError> {
        self.complete_cpu_after_cleanup(slot, completion, result, accepted_result_ids)
    }

    fn complete_cpu_after_cleanup(
        &mut self,
        slot: &CpuDecisionSlot,
        completion: Completion,
        result: ResultTerminal,
        accepted_result_ids: &mut BTreeSet<String>,
    ) -> Result<(), DaemonError> {
        match self
            .core
            .complete(slot.job_id, completion, Vec::new(), Instant::now())
        {
            Ok(_) => {
                self.results
                    .terminal(&slot.request_id, result)
                    .map_err(|_| DaemonError::ResultBufferInvariant)?;
                accepted_result_ids.remove(&slot.request_id);
                Ok(())
            }
            Err(_) => {
                self.core.quarantine();
                // A failed durable terminal update is never converted into
                // success. Attempt to persist interruption while retaining the
                // reservation; if this succeeds, cleanup has already been
                // independently proved above and may be acknowledged.
                if self
                    .core
                    .complete(
                        slot.job_id,
                        Completion::Uncertain,
                        Vec::new(),
                        Instant::now(),
                    )
                    .is_ok()
                {
                    let _ = self.core.acknowledge_reconciled_cleanup(slot.job_id);
                }
                self.results
                    .terminal(&slot.request_id, ResultTerminal::Interrupted)
                    .map_err(|_| DaemonError::ResultBufferInvariant)?;
                accepted_result_ids.remove(&slot.request_id);
                Err(DaemonError::HistoryUnavailable)
            }
        }
    }

    fn mark_cpu_slot_uncertain(
        &mut self,
        slot: &CpuDecisionSlot,
        accepted_result_ids: &mut BTreeSet<String>,
    ) {
        self.core.quarantine();
        let _ = self.core.complete(
            slot.job_id,
            Completion::Uncertain,
            Vec::new(),
            Instant::now(),
        );
        let _ = self
            .results
            .terminal(&slot.request_id, ResultTerminal::Interrupted);
        accepted_result_ids.remove(&slot.request_id);
    }

    fn consume_cpu_completion(
        &mut self,
        completion: DecisionWorkerCompletion,
        active_ticket: &mut Option<CpuDecisionSlot>,
        accepted_result_ids: &mut BTreeSet<String>,
    ) -> Result<bool, DaemonError> {
        let Some(slot) = active_ticket.take() else {
            self.core.quarantine();
            return Ok(false);
        };
        let (job_id, request_id, outcome) = completion.into_parts();
        if job_id != slot.job_id || request_id != slot.request_id {
            self.mark_cpu_slot_uncertain(&slot, accepted_result_ids);
            return Ok(false);
        }

        let cleanup = outcome.cleanup();
        if cleanup == ProcessFailureDisposition::OwnershipUncertain {
            self.mark_cpu_slot_uncertain(&slot, accepted_result_ids);
            return Ok(false);
        }
        match outcome {
            DecisionWorkerOutcome::Succeeded(native) => {
                if cleanup != ProcessFailureDisposition::ChildGroupReaped {
                    self.mark_cpu_slot_uncertain(&slot, accepted_result_ids);
                    return Ok(false);
                }
                let super::requests::InferenceResult::Decision { answers } = native.result else {
                    return self
                        .complete_cpu_after_cleanup(
                            &slot,
                            Completion::Failed,
                            ResultTerminal::Failed(ResultFailureCode::InvalidOutput),
                            accepted_result_ids,
                        )
                        .map(|()| true);
                };
                if self
                    .results
                    .append(
                        &slot.request_id,
                        ResultEvent::DecisionResult {
                            answers,
                            native_latency: native.native_latency,
                            native_usage: native.native_usage,
                        },
                    )
                    .is_err()
                {
                    return self
                        .complete_cpu_after_cleanup(
                            &slot,
                            Completion::Failed,
                            ResultTerminal::Failed(ResultFailureCode::Internal),
                            accepted_result_ids,
                        )
                        .map(|()| true);
                }
                self.complete_cpu_after_cleanup(
                    &slot,
                    Completion::Succeeded,
                    ResultTerminal::Succeeded,
                    accepted_result_ids,
                )?;
                Ok(true)
            }
            DecisionWorkerOutcome::Failed(failure) => {
                if !matches!(
                    cleanup,
                    ProcessFailureDisposition::NoChildSpawned
                        | ProcessFailureDisposition::ChildGroupReaped
                ) {
                    self.mark_cpu_slot_uncertain(&slot, accepted_result_ids);
                    return Ok(false);
                }
                let (completion, result) = worker_failure_result(failure.kind());
                self.complete_cpu_after_cleanup(&slot, completion, result, accepted_result_ids)?;
                Ok(true)
            }
        }
    }

    fn cancel_accepted_cpu_requests(&mut self, accepted_result_ids: &mut BTreeSet<String>) -> bool {
        let mut uncertain = false;
        let mut interrupt = Vec::new();
        for request_id in accepted_result_ids.iter() {
            match self.core.authoritative_request_status(request_id) {
                Some(RequestStatus::Queued | RequestStatus::Running) => {
                    if self.core.cancel_request(request_id).is_err() {
                        self.core.quarantine();
                        uncertain = true;
                        interrupt.push(request_id.clone());
                    }
                }
                Some(
                    RequestStatus::Succeeded
                    | RequestStatus::Failed
                    | RequestStatus::Cancelled
                    | RequestStatus::Interrupted,
                ) => {}
                Some(RequestStatus::NotFound) | None => {
                    self.core.quarantine();
                    interrupt.push(request_id.clone());
                    uncertain = true;
                }
            }
        }
        for request_id in interrupt {
            if self
                .results
                .terminal(&request_id, ResultTerminal::Interrupted)
                .is_err()
            {
                uncertain = true;
            }
            accepted_result_ids.remove(&request_id);
        }
        if self
            .settle_terminal_result_slots(accepted_result_ids)
            .is_err()
        {
            uncertain = true;
        }
        uncertain
    }

    fn shutdown_cpu_worker(
        &mut self,
        worker: &mut CpuDecisionWorker,
        active_ticket: &mut Option<CpuDecisionSlot>,
        accepted_result_ids: &mut BTreeSet<String>,
        recovery: RecoveryPolicy,
    ) -> bool {
        worker.shutdown();
        let grace = recovery.timeout().saturating_add(CPU_JOIN_POLL);
        let deadline = Instant::now()
            .checked_add(grace)
            .unwrap_or_else(Instant::now);
        let mut completion = None;
        let mut channel_closed = false;
        let mut joined = false;

        loop {
            if completion.is_none() && worker.is_in_flight() {
                let poll_deadline = (Instant::now() + CPU_JOIN_POLL).min(deadline);
                match worker.receive_until(poll_deadline) {
                    Ok(Some(received)) => completion = Some(received),
                    Ok(None) => {}
                    Err(DecisionWorkerReceiveError::Closed) => channel_closed = true,
                }
            }

            match worker.try_join_finished() {
                Ok(true) => {
                    joined = true;
                    break;
                }
                Ok(false) => {}
                Err(_) => {
                    if let Some(slot) = active_ticket.take() {
                        self.mark_cpu_slot_uncertain(&slot, accepted_result_ids);
                    } else {
                        self.core.quarantine();
                    }
                    return true;
                }
            }
            if channel_closed || Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(CPU_JOIN_POLL);
        }

        if !joined {
            joined = matches!(worker.try_join_finished(), Ok(true));
        }
        if !joined {
            if let Some(slot) = active_ticket.take() {
                self.mark_cpu_slot_uncertain(&slot, accepted_result_ids);
            } else {
                self.core.quarantine();
            }
            return true;
        }

        if completion.is_none() && worker.is_in_flight() {
            match worker.try_receive() {
                Ok(Some(received)) => completion = Some(received),
                Ok(None) | Err(DecisionWorkerReceiveError::Closed) => {}
            }
        }
        if let Some(completion) = completion {
            return match self.consume_cpu_completion(completion, active_ticket, accepted_result_ids)
            {
                Ok(cleanup_verified) => !cleanup_verified,
                Err(_) => true,
            };
        }
        if let Some(slot) = active_ticket.take() {
            self.mark_cpu_slot_uncertain(&slot, accepted_result_ids);
            true
        } else {
            false
        }
    }

    fn handle_results_query(&self, query: ResultsQuery) -> ResponseEnvelope {
        let Some(state) = self.core.authoritative_request_status(&query.request_id) else {
            return results_error(query.request_id, ResponseErrorCode::Internal);
        };
        let page = match self.results.page(
            &query.request_id,
            query.after_cursor,
            query.byte_budget as usize,
        ) {
            Ok(page) => page,
            Err(error) => {
                let code = match error {
                    ResultBufferError::InvalidRequestId
                    | ResultBufferError::InvalidCursor
                    | ResultBufferError::InvalidPageBudget
                    | ResultBufferError::PageBudgetTooSmall => ResponseErrorCode::InvalidRequest,
                    _ => ResponseErrorCode::Internal,
                };
                return results_error(query.request_id, code);
            }
        };
        ResponseEnvelope::v2_results(ResponseMessage::Results {
            request_id: query.request_id,
            after_cursor: query.after_cursor,
            byte_budget: query.byte_budget,
            state,
            page,
        })
    }
}

fn serve_managed_connection(
    core: &mut CoordinatorCore,
    results: &mut ResultBuffer,
    connection: &mut super::local_ipc::LocalConnection,
    context: &OperationContext<'_>,
    accepted_result_ids: &mut BTreeSet<String>,
) -> Result<(), LocalIpcError> {
    let request = connection.read_incoming_request(context)?;
    if context.is_cancelled() {
        return Err(LocalIpcError::Cancelled);
    }
    if context.remaining().is_zero() {
        return Err(LocalIpcError::Deadline);
    }

    let reservation = match &request {
        IncomingRequest::Control(ClientRequest::Submit {
            request_id,
            request,
            ..
        }) => Some((request_id.clone(), request.lane())),
        _ => None,
    };
    if let Some((request_id, _)) = &reservation {
        if let Err(error) = results.can_begin_reserved(request_id) {
            let code = match error {
                ResultBufferError::InvalidRequestId => ResponseErrorCode::InvalidRequest,
                ResultBufferError::DuplicateRequest => ResponseErrorCode::DuplicateRequest,
                ResultBufferError::CapacityExceeded => ResponseErrorCode::QueueFull,
                _ => ResponseErrorCode::Internal,
            };
            return connection.write_response(
                &ResponseEnvelope::v1(ResponseMessage::Error {
                    request_id: Some(request_id.clone()),
                    code,
                }),
                context,
            );
        }
    }

    let response = match request {
        IncomingRequest::Control(request) => core.handle_client_request(request, Instant::now()),
        IncomingRequest::Results(query) => managed_results_query(core, results, query),
    };
    if let (
        Some((expected_request_id, lane)),
        ResponseMessage::Accepted {
            request_id,
            operation: AcceptedOperation::Submit,
            state: RequestStatus::Queued,
        },
    ) = (reservation, &response.message)
    {
        if &expected_request_id != request_id || results.begin_reserved(request_id, lane).is_err() {
            core.quarantine();
            let _ = core.cancel_request(request_id);
            return connection.write_response(
                &ResponseEnvelope::v1(ResponseMessage::Error {
                    request_id: Some(request_id.clone()),
                    code: ResponseErrorCode::Internal,
                }),
                context,
            );
        }
        accepted_result_ids.insert(request_id.clone());
    }
    connection.write_response(&response, context)
}

fn managed_results_query(
    core: &CoordinatorCore,
    results: &ResultBuffer,
    query: ResultsQuery,
) -> ResponseEnvelope {
    let Some(state) = core.authoritative_request_status(&query.request_id) else {
        return results_error(query.request_id, ResponseErrorCode::Internal);
    };
    let page = match results.page(
        &query.request_id,
        query.after_cursor,
        query.byte_budget as usize,
    ) {
        Ok(page) => page,
        Err(error) => {
            let code = match error {
                ResultBufferError::InvalidRequestId
                | ResultBufferError::InvalidCursor
                | ResultBufferError::InvalidPageBudget
                | ResultBufferError::PageBudgetTooSmall => ResponseErrorCode::InvalidRequest,
                _ => ResponseErrorCode::Internal,
            };
            return results_error(query.request_id, code);
        }
    };
    ResponseEnvelope::v2_results(ResponseMessage::Results {
        request_id: query.request_id,
        after_cursor: query.after_cursor,
        byte_budget: query.byte_budget,
        state,
        page,
    })
}

fn settle_managed_terminal_result_slots(
    core: &mut CoordinatorCore,
    results: &mut ResultBuffer,
    accepted_result_ids: &mut BTreeSet<String>,
    active_slot: Option<&super::managed_session::ManagedJobKey>,
) -> Result<(), DaemonError> {
    let mut terminal = Vec::new();
    for request_id in accepted_result_ids.iter() {
        if active_slot.is_some_and(|slot| slot.request_id == *request_id) {
            continue;
        }
        let state = core.authoritative_request_status(request_id);
        let result = match state {
            Some(RequestStatus::Queued | RequestStatus::Running) => continue,
            Some(RequestStatus::Succeeded) => ResultTerminal::Succeeded,
            Some(RequestStatus::Failed) => ResultTerminal::Failed(ResultFailureCode::TimedOut),
            Some(RequestStatus::Cancelled) => ResultTerminal::Cancelled,
            Some(RequestStatus::Interrupted | RequestStatus::NotFound) | None => {
                core.quarantine();
                continue;
            }
        };
        terminal.push((request_id.clone(), result));
    }
    for (request_id, result) in terminal {
        results
            .terminal(&request_id, result)
            .map_err(|_| DaemonError::ResultBufferInvariant)?;
        accepted_result_ids.remove(&request_id);
    }
    Ok(())
}

fn cancel_managed_queued(
    core: &mut CoordinatorCore,
    results: &mut ResultBuffer,
    accepted_result_ids: &mut BTreeSet<String>,
    active_slot: Option<&super::managed_session::ManagedJobKey>,
) -> Result<(), DaemonError> {
    let queued = accepted_result_ids
        .iter()
        .filter(|request_id| !active_slot.is_some_and(|slot| slot.request_id == **request_id))
        .filter(|request_id| {
            core.authoritative_request_status(request_id) == Some(RequestStatus::Queued)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut failure = None;
    for request_id in queued {
        if core.cancel_request(&request_id).is_err() {
            core.quarantine();
            failure = Some(DaemonError::HistoryUnavailable);
            continue;
        }
        if results
            .terminal(&request_id, ResultTerminal::Cancelled)
            .is_err()
        {
            failure = Some(DaemonError::ResultBufferInvariant);
        }
        accepted_result_ids.remove(&request_id);
    }
    failure.map_or(Ok(()), Err)
}

fn consume_managed_completion(
    core: &mut CoordinatorCore,
    results: &mut ResultBuffer,
    completion: super::managed_session::ManagedJobCompletion,
    active_slot: &mut Option<super::managed_session::ManagedJobKey>,
    accepted_result_ids: &mut BTreeSet<String>,
    cleanup_ack_pending: &mut bool,
) -> Result<(), DaemonError> {
    let Some(active) = active_slot.as_ref() else {
        core.quarantine();
        return Err(DaemonError::ManagedSessionUnavailable);
    };
    if active != &completion.key {
        core.quarantine();
        return Err(DaemonError::ManagedSessionUnavailable);
    }
    let slot = active.clone();

    if completion.completion == Completion::Uncertain {
        if core
            .complete(
                slot.job_id,
                Completion::Uncertain,
                Vec::new(),
                Instant::now(),
            )
            .is_err()
        {
            core.quarantine();
            return Err(DaemonError::HistoryUnavailable);
        }
        if completion.cleanup_proven {
            core.acknowledge_reconciled_cleanup(slot.job_id)
                .map_err(|_| DaemonError::HistoryUnavailable)?;
            results
                .terminal(&slot.request_id, ResultTerminal::Interrupted)
                .map_err(|_| DaemonError::ResultBufferInvariant)?;
            accepted_result_ids.remove(&slot.request_id);
            *active_slot = None;
        }
        return Ok(());
    }

    if core
        .complete(
            slot.job_id,
            completion.completion,
            completion.artifacts.clone(),
            Instant::now(),
        )
        .is_err()
    {
        core.quarantine();
        if core
            .complete(
                slot.job_id,
                Completion::Uncertain,
                Vec::new(),
                Instant::now(),
            )
            .is_err()
        {
            return Err(DaemonError::HistoryUnavailable);
        }
        if completion.cleanup_proven {
            core.acknowledge_reconciled_cleanup(slot.job_id)
                .map_err(|_| DaemonError::HistoryUnavailable)?;
            *active_slot = None;
            results
                .terminal(&slot.request_id, ResultTerminal::Interrupted)
                .map_err(|_| DaemonError::ResultBufferInvariant)?;
            accepted_result_ids.remove(&slot.request_id);
        } else {
            *cleanup_ack_pending = true;
        }
        return Ok(());
    }

    *active_slot = None;
    results
        .terminal(&slot.request_id, completion.result_terminal)
        .map_err(|_| DaemonError::ResultBufferInvariant)?;
    accepted_result_ids.remove(&slot.request_id);
    if completion.stop_session {
        core.quarantine();
    }
    Ok(())
}

struct CpuDecisionSlot {
    job_id: u64,
    request_id: String,
}

fn dispatch_error_result(error: DispatchBeginError) -> (Completion, ResultTerminal) {
    match error {
        DispatchBeginError::Cancelled => (Completion::Cancelled, ResultTerminal::Cancelled),
        DispatchBeginError::DeadlineExceeded => (
            Completion::Failed,
            ResultTerminal::Failed(ResultFailureCode::TimedOut),
        ),
        _ => (
            Completion::Failed,
            ResultTerminal::Failed(ResultFailureCode::Internal),
        ),
    }
}

fn worker_failure_result(kind: DecisionWorkerFailureKind) -> (Completion, ResultTerminal) {
    match kind {
        DecisionWorkerFailureKind::Cancelled => (Completion::Cancelled, ResultTerminal::Cancelled),
        DecisionWorkerFailureKind::DeadlineExceeded => (
            Completion::Failed,
            ResultTerminal::Failed(ResultFailureCode::TimedOut),
        ),
        DecisionWorkerFailureKind::Unavailable => (
            Completion::Failed,
            ResultTerminal::Failed(ResultFailureCode::Unavailable),
        ),
        DecisionWorkerFailureKind::ProcessFailed => (
            Completion::Failed,
            ResultTerminal::Failed(ResultFailureCode::ResourceFailure),
        ),
        DecisionWorkerFailureKind::InvalidOutput(_) => (
            Completion::Failed,
            ResultTerminal::Failed(ResultFailureCode::InvalidOutput),
        ),
        DecisionWorkerFailureKind::UnsupportedRoute | DecisionWorkerFailureKind::Internal => (
            Completion::Failed,
            ResultTerminal::Failed(ResultFailureCode::Internal),
        ),
    }
}

fn results_error(request_id: String, code: ResponseErrorCode) -> ResponseEnvelope {
    ResponseEnvelope::v2_results(ResponseMessage::Error {
        request_id: Some(request_id),
        code,
    })
}

/// Check before binding or loading history. Callers must pass the same legacy
/// store their standalone clients use; trusted directory ancestors remain
/// stable during the session. This is not a lock respected by legacy writers.
fn verify_history_isolation(data: &Path, standalone: &Path) -> Result<(), DaemonError> {
    use std::os::unix::fs::MetadataExt;
    if !standalone.is_absolute() {
        return Err(DaemonError::InvalidConfig);
    }
    let standalone = crate::fs_util::resolve_allow_missing(standalone)
        .map_err(|_| DaemonError::HistoryConflict)?;
    if data == standalone {
        return Err(DaemonError::HistoryConflict);
    }
    let data_metadata = data.metadata().map_err(|_| DaemonError::HistoryConflict)?;
    match standalone.metadata() {
        Ok(metadata) => {
            if !metadata.is_dir()
                || (metadata.dev(), metadata.ino()) == (data_metadata.dev(), data_metadata.ino())
            {
                return Err(DaemonError::HistoryConflict);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(DaemonError::HistoryConflict),
    }
    match std::fs::symlink_metadata(data.join("jobs.json")) {
        Ok(metadata) if !metadata.is_file() || metadata.nlink() != 1 => {
            return Err(DaemonError::HistoryConflict);
        }
        Ok(metadata) => match standalone.join("jobs.json").metadata() {
            Ok(legacy) if (metadata.dev(), metadata.ino()) == (legacy.dev(), legacy.ino()) => {
                return Err(DaemonError::HistoryConflict);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(DaemonError::HistoryConflict),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(DaemonError::HistoryConflict),
    }
    Ok(())
}

// Forward only cancellation, not the peer deadline, to the bounded context.
struct ForwardCancellation<'a>(&'a OperationContext<'a>);
impl super::runtime::Cancellation for ForwardCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

fn stop_reason(context: &OperationContext<'_>) -> Option<StopReason> {
    if context.is_cancelled() {
        Some(StopReason::Cancelled)
    } else if context.remaining().is_zero() {
        Some(StopReason::Deadline)
    } else {
        None
    }
}

fn managed_ready_may_open_admission(stopping: bool, session_cancelled: bool) -> bool {
    !stopping && !session_cancelled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::ipc::{
        RequestEnvelope, RequestStatus, ResponseEnvelope, ResponseErrorCode, ResponseMessage,
        WireChatRole, WireChatTurn, WireInferenceRequest, WireRequestMessage,
    };
    use crate::control::local_ipc::LocalConnection;
    use crate::control::ownership::CoordinatorLock;
    use crate::control::result_buffer::{
        Availability, PageEvent, ResultEvent, ResultTerminal, MIN_PAGE_BYTES,
    };
    use crate::control::runtime::{Cancellation, NeverCancel};
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn managed_late_ready_cannot_reopen_admission_after_stopping_begins() {
        assert!(managed_ready_may_open_admission(false, false));
        assert!(!managed_ready_may_open_admission(true, false));
        assert!(!managed_ready_may_open_admission(false, true));
        assert!(!managed_ready_may_open_admission(true, true));
    }

    struct Fixture {
        socket: TempDir,
        locks: TempDir,
        data: TempDir,
        standalone: TempDir,
    }
    impl Fixture {
        fn new() -> Self {
            let fixture = Self {
                socket: TempDir::new().unwrap(),
                locks: TempDir::new().unwrap(),
                data: TempDir::new().unwrap(),
                standalone: TempDir::new().unwrap(),
            };
            for path in [
                fixture.socket.path(),
                fixture.locks.path(),
                fixture.data.path(),
            ] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
            }
            fixture
        }
        fn config(&self, limit: usize) -> DaemonConfig {
            DaemonConfig {
                socket_directory: self.socket.path().into(),
                lock_directory: self.locks.path().into(),
                data_directory: self.data.path().into(),
                standalone_data_directory: self.standalone.path().into(),
                gpus: vec![0],
                queue_capacity: 8,
                peer_timeout: Duration::from_millis(100),
                max_connections: limit,
            }
        }
        fn bind(&self, config: DaemonConfig) -> Result<ControlDaemon, DaemonError> {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"));
            ControlDaemon::bind(config, Catalog::load(root).unwrap(), BTreeMap::new(), root)
        }
        fn released(&self) {
            assert!(!self.socket.path().join("coordinator.sock").exists());
            assert!(CoordinatorLock::acquire(self.locks.path(), [0]).is_ok());
        }
    }

    struct CpuFixture {
        root: TempDir,
        socket: TempDir,
        locks: TempDir,
        data: TempDir,
        standalone: TempDir,
    }

    #[test]
    fn managed_uncertain_completion_retains_reservation_until_cleanup_is_proven() {
        use crate::control::catalog::{Lane, ProfileKey};
        use crate::control::managed_session::{ManagedJobCompletion, ManagedJobKey};
        use crate::control::requests::{ChatRole, ChatTurn, InferenceRequest};

        for cleanup_proven in [false, true] {
            let fixture = CpuFixture::new("# not executed\n");
            let catalog = Catalog::load(fixture.root.path()).unwrap();
            let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
            let mut core = CoordinatorCore::load(
                catalog,
                BTreeMap::from([(runtime, ResourceClaim::ExclusiveGpu(0))]),
                fixture.data.path(),
                fixture.root.path(),
                8,
            )
            .unwrap();
            // This fixture exercises completion accounting, not native readiness.
            core.confirm_reconciled().unwrap();
            let request_id = "uncertain-language";
            let job_id = core
                .submit(
                    request_id.into(),
                    InferenceRequest::Chat {
                        profile: ProfileKey::new(Lane::Language, "fixture").unwrap(),
                        turns: vec![ChatTurn {
                            role: ChatRole::User,
                            content: "fixture".into(),
                        }],
                        max_tokens: 64,
                    },
                    Duration::from_secs(30),
                    Instant::now(),
                )
                .unwrap();
            let ticket = core.reserve_next(Instant::now()).unwrap().unwrap();
            core.begin_dispatch(&ticket, Instant::now()).unwrap();
            let key = ManagedJobKey {
                job_id,
                request_id: request_id.into(),
            };
            let mut active = Some(key.clone());
            let mut accepted = BTreeSet::from([request_id.into()]);
            let mut results = ResultBuffer::new();
            results.begin(request_id, Lane::Language).unwrap();
            let mut cleanup_pending = false;
            consume_managed_completion(
                &mut core,
                &mut results,
                ManagedJobCompletion {
                    key,
                    completion: Completion::Uncertain,
                    result_terminal: ResultTerminal::Interrupted,
                    cleanup_proven,
                    stop_session: true,
                    artifacts: Vec::new(),
                },
                &mut active,
                &mut accepted,
                &mut cleanup_pending,
            )
            .unwrap();
            assert_eq!(core.active_count(), usize::from(!cleanup_proven));
            assert_eq!(active.is_some(), !cleanup_proven);
            assert_eq!(accepted.contains(request_id), !cleanup_proven);
            assert_eq!(
                core.authoritative_request_status(request_id),
                Some(RequestStatus::Interrupted)
            );
            assert_ne!(
                core.admission(),
                crate::control::inference::AdmissionState::Ready
            );
        }
    }

    impl CpuFixture {
        fn new(runner: &str) -> Self {
            let fixture = Self {
                root: TempDir::new().unwrap(),
                socket: TempDir::new().unwrap(),
                locks: TempDir::new().unwrap(),
                data: TempDir::new().unwrap(),
                standalone: TempDir::new().unwrap(),
            };
            for path in [
                fixture.socket.path(),
                fixture.locks.path(),
                fixture.data.path(),
            ] {
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
            }
            let media = fixture.root.path().join("toolkits/l2m2");
            let decisions = fixture.root.path().join("toolkits/l3ds");
            std::fs::create_dir_all(&media).unwrap();
            std::fs::create_dir_all(decisions.join("decision-models")).unwrap();
            std::fs::create_dir_all(decisions.join(".venv/bin")).unwrap();
            std::fs::write(
                fixture.root.path().join("llama-swap.yaml"),
                "models:\n  fixture:\n    cmd: fixture\n",
            )
            .unwrap();
            std::fs::write(
                media.join("media-runtimes.json"),
                r#"{"schema_version":1,"profiles":[]}"#,
            )
            .unwrap();
            std::fs::write(
                decisions.join("decision-runtimes.json"),
                r#"{"version":"1.0","profiles":[
                    {"id":"fixture-cpu","name":"Fixture CPU","engine":"nagi","type":"local","device":"cpu","model":"fixture-model","entrypoint":"decision-models/runner.py"},
                    {"id":"fixture-gpu","name":"Fixture GPU","engine":"nagi","type":"local","device":"cuda","model":"fixture-model","entrypoint":"decision-models/runner.py"}
                ]}"#,
            )
            .unwrap();
            let _guard = super::super::process::executable_fixture_guard();
            let interpreter = decisions.join(".venv/bin/python");
            std::fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
            std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o700)).unwrap();
            let runner_path = decisions.join("decision-models/runner.py");
            std::fs::write(&runner_path, runner).unwrap();
            std::fs::set_permissions(&runner_path, std::fs::Permissions::from_mode(0o700)).unwrap();
            drop(_guard);
            fixture
        }

        fn config(&self, max_connections: usize) -> DaemonConfig {
            DaemonConfig {
                socket_directory: self.socket.path().into(),
                lock_directory: self.locks.path().into(),
                data_directory: self.data.path().into(),
                standalone_data_directory: self.standalone.path().into(),
                gpus: Vec::new(),
                queue_capacity: 8,
                peer_timeout: Duration::from_millis(100),
                max_connections,
            }
        }

        fn bind(&self, max_connections: usize) -> ControlDaemon {
            let catalog = Catalog::load(self.root.path()).unwrap();
            let runtime =
                RuntimeIdentity::new(crate::control::catalog::Lane::Decisions, "nagi").unwrap();
            ControlDaemon::bind(
                self.config(max_connections),
                catalog,
                BTreeMap::from([(runtime, ResourceClaim::None)]),
                self.root.path(),
            )
            .unwrap()
        }

        fn released(&self) {
            assert!(!self.socket.path().join("coordinator.sock").exists());
            assert!(CoordinatorLock::acquire(self.locks.path(), []).is_ok());
        }
    }

    fn cpu_submit(request_id: &str, profile_id: &str, timeout_ms: u64) -> WireRequestMessage {
        WireRequestMessage::Submit {
            request_id: request_id.to_owned(),
            timeout_ms,
            request: WireInferenceRequest::Decisions {
                profile_id: profile_id.to_owned(),
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
            },
        }
    }

    fn successful_cpu_runner(marker: &Path) -> String {
        format!(
            "#!/bin/sh\nprintf x >> {}\ncat >/dev/null\nprintf '%s\\n' '{{\"engine\":\"nagi-native\",\"answers\":{{\"route\":{{\"choice\":\"secret-answer\"}}}},\"latency_ms\":12.5}}'\n",
            shell_quote(marker)
        )
    }

    fn sleeping_cpu_runner(marker: &Path) -> String {
        format!(
            "#!/bin/sh\ntouch {}\nexec /bin/sleep 30\n",
            shell_quote(marker)
        )
    }

    fn shell_quote(path: &Path) -> String {
        format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
    }

    fn shared_cancellation() -> Arc<Flag> {
        Arc::new(Flag(AtomicBool::new(false)))
    }

    fn spawn_cpu_daemon(
        daemon: ControlDaemon,
        cancellation: Arc<Flag>,
    ) -> std::thread::JoinHandle<Result<RunReport, DaemonError>> {
        std::thread::spawn(move || {
            let context = OperationContext::new(
                Instant::now() + Duration::from_secs(15),
                cancellation.as_ref(),
            );
            daemon.run_cpu_decisions(
                &context,
                RecoveryPolicy::new(Duration::from_secs(2)).unwrap(),
            )
        })
    }

    fn wait_for_request_state(directory: &Path, request_id: &str, terminal: bool) -> RequestStatus {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let response = exchange(
                directory,
                WireRequestMessage::Status {
                    request_id: request_id.to_owned(),
                },
            );
            if let ResponseMessage::Status { state, .. } = response {
                if !terminal
                    || matches!(
                        state,
                        RequestStatus::Succeeded
                            | RequestStatus::Failed
                            | RequestStatus::Cancelled
                            | RequestStatus::Interrupted
                    )
                {
                    return state;
                }
            }
            assert!(
                Instant::now() < deadline,
                "request did not reach expected state"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn wait_for_path(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !path.exists() {
            assert!(Instant::now() < deadline, "fixture runner did not start");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn cancel_outer_and_join(
        cancellation: &Flag,
        daemon: std::thread::JoinHandle<Result<RunReport, DaemonError>>,
    ) -> RunReport {
        cancellation.0.store(true, Ordering::Release);
        daemon.join().unwrap().unwrap()
    }

    fn result_page(directory: &Path, request_id: &str) -> super::super::result_buffer::ResultPage {
        match exchange_envelope(
            directory,
            RequestEnvelope::v2_results(request_id, 0, MIN_PAGE_BYTES as u32),
        )
        .message
        {
            ResponseMessage::Results { page, .. } => page,
            other => panic!("unexpected results response: {other:?}"),
        }
    }

    #[test]
    fn cpu_decision_submit_lost_ack_result_pages_and_restart_never_replays() {
        let marker_dir = TempDir::new().unwrap();
        let marker = marker_dir.path().join("runs");
        let fixture = CpuFixture::new(&successful_cpu_runner(&marker));
        let daemon = fixture.bind(128);
        let directory = fixture.socket.path().to_owned();
        let cancellation = shared_cancellation();
        let worker = spawn_cpu_daemon(daemon, Arc::clone(&cancellation));

        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let mut lost_ack = LocalConnection::connect(&directory, &context).unwrap();
        lost_ack
            .write_request(
                &RequestEnvelope::v1(cpu_submit("cpu-once", "fixture-cpu", 5_000)),
                &context,
            )
            .unwrap();
        drop(lost_ack);

        assert_eq!(
            wait_for_request_state(&directory, "cpu-once", true),
            RequestStatus::Succeeded
        );
        let page = result_page(&directory, "cpu-once");
        assert_eq!(page.terminal, Some(ResultTerminal::Succeeded));
        assert_eq!(page.events.len(), 2);
        let PageEvent::DecisionJsonPart { json, .. } = &page.events[0].event else {
            panic!("expected native decision result page");
        };
        let native: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(
            native,
            serde_json::json!({"route":{"choice":"secret-answer"}})
        );
        let PageEvent::DecisionMetadataJsonPart { json, .. } = &page.events[1].event else {
            panic!("expected native decision metadata page");
        };
        let metadata: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(metadata["native_latency_ns"], 12_500_000);
        assert_eq!(std::fs::read(&marker).unwrap(), b"x");

        let first_report = cancel_outer_and_join(cancellation.as_ref(), worker);
        assert_eq!(first_report.stopped, StopReason::Cancelled);
        fixture.released();

        let restarted = fixture.bind(2);
        let restarted_worker = spawn_cpu_daemon(restarted, shared_cancellation());
        let status = exchange(
            &directory,
            WireRequestMessage::Status {
                request_id: "cpu-once".into(),
            },
        );
        assert_eq!(
            status,
            ResponseMessage::Status {
                request_id: "cpu-once".into(),
                state: RequestStatus::Succeeded,
            }
        );
        let restarted_page = result_page(&directory, "cpu-once");
        assert_eq!(restarted_page.availability, Availability::Unknown);
        assert_eq!(restarted_page.terminal, None);
        let restarted_report = restarted_worker.join().unwrap().unwrap();
        assert_eq!(restarted_report.stopped, StopReason::ConnectionLimit);
        assert_eq!(std::fs::read(&marker).unwrap(), b"x");
        fixture.released();
    }

    #[test]
    fn cpu_decision_running_cancel_waits_for_cleanup_and_terminalizes_page() {
        let marker_dir = TempDir::new().unwrap();
        let started = marker_dir.path().join("started");
        let fixture = CpuFixture::new(&sleeping_cpu_runner(&started));
        let daemon = fixture.bind(64);
        let directory = fixture.socket.path().to_owned();
        let cancellation = shared_cancellation();
        let worker = spawn_cpu_daemon(daemon, Arc::clone(&cancellation));

        assert!(matches!(
            exchange(&directory, cpu_submit("cpu-cancel", "fixture-cpu", 5_000)),
            ResponseMessage::Accepted {
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued,
                ..
            }
        ));
        wait_for_path(&started);
        assert_eq!(
            exchange(
                &directory,
                WireRequestMessage::Status {
                    request_id: "cpu-cancel".into(),
                }
            ),
            ResponseMessage::Status {
                request_id: "cpu-cancel".into(),
                state: RequestStatus::Running,
            }
        );
        assert!(matches!(
            exchange(
                &directory,
                WireRequestMessage::Cancel {
                    request_id: "cpu-cancel".into(),
                }
            ),
            ResponseMessage::Accepted {
                operation: AcceptedOperation::Cancel,
                state: RequestStatus::Running,
                ..
            }
        ));
        assert_eq!(
            wait_for_request_state(&directory, "cpu-cancel", true),
            RequestStatus::Cancelled
        );
        let page = result_page(&directory, "cpu-cancel");
        assert_eq!(page.terminal, Some(ResultTerminal::Cancelled));
        assert!(page.events.is_empty());

        let report = cancel_outer_and_join(cancellation.as_ref(), worker);
        assert_eq!(report.stopped, StopReason::Cancelled);
        fixture.released();
    }

    #[test]
    fn cpu_decision_timeout_cleans_child_before_failed_terminal_is_reported() {
        let marker_dir = TempDir::new().unwrap();
        let started = marker_dir.path().join("started");
        let fixture = CpuFixture::new(&sleeping_cpu_runner(&started));
        let daemon = fixture.bind(64);
        let directory = fixture.socket.path().to_owned();
        let cancellation = shared_cancellation();
        let worker = spawn_cpu_daemon(daemon, Arc::clone(&cancellation));

        assert!(matches!(
            exchange(&directory, cpu_submit("cpu-timeout", "fixture-cpu", 250)),
            ResponseMessage::Accepted {
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued,
                ..
            }
        ));
        wait_for_path(&started);
        assert_eq!(
            wait_for_request_state(&directory, "cpu-timeout", true),
            RequestStatus::Failed
        );
        let page = result_page(&directory, "cpu-timeout");
        assert_eq!(
            page.terminal,
            Some(ResultTerminal::Failed(ResultFailureCode::TimedOut))
        );
        assert!(page.events.is_empty());

        let report = cancel_outer_and_join(cancellation.as_ref(), worker);
        assert_eq!(report.stopped, StopReason::Cancelled);
        fixture.released();
    }

    #[test]
    fn cpu_decision_wrong_gpu_profile_is_rejected_before_acceptance() {
        let marker_dir = TempDir::new().unwrap();
        let marker = marker_dir.path().join("runs");
        let fixture = CpuFixture::new(&successful_cpu_runner(&marker));
        let daemon = fixture.bind(2);
        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            let submit = exchange(&directory, cpu_submit("gpu-refused", "fixture-gpu", 5_000));
            let status = exchange(
                &directory,
                WireRequestMessage::Status {
                    request_id: "gpu-refused".into(),
                },
            );
            (submit, status)
        });
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let report = daemon.run_cpu_decisions(
            &context,
            RecoveryPolicy::new(Duration::from_secs(2)).unwrap(),
        );
        let (submit, status) = client.join().unwrap();
        assert!(matches!(
            submit,
            ResponseMessage::Error {
                request_id: Some(ref id),
                code: ResponseErrorCode::Unavailable,
            } if id == "gpu-refused"
        ));
        assert_eq!(
            status,
            ResponseMessage::Status {
                request_id: "gpu-refused".into(),
                state: RequestStatus::NotFound,
            }
        );
        assert_eq!(report.unwrap().stopped, StopReason::ConnectionLimit);
        assert!(!marker.exists());
        fixture.released();
    }

    #[test]
    fn cpu_decision_capacity_is_reserved_before_request_is_accepted() {
        let fixture = CpuFixture::new("#!/bin/sh\nexit 99\n");
        let mut daemon = fixture.bind(2);
        let mut filled = 0;
        loop {
            match daemon.results.begin_reserved(
                &format!("held-{filled}"),
                crate::control::catalog::Lane::Decisions,
            ) {
                Ok(()) => filled += 1,
                Err(ResultBufferError::CapacityExceeded) => break,
                Err(error) => panic!("unexpected fixture buffer error: {error:?}"),
            }
        }
        assert!(filled > 0);

        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            let submit = exchange(
                &directory,
                cpu_submit("no-result-capacity", "fixture-cpu", 5_000),
            );
            let status = exchange(
                &directory,
                WireRequestMessage::Status {
                    request_id: "no-result-capacity".into(),
                },
            );
            (submit, status)
        });
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let report = daemon.run_cpu_decisions(
            &context,
            RecoveryPolicy::new(Duration::from_secs(2)).unwrap(),
        );
        let (submit, status) = client.join().unwrap();
        assert!(matches!(
            submit,
            ResponseMessage::Error {
                code: ResponseErrorCode::QueueFull,
                ..
            }
        ));
        assert!(matches!(
            status,
            ResponseMessage::Status {
                state: RequestStatus::NotFound,
                ..
            }
        ));
        assert_eq!(report.unwrap().stopped, StopReason::ConnectionLimit);
        fixture.released();
    }

    #[test]
    fn cpu_decision_active_shutdown_cancels_and_waits_for_worker_cleanup() {
        let marker_dir = TempDir::new().unwrap();
        let started = marker_dir.path().join("started");
        let fixture = CpuFixture::new(&sleeping_cpu_runner(&started));
        let daemon = fixture.bind(2);
        let directory = fixture.socket.path().to_owned();
        let cancellation = shared_cancellation();
        let worker = spawn_cpu_daemon(daemon, Arc::clone(&cancellation));
        assert!(matches!(
            exchange(
                &directory,
                cpu_submit("cpu-shutdown", "fixture-cpu", 10_000)
            ),
            ResponseMessage::Accepted {
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued,
                ..
            }
        ));
        wait_for_path(&started);
        assert_eq!(
            exchange(
                &directory,
                WireRequestMessage::Status {
                    request_id: "cpu-shutdown".into(),
                }
            ),
            ResponseMessage::Status {
                request_id: "cpu-shutdown".into(),
                state: RequestStatus::Running,
            }
        );
        let report = worker.join().unwrap().unwrap();
        assert_eq!(report.stopped, StopReason::ConnectionLimit);
        fixture.released();

        let restarted = fixture.bind(1);
        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            exchange(
                &directory,
                WireRequestMessage::Status {
                    request_id: "cpu-shutdown".into(),
                },
            )
        });
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let restart_report = restarted.run(&context).unwrap();
        assert_eq!(restart_report.completed_exchanges, 1);
        assert_eq!(
            client.join().unwrap(),
            ResponseMessage::Status {
                request_id: "cpu-shutdown".into(),
                state: RequestStatus::Cancelled,
            }
        );
        fixture.released();
    }

    fn exchange(directory: &Path, message: WireRequestMessage) -> ResponseMessage {
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let mut peer = LocalConnection::connect(directory, &context).unwrap();
        peer.write_request(&RequestEnvelope::v1(message), &context)
            .unwrap();
        peer.read_response(&context).unwrap().message
    }

    fn exchange_envelope(directory: &Path, request: RequestEnvelope) -> ResponseEnvelope {
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let mut peer = LocalConnection::connect(directory, &context).unwrap();
        peer.write_request(&request, &context).unwrap();
        peer.read_response(&context).unwrap()
    }

    #[test]
    fn status_is_served_but_submission_stays_closed_and_owner_is_released() {
        let fixture = Fixture::new();
        let daemon = fixture.bind(fixture.config(2)).unwrap();
        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            assert_eq!(
                exchange(
                    &directory,
                    WireRequestMessage::Status {
                        request_id: "missing".into(),
                    }
                ),
                ResponseMessage::Status {
                    request_id: "missing".into(),
                    state: RequestStatus::NotFound,
                }
            );
            assert_eq!(
                exchange(
                    &directory,
                    WireRequestMessage::Submit {
                        request_id: "closed".into(),
                        timeout_ms: 1000,
                        request: WireInferenceRequest::Language {
                            profile_id: "gemma-4-26b-qat-mtp".into(),
                            max_tokens: 8,
                            turns: vec![WireChatTurn {
                                role: WireChatRole::User,
                                content: "PRIVATE_PROMPT".into()
                            }],
                        },
                    }
                ),
                ResponseMessage::Error {
                    request_id: Some("closed".into()),
                    code: ResponseErrorCode::Unavailable,
                }
            );
        });
        let ctx = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let report = daemon.run(&ctx).unwrap();
        client.join().unwrap();
        assert_eq!(
            report,
            RunReport {
                completed_exchanges: 2,
                failed_connections: 0,
                stopped: StopReason::ConnectionLimit
            }
        );
        assert!(!fixture.data.path().join("jobs.json").exists());
        fixture.released();
    }

    #[test]
    fn results_keep_authoritative_queued_and_not_found_separate_from_empty_cache() {
        let fixture = Fixture::new();
        let mut daemon = fixture.bind(fixture.config(2)).unwrap();
        daemon
            .core
            .seed_test_request_status("retained-queued", "queued")
            .unwrap();
        assert_eq!(daemon.core.queued_count(), 0);
        assert_eq!(daemon.core.active_count(), 0);
        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            let queued = exchange_envelope(
                &directory,
                RequestEnvelope::v2_results("retained-queued", 0, MIN_PAGE_BYTES as u32),
            );
            let missing = exchange_envelope(
                &directory,
                RequestEnvelope::v2_results("not-retained", 0, MIN_PAGE_BYTES as u32),
            );
            (queued, missing)
        });
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let report = daemon.run(&context).unwrap();
        let (queued, missing) = client.join().unwrap();
        assert_eq!(report.completed_exchanges, 2);
        assert!(matches!(
            queued.message,
            ResponseMessage::Results {
                state: RequestStatus::Queued,
                page,
                ..
            } if page.availability == Availability::Unknown
                && page.events.is_empty()
        ));
        assert!(matches!(
            missing.message,
            ResponseMessage::Results {
                state: RequestStatus::NotFound,
                page,
                ..
            } if page.availability == Availability::Unknown
        ));
        fixture.released();
    }

    #[test]
    fn completed_result_pages_are_bounded_repeatable_after_lost_response() {
        let fixture = Fixture::new();
        let mut daemon = fixture.bind(fixture.config(2)).unwrap();
        daemon
            .core
            .seed_test_request_status("completed-result", "succeeded")
            .unwrap();
        daemon
            .results
            .begin("completed-result", crate::control::catalog::Lane::Language)
            .unwrap();
        daemon
            .results
            .append(
                "completed-result",
                ResultEvent::ChatContent("private completed body".into()),
            )
            .unwrap();
        daemon
            .results
            .terminal("completed-result", ResultTerminal::Succeeded)
            .unwrap();
        assert_eq!(daemon.core.queued_count(), 0);
        assert_eq!(daemon.core.active_count(), 0);

        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            let context =
                OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
            let mut lost = LocalConnection::connect(&directory, &context).unwrap();
            lost.write_request(
                &RequestEnvelope::v2_results("completed-result", 0, MIN_PAGE_BYTES as u32),
                &context,
            )
            .unwrap();
            drop(lost); // Simulate losing the first acknowledgement.

            exchange_envelope(
                &directory,
                RequestEnvelope::v2_results("completed-result", 0, MIN_PAGE_BYTES as u32),
            )
        });
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let report = daemon.run(&context).unwrap();
        let response = client.join().unwrap();
        assert_eq!(report.completed_exchanges + report.failed_connections, 2);
        assert!(matches!(
            response.message,
            ResponseMessage::Results {
                state: RequestStatus::Succeeded,
                byte_budget,
                page,
                ..
            } if byte_budget == MIN_PAGE_BYTES as u32
                && page.availability == Availability::Available
                && page.events.len() == 1
                && page.events[0].cursor == 1
                && page.next_cursor == 1
                && serde_json::to_vec(&page).unwrap().len() <= byte_budget as usize
        ));
        fixture.released();
    }

    #[test]
    fn restart_preserves_terminal_history_but_not_result_bodies_or_execution() {
        let fixture = Fixture::new();
        let mut first = fixture.bind(fixture.config(1)).unwrap();
        first
            .core
            .seed_test_request_status("restart-results", "succeeded")
            .unwrap();
        first
            .results
            .begin("restart-results", crate::control::catalog::Lane::Language)
            .unwrap();
        first
            .results
            .append(
                "restart-results",
                ResultEvent::ChatContent("body disappears on restart".into()),
            )
            .unwrap();
        first
            .results
            .terminal("restart-results", ResultTerminal::Succeeded)
            .unwrap();
        drop(first);

        let restarted = fixture.bind(fixture.config(1)).unwrap();
        assert_eq!(
            restarted.core.request_state("restart-results"),
            Some("succeeded")
        );
        assert_eq!(restarted.core.queued_count(), 0);
        assert_eq!(restarted.core.active_count(), 0);
        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            exchange_envelope(
                &directory,
                RequestEnvelope::v2_results("restart-results", 0, MIN_PAGE_BYTES as u32),
            )
        });
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let report = restarted.run(&context).unwrap();
        let response = client.join().unwrap();
        assert_eq!(report.completed_exchanges, 1);
        assert!(matches!(
            response.message,
            ResponseMessage::Results {
                state: RequestStatus::Succeeded,
                page,
                ..
            } if page.availability == Availability::Unknown
                && page.events.is_empty()
                && page.terminal.is_none()
        ));
        fixture.released();
    }

    #[test]
    fn malformed_and_stalled_peers_do_not_block_next_peer() {
        let fixture = Fixture::new();
        let daemon = fixture.bind(fixture.config(3)).unwrap();
        let directory = fixture.socket.path().to_owned();
        // Fill two connections before the good peer. First frame is invalid,
        // second holds an incomplete length prefix beyond the peer deadline.
        let mut malformed = UnixStream::connect(directory.join("coordinator.sock")).unwrap();
        malformed.write_all(&[0, 0, 0, 1, b'{']).unwrap();
        let mut stalled = UnixStream::connect(directory.join("coordinator.sock")).unwrap();
        stalled.write_all(&[0]).unwrap();
        let client = std::thread::spawn(move || {
            exchange(
                &directory,
                WireRequestMessage::Status {
                    request_id: "after-bad-peers".into(),
                },
            )
        });
        let ctx = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let report = daemon.run(&ctx).unwrap();
        assert!(matches!(
            client.join().unwrap(),
            ResponseMessage::Status {
                state: RequestStatus::NotFound,
                ..
            }
        ));
        assert_eq!(
            report,
            RunReport {
                completed_exchanges: 1,
                failed_connections: 2,
                stopped: StopReason::ConnectionLimit
            }
        );
        fixture.released();
    }

    struct Flag(AtomicBool);
    impl Cancellation for Flag {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }
    #[test]
    fn cancellation_interrupts_stalled_peer_and_releases_ownership() {
        let fixture = Fixture::new();
        let mut config = fixture.config(10);
        config.peer_timeout = Duration::from_secs(30);
        let daemon = fixture.bind(config).unwrap();
        let _stalled = UnixStream::connect(fixture.socket.path().join("coordinator.sock")).unwrap();
        let cancelled = Flag(AtomicBool::new(false));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(50));
                cancelled.0.store(true, Ordering::Release);
            });
            let ctx = OperationContext::new(Instant::now() + Duration::from_secs(3), &cancelled);
            assert_eq!(daemon.run(&ctx).unwrap().stopped, StopReason::Cancelled);
        });
        fixture.released();
    }

    #[test]
    fn overall_deadline_limits_idle_listener_and_stalled_peer() {
        for stalled in [false, true] {
            let fixture = Fixture::new();
            let daemon = fixture.bind(fixture.config(10)).unwrap();
            let _peer = stalled.then(|| {
                UnixStream::connect(fixture.socket.path().join("coordinator.sock")).unwrap()
            });
            let ctx =
                OperationContext::new(Instant::now() + Duration::from_millis(40), &NeverCancel);
            assert_eq!(daemon.run(&ctx).unwrap().stopped, StopReason::Deadline);
            fixture.released();
        }
    }

    #[test]
    fn competing_owner_does_not_touch_history() {
        let fixture = Fixture::new();
        std::fs::write(fixture.data.path().join("jobs.json"), "untouched").unwrap();
        let _guard = CoordinatorLock::acquire(fixture.locks.path(), [0]).unwrap();
        assert!(matches!(
            fixture.bind(fixture.config(1)),
            Err(DaemonError::Transport(LocalIpcError::OwnershipUnavailable))
        ));
        assert_eq!(
            std::fs::read_to_string(fixture.data.path().join("jobs.json")).unwrap(),
            "untouched"
        );
        assert!(!fixture.socket.path().join("coordinator.sock").exists());
    }

    #[test]
    fn nonregular_history_fails_before_binding() {
        let fixture = Fixture::new();
        std::fs::create_dir(fixture.data.path().join("jobs.json")).unwrap();
        assert!(matches!(
            fixture.bind(fixture.config(1)),
            Err(DaemonError::HistoryConflict)
        ));
        fixture.released();
    }

    #[test]
    fn standalone_directory_and_symlink_aliases_are_rejected_without_writes() {
        let fixture = Fixture::new();
        let history = fixture.data.path().join("jobs.json");
        std::fs::write(&history, "preserve me").unwrap();
        let mut config = fixture.config(1);
        config.standalone_data_directory = fixture.data.path().into();
        assert!(matches!(
            fixture.bind(config),
            Err(DaemonError::HistoryConflict)
        ));
        let alias = fixture.standalone.path().join("alias");
        std::os::unix::fs::symlink(fixture.data.path(), &alias).unwrap();
        let mut config = fixture.config(1);
        config.standalone_data_directory = alias;
        assert!(matches!(
            fixture.bind(config),
            Err(DaemonError::HistoryConflict)
        ));
        assert_eq!(std::fs::read_to_string(history).unwrap(), "preserve me");
        fixture.released();
    }

    #[test]
    fn aliased_history_files_are_rejected_and_preserved() {
        for symlink in [false, true] {
            let fixture = Fixture::new();
            let legacy = fixture.standalone.path().join("jobs.json");
            let coordinated = fixture.data.path().join("jobs.json");
            std::fs::write(&legacy, "legacy contents").unwrap();
            if symlink {
                std::os::unix::fs::symlink(&legacy, &coordinated).unwrap();
            } else {
                std::fs::hard_link(&legacy, &coordinated).unwrap();
            }
            assert!(matches!(
                fixture.bind(fixture.config(1)),
                Err(DaemonError::HistoryConflict)
            ));
            assert_eq!(std::fs::read_to_string(legacy).unwrap(), "legacy contents");
            fixture.released();
        }
        // Legacy clients follow symlinks too. The coordinator's regular file
        // can have nlink=1 while the legacy path resolves to the same inode.
        let fixture = Fixture::new();
        let coordinated = fixture.data.path().join("jobs.json");
        std::fs::write(&coordinated, "coordinator contents").unwrap();
        std::os::unix::fs::symlink(&coordinated, fixture.standalone.path().join("jobs.json"))
            .unwrap();
        assert!(matches!(
            fixture.bind(fixture.config(1)),
            Err(DaemonError::HistoryConflict)
        ));
        assert_eq!(
            std::fs::read_to_string(coordinated).unwrap(),
            "coordinator contents"
        );
        fixture.released();
    }

    #[test]
    fn absent_standalone_store_is_not_created_and_relative_path_is_rejected() {
        let fixture = Fixture::new();
        let absent = fixture.standalone.path().join("not-created");
        let mut config = fixture.config(1);
        config.standalone_data_directory = absent.clone();
        drop(fixture.bind(config).unwrap());
        assert!(!absent.exists());
        let mut config = fixture.config(1);
        config.standalone_data_directory = "relative".into();
        assert!(matches!(
            fixture.bind(config),
            Err(DaemonError::InvalidConfig)
        ));
        fixture.released();
    }

    #[test]
    fn invalid_configuration_and_nonprivate_data_fail_before_binding() {
        let fixture = Fixture::new();
        for invalid in 0..4 {
            let mut config = fixture.config(1);
            match invalid {
                0 => config.max_connections = 0,
                1 => config.queue_capacity = 129,
                2 => config.peer_timeout = Duration::ZERO,
                _ => config.peer_timeout = Duration::from_secs(31),
            }
            assert!(matches!(
                fixture.bind(config),
                Err(DaemonError::InvalidConfig)
            ));
        }
        std::fs::set_permissions(fixture.data.path(), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert!(matches!(
            fixture.bind(fixture.config(1)),
            Err(DaemonError::Transport(LocalIpcError::UnsafePath))
        ));
        fixture.released();
    }

    #[test]
    fn startup_interrupts_durable_work_without_replay() {
        use crate::job_history::{CrossToolkitJobStart, JobHistory};
        let fixture = Fixture::new();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let (mut history, _) = JobHistory::load_in(fixture.data.path(), root).unwrap();
        history
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: "l3ms".into(),
                request_id: Some("before-restart".into()),
                ..CrossToolkitJobStart::default()
            })
            .unwrap();
        history.persist_in(fixture.data.path()).unwrap();
        drop(history);
        let daemon = fixture.bind(fixture.config(1)).unwrap();
        assert_eq!(daemon.core.queued_count(), 0);
        assert_eq!(daemon.core.active_count(), 0);
        let directory = fixture.socket.path().to_owned();
        let client = std::thread::spawn(move || {
            exchange(
                &directory,
                WireRequestMessage::Status {
                    request_id: "before-restart".into(),
                },
            )
        });
        let ctx = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        daemon.run(&ctx).unwrap();
        assert_eq!(
            client.join().unwrap(),
            ResponseMessage::Status {
                request_id: "before-restart".into(),
                state: RequestStatus::Interrupted,
            }
        );
        fixture.released();
    }

    #[test]
    fn every_bound_gpu_requires_an_owned_device_lock() {
        use crate::control::catalog::Lane;
        let fixture = Fixture::new();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let bindings = BTreeMap::from([(
            RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap(),
            ResourceClaim::ExclusiveGpu(1),
        )]);
        assert!(matches!(
            ControlDaemon::bind(
                fixture.config(1),
                Catalog::load(root).unwrap(),
                bindings,
                root,
            ),
            Err(DaemonError::InvalidConfig)
        ));
        fixture.released();
    }
}
