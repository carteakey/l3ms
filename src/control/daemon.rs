//! Internal Linux control-loop owner; not an installed or dispatching daemon.
//!
//! Binding acquires listener/host/device ownership before history is loaded or
//! reconciled. Admission stays closed: this loop cannot dispatch, reconcile a
//! runtime, or expose its core for mutation. Each invocation consumes the owner
//! and releases its endpoint and locks on every return path.
//!
//! One authenticated peer is served at a time, one exchange per connection.
//! Peer time and accepted/denied connection attempts are bounded. Same-user
//! denial of service is still possible; UID authentication is not app isolation.
//! Local filesystem operations are synchronous and cannot be preempted. Caller
//! supplied private local directories and their ancestors must remain trusted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::catalog::{Catalog, RuntimeIdentity};
use super::inference::CoordinatorCore;
use super::ipc::{
    IncomingRequest, ResponseEnvelope, ResponseErrorCode, ResponseMessage, ResultsQuery,
};
use super::local_ipc::{private_directory, LocalIpcError, LocalListener};
use super::result_buffer::{ResultBuffer, ResultBufferError};
use super::runtime::{OperationContext, ResourceClaim};

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
        Availability, ResultEvent, ResultTerminal, MIN_PAGE_BYTES,
    };
    use crate::control::runtime::{Cancellation, NeverCancel};
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tempfile::TempDir;

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
