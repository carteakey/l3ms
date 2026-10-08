//! Linux-only private Unix-socket transport for IPC v1/v2, not an installed daemon.
//!
//! The caller supplies pre-existing private local directories and acquires this
//! listener before loading/writing coordinator history. The host guard lives
//! with the listener. Existing socket paths are never silently reclaimed. Same
//! effective-UID peer checks are not isolation from other programs of that user.
//! Trusted directories/ancestors must not be maliciously replaced concurrently.
//!
//! One connection carries one request/reply; v2 results are bounded pages, not
//! a stream. Deadlines apply across each whole read/write, not separately to each byte. A transport
//! failure after starting a request write is uncertain; clients must reconnect
//! by request ID rather than resubmit. No retry or runtime operation occurs here.

use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::ipc::{
    self, ClientRequest, IncomingRequest, ProtocolError, RequestEnvelope, ResponseEnvelope,
};
use super::ownership::CoordinatorLock;
use super::runtime::OperationContext;

const SOCKET_NAME: &str = "coordinator.sock";
const POLL: Duration = Duration::from_millis(25);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalIpcError {
    UnsafePath,
    ExistingSocket,
    OwnershipUnavailable,
    PeerDenied,
    Unavailable,
    Io,
    Cancelled,
    Deadline,
    OutOfOrder,
    ResponseMismatch,
    Protocol(ProtocolError),
}
impl fmt::Display for LocalIpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "local IPC: {self:?}")
    }
}
impl Error for LocalIpcError {}

/// Owns the listening socket and host/device guard. Not Clone: only its owner
/// may dispatch or mutate history. No service is started by constructing it.
pub struct LocalListener {
    listener: UnixListener,
    path: PathBuf,
    identity: (u64, u64),
    uid: libc::uid_t,
    _ownership: CoordinatorLock,
}
impl LocalListener {
    pub fn bind(
        socket_directory: &Path,
        host_lock_directory: &Path,
        gpus: impl IntoIterator<Item = u32>,
    ) -> Result<Self, LocalIpcError> {
        let path = socket_path(socket_directory)?;
        private_directory(host_lock_directory)?;
        let ownership = CoordinatorLock::acquire(host_lock_directory, gpus)
            .map_err(|_| LocalIpcError::OwnershipUnavailable)?;
        match path.symlink_metadata() {
            Ok(_) => return Err(LocalIpcError::ExistingSocket),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(_) => return Err(LocalIpcError::UnsafePath),
        }
        let listener = UnixListener::bind(&path).map_err(|_| LocalIpcError::Unavailable)?;
        let metadata = path.symlink_metadata().map_err(|_| LocalIpcError::Io)?;
        let bound = Self {
            listener,
            path,
            identity: (metadata.dev(), metadata.ino()),
            uid: unsafe { libc::geteuid() },
            _ownership: ownership,
        };
        // The 0700 parent protects the bind-to-chmod interval without changing
        // process-global umask. RAII removes only our inode on setup failure.
        std::fs::set_permissions(&bound.path, std::fs::Permissions::from_mode(0o600))
            .map_err(|_| LocalIpcError::Io)?;
        bound
            .listener
            .set_nonblocking(true)
            .map_err(|_| LocalIpcError::Io)?;
        Ok(bound)
    }

    pub fn accept(&self, context: &OperationContext<'_>) -> Result<LocalConnection, LocalIpcError> {
        loop {
            active(context)?;
            match self.listener.accept() {
                Ok((stream, _)) => {
                    return LocalConnection::authenticated(stream, self.uid, Phase::ServerRequest)
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => pause(context)?,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => return Err(LocalIpcError::Io),
            }
        }
    }

    /// Borrow the host/device guard retained for this listener's full
    /// lifetime. Managed execution owners must receive this exact guard;
    /// acquiring a second lock would neither strengthen nor replace the
    /// listener's ownership.
    pub(crate) fn ownership_guard(&self) -> &CoordinatorLock {
        &self._ownership
    }
}
impl Drop for LocalListener {
    fn drop(&mut self) {
        // Do not delete an endpoint replaced by another writer or a symlink.
        if self.path.symlink_metadata().is_ok_and(|entry| {
            entry.file_type().is_socket()
                && (entry.dev(), entry.ino()) == self.identity
                && entry.uid() == self.uid
        }) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// A checked, nonblocking peer connection. No Debug to discourage logging
/// descriptors, endpoint paths, credentials or request payloads.
pub struct LocalConnection {
    stream: UnixStream,
    phase: Phase,
    request_id: Option<String>,
    request_version: Option<u16>,
    result_query: Option<(u64, u32)>,
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    ClientRequest,
    ClientResponse,
    ServerRequest,
    ServerResponse,
    Done,
}
impl LocalConnection {
    fn authenticated(
        stream: UnixStream,
        uid: libc::uid_t,
        phase: Phase,
    ) -> Result<Self, LocalIpcError> {
        verify_peer(&stream, uid)?;
        stream
            .set_nonblocking(true)
            .map_err(|_| LocalIpcError::Io)?;
        Ok(Self {
            stream,
            phase,
            request_id: None,
            request_version: None,
            result_query: None,
        })
    }

    /// A missing/refused coordinator fails explicitly; no direct-engine fallback.
    /// Connection setup is nonblocking and bounded, including a full backlog.
    pub fn connect(
        directory: &Path,
        context: &OperationContext<'_>,
    ) -> Result<Self, LocalIpcError> {
        active(context)?;
        let path = socket_path(directory)?;
        let uid = unsafe { libc::geteuid() };
        let entry = path
            .symlink_metadata()
            .map_err(|_| LocalIpcError::Unavailable)?;
        if !entry.file_type().is_socket() || entry.uid() != uid || entry.mode() & 0o077 != 0 {
            return Err(LocalIpcError::UnsafePath);
        }
        let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
        address.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let bytes = path.as_os_str().as_bytes();
        for (target, source) in address.sun_path.iter_mut().zip(bytes) {
            *target = *source as libc::c_char;
        }
        let length = (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1)
            as libc::socklen_t;
        loop {
            active(context)?;
            let fd = unsafe {
                libc::socket(
                    libc::AF_UNIX,
                    libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
                    0,
                )
            };
            if fd < 0 {
                return Err(LocalIpcError::Io);
            }
            let stream = unsafe { UnixStream::from_raw_fd(fd) };
            let result =
                unsafe { libc::connect(fd, (&address as *const libc::sockaddr_un).cast(), length) };
            if result == 0 {
                active(context)?;
                return Self::authenticated(stream, uid, Phase::ClientRequest);
            }
            let error = io::Error::last_os_error();
            match error.raw_os_error() {
                // Unix EAGAIN means backlog full, NOT a pending connect. Retry
                // only connection establishment, on a fresh socket, never a frame.
                Some(libc::EAGAIN) => pause(context)?,
                Some(libc::EINPROGRESS) => loop {
                    active(context)?;
                    let mut poll = libc::pollfd {
                        fd,
                        events: libc::POLLOUT,
                        revents: 0,
                    };
                    let wait = context.remaining().min(POLL).as_millis().max(1) as i32;
                    let ready = unsafe { libc::poll(&mut poll, 1, wait) };
                    if ready < 0 {
                        if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                            continue;
                        }
                        return Err(LocalIpcError::Io);
                    }
                    if ready > 0 {
                        if stream
                            .take_error()
                            .map_err(|_| LocalIpcError::Io)?
                            .is_some()
                        {
                            return Err(LocalIpcError::Unavailable);
                        }
                        active(context)?;
                        return Self::authenticated(stream, uid, Phase::ClientRequest);
                    }
                },
                Some(libc::EINTR) => continue,
                _ => return Err(LocalIpcError::Unavailable),
            }
        }
    }

    pub fn read_request(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<ClientRequest, LocalIpcError> {
        self.start(Phase::ServerRequest)?;
        let result = ipc::read_request(&mut BoundedIo {
            stream: &mut self.stream,
            context,
        });
        let request = checked(result, context)?;
        self.request_id = Some(request.request_id().to_owned());
        self.request_version = Some(ipc::PROTOCOL_VERSION);
        self.phase = Phase::ServerResponse;
        Ok(request)
    }
    pub fn read_incoming_request(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<IncomingRequest, LocalIpcError> {
        self.start(Phase::ServerRequest)?;
        let result = ipc::read_incoming_request(&mut BoundedIo {
            stream: &mut self.stream,
            context,
        });
        let request = checked(result, context)?;
        self.request_id = Some(request.request_id().to_owned());
        self.request_version = Some(request.version());
        self.result_query = match &request {
            IncomingRequest::Results(query) => Some((query.after_cursor, query.byte_budget)),
            IncomingRequest::Control(_) => None,
        };
        self.phase = Phase::ServerResponse;
        Ok(request)
    }
    pub fn write_request(
        &mut self,
        request: &RequestEnvelope,
        context: &OperationContext<'_>,
    ) -> Result<(), LocalIpcError> {
        self.start(Phase::ClientRequest)?;
        let id = match &request.message {
            ipc::WireRequestMessage::Submit { request_id, .. }
            | ipc::WireRequestMessage::Status { request_id }
            | ipc::WireRequestMessage::Cancel { request_id }
            | ipc::WireRequestMessage::Results { request_id, .. } => request_id,
        };
        let result = ipc::write_request(
            &mut BoundedIo {
                stream: &mut self.stream,
                context,
            },
            request,
        );
        checked(result, context)?;
        self.request_id = Some(id.clone());
        self.request_version = Some(request.version);
        self.result_query = match &request.message {
            ipc::WireRequestMessage::Results {
                after_cursor,
                byte_budget,
                ..
            } => Some((*after_cursor, *byte_budget)),
            _ => None,
        };
        self.phase = Phase::ClientResponse;
        Ok(())
    }
    pub fn read_response(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<ResponseEnvelope, LocalIpcError> {
        self.start(Phase::ClientResponse)?;
        let result = ipc::read_response(&mut BoundedIo {
            stream: &mut self.stream,
            context,
        });
        let response = checked(result, context)?;
        self.match_response(&response)?;
        Ok(response)
    }
    pub fn write_response(
        &mut self,
        response: &ResponseEnvelope,
        context: &OperationContext<'_>,
    ) -> Result<(), LocalIpcError> {
        self.start(Phase::ServerResponse)?;
        self.match_response(response)?;
        let result = ipc::write_response(
            &mut BoundedIo {
                stream: &mut self.stream,
                context,
            },
            response,
        );
        checked(result, context)
    }

    fn start(&mut self, expected: Phase) -> Result<(), LocalIpcError> {
        if self.phase != expected {
            return Err(LocalIpcError::OutOfOrder);
        }
        self.phase = Phase::Done; // Partial IO is never resumed/replayed.
        Ok(())
    }
    fn match_response(&self, response: &ResponseEnvelope) -> Result<(), LocalIpcError> {
        let id = match &response.message {
            ipc::ResponseMessage::Accepted { request_id, .. }
            | ipc::ResponseMessage::Status { request_id, .. } => Some(request_id.as_str()),
            ipc::ResponseMessage::Results { request_id, .. } => Some(request_id.as_str()),
            ipc::ResponseMessage::Error { request_id, .. } => request_id.as_deref(),
        };
        if self.request_version != Some(response.version)
            || (id.is_some() && id != self.request_id.as_deref())
            || (self.request_version == Some(ipc::RESULTS_PROTOCOL_VERSION)
                && matches!(
                    &response.message,
                    ipc::ResponseMessage::Error {
                        request_id: None,
                        ..
                    }
                ))
        {
            return Err(LocalIpcError::ResponseMismatch);
        }
        if let ipc::ResponseMessage::Results {
            after_cursor,
            byte_budget,
            ..
        } = &response.message
        {
            if self.result_query != Some((*after_cursor, *byte_budget)) {
                return Err(LocalIpcError::ResponseMismatch);
            }
        } else if self.result_query.is_some()
            && !matches!(response.message, ipc::ResponseMessage::Error { .. })
        {
            return Err(LocalIpcError::ResponseMismatch);
        }
        Ok(())
    }
}

fn checked<T>(
    result: Result<T, ProtocolError>,
    context: &OperationContext<'_>,
) -> Result<T, LocalIpcError> {
    active(context)?;
    result.map_err(LocalIpcError::Protocol)
}
pub(crate) fn private_directory(directory: &Path) -> Result<PathBuf, LocalIpcError> {
    let directory = directory
        .canonicalize()
        .map_err(|_| LocalIpcError::UnsafePath)?;
    let entry = directory
        .metadata()
        .map_err(|_| LocalIpcError::UnsafePath)?;
    if !entry.is_dir() || entry.uid() != unsafe { libc::geteuid() } || entry.mode() & 0o077 != 0 {
        return Err(LocalIpcError::UnsafePath);
    }
    Ok(directory)
}
fn socket_path(directory: &Path) -> Result<PathBuf, LocalIpcError> {
    let path = private_directory(directory)?.join(SOCKET_NAME);
    // Linux pathname sockets have a 108-byte sun_path including trailing NUL.
    let bytes = path.as_os_str().as_bytes();
    if bytes.len() > 100 || bytes.contains(&0) {
        return Err(LocalIpcError::UnsafePath);
    }
    Ok(path)
}
fn verify_peer(stream: &UnixStream, uid: libc::uid_t) -> Result<(), LocalIpcError> {
    let mut credentials: libc::ucred = unsafe { std::mem::zeroed() };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let result = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut length,
        )
    };
    if result != 0
        || length as usize != std::mem::size_of::<libc::ucred>()
        || credentials.uid != uid
    {
        return Err(LocalIpcError::PeerDenied);
    }
    Ok(())
}
fn active(context: &OperationContext<'_>) -> Result<(), LocalIpcError> {
    if context.is_cancelled() {
        Err(LocalIpcError::Cancelled)
    } else if context.remaining().is_zero() {
        Err(LocalIpcError::Deadline)
    } else {
        Ok(())
    }
}
fn pause(context: &OperationContext<'_>) -> Result<(), LocalIpcError> {
    active(context)?;
    std::thread::sleep(context.remaining().min(POLL));
    active(context)
}

struct BoundedIo<'a, 'b> {
    stream: &'a mut UnixStream,
    context: &'a OperationContext<'b>,
}
impl BoundedIo<'_, '_> {
    fn wait(&self) -> io::Result<()> {
        pause(self.context).map_err(|_| io::ErrorKind::TimedOut.into())
    }
    fn check(&self) -> io::Result<()> {
        active(self.context).map_err(|_| io::ErrorKind::TimedOut.into())
    }
}
impl Read for BoundedIo<'_, '_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        loop {
            self.check()?;
            match self.stream.read(bytes) {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => self.wait()?,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => return result,
            }
        }
    }
}
impl Write for BoundedIo<'_, '_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        loop {
            self.check()?;
            match self.stream.write(bytes) {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => self.wait()?,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.check()
    }
}

#[cfg(test)]
mod tests {
    use super::super::ipc::{RequestStatus, ResponseMessage, WireRequestMessage};
    use super::super::runtime::{Cancellation, NeverCancel};
    use super::*;
    use std::time::Instant;

    fn context(ms: u64) -> OperationContext<'static> {
        OperationContext::new(Instant::now() + Duration::from_millis(ms), &NeverCancel)
    }
    fn private() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }
    fn status_request(id: &str) -> RequestEnvelope {
        RequestEnvelope::v1(WireRequestMessage::Status {
            request_id: id.into(),
        })
    }
    fn response(id: &str) -> ResponseEnvelope {
        ResponseEnvelope::v1(ResponseMessage::Status {
            request_id: id.into(),
            state: RequestStatus::NotFound,
        })
    }
    fn pair() -> (LocalConnection, UnixStream) {
        let (server, peer) = UnixStream::pair().unwrap();
        (
            LocalConnection::authenticated(
                server,
                unsafe { libc::geteuid() },
                Phase::ServerRequest,
            )
            .unwrap(),
            peer,
        )
    }

    #[test]
    fn private_listener_round_trip_holds_host_guard_and_cleans_its_socket() {
        let directory = private();
        let locks = private();
        let listener = LocalListener::bind(directory.path(), locks.path(), [0]).unwrap();
        let path = directory.path().join(SOCKET_NAME);
        assert_eq!(path.metadata().unwrap().mode() & 0o777, 0o600);
        assert!(CoordinatorLock::acquire(locks.path(), [0]).is_err());
        let socket_directory = directory.path().to_owned();
        let client = std::thread::spawn(move || {
            let mut connection =
                LocalConnection::connect(&socket_directory, &context(2000)).unwrap();
            connection
                .write_request(&status_request("client-a"), &context(2000))
                .unwrap();
            assert_eq!(
                connection
                    .write_request(&status_request("client-a"), &context(2000))
                    .unwrap_err(),
                LocalIpcError::OutOfOrder
            );
            let result = connection.read_response(&context(2000)).unwrap();
            assert_eq!(result.message, response("client-a").message);
            assert_eq!(
                connection.read_response(&context(100)).err().unwrap(),
                LocalIpcError::OutOfOrder
            );
        });
        let mut connection = listener.accept(&context(2000)).unwrap();
        let request = connection.read_request(&context(2000)).unwrap();
        assert_eq!(request.request_id(), "client-a");
        connection
            .write_response(&response("client-a"), &context(2000))
            .unwrap();
        client.join().unwrap();
        drop(listener);
        assert!(!path.exists());
        assert!(CoordinatorLock::acquire(locks.path(), [0]).is_ok());
    }

    #[test]
    fn child_process_transport_probe() {
        let Some(directory) = std::env::var_os("L3MS_TEST_LOCAL_IPC_DIRECTORY") else {
            return;
        };
        let id = std::env::var("L3MS_TEST_LOCAL_IPC_REQUEST_ID").unwrap();
        let ctx = context(3000);
        let mut connection = LocalConnection::connect(Path::new(&directory), &ctx).unwrap();
        connection
            .write_request(&status_request(&id), &ctx)
            .unwrap();
        assert_eq!(
            connection.read_response(&ctx).unwrap().message,
            response(&id).message
        );
    }

    #[test]
    fn two_independent_processes_use_the_same_private_listener() {
        let directory = private();
        let locks = private();
        let listener = LocalListener::bind(directory.path(), locks.path(), []).unwrap();
        let mut children = Vec::new();
        for id in ["proc-a", "proc-b"] {
            children.push(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "control::local_ipc::tests::child_process_transport_probe",
                        "--nocapture",
                    ])
                    .env("L3MS_TEST_LOCAL_IPC_DIRECTORY", directory.path())
                    .env("L3MS_TEST_LOCAL_IPC_REQUEST_ID", id)
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .unwrap(),
            );
        }
        let mut ids = std::collections::BTreeSet::new();
        for _ in 0..2 {
            let ctx = context(3000);
            let mut connection = listener.accept(&ctx).unwrap();
            let request = connection.read_request(&ctx).unwrap();
            let id = request.request_id().to_owned();
            connection.write_response(&response(&id), &ctx).unwrap();
            ids.insert(id);
        }
        for mut child in children {
            assert!(child.wait().unwrap().success());
        }
        assert_eq!(
            ids,
            std::collections::BTreeSet::from(["proc-a".into(), "proc-b".into()])
        );
    }

    #[test]
    fn unsafe_paths_and_existing_entries_are_preserved() {
        let directory = private();
        let locks = private();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(
            LocalListener::bind(directory.path(), locks.path(), [])
                .err()
                .unwrap(),
            LocalIpcError::UnsafePath
        );
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory.path().join(SOCKET_NAME);
        std::fs::write(&path, "preserve").unwrap();
        assert_eq!(
            LocalListener::bind(directory.path(), locks.path(), [])
                .err()
                .unwrap(),
            LocalIpcError::ExistingSocket
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "preserve");
        assert_eq!(
            LocalConnection::connect(directory.path(), &context(100))
                .err()
                .unwrap(),
            LocalIpcError::UnsafePath
        );
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink("missing", &path).unwrap();
        assert_eq!(
            LocalListener::bind(directory.path(), locks.path(), [])
                .err()
                .unwrap(),
            LocalIpcError::ExistingSocket
        );
        assert!(path.symlink_metadata().unwrap().file_type().is_symlink());
    }

    #[test]
    fn competitor_and_stale_socket_are_not_silently_reclaimed() {
        let directory = private();
        let other = private();
        let locks = private();
        let listener = LocalListener::bind(directory.path(), locks.path(), []).unwrap();
        assert_eq!(
            LocalListener::bind(other.path(), locks.path(), [])
                .err()
                .unwrap(),
            LocalIpcError::OwnershipUnavailable
        );
        drop(listener);
        let path = directory.path().join(SOCKET_NAME);
        let stale = UnixListener::bind(&path).unwrap();
        drop(stale);
        assert_eq!(
            LocalListener::bind(directory.path(), locks.path(), [])
                .err()
                .unwrap(),
            LocalIpcError::ExistingSocket
        );
        assert!(path.symlink_metadata().unwrap().file_type().is_socket());
    }

    #[test]
    fn drop_never_removes_a_replaced_entry() {
        let directory = private();
        let locks = private();
        let listener = LocalListener::bind(directory.path(), locks.path(), []).unwrap();
        let path = directory.path().join(SOCKET_NAME);
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, "replacement").unwrap();
        drop(listener);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replacement");
    }

    #[test]
    fn peer_uid_mismatch_is_denied_before_payload_io() {
        let (first, _second) = UnixStream::pair().unwrap();
        assert_eq!(
            LocalConnection::authenticated(
                first,
                unsafe { libc::geteuid() }.wrapping_add(1),
                Phase::ServerRequest
            )
            .err()
            .unwrap(),
            LocalIpcError::PeerDenied
        );
    }

    #[test]
    fn trickling_header_does_not_reset_the_whole_read_deadline() {
        let (mut connection, mut peer) = pair();
        let trickle = std::thread::spawn(move || {
            for byte in [0u8, 0, 1, 0] {
                if peer.write_all(&[byte]).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(40));
            }
        });
        let start = Instant::now();
        assert_eq!(
            connection.read_request(&context(90)).err().unwrap(),
            LocalIpcError::Deadline
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(
            connection.read_request(&context(100)).err().unwrap(),
            LocalIpcError::OutOfOrder
        );
        drop(connection);
        trickle.join().unwrap();
    }

    #[test]
    fn malformed_and_oversized_frames_are_bounded_and_redacted() {
        for payload in [
            (ipc::MAX_FRAME_BYTES as u32 + 1).to_be_bytes().to_vec(),
            vec![0, 0, 0, 3, b'x', b'y', b'z'],
        ] {
            let (mut connection, mut peer) = pair();
            peer.write_all(&payload).unwrap();
            let error = connection.read_request(&context(100)).err().unwrap();
            assert!(matches!(
                error,
                LocalIpcError::Protocol(
                    ProtocolError::FrameTooLarge | ProtocolError::MalformedFrame
                )
            ));
            assert!(!error.to_string().contains("xyz"));
        }
    }

    #[test]
    fn stalled_write_is_bounded_and_never_resumed() {
        let (client, _peer) = UnixStream::pair().unwrap();
        let size: libc::c_int = 4096;
        assert_eq!(
            unsafe {
                libc::setsockopt(
                    client.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_SNDBUF,
                    (&size as *const libc::c_int).cast(),
                    std::mem::size_of_val(&size) as libc::socklen_t,
                )
            },
            0
        );
        let mut connection = LocalConnection::authenticated(
            client,
            unsafe { libc::geteuid() },
            Phase::ClientRequest,
        )
        .unwrap();
        let request = RequestEnvelope::v1(WireRequestMessage::Submit {
            request_id: "large".into(),
            timeout_ms: 1000,
            request: ipc::WireInferenceRequest::Language {
                profile_id: "fixture".into(),
                turns: vec![ipc::WireChatTurn {
                    role: ipc::WireChatRole::User,
                    content: "x".repeat(900_000),
                }],
                max_tokens: 8,
            },
        });
        let start = Instant::now();
        assert_eq!(
            connection
                .write_request(&request, &context(80))
                .unwrap_err(),
            LocalIpcError::Deadline
        );
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(
            connection
                .write_request(&request, &context(100))
                .unwrap_err(),
            LocalIpcError::OutOfOrder
        );
    }

    struct Cancel;
    impl Cancellation for Cancel {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    #[test]
    fn accept_and_connect_observe_deadline_cancellation_and_unavailability() {
        let directory = private();
        let locks = private();
        assert_eq!(
            LocalConnection::connect(directory.path(), &context(100))
                .err()
                .unwrap(),
            LocalIpcError::Unavailable
        );
        let listener = LocalListener::bind(directory.path(), locks.path(), []).unwrap();
        assert_eq!(
            listener.accept(&context(30)).err().unwrap(),
            LocalIpcError::Deadline
        );
        let cancelled = OperationContext::new(Instant::now() + Duration::from_secs(1), &Cancel);
        assert_eq!(
            listener.accept(&cancelled).err().unwrap(),
            LocalIpcError::Cancelled
        );
        assert_eq!(
            LocalConnection::connect(directory.path(), &cancelled)
                .err()
                .unwrap(),
            LocalIpcError::Cancelled
        );
    }

    #[test]
    fn full_socket_backlog_does_not_create_an_unbounded_connect() {
        let directory = private();
        let locks = private();
        let listener = LocalListener::bind(directory.path(), locks.path(), []).unwrap();
        assert_eq!(unsafe { libc::listen(listener.listener.as_raw_fd(), 1) }, 0);
        let _first = LocalConnection::connect(directory.path(), &context(1000)).unwrap();
        let _second = LocalConnection::connect(directory.path(), &context(1000)).unwrap();
        let start = Instant::now();
        assert_eq!(
            LocalConnection::connect(directory.path(), &context(80))
                .err()
                .unwrap(),
            LocalIpcError::Deadline
        );
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn unix_backlog_eagain_is_not_a_connected_or_pending_socket() {
        let directory = private();
        let locks = private();
        let listener = LocalListener::bind(directory.path(), locks.path(), []).unwrap();
        assert_eq!(unsafe { libc::listen(listener.listener.as_raw_fd(), 1) }, 0);
        let _first = LocalConnection::connect(directory.path(), &context(1000)).unwrap();
        let _second = LocalConnection::connect(directory.path(), &context(1000)).unwrap();
        let fd = unsafe {
            libc::socket(
                libc::AF_UNIX,
                libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
                0,
            )
        };
        assert!(fd >= 0);
        let raw = unsafe { UnixStream::from_raw_fd(fd) };
        let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
        address.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let path = socket_path(directory.path()).unwrap();
        let bytes = path.as_os_str().as_bytes();
        for (target, source) in address.sun_path.iter_mut().zip(bytes) {
            *target = *source as libc::c_char;
        }
        let length = (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1)
            as libc::socklen_t;
        let connect =
            || unsafe { libc::connect(fd, (&address as *const libc::sockaddr_un).cast(), length) };
        assert_eq!(connect(), -1);
        assert_eq!(
            io::Error::last_os_error().raw_os_error(),
            Some(libc::EAGAIN)
        );
        // A zero SO_ERROR is not sufficient to establish this connection.
        assert!(raw.take_error().unwrap().is_none());
        assert_eq!(
            raw.peer_addr().unwrap_err().raw_os_error(),
            Some(libc::ENOTCONN)
        );
        let _accepted = listener.accept(&context(100)).unwrap();
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(
            raw.peer_addr().unwrap_err().raw_os_error(),
            Some(libc::ENOTCONN)
        );
        // Freeing backlog alone did not finish it; connect must be called again.
        assert_eq!(connect(), 0);
        assert!(raw.peer_addr().is_ok());
    }

    #[test]
    fn mismatched_response_is_rejected_and_connection_cannot_be_reused() {
        let (mut connection, mut peer) = pair();
        ipc::write_request(&mut peer, &status_request("owned")).unwrap();
        connection.read_request(&context(100)).unwrap();
        assert_eq!(
            connection
                .write_response(&response("unrelated"), &context(100))
                .unwrap_err(),
            LocalIpcError::ResponseMismatch
        );
        assert_eq!(
            connection
                .write_response(&response("owned"), &context(100))
                .unwrap_err(),
            LocalIpcError::OutOfOrder
        );
    }
}
