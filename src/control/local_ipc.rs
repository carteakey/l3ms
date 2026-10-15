//! Linux-only client transport for compatibility with an explicitly supplied
//! legacy local IPC server. It does not create or own a server.
//!
//! Same effective-UID peer checks are not isolation from other programs of
//! that user. Trusted directories/ancestors must not be maliciously replaced
//! concurrently. One connection carries one request/reply; v2 results are
//! bounded pages, not a stream. Deadlines apply across each whole read/write,
//! not separately to each byte. A transport failure after starting a request
//! write is uncertain; clients must reconnect by request ID rather than
//! resubmit. No retry or runtime operation occurs here.

use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::ipc::{self, ProtocolError, RequestEnvelope, ResponseEnvelope};
use super::runtime::OperationContext;

const SOCKET_NAME: &str = "coordinator.sock";
const POLL: Duration = Duration::from_millis(25);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalIpcError {
    UnsafePath,
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

/// A checked, nonblocking peer connection. No Debug to discourage logging
/// descriptors, endpoint paths, credentials or request payloads.
pub struct LocalConnection {
    stream: UnixStream,
    phase: Phase,
    request_id: Option<String>,
    request_version: Option<u16>,
    result_query: Option<(u64, u32)>,
    artifact_query: Option<ipc::ArtifactQuery>,
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    ClientRequest,
    ClientResponse,
    Done,
}
impl LocalConnection {
    fn authenticated(stream: UnixStream, uid: libc::uid_t) -> Result<Self, LocalIpcError> {
        verify_peer(&stream, uid)?;
        stream
            .set_nonblocking(true)
            .map_err(|_| LocalIpcError::Io)?;
        Ok(Self {
            stream,
            phase: Phase::ClientRequest,
            request_id: None,
            request_version: None,
            result_query: None,
            artifact_query: None,
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
                return Self::authenticated(stream, uid);
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
                        return Self::authenticated(stream, uid);
                    }
                },
                Some(libc::EINTR) => continue,
                _ => return Err(LocalIpcError::Unavailable),
            }
        }
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
            | ipc::WireRequestMessage::Results { request_id, .. }
            | ipc::WireRequestMessage::Artifact { request_id, .. } => request_id,
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
        self.artifact_query = match &request.message {
            ipc::WireRequestMessage::Artifact {
                request_id,
                artifact,
                offset,
                byte_budget,
                artifact_version,
            } => Some(ipc::ArtifactQuery {
                request_id: request_id.clone(),
                artifact: artifact.clone(),
                offset: *offset,
                byte_budget: *byte_budget,
                artifact_version: artifact_version.clone(),
            }),
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
            ipc::ResponseMessage::Results { request_id, .. }
            | ipc::ResponseMessage::Artifact { request_id, .. } => Some(request_id.as_str()),
            ipc::ResponseMessage::Error { request_id, .. } => request_id.as_deref(),
        };
        if self.request_version != Some(response.version)
            || (id.is_some() && id != self.request_id.as_deref())
            || (matches!(
                self.request_version,
                Some(ipc::RESULTS_PROTOCOL_VERSION | ipc::ARTIFACT_PROTOCOL_VERSION)
            ) && matches!(
                &response.message,
                ipc::ResponseMessage::Error {
                    request_id: None,
                    ..
                }
            ))
        {
            return Err(LocalIpcError::ResponseMismatch);
        }
        if let Some(query) = &self.artifact_query {
            if !matches!(response.message, ipc::ResponseMessage::Error { .. }) {
                ipc::validate_artifact_reply(
                    &query.request_id,
                    &query.artifact,
                    query.offset,
                    query.byte_budget,
                    query.artifact_version.as_ref(),
                    &response.message,
                )
                .map_err(|_| LocalIpcError::ResponseMismatch)?;
            }
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
    use super::super::runtime::{Cancellation, NeverCancel};
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;
    use std::time::Instant;

    fn context(ms: u64) -> OperationContext<'static> {
        OperationContext::new(Instant::now() + Duration::from_millis(ms), &NeverCancel)
    }

    fn private() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }

    fn fixture_listener(directory: &Path) -> UnixListener {
        let path = directory.join(SOCKET_NAME);
        let listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        listener.set_nonblocking(true).unwrap();
        listener
    }

    #[test]
    fn connect_rejects_unsafe_socket_entries() {
        let directory = private();
        let path = directory.path().join(SOCKET_NAME);

        assert_eq!(
            LocalConnection::connect(directory.path(), &context(100))
                .err()
                .unwrap(),
            LocalIpcError::Unavailable
        );

        std::fs::write(&path, "preserve").unwrap();
        assert_eq!(
            LocalConnection::connect(directory.path(), &context(100))
                .err()
                .unwrap(),
            LocalIpcError::UnsafePath
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "preserve");

        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink("missing", &path).unwrap();
        assert_eq!(
            LocalConnection::connect(directory.path(), &context(100))
                .err()
                .unwrap(),
            LocalIpcError::UnsafePath
        );
        assert!(path.symlink_metadata().unwrap().file_type().is_symlink());
    }

    #[test]
    fn peer_uid_mismatch_is_denied_before_payload_io() {
        let (stream, _peer) = UnixStream::pair().unwrap();
        assert_eq!(
            LocalConnection::authenticated(stream, unsafe { libc::geteuid() }.wrapping_add(1))
                .err()
                .unwrap(),
            LocalIpcError::PeerDenied
        );
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
        let mut connection =
            LocalConnection::authenticated(client, unsafe { libc::geteuid() }).unwrap();
        let request = RequestEnvelope::v1(ipc::WireRequestMessage::Submit {
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
    fn connect_observes_deadline_cancellation_and_unavailability() {
        let directory = private();
        assert_eq!(
            LocalConnection::connect(directory.path(), &context(100))
                .err()
                .unwrap(),
            LocalIpcError::Unavailable
        );
        let cancelled = OperationContext::new(Instant::now() + Duration::from_secs(1), &Cancel);
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
        let listener = fixture_listener(directory.path());
        assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 1) }, 0);
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
        let listener = fixture_listener(directory.path());
        assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 1) }, 0);
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
        assert!(raw.take_error().unwrap().is_none());
        assert_eq!(
            raw.peer_addr().unwrap_err().raw_os_error(),
            Some(libc::ENOTCONN)
        );
        let _accepted = listener.accept().unwrap();
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(
            raw.peer_addr().unwrap_err().raw_os_error(),
            Some(libc::ENOTCONN)
        );
        assert_eq!(connect(), 0);
        assert!(raw.peer_addr().is_ok());
    }
}
