//! Private byte relay for a backend inside an isolated network namespace.
//!
//! The helper process binds a private Unix socket while its network namespace
//! differs from the coordinator captured at launch. It accepts only that
//! coordinator's live PID/UID/start-time identity, and connects exclusively to
//! 127.0.0.1 at a validated port. This is an opaque HTTP-byte transport: it
//! does not inspect prompts, retry requests, prove admission closure, drain
//! asynchronous jobs, or establish GPU release.
//!
//! The caller must use a dedicated pre-existing 0700 directory on a trusted,
//! stable local filesystem and retain its singleton ownership lock.
//! The trusted coordinator must remain in its captured network namespace;
//! the handed-off namespace file pins that namespace, not future setns calls.
//! A relay error after any backend request byte was written is uncertain; callers must
//! not replay that request automatically. The 16 MiB limit is per connection
//! and direction, with bounded backpressure buffers. A relay receipt only says
//! both byte streams reached EOF and drained; it is not proof that one native
//! HTTP request or asynchronous job completed. Keep-alive protocols may not
//! reach EOF after a response. Lifecycle adapters must manage request framing,
//! client-pool shutdown, and stream/job draining separately; this module never
//! parses HTTP.

use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::num::NonZeroU16;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::runtime::OperationContext;

pub const ENGINE_SOCKET_NAME: &str = "engine.sock";
const SOCKET_NAME: &str = ENGINE_SOCKET_NAME;
const MAX_DIRECTION_BYTES: u64 = 16 * 1024 * 1024;
const BUFFER_BYTES: usize = 64 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Fixed, redacted setup errors. No variant carries an OS error or path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineTransportError {
    InvalidIdentity,
    InvalidBackendPort,
    UnsafeDirectory,
    ExistingSocket,
    CoordinatorUnavailable,
    SharedNetworkNamespace,
    SocketUnavailable,
    Io,
}

impl fmt::Display for EngineTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidIdentity => "engine transport: invalid coordinator identity",
            Self::InvalidBackendPort => "engine transport: invalid backend port",
            Self::UnsafeDirectory => "engine transport: unsafe private directory",
            Self::ExistingSocket => "engine transport: endpoint already exists",
            Self::CoordinatorUnavailable => "engine transport: coordinator identity unavailable",
            Self::SharedNetworkNamespace => {
                "engine transport: coordinator shares the engine network namespace"
            }
            Self::SocketUnavailable => "engine transport: private endpoint unavailable",
            Self::Io => "engine transport: local I/O failed",
        })
    }
}

impl Error for EngineTransportError {}

/// Coordinator identity captured before launching the isolated helper.
///
/// The fields are launch metadata, not credentials. A helper still validates
/// every field against the live process, its pidfd, and `/proc` before binding
/// or relaying. Custom Debug intentionally omits even this metadata so callers
/// do not casually log process identity.
#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorIdentityDescriptor {
    pid: u32,
    uid: u32,
    start_time_ticks: u64,
    network_namespace_device: u64,
    network_namespace_inode: u64,
}

impl fmt::Debug for CoordinatorIdentityDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CoordinatorIdentityDescriptor([redacted])")
    }
}

impl CoordinatorIdentityDescriptor {
    /// Captures the current daemon process for a launch descriptor.
    pub fn capture_current() -> Result<Self, EngineTransportError> {
        let pid = unsafe { libc::getpid() };
        let uid = unsafe { libc::geteuid() };
        let process = process_info(pid)?;
        if process.uid != uid {
            return Err(EngineTransportError::InvalidIdentity);
        }
        let namespace = namespace_info(Path::new("/proc/self/ns/net"))?;
        let process_after = process_info(pid)?;
        let namespace_after = namespace_info(Path::new("/proc/self/ns/net"))?;
        if process_after != process || namespace_after != namespace {
            return Err(EngineTransportError::CoordinatorUnavailable);
        }
        let descriptor = Self {
            pid: u32::try_from(pid).map_err(|_| EngineTransportError::InvalidIdentity)?,
            uid,
            start_time_ticks: process.start_time_ticks,
            network_namespace_device: namespace.0,
            network_namespace_inode: namespace.1,
        };
        descriptor.validate_shape()?;
        Ok(descriptor)
    }

    /// Captures both scalar process identity and a retained network-namespace
    /// file for a trusted launch handoff. The helper must receive this file
    /// descriptor (for example through a manager-opened `OpenFile` descriptor
    /// or `SCM_RIGHTS`); the scalar namespace numbers are not a substitute.
    pub fn capture_current_handoff() -> Result<CoordinatorIdentityHandoff, EngineTransportError> {
        let descriptor = Self::capture_current()?;
        let namespace = File::open("/proc/self/ns/net")
            .map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
        let handoff = CoordinatorIdentityHandoff::from_namespace_file(descriptor, namespace)?;
        let process = process_info(descriptor.pid as libc::pid_t)?;
        if process.uid != descriptor.uid || process.start_time_ticks != descriptor.start_time_ticks
        {
            return Err(EngineTransportError::CoordinatorUnavailable);
        }
        Ok(handoff)
    }

    /// Constructs a descriptor received through the trusted launch handoff.
    /// The helper performs authoritative live-process checks before use.
    pub fn from_parts(
        pid: u32,
        uid: u32,
        start_time_ticks: u64,
        network_namespace_device: u64,
        network_namespace_inode: u64,
    ) -> Result<Self, EngineTransportError> {
        let descriptor = Self {
            pid,
            uid,
            start_time_ticks,
            network_namespace_device,
            network_namespace_inode,
        };
        descriptor.validate_shape()?;
        Ok(descriptor)
    }

    pub fn pid(self) -> u32 {
        self.pid
    }

    pub fn uid(self) -> u32 {
        self.uid
    }

    pub fn start_time_ticks(self) -> u64 {
        self.start_time_ticks
    }

    pub fn network_namespace_device(self) -> u64 {
        self.network_namespace_device
    }

    pub fn network_namespace_inode(self) -> u64 {
        self.network_namespace_inode
    }

    fn validate_shape(self) -> Result<(), EngineTransportError> {
        if self.pid <= 1
            || self.pid > i32::MAX as u32
            || self.start_time_ticks == 0
            || self.network_namespace_inode == 0
        {
            return Err(EngineTransportError::InvalidIdentity);
        }
        Ok(())
    }
}

/// Non-cloneable launch handoff. Its retained file descriptor is authoritative
/// namespace evidence; the PID/UID/start-time descriptor alone cannot bind an
/// engine helper. Callers must pass the FD explicitly to the helper before it
/// enters its private network namespace.
pub struct CoordinatorIdentityHandoff {
    descriptor: CoordinatorIdentityDescriptor,
    network_namespace: File,
}

impl fmt::Debug for CoordinatorIdentityHandoff {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CoordinatorIdentityHandoff([redacted])")
    }
}

impl CoordinatorIdentityHandoff {
    /// Adopts a manager- or coordinator-passed namespace FD and verifies its
    /// filesystem type and identity against the launch descriptor.
    pub fn from_namespace_file(
        descriptor: CoordinatorIdentityDescriptor,
        network_namespace: File,
    ) -> Result<Self, EngineTransportError> {
        descriptor.validate_shape()?;
        let actual = namespace_file_info(&network_namespace)?;
        if actual
            != (
                descriptor.network_namespace_device,
                descriptor.network_namespace_inode,
            )
        {
            return Err(EngineTransportError::InvalidIdentity);
        }
        Ok(Self {
            descriptor,
            network_namespace,
        })
    }

    pub fn descriptor(&self) -> CoordinatorIdentityDescriptor {
        self.descriptor
    }

    pub fn network_namespace_file(&self) -> &File {
        &self.network_namespace
    }

    /// Transfers ownership of the descriptor and retained namespace FD to a
    /// launch adapter that passes the FD to the helper.
    pub fn into_parts(self) -> (CoordinatorIdentityDescriptor, File) {
        (self.descriptor, self.network_namespace)
    }
}

/// A non-zero TCP port on the engine helper's loopback interface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendPort(NonZeroU16);

impl BackendPort {
    pub fn new(port: u16) -> Result<Self, EngineTransportError> {
        NonZeroU16::new(port)
            .map(Self)
            .ok_or(EngineTransportError::InvalidBackendPort)
    }

    pub fn get(self) -> u16 {
        self.0.get()
    }
}

/// The reason a relay stopped before any request byte reached the backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoDispatchReason {
    CoordinatorUnavailable,
    PeerDenied,
    Cancelled,
    Deadline,
    BackendUnavailable,
    RequestTooLarge,
    ResponseTooLarge,
    Io,
}

/// A relay failure is explicit about whether backend dispatch may have begun.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelayFailure {
    NoDispatch(NoDispatchReason),
    Uncertain,
}

impl fmt::Display for RelayFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NoDispatch(NoDispatchReason::CoordinatorUnavailable) => {
                "engine transport: coordinator unavailable before dispatch"
            }
            Self::NoDispatch(NoDispatchReason::PeerDenied) => {
                "engine transport: peer identity denied"
            }
            Self::NoDispatch(NoDispatchReason::Cancelled) => {
                "engine transport: cancelled before dispatch"
            }
            Self::NoDispatch(NoDispatchReason::Deadline) => {
                "engine transport: deadline expired before dispatch"
            }
            Self::NoDispatch(NoDispatchReason::BackendUnavailable) => {
                "engine transport: backend unavailable before dispatch"
            }
            Self::NoDispatch(NoDispatchReason::RequestTooLarge) => {
                "engine transport: request exceeds relay limit before dispatch"
            }
            Self::NoDispatch(NoDispatchReason::ResponseTooLarge) => {
                "engine transport: response exceeds relay limit before dispatch"
            }
            Self::NoDispatch(NoDispatchReason::Io) => {
                "engine transport: relay I/O failed before dispatch"
            }
            Self::Uncertain => "engine transport: result uncertain; request was not retried",
        })
    }
}

impl Error for RelayFailure {}

/// Byte counts only; no request or response data is retained or exposed.
/// Success means both connection directions reached EOF and drained, not that
/// a native HTTP request or asynchronous backend job has completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelayReceipt {
    pub request_bytes: u64,
    pub response_bytes: u64,
}

struct PinnedCoordinator {
    descriptor: CoordinatorIdentityDescriptor,
    pidfd: File,
    network_namespace: File,
}

impl PinnedCoordinator {
    fn from_expected(
        descriptor: CoordinatorIdentityDescriptor,
        network_namespace: File,
    ) -> Result<Self, EngineTransportError> {
        descriptor.validate_shape()?;
        if descriptor.uid != unsafe { libc::geteuid() } {
            return Err(EngineTransportError::InvalidIdentity);
        }
        let handoff =
            CoordinatorIdentityHandoff::from_namespace_file(descriptor, network_namespace)?;
        let pid =
            i32::try_from(descriptor.pid).map_err(|_| EngineTransportError::InvalidIdentity)?;
        let pidfd_raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
        if pidfd_raw < 0 {
            return Err(EngineTransportError::CoordinatorUnavailable);
        }
        let pidfd = unsafe { File::from_raw_fd(pidfd_raw as i32) };
        let identity = Self {
            descriptor,
            pidfd,
            network_namespace: handoff.network_namespace,
        };
        identity.verify_live_process()?;
        identity.verify_namespace_matches_descriptor()?;
        Ok(identity)
    }

    fn verify_live_process(&self) -> Result<(), EngineTransportError> {
        if pidfd_exited(self.pidfd.as_raw_fd())? {
            return Err(EngineTransportError::CoordinatorUnavailable);
        }
        let pid = i32::try_from(self.descriptor.pid)
            .map_err(|_| EngineTransportError::InvalidIdentity)?;
        let actual = process_info(pid)?;
        if actual.uid != self.descriptor.uid
            || actual.start_time_ticks != self.descriptor.start_time_ticks
        {
            return Err(EngineTransportError::CoordinatorUnavailable);
        }
        if pidfd_exited(self.pidfd.as_raw_fd())? {
            return Err(EngineTransportError::CoordinatorUnavailable);
        }
        Ok(())
    }

    fn verify_namespace_matches_descriptor(&self) -> Result<(), EngineTransportError> {
        self.verify_live_process()?;
        let namespace = namespace_file_info(&self.network_namespace)?;
        self.verify_live_process()?;
        if namespace
            != (
                self.descriptor.network_namespace_device,
                self.descriptor.network_namespace_inode,
            )
        {
            return Err(EngineTransportError::CoordinatorUnavailable);
        }
        Ok(())
    }

    fn verify_peer(&self, stream: &UnixStream) -> Result<(), EngineTransportError> {
        self.verify_live_process()?;
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
            || credentials.uid != self.descriptor.uid
            || credentials.pid != self.descriptor.pid as libc::pid_t
        {
            return Err(EngineTransportError::InvalidIdentity);
        }
        self.verify_live_process()?;
        Ok(())
    }
}

struct PinnedDirectory {
    path: PathBuf,
    file: File,
    identity: (u64, u64),
    uid: u32,
}

impl PinnedDirectory {
    fn open(path: &Path) -> Result<Self, EngineTransportError> {
        if !path.is_absolute() {
            return Err(EngineTransportError::UnsafeDirectory);
        }
        let input =
            fs::symlink_metadata(path).map_err(|_| EngineTransportError::UnsafeDirectory)?;
        if !input.is_dir() {
            return Err(EngineTransportError::UnsafeDirectory);
        }
        let canonical = path
            .canonicalize()
            .map_err(|_| EngineTransportError::UnsafeDirectory)?;
        let socket_path = canonical.join(SOCKET_NAME);
        let socket_path_bytes = socket_path.as_os_str().as_bytes();
        let sun_path_bytes = std::mem::size_of::<libc::sockaddr_un>()
            - std::mem::offset_of!(libc::sockaddr_un, sun_path);
        if socket_path_bytes.len() + 1 > sun_path_bytes {
            return Err(EngineTransportError::UnsafeDirectory);
        }
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let file = options
            .open(&canonical)
            .map_err(|_| EngineTransportError::UnsafeDirectory)?;
        let metadata = file
            .metadata()
            .map_err(|_| EngineTransportError::UnsafeDirectory)?;
        let uid = unsafe { libc::geteuid() };
        if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o777 != 0o700 {
            return Err(EngineTransportError::UnsafeDirectory);
        }
        let path_metadata =
            fs::symlink_metadata(&canonical).map_err(|_| EngineTransportError::UnsafeDirectory)?;
        if !path_metadata.is_dir()
            || (path_metadata.dev(), path_metadata.ino()) != (metadata.dev(), metadata.ino())
        {
            return Err(EngineTransportError::UnsafeDirectory);
        }
        Ok(Self {
            path: canonical,
            file,
            identity: (metadata.dev(), metadata.ino()),
            uid,
        })
    }

    fn verify(&self) -> Result<(), EngineTransportError> {
        let fd_metadata = self
            .file
            .metadata()
            .map_err(|_| EngineTransportError::UnsafeDirectory)?;
        let path_metadata =
            fs::symlink_metadata(&self.path).map_err(|_| EngineTransportError::UnsafeDirectory)?;
        if !fd_metadata.is_dir()
            || !path_metadata.is_dir()
            || fd_metadata.uid() != self.uid
            || path_metadata.uid() != self.uid
            || fd_metadata.mode() & 0o777 != 0o700
            || path_metadata.mode() & 0o777 != 0o700
            || (fd_metadata.dev(), fd_metadata.ino()) != self.identity
            || (path_metadata.dev(), path_metadata.ino()) != self.identity
        {
            return Err(EngineTransportError::UnsafeDirectory);
        }
        Ok(())
    }
}

struct SocketPathGuard {
    path: PathBuf,
    directory_identity: (u64, u64),
    socket_identity: (u64, u64),
    uid: u32,
}

impl Drop for SocketPathGuard {
    fn drop(&mut self) {
        let Ok(directory) = fs::symlink_metadata(self.path.parent().unwrap_or(Path::new("")))
        else {
            return;
        };
        if !directory.is_dir()
            || (directory.dev(), directory.ino()) != self.directory_identity
            || directory.uid() != self.uid
            || directory.mode() & 0o777 != 0o700
        {
            return;
        }
        if fs::symlink_metadata(&self.path).is_ok_and(|socket| {
            socket.file_type().is_socket()
                && socket.uid() == self.uid
                && (socket.dev(), socket.ino()) == self.socket_identity
        }) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// A non-cloneable endpoint owned by an engine-side helper process.
pub struct EngineTransport {
    listener: UnixListener,
    directory: PinnedDirectory,
    socket: SocketPathGuard,
    coordinator: PinnedCoordinator,
    backend: BackendPort,
}

impl EngineTransport {
    /// Pins the supplied launch identity and binds a private Unix listener.
    /// The coordinator must already be running in a different network
    /// namespace. No directory or endpoint is created or reclaimed here.
    pub fn from_expected(
        directory: &Path,
        expected_coordinator: CoordinatorIdentityDescriptor,
        coordinator_network_namespace: File,
        backend: BackendPort,
    ) -> Result<Self, EngineTransportError> {
        let coordinator =
            PinnedCoordinator::from_expected(expected_coordinator, coordinator_network_namespace)?;
        let directory = PinnedDirectory::open(directory)?;
        verify_distinct_network_namespaces(&coordinator)?;
        directory.verify()?;

        let path = directory.path.join(SOCKET_NAME);
        if fs::symlink_metadata(&path).is_ok() {
            return Err(EngineTransportError::ExistingSocket);
        }
        let listener =
            UnixListener::bind(&path).map_err(|_| EngineTransportError::SocketUnavailable)?;
        let metadata =
            fs::symlink_metadata(&path).map_err(|_| EngineTransportError::SocketUnavailable)?;
        if !metadata.file_type().is_socket() || metadata.uid() != directory.uid {
            return Err(EngineTransportError::SocketUnavailable);
        }
        let socket = SocketPathGuard {
            path,
            directory_identity: directory.identity,
            socket_identity: (metadata.dev(), metadata.ino()),
            uid: directory.uid,
        };
        fs::set_permissions(&socket.path, fs::Permissions::from_mode(0o600))
            .map_err(|_| EngineTransportError::SocketUnavailable)?;
        if !fs::symlink_metadata(&socket.path).is_ok_and(|entry| {
            entry.file_type().is_socket()
                && entry.uid() == directory.uid
                && entry.mode() & 0o777 == 0o600
                && (entry.dev(), entry.ino()) == socket.socket_identity
        }) {
            return Err(EngineTransportError::SocketUnavailable);
        }
        listener
            .set_nonblocking(true)
            .map_err(|_| EngineTransportError::SocketUnavailable)?;
        directory.verify()?;
        coordinator.verify_live_process()?;
        verify_distinct_network_namespaces(&coordinator)?;
        Ok(Self {
            listener,
            directory,
            socket,
            coordinator,
            backend,
        })
    }

    /// The fixed Unix endpoint used by the captured coordinator.
    pub fn socket_path(&self) -> &Path {
        &self.socket.path
    }

    /// Accepts one coordinator connection and relays opaque bytes exactly
    /// once. The 16 MiB cap applies to each direction of this whole connection;
    /// a receipt is not request/job completion. It never retries a request or
    /// exposes its contents to diagnostics. HTTP keep-alive pools must be
    /// bounded and shut down by their caller, not inferred here.
    pub fn relay_one(&self, context: &OperationContext<'_>) -> Result<RelayReceipt, RelayFailure> {
        let mut coordinator_stream = self.accept_coordinator(context)?;
        if coordinator_stream.set_nonblocking(true).is_err() {
            return Err(RelayFailure::NoDispatch(NoDispatchReason::Io));
        }
        let mut backend = connect_backend(self.backend, &self.coordinator, context)
            .map_err(RelayFailure::NoDispatch)?;
        self.verify_runtime().map_err(|error| match error {
            EngineTransportError::CoordinatorUnavailable => {
                RelayFailure::NoDispatch(NoDispatchReason::CoordinatorUnavailable)
            }
            _ => RelayFailure::NoDispatch(NoDispatchReason::Io),
        })?;
        relay_bytes(
            &mut coordinator_stream,
            &mut backend,
            &self.coordinator,
            &self.directory,
            self.socket.socket_identity,
            context,
        )
    }

    fn accept_coordinator(
        &self,
        context: &OperationContext<'_>,
    ) -> Result<UnixStream, RelayFailure> {
        loop {
            active(context).map_err(RelayFailure::NoDispatch)?;
            self.verify_runtime().map_err(|error| match error {
                EngineTransportError::CoordinatorUnavailable => {
                    RelayFailure::NoDispatch(NoDispatchReason::CoordinatorUnavailable)
                }
                _ => RelayFailure::NoDispatch(NoDispatchReason::Io),
            })?;
            match self.listener.accept() {
                Ok((stream, _)) => {
                    self.coordinator
                        .verify_peer(&stream)
                        .map_err(|error| match error {
                            EngineTransportError::InvalidIdentity => {
                                RelayFailure::NoDispatch(NoDispatchReason::PeerDenied)
                            }
                            _ => RelayFailure::NoDispatch(NoDispatchReason::CoordinatorUnavailable),
                        })?;
                    return Ok(stream);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    poll_until(
                        context,
                        &self.coordinator,
                        &[(&self.listener, libc::POLLIN)],
                    )
                    .map_err(RelayFailure::NoDispatch)?;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => return Err(RelayFailure::NoDispatch(NoDispatchReason::Io)),
            }
        }
    }

    fn verify_runtime(&self) -> Result<(), EngineTransportError> {
        self.directory.verify()?;
        self.coordinator.verify_live_process()?;
        if !socket_matches(&self.directory, self.socket.socket_identity) {
            return Err(EngineTransportError::SocketUnavailable);
        }
        verify_distinct_network_namespaces(&self.coordinator)
    }
}

fn verify_distinct_network_namespaces(
    coordinator: &PinnedCoordinator,
) -> Result<(), EngineTransportError> {
    coordinator.verify_namespace_matches_descriptor()?;
    let engine_namespace = namespace_info(Path::new("/proc/self/ns/net"))?;
    if engine_namespace
        == (
            coordinator.descriptor.network_namespace_device,
            coordinator.descriptor.network_namespace_inode,
        )
    {
        return Err(EngineTransportError::SharedNetworkNamespace);
    }
    coordinator.verify_namespace_matches_descriptor()?;
    Ok(())
}

fn connect_backend(
    backend: BackendPort,
    coordinator: &PinnedCoordinator,
    context: &OperationContext<'_>,
) -> Result<TcpStream, NoDispatchReason> {
    active(context)?;
    coordinator
        .verify_live_process()
        .map_err(|_| NoDispatchReason::CoordinatorUnavailable)?;
    let raw = unsafe {
        libc::socket(
            libc::AF_INET,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    if raw < 0 {
        return Err(NoDispatchReason::BackendUnavailable);
    }
    let stream = unsafe { TcpStream::from_raw_fd(raw) };
    let address = libc::sockaddr_in {
        sin_family: libc::AF_INET as libc::sa_family_t,
        sin_port: backend.get().to_be(),
        sin_addr: libc::in_addr {
            s_addr: u32::from_ne_bytes([127, 0, 0, 1]),
        },
        sin_zero: [0; 8],
    };
    let result = unsafe {
        libc::connect(
            stream.as_raw_fd(),
            (&address as *const libc::sockaddr_in).cast(),
            std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
        )
    };
    if result == 0 {
        return Ok(stream);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() != Some(libc::EINPROGRESS) {
        return Err(NoDispatchReason::BackendUnavailable);
    }
    loop {
        active(context)?;
        coordinator
            .verify_live_process()
            .map_err(|_| NoDispatchReason::CoordinatorUnavailable)?;
        let mut descriptors = [
            libc::pollfd {
                fd: stream.as_raw_fd(),
                events: libc::POLLOUT,
                revents: 0,
            },
            libc::pollfd {
                fd: coordinator.pidfd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        let timeout = timeout_millis(context);
        let ready = unsafe { libc::poll(descriptors.as_mut_ptr(), 2, timeout) };
        if ready < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(NoDispatchReason::BackendUnavailable);
        }
        if descriptors[1].revents != 0 {
            return Err(NoDispatchReason::CoordinatorUnavailable);
        }
        if ready == 0 {
            continue;
        }
        let error = stream
            .take_error()
            .map_err(|_| NoDispatchReason::BackendUnavailable)?;
        return if error.is_none() {
            Ok(stream)
        } else {
            Err(NoDispatchReason::BackendUnavailable)
        };
    }
}

fn relay_bytes(
    coordinator: &mut UnixStream,
    backend: &mut TcpStream,
    identity: &PinnedCoordinator,
    directory: &PinnedDirectory,
    socket_identity: (u64, u64),
    context: &OperationContext<'_>,
) -> Result<RelayReceipt, RelayFailure> {
    let mut request = ByteQueue::new();
    let mut response = ByteQueue::new();
    let mut coordinator_eof = false;
    let mut backend_eof = false;
    let mut backend_write_closed = false;
    let mut coordinator_write_closed = false;
    let mut request_bytes = 0_u64;
    let mut response_bytes = 0_u64;
    let mut dispatched = false;

    loop {
        if let Err(reason) = active(context) {
            return Err(classify_relay_failure(reason, dispatched));
        }
        if identity.verify_live_process().is_err()
            || verify_distinct_network_namespaces(identity).is_err()
            || directory.verify().is_err()
            || !socket_matches(directory, socket_identity)
        {
            return Err(classify_relay_failure(
                NoDispatchReason::CoordinatorUnavailable,
                dispatched,
            ));
        }

        match write_pending(backend, &mut request) {
            Ok(wrote) => dispatched |= wrote,
            Err(reason) => return Err(classify_relay_failure(reason, dispatched)),
        }
        if let Err(reason) = write_pending(coordinator, &mut response) {
            return Err(classify_relay_failure(reason, dispatched));
        }

        if coordinator_eof && request.is_empty() && !backend_write_closed {
            backend
                .shutdown(Shutdown::Write)
                .map_err(|_| classify_relay_failure(NoDispatchReason::Io, dispatched))?;
            backend_write_closed = true;
        }
        if backend_eof && response.is_empty() && !coordinator_write_closed {
            coordinator
                .shutdown(Shutdown::Write)
                .map_err(|_| classify_relay_failure(NoDispatchReason::Io, dispatched))?;
            coordinator_write_closed = true;
        }
        if coordinator_eof && backend_eof && request.is_empty() && response.is_empty() {
            return Ok(RelayReceipt {
                request_bytes,
                response_bytes,
            });
        }

        if !coordinator_eof && request.has_space() {
            match read_bounded(coordinator, &mut request, request_bytes) {
                Ok(Some(0)) => coordinator_eof = true,
                Ok(Some(count)) => request_bytes += count as u64,
                Ok(None) => {}
                Err(reason) => {
                    return Err(classify_relay_failure(reason, dispatched));
                }
            }
        }
        if !backend_eof && response.has_space() {
            match read_bounded(backend, &mut response, response_bytes) {
                Ok(Some(0)) => backend_eof = true,
                Ok(Some(count)) => response_bytes += count as u64,
                Ok(None) => {}
                Err(reason) => {
                    let reason = if reason == NoDispatchReason::RequestTooLarge {
                        NoDispatchReason::ResponseTooLarge
                    } else {
                        reason
                    };
                    return Err(classify_relay_failure(reason, dispatched));
                }
            }
        }

        let coordinator_events = (if !coordinator_eof && request.has_space() {
            libc::POLLIN
        } else {
            0
        }) | (if !response.is_empty() {
            libc::POLLOUT
        } else {
            0
        });
        let backend_events = (if !backend_eof && response.has_space() {
            libc::POLLIN
        } else {
            0
        }) | (if !request.is_empty() {
            libc::POLLOUT
        } else {
            0
        });
        let mut descriptors = [
            libc::pollfd {
                fd: coordinator.as_raw_fd(),
                events: coordinator_events,
                revents: 0,
            },
            libc::pollfd {
                fd: backend.as_raw_fd(),
                events: backend_events,
                revents: 0,
            },
            libc::pollfd {
                fd: identity.pidfd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        let ready = unsafe { libc::poll(descriptors.as_mut_ptr(), 3, timeout_millis(context)) };
        if ready < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(classify_relay_failure(NoDispatchReason::Io, dispatched));
        }
        if descriptors[2].revents != 0 {
            return Err(classify_relay_failure(
                NoDispatchReason::CoordinatorUnavailable,
                dispatched,
            ));
        }
        if descriptors[0].revents & libc::POLLNVAL != 0
            || descriptors[1].revents & libc::POLLNVAL != 0
        {
            return Err(classify_relay_failure(NoDispatchReason::Io, dispatched));
        }
    }
}

fn write_pending<W: Write>(
    writer: &mut W,
    queue: &mut ByteQueue,
) -> Result<bool, NoDispatchReason> {
    if queue.is_empty() {
        return Ok(false);
    }
    match writer.write(queue.front()) {
        Ok(0) => Err(NoDispatchReason::Io),
        Ok(count) => {
            queue.consume(count);
            Ok(count > 0)
        }
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(false),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(false),
        Err(_) => Err(NoDispatchReason::Io),
    }
}

fn read_bounded<R: Read>(
    reader: &mut R,
    queue: &mut ByteQueue,
    already_read: u64,
) -> Result<Option<usize>, NoDispatchReason> {
    let remaining_plus_one = MAX_DIRECTION_BYTES
        .saturating_sub(already_read)
        .saturating_add(1);
    let max_read = queue
        .free_len()
        .min(usize::try_from(remaining_plus_one).unwrap_or(usize::MAX));
    if max_read == 0 {
        return Ok(None);
    }
    match reader.read(queue.space(max_read)) {
        Ok(0) => Ok(Some(0)),
        Ok(count) if already_read + count as u64 > MAX_DIRECTION_BYTES => {
            Err(NoDispatchReason::RequestTooLarge)
        }
        Ok(count) => {
            queue.produced(count);
            Ok(Some(count))
        }
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(None),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(None),
        Err(_) => Err(NoDispatchReason::Io),
    }
}

fn classify_relay_failure(reason: NoDispatchReason, dispatched: bool) -> RelayFailure {
    if dispatched {
        RelayFailure::Uncertain
    } else {
        RelayFailure::NoDispatch(reason)
    }
}

struct ByteQueue {
    bytes: Box<[u8; BUFFER_BYTES]>,
    start: usize,
    end: usize,
}

impl ByteQueue {
    fn new() -> Self {
        Self {
            bytes: Box::new([0; BUFFER_BYTES]),
            start: 0,
            end: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    fn has_space(&mut self) -> bool {
        if self.end == self.bytes.len() && self.start > 0 {
            self.bytes.copy_within(self.start..self.end, 0);
            self.end -= self.start;
            self.start = 0;
        }
        self.end < self.bytes.len()
    }

    fn free_len(&mut self) -> usize {
        if !self.has_space() {
            0
        } else {
            self.bytes.len() - self.end
        }
    }

    fn space(&mut self, max_len: usize) -> &mut [u8] {
        let free = self.free_len();
        &mut self.bytes[self.end..self.end + free.min(max_len)]
    }

    fn produced(&mut self, count: usize) {
        self.end += count;
    }

    fn front(&self) -> &[u8] {
        &self.bytes[self.start..self.end]
    }

    fn consume(&mut self, count: usize) {
        self.start += count;
        if self.start == self.end {
            self.start = 0;
            self.end = 0;
        }
    }
}

fn active(context: &OperationContext<'_>) -> Result<(), NoDispatchReason> {
    if context.is_cancelled() {
        Err(NoDispatchReason::Cancelled)
    } else if context.remaining().is_zero() {
        Err(NoDispatchReason::Deadline)
    } else {
        Ok(())
    }
}

fn timeout_millis(context: &OperationContext<'_>) -> i32 {
    context.remaining().min(POLL_INTERVAL).as_millis().max(1) as i32
}

fn poll_until(
    context: &OperationContext<'_>,
    coordinator: &PinnedCoordinator,
    sockets: &[(&impl AsRawFd, libc::c_short)],
) -> Result<(), NoDispatchReason> {
    active(context)?;
    coordinator
        .verify_live_process()
        .map_err(|_| NoDispatchReason::CoordinatorUnavailable)?;
    let mut descriptors = Vec::with_capacity(sockets.len() + 1);
    for (socket, events) in sockets {
        descriptors.push(libc::pollfd {
            fd: socket.as_raw_fd(),
            events: *events,
            revents: 0,
        });
    }
    descriptors.push(libc::pollfd {
        fd: coordinator.pidfd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    });
    let ready = unsafe {
        libc::poll(
            descriptors.as_mut_ptr(),
            descriptors.len() as libc::nfds_t,
            timeout_millis(context),
        )
    };
    if ready < 0 {
        if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
            return Ok(());
        }
        return Err(NoDispatchReason::Io);
    }
    if descriptors
        .last()
        .is_some_and(|descriptor| descriptor.revents != 0)
    {
        return Err(NoDispatchReason::CoordinatorUnavailable);
    }
    Ok(())
}

fn pidfd_exited(pidfd: i32) -> Result<bool, EngineTransportError> {
    let mut descriptor = libc::pollfd {
        fd: pidfd,
        events: libc::POLLIN,
        revents: 0,
    };
    let result = unsafe { libc::poll(&mut descriptor, 1, 0) };
    if result < 0 {
        return Err(EngineTransportError::CoordinatorUnavailable);
    }
    Ok(result > 0 && descriptor.revents != 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcessInfo {
    uid: u32,
    start_time_ticks: u64,
}

fn process_info(pid: libc::pid_t) -> Result<ProcessInfo, EngineTransportError> {
    let base = PathBuf::from(format!("/proc/{pid}"));
    let stat = fs::read_to_string(base.join("stat"))
        .map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
    let (prefix, suffix) = stat
        .rsplit_once(')')
        .ok_or(EngineTransportError::CoordinatorUnavailable)?;
    let stat_pid = prefix
        .split_whitespace()
        .next()
        .ok_or(EngineTransportError::CoordinatorUnavailable)?
        .parse::<libc::pid_t>()
        .map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
    if stat_pid != pid {
        return Err(EngineTransportError::CoordinatorUnavailable);
    }
    let start_time_ticks = suffix
        .split_whitespace()
        .nth(19)
        .ok_or(EngineTransportError::CoordinatorUnavailable)?
        .parse::<u64>()
        .map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
    let status = fs::read_to_string(base.join("status"))
        .map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
    let uid = status
        .lines()
        .find_map(|line| line.strip_prefix("Uid:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or(EngineTransportError::CoordinatorUnavailable)?
        .parse::<u32>()
        .map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
    Ok(ProcessInfo {
        uid,
        start_time_ticks,
    })
}

fn namespace_info(path: &Path) -> Result<(u64, u64), EngineTransportError> {
    let namespace = File::open(path).map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
    let metadata = namespace
        .metadata()
        .map_err(|_| EngineTransportError::CoordinatorUnavailable)?;
    Ok((metadata.dev(), metadata.ino()))
}

fn namespace_file_info(namespace: &File) -> Result<(u64, u64), EngineTransportError> {
    let metadata = namespace
        .metadata()
        .map_err(|_| EngineTransportError::InvalidIdentity)?;
    let mut filesystem: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatfs(namespace.as_raw_fd(), &mut filesystem) } != 0
        || filesystem.f_type != libc::NSFS_MAGIC
    {
        return Err(EngineTransportError::InvalidIdentity);
    }
    Ok((metadata.dev(), metadata.ino()))
}

fn socket_matches(directory: &PinnedDirectory, expected: (u64, u64)) -> bool {
    directory.verify().is_ok()
        && fs::symlink_metadata(directory.path.join(SOCKET_NAME)).is_ok_and(|socket| {
            socket.file_type().is_socket()
                && socket.uid() == directory.uid
                && socket.mode() & 0o777 == 0o600
                && (socket.dev(), socket.ino()) == expected
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::runtime::NeverCancel;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;
    use std::time::Instant;

    const CHILD_ROLE: &str = "L3MS_ENGINE_TRANSPORT_TEST_CHILD";
    const CHILD_DIRECTORY: &str = "L3MS_ENGINE_TRANSPORT_TEST_DIRECTORY";
    const CHILD_IDENTITY: &str = "L3MS_ENGINE_TRANSPORT_TEST_IDENTITY";
    const CHILD_NAMESPACE_FD: &str = "L3MS_ENGINE_TRANSPORT_TEST_NAMESPACE_FD";
    const CHILD_SCENARIO: &str = "L3MS_ENGINE_TRANSPORT_TEST_SCENARIO";

    fn private_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }

    fn context(milliseconds: u64) -> OperationContext<'static> {
        OperationContext::new(
            Instant::now() + Duration::from_millis(milliseconds),
            &NeverCancel,
        )
    }

    fn descriptor_for_pid(pid: libc::pid_t) -> CoordinatorIdentityDescriptor {
        let process = process_info(pid).unwrap();
        let namespace = namespace_info(&PathBuf::from(format!("/proc/{pid}/ns/net"))).unwrap();
        CoordinatorIdentityDescriptor::from_parts(
            pid as u32,
            process.uid,
            process.start_time_ticks,
            namespace.0,
            namespace.1,
        )
        .unwrap()
    }

    fn namespace_file_for_pid(pid: libc::pid_t) -> File {
        File::open(format!("/proc/{pid}/ns/net")).unwrap()
    }

    fn large_request() -> Vec<u8> {
        let mut request =
            b"POST /api/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 524288\r\n\r\n"
                .to_vec();
        request.extend(vec![b'Q'; 512 * 1024]);
        request
    }

    fn spawn_unshared_helper(
        directory: &Path,
        handoff: &CoordinatorIdentityHandoff,
        scenario: &str,
    ) -> std::process::Child {
        let descriptor = handoff.descriptor();
        let identity_json = serde_json::to_string(&descriptor).unwrap();
        let namespace = handoff.network_namespace_file().try_clone().unwrap();
        let namespace_fd = namespace.as_raw_fd();
        let descriptor_flags = unsafe { libc::fcntl(namespace_fd, libc::F_GETFD) };
        assert!(descriptor_flags >= 0);
        assert_eq!(
            unsafe {
                libc::fcntl(
                    namespace_fd,
                    libc::F_SETFD,
                    descriptor_flags & !libc::FD_CLOEXEC,
                )
            },
            0
        );
        let child_result = Command::new("unshare")
            .args([
                "--user",
                "--map-current-user",
                "--net",
                "--keep-caps",
                "--",
                "sh",
                "-c",
                "ip link set lo up && exec \"$@\"",
                "engine-transport-test",
            ])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "control::engine_transport::tests::real_unshare_helper_entrypoint",
                "--nocapture",
            ])
            .env(CHILD_ROLE, "helper")
            .env(CHILD_DIRECTORY, directory)
            .env(CHILD_IDENTITY, identity_json)
            .env(CHILD_NAMESPACE_FD, namespace_fd.to_string())
            .env(CHILD_SCENARIO, scenario)
            .spawn();
        assert_eq!(
            unsafe { libc::fcntl(namespace_fd, libc::F_SETFD, descriptor_flags) },
            0
        );
        drop(namespace);
        child_result.unwrap()
    }

    fn wait_for_ready(child: &mut std::process::Child, directory: &Path) {
        let ready = directory.join("ready");
        let deadline = Instant::now() + Duration::from_secs(8);
        while !ready.exists() && Instant::now() < deadline {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("isolated helper exited before binding: {status}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(ready.exists(), "isolated helper did not bind in time");
    }

    #[test]
    fn current_identity_is_a_strict_bounded_launch_descriptor() {
        let descriptor = CoordinatorIdentityDescriptor::capture_current().unwrap();
        let encoded = serde_json::to_string(&descriptor).unwrap();
        assert!(encoded.len() <= 256);
        assert_eq!(
            serde_json::from_str::<CoordinatorIdentityDescriptor>(&encoded).unwrap(),
            descriptor
        );
        assert!(serde_json::from_str::<CoordinatorIdentityDescriptor>(
            r#"{"pid":2,"uid":3,"start_time_ticks":4,"network_namespace_device":5,"network_namespace_inode":6,"extra":7}"#
        )
        .is_err());
        assert_eq!(
            format!("{descriptor:?}"),
            "CoordinatorIdentityDescriptor([redacted])"
        );
        assert_eq!(
            BackendPort::new(0),
            Err(EngineTransportError::InvalidBackendPort)
        );
    }

    #[test]
    fn helper_refuses_to_bind_in_the_coordinators_network_namespace() {
        let directory = private_directory();
        let (descriptor, namespace) = CoordinatorIdentityDescriptor::capture_current_handoff()
            .unwrap()
            .into_parts();
        assert!(matches!(
            EngineTransport::from_expected(
                directory.path(),
                descriptor,
                namespace,
                BackendPort::new(12345).unwrap()
            ),
            Err(EngineTransportError::SharedNetworkNamespace)
        ));
        assert!(!directory.path().join(SOCKET_NAME).exists());
    }

    #[test]
    fn pidfd_detects_coordinator_exit_without_trusting_a_reused_pid() {
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        let descriptor = descriptor_for_pid(child.id() as libc::pid_t);
        let namespace = namespace_file_for_pid(child.id() as libc::pid_t);
        let coordinator = PinnedCoordinator::from_expected(descriptor, namespace).unwrap();
        child.kill().unwrap();
        child.wait().unwrap();
        assert_eq!(
            coordinator.verify_live_process(),
            Err(EngineTransportError::CoordinatorUnavailable)
        );
    }

    #[test]
    fn same_uid_child_process_is_not_accepted_as_the_captured_coordinator() {
        let directory = private_directory();
        let path = directory.path().join("peer-test.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let (descriptor, namespace) = CoordinatorIdentityDescriptor::capture_current_handoff()
            .unwrap()
            .into_parts();
        let expected = PinnedCoordinator::from_expected(descriptor, namespace).unwrap();

        let address_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
        address.sun_family = libc::AF_UNIX as libc::sa_family_t;
        for (target, source) in address.sun_path.iter_mut().zip(address_path.as_bytes()) {
            *target = *source as libc::c_char;
        }
        let address_length = (std::mem::offset_of!(libc::sockaddr_un, sun_path)
            + address_path.as_bytes().len()
            + 1) as libc::socklen_t;
        let child_pid = unsafe { libc::fork() };
        assert!(child_pid >= 0);
        if child_pid == 0 {
            let socket =
                unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
            if socket < 0 {
                unsafe { libc::_exit(10) };
            }
            let result = unsafe {
                libc::connect(
                    socket,
                    (&address as *const libc::sockaddr_un).cast(),
                    address_length,
                )
            };
            unsafe {
                libc::close(socket);
                libc::_exit(if result == 0 { 0 } else { 11 });
            }
        }
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(
            expected.verify_peer(&stream),
            Err(EngineTransportError::InvalidIdentity)
        );
        let mut status = 0;
        assert_eq!(
            unsafe { libc::waitpid(child_pid, &mut status, 0) },
            child_pid
        );
        assert!(libc::WIFEXITED(status));
        assert_eq!(libc::WEXITSTATUS(status), 0);
    }

    /// Invoked as a second test process by the real-unshare integration test.
    #[test]
    fn real_unshare_helper_entrypoint() {
        if std::env::var_os(CHILD_ROLE).is_none() {
            return;
        }
        let directory = PathBuf::from(std::env::var_os(CHILD_DIRECTORY).unwrap());
        let descriptor: CoordinatorIdentityDescriptor =
            serde_json::from_str(&std::env::var(CHILD_IDENTITY).unwrap()).unwrap();
        let namespace_fd = std::env::var(CHILD_NAMESPACE_FD)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        let namespace = unsafe { File::from_raw_fd(namespace_fd) };
        let scenario = std::env::var(CHILD_SCENARIO).unwrap();
        let request = large_request();
        let response_header = b"HTTP/1.1 200 OK\r\nContent-Length: 524288\r\n\r\n";
        let mut response = response_header.to_vec();
        response.extend(vec![b'R'; 512 * 1024]);
        let backend = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = backend.local_addr().unwrap().port();
        let expected_request = request.clone();
        let backend_thread = match scenario.as_str() {
            "duplex" => {
                let response = response.clone();
                std::thread::spawn(move || {
                    let (mut stream, _) = backend.accept().unwrap();
                    stream.write_all(b"EARLY").unwrap();
                    let mut received = Vec::new();
                    stream.read_to_end(&mut received).unwrap();
                    assert_eq!(received, expected_request);
                    stream.write_all(&response).unwrap();
                    stream.shutdown(Shutdown::Write).unwrap();
                })
            }
            "partial-write" => std::thread::spawn(move || {
                let (mut stream, _) = backend.accept().unwrap();
                let mut first_byte = [0_u8; 1];
                stream.read_exact(&mut first_byte).unwrap();
            }),
            _ => panic!("unknown test scenario"),
        };
        let transport = EngineTransport::from_expected(
            &directory,
            descriptor,
            namespace,
            BackendPort::new(port).unwrap(),
        )
        .unwrap();
        fs::write(directory.join("ready"), b"ready").unwrap();
        match scenario.as_str() {
            "duplex" => {
                let receipt = transport.relay_one(&context(10_000)).unwrap();
                assert_eq!(receipt.request_bytes, request.len() as u64);
                assert_eq!(receipt.response_bytes, (response.len() + 5) as u64);
                backend_thread.join().unwrap();
            }
            "partial-write" => {
                let outcome = transport.relay_one(&context(1_000));
                assert_eq!(outcome, Err(RelayFailure::Uncertain));
                backend_thread.join().unwrap();
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn real_unshare_isolation_relays_only_through_the_private_unix_peer() {
        let probe = Command::new("unshare")
            .args([
                "--user",
                "--map-current-user",
                "--net",
                "--keep-caps",
                "--",
                "sh",
                "-c",
                "ip link set lo up && exec true",
            ])
            .output();
        let Ok(probe) = probe else {
            eprintln!("SKIP engine transport unshare integration: unshare is unavailable");
            return;
        };
        if !probe.status.success() {
            eprintln!(
                "SKIP engine transport unshare integration: user/network namespaces or loopback setup are unavailable"
            );
            return;
        }

        let directory = private_directory();
        let handoff = CoordinatorIdentityDescriptor::capture_current_handoff().unwrap();
        let mut child = spawn_unshared_helper(directory.path(), &handoff, "duplex");
        wait_for_ready(&mut child, directory.path());

        let mut client = UnixStream::connect(directory.path().join(SOCKET_NAME)).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let request = large_request();
        let response_header = b"HTTP/1.1 200 OK\r\nContent-Length: 524288\r\n\r\n";
        let mut expected_response = response_header.to_vec();
        expected_response.extend(vec![b'R'; 512 * 1024]);
        let split = 31;
        client.write_all(&request[..split]).unwrap();
        let mut early = [0_u8; 5];
        client.read_exact(&mut early).unwrap();
        assert_eq!(&early, b"EARLY");
        client.write_all(&request[split..]).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        // Stall the reader briefly so the 512 KiB backend body exercises the
        // bounded response queue and downstream backpressure.
        std::thread::sleep(Duration::from_millis(50));
        let mut received = Vec::new();
        client.read_to_end(&mut received).unwrap();
        assert_eq!(received, expected_response);
        assert!(child.wait().unwrap().success());

        let partial_directory = private_directory();
        let mut partial =
            spawn_unshared_helper(partial_directory.path(), &handoff, "partial-write");
        wait_for_ready(&mut partial, partial_directory.path());
        let mut client = UnixStream::connect(partial_directory.path().join(SOCKET_NAME)).unwrap();
        client
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let request = large_request();
        // A peer close after the backend accepts the first request byte forces
        // the relay to report uncertainty after dispatch; this must not retry.
        let _ = client.write_all(&request);
        let _ = client.shutdown(Shutdown::Write);
        let mut ignored = Vec::new();
        let _ = client.read_to_end(&mut ignored);
        assert!(partial.wait().unwrap().success());
    }
}
