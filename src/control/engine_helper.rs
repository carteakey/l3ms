//! Single-unit supervisor for a backend inside a systemd-created private
//! network namespace.
//!
//! The owned transient service runs this process with `PrivateNetwork=yes`
//! and a manager-opened coordinator network-namespace descriptor. This
//! supervisor proves the descriptor and live coordinator identity through
//! `EngineTransport` before binding its private Unix listener, then starts the
//! backend directly (never through a shell) in the same unit/cgroup/netns.
//! It supervises the backend during both idle time and relay activity and
//! tears down its process group on exit. `READY=1` reports transport setup
//! only; a lifecycle adapter must perform a backend-native health check before
//! admitting inference. This module does not parse HTTP, retry uncertain
//! requests, drain asynchronous jobs, or prove GPU release.

use std::error::Error;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::cuda_device::CudaDevice;
use super::engine_transport::{
    BackendPort, CoordinatorIdentityDescriptor, EngineTransport, EngineTransportError,
    NoDispatchReason, RelayFailure,
};
use super::managed_service::VerifiedPythonInterpreter;
use super::runtime::{Cancellation, OperationContext};

const ARGUMENT_MARKER: &str = "--l3ms-isolated-engine-supervisor-v1";
const STACK_ROOT_ARGUMENT: &str = "--l3ms-stack-root";
const CUDA_DEVICE_ARGUMENT: &str = "--l3ms-cuda-device";
const VERIFIED_PYTHON_ARGUMENT: &str = "--l3ms-verified-python-v1";
const MAX_ARGUMENTS: usize = 4096;
const MAX_HELPER_ARGUMENTS: usize = 20;
const MAX_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_AGGREGATE_ARGUMENT_BYTES: usize = 256 * 1024;
const MAX_DIRECTORY_BYTES: usize = 4096;
const MAX_PATH_BYTES: usize = 4096;
const LISTEN_FD: i32 = 3;
const LISTEN_FD_NAME: &str = "l3ms-coordinator-netns";
const RELAY_DEADLINE: Duration = Duration::from_secs(24 * 60 * 60);
const CHILD_TERM_GRACE: Duration = Duration::from_secs(2);
const CHILD_KILL_GRACE: Duration = Duration::from_secs(2);
const CHILD_POLL_QUANTUM: Duration = Duration::from_millis(25);

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Fixed, redacted helper failures. No executable, argv, path, request body,
/// manager output, or OS error is retained in this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineHelperError {
    InvalidArguments,
    InvalidActivationEnvironment,
    InvalidNamespaceDescriptor,
    TransportUnavailable,
    BackendSpawnFailed,
    BackendExited,
    NotificationUnavailable,
    RelayUnavailable,
    ChildCleanupUncertain,
}

impl fmt::Display for EngineHelperError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidArguments => "isolated engine helper: invalid launch arguments",
            Self::InvalidActivationEnvironment => {
                "isolated engine helper: invalid manager descriptor handoff"
            }
            Self::InvalidNamespaceDescriptor => "isolated engine helper: namespace proof failed",
            Self::TransportUnavailable => "isolated engine helper: transport setup failed",
            Self::BackendSpawnFailed => "isolated engine helper: backend launch failed",
            Self::BackendExited => "isolated engine helper: backend process exited",
            Self::NotificationUnavailable => {
                "isolated engine helper: manager readiness notification failed"
            }
            Self::RelayUnavailable => "isolated engine helper: byte relay failed",
            Self::ChildCleanupUncertain => {
                "isolated engine helper: backend termination is uncertain"
            }
        })
    }
}

impl Error for EngineHelperError {}

/// Returns whether the first argument requests the private helper entrypoint.
/// The CLI should check this before normal argument parsing and pass the full
/// argument vector, including this marker, to [`run_supervisor`].
pub fn is_helper_entrypoint(first_argument: &OsStr) -> bool {
    first_argument == ARGUMENT_MARKER
}

/// Canonical, bounded root directory explicitly passed to an isolated
/// backend. Its path is launch configuration, not ambient manager state.
pub(crate) struct StackRoot {
    path: PathBuf,
}

impl StackRoot {
    pub(crate) fn new(path: impl AsRef<Path>) -> Result<Self, EngineHelperError> {
        let supplied_path = path.as_ref();
        if !supplied_path.is_absolute()
            || supplied_path.as_os_str().as_bytes().len() > MAX_PATH_BYTES
        {
            return Err(EngineHelperError::InvalidArguments);
        }
        let path = supplied_path
            .canonicalize()
            .map_err(|_| EngineHelperError::InvalidArguments)?;
        if !path.is_absolute() || path.as_os_str().as_bytes().len() > MAX_PATH_BYTES {
            return Err(EngineHelperError::InvalidArguments);
        }
        let metadata = path
            .metadata()
            .map_err(|_| EngineHelperError::InvalidArguments)?;
        if !metadata.is_dir() {
            return Err(EngineHelperError::InvalidArguments);
        }
        Ok(Self { path })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl fmt::Debug for StackRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StackRoot([redacted])")
    }
}

/// Validated backend process details encoded in the unit's bounded argv.
/// Debug deliberately omits all launch data.
pub struct BackendProcessSpec {
    executable: PathBuf,
    working_directory: PathBuf,
    arguments: Vec<OsString>,
    stack_root: Option<StackRoot>,
    cuda_device: Option<CudaDevice>,
    verified_python: Option<VerifiedPythonInterpreter>,
}

impl BackendProcessSpec {
    pub fn new(
        executable: impl AsRef<Path>,
        working_directory: impl AsRef<Path>,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Self, EngineHelperError> {
        Self::build(
            executable.as_ref(),
            working_directory.as_ref(),
            arguments,
            None,
        )
    }

    pub(crate) fn new_with_verified_python(
        interpreter: &VerifiedPythonInterpreter,
        working_directory: impl AsRef<Path>,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Self, EngineHelperError> {
        interpreter
            .verify()
            .map_err(|_| EngineHelperError::InvalidArguments)?;
        Self::build(
            interpreter.configured_path(),
            working_directory.as_ref(),
            arguments,
            Some(interpreter.clone()),
        )
    }

    fn build(
        executable_path: &Path,
        working_path: &Path,
        arguments: impl IntoIterator<Item = OsString>,
        verified_python: Option<VerifiedPythonInterpreter>,
    ) -> Result<Self, EngineHelperError> {
        if !executable_path.is_absolute()
            || !working_path.is_absolute()
            || executable_path.as_os_str().as_bytes().len() > MAX_PATH_BYTES
            || working_path.as_os_str().as_bytes().len() > MAX_PATH_BYTES
        {
            return Err(EngineHelperError::InvalidArguments);
        }
        let executable = if let Some(interpreter) = &verified_python {
            interpreter
                .verify()
                .map_err(|_| EngineHelperError::InvalidArguments)?;
            if executable_path != interpreter.configured_path() {
                return Err(EngineHelperError::InvalidArguments);
            }
            executable_path.to_path_buf()
        } else {
            executable_path
                .canonicalize()
                .map_err(|_| EngineHelperError::InvalidArguments)?
        };
        let metadata = executable
            .metadata()
            .map_err(|_| EngineHelperError::InvalidArguments)?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
            return Err(EngineHelperError::InvalidArguments);
        }
        let working_directory = working_path
            .canonicalize()
            .map_err(|_| EngineHelperError::InvalidArguments)?;
        if !working_directory.is_dir() {
            return Err(EngineHelperError::InvalidArguments);
        }
        let arguments = validate_arguments(arguments.into_iter().collect())?;
        Ok(Self {
            executable,
            working_directory,
            arguments,
            stack_root: None,
            cuda_device: None,
            verified_python,
        })
    }

    pub(crate) fn with_stack_root(mut self, stack_root: StackRoot) -> Self {
        self.stack_root = Some(stack_root);
        self
    }

    /// Explicitly select exactly one full physical GPU UUID for the backend.
    pub fn with_cuda_device(mut self, cuda_device: CudaDevice) -> Self {
        self.cuda_device = Some(cuda_device);
        self
    }
}

impl fmt::Debug for BackendProcessSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BackendProcessSpec([redacted])")
    }
}

/// Build the exact bounded argv for the hidden helper entrypoint. The
/// generated values contain only launch metadata; native request data never
/// enters this protocol. The caller still persists the typed role before
/// dispatching this recipe.
pub fn supervisor_arguments(
    transport_directory: &Path,
    backend_port: BackendPort,
    coordinator: CoordinatorIdentityDescriptor,
    backend: &BackendProcessSpec,
) -> Result<Vec<OsString>, EngineHelperError> {
    let directory = transport_directory;
    if !directory.is_absolute()
        || directory.as_os_str().as_bytes().len() > MAX_DIRECTORY_BYTES
        || directory.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
    {
        return Err(EngineHelperError::InvalidArguments);
    }
    let link_metadata =
        std::fs::symlink_metadata(directory).map_err(|_| EngineHelperError::InvalidArguments)?;
    if !link_metadata.is_dir() {
        return Err(EngineHelperError::InvalidArguments);
    }
    let directory = directory
        .canonicalize()
        .map_err(|_| EngineHelperError::InvalidArguments)?;
    let port = backend_port.get().to_string();
    let fields = [
        coordinator.pid().to_string(),
        coordinator.uid().to_string(),
        coordinator.start_time_ticks().to_string(),
        coordinator.network_namespace_device().to_string(),
        coordinator.network_namespace_inode().to_string(),
    ];
    let executable = backend.executable.as_os_str();
    let working_directory = backend.working_directory.as_os_str();
    let mut args = Vec::with_capacity(15 + backend.arguments.len());
    args.push(OsString::from(ARGUMENT_MARKER));
    args.push(directory.into_os_string());
    args.push(OsString::from(port));
    args.extend(fields.into_iter().map(OsString::from));
    if let Some(stack_root) = &backend.stack_root {
        // Revalidate just before the caller persists and dispatches the
        // prepared service; the helper validates it again before spawning.
        let stack_root = StackRoot::new(stack_root.path())?;
        args.push(OsString::from(STACK_ROOT_ARGUMENT));
        args.push(stack_root.path().as_os_str().to_os_string());
    }
    if let Some(cuda_device) = &backend.cuda_device {
        args.push(OsString::from(CUDA_DEVICE_ARGUMENT));
        args.push(OsString::from(cuda_device.gpu_uuid()));
    }
    if let Some(interpreter) = &backend.verified_python {
        interpreter
            .verify()
            .map_err(|_| EngineHelperError::InvalidArguments)?;
        args.push(OsString::from(VERIFIED_PYTHON_ARGUMENT));
        args.push(interpreter.configured_path().as_os_str().to_os_string());
        args.push(interpreter.target_path().as_os_str().to_os_string());
        args.push(interpreter.venv_directory().as_os_str().to_os_string());
        args.push(OsString::from(interpreter.identity_wire()));
    }
    args.push(executable.to_os_string());
    args.push(working_directory.to_os_string());
    args.push(OsString::from(backend.arguments.len().to_string()));
    args.extend(backend.arguments.iter().cloned());
    validate_arguments(args.clone())?;
    Ok(args)
}

/// Run the hidden helper entrypoint with the full raw argument vector,
/// including the fixed `--l3ms-isolated-engine-supervisor-v1` marker. The CLI
/// must call this before ordinary parsing when [`is_helper_entrypoint`]
/// matches. This returns only fixed error categories and never prints or
/// records backend argv or transport bytes.
pub fn run_supervisor(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<(), EngineHelperError> {
    let config = parse_arguments(arguments.into_iter().collect())?;
    install_shutdown_handler()?;
    let (descriptor, namespace) = take_activation_fd(config.coordinator)?;
    let backend =
        BackendPort::new(config.backend_port).map_err(|_| EngineHelperError::InvalidArguments)?;
    let transport =
        EngineTransport::from_expected(&config.transport_directory, descriptor, namespace, backend)
            .map_err(|error| match error {
                EngineTransportError::InvalidIdentity
                | EngineTransportError::SharedNetworkNamespace => {
                    EngineHelperError::InvalidNamespaceDescriptor
                }
                _ => EngineHelperError::TransportUnavailable,
            })?;

    let mut child = spawn_backend(&config.backend)?;
    let mut monitor = match ChildMonitor::start(child.id()) {
        Ok(monitor) => monitor,
        Err(()) => {
            terminate_child_group_without_monitor(&mut child)
                .map_err(|_| EngineHelperError::ChildCleanupUncertain)?;
            return Err(EngineHelperError::BackendSpawnFailed);
        }
    };
    let child_pid = i32::try_from(child.id()).map_err(|_| EngineHelperError::BackendSpawnFailed)?;
    if monitor.exited.load(Ordering::Acquire)
        || monitor.failed.load(Ordering::Acquire)
        || wait_child_exit(child_pid, &monitor, Duration::ZERO).unwrap_or(true)
    {
        let cleanup = terminate_child_group(&mut child, &monitor);
        monitor.stop_and_join();
        cleanup.map_err(|_| EngineHelperError::ChildCleanupUncertain)?;
        return Err(EngineHelperError::BackendExited);
    }
    if let Err(()) = notify_transport_ready() {
        let cleanup = terminate_child_group(&mut child, &monitor);
        monitor.stop_and_join();
        cleanup.map_err(|_| EngineHelperError::ChildCleanupUncertain)?;
        return Err(EngineHelperError::NotificationUnavailable);
    }

    let cancellation = HelperCancellation {
        child_exited: Arc::clone(&monitor.exited),
        monitor_failed: Arc::clone(&monitor.failed),
    };
    let mut backend_exited = false;
    let mut relay_failure_seen = false;
    loop {
        if SHUTDOWN_REQUESTED.load(Ordering::Acquire) {
            break;
        }
        if monitor.failed.load(Ordering::Acquire) {
            relay_failure_seen = true;
            break;
        }
        if monitor.exited.load(Ordering::Acquire) {
            backend_exited = true;
            break;
        }
        let context = OperationContext::new(Instant::now() + RELAY_DEADLINE, &cancellation);
        match transport.relay_one(&context) {
            Ok(_) => {}
            Err(RelayFailure::NoDispatch(NoDispatchReason::PeerDenied)) => {}
            Err(RelayFailure::NoDispatch(NoDispatchReason::Cancelled))
                if SHUTDOWN_REQUESTED.load(Ordering::Acquire)
                    || monitor.exited.load(Ordering::Acquire)
                    || monitor.failed.load(Ordering::Acquire) =>
            {
                backend_exited = monitor.exited.load(Ordering::Acquire);
                if monitor.failed.load(Ordering::Acquire)
                    && !SHUTDOWN_REQUESTED.load(Ordering::Acquire)
                {
                    relay_failure_seen = true;
                }
                break;
            }
            Err(RelayFailure::NoDispatch(NoDispatchReason::BackendUnavailable)) => {}
            Err(RelayFailure::NoDispatch(
                NoDispatchReason::Deadline
                | NoDispatchReason::RequestTooLarge
                | NoDispatchReason::ResponseTooLarge,
            )) => {}
            Err(RelayFailure::NoDispatch(NoDispatchReason::CoordinatorUnavailable)) => {
                relay_failure_seen = true;
                break;
            }
            Err(RelayFailure::Uncertain) => {
                // The transport never retries. A later, distinct connection
                // remains the coordinator's explicit decision.
                relay_failure_seen = true;
            }
            Err(RelayFailure::NoDispatch(NoDispatchReason::Io | NoDispatchReason::Cancelled)) => {
                relay_failure_seen = true;
                break;
            }
        }
    }

    let cleanup = terminate_child_group(&mut child, &monitor);
    monitor.stop_and_join();
    cleanup.map_err(|_| EngineHelperError::ChildCleanupUncertain)?;
    if backend_exited {
        return Err(EngineHelperError::BackendExited);
    }
    if relay_failure_seen && !SHUTDOWN_REQUESTED.load(Ordering::Acquire) {
        return Err(EngineHelperError::RelayUnavailable);
    }
    Ok(())
}

struct ParsedConfig {
    transport_directory: PathBuf,
    backend_port: u16,
    coordinator: CoordinatorIdentityDescriptor,
    backend: BackendProcessSpec,
}

fn parse_arguments(arguments: Vec<OsString>) -> Result<ParsedConfig, EngineHelperError> {
    validate_arguments(arguments.clone())?;
    let mut cursor = 0usize;
    if take_utf8(&arguments, &mut cursor)? != ARGUMENT_MARKER {
        return Err(EngineHelperError::InvalidArguments);
    }
    let transport_directory = take_path(&arguments, &mut cursor, MAX_DIRECTORY_BYTES)?;
    let backend_port = take_number::<u16>(&arguments, &mut cursor)?;
    let coordinator = CoordinatorIdentityDescriptor::from_parts(
        take_number::<u32>(&arguments, &mut cursor)?,
        take_number::<u32>(&arguments, &mut cursor)?,
        take_number::<u64>(&arguments, &mut cursor)?,
        take_number::<u64>(&arguments, &mut cursor)?,
        take_number::<u64>(&arguments, &mut cursor)?,
    )
    .map_err(|_| EngineHelperError::InvalidArguments)?;
    let stack_root = if arguments
        .get(cursor)
        .is_some_and(|argument| argument == STACK_ROOT_ARGUMENT)
    {
        cursor += 1;
        let stack_root = StackRoot::new(take_path(&arguments, &mut cursor, MAX_PATH_BYTES)?)?;
        if arguments
            .get(cursor)
            .is_some_and(|argument| argument == STACK_ROOT_ARGUMENT)
        {
            return Err(EngineHelperError::InvalidArguments);
        }
        Some(stack_root)
    } else {
        None
    };
    let cuda_device = if arguments
        .get(cursor)
        .is_some_and(|argument| argument == CUDA_DEVICE_ARGUMENT)
    {
        cursor += 1;
        let uuid = take_utf8(&arguments, &mut cursor)?;
        Some(CudaDevice::new(uuid).map_err(|_| EngineHelperError::InvalidArguments)?)
    } else {
        None
    };
    let verified_python = if arguments
        .get(cursor)
        .is_some_and(|argument| argument == VERIFIED_PYTHON_ARGUMENT)
    {
        cursor += 1;
        let configured_path = take_path(&arguments, &mut cursor, MAX_PATH_BYTES)?;
        let target_path = take_path(&arguments, &mut cursor, MAX_PATH_BYTES)?;
        let venv_directory = take_path(&arguments, &mut cursor, MAX_PATH_BYTES)?;
        let identity_wire = take_utf8(&arguments, &mut cursor)?;
        Some(
            VerifiedPythonInterpreter::from_helper_parts(
                configured_path,
                target_path,
                venv_directory,
                identity_wire,
            )
            .map_err(|_| EngineHelperError::InvalidArguments)?,
        )
    } else {
        None
    };
    let executable = take_path(&arguments, &mut cursor, MAX_PATH_BYTES)?;
    let working_directory = take_path(&arguments, &mut cursor, MAX_PATH_BYTES)?;
    let argument_count = take_number::<usize>(&arguments, &mut cursor)?;
    if argument_count > MAX_ARGUMENTS || arguments.len().saturating_sub(cursor) != argument_count {
        return Err(EngineHelperError::InvalidArguments);
    }
    let backend_arguments = arguments.into_iter().skip(cursor).collect::<Vec<_>>();
    let mut backend = if let Some(interpreter) = &verified_python {
        if executable != interpreter.configured_path() {
            return Err(EngineHelperError::InvalidArguments);
        }
        BackendProcessSpec::new_with_verified_python(
            interpreter,
            working_directory,
            backend_arguments,
        )?
    } else {
        BackendProcessSpec::new(executable, working_directory, backend_arguments)?
    };
    if let Some(stack_root) = stack_root {
        backend = backend.with_stack_root(stack_root);
    }
    if let Some(cuda_device) = cuda_device {
        backend = backend.with_cuda_device(cuda_device);
    }
    let directory = transport_directory
        .canonicalize()
        .map_err(|_| EngineHelperError::InvalidArguments)?;
    if !directory.is_dir() {
        return Err(EngineHelperError::InvalidArguments);
    }
    Ok(ParsedConfig {
        transport_directory: directory,
        backend_port,
        coordinator,
        backend,
    })
}

fn take_activation_fd(
    descriptor: CoordinatorIdentityDescriptor,
) -> Result<(CoordinatorIdentityDescriptor, File), EngineHelperError> {
    let pid =
        std::env::var("LISTEN_PID").map_err(|_| EngineHelperError::InvalidActivationEnvironment)?;
    let fds =
        std::env::var("LISTEN_FDS").map_err(|_| EngineHelperError::InvalidActivationEnvironment)?;
    let names = std::env::var("LISTEN_FDNAMES")
        .map_err(|_| EngineHelperError::InvalidActivationEnvironment)?;
    validate_activation_values(&pid, &fds, &names, std::process::id())?;
    let flags = unsafe { libc::fcntl(LISTEN_FD, libc::F_GETFD) };
    if flags < 0 || unsafe { libc::fcntl(LISTEN_FD, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
        return Err(EngineHelperError::InvalidActivationEnvironment);
    }
    let namespace = unsafe { File::from_raw_fd(LISTEN_FD) };
    super::engine_transport::CoordinatorIdentityHandoff::from_namespace_file(descriptor, namespace)
        .map_err(|_| EngineHelperError::InvalidNamespaceDescriptor)
        .map(|handoff| handoff.into_parts())
}

fn validate_activation_values(
    listen_pid: &str,
    listen_fds: &str,
    listen_names: &str,
    current_pid: u32,
) -> Result<(), EngineHelperError> {
    if listen_pid.parse::<u32>().ok() != Some(current_pid)
        || listen_fds != "1"
        || listen_names != LISTEN_FD_NAME
    {
        return Err(EngineHelperError::InvalidActivationEnvironment);
    }
    Ok(())
}

fn spawn_backend(backend: &BackendProcessSpec) -> Result<Child, EngineHelperError> {
    if let Some(interpreter) = &backend.verified_python {
        interpreter
            .verify()
            .map_err(|_| EngineHelperError::InvalidArguments)?;
        if backend.executable != interpreter.configured_path() {
            return Err(EngineHelperError::InvalidArguments);
        }
    }
    let expected_parent_pid = unsafe { libc::getpid() };
    let mut command = Command::new(&backend.executable);
    if backend.cuda_device.is_some() {
        // An explicit physical selection is intended to be the backend's
        // complete CUDA visibility policy, not an addition to inherited
        // ordinal/list/MIG settings or unrelated ambient launch options.
        command.env_clear();
    }
    command
        .args(&backend.arguments)
        .current_dir(&backend.working_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env_remove("LISTEN_PID")
        .env_remove("LISTEN_FDS")
        .env_remove("LISTEN_FDNAMES")
        .env_remove("NOTIFY_SOCKET");
    if backend.verified_python.is_some() {
        // A validated venv must determine its own prefix and site packages;
        // ambient Python path overrides apply to no other backend launch.
        command.env_remove("PYTHONHOME").env_remove("PYTHONPATH");
    }
    if let Some(stack_root) = &backend.stack_root {
        command.env("L3MS_ROOT", stack_root.path());
    }
    if let Some(cuda_device) = &backend.cuda_device {
        cuda_device.apply_to(&mut command);
    }
    // The child is placed in a dedicated process group and gets a parent-death
    // signal as a second containment layer in addition to systemd's cgroup.
    unsafe {
        command.process_group(0).pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::getppid() != expected_parent_pid {
                libc::kill(libc::getpid(), libc::SIGKILL);
                return Err(io::Error::from_raw_os_error(libc::ECHILD));
            }
            Ok(())
        });
    }
    command
        .spawn()
        .map_err(|_| EngineHelperError::BackendSpawnFailed)
}

fn install_shutdown_handler() -> Result<(), EngineHelperError> {
    SHUTDOWN_REQUESTED.store(false, Ordering::Release);
    let previous = unsafe {
        libc::signal(
            libc::SIGTERM,
            shutdown_signal_handler as *const () as libc::sighandler_t,
        )
    };
    if previous == libc::SIG_ERR {
        return Err(EngineHelperError::TransportUnavailable);
    }
    Ok(())
}

extern "C" fn shutdown_signal_handler(_: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::Release);
}

struct HelperCancellation {
    child_exited: Arc<AtomicBool>,
    monitor_failed: Arc<AtomicBool>,
}

impl Cancellation for HelperCancellation {
    fn is_cancelled(&self) -> bool {
        SHUTDOWN_REQUESTED.load(Ordering::Acquire)
            || self.child_exited.load(Ordering::Acquire)
            || self.monitor_failed.load(Ordering::Acquire)
    }
}

struct ChildMonitor {
    pidfd: File,
    exited: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl ChildMonitor {
    fn start(pid: u32) -> Result<Self, ()> {
        let pid = i32::try_from(pid).map_err(|_| ())?;
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
        if fd < 0 {
            return Err(());
        }
        let pidfd = unsafe { File::from_raw_fd(fd as i32) };
        let thread_pidfd = pidfd.try_clone().map_err(|_| ())?;
        let exited = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_exited = Arc::clone(&exited);
        let thread_failed = Arc::clone(&failed);
        let thread_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("l3ms-engine-child-monitor".to_owned())
            .spawn(move || {
                while !thread_stop.load(Ordering::Acquire) {
                    let mut pollfd = libc::pollfd {
                        fd: thread_pidfd.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    let result = unsafe {
                        libc::poll(
                            &mut pollfd,
                            1,
                            CHILD_POLL_QUANTUM.as_millis() as libc::c_int,
                        )
                    };
                    if result > 0 && pollfd.revents & libc::POLLIN != 0 {
                        thread_exited.store(true, Ordering::Release);
                        break;
                    }
                    if result > 0 && pollfd.revents & (libc::POLLHUP | libc::POLLERR) != 0 {
                        thread_failed.store(true, Ordering::Release);
                        break;
                    }
                    if result < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted
                    {
                        thread_failed.store(true, Ordering::Release);
                        break;
                    }
                }
            })
            .map_err(|_| ())?;
        Ok(Self {
            pidfd,
            exited,
            failed,
            stop,
            thread: Some(thread),
        })
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    fn wait_exited(&self, duration: Duration) -> Result<bool, ()> {
        let deadline = Instant::now() + duration;
        loop {
            let mut pollfd = libc::pollfd {
                fd: self.pidfd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            let timeout = remaining
                .as_millis()
                .min(i32::MAX as u128)
                .try_into()
                .unwrap_or(i32::MAX);
            let result = unsafe { libc::poll(&mut pollfd, 1, timeout) };
            if result > 0 {
                if pollfd.revents & libc::POLLIN != 0 {
                    return Ok(true);
                }
                if pollfd.revents & (libc::POLLHUP | libc::POLLERR) != 0 {
                    return Err(());
                }
                return Err(());
            }
            if result == 0 {
                return Ok(false);
            }
            if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                return Err(());
            }
        }
    }
}

impl Drop for ChildMonitor {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn terminate_child_group(child: &mut Child, monitor: &ChildMonitor) -> Result<(), ()> {
    let group = i32::try_from(child.id()).map_err(|_| ())?;
    signal_process_group(group, libc::SIGTERM)?;
    let mut child_exited = wait_child_exit(group, monitor, CHILD_TERM_GRACE)?;
    let kill_deadline = Instant::now() + CHILD_KILL_GRACE;
    while !child_exited && Instant::now() < kill_deadline {
        // The direct child is intentionally still unreaped, pinning this
        // numeric PGID to the exact process group created for this child.
        signal_process_group(group, libc::SIGKILL)?;
        child_exited = wait_child_exit(group, monitor, CHILD_POLL_QUANTUM)?;
    }
    if !child_exited {
        signal_process_group(group, libc::SIGKILL)?;
        return Err(());
    }

    // Final group signal while the leader remains unreaped. Never signal this
    // numeric PGID after wait(), when the number could have been reused.
    signal_process_group(group, libc::SIGKILL)?;
    child.wait().map_err(|_| ())?;
    wait_process_group_empty(group, CHILD_KILL_GRACE).and_then(
        |empty| {
            if empty {
                Ok(())
            } else {
                Err(())
            }
        },
    )
}

fn terminate_child_group_without_monitor(child: &mut Child) -> Result<(), ()> {
    let group = i32::try_from(child.id()).map_err(|_| ())?;
    signal_process_group(group, libc::SIGTERM)?;
    let mut child_exited = waitid_child_exited(group, CHILD_TERM_GRACE)?;
    let kill_deadline = Instant::now() + CHILD_KILL_GRACE;
    while !child_exited && Instant::now() < kill_deadline {
        signal_process_group(group, libc::SIGKILL)?;
        child_exited = waitid_child_exited(group, CHILD_POLL_QUANTUM)?;
    }
    if !child_exited {
        signal_process_group(group, libc::SIGKILL)?;
        return Err(());
    }
    signal_process_group(group, libc::SIGKILL)?;
    child.wait().map_err(|_| ())?;
    wait_process_group_empty(group, CHILD_KILL_GRACE).and_then(
        |empty| {
            if empty {
                Ok(())
            } else {
                Err(())
            }
        },
    )
}

fn signal_process_group(process_group: i32, signal: i32) -> Result<(), ()> {
    let result = unsafe { libc::kill(-process_group, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(())
    }
}

fn wait_process_group_empty(process_group: i32, duration: Duration) -> Result<bool, ()> {
    let deadline = Instant::now() + duration;
    loop {
        let result = unsafe { libc::kill(-process_group, 0) };
        if result != 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                return Ok(true);
            }
            if error.raw_os_error() != Some(libc::EPERM) {
                return Err(());
            }
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(CHILD_POLL_QUANTUM);
    }
}

fn waitid_child_exited(pid: i32, duration: Duration) -> Result<bool, ()> {
    let deadline = Instant::now() + duration;
    loop {
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                pid as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result != 0 {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(());
            }
        } else if unsafe { info.si_pid() } != 0 {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(CHILD_POLL_QUANTUM);
    }
}

/// Uses pidfd readiness to wake promptly, but only `waitid(..., WNOWAIT)` can
/// certify that the still-owned child has exited. This keeps its PID/PGID
/// reserved until the final process-group signal and bounded cleanup finish.
fn wait_child_exit(pid: i32, monitor: &ChildMonitor, duration: Duration) -> Result<bool, ()> {
    let deadline = Instant::now() + duration;
    loop {
        if waitid_child_exited(pid, Duration::ZERO)? {
            return Ok(true);
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        let quantum = deadline
            .saturating_duration_since(now)
            .min(CHILD_POLL_QUANTUM);
        // A monitor error does not imply process exit. Keep the bounded wait
        // moving and retry the authoritative WNOWAIT observation.
        if monitor.wait_exited(quantum).is_err() {
            thread::sleep(quantum);
        }
    }
}

fn notify_transport_ready() -> Result<(), ()> {
    let address = std::env::var_os("NOTIFY_SOCKET").ok_or(())?;
    let bytes = address.as_os_str().as_bytes();
    if bytes.is_empty() || bytes.contains(&0) {
        return Err(());
    }
    let abstract_name = bytes[0] == b'@';
    let path_bytes = if abstract_name { &bytes[1..] } else { bytes };
    let max = std::mem::size_of::<libc::sockaddr_un>()
        - std::mem::offset_of!(libc::sockaddr_un, sun_path);
    if path_bytes.is_empty() || path_bytes.len() + usize::from(!abstract_name) > max {
        return Err(());
    }
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(());
    }
    let socket = unsafe { File::from_raw_fd(fd) };
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let path_ptr = address.sun_path.as_mut_ptr().cast::<u8>();
    if abstract_name {
        unsafe {
            std::ptr::copy_nonoverlapping(path_bytes.as_ptr(), path_ptr.add(1), path_bytes.len());
        }
    } else {
        unsafe {
            std::ptr::copy_nonoverlapping(path_bytes.as_ptr(), path_ptr, path_bytes.len());
        }
    }
    let address_len = (std::mem::offset_of!(libc::sockaddr_un, sun_path) + path_bytes.len() + 1)
        as libc::socklen_t;
    let message = b"READY=1\nSTATUS=transport ready; backend health not verified";
    let result = unsafe {
        libc::sendto(
            socket.as_raw_fd(),
            message.as_ptr().cast(),
            message.len(),
            libc::MSG_NOSIGNAL,
            (&address as *const libc::sockaddr_un).cast(),
            address_len,
        )
    };
    if result == message.len() as isize {
        Ok(())
    } else {
        Err(())
    }
}

fn validate_arguments(arguments: Vec<OsString>) -> Result<Vec<OsString>, EngineHelperError> {
    if arguments.len() > MAX_ARGUMENTS + MAX_HELPER_ARGUMENTS {
        return Err(EngineHelperError::InvalidArguments);
    }
    let mut total = 0usize;
    for argument in &arguments {
        let bytes = argument.as_os_str().as_bytes();
        if bytes.len() > MAX_ARGUMENT_BYTES || bytes.contains(&0) {
            return Err(EngineHelperError::InvalidArguments);
        }
        total = total
            .checked_add(bytes.len())
            .ok_or(EngineHelperError::InvalidArguments)?;
    }
    if total > MAX_AGGREGATE_ARGUMENT_BYTES {
        return Err(EngineHelperError::InvalidArguments);
    }
    Ok(arguments)
}

fn take_utf8<'a>(
    arguments: &'a [OsString],
    cursor: &mut usize,
) -> Result<&'a str, EngineHelperError> {
    let value = arguments
        .get(*cursor)
        .ok_or(EngineHelperError::InvalidArguments)?;
    *cursor += 1;
    value.to_str().ok_or(EngineHelperError::InvalidArguments)
}

fn take_number<T: std::str::FromStr>(
    arguments: &[OsString],
    cursor: &mut usize,
) -> Result<T, EngineHelperError> {
    take_utf8(arguments, cursor)?
        .parse()
        .map_err(|_| EngineHelperError::InvalidArguments)
}

fn take_path(
    arguments: &[OsString],
    cursor: &mut usize,
    maximum: usize,
) -> Result<PathBuf, EngineHelperError> {
    let value = arguments
        .get(*cursor)
        .ok_or(EngineHelperError::InvalidArguments)?;
    *cursor += 1;
    let path = PathBuf::from(value);
    if !path.is_absolute() || path.as_os_str().as_bytes().len() > maximum {
        return Err(EngineHelperError::InvalidArguments);
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn private_fixture_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "l3ms-engine-helper-test-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    #[test]
    fn helper_errors_are_fixed_and_debug_redacted() {
        assert_eq!(
            EngineHelperError::BackendSpawnFailed.to_string(),
            "isolated engine helper: backend launch failed"
        );
        assert_eq!(
            format!("{:?}", EngineHelperError::BackendSpawnFailed),
            "BackendSpawnFailed"
        );
    }

    #[test]
    fn argument_validation_rejects_nul_and_oversize_values() {
        assert_eq!(
            validate_arguments(vec![OsString::from_vec(b"bad\0arg".to_vec())]),
            Err(EngineHelperError::InvalidArguments)
        );
        assert_eq!(
            validate_arguments(vec![OsString::from("x".repeat(MAX_ARGUMENT_BYTES + 1))]),
            Err(EngineHelperError::InvalidArguments)
        );
    }

    #[test]
    fn argument_parser_rejects_wrong_marker_and_extra_fields() {
        assert_eq!(
            parse_arguments(vec![OsString::from("wrong")]).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
        assert_eq!(
            parse_arguments(vec![
                OsString::from(ARGUMENT_MARKER),
                OsString::from("extra")
            ])
            .map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
    }

    #[test]
    fn supervisor_roundtrip_allows_a_backend_without_arguments_but_not_empty_helper_config() {
        let directory = private_fixture_dir();
        let backend = BackendProcessSpec::new(
            std::env::current_exe().unwrap(),
            std::env::current_dir().unwrap(),
            [],
        )
        .unwrap();
        let arguments = supervisor_arguments(
            &directory,
            BackendPort::new(8188).unwrap(),
            CoordinatorIdentityDescriptor::capture_current().unwrap(),
            &backend,
        )
        .unwrap();
        assert!(!arguments
            .iter()
            .any(|argument| argument == STACK_ROOT_ARGUMENT));
        assert!(!arguments
            .iter()
            .any(|argument| argument == CUDA_DEVICE_ARGUMENT));
        assert!(parse_arguments(arguments)
            .unwrap()
            .backend
            .arguments
            .is_empty());
        assert_eq!(
            parse_arguments(Vec::new()).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn supervisor_argument_roundtrip_preserves_non_utf8_backend_arguments() {
        let directory = private_fixture_dir();
        let backend = BackendProcessSpec::new(
            std::env::current_exe().unwrap(),
            std::env::current_dir().unwrap(),
            [OsString::from_vec(vec![b'k', 0xff, b'y'])],
        )
        .unwrap();
        let descriptor = CoordinatorIdentityDescriptor::capture_current().unwrap();
        let arguments = supervisor_arguments(
            &directory,
            BackendPort::new(8188).unwrap(),
            descriptor,
            &backend,
        )
        .unwrap();
        let parsed = parse_arguments(arguments).unwrap();
        assert_eq!(parsed.backend.arguments, backend.arguments);
        assert_eq!(parsed.coordinator, descriptor);
        assert_eq!(parsed.backend_port, 8188);
        assert_eq!(
            parsed.transport_directory,
            directory.canonicalize().unwrap()
        );
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn verified_python_roundtrip_executes_the_configured_venv_interpreter() {
        let Some(target) = [
            Path::new("/usr/bin/python3"),
            Path::new("/usr/local/bin/python3"),
        ]
        .into_iter()
        .find_map(|path| {
            path.canonicalize().ok().filter(|target| {
                target.metadata().is_ok_and(|metadata| {
                    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
                })
            })
        }) else {
            return;
        };

        let directory = private_fixture_dir().canonicalize().unwrap();
        let venv_directory = directory.join(".venv-comfy");
        let bin_directory = venv_directory.join("bin");
        std::fs::create_dir_all(&bin_directory).unwrap();
        std::os::unix::fs::symlink(&target, bin_directory.join("python")).unwrap();
        std::fs::write(
            venv_directory.join("pyvenv.cfg"),
            format!(
                "home = {}\ninclude-system-site-packages = false\n",
                target.parent().unwrap().display()
            ),
        )
        .unwrap();
        let interpreter =
            VerifiedPythonInterpreter::capture(bin_directory.join("python"), &venv_directory)
                .unwrap();
        let observed_prefix = directory.join("observed-prefix");
        let backend = BackendProcessSpec::new_with_verified_python(
            &interpreter,
            std::env::current_dir().unwrap(),
            [
                OsString::from("-c"),
                OsString::from(
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.prefix)",
                ),
                observed_prefix.as_os_str().to_os_string(),
            ],
        )
        .unwrap();
        assert_eq!(backend.executable, interpreter.configured_path());

        let arguments = supervisor_arguments(
            &directory,
            BackendPort::new(8188).unwrap(),
            CoordinatorIdentityDescriptor::capture_current().unwrap(),
            &backend,
        )
        .unwrap();
        let parsed = parse_arguments(arguments).unwrap();
        assert_eq!(parsed.backend.executable, interpreter.configured_path());
        assert!(parsed.backend.verified_python.is_some());
        let mut child = spawn_backend(&parsed.backend).unwrap();
        assert!(child.wait().unwrap().success());
        assert_eq!(
            std::fs::read_to_string(&observed_prefix).unwrap(),
            venv_directory.to_string_lossy().to_string()
        );

        std::fs::remove_file(bin_directory.join("python")).unwrap();
        std::os::unix::fs::symlink("/bin/false", bin_directory.join("python")).unwrap();
        assert_eq!(
            interpreter.verify(),
            Err(super::super::managed_service::ManagedServiceErrorKind::InvalidRecipe)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn stack_root_is_canonical_existing_bounded_directory_and_redacted() {
        let directory = private_fixture_dir();
        let root = StackRoot::new(&directory).unwrap();
        assert_eq!(root.path(), directory.canonicalize().unwrap());
        assert_eq!(format!("{root:?}"), "StackRoot([redacted])");
        assert_eq!(
            StackRoot::new("relative").map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
        let too_long = PathBuf::from(format!("/{}", "x".repeat(MAX_PATH_BYTES)));
        assert_eq!(
            StackRoot::new(too_long).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
        let file = directory.join("not-a-directory");
        std::fs::write(&file, b"fixture").unwrap();
        assert_eq!(
            StackRoot::new(&file).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn optional_stack_root_flag_roundtrips_and_rejects_duplicates() {
        let directory = private_fixture_dir();
        let stack_root_path = directory.join("stack-root");
        std::fs::create_dir(&stack_root_path).unwrap();
        let backend = BackendProcessSpec::new(
            std::env::current_exe().unwrap(),
            std::env::current_dir().unwrap(),
            [],
        )
        .unwrap()
        .with_stack_root(StackRoot::new(&stack_root_path).unwrap());
        assert!(!format!("{backend:?}").contains(stack_root_path.to_str().unwrap()));
        let arguments = supervisor_arguments(
            &directory,
            BackendPort::new(8188).unwrap(),
            CoordinatorIdentityDescriptor::capture_current().unwrap(),
            &backend,
        )
        .unwrap();
        let parsed = parse_arguments(arguments.clone()).unwrap();
        let canonical_stack_root = stack_root_path.canonicalize().unwrap();
        assert_eq!(
            parsed.backend.stack_root.as_ref().map(StackRoot::path),
            Some(canonical_stack_root.as_path())
        );

        let flag_index = arguments
            .iter()
            .position(|argument| argument == STACK_ROOT_ARGUMENT)
            .unwrap();
        let mut duplicate = arguments.clone();
        duplicate.insert(flag_index + 2, OsString::from(STACK_ROOT_ARGUMENT));
        assert_eq!(
            parse_arguments(duplicate).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );

        let mut missing_root = arguments;
        missing_root.remove(flag_index + 1);
        assert_eq!(
            parse_arguments(missing_root).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn optional_cuda_device_flag_roundtrips_strictly_after_other_launch_metadata() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let directory = private_fixture_dir();
        let stack_root_path = directory.join("stack-root");
        std::fs::create_dir(&stack_root_path).unwrap();
        let backend = BackendProcessSpec::new(
            std::env::current_exe().unwrap(),
            std::env::current_dir().unwrap(),
            [],
        )
        .unwrap()
        .with_stack_root(StackRoot::new(&stack_root_path).unwrap())
        .with_cuda_device(CudaDevice::new(UUID).unwrap());
        let arguments = supervisor_arguments(
            &directory,
            BackendPort::new(8188).unwrap(),
            CoordinatorIdentityDescriptor::capture_current().unwrap(),
            &backend,
        )
        .unwrap();
        let parsed = parse_arguments(arguments.clone()).unwrap();
        assert_eq!(
            parsed
                .backend
                .cuda_device
                .as_ref()
                .map(CudaDevice::gpu_uuid),
            Some(UUID)
        );

        let device_index = arguments
            .iter()
            .position(|argument| argument == CUDA_DEVICE_ARGUMENT)
            .unwrap();
        assert_eq!(
            arguments[device_index + 1].to_str(),
            Some(UUID),
            "the UUID follows its typed marker"
        );
        let stack_index = arguments
            .iter()
            .position(|argument| argument == STACK_ROOT_ARGUMENT)
            .unwrap();
        assert!(stack_index < device_index);

        let mut duplicate = arguments.clone();
        duplicate.splice(
            device_index + 2..device_index + 2,
            [OsString::from(CUDA_DEVICE_ARGUMENT), OsString::from(UUID)],
        );
        assert_eq!(
            parse_arguments(duplicate).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );

        let mut missing_uuid = arguments.clone();
        missing_uuid.remove(device_index + 1);
        assert_eq!(
            parse_arguments(missing_uuid).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );

        let mut invalid_uuid = arguments.clone();
        invalid_uuid[device_index + 1] = OsString::from("0");
        assert_eq!(
            parse_arguments(invalid_uuid).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );

        let mut out_of_order = arguments;
        let cuda_pair = out_of_order
            .drain(device_index..device_index + 2)
            .collect::<Vec<_>>();
        let stack_index = out_of_order
            .iter()
            .position(|argument| argument == STACK_ROOT_ARGUMENT)
            .unwrap();
        out_of_order.splice(stack_index..stack_index, cuda_pair);
        assert_eq!(
            parse_arguments(out_of_order).map(|_| ()),
            Err(EngineHelperError::InvalidArguments)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn backend_environment_probe() {
        if std::env::var_os("L3MS_STACK_ROOT_PROBE").as_deref() != Some(OsStr::new("1")) {
            return;
        }
        let stack_root = StackRoot::new(PathBuf::from(
            std::env::var_os("L3MS_STACK_ROOT_PROBE_ROOT").expect("probe root supplied"),
        ))
        .unwrap();
        let backend = BackendProcessSpec::new(
            "/bin/sh",
            std::env::current_dir().unwrap(),
            [
                OsString::from("-c"),
                OsString::from(r#"printf '%s' "$L3MS_ROOT" > "$L3MS_ROOT/observed""#),
            ],
        )
        .unwrap()
        .with_stack_root(stack_root);
        let mut child = spawn_backend(&backend).unwrap();
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn backend_cuda_environment_probe() {
        if std::env::var_os("L3MS_CUDA_PROBE").as_deref() != Some(OsStr::new("1")) {
            return;
        }
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let stack_root = StackRoot::new(PathBuf::from(
            std::env::var_os("L3MS_CUDA_PROBE_ROOT").expect("probe root supplied"),
        ))
        .unwrap();
        let device =
            CudaDevice::new(&std::env::var("L3MS_CUDA_PROBE_UUID").expect("probe UUID supplied"))
                .unwrap();
        assert_eq!(device.gpu_uuid(), UUID);
        let backend = BackendProcessSpec::new(
            "/bin/sh",
            std::env::current_dir().unwrap(),
            [
                OsString::from("-c"),
                OsString::from(
                    r#"printf '%s\n%s\n%s\n' "$CUDA_VISIBLE_DEVICES" "$L3MS_ROOT" "${L3MS_AMBIENT_SENTINEL-unset}" > "$L3MS_ROOT/observed""#,
                ),
            ],
        )
        .unwrap()
        .with_stack_root(stack_root)
        .with_cuda_device(device);
        let mut child = spawn_backend(&backend).unwrap();
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn selected_physical_uuid_replaces_ambient_child_cuda_environment() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let directory = private_fixture_dir();
        let explicit_root = directory.join("explicit");
        let ambient_root = directory.join("ambient");
        std::fs::create_dir(&explicit_root).unwrap();
        std::fs::create_dir(&ambient_root).unwrap();
        let executable = std::env::current_exe().unwrap();
        let listing = Command::new(&executable).arg("--list").output().unwrap();
        assert!(listing.status.success());
        let test_listing = String::from_utf8(listing.stdout).unwrap();
        let test_name = test_listing
            .lines()
            .find_map(|line| {
                line.strip_suffix(": test")
                    .filter(|name| name.ends_with("::backend_cuda_environment_probe"))
            })
            .expect("backend CUDA environment probe test is listed");
        let status = Command::new(executable)
            .arg("--exact")
            .arg(test_name)
            .arg("--nocapture")
            .env("CUDA_VISIBLE_DEVICES", "ambient,selection")
            .env("L3MS_ROOT", &ambient_root)
            .env("L3MS_AMBIENT_SENTINEL", "must-not-be-forwarded")
            .env("L3MS_CUDA_PROBE", "1")
            .env("L3MS_CUDA_PROBE_ROOT", &explicit_root)
            .env("L3MS_CUDA_PROBE_UUID", UUID)
            .status()
            .unwrap();
        assert!(status.success());
        let observed = std::fs::read_to_string(explicit_root.join("observed")).unwrap();
        let mut lines = observed.lines();
        assert_eq!(lines.next(), Some(UUID));
        assert_eq!(
            lines.next(),
            Some(explicit_root.canonicalize().unwrap().to_str().unwrap())
        );
        assert_eq!(lines.next(), Some("unset"));
        assert!(lines.next().is_none());
        assert!(!ambient_root.join("observed").exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn backend_child_gets_explicit_l3ms_root_over_ambient_value() {
        let directory = private_fixture_dir();
        let explicit_root = directory.join("explicit");
        let ambient_root = directory.join("ambient");
        std::fs::create_dir(&explicit_root).unwrap();
        std::fs::create_dir(&ambient_root).unwrap();
        let executable = std::env::current_exe().unwrap();
        let listing = Command::new(&executable).arg("--list").output().unwrap();
        assert!(listing.status.success());
        let test_listing = String::from_utf8(listing.stdout).unwrap();
        let test_name = test_listing
            .lines()
            .find_map(|line| {
                line.strip_suffix(": test")
                    .filter(|name| name.ends_with("::backend_environment_probe"))
            })
            .expect("backend environment probe test is listed");
        let status = Command::new(executable)
            .arg("--exact")
            .arg(test_name)
            .arg("--nocapture")
            .env("L3MS_ROOT", &ambient_root)
            .env("L3MS_STACK_ROOT_PROBE", "1")
            .env("L3MS_STACK_ROOT_PROBE_ROOT", &explicit_root)
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(
            std::fs::read_to_string(explicit_root.join("observed")).unwrap(),
            explicit_root
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        );
        assert!(!ambient_root.join("observed").exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn manager_descriptor_handoff_requires_exact_pid_count_and_name() {
        let pid = std::process::id();
        assert!(validate_activation_values(&pid.to_string(), "1", LISTEN_FD_NAME, pid).is_ok());
        assert_eq!(
            validate_activation_values(&pid.to_string(), "2", LISTEN_FD_NAME, pid),
            Err(EngineHelperError::InvalidActivationEnvironment)
        );
        assert_eq!(
            validate_activation_values(&pid.to_string(), "1", "wrong-name", pid),
            Err(EngineHelperError::InvalidActivationEnvironment)
        );
        assert_eq!(
            validate_activation_values("1", "1", LISTEN_FD_NAME, pid),
            Err(EngineHelperError::InvalidActivationEnvironment)
        );
    }

    fn spawn_test_child(script: &str) -> Child {
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(script)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command.process_group(0);
        command.spawn().unwrap()
    }

    fn terminate_test_child(child: &mut Child, monitor: &mut ChildMonitor) {
        let started = Instant::now();
        let result = terminate_child_group(child, monitor);
        monitor.stop_and_join();
        assert!(result.is_ok(), "bounded child-group cleanup failed");
        assert!(started.elapsed() < CHILD_TERM_GRACE + CHILD_KILL_GRACE + Duration::from_secs(2));
    }

    #[test]
    fn cleanup_terminates_live_child_group() {
        let mut child = spawn_test_child("exec sleep 30");
        let mut monitor = ChildMonitor::start(child.id()).unwrap();
        terminate_test_child(&mut child, &mut monitor);
    }

    #[test]
    fn cleanup_kills_term_resistant_child_group_within_bound() {
        let mut child = spawn_test_child("trap '' TERM; exec sleep 30");
        let mut monitor = ChildMonitor::start(child.id()).unwrap();
        terminate_test_child(&mut child, &mut monitor);
    }

    #[test]
    fn exited_leader_stays_unreaped_until_lingering_group_is_signalled() {
        let mut child = spawn_test_child("sleep 30 & exit 0");
        let mut monitor = ChildMonitor::start(child.id()).unwrap();
        let leader = i32::try_from(child.id()).unwrap();
        assert!(waitid_child_exited(leader, CHILD_TERM_GRACE).unwrap());
        // WNOWAIT must leave the direct child as an unreaped zombie while the
        // final process-group signal reaches its still-running descendant.
        assert!(terminate_child_group(&mut child, &monitor).is_ok());
        monitor.stop_and_join();
    }

    #[test]
    fn monitor_failure_never_certifies_exit_or_skips_authoritative_cleanup() {
        let mut child = spawn_test_child("exec sleep 30");
        let mut monitor = ChildMonitor::start(child.id()).unwrap();
        let leader = i32::try_from(child.id()).unwrap();
        monitor.failed.store(true, Ordering::Release);
        assert!(!wait_child_exit(leader, &monitor, Duration::ZERO).unwrap());
        assert!(!waitid_child_exited(leader, Duration::ZERO).unwrap());
        terminate_test_child(&mut child, &mut monitor);
    }
}
