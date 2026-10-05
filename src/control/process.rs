//! Bounded execution of one caller-supplied native child process.
//!
//! Each child starts in a new process group owned by this invocation. Parent
//! Pipe IO is nonblocking. Input and separate stdout/stderr capture share hard
//! byte limits, and waits use the caller's `OperationContext`. Cleanup after
//! cancellation, timeout, output overflow, or leader exit uses one finite
//! `RecoveryPolicy` budget. Errors and `Debug` output are deliberately
//! redacted because command arguments and captured streams may contain prompts
//! or credentials.
//!
//! The containment boundary is the dedicated process group. A child that
//! deliberately calls `setsid`, changes process groups, or otherwise
//! daemonizes can escape that boundary; this module does not claim arbitrary
//! process-tree containment and callers must trust the supplied executable.
//!
//! Cleaning up this local process group does not prove that a remote Comfy job
//! or GPU lease was released. Callers must verify backend-specific remote
//! ownership separately.

use std::error::Error;
use std::fmt;
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

use super::runtime::{OperationContext, RecoveryPolicy};

/// Hard ceiling for any stdin payload accepted by this module.
pub const MAX_STDIN_BYTES: usize = 16 * 1024 * 1024;
/// Hard ceiling for the combined byte count captured across stdout and stderr.
pub const MAX_CAPTURE_BYTES: usize = 16 * 1024 * 1024;

const POLL_QUANTUM: Duration = Duration::from_millis(25);
const POST_EXIT_DRAIN: Duration = Duration::from_millis(200);
const IO_CHUNK_BYTES: usize = 16 * 1024;

// Parallel fixture writers must not overlap this runner's fork/exec window:
// a fork can inherit another thread's writable script descriptor until exec,
// making an otherwise closed fixture executable fail with ETXTBSY.
#[cfg(test)]
pub(crate) fn executable_fixture_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The strongest conclusion this executor can make after an attempted run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessFailureDisposition {
    /// Validation, cancellation, setup, or spawn failed before a child existed.
    NoChildSpawned,
    /// The fresh child group was signalled as needed, the leader was reaped,
    /// and its output pipes reached EOF and the group disappeared within the
    /// cleanup budget.
    /// This does not account for descendants that deliberately escaped the
    /// dedicated process group.
    ChildGroupReaped,
    /// Cleanup could not establish that the child group was contained.
    OwnershipUncertain,
}

/// A safe, non-secret description of why a child operation failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessFailureKind {
    UnsupportedPlatform,
    InvalidLimits,
    StdinTooLarge { limit: usize },
    Cancelled,
    DeadlineExceeded,
    PipeSetupFailed,
    SpawnFailed,
    IoFailed,
    PollFailed,
    WaitFailed,
    InvalidPipeDescriptor,
    PipeReadFailed,
    InputWriteFailed,
    OutputLimitExceeded,
    CleanupUnconfirmed,
    ChildExitedUnsuccessfully { code: Option<i32> },
}

/// A process failure that never retains a command, OS error string, or output.
pub struct ProcessFailure {
    disposition: ProcessFailureDisposition,
    kind: ProcessFailureKind,
}

impl ProcessFailure {
    fn new(disposition: ProcessFailureDisposition, kind: ProcessFailureKind) -> Self {
        Self { disposition, kind }
    }

    pub fn disposition(&self) -> ProcessFailureDisposition {
        self.disposition
    }

    pub fn kind(&self) -> ProcessFailureKind {
        self.kind
    }
}

impl fmt::Debug for ProcessFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessFailure")
            .field("disposition", &self.disposition)
            .field("kind", &self.kind)
            .finish()
    }
}

impl fmt::Display for ProcessFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ProcessFailureKind::UnsupportedPlatform => {
                formatter.write_str("bounded child execution is supported only on Linux")?
            }
            ProcessFailureKind::InvalidLimits => {
                formatter.write_str("process limits exceed the supported bounds")?
            }
            ProcessFailureKind::StdinTooLarge { limit } => {
                write!(formatter, "stdin payload exceeds the {limit}-byte limit")?
            }
            ProcessFailureKind::Cancelled => {
                formatter.write_str("child execution was cancelled")?
            }
            ProcessFailureKind::DeadlineExceeded => {
                formatter.write_str("child execution deadline expired")?
            }
            ProcessFailureKind::PipeSetupFailed => {
                formatter.write_str("child process pipes could not be prepared")?
            }
            ProcessFailureKind::SpawnFailed => {
                formatter.write_str("child process could not be started")?
            }
            ProcessFailureKind::IoFailed
            | ProcessFailureKind::PollFailed
            | ProcessFailureKind::WaitFailed
            | ProcessFailureKind::InvalidPipeDescriptor
            | ProcessFailureKind::PipeReadFailed
            | ProcessFailureKind::InputWriteFailed => {
                formatter.write_str("child process I/O failed")?
            }
            ProcessFailureKind::OutputLimitExceeded => {
                formatter.write_str("combined child output exceeded its configured limit")?
            }
            ProcessFailureKind::CleanupUnconfirmed => formatter.write_str(
                "child process group cleanup was not confirmed within the recovery budget",
            )?,
            ProcessFailureKind::ChildExitedUnsuccessfully { code: Some(code) } => {
                write!(formatter, "child process exited with status code {code}")?
            }
            ProcessFailureKind::ChildExitedUnsuccessfully { code: None } => {
                formatter.write_str("child process terminated unsuccessfully")?
            }
        }
        formatter.write_str(match self.disposition {
            ProcessFailureDisposition::NoChildSpawned => "; disposition: no child spawned",
            ProcessFailureDisposition::ChildGroupReaped => "; disposition: child group reaped",
            ProcessFailureDisposition::OwnershipUncertain => "; disposition: ownership uncertain",
        })
    }
}

impl Error for ProcessFailure {}

/// Validated byte limits for one execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessLimits {
    max_stdin_bytes: usize,
    max_output_bytes: usize,
}

impl ProcessLimits {
    /// Build limits no larger than this module's hard ceilings.
    pub fn new(max_stdin_bytes: usize, max_output_bytes: usize) -> Result<Self, ProcessFailure> {
        if max_stdin_bytes > MAX_STDIN_BYTES || max_output_bytes > MAX_CAPTURE_BYTES {
            return Err(ProcessFailure::new(
                ProcessFailureDisposition::NoChildSpawned,
                ProcessFailureKind::InvalidLimits,
            ));
        }
        Ok(Self {
            max_stdin_bytes,
            max_output_bytes,
        })
    }

    pub fn max_stdin_bytes(self) -> usize {
        self.max_stdin_bytes
    }

    pub fn max_output_bytes(self) -> usize {
        self.max_output_bytes
    }
}

impl Default for ProcessLimits {
    fn default() -> Self {
        Self {
            max_stdin_bytes: MAX_STDIN_BYTES,
            max_output_bytes: MAX_CAPTURE_BYTES,
        }
    }
}

/// Successful local process completion and its bounded, combined output.
pub struct ProcessOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: ExitStatus,
}

impl ProcessOutput {
    /// Return captured stdout bytes. Treat these bytes as sensitive data.
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    /// Return captured stderr bytes. Treat these bytes as sensitive data.
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }

    pub fn status(&self) -> ExitStatus {
        self.status
    }
}

impl fmt::Debug for ProcessOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessOutput")
            .field("stdout", &"<redacted>")
            .field("stderr", &"<redacted>")
            .field("status", &self.status)
            .finish()
    }
}

/// Execute `command` in a new process group with bounded optional stdin and
/// separate stdout and stderr capture sharing one output-byte limit.
///
/// The executor checks `context` before setup and immediately before spawn,
/// then polls nonblocking parent pipes while checking cancellation and the
/// deadline. On cancellation, timeout, output overflow, or incomplete leader
/// exit it sends `SIGKILL` only to the process group created for this child,
/// reaps the leader, drains both output pipes, and probes for group absence
/// under one independent `recovery_policy` deadline. Nonzero status is
/// returned as a redacted typed failure; captured output is available only on
/// success.
#[cfg(target_os = "linux")]
pub fn execute(
    command: Command,
    stdin_payload: Option<&[u8]>,
    limits: ProcessLimits,
    context: &OperationContext<'_>,
    recovery_policy: RecoveryPolicy,
) -> Result<ProcessOutput, ProcessFailure> {
    execute_linux(command, stdin_payload, limits, context, recovery_policy)
}

/// Fail closed on platforms where process-group setup and cleanup have not
/// been implemented.
#[cfg(not(target_os = "linux"))]
pub fn execute(
    _command: Command,
    _stdin_payload: Option<&[u8]>,
    _limits: ProcessLimits,
    _context: &OperationContext<'_>,
    _recovery_policy: RecoveryPolicy,
) -> Result<ProcessOutput, ProcessFailure> {
    Err(ProcessFailure::new(
        ProcessFailureDisposition::NoChildSpawned,
        ProcessFailureKind::UnsupportedPlatform,
    ))
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::fs::File;
    use std::io::{self, Read, Write};
    use std::os::fd::{FromRawFd, RawFd};
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;

    struct Capture {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        limit: usize,
        total: usize,
        overflowed: bool,
    }

    impl Capture {
        fn new(limit: usize) -> Self {
            Self {
                stdout: Vec::with_capacity(limit.min(32 * 1024)),
                stderr: Vec::with_capacity(limit.min(32 * 1024)),
                limit,
                total: 0,
                overflowed: false,
            }
        }

        fn record(&mut self, stream: OutputStream, bytes: &[u8]) {
            if self.overflowed {
                return;
            }
            if bytes.len() > self.limit.saturating_sub(self.total) {
                self.overflowed = true;
                return;
            }
            match stream {
                OutputStream::Stdout => self.stdout.extend_from_slice(bytes),
                OutputStream::Stderr => self.stderr.extend_from_slice(bytes),
            }
            self.total += bytes.len();
        }
    }

    #[derive(Clone, Copy)]
    enum OutputStream {
        Stdout,
        Stderr,
    }

    struct OutputPipes {
        stdout: Option<File>,
        stderr: Option<File>,
    }

    impl OutputPipes {
        fn reached_eof(&self) -> bool {
            self.stdout.is_none() && self.stderr.is_none()
        }
    }

    struct CleanupReport {
        disposition: ProcessFailureDisposition,
        status: Option<ExitStatus>,
    }

    enum Readiness {
        Ready {
            stdout: i16,
            stderr: i16,
            input: i16,
        },
    }

    pub(super) fn execute_linux(
        mut command: Command,
        stdin_payload: Option<&[u8]>,
        limits: ProcessLimits,
        context: &OperationContext<'_>,
        recovery_policy: RecoveryPolicy,
    ) -> Result<ProcessOutput, ProcessFailure> {
        if let Some(kind) = inactive_reason(context) {
            return Err(ProcessFailure::new(
                ProcessFailureDisposition::NoChildSpawned,
                kind,
            ));
        }

        if let Some(payload) = stdin_payload {
            if payload.len() > limits.max_stdin_bytes {
                return Err(ProcessFailure::new(
                    ProcessFailureDisposition::NoChildSpawned,
                    ProcessFailureKind::StdinTooLarge {
                        limit: limits.max_stdin_bytes,
                    },
                ));
            }
        }

        let (stdout_reader, stdout_writer) = output_pipe().map_err(|_| {
            ProcessFailure::new(
                ProcessFailureDisposition::NoChildSpawned,
                ProcessFailureKind::PipeSetupFailed,
            )
        })?;
        let (stderr_reader, stderr_writer) = output_pipe().map_err(|_| {
            ProcessFailure::new(
                ProcessFailureDisposition::NoChildSpawned,
                ProcessFailureKind::PipeSetupFailed,
            )
        })?;

        let input = match stdin_payload {
            Some(_) => Some(stdin_pipe().map_err(|_| {
                ProcessFailure::new(
                    ProcessFailureDisposition::NoChildSpawned,
                    ProcessFailureKind::PipeSetupFailed,
                )
            })?),
            None => None,
        };

        let (mut input_writer, input_reader) = match input {
            Some((writer, reader)) => (Some(writer), Some(reader)),
            None => (None, None),
        };

        command
            .stdin(input_reader.map_or_else(Stdio::null, Stdio::from))
            .stdout(Stdio::from(stdout_writer))
            .stderr(Stdio::from(stderr_writer));
        command.process_group(0);

        // This final check closes the cancellation race during pipe setup.
        if let Some(kind) = inactive_reason(context) {
            return Err(ProcessFailure::new(
                ProcessFailureDisposition::NoChildSpawned,
                kind,
            ));
        }

        #[cfg(test)]
        let fixture_guard = executable_fixture_guard();
        let spawned = command.spawn();
        #[cfg(test)]
        drop(fixture_guard);
        let mut child = spawned.map_err(|_| {
            ProcessFailure::new(
                ProcessFailureDisposition::NoChildSpawned,
                ProcessFailureKind::SpawnFailed,
            )
        })?;
        // Drop the parent's copies of child-side descriptors immediately so
        // EOF accurately indicates that process-group writers are gone.
        drop(command);

        let child_pid = child.id() as libc::pid_t;
        let mut output_pipes = OutputPipes {
            stdout: Some(stdout_reader),
            stderr: Some(stderr_reader),
        };
        if stdin_payload.is_none_or(|payload| payload.is_empty()) {
            input_writer.take();
        }

        let mut capture = Capture::new(limits.max_output_bytes);
        let mut input_offset = 0usize;
        let mut leader_exit_at = None;

        loop {
            if let Some(kind) = inactive_reason(context) {
                return Err(fail_after_spawn(
                    &mut child,
                    child_pid,
                    &mut output_pipes,
                    &mut input_writer,
                    &mut capture,
                    recovery_policy,
                    kind,
                ));
            }

            match child_has_exited(child_pid) {
                Ok(true) => {
                    leader_exit_at.get_or_insert_with(Instant::now);
                }
                Ok(false) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    return Err(fail_after_spawn(
                        &mut child,
                        child_pid,
                        &mut output_pipes,
                        &mut input_writer,
                        &mut capture,
                        recovery_policy,
                        ProcessFailureKind::WaitFailed,
                    ));
                }
            }

            if let Some(exited_at) = leader_exit_at {
                if output_pipes.reached_eof()
                    || Instant::now().saturating_duration_since(exited_at) >= POST_EXIT_DRAIN
                {
                    if input_offset < stdin_payload.map_or(0, <[u8]>::len) {
                        return Err(fail_after_spawn(
                            &mut child,
                            child_pid,
                            &mut output_pipes,
                            &mut input_writer,
                            &mut capture,
                            recovery_policy,
                            ProcessFailureKind::InputWriteFailed,
                        ));
                    }
                    return finish_after_exit(
                        &mut child,
                        child_pid,
                        &mut output_pipes,
                        &mut input_writer,
                        &mut capture,
                        recovery_policy,
                    );
                }
            }

            let timeout = main_poll_timeout(context, leader_exit_at);
            let readiness = match poll_io(&output_pipes, input_writer.as_ref(), timeout) {
                Ok(readiness) => readiness,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    return Err(fail_after_spawn(
                        &mut child,
                        child_pid,
                        &mut output_pipes,
                        &mut input_writer,
                        &mut capture,
                        recovery_policy,
                        ProcessFailureKind::PollFailed,
                    ));
                }
            };

            let Readiness::Ready {
                stdout,
                stderr,
                input,
            } = readiness;
            if (stdout | stderr | input) & libc::POLLNVAL != 0 {
                return Err(fail_after_spawn(
                    &mut child,
                    child_pid,
                    &mut output_pipes,
                    &mut input_writer,
                    &mut capture,
                    recovery_policy,
                    ProcessFailureKind::InvalidPipeDescriptor,
                ));
            }

            if output_pipes.stdout.is_some()
                && stdout & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0
            {
                match drain_output(&mut output_pipes.stdout, &mut capture, OutputStream::Stdout) {
                    Ok(true) => {
                        output_pipes.stdout.take();
                    }
                    Ok(false) => {}
                    Err(_) => {
                        return Err(fail_after_spawn(
                            &mut child,
                            child_pid,
                            &mut output_pipes,
                            &mut input_writer,
                            &mut capture,
                            recovery_policy,
                            ProcessFailureKind::PipeReadFailed,
                        ));
                    }
                }
                if capture.overflowed {
                    return Err(fail_after_spawn(
                        &mut child,
                        child_pid,
                        &mut output_pipes,
                        &mut input_writer,
                        &mut capture,
                        recovery_policy,
                        ProcessFailureKind::OutputLimitExceeded,
                    ));
                }
            }

            if output_pipes.stderr.is_some()
                && stderr & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0
            {
                match drain_output(&mut output_pipes.stderr, &mut capture, OutputStream::Stderr) {
                    Ok(true) => {
                        output_pipes.stderr.take();
                    }
                    Ok(false) => {}
                    Err(_) => {
                        return Err(fail_after_spawn(
                            &mut child,
                            child_pid,
                            &mut output_pipes,
                            &mut input_writer,
                            &mut capture,
                            recovery_policy,
                            ProcessFailureKind::PipeReadFailed,
                        ));
                    }
                }
                if capture.overflowed {
                    return Err(fail_after_spawn(
                        &mut child,
                        child_pid,
                        &mut output_pipes,
                        &mut input_writer,
                        &mut capture,
                        recovery_policy,
                        ProcessFailureKind::OutputLimitExceeded,
                    ));
                }
            }

            if input_writer.is_some()
                && input & (libc::POLLOUT | libc::POLLHUP | libc::POLLERR) != 0
            {
                let payload = stdin_payload.unwrap_or_default();
                match send_input(&mut input_writer, payload, &mut input_offset) {
                    Ok(InputWrite::Progress) => {
                        if input_offset >= payload.len() {
                            input_writer.take();
                        }
                    }
                    Ok(InputWrite::Closed) => {
                        input_writer.take();
                    }
                    Err(_) => {
                        return Err(fail_after_spawn(
                            &mut child,
                            child_pid,
                            &mut output_pipes,
                            &mut input_writer,
                            &mut capture,
                            recovery_policy,
                            ProcessFailureKind::InputWriteFailed,
                        ));
                    }
                }
            }
        }
    }

    fn inactive_reason(context: &OperationContext<'_>) -> Option<ProcessFailureKind> {
        if context.is_cancelled() {
            Some(ProcessFailureKind::Cancelled)
        } else if context.remaining().is_zero() {
            Some(ProcessFailureKind::DeadlineExceeded)
        } else {
            None
        }
    }

    fn output_pipe() -> io::Result<(File, File)> {
        let mut descriptors = [-1; 2];
        let result = unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        let reader = unsafe { File::from_raw_fd(descriptors[0]) };
        let writer = unsafe { File::from_raw_fd(descriptors[1]) };
        set_nonblocking(reader.as_raw_fd())?;
        Ok((reader, writer))
    }

    fn stdin_pipe() -> io::Result<(File, File)> {
        let mut descriptors = [-1; 2];
        let result = unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        let reader = unsafe { File::from_raw_fd(descriptors[0]) };
        let writer = unsafe { File::from_raw_fd(descriptors[1]) };
        set_nonblocking(writer.as_raw_fd())?;
        Ok((writer, reader))
    }

    fn set_nonblocking(fd: RawFd) -> io::Result<()> {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    enum InputWrite {
        Progress,
        Closed,
    }

    fn send_input(
        writer: &mut Option<File>,
        payload: &[u8],
        offset: &mut usize,
    ) -> io::Result<InputWrite> {
        let Some(writer) = writer.as_mut() else {
            return Ok(InputWrite::Closed);
        };
        if *offset >= payload.len() {
            return Ok(InputWrite::Closed);
        }
        let end = (*offset + IO_CHUNK_BYTES).min(payload.len());
        let slice = &payload[*offset..end];
        match writer.write(slice) {
            Ok(written) => {
                *offset += written;
                Ok(InputWrite::Progress)
            }
            Err(error) => match error.kind() {
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted => Ok(InputWrite::Progress),
                _ => Err(error),
            },
        }
    }

    fn drain_output(
        reader: &mut Option<File>,
        capture: &mut Capture,
        stream: OutputStream,
    ) -> io::Result<bool> {
        let Some(reader) = reader.as_mut() else {
            return Ok(true);
        };
        let mut buffer = [0u8; IO_CHUNK_BYTES];
        // Bound each turn so a continuously writing child cannot prevent the
        // executor from checking its deadline or initiating cleanup.
        for _ in 0..4 {
            match reader.read(&mut buffer) {
                Ok(0) => return Ok(true),
                Ok(count) => {
                    capture.record(stream, &buffer[..count]);
                    if capture.overflowed {
                        return Ok(false);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        Ok(false)
    }

    fn poll_io(
        output_pipes: &OutputPipes,
        input_writer: Option<&File>,
        timeout: Duration,
    ) -> io::Result<Readiness> {
        let mut descriptors = [
            libc::pollfd {
                fd: output_pipes.stdout.as_ref().map_or(-1, AsRawFd::as_raw_fd),
                events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
                revents: 0,
            },
            libc::pollfd {
                fd: output_pipes.stderr.as_ref().map_or(-1, AsRawFd::as_raw_fd),
                events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
                revents: 0,
            },
            libc::pollfd {
                fd: input_writer.map_or(-1, AsRawFd::as_raw_fd),
                events: libc::POLLOUT | libc::POLLHUP | libc::POLLERR,
                revents: 0,
            },
        ];
        let timeout_ms = duration_ceil_millis(timeout);
        let result = unsafe { libc::poll(descriptors.as_mut_ptr(), 3, timeout_ms) };
        if result < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Readiness::Ready {
            stdout: descriptors[0].revents,
            stderr: descriptors[1].revents,
            input: descriptors[2].revents,
        })
    }

    fn child_has_exited(pid: libc::pid_t) -> io::Result<bool> {
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                pid as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result == 0 {
            return Ok(unsafe { info.si_pid() } != 0);
        }
        Err(io::Error::last_os_error())
    }

    fn signal_child_group(pid: libc::pid_t) -> bool {
        let result = unsafe { libc::kill(-pid, libc::SIGKILL) };
        if result == 0 {
            return true;
        }
        io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }

    fn fail_after_spawn(
        child: &mut std::process::Child,
        child_pid: libc::pid_t,
        output_pipes: &mut OutputPipes,
        input_writer: &mut Option<File>,
        capture: &mut Capture,
        recovery_policy: RecoveryPolicy,
        kind: ProcessFailureKind,
    ) -> ProcessFailure {
        let report = cleanup_group(
            child,
            child_pid,
            output_pipes,
            input_writer,
            capture,
            recovery_policy,
        );
        ProcessFailure::new(report.disposition, kind)
    }

    fn finish_after_exit(
        child: &mut std::process::Child,
        child_pid: libc::pid_t,
        output_pipes: &mut OutputPipes,
        input_writer: &mut Option<File>,
        capture: &mut Capture,
        recovery_policy: RecoveryPolicy,
    ) -> Result<ProcessOutput, ProcessFailure> {
        let report = cleanup_group(
            child,
            child_pid,
            output_pipes,
            input_writer,
            capture,
            recovery_policy,
        );
        if report.disposition != ProcessFailureDisposition::ChildGroupReaped {
            return Err(ProcessFailure::new(
                ProcessFailureDisposition::OwnershipUncertain,
                ProcessFailureKind::CleanupUnconfirmed,
            ));
        }
        if capture.overflowed {
            return Err(ProcessFailure::new(
                report.disposition,
                ProcessFailureKind::OutputLimitExceeded,
            ));
        }
        let status = report.status.expect("reaped child has an exit status");
        if !status.success() {
            return Err(ProcessFailure::new(
                report.disposition,
                ProcessFailureKind::ChildExitedUnsuccessfully {
                    code: status.code(),
                },
            ));
        }
        Ok(ProcessOutput {
            stdout: std::mem::take(&mut capture.stdout),
            stderr: std::mem::take(&mut capture.stderr),
            status,
        })
    }

    fn cleanup_group(
        child: &mut std::process::Child,
        child_pid: libc::pid_t,
        output_pipes: &mut OutputPipes,
        input_writer: &mut Option<File>,
        capture: &mut Capture,
        recovery_policy: RecoveryPolicy,
    ) -> CleanupReport {
        input_writer.take();
        let deadline = match Instant::now().checked_add(recovery_policy.timeout()) {
            Some(deadline) => deadline,
            None => {
                return CleanupReport {
                    disposition: ProcessFailureDisposition::OwnershipUncertain,
                    status: None,
                };
            }
        };

        // `waitid(..., WNOWAIT)` keeps the leader's PID reserved until the
        // final group signal. That prevents its numeric process-group ID from
        // being reused before we finish signaling this owned group.
        let mut group_signal_ok = signal_child_group(child_pid);
        let mut pipe_io_ok = true;
        let mut leader_exited = false;
        let mut next_signal = Instant::now() + POLL_QUANTUM;

        while Instant::now() < deadline {
            if drain_pipes(output_pipes, capture).is_err() {
                pipe_io_ok = false;
            }

            match child_has_exited(child_pid) {
                Ok(true) => {
                    leader_exited = true;
                    break;
                }
                Ok(false) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    pipe_io_ok = false;
                    break;
                }
            }

            if Instant::now() >= next_signal {
                group_signal_ok |= signal_child_group(child_pid);
                next_signal = Instant::now() + POLL_QUANTUM;
            }

            let timeout = bounded_remaining(deadline, POLL_QUANTUM);
            match poll_io(output_pipes, None, timeout) {
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    pipe_io_ok = false;
                    // Keep the cleanup clock moving without a blocking wait.
                    std::thread::sleep(bounded_remaining(deadline, POLL_QUANTUM));
                }
            }
        }

        if !leader_exited {
            output_pipes.stdout.take();
            output_pipes.stderr.take();
            return CleanupReport {
                disposition: ProcessFailureDisposition::OwnershipUncertain,
                status: None,
            };
        }

        // The leader remains an unreaped zombie here, so this final signal
        // still addresses only its original, invocation-owned process group.
        group_signal_ok |= signal_child_group(child_pid);
        let status = child.wait().ok();

        // Do not signal after `wait()` releases the leader PID: the numeric
        // process-group ID could then be reused. A zero-signal probe is safe
        // after reaping and lets us wait for every remaining group member,
        // including members that closed their inherited output descriptors.
        let mut group_absent = false;
        let mut probe_reliable = true;
        while Instant::now() < deadline && (!group_absent || !output_pipes.reached_eof()) {
            if drain_pipes(output_pipes, capture).is_err() {
                pipe_io_ok = false;
            }
            match probe_child_group_absent(child_pid) {
                Ok(true) => group_absent = true,
                Ok(false) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    probe_reliable = false;
                    break;
                }
            }
            if group_absent && output_pipes.reached_eof() {
                break;
            }
            let timeout = bounded_remaining(deadline, POLL_QUANTUM);
            match poll_io(output_pipes, None, timeout) {
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    pipe_io_ok = false;
                    std::thread::sleep(bounded_remaining(deadline, POLL_QUANTUM));
                }
            }
        }

        let disposition = if group_signal_ok
            && pipe_io_ok
            && probe_reliable
            && group_absent
            && status.is_some()
            && output_pipes.reached_eof()
        {
            ProcessFailureDisposition::ChildGroupReaped
        } else {
            output_pipes.stdout.take();
            output_pipes.stderr.take();
            ProcessFailureDisposition::OwnershipUncertain
        };
        CleanupReport {
            disposition,
            status,
        }
    }

    fn drain_one(
        reader: &mut Option<File>,
        capture: &mut Capture,
        stream: OutputStream,
    ) -> io::Result<()> {
        if reader.is_some() && drain_output(reader, capture, stream)? {
            reader.take();
        }
        Ok(())
    }

    fn drain_pipes(pipes: &mut OutputPipes, capture: &mut Capture) -> io::Result<()> {
        let stdout_result = drain_one(&mut pipes.stdout, capture, OutputStream::Stdout);
        let stderr_result = drain_one(&mut pipes.stderr, capture, OutputStream::Stderr);
        stdout_result.and(stderr_result)
    }

    fn probe_child_group_absent(pid: libc::pid_t) -> io::Result<bool> {
        let result = unsafe { libc::kill(-pid, 0) };
        if result == 0 {
            return Ok(false);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ESRCH) => Ok(true),
            Some(libc::EPERM) => Ok(false),
            _ => Err(error),
        }
    }

    fn main_poll_timeout(
        context: &OperationContext<'_>,
        leader_exit_at: Option<Instant>,
    ) -> Duration {
        let mut timeout = context.remaining().min(POLL_QUANTUM);
        if let Some(exited_at) = leader_exit_at {
            timeout = timeout.min(
                POST_EXIT_DRAIN.saturating_sub(Instant::now().saturating_duration_since(exited_at)),
            );
        }
        timeout
    }

    fn bounded_remaining(deadline: Instant, maximum: Duration) -> Duration {
        deadline
            .saturating_duration_since(Instant::now())
            .min(maximum)
    }

    fn duration_ceil_millis(duration: Duration) -> i32 {
        if duration.is_zero() {
            return 0;
        }
        let nanos = duration.as_nanos();
        let millis = nanos.saturating_add(999_999) / 1_000_000;
        millis.clamp(1, POLL_QUANTUM.as_millis()) as i32
    }

    use std::os::fd::AsRawFd;
}

#[cfg(target_os = "linux")]
use linux::execute_linux;

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    use super::super::runtime::{Cancellation, NeverCancel};

    #[test]
    #[cfg(target_os = "linux")]
    fn parallel_fresh_executable_fixtures_do_not_race_fork_exec() {
        use std::os::unix::fs::PermissionsExt;
        thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let directory = tempfile::tempdir().unwrap();
                    for index in 0..50 {
                        let path = directory.path().join(format!("fixture-{index}"));
                        {
                            let _guard = executable_fixture_guard();
                            std::fs::write(&path, "#!/bin/sh\nprintf fixture\n").unwrap();
                            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))
                                .unwrap();
                        }
                        let context = OperationContext::new(
                            Instant::now() + Duration::from_secs(5),
                            &NeverCancel,
                        );
                        let output = execute(
                            Command::new(path),
                            None,
                            limits(0, 64),
                            &context,
                            recovery(),
                        )
                        .unwrap();
                        assert_eq!(output.stdout(), b"fixture");
                    }
                });
            }
        });
    }

    fn limits(stdin: usize, output: usize) -> ProcessLimits {
        ProcessLimits::new(stdin, output).unwrap()
    }

    fn recovery() -> RecoveryPolicy {
        RecoveryPolicy::new(Duration::from_secs(2)).unwrap()
    }

    fn shell(script: &str) -> Command {
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(script);
        command
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn captures_stdout_and_stderr_within_limit() {
        let context = OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel);
        let output = execute(
            shell("IFS= read -r data; printf 'stdout:%s' \"$data\"; printf stderr >&2"),
            Some(b"input\n"),
            limits(16, 64),
            &context,
            recovery(),
        )
        .unwrap();

        let stdout = String::from_utf8_lossy(output.stdout());
        let stderr = String::from_utf8_lossy(output.stderr());
        assert!(stdout.contains("stdout:input"));
        assert!(stderr.contains("stderr"));
        assert!(output.status().success());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn early_stdin_close_cannot_report_success_with_truncated_input() {
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let payload = vec![b'x'; MAX_STDIN_BYTES];
        let failure = execute(
            shell("exec 0<&-; printf misleading-success; sleep 0.05"),
            Some(&payload),
            limits(MAX_STDIN_BYTES, 128),
            &context,
            recovery(),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), ProcessFailureKind::InputWriteFailed);
        assert_eq!(
            failure.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert!(!format!("{failure:?}").contains("misleading-success"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn output_overflow_kills_and_reaps_the_group() {
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let failure = execute(
            shell(
                "i=0; while [ $i -lt 1000 ]; do printf 12345; printf 67890 >&2; i=$((i + 1)); done",
            ),
            None,
            limits(0, 128),
            &context,
            recovery(),
        )
        .unwrap_err();

        assert_eq!(failure.kind(), ProcessFailureKind::OutputLimitExceeded);
        assert_eq!(
            failure.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
    }

    struct AtomicCancellation(Arc<AtomicBool>);

    impl Cancellation for AtomicCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cancellation_kills_and_reaps_the_group() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let trigger = Arc::clone(&cancelled);
        let cancellation_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(75));
            trigger.store(true, Ordering::SeqCst);
        });
        let token = AtomicCancellation(cancelled);
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &token);
        let failure = execute(
            shell("sleep 10"),
            None,
            limits(0, 1024),
            &context,
            recovery(),
        )
        .unwrap_err();
        cancellation_thread.join().unwrap();

        assert_eq!(failure.kind(), ProcessFailureKind::Cancelled);
        assert_eq!(
            failure.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn deadline_kills_and_reaps_the_group() {
        let context =
            OperationContext::new(Instant::now() + Duration::from_millis(60), &NeverCancel);
        let failure = execute(
            shell("sleep 10"),
            None,
            limits(0, 1024),
            &context,
            recovery(),
        )
        .unwrap_err();

        assert_eq!(failure.kind(), ProcessFailureKind::DeadlineExceeded);
        assert_eq!(
            failure.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn nonzero_exit_is_typed_without_echoing_streams_or_command() {
        let secret = "secret-process-output-and-argument";
        let mut command = shell("printf '%s' \"$SECRET\"; printf '%s' \"$SECRET\" >&2; exit 23");
        command.env("SECRET", secret);
        let context = OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel);
        let failure = execute(command, None, limits(0, 1024), &context, recovery()).unwrap_err();

        assert_eq!(
            failure.kind(),
            ProcessFailureKind::ChildExitedUnsuccessfully { code: Some(23) }
        );
        let diagnostic = format!("{failure:?} {failure}");
        assert!(!diagnostic.contains(secret));
        assert!(!diagnostic.contains("stdout"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pre_cancelled_call_does_not_spawn() {
        struct Cancelled;
        impl Cancellation for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }

        let cancellation = Cancelled;
        let context = OperationContext::new(Instant::now() + Duration::from_secs(2), &cancellation);
        let failure = execute(
            shell("printf should-not-run"),
            None,
            limits(0, 1024),
            &context,
            recovery(),
        )
        .unwrap_err();

        assert_eq!(failure.kind(), ProcessFailureKind::Cancelled);
        assert_eq!(
            failure.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descendant_holding_output_pipe_is_cleaned_up_after_leader_exit() {
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel);
        let started = Instant::now();
        let result = execute(
            shell("sleep 10 & exit 0"),
            None,
            limits(0, 1024),
            &context,
            recovery(),
        );

        match result {
            Ok(output) => assert!(output.status().success()),
            Err(failure) => {
                assert_eq!(
                    failure.disposition(),
                    ProcessFailureDisposition::OwnershipUncertain
                );
                assert_eq!(failure.kind(), ProcessFailureKind::CleanupUnconfirmed);
            }
        }
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failure_and_success_debug_redact_secret_output() {
        let secret = "sensitive-prompt-fragment";
        let mut command = shell("printf '%s' \"$SECRET\"");
        command.env("SECRET", secret);
        let context = OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel);
        let output = execute(command, None, limits(0, 1024), &context, recovery()).unwrap();
        let debug = format!("{output:?}");

        assert!(String::from_utf8_lossy(output.stdout()).contains(secret));
        assert!(!debug.contains(secret));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn limits_cannot_exceed_hard_caps() {
        let invalid = ProcessLimits::new(MAX_STDIN_BYTES + 1, 1).unwrap_err();
        assert_eq!(invalid.kind(), ProcessFailureKind::InvalidLimits);
        assert_eq!(
            invalid.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
    }
}
