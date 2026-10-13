//! Narrow, in-memory ownership checks for the pre-existing user llama-swap unit.
//!
//! This adapter is deliberately bound to one unit and one caller-owned router
//! configuration. It is not a generic service manager or a crash-recovery
//! journal. Its checks assume that other same-user systemd clients and router
//! configuration writers are quiesced for the complete operation.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::{CStr, OsStr};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Take};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use super::process::{self, ProcessLimits};
use super::runtime::{OperationContext, RecoveryPolicy};

const UNIT: &str = "llama-swap.service";
const SYSTEMCTL: &str = "/usr/bin/systemctl";
const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const MAX_MANAGER_BYTES: usize = 32 * 1024;
const MAX_FILE_BYTES: u64 = 1024 * 1024;
const MAX_SNAPSHOT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_DROP_INS: usize = 32;
const MAX_CGROUP_DIRS: usize = 256;
const MAX_CGROUP_PIDS: usize = 4096;
const MAX_CGROUP_PROCS_BYTES: usize = 64 * 1024;
const PROC_FILE_BYTES: usize = 16 * 1024;
const MANAGER_RECOVERY: std::time::Duration = std::time::Duration::from_secs(5);

const PROPERTIES: [&str; 14] = [
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "InvocationID",
    "ControlGroup",
    "KillMode",
    "SendSIGKILL",
    "Restart",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "ExecStart",
];

/// Fixed, redacted failures. No manager output, paths, PID, file contents, or
/// syscall message is retained in an error value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidentUnitError {
    InvalidInput,
    ContextUnavailable,
    ManagerUnavailable,
    InvalidManagerReply,
    UnitNotSupported,
    DefinitionChanged,
    UnsafeFile,
    UnsafeCgroup,
    ProcessIdentityUnavailable,
    StopUnconfirmed,
    StartUnconfirmed,
    OperationAlreadyAttempted,
}

impl fmt::Display for ResidentUnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidInput => "resident unit input is invalid",
            Self::ContextUnavailable => "resident unit operation is cancelled or expired",
            Self::ManagerUnavailable => "user service manager operation failed",
            Self::InvalidManagerReply => "user service manager reply is invalid",
            Self::UnitNotSupported => "resident llama-swap unit state is unsupported",
            Self::DefinitionChanged => "resident unit definition or configuration changed",
            Self::UnsafeFile => "resident unit definition or configuration is unsafe",
            Self::UnsafeCgroup => "resident unit cgroup could not be verified",
            Self::ProcessIdentityUnavailable => {
                "resident unit process identity could not be verified"
            }
            Self::StopUnconfirmed => "resident unit stop was not confirmed",
            Self::StartUnconfirmed => "resident unit start was not confirmed",
            Self::OperationAlreadyAttempted => "resident unit operation was already attempted",
        })
    }
}

impl Error for ResidentUnitError {}

/// Opaque snapshot of the one supported resident user service.
///
/// This stores bounded manager-definition and config-file snapshots in memory
/// only. It intentionally has no `Debug` implementation and exposes no raw
/// properties, paths, process identities, or file contents.
pub(crate) struct ResidentUnit {
    systemctl: PathBuf,
    cgroup_root: PathBuf,
    proc_root: PathBuf,
    router_config: PathBuf,
    uid: u32,
    definition: UnitDefinition,
    original: Generation,
    was_active: bool,
    members: BTreeMap<u32, ProcessIdentity>,
    pause_called: bool,
    stop_attempted: bool,
    paused: bool,
    restore_called: bool,
    start_attempted: bool,
    restored: Option<Generation>,
}

#[derive(Clone, Eq, PartialEq)]
struct Generation {
    invocation_id: String,
    cgroup: String,
    main_pid: u32,
    main_identity: Option<ProcessIdentity>,
}

#[derive(Clone, Eq, PartialEq)]
struct ProcessIdentity {
    start_ticks: u64,
    uid: u32,
    cgroup: String,
}

#[derive(Clone, Eq, PartialEq)]
struct UnitDefinition {
    id: String,
    kill_mode: String,
    send_sigkill: String,
    restart: String,
    unit_file_state: String,
    fragment_path: PathBuf,
    drop_in_paths: Vec<PathBuf>,
    exec_start: String,
    files: Vec<FileSnapshot>,
}

#[derive(Clone, Eq, PartialEq)]
struct FileSnapshot {
    path: PathBuf,
    dev: u64,
    ino: u64,
    uid: u32,
    gid: u32,
    mode: u32,
    nlink: u64,
    len: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
    bytes: Vec<u8>,
}

struct ManagerSnapshot {
    id: String,
    active_state: String,
    sub_state: String,
    main_pid: u32,
    invocation_id: String,
    cgroup: String,
    kill_mode: String,
    send_sigkill: String,
    restart: String,
    unit_file_state: String,
    fragment_path: PathBuf,
    drop_in_paths: Vec<PathBuf>,
    exec_start: String,
}

impl ResidentUnit {
    /// Capture the fixed user service and a trusted private router config.
    pub(crate) fn capture(
        router_config: &Path,
        context: &OperationContext<'_>,
    ) -> Result<Self, ResidentUnitError> {
        Self::capture_with_paths(
            router_config,
            Path::new(SYSTEMCTL),
            Path::new(CGROUP_ROOT),
            Path::new("/proc"),
            context,
            false,
        )
    }

    fn capture_with_paths(
        router_config: &Path,
        systemctl: &Path,
        cgroup_root: &Path,
        proc_root: &Path,
        context: &OperationContext<'_>,
        test_paths: bool,
    ) -> Result<Self, ResidentUnitError> {
        check_context(context)?;
        let uid = unsafe { libc::geteuid() };
        let router_config = validate_router_config_path(router_config)?;
        validate_executable(systemctl, test_paths)?;
        validate_cgroup_root(cgroup_root, !test_paths)?;

        let manager = show(systemctl, context)?;
        verify_effective_router_config(systemctl, &router_config, uid, context)?;
        let generation = generation_from_manager(&manager, uid, proc_root)?;
        let definition = capture_definition(&manager, &router_config, uid, context)?;
        let members = if generation.cgroup.is_empty() {
            BTreeMap::new()
        } else {
            scan_members(cgroup_root, &generation.cgroup, proc_root, uid, context)?
        };
        if manager.active_state == "active" {
            let main = members
                .get(&generation.main_pid)
                .ok_or(ResidentUnitError::ProcessIdentityUnavailable)?;
            if Some(main.clone()) != generation.main_identity {
                return Err(ResidentUnitError::ProcessIdentityUnavailable);
            }
            verify_members_router_config(proc_root, &members, &router_config, uid, context)?;
        } else if !members.is_empty() {
            return Err(ResidentUnitError::UnsafeCgroup);
        }

        Ok(Self {
            systemctl: systemctl.to_owned(),
            cgroup_root: cgroup_root.to_owned(),
            proc_root: proc_root.to_owned(),
            router_config,
            uid,
            definition,
            original: generation,
            was_active: manager.active_state == "active",
            members,
            pause_called: false,
            stop_attempted: false,
            paused: false,
            restore_called: false,
            start_attempted: false,
            restored: None,
        })
    }

    pub(crate) fn was_active(&self) -> bool {
        self.was_active
    }

    /// Check only identities captured from the current verified unit subtree.
    /// A false result is absence of this unit's proof, not proof that the PID
    /// is harmless or that the GPU is idle.
    pub(crate) fn accepts_claimant(&self, pid: u32) -> bool {
        if pid <= 1 {
            return false;
        }
        let Some(expected) = self.members.get(&pid) else {
            return false;
        };
        read_process_identity(&self.proc_root, pid, self.uid)
            .map(|actual| actual == *expected)
            .unwrap_or(false)
    }

    /// Stop the original active generation at most once. An originally
    /// inactive unit is only re-verified; this method never starts or stops it.
    pub(crate) fn pause(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<(), ResidentUnitError> {
        if self.pause_called {
            return Err(ResidentUnitError::OperationAlreadyAttempted);
        }
        self.pause_called = true;
        check_context(context)?;
        let current = self.observe_unchanged(context)?;
        if self.was_active {
            self.require_generation(&current, &self.original, true)?;
            let members = scan_members(
                &self.cgroup_root,
                &self.original.cgroup,
                &self.proc_root,
                self.uid,
                context,
            )?;
            if members.get(&self.original.main_pid) != self.original.main_identity.as_ref() {
                return Err(ResidentUnitError::ProcessIdentityUnavailable);
            }
            verify_members_router_config(
                &self.proc_root,
                &members,
                &self.router_config,
                self.uid,
                context,
            )?;
            self.stop_attempted = true;
            manager_action(&self.systemctl, "stop", context)
                .map_err(|_| ResidentUnitError::StopUnconfirmed)?;
            let after = self.observe_unchanged(context)?;
            if after.active_state != "inactive"
                || after.sub_state != "dead"
                || after.main_pid != 0
                || !inactive_cgroup_matches(&after.cgroup, &self.original.cgroup)
                || (!after.invocation_id.is_empty()
                    && after.invocation_id != self.original.invocation_id)
            {
                return Err(ResidentUnitError::StopUnconfirmed);
            }
            if !scan_members(
                &self.cgroup_root,
                &self.original.cgroup,
                &self.proc_root,
                self.uid,
                context,
            )?
            .is_empty()
            {
                return Err(ResidentUnitError::StopUnconfirmed);
            }
            self.members.clear();
        } else if !self.original.cgroup.is_empty() {
            require_inactive(&current)?;
            if !scan_members(
                &self.cgroup_root,
                &self.original.cgroup,
                &self.proc_root,
                self.uid,
                context,
            )?
            .is_empty()
            {
                return Err(ResidentUnitError::UnsafeCgroup);
            }
        } else {
            require_inactive(&current)?;
        }
        self.paused = true;
        Ok(())
    }

    /// Restore the captured state at most once. An originally inactive unit is
    /// verified and left untouched; an active one gets one start attempt only
    /// after the acknowledged old generation and unchanged definition verify.
    pub(crate) fn restore(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<(), ResidentUnitError> {
        if self.restore_called {
            return Err(ResidentUnitError::OperationAlreadyAttempted);
        }
        if !self.pause_called {
            return Err(ResidentUnitError::OperationAlreadyAttempted);
        }
        self.restore_called = true;
        check_context(context)?;
        let current = self.observe_unchanged(context)?;
        if !self.was_active {
            require_inactive(&current)?;
            if !self.original.cgroup.is_empty()
                && !scan_members(
                    &self.cgroup_root,
                    &self.original.cgroup,
                    &self.proc_root,
                    self.uid,
                    context,
                )?
                .is_empty()
            {
                return Err(ResidentUnitError::UnsafeCgroup);
            }
            self.restored = Some(self.original.clone());
            return check_context(context);
        }
        if !self.paused || !self.stop_attempted || self.start_attempted {
            return Err(ResidentUnitError::OperationAlreadyAttempted);
        }
        if current.active_state != "inactive"
            || current.sub_state != "dead"
            || current.main_pid != 0
            || !inactive_cgroup_matches(&current.cgroup, &self.original.cgroup)
        {
            return Err(ResidentUnitError::DefinitionChanged);
        }
        if !self.original.cgroup.is_empty()
            && !scan_members(
                &self.cgroup_root,
                &self.original.cgroup,
                &self.proc_root,
                self.uid,
                context,
            )?
            .is_empty()
        {
            return Err(ResidentUnitError::UnsafeCgroup);
        }
        self.start_attempted = true;
        manager_action(&self.systemctl, "start", context)
            .map_err(|_| ResidentUnitError::StartUnconfirmed)?;

        let after = self.observe_unchanged(context)?;
        let generation = generation_from_manager(&after, self.uid, &self.proc_root)
            .map_err(|_| ResidentUnitError::StartUnconfirmed)?;
        if after.active_state != "active"
            || after.sub_state != "running"
            || generation.invocation_id == self.original.invocation_id
            || generation.cgroup != self.original.cgroup
        {
            return Err(ResidentUnitError::StartUnconfirmed);
        }
        let members = scan_members(
            &self.cgroup_root,
            &generation.cgroup,
            &self.proc_root,
            self.uid,
            context,
        )
        .map_err(|_| ResidentUnitError::StartUnconfirmed)?;
        if members.get(&generation.main_pid) != generation.main_identity.as_ref() {
            return Err(ResidentUnitError::StartUnconfirmed);
        }
        verify_members_router_config(
            &self.proc_root,
            &members,
            &self.router_config,
            self.uid,
            context,
        )
        .map_err(|_| ResidentUnitError::StartUnconfirmed)?;
        self.members = members;
        self.restored = Some(generation);
        check_context(context)
    }

    /// Verify the restored generation and refresh the bounded current PID
    /// identity set. Callers may use `accepts_claimant` only as a membership
    /// predicate for the most recently verified unit subtree.
    pub(crate) fn verify_restored(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<(), ResidentUnitError> {
        if !self.restore_called {
            return Err(ResidentUnitError::OperationAlreadyAttempted);
        }
        check_context(context)?;
        let current = self.observe_unchanged(context)?;
        let expected = self
            .restored
            .as_ref()
            .ok_or(ResidentUnitError::StartUnconfirmed)?;
        self.require_generation(&current, expected, self.was_active)?;
        let members = if self.was_active {
            let members = scan_members(
                &self.cgroup_root,
                &expected.cgroup,
                &self.proc_root,
                self.uid,
                context,
            )?;
            if members.get(&expected.main_pid) != expected.main_identity.as_ref() {
                return Err(ResidentUnitError::ProcessIdentityUnavailable);
            }
            verify_members_router_config(
                &self.proc_root,
                &members,
                &self.router_config,
                self.uid,
                context,
            )?;
            members
        } else {
            let members = scan_members(
                &self.cgroup_root,
                &expected.cgroup,
                &self.proc_root,
                self.uid,
                context,
            )?;
            if !members.is_empty() {
                return Err(ResidentUnitError::UnsafeCgroup);
            }
            members
        };
        self.members = members;
        check_context(context)
    }

    fn require_generation(
        &self,
        current: &ManagerSnapshot,
        expected: &Generation,
        active: bool,
    ) -> Result<(), ResidentUnitError> {
        let (state, substate) = if active {
            ("active", "running")
        } else {
            ("inactive", "dead")
        };
        let invocation_matches = current.invocation_id == expected.invocation_id;
        if current.active_state != state
            || current.sub_state != substate
            || current.main_pid != expected.main_pid
            || if active {
                current.cgroup != expected.cgroup
            } else {
                !inactive_cgroup_matches(&current.cgroup, &expected.cgroup)
            }
            || !invocation_matches
        {
            return Err(ResidentUnitError::DefinitionChanged);
        }
        if active {
            let actual = read_process_identity(&self.proc_root, expected.main_pid, self.uid)
                .ok_or(ResidentUnitError::ProcessIdentityUnavailable)?;
            if Some(actual) != expected.main_identity {
                return Err(ResidentUnitError::ProcessIdentityUnavailable);
            }
        }
        Ok(())
    }

    fn observe_unchanged(
        &self,
        context: &OperationContext<'_>,
    ) -> Result<ManagerSnapshot, ResidentUnitError> {
        let current = show(&self.systemctl, context)?;
        verify_effective_router_config(&self.systemctl, &self.router_config, self.uid, context)?;
        let actual = capture_definition(&current, &self.router_config, self.uid, context)?;
        if actual != self.definition {
            return Err(ResidentUnitError::DefinitionChanged);
        }
        Ok(current)
    }
}

fn check_context(context: &OperationContext<'_>) -> Result<(), ResidentUnitError> {
    context
        .check_active("resident unit operation")
        .map_err(|_| ResidentUnitError::ContextUnavailable)
}

fn validate_executable(path: &Path, test_paths: bool) -> Result<(), ResidentUnitError> {
    if !path.is_absolute() || !test_paths && path != Path::new(SYSTEMCTL) {
        return Err(ResidentUnitError::InvalidInput);
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.mode() & 0o111 == 0
        || !test_paths && (metadata.uid() != 0 || metadata.mode() & 0o022 != 0)
    {
        return Err(ResidentUnitError::ManagerUnavailable);
    }
    Ok(())
}

fn validate_cgroup_root(path: &Path, require_cgroup2: bool) -> Result<(), ResidentUnitError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| ResidentUnitError::UnsafeCgroup)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || path
            .canonicalize()
            .map_err(|_| ResidentUnitError::UnsafeCgroup)?
            != path
    {
        return Err(ResidentUnitError::UnsafeCgroup);
    }
    if require_cgroup2 {
        let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
        let file = File::open(path).map_err(|_| ResidentUnitError::UnsafeCgroup)?;
        if unsafe { libc::fstatfs(std::os::fd::AsRawFd::as_raw_fd(&file), stat.as_mut_ptr()) } != 0
            || unsafe { stat.assume_init() }.f_type != 0x6367_7270
        {
            return Err(ResidentUnitError::UnsafeCgroup);
        }
    }
    Ok(())
}

fn validate_router_config_path(path: &Path) -> Result<PathBuf, ResidentUnitError> {
    if !path.is_absolute() || !safe_plain_path(path, false) {
        return Err(ResidentUnitError::InvalidInput);
    }
    let canonical = path
        .canonicalize()
        .map_err(|_| ResidentUnitError::UnsafeFile)?;
    if canonical != path {
        return Err(ResidentUnitError::UnsafeFile);
    }
    Ok(canonical)
}

fn show(
    systemctl: &Path,
    context: &OperationContext<'_>,
) -> Result<ManagerSnapshot, ResidentUnitError> {
    check_context(context)?;
    let mut command = Command::new(systemctl);
    command
        .arg("--user")
        .arg("--no-pager")
        .arg("--no-ask-password")
        .arg("show")
        .arg("--all");
    for property in PROPERTIES {
        command.arg(format!("--property={property}"));
    }
    command.arg(UNIT);
    let limits = ProcessLimits::new(0, MAX_MANAGER_BYTES)
        .map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    let recovery =
        RecoveryPolicy::new(MANAGER_RECOVERY).map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    let output = process::execute(command, None, limits, context, recovery)
        .map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    if !output.stderr().iter().all(u8::is_ascii_whitespace) {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    parse_manager_snapshot(output.stdout())
}

fn manager_action(
    systemctl: &Path,
    action: &str,
    context: &OperationContext<'_>,
) -> Result<(), ResidentUnitError> {
    check_context(context)?;
    let mut command = Command::new(systemctl);
    command
        .arg("--user")
        .arg("--no-pager")
        .arg("--no-ask-password")
        .arg(action)
        .arg(UNIT);
    let limits = ProcessLimits::new(0, MAX_MANAGER_BYTES)
        .map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    let recovery =
        RecoveryPolicy::new(MANAGER_RECOVERY).map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    let output = process::execute(command, None, limits, context, recovery)
        .map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    if !output.stdout().iter().all(u8::is_ascii_whitespace)
        || !output.stderr().iter().all(u8::is_ascii_whitespace)
    {
        return Err(ResidentUnitError::ManagerUnavailable);
    }
    check_context(context)
}

fn verify_effective_router_config(
    systemctl: &Path,
    router_config: &Path,
    uid: u32,
    context: &OperationContext<'_>,
) -> Result<(), ResidentUnitError> {
    check_context(context)?;
    let mut command = Command::new(systemctl);
    command
        .arg("--user")
        .arg("--no-pager")
        .arg("--no-ask-password")
        .arg("show")
        .arg("--property=Environment")
        .arg(UNIT);
    let limits = ProcessLimits::new(0, MAX_MANAGER_BYTES)
        .map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    let recovery =
        RecoveryPolicy::new(MANAGER_RECOVERY).map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    let output = process::execute(command, None, limits, context, recovery)
        .map_err(|_| ResidentUnitError::ManagerUnavailable)?;
    if !output.stderr().iter().all(u8::is_ascii_whitespace) {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    let root = parse_effective_router_root(output.stdout())?;
    // The full manager environment is intentionally ephemeral: only the
    // selected L3MS_ROOT survives this point, and it is immediately resolved.
    drop(output);
    let expanded = resolve_user_path(&root, uid)?;
    let configured = expanded
        .join("llama-swap.yaml")
        .canonicalize()
        .map_err(|_| ResidentUnitError::UnsafeFile)?;
    if configured != router_config {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    check_context(context)
}

fn parse_effective_router_root(bytes: &[u8]) -> Result<String, ResidentUnitError> {
    if bytes.len() > MAX_MANAGER_BYTES {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| ResidentUnitError::InvalidManagerReply)?;
    let mut lines = text.lines();
    let line = lines.next().ok_or(ResidentUnitError::InvalidManagerReply)?;
    if lines.next().is_some() {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    let value = line
        .strip_prefix("Environment=")
        .ok_or(ResidentUnitError::InvalidManagerReply)?;
    let mut root = None;
    for token in value.split_ascii_whitespace() {
        if let Some(path) = token.strip_prefix("L3MS_ROOT=") {
            if path.is_empty()
                || path
                    .bytes()
                    .any(|byte| matches!(byte, b'\\' | b'\'' | b'"' | b'$'))
                || path.chars().any(char::is_control)
                || root.is_some()
            {
                return Err(ResidentUnitError::InvalidManagerReply);
            }
            root = Some(path.to_owned());
        } else if token.contains("L3MS_ROOT") {
            return Err(ResidentUnitError::InvalidManagerReply);
        }
    }
    root.ok_or(ResidentUnitError::UnitNotSupported)
}

fn resolve_user_path(value: &str, uid: u32) -> Result<PathBuf, ResidentUnitError> {
    let path = if value == "%h" || value.starts_with("%h/") {
        let suffix = value.strip_prefix("%h").unwrap_or_default();
        user_home(uid)?.join(suffix.trim_start_matches('/'))
    } else if value.starts_with('/') {
        PathBuf::from(value)
    } else {
        return Err(ResidentUnitError::UnitNotSupported);
    };
    if !safe_plain_path(&path, false) {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    Ok(path)
}

fn verify_members_router_config(
    proc_root: &Path,
    members: &BTreeMap<u32, ProcessIdentity>,
    router_config: &Path,
    uid: u32,
    context: &OperationContext<'_>,
) -> Result<(), ResidentUnitError> {
    let expected = router_config
        .to_str()
        .ok_or(ResidentUnitError::UnsafeFile)?;
    let mut matches = 0usize;
    for (&pid, identity) in members {
        check_context(context)?;
        let before = read_process_identity(proc_root, pid, uid)
            .ok_or(ResidentUnitError::ProcessIdentityUnavailable)?;
        if before != *identity {
            return Err(ResidentUnitError::ProcessIdentityUnavailable);
        }
        let bytes = read_bounded_regular(
            &proc_root.join(pid.to_string()).join("cmdline"),
            PROC_FILE_BYTES,
        )
        .map_err(|_| ResidentUnitError::ProcessIdentityUnavailable)?;
        let args: Vec<_> = bytes
            .split(|byte| *byte == 0)
            .filter(|arg| !arg.is_empty())
            .map(|arg| {
                std::str::from_utf8(arg).map_err(|_| ResidentUnitError::ProcessIdentityUnavailable)
            })
            .collect::<Result<_, _>>()?;
        for (index, arg) in args.iter().enumerate() {
            if *arg == "-config" {
                if args.get(index + 1).copied() != Some(expected) {
                    return Err(ResidentUnitError::UnitNotSupported);
                }
                matches += 1;
            }
        }
        let after = read_process_identity(proc_root, pid, uid)
            .ok_or(ResidentUnitError::ProcessIdentityUnavailable)?;
        if after != before {
            return Err(ResidentUnitError::ProcessIdentityUnavailable);
        }
    }
    if matches != 1 {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    check_context(context)
}

fn parse_manager_snapshot(bytes: &[u8]) -> Result<ManagerSnapshot, ResidentUnitError> {
    if bytes.len() > MAX_MANAGER_BYTES {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| ResidentUnitError::InvalidManagerReply)?;
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .ok_or(ResidentUnitError::InvalidManagerReply)?;
        if !PROPERTIES.contains(&key) || values.insert(key, value).is_some() {
            return Err(ResidentUnitError::InvalidManagerReply);
        }
    }
    if values.len() != PROPERTIES.len() {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    let get = |key: &str| {
        values
            .get(key)
            .copied()
            .ok_or(ResidentUnitError::InvalidManagerReply)
    };
    let id = get("Id")?.to_owned();
    let load_state = get("LoadState")?;
    let active_state = get("ActiveState")?.to_owned();
    let sub_state = get("SubState")?.to_owned();
    let main_pid = get("MainPID")?
        .parse::<u32>()
        .map_err(|_| ResidentUnitError::InvalidManagerReply)?;
    let invocation_id = get("InvocationID")?.to_owned();
    let cgroup = get("ControlGroup")?.to_owned();
    let kill_mode = get("KillMode")?.to_owned();
    let send_sigkill = get("SendSIGKILL")?.to_owned();
    let restart = get("Restart")?.to_owned();
    let unit_file_state = get("UnitFileState")?.to_owned();
    let fragment_path = PathBuf::from(get("FragmentPath")?);
    let drop_in_paths = parse_drop_in_paths(get("DropInPaths")?)?;
    let exec_start = normalize_exec_start(get("ExecStart")?)?;
    if id != UNIT
        || load_state != "loaded"
        || kill_mode != "control-group"
        || send_sigkill != "yes"
        || restart.is_empty()
        || unit_file_state.is_empty()
        || !safe_plain_path(&fragment_path, false)
        || !cgroup.is_empty() && !valid_cgroup_text(&cgroup)
        || !invocation_id.is_empty() && !valid_invocation_id(&invocation_id)
    {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    let valid_active = active_state == "active" && sub_state == "running" && main_pid > 1;
    let valid_inactive = active_state == "inactive" && sub_state == "dead" && main_pid == 0;
    if !valid_active && !valid_inactive {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    if valid_active && cgroup.is_empty() {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    if valid_active && !valid_invocation_id(&invocation_id) {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    Ok(ManagerSnapshot {
        id,
        active_state,
        sub_state,
        main_pid,
        invocation_id,
        cgroup,
        kill_mode,
        send_sigkill,
        restart,
        unit_file_state,
        fragment_path,
        drop_in_paths,
        exec_start,
    })
}

fn parse_drop_in_paths(raw: &str) -> Result<Vec<PathBuf>, ResidentUnitError> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    if raw.contains('\\') || raw.chars().any(char::is_control) {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    let mut paths = Vec::new();
    let mut seen = BTreeSet::new();
    for part in raw.split_ascii_whitespace() {
        let path = PathBuf::from(part);
        if !safe_plain_path(&path, false) || !seen.insert(path.clone()) {
            return Err(ResidentUnitError::InvalidManagerReply);
        }
        paths.push(path);
        if paths.len() > MAX_DROP_INS {
            return Err(ResidentUnitError::InvalidManagerReply);
        }
    }
    if paths.is_empty() {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    Ok(paths)
}

fn normalize_exec_start(raw: &str) -> Result<String, ResidentUnitError> {
    if raw.is_empty() || raw.chars().any(char::is_control) {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    let normalized = raw.split(" ; start_time=").next().unwrap_or(raw);
    if !normalized.contains("path=") || !normalized.contains("argv[]=") {
        return Err(ResidentUnitError::InvalidManagerReply);
    }
    Ok(normalized.to_owned())
}

fn generation_from_manager(
    manager: &ManagerSnapshot,
    uid: u32,
    proc_root: &Path,
) -> Result<Generation, ResidentUnitError> {
    let active = manager.active_state == "active";
    if active || !manager.cgroup.is_empty() {
        validate_user_cgroup(&manager.cgroup, uid)?;
    }
    let main_identity = if active {
        Some(
            read_process_identity(proc_root, manager.main_pid, uid)
                .ok_or(ResidentUnitError::ProcessIdentityUnavailable)?,
        )
    } else {
        None
    };
    Ok(Generation {
        invocation_id: manager.invocation_id.clone(),
        cgroup: manager.cgroup.clone(),
        main_pid: manager.main_pid,
        main_identity,
    })
}

fn inactive_cgroup_matches(current: &str, original: &str) -> bool {
    current == original || !original.is_empty() && current.is_empty()
}

fn validate_user_cgroup(path: &str, uid: u32) -> Result<(), ResidentUnitError> {
    let prefix = format!("/user.slice/user-{uid}.slice/user@{uid}.service/");
    if !valid_cgroup_text(path)
        || !path.starts_with(&prefix)
        || !path.ends_with("/llama-swap.service")
    {
        return Err(ResidentUnitError::UnsafeCgroup);
    }
    Ok(())
}

fn valid_cgroup_text(path: &str) -> bool {
    path.starts_with('/')
        && path.len() <= 4096
        && !path.ends_with('/')
        && !path.contains("//")
        && !path.contains('\\')
        && !path.chars().any(char::is_control)
        && path
            .split('/')
            .skip(1)
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn valid_invocation_id(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn capture_definition(
    manager: &ManagerSnapshot,
    router_config: &Path,
    uid: u32,
    context: &OperationContext<'_>,
) -> Result<UnitDefinition, ResidentUnitError> {
    let mut paths = Vec::with_capacity(manager.drop_in_paths.len() + 2);
    paths.push(manager.fragment_path.clone());
    paths.extend(manager.drop_in_paths.iter().cloned());
    paths.push(router_config.to_owned());
    let mut files = Vec::with_capacity(paths.len());
    let mut total = 0u64;
    for (index, path) in paths.iter().enumerate() {
        check_context(context)?;
        let snapshot = read_file_snapshot(path, uid, index + 1 == paths.len(), context)?;
        total = total
            .checked_add(snapshot.len)
            .ok_or(ResidentUnitError::UnsafeFile)?;
        if total > MAX_SNAPSHOT_BYTES {
            return Err(ResidentUnitError::UnsafeFile);
        }
        files.push(snapshot);
    }
    let definition = UnitDefinition {
        id: manager.id.clone(),
        kill_mode: manager.kill_mode.clone(),
        send_sigkill: manager.send_sigkill.clone(),
        restart: manager.restart.clone(),
        unit_file_state: manager.unit_file_state.clone(),
        fragment_path: manager.fragment_path.clone(),
        drop_in_paths: manager.drop_in_paths.clone(),
        exec_start: manager.exec_start.clone(),
        files,
    };
    verify_router_exec_config(&definition, router_config, uid)?;
    Ok(definition)
}

fn verify_router_exec_config(
    definition: &UnitDefinition,
    _router_config: &Path,
    _uid: u32,
) -> Result<(), ResidentUnitError> {
    if definition.exec_start.matches("-config").count() != 1
        || !definition
            .exec_start
            .contains(" -config ${L3MS_ROOT}/llama-swap.yaml")
    {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    for snapshot in definition
        .files
        .iter()
        .take(definition.files.len().saturating_sub(1))
    {
        let text =
            std::str::from_utf8(&snapshot.bytes).map_err(|_| ResidentUnitError::UnsafeFile)?;
        let mut in_service = false;
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('[') && line.ends_with(']') {
                in_service = &line[1..line.len() - 1] == "Service";
                continue;
            }
            if !in_service || line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            match key.trim() {
                "EnvironmentFile" => return Err(ResidentUnitError::UnitNotSupported),
                "UnsetEnvironment" if value.contains("L3MS_ROOT") => {
                    return Err(ResidentUnitError::UnitNotSupported);
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn user_home(uid: u32) -> Result<PathBuf, ResidentUnitError> {
    let mut password = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut result = std::ptr::null_mut();
    let mut storage = vec![0u8; 16 * 1024];
    let code = unsafe {
        libc::getpwuid_r(
            uid,
            password.as_mut_ptr(),
            storage.as_mut_ptr().cast(),
            storage.len(),
            &mut result,
        )
    };
    if code != 0 || result.is_null() {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    let password = unsafe { password.assume_init() };
    if password.pw_dir.is_null() {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    let home = PathBuf::from(OsStr::from_bytes(unsafe {
        CStr::from_ptr(password.pw_dir).to_bytes()
    }));
    if !home.is_absolute() {
        return Err(ResidentUnitError::UnitNotSupported);
    }
    Ok(home)
}

fn safe_plain_path(path: &Path, allow_spaces: bool) -> bool {
    path.is_absolute()
        && !path.as_os_str().is_empty()
        && path.to_str().is_some_and(|value| {
            !value.contains('\\')
                && !value.chars().any(char::is_control)
                && (allow_spaces || !value.chars().any(char::is_whitespace))
        })
        && path.components().all(|component| {
            matches!(component, Component::RootDir | Component::Normal(_))
                && !matches!(component, Component::Normal(value) if value == OsStr::new("."))
        })
}

fn read_file_snapshot(
    path: &Path,
    uid: u32,
    router_config: bool,
    context: &OperationContext<'_>,
) -> Result<FileSnapshot, ResidentUnitError> {
    check_context(context)?;
    if !safe_plain_path(path, false) {
        return Err(ResidentUnitError::UnsafeFile);
    }
    if path
        .canonicalize()
        .map_err(|_| ResidentUnitError::UnsafeFile)?
        != path
    {
        return Err(ResidentUnitError::UnsafeFile);
    }
    let before = fs::symlink_metadata(path).map_err(|_| ResidentUnitError::UnsafeFile)?;
    let owner_ok = if router_config {
        before.uid() == uid && before.mode() & 0o022 == 0 && before.nlink() == 1
    } else {
        (before.uid() == uid || before.uid() == 0) && before.mode() & 0o022 == 0
    };
    if before.file_type().is_symlink()
        || !before.is_file()
        || !owner_ok
        || before.len() > MAX_FILE_BYTES
    {
        return Err(ResidentUnitError::UnsafeFile);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| ResidentUnitError::UnsafeFile)?;
    let opened = file.metadata().map_err(|_| ResidentUnitError::UnsafeFile)?;
    if !opened.is_file()
        || (opened.dev(), opened.ino()) != (before.dev(), before.ino())
        || opened.uid() != before.uid()
        || opened.mode() != before.mode()
        || opened.nlink() != before.nlink()
        || opened.len() != before.len()
    {
        return Err(ResidentUnitError::UnsafeFile);
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    check_context(context)?;
    let mut bounded: Take<&File> = (&file).take(MAX_FILE_BYTES + 1);
    bounded
        .read_to_end(&mut bytes)
        .map_err(|_| ResidentUnitError::UnsafeFile)?;
    check_context(context)?;
    let after = file.metadata().map_err(|_| ResidentUnitError::UnsafeFile)?;
    if bytes.len() as u64 > MAX_FILE_BYTES
        || (
            after.dev(),
            after.ino(),
            after.uid(),
            after.gid(),
            after.mode(),
            after.nlink(),
            after.len(),
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec(),
        ) != (
            opened.dev(),
            opened.ino(),
            opened.uid(),
            opened.gid(),
            opened.mode(),
            opened.nlink(),
            opened.len(),
            opened.mtime(),
            opened.mtime_nsec(),
            opened.ctime(),
            opened.ctime_nsec(),
        )
    {
        return Err(ResidentUnitError::UnsafeFile);
    }
    Ok(FileSnapshot {
        path: path.to_owned(),
        dev: after.dev(),
        ino: after.ino(),
        uid: after.uid(),
        gid: after.gid(),
        mode: after.mode(),
        nlink: after.nlink(),
        len: after.len(),
        mtime: after.mtime(),
        mtime_nsec: after.mtime_nsec(),
        ctime: after.ctime(),
        ctime_nsec: after.ctime_nsec(),
        bytes,
    })
}

fn require_inactive(manager: &ManagerSnapshot) -> Result<(), ResidentUnitError> {
    if manager.active_state == "inactive" && manager.sub_state == "dead" && manager.main_pid == 0 {
        Ok(())
    } else {
        Err(ResidentUnitError::DefinitionChanged)
    }
}

fn scan_members(
    root: &Path,
    cgroup: &str,
    proc_root: &Path,
    uid: u32,
    context: &OperationContext<'_>,
) -> Result<BTreeMap<u32, ProcessIdentity>, ResidentUnitError> {
    check_context(context)?;
    if cgroup.is_empty() {
        return Ok(BTreeMap::new());
    }
    if !valid_cgroup_text(cgroup) {
        return Err(ResidentUnitError::UnsafeCgroup);
    }
    let mut members = BTreeMap::new();
    let Some(target) = resolve_cgroup_directory(root, cgroup)? else {
        return Ok(members);
    };
    let mut pending = vec![target];
    let mut directories = 0usize;
    while let Some(directory) = pending.pop() {
        check_context(context)?;
        directories += 1;
        if directories > MAX_CGROUP_DIRS {
            return Err(ResidentUnitError::UnsafeCgroup);
        }
        let current_meta =
            fs::symlink_metadata(&directory).map_err(|_| ResidentUnitError::UnsafeCgroup)?;
        if current_meta.file_type().is_symlink() || !current_meta.is_dir() {
            return Err(ResidentUnitError::UnsafeCgroup);
        }
        let procs_path = directory.join("cgroup.procs");
        let procs = read_bounded_regular(&procs_path, MAX_CGROUP_PROCS_BYTES)
            .map_err(|_| ResidentUnitError::UnsafeCgroup)?;
        let rel = directory
            .strip_prefix(root)
            .map_err(|_| ResidentUnitError::UnsafeCgroup)?;
        let cgroup_path = format!("/{}", rel.to_string_lossy());
        if !valid_cgroup_text(&cgroup_path) || !cgroup_path_is_within(&cgroup_path, cgroup) {
            return Err(ResidentUnitError::UnsafeCgroup);
        }
        for row in procs.split(|byte| *byte == b'\n') {
            if row.is_empty() {
                continue;
            }
            check_context(context)?;
            let text = std::str::from_utf8(row).map_err(|_| ResidentUnitError::UnsafeCgroup)?;
            let pid = text
                .parse::<u32>()
                .map_err(|_| ResidentUnitError::UnsafeCgroup)?;
            if pid <= 1 || pid > i32::MAX as u32 || members.len() >= MAX_CGROUP_PIDS {
                return Err(ResidentUnitError::UnsafeCgroup);
            }
            let identity = read_process_identity(proc_root, pid, uid)
                .ok_or(ResidentUnitError::ProcessIdentityUnavailable)?;
            if identity.cgroup != cgroup_path || members.insert(pid, identity).is_some() {
                return Err(ResidentUnitError::ProcessIdentityUnavailable);
            }
        }
        let entries = fs::read_dir(&directory).map_err(|_| ResidentUnitError::UnsafeCgroup)?;
        for entry in entries {
            check_context(context)?;
            let entry = entry.map_err(|_| ResidentUnitError::UnsafeCgroup)?;
            let name = entry.file_name();
            if name == OsStr::new("cgroup.procs") || name == OsStr::new("cgroup.threads") {
                continue;
            }
            let metadata =
                fs::symlink_metadata(entry.path()).map_err(|_| ResidentUnitError::UnsafeCgroup)?;
            if metadata.file_type().is_symlink() {
                return Err(ResidentUnitError::UnsafeCgroup);
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    check_context(context)?;
    Ok(members)
}

fn resolve_cgroup_directory(
    root: &Path,
    cgroup: &str,
) -> Result<Option<PathBuf>, ResidentUnitError> {
    let mut current = root.to_owned();
    for part in cgroup.trim_start_matches('/').split('/') {
        current.push(part);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(ResidentUnitError::UnsafeCgroup),
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(ResidentUnitError::UnsafeCgroup);
        }
    }
    Ok(Some(current))
}

fn cgroup_path_is_within(candidate: &str, root: &str) -> bool {
    candidate == root
        || candidate
            .strip_prefix(root)
            .is_some_and(|tail| tail.starts_with('/'))
}

fn read_process_identity(root: &Path, pid: u32, expected_uid: u32) -> Option<ProcessIdentity> {
    if pid <= 1 || pid > i32::MAX as u32 {
        return None;
    }
    let directory = root.join(pid.to_string());
    let before = fs::symlink_metadata(&directory).ok()?;
    if before.file_type().is_symlink() || !before.is_dir() || before.uid() != expected_uid {
        return None;
    }
    let stat_path = directory.join("stat");
    let stat_bytes = read_bounded_regular(&stat_path, PROC_FILE_BYTES).ok()?;
    let start_ticks = parse_proc_start_ticks(&stat_bytes)?;
    let cgroup_path = directory.join("cgroup");
    let cgroup_bytes = read_bounded_regular(&cgroup_path, PROC_FILE_BYTES).ok()?;
    let cgroup = parse_proc_cgroup(&cgroup_bytes)?;
    let after = fs::symlink_metadata(&directory).ok()?;
    if after.file_type().is_symlink()
        || !after.is_dir()
        || after.uid() != expected_uid
        || (before.dev(), before.ino()) != (after.dev(), after.ino())
    {
        return None;
    }
    let final_ticks =
        parse_proc_start_ticks(&read_bounded_regular(&stat_path, PROC_FILE_BYTES).ok()?)?;
    if start_ticks != final_ticks {
        return None;
    }
    Some(ProcessIdentity {
        start_ticks,
        uid: expected_uid,
        cgroup,
    })
}

fn parse_proc_start_ticks(bytes: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(bytes).ok()?;
    let close = text.rfind(')')?;
    let fields: Vec<_> = text[close + 1..].split_ascii_whitespace().collect();
    fields.get(19)?.parse().ok()
}

fn parse_proc_cgroup(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut path = None;
    for line in text.lines() {
        let (hierarchy, rest) = line.split_once(':')?;
        let (controllers, candidate) = rest.split_once(':')?;
        if hierarchy == "0" && controllers.is_empty() {
            if path.is_some() || !valid_cgroup_text(candidate) {
                return None;
            }
            path = Some(candidate.to_owned());
        } else {
            return None;
        }
    }
    path
}

fn read_bounded_regular(path: &Path, limit: usize) -> Result<Vec<u8>, ResidentUnitError> {
    let before = fs::symlink_metadata(path).map_err(|_| ResidentUnitError::UnsafeFile)?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > limit as u64 {
        return Err(ResidentUnitError::UnsafeFile);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| ResidentUnitError::UnsafeFile)?;
    let opened = file.metadata().map_err(|_| ResidentUnitError::UnsafeFile)?;
    if !opened.is_file()
        || (before.dev(), before.ino()) != (opened.dev(), opened.ino())
        || opened.len() > limit as u64
    {
        return Err(ResidentUnitError::UnsafeFile);
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    let mut bounded: Take<&File> = (&file).take(limit as u64 + 1);
    bounded
        .read_to_end(&mut bytes)
        .map_err(|_| ResidentUnitError::UnsafeFile)?;
    if bytes.len() > limit {
        return Err(ResidentUnitError::UnsafeFile);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::runtime::NeverCancel;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};
    use tempfile::{tempdir, TempDir};

    fn context() -> OperationContext<'static> {
        static NEVER: NeverCancel = NeverCancel;
        OperationContext::new(Instant::now() + Duration::from_secs(10), &NEVER)
    }

    fn properties(overrides: &[(&str, &str)]) -> Vec<u8> {
        let mut values = BTreeMap::from([
            ("Id", UNIT),
            ("LoadState", "loaded"),
            ("ActiveState", "inactive"),
            ("SubState", "dead"),
            ("MainPID", "0"),
            ("InvocationID", ""),
            (
                "ControlGroup",
                "/user.slice/user-1000.slice/user@1000.service/app.slice/llama-swap.service",
            ),
            ("KillMode", "control-group"),
            ("SendSIGKILL", "yes"),
            ("Restart", "on-failure"),
            ("UnitFileState", "enabled"),
            ("FragmentPath", "/tmp/llama-swap.service"),
            ("DropInPaths", ""),
            (
                "ExecStart",
                "{ path=/bin/sh ; argv[]=/bin/sh -c /router ; ignore_errors=no }",
            ),
        ]);
        for (key, value) in overrides {
            values.insert(key, value);
        }
        PROPERTIES
            .iter()
            .map(|key| format!("{key}={}\n", values[key]))
            .collect::<String>()
            .into_bytes()
    }

    const INITIAL_INVOCATION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const RESTORED_INVOCATION: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const FIXTURE_CGROUP_TAIL: &str = "app.slice/llama-swap.service";

    struct UnitFixture {
        _directory: TempDir,
        systemctl: PathBuf,
        cgroup_root: PathBuf,
        proc_root: PathBuf,
        router_config: PathBuf,
        fragment: PathBuf,
        actions: PathBuf,
        cgroup_procs: PathBuf,
        cgroup: String,
    }

    impl UnitFixture {
        fn new(active: bool, fail_stop: bool) -> Self {
            let directory = tempdir().unwrap();
            let base = directory.path().canonicalize().unwrap();
            let root = base.join("stack");
            fs::create_dir_all(&root).unwrap();
            let router_config = root.join("llama-swap.yaml");
            write_private_file(&router_config, b"models: {}\n");

            let fragment = base.join("llama-swap.service");
            let fragment_text = format!(
                "[Service]\nEnvironment=L3MS_ROOT={}\nExecStart=/bin/sh -c '${{LLAMA_SWAP_BIN}} -config ${{L3MS_ROOT}}/llama-swap.yaml'\nKillMode=control-group\n",
                root.display()
            );
            write_private_file(&fragment, fragment_text.as_bytes());

            let uid = unsafe { libc::geteuid() };
            let cgroup =
                format!("/user.slice/user-{uid}.slice/user@{uid}.service/{FIXTURE_CGROUP_TAIL}");
            let cgroup_root = base.join("cgroups");
            let cgroup_directory = cgroup_root.join(cgroup.trim_start_matches('/'));
            fs::create_dir_all(&cgroup_directory).unwrap();
            let cgroup_procs = cgroup_directory.join("cgroup.procs");
            fs::write(&cgroup_procs, if active { "1101\n1102\n" } else { "" }).unwrap();

            let proc_root = base.join("proc");
            fs::create_dir_all(&proc_root).unwrap();
            if active {
                write_process(&proc_root, &cgroup, 1101, 101, &shell_args());
                write_process(&proc_root, &cgroup, 1102, 102, &router_args(&router_config));
            }
            write_process(&proc_root, &cgroup, 2201, 201, &shell_args());
            write_process(&proc_root, &cgroup, 2202, 202, &router_args(&router_config));

            let state = base.join("manager-state");
            let inactive = base.join("manager-inactive");
            let restored = base.join("manager-restored");
            let active_state =
                fixture_properties(true, 1101, INITIAL_INVOCATION, &cgroup, &fragment);
            let inactive_state = fixture_properties(false, 0, INITIAL_INVOCATION, "", &fragment);
            let restored_state =
                fixture_properties(true, 2201, RESTORED_INVOCATION, &cgroup, &fragment);
            let initial_state = if active {
                active_state
            } else {
                fixture_properties(false, 0, "", "", &fragment)
            };
            write_private_file(&state, &initial_state);
            write_private_file(&inactive, &inactive_state);
            write_private_file(&restored, &restored_state);

            let actions = base.join("actions");
            fs::write(&actions, b"").unwrap();
            let fail_stop_path = base.join("fail-stop");
            if fail_stop {
                fs::write(&fail_stop_path, b"fail").unwrap();
            }
            let script = format!(
                "#!/bin/sh\ncase \"$4\" in\n show)\n  if [ \"$5\" = \"--property=Environment\" ]; then printf '%s\\n' 'Environment=L3MS_ROOT={}'; else /bin/cat '{}'; fi\n  ;;\n stop)\n  printf 'stop\\n' >> '{}';\n  if [ -e '{}' ]; then exit 1; fi\n  /bin/cp '{}' '{}'; : > '{}';\n  ;;\n start)\n  printf 'start\\n' >> '{}';\n  /bin/cp '{}' '{}'; printf '2201\\n2202\\n' > '{}';\n  ;;\n *) exit 9 ;;\nesac\n",
                root.display(),
                state.display(),
                actions.display(),
                fail_stop_path.display(),
                inactive.display(),
                state.display(),
                cgroup_procs.display(),
                actions.display(),
                restored.display(),
                state.display(),
                cgroup_procs.display(),
            );
            let systemctl = base.join("systemctl-fixture");
            write_executable(&systemctl, script.as_bytes());

            Self {
                _directory: directory,
                systemctl,
                cgroup_root,
                proc_root,
                router_config,
                fragment,
                actions,
                cgroup_procs,
                cgroup,
            }
        }

        fn capture(&self) -> Result<ResidentUnit, ResidentUnitError> {
            ResidentUnit::capture_with_paths(
                &self.router_config,
                &self.systemctl,
                &self.cgroup_root,
                &self.proc_root,
                &context(),
                true,
            )
        }

        fn add_model_child(&self, pid: u32) {
            write_process(
                &self.proc_root,
                &self.cgroup,
                pid,
                303,
                &[
                    "/opt/llama-server".to_owned(),
                    "-m".to_owned(),
                    "/models/ready.gguf".to_owned(),
                ],
            );
            fs::write(&self.cgroup_procs, b"2201\n2202\n3303\n").unwrap();
        }

        fn actions(&self) -> String {
            fs::read_to_string(&self.actions).unwrap()
        }
    }

    fn fixture_properties(
        active: bool,
        main_pid: u32,
        invocation: &str,
        cgroup: &str,
        fragment: &Path,
    ) -> Vec<u8> {
        let active_state = if active { "active" } else { "inactive" };
        let sub_state = if active { "running" } else { "dead" };
        let values = BTreeMap::from([
            ("Id", UNIT.to_owned()),
            ("LoadState", "loaded".to_owned()),
            ("ActiveState", active_state.to_owned()),
            ("SubState", sub_state.to_owned()),
            ("MainPID", main_pid.to_string()),
            ("InvocationID", invocation.to_owned()),
            ("ControlGroup", cgroup.to_owned()),
            ("KillMode", "control-group".to_owned()),
            ("SendSIGKILL", "yes".to_owned()),
            ("Restart", "on-failure".to_owned()),
            ("UnitFileState", "enabled".to_owned()),
            ("FragmentPath", fragment.display().to_string()),
            ("DropInPaths", String::new()),
            (
                "ExecStart",
                "{ path=/bin/sh ; argv[]=/bin/sh -c ${LLAMA_SWAP_BIN} -config ${L3MS_ROOT}/llama-swap.yaml ; ignore_errors=no }".to_owned(),
            ),
        ]);
        PROPERTIES
            .iter()
            .map(|key| format!("{key}={}\n", values[key]))
            .collect::<String>()
            .into_bytes()
    }

    fn shell_args() -> Vec<String> {
        vec![
            "/bin/sh".to_owned(),
            "-c".to_owned(),
            "${LLAMA_SWAP_BIN} -config ${L3MS_ROOT}/llama-swap.yaml".to_owned(),
        ]
    }

    fn router_args(router_config: &Path) -> Vec<String> {
        vec![
            "/opt/llama-swap".to_owned(),
            "-config".to_owned(),
            router_config.to_string_lossy().into_owned(),
        ]
    }

    fn write_process(root: &Path, cgroup: &str, pid: u32, ticks: u64, args: &[String]) {
        let directory = root.join(pid.to_string());
        fs::create_dir_all(&directory).unwrap();
        let mut stat = format!("{pid} (fixture process) S").into_bytes();
        for index in 0..20 {
            let value = if index == 18 {
                ticks.to_string()
            } else {
                "1".to_owned()
            };
            stat.extend_from_slice(format!(" {value}").as_bytes());
        }
        stat.push(b'\n');
        fs::write(directory.join("stat"), stat).unwrap();
        fs::write(
            directory.join("cgroup"),
            format!("0::{cgroup}\n").as_bytes(),
        )
        .unwrap();
        let mut cmdline = args.join("\0").into_bytes();
        cmdline.push(0);
        fs::write(directory.join("cmdline"), cmdline).unwrap();
    }

    fn write_private_file(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }

    fn write_executable(path: &Path, bytes: &[u8]) {
        {
            let _guard = process::executable_fixture_guard();
            fs::write(path, bytes).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }
    }

    #[test]
    fn manager_reply_requires_exact_unique_property_set_and_safe_state() {
        assert!(parse_manager_snapshot(&properties(&[])).is_ok());
        let mut duplicate = properties(&[]);
        duplicate.extend_from_slice(b"Id=llama-swap.service\n");
        assert!(matches!(
            parse_manager_snapshot(&duplicate).err(),
            Some(ResidentUnitError::InvalidManagerReply)
        ));
        assert!(parse_manager_snapshot(&properties(&[("KillMode", "process")])).is_err());
        assert!(
            parse_manager_snapshot(&properties(&[("DropInPaths", "/etc/a\\x20b.conf")])).is_err()
        );
    }

    #[test]
    fn config_snapshot_binds_file_contents_and_metadata() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("router.json");
        fs::write(&path, b"{}\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let uid = unsafe { libc::geteuid() };
        let context = context();
        let original = read_file_snapshot(&path, uid, true, &context).unwrap();
        fs::write(&path, b"{\"models\":[]}\n").unwrap();
        let changed = read_file_snapshot(&path, uid, true, &context).unwrap();
        assert!(original != changed);
    }

    #[test]
    fn process_parsers_use_birth_identity_and_exact_v2_membership() {
        let mut stat = b"123 (fake process name) S".to_vec();
        for index in 0..20 {
            stat.extend_from_slice(
                format!(" {}", if index == 18 { "987654" } else { "1" }).as_bytes(),
            );
        }
        stat.push(b'\n');
        assert_eq!(parse_proc_start_ticks(&stat), Some(987654));
        assert_eq!(
            parse_proc_cgroup(b"0::/user.slice/test\n"),
            Some("/user.slice/test".to_owned())
        );
        assert_eq!(parse_proc_cgroup(b"2:cpu:/bad\n0::/good\n"), None);
    }

    #[test]
    fn inactive_capture_helpers_do_not_treat_empty_cgroup_as_a_start_receipt() {
        let snapshot = parse_manager_snapshot(&properties(&[])).unwrap();
        require_inactive(&snapshot).unwrap();
        let active = parse_manager_snapshot(&properties(&[
            ("ActiveState", "active"),
            ("SubState", "running"),
            ("MainPID", "42"),
            ("InvocationID", "0123456789abcdef0123456789abcdef"),
        ]));
        assert!(active.is_ok());
    }

    #[test]
    fn active_unit_stops_restores_and_refreshes_ready_child_membership() {
        let fixture = UnitFixture::new(true, false);
        let mut unit = fixture.capture().unwrap();
        assert!(unit.was_active());
        assert!(unit.accepts_claimant(1102));

        unit.pause(&context()).unwrap();
        assert!(!unit.accepts_claimant(1102));
        unit.restore(&context()).unwrap();
        unit.verify_restored(&context()).unwrap();
        assert!(unit.accepts_claimant(2202));
        assert!(!unit.accepts_claimant(1102));

        fixture.add_model_child(3303);
        unit.verify_restored(&context()).unwrap();
        assert!(unit.accepts_claimant(3303));
        assert_eq!(fixture.actions(), "stop\nstart\n");
    }

    #[test]
    fn inactive_empty_cgroup_is_left_untouched() {
        let fixture = UnitFixture::new(false, false);
        let mut unit = fixture.capture().unwrap();
        assert!(!unit.was_active());
        unit.pause(&context()).unwrap();
        unit.restore(&context()).unwrap();
        unit.verify_restored(&context()).unwrap();
        assert!(fixture.actions().is_empty());
    }

    #[test]
    fn failed_stop_is_attempted_only_once() {
        let fixture = UnitFixture::new(true, true);
        let mut unit = fixture.capture().unwrap();
        assert_eq!(
            unit.pause(&context()),
            Err(ResidentUnitError::StopUnconfirmed)
        );
        assert_eq!(
            unit.pause(&context()),
            Err(ResidentUnitError::OperationAlreadyAttempted)
        );
        assert_eq!(fixture.actions(), "stop\n");
    }

    #[test]
    fn changed_definition_refuses_stop_before_manager_action() {
        let fixture = UnitFixture::new(true, false);
        let mut unit = fixture.capture().unwrap();
        fs::write(&fixture.fragment, b"[Service]\n# changed after capture\n").unwrap();
        assert_eq!(
            unit.pause(&context()),
            Err(ResidentUnitError::DefinitionChanged)
        );
        assert!(fixture.actions().is_empty());
    }
}
