//! Narrow, in-memory checks for the fixed user services used by local GPU control.
//!
//! The llama-swap adapter is bound to one unit and caller-owned router config;
//! the desktop baseline only observes one fixed KDE compositor unit. Neither is
//! a generic service manager or a crash-recovery journal. The llama-swap checks
//! assume same-user systemd clients and config writers are quiesced throughout.

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Take};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use super::process::{self, ProcessLimits};
use super::runtime::{OperationContext, RecoveryPolicy};

const SYSTEMCTL: &str = "/usr/bin/systemctl";
const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const MAX_MANAGER_BYTES: usize = 32 * 1024;
const MAX_FILE_BYTES: u64 = 1024 * 1024;
const MAX_CGROUP_DIRS: usize = 256;
const MAX_CGROUP_PIDS: usize = 4096;
const MAX_CGROUP_PROCS_BYTES: usize = 64 * 1024;
const PROC_FILE_BYTES: usize = 16 * 1024;
const MANAGER_RECOVERY: std::time::Duration = std::time::Duration::from_secs(5);
const DESKTOP_UNIT: &str = "plasma-kwin_wayland.service";
const DESKTOP_COMPOSITOR: &str = "/usr/bin/kwin_wayland";
const DESKTOP_LAUNCHER: &str = "/usr/bin/kwin_wayland_wrapper";
const DESKTOP_FRAGMENT: &str = "/usr/lib/systemd/user/plasma-kwin_wayland.service";

const DESKTOP_PROPERTIES: [&str; 10] = [
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "InvocationID",
    "ControlGroup",
    "FragmentPath",
    "DropInPaths",
    "ExecStart",
];

/// Fixed, redacted failures. No manager output, paths, PID, file contents, or
/// syscall message is retained in an error value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DesktopGpuError {
    InvalidInput,
    ContextUnavailable,
    ManagerUnavailable,
    InvalidManagerReply,
    UnitNotSupported,
    DefinitionChanged,
    UnsafeFile,
    UnsafeCgroup,
    ProcessIdentityUnavailable,
}

impl fmt::Display for DesktopGpuError {
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
        })
    }
}

impl Error for DesktopGpuError {}

#[derive(Clone, Eq, PartialEq)]
struct ProcessIdentity {
    start_ticks: u64,
    uid: u32,
    cgroup: String,
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

/// Opaque, read-only snapshot of the captured KDE launcher's compositor child.
///
/// This baseline never changes the desktop unit. It accepts only the exact
/// compositor PID captured from the fixed unit's verified cgroup. It is only
/// a point-in-time observation, not a GPU admission fence, lease, or ownership
/// proof. Callers must verify the original generation and full member set,
/// own service-cgroup cleanup, and exclude direct device bypass before
/// switching workloads. It inspects only this unit, not all graphics
/// processes; its bounded manager query is still a synchronous subprocess.
/// This trusts the user manager and launcher environment, not hostile same-UID
/// processes. The child's protected executable link is not attested; argv[0]
/// only selects a direct child of the verified packaged launcher.
pub(crate) struct DesktopGpuBaseline {
    systemctl: PathBuf,
    cgroup_root: PathBuf,
    proc_root: PathBuf,
    fragment_path: PathBuf,
    uid: u32,
    generation: DesktopGeneration,
    definition: DesktopUnitDefinition,
    members: BTreeMap<u32, ProcessIdentity>,
    compositor_pid: u32,
    compositor_identity: ProcessIdentity,
}

#[derive(Clone, Eq, PartialEq)]
struct DesktopGeneration {
    invocation_id: String,
    main_pid: u32,
    main_identity: ProcessIdentity,
    cgroup: String,
}

#[derive(Clone, Eq, PartialEq)]
struct DesktopUnitDefinition {
    exec_start: String,
    files: Vec<FileSnapshot>,
}

struct DesktopManagerSnapshot {
    main_pid: u32,
    invocation_id: String,
    cgroup: String,
    exec_start: String,
}

impl DesktopGpuBaseline {
    /// Capture the currently running packaged KDE launcher's compositor child.
    ///
    /// This is a point-in-time read only. It is not a GPU admission fence,
    /// lease, or proof that no process can begin using the device later.
    pub(crate) fn capture(context: &OperationContext<'_>) -> Result<Self, DesktopGpuError> {
        Self::capture_at(
            Path::new(SYSTEMCTL),
            Path::new(CGROUP_ROOT),
            Path::new("/proc"),
            Path::new(DESKTOP_FRAGMENT),
            context,
            false,
        )
    }

    #[cfg(test)]
    fn capture_with_paths(
        systemctl: &Path,
        cgroup_root: &Path,
        proc_root: &Path,
        fragment: &Path,
        context: &OperationContext<'_>,
    ) -> Result<Self, DesktopGpuError> {
        Self::capture_at(systemctl, cgroup_root, proc_root, fragment, context, true)
    }

    fn capture_at(
        systemctl: &Path,
        cgroup_root: &Path,
        proc_root: &Path,
        fragment: &Path,
        context: &OperationContext<'_>,
        test_paths: bool,
    ) -> Result<Self, DesktopGpuError> {
        check_context(context)?;
        let uid = unsafe { libc::geteuid() };
        validate_cgroup_root(cgroup_root, !test_paths)?;

        let manager = show_desktop(systemctl, fragment, uid, test_paths, context)?;
        let definition = capture_desktop_definition(&manager, fragment, uid, test_paths, context)?;
        let members = scan_members(cgroup_root, &manager.cgroup, proc_root, uid, context)?;
        let main_identity = members
            .get(&manager.main_pid)
            .cloned()
            .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
        verify_desktop_launcher(proc_root, manager.main_pid, test_paths)?;
        let compositor =
            find_desktop_compositor(&members, proc_root, uid, manager.main_pid, context)?;
        let baseline = Self {
            systemctl: systemctl.to_owned(),
            cgroup_root: cgroup_root.to_owned(),
            proc_root: proc_root.to_owned(),
            fragment_path: fragment.to_owned(),
            uid,
            generation: DesktopGeneration {
                invocation_id: manager.invocation_id,
                main_pid: manager.main_pid,
                main_identity,
                cgroup: manager.cgroup,
            },
            definition,
            members,
            compositor_pid: compositor.0,
            compositor_identity: compositor.1,
        };
        baseline.verify_at(context, test_paths)?;
        Ok(baseline)
    }

    /// Recheck that the captured desktop generation, files, and exact full
    /// cgroup membership set have not changed.
    pub(crate) fn verify(&self, context: &OperationContext<'_>) -> Result<(), DesktopGpuError> {
        self.verify_at(context, false)
    }

    fn verify_at(
        &self,
        context: &OperationContext<'_>,
        test_paths: bool,
    ) -> Result<(), DesktopGpuError> {
        check_context(context)?;
        let manager = show_desktop(
            &self.systemctl,
            &self.fragment_path,
            self.uid,
            test_paths,
            context,
        )?;
        if manager.main_pid != self.generation.main_pid
            || manager.invocation_id != self.generation.invocation_id
            || manager.cgroup != self.generation.cgroup
        {
            return Err(DesktopGpuError::DefinitionChanged);
        }
        let main_identity = read_process_identity(&self.proc_root, manager.main_pid, self.uid)
            .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
        if main_identity != self.generation.main_identity {
            return Err(DesktopGpuError::ProcessIdentityUnavailable);
        }
        verify_desktop_launcher(&self.proc_root, manager.main_pid, test_paths)?;
        let definition = capture_desktop_definition(
            &manager,
            &self.fragment_path,
            self.uid,
            test_paths,
            context,
        )?;
        if definition != self.definition {
            return Err(DesktopGpuError::DefinitionChanged);
        }
        let members = scan_members(
            &self.cgroup_root,
            &manager.cgroup,
            &self.proc_root,
            self.uid,
            context,
        )?;
        if members != self.members {
            return Err(DesktopGpuError::ProcessIdentityUnavailable);
        }
        let compositor = find_desktop_compositor(
            &members,
            &self.proc_root,
            self.uid,
            manager.main_pid,
            context,
        )?;
        if compositor.0 != self.compositor_pid || compositor.1 != self.compositor_identity {
            return Err(DesktopGpuError::ProcessIdentityUnavailable);
        }
        check_context(context)
    }

    /// Accept only the exact captured compositor child, not other
    /// members of the KDE unit. Call `verify` before relying on this snapshot.
    pub(crate) fn accepts_claimant(&self, pid: u32) -> bool {
        if pid != self.compositor_pid
            || read_process_identity(&self.proc_root, self.generation.main_pid, self.uid).as_ref()
                != Some(&self.generation.main_identity)
            || self.members.get(&pid) != Some(&self.compositor_identity)
            || read_process_identity(&self.proc_root, pid, self.uid).as_ref()
                != Some(&self.compositor_identity)
        {
            return false;
        }
        desktop_child_matches(&self.proc_root, pid, self.generation.main_pid).unwrap_or(false)
    }
}

fn show_desktop(
    systemctl: &Path,
    fragment: &Path,
    uid: u32,
    test_paths: bool,
    context: &OperationContext<'_>,
) -> Result<DesktopManagerSnapshot, DesktopGpuError> {
    check_context(context)?;
    validate_executable(systemctl, test_paths)?;
    let mut command = Command::new(systemctl);
    command
        .arg("--user")
        .arg("--no-pager")
        .arg("--no-ask-password")
        .arg("show")
        .arg("--all");
    for property in DESKTOP_PROPERTIES {
        command.arg(format!("--property={property}"));
    }
    command.arg(DESKTOP_UNIT);
    let limits = ProcessLimits::new(0, MAX_MANAGER_BYTES)
        .map_err(|_| DesktopGpuError::ManagerUnavailable)?;
    let recovery =
        RecoveryPolicy::new(MANAGER_RECOVERY).map_err(|_| DesktopGpuError::ManagerUnavailable)?;
    let output = process::execute(command, None, limits, context, recovery)
        .map_err(|_| DesktopGpuError::ManagerUnavailable)?;
    if !output.stderr().iter().all(u8::is_ascii_whitespace) {
        return Err(DesktopGpuError::InvalidManagerReply);
    }
    let snapshot = parse_desktop_snapshot(output.stdout(), fragment, uid)?;
    drop(output);
    check_context(context)?;
    Ok(snapshot)
}

fn parse_desktop_snapshot(
    bytes: &[u8],
    expected_fragment: &Path,
    uid: u32,
) -> Result<DesktopManagerSnapshot, DesktopGpuError> {
    if bytes.len() > MAX_MANAGER_BYTES {
        return Err(DesktopGpuError::InvalidManagerReply);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| DesktopGpuError::InvalidManagerReply)?;
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .ok_or(DesktopGpuError::InvalidManagerReply)?;
        if !DESKTOP_PROPERTIES.contains(&key) || values.insert(key, value).is_some() {
            return Err(DesktopGpuError::InvalidManagerReply);
        }
    }
    if values.len() != DESKTOP_PROPERTIES.len() {
        return Err(DesktopGpuError::InvalidManagerReply);
    }
    let get = |key: &str| {
        values
            .get(key)
            .copied()
            .ok_or(DesktopGpuError::InvalidManagerReply)
    };
    let main_pid = get("MainPID")?
        .parse::<u32>()
        .map_err(|_| DesktopGpuError::InvalidManagerReply)?;
    let invocation_id = get("InvocationID")?.to_owned();
    let cgroup = get("ControlGroup")?.to_owned();
    let fragment_path = PathBuf::from(get("FragmentPath")?);
    let raw_exec_start = get("ExecStart")?;
    let exec_start = normalize_exec_start(raw_exec_start)?;
    if get("Id")? != DESKTOP_UNIT
        || get("LoadState")? != "loaded"
        || get("ActiveState")? != "active"
        || get("SubState")? != "running"
        || main_pid <= 1
        || !valid_invocation_id(&invocation_id)
        || !valid_cgroup_text(&cgroup)
        || !valid_desktop_cgroup(&cgroup, uid)
        || !safe_plain_path(&fragment_path, false)
        || fragment_path != expected_fragment
        || !get("DropInPaths")?.is_empty()
        || !desktop_exec_start_matches(raw_exec_start)
    {
        return Err(DesktopGpuError::UnitNotSupported);
    }
    Ok(DesktopManagerSnapshot {
        main_pid,
        invocation_id,
        cgroup,
        exec_start,
    })
}

fn valid_desktop_cgroup(path: &str, uid: u32) -> bool {
    let prefix = format!("/user.slice/user-{uid}.slice/user@{uid}.service/");
    valid_cgroup_text(path)
        && path.starts_with(&prefix)
        && path.ends_with("/session.slice/plasma-kwin_wayland.service")
}

fn desktop_exec_start_matches(raw: &str) -> bool {
    let Some(body) = raw
        .strip_prefix("{ ")
        .and_then(|value| value.strip_suffix(" }"))
    else {
        return false;
    };
    let mut path_count = 0;
    let mut argv_count = 0;
    for field in body.split(';') {
        let field = field.trim();
        if let Some(path) = field.strip_prefix("path=") {
            path_count += 1;
            if path != DESKTOP_LAUNCHER {
                return false;
            }
        } else if let Some(argv) = field.strip_prefix("argv[]=") {
            argv_count += 1;
            if argv.split_ascii_whitespace().next() != Some(DESKTOP_LAUNCHER) {
                return false;
            }
        }
    }
    path_count == 1 && argv_count == 1
}

fn capture_desktop_definition(
    manager: &DesktopManagerSnapshot,
    fragment: &Path,
    uid: u32,
    test_paths: bool,
    context: &OperationContext<'_>,
) -> Result<DesktopUnitDefinition, DesktopGpuError> {
    let file = read_file_snapshot(fragment, uid, false, context)?;
    if !test_paths && file.uid != 0 {
        return Err(DesktopGpuError::UnsafeFile);
    }
    if !unit_file_exec_start_matches(&file.bytes) {
        return Err(DesktopGpuError::UnitNotSupported);
    }
    Ok(DesktopUnitDefinition {
        exec_start: manager.exec_start.clone(),
        files: vec![file],
    })
}

fn unit_file_exec_start_matches(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let mut in_service = false;
    let mut count = 0;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_service = &line[1..line.len() - 1] == "Service";
            continue;
        }
        if !in_service {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.starts_with("ExecStart") {
            if key != "ExecStart" || value.split_ascii_whitespace().next() != Some(DESKTOP_LAUNCHER)
            {
                return false;
            }
            count += 1;
        }
    }
    count == 1
}

fn find_desktop_compositor(
    members: &BTreeMap<u32, ProcessIdentity>,
    proc_root: &Path,
    uid: u32,
    launcher_pid: u32,
    context: &OperationContext<'_>,
) -> Result<(u32, ProcessIdentity), DesktopGpuError> {
    let mut found = None;
    for (&pid, identity) in members {
        check_context(context)?;
        let before = read_process_identity(proc_root, pid, uid)
            .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
        if before != *identity {
            return Err(DesktopGpuError::ProcessIdentityUnavailable);
        }
        let is_compositor = desktop_child_matches(proc_root, pid, launcher_pid)?;
        let after = read_process_identity(proc_root, pid, uid)
            .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
        if after != before {
            return Err(DesktopGpuError::ProcessIdentityUnavailable);
        }
        if is_compositor {
            if found.is_some() {
                return Err(DesktopGpuError::UnitNotSupported);
            }
            found = Some((pid, after));
        }
    }
    check_context(context)?;
    found.ok_or(DesktopGpuError::UnitNotSupported)
}

fn verify_desktop_launcher(
    proc_root: &Path,
    pid: u32,
    test_paths: bool,
) -> Result<(), DesktopGpuError> {
    let executable = proc_root.join(pid.to_string()).join("exe");
    if fs::read_link(&executable).ok().as_deref() != Some(Path::new(DESKTOP_LAUNCHER)) {
        return Err(DesktopGpuError::ProcessIdentityUnavailable);
    }
    if !test_paths {
        let installed = fs::symlink_metadata(DESKTOP_LAUNCHER)
            .map_err(|_| DesktopGpuError::ProcessIdentityUnavailable)?;
        let running =
            fs::metadata(executable).map_err(|_| DesktopGpuError::ProcessIdentityUnavailable)?;
        if !installed.is_file()
            || installed.uid() != 0
            || installed.mode() & 0o022 != 0
            || installed.mode() & 0o111 == 0
            || (installed.dev(), installed.ino()) != (running.dev(), running.ino())
        {
            return Err(DesktopGpuError::ProcessIdentityUnavailable);
        }
    }
    Ok(())
}

fn desktop_child_matches(
    proc_root: &Path,
    pid: u32,
    launcher_pid: u32,
) -> Result<bool, DesktopGpuError> {
    let bytes = read_bounded_regular(
        &proc_root.join(pid.to_string()).join("cmdline"),
        PROC_FILE_BYTES,
    )
    .map_err(|_| DesktopGpuError::ProcessIdentityUnavailable)?;
    let first = bytes
        .split(|byte| *byte == 0)
        .next()
        .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
    if first.is_empty() || !bytes.contains(&0) {
        return Err(DesktopGpuError::ProcessIdentityUnavailable);
    }
    if first != DESKTOP_COMPOSITOR.as_bytes() {
        return Ok(false);
    }
    let stat = read_bounded_regular(
        &proc_root.join(pid.to_string()).join("stat"),
        PROC_FILE_BYTES,
    )
    .map_err(|_| DesktopGpuError::ProcessIdentityUnavailable)?;
    let text =
        std::str::from_utf8(&stat).map_err(|_| DesktopGpuError::ProcessIdentityUnavailable)?;
    let close = text
        .rfind(')')
        .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
    let parent = text[close + 1..]
        .split_ascii_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
    Ok(parent == launcher_pid)
}

fn check_context(context: &OperationContext<'_>) -> Result<(), DesktopGpuError> {
    context
        .check_active("resident unit operation")
        .map_err(|_| DesktopGpuError::ContextUnavailable)
}

fn validate_executable(path: &Path, test_paths: bool) -> Result<(), DesktopGpuError> {
    if !path.is_absolute() || !test_paths && path != Path::new(SYSTEMCTL) {
        return Err(DesktopGpuError::InvalidInput);
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| DesktopGpuError::ManagerUnavailable)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.mode() & 0o111 == 0
        || !test_paths && (metadata.uid() != 0 || metadata.mode() & 0o022 != 0)
    {
        return Err(DesktopGpuError::ManagerUnavailable);
    }
    Ok(())
}

fn validate_cgroup_root(path: &Path, require_cgroup2: bool) -> Result<(), DesktopGpuError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| DesktopGpuError::UnsafeCgroup)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || path
            .canonicalize()
            .map_err(|_| DesktopGpuError::UnsafeCgroup)?
            != path
    {
        return Err(DesktopGpuError::UnsafeCgroup);
    }
    if require_cgroup2 {
        let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
        let file = File::open(path).map_err(|_| DesktopGpuError::UnsafeCgroup)?;
        if unsafe { libc::fstatfs(std::os::fd::AsRawFd::as_raw_fd(&file), stat.as_mut_ptr()) } != 0
            || unsafe { stat.assume_init() }.f_type != 0x6367_7270
        {
            return Err(DesktopGpuError::UnsafeCgroup);
        }
    }
    Ok(())
}

fn normalize_exec_start(raw: &str) -> Result<String, DesktopGpuError> {
    if raw.is_empty() || raw.chars().any(char::is_control) {
        return Err(DesktopGpuError::InvalidManagerReply);
    }
    let normalized = raw.split(" ; start_time=").next().unwrap_or(raw);
    if !normalized.contains("path=") || !normalized.contains("argv[]=") {
        return Err(DesktopGpuError::InvalidManagerReply);
    }
    Ok(normalized.to_owned())
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
) -> Result<FileSnapshot, DesktopGpuError> {
    check_context(context)?;
    if !safe_plain_path(path, false) {
        return Err(DesktopGpuError::UnsafeFile);
    }
    if path
        .canonicalize()
        .map_err(|_| DesktopGpuError::UnsafeFile)?
        != path
    {
        return Err(DesktopGpuError::UnsafeFile);
    }
    let before = fs::symlink_metadata(path).map_err(|_| DesktopGpuError::UnsafeFile)?;
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
        return Err(DesktopGpuError::UnsafeFile);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| DesktopGpuError::UnsafeFile)?;
    let opened = file.metadata().map_err(|_| DesktopGpuError::UnsafeFile)?;
    if !opened.is_file()
        || (opened.dev(), opened.ino()) != (before.dev(), before.ino())
        || opened.uid() != before.uid()
        || opened.mode() != before.mode()
        || opened.nlink() != before.nlink()
        || opened.len() != before.len()
    {
        return Err(DesktopGpuError::UnsafeFile);
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    check_context(context)?;
    let mut bounded: Take<&File> = (&file).take(MAX_FILE_BYTES + 1);
    bounded
        .read_to_end(&mut bytes)
        .map_err(|_| DesktopGpuError::UnsafeFile)?;
    check_context(context)?;
    let after = file.metadata().map_err(|_| DesktopGpuError::UnsafeFile)?;
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
        return Err(DesktopGpuError::UnsafeFile);
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

fn scan_members(
    root: &Path,
    cgroup: &str,
    proc_root: &Path,
    uid: u32,
    context: &OperationContext<'_>,
) -> Result<BTreeMap<u32, ProcessIdentity>, DesktopGpuError> {
    check_context(context)?;
    if cgroup.is_empty() {
        return Ok(BTreeMap::new());
    }
    if !valid_cgroup_text(cgroup) {
        return Err(DesktopGpuError::UnsafeCgroup);
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
            return Err(DesktopGpuError::UnsafeCgroup);
        }
        let current_meta =
            fs::symlink_metadata(&directory).map_err(|_| DesktopGpuError::UnsafeCgroup)?;
        if current_meta.file_type().is_symlink() || !current_meta.is_dir() {
            return Err(DesktopGpuError::UnsafeCgroup);
        }
        let procs_path = directory.join("cgroup.procs");
        let procs = read_bounded_regular(&procs_path, MAX_CGROUP_PROCS_BYTES)
            .map_err(|_| DesktopGpuError::UnsafeCgroup)?;
        let rel = directory
            .strip_prefix(root)
            .map_err(|_| DesktopGpuError::UnsafeCgroup)?;
        let cgroup_path = format!("/{}", rel.to_string_lossy());
        if !valid_cgroup_text(&cgroup_path) || !cgroup_path_is_within(&cgroup_path, cgroup) {
            return Err(DesktopGpuError::UnsafeCgroup);
        }
        for row in procs.split(|byte| *byte == b'\n') {
            if row.is_empty() {
                continue;
            }
            check_context(context)?;
            let text = std::str::from_utf8(row).map_err(|_| DesktopGpuError::UnsafeCgroup)?;
            let pid = text
                .parse::<u32>()
                .map_err(|_| DesktopGpuError::UnsafeCgroup)?;
            if pid <= 1 || pid > i32::MAX as u32 || members.len() >= MAX_CGROUP_PIDS {
                return Err(DesktopGpuError::UnsafeCgroup);
            }
            let identity = read_process_identity(proc_root, pid, uid)
                .ok_or(DesktopGpuError::ProcessIdentityUnavailable)?;
            if identity.cgroup != cgroup_path || members.insert(pid, identity).is_some() {
                return Err(DesktopGpuError::ProcessIdentityUnavailable);
            }
        }
        let entries = fs::read_dir(&directory).map_err(|_| DesktopGpuError::UnsafeCgroup)?;
        for entry in entries {
            check_context(context)?;
            let entry = entry.map_err(|_| DesktopGpuError::UnsafeCgroup)?;
            let name = entry.file_name();
            if name == OsStr::new("cgroup.procs") || name == OsStr::new("cgroup.threads") {
                continue;
            }
            let metadata =
                fs::symlink_metadata(entry.path()).map_err(|_| DesktopGpuError::UnsafeCgroup)?;
            if metadata.file_type().is_symlink() {
                return Err(DesktopGpuError::UnsafeCgroup);
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    check_context(context)?;
    Ok(members)
}

fn resolve_cgroup_directory(root: &Path, cgroup: &str) -> Result<Option<PathBuf>, DesktopGpuError> {
    let mut current = root.to_owned();
    for part in cgroup.trim_start_matches('/').split('/') {
        current.push(part);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(DesktopGpuError::UnsafeCgroup),
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(DesktopGpuError::UnsafeCgroup);
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

fn read_bounded_regular(path: &Path, limit: usize) -> Result<Vec<u8>, DesktopGpuError> {
    let before = fs::symlink_metadata(path).map_err(|_| DesktopGpuError::UnsafeFile)?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > limit as u64 {
        return Err(DesktopGpuError::UnsafeFile);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| DesktopGpuError::UnsafeFile)?;
    let opened = file.metadata().map_err(|_| DesktopGpuError::UnsafeFile)?;
    if !opened.is_file()
        || (before.dev(), before.ino()) != (opened.dev(), opened.ino())
        || opened.len() > limit as u64
    {
        return Err(DesktopGpuError::UnsafeFile);
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    let mut bounded: Take<&File> = (&file).take(limit as u64 + 1);
    bounded
        .read_to_end(&mut bytes)
        .map_err(|_| DesktopGpuError::UnsafeFile)?;
    if bytes.len() > limit {
        return Err(DesktopGpuError::UnsafeFile);
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

    struct DesktopFixture {
        _directory: TempDir,
        systemctl: PathBuf,
        cgroup_root: PathBuf,
        proc_root: PathBuf,
        fragment: PathBuf,
        manager_state: PathBuf,
        cgroup: String,
    }

    impl DesktopFixture {
        fn capture(&self) -> Result<DesktopGpuBaseline, DesktopGpuError> {
            DesktopGpuBaseline::capture_with_paths(
                &self.systemctl,
                &self.cgroup_root,
                &self.proc_root,
                &self.fragment,
                &context(),
            )
        }

        fn new() -> Self {
            let directory = tempdir().unwrap();
            let base = directory.path().canonicalize().unwrap();
            let uid = unsafe { libc::geteuid() };
            let cgroup = format!(
                "/user.slice/user-{uid}.slice/user@{uid}.service/session.slice/{DESKTOP_UNIT}"
            );
            let cgroup_root = base.join("cgroups");
            let cgroup_directory = cgroup_root.join(cgroup.trim_start_matches('/'));
            fs::create_dir_all(&cgroup_directory).unwrap();
            fs::write(cgroup_directory.join("cgroup.procs"), b"9001\n9002\n").unwrap();

            let proc_root = base.join("proc");
            fs::create_dir_all(&proc_root).unwrap();
            write_process(
                &proc_root,
                &cgroup,
                9001,
                101,
                &["/usr/bin/systemd".to_owned()],
            );
            write_process(
                &proc_root,
                &cgroup,
                9002,
                102,
                &[DESKTOP_COMPOSITOR.to_owned()],
            );
            std::os::unix::fs::symlink(DESKTOP_LAUNCHER, proc_root.join("9001/exe")).unwrap();
            let stat_path = proc_root.join("9002/stat");
            let stat = fs::read_to_string(&stat_path)
                .unwrap()
                .replacen(") S 1 ", ") S 9001 ", 1);
            fs::write(stat_path, stat).unwrap();

            let fragment = base.join("plasma-kwin_wayland.service");
            write_private_file(
                &fragment,
                format!(
                    "[Unit]\nDescription=KDE Wayland compositor\n[Service]\nExecStart={DESKTOP_LAUNCHER}\n"
                )
                .as_bytes(),
            );
            let manager_state = base.join("manager-state");
            write_private_file(
                &manager_state,
                &desktop_fixture_properties(
                    9001,
                    "cccccccccccccccccccccccccccccccc",
                    &cgroup,
                    &fragment,
                ),
            );
            let systemctl = base.join("systemctl-fixture");
            write_executable(
                &systemctl,
                format!("#!/bin/sh\nexec /bin/cat '{}'\n", manager_state.display()).as_bytes(),
            );

            Self {
                _directory: directory,
                systemctl,
                cgroup_root,
                proc_root,
                fragment,
                manager_state,
                cgroup,
            }
        }
    }

    fn desktop_fixture_properties(
        main_pid: u32,
        invocation: &str,
        cgroup: &str,
        fragment: &Path,
    ) -> Vec<u8> {
        let values = BTreeMap::from([
            ("Id", DESKTOP_UNIT.to_owned()),
            ("LoadState", "loaded".to_owned()),
            ("ActiveState", "active".to_owned()),
            ("SubState", "running".to_owned()),
            ("MainPID", main_pid.to_string()),
            ("InvocationID", invocation.to_owned()),
            ("ControlGroup", cgroup.to_owned()),
            ("FragmentPath", fragment.display().to_string()),
            ("DropInPaths", String::new()),
            (
                "ExecStart",
                format!(
                    "{{ path={DESKTOP_LAUNCHER} ; argv[]={DESKTOP_LAUNCHER} ; ignore_errors=no ; start_time=[Mon 2026-10-12 11:12:13 EDT]; stop_time=[n/a]; pid={main_pid}; code=(null); status=0/0 }}"
                ),
            ),
        ]);
        DESKTOP_PROPERTIES
            .iter()
            .map(|key| format!("{key}={}\n", values[key]))
            .collect::<String>()
            .into_bytes()
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
    fn desktop_baseline_rejects_a_named_process_without_the_packaged_launcher() {
        let fixture = DesktopFixture::new();
        // A matching argv[0] in the right cgroup is not executable identity.
        fs::remove_file(fixture.proc_root.join("9001/exe")).unwrap();
        assert!(fixture.capture().is_err());

        std::os::unix::fs::symlink("/usr/bin/false", fixture.proc_root.join("9001/exe")).unwrap();
        assert!(fixture.capture().is_err());
    }

    #[test]
    fn desktop_baseline_rejects_named_process_with_an_unrelated_parent() {
        let fixture = DesktopFixture::new();
        let baseline = fixture.capture().unwrap();
        let stat_path = fixture.proc_root.join("9002/stat");
        let stat = fs::read_to_string(&stat_path)
            .unwrap()
            .replacen(") S 9001 ", ") S 1 ", 1);
        fs::write(stat_path, stat).unwrap();
        assert!(!baseline.accepts_claimant(9002));
        assert!(baseline.verify_at(&context(), true).is_err());
        assert!(fixture.capture().is_err());
    }

    #[test]
    fn desktop_baseline_accepts_only_the_captured_native_compositor() {
        let fixture = DesktopFixture::new();
        let baseline = fixture.capture().unwrap();
        assert!(baseline.accepts_claimant(9002));
        assert!(!baseline.accepts_claimant(9001));
        assert!(!baseline.accepts_claimant(9003));

        let changed_time = fs::read_to_string(&fixture.manager_state)
            .unwrap()
            .replace("Mon 2026-10-12 11:12:13 EDT", "Tue 2026-10-13 12:13:14 EDT");
        write_private_file(&fixture.manager_state, changed_time.as_bytes());
        baseline.verify_at(&context(), true).unwrap();
    }

    #[test]
    fn desktop_baseline_rejects_changed_generation_or_process_identity() {
        let fixture = DesktopFixture::new();
        let baseline = fixture.capture().unwrap();
        write_private_file(
            &fixture.manager_state,
            &desktop_fixture_properties(
                9001,
                "dddddddddddddddddddddddddddddddd",
                &fixture.cgroup,
                &fixture.fragment,
            ),
        );
        assert_eq!(
            baseline.verify_at(&context(), true),
            Err(DesktopGpuError::DefinitionChanged)
        );

        write_private_file(
            &fixture.manager_state,
            &desktop_fixture_properties(
                9001,
                "cccccccccccccccccccccccccccccccc",
                &fixture.cgroup,
                &fixture.fragment,
            ),
        );
        write_process(
            &fixture.proc_root,
            &fixture.cgroup,
            9002,
            999,
            &[DESKTOP_COMPOSITOR.to_owned()],
        );
        assert_eq!(
            baseline.verify_at(&context(), true),
            Err(DesktopGpuError::ProcessIdentityUnavailable)
        );
    }

    #[test]
    fn desktop_baseline_rejects_untrusted_exec_start_or_missing_compositor() {
        let fixture = DesktopFixture::new();
        fs::write(
            &fixture.fragment,
            format!("[Service]\nExecStart={DESKTOP_LAUNCHER}\nExecStartPost=/bin/false\n")
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(
            fixture.capture().err(),
            Some(DesktopGpuError::UnitNotSupported)
        );

        let fixture = DesktopFixture::new();
        let invalid_exec = fs::read_to_string(&fixture.manager_state)
            .unwrap()
            .replace(DESKTOP_LAUNCHER, "/usr/bin/kwin_launcher_unknown");
        write_private_file(&fixture.manager_state, invalid_exec.as_bytes());
        assert_eq!(
            fixture.capture().err(),
            Some(DesktopGpuError::UnitNotSupported)
        );

        let fixture = DesktopFixture::new();
        write_process(
            &fixture.proc_root,
            &fixture.cgroup,
            9002,
            102,
            &["/usr/bin/kwin_wayland_helper".to_owned()],
        );
        assert_eq!(
            fixture.capture().err(),
            Some(DesktopGpuError::UnitNotSupported)
        );
    }

    #[test]
    #[ignore = "requires a live supported KDE user session; read-only"]
    fn real_kde_desktop_baseline_capture() {
        let baseline = DesktopGpuBaseline::capture(&context()).unwrap();
        baseline.verify(&context()).unwrap();
        assert!(baseline.accepts_claimant(baseline.compositor_pid));
        assert!(!baseline.accepts_claimant(1));
    }

    #[test]
    #[ignore = "requires explicit GPU UUID and an idle supported KDE session; read-only"]
    fn real_kde_baseline_accepts_idle_desktop() {
        use super::super::gpu_probe::{GpuProbeConfig, NvidiaSmiGpuProbe};
        let uuid = std::env::var("L3MS_LIVE_GPU_UUID").expect("set the target physical GPU UUID");
        let baseline = DesktopGpuBaseline::capture(&context()).unwrap();
        let probe = NvidiaSmiGpuProbe::new(GpuProbeConfig::new(uuid, 0, 512).unwrap());
        let observed = probe.observe_idle_with_baseline(
            &context(),
            RecoveryPolicy::new(MANAGER_RECOVERY).unwrap(),
            |pid| baseline.accepts_claimant(pid),
        );
        baseline.verify(&context()).unwrap();
        assert!(observed.unwrap().memory_used_mib() <= 512);
    }

    #[test]
    #[ignore = "requires explicit GPU UUID, KDE and a foreign compute claimant; read-only"]
    fn real_kde_baseline_refuses_foreign_compute() {
        use super::super::gpu_probe::{GpuProbeConfig, GpuProbeFailureKind, NvidiaSmiGpuProbe};
        let uuid = std::env::var("L3MS_LIVE_GPU_UUID").expect("set the target physical GPU UUID");
        let baseline = DesktopGpuBaseline::capture(&context()).unwrap();
        let probe = NvidiaSmiGpuProbe::new(GpuProbeConfig::new(uuid, 0, 512).unwrap());
        let recovery = RecoveryPolicy::new(MANAGER_RECOVERY).unwrap();
        let observed = probe
            .observe_idle_with_baseline(&context(), recovery, |pid| baseline.accepts_claimant(pid));
        baseline.verify(&context()).unwrap();
        assert_eq!(
            observed.unwrap_err().kind(),
            GpuProbeFailureKind::ClaimantPresent
        );
    }
}
