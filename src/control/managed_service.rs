//! Conservative ownership of one coordinator-created Linux transient service.
//!
//! This is a concrete service owner, not a generic lifecycle adapter. Callers
//! persist the prepared intent before dispatch. A local
//! systemd-run timeout can happen after the manager accepted the start, so all
//! post-spawn failures retain that intent for explicit reconciliation. No
//! failure path stops a unit by name.
//!
//! The owner is a uniquely named transient user service started with
//! KillMode=control-group and Restart=no. A handle is issued only after
//! systemd reports the expected transient marker, active state, InvocationID,
//! and cgroup. This is not proof of GPU release, ingress closure, admission,
//! or runtime health.
//!
//! Callers serialize coordinator operations with the shared host lock. That
//! lock does not fence another same-user systemd client; no external actor may
//! create, restart, or replace these transient units concurrently.
//!
//! Stop is serialized with other coordinator operations by the host lock.
//! systemctl addresses a unit name, not an InvocationID, so the generation
//! check immediately before stop is not atomic against an external same-user
//! restart or replacement. This primitive assumes no such concurrent manager
//! operation. It does not independently prove that a cgroup is empty.

use std::error::Error;
use std::ffi::{CString, OsString};
use std::fmt;
use std::fs;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use crate::state_store::{
    ComfyManagerGenerationReference, ComfyServiceGenerationReference, ComfyServiceIntentReference,
    ComfySupervisorCoordinatorReference,
};

use super::catalog::{Lane, RuntimeIdentity};
use super::process::{self, ProcessLimits};
use super::runtime::{OperationContext, RecoveryPolicy};

const SYSTEMD_RUN: &str = "/usr/bin/systemd-run";
const SYSTEMCTL: &str = "/usr/bin/systemctl";
const MAX_COORDINATOR_ID_BYTES: usize = 32;
const MAX_RUNTIME_ID_BYTES: usize = 128;
const MAX_OWNED_JOB_REQUEST_ID_BYTES: usize = 128;
const MAX_OWNED_JOB_DIRECTORY_BYTES: usize = 4096;
const OWNED_JOB_DIRECTORY_MODE: u32 = 0o700;
const MAX_WORKING_DIRECTORY_BYTES: usize = 4096;
const MAX_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_ARGUMENTS: usize = 4096;
const MAX_AGGREGATE_ARGUMENT_BYTES: usize = 256 * 1024;
const MAX_MANAGER_OUTPUT_BYTES: usize = 32 * 1024;
const UNIT_RANDOM_BYTES: usize = 16;
const MAX_STOP_TIMEOUT: Duration = Duration::from_secs(300);
const POLL_QUANTUM: Duration = Duration::from_millis(25);

/// Fixed, redacted failure categories. Command output, argv, and OS errors are
/// never retained in this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedServiceErrorKind {
    InvalidCoordinatorId,
    InvalidRuntimeIdentity,
    InvalidRecipe,
    InvalidBackendPort,
    InvalidLaunchRole,
    InvalidOwnedJobBinding,
    CoordinatorIdentityUnavailable,
    InvalidStopTimeout,
    RandomnessUnavailable,
    ExecutableUnavailable,
    ManagerCommandFailed,
    UnitAlreadyExists,
    UnitStateUnverifiable,
    OwnershipMismatch,
    ServiceNotActive,
    StopUnconfirmed,
    DeadlineExceeded,
    Cancelled,
}

impl fmt::Display for ManagedServiceErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidCoordinatorId => "invalid coordinator identity",
            Self::InvalidRuntimeIdentity => "invalid runtime identity",
            Self::InvalidRecipe => "invalid service launch recipe",
            Self::InvalidBackendPort => "invalid isolated backend port",
            Self::InvalidLaunchRole => "invalid managed service launch role",
            Self::InvalidOwnedJobBinding => "invalid owned local-job binding",
            Self::CoordinatorIdentityUnavailable => "coordinator namespace identity is unavailable",
            Self::InvalidStopTimeout => "invalid service stop timeout",
            Self::RandomnessUnavailable => "unique service identity is unavailable",
            Self::ExecutableUnavailable => "required service executable is unavailable",
            Self::ManagerCommandFailed => "service manager operation failed",
            Self::UnitAlreadyExists => "generated service name is already in use",
            Self::UnitStateUnverifiable => "service ownership state could not be verified",
            Self::OwnershipMismatch => "service ownership proof does not match",
            Self::ServiceNotActive => "owned service is not confirmed active",
            Self::StopUnconfirmed => "service stop could not be confirmed",
            Self::DeadlineExceeded => "service operation deadline expired",
            Self::Cancelled => "service operation was cancelled",
        })
    }
}

impl Error for ManagedServiceErrorKind {}

/// Bounded immutable argv. The executable is resolved and checked when the
/// recipe is created; no shell command string or environment override exists.
pub struct LaunchRecipe {
    executable: PathBuf,
    working_directory: PathBuf,
    arguments: Vec<OsString>,
    verified_python: Option<VerifiedPythonInterpreter>,
}

const MAX_PYTHON_CONFIG_BYTES: u64 = 16 * 1024;
const MAX_PYTHON_IDENTITY_BYTES: usize = 1024;

#[derive(Clone, Copy, Eq, PartialEq)]
struct PythonFileIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    mode: u32,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl PythonFileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            uid: metadata.uid(),
            mode: metadata.mode() & 0o7777,
            size: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    fn append_wire_fields(self, fields: &mut Vec<String>) {
        fields.extend([
            self.device.to_string(),
            self.inode.to_string(),
            self.uid.to_string(),
            self.mode.to_string(),
            self.size.to_string(),
            self.modified_seconds.to_string(),
            self.modified_nanoseconds.to_string(),
            self.changed_seconds.to_string(),
            self.changed_nanoseconds.to_string(),
        ]);
    }

    fn parse_wire_fields<'a>(
        fields: &mut impl Iterator<Item = &'a str>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let device = parse_python_wire_u64(fields)?;
        let inode = parse_python_wire_u64(fields)?;
        let uid = parse_python_wire_u32(fields)?;
        let mode = parse_python_wire_u32(fields)?;
        let size = parse_python_wire_u64(fields)?;
        let modified_seconds = parse_python_wire_i64(fields)?;
        let modified_nanoseconds = parse_python_wire_i64(fields)?;
        let changed_seconds = parse_python_wire_i64(fields)?;
        let changed_nanoseconds = parse_python_wire_i64(fields)?;
        if modified_nanoseconds.unsigned_abs() >= 1_000_000_000
            || changed_nanoseconds.unsigned_abs() >= 1_000_000_000
            || mode & !0o7777 != 0
        {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        Ok(Self {
            device,
            inode,
            uid,
            mode,
            size,
            modified_seconds,
            modified_nanoseconds,
            changed_seconds,
            changed_nanoseconds,
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct PythonDirectoryIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    mode: u32,
}

impl PythonDirectoryIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            uid: metadata.uid(),
            mode: metadata.mode() & 0o7777,
        }
    }

    fn append_wire_fields(self, fields: &mut Vec<String>) {
        fields.extend([
            self.device.to_string(),
            self.inode.to_string(),
            self.uid.to_string(),
            self.mode.to_string(),
        ]);
    }

    fn parse_wire_fields<'a>(
        fields: &mut impl Iterator<Item = &'a str>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let device = parse_python_wire_u64(fields)?;
        let inode = parse_python_wire_u64(fields)?;
        let uid = parse_python_wire_u32(fields)?;
        let mode = parse_python_wire_u32(fields)?;
        if mode & !0o7777 != 0 {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        Ok(Self {
            device,
            inode,
            uid,
            mode,
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct VerifiedPythonIdentity {
    configured_entry_is_symlink: bool,
    configured_entry: PythonFileIdentity,
    target: PythonFileIdentity,
    venv_directory: PythonDirectoryIdentity,
    bin_directory: PythonDirectoryIdentity,
    pyvenv_config: PythonFileIdentity,
}

impl VerifiedPythonIdentity {
    fn to_wire(self) -> String {
        let mut fields = Vec::with_capacity(36);
        fields.push(if self.configured_entry_is_symlink {
            "1".to_owned()
        } else {
            "0".to_owned()
        });
        self.configured_entry.append_wire_fields(&mut fields);
        self.target.append_wire_fields(&mut fields);
        self.venv_directory.append_wire_fields(&mut fields);
        self.bin_directory.append_wire_fields(&mut fields);
        self.pyvenv_config.append_wire_fields(&mut fields);
        fields.join(":")
    }

    fn from_wire(value: &str) -> Result<Self, ManagedServiceErrorKind> {
        if value.len() > MAX_PYTHON_IDENTITY_BYTES {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        let mut fields = value.split(':');
        let configured_entry_is_symlink = match fields.next() {
            Some("0") => false,
            Some("1") => true,
            _ => return Err(ManagedServiceErrorKind::InvalidRecipe),
        };
        let identity = Self {
            configured_entry_is_symlink,
            configured_entry: PythonFileIdentity::parse_wire_fields(&mut fields)?,
            target: PythonFileIdentity::parse_wire_fields(&mut fields)?,
            venv_directory: PythonDirectoryIdentity::parse_wire_fields(&mut fields)?,
            bin_directory: PythonDirectoryIdentity::parse_wire_fields(&mut fields)?,
            pyvenv_config: PythonFileIdentity::parse_wire_fields(&mut fields)?,
        };
        if fields.next().is_some() || identity.to_wire() != value {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        Ok(identity)
    }
}

/// A Comfy-only Python invocation that keeps the declared venv path while
/// binding its symlink, executable target, venv directories, and pyvenv.cfg to
/// fresh filesystem identities. Generic launch constructors do not accept or
/// infer this marker.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct VerifiedPythonInterpreter {
    configured_path: PathBuf,
    target_path: PathBuf,
    venv_directory: PathBuf,
    identity: VerifiedPythonIdentity,
}

impl VerifiedPythonInterpreter {
    /// Pin the exact `<venv>/.venv-comfy/bin/python` invocation path and its
    /// existing executable target. Call `verify` again before launch.
    pub(crate) fn capture(
        configured_path: impl AsRef<Path>,
        expected_venv_directory: impl AsRef<Path>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        Self::inspect(configured_path.as_ref(), expected_venv_directory.as_ref())
    }

    pub(crate) fn verify(&self) -> Result<(), ManagedServiceErrorKind> {
        let current = Self::inspect(&self.configured_path, &self.venv_directory)?;
        if current == *self {
            Ok(())
        } else {
            Err(ManagedServiceErrorKind::InvalidRecipe)
        }
    }

    pub(crate) fn configured_path(&self) -> &Path {
        &self.configured_path
    }

    pub(crate) fn target_path(&self) -> &Path {
        &self.target_path
    }

    pub(crate) fn venv_directory(&self) -> &Path {
        &self.venv_directory
    }

    pub(crate) fn identity_wire(&self) -> String {
        self.identity.to_wire()
    }

    pub(crate) fn from_helper_parts(
        configured_path: PathBuf,
        target_path: PathBuf,
        venv_directory: PathBuf,
        identity_wire: &str,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let expected_identity = VerifiedPythonIdentity::from_wire(identity_wire)?;
        let current = Self::inspect(&configured_path, &venv_directory)?;
        if current.target_path != target_path || current.identity != expected_identity {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        Ok(current)
    }

    fn inspect(
        configured_path: &Path,
        expected_venv_directory: &Path,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let invalid = || ManagedServiceErrorKind::InvalidRecipe;
        if !configured_path.is_absolute()
            || !expected_venv_directory.is_absolute()
            || configured_path.as_os_str().as_bytes().len() > MAX_WORKING_DIRECTORY_BYTES
            || expected_venv_directory.as_os_str().as_bytes().len() > MAX_WORKING_DIRECTORY_BYTES
            || configured_path.as_os_str().as_bytes().contains(&0)
            || expected_venv_directory.as_os_str().as_bytes().contains(&0)
            || configured_path
                .components()
                .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
            || expected_venv_directory
                .components()
                .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
            || expected_venv_directory.file_name() != Some(std::ffi::OsStr::new(".venv-comfy"))
            || configured_path != expected_venv_directory.join("bin/python")
            || expected_venv_directory.canonicalize().ok()
                != Some(expected_venv_directory.to_path_buf())
        {
            return Err(invalid());
        }

        let venv_metadata = fs::symlink_metadata(expected_venv_directory).map_err(|_| invalid())?;
        if !venv_metadata.is_dir() || venv_metadata.file_type().is_symlink() {
            return Err(invalid());
        }
        let bin_directory = expected_venv_directory.join("bin");
        let bin_metadata = fs::symlink_metadata(&bin_directory).map_err(|_| invalid())?;
        if !bin_metadata.is_dir()
            || bin_metadata.file_type().is_symlink()
            || bin_directory.canonicalize().ok() != Some(bin_directory.clone())
        {
            return Err(invalid());
        }

        let configured_metadata = fs::symlink_metadata(configured_path).map_err(|_| invalid())?;
        let configured_entry_is_symlink = configured_metadata.file_type().is_symlink();
        if !configured_entry_is_symlink && !configured_metadata.is_file() {
            return Err(invalid());
        }
        let target_path = configured_path.canonicalize().map_err(|_| invalid())?;
        if !target_path.is_absolute()
            || target_path.as_os_str().as_bytes().len() > MAX_WORKING_DIRECTORY_BYTES
        {
            return Err(invalid());
        }
        let target_metadata = fs::metadata(&target_path).map_err(|_| invalid())?;
        if !target_metadata.is_file() || target_metadata.permissions().mode() & 0o111 == 0 {
            return Err(invalid());
        }

        let config_path = expected_venv_directory.join("pyvenv.cfg");
        let config_link_metadata = fs::symlink_metadata(&config_path).map_err(|_| invalid())?;
        if config_link_metadata.file_type().is_symlink()
            || !config_link_metadata.is_file()
            || config_link_metadata.len() == 0
            || config_link_metadata.len() > MAX_PYTHON_CONFIG_BYTES
        {
            return Err(invalid());
        }

        Ok(Self {
            configured_path: configured_path.to_path_buf(),
            target_path,
            venv_directory: expected_venv_directory.to_path_buf(),
            identity: VerifiedPythonIdentity {
                configured_entry_is_symlink,
                configured_entry: PythonFileIdentity::from_metadata(&configured_metadata),
                target: PythonFileIdentity::from_metadata(&target_metadata),
                venv_directory: PythonDirectoryIdentity::from_metadata(&venv_metadata),
                bin_directory: PythonDirectoryIdentity::from_metadata(&bin_metadata),
                pyvenv_config: PythonFileIdentity::from_metadata(&config_link_metadata),
            },
        })
    }
}

impl fmt::Debug for VerifiedPythonInterpreter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VerifiedPythonInterpreter([redacted])")
    }
}

fn parse_python_wire_u64<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
) -> Result<u64, ManagedServiceErrorKind> {
    fields
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or(ManagedServiceErrorKind::InvalidRecipe)
}

fn parse_python_wire_u32<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
) -> Result<u32, ManagedServiceErrorKind> {
    fields
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or(ManagedServiceErrorKind::InvalidRecipe)
}

fn parse_python_wire_i64<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
) -> Result<i64, ManagedServiceErrorKind> {
    fields
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or(ManagedServiceErrorKind::InvalidRecipe)
}

/// Validated, payload-free binding between one local job and its owned
/// service. The directory identity is captured from an existing private
/// directory and rechecked immediately before manager dispatch. Persisting
/// this value never authorizes a restart or payload execution.
#[derive(Clone, Eq, PartialEq)]
pub struct OwnedLocalJobBinding {
    runtime_identity: RuntimeIdentity,
    job_id: u64,
    request_id: String,
    job_directory: PathBuf,
    directory_device: u64,
    directory_inode: u64,
    coordinator: super::engine_transport::CoordinatorIdentityDescriptor,
}

impl OwnedLocalJobBinding {
    /// Capture one existing canonical current-UID mode-0700 job directory.
    /// Every path component is opened without following symlinks. This stores
    /// only safe identifiers, directory identity, runtime, and coordinator
    /// identity; it never stores request content or command arguments.
    pub fn capture(
        runtime_identity: RuntimeIdentity,
        job_id: u64,
        request_id: impl Into<String>,
        job_directory: impl AsRef<Path>,
        coordinator: super::engine_transport::CoordinatorIdentityDescriptor,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let request_id = request_id.into();
        validate_owned_job_fields(&runtime_identity, job_id, &request_id, coordinator)?;
        let supplied_path = job_directory.as_ref();
        if !supplied_job_directory_path_is_acceptable(supplied_path) {
            return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
        }
        let current_coordinator =
            super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                .map_err(|_| ManagedServiceErrorKind::InvalidOwnedJobBinding)?;
        if coordinator != current_coordinator {
            return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
        }

        let supplied_directory = open_owned_job_directory(supplied_path)?;
        let (directory_device, directory_inode) =
            owned_job_directory_identity(&supplied_directory)?;
        let canonical_directory = fs::canonicalize(supplied_path)
            .map_err(|_| ManagedServiceErrorKind::InvalidOwnedJobBinding)?;
        if !canonical_owned_job_directory(&canonical_directory) {
            return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
        }
        let canonical_directory_handle = open_owned_job_directory(&canonical_directory)?;
        if owned_job_directory_identity(&canonical_directory_handle)?
            != (directory_device, directory_inode)
        {
            return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
        }
        validate_owned_job_binding_parts(
            &runtime_identity,
            job_id,
            &request_id,
            &canonical_directory,
            directory_device,
            directory_inode,
            coordinator,
        )?;

        Ok(Self {
            runtime_identity,
            job_id,
            request_id,
            job_directory: canonical_directory,
            directory_device,
            directory_inode,
            coordinator,
        })
    }

    pub(super) fn from_persisted_parts(
        runtime_identity: RuntimeIdentity,
        job_id: u64,
        request_id: String,
        job_directory: PathBuf,
        directory_device: u64,
        directory_inode: u64,
        coordinator: super::engine_transport::CoordinatorIdentityDescriptor,
    ) -> Result<Self, ManagedServiceErrorKind> {
        validate_owned_job_binding_parts(
            &runtime_identity,
            job_id,
            &request_id,
            &job_directory,
            directory_device,
            directory_inode,
            coordinator,
        )?;
        Ok(Self {
            runtime_identity,
            job_id,
            request_id,
            job_directory,
            directory_device,
            directory_inode,
            coordinator,
        })
    }

    pub fn runtime_identity(&self) -> &RuntimeIdentity {
        &self.runtime_identity
    }

    pub fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn job_directory(&self) -> &Path {
        &self.job_directory
    }

    pub fn directory_device(&self) -> u64 {
        self.directory_device
    }

    pub fn directory_inode(&self) -> u64 {
        self.directory_inode
    }

    pub fn coordinator_identity(&self) -> super::engine_transport::CoordinatorIdentityDescriptor {
        self.coordinator
    }

    fn validate_directory_identity(&self) -> Result<(), ManagedServiceErrorKind> {
        let directory = open_owned_job_directory(&self.job_directory)?;
        if owned_job_directory_identity(&directory)?
            == (self.directory_device, self.directory_inode)
        {
            Ok(())
        } else {
            Err(ManagedServiceErrorKind::InvalidOwnedJobBinding)
        }
    }
}

impl fmt::Debug for OwnedLocalJobBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OwnedLocalJobBinding([redacted])")
    }
}

/// Fixed manager topology generated by this controller. Ordinary and owned
/// local-job callers keep `Type=exec`; only the typed isolated-supervisor
/// role can request namespace properties.
#[derive(Clone, Eq, PartialEq)]
pub(super) enum ServiceLaunchRole {
    Ordinary,
    IsolatedEngineSupervisor(super::engine_transport::CoordinatorIdentityDescriptor),
    OwnedLocalJob(OwnedLocalJobBinding),
}

impl fmt::Debug for ServiceLaunchRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Ordinary => "Ordinary",
            Self::IsolatedEngineSupervisor(_) => "IsolatedEngineSupervisor([redacted])",
            Self::OwnedLocalJob(_) => "OwnedLocalJob([redacted])",
        })
    }
}

impl LaunchRecipe {
    pub fn new(
        executable: impl AsRef<Path>,
        working_directory: impl AsRef<Path>,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let path = executable.as_ref();
        if !path.is_absolute() {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        let executable = path
            .canonicalize()
            .map_err(|_| ManagedServiceErrorKind::InvalidRecipe)?;

        Self::new_with_executable(executable, working_directory, arguments, None)
    }

    /// Create a recipe for the one explicitly verified Comfy venv interpreter.
    /// Unlike `new`, this preserves the configured symlink path so Python can
    /// discover `pyvenv.cfg`; the marker is checked again by the helper before
    /// it spawns the backend. Ordinary launch recipes remain canonicalized.
    pub(crate) fn new_with_verified_python(
        interpreter: &VerifiedPythonInterpreter,
        working_directory: impl AsRef<Path>,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        interpreter.verify()?;
        Self::new_with_executable(
            interpreter.configured_path().to_path_buf(),
            working_directory,
            arguments,
            Some(interpreter.clone()),
        )
    }

    fn new_with_executable(
        executable: PathBuf,
        working_directory: impl AsRef<Path>,
        arguments: impl IntoIterator<Item = OsString>,
        verified_python: Option<VerifiedPythonInterpreter>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        if !executable.is_absolute()
            || executable.as_os_str().as_bytes().contains(&0)
            || executable.as_os_str().as_bytes().len() > MAX_WORKING_DIRECTORY_BYTES
        {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        if let Some(interpreter) = &verified_python {
            interpreter.verify()?;
            if executable != interpreter.configured_path() {
                return Err(ManagedServiceErrorKind::InvalidRecipe);
            }
        } else if executable.canonicalize().ok().as_ref() != Some(&executable) {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        let metadata =
            fs::metadata(&executable).map_err(|_| ManagedServiceErrorKind::InvalidRecipe)?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }

        let requested_working_directory = working_directory.as_ref();
        if !requested_working_directory.is_absolute()
            || requested_working_directory
                .as_os_str()
                .as_bytes()
                .contains(&0)
            || requested_working_directory.as_os_str().as_bytes().len()
                > MAX_WORKING_DIRECTORY_BYTES
        {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        let working_directory = requested_working_directory
            .canonicalize()
            .map_err(|_| ManagedServiceErrorKind::InvalidRecipe)?;
        let working_directory_metadata = working_directory
            .metadata()
            .map_err(|_| ManagedServiceErrorKind::InvalidRecipe)?;
        if !working_directory_metadata.is_dir()
            || working_directory.as_os_str().as_bytes().contains(&0)
            || working_directory.as_os_str().as_bytes().len() > MAX_WORKING_DIRECTORY_BYTES
            || working_directory.to_str().is_none()
        {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }

        let mut collected = Vec::new();
        let mut total = executable
            .as_os_str()
            .as_bytes()
            .len()
            .saturating_add(working_directory.as_os_str().as_bytes().len());
        if total > MAX_AGGREGATE_ARGUMENT_BYTES {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        for argument in arguments {
            let bytes = argument.as_os_str().as_bytes();
            if bytes.len() > MAX_ARGUMENT_BYTES
                || bytes.contains(&0)
                || collected.len() >= MAX_ARGUMENTS
            {
                return Err(ManagedServiceErrorKind::InvalidRecipe);
            }
            total = total.saturating_add(bytes.len());
            if total > MAX_AGGREGATE_ARGUMENT_BYTES {
                return Err(ManagedServiceErrorKind::InvalidRecipe);
            }
            collected.push(argument);
        }
        Ok(Self {
            executable,
            working_directory,
            arguments: collected,
            verified_python,
        })
    }

    /// Stable opaque identity for the immutable executable and argv, suitable
    /// for comparing a persisted intent with a refreshed recipe. It does not
    /// expose the command or its arguments.
    pub fn launch_identity(&self) -> String {
        launch_fingerprint(self)
    }

    pub(super) fn executable(&self) -> &Path {
        &self.executable
    }

    pub(super) fn working_directory(&self) -> &Path {
        &self.working_directory
    }

    pub(super) fn arguments(&self) -> &[OsString] {
        &self.arguments
    }

    pub(super) fn verified_python(&self) -> Option<&VerifiedPythonInterpreter> {
        self.verified_python.as_ref()
    }
}

impl fmt::Debug for LaunchRecipe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LaunchRecipe(<redacted>)")
    }
}

/// Safe durable description of a start intent. This record is not authority
/// to stop anything; reconciliation still requires a live generation and
/// cgroup match.
#[derive(Clone, Eq, PartialEq)]
pub struct ServiceIntentRecord {
    coordinator_id: String,
    unit_name: String,
    launch_identity: String,
    stop_timeout_millis: u64,
    runtime_identity: Option<RuntimeIdentity>,
    launch_role: ServiceLaunchRole,
}

impl ServiceIntentRecord {
    /// Rebuild a durable pending-intent record after restart. This restores
    /// data only; it does not create an owned handle or authorize any action.
    pub fn from_persisted_parts(
        coordinator_id: String,
        unit_name: String,
        launch_identity: String,
        stop_timeout_millis: u64,
    ) -> Result<Self, ManagedServiceErrorKind> {
        Self::from_persisted_parts_with_runtime(
            coordinator_id,
            unit_name,
            launch_identity,
            stop_timeout_millis,
            None,
        )
    }

    pub(super) fn from_persisted_parts_with_runtime(
        coordinator_id: String,
        unit_name: String,
        launch_identity: String,
        stop_timeout_millis: u64,
        runtime_identity: Option<RuntimeIdentity>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        Self::from_persisted_parts_with_role(
            coordinator_id,
            unit_name,
            launch_identity,
            stop_timeout_millis,
            runtime_identity,
            ServiceLaunchRole::Ordinary,
        )
    }

    pub(super) fn from_persisted_parts_with_role(
        coordinator_id: String,
        unit_name: String,
        launch_identity: String,
        stop_timeout_millis: u64,
        runtime_identity: Option<RuntimeIdentity>,
        launch_role: ServiceLaunchRole,
    ) -> Result<Self, ManagedServiceErrorKind> {
        if coordinator_id.len() > MAX_COORDINATOR_ID_BYTES
            || unit_name.len() > 96
            || launch_identity.len() > 64
            || stop_timeout_millis == 0
            || stop_timeout_millis > MAX_STOP_TIMEOUT.as_millis() as u64
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        let record = Self {
            coordinator_id,
            unit_name,
            launch_identity,
            stop_timeout_millis,
            runtime_identity,
            launch_role,
        };
        if !valid_intent(&record) {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        Ok(record)
    }

    pub fn coordinator_id(&self) -> &str {
        &self.coordinator_id
    }

    pub fn unit_name(&self) -> &str {
        &self.unit_name
    }

    /// Stable opaque fingerprint for distinguishing immutable recipes in
    /// persisted intent records. This is metadata, not a secret or ownership
    /// proof.
    pub fn launch_identity(&self) -> &str {
        &self.launch_identity
    }

    pub fn stop_timeout_millis(&self) -> u64 {
        self.stop_timeout_millis
    }

    /// Optional catalog identity this intent was prepared to own. An absent
    /// binding denotes a legacy/generic intent and must not be inferred from
    /// its executable, unit name, or runtime observations.
    pub fn runtime_identity(&self) -> Option<&RuntimeIdentity> {
        self.runtime_identity.as_ref()
    }

    pub(super) fn launch_role(&self) -> &ServiceLaunchRole {
        &self.launch_role
    }

    /// Safe identifiers and private directory identity for an owned local
    /// job role. Presence is metadata only and does not authorize replay.
    pub fn owned_local_job_binding(&self) -> Option<&OwnedLocalJobBinding> {
        match &self.launch_role {
            ServiceLaunchRole::OwnedLocalJob(binding) => Some(binding),
            _ => None,
        }
    }
}

impl fmt::Debug for ServiceIntentRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServiceIntentRecord")
            .field("coordinator_id", &self.coordinator_id)
            .field("unit_name", &self.unit_name)
            .field("runtime_identity", &self.runtime_identity)
            .field("launch_role", &self.launch_role)
            .field("launch_identity", &"<redacted>")
            .field("stop_timeout_millis", &self.stop_timeout_millis)
            .finish()
    }
}

/// A single-use prepared start. Persist intent_record before moving this token
/// into dispatch. The token is intentionally not Clone.
pub struct PreparedServiceStart {
    intent: ServiceIntentRecord,
    recipe: LaunchRecipe,
}

impl PreparedServiceStart {
    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }
}

impl fmt::Debug for PreparedServiceStart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedServiceStart")
            .field("intent", &self.intent)
            .field("recipe", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StartDisposition {
    /// The manager command was not attempted, or preflight rejected the name.
    NotDispatched,
    /// systemd-run may have submitted a start, so the intent must be retained.
    OwnershipUncertain,
}

/// Start failures preserve their generated intent, including timeout and
/// partial-start uncertainty.
pub struct StartFailure {
    intent: Box<ServiceIntentRecord>,
    disposition: StartDisposition,
    kind: ManagedServiceErrorKind,
}

impl StartFailure {
    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }

    pub fn disposition(&self) -> StartDisposition {
        self.disposition
    }

    pub fn kind(&self) -> ManagedServiceErrorKind {
        self.kind
    }
}

impl fmt::Debug for StartFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StartFailure")
            .field("intent", &self.intent)
            .field("disposition", &self.disposition)
            .field("kind", &self.kind)
            .finish()
    }
}

impl fmt::Display for StartFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({:?})", self.kind, self.disposition)
    }
}

impl Error for StartFailure {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedServiceState {
    Running,
    Stopped,
    Unknown,
}

/// Reconciliation describes the manager query at this instant. AbsentNow
/// does not erase the durable intent or establish resource release.
pub enum ReconcileResult {
    Owned(Box<OwnedService>),
    StopOnly(Box<StopOnlyService>),
    AbsentNow,
    Unknown,
}

impl fmt::Debug for ReconcileResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned(handle) => formatter.debug_tuple("Owned").field(handle).finish(),
            Self::StopOnly(handle) => formatter.debug_tuple("StopOnly").field(handle).finish(),
            Self::AbsentNow => formatter.write_str("AbsentNow"),
            Self::Unknown => formatter.write_str("Unknown"),
        }
    }
}

/// Private, non-Clone ownership capability. The unit name alone is never
/// accepted by state or stop operations.
pub struct OwnedService {
    intent: ServiceIntentRecord,
    invocation_id: String,
    cgroup: String,
    main_pid: u32,
}

/// Cloneable comparison evidence for one exact owner-created manager
/// generation. This is not an operation capability: it cannot start, inspect,
/// or stop a service. Its private intent association prevents matching an
/// otherwise-identical manager snapshot from another launch/runtime binding.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct OwnedServiceGenerationIdentity {
    intent: ServiceIntentRecord,
    invocation_id: String,
    cgroup: String,
    main_pid: u32,
}

impl fmt::Debug for OwnedServiceGenerationIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OwnedServiceGenerationIdentity([redacted])")
    }
}

impl OwnedServiceGenerationIdentity {
    /// Project this exact owner-issued Comfy supervisor identity into bounded,
    /// payload-free history evidence. Other runtime identities and launch
    /// roles cannot be projected into the Comfy dispatch path.
    pub(crate) fn comfy_service_generation_reference(
        &self,
    ) -> Result<ComfyServiceGenerationReference, ManagedServiceErrorKind> {
        if !self
            .intent
            .runtime_identity()
            .is_some_and(|runtime| runtime.lane == Lane::Media && runtime.as_str() == "ComfyUI")
        {
            return Err(ManagedServiceErrorKind::InvalidRuntimeIdentity);
        }
        let ServiceLaunchRole::IsolatedEngineSupervisor(coordinator) = self.intent.launch_role()
        else {
            return Err(ManagedServiceErrorKind::InvalidLaunchRole);
        };

        let reference = ComfyServiceGenerationReference {
            schema_version: 1,
            intent: ComfyServiceIntentReference {
                coordinator_id: self.intent.coordinator_id.clone(),
                unit_name: self.intent.unit_name.clone(),
                launch_identity: self.intent.launch_identity.clone(),
                stop_timeout_millis: self.intent.stop_timeout_millis,
                runtime_lane: "media".to_owned(),
                runtime_name: "ComfyUI".to_owned(),
                launch_role: "isolated_engine_supervisor".to_owned(),
                supervisor_coordinator: ComfySupervisorCoordinatorReference {
                    pid: coordinator.pid(),
                    uid: coordinator.uid(),
                    start_time_ticks: coordinator.start_time_ticks(),
                    network_namespace_device: coordinator.network_namespace_device(),
                    network_namespace_inode: coordinator.network_namespace_inode(),
                },
            },
            generation: ComfyManagerGenerationReference {
                invocation_id: self.invocation_id.clone(),
                cgroup: self.cgroup.clone(),
                main_pid: self.main_pid,
            },
        };
        if !reference.is_valid() {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        Ok(reference)
    }

    #[cfg(test)]
    pub(crate) fn for_test_comfy_supervisor() -> Result<Self, ManagedServiceErrorKind> {
        Self::for_test_comfy_supervisor_with_main_pid(2345)
    }

    #[cfg(test)]
    pub(crate) fn for_test_comfy_supervisor_with_main_pid(
        main_pid: u32,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let coordinator = super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
            .map_err(|_| ManagedServiceErrorKind::CoordinatorIdentityUnavailable)?;
        let runtime = RuntimeIdentity::new(Lane::Media, "ComfyUI")
            .map_err(|_| ManagedServiceErrorKind::InvalidRuntimeIdentity)?;
        let unit_name = "l3ms-test-owner-0123456789abcdef0123456789abcdef.service".to_owned();
        let intent = ServiceIntentRecord::from_persisted_parts_with_role(
            "test-owner".to_owned(),
            unit_name.clone(),
            "b".repeat(64),
            30_000,
            Some(runtime),
            ServiceLaunchRole::IsolatedEngineSupervisor(coordinator),
        )?;
        Ok(Self {
            intent,
            invocation_id: "c".repeat(32),
            cgroup: format!("/user.slice/{unit_name}"),
            main_pid,
        })
    }
}

impl OwnedService {
    pub fn unit_name(&self) -> &str {
        &self.intent.unit_name
    }

    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }

    /// Peer identity of the owned isolated-engine supervisor on `engine.sock`.
    /// The helper is launched by the coordinator's user manager, so it shares
    /// the captured coordinator UID. Ordinary services deliberately expose no
    /// peer identity to native-engine health probes.
    pub(crate) fn supervisor_peer_identity(&self) -> Option<(u32, u32)> {
        let ServiceLaunchRole::IsolatedEngineSupervisor(coordinator) = &self.intent.launch_role
        else {
            return None;
        };
        Some((self.main_pid, coordinator.uid()))
    }

    /// Capture opaque identity from this owner-created handle only. The
    /// enclosing `ManagedRuntimeOwner` validates current WAL/state before
    /// publishing this comparison evidence to another module.
    pub(super) fn generation_identity(&self) -> OwnedServiceGenerationIdentity {
        OwnedServiceGenerationIdentity {
            intent: self.intent.clone(),
            invocation_id: self.invocation_id.clone(),
            cgroup: self.cgroup.clone(),
            main_pid: self.main_pid,
        }
    }
}

impl fmt::Debug for OwnedService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedService")
            .field("unit_name", &self.intent.unit_name)
            .field("invocation_id", &"<redacted>")
            .field("cgroup", &"<redacted>")
            .field("main_pid", &"<redacted>")
            .finish()
    }
}

/// Restricted cleanup capability recovered from a secure WAL record when the
/// original isolated-supervisor coordinator identity is no longer available.
/// It is intentionally non-Clone and carries only one observed manager
/// generation. It cannot be used for state, start, replay, or admission.
pub struct StopOnlyService {
    intent: ServiceIntentRecord,
    invocation_id: String,
    cgroup: String,
    main_pid: u32,
}

impl StopOnlyService {
    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }
}

impl fmt::Debug for StopOnlyService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StopOnlyService")
            .field("unit_name", &self.intent.unit_name)
            .field("invocation_id", &"<redacted>")
            .field("cgroup", &"<redacted>")
            .field("main_pid", &"<redacted>")
            .finish()
    }
}

/// Evidence that the manager acknowledged stopping this exact owned
/// generation. This is not evidence that its cgroup is empty or a resource
/// has been released. The private fields prevent callers from manufacturing a
/// receipt from a unit name or an arbitrary cgroup path. The original intent
/// is retained privately so later release checks use the exact runtime and
/// launch identity that were durably prepared.
#[derive(Eq, PartialEq)]
pub struct StoppedService {
    intent: ServiceIntentRecord,
    invocation_id: String,
    cgroup: String,
}

impl StoppedService {
    /// The generation-bound path is consumed only by the cgroup observation
    /// module. It is deliberately not part of this receipt's Debug output.
    pub(crate) fn cgroup_path(&self) -> &str {
        &self.cgroup
    }

    pub(crate) fn unit_name(&self) -> &str {
        self.intent.unit_name()
    }

    pub(crate) fn invocation_id(&self) -> &str {
        &self.invocation_id
    }

    /// Original validated intent metadata. This does not authorize any manager
    /// operation and is retained so callers can reconcile after a failed
    /// ledger retirement.
    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        unit_name: String,
        invocation_id: String,
        cgroup: String,
    ) -> Option<Self> {
        let stem = unit_name.strip_prefix("l3ms-")?.strip_suffix(".service")?;
        let (coordinator_id, random) = stem.rsplit_once('-')?;
        if unit_name.len() <= 96
            && coordinator_id.len() <= MAX_COORDINATOR_ID_BYTES
            && random.len() == UNIT_RANDOM_BYTES * 2
            && valid_invocation_id(&invocation_id)
            && valid_cgroup(&cgroup, &unit_name)
        {
            let intent = ServiceIntentRecord::from_persisted_parts(
                coordinator_id.to_owned(),
                unit_name,
                "b".repeat(64),
                30_000,
            )
            .ok()?;
            Some(Self {
                intent,
                invocation_id,
                cgroup,
            })
        } else {
            None
        }
    }
}

impl fmt::Debug for StoppedService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StoppedService(<redacted>)")
    }
}

/// Coordinates only transient user services created from validated recipes.
pub struct ManagedServiceController {
    coordinator_id: String,
    systemd_run: PathBuf,
    systemctl: PathBuf,
}

impl ManagedServiceController {
    /// Use fixed, root-owned systemd client binaries. No PATH lookup is used.
    pub fn production(coordinator_id: impl Into<String>) -> Result<Self, ManagedServiceErrorKind> {
        let coordinator_id = coordinator_id.into();
        validate_coordinator_id(&coordinator_id)?;
        let systemd_run = PathBuf::from(SYSTEMD_RUN);
        let systemctl = PathBuf::from(SYSTEMCTL);
        verify_production_binary(&systemd_run)?;
        verify_production_binary(&systemctl)?;
        Ok(Self {
            coordinator_id,
            systemd_run,
            systemctl,
        })
    }

    /// Generate and expose the durable identifier before any manager command.
    pub fn prepare_start(
        &self,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
    ) -> Result<PreparedServiceStart, ManagedServiceErrorKind> {
        self.prepare_start_inner(recipe, stop_timeout, None, ServiceLaunchRole::Ordinary)
    }

    /// Prepare an intent bound to a catalog runtime identity. The binding is
    /// persisted with the start intent and included in manager reconciliation;
    /// it is metadata, not standalone admission or resource ownership proof.
    pub fn prepare_start_for_runtime(
        &self,
        runtime_identity: RuntimeIdentity,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
    ) -> Result<PreparedServiceStart, ManagedServiceErrorKind> {
        self.prepare_start_inner(
            recipe,
            stop_timeout,
            Some(runtime_identity),
            ServiceLaunchRole::Ordinary,
        )
    }

    /// Prepare the one supported isolated-engine supervisor role. Its single
    /// transient unit owns the helper and backend child in one private network
    /// namespace and one systemd cgroup.
    pub(super) fn prepare_isolated_engine_supervisor(
        &self,
        runtime_identity: RuntimeIdentity,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        coordinator: super::engine_transport::CoordinatorIdentityDescriptor,
    ) -> Result<PreparedServiceStart, ManagedServiceErrorKind> {
        self.prepare_start_inner(
            recipe,
            stop_timeout,
            Some(runtime_identity),
            ServiceLaunchRole::IsolatedEngineSupervisor(coordinator),
        )
    }

    /// Prepare an ordinary `Type=exec` service bound to exactly one local job.
    /// The durable binding is included in the launch fingerprint. It is a
    /// no-replay ownership record, not permission to execute payload from the
    /// retained store.
    pub(super) fn prepare_owned_local_job(
        &self,
        binding: OwnedLocalJobBinding,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
    ) -> Result<PreparedServiceStart, ManagedServiceErrorKind> {
        self.prepare_start_inner(
            recipe,
            stop_timeout,
            Some(binding.runtime_identity().clone()),
            ServiceLaunchRole::OwnedLocalJob(binding),
        )
    }

    fn prepare_start_inner(
        &self,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        runtime_identity: Option<RuntimeIdentity>,
        launch_role: ServiceLaunchRole,
    ) -> Result<PreparedServiceStart, ManagedServiceErrorKind> {
        if stop_timeout.is_zero() || stop_timeout > MAX_STOP_TIMEOUT {
            return Err(ManagedServiceErrorKind::InvalidStopTimeout);
        }
        if runtime_identity
            .as_ref()
            .is_some_and(|identity| !valid_runtime_identity(identity))
        {
            return Err(ManagedServiceErrorKind::InvalidRuntimeIdentity);
        }
        validate_role_runtime_binding(&launch_role, runtime_identity.as_ref())?;
        let suffix = random_unit_suffix()?;
        let unit_name = format!("l3ms-{}-{suffix}.service", self.coordinator_id);
        validate_launch_role(&launch_role)?;
        let launch_identity =
            launch_fingerprint_for_role(&recipe, &launch_role, runtime_identity.as_ref());
        let stop_timeout_millis = duration_millis_ceil(stop_timeout) as u64;
        Ok(PreparedServiceStart {
            intent: ServiceIntentRecord {
                coordinator_id: self.coordinator_id.clone(),
                unit_name,
                launch_identity,
                stop_timeout_millis,
                runtime_identity,
                launch_role,
            },
            recipe,
        })
    }

    /// Consume a prepared start exactly once. The caller persists the intent
    /// record first. Post-spawn failures never trigger name-only cleanup.
    pub fn dispatch(
        &self,
        prepared: PreparedServiceStart,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, StartFailure> {
        let intent = prepared.intent;
        let fail = |disposition, kind| StartFailure {
            intent: Box::new(intent.clone()),
            disposition,
            kind,
        };
        if intent.coordinator_id != self.coordinator_id
            || !valid_intent(&intent)
            || intent.launch_identity
                != launch_fingerprint_for_role(
                    &prepared.recipe,
                    &intent.launch_role,
                    intent.runtime_identity.as_ref(),
                )
        {
            return Err(fail(
                StartDisposition::NotDispatched,
                ManagedServiceErrorKind::OwnershipMismatch,
            ));
        }
        if let Err(kind) = context_error(context) {
            return Err(fail(StartDisposition::NotDispatched, kind));
        }
        if let ServiceLaunchRole::OwnedLocalJob(binding) = &intent.launch_role {
            if binding.validate_directory_identity().is_err()
                || !coordinator_identity_is_current(binding.coordinator_identity())
            {
                return Err(fail(
                    StartDisposition::NotDispatched,
                    ManagedServiceErrorKind::OwnershipMismatch,
                ));
            }
        }
        match self.preflight_unit(&intent, context) {
            Ok(true) => {}
            Ok(false) => {
                return Err(fail(
                    StartDisposition::NotDispatched,
                    ManagedServiceErrorKind::UnitAlreadyExists,
                ));
            }
            Err(kind) => return Err(fail(StartDisposition::NotDispatched, kind)),
        }

        let description = description_for(&intent);
        let timeout = format!("{}ms", intent.stop_timeout_millis);
        let mut command = Command::new(&self.systemd_run);
        command
            .arg("--user")
            .arg("--no-ask-password")
            .arg("--quiet")
            .arg("--no-block")
            .arg("--expand-environment=no")
            .arg(match &intent.launch_role {
                ServiceLaunchRole::IsolatedEngineSupervisor(_) => "--service-type=notify",
                ServiceLaunchRole::Ordinary | ServiceLaunchRole::OwnedLocalJob(_) => {
                    "--service-type=exec"
                }
            })
            .arg(format!("--unit={}", intent.unit_name))
            .arg(format!("--description={description}"))
            .arg("--property=KillMode=control-group")
            .arg("--property=Restart=no")
            .arg(format!("--property=TimeoutStopSec={timeout}"))
            .arg({
                let mut property = OsString::from("--property=WorkingDirectory=");
                property.push(&prepared.recipe.working_directory);
                property
            });
        match &intent.launch_role {
            ServiceLaunchRole::Ordinary | ServiceLaunchRole::OwnedLocalJob(_) => {}
            ServiceLaunchRole::IsolatedEngineSupervisor(coordinator) => {
                command.arg("--property=PrivateNetwork=yes").arg(format!(
                    "--property=OpenFile=/proc/{}/ns/net:l3ms-coordinator-netns:read-only",
                    coordinator.pid()
                ));
            }
        }
        command
            .arg("--")
            .arg(&prepared.recipe.executable)
            .args(&prepared.recipe.arguments);
        match execute(command, context) {
            Ok(_) => {}
            Err((kind, false)) => {
                return Err(fail(StartDisposition::NotDispatched, kind));
            }
            Err((kind, true)) => {
                return Err(fail(StartDisposition::OwnershipUncertain, kind));
            }
        }

        loop {
            if let Err(kind) = context_error(context) {
                return Err(fail(StartDisposition::OwnershipUncertain, kind));
            }
            if let Ok(Some(snapshot)) = self.show_unit(&intent, context) {
                match owned_snapshot(&intent, &snapshot, None) {
                    Ok(ManagedServiceState::Running) => {
                        if matches!(
                            &intent.launch_role,
                            ServiceLaunchRole::IsolatedEngineSupervisor(_)
                        ) {
                            let namespace_verified = match self
                                .verify_supervisor_namespace(&intent, &snapshot, context)
                            {
                                Ok(verified) => verified,
                                Err(kind) => {
                                    return Err(fail(StartDisposition::OwnershipUncertain, kind));
                                }
                            };
                            if !namespace_verified {
                                return Err(fail(
                                    StartDisposition::OwnershipUncertain,
                                    ManagedServiceErrorKind::OwnershipMismatch,
                                ));
                            }
                        }
                        if let ServiceLaunchRole::OwnedLocalJob(binding) = &intent.launch_role {
                            if !coordinator_identity_is_current(binding.coordinator_identity()) {
                                return Err(fail(
                                    StartDisposition::OwnershipUncertain,
                                    ManagedServiceErrorKind::OwnershipMismatch,
                                ));
                            }
                        }
                        let invocation_id = snapshot.invocation_id.clone();
                        let cgroup = snapshot.control_group.clone();
                        let main_pid = snapshot.main_pid;
                        return Ok(OwnedService {
                            intent: intent.clone(),
                            invocation_id,
                            cgroup,
                            main_pid,
                        });
                    }
                    Ok(ManagedServiceState::Unknown) => {}
                    Ok(ManagedServiceState::Stopped) | Err(_) => {
                        return Err(fail(
                            StartDisposition::OwnershipUncertain,
                            ManagedServiceErrorKind::OwnershipMismatch,
                        ));
                    }
                }
            }
            if context.remaining().is_zero() {
                return Err(fail(
                    StartDisposition::OwnershipUncertain,
                    ManagedServiceErrorKind::DeadlineExceeded,
                ));
            }
            thread::sleep(POLL_QUANTUM.min(context.remaining()));
        }
    }

    /// Reconcile a pending durable intent. Only a fully matching active
    /// transient service produces a handle; this method never sends stop.
    pub fn reconcile(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<ReconcileResult, ManagedServiceErrorKind> {
        self.reconcile_inner(intent, context, false)
    }

    /// Reconcile an intent returned by the secure retained-intent store. Only
    /// this path may issue stop-only recovery for an isolated supervisor whose
    /// original coordinator process can no longer be verified.
    pub(super) fn reconcile_restored_intent(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<ReconcileResult, ManagedServiceErrorKind> {
        self.reconcile_inner(intent, context, true)
    }

    fn reconcile_inner(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
        restored_intent: bool,
    ) -> Result<ReconcileResult, ManagedServiceErrorKind> {
        if intent.coordinator_id != self.coordinator_id || !valid_intent(intent) {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        context_error(context)?;
        let Some(snapshot) = self.show_unit(intent, context)? else {
            return Ok(ReconcileResult::AbsentNow);
        };
        match owned_snapshot(intent, &snapshot, None) {
            Ok(ManagedServiceState::Running) => {
                if let ServiceLaunchRole::OwnedLocalJob(binding) = &intent.launch_role {
                    if restored_intent {
                        return Ok(ReconcileResult::StopOnly(Box::new(StopOnlyService {
                            intent: intent.clone(),
                            invocation_id: snapshot.invocation_id,
                            cgroup: snapshot.control_group,
                            main_pid: snapshot.main_pid,
                        })));
                    }
                    if !coordinator_identity_is_current(binding.coordinator_identity()) {
                        return Err(ManagedServiceErrorKind::OwnershipMismatch);
                    }
                }
                if matches!(
                    &intent.launch_role,
                    ServiceLaunchRole::IsolatedEngineSupervisor(_)
                ) && !self.verify_supervisor_namespace(intent, &snapshot, context)?
                {
                    if restored_intent {
                        return Ok(ReconcileResult::StopOnly(Box::new(StopOnlyService {
                            intent: intent.clone(),
                            invocation_id: snapshot.invocation_id,
                            cgroup: snapshot.control_group,
                            main_pid: snapshot.main_pid,
                        })));
                    }
                    return Err(ManagedServiceErrorKind::OwnershipMismatch);
                }
                Ok(ReconcileResult::Owned(Box::new(OwnedService {
                    intent: intent.clone(),
                    invocation_id: snapshot.invocation_id,
                    cgroup: snapshot.control_group,
                    main_pid: snapshot.main_pid,
                })))
            }
            Ok(ManagedServiceState::Stopped | ManagedServiceState::Unknown) | Err(_) => {
                Ok(ReconcileResult::Unknown)
            }
        }
    }

    /// Query state only when the current unit still matches the handle's
    /// InvocationID and cgroup. A disappeared unit is unknown to this query.
    pub fn state(
        &self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<ManagedServiceState, ManagedServiceErrorKind> {
        self.validate_handle(service)?;
        context_error(context)?;
        let Some(snapshot) = self.show_unit(&service.intent, context)? else {
            return Ok(ManagedServiceState::Unknown);
        };
        if snapshot.active_state == "active" && snapshot.main_pid != service.main_pid {
            return Ok(ManagedServiceState::Unknown);
        }
        let state = owned_snapshot(
            &service.intent,
            &snapshot,
            Some((&service.invocation_id, &service.cgroup)),
        )
        .unwrap_or(ManagedServiceState::Unknown);
        if state == ManagedServiceState::Running
            && matches!(
                &service.intent.launch_role,
                ServiceLaunchRole::IsolatedEngineSupervisor(_)
            )
            && !self.verify_supervisor_namespace(&service.intent, &snapshot, context)?
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        if state == ManagedServiceState::Running {
            if let ServiceLaunchRole::OwnedLocalJob(binding) = &service.intent.launch_role {
                if !coordinator_identity_is_current(binding.coordinator_identity()) {
                    return Ok(ManagedServiceState::Unknown);
                }
            }
        }
        Ok(state)
    }

    /// Recheck a stop receipt without issuing another stop command. A
    /// missing transient unit is acceptable only because this opaque receipt
    /// already includes a generation-matched stop acknowledgment. If a unit
    /// remains, it must still be the exact inactive invocation and cgroup;
    /// active, replaced, or unknown state fails closed. The caller must keep
    /// the host guard and trust that no concurrent same-user manager client
    /// replaces or restarts this unit.
    pub(crate) fn verify_stopped_receipt(
        &self,
        receipt: &StoppedService,
        context: &OperationContext<'_>,
    ) -> Result<(), ManagedServiceErrorKind> {
        let intent = receipt.intent_record();
        if intent.coordinator_id != self.coordinator_id
            || !valid_intent(intent)
            || !valid_invocation_id(&receipt.invocation_id)
            || !valid_cgroup(&receipt.cgroup, &intent.unit_name)
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        context_error(context)?;
        let Some(snapshot) = self.show_unit(intent, context)? else {
            context_error(context)?;
            return Ok(());
        };
        match owned_snapshot(
            intent,
            &snapshot,
            Some((&receipt.invocation_id, &receipt.cgroup)),
        )? {
            ManagedServiceState::Stopped => Ok(()),
            ManagedServiceState::Running => Err(ManagedServiceErrorKind::ServiceNotActive),
            ManagedServiceState::Unknown => Err(ManagedServiceErrorKind::UnitStateUnverifiable),
        }
    }

    /// Stop only after the active generation matches InvocationID and cgroup.
    /// systemctl stop is name-targeted, so the coordinator host lock must
    /// serialize its callers, and external same-user restarts/replacements
    /// must not race this method. The generation check is not atomic. Success
    /// is a manager stop acknowledgment, not proof that the cgroup is empty or
    /// that a GPU/resource was released.
    pub fn stop(
        &self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<StoppedService, ManagedServiceErrorKind> {
        self.validate_handle(service)?;
        self.stop_exact_generation(
            &service.intent,
            &service.invocation_id,
            &service.cgroup,
            service.main_pid,
            context,
        )
    }

    /// Stop the exact generation represented by a restricted recovery token.
    /// This path is available only to the owner after secure WAL restoration;
    /// it never creates an OwnedService handle.
    pub(super) fn stop_stop_only(
        &self,
        service: &StopOnlyService,
        context: &OperationContext<'_>,
    ) -> Result<StoppedService, ManagedServiceErrorKind> {
        self.validate_stop_only(service)?;
        self.stop_exact_generation(
            &service.intent,
            &service.invocation_id,
            &service.cgroup,
            service.main_pid,
            context,
        )
    }

    fn stop_exact_generation(
        &self,
        intent: &ServiceIntentRecord,
        invocation_id: &str,
        cgroup: &str,
        main_pid: u32,
        context: &OperationContext<'_>,
    ) -> Result<StoppedService, ManagedServiceErrorKind> {
        context_error(context)?;
        let Some(snapshot) = self.show_unit(intent, context)? else {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        };
        if snapshot.active_state == "active" && snapshot.main_pid != main_pid {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        match owned_snapshot(intent, &snapshot, Some((invocation_id, cgroup)))? {
            ManagedServiceState::Running if snapshot.main_pid == main_pid => {}
            ManagedServiceState::Running => {
                return Err(ManagedServiceErrorKind::OwnershipMismatch);
            }
            ManagedServiceState::Stopped | ManagedServiceState::Unknown => {
                return Err(ManagedServiceErrorKind::ServiceNotActive);
            }
        }

        let mut command = Command::new(&self.systemctl);
        command
            .arg("--user")
            .arg("--no-ask-password")
            .arg("stop")
            .arg(&intent.unit_name);
        execute(command, context).map_err(|(kind, _)| kind)?;

        let Some(after) = self.show_unit(intent, context)? else {
            // A successful stop followed by transient-unit unloading confirms
            // that no unit is currently addressable under this generated name.
            // The matching pre-stop generation and successful systemctl
            // acknowledgment are required before issuing the receipt.
            context_error(context)?;
            return Ok(stopped_receipt_for(intent, invocation_id, cgroup));
        };
        let after_state = owned_snapshot(intent, &after, Some((invocation_id, cgroup)))
            .unwrap_or(ManagedServiceState::Unknown);
        if after.active_state == "active" && after.main_pid != main_pid {
            return Err(ManagedServiceErrorKind::StopUnconfirmed);
        }
        match after_state {
            ManagedServiceState::Stopped => {
                context_error(context)?;
                Ok(stopped_receipt_for(intent, invocation_id, cgroup))
            }
            ManagedServiceState::Running | ManagedServiceState::Unknown => {
                Err(ManagedServiceErrorKind::StopUnconfirmed)
            }
        }
    }

    fn validate_handle(&self, service: &OwnedService) -> Result<(), ManagedServiceErrorKind> {
        if service.intent.coordinator_id != self.coordinator_id
            || !valid_intent(&service.intent)
            || !valid_invocation_id(&service.invocation_id)
            || !valid_cgroup(&service.cgroup, &service.intent.unit_name)
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        Ok(())
    }

    fn validate_stop_only(&self, service: &StopOnlyService) -> Result<(), ManagedServiceErrorKind> {
        if service.intent.coordinator_id != self.coordinator_id
            || !valid_intent(&service.intent)
            || !matches!(
                &service.intent.launch_role,
                ServiceLaunchRole::IsolatedEngineSupervisor(_)
                    | ServiceLaunchRole::OwnedLocalJob(_)
            )
            || !valid_invocation_id(&service.invocation_id)
            || !valid_cgroup(&service.cgroup, &service.intent.unit_name)
            || service.main_pid <= 1
            || service.main_pid > i32::MAX as u32
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        Ok(())
    }

    fn preflight_unit(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<bool, ManagedServiceErrorKind> {
        // Check loaded units and installed unit files. Static or pre-existing
        // services are never claimed, including inactive ones.
        let loaded = self.list_unit("list-units", intent, context)?;
        if !loaded.stdout().iter().all(u8::is_ascii_whitespace) {
            return Ok(false);
        }
        let installed = self.list_unit("list-unit-files", intent, context)?;
        Ok(installed.stdout().iter().all(u8::is_ascii_whitespace))
    }

    fn list_unit(
        &self,
        verb: &str,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<process::ProcessOutput, ManagedServiceErrorKind> {
        let mut command = Command::new(&self.systemctl);
        command
            .arg("--user")
            .arg("--no-pager")
            .arg("--no-legend")
            .arg(verb)
            .arg("--all")
            .arg(&intent.unit_name);
        execute(command, context).map_err(|(kind, _)| kind)
    }

    fn show_unit(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<Option<UnitSnapshot>, ManagedServiceErrorKind> {
        let mut command = Command::new(&self.systemctl);
        command
            .arg("--user")
            .arg("--no-pager")
            .arg("show")
            .arg("--all");
        for property in [
            "LoadState",
            "ActiveState",
            "MainPID",
            "Transient",
            "InvocationID",
            "ControlGroup",
            "FragmentPath",
            "Description",
            "Restart",
            "KillMode",
            "Type",
            "PrivateNetwork",
            "JoinsNamespaceOf",
        ] {
            command.arg(format!("--property={property}"));
        }
        command.arg(&intent.unit_name);
        let output = execute(command, context).map_err(|(kind, _)| kind)?;
        parse_snapshot(output.stdout())
    }

    fn verify_supervisor_namespace(
        &self,
        intent: &ServiceIntentRecord,
        before: &UnitSnapshot,
        context: &OperationContext<'_>,
    ) -> Result<bool, ManagedServiceErrorKind> {
        let ServiceLaunchRole::IsolatedEngineSupervisor(coordinator) = &intent.launch_role else {
            return Ok(true);
        };
        let current_coordinator =
            match super::engine_transport::CoordinatorIdentityDescriptor::capture_current() {
                Ok(current) => current,
                Err(_) => return Ok(false),
            };
        if current_coordinator != *coordinator {
            return Ok(false);
        }
        let pid = before.main_pid;
        if pid <= 1 || pid > i32::MAX as u32 {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        context_error(context)?;
        let first_namespace = fs::File::open(format!("/proc/{pid}/ns/net"))
            .map_err(|_| ManagedServiceErrorKind::OwnershipMismatch)?;
        let first_identity = namespace_file_identity(&first_namespace)?;
        if first_identity
            == (
                coordinator.network_namespace_device(),
                coordinator.network_namespace_inode(),
            )
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }

        context_error(context)?;
        let after = self
            .show_unit(intent, context)?
            .ok_or(ManagedServiceErrorKind::OwnershipMismatch)?;
        if owned_snapshot(
            intent,
            &after,
            Some((&before.invocation_id, &before.control_group)),
        )? != ManagedServiceState::Running
            || after.main_pid != pid
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        context_error(context)?;
        let second_namespace = fs::File::open(format!("/proc/{pid}/ns/net"))
            .map_err(|_| ManagedServiceErrorKind::OwnershipMismatch)?;
        let second_identity = namespace_file_identity(&second_namespace)?;
        if second_identity != first_identity
            || second_identity
                == (
                    coordinator.network_namespace_device(),
                    coordinator.network_namespace_inode(),
                )
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        context_error(context)?;
        let current_coordinator =
            match super::engine_transport::CoordinatorIdentityDescriptor::capture_current() {
                Ok(current) => current,
                Err(_) => return Ok(false),
            };
        if current_coordinator != *coordinator {
            return Ok(false);
        }
        Ok(true)
    }
}

#[cfg(test)]
impl ManagedServiceController {
    pub(crate) fn with_test_executables(
        coordinator_id: impl Into<String>,
        systemd_run: PathBuf,
        systemctl: PathBuf,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let coordinator_id = coordinator_id.into();
        validate_coordinator_id(&coordinator_id)?;
        verify_executable(&systemd_run)?;
        verify_executable(&systemctl)?;
        Ok(Self {
            coordinator_id,
            systemd_run,
            systemctl,
        })
    }
}

fn stopped_receipt_for(
    intent: &ServiceIntentRecord,
    invocation_id: &str,
    cgroup: &str,
) -> StoppedService {
    StoppedService {
        intent: intent.clone(),
        invocation_id: invocation_id.to_owned(),
        cgroup: cgroup.to_owned(),
    }
}

struct UnitSnapshot {
    load_state: String,
    active_state: String,
    main_pid: u32,
    transient: String,
    invocation_id: String,
    control_group: String,
    _fragment_path: String,
    description: String,
    restart: String,
    kill_mode: String,
    unit_type: String,
    private_network: String,
    joins_namespace_of: String,
}

fn parse_snapshot(stdout: &[u8]) -> Result<Option<UnitSnapshot>, ManagedServiceErrorKind> {
    let text =
        std::str::from_utf8(stdout).map_err(|_| ManagedServiceErrorKind::UnitStateUnverifiable)?;
    let mut fields = std::collections::BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            return Err(ManagedServiceErrorKind::UnitStateUnverifiable);
        };
        if fields.insert(key, value).is_some() {
            return Err(ManagedServiceErrorKind::UnitStateUnverifiable);
        }
    }
    let get = |name| {
        fields
            .get(name)
            .copied()
            .ok_or(ManagedServiceErrorKind::UnitStateUnverifiable)
    };
    let load_state = get("LoadState")?.to_owned();
    if load_state == "not-found" {
        return Ok(None);
    }
    let snapshot = UnitSnapshot {
        load_state,
        active_state: get("ActiveState")?.to_owned(),
        main_pid: get("MainPID")?
            .parse::<u32>()
            .ok()
            .filter(|pid| *pid <= i32::MAX as u32)
            .ok_or(ManagedServiceErrorKind::UnitStateUnverifiable)?,
        transient: get("Transient")?.to_owned(),
        invocation_id: get("InvocationID")?.to_owned(),
        control_group: get("ControlGroup")?.to_owned(),
        _fragment_path: get("FragmentPath")?.to_owned(),
        description: get("Description")?.to_owned(),
        restart: get("Restart")?.to_owned(),
        kill_mode: get("KillMode")?.to_owned(),
        unit_type: get("Type")?.to_owned(),
        private_network: get("PrivateNetwork")?.to_owned(),
        joins_namespace_of: get("JoinsNamespaceOf")?.to_owned(),
    };
    Ok(Some(snapshot))
}

fn owned_snapshot(
    intent: &ServiceIntentRecord,
    snapshot: &UnitSnapshot,
    generation: Option<(&str, &str)>,
) -> Result<ManagedServiceState, ManagedServiceErrorKind> {
    if snapshot.load_state != "loaded"
        || snapshot.transient != "yes"
        || snapshot.description != description_for(intent)
        || snapshot.restart != "no"
        || snapshot.kill_mode != "control-group"
        || !snapshot_matches_role(intent, snapshot)
    {
        return Err(ManagedServiceErrorKind::OwnershipMismatch);
    }
    let expected = generation;
    match snapshot.active_state.as_str() {
        // The transient job may be visible before systemd has assigned an
        // invocation ID or cgroup. Keep polling instead of failing ownership.
        "activating" | "deactivating" | "reloading" => {
            if let Some((invocation_id, cgroup)) = expected {
                if (!snapshot.invocation_id.is_empty() && snapshot.invocation_id != invocation_id)
                    || (!snapshot.control_group.is_empty() && snapshot.control_group != cgroup)
                {
                    return Err(ManagedServiceErrorKind::OwnershipMismatch);
                }
            }
            Ok(ManagedServiceState::Unknown)
        }
        "active" | "inactive" => {
            if !valid_invocation_id(&snapshot.invocation_id)
                || !valid_cgroup(&snapshot.control_group, &intent.unit_name)
            {
                return Err(ManagedServiceErrorKind::OwnershipMismatch);
            }
            if let Some((invocation_id, cgroup)) = expected {
                if snapshot.invocation_id != invocation_id || snapshot.control_group != cgroup {
                    return Err(ManagedServiceErrorKind::OwnershipMismatch);
                }
            }
            if snapshot.active_state == "active" {
                if snapshot.main_pid <= 1 {
                    return Err(ManagedServiceErrorKind::OwnershipMismatch);
                }
                Ok(ManagedServiceState::Running)
            } else {
                Ok(ManagedServiceState::Stopped)
            }
        }
        _ => Ok(ManagedServiceState::Unknown),
    }
}

fn snapshot_matches_role(intent: &ServiceIntentRecord, snapshot: &UnitSnapshot) -> bool {
    match &intent.launch_role {
        ServiceLaunchRole::Ordinary | ServiceLaunchRole::OwnedLocalJob(_) => {
            snapshot.unit_type == "exec"
                && snapshot.private_network == "no"
                && snapshot.joins_namespace_of.is_empty()
        }
        ServiceLaunchRole::IsolatedEngineSupervisor(_) => {
            snapshot.unit_type == "notify"
                && snapshot.private_network == "yes"
                && snapshot.joins_namespace_of.is_empty()
        }
    }
}

fn namespace_file_identity(file: &fs::File) -> Result<(u64, u64), ManagedServiceErrorKind> {
    let metadata = file
        .metadata()
        .map_err(|_| ManagedServiceErrorKind::OwnershipMismatch)?;
    let mut filesystem: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatfs(file.as_raw_fd(), &mut filesystem) } != 0
        || filesystem.f_type != libc::NSFS_MAGIC
    {
        return Err(ManagedServiceErrorKind::OwnershipMismatch);
    }
    Ok((metadata.dev(), metadata.ino()))
}

fn execute(
    command: Command,
    context: &OperationContext<'_>,
) -> Result<process::ProcessOutput, (ManagedServiceErrorKind, bool)> {
    let limits = ProcessLimits::new(0, MAX_MANAGER_OUTPUT_BYTES)
        .map_err(|_| (ManagedServiceErrorKind::ManagerCommandFailed, false))?;
    process::execute(
        command,
        None,
        limits,
        context,
        RecoveryPolicy::new(Duration::from_secs(5))
            .expect("fixed process recovery timeout is valid"),
    )
    .map_err(|failure| {
        use process::ProcessFailureDisposition as Disposition;
        let may_have_spawned = !matches!(failure.disposition(), Disposition::NoChildSpawned);
        let kind = match failure.kind() {
            process::ProcessFailureKind::Cancelled => ManagedServiceErrorKind::Cancelled,
            process::ProcessFailureKind::DeadlineExceeded => {
                ManagedServiceErrorKind::DeadlineExceeded
            }
            _ if context.is_cancelled() => ManagedServiceErrorKind::Cancelled,
            _ if context.remaining().is_zero() => ManagedServiceErrorKind::DeadlineExceeded,
            _ => ManagedServiceErrorKind::ManagerCommandFailed,
        };
        (kind, may_have_spawned)
    })
}

fn context_error(context: &OperationContext<'_>) -> Result<(), ManagedServiceErrorKind> {
    if context.is_cancelled() {
        Err(ManagedServiceErrorKind::Cancelled)
    } else if context.remaining().is_zero() {
        Err(ManagedServiceErrorKind::DeadlineExceeded)
    } else {
        Ok(())
    }
}

fn validate_coordinator_id(value: &str) -> Result<(), ManagedServiceErrorKind> {
    if value.is_empty()
        || value.len() > MAX_COORDINATOR_ID_BYTES
        || !value.as_bytes()[0].is_ascii_lowercase()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || value.ends_with('-')
    {
        return Err(ManagedServiceErrorKind::InvalidCoordinatorId);
    }
    Ok(())
}

fn valid_intent(intent: &ServiceIntentRecord) -> bool {
    if validate_coordinator_id(&intent.coordinator_id).is_err()
        || intent.launch_identity.len() != 64
        || intent.stop_timeout_millis == 0
        || intent.stop_timeout_millis > MAX_STOP_TIMEOUT.as_millis() as u64
        || !intent
            .launch_identity
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return false;
    }
    if intent
        .runtime_identity
        .as_ref()
        .is_some_and(|identity| !valid_runtime_identity(identity))
    {
        return false;
    }
    if validate_role_runtime_binding(&intent.launch_role, intent.runtime_identity.as_ref()).is_err()
    {
        return false;
    }
    if validate_launch_role(&intent.launch_role).is_err() {
        return false;
    }
    let prefix = format!("l3ms-{}-", intent.coordinator_id);
    let Some(random) = intent
        .unit_name
        .strip_prefix(&prefix)
        .and_then(|name| name.strip_suffix(".service"))
    else {
        return false;
    };
    random.len() == UNIT_RANDOM_BYTES * 2
        && random
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn description_for(intent: &ServiceIntentRecord) -> String {
    let description = format!(
        "l3ms-managed:{}:{}:{}:{}",
        intent.coordinator_id, intent.unit_name, intent.launch_identity, intent.stop_timeout_millis
    );
    match &intent.runtime_identity {
        Some(identity) => format!(
            "{description}:runtime={}:{}",
            identity.lane,
            identity.as_str()
        ),
        None => description,
    }
}

fn valid_runtime_identity(identity: &RuntimeIdentity) -> bool {
    let name = identity.as_str();
    !name.is_empty()
        && name.len() <= MAX_RUNTIME_ID_BYTES
        && name.as_bytes()[0].is_ascii_alphanumeric()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

pub(super) fn runtime_identity_from_persisted_parts(
    lane: &str,
    name: String,
) -> Result<RuntimeIdentity, ManagedServiceErrorKind> {
    let lane = match lane {
        "language" => Lane::Language,
        "media" => Lane::Media,
        "decisions" => Lane::Decisions,
        _ => return Err(ManagedServiceErrorKind::InvalidRuntimeIdentity),
    };
    let identity = RuntimeIdentity::new(lane, name)
        .map_err(|_| ManagedServiceErrorKind::InvalidRuntimeIdentity)?;
    if !valid_runtime_identity(&identity) {
        return Err(ManagedServiceErrorKind::InvalidRuntimeIdentity);
    }
    Ok(identity)
}

fn valid_invocation_id(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn valid_cgroup(value: &str, unit_name: &str) -> bool {
    value.starts_with('/')
        && value.len() <= 4096
        && value.rsplit('/').next() == Some(unit_name)
        && value.split('/').skip(1).all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_.:@-".contains(&byte))
        })
}

fn launch_fingerprint(recipe: &LaunchRecipe) -> String {
    // Four independently seeded FNV-1a streams produce a stable recipe
    // fingerprint. It distinguishes persisted intents; random unit names and
    // systemd generation fields remain the ownership proof.
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut states = [
        0xcbf2_9ce4_8422_2325,
        0x8422_2325_cbf2_9ce4,
        0x9e37_79b9_7f4a_7c15,
        0xd6e8_feb8_6659_fd93,
    ];
    let mut feed = |bytes: &[u8]| {
        for state in &mut states {
            for byte in (bytes.len() as u64)
                .to_le_bytes()
                .into_iter()
                .chain(bytes.iter().copied())
            {
                *state = (*state ^ u64::from(byte)).wrapping_mul(PRIME);
            }
        }
    };
    feed(b"l3ms-launch-recipe-v1");
    feed(recipe.executable.as_os_str().as_bytes());
    feed(recipe.working_directory.as_os_str().as_bytes());
    for argument in &recipe.arguments {
        feed(argument.as_os_str().as_bytes());
    }
    if let Some(interpreter) = &recipe.verified_python {
        feed(b"verified-python-interpreter-v1");
        feed(interpreter.configured_path().as_os_str().as_bytes());
        feed(interpreter.target_path().as_os_str().as_bytes());
        feed(interpreter.venv_directory().as_os_str().as_bytes());
        feed(interpreter.identity_wire().as_bytes());
    }
    format!(
        "{:016x}{:016x}{:016x}{:016x}",
        states[0], states[1], states[2], states[3]
    )
}

fn launch_fingerprint_for_role(
    recipe: &LaunchRecipe,
    role: &ServiceLaunchRole,
    runtime_identity: Option<&RuntimeIdentity>,
) -> String {
    if matches!(role, ServiceLaunchRole::Ordinary) {
        return launch_fingerprint(recipe);
    }
    let base = launch_fingerprint(recipe);
    let role = role_bytes(role);
    let lane = runtime_identity
        .map(|identity| identity.lane.as_str())
        .unwrap_or("");
    let runtime = runtime_identity.map(RuntimeIdentity::as_str).unwrap_or("");
    fingerprint_fields(
        b"l3ms-launch-role-v1",
        &[
            base.as_bytes(),
            role.as_slice(),
            lane.as_bytes(),
            runtime.as_bytes(),
        ],
    )
}

fn role_bytes(role: &ServiceLaunchRole) -> Vec<u8> {
    let mut bytes = Vec::new();
    match role {
        ServiceLaunchRole::Ordinary => bytes.extend_from_slice(b"ordinary"),
        ServiceLaunchRole::IsolatedEngineSupervisor(coordinator) => {
            bytes.extend_from_slice(b"isolated-engine-supervisor");
            bytes.extend_from_slice(&coordinator.pid().to_le_bytes());
            bytes.extend_from_slice(&coordinator.uid().to_le_bytes());
            bytes.extend_from_slice(&coordinator.start_time_ticks().to_le_bytes());
            bytes.extend_from_slice(&coordinator.network_namespace_device().to_le_bytes());
            bytes.extend_from_slice(&coordinator.network_namespace_inode().to_le_bytes());
        }
        ServiceLaunchRole::OwnedLocalJob(binding) => {
            bytes.extend_from_slice(b"owned-local-job-v1");
            bytes.extend_from_slice(&binding.job_id.to_le_bytes());
            append_fingerprint_field(&mut bytes, binding.request_id.as_bytes());
            append_fingerprint_field(&mut bytes, binding.job_directory.as_os_str().as_bytes());
            bytes.extend_from_slice(&binding.directory_device.to_le_bytes());
            bytes.extend_from_slice(&binding.directory_inode.to_le_bytes());
            bytes.extend_from_slice(&binding.coordinator.pid().to_le_bytes());
            bytes.extend_from_slice(&binding.coordinator.uid().to_le_bytes());
            bytes.extend_from_slice(&binding.coordinator.start_time_ticks().to_le_bytes());
            bytes.extend_from_slice(&binding.coordinator.network_namespace_device().to_le_bytes());
            bytes.extend_from_slice(&binding.coordinator.network_namespace_inode().to_le_bytes());
        }
    }
    bytes
}

fn append_fingerprint_field(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value);
}

fn fingerprint_fields(tag: &[u8], fields: &[&[u8]]) -> String {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut states = [
        0xcbf2_9ce4_8422_2325,
        0x8422_2325_cbf2_9ce4,
        0x9e37_79b9_7f4a_7c15,
        0xd6e8_feb8_6659_fd93,
    ];
    let mut feed = |bytes: &[u8]| {
        for state in &mut states {
            for byte in (bytes.len() as u64)
                .to_le_bytes()
                .into_iter()
                .chain(bytes.iter().copied())
            {
                *state = (*state ^ u64::from(byte)).wrapping_mul(PRIME);
            }
        }
    };
    feed(tag);
    for field in fields {
        feed(field);
    }
    format!(
        "{:016x}{:016x}{:016x}{:016x}",
        states[0], states[1], states[2], states[3]
    )
}

fn validate_launch_role(role: &ServiceLaunchRole) -> Result<(), ManagedServiceErrorKind> {
    match role {
        ServiceLaunchRole::Ordinary => Ok(()),
        ServiceLaunchRole::IsolatedEngineSupervisor(descriptor) => {
            validate_coordinator_descriptor(*descriptor)
        }
        ServiceLaunchRole::OwnedLocalJob(binding) => validate_owned_job_binding_parts(
            &binding.runtime_identity,
            binding.job_id,
            &binding.request_id,
            &binding.job_directory,
            binding.directory_device,
            binding.directory_inode,
            binding.coordinator,
        ),
    }
}

fn validate_role_runtime_binding(
    role: &ServiceLaunchRole,
    runtime_identity: Option<&RuntimeIdentity>,
) -> Result<(), ManagedServiceErrorKind> {
    match role {
        ServiceLaunchRole::Ordinary => Ok(()),
        ServiceLaunchRole::IsolatedEngineSupervisor(_) if runtime_identity.is_some() => Ok(()),
        ServiceLaunchRole::OwnedLocalJob(binding)
            if runtime_identity == Some(&binding.runtime_identity) =>
        {
            Ok(())
        }
        _ => Err(ManagedServiceErrorKind::InvalidRuntimeIdentity),
    }
}

fn validate_owned_job_fields(
    runtime_identity: &RuntimeIdentity,
    job_id: u64,
    request_id: &str,
    coordinator: super::engine_transport::CoordinatorIdentityDescriptor,
) -> Result<(), ManagedServiceErrorKind> {
    if !valid_runtime_identity(runtime_identity) {
        return Err(ManagedServiceErrorKind::InvalidRuntimeIdentity);
    }
    if job_id == 0
        || request_id.is_empty()
        || request_id.len() > MAX_OWNED_JOB_REQUEST_ID_BYTES
        || super::ipc::validate_request_id(request_id).is_err()
    {
        return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
    }
    validate_coordinator_descriptor(coordinator)
        .map_err(|_| ManagedServiceErrorKind::InvalidOwnedJobBinding)
}

fn validate_owned_job_binding_parts(
    runtime_identity: &RuntimeIdentity,
    job_id: u64,
    request_id: &str,
    job_directory: &Path,
    _directory_device: u64,
    directory_inode: u64,
    coordinator: super::engine_transport::CoordinatorIdentityDescriptor,
) -> Result<(), ManagedServiceErrorKind> {
    validate_owned_job_fields(runtime_identity, job_id, request_id, coordinator)?;
    if !canonical_owned_job_directory(job_directory) || directory_inode == 0 {
        return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
    }
    Ok(())
}

fn supplied_job_directory_path_is_acceptable(path: &Path) -> bool {
    path.is_absolute()
        && path.as_os_str().as_bytes().len() <= MAX_OWNED_JOB_DIRECTORY_BYTES
        && !path.as_os_str().as_bytes().contains(&0)
        && !path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
}

fn canonical_owned_job_directory(path: &Path) -> bool {
    if !path.is_absolute()
        || path.as_os_str().as_bytes().len() > MAX_OWNED_JOB_DIRECTORY_BYTES
        || path.as_os_str().as_bytes().contains(&0)
        || path.to_str().is_none()
    {
        return false;
    }
    let mut normalized = PathBuf::new();
    let mut normal_components = 0usize;
    for component in path.components() {
        match component {
            Component::RootDir => normalized.push("/"),
            Component::Normal(part) => {
                normal_components += 1;
                normalized.push(part);
            }
            _ => return false,
        }
    }
    normal_components > 0 && normalized.as_os_str().as_bytes() == path.as_os_str().as_bytes()
}

fn open_owned_job_directory(path: &Path) -> Result<fs::File, ManagedServiceErrorKind> {
    if !supplied_job_directory_path_is_acceptable(path) {
        return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
    }
    let root = CString::new("/").expect("static path has no NUL");
    let root_fd = unsafe {
        libc::open(
            root.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if root_fd < 0 {
        return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
    }
    let mut current = unsafe { fs::File::from_raw_fd(root_fd) };
    for component in path.components() {
        let Component::Normal(part) = component else {
            if matches!(component, Component::RootDir) {
                continue;
            }
            return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
        };
        let name = CString::new(part.as_bytes())
            .map_err(|_| ManagedServiceErrorKind::InvalidOwnedJobBinding)?;
        let fd = unsafe {
            libc::openat(
                current.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
        }
        current = unsafe { fs::File::from_raw_fd(fd) };
    }
    Ok(current)
}

fn owned_job_directory_identity(
    directory: &fs::File,
) -> Result<(u64, u64), ManagedServiceErrorKind> {
    let metadata = directory
        .metadata()
        .map_err(|_| ManagedServiceErrorKind::InvalidOwnedJobBinding)?;
    if !metadata.is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o7777 != OWNED_JOB_DIRECTORY_MODE
    {
        return Err(ManagedServiceErrorKind::InvalidOwnedJobBinding);
    }
    Ok((metadata.dev(), metadata.ino()))
}

fn validate_coordinator_descriptor(
    descriptor: super::engine_transport::CoordinatorIdentityDescriptor,
) -> Result<(), ManagedServiceErrorKind> {
    if descriptor.uid() != unsafe { libc::geteuid() }
        || super::engine_transport::CoordinatorIdentityDescriptor::from_parts(
            descriptor.pid(),
            descriptor.uid(),
            descriptor.start_time_ticks(),
            descriptor.network_namespace_device(),
            descriptor.network_namespace_inode(),
        )
        .is_err()
    {
        return Err(ManagedServiceErrorKind::InvalidLaunchRole);
    }
    Ok(())
}

fn coordinator_identity_is_current(
    expected: super::engine_transport::CoordinatorIdentityDescriptor,
) -> bool {
    super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
        .is_ok_and(|current| current == expected)
}

fn random_unit_suffix() -> Result<String, ManagedServiceErrorKind> {
    let mut bytes = [0u8; UNIT_RANDOM_BYTES];
    let mut offset = 0;
    while offset < bytes.len() {
        let result = unsafe {
            libc::getrandom(
                bytes[offset..].as_mut_ptr().cast(),
                bytes.len() - offset,
                libc::GRND_NONBLOCK,
            )
        };
        if result > 0 {
            offset += result as usize;
            continue;
        }
        if result < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(ManagedServiceErrorKind::RandomnessUnavailable);
    }
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use fmt::Write;
        let _ = write!(encoded, "{byte:02x}");
    }
    Ok(encoded)
}

fn duration_millis_ceil(value: Duration) -> u128 {
    value.as_millis()
        + if !value.subsec_nanos().is_multiple_of(1_000_000) {
            1
        } else {
            0
        }
}

fn verify_production_binary(path: &Path) -> Result<(), ManagedServiceErrorKind> {
    verify_executable(path)?;
    let metadata =
        fs::symlink_metadata(path).map_err(|_| ManagedServiceErrorKind::ExecutableUnavailable)?;
    if metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
        return Err(ManagedServiceErrorKind::ExecutableUnavailable);
    }
    Ok(())
}

fn verify_executable(path: &Path) -> Result<(), ManagedServiceErrorKind> {
    let metadata =
        fs::metadata(path).map_err(|_| ManagedServiceErrorKind::ExecutableUnavailable)?;
    if !path.is_absolute() || !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
        return Err(ManagedServiceErrorKind::ExecutableUnavailable);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Instant;
    use tempfile::TempDir;

    static NEVER_CANCEL: super::super::runtime::NeverCancel = super::super::runtime::NeverCancel;
    const INVOCATION: &str = "0123456789abcdef0123456789abcdef";

    struct Fixture {
        directory: TempDir,
        state: PathBuf,
        runner_arguments: PathBuf,
        recipe_executable: PathBuf,
        controller: ManagedServiceController,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
            let state = directory.path().join("state");
            let runner_arguments = directory.path().join("runner-arguments");
            let runner = directory.path().join("systemd-run");
            let systemctl = directory.path().join("systemctl");
            let recipe_executable = directory.path().join("service");
            write_executable(&runner, "#!/bin/sh\nexit 0\n");
            write_executable(&systemctl, "#!/bin/sh\nexit 0\n");
            write_executable(&recipe_executable, "#!/bin/sh\nexit 0\n");
            let controller = ManagedServiceController::with_test_executables(
                "test-coordinator",
                runner,
                systemctl,
            )
            .unwrap();
            Self {
                directory,
                state,
                runner_arguments,
                recipe_executable,
                controller,
            }
        }

        fn recipe(&self) -> LaunchRecipe {
            LaunchRecipe::new(
                &self.recipe_executable,
                self.directory.path(),
                [OsString::from("fixture-arg")],
            )
            .unwrap()
        }

        fn owned_job_binding(&self, job_id: u64, request_id: &str) -> OwnedLocalJobBinding {
            let directory = self.directory.path().join(format!("job-{job_id}"));
            fs::create_dir(&directory).unwrap();
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
            OwnedLocalJobBinding::capture(
                RuntimeIdentity::new(Lane::Media, "breeze-speech").unwrap(),
                job_id,
                request_id,
                directory,
                super::super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                    .unwrap(),
            )
            .unwrap()
        }

        fn install_scripts(&self, intent: &ServiceIntentRecord, launch_result: &str) {
            self.install_scripts_with_cgroup(intent, launch_result, false);
        }

        fn install_scripts_with_cgroup(
            &self,
            intent: &ServiceIntentRecord,
            launch_result: &str,
            clear_cgroup_when_inactive: bool,
        ) {
            let runner = self.directory.path().join("systemd-run");
            let runner_script = format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf active > '{}'\nexit {}\n",
                self.runner_arguments.display(),
                self.state.display(),
                launch_result
            );
            write_executable(&runner, &runner_script);
            let systemctl = self.directory.path().join("systemctl");
            let cgroup_output = if clear_cgroup_when_inactive {
                format!(
                    "if [ \"$active\" = inactive ]; then printf 'ControlGroup=\\n'; else printf 'ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/{}\\n'; fi",
                    intent.unit_name
                )
            } else {
                format!(
                    "printf 'ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/{}\\n'",
                    intent.unit_name
                )
            };
            let script = format!(
                r#"#!/bin/sh
verb=
for arg in "$@"; do
  case "$arg" in
    show|stop|list-units|list-unit-files) verb="$arg" ;;
  esac
done
case "$verb" in
  list-units|list-unit-files) exit 0 ;;
  stop) printf inactive > '{state}'; exit 0 ;;
  show)
    active=$(cat '{state}' 2>/dev/null || printf active)
    printf 'LoadState=loaded\n'
    printf 'ActiveState=%s\n' "$active"
            if [ "$active" = active ]; then printf 'MainPID=1234\n'; else printf 'MainPID=0\n'; fi
            printf 'InvocationID={invocation}\n'
            {cgroup_output}
            printf 'FragmentPath=/run/user/1000/systemd/transient/{unit_name}\n'
            printf 'Transient=yes\n'
            printf 'Description={description}\n'
            printf 'Restart=no\nKillMode=control-group\n'
            printf 'Type=exec\nPrivateNetwork=no\nJoinsNamespaceOf=\n'
    exit 0
    ;;
esac
exit 1
"#,
                state = self.state.display(),
                invocation = INVOCATION,
                description = description_for(intent),
                unit_name = intent.unit_name,
                cgroup_output = cgroup_output,
            );
            write_executable(&systemctl, &script);
        }

        fn context(&self, duration: Duration) -> OperationContext<'static> {
            OperationContext::new(Instant::now() + duration, &NEVER_CANCEL)
        }
    }

    fn write_executable(path: &Path, contents: &str) {
        let _guard = process::executable_fixture_guard();
        fs::write(path, contents).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[test]
    fn prepared_intent_is_unique_and_available_before_dispatch() {
        let fixture = Fixture::new();
        let first = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let second = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        assert_ne!(
            first.intent_record().unit_name(),
            second.intent_record().unit_name()
        );
        assert!(first
            .intent_record()
            .unit_name()
            .starts_with("l3ms-test-coordinator-"));
        assert_eq!(first.intent_record().launch_identity().len(), 64);
        assert_eq!(
            first.intent_record().launch_identity(),
            fixture.recipe().launch_identity()
        );
        assert!(!fixture.state.exists());
    }

    #[test]
    fn prepared_runtime_intent_is_bound_and_validated() {
        let fixture = Fixture::new();
        let runtime = RuntimeIdentity::new(Lane::Media, "comfyui").unwrap();
        let prepared = fixture
            .controller
            .prepare_start_for_runtime(runtime.clone(), fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        assert_eq!(prepared.intent_record().runtime_identity(), Some(&runtime));
        assert!(description_for(prepared.intent_record()).ends_with(":runtime=media:comfyui"));

        let invalid = RuntimeIdentity::new(Lane::Language, "bad/name").unwrap();
        assert_eq!(
            fixture
                .controller
                .prepare_start_for_runtime(invalid, fixture.recipe(), Duration::from_secs(10))
                .err(),
            Some(ManagedServiceErrorKind::InvalidRuntimeIdentity)
        );
    }

    #[test]
    fn owned_local_job_binding_captures_canonical_private_directory_and_redacts() {
        let fixture = Fixture::new();
        let binding = fixture.owned_job_binding(7, "caller:item-17");
        let canonical = fs::canonicalize(fixture.directory.path().join("job-7")).unwrap();
        assert_eq!(binding.job_id(), 7);
        assert_eq!(binding.request_id(), "caller:item-17");
        assert_eq!(binding.job_directory(), canonical);
        assert_eq!(binding.runtime_identity().as_str(), "breeze-speech");
        assert_eq!(
            binding.coordinator_identity(),
            super::super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                .unwrap()
        );
        assert!(binding.directory_inode() > 0);
        let debug = format!("{binding:?}");
        assert_eq!(debug, "OwnedLocalJobBinding([redacted])");
        assert!(!debug.contains("caller:item-17"));
        assert!(!debug.contains(canonical.to_str().unwrap()));
    }

    #[test]
    fn owned_local_job_binding_rejects_invalid_identity_and_unsafe_directories() {
        let fixture = Fixture::new();
        let directory = fixture.directory.path().join("unsafe-job");
        fs::create_dir(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "breeze-speech").unwrap();
        let coordinator =
            super::super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                .unwrap();
        assert_eq!(
            OwnedLocalJobBinding::capture(
                runtime.clone(),
                0,
                "caller:item-1",
                &directory,
                coordinator,
            )
            .unwrap_err(),
            ManagedServiceErrorKind::InvalidOwnedJobBinding
        );
        assert_eq!(
            OwnedLocalJobBinding::capture(
                runtime.clone(),
                1,
                "password=not-a-safe-id",
                &directory,
                coordinator,
            )
            .unwrap_err(),
            ManagedServiceErrorKind::InvalidOwnedJobBinding
        );
        assert_eq!(
            OwnedLocalJobBinding::capture(
                RuntimeIdentity::new(Lane::Media, "bad/name").unwrap(),
                1,
                "caller:item-1",
                &directory,
                coordinator,
            )
            .unwrap_err(),
            ManagedServiceErrorKind::InvalidRuntimeIdentity
        );

        let symlink_target = fixture.directory.path().join("symlink-target");
        fs::create_dir(&symlink_target).unwrap();
        fs::set_permissions(&symlink_target, fs::Permissions::from_mode(0o700)).unwrap();
        let symlink = fixture.directory.path().join("symlink-job");
        std::os::unix::fs::symlink(&symlink_target, &symlink).unwrap();
        assert_eq!(
            OwnedLocalJobBinding::capture(
                runtime.clone(),
                2,
                "caller:item-2",
                &symlink,
                coordinator,
            )
            .unwrap_err(),
            ManagedServiceErrorKind::InvalidOwnedJobBinding
        );

        fs::set_permissions(&directory, fs::Permissions::from_mode(0o750)).unwrap();
        assert_eq!(
            OwnedLocalJobBinding::capture(runtime, 3, "caller:item-3", &directory, coordinator,)
                .unwrap_err(),
            ManagedServiceErrorKind::InvalidOwnedJobBinding
        );
    }

    #[test]
    fn owned_local_job_fingerprint_binds_every_directory_and_job_identity() {
        let fixture = Fixture::new();
        let recipe = fixture.recipe();
        let binding = fixture.owned_job_binding(11, "caller:item-11");
        let prepared = fixture
            .controller
            .prepare_owned_local_job(binding.clone(), recipe, Duration::from_secs(10))
            .unwrap();
        let original_fingerprint = prepared.intent_record().launch_identity().to_owned();
        assert_eq!(
            prepared.intent_record().runtime_identity(),
            Some(binding.runtime_identity())
        );

        let other_job = fixture.owned_job_binding(12, "caller:item-12");
        let other = fixture
            .controller
            .prepare_owned_local_job(other_job, fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        assert_ne!(
            original_fingerprint,
            other.intent_record().launch_identity()
        );

        let restored_with_changed_request = OwnedLocalJobBinding::from_persisted_parts(
            binding.runtime_identity().clone(),
            binding.job_id(),
            "caller:item-other".to_owned(),
            binding.job_directory().to_path_buf(),
            binding.directory_device(),
            binding.directory_inode(),
            binding.coordinator_identity(),
        )
        .unwrap();
        let changed_role = ServiceLaunchRole::OwnedLocalJob(restored_with_changed_request);
        assert_ne!(
            original_fingerprint,
            launch_fingerprint_for_role(
                &fixture.recipe(),
                &changed_role,
                Some(binding.runtime_identity())
            )
        );
    }

    #[test]
    fn owned_local_job_dispatch_rechecks_directory_and_recipe_fingerprint() {
        let fixture = Fixture::new();
        let binding = fixture.owned_job_binding(21, "caller:item-21");
        let job_directory = binding.job_directory().to_path_buf();
        let prepared = fixture
            .controller
            .prepare_owned_local_job(binding, fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let moved_directory = job_directory.with_extension("moved");
        fs::rename(&job_directory, &moved_directory).unwrap();
        fs::create_dir(&job_directory).unwrap();
        fs::set_permissions(&job_directory, fs::Permissions::from_mode(0o700)).unwrap();
        let failure = fixture
            .controller
            .dispatch(prepared, &fixture.context(Duration::from_secs(2)))
            .unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::NotDispatched);
        assert_eq!(failure.kind(), ManagedServiceErrorKind::OwnershipMismatch);
        assert!(!fixture.runner_arguments.exists());
        assert!(!fixture.state.exists());

        let binding = fixture.owned_job_binding(22, "caller:item-22");
        let mut tampered = fixture
            .controller
            .prepare_owned_local_job(binding.clone(), fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        tampered.intent.launch_role = ServiceLaunchRole::OwnedLocalJob(
            OwnedLocalJobBinding::from_persisted_parts(
                binding.runtime_identity().clone(),
                binding.job_id(),
                "caller:item-replaced".to_owned(),
                binding.job_directory().to_path_buf(),
                binding.directory_device(),
                binding.directory_inode(),
                binding.coordinator_identity(),
            )
            .unwrap(),
        );
        let failure = fixture
            .controller
            .dispatch(tampered, &fixture.context(Duration::from_secs(2)))
            .unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::NotDispatched);
        assert_eq!(failure.kind(), ManagedServiceErrorKind::OwnershipMismatch);
        assert!(!fixture.runner_arguments.exists());
        assert!(!fixture.state.exists());
    }

    #[test]
    fn owned_local_job_uses_ordinary_exec_topology_and_restored_role_is_stop_only() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_owned_local_job(
                fixture.owned_job_binding(31, "caller:item-31"),
                fixture.recipe(),
                Duration::from_secs(10),
            )
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let service = fixture
            .controller
            .dispatch(prepared, &fixture.context(Duration::from_secs(2)))
            .unwrap();
        let runner_arguments = fs::read_to_string(&fixture.runner_arguments).unwrap();
        assert!(runner_arguments
            .lines()
            .any(|line| line == "--service-type=exec"));
        assert!(runner_arguments
            .lines()
            .any(|line| line == "--property=KillMode=control-group"));
        assert!(runner_arguments
            .lines()
            .any(|line| line == "--property=Restart=no"));
        assert!(!runner_arguments
            .lines()
            .any(|line| line.starts_with("--property=PrivateNetwork=")));
        assert_eq!(service.intent_record(), &intent);

        let restored = fixture
            .controller
            .reconcile_restored_intent(&intent, &fixture.context(Duration::from_secs(2)))
            .unwrap();
        let stop_only = match restored {
            ReconcileResult::StopOnly(stop_only) => stop_only,
            other => panic!("restored job record must be stop-only, got {other:?}"),
        };
        assert_eq!(stop_only.intent_record(), &intent);
        let stopped = fixture
            .controller
            .stop_stop_only(&stop_only, &fixture.context(Duration::from_secs(2)))
            .unwrap();
        assert_eq!(stopped.intent_record(), &intent);
        assert_eq!(
            fs::read_to_string(fixture.directory.path().join("state")).unwrap(),
            "inactive"
        );
    }

    #[test]
    fn supervisor_reconciliation_does_not_treat_manager_flags_as_namespace_proof() {
        let fixture = Fixture::new();
        let coordinator =
            super::super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                .unwrap();
        let prepared = fixture
            .controller
            .prepare_isolated_engine_supervisor(
                RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
                fixture.recipe(),
                Duration::from_secs(10),
                coordinator,
            )
            .unwrap();
        let intent = prepared.intent_record().clone();
        let pid = std::process::id();
        let script = format!(
            "#!/bin/sh\nprintf 'LoadState=loaded\\nActiveState=active\\nMainPID={pid}\\nTransient=yes\\nInvocationID={invocation}\\nControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/{unit}\\nFragmentPath=/run/user/1000/systemd/transient/{unit}\\nDescription={description}\\nRestart=no\\nKillMode=control-group\\nType=notify\\nPrivateNetwork=yes\\nJoinsNamespaceOf=\\n'\n",
            pid = pid,
            invocation = INVOCATION,
            unit = intent.unit_name,
            description = description_for(&intent),
        );
        write_executable(&fixture.directory.path().join("systemctl"), &script);

        assert!(matches!(
            fixture
                .controller
                .reconcile(&intent, &fixture.context(Duration::from_secs(2))),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        ));
    }

    #[test]
    fn dispatch_refuses_a_recipe_that_differs_from_the_prepared_intent() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start_for_runtime(
                RuntimeIdentity::new(Lane::Media, "comfyui").unwrap(),
                fixture.recipe(),
                Duration::from_secs(10),
            )
            .unwrap();
        let alternate_directory = tempfile::tempdir().unwrap();
        let changed_recipe = LaunchRecipe::new(
            &fixture.recipe_executable,
            alternate_directory.path(),
            [OsString::from("fixture-arg")],
        )
        .unwrap();
        let forged = PreparedServiceStart {
            intent: prepared.intent,
            recipe: changed_recipe,
        };
        let failure = fixture
            .controller
            .dispatch(forged, &fixture.context(Duration::from_secs(2)))
            .unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::NotDispatched);
        assert_eq!(failure.kind(), ManagedServiceErrorKind::OwnershipMismatch);
        assert!(!fixture.state.exists());
    }

    #[test]
    fn working_directory_is_explicit_and_part_of_recipe_identity() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        let another_directory = tempfile::tempdir().unwrap();
        let other_recipe = LaunchRecipe::new(
            &fixture.recipe_executable,
            another_directory.path(),
            [OsString::from("fixture-arg")],
        )
        .unwrap();
        assert_ne!(
            intent.launch_identity(),
            other_recipe.launch_identity().as_str()
        );

        fixture.install_scripts(&intent, "0");
        let context = fixture.context(Duration::from_secs(2));
        let _handle = fixture.controller.dispatch(prepared, &context).unwrap();
        let runner_arguments = fs::read_to_string(&fixture.runner_arguments).unwrap();
        assert!(runner_arguments
            .lines()
            .any(|argument| argument == "--expand-environment=no"));
        let property = format!(
            "--property=WorkingDirectory={}",
            fixture.directory.path().canonicalize().unwrap().display()
        );
        assert!(runner_arguments
            .lines()
            .any(|argument| argument == property));
    }

    #[test]
    fn rejects_invalid_working_directories() {
        let fixture = Fixture::new();
        let args = [OsString::from("fixture-arg")];
        assert_eq!(
            LaunchRecipe::new(&fixture.recipe_executable, "relative", args.clone()).err(),
            Some(ManagedServiceErrorKind::InvalidRecipe)
        );
        assert_eq!(
            LaunchRecipe::new(
                &fixture.recipe_executable,
                fixture.directory.path().join("missing"),
                args.clone()
            )
            .err(),
            Some(ManagedServiceErrorKind::InvalidRecipe)
        );
        assert_eq!(
            LaunchRecipe::new(
                &fixture.recipe_executable,
                &fixture.recipe_executable,
                args.clone()
            )
            .err(),
            Some(ManagedServiceErrorKind::InvalidRecipe)
        );
        let nul_path = PathBuf::from(OsString::from_vec(b"/tmp/dir\0suffix".to_vec()));
        assert_eq!(
            LaunchRecipe::new(&fixture.recipe_executable, nul_path, args).err(),
            Some(ManagedServiceErrorKind::InvalidRecipe)
        );
    }

    #[test]
    fn dispatch_returns_handle_only_after_generation_and_cgroup_proof() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let context = fixture.context(Duration::from_secs(2));
        let handle = fixture.controller.dispatch(prepared, &context).unwrap();
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Running
        );
        let identity = handle.generation_identity();
        assert_eq!(identity, handle.generation_identity());
        assert_eq!(
            format!("{identity:?}"),
            "OwnedServiceGenerationIdentity([redacted])"
        );

        let mut different_coordinator = identity.clone();
        different_coordinator
            .intent
            .coordinator_id
            .push_str("-other");
        assert_ne!(identity, different_coordinator);
        let mut different_launch = identity.clone();
        different_launch.intent.launch_identity.push_str("-other");
        assert_ne!(identity, different_launch);
        let mut different_invocation = identity.clone();
        different_invocation.invocation_id.push_str("-other");
        assert_ne!(identity, different_invocation);
        let mut different_cgroup = identity.clone();
        different_cgroup.cgroup.push_str("-other");
        assert_ne!(identity, different_cgroup);
        let mut different_main_pid = identity.clone();
        different_main_pid.main_pid += 1;
        assert_ne!(identity, different_main_pid);

        let stopped = fixture.controller.stop(&handle, &context).unwrap();
        assert_eq!(stopped.cgroup_path(), handle.cgroup);
        assert!(!format!("{stopped:?}").contains(&handle.cgroup));
        assert!(!format!("{stopped:?}").contains(INVOCATION));
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Stopped
        );
    }

    #[test]
    fn comfy_generation_projection_requires_exact_runtime_and_supervisor_role() {
        let identity = OwnedServiceGenerationIdentity::for_test_comfy_supervisor().unwrap();
        let reference = identity.comfy_service_generation_reference().unwrap();
        assert!(reference.is_valid());
        assert_eq!(
            format!("{reference:?}"),
            "ComfyServiceGenerationReference([redacted])"
        );
        assert_eq!(
            reference.intent.coordinator_id,
            identity.intent.coordinator_id
        );
        assert_eq!(reference.intent.unit_name, identity.intent.unit_name);
        assert_eq!(
            reference.intent.launch_identity,
            identity.intent.launch_identity
        );
        assert_eq!(
            reference.intent.stop_timeout_millis,
            identity.intent.stop_timeout_millis
        );
        assert_eq!(reference.generation.invocation_id, identity.invocation_id);
        assert_eq!(reference.generation.cgroup, identity.cgroup);
        assert_eq!(reference.generation.main_pid, identity.main_pid);
        assert!(!valid_cgroup(
            &format!("/user.slice/prefix-{}", identity.intent.unit_name),
            &identity.intent.unit_name,
        ));

        let ServiceLaunchRole::IsolatedEngineSupervisor(coordinator) =
            identity.intent.launch_role()
        else {
            panic!("test identity must use the isolated supervisor role");
        };
        assert_eq!(
            reference.intent.supervisor_coordinator.pid,
            coordinator.pid()
        );
        assert_eq!(
            reference.intent.supervisor_coordinator.uid,
            coordinator.uid()
        );
        assert_eq!(
            reference.intent.supervisor_coordinator.start_time_ticks,
            coordinator.start_time_ticks()
        );
        assert_eq!(
            reference
                .intent
                .supervisor_coordinator
                .network_namespace_device,
            coordinator.network_namespace_device()
        );
        assert_eq!(
            reference
                .intent
                .supervisor_coordinator
                .network_namespace_inode,
            coordinator.network_namespace_inode()
        );

        let mut wrong_runtime = identity.clone();
        wrong_runtime.intent.runtime_identity =
            Some(RuntimeIdentity::new(Lane::Media, "comfyui").unwrap());
        assert_eq!(
            wrong_runtime.comfy_service_generation_reference().err(),
            Some(ManagedServiceErrorKind::InvalidRuntimeIdentity)
        );

        let mut ordinary = identity.clone();
        ordinary.intent.launch_role = ServiceLaunchRole::Ordinary;
        assert_eq!(
            ordinary.comfy_service_generation_reference().err(),
            Some(ManagedServiceErrorKind::InvalidLaunchRole)
        );

        let fixture = Fixture::new();
        let directory = fixture.directory.path().join("job-local");
        fs::create_dir(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        let binding = OwnedLocalJobBinding::capture(
            RuntimeIdentity::new(Lane::Media, "ComfyUI").unwrap(),
            7,
            "req-local-job",
            directory,
            super::super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                .unwrap(),
        )
        .unwrap();
        let mut local_job = identity;
        local_job.intent.launch_role = ServiceLaunchRole::OwnedLocalJob(binding);
        assert_eq!(
            local_job.comfy_service_generation_reference().err(),
            Some(ManagedServiceErrorKind::InvalidLaunchRole)
        );
    }

    #[test]
    fn inactive_snapshot_without_cgroup_does_not_claim_stop_cleanup() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts_with_cgroup(&intent, "0", true);
        let context = fixture.context(Duration::from_secs(2));
        let handle = fixture.controller.dispatch(prepared, &context).unwrap();
        assert_eq!(
            fixture.controller.stop(&handle, &context),
            Err(ManagedServiceErrorKind::StopUnconfirmed)
        );
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Unknown
        );
    }

    #[test]
    fn non_transient_unit_never_yields_an_owned_handle() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let systemctl = fixture.directory.path().join("systemctl");
        let changed = fs::read_to_string(&systemctl)
            .unwrap()
            .replace("Transient=yes", "Transient=no");
        write_executable(&systemctl, &changed);
        let context = fixture.context(Duration::from_secs(2));
        let failure = fixture.controller.dispatch(prepared, &context).unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::OwnershipUncertain);
        assert_eq!(failure.kind(), ManagedServiceErrorKind::OwnershipMismatch);
        assert_eq!(failure.intent_record(), &intent);
    }

    #[test]
    fn failed_launcher_keeps_uncertain_intent() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "1");
        let context = fixture.context(Duration::from_secs(2));
        let failure = fixture.controller.dispatch(prepared, &context).unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::OwnershipUncertain);
        assert_eq!(failure.intent_record(), &intent);
        assert_eq!(
            failure.kind(),
            ManagedServiceErrorKind::ManagerCommandFailed
        );
    }

    #[test]
    fn mismatched_generation_never_reaches_stop() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let context = fixture.context(Duration::from_secs(2));
        let handle = fixture.controller.dispatch(prepared, &context).unwrap();
        let systemctl = fixture.directory.path().join("systemctl");
        let changed = fs::read_to_string(&systemctl)
            .unwrap()
            .replace(INVOCATION, "ffffffffffffffffffffffffffffffff");
        write_executable(&systemctl, &changed);
        assert_eq!(
            fixture.controller.stop(&handle, &context),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        );
        assert_eq!(fs::read_to_string(&fixture.state).unwrap(), "active");
    }

    #[test]
    fn refuses_preexisting_unit_file_before_dispatch() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        let systemctl = fixture.directory.path().join("systemctl");
        let script = format!(
            "#!/bin/sh\nfor arg in \"$@\"; do\n  if [ \"$arg\" = list-unit-files ]; then printf '{} disabled\\n'; fi\ndone\nexit 0\n",
            intent.unit_name
        );
        write_executable(&systemctl, &script);
        let context = fixture.context(Duration::from_secs(2));
        let failure = fixture.controller.dispatch(prepared, &context).unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::NotDispatched);
        assert_eq!(failure.kind(), ManagedServiceErrorKind::UnitAlreadyExists);
        assert!(!fixture.state.exists());
    }

    #[test]
    fn restored_intent_reconciles_to_handle_only_with_live_generation_proof() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        fs::write(&fixture.state, "active").unwrap();
        let restored = ServiceIntentRecord::from_persisted_parts(
            intent.coordinator_id().to_owned(),
            intent.unit_name().to_owned(),
            intent.launch_identity().to_owned(),
            intent.stop_timeout_millis(),
        )
        .unwrap();
        let context = fixture.context(Duration::from_secs(2));
        let ReconcileResult::Owned(handle) =
            fixture.controller.reconcile(&restored, &context).unwrap()
        else {
            panic!("fixture should prove the matching active generation");
        };
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Running
        );
    }

    #[test]
    fn restored_runtime_binding_must_match_the_manager_generation() {
        let fixture = Fixture::new();
        let runtime = RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap();
        let prepared = fixture
            .controller
            .prepare_start_for_runtime(runtime.clone(), fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let context = fixture.context(Duration::from_secs(2));
        let _handle = fixture.controller.dispatch(prepared, &context).unwrap();

        let restored = ServiceIntentRecord::from_persisted_parts_with_runtime(
            intent.coordinator_id().to_owned(),
            intent.unit_name().to_owned(),
            intent.launch_identity().to_owned(),
            intent.stop_timeout_millis(),
            Some(runtime.clone()),
        )
        .unwrap();
        assert_eq!(restored.runtime_identity(), Some(&runtime));
        assert!(matches!(
            fixture.controller.reconcile(&restored, &context).unwrap(),
            ReconcileResult::Owned(_)
        ));

        let changed_binding = ServiceIntentRecord::from_persisted_parts_with_runtime(
            intent.coordinator_id().to_owned(),
            intent.unit_name().to_owned(),
            intent.launch_identity().to_owned(),
            intent.stop_timeout_millis(),
            Some(RuntimeIdentity::new(Lane::Media, "comfyui").unwrap()),
        )
        .unwrap();
        assert!(matches!(
            fixture
                .controller
                .reconcile(&changed_binding, &context)
                .unwrap(),
            ReconcileResult::Unknown
        ));
    }

    #[test]
    fn rejects_invalid_names_and_stop_timeouts() {
        assert_eq!(
            ManagedServiceController::with_test_executables(
                "../bad",
                PathBuf::from("/bin/true"),
                PathBuf::from("/bin/true")
            )
            .err(),
            Some(ManagedServiceErrorKind::InvalidCoordinatorId)
        );
        let fixture = Fixture::new();
        assert_eq!(
            fixture
                .controller
                .prepare_start(fixture.recipe(), Duration::ZERO)
                .err(),
            Some(ManagedServiceErrorKind::InvalidStopTimeout)
        );
    }

    #[test]
    fn persisted_intent_restore_is_bounded_and_not_a_handle() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record();
        assert_eq!(
            ServiceIntentRecord::from_persisted_parts(
                intent.coordinator_id().to_owned(),
                intent.unit_name().to_owned(),
                intent.launch_identity().to_owned(),
                intent.stop_timeout_millis(),
            )
            .unwrap(),
            *intent
        );
        assert_eq!(
            ServiceIntentRecord::from_persisted_parts(
                intent.coordinator_id().to_owned(),
                "llama-swap.service".to_owned(),
                intent.launch_identity().to_owned(),
                intent.stop_timeout_millis(),
            )
            .err(),
            Some(ManagedServiceErrorKind::OwnershipMismatch)
        );
        assert_eq!(
            ServiceIntentRecord::from_persisted_parts(
                intent.coordinator_id().to_owned(),
                intent.unit_name().to_owned(),
                "x".repeat(100_000),
                intent.stop_timeout_millis(),
            )
            .err(),
            Some(ManagedServiceErrorKind::OwnershipMismatch)
        );
    }
}
