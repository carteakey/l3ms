//! Durable pending start-intent records for the managed-service coordinator.
//!
//! The caller must hold the host singleton lock for every store operation.
//! The existing absolute directory must be a dedicated, trusted directory on
//! a stable local filesystem. This module neither creates directories nor
//! repairs records. Its only removal path is crate-private and requires the
//! owner to present a generation-bound stopped-service proof. A retained
//! record is a write-ahead fact only; its absence does not establish that a
//! service stopped.

use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};

use serde::{Deserialize, Deserializer, Serialize};

use super::managed_service::{
    runtime_identity_from_persisted_parts, ServiceIntentRecord, ServiceLaunchRole,
};
use super::runtime::OperationContext;

const MAX_INTENTS: usize = 128;
const MAX_RECORD_BYTES: usize = 4 * 1024;
const DIRECTORY_MODE: u32 = 0o700;
const FILE_MODE: u32 = 0o600;
const JSON_SUFFIX: &str = ".json";
const UNIT_SUFFIX: &str = ".service";

/// Fixed, redacted failures from the retained intent store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceIntentStoreError {
    UnsafeDirectory,
    InvalidRecord,
    DuplicateIntent,
    CapacityExceeded,
    CorruptStore,
    IntentMismatch,
    Cancelled,
    DeadlineExceeded,
    Io,
}

impl fmt::Display for ServiceIntentStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsafeDirectory => "service intent store directory is unsafe",
            Self::InvalidRecord => "service intent record is invalid",
            Self::DuplicateIntent => "service intent record already exists",
            Self::CapacityExceeded => "service intent store capacity exceeded",
            Self::CorruptStore => "service intent store contents are invalid",
            Self::IntentMismatch => "retained service intent does not match the receipt",
            Self::Cancelled => "service intent retirement was cancelled",
            Self::DeadlineExceeded => "service intent retirement deadline expired",
            Self::Io => "service intent store operation failed",
        })
    }
}

impl std::error::Error for ServiceIntentStoreError {}

/// Linux write-ahead store for pending service start intents.
///
/// There is no public deletion operation. The crate-private retirement path
/// accepts a complete validated record only and is called by the owner after
/// generation, manager, and cgroup checks. The store must not be shared across
/// coordinators without the caller's host singleton lock.
pub struct ServiceIntentStore {
    directory: File,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServiceIntentRetireDisposition {
    NotRetired,
    DurabilityUncertain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ServiceIntentRetireFailure {
    kind: ServiceIntentStoreError,
    disposition: ServiceIntentRetireDisposition,
}

impl ServiceIntentRetireFailure {
    pub(crate) fn kind(self) -> ServiceIntentStoreError {
        self.kind
    }

    pub(crate) fn disposition(self) -> ServiceIntentRetireDisposition {
        self.disposition
    }
}

impl fmt::Debug for ServiceIntentStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ServiceIntentStore(<redacted>)")
    }
}

impl ServiceIntentStore {
    /// Opens an existing absolute directory only when it is owned by the
    /// current effective UID and has exactly mode 0700. No path component may
    /// be a symlink. The directory must be dedicated to these records.
    pub fn open(path: &Path) -> Result<Self, ServiceIntentStoreError> {
        let directory = File::from(open_directory_without_symlinks(path)?);
        let stat = stat_fd(directory.as_raw_fd()).map_err(|_| ServiceIntentStoreError::Io)?;
        if !is_directory(&stat)
            || stat.st_uid != unsafe { libc::geteuid() }
            || stat.st_mode & 0o7777 != DIRECTORY_MODE
        {
            return Err(ServiceIntentStoreError::UnsafeDirectory);
        }
        Ok(Self { directory })
    }

    /// Persists an intent before dispatch. Success is returned only after the
    /// new file and its parent directory have both been synchronized. A
    /// failed write is deliberately left in place for caller quarantine.
    pub fn record(&self, intent: &ServiceIntentRecord) -> Result<(), ServiceIntentStoreError> {
        let persisted = PersistedIntent::from_record(intent)?;
        let existing = self.list()?;
        if existing
            .iter()
            .any(|record| record.unit_name() == intent.unit_name())
        {
            return Err(ServiceIntentStoreError::DuplicateIntent);
        }
        if existing.len() >= MAX_INTENTS {
            return Err(ServiceIntentStoreError::CapacityExceeded);
        }

        let bytes = serde_json::to_vec(&persisted).map_err(|_| ServiceIntentStoreError::Io)?;
        if bytes.len() > MAX_RECORD_BYTES {
            return Err(ServiceIntentStoreError::InvalidRecord);
        }
        let filename = format!("{}{}", intent.unit_name(), JSON_SUFFIX);
        let name = CString::new(filename).map_err(|_| ServiceIntentStoreError::InvalidRecord)?;
        let fd = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                FILE_MODE as libc::mode_t,
            )
        };
        if fd < 0 {
            return if io::Error::last_os_error().kind() == io::ErrorKind::AlreadyExists {
                Err(ServiceIntentStoreError::DuplicateIntent)
            } else {
                Err(ServiceIntentStoreError::Io)
            };
        }

        // The descriptor is now the only path to this newly created file.
        // Any later failure leaves it in place; no cleanup or retry is safe.
        let mut file = unsafe { File::from_raw_fd(fd) };
        let stat = file.metadata().map_err(|_| ServiceIntentStoreError::Io)?;
        if !valid_file_metadata(&stat, unsafe { libc::geteuid() }) {
            return Err(ServiceIntentStoreError::CorruptStore);
        }
        file.write_all(&bytes)
            .map_err(|_| ServiceIntentStoreError::Io)?;
        file.sync_all().map_err(|_| ServiceIntentStoreError::Io)?;
        drop(file);
        self.directory
            .sync_all()
            .map_err(|_| ServiceIntentStoreError::Io)
    }

    /// Reads and validates every retained intent, failing closed on any
    /// unexpected entry, malformed file, unsupported schema, or unsafe inode.
    pub fn list(&self) -> Result<Vec<ServiceIntentRecord>, ServiceIntentStoreError> {
        let names = self.entry_names()?;
        if names.len() > MAX_INTENTS {
            return Err(ServiceIntentStoreError::CapacityExceeded);
        }

        let mut records = Vec::with_capacity(names.len());
        for filename in names {
            if !valid_filename(&filename) {
                return Err(ServiceIntentStoreError::CorruptStore);
            }
            records.push(self.read_record(&filename)?);
        }
        records.sort_by(|left, right| left.unit_name().cmp(right.unit_name()));
        Ok(records)
    }

    /// Removes only the exact, fully validated retained record associated
    /// with a stopped-service receipt. This is crate-private so callers cannot
    /// delete intents by unit name or bypass the owner's fresh cgroup and
    /// manager observations. Once unlink succeeds, parent-directory sync is
    /// mandatory; a sync failure is reported as durability-uncertain and must
    /// not be retried automatically.
    pub(crate) fn retire_record(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<(), ServiceIntentRetireFailure> {
        self.retire_record_with_sync(intent, context, |directory| directory.sync_all())
    }

    fn retire_record_with_sync(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
        sync_directory: impl FnOnce(&File) -> io::Result<()>,
    ) -> Result<(), ServiceIntentRetireFailure> {
        let not_retired = |kind| ServiceIntentRetireFailure {
            kind,
            disposition: ServiceIntentRetireDisposition::NotRetired,
        };
        let uncertain = |kind| ServiceIntentRetireFailure {
            kind,
            disposition: ServiceIntentRetireDisposition::DurabilityUncertain,
        };

        check_retirement_context(context).map_err(not_retired)?;
        PersistedIntent::from_record(intent).map_err(not_retired)?;
        let records = self.list().map_err(not_retired)?;
        check_retirement_context(context).map_err(not_retired)?;
        if !records.iter().any(|record| record == intent) {
            return Err(not_retired(ServiceIntentStoreError::IntentMismatch));
        }

        let filename = format!("{}{}", intent.unit_name(), JSON_SUFFIX);
        let name = CString::new(filename.as_str())
            .map_err(|_| not_retired(ServiceIntentStoreError::IntentMismatch))?;
        check_retirement_context(context).map_err(not_retired)?;
        let entry_stat = stat_at_nofollow(self.directory.as_raw_fd(), &name)
            .map_err(|_| not_retired(ServiceIntentStoreError::CorruptStore))?;
        check_retirement_context(context).map_err(not_retired)?;
        if !valid_stat_file(&entry_stat, unsafe { libc::geteuid() })
            || entry_stat.st_size < 0
            || entry_stat.st_size as usize > MAX_RECORD_BYTES
        {
            return Err(not_retired(ServiceIntentStoreError::CorruptStore));
        }

        check_retirement_context(context).map_err(not_retired)?;
        let fd = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(not_retired(ServiceIntentStoreError::CorruptStore));
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        check_retirement_context(context).map_err(not_retired)?;
        let opened =
            stat_fd(file.as_raw_fd()).map_err(|_| not_retired(ServiceIntentStoreError::Io))?;
        check_retirement_context(context).map_err(not_retired)?;
        if !valid_stat_file(&opened, unsafe { libc::geteuid() })
            || opened.st_size != entry_stat.st_size
            || !same_inode(&opened, &entry_stat)
        {
            return Err(not_retired(ServiceIntentStoreError::CorruptStore));
        }

        let mut bytes = Vec::with_capacity(opened.st_size as usize);
        {
            let mut bounded = Read::by_ref(&mut file).take((MAX_RECORD_BYTES + 1) as u64);
            bounded
                .read_to_end(&mut bytes)
                .map_err(|_| not_retired(ServiceIntentStoreError::Io))?;
        }
        check_retirement_context(context).map_err(not_retired)?;
        if bytes.len() > MAX_RECORD_BYTES || bytes.len() as i64 != opened.st_size {
            return Err(not_retired(ServiceIntentStoreError::CorruptStore));
        }
        let after_read =
            stat_fd(file.as_raw_fd()).map_err(|_| not_retired(ServiceIntentStoreError::Io))?;
        check_retirement_context(context).map_err(not_retired)?;
        if !valid_stat_file(&after_read, unsafe { libc::geteuid() })
            || !same_inode(&opened, &after_read)
            || after_read.st_size != opened.st_size
        {
            return Err(not_retired(ServiceIntentStoreError::CorruptStore));
        }
        let persisted: PersistedIntent = serde_json::from_slice(&bytes)
            .map_err(|_| not_retired(ServiceIntentStoreError::CorruptStore))?;
        let stored = persisted
            .into_record()
            .map_err(|_| not_retired(ServiceIntentStoreError::CorruptStore))?;
        if stored != *intent || format!("{}{}", stored.unit_name(), JSON_SUFFIX) != filename {
            return Err(not_retired(ServiceIntentStoreError::IntentMismatch));
        }

        check_retirement_context(context).map_err(not_retired)?;
        let directory_stat = stat_fd(self.directory.as_raw_fd())
            .map_err(|_| not_retired(ServiceIntentStoreError::Io))?;
        check_retirement_context(context).map_err(not_retired)?;
        if !is_directory(&directory_stat)
            || directory_stat.st_uid != unsafe { libc::geteuid() }
            || directory_stat.st_mode & 0o7777 != DIRECTORY_MODE
        {
            return Err(not_retired(ServiceIntentStoreError::UnsafeDirectory));
        }
        let current_name = stat_at_nofollow(self.directory.as_raw_fd(), &name)
            .map_err(|_| not_retired(ServiceIntentStoreError::CorruptStore))?;
        check_retirement_context(context).map_err(not_retired)?;
        if !valid_stat_file(&current_name, unsafe { libc::geteuid() })
            || !same_inode(&opened, &current_name)
            || current_name.st_size != opened.st_size
        {
            return Err(not_retired(ServiceIntentStoreError::CorruptStore));
        }
        check_retirement_context(context).map_err(not_retired)?;
        if unsafe { libc::unlinkat(self.directory.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            return Err(not_retired(ServiceIntentStoreError::Io));
        }

        // Cancellation after unlink cannot roll back the commit. Still sync
        // the directory to resolve durability; success is final even if the
        // caller's context expires while these non-preemptible operations run.
        let _context_after_unlink = check_retirement_context(context);
        if sync_directory(&self.directory).is_err() {
            return Err(uncertain(ServiceIntentStoreError::Io));
        }
        let _context_after_sync = check_retirement_context(context);
        Ok(())
    }

    fn entry_names(&self) -> Result<Vec<String>, ServiceIntentStoreError> {
        // openat(".") gives the enumeration an independent directory offset,
        // so repeated list() calls each see the complete directory.
        let dot = CString::new(".").expect("static path has no NUL");
        let fd = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                dot.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(ServiceIntentStoreError::Io);
        }
        let stream = unsafe { libc::fdopendir(fd) };
        if stream.is_null() {
            unsafe { libc::close(fd) };
            return Err(ServiceIntentStoreError::Io);
        }
        let stream = DirectoryStream(stream);
        let mut names = Vec::new();
        loop {
            unsafe { *libc::__errno_location() = 0 };
            let entry = unsafe { libc::readdir(stream.0) };
            if entry.is_null() {
                let errno = unsafe { *libc::__errno_location() };
                if errno != 0 {
                    return Err(ServiceIntentStoreError::Io);
                }
                break;
            }
            let raw_name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
            if raw_name == b"." || raw_name == b".." {
                continue;
            }
            if names.len() == MAX_INTENTS {
                return Err(ServiceIntentStoreError::CapacityExceeded);
            }
            let filename =
                std::str::from_utf8(raw_name).map_err(|_| ServiceIntentStoreError::CorruptStore)?;
            names.push(filename.to_owned());
        }
        names.sort_unstable();
        Ok(names)
    }

    fn read_record(&self, filename: &str) -> Result<ServiceIntentRecord, ServiceIntentStoreError> {
        let name = CString::new(filename).map_err(|_| ServiceIntentStoreError::CorruptStore)?;
        // Reject special files before opening them. O_NONBLOCK below also
        // prevents a replacement FIFO from hanging the reader between this
        // check and openat; the descriptor's fstat remains authoritative.
        let entry_stat = stat_at_nofollow(self.directory.as_raw_fd(), &name)
            .map_err(|_| ServiceIntentStoreError::CorruptStore)?;
        if !valid_stat_file(&entry_stat, unsafe { libc::geteuid() })
            || entry_stat.st_size < 0
            || entry_stat.st_size as usize > MAX_RECORD_BYTES
        {
            return Err(ServiceIntentStoreError::CorruptStore);
        }
        let fd = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(ServiceIntentStoreError::CorruptStore);
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let before = stat_fd(file.as_raw_fd()).map_err(|_| ServiceIntentStoreError::Io)?;
        if !valid_stat_file(&before, unsafe { libc::geteuid() })
            || before.st_size < 0
            || before.st_size as usize > MAX_RECORD_BYTES
        {
            return Err(ServiceIntentStoreError::CorruptStore);
        }

        let mut bytes = Vec::with_capacity(before.st_size as usize);
        {
            let mut bounded = Read::by_ref(&mut file).take((MAX_RECORD_BYTES + 1) as u64);
            bounded
                .read_to_end(&mut bytes)
                .map_err(|_| ServiceIntentStoreError::Io)?;
        }
        if bytes.len() > MAX_RECORD_BYTES || bytes.len() as i64 != before.st_size {
            return Err(ServiceIntentStoreError::CorruptStore);
        }
        let after = stat_fd(file.as_raw_fd()).map_err(|_| ServiceIntentStoreError::Io)?;
        if !valid_stat_file(&after, unsafe { libc::geteuid() }) || after.st_size != before.st_size {
            return Err(ServiceIntentStoreError::CorruptStore);
        }

        let persisted: PersistedIntent =
            serde_json::from_slice(&bytes).map_err(|_| ServiceIntentStoreError::CorruptStore)?;
        let record = persisted
            .into_record()
            .map_err(|_| ServiceIntentStoreError::CorruptStore)?;
        let expected_filename = format!("{}{}", record.unit_name(), JSON_SUFFIX);
        if expected_filename != filename {
            return Err(ServiceIntentStoreError::CorruptStore);
        }
        Ok(record)
    }
}

fn check_retirement_context(context: &OperationContext<'_>) -> Result<(), ServiceIntentStoreError> {
    if context.is_cancelled() {
        Err(ServiceIntentStoreError::Cancelled)
    } else if context.remaining().is_zero() {
        Err(ServiceIntentStoreError::DeadlineExceeded)
    } else {
        Ok(())
    }
}

fn same_inode(left: &libc::stat, right: &libc::stat) -> bool {
    left.st_dev == right.st_dev && left.st_ino == right.st_ino
}

struct DirectoryStream(*mut libc::DIR);

impl Drop for DirectoryStream {
    fn drop(&mut self) {
        unsafe { libc::closedir(self.0) };
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedIntent {
    schema: u8,
    coordinator_id: String,
    unit_name: String,
    launch_identity: String,
    stop_timeout_millis: u64,
    #[serde(
        default,
        deserialize_with = "deserialize_binding_field",
        skip_serializing_if = "Option::is_none"
    )]
    runtime_lane: Option<Option<String>>,
    #[serde(
        default,
        deserialize_with = "deserialize_binding_field",
        skip_serializing_if = "Option::is_none"
    )]
    runtime_name: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    launch_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    coordinator_identity: Option<PersistedCoordinatorIdentity>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedCoordinatorIdentity {
    pid: u32,
    uid: u32,
    start_time_ticks: u64,
    network_namespace_device: u64,
    network_namespace_inode: u64,
}

// Distinguish an absent schema-1 field from an explicitly supplied null.
// Ordinary nested Option deserialization collapses both to None.
fn deserialize_binding_field<'de, D>(deserializer: D) -> Result<Option<Option<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(Some)
}

impl PersistedIntent {
    fn from_record(record: &ServiceIntentRecord) -> Result<Self, ServiceIntentStoreError> {
        ServiceIntentRecord::from_persisted_parts_with_role(
            record.coordinator_id().to_owned(),
            record.unit_name().to_owned(),
            record.launch_identity().to_owned(),
            record.stop_timeout_millis(),
            record.runtime_identity().cloned(),
            *record.launch_role(),
        )
        .map_err(|_| ServiceIntentStoreError::InvalidRecord)?;
        let runtime_lane = record
            .runtime_identity()
            .map(|identity| Some(identity.lane.as_str().to_owned()));
        let runtime_name = record
            .runtime_identity()
            .map(|identity| Some(identity.as_str().to_owned()));
        let (launch_role, coordinator_identity) = match record.launch_role() {
            ServiceLaunchRole::Ordinary => (None, None),
            ServiceLaunchRole::IsolatedEngineSupervisor(coordinator) => (
                Some("isolated_engine_supervisor".to_owned()),
                Some(PersistedCoordinatorIdentity {
                    pid: coordinator.pid(),
                    uid: coordinator.uid(),
                    start_time_ticks: coordinator.start_time_ticks(),
                    network_namespace_device: coordinator.network_namespace_device(),
                    network_namespace_inode: coordinator.network_namespace_inode(),
                }),
            ),
        };
        Ok(Self {
            schema: if launch_role.is_some() {
                3
            } else if runtime_lane.is_some() {
                2
            } else {
                1
            },
            coordinator_id: record.coordinator_id().to_owned(),
            unit_name: record.unit_name().to_owned(),
            launch_identity: record.launch_identity().to_owned(),
            stop_timeout_millis: record.stop_timeout_millis(),
            runtime_lane,
            runtime_name,
            launch_role,
            coordinator_identity,
        })
    }

    fn into_record(self) -> Result<ServiceIntentRecord, ServiceIntentStoreError> {
        let Self {
            schema,
            coordinator_id,
            unit_name,
            launch_identity,
            stop_timeout_millis,
            runtime_lane,
            runtime_name,
            launch_role,
            coordinator_identity,
        } = self;
        let runtime_identity = match (schema, runtime_lane, runtime_name) {
            (1, None, None) => None,
            (2 | 3, Some(Some(lane)), Some(Some(name))) => Some(
                runtime_identity_from_persisted_parts(&lane, name)
                    .map_err(|_| ServiceIntentStoreError::CorruptStore)?,
            ),
            _ => return Err(ServiceIntentStoreError::CorruptStore),
        };
        let launch_role = match (schema, launch_role.as_deref(), coordinator_identity) {
            (1 | 2, None, None) => ServiceLaunchRole::Ordinary,
            (3, Some("isolated_engine_supervisor"), Some(coordinator)) => {
                if runtime_identity.is_none() {
                    return Err(ServiceIntentStoreError::CorruptStore);
                }
                let coordinator =
                    super::engine_transport::CoordinatorIdentityDescriptor::from_parts(
                        coordinator.pid,
                        coordinator.uid,
                        coordinator.start_time_ticks,
                        coordinator.network_namespace_device,
                        coordinator.network_namespace_inode,
                    )
                    .map_err(|_| ServiceIntentStoreError::CorruptStore)?;
                ServiceLaunchRole::IsolatedEngineSupervisor(coordinator)
            }
            _ => return Err(ServiceIntentStoreError::CorruptStore),
        };
        ServiceIntentRecord::from_persisted_parts_with_role(
            coordinator_id,
            unit_name,
            launch_identity,
            stop_timeout_millis,
            runtime_identity,
            launch_role,
        )
        .map_err(|_| ServiceIntentStoreError::CorruptStore)
    }
}

fn valid_filename(filename: &str) -> bool {
    if filename.len() > 96 + UNIT_SUFFIX.len() + JSON_SUFFIX.len() || !filename.is_ascii() {
        return false;
    }
    let Some(unit_name) = filename.strip_suffix(JSON_SUFFIX) else {
        return false;
    };
    let Some(unit_stem) = unit_name.strip_suffix(UNIT_SUFFIX) else {
        return false;
    };
    !unit_stem.is_empty()
        && unit_name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'.'
        })
}

fn open_directory_without_symlinks(path: &Path) -> Result<OwnedFd, ServiceIntentStoreError> {
    if !path.is_absolute() {
        return Err(ServiceIntentStoreError::UnsafeDirectory);
    }
    let root = CString::new("/").expect("static path has no NUL");
    let root_fd = unsafe {
        libc::open(
            root.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if root_fd < 0 {
        return Err(ServiceIntentStoreError::Io);
    }
    let mut current = unsafe { OwnedFd::from_raw_fd(root_fd) };
    for component in path.components() {
        let Component::Normal(part) = component else {
            if matches!(component, Component::RootDir) {
                continue;
            }
            return Err(ServiceIntentStoreError::UnsafeDirectory);
        };
        let name =
            CString::new(part.as_bytes()).map_err(|_| ServiceIntentStoreError::UnsafeDirectory)?;
        let fd = unsafe {
            libc::openat(
                current.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(ServiceIntentStoreError::UnsafeDirectory);
        }
        current = unsafe { OwnedFd::from_raw_fd(fd) };
    }
    Ok(current)
}

fn stat_fd(fd: RawFd) -> io::Result<libc::stat> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { stat.assume_init() })
}

fn stat_at_nofollow(directory_fd: RawFd, name: &CStr) -> io::Result<libc::stat> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    if unsafe {
        libc::fstatat(
            directory_fd,
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { stat.assume_init() })
}

fn is_directory(stat: &libc::stat) -> bool {
    stat.st_mode & libc::S_IFMT == libc::S_IFDIR
}

fn is_regular_file(stat: &libc::stat) -> bool {
    stat.st_mode & libc::S_IFMT == libc::S_IFREG
}

fn valid_stat_file(stat: &libc::stat, uid: libc::uid_t) -> bool {
    is_regular_file(stat)
        && stat.st_uid == uid
        && stat.st_mode & 0o7777 == FILE_MODE
        && stat.st_nlink == 1
}

#[cfg(unix)]
fn valid_file_metadata(metadata: &std::fs::Metadata, uid: libc::uid_t) -> bool {
    use std::os::unix::fs::MetadataExt;

    metadata.is_file()
        && metadata.uid() == uid
        && metadata.mode() & 0o7777 == FILE_MODE
        && metadata.nlink() == 1
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{Lane, RuntimeIdentity};
    use super::super::runtime::NeverCancel;
    use super::*;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{symlink, PermissionsExt};
    use std::time::{Duration, Instant};

    static NEVER_CANCEL: NeverCancel = NeverCancel;

    fn private_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(
            directory.path(),
            std::fs::Permissions::from_mode(DIRECTORY_MODE),
        )
        .unwrap();
        directory
    }

    fn intent(suffix: u128) -> ServiceIntentRecord {
        let random = format!("{suffix:032x}");
        ServiceIntentRecord::from_persisted_parts(
            "coordinator".to_owned(),
            format!("l3ms-coordinator-{random}.service"),
            "b".repeat(64),
            30_000,
        )
        .unwrap()
    }

    fn runtime_intent(suffix: u128) -> ServiceIntentRecord {
        let random = format!("{suffix:032x}");
        ServiceIntentRecord::from_persisted_parts_with_runtime(
            "coordinator".to_owned(),
            format!("l3ms-coordinator-{random}.service"),
            "c".repeat(64),
            45_000,
            Some(RuntimeIdentity::new(Lane::Media, "comfyui").unwrap()),
        )
        .unwrap()
    }

    fn supervisor_intent(suffix: u128) -> ServiceIntentRecord {
        let random = format!("{suffix:032x}");
        ServiceIntentRecord::from_persisted_parts_with_role(
            "coordinator".to_owned(),
            format!("l3ms-coordinator-{random}.service"),
            "d".repeat(64),
            45_000,
            Some(RuntimeIdentity::new(Lane::Media, "comfyui").unwrap()),
            ServiceLaunchRole::IsolatedEngineSupervisor(
                super::super::engine_transport::CoordinatorIdentityDescriptor::capture_current()
                    .unwrap(),
            ),
        )
        .unwrap()
    }

    fn filename(record: &ServiceIntentRecord) -> String {
        format!("{}{}", record.unit_name(), JSON_SUFFIX)
    }

    fn write_private_file(path: &Path, bytes: &[u8], mode: u32) {
        std::fs::write(path, bytes).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    fn raw_record(record: &ServiceIntentRecord) -> Vec<u8> {
        serde_json::to_vec(&PersistedIntent::from_record(record).unwrap()).unwrap()
    }

    fn context() -> OperationContext<'static> {
        OperationContext::new(Instant::now() + Duration::from_secs(3), &NEVER_CANCEL)
    }

    #[test]
    fn record_is_durable_and_lists_a_restorable_record() {
        let directory = private_directory();
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        let original = intent(7);

        store.record(&original).unwrap();

        let persisted: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join(filename(&original))).unwrap(),
        )
        .unwrap();
        assert_eq!(persisted["schema"], 1);
        assert!(persisted.get("runtime_lane").is_none());
        assert!(persisted.get("runtime_name").is_none());

        assert_eq!(store.list().unwrap(), vec![original.clone()]);
        let restored = store.list().unwrap().pop().unwrap();
        assert_eq!(
            ServiceIntentRecord::from_persisted_parts(
                restored.coordinator_id().to_owned(),
                restored.unit_name().to_owned(),
                restored.launch_identity().to_owned(),
                restored.stop_timeout_millis(),
            )
            .unwrap(),
            original
        );
    }

    #[test]
    fn bound_record_round_trips_only_bounded_runtime_metadata() {
        let directory = private_directory();
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        let original = runtime_intent(15);
        store.record(&original).unwrap();

        let persisted: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join(filename(&original))).unwrap(),
        )
        .unwrap();
        assert_eq!(persisted["schema"], 2);
        assert_eq!(persisted["runtime_lane"], "media");
        assert_eq!(persisted["runtime_name"], "comfyui");
        assert_eq!(persisted.as_object().unwrap().len(), 7);
        assert!(persisted.get("argv").is_none());
        assert!(persisted.get("env").is_none());

        let restored = store.list().unwrap().pop().unwrap();
        assert_eq!(restored, original);
        assert_eq!(
            restored.runtime_identity(),
            Some(&RuntimeIdentity::new(Lane::Media, "comfyui").unwrap())
        );
    }

    #[test]
    fn supervisor_record_round_trips_exact_typed_launch_metadata() {
        let directory = private_directory();
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        let original = supervisor_intent(17);
        store.record(&original).unwrap();

        let persisted: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join(filename(&original))).unwrap(),
        )
        .unwrap();
        assert_eq!(persisted["schema"], 3);
        assert_eq!(persisted["launch_role"], "isolated_engine_supervisor");
        assert_eq!(persisted["runtime_lane"], "media");
        assert_eq!(persisted["runtime_name"], "comfyui");
        assert_eq!(persisted.as_object().unwrap().len(), 9);
        let identity = persisted["coordinator_identity"].as_object().unwrap();
        assert_eq!(identity.len(), 5);
        assert_eq!(identity["pid"], std::process::id());
        assert!(persisted.get("argv").is_none());
        assert!(persisted.get("environment").is_none());

        let restored = store.list().unwrap().pop().unwrap();
        assert_eq!(restored, original);
        assert_eq!(
            restored.runtime_identity(),
            Some(&RuntimeIdentity::new(Lane::Media, "comfyui").unwrap())
        );
    }

    #[test]
    fn duplicate_is_rejected_without_overwriting_the_retained_record() {
        let directory = private_directory();
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        let original = intent(8);
        store.record(&original).unwrap();
        let path = directory.path().join(filename(&original));
        let before = std::fs::read(&path).unwrap();

        assert_eq!(
            store.record(&original),
            Err(ServiceIntentStoreError::DuplicateIntent)
        );
        assert_eq!(std::fs::read(path).unwrap(), before);
    }

    #[test]
    fn private_retirement_requires_exact_bound_record_and_syncs_removal() {
        let directory = private_directory();
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        let original = runtime_intent(18);
        store.record(&original).unwrap();
        let path = directory.path().join(filename(&original));
        let before = std::fs::read(&path).unwrap();
        let mismatched = ServiceIntentRecord::from_persisted_parts_with_runtime(
            original.coordinator_id().to_owned(),
            original.unit_name().to_owned(),
            original.launch_identity().to_owned(),
            original.stop_timeout_millis(),
            Some(RuntimeIdentity::new(Lane::Language, "llama-swap").unwrap()),
        )
        .unwrap();

        let failure = store.retire_record(&mismatched, &context()).unwrap_err();
        assert_eq!(failure.kind(), ServiceIntentStoreError::IntentMismatch);
        assert_eq!(
            failure.disposition(),
            ServiceIntentRetireDisposition::NotRetired
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);

        store.retire_record(&original, &context()).unwrap();
        assert!(!path.exists());
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn post_unlink_sync_failure_is_durability_uncertain_not_retryable_success() {
        let directory = private_directory();
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        let original = runtime_intent(19);
        store.record(&original).unwrap();
        let path = directory.path().join(filename(&original));

        let failure = store
            .retire_record_with_sync(&original, &context(), |_| {
                Err(io::Error::other("fixture sync failure"))
            })
            .unwrap_err();
        assert_eq!(failure.kind(), ServiceIntentStoreError::Io);
        assert_eq!(
            failure.disposition(),
            ServiceIntentRetireDisposition::DurabilityUncertain
        );
        assert!(!path.exists());
    }

    #[test]
    fn malformed_schema_and_fields_fail_closed() {
        let record = intent(9);
        let filename = filename(&record);
        let malformed = [
            br#"{"schema":2,"coordinator_id":"coordinator","unit_name":"l3ms-coordinator-00000000000000000000000000000009.service","launch_identity":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","stop_timeout_millis":30000}"#.to_vec(),
            br#"{"schema":1,"coordinator_id":"coordinator","unit_name":"l3ms-coordinator-00000000000000000000000000000009.service","launch_identity":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","stop_timeout_millis":30000,"argv":["secret"]}"#.to_vec(),
            br#"{"schema":1,"coordinator_id":"coordinator","unit_name":"l3ms-coordinator-00000000000000000000000000000009.service","launch_identity":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}"#.to_vec(),
        ];

        for bytes in malformed {
            let directory = private_directory();
            write_private_file(&directory.path().join(&filename), &bytes, FILE_MODE);
            let store = ServiceIntentStore::open(directory.path()).unwrap();
            assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));
        }
    }

    #[test]
    fn invalid_runtime_binding_unknown_fields_and_corruption_fail_closed() {
        let record = runtime_intent(16);
        let filename = filename(&record);
        let valid = serde_json::to_value(PersistedIntent::from_record(&record).unwrap()).unwrap();
        let mut cases = Vec::new();

        let mut invalid_lane = valid.clone();
        invalid_lane["runtime_lane"] = serde_json::Value::String("unknown".to_owned());
        cases.push(serde_json::to_vec(&invalid_lane).unwrap());

        let mut invalid_name = valid.clone();
        invalid_name["runtime_name"] = serde_json::Value::String("bad/name".to_owned());
        cases.push(serde_json::to_vec(&invalid_name).unwrap());

        let mut oversized_name = valid.clone();
        oversized_name["runtime_name"] = serde_json::Value::String("x".repeat(129));
        cases.push(serde_json::to_vec(&oversized_name).unwrap());

        let mut missing_lane = valid.clone();
        missing_lane.as_object_mut().unwrap().remove("runtime_lane");
        cases.push(serde_json::to_vec(&missing_lane).unwrap());

        let mut explicit_null = valid.clone();
        explicit_null["runtime_lane"] = serde_json::Value::Null;
        cases.push(serde_json::to_vec(&explicit_null).unwrap());

        let mut unbound_with_nulls = valid.clone();
        unbound_with_nulls["schema"] = serde_json::Value::from(1);
        unbound_with_nulls["runtime_lane"] = serde_json::Value::Null;
        unbound_with_nulls["runtime_name"] = serde_json::Value::Null;
        cases.push(serde_json::to_vec(&unbound_with_nulls).unwrap());

        let mut wrong_schema = valid.clone();
        wrong_schema["schema"] = serde_json::Value::from(1);
        cases.push(serde_json::to_vec(&wrong_schema).unwrap());

        let mut unknown_field = valid;
        unknown_field["argv"] = serde_json::json!(["must-not-be-persisted"]);
        cases.push(serde_json::to_vec(&unknown_field).unwrap());
        cases.push(b"{truncated".to_vec());

        for bytes in cases {
            let directory = private_directory();
            write_private_file(&directory.path().join(&filename), &bytes, FILE_MODE);
            let store = ServiceIntentStore::open(directory.path()).unwrap();
            assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));
        }
    }

    #[test]
    fn malformed_supervisor_role_identity_and_binding_fail_closed() {
        let record = supervisor_intent(20);
        let filename = filename(&record);
        let valid = serde_json::to_value(PersistedIntent::from_record(&record).unwrap()).unwrap();
        let mut cases = Vec::new();

        let mut missing_identity = valid.clone();
        missing_identity
            .as_object_mut()
            .unwrap()
            .remove("coordinator_identity");
        cases.push(serde_json::to_vec(&missing_identity).unwrap());

        let mut missing_binding = valid.clone();
        missing_binding
            .as_object_mut()
            .unwrap()
            .remove("runtime_name");
        cases.push(serde_json::to_vec(&missing_binding).unwrap());

        let mut invalid_pid = valid.clone();
        invalid_pid["coordinator_identity"]["pid"] = serde_json::Value::from(0);
        cases.push(serde_json::to_vec(&invalid_pid).unwrap());

        let mut unknown_role = valid.clone();
        unknown_role["launch_role"] = serde_json::Value::String("engine_bridge".to_owned());
        cases.push(serde_json::to_vec(&unknown_role).unwrap());

        let mut legacy_schema_with_role = valid;
        legacy_schema_with_role["schema"] = serde_json::Value::from(2);
        cases.push(serde_json::to_vec(&legacy_schema_with_role).unwrap());

        for bytes in cases {
            let directory = private_directory();
            write_private_file(&directory.path().join(&filename), &bytes, FILE_MODE);
            let store = ServiceIntentStore::open(directory.path()).unwrap();
            assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));
        }
    }

    #[test]
    fn oversized_records_fail_closed_after_bounded_read() {
        let directory = private_directory();
        let record = intent(10);
        write_private_file(
            &directory.path().join(filename(&record)),
            &vec![b'x'; MAX_RECORD_BYTES + 1],
            FILE_MODE,
        );
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));
    }

    #[test]
    fn symlinks_and_hardlinks_are_rejected() {
        let symlink_directory = private_directory();
        let target = symlink_directory.path().join("outside");
        std::fs::write(&target, b"retained elsewhere").unwrap();
        symlink(
            &target,
            symlink_directory.path().join(filename(&intent(11))),
        )
        .unwrap();
        let store = ServiceIntentStore::open(symlink_directory.path()).unwrap();
        assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));
        assert_eq!(std::fs::read(target).unwrap(), b"retained elsewhere");

        let hardlink_directory = private_directory();
        let hardlink_record = intent(12);
        let target = hardlink_directory.path().join("outside");
        write_private_file(&target, &raw_record(&hardlink_record), FILE_MODE);
        std::fs::hard_link(
            &target,
            hardlink_directory.path().join(filename(&hardlink_record)),
        )
        .unwrap();
        let store = ServiceIntentStore::open(hardlink_directory.path()).unwrap();
        assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));
    }

    #[test]
    fn fifo_with_a_valid_intent_filename_is_rejected_without_blocking() {
        let directory = private_directory();
        let path = directory.path().join(filename(&intent(14)));
        let path_c = CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { libc::mkfifo(path_c.as_ptr(), FILE_MODE as libc::mode_t) },
            0
        );

        let store = ServiceIntentStore::open(directory.path()).unwrap();
        assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));
    }

    #[test]
    fn unsafe_permissions_unexpected_files_and_symlink_directories_are_rejected() {
        let directory = private_directory();
        let record = intent(13);
        write_private_file(
            &directory.path().join(filename(&record)),
            &raw_record(&record),
            0o640,
        );
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));

        let directory = private_directory();
        write_private_file(&directory.path().join("notes"), b"unexpected", FILE_MODE);
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        assert_eq!(store.list(), Err(ServiceIntentStoreError::CorruptStore));

        let directory = private_directory();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o750)).unwrap();
        assert_eq!(
            ServiceIntentStore::open(directory.path()).err(),
            Some(ServiceIntentStoreError::UnsafeDirectory)
        );

        let outer = tempfile::tempdir().unwrap();
        let target = private_directory();
        let alias = outer.path().join("alias");
        symlink(target.path(), &alias).unwrap();
        assert_eq!(
            ServiceIntentStore::open(&alias).err(),
            Some(ServiceIntentStoreError::UnsafeDirectory)
        );
    }

    #[test]
    fn missing_directory_is_not_created_and_store_limit_is_enforced() {
        let parent = tempfile::tempdir().unwrap();
        let absent = parent.path().join("not-created").join("intents");
        assert!(ServiceIntentStore::open(&absent).is_err());
        assert!(!parent.path().join("not-created").exists());

        let directory = private_directory();
        for number in 0..=MAX_INTENTS {
            let record = intent(number as u128);
            write_private_file(
                &directory.path().join(filename(&record)),
                &raw_record(&record),
                FILE_MODE,
            );
        }
        let store = ServiceIntentStore::open(directory.path()).unwrap();
        assert_eq!(store.list(), Err(ServiceIntentStoreError::CapacityExceeded));
    }
}
