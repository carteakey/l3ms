//! Durable pending start-intent records for the managed-service coordinator.
//!
//! The caller must hold the host singleton lock for every store operation.
//! The existing absolute directory must be a dedicated, trusted directory on
//! a stable local filesystem. This module neither creates directories nor
//! removes, repairs, or reclaims records. A retained record is a write-ahead
//! fact only; its absence does not establish that a service stopped.

use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};

use serde::{Deserialize, Deserializer, Serialize};

use super::managed_service::{runtime_identity_from_persisted_parts, ServiceIntentRecord};

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
            Self::Io => "service intent store operation failed",
        })
    }
}

impl std::error::Error for ServiceIntentStoreError {}

/// Linux write-ahead store for pending service start intents.
///
/// The store retains each record permanently. It exposes no retirement or
/// deletion operation and must not be shared across coordinators without the
/// caller's host singleton lock.
pub struct ServiceIntentStore {
    directory: File,
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
        ServiceIntentRecord::from_persisted_parts_with_runtime(
            record.coordinator_id().to_owned(),
            record.unit_name().to_owned(),
            record.launch_identity().to_owned(),
            record.stop_timeout_millis(),
            record.runtime_identity().cloned(),
        )
        .map_err(|_| ServiceIntentStoreError::InvalidRecord)?;
        let runtime_lane = record
            .runtime_identity()
            .map(|identity| Some(identity.lane.as_str().to_owned()));
        let runtime_name = record
            .runtime_identity()
            .map(|identity| Some(identity.as_str().to_owned()));
        Ok(Self {
            schema: if runtime_lane.is_some() { 2 } else { 1 },
            coordinator_id: record.coordinator_id().to_owned(),
            unit_name: record.unit_name().to_owned(),
            launch_identity: record.launch_identity().to_owned(),
            stop_timeout_millis: record.stop_timeout_millis(),
            runtime_lane,
            runtime_name,
        })
    }

    fn into_record(self) -> Result<ServiceIntentRecord, ServiceIntentStoreError> {
        let runtime_identity = match (self.schema, self.runtime_lane, self.runtime_name) {
            (1, None, None) => None,
            (2, Some(Some(lane)), Some(Some(name))) => Some(
                runtime_identity_from_persisted_parts(&lane, name)
                    .map_err(|_| ServiceIntentStoreError::CorruptStore)?,
            ),
            _ => return Err(ServiceIntentStoreError::CorruptStore),
        };
        ServiceIntentRecord::from_persisted_parts_with_runtime(
            self.coordinator_id,
            self.unit_name,
            self.launch_identity,
            self.stop_timeout_millis,
            runtime_identity,
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
    use super::*;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{symlink, PermissionsExt};

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
