//! Pinned, in-memory registry for already-published native media artifacts.
//!
//! Artifacts are captured only after the owning runtime has independently
//! completed its stop and release proof. The store does not open
//! request-selected paths: capture is an internal publication step, while
//! retrieval resolves an exact `(request_id, MediaArtifact)` pair to a pinned
//! descriptor and returns one bounded byte range.

use std::collections::HashMap;
use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::ipc::validate_request_id;
#[cfg(test)]
use super::result_buffer::ArtifactKind;
use super::result_buffer::MediaArtifact;
use super::runtime::OperationContext;

const PRIVATE_DIRECTORY_MODE: u32 = 0o700;
const MAX_ROOT_PATH_BYTES: usize = 4096;
const MAX_ROOT_COMPONENTS: usize = 64;
const MAX_NATIVE_PATH_BYTES: usize = 4096;
const MAX_NATIVE_COMPONENT_BYTES: usize = 255;
const MAX_NATIVE_COMPONENTS: usize = 64;
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ARTIFACTS_PER_REQUEST: usize = 128;
const MAX_REQUESTS: usize = 128;
const MAX_PINNED_FILE_DESCRIPTORS: usize = 256;
const MIN_CHUNK_BYTES: usize = 1;
const MAX_CHUNK_BYTES: usize = 64 * 1024;
const VERSION_BYTES: usize = 32;

/// Fixed failures from artifact publication and retrieval. No paths, OS
/// errors, contents, or backend data are retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArtifactStoreError {
    Invalid,
    Unavailable,
    Changed,
    Duplicate,
    CapacityExceeded,
    InvalidBudget,
    InvalidOffset,
    RandomnessUnavailable,
}

impl fmt::Display for ArtifactStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Invalid => "media artifact reference is invalid",
            Self::Unavailable => "published media artifact is unavailable",
            Self::Changed => "published media artifact changed",
            Self::Duplicate => "media artifact publication collides with retained data",
            Self::CapacityExceeded => "media artifact store capacity exceeded",
            Self::InvalidBudget => "media artifact byte budget is invalid",
            Self::InvalidOffset => "media artifact offset is invalid",
            Self::RandomnessUnavailable => "media artifact version is unavailable",
        })
    }
}

impl Error for ArtifactStoreError {}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    mode: u32,
}

impl DirectoryIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            uid: metadata.uid(),
            mode: metadata.mode() & 0o7777,
        }
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    links: u64,
    length: u64,
    uid: u32,
    mode: u32,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            links: metadata.nlink(),
            length: metadata.len(),
            uid: metadata.uid(),
            mode: metadata.mode() & 0o7777,
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

struct PinnedRoot {
    directory: File,
    path: PathBuf,
    identity: DirectoryIdentity,
    uid: u32,
}

impl PinnedRoot {
    fn matches_key(&self, other: &Self) -> bool {
        self.path == other.path && self.identity == other.identity && self.uid == other.uid
    }

    fn verify_named(&self) -> Result<(), ArtifactStoreError> {
        verify_private_root_fd(&self.directory, self.identity, self.uid)
            .map_err(|_| ArtifactStoreError::Changed)?;
        let (current, identity) =
            open_private_root(&self.path).map_err(|_| ArtifactStoreError::Changed)?;
        drop(current);
        if identity != self.identity {
            return Err(ArtifactStoreError::Changed);
        }
        Ok(())
    }
}

/// Opaque evidence for one already-published file. It is intentionally not
/// cloneable or serializable; dropping it releases its descriptor unless the
/// registry retains it. Debug output never includes native or public paths.
pub(crate) struct PublishedArtifact {
    artifact: MediaArtifact,
    native_relative_path: String,
    root: Arc<PinnedRoot>,
    file: File,
    identity: FileIdentity,
    version: String,
}

impl PublishedArtifact {
    /// Pin one file inside an existing current-UID mode-0700 private root.
    /// All root and file components are opened with `O_NOFOLLOW`; the file
    /// itself is opened read-only and must be regular, single-link, nonempty,
    /// and no larger than 512 MiB.
    pub(crate) fn capture(
        private_job_root: &Path,
        artifact: MediaArtifact,
        native_relative_path: &str,
        context: &OperationContext<'_>,
    ) -> Result<Self, ArtifactStoreError> {
        let artifact = MediaArtifact::new(artifact.kind, artifact.relative_path.clone())
            .map_err(|_| ArtifactStoreError::Invalid)?;
        validate_native_relative_path(native_relative_path)?;
        check_context(context)?;

        let (directory, identity) = open_private_root(private_job_root)?;
        let root = Arc::new(PinnedRoot {
            directory,
            path: private_job_root.to_path_buf(),
            identity,
            uid: effective_uid(),
        });
        root.verify_named()?;

        let file = open_native_file(&root, native_relative_path)?;
        let file_identity = validate_published_file(&file, root.uid)?;
        root.verify_named()?;
        verify_named_file(&root, native_relative_path, file_identity)?;
        let version = generate_version(context)?;

        Ok(Self {
            artifact,
            native_relative_path: native_relative_path.to_owned(),
            root,
            file,
            identity: file_identity,
            version,
        })
    }

    pub(crate) fn reference(&self) -> &MediaArtifact {
        &self.artifact
    }

    fn verify_current(&self) -> Result<(), ArtifactStoreError> {
        self.root.verify_named()?;
        let identity = validate_published_file(&self.file, self.root.uid)
            .map_err(|_| ArtifactStoreError::Changed)?;
        if identity != self.identity {
            return Err(ArtifactStoreError::Changed);
        }
        verify_named_file(&self.root, &self.native_relative_path, self.identity)?;
        self.root.verify_named()?;
        Ok(())
    }
}

impl fmt::Debug for PublishedArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PublishedArtifact([pinned file, redacted])")
    }
}

/// One bounded page from an exact published artifact. Contents are returned
/// only to the explicit caller and are never included in Debug output.
pub(crate) struct ArtifactChunk {
    pub(crate) artifact: MediaArtifact,
    pub(crate) offset: u64,
    pub(crate) total_bytes: u64,
    pub(crate) version: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) next_offset: u64,
    pub(crate) eof: bool,
}

impl fmt::Debug for ArtifactChunk {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactChunk")
            .field("artifact", &self.artifact)
            .field("offset", &self.offset)
            .field("total_bytes", &self.total_bytes)
            .field("version", &"<redacted>")
            .field("byte_count", &self.bytes.len())
            .field("next_offset", &self.next_offset)
            .field("eof", &self.eof)
            .finish()
    }
}

/// In-memory, non-restorable registry for artifacts whose native owners have
/// already proved release. Requests and references are exact-match keys; no
/// client-supplied filesystem path is ever opened during retrieval.
pub(crate) struct ArtifactStore {
    requests: HashMap<String, Vec<PublishedArtifact>>,
}

impl ArtifactStore {
    pub(crate) fn new() -> Self {
        Self {
            requests: HashMap::new(),
        }
    }

    /// Atomically retain a complete artifact batch for one request. Duplicate
    /// request IDs and public references collide; a failed batch retains
    /// nothing and releases every handle passed to this call.
    pub(crate) fn register(
        &mut self,
        request_id: &str,
        mut artifacts: Vec<PublishedArtifact>,
    ) -> Result<(), ArtifactStoreError> {
        validate_request_id(request_id).map_err(|_| ArtifactStoreError::Invalid)?;
        if artifacts.is_empty() || artifacts.len() > MAX_ARTIFACTS_PER_REQUEST {
            return Err(ArtifactStoreError::CapacityExceeded);
        }
        if self.requests.contains_key(request_id) {
            return Err(ArtifactStoreError::Duplicate);
        }
        if self.requests.len() >= MAX_REQUESTS {
            return Err(ArtifactStoreError::CapacityExceeded);
        }

        for (index, artifact) in artifacts.iter().enumerate() {
            artifact.verify_current()?;
            if artifacts[..index]
                .iter()
                .any(|other| other.artifact == artifact.artifact)
                || self
                    .requests
                    .values()
                    .flatten()
                    .any(|other| other.artifact == artifact.artifact)
            {
                return Err(ArtifactStoreError::Duplicate);
            }
        }

        let mut roots = Vec::<Arc<PinnedRoot>>::new();
        for root in self
            .requests
            .values()
            .flatten()
            .map(|artifact| Arc::clone(&artifact.root))
            .chain(artifacts.iter().map(|artifact| Arc::clone(&artifact.root)))
        {
            if !roots.iter().any(|known| known.matches_key(&root)) {
                roots.push(root);
            }
        }
        for artifact in &mut artifacts {
            let root = roots
                .iter()
                .find(|known| known.matches_key(&artifact.root))
                .ok_or(ArtifactStoreError::Unavailable)?;
            artifact.root = Arc::clone(root);
        }

        let artifact_count = self
            .requests
            .values()
            .map(Vec::len)
            .sum::<usize>()
            .saturating_add(artifacts.len());
        let descriptor_count = artifact_count.saturating_add(roots.len());
        if descriptor_count > MAX_PINNED_FILE_DESCRIPTORS {
            return Err(ArtifactStoreError::CapacityExceeded);
        }

        self.requests.insert(request_id.to_owned(), artifacts);
        Ok(())
    }

    /// Read one page from an exact registered public reference. The first
    /// page must start at offset zero and omits the version token; all later
    /// pages must echo the same opaque 64-character version.
    pub(crate) fn read_chunk(
        &self,
        request_id: &str,
        artifact: &MediaArtifact,
        offset: u64,
        budget: usize,
        version: Option<&str>,
        context: &OperationContext<'_>,
    ) -> Result<ArtifactChunk, ArtifactStoreError> {
        validate_request_id(request_id).map_err(|_| ArtifactStoreError::Invalid)?;
        MediaArtifact::new(artifact.kind, artifact.relative_path.clone())
            .map_err(|_| ArtifactStoreError::Invalid)?;
        if !(MIN_CHUNK_BYTES..=MAX_CHUNK_BYTES).contains(&budget) {
            return Err(ArtifactStoreError::InvalidBudget);
        }
        match version {
            None if offset != 0 => return Err(ArtifactStoreError::InvalidOffset),
            Some(value) if !valid_version(value) => return Err(ArtifactStoreError::Invalid),
            _ => {}
        }
        check_context(context)?;

        let published = self
            .requests
            .get(request_id)
            .and_then(|registered| registered.iter().find(|item| item.artifact == *artifact))
            .ok_or(ArtifactStoreError::Unavailable)?;
        if version.is_some_and(|value| value != published.version.as_str()) {
            return Err(ArtifactStoreError::Changed);
        }
        if offset > published.identity.length {
            return Err(ArtifactStoreError::InvalidOffset);
        }
        published.verify_current()?;
        check_context(context)?;

        let remaining = published.identity.length - offset;
        let wanted = usize::try_from(remaining.min(budget as u64))
            .map_err(|_| ArtifactStoreError::InvalidOffset)?;
        let mut bytes = Vec::with_capacity(wanted);
        while bytes.len() < wanted {
            check_context(context)?;
            let read_offset = offset
                .checked_add(bytes.len() as u64)
                .ok_or(ArtifactStoreError::InvalidOffset)?;
            let result = unsafe {
                libc::pread(
                    published.file.as_raw_fd(),
                    bytes.spare_capacity_mut().as_mut_ptr().cast(),
                    wanted - bytes.len(),
                    read_offset as libc::off_t,
                )
            };
            if result > 0 {
                let read = result as usize;
                unsafe { bytes.set_len(bytes.len() + read) };
                continue;
            }
            if result == 0 {
                return Err(ArtifactStoreError::Changed);
            }
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(ArtifactStoreError::Unavailable);
        }

        published.verify_current()?;
        check_context(context)?;
        let next_offset = offset
            .checked_add(bytes.len() as u64)
            .ok_or(ArtifactStoreError::InvalidOffset)?;
        Ok(ArtifactChunk {
            artifact: published.artifact.clone(),
            offset,
            total_bytes: published.identity.length,
            version: published.version.clone(),
            bytes,
            next_offset,
            eof: next_offset == published.identity.length,
        })
    }

    /// Explicitly release all pins retained for one request. Nothing expires
    /// or restores implicitly across requests or process restarts.
    pub(crate) fn forget(&mut self, request_id: &str) -> Result<(), ArtifactStoreError> {
        validate_request_id(request_id).map_err(|_| ArtifactStoreError::Invalid)?;
        self.requests
            .remove(request_id)
            .map(|_| ())
            .ok_or(ArtifactStoreError::Unavailable)
    }

    #[cfg(test)]
    fn pinned_descriptor_count(&self) -> usize {
        let artifacts = self.requests.values().map(Vec::len).sum::<usize>();
        let mut roots = Vec::<&PinnedRoot>::new();
        for root in self
            .requests
            .values()
            .flatten()
            .map(|artifact| &*artifact.root)
        {
            if !roots.iter().any(|known| known.matches_key(root)) {
                roots.push(root);
            }
        }
        artifacts + roots.len()
    }
}

impl Default for ArtifactStore {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ArtifactStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactStore")
            .field("request_count", &self.requests.len())
            .field(
                "artifact_count",
                &self.requests.values().map(Vec::len).sum::<usize>(),
            )
            .field("native_paths", &"<redacted>")
            .finish()
    }
}

fn validate_native_relative_path(path: &str) -> Result<(), ArtifactStoreError> {
    if path.is_empty()
        || path.len() > MAX_NATIVE_PATH_BYTES
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains(':')
        || path.chars().any(char::is_control)
        || path.split('/').count() > MAX_NATIVE_COMPONENTS
        || path.split('/').any(|component| {
            component.is_empty()
                || component == "."
                || component == ".."
                || component.len() > MAX_NATIVE_COMPONENT_BYTES
        })
    {
        return Err(ArtifactStoreError::Invalid);
    }
    Ok(())
}

fn validate_root_path(path: &Path) -> Result<(), ArtifactStoreError> {
    let bytes = path.as_os_str().as_bytes();
    if !path.is_absolute()
        || bytes.len() < 2
        || bytes.len() > MAX_ROOT_PATH_BYTES
        || bytes.contains(&0)
        || bytes[0] != b'/'
        || bytes[1..].split(|byte| *byte == b'/').count() > MAX_ROOT_COMPONENTS
        || bytes[1..]
            .split(|byte| *byte == b'/')
            .any(|component| component.is_empty() || component == b"." || component == b"..")
    {
        return Err(ArtifactStoreError::Invalid);
    }
    Ok(())
}

fn open_private_root(path: &Path) -> Result<(File, DirectoryIdentity), ArtifactStoreError> {
    validate_root_path(path)?;
    let root_name = CString::new("/").map_err(|_| ArtifactStoreError::Invalid)?;
    let root_fd = unsafe {
        libc::open(
            root_name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if root_fd < 0 {
        return Err(ArtifactStoreError::Unavailable);
    }
    let mut directory = unsafe { File::from_raw_fd(root_fd) };
    for component in path.as_os_str().as_bytes()[1..].split(|byte| *byte == b'/') {
        let name = CString::new(component).map_err(|_| ArtifactStoreError::Invalid)?;
        let next = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if next < 0 {
            return Err(ArtifactStoreError::Unavailable);
        }
        directory = unsafe { File::from_raw_fd(next) };
    }
    let uid = effective_uid();
    let metadata = directory
        .metadata()
        .map_err(|_| ArtifactStoreError::Unavailable)?;
    if !metadata.is_dir()
        || metadata.uid() != uid
        || metadata.mode() & 0o7777 != PRIVATE_DIRECTORY_MODE
    {
        return Err(ArtifactStoreError::Invalid);
    }
    Ok((directory, DirectoryIdentity::from_metadata(&metadata)))
}

fn verify_private_root_fd(
    directory: &File,
    identity: DirectoryIdentity,
    uid: u32,
) -> Result<(), ArtifactStoreError> {
    let metadata = directory
        .metadata()
        .map_err(|_| ArtifactStoreError::Changed)?;
    if !metadata.is_dir()
        || metadata.uid() != uid
        || metadata.mode() & 0o7777 != PRIVATE_DIRECTORY_MODE
        || DirectoryIdentity::from_metadata(&metadata) != identity
    {
        return Err(ArtifactStoreError::Changed);
    }
    Ok(())
}

fn open_native_file(root: &PinnedRoot, relative_path: &str) -> Result<File, ArtifactStoreError> {
    validate_native_relative_path(relative_path)?;
    let mut directory = root
        .directory
        .try_clone()
        .map_err(|_| ArtifactStoreError::Unavailable)?;
    let mut components = relative_path.split('/').peekable();
    while let Some(component) = components.next() {
        let name = CString::new(component).map_err(|_| ArtifactStoreError::Invalid)?;
        if components.peek().is_some() {
            let fd = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            };
            if fd < 0 {
                return Err(ArtifactStoreError::Unavailable);
            }
            directory = unsafe { File::from_raw_fd(fd) };
            let metadata = directory
                .metadata()
                .map_err(|_| ArtifactStoreError::Unavailable)?;
            if !metadata.is_dir() || metadata.uid() != root.uid {
                return Err(ArtifactStoreError::Unavailable);
            }
        } else {
            let fd = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
                )
            };
            if fd < 0 {
                return Err(ArtifactStoreError::Unavailable);
            }
            return Ok(unsafe { File::from_raw_fd(fd) });
        }
    }
    Err(ArtifactStoreError::Invalid)
}

fn validate_published_file(file: &File, uid: u32) -> Result<FileIdentity, ArtifactStoreError> {
    let metadata = file
        .metadata()
        .map_err(|_| ArtifactStoreError::Unavailable)?;
    let identity = FileIdentity::from_metadata(&metadata);
    if !metadata.is_file() || identity.uid != uid || identity.links != 1 || identity.length == 0 {
        return Err(ArtifactStoreError::Invalid);
    }
    if identity.length > MAX_ARTIFACT_BYTES {
        return Err(ArtifactStoreError::CapacityExceeded);
    }
    Ok(identity)
}

fn verify_named_file(
    root: &PinnedRoot,
    relative_path: &str,
    expected: FileIdentity,
) -> Result<(), ArtifactStoreError> {
    let file = open_native_file(root, relative_path).map_err(|_| ArtifactStoreError::Changed)?;
    let identity =
        validate_published_file(&file, root.uid).map_err(|_| ArtifactStoreError::Changed)?;
    if identity == expected {
        Ok(())
    } else {
        Err(ArtifactStoreError::Changed)
    }
}

fn effective_uid() -> u32 {
    unsafe { libc::geteuid() }
}

fn check_context(context: &OperationContext<'_>) -> Result<(), ArtifactStoreError> {
    context
        .check_active("media artifact access")
        .map_err(|_| ArtifactStoreError::Unavailable)
}

fn generate_version(context: &OperationContext<'_>) -> Result<String, ArtifactStoreError> {
    let mut bytes = [0u8; VERSION_BYTES];
    let mut offset = 0usize;
    while offset < bytes.len() {
        check_context(context)?;
        let count = unsafe {
            libc::getrandom(
                bytes[offset..].as_mut_ptr().cast(),
                bytes.len() - offset,
                libc::GRND_NONBLOCK,
            )
        };
        if count > 0 {
            offset += count as usize;
            continue;
        }
        if count < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(ArtifactStoreError::RandomnessUnavailable);
    }
    let mut version = String::with_capacity(VERSION_BYTES * 2);
    for byte in bytes {
        use fmt::Write;
        let _ = write!(version, "{byte:02x}");
    }
    Ok(version)
}

fn valid_version(version: &str) -> bool {
    version.len() == VERSION_BYTES * 2
        && version
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, PermissionsExt};
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    static NEVER_CANCEL: super::super::runtime::NeverCancel = super::super::runtime::NeverCancel;

    struct Fixture {
        temp: TempDir,
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let root = temp.path().join("job");
            std::fs::create_dir(&root).unwrap();
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
            Self { temp, root }
        }

        fn write(&self, relative: &str, bytes: &[u8]) {
            let path = self.root.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, bytes).unwrap();
        }

        fn capture(&self, public: &str, native: &str) -> PublishedArtifact {
            PublishedArtifact::capture(
                &self.root,
                MediaArtifact::new(ArtifactKind::LocalJob, public).unwrap(),
                native,
                &context(),
            )
            .unwrap()
        }
    }

    fn context() -> OperationContext<'static> {
        OperationContext::new(Instant::now() + Duration::from_secs(5), &NEVER_CANCEL)
    }

    fn file_inode(path: &Path) -> (u64, u64) {
        let metadata = std::fs::metadata(path).unwrap();
        (metadata.dev(), metadata.ino())
    }

    fn fd_count_for_inode(expected: (u64, u64)) -> usize {
        std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|entry| std::fs::metadata(entry.path()).ok())
            .filter(|metadata| (metadata.dev(), metadata.ino()) == expected)
            .count()
    }

    fn file_name(index: usize) -> String {
        format!("file-{index:03}.bin")
    }

    #[test]
    fn valid_artifacts_are_paged_with_one_opaque_version() {
        let fixture = Fixture::new();
        fixture.write("outputs/audio.bin", b"0123456789");
        let artifact = fixture.capture("jobs/41/audio.bin", "outputs/audio.bin");
        assert_eq!(
            format!("{artifact:?}"),
            "PublishedArtifact([pinned file, redacted])"
        );

        let mut store = ArtifactStore::new();
        store
            .register("caller:artifact-41", vec![artifact])
            .unwrap();
        let public = MediaArtifact::new(ArtifactKind::LocalJob, "jobs/41/audio.bin").unwrap();
        let first = store
            .read_chunk("caller:artifact-41", &public, 0, 4, None, &context())
            .unwrap();
        assert_eq!(first.bytes, b"0123");
        assert_eq!(first.offset, 0);
        assert_eq!(first.total_bytes, 10);
        assert_eq!(first.next_offset, 4);
        assert!(!first.eof);
        assert!(valid_version(&first.version));

        let second = store
            .read_chunk(
                "caller:artifact-41",
                &public,
                first.next_offset,
                4,
                Some(&first.version),
                &context(),
            )
            .unwrap();
        assert_eq!(second.bytes, b"4567");
        assert_eq!(second.version, first.version);
        let last = store
            .read_chunk(
                "caller:artifact-41",
                &public,
                second.next_offset,
                64 * 1024,
                Some(&first.version),
                &context(),
            )
            .unwrap();
        assert_eq!(last.bytes, b"89");
        assert_eq!(last.next_offset, 10);
        assert!(last.eof);
        assert!(!format!("{last:?}").contains("89"));
        assert!(matches!(
            store.read_chunk("caller:artifact-41", &public, 1, 1, None, &context()),
            Err(ArtifactStoreError::InvalidOffset)
        ));
        assert!(matches!(
            store.read_chunk("caller:artifact-41", &public, 0, 0, None, &context()),
            Err(ArtifactStoreError::InvalidBudget)
        ));
        assert!(matches!(
            store.read_chunk(
                "caller:artifact-41",
                &public,
                0,
                MAX_CHUNK_BYTES + 1,
                None,
                &context()
            ),
            Err(ArtifactStoreError::InvalidBudget)
        ));
        assert!(matches!(
            store.read_chunk(
                "caller:artifact-41",
                &public,
                4,
                4,
                Some("0".repeat(VERSION_BYTES * 2).as_str()),
                &context()
            ),
            Err(ArtifactStoreError::Changed)
        ));
        assert!(matches!(
            store.read_chunk(
                "caller:artifact-41",
                &public,
                0,
                4,
                Some("invalid"),
                &context()
            ),
            Err(ArtifactStoreError::Invalid)
        ));
        store.forget("caller:artifact-41").unwrap();
        assert!(matches!(
            store.read_chunk("caller:artifact-41", &public, 0, 4, None, &context()),
            Err(ArtifactStoreError::Unavailable)
        ));
        drop(fixture.temp);
    }

    #[test]
    fn capture_rejects_traversal_symlinks_fifo_empty_and_oversized_files() {
        let fixture = Fixture::new();
        fixture.write("target.bin", b"payload");
        assert_eq!(
            PublishedArtifact::capture(
                &fixture.root,
                MediaArtifact::new(ArtifactKind::LocalJob, "public/file.bin").unwrap(),
                "../target.bin",
                &context()
            )
            .map(|_| ()),
            Err(ArtifactStoreError::Invalid)
        );

        let public_root = fixture.temp.path().join("not-private");
        std::fs::create_dir(&public_root).unwrap();
        std::fs::set_permissions(&public_root, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(public_root.join("file.bin"), b"payload").unwrap();
        assert_eq!(
            PublishedArtifact::capture(
                &public_root,
                MediaArtifact::new(ArtifactKind::LocalJob, "public/file.bin").unwrap(),
                "file.bin",
                &context()
            )
            .map(|_| ()),
            Err(ArtifactStoreError::Invalid)
        );

        symlink("target.bin", fixture.root.join("link.bin")).unwrap();
        assert!(PublishedArtifact::capture(
            &fixture.root,
            MediaArtifact::new(ArtifactKind::LocalJob, "public/link.bin").unwrap(),
            "link.bin",
            &context()
        )
        .is_err());

        fixture.write("real-dir/nested.bin", b"nested");
        symlink("real-dir", fixture.root.join("dir-link")).unwrap();
        assert!(PublishedArtifact::capture(
            &fixture.root,
            MediaArtifact::new(ArtifactKind::LocalJob, "public/nested.bin").unwrap(),
            "dir-link/nested.bin",
            &context()
        )
        .is_err());

        let fifo = fixture.root.join("fifo.bin");
        let fifo_name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
        assert!(PublishedArtifact::capture(
            &fixture.root,
            MediaArtifact::new(ArtifactKind::LocalJob, "public/fifo.bin").unwrap(),
            "fifo.bin",
            &context()
        )
        .is_err());

        fixture.write("empty.bin", b"");
        assert!(PublishedArtifact::capture(
            &fixture.root,
            MediaArtifact::new(ArtifactKind::LocalJob, "public/empty.bin").unwrap(),
            "empty.bin",
            &context()
        )
        .is_err());

        let large = File::create(fixture.root.join("large.bin")).unwrap();
        large.set_len(MAX_ARTIFACT_BYTES + 1).unwrap();
        assert_eq!(
            PublishedArtifact::capture(
                &fixture.root,
                MediaArtifact::new(ArtifactKind::LocalJob, "public/large.bin").unwrap(),
                "large.bin",
                &context()
            )
            .map(|_| ()),
            Err(ArtifactStoreError::CapacityExceeded)
        );
        assert!(fixture.temp.path().exists());
    }

    #[test]
    fn root_and_file_replacement_or_mutation_invalidate_a_published_pin() {
        let fixture = Fixture::new();
        fixture.write("file.bin", b"original");
        let root_replaced = fixture.capture("job/root.bin", "file.bin");
        let moved = fixture.root.with_extension("moved");
        std::fs::rename(&fixture.root, &moved).unwrap();
        std::fs::create_dir(&fixture.root).unwrap();
        std::fs::set_permissions(&fixture.root, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut root_store = ArtifactStore::new();
        root_store
            .register("caller:root-replaced", vec![root_replaced])
            .unwrap_err();

        let file_pin = PublishedArtifact::capture(
            &moved,
            MediaArtifact::new(ArtifactKind::LocalJob, "job/file.bin").unwrap(),
            "file.bin",
            &context(),
        )
        .unwrap();
        let mut file_store = ArtifactStore::new();
        file_store
            .register("caller:file-replaced", vec![file_pin])
            .unwrap();
        std::fs::rename(moved.join("file.bin"), moved.join("old.bin")).unwrap();
        std::fs::write(moved.join("file.bin"), b"replacement").unwrap();
        let public = MediaArtifact::new(ArtifactKind::LocalJob, "job/file.bin").unwrap();
        assert!(matches!(
            file_store.read_chunk("caller:file-replaced", &public, 0, 64, None, &context()),
            Err(ArtifactStoreError::Changed)
        ));

        let mutation_pin = PublishedArtifact::capture(
            &moved,
            MediaArtifact::new(ArtifactKind::LocalJob, "job/old.bin").unwrap(),
            "old.bin",
            &context(),
        )
        .unwrap();
        let mut mutation_store = ArtifactStore::new();
        mutation_store
            .register("caller:file-mutated", vec![mutation_pin])
            .unwrap();
        std::fs::write(moved.join("old.bin"), b"mutated!").unwrap();
        let public = MediaArtifact::new(ArtifactKind::LocalJob, "job/old.bin").unwrap();
        assert!(matches!(
            mutation_store.read_chunk("caller:file-mutated", &public, 0, 64, None, &context()),
            Err(ArtifactStoreError::Changed)
        ));
    }

    #[test]
    fn foreign_job_and_public_reference_collisions_fail_closed() {
        let fixture = Fixture::new();
        fixture.write("one.bin", b"one");
        fixture.write("two.bin", b"two");
        let first = fixture.capture("jobs/shared.bin", "one.bin");
        let second = fixture.capture("jobs/shared.bin", "two.bin");
        let mut store = ArtifactStore::new();
        store.register("caller:job-a", vec![first]).unwrap();
        assert_eq!(
            store.register("caller:job-b", vec![second]),
            Err(ArtifactStoreError::Duplicate)
        );
        let public = MediaArtifact::new(ArtifactKind::LocalJob, "jobs/shared.bin").unwrap();
        assert!(matches!(
            store.read_chunk("caller:job-b", &public, 0, 3, None, &context()),
            Err(ArtifactStoreError::Unavailable)
        ));
        assert_eq!(
            store.register("caller:job-a", vec![]),
            Err(ArtifactStoreError::CapacityExceeded)
        );
    }

    #[test]
    fn quotas_are_transactional_for_jobs_artifacts_and_pinned_descriptors() {
        let fixture = Fixture::new();
        let mut store = ArtifactStore::new();
        for request in 0..MAX_REQUESTS {
            let name = file_name(request);
            fixture.write(&name, b"x");
            let artifact = fixture.capture(&format!("jobs/{request}.bin"), &name);
            store
                .register(&format!("caller:quota-{request}"), vec![artifact])
                .unwrap();
        }
        assert_eq!(store.requests.len(), MAX_REQUESTS);
        let extra_name = file_name(MAX_REQUESTS);
        fixture.write(&extra_name, b"x");
        let extra = fixture.capture("jobs/extra.bin", &extra_name);
        assert_eq!(
            store.register("caller:quota-extra", vec![extra]),
            Err(ArtifactStoreError::CapacityExceeded)
        );
        store.forget("caller:quota-0").unwrap();
        let extra = fixture.capture("jobs/extra.bin", &extra_name);
        store.register("caller:quota-extra", vec![extra]).unwrap();

        let mut over_artifact_limit = Vec::with_capacity(MAX_ARTIFACTS_PER_REQUEST + 1);
        for index in 0..=MAX_ARTIFACTS_PER_REQUEST {
            let name = format!("batch-{}-file.bin", index);
            fixture.write(&name, b"x");
            over_artifact_limit.push(fixture.capture(&format!("batch/{index}.bin"), &name));
        }
        assert_eq!(
            store.register("caller:too-many-artifacts", over_artifact_limit),
            Err(ArtifactStoreError::CapacityExceeded)
        );

        let mut descriptor_store = ArtifactStore::new();
        let mut first_batch = Vec::with_capacity(MAX_ARTIFACTS_PER_REQUEST);
        let mut first_batch_inodes = Vec::with_capacity(MAX_ARTIFACTS_PER_REQUEST);
        for index in 0..MAX_ARTIFACTS_PER_REQUEST {
            let name = format!("fd-first-{index}.bin");
            fixture.write(&name, b"x");
            first_batch_inodes.push(file_inode(&fixture.root.join(&name)));
            first_batch.push(fixture.capture(&format!("fd-first/{index}.bin"), &name));
        }
        descriptor_store
            .register("caller:fd-first", first_batch)
            .unwrap();
        let before = descriptor_store.pinned_descriptor_count();
        let mut descriptor_batch = Vec::with_capacity(MAX_ARTIFACTS_PER_REQUEST);
        let mut rejected_batch_inodes = Vec::with_capacity(MAX_ARTIFACTS_PER_REQUEST);
        for index in 0..MAX_ARTIFACTS_PER_REQUEST {
            let name = format!("fd-batch-{index}.bin");
            fixture.write(&name, b"x");
            rejected_batch_inodes.push(file_inode(&fixture.root.join(&name)));
            descriptor_batch.push(fixture.capture(&format!("fd-batch/{index}.bin"), &name));
        }
        assert_eq!(
            descriptor_store.register("caller:fd-batch", descriptor_batch),
            Err(ArtifactStoreError::CapacityExceeded)
        );
        assert_eq!(descriptor_store.pinned_descriptor_count(), before);
        assert!(first_batch_inodes
            .into_iter()
            .all(|inode| fd_count_for_inode(inode) == 1));
        assert!(rejected_batch_inodes
            .into_iter()
            .all(|inode| fd_count_for_inode(inode) == 0));
    }

    #[test]
    fn root_path_symlinks_and_replaced_roots_are_rejected() {
        let fixture = Fixture::new();
        fixture.write("file.bin", b"payload");
        let alias = fixture.temp.path().join("alias");
        symlink(&fixture.root, &alias).unwrap();
        assert!(PublishedArtifact::capture(
            &alias,
            MediaArtifact::new(ArtifactKind::LocalJob, "job/file.bin").unwrap(),
            "file.bin",
            &context()
        )
        .is_err());
        let pin = fixture.capture("job/file.bin", "file.bin");
        let mut store = ArtifactStore::new();
        store.register("caller:root-symlink", vec![pin]).unwrap();
        let public = MediaArtifact::new(ArtifactKind::LocalJob, "job/file.bin").unwrap();
        std::fs::rename(&fixture.root, fixture.root.with_extension("moved")).unwrap();
        symlink(fixture.root.with_extension("moved"), &fixture.root).unwrap();
        assert!(matches!(
            store.read_chunk("caller:root-symlink", &public, 0, 64, None, &context()),
            Err(ArtifactStoreError::Changed)
        ));
    }
}
