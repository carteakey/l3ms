//! Read-only observation of a stopped service's owned cgroup-v2 subtree.
//!
//! This module accepts only a `StoppedService` receipt minted after a
//! generation-matched manager stop acknowledgment. It does not inspect an
//! arbitrary caller-supplied path, kill processes, or prove GPU release,
//! admission safety, or runtime health. `Empty` is an observation at one
//! instant, not a reservation or a lifecycle admission decision.
//!
//! The `Removed` observation is intentionally narrow: it is available only
//! after a receipt binds the original unit, InvocationID, and cgroup. It means
//! the acknowledged-stop cgroup path is absent at observation time. This
//! inference assumes no concurrent same-user systemd client can replace or
//! restart the transient unit and that the launched executable tree is trusted
//! not to escape its assigned cgroup. The coordinator host lock serializes
//! coordinator operations but does not fence other same-user manager clients.

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::File;
use std::io::{Read, Take};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};

use super::managed_service::StoppedService;
use super::runtime::OperationContext;

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const CGROUP2_SUPER_MAGIC: libc::c_long = 0x6367_7270;
const MAX_EVENTS_BYTES: usize = 4 * 1024;

/// Fixed redacted failures. No path, event output, syscall text, or generation
/// identifier is retained in this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CgroupReleaseErrorKind {
    UnsupportedFilesystem,
    InvalidReceipt,
    UnsafePath,
    MissingEvents,
    NonRegularEvents,
    MalformedEvents,
    CgroupNotEmpty,
    OutputTooLarge,
    Io,
    DeadlineExceeded,
    Cancelled,
}

impl fmt::Display for CgroupReleaseErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsupportedFilesystem => "cgroup-v2 filesystem is unavailable",
            Self::InvalidReceipt => "stopped-service receipt is invalid",
            Self::UnsafePath => "owned cgroup path is unsafe",
            Self::MissingEvents => "owned cgroup events are unavailable",
            Self::NonRegularEvents => "owned cgroup events are not a regular file",
            Self::MalformedEvents => "owned cgroup events are malformed",
            Self::CgroupNotEmpty => "owned cgroup still contains processes",
            Self::OutputTooLarge => "owned cgroup events exceed the read limit",
            Self::Io => "owned cgroup observation failed",
            Self::DeadlineExceeded => "cgroup observation deadline expired",
            Self::Cancelled => "cgroup observation was cancelled",
        })
    }
}

impl Error for CgroupReleaseErrorKind {}

/// A point-in-time observation, not a GPU-release or admission proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CgroupObservationKind {
    Empty,
    Removed,
}

/// A private-field observation badge bound to the stop receipt it inspected.
/// This result is not a GPU-release or admission proof and cannot be created
/// directly by callers.
#[derive(Eq, PartialEq)]
pub struct CgroupObservation {
    kind: CgroupObservationKind,
    unit_name: String,
    invocation_id: String,
    cgroup: String,
}

impl CgroupObservation {
    pub fn kind(&self) -> CgroupObservationKind {
        self.kind
    }

    /// Prevents an observation from being accidentally applied to another
    /// receipt, even if both happen to share an apparent unit name.
    pub fn matches(&self, receipt: &StoppedService) -> bool {
        self.unit_name == receipt.unit_name()
            && self.invocation_id == receipt.invocation_id()
            && self.cgroup == receipt.cgroup_path()
    }
}

impl fmt::Debug for CgroupObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CgroupObservation")
            .field("kind", &self.kind)
            .field("receipt_identity", &"<redacted>")
            .finish()
    }
}

/// Descriptor-relative, read-only observer rooted at a pinned cgroup-v2
/// filesystem. The production constructor always uses `/sys/fs/cgroup` and
/// verifies its filesystem type. Test roots are available only in unit tests.
pub struct CgroupReleaseProbe {
    root: File,
}

impl fmt::Debug for CgroupReleaseProbe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CgroupReleaseProbe(<pinned-root>)")
    }
}

impl CgroupReleaseProbe {
    /// Pin and verify the host's fixed cgroup-v2 root. No caller-provided
    /// production path is accepted.
    pub fn production() -> Result<Self, CgroupReleaseErrorKind> {
        let root = open_directory_without_symlinks(Path::new(CGROUP_ROOT))?;
        let mut filesystem = MaybeUninit::<libc::statfs>::uninit();
        if unsafe { libc::fstatfs(root.as_raw_fd(), filesystem.as_mut_ptr()) } != 0 {
            return Err(CgroupReleaseErrorKind::Io);
        }
        let filesystem = unsafe { filesystem.assume_init() };
        if filesystem.f_type != CGROUP2_SUPER_MAGIC {
            return Err(CgroupReleaseErrorKind::UnsupportedFilesystem);
        }
        Ok(Self { root })
    }

    /// Test adapter for private regular-file fixtures. The only bypass is the
    /// filesystem-magic check; absolute no-symlink directory opening and all
    /// receipt/path/file checks remain identical to production.
    #[cfg(test)]
    pub(crate) fn with_test_root(path: &Path) -> Result<Self, CgroupReleaseErrorKind> {
        let root = open_directory_without_symlinks(path)?;
        Ok(Self { root })
    }

    /// Observe the generation-bound cgroup represented by a stop receipt.
    /// `Removed` is returned only when an owned path component is absent; a
    /// missing events file or inaccessible path fails closed instead.
    pub fn observe(
        &self,
        receipt: &StoppedService,
        context: &OperationContext<'_>,
    ) -> Result<CgroupObservation, CgroupReleaseErrorKind> {
        check_context(context)?;
        let cgroup_path = receipt.cgroup_path();
        if !valid_absolute_cgroup_path(cgroup_path) {
            return Err(CgroupReleaseErrorKind::InvalidReceipt);
        }

        let Some(directory) = self.open_owned_directory(cgroup_path, context)? else {
            // This is not a generic path-existence result: the path came from
            // an opaque generation-bearing receipt after manager stop ack.
            check_context(context)?;
            return Ok(observation(receipt, CgroupObservationKind::Removed));
        };
        check_context(context)?;
        let mut events = self.open_events(&directory)?;
        check_context(context)?;
        let mut bytes = Vec::with_capacity(128);
        let mut bounded: Take<&mut File> = (&mut events).take((MAX_EVENTS_BYTES + 1) as u64);
        bounded
            .read_to_end(&mut bytes)
            .map_err(|_| CgroupReleaseErrorKind::Io)?;
        check_context(context)?;
        if bytes.len() > MAX_EVENTS_BYTES {
            return Err(CgroupReleaseErrorKind::OutputTooLarge);
        }
        match parse_events(&bytes)? {
            EventPopulation::Empty => Ok(observation(receipt, CgroupObservationKind::Empty)),
            EventPopulation::Populated => Err(CgroupReleaseErrorKind::CgroupNotEmpty),
        }
    }

    fn open_owned_directory(
        &self,
        cgroup_path: &str,
        context: &OperationContext<'_>,
    ) -> Result<Option<File>, CgroupReleaseErrorKind> {
        let relative = cgroup_path
            .strip_prefix('/')
            .ok_or(CgroupReleaseErrorKind::InvalidReceipt)?;
        let dot = CString::new(".").expect("static path has no NUL");
        let mut current = open_child_directory(self.root.as_raw_fd(), &dot)?;
        for component in relative.split('/') {
            check_context(context)?;
            let name =
                CString::new(component).map_err(|_| CgroupReleaseErrorKind::InvalidReceipt)?;
            let fd = unsafe {
                libc::openat(
                    current.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY
                        | libc::O_DIRECTORY
                        | libc::O_CLOEXEC
                        | libc::O_NOFOLLOW
                        | libc::O_NONBLOCK,
                )
            };
            if fd < 0 {
                if std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT) {
                    return Ok(None);
                }
                return Err(CgroupReleaseErrorKind::UnsafePath);
            }
            let next = unsafe { File::from_raw_fd(fd) };
            let stat = stat_fd(next.as_raw_fd()).map_err(|_| CgroupReleaseErrorKind::Io)?;
            if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
                return Err(CgroupReleaseErrorKind::UnsafePath);
            }
            current = next;
        }
        Ok(Some(current))
    }

    fn open_events(&self, directory: &File) -> Result<File, CgroupReleaseErrorKind> {
        let events_name = CString::new("cgroup.events").expect("static filename has no NUL");
        let entry = stat_at_nofollow(directory.as_raw_fd(), &events_name).map_err(|error| {
            match error.raw_os_error() {
                Some(libc::ENOENT) => CgroupReleaseErrorKind::MissingEvents,
                _ => CgroupReleaseErrorKind::Io,
            }
        })?;
        if entry.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err(CgroupReleaseErrorKind::NonRegularEvents);
        }
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                events_name.as_ptr(),
                libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(CgroupReleaseErrorKind::Io);
        }
        let file = unsafe { File::from_raw_fd(fd) };
        let opened = stat_fd(file.as_raw_fd()).map_err(|_| CgroupReleaseErrorKind::Io)?;
        if opened.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err(CgroupReleaseErrorKind::NonRegularEvents);
        }
        Ok(file)
    }
}

fn observation(receipt: &StoppedService, kind: CgroupObservationKind) -> CgroupObservation {
    CgroupObservation {
        kind,
        unit_name: receipt.unit_name().to_owned(),
        invocation_id: receipt.invocation_id().to_owned(),
        cgroup: receipt.cgroup_path().to_owned(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EventPopulation {
    Empty,
    Populated,
}

fn parse_events(bytes: &[u8]) -> Result<EventPopulation, CgroupReleaseErrorKind> {
    let text = std::str::from_utf8(bytes).map_err(|_| CgroupReleaseErrorKind::MalformedEvents)?;
    let mut populated = None;
    let mut keys = std::collections::BTreeSet::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(' ') else {
            return Err(CgroupReleaseErrorKind::MalformedEvents);
        };
        if key.is_empty()
            || value.is_empty()
            || value.contains(' ')
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
            || !value.bytes().all(|byte| byte.is_ascii_digit())
            || !keys.insert(key)
        {
            return Err(CgroupReleaseErrorKind::MalformedEvents);
        }
        if key == "populated" {
            populated = Some(match value {
                "0" => EventPopulation::Empty,
                "1" => EventPopulation::Populated,
                _ => return Err(CgroupReleaseErrorKind::MalformedEvents),
            });
        }
    }
    populated.ok_or(CgroupReleaseErrorKind::MalformedEvents)
}

fn valid_absolute_cgroup_path(path: &str) -> bool {
    path.len() <= 4096
        && path.starts_with('/')
        && path.len() > 1
        && path.split('/').skip(1).all(|component| {
            !component.is_empty()
                && component != "."
                && component != ".."
                && component
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_.:@-".contains(&byte))
        })
}

fn open_directory_without_symlinks(path: &Path) -> Result<File, CgroupReleaseErrorKind> {
    if !path.is_absolute() {
        return Err(CgroupReleaseErrorKind::UnsafePath);
    }
    let root_name = CString::new("/").expect("static path has no NUL");
    let root_fd = unsafe {
        libc::open(
            root_name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if root_fd < 0 {
        return Err(CgroupReleaseErrorKind::Io);
    }
    let mut current = unsafe { File::from_raw_fd(root_fd) };
    for component in path.components() {
        let Component::Normal(part) = component else {
            if matches!(component, Component::RootDir) {
                continue;
            }
            return Err(CgroupReleaseErrorKind::UnsafePath);
        };
        let name = CString::new(part.as_bytes()).map_err(|_| CgroupReleaseErrorKind::UnsafePath)?;
        let next = open_child_directory(current.as_raw_fd(), &name)?;
        current = next;
    }
    let stat = stat_fd(current.as_raw_fd()).map_err(|_| CgroupReleaseErrorKind::Io)?;
    if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
        return Err(CgroupReleaseErrorKind::UnsafePath);
    }
    Ok(current)
}

fn open_child_directory(parent: libc::c_int, name: &CStr) -> Result<File, CgroupReleaseErrorKind> {
    let fd = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(CgroupReleaseErrorKind::UnsafePath);
    }
    let directory = unsafe { File::from_raw_fd(fd) };
    let stat = stat_fd(directory.as_raw_fd()).map_err(|_| CgroupReleaseErrorKind::Io)?;
    if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
        return Err(CgroupReleaseErrorKind::UnsafePath);
    }
    Ok(directory)
}

fn stat_fd(fd: libc::c_int) -> std::io::Result<libc::stat> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { stat.assume_init() })
}

fn stat_at_nofollow(directory: libc::c_int, name: &CStr) -> std::io::Result<libc::stat> {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    if unsafe {
        libc::fstatat(
            directory,
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { stat.assume_init() })
}

fn check_context(context: &OperationContext<'_>) -> Result<(), CgroupReleaseErrorKind> {
    if context.is_cancelled() {
        Err(CgroupReleaseErrorKind::Cancelled)
    } else if context.remaining().is_zero() {
        Err(CgroupReleaseErrorKind::DeadlineExceeded)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::runtime::{Cancellation, NeverCancel};
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::symlink;
    use std::time::{Duration, Instant};

    const INVOCATION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    struct Fixture {
        root: tempfile::TempDir,
        receipt: StoppedService,
        cgroup_path: String,
    }

    impl Fixture {
        fn new() -> Self {
            let unit = format!("l3ms-fixture-{}.service", "a".repeat(32));
            let cgroup_path = format!("/user.slice/user@1000.service/app.slice/{unit}");
            let receipt =
                StoppedService::for_test(unit.clone(), INVOCATION.to_owned(), cgroup_path.clone())
                    .unwrap();
            Self {
                root: tempfile::tempdir().unwrap(),
                receipt,
                cgroup_path,
            }
        }

        fn probe(&self) -> CgroupReleaseProbe {
            CgroupReleaseProbe::with_test_root(self.root.path()).unwrap()
        }

        fn leaf(&self) -> std::path::PathBuf {
            self.root
                .path()
                .join(self.cgroup_path.trim_start_matches('/'))
        }

        fn events(&self, value: &str) -> std::path::PathBuf {
            let leaf = self.leaf();
            fs::create_dir_all(&leaf).unwrap();
            let events = leaf.join("cgroup.events");
            fs::write(&events, value).unwrap();
            events
        }

        fn context(&self) -> OperationContext<'static> {
            OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel)
        }

        fn observe(&self) -> Result<CgroupObservationKind, CgroupReleaseErrorKind> {
            let observation = self.probe().observe(&self.receipt, &self.context())?;
            assert!(observation.matches(&self.receipt));
            Ok(observation.kind())
        }
    }

    #[test]
    fn exact_hierarchical_zero_marker_returns_only_point_in_time_empty_observation() {
        let fixture = Fixture::new();
        fixture.events("populated 0\nfrozen 0\n");
        assert_eq!(fixture.observe(), Ok(CgroupObservationKind::Empty));
        let probe = fixture.probe();
        let observation = probe.observe(&fixture.receipt, &fixture.context()).unwrap();
        let other_receipt = StoppedService::for_test(
            fixture.receipt.unit_name().to_owned(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            fixture.cgroup_path.clone(),
        )
        .unwrap();
        assert!(!observation.matches(&other_receipt));
        assert!(!format!("{observation:?}").contains(&fixture.cgroup_path));
        assert!(!format!("{observation:?}").contains(INVOCATION));
    }

    #[test]
    fn populated_one_fails_closed() {
        let fixture = Fixture::new();
        fixture.events("populated 1\nfrozen 0\n");
        assert_eq!(
            fixture.observe(),
            Err(CgroupReleaseErrorKind::CgroupNotEmpty)
        );
    }

    #[test]
    fn missing_acknowledged_owned_cgroup_is_only_a_removal_observation() {
        let fixture = Fixture::new();
        assert_eq!(fixture.observe(), Ok(CgroupObservationKind::Removed));
    }

    #[test]
    fn missing_events_file_is_not_misreported_as_cgroup_removal() {
        let fixture = Fixture::new();
        fs::create_dir_all(fixture.leaf()).unwrap();
        assert_eq!(
            fixture.observe(),
            Err(CgroupReleaseErrorKind::MissingEvents)
        );
    }

    #[test]
    fn rejects_nonregular_events_without_opening_a_fifo() {
        let fixture = Fixture::new();
        let events = fixture.events("populated 0\n");
        fs::remove_file(&events).unwrap();
        let name = CString::new(events.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        assert_eq!(
            fixture.observe(),
            Err(CgroupReleaseErrorKind::NonRegularEvents)
        );
    }

    #[test]
    fn rejects_symlinked_cgroup_component_and_events_file() {
        let fixture = Fixture::new();
        let target = fixture.root.path().join("target");
        let target_leaf = target.join(fixture.cgroup_path.trim_start_matches('/'));
        fs::create_dir_all(&target_leaf).unwrap();
        fs::write(target_leaf.join("cgroup.events"), "populated 0\n").unwrap();
        symlink(&target, fixture.root.path().join("user.slice")).unwrap();
        assert_eq!(fixture.observe(), Err(CgroupReleaseErrorKind::UnsafePath));

        let file_fixture = Fixture::new();
        let events = file_fixture.events("populated 0\n");
        fs::remove_file(&events).unwrap();
        let outside = file_fixture.root.path().join("outside");
        fs::write(&outside, "populated 0\n").unwrap();
        symlink(outside, &events).unwrap();
        assert_eq!(
            file_fixture.observe(),
            Err(CgroupReleaseErrorKind::NonRegularEvents)
        );
    }

    #[test]
    fn malformed_or_missing_populated_marker_fails_closed() {
        for contents in [
            "frozen 0\n",
            "populated x\n",
            "populated 0 extra\n",
            "populated 0\npopulated 0\n",
            "populated 0\nfrozen\n",
        ] {
            let fixture = Fixture::new();
            fixture.events(contents);
            assert_eq!(
                fixture.observe(),
                Err(CgroupReleaseErrorKind::MalformedEvents),
                "contents={contents:?}"
            );
        }
    }

    #[test]
    fn events_output_is_bounded_and_not_reflected_in_errors() {
        let fixture = Fixture::new();
        fixture.events(&format!(
            "populated 0\nsensitive-fixture-output{}",
            "x".repeat(MAX_EVENTS_BYTES + 1)
        ));
        let error = fixture
            .probe()
            .observe(&fixture.receipt, &fixture.context())
            .unwrap_err();
        assert_eq!(error, CgroupReleaseErrorKind::OutputTooLarge);
        assert!(!format!("{error:?}").contains("sensitive-fixture-output"));
    }

    #[test]
    fn rejects_traversal_when_test_receipt_is_built() {
        let unit = format!("l3ms-fixture-{}.service", "a".repeat(32));
        let path = format!("/user.slice/../{unit}");
        assert!(StoppedService::for_test(unit, INVOCATION.to_owned(), path).is_none());
    }

    #[test]
    fn checks_cancellation_and_deadline_before_filesystem_observation() {
        struct Cancel;
        impl Cancellation for Cancel {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let fixture = Fixture::new();
        let probe = fixture.probe();
        let cancelled = OperationContext::new(Instant::now() + Duration::from_secs(2), &Cancel);
        assert_eq!(
            probe
                .observe(&fixture.receipt, &cancelled)
                .map(|o| o.kind()),
            Err(CgroupReleaseErrorKind::Cancelled)
        );
        let expired = OperationContext::new(Instant::now(), &NeverCancel);
        assert_eq!(
            probe.observe(&fixture.receipt, &expired).map(|o| o.kind()),
            Err(CgroupReleaseErrorKind::DeadlineExceeded)
        );
    }
}
