//! Explicit foreground CPU coordinator startup. No installation, downloads,
//! directory creation, stale socket removal, or adoption of existing engines.
//!
//! Only a CLI process dedicated to this foreground operation may call `run`:
//! it temporarily owns SIGINT/SIGTERM handlers to request bounded worker
//! cleanup. Configuration and native repository ancestors must remain trusted.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::catalog::{Catalog, Lane, RuntimeIdentity};
use super::daemon::{ControlDaemon, DaemonConfig, RunReport};
use super::runtime::{Cancellation, OperationContext, RecoveryPolicy, ResourceClaim};

const MAX_CONFIG_BYTES: usize = 16 * 1024;
static SIGNAL_OWNER: Mutex<()> = Mutex::new(());
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchError {
    InvalidConfig,
    ConfigUnavailable,
    CatalogUnavailable,
    SignalUnavailable,
    CoordinatorUnavailable,
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidConfig => "CPU coordinator configuration is invalid",
            Self::ConfigUnavailable => "CPU coordinator configuration is unavailable",
            Self::CatalogUnavailable => "CPU coordinator catalog is unavailable",
            Self::SignalUnavailable => "CPU coordinator shutdown handler is unavailable",
            Self::CoordinatorUnavailable => {
                "CPU coordinator operation failed; inspect retained history before retrying"
            }
        })
    }
}
impl std::error::Error for LaunchError {}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CpuRuntime {
    Nagi,
    Laya,
}

/// Strict local configuration, deliberately without portable defaults or raw
/// commands. Native manifests remain authoritative for executable/model routes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema_version: u32,
    repository_root: PathBuf,
    socket_directory: PathBuf,
    lock_directory: PathBuf,
    data_directory: PathBuf,
    standalone_data_directory: PathBuf,
    runtimes: Vec<CpuRuntime>,
    queue_capacity: usize,
    peer_timeout_ms: u64,
    max_connections: usize,
    session_timeout_seconds: u64,
    recovery_timeout_seconds: u64,
}

impl Config {
    fn validate(&self) -> Result<(), LaunchError> {
        if self.schema_version != 1
            || !(1..=128).contains(&self.queue_capacity)
            || !(1..=30_000).contains(&self.peer_timeout_ms)
            || !(1..=4096).contains(&self.max_connections)
            || !(1..=3600).contains(&self.session_timeout_seconds)
            || !(1..=300).contains(&self.recovery_timeout_seconds)
            || self.runtimes.is_empty()
            || self.runtimes.len() > 2
            || (self.runtimes.len() == 2 && self.runtimes[0] == self.runtimes[1])
            || [
                &self.repository_root,
                &self.socket_directory,
                &self.lock_directory,
                &self.data_directory,
                &self.standalone_data_directory,
            ]
            .iter()
            .any(|path| !path.is_absolute())
        {
            return Err(LaunchError::InvalidConfig);
        }
        Ok(())
    }
}

fn read_config(path: &Path) -> Result<Config, LaunchError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| LaunchError::ConfigUnavailable)?;
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_file()
        || metadata.uid() != uid
        || metadata.mode() & 0o7777 != 0o600
        || metadata.nlink() != 1
        || metadata.len() > MAX_CONFIG_BYTES as u64
    {
        return Err(LaunchError::ConfigUnavailable);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| LaunchError::ConfigUnavailable)?;
    let opened = file
        .metadata()
        .map_err(|_| LaunchError::ConfigUnavailable)?;
    if !opened.is_file()
        || opened.uid() != uid
        || opened.mode() & 0o7777 != 0o600
        || opened.nlink() != 1
        || (opened.dev(), opened.ino()) != (metadata.dev(), metadata.ino())
    {
        return Err(LaunchError::ConfigUnavailable);
    }
    let mut bytes = Vec::new();
    file.take((MAX_CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| LaunchError::ConfigUnavailable)?;
    if bytes.len() > MAX_CONFIG_BYTES {
        return Err(LaunchError::ConfigUnavailable);
    }
    let config: Config = serde_json::from_slice(&bytes).map_err(|_| LaunchError::InvalidConfig)?;
    config.validate()?;
    Ok(config)
}

extern "C" fn request_stop(_: libc::c_int) {
    // AtomicBool is lock-free on supported Linux targets; no allocation, IO,
    // logging or locking is performed in a signal handler.
    STOP_REQUESTED.store(true, Ordering::Relaxed);
}

struct ShutdownSignals {
    _owner: MutexGuard<'static, ()>,
    interrupt: libc::sigaction,
    terminate: libc::sigaction,
}

impl ShutdownSignals {
    fn install() -> Result<Self, LaunchError> {
        let owner = SIGNAL_OWNER
            .try_lock()
            .map_err(|_| LaunchError::SignalUnavailable)?;
        STOP_REQUESTED.store(false, Ordering::Relaxed);
        // All members are initialized before use; sigemptyset supplies the
        // platform-specific empty mask representation.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut interrupt: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut terminate: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = request_stop as *const () as usize;
        unsafe {
            if libc::sigemptyset(&mut action.sa_mask) != 0
                || libc::sigaction(libc::SIGINT, &action, &mut interrupt) != 0
            {
                return Err(LaunchError::SignalUnavailable);
            }
            if libc::sigaction(libc::SIGTERM, &action, &mut terminate) != 0 {
                libc::sigaction(libc::SIGINT, &interrupt, std::ptr::null_mut());
                return Err(LaunchError::SignalUnavailable);
            }
        }
        Ok(Self {
            _owner: owner,
            interrupt,
            terminate,
        })
    }
}

impl Cancellation for ShutdownSignals {
    fn is_cancelled(&self) -> bool {
        STOP_REQUESTED.load(Ordering::Relaxed)
    }
}

impl Drop for ShutdownSignals {
    fn drop(&mut self) {
        unsafe {
            libc::sigaction(libc::SIGINT, &self.interrupt, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &self.terminate, std::ptr::null_mut());
        }
    }
}

/// Run a bounded foreground coordinator with CPU-local decision admission.
/// All directories must already exist and satisfy the daemon's private-path
/// checks. No GPU claim is acquired; CPU must be explicitly declared in native
/// profiles. SIGINT/SIGTERM request cleanup, never inference replay. A fixed
/// error can follow durable acceptance; inspect history by ID before retrying.
pub fn run(config_path: &Path) -> Result<RunReport, LaunchError> {
    let config = read_config(config_path)?;
    let catalog =
        Catalog::load(&config.repository_root).map_err(|_| LaunchError::CatalogUnavailable)?;
    let mut bindings = BTreeMap::new();
    for runtime in &config.runtimes {
        let name = match runtime {
            CpuRuntime::Nagi => "nagi",
            CpuRuntime::Laya => "laya",
        };
        let identity =
            RuntimeIdentity::new(Lane::Decisions, name).map_err(|_| LaunchError::InvalidConfig)?;
        bindings.insert(identity, ResourceClaim::None);
    }
    let signals = ShutdownSignals::install()?;
    let context = OperationContext::new(
        Instant::now() + Duration::from_secs(config.session_timeout_seconds),
        &signals,
    );
    let recovery = RecoveryPolicy::new(Duration::from_secs(config.recovery_timeout_seconds))
        .map_err(|_| LaunchError::InvalidConfig)?;
    let daemon = ControlDaemon::bind(
        DaemonConfig {
            socket_directory: config.socket_directory,
            lock_directory: config.lock_directory,
            data_directory: config.data_directory,
            standalone_data_directory: config.standalone_data_directory,
            gpus: Vec::new(),
            queue_capacity: config.queue_capacity,
            peer_timeout: Duration::from_millis(config.peer_timeout_ms),
            max_connections: config.max_connections,
        },
        catalog,
        bindings,
        &config.repository_root,
    )
    .map_err(|_| LaunchError::CoordinatorUnavailable)?;
    daemon
        .run_cpu_decisions(&context, recovery)
        .map_err(|_| LaunchError::CoordinatorUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, PermissionsExt};

    fn config_json() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "repository_root": "/repo",
            "socket_directory": "/private/socket",
            "lock_directory": "/private/locks",
            "data_directory": "/private/history",
            "standalone_data_directory": "/standalone/history",
            "runtimes": ["nagi"],
            "queue_capacity": 8,
            "peer_timeout_ms": 100,
            "max_connections": 100,
            "session_timeout_seconds": 60,
            "recovery_timeout_seconds": 5
        })
    }

    #[test]
    fn strict_configuration_accepts_only_explicit_bounded_cpu_scope() {
        let valid = config_json();
        serde_json::from_value::<Config>(valid.clone())
            .unwrap()
            .validate()
            .unwrap();
        for (field, value) in [
            ("schema_version", serde_json::json!(2)),
            ("repository_root", serde_json::json!("relative")),
            ("queue_capacity", serde_json::json!(0)),
            ("peer_timeout_ms", serde_json::json!(30_001)),
            ("max_connections", serde_json::json!(4097)),
            ("session_timeout_seconds", serde_json::json!(3601)),
            ("recovery_timeout_seconds", serde_json::json!(0)),
            ("runtimes", serde_json::json!([])),
            ("runtimes", serde_json::json!(["nagi", "nagi"])),
        ] {
            let mut changed = valid.clone();
            changed[field] = value;
            assert_eq!(
                serde_json::from_value::<Config>(changed)
                    .unwrap()
                    .validate(),
                Err(LaunchError::InvalidConfig)
            );
        }
        for (field, value) in [
            ("runtimes", serde_json::json!(["jev"])),
            ("gpus", serde_json::json!([0])),
            ("command", serde_json::json!("secret-command")),
        ] {
            let mut changed = valid.clone();
            changed[field] = value;
            assert!(serde_json::from_value::<Config>(changed).is_err());
        }
    }

    #[test]
    fn config_file_refuses_aliases_permissions_and_oversize_without_echoing_body() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.json");
        fs::write(&path, serde_json::to_vec(&config_json()).unwrap()).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_config(&path).is_ok());
        let alias = directory.path().join("alias");
        symlink(&path, &alias).unwrap();
        assert!(matches!(
            read_config(&alias),
            Err(LaunchError::ConfigUnavailable)
        ));
        fs::hard_link(&path, directory.path().join("hardlink")).unwrap();
        assert!(matches!(
            read_config(&path),
            Err(LaunchError::ConfigUnavailable)
        ));
        fs::remove_file(directory.path().join("hardlink")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            read_config(&path),
            Err(LaunchError::ConfigUnavailable)
        ));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&path, vec![b'x'; MAX_CONFIG_BYTES + 1]).unwrap();
        assert!(matches!(
            read_config(&path),
            Err(LaunchError::ConfigUnavailable)
        ));
        fs::write(&path, b"secret-config-body").unwrap();
        let error = read_config(&path).err().unwrap();
        assert!(!format!("{error:?} {error}").contains("secret-config-body"));
    }

    #[test]
    fn signal_owner_is_exclusive_and_requests_cooperative_stop() {
        let signals = ShutdownSignals::install().unwrap();
        assert!(!signals.is_cancelled());
        assert!(matches!(
            ShutdownSignals::install(),
            Err(LaunchError::SignalUnavailable)
        ));
        request_stop(libc::SIGTERM);
        assert!(signals.is_cancelled());
        drop(signals);
        let signals = ShutdownSignals::install().unwrap();
        assert!(!signals.is_cancelled());
    }
}
