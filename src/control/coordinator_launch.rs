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
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::catalog::{Catalog, Lane, RuntimeIdentity};
use super::daemon::{ControlDaemon, DaemonConfig, RunReport};
use super::foreground_shutdown::ShutdownSignals;
use super::runtime::{OperationContext, RecoveryPolicy, ResourceClaim};

const MAX_CONFIG_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchError {
    InvalidConfig,
    ConfigUnavailable,
    CatalogUnavailable,
    SignalUnavailable,
    CoordinatorUnavailable,
    ManagedSessionUnavailable,
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
            Self::ManagedSessionUnavailable => {
                "managed language session failed; inspect retained ownership before retrying"
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
    let config: Config = read_private_json(path)?;
    config.validate()?;
    Ok(config)
}

fn read_private_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LaunchError> {
    let bytes = read_private_json_bytes(path)?;
    serde_json::from_slice(&bytes).map_err(|_| LaunchError::InvalidConfig)
}

fn read_private_json_object<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LaunchError> {
    let bytes = read_private_json_bytes(path)?;
    if bytes
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        != Some(b'{')
    {
        return Err(LaunchError::InvalidConfig);
    }
    serde_json::from_slice(&bytes).map_err(|_| LaunchError::InvalidConfig)
}

fn read_private_json_bytes(path: &Path) -> Result<Vec<u8>, LaunchError> {
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
    Ok(bytes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedConfig {
    schema_version: u32,
    backend_contract: super::managed_session::BackendContract,
    repository_root: PathBuf,
    socket_directory: PathBuf,
    lock_directory: PathBuf,
    data_directory: PathBuf,
    standalone_data_directory: PathBuf,
    service_intent_directory: PathBuf,
    transport_directory: PathBuf,
    coordinator_id: String,
    startup_profile: String,
    supported_model_tasks: BTreeMap<String, std::collections::BTreeSet<String>>,
    #[serde(default)]
    local_job_profiles: Vec<String>,
    #[serde(default)]
    job_directory_root: Option<PathBuf>,
    #[serde(default)]
    comfy: Option<super::managed_comfy_config::ManagedComfyLaunchInput>,
    private_router_config: PathBuf,
    router_executable: PathBuf,
    router_working_directory: PathBuf,
    router_port: u16,
    gpu_uuid: String,
    gpu_index: u32,
    gpu_idle_mib: u64,
    queue_capacity: usize,
    peer_timeout_ms: u64,
    max_connections: usize,
    session_timeout_seconds: u64,
    request_timeout_seconds: u64,
    activation_timeout_seconds: u64,
    stop_timeout_seconds: u64,
    recovery_timeout_seconds: u64,
}

impl ManagedConfig {
    fn validate(&self) -> Result<(), LaunchError> {
        if self.schema_version != 1
            || !(1..=128).contains(&self.queue_capacity)
            || !(1..=30_000).contains(&self.peer_timeout_ms)
            || !(1..=4096).contains(&self.max_connections)
            || !(1..=3600).contains(&self.session_timeout_seconds)
            || !(1..=3600).contains(&self.request_timeout_seconds)
            || !(1..=500).contains(&self.activation_timeout_seconds)
            || !(1..=300).contains(&self.stop_timeout_seconds)
            || !(1..=300).contains(&self.recovery_timeout_seconds)
            || self.router_port == 0
            || self.gpu_index > 1023
            || self.coordinator_id.is_empty()
            || self.coordinator_id.len() > 32
            || !self
                .coordinator_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || [
                &self.repository_root,
                &self.socket_directory,
                &self.lock_directory,
                &self.data_directory,
                &self.standalone_data_directory,
                &self.service_intent_directory,
                &self.transport_directory,
                &self.private_router_config,
                &self.router_executable,
                &self.router_working_directory,
            ]
            .iter()
            .any(|path| !path.is_absolute())
            || self.service_intent_directory == self.transport_directory
        {
            return Err(LaunchError::InvalidConfig);
        }
        if let Some(comfy) = self.comfy.as_ref() {
            if comfy.profiles.is_empty()
                || comfy.profiles.len() > 16
                || comfy.port == 0
                || comfy.profiles.iter().any(|profile| {
                    profile.is_empty() || profile.len() > 128 || profile.trim() != profile
                })
                || [
                    &comfy.python_executable,
                    &comfy.comfy_checkout,
                    &comfy.private_user_directory_root,
                    &comfy.private_output_directory_root,
                ]
                .iter()
                .any(|path| !path.is_absolute())
                || comfy
                    .extra_model_paths_config
                    .as_ref()
                    .is_some_and(|path| !path.is_absolute())
            {
                return Err(LaunchError::InvalidConfig);
            }
        }
        Ok(())
    }
}

/// Run a strict managed language session with optional allowlisted local jobs
/// and an explicit managed Comfy workflow opt-in. This is separate from the
/// CPU-only schema; local media/decision execution requires a profile
/// allowlist and pinned private job root.
pub fn run_managed_session(config_path: &Path) -> Result<RunReport, LaunchError> {
    let config: ManagedConfig = read_private_json_object(config_path)?;
    config.validate()?;
    let repository_root = config
        .repository_root
        .canonicalize()
        .map_err(|_| LaunchError::InvalidConfig)?;
    if !repository_root.is_dir() {
        return Err(LaunchError::InvalidConfig);
    }
    let catalog = Catalog::load(&repository_root).map_err(|_| LaunchError::CatalogUnavailable)?;
    let recovery = RecoveryPolicy::new(Duration::from_secs(config.recovery_timeout_seconds))
        .map_err(|_| LaunchError::InvalidConfig)?;
    let managed = super::managed_session::ManagedSessionConfig::validate(
        super::managed_session::ManagedSessionConfigInput {
            contract: config.backend_contract,
            repository_root: repository_root.clone(),
            data_directory: config.data_directory.clone(),
            standalone_data_directory: config.standalone_data_directory.clone(),
            service_intent_directory: config.service_intent_directory,
            transport_directory: config.transport_directory,
            coordinator_id: config.coordinator_id,
            startup_profile: config.startup_profile,
            supported_model_tasks: config.supported_model_tasks,
            local_job_profiles: config.local_job_profiles,
            comfy: config.comfy,
            job_directory_root: config.job_directory_root,
            gpu_uuid: config.gpu_uuid,
            gpu_index: config.gpu_index,
            gpu_idle_mib: config.gpu_idle_mib,
            router_executable: config.router_executable,
            router_working_directory: config.router_working_directory,
            private_router_config: config.private_router_config,
            router_port: config.router_port,
            activation_timeout: Duration::from_secs(config.activation_timeout_seconds),
            request_timeout: Duration::from_secs(config.request_timeout_seconds),
            stop_timeout: Duration::from_secs(config.stop_timeout_seconds),
            recovery,
        },
        &catalog,
    )
    .map_err(|_| LaunchError::InvalidConfig)?;
    let signals = ShutdownSignals::install().map_err(|()| LaunchError::SignalUnavailable)?;
    let context = OperationContext::new(
        Instant::now() + Duration::from_secs(config.session_timeout_seconds),
        &signals,
    );
    let mut bindings = BTreeMap::new();
    bindings.insert(
        managed.runtime.clone(),
        ResourceClaim::ExclusiveGpu(config.gpu_index),
    );
    for profile in &managed.local_job_profiles {
        let route = super::routes::resolve_profile_route(&catalog, profile)
            .map_err(|_| LaunchError::InvalidConfig)?;
        let resource = match route.device {
            Some(super::catalog::Device::Cpu) => ResourceClaim::None,
            Some(super::catalog::Device::Cuda) => ResourceClaim::ExclusiveGpu(config.gpu_index),
            _ => return Err(LaunchError::InvalidConfig),
        };
        if bindings
            .insert(route.runtime, resource)
            .is_some_and(|previous| previous != resource)
        {
            return Err(LaunchError::InvalidConfig);
        }
    }
    if let Some(comfy) = managed.comfy.as_ref() {
        let runtime =
            RuntimeIdentity::new(Lane::Media, "ComfyUI").map_err(|_| LaunchError::InvalidConfig)?;
        for profile in comfy.profiles() {
            let route = super::routes::resolve_profile_route(&catalog, profile)
                .map_err(|_| LaunchError::InvalidConfig)?;
            if route.runtime != runtime || route.device != Some(super::catalog::Device::Cuda) {
                return Err(LaunchError::InvalidConfig);
            }
        }
        if bindings
            .insert(runtime, ResourceClaim::ExclusiveGpu(config.gpu_index))
            .is_some_and(|previous| previous != ResourceClaim::ExclusiveGpu(config.gpu_index))
        {
            return Err(LaunchError::InvalidConfig);
        }
    }
    let daemon = ControlDaemon::bind(
        DaemonConfig {
            socket_directory: config.socket_directory,
            lock_directory: config.lock_directory,
            data_directory: managed.data_directory.clone(),
            standalone_data_directory: managed.standalone_data_directory.clone(),
            gpus: vec![config.gpu_index],
            queue_capacity: config.queue_capacity,
            peer_timeout: Duration::from_millis(config.peer_timeout_ms),
            max_connections: config.max_connections,
        },
        catalog,
        bindings,
        &managed.repository_root,
    )
    .map_err(|_| LaunchError::CoordinatorUnavailable)?;
    daemon
        .run_managed_language(managed, &context)
        .map_err(|_| LaunchError::ManagedSessionUnavailable)
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
    let signals = ShutdownSignals::install().map_err(|()| LaunchError::SignalUnavailable)?;
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
    fn legacy_managed_config_keeps_local_jobs_disabled_unless_explicitly_declared() {
        let mut value = serde_json::json!({
            "schema_version": 1,
            "backend_contract": "llama_swap_v247_llama_server",
            "repository_root": "/repo", "socket_directory": "/socket",
            "lock_directory": "/locks", "data_directory": "/data",
            "standalone_data_directory": "/standalone",
            "service_intent_directory": "/intents", "transport_directory": "/transport",
            "coordinator_id": "test", "startup_profile": "language:test",
            "supported_model_tasks": {"test": ["chat"]},
            "private_router_config": "/router.json", "router_executable": "/router",
            "router_working_directory": "/repo", "router_port": 8080,
            "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
            "gpu_index": 0, "gpu_idle_mib": 128, "queue_capacity": 8,
            "peer_timeout_ms": 1000, "max_connections": 8,
            "session_timeout_seconds": 60, "request_timeout_seconds": 30,
            "activation_timeout_seconds": 30, "stop_timeout_seconds": 10,
            "recovery_timeout_seconds": 10
        });
        let legacy: ManagedConfig = serde_json::from_value(value.clone()).unwrap();
        assert!(legacy.local_job_profiles.is_empty());
        assert!(legacy.job_directory_root.is_none());
        assert!(legacy.comfy.is_none());
        assert_eq!(legacy.validate(), Ok(()));

        value["local_job_profiles"] = serde_json::json!(["media:breeze-tts-2"]);
        value["job_directory_root"] = serde_json::json!("/private-jobs");
        let explicit: ManagedConfig = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(explicit.local_job_profiles, ["media:breeze-tts-2"]);
        assert_eq!(
            explicit.job_directory_root,
            Some(PathBuf::from("/private-jobs"))
        );
        // Parsing is not route/path acceptance: full validation remains in
        // ManagedSessionConfig against the fresh catalog and pinned roots.
        value["local_job_profiles"] = serde_json::json!("media:breeze-tts-2");
        assert!(serde_json::from_value::<ManagedConfig>(value).is_err());
    }

    #[test]
    fn managed_comfy_is_explicit_optional_and_strictly_shaped() {
        let base = serde_json::json!({
            "schema_version": 1,
            "backend_contract": "llama_swap_v247_llama_server",
            "repository_root": "/repo", "socket_directory": "/socket",
            "lock_directory": "/locks", "data_directory": "/data",
            "standalone_data_directory": "/standalone",
            "service_intent_directory": "/intents", "transport_directory": "/transport",
            "coordinator_id": "test", "startup_profile": "language:test",
            "supported_model_tasks": {"test": ["chat"]},
            "private_router_config": "/router.json", "router_executable": "/router",
            "router_working_directory": "/repo", "router_port": 8080,
            "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
            "gpu_index": 0, "gpu_idle_mib": 128, "queue_capacity": 8,
            "peer_timeout_ms": 1000, "max_connections": 8,
            "session_timeout_seconds": 60, "request_timeout_seconds": 30,
            "activation_timeout_seconds": 30, "stop_timeout_seconds": 10,
            "recovery_timeout_seconds": 10
        });
        let comfy = serde_json::json!({
            "profiles": ["media:qwen-image-2.1"],
            "python_executable": "/repo/toolkits/l2m2/.venv-comfy/bin/python",
            "comfy_checkout": "/repo/toolkits/l2m2/vendor/ComfyUI",
            "private_user_directory_root": "/private/comfy-user",
            "private_output_directory_root": "/private/comfy-output",
            "port": 8188
        });
        let mut explicit = base.clone();
        explicit["comfy"] = comfy.clone();
        let parsed: ManagedConfig = serde_json::from_value(explicit.clone()).unwrap();
        parsed.validate().unwrap();
        assert_eq!(
            parsed.comfy.as_ref().unwrap().profiles,
            ["media:qwen-image-2.1"]
        );

        let mut missing = base.clone();
        missing["comfy"] = serde_json::json!({"profiles": ["media:qwen-image-2.1"]});
        assert!(serde_json::from_value::<ManagedConfig>(missing).is_err());

        let mut empty_profiles = base.clone();
        let mut empty = comfy.clone();
        empty["profiles"] = serde_json::json!([]);
        empty_profiles["comfy"] = empty;
        assert_eq!(
            serde_json::from_value::<ManagedConfig>(empty_profiles)
                .unwrap()
                .validate(),
            Err(LaunchError::InvalidConfig)
        );

        let mut unknown_field = comfy.clone();
        unknown_field["raw_command"] = serde_json::json!("secret command");
        let mut config = base.clone();
        config["comfy"] = unknown_field;
        assert!(serde_json::from_value::<ManagedConfig>(config).is_err());
        let duplicate_comfy =
            format!("{}", explicit).replacen("\"comfy\":", "\"comfy\":{},\"comfy\":", 1);
        assert!(serde_json::from_str::<ManagedConfig>(&duplicate_comfy).is_err());
    }

    #[test]
    fn managed_config_reader_requires_a_top_level_json_object() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("managed.json");
        fs::write(&path, br#" {"schema_version":1} "#).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let object: serde_json::Value = read_private_json_object(&path).unwrap();
        assert_eq!(object["schema_version"], 1);

        for body in [
            b"[]".as_slice(),
            b"null".as_slice(),
            b"\"object-like\"".as_slice(),
        ] {
            fs::write(&path, body).unwrap();
            assert_eq!(
                read_private_json_object::<serde_json::Value>(&path),
                Err(LaunchError::InvalidConfig)
            );
        }
    }
}
