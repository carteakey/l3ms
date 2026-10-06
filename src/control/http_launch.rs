//! Explicit foreground loopback gateway; never installs or starts a coordinator.
//! Secrets come only from a bounded private configuration, not argv or defaults.

use std::fs::{self, OpenOptions};
use std::io::Read;
use std::net::SocketAddr;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::foreground_shutdown::ShutdownSignals;
use super::http_gateway::{BearerToken, HttpGateway, HttpGatewayConfig, ServeReport};
use super::runtime::OperationContext;

const MAX_CONFIG_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchError {
    ConfigUnavailable,
    InvalidConfig,
    SignalUnavailable,
    ListenerUnavailable,
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ConfigUnavailable => "HTTP gateway configuration is unavailable",
            Self::InvalidConfig => "HTTP gateway configuration is invalid",
            Self::SignalUnavailable => "HTTP gateway shutdown handler is unavailable",
            Self::ListenerUnavailable => "HTTP gateway listener is unavailable",
        })
    }
}
impl std::error::Error for LaunchError {}

// Deliberately no Debug: this includes a bearer credential.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema_version: u32,
    bind: SocketAddr,
    coordinator_socket_directory: PathBuf,
    bearer_token: String,
    request_timeout_ms: u64,
    session_attempts: usize,
    peer_attempts: usize,
    session_timeout_seconds: u64,
}

impl Config {
    fn into_gateway(self) -> Result<(HttpGatewayConfig, Duration), LaunchError> {
        if self.schema_version != 1
            || !self.coordinator_socket_directory.is_absolute()
            || !(1..=30_000).contains(&self.request_timeout_ms)
            || !(1..=3600).contains(&self.session_timeout_seconds)
        {
            return Err(LaunchError::InvalidConfig);
        }
        let token = BearerToken::new(self.bearer_token).map_err(|_| LaunchError::InvalidConfig)?;
        let gateway = HttpGatewayConfig::new(self.bind, self.coordinator_socket_directory, token)
            .and_then(|gateway| {
                gateway.with_limits(
                    Duration::from_millis(self.request_timeout_ms),
                    self.session_attempts,
                    self.peer_attempts,
                )
            })
            .map_err(|_| LaunchError::InvalidConfig)?;
        Ok((gateway, Duration::from_secs(self.session_timeout_seconds)))
    }
}

fn read_config(path: &Path) -> Result<Config, LaunchError> {
    let before = fs::symlink_metadata(path).map_err(|_| LaunchError::ConfigUnavailable)?;
    let uid = unsafe { libc::geteuid() };
    if !before.is_file()
        || before.uid() != uid
        || before.mode() & 0o7777 != 0o600
        || before.nlink() != 1
        || before.len() > MAX_CONFIG_BYTES as u64
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
        || (opened.dev(), opened.ino()) != (before.dev(), before.ino())
    {
        return Err(LaunchError::ConfigUnavailable);
    }
    let mut bytes = Vec::new();
    file.take(MAX_CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| LaunchError::ConfigUnavailable)?;
    if bytes.len() > MAX_CONFIG_BYTES {
        return Err(LaunchError::InvalidConfig);
    }
    serde_json::from_slice(&bytes).map_err(|_| LaunchError::InvalidConfig)
}

/// Dedicated foreground CLI only: takes scoped SIGINT/SIGTERM ownership.
/// Listener/session failure never retries a request or changes engine state.
pub fn run(path: &Path) -> Result<ServeReport, LaunchError> {
    let (config, timeout) = read_config(path)?.into_gateway()?;
    let signals = ShutdownSignals::install().map_err(|()| LaunchError::SignalUnavailable)?;
    let gateway = HttpGateway::bind(config).map_err(|_| LaunchError::ListenerUnavailable)?;
    let context = OperationContext::new(Instant::now() + timeout, &signals);
    Ok(gateway.serve(&context))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, PermissionsExt};

    fn config() -> serde_json::Value {
        serde_json::json!({"schema_version":1,"bind":"127.0.0.1:0",
            "coordinator_socket_directory":"/absolute/private/coordinator",
            "bearer_token":"01234567890123456789012345678901",
            "request_timeout_ms":100,"session_attempts":4,"peer_attempts":4,
            "session_timeout_seconds":1})
    }

    #[test]
    fn strict_config_rejects_public_bind_bad_limits_and_unknown_fields() {
        assert!(serde_json::from_value::<Config>(config())
            .unwrap()
            .into_gateway()
            .is_ok());
        for (field, value) in [
            ("bind", serde_json::json!("0.0.0.0:8080")),
            (
                "coordinator_socket_directory",
                serde_json::json!("relative"),
            ),
            ("bearer_token", serde_json::json!("short")),
            ("request_timeout_ms", serde_json::json!(0)),
            ("session_attempts", serde_json::json!(257)),
            ("peer_attempts", serde_json::json!(5)),
            ("session_timeout_seconds", serde_json::json!(3601)),
        ] {
            let mut value_config = config();
            value_config[field] = value;
            assert!(serde_json::from_value::<Config>(value_config)
                .unwrap()
                .into_gateway()
                .is_err());
        }
        let mut value = config();
        value["unknown"] = serde_json::json!(true);
        assert!(serde_json::from_value::<Config>(value).is_err());
    }

    #[test]
    fn private_config_refuses_symlinks_permissions_and_sensitive_errors() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("gateway.json");
        fs::write(&path, serde_json::to_vec(&config()).unwrap()).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_config(&path).is_ok());
        let alias = directory.path().join("alias.json");
        symlink(&path, &alias).unwrap();
        assert!(matches!(
            read_config(&alias),
            Err(LaunchError::ConfigUnavailable)
        ));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            read_config(&path),
            Err(LaunchError::ConfigUnavailable)
        ));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&path, b"PRIVATE_GATEWAY_CREDENTIAL").unwrap();
        let error = read_config(&path).err().unwrap();
        assert_eq!(error.to_string(), "HTTP gateway configuration is invalid");
        assert!(!format!("{error:?} {error}").contains("PRIVATE_GATEWAY_CREDENTIAL"));
    }
}
