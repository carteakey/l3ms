//! Persisted operator settings and portable profile bundles.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{config_store, control::catalog::ProfileKey, script_store};

const SETTINGS_FILE: &str = "settings.json";
const PROFILE_MANIFEST: &str = "profile.json";
const PROFILE_CONFIG: &str = "models_config.json";
const PROFILE_SCRIPTS: &str = "scripts";
const MAX_SETTINGS_BYTES: usize = 256 * 1024;
const MAX_PROFILE_FILE_BYTES: usize = 2 * 1024 * 1024;
const MAX_PROFILE_SCRIPTS: usize = 1_000;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub default_port: u16,
    pub base_models_dir: String,
    pub binary_path: String,
    pub theme: String,
    #[serde(skip_serializing_if = "StackPreferences::is_empty")]
    pub stack: StackPreferences,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            default_port: 8080,
            base_models_dir: String::new(),
            binary_path: String::new(),
            theme: "default".to_owned(),
            stack: StackPreferences::default(),
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.default_port == 0 {
            errors.push("default_port must be between 1 and 65535".to_owned());
        }
        if self.base_models_dir.contains('\0') {
            errors.push("base_models_dir must not contain NUL".to_owned());
        }
        if self.binary_path.contains('\0') {
            errors.push("binary_path must not contain NUL".to_owned());
        }
        if self.theme.trim().is_empty() || self.theme.len() > 64 {
            errors.push("theme must be 1-64 bytes".to_owned());
        }
        errors.extend(self.stack.validate());
        errors
    }
}

/// Cross-toolkit preferences kept in the existing settings.json store. Empty
/// preferences are omitted, so a legacy settings file keeps its original shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StackPreferences {
    pub schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_toolkit_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decisions_toolkit_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_policy: Option<GpuPolicy>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub task_defaults: BTreeMap<String, String>,
}

impl Default for StackPreferences {
    fn default() -> Self {
        Self {
            schema_version: 1,
            media_toolkit_root: None,
            decisions_toolkit_root: None,
            artifact_root: None,
            gpu_policy: None,
            task_defaults: BTreeMap::new(),
        }
    }
}

impl StackPreferences {
    fn is_empty(&self) -> bool {
        self.schema_version == 1
            && self.media_toolkit_root.is_none()
            && self.decisions_toolkit_root.is_none()
            && self.artifact_root.is_none()
            && self.gpu_policy.is_none()
            && self.task_defaults.is_empty()
    }

    fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != 1 {
            errors.push(format!(
                "unsupported stack settings schema {} (expected 1)",
                self.schema_version
            ));
        }
        for (name, value) in [
            ("media_toolkit_root", &self.media_toolkit_root),
            ("decisions_toolkit_root", &self.decisions_toolkit_root),
            ("artifact_root", &self.artifact_root),
        ] {
            if value
                .as_ref()
                .is_some_and(|value| value.trim().is_empty() || value.contains('\0'))
            {
                errors.push(format!("{name} must be a non-empty path without NUL"));
            }
        }
        for (task, profile) in &self.task_defaults {
            if task.is_empty()
                || task.len() > 64
                || !task
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                || !valid_profile_key(profile)
            {
                errors.push(format!("invalid task default {task:?} = {profile:?}"));
            }
        }
        errors
    }
}

fn valid_profile_key(value: &str) -> bool {
    value.parse::<ProfileKey>().is_ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuPolicy {
    Manual,
    Exclusive,
}

/// Effective paths and defaults after the declared precedence is applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedStackSettings {
    pub media_toolkit_root: PathBuf,
    pub decisions_toolkit_root: PathBuf,
    pub artifact_root: Option<PathBuf>,
    pub gpu_policy: GpuPolicy,
    pub task_defaults: BTreeMap<String, String>,
}

/// CLI-only overrides; relative paths are interpreted from `invocation_dir`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StackOverrides {
    pub media_toolkit_root: Option<PathBuf>,
    pub decisions_toolkit_root: Option<PathBuf>,
    pub artifact_root: Option<PathBuf>,
    pub gpu_policy: Option<GpuPolicy>,
    pub task_defaults: BTreeMap<String, String>,
}

/// Resolve built-ins, selected host settings, user settings, supported
/// environment overrides, then explicit CLI values. Stored relative paths are
/// anchored to their settings-file directory, never the process CWD.
pub fn resolve_stack_settings(
    repo_root: &Path,
    host: Option<(&Settings, &Path)>,
    user: Option<(&Settings, &Path)>,
    environment: &BTreeMap<String, String>,
    cli: &StackOverrides,
    invocation_dir: &Path,
) -> Result<ResolvedStackSettings> {
    let mut resolved = ResolvedStackSettings {
        media_toolkit_root: repo_root.join("toolkits/l2m2"),
        decisions_toolkit_root: repo_root.join("toolkits/l3ds"),
        artifact_root: None,
        gpu_policy: GpuPolicy::Manual,
        task_defaults: BTreeMap::new(),
    };
    for (settings, declaring_dir) in [host, user].into_iter().flatten() {
        let errors = settings.validate();
        if !errors.is_empty() {
            return Err(anyhow!(errors.join("; ")));
        }
        let layer = &settings.stack;
        if let Some(root) = &layer.media_toolkit_root {
            resolved.media_toolkit_root = resolve_declared_path(declaring_dir, Path::new(root));
        }
        if let Some(root) = &layer.decisions_toolkit_root {
            resolved.decisions_toolkit_root = resolve_declared_path(declaring_dir, Path::new(root));
        }
        if let Some(path) = &layer.artifact_root {
            resolved.artifact_root = Some(resolve_declared_path(declaring_dir, Path::new(path)));
        }
        if let Some(policy) = layer.gpu_policy {
            resolved.gpu_policy = policy;
        }
        resolved.task_defaults.extend(layer.task_defaults.clone());
    }
    for (key, target) in [
        ("L3MS_MEDIA_TOOLKIT_ROOT", &mut resolved.media_toolkit_root),
        (
            "L3MS_DECISIONS_TOOLKIT_ROOT",
            &mut resolved.decisions_toolkit_root,
        ),
    ] {
        if let Some(value) = environment.get(key) {
            if value.trim().is_empty() || value.contains('\0') {
                return Err(anyhow!("{key} must be a non-empty path without NUL"));
            }
            *target = resolve_declared_path(invocation_dir, Path::new(value));
        }
    }
    if let Some(value) = environment.get("L3MS_GPU_POLICY") {
        resolved.gpu_policy = parse_gpu_policy(value)?;
    }
    for (key, target) in [("L3MS_ARTIFACT_ROOT", &mut resolved.artifact_root)] {
        if let Some(value) = environment.get(key) {
            if value.trim().is_empty() || value.contains('\0') {
                return Err(anyhow!("{key} must be a non-empty path without NUL"));
            }
            *target = Some(resolve_declared_path(invocation_dir, Path::new(value)));
        }
    }
    for (key, profile) in environment {
        if let Some(task) = key.strip_prefix("L3MS_TASK_DEFAULT_") {
            let task = task.to_ascii_lowercase();
            if task.is_empty()
                || task.len() > 64
                || !task
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                || !valid_profile_key(profile)
            {
                return Err(anyhow!("invalid {key} task default"));
            }
            resolved.task_defaults.insert(task, profile.clone());
        }
    }
    for (task, profile) in &cli.task_defaults {
        if !valid_profile_key(profile)
            || task.is_empty()
            || task.len() > 64
            || !task
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(anyhow!("invalid CLI task default {task:?} = {profile:?}"));
        }
        resolved.task_defaults.insert(task.clone(), profile.clone());
    }
    for (value, target) in [
        (&cli.media_toolkit_root, &mut resolved.media_toolkit_root),
        (
            &cli.decisions_toolkit_root,
            &mut resolved.decisions_toolkit_root,
        ),
    ] {
        if let Some(path) = value {
            if path.as_os_str().is_empty() {
                return Err(anyhow!("CLI toolkit root must not be empty"));
            }
            *target = resolve_declared_path(invocation_dir, path);
        }
    }
    if let Some(policy) = cli.gpu_policy {
        resolved.gpu_policy = policy;
    }
    for (value, target) in [(&cli.artifact_root, &mut resolved.artifact_root)] {
        if let Some(path) = value {
            if path.as_os_str().is_empty() || path.to_string_lossy().contains('\0') {
                return Err(anyhow!("CLI control path must not be empty or contain NUL"));
            }
            *target = Some(resolve_declared_path(invocation_dir, path));
        }
    }
    Ok(resolved)
}

/// Load both optional host and user layers through the existing settings
/// store, then apply the remaining overrides. No config file is written.
pub fn load_resolved_stack_settings_in(
    repo_root: &Path,
    host_data_root: Option<&Path>,
    user_data_root: &Path,
    environment: &BTreeMap<String, String>,
    cli: &StackOverrides,
    invocation_dir: &Path,
) -> Result<ResolvedStackSettings> {
    let host_settings = host_data_root.map(load_settings_in).transpose()?;
    let user_settings = load_settings_in(user_data_root)?;
    resolve_stack_settings(
        repo_root,
        host_settings.as_ref().zip(host_data_root),
        Some((&user_settings, user_data_root)),
        environment,
        cli,
        invocation_dir,
    )
}

/// Resolve the current operator's stack roots without changing settings.
pub fn load_current_stack_settings_in(
    repo_root: &Path,
    user_data_root: &Path,
) -> Result<ResolvedStackSettings> {
    let host_data_root = std::env::var_os("L3MS_HOST_SETTINGS_DIR")
        .map(|value| {
            anyhow::ensure!(
                !value.is_empty(),
                "L3MS_HOST_SETTINGS_DIR must not be empty"
            );
            Ok(PathBuf::from(value))
        })
        .transpose()?;
    let mut environment = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        let Some(key) = key.to_str() else { continue };
        if !matches!(
            key,
            "L3MS_MEDIA_TOOLKIT_ROOT"
                | "L3MS_DECISIONS_TOOLKIT_ROOT"
                | "L3MS_GPU_POLICY"
                | "L3MS_ARTIFACT_ROOT"
        ) && !key.starts_with("L3MS_TASK_DEFAULT_")
        {
            continue;
        }
        let value = value
            .into_string()
            .map_err(|_| anyhow!("{key} must be UTF-8 for stack settings resolution"))?;
        environment.insert(key.to_owned(), value);
    }
    load_resolved_stack_settings_in(
        repo_root,
        host_data_root.as_deref(),
        user_data_root,
        &environment,
        &StackOverrides::default(),
        &std::env::current_dir().context("resolve current directory")?,
    )
}

fn resolve_declared_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn parse_gpu_policy(value: &str) -> Result<GpuPolicy> {
    match value {
        "manual" => Ok(GpuPolicy::Manual),
        "exclusive" => Ok(GpuPolicy::Exclusive),
        _ => Err(anyhow!("L3MS_GPU_POLICY must be manual or exclusive")),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileManifest {
    pub schema_version: u32,
    pub settings: Settings,
    pub config_file: String,
    pub scripts_directory: String,
    pub scripts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilePreview {
    pub manifest: ProfileManifest,
    pub config: config_store::DownloadConfig,
    pub scripts: BTreeMap<String, String>,
}

pub fn settings_path(data_root: impl AsRef<Path>) -> PathBuf {
    data_root.as_ref().join(SETTINGS_FILE)
}

pub fn load_settings(data_root: impl AsRef<Path>) -> Result<Settings> {
    load_settings_in(data_root.as_ref())
}

pub fn load_settings_in(data_root: &Path) -> Result<Settings> {
    let path = settings_path(data_root);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Settings::default()),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()))
        }
    };
    if bytes.len() > MAX_SETTINGS_BYTES {
        return Err(anyhow!("settings file exceeds {MAX_SETTINGS_BYTES} bytes"));
    }
    let settings: Settings = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse settings {}", path.display()))?;
    let errors = settings.validate();
    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }
    Ok(settings)
}

pub fn save_settings(data_root: impl AsRef<Path>, settings: &Settings) -> Result<()> {
    save_settings_in(data_root.as_ref(), settings)
}

pub fn save_settings_in(data_root: &Path, settings: &Settings) -> Result<()> {
    let errors = settings.validate();
    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }
    let mut bytes = serde_json::to_vec_pretty(settings).context("serialize settings")?;
    bytes.push(b'\n');
    if bytes.len() > MAX_SETTINGS_BYTES {
        return Err(anyhow!(
            "serialized settings exceed {MAX_SETTINGS_BYTES} bytes"
        ));
    }
    let path = settings_path(data_root);
    if path.exists() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let snapshot = path.with_file_name(format!("{SETTINGS_FILE}.bak-{stamp}-{nonce}"));
        fs::copy(&path, &snapshot)
            .with_context(|| format!("snapshot settings {}", path.display()))?;
    }
    atomic_write(&path, &bytes)
}

/// Export a deterministic directory bundle. The bundle intentionally contains
/// only model configuration, script source, and non-secret settings; absolute
/// machine paths are not copied into the manifest.
pub fn export_profile(
    repo_root: impl AsRef<Path>,
    data_root: impl AsRef<Path>,
    target: impl AsRef<Path>,
) -> Result<ProfileManifest> {
    export_profile_in(repo_root.as_ref(), data_root.as_ref(), target.as_ref())
}

pub fn export_profile_in(
    repo_root: &Path,
    data_root: &Path,
    target: &Path,
) -> Result<ProfileManifest> {
    let mut settings = load_settings_in(data_root)?;
    // Profile bundles are portable: keep namespaced task choices, but do not
    // export machine-local toolkit/control paths or host GPU policy.
    settings.stack.media_toolkit_root = None;
    settings.stack.decisions_toolkit_root = None;
    settings.stack.artifact_root = None;
    settings.stack.gpu_policy = None;
    let config_path = repo_root.join("model_downloader/models_config.json");
    let config = config_store::load_config_strict(&config_path)
        .with_context(|| format!("load profile config {}", config_path.display()))?;
    let scripts_root = target.join(PROFILE_SCRIPTS);
    fs::create_dir_all(&scripts_root)
        .with_context(|| format!("create profile directory {}", scripts_root.display()))?;
    let config_bytes = serde_json::to_vec_pretty(&config).context("serialize profile config")?;
    atomic_write(&target.join(PROFILE_CONFIG), &config_bytes)?;

    let mut scripts = Vec::new();
    for mode in [
        script_store::ScriptMode::Bench,
        script_store::ScriptMode::Run,
    ] {
        for script in script_store::collect_scripts_in(repo_root, mode)? {
            let relative = script
                .strip_prefix(repo_root)
                .map_err(|_| anyhow!("script outside repository"))?;
            let relative = relative.to_string_lossy().replace('\\', "/");
            let destination = scripts_root.join(&relative);
            let parent = destination
                .parent()
                .ok_or_else(|| anyhow!("script destination has no parent"))?;
            fs::create_dir_all(parent)?;
            let content =
                fs::read(&script).with_context(|| format!("read script {}", script.display()))?;
            if content.len() > MAX_PROFILE_FILE_BYTES {
                return Err(anyhow!(
                    "script {} exceeds {MAX_PROFILE_FILE_BYTES} bytes",
                    script.display()
                ));
            }
            atomic_write(&destination, &content)?;
            scripts.push(relative);
        }
    }
    scripts.sort();
    let manifest = ProfileManifest {
        schema_version: 1,
        settings,
        config_file: PROFILE_CONFIG.to_owned(),
        scripts_directory: PROFILE_SCRIPTS.to_owned(),
        scripts,
    };
    let mut manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("serialize profile manifest")?;
    manifest_bytes.push(b'\n');
    atomic_write(&target.join(PROFILE_MANIFEST), &manifest_bytes)?;
    Ok(manifest)
}

/// Validate a profile completely before any target file is changed.
pub fn preview_profile(source: impl AsRef<Path>) -> Result<ProfilePreview> {
    preview_profile_in(source.as_ref())
}

pub fn preview_profile_in(source: &Path) -> Result<ProfilePreview> {
    let manifest_path = source.join(PROFILE_MANIFEST);
    let manifest_bytes = bounded_read(&manifest_path)?;
    let manifest: ProfileManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse profile manifest {}", manifest_path.display()))?;
    if manifest.schema_version != 1 {
        return Err(anyhow!(
            "unsupported profile schema {}",
            manifest.schema_version
        ));
    }
    let errors = manifest.settings.validate();
    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }
    if manifest.scripts.len() > MAX_PROFILE_SCRIPTS {
        return Err(anyhow!("profile contains too many scripts"));
    }
    let config_path = source.join(&manifest.config_file);
    ensure_relative_file(&manifest.config_file)?;
    let config = config_store::load_config_strict(&config_path)?;
    let mut scripts = BTreeMap::new();
    for relative in &manifest.scripts {
        ensure_relative_file(relative)?;
        if !relative.starts_with("bench-models/") && !relative.starts_with("run-models/") {
            return Err(anyhow!(
                "profile script must be under bench-models or run-models: {relative}"
            ));
        }
        let path = source.join(&manifest.scripts_directory).join(relative);
        let bytes = bounded_read(&path)?;
        let content =
            String::from_utf8(bytes).with_context(|| format!("script {relative} is not UTF-8"))?;
        scripts.insert(relative.clone(), content);
    }
    Ok(ProfilePreview {
        manifest,
        config,
        scripts,
    })
}

/// Apply a previously validated profile. Every changed config/script uses the
/// existing snapshot-aware stores; callers should show [`preview_profile`]
/// before invoking this mutating operation.
pub fn import_profile(
    repo_root: impl AsRef<Path>,
    data_root: impl AsRef<Path>,
    source: impl AsRef<Path>,
) -> Result<ProfileManifest> {
    import_profile_in(repo_root.as_ref(), data_root.as_ref(), source.as_ref())
}

pub fn import_profile_in(
    repo_root: &Path,
    data_root: &Path,
    source: &Path,
) -> Result<ProfileManifest> {
    let preview = preview_profile_in(source)?;
    save_settings_in(data_root, &preview.manifest.settings)?;
    let config_path = repo_root.join("model_downloader/models_config.json");
    let version_root = repo_root.join(".toolkit/download_config_versions");
    config_store::save_config_in(
        &config_path,
        &preview.config,
        "profile-import",
        &version_root,
    )?;
    let script_versions = repo_root.join(".toolkit/script_versions");
    for (relative, content) in &preview.scripts {
        let destination = repo_root.join(relative);
        script_store::save_script_with_version_in(
            &destination,
            content,
            "profile-import",
            repo_root,
            &script_versions,
        )?;
    }
    Ok(preview.manifest)
}

fn ensure_relative_file(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(anyhow!(
            "profile path must be a safe relative file: {value:?}"
        ));
    }
    Ok(())
}

fn bounded_read(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path).with_context(|| format!("read profile file {}", path.display()))?;
    if bytes.len() > MAX_PROFILE_FILE_BYTES {
        return Err(anyhow!(
            "profile file {} exceeds {MAX_PROFILE_FILE_BYTES} bytes",
            path.display()
        ));
    }
    Ok(bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("target has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("profile"),
        nonce
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temp, path).with_context(|| format!("replace {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn settings_default_and_validation_are_strict() {
        assert_eq!(Settings::default().default_port, 8080);
        let settings = Settings {
            default_port: 0,
            ..Settings::default()
        };
        assert!(!settings.validate().is_empty());
    }

    #[test]
    fn settings_round_trip_atomically() {
        let temp = TempDir::new().unwrap();
        let settings = Settings {
            default_port: 9000,
            base_models_dir: "models".into(),
            binary_path: "llama-server".into(),
            theme: "dark".into(),
            stack: StackPreferences::default(),
        };
        save_settings_in(temp.path(), &settings).unwrap();
        assert_eq!(load_settings_in(temp.path()).unwrap(), settings);
    }

    #[test]
    fn legacy_settings_keep_their_shape_and_gain_empty_stack_defaults() {
        let temp = TempDir::new().unwrap();
        fs::write(
            settings_path(temp.path()),
            r#"{"default_port":8001,"base_models_dir":"","binary_path":"","theme":"default"}"#,
        )
        .unwrap();
        let settings = load_settings_in(temp.path()).unwrap();
        assert!(settings.stack.is_empty());
        save_settings_in(temp.path(), &settings).unwrap();
        let saved = fs::read_to_string(settings_path(temp.path())).unwrap();
        assert!(!saved.contains("\"stack\""));
        let snapshots = fs::read_dir(temp.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("settings.json.bak-")
            })
            .collect::<Vec<_>>();
        assert_eq!(snapshots.len(), 1);
        assert!(fs::read_to_string(snapshots[0].path())
            .unwrap()
            .contains("\"default_port\":8001"));
    }

    #[test]
    fn stack_resolution_obeys_precedence_and_declaring_paths() {
        let temp = TempDir::new().unwrap();
        let repo = temp.path().join("repo");
        let host_dir = temp.path().join("host");
        let user_dir = temp.path().join("user");
        let invocation_dir = temp.path().join("some-other-working-directory");
        let mut host = Settings::default();
        host.stack.media_toolkit_root = Some("host-media".into());
        host.stack.gpu_policy = Some(GpuPolicy::Exclusive);
        host.stack
            .task_defaults
            .insert("tts".into(), "media:host-voice".into());
        let mut user = Settings::default();
        user.stack.decisions_toolkit_root = Some("user-decisions".into());
        user.stack
            .task_defaults
            .insert("tts".into(), "media:user-voice".into());
        let environment = BTreeMap::from([
            ("L3MS_MEDIA_TOOLKIT_ROOT".into(), "env-media".into()),
            ("L3MS_TASK_DEFAULT_TTS".into(), "media:env-voice".into()),
        ]);
        let cli = StackOverrides {
            media_toolkit_root: Some(PathBuf::from("cli-media")),
            task_defaults: BTreeMap::from([("tts".into(), "media:cli-voice".into())]),
            ..StackOverrides::default()
        };
        let resolved = resolve_stack_settings(
            &repo,
            Some((&host, &host_dir)),
            Some((&user, &user_dir)),
            &environment,
            &cli,
            &invocation_dir,
        )
        .unwrap();
        assert_eq!(
            resolved.media_toolkit_root,
            invocation_dir.join("cli-media")
        );
        assert_eq!(
            resolved.decisions_toolkit_root,
            user_dir.join("user-decisions")
        );
        assert_eq!(resolved.gpu_policy, GpuPolicy::Exclusive);
        assert_eq!(resolved.task_defaults["tts"], "media:cli-voice");
    }

    #[test]
    fn control_paths_are_opt_in_and_share_layered_resolution_without_creating_directories() {
        let temp = TempDir::new().unwrap();
        let repo = temp.path().join("repo");
        let host_dir = temp.path().join("host");
        let user_dir = temp.path().join("user");
        let invocation = temp.path().join("invocation");
        let empty = resolve_stack_settings(
            &repo,
            None,
            None,
            &BTreeMap::new(),
            &StackOverrides::default(),
            &invocation,
        )
        .unwrap();
        assert!(empty.artifact_root.is_none());
        let mut host = Settings::default();
        host.stack.artifact_root = Some("host-artifacts".into());
        let mut user = Settings::default();
        user.stack.artifact_root = Some("user-artifacts".into());
        let layered = resolve_stack_settings(
            &repo,
            Some((&host, &host_dir)),
            Some((&user, &user_dir)),
            &BTreeMap::new(),
            &StackOverrides::default(),
            &invocation,
        )
        .unwrap();
        assert_eq!(layered.artifact_root, Some(user_dir.join("user-artifacts")));
        let environment = BTreeMap::from([("L3MS_ARTIFACT_ROOT".into(), "env-artifacts".into())]);
        let cli = StackOverrides::default();
        let overridden = resolve_stack_settings(
            &repo,
            Some((&host, &host_dir)),
            Some((&user, &user_dir)),
            &environment,
            &cli,
            &invocation,
        )
        .unwrap();
        assert_eq!(
            overridden.artifact_root,
            Some(invocation.join("env-artifacts"))
        );
        assert!(!host_dir.exists() && !user_dir.exists() && !invocation.exists());
    }

    #[test]
    fn invalid_control_paths_fail_before_settings_writes_or_resolution() {
        let temp = TempDir::new().unwrap();
        for invalid in ["", "\0", "  "] {
            let mut settings = Settings::default();
            settings.stack.artifact_root = Some(invalid.into());
            assert!(save_settings_in(temp.path(), &settings).is_err());
            assert!(resolve_stack_settings(
                temp.path(),
                None,
                None,
                &BTreeMap::from([("L3MS_ARTIFACT_ROOT".into(), invalid.into())]),
                &StackOverrides::default(),
                temp.path()
            )
            .is_err());
        }
        assert!(!settings_path(temp.path()).exists());
    }

    #[test]
    fn current_control_settings_read_environment_in_an_isolated_process() {
        const MARKER: &str = "L3MS_TEST_CONTROL_PATH_ROOT";
        if let Some(root) = std::env::var_os(MARKER) {
            let root = PathBuf::from(root);
            let resolved = load_current_stack_settings_in(&root, &root.join("user")).unwrap();
            assert_eq!(resolved.artifact_root, Some(root.join("artifacts")));
            assert!(!root.join("user").exists());
            return;
        }
        let temp = TempDir::new().unwrap();
        let output = {
            let _fixture_guard = crate::control::process::executable_fixture_guard();
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "settings::tests::current_control_settings_read_environment_in_an_isolated_process",
                ])
                .env(MARKER, temp.path())
                .env("L3MS_ARTIFACT_ROOT", temp.path().join("artifacts"))
                .env_remove("L3MS_HOST_SETTINGS_DIR")
                .output()
                .unwrap()
        };
        assert!(
            output.status.success(),
            "isolated settings test failed: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    #[test]
    fn stack_settings_reject_unknown_schema_and_bad_defaults() {
        let mut settings = Settings::default();
        settings.stack.schema_version = 2;
        settings
            .stack
            .task_defaults
            .insert("tts".into(), "unknown:voice".into());
        assert_eq!(settings.validate().len(), 2);
        let temp = TempDir::new().unwrap();
        assert!(save_settings_in(temp.path(), &settings).is_err());
        assert!(!settings_path(temp.path()).exists());
    }

    #[test]
    fn stack_loader_reads_existing_host_and_user_settings_stores() {
        let temp = TempDir::new().unwrap();
        let host_dir = temp.path().join("host");
        let user_dir = temp.path().join("user");
        let mut host = Settings::default();
        host.stack.media_toolkit_root = Some("media".into());
        save_settings_in(&host_dir, &host).unwrap();
        let mut user = Settings::default();
        user.stack.decisions_toolkit_root = Some("decisions".into());
        save_settings_in(&user_dir, &user).unwrap();
        let resolved = load_resolved_stack_settings_in(
            &temp.path().join("repo"),
            Some(&host_dir),
            &user_dir,
            &BTreeMap::new(),
            &StackOverrides::default(),
            temp.path(),
        )
        .unwrap();
        assert_eq!(resolved.media_toolkit_root, host_dir.join("media"));
        assert_eq!(resolved.decisions_toolkit_root, user_dir.join("decisions"));
    }

    #[test]
    fn profile_preview_rejects_traversal() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join(PROFILE_MANIFEST),
            serde_json::json!({
                "schema_version": 1,
                "settings": Settings::default(),
                "config_file": "models_config.json",
                "scripts_directory": "scripts",
                "scripts": ["../escape.sh"]
            })
            .to_string(),
        )
        .unwrap();
        assert!(preview_profile_in(temp.path()).is_err());
    }

    #[test]
    fn profile_export_and_import_round_trip_config_scripts_and_settings() {
        let repository = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        fs::create_dir_all(repository.path().join("model_downloader")).unwrap();
        fs::create_dir_all(repository.path().join("bench-models")).unwrap();
        fs::create_dir_all(repository.path().join("run-models")).unwrap();
        fs::write(
            repository.path().join("model_downloader/models_config.json"),
            r#"{"base_models_dir":"models","models":[{"repo_id":"org/model","description":"demo"}]}"#,
        )
        .unwrap();
        fs::write(
            repository.path().join("bench-models/bench-demo.sh"),
            "#!/bin/sh\necho bench\n",
        )
        .unwrap();
        fs::write(
            repository.path().join("run-models/run-llama-cpp-demo.sh"),
            "#!/bin/sh\necho run\n",
        )
        .unwrap();
        let settings = Settings {
            default_port: 9001,
            base_models_dir: "models".into(),
            binary_path: "llama-server".into(),
            theme: "dark".into(),
            stack: StackPreferences::default(),
        };
        save_settings_in(data.path(), &settings).unwrap();

        let bundle = repository.path().join("bundle");
        let manifest = export_profile_in(repository.path(), data.path(), &bundle).unwrap();
        assert_eq!(
            manifest.scripts,
            [
                "bench-models/bench-demo.sh",
                "run-models/run-llama-cpp-demo.sh"
            ]
        );
        assert_eq!(
            preview_profile_in(&bundle).unwrap().manifest.settings,
            settings
        );

        fs::write(
            repository
                .path()
                .join("model_downloader/models_config.json"),
            r#"{"base_models_dir":"changed","models":[]}"#,
        )
        .unwrap();
        fs::write(
            repository.path().join("bench-models/bench-demo.sh"),
            "#!/bin/sh\necho changed\n",
        )
        .unwrap();
        import_profile_in(repository.path(), data.path(), &bundle).unwrap();
        assert_eq!(load_settings_in(data.path()).unwrap(), settings);
        assert!(
            fs::read_to_string(repository.path().join("bench-models/bench-demo.sh"))
                .unwrap()
                .contains("echo bench")
        );
        assert!(fs::read_to_string(
            repository
                .path()
                .join("model_downloader/models_config.json")
        )
        .unwrap()
        .contains("org/model"));
    }

    #[test]
    fn profile_export_strips_machine_local_stack_roots() {
        let repository = TempDir::new().unwrap();
        let data = TempDir::new().unwrap();
        fs::create_dir_all(repository.path().join("model_downloader")).unwrap();
        fs::create_dir_all(repository.path().join("bench-models")).unwrap();
        fs::create_dir_all(repository.path().join("run-models")).unwrap();
        fs::write(
            repository.path().join("model_downloader/models_config.json"),
            r#"{"base_models_dir":"models","models":[{"repo_id":"org/model","description":"demo"}]}"#,
        )
        .unwrap();
        let mut settings = Settings::default();
        settings.stack.media_toolkit_root = Some("/private/media".into());
        settings.stack.decisions_toolkit_root = Some("/private/decisions".into());
        settings.stack.artifact_root = Some("/private/artifacts".into());
        settings.stack.gpu_policy = Some(GpuPolicy::Exclusive);
        settings
            .stack
            .task_defaults
            .insert("tts".into(), "media:breeze-tts-2".into());
        save_settings_in(data.path(), &settings).unwrap();

        let bundle = repository.path().join("bundle");
        let exported = export_profile_in(repository.path(), data.path(), &bundle).unwrap();
        assert_eq!(exported.settings.stack.media_toolkit_root, None);
        assert_eq!(exported.settings.stack.decisions_toolkit_root, None);
        assert_eq!(exported.settings.stack.artifact_root, None);
        assert_eq!(exported.settings.stack.gpu_policy, None);
        assert_eq!(
            exported.settings.stack.task_defaults["tts"],
            "media:breeze-tts-2"
        );
        assert!(!fs::read_to_string(bundle.join("profile.json"))
            .unwrap()
            .contains("/private/"));
        assert_eq!(load_settings_in(data.path()).unwrap(), settings);
    }
}
