//! Persisted operator settings and portable profile bundles.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{config_store, script_store};

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
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            default_port: 8080,
            base_models_dir: String::new(),
            binary_path: String::new(),
            theme: "default".to_owned(),
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
        errors
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
    let settings = load_settings_in(data_root)?;
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
        };
        save_settings_in(temp.path(), &settings).unwrap();
        assert_eq!(load_settings_in(temp.path()).unwrap(), settings);
    }

    #[test]
    fn legacy_settings_with_a_stack_key_still_load_and_drop_it_on_save() {
        let temp = TempDir::new().unwrap();
        fs::write(
            settings_path(temp.path()),
            r#"{"default_port":8001,"base_models_dir":"","binary_path":"","theme":"default","stack":{"media_toolkit_root":"/old/media","gpu_policy":"exclusive"}}"#,
        )
        .unwrap();
        let settings = load_settings_in(temp.path()).unwrap();
        assert_eq!(settings.default_port, 8001);
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
            .count();
        assert_eq!(snapshots, 1);
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
}
