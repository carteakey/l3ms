use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::fs_util::{atomic_write, ensure_single_component, resolve_allow_missing, safe_stamp,
    write_unique_snapshot};

const DEFAULT_CONFIG_RELATIVE_PATH: &str = "model_downloader/models_config.json";
const VERSIONS_RELATIVE_PATH: &str = ".toolkit/download_config_versions";

/// A normalized model downloader entry.
///
/// Field order intentionally matches the Python store so serialized files keep
/// the same stable, human-friendly layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub enabled: bool,
    pub repo_id: String,
    pub local_dir: String,
    pub allow_patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
    pub revision: String,
    pub force_download: bool,
    pub max_workers: Option<u64>,
    pub description: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            repo_id: String::new(),
            local_dir: String::new(),
            allow_patterns: Vec::new(),
            ignore_patterns: Vec::new(),
            revision: String::new(),
            force_download: false,
            max_workers: None,
            description: String::new(),
        }
    }
}

/// The normalized downloader configuration persisted as JSON.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DownloadConfig {
    pub base_models_dir: String,
    pub models: Vec<ModelConfig>,
}

pub fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn default_config_path() -> PathBuf {
    repository_root().join(DEFAULT_CONFIG_RELATIVE_PATH)
}

pub fn versions_root() -> PathBuf {
    repository_root().join(VERSIONS_RELATIVE_PATH)
}

/// Convert a JSON model object to the canonical persisted representation.
/// Unknown keys are ignored, strings are trimmed, empty patterns are removed,
/// and non-positive or non-integral worker counts become `None`.
pub fn normalize_model(raw: &Value) -> ModelConfig {
    let Some(model) = raw.as_object() else {
        return ModelConfig::default();
    };

    ModelConfig {
        enabled: model.get("enabled").map(json_truthy).unwrap_or(true),
        repo_id: model
            .get("repo_id")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        local_dir: model
            .get("local_dir")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        allow_patterns: normalize_patterns(model.get("allow_patterns")),
        ignore_patterns: normalize_patterns(model.get("ignore_patterns")),
        revision: model
            .get("revision")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        force_download: model
            .get("force_download")
            .map(json_truthy)
            .unwrap_or(false),
        max_workers: model
            .get("max_workers")
            .and_then(Value::as_u64)
            .filter(|workers| *workers > 0),
        description: model
            .get("description")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
    }
}

/// Normalize arbitrary JSON using the same schema and defaults as the Python
/// implementation. Non-object roots and non-array `models` values yield an
/// empty configuration rather than failing a load.
pub fn normalize_config(raw: &Value) -> DownloadConfig {
    let Some(object) = raw.as_object() else {
        return DownloadConfig::default();
    };

    let models = object
        .get("models")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| item.is_object())
                .map(normalize_model)
                .collect()
        })
        .unwrap_or_default();

    DownloadConfig {
        base_models_dir: object
            .get("base_models_dir")
            .map(python_string)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        models,
    }
}

/// Normalize an already typed config before persistence.
pub fn normalize_download_config(config: &DownloadConfig) -> DownloadConfig {
    DownloadConfig {
        base_models_dir: config.base_models_dir.trim().to_owned(),
        models: config
            .models
            .iter()
            .map(|model| ModelConfig {
                enabled: model.enabled,
                repo_id: model.repo_id.trim().to_owned(),
                local_dir: model.local_dir.trim().to_owned(),
                allow_patterns: normalize_string_patterns(&model.allow_patterns),
                ignore_patterns: normalize_string_patterns(&model.ignore_patterns),
                revision: model.revision.trim().to_owned(),
                force_download: model.force_download,
                max_workers: model.max_workers.filter(|workers| *workers > 0),
                description: model.description.trim().to_owned(),
            })
            .collect(),
    }
}

/// Validate a typed configuration and return all errors in deterministic order.
pub fn validate_config(config: &DownloadConfig) -> Vec<String> {
    let mut errors = Vec::new();

    for (index, model) in config.models.iter().enumerate() {
        if model.repo_id.trim().is_empty() {
            errors.push(format!("models[{index}].repo_id is required"));
        }
        if matches!(model.max_workers, Some(0)) {
            errors.push(format!(
                "models[{index}].max_workers must be null or positive integer"
            ));
        }
    }

    errors
}

/// Validate unnormalized JSON. This is useful at API boundaries where the
/// caller needs the Python store's structural error messages before coercion.
pub fn validate_json_config(config: &Value) -> Vec<String> {
    let Some(object) = config.as_object() else {
        return vec!["Config must be a JSON object".to_owned()];
    };

    let Some(models) = object.get("models").and_then(Value::as_array) else {
        return vec!["models must be an array".to_owned()];
    };

    let mut errors = Vec::new();
    for (index, model) in models.iter().enumerate() {
        let Some(model) = model.as_object() else {
            errors.push(format!("models[{index}] must be an object"));
            continue;
        };

        let repo_id = model.get("repo_id").map(python_string).unwrap_or_default();
        if repo_id.trim().is_empty() {
            errors.push(format!("models[{index}].repo_id is required"));
        }

        for key in ["allow_patterns", "ignore_patterns"] {
            match model.get(key) {
                None | Some(Value::Null) => {}
                Some(Value::Array(entries)) => {
                    if entries.iter().any(|entry| !entry.is_string()) {
                        errors.push(format!("models[{index}].{key} entries must be strings"));
                    }
                }
                Some(_) => errors.push(format!("models[{index}].{key} must be an array")),
            }
        }

        if let Some(workers) = model.get("max_workers") {
            let valid = match workers {
                Value::Null => true,
                // Python's bool is an int subclass: True passes this check and
                // False fails it. Keep that compatibility for raw validation.
                Value::Bool(value) => *value,
                Value::Number(value) => {
                    value.as_i64().is_some_and(|number| number > 0)
                        || value.as_u64().is_some_and(|number| number > 0)
                }
                _ => false,
            };
            if !valid {
                errors.push(format!(
                    "models[{index}].max_workers must be null or positive integer"
                ));
            }
        }
    }

    errors
}

/// Load and normalize a JSON config. Missing, unreadable, malformed, and
/// non-object files intentionally produce an empty config, matching Python.
pub fn load_config(path: impl AsRef<Path>) -> DownloadConfig {
    let Ok(contents) = fs::read_to_string(path.as_ref()) else {
        return DownloadConfig::default();
    };
    let Ok(raw) = serde_json::from_str::<Value>(&contents) else {
        return DownloadConfig::default();
    };
    normalize_config(&raw)
}

/// Load a config without the compatibility fallback used by [`load_config`].
/// I/O, JSON syntax, root-shape, and schema validation errors are returned to
/// the caller so an interactive surface can make data loss or corruption
/// visible instead of presenting an apparently empty configuration.
pub fn load_config_strict(path: impl AsRef<Path>) -> Result<DownloadConfig> {
    let path = path.as_ref();
    let contents =
        fs::read(path).with_context(|| format!("failed to read config {}", path.display()))?;
    parse_config_strict_bytes(&contents, path)
}

/// Save a normalized config and snapshot the previous bytes when the target
/// already exists.
pub fn save_config(path: impl AsRef<Path>, config: &DownloadConfig, note: &str) -> Result<()> {
    save_config_in(path.as_ref(), config, note, &versions_root())
}

/// List config snapshots newest-first by their stable filename ordering.
pub fn list_versions(path: impl AsRef<Path>) -> Result<Vec<String>> {
    list_versions_in(path.as_ref(), &versions_root())
}

/// Restore a valid config snapshot verbatim. Snapshot names must be a single
/// path component so callers cannot escape the version directory. Invalid
/// snapshot bytes are rejected before the target is changed.
pub fn restore_version(path: impl AsRef<Path>, version_name: &str) -> Result<()> {
    restore_version_in(path.as_ref(), version_name, &versions_root())
}

/// Restore a valid snapshot and return the normalized config parsed from the
/// exact bytes written to disk.
pub fn restore_version_and_load(
    path: impl AsRef<Path>,
    version_name: &str,
) -> Result<DownloadConfig> {
    restore_version_and_load_in(path.as_ref(), version_name, &versions_root())
}

pub fn csv_to_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Save using an explicit snapshot root. Runtime applications should prefer
/// this over [`save_config`] when they operate on a repository other than the
/// one used to compile the binary.
pub fn save_config_in(
    path: &Path,
    config: &DownloadConfig,
    note: &str,
    version_root: &Path,
) -> Result<()> {
    let normalized = normalize_download_config(config);
    let errors = validate_config(&normalized);
    if !errors.is_empty() {
        return Err(anyhow!(errors.join("; ")));
    }

    let target = resolve_allow_missing(path)
        .with_context(|| format!("failed to resolve config path {}", path.display()))?;
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("config path has no parent: {}", target.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create config directory {}", parent.display()))?;

    #[cfg(unix)]
    let target_mode = fs::metadata(&target).ok().map(|metadata| {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode()
    });
    #[cfg(not(unix))]
    let target_mode = None;

    if target.is_file() {
        let previous = fs::read(&target)
            .with_context(|| format!("failed to read existing config {}", target.display()))?;
        let version_dir = version_dir_for_config_in(&target, version_root)?;
        fs::create_dir_all(&version_dir).with_context(|| {
            format!(
                "failed to create config version directory {}",
                version_dir.display()
            )
        })?;
        write_unique_snapshot(
            &version_dir,
            &safe_stamp(),
            &sanitize_note(note, "save", None),
            ".json",
            &previous,
        )?;
    }

    let mut serialized = serde_json::to_vec_pretty(&normalized)
        .context("failed to serialize normalized download config")?;
    serialized.push(b'\n');
    atomic_write(&target, &serialized, target_mode)
        .with_context(|| format!("failed to save config {}", target.display()))
}

/// List versions using an explicit snapshot root.
///
/// New snapshots live in a path-hashed namespace. Names from the former
/// sanitized-path-only namespace remain visible so existing installations can
/// restore their history without a migration.
pub fn list_versions_in(path: &Path, version_root: &Path) -> Result<Vec<String>> {
    let version_dir = version_dir_for_config_in(path, version_root)?;
    let legacy_version_dir = legacy_version_dir_for_config_in(path, version_root)?;
    let mut versions = BTreeSet::new();
    collect_version_names(&version_dir, &mut versions)?;
    collect_version_names(&legacy_version_dir, &mut versions)?;
    Ok(versions.into_iter().rev().collect())
}

fn collect_version_names(directory: &Path, versions: &mut BTreeSet<String>) -> Result<()> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to read config version directory {}",
                    directory.display()
                )
            })
        }
    };

    for entry in entries {
        let entry = entry.with_context(|| {
            format!(
                "failed to inspect config version directory {}",
                directory.display()
            )
        })?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension == "json")
        {
            versions.insert(entry.file_name().to_string_lossy().into_owned());
        }
    }
    Ok(())
}

/// Restore a valid version from an explicit snapshot root.
///
/// This compatibility wrapper discards the parsed config. Interactive callers
/// should prefer [`restore_version_and_load_in`] so their in-memory state can
/// advance from the same bytes without a fallible post-restore reload.
pub fn restore_version_in(path: &Path, version_name: &str, version_root: &Path) -> Result<()> {
    restore_version_and_load_in(path, version_name, version_root)?;
    Ok(())
}

/// Validate a contained snapshot, snapshot any displaced target bytes, then
/// atomically restore the exact snapshot bytes and return their normalized
/// typed representation.
///
/// The snapshot is read once and parsed before the target is mutated. An I/O,
/// JSON, or schema error therefore leaves the target untouched and creates no
/// undo snapshot. A valid restore cannot replace an existing target unless its
/// exact displaced bytes were first durably snapshotted. After the atomic
/// replacement succeeds there are no further fallible operations, so a caller
/// can update in-memory state from the returned value without creating a
/// disk/memory split-brain window.
pub fn restore_version_and_load_in(
    path: &Path,
    version_name: &str,
    version_root: &Path,
) -> Result<DownloadConfig> {
    ensure_single_component(version_name)?;
    let target = resolve_allow_missing(path)
        .with_context(|| format!("failed to resolve config path {}", path.display()))?;
    let (version_dir, source) = version_source_for_config_in(&target, version_name, version_root)?;
    let resolved_dir = resolve_allow_missing(&version_dir)?;
    let resolved_source = resolve_allow_missing(&source)?;
    if !resolved_source.starts_with(&resolved_dir) {
        return Err(anyhow!("invalid version path"));
    }

    let content = fs::read(&resolved_source)
        .with_context(|| format!("failed to read config version {}", source.display()))?;
    let restored = parse_config_strict_bytes(&content, &resolved_source)?;

    let displaced = match fs::read(&target) {
        Ok(content) => Some(content),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read existing config {}", target.display()))
        }
    };

    #[cfg(unix)]
    let mode = if displaced.is_some() {
        use std::os::unix::fs::PermissionsExt;
        Some(
            fs::metadata(&target)
                .with_context(|| format!("failed to inspect config {}", target.display()))?
                .permissions()
                .mode(),
        )
    } else {
        None
    };
    #[cfg(not(unix))]
    let mode = None;

    if let Some(displaced) = displaced {
        let undo_dir = version_dir_for_config_in(&target, version_root)?;
        fs::create_dir_all(&undo_dir).with_context(|| {
            format!(
                "failed to create config version directory {}",
                undo_dir.display()
            )
        })?;
        let undo_note = sanitize_note(&format!("restore-{version_name}"), "restore", Some(96));
        write_unique_snapshot(&undo_dir, &safe_stamp(), &undo_note, ".json", &displaced)?;
    }

    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("config path has no parent: {}", target.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create config directory {}", parent.display()))?;

    atomic_write(&target, &content, mode)
        .with_context(|| format!("failed to restore config {}", target.display()))?;
    Ok(restored)
}

fn parse_config_strict_bytes(contents: &[u8], path: &Path) -> Result<DownloadConfig> {
    let raw: Value = serde_json::from_slice(contents)
        .with_context(|| format!("failed to parse config {}", path.display()))?;
    let errors = validate_json_config(&raw);
    if !errors.is_empty() {
        return Err(anyhow!(
            "invalid config {}: {}",
            path.display(),
            errors.join("; ")
        ));
    }
    Ok(normalize_config(&raw))
}

fn version_dir_for_config_in(path: &Path, version_root: &Path) -> Result<PathBuf> {
    let resolved = resolve_allow_missing(path)
        .with_context(|| format!("failed to resolve config path {}", path.display()))?;
    Ok(version_root.join(config_path_key(&resolved)))
}

fn legacy_version_dir_for_config_in(path: &Path, version_root: &Path) -> Result<PathBuf> {
    let resolved = resolve_allow_missing(path)
        .with_context(|| format!("failed to resolve config path {}", path.display()))?;
    Ok(version_root.join(legacy_config_path_key(&resolved)))
}

fn version_source_for_config_in(
    path: &Path,
    version_name: &str,
    version_root: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let version_dir = version_dir_for_config_in(path, version_root)?;
    let source = version_dir.join(version_name);
    if source.is_file() {
        return Ok((version_dir, source));
    }

    let legacy_version_dir = legacy_version_dir_for_config_in(path, version_root)?;
    let legacy_source = legacy_version_dir.join(version_name);
    if legacy_source.is_file() {
        Ok((legacy_version_dir, legacy_source))
    } else {
        Err(anyhow!("version not found"))
    }
}

fn config_path_key(path: &Path) -> String {
    format!(
        "{}__{:016x}",
        legacy_config_path_key(path),
        config_path_hash(path)
    )
}

fn legacy_config_path_key(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let mut sanitized = String::with_capacity(raw.len());
    let mut in_replacement = false;

    for character in raw.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '/' | '-') {
            sanitized.push(character);
            in_replacement = false;
        } else if !in_replacement {
            sanitized.push('-');
            in_replacement = true;
        }
    }

    let key = sanitized.trim_matches('/').replace('/', "__");
    if key.is_empty() {
        "models_config_json".to_owned()
    } else {
        key
    }
}

#[cfg(unix)]
fn config_path_hash(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;

    fnv1a64(path.as_os_str().as_bytes())
}

#[cfg(windows)]
fn config_path_hash(path: &Path) -> u64 {
    use std::os::windows::ffi::OsStrExt;

    let mut bytes = Vec::new();
    for unit in path.as_os_str().encode_wide() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    fnv1a64(&bytes)
}

#[cfg(not(any(unix, windows)))]
fn config_path_hash(path: &Path) -> u64 {
    fnv1a64(path.to_string_lossy().as_bytes())
}

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x00000100000001b3;

fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_continue(FNV1A64_OFFSET_BASIS, bytes)
}

fn fnv1a64_continue(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn normalize_patterns(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(patterns)) => patterns
            .iter()
            .map(python_string)
            .map(|pattern| pattern.trim().to_owned())
            .filter(|pattern| !pattern.is_empty())
            .collect(),
        // Python iterates truthy strings and dictionaries here. Preserve that
        // compatibility while handling non-iterable malformed values safely.
        Some(Value::String(pattern)) => pattern
            .chars()
            .map(|character| character.to_string())
            .filter(|character| !character.trim().is_empty())
            .collect(),
        Some(Value::Object(patterns)) => patterns
            .keys()
            .map(|pattern| pattern.trim().to_owned())
            .filter(|pattern| !pattern.is_empty())
            .collect(),
        Some(value) if !json_truthy(value) => Vec::new(),
        Some(_) => Vec::new(),
    }
}

fn normalize_string_patterns(patterns: &[String]) -> Vec<String> {
    patterns
        .iter()
        .map(|pattern| pattern.trim())
        .filter(|pattern| !pattern.is_empty())
        .map(str::to_owned)
        .collect()
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_i64().map_or_else(
            || number.as_f64().is_some_and(|value| value != 0.0),
            |value| value != 0,
        ),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn sanitize_note(note: &str, fallback: &str, max_len: Option<usize>) -> String {
    let mut clean = String::with_capacity(note.len());
    let mut in_replacement = false;
    for character in note.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
            clean.push(character);
            in_replacement = false;
        } else if !in_replacement {
            clean.push('-');
            in_replacement = true;
        }
    }

    let clean = clean.trim_matches('-');
    let clean = if clean.is_empty() { fallback } else { clean };
    match max_len {
        Some(max_len) => clean.chars().take(max_len).collect(),
        None => clean.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn normalizes_models_and_discards_unknown_fields() {
        let raw = json!({
            "enabled": 0,
            "repo_id": "  org/model  ",
            "local_dir": " models/model ",
            "allow_patterns": [" *.gguf ", "", 17],
            "ignore_patterns": null,
            "revision": " main ",
            "force_download": "yes",
            "max_workers": -2,
            "description": " demo ",
            "unknown": "discard me"
        });

        let model = normalize_model(&raw);
        assert!(!model.enabled);
        assert_eq!(model.repo_id, "org/model");
        assert_eq!(model.local_dir, "models/model");
        assert_eq!(model.allow_patterns, ["*.gguf", "17"]);
        assert!(model.ignore_patterns.is_empty());
        assert_eq!(model.revision, "main");
        assert!(model.force_download);
        assert_eq!(model.max_workers, None);
        assert_eq!(model.description, "demo");
    }

    #[test]
    fn normalizes_config_shape_and_typed_values() {
        assert_eq!(normalize_config(&json!([])), DownloadConfig::default());
        assert_eq!(
            normalize_config(&json!({"base_models_dir": " /models ", "models": {}})),
            DownloadConfig {
                base_models_dir: "/models".to_owned(),
                models: Vec::new(),
            }
        );

        let config = DownloadConfig {
            base_models_dir: " /models ".to_owned(),
            models: vec![ModelConfig {
                repo_id: " org/model ".to_owned(),
                allow_patterns: vec![" *.gguf ".to_owned(), " ".to_owned()],
                max_workers: Some(0),
                ..ModelConfig::default()
            }],
        };
        let normalized = normalize_download_config(&config);
        assert_eq!(normalized.base_models_dir, "/models");
        assert_eq!(normalized.models[0].repo_id, "org/model");
        assert_eq!(normalized.models[0].allow_patterns, ["*.gguf"]);
        assert_eq!(normalized.models[0].max_workers, None);
    }

    #[test]
    fn validates_raw_structure_with_python_compatible_messages() {
        assert_eq!(
            validate_json_config(&json!([])),
            ["Config must be a JSON object"]
        );
        assert_eq!(
            validate_json_config(&json!({})),
            ["models must be an array"]
        );

        let errors = validate_json_config(&json!({
            "models": [
                null,
                {
                    "repo_id": " ",
                    "allow_patterns": "*.gguf",
                    "ignore_patterns": ["ok", 3],
                    "max_workers": 0
                },
                {"repo_id": "valid", "max_workers": true}
            ]
        }));
        assert_eq!(
            errors,
            [
                "models[0] must be an object",
                "models[1].repo_id is required",
                "models[1].allow_patterns must be an array",
                "models[1].ignore_patterns entries must be strings",
                "models[1].max_workers must be null or positive integer",
            ]
        );
    }

    #[test]
    fn typed_validation_finds_required_repo_and_zero_workers() {
        let config = DownloadConfig {
            models: vec![ModelConfig {
                max_workers: Some(0),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };
        assert_eq!(
            validate_config(&config),
            [
                "models[0].repo_id is required",
                "models[0].max_workers must be null or positive integer",
            ]
        );
    }

    #[test]
    fn load_returns_empty_for_missing_or_malformed_files_and_normalizes_valid_json() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        assert_eq!(load_config(&path), DownloadConfig::default());

        fs::write(&path, "not json").unwrap();
        assert_eq!(load_config(&path), DownloadConfig::default());

        fs::write(
            &path,
            r#"{"base_models_dir":" /models ","models":[{"repo_id":" org/model "},2]}"#,
        )
        .unwrap();
        let loaded = load_config(&path);
        assert_eq!(loaded.base_models_dir, "/models");
        assert_eq!(loaded.models.len(), 1);
        assert_eq!(loaded.models[0].repo_id, "org/model");
    }

    #[test]
    fn strict_load_surfaces_io_parse_and_schema_errors() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");

        assert!(load_config_strict(&path)
            .unwrap_err()
            .to_string()
            .contains("failed to read config"));

        fs::write(&path, "not json").unwrap();
        assert!(load_config_strict(&path)
            .unwrap_err()
            .to_string()
            .contains("failed to parse config"));

        fs::write(&path, r#"{"models":"wrong"}"#).unwrap();
        assert!(load_config_strict(&path)
            .unwrap_err()
            .to_string()
            .contains("models must be an array"));

        fs::write(&path, r#"{"base_models_dir":" /models ","models":[]}"#).unwrap();
        assert_eq!(
            load_config_strict(&path).unwrap().base_models_dir,
            "/models"
        );
    }

    #[test]
    fn save_snapshots_previous_bytes_and_restore_is_contained() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let original =
            b"{\"base_models_dir\":\"/legacy\",\"models\":[{\"repo_id\":\"legacy/model\"}]}\n";
        fs::write(&path, original).unwrap();

        let config = DownloadConfig {
            base_models_dir: " /models ".to_owned(),
            models: vec![ModelConfig {
                repo_id: " org/model ".to_owned(),
                allow_patterns: vec![" *.gguf ".to_owned()],
                max_workers: Some(4),
                ..ModelConfig::default()
            }],
        };
        save_config_in(&path, &config, " first save! ", &version_root).unwrap();

        let saved = fs::read_to_string(&path).unwrap();
        let saved_bytes = saved.as_bytes().to_vec();
        assert!(saved.ends_with('\n'));
        let parsed: Value = serde_json::from_str(&saved).unwrap();
        assert_eq!(parsed["base_models_dir"], "/models");
        assert_eq!(parsed["models"][0]["repo_id"], "org/model");
        assert_eq!(parsed["models"][0]["allow_patterns"], json!(["*.gguf"]));

        let versions = list_versions_in(&path, &version_root).unwrap();
        assert_eq!(versions.len(), 1);
        assert!(versions[0].ends_with("__first-save.json"));
        let restored_version = versions[0].clone();
        let version_dir = version_dir_for_config_in(&path, &version_root).unwrap();
        assert_eq!(
            fs::read(version_dir.join(&restored_version)).unwrap(),
            original
        );

        let restored =
            restore_version_and_load_in(&path, &restored_version, &version_root).unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(restored.base_models_dir, "/legacy");
        assert_eq!(restored.models[0].repo_id, "legacy/model");
        let versions_after_restore = list_versions_in(&path, &version_root).unwrap();
        assert_eq!(versions_after_restore.len(), 2);
        let undo_version = versions_after_restore
            .iter()
            .find(|version| *version != &restored_version)
            .unwrap();
        assert!(undo_version.contains("__restore-"));
        assert_eq!(
            fs::read(version_dir.join(undo_version)).unwrap(),
            saved_bytes
        );

        let outside = temp.path().join("outside.json");
        fs::write(&outside, "outside").unwrap();
        let error = restore_version_in(&path, "../outside.json", &version_root).unwrap_err();
        assert!(error.to_string().contains("invalid version path"));
    }

    #[test]
    fn invalid_snapshot_is_rejected_before_target_replacement() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "not json").unwrap();

        let current = DownloadConfig {
            base_models_dir: "/models".to_owned(),
            models: vec![ModelConfig {
                repo_id: "org/current".to_owned(),
                ..ModelConfig::default()
            }],
        };
        save_config_in(&path, &current, "replace-invalid", &version_root).unwrap();
        let current_bytes = fs::read(&path).unwrap();
        let versions_before = list_versions_in(&path, &version_root).unwrap();
        let invalid_version = versions_before[0].clone();

        #[cfg(unix)]
        let mode_before = {
            use std::os::unix::fs::PermissionsExt;
            fs::metadata(&path).unwrap().permissions().mode() & 0o777
        };

        let error =
            restore_version_and_load_in(&path, &invalid_version, &version_root).unwrap_err();

        assert!(error.to_string().contains("failed to parse config"));
        assert_eq!(fs::read(&path).unwrap(), current_bytes);
        assert_eq!(load_config_strict(&path).unwrap(), current);
        assert_eq!(
            list_versions_in(&path, &version_root).unwrap(),
            versions_before,
            "an invalid source must not create an undo snapshot"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                mode_before
            );
        }
    }

    #[test]
    fn fnv1a64_matches_the_published_test_vector() {
        assert_eq!(fnv1a64(b"hello"), 0xa430_d846_80aa_bd0b);
    }

    #[test]
    fn hashed_snapshot_namespaces_separate_legacy_key_collisions() {
        let temp = TempDir::new().unwrap();
        let first = temp.path().join("collision/a/models.json");
        let second = temp.path().join("collision/a__models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        let first_bytes =
            b"{\"base_models_dir\":\"/first\",\"models\":[{\"repo_id\":\"org/first\"}]}\n";
        let second_bytes =
            b"{\"base_models_dir\":\"/second\",\"models\":[{\"repo_id\":\"org/second\"}]}\n";
        fs::write(&first, first_bytes).unwrap();
        fs::write(&second, second_bytes).unwrap();

        let first_legacy = legacy_version_dir_for_config_in(&first, &version_root).unwrap();
        let second_legacy = legacy_version_dir_for_config_in(&second, &version_root).unwrap();
        assert_eq!(first_legacy, second_legacy, "the legacy keys must collide");

        let first_version_dir = version_dir_for_config_in(&first, &version_root).unwrap();
        let second_version_dir = version_dir_for_config_in(&second, &version_root).unwrap();
        assert_ne!(first_version_dir, second_version_dir);

        let replacement = DownloadConfig {
            models: vec![ModelConfig {
                repo_id: "org/replacement".to_owned(),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };
        save_config_in(&first, &replacement, "collision", &version_root).unwrap();
        save_config_in(&second, &replacement, "collision", &version_root).unwrap();

        let first_versions = list_versions_in(&first, &version_root).unwrap();
        let second_versions = list_versions_in(&second, &version_root).unwrap();
        assert_eq!(first_versions.len(), 1);
        assert_eq!(second_versions.len(), 1);
        assert_eq!(
            fs::read(first_version_dir.join(&first_versions[0])).unwrap(),
            first_bytes
        );
        assert_eq!(
            fs::read(second_version_dir.join(&second_versions[0])).unwrap(),
            second_bytes
        );
    }

    #[test]
    fn legacy_snapshot_directory_remains_listable_and_restorable() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let current =
            b"{\"base_models_dir\":\"/current\",\"models\":[{\"repo_id\":\"org/current\"}]}\n";
        let legacy =
            b"{\"base_models_dir\":\"/legacy\",\"models\":[{\"repo_id\":\"org/legacy\"}]}\n";
        fs::write(&path, current).unwrap();

        let legacy_dir = legacy_version_dir_for_config_in(&path, &version_root).unwrap();
        fs::create_dir_all(&legacy_dir).unwrap();
        let legacy_version = "20240101T000000Z__legacy.json";
        fs::write(legacy_dir.join(legacy_version), legacy).unwrap();

        assert_eq!(
            list_versions_in(&path, &version_root).unwrap(),
            [legacy_version]
        );
        let restored = restore_version_and_load_in(&path, legacy_version, &version_root).unwrap();
        assert_eq!(fs::read(&path).unwrap(), legacy);
        assert_eq!(restored.base_models_dir, "/legacy");
        assert_eq!(restored.models[0].repo_id, "org/legacy");

        let hashed_dir = version_dir_for_config_in(&path, &version_root).unwrap();
        let hashed_versions = fs::read_dir(&hashed_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(hashed_versions.len(), 1);
        assert_eq!(fs::read(&hashed_versions[0]).unwrap(), current);
        assert_eq!(list_versions_in(&path, &version_root).unwrap().len(), 2);
    }

    #[test]
    fn restore_leaves_target_untouched_when_undo_snapshot_cannot_be_written() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("configs/models.json");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let current =
            b"{\"base_models_dir\":\"/current\",\"models\":[{\"repo_id\":\"org/current\"}]}\n";
        let previous =
            b"{\"base_models_dir\":\"/previous\",\"models\":[{\"repo_id\":\"org/previous\"}]}\n";
        fs::write(&path, current).unwrap();

        let legacy_dir = legacy_version_dir_for_config_in(&path, &version_root).unwrap();
        fs::create_dir_all(&legacy_dir).unwrap();
        let version = "20240101T000000Z__previous.json";
        fs::write(legacy_dir.join(version), previous).unwrap();

        let hashed_dir = version_dir_for_config_in(&path, &version_root).unwrap();
        fs::write(&hashed_dir, "blocks undo directory creation").unwrap();

        let error = restore_version_and_load_in(&path, version, &version_root).unwrap_err();

        assert!(error
            .to_string()
            .contains("failed to create config version directory"));
        assert_eq!(fs::read(&path).unwrap(), current);
        assert_eq!(fs::read(legacy_dir.join(version)).unwrap(), previous);
    }

    #[test]
    fn rapid_saves_allocate_unique_snapshot_names() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        let config = DownloadConfig {
            models: vec![ModelConfig {
                repo_id: "org/model".to_owned(),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };

        fs::write(&path, "one").unwrap();
        save_config_in(&path, &config, "manual", &version_root).unwrap();
        save_config_in(&path, &config, "manual", &version_root).unwrap();

        let versions = list_versions_in(&path, &version_root).unwrap();
        assert_eq!(versions.len(), 2);
        assert_ne!(versions[0], versions[1]);
    }

    #[test]
    fn context_aware_api_places_snapshots_under_supplied_root() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("runtime-repo/models.json");
        let supplied_root = temp.path().join("runtime-state/config-versions");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "previous").unwrap();
        let config = DownloadConfig {
            models: vec![ModelConfig {
                repo_id: "org/model".to_owned(),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };

        save_config_in(&path, &config, "runtime", &supplied_root).unwrap();

        let versions = list_versions_in(&path, &supplied_root).unwrap();
        assert_eq!(versions.len(), 1);
        let version_dir = version_dir_for_config_in(&path, &supplied_root).unwrap();
        assert!(version_dir.starts_with(&supplied_root));
        assert_eq!(
            fs::read_to_string(version_dir.join(&versions[0])).unwrap(),
            "previous"
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_save_preserves_existing_unix_mode() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        fs::write(&path, "old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        let config = DownloadConfig {
            models: vec![ModelConfig {
                repo_id: "org/model".to_owned(),
                ..ModelConfig::default()
            }],
            ..DownloadConfig::default()
        };

        save_config_in(&path, &config, "mode", &version_root).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_restore_preserves_existing_unix_mode() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new().unwrap();
        let path = temp.path().join("models.json");
        let version_root = temp.path().join("versions");
        let current =
            b"{\"base_models_dir\":\"/current\",\"models\":[{\"repo_id\":\"org/current\"}]}\n";
        let previous =
            b"{\"base_models_dir\":\"/previous\",\"models\":[{\"repo_id\":\"org/previous\"}]}\n";
        fs::write(&path, current).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();

        let version_dir = version_dir_for_config_in(&path, &version_root).unwrap();
        fs::create_dir_all(&version_dir).unwrap();
        let version = "20240101T000000Z__previous.json";
        fs::write(version_dir.join(version), previous).unwrap();

        restore_version_and_load_in(&path, version, &version_root).unwrap();

        assert_eq!(fs::read(&path).unwrap(), previous);
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
        let undo = list_versions_in(&path, &version_root)
            .unwrap()
            .into_iter()
            .find(|candidate| candidate != version)
            .unwrap();
        assert_eq!(fs::read(version_dir.join(undo)).unwrap(), current);
    }

    #[test]
    fn csv_parser_trims_and_discards_empty_fields() {
        assert_eq!(csv_to_list(" one, ,two ,, three "), ["one", "two", "three"]);
    }

}
