use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{anyhow, Context, Result};

use crate::fs_util::{atomic_write, ensure_single_component, resolve_allow_missing, safe_stamp,
    write_unique_snapshot};

const VERSIONS_RELATIVE_PATH: &str = ".toolkit/script_versions";
const RUN_DIRECTORY: &str = "run-models";
const BENCH_DIRECTORY: &str = "bench-models";
const RUN_PREFIX: &str = "run-llama-cpp-";
const BENCH_PREFIX: &str = "bench-";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptMode {
    Run,
    Bench,
}

impl ScriptMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Bench => "bench",
        }
    }

    fn directory(self) -> &'static str {
        match self {
            Self::Run => RUN_DIRECTORY,
            Self::Bench => BENCH_DIRECTORY,
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Run => RUN_PREFIX,
            Self::Bench => BENCH_PREFIX,
        }
    }
}

impl fmt::Display for ScriptMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ScriptMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "run" => Ok(Self::Run),
            "bench" => Ok(Self::Bench),
            _ => Err(anyhow!("unsupported script mode: {value}")),
        }
    }
}

pub fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn versions_root() -> PathBuf {
    repository_root().join(VERSIONS_RELATIVE_PATH)
}

/// Build the direct, non-shell command used to execute a supported script.
/// Arguments remain separate so spaces and metacharacters are never re-parsed.
pub fn command_for_script(path: impl AsRef<Path>, extra_args: &[String]) -> Vec<String> {
    let path = path.as_ref();
    let suffix = extension_lowercase(path);
    let mut command = match suffix.as_deref() {
        Some("sh") => vec!["bash".to_owned(), path.to_string_lossy().into_owned()],
        Some("ps1") => vec![
            "pwsh".to_owned(),
            "-File".to_owned(),
            path.to_string_lossy().into_owned(),
        ],
        Some("bat" | "cmd") => vec![
            "cmd".to_owned(),
            "/c".to_owned(),
            path.to_string_lossy().into_owned(),
        ],
        _ => vec!["bash".to_owned(), path.to_string_lossy().into_owned()],
    };
    command.extend(extra_args.iter().cloned());
    command
}

/// Parse the launcher `--extra` value using POSIX shell word rules without
/// invoking a shell.
pub fn parse_extra_args(raw: &str) -> Result<Vec<String>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    shell_words::split(raw).map_err(|error| anyhow!("invalid extra arguments: {error}"))
}

/// Discover the repository's run or bench scripts in stable path order.
pub fn collect_scripts(mode: ScriptMode) -> Result<Vec<PathBuf>> {
    collect_scripts_in(repository_root(), mode)
}

/// Discover scripts below an explicit root. This is public for callers that
/// embed L3MS against a checked-out repository other than the build root.
pub fn collect_scripts_in(root: impl AsRef<Path>, mode: ScriptMode) -> Result<Vec<PathBuf>> {
    let directory = root.as_ref().join(mode.directory());
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!("failed to read script directory {}", directory.display())
            })
        }
    };

    let expected_prefix = mode.prefix();
    let mut scripts = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| {
            format!("failed to inspect script directory {}", directory.display())
        })?;
        let path = entry.path();
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if path.is_file() && file_name.starts_with(expected_prefix) && file_name.ends_with(".sh") {
            scripts.push(path);
        }
    }
    scripts.sort_unstable();
    Ok(scripts)
}

pub fn pretty_name(path: impl AsRef<Path>) -> String {
    let stem = path
        .as_ref()
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    for prefix in [
        RUN_PREFIX,
        "bench-ik-llama-cpp-",
        "bench-llama-cpp-",
        BENCH_PREFIX,
    ] {
        if let Some(name) = stem.strip_prefix(prefix) {
            return name.to_owned();
        }
    }
    stem.into_owned()
}

/// Resolve a script path and verify that it stays inside the repository and
/// has one of the supported executable-script extensions.
pub fn resolve_script(path: impl AsRef<Path>) -> Result<PathBuf> {
    resolve_script_in(path.as_ref(), &repository_root())
}

pub fn version_dir_for_script(path: impl AsRef<Path>) -> Result<PathBuf> {
    version_dir_for_script_in(path.as_ref(), &repository_root(), &versions_root())
}

pub fn list_script_versions(path: impl AsRef<Path>) -> Result<Vec<String>> {
    list_script_versions_in(path.as_ref(), &repository_root(), &versions_root())
}

pub fn load_script(path: impl AsRef<Path>) -> Result<String> {
    load_script_in(path.as_ref(), &repository_root())
}

pub fn save_script_with_version(path: impl AsRef<Path>, content: &str, note: &str) -> Result<()> {
    save_script_with_version_in(
        path.as_ref(),
        content,
        note,
        &repository_root(),
        &versions_root(),
    )
}

pub fn restore_script_version(path: impl AsRef<Path>, version_name: &str) -> Result<()> {
    restore_script_version_in(
        path.as_ref(),
        version_name,
        &repository_root(),
        &versions_root(),
    )
}

/// Resolve and contain a script against an explicit runtime repository root.
pub fn resolve_script_in(path: &Path, root: &Path) -> Result<PathBuf> {
    let root = resolve_allow_missing(root)
        .with_context(|| format!("failed to resolve repository root {}", root.display()))?;
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let target = resolve_allow_missing(&candidate)
        .with_context(|| format!("failed to resolve script path {}", path.display()))?;

    if !target.starts_with(&root) {
        return Err(anyhow!("script path must stay inside repository"));
    }
    if !is_allowed_extension(&target) {
        return Err(anyhow!("unsupported script extension"));
    }
    Ok(target)
}

/// Resolve a script's snapshot directory from explicit runtime roots.
pub fn version_dir_for_script_in(path: &Path, root: &Path, version_root: &Path) -> Result<PathBuf> {
    let root = resolve_allow_missing(root)
        .with_context(|| format!("failed to resolve repository root {}", root.display()))?;
    let target = resolve_script_in(path, &root)?;
    let relative = target
        .strip_prefix(&root)
        .map_err(|_| anyhow!("script path must stay inside repository"))?;
    Ok(version_root.join(relative))
}

/// List script versions using explicit runtime roots.
pub fn list_script_versions_in(
    path: &Path,
    root: &Path,
    version_root: &Path,
) -> Result<Vec<String>> {
    let version_dir = version_dir_for_script_in(path, root, version_root)?;
    let entries = match fs::read_dir(&version_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to read script version directory {}",
                    version_dir.display()
                )
            })
        }
    };

    let mut versions = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| {
            format!(
                "failed to inspect script version directory {}",
                version_dir.display()
            )
        })?;
        if entry.path().is_file() {
            versions.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    versions.sort_unstable_by(|left, right| right.cmp(left));
    Ok(versions)
}

/// Load a contained script using an explicit runtime repository root.
pub fn load_script_in(path: &Path, root: &Path) -> Result<String> {
    let target = resolve_script_in(path, root)?;
    if !target.is_file() {
        return Err(anyhow!("script not found"));
    }
    fs::read_to_string(&target)
        .with_context(|| format!("failed to read script {}", target.display()))
}

/// Save and snapshot a script using explicit runtime roots.
pub fn save_script_with_version_in(
    path: &Path,
    content: &str,
    note: &str,
    root: &Path,
    version_root: &Path,
) -> Result<()> {
    let target = resolve_script_in(path, root)?;

    #[cfg(unix)]
    let default_mode = Some(0o755);
    #[cfg(not(unix))]
    let default_mode = None;

    let (previous, mode) = if target.is_file() {
        let previous = fs::read_to_string(&target)
            .with_context(|| format!("failed to read existing script {}", target.display()))?;
        #[cfg(unix)]
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            Some(fs::metadata(&target)?.permissions().mode())
        };
        #[cfg(not(unix))]
        let mode = None;
        (previous, mode)
    } else {
        (String::new(), default_mode)
    };

    let version_dir = version_dir_for_script_in(&target, root, version_root)?;
    fs::create_dir_all(&version_dir).with_context(|| {
        format!(
            "failed to create script version directory {}",
            version_dir.display()
        )
    })?;
    let extension = target
        .extension()
        .map(|extension| format!(".{}", extension.to_string_lossy()))
        .unwrap_or_default();
    write_unique_snapshot(
        &version_dir,
        &safe_stamp(),
        &sanitize_note(note),
        &extension,
        previous.as_bytes(),
    )?;

    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("script path has no parent: {}", target.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create script directory {}", parent.display()))?;
    atomic_write(&target, content.as_bytes(), mode)
        .with_context(|| format!("failed to save script {}", target.display()))
}

/// Restore and snapshot a script using explicit runtime roots.
pub fn restore_script_version_in(
    path: &Path,
    version_name: &str,
    root: &Path,
    version_root: &Path,
) -> Result<()> {
    ensure_single_component(version_name)?;
    let target = resolve_script_in(path, root)?;
    let version_dir = version_dir_for_script_in(&target, root, version_root)?;
    let version_path = version_dir.join(version_name);
    if !version_path.is_file() {
        return Err(anyhow!("version not found"));
    }

    let resolved_dir = resolve_allow_missing(&version_dir)?;
    let resolved_version = resolve_allow_missing(&version_path)?;
    if !resolved_version.starts_with(&resolved_dir) {
        return Err(anyhow!("invalid version path"));
    }
    let content = fs::read_to_string(&resolved_version)
        .with_context(|| format!("failed to read script version {}", version_path.display()))?;
    save_script_with_version_in(
        &target,
        &content,
        &format!("restore-{version_name}"),
        root,
        version_root,
    )
}

fn is_allowed_extension(path: &Path) -> bool {
    matches!(
        extension_lowercase(path).as_deref(),
        Some("sh" | "ps1" | "bat" | "cmd")
    )
}

fn extension_lowercase(path: &Path) -> Option<String> {
    path.extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
}

fn sanitize_note(note: &str) -> String {
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
    let clean = if clean.is_empty() { "save" } else { clean };
    clean.chars().take(40).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn constructs_commands_without_reparsing_arguments() {
        let extra = vec!["--name".to_owned(), "two words".to_owned()];
        assert_eq!(
            command_for_script("model.sh", &extra),
            ["bash", "model.sh", "--name", "two words"]
        );
        assert_eq!(
            command_for_script("model.PS1", &extra),
            ["pwsh", "-File", "model.PS1", "--name", "two words"]
        );
        assert_eq!(
            command_for_script("model.cmd", &extra),
            ["cmd", "/c", "model.cmd", "--name", "two words"]
        );
        assert_eq!(
            command_for_script("model.unknown", &extra),
            ["bash", "model.unknown", "--name", "two words"]
        );
    }

    #[test]
    fn parses_shell_words_without_invoking_a_shell() {
        assert_eq!(
            parse_extra_args(r#"--ctx-size 32768 --name "two words""#).unwrap(),
            ["--ctx-size", "32768", "--name", "two words"]
        );
        assert!(parse_extra_args("'unterminated").is_err());
        assert!(parse_extra_args("  ").unwrap().is_empty());
    }

    #[test]
    fn inventory_matches_authoritative_entrypoints_and_pretty_names() {
        let temp = TempDir::new().unwrap();
        let run_dir = temp.path().join(RUN_DIRECTORY);
        let bench_dir = temp.path().join(BENCH_DIRECTORY);
        fs::create_dir_all(&run_dir).unwrap();
        fs::create_dir_all(&bench_dir).unwrap();
        fs::write(run_dir.join("run-llama-cpp-zeta.sh"), "").unwrap();
        fs::write(run_dir.join("run-llama-cpp-alpha.sh"), "").unwrap();
        fs::write(run_dir.join("ignore.sh"), "").unwrap();
        fs::write(bench_dir.join("bench-llama-cpp-beta.sh"), "").unwrap();
        fs::write(bench_dir.join("bench-ik-llama-cpp-gamma.sh"), "").unwrap();
        fs::write(bench_dir.join("run-llama-bench.sh"), "").unwrap();

        let run = collect_scripts_in(temp.path(), ScriptMode::Run).unwrap();
        assert_eq!(run.len(), 2);
        assert!(run[0].ends_with("run-llama-cpp-alpha.sh"));
        assert!(run[1].ends_with("run-llama-cpp-zeta.sh"));
        assert_eq!(pretty_name(&run[0]), "alpha");

        let bench = collect_scripts_in(temp.path(), ScriptMode::Bench).unwrap();
        assert_eq!(bench.len(), 2);
        assert!(bench[0].ends_with("bench-ik-llama-cpp-gamma.sh"));
        assert_eq!(pretty_name(&bench[0]), "gamma");
        assert_eq!(pretty_name(&bench[1]), "beta");
        assert_eq!(pretty_name("custom.sh"), "custom");
    }

    #[test]
    fn resolves_only_supported_scripts_inside_root() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let outside = temp.path().join("outside.sh");
        fs::create_dir_all(root.join("bench-models")).unwrap();
        fs::write(root.join("bench-models/model.sh"), "#!/bin/sh\n").unwrap();
        fs::write(root.join("notes.txt"), "notes").unwrap();
        fs::write(&outside, "#!/bin/sh\n").unwrap();

        assert!(resolve_script_in(Path::new("bench-models/model.sh"), &root).is_ok());
        assert!(resolve_script_in(Path::new("notes.txt"), &root)
            .unwrap_err()
            .to_string()
            .contains("unsupported script extension"));
        assert!(resolve_script_in(Path::new("../outside.sh"), &root)
            .unwrap_err()
            .to_string()
            .contains("inside repository"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks_that_escape_root() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let outside = temp.path().join("outside.sh");
        fs::create_dir_all(&root).unwrap();
        fs::write(&outside, "#!/bin/sh\n").unwrap();
        symlink(&outside, root.join("escape.sh")).unwrap();

        let error = resolve_script_in(&root.join("escape.sh"), &root).unwrap_err();
        assert!(error.to_string().contains("inside repository"));
    }

    #[cfg(unix)]
    #[test]
    fn new_scripts_are_executable_and_existing_modes_are_preserved() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("bench-models/model.sh");

        save_script_with_version_in(
            &path,
            "#!/bin/sh\necho first\n",
            "create",
            &root,
            &version_root,
        )
        .unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o755
        );

        fs::set_permissions(&path, fs::Permissions::from_mode(0o740)).unwrap();
        save_script_with_version_in(
            &path,
            "#!/bin/sh\necho second\n",
            "edit",
            &root,
            &version_root,
        )
        .unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o740
        );
        assert_eq!(
            load_script_in(&path, &root).unwrap(),
            "#!/bin/sh\necho second\n"
        );
    }

    #[test]
    fn save_creates_snapshots_with_sanitized_unique_names() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let version_root = temp.path().join("versions");
        let path = root.join("bench-models/model.sh");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "first").unwrap();

        let long_note = format!(" unsafe note {} ", "x".repeat(80));
        save_script_with_version_in(&path, "second", &long_note, &root, &version_root).unwrap();
        save_script_with_version_in(&path, "third", &long_note, &root, &version_root).unwrap();

        let versions = list_script_versions_in(&path, &root, &version_root).unwrap();
        assert_eq!(versions.len(), 2);
        assert_ne!(versions[0], versions[1]);
        assert!(versions.iter().all(|name| name.ends_with(".sh")));
        assert!(versions.iter().all(|name| !name.contains(' ')));

        let version_dir = version_dir_for_script_in(&path, &root, &version_root).unwrap();
        let contents: Vec<String> = versions
            .iter()
            .map(|name| fs::read_to_string(version_dir.join(name)).unwrap())
            .collect();
        assert!(contents.contains(&"first".to_owned()));
        assert!(contents.contains(&"second".to_owned()));
    }

    #[test]
    fn context_aware_api_places_script_snapshots_under_supplied_root() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("runtime-repo");
        let supplied_root = temp.path().join("runtime-state/script-versions");
        let path = root.join("bench-models/model.sh");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "previous").unwrap();

        save_script_with_version_in(&path, "current", "runtime", &root, &supplied_root).unwrap();

        let versions = list_script_versions_in(&path, &root, &supplied_root).unwrap();
        assert_eq!(versions.len(), 1);
        let version_dir = version_dir_for_script_in(&path, &root, &supplied_root).unwrap();
        assert!(version_dir.starts_with(&supplied_root));
        assert_eq!(
            fs::read_to_string(version_dir.join(&versions[0])).unwrap(),
            "previous"
        );
    }

    #[test]
    fn restore_replaces_content_and_snapshots_the_displaced_script() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let version_root = temp.path().join("versions");
        let path = root.join("bench-models/model.sh");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "first").unwrap();
        save_script_with_version_in(&path, "second", "edit", &root, &version_root).unwrap();

        let version_dir = version_dir_for_script_in(&path, &root, &version_root).unwrap();
        let first_version = list_script_versions_in(&path, &root, &version_root)
            .unwrap()
            .into_iter()
            .find(|name| fs::read_to_string(version_dir.join(name)).unwrap() == "first")
            .unwrap();
        restore_script_version_in(&path, &first_version, &root, &version_root).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "first");

        let versions = list_script_versions_in(&path, &root, &version_root).unwrap();
        assert_eq!(versions.len(), 2);
        assert!(versions
            .iter()
            .any(|name| fs::read_to_string(version_dir.join(name)).unwrap() == "second"));

        let outside = temp.path().join("outside.sh");
        fs::write(&outside, "outside").unwrap();
        let error =
            restore_script_version_in(&path, "../outside.sh", &root, &version_root).unwrap_err();
        assert!(error.to_string().contains("invalid version path"));
    }

    #[test]
    fn missing_scripts_and_versions_have_stable_behavior() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("repo");
        let version_root = temp.path().join("versions");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("missing.sh");

        assert!(load_script_in(&path, &root)
            .unwrap_err()
            .to_string()
            .contains("script not found"));
        assert!(list_script_versions_in(&path, &root, &version_root)
            .unwrap()
            .is_empty());
        assert!(
            restore_script_version_in(&path, "missing.sh", &root, &version_root)
                .unwrap_err()
                .to_string()
                .contains("version not found")
        );
    }

}
