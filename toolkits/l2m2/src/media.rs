//! Declarative media-generation profiles and their safe CLI boundary.
//!
//! The actual runtimes remain editable scripts.  This module owns discovery,
//! filtering, and process invocation so prompts and input paths are passed as
//! argv values rather than reparsed by a shell.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

const MANIFEST_FILE: &str = "media-runtimes.json";
const MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct MediaManifest {
    pub schema_version: u32,
    pub profiles: Vec<MediaProfile>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct MediaProfile {
    pub id: String,
    pub name: String,
    pub runtime: String,
    pub variant: String,
    pub tasks: Vec<String>,
    pub inputs: Vec<String>,
    pub status: String,
    pub script: String,
    pub description: String,
    #[serde(default)]
    pub requirements: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct CommandSpec {
    program: OsString,
    args: Vec<OsString>,
}

pub fn load_manifest(root: &Path) -> Result<MediaManifest> {
    let path = root.join(MANIFEST_FILE);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let manifest: MediaManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid media manifest: {}", path.display()))?;

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        bail!(
            "unsupported media manifest schema {} (expected {})",
            manifest.schema_version,
            MANIFEST_SCHEMA_VERSION
        );
    }
    if manifest.profiles.is_empty() {
        bail!("media manifest contains no profiles: {}", path.display());
    }

    for profile in &manifest.profiles {
        if profile.id.trim().is_empty() || profile.name.trim().is_empty() {
            bail!("media profiles must have non-empty id and name");
        }
        if profile.script.trim().is_empty() {
            bail!("media profile {:?} has no script", profile.id);
        }
        let script = root.join(&profile.script);
        if !script.is_file() {
            bail!(
                "media profile {:?} script does not exist: {}",
                profile.id,
                script.display()
            );
        }
    }

    Ok(manifest)
}

pub fn filter_profiles(profiles: Vec<MediaProfile>, filter: &str) -> Vec<MediaProfile> {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return profiles;
    }

    profiles
        .into_iter()
        .filter(|profile| {
            [
                profile.id.as_str(),
                profile.name.as_str(),
                profile.runtime.as_str(),
                profile.variant.as_str(),
                profile.status.as_str(),
                profile.description.as_str(),
            ]
            .iter()
            .any(|value| value.to_ascii_lowercase().contains(&filter))
                || profile
                    .tasks
                    .iter()
                    .chain(profile.inputs.iter())
                    .any(|value| value.to_ascii_lowercase().contains(&filter))
        })
        .collect()
}

pub fn print_profile_list(profiles: &[MediaProfile]) {
    if profiles.is_empty() {
        println!("No media-generation profiles found");
        return;
    }

    println!("MEDIA profiles ({}):", profiles.len());
    for (index, profile) in profiles.iter().enumerate() {
        println!(
            "  {:>2}. {:<24} {:<16} {:<12} {}",
            index + 1,
            profile.id,
            profile.runtime,
            profile.status,
            profile.name
        );
        println!(
            "      tasks={} inputs={} variant={}",
            profile.tasks.join(","),
            profile.inputs.join(","),
            profile.variant
        );
    }
}

pub fn interactive_media(root: &Path, filter: &str, extra: &str) -> Result<u8> {
    let manifest = load_manifest(root)?;
    let profiles = filter_profiles(manifest.profiles, filter);
    if profiles.is_empty() {
        eprintln!("No media-generation profiles found for filter: {filter:?}");
        return Ok(1);
    }

    print_profile_list(&profiles);
    let index = if profiles.len() == 1 && !extra.trim().is_empty() {
        println!("Auto-selecting the only matching media profile for --extra.");
        0
    } else {
        let Some(index) = choose_index(profiles.len(), "media profile")? else {
            println!("Cancelled.");
            return Ok(0);
        };
        index
    };

    let profile = &profiles[index];
    let extra_args = parse_extra_args(extra)?;
    let script = safe_script_path(root, &profile.script)?;
    let command = command_for_script(&script, &extra_args);
    println!("$ {}", format_command(&command));

    let status = Command::new(&command.program)
        .args(&command.args)
        .current_dir(root)
        .status()
        .with_context(|| format!("failed to execute media profile {:?}", profile.id))?;
    let code = status.code().unwrap_or(1);
    println!("Exited with code {code}");
    Ok(u8::try_from(code).unwrap_or(1))
}

fn safe_script_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let root =
        fs::canonicalize(root).with_context(|| format!("invalid L2M2 root: {}", root.display()))?;
    let script = fs::canonicalize(root.join(relative))
        .with_context(|| format!("invalid media script path: {relative}"))?;
    if !script.starts_with(&root) {
        bail!("media script escapes the repository root: {relative}");
    }
    Ok(script)
}

fn choose_index(count: usize, item_name: &str) -> Result<Option<usize>> {
    println!("Select {item_name} index, or 'q' to quit.");
    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("> ");
        io::stdout().flush().context("failed to flush prompt")?;
        input.clear();
        let bytes = stdin
            .read_line(&mut input)
            .context("failed to read selection")?;
        if bytes == 0 {
            return Ok(None);
        }

        let value = input.trim();
        if value.is_empty() && count == 1 {
            return Ok(Some(0));
        }
        if matches!(value.to_ascii_lowercase().as_str(), "q" | "quit" | "exit") {
            return Ok(None);
        }
        if let Ok(index) = value.parse::<usize>() {
            if (1..=count).contains(&index) {
                return Ok(Some(index - 1));
            }
        }
        println!("Invalid selection: {value:?}. Enter 1-{count} or q.");
    }
}

fn parse_extra_args(raw: &str) -> Result<Vec<String>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    shell_words::split(raw).context("invalid --extra value")
}

fn command_for_script(path: &Path, extra_args: &[String]) -> CommandSpec {
    let mut args = Vec::with_capacity(extra_args.len() + 1);
    args.push(path.as_os_str().to_owned());
    args.extend(extra_args.iter().map(OsString::from));
    CommandSpec {
        program: OsString::from("bash"),
        args,
    }
}

fn format_command(command: &CommandSpec) -> String {
    std::iter::once(command.program.as_os_str())
        .chain(command.args.iter().map(OsString::as_os_str))
        .map(quote_for_display)
        .collect::<Vec<_>>()
        .join(" ")
}

fn quote_for_display(value: &OsStr) -> String {
    let value = value.to_string_lossy();
    if !value.is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_@%+=:,./-".contains(character))
    {
        return value.into_owned();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn profile(id: &str, runtime: &str, task: &str) -> MediaProfile {
        MediaProfile {
            id: id.into(),
            name: id.into(),
            runtime: runtime.into(),
            variant: "test".into(),
            tasks: vec![task.into()],
            inputs: vec!["text".into()],
            status: "ready".into(),
            script: "media-models/test.sh".into(),
            description: "test profile".into(),
            requirements: Vec::new(),
            notes: Vec::new(),
        }
    }

    #[test]
    fn filters_profiles_across_runtime_task_and_input() {
        let profiles = vec![
            profile("minimax-h3", "audio.cpp", "video"),
            profile("ltx-2.5", "LTX-2", "video"),
        ];
        assert_eq!(filter_profiles(profiles.clone(), "AUDIO").len(), 1);
        assert_eq!(filter_profiles(profiles.clone(), "VIDEO").len(), 2);
        assert_eq!(filter_profiles(profiles.clone(), "LTX")[0].id, "ltx-2.5");
        assert_eq!(filter_profiles(profiles, "").len(), 2);
    }

    #[test]
    fn loads_and_validates_manifest() {
        let temp = tempdir().unwrap();
        let script = temp.path().join("media-models/test.sh");
        fs::create_dir_all(script.parent().unwrap()).unwrap();
        fs::write(&script, "#!/bin/sh\n").unwrap();
        let manifest = r#"{
          "schema_version": 1,
          "profiles": [{
            "id": "test",
            "name": "Test",
            "runtime": "test",
            "variant": "test",
            "tasks": ["music"],
            "inputs": ["text"],
            "status": "ready",
            "script": "media-models/test.sh",
            "description": "test"
          }]
        }"#;
        fs::write(temp.path().join(MANIFEST_FILE), manifest).unwrap();
        assert_eq!(load_manifest(temp.path()).unwrap().profiles.len(), 1);
    }

    #[test]
    fn rejects_media_script_escape() {
        let temp = tempdir().unwrap();
        let script = temp.path().join("media-models/test.sh");
        fs::create_dir_all(script.parent().unwrap()).unwrap();
        fs::write(&script, "#!/bin/sh\n").unwrap();
        assert!(safe_script_path(temp.path(), "../outside.sh").is_err());
    }

    #[test]
    fn command_keeps_prompt_as_one_argument() {
        let command = command_for_script(
            Path::new("/repo/media-models/generate.sh"),
            &["--prompt".into(), "bright blue sky".into()],
        );
        assert_eq!(command.program, OsString::from("bash"));
        assert_eq!(command.args[2], OsString::from("bright blue sky"));
        assert_eq!(
            format_command(&command),
            "bash /repo/media-models/generate.sh --prompt 'bright blue sky'"
        );
    }
}
