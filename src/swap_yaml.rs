//! Deterministic, text-level handling of `llama-swap.yaml`.
//!
//! llama-swap v247 has no runtime enable/disable API, so model enablement is
//! a config edit plus SIGHUP hot-reload. This module keeps that edit
//! deterministic: parse top-level model entries, toggle a `disabled:` line
//! inside the entry block, snapshot the file before writing, then signal the
//! router. No YAML library — the file is house-authored text and the parser
//! only needs the subset of structure we write ourselves.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::llama_swap::SwapModel;

/// A top-level model entry parsed from the yaml text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapYamlEntry {
    pub id: String,
    pub disabled: bool,
    /// First `-m` (or `--model`) path found in the entry's cmd block.
    pub model_path: Option<String>,
}

const MODELS_SECTION: &str = "models:";

/// Byte-range (line indices) of the `models:` section: from the `models:`
/// line to the next top-level (0-indent, non-comment) key. Everything
/// inside is indented, so a 0-indent key definitively ends the section.
fn models_section_span(lines: &[&str]) -> Option<(usize, usize)> {
    let start = lines.iter().position(|line| *line == MODELS_SECTION)?;
    let mut end = lines.len();
    for (offset, line) in lines[start + 1..].iter().enumerate() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !trimmed.starts_with(' ') {
            end = start + 1 + offset;
            break;
        }
    }
    Some((start, end))
}

/// Line span (start index of the entry key line, exclusive end) of the
/// top-level entry with key `id`, or `None`. Used by text-level entry edits.
pub(crate) fn entry_span(lines: &[&str], id: &str) -> Option<(usize, usize)> {
    let (section_start, section_end) = models_section_span(lines)?;
    let mut index = section_start + 1;
    while index < section_end {
        let mut end = index + 1;
        while end < section_end && !is_entry_boundary(lines[end]) {
            end += 1;
        }
        if entry_key(lines[index]) == Some(id) {
            return Some((index, end));
        }
        index = end;
    }
    None
}

/// Parse every active (non-commented) top-level model entry.
pub fn parse_entries(text: &str) -> Vec<SwapYamlEntry> {
    let lines: Vec<&str> = text.lines().collect();
    let Some((section_start, section_end)) = models_section_span(&lines) else {
        return Vec::new();
    };

    let mut entries = Vec::new();
    let mut index = section_start + 1;
    while index < section_end {
        let Some(id) = entry_key(lines[index]) else {
            index += 1;
            continue;
        };
        let mut end = index + 1;
        while end < section_end && !is_entry_boundary(lines[end]) {
            end += 1;
        }
        let body = &lines[index + 1..end];
        entries.push(SwapYamlEntry {
            id: id.to_owned(),
            disabled: body.iter().any(|line| is_disabled_line(line)),
            model_path: body.iter().find_map(|line| model_path(line)),
        });
        index = end;
    }
    entries
}

/// Parse model entries and merge their config-level state into `models`.
pub fn merge_into(models: &mut [SwapModel], text: &str) {
    let entries = parse_entries(text);
    for model in models.iter_mut() {
        if let Some(entry) = entries.iter().find(|entry| entry.id == model.id) {
            model.disabled = entry.disabled;
            model.size_bytes = entry.model_path.as_deref().and_then(disk_size);
        }
    }
}

/// Toggle `disabled: true` inside the entry block for `id`.
///
/// Disabling inserts `    disabled: true` directly under the entry key.
/// Enabling removes any active `disabled:` line from the block. Commented
/// lines are never touched, matching the house style of commented-out
/// historical config.
pub fn set_disabled(text: &str, id: &str, disable: bool) -> Result<String> {
    let trailing_newline = text.ends_with('\n');
    let lines: Vec<&str> = text.lines().collect();
    let Some((section_start, section_end)) = models_section_span(&lines) else {
        bail!("no models section found");
    };

    let mut index = section_start + 1;
    while index < section_end {
        let Some(entry_id) = entry_key(lines[index]) else {
            index += 1;
            continue;
        };
        let mut end = index + 1;
        while end < section_end && !is_entry_boundary(lines[end]) {
            end += 1;
        }
        if entry_id != id {
            index = end;
            continue;
        }

        let mut updated: Vec<&str> = Vec::with_capacity(lines.len() + 1);
        if disable {
            if lines[index + 1..end]
                .iter()
                .any(|line| is_disabled_line(line))
            {
                return Ok(text.to_owned());
            }
            updated.extend_from_slice(&lines[..index + 1]);
            updated.push("    disabled: true");
            updated.extend_from_slice(&lines[index + 1..]);
        } else {
            updated.extend_from_slice(&lines[..index + 1]);
            updated.extend(
                lines[index + 1..end]
                    .iter()
                    .filter(|line| !is_disabled_line(line)),
            );
            updated.extend_from_slice(&lines[end..]);
        }
        let mut joined = updated.join("\n");
        if trailing_newline {
            joined.push('\n');
        }
        return Ok(joined);
    }

    bail!("model {id:?} not found in llama-swap.yaml");
}

/// Snapshot the yaml beside itself before a write, per repo convention.
pub fn snapshot(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let stamp = format_utc_timestamp(seconds);
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "llama-swap.yaml".into());
    let mut candidate = path.with_file_name(format!("{file_name}.bak-{stamp}"));
    let mut suffix = 1;
    while candidate.exists() {
        candidate = path.with_file_name(format!("{file_name}.bak-{stamp}-{suffix}"));
        suffix += 1;
    }
    fs::copy(path, &candidate).with_context(|| format!("failed to snapshot {}", path.display()))?;
    Ok(candidate)
}

/// Atomically write `content` to `path` via a temporary file in the same directory.
pub fn atomic_write(path: impl AsRef<Path>, content: &str) -> Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".into());
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .with_context(|| format!("failed to create temporary file for {}", path.display()))?;
    let write_result = (|| -> Result<()> {
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path).with_context(|| format!("failed to replace {}", path.display()))?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    write_result
}

fn format_utc_timestamp(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);

    format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}")
}

/// SIGHUP the running llama-swap router so config edits hot-reload.
pub fn reload_router() -> Result<usize> {
    let output = Command::new("pgrep")
        .args(["-x", "llama-swap"])
        .output()
        .context("failed to run pgrep")?;
    let text = String::from_utf8_lossy(&output.stdout);
    let mut signalled = 0;
    for pid in text
        .lines()
        .filter_map(|line| line.trim().parse::<i32>().ok())
        .filter(|pid| *pid > 1)
    {
        Command::new("kill")
            .args(["-HUP", &pid.to_string()])
            .output()
            .context("failed to run kill")?;
        signalled += 1;
    }
    if signalled == 0 {
        bail!("no running llama-swap process found");
    }
    Ok(signalled)
}

/// Match a top-level (exactly 2-space indented, non-comment) entry key line
/// under `models:` and return its unquoted id. Model entries are mappings,
/// so the key must end with a bare colon — scalar children like
/// `preload: [...]` under sibling top-level blocks are skipped.
fn entry_key(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("  ")?;
    if rest.starts_with(' ') || rest.starts_with('#') {
        return None;
    }
    let colon = rest.find(':')?;
    if !rest[colon + 1..].trim().is_empty() {
        return None;
    }
    let key = rest[..colon].trim();
    if key.is_empty() || key.contains(' ') {
        return None;
    }
    let key = key.strip_prefix('"').unwrap_or(key);
    let key = key.strip_suffix('"').unwrap_or(key);
    Some(key)
}

fn is_entry_boundary(line: &str) -> bool {
    entry_key(line).is_some()
}

fn is_disabled_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("disabled:")
        && trimmed
            .trim_start_matches("disabled:")
            .trim()
            .starts_with("true")
}

fn model_path(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        return None;
    }
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    tokens
        .windows(2)
        .find(|pair| pair[0] == "-m" || pair[0] == "--model")
        .and_then(|pair| {
            let path = pair[1];
            (!path.is_empty() && !path.starts_with('-')).then(|| path.to_owned())
        })
}

/// Size in bytes of the model file at `raw`, expanding GGUF shard sets so a
/// `-00001-of-00033.gguf` entry reports the full shard set size.
fn disk_size(raw: &str) -> Option<u64> {
    let path = Path::new(raw.trim());
    if let Some(((start, end), count)) = shard_span(&path.file_name()?.to_string_lossy()) {
        if count > 1 {
            let summed = sum_shards(path, start, end);
            if summed.is_some() {
                return summed;
            }
        }
    }
    let meta = fs::metadata(path).ok()?;
    meta.is_file().then_some(meta.len())
}

/// Sum every sibling shard matching `prefix*-of-NNNNN.gguf` where the prefix
/// and suffix have the same length as the reference shard's name.
fn sum_shards(path: &Path, index_start: usize, index_end: usize) -> Option<u64> {
    let file_name = path.file_name()?.to_string_lossy().into_owned();
    let prefix = &file_name[..index_start];
    let suffix = &file_name[index_end..];
    let parent = path.parent()?;
    let mut total = 0u64;
    let mut found = false;
    for entry in fs::read_dir(parent).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.len() == file_name.len() && name.starts_with(prefix) && name.ends_with(suffix) {
            total += entry.metadata().ok()?.len();
            found = true;
        }
    }
    found.then_some(total)
}

/// Match names shaped `*-NNNNN-of-NNNNN.gguf` and return the digit span of
/// the shard index plus the shard count.
fn shard_span(file_name: &str) -> Option<((usize, usize), usize)> {
    let of = file_name.find("-of-")?;
    let index_start = file_name[..of]
        .char_indices()
        .rev()
        .take_while(|(_, character)| character.is_ascii_digit())
        .map(|(index, _)| index)
        .last()?;
    if index_start == of {
        return None;
    }
    let index: usize = file_name[index_start..of].parse().ok()?;
    let count: usize = file_name[of + 4..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    (index <= count).then_some(((index_start, of), count))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"apiKeys: []
startPort: 10001

models:
  "alpha":
    name: "Alpha"
    cmd: |
      llama-server -m /models/alpha/model.gguf --port ${PORT}

  "beta-shard":
    name: "Beta"
    cmd: |
      llama-server
      -m /models/beta/beta-00001-of-00003.gguf
      --port ${PORT}

on_startup:
  preload: []
"#;
    #[test]
    fn parses_entries_with_state_and_paths() {
        let entries = parse_entries(SAMPLE);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "alpha");
        assert!(!entries[0].disabled);
        assert_eq!(
            entries[0].model_path.as_deref(),
            Some("/models/alpha/model.gguf")
        );
        assert_eq!(entries[1].id, "beta-shard");
        assert_eq!(
            entries[1].model_path.as_deref(),
            Some("/models/beta/beta-00001-of-00003.gguf")
        );
    }

    #[test]
    fn skips_commented_and_top_level_keys() {
        let text = "models:\n  # \"old\":\n  #   cmd: x\nother: 1\n";
        assert!(parse_entries(text).is_empty());
        assert!(parse_entries("no models here").is_empty());
    }

    #[test]
    fn scalar_children_of_sibling_blocks_are_not_entries() {
        let text = "models:\n  \"alpha\":\n    cmd: x\nhooks:\n  on_startup:\n    preload:\n      - \"m\"\n";
        let entries = parse_entries(text);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "alpha");
    }

    #[test]
    fn round_trips_disable_and_enable() {
        let disabled = set_disabled(SAMPLE, "alpha", true).unwrap();
        assert!(disabled.contains("  \"alpha\":\n    disabled: true\n"));
        let entries = parse_entries(&disabled);
        assert!(entries[0].disabled);

        let enabled = set_disabled(&disabled, "alpha", false).unwrap();
        assert_eq!(enabled, SAMPLE);
        assert!(!parse_entries(&enabled)[0].disabled);
    }

    #[test]
    fn disable_is_idempotent_and_unknown_model_errors() {
        let once = set_disabled(SAMPLE, "beta-shard", true).unwrap();
        assert_eq!(set_disabled(&once, "beta-shard", true).unwrap(), once);
        assert!(set_disabled(SAMPLE, "missing", true).is_err());
        assert!(set_disabled("other: 1", "alpha", true).is_err());
    }

    #[test]
    fn enable_without_flag_is_noop() {
        assert_eq!(set_disabled(SAMPLE, "alpha", false).unwrap(), SAMPLE);
    }

    #[test]
    fn shard_span_matches_gguf_shard_names() {
        let name = "model-00001-of-00033.gguf";
        let ((start, end), count) = shard_span(name).unwrap();
        assert_eq!(count, 33);
        assert_eq!(&name[..start], "model-");
        assert_eq!(&name[end..], "-of-00033.gguf");
        assert!(shard_span("model.gguf").is_none());
        assert!(shard_span("model-00001-of-00033.bin").is_some());
    }

    #[test]
    fn disk_size_sums_shards() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        for name in ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf"] {
            fs::write(dir.join(name), vec![0u8; 10]).unwrap();
        }
        assert_eq!(
            disk_size(dir.join("m-00001-of-00003.gguf").to_str().unwrap()),
            Some(20)
        );
        fs::write(dir.join("single.gguf"), vec![0u8; 7]).unwrap();
        assert_eq!(
            disk_size(dir.join("single.gguf").to_str().unwrap()),
            Some(7)
        );
        assert_eq!(
            disk_size(dir.join("missing-00001-of-00003.gguf").to_str().unwrap()),
            None
        );
    }

    #[test]
    fn merge_into_fills_disabled_and_size() {
        let temp = tempfile::tempdir().unwrap();
        let model_file = temp.path().join("model.gguf");
        fs::write(&model_file, vec![0u8; 5]).unwrap();
        let text = format!(
            "models:\n  \"alpha\":\n    disabled: true\n    cmd: |\n      -m {}\n",
            model_file.display()
        );
        let mut models = vec![SwapModel {
            id: "alpha".into(),
            state: "unloaded".into(),
            name: String::new(),
            description: String::new(),
            created: None,
            disabled: false,
            size_bytes: None,
        }];
        merge_into(&mut models, &text);
        assert!(models[0].disabled);
        assert_eq!(models[0].size_bytes, Some(5));
    }

    #[test]
    fn format_utc_timestamp_matches_known_dates() {
        assert_eq!(format_utc_timestamp(0), "19700101-000000");
        assert_eq!(format_utc_timestamp(1_709_164_800), "20240229-000000");
    }

    #[test]
    fn snapshot_and_atomic_write_round_trip() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("llama-swap.yaml");
        atomic_write(&target, "models: {}\n").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "models: {}\n");

        let backup = snapshot(&target).unwrap();
        assert!(backup.is_file());
        assert_eq!(fs::read_to_string(&backup).unwrap(), "models: {}\n");

        atomic_write(&target, "models:\n  a: {}\n").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "models:\n  a: {}\n");
        assert_eq!(fs::read_to_string(&backup).unwrap(), "models: {}\n");
    }
}
