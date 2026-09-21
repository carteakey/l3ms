//! Shared filesystem utilities used by the storage modules.
//!
//! All writes go through [`atomic_write`] (write → fsync → rename) so a
//! crash at any point leaves either the old file or the new one intact.
//! Snapshots use [`write_unique_snapshot`] with a timestamp-plus-counter
//! name to guarantee uniqueness without needing a lock.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};

pub(crate) static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `content` to `path` atomically (temp → fsync → rename).
///
/// On Unix the target's existing permission bits are preserved when the
/// file already exists; `unix_mode` overrides the mode for new files or
/// when the caller wants to force a specific mode.
pub(crate) fn atomic_write(path: &Path, content: &[u8], unix_mode: Option<u32>) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create directory {}", parent.display()))?;

    let file_name = path
        .file_name()
        .map_or_else(|| OsString::from("file"), OsString::from)
        .to_string_lossy()
        .into_owned();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for attempt in 0_u32..100 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.l3ms-tmp-{}-{nonce}-{counter}-{attempt}",
            std::process::id()
        ));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        };

        let result = (|| -> Result<()> {
            file.write_all(content)?;
            file.sync_all()?;
            drop(file);

            #[cfg(unix)]
            if let Some(mode) = unix_mode {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&temporary, fs::Permissions::from_mode(mode))?;
            }
            #[cfg(not(unix))]
            let _ = unix_mode;

            fs::rename(&temporary, path)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        return result;
    }

    Err(anyhow!("could not allocate a temporary file"))
}

/// Write all of `content` to `file` and fsync.
pub(crate) fn write_and_sync(file: &mut File, content: &[u8]) -> std::io::Result<()> {
    file.write_all(content)?;
    file.sync_all()
}

/// Create a uniquely-named snapshot file in `directory`.
///
/// The name format is `{stamp}__{note}{extension}`, with a counter suffix
/// appended on collisions.  Up to 10,000 suffixes are tried before failing.
pub(crate) fn write_unique_snapshot(
    directory: &Path,
    stamp: &str,
    note: &str,
    extension: &str,
    content: &[u8],
) -> Result<PathBuf> {
    for collision in 0_u32..10_000 {
        let stamp = if collision == 0 {
            stamp.to_owned()
        } else {
            // `~` sorts after the base name's `_`, so reverse filename order
            // still places a later same-second snapshot first.
            format!("{stamp}~{collision:04}")
        };
        let path = directory.join(format!("{stamp}__{note}{extension}"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                if let Err(error) = write_and_sync(&mut file, content) {
                    drop(file);
                    let _ = fs::remove_file(&path);
                    return Err(anyhow::Error::new(error)).with_context(|| {
                        format!("failed to write snapshot {}", path.display())
                    });
                }
                return Ok(path);
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create snapshot {}", path.display())
                })
            }
        }
    }
    Err(anyhow!("could not allocate a unique snapshot name"))
}

/// Current UTC time as `YYYYMMDDTHHMMSSz` for use in snapshot filenames.
pub(crate) fn safe_stamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_utc_seconds(seconds)
}

/// Format a Unix timestamp as `YYYYMMDDTHHMMSSz`.
///
/// Uses Howard Hinnant's public-domain `civil_from_days` Gregorian
/// conversion so no date/time dependency is needed.
pub(crate) fn format_utc_seconds(seconds: u64) -> String {
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

    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

/// Resolve `path` to an absolute path without requiring it to exist.
///
/// Symlink components that *do* exist are canonicalized eagerly; missing
/// trailing components are left as-is so callers can create them later.
pub(crate) fn resolve_allow_missing(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to determine current directory")?
            .join(path)
    };

    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(part) => {
                resolved.push(part);
                if fs::symlink_metadata(&resolved).is_ok() {
                    resolved = fs::canonicalize(&resolved).with_context(|| {
                        format!("failed to resolve path component {}", resolved.display())
                    })?;
                }
            }
        }
    }
    Ok(resolved)
}

/// Verify that `name` is a single, plain path component (no `/` or `..`).
pub(crate) fn ensure_single_component(name: &str) -> Result<()> {
    let path = Path::new(name);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(anyhow!("invalid version path")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn format_utc_seconds_known_dates() {
        assert_eq!(format_utc_seconds(0), "19700101T000000Z");
        assert_eq!(format_utc_seconds(1_709_164_800), "20240229T000000Z");
    }

    #[test]
    fn resolve_allow_missing_normalizes_dotdot() {
        let temp = TempDir::new().unwrap();
        let resolved = resolve_allow_missing(&temp.path().join("a/../b")).unwrap();
        assert_eq!(resolved, temp.path().join("b"));
    }

    #[test]
    fn ensure_single_component_rejects_paths() {
        assert!(ensure_single_component("ok").is_ok());
        assert!(ensure_single_component("a/b").is_err());
        assert!(ensure_single_component("..").is_err());
        assert!(ensure_single_component("").is_err());
    }

    #[test]
    fn atomic_write_round_trips() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("target.txt");
        atomic_write(&path, b"hello", None).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello");
        atomic_write(&path, b"world", None).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"world");
    }

    #[test]
    fn write_unique_snapshot_creates_distinct_files() {
        let temp = TempDir::new().unwrap();
        let p1 = write_unique_snapshot(temp.path(), "20260101T120000Z", "note", ".json", b"a").unwrap();
        let p2 = write_unique_snapshot(temp.path(), "20260101T120000Z", "note", ".json", b"b").unwrap();
        assert_ne!(p1, p2);
        assert_eq!(fs::read(p1).unwrap(), b"a");
        assert_eq!(fs::read(p2).unwrap(), b"b");
    }
}
