//! Advisory lifetime locks for the future host-local coordinator.
//!
//! All managed entry points must agree on one pre-existing, caller-owned,
//! private lock directory on a local filesystem, independently of checkout or
//! job-store location. These locks do not fence legacy direct engine clients,
//! prove GPU release, or replace startup reconciliation. Never unlink a lock
//! file: replacing its inode can allow two holders to believe they own it.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::path::Path;

use anyhow::{ensure, Context, Result};

/// Dropping this guard releases all acquired locks, including after a crash.
/// The persistent empty files are deliberately left in place.
pub struct CoordinatorLock {
    _files: Vec<File>,
}

impl CoordinatorLock {
    #[cfg(unix)]
    pub fn acquire(directory: &Path, gpus: impl IntoIterator<Item = u32>) -> Result<Self> {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

        let directory = directory
            .canonicalize()
            .context("resolve host lock directory")?;
        let metadata = directory.metadata()?;
        ensure!(metadata.is_dir(), "host lock location is not a directory");
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "host lock directory must not be accessible to group or other users"
        );
        let mut unique_gpus = BTreeSet::new();
        for (index, gpu) in gpus.into_iter().enumerate() {
            ensure!(index < 16, "too many GPU locks requested");
            ensure!(gpu <= 1023, "GPU lock index exceeds 1023");
            unique_gpus.insert(gpu);
        }
        let names = std::iter::once("coordinator.lock".to_owned())
            .chain(unique_gpus.into_iter().map(|gpu| format!("gpu-{gpu}.lock")));
        let mut files = Vec::new();
        for name in names {
            let path = directory.join(name);
            let file = match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let entry = path
                        .symlink_metadata()
                        .context("inspect existing host lock")?;
                    ensure!(
                        entry.is_file() && !entry.file_type().is_symlink(),
                        "host lock must be a regular, non-symlink file"
                    );
                    let file = OpenOptions::new().read(true).write(true).open(&path)?;
                    let opened = file.metadata()?;
                    ensure!(
                        entry.dev() == opened.dev() && entry.ino() == opened.ino(),
                        "host lock changed while opening"
                    );
                    file
                }
                Err(error) => return Err(error).context("open host ownership lock"),
            };
            let opened = file.metadata()?;
            ensure!(
                opened.is_file() && opened.nlink() == 1 && opened.uid() == metadata.uid(),
                "host lock must have one link and the directory owner"
            );
            ensure!(
                opened.permissions().mode() & 0o077 == 0,
                "host lock must be private"
            );
            file.try_lock().map_err(|error| {
                anyhow::anyhow!("host coordinator ownership unavailable: {error}")
            })?;
            files.push(file);
        }
        Ok(Self { _files: files })
    }

    #[cfg(not(unix))]
    pub fn acquire(_directory: &Path, _gpus: impl IntoIterator<Item = u32>) -> Result<Self> {
        anyhow::bail!("host coordinator ownership is only implemented for Unix")
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, PermissionsExt};

    fn private_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }

    #[test]
    fn lifetime_lock_refuses_competitors_and_retains_inode_after_release() {
        let directory = private_directory();
        let first = CoordinatorLock::acquire(directory.path(), [0, 0, 1]).unwrap();
        assert!(CoordinatorLock::acquire(directory.path(), []).is_err());
        let file = directory.path().join("coordinator.lock");
        assert!(file.is_file());
        drop(first);
        assert!(file.is_file());
        assert!(CoordinatorLock::acquire(directory.path(), [0]).is_ok());
    }

    #[test]
    fn invalid_or_shared_paths_fail_closed() {
        let directory = private_directory();
        assert!(CoordinatorLock::acquire(directory.path(), [1024]).is_err());
        assert!(CoordinatorLock::acquire(directory.path(), std::iter::repeat(0)).is_err());
        assert!(!directory.path().join("coordinator.lock").exists());
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(CoordinatorLock::acquire(directory.path(), []).is_err());
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let target = directory.path().join("target");
        std::fs::write(&target, "preserve").unwrap();
        symlink(&target, directory.path().join("coordinator.lock")).unwrap();
        assert!(CoordinatorLock::acquire(directory.path(), []).is_err());
        assert_eq!(std::fs::read_to_string(target).unwrap(), "preserve");
    }

    #[test]
    fn child_process_probe() {
        let Some(directory) = std::env::var_os("L3MS_TEST_LOCK_DIRECTORY") else {
            return;
        };
        assert!(CoordinatorLock::acquire(Path::new(&directory), [0]).is_err());
    }

    #[test]
    fn independent_process_cannot_acquire_held_host_lock() {
        let directory = private_directory();
        let _guard = CoordinatorLock::acquire(directory.path(), [0]).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "control::ownership::tests::child_process_probe",
                "--nocapture",
            ])
            .env("L3MS_TEST_LOCK_DIRECTORY", directory.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
