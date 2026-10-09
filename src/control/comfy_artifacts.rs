//! Validate ComfyUI output descriptors against one pinned, private job output
//! directory and map them to backend-relative result references.
//!
//! Open this handle before launching the exact owned Comfy engine generation,
//! while the configured output directory is fresh and private to that job.
//! Call [`ComfyArtifactRoot::validate`] only after the caller has independently
//! stopped that exact generation and established its engine, cgroup, and GPU
//! release. This module validates descriptor paths and current file metadata;
//! it does not prove release, open or retrieve artifact bytes, or claim that
//! bytes were delivered. The reference prefix must be a unique per-job path
//! under the client's configured backend artifact root. Keep output roots
//! isolated per job, or guarantee that output filenames are never reused.
//!
//! Only Comfy's `output` storage type is supported. The returned paths preserve
//! descriptor order and the descriptor's explicit `subfolder` and `filename`;
//! `node_id` and `media_type` remain descriptor metadata because
//! [`MediaArtifact`] has no fields for them.

#![cfg(target_os = "linux")]

use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use super::comfy::ComfyArtifact;
use super::result_buffer::{ArtifactKind, MediaArtifact};

const PRIVATE_DIRECTORY_MODE: u32 = 0o700;
const MAX_COMFY_ARTIFACTS: usize = 128;
const MAX_SINGLE_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOTAL_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_COMFY_PATH_BYTES: usize = 512;
const MAX_COMFY_COMPONENT_BYTES: usize = 128;
const MAX_BACKEND_REFERENCE_BYTES: usize = 2048;

/// Fixed, path-redacted failures from Comfy output artifact validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComfyArtifactError {
    UnsafeOutputRoot,
    InvalidBackendPrefix,
    InvalidDescriptor,
    UnsupportedStorage,
    DuplicateArtifact,
    TooManyArtifacts,
    InvalidBounds,
    InvalidArtifactFile,
    ArtifactTooLarge,
}

impl fmt::Display for ComfyArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsafeOutputRoot => "Comfy output directory is unavailable or unsafe",
            Self::InvalidBackendPrefix => "Comfy backend artifact prefix is invalid",
            Self::InvalidDescriptor => "Comfy output descriptor is invalid",
            Self::UnsupportedStorage => "Comfy output storage type is unsupported",
            Self::DuplicateArtifact => "Comfy output contains a duplicate artifact path",
            Self::TooManyArtifacts => "Comfy output contains too many artifacts",
            Self::InvalidBounds => "Comfy output artifact bounds are invalid",
            Self::InvalidArtifactFile => "Comfy output artifact file is unavailable or unsafe",
            Self::ArtifactTooLarge => "Comfy output artifact exceeds its size limit",
        })
    }
}

impl Error for ComfyArtifactError {}

#[derive(Clone, Copy, Eq, PartialEq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    mode: u32,
}

/// A pinned output root for one Comfy job.
///
/// The root must be a fresh, existing directory owned by the current effective
/// user with exact mode 0700. Every path component is opened without following
/// symlinks. The directory descriptor stays pinned across engine launch; each
/// validation also confirms that the configured path still names that same
/// directory before and after checking every descriptor.
pub struct ComfyArtifactRoot {
    directory: File,
    path: PathBuf,
    identity: DirectoryIdentity,
    uid: u32,
    backend_reference_prefix: String,
}

impl fmt::Debug for ComfyArtifactRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ComfyArtifactRoot(<pinned private directory>)")
    }
}

impl ComfyArtifactRoot {
    /// Pin a fresh per-job Comfy output directory and its unique logical prefix
    /// under the client's configured backend artifact root.
    ///
    /// Call this before launching the exact owned Comfy engine generation.
    /// The prefix is a validated relative path such as `jobs/<job-id>`; callers
    /// must ensure it is unique for the job and agrees with the backend root
    /// configured for artifact retrieval.
    pub fn open(
        output_root: impl AsRef<Path>,
        backend_reference_prefix: &str,
    ) -> Result<Self, ComfyArtifactError> {
        if !safe_comfy_relative(backend_reference_prefix, false) {
            return Err(ComfyArtifactError::InvalidBackendPrefix);
        }

        let path = output_root.as_ref();
        let directory = open_absolute_directory_nofollow(path)?;
        let uid = effective_uid();
        let identity = validate_private_root(&directory, uid)?;
        let root = Self {
            directory,
            path: path.to_path_buf(),
            identity,
            uid,
            backend_reference_prefix: backend_reference_prefix.to_owned(),
        };
        root.verify_named_root()?;
        Ok(root)
    }

    /// Revalidate an explicit Comfy descriptor list against the pinned output
    /// root and return same-order backend-relative artifact references.
    ///
    /// The caller must invoke this only after releasing the exact owned engine
    /// generation. The method accepts only `storage_type == "output"`, rejects
    /// duplicate or unsafe paths, and freshly opens each native file without
    /// following symlinks. It returns references only; it does not read or
    /// deliver file contents.
    pub fn validate(
        &self,
        descriptors: &[ComfyArtifact],
        max_file_bytes: u64,
    ) -> Result<Vec<MediaArtifact>, ComfyArtifactError> {
        if max_file_bytes == 0 || max_file_bytes > MAX_SINGLE_FILE_BYTES {
            return Err(ComfyArtifactError::InvalidBounds);
        }
        if descriptors.len() > MAX_COMFY_ARTIFACTS {
            return Err(ComfyArtifactError::TooManyArtifacts);
        }

        self.verify_named_root()?;
        let mut seen = BTreeSet::new();
        let mut total_bytes = 0_u64;
        let mut mapped = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            validate_descriptor(descriptor)?;
            if descriptor.storage_type != "output" {
                return Err(ComfyArtifactError::UnsupportedStorage);
            }

            let relative_path = self.backend_reference_path(descriptor)?;
            if !seen.insert(relative_path.clone()) {
                return Err(ComfyArtifactError::DuplicateArtifact);
            }

            self.verify_named_root()?;
            let file_size = self.validate_native_file(descriptor)?;
            self.verify_named_root()?;

            if file_size > max_file_bytes {
                return Err(ComfyArtifactError::ArtifactTooLarge);
            }
            total_bytes = total_bytes
                .checked_add(file_size)
                .ok_or(ComfyArtifactError::ArtifactTooLarge)?;
            if total_bytes > MAX_TOTAL_FILE_BYTES {
                return Err(ComfyArtifactError::ArtifactTooLarge);
            }

            mapped.push(
                MediaArtifact::new(ArtifactKind::Backend, relative_path)
                    .map_err(|_| ComfyArtifactError::InvalidDescriptor)?,
            );
        }

        self.verify_named_root()?;
        Ok(mapped)
    }

    fn backend_reference_path(
        &self,
        descriptor: &ComfyArtifact,
    ) -> Result<String, ComfyArtifactError> {
        let mut path = self.backend_reference_prefix.clone();
        if !descriptor.subfolder.is_empty() {
            path.push('/');
            path.push_str(&descriptor.subfolder);
        }
        path.push('/');
        path.push_str(&descriptor.filename);
        if path.len() > MAX_BACKEND_REFERENCE_BYTES {
            return Err(ComfyArtifactError::InvalidDescriptor);
        }
        Ok(path)
    }

    fn validate_native_file(&self, descriptor: &ComfyArtifact) -> Result<u64, ComfyArtifactError> {
        let mut directory = self
            .directory
            .try_clone()
            .map_err(|_| ComfyArtifactError::UnsafeOutputRoot)?;

        if !descriptor.subfolder.is_empty() {
            for component in descriptor.subfolder.split('/') {
                let name = c_string(component)?;
                let fd = unsafe {
                    libc::openat(
                        directory.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                    )
                };
                if fd < 0 {
                    return Err(ComfyArtifactError::InvalidArtifactFile);
                }
                directory = unsafe { File::from_raw_fd(fd) };
                let metadata = directory
                    .metadata()
                    .map_err(|_| ComfyArtifactError::InvalidArtifactFile)?;
                if !metadata.is_dir() || metadata.uid() != self.uid {
                    return Err(ComfyArtifactError::InvalidArtifactFile);
                }
            }
        }

        let filename = c_string(&descriptor.filename)?;
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                filename.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            )
        };
        if fd < 0 {
            return Err(ComfyArtifactError::InvalidArtifactFile);
        }
        let file = unsafe { File::from_raw_fd(fd) };
        let metadata = file
            .metadata()
            .map_err(|_| ComfyArtifactError::InvalidArtifactFile)?;
        if !metadata.is_file() || metadata.uid() != self.uid || metadata.nlink() != 1 {
            return Err(ComfyArtifactError::InvalidArtifactFile);
        }
        if metadata.len() == 0 {
            return Err(ComfyArtifactError::InvalidArtifactFile);
        }
        if metadata.len() > MAX_SINGLE_FILE_BYTES {
            return Err(ComfyArtifactError::ArtifactTooLarge);
        }
        Ok(metadata.len())
    }

    fn verify_named_root(&self) -> Result<(), ComfyArtifactError> {
        let pinned_identity = validate_private_root(&self.directory, self.uid)?;
        if pinned_identity != self.identity {
            return Err(ComfyArtifactError::UnsafeOutputRoot);
        }
        let named_directory = open_absolute_directory_nofollow(&self.path)?;
        let named_identity = validate_private_root(&named_directory, self.uid)?;
        if named_identity != self.identity {
            return Err(ComfyArtifactError::UnsafeOutputRoot);
        }
        Ok(())
    }
}

fn validate_descriptor(descriptor: &ComfyArtifact) -> Result<(), ComfyArtifactError> {
    if descriptor.node_id.is_empty()
        || descriptor.node_id.len() > MAX_COMFY_COMPONENT_BYTES
        || descriptor.node_id.chars().any(char::is_control)
        || descriptor.media_type.is_empty()
        || descriptor.media_type.len() > MAX_COMFY_COMPONENT_BYTES
        || descriptor.media_type.chars().any(char::is_control)
        || !safe_comfy_relative(&descriptor.filename, false)
        || descriptor.filename.contains('/')
        || !safe_comfy_relative(&descriptor.subfolder, true)
    {
        return Err(ComfyArtifactError::InvalidDescriptor);
    }
    Ok(())
}

/// Match the native parser's path policy at this filesystem trust boundary.
fn safe_comfy_relative(path: &str, allow_empty: bool) -> bool {
    (allow_empty && path.is_empty())
        || (!path.is_empty()
            && path.len() <= MAX_COMFY_PATH_BYTES
            && !path.contains(['\\', ':', '%', '?', '#'])
            && !path.chars().any(char::is_control)
            && path.split('/').all(|part| {
                !part.is_empty()
                    && part != "."
                    && part != ".."
                    && part.len() <= MAX_COMFY_COMPONENT_BYTES
            }))
}

fn validate_private_root(
    directory: &File,
    uid: u32,
) -> Result<DirectoryIdentity, ComfyArtifactError> {
    let metadata = directory
        .metadata()
        .map_err(|_| ComfyArtifactError::UnsafeOutputRoot)?;
    let mode = metadata.mode() & 0o7777;
    if !metadata.is_dir() || metadata.uid() != uid || mode != PRIVATE_DIRECTORY_MODE {
        return Err(ComfyArtifactError::UnsafeOutputRoot);
    }
    Ok(DirectoryIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        uid,
        mode,
    })
}

fn open_absolute_directory_nofollow(path: &Path) -> Result<File, ComfyArtifactError> {
    let raw = path.as_os_str().as_bytes();
    if raw.first() != Some(&b'/') || raw.contains(&0) {
        return Err(ComfyArtifactError::UnsafeOutputRoot);
    }
    let components = raw[1..].split(|byte| *byte == b'/');
    let root_name = CString::new("/").expect("literal has no NUL");
    let root_fd = unsafe {
        libc::open(
            root_name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if root_fd < 0 {
        return Err(ComfyArtifactError::UnsafeOutputRoot);
    }
    let mut directory = unsafe { File::from_raw_fd(root_fd) };

    for component in components {
        if component.is_empty() || component == b"." || component == b".." {
            return Err(ComfyArtifactError::UnsafeOutputRoot);
        }
        let name = CString::new(component).map_err(|_| ComfyArtifactError::UnsafeOutputRoot)?;
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(ComfyArtifactError::UnsafeOutputRoot);
        }
        directory = unsafe { File::from_raw_fd(fd) };
    }
    Ok(directory)
}

fn c_string(value: &str) -> Result<CString, ComfyArtifactError> {
    CString::new(value).map_err(|_| ComfyArtifactError::InvalidDescriptor)
}

fn effective_uid() -> u32 {
    unsafe { libc::geteuid() as u32 }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{symlink, PermissionsExt};
    use std::path::Path;

    use super::{ComfyArtifactError, ComfyArtifactRoot, MAX_COMFY_ARTIFACTS};
    use crate::control::comfy::ComfyArtifact;
    use crate::control::result_buffer::ArtifactKind;

    fn private_dir(parent: &Path, name: &str) -> std::path::PathBuf {
        let path = parent.join(name);
        fs::create_dir(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn descriptor(filename: &str, subfolder: &str) -> ComfyArtifact {
        ComfyArtifact {
            node_id: "12".to_owned(),
            media_type: "images".to_owned(),
            filename: filename.to_owned(),
            subfolder: subfolder.to_owned(),
            storage_type: "output".to_owned(),
        }
    }

    fn root(path: &Path) -> ComfyArtifactRoot {
        ComfyArtifactRoot::open(path, "jobs/job-unique").unwrap()
    }

    #[test]
    fn maps_valid_nested_file_paths_in_descriptor_order() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        let images = private_dir(&output, "images");
        fs::write(images.join("frame.png"), b"png").unwrap();
        fs::write(output.join("preview.webp"), b"webp").unwrap();

        let artifacts = root(&output)
            .validate(
                &[
                    descriptor("frame.png", "images"),
                    descriptor("preview.webp", ""),
                ],
                16,
            )
            .unwrap();

        assert_eq!(artifacts.len(), 2);
        assert!(artifacts
            .iter()
            .all(|artifact| artifact.kind == ArtifactKind::Backend));
        assert_eq!(
            artifacts[0].relative_path,
            "jobs/job-unique/images/frame.png"
        );
        assert_eq!(artifacts[1].relative_path, "jobs/job-unique/preview.webp");
    }

    #[test]
    fn accepts_no_descriptors_but_rejects_empty_or_unsafe_descriptor_paths() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        let output_root = root(&output);
        assert!(output_root.validate(&[], 16).unwrap().is_empty());

        let mut empty_filename = descriptor("", "");
        assert_eq!(
            output_root.validate(&[empty_filename.clone()], 16),
            Err(ComfyArtifactError::InvalidDescriptor)
        );
        empty_filename.filename = "../escape.png".to_owned();
        assert_eq!(
            output_root.validate(&[empty_filename], 16),
            Err(ComfyArtifactError::InvalidDescriptor)
        );
        assert_eq!(
            output_root.validate(&[descriptor("frame.png", "../outside")], 16),
            Err(ComfyArtifactError::InvalidDescriptor)
        );
        assert_eq!(
            output_root.validate(&[descriptor("/tmp/frame.png", "")], 16),
            Err(ComfyArtifactError::InvalidDescriptor)
        );
        assert_eq!(
            output_root.validate(&[descriptor("nested/frame.png", "")], 16),
            Err(ComfyArtifactError::InvalidDescriptor)
        );
    }

    #[test]
    fn rejects_non_output_storage_types() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        let output_root = root(&output);
        for storage_type in ["temp", "input", "unknown"] {
            let mut item = descriptor("frame.png", "");
            item.storage_type = storage_type.to_owned();
            assert_eq!(
                output_root.validate(&[item], 16),
                Err(ComfyArtifactError::UnsupportedStorage)
            );
        }
    }

    #[test]
    fn rejects_symlinked_file_and_subfolder_components() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        let outside = private_dir(temp.path(), "outside");
        fs::write(outside.join("frame.png"), b"png").unwrap();
        symlink(&outside, output.join("linked-dir")).unwrap();
        symlink(outside.join("frame.png"), output.join("linked.png")).unwrap();
        let output_root = root(&output);

        assert_eq!(
            output_root.validate(&[descriptor("frame.png", "linked-dir")], 16),
            Err(ComfyArtifactError::InvalidArtifactFile)
        );
        assert_eq!(
            output_root.validate(&[descriptor("linked.png", "")], 16),
            Err(ComfyArtifactError::InvalidArtifactFile)
        );
    }

    #[test]
    fn rejects_special_and_empty_files() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        fs::write(output.join("empty.png"), b"").unwrap();
        let fifo = output.join("pipe.png");
        let fifo_name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
        let result = unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) };
        assert_eq!(result, 0);
        let output_root = root(&output);

        assert_eq!(
            output_root.validate(&[descriptor("empty.png", "")], 16),
            Err(ComfyArtifactError::InvalidArtifactFile)
        );
        assert_eq!(
            output_root.validate(&[descriptor("pipe.png", "")], 16),
            Err(ComfyArtifactError::InvalidArtifactFile)
        );
    }

    #[test]
    fn rejects_duplicate_paths_and_overbounds() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        fs::write(output.join("frame.png"), b"png").unwrap();
        let output_root = root(&output);
        assert_eq!(
            output_root.validate(
                &[descriptor("frame.png", ""), descriptor("frame.png", "")],
                16
            ),
            Err(ComfyArtifactError::DuplicateArtifact)
        );

        let too_many = (0..=MAX_COMFY_ARTIFACTS)
            .map(|index| descriptor(&format!("frame-{index}.png"), ""))
            .collect::<Vec<_>>();
        assert_eq!(
            output_root.validate(&too_many, 16),
            Err(ComfyArtifactError::TooManyArtifacts)
        );
        assert_eq!(
            output_root.validate(&[descriptor("frame.png", "")], 2),
            Err(ComfyArtifactError::ArtifactTooLarge)
        );
        assert_eq!(
            output_root.validate(&[descriptor("frame.png", "")], 0),
            Err(ComfyArtifactError::InvalidBounds)
        );
    }

    #[test]
    fn detects_named_root_replacement_after_pinning() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        fs::write(output.join("frame.png"), b"png").unwrap();
        let output_root = root(&output);
        let displaced = temp.path().join("displaced-output");
        fs::rename(&output, &displaced).unwrap();
        let replacement = private_dir(temp.path(), "output");
        fs::write(replacement.join("frame.png"), b"other").unwrap();

        assert_eq!(
            output_root.validate(&[descriptor("frame.png", "")], 16),
            Err(ComfyArtifactError::UnsafeOutputRoot)
        );
    }

    #[test]
    fn rejects_hard_linked_output_and_invalid_root_aliases() {
        let temp = tempfile::tempdir().unwrap();
        let output = private_dir(temp.path(), "output");
        fs::write(output.join("frame.png"), b"png").unwrap();
        fs::hard_link(output.join("frame.png"), output.join("alias.png")).unwrap();
        assert_eq!(
            root(&output).validate(&[descriptor("frame.png", "")], 16),
            Err(ComfyArtifactError::InvalidArtifactFile)
        );

        let symlinked_root = temp.path().join("output-alias");
        symlink(&output, &symlinked_root).unwrap();
        assert!(matches!(
            ComfyArtifactRoot::open(&symlinked_root, "jobs/job-unique"),
            Err(ComfyArtifactError::UnsafeOutputRoot)
        ));
    }
}
