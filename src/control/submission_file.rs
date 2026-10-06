//! Shared explicit Submit-file input for CLI and keyboard-first clients.
//! Never writes, connects, starts engines, generates IDs, or retries work.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::catalog::Catalog;
use super::ipc::{self, ClientRequest, RequestEnvelope, WireRequestMessage};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionFileError {
    Unsafe,
    Read,
    TooLarge,
    Invalid,
    WrongOperation,
}
impl std::fmt::Display for SubmissionFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unsafe => "submit request file is unavailable or unsafe",
            Self::Read => "submit request file could not be read",
            Self::TooLarge => "submit request file exceeds the size limit",
            Self::Invalid => "submit request file is invalid",
            Self::WrongOperation => "submit request file must contain a v1 submit envelope",
        })
    }
}
impl std::error::Error for SubmissionFileError {}

/// Read one bounded, strict v1 native Submit against a refreshed caller-owned
/// catalog. Symlinks, devices and FIFOs are refused before reading; a file
/// replaced during opening is refused as well. Same-user file writers and
/// toolkit ancestors are trusted, not sandboxed. No path/body enters errors.
pub fn read(path: &Path, catalog: &Catalog) -> Result<RequestEnvelope, SubmissionFileError> {
    let file = open(path)?;
    let mut bytes = Vec::new();
    file.take(ipc::MAX_FRAME_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| SubmissionFileError::Read)?;
    if bytes.len() > ipc::MAX_FRAME_BYTES {
        return Err(SubmissionFileError::TooLarge);
    }
    let envelope: RequestEnvelope =
        serde_json::from_slice(&bytes).map_err(|_| SubmissionFileError::Invalid)?;
    if envelope.version != ipc::PROTOCOL_VERSION
        || !matches!(&envelope.message, WireRequestMessage::Submit { .. })
    {
        return Err(SubmissionFileError::WrongOperation);
    }
    ipc::validate_request_envelope(&envelope).map_err(|_| SubmissionFileError::Invalid)?;
    let ClientRequest::Submit { request, .. } = envelope
        .clone()
        .into_client_request()
        .map_err(|_| SubmissionFileError::Invalid)?
    else {
        return Err(SubmissionFileError::WrongOperation);
    };
    let refreshed = catalog
        .refresh()
        .map_err(|_| SubmissionFileError::Invalid)?;
    request
        .validate(&refreshed)
        .map_err(|_| SubmissionFileError::Invalid)?;
    Ok(envelope)
}

#[cfg(unix)]
fn open(path: &Path) -> Result<File, SubmissionFileError> {
    use std::fs::{self, OpenOptions};
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let before = fs::symlink_metadata(path).map_err(|_| SubmissionFileError::Unsafe)?;
    if !before.is_file() {
        return Err(SubmissionFileError::Unsafe);
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| SubmissionFileError::Unsafe)?;
    let opened = file.metadata().map_err(|_| SubmissionFileError::Unsafe)?;
    if !opened.is_file() || (before.dev(), before.ino()) != (opened.dev(), opened.ino()) {
        return Err(SubmissionFileError::Unsafe);
    }
    Ok(file)
}
#[cfg(not(unix))]
fn open(_: &Path) -> Result<File, SubmissionFileError> {
    Err(SubmissionFileError::Unsafe)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn shared_reader_accepts_exact_native_submit_and_refuses_aliases_or_wrong_operations() {
        let catalog = Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("submit.json");
        let envelope = RequestEnvelope::v1(WireRequestMessage::Submit {
            request_id: "caller-owned-id".into(),
            timeout_ms: 10_000,
            request: ipc::WireInferenceRequest::Language {
                profile_id: "gemma-4-26b-qat-mtp".into(),
                max_tokens: 64,
                turns: vec![ipc::WireChatTurn {
                    role: ipc::WireChatRole::User,
                    content: "private-prompt".into(),
                }],
            },
        });
        std::fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_vec(&read(&path, &catalog).unwrap()).unwrap(),
            serde_json::to_vec(&envelope).unwrap()
        );
        let alias = directory.path().join("alias");
        symlink(&path, &alias).unwrap();
        assert!(matches!(
            read(&alias, &catalog),
            Err(SubmissionFileError::Unsafe)
        ));
        let status = RequestEnvelope::v1(WireRequestMessage::Status {
            request_id: "caller-owned-id".into(),
        });
        std::fs::write(&path, serde_json::to_vec(&status).unwrap()).unwrap();
        assert!(matches!(
            read(&path, &catalog),
            Err(SubmissionFileError::WrongOperation)
        ));
        std::fs::write(&path, vec![b'x'; ipc::MAX_FRAME_BYTES + 1]).unwrap();
        assert!(matches!(
            read(&path, &catalog),
            Err(SubmissionFileError::TooLarge)
        ));
        std::fs::write(&path, b"private-prompt").unwrap();
        let error = read(&path, &catalog).err().unwrap();
        assert!(!format!("{error:?} {error}").contains("private-prompt"));
    }
}
