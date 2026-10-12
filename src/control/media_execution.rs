//! Bounded execution and local artifact validation for Breeze-TTS-2.
//!
//! The caller allocates a fresh, current-UID mode-0700 job directory before
//! dispatch. This module never creates or removes it. The native wrapper gets
//! explicit output paths, so inherited output-directory settings cannot redirect
//! generated files. This is not a filesystem sandbox: the toolkit and its
//! dependencies remain trusted executables. The caller must retain exclusive,
//! trusted ownership of the job directory and its ancestors throughout the run;
//! local filesystem operations cannot be preempted between context checks.

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::catalog::{Catalog, Device};
use super::cuda_device::CudaDevice;
use super::native_commands::{
    is_supported_cpu_breeze_speech, prepare_breeze_speech_recipe,
    prepare_breeze_speech_recipe_for_route, NativeOutputExpectation, CPU_BREEZE_PROFILE_ID,
};
use super::process::{self, ProcessFailureDisposition, ProcessFailureKind, ProcessLimits};
use super::requests::{InferenceRequest, MediaInput, SpeechMode};
use super::routes::{NativeRoute, ResolvedProfileRoute};
use super::runtime::{OperationContext, RecoveryPolicy};

const OUTPUT_NAME: &str = "result.wav";
const CHUNKS_NAME: &str = "result.wav.chunks";
const MAX_CAPTURE_BYTES: usize = 4 * 1024 * 1024;
const MAX_STREAM_CHUNKS: usize = 63;
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOTAL_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;

/// Fixed, redacted failure categories. No path, prompt, argv, or process output
/// is retained; process disposition remains available for safe reconciliation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaExecutionErrorKind {
    InvalidRequest,
    InvalidOutputDirectory,
    OutputAlreadyExists,
    Process(ProcessFailureKind),
    InvalidArtifact,
    ArtifactTooLarge,
    TooManyArtifacts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaExecutionFailure {
    disposition: ProcessFailureDisposition,
    kind: MediaExecutionErrorKind,
}

impl MediaExecutionFailure {
    pub fn disposition(self) -> ProcessFailureDisposition {
        self.disposition
    }

    pub fn kind(self) -> MediaExecutionErrorKind {
        self.kind
    }
}

impl fmt::Display for MediaExecutionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Breeze-TTS execution failed ({:?}, {:?})",
            self.disposition, self.kind
        )
    }
}
impl Error for MediaExecutionFailure {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaArtifactKind {
    FinalWave,
    StreamingChunk,
}

/// A validated path relative to the in-memory output root. The root is not
/// included in persistent job metadata; callers can persist this relative ref.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaArtifact {
    relative_path: PathBuf,
    kind: MediaArtifactKind,
}

impl MediaArtifact {
    pub fn relative_path(&self) -> &Path {
        &self.relative_path
    }

    pub fn kind(&self) -> MediaArtifactKind {
        self.kind
    }
}

/// Successful native output. Its `output_root` is in-memory routing metadata;
/// artifact references themselves are deterministic relative paths.
pub struct MediaExecutionResult {
    output_root: PathBuf,
    mode: SpeechMode,
    artifacts: Vec<MediaArtifact>,
}

impl MediaExecutionResult {
    pub fn output_root(&self) -> &Path {
        &self.output_root
    }

    pub fn mode(&self) -> SpeechMode {
        self.mode
    }

    pub fn artifacts(&self) -> &[MediaArtifact] {
        &self.artifacts
    }
}

impl fmt::Debug for MediaExecutionResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaExecutionResult")
            .field("output_root", &"<private job directory>")
            .field("mode", &self.mode)
            .field("artifact_count", &self.artifacts.len())
            .finish()
    }
}

/// Run one Breeze speech request using a caller-allocated job directory.
/// `job_directory` must already exist, belong to the current UID, and have
/// mode 0700. The result and streaming-chunk targets must not exist yet. No
/// job directory creation, cleanup, default-directory scan, or stdout parsing
/// occurs here.
pub fn execute_breeze_speech(
    catalog: &Catalog,
    request: &InferenceRequest,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    execute_breeze_inner(
        catalog,
        request,
        BreezeBinding {
            route: None,
            device: None,
            hide_cuda: false,
        },
        job_directory,
        context,
        recovery,
    )
}

/// Execute against a complete previously accepted route. The caller must
/// already have passed the coordinator's dispatch gate and retained its
/// ownership/reservation. This checks route freshness, not GPU admission or
/// release; uncertain process cleanup must still quarantine conflicting work.
pub fn execute_breeze_speech_for_route(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    execute_breeze_inner(
        catalog,
        request,
        BreezeBinding {
            route: Some(accepted_route),
            device: None,
            hide_cuda: false,
        },
        job_directory,
        context,
        recovery,
    )
}

/// Execute an accepted CUDA Breeze route on exactly one configured physical
/// device. Native artifacts are candidates only: the owner must separately
/// stop the exact service, prove cgroup/GPU release and revalidate artifacts.
pub fn execute_breeze_speech_on_device(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    device: &CudaDevice,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    if accepted_route.device != Some(Device::Cuda) {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }
    execute_breeze_inner(
        catalog,
        request,
        BreezeBinding {
            route: Some(accepted_route),
            device: Some(device),
            hide_cuda: false,
        },
        job_directory,
        context,
        recovery,
    )
}

/// Execute standard or voice-design Breeze speech explicitly declared for CPU.
///
/// The accepted route must identify `media:breeze-tts-2-cpu` and declare CPU;
/// the child receives an empty CUDA visibility setting and an explicit CPU
/// backend flag. Clone and streaming modes remain unsupported. The output
/// directory remains caller-owned.
pub fn execute_breeze_speech_on_cpu(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    if !is_supported_cpu_breeze_route(request, accepted_route) {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }
    execute_breeze_inner(
        catalog,
        request,
        BreezeBinding {
            route: Some(accepted_route),
            device: None,
            hide_cuda: true,
        },
        job_directory,
        context,
        recovery,
    )
}

struct BreezeBinding<'a> {
    route: Option<&'a ResolvedProfileRoute>,
    device: Option<&'a CudaDevice>,
    hide_cuda: bool,
}

pub(crate) fn is_supported_cpu_breeze_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
) -> bool {
    let InferenceRequest::Media {
        profile,
        input: MediaInput::Speech(input),
    } = request
    else {
        return false;
    };
    matches!(
        &route.native_route,
        NativeRoute::MediaScript { path }
            if path.file_name().and_then(|name| name.to_str()) == Some("generate-breeze-tts-2.sh")
    ) && profile == &route.profile
        && profile.lane == super::catalog::Lane::Media
        && profile.profile_id.as_str() == CPU_BREEZE_PROFILE_ID
        && route.runtime.lane == super::catalog::Lane::Media
        && route.runtime.as_str() == "audio.cpp"
        && route.device.as_ref() == Some(&Device::Cpu)
        && is_supported_cpu_breeze_speech(input)
}

/// Re-read the bounded native artifact contract after the owner has stopped
/// the exact job service and proven its cgroup empty. This function does not
/// establish that prerequisite: callers must hold the directory exclusively
/// and compare these fresh references with their native result candidate.
pub fn revalidate_breeze_artifacts(
    job_directory: &Path,
    mode: SpeechMode,
    context: &OperationContext<'_>,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    let output_root = validate_job_directory(job_directory)?;
    check_after_child(context)?;
    let artifacts = discover_artifacts(&output_root, mode, context)?;
    check_after_child(context)?;
    Ok(MediaExecutionResult {
        output_root,
        mode,
        artifacts,
    })
}

fn execute_breeze_inner(
    catalog: &Catalog,
    request: &InferenceRequest,
    binding: BreezeBinding<'_>,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    if !matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Speech(_),
            ..
        }
    ) {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }
    if request.profile().profile_id.as_str() == CPU_BREEZE_PROFILE_ID && !binding.hide_cuda {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }

    let output_root = validate_job_directory(job_directory)?;
    ensure_absent(&output_root.join(OUTPUT_NAME))?;
    ensure_absent(&output_root.join(CHUNKS_NAME))?;

    let recipe = match binding.route {
        Some(route) => {
            prepare_breeze_speech_recipe_for_route(catalog, request, route, &output_root)
        }
        None => prepare_breeze_speech_recipe(catalog, request, &output_root),
    }
    .map_err(|_| not_dispatched(MediaExecutionErrorKind::InvalidRequest))?;
    let mode = match &recipe.expectation {
        NativeOutputExpectation::BreezeSpeech { mode, .. } => *mode,
        _ => return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest)),
    };

    let mut command = Command::new(&recipe.executable);
    command.args(&recipe.argv).current_dir(&recipe.cwd);
    if let Some(device) = binding.device {
        device.apply_to(&mut command);
    }
    if binding.hide_cuda {
        command.env("CUDA_VISIBLE_DEVICES", "");
    }
    let limits = ProcessLimits::new(0, MAX_CAPTURE_BYTES)
        .map_err(|error| process_failure(error.disposition(), error.kind()))?;
    let output = process::execute(command, None, limits, context, recovery)
        .map_err(|error| process_failure(error.disposition(), error.kind()))?;
    // Captured logs may contain prompts or native paths. They are deliberately
    // not parsed, returned, or included in errors.
    drop(output);

    check_after_child(context)?;
    let artifacts = discover_artifacts(&output_root, mode, context)?;
    check_after_child(context)?;
    Ok(MediaExecutionResult {
        output_root,
        mode,
        artifacts,
    })
}

fn validate_job_directory(path: &Path) -> Result<PathBuf, MediaExecutionFailure> {
    if !path.is_absolute() {
        return Err(not_dispatched(
            MediaExecutionErrorKind::InvalidOutputDirectory,
        ));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| not_dispatched(MediaExecutionErrorKind::InvalidOutputDirectory))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(not_dispatched(
            MediaExecutionErrorKind::InvalidOutputDirectory,
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let uid = unsafe { libc::geteuid() };
        if metadata.uid() != uid || metadata.permissions().mode() & 0o7777 != 0o700 {
            return Err(not_dispatched(
                MediaExecutionErrorKind::InvalidOutputDirectory,
            ));
        }
    }
    #[cfg(not(unix))]
    {
        return Err(not_dispatched(
            MediaExecutionErrorKind::InvalidOutputDirectory,
        ));
    }
    fs::canonicalize(path)
        .map_err(|_| not_dispatched(MediaExecutionErrorKind::InvalidOutputDirectory))
}

fn ensure_absent(path: &Path) -> Result<(), MediaExecutionFailure> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(not_dispatched(MediaExecutionErrorKind::OutputAlreadyExists)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(not_dispatched(
            MediaExecutionErrorKind::InvalidOutputDirectory,
        )),
    }
}

fn discover_artifacts(
    root: &Path,
    mode: SpeechMode,
    context: &OperationContext<'_>,
) -> Result<Vec<MediaArtifact>, MediaExecutionFailure> {
    check_after_child(context)?;
    let final_wave = root.join(OUTPUT_NAME);
    let final_size = validate_wave(&final_wave, root, context)?;
    let mut total_size = final_size;
    let mut artifacts = vec![MediaArtifact {
        relative_path: PathBuf::from(OUTPUT_NAME),
        kind: MediaArtifactKind::FinalWave,
    }];

    if mode == SpeechMode::Streaming {
        let chunks_directory = root.join(CHUNKS_NAME);
        let directory_metadata =
            fs::symlink_metadata(&chunks_directory).map_err(|_| invalid_artifact())?;
        if directory_metadata.file_type().is_symlink() || !directory_metadata.is_dir() {
            return Err(invalid_artifact());
        }
        let canonical_chunks =
            fs::canonicalize(&chunks_directory).map_err(|_| invalid_artifact())?;
        if !canonical_chunks.starts_with(root) {
            return Err(invalid_artifact());
        }

        let entries = fs::read_dir(&chunks_directory).map_err(|_| invalid_artifact())?;
        let mut chunks = Vec::new();
        for entry in entries {
            check_after_child(context)?;
            let entry = entry.map_err(|_| invalid_artifact())?;
            let name = entry.file_name();
            let Some(name_text) = name.to_str().map(str::to_owned) else {
                return Err(invalid_artifact());
            };
            if !safe_chunk_name(&name_text) {
                return Err(invalid_artifact());
            }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(|_| invalid_artifact())?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(invalid_artifact());
            }
            if chunks.len() >= MAX_STREAM_CHUNKS {
                return Err(MediaExecutionFailure {
                    disposition: ProcessFailureDisposition::ChildGroupReaped,
                    kind: MediaExecutionErrorKind::TooManyArtifacts,
                });
            }
            let size = validate_wave(&path, &canonical_chunks, context)?;
            total_size = total_size.saturating_add(size);
            if total_size > MAX_TOTAL_ARTIFACT_BYTES {
                return Err(MediaExecutionFailure {
                    disposition: ProcessFailureDisposition::ChildGroupReaped,
                    kind: MediaExecutionErrorKind::ArtifactTooLarge,
                });
            }
            chunks.push((name, name_text));
        }
        if chunks.is_empty() {
            return Err(invalid_artifact());
        }
        chunks.sort_by(|left, right| left.0.cmp(&right.0));
        artifacts.extend(chunks.into_iter().map(|(_, name)| MediaArtifact {
            relative_path: PathBuf::from(CHUNKS_NAME).join(name),
            kind: MediaArtifactKind::StreamingChunk,
        }));
    } else {
        match fs::symlink_metadata(root.join(CHUNKS_NAME)) {
            Ok(_) => {
                // A non-streaming run has no native contract for chunk output.
                return Err(invalid_artifact());
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(invalid_artifact()),
        }
    }

    check_after_child(context)?;
    Ok(artifacts)
}

fn validate_wave(
    path: &Path,
    allowed_root: &Path,
    context: &OperationContext<'_>,
) -> Result<u64, MediaExecutionFailure> {
    check_after_child(context)?;
    let metadata = fs::symlink_metadata(path).map_err(|_| invalid_artifact())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid_artifact());
    }
    if metadata.len() < 12 {
        return Err(invalid_artifact());
    }
    if metadata.len() > MAX_ARTIFACT_BYTES {
        return Err(MediaExecutionFailure {
            disposition: ProcessFailureDisposition::ChildGroupReaped,
            kind: MediaExecutionErrorKind::ArtifactTooLarge,
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() != 1 {
            return Err(invalid_artifact());
        }
    }
    #[cfg(not(unix))]
    {
        return Err(invalid_artifact());
    }
    check_after_child(context)?;
    let canonical = fs::canonicalize(path).map_err(|_| invalid_artifact())?;
    if !canonical.starts_with(allowed_root) {
        return Err(invalid_artifact());
    }
    check_after_child(context)?;
    let mut file = File::open(&canonical).map_err(|_| invalid_artifact())?;
    validate_wave_container(&mut file, metadata.len(), context)?;
    check_after_child(context)?;
    Ok(metadata.len())
}

/// Validate the native PCM/float RIFF layout without loading sample data.
/// A WAVE marker alone (or an empty data chunk) is not a completed artifact.
/// RF64/BW64 and compressed formats require a separate explicit contract.
fn validate_wave_container(
    file: &mut File,
    length: u64,
    context: &OperationContext<'_>,
) -> Result<(), MediaExecutionFailure> {
    let mut header = [0u8; 12];
    file.read_exact(&mut header)
        .map_err(|_| invalid_artifact())?;
    let big_endian = match &header[..4] {
        b"RIFF" => false,
        b"RIFX" => true,
        _ => return Err(invalid_artifact()),
    };
    let integer = |bytes: &[u8]| -> u32 {
        let bytes: [u8; 4] = bytes.try_into().expect("fixed WAV field");
        if big_endian {
            u32::from_be_bytes(bytes)
        } else {
            u32::from_le_bytes(bytes)
        }
    };
    let short = |bytes: &[u8]| -> u16 {
        let bytes: [u8; 2] = bytes.try_into().expect("fixed WAV field");
        if big_endian {
            u16::from_be_bytes(bytes)
        } else {
            u16::from_le_bytes(bytes)
        }
    };
    if &header[8..] != b"WAVE" || u64::from(integer(&header[4..8])) + 8 != length {
        return Err(invalid_artifact());
    }
    let mut offset = 12u64;
    let mut block_align = None;
    let mut has_data = false;
    for _ in 0..128 {
        check_after_child(context)?;
        if offset == length {
            return if has_data {
                Ok(())
            } else {
                Err(invalid_artifact())
            };
        }
        if length.saturating_sub(offset) < 8 {
            return Err(invalid_artifact());
        }
        let mut chunk = [0u8; 8];
        file.read_exact(&mut chunk)
            .map_err(|_| invalid_artifact())?;
        let size = u64::from(integer(&chunk[4..]));
        let next = offset + 8 + size + size % 2;
        if next > length {
            return Err(invalid_artifact());
        }
        match &chunk[..4] {
            b"fmt " => {
                if block_align.is_some() || size < 16 {
                    return Err(invalid_artifact());
                }
                let mut format = [0u8; 16];
                file.read_exact(&mut format)
                    .map_err(|_| invalid_artifact())?;
                let encoding = short(&format[..2]);
                let channels = short(&format[2..4]);
                let sample_rate = integer(&format[4..8]);
                let byte_rate = integer(&format[8..12]);
                let align = short(&format[12..14]);
                let bits = short(&format[14..]);
                if !matches!(encoding, 1 | 3)
                    || !(1..=8).contains(&channels)
                    || !(8000..=384000).contains(&sample_rate)
                    || !matches!(bits, 8 | 16 | 24 | 32 | 64)
                    || (encoding == 3 && !matches!(bits, 32 | 64))
                    || (encoding == 1 && bits == 64)
                    || align != channels * (bits / 8)
                    || byte_rate != sample_rate * u32::from(align)
                {
                    return Err(invalid_artifact());
                }
                block_align = Some(u64::from(align));
            }
            b"data" => {
                let align = block_align.ok_or_else(invalid_artifact)?;
                if has_data || size == 0 || size % align != 0 {
                    return Err(invalid_artifact());
                }
                has_data = true;
            }
            _ => {}
        }
        file.seek(SeekFrom::Start(next))
            .map_err(|_| invalid_artifact())?;
        offset = next;
    }
    check_after_child(context)?;
    if offset == length && has_data {
        Ok(())
    } else {
        Err(invalid_artifact())
    }
}

fn check_after_child(context: &OperationContext<'_>) -> Result<(), MediaExecutionFailure> {
    let kind = if context.is_cancelled() {
        Some(ProcessFailureKind::Cancelled)
    } else if context.remaining().is_zero() {
        Some(ProcessFailureKind::DeadlineExceeded)
    } else {
        None
    };
    match kind {
        Some(kind) => Err(process_failure(
            ProcessFailureDisposition::ChildGroupReaped,
            kind,
        )),
        None => Ok(()),
    }
}

fn safe_chunk_name(name: &str) -> bool {
    name.len() <= 128
        && name.ends_with(".wav")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        && name != ".wav"
}

fn not_dispatched(kind: MediaExecutionErrorKind) -> MediaExecutionFailure {
    MediaExecutionFailure {
        disposition: ProcessFailureDisposition::NoChildSpawned,
        kind,
    }
}

fn invalid_artifact() -> MediaExecutionFailure {
    MediaExecutionFailure {
        disposition: ProcessFailureDisposition::ChildGroupReaped,
        kind: MediaExecutionErrorKind::InvalidArtifact,
    }
}

fn process_failure(
    disposition: ProcessFailureDisposition,
    kind: ProcessFailureKind,
) -> MediaExecutionFailure {
    MediaExecutionFailure {
        disposition,
        kind: MediaExecutionErrorKind::Process(kind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::catalog::{Lane, ProfileKey};
    use crate::control::requests::{MediaInput, SpeechInput};
    use crate::control::runtime::{Cancellation, NeverCancel};
    use std::fs;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    const FIXTURE_SCRIPT: &str = r#"#!/usr/bin/env bash
set -euo pipefail
output=""
out_dir=""
mode="offline"
text=""
while (($#)); do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --text) text="$2"; shift 2 ;;
    --task|--instruction|--voice-ref|--reference-text) shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$output" && -n "$out_dir" ]]
[[ "$output" == "$out_dir/result.wav" ]]
write_wave() {
  printf 'RIFF\046\000\000\000WAVEfmt \020\000\000\000\001\000\001\000\300\135\000\000\200\273\000\000\002\000\020\000data\002\000\000\000\000\000' > "$1"
}
case "$text" in
  *child-failure*) exit 7 ;;
  child-sleep) sleep 5; write_wave "$output"; exit 0 ;;
  no-output) exit 0 ;;
  empty-output) : > "$output"; exit 0 ;;
  bad-header) printf 'not-a-wave!' > "$output"; exit 0 ;;
  directory-output) mkdir "$output"; exit 0 ;;
  symlink-output) ln -s /etc/passwd "$output"; exit 0 ;;
  bad-chunks)
    write_wave "$output"
    mkdir "$output.chunks"
    ln -s /etc/passwd "$output.chunks/chunk_0001.wav"
    exit 0
    ;;
  many-chunks)
    write_wave "$output"
    mkdir "$output.chunks"
    for i in $(seq -w 1 64); do write_wave "$output.chunks/chunk_${i}.wav"; done
    exit 0
    ;;
esac
write_wave "$output"
if [[ "$mode" == "streaming" ]]; then
  mkdir "$output.chunks"
  write_wave "$output.chunks/chunk_0002.wav"
  write_wave "$output.chunks/chunk_0001.wav"
fi
"#;

    struct Fixture {
        _parent: TempDir,
        language_root: PathBuf,
        media_root: PathBuf,
        decisions_root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let parent = tempfile::tempdir().unwrap();
            let language_root = parent.path().join("language");
            let media_root = parent.path().join("media");
            let decisions_root = parent.path().join("decisions");
            fs::create_dir_all(&language_root).unwrap();
            fs::create_dir_all(media_root.join("media-models")).unwrap();
            fs::create_dir_all(&decisions_root).unwrap();
            fs::write(
                language_root.join("llama-swap.yaml"),
                "models:\n  fixture-model:\n    cmd: fixture\n",
            )
            .unwrap();
            fs::write(
                media_root.join("media-runtimes.json"),
                serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "breeze-tts-2",
                        "name": "Fixture Breeze",
                        "runtime": "audio.cpp",
                        "tasks": ["tts", "clon", "design"],
                        "script": "media-models/generate-breeze-tts-2.sh"
                    }]
                })
                .to_string(),
            )
            .unwrap();
            fs::write(
                media_root.join("media-models/generate-breeze-tts-2.sh"),
                FIXTURE_SCRIPT,
            )
            .unwrap();
            fs::write(
                decisions_root.join("decision-runtimes.json"),
                r#"{"version":"1.0","profiles":[]}"#,
            )
            .unwrap();
            Self {
                _parent: parent,
                language_root,
                media_root,
                decisions_root,
            }
        }

        fn catalog(&self) -> Catalog {
            Catalog::load_from_roots(&self.language_root, &self.media_root, &self.decisions_root)
                .unwrap()
        }

        fn output_directory(&self, name: &str) -> PathBuf {
            let path = self._parent.path().join(name);
            fs::create_dir(&path).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            }
            path
        }
    }

    fn request(mode: SpeechMode, text: &str, fixture: &Fixture) -> InferenceRequest {
        let voice_reference = if mode == SpeechMode::VoiceClone {
            let path = fixture._parent.path().join("speaker.wav");
            fs::write(&path, b"fixture reference").unwrap();
            Some(path)
        } else {
            None
        };
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap(),
            input: MediaInput::Speech(SpeechInput {
                text: text.into(),
                mode,
                instruction: (mode == SpeechMode::VoiceDesign)
                    .then(|| "fixture voice direction".into()),
                voice_reference,
                reference_text: (mode == SpeechMode::VoiceClone)
                    .then(|| "fixture transcript".into()),
            }),
        }
    }

    fn cpu_request(mode: SpeechMode, text: &str, fixture: &Fixture) -> InferenceRequest {
        let mut request = request(mode, text, fixture);
        let InferenceRequest::Media { profile, .. } = &mut request else {
            unreachable!();
        };
        *profile = ProfileKey::new(Lane::Media, CPU_BREEZE_PROFILE_ID).unwrap();
        request
    }

    fn context<'a>(cancel: &'a dyn Cancellation, duration: Duration) -> OperationContext<'a> {
        OperationContext::new(Instant::now() + duration, cancel)
    }

    fn recovery() -> RecoveryPolicy {
        RecoveryPolicy::new(Duration::from_secs(2)).unwrap()
    }

    fn pcm_wave() -> Vec<u8> {
        let mut bytes = vec![0u8; 46];
        bytes[..4].copy_from_slice(b"RIFF");
        bytes[4..8].copy_from_slice(&38u32.to_le_bytes());
        bytes[8..16].copy_from_slice(b"WAVEfmt ");
        bytes[16..20].copy_from_slice(&16u32.to_le_bytes());
        bytes[20..22].copy_from_slice(&1u16.to_le_bytes());
        bytes[22..24].copy_from_slice(&1u16.to_le_bytes());
        bytes[24..28].copy_from_slice(&24000u32.to_le_bytes());
        bytes[28..32].copy_from_slice(&48000u32.to_le_bytes());
        bytes[32..34].copy_from_slice(&2u16.to_le_bytes());
        bytes[34..36].copy_from_slice(&16u16.to_le_bytes());
        bytes[36..40].copy_from_slice(b"data");
        bytes[40..44].copy_from_slice(&2u32.to_le_bytes());
        bytes
    }

    #[test]
    fn validates_pcm_data_not_just_wave_markers_and_rejects_hard_links() {
        let fixture = Fixture::new();
        let root = fixture.output_directory("wave-validation");
        let path = root.join("sample.wav");
        let ctx = context(&NeverCancel, Duration::from_secs(5));
        let valid = pcm_wave();
        fs::write(&path, &valid).unwrap();
        assert_eq!(validate_wave(&path, &root, &ctx).unwrap(), 46);
        let mut empty = valid.clone();
        empty.truncate(44);
        empty[4..8].copy_from_slice(&36u32.to_le_bytes());
        empty[40..44].copy_from_slice(&0u32.to_le_bytes());
        let mut malformed_align = valid.clone();
        malformed_align[32..34].copy_from_slice(&0u16.to_le_bytes());
        let mut odd_data = valid.clone();
        odd_data[40..44].copy_from_slice(&1u32.to_le_bytes());
        for bad in [
            b"RIFF0000WAVE".to_vec(),
            valid[..45].to_vec(),
            empty,
            malformed_align,
            odd_data,
        ] {
            fs::write(&path, bad).unwrap();
            assert_eq!(
                validate_wave(&path, &root, &ctx).unwrap_err().kind(),
                MediaExecutionErrorKind::InvalidArtifact
            );
        }
        fs::write(&path, valid).unwrap();
        fs::hard_link(&path, root.join("duplicate.wav")).unwrap();
        assert_eq!(
            validate_wave(&path, &root, &ctx).unwrap_err().kind(),
            MediaExecutionErrorKind::InvalidArtifact
        );
    }

    fn run(
        fixture: &Fixture,
        mode: SpeechMode,
        text: &str,
        directory_name: &str,
        cancel: &dyn Cancellation,
        duration: Duration,
    ) -> Result<MediaExecutionResult, MediaExecutionFailure> {
        execute_breeze_speech(
            &fixture.catalog(),
            &request(mode, text, fixture),
            &fixture.output_directory(directory_name),
            &context(cancel, duration),
            recovery(),
        )
    }

    #[test]
    fn fresh_artifact_validation_rejects_mutation_after_native_result() {
        let fixture = Fixture::new();
        let output = fixture.output_directory("revalidate");
        let ctx = context(&NeverCancel, Duration::from_secs(5));
        let candidate = execute_breeze_speech(
            &fixture.catalog(),
            &request(SpeechMode::Standard, "fixture speech", &fixture),
            &output,
            &ctx,
            recovery(),
        )
        .unwrap();
        let fresh = revalidate_breeze_artifacts(&output, candidate.mode(), &ctx).unwrap();
        assert_eq!(candidate.artifacts(), fresh.artifacts());
        fs::write(output.join(OUTPUT_NAME), b"truncated wave").unwrap();
        assert_eq!(
            revalidate_breeze_artifacts(&output, candidate.mode(), &ctx)
                .unwrap_err()
                .kind(),
            MediaExecutionErrorKind::InvalidArtifact
        );
    }

    #[test]
    fn accepted_cuda_breeze_passes_exact_physical_uuid_and_rejects_cpu_route() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let manifest_path = fixture.media_root.join("media-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = serde_json::json!("cuda");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        fs::write(
            fixture
                .media_root
                .join("media-models/generate-breeze-tts-2.sh"),
            format!("[ \"$CUDA_VISIBLE_DEVICES\" = '{UUID}' ] || exit 8\n{FIXTURE_SCRIPT}"),
        )
        .unwrap();
        let catalog = fixture.catalog();
        let request = request(SpeechMode::Standard, "fixture speech", &fixture);
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
        let device = CudaDevice::new(UUID).unwrap();
        let output = fixture.output_directory("cuda-selected");
        let ctx = context(&NeverCancel, Duration::from_secs(5));
        let result = execute_breeze_speech_on_device(
            &catalog,
            &request,
            &accepted,
            &device,
            &output,
            &ctx,
            recovery(),
        )
        .unwrap();
        assert_eq!(result.artifacts().len(), 1);
        let mut cpu = accepted.clone();
        cpu.device = Some(Device::Cpu);
        let rejected_output = fixture.output_directory("cpu-rejected");
        let error = execute_breeze_speech_on_device(
            &catalog,
            &request,
            &cpu,
            &device,
            &rejected_output,
            &ctx,
            recovery(),
        )
        .unwrap_err();
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert!(!rejected_output.join(OUTPUT_NAME).exists());
        manifest["profiles"][0]["device"] = serde_json::json!("cpu");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let error = execute_breeze_speech_on_device(
            &fixture.catalog(),
            &request,
            &accepted,
            &device,
            &rejected_output,
            &ctx,
            recovery(),
        )
        .unwrap_err();
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert!(!rejected_output.join(OUTPUT_NAME).exists());
    }

    #[test]
    fn cpu_breeze_standard_and_design_are_private_cpu_runs_and_other_modes_are_rejected() {
        let fixture = Fixture::new();
        let manifest_path = fixture.media_root.join("media-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["id"] = serde_json::json!(CPU_BREEZE_PROFILE_ID);
        manifest["profiles"][0]["device"] = serde_json::json!("cpu");
        manifest["profiles"][0]["tasks"] = serde_json::json!(["tts", "design"]);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let catalog = fixture.catalog();
        let accepted = crate::control::routes::resolve_profile_route(
            &catalog,
            &ProfileKey::new(Lane::Media, CPU_BREEZE_PROFILE_ID).unwrap(),
        )
        .unwrap();
        let script_path = fixture
            .media_root
            .join("media-models/generate-breeze-tts-2.sh");
        for (mode, name, instruction_check) in [
            (SpeechMode::Standard, "cpu-standard", ""),
            (
                SpeechMode::VoiceDesign,
                "cpu-design",
                "[[ \"$*\" == *\"--instruction fixture voice direction\"* ]] || exit 10\n",
            ),
        ] {
            fs::write(
                &script_path,
                format!(
                    "[[ \"${{CUDA_VISIBLE_DEVICES-unset}}\" == \"\" ]] || exit 8\n[[ \"$*\" == *\"--backend cpu\"* ]] || exit 9\n{instruction_check}{FIXTURE_SCRIPT}"
                ),
            )
            .unwrap();
            let request = cpu_request(mode, "fixture speech", &fixture);
            let output = fixture.output_directory(name);
            let result = execute_breeze_speech_on_cpu(
                &catalog,
                &request,
                &accepted,
                &output,
                &context(&NeverCancel, Duration::from_secs(5)),
                recovery(),
            )
            .unwrap();
            assert_eq!(result.mode(), mode);
            assert_eq!(result.artifacts().len(), 1);
            assert_eq!(
                result.artifacts()[0].relative_path(),
                Path::new(OUTPUT_NAME)
            );
            assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::FinalWave);
        }

        for (mode, name) in [
            (SpeechMode::VoiceClone, "cpu-clone-rejected"),
            (SpeechMode::Streaming, "cpu-streaming-rejected"),
        ] {
            let request = cpu_request(mode, "fixture speech", &fixture);
            let output = fixture.output_directory(name);
            let error = execute_breeze_speech_on_cpu(
                &catalog,
                &request,
                &accepted,
                &output,
                &context(&NeverCancel, Duration::from_secs(5)),
                recovery(),
            )
            .expect_err("CPU executor keeps clone and streaming unsupported");
            assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::NoChildSpawned
            );
            assert_eq!(fs::read_dir(&output).unwrap().count(), 0);
        }

        let mut wrong_device = accepted.clone();
        wrong_device.device = Some(Device::Cuda);
        let rejected_output = fixture.output_directory("cpu-route-rejected");
        let design_request = cpu_request(SpeechMode::VoiceDesign, "fixture speech", &fixture);
        let error = execute_breeze_speech_on_cpu(
            &catalog,
            &design_request,
            &wrong_device,
            &rejected_output,
            &context(&NeverCancel, Duration::from_secs(5)),
            recovery(),
        )
        .expect_err("CPU executor rejects a non-CPU accepted route");
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(fs::read_dir(&rejected_output).unwrap().count(), 0);

        manifest["profiles"][0]["tasks"] = serde_json::json!(["tts"]);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let drift_output = fixture.output_directory("cpu-design-route-drift");
        let error = execute_breeze_speech_on_cpu(
            &fixture.catalog(),
            &design_request,
            &accepted,
            &drift_output,
            &context(&NeverCancel, Duration::from_secs(5)),
            recovery(),
        )
        .expect_err("accepted design task drift fails before child spawn");
        assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(fs::read_dir(&drift_output).unwrap().count(), 0);
    }

    #[test]
    fn accepted_route_executes_once_and_rejects_refreshed_route_drift_before_spawn() {
        use crate::control::routes::resolve_profile_route;

        let fixture = Fixture::new();
        let catalog = fixture.catalog();
        let request = request(SpeechMode::Standard, "fixture speech", &fixture);
        let accepted = resolve_profile_route(&catalog, request.profile()).unwrap();
        let result = execute_breeze_speech_for_route(
            &catalog,
            &request,
            &accepted,
            &fixture.output_directory("accepted"),
            &context(&NeverCancel, Duration::from_secs(5)),
            recovery(),
        )
        .unwrap();
        assert_eq!(result.artifacts().len(), 1);

        let manifest_path = fixture.media_root.join("media-runtimes.json");
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        for (field, replacement) in [
            ("runtime", serde_json::json!("different-runtime")),
            ("device", serde_json::json!("cpu")),
            ("script", serde_json::json!("media-models/changed.sh")),
            ("tasks", serde_json::json!(["tts"])),
        ] {
            let mut changed = original.clone();
            changed["profiles"][0][field] = replacement;
            fs::write(&manifest_path, serde_json::to_vec(&changed).unwrap()).unwrap();
            // A caller with a refreshed catalog must still retain the old
            // acceptance route, not silently substitute the refreshed one.
            let refreshed = fixture.catalog();
            let output = fixture.output_directory(&format!("drift-{field}"));
            let error = execute_breeze_speech_for_route(
                &refreshed,
                &request,
                &accepted,
                &output,
                &context(&NeverCancel, Duration::from_secs(5)),
                recovery(),
            )
            .unwrap_err();
            assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::NoChildSpawned
            );
            assert_eq!(fs::read_dir(&output).unwrap().count(), 0);
        }
    }

    #[test]
    fn executes_all_four_modes_with_explicit_private_targets_and_sorted_chunks() {
        let fixture = Fixture::new();
        let cancel = NeverCancel;
        for (mode, name) in [
            (SpeechMode::Standard, "standard"),
            (SpeechMode::VoiceDesign, "design"),
            (SpeechMode::VoiceClone, "clone"),
        ] {
            let result = run(
                &fixture,
                mode,
                "fixture speech",
                name,
                &cancel,
                Duration::from_secs(5),
            )
            .unwrap();
            assert_eq!(result.mode(), mode);
            assert_eq!(result.artifacts().len(), 1);
            assert_eq!(
                result.artifacts()[0].relative_path(),
                Path::new(OUTPUT_NAME)
            );
            assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::FinalWave);
            assert!(result
                .output_root()
                .join(result.artifacts()[0].relative_path())
                .is_file());
        }

        let streaming = run(
            &fixture,
            SpeechMode::Streaming,
            "fixture speech",
            "streaming",
            &cancel,
            Duration::from_secs(5),
        )
        .unwrap();
        let paths = streaming
            .artifacts()
            .iter()
            .map(|artifact| artifact.relative_path().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            [
                "result.wav",
                "result.wav.chunks/chunk_0001.wav",
                "result.wav.chunks/chunk_0002.wav"
            ]
        );
        assert!(streaming
            .artifacts()
            .iter()
            .skip(1)
            .all(|artifact| artifact.kind() == MediaArtifactKind::StreamingChunk));
    }

    #[test]
    fn rejects_bad_or_missing_output_files_without_returning_artifacts() {
        let fixture = Fixture::new();
        for (marker, mode) in [
            ("no-output", SpeechMode::Standard),
            ("empty-output", SpeechMode::Standard),
            ("bad-header", SpeechMode::Standard),
            ("directory-output", SpeechMode::Standard),
            ("symlink-output", SpeechMode::Standard),
            ("bad-chunks", SpeechMode::Streaming),
            ("many-chunks", SpeechMode::Streaming),
        ] {
            let error = run(
                &fixture,
                mode,
                marker,
                marker,
                &NeverCancel,
                Duration::from_secs(5),
            )
            .unwrap_err();
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::ChildGroupReaped,
                "{marker}"
            );
            assert!(matches!(
                error.kind(),
                MediaExecutionErrorKind::InvalidArtifact
                    | MediaExecutionErrorKind::ArtifactTooLarge
                    | MediaExecutionErrorKind::TooManyArtifacts
            ));
        }
    }

    #[test]
    fn requires_a_fresh_current_uid_private_directory_before_spawn() {
        let fixture = Fixture::new();
        let public = fixture.output_directory("public");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&public, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let error = execute_breeze_speech(
            &fixture.catalog(),
            &request(SpeechMode::Standard, "fixture speech", &fixture),
            &public,
            &context(&NeverCancel, Duration::from_secs(5)),
            recovery(),
        )
        .unwrap_err();
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(
            error.kind(),
            MediaExecutionErrorKind::InvalidOutputDirectory
        );

        let occupied = fixture.output_directory("occupied");
        fs::write(occupied.join(OUTPUT_NAME), b"preexisting").unwrap();
        let error = execute_breeze_speech(
            &fixture.catalog(),
            &request(SpeechMode::Standard, "fixture speech", &fixture),
            &occupied,
            &context(&NeverCancel, Duration::from_secs(5)),
            recovery(),
        )
        .unwrap_err();
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(error.kind(), MediaExecutionErrorKind::OutputAlreadyExists);
    }

    #[test]
    fn child_failure_is_redacted_and_preserves_reaped_disposition() {
        let fixture = Fixture::new();
        let error = run(
            &fixture,
            SpeechMode::Standard,
            "private prompt child-failure",
            "failure",
            &NeverCancel,
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(
            error.kind(),
            MediaExecutionErrorKind::Process(ProcessFailureKind::ChildExitedUnsuccessfully {
                code: Some(7)
            })
        );
        let display = error.to_string();
        assert!(!display.contains("private prompt"));
        assert!(!display.contains("failure"));
    }

    struct SharedCancellation(Arc<AtomicBool>);
    impl Cancellation for SharedCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[test]
    fn cancellation_and_deadline_clean_up_the_owned_child_group() {
        let fixture = Fixture::new();
        let flag = Arc::new(AtomicBool::new(false));
        let cancel = SharedCancellation(flag.clone());
        let signal = thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            flag.store(true, Ordering::Release);
        });
        let cancelled = run(
            &fixture,
            SpeechMode::Standard,
            "child-sleep",
            "cancelled",
            &cancel,
            Duration::from_secs(5),
        )
        .unwrap_err();
        signal.join().unwrap();
        assert_eq!(
            cancelled.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(
            cancelled.kind(),
            MediaExecutionErrorKind::Process(ProcessFailureKind::Cancelled)
        );

        let deadline = run(
            &fixture,
            SpeechMode::Standard,
            "child-sleep",
            "deadline",
            &NeverCancel,
            Duration::from_millis(200),
        )
        .unwrap_err();
        assert_eq!(
            deadline.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(
            deadline.kind(),
            MediaExecutionErrorKind::Process(ProcessFailureKind::DeadlineExceeded)
        );
    }
}
