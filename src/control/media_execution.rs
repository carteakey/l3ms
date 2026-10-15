//! Bounded execution and local artifact validation for native media.
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
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::de::{DeserializeSeed, Error as DeError, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use super::catalog::{Catalog, Device};
use super::cuda_device::CudaDevice;
use super::language::SpeechAudio;
use super::native_commands::{
    is_supported_blender_request_route, is_supported_cpu_breeze_speech,
    is_supported_h3_request_route, is_supported_heartmula_request_route,
    is_supported_ltx_request_route, prepare_blender_recipe_for_route, prepare_breeze_speech_recipe,
    prepare_breeze_speech_recipe_for_route, prepare_h3_recipe_for_route,
    prepare_heartmula_music_recipe_for_route, prepare_ltx_recipe_for_route,
    NativeOutputExpectation, CPU_BREEZE_PROFILE_ID,
};
use super::process::{self, ProcessFailureDisposition, ProcessFailureKind, ProcessLimits};
use super::requests::{InferenceRequest, MediaInput, SpeechMode};
use super::routes::{NativeRoute, ResolvedProfileRoute};
use super::runtime::{OperationContext, RecoveryPolicy};

const OUTPUT_NAME: &str = "result.wav";
const CHUNKS_NAME: &str = "result.wav.chunks";
const H3_RAW_VIDEO_NAME: &str = "minimax_h3_video_rgb24.json";
const H3_VIDEO_NAME: &str = "result.mp4";
const LTX_VIDEO_NAME: &str = H3_VIDEO_NAME;
const BLENDER_IMAGE_NAME: &str = "result_0001.png";
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
            "Native media execution failed ({:?}, {:?})",
            self.disposition, self.kind
        )
    }
}
impl Error for MediaExecutionFailure {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaArtifactKind {
    FinalWave,
    StreamingChunk,
    RawRgb24Video,
    Video,
    Image,
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

/// Execute HeartMuLa music for one exact accepted CUDA route. The caller must
/// establish GPU admission and retain its ownership through process cleanup;
/// this function only pins child visibility and validates the final WAV.
pub(crate) fn execute_heartmula_music_on_device(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    device: &CudaDevice,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    if !is_supported_heartmula_request_route(request, accepted_route) {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }

    let output_root = validate_job_directory(job_directory)?;
    ensure_absent(&output_root.join(OUTPUT_NAME))?;
    ensure_absent(&output_root.join(CHUNKS_NAME))?;
    let recipe = prepare_heartmula_music_recipe_for_route(
        catalog,
        request,
        accepted_route,
        &output_root.join(OUTPUT_NAME),
    )
    .map_err(|_| not_dispatched(MediaExecutionErrorKind::InvalidRequest))?;

    execute_script_recipe(
        &recipe,
        &output_root,
        Some(device),
        context,
        recovery,
        ScriptMode::Default,
    )?;
    // The shared validator's Standard mode means one final WAV and no chunks;
    // it does not imply that the HeartMuLa result is speech.
    let artifacts = discover_artifacts(&output_root, SpeechMode::Standard, context)?;
    check_after_child(context)?;
    Ok(MediaExecutionResult {
        output_root,
        mode: SpeechMode::Standard,
        artifacts,
    })
}

/// Run the declared MiniMax-H3 audio or synchronized audio/video script on its
/// exact accepted CUDA route. GPU admission/release remains with the caller.
pub(crate) fn execute_h3_on_device(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    device: &CudaDevice,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    if !is_supported_h3_request_route(request, accepted_route) {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }
    let InferenceRequest::Media {
        input: MediaInput::Script { task, .. },
        ..
    } = request
    else {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    };
    let video_task = task.as_str() == "video";
    let output_root = validate_job_directory(job_directory)?;
    for name in [OUTPUT_NAME, CHUNKS_NAME, H3_RAW_VIDEO_NAME, H3_VIDEO_NAME] {
        ensure_absent(&output_root.join(name))?;
    }
    let video_output = video_task.then(|| output_root.join(H3_VIDEO_NAME));
    let recipe = prepare_h3_recipe_for_route(
        catalog,
        request,
        accepted_route,
        &output_root.join(OUTPUT_NAME),
        video_output.as_deref(),
    )
    .map_err(|_| not_dispatched(MediaExecutionErrorKind::InvalidRequest))?;
    execute_script_recipe(
        &recipe,
        &output_root,
        Some(device),
        context,
        recovery,
        ScriptMode::Default,
    )?;

    let artifacts = discover_h3_artifacts(&output_root, video_task, context)?;
    check_after_child(context)?;
    Ok(MediaExecutionResult {
        output_root,
        mode: SpeechMode::Standard,
        artifacts,
    })
}

/// Run one LTX-2.5 synchronized audio/video request on its exact accepted CUDA
/// route. The prepared-only script path must use the already-installed Python
/// interpreter and weights; this function does not prepare or download either.
pub(crate) fn execute_ltx_on_device(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    device: &CudaDevice,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    if !is_supported_ltx_request_route(request, accepted_route) {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }
    let output_root = validate_job_directory(job_directory)?;
    let output_path = output_root.join(LTX_VIDEO_NAME);
    ensure_absent(&output_path)?;
    let recipe = prepare_ltx_recipe_for_route(catalog, request, accepted_route, &output_path)
        .map_err(|_| not_dispatched(MediaExecutionErrorKind::InvalidRequest))?;

    execute_script_recipe(
        &recipe,
        &output_root,
        Some(device),
        context,
        recovery,
        ScriptMode::LtxPrepared,
    )?;
    let artifacts = discover_ltx_artifacts(&output_root, context)?;
    check_after_child(context)?;
    Ok(MediaExecutionResult {
        output_root,
        // The result type predates synchronized video. Callers distinguish it
        // by its single Video artifact, not this speech-mode compatibility tag.
        mode: SpeechMode::Standard,
        artifacts,
    })
}

fn execute_script_recipe(
    recipe: &super::native_commands::NativeCommandRecipe,
    output_root: &Path,
    device: Option<&CudaDevice>,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
    mode: ScriptMode,
) -> Result<(), MediaExecutionFailure> {
    let mut command = Command::new(&recipe.executable);
    command
        .args(&recipe.argv)
        .current_dir(&recipe.cwd)
        // HeartMuLa/H3 wrappers create defaults even for explicit targets; keep
        // output and the H3 muxer's temporary raw stream in this private job.
        .env("L3MS_MEDIA_OUTPUT_DIR", output_root)
        .env("TMPDIR", output_root);
    match mode {
        ScriptMode::Default => {}
        ScriptMode::LtxPrepared => {
            command.env("L3MS_LTX_PREPARED_ONLY", "1");
        }
        ScriptMode::BlenderHeadless => {
            command.env("L3MS_BLENDER_HEADLESS_ONLY", "1");
        }
    }
    if let Some(device) = device {
        device.apply_to(&mut command);
    }
    let limits = ProcessLimits::new(0, MAX_CAPTURE_BYTES)
        .map_err(|error| process_failure(error.disposition(), error.kind()))?;
    let output = process::execute(command, None, limits, context, recovery)
        .map_err(|error| process_failure(error.disposition(), error.kind()))?;
    // Native output may contain prompts, paths, or result metadata.
    drop(output);
    check_after_child(context)
}

#[derive(Clone, Copy)]
enum ScriptMode {
    Default,
    LtxPrepared,
    BlenderHeadless,
}

/// Run one accepted Blender headless action. This executes trusted local
/// Python without a filesystem or network sandbox; code/script actions return
/// no artifact claim, while render/image actions validate only the fixed PNG.
pub(crate) fn execute_blender_headless(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    job_directory: &Path,
    context: &OperationContext<'_>,
    recovery: RecoveryPolicy,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    if !is_supported_blender_request_route(request, accepted_route) {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    }
    let InferenceRequest::Media {
        input: MediaInput::Script { task, .. },
        ..
    } = request
    else {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    };
    let render_task = matches!(task.as_str(), "render" | "image");
    let output_root = validate_job_directory(job_directory)?;
    if render_task {
        ensure_absent(&output_root.join(BLENDER_IMAGE_NAME))?;
    }
    let recipe = prepare_blender_recipe_for_route(catalog, request, accepted_route, &output_root)
        .map_err(|_| not_dispatched(MediaExecutionErrorKind::InvalidRequest))?;
    execute_script_recipe(
        &recipe,
        &output_root,
        None,
        context,
        recovery,
        ScriptMode::BlenderHeadless,
    )?;

    let artifacts = if render_task {
        validate_blender_png(&output_root.join(BLENDER_IMAGE_NAME), &output_root, context)?;
        vec![MediaArtifact {
            relative_path: PathBuf::from(BLENDER_IMAGE_NAME),
            kind: MediaArtifactKind::Image,
        }]
    } else {
        Vec::new()
    };
    check_after_child(context)?;
    Ok(MediaExecutionResult {
        output_root,
        // This compatibility field predates non-speech media. Callers
        // distinguish Blender results by their image artifact, if any.
        mode: SpeechMode::Standard,
        artifacts,
    })
}

/// Execute an explicitly supported Breeze speech mode on a declared CPU route.
///
/// The accepted route must identify `media:breeze-tts-2-cpu` and declare CPU;
/// the child receives an empty CUDA visibility setting and an explicit CPU
/// backend flag. Streaming chunk references are returned only after native
/// completion; this is not live audio delivery. The output directory remains
/// caller-owned.
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
            hide_cuda: true,
        },
        job_directory,
        context,
        recovery,
    )
}

struct BreezeBinding<'a> {
    route: Option<&'a ResolvedProfileRoute>,
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

/// Publish speech that llama-swap's `/audiocpp/` route returned into a
/// caller-allocated job directory. The files and validation are exactly those
/// of a script run (`result.wav`, plus `result.wav.chunks/` when streaming), so
/// every consumer of `MediaExecutionResult` is unchanged. Nothing is written
/// until the directory and target names pass the same checks.
pub(crate) fn publish_routed_speech(
    request: &InferenceRequest,
    audio: &SpeechAudio,
    job_directory: &Path,
    context: &OperationContext<'_>,
) -> Result<MediaExecutionResult, MediaExecutionFailure> {
    let InferenceRequest::Media {
        input: MediaInput::Speech(input),
        ..
    } = request
    else {
        return Err(not_dispatched(MediaExecutionErrorKind::InvalidRequest));
    };
    let mode = input.mode;
    if (mode == SpeechMode::Streaming) != !audio.chunks.is_empty() {
        return Err(invalid_artifact());
    }
    let output_root = validate_job_directory(job_directory)?;
    ensure_absent(&output_root.join(OUTPUT_NAME))?;
    ensure_absent(&output_root.join(CHUNKS_NAME))?;
    write_private_file(&output_root.join(OUTPUT_NAME), &audio.wave)?;
    if mode == SpeechMode::Streaming {
        let chunks_directory = output_root.join(CHUNKS_NAME);
        create_private_directory(&chunks_directory)?;
        for (index, chunk) in audio.chunks.iter().enumerate() {
            write_private_file(
                &chunks_directory.join(format!("chunk_{:04}.wav", index + 1)),
                chunk,
            )?;
        }
    }
    let artifacts = discover_artifacts(&output_root, mode, context)?;
    check_after_child(context)?;
    Ok(MediaExecutionResult {
        output_root,
        mode,
        artifacts,
    })
}

fn write_private_file(path: &Path, bytes: &[u8]) -> Result<(), MediaExecutionFailure> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|_| invalid_artifact())?;
    file.write_all(bytes).map_err(|_| invalid_artifact())
}

fn create_private_directory(path: &Path) -> Result<(), MediaExecutionFailure> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new()
        .mode(0o700)
        .create(path)
        .map_err(|_| invalid_artifact())
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

fn discover_ltx_artifacts(
    root: &Path,
    context: &OperationContext<'_>,
) -> Result<Vec<MediaArtifact>, MediaExecutionFailure> {
    check_after_child(context)?;
    validate_mp4(&root.join(LTX_VIDEO_NAME), root, context)?;
    for name in [OUTPUT_NAME, CHUNKS_NAME, H3_RAW_VIDEO_NAME] {
        require_missing_after_child(&root.join(name))?;
    }
    check_after_child(context)?;
    Ok(vec![MediaArtifact {
        relative_path: PathBuf::from(LTX_VIDEO_NAME),
        kind: MediaArtifactKind::Video,
    }])
}

fn discover_h3_artifacts(
    root: &Path,
    video_task: bool,
    context: &OperationContext<'_>,
) -> Result<Vec<MediaArtifact>, MediaExecutionFailure> {
    check_after_child(context)?;
    let wave_path = root.join(OUTPUT_NAME);
    let wave_size = validate_wave(&wave_path, root, context)?;
    let mut total_size = wave_size;
    let mut artifacts = vec![MediaArtifact {
        relative_path: PathBuf::from(OUTPUT_NAME),
        kind: MediaArtifactKind::FinalWave,
    }];
    require_missing_after_child(&root.join(CHUNKS_NAME))?;

    if video_task {
        let raw_path = root.join(H3_RAW_VIDEO_NAME);
        let (raw_size, _decoded_rgb_bytes) = validate_h3_rgb24_json(&raw_path, root, context)?;
        total_size = total_size.saturating_add(raw_size);
        if total_size > MAX_TOTAL_ARTIFACT_BYTES {
            return Err(artifact_too_large());
        }
        artifacts.push(MediaArtifact {
            relative_path: PathBuf::from(H3_RAW_VIDEO_NAME),
            kind: MediaArtifactKind::RawRgb24Video,
        });

        let mp4_path = root.join(H3_VIDEO_NAME);
        match fs::symlink_metadata(&mp4_path) {
            Ok(_) => {
                let mp4_size = validate_mp4(&mp4_path, root, context)?;
                total_size = total_size.saturating_add(mp4_size);
                if total_size > MAX_TOTAL_ARTIFACT_BYTES {
                    return Err(artifact_too_large());
                }
                artifacts.push(MediaArtifact {
                    relative_path: PathBuf::from(H3_VIDEO_NAME),
                    kind: MediaArtifactKind::Video,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(invalid_artifact()),
        }
    } else {
        require_missing_after_child(&root.join(H3_RAW_VIDEO_NAME))?;
        require_missing_after_child(&root.join(H3_VIDEO_NAME))?;
    }

    check_after_child(context)?;
    Ok(artifacts)
}

#[derive(Deserialize)]
struct H3Rgb24Metadata {
    width: u32,
    height: u32,
    frames: u32,
    fps: f64,
}

struct H3Rgb24Json {
    metadata: H3Rgb24Metadata,
    decoded_rgb_bytes: u64,
}

struct H3Rgb24Seed<'ctx, 'op> {
    context: &'ctx OperationContext<'op>,
    interrupted: &'ctx std::cell::Cell<Option<MediaExecutionFailure>>,
}

impl<'de, 'ctx, 'op> DeserializeSeed<'de> for H3Rgb24Seed<'ctx, 'op> {
    type Value = H3Rgb24Json;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(H3Rgb24Visitor {
            context: self.context,
            interrupted: self.interrupted,
        })
    }
}

struct H3Rgb24Visitor<'ctx, 'op> {
    context: &'ctx OperationContext<'op>,
    interrupted: &'ctx std::cell::Cell<Option<MediaExecutionFailure>>,
}

impl<'de, 'ctx, 'op> Visitor<'de> for H3Rgb24Visitor<'ctx, 'op> {
    type Value = H3Rgb24Json;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an H3 RGB24 artifact object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut metadata = None;
        let mut decoded_rgb_bytes = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "meta" => {
                    if metadata.is_some() {
                        return Err(M::Error::custom("duplicate H3 video metadata"));
                    }
                    metadata = Some(map.next_value::<H3Rgb24Metadata>()?);
                }
                "payload_hex" => {
                    if decoded_rgb_bytes.is_some() {
                        return Err(M::Error::custom("duplicate H3 RGB24 payload"));
                    }
                    decoded_rgb_bytes = Some(map.next_value_seed(H3Rgb24HexSeed {
                        context: self.context,
                        interrupted: self.interrupted,
                    })?);
                }
                _ => {
                    let _ = map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
        }
        Ok(H3Rgb24Json {
            metadata: metadata.ok_or_else(|| M::Error::custom("missing H3 video metadata"))?,
            decoded_rgb_bytes: decoded_rgb_bytes
                .ok_or_else(|| M::Error::custom("missing H3 RGB24 payload"))?,
        })
    }
}

struct H3Rgb24HexSeed<'ctx, 'op> {
    context: &'ctx OperationContext<'op>,
    interrupted: &'ctx std::cell::Cell<Option<MediaExecutionFailure>>,
}

impl<'de, 'ctx, 'op> DeserializeSeed<'de> for H3Rgb24HexSeed<'ctx, 'op> {
    type Value = u64;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(H3Rgb24HexVisitor {
            context: self.context,
            interrupted: self.interrupted,
        })
    }
}

struct H3Rgb24HexVisitor<'ctx, 'op> {
    context: &'ctx OperationContext<'op>,
    interrupted: &'ctx std::cell::Cell<Option<MediaExecutionFailure>>,
}

impl H3Rgb24HexVisitor<'_, '_> {
    fn validate<E: DeError>(&self, value: &str) -> Result<u64, E> {
        if !value.len().is_multiple_of(2) {
            return Err(E::custom("odd-length H3 RGB24 hex payload"));
        }
        let mut decoded = 0u64;
        for (index, byte) in value.bytes().enumerate() {
            if !byte.is_ascii_hexdigit() {
                return Err(E::custom("invalid H3 RGB24 hex payload"));
            }
            if index % 2 == 1 {
                decoded += 1;
            }
            if index % 65_536 == 0 {
                if let Err(failure) = check_after_child(self.context) {
                    self.interrupted.set(Some(failure));
                    return Err(E::custom("H3 video validation interrupted"));
                }
            }
        }
        Ok(decoded)
    }
}

impl<'de> Visitor<'de> for H3Rgb24HexVisitor<'_, '_> {
    type Value = u64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded hexadecimal RGB24 payload")
    }

    fn visit_str<E: DeError>(self, value: &str) -> Result<Self::Value, E> {
        self.validate(value)
    }

    fn visit_borrowed_str<E: DeError>(self, value: &'de str) -> Result<Self::Value, E> {
        self.validate(value)
    }

    fn visit_string<E: DeError>(self, value: String) -> Result<Self::Value, E> {
        self.validate(&value)
    }
}

fn validate_h3_rgb24_json(
    path: &Path,
    root: &Path,
    context: &OperationContext<'_>,
) -> Result<(u64, u64), MediaExecutionFailure> {
    let (canonical, file_length) = validate_regular_media_file(path, root, context)?;
    if file_length == 0 {
        return Err(invalid_artifact());
    }
    if file_length > MAX_ARTIFACT_BYTES {
        return Err(artifact_too_large());
    }
    let file = File::open(canonical).map_err(|_| invalid_artifact())?;
    let interrupted = std::cell::Cell::new(None);
    let reader = ContextCheckedReader {
        reader: BufReader::new(file),
        context,
        interrupted: &interrupted,
    };
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let parsed = H3Rgb24Seed {
        context,
        interrupted: &interrupted,
    }
    .deserialize(&mut deserializer);
    if let Some(failure) = interrupted.get() {
        return Err(failure);
    }
    let parsed = parsed.map_err(|_| invalid_artifact())?;
    let end = deserializer.end();
    if let Some(failure) = interrupted.get() {
        return Err(failure);
    }
    end.map_err(|_| invalid_artifact())?;

    let metadata = parsed.metadata;
    if metadata.width == 0
        || metadata.height == 0
        || metadata.frames == 0
        || !metadata.fps.is_finite()
        || metadata.fps <= 0.0
    {
        return Err(invalid_artifact());
    }
    let expected = u64::from(metadata.width)
        .checked_mul(u64::from(metadata.height))
        .and_then(|size| size.checked_mul(u64::from(metadata.frames)))
        .and_then(|size| size.checked_mul(3))
        .ok_or_else(artifact_too_large)?;
    if expected > MAX_TOTAL_ARTIFACT_BYTES {
        return Err(artifact_too_large());
    }
    if parsed.decoded_rgb_bytes != expected {
        return Err(invalid_artifact());
    }
    check_after_child(context)?;
    Ok((file_length, expected))
}

struct ContextCheckedReader<'ctx, 'op> {
    reader: BufReader<File>,
    context: &'ctx OperationContext<'op>,
    interrupted: &'ctx std::cell::Cell<Option<MediaExecutionFailure>>,
}

impl Read for ContextCheckedReader<'_, '_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if let Err(failure) = check_after_child(self.context) {
            self.interrupted.set(Some(failure));
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "H3 artifact validation interrupted",
            ));
        }
        self.reader.read(buffer)
    }
}

fn validate_mp4(
    path: &Path,
    root: &Path,
    context: &OperationContext<'_>,
) -> Result<u64, MediaExecutionFailure> {
    let (canonical, length) = validate_regular_media_file(path, root, context)?;
    if length < 12 {
        return Err(invalid_artifact());
    }
    let mut file = File::open(canonical).map_err(|_| invalid_artifact())?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header)
        .map_err(|_| invalid_artifact())?;
    let brand_box_length = u64::from(u32::from_be_bytes(header[..4].try_into().unwrap()));
    if &header[4..8] != b"ftyp" || brand_box_length < 12 || brand_box_length > length {
        return Err(invalid_artifact());
    }
    check_after_child(context)?;
    Ok(length)
}

fn validate_blender_png(
    path: &Path,
    root: &Path,
    context: &OperationContext<'_>,
) -> Result<(), MediaExecutionFailure> {
    let (canonical, length) = validate_regular_media_file(path, root, context)?;
    if length < 33 {
        return Err(invalid_artifact());
    }
    let mut file = File::open(canonical).map_err(|_| invalid_artifact())?;
    let mut header = [0u8; 33];
    file.read_exact(&mut header)
        .map_err(|_| invalid_artifact())?;
    let width = u32::from_be_bytes(header[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(header[20..24].try_into().unwrap());
    if header[..8] != [137, 80, 78, 71, 13, 10, 26, 10]
        || header[8..12] != 13u32.to_be_bytes()
        || &header[12..16] != b"IHDR"
        || width == 0
        || height == 0
        || width > i32::MAX as u32
        || height > i32::MAX as u32
    {
        return Err(invalid_artifact());
    }
    check_after_child(context)
}

fn validate_regular_media_file(
    path: &Path,
    root: &Path,
    context: &OperationContext<'_>,
) -> Result<(PathBuf, u64), MediaExecutionFailure> {
    check_after_child(context)?;
    let metadata = fs::symlink_metadata(path).map_err(|_| invalid_artifact())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(invalid_artifact());
    }
    if metadata.len() > MAX_ARTIFACT_BYTES {
        return Err(artifact_too_large());
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
    let canonical = fs::canonicalize(path).map_err(|_| invalid_artifact())?;
    if !canonical.starts_with(root) {
        return Err(invalid_artifact());
    }
    check_after_child(context)?;
    Ok((canonical, metadata.len()))
}

fn require_missing_after_child(path: &Path) -> Result<(), MediaExecutionFailure> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(invalid_artifact()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(invalid_artifact()),
    }
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

fn artifact_too_large() -> MediaExecutionFailure {
    MediaExecutionFailure {
        disposition: ProcessFailureDisposition::ChildGroupReaped,
        kind: MediaExecutionErrorKind::ArtifactTooLarge,
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

    const HEARTMULA_FIXTURE_SCRIPT: &str = r#"#!/usr/bin/env bash
set -euo pipefail
output=""
out_dir=""
while (($#)); do
  case "$1" in
    --prompt|--text|--prompt-file|--tags|--lyrics|--lyrics-file|--duration|--duration-seconds|--steps|--num-inference-steps|--seed|--temperature|--top-k|--guidance-scale)
      (($# >= 2)) || exit 91
      shift 2
      ;;
    --instrumental) shift ;;
    --output) output="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    *) exit 92 ;;
  esac
done
[[ "${CUDA_VISIBLE_DEVICES-unset}" == "GPU-01234567-89ab-cdef-0123-456789abcdef" ]] || exit 93
[[ -n "$out_dir" && "$L3MS_MEDIA_OUTPUT_DIR" == "$out_dir" ]] || exit 94
[[ "$output" == "$out_dir/result.wav" ]] || exit 95
printf 'RIFF\046\000\000\000WAVEfmt \020\000\000\000\001\000\001\000\300\135\000\000\200\273\000\000\002\000\020\000data\002\000\000\000\000\000' > "$output"
"#;

    const H3_FIXTURE_SCRIPT: &str = r#"#!/usr/bin/env bash
set -euo pipefail
output=""
out_dir=""
video_output=""
video=false
while (($#)); do
  case "$1" in
    --prompt|--text|--prompt-file|--steps|--num-inference-steps|--seed|--width|--height|--frames|--request-option)
      (($# >= 2)) || exit 91
      shift 2
      ;;
    --video) video=true; shift ;;
    --audio-only) video=false; shift ;;
    --output) output="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --video-output) video_output="$2"; shift 2 ;;
    *) exit 92 ;;
  esac
done
[[ "${CUDA_VISIBLE_DEVICES-unset}" == "GPU-01234567-89ab-cdef-0123-456789abcdef" ]] || exit 93
[[ -n "$out_dir" && "$L3MS_MEDIA_OUTPUT_DIR" == "$out_dir" && "$TMPDIR" == "$out_dir" ]] || exit 94
[[ "$output" == "$out_dir/result.wav" ]] || exit 95
printf 'RIFF\046\000\000\000WAVEfmt \020\000\000\000\001\000\001\000\300\135\000\000\200\273\000\000\002\000\020\000data\002\000\000\000\000\000' > "$output"
if [[ "$video" == true ]]; then
  [[ "$video_output" == "$out_dir/result.mp4" ]] || exit 96
  printf '{"meta":{"width":2,"height":1,"frames":2,"fps":24},"payload_hex":"000102030405060708090a0b"}' > "$out_dir/minimax_h3_video_rgb24.json"
  # H3_MUX_FIXTURE
else
  [[ -z "$video_output" ]] || exit 97
fi
"#;

    const LTX_FIXTURE_SCRIPT: &str = r#"#!/usr/bin/env bash
set -euo pipefail
output=""
prompt=""
while (($#)); do
  case "$1" in
    --prompt|--prompt-file) prompt="$2"; shift 2 ;;
    --frames|--num-frames|--seed|--transformer|--quantization|--offload) shift 2 ;;
    --image|--input-image) shift 4 ;;
    --output) output="$2"; shift 2 ;;
    *) exit 92 ;;
  esac
done
[[ "${CUDA_VISIBLE_DEVICES-unset}" == "GPU-01234567-89ab-cdef-0123-456789abcdef" ]] || exit 93
[[ "${L3MS_LTX_PREPARED_ONLY-unset}" == "1" ]] || exit 94
[[ -n "$L3MS_MEDIA_OUTPUT_DIR" && "$TMPDIR" == "$L3MS_MEDIA_OUTPUT_DIR" ]] || exit 95
[[ "$output" == "$L3MS_MEDIA_OUTPUT_DIR/result.mp4" ]] || exit 96
printf x >> '__ATTEMPT_MARKER__'
if [[ "$prompt" == *child-failure* ]]; then
  printf 'private LTX log'
  exit 9
fi
if [[ "$prompt" == *invalid-mp4* ]]; then
  printf 'not an mp4' > "$output"
  exit 0
fi
printf '\000\000\000\014ftypisom' > "$output"
"#;

    const BLENDER_FIXTURE_SCRIPT: &str = r#"#!/usr/bin/env bash
set -euo pipefail
printf x >> '__ATTEMPT_MARKER__'
[[ "${L3MS_BLENDER_HEADLESS_ONLY-unset}" == "1" ]] || exit 81
[[ "$L3MS_MEDIA_OUTPUT_DIR" == "$TMPDIR" ]] || exit 82
case "$1" in
  --code|--script)
    [[ $# -eq 2 ]] || exit 83
    if [[ "$2" == "failure" ]]; then
      printf 'private blender log' >&2
      exit 9
    fi
    ;;
  --render)
    [[ $# -eq 4 && "$3" == "--output" ]] || exit 84
    [[ "$4" == "$L3MS_MEDIA_OUTPUT_DIR" ]] || exit 85
    case "${2##*/}" in
      missing.png) ;;
      malformed.png) printf 'not a png' > "$4/result_0001.png" ;;
      *) printf '\211PNG\015\012\032\012\000\000\000\015IHDR\000\000\000\002\000\000\000\003\010\002\000\000\000\000\000\000\000\000' > "$4/result_0001.png" ;;
    esac
    ;;
  *) exit 86 ;;
esac
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
                    "profiles": [
                        {
                            "id": "breeze-tts-2",
                            "name": "Fixture Breeze",
                            "runtime": "audio.cpp",
                            "tasks": ["tts", "clon", "design"],
                            "script": "media-models/generate-breeze-tts-2.sh"
                        },
                        {
                            "id": "heartmula-music",
                            "name": "Fixture HeartMuLa",
                            "runtime": "audio.cpp",
                            "device": "cuda",
                            "tasks": ["music"],
                            "script": "media-models/generate-heartmula-music.sh"
                        },
                        {
                            "id": "minimax-h3",
                            "name": "Fixture H3",
                            "runtime": "audio.cpp",
                            "device": "cuda",
                            "tasks": ["music", "tts", "video"],
                            "script": "media-models/generate-minimax-h3.sh"
                        },
                        {
                            "id": "ltx-2.5",
                            "name": "Fixture LTX",
                            "runtime": "LTX-2 Python",
                            "device": "cuda",
                            "tasks": ["video", "music"],
                            "script": "media-models/generate-ltx-2.5.sh"
                        },
                        {
                            "id": "blender-3d",
                            "name": "Fixture Blender",
                            "runtime": "Blender 5.2 / blender-mcp",
                            "tasks": ["3d", "render", "image"],
                            "script": "media-models/generate-blender-3d.sh"
                        }
                    ]
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
                media_root.join("media-models/generate-heartmula-music.sh"),
                HEARTMULA_FIXTURE_SCRIPT,
            )
            .unwrap();
            fs::write(
                media_root.join("media-models/generate-minimax-h3.sh"),
                H3_FIXTURE_SCRIPT,
            )
            .unwrap();
            fs::write(
                media_root.join("media-models/generate-ltx-2.5.sh"),
                "#!/usr/bin/env bash\nexit 97\n",
            )
            .unwrap();
            fs::write(
                media_root.join("media-models/generate-blender-3d.sh"),
                "#!/usr/bin/env bash\nexit 97\n",
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

    fn music_request(args: &[&str]) -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "heartmula-music").unwrap(),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("music").unwrap(),
                args: args.iter().map(|argument| (*argument).to_owned()).collect(),
            },
        }
    }

    fn h3_request(task: &str, args: &[&str]) -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "minimax-h3").unwrap(),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new(task).unwrap(),
                args: args.iter().map(|argument| (*argument).to_owned()).collect(),
            },
        }
    }

    fn ltx_request(task: &str, args: &[&str]) -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "ltx-2.5").unwrap(),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new(task).unwrap(),
                args: args.iter().map(|argument| (*argument).to_owned()).collect(),
            },
        }
    }

    fn blender_request(task: &str, args: &[&str]) -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "blender-3d").unwrap(),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new(task).unwrap(),
                args: args.iter().map(|argument| (*argument).to_owned()).collect(),
            },
        }
    }

    fn install_blender_fixture(fixture: &Fixture, marker: &Path) {
        let script =
            BLENDER_FIXTURE_SCRIPT.replace("__ATTEMPT_MARKER__", &marker.display().to_string());
        let _guard = process::executable_fixture_guard();
        fs::write(
            fixture
                .media_root
                .join("media-models/generate-blender-3d.sh"),
            script,
        )
        .unwrap();
    }

    fn context<'a>(cancel: &'a dyn Cancellation, duration: Duration) -> OperationContext<'a> {
        OperationContext::new(Instant::now() + duration, cancel)
    }

    fn recovery() -> RecoveryPolicy {
        RecoveryPolicy::new(Duration::from_secs(2)).unwrap()
    }

    #[test]
    fn blender_code_and_script_actions_return_no_artifact_and_fail_once() {
        let fixture = {
            let _guard = process::executable_fixture_guard();
            Fixture::new()
        };
        let marker = fixture._parent.path().join("blender-attempts");
        install_blender_fixture(&fixture, &marker);
        let catalog = fixture.catalog();
        let request = blender_request("3d", &["--code", "print('fixture')"]);
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
        let never = NeverCancel;
        let operation = context(&never, Duration::from_secs(5));

        let code_output = fixture.output_directory("blender-code-output");
        let code_result = execute_blender_headless(
            &catalog,
            &request,
            &accepted,
            &code_output,
            &operation,
            recovery(),
        )
        .unwrap();
        assert!(code_result.artifacts().is_empty());
        assert_eq!(fs::read(&marker).unwrap(), b"x");

        let script_path = fixture._parent.path().join("fixture.py");
        fs::write(&script_path, "# fixture\n").unwrap();
        let script_request = blender_request("3d", &["--script", script_path.to_str().unwrap()]);
        let script_output = fixture.output_directory("blender-script-output");
        let script_result = execute_blender_headless(
            &catalog,
            &script_request,
            &accepted,
            &script_output,
            &operation,
            recovery(),
        )
        .unwrap();
        assert!(script_result.artifacts().is_empty());
        assert_eq!(fs::read(&marker).unwrap(), b"xx");

        let failing_request = blender_request("3d", &["--code", "failure"]);
        let failing_output = fixture.output_directory("blender-failure-output");
        let error = execute_blender_headless(
            &catalog,
            &failing_request,
            &accepted,
            &failing_output,
            &operation,
            recovery(),
        )
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            MediaExecutionErrorKind::Process(ProcessFailureKind::ChildExitedUnsuccessfully {
                code: Some(9)
            })
        ));
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(fs::read(&marker).unwrap(), b"xxx");
        assert!(!format!("{error:?}").contains("private blender log"));
        assert!(!error.to_string().contains("private blender log"));
    }

    #[test]
    fn blender_render_returns_only_a_private_png_and_rejects_bad_or_existing_targets() {
        let fixture = {
            let _guard = process::executable_fixture_guard();
            Fixture::new()
        };
        let marker = fixture._parent.path().join("blender-render-attempts");
        install_blender_fixture(&fixture, &marker);
        let catalog = fixture.catalog();
        let accepted = crate::control::routes::resolve_profile_route(
            &catalog,
            &ProfileKey::new(Lane::Media, "blender-3d").unwrap(),
        )
        .unwrap();
        let never = NeverCancel;
        let operation = context(&never, Duration::from_secs(5));

        let valid_scene = fixture._parent.path().join("scene.blend");
        fs::write(&valid_scene, b"fixture scene").unwrap();
        let valid_scene = valid_scene.to_str().unwrap();
        let valid_request = blender_request("render", &["--render", valid_scene]);
        let valid_output = fixture.output_directory("blender-render-output");
        let result = execute_blender_headless(
            &catalog,
            &valid_request,
            &accepted,
            &valid_output,
            &operation,
            recovery(),
        )
        .unwrap();
        assert_eq!(result.artifacts().len(), 1);
        assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::Image);
        assert_eq!(
            result.artifacts()[0].relative_path(),
            Path::new(BLENDER_IMAGE_NAME)
        );
        assert_eq!(fs::read_dir(&valid_output).unwrap().count(), 1);
        assert_eq!(fs::read(&marker).unwrap(), b"x");

        let existing_output = fixture.output_directory("blender-existing-output");
        fs::write(existing_output.join(BLENDER_IMAGE_NAME), b"already here").unwrap();
        let error = execute_blender_headless(
            &catalog,
            &valid_request,
            &accepted,
            &existing_output,
            &operation,
            recovery(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), MediaExecutionErrorKind::OutputAlreadyExists);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(fs::read(&marker).unwrap(), b"x");

        for (name, scene, expected_attempts) in [
            (
                "blender-malformed-output",
                "malformed.png",
                b"xx".as_slice(),
            ),
            ("blender-missing-output", "missing.png", b"xxx".as_slice()),
        ] {
            let scene_path = fixture._parent.path().join(scene);
            fs::write(&scene_path, b"fixture scene").unwrap();
            let scene_path = scene_path.to_str().unwrap();
            let request = blender_request("image", &["--render", scene_path]);
            let output = fixture.output_directory(name);
            let error = execute_blender_headless(
                &catalog,
                &request,
                &accepted,
                &output,
                &operation,
                recovery(),
            )
            .unwrap_err();
            assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidArtifact);
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::ChildGroupReaped
            );
            assert_eq!(fs::read(&marker).unwrap(), expected_attempts);
        }
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

    /// Script execution without a device pin, for fixture tests of the shared
    /// inner executor (production CUDA Breeze goes through the router instead).
    fn execute_breeze_speech(
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
                hide_cuda: false,
            },
            job_directory,
            context,
            recovery,
        )
    }

    fn execute_breeze_speech_for_route(
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
                hide_cuda: false,
            },
            job_directory,
            context,
            recovery,
        )
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

    fn sine_pcm(samples: usize) -> Vec<u8> {
        (0..samples)
            .map(|index| (((index as f32) * 0.3).sin() * 8000.0) as i16)
            .flat_map(i16::to_le_bytes)
            .collect()
    }

    #[test]
    fn routed_speech_uses_script_compatible_names_and_is_fully_validated() {
        use crate::control::language::wav_from_pcm16;
        let fixture = Fixture::new();
        let ctx = context(&NeverCancel, Duration::from_secs(5));
        let (first, second) = (sine_pcm(1200), sine_pcm(800));
        let streaming_audio = SpeechAudio {
            wave: wav_from_pcm16(&[first.clone(), second.clone()].concat()),
            chunks: vec![wav_from_pcm16(&first), wav_from_pcm16(&second)],
        };
        let output = fixture.output_directory("routed-stream");
        let request_streaming = request(SpeechMode::Streaming, "stream", &fixture);
        let result =
            publish_routed_speech(&request_streaming, &streaming_audio, &output, &ctx).unwrap();
        assert_eq!(result.mode(), SpeechMode::Streaming);
        let names: Vec<_> = result
            .artifacts()
            .iter()
            .map(|artifact| artifact.relative_path().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "result.wav",
                "result.wav.chunks/chunk_0001.wav",
                "result.wav.chunks/chunk_0002.wav"
            ]
        );
        // The same files pass the independent fresh re-validation used after script runs.
        assert_eq!(
            revalidate_breeze_artifacts(&output, SpeechMode::Streaming, &ctx)
                .unwrap()
                .artifacts(),
            result.artifacts()
        );

        // A second publish into the same directory never overwrites.
        let error =
            publish_routed_speech(&request_streaming, &streaming_audio, &output, &ctx).unwrap_err();
        assert_eq!(error.kind(), MediaExecutionErrorKind::OutputAlreadyExists);

        // Chunks only belong to streaming, and streaming must carry chunks.
        let standard = request(SpeechMode::Standard, "plain", &fixture);
        let error = publish_routed_speech(
            &standard,
            &streaming_audio,
            &fixture.output_directory("routed-mismatch"),
            &ctx,
        )
        .unwrap_err();
        assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidArtifact);

        // Whatever the router returned is validated like native output.
        let junk = SpeechAudio {
            wave: b"RIFF not a wave".to_vec(),
            chunks: Vec::new(),
        };
        let error = publish_routed_speech(
            &standard,
            &junk,
            &fixture.output_directory("routed-junk"),
            &ctx,
        )
        .unwrap_err();
        assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidArtifact);
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
    fn heartmula_music_runs_once_with_uuid_and_only_private_wav_targets() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let catalog = fixture.catalog();
        let request = music_request(&["--prompt", "fixture ambient song", "--instrumental"]);
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
        let device = CudaDevice::new(UUID).unwrap();
        let output = fixture.output_directory("heartmula-private");
        let context = context(&NeverCancel, Duration::from_secs(5));
        let result = execute_heartmula_music_on_device(
            &catalog,
            &request,
            &accepted,
            &device,
            &output,
            &context,
            recovery(),
        )
        .unwrap();

        assert_eq!(result.mode(), SpeechMode::Standard);
        assert_eq!(result.artifacts().len(), 1);
        assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::FinalWave);
        assert_eq!(
            result.artifacts()[0].relative_path(),
            Path::new(OUTPUT_NAME)
        );
        assert_eq!(result.output_root(), fs::canonicalize(&output).unwrap());
        assert_eq!(
            fs::read_dir(&output).unwrap().count(),
            1,
            "the wrapper output directory remains scoped to this job"
        );

        let bad_args = [
            music_request(&["--prompt"]),
            music_request(&["--prompt", "song", "--output", "/tmp/escape.wav"]),
            music_request(&["--prompt", "song", "--out-dir", "/tmp/escape"]),
            music_request(&["--prompt", "song", "--backend", "cpu"]),
        ];
        for (index, invalid_request) in bad_args.iter().enumerate() {
            let rejected_output = fixture.output_directory(&format!("heartmula-rejected-{index}"));
            let error = execute_heartmula_music_on_device(
                &catalog,
                invalid_request,
                &accepted,
                &device,
                &rejected_output,
                &context,
                recovery(),
            )
            .expect_err("invalid HeartMuLa argument shape must not dispatch");
            assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::NoChildSpawned
            );
            assert_eq!(fs::read_dir(rejected_output).unwrap().count(), 0);
        }

        let mut wrong_device = accepted.clone();
        wrong_device.device = Some(Device::Cpu);
        let route_rejected_output = fixture.output_directory("heartmula-route-rejected");
        let error = execute_heartmula_music_on_device(
            &catalog,
            &request,
            &wrong_device,
            &device,
            &route_rejected_output,
            &context,
            recovery(),
        )
        .expect_err("HeartMuLa must remain bound to its declared CUDA route");
        assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(fs::read_dir(route_rejected_output).unwrap().count(), 0);

        let mut drifted_route = accepted.clone();
        drifted_route.tasks.clear();
        let drift_rejected_output = fixture.output_directory("heartmula-task-drift");
        let error = execute_heartmula_music_on_device(
            &catalog,
            &request,
            &drifted_route,
            &device,
            &drift_rejected_output,
            &context,
            recovery(),
        )
        .expect_err("accepted music task drift must fail before spawn");
        assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(fs::read_dir(drift_rejected_output).unwrap().count(), 0);
    }

    #[test]
    fn heartmula_process_failure_is_not_retried_and_logs_are_not_returned() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let marker = fixture._parent.path().join("heartmula-attempts");
        fs::write(
            fixture
                .media_root
                .join("media-models/generate-heartmula-music.sh"),
            format!(
                "printf x >> '{}'; printf 'private native log'; exit 7\n",
                marker.display()
            ),
        )
        .unwrap();
        let catalog = fixture.catalog();
        let request = music_request(&["--prompt", "fixture ambient song"]);
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
        let output = fixture.output_directory("heartmula-one-attempt");
        let error = execute_heartmula_music_on_device(
            &catalog,
            &request,
            &accepted,
            &CudaDevice::new(UUID).unwrap(),
            &output,
            &context(&NeverCancel, Duration::from_secs(5)),
            recovery(),
        )
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            MediaExecutionErrorKind::Process(ProcessFailureKind::ChildExitedUnsuccessfully {
                code: Some(7)
            })
        ));
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(fs::read(marker).unwrap(), b"x");
        assert_eq!(fs::read_dir(output).unwrap().count(), 0);
        assert!(!format!("{error:?}").contains("private native log"));
        assert!(!error.to_string().contains("private native log"));
    }

    #[test]
    fn h3_audio_tasks_and_video_raw_rgb24_optional_mp4_are_returned() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let device = CudaDevice::new(UUID).unwrap();
        let context = context(&NeverCancel, Duration::from_secs(5));
        for (index, task) in ["music", "tts"].iter().enumerate() {
            let catalog = fixture.catalog();
            let request = h3_request(task, &["--prompt", "fixture audio", "--steps", "2"]);
            let accepted =
                crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
            let output = fixture.output_directory(&format!("h3-audio-{index}"));
            let result = execute_h3_on_device(
                &catalog,
                &request,
                &accepted,
                &device,
                &output,
                &context,
                recovery(),
            )
            .unwrap();
            assert_eq!(result.mode(), SpeechMode::Standard);
            assert_eq!(result.artifacts().len(), 1);
            assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::FinalWave);
            assert_eq!(fs::read_dir(output).unwrap().count(), 1);
        }

        let video_request = h3_request(
            "video",
            &[
                "--prompt",
                "fixture video",
                "--width",
                "2",
                "--height",
                "1",
                "--frames",
                "2",
                "--request-option",
                "dit_layerwise=true",
            ],
        );
        let catalog = fixture.catalog();
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, video_request.profile())
                .unwrap();
        let raw_only = fixture.output_directory("h3-video-raw-only");
        // The wrapper reads these fields, but native artifacts may include
        // producer/layout metadata too. Keep validation strict for the known
        // fields without imposing an unverified closed schema.
        let extended_metadata_script = H3_FIXTURE_SCRIPT.replace(
            "\"meta\":{\"width\":2",
            "\"producer\":{\"fixture\":true},\"meta\":{\"layout\":\"NHWC\",\"width\":2",
        );
        fs::write(
            fixture
                .media_root
                .join("media-models/generate-minimax-h3.sh"),
            extended_metadata_script,
        )
        .unwrap();
        let result = execute_h3_on_device(
            &catalog,
            &video_request,
            &accepted,
            &device,
            &raw_only,
            &context,
            recovery(),
        )
        .unwrap();
        assert_eq!(result.artifacts().len(), 2);
        assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::FinalWave);
        assert_eq!(
            result.artifacts()[1].kind(),
            MediaArtifactKind::RawRgb24Video
        );
        assert_eq!(
            result.artifacts()[1].relative_path(),
            Path::new(H3_RAW_VIDEO_NAME)
        );
        assert_eq!(fs::read_dir(&raw_only).unwrap().count(), 2);

        let with_mp4_script = H3_FIXTURE_SCRIPT.replace(
            "# H3_MUX_FIXTURE",
            r#"printf '\000\000\000\014ftypisom' > "$video_output""#,
        );
        fs::write(
            fixture
                .media_root
                .join("media-models/generate-minimax-h3.sh"),
            with_mp4_script,
        )
        .unwrap();
        let catalog = fixture.catalog();
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, video_request.profile())
                .unwrap();
        let with_mp4 = fixture.output_directory("h3-video-with-mp4");
        let result = execute_h3_on_device(
            &catalog,
            &video_request,
            &accepted,
            &device,
            &with_mp4,
            &context,
            recovery(),
        )
        .unwrap();
        assert_eq!(result.artifacts().len(), 3);
        assert_eq!(result.artifacts()[2].kind(), MediaArtifactKind::Video);
        assert_eq!(
            result.artifacts()[2].relative_path(),
            Path::new(H3_VIDEO_NAME)
        );
        assert_eq!(fs::read_dir(&with_mp4).unwrap().count(), 3);
    }

    #[test]
    fn h3_rejects_bad_hex_dimensions_outputs_and_wrong_routes_before_success() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let video_request = h3_request("video", &["--prompt", "fixture video"]);
        let device = CudaDevice::new(UUID).unwrap();
        let context = context(&NeverCancel, Duration::from_secs(5));

        for (index, (needle, replacement)) in [
            ("000102030405060708090a0b", "000102030405060708090a0g"),
            ("\"width\":2", "\"width\":3"),
            ("\"width\":2", "\"width\":2,\"width\":2"),
            (
                "\"payload_hex\":\"000102030405060708090a0b\"",
                "\"payload_hex\":\"000102030405060708090a0b\",\"payload_hex\":\"000102030405060708090a0b\"",
            ),
            ("\"fps\":24", "\"fps\":0"),
        ]
        .into_iter()
        .enumerate()
        {
            fs::write(
                fixture
                    .media_root
                    .join("media-models/generate-minimax-h3.sh"),
                H3_FIXTURE_SCRIPT.replace(needle, replacement),
            )
            .unwrap();
            let catalog = fixture.catalog();
            let accepted =
                crate::control::routes::resolve_profile_route(&catalog, video_request.profile())
                    .unwrap();
            let output = fixture.output_directory(&format!("h3-invalid-{index}"));
            let error = execute_h3_on_device(
                &catalog,
                &video_request,
                &accepted,
                &device,
                &output,
                &context,
                recovery(),
            )
            .expect_err("malformed H3 video artifact must not be reported as success");
            assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidArtifact);
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::ChildGroupReaped
            );
        }

        fs::write(
            fixture
                .media_root
                .join("media-models/generate-minimax-h3.sh"),
            H3_FIXTURE_SCRIPT,
        )
        .unwrap();
        for (index, args) in [
            vec!["--prompt", "scene", "--video-output", "/tmp/escape.mp4"],
            vec![
                "--prompt",
                "scene",
                "--request-option",
                "return_video=false",
            ],
            vec!["--prompt", "scene", "--request-option", "output=/tmp/x"],
            vec!["--prompt", "scene", "--request-option", "width=32"],
        ]
        .iter()
        .enumerate()
        {
            let request = h3_request("video", args);
            let catalog = fixture.catalog();
            let accepted =
                crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
            let output = fixture.output_directory(&format!("h3-override-{index}"));
            let error = execute_h3_on_device(
                &catalog,
                &request,
                &accepted,
                &device,
                &output,
                &context,
                recovery(),
            )
            .expect_err("caller cannot override H3 task mode or output paths");
            assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
            assert_eq!(
                error.disposition(),
                ProcessFailureDisposition::NoChildSpawned
            );
            assert_eq!(fs::read_dir(output).unwrap().count(), 0);
        }

        let catalog = fixture.catalog();
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, video_request.profile())
                .unwrap();
        let mut wrong_route = accepted.clone();
        wrong_route.device = Some(Device::Cpu);
        let output = fixture.output_directory("h3-wrong-device");
        let error = execute_h3_on_device(
            &catalog,
            &video_request,
            &wrong_route,
            &device,
            &output,
            &context,
            recovery(),
        )
        .expect_err("H3 execution cannot cross a changed CUDA route");
        assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(fs::read_dir(output).unwrap().count(), 0);
    }

    #[test]
    fn h3_process_failure_is_not_retried_and_native_logs_are_redacted() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let marker = fixture._parent.path().join("h3-attempts");
        fs::write(
            fixture
                .media_root
                .join("media-models/generate-minimax-h3.sh"),
            format!(
                "printf x >> '{}'; printf 'private h3 log'; exit 9\n",
                marker.display()
            ),
        )
        .unwrap();
        let catalog = fixture.catalog();
        let request = h3_request("music", &["--prompt", "fixture audio"]);
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
        let output = fixture.output_directory("h3-one-attempt");
        let error = execute_h3_on_device(
            &catalog,
            &request,
            &accepted,
            &CudaDevice::new(UUID).unwrap(),
            &output,
            &context(&NeverCancel, Duration::from_secs(5)),
            recovery(),
        )
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            MediaExecutionErrorKind::Process(ProcessFailureKind::ChildExitedUnsuccessfully {
                code: Some(9)
            })
        ));
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(fs::read(marker).unwrap(), b"x");
        assert!(!format!("{error:?}").contains("private h3 log"));
        assert!(!error.to_string().contains("private h3 log"));
    }

    #[test]
    fn ltx_runs_once_with_private_output_and_returns_only_a_video_artifact() {
        const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";
        let fixture = Fixture::new();
        let marker = fixture._parent.path().join("ltx-attempts");
        let script =
            LTX_FIXTURE_SCRIPT.replace("__ATTEMPT_MARKER__", &marker.display().to_string());
        fs::write(
            fixture.media_root.join("media-models/generate-ltx-2.5.sh"),
            script,
        )
        .unwrap();
        let conditioning = fixture._parent.path().join("conditioning.png");
        fs::write(&conditioning, b"fixture image").unwrap();
        let conditioning = conditioning.to_str().unwrap();
        let request = ltx_request(
            "video",
            &[
                "--prompt",
                "the camera approaches the boat",
                "--input-image",
                conditioning,
                "0",
                "0.8",
                "--frames",
                "121",
            ],
        );
        let catalog = fixture.catalog();
        let accepted =
            crate::control::routes::resolve_profile_route(&catalog, request.profile()).unwrap();
        let device = CudaDevice::new(UUID).unwrap();
        let context = context(&NeverCancel, Duration::from_secs(5));
        let output = fixture.output_directory("ltx-private-output");
        let result = execute_ltx_on_device(
            &catalog,
            &request,
            &accepted,
            &device,
            &output,
            &context,
            recovery(),
        )
        .unwrap();
        assert_eq!(result.artifacts().len(), 1);
        assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::Video);
        assert_eq!(
            result.artifacts()[0].relative_path(),
            Path::new("result.mp4")
        );
        assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
        assert_eq!(fs::read(&marker).unwrap(), b"x");

        let music_request = ltx_request("music", &["--prompt", "ambient orchestral music"]);
        let music_output = fixture.output_directory("ltx-music-output");
        let music_result = execute_ltx_on_device(
            &catalog,
            &music_request,
            &accepted,
            &device,
            &music_output,
            &context,
            recovery(),
        )
        .unwrap();
        assert_eq!(music_result.artifacts().len(), 1);
        assert_eq!(music_result.artifacts()[0].kind(), MediaArtifactKind::Video);
        assert_eq!(fs::read_dir(&music_output).unwrap().count(), 1);
        assert_eq!(fs::read(&marker).unwrap(), b"xx");

        let existing_output = fixture.output_directory("ltx-existing-output");
        fs::write(existing_output.join("result.mp4"), b"existing").unwrap();
        let error = execute_ltx_on_device(
            &catalog,
            &request,
            &accepted,
            &device,
            &existing_output,
            &context,
            recovery(),
        )
        .expect_err("LTX must refuse to clobber an existing result");
        assert_eq!(error.kind(), MediaExecutionErrorKind::OutputAlreadyExists);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::NoChildSpawned
        );
        assert_eq!(fs::read(&marker).unwrap(), b"xx");

        let invalid_request = ltx_request("video", &["--prompt", "invalid-mp4 output"]);
        let invalid_output = fixture.output_directory("ltx-invalid-mp4");
        let error = execute_ltx_on_device(
            &catalog,
            &invalid_request,
            &accepted,
            &device,
            &invalid_output,
            &context,
            recovery(),
        )
        .expect_err("malformed MP4 must not be reported as a video artifact");
        assert_eq!(error.kind(), MediaExecutionErrorKind::InvalidArtifact);
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(fs::read(&marker).unwrap(), b"xxx");

        let failed_request = ltx_request("music", &["--prompt", "child-failure prompt"]);
        let failed_output = fixture.output_directory("ltx-child-failure");
        let error = execute_ltx_on_device(
            &catalog,
            &failed_request,
            &accepted,
            &device,
            &failed_output,
            &context,
            recovery(),
        )
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            MediaExecutionErrorKind::Process(ProcessFailureKind::ChildExitedUnsuccessfully {
                code: Some(9)
            })
        ));
        assert_eq!(
            error.disposition(),
            ProcessFailureDisposition::ChildGroupReaped
        );
        assert_eq!(fs::read(&marker).unwrap(), b"xxxx");
        assert_eq!(fs::read_dir(&failed_output).unwrap().count(), 0);
        assert!(!format!("{error:?}").contains("private LTX log"));
        assert!(!error.to_string().contains("private LTX log"));
    }

    #[test]
    fn cpu_breeze_supported_modes_run_privately_and_invalid_fields_are_rejected() {
        let fixture = Fixture::new();
        let manifest_path = fixture.media_root.join("media-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["id"] = serde_json::json!(CPU_BREEZE_PROFILE_ID);
        manifest["profiles"][0]["device"] = serde_json::json!("cpu");
        manifest["profiles"][0]["tasks"] = serde_json::json!(["tts", "clon", "design"]);
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
            (SpeechMode::VoiceClone, "cpu-clone", ""),
            (SpeechMode::Streaming, "cpu-streaming", ""),
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
            assert_eq!(
                result.artifacts().len(),
                if mode == SpeechMode::Streaming { 3 } else { 1 }
            );
            assert_eq!(
                result.artifacts()[0].relative_path(),
                Path::new(OUTPUT_NAME)
            );
            assert_eq!(result.artifacts()[0].kind(), MediaArtifactKind::FinalWave);
            if mode == SpeechMode::Streaming {
                assert!(result.artifacts()[1..]
                    .iter()
                    .all(|artifact| artifact.kind() == MediaArtifactKind::StreamingChunk));
            }
        }

        for (name, mode) in [
            ("cpu-clone-missing-transcript", SpeechMode::VoiceClone),
            ("cpu-streaming-with-reference", SpeechMode::Streaming),
        ] {
            let mut request = cpu_request(mode, "fixture speech", &fixture);
            let InferenceRequest::Media {
                input: MediaInput::Speech(input),
                ..
            } = &mut request
            else {
                unreachable!();
            };
            if mode == SpeechMode::VoiceClone {
                input.reference_text = None;
            } else {
                input.voice_reference = Some(PathBuf::from("reference.wav"));
            }
            let output = fixture.output_directory(name);
            let error = execute_breeze_speech_on_cpu(
                &catalog,
                &request,
                &accepted,
                &output,
                &context(&NeverCancel, Duration::from_secs(5)),
                recovery(),
            )
            .expect_err("CPU executor rejects mode-specific field conflicts");
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
