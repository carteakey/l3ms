//! Validated, inert command recipes for native media and decision dispatch.
//!
//! This module never spawns a process, creates files, prepares environments,
//! loads weights, or performs GPU work. The caller must own the resource lease
//! and admission decision, persist only bounded metadata, and execute recipes
//! with a bounded runner. Arguments can contain prompts, speech, or structured
//! state, so recipe/argument debug output is redacted and values must never be
//! persisted to job history.
//!
//! Native argv items are capped at 64 KiB and total argv bytes at 256 KiB.
//! This keeps a single speech/script argument below Linux's per-argument
//! limit, but is not a guarantee against platform `ARG_MAX` or a large inherited
//! environment. Decision state/questions avoid argv limits by using bounded
//! stdin instead. Hosted decision credentials and endpoint configuration are
//! intentionally absent from the recipe; the executor must provide any needed
//! values through a controlled environment without logging or persisting them.
//! A decision executor must write the complete stdin payload, close the pipe,
//! and enforce a deadline; this recipe is not a runner or IO timeout.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::{Map, Value};

use super::catalog::{Catalog, Device, Lane, ProfileKey};
use super::native_paths::{verify_native_entrypoint, NativeEntrypointKind, VerifiedNativePath};
use super::requests::{InferenceRequest, MediaInput, SpeechInput, SpeechMode};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};

const BREEZE_PROFILE_ID: &str = "breeze-tts-2";
pub(crate) const CPU_BREEZE_PROFILE_ID: &str = "breeze-tts-2-cpu";
const BREEZE_SCRIPT: &str = "media-models/generate-breeze-tts-2.sh";
const DECISION_PYTHON: &str = ".venv/bin/python";
const HOSTED_DECISION_PYTHON: &str = ".venv-hosted/bin/python";
pub(crate) const HOSTED_DECISION_PYTHON_ENV: &str = "L3MS_HOSTED_DECISIONS_PYTHON";
const BASH_EXECUTABLE: &str = "/bin/bash";
const MAX_NATIVE_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_NATIVE_ARGV_BYTES: usize = 256 * 1024;
const MAX_DECISION_STDIN_BYTES: usize = 1024 * 1024;

/// One argument is a single OS argument, never shell-parsed. Its contents are
/// intentionally hidden in `Debug` because it may be a prompt or credential.
#[derive(Clone, PartialEq, Eq)]
pub struct CommandArgument(OsString);

impl CommandArgument {
    pub fn as_os_str(&self) -> &OsStr {
        &self.0
    }
}

impl AsRef<OsStr> for CommandArgument {
    fn as_ref(&self) -> &OsStr {
        self.as_os_str()
    }
}

impl fmt::Debug for CommandArgument {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CommandArgument(<redacted>)")
    }
}

/// Bounded decision input that the executor writes to the runner's stdin.
/// Content is deliberately omitted from debug output and must never enter
/// persisted job history.
#[derive(Clone, PartialEq, Eq)]
pub struct SensitivePayload(Vec<u8>);

impl SensitivePayload {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for SensitivePayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SensitivePayload(<redacted>)")
    }
}

/// Where the child runner reports or places the native result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreezeArtifacts {
    /// The wrapper writes its final WAV using its native output-directory default.
    FinalWaveAtWrapperDefault,
    /// The wrapper writes a final WAV and a sibling `<output>.chunks` directory.
    FinalWaveAndChunksAtWrapperDefault {
        chunk_directory_suffix: &'static str,
    },
}

/// Lane-specific expectations for interpreting a command's output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeOutputExpectation {
    /// Artifacts are defined by this profile's own native media script.
    MediaScript { task: String },
    /// Breeze's wrapper task alias and output behavior. Voice design uses the
    /// wrapper's `design` alias, which the script maps to audio.cpp `tts`.
    BreezeSpeech {
        wrapper_task: &'static str,
        audio_cpp_task: &'static str,
        mode: SpeechMode,
        artifacts: BreezeArtifacts,
    },
    /// The native decision runner writes its decision result as JSON to stdout.
    /// Type and tier are retained as route metadata; its CLI has no flags for them.
    DecisionJsonStdout {
        engine: String,
        model: String,
        profile_type: String,
        tier: Option<String>,
    },
}

/// An inert argv-based recipe. `argv` contains arguments after argv[0]; the
/// script/runner entrypoint is its first item. No shell-string is constructed.
#[derive(Clone, PartialEq, Eq)]
pub struct NativeCommandRecipe {
    pub executable: PathBuf,
    pub argv: Vec<CommandArgument>,
    pub cwd: PathBuf,
    /// Present only for the L3DS decision runner's bounded `--input-stdin`
    /// contract. The caller must pipe this payload without logging it.
    pub stdin_payload: Option<SensitivePayload>,
    pub expectation: NativeOutputExpectation,
}

impl fmt::Debug for NativeCommandRecipe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeCommandRecipe")
            .field("executable", &self.executable)
            .field("argv", &"<redacted>")
            .field("argv_count", &self.argv.len())
            .field("cwd", &self.cwd)
            .field(
                "stdin_payload",
                &self.stdin_payload.as_ref().map(|_| "<redacted>"),
            )
            .field("expectation", &self.expectation)
            .finish()
    }
}

/// Refresh and validate a request, verify its current native entrypoint, and
/// prepare an argv recipe for a supported media script, Breeze speech mode, or
/// L3DS decision. Catalog route metadata is refreshed again after path
/// verification; a changed runtime identity is rejected so the caller cannot
/// reuse a stale resource reservation.
pub fn prepare_native_command(
    catalog: &Catalog,
    request: &InferenceRequest,
) -> Result<NativeCommandRecipe> {
    prepare_native_command_inner(catalog, request, None)
}

/// Prepare against an already accepted native route. Every refreshed route
/// must match that snapshot, including the declared device. This is not
/// admission or resource ownership; the coordinator must authorize dispatch.
pub fn prepare_native_command_for_route(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted: &ResolvedProfileRoute,
) -> Result<NativeCommandRecipe> {
    prepare_native_command_inner(catalog, request, Some(accepted))
}

fn prepare_native_command_inner(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted: Option<&ResolvedProfileRoute>,
) -> Result<NativeCommandRecipe> {
    let key = request.profile();
    if let Some(accepted) = accepted {
        ensure!(
            resolve_profile_route(catalog, key)? == *accepted,
            "accepted native route changed"
        );
    }
    let cached_profile = catalog
        .inspect(key)
        .with_context(|| format!("unknown profile {key}"))?;
    let reserved_runtime = cached_profile.runtime.clone();
    let reserved_device = cached_profile.device.clone();

    let validated_catalog = catalog
        .refresh()
        .context("refresh native catalog before command preparation")?;
    if let Some(accepted) = accepted {
        ensure!(
            resolve_profile_route(&validated_catalog, key)? == *accepted,
            "accepted native route changed"
        );
    }
    let validated_profile = request
        .validate(&validated_catalog)
        .map_err(|_| anyhow!("native inference request failed validation"))?;
    ensure!(
        validated_profile.runtime == reserved_runtime,
        "runtime identity for {key} changed; executor must rebind and reconcile resources"
    );
    ensure!(
        validated_profile.device == reserved_device,
        "native device changed; executor must rebind and reconcile resources"
    );

    if matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Workflow { .. },
            ..
        }
    ) {
        bail!("workflow inputs require the native ComfyUI HTTP adapter");
    }
    ensure!(
        matches!(
            request,
            InferenceRequest::Media { .. } | InferenceRequest::Decision { .. }
        ),
        "native command preparation supports only media and decision requests"
    );

    let verified = verify_native_entrypoint(&validated_catalog, key)?;

    // Load route fields after verification so model/type/tier are not copied
    // from a stale manifest snapshot. The path must still be the one verified.
    let current_catalog = catalog
        .refresh()
        .context("refresh native route after entrypoint verification")?;
    let profile = request
        .validate(&current_catalog)
        .map_err(|_| anyhow!("native inference request failed validation"))?;
    ensure!(
        profile.runtime == reserved_runtime,
        "runtime identity for {key} changed; executor must rebind and reconcile resources"
    );
    ensure!(
        profile.device == reserved_device,
        "native device changed; executor must rebind and reconcile resources"
    );
    let route = resolve_profile_route(&current_catalog, key)
        .with_context(|| format!("resolve current native route for {key}"))?;
    if let Some(accepted) = accepted {
        ensure!(route == *accepted, "accepted native route changed");
    }
    ensure_route_matches_verified_path(&route, &verified)?;

    match request {
        InferenceRequest::Media {
            input: MediaInput::Script { task, args },
            ..
        } => media_script_recipe(&verified, task.as_str(), args),
        InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } => breeze_speech_recipe(key, &route, &verified, input),
        InferenceRequest::Media {
            input: MediaInput::Workflow { .. },
            ..
        } => unreachable!("workflow requests were rejected above"),
        InferenceRequest::Decision {
            state, questions, ..
        } => decision_recipe(profile, &route, &verified, state, questions),
        InferenceRequest::Chat { .. } | InferenceRequest::Embeddings { .. } => {
            unreachable!("language requests were rejected above")
        }
    }
}

/// Prepare Breeze speech with an explicit caller-owned output directory.
/// Existing callers of `prepare_native_command` retain the native wrapper's
/// default output behavior; the bounded executor uses this helper to prevent
/// timestamp-based writes into a shared inherited output directory.
pub fn prepare_breeze_speech_recipe(
    catalog: &Catalog,
    request: &InferenceRequest,
    output_directory: &std::path::Path,
) -> Result<NativeCommandRecipe> {
    breeze_recipe_with_output(prepare_native_command(catalog, request)?, output_directory)
}

/// Prepare explicit Breeze output against the complete route retained at
/// acceptance. A refreshed device, runtime, task set or entrypoint may not
/// replace that route while keeping the old coordinator reservation.
pub fn prepare_breeze_speech_recipe_for_route(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    output_directory: &std::path::Path,
) -> Result<NativeCommandRecipe> {
    breeze_recipe_with_output(
        prepare_native_command_for_route(catalog, request, accepted_route)?,
        output_directory,
    )
}

/// Prepare the declared HeartMuLa music script against the accepted CUDA route
/// and append only the caller-owned output target. The script's native output
/// aliases are not accepted from request arguments.
pub(crate) fn prepare_heartmula_music_recipe_for_route(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    output_file: &std::path::Path,
) -> Result<NativeCommandRecipe> {
    ensure!(
        is_supported_heartmula_request_route(request, accepted_route),
        "request is not a supported HeartMuLa music route"
    );
    let recipe = prepare_native_command_for_route(catalog, request, accepted_route)?;
    ensure!(
        matches!(
            &recipe.expectation,
            NativeOutputExpectation::MediaScript { task } if task == "music"
        ),
        "HeartMuLa preparation did not resolve to the declared music script"
    );
    append_private_output_controls(recipe, output_file, None)
}

/// Prepare audio.cpp MiniMax-H3 for one declared task, keeping generation mode
/// and output locations under caller control rather than the request payload.
pub(crate) fn prepare_h3_recipe_for_route(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    output_wav: &std::path::Path,
    video_output: Option<&std::path::Path>,
) -> Result<NativeCommandRecipe> {
    ensure!(
        is_supported_h3_request_route(request, accepted_route),
        "request is not a supported MiniMax-H3 route"
    );
    let InferenceRequest::Media {
        input: MediaInput::Script { task, .. },
        ..
    } = request
    else {
        unreachable!("supported H3 route requires script media")
    };
    let video_task = task.as_str() == "video";
    let expected_task = task.as_str();
    ensure!(
        video_task == video_output.is_some(),
        "only video tasks have an MP4 output target"
    );
    let mut recipe = prepare_native_command_for_route(catalog, request, accepted_route)?;
    ensure!(
        matches!(
            &recipe.expectation,
            NativeOutputExpectation::MediaScript { task } if task == expected_task
        ),
        "MiniMax-H3 preparation did not resolve to the accepted task"
    );
    recipe.argv.push(text_argument(if video_task {
        "--video"
    } else {
        "--audio-only"
    })?);
    append_private_output_controls(recipe, output_wav, video_output)
}

fn append_private_output_controls(
    mut recipe: NativeCommandRecipe,
    output_wav: &std::path::Path,
    video_output: Option<&std::path::Path>,
) -> Result<NativeCommandRecipe> {
    ensure!(
        output_wav.is_absolute()
            && output_wav.file_name().and_then(|name| name.to_str()) == Some("result.wav"),
        "media WAV target must use the fixed absolute result filename"
    );
    let output_directory = output_wav
        .parent()
        .ok_or_else(|| anyhow!("media WAV target has no parent directory"))?;
    if let Some(video_output) = video_output {
        ensure!(
            video_output.is_absolute()
                && video_output.file_name().and_then(|name| name.to_str()) == Some("result.mp4")
                && video_output.parent() == Some(output_directory),
            "media video target must use the fixed result filename in the WAV directory"
        );
        recipe.argv.push(text_argument("--video-output")?);
        recipe.argv.push(path_argument(video_output)?);
    }
    recipe.argv.push(text_argument("--output")?);
    recipe.argv.push(path_argument(output_wav)?);
    recipe.argv.push(text_argument("--out-dir")?);
    recipe.argv.push(path_argument(output_directory)?);
    ensure_argv_budget(&recipe.executable, &recipe.argv)?;
    Ok(recipe)
}

/// Shared shape-only admission for HeartMuLa wrapper arguments. Values remain
/// opaque native input; this checks only the wrapper's supported option
/// structure and mutually exclusive source fields.
pub(crate) fn is_supported_heartmula_args(arguments: &[String]) -> bool {
    // Prompt, tags, lyrics, duration, steps, seed, temperature, top-k,
    // guidance, and instrumental. Aliases share their option slot.
    let mut seen = [false; 10];
    let mut has_prompt = false;
    let mut has_lyrics = false;
    let mut instrumental = false;
    let mut index = 0;

    while index < arguments.len() {
        let (slot, takes_value) = match arguments[index].as_str() {
            "--prompt" | "--text" | "--prompt-file" => (0, true),
            "--tags" => (1, true),
            "--lyrics" | "--lyrics-file" => (2, true),
            "--duration" | "--duration-seconds" => (3, true),
            "--steps" | "--num-inference-steps" => (4, true),
            "--seed" => (5, true),
            "--temperature" => (6, true),
            "--top-k" => (7, true),
            "--guidance-scale" => (8, true),
            "--instrumental" => (9, false),
            // In particular, output controls, help, backend overrides, the
            // option delimiter, and all unknown flags are never caller-owned.
            _ => return false,
        };
        if seen[slot] {
            return false;
        }
        seen[slot] = true;
        if takes_value {
            let Some(value) = arguments.get(index + 1) else {
                return false;
            };
            if value.is_empty() || value.contains('\0') || value.starts_with("--") {
                return false;
            }
            match slot {
                0 => has_prompt = true,
                2 => has_lyrics = true,
                _ => {}
            }
            index += 2;
        } else {
            instrumental = true;
            index += 1;
        }
    }

    has_prompt && !(instrumental && has_lyrics)
}

pub(crate) fn is_supported_heartmula_request_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
) -> bool {
    let InferenceRequest::Media {
        profile,
        input: MediaInput::Script { task, args },
    } = request
    else {
        return false;
    };
    let NativeRoute::MediaScript { path } = &route.native_route else {
        return false;
    };
    profile == &route.profile
        && profile.lane == Lane::Media
        && profile.profile_id.as_str() == "heartmula-music"
        && task.as_str() == "music"
        && route.runtime.lane == Lane::Media
        && route.runtime.as_str() == "audio.cpp"
        && route.device.as_ref() == Some(&Device::Cuda)
        && route
            .tasks
            .iter()
            .any(|declared| declared.as_str() == "music")
        && path.ends_with(std::path::Path::new(
            "media-models/generate-heartmula-music.sh",
        ))
        && is_supported_heartmula_args(args)
}

/// Shape-only H3 wrapper admission. Request options are limited to the
/// layerwise controls evidenced in the wrapper; mode/output and `return_video`
/// remain controlled by the route/task adapter.
pub(crate) fn is_supported_h3_args(arguments: &[String], task: &str) -> bool {
    if !matches!(task, "music" | "tts" | "video") {
        return false;
    }
    let mut prompt_seen = false;
    let mut seen = [false; 5]; // steps, seed, width, height, frames
    let mut request_options = std::collections::BTreeSet::new();
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        if option == "--prompt" || option == "--text" || option == "--prompt-file" {
            if prompt_seen || !valid_h3_value(arguments.get(index + 1)) {
                return false;
            }
            prompt_seen = true;
            index += 2;
            continue;
        }
        let slot = match option {
            "--steps" | "--num-inference-steps" => Some(0),
            "--seed" => Some(1),
            "--width" => Some(2),
            "--height" => Some(3),
            "--frames" => Some(4),
            _ => None,
        };
        if let Some(slot) = slot {
            if seen[slot] || !valid_h3_value(arguments.get(index + 1)) {
                return false;
            }
            seen[slot] = true;
            index += 2;
            continue;
        }
        if option == "--request-option" {
            let Some(value) = arguments.get(index + 1) else {
                return false;
            };
            let Some((key, option_value)) = value.split_once('=') else {
                return false;
            };
            const ALLOWED_REQUEST_KEYS: [&str; 5] = [
                "text_layerwise",
                "text_layerwise_batch",
                "dit_layerwise",
                "dit_layerwise_batch",
                "dit_mlp_chunk_tokens",
            ];
            if !ALLOWED_REQUEST_KEYS.contains(&key)
                || option_value.is_empty()
                || option_value.contains('\0')
                || option_value.starts_with("--")
                || !request_options.insert(key)
            {
                return false;
            }
            index += 2;
            continue;
        }
        // --video/--audio-only, all output controls, help, backend selection,
        // delimiters, and unknown flags are not request-owned.
        return false;
    }
    prompt_seen
}

pub(crate) fn is_supported_h3_request_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
) -> bool {
    let InferenceRequest::Media {
        profile,
        input: MediaInput::Script { task, args },
    } = request
    else {
        return false;
    };
    let NativeRoute::MediaScript { path } = &route.native_route else {
        return false;
    };
    profile == &route.profile
        && profile.lane == Lane::Media
        && profile.profile_id.as_str() == "minimax-h3"
        && route.runtime.lane == Lane::Media
        && route.runtime.as_str() == "audio.cpp"
        && route.device.as_ref() == Some(&Device::Cuda)
        && route.tasks.iter().any(|declared| declared == task)
        && path.ends_with(std::path::Path::new("media-models/generate-minimax-h3.sh"))
        && is_supported_h3_args(args, task.as_str())
}

fn valid_h3_value(value: Option<&String>) -> bool {
    value
        .is_some_and(|value| !value.is_empty() && !value.contains('\0') && !value.starts_with("--"))
}

fn breeze_recipe_with_output(
    mut recipe: NativeCommandRecipe,
    output_directory: &std::path::Path,
) -> Result<NativeCommandRecipe> {
    ensure!(
        matches!(
            &recipe.expectation,
            NativeOutputExpectation::BreezeSpeech { .. }
        ),
        "explicit Breeze output requires a Breeze speech request"
    );
    let output = output_directory.join("result.wav");
    recipe.argv.push(text_argument("--output")?);
    recipe.argv.push(path_argument(&output)?);
    recipe.argv.push(text_argument("--out-dir")?);
    recipe.argv.push(path_argument(output_directory)?);
    ensure_argv_budget(&recipe.executable, &recipe.argv)?;
    Ok(recipe)
}

fn ensure_route_matches_verified_path(
    route: &ResolvedProfileRoute,
    verified: &VerifiedNativePath,
) -> Result<()> {
    let declared_path = match &route.native_route {
        NativeRoute::MediaScript { path } => path,
        NativeRoute::DecisionRunner { entrypoint, .. } => entrypoint,
        NativeRoute::LlamaSwapModel { .. } => {
            bail!("language route has no native local command entrypoint")
        }
    };
    let current_path = fs::canonicalize(declared_path)
        .context("recheck current manifest-declared entrypoint path")?;
    ensure!(
        current_path == verified.entrypoint,
        "native route changed while preparing command; retry dispatch"
    );
    ensure!(
        verified.profile == route.profile,
        "verified path profile does not match current route"
    );
    Ok(())
}

fn media_script_recipe(
    verified: &VerifiedNativePath,
    task: &str,
    native_args: &[String],
) -> Result<NativeCommandRecipe> {
    ensure!(
        verified.kind == NativeEntrypointKind::MediaScript,
        "media script request resolved to a non-media entrypoint"
    );
    ensure!(
        verified.entrypoint.extension().and_then(|ext| ext.to_str()) == Some("sh"),
        "native media command adapter supports only shell scripts"
    );
    let bash = checked_executable(
        PathBuf::from(BASH_EXECUTABLE),
        &format!("bash interpreter at {BASH_EXECUTABLE}"),
    )?;
    ensure_no_simulation_args(native_args)?;

    let mut argv = vec![path_argument(&verified.entrypoint)?];
    for argument in native_args {
        argv.push(text_argument(argument)?);
    }
    ensure_argv_budget(&bash, &argv)?;
    Ok(NativeCommandRecipe {
        executable: bash,
        argv,
        cwd: verified.toolkit_root.clone(),
        stdin_payload: None,
        expectation: NativeOutputExpectation::MediaScript {
            task: task.to_owned(),
        },
    })
}

fn ensure_no_simulation_args(arguments: &[String]) -> Result<()> {
    ensure!(
        !arguments
            .iter()
            .any(|argument| argument == "--simulate" || argument.starts_with("--simulate=")),
        "simulation options are not accepted by native command preparation"
    );
    Ok(())
}

fn breeze_speech_recipe(
    key: &ProfileKey,
    route: &ResolvedProfileRoute,
    verified: &VerifiedNativePath,
    input: &SpeechInput,
) -> Result<NativeCommandRecipe> {
    let cpu_profile = key.profile_id.as_str() == CPU_BREEZE_PROFILE_ID;
    ensure!(
        key.lane == Lane::Media
            && (key.profile_id.as_str() == BREEZE_PROFILE_ID || cpu_profile)
            && route.profile == *key,
        "speech inputs are supported only by declared Breeze-TTS-2 media profiles"
    );
    if cpu_profile {
        ensure!(
            route.runtime.lane == Lane::Media
                && route.runtime.as_str() == "audio.cpp"
                && route.device.as_ref() == Some(&Device::Cpu)
                && is_supported_cpu_breeze_speech(input),
            "the CPU Breeze profile does not support these speech fields or mode"
        );
    } else {
        ensure!(
            route.device.as_ref() != Some(&Device::Cpu),
            "the CUDA Breeze profile cannot be dispatched as CPU speech"
        );
    }
    ensure!(
        verified.kind == NativeEntrypointKind::MediaScript,
        "Breeze speech request resolved to a non-media entrypoint"
    );
    let expected_script = fs::canonicalize(verified.toolkit_root.join(BREEZE_SCRIPT))
        .context("resolve Breeze-TTS-2's native generation script")?;
    ensure!(
        verified.entrypoint == expected_script,
        "Breeze-TTS-2 profile does not resolve to its native generation script"
    );
    let bash = checked_executable(
        PathBuf::from(BASH_EXECUTABLE),
        &format!("bash interpreter at {BASH_EXECUTABLE}"),
    )?;

    let (wrapper_task, audio_cpp_task) = match input.mode {
        SpeechMode::Standard => ("tts", "tts"),
        SpeechMode::VoiceDesign => ("design", "tts"),
        SpeechMode::VoiceClone => ("clon", "clon"),
        SpeechMode::Streaming => ("tts", "tts"),
    };
    if !cpu_profile && input.mode != SpeechMode::VoiceClone {
        ensure!(
            input.voice_reference.is_none() && input.reference_text.is_none(),
            "reference audio and transcript are supported only for voice cloning"
        );
    }

    let mut argv = vec![
        path_argument(&verified.entrypoint)?,
        text_argument("--task")?,
        text_argument(wrapper_task)?,
        text_argument("--text")?,
        text_argument(&input.text)?,
        text_argument("--mode")?,
        text_argument(if input.mode == SpeechMode::Streaming {
            "streaming"
        } else {
            "offline"
        })?,
    ];
    if cpu_profile {
        argv.push(text_argument("--backend")?);
        argv.push(text_argument("cpu")?);
    }
    if let Some(instruction) = &input.instruction {
        argv.push(text_argument("--instruction")?);
        argv.push(text_argument(instruction)?);
    }
    if let Some(voice_reference) = &input.voice_reference {
        argv.push(text_argument("--voice-ref")?);
        let resolved_reference = resolve_voice_reference(&verified.toolkit_root, voice_reference)?;
        argv.push(path_argument(&resolved_reference)?);
    }
    if let Some(reference_text) = &input.reference_text {
        argv.push(text_argument("--reference-text")?);
        argv.push(text_argument(reference_text)?);
    }
    ensure_argv_budget(&bash, &argv)?;

    let artifacts = if input.mode == SpeechMode::Streaming {
        BreezeArtifacts::FinalWaveAndChunksAtWrapperDefault {
            chunk_directory_suffix: ".chunks",
        }
    } else {
        BreezeArtifacts::FinalWaveAtWrapperDefault
    };
    Ok(NativeCommandRecipe {
        executable: bash,
        argv,
        cwd: verified.toolkit_root.clone(),
        stdin_payload: None,
        expectation: NativeOutputExpectation::BreezeSpeech {
            wrapper_task,
            audio_cpp_task,
            mode: input.mode,
            artifacts,
        },
    })
}

/// Keep CPU Breeze mode/field admission shared by recipe preparation, execution,
/// and direct-route checks. Streaming artifacts are collected after the native
/// command finishes; this does not provide live audio delivery.
pub(crate) fn is_supported_cpu_breeze_speech(input: &SpeechInput) -> bool {
    let no_reference = input.voice_reference.is_none() && input.reference_text.is_none();
    match input.mode {
        SpeechMode::Standard => input.instruction.is_none() && no_reference,
        SpeechMode::Streaming => input.instruction.is_none() && no_reference,
        SpeechMode::VoiceDesign => {
            input
                .instruction
                .as_deref()
                .is_some_and(|instruction| !instruction.trim().is_empty())
                && no_reference
        }
        SpeechMode::VoiceClone => {
            input.instruction.is_none()
                && input
                    .voice_reference
                    .as_ref()
                    .is_some_and(|path| !path.as_os_str().is_empty())
                && input
                    .reference_text
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty())
        }
    }
}

fn decision_recipe(
    profile: &super::catalog::CatalogProfile,
    route: &ResolvedProfileRoute,
    verified: &VerifiedNativePath,
    state: &Value,
    questions: &serde_json::Map<String, Value>,
) -> Result<NativeCommandRecipe> {
    ensure!(
        verified.kind == NativeEntrypointKind::DecisionRunner,
        "decision request resolved to a non-decision entrypoint"
    );
    let NativeRoute::DecisionRunner {
        model,
        profile_type,
        tier,
        ..
    } = &route.native_route
    else {
        bail!("decision profile has no native runner route")
    };
    let engine = route.runtime.as_str();
    ensure!(
        matches!(engine, "nagi" | "laya" | "semif" | "jev"),
        "decision runner does not support the declared engine"
    );
    ensure!(
        (engine == "jev" && profile_type == "api")
            || (matches!(engine, "nagi" | "laya" | "semif") && profile_type == "local"),
        "decision engine and native profile type are incompatible"
    );
    ensure!(
        !model.is_empty() && !model.contains('\0'),
        "decision model identifier is invalid"
    );

    let hosted = engine == "jev" && profile_type == "api";
    let python = if hosted {
        configured_hosted_decision_interpreter(
            &verified.toolkit_root,
            std::env::var_os(HOSTED_DECISION_PYTHON_ENV),
        )?
    } else {
        checked_executable(
            verified.toolkit_root.join(DECISION_PYTHON),
            &format!("child-owned decision interpreter at {DECISION_PYTHON}"),
        )?
    };

    let stdin_payload = encode_decision_stdin_payload(state, questions)?;
    let mut argv = vec![
        path_argument(&verified.entrypoint)?,
        text_argument("--engine")?,
        text_argument(engine)?,
        text_argument("--model")?,
        text_argument(model)?,
    ];
    let device = decision_device(profile.device.as_ref(), engine, profile_type)?;
    argv.push(text_argument("--device")?);
    argv.push(text_argument(device)?);
    argv.push(text_argument("--input-stdin")?);
    if hosted {
        argv.push(text_argument("--single-attempt")?);
    }
    ensure_argv_budget(&python, &argv)?;

    Ok(NativeCommandRecipe {
        executable: python,
        argv,
        cwd: verified.toolkit_root.clone(),
        stdin_payload: Some(stdin_payload),
        expectation: NativeOutputExpectation::DecisionJsonStdout {
            engine: engine.to_owned(),
            model: model.clone(),
            profile_type: profile_type.clone(),
            tier: tier.clone(),
        },
    })
}

fn checked_executable(path: PathBuf, label: &str) -> Result<PathBuf> {
    let metadata = fs::metadata(&path).with_context(|| format!("inspect {label}"))?;
    ensure!(metadata.is_file(), "{label} is not a regular file");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o111 != 0,
            "{label} is not executable"
        );
    }
    Ok(path)
}

fn configured_hosted_decision_interpreter(
    toolkit_root: &std::path::Path,
    configured: Option<std::ffi::OsString>,
) -> Result<PathBuf> {
    let (path, label) = match configured {
        Some(configured) => (
            PathBuf::from(configured),
            "configured hosted decision interpreter",
        ),
        None => (
            toolkit_root.join(HOSTED_DECISION_PYTHON),
            "hosted decision interpreter at .venv-hosted/bin/python",
        ),
    };
    ensure!(
        path.is_absolute(),
        "hosted decision interpreter must be an absolute path"
    );
    ensure!(
        !path.to_string_lossy().contains('\0'),
        "hosted decision interpreter path is invalid"
    );
    checked_executable(path, label)
}

#[derive(Serialize)]
struct DecisionStdinEnvelope<'a> {
    state: &'a Value,
    questions: &'a Map<String, Value>,
}

fn encode_decision_stdin_payload(
    state: &Value,
    questions: &Map<String, Value>,
) -> Result<SensitivePayload> {
    let payload = serde_json::to_vec(&DecisionStdinEnvelope { state, questions })
        .context("encode native decision stdin envelope")?;
    ensure!(
        payload.len() <= MAX_DECISION_STDIN_BYTES,
        "native decision stdin envelope exceeds 1 MiB"
    );
    Ok(SensitivePayload(payload))
}

fn decision_device<'a>(
    device: Option<&'a Device>,
    engine: &str,
    profile_type: &str,
) -> Result<&'a str> {
    if engine == "jev" && profile_type == "api" {
        ensure!(
            matches!(device, Some(Device::Remote)),
            "hosted Jev decision route must explicitly declare remote device"
        );
        return Ok("remote");
    }
    match device {
        Some(Device::Cpu) => Ok("cpu"),
        Some(Device::Cuda) => Ok("cuda"),
        Some(Device::Remote) => Ok("remote"),
        Some(Device::Other(value)) => {
            ensure!(!value.trim().is_empty(), "decision device is invalid");
            ensure!(!value.contains('\0'), "decision device is invalid");
            Ok(value)
        }
        None => bail!("decision profile has no declared device; refusing runner default"),
    }
}

/// Canonicalize a Breeze clone sample before dispatch. Relative paths are
/// anchored to the verified toolkit root (also the recipe's native child CWD)
/// and may not escape it; absolute paths may refer to external sample files.
/// Either form must resolve to a non-empty regular file. This checks input
/// shape only and does not inspect audio content or claim it is loadable.
fn resolve_voice_reference(
    toolkit_root: &std::path::Path,
    input_path: &std::path::Path,
) -> Result<PathBuf> {
    let is_relative = !input_path.is_absolute();
    let candidate = if is_relative {
        toolkit_root.join(input_path)
    } else {
        input_path.to_path_buf()
    };
    let canonical = fs::canonicalize(&candidate).context("resolve voice reference audio path")?;
    if is_relative {
        ensure!(
            canonical.starts_with(toolkit_root),
            "relative voice reference escapes its toolkit root"
        );
    }
    let metadata = fs::metadata(&canonical).context("inspect voice reference audio")?;
    ensure!(metadata.is_file(), "voice reference is not a regular file");
    ensure!(metadata.len() > 0, "voice reference audio is empty");
    Ok(canonical)
}

fn text_argument(value: &str) -> Result<CommandArgument> {
    ensure!(
        !value.contains('\0'),
        "native command argument contains NUL"
    );
    ensure!(
        value.len() <= MAX_NATIVE_ARGUMENT_BYTES,
        "native command argument exceeds 64 KiB"
    );
    Ok(CommandArgument(value.into()))
}

fn path_argument(value: &std::path::Path) -> Result<CommandArgument> {
    ensure!(
        value.as_os_str().as_encoded_bytes().len() <= MAX_NATIVE_ARGUMENT_BYTES,
        "native command path exceeds 64 KiB"
    );
    ensure!(
        !value.as_os_str().as_encoded_bytes().contains(&0),
        "native command path contains NUL"
    );
    Ok(CommandArgument(value.as_os_str().to_os_string()))
}

fn ensure_argv_budget(executable: &std::path::Path, arguments: &[CommandArgument]) -> Result<()> {
    let executable_bytes = executable.as_os_str().as_encoded_bytes().len();
    ensure!(
        executable_bytes <= MAX_NATIVE_ARGUMENT_BYTES,
        "native executable path exceeds 64 KiB"
    );
    let total = arguments
        .iter()
        .fold(executable_bytes.saturating_add(1), |bytes, argument| {
            bytes.saturating_add(
                argument
                    .as_os_str()
                    .as_encoded_bytes()
                    .len()
                    .saturating_add(1),
            )
        });
    ensure!(
        total <= MAX_NATIVE_ARGV_BYTES,
        "native command argv exceeds 256 KiB"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    use tempfile::TempDir;

    struct Fixture {
        _parent: TempDir,
        language_root: PathBuf,
        media_root: PathBuf,
        decisions_root: PathBuf,
    }

    impl Fixture {
        fn new(media_id: &str, media_script: &str, media_tasks: &[&str]) -> Self {
            let parent = tempfile::tempdir().expect("create fixture root");
            let language_root = parent.path().join("language");
            let media_root = parent.path().join("configured-media-toolkit");
            let decisions_root = parent.path().join("configured-decisions-toolkit");
            fs::create_dir_all(&language_root).unwrap();
            fs::create_dir_all(&media_root).unwrap();
            fs::create_dir_all(&decisions_root).unwrap();
            fs::write(
                language_root.join("llama-swap.yaml"),
                "models:\n  language-model:\n    cmd: model\n",
            )
            .unwrap();
            fs::write(
                media_root.join("media-runtimes.json"),
                serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": media_id,
                        "name": "Fixture media",
                        "runtime": "audio.cpp",
                        "tasks": media_tasks,
                        "script": media_script,
                    }]
                })
                .to_string(),
            )
            .unwrap();
            fs::write(
                decisions_root.join("decision-runtimes.json"),
                serde_json::json!({
                    "version": "1.0",
                    "profiles": [{
                        "id": "nagi-smol",
                        "name": "Fixture Nagi",
                        "engine": "nagi",
                        "tier": "smol",
                        "type": "local",
                        "model": "org/model-name",
                        "device": "cuda",
                        "entrypoint": "decision-models/runner.py",
                    }]
                })
                .to_string(),
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

        fn write_media_script(&self, relative: &str) {
            write_nonempty(&self.media_root.join(relative));
        }

        fn write_decision_files(&self, with_python: bool) {
            write_nonempty(&self.decisions_root.join("decision-models/runner.py"));
            if with_python {
                let python = self.decisions_root.join(DECISION_PYTHON);
                write_nonempty(&python);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
                }
            }
        }

        fn write_hosted_decision_python(&self) {
            let python = self.decisions_root.join(HOSTED_DECISION_PYTHON);
            write_nonempty(&python);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
            }
        }
    }

    fn write_nonempty(path: &Path) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "fixture content\n").unwrap();
    }

    fn write_tiny_wav(path: &Path) {
        let mut wav = vec![0; 44];
        wav[0..4].copy_from_slice(b"RIFF");
        wav[8..12].copy_from_slice(b"WAVE");
        wav[12..16].copy_from_slice(b"fmt ");
        wav[16..20].copy_from_slice(&16_u32.to_le_bytes());
        wav[20..22].copy_from_slice(&1_u16.to_le_bytes());
        wav[22..24].copy_from_slice(&1_u16.to_le_bytes());
        wav[24..28].copy_from_slice(&8_000_u32.to_le_bytes());
        wav[28..32].copy_from_slice(&16_000_u32.to_le_bytes());
        wav[32..34].copy_from_slice(&2_u16.to_le_bytes());
        wav[34..36].copy_from_slice(&16_u16.to_le_bytes());
        wav[36..40].copy_from_slice(b"data");
        fs::write(path, wav).unwrap();
    }

    fn key(lane: Lane, id: &str) -> ProfileKey {
        ProfileKey::new(lane, id).unwrap()
    }

    fn argv_strings(recipe: &NativeCommandRecipe) -> Vec<String> {
        recipe
            .argv
            .iter()
            .map(|arg| arg.as_os_str().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn heartmula_argument_admission_is_an_exact_wrapper_allowlist() {
        for accepted in [
            vec!["--prompt", "soft ambient music"],
            vec!["--text", "soft ambient music", "--instrumental"],
            vec!["--prompt-file", "lyrics/prompt.txt", "--tags", "ambient"],
            vec!["--prompt", "song", "--lyrics-file", "lyrics.txt"],
            vec![
                "--prompt",
                "song",
                "--duration-seconds",
                "30",
                "--num-inference-steps",
                "10",
                "--seed",
                "42",
                "--temperature",
                "1.0",
                "--top-k",
                "50",
                "--guidance-scale",
                "1.5",
            ],
        ] {
            assert!(is_supported_heartmula_args(
                &accepted.into_iter().map(str::to_owned).collect::<Vec<_>>()
            ));
        }

        for rejected in [
            vec![],
            vec!["--prompt"],
            vec!["--prompt", ""],
            vec!["--prompt", "--instrumental"],
            vec!["--prompt", "song", "--prompt-file", "prompt.txt"],
            vec!["--prompt", "song", "--text", "another song"],
            vec!["--prompt", "song", "--lyrics", "words", "--instrumental"],
            vec!["--prompt", "song", "--lyrics"],
            vec!["--prompt", "song", "--duration"],
            vec!["--prompt", "song", "--output", "/tmp/arbitrary.wav"],
            vec!["--prompt", "song", "--out", "/tmp/arbitrary.wav"],
            vec!["--prompt", "song", "--out-dir", "/tmp/arbitrary"],
            vec!["--prompt", "song", "--help"],
            vec!["--prompt", "song", "--"],
            vec!["--prompt", "song", "--backend", "cuda"],
            vec!["--prompt", "song", "--other", "value"],
        ] {
            assert!(
                !is_supported_heartmula_args(
                    &rejected.into_iter().map(str::to_owned).collect::<Vec<_>>()
                ),
                "accepted unsupported HeartMuLa argument shape"
            );
        }
    }

    #[test]
    fn heartmula_recipe_pins_both_script_output_controls_to_private_target() {
        let fixture = Fixture::new(
            "heartmula-music",
            "media-models/generate-heartmula-music.sh",
            &["music"],
        );
        fixture.write_media_script("media-models/generate-heartmula-music.sh");
        let manifest_path = fixture.media_root.join("media-runtimes.json");
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = Value::String("cuda".into());
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let catalog = fixture.catalog();
        let request = InferenceRequest::Media {
            profile: key(Lane::Media, "heartmula-music"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("music").unwrap(),
                args: vec!["--prompt".into(), "fixture song".into()],
            },
        };
        let accepted = resolve_profile_route(&catalog, request.profile()).unwrap();
        assert!(is_supported_heartmula_request_route(&request, &accepted));
        let output = fixture._parent.path().join("private-job/result.wav");
        let recipe =
            prepare_heartmula_music_recipe_for_route(&catalog, &request, &accepted, &output)
                .unwrap();
        let argv = argv_strings(&recipe);
        assert_eq!(
            &argv[argv.len() - 4..],
            [
                "--output",
                output.to_str().unwrap(),
                "--out-dir",
                output.parent().unwrap().to_str().unwrap(),
            ]
        );
        let mut wrong_device = accepted;
        wrong_device.device = Some(Device::Cpu);
        assert!(!is_supported_heartmula_request_route(
            &request,
            &wrong_device
        ));
        assert!(prepare_heartmula_music_recipe_for_route(
            &catalog,
            &request,
            &wrong_device,
            &output,
        )
        .is_err());
    }

    #[test]
    fn h3_argument_admission_is_task_bound_and_limits_request_options() {
        for accepted in [
            (
                "music",
                vec!["--prompt", "soft ambient music", "--steps", "8"],
            ),
            ("tts", vec!["--prompt-file", "prompt.txt", "--seed", "42"]),
            (
                "video",
                vec![
                    "--text",
                    "rainy neon street",
                    "--width",
                    "832",
                    "--height",
                    "480",
                    "--frames",
                    "121",
                    "--request-option",
                    "text_layerwise=true",
                    "--request-option",
                    "dit_layerwise=true",
                    "--request-option",
                    "dit_mlp_chunk_tokens=512",
                ],
            ),
        ] {
            assert!(is_supported_h3_args(
                &accepted
                    .1
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
                accepted.0
            ));
        }

        for (task, rejected) in [
            ("music", vec![]),
            ("music", vec!["--prompt"]),
            ("video", vec!["--prompt", "scene", "--audio-only"]),
            ("music", vec!["--prompt", "song", "--video"]),
            (
                "video",
                vec!["--prompt", "scene", "--video-output", "/tmp/x.mp4"],
            ),
            ("music", vec!["--prompt", "song", "--output", "/tmp/x.wav"]),
            ("music", vec!["--prompt", "song", "--out", "/tmp/x.wav"]),
            ("music", vec!["--prompt", "song", "--out-dir", "/tmp/x"]),
            ("video", vec!["--prompt", "scene", "--help"]),
            ("video", vec!["--prompt", "scene", "--"]),
            ("video", vec!["--prompt", "scene", "--backend", "cuda"]),
            ("video", vec!["--prompt", "scene", "--request-option"]),
            (
                "video",
                vec![
                    "--prompt",
                    "scene",
                    "--request-option",
                    "return_video=false",
                ],
            ),
            (
                "video",
                vec!["--prompt", "scene", "--request-option", "width=32"],
            ),
            (
                "video",
                vec![
                    "--prompt",
                    "scene",
                    "--request-option",
                    "dit_layerwise=true",
                    "--request-option",
                    "dit_layerwise=false",
                ],
            ),
            ("image", vec!["--prompt", "scene"]),
        ] {
            assert!(
                !is_supported_h3_args(
                    &rejected.into_iter().map(str::to_owned).collect::<Vec<_>>(),
                    task
                ),
                "accepted unsupported H3 argument shape"
            );
        }
    }

    #[test]
    fn h3_recipe_injects_task_mode_and_fixed_private_targets() {
        let fixture = Fixture::new(
            "minimax-h3",
            "media-models/generate-minimax-h3.sh",
            &["music", "tts", "video"],
        );
        fixture.write_media_script("media-models/generate-minimax-h3.sh");
        let manifest_path = fixture.media_root.join("media-runtimes.json");
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = Value::String("cuda".into());
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let catalog = fixture.catalog();

        let video_request = InferenceRequest::Media {
            profile: key(Lane::Media, "minimax-h3"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("video").unwrap(),
                args: vec![
                    "--prompt".into(),
                    "fixture video".into(),
                    "--request-option".into(),
                    "dit_layerwise=true".into(),
                ],
            },
        };
        let accepted = resolve_profile_route(&catalog, video_request.profile()).unwrap();
        assert!(is_supported_h3_request_route(&video_request, &accepted));
        let output = fixture._parent.path().join("private-job/result.wav");
        let mp4 = output.parent().unwrap().join("result.mp4");
        let recipe =
            prepare_h3_recipe_for_route(&catalog, &video_request, &accepted, &output, Some(&mp4))
                .unwrap();
        let argv = argv_strings(&recipe);
        assert!(argv.contains(&"--video".to_owned()));
        assert!(!argv.contains(&"--audio-only".to_owned()));
        assert_eq!(
            &argv[argv.len() - 6..],
            [
                "--video-output",
                mp4.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--out-dir",
                output.parent().unwrap().to_str().unwrap(),
            ]
        );

        let audio_request = InferenceRequest::Media {
            profile: key(Lane::Media, "minimax-h3"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("tts").unwrap(),
                args: vec!["--prompt".into(), "fixture audio".into()],
            },
        };
        let audio_route = resolve_profile_route(&catalog, audio_request.profile()).unwrap();
        let audio_recipe =
            prepare_h3_recipe_for_route(&catalog, &audio_request, &audio_route, &output, None)
                .unwrap();
        let audio_argv = argv_strings(&audio_recipe);
        assert!(audio_argv.contains(&"--audio-only".to_owned()));
        assert!(!audio_argv.contains(&"--video".to_owned()));
        assert!(!audio_argv.contains(&"--video-output".to_owned()));
    }

    fn speech_request(profile: &str, mode: SpeechMode, fixture: &Fixture) -> InferenceRequest {
        let voice_reference = if mode == SpeechMode::VoiceClone {
            let path = fixture._parent.path().join("speaker.wav");
            write_tiny_wav(&path);
            Some(path)
        } else {
            None
        };
        InferenceRequest::Media {
            profile: key(Lane::Media, profile),
            input: MediaInput::Speech(SpeechInput {
                text: "speech prompt".into(),
                mode,
                instruction: (mode == SpeechMode::VoiceDesign).then(|| "warm, clear voice".into()),
                voice_reference,
                reference_text: (mode == SpeechMode::VoiceClone)
                    .then(|| "speaker reference words".into()),
            }),
        }
    }

    fn voice_clone_request(profile: &str, reference: PathBuf) -> InferenceRequest {
        InferenceRequest::Media {
            profile: key(Lane::Media, profile),
            input: MediaInput::Speech(SpeechInput {
                text: "clone fixture voice".into(),
                mode: SpeechMode::VoiceClone,
                instruction: None,
                voice_reference: Some(reference),
                reference_text: Some("fixture transcript".into()),
            }),
        }
    }

    fn decision_request(profile: &str) -> InferenceRequest {
        InferenceRequest::Decision {
            profile: key(Lane::Decisions, profile),
            state: serde_json::json!({"fixture":true}),
            questions: serde_json::json!({
                "q": {"type":"score", "range":[0,1]}
            })
            .as_object()
            .unwrap()
            .clone(),
        }
    }

    #[test]
    fn prepares_shell_media_script_as_separate_native_arguments() {
        let fixture = Fixture::new("generic-media", "media-models/generate.sh", &["tts"]);
        fixture.write_media_script("media-models/generate.sh");
        let catalog = fixture.catalog();
        let request = InferenceRequest::Media {
            profile: key(Lane::Media, "generic-media"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("tts").unwrap(),
                args: vec![
                    "--text".into(),
                    "two words stay together".into(),
                    "--seed".into(),
                    "7".into(),
                ],
            },
        };
        let recipe = prepare_native_command(&catalog, &request).unwrap();

        assert_eq!(recipe.executable, PathBuf::from(BASH_EXECUTABLE));
        assert_eq!(recipe.cwd, fs::canonicalize(&fixture.media_root).unwrap());
        let argv = argv_strings(&recipe);
        assert_eq!(
            argv[0],
            fs::canonicalize(fixture.media_root.join("media-models/generate.sh"))
                .unwrap()
                .to_string_lossy()
        );
        assert_eq!(
            &argv[1..],
            ["--text", "two words stay together", "--seed", "7"]
        );
        assert!(
            matches!(recipe.expectation, NativeOutputExpectation::MediaScript { ref task } if task == "tts")
        );
        assert!(recipe.stdin_payload.is_none());
    }

    #[test]
    fn rejects_workflow_simulation_and_nul_argument_inputs() {
        let fixture = Fixture::new(
            "comfy-profile",
            "media-models/generate-comfy.sh",
            &["image"],
        );
        fixture.write_media_script("media-models/generate-comfy.sh");
        let catalog = fixture.catalog();

        let workflow = InferenceRequest::Media {
            profile: key(Lane::Media, "comfy-profile"),
            input: MediaInput::Workflow {
                task: super::super::catalog::Task::new("image").unwrap(),
                graph: serde_json::json!({"nodes":{}}),
            },
        };
        assert!(prepare_native_command(&catalog, &workflow)
            .unwrap_err()
            .to_string()
            .contains("ComfyUI HTTP adapter"));

        let simulated = InferenceRequest::Media {
            profile: key(Lane::Media, "comfy-profile"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("image").unwrap(),
                args: vec!["--simulate".into()],
            },
        };
        assert!(prepare_native_command(&catalog, &simulated)
            .unwrap_err()
            .to_string()
            .contains("simulation options"));

        let nul_arg = InferenceRequest::Media {
            profile: key(Lane::Media, "comfy-profile"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("image").unwrap(),
                args: vec!["bad\0argument".into()],
            },
        };
        assert!(prepare_native_command(&catalog, &nul_arg).is_err());
    }

    #[test]
    fn rejects_runtime_identity_changes_after_catalog_reservation() {
        let fixture = Fixture::new("generic-media", "media-models/generate.sh", &["tts"]);
        fixture.write_media_script("media-models/generate.sh");
        let catalog = fixture.catalog();
        fs::write(
            fixture.media_root.join("media-runtimes.json"),
            serde_json::json!({
                "schema_version": 1,
                "profiles": [{
                    "id": "generic-media",
                    "name": "Fixture media",
                    "runtime": "changed-runtime",
                    "tasks": ["tts"],
                    "script": "media-models/generate.sh",
                }]
            })
            .to_string(),
        )
        .unwrap();
        let request = InferenceRequest::Media {
            profile: key(Lane::Media, "generic-media"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("tts").unwrap(),
                args: Vec::new(),
            },
        };

        let error = prepare_native_command(&catalog, &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("runtime identity"));
        assert!(error.contains("rebind and reconcile"));
    }

    #[test]
    fn bounds_script_and_speech_argv_by_encoded_byte_size() {
        let generic = Fixture::new("generic-media", "media-models/generate.sh", &["tts"]);
        generic.write_media_script("media-models/generate.sh");
        let generic_catalog = generic.catalog();

        let large_script_arg = InferenceRequest::Media {
            profile: key(Lane::Media, "generic-media"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("tts").unwrap(),
                args: vec!["x".repeat(MAX_NATIVE_ARGUMENT_BYTES + 1)],
            },
        };
        assert!(prepare_native_command(&generic_catalog, &large_script_arg)
            .unwrap_err()
            .to_string()
            .contains("argument exceeds 64 KiB"));

        let large_aggregate = InferenceRequest::Media {
            profile: key(Lane::Media, "generic-media"),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new("tts").unwrap(),
                args: vec!["x".repeat(60 * 1024); 5],
            },
        };
        assert!(prepare_native_command(&generic_catalog, &large_aggregate)
            .unwrap_err()
            .to_string()
            .contains("argv exceeds 256 KiB"));

        let breeze = Fixture::new(BREEZE_PROFILE_ID, BREEZE_SCRIPT, &["tts", "design", "clon"]);
        breeze.write_media_script(BREEZE_SCRIPT);
        let breeze_catalog = breeze.catalog();
        let oversized_multibyte_speech = InferenceRequest::Media {
            profile: key(Lane::Media, BREEZE_PROFILE_ID),
            input: MediaInput::Speech(SpeechInput {
                text: "é".repeat(MAX_NATIVE_ARGUMENT_BYTES / 2 + 1),
                mode: SpeechMode::Standard,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            }),
        };
        assert!(
            prepare_native_command(&breeze_catalog, &oversized_multibyte_speech)
                .unwrap_err()
                .to_string()
                .contains("argument exceeds 64 KiB")
        );
    }

    #[test]
    fn maps_breeze_speech_modes_to_wrapper_flags_and_output_contracts() {
        let fixture = Fixture::new(BREEZE_PROFILE_ID, BREEZE_SCRIPT, &["tts", "design", "clon"]);
        fixture.write_media_script(BREEZE_SCRIPT);
        let catalog = fixture.catalog();

        for (mode, expected_wrapper_task, expected_cli_task, expected_mode) in [
            (SpeechMode::Standard, "tts", "tts", "offline"),
            (SpeechMode::VoiceDesign, "design", "tts", "offline"),
            (SpeechMode::VoiceClone, "clon", "clon", "offline"),
            (SpeechMode::Streaming, "tts", "tts", "streaming"),
        ] {
            let request = speech_request(BREEZE_PROFILE_ID, mode, &fixture);
            let recipe = prepare_native_command(&catalog, &request).unwrap();
            assert_eq!(recipe.executable, PathBuf::from(BASH_EXECUTABLE));
            assert!(recipe.stdin_payload.is_none());
            let args = argv_strings(&recipe);
            assert_eq!(
                args[0],
                fs::canonicalize(fixture.media_root.join(BREEZE_SCRIPT))
                    .unwrap()
                    .to_string_lossy()
            );
            assert_eq!(args[1..3], ["--task", expected_wrapper_task]);
            assert_eq!(args[3..5], ["--text", "speech prompt"]);
            assert!(!args.iter().any(|arg| arg == "--backend"));
            let mode_index = args.iter().position(|arg| arg == "--mode").unwrap();
            assert_eq!(args[mode_index + 1], expected_mode);
            assert!(!args.iter().any(|arg| arg == "--simulate"));
            assert!(matches!(
                recipe.expectation,
                NativeOutputExpectation::BreezeSpeech {
                    wrapper_task,
                    audio_cpp_task,
                    mode: actual_mode,
                    ..
                } if wrapper_task == expected_wrapper_task
                    && audio_cpp_task == expected_cli_task
                    && actual_mode == mode
            ));
            if mode == SpeechMode::Streaming {
                assert!(matches!(
                    recipe.expectation,
                    NativeOutputExpectation::BreezeSpeech {
                        artifacts: BreezeArtifacts::FinalWaveAndChunksAtWrapperDefault {
                            chunk_directory_suffix: ".chunks"
                        },
                        ..
                    }
                ));
            }
            if mode == SpeechMode::VoiceClone {
                assert!(args.iter().any(|arg| arg == "--voice-ref"));
                assert!(args.iter().any(|arg| arg == "--reference-text"));
                let voice_index = args.iter().position(|arg| arg == "--voice-ref").unwrap();
                assert_eq!(
                    args[voice_index + 1],
                    fs::canonicalize(fixture._parent.path().join("speaker.wav"))
                        .unwrap()
                        .to_string_lossy()
                );
            }
        }
    }

    #[test]
    fn cpu_breeze_maps_supported_modes_and_rejects_mode_field_conflicts() {
        let fixture = Fixture::new(
            CPU_BREEZE_PROFILE_ID,
            BREEZE_SCRIPT,
            &["tts", "design", "clon"],
        );
        fixture.write_media_script(BREEZE_SCRIPT);
        let manifest_path = fixture.media_root.join("media-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = serde_json::json!("cpu");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let catalog = fixture.catalog();
        let accepted = super::super::routes::resolve_profile_route(
            &catalog,
            &key(Lane::Media, CPU_BREEZE_PROFILE_ID),
        )
        .unwrap();
        for (mode, expected_wrapper_task, expected_cli_task, instruction) in [
            (SpeechMode::Standard, "tts", "tts", None),
            (
                SpeechMode::VoiceDesign,
                "design",
                "tts",
                Some("warm, clear voice"),
            ),
            (SpeechMode::VoiceClone, "clon", "clon", None),
            (SpeechMode::Streaming, "tts", "tts", None),
        ] {
            let request = speech_request(CPU_BREEZE_PROFILE_ID, mode, &fixture);
            let recipe = prepare_native_command_for_route(&catalog, &request, &accepted).unwrap();
            let args = argv_strings(&recipe);
            let backend = args.iter().position(|arg| arg == "--backend").unwrap();
            assert_eq!(&args[backend..backend + 2], ["--backend", "cpu"]);
            let task = args.iter().position(|arg| arg == "--task").unwrap();
            assert_eq!(&args[task..task + 2], ["--task", expected_wrapper_task]);
            assert_eq!(
                args.iter()
                    .position(|arg| arg == "--instruction")
                    .map(|index| args[index + 1].as_str()),
                instruction
            );
            match recipe.expectation {
                NativeOutputExpectation::BreezeSpeech {
                    wrapper_task,
                    audio_cpp_task,
                    mode: actual_mode,
                    artifacts,
                } => {
                    assert_eq!(wrapper_task, expected_wrapper_task);
                    assert_eq!(audio_cpp_task, expected_cli_task);
                    assert_eq!(actual_mode, mode);
                    if mode == SpeechMode::Streaming {
                        assert!(matches!(
                            artifacts,
                            BreezeArtifacts::FinalWaveAndChunksAtWrapperDefault {
                                chunk_directory_suffix: ".chunks"
                            }
                        ));
                    } else {
                        assert_eq!(artifacts, BreezeArtifacts::FinalWaveAtWrapperDefault);
                    }
                }
                _ => panic!("CPU Breeze produces a speech recipe"),
            }

            let mut cuda_route = accepted.clone();
            cuda_route.device = Some(Device::Cuda);
            assert!(prepare_native_command_for_route(&catalog, &request, &cuda_route).is_err());
        }

        let mut malformed_design =
            speech_request(CPU_BREEZE_PROFILE_ID, SpeechMode::VoiceDesign, &fixture);
        let InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } = &mut malformed_design
        else {
            unreachable!();
        };
        input.instruction = None;
        assert!(prepare_native_command_for_route(&catalog, &malformed_design, &accepted).is_err());

        let mut malformed_clone =
            speech_request(CPU_BREEZE_PROFILE_ID, SpeechMode::VoiceClone, &fixture);
        let InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } = &mut malformed_clone
        else {
            unreachable!();
        };
        input.reference_text = Some("   ".into());
        assert!(prepare_native_command_for_route(&catalog, &malformed_clone, &accepted).is_err());

        let mut malformed_streaming =
            speech_request(CPU_BREEZE_PROFILE_ID, SpeechMode::Streaming, &fixture);
        let InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } = &mut malformed_streaming
        else {
            unreachable!();
        };
        input.instruction = Some("not allowed".into());
        assert!(
            prepare_native_command_for_route(&catalog, &malformed_streaming, &accepted).is_err()
        );

        let cuda_fixture = Fixture::new(BREEZE_PROFILE_ID, BREEZE_SCRIPT, &["tts"]);
        cuda_fixture.write_media_script(BREEZE_SCRIPT);
        let cuda_manifest_path = cuda_fixture.media_root.join("media-runtimes.json");
        let mut cuda_manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&cuda_manifest_path).unwrap()).unwrap();
        cuda_manifest["profiles"][0]["device"] = serde_json::json!("cpu");
        fs::write(
            &cuda_manifest_path,
            serde_json::to_vec(&cuda_manifest).unwrap(),
        )
        .unwrap();
        let cuda_catalog = cuda_fixture.catalog();
        let cuda_request = speech_request(BREEZE_PROFILE_ID, SpeechMode::Standard, &cuda_fixture);
        assert!(prepare_native_command(&cuda_catalog, &cuda_request).is_err());
    }

    #[test]
    fn resolves_relative_clone_audio_from_toolkit_root_and_validates_files() {
        let fixture = Fixture::new(BREEZE_PROFILE_ID, BREEZE_SCRIPT, &["tts", "design", "clon"]);
        fixture.write_media_script(BREEZE_SCRIPT);
        let catalog = fixture.catalog();

        let relative_reference = PathBuf::from("samples/reference.wav");
        let toolkit_reference = fixture.media_root.join(&relative_reference);
        fs::create_dir_all(toolkit_reference.parent().unwrap()).unwrap();
        write_tiny_wav(&toolkit_reference);
        let recipe = prepare_native_command(
            &catalog,
            &voice_clone_request(BREEZE_PROFILE_ID, relative_reference),
        )
        .unwrap();
        let args = argv_strings(&recipe);
        let voice_index = args.iter().position(|arg| arg == "--voice-ref").unwrap();
        assert_eq!(
            args[voice_index + 1],
            fs::canonicalize(&toolkit_reference)
                .unwrap()
                .to_string_lossy()
        );

        let escaped_reference = fixture._parent.path().join("outside.wav");
        write_tiny_wav(&escaped_reference);
        let escaped = prepare_native_command(
            &catalog,
            &voice_clone_request(BREEZE_PROFILE_ID, PathBuf::from("../outside.wav")),
        )
        .unwrap_err()
        .to_string();
        assert!(escaped.contains("relative voice reference escapes"));

        let missing = prepare_native_command(
            &catalog,
            &voice_clone_request(
                BREEZE_PROFILE_ID,
                fixture._parent.path().join("missing.wav"),
            ),
        )
        .unwrap_err()
        .to_string();
        assert!(missing.contains("resolve voice reference audio path"));

        let directory_reference = fixture.media_root.join("reference-directory");
        fs::create_dir_all(&directory_reference).unwrap();
        let directory = prepare_native_command(
            &catalog,
            &voice_clone_request(BREEZE_PROFILE_ID, directory_reference),
        )
        .unwrap_err()
        .to_string();
        assert!(directory.contains("not a regular file"));

        let empty_reference = fixture.media_root.join("empty.wav");
        fs::write(&empty_reference, []).unwrap();
        let empty = prepare_native_command(
            &catalog,
            &voice_clone_request(BREEZE_PROFILE_ID, empty_reference),
        )
        .unwrap_err()
        .to_string();
        assert!(empty.contains("voice reference audio is empty"));
    }

    #[test]
    fn speech_is_rejected_for_profiles_other_than_breeze() {
        let fixture = Fixture::new("other-tts", "media-models/other.sh", &["tts"]);
        fixture.write_media_script("media-models/other.sh");
        let catalog = fixture.catalog();
        let request = speech_request("other-tts", SpeechMode::Standard, &fixture);
        assert!(prepare_native_command(&catalog, &request)
            .unwrap_err()
            .to_string()
            .contains("only by declared Breeze-TTS-2"));
    }

    #[test]
    fn maps_decision_runner_actual_cli_with_native_json_without_simulation_or_secrets() {
        let fixture = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        fixture.write_decision_files(true);
        let catalog = fixture.catalog();
        let questions = serde_json::json!({
            "move": {"type":"choice", "criteria":{"yes":"Go", "no":"Stay"}}
        })
        .as_object()
        .unwrap()
        .clone();
        let state = serde_json::json!({"position":{"x":4,"y":8}});
        let request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: state.clone(),
            questions: questions.clone(),
        };
        let recipe = prepare_native_command(&catalog, &request).unwrap();
        let args = argv_strings(&recipe);
        assert_eq!(
            recipe.executable,
            fixture.decisions_root.join(DECISION_PYTHON)
        );
        assert_eq!(
            recipe.cwd,
            fs::canonicalize(&fixture.decisions_root).unwrap()
        );
        assert_eq!(
            args[0],
            fs::canonicalize(fixture.decisions_root.join("decision-models/runner.py"))
                .unwrap()
                .to_string_lossy()
        );
        assert_eq!(
            &args[1..5],
            ["--engine", "nagi", "--model", "org/model-name"]
        );
        assert_eq!(&args[5..7], ["--device", "cuda"]);
        assert_eq!(args.get(7).map(String::as_str), Some("--input-stdin"));
        assert!(!args
            .iter()
            .any(|arg| matches!(arg.as_str(), "--state" | "--question")));
        assert!(!args.iter().any(|arg| matches!(
            arg.as_str(),
            "--simulate" | "--api-key" | "--base-url" | "--tier" | "--type"
        )));
        let stdin_payload = recipe.stdin_payload.as_ref().unwrap();
        let envelope: Value = serde_json::from_slice(stdin_payload.as_bytes()).unwrap();
        assert_eq!(envelope["state"], state);
        assert_eq!(envelope["questions"], serde_json::json!(questions));
        assert!(stdin_payload.as_bytes().len() <= MAX_DECISION_STDIN_BYTES);
        assert!(matches!(
            &recipe.expectation,
            NativeOutputExpectation::DecisionJsonStdout {
                engine,
                profile_type,
                tier: Some(tier),
                ..
            } if engine == "nagi" && profile_type == "local" && tier == "smol"
        ));
        let debug = format!("{recipe:?}");
        assert!(!debug.contains(&serde_json::to_string(&state).unwrap()));
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("speech prompt"));
        assert!(recipe.stdin_payload.is_some());
    }

    #[test]
    fn accepted_route_preparation_refuses_model_only_changes() {
        let fixture = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        fixture.write_decision_files(true);
        let catalog = fixture.catalog();
        let request = decision_request("nagi-smol");
        let accepted = resolve_profile_route(&catalog, request.profile()).unwrap();
        let manifest_path = fixture.decisions_root.join("decision-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["model"] = serde_json::json!("different/model");
        fs::write(&manifest_path, manifest.to_string()).unwrap();
        let error = prepare_native_command_for_route(&catalog, &request, &accepted).unwrap_err();
        assert!(error.to_string().contains("accepted native route changed"));
    }

    #[test]
    fn stale_native_device_never_reuses_the_catalog_resource_assumption() {
        let fixture = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        fixture.write_decision_files(true);
        let catalog = fixture.catalog();
        let manifest_path = fixture.decisions_root.join("decision-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = serde_json::json!("cpu");
        fs::write(&manifest_path, manifest.to_string()).unwrap();
        let error = prepare_native_command(&catalog, &decision_request("nagi-smol")).unwrap_err();
        assert!(error.to_string().contains("native device changed"));
    }

    #[test]
    fn requires_explicit_local_device_and_uses_explicit_remote_for_hosted_jev() {
        let local = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        local.write_decision_files(true);
        fs::write(
            local.decisions_root.join("decision-runtimes.json"),
            serde_json::json!({
                "version": "1.0",
                "profiles": [{
                    "id": "nagi-smol",
                    "name": "Fixture Nagi",
                    "engine": "nagi",
                    "tier": "smol",
                    "type": "local",
                    "model": "org/model-name",
                    "entrypoint": "decision-models/runner.py"
                }]
            })
            .to_string(),
        )
        .unwrap();
        let local_catalog = local.catalog();
        let missing_device = prepare_native_command(&local_catalog, &decision_request("nagi-smol"))
            .unwrap_err()
            .to_string();
        assert!(missing_device.contains("no declared device"));

        let hosted = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        hosted.write_decision_files(true);
        hosted.write_hosted_decision_python();
        fs::write(
            hosted.decisions_root.join("decision-runtimes.json"),
            serde_json::json!({
                "version": "1.0",
                "profiles": [{
                    "id": "jev-api",
                    "name": "Fixture hosted Jev",
                    "engine": "jev",
                    "tier": "hosted",
                    "type": "api",
                    "device": "remote",
                    "model": "typesafe/jev-latest",
                    "entrypoint": "decision-models/runner.py"
                }]
            })
            .to_string(),
        )
        .unwrap();
        let hosted_catalog = hosted.catalog();
        let recipe = prepare_native_command(&hosted_catalog, &decision_request("jev-api")).unwrap();
        let args = argv_strings(&recipe);
        let expected_python = std::env::var_os(HOSTED_DECISION_PYTHON_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| hosted.decisions_root.join(HOSTED_DECISION_PYTHON));
        assert_eq!(recipe.executable, expected_python);
        assert_eq!(
            &args[1..5],
            ["--engine", "jev", "--model", "typesafe/jev-latest"]
        );
        assert_eq!(&args[5..7], ["--device", "remote"]);
        assert_eq!(&args[7..9], ["--input-stdin", "--single-attempt"]);
    }

    #[test]
    fn hosted_interpreter_override_must_be_absolute_and_executable() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let default = root.path().join(HOSTED_DECISION_PYTHON);
        fs::create_dir_all(default.parent().unwrap()).unwrap();
        fs::write(&default, "fixture").unwrap();
        fs::set_permissions(&default, fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(
            configured_hosted_decision_interpreter(root.path(), None).unwrap(),
            default
        );

        let explicit = root.path().join("custom/python");
        fs::create_dir_all(explicit.parent().unwrap()).unwrap();
        fs::write(&explicit, "fixture").unwrap();
        fs::set_permissions(&explicit, fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(
            configured_hosted_decision_interpreter(
                root.path(),
                Some(explicit.clone().into_os_string())
            )
            .unwrap(),
            explicit
        );
        assert!(configured_hosted_decision_interpreter(
            root.path(),
            Some("relative/python".into())
        )
        .is_err());
    }

    #[test]
    fn invalid_request_errors_do_not_echo_question_ids_or_payload_details() {
        let fixture = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        let catalog = fixture.catalog();
        let sensitive_id = "credential-like-question-sentinel";
        let request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: serde_json::json!({"payload":"private-state"}),
            questions: serde_json::json!({
                (sensitive_id): {"type":"choice"}
            })
            .as_object()
            .unwrap()
            .clone(),
        };

        let error = prepare_native_command(&catalog, &request)
            .unwrap_err()
            .to_string();
        assert_eq!(error, "native inference request failed validation");
        assert!(!error.contains(sensitive_id));
        assert!(!error.contains("private-state"));
    }

    #[test]
    fn rejects_stdin_envelope_overhead_that_exceeds_runner_limit() {
        let fixture = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        fixture.write_decision_files(true);
        let catalog = fixture.catalog();
        let questions = serde_json::json!({
            "q": {"type":"score", "range":[0,1]}
        })
        .as_object()
        .unwrap()
        .clone();
        let question_bytes = serde_json::to_vec(&questions).unwrap().len();
        let mut valid_request = None;

        // Request validation budgets state+questions, while the native stdin
        // protocol also needs an envelope. Find a valid input at that boundary
        // and ensure envelope overhead is checked before returning a recipe.
        for padding_len in (0..MAX_DECISION_STDIN_BYTES - question_bytes).rev() {
            let state = serde_json::json!({"padding": "x".repeat(padding_len)});
            let request = InferenceRequest::Decision {
                profile: key(Lane::Decisions, "nagi-smol"),
                state,
                questions: questions.clone(),
            };
            if request.validate(&catalog).is_ok() {
                valid_request = Some(request);
                break;
            }
        }
        let request = valid_request.expect("find an input within request budget");
        let error = prepare_native_command(&catalog, &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("stdin envelope exceeds 1 MiB"));
    }

    #[test]
    fn requires_the_child_owned_python_file_without_running_it() {
        let fixture = Fixture::new("media-profile", "media-models/unused.sh", &["tts"]);
        fixture.write_decision_files(false);
        let catalog = fixture.catalog();
        let request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: serde_json::json!({"x":1}),
            questions: serde_json::json!({"q":{"type":"score","range":[0,1]}})
                .as_object()
                .unwrap()
                .clone(),
        };
        let error = prepare_native_command(&catalog, &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains(DECISION_PYTHON));
    }
}
