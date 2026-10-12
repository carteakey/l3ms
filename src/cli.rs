//! Command-line launcher for the Rust L3MS application.

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{ArgGroup, Parser, ValueEnum};

use crate::{
    bench_results::{load_results_in, sort_results, ResultSort},
    control::catalog::{Catalog, CatalogProfile, Device, Readiness},
    llama_swap::{SwapClient, SwapModel},
    prompt_store::list_prompts,
    script_store::{collect_scripts_in, ScriptMode},
    settings::{
        export_profile_in, import_profile_in, load_current_stack_settings_in, load_settings_in,
    },
};

#[cfg(target_os = "linux")]
const COORDINATOR_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_COORDINATOR_RESPONSE_BYTES: usize = 2 * 1024;
const MAX_COORDINATOR_RESULTS_ENVELOPE_BYTES: usize = 4 * 1024;
const MAX_COORDINATOR_NATIVE_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_DIRECT_NATIVE_INPUT_BYTES: usize = 1024 * 1024;
const MAX_DIRECT_NATIVE_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_DIRECT_NATIVE_TIMEOUT_MS: u64 = 300_000;
const MAX_DIRECT_NATIVE_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const DEFAULT_DIRECT_COMFY_URL: &str = "http://127.0.0.1:8188";
#[cfg(target_os = "linux")]
const MAX_COORDINATOR_ARTIFACT_CHUNK_JSON_BYTES: usize = 160 * 1024;

#[derive(Debug, Parser)]
#[command(
    name = "l3ms",
    version,
    about = "L3MS launcher (TUI + interactive model and bench CLI)",
    group(
        ArgGroup::new("action")
            .args(["run", "run_profile", "bench", "list", "quickstart", "settings", "export_profile", "import_profile", "compare_results", "coordinator_status", "coordinator_cancel", "request_results", "request_output", "artifact_chunk", "coordinator_cpu_daemon", "coordinator_managed_session"])
            .multiple(false)
    )
)]
struct Cli {
    /// Interactively select and load a llama-swap model.
    #[arg(
        long,
        value_name = "FILTER",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    run: Option<String>,

    /// Execute one explicit native profile with strict native JSON input.
    #[arg(long, value_name = "NAMESPACED_ID", requires_all = ["task", "input"])]
    run_profile: Option<String>,

    /// Declared task name (decision, tts, clon, design, image, video, or embeddings) required by --run-profile.
    #[arg(long, value_name = "TASK", requires = "run_profile")]
    task: Option<String>,

    /// Strict native input JSON file required by --run-profile.
    #[arg(long, value_name = "PATH", requires = "run_profile")]
    input: Option<PathBuf>,

    /// Existing private empty output directory required for speech and Comfy workflows.
    #[arg(long, value_name = "PATH", requires = "run_profile")]
    output_dir: Option<PathBuf>,

    /// Bound the local native operation (default 300000 milliseconds).
    #[arg(long, value_name = "MILLISECONDS", requires = "run_profile")]
    task_timeout_ms: Option<u64>,

    /// Interactively select and execute a benchmark script.
    #[arg(
        long,
        value_name = "FILTER",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    bench: Option<String>,

    /// List live models, offline stack profiles, scripts, results, prompts, or everything.
    #[arg(long, value_name = "MODE")]
    list: Option<ListMode>,

    /// Print a quick-start guide and exit.
    #[arg(long)]
    quickstart: bool,

    /// Print the persisted settings and exit.
    #[arg(long)]
    settings: bool,

    /// Export models, scripts, and non-secret settings to a directory.
    #[arg(long, value_name = "DIRECTORY")]
    export_profile: Option<PathBuf>,

    /// Preview and import a profile directory.
    #[arg(long, value_name = "DIRECTORY")]
    import_profile: Option<PathBuf>,

    /// Compare two benchmark result files from the bounded result browser.
    #[arg(long, value_names = ["LEFT", "RIGHT"], num_args = 2)]
    compare_results: Option<Vec<PathBuf>>,

    /// Query one retained coordinator request by caller-owned request ID.
    #[arg(long, value_name = "ID")]
    coordinator_status: Option<String>,

    /// Cancel one retained queued/running coordinator request by request ID.
    #[arg(long, value_name = "ID")]
    coordinator_cancel: Option<String>,

    /// Read one bounded retained-results page by caller-owned request ID.
    #[arg(long, value_name = "ID")]
    request_results: Option<String>,

    /// Read currently retained and reassembled native output for one request.
    #[arg(long, value_name = "ID")]
    request_output: Option<String>,

    /// Read exactly one bounded chunk from an already-published artifact.
    #[arg(long, value_name = "ID")]
    artifact_chunk: Option<String>,

    /// Published artifact kind required by --artifact-chunk.
    #[arg(long, value_enum, value_name = "KIND")]
    artifact_kind: Option<ArtifactChunkKind>,

    /// Published relative artifact path required by --artifact-chunk.
    #[arg(long, value_name = "PATH")]
    artifact_path: Option<String>,

    /// Byte offset required by --artifact-chunk.
    #[arg(long, value_name = "OFFSET")]
    artifact_offset: Option<u64>,

    /// Maximum decoded bytes to read (1..=65536).
    #[arg(long, value_name = "BYTES")]
    artifact_byte_budget: Option<u32>,

    /// Optional exact lowercase 64-hex version token for a published artifact.
    #[arg(long, value_name = "TOKEN")]
    artifact_version: Option<String>,

    /// Native output lane required by --request-output.
    #[arg(long, value_enum, value_name = "LANE")]
    request_lane: Option<RequestOutputLane>,

    /// Run an explicitly configured foreground CPU-local coordinator session.
    #[arg(long, value_name = "CONFIG")]
    coordinator_cpu_daemon: Option<PathBuf>,

    /// Run one explicitly configured foreground managed language session.
    #[arg(long, value_name = "CONFIG")]
    coordinator_managed_session: Option<PathBuf>,

    /// Results cursor to read after; defaults to 0 and requires --request-results.
    #[arg(long, value_name = "CURSOR")]
    after_cursor: Option<u64>,

    /// Maximum serialized result-page bytes; defaults to 131072 and requires --request-results.
    #[arg(long, value_name = "BYTES")]
    result_page_bytes: Option<u32>,

    /// Private directory containing coordinator.sock; overrides shared settings.
    #[arg(long, value_name = "DIRECTORY")]
    coordinator_socket_dir: Option<PathBuf>,

    /// Shell-style arguments appended to the selected benchmark script.
    #[arg(long, value_name = "ARGS", allow_hyphen_values = true)]
    extra: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ListMode {
    Run,
    Bench,
    Results,
    Prompts,
    Stack,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RequestOutputLane {
    Language,
    Media,
    Decisions,
}

struct DirectNativeSpec<'a> {
    profile: &'a str,
    task: &'a str,
    input: &'a Path,
    output_dir: Option<&'a Path>,
    timeout_ms: u64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectDecisionInput {
    state: serde_json::Value,
    questions: serde_json::Map<String, serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectSpeechInput {
    text: String,
    #[serde(default)]
    mode: DirectSpeechMode,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum DirectSpeechMode {
    #[default]
    Offline,
    Streaming,
}

impl DirectSpeechMode {
    fn native_mode(self) -> crate::control::requests::SpeechMode {
        match self {
            Self::Offline => crate::control::requests::SpeechMode::Standard,
            Self::Streaming => crate::control::requests::SpeechMode::Streaming,
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectVoiceCloneInput {
    text: String,
    voice_reference: String,
    reference_text: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectVoiceDesignInput {
    text: String,
    instruction: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectComfyInput {
    graph: serde_json::Map<String, serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectEmbeddingInput {
    input: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectInputError {
    TooLarge,
    Invalid,
}

impl std::fmt::Display for DirectInputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::TooLarge => "native input exceeds the 1 MiB limit",
            Self::Invalid => "native input must be a strict JSON object for the selected task",
        })
    }
}

impl std::error::Error for DirectInputError {}

fn parse_direct_decision_input(bytes: &[u8]) -> Result<DirectDecisionInput, DirectInputError> {
    if bytes.len() > MAX_DIRECT_NATIVE_INPUT_BYTES {
        return Err(DirectInputError::TooLarge);
    }
    serde_json::from_slice(bytes).map_err(|_| DirectInputError::Invalid)
}

fn parse_direct_speech_input(bytes: &[u8]) -> Result<DirectSpeechInput, DirectInputError> {
    if bytes.len() > MAX_DIRECT_NATIVE_INPUT_BYTES {
        return Err(DirectInputError::TooLarge);
    }
    serde_json::from_slice(bytes).map_err(|_| DirectInputError::Invalid)
}

fn parse_direct_voice_clone_input(bytes: &[u8]) -> Result<DirectVoiceCloneInput, DirectInputError> {
    if bytes.len() > MAX_DIRECT_NATIVE_INPUT_BYTES {
        return Err(DirectInputError::TooLarge);
    }
    serde_json::from_slice(bytes).map_err(|_| DirectInputError::Invalid)
}

fn parse_direct_voice_design_input(
    bytes: &[u8],
) -> Result<DirectVoiceDesignInput, DirectInputError> {
    if bytes.len() > MAX_DIRECT_NATIVE_INPUT_BYTES {
        return Err(DirectInputError::TooLarge);
    }
    serde_json::from_slice(bytes).map_err(|_| DirectInputError::Invalid)
}

fn parse_direct_comfy_input(bytes: &[u8]) -> Result<serde_json::Value, DirectInputError> {
    if bytes.len() > MAX_DIRECT_NATIVE_INPUT_BYTES {
        return Err(DirectInputError::TooLarge);
    }
    let parsed: DirectComfyInput =
        serde_json::from_slice(bytes).map_err(|_| DirectInputError::Invalid)?;
    if parsed.graph.is_empty() {
        return Err(DirectInputError::Invalid);
    }
    Ok(serde_json::Value::Object(parsed.graph))
}

fn parse_direct_embedding_input(bytes: &[u8]) -> Result<DirectEmbeddingInput, DirectInputError> {
    if bytes.len() > MAX_DIRECT_NATIVE_INPUT_BYTES {
        return Err(DirectInputError::TooLarge);
    }
    serde_json::from_slice(bytes).map_err(|_| DirectInputError::Invalid)
}

fn direct_native_spec(cli: &Cli) -> Result<Option<DirectNativeSpec<'_>>> {
    let Some(profile) = cli.run_profile.as_deref() else {
        if cli.task.is_some()
            || cli.input.is_some()
            || cli.output_dir.is_some()
            || cli.task_timeout_ms.is_some()
        {
            bail!("--task, --input, --output-dir, and --task-timeout-ms require --run-profile");
        }
        return Ok(None);
    };
    if cli.extra.is_some() {
        bail!("--extra cannot be used with --run-profile");
    }
    if coordinator_options_present(cli) {
        bail!("--run-profile cannot be combined with coordinator options");
    }
    let task = cli
        .task
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--run-profile requires --task"))?;
    let input = cli
        .input
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--run-profile requires --input"))?;
    let output_dir = cli.output_dir.as_deref();
    match (task, output_dir) {
        ("decision", None)
        | ("embeddings", None)
        | ("tts" | "clon" | "design" | "image" | "video", Some(_)) => {}
        ("decision" | "embeddings", Some(_)) => {
            bail!("--output-dir is only valid with speech or Comfy workflow tasks")
        }
        ("tts" | "clon" | "design" | "image" | "video", None) => {
            bail!("--output-dir is required with speech and Comfy workflow tasks")
        }
        (_, Some(_)) => bail!("--output-dir is only valid with speech or Comfy workflow tasks"),
        _ => bail!(
            "--run-profile supports only --task decision, --task tts, --task clon, --task design, --task image, --task video, or --task embeddings"
        ),
    }
    let timeout_ms = cli
        .task_timeout_ms
        .unwrap_or(DEFAULT_DIRECT_NATIVE_TIMEOUT_MS);
    if timeout_ms == 0 || timeout_ms > MAX_DIRECT_NATIVE_TIMEOUT_MS {
        bail!("--task-timeout-ms must be between 1 and 86400000");
    }
    Ok(Some(DirectNativeSpec {
        profile,
        task,
        input,
        output_dir,
        timeout_ms,
    }))
}

fn coordinator_options_present(cli: &Cli) -> bool {
    cli.coordinator_status.is_some()
        || cli.coordinator_cancel.is_some()
        || cli.request_results.is_some()
        || cli.request_output.is_some()
        || cli.artifact_chunk.is_some()
        || cli.artifact_kind.is_some()
        || cli.artifact_path.is_some()
        || cli.artifact_offset.is_some()
        || cli.artifact_byte_budget.is_some()
        || cli.artifact_version.is_some()
        || cli.request_lane.is_some()
        || cli.coordinator_cpu_daemon.is_some()
        || cli.coordinator_managed_session.is_some()
        || cli.after_cursor.is_some()
        || cli.result_page_bytes.is_some()
        || cli.coordinator_socket_dir.is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ArtifactChunkKind {
    #[value(name = "local_job")]
    LocalJob,
    Backend,
}

impl ArtifactChunkKind {
    fn artifact_kind(self) -> crate::control::result_buffer::ArtifactKind {
        match self {
            Self::LocalJob => crate::control::result_buffer::ArtifactKind::LocalJob,
            Self::Backend => crate::control::result_buffer::ArtifactKind::Backend,
        }
    }
}

struct ArtifactChunkSpec {
    request_id: String,
    artifact: crate::control::result_buffer::MediaArtifact,
    offset: u64,
    byte_budget: u32,
    artifact_version: Option<crate::control::ipc::ArtifactVersion>,
}

impl std::fmt::Debug for ArtifactChunkSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ArtifactChunkSpec(<redacted>)")
    }
}

impl RequestOutputLane {
    fn lane(self) -> crate::control::catalog::Lane {
        match self {
            Self::Language => crate::control::catalog::Lane::Language,
            Self::Media => crate::control::catalog::Lane::Media,
            Self::Decisions => crate::control::catalog::Lane::Decisions,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct CommandSpec {
    program: OsString,
    args: Vec<OsString>,
}

/// Parse process arguments and launch the requested mode.
pub fn run() -> Result<u8> {
    #[cfg(target_os = "linux")]
    {
        // The owned manager invokes this internal operation directly. Preserve
        // OS-native backend argv and bypass ordinary settings/interactive startup.
        let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
        if let Some(result) = isolated_engine_action(arguments) {
            return result;
        }
        let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
        if arguments.first().is_some_and(|argument| {
            crate::control::owned_job_helper::is_helper_entrypoint(argument)
        }) {
            return crate::control::owned_job_helper::run_helper(arguments)
                .map(|()| 0)
                .map_err(anyhow::Error::new);
        }
    }
    dispatch(Cli::parse())
}

#[cfg(target_os = "linux")]
fn isolated_engine_action(arguments: Vec<OsString>) -> Option<Result<u8>> {
    if !arguments
        .first()
        .is_some_and(|argument| crate::control::engine_helper::is_helper_entrypoint(argument))
    {
        return None;
    }
    Some(
        crate::control::engine_helper::run_supervisor(arguments)
            .map(|()| 0)
            .map_err(anyhow::Error::new),
    )
}

fn dispatch(cli: Cli) -> Result<u8> {
    if let Some(code) = direct_native_action(&cli)? {
        return Ok(code);
    }
    if let Some(config_path) = coordinator_cpu_daemon_action(&cli)? {
        return run_cpu_coordinator_action(config_path);
    }
    if let Some(config_path) = coordinator_managed_session_action(&cli)? {
        return run_managed_session_action(config_path);
    }
    let output_spec = coordinator_output_spec(&cli)?;
    let artifact_chunk_spec = coordinator_artifact_chunk_spec(&cli)?;

    let configured_socket_dir = if (cli.coordinator_status.is_some()
        || cli.coordinator_cancel.is_some()
        || cli.request_results.is_some()
        || cli.request_output.is_some()
        || cli.artifact_chunk.is_some())
        && cli.coordinator_socket_dir.is_none()
        && cli.extra.is_none()
    {
        configured_coordinator_socket_dir()?
    } else {
        None
    };
    if let Some(spec) = artifact_chunk_spec {
        let socket_directory = coordinator_socket_directory(&cli, configured_socket_dir)?;
        return coordinator_artifact_chunk_action(&socket_directory, spec);
    }
    if let Some((request_id, lane)) = output_spec {
        let socket_directory = coordinator_socket_directory(&cli, configured_socket_dir)?;
        let consumer = coordinator_native_results_exchange(&socket_directory, request_id, lane)?;
        let json = coordinator_native_output_json(request_id, &consumer)?;
        println!("{json}");
        return Ok(0);
    }
    if let Some((request, socket_directory)) = coordinator_action(&cli, configured_socket_dir)? {
        let response = coordinator_exchange(&socket_directory, request)?;
        let json = coordinator_response_json(&response)?;
        let exit_code = u8::from(matches!(
            response,
            crate::control::ipc::ResponseMessage::Error { .. }
        ));
        println!("{json}");
        return Ok(exit_code);
    }

    if cli.quickstart {
        print_quickstart();
        return Ok(0);
    }

    if cli.settings {
        let data_root = crate::state_store::data_root()?;
        let settings = load_settings_in(&data_root)?;
        println!(
            "settings: {}",
            crate::settings::settings_path(&data_root).display()
        );
        println!("{}", serde_json::to_string_pretty(&settings)?);
        return Ok(0);
    }

    if let Some(target) = cli.export_profile {
        let root = repository_root()?;
        let data_root = crate::state_store::data_root()?;
        let manifest = export_profile_in(&root, &data_root, &target)?;
        println!(
            "exported profile to {} ({} script(s))",
            target.display(),
            manifest.scripts.len()
        );
        return Ok(0);
    }

    if let Some(source) = cli.import_profile {
        let root = repository_root()?;
        let data_root = crate::state_store::data_root()?;
        let manifest = import_profile_in(&root, &data_root, &source)?;
        println!(
            "imported profile from {} ({} script(s))",
            source.display(),
            manifest.scripts.len()
        );
        return Ok(0);
    }

    if let Some(paths) = cli.compare_results {
        let root = repository_root()?;
        compare_result_files(&root, &paths)?;
        return Ok(0);
    }

    if let Some(mode) = cli.list {
        if !cli.extra.as_deref().unwrap_or_default().trim().is_empty() {
            bail!("--extra is only valid with --bench");
        }
        return list(mode);
    }

    if let Some(filter) = cli.run {
        if !cli.extra.as_deref().unwrap_or_default().trim().is_empty() {
            bail!("--extra is only valid with --bench");
        }
        return interactive_run(&filter);
    }

    if let Some(filter) = cli.bench {
        return interactive_bench(&filter, cli.extra.as_deref().unwrap_or_default());
    }

    if !cli.extra.as_deref().unwrap_or_default().trim().is_empty() {
        bail!("--extra is only valid with --bench");
    }

    crate::app::run_tui()?;
    Ok(0)
}

fn coordinator_cpu_daemon_action(cli: &Cli) -> Result<Option<&Path>> {
    let Some(config_path) = cli.coordinator_cpu_daemon.as_deref() else {
        return Ok(None);
    };
    if cli.coordinator_socket_dir.is_some()
        || cli.coordinator_status.is_some()
        || cli.coordinator_cancel.is_some()
        || cli.request_results.is_some()
        || cli.request_output.is_some()
        || cli.artifact_chunk.is_some()
        || cli.artifact_kind.is_some()
        || cli.artifact_path.is_some()
        || cli.artifact_offset.is_some()
        || cli.artifact_byte_budget.is_some()
        || cli.artifact_version.is_some()
        || cli.request_lane.is_some()
        || cli.after_cursor.is_some()
        || cli.result_page_bytes.is_some()
        || cli.extra.is_some()
    {
        bail!("--coordinator-cpu-daemon cannot be combined with other coordinator options");
    }
    Ok(Some(config_path))
}

fn coordinator_managed_session_action(cli: &Cli) -> Result<Option<&Path>> {
    let Some(config_path) = cli.coordinator_managed_session.as_deref() else {
        return Ok(None);
    };
    if cli.coordinator_socket_dir.is_some()
        || cli.coordinator_status.is_some()
        || cli.coordinator_cancel.is_some()
        || cli.request_results.is_some()
        || cli.request_output.is_some()
        || cli.artifact_chunk.is_some()
        || cli.artifact_kind.is_some()
        || cli.artifact_path.is_some()
        || cli.artifact_offset.is_some()
        || cli.artifact_byte_budget.is_some()
        || cli.artifact_version.is_some()
        || cli.request_lane.is_some()
        || cli.after_cursor.is_some()
        || cli.result_page_bytes.is_some()
        || cli.extra.is_some()
    {
        bail!("--coordinator-managed-session cannot be combined with other coordinator options");
    }
    Ok(Some(config_path))
}

fn coordinator_output_spec(cli: &Cli) -> Result<Option<(&str, crate::control::catalog::Lane)>> {
    match (cli.request_output.as_deref(), cli.request_lane) {
        (None, None) => Ok(None),
        (None, Some(_)) => bail!("--request-lane requires --request-output"),
        (Some(_), None) => bail!("--request-output requires --request-lane"),
        (Some(request_id), Some(lane)) => {
            if cli.coordinator_status.is_some()
                || cli.coordinator_cancel.is_some()
                || cli.request_results.is_some()
            {
                bail!("--request-output must be the only coordinator operation");
            }
            if cli.after_cursor.is_some() || cli.result_page_bytes.is_some() {
                bail!("result cursors and page budgets cannot be used with --request-output");
            }
            if cli.extra.is_some() {
                bail!("--extra cannot be used with coordinator operations");
            }
            Ok(Some((request_id, lane.lane())))
        }
    }
}

fn coordinator_artifact_chunk_spec(cli: &Cli) -> Result<Option<ArtifactChunkSpec>> {
    let has_artifact_options = cli.artifact_kind.is_some()
        || cli.artifact_path.is_some()
        || cli.artifact_offset.is_some()
        || cli.artifact_byte_budget.is_some()
        || cli.artifact_version.is_some();
    let Some(request_id) = cli.artifact_chunk.as_deref() else {
        if has_artifact_options {
            bail!("artifact chunk options require --artifact-chunk");
        }
        return Ok(None);
    };

    if cli.artifact_kind.is_none()
        || cli.artifact_path.is_none()
        || cli.artifact_offset.is_none()
        || cli.artifact_byte_budget.is_none()
    {
        bail!(
            "--artifact-chunk requires --artifact-kind, --artifact-path, --artifact-offset, and --artifact-byte-budget"
        );
    }
    if cli.coordinator_status.is_some()
        || cli.coordinator_cancel.is_some()
        || cli.request_results.is_some()
        || cli.request_output.is_some()
        || cli.request_lane.is_some()
        || cli.coordinator_cpu_daemon.is_some()
        || cli.coordinator_managed_session.is_some()
        || cli.after_cursor.is_some()
        || cli.result_page_bytes.is_some()
        || cli.extra.is_some()
    {
        bail!("--artifact-chunk must be the only coordinator operation");
    }
    if crate::control::ipc::validate_request_id(request_id).is_err() {
        bail!("--artifact-chunk request ID is invalid");
    }

    let artifact = crate::control::result_buffer::MediaArtifact::new(
        cli.artifact_kind.expect("required above").artifact_kind(),
        cli.artifact_path.as_deref().expect("required above"),
    )
    .map_err(|_| anyhow::anyhow!("--artifact-path is invalid"))?;
    let byte_budget = cli.artifact_byte_budget.expect("required above");
    if !(1..=crate::control::ipc::MAX_ARTIFACT_CHUNK_BYTES as u32).contains(&byte_budget) {
        bail!("--artifact-byte-budget must be between 1 and 65536");
    }
    let offset = cli.artifact_offset.expect("required above");
    let artifact_version = cli
        .artifact_version
        .as_deref()
        .map(|token| {
            if !is_artifact_version_token(token) {
                bail!("--artifact-version must be exactly 64 lowercase hexadecimal characters");
            }
            crate::control::ipc::ArtifactVersion::new(token)
                .map_err(|_| anyhow::anyhow!("--artifact-version is invalid"))
        })
        .transpose()?;
    if offset > 0 && artifact_version.is_none() {
        bail!("--artifact-version is required when --artifact-offset is nonzero");
    }

    Ok(Some(ArtifactChunkSpec {
        request_id: request_id.to_owned(),
        artifact,
        offset,
        byte_budget,
        artifact_version,
    }))
}

fn is_artifact_version_token(token: &str) -> bool {
    token.len() == 64
        && token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(target_os = "linux")]
fn run_cpu_coordinator_action(config_path: &Path) -> Result<u8> {
    let report = run_cpu_coordinator(config_path)?;
    println!("{}", cpu_coordinator_report_json(report));
    Ok(u8::from(
        report.stopped == crate::control::daemon::StopReason::WorkerUncertain,
    ))
}

#[cfg(not(target_os = "linux"))]
fn run_cpu_coordinator_action(_: &Path) -> Result<u8> {
    bail!("CPU coordinator is supported only on Linux");
}

#[cfg(target_os = "linux")]
fn run_cpu_coordinator(config_path: &Path) -> Result<crate::control::daemon::RunReport> {
    crate::control::coordinator_launch::run(config_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(target_os = "linux")]
fn run_managed_session_action(config_path: &Path) -> Result<u8> {
    let report = run_managed_session(config_path)?;
    println!("{}", cpu_coordinator_report_json(report));
    Ok(u8::from(
        report.stopped == crate::control::daemon::StopReason::WorkerUncertain,
    ))
}

#[cfg(not(target_os = "linux"))]
fn run_managed_session_action(_: &Path) -> Result<u8> {
    bail!("managed coordinator sessions are supported only on Linux");
}

#[cfg(target_os = "linux")]
fn run_managed_session(config_path: &Path) -> Result<crate::control::daemon::RunReport> {
    crate::control::coordinator_launch::run_managed_session(config_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(target_os = "linux")]
fn cpu_coordinator_report_json(report: crate::control::daemon::RunReport) -> String {
    let stopped = match report.stopped {
        crate::control::daemon::StopReason::Cancelled => "cancelled",
        crate::control::daemon::StopReason::Deadline => "deadline",
        crate::control::daemon::StopReason::ConnectionLimit => "connection_limit",
        crate::control::daemon::StopReason::WorkerUncertain => "worker_uncertain",
    };
    format!(
        "{{\"completed_exchanges\":{},\"failed_connections\":{},\"stopped\":\"{stopped}\"}}",
        report.completed_exchanges, report.failed_connections
    )
}

fn coordinator_action(
    cli: &Cli,
    configured_socket_dir: Option<PathBuf>,
) -> Result<Option<(crate::control::ipc::RequestEnvelope, PathBuf)>> {
    let is_results_action = cli.request_results.is_some();
    if !is_results_action && (cli.after_cursor.is_some() || cli.result_page_bytes.is_some()) {
        bail!("--after-cursor and --result-page-bytes require --request-results");
    }
    let action_count = [
        cli.coordinator_status.is_some(),
        cli.coordinator_cancel.is_some(),
        cli.request_results.is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    if action_count > 1 {
        bail!("choose only one coordinator operation");
    }
    let action = match (
        cli.coordinator_status.as_deref(),
        cli.coordinator_cancel.as_deref(),
        cli.request_results.as_deref(),
    ) {
        (Some(request_id), None, None) => Some(ParsedCoordinatorAction::Status(request_id)),
        (None, Some(request_id), None) => Some(ParsedCoordinatorAction::Cancel(request_id)),
        (None, None, Some(request_id)) => Some(ParsedCoordinatorAction::Results(request_id)),
        (None, None, None) => None,
        _ => bail!("choose only one coordinator operation"),
    };

    let Some(action) = action else {
        if cli.coordinator_socket_dir.is_some() {
            bail!("--coordinator-socket-dir requires a coordinator operation");
        }
        return Ok(None);
    };

    if cli.extra.is_some() {
        bail!("--extra cannot be used with coordinator operations");
    }
    let socket_directory = coordinator_socket_directory(cli, configured_socket_dir)?;
    let request = match action {
        ParsedCoordinatorAction::Status(request_id) => crate::control::ipc::RequestEnvelope::v1(
            crate::control::ipc::WireRequestMessage::Status {
                request_id: request_id.to_owned(),
            },
        ),
        ParsedCoordinatorAction::Cancel(request_id) => crate::control::ipc::RequestEnvelope::v1(
            crate::control::ipc::WireRequestMessage::Cancel {
                request_id: request_id.to_owned(),
            },
        ),
        ParsedCoordinatorAction::Results(request_id) => {
            let byte_budget = cli
                .result_page_bytes
                .unwrap_or(crate::control::result_buffer::MAX_PAGE_BYTES as u32);
            if !(crate::control::result_buffer::MIN_PAGE_BYTES as u32
                ..=crate::control::result_buffer::MAX_PAGE_BYTES as u32)
                .contains(&byte_budget)
            {
                bail!("--result-page-bytes must be between 1024 and 131072");
            }
            crate::control::ipc::RequestEnvelope::v2_results(
                request_id,
                cli.after_cursor.unwrap_or(0),
                byte_budget,
            )
        }
    };
    Ok(Some((request, socket_directory)))
}

fn coordinator_socket_directory(
    cli: &Cli,
    configured_socket_dir: Option<PathBuf>,
) -> Result<PathBuf> {
    cli.coordinator_socket_dir
        .clone()
        .or(configured_socket_dir)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "coordinator operations require --coordinator-socket-dir or a configured coordinator socket directory"
            )
        })
}

enum ParsedCoordinatorAction<'a> {
    Status(&'a str),
    Cancel(&'a str),
    Results(&'a str),
}

fn configured_coordinator_socket_dir() -> Result<Option<PathBuf>> {
    let resolved = (|| {
        let root = repository_root()?;
        let data_root = crate::state_store::data_root()?;
        load_current_stack_settings_in(&root, &data_root)
    })()
    .map_err(|_| anyhow::anyhow!("coordinator settings are unavailable"))?;
    Ok(resolved.coordinator_socket_dir)
}

fn coordinator_exchange(
    socket_directory: &Path,
    request: crate::control::ipc::RequestEnvelope,
) -> Result<crate::control::ipc::ResponseMessage> {
    #[cfg(target_os = "linux")]
    {
        let never_cancel = crate::control::runtime::NeverCancel;
        let context = crate::control::runtime::OperationContext::new(
            Instant::now() + COORDINATOR_TIMEOUT,
            &never_cancel,
        );
        Ok(crate::control::client::exchange(
            socket_directory,
            request,
            &context,
        )?)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (socket_directory, request);
        bail!("coordinator IPC is supported only on Linux");
    }
}

#[cfg(target_os = "linux")]
fn coordinator_native_results_exchange(
    socket_directory: &Path,
    request_id: &str,
    lane: crate::control::catalog::Lane,
) -> Result<crate::control::result_consumer::ResultConsumer> {
    let never_cancel = crate::control::runtime::NeverCancel;
    let context = crate::control::runtime::OperationContext::new(
        Instant::now() + COORDINATOR_TIMEOUT,
        &never_cancel,
    );
    crate::control::client::read_native_results(socket_directory, request_id, lane, &context)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(not(target_os = "linux"))]
fn coordinator_native_results_exchange(
    _: &Path,
    _: &str,
    _: crate::control::catalog::Lane,
) -> Result<crate::control::result_consumer::ResultConsumer> {
    bail!("coordinator IPC is supported only on Linux");
}

#[derive(serde::Serialize)]
struct NativeOutputResponse<'a> {
    request_id: &'a str,
    cursor: u64,
    latest_cursor: u64,
    terminal: Option<crate::control::result_buffer::ResultTerminal>,
    complete: bool,
    output: &'a crate::control::result_consumer::NativeOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_decision_metadata: Option<&'a crate::control::result_consumer::NativeDecisionMetadata>,
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl BoundedJsonWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
        }
    }
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "bounded coordinator output",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn coordinator_native_output_json(
    request_id: &str,
    consumer: &crate::control::result_consumer::ResultConsumer,
) -> Result<String> {
    let response = NativeOutputResponse {
        request_id,
        cursor: consumer.cursor(),
        latest_cursor: consumer.latest_cursor(),
        terminal: consumer.terminal(),
        complete: consumer.is_complete(),
        output: consumer.output(),
        native_decision_metadata: consumer.decision_metadata(),
    };
    let mut writer = BoundedJsonWriter::new(MAX_COORDINATOR_NATIVE_OUTPUT_BYTES);
    if serde_json::to_writer(&mut writer, &response).is_err() {
        if writer.exceeded {
            bail!("coordinator native output exceeded the size limit");
        }
        bail!("coordinator native output could not be encoded");
    }
    String::from_utf8(writer.bytes)
        .map_err(|_| anyhow::anyhow!("coordinator native output could not be encoded"))
}

#[cfg(target_os = "linux")]
#[derive(serde::Serialize)]
struct ArtifactChunkOutput<'a> {
    request_id: &'a str,
    artifact: &'a crate::control::result_buffer::MediaArtifact,
    availability: crate::control::result_buffer::Availability,
    offset: u64,
    byte_budget: u32,
    total_bytes: u64,
    next_offset: u64,
    end_of_file: bool,
    artifact_version: Option<&'a str>,
    bytes_hex: String,
}

#[cfg(target_os = "linux")]
fn coordinator_artifact_chunk_action(
    socket_directory: &Path,
    spec: ArtifactChunkSpec,
) -> Result<u8> {
    let never_cancel = crate::control::runtime::NeverCancel;
    let context = crate::control::runtime::OperationContext::new(
        Instant::now() + COORDINATOR_TIMEOUT,
        &never_cancel,
    );
    let chunk = crate::control::client::artifact_chunk(
        socket_directory,
        &spec.request_id,
        &spec.artifact,
        spec.offset,
        spec.byte_budget,
        spec.artifact_version.as_ref(),
        &context,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    println!(
        "{}",
        coordinator_artifact_chunk_json(&chunk, spec.byte_budget)?
    );
    Ok(0)
}

#[cfg(not(target_os = "linux"))]
fn coordinator_artifact_chunk_action(_: &Path, _: ArtifactChunkSpec) -> Result<u8> {
    bail!("coordinator IPC is supported only on Linux");
}

#[cfg(target_os = "linux")]
fn coordinator_artifact_chunk_json(
    chunk: &crate::control::client::ArtifactChunk,
    byte_budget: u32,
) -> Result<String> {
    if chunk.bytes.len() > byte_budget as usize
        || chunk.bytes.len() > crate::control::ipc::MAX_ARTIFACT_CHUNK_BYTES
        || chunk
            .artifact_version
            .as_ref()
            .is_some_and(|token| !is_artifact_version_token(token.as_str()))
    {
        bail!("coordinator artifact response is invalid");
    }
    let response = ArtifactChunkOutput {
        request_id: &chunk.request_id,
        artifact: &chunk.artifact,
        availability: chunk.availability,
        offset: chunk.offset,
        byte_budget,
        total_bytes: chunk.total_bytes,
        next_offset: chunk.next_offset,
        end_of_file: chunk.end_of_file,
        artifact_version: chunk.artifact_version.as_ref().map(|token| token.as_str()),
        bytes_hex: lower_hex(&chunk.bytes),
    };
    let mut writer = BoundedJsonWriter::new(MAX_COORDINATOR_ARTIFACT_CHUNK_JSON_BYTES);
    if serde_json::to_writer(&mut writer, &response).is_err() {
        if writer.exceeded {
            bail!("coordinator artifact chunk exceeded the output limit");
        }
        bail!("coordinator artifact chunk could not be encoded");
    }
    String::from_utf8(writer.bytes)
        .map_err(|_| anyhow::anyhow!("coordinator artifact chunk could not be encoded"))
}

#[cfg(target_os = "linux")]
fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn coordinator_response_json(response: &crate::control::ipc::ResponseMessage) -> Result<String> {
    let encoded = serde_json::to_vec(response)
        .map_err(|_| anyhow::anyhow!("coordinator response could not be encoded"))?;
    match response {
        crate::control::ipc::ResponseMessage::Results {
            byte_budget, page, ..
        } => {
            let page_bytes = serde_json::to_vec(page)
                .map_err(|_| anyhow::anyhow!("coordinator results page is invalid"))?;
            if page.validate().is_err()
                || page_bytes.len() > crate::control::result_buffer::MAX_PAGE_BYTES
                || page_bytes.len() > *byte_budget as usize
                || encoded.len().saturating_sub(page_bytes.len())
                    > MAX_COORDINATOR_RESULTS_ENVELOPE_BYTES
            {
                bail!("coordinator results response exceeded the output limit");
            }
        }
        _ if encoded.len() > MAX_COORDINATOR_RESPONSE_BYTES => {
            bail!("coordinator response exceeded the output limit");
        }
        _ => {}
    }
    String::from_utf8(encoded)
        .map_err(|_| anyhow::anyhow!("coordinator response could not be encoded"))
}

fn print_quickstart() {
    println!("L3MS quick start");
    println!();
    println!("  1) Open the TUI");
    println!("     l3ms");
    println!();
    println!("  2) Load a llama-swap model without entering the TUI");
    println!("     l3ms --run");
    println!();
    println!("  3) Run a benchmark script");
    println!("     l3ms --bench");
    println!();
    println!("  4) Discover available models and scripts");
    println!("     l3ms --list all");
    println!("     l3ms --list stack  # offline three-lane manifest catalog");
    println!();
    println!("  5) Pass extra arguments to a selected benchmark");
    println!(r#"     l3ms --bench qwen --extra "--ctx-size 32768""#);
    println!();
    println!("  6) Inspect or move non-secret local profile settings");
    println!("     l3ms --settings");
    println!("     l3ms --export-profile ./profile");
    println!();
    println!("  7) Compare two benchmark result files");
    println!("     l3ms --compare-results bench-results/left.md bench-results/right.md");
}

fn list(mode: ListMode) -> Result<u8> {
    match mode {
        ListMode::Stack => print_stack_list()?,
        ListMode::Run => {
            let client = SwapClient::from_env()?;
            print_model_list(&client.list_models()?);
        }
        ListMode::Bench => {
            let root = repository_root()?;
            print_script_list(&root, &collect_bench_scripts(&root)?);
        }
        ListMode::Results => {
            let root = repository_root()?;
            print_result_list(&root)?;
        }
        ListMode::Prompts => {
            let data_root = crate::state_store::data_root()?;
            let prompts = list_prompts(&data_root)?;
            println!("CHAT prompts ({}):", prompts.len());
            for prompt in prompts {
                println!("  {prompt}");
            }
        }
        ListMode::All => {
            let client = SwapClient::from_env()?;
            print_model_list(&client.list_models()?);
            println!();
            let root = repository_root()?;
            print_script_list(&root, &collect_bench_scripts(&root)?);
            println!();
            print_result_list(&root)?;
            let data_root = crate::state_store::data_root()?;
            let prompts = list_prompts(&data_root)?;
            println!();
            println!("CHAT prompts ({}):", prompts.len());
            for prompt in prompts {
                println!("  {prompt}");
            }
        }
    }
    Ok(0)
}

fn print_stack_list() -> Result<()> {
    let root = repository_root()?;
    let user_data_root = crate::state_store::data_root()?;
    let resolved = load_current_stack_settings_in(&root, &user_data_root)?;
    let catalog = Catalog::load_from_roots(
        &root,
        &resolved.media_toolkit_root,
        &resolved.decisions_toolkit_root,
    )?;
    println!("L3MS stack profiles ({}):", catalog.list().len());
    for profile in catalog.list() {
        println!("{}", format_stack_profile(profile));
    }
    Ok(())
}

fn format_stack_profile(profile: &CatalogProfile) -> String {
    let tasks = profile
        .tasks
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let device = match profile.device.as_ref() {
        Some(Device::Cpu) => "cpu".to_owned(),
        Some(Device::Cuda) => "cuda".to_owned(),
        Some(Device::Remote) => "remote".to_owned(),
        Some(Device::Other(value)) => value.clone(),
        None => "unspecified".to_owned(),
    };
    let readiness = match profile.readiness {
        Readiness::ManifestPresent => "manifest-only",
        Readiness::Installed => "installed",
        Readiness::Healthy => "healthy",
    };
    format!(
        "  {:<36} tasks={:<20} runtime={:<24} device={:<11} readiness={}{}\n    source={}",
        profile.key,
        tasks,
        profile.runtime,
        device,
        readiness,
        if profile.disabled { " disabled" } else { "" },
        profile.source_path.display()
    )
}

fn interactive_run(filter: &str) -> Result<u8> {
    let client = SwapClient::from_env()?;
    let models = filter_models(client.list_models()?, filter);
    if models.is_empty() {
        eprintln!("No llama-swap models found for filter: {filter:?}");
        return Ok(1);
    }

    print_model_list(&models);
    let Some(index) = choose_index(models.len(), "model")? else {
        println!("Cancelled.");
        return Ok(0);
    };

    let model = &models[index];
    println!("POST {}/models/load  model={}", client.base_url(), model.id);
    println!("{}", client.load_model(&model.id)?);
    Ok(0)
}

fn interactive_bench(filter: &str, extra: &str) -> Result<u8> {
    let extra_args = parse_extra_args(extra)?;
    let root = repository_root()?;
    let scripts = filter_scripts(collect_bench_scripts(&root)?, &root, filter);
    if scripts.is_empty() {
        eprintln!("No benchmark scripts found for filter: {filter:?}");
        return Ok(1);
    }

    print_script_list(&root, &scripts);
    let Some(index) = choose_index(scripts.len(), "script")? else {
        println!("Cancelled.");
        return Ok(0);
    };

    let command = command_for_script(&scripts[index], &extra_args);
    println!("$ {}", format_command(&command));
    let status = Command::new(&command.program)
        .args(&command.args)
        .current_dir(&root)
        .status()
        .with_context(|| format!("failed to execute {}", scripts[index].display()))?;
    let code = status.code().unwrap_or(1);
    println!("Exited with code {code}");
    Ok(u8::try_from(code).unwrap_or(1))
}

fn collect_bench_scripts(root: &Path) -> Result<Vec<PathBuf>> {
    collect_scripts_in(root, ScriptMode::Bench)
}

fn filter_models(models: Vec<SwapModel>, filter: &str) -> Vec<SwapModel> {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return models;
    }

    models
        .into_iter()
        .filter(|model| {
            [&model.id, &model.name, &model.description]
                .iter()
                .any(|value| value.to_ascii_lowercase().contains(&filter))
        })
        .collect()
}

fn filter_scripts(scripts: Vec<PathBuf>, root: &Path, filter: &str) -> Vec<PathBuf> {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return scripts;
    }

    scripts
        .into_iter()
        .filter(|script| {
            let relative = script.strip_prefix(root).unwrap_or(script);
            relative
                .to_string_lossy()
                .to_ascii_lowercase()
                .contains(&filter)
                || pretty_name(script).to_ascii_lowercase().contains(&filter)
        })
        .collect()
}

fn pretty_name(path: &Path) -> String {
    let stem = path.file_stem().and_then(OsStr::to_str).unwrap_or_default();
    for prefix in ["bench-ik-llama-cpp-", "bench-llama-cpp-", "bench-"] {
        if let Some(name) = stem.strip_prefix(prefix) {
            return name.to_owned();
        }
    }
    stem.to_owned()
}

fn print_model_list(models: &[SwapModel]) {
    if models.is_empty() {
        println!("No llama-swap models found");
        return;
    }

    println!("RUN models ({}):", models.len());
    for (index, model) in models.iter().enumerate() {
        println!(
            "  {:>2}. {:<10} {:<36} {}",
            index + 1,
            model.state,
            model.id,
            model.name
        );
    }
}

fn print_script_list(root: &Path, scripts: &[PathBuf]) {
    if scripts.is_empty() {
        println!("No benchmark scripts found");
        return;
    }

    println!("BENCH scripts ({}):", scripts.len());
    for (index, script) in scripts.iter().enumerate() {
        let relative = script.strip_prefix(root).unwrap_or(script);
        println!(
            "  {:>2}. {:<36} {}",
            index + 1,
            pretty_name(script),
            relative.display()
        );
    }
}

fn print_result_list(root: &Path) -> Result<()> {
    let mut loaded = load_results_in(root)?;
    sort_results(&mut loaded.results, ResultSort::TimestampDesc);
    if loaded.results.is_empty() {
        println!("BENCH results (0): no result records found");
    } else {
        println!("BENCH results ({}):", loaded.results.len());
        for (index, result) in loaded.results.iter().enumerate() {
            let pp = result
                .pp_ts
                .map_or_else(|| "—".to_owned(), |value| format!("{value:.2}"));
            let tg = result
                .tg_ts
                .map_or_else(|| "—".to_owned(), |value| format!("{value:.2}"));
            println!(
                "  {:>2}. {:<28} {:<12} pp={:<8} tg={:<8} {}",
                index + 1,
                result.model_key,
                result.strategy,
                pp,
                tg,
                result.ts
            );
        }
    }
    for issue in loaded.issues {
        let line = issue
            .line
            .map_or_else(String::new, |line| format!(":{line}"));
        eprintln!(
            "warning: {}{}: {}",
            issue.source.display(),
            line,
            issue.error
        );
    }
    if loaded.truncated_rows > 0 {
        eprintln!("warning: truncated {} result rows", loaded.truncated_rows);
    }
    Ok(())
}

fn compare_result_files(root: &Path, paths: &[PathBuf]) -> Result<()> {
    anyhow::ensure!(
        paths.len() == 2,
        "--compare-results requires exactly two files"
    );
    let load = load_results_in(root)?;
    let records = paths
        .iter()
        .map(|path| {
            let candidate = if path.is_absolute() {
                path.clone()
            } else {
                root.join(path)
            };
            load.results
                .iter()
                .find(|result| result.source == candidate)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no result record found for {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let comparison = crate::bench_results::compare_results(&records[0], &records[1]);
    println!("BENCH comparison");
    println!("  left:  {}", records[0].source.display());
    println!("  right: {}", records[1].source.display());
    for (name, delta) in comparison.metrics {
        let value = delta
            .delta
            .map_or_else(|| "—".to_owned(), |value| format!("{value:+.3}"));
        println!(
            "  {name}: left={} right={} delta={value}",
            delta
                .left
                .map_or_else(|| "—".to_owned(), |value| format!("{value:.3}")),
            delta
                .right
                .map_or_else(|| "—".to_owned(), |value| format!("{value:.3}"))
        );
    }
    Ok(())
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

fn repository_root() -> Result<PathBuf> {
    if let Some(root) = env::var_os("L3MS_ROOT").map(PathBuf::from) {
        if is_repository_root(&root) {
            return Ok(root);
        }
        bail!("L3MS_ROOT is not an L3MS repository: {}", root.display());
    }

    if let Ok(executable) = env::current_exe() {
        if let Some(root) = find_repository_root(&executable) {
            return Ok(root);
        }
    }
    if let Ok(cwd) = env::current_dir() {
        if let Some(root) = find_repository_root(&cwd) {
            return Ok(root);
        }
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = find_repository_root(manifest_dir) {
        return Ok(root);
    }

    bail!("could not locate the L3MS repository; run from the checkout or set L3MS_ROOT")
}

fn find_repository_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|candidate| is_repository_root(candidate))
        .map(Path::to_path_buf)
}

fn is_repository_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("llama-swap.yaml").is_file()
        && path.join("bench-models").is_dir()
}

fn direct_native_action(cli: &Cli) -> Result<Option<u8>> {
    let Some(spec) = direct_native_spec(cli)? else {
        return Ok(None);
    };
    run_direct_native(&spec).map(Some)
}

#[cfg(target_os = "linux")]
fn direct_comfy_adapter_from_environment() -> Result<crate::control::comfy::ComfyAdapter> {
    fn optional_environment_value(name: &str) -> Result<Option<String>> {
        env::var_os(name)
            .map(|value| {
                value
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("Comfy environment configuration is invalid"))
            })
            .transpose()
    }

    let scoped_jobs = optional_environment_value("L3MS_COMFY_SCOPED_JOBS")?;
    let endpoint = optional_environment_value("L2M2_COMFY_URL")?;
    let api_key = optional_environment_value("L3MS_COMFY_API_KEY")?;
    direct_comfy_adapter(
        scoped_jobs.as_deref(),
        endpoint.as_deref(),
        api_key.as_deref(),
    )
}

fn direct_comfy_adapter(
    scoped_jobs_assertion: Option<&str>,
    endpoint: Option<&str>,
    api_key: Option<&str>,
) -> Result<crate::control::comfy::ComfyAdapter> {
    if scoped_jobs_assertion != Some("1") {
        bail!(
            "Comfy dispatch requires L3MS_COMFY_SCOPED_JOBS=1 asserting canonical IDs and atomic scoped cancellation"
        );
    }
    crate::control::comfy::ComfyAdapter::new(
        endpoint.unwrap_or(DEFAULT_DIRECT_COMFY_URL),
        api_key,
        crate::control::comfy::ComfyContract::CanonicalIdsAndAtomicScopedCancellation,
    )
    .map_err(|_| anyhow::anyhow!("Comfy adapter configuration is invalid"))
}

#[cfg(target_os = "linux")]
fn absolute_comfy_output_path(path: &Path) -> Result<PathBuf> {
    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .context("working directory is unavailable")?
            .join(path)
    };
    Ok(absolute_path)
}

#[cfg(target_os = "linux")]
fn run_direct_native(spec: &DirectNativeSpec<'_>) -> Result<u8> {
    let profile = spec
        .profile
        .parse::<crate::control::catalog::ProfileKey>()
        .map_err(|_| anyhow::anyhow!("--run-profile must be a valid namespaced profile ID"))?;
    let expected_lane = match spec.task {
        "decision" => crate::control::catalog::Lane::Decisions,
        "tts" | "clon" | "design" | "image" | "video" => crate::control::catalog::Lane::Media,
        "embeddings" => crate::control::catalog::Lane::Language,
        _ => bail!(
            "--run-profile supports only --task decision, --task tts, --task clon, --task design, --task image, --task video, or --task embeddings"
        ),
    };
    if profile.lane != expected_lane {
        bail!("--run-profile profile lane does not match --task");
    }
    let task = crate::control::catalog::Task::new(spec.task.to_owned())
        .map_err(|_| anyhow::anyhow!("--task is invalid"))?;
    let input = read_direct_native_input(spec.input, spec.task)?;
    let is_comfy_workflow = matches!(spec.task, "image" | "video");
    let comfy_output_directory = if is_comfy_workflow {
        Some(absolute_comfy_output_path(spec.output_dir.ok_or_else(
            || anyhow::anyhow!("Comfy output directory is required"),
        )?)?)
    } else {
        None
    };

    let repository_root =
        repository_root().map_err(|_| anyhow::anyhow!("repository location is unavailable"))?;
    let data_root = crate::state_store::data_root()
        .map_err(|_| anyhow::anyhow!("stack settings are unavailable"))?;
    let settings = load_current_stack_settings_in(&repository_root, &data_root)
        .map_err(|_| anyhow::anyhow!("stack settings are unavailable"))?;
    let catalog = Catalog::load_from_roots(
        &repository_root,
        &settings.media_toolkit_root,
        &settings.decisions_toolkit_root,
    )
    .map_err(|_| anyhow::anyhow!("stack catalog is unavailable"))?;

    let (fields, output_dir) = match input {
        DirectNativeInput::Decision(input) => (
            crate::control::task_form::TaskFields::Decisions {
                state: input.state,
                questions: input.questions,
            },
            None,
        ),
        DirectNativeInput::Speech(input) => (
            crate::control::task_form::TaskFields::Speech {
                text: input.text,
                mode: input.mode.native_mode(),
                instruction: None,
                voice_reference: None,
                reference_text: None,
            },
            spec.output_dir,
        ),
        DirectNativeInput::VoiceClone(input) => (
            crate::control::task_form::TaskFields::Speech {
                text: input.text,
                mode: crate::control::requests::SpeechMode::VoiceClone,
                instruction: None,
                voice_reference: Some(input.voice_reference),
                reference_text: Some(input.reference_text),
            },
            spec.output_dir,
        ),
        DirectNativeInput::VoiceDesign(input) => (
            crate::control::task_form::TaskFields::Speech {
                text: input.text,
                mode: crate::control::requests::SpeechMode::VoiceDesign,
                instruction: Some(input.instruction),
                voice_reference: None,
                reference_text: None,
            },
            spec.output_dir,
        ),
        DirectNativeInput::ComfyWorkflow(graph) => (
            crate::control::task_form::TaskFields::MediaWorkflow { graph },
            spec.output_dir,
        ),
        DirectNativeInput::Embeddings(input) => (
            crate::control::task_form::TaskFields::LanguageEmbeddings { input: input.input },
            None,
        ),
    };
    let draft = crate::control::task_form::TaskFormDraft::new(
        profile,
        task,
        "direct".to_owned(),
        spec.timeout_ms,
        fields,
    );
    let (request, timeout) = crate::control::task_form::build_direct_native(&catalog, draft)
        .map_err(|_| anyhow::anyhow!("native request is invalid"))?;
    let output_dir = comfy_output_directory.as_deref().or(output_dir);
    let language = if spec.task == "embeddings" {
        let endpoint = env::var("LLAMA_SWAP_URL")
            .unwrap_or_else(|_| crate::llama_swap::DEFAULT_BASE_URL.to_owned());
        let api_key = env::var("LLAMA_SWAP_API_KEY").ok();
        Some(
            crate::control::language::LanguageAdapter::new(&endpoint, api_key.as_deref())
                .map_err(|_| anyhow::anyhow!("language adapter configuration is invalid"))?,
        )
    } else {
        None
    };
    let comfy_adapter = if is_comfy_workflow {
        Some(direct_comfy_adapter_from_environment()?)
    } else {
        None
    };
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| anyhow::anyhow!("native task timeout is invalid"))?;
    let signals = crate::control::foreground_shutdown::ShutdownSignals::install()
        .map_err(|_| anyhow::anyhow!("could not install native task cancellation handler"))?;
    let context = crate::control::runtime::OperationContext::new(deadline, &signals);
    let native_id = if comfy_adapter.is_some() {
        let candidate = crate::control::comfy::fresh_comfy_candidate(&context)
            .map_err(|_| anyhow::anyhow!("Comfy job ID could not be generated"))?;
        eprintln!("native_job_id={candidate}");
        Some(candidate)
    } else {
        None
    };
    let connection = match (
        language.as_ref(),
        comfy_adapter.as_ref(),
        native_id.as_deref(),
    ) {
        (Some(adapter), None, None) => Some(
            crate::control::direct_run::DirectConnection::Language(adapter),
        ),
        (None, Some(adapter), Some(native_id)) => {
            Some(crate::control::direct_run::DirectConnection::Comfy { adapter, native_id })
        }
        (None, None, None) => None,
        _ => bail!("native connection configuration is invalid"),
    };
    let result = crate::control::direct_run::run_native(
        &catalog, &request, output_dir, connection, &context,
    )
    .map_err(|error| {
        if spec.task == "embeddings" && error.blocks_following_work() {
            anyhow::anyhow!("remote completion is unconfirmed; do not retry automatically")
        } else if is_comfy_workflow && error.blocks_following_work() {
            anyhow::anyhow!(
                "Comfy completion is unconfirmed; do not retry or resubmit automatically"
            )
        } else {
            anyhow::anyhow!(error.to_string())
        }
    })?;
    let output = match result {
        crate::control::direct_run::DirectRunResult::Decision(result) => {
            direct_decision_output_json(&result)?
        }
        crate::control::direct_run::DirectRunResult::Speech(result) => {
            direct_speech_output_json(&result)?
        }
        crate::control::direct_run::DirectRunResult::Embeddings(result) => {
            direct_embedding_output_json(&result)?
        }
        crate::control::direct_run::DirectRunResult::Comfy(result) => {
            if native_id.as_deref() != Some(result.native_id.as_str()) {
                bail!("native Comfy result does not match the dispatched job ID");
            }
            direct_comfy_output_json(
                &result.native_id,
                &result.output_root,
                result.artifacts.iter().map(|artifact| {
                    (
                        Path::new(&artifact.relative_path),
                        artifact.node_id.as_str(),
                        artifact.media_type.as_str(),
                    )
                }),
            )?
        }
    };
    println!("{output}");
    Ok(0)
}

#[cfg(not(target_os = "linux"))]
fn run_direct_native(_: &DirectNativeSpec<'_>) -> Result<u8> {
    bail!("direct native execution is supported only on Linux");
}

#[cfg(target_os = "linux")]
enum DirectNativeInput {
    Decision(DirectDecisionInput),
    Speech(DirectSpeechInput),
    VoiceClone(DirectVoiceCloneInput),
    VoiceDesign(DirectVoiceDesignInput),
    ComfyWorkflow(serde_json::Value),
    Embeddings(DirectEmbeddingInput),
}

#[cfg(target_os = "linux")]
fn read_direct_native_input(path: &Path, task: &str) -> Result<DirectNativeInput> {
    use std::os::unix::fs::OpenOptionsExt;

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| anyhow::anyhow!("native input file is unavailable"))?;
    let metadata = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("native input file is unavailable"))?;
    if !metadata.is_file() || metadata.len() > MAX_DIRECT_NATIVE_INPUT_BYTES as u64 {
        bail!("native input must be a regular file no larger than 1 MiB");
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take((MAX_DIRECT_NATIVE_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| anyhow::anyhow!("native input file could not be read"))?;
    match task {
        "decision" => parse_direct_decision_input(&bytes)
            .map(DirectNativeInput::Decision)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        "tts" => parse_direct_speech_input(&bytes)
            .map(DirectNativeInput::Speech)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        "clon" => parse_direct_voice_clone_input(&bytes)
            .map(DirectNativeInput::VoiceClone)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        "design" => parse_direct_voice_design_input(&bytes)
            .map(DirectNativeInput::VoiceDesign)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        "image" | "video" => parse_direct_comfy_input(&bytes)
            .map(DirectNativeInput::ComfyWorkflow)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        "embeddings" => parse_direct_embedding_input(&bytes)
            .map(DirectNativeInput::Embeddings)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        _ => bail!(
            "--run-profile supports only --task decision, --task tts, --task clon, --task design, --task image, --task video, or --task embeddings"
        ),
    }
}

#[derive(serde::Serialize)]
struct DirectDecisionOutput<'a> {
    answers: &'a serde_json::Map<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_latency_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_usage: Option<&'a serde_json::Map<String, serde_json::Value>>,
}

fn direct_decision_output_json(
    result: &crate::control::decision::NativeDecisionResult,
) -> Result<String> {
    let crate::control::requests::InferenceResult::Decision { answers } = &result.result else {
        bail!("native decision runner returned an incompatible result");
    };
    let response = DirectDecisionOutput {
        answers,
        native_latency_ns: result
            .native_latency
            .map(|latency| u64::try_from(latency.as_nanos()))
            .transpose()
            .map_err(|_| anyhow::anyhow!("native decision latency is out of range"))?,
        native_usage: result.native_usage.as_ref(),
    };
    let mut writer = BoundedJsonWriter::new(MAX_DIRECT_NATIVE_OUTPUT_BYTES);
    if serde_json::to_writer(&mut writer, &response).is_err() {
        if writer.exceeded {
            bail!("native decision output exceeds the 2 MiB limit");
        }
        bail!("native decision output could not be encoded");
    }
    String::from_utf8(writer.bytes)
        .map_err(|_| anyhow::anyhow!("native decision output could not be encoded"))
}

#[derive(serde::Serialize)]
struct DirectSpeechOutput {
    mode: &'static str,
    artifacts: Vec<DirectSpeechArtifactOutput>,
}

#[derive(serde::Serialize)]
struct DirectSpeechArtifactOutput {
    path: PathBuf,
    native_kind: &'static str,
}

fn direct_speech_output_json(
    result: &crate::control::media_execution::MediaExecutionResult,
) -> Result<String> {
    let artifacts = result
        .artifacts()
        .iter()
        .map(|artifact| DirectSpeechArtifactOutput {
            path: result.output_root().join(artifact.relative_path()),
            native_kind: match artifact.kind() {
                crate::control::media_execution::MediaArtifactKind::FinalWave => "final_wave",
                crate::control::media_execution::MediaArtifactKind::StreamingChunk => {
                    "streaming_chunk"
                }
            },
        })
        .collect();
    direct_speech_output_parts_json(result.mode(), artifacts)
}

fn direct_speech_output_parts_json(
    mode: crate::control::requests::SpeechMode,
    artifacts: Vec<DirectSpeechArtifactOutput>,
) -> Result<String> {
    let response = DirectSpeechOutput {
        mode: match mode {
            crate::control::requests::SpeechMode::Standard => "standard",
            crate::control::requests::SpeechMode::VoiceDesign => "voice_design",
            crate::control::requests::SpeechMode::VoiceClone => "voice_clone",
            crate::control::requests::SpeechMode::Streaming => "streaming",
        },
        artifacts,
    };
    let mut writer = BoundedJsonWriter::new(MAX_DIRECT_NATIVE_OUTPUT_BYTES);
    if serde_json::to_writer(&mut writer, &response).is_err() {
        if writer.exceeded {
            bail!("native speech output exceeds the 2 MiB limit");
        }
        bail!("native speech output could not be encoded");
    }
    String::from_utf8(writer.bytes)
        .map_err(|_| anyhow::anyhow!("native speech output could not be encoded"))
}

fn direct_embedding_output_json(
    result: &crate::control::requests::EmbeddingPayload,
) -> Result<String> {
    let mut writer = BoundedJsonWriter::new(crate::control::requests::MAX_EMBEDDING_JSON_BYTES);
    if serde_json::to_writer(&mut writer, result).is_err() {
        if writer.exceeded {
            bail!("native embedding output exceeds the 4 MiB limit");
        }
        bail!("native embedding output could not be encoded");
    }
    String::from_utf8(writer.bytes)
        .map_err(|_| anyhow::anyhow!("native embedding output could not be encoded"))
}

#[derive(serde::Serialize)]
struct DirectComfyOutput {
    native_job_id: String,
    artifacts: Vec<DirectComfyArtifactOutput>,
}

#[derive(serde::Serialize)]
struct DirectComfyArtifactOutput {
    path: PathBuf,
    native_media_type: String,
    node_id: String,
}

fn direct_comfy_output_json<'a>(
    native_job_id: &str,
    output_root: &Path,
    artifacts: impl IntoIterator<Item = (&'a Path, &'a str, &'a str)>,
) -> Result<String> {
    if !output_root.is_absolute() {
        bail!("Comfy output directory is invalid");
    }
    let mut serialized_artifacts = Vec::new();
    for (relative_path, node_id, media_type) in artifacts {
        if relative_path.as_os_str().is_empty()
            || relative_path.is_absolute()
            || !relative_path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
            || relative_path
                .to_string_lossy()
                .contains(['\\', ':', '%', '?', '#'])
            || node_id.is_empty()
            || media_type.is_empty()
            || node_id.chars().any(char::is_control)
            || media_type.chars().any(char::is_control)
        {
            bail!("Comfy output artifact is invalid");
        }
        serialized_artifacts.push(DirectComfyArtifactOutput {
            path: output_root.join(relative_path),
            native_media_type: media_type.to_owned(),
            node_id: node_id.to_owned(),
        });
    }
    let response = DirectComfyOutput {
        native_job_id: native_job_id.to_owned(),
        artifacts: serialized_artifacts,
    };
    let mut writer = BoundedJsonWriter::new(MAX_DIRECT_NATIVE_OUTPUT_BYTES);
    if serde_json::to_writer(&mut writer, &response).is_err() {
        if writer.exceeded {
            bail!("native Comfy output exceeds the 2 MiB limit");
        }
        bail!("native Comfy output could not be encoded");
    }
    String::from_utf8(writer.bytes)
        .map_err(|_| anyhow::anyhow!("native Comfy output could not be encoded"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn hidden_engine_action_is_exact_and_malformed_launch_stays_redacted() {
        assert!(isolated_engine_action(vec![OsString::from("--quickstart")]).is_none());
        assert!(isolated_engine_action(Vec::new()).is_none());
        let error =
            isolated_engine_action(vec![OsString::from("--l3ms-isolated-engine-supervisor-v1")])
                .unwrap()
                .unwrap_err();
        assert_eq!(
            error.to_string(),
            "isolated engine helper: invalid launch arguments"
        );
    }

    #[test]
    fn direct_native_action_requires_explicit_inputs_and_uses_bounded_default() {
        let cli = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "decisions:nagi-smol-cpu",
            "--task",
            "decision",
            "--input",
            "input.json",
        ])
        .unwrap();
        let Some(spec) = direct_native_spec(&cli).unwrap() else {
            panic!("direct native action is present");
        };
        assert_eq!(spec.profile, "decisions:nagi-smol-cpu");
        assert_eq!(spec.task, "decision");
        assert_eq!(spec.input, Path::new("input.json"));
        assert_eq!(spec.output_dir, None);
        assert_eq!(spec.timeout_ms, DEFAULT_DIRECT_NATIVE_TIMEOUT_MS);

        let cli = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "decisions:nagi-smol-cpu",
            "--task",
            "decision",
            "--input",
            "input.json",
            "--task-timeout-ms",
            "1200",
        ])
        .unwrap();
        let Some(spec) = direct_native_spec(&cli).unwrap() else {
            panic!("direct native action is present");
        };
        assert_eq!(spec.timeout_ms, 1200);
    }

    #[test]
    fn direct_native_speech_requires_output_directory_and_rejects_it_elsewhere() {
        let args = [
            "l3ms",
            "--run-profile",
            "media:breeze-tts-2-cpu",
            "--task",
            "tts",
            "--input",
            "input.json",
            "--output-dir",
            "/private/job-1",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        let Some(spec) = direct_native_spec(&cli).unwrap() else {
            panic!("direct speech action is present");
        };
        assert_eq!(spec.task, "tts");
        assert_eq!(spec.output_dir, Some(Path::new("/private/job-1")));

        let missing_output = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "media:breeze-tts-2-cpu",
            "--task",
            "tts",
            "--input",
            "input.json",
        ])
        .unwrap();
        assert!(direct_native_spec(&missing_output).is_err());

        let decision_with_output = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "decisions:nagi-smol-cpu",
            "--task",
            "decision",
            "--input",
            "input.json",
            "--output-dir",
            "/private/job-1",
        ])
        .unwrap();
        assert!(direct_native_spec(&decision_with_output).is_err());

        let clone = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "media:breeze-tts-2-cpu",
            "--task",
            "clon",
            "--input",
            "input.json",
            "--output-dir",
            "/private/job-1",
        ])
        .unwrap();
        let Some(clone_spec) = direct_native_spec(&clone).unwrap() else {
            panic!("direct clone action is present");
        };
        assert_eq!(clone_spec.task, "clon");
        assert_eq!(clone_spec.output_dir, Some(Path::new("/private/job-1")));
        let clone_without_output = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "media:breeze-tts-2-cpu",
            "--task",
            "clon",
            "--input",
            "input.json",
        ])
        .unwrap();
        assert!(direct_native_spec(&clone_without_output).is_err());

        assert!(Cli::try_parse_from(["l3ms", "--output-dir", "/private/job-1"]).is_err());
    }

    #[test]
    fn direct_native_voice_design_requires_output_directory() {
        let cli = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "media:breeze-tts-2-cpu",
            "--task",
            "design",
            "--input",
            "input.json",
            "--output-dir",
            "/private/job-1",
        ])
        .unwrap();
        let Some(spec) = direct_native_spec(&cli).unwrap() else {
            panic!("direct voice-design action is present");
        };
        assert_eq!(spec.profile, "media:breeze-tts-2-cpu");
        assert_eq!(spec.task, "design");
        assert_eq!(spec.output_dir, Some(Path::new("/private/job-1")));

        let missing_output = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "media:breeze-tts-2-cpu",
            "--task",
            "design",
            "--input",
            "input.json",
        ])
        .unwrap();
        assert!(direct_native_spec(&missing_output).is_err());
    }

    #[test]
    fn direct_native_comfy_tasks_require_explicit_output_directory() {
        for task in ["image", "video"] {
            let cli = Cli::try_parse_from([
                "l3ms",
                "--run-profile",
                "media:fixture-comfy",
                "--task",
                task,
                "--input",
                "workflow.json",
                "--output-dir",
                "/private/job-1",
            ])
            .unwrap();
            let Some(spec) = direct_native_spec(&cli).unwrap() else {
                panic!("Comfy workflow action is present");
            };
            assert_eq!(spec.task, task);
            assert_eq!(spec.output_dir, Some(Path::new("/private/job-1")));

            let missing_output = Cli::try_parse_from([
                "l3ms",
                "--run-profile",
                "media:fixture-comfy",
                "--task",
                task,
                "--input",
                "workflow.json",
            ])
            .unwrap();
            assert!(direct_native_spec(&missing_output).is_err());
        }
    }

    #[test]
    fn direct_native_embeddings_require_language_profile_and_reject_output_directory() {
        let cli = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "language:nomic-embed-text-v1.5",
            "--task",
            "embeddings",
            "--input",
            "input.json",
        ])
        .unwrap();
        let Some(spec) = direct_native_spec(&cli).unwrap() else {
            panic!("direct embeddings action is present");
        };
        assert_eq!(spec.task, "embeddings");
        assert_eq!(spec.output_dir, None);

        let with_output_dir = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "language:nomic-embed-text-v1.5",
            "--task",
            "embeddings",
            "--input",
            "input.json",
            "--output-dir",
            "/private/job-1",
        ])
        .unwrap();
        assert!(direct_native_spec(&with_output_dir).is_err());
    }

    #[test]
    fn direct_native_action_rejects_other_actions_extra_and_coordinator_options() {
        assert!(Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "decisions:nagi-smol-cpu",
            "--task",
            "decision",
            "--input",
            "input.json",
            "--quickstart",
        ])
        .is_err());

        let cli = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "decisions:nagi-smol-cpu",
            "--task",
            "decision",
            "--input",
            "input.json",
            "--extra",
            "--unsafe",
        ])
        .unwrap();
        assert!(direct_native_spec(&cli).is_err());

        let cli = Cli::try_parse_from([
            "l3ms",
            "--run-profile",
            "decisions:nagi-smol-cpu",
            "--task",
            "decision",
            "--input",
            "input.json",
            "--coordinator-socket-dir",
            "/private/socket",
        ])
        .unwrap();
        assert!(direct_native_spec(&cli).is_err());
    }

    #[test]
    fn direct_decision_input_requires_unique_known_top_level_fields_and_is_bounded() {
        let valid = br#"{"state":{"ready":true},"questions":{"route":{"type":"choice","options":["yes"]}}}"#;
        let parsed = parse_direct_decision_input(valid).unwrap();
        assert_eq!(parsed.state["ready"], true);
        assert!(parsed.questions.contains_key("route"));

        for invalid in [
            br#"{"state":{},"state":{},"questions":{}}"#.as_slice(),
            br#"{"state":{},"questions":{},"unexpected":true}"#.as_slice(),
            br#"{"state":{}}"#.as_slice(),
            br#"[]"#.as_slice(),
            br#"{"state":{},"questions":{}} trailing"#.as_slice(),
        ] {
            assert!(parse_direct_decision_input(invalid).is_err());
        }

        let oversized = vec![b' '; MAX_DIRECT_NATIVE_INPUT_BYTES + 1];
        assert_eq!(
            parse_direct_decision_input(&oversized).err(),
            Some(DirectInputError::TooLarge)
        );
    }

    #[test]
    fn direct_speech_input_defaults_to_offline_and_accepts_streaming_mode() {
        let parsed = parse_direct_speech_input(br#"{"text":"hello"}"#).unwrap();
        assert_eq!(parsed.text, "hello");
        assert_eq!(parsed.mode, DirectSpeechMode::Offline);
        assert_eq!(
            parsed.mode.native_mode(),
            crate::control::requests::SpeechMode::Standard
        );

        let offline = parse_direct_speech_input(br#"{"text":"hello","mode":"offline"}"#).unwrap();
        assert_eq!(offline.mode, DirectSpeechMode::Offline);

        let streamed =
            parse_direct_speech_input(br#"{"text":"hello","mode":"streaming"}"#).unwrap();
        assert_eq!(streamed.mode, DirectSpeechMode::Streaming);
        assert_eq!(
            streamed.mode.native_mode(),
            crate::control::requests::SpeechMode::Streaming
        );

        for invalid in [
            br#"{"text":"hello","text":"again"}"#.as_slice(),
            br#"{"text":"hello","unexpected":true}"#.as_slice(),
            br#"{"text":1}"#.as_slice(),
            br#"{"text":"hello","mode":"invalid"}"#.as_slice(),
            br#"{"text":"hello","mode":"offline","mode":"streaming"}"#.as_slice(),
            br#"[]"#.as_slice(),
            br#"{"text":"hello"} trailing"#.as_slice(),
        ] {
            assert!(parse_direct_speech_input(invalid).is_err());
        }

        let oversized = vec![b' '; MAX_DIRECT_NATIVE_INPUT_BYTES + 1];
        assert_eq!(
            parse_direct_speech_input(&oversized).err(),
            Some(DirectInputError::TooLarge)
        );
    }

    #[test]
    fn direct_voice_clone_input_requires_all_known_fields_and_is_bounded() {
        let parsed = parse_direct_voice_clone_input(
            br#"{"text":"spoken words","voice_reference":"samples/reference.wav","reference_text":"reference words"}"#,
        )
        .unwrap();
        assert_eq!(parsed.text, "spoken words");
        assert_eq!(parsed.voice_reference, "samples/reference.wav");
        assert_eq!(parsed.reference_text, "reference words");

        for invalid in [
            br#"{"text":"spoken words","voice_reference":"samples/reference.wav"}"#.as_slice(),
            br#"{"text":"spoken words","voice_reference":"samples/reference.wav","reference_text":"reference words","extra":true}"#.as_slice(),
            br#"{"text":"spoken words","text":"again","voice_reference":"samples/reference.wav","reference_text":"reference words"}"#.as_slice(),
            br#"{"text":1,"voice_reference":"samples/reference.wav","reference_text":"reference words"}"#.as_slice(),
            br#"[]"#.as_slice(),
            br#"{"text":"spoken words","voice_reference":"samples/reference.wav","reference_text":"reference words"} trailing"#.as_slice(),
        ] {
            assert!(parse_direct_voice_clone_input(invalid).is_err());
        }

        let oversized = vec![b' '; MAX_DIRECT_NATIVE_INPUT_BYTES + 1];
        assert_eq!(
            parse_direct_voice_clone_input(&oversized).err(),
            Some(DirectInputError::TooLarge)
        );
    }

    #[test]
    fn direct_voice_design_input_is_strict_duplicate_safe_and_bounded() {
        let parsed = parse_direct_voice_design_input(
            br#"{"text":"hello","instruction":"warm, clear alto"}"#,
        )
        .unwrap();
        assert_eq!(parsed.text, "hello");
        assert_eq!(parsed.instruction, "warm, clear alto");

        for invalid in [
            br#"{"text":"hello","text":"again","instruction":"warm"}"#.as_slice(),
            br#"{"text":"hello","instruction":"warm","instruction":"bright"}"#.as_slice(),
            br#"{"text":"hello","instruction":"warm","unexpected":true}"#.as_slice(),
            br#"{"text":"hello"}"#.as_slice(),
            br#"{"text":1,"instruction":"warm"}"#.as_slice(),
            br#"[]"#.as_slice(),
            br#"{"text":"hello","instruction":"warm"} trailing"#.as_slice(),
        ] {
            assert!(parse_direct_voice_design_input(invalid).is_err());
        }

        let oversized = vec![b' '; MAX_DIRECT_NATIVE_INPUT_BYTES + 1];
        assert_eq!(
            parse_direct_voice_design_input(&oversized).err(),
            Some(DirectInputError::TooLarge)
        );
    }

    #[test]
    fn direct_comfy_input_requires_strict_nonempty_workflow_object() {
        let parsed = parse_direct_comfy_input(
            br#"{"graph":{"3":{"class_type":"LoadImage","inputs":{"image":"fixture.png"}}}}"#,
        )
        .unwrap();
        assert!(parsed.get("3").is_some());

        for invalid in [
            br#"{"graph":{},"graph":{}}"#.as_slice(),
            br#"{"graph":{},"unexpected":true}"#.as_slice(),
            br#"{"graph":[]}"#.as_slice(),
            br#"{"graph":{}}"#.as_slice(),
            br#"[]"#.as_slice(),
            br#"{"graph":{"3":{}}} trailing"#.as_slice(),
        ] {
            assert!(parse_direct_comfy_input(invalid).is_err());
        }

        let oversized = vec![b' '; MAX_DIRECT_NATIVE_INPUT_BYTES + 1];
        assert_eq!(
            parse_direct_comfy_input(&oversized).err(),
            Some(DirectInputError::TooLarge)
        );
    }

    #[test]
    fn direct_comfy_configuration_requires_scoped_jobs_assertion_and_uses_loopback_default() {
        assert!(direct_comfy_adapter(None, None, None).is_err());
        assert!(direct_comfy_adapter(Some("0"), None, None).is_err());
        assert!(direct_comfy_adapter(Some("1"), None, None).is_ok());
        assert!(direct_comfy_adapter(Some("1"), Some("not a URL"), None).is_err());
    }

    #[test]
    fn direct_comfy_output_lists_absolute_validated_artifacts_and_is_bounded() {
        let root = Path::new("/private/job-1");
        let artifact_path = Path::new("images/result.png");
        let output: serde_json::Value = serde_json::from_str(
            &direct_comfy_output_json(
                "85c2c494-e762-4f10-8c45-6a8245b43962",
                root,
                [(artifact_path, "4", "image/png")],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            output["native_job_id"],
            "85c2c494-e762-4f10-8c45-6a8245b43962"
        );
        assert_eq!(
            output["artifacts"][0]["path"],
            "/private/job-1/images/result.png"
        );
        assert_eq!(output["artifacts"][0]["native_media_type"], "image/png");
        assert_eq!(output["artifacts"][0]["node_id"], "4");
        assert!(direct_comfy_output_json(
            "id",
            root,
            [(Path::new("../escape.png"), "4", "image/png")],
        )
        .is_err());

        let oversized_media_type = "x".repeat(MAX_DIRECT_NATIVE_OUTPUT_BYTES);
        assert_eq!(
            direct_comfy_output_json(
                "id",
                root,
                [(Path::new("image.png"), "4", oversized_media_type.as_str())],
            )
            .unwrap_err()
            .to_string(),
            "native Comfy output exceeds the 2 MiB limit"
        );
    }

    #[test]
    fn direct_embedding_input_is_strict_duplicate_safe_and_bounded() {
        let parsed = parse_direct_embedding_input(br#"{"input":["first","second"]}"#).unwrap();
        assert_eq!(
            parsed.input,
            vec![String::from("first"), String::from("second")]
        );

        for invalid in [
            br#"{"input":["first"],"input":["second"]}"#.as_slice(),
            br#"{"input":["first"],"unexpected":true}"#.as_slice(),
            br#"{"input":"first"}"#.as_slice(),
            br#"{}"#.as_slice(),
            br#"[]"#.as_slice(),
            br#"{"input":["first"]} trailing"#.as_slice(),
        ] {
            assert!(parse_direct_embedding_input(invalid).is_err());
        }

        let oversized = vec![b' '; MAX_DIRECT_NATIVE_INPUT_BYTES + 1];
        assert_eq!(
            parse_direct_embedding_input(&oversized).err(),
            Some(DirectInputError::TooLarge)
        );
    }

    #[test]
    fn direct_decision_output_preserves_native_latency_and_enforces_size_limit() {
        let mut answers = serde_json::Map::new();
        answers.insert("answer".into(), serde_json::json!("yes"));
        let result = crate::control::decision::NativeDecisionResult {
            result: crate::control::requests::InferenceResult::Decision { answers },
            native_latency: Some(std::time::Duration::from_nanos(1234)),
            native_usage: None,
        };
        let output: serde_json::Value =
            serde_json::from_str(&direct_decision_output_json(&result).unwrap()).unwrap();
        assert_eq!(output["answers"]["answer"], "yes");
        assert_eq!(output["native_latency_ns"], 1234);
        assert!(output.get("native_usage").is_none());

        let mut answers = serde_json::Map::new();
        answers.insert(
            "answer".into(),
            serde_json::Value::String("x".repeat(MAX_DIRECT_NATIVE_OUTPUT_BYTES)),
        );
        let oversized = crate::control::decision::NativeDecisionResult {
            result: crate::control::requests::InferenceResult::Decision { answers },
            native_latency: None,
            native_usage: None,
        };
        assert!(direct_decision_output_json(&oversized).is_err());
    }

    #[test]
    fn direct_embedding_output_preserves_vectors_and_usage_with_four_mib_limit() {
        let result = crate::control::requests::EmbeddingPayload {
            vectors: vec![vec![0.125, -0.25]],
            usage: crate::control::requests::EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        };
        let output: serde_json::Value =
            serde_json::from_str(&direct_embedding_output_json(&result).unwrap()).unwrap();
        assert_eq!(output["vectors"][0], serde_json::json!([0.125, -0.25]));
        assert_eq!(output["usage"]["prompt_tokens"], 3);
        assert_eq!(output["usage"]["total_tokens"], 3);

        let oversized = crate::control::requests::EmbeddingPayload {
            vectors: vec![vec![1.0; 1_100_000]],
            usage: crate::control::requests::EmbeddingUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
        };
        assert_eq!(
            direct_embedding_output_json(&oversized)
                .unwrap_err()
                .to_string(),
            "native embedding output exceeds the 4 MiB limit"
        );
    }

    #[test]
    fn retired_http_gateway_option_is_not_exposed() {
        assert!(
            Cli::try_parse_from(["l3ms", "--coordinator-http-gateway", "private.json"]).is_err()
        );
    }

    #[test]
    fn direct_voice_design_output_uses_existing_speech_json_shape_and_limit() {
        let artifact = DirectSpeechArtifactOutput {
            path: PathBuf::from("/private/job-1/final.wav"),
            native_kind: "final_wave",
        };
        let output: serde_json::Value = serde_json::from_str(
            &direct_speech_output_parts_json(
                crate::control::requests::SpeechMode::VoiceDesign,
                vec![artifact],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(output["mode"], "voice_design");
        assert_eq!(output["artifacts"][0]["path"], "/private/job-1/final.wav");
        assert_eq!(output["artifacts"][0]["native_kind"], "final_wave");

        for (mode, expected) in [
            (
                crate::control::requests::SpeechMode::VoiceClone,
                "voice_clone",
            ),
            (crate::control::requests::SpeechMode::Streaming, "streaming"),
        ] {
            let output: serde_json::Value =
                serde_json::from_str(&direct_speech_output_parts_json(mode, Vec::new()).unwrap())
                    .unwrap();
            assert_eq!(output["mode"], expected);
            assert_eq!(output["artifacts"], serde_json::json!([]));
        }

        let oversized = DirectSpeechArtifactOutput {
            path: PathBuf::from(format!(
                "/private/{}",
                "x".repeat(MAX_DIRECT_NATIVE_OUTPUT_BYTES)
            )),
            native_kind: "final_wave",
        };
        assert_eq!(
            direct_speech_output_parts_json(
                crate::control::requests::SpeechMode::VoiceDesign,
                vec![oversized]
            )
            .unwrap_err()
            .to_string(),
            "native speech output exceeds the 2 MiB limit"
        );
    }

    fn model(id: &str, name: &str, description: &str) -> SwapModel {
        SwapModel {
            id: id.into(),
            state: "unknown".into(),
            name: name.into(),
            description: description.into(),
            created: None,
            disabled: false,
            size_bytes: None,
        }
    }

    #[test]
    fn parses_optional_filters_and_modes() {
        let cli = Cli::try_parse_from(["l3ms", "--run"]).unwrap();
        assert_eq!(cli.run.as_deref(), Some(""));

        let cli = Cli::try_parse_from(["l3ms", "--run", "qwen"]).unwrap();
        assert_eq!(cli.run.as_deref(), Some("qwen"));

        let cli = Cli::try_parse_from(["l3ms", "--bench"]).unwrap();
        assert_eq!(cli.bench.as_deref(), Some(""));

        assert!(Cli::try_parse_from(["l3ms", "--media", "h3"]).is_err());

        let cli = Cli::try_parse_from(["l3ms", "--list", "all"]).unwrap();
        assert_eq!(cli.list, Some(ListMode::All));
        let cli = Cli::try_parse_from(["l3ms", "--list", "results"]).unwrap();
        assert_eq!(cli.list, Some(ListMode::Results));
        let cli = Cli::try_parse_from(["l3ms", "--list", "prompts"]).unwrap();
        assert_eq!(cli.list, Some(ListMode::Prompts));
        let cli = Cli::try_parse_from(["l3ms", "--list", "stack"]).unwrap();
        assert_eq!(cli.list, Some(ListMode::Stack));
        assert!(
            Cli::try_parse_from(["l3ms", "--settings"])
                .unwrap()
                .settings
        );
        let cli = Cli::try_parse_from([
            "l3ms",
            "--compare-results",
            "bench-results/left.md",
            "bench-results/right.md",
        ])
        .unwrap();
        assert_eq!(cli.compare_results.unwrap().len(), 2);

        let cli = Cli::try_parse_from([
            "l3ms",
            "--coordinator-status",
            "request-1",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
        ])
        .unwrap();
        assert_eq!(cli.coordinator_status.as_deref(), Some("request-1"));
        assert_eq!(
            cli.coordinator_socket_dir.as_deref(),
            Some(Path::new("/run/user/1000/l3ms"))
        );
        assert!(Cli::try_parse_from([
            "l3ms",
            "--coordinator-status",
            "request-1",
            "--coordinator-cancel",
            "request-1",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
        ])
        .is_err());

        assert!(Cli::try_parse_from(["l3ms", "--request-submit", "request.json"]).is_err());
        let cli = Cli::try_parse_from(["l3ms", "--coordinator-cpu-daemon", "daemon.json"]).unwrap();
        assert_eq!(
            cli.coordinator_cpu_daemon.as_deref(),
            Some(Path::new("daemon.json"))
        );
        assert!(Cli::try_parse_from([
            "l3ms",
            "--coordinator-cpu-daemon",
            "daemon.json",
            "--coordinator-status",
            "request-1",
        ])
        .is_err());
        assert!(Cli::try_parse_from([
            "l3ms",
            "--coordinator-cpu-daemon",
            "daemon.json",
            "--request-results",
            "request-1",
        ])
        .is_err());
        assert!(Cli::try_parse_from([
            "l3ms",
            "--coordinator-cpu-daemon",
            "daemon.json",
            "--list",
            "stack",
        ])
        .is_err());
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-results",
            "request-1",
            "--after-cursor",
            "42",
            "--result-page-bytes",
            "4096",
        ])
        .unwrap();
        assert_eq!(cli.request_results.as_deref(), Some("request-1"));
        assert_eq!(cli.after_cursor, Some(42));
        assert_eq!(cli.result_page_bytes, Some(4096));
        assert!(Cli::try_parse_from([
            "l3ms",
            "--request-results",
            "request-1",
            "--coordinator-cancel",
            "request-1",
        ])
        .is_err());
        assert!(Cli::try_parse_from(
            ["l3ms", "--request-results", "request-1", "--list", "stack",]
        )
        .is_err());
        let cli = Cli::try_parse_from(["l3ms", "--coordinator-status", "request-1"]).unwrap();
        assert!(coordinator_action(&cli, None).is_err());
        assert!(Cli::try_parse_from([
            "l3ms",
            "--coordinator-status",
            "request-1",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
            "--list",
            "stack",
        ])
        .is_err());

        assert!(Cli::try_parse_from(["l3ms", "--list", "media"]).is_err());

        let cli = Cli::try_parse_from(["l3ms", "--bench", "qwen", "--extra", "--ctx-size 32768"])
            .unwrap();
        assert_eq!(cli.extra.as_deref(), Some("--ctx-size 32768"));

        assert!(Cli::try_parse_from(["l3ms", "--run", "--bench"]).is_err());
        assert!(Cli::try_parse_from(["l3ms", "--list", "invalid"]).is_err());
    }

    #[test]
    fn coordinator_options_require_action_and_reject_any_extra_argument() {
        let cli = Cli::try_parse_from(["l3ms", "--coordinator-socket-dir", "/run/user/1000/l3ms"])
            .unwrap();
        assert!(coordinator_action(&cli, None).is_err());

        let cli = Cli::try_parse_from(["l3ms", "--after-cursor", "2"]).unwrap();
        assert!(coordinator_action(&cli, None).is_err());
        let cli = Cli::try_parse_from(["l3ms", "--result-page-bytes", "4096"]).unwrap();
        assert!(coordinator_action(&cli, None).is_err());

        let cli = Cli::try_parse_from([
            "l3ms",
            "--coordinator-status",
            "request-1",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
            "--extra",
            "",
        ])
        .unwrap();
        assert!(coordinator_action(&cli, None).is_err());

        let cli = Cli::try_parse_from([
            "l3ms",
            "--coordinator-cancel",
            "request-1",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
        ])
        .unwrap();
        let Some((request, socket_directory)) = coordinator_action(&cli, None).unwrap() else {
            panic!("coordinator action should be present");
        };
        assert_eq!(socket_directory, Path::new("/run/user/1000/l3ms"));
        assert!(matches!(
            request.message,
            crate::control::ipc::WireRequestMessage::Cancel { request_id }
                if request_id == "request-1"
        ));

        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-results",
            "request-3",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
            "--extra",
            "",
        ])
        .unwrap();
        assert!(coordinator_action(&cli, None).is_err());

        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-results",
            "request-3",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
        ])
        .unwrap();
        let Some((request, _)) = coordinator_action(&cli, None).unwrap() else {
            panic!("results action should be present");
        };
        assert_eq!(
            request.version,
            crate::control::ipc::RESULTS_PROTOCOL_VERSION
        );
        assert!(matches!(
            request.message,
            crate::control::ipc::WireRequestMessage::Results {
                request_id,
                after_cursor: 0,
                byte_budget,
            } if request_id == "request-3"
                && byte_budget == crate::control::result_buffer::MAX_PAGE_BYTES as u32
        ));
    }

    #[test]
    fn coordinator_socket_resolution_prefers_cli_then_shared_settings() {
        let from_settings = PathBuf::from("/run/user/1000/shared-l3ms");
        let cli = Cli::try_parse_from(["l3ms", "--coordinator-status", "request-1"]).unwrap();
        assert!(coordinator_action(&cli, None).is_err());
        let Some((request, directory)) =
            coordinator_action(&cli, Some(from_settings.clone())).unwrap()
        else {
            panic!("configured coordinator action should be present");
        };
        assert_eq!(directory, from_settings);
        assert!(matches!(
            request.message,
            crate::control::ipc::WireRequestMessage::Status { request_id }
                if request_id == "request-1"
        ));

        let explicit = PathBuf::from("/run/user/1000/explicit-l3ms");
        let cli = Cli::try_parse_from([
            "l3ms",
            "--coordinator-cancel",
            "request-2",
            "--coordinator-socket-dir",
            "/run/user/1000/explicit-l3ms",
        ])
        .unwrap();
        let Some((request, directory)) =
            coordinator_action(&cli, Some(from_settings.clone())).unwrap()
        else {
            panic!("explicit coordinator action should be present");
        };
        assert_eq!(directory, explicit);
        assert!(matches!(
            request.message,
            crate::control::ipc::WireRequestMessage::Cancel { request_id }
                if request_id == "request-2"
        ));

        let cli = Cli::try_parse_from(["l3ms", "--request-results", "request-4"]).unwrap();
        let Some((request, directory)) =
            coordinator_action(&cli, Some(from_settings.clone())).unwrap()
        else {
            panic!("configured results action should be present");
        };
        assert_eq!(directory, from_settings);
        assert!(matches!(
            request.message,
            crate::control::ipc::WireRequestMessage::Results {
                request_id,
                after_cursor: 0,
                byte_budget,
            } if request_id == "request-4"
                && byte_budget == crate::control::result_buffer::MAX_PAGE_BYTES as u32
        ));

        let explicit = PathBuf::from("/run/user/1000/explicit-results");
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-results",
            "request-5",
            "--coordinator-socket-dir",
            "/run/user/1000/explicit-results",
            "--after-cursor",
            "9",
            "--result-page-bytes",
            "2048",
        ])
        .unwrap();
        let Some((request, directory)) = coordinator_action(&cli, Some(from_settings)).unwrap()
        else {
            panic!("explicit results action should be present");
        };
        assert_eq!(directory, explicit);
        assert!(matches!(
            request.message,
            crate::control::ipc::WireRequestMessage::Results {
                request_id,
                after_cursor: 9,
                byte_budget: 2048,
            } if request_id == "request-5"
        ));
    }

    #[test]
    fn request_output_requires_lane_and_rejects_modifiers() {
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-output",
            "caller:output-1",
            "--request-lane",
            "language",
        ])
        .unwrap();
        assert_eq!(
            coordinator_output_spec(&cli).unwrap(),
            Some(("caller:output-1", crate::control::catalog::Lane::Language))
        );
        assert!(Cli::try_parse_from([
            "l3ms",
            "--request-output",
            "caller:output-1",
            "--request-lane",
            "unknown",
        ])
        .is_err());

        let lane_without_output = Cli::try_parse_from(["l3ms", "--request-lane", "media"]).unwrap();
        assert!(coordinator_output_spec(&lane_without_output).is_err());
        let output_without_lane =
            Cli::try_parse_from(["l3ms", "--request-output", "caller:output-1"]).unwrap();
        assert!(coordinator_output_spec(&output_without_lane).is_err());

        for args in [
            vec![
                "l3ms",
                "--request-output",
                "caller:output-1",
                "--request-lane",
                "decisions",
                "--after-cursor",
                "4",
            ],
            vec![
                "l3ms",
                "--request-output",
                "caller:output-1",
                "--request-lane",
                "media",
                "--result-page-bytes",
                "4096",
            ],
            vec![
                "l3ms",
                "--request-output",
                "caller:output-1",
                "--request-lane",
                "language",
                "--extra",
                "",
            ],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(coordinator_output_spec(&cli).is_err());
        }

        assert!(Cli::try_parse_from([
            "l3ms",
            "--request-output",
            "caller:output-1",
            "--request-lane",
            "language",
            "--coordinator-status",
            "caller:status-1",
        ])
        .is_err());
        assert!(Cli::try_parse_from([
            "l3ms",
            "--request-output",
            "caller:output-1",
            "--request-lane",
            "language",
            "unexpected",
        ])
        .is_err());
    }

    #[test]
    fn request_output_socket_directory_uses_explicit_then_shared_config() {
        let shared = PathBuf::from("/private/shared-coordinator");
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-output",
            "caller:output-1",
            "--request-lane",
            "language",
        ])
        .unwrap();
        assert_eq!(
            coordinator_socket_directory(&cli, Some(shared.clone())).unwrap(),
            shared
        );
        let error = coordinator_socket_directory(&cli, None).unwrap_err();
        assert!(error
            .to_string()
            .contains("configured coordinator socket directory"));
        assert!(!error.to_string().contains("caller:output-1"));
        assert!(!error.to_string().contains("/private"));

        let explicit = PathBuf::from("/private/explicit-coordinator");
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-output",
            "caller:output-2",
            "--request-lane",
            "decisions",
            "--coordinator-socket-dir",
            explicit.to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(
            coordinator_socket_directory(&cli, Some(shared)).unwrap(),
            explicit
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_coordinator_launcher_rejects_modifiers_and_maps_fixed_errors() {
        let normal = Cli::try_parse_from(["l3ms"]).unwrap();
        assert!(coordinator_cpu_daemon_action(&normal).unwrap().is_none());

        let config = Path::new("/private/daemon-secret.json");
        let cli =
            Cli::try_parse_from(["l3ms", "--coordinator-cpu-daemon", config.to_str().unwrap()])
                .unwrap();
        assert_eq!(coordinator_cpu_daemon_action(&cli).unwrap(), Some(config));

        for args in [
            vec![
                "l3ms",
                "--coordinator-cpu-daemon",
                "/private/daemon.json",
                "--coordinator-socket-dir",
                "/private/socket",
            ],
            vec![
                "l3ms",
                "--coordinator-cpu-daemon",
                "/private/daemon.json",
                "--after-cursor",
                "1",
            ],
            vec![
                "l3ms",
                "--coordinator-cpu-daemon",
                "/private/daemon.json",
                "--result-page-bytes",
                "2048",
            ],
            vec![
                "l3ms",
                "--coordinator-cpu-daemon",
                "/private/daemon.json",
                "--extra",
                "",
            ],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(coordinator_cpu_daemon_action(&cli).is_err());
        }

        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("secret-config.json");
        let error = run_cpu_coordinator(&missing).unwrap_err();
        assert!(error
            .to_string()
            .contains("CPU coordinator configuration is unavailable"));
        assert!(!error.to_string().contains(missing.to_str().unwrap()));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_coordinator_report_contains_only_compact_counts_and_stop_reason() {
        let report = crate::control::daemon::RunReport {
            completed_exchanges: 7,
            failed_connections: 2,
            stopped: crate::control::daemon::StopReason::Deadline,
        };
        assert_eq!(
            cpu_coordinator_report_json(report),
            r#"{"completed_exchanges":7,"failed_connections":2,"stopped":"deadline"}"#
        );
    }

    #[test]
    fn results_budget_is_bounded_and_rejected_before_socket_connect() {
        let directory = tempfile::tempdir().unwrap();
        for budget in [
            crate::control::result_buffer::MIN_PAGE_BYTES as u32,
            crate::control::result_buffer::MAX_PAGE_BYTES as u32,
        ] {
            let cli = Cli::try_parse_from([
                "l3ms",
                "--request-results",
                "request-1",
                "--coordinator-socket-dir",
                directory.path().to_str().unwrap(),
                "--result-page-bytes",
                &budget.to_string(),
            ])
            .unwrap();
            assert!(coordinator_action(&cli, None).is_ok());
        }

        for budget in [
            crate::control::result_buffer::MIN_PAGE_BYTES as u32 - 1,
            crate::control::result_buffer::MAX_PAGE_BYTES as u32 + 1,
        ] {
            let cli = Cli::try_parse_from([
                "l3ms",
                "--request-results",
                "request-1",
                "--coordinator-socket-dir",
                directory.path().to_str().unwrap(),
                "--result-page-bytes",
                &budget.to_string(),
            ])
            .unwrap();
            let error = dispatch(cli).unwrap_err();
            assert!(error.to_string().contains("between 1024 and 131072"));
            assert!(!directory.path().join("coordinator.sock").exists());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn coordinator_cli_delegates_to_the_shared_client() {
        use crate::control::ipc::{
            ClientRequest, RequestStatus, ResponseEnvelope, ResponseMessage,
        };
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket_path = directory.path().join("coordinator.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            assert!(matches!(
                crate::control::ipc::read_request(&mut stream).unwrap(),
                ClientRequest::Status { request_id } if request_id == "request-1"
            ));
            crate::control::ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v1(ResponseMessage::Status {
                    request_id: "request-1".into(),
                    state: RequestStatus::Running,
                }),
            )
            .unwrap();
        });

        let cli = Cli::try_parse_from([
            "l3ms",
            "--coordinator-status",
            "request-1",
            "--coordinator-socket-dir",
            directory.path().to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(dispatch(cli).unwrap(), 0);
        server.join().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn coordinator_results_error_is_fixed_and_exits_nonzero() {
        use crate::control::ipc::{
            IncomingRequest, ResponseEnvelope, ResponseErrorCode, ResponseMessage,
        };
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket_path = directory.path().join("coordinator.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            assert!(matches!(
                crate::control::ipc::read_incoming_request(&mut stream).unwrap(),
                IncomingRequest::Results(ref query)
                    if query.request_id == "request-1"
                        && query.after_cursor == 0
                        && query.byte_budget
                            == crate::control::result_buffer::MAX_PAGE_BYTES as u32
            ));
            crate::control::ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v2_results(ResponseMessage::Error {
                    request_id: Some("request-1".into()),
                    code: ResponseErrorCode::Unavailable,
                }),
            )
            .unwrap();
        });

        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-results",
            "request-1",
            "--coordinator-socket-dir",
            directory.path().to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(dispatch(cli).unwrap(), 1);
        server.join().unwrap();
    }

    #[test]
    fn coordinator_response_json_is_compact_and_bounded() {
        use crate::control::ipc::{RequestStatus, ResponseMessage};

        let json = coordinator_response_json(&ResponseMessage::Status {
            request_id: "request-1".into(),
            state: RequestStatus::Queued,
        })
        .unwrap();
        assert!(json.len() <= MAX_COORDINATOR_RESPONSE_BYTES);
        assert!(!json.contains('\n'));
        assert!(json.contains("\"type\":\"status\""));
        assert!(coordinator_response_json(&ResponseMessage::Status {
            request_id: "x".repeat(MAX_COORDINATOR_RESPONSE_BYTES + 1),
            state: RequestStatus::Queued,
        })
        .is_err());
    }

    #[test]
    fn result_response_json_is_compact_escaped_and_has_its_own_bound() {
        use crate::control::catalog::Lane;
        use crate::control::ipc::{RequestStatus, ResponseMessage};
        use crate::control::result_buffer::{
            ResultBuffer, ResultEvent, ResultTerminal, MAX_PAGE_BYTES, MIN_PAGE_BYTES,
        };

        let mut buffer = ResultBuffer::new();
        buffer.begin("result:cli-json", Lane::Language).unwrap();
        buffer
            .append(
                "result:cli-json",
                ResultEvent::ChatContent("line\u{1}break".into()),
            )
            .unwrap();
        for _ in 0..20 {
            buffer
                .append("result:cli-json", ResultEvent::ChatContent("x".repeat(64)))
                .unwrap();
        }
        buffer
            .terminal("result:cli-json", ResultTerminal::Succeeded)
            .unwrap();
        let page = buffer.page("result:cli-json", 0, MAX_PAGE_BYTES).unwrap();
        let json = coordinator_response_json(&ResponseMessage::Results {
            request_id: "result:cli-json".into(),
            after_cursor: 0,
            byte_budget: MAX_PAGE_BYTES as u32,
            state: RequestStatus::Succeeded,
            page: page.clone(),
        })
        .unwrap();
        assert!(!json.contains('\n'));
        assert!(json.contains("\\u0001"));
        assert!(!json.chars().any(char::is_control));
        assert!(json.len() <= MAX_PAGE_BYTES + MAX_COORDINATOR_RESULTS_ENVELOPE_BYTES);

        let too_small_json = coordinator_response_json(&ResponseMessage::Results {
            request_id: "result:cli-json".into(),
            after_cursor: 0,
            byte_budget: MIN_PAGE_BYTES as u32,
            state: RequestStatus::Succeeded,
            page,
        });
        assert!(too_small_json.is_err());
    }

    #[test]
    fn native_decision_export_keeps_optional_metadata_separate_and_exact() {
        use crate::control::catalog::Lane;
        use crate::control::result_buffer::{ResultBuffer, ResultEvent, MAX_PAGE_BYTES};
        use crate::control::result_consumer::ResultConsumer;
        let mut buffer = ResultBuffer::new();
        buffer.begin("decision-export", Lane::Decisions).unwrap();
        buffer
            .append(
                "decision-export",
                ResultEvent::DecisionResult {
                    answers: serde_json::json!({"choice": "yes"})
                        .as_object()
                        .unwrap()
                        .clone(),
                    native_latency: Some(Duration::from_nanos(12_500_001)),
                    native_usage: Some(
                        serde_json::json!({"provider_units": 3, "detail": "private\nusage"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                },
            )
            .unwrap();
        let mut consumer = ResultConsumer::new("decision-export", Lane::Decisions).unwrap();
        consumer
            .consume(&buffer.page("decision-export", 0, MAX_PAGE_BYTES).unwrap())
            .unwrap();
        let json = coordinator_native_output_json("decision-export", &consumer).unwrap();
        assert!(!json.chars().any(char::is_control));
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value["native_decision_metadata"]["native_latency_ns"],
            12_500_001
        );
        assert_eq!(
            value["native_decision_metadata"]["native_usage"]["provider_units"],
            3
        );
        assert_eq!(
            value["output"]["answers"],
            serde_json::json!([{"choice": "yes"}])
        );
        assert!(value["output"].get("native_latency_ns").is_none());
    }

    #[test]
    fn native_output_json_is_compact_escaped_and_preserves_completion_metadata() {
        use crate::control::catalog::Lane;
        use crate::control::result_buffer::{
            ResultBuffer, ResultEvent, ResultTerminal, MAX_PAGE_BYTES,
        };
        use crate::control::result_consumer::{NativeOutput, ResultConsumer};

        let request_id = "caller:native-output";
        let mut buffer = ResultBuffer::new();
        buffer.begin(request_id, Lane::Language).unwrap();
        buffer
            .append(
                request_id,
                ResultEvent::ChatContent("visible\ntext\u{1}".into()),
            )
            .unwrap();
        buffer
            .append(
                request_id,
                ResultEvent::ChatReasoning("kept separate".into()),
            )
            .unwrap();
        buffer
            .terminal(request_id, ResultTerminal::Succeeded)
            .unwrap();

        let page = buffer.page(request_id, 0, MAX_PAGE_BYTES).unwrap();
        let mut consumer = ResultConsumer::new(request_id, Lane::Language).unwrap();
        consumer.consume(&page).unwrap();
        let json = coordinator_native_output_json(request_id, &consumer).unwrap();
        assert!(!json.contains('\n'));
        assert!(json.contains("visible\\ntext\\u0001"));
        assert!(json.contains("\"terminal\":{\"status\":\"succeeded\"}"));
        assert!(json.contains("\"complete\":true"));
        assert!(json.contains("\"reasoning\":\"kept separate\""));
        assert!(json.len() <= MAX_COORDINATOR_NATIVE_OUTPUT_BYTES);
        assert!(!json.chars().any(char::is_control));
        assert!(!json.contains("native_decision_metadata"));
        assert!(matches!(
            consumer.output(),
            NativeOutput::Language { content, reasoning }
                if content == "visible\ntext\u{1}" && reasoning == "kept separate"
        ));

        let mut writer = BoundedJsonWriter::new(3);
        assert!(writer.write_all(b"four").is_err());
        assert!(writer.exceeded);
        assert_eq!(writer.bytes, b"");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn request_output_cli_delegates_one_v2_read_and_prints_reassembled_native_output() {
        use crate::control::catalog::Lane;
        use crate::control::ipc::{
            IncomingRequest, RequestStatus, ResponseEnvelope, ResponseMessage,
        };
        use crate::control::result_buffer::{
            ResultBuffer, ResultEvent, ResultTerminal, MAX_PAGE_BYTES,
        };
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket_path = directory.path().join("coordinator.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let IncomingRequest::Results(query) =
                crate::control::ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("native output must only request results");
            };
            assert_eq!(query.request_id, "caller:native-output-cli");
            assert_eq!(query.after_cursor, 0);
            assert_eq!(query.byte_budget, MAX_PAGE_BYTES as u32);

            let mut buffer = ResultBuffer::new();
            buffer.begin(&query.request_id, Lane::Language).unwrap();
            buffer
                .append(
                    &query.request_id,
                    ResultEvent::ChatContent("cli result\nvisible".into()),
                )
                .unwrap();
            buffer
                .terminal(&query.request_id, ResultTerminal::Succeeded)
                .unwrap();
            let page = buffer
                .page(
                    &query.request_id,
                    query.after_cursor,
                    query.byte_budget as usize,
                )
                .unwrap();
            crate::control::ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v2_results(ResponseMessage::Results {
                    request_id: query.request_id,
                    after_cursor: query.after_cursor,
                    byte_budget: query.byte_budget,
                    state: RequestStatus::Succeeded,
                    page,
                }),
            )
            .unwrap();
        });

        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-output",
            "caller:native-output-cli",
            "--request-lane",
            "language",
            "--coordinator-socket-dir",
            directory.path().to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(dispatch(cli).unwrap(), 0);
        server.join().unwrap();
    }

    #[test]
    fn offline_stack_list_formats_manifest_only_profiles() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let catalog = Catalog::load(&root).unwrap();
        let key = "media:breeze-tts-2".parse().unwrap();
        let profile = catalog.inspect(&key).unwrap();
        let formatted = format_stack_profile(profile);
        assert!(formatted.contains("media:breeze-tts-2"));
        assert!(formatted.contains("tts"));
        assert!(formatted.contains("manifest-only"));
        assert!(formatted.contains("media-runtimes.json"));
    }

    #[test]
    fn parses_shell_style_extra_arguments() {
        assert_eq!(
            parse_extra_args(r#"--ctx-size 32768 --prompt "hello world""#).unwrap(),
            ["--ctx-size", "32768", "--prompt", "hello world"]
        );
        assert!(parse_extra_args("'unterminated").is_err());
        assert!(parse_extra_args("  ").unwrap().is_empty());
    }

    #[test]
    fn filters_models_case_insensitively_across_metadata() {
        let models = vec![
            model("qwen3-coder", "Coder", "large coding model"),
            model("gemma", "Vision Model", "multimodal"),
        ];

        assert_eq!(filter_models(models.clone(), "QWEN")[0].id, "qwen3-coder");
        assert_eq!(filter_models(models.clone(), "vision")[0].id, "gemma");
        assert_eq!(filter_models(models.clone(), "MULTI")[0].id, "gemma");
        assert_eq!(filter_models(models.clone(), "").len(), 2);
        assert!(filter_models(models, "missing").is_empty());
    }

    #[test]
    fn filters_scripts_by_relative_path() {
        let root = Path::new("/repo");
        let scripts = vec![
            root.join("bench-models/bench-llama-cpp-qwen.sh"),
            root.join("bench-models/bench-llama-cpp-gemma.sh"),
        ];
        assert_eq!(
            filter_scripts(scripts.clone(), root, "QWEN"),
            vec![scripts[0].clone()]
        );
        assert_eq!(
            filter_scripts(scripts.clone(), root, "bench-models").len(),
            2
        );
        assert_eq!(filter_scripts(scripts, root, "").len(), 2);
    }

    #[test]
    fn creates_readable_benchmark_names() {
        assert_eq!(
            pretty_name(Path::new("bench-models/bench-llama-cpp-qwen3.sh")),
            "qwen3"
        );
        assert_eq!(
            pretty_name(Path::new("bench-models/bench-ik-llama-cpp-gpt-oss.sh")),
            "gpt-oss"
        );
        assert_eq!(
            pretty_name(Path::new("bench-models/bench-custom.sh")),
            "custom"
        );
    }

    #[test]
    fn builds_bash_command_without_reparsing_arguments() {
        let script = Path::new("/repo/bench-models/bench model.sh");
        let extra = vec!["--prompt".into(), "hello world".into()];
        let command = command_for_script(script, &extra);

        assert_eq!(command.program, OsString::from("bash"));
        assert_eq!(
            command.args,
            vec![
                OsString::from("/repo/bench-models/bench model.sh"),
                OsString::from("--prompt"),
                OsString::from("hello world")
            ]
        );
        assert_eq!(
            format_command(&command),
            "bash '/repo/bench-models/bench model.sh' --prompt 'hello world'"
        );
    }

    #[test]
    fn finds_repository_above_target_binary() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let executable = root.join("target/debug/deps/l3ms-test");
        assert_eq!(find_repository_root(&executable).as_deref(), Some(root));
    }

    #[test]
    fn artifact_chunk_parsing_checks_required_fields_bounds_and_redaction() {
        let token = "a".repeat(64);
        let cli = Cli::try_parse_from([
            "l3ms",
            "--artifact-chunk",
            "caller:artifact-cli",
            "--artifact-kind",
            "backend",
            "--artifact-path",
            "jobs/job-1/frame.png",
            "--artifact-offset",
            "0",
            "--artifact-byte-budget",
            "65536",
            "--artifact-version",
            token.as_str(),
        ])
        .unwrap();
        let spec = coordinator_artifact_chunk_spec(&cli).unwrap().unwrap();
        assert_eq!(spec.request_id, "caller:artifact-cli");
        assert_eq!(
            spec.artifact.kind,
            crate::control::result_buffer::ArtifactKind::Backend
        );
        assert_eq!(spec.artifact.relative_path, "jobs/job-1/frame.png");
        assert_eq!(spec.offset, 0);
        assert_eq!(spec.byte_budget, 65_536);
        assert_eq!(spec.artifact_version.as_ref().unwrap().as_str(), token);

        for budget in ["0", "65537"] {
            let cli = Cli::try_parse_from([
                "l3ms",
                "--artifact-chunk",
                "caller:artifact-cli",
                "--artifact-kind",
                "local_job",
                "--artifact-path",
                "job/output.wav",
                "--artifact-offset",
                "0",
                "--artifact-byte-budget",
                budget,
            ])
            .unwrap();
            assert!(coordinator_artifact_chunk_spec(&cli).is_err());
        }

        let missing_version = Cli::try_parse_from([
            "l3ms",
            "--artifact-chunk",
            "caller:artifact-cli",
            "--artifact-kind",
            "backend",
            "--artifact-path",
            "jobs/job-1/frame.png",
            "--artifact-offset",
            "1",
            "--artifact-byte-budget",
            "8",
        ])
        .unwrap();
        assert!(coordinator_artifact_chunk_spec(&missing_version).is_err());

        assert!(Cli::try_parse_from([
            "l3ms",
            "--artifact-chunk",
            "caller:artifact-cli",
            "--artifact-kind",
            "local-job",
            "--artifact-path",
            "job/output.wav",
            "--artifact-offset",
            "0",
            "--artifact-byte-budget",
            "8",
        ])
        .is_err());

        let secret_path = "../private/native/output.png";
        let invalid_path = Cli::try_parse_from([
            "l3ms",
            "--artifact-chunk",
            "caller:artifact-cli",
            "--artifact-kind",
            "backend",
            "--artifact-path",
            secret_path,
            "--artifact-offset",
            "0",
            "--artifact-byte-budget",
            "8",
        ])
        .unwrap();
        let error = coordinator_artifact_chunk_spec(&invalid_path)
            .unwrap_err()
            .to_string();
        assert!(!error.contains(secret_path));

        let secret_token = format!("{}!", "b".repeat(63));
        let invalid_version = Cli::try_parse_from([
            "l3ms",
            "--artifact-chunk",
            "caller:artifact-cli",
            "--artifact-kind",
            "backend",
            "--artifact-path",
            "jobs/job-1/frame.png",
            "--artifact-offset",
            "0",
            "--artifact-byte-budget",
            "8",
            "--artifact-version",
            secret_token.as_str(),
        ])
        .unwrap();
        let error = coordinator_artifact_chunk_spec(&invalid_version)
            .unwrap_err()
            .to_string();
        assert!(!error.contains(&secret_token));
    }

    #[test]
    fn artifact_chunk_is_exclusive_and_does_not_change_results_cli_flow() {
        let args = [
            "l3ms",
            "--artifact-chunk",
            "caller:artifact-cli",
            "--artifact-kind",
            "backend",
            "--artifact-path",
            "jobs/job-1/frame.png",
            "--artifact-offset",
            "0",
            "--artifact-byte-budget",
            "8",
        ];
        let mut mixed_results = args.to_vec();
        mixed_results.extend(["--request-results", "caller:other"]);
        assert!(Cli::try_parse_from(mixed_results).is_err());
        let mut mixed_run = args.to_vec();
        mixed_run.push("--run");
        assert!(Cli::try_parse_from(mixed_run).is_err());

        let results_cli = Cli::try_parse_from([
            "l3ms",
            "--request-results",
            "caller:results-cli",
            "--after-cursor",
            "9",
            "--result-page-bytes",
            "2048",
        ])
        .unwrap();
        assert!(coordinator_artifact_chunk_spec(&results_cli)
            .unwrap()
            .is_none());
        let Some((request, _)) =
            coordinator_action(&results_cli, Some(PathBuf::from("/private/coordinator"))).unwrap()
        else {
            panic!("legacy result page action should remain available");
        };
        assert!(matches!(
            request.message,
            crate::control::ipc::WireRequestMessage::Results {
                request_id,
                after_cursor: 9,
                byte_budget: 2048,
            } if request_id == "caller:results-cli"
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn artifact_chunk_json_is_bounded_and_keeps_sensitive_debug_data_redacted() {
        let token = "c".repeat(64);
        let chunk = crate::control::client::ArtifactChunk {
            request_id: "caller:artifact-cli".to_owned(),
            artifact: crate::control::result_buffer::MediaArtifact::new(
                crate::control::result_buffer::ArtifactKind::Backend,
                "jobs/job-1/private-frame.png",
            )
            .unwrap(),
            availability: crate::control::result_buffer::Availability::Available,
            offset: 0,
            total_bytes: 3,
            next_offset: 3,
            end_of_file: true,
            artifact_version: Some(
                crate::control::ipc::ArtifactVersion::new(token.as_str()).unwrap(),
            ),
            bytes: vec![0, 1, 255],
        };
        let debug = format!("{chunk:?}");
        assert!(!debug.contains("jobs/job-1/private-frame.png"));
        assert!(!debug.contains(&token));
        assert!(!debug.contains("0001ff"));

        let json = coordinator_artifact_chunk_json(&chunk, 3).unwrap();
        assert!(json.len() <= MAX_COORDINATOR_ARTIFACT_CHUNK_JSON_BYTES);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["availability"], "available");
        assert_eq!(parsed["total_bytes"], 3);
        assert_eq!(parsed["next_offset"], 3);
        assert_eq!(parsed["end_of_file"], true);
        assert_eq!(parsed["artifact_version"], token);
        assert_eq!(parsed["bytes_hex"], "0001ff");
        assert_eq!(
            parsed["artifact"]["relative_path"],
            "jobs/job-1/private-frame.png"
        );

        let large_chunk = crate::control::client::ArtifactChunk {
            bytes: vec![0; crate::control::ipc::MAX_ARTIFACT_CHUNK_BYTES],
            total_bytes: crate::control::ipc::MAX_ARTIFACT_CHUNK_BYTES as u64,
            next_offset: crate::control::ipc::MAX_ARTIFACT_CHUNK_BYTES as u64,
            ..chunk
        };
        let large_json = coordinator_artifact_chunk_json(
            &large_chunk,
            crate::control::ipc::MAX_ARTIFACT_CHUNK_BYTES as u32,
        )
        .unwrap();
        assert!(large_json.len() <= MAX_COORDINATOR_ARTIFACT_CHUNK_JSON_BYTES);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&large_json).unwrap()["bytes_hex"]
                .as_str()
                .unwrap()
                .len(),
            crate::control::ipc::MAX_ARTIFACT_CHUNK_BYTES * 2
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn artifact_chunk_cli_makes_one_exact_read_only_v3_exchange() {
        use crate::control::ipc::{
            ArtifactVersion, IncomingRequest, ResponseEnvelope, ResponseMessage,
        };
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket_path = directory.path().join("coordinator.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let version = "d".repeat(64);

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let IncomingRequest::Artifact(query) =
                crate::control::ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("artifact chunk should use one read-only artifact operation");
            };
            assert_eq!(query.request_id, "caller:artifact-cli");
            assert_eq!(
                query.artifact.kind,
                crate::control::result_buffer::ArtifactKind::LocalJob
            );
            assert_eq!(query.artifact.relative_path, "job-41/output.png");
            assert_eq!(query.offset, 0);
            assert_eq!(query.byte_budget, 3);
            assert!(query.artifact_version.is_none());
            crate::control::ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v3_artifact(ResponseMessage::Artifact {
                    request_id: query.request_id,
                    artifact: query.artifact,
                    availability: crate::control::result_buffer::Availability::Available,
                    offset: query.offset,
                    total_bytes: 3,
                    next_offset: 3,
                    end_of_file: true,
                    artifact_version: Some(ArtifactVersion::new(&version).unwrap()),
                    bytes: vec![1, 2, 3],
                }),
            )
            .unwrap();
        });

        let cli = Cli::try_parse_from([
            "l3ms",
            "--artifact-chunk",
            "caller:artifact-cli",
            "--artifact-kind",
            "local_job",
            "--artifact-path",
            "job-41/output.png",
            "--artifact-offset",
            "0",
            "--artifact-byte-budget",
            "3",
            "--coordinator-socket-dir",
            directory.path().to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(dispatch(cli).unwrap(), 0);
        server.join().unwrap();
    }
}
