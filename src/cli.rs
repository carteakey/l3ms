//! Command-line launcher for the Rust L3MS application.

use std::env;
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
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
#[cfg(target_os = "linux")]
const MAX_COORDINATOR_ARTIFACT_CHUNK_JSON_BYTES: usize = 160 * 1024;

#[derive(Debug, Parser)]
#[command(
    name = "l3ms",
    version,
    about = "L3MS launcher (TUI + interactive model and bench CLI)",
    group(
        ArgGroup::new("action")
            .args(["run", "bench", "list", "quickstart", "settings", "export_profile", "import_profile", "compare_results", "coordinator_status", "coordinator_cancel", "request_results", "request_submit", "request_output", "artifact_chunk", "coordinator_cpu_daemon", "coordinator_managed_session", "coordinator_http_gateway"])
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

    /// Submit one strict v1 request envelope from a bounded JSON file.
    #[arg(long, value_name = "PATH")]
    request_submit: Option<PathBuf>,

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

    /// Run an authenticated loopback HTTP gateway to an existing coordinator.
    #[arg(long, value_name = "CONFIG")]
    coordinator_http_gateway: Option<PathBuf>,

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
    if let Some(config_path) = http_gateway_action(&cli)? {
        return run_http_gateway_action(config_path);
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
        || cli.request_submit.is_some()
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

fn http_gateway_action(cli: &Cli) -> Result<Option<&Path>> {
    let Some(path) = cli.coordinator_http_gateway.as_deref() else {
        return Ok(None);
    };
    if cli.request_lane.is_some()
        || cli.coordinator_socket_dir.is_some()
        || cli.after_cursor.is_some()
        || cli.result_page_bytes.is_some()
        || cli.artifact_chunk.is_some()
        || cli.artifact_kind.is_some()
        || cli.artifact_path.is_some()
        || cli.artifact_offset.is_some()
        || cli.artifact_byte_budget.is_some()
        || cli.artifact_version.is_some()
        || cli.extra.is_some()
    {
        bail!("HTTP gateway accepts only its explicit configuration file");
    }
    Ok(Some(path))
}

#[cfg(target_os = "linux")]
fn run_http_gateway_action(path: &Path) -> Result<u8> {
    let report = crate::control::http_launch::run(path).map_err(anyhow::Error::new)?;
    let stopped = match report.stop {
        crate::control::http_gateway::SessionStop::Cancelled => "cancelled",
        crate::control::http_gateway::SessionStop::Deadline => "deadline",
        crate::control::http_gateway::SessionStop::AttemptLimit => "attempt_limit",
        crate::control::http_gateway::SessionStop::ListenerUnavailable => "listener_unavailable",
    };
    println!(
        "{}",
        serde_json::json!({
            "accepted_attempts": report.accepted_attempts,
            "completed_http_connections": report.completed_http_connections,
            "denied_non_loopback_peers": report.denied_non_loopback_peers,
            "denied_peer_limit": report.denied_peer_limit,
            "stopped": stopped,
        })
    );
    Ok(u8::from(
        report.stop == crate::control::http_gateway::SessionStop::ListenerUnavailable,
    ))
}

#[cfg(not(target_os = "linux"))]
fn run_http_gateway_action(_: &Path) -> Result<u8> {
    bail!("HTTP gateway is supported only on Linux");
}

fn coordinator_cpu_daemon_action(cli: &Cli) -> Result<Option<&Path>> {
    let Some(config_path) = cli.coordinator_cpu_daemon.as_deref() else {
        return Ok(None);
    };
    if cli.coordinator_socket_dir.is_some()
        || cli.coordinator_status.is_some()
        || cli.coordinator_cancel.is_some()
        || cli.request_results.is_some()
        || cli.request_submit.is_some()
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
        || cli.request_submit.is_some()
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
                || cli.request_submit.is_some()
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
        bail!("--artifact-chunk requires --artifact-kind, --artifact-path, --artifact-offset, and --artifact-byte-budget");
    }
    if cli.coordinator_status.is_some()
        || cli.coordinator_cancel.is_some()
        || cli.request_results.is_some()
        || cli.request_submit.is_some()
        || cli.request_output.is_some()
        || cli.request_lane.is_some()
        || cli.coordinator_cpu_daemon.is_some()
        || cli.coordinator_managed_session.is_some()
        || cli.coordinator_http_gateway.is_some()
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
        cli.request_submit.is_some(),
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
        cli.request_submit.as_deref(),
    ) {
        (Some(request_id), None, None, None) => Some(ParsedCoordinatorAction::Status(request_id)),
        (None, Some(request_id), None, None) => Some(ParsedCoordinatorAction::Cancel(request_id)),
        (None, None, Some(request_id), None) => Some(ParsedCoordinatorAction::Results(request_id)),
        (None, None, None, Some(path)) => Some(ParsedCoordinatorAction::Submit(path)),
        (None, None, None, None) => None,
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
        ParsedCoordinatorAction::Submit(path) => read_submit_envelope(path)?,
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
    Submit(&'a Path),
}

fn read_submit_envelope(path: &Path) -> Result<crate::control::ipc::RequestEnvelope> {
    let root = repository_root().map_err(|_| anyhow::anyhow!("submit request file is invalid"))?;
    let data_root = crate::state_store::data_root()
        .map_err(|_| anyhow::anyhow!("submit request file is invalid"))?;
    let resolved = load_current_stack_settings_in(&root, &data_root)
        .map_err(|_| anyhow::anyhow!("submit request file is invalid"))?;
    let catalog = Catalog::load_from_roots(
        &root,
        &resolved.media_toolkit_root,
        &resolved.decisions_toolkit_root,
    )
    .map_err(|_| anyhow::anyhow!("submit request file is invalid"))?;
    crate::control::submission_file::read(path, &catalog)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
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
    fn http_gateway_is_exclusive_and_rejects_orphan_options_before_launch() {
        let cli =
            Cli::try_parse_from(["l3ms", "--coordinator-http-gateway", "private.json"]).unwrap();
        assert_eq!(
            http_gateway_action(&cli).unwrap(),
            Some(Path::new("private.json"))
        );
        assert!(Cli::try_parse_from([
            "l3ms",
            "--coordinator-http-gateway",
            "private.json",
            "--quickstart"
        ])
        .is_err());
        for options in [
            vec!["--extra", ""],
            vec!["--coordinator-socket-dir", "/private"],
            vec!["--request-lane", "decisions"],
            vec!["--after-cursor", "0"],
            vec!["--result-page-bytes", "4096"],
        ] {
            let mut arguments = vec!["l3ms", "--coordinator-http-gateway", "private.json"];
            arguments.extend(options);
            let cli = Cli::try_parse_from(arguments).unwrap();
            assert_eq!(
                http_gateway_action(&cli).unwrap_err().to_string(),
                "HTTP gateway accepts only its explicit configuration file"
            );
        }
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

        let cli = Cli::try_parse_from(["l3ms", "--request-submit", "request.json"]).unwrap();
        assert_eq!(
            cli.request_submit.as_deref(),
            Some(Path::new("request.json"))
        );
        assert!(Cli::try_parse_from([
            "l3ms",
            "--request-submit",
            "request.json",
            "--request-results",
            "request-1",
        ])
        .is_err());
        let cli = Cli::try_parse_from(["l3ms", "--coordinator-cpu-daemon", "daemon.json"]).unwrap();
        assert_eq!(
            cli.coordinator_cpu_daemon.as_deref(),
            Some(Path::new("daemon.json"))
        );
        assert!(Cli::try_parse_from([
            "l3ms",
            "--coordinator-cpu-daemon",
            "daemon.json",
            "--request-submit",
            "request.json",
        ])
        .is_err());
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
        assert!(Cli::try_parse_from([
            "l3ms",
            "--request-submit",
            "request.json",
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
        let cli = Cli::try_parse_from(["l3ms", "--request-submit", "request.json"]).unwrap();
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

    fn valid_submit_envelope() -> crate::control::ipc::RequestEnvelope {
        use crate::control::ipc::{
            RequestEnvelope, WireChatRole, WireChatTurn, WireInferenceRequest, WireRequestMessage,
        };

        RequestEnvelope::v1(WireRequestMessage::Submit {
            request_id: "caller:cli-submit-1".into(),
            timeout_ms: 15_000,
            request: WireInferenceRequest::Language {
                profile_id: "gemma-4-26b-qat-mtp".into(),
                turns: vec![WireChatTurn {
                    role: WireChatRole::User,
                    content: "private submit prompt".into(),
                }],
                max_tokens: 32,
            },
        })
    }

    fn write_submit_envelope(path: &Path, envelope: &crate::control::ipc::RequestEnvelope) {
        let bytes = serde_json::to_vec(envelope).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn submit_selection_resolves_explicit_or_shared_socket_and_preserves_envelope() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("submit.json");
        write_submit_envelope(&file, &valid_submit_envelope());
        let shared_socket = PathBuf::from("/run/user/1000/shared-l3ms");
        let cli =
            Cli::try_parse_from(["l3ms", "--request-submit", file.to_str().unwrap()]).unwrap();
        let Some((request, directory)) =
            coordinator_action(&cli, Some(shared_socket.clone())).unwrap()
        else {
            panic!("submit action should be present");
        };
        assert_eq!(directory, shared_socket);
        assert_eq!(request.version, crate::control::ipc::PROTOCOL_VERSION);
        let crate::control::ipc::WireRequestMessage::Submit {
            request_id,
            timeout_ms,
            request:
                crate::control::ipc::WireInferenceRequest::Language {
                    profile_id,
                    turns,
                    max_tokens,
                },
        } = request.message
        else {
            panic!("submit envelope should be preserved");
        };
        assert!(request_id == "caller:cli-submit-1");
        assert!(timeout_ms == 15_000);
        assert!(profile_id == "gemma-4-26b-qat-mtp");
        assert!(turns.len() == 1 && turns[0].content == "private submit prompt");
        assert!(max_tokens == 32);

        for args in [
            vec![
                "l3ms",
                "--request-submit",
                file.to_str().unwrap(),
                "--after-cursor",
                "1",
                "--coordinator-socket-dir",
                "/run/user/1000/explicit-l3ms",
            ],
            vec![
                "l3ms",
                "--request-submit",
                file.to_str().unwrap(),
                "--result-page-bytes",
                "2048",
                "--coordinator-socket-dir",
                "/run/user/1000/explicit-l3ms",
            ],
            vec![
                "l3ms",
                "--request-submit",
                file.to_str().unwrap(),
                "--extra",
                "",
                "--coordinator-socket-dir",
                "/run/user/1000/explicit-l3ms",
            ],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(coordinator_action(&cli, None).is_err());
        }

        let explicit_socket = PathBuf::from("/run/user/1000/explicit-l3ms");
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-submit",
            file.to_str().unwrap(),
            "--coordinator-socket-dir",
            explicit_socket.to_str().unwrap(),
        ])
        .unwrap();
        let Some((_, directory)) =
            coordinator_action(&cli, Some(PathBuf::from("/run/user/1000/shared-l3ms"))).unwrap()
        else {
            panic!("explicit submit action should be present");
        };
        assert_eq!(directory, explicit_socket);
    }

    #[test]
    fn submit_rejects_invalid_version_operation_and_native_payload_before_connect() {
        use crate::control::ipc::{RequestEnvelope, WireInferenceRequest, WireRequestMessage};

        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("request.json");
        let invalid_envelopes = [
            RequestEnvelope {
                version: crate::control::ipc::RESULTS_PROTOCOL_VERSION,
                message: valid_submit_envelope().message,
            },
            RequestEnvelope::v1(WireRequestMessage::Status {
                request_id: "caller:submit-status".into(),
            }),
            RequestEnvelope::v1(WireRequestMessage::Submit {
                request_id: "caller:invalid-native".into(),
                timeout_ms: 5_000,
                request: WireInferenceRequest::Language {
                    profile_id: "gemma-4-26b-qat-mtp".into(),
                    turns: Vec::new(),
                    max_tokens: 0,
                },
            }),
        ];

        for envelope in invalid_envelopes {
            write_submit_envelope(&file, &envelope);
            let cli = Cli::try_parse_from([
                "l3ms",
                "--request-submit",
                file.to_str().unwrap(),
                "--coordinator-socket-dir",
                directory.path().to_str().unwrap(),
            ])
            .unwrap();
            let error = dispatch(cli).unwrap_err();
            assert!(error.to_string().contains("submit request file"));
            assert!(!error.to_string().contains("private submit prompt"));
            assert!(!directory.path().join("coordinator.sock").exists());
        }

        let invalid_role_shape = serde_json::json!({
            "version": 1,
            "message": {
                "type": "submit",
                "request_id": "caller:strict-json",
                "timeout_ms": 5000,
                "request": {
                    "lane": "language",
                    "profile_id": "gemma-4-26b-qat-mtp",
                    "turns": [{ "role": "user", "content": "secret" }],
                    "max_tokens": 16,
                    "unknown": true
                }
            }
        });
        std::fs::write(&file, serde_json::to_vec(&invalid_role_shape).unwrap()).unwrap();
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-submit",
            file.to_str().unwrap(),
            "--coordinator-socket-dir",
            directory.path().to_str().unwrap(),
        ])
        .unwrap();
        let error = dispatch(cli).unwrap_err();
        assert!(error.to_string().contains("submit request file is invalid"));
        assert!(!error.to_string().contains("secret"));
        assert!(!directory.path().join("coordinator.sock").exists());
    }

    #[test]
    fn submit_file_must_be_regular_nonsymlink_and_bounded() {
        use crate::control::ipc::MAX_FRAME_BYTES;

        let directory = tempfile::tempdir().unwrap();
        let regular = directory.path().join("regular.json");
        write_submit_envelope(&regular, &valid_submit_envelope());
        let oversized = directory.path().join("oversized.json");
        std::fs::write(&oversized, vec![b'x'; MAX_FRAME_BYTES + 1]).unwrap();

        let oversized_error = read_submit_envelope(&oversized).err().unwrap();
        assert!(oversized_error
            .to_string()
            .contains("submit request file exceeds the size limit"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let symbolic = directory.path().join("symbolic.json");
            symlink(&regular, &symbolic).unwrap();
            assert_eq!(
                read_submit_envelope(&symbolic).err().unwrap().to_string(),
                "submit request file is unavailable or unsafe"
            );

            let fifo = directory.path().join("request.fifo");
            let fifo_c = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
            // SAFETY: fifo_c is a valid nul-terminated path and the FIFO lives
            // inside this test's private temporary directory.
            assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);
            assert_eq!(
                read_submit_envelope(&fifo).err().unwrap().to_string(),
                "submit request file is unavailable or unsafe"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn submit_cli_delegates_exact_request_to_the_shared_client() {
        use crate::control::ipc::{
            AcceptedOperation, ClientRequest, RequestStatus, ResponseEnvelope, ResponseMessage,
        };
        use crate::control::requests::InferenceRequest;
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket_path = directory.path().join("coordinator.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let ClientRequest::Submit {
                request_id,
                request,
                timeout,
            } = crate::control::ipc::read_request(&mut stream).unwrap()
            else {
                panic!("server should receive a submit");
            };
            assert!(request_id == "caller:cli-submit-1");
            assert!(timeout == std::time::Duration::from_millis(15_000));
            let InferenceRequest::Chat {
                turns, max_tokens, ..
            } = request
            else {
                panic!("server should receive the language request");
            };
            assert!(turns.len() == 1 && turns[0].content == "private submit prompt");
            assert!(max_tokens == 32);
            crate::control::ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v1(ResponseMessage::Accepted {
                    request_id: "caller:cli-submit-1".into(),
                    operation: AcceptedOperation::Submit,
                    state: RequestStatus::Queued,
                }),
            )
            .unwrap();
        });

        let file = directory.path().join("submit.json");
        write_submit_envelope(&file, &valid_submit_envelope());
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-submit",
            file.to_str().unwrap(),
            "--coordinator-socket-dir",
            directory.path().to_str().unwrap(),
        ])
        .unwrap();
        assert_eq!(dispatch(cli).unwrap(), 0);
        server.join().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn submit_lost_ack_is_not_retried() {
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket_path = directory.path().join("coordinator.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = crate::control::ipc::read_request(&mut stream).unwrap();
            let _ = stream.shutdown(std::net::Shutdown::Both);
            drop(stream);
            listener.set_nonblocking(true).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
            let mut extra_connections = 0;
            while std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((stream, _)) => {
                        extra_connections += 1;
                        drop(stream);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
            extra_connections
        });

        let file = directory.path().join("submit.json");
        write_submit_envelope(&file, &valid_submit_envelope());
        let cli = Cli::try_parse_from([
            "l3ms",
            "--request-submit",
            file.to_str().unwrap(),
            "--coordinator-socket-dir",
            directory.path().to_str().unwrap(),
        ])
        .unwrap();
        let error = dispatch(cli).unwrap_err();
        assert!(!error.to_string().contains("private submit prompt"));
        assert!(!error.to_string().contains(file.to_str().unwrap()));
        assert_eq!(server.join().unwrap(), 0);
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
