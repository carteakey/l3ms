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

#[derive(Debug, Parser)]
#[command(
    name = "l3ms",
    version,
    about = "L3MS launcher (TUI + interactive model and bench CLI)",
    group(
        ArgGroup::new("action")
            .args(["run", "bench", "list", "quickstart", "settings", "export_profile", "import_profile", "compare_results", "coordinator_status", "coordinator_cancel"])
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
    #[arg(long, value_name = "ID", requires = "coordinator_socket_dir")]
    coordinator_status: Option<String>,

    /// Cancel one retained queued/running coordinator request by request ID.
    #[arg(long, value_name = "ID", requires = "coordinator_socket_dir")]
    coordinator_cancel: Option<String>,

    /// Existing private directory containing coordinator.sock; never inferred.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoordinatorOperation {
    Status,
    Cancel,
}

#[derive(Debug, PartialEq, Eq)]
struct CommandSpec {
    program: OsString,
    args: Vec<OsString>,
}

/// Parse process arguments and launch the requested mode.
pub fn run() -> Result<u8> {
    dispatch(Cli::parse())
}

fn dispatch(cli: Cli) -> Result<u8> {
    if let Some((operation, request_id, socket_directory)) = coordinator_action(&cli)? {
        let response = coordinator_exchange(socket_directory, operation, request_id)?;
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

fn coordinator_action(cli: &Cli) -> Result<Option<(CoordinatorOperation, &str, &Path)>> {
    let action = match (
        cli.coordinator_status.as_deref(),
        cli.coordinator_cancel.as_deref(),
    ) {
        (Some(_), Some(_)) => bail!("choose only one coordinator operation"),
        (Some(request_id), None) => Some((CoordinatorOperation::Status, request_id)),
        (None, Some(request_id)) => Some((CoordinatorOperation::Cancel, request_id)),
        (None, None) => None,
    };

    let Some((operation, request_id)) = action else {
        if cli.coordinator_socket_dir.is_some() {
            bail!("--coordinator-socket-dir requires --coordinator-status or --coordinator-cancel");
        }
        return Ok(None);
    };

    if cli.extra.is_some() {
        bail!("--extra cannot be used with coordinator operations");
    }
    let socket_directory = cli.coordinator_socket_dir.as_deref().ok_or_else(|| {
        anyhow::anyhow!("coordinator operations require --coordinator-socket-dir")
    })?;
    Ok(Some((operation, request_id, socket_directory)))
}

fn coordinator_exchange(
    socket_directory: &Path,
    operation: CoordinatorOperation,
    request_id: &str,
) -> Result<crate::control::ipc::ResponseMessage> {
    #[cfg(target_os = "linux")]
    {
        coordinator_exchange_linux(socket_directory, operation, request_id)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (socket_directory, operation, request_id);
        bail!("coordinator IPC is supported only on Linux");
    }
}

#[cfg(target_os = "linux")]
fn coordinator_exchange_linux(
    socket_directory: &Path,
    operation: CoordinatorOperation,
    request_id: &str,
) -> Result<crate::control::ipc::ResponseMessage> {
    use crate::control::{
        ipc::{ProtocolError, RequestEnvelope, WireRequestMessage},
        local_ipc::{LocalConnection, LocalIpcError},
        runtime::{NeverCancel, OperationContext},
    };

    crate::control::ipc::validate_request_id(request_id)
        .map_err(|_| anyhow::anyhow!("invalid coordinator request ID"))?;
    let message = match operation {
        CoordinatorOperation::Status => WireRequestMessage::Status {
            request_id: request_id.to_owned(),
        },
        CoordinatorOperation::Cancel => WireRequestMessage::Cancel {
            request_id: request_id.to_owned(),
        },
    };

    let context = OperationContext::new(Instant::now() + COORDINATOR_TIMEOUT, &NeverCancel);
    let mut connection = LocalConnection::connect(socket_directory, &context)
        .map_err(|error| safe_coordinator_error(error, true))?;
    connection
        .write_request(&RequestEnvelope::v1(message), &context)
        .map_err(|error| safe_coordinator_error(error, false))?;
    connection
        .read_response(&context)
        .map(|response| response.message)
        .map_err(|error| {
            let error = match error {
                LocalIpcError::Protocol(ProtocolError::InvalidRequestId) => {
                    anyhow::anyhow!("invalid coordinator request ID")
                }
                other => safe_coordinator_error(other, false),
            };
            error
        })
        .and_then(|response| validate_coordinator_response(operation, response))
}

#[cfg(target_os = "linux")]
fn safe_coordinator_error(
    error: crate::control::local_ipc::LocalIpcError,
    connecting: bool,
) -> anyhow::Error {
    use crate::control::local_ipc::LocalIpcError;

    let message = match error {
        LocalIpcError::UnsafePath => "coordinator directory or socket is unsafe",
        LocalIpcError::Unavailable if connecting => "coordinator is unavailable",
        LocalIpcError::Unavailable => "coordinator exchange failed",
        LocalIpcError::Deadline | LocalIpcError::Cancelled => "coordinator operation timed out",
        LocalIpcError::Protocol(crate::control::ipc::ProtocolError::InvalidRequestId) => {
            "invalid coordinator request ID"
        }
        LocalIpcError::Protocol(_) | LocalIpcError::ResponseMismatch => {
            "coordinator response is invalid"
        }
        LocalIpcError::PeerDenied => "coordinator peer is not authorized",
        LocalIpcError::OwnershipUnavailable => "coordinator ownership is unavailable",
        LocalIpcError::ExistingSocket => "coordinator endpoint is unavailable",
        LocalIpcError::Io | LocalIpcError::OutOfOrder => "coordinator exchange failed",
    };
    anyhow::anyhow!(message)
}

fn coordinator_response_json(response: &crate::control::ipc::ResponseMessage) -> Result<String> {
    let json = serde_json::to_string(response)
        .map_err(|_| anyhow::anyhow!("coordinator response could not be encoded"))?;
    if json.len() > MAX_COORDINATOR_RESPONSE_BYTES {
        bail!("coordinator response exceeded the output limit");
    }
    Ok(json)
}

fn validate_coordinator_response(
    operation: CoordinatorOperation,
    response: crate::control::ipc::ResponseMessage,
) -> Result<crate::control::ipc::ResponseMessage> {
    use crate::control::ipc::{AcceptedOperation, RequestStatus, ResponseMessage};

    let matches_operation = matches!(
        (&operation, &response),
        (_, ResponseMessage::Error { .. })
            | (CoordinatorOperation::Status, ResponseMessage::Status { .. })
            | (
                CoordinatorOperation::Cancel,
                ResponseMessage::Accepted {
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Queued
                        | RequestStatus::Running
                        | RequestStatus::Cancelled,
                    ..
                }
            )
    );
    if !matches_operation {
        bail!("coordinator response is invalid");
    }
    Ok(response)
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
        assert!(Cli::try_parse_from(["l3ms", "--coordinator-status", "request-1",]).is_err());
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
        assert!(coordinator_action(&cli).is_err());

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
        assert!(coordinator_action(&cli).is_err());

        let cli = Cli::try_parse_from([
            "l3ms",
            "--coordinator-cancel",
            "request-1",
            "--coordinator-socket-dir",
            "/run/user/1000/l3ms",
        ])
        .unwrap();
        assert_eq!(
            coordinator_action(&cli).unwrap(),
            Some((
                CoordinatorOperation::Cancel,
                "request-1",
                Path::new("/run/user/1000/l3ms")
            ))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn coordinator_status_and_cancel_use_typed_private_socket_exchanges() {
        use crate::control::ipc::{
            AcceptedOperation, ClientRequest, RequestStatus, ResponseEnvelope, ResponseMessage,
        };
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        for (operation, response) in [
            (
                CoordinatorOperation::Status,
                ResponseMessage::Status {
                    request_id: "request-1".into(),
                    state: RequestStatus::Running,
                },
            ),
            (
                CoordinatorOperation::Cancel,
                ResponseMessage::Accepted {
                    request_id: "request-1".into(),
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Cancelled,
                },
            ),
        ] {
            let directory = tempfile::tempdir().unwrap();
            std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
            let socket_path = directory.path().join("coordinator.sock");
            let listener = UnixListener::bind(&socket_path).unwrap();
            std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let request = crate::control::ipc::read_request(&mut stream).unwrap();
                crate::control::ipc::write_response(
                    &mut stream,
                    &ResponseEnvelope::v1(response.clone()),
                )
                .unwrap();
                request
            });

            let response = coordinator_exchange(directory.path(), operation, "request-1").unwrap();
            let request = server.join().unwrap();
            match (operation, request) {
                (CoordinatorOperation::Status, ClientRequest::Status { request_id }) => {
                    assert_eq!(request_id, "request-1");
                    assert_eq!(
                        response,
                        ResponseMessage::Status {
                            request_id,
                            state: RequestStatus::Running
                        }
                    );
                }
                (CoordinatorOperation::Cancel, ClientRequest::Cancel { request_id }) => {
                    assert_eq!(request_id, "request-1");
                    assert_eq!(
                        response,
                        ResponseMessage::Accepted {
                            request_id,
                            operation: AcceptedOperation::Cancel,
                            state: RequestStatus::Cancelled
                        }
                    );
                }
                _ => panic!("coordinator operation did not match its typed IPC request"),
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn coordinator_rejects_wrong_operation_response() {
        use crate::control::ipc::{
            AcceptedOperation, ClientRequest, RequestStatus, ResponseEnvelope, ResponseMessage,
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
            let request = crate::control::ipc::read_request(&mut stream).unwrap();
            crate::control::ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v1(ResponseMessage::Accepted {
                    request_id: "request-1".into(),
                    operation: AcceptedOperation::Submit,
                    state: RequestStatus::Queued,
                }),
            )
            .unwrap();
            request
        });

        let error =
            coordinator_exchange(directory.path(), CoordinatorOperation::Status, "request-1")
                .unwrap_err();
        assert_eq!(error.to_string(), "coordinator response is invalid");
        assert!(matches!(
            server.join().unwrap(),
            ClientRequest::Status { request_id } if request_id == "request-1"
        ));
    }

    #[test]
    fn coordinator_response_json_is_compact_and_bounded() {
        use crate::control::ipc::{AcceptedOperation, RequestStatus, ResponseMessage};

        for state in [
            RequestStatus::Queued,
            RequestStatus::Running,
            RequestStatus::Cancelled,
        ] {
            assert!(validate_coordinator_response(
                CoordinatorOperation::Cancel,
                ResponseMessage::Accepted {
                    request_id: "request-1".into(),
                    operation: AcceptedOperation::Cancel,
                    state,
                }
            )
            .is_ok());
        }
        for state in [
            RequestStatus::Succeeded,
            RequestStatus::Failed,
            RequestStatus::Interrupted,
            RequestStatus::NotFound,
        ] {
            assert!(validate_coordinator_response(
                CoordinatorOperation::Cancel,
                ResponseMessage::Accepted {
                    request_id: "request-1".into(),
                    operation: AcceptedOperation::Cancel,
                    state,
                }
            )
            .is_err());
        }

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

    #[cfg(target_os = "linux")]
    #[test]
    fn coordinator_transport_failures_use_fixed_messages_without_fallback() {
        use std::os::unix::fs::PermissionsExt;

        let missing = tempfile::tempdir().unwrap();
        std::fs::set_permissions(missing.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let unavailable =
            coordinator_exchange(missing.path(), CoordinatorOperation::Status, "request-1")
                .unwrap_err();
        assert_eq!(unavailable.to_string(), "coordinator is unavailable");
        assert!(!missing.path().join("coordinator.sock").exists());

        let unsafe_directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(
            unsafe_directory.path(),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        let unsafe_path = coordinator_exchange(
            unsafe_directory.path(),
            CoordinatorOperation::Cancel,
            "request-1",
        )
        .unwrap_err();
        assert_eq!(
            unsafe_path.to_string(),
            "coordinator directory or socket is unsafe"
        );
        assert!(!unsafe_path
            .to_string()
            .contains(unsafe_directory.path().to_str().unwrap()));
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
}
