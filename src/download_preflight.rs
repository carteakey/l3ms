//! Strict download-estimate parsing and bounded preflight subprocesses.
//!
//! The Python downloader remains the authority for Hugging Face dry-run
//! semantics. This module treats its JSON as an untrusted process boundary and
//! keeps command execution suitable for a background thread: both pipes are
//! drained concurrently, retained output is capped, and every spawned child is
//! reaped on success, failure, timeout, or output overflow.
//!
//! On Unix, subprocesses are placed in their own process group. Timeout,
//! cancellation, and output overflow make a best-effort `/bin/kill` call for
//! that group before the direct child is killed and reaped. This also closes
//! inherited pipes in ordinary descendant processes. The group step depends on
//! the platform utility being present; outside Unix only the direct child can
//! be cleaned up without another dependency.

use std::{
    ffi::OsString,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

pub const ESTIMATE_SCHEMA_VERSION: u64 = 1;
pub const MAX_ESTIMATE_JSON_BYTES: usize = 1024 * 1024;
pub const MAX_ESTIMATE_MODELS: usize = 256;
pub const MAX_REPO_ID_BYTES: usize = 512;
pub const MAX_REVISION_BYTES: usize = 1024;
pub const MAX_MATCHED_FILES: u64 = 10_000;
pub const MAX_ESTIMATE_BYTES: u64 = 1_u64 << 60;

const MAX_ESTIMATE_STDERR_BYTES: usize = 64 * 1024;
const MAX_DF_STDOUT_BYTES: usize = 64 * 1024;
const MAX_DF_STDERR_BYTES: usize = 16 * 1024;
const PIPE_CHUNK_BYTES: usize = 8 * 1024;
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);
const PIPE_CLOSE_GRACE: Duration = Duration::from_millis(500);

/// One repository estimate emitted by the Python downloader.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEstimate {
    pub repo_id: String,
    pub revision: String,
    pub matched_files: u64,
    pub total_bytes: u64,
    pub download_bytes: u64,
    pub cached_bytes: u64,
}

/// Aggregate estimate fields emitted by the Python downloader.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EstimateTotals {
    pub models: u64,
    pub matched_files: u64,
    pub total_bytes: u64,
    pub download_bytes: u64,
    pub cached_bytes: u64,
}

/// Versioned estimate document emitted by `download_hf_model.py --estimate-json`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadEstimate {
    pub schema_version: u64,
    pub models: Vec<ModelEstimate>,
    pub totals: EstimateTotals,
}

/// Complete preflight result. Disk probing is advisory, so an estimate can
/// remain usable when free-space discovery fails or is unsupported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadPreflight {
    pub estimate: DownloadEstimate,
    pub disk_space: Option<DiskSpace>,
    pub warning: Option<String>,
}

/// Total and currently available bytes reported by POSIX `df -Pk`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskSpace {
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// Parse and validate a capped downloader estimate document.
pub fn parse_estimate_json(bytes: &[u8]) -> Result<DownloadEstimate> {
    if bytes.is_empty() {
        bail!("download estimate JSON is empty");
    }
    if bytes.len() > MAX_ESTIMATE_JSON_BYTES {
        bail!("download estimate JSON exceeds {MAX_ESTIMATE_JSON_BYTES} bytes");
    }

    let estimate: DownloadEstimate =
        serde_json::from_slice(bytes).context("parse download estimate JSON")?;
    validate_estimate(&estimate)?;
    Ok(estimate)
}

/// Run an argv-only estimate command with an explicit deadline and parse its
/// stdout as the strict schema above.
pub fn run_estimate_command(argv: Vec<OsString>, timeout: Duration) -> Result<DownloadEstimate> {
    let cancellation = AtomicBool::new(false);
    run_estimate_command_cancellable(argv, timeout, &cancellation)
}

/// Cancellable form of [`run_estimate_command`]. Setting `cancellation` stops,
/// terminates, and reaps an in-flight estimator.
pub fn run_estimate_command_cancellable(
    argv: Vec<OsString>,
    timeout: Duration,
    cancellation: &AtomicBool,
) -> Result<DownloadEstimate> {
    let output = run_capped_command(
        &argv,
        timeout,
        MAX_ESTIMATE_JSON_BYTES,
        MAX_ESTIMATE_STDERR_BYTES,
        cancellation,
    )?;
    if cancellation.load(Ordering::Acquire) {
        bail!("download estimate cancelled");
    }
    parse_estimate_json(&output.stdout)
}

/// Run the estimate and advisory free-space probe as one background-friendly
/// operation. Estimate failures are fatal; disk-probe failures become a
/// warning so callers can still show the remote size.
pub fn run_download_preflight(
    argv: Vec<OsString>,
    target: &Path,
    estimate_timeout: Duration,
    disk_timeout: Duration,
) -> Result<DownloadPreflight> {
    let cancellation = AtomicBool::new(false);
    run_download_preflight_cancellable(argv, target, estimate_timeout, disk_timeout, &cancellation)
}

/// Cancellable form of [`run_download_preflight`]. Cancellation is fatal,
/// including while the otherwise-advisory disk probe is running.
pub fn run_download_preflight_cancellable(
    argv: Vec<OsString>,
    target: &Path,
    estimate_timeout: Duration,
    disk_timeout: Duration,
    cancellation: &AtomicBool,
) -> Result<DownloadPreflight> {
    let estimate = run_estimate_command_cancellable(argv, estimate_timeout, cancellation)?;
    if cancellation.load(Ordering::Acquire) {
        bail!("download preflight cancelled");
    }

    let disk_result = probe_disk_space_cancellable(target, disk_timeout, cancellation);
    if cancellation.load(Ordering::Acquire) {
        bail!("download preflight cancelled");
    }
    match disk_result {
        Ok(disk_space) => {
            let warning = (disk_space.free_bytes < estimate.totals.download_bytes).then(|| {
                format!(
                    "estimated download requires {} bytes but only {} bytes are free",
                    estimate.totals.download_bytes, disk_space.free_bytes
                )
            });
            Ok(DownloadPreflight {
                estimate,
                disk_space: Some(disk_space),
                warning,
            })
        }
        Err(error) => Ok(DownloadPreflight {
            estimate,
            disk_space: None,
            warning: Some(format!("free-space probe unavailable: {error:#}")),
        }),
    }
}

/// Return the nearest existing ancestor for a possibly not-yet-created target.
pub fn nearest_existing_ancestor(target: &Path) -> Result<PathBuf> {
    if target.as_os_str().is_empty() {
        bail!("free-space target is empty");
    }

    let mut candidate = if target.is_absolute() {
        target.to_path_buf()
    } else {
        std::env::current_dir()
            .context("resolve current directory for free-space target")?
            .join(target)
    };

    loop {
        match candidate.try_exists() {
            Ok(true) => return Ok(candidate),
            Ok(false) => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("inspect free-space target ancestor {}", candidate.display())
                })
            }
        }
        if !candidate.pop() {
            bail!(
                "free-space target {} has no existing ancestor",
                target.display()
            );
        }
    }
}

/// Probe total and available bytes for the nearest existing ancestor.
pub fn probe_disk_space(target: &Path, timeout: Duration) -> Result<DiskSpace> {
    let cancellation = AtomicBool::new(false);
    probe_disk_space_cancellable(target, timeout, &cancellation)
}

/// Probe total and available bytes with
/// `df -Pk -- <nearest-existing-ancestor>`.
#[cfg(unix)]
fn probe_disk_space_cancellable(
    target: &Path,
    timeout: Duration,
    cancellation: &AtomicBool,
) -> Result<DiskSpace> {
    let ancestor = nearest_existing_ancestor(target)?;
    let argv = vec![
        OsString::from("df"),
        OsString::from("-Pk"),
        OsString::from("--"),
        ancestor.into_os_string(),
    ];
    let output = run_capped_command(
        &argv,
        timeout,
        MAX_DF_STDOUT_BYTES,
        MAX_DF_STDERR_BYTES,
        cancellation,
    )?;
    parse_df_disk_space(&output.stdout)
}

/// Disk-space probing has no dependency-free portable equivalent outside Unix.
#[cfg(not(unix))]
fn probe_disk_space_cancellable(
    _target: &Path,
    _timeout: Duration,
    _cancellation: &AtomicBool,
) -> Result<DiskSpace> {
    bail!("disk-space probing is unsupported on this platform")
}

/// Compatibility wrapper returning only available bytes.
pub fn probe_free_space(target: &Path, timeout: Duration) -> Result<u64> {
    Ok(probe_disk_space(target, timeout)?.free_bytes)
}

/// Parse total and available 1 KiB block fields from POSIX `df -Pk` output.
///
/// The capacity column is used as the anchor so spaces in filesystem or mount
/// names do not shift the numeric fields.
pub fn parse_df_disk_space(output: &[u8]) -> Result<DiskSpace> {
    if output.len() > MAX_DF_STDOUT_BYTES {
        bail!("df output exceeds {MAX_DF_STDOUT_BYTES} bytes");
    }
    let text = String::from_utf8_lossy(output);
    for line in text.lines().rev() {
        let columns = line.split_whitespace().collect::<Vec<_>>();
        for capacity_index in 4..columns.len().saturating_sub(1) {
            let Some(capacity) = columns[capacity_index].strip_suffix('%') else {
                continue;
            };
            if capacity.is_empty() || !capacity.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }

            let total_blocks = match columns[capacity_index - 3].parse::<u64>() {
                Ok(value) => value,
                Err(_) => continue,
            };
            let used_blocks = match columns[capacity_index - 2].parse::<u64>() {
                Ok(value) => value,
                Err(_) => continue,
            };
            let available_blocks = match columns[capacity_index - 1].parse::<u64>() {
                Ok(value) => value,
                Err(_) => continue,
            };
            if used_blocks > total_blocks || available_blocks > total_blocks {
                bail!("df output contains inconsistent block counts");
            }
            let total_bytes = total_blocks
                .checked_mul(1024)
                .ok_or_else(|| anyhow!("df total byte count overflow"))?;
            let free_bytes = available_blocks
                .checked_mul(1024)
                .ok_or_else(|| anyhow!("df available byte count overflow"))?;
            return Ok(DiskSpace {
                total_bytes,
                free_bytes,
            });
        }
    }
    bail!("df output contains no parseable filesystem row")
}

/// Compatibility wrapper returning only the parsed available bytes.
pub fn parse_df_available_bytes(output: &[u8]) -> Result<u64> {
    Ok(parse_df_disk_space(output)?.free_bytes)
}

fn validate_estimate(estimate: &DownloadEstimate) -> Result<()> {
    if estimate.schema_version != ESTIMATE_SCHEMA_VERSION {
        bail!(
            "unsupported download estimate schema version {}; expected {ESTIMATE_SCHEMA_VERSION}",
            estimate.schema_version
        );
    }
    if estimate.models.len() > MAX_ESTIMATE_MODELS {
        bail!(
            "download estimate contains {} models; maximum is {MAX_ESTIMATE_MODELS}",
            estimate.models.len()
        );
    }

    let mut calculated = EstimateTotals {
        models: u64::try_from(estimate.models.len()).context("estimate model count overflow")?,
        ..EstimateTotals::default()
    };
    for (index, model) in estimate.models.iter().enumerate() {
        validate_required_string(
            &format!("models[{index}].repo_id"),
            &model.repo_id,
            MAX_REPO_ID_BYTES,
        )?;
        validate_required_string(
            &format!("models[{index}].revision"),
            &model.revision,
            MAX_REVISION_BYTES,
        )?;
        if model.matched_files > MAX_MATCHED_FILES {
            bail!("models[{index}].matched_files exceeds {MAX_MATCHED_FILES}");
        }
        validate_model_bytes(index, model)?;

        calculated.matched_files = calculated
            .matched_files
            .checked_add(model.matched_files)
            .ok_or_else(|| anyhow!("download estimate matched_files overflow"))?;
        calculated.total_bytes = calculated
            .total_bytes
            .checked_add(model.total_bytes)
            .ok_or_else(|| anyhow!("download estimate total_bytes overflow"))?;
        calculated.download_bytes = calculated
            .download_bytes
            .checked_add(model.download_bytes)
            .ok_or_else(|| anyhow!("download estimate download_bytes overflow"))?;
        calculated.cached_bytes = calculated
            .cached_bytes
            .checked_add(model.cached_bytes)
            .ok_or_else(|| anyhow!("download estimate cached_bytes overflow"))?;
    }

    if calculated.matched_files > MAX_MATCHED_FILES {
        bail!("download estimate total matched_files exceeds {MAX_MATCHED_FILES}");
    }
    validate_bounded_bytes("download estimate total_bytes", calculated.total_bytes)?;
    validate_bounded_bytes(
        "download estimate download_bytes",
        calculated.download_bytes,
    )?;
    validate_bounded_bytes("download estimate cached_bytes", calculated.cached_bytes)?;

    if estimate.totals != calculated {
        bail!(
            "download estimate totals are inconsistent: expected {calculated:?}, received {:?}",
            estimate.totals
        );
    }
    Ok(())
}

fn validate_required_string(name: &str, value: &str, max_bytes: usize) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{name} is required");
    }
    if value.len() > max_bytes {
        bail!("{name} exceeds {max_bytes} bytes");
    }
    if value.trim() != value {
        bail!("{name} must not have surrounding whitespace");
    }
    if value.chars().any(char::is_control) {
        bail!("{name} must not contain control characters");
    }
    Ok(())
}

fn validate_model_bytes(index: usize, model: &ModelEstimate) -> Result<()> {
    validate_bounded_bytes(&format!("models[{index}].total_bytes"), model.total_bytes)?;
    validate_bounded_bytes(
        &format!("models[{index}].download_bytes"),
        model.download_bytes,
    )?;
    validate_bounded_bytes(&format!("models[{index}].cached_bytes"), model.cached_bytes)?;
    if model.download_bytes > model.total_bytes {
        bail!("models[{index}].download_bytes exceeds total_bytes");
    }
    if model.cached_bytes > model.total_bytes {
        bail!("models[{index}].cached_bytes exceeds total_bytes");
    }
    Ok(())
}

fn validate_bounded_bytes(name: &str, value: u64) -> Result<()> {
    if value > MAX_ESTIMATE_BYTES {
        bail!("{name} exceeds {MAX_ESTIMATE_BYTES} bytes");
    }
    Ok(())
}

#[derive(Debug)]
struct CapturedCommand {
    stdout: Vec<u8>,
}

#[derive(Debug)]
struct CapturedPipe {
    bytes: Vec<u8>,
    overflow: bool,
}

enum ChildEnd {
    Exited(ExitStatus),
    Cancelled,
    TimedOut,
    Oversized,
    PollFailed(io::Error),
}

fn run_capped_command(
    argv: &[OsString],
    timeout: Duration,
    stdout_limit: usize,
    stderr_limit: usize,
    cancellation: &AtomicBool,
) -> Result<CapturedCommand> {
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| anyhow!("child command argv is empty"))?;
    if program.is_empty() {
        bail!("child command program is empty");
    }
    if cancellation.load(Ordering::Acquire) {
        bail!("child command cancelled before spawn");
    }

    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let spawned = {
        #[cfg(test)]
        let _fixture_guard = crate::control::process::executable_fixture_guard();
        command.spawn()
    };
    let mut child = spawned.with_context(|| format!("spawn child command {:?}", program))?;
    let stdout = child
        .stdout
        .take()
        .expect("piped child stdout must be available after spawn");
    let stderr = child
        .stderr
        .take()
        .expect("piped child stderr must be available after spawn");
    let output_overflow = Arc::new(AtomicBool::new(false));
    let stdout_reader = spawn_pipe_reader(stdout, stdout_limit, Arc::clone(&output_overflow));
    let stderr_reader = spawn_pipe_reader(stderr, stderr_limit, Arc::clone(&output_overflow));

    let started = Instant::now();
    let mut exit_status = None;
    let child_end = loop {
        if cancellation.load(Ordering::Acquire) {
            break ChildEnd::Cancelled;
        }
        if output_overflow.load(Ordering::Acquire) {
            break ChildEnd::Oversized;
        }
        if exit_status.is_none() {
            match child.try_wait() {
                Ok(Some(status)) => exit_status = Some(status),
                Ok(None) => {}
                Err(error) => break ChildEnd::PollFailed(error),
            }
        }
        if exit_status.is_some() && stdout_reader.is_finished() && stderr_reader.is_finished() {
            break ChildEnd::Exited(exit_status.take().expect("exit status was checked"));
        }
        let elapsed = started.elapsed();
        if elapsed >= timeout {
            break ChildEnd::TimedOut;
        }
        thread::sleep(CHILD_POLL_INTERVAL.min(timeout.saturating_sub(elapsed)));
    };

    let status = match child_end {
        ChildEnd::Exited(status) => status,
        abnormal => {
            let leader_reaped = exit_status.is_some();
            let cleanup_result = terminate_and_reap(&mut child, leader_reaped);
            finish_terminated_pipe_readers(stdout_reader, stderr_reader);
            cleanup_result?;
            match abnormal {
                ChildEnd::Cancelled => bail!("child command cancelled"),
                ChildEnd::TimedOut => {
                    bail!("child command timed out after {} ms", timeout.as_millis())
                }
                ChildEnd::Oversized => {
                    bail!("child command output exceeded its configured limit")
                }
                ChildEnd::PollFailed(error) => return Err(error).context("poll child command"),
                ChildEnd::Exited(_) => unreachable!("exited child handled above"),
            }
        }
    };
    let stdout = join_pipe_reader(stdout_reader, "stdout")?;
    let stderr = join_pipe_reader(stderr_reader, "stderr")?;

    if stdout.overflow {
        bail!("child stdout exceeds {stdout_limit} bytes");
    }
    if stderr.overflow {
        bail!("child stderr exceeds {stderr_limit} bytes");
    }

    if !status.success() {
        let preview = utf8_safe_preview(&stderr.bytes, stderr_limit);
        if preview.is_empty() {
            bail!("child command exited with {status}");
        }
        bail!("child command exited with {status}: {preview}");
    }
    Ok(CapturedCommand {
        stdout: stdout.bytes,
    })
}

fn spawn_pipe_reader(
    mut pipe: impl Read + Send + 'static,
    limit: usize,
    output_overflow: Arc<AtomicBool>,
) -> thread::JoinHandle<io::Result<CapturedPipe>> {
    thread::spawn(move || {
        let mut bytes = Vec::with_capacity(limit.min(PIPE_CHUNK_BYTES));
        let mut overflow = false;
        let mut chunk = [0_u8; PIPE_CHUNK_BYTES];
        loop {
            let read = pipe.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            let remaining = limit.saturating_sub(bytes.len());
            let retained = remaining.min(read);
            bytes.extend_from_slice(&chunk[..retained]);
            if retained < read {
                overflow = true;
                output_overflow.store(true, Ordering::Release);
            }
        }
        Ok(CapturedPipe { bytes, overflow })
    })
}

fn join_pipe_reader(
    reader: thread::JoinHandle<io::Result<CapturedPipe>>,
    name: &str,
) -> Result<CapturedPipe> {
    reader
        .join()
        .map_err(|_| anyhow!("child {name} reader panicked"))?
        .with_context(|| format!("read child {name}"))
}

fn terminate_and_reap(child: &mut Child, leader_reaped: bool) -> Result<()> {
    #[cfg(unix)]
    terminate_process_group(child.id());
    if leader_reaped {
        return Ok(());
    }
    let _ = child.kill();
    child.wait().context("reap terminated child command")?;
    Ok(())
}

fn finish_terminated_pipe_readers(
    stdout_reader: thread::JoinHandle<io::Result<CapturedPipe>>,
    stderr_reader: thread::JoinHandle<io::Result<CapturedPipe>>,
) {
    let started = Instant::now();
    while !(stdout_reader.is_finished() && stderr_reader.is_finished())
        && started.elapsed() < PIPE_CLOSE_GRACE
    {
        thread::sleep(CHILD_POLL_INTERVAL.min(PIPE_CLOSE_GRACE.saturating_sub(started.elapsed())));
    }

    // A descendant outside the process group can retain an inherited pipe.
    // Dropping an unfinished handle detaches that reader rather than violating
    // the caller's deadline indefinitely.
    if stdout_reader.is_finished() {
        let _ = stdout_reader.join();
    }
    if stderr_reader.is_finished() {
        let _ = stderr_reader.join();
    }
}

#[cfg(unix)]
fn terminate_process_group(process_id: u32) {
    let group = format!("-{process_id}");
    let _ = {
        #[cfg(test)]
        let _fixture_guard = crate::control::process::executable_fixture_guard();
        Command::new("/bin/kill")
            .args(["-KILL", "--", group.as_str()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
    };
}

fn utf8_safe_preview(bytes: &[u8], limit: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut preview = String::with_capacity(text.len().min(limit));
    for character in text.chars() {
        if preview.len() + character.len_utf8() > limit {
            break;
        }
        preview.push(character);
    }
    preview.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsStr, fs, process::Command, sync::Arc};

    use serde_json::{json, Value};
    use tempfile::TempDir;

    use super::*;

    fn model(repo_id: &str, files: u64, total: u64, download: u64, cached: u64) -> Value {
        json!({
            "repo_id": repo_id,
            "revision": "main",
            "matched_files": files,
            "total_bytes": total,
            "download_bytes": download,
            "cached_bytes": cached,
        })
    }

    fn document(models: Vec<Value>) -> Value {
        let mut matched_files = 0_u64;
        let mut total_bytes = 0_u64;
        let mut download_bytes = 0_u64;
        let mut cached_bytes = 0_u64;
        for model in &models {
            matched_files += model["matched_files"].as_u64().unwrap();
            total_bytes += model["total_bytes"].as_u64().unwrap();
            download_bytes += model["download_bytes"].as_u64().unwrap();
            cached_bytes += model["cached_bytes"].as_u64().unwrap();
        }
        json!({
            "schema_version": ESTIMATE_SCHEMA_VERSION,
            "totals": {
                "models": models.len() as u64,
                "matched_files": matched_files,
                "total_bytes": total_bytes,
                "download_bytes": download_bytes,
                "cached_bytes": cached_bytes,
            },
            "models": models,
        })
    }

    fn parse_value(value: &Value) -> Result<DownloadEstimate> {
        parse_estimate_json(&serde_json::to_vec(value).unwrap())
    }

    #[test]
    fn parses_valid_versioned_estimate_and_allows_non_partitioned_cache_states() {
        let value = document(vec![
            model("org/one", 3, 100, 80, 40),
            model("org/two", 2, 50, 10, 50),
        ]);

        let estimate = parse_value(&value).unwrap();

        assert_eq!(estimate.schema_version, ESTIMATE_SCHEMA_VERSION);
        assert_eq!(estimate.models.len(), 2);
        assert_eq!(estimate.totals.models, 2);
        assert_eq!(estimate.totals.matched_files, 5);
        assert_eq!(estimate.totals.total_bytes, 150);
        assert_eq!(estimate.totals.download_bytes, 90);
        assert_eq!(estimate.totals.cached_bytes, 90);
    }

    #[test]
    fn rejects_schema_strings_model_and_file_bounds() {
        let mut wrong_version = document(vec![]);
        wrong_version["schema_version"] = json!(2);
        assert!(parse_value(&wrong_version)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));

        let too_many = document(
            (0..=MAX_ESTIMATE_MODELS)
                .map(|index| model(&format!("org/{index}"), 0, 0, 0, 0))
                .collect(),
        );
        assert!(parse_value(&too_many)
            .unwrap_err()
            .to_string()
            .contains("maximum"));

        let long_repo = document(vec![model(&"x".repeat(MAX_REPO_ID_BYTES + 1), 1, 1, 1, 0)]);
        assert!(parse_value(&long_repo)
            .unwrap_err()
            .to_string()
            .contains("repo_id"));

        let mut long_revision = model("org/revision", 1, 1, 1, 0);
        long_revision["revision"] = json!("x".repeat(MAX_REVISION_BYTES + 1));
        assert!(parse_value(&document(vec![long_revision]))
            .unwrap_err()
            .to_string()
            .contains("revision"));

        let too_many_files = document(vec![model("org/files", MAX_MATCHED_FILES + 1, 1, 1, 0)]);
        assert!(parse_value(&too_many_files)
            .unwrap_err()
            .to_string()
            .contains("matched_files"));
    }

    #[test]
    fn rejects_byte_relations_overflow_inconsistent_totals_and_unknown_fields() {
        let download_too_large = document(vec![model("org/bad", 1, 10, 11, 0)]);
        assert!(parse_value(&download_too_large)
            .unwrap_err()
            .to_string()
            .contains("download_bytes exceeds"));

        let bytes_over_bound = document(vec![model("org/huge", 1, MAX_ESTIMATE_BYTES + 1, 0, 0)]);
        assert!(parse_value(&bytes_over_bound)
            .unwrap_err()
            .to_string()
            .contains("total_bytes"));

        let overflow_models = (0..16)
            .map(|index| model(&format!("org/{index}"), 1, MAX_ESTIMATE_BYTES, 0, 0))
            .collect::<Vec<_>>();
        let overflow = json!({
            "schema_version": ESTIMATE_SCHEMA_VERSION,
            "models": overflow_models,
            "totals": {
                "models": 16,
                "matched_files": 16,
                "total_bytes": 0,
                "download_bytes": 0,
                "cached_bytes": 0,
            },
        });
        assert!(parse_value(&overflow)
            .unwrap_err()
            .to_string()
            .contains("overflow"));

        let mut inconsistent = document(vec![model("org/one", 1, 10, 4, 6)]);
        inconsistent["totals"]["download_bytes"] = json!(5);
        assert!(parse_value(&inconsistent)
            .unwrap_err()
            .to_string()
            .contains("inconsistent"));

        let mut unknown = document(vec![]);
        unknown["unexpected"] = json!(true);
        assert!(parse_value(&unknown).is_err());
    }

    #[test]
    fn rejects_json_over_the_input_cap_before_parsing() {
        let oversized = vec![b' '; MAX_ESTIMATE_JSON_BYTES + 1];
        assert!(parse_estimate_json(&oversized)
            .unwrap_err()
            .to_string()
            .contains("exceeds"));
    }

    #[test]
    fn parses_df_with_headers_and_spaces_in_source_and_mount() {
        let output = concat!(
            "Filesystem name 1024-blocks Used Available Capacity Mounted on\n",
            "network source with spaces 1000 250 750 25% /Volumes/Model Store\n",
        );
        assert_eq!(
            parse_df_disk_space(output.as_bytes()).unwrap(),
            DiskSpace {
                total_bytes: 1000 * 1024,
                free_bytes: 750 * 1024,
            }
        );
        assert_eq!(
            parse_df_available_bytes(output.as_bytes()).unwrap(),
            750 * 1024
        );

        assert!(parse_df_disk_space(b"source 1000 250 750 25%\n").is_err());
        assert!(parse_df_disk_space(b"source 1000 1250 750 25% /\n").is_err());
    }

    #[test]
    fn nearest_ancestor_handles_a_missing_target_tree() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("not-created/deeper/models");
        assert_eq!(nearest_existing_ancestor(&target).unwrap(), temp.path());
        assert!(nearest_existing_ancestor(Path::new("")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn free_space_probe_uses_the_nearest_existing_ancestor() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("not-created/deeper/models");
        let disk_space = probe_disk_space(&target, Duration::from_secs(2)).unwrap();
        assert!(disk_space.total_bytes >= disk_space.free_bytes);
        assert!(disk_space.free_bytes > 0);
        assert!(probe_free_space(&target, Duration::from_secs(2)).unwrap() > 0);
    }

    #[cfg(unix)]
    fn shell_argv(script: impl AsRef<OsStr>) -> Vec<OsString> {
        vec![
            OsString::from("sh"),
            OsString::from("-c"),
            script.as_ref().into(),
        ]
    }

    #[cfg(unix)]
    #[test]
    fn estimate_runner_accepts_valid_json_and_rejects_nonzero_status() {
        let json = serde_json::to_string(&document(vec![model("org/one", 1, 5, 5, 0)])).unwrap();
        let valid = vec![
            OsString::from("sh"),
            OsString::from("-c"),
            OsString::from("printf '%s' \"$1\""),
            OsString::from("sh"),
            OsString::from(json),
        ];
        let estimate = run_estimate_command(valid, Duration::from_secs(2)).unwrap();
        assert_eq!(estimate.totals.download_bytes, 5);

        let error = run_estimate_command(
            shell_argv("printf 'estimate failed' >&2; exit 7"),
            Duration::from_secs(2),
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("exit status: 7") || message.contains("exited with 7"));
        assert!(message.contains("estimate failed"));
    }

    #[cfg(unix)]
    #[test]
    fn preflight_keeps_a_valid_estimate_when_disk_probe_is_unavailable() {
        let json = serde_json::to_string(&document(vec![model("org/one", 1, 5, 5, 0)])).unwrap();
        let argv = vec![
            OsString::from("sh"),
            OsString::from("-c"),
            OsString::from("printf '%s' \"$1\""),
            OsString::from("sh"),
            OsString::from(json),
        ];

        let result = run_download_preflight(
            argv,
            Path::new(""),
            Duration::from_secs(2),
            Duration::from_secs(2),
        )
        .unwrap();

        assert_eq!(result.estimate.totals.download_bytes, 5);
        assert_eq!(result.disk_space, None);
        assert!(result
            .warning
            .as_deref()
            .is_some_and(|warning| warning.contains("free-space target is empty")));
    }

    #[cfg(unix)]
    #[test]
    fn estimate_runner_rejects_timeout_and_oversized_stdout() {
        let started = Instant::now();
        let timeout_error =
            run_estimate_command(shell_argv("exec sleep 5"), Duration::from_millis(100))
                .unwrap_err();
        assert!(timeout_error.to_string().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));

        let oversize_error = run_estimate_command(
            shell_argv(format!("head -c {} /dev/zero", MAX_ESTIMATE_JSON_BYTES + 1)),
            Duration::from_secs(2),
        )
        .unwrap_err();
        assert!(oversize_error.to_string().contains("output exceeded"));
    }

    #[cfg(unix)]
    #[test]
    fn timed_out_child_is_killed_and_reaped() {
        let temp = TempDir::new().unwrap();
        let pid_file = temp.path().join("child.pid");
        let argv = vec![
            OsString::from("sh"),
            OsString::from("-c"),
            OsString::from("printf '%s' \"$$\" > \"$1\"; exec sleep 5"),
            OsString::from("sh"),
            pid_file.clone().into_os_string(),
        ];

        let error = run_estimate_command(argv, Duration::from_millis(150)).unwrap_err();

        assert!(error.to_string().contains("timed out"));
        let process_id = fs::read_to_string(&pid_file).unwrap();
        assert_process_gone(process_id.trim(), "timed-out child");
    }

    #[cfg(unix)]
    #[test]
    fn timeout_kills_descendant_that_keeps_output_pipes_open() {
        let temp = TempDir::new().unwrap();
        let pid_file = temp.path().join("descendant.pid");
        let argv = vec![
            OsString::from("sh"),
            OsString::from("-c"),
            OsString::from("sleep 5 & printf '%s' \"$!\" > \"$1\"; exit 0"),
            OsString::from("sh"),
            pid_file.clone().into_os_string(),
        ];

        let started = Instant::now();
        let error = run_estimate_command(argv, Duration::from_millis(150)).unwrap_err();

        assert!(error.to_string().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
        let process_id = fs::read_to_string(&pid_file).unwrap();
        assert_process_gone(process_id.trim(), "pipe-holding descendant");
    }

    #[cfg(unix)]
    #[test]
    fn cancellable_preflight_returns_promptly_and_reaps_estimator() {
        let temp = TempDir::new().unwrap();
        let pid_file = temp.path().join("estimator.pid");
        let target = temp.path().to_path_buf();
        let argv = vec![
            OsString::from("sh"),
            OsString::from("-c"),
            OsString::from("printf '%s' \"$$\" > \"$1\"; exec sleep 5"),
            OsString::from("sh"),
            pid_file.clone().into_os_string(),
        ];
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation);
        let worker = thread::spawn(move || {
            run_download_preflight_cancellable(
                argv,
                &target,
                Duration::from_secs(10),
                Duration::from_secs(2),
                &worker_cancellation,
            )
        });

        let wait_started = Instant::now();
        while !pid_file.exists() && wait_started.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(pid_file.exists(), "estimator did not start");

        let cancelled_at = Instant::now();
        cancellation.store(true, Ordering::Release);
        let error = worker.join().unwrap().unwrap_err();

        assert!(error.to_string().contains("cancelled"));
        assert!(cancelled_at.elapsed() < Duration::from_secs(2));
        let process_id = fs::read_to_string(&pid_file).unwrap();
        assert_process_gone(process_id.trim(), "cancelled estimator");
    }

    #[cfg(unix)]
    fn assert_process_gone(process_id: &str, description: &str) {
        let started = Instant::now();
        while process_exists(process_id) && started.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!process_exists(process_id), "{description} still exists");
    }

    #[cfg(unix)]
    fn process_exists(process_id: &str) -> bool {
        {
            let _fixture_guard = crate::control::process::executable_fixture_guard();
            Command::new("/bin/kill")
                .args(["-0", process_id])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        }
        .is_ok_and(|status| status.success())
    }
}
