//! Bounded, point-in-time NVIDIA GPU idle observation.
//!
//! This module asks a fixed `nvidia-smi` executable for one configured GPU's
//! identity and memory, checks compute-process claimants, then repeats the GPU
//! snapshot. It does not enumerate graphics/desktop processes, establish an
//! admission fence, prove service or cgroup cleanup, or prove GPU lease
//! ownership. A caller considering a runtime switch must own service/cgroup
//! cleanup and exclude direct bypasses before relying on this observation.
//!
//! Each query uses the bounded Linux process runner with a 64 KiB combined
//! output cap and the caller's deadline/cancellation and recovery policy.
//! Synchronous OS process creation itself is not preemptible, as documented by
//! the process runner. The executable is trusted not to daemonize or escape its
//! process group.

use std::error::Error;
use std::fmt;
#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use super::process::{self, ProcessFailure, ProcessFailureKind, ProcessLimits};
use super::runtime::{OperationContext, RecoveryPolicy};

/// Fixed system installation selected and reviewed for this adapter.
pub const NVIDIA_SMI_PATH: &str = "/usr/bin/nvidia-smi";
/// Maximum combined stdout/stderr bytes retained from any single query.
pub const MAX_QUERY_BYTES: usize = 64 * 1024;
/// Maximum rows accepted from the compute-app query.
pub const MAX_COMPUTE_APP_ROWS: usize = 256;
const MAX_GPU_ROWS: usize = 8;

/// Explicitly configured identity and idle-memory allowance for one physical
/// NVIDIA GPU. There is deliberately no automatic or sampled baseline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuProbeConfig {
    gpu_uuid: String,
    expected_index: u32,
    max_idle_mib: u64,
}

impl GpuProbeConfig {
    /// Configure an exact physical GPU UUID, expected current index, and
    /// explicit maximum idle memory usage in MiB.
    pub fn new(
        gpu_uuid: impl Into<String>,
        expected_index: u32,
        max_idle_mib: u64,
    ) -> Result<Self, GpuProbeFailure> {
        let gpu_uuid = gpu_uuid.into();
        if !valid_gpu_uuid(&gpu_uuid) {
            return Err(GpuProbeFailure::new(
                GpuProbeFailureKind::InvalidConfiguration,
            ));
        }
        Ok(Self {
            gpu_uuid,
            expected_index,
            max_idle_mib,
        })
    }

    pub fn gpu_uuid(&self) -> &str {
        &self.gpu_uuid
    }

    pub fn expected_index(&self) -> u32 {
        self.expected_index
    }

    pub fn max_idle_mib(&self) -> u64 {
        self.max_idle_mib
    }
}

/// The strongest positive result this probe produces: a bounded observation
/// of the last NVIDIA memory snapshot after a compute-app query.
///
/// This value is not an admission fence, service-cleanup acknowledgment, GPU
/// lease, or guarantee that the device remains idle after the query returns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdleGpuObservation {
    gpu_uuid: String,
    index: u32,
    memory_used_mib: u64,
    memory_total_mib: u64,
    max_idle_mib: u64,
}

impl IdleGpuObservation {
    pub fn gpu_uuid(&self) -> &str {
        &self.gpu_uuid
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn memory_used_mib(&self) -> u64 {
        self.memory_used_mib
    }

    pub fn memory_total_mib(&self) -> u64 {
        self.memory_total_mib
    }

    pub fn max_idle_mib(&self) -> u64 {
        self.max_idle_mib
    }
}

/// Fixed, redacted failure categories. No query output, process ID, process
/// name, command detail, or operating-system error is retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuProbeFailureKind {
    InvalidConfiguration,
    Cancelled,
    DeadlineExceeded,
    QueryFailed,
    MalformedOutput,
    DeviceMismatch,
    ClaimantPresent,
    MemoryBusy,
}

/// A GPU observation failure containing only a fixed category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuProbeFailure {
    kind: GpuProbeFailureKind,
}

impl GpuProbeFailure {
    fn new(kind: GpuProbeFailureKind) -> Self {
        Self { kind }
    }

    pub fn kind(&self) -> GpuProbeFailureKind {
        self.kind
    }
}

impl fmt::Display for GpuProbeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            GpuProbeFailureKind::InvalidConfiguration => "GPU probe configuration is invalid",
            GpuProbeFailureKind::Cancelled => "GPU observation was cancelled",
            GpuProbeFailureKind::DeadlineExceeded => "GPU observation deadline expired",
            GpuProbeFailureKind::QueryFailed => "NVIDIA GPU query failed",
            GpuProbeFailureKind::MalformedOutput => "NVIDIA GPU query output is malformed",
            GpuProbeFailureKind::DeviceMismatch => "NVIDIA GPU query returned a different device",
            GpuProbeFailureKind::ClaimantPresent => "NVIDIA GPU has one or more compute claimants",
            GpuProbeFailureKind::MemoryBusy => {
                "NVIDIA GPU memory use exceeds the configured idle limit"
            }
        })
    }
}

impl Error for GpuProbeFailure {}

/// Read-only NVIDIA GPU probe using the fixed production executable.
pub struct NvidiaSmiGpuProbe {
    config: GpuProbeConfig,
    executable: PathBuf,
}

impl NvidiaSmiGpuProbe {
    /// Construct a production probe using `/usr/bin/nvidia-smi`.
    pub fn new(config: GpuProbeConfig) -> Self {
        Self {
            config,
            executable: PathBuf::from(NVIDIA_SMI_PATH),
        }
    }

    /// Take a bounded observation of the configured GPU.
    ///
    /// A successful result means both memory snapshots were within the
    /// configured allowance and the intervening compute-app query had no
    /// rows. The result is only point-in-time evidence: callers must separately
    /// own runtime cleanup and prevent direct admission/bypass before a switch.
    pub fn observe_idle(
        &self,
        context: &OperationContext<'_>,
        recovery_policy: RecoveryPolicy,
    ) -> Result<IdleGpuObservation, GpuProbeFailure> {
        check_active(context)?;

        let before_bytes = self.query_gpu(context, recovery_policy)?;
        check_active(context)?;
        let before = parse_gpu_snapshot(&before_bytes, &self.config)?;

        let apps_bytes = self.query_compute_apps(context, recovery_policy)?;
        check_active(context)?;
        let has_compute_claimants = parse_compute_apps(&apps_bytes, &self.config)?;

        let after_bytes = self.query_gpu(context, recovery_policy)?;
        check_active(context)?;
        let after = parse_gpu_snapshot(&after_bytes, &self.config)?;

        if has_compute_claimants {
            return Err(GpuProbeFailure::new(GpuProbeFailureKind::ClaimantPresent));
        }
        if before.memory_total_mib != after.memory_total_mib {
            return Err(GpuProbeFailure::new(GpuProbeFailureKind::DeviceMismatch));
        }
        for snapshot in [&before, &after] {
            if snapshot.memory_used_mib > self.config.max_idle_mib
                || snapshot.memory_total_mib <= self.config.max_idle_mib
                || snapshot.memory_used_mib > snapshot.memory_total_mib
            {
                return Err(GpuProbeFailure::new(GpuProbeFailureKind::MemoryBusy));
            }
        }
        check_active(context)?;

        Ok(IdleGpuObservation {
            gpu_uuid: self.config.gpu_uuid.clone(),
            index: after.index,
            memory_used_mib: after.memory_used_mib,
            memory_total_mib: after.memory_total_mib,
            max_idle_mib: self.config.max_idle_mib,
        })
    }

    fn query_gpu(
        &self,
        context: &OperationContext<'_>,
        recovery_policy: RecoveryPolicy,
    ) -> Result<Vec<u8>, GpuProbeFailure> {
        let mut command = Command::new(&self.executable);
        command
            .arg("--id")
            .arg(&self.config.gpu_uuid)
            .arg("--query-gpu=index,uuid,memory.used,memory.total")
            .arg("--format=csv,noheader,nounits");
        self.query(command, context, recovery_policy)
    }

    fn query_compute_apps(
        &self,
        context: &OperationContext<'_>,
        recovery_policy: RecoveryPolicy,
    ) -> Result<Vec<u8>, GpuProbeFailure> {
        let mut command = Command::new(&self.executable);
        command
            .arg("--id")
            .arg(&self.config.gpu_uuid)
            .arg("--query-compute-apps=gpu_uuid,pid")
            .arg("--format=csv,noheader,nounits");
        self.query(command, context, recovery_policy)
    }

    fn query(
        &self,
        command: Command,
        context: &OperationContext<'_>,
        recovery_policy: RecoveryPolicy,
    ) -> Result<Vec<u8>, GpuProbeFailure> {
        check_active(context)?;
        let limits = ProcessLimits::new(0, MAX_QUERY_BYTES)
            .map_err(|_| GpuProbeFailure::new(GpuProbeFailureKind::QueryFailed))?;
        let output = process::execute(command, None, limits, context, recovery_policy)
            .map_err(map_process_failure)?;
        if !output.stderr().is_empty() {
            return Err(GpuProbeFailure::new(GpuProbeFailureKind::QueryFailed));
        }
        Ok(output.stdout().to_vec())
    }

    #[cfg(test)]
    fn with_program(config: GpuProbeConfig, executable: impl AsRef<Path>) -> Self {
        Self {
            config,
            executable: executable.as_ref().to_path_buf(),
        }
    }
}

fn check_active(context: &OperationContext<'_>) -> Result<(), GpuProbeFailure> {
    if context.is_cancelled() {
        Err(GpuProbeFailure::new(GpuProbeFailureKind::Cancelled))
    } else if context.remaining().is_zero() {
        Err(GpuProbeFailure::new(GpuProbeFailureKind::DeadlineExceeded))
    } else {
        Ok(())
    }
}

fn map_process_failure(error: ProcessFailure) -> GpuProbeFailure {
    let kind = match error.kind() {
        ProcessFailureKind::Cancelled => GpuProbeFailureKind::Cancelled,
        ProcessFailureKind::DeadlineExceeded => GpuProbeFailureKind::DeadlineExceeded,
        _ => GpuProbeFailureKind::QueryFailed,
    };
    // A cleanup uncertainty remains a failed observation; no GPU ownership or
    // release conclusion is inferred from the local child disposition.
    GpuProbeFailure::new(kind)
}

#[derive(Clone, Copy)]
struct GpuSnapshot {
    index: u32,
    memory_used_mib: u64,
    memory_total_mib: u64,
}

fn parse_gpu_snapshot(
    bytes: &[u8],
    config: &GpuProbeConfig,
) -> Result<GpuSnapshot, GpuProbeFailure> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GpuProbeFailure::new(GpuProbeFailureKind::MalformedOutput))?;
    let mut row_count = 0usize;
    let mut parsed = None;

    for row in text.lines() {
        row_count += 1;
        if row_count > MAX_GPU_ROWS {
            return Err(GpuProbeFailure::new(GpuProbeFailureKind::MalformedOutput));
        }
        let mut fields = row.split(',');
        let index = numeric_field(fields.next())
            .and_then(parse_u32)
            .ok_or_else(malformed_output)?;
        let uuid = text_field(fields.next()).ok_or_else(malformed_output)?;
        let memory_used_mib = numeric_field(fields.next())
            .and_then(parse_u64)
            .ok_or_else(malformed_output)?;
        let memory_total_mib = numeric_field(fields.next())
            .and_then(parse_u64)
            .filter(|value| *value > 0)
            .ok_or_else(malformed_output)?;
        if memory_used_mib > memory_total_mib {
            return Err(malformed_output());
        }
        if fields.next().is_some() || !valid_gpu_uuid(uuid) {
            return Err(malformed_output());
        }
        if uuid != config.gpu_uuid || index != config.expected_index {
            return Err(GpuProbeFailure::new(GpuProbeFailureKind::DeviceMismatch));
        }
        parsed = Some(GpuSnapshot {
            index,
            memory_used_mib,
            memory_total_mib,
        });
    }

    if row_count != 1 {
        return Err(malformed_output());
    }
    parsed.ok_or_else(malformed_output)
}

/// Validate every compute-app row without retaining a PID or other row data.
/// An empty response is the only representation of no compute claimants.
fn parse_compute_apps(bytes: &[u8], config: &GpuProbeConfig) -> Result<bool, GpuProbeFailure> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| GpuProbeFailure::new(GpuProbeFailureKind::MalformedOutput))?;
    let mut row_count = 0usize;

    for row in text.lines() {
        row_count += 1;
        if row_count > MAX_COMPUTE_APP_ROWS {
            return Err(GpuProbeFailure::new(GpuProbeFailureKind::MalformedOutput));
        }
        let mut fields = row.split(',');
        let uuid = text_field(fields.next()).ok_or_else(malformed_output)?;
        let pid = numeric_field(fields.next())
            .and_then(parse_u32)
            .filter(|pid| *pid > 0)
            .ok_or_else(malformed_output)?;
        if fields.next().is_some() || !valid_gpu_uuid(uuid) {
            return Err(malformed_output());
        }
        if uuid != config.gpu_uuid {
            return Err(GpuProbeFailure::new(GpuProbeFailureKind::DeviceMismatch));
        }
        // Deliberately consume only a positive numeric check; never store or
        // report the PID.
        let _ = pid;
    }

    Ok(row_count > 0)
}

fn text_field(value: Option<&str>) -> Option<&str> {
    let value = value?.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("N/A") {
        None
    } else {
        Some(value)
    }
}

fn numeric_field(value: Option<&str>) -> Option<&str> {
    let value = text_field(value)?;
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit())
        .then_some(value)
}

fn parse_u32(value: &str) -> Option<u32> {
    value.parse().ok()
}

fn parse_u64(value: &str) -> Option<u64> {
    value.parse().ok()
}

fn valid_gpu_uuid(uuid: &str) -> bool {
    let Some(value) = uuid.strip_prefix("GPU-") else {
        return false;
    };
    let bytes = value.as_bytes();
    bytes.len() == 36
        && [8, 13, 18, 23]
            .into_iter()
            .all(|index| bytes[index] == b'-')
        && bytes.iter().enumerate().all(|(index, byte)| {
            if [8, 13, 18, 23].contains(&index) {
                *byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

fn malformed_output() -> GpuProbeFailure {
    GpuProbeFailure::new(GpuProbeFailureKind::MalformedOutput)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::control::runtime::{Cancellation, NeverCancel};
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";

    struct FakeProbe {
        directory: TempDir,
        probe: NvidiaSmiGpuProbe,
    }

    fn fake_probe(gpu_before: &str, apps: &str, gpu_after: &str) -> FakeProbe {
        fake_probe_with(gpu_before, apps, gpu_after, "")
    }

    fn fake_probe_with(
        gpu_before: &str,
        apps: &str,
        gpu_after: &str,
        gpu_delay: &str,
    ) -> FakeProbe {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("fake-nvidia-smi");
        let gpu_count = directory.path().join("gpu-count");
        let app_count = directory.path().join("app-count");
        fs::write(&gpu_count, "0\n").unwrap();
        fs::write(&app_count, "0\n").unwrap();
        let script = format!(
            "#!/bin/sh\ncase \"$*\" in\n  *'--query-gpu=index,uuid,memory.used,memory.total'*)\n    read count < {gpu_count} 2>/dev/null || count=0\n    count=$((count + 1))\n    printf '%s\\n' \"$count\" > {gpu_count}\n    {gpu_delay}\n    case \"$count\" in\n      1) printf '%s' {gpu_before} ;;\n      *) printf '%s' {gpu_after} ;;\n    esac\n    ;;\n  *'--query-compute-apps=gpu_uuid,pid'*)\n    read count < {app_count} 2>/dev/null || count=0\n    count=$((count + 1))\n    printf '%s\\n' \"$count\" > {app_count}\n    printf '%s' {apps}\n    ;;\n  *) exit 23 ;;\nesac\n",
            gpu_count = shell_quote(&gpu_count.to_string_lossy()),
            app_count = shell_quote(&app_count.to_string_lossy()),
            gpu_before = shell_quote(gpu_before),
            gpu_after = shell_quote(gpu_after),
            apps = shell_quote(apps),
            gpu_delay = gpu_delay,
        );
        let fixture_guard = super::super::process::executable_fixture_guard();
        fs::write(&executable, script).unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&executable, permissions).unwrap();
        drop(fixture_guard);
        let config = GpuProbeConfig::new(UUID, 0, 64).unwrap();
        let probe = NvidiaSmiGpuProbe::with_program(config, &executable);
        FakeProbe { directory, probe }
    }

    fn shell_quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\\''"))
    }

    fn snapshot(index: u32, uuid: &str, used: u64, total: u64) -> String {
        format!("{index},{uuid},{used},{total}\n")
    }

    fn context<'a>(timeout: Duration, cancellation: &'a dyn Cancellation) -> OperationContext<'a> {
        OperationContext::new(Instant::now() + timeout, cancellation)
    }

    fn recovery() -> RecoveryPolicy {
        RecoveryPolicy::new(Duration::from_secs(2)).unwrap()
    }

    fn assert_kind(error: GpuProbeFailure, expected: GpuProbeFailureKind) {
        assert_eq!(error.kind(), expected);
    }

    #[test]
    fn successful_observation_runs_both_snapshots_and_compute_query() {
        let fixture = fake_probe(
            &snapshot(0, UUID, 32, 8192),
            "",
            &snapshot(0, UUID, 48, 8192),
        );
        let result = fixture
            .probe
            .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
            .unwrap();
        assert_eq!(result.gpu_uuid(), UUID);
        assert_eq!(result.index(), 0);
        assert_eq!(result.memory_used_mib(), 48);
        assert_eq!(result.memory_total_mib(), 8192);
        assert_eq!(result.max_idle_mib(), 64);
        assert_eq!(
            fs::read_to_string(fixture.directory.path().join("gpu-count")).unwrap(),
            "2\n"
        );
        assert_eq!(
            fs::read_to_string(fixture.directory.path().join("app-count")).unwrap(),
            "1\n"
        );
    }

    #[test]
    fn malformed_n_a_and_missing_or_extra_rows_fail_closed() {
        for gpu_output in [
            "0,GPU-01234567-89ab-cdef-0123-456789abcdef,N/A,8192\n",
            "0,GPU-01234567-89ab-cdef-0123-456789abcdef,10,0\n",
            "0,GPU-01234567-89ab-cdef-0123-456789abcdef,10,8192,extra\n",
            "",
            "0,GPU-01234567-89ab-cdef-0123-456789abcdef,10,8192\n\n",
        ] {
            let fixture = fake_probe(gpu_output, "", gpu_output);
            let error = fixture
                .probe
                .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
                .unwrap_err();
            assert_kind(error, GpuProbeFailureKind::MalformedOutput);
        }
    }

    #[test]
    fn other_gpu_and_reordered_index_are_device_mismatches() {
        let other_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
        let cases = [snapshot(0, other_uuid, 0, 8192), snapshot(1, UUID, 0, 8192)];
        for gpu_output in cases {
            let fixture = fake_probe(&gpu_output, "", &gpu_output);
            let error = fixture
                .probe
                .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
                .unwrap_err();
            assert_kind(error, GpuProbeFailureKind::DeviceMismatch);
        }
    }

    #[test]
    fn over_baseline_or_baseline_at_capacity_is_memory_busy() {
        let cases = [snapshot(0, UUID, 65, 8192), snapshot(0, UUID, 0, 64)];
        for gpu_output in cases {
            let fixture = fake_probe(&gpu_output, "", &gpu_output);
            let error = fixture
                .probe
                .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
                .unwrap_err();
            assert_kind(error, GpuProbeFailureKind::MemoryBusy);
        }
    }

    #[test]
    fn used_above_total_is_malformed_and_changing_capacity_is_device_mismatch() {
        let malformed = snapshot(0, UUID, 8193, 8192);
        let fixture = fake_probe(&malformed, "", &malformed);
        let error = fixture
            .probe
            .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
            .unwrap_err();
        assert_kind(error, GpuProbeFailureKind::MalformedOutput);

        let fixture = fake_probe(&snapshot(0, UUID, 0, 8192), "", &snapshot(0, UUID, 0, 8191));
        let error = fixture
            .probe
            .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
            .unwrap_err();
        assert_kind(error, GpuProbeFailureKind::DeviceMismatch);
    }

    #[test]
    fn compute_claimants_are_reported_without_retaining_the_pid() {
        let fixture = fake_probe(
            &snapshot(0, UUID, 0, 8192),
            &format!("{UUID},123456\n"),
            &snapshot(0, UUID, 0, 8192),
        );
        let error = fixture
            .probe
            .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
            .unwrap_err();
        assert_kind(error, GpuProbeFailureKind::ClaimantPresent);
        assert!(!error.to_string().contains("123456"));
        assert!(!format!("{error:?}").contains("123456"));
        assert_eq!(
            fs::read_to_string(fixture.directory.path().join("gpu-count")).unwrap(),
            "2\n"
        );
    }

    #[test]
    fn malformed_or_foreign_compute_rows_fail_closed() {
        for (apps, expected) in [
            (
                format!("{UUID},N/A\n"),
                GpuProbeFailureKind::MalformedOutput,
            ),
            (
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,42\n".to_owned(),
                GpuProbeFailureKind::DeviceMismatch,
            ),
        ] {
            let fixture = fake_probe(
                &snapshot(0, UUID, 0, 8192),
                &apps,
                &snapshot(0, UUID, 0, 8192),
            );
            let error = fixture
                .probe
                .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
                .unwrap_err();
            assert_kind(error, expected);
        }
    }

    #[test]
    fn oversized_output_and_nonzero_query_are_fixed_query_failures() {
        let large_gpu = format!("{}\n", "0".repeat(MAX_QUERY_BYTES + 1));
        let fixture = fake_probe(&large_gpu, "", "");
        let error = fixture
            .probe
            .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
            .unwrap_err();
        assert_kind(error, GpuProbeFailureKind::QueryFailed);

        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("fake-nvidia-smi");
        fs::write(
            &executable,
            "#!/bin/sh\nprintf '%s' 'private process details' >&2\nexit 7\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&executable, permissions).unwrap();
        let probe =
            NvidiaSmiGpuProbe::with_program(GpuProbeConfig::new(UUID, 0, 64).unwrap(), &executable);
        let error = probe
            .observe_idle(&context(Duration::from_secs(3), &NeverCancel), recovery())
            .unwrap_err();
        assert_kind(error, GpuProbeFailureKind::QueryFailed);
        assert!(!error.to_string().contains("private process details"));
    }

    #[test]
    fn timeout_and_cancellation_fail_closed() {
        let fixture = fake_probe_with(
            &snapshot(0, UUID, 0, 8192),
            "",
            &snapshot(0, UUID, 0, 8192),
            "sleep 2",
        );
        let error = fixture
            .probe
            .observe_idle(
                &context(Duration::from_millis(100), &NeverCancel),
                recovery(),
            )
            .unwrap_err();
        assert_kind(error, GpuProbeFailureKind::DeadlineExceeded);

        struct SharedCancellation(Arc<AtomicBool>);
        impl Cancellation for SharedCancellation {
            fn is_cancelled(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
        }
        let fixture = fake_probe_with(
            &snapshot(0, UUID, 0, 8192),
            "",
            &snapshot(0, UUID, 0, 8192),
            "sleep 2",
        );
        let flag = Arc::new(AtomicBool::new(false));
        let cancellation = SharedCancellation(flag.clone());
        let signal = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            flag.store(true, Ordering::Release);
        });
        let error = fixture
            .probe
            .observe_idle(&context(Duration::from_secs(3), &cancellation), recovery())
            .unwrap_err();
        signal.join().unwrap();
        assert_kind(error, GpuProbeFailureKind::Cancelled);
    }

    #[test]
    fn explicit_configuration_rejects_non_gpu_uuid_and_has_no_default_baseline() {
        assert_eq!(
            GpuProbeConfig::new("0", 0, 64).unwrap_err().kind(),
            GpuProbeFailureKind::InvalidConfiguration
        );
        assert_eq!(GpuProbeConfig::new(UUID, 0, 0).unwrap().max_idle_mib(), 0);
    }
}
