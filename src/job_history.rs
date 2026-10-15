use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::process::Command;

use anyhow::{Context, Result};

use crate::{
    script_store::{command_for_script, resolve_script_in},
    state_store::{
        self, JobHistoryLoad as PersistedLoad, JobMetadata, JobRecord as PersistedJob, JOBS_MAX,
        JOB_INTERRUPTED_EXIT, JOB_RUNNING_EXIT,
    },
};

/// Safe identifiers for a cross-toolkit job. This API deliberately has no
/// argv, command, or environment fields; those remain transient supervisor
/// inputs and are never copied into the persistent job contract.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrossToolkitJobStart {
    pub toolkit: String,
    pub profile: Option<String>,
    pub task: Option<String>,
    pub request_id: Option<String>,
}

/// Bounded, typed updates supplied as a dispatched job progresses.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrossToolkitJobUpdate {
    pub backend_id: Option<String>,
    pub result_state: Option<String>,
    /// When provided, replaces the complete artifact-reference list.
    pub artifact_refs: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct JobRecord {
    pub id: u64,
    pub name: String,
    pub kind: String,
    pub status: String,
    pub command: Vec<String>,
    pub started_label: String,
    pub elapsed_label: String,
    pub exit_label: String,
    pub mode: String,
    pub script_path: String,
    pub exit_code: Option<i32>,
    pub metadata: JobMetadata,
    started_at: Option<Instant>,
}

impl JobRecord {
    pub fn elapsed_seconds(&self) -> Option<u64> {
        self.started_at.map(|started| started.elapsed().as_secs())
    }

    pub fn is_running(&self) -> bool {
        self.exit_label == JOB_RUNNING_EXIT
    }

    fn persisted(&self) -> PersistedJob {
        PersistedJob {
            name: self.name.clone(),
            started: self.started_label.clone(),
            elapsed: self.elapsed_label.clone(),
            exit: self.exit_label.clone(),
            mode: self.mode.clone(),
            script_path: self.script_path.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobHistory {
    jobs: Vec<JobRecord>,
    next_id: u64,
    persistence_enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadNotice {
    pub reconciled_running: usize,
    pub truncated_rows: usize,
    pub malformed_rows: usize,
    pub normalization_error: Option<String>,
}

impl LoadNotice {
    pub fn is_empty(&self) -> bool {
        self.reconciled_running == 0
            && self.truncated_rows == 0
            && self.malformed_rows == 0
            && self.normalization_error.is_none()
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.reconciled_running > 0 {
            parts.push(format!(
                "reconciled {} interrupted job(s)",
                self.reconciled_running
            ));
        }
        if self.truncated_rows > 0 {
            parts.push(format!("trimmed {} old job(s)", self.truncated_rows));
        }
        if self.malformed_rows > 0 {
            parts.push(format!("ignored {} malformed job(s)", self.malformed_rows));
        }
        if let Some(error) = &self.normalization_error {
            parts.push(format!("normalization save failed: {error}"));
        }
        parts.join("; ")
    }
}

impl Default for JobHistory {
    fn default() -> Self {
        Self {
            jobs: Vec::new(),
            next_id: 1,
            persistence_enabled: true,
        }
    }
}

impl JobHistory {
    pub fn load(repository_root: &Path) -> Result<(Self, LoadNotice)> {
        let data_root = state_store::data_root()?;
        Self::load_in(&data_root, repository_root)
    }

    pub fn load_in(data_root: &Path, repository_root: &Path) -> Result<(Self, LoadNotice)> {
        let loaded = state_store::load_jobs_in(data_root)?;
        Self::load_from_loaded(data_root, repository_root, loaded)
    }

    fn load_from_loaded(
        data_root: &Path,
        repository_root: &Path,
        loaded: PersistedLoad,
    ) -> Result<(Self, LoadNotice)> {
        let mut notice = notice_for(&loaded);
        if !notice.is_empty() {
            if let Err(error) = state_store::save_jobs_in(data_root, &loaded.jobs) {
                notice.normalization_error = Some(format!("{error:#}"));
            }
        }
        let jobs = loaded
            .jobs
            .into_iter()
            .enumerate()
            .rev()
            .map(|(index, job)| hydrate_job(index as u64 + 1, job, repository_root))
            .collect::<Vec<_>>();
        let next_id = jobs.iter().map(|job| job.id).max().unwrap_or(0) + 1;
        Ok((
            Self {
                jobs,
                next_id,
                persistence_enabled: true,
            },
            notice,
        ))
    }

    pub fn records(&self) -> &[JobRecord] {
        &self.jobs
    }

    pub fn get(&self, index: usize) -> Option<&JobRecord> {
        self.jobs.get(index)
    }

    pub fn unavailable() -> Self {
        Self {
            persistence_enabled: false,
            ..Self::default()
        }
    }

    pub fn persistence_enabled(&self) -> bool {
        self.persistence_enabled
    }

    pub fn begin(
        &mut self,
        name: String,
        kind: String,
        command: Vec<String>,
        mode: String,
        script_path: String,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.jobs.insert(
            0,
            JobRecord {
                id,
                name,
                kind,
                status: "running".into(),
                command,
                started_label: current_clock_label(),
                elapsed_label: "-".into(),
                exit_label: JOB_RUNNING_EXIT.into(),
                mode,
                script_path,
                exit_code: None,
                metadata: JobMetadata::default(),
                started_at: Some(Instant::now()),
            },
        );
        self.jobs.truncate(JOBS_MAX);
        id
    }

    pub fn finish(&mut self, id: u64, exit_code: i32) -> Option<&JobRecord> {
        let job = self.jobs.iter_mut().find(|job| job.id == id)?;
        let elapsed = job
            .started_at
            .map(|started| started.elapsed().as_secs())
            .unwrap_or_default();
        job.elapsed_label = format_elapsed(elapsed);
        job.exit_label = exit_code.to_string();
        job.exit_code = Some(exit_code);
        job.status = if exit_code == 0 {
            "done".into()
        } else {
            "failed".into()
        };
        if job.metadata.toolkit.is_some() {
            job.metadata.result_state = Some(if exit_code == 0 {
                "succeeded".into()
            } else {
                "failed".into()
            });
            job.metadata.finished_at = Some(state_store::utc_timestamp_now());
        }
        job.started_at = None;
        Some(job)
    }

    /// Record accepted cross-toolkit work as queued. Callers must persist this
    /// history before dispatching; update the entry to `running` when execution
    /// actually starts. The row uses the same bounded history/store as all jobs.
    pub fn begin_cross_toolkit(&mut self, request: CrossToolkitJobStart) -> Result<u64> {
        let timestamp = state_store::utc_timestamp_now();
        let metadata = JobMetadata {
            toolkit: Some(request.toolkit),
            profile: request.profile,
            task: request.task,
            request_id: request.request_id,
            submitted_at: Some(timestamp),
            result_state: Some("queued".into()),
            ..JobMetadata::default()
        };
        let errors = state_store::validate_job_metadata(&metadata);
        anyhow::ensure!(
            errors.is_empty(),
            "invalid cross-toolkit job metadata: {}",
            errors.join("; ")
        );

        let toolkit = metadata.toolkit.as_deref().unwrap_or("toolkit");
        let name = match metadata.task.as_deref() {
            Some(task) => format!("{toolkit}:{task}"),
            None => format!("{toolkit}:job"),
        };
        let id = self.begin(
            name,
            "cross-toolkit".into(),
            Vec::new(),
            "cross-toolkit".into(),
            String::new(),
        );
        if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
            job.metadata = metadata;
            job.status = "queued".into();
            job.started_at = None;
        }
        Ok(id)
    }

    /// Update backend/result/artifact fields on an existing cross-toolkit
    /// entry. Metadata is validated before mutating the live history.
    pub fn update_cross_toolkit(
        &mut self,
        id: u64,
        update: CrossToolkitJobUpdate,
    ) -> Result<&JobRecord> {
        let job_index = self
            .jobs
            .iter()
            .position(|job| job.id == id)
            .ok_or_else(|| anyhow::anyhow!("unknown job id {id}"))?;
        let mut metadata = self.jobs[job_index].metadata.clone();
        let previous_state = metadata.result_state.clone();
        anyhow::ensure!(
            metadata.toolkit.is_some(),
            "job {id} is not a cross-toolkit job"
        );
        if let Some(backend_id) = update.backend_id {
            metadata.backend_id = Some(backend_id);
        }
        if let Some(result_state) = update.result_state {
            let previous = metadata.result_state.as_deref().unwrap_or("running");
            anyhow::ensure!(
                valid_state_transition(previous, &result_state),
                "invalid job result transition from {previous} to {result_state}"
            );
            metadata.result_state = Some(result_state);
        }
        if let Some(artifact_refs) = update.artifact_refs {
            metadata.artifact_refs = artifact_refs;
        }
        let entering_running = previous_state.as_deref() == Some("queued")
            && metadata.result_state.as_deref() == Some("running");
        if entering_running {
            metadata.started_at = Some(state_store::utc_timestamp_now());
        }
        let errors = state_store::validate_job_metadata(&metadata);
        anyhow::ensure!(
            errors.is_empty(),
            "invalid cross-toolkit job metadata: {}",
            errors.join("; ")
        );

        let terminal_transition = metadata
            .result_state
            .as_deref()
            .is_some_and(is_terminal_state)
            && !self.jobs[job_index]
                .metadata
                .result_state
                .as_deref()
                .is_some_and(is_terminal_state);
        if terminal_transition {
            metadata.finished_at = Some(state_store::utc_timestamp_now());
        }
        let job = &mut self.jobs[job_index];
        if terminal_transition
            && !job
                .metadata
                .result_state
                .as_deref()
                .is_some_and(is_terminal_state)
        {
            finish_with_result_state(job, metadata.result_state.as_deref().unwrap());
        } else if metadata.result_state.as_deref() == Some("queued") {
            job.status = "queued".into();
        } else if metadata.result_state.as_deref() == Some("running") {
            job.status = "running".into();
            if entering_running {
                job.started_label = current_clock_label();
                job.started_at = Some(Instant::now());
            }
        }
        job.metadata = metadata;
        Ok(job)
    }

    /// Persist the one-shot ComfyUI prompt candidate for a running media job.
    /// The candidate is intent only: it is kept separate from `backend_id`,
    /// which is written only after a positive native acceptance response.
    pub fn record_comfy_dispatch_candidate(
        &mut self,
        id: u64,
        candidate_id: String,
    ) -> Result<&JobRecord> {
        self.record_comfy_dispatch_candidate_inner(id, candidate_id)
    }

    fn record_comfy_dispatch_candidate_inner(
        &mut self,
        id: u64,
        candidate_id: String,
    ) -> Result<&JobRecord> {
        let job_index = self
            .jobs
            .iter()
            .position(|job| job.id == id)
            .ok_or_else(|| anyhow::anyhow!("unknown job id {id}"))?;
        let job = &self.jobs[job_index];
        anyhow::ensure!(
            job.kind == "cross-toolkit"
                && job.status == "running"
                && job.metadata.toolkit.as_deref() == Some("l2m2")
                && job.metadata.result_state.as_deref() == Some("running")
                && job.metadata.profile.is_some()
                && job.metadata.task.is_some()
                && job.metadata.request_id.is_some()
                && job.metadata.backend_id.is_none()
                && job.metadata.comfy_service_generation.is_none(),
            "job {id} is not an unacknowledged running Comfy dispatch"
        );
        anyhow::ensure!(
            job.metadata.comfy_dispatch_candidate.is_none(),
            "job {id} already has a Comfy dispatch candidate"
        );
        anyhow::ensure!(
            !self.jobs.iter().enumerate().any(|(index, existing)| {
                index != job_index
                    && existing.metadata.comfy_dispatch_candidate.as_deref()
                        == Some(candidate_id.as_str())
            }),
            "Comfy dispatch candidate already belongs to another retained job"
        );

        let mut metadata = job.metadata.clone();
        metadata.comfy_dispatch_candidate = Some(candidate_id);
        let errors = state_store::validate_job_metadata(&metadata);
        anyhow::ensure!(
            errors.is_empty(),
            "invalid Comfy dispatch candidate metadata: {}",
            errors.join("; ")
        );
        self.jobs[job_index].metadata = metadata;
        Ok(&self.jobs[job_index])
    }

    pub fn clear_for_recovery(&mut self) {
        self.jobs.clear();
        self.persistence_enabled = true;
    }

    pub fn persist(&self) -> Result<()> {
        let data_root = state_store::data_root()?;
        self.persist_in(&data_root)
    }

    pub fn persist_in(&self, data_root: &Path) -> Result<()> {
        anyhow::ensure!(
            self.persistence_enabled,
            "job history persistence is disabled until the unreadable history is explicitly cleared"
        );
        let oldest_first = self
            .jobs
            .iter()
            .rev()
            .map(JobRecord::persisted)
            .collect::<Vec<_>>();
        state_store::save_jobs_in(data_root, &oldest_first)
    }
}

fn notice_for(loaded: &PersistedLoad) -> LoadNotice {
    LoadNotice {
        reconciled_running: loaded.reconciled_running,
        truncated_rows: loaded.truncated_rows,
        malformed_rows: loaded.issues.len(),
        normalization_error: None,
    }
}

fn hydrate_job(id: u64, persisted: PersistedJob, repository_root: &Path) -> JobRecord {
    let (kind, command) = retry_command(&persisted, repository_root);
    let exit_code = persisted.exit.parse::<i32>().ok();
    let status = match persisted.metadata.result_state.as_deref() {
        Some("queued") => "queued",
        Some("running") => "running",
        Some("succeeded") => "done",
        Some("failed") => "failed",
        Some("cancelled") => "cancelled",
        Some("interrupted") => "interrupted",
        _ => match persisted.exit.as_str() {
            JOB_INTERRUPTED_EXIT => "interrupted",
            "0" => "done",
            _ if exit_code.is_some() => "failed",
            _ => "finished",
        },
    }
    .to_owned();
    JobRecord {
        id,
        name: persisted.name,
        kind,
        status,
        command,
        started_label: persisted.started,
        elapsed_label: persisted.elapsed,
        exit_label: persisted.exit,
        mode: persisted.mode,
        script_path: persisted.script_path,
        exit_code,
        metadata: persisted.metadata,
        started_at: None,
    }
}

fn is_terminal_state(state: &str) -> bool {
    matches!(state, "succeeded" | "failed" | "cancelled" | "interrupted")
}

fn valid_state_transition(previous: &str, next: &str) -> bool {
    if previous == next {
        return true;
    }
    match previous {
        "queued" => next == "running" || is_terminal_state(next),
        "running" => is_terminal_state(next),
        _ => false,
    }
}

fn finish_with_result_state(job: &mut JobRecord, result_state: &str) {
    let elapsed = job
        .started_at
        .map(|started| started.elapsed().as_secs())
        .unwrap_or_default();
    job.elapsed_label = format_elapsed(elapsed);
    match result_state {
        "succeeded" => {
            job.exit_label = "0".into();
            job.exit_code = Some(0);
            job.status = "done".into();
        }
        "failed" => {
            job.exit_label = "1".into();
            job.exit_code = Some(1);
            job.status = "failed".into();
        }
        "cancelled" => {
            job.exit_label = "cancelled".into();
            job.exit_code = None;
            job.status = "cancelled".into();
        }
        "interrupted" => {
            job.exit_label = JOB_INTERRUPTED_EXIT.into();
            job.exit_code = None;
            job.status = "interrupted".into();
        }
        _ => return,
    }
    job.started_at = None;
}

fn retry_command(persisted: &PersistedJob, repository_root: &Path) -> (String, Vec<String>) {
    match persisted.mode.as_str() {
        "run" if !persisted.script_path.trim().is_empty() => (
            "model-load".into(),
            vec![
                "llama-swap".into(),
                "load".into(),
                persisted.script_path.clone(),
            ],
        ),
        "bench" | "maintenance" if !persisted.script_path.trim().is_empty() => {
            let mode = persisted.mode.clone();
            let path = PathBuf::from(&persisted.script_path);
            let expected_directory = repository_root.join(if mode == "bench" {
                "bench-models"
            } else {
                "maintenance"
            });
            let expected_directory =
                fs::canonicalize(&expected_directory).unwrap_or(expected_directory);
            let safe_path = resolve_script_in(&path, repository_root)
                .ok()
                .filter(|resolved| resolved.is_file() && resolved.starts_with(expected_directory));
            match safe_path {
                Some(path) => (mode, command_for_script(path, &[])),
                None => (mode, Vec::new()),
            }
        }
        mode if !mode.is_empty() => (mode.to_owned(), Vec::new()),
        _ => ("legacy".into(), Vec::new()),
    }
}

fn clock_label(now: SystemTime) -> String {
    let seconds = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() % 86_400;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}

fn current_clock_label() -> String {
    #[cfg(unix)]
    {
        if let Ok(output) = Command::new("date").arg("+%H:%M:%S").output() {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if output.status.success()
                && value.len() == 8
                && value.chars().enumerate().all(|(index, character)| {
                    matches!(index, 2 | 5) && character == ':'
                        || !matches!(index, 2 | 5) && character.is_ascii_digit()
                })
            {
                return value;
            }
        }
    }
    clock_label(SystemTime::now())
}

fn format_elapsed(seconds: u64) -> String {
    if seconds < 120 {
        format!("{seconds}s")
    } else {
        format!("{:.1}m", seconds as f64 / 60.0)
    }
}

pub fn persist_with_context(history: &JobHistory, context: &str) -> Result<()> {
    history
        .persist()
        .with_context(|| format!("persist job history after {context}"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;
    use crate::state_store::{
        ComfyManagerGenerationReference, ComfyServiceGenerationReference,
        ComfyServiceIntentReference, ComfySupervisorCoordinatorReference,
    };

    fn comfy_generation_reference() -> ComfyServiceGenerationReference {
        let unit_name = "l3ms-test-coordinator-0123456789abcdef0123456789abcdef.service".to_owned();
        ComfyServiceGenerationReference {
            schema_version: 1,
            intent: ComfyServiceIntentReference {
                coordinator_id: "test-coordinator".to_owned(),
                unit_name: unit_name.clone(),
                launch_identity: "b".repeat(64),
                stop_timeout_millis: 30_000,
                runtime_lane: "media".to_owned(),
                runtime_name: "ComfyUI".to_owned(),
                launch_role: "isolated_engine_supervisor".to_owned(),
                supervisor_coordinator: ComfySupervisorCoordinatorReference {
                    pid: 123,
                    uid: 1000,
                    start_time_ticks: 456,
                    network_namespace_device: 7,
                    network_namespace_inode: 4_026_531_993,
                },
            },
            generation: ComfyManagerGenerationReference {
                invocation_id: "a".repeat(32),
                cgroup: format!("/user.slice/{unit_name}"),
                main_pid: 2345,
            },
        }
    }

    #[test]
    fn loads_newest_first_and_reconciles_running_rows() {
        let data = TempDir::new().unwrap();
        let repository = repository_fixture();
        state_store::save_jobs_in(
            data.path(),
            &[
                persisted("old", "0", "", ""),
                persisted("new", JOB_RUNNING_EXIT, "run", "model-a"),
            ],
        )
        .unwrap();

        let (history, notice) = JobHistory::load_in(data.path(), repository.path()).unwrap();
        assert_eq!(history.records()[0].name, "new");
        assert_eq!(history.records()[0].status, "interrupted");
        assert_eq!(history.records()[0].command[2], "model-a");
        assert_eq!(notice.reconciled_running, 1);
        let normalized = state_store::load_jobs_in(data.path()).unwrap();
        assert_eq!(normalized.reconciled_running, 0);
    }

    #[test]
    fn startup_does_not_replay_a_persisted_queued_cross_toolkit_job() {
        let data = TempDir::new().unwrap();
        let mut queued = persisted("queued", JOB_RUNNING_EXIT, "cross-toolkit", "");
        queued.metadata = JobMetadata {
            toolkit: Some("l2m2".into()),
            submitted_at: Some("2026-09-28T15:04:05Z".into()),
            result_state: Some("queued".into()),
            ..JobMetadata::default()
        };
        state_store::save_jobs_in(data.path(), &[queued]).unwrap();

        let (history, notice) =
            JobHistory::load_in(data.path(), repository_fixture().path()).unwrap();
        assert_eq!(notice.reconciled_running, 1);
        assert_eq!(history.records()[0].status, "interrupted");
        assert_eq!(
            history.records()[0].metadata.result_state.as_deref(),
            Some("interrupted")
        );
        assert!(history.records()[0].metadata.finished_at.is_some());
    }

    #[test]
    fn begin_finish_and_persist_keep_legacy_order_and_shape() {
        let data = TempDir::new().unwrap();
        let mut history = JobHistory::default();
        let first = history.begin(
            "first".into(),
            "bench".into(),
            vec!["bash".into(), "/repo/bench-models/bench-a.sh".into()],
            "bench".into(),
            "/repo/bench-models/bench-a.sh".into(),
        );
        history.finish(first, 0);
        history.begin(
            "second".into(),
            "download".into(),
            vec!["download".into()],
            "download".into(),
            String::new(),
        );
        history.persist_in(data.path()).unwrap();

        let persisted = state_store::load_jobs_in(data.path()).unwrap();
        assert_eq!(persisted.jobs[0].name, "first");
        assert_eq!(persisted.jobs[0].exit, "0");
        assert_eq!(persisted.jobs[1].name, "second");
        assert_eq!(persisted.jobs[1].exit, JOB_INTERRUPTED_EXIT);
    }

    #[test]
    fn cross_toolkit_jobs_use_existing_history_and_persist_only_contract_metadata() {
        let data = TempDir::new().unwrap();
        let mut history = JobHistory::default();
        let id = history
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: "l2m2".into(),
                profile: Some("whisper-large".into()),
                task: Some("transcribe-42".into()),
                request_id: Some("req-123".into()),
            })
            .unwrap();
        let record = history.records().iter().find(|job| job.id == id).unwrap();
        assert!(record.command.is_empty());
        assert!(record.script_path.is_empty());
        assert_eq!(record.status, "queued");
        assert_eq!(record.metadata.result_state.as_deref(), Some("queued"));
        assert!(record
            .metadata
            .submitted_at
            .as_deref()
            .is_some_and(state_store::is_utc_timestamp));
        assert!(record.metadata.started_at.is_none());
        assert!(record.started_at.is_none());
        history.persist_in(data.path()).unwrap();
        let queued_on_disk: Vec<PersistedJob> =
            serde_json::from_slice(&fs::read(state_store::jobs_path_in(data.path())).unwrap())
                .unwrap();
        assert_eq!(
            queued_on_disk[0].metadata.result_state.as_deref(),
            Some("queued")
        );
        assert!(queued_on_disk[0].metadata.started_at.is_none());

        history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    backend_id: Some("llama.cpp:gpu0".into()),
                    artifact_refs: Some(vec!["artifacts/transcript.json".into()]),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();
        history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    result_state: Some("running".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();
        assert_eq!(history.records()[0].status, "running");
        assert!(history.records()[0].started_at.is_some());
        assert!(history.records()[0]
            .metadata
            .started_at
            .as_deref()
            .is_some_and(state_store::is_utc_timestamp));

        history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    result_state: Some("succeeded".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();
        history.persist_in(data.path()).unwrap();

        let bytes = fs::read(state_store::jobs_path_in(data.path())).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(!text.contains("argv"));
        assert!(!text.contains("environment"));
        assert!(!text.contains("--api-key"));
        let persisted = state_store::load_jobs_in(data.path()).unwrap();
        let job = &persisted.jobs[0];
        assert_eq!(job.metadata.toolkit.as_deref(), Some("l2m2"));
        assert_eq!(job.metadata.backend_id.as_deref(), Some("llama.cpp:gpu0"));
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        assert!(job
            .metadata
            .finished_at
            .as_deref()
            .is_some_and(state_store::is_utc_timestamp));
        assert!(job.script_path.is_empty());
    }

    #[test]
    fn comfy_candidate_persists_as_intent_and_restart_never_treats_it_as_acknowledgment() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let data = TempDir::new().unwrap();
        let mut history = JobHistory::default();
        let id = running_comfy_job(&mut history, "req-comfy-1");

        history
            .record_comfy_dispatch_candidate(id, CANDIDATE.into())
            .unwrap();
        let row = history.records().iter().find(|job| job.id == id).unwrap();
        assert_eq!(
            row.metadata.comfy_dispatch_candidate.as_deref(),
            Some(CANDIDATE)
        );
        assert!(row.metadata.backend_id.is_none());
        assert!(row.metadata.comfy_service_generation.is_none());
        history.persist_in(data.path()).unwrap();

        let raw = fs::read_to_string(state_store::jobs_path_in(data.path())).unwrap();
        assert!(raw.contains("\"comfy_dispatch_candidate\""));
        assert!(raw.contains(CANDIDATE));
        assert!(!raw.contains("comfy_service_generation"));
        assert!(!raw.contains("workflow-payload-secret"));
        assert!(!raw.contains("backend_id"));

        let (restarted, notice) =
            JobHistory::load_in(data.path(), repository_fixture().path()).unwrap();
        assert_eq!(notice.reconciled_running, 1);
        let row = restarted.records().iter().find(|job| job.id == id).unwrap();
        assert_eq!(row.status, "interrupted");
        assert_eq!(row.metadata.result_state.as_deref(), Some("interrupted"));
        assert_eq!(
            row.metadata.comfy_dispatch_candidate.as_deref(),
            Some(CANDIDATE)
        );
        assert!(row.metadata.backend_id.is_none());
        assert!(row.metadata.comfy_service_generation.is_none());
    }

    #[test]
    fn persisted_comfy_generation_metadata_remains_readable_after_restart() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let data = TempDir::new().unwrap();
        let generation = comfy_generation_reference();
        let persisted_row = serde_json::json!([{
            "name": "managed-comfy",
            "started": "12:34:56",
            "elapsed": "-",
            "exit": JOB_RUNNING_EXIT,
            "mode": "cross-toolkit",
            "script_path": "",
            "toolkit": "l2m2",
            "profile": "fixture-profile",
            "task": "image",
            "request_id": "req-managed-comfy",
            "result_state": "running",
            "comfy_dispatch_candidate": CANDIDATE,
            "comfy_service_generation": serde_json::to_value(&generation).unwrap()
        }]);
        fs::write(
            state_store::jobs_path_in(data.path()),
            serde_json::to_vec(&persisted_row).unwrap(),
        )
        .unwrap();

        let (restarted, notice) =
            JobHistory::load_in(data.path(), repository_fixture().path()).unwrap();
        assert_eq!(notice.reconciled_running, 1);
        let row = restarted.records().first().unwrap();
        assert_eq!(row.status, "interrupted");
        assert_eq!(row.metadata.result_state.as_deref(), Some("interrupted"));
        assert_eq!(
            row.metadata.comfy_dispatch_candidate.as_deref(),
            Some(CANDIDATE)
        );
        assert_eq!(row.metadata.comfy_service_generation, Some(generation));
        assert!(row.metadata.backend_id.is_none());
    }

    #[test]
    fn comfy_candidate_is_immutable_unique_and_scoped_to_its_running_media_job() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let mut history = JobHistory::default();
        let first = running_comfy_job(&mut history, "req-comfy-1");
        let second = running_comfy_job(&mut history, "req-comfy-2");

        history
            .record_comfy_dispatch_candidate(first, CANDIDATE.into())
            .unwrap();
        assert!(history
            .record_comfy_dispatch_candidate(first, CANDIDATE.into())
            .is_err());
        assert!(history
            .record_comfy_dispatch_candidate(second, CANDIDATE.into())
            .is_err());
        assert!(history
            .records()
            .iter()
            .find(|job| job.id == second)
            .unwrap()
            .metadata
            .comfy_dispatch_candidate
            .is_none());

        let unrelated = history.begin(
            "ordinary job".into(),
            "bench".into(),
            Vec::new(),
            "bench".into(),
            String::new(),
        );
        assert!(history
            .record_comfy_dispatch_candidate(unrelated, CANDIDATE.into())
            .is_err());
    }

    #[test]
    fn comfy_candidate_rejects_noncanonical_uuid_and_never_overwrites_backend_id() {
        let mut history = JobHistory::default();
        let id = running_comfy_job(&mut history, "req-comfy-1");
        assert!(history
            .record_comfy_dispatch_candidate(id, "123E4567-E89B-12D3-A456-426614174000".into())
            .is_err());
        assert!(history.records()[0]
            .metadata
            .comfy_dispatch_candidate
            .is_none());

        history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    backend_id: Some("already-acknowledged".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();
        assert!(history
            .record_comfy_dispatch_candidate(id, "123e4567-e89b-12d3-a456-426614174000".into())
            .is_err());
    }

    #[test]
    fn cross_toolkit_updates_validate_before_mutation_and_keep_terminal_state() {
        let mut history = JobHistory::default();
        let id = history
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: "l2m2".into(),
                ..CrossToolkitJobStart::default()
            })
            .unwrap();
        let before = history.records()[0].metadata.clone();
        let invalid = history.update_cross_toolkit(
            id,
            CrossToolkitJobUpdate {
                backend_id: Some("Bearer top-secret".into()),
                ..CrossToolkitJobUpdate::default()
            },
        );
        assert!(invalid.is_err());
        assert_eq!(history.records()[0].metadata, before);

        history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    result_state: Some("cancelled".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();
        assert_eq!(history.records()[0].status, "cancelled");
        assert!(history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    result_state: Some("running".into()),
                    ..CrossToolkitJobUpdate::default()
                }
            )
            .is_err());
    }

    #[test]
    fn existing_finish_api_completes_cross_toolkit_metadata() {
        let mut history = JobHistory::default();
        let id = history
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: "l2m2".into(),
                ..CrossToolkitJobStart::default()
            })
            .unwrap();

        history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    result_state: Some("running".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();
        let finished = history.finish(id, 0).unwrap();
        assert_eq!(finished.metadata.result_state.as_deref(), Some("succeeded"));
        assert!(finished
            .metadata
            .finished_at
            .as_deref()
            .is_some_and(state_store::is_utc_timestamp));
    }

    #[test]
    fn only_reconstructs_contained_scripts_in_the_expected_directory() {
        let repository = repository_fixture();
        let bench = repository.path().join("bench-models/bench-a.sh");
        fs::write(&bench, "#!/bin/sh\n").unwrap();
        let good = persisted("good", "0", "bench", &bench.to_string_lossy());
        let (kind, command) = retry_command(&good, repository.path());
        assert_eq!(kind, "bench");
        let canonical_bench = fs::canonicalize(&bench).unwrap();
        assert_eq!(command, ["bash", canonical_bench.to_str().unwrap()]);

        let outside = TempDir::new().unwrap();
        let outside_script = outside.path().join("bench-evil.sh");
        fs::write(&outside_script, "#!/bin/sh\n").unwrap();
        let bad = persisted("bad", "0", "bench", &outside_script.to_string_lossy());
        assert!(retry_command(&bad, repository.path()).1.is_empty());
    }

    #[test]
    fn formatting_matches_legacy_display_shapes() {
        assert_eq!(clock_label(UNIX_EPOCH), "00:00:00");
        assert_eq!(format_elapsed(42), "42s");
        assert_eq!(format_elapsed(150), "2.5m");
    }

    #[test]
    fn unavailable_history_cannot_overwrite_state_until_explicit_recovery() {
        let data = TempDir::new().unwrap();
        let path = state_store::jobs_path_in(data.path());
        fs::write(&path, "not json").unwrap();
        assert!(JobHistory::load_in(data.path(), repository_fixture().path()).is_err());

        let mut history = JobHistory::unavailable();
        history.begin(
            "new".into(),
            "download".into(),
            vec!["download".into()],
            "download".into(),
            String::new(),
        );
        assert!(history.persist_in(data.path()).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "not json");

        history.clear_for_recovery();
        history.persist_in(data.path()).unwrap();
        assert_eq!(
            state_store::load_jobs_in(data.path()).unwrap().jobs.len(),
            0
        );
    }

    fn persisted(name: &str, exit: &str, mode: &str, script_path: &str) -> PersistedJob {
        PersistedJob {
            name: name.into(),
            started: "12:34:56".into(),
            elapsed: "1s".into(),
            exit: exit.into(),
            mode: mode.into(),
            script_path: script_path.into(),
            metadata: JobMetadata::default(),
        }
    }

    fn running_comfy_job(history: &mut JobHistory, request_id: &str) -> u64 {
        let id = history
            .begin_cross_toolkit(CrossToolkitJobStart {
                toolkit: "l2m2".into(),
                profile: Some("qwen-image-2.1".into()),
                task: Some("image".into()),
                request_id: Some(request_id.into()),
            })
            .unwrap();
        history
            .update_cross_toolkit(
                id,
                CrossToolkitJobUpdate {
                    result_state: Some("running".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .unwrap();
        id
    }

    fn repository_fixture() -> TempDir {
        let repository = TempDir::new().unwrap();
        fs::create_dir(repository.path().join("bench-models")).unwrap();
        fs::create_dir(repository.path().join("maintenance")).unwrap();
        repository
    }
}
