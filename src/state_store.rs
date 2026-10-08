use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::de::{Error as DeError, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const JOBS_MAX: usize = 200;
pub const JOB_RUNNING_EXIT: &str = "…";
pub const JOB_INTERRUPTED_EXIT: &str = "interrupted";

pub const CHAT_MAX_MESSAGES: usize = 4_096;
pub const CHAT_MAX_MESSAGE_BYTES: usize = 2 * 1_024 * 1_024;
pub const CHAT_MAX_TOTAL_CONTENT_BYTES: usize = 16 * 1_024 * 1_024;
pub const CHAT_MAX_SESSION_FILE_BYTES: usize = 32 * 1_024 * 1_024;
pub const CHAT_LIST_MAX: usize = 1_000;

const JOBS_FILE_NAME: &str = "jobs.json";
const CHATS_DIRECTORY_NAME: &str = "chats";
const JOB_MAX_FIELD_BYTES: usize = 64 * 1_024;
const JOB_METADATA_ID_MAX_BYTES: usize = 128;
const JOB_METADATA_ARTIFACT_MAX_BYTES: usize = 2 * 1_024;
const JOB_METADATA_ARTIFACTS_MAX: usize = 64;
const JOB_METADATA_TOTAL_MAX_BYTES: usize = 16 * 1_024;
const JOBS_MAX_FILE_BYTES: usize = 16 * 1_024 * 1_024;
const CHAT_MAX_SAVED_BYTES: usize = 256;
const CHAT_MAX_ROLE_BYTES: usize = 64;
const CHAT_MAX_FILE_NAME_BYTES: usize = 255;
const UNIQUE_NAME_ATTEMPTS: u32 = 10_000;
const ATOMIC_WRITE_ATTEMPTS: u32 = 100;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Metadata shared by jobs dispatched to another toolkit or backend.
///
/// The fields are flattened into the existing jobs.json row and omitted when
/// empty, so legacy rows retain their original JSON shape. Only identifiers,
/// lifecycle state, timestamps, and bounded artifact references belong here;
/// command arguments and environment values must never be stored in metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toolkit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_state: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_refs: Vec<String>,
    /// A ComfyUI prompt UUID durably allocated before one POST attempt. This is
    /// intent only; it is never evidence that the native backend accepted work.
    /// The exact route/runtime and service-generation binding is not persisted;
    /// a transient permit binds the route only to its current coordinator.
    /// Restart must reconcile and must not adopt or replay this identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comfy_dispatch_candidate: Option<String>,
}

/// A persisted job row in the same field order and string representation as
/// the legacy Python `jobs.json` file.
///
/// Histories are stored oldest-to-newest. `mode` and `script_path` default to
/// empty strings when reading early legacy rows that predate those fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRecord {
    pub name: String,
    pub started: String,
    pub elapsed: String,
    pub exit: String,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub script_path: String,
    #[serde(flatten)]
    pub metadata: JobMetadata,
}

/// A malformed recent row omitted while loading a job history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRowIssue {
    pub source_index: usize,
    pub error: String,
}

/// The bounded, normalized result of loading `jobs.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobHistoryLoad {
    pub jobs: Vec<JobRecord>,
    pub issues: Vec<JobRowIssue>,
    pub reconciled_running: usize,
    pub truncated_rows: usize,
}

/// A role/content message compatible with legacy saved chat sessions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// The machine-readable legacy chat session shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatSession {
    pub saved: String,
    pub history: Vec<ChatMessage>,
}

/// Paths and typed data returned after a chat session is saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedChatSession {
    pub session: ChatSession,
    pub json_path: PathBuf,
    pub markdown_path: PathBuf,
}

/// Lightweight, validated metadata for a saved chat session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatSessionSummary {
    pub file_name: String,
    pub saved: String,
    pub message_count: usize,
    pub json_path: PathBuf,
    pub markdown_path: Option<PathBuf>,
}

/// A malformed session omitted from a chat listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatSessionIssue {
    pub file_name: String,
    pub error: String,
}

/// A newest-first, bounded chat session listing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatSessionList {
    pub sessions: Vec<ChatSessionSummary>,
    pub issues: Vec<ChatSessionIssue>,
    pub truncated_entries: usize,
    pub ignored_entries: usize,
}

/// Resolve the L3MS state root from `L3MS_DATA_DIR`, falling back to
/// `~/.l3ms`. An explicitly empty `L3MS_DATA_DIR` is rejected rather than
/// accidentally treating the process working directory as application data.
pub fn data_root() -> Result<PathBuf> {
    if let Some(configured) = std::env::var_os("L3MS_DATA_DIR") {
        if configured.is_empty() {
            return Err(anyhow!("L3MS_DATA_DIR must not be empty"));
        }
        return Ok(PathBuf::from(configured));
    }

    home_directory()
        .map(|home| home.join(".l3ms"))
        .ok_or_else(|| anyhow!("could not determine home directory for L3MS state"))
}

pub fn jobs_path() -> Result<PathBuf> {
    Ok(jobs_path_in(data_root()?))
}

pub fn chats_path() -> Result<PathBuf> {
    Ok(chats_path_in(data_root()?))
}

pub fn jobs_path_in(data_root: impl AsRef<Path>) -> PathBuf {
    data_root.as_ref().join(JOBS_FILE_NAME)
}

pub fn chats_path_in(data_root: impl AsRef<Path>) -> PathBuf {
    data_root.as_ref().join(CHATS_DIRECTORY_NAME)
}

/// Load the configured job history. Missing state is an empty history.
pub fn load_jobs() -> Result<JobHistoryLoad> {
    load_jobs_in(data_root()?)
}

/// Load and normalize the most recent 200 persisted job rows.
///
/// Malformed rows are skipped and reported in `issues`. A stale running
/// marker is returned as `exit == "interrupted"`; cross-toolkit queued and
/// running states are also marked interrupted to prevent implicit replay.
/// Callers can persist this reconciliation with `save_jobs` after loading.
pub fn load_jobs_in(data_root: impl AsRef<Path>) -> Result<JobHistoryLoad> {
    let path = jobs_path_in(data_root);
    let Some(bytes) = read_bounded_optional(&path, JOBS_MAX_FILE_BYTES)? else {
        return Ok(JobHistoryLoad::default());
    };
    let raw: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse job history {}", path.display()))?;
    load_jobs_from_value(raw, &path)
}

/// Coordinator-only strict load. Unlike legacy normalization, this rejects
/// duplicate object keys before serde_json::Value can collapse them.
pub(crate) fn load_jobs_strict_in(data_root: impl AsRef<Path>) -> Result<JobHistoryLoad> {
    let path = jobs_path_in(data_root);
    let Some(bytes) = read_bounded_optional(&path, JOBS_MAX_FILE_BYTES)? else {
        return Ok(JobHistoryLoad::default());
    };
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    let StrictJsonValue(raw) = StrictJsonValue::deserialize(&mut deserializer)
        .with_context(|| format!("failed to parse strict job history {}", path.display()))?;
    deserializer
        .end()
        .with_context(|| format!("failed to parse strict job history {}", path.display()))?;
    load_jobs_from_value(raw, &path)
}

fn load_jobs_from_value(raw: Value, path: &Path) -> Result<JobHistoryLoad> {
    let rows = raw
        .as_array()
        .ok_or_else(|| anyhow!("job history must be a JSON array: {}", path.display()))?;

    let first = rows.len().saturating_sub(JOBS_MAX);
    let mut result = JobHistoryLoad {
        jobs: Vec::with_capacity(rows.len() - first),
        issues: Vec::new(),
        reconciled_running: 0,
        truncated_rows: first,
    };
    let mut seen_comfy_candidates = BTreeSet::new();

    for (source_index, row) in rows.iter().enumerate().skip(first) {
        if !row.is_object() {
            result.issues.push(JobRowIssue {
                source_index,
                error: "job row must be a JSON object".to_owned(),
            });
            continue;
        }

        let mut job: JobRecord = match serde_json::from_value(row.clone()) {
            Ok(job) => job,
            Err(error) => {
                result.issues.push(JobRowIssue {
                    source_index,
                    error: format!("invalid job row: {error}"),
                });
                continue;
            }
        };
        let errors = validate_job_record(&job);
        if !errors.is_empty() {
            result.issues.push(JobRowIssue {
                source_index,
                error: errors.join("; "),
            });
            continue;
        }
        if let Some(candidate) = job.metadata.comfy_dispatch_candidate.as_deref() {
            if !seen_comfy_candidates.insert(candidate.to_owned()) {
                result.issues.push(JobRowIssue {
                    source_index,
                    error: "duplicate comfy_dispatch_candidate in retained history".to_owned(),
                });
                continue;
            }
        }

        if job.exit == JOB_RUNNING_EXIT {
            job.exit = JOB_INTERRUPTED_EXIT.to_owned();
            if job.metadata.toolkit.is_some() {
                job.metadata.result_state = Some("interrupted".to_owned());
                if job.metadata.finished_at.is_none() {
                    job.metadata.finished_at = Some(utc_timestamp_now());
                }
            }
            result.reconciled_running += 1;
        }
        result.jobs.push(job);
    }

    Ok(result)
}

struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonValueVisitor)
    }
}

struct StrictJsonValueVisitor;

impl<'de> Visitor<'de> for StrictJsonValueVisitor {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: DeError,
    {
        serde_json::Number::from_f64(value)
            .map(|number| StrictJsonValue(Value::Number(number)))
            .ok_or_else(|| E::custom("invalid JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::String(value)))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(StrictJsonValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(StrictJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom("duplicate JSON object key"));
            }
            let StrictJsonValue(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(StrictJsonValue(Value::Object(values)))
    }
}

/// Persist the configured job history atomically, retaining its most recent
/// 200 rows in legacy oldest-to-newest order.
pub fn save_jobs(jobs: &[JobRecord]) -> Result<()> {
    save_jobs_in(data_root()?, jobs)
}

/// Persist a job history below an explicit data root.
pub fn save_jobs_in(data_root: impl AsRef<Path>, jobs: &[JobRecord]) -> Result<()> {
    let first = jobs.len().saturating_sub(JOBS_MAX);
    let bounded = &jobs[first..];
    let mut seen_comfy_candidates = BTreeSet::new();
    for (offset, job) in bounded.iter().enumerate() {
        let errors = validate_job_record(job);
        if !errors.is_empty() {
            return Err(anyhow!(
                "cannot save job row {}: {}",
                first + offset,
                errors.join("; ")
            ));
        }
        if let Some(candidate) = job.metadata.comfy_dispatch_candidate.as_deref() {
            if !seen_comfy_candidates.insert(candidate) {
                return Err(anyhow!(
                    "cannot save duplicate comfy_dispatch_candidate in retained history"
                ));
            }
        }
    }

    let mut serialized = serde_json::to_vec_pretty(bounded).context("failed to serialize jobs")?;
    serialized.push(b'\n');
    if serialized.len() > JOBS_MAX_FILE_BYTES {
        return Err(anyhow!(
            "serialized job history exceeds {} bytes",
            JOBS_MAX_FILE_BYTES
        ));
    }
    atomic_write(&jobs_path_in(data_root), &serialized)
}

/// Validate a single typed job record without mutating it.
pub fn validate_job_record(job: &JobRecord) -> Vec<String> {
    let mut errors = Vec::new();
    validate_required_bounded_field("name", &job.name, JOB_MAX_FIELD_BYTES, &mut errors);
    validate_required_bounded_field("started", &job.started, JOB_MAX_FIELD_BYTES, &mut errors);
    validate_required_bounded_field("elapsed", &job.elapsed, JOB_MAX_FIELD_BYTES, &mut errors);
    validate_required_bounded_field("exit", &job.exit, JOB_MAX_FIELD_BYTES, &mut errors);
    validate_bounded_field("mode", &job.mode, JOB_MAX_FIELD_BYTES, &mut errors);
    validate_bounded_field(
        "script_path",
        &job.script_path,
        JOB_MAX_FIELD_BYTES,
        &mut errors,
    );
    errors.extend(validate_job_metadata(&job.metadata));
    errors
}

/// Validate cross-toolkit metadata. This intentionally accepts only
/// non-secret identifiers, known lifecycle values, strict UTC timestamps,
/// and local paths or credential-free HTTPS artifact URLs.
pub fn validate_job_metadata(metadata: &JobMetadata) -> Vec<String> {
    let mut errors = Vec::new();
    let mut total_bytes = 0_usize;
    for (field, value) in [
        ("toolkit", metadata.toolkit.as_deref()),
        ("profile", metadata.profile.as_deref()),
        ("task", metadata.task.as_deref()),
        ("request_id", metadata.request_id.as_deref()),
        ("backend_id", metadata.backend_id.as_deref()),
        (
            "comfy_dispatch_candidate",
            metadata.comfy_dispatch_candidate.as_deref(),
        ),
    ] {
        if let Some(value) = value {
            total_bytes = total_bytes.saturating_add(value.len());
            validate_metadata_identifier(field, value, &mut errors);
        }
    }

    if let Some(candidate) = metadata.comfy_dispatch_candidate.as_deref() {
        if !is_canonical_lower_hex_uuid(candidate) {
            errors.push("comfy_dispatch_candidate must be a canonical lowercase UUID".to_owned());
        }
        if metadata.toolkit.as_deref() != Some("l2m2") {
            errors.push("comfy_dispatch_candidate requires the l2m2 toolkit".to_owned());
        }
        for (field, value) in [
            ("profile", metadata.profile.as_deref()),
            ("task", metadata.task.as_deref()),
            ("request_id", metadata.request_id.as_deref()),
        ] {
            if value.is_none() {
                errors.push(format!("comfy_dispatch_candidate requires {field}"));
            }
        }
        if metadata
            .backend_id
            .as_deref()
            .is_some_and(|backend_id| backend_id != candidate)
        {
            errors.push(
                "backend_id must match comfy_dispatch_candidate when both are present".to_owned(),
            );
        }
    }

    for (field, value) in [
        ("submitted_at", metadata.submitted_at.as_deref()),
        ("started_at", metadata.started_at.as_deref()),
        ("finished_at", metadata.finished_at.as_deref()),
    ] {
        if let Some(value) = value {
            total_bytes = total_bytes.saturating_add(value.len());
            if !is_utc_timestamp(value) {
                errors.push(format!(
                    "{field} must be a UTC timestamp in YYYY-MM-DDTHH:MM:SSZ form"
                ));
            }
        }
    }

    if let Some(state) = metadata.result_state.as_deref() {
        total_bytes = total_bytes.saturating_add(state.len());
        if !matches!(
            state,
            "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted"
        ) {
            errors.push("result_state is not a supported job state".to_owned());
        }
    }

    if metadata.artifact_refs.len() > JOB_METADATA_ARTIFACTS_MAX {
        errors.push(format!(
            "artifact_refs contains {}; maximum is {}",
            metadata.artifact_refs.len(),
            JOB_METADATA_ARTIFACTS_MAX
        ));
    }
    for (index, artifact) in metadata.artifact_refs.iter().enumerate() {
        total_bytes = total_bytes.saturating_add(artifact.len());
        if artifact.len() > JOB_METADATA_ARTIFACT_MAX_BYTES {
            errors.push(format!(
                "artifact_refs[{index}] exceeds {JOB_METADATA_ARTIFACT_MAX_BYTES} bytes"
            ));
        }
        if artifact.is_empty() || artifact.chars().any(char::is_control) {
            errors.push(format!(
                "artifact_refs[{index}] must be non-empty and contain no control characters"
            ));
        }
        if contains_obvious_secret(artifact) {
            errors.push(format!(
                "artifact_refs[{index}] appears to contain credential material"
            ));
        }
        if !is_safe_artifact_reference(artifact) {
            errors.push(format!(
                "artifact_refs[{index}] must be a local path or credential-free HTTPS URL without query or fragment"
            ));
        }
    }
    if total_bytes > JOB_METADATA_TOTAL_MAX_BYTES {
        errors.push(format!(
            "job metadata exceeds {JOB_METADATA_TOTAL_MAX_BYTES} total bytes"
        ));
    }
    errors
}

fn is_canonical_lower_hex_uuid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            }
        })
}

fn validate_metadata_identifier(field: &str, value: &str, errors: &mut Vec<String>) {
    if value.is_empty() || value.len() > JOB_METADATA_ID_MAX_BYTES {
        errors.push(format!(
            "{field} must contain 1 to {JOB_METADATA_ID_MAX_BYTES} bytes"
        ));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte))
    {
        errors.push(format!("{field} must use only safe identifier characters"));
    }
    if contains_obvious_secret(value) {
        errors.push(format!("{field} appears to contain credential material"));
    }
}

fn contains_obvious_secret(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        "bearer ",
        "api_key=",
        "apikey=",
        "access_token=",
        "refresh_token=",
        "password=",
        "secret=",
        "token=",
        "sk-",
        "hf_",
        "ghp_",
        "github_pat_",
        "xoxb-",
        "xoxp-",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        || lower.starts_with("akia")
}

fn is_safe_artifact_reference(value: &str) -> bool {
    if value.contains('?') || value.contains('#') || value.contains('\\') {
        return false;
    }
    if let Some(url) = value.strip_prefix("https://") {
        let Some((authority, path)) = url.split_once('/') else {
            return !url.is_empty() && !url.contains('@') && !url.chars().any(char::is_whitespace);
        };
        !authority.is_empty()
            && !authority.contains('@')
            && !authority.chars().any(char::is_whitespace)
            && !path.is_empty()
            && !path.chars().any(char::is_whitespace)
    } else {
        !value.contains("://")
            && !value.chars().any(char::is_whitespace)
            && !value.starts_with("//")
    }
}

pub(crate) fn is_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
        || !bytes.iter().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7 | 10 | 13 | 16 | 19) || byte.is_ascii_digit()
        })
    {
        return false;
    }
    let number =
        |range: std::ops::Range<usize>| value.get(range).and_then(|part| part.parse::<u32>().ok());
    let Some(year) = number(0..4) else {
        return false;
    };
    let Some(month) = number(5..7) else {
        return false;
    };
    let Some(day) = number(8..10) else {
        return false;
    };
    let Some(hour) = number(11..13) else {
        return false;
    };
    let Some(minute) = number(14..16) else {
        return false;
    };
    let Some(second) = number(17..19) else {
        return false;
    };
    if !(1..=12).contains(&month) || !(0..=23).contains(&hour) || minute > 59 || second > 59 {
        return false;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    (1..=days_in_month).contains(&day)
}

pub(crate) fn utc_timestamp_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (year, month, day, hour, minute, second) = utc_parts(seconds);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Save a non-empty chat history as a unique JSON/Markdown pair.
pub fn save_chat_session(history: &[ChatMessage]) -> Result<SavedChatSession> {
    save_chat_session_in(data_root()?, history)
}

/// Save a chat history below an explicit data root.
pub fn save_chat_session_in(
    data_root: impl AsRef<Path>,
    history: &[ChatMessage],
) -> Result<SavedChatSession> {
    save_chat_session_at_in(data_root.as_ref(), history, SystemTime::now())
}

/// Load a chat session selected by a filename returned from
/// `list_chat_sessions`, or by an absolute path contained in the chats root.
pub fn load_chat_session(session_file: impl AsRef<Path>) -> Result<ChatSession> {
    load_chat_session_in(data_root()?, session_file)
}

/// Load and validate a legacy-compatible chat JSON file below an explicit
/// data root. Relative requests must be a single filename; absolute requests
/// are canonicalized and must remain inside that root's `chats` directory.
pub fn load_chat_session_in(
    data_root: impl AsRef<Path>,
    session_file: impl AsRef<Path>,
) -> Result<ChatSession> {
    let chats = chats_path_in(data_root);
    let path = resolve_chat_session_path(&chats, session_file.as_ref())?;
    let bytes = read_bounded(&path, CHAT_MAX_SESSION_FILE_BYTES)?;
    let session: ChatSession = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse chat session {}", path.display()))?;
    let errors = validate_chat_session(&session);
    if !errors.is_empty() {
        return Err(anyhow!(
            "invalid chat session {}: {}",
            path.display(),
            errors.join("; ")
        ));
    }
    Ok(session)
}

/// List up to 1,000 valid chat sessions newest-first. Individual malformed
/// files are returned as issues so one bad session does not hide the rest.
pub fn list_chat_sessions() -> Result<ChatSessionList> {
    list_chat_sessions_in(data_root()?)
}

/// List chat sessions below an explicit data root.
pub fn list_chat_sessions_in(data_root: impl AsRef<Path>) -> Result<ChatSessionList> {
    let data_root = data_root.as_ref();
    let chats = chats_path_in(data_root);
    let entries = match fs::read_dir(&chats) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(ChatSessionList::default()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read chats directory {}", chats.display()))
        }
    };

    // Keep only the lexicographically newest bounded set while scanning, so a
    // directory with an unexpectedly large number of sessions cannot make the
    // listing allocate one path per file.
    let mut candidates: BinaryHeap<Reverse<(String, PathBuf)>> = BinaryHeap::new();
    let mut json_entries = 0_usize;
    let mut ignored_entries = 0_usize;
    for entry in entries {
        let entry = entry
            .with_context(|| format!("failed to inspect chats directory {}", chats.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect chat entry {}", entry.path().display()))?;
        if !file_type.is_file()
            || entry.path().extension().and_then(|value| value.to_str()) != Some("json")
        {
            ignored_entries += 1;
            continue;
        }
        let Some(file_name) = entry.file_name().to_str().map(str::to_owned) else {
            ignored_entries += 1;
            continue;
        };

        json_entries += 1;
        candidates.push(Reverse((file_name, entry.path())));
        if candidates.len() > CHAT_LIST_MAX {
            candidates.pop();
        }
    }

    let mut candidates: Vec<(String, PathBuf)> = candidates
        .into_iter()
        .map(|Reverse(candidate)| candidate)
        .collect();
    candidates.sort_unstable_by(|left, right| right.0.cmp(&left.0));

    let mut result = ChatSessionList {
        sessions: Vec::with_capacity(candidates.len()),
        issues: Vec::new(),
        truncated_entries: json_entries.saturating_sub(CHAT_LIST_MAX),
        ignored_entries,
    };
    for (file_name, path) in candidates {
        match load_chat_session_in(data_root, Path::new(&file_name)) {
            Ok(session) => {
                let json_path = fs::canonicalize(&path).with_context(|| {
                    format!("failed to resolve chat session {}", path.display())
                })?;
                let markdown_candidate = path.with_extension("md");
                let markdown_path = match fs::symlink_metadata(&markdown_candidate) {
                    Ok(metadata) if metadata.file_type().is_file() => {
                        Some(fs::canonicalize(&markdown_candidate).with_context(|| {
                            format!(
                                "failed to resolve chat markdown {}",
                                markdown_candidate.display()
                            )
                        })?)
                    }
                    _ => None,
                };
                result.sessions.push(ChatSessionSummary {
                    file_name,
                    saved: session.saved,
                    message_count: session.history.len(),
                    json_path,
                    markdown_path,
                });
            }
            Err(error) => result.issues.push(ChatSessionIssue {
                file_name,
                error: format!("{error:#}"),
            }),
        }
    }
    Ok(result)
}

/// Validate a typed chat session and return every bounded-validation error in
/// deterministic order.
pub fn validate_chat_session(session: &ChatSession) -> Vec<String> {
    let mut errors = Vec::new();
    validate_required_bounded_field("saved", &session.saved, CHAT_MAX_SAVED_BYTES, &mut errors);
    if session.saved.chars().any(char::is_control) {
        errors.push("saved must not contain control characters".to_owned());
    }
    errors.extend(validate_chat_history(&session.history));
    errors
}

/// Validate chat messages without requiring session metadata.
pub fn validate_chat_history(history: &[ChatMessage]) -> Vec<String> {
    let mut errors = Vec::new();
    if history.len() > CHAT_MAX_MESSAGES {
        errors.push(format!(
            "history contains {} messages; maximum is {}",
            history.len(),
            CHAT_MAX_MESSAGES
        ));
    }

    let mut total_content_bytes = 0_usize;
    for (index, message) in history.iter().take(CHAT_MAX_MESSAGES + 1).enumerate() {
        if message.role.trim().is_empty() {
            errors.push(format!("history[{index}].role is required"));
        }
        if message.role.len() > CHAT_MAX_ROLE_BYTES {
            errors.push(format!(
                "history[{index}].role exceeds {CHAT_MAX_ROLE_BYTES} bytes"
            ));
        }
        if message.role.chars().any(char::is_control) {
            errors.push(format!(
                "history[{index}].role must not contain control characters"
            ));
        }
        if message.content.len() > CHAT_MAX_MESSAGE_BYTES {
            errors.push(format!(
                "history[{index}].content exceeds {CHAT_MAX_MESSAGE_BYTES} bytes"
            ));
        }
        total_content_bytes = total_content_bytes.saturating_add(message.content.len());
    }
    if total_content_bytes > CHAT_MAX_TOTAL_CONTENT_BYTES {
        errors.push(format!(
            "history content exceeds {CHAT_MAX_TOTAL_CONTENT_BYTES} bytes"
        ));
    }
    errors
}

fn save_chat_session_at_in(
    data_root: &Path,
    history: &[ChatMessage],
    saved_at: SystemTime,
) -> Result<SavedChatSession> {
    if history.is_empty() {
        return Err(anyhow!("cannot save an empty chat session"));
    }
    let history_errors = validate_chat_history(history);
    if !history_errors.is_empty() {
        return Err(anyhow!(
            "cannot save chat session: {}",
            history_errors.join("; ")
        ));
    }

    let seconds = saved_at
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let base_stamp = format_chat_stamp(seconds);
    let chats = chats_path_in(data_root);
    fs::create_dir_all(&chats)
        .with_context(|| format!("failed to create chats directory {}", chats.display()))?;
    let reserved = reserve_chat_paths(&chats, &base_stamp)?;
    let session = ChatSession {
        saved: reserved.stem.clone(),
        history: history.to_vec(),
    };
    let errors = validate_chat_session(&session);
    if !errors.is_empty() {
        return Err(anyhow!("cannot save chat session: {}", errors.join("; ")));
    }

    let mut json = serde_json::to_vec_pretty(&session).context("failed to serialize chat")?;
    json.push(b'\n');
    if json.len() > CHAT_MAX_SESSION_FILE_BYTES {
        return Err(anyhow!(
            "serialized chat session exceeds {} bytes",
            CHAT_MAX_SESSION_FILE_BYTES
        ));
    }
    let markdown = render_chat_markdown(seconds, history);

    // Publish Markdown first and JSON last. Listings only discover JSON, so a
    // process interruption cannot expose a session whose companion Markdown
    // has not yet been written.
    atomic_write(&reserved.markdown_path, markdown.as_bytes()).with_context(|| {
        format!(
            "failed to save chat markdown {}",
            reserved.markdown_path.display()
        )
    })?;
    if let Err(error) = atomic_write(&reserved.json_path, &json)
        .with_context(|| format!("failed to save chat JSON {}", reserved.json_path.display()))
    {
        let _ = fs::remove_file(&reserved.markdown_path);
        return Err(error);
    }

    Ok(SavedChatSession {
        session,
        json_path: reserved.json_path.clone(),
        markdown_path: reserved.markdown_path.clone(),
    })
}

struct ReservedChatPaths {
    stem: String,
    json_path: PathBuf,
    markdown_path: PathBuf,
    reservation_path: PathBuf,
}

impl Drop for ReservedChatPaths {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.reservation_path);
    }
}

fn reserve_chat_paths(chats: &Path, base_stamp: &str) -> Result<ReservedChatPaths> {
    for collision in 0..UNIQUE_NAME_ATTEMPTS {
        let stem = if collision == 0 {
            base_stamp.to_owned()
        } else {
            format!("{base_stamp}_{collision:04}")
        };
        let json_path = chats.join(format!("{stem}.json"));
        let markdown_path = chats.join(format!("{stem}.md"));
        let reservation_path = chats.join(format!(".{stem}.l3ms-reserve"));

        if json_path.exists() || markdown_path.exists() {
            continue;
        }
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&reservation_path)
        {
            Ok(file) => {
                drop(file);
                if json_path.exists() || markdown_path.exists() {
                    let _ = fs::remove_file(&reservation_path);
                    continue;
                }
                return Ok(ReservedChatPaths {
                    stem,
                    json_path,
                    markdown_path,
                    reservation_path,
                });
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to reserve chat session name in {}", chats.display())
                })
            }
        }
    }
    Err(anyhow!(
        "could not allocate a unique chat session name after {UNIQUE_NAME_ATTEMPTS} attempts"
    ))
}

fn resolve_chat_session_path(chats: &Path, requested: &Path) -> Result<PathBuf> {
    if requested.as_os_str().is_empty() {
        return Err(anyhow!("chat session path must not be empty"));
    }
    if requested
        .components()
        .any(|component| component == Component::ParentDir)
    {
        return Err(anyhow!(
            "chat session path must stay inside chats directory"
        ));
    }

    let candidate = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        let mut components = requested.components();
        match (components.next(), components.next()) {
            (Some(Component::Normal(_)), None) => chats.join(requested),
            _ => {
                return Err(anyhow!(
                    "relative chat session path must be a single filename"
                ))
            }
        }
    };
    let file_name = candidate
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("chat session filename must be valid UTF-8"))?;
    if file_name.len() > CHAT_MAX_FILE_NAME_BYTES {
        return Err(anyhow!(
            "chat session filename exceeds {CHAT_MAX_FILE_NAME_BYTES} bytes"
        ));
    }
    if candidate.extension().and_then(|value| value.to_str()) != Some("json") {
        return Err(anyhow!("chat session path must have a .json extension"));
    }

    let canonical_chats = fs::canonicalize(chats)
        .with_context(|| format!("failed to resolve chats directory {}", chats.display()))?;
    let canonical_candidate = fs::canonicalize(&candidate)
        .with_context(|| format!("failed to resolve chat session {}", candidate.display()))?;
    if !canonical_candidate.starts_with(&canonical_chats) {
        return Err(anyhow!(
            "chat session path must stay inside chats directory"
        ));
    }
    let metadata = fs::metadata(&canonical_candidate).with_context(|| {
        format!(
            "failed to inspect chat session {}",
            canonical_candidate.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(anyhow!("chat session path must identify a regular file"));
    }
    Ok(canonical_candidate)
}

fn validate_required_bounded_field(
    name: &str,
    value: &str,
    max_bytes: usize,
    errors: &mut Vec<String>,
) {
    if value.trim().is_empty() {
        errors.push(format!("{name} is required"));
    }
    validate_bounded_field(name, value, max_bytes, errors);
}

fn validate_bounded_field(name: &str, value: &str, max_bytes: usize, errors: &mut Vec<String>) {
    if value.len() > max_bytes {
        errors.push(format!("{name} exceeds {max_bytes} bytes"));
    }
    if value.contains('\0') {
        errors.push(format!("{name} must not contain NUL"));
    }
}

fn home_directory() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| {
            let drive = std::env::var_os("HOMEDRIVE")?;
            let path = std::env::var_os("HOMEPATH")?;
            if drive.is_empty() || path.is_empty() {
                return None;
            }
            let mut home = PathBuf::from(drive);
            home.push(path);
            Some(home)
        })
}

fn read_bounded_optional(path: &Path, max_bytes: usize) -> Result<Option<Vec<u8>>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to open {}", path.display()))
        }
    };
    read_bounded_file(file, path, max_bytes).map(Some)
}

fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    read_bounded_file(file, path, max_bytes)
}

fn read_bounded_file(file: File, path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    if metadata.len() > max_bytes as u64 {
        return Err(anyhow!(
            "{} exceeds maximum size of {} bytes",
            path.display(),
            max_bytes
        ));
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    let mut reader = file.take(max_bytes as u64 + 1);
    reader
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() > max_bytes {
        return Err(anyhow!(
            "{} exceeds maximum size of {} bytes",
            path.display(),
            max_bytes
        ));
    }
    Ok(bytes)
}

fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create directory {}", parent.display()))?;

    let file_name = path
        .file_name()
        .map_or_else(|| OsString::from("file"), OsString::from)
        .to_string_lossy()
        .into_owned();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let unix_mode = existing_unix_mode(path);

    for attempt in 0..ATOMIC_WRITE_ATTEMPTS {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.l3ms-tmp-{}-{nonce}-{counter}-{attempt}",
            std::process::id()
        ));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create temporary file for {}", path.display())
                })
            }
        };

        let result = (|| -> Result<()> {
            file.write_all(content)?;
            file.sync_all()?;
            drop(file);

            #[cfg(unix)]
            if let Some(mode) = unix_mode {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&temporary, fs::Permissions::from_mode(mode))?;
            }
            #[cfg(not(unix))]
            let _ = unix_mode;

            fs::rename(&temporary, path)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        return result.with_context(|| format!("failed to atomically write {}", path.display()));
    }
    Err(anyhow!(
        "could not allocate a temporary file for {}",
        path.display()
    ))
}

#[cfg(unix)]
fn existing_unix_mode(path: &Path) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path)
        .ok()
        .map(|metadata| metadata.permissions().mode())
}

#[cfg(not(unix))]
fn existing_unix_mode(_path: &Path) -> Option<u32> {
    None
}

fn render_chat_markdown(saved_at_seconds: u64, history: &[ChatMessage]) -> String {
    let mut markdown = format!("# Chat - {} UTC\n", format_human_stamp(saved_at_seconds));
    for message in history {
        let role = match message.role.as_str() {
            "user" => "You",
            "assistant" => "Assistant",
            other => other,
        };
        markdown.push_str("\n## ");
        markdown.push_str(role);
        markdown.push('\n');
        markdown.push_str(&message.content);
        markdown.push('\n');
    }
    markdown
}

fn format_chat_stamp(seconds: u64) -> String {
    let (year, month, day, hour, minute, second) = utc_parts(seconds);
    format!("{year:04}{month:02}{day:02}_{hour:02}{minute:02}{second:02}")
}

fn format_human_stamp(seconds: u64) -> String {
    let (year, month, day, hour, minute, second) = utc_parts(seconds);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}")
}

fn utc_parts(seconds: u64) -> (i64, i64, i64, u64, u64, u64) {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    // Civil date conversion by Howard Hinnant, expressed with integer
    // arithmetic so state naming does not require a date/time dependency.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn job(name: impl Into<String>, exit: impl Into<String>) -> JobRecord {
        JobRecord {
            name: name.into(),
            started: "2026-07-17 12:00:00".to_owned(),
            elapsed: "1s".to_owned(),
            exit: exit.into(),
            mode: "bench".to_owned(),
            script_path: "bench-models/bench-example.sh".to_owned(),
            metadata: JobMetadata::default(),
        }
    }

    fn history() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "user".to_owned(),
                content: "Hello".to_owned(),
            },
            ChatMessage {
                role: "assistant".to_owned(),
                content: "Hi there".to_owned(),
            },
        ]
    }

    #[test]
    fn missing_jobs_file_loads_empty() {
        let temp = TempDir::new().unwrap();
        assert_eq!(
            load_jobs_in(temp.path()).unwrap(),
            JobHistoryLoad::default()
        );
    }

    #[test]
    fn loads_legacy_job_rows_and_defaults_early_optional_fields() {
        let temp = TempDir::new().unwrap();
        fs::write(
            jobs_path_in(temp.path()),
            r#"[
              {"name":"one","started":"now","elapsed":"2s","exit":"0","mode":"bench","script_path":"bench.sh"},
              {"name":"old","started":"then","elapsed":"3s","exit":"1"}
            ]"#,
        )
        .unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert!(loaded.issues.is_empty());
        assert_eq!(loaded.jobs.len(), 2);
        assert_eq!(loaded.jobs[0].script_path, "bench.sh");
        assert_eq!(loaded.jobs[1].mode, "");
        assert_eq!(loaded.jobs[1].script_path, "");
        assert_eq!(loaded.jobs[1].metadata, JobMetadata::default());
    }

    #[test]
    fn legacy_job_rows_round_trip_without_changing_their_json_shape() {
        let legacy = r#"{"name":"old","started":"then","elapsed":"3s","exit":"0","mode":"","script_path":""}"#;
        let job: JobRecord = serde_json::from_str(legacy).unwrap();
        assert_eq!(serde_json::to_string(&job).unwrap(), legacy);
    }

    #[test]
    fn validates_cross_toolkit_metadata_and_rejects_credentials_or_oversize() {
        let valid = JobMetadata {
            toolkit: Some("l2m2".into()),
            profile: Some("qwen3-8b".into()),
            task: Some("transcribe-42".into()),
            request_id: Some("req:123".into()),
            backend_id: Some("llama.cpp:worker-1".into()),
            submitted_at: Some("2026-09-28T15:04:05Z".into()),
            started_at: Some("2026-09-28T15:04:05Z".into()),
            finished_at: None,
            result_state: Some("running".into()),
            artifact_refs: vec![
                "artifacts/transcript.json".into(),
                "https://store.example/a.wav".into(),
            ],
            comfy_dispatch_candidate: None,
        };
        assert!(validate_job_metadata(&valid).is_empty());

        let mut candidate = valid.clone();
        candidate.toolkit = Some("l2m2".into());
        candidate.profile = Some("qwen-image-2.1".into());
        candidate.task = Some("image".into());
        candidate.request_id = Some("req-comfy-1".into());
        candidate.backend_id = None;
        candidate.comfy_dispatch_candidate = Some("123e4567-e89b-12d3-a456-426614174000".into());
        assert!(validate_job_metadata(&candidate).is_empty());

        let mut invalid = valid.clone();
        invalid.request_id = Some("Bearer abc123".into());
        invalid.submitted_at = Some("2026-02-30T15:04:05Z".into());
        invalid.result_state = Some("running shell with args".into());
        invalid.artifact_refs = vec!["https://user:password@store.example/a.wav".into()];
        assert!(!validate_job_metadata(&invalid).is_empty());

        let mut invalid_candidate = candidate.clone();
        invalid_candidate.comfy_dispatch_candidate =
            Some("123E4567-E89B-12D3-A456-426614174000".into());
        assert!(validate_job_metadata(&invalid_candidate)
            .iter()
            .any(|error| error.contains("canonical lowercase UUID")));

        invalid_candidate = candidate.clone();
        invalid_candidate.backend_id = Some("different-native-id".into());
        assert!(validate_job_metadata(&invalid_candidate)
            .iter()
            .any(|error| error.contains("must match comfy_dispatch_candidate")));

        invalid_candidate = candidate.clone();
        invalid_candidate.profile = None;
        assert!(validate_job_metadata(&invalid_candidate)
            .iter()
            .any(|error| error.contains("requires profile")));

        invalid.artifact_refs = vec!["https://store.example/a.wav?token=abc".into()];
        assert!(!validate_job_metadata(&invalid).is_empty());

        let mut oversized = valid;
        oversized.artifact_refs = vec!["a".repeat(JOB_METADATA_ARTIFACT_MAX_BYTES + 1)];
        assert!(validate_job_metadata(&oversized)
            .iter()
            .any(|error| error.contains("exceeds")));
    }

    #[test]
    fn malformed_persisted_metadata_is_reported_and_omitted() {
        let temp = TempDir::new().unwrap();
        fs::write(
            jobs_path_in(temp.path()),
            r#"[{"name":"bad","started":"now","elapsed":"1s","exit":"0","result_state":"mystery"},{"name":"good","started":"now","elapsed":"1s","exit":"0","toolkit":"l2m2","result_state":"succeeded","finished_at":"2026-09-28T15:04:05Z"}]"#,
        )
        .unwrap();
        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.jobs.len(), 1);
        assert_eq!(loaded.jobs[0].metadata.toolkit.as_deref(), Some("l2m2"));
        assert_eq!(loaded.issues.len(), 1);
    }

    #[test]
    fn duplicate_comfy_candidates_are_rejected_on_save_and_load() {
        const CANDIDATE: &str = "123e4567-e89b-12d3-a456-426614174000";
        let temp = TempDir::new().unwrap();
        let make_job = |name: &str| JobRecord {
            name: name.into(),
            started: "12:34:56".into(),
            elapsed: "-".into(),
            exit: JOB_RUNNING_EXIT.into(),
            mode: "cross-toolkit".into(),
            script_path: String::new(),
            metadata: JobMetadata {
                toolkit: Some("l2m2".into()),
                profile: Some("qwen-image-2.1".into()),
                task: Some("image".into()),
                request_id: Some(format!("req-{name}")),
                result_state: Some("running".into()),
                comfy_dispatch_candidate: Some(CANDIDATE.into()),
                ..JobMetadata::default()
            },
        };
        assert!(save_jobs_in(temp.path(), &[make_job("one"), make_job("two")]).is_err());

        fs::write(
            jobs_path_in(temp.path()),
            format!(
                r#"[
                  {{"name":"one","started":"12:34:56","elapsed":"-","exit":"…","mode":"cross-toolkit","script_path":"","toolkit":"l2m2","profile":"qwen-image-2.1","task":"image","request_id":"req-one","result_state":"running","comfy_dispatch_candidate":"{CANDIDATE}"}},
                  {{"name":"two","started":"12:34:56","elapsed":"-","exit":"…","mode":"cross-toolkit","script_path":"","toolkit":"l2m2","profile":"qwen-image-2.1","task":"image","request_id":"req-two","result_state":"running","comfy_dispatch_candidate":"{CANDIDATE}"}}
                ]"#
            ),
        )
        .unwrap();
        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.jobs.len(), 1);
        assert_eq!(loaded.issues.len(), 1);
        assert_eq!(loaded.issues[0].source_index, 1);
        assert_eq!(
            loaded.jobs[0].metadata.comfy_dispatch_candidate.as_deref(),
            Some(CANDIDATE)
        );
    }

    #[test]
    fn reports_malformed_job_rows_without_losing_valid_recent_rows() {
        let temp = TempDir::new().unwrap();
        fs::write(
            jobs_path_in(temp.path()),
            r#"[
              {"name":"valid","started":"now","elapsed":"2s","exit":"0","mode":"run","script_path":"model"},
              "not an object",
              {"name":42,"started":"now","elapsed":"2s","exit":"0"},
              {"name":"","started":"now","elapsed":"2s","exit":"0"}
            ]"#,
        )
        .unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.jobs.len(), 1);
        assert_eq!(loaded.issues.len(), 3);
        assert_eq!(loaded.issues[0].source_index, 1);
        assert!(loaded.issues[0].error.contains("JSON object"));
        assert!(loaded.issues[1].error.contains("invalid job row"));
        assert!(loaded.issues[2].error.contains("name is required"));
    }

    #[test]
    fn rejects_malformed_job_json_and_non_array_roots() {
        let temp = TempDir::new().unwrap();
        fs::write(jobs_path_in(temp.path()), "[").unwrap();
        assert!(load_jobs_in(temp.path())
            .unwrap_err()
            .to_string()
            .contains("parse"));

        fs::write(jobs_path_in(temp.path()), r#"{"jobs":[]}"#).unwrap();
        assert!(load_jobs_in(temp.path())
            .unwrap_err()
            .to_string()
            .contains("JSON array"));
    }

    #[test]
    fn reconciles_stale_running_jobs_on_load() {
        let temp = TempDir::new().unwrap();
        save_jobs_in(temp.path(), &[job("stale", JOB_RUNNING_EXIT)]).unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.reconciled_running, 1);
        assert_eq!(loaded.jobs[0].exit, JOB_INTERRUPTED_EXIT);
        // Loading is intentionally non-destructive. Persisting the typed result
        // makes reconciliation durable at the caller's transaction boundary.
        let disk: Vec<JobRecord> =
            serde_json::from_slice(&fs::read(jobs_path_in(temp.path())).unwrap()).unwrap();
        assert_eq!(disk[0].exit, JOB_RUNNING_EXIT);
        save_jobs_in(temp.path(), &loaded.jobs).unwrap();
        let disk: Vec<JobRecord> =
            serde_json::from_slice(&fs::read(jobs_path_in(temp.path())).unwrap()).unwrap();
        assert_eq!(disk[0].exit, JOB_INTERRUPTED_EXIT);
    }

    #[test]
    fn stale_cross_toolkit_jobs_reconcile_result_state_and_timestamp() {
        let temp = TempDir::new().unwrap();
        let mut stale = job("stale", JOB_RUNNING_EXIT);
        stale.metadata = JobMetadata {
            toolkit: Some("l2m2".into()),
            result_state: Some("running".into()),
            ..JobMetadata::default()
        };
        save_jobs_in(temp.path(), &[stale]).unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.jobs[0].exit, JOB_INTERRUPTED_EXIT);
        assert_eq!(
            loaded.jobs[0].metadata.result_state.as_deref(),
            Some("interrupted")
        );
        assert!(loaded.jobs[0]
            .metadata
            .finished_at
            .as_deref()
            .is_some_and(is_utc_timestamp));
    }

    #[test]
    fn stale_queued_cross_toolkit_jobs_reconcile_as_interrupted() {
        let temp = TempDir::new().unwrap();
        let mut queued = job("queued", JOB_RUNNING_EXIT);
        queued.metadata = JobMetadata {
            toolkit: Some("l2m2".into()),
            submitted_at: Some("2026-09-28T15:04:05Z".into()),
            result_state: Some("queued".into()),
            ..JobMetadata::default()
        };
        save_jobs_in(temp.path(), &[queued]).unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.reconciled_running, 1);
        assert_eq!(loaded.jobs[0].exit, JOB_INTERRUPTED_EXIT);
        assert_eq!(
            loaded.jobs[0].metadata.result_state.as_deref(),
            Some("interrupted")
        );
        assert!(loaded.jobs[0].metadata.started_at.is_none());
        assert!(loaded.jobs[0]
            .metadata
            .finished_at
            .as_deref()
            .is_some_and(is_utc_timestamp));
    }

    #[test]
    fn bounds_jobs_to_the_most_recent_200_rows() {
        let temp = TempDir::new().unwrap();
        let jobs: Vec<JobRecord> = (0..250).map(|index| job(index.to_string(), "0")).collect();
        save_jobs_in(temp.path(), &jobs).unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.jobs.len(), JOBS_MAX);
        assert_eq!(loaded.jobs.first().unwrap().name, "50");
        assert_eq!(loaded.jobs.last().unwrap().name, "249");

        let raw: Vec<Value> =
            serde_json::from_slice(&fs::read(jobs_path_in(temp.path())).unwrap()).unwrap();
        assert_eq!(raw.len(), JOBS_MAX);
    }

    #[test]
    fn reports_truncated_legacy_job_rows_on_load() {
        let temp = TempDir::new().unwrap();
        let jobs: Vec<JobRecord> = (0..205).map(|index| job(index.to_string(), "0")).collect();
        fs::write(
            jobs_path_in(temp.path()),
            serde_json::to_vec(&jobs).unwrap(),
        )
        .unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.truncated_rows, 5);
        assert_eq!(loaded.jobs[0].name, "5");
    }

    #[test]
    fn atomically_replaces_jobs_and_leaves_no_temporary_files() {
        let temp = TempDir::new().unwrap();
        save_jobs_in(temp.path(), &[job("first", "0")]).unwrap();
        save_jobs_in(temp.path(), &[job("replacement", "1")]).unwrap();

        let loaded = load_jobs_in(temp.path()).unwrap();
        assert_eq!(loaded.jobs, vec![job("replacement", "1")]);
        let names: Vec<String> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, [JOBS_FILE_NAME]);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_job_replacement_preserves_existing_mode() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new().unwrap();
        let path = jobs_path_in(temp.path());
        save_jobs_in(temp.path(), &[job("first", "0")]).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        save_jobs_in(temp.path(), &[job("replacement", "0")]).unwrap();
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[test]
    fn saves_legacy_compatible_json_and_human_readable_markdown() {
        let temp = TempDir::new().unwrap();
        let saved = save_chat_session_at_in(
            temp.path(),
            &history(),
            UNIX_EPOCH + std::time::Duration::from_secs(1_721_224_245),
        )
        .unwrap();

        assert_eq!(saved.session.saved, "20240717_135045");
        let raw: Value = serde_json::from_slice(&fs::read(&saved.json_path).unwrap()).unwrap();
        assert_eq!(raw["saved"], saved.session.saved);
        assert_eq!(raw["history"][0]["role"], "user");
        assert_eq!(raw["history"][1]["content"], "Hi there");
        let markdown = fs::read_to_string(&saved.markdown_path).unwrap();
        assert!(markdown.contains("# Chat - 2024-07-17 13:50:45 UTC"));
        assert!(markdown.contains("## You\nHello"));
        assert!(markdown.contains("## Assistant\nHi there"));
    }

    #[test]
    fn same_second_chat_saves_are_unique_and_complete() {
        let temp = TempDir::new().unwrap();
        let instant = UNIX_EPOCH + std::time::Duration::from_secs(1_721_224_245);
        let first = save_chat_session_at_in(temp.path(), &history(), instant).unwrap();
        let second = save_chat_session_at_in(temp.path(), &history(), instant).unwrap();

        assert_ne!(first.json_path, second.json_path);
        assert_eq!(first.session.saved, "20240717_135045");
        assert_eq!(second.session.saved, "20240717_135045_0001");
        assert!(first.json_path.is_file());
        assert!(first.markdown_path.is_file());
        assert!(second.json_path.is_file());
        assert!(second.markdown_path.is_file());
        assert!(!fs::read_dir(chats_path_in(temp.path()))
            .unwrap()
            .any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("reserve")));
    }

    #[test]
    fn loads_a_legacy_chat_fixture_by_name_and_contained_absolute_path() {
        let temp = TempDir::new().unwrap();
        let chats = chats_path_in(temp.path());
        fs::create_dir_all(&chats).unwrap();
        let path = chats.join("20260102_030405.json");
        fs::write(
            &path,
            r#"{
              "saved":"20260102_030405",
              "history":[
                {"role":"user","content":"question"},
                {"role":"assistant","content":"answer"}
              ]
            }"#,
        )
        .unwrap();

        let by_name = load_chat_session_in(temp.path(), "20260102_030405.json").unwrap();
        let by_path = load_chat_session_in(temp.path(), fs::canonicalize(path).unwrap()).unwrap();
        assert_eq!(by_name, by_path);
        assert_eq!(by_name.history.len(), 2);
    }

    #[test]
    fn rejects_malformed_and_unbounded_chat_sessions() {
        let temp = TempDir::new().unwrap();
        let chats = chats_path_in(temp.path());
        fs::create_dir_all(&chats).unwrap();
        fs::write(chats.join("broken.json"), "{").unwrap();
        assert!(load_chat_session_in(temp.path(), "broken.json")
            .unwrap_err()
            .to_string()
            .contains("parse"));

        let oversized = ChatSession {
            saved: "20260102_030405".to_owned(),
            history: vec![ChatMessage {
                role: "user".to_owned(),
                content: "x".repeat(CHAT_MAX_MESSAGE_BYTES + 1),
            }],
        };
        fs::write(
            chats.join("oversized.json"),
            serde_json::to_vec(&oversized).unwrap(),
        )
        .unwrap();
        let error = load_chat_session_in(temp.path(), "oversized.json").unwrap_err();
        assert!(format!("{error:#}").contains("content exceeds"));
    }

    #[test]
    fn chat_validation_bounds_message_count_roles_and_total_content() {
        let too_many = vec![
            ChatMessage {
                role: "user".to_owned(),
                content: String::new(),
            };
            CHAT_MAX_MESSAGES + 1
        ];
        assert!(validate_chat_history(&too_many)
            .iter()
            .any(|error| error.contains("maximum")));

        let bad_role = [ChatMessage {
            role: "bad\nrole".to_owned(),
            content: String::new(),
        }];
        assert!(validate_chat_history(&bad_role)
            .iter()
            .any(|error| error.contains("control")));
    }

    #[test]
    fn refuses_to_save_empty_chat_history() {
        let temp = TempDir::new().unwrap();
        assert!(save_chat_session_in(temp.path(), &[])
            .unwrap_err()
            .to_string()
            .contains("empty"));
    }

    #[test]
    fn lists_valid_chats_newest_first_and_reports_bad_files() {
        let temp = TempDir::new().unwrap();
        let first = save_chat_session_at_in(
            temp.path(),
            &history(),
            UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000),
        )
        .unwrap();
        let second = save_chat_session_at_in(
            temp.path(),
            &history(),
            UNIX_EPOCH + std::time::Duration::from_secs(1_800_000_000),
        )
        .unwrap();
        fs::write(chats_path_in(temp.path()).join("zzzz-bad.json"), "not JSON").unwrap();
        fs::write(chats_path_in(temp.path()).join("note.txt"), "ignored").unwrap();

        let listed = list_chat_sessions_in(temp.path()).unwrap();
        assert_eq!(listed.sessions.len(), 2);
        assert_eq!(listed.issues.len(), 1);
        assert_eq!(listed.issues[0].file_name, "zzzz-bad.json");
        // The two Markdown companion files and note.txt are intentionally not
        // candidates for JSON session loading.
        assert_eq!(listed.ignored_entries, 3);
        assert_eq!(
            listed.sessions[0].file_name,
            second.json_path.file_name().unwrap().to_string_lossy()
        );
        assert_eq!(
            listed.sessions[1].file_name,
            first.json_path.file_name().unwrap().to_string_lossy()
        );
        assert_eq!(listed.sessions[0].message_count, 2);
        assert!(listed.sessions[0].markdown_path.is_some());
    }

    #[test]
    fn rejects_relative_traversal_and_non_json_chat_paths() {
        let temp = TempDir::new().unwrap();
        let chats = chats_path_in(temp.path());
        fs::create_dir_all(&chats).unwrap();
        fs::write(
            temp.path().join("outside.json"),
            r#"{"saved":"x","history":[]}"#,
        )
        .unwrap();
        fs::write(chats.join("session.md"), "text").unwrap();

        for path in ["../outside.json", "sub/session.json", "session.md"] {
            assert!(load_chat_session_in(temp.path(), path).is_err(), "{path}");
        }
    }

    #[test]
    fn rejects_absolute_paths_outside_chat_root() {
        let temp = TempDir::new().unwrap();
        fs::create_dir_all(chats_path_in(temp.path())).unwrap();
        let outside = temp.path().join("outside.json");
        fs::write(&outside, r#"{"saved":"x","history":[]}"#).unwrap();
        let error =
            load_chat_session_in(temp.path(), fs::canonicalize(outside).unwrap()).unwrap_err();
        assert!(error.to_string().contains("inside chats"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_chat_symlinks_that_escape_the_data_root() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let chats = chats_path_in(temp.path());
        fs::create_dir_all(&chats).unwrap();
        let outside = temp.path().join("outside.json");
        fs::write(&outside, r#"{"saved":"x","history":[]}"#).unwrap();
        symlink(&outside, chats.join("escape.json")).unwrap();

        assert!(load_chat_session_in(temp.path(), "escape.json")
            .unwrap_err()
            .to_string()
            .contains("inside chats"));
        let listed = list_chat_sessions_in(temp.path()).unwrap();
        assert!(listed.sessions.is_empty());
        assert_eq!(listed.ignored_entries, 1);
    }

    #[test]
    fn format_stamp_matches_legacy_filename_shape() {
        assert_eq!(format_chat_stamp(0), "19700101_000000");
        assert_eq!(format_human_stamp(0), "1970-01-01 00:00:00");
    }
}
