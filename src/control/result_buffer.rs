//! Bounded in-memory result/event retention for reconnectable coordinator IPC.
//!
//! Results are lane-native and remain in memory only. A process restart loses
//! retained result bodies; it never causes inference to be replayed. The
//! durable acceptance/history layer remains responsible for replay prevention
//! after this buffer and its bounded tombstones are gone.
//!
//! The buffer retains at most 64 jobs, 8 MiB of serialized cursor events per
//! job, and 64 MiB total, plus at most 256 LRU tombstones. Result pages are
//! measured from their actual serialized JSON and accept budgets from 1 KiB
//! through 128 KiB.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::io::{self, Write};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::catalog::Lane;
use super::ipc::validate_request_id;
use super::requests::{serialize_embedding_payload, EmbeddingUsage, MAX_EMBEDDING_JSON_BYTES};

pub const MAX_RETAINED_JOBS: usize = 64;
pub const MAX_JOB_EVENT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_TOTAL_EVENT_BYTES: usize = 64 * 1024 * 1024;
pub const MIN_PAGE_BYTES: usize = 1024;
pub const MAX_PAGE_BYTES: usize = 128 * 1024;

const MAX_TOMBSTONES: usize = 256;
const MAX_CHAT_RAW_BYTES_PER_JOB: usize = 1024 * 1024;
const MAX_DECISION_JSON_BYTES: usize = 1024 * 1024;
const MAX_ARTIFACT_PATH_BYTES: usize = 2048;
const MAX_ARTIFACTS_PER_APPEND: usize = 16 * 1024;
const CHAT_FRAGMENT_BYTES: usize = 64;
const DECISION_PART_BYTES: usize = 128;
const MAX_DECISION_PARTS: usize = MAX_DECISION_JSON_BYTES.div_ceil(DECISION_PART_BYTES - 3);
const EMBEDDING_PART_BYTES: usize = 512;
const MAX_EMBEDDING_PARTS: usize = MAX_EMBEDDING_JSON_BYTES.div_ceil(EMBEDDING_PART_BYTES);

/// Fixed redacted failures from the in-memory result buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultBufferError {
    InvalidRequestId,
    DuplicateRequest,
    UnknownRequest,
    ExpiredRequest,
    CapacityExceeded,
    WrongLane,
    AlreadyTerminal,
    NotTerminal,
    EventTooLarge,
    InvalidArtifactPath,
    InvalidEmbedding,
    InvalidDecisionMetadata,
    InvalidChatEvent,
    ReservationRequired,
    InvalidCursor,
    InvalidPageBudget,
    PageBudgetTooSmall,
    Serialization,
}

impl fmt::Display for ResultBufferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequestId => "result request ID is invalid",
            Self::DuplicateRequest => "result request ID is already retained",
            Self::UnknownRequest => "result request is unknown",
            Self::ExpiredRequest => "result request result has expired",
            Self::CapacityExceeded => "result buffer capacity exceeded",
            Self::WrongLane => "result event does not match its native lane",
            Self::AlreadyTerminal => "result request is already terminal",
            Self::NotTerminal => "only terminal results can be evicted",
            Self::EventTooLarge => "result event exceeds the bounded result limit",
            Self::InvalidArtifactPath => "media artifact reference is invalid",
            Self::InvalidEmbedding => "embedding result is invalid or too large",
            Self::InvalidDecisionMetadata => "native decision metadata is invalid or too large",
            Self::InvalidChatEvent => "chat stream event sequence is invalid",
            Self::ReservationRequired => "chat streaming requires a reserved result slot",
            Self::InvalidCursor => "result cursor is invalid",
            Self::InvalidPageBudget => "result page byte budget is invalid",
            Self::PageBudgetTooSmall => "result page budget cannot hold the next event",
            Self::Serialization => "result page serialization failed",
        })
    }
}

impl std::error::Error for ResultBufferError {}

/// The stable wire availability state for a result lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Availability {
    Available,
    Expired,
    Unknown,
}

/// Safe terminal metadata; it never carries native error text or paths.
#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "status",
    content = "code",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ResultTerminal {
    Succeeded,
    Failed(ResultFailureCode),
    Cancelled,
    Interrupted,
}

impl fmt::Debug for ResultTerminal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Succeeded => formatter.write_str("Succeeded"),
            Self::Failed(code) => formatter.debug_tuple("Failed").field(code).finish(),
            Self::Cancelled => formatter.write_str("Cancelled"),
            Self::Interrupted => formatter.write_str("Interrupted"),
        }
    }
}

/// Bounded stable error categories for terminal result metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ResultFailureCode {
    Unavailable,
    TimedOut,
    InvalidOutput,
    ResourceFailure,
    Internal,
}

/// Media output kind remains explicit across storage and IPC retrieval.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactKind {
    LocalJob,
    Backend,
}

/// A path relative to its declared native artifact root, never an absolute
/// root or URL. `append` and [`ResultPage::validate`] recheck this value.
#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaArtifact {
    pub kind: ArtifactKind,
    pub relative_path: String,
}

impl MediaArtifact {
    pub fn new(
        kind: ArtifactKind,
        relative_path: impl Into<String>,
    ) -> Result<Self, ResultBufferError> {
        let artifact = Self {
            kind,
            relative_path: relative_path.into(),
        };
        validate_relative_artifact_path(&artifact.relative_path)?;
        Ok(artifact)
    }
}

impl fmt::Debug for MediaArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaArtifact")
            .field("kind", &self.kind)
            .field("relative_path", &"<redacted>")
            .finish()
    }
}

/// Native lane-specific result data accepted by [`ResultBuffer::append`].
/// These input values are converted into bounded page chunks before commit.
#[derive(Clone)]
pub enum ResultEvent {
    ChatContent(String),
    ChatReasoning(String),
    ChatUsage {
        completion_tokens: u64,
    },
    ChatStreamComplete,
    Embeddings {
        vectors: Vec<Vec<f32>>,
        usage: EmbeddingUsage,
    },
    DecisionAnswers(Map<String, Value>),
    /// Atomically append one native decision answer object and its optional
    /// provider-supplied metadata. Existing `DecisionAnswers` callers retain
    /// their exact event and wire format.
    DecisionResult {
        answers: Map<String, Value>,
        native_latency: Option<Duration>,
        native_usage: Option<Map<String, Value>>,
    },
    MediaArtifacts(Vec<MediaArtifact>),
}

impl fmt::Debug for ResultEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatContent(text) => formatter
                .debug_struct("ChatContent")
                .field("bytes", &text.len())
                .field("text", &"<redacted>")
                .finish(),
            Self::ChatReasoning(text) => formatter
                .debug_struct("ChatReasoning")
                .field("bytes", &text.len())
                .field("text", &"<redacted>")
                .finish(),
            Self::ChatUsage { completion_tokens } => formatter
                .debug_struct("ChatUsage")
                .field("completion_tokens", completion_tokens)
                .finish(),
            Self::ChatStreamComplete => formatter.write_str("ChatStreamComplete"),
            Self::Embeddings { vectors, usage } => formatter
                .debug_struct("Embeddings")
                .field("vector_count", &vectors.len())
                .field("dimensions", &vectors.first().map(Vec::len))
                .field("prompt_tokens", &usage.prompt_tokens)
                .field("total_tokens", &usage.total_tokens)
                .field("vectors", &"<redacted>")
                .finish(),
            Self::DecisionAnswers(answers) => formatter
                .debug_struct("DecisionAnswers")
                .field("answer_count", &answers.len())
                .field("answers", &"<redacted>")
                .finish(),
            Self::DecisionResult {
                answers,
                native_latency,
                native_usage,
            } => formatter
                .debug_struct("DecisionResult")
                .field("answer_count", &answers.len())
                .field("native_latency", native_latency)
                .field("has_native_usage", &native_usage.is_some())
                .field("answers", &"<redacted>")
                .field("native_usage", &"<redacted>")
                .finish(),
            Self::MediaArtifacts(artifacts) => formatter
                .debug_struct("MediaArtifacts")
                .field("artifact_count", &artifacts.len())
                .field("artifacts", &"<redacted>")
                .finish(),
        }
    }
}

/// Serialized page event. Text and decision outputs are split to fit the
/// minimum page budget; a long media path can require a larger requested page.
#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PageEvent {
    ChatContent {
        text: String,
    },
    ChatReasoning {
        text: String,
    },
    ChatUsage {
        completion_tokens: u64,
    },
    /// Native stream completion only. This is not the coordinator's terminal
    /// job state and carries no lifecycle/GPU-release meaning.
    ChatStreamComplete,
    EmbeddingJsonPart {
        result_id: u64,
        part_index: u32,
        part_count: u32,
        json: String,
    },
    DecisionJsonPart {
        result_id: u64,
        part_index: u32,
        part_count: u32,
        json: String,
    },
    /// Optional native decision metadata, separate from the unchanged answer
    /// object parts. The JSON payload is chunked with the same page bounds.
    DecisionMetadataJsonPart {
        result_id: u64,
        part_index: u32,
        part_count: u32,
        json: String,
    },
    MediaArtifactBatch {
        result_id: u64,
        batch_index: u32,
        batch_count: u32,
        artifacts: Vec<MediaArtifact>,
    },
}

impl fmt::Debug for PageEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatContent { text } => formatter
                .debug_struct("ChatContent")
                .field("bytes", &text.len())
                .field("text", &"<redacted>")
                .finish(),
            Self::ChatReasoning { text } => formatter
                .debug_struct("ChatReasoning")
                .field("bytes", &text.len())
                .field("text", &"<redacted>")
                .finish(),
            Self::ChatUsage { completion_tokens } => formatter
                .debug_struct("ChatUsage")
                .field("completion_tokens", completion_tokens)
                .finish(),
            Self::ChatStreamComplete => formatter.write_str("ChatStreamComplete"),
            Self::EmbeddingJsonPart {
                result_id,
                part_index,
                part_count,
                json,
            } => formatter
                .debug_struct("EmbeddingJsonPart")
                .field("result_id", result_id)
                .field("part_index", part_index)
                .field("part_count", part_count)
                .field("json_bytes", &json.len())
                .field("json", &"<redacted>")
                .finish(),
            Self::DecisionJsonPart {
                result_id,
                part_index,
                part_count,
                json,
            } => formatter
                .debug_struct("DecisionJsonPart")
                .field("result_id", result_id)
                .field("part_index", part_index)
                .field("part_count", part_count)
                .field("json_bytes", &json.len())
                .field("json", &"<redacted>")
                .finish(),
            Self::DecisionMetadataJsonPart {
                result_id,
                part_index,
                part_count,
                json,
            } => formatter
                .debug_struct("DecisionMetadataJsonPart")
                .field("result_id", result_id)
                .field("part_index", part_index)
                .field("part_count", part_count)
                .field("json_bytes", &json.len())
                .field("json", &"<redacted>")
                .finish(),
            Self::MediaArtifactBatch {
                result_id,
                batch_index,
                batch_count,
                artifacts,
            } => formatter
                .debug_struct("MediaArtifactBatch")
                .field("result_id", result_id)
                .field("batch_index", batch_index)
                .field("batch_count", batch_count)
                .field("artifact_count", &artifacts.len())
                .field("artifacts", &"<redacted>")
                .finish(),
        }
    }
}

/// One immutable cursor-addressed event in a serialized result page.
#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CursorEvent {
    pub cursor: u64,
    pub event: PageEvent,
}

impl fmt::Debug for CursorEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CursorEvent")
            .field("cursor", &self.cursor)
            .field("event", &"<redacted>")
            .finish()
    }
}

/// A bounded serialized response. Repeating the same `after_cursor` is
/// read-only and returns the same events unless new events were appended.
#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultPage {
    pub request_id: String,
    pub availability: Availability,
    pub events: Vec<CursorEvent>,
    pub next_cursor: u64,
    pub latest_cursor: u64,
    pub terminal: Option<ResultTerminal>,
}

impl fmt::Debug for ResultPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResultPage")
            .field("request_id", &"<redacted>")
            .field("availability", &self.availability)
            .field("event_count", &self.events.len())
            .field("next_cursor", &self.next_cursor)
            .field("latest_cursor", &self.latest_cursor)
            .field("terminal", &self.terminal)
            .finish()
    }
}

impl ResultPage {
    /// Validates an untrusted decoded page before client code consumes it.
    /// Deserialization alone does not establish cursor, part, or path safety.
    pub fn validate(&self) -> Result<(), ResultBufferError> {
        validate_id(&self.request_id)?;
        if serde_json::to_vec(self)
            .map_err(|_| ResultBufferError::Serialization)?
            .len()
            > MAX_PAGE_BYTES
        {
            return Err(ResultBufferError::EventTooLarge);
        }
        let mut decision_parts: HashMap<u64, (u32, HashMap<u32, u64>)> = HashMap::new();
        let mut decision_metadata_parts: HashMap<u64, (u32, HashMap<u32, u64>)> = HashMap::new();
        let mut embedding_parts: HashMap<u64, (u32, HashMap<u32, u64>)> = HashMap::new();
        let mut artifact_batches: HashMap<u64, (u32, HashMap<u32, u64>)> = HashMap::new();
        let mut native_lane = None;
        let mut language_task = None;
        match self.availability {
            Availability::Unknown => {
                if !self.events.is_empty()
                    || self.next_cursor != 0
                    || self.latest_cursor != 0
                    || self.terminal.is_some()
                {
                    return Err(ResultBufferError::InvalidCursor);
                }
            }
            Availability::Expired => {
                if !self.events.is_empty()
                    || self.next_cursor != self.latest_cursor
                    || self.terminal.is_none()
                {
                    return Err(ResultBufferError::InvalidCursor);
                }
            }
            Availability::Available => {
                if self.next_cursor > self.latest_cursor {
                    return Err(ResultBufferError::InvalidCursor);
                }
                let mut previous = 0;
                let mut chat_usage_seen = false;
                let mut chat_stream_complete = false;
                for item in &self.events {
                    let lane = match &item.event {
                        PageEvent::ChatContent { .. }
                        | PageEvent::ChatReasoning { .. }
                        | PageEvent::ChatUsage { .. }
                        | PageEvent::ChatStreamComplete
                        | PageEvent::EmbeddingJsonPart { .. } => Lane::Language,
                        PageEvent::DecisionJsonPart { .. }
                        | PageEvent::DecisionMetadataJsonPart { .. } => Lane::Decisions,
                        PageEvent::MediaArtifactBatch { .. } => Lane::Media,
                    };
                    if native_lane.is_some_and(|expected| expected != lane) {
                        return Err(ResultBufferError::WrongLane);
                    }
                    native_lane = Some(lane);
                    if lane == Lane::Language {
                        let is_embedding =
                            matches!(&item.event, PageEvent::EmbeddingJsonPart { .. });
                        if language_task.is_some_and(|expected| expected != is_embedding) {
                            return Err(ResultBufferError::WrongLane);
                        }
                        language_task = Some(is_embedding);
                    }
                    if item.cursor <= previous
                        || (previous != 0 && item.cursor != previous + 1)
                        || item.cursor > self.next_cursor
                    {
                        return Err(ResultBufferError::InvalidCursor);
                    }
                    match &item.event {
                        PageEvent::ChatContent { .. } | PageEvent::ChatReasoning { .. } => {
                            if chat_stream_complete {
                                return Err(ResultBufferError::InvalidChatEvent);
                            }
                        }
                        PageEvent::ChatUsage { .. } => {
                            if chat_stream_complete || chat_usage_seen {
                                return Err(ResultBufferError::InvalidChatEvent);
                            }
                            chat_usage_seen = true;
                        }
                        PageEvent::ChatStreamComplete => {
                            if chat_stream_complete {
                                return Err(ResultBufferError::InvalidChatEvent);
                            }
                            chat_stream_complete = true;
                        }
                        PageEvent::DecisionJsonPart {
                            result_id,
                            part_index,
                            part_count,
                            ..
                        } => {
                            validate_group_part(
                                &mut decision_parts,
                                *result_id,
                                *part_index,
                                *part_count,
                                item.cursor,
                                MAX_DECISION_PARTS as u32,
                            )?;
                        }
                        PageEvent::DecisionMetadataJsonPart {
                            result_id,
                            part_index,
                            part_count,
                            ..
                        } => {
                            validate_group_part(
                                &mut decision_metadata_parts,
                                *result_id,
                                *part_index,
                                *part_count,
                                item.cursor,
                                MAX_DECISION_PARTS as u32,
                            )?;
                        }
                        PageEvent::EmbeddingJsonPart {
                            result_id,
                            part_index,
                            part_count,
                            ..
                        } => {
                            validate_group_part(
                                &mut embedding_parts,
                                *result_id,
                                *part_index,
                                *part_count,
                                item.cursor,
                                MAX_EMBEDDING_PARTS as u32,
                            )?;
                        }
                        PageEvent::MediaArtifactBatch {
                            result_id,
                            batch_index,
                            batch_count,
                            ..
                        } => {
                            validate_group_part(
                                &mut artifact_batches,
                                *result_id,
                                *batch_index,
                                *batch_count,
                                item.cursor,
                                MAX_ARTIFACTS_PER_APPEND as u32,
                            )?;
                        }
                    }
                    previous = item.cursor;
                }
                if let Some(last) = self.events.last() {
                    if last.cursor != self.next_cursor {
                        return Err(ResultBufferError::InvalidCursor);
                    }
                }
            }
        }
        for item in &self.events {
            validate_page_event(item)?;
        }
        Ok(())
    }
}

/// One-coordinator in-memory result/event retention. All mutation is through
/// `&mut self`; reading a page never advances a cursor.
pub struct ResultBuffer {
    jobs: HashMap<String, ResultEntry>,
    total_event_bytes: usize,
    total_reserved_bytes: usize,
    tombstones: HashMap<String, Tombstone>,
    tombstone_order: VecDeque<String>,
}

impl fmt::Debug for ResultBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResultBuffer")
            .field("retained_jobs", &self.jobs.len())
            .field("retained_tombstones", &self.tombstones.len())
            .field("event_bytes", &self.total_event_bytes)
            .field("reserved_bytes", &self.total_reserved_bytes)
            .finish()
    }
}

impl Default for ResultBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultBuffer {
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            total_event_bytes: 0,
            total_reserved_bytes: 0,
            tombstones: HashMap::new(),
            tombstone_order: VecDeque::new(),
        }
    }

    /// Begins one request result. IDs remain unavailable for reuse while a
    /// retained job or bounded tombstone carries that ID.
    pub fn begin(&mut self, request_id: &str, lane: Lane) -> Result<(), ResultBufferError> {
        self.begin_with_budget(request_id, lane, 0)
    }

    /// Read-only acceptance check for a full per-job output reservation. The
    /// single daemon owner must call begin_reserved after durable acceptance,
    /// without intervening mutation. No cache entry or tombstone is created.
    pub fn can_begin_reserved(&self, request_id: &str) -> Result<(), ResultBufferError> {
        self.check_begin(request_id, MAX_JOB_EVENT_BYTES)
    }

    /// Reserve the full 8 MiB serialized-event allowance before dispatch.
    /// Appends consume this reservation; terminal acknowledgment frees only
    /// unused quota. Retained output still counts against the global cap.
    pub fn begin_reserved(
        &mut self,
        request_id: &str,
        lane: Lane,
    ) -> Result<(), ResultBufferError> {
        self.begin_with_budget(request_id, lane, MAX_JOB_EVENT_BYTES)
    }

    /// Read-only preflight for the incremental language chat publisher. The
    /// caller must already have durably accepted the request and created its
    /// reserved result slot; this check neither admits work nor changes quota.
    pub fn can_append_reserved_chat_stream(
        &self,
        request_id: &str,
    ) -> Result<(), ResultBufferError> {
        validate_id(request_id)?;
        let job = self.job(request_id)?;
        if job.lane != Lane::Language {
            return Err(ResultBufferError::WrongLane);
        }
        if job.terminal.is_some() {
            return Err(ResultBufferError::AlreadyTerminal);
        }
        if job.latest_cursor != 0
            || job.language_kind.is_some()
            || job.chat_usage_seen
            || job.chat_stream_complete
        {
            return Err(ResultBufferError::InvalidChatEvent);
        }
        if job.reserved_bytes != MAX_JOB_EVENT_BYTES {
            return Err(ResultBufferError::ReservationRequired);
        }
        Ok(())
    }

    fn check_begin(&self, request_id: &str, reserved: usize) -> Result<(), ResultBufferError> {
        validate_id(request_id)?;
        if self.jobs.contains_key(request_id) || self.tombstones.contains_key(request_id) {
            return Err(ResultBufferError::DuplicateRequest);
        }
        if self.jobs.len() >= MAX_RETAINED_JOBS {
            return Err(ResultBufferError::CapacityExceeded);
        }
        if self
            .total_event_bytes
            .checked_add(self.total_reserved_bytes)
            .and_then(|bytes| bytes.checked_add(reserved))
            .is_none_or(|bytes| bytes > MAX_TOTAL_EVENT_BYTES)
        {
            return Err(ResultBufferError::CapacityExceeded);
        }
        Ok(())
    }

    fn begin_with_budget(
        &mut self,
        request_id: &str,
        lane: Lane,
        reserved: usize,
    ) -> Result<(), ResultBufferError> {
        self.check_begin(request_id, reserved)?;
        let mut entry = ResultEntry::new(lane);
        entry.reserved_bytes = reserved;
        self.jobs.insert(request_id.to_owned(), entry);
        self.total_reserved_bytes += reserved;
        Ok(())
    }

    /// Appends a native lane event transactionally. Chat content and reasoning
    /// share a 1 MiB raw per-job cap. Oversized or over-cap input leaves every
    /// cursor and retained byte count unchanged.
    pub fn append(
        &mut self,
        request_id: &str,
        event: ResultEvent,
    ) -> Result<(), ResultBufferError> {
        validate_id(request_id)?;
        let (
            lane,
            language_kind,
            latest_cursor,
            event_bytes,
            chat_raw_bytes,
            terminal,
            reserved_bytes,
            chat_usage_seen,
            chat_stream_complete,
            decision_metadata_seen,
        ) = {
            let job = self.job(request_id)?;
            (
                job.lane,
                job.language_kind,
                job.latest_cursor,
                job.event_bytes,
                job.chat_raw_bytes,
                job.terminal,
                job.reserved_bytes,
                job.chat_usage_seen,
                job.chat_stream_complete,
                job.decision_metadata_seen,
            )
        };
        if terminal.is_some() {
            return Err(ResultBufferError::AlreadyTerminal);
        }
        let event_lane = event.lane();
        if lane != event_lane {
            return Err(ResultBufferError::WrongLane);
        }
        if decision_metadata_seen
            && matches!(
                &event,
                ResultEvent::DecisionAnswers(_) | ResultEvent::DecisionResult { .. }
            )
        {
            return Err(ResultBufferError::InvalidDecisionMetadata);
        }
        let incoming_language_kind = event.language_kind();
        if matches!(
            (language_kind, incoming_language_kind),
            (Some(LanguageResultKind::Embeddings), Some(_))
                | (
                    Some(LanguageResultKind::Chat),
                    Some(LanguageResultKind::Embeddings)
                )
        ) {
            return Err(ResultBufferError::WrongLane);
        }
        if incoming_language_kind == Some(LanguageResultKind::Chat)
            && (chat_stream_complete
                || (matches!(&event, ResultEvent::ChatUsage { .. }) && chat_usage_seen))
        {
            return Err(ResultBufferError::InvalidChatEvent);
        }
        let chat_usage_event = matches!(&event, ResultEvent::ChatUsage { .. });
        let chat_stream_complete_event = matches!(&event, ResultEvent::ChatStreamComplete);
        let decision_metadata_event = event.has_native_decision_metadata();
        let added_chat_bytes = event.raw_chat_bytes();
        let new_chat_raw_bytes = chat_raw_bytes
            .checked_add(added_chat_bytes)
            .filter(|bytes| *bytes <= MAX_CHAT_RAW_BYTES_PER_JOB)
            .ok_or(ResultBufferError::EventTooLarge)?;
        let first_cursor = latest_cursor
            .checked_add(1)
            .ok_or(ResultBufferError::CapacityExceeded)?;
        let staged = prepare_events(request_id, first_cursor, event)?;
        if staged.is_empty() {
            return Err(ResultBufferError::Serialization);
        }

        let mut staged_bytes = 0usize;
        let mut staged_records = Vec::with_capacity(staged.len());
        for (offset, event) in staged.iter().enumerate() {
            let cursor = first_cursor
                .checked_add(offset as u64)
                .ok_or(ResultBufferError::CapacityExceeded)?;
            let fits = if matches!(event, PageEvent::MediaArtifactBatch { .. }) {
                page_event_fits_maximum(request_id, cursor, event)?
            } else {
                page_event_fits_minimum(request_id, cursor, event)?
            };
            if !fits {
                return Err(ResultBufferError::PageBudgetTooSmall);
            }
            let event_bytes = serialized_event_bytes(cursor, event)?;
            staged_bytes = staged_bytes
                .checked_add(event_bytes)
                .ok_or(ResultBufferError::CapacityExceeded)?;
            staged_records.push(StoredResultEvent {
                record: CursorEvent {
                    cursor,
                    event: event.clone(),
                },
                encoded_bytes: event_bytes,
            });
        }
        if event_bytes
            .checked_add(staged_bytes)
            .filter(|bytes| *bytes <= MAX_JOB_EVENT_BYTES)
            .is_none()
            || self
                .total_event_bytes
                .checked_add(self.total_reserved_bytes)
                .and_then(|bytes| bytes.checked_add(staged_bytes))
                .and_then(|bytes| bytes.checked_sub(staged_bytes.min(reserved_bytes)))
                .filter(|bytes| *bytes <= MAX_TOTAL_EVENT_BYTES)
                .is_none()
        {
            return Err(ResultBufferError::CapacityExceeded);
        }

        let staged_count = staged_records.len() as u64;
        let last_cursor = first_cursor
            .checked_add(staged_count - 1)
            .ok_or(ResultBufferError::CapacityExceeded)?;
        let job = self
            .jobs
            .get_mut(request_id)
            .ok_or(ResultBufferError::UnknownRequest)?;
        job.events.extend(staged_records);
        if incoming_language_kind.is_some() {
            job.language_kind = incoming_language_kind;
        }
        if chat_usage_event {
            job.chat_usage_seen = true;
        }
        if chat_stream_complete_event {
            job.chat_stream_complete = true;
        }
        if decision_metadata_event {
            job.decision_metadata_seen = true;
        }
        if decision_metadata_event {
            job.decision_metadata_seen = true;
        }
        job.latest_cursor = last_cursor;
        job.event_bytes += staged_bytes;
        job.chat_raw_bytes = new_chat_raw_bytes;
        let consumed_reservation = staged_bytes.min(job.reserved_bytes);
        job.reserved_bytes -= consumed_reservation;
        self.total_reserved_bytes -= consumed_reservation;
        self.total_event_bytes += staged_bytes;
        Ok(())
    }

    /// Marks a result terminal exactly once. Terminal metadata is fixed and
    /// contains no free-form native error string, body, or path.
    pub fn terminal(
        &mut self,
        request_id: &str,
        terminal: ResultTerminal,
    ) -> Result<(), ResultBufferError> {
        validate_id(request_id)?;
        let freed = {
            let job = self.job_mut(request_id)?;
            if job.terminal.is_some() {
                return Err(ResultBufferError::AlreadyTerminal);
            }
            job.terminal = Some(terminal);
            let freed = job.reserved_bytes;
            job.reserved_bytes = 0;
            freed
        };
        self.total_reserved_bytes -= freed;
        Ok(())
    }

    /// Retention state without copying events or exposing their payloads.
    #[cfg(target_os = "linux")]
    pub(crate) fn availability(&self, request_id: &str) -> Result<Availability, ResultBufferError> {
        validate_id(request_id)?;
        Ok(if self.jobs.contains_key(request_id) {
            Availability::Available
        } else if self.tombstones.contains_key(request_id) {
            Availability::Expired
        } else {
            Availability::Unknown
        })
    }

    /// Returns an immutable bounded page. A repeated cursor returns the same
    /// events and is safe to retry after a lost acknowledgement.
    pub fn page(
        &self,
        request_id: &str,
        after_cursor: u64,
        byte_budget: usize,
    ) -> Result<ResultPage, ResultBufferError> {
        validate_id(request_id)?;
        if !(MIN_PAGE_BYTES..=MAX_PAGE_BYTES).contains(&byte_budget) {
            return Err(ResultBufferError::InvalidPageBudget);
        }
        let Some(job) = self.jobs.get(request_id) else {
            if let Some(tombstone) = self.tombstones.get(request_id) {
                if after_cursor > tombstone.latest_cursor {
                    return Err(ResultBufferError::InvalidCursor);
                }
                return make_page(
                    request_id,
                    Availability::Expired,
                    Vec::new(),
                    tombstone.latest_cursor,
                    tombstone.latest_cursor,
                    Some(tombstone.terminal),
                    byte_budget,
                );
            }
            return make_page(
                request_id,
                Availability::Unknown,
                Vec::new(),
                0,
                0,
                None,
                byte_budget,
            );
        };
        if after_cursor > job.latest_cursor {
            return Err(ResultBufferError::InvalidCursor);
        }

        let mut page = ResultPage {
            request_id: request_id.to_owned(),
            availability: Availability::Available,
            events: Vec::new(),
            next_cursor: after_cursor,
            latest_cursor: job.latest_cursor,
            terminal: job.terminal,
        };
        let base_bytes = serialized_page_bytes(&page)?;
        if base_bytes > byte_budget {
            return Err(ResultBufferError::PageBudgetTooSmall);
        }
        let base_cursor_digits = cursor_digits(after_cursor);
        let mut included_event_bytes = 0usize;
        for stored in job
            .events
            .iter()
            .filter(|event| event.record.cursor > after_cursor)
        {
            let candidate_bytes = base_bytes
                .checked_sub(base_cursor_digits)
                .and_then(|bytes| bytes.checked_add(cursor_digits(stored.record.cursor)))
                .and_then(|bytes| bytes.checked_add(included_event_bytes))
                .and_then(|bytes| bytes.checked_add(stored.encoded_bytes))
                .and_then(|bytes| bytes.checked_add(page.events.len()))
                .ok_or(ResultBufferError::Serialization)?;
            if candidate_bytes > byte_budget {
                if page.events.is_empty() {
                    return Err(ResultBufferError::PageBudgetTooSmall);
                }
                break;
            }
            included_event_bytes += stored.encoded_bytes;
            page.events.push(stored.record.clone());
            page.next_cursor = stored.record.cursor;
        }
        if serialized_page_bytes(&page)? > byte_budget {
            return Err(ResultBufferError::PageBudgetTooSmall);
        }
        Ok(page)
    }

    /// Explicitly removes a terminal payload and retains a bounded tombstone.
    /// Active jobs are never evicted. Old tombstones age out after 256 later
    /// evictions; durable acceptance/history owns replay prevention beyond it.
    pub fn evict_terminal(&mut self, request_id: &str) -> Result<(), ResultBufferError> {
        validate_id(request_id)?;
        let Some(job) = self.jobs.get(request_id) else {
            return if self.tombstones.contains_key(request_id) {
                Err(ResultBufferError::ExpiredRequest)
            } else {
                Err(ResultBufferError::UnknownRequest)
            };
        };
        let Some(terminal) = job.terminal else {
            return Err(ResultBufferError::NotTerminal);
        };
        let latest_cursor = job.latest_cursor;
        let removed = self
            .jobs
            .remove(request_id)
            .ok_or(ResultBufferError::UnknownRequest)?;
        self.total_event_bytes = self.total_event_bytes.saturating_sub(removed.event_bytes);
        self.tombstones.insert(
            request_id.to_owned(),
            Tombstone {
                terminal,
                latest_cursor,
            },
        );
        self.tombstone_order.push_back(request_id.to_owned());
        while self.tombstone_order.len() > MAX_TOMBSTONES {
            if let Some(expired) = self.tombstone_order.pop_front() {
                self.tombstones.remove(&expired);
            }
        }
        Ok(())
    }

    fn job(&self, request_id: &str) -> Result<&ResultEntry, ResultBufferError> {
        self.jobs.get(request_id).ok_or_else(|| {
            if self.tombstones.contains_key(request_id) {
                ResultBufferError::ExpiredRequest
            } else {
                ResultBufferError::UnknownRequest
            }
        })
    }

    fn job_mut(&mut self, request_id: &str) -> Result<&mut ResultEntry, ResultBufferError> {
        if self.jobs.contains_key(request_id) {
            return self
                .jobs
                .get_mut(request_id)
                .ok_or(ResultBufferError::UnknownRequest);
        }
        if self.tombstones.contains_key(request_id) {
            Err(ResultBufferError::ExpiredRequest)
        } else {
            Err(ResultBufferError::UnknownRequest)
        }
    }
}

struct ResultEntry {
    reserved_bytes: usize,
    lane: Lane,
    language_kind: Option<LanguageResultKind>,
    events: Vec<StoredResultEvent>,
    latest_cursor: u64,
    event_bytes: usize,
    chat_raw_bytes: usize,
    chat_usage_seen: bool,
    chat_stream_complete: bool,
    decision_metadata_seen: bool,
    terminal: Option<ResultTerminal>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum LanguageResultKind {
    Chat,
    Embeddings,
}

struct StoredResultEvent {
    record: CursorEvent,
    encoded_bytes: usize,
}

impl ResultEntry {
    fn new(lane: Lane) -> Self {
        Self {
            reserved_bytes: 0,
            lane,
            language_kind: None,
            events: Vec::new(),
            latest_cursor: 0,
            event_bytes: 0,
            chat_raw_bytes: 0,
            chat_usage_seen: false,
            chat_stream_complete: false,
            decision_metadata_seen: false,
            terminal: None,
        }
    }
}

#[derive(Clone, Copy)]
struct Tombstone {
    terminal: ResultTerminal,
    latest_cursor: u64,
}

impl ResultEvent {
    fn language_kind(&self) -> Option<LanguageResultKind> {
        match self {
            Self::ChatContent(_)
            | Self::ChatReasoning(_)
            | Self::ChatUsage { .. }
            | Self::ChatStreamComplete => Some(LanguageResultKind::Chat),
            Self::Embeddings { .. } => Some(LanguageResultKind::Embeddings),
            Self::DecisionAnswers(_) | Self::DecisionResult { .. } | Self::MediaArtifacts(_) => {
                None
            }
        }
    }

    fn lane(&self) -> Lane {
        match self {
            Self::ChatContent(_)
            | Self::ChatReasoning(_)
            | Self::ChatUsage { .. }
            | Self::ChatStreamComplete
            | Self::Embeddings { .. } => Lane::Language,
            Self::DecisionAnswers(_) | Self::DecisionResult { .. } => Lane::Decisions,
            Self::MediaArtifacts(_) => Lane::Media,
        }
    }

    fn raw_chat_bytes(&self) -> usize {
        match self {
            Self::ChatContent(text) | Self::ChatReasoning(text) => text.len(),
            Self::ChatUsage { .. }
            | Self::ChatStreamComplete
            | Self::Embeddings { .. }
            | Self::DecisionAnswers(_)
            | Self::DecisionResult { .. }
            | Self::MediaArtifacts(_) => 0,
        }
    }

    fn has_native_decision_metadata(&self) -> bool {
        matches!(
            self,
            Self::DecisionResult {
                native_latency: Some(_),
                ..
            } | Self::DecisionResult {
                native_usage: Some(_),
                ..
            }
        )
    }
}

fn prepare_events(
    request_id: &str,
    first_cursor: u64,
    event: ResultEvent,
) -> Result<Vec<PageEvent>, ResultBufferError> {
    match event {
        ResultEvent::ChatContent(text) => Ok(split_text(&text, CHAT_FRAGMENT_BYTES)
            .into_iter()
            .map(|text| PageEvent::ChatContent { text })
            .collect()),
        ResultEvent::ChatReasoning(text) => Ok(split_text(&text, CHAT_FRAGMENT_BYTES)
            .into_iter()
            .map(|text| PageEvent::ChatReasoning { text })
            .collect()),
        ResultEvent::ChatUsage { completion_tokens } => {
            Ok(vec![PageEvent::ChatUsage { completion_tokens }])
        }
        ResultEvent::ChatStreamComplete => Ok(vec![PageEvent::ChatStreamComplete]),
        ResultEvent::Embeddings { vectors, usage } => {
            let json = serialize_embedding_payload(&vectors, &usage)
                .map_err(|_| ResultBufferError::InvalidEmbedding)?;
            let json = String::from_utf8(json).map_err(|_| ResultBufferError::Serialization)?;
            let parts = split_text(&json, EMBEDDING_PART_BYTES);
            let part_count =
                u32::try_from(parts.len()).map_err(|_| ResultBufferError::EventTooLarge)?;
            Ok(parts
                .into_iter()
                .enumerate()
                .map(|(part_index, json)| PageEvent::EmbeddingJsonPart {
                    result_id: first_cursor,
                    part_index: part_index as u32,
                    part_count,
                    json,
                })
                .collect())
        }
        ResultEvent::DecisionAnswers(answers) => {
            prepare_decision_answer_events(first_cursor, answers)
        }
        ResultEvent::DecisionResult {
            answers,
            native_latency,
            native_usage,
        } => {
            let answers_json = serialize_decision_answers(&answers)?;
            let metadata_json = serialize_decision_metadata(native_latency, native_usage)?;
            if answers_json
                .len()
                .checked_add(metadata_json.as_ref().map_or(0, String::len))
                .is_none_or(|bytes| bytes > MAX_DECISION_JSON_BYTES)
            {
                return Err(ResultBufferError::InvalidDecisionMetadata);
            }
            let mut events = prepare_decision_answer_json(first_cursor, &answers_json)?;
            if let Some(metadata_json) = metadata_json {
                let metadata_cursor = first_cursor
                    .checked_add(events.len() as u64)
                    .ok_or(ResultBufferError::CapacityExceeded)?;
                events.extend(prepare_decision_metadata_json(
                    metadata_cursor,
                    &metadata_json,
                )?);
            }
            Ok(events)
        }
        ResultEvent::MediaArtifacts(artifacts) => {
            if artifacts.len() > MAX_ARTIFACTS_PER_APPEND || artifacts.len() > u32::MAX as usize {
                return Err(ResultBufferError::EventTooLarge);
            }
            for artifact in &artifacts {
                validate_relative_artifact_path(&artifact.relative_path)?;
            }
            prepare_artifact_events(request_id, first_cursor, artifacts)
        }
    }
}

fn prepare_artifact_events(
    request_id: &str,
    first_cursor: u64,
    artifacts: Vec<MediaArtifact>,
) -> Result<Vec<PageEvent>, ResultBufferError> {
    let batch_count =
        u32::try_from(artifacts.len().max(1)).map_err(|_| ResultBufferError::EventTooLarge)?;
    let mut events = Vec::with_capacity(artifacts.len().max(1));
    let mut encoded_total = 0usize;
    if artifacts.is_empty() {
        let event = PageEvent::MediaArtifactBatch {
            result_id: first_cursor,
            batch_index: 0,
            batch_count: 1,
            artifacts: Vec::new(),
        };
        if !page_event_fits_maximum(request_id, first_cursor, &event)? {
            return Err(ResultBufferError::PageBudgetTooSmall);
        }
        return Ok(vec![event]);
    }
    for (index, artifact) in artifacts.into_iter().enumerate() {
        let cursor = first_cursor
            .checked_add(index as u64)
            .ok_or(ResultBufferError::CapacityExceeded)?;
        let event = PageEvent::MediaArtifactBatch {
            result_id: first_cursor,
            batch_index: index as u32,
            batch_count,
            artifacts: vec![artifact],
        };
        if !page_event_fits_maximum(request_id, cursor, &event)? {
            return Err(ResultBufferError::EventTooLarge);
        }
        encoded_total = encoded_total
            .checked_add(serialized_event_bytes(cursor, &event)?)
            .filter(|bytes| *bytes <= MAX_JOB_EVENT_BYTES)
            .ok_or(ResultBufferError::EventTooLarge)?;
        events.push(event);
    }
    Ok(events)
}

fn serialize_decision_answers(answers: &Map<String, Value>) -> Result<String, ResultBufferError> {
    let mut bounded = LimitedJsonBuffer {
        bytes: Vec::new(),
        exceeded: false,
    };
    if serde_json::to_writer(&mut bounded, answers).is_err() {
        return Err(if bounded.exceeded {
            ResultBufferError::EventTooLarge
        } else {
            ResultBufferError::Serialization
        });
    }
    String::from_utf8(bounded.bytes).map_err(|_| ResultBufferError::Serialization)
}

fn serialize_decision_metadata(
    native_latency: Option<Duration>,
    native_usage: Option<Map<String, Value>>,
) -> Result<Option<String>, ResultBufferError> {
    let mut payload = Map::new();
    if let Some(latency) = native_latency {
        let nanos = u64::try_from(latency.as_nanos())
            .map_err(|_| ResultBufferError::InvalidDecisionMetadata)?;
        payload.insert("native_latency_ns".to_owned(), Value::from(nanos));
    }
    if let Some(usage) = native_usage {
        payload.insert("native_usage".to_owned(), Value::Object(usage));
    }
    if payload.is_empty() {
        return Ok(None);
    }
    serialize_decision_answers(&payload)
        .map(Some)
        .map_err(|error| match error {
            ResultBufferError::EventTooLarge => ResultBufferError::InvalidDecisionMetadata,
            other => other,
        })
}

fn prepare_decision_answer_events(
    first_cursor: u64,
    answers: Map<String, Value>,
) -> Result<Vec<PageEvent>, ResultBufferError> {
    let json = serialize_decision_answers(&answers)?;
    prepare_decision_answer_json(first_cursor, &json)
}

fn prepare_decision_answer_json(
    first_cursor: u64,
    json: &str,
) -> Result<Vec<PageEvent>, ResultBufferError> {
    let parts = split_text(json, DECISION_PART_BYTES);
    let part_count = u32::try_from(parts.len()).map_err(|_| ResultBufferError::EventTooLarge)?;
    Ok(parts
        .into_iter()
        .enumerate()
        .map(|(part_index, json)| PageEvent::DecisionJsonPart {
            result_id: first_cursor,
            part_index: part_index as u32,
            part_count,
            json,
        })
        .collect())
}

fn prepare_decision_metadata_json(
    first_cursor: u64,
    json: &str,
) -> Result<Vec<PageEvent>, ResultBufferError> {
    let parts = split_text(json, DECISION_PART_BYTES);
    let part_count = u32::try_from(parts.len()).map_err(|_| ResultBufferError::EventTooLarge)?;
    Ok(parts
        .into_iter()
        .enumerate()
        .map(|(part_index, json)| PageEvent::DecisionMetadataJsonPart {
            result_id: first_cursor,
            part_index: part_index as u32,
            part_count,
            json,
        })
        .collect())
}

struct LimitedJsonBuffer {
    bytes: Vec<u8>,
    exceeded: bool,
}

impl Write for LimitedJsonBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_DECISION_JSON_BYTES.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "bounded result exceeded",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn split_text(text: &str, max_bytes: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut result = Vec::with_capacity(text.len().div_ceil(max_bytes));
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + max_bytes).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        result.push(text[start..end].to_owned());
        start = end;
    }
    result
}

fn validate_id(request_id: &str) -> Result<(), ResultBufferError> {
    validate_request_id(request_id).map_err(|_| ResultBufferError::InvalidRequestId)
}

fn validate_relative_artifact_path(path: &str) -> Result<(), ResultBufferError> {
    if path.is_empty()
        || path.len() > MAX_ARTIFACT_PATH_BYTES
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains(':')
        || path.chars().any(char::is_control)
    {
        return Err(ResultBufferError::InvalidArtifactPath);
    }
    if path
        .split('/')
        .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(ResultBufferError::InvalidArtifactPath);
    }
    Ok(())
}

fn serialized_event_bytes(cursor: u64, event: &PageEvent) -> Result<usize, ResultBufferError> {
    serde_json::to_vec(&CursorEventRef { cursor, event })
        .map(|bytes| bytes.len())
        .map_err(|_| ResultBufferError::Serialization)
}

fn cursor_digits(cursor: u64) -> usize {
    cursor.to_string().len()
}

#[derive(Serialize)]
struct CursorEventRef<'a> {
    cursor: u64,
    event: &'a PageEvent,
}

fn page_event_fits_minimum(
    request_id: &str,
    cursor: u64,
    event: &PageEvent,
) -> Result<bool, ResultBufferError> {
    let page = ResultPage {
        request_id: request_id.to_owned(),
        availability: Availability::Available,
        events: vec![CursorEvent {
            cursor,
            event: event.clone(),
        }],
        next_cursor: cursor,
        latest_cursor: cursor,
        // Account for terminal metadata in the minimum-page fit guarantee.
        terminal: Some(ResultTerminal::Failed(ResultFailureCode::ResourceFailure)),
    };
    Ok(serialized_page_bytes(&page)? <= MIN_PAGE_BYTES)
}

fn page_event_fits_maximum(
    request_id: &str,
    cursor: u64,
    event: &PageEvent,
) -> Result<bool, ResultBufferError> {
    let page = ResultPage {
        request_id: request_id.to_owned(),
        availability: Availability::Available,
        events: vec![CursorEvent {
            cursor,
            event: event.clone(),
        }],
        next_cursor: cursor,
        latest_cursor: cursor,
        terminal: Some(ResultTerminal::Failed(ResultFailureCode::ResourceFailure)),
    };
    Ok(serialized_page_bytes(&page)? <= MAX_PAGE_BYTES)
}

fn make_page(
    request_id: &str,
    availability: Availability,
    events: Vec<CursorEvent>,
    next_cursor: u64,
    latest_cursor: u64,
    terminal: Option<ResultTerminal>,
    byte_budget: usize,
) -> Result<ResultPage, ResultBufferError> {
    let page = ResultPage {
        request_id: request_id.to_owned(),
        availability,
        events,
        next_cursor,
        latest_cursor,
        terminal,
    };
    if serialized_page_bytes(&page)? > byte_budget {
        return Err(ResultBufferError::PageBudgetTooSmall);
    }
    Ok(page)
}

fn serialized_page_bytes(page: &ResultPage) -> Result<usize, ResultBufferError> {
    serde_json::to_vec(page)
        .map(|bytes| bytes.len())
        .map_err(|_| ResultBufferError::Serialization)
}

fn validate_page_event(item: &CursorEvent) -> Result<(), ResultBufferError> {
    match &item.event {
        PageEvent::ChatContent { text } | PageEvent::ChatReasoning { text } => {
            if text.len() > CHAT_FRAGMENT_BYTES {
                return Err(ResultBufferError::EventTooLarge);
            }
        }
        PageEvent::ChatUsage { .. } | PageEvent::ChatStreamComplete => {}
        PageEvent::DecisionJsonPart {
            result_id,
            part_index,
            part_count,
            json,
        } => {
            if *result_id == 0
                || *result_id > item.cursor
                || *part_count == 0
                || *part_index >= *part_count
                || *part_count as usize > MAX_DECISION_PARTS
                || json.len() > DECISION_PART_BYTES
                || json.is_empty()
                || (*part_index == 0 && *result_id != item.cursor)
            {
                return Err(ResultBufferError::InvalidCursor);
            }
        }
        PageEvent::DecisionMetadataJsonPart {
            result_id,
            part_index,
            part_count,
            json,
        } => {
            if *result_id == 0
                || *result_id > item.cursor
                || *part_count == 0
                || *part_index >= *part_count
                || *part_count as usize > MAX_DECISION_PARTS
                || json.len() > DECISION_PART_BYTES
                || json.is_empty()
                || (*part_index == 0 && *result_id != item.cursor)
            {
                return Err(ResultBufferError::InvalidCursor);
            }
        }
        PageEvent::EmbeddingJsonPart {
            result_id,
            part_index,
            part_count,
            json,
        } => {
            if *result_id == 0
                || *result_id > item.cursor
                || *part_count == 0
                || *part_index >= *part_count
                || *part_count as usize > MAX_EMBEDDING_PARTS
                || json.len() > EMBEDDING_PART_BYTES
                || json.is_empty()
                || !json.is_ascii()
                || (*part_index == 0 && *result_id != item.cursor)
            {
                return Err(ResultBufferError::InvalidCursor);
            }
        }
        PageEvent::MediaArtifactBatch {
            result_id,
            batch_index,
            batch_count,
            artifacts,
        } => {
            if *result_id == 0
                || *result_id > item.cursor
                || *batch_count == 0
                || *batch_count as usize > MAX_ARTIFACTS_PER_APPEND
                || *batch_index >= *batch_count
                || (*batch_index == 0 && *result_id != item.cursor)
                || artifacts.len() > 1
                || (artifacts.is_empty() && (*batch_count != 1 || *batch_index != 0))
            {
                return Err(ResultBufferError::InvalidCursor);
            }
            for artifact in artifacts {
                validate_relative_artifact_path(&artifact.relative_path)?;
            }
        }
    }
    Ok(())
}

fn validate_group_part(
    groups: &mut HashMap<u64, (u32, HashMap<u32, u64>)>,
    result_id: u64,
    part_index: u32,
    part_count: u32,
    cursor: u64,
    max_parts: u32,
) -> Result<(), ResultBufferError> {
    if result_id == 0 || result_id > cursor || part_count == 0 || part_count > max_parts {
        return Err(ResultBufferError::InvalidCursor);
    }
    if result_id.checked_add(part_index as u64) != Some(cursor) {
        return Err(ResultBufferError::InvalidCursor);
    }
    let (stored_count, indexes) = groups
        .entry(result_id)
        .or_insert_with(|| (part_count, HashMap::new()));
    if *stored_count != part_count || indexes.insert(part_index, cursor).is_some() {
        return Err(ResultBufferError::InvalidCursor);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn overflowing_native_latency_rolls_back_answers_and_cursor() {
        let mut buffer = super::ResultBuffer::new();
        buffer.begin("native", super::Lane::Decisions).unwrap();
        let before = buffer.page("native", 0, super::MIN_PAGE_BYTES).unwrap();
        assert_eq!(
            buffer.append(
                "native",
                super::ResultEvent::DecisionResult {
                    answers: serde_json::json!({"answer": true})
                        .as_object()
                        .unwrap()
                        .clone(),
                    native_latency: Some(std::time::Duration::from_secs(u64::MAX)),
                    native_usage: None,
                }
            ),
            Err(super::ResultBufferError::InvalidDecisionMetadata)
        );
        assert_eq!(
            buffer.page("native", 0, super::MIN_PAGE_BYTES).unwrap(),
            before
        );
    }
    use super::*;

    #[test]
    fn reserved_outputs_cannot_be_consumed_by_other_jobs() {
        let mut buffer = ResultBuffer::new();
        for index in 0..MAX_TOTAL_EVENT_BYTES / MAX_JOB_EVENT_BYTES {
            buffer
                .begin_reserved(&format!("reserved-{index}"), Lane::Language)
                .unwrap();
        }
        assert_eq!(buffer.total_reserved_bytes, MAX_TOTAL_EVENT_BYTES);
        assert_eq!(
            buffer.can_begin_reserved("one-more"),
            Err(ResultBufferError::CapacityExceeded)
        );
        buffer.begin("unreserved", Lane::Language).unwrap();
        assert_eq!(
            buffer.append("unreserved", ResultEvent::ChatContent("small".into())),
            Err(ResultBufferError::CapacityExceeded)
        );
        assert_eq!(buffer.total_event_bytes, 0);
        buffer
            .append("reserved-0", ResultEvent::ChatContent("small".into()))
            .unwrap();
        assert!(buffer.total_event_bytes > 0);
        assert_eq!(
            buffer.total_event_bytes + buffer.total_reserved_bytes,
            MAX_TOTAL_EVENT_BYTES
        );
        buffer
            .terminal("reserved-0", ResultTerminal::Succeeded)
            .unwrap();
        buffer
            .terminal("reserved-1", ResultTerminal::Cancelled)
            .unwrap();
        buffer.can_begin_reserved("one-more").unwrap();
        buffer.begin_reserved("one-more", Lane::Decisions).unwrap();
        assert!(buffer.total_event_bytes + buffer.total_reserved_bytes <= MAX_TOTAL_EVENT_BYTES);
        let retained = buffer.total_event_bytes;
        buffer.evict_terminal("reserved-0").unwrap();
        assert_eq!(buffer.total_event_bytes, 0);
        assert!(retained > 0);
        assert_eq!(
            buffer.can_begin_reserved("reserved-0"),
            Err(ResultBufferError::DuplicateRequest)
        );
    }

    #[test]
    fn reserved_acceptance_check_is_read_only_and_failed_append_keeps_quota() {
        let mut buffer = ResultBuffer::new();
        buffer.can_begin_reserved("checked").unwrap();
        buffer.can_begin_reserved("checked").unwrap();
        assert_eq!(
            buffer
                .page("checked", 0, MAX_PAGE_BYTES)
                .unwrap()
                .availability,
            Availability::Unknown
        );
        assert_eq!(buffer.total_reserved_bytes, 0);
        buffer.begin_reserved("checked", Lane::Decisions).unwrap();
        assert_eq!(
            buffer.can_begin_reserved("checked"),
            Err(ResultBufferError::DuplicateRequest)
        );
        assert_eq!(
            buffer.append("checked", ResultEvent::ChatContent("wrong-lane".into())),
            Err(ResultBufferError::WrongLane)
        );
        assert_eq!(buffer.total_reserved_bytes, MAX_JOB_EVENT_BYTES);
        buffer
            .terminal("checked", ResultTerminal::Interrupted)
            .unwrap();
        assert_eq!(buffer.total_reserved_bytes, 0);
        assert_eq!(
            buffer.terminal("checked", ResultTerminal::Cancelled),
            Err(ResultBufferError::AlreadyTerminal)
        );
        assert_eq!(buffer.total_reserved_bytes, 0);
    }

    #[test]
    fn chat_stream_preflight_is_read_only_and_native_done_is_not_job_terminal() {
        let mut buffer = ResultBuffer::new();
        buffer.begin("unreserved-chat", Lane::Language).unwrap();
        assert_eq!(
            buffer.can_append_reserved_chat_stream("unreserved-chat"),
            Err(ResultBufferError::ReservationRequired)
        );
        buffer
            .begin_reserved("stream-chat", Lane::Language)
            .unwrap();
        let reserved_before = buffer.total_reserved_bytes;
        buffer
            .can_append_reserved_chat_stream("stream-chat")
            .unwrap();
        buffer
            .can_append_reserved_chat_stream("stream-chat")
            .unwrap();
        assert_eq!(buffer.total_reserved_bytes, reserved_before);
        assert_eq!(
            buffer
                .page("stream-chat", 0, MAX_PAGE_BYTES)
                .unwrap()
                .latest_cursor,
            0
        );

        buffer
            .append("stream-chat", ResultEvent::ChatContent("first".into()))
            .unwrap();
        assert_eq!(
            buffer.can_append_reserved_chat_stream("stream-chat"),
            Err(ResultBufferError::InvalidChatEvent)
        );
        buffer
            .append(
                "stream-chat",
                ResultEvent::ChatUsage {
                    completion_tokens: u64::MAX,
                },
            )
            .unwrap();
        let before_duplicate_usage = buffer.page("stream-chat", 0, MAX_PAGE_BYTES).unwrap();
        assert_eq!(
            buffer.append(
                "stream-chat",
                ResultEvent::ChatUsage {
                    completion_tokens: 1,
                }
            ),
            Err(ResultBufferError::InvalidChatEvent)
        );
        assert_eq!(
            buffer.page("stream-chat", 0, MAX_PAGE_BYTES).unwrap(),
            before_duplicate_usage
        );
        buffer
            .append("stream-chat", ResultEvent::ChatStreamComplete)
            .unwrap();
        let completed_stream = buffer.page("stream-chat", 0, MAX_PAGE_BYTES).unwrap();
        assert_eq!(completed_stream.terminal, None);
        assert!(matches!(
            &completed_stream.events.last().unwrap().event,
            PageEvent::ChatStreamComplete
        ));
        for late_event in [
            ResultEvent::ChatContent("late".into()),
            ResultEvent::ChatUsage {
                completion_tokens: 2,
            },
            ResultEvent::ChatStreamComplete,
        ] {
            assert_eq!(
                buffer.append("stream-chat", late_event),
                Err(ResultBufferError::InvalidChatEvent)
            );
        }
        assert_eq!(
            buffer.page("stream-chat", 0, MAX_PAGE_BYTES).unwrap(),
            completed_stream
        );
    }

    fn request_id(index: usize) -> String {
        format!("result:{index}")
    }

    fn decision_answers() -> Map<String, Value> {
        serde_json::from_str(r#"{"answer": {"label": "héllo \"world\"", "ok": true}}"#).unwrap()
    }

    #[test]
    fn decoded_page_cannot_mix_native_lanes() {
        let page = ResultPage {
            request_id: request_id(21),
            availability: Availability::Available,
            events: vec![
                CursorEvent {
                    cursor: 1,
                    event: PageEvent::ChatContent {
                        text: "text".into(),
                    },
                },
                CursorEvent {
                    cursor: 2,
                    event: PageEvent::DecisionJsonPart {
                        result_id: 2,
                        part_index: 0,
                        part_count: 1,
                        json: "{}".into(),
                    },
                },
            ],
            next_cursor: 2,
            latest_cursor: 2,
            terminal: None,
        };
        assert_eq!(page.validate(), Err(ResultBufferError::WrongLane));
    }

    #[test]
    fn decoded_chat_page_rejects_duplicate_and_post_complete_events() {
        let page = |events: Vec<CursorEvent>| ResultPage {
            request_id: request_id(22),
            availability: Availability::Available,
            next_cursor: events.len() as u64,
            latest_cursor: events.len() as u64,
            events,
            terminal: None,
        };
        let late_delta = page(vec![
            CursorEvent {
                cursor: 1,
                event: PageEvent::ChatStreamComplete,
            },
            CursorEvent {
                cursor: 2,
                event: PageEvent::ChatContent {
                    text: "late".into(),
                },
            },
        ]);
        assert_eq!(
            late_delta.validate(),
            Err(ResultBufferError::InvalidChatEvent)
        );
        let duplicate_usage = page(vec![
            CursorEvent {
                cursor: 1,
                event: PageEvent::ChatUsage {
                    completion_tokens: 1,
                },
            },
            CursorEvent {
                cursor: 2,
                event: PageEvent::ChatUsage {
                    completion_tokens: 2,
                },
            },
        ]);
        assert_eq!(
            duplicate_usage.validate(),
            Err(ResultBufferError::InvalidChatEvent)
        );
    }

    #[test]
    fn duplicate_ids_lane_mismatch_and_terminal_once_are_enforced() {
        let mut buffer = ResultBuffer::new();
        let id = request_id(1);
        buffer.begin(&id, Lane::Language).unwrap();
        assert_eq!(
            buffer.begin(&id, Lane::Language),
            Err(ResultBufferError::DuplicateRequest)
        );
        assert_eq!(
            buffer.append(&id, ResultEvent::DecisionAnswers(decision_answers())),
            Err(ResultBufferError::WrongLane)
        );
        buffer
            .append(&id, ResultEvent::ChatContent("hello".to_owned()))
            .unwrap();
        buffer.terminal(&id, ResultTerminal::Succeeded).unwrap();
        assert_eq!(
            buffer.terminal(&id, ResultTerminal::Succeeded),
            Err(ResultBufferError::AlreadyTerminal)
        );
        assert_eq!(
            buffer.append(&id, ResultEvent::ChatContent("late".to_owned())),
            Err(ResultBufferError::AlreadyTerminal)
        );
    }

    #[test]
    fn escaped_utf8_text_pages_with_a_repeatable_cursor_and_exact_budget() {
        let mut buffer = ResultBuffer::new();
        let id = request_id(2);
        buffer.begin(&id, Lane::Language).unwrap();
        let text = "quote=\" slash=\\ nul=\u{0} snow=雪🙂\n".repeat(220);
        buffer
            .append(&id, ResultEvent::ChatContent(text.clone()))
            .unwrap();

        let first = buffer.page(&id, 0, MIN_PAGE_BYTES).unwrap();
        let first_json = serde_json::to_vec(&first).unwrap();
        assert!(first_json.len() <= MIN_PAGE_BYTES);
        assert!(first.next_cursor > 0);
        assert!(first.next_cursor < first.latest_cursor);
        assert_eq!(
            first_json,
            serde_json::to_vec(&buffer.page(&id, 0, MIN_PAGE_BYTES).unwrap()).unwrap()
        );

        let mut reconstructed = String::new();
        let mut cursor = 0;
        while cursor < first.latest_cursor {
            let page = buffer.page(&id, cursor, MIN_PAGE_BYTES).unwrap();
            assert!(serde_json::to_vec(&page).unwrap().len() <= MIN_PAGE_BYTES);
            for item in page.events {
                if let PageEvent::ChatContent { text } = item.event {
                    reconstructed.push_str(&text);
                } else {
                    panic!("unexpected result event kind");
                }
            }
            assert!(page.next_cursor > cursor);
            cursor = page.next_cursor;
        }
        assert_eq!(reconstructed, text);
    }

    #[test]
    fn decision_answers_are_split_as_canonical_json_parts() {
        let mut buffer = ResultBuffer::new();
        let id = request_id(3);
        buffer.begin(&id, Lane::Decisions).unwrap();
        let mut answers = Map::new();
        answers.insert(
            "answer".to_owned(),
            Value::String(format!("{}雪\\\"", "x".repeat(700))),
        );
        let expected = answers.clone();
        buffer
            .append(&id, ResultEvent::DecisionAnswers(answers))
            .unwrap();

        let mut cursor = 0;
        let mut parts = Vec::new();
        loop {
            let page = buffer.page(&id, cursor, MIN_PAGE_BYTES).unwrap();
            assert!(serde_json::to_vec(&page).unwrap().len() <= MIN_PAGE_BYTES);
            for item in page.events {
                if let PageEvent::DecisionJsonPart {
                    result_id,
                    part_index,
                    part_count,
                    json,
                } = item.event
                {
                    parts.push((result_id, part_index, part_count, json));
                } else {
                    panic!("unexpected result event kind");
                }
            }
            if page.next_cursor == page.latest_cursor {
                break;
            }
            assert!(page.next_cursor > cursor);
            cursor = page.next_cursor;
        }
        parts.sort_by_key(|part| part.1);
        assert_eq!(parts[0].1, 0);
        assert_eq!(parts.len(), parts[0].2 as usize);
        assert!(parts.iter().all(|part| part.0 == parts[0].0));
        assert!(parts
            .iter()
            .enumerate()
            .all(|(index, part)| part.1 as usize == index));
        let json = parts.into_iter().map(|part| part.3).collect::<String>();
        let restored: Map<String, Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, expected);
    }

    #[test]
    fn media_artifacts_keep_kind_and_reject_unsafe_relative_paths() {
        let mut buffer = ResultBuffer::new();
        let id = request_id(4);
        buffer.begin(&id, Lane::Media).unwrap();
        for path in [
            "/private/file.png",
            "../escape.png",
            "a/../b",
            "a\\b",
            "https://host/file",
        ] {
            assert_eq!(
                MediaArtifact::new(ArtifactKind::Backend, path),
                Err(ResultBufferError::InvalidArtifactPath)
            );
        }
        buffer
            .append(
                &id,
                ResultEvent::MediaArtifacts(vec![
                    MediaArtifact::new(ArtifactKind::LocalJob, "outputs/image.png").unwrap(),
                    MediaArtifact::new(ArtifactKind::Backend, "history/42/image.png").unwrap(),
                ]),
            )
            .unwrap();
        let page = buffer.page(&id, 0, MIN_PAGE_BYTES).unwrap();
        assert!(page.validate().is_ok());
        let artifacts = page
            .events
            .iter()
            .flat_map(|item| match &item.event {
                PageEvent::MediaArtifactBatch { artifacts, .. } => artifacts.as_slice(),
                _ => &[],
            })
            .collect::<Vec<_>>();
        assert_eq!(artifacts[0].kind, ArtifactKind::LocalJob);
        assert_eq!(artifacts[1].kind, ArtifactKind::Backend);
    }

    #[test]
    fn long_valid_artifact_is_retained_but_small_page_does_not_advance_past_it() {
        let mut buffer = ResultBuffer::new();
        let id = request_id(8);
        buffer.begin(&id, Lane::Media).unwrap();
        let path = format!("outputs/{}.png", "a".repeat(1800));
        buffer
            .append(
                &id,
                ResultEvent::MediaArtifacts(vec![MediaArtifact::new(
                    ArtifactKind::LocalJob,
                    path.clone(),
                )
                .unwrap()]),
            )
            .unwrap();
        assert_eq!(
            buffer.page(&id, 0, MIN_PAGE_BYTES),
            Err(ResultBufferError::PageBudgetTooSmall)
        );
        let page = buffer.page(&id, 0, 8 * 1024).unwrap();
        assert_eq!(page.next_cursor, 1);
        assert_eq!(page.latest_cursor, 1);
        let PageEvent::MediaArtifactBatch { artifacts, .. } = &page.events[0].event else {
            panic!("expected media artifact batch");
        };
        assert_eq!(artifacts[0].relative_path, path);
    }

    #[test]
    fn event_debug_redacts_text_decision_values_and_artifact_paths() {
        let private_path = "/home/user/private/output.png";
        let artifact = MediaArtifact {
            kind: ArtifactKind::LocalJob,
            relative_path: "private/output.png".to_owned(),
        };
        let page = ResultPage {
            request_id: "result:debug".to_owned(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 1,
                event: PageEvent::ChatContent {
                    text: "private page body".to_owned(),
                },
            }],
            next_cursor: 1,
            latest_cursor: 1,
            terminal: None,
        };
        let values = [
            format!(
                "{:?}",
                ResultEvent::ChatContent("private prompt".to_owned())
            ),
            format!(
                "{:?}",
                ResultEvent::DecisionAnswers(Map::from_iter([(
                    "answer".to_owned(),
                    Value::String("private answer".to_owned()),
                )]))
            ),
            format!("{artifact:?}"),
            format!(
                "{:?}",
                PageEvent::MediaArtifactBatch {
                    result_id: 1,
                    batch_index: 0,
                    batch_count: 1,
                    artifacts: vec![artifact],
                }
            ),
            format!("{page:?}"),
        ];
        for debug in values {
            assert!(!debug.contains("private prompt"));
            assert!(!debug.contains("private answer"));
            assert!(!debug.contains("private/output.png"));
            assert!(!debug.contains(private_path));
        }
    }

    #[test]
    fn only_terminal_results_are_evicted_and_tombstones_block_reuse() {
        let mut buffer = ResultBuffer::new();
        let first = request_id(5);
        buffer.begin(&first, Lane::Language).unwrap();
        buffer
            .append(&first, ResultEvent::ChatContent("retained".to_owned()))
            .unwrap();
        assert_eq!(
            buffer.evict_terminal(&first),
            Err(ResultBufferError::NotTerminal)
        );
        buffer
            .terminal(&first, ResultTerminal::Interrupted)
            .unwrap();
        assert!(buffer.total_event_bytes > 0);
        buffer.evict_terminal(&first).unwrap();
        assert_eq!(buffer.total_event_bytes, 0);
        let expired = buffer.page(&first, 0, MIN_PAGE_BYTES).unwrap();
        assert_eq!(expired.availability, Availability::Expired);
        assert_eq!(expired.terminal, Some(ResultTerminal::Interrupted));
        assert_eq!(
            buffer.begin(&first, Lane::Language),
            Err(ResultBufferError::DuplicateRequest)
        );

        for number in 10..10 + MAX_TOMBSTONES {
            let id = request_id(number);
            buffer.begin(&id, Lane::Media).unwrap();
            buffer.terminal(&id, ResultTerminal::Succeeded).unwrap();
            buffer.evict_terminal(&id).unwrap();
        }
        assert_eq!(
            buffer.page(&first, 0, MIN_PAGE_BYTES).unwrap().availability,
            Availability::Unknown
        );
        buffer.begin(&first, Lane::Language).unwrap();
    }

    #[test]
    fn full_active_slots_are_never_evicted_implicitly_and_pages_validate() {
        let mut buffer = ResultBuffer::new();
        for number in 0..MAX_RETAINED_JOBS {
            let id = request_id(100 + number);
            buffer.begin(&id, Lane::Language).unwrap();
            buffer
                .append(&id, ResultEvent::ChatContent("kept".to_owned()))
                .unwrap();
        }
        assert_eq!(
            buffer.begin("result:too-many", Lane::Language),
            Err(ResultBufferError::CapacityExceeded)
        );
        let retained = buffer.page(&request_id(100), 0, MIN_PAGE_BYTES).unwrap();
        assert!(retained.validate().is_ok());
        assert_eq!(retained.events.len(), 1);
        assert_eq!(
            buffer.page(&request_id(100), retained.latest_cursor + 1, MIN_PAGE_BYTES),
            Err(ResultBufferError::InvalidCursor)
        );
    }

    #[test]
    fn combined_language_raw_cap_rejects_without_adding_partial_chunks() {
        let mut buffer = ResultBuffer::new();
        let id = request_id(6);
        buffer.begin(&id, Lane::Language).unwrap();
        let first = "\0".repeat(MAX_CHAT_RAW_BYTES_PER_JOB);
        buffer.append(&id, ResultEvent::ChatContent(first)).unwrap();
        assert!(buffer.total_event_bytes > 5 * 1024 * 1024);
        assert!(buffer.total_event_bytes <= MAX_JOB_EVENT_BYTES);
        let first_page = buffer.page(&id, 0, MAX_PAGE_BYTES).unwrap();
        assert!(serde_json::to_vec(&first_page).unwrap().len() <= MAX_PAGE_BYTES);
        assert!(first_page.next_cursor > 0);
        assert!(first_page.next_cursor < first_page.latest_cursor);
        let cursor = first_page.latest_cursor;
        assert_eq!(
            cursor as usize,
            MAX_CHAT_RAW_BYTES_PER_JOB / CHAT_FRAGMENT_BYTES
        );
        assert_eq!(
            buffer.append(&id, ResultEvent::ChatReasoning("x".to_owned())),
            Err(ResultBufferError::EventTooLarge)
        );
        assert_eq!(
            buffer
                .page(&id, cursor, MIN_PAGE_BYTES)
                .unwrap()
                .latest_cursor,
            cursor
        );
    }

    #[test]
    fn page_budget_and_wire_validation_are_bounded() {
        let mut buffer = ResultBuffer::new();
        let id = request_id(7);
        buffer.begin(&id, Lane::Decisions).unwrap();
        buffer
            .append(&id, ResultEvent::DecisionAnswers(decision_answers()))
            .unwrap();
        assert_eq!(
            buffer.page(&id, 0, MIN_PAGE_BYTES - 1),
            Err(ResultBufferError::InvalidPageBudget)
        );
        let page = buffer.page(&id, 0, MAX_PAGE_BYTES).unwrap();
        assert!(serde_json::to_vec(&page).unwrap().len() <= MAX_PAGE_BYTES);
        assert!(page.validate().is_ok());
        let decoded: ResultPage =
            serde_json::from_slice(&serde_json::to_vec(&page).unwrap()).unwrap();
        assert!(decoded.validate().is_ok());
        let mut with_unknown_field = serde_json::to_value(&page).unwrap();
        with_unknown_field
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_owned(), Value::Null);
        assert!(serde_json::from_value::<ResultPage>(with_unknown_field).is_err());
        assert!(serde_json::from_str::<PageEvent>(
            r#"{"type":"chat_content","text":"x","unexpected":true}"#
        )
        .is_err());

        let forged_part = ResultPage {
            request_id: id.clone(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 3,
                event: PageEvent::DecisionJsonPart {
                    result_id: 1,
                    part_index: 1,
                    part_count: 3,
                    json: "{}".to_owned(),
                },
            }],
            next_cursor: 3,
            latest_cursor: 3,
            terminal: None,
        };
        assert_eq!(
            forged_part.validate(),
            Err(ResultBufferError::InvalidCursor)
        );
        let forged_batch = ResultPage {
            request_id: id,
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 3,
                event: PageEvent::MediaArtifactBatch {
                    result_id: 1,
                    batch_index: 1,
                    batch_count: 2,
                    artifacts: vec![
                        MediaArtifact::new(ArtifactKind::Backend, "outputs/image.png").unwrap(),
                    ],
                },
            }],
            next_cursor: 3,
            latest_cursor: 3,
            terminal: None,
        };
        assert_eq!(
            forged_batch.validate(),
            Err(ResultBufferError::InvalidCursor)
        );
    }

    #[test]
    fn capacity_failures_are_transactional_and_eviction_reclaims_event_bytes() {
        let mut per_job = ResultBuffer::new();
        let id = request_id(20);
        per_job.begin(&id, Lane::Language).unwrap();
        per_job.jobs.get_mut(&id).unwrap().event_bytes = MAX_JOB_EVENT_BYTES;
        assert_eq!(
            per_job.append(&id, ResultEvent::ChatContent("x".to_owned())),
            Err(ResultBufferError::CapacityExceeded)
        );
        let job = per_job.jobs.get(&id).unwrap();
        assert_eq!(job.event_bytes, MAX_JOB_EVENT_BYTES);
        assert_eq!(job.latest_cursor, 0);
        assert!(job.events.is_empty());
        assert_eq!(per_job.total_event_bytes, 0);

        let mut global = ResultBuffer::new();
        let id = request_id(21);
        global.begin(&id, Lane::Language).unwrap();
        global.total_event_bytes = MAX_TOTAL_EVENT_BYTES;
        assert_eq!(
            global.append(&id, ResultEvent::ChatContent("x".to_owned())),
            Err(ResultBufferError::CapacityExceeded)
        );
        let job = global.jobs.get(&id).unwrap();
        assert_eq!(job.event_bytes, 0);
        assert_eq!(job.latest_cursor, 0);
        assert!(job.events.is_empty());
        assert_eq!(global.total_event_bytes, MAX_TOTAL_EVENT_BYTES);
    }

    #[test]
    fn language_result_kind_guard_rejects_mixed_or_repeated_embeddings_transactionally() {
        let usage = EmbeddingUsage {
            prompt_tokens: 0,
            total_tokens: 0,
        };
        let mut chat_first = ResultBuffer::new();
        chat_first.begin("chat-first", Lane::Language).unwrap();
        chat_first
            .append("chat-first", ResultEvent::ChatContent("answer".into()))
            .unwrap();
        let before = (
            chat_first.jobs["chat-first"].latest_cursor,
            chat_first.jobs["chat-first"].event_bytes,
            chat_first.total_event_bytes,
            chat_first.total_reserved_bytes,
        );
        assert_eq!(
            chat_first.append(
                "chat-first",
                ResultEvent::Embeddings {
                    vectors: vec![vec![1.0]],
                    usage,
                }
            ),
            Err(ResultBufferError::WrongLane)
        );
        assert_eq!(
            (
                chat_first.jobs["chat-first"].latest_cursor,
                chat_first.jobs["chat-first"].event_bytes,
                chat_first.total_event_bytes,
                chat_first.total_reserved_bytes,
            ),
            before
        );

        let mut embeddings_first = ResultBuffer::new();
        embeddings_first
            .begin("embeddings-first", Lane::Language)
            .unwrap();
        embeddings_first
            .append(
                "embeddings-first",
                ResultEvent::Embeddings {
                    vectors: vec![vec![1.0]],
                    usage,
                },
            )
            .unwrap();
        let before = (
            embeddings_first.jobs["embeddings-first"].latest_cursor,
            embeddings_first.jobs["embeddings-first"].event_bytes,
            embeddings_first.total_event_bytes,
            embeddings_first.total_reserved_bytes,
        );
        for invalid in [
            ResultEvent::ChatContent("answer".into()),
            ResultEvent::Embeddings {
                vectors: vec![vec![2.0]],
                usage,
            },
        ] {
            assert_eq!(
                embeddings_first.append("embeddings-first", invalid),
                Err(ResultBufferError::WrongLane)
            );
            assert_eq!(
                (
                    embeddings_first.jobs["embeddings-first"].latest_cursor,
                    embeddings_first.jobs["embeddings-first"].event_bytes,
                    embeddings_first.total_event_bytes,
                    embeddings_first.total_reserved_bytes,
                ),
                before
            );
        }

        let mut invalid_vectors = ResultBuffer::new();
        invalid_vectors
            .begin("invalid-vector", Lane::Language)
            .unwrap();
        assert_eq!(
            invalid_vectors.append(
                "invalid-vector",
                ResultEvent::Embeddings {
                    vectors: vec![vec![f32::NAN]],
                    usage,
                }
            ),
            Err(ResultBufferError::InvalidEmbedding)
        );
        assert_eq!(invalid_vectors.jobs["invalid-vector"].latest_cursor, 0);
        assert_eq!(invalid_vectors.jobs["invalid-vector"].event_bytes, 0);
    }
}
