//! Versioned, single-frame IPC protocol contracts for the future coordinator.
//!
//! This module only defines the wire format and bounded framing. It does not
//! open sockets, authenticate peers, acquire a host-wide lock, dispatch work,
//! or perform lifecycle operations. A future daemon must supply those
//! guarantees separately. The generic `Read`/`Write` helpers are blocking and
//! do not enforce IO deadlines; the Linux `local_ipc` transport bounds peer IO
//! outside this module. Requests intentionally have no `Debug` output so prompts,
//! workflow graphs, speech text, and native arguments are not dumped
//! accidentally by diagnostic logging. Version 1 returns acceptance and
//! lifecycle status only. Version 2 adds a read-only bounded results-page
//! query, and version 3 adds bounded reads of already-published media
//! artifacts; neither version starts runtimes or streams inference output.

use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Duration;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::catalog::{Lane, ProfileKey, Task};
use super::requests::{ChatRole, ChatTurn, InferenceRequest, MediaInput, SpeechInput, SpeechMode};
use super::result_buffer::{
    Availability, MediaArtifact, ResultBufferError, ResultPage, MAX_PAGE_BYTES, MIN_PAGE_BYTES,
};
use crate::state_store::{validate_job_metadata, JobMetadata};

pub const PROTOCOL_VERSION: u16 = 1;
/// Result retrieval is deliberately additive: v1 submit/status/cancel remain
/// unchanged and are never interpreted as v2 operations.
pub const RESULTS_PROTOCOL_VERSION: u16 = 2;
/// Published artifact chunks use a separate read-only protocol version. The
/// existing v1 control and v2 page-query wire shapes stay unchanged.
pub const ARTIFACT_PROTOCOL_VERSION: u16 = 3;
/// Artifact reads are single exchanges capped at 64 KiB of decoded bytes.
pub const MAX_ARTIFACT_CHUNK_BYTES: usize = 64 * 1024;
const ARTIFACT_VERSION_BYTES: usize = 64;
const MAX_ARTIFACT_CHUNK_HEX_BYTES: usize = MAX_ARTIFACT_CHUNK_BYTES * 2;
/// One native request is limited to 1 MiB; this small allowance covers its
/// versioned envelope and JSON field names without permitting unbounded reads.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024 + 16 * 1024;

const MAX_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;

/// Errors deliberately contain no decoded request data, prompts, or parser
/// excerpts. `UnsupportedVersion` includes only the bounded numeric version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolError {
    Io,
    EmptyFrame,
    FrameTooLarge,
    TruncatedFrame,
    MalformedFrame,
    UnsupportedVersion(u16),
    InvalidRequestId,
    InvalidTimeout,
    InvalidRequest,
    InvalidArtifactReference,
    InvalidArtifactVersion,
    InvalidArtifactChunk,
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io => formatter.write_str("IPC I/O failed"),
            Self::EmptyFrame => formatter.write_str("IPC frame is empty"),
            Self::FrameTooLarge => formatter.write_str("IPC frame exceeds the size limit"),
            Self::TruncatedFrame => formatter.write_str("IPC frame is truncated"),
            Self::MalformedFrame => formatter.write_str("IPC frame is malformed"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported IPC protocol version {version}")
            }
            Self::InvalidRequestId => formatter.write_str("request ID is invalid"),
            Self::InvalidTimeout => formatter.write_str("request timeout is invalid"),
            Self::InvalidRequest => formatter.write_str("inference request is invalid"),
            Self::InvalidArtifactReference => formatter.write_str("artifact reference is invalid"),
            Self::InvalidArtifactVersion => formatter.write_str("artifact version is invalid"),
            Self::InvalidArtifactChunk => formatter.write_str("artifact chunk is invalid"),
        }
    }
}

impl Error for ProtocolError {}

/// Opaque, bounded server token pinning one published artifact version.
/// The value is intentionally omitted from Debug output and is never a path
/// or URL. The server chooses the token; clients echo it on later chunks.
#[derive(Clone, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ArtifactVersion(String);

impl ArtifactVersion {
    pub fn new(value: &str) -> Result<Self, ProtocolError> {
        if value.len() != ARTIFACT_VERSION_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ProtocolError::InvalidArtifactVersion);
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ArtifactVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(&value).map_err(de::Error::custom)
    }
}

impl fmt::Debug for ArtifactVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ArtifactVersion([redacted])")
    }
}

/// One JSON request envelope. Unknown fields are rejected by deserialization.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestEnvelope {
    pub version: u16,
    pub message: WireRequestMessage,
}

impl RequestEnvelope {
    pub fn v1(message: WireRequestMessage) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            message,
        }
    }

    pub fn v2_results(request_id: impl Into<String>, after_cursor: u64, byte_budget: u32) -> Self {
        Self {
            version: RESULTS_PROTOCOL_VERSION,
            message: WireRequestMessage::Results {
                request_id: request_id.into(),
                after_cursor,
                byte_budget,
            },
        }
    }

    /// Construct one read-only v3 query for an already-published artifact.
    /// The byte budget counts decoded bytes and is checked before writing.
    pub fn v3_artifact(
        request_id: impl Into<String>,
        artifact: MediaArtifact,
        offset: u64,
        byte_budget: u32,
        artifact_version: Option<ArtifactVersion>,
    ) -> Self {
        Self {
            version: ARTIFACT_PROTOCOL_VERSION,
            message: WireRequestMessage::Artifact {
                request_id: request_id.into(),
                artifact,
                offset,
                byte_budget,
                artifact_version,
            },
        }
    }

    /// Convert the wire contract to the existing lane-specific in-memory
    /// request. Catalog ownership, declared tasks, and all runtime readiness
    /// checks remain the coordinator's responsibility.
    pub fn into_client_request(self) -> Result<ClientRequest, ProtocolError> {
        if self.version != PROTOCOL_VERSION {
            return Err(ProtocolError::UnsupportedVersion(self.version));
        }
        match self.message {
            WireRequestMessage::Submit {
                request_id,
                timeout_ms,
                request,
            } => {
                validate_request_id(&request_id)?;
                if timeout_ms == 0 || timeout_ms > MAX_TIMEOUT_MS {
                    return Err(ProtocolError::InvalidTimeout);
                }
                Ok(ClientRequest::Submit {
                    request_id,
                    timeout: Duration::from_millis(timeout_ms),
                    request: request.into_inference_request()?,
                })
            }
            WireRequestMessage::Cancel { request_id } => {
                validate_request_id(&request_id)?;
                Ok(ClientRequest::Cancel { request_id })
            }
            WireRequestMessage::Status { request_id } => {
                validate_request_id(&request_id)?;
                Ok(ClientRequest::Status { request_id })
            }
            WireRequestMessage::Results { .. } | WireRequestMessage::Artifact { .. } => {
                Err(ProtocolError::InvalidRequest)
            }
        }
    }

    /// Decode a protocol-versioned operation without changing the v1 client
    /// request interface used by existing submit/status/cancel callers.
    pub fn into_incoming_request(self) -> Result<IncomingRequest, ProtocolError> {
        match self.version {
            PROTOCOL_VERSION => self.into_client_request().map(IncomingRequest::Control),
            RESULTS_PROTOCOL_VERSION => match self.message {
                WireRequestMessage::Results {
                    request_id,
                    after_cursor,
                    byte_budget,
                } => {
                    validate_request_id(&request_id)?;
                    validate_page_budget(byte_budget)?;
                    Ok(IncomingRequest::Results(ResultsQuery {
                        request_id,
                        after_cursor,
                        byte_budget,
                    }))
                }
                _ => Err(ProtocolError::InvalidRequest),
            },
            ARTIFACT_PROTOCOL_VERSION => match self.message {
                WireRequestMessage::Artifact {
                    request_id,
                    artifact,
                    offset,
                    byte_budget,
                    artifact_version,
                } => {
                    validate_artifact_request_fields(
                        &request_id,
                        &artifact,
                        offset,
                        byte_budget,
                        artifact_version.as_ref(),
                    )?;
                    Ok(IncomingRequest::Artifact(ArtifactQuery {
                        request_id,
                        artifact,
                        offset,
                        byte_budget,
                        artifact_version,
                    }))
                }
                _ => Err(ProtocolError::InvalidRequest),
            },
            version => Err(ProtocolError::UnsupportedVersion(version)),
        }
    }
}

/// The request operation. `request_id` is caller-owned, bounded, and remains
/// the lookup key across daemon restarts while its bounded history row is
/// retained; numeric history job IDs are intentionally absent from the wire.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WireRequestMessage {
    Submit {
        request_id: String,
        timeout_ms: u64,
        request: WireInferenceRequest,
    },
    Cancel {
        request_id: String,
    },
    Status {
        request_id: String,
    },
    /// Bounded, repeatable result lookup. Valid only in protocol v2.
    Results {
        request_id: String,
        after_cursor: u64,
        byte_budget: u32,
    },
    /// One bounded read from an already-published relative artifact reference.
    /// Valid only in protocol v3; it never accepts a URL or root path.
    Artifact {
        request_id: String,
        artifact: MediaArtifact,
        offset: u64,
        byte_budget: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        artifact_version: Option<ArtifactVersion>,
    },
}

/// A checked result lookup decoded from a v2 envelope.
#[derive(Clone, Eq, PartialEq)]
pub struct ResultsQuery {
    pub request_id: String,
    pub after_cursor: u64,
    pub byte_budget: u32,
}

/// Versioned incoming requests for the coordinator. Existing v1 request
/// handlers continue to receive `ClientRequest` unchanged.
pub enum IncomingRequest {
    Control(ClientRequest),
    Results(ResultsQuery),
    Artifact(ArtifactQuery),
}

impl IncomingRequest {
    pub fn request_id(&self) -> &str {
        match self {
            Self::Control(request) => request.request_id(),
            Self::Results(query) => &query.request_id,
            Self::Artifact(query) => &query.request_id,
        }
    }

    pub fn version(&self) -> u16 {
        match self {
            Self::Control(_) => PROTOCOL_VERSION,
            Self::Results(_) => RESULTS_PROTOCOL_VERSION,
            Self::Artifact(_) => ARTIFACT_PROTOCOL_VERSION,
        }
    }
}

/// Checked artifact query delivered to a coordinator. The reference must be
/// present in already-published media output for this exact request. Offset
/// zero may omit the version token; every nonzero offset must echo the token
/// from the first available chunk, preventing mixed-version assembly.
#[derive(Clone, Eq, PartialEq)]
pub struct ArtifactQuery {
    pub request_id: String,
    pub artifact: MediaArtifact,
    pub offset: u64,
    pub byte_budget: u32,
    pub artifact_version: Option<ArtifactVersion>,
}

impl fmt::Debug for ArtifactQuery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactQuery")
            .field("request_id", &"<redacted>")
            .field("artifact", &self.artifact)
            .field("offset", &self.offset)
            .field("byte_budget", &self.byte_budget)
            .field("artifact_version", &"<redacted>")
            .finish()
    }
}

/// Native inference inputs remain explicitly lane-qualified on the wire.
#[derive(Clone)]
pub enum WireInferenceRequest {
    Language {
        profile_id: String,
        turns: Vec<WireChatTurn>,
        max_tokens: u32,
    },
    Embeddings {
        profile_id: String,
        input: Vec<String>,
    },
    Media {
        profile_id: String,
        input: WireMediaInput,
    },
    Decisions {
        profile_id: String,
        state: Value,
        questions: Map<String, Value>,
    },
}

impl Serialize for WireInferenceRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Language {
                profile_id,
                turns,
                max_tokens,
            } => {
                let mut state = serializer.serialize_struct("WireInferenceRequest", 4)?;
                state.serialize_field("lane", "language")?;
                state.serialize_field("profile_id", profile_id)?;
                state.serialize_field("turns", turns)?;
                state.serialize_field("max_tokens", max_tokens)?;
                state.end()
            }
            Self::Embeddings { profile_id, input } => {
                let mut state = serializer.serialize_struct("WireInferenceRequest", 4)?;
                state.serialize_field("lane", "language")?;
                state.serialize_field("task", "embeddings")?;
                state.serialize_field("profile_id", profile_id)?;
                state.serialize_field("input", input)?;
                state.end()
            }
            Self::Media { profile_id, input } => {
                let mut state = serializer.serialize_struct("WireInferenceRequest", 3)?;
                state.serialize_field("lane", "media")?;
                state.serialize_field("profile_id", profile_id)?;
                state.serialize_field("input", input)?;
                state.end()
            }
            Self::Decisions {
                profile_id,
                state: decision_state,
                questions,
            } => {
                let mut state = serializer.serialize_struct("WireInferenceRequest", 4)?;
                state.serialize_field("lane", "decisions")?;
                state.serialize_field("profile_id", profile_id)?;
                state.serialize_field("state", decision_state)?;
                state.serialize_field("questions", questions)?;
                state.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for WireInferenceRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(WireInferenceRequestVisitor)
    }
}

struct WireInferenceRequestVisitor;

#[derive(Deserialize)]
#[serde(untagged)]
enum WireRequestInput {
    Embeddings(Vec<String>),
    Media(WireMediaInput),
}

impl<'de> Visitor<'de> for WireInferenceRequestVisitor {
    type Value = WireInferenceRequest;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a strict lane-qualified inference request")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut lane: Option<String> = None;
        let mut task: Option<String> = None;
        let mut profile_id: Option<String> = None;
        let mut turns: Option<Vec<WireChatTurn>> = None;
        let mut max_tokens: Option<u32> = None;
        let mut input: Option<WireRequestInput> = None;
        let mut state: Option<Value> = None;
        let mut questions: Option<Map<String, Value>> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "lane" => {
                    if lane.is_some() {
                        return Err(de::Error::duplicate_field("lane"));
                    }
                    lane = Some(map.next_value()?);
                }
                "task" => {
                    if task.is_some() {
                        return Err(de::Error::duplicate_field("task"));
                    }
                    task = Some(map.next_value()?);
                }
                "profile_id" => {
                    if profile_id.is_some() {
                        return Err(de::Error::duplicate_field("profile_id"));
                    }
                    profile_id = Some(map.next_value()?);
                }
                "turns" => {
                    if turns.is_some() {
                        return Err(de::Error::duplicate_field("turns"));
                    }
                    turns = Some(map.next_value()?);
                }
                "max_tokens" => {
                    if max_tokens.is_some() {
                        return Err(de::Error::duplicate_field("max_tokens"));
                    }
                    max_tokens = Some(map.next_value()?);
                }
                "input" => {
                    if input.is_some() {
                        return Err(de::Error::duplicate_field("input"));
                    }
                    input = Some(map.next_value()?);
                }
                "state" => {
                    if state.is_some() {
                        return Err(de::Error::duplicate_field("state"));
                    }
                    state = Some(map.next_value()?);
                }
                "questions" => {
                    if questions.is_some() {
                        return Err(de::Error::duplicate_field("questions"));
                    }
                    questions = Some(map.next_value()?);
                }
                _ => return Err(de::Error::unknown_field(&key, WIRE_INFERENCE_FIELDS)),
            }
        }

        let lane = lane.ok_or_else(|| de::Error::missing_field("lane"))?;
        let profile_id = profile_id.ok_or_else(|| de::Error::missing_field("profile_id"))?;
        match lane.as_str() {
            "language" if task.is_none() => {
                if input.is_some() || state.is_some() || questions.is_some() {
                    return Err(de::Error::custom("mixed language request fields"));
                }
                Ok(WireInferenceRequest::Language {
                    profile_id,
                    turns: turns.ok_or_else(|| de::Error::missing_field("turns"))?,
                    max_tokens: max_tokens.ok_or_else(|| de::Error::missing_field("max_tokens"))?,
                })
            }
            "language" if task.as_deref() == Some("embeddings") => {
                if turns.is_some() || max_tokens.is_some() || state.is_some() || questions.is_some()
                {
                    return Err(de::Error::custom("mixed embeddings request fields"));
                }
                let Some(WireRequestInput::Embeddings(input)) = input else {
                    return Err(de::Error::missing_field("input"));
                };
                Ok(WireInferenceRequest::Embeddings { profile_id, input })
            }
            "media" if task.is_none() => {
                if turns.is_some() || max_tokens.is_some() || state.is_some() || questions.is_some()
                {
                    return Err(de::Error::custom("mixed media request fields"));
                }
                let Some(WireRequestInput::Media(input)) = input else {
                    return Err(de::Error::missing_field("input"));
                };
                Ok(WireInferenceRequest::Media { profile_id, input })
            }
            "decisions" if task.is_none() => {
                if turns.is_some() || max_tokens.is_some() || input.is_some() {
                    return Err(de::Error::custom("mixed decision request fields"));
                }
                Ok(WireInferenceRequest::Decisions {
                    profile_id,
                    state: state.ok_or_else(|| de::Error::missing_field("state"))?,
                    questions: questions.ok_or_else(|| de::Error::missing_field("questions"))?,
                })
            }
            "language" | "media" | "decisions" => Err(de::Error::custom(
                "invalid or mixed inference request fields",
            )),
            _ => Err(de::Error::unknown_variant(
                &lane,
                &["language", "media", "decisions"],
            )),
        }
    }
}

const WIRE_INFERENCE_FIELDS: &[&str] = &[
    "lane",
    "task",
    "profile_id",
    "turns",
    "max_tokens",
    "input",
    "state",
    "questions",
];

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireChatTurn {
    pub role: WireChatRole,
    pub content: String,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireChatRole {
    System,
    User,
    Assistant,
}

/// Media payloads preserve the native workflow/script/speech distinction.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WireMediaInput {
    Workflow {
        task: String,
        graph: Value,
    },
    Script {
        task: String,
        args: Vec<String>,
    },
    Speech {
        text: String,
        mode: WireSpeechMode,
        instruction: Option<String>,
        voice_reference: Option<String>,
        reference_text: Option<String>,
    },
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireSpeechMode {
    Standard,
    VoiceDesign,
    VoiceClone,
    Streaming,
}

impl WireInferenceRequest {
    fn into_inference_request(self) -> Result<InferenceRequest, ProtocolError> {
        match self {
            Self::Language {
                profile_id,
                turns,
                max_tokens,
            } => Ok(InferenceRequest::Chat {
                profile: profile_key(Lane::Language, profile_id)?,
                turns: turns
                    .into_iter()
                    .map(|turn| ChatTurn {
                        role: match turn.role {
                            WireChatRole::System => ChatRole::System,
                            WireChatRole::User => ChatRole::User,
                            WireChatRole::Assistant => ChatRole::Assistant,
                        },
                        content: turn.content,
                    })
                    .collect(),
                max_tokens,
            }),
            Self::Embeddings { profile_id, input } => Ok(InferenceRequest::Embeddings {
                profile: profile_key(Lane::Language, profile_id)?,
                input,
            }),
            Self::Media { profile_id, input } => Ok(InferenceRequest::Media {
                profile: profile_key(Lane::Media, profile_id)?,
                input: match input {
                    WireMediaInput::Workflow { task, graph } => MediaInput::Workflow {
                        task: Task::new(task).map_err(|_| ProtocolError::InvalidRequest)?,
                        graph,
                    },
                    WireMediaInput::Script { task, args } => MediaInput::Script {
                        task: Task::new(task).map_err(|_| ProtocolError::InvalidRequest)?,
                        args,
                    },
                    WireMediaInput::Speech {
                        text,
                        mode,
                        instruction,
                        voice_reference,
                        reference_text,
                    } => MediaInput::Speech(SpeechInput {
                        text,
                        mode: match mode {
                            WireSpeechMode::Standard => SpeechMode::Standard,
                            WireSpeechMode::VoiceDesign => SpeechMode::VoiceDesign,
                            WireSpeechMode::VoiceClone => SpeechMode::VoiceClone,
                            WireSpeechMode::Streaming => SpeechMode::Streaming,
                        },
                        instruction,
                        voice_reference: voice_reference.map(PathBuf::from),
                        reference_text,
                    }),
                },
            }),
            Self::Decisions {
                profile_id,
                state,
                questions,
            } => Ok(InferenceRequest::Decision {
                profile: profile_key(Lane::Decisions, profile_id)?,
                state,
                questions,
                images: Vec::new(),
            }),
        }
    }
}

fn profile_key(lane: Lane, profile_id: String) -> Result<ProfileKey, ProtocolError> {
    ProfileKey::new(lane, profile_id).map_err(|_| ProtocolError::InvalidRequest)
}

/// Validated request delivered to the future daemon's coordinator adapter.
/// This deliberately does not implement `Debug`: it contains native payloads.
pub enum ClientRequest {
    Submit {
        request_id: String,
        request: InferenceRequest,
        timeout: Duration,
    },
    Cancel {
        request_id: String,
    },
    Status {
        request_id: String,
    },
}

impl ClientRequest {
    pub fn request_id(&self) -> &str {
        match self {
            Self::Submit { request_id, .. }
            | Self::Cancel { request_id }
            | Self::Status { request_id } => request_id,
        }
    }
}

/// Stable lifecycle values returned by submit/cancel acceptance and status.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
    NotFound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptedOperation {
    Submit,
    Cancel,
}

/// Error categories are stable and carry no free-form text that could disclose
/// prompts, native arguments, paths, or credentials.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseErrorCode {
    InvalidFrame,
    UnsupportedVersion,
    InvalidRequest,
    DuplicateRequest,
    QueueFull,
    Unavailable,
    Internal,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseEnvelope {
    pub version: u16,
    pub message: ResponseMessage,
}

impl ResponseEnvelope {
    pub fn v1(message: ResponseMessage) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            message,
        }
    }

    pub fn v2_results(message: ResponseMessage) -> Self {
        Self {
            version: RESULTS_PROTOCOL_VERSION,
            message,
        }
    }

    pub fn v3_artifact(message: ResponseMessage) -> Self {
        Self {
            version: ARTIFACT_PROTOCOL_VERSION,
            message,
        }
    }
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseMessage {
    /// A submit may be acknowledged only after queued acceptance is durable.
    /// For cancel, acceptance means cancellation was recorded or signalled; a
    /// running request can remain `running` until native cleanup completes.
    Accepted {
        request_id: String,
        operation: AcceptedOperation,
        state: RequestStatus,
    },
    Status {
        request_id: String,
        state: RequestStatus,
    },
    /// Bounded result page plus authoritative durable lifecycle state.
    Results {
        request_id: String,
        after_cursor: u64,
        byte_budget: u32,
        state: RequestStatus,
        page: ResultPage,
    },
    /// One bounded read from a published artifact. `bytes` is serialized as
    /// canonical lowercase hexadecimal to keep the wire representation small
    /// without introducing another package dependency. For `unknown` or
    /// `expired`, the body/version are empty, `total_bytes` is zero,
    /// `next_offset` echoes `offset`, and `end_of_file` is false.
    Artifact {
        request_id: String,
        artifact: MediaArtifact,
        availability: Availability,
        offset: u64,
        total_bytes: u64,
        next_offset: u64,
        end_of_file: bool,
        artifact_version: Option<ArtifactVersion>,
        #[serde(rename = "bytes_hex", with = "artifact_bytes_hex")]
        bytes: Vec<u8>,
    },
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
        code: ResponseErrorCode,
    },
}

impl fmt::Debug for ResponseMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted {
                operation, state, ..
            } => formatter
                .debug_struct("Accepted")
                .field("request_id", &"<redacted>")
                .field("operation", operation)
                .field("state", state)
                .finish(),
            Self::Status { state, .. } => formatter
                .debug_struct("Status")
                .field("request_id", &"<redacted>")
                .field("state", state)
                .finish(),
            Self::Results {
                after_cursor,
                byte_budget,
                state,
                ..
            } => formatter
                .debug_struct("Results")
                .field("request_id", &"<redacted>")
                .field("after_cursor", after_cursor)
                .field("byte_budget", byte_budget)
                .field("state", state)
                .field("page", &"<redacted>")
                .finish(),
            Self::Artifact {
                availability,
                offset,
                total_bytes,
                next_offset,
                end_of_file,
                ..
            } => formatter
                .debug_struct("Artifact")
                .field("request_id", &"<redacted>")
                .field("artifact", &"<redacted>")
                .field("availability", availability)
                .field("offset", offset)
                .field("total_bytes", total_bytes)
                .field("next_offset", next_offset)
                .field("end_of_file", end_of_file)
                .field("artifact_version", &"<redacted>")
                .field("bytes", &"<redacted>")
                .finish(),
            Self::Error { code, .. } => formatter
                .debug_struct("Error")
                .field("request_id", &"<redacted>")
                .field("code", code)
                .finish(),
        }
    }
}

/// Read exactly one bounded length-prefixed JSON frame and decode a request.
/// The 4-byte big-endian length is checked before allocating the frame buffer.
pub fn read_request<R: Read>(reader: &mut R) -> Result<ClientRequest, ProtocolError> {
    let payload = read_payload(reader)?;
    let envelope: RequestEnvelope = decode_json(&payload)?;
    envelope.into_client_request()
}

/// Read one bounded frame as a protocol-versioned coordinator operation.
pub fn read_incoming_request<R: Read>(reader: &mut R) -> Result<IncomingRequest, ProtocolError> {
    let payload = read_payload(reader)?;
    let envelope: RequestEnvelope = decode_json(&payload)?;
    envelope.into_incoming_request()
}

/// Serialize and write one request frame. Version, request ID, and timeout are
/// checked before any bytes are written; full catalog validation belongs to
/// the coordinator after decoding.
pub fn write_request<W: Write>(
    writer: &mut W,
    envelope: &RequestEnvelope,
) -> Result<(), ProtocolError> {
    validate_request_envelope(envelope)?;
    let payload = encode_json(envelope)?;
    write_payload(writer, &payload)
}

pub fn read_response<R: Read>(reader: &mut R) -> Result<ResponseEnvelope, ProtocolError> {
    let payload = read_payload(reader)?;
    let envelope: ResponseEnvelope = decode_json(&payload)?;
    validate_response_envelope(&envelope)?;
    Ok(envelope)
}

pub fn write_response<W: Write>(
    writer: &mut W,
    envelope: &ResponseEnvelope,
) -> Result<(), ProtocolError> {
    validate_response_envelope(envelope)?;
    let payload = encode_json(envelope)?;
    write_payload(writer, &payload)
}

pub(crate) fn validate_request_envelope(envelope: &RequestEnvelope) -> Result<(), ProtocolError> {
    validate_version(envelope.version)?;
    match (envelope.version, &envelope.message) {
        (PROTOCOL_VERSION, WireRequestMessage::Submit { .. })
        | (PROTOCOL_VERSION, WireRequestMessage::Cancel { .. })
        | (PROTOCOL_VERSION, WireRequestMessage::Status { .. }) => {
            validate_wire_request_headers(&envelope.message)
        }
        (
            RESULTS_PROTOCOL_VERSION,
            WireRequestMessage::Results {
                request_id,
                byte_budget,
                ..
            },
        ) => {
            validate_request_id(request_id)?;
            validate_page_budget(*byte_budget)
        }
        (
            ARTIFACT_PROTOCOL_VERSION,
            WireRequestMessage::Artifact {
                request_id,
                artifact,
                offset,
                byte_budget,
                artifact_version,
                ..
            },
        ) => validate_artifact_request_fields(
            request_id,
            artifact,
            *offset,
            *byte_budget,
            artifact_version.as_ref(),
        ),
        _ => Err(ProtocolError::InvalidRequest),
    }
}

fn validate_response_envelope(envelope: &ResponseEnvelope) -> Result<(), ProtocolError> {
    validate_version(envelope.version)?;
    validate_response_ids(&envelope.message)?;
    match (envelope.version, &envelope.message) {
        (PROTOCOL_VERSION, ResponseMessage::Accepted { .. })
        | (PROTOCOL_VERSION, ResponseMessage::Status { .. })
        | (PROTOCOL_VERSION, ResponseMessage::Error { .. }) => Ok(()),
        (
            RESULTS_PROTOCOL_VERSION,
            ResponseMessage::Results {
                after_cursor,
                byte_budget,
                page,
                ..
            },
        ) => validate_result_page(*after_cursor, *byte_budget, page),
        (RESULTS_PROTOCOL_VERSION, ResponseMessage::Error { .. }) => Ok(()),
        (ARTIFACT_PROTOCOL_VERSION, ResponseMessage::Artifact { .. }) => {
            validate_artifact_response_fields(&envelope.message)
        }
        (ARTIFACT_PROTOCOL_VERSION, ResponseMessage::Error { .. }) => Ok(()),
        _ => Err(ProtocolError::InvalidRequest),
    }
}

fn validate_version(version: u16) -> Result<(), ProtocolError> {
    if version == PROTOCOL_VERSION
        || version == RESULTS_PROTOCOL_VERSION
        || version == ARTIFACT_PROTOCOL_VERSION
    {
        Ok(())
    } else {
        Err(ProtocolError::UnsupportedVersion(version))
    }
}

mod artifact_bytes_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::{MAX_ARTIFACT_CHUNK_BYTES, MAX_ARTIFACT_CHUNK_HEX_BYTES};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if bytes.len() > MAX_ARTIFACT_CHUNK_BYTES {
            return Err(<S::Error as serde::ser::Error>::custom(
                "artifact chunk exceeds the size limit",
            ));
        }
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0x0f) as usize] as char);
        }
        serializer.serialize_str(&encoded)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        if encoded.len() > MAX_ARTIFACT_CHUNK_HEX_BYTES || encoded.len() % 2 != 0 {
            return Err(<D::Error as serde::de::Error>::custom(
                "invalid bounded artifact bytes",
            ));
        }
        let mut bytes = Vec::with_capacity(encoded.len() / 2);
        for pair in encoded.as_bytes().chunks_exact(2) {
            let high = nibble(pair[0])
                .ok_or_else(|| <D::Error as serde::de::Error>::custom("invalid artifact hex"))?;
            let low = nibble(pair[1])
                .ok_or_else(|| <D::Error as serde::de::Error>::custom("invalid artifact hex"))?;
            bytes.push((high << 4) | low);
        }
        Ok(bytes)
    }

    fn nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
}

fn validate_wire_request_headers(message: &WireRequestMessage) -> Result<(), ProtocolError> {
    match message {
        WireRequestMessage::Submit {
            request_id,
            timeout_ms,
            ..
        } => {
            validate_request_id(request_id)?;
            if *timeout_ms == 0 || *timeout_ms > MAX_TIMEOUT_MS {
                return Err(ProtocolError::InvalidTimeout);
            }
        }
        WireRequestMessage::Cancel { request_id } | WireRequestMessage::Status { request_id } => {
            validate_request_id(request_id)?;
        }
        WireRequestMessage::Results { .. } | WireRequestMessage::Artifact { .. } => {
            return Err(ProtocolError::InvalidRequest);
        }
    }
    Ok(())
}

fn validate_response_ids(message: &ResponseMessage) -> Result<(), ProtocolError> {
    match message {
        ResponseMessage::Accepted { request_id, .. }
        | ResponseMessage::Status { request_id, .. } => validate_request_id(request_id),
        ResponseMessage::Results {
            request_id, page, ..
        } => {
            validate_request_id(request_id)?;
            if page.request_id != *request_id {
                return Err(ProtocolError::InvalidRequest);
            }
            Ok(())
        }
        ResponseMessage::Artifact { .. } => validate_artifact_response_fields(message),
        ResponseMessage::Error { request_id, .. } => {
            if let Some(request_id) = request_id {
                validate_request_id(request_id)?;
            }
            Ok(())
        }
    }
}

fn validate_page_budget(byte_budget: u32) -> Result<(), ProtocolError> {
    if (MIN_PAGE_BYTES..=MAX_PAGE_BYTES).contains(&(byte_budget as usize)) {
        Ok(())
    } else {
        Err(ProtocolError::InvalidRequest)
    }
}

fn validate_artifact_request_fields(
    request_id: &str,
    artifact: &MediaArtifact,
    offset: u64,
    byte_budget: u32,
    artifact_version: Option<&ArtifactVersion>,
) -> Result<(), ProtocolError> {
    validate_request_id(request_id)?;
    validate_artifact_reference(artifact)?;
    validate_artifact_byte_budget(byte_budget)?;
    if (offset > 0 && artifact_version.is_none())
        || artifact_version.is_some_and(|version| ArtifactVersion::new(version.as_str()).is_err())
    {
        return Err(ProtocolError::InvalidArtifactVersion);
    }
    Ok(())
}

fn validate_artifact_reference(artifact: &MediaArtifact) -> Result<(), ProtocolError> {
    MediaArtifact::new(artifact.kind, artifact.relative_path.clone())
        .map(|_| ())
        .map_err(|_| ProtocolError::InvalidArtifactReference)
}

fn validate_artifact_byte_budget(byte_budget: u32) -> Result<(), ProtocolError> {
    if (1..=MAX_ARTIFACT_CHUNK_BYTES as u32).contains(&byte_budget) {
        Ok(())
    } else {
        Err(ProtocolError::InvalidRequest)
    }
}

fn validate_artifact_response_fields(response: &ResponseMessage) -> Result<(), ProtocolError> {
    let ResponseMessage::Artifact {
        request_id,
        artifact,
        availability,
        offset,
        total_bytes,
        next_offset,
        end_of_file,
        artifact_version,
        bytes,
    } = response
    else {
        return Err(ProtocolError::InvalidArtifactChunk);
    };
    validate_request_id(request_id)?;
    validate_artifact_reference(artifact)?;
    if bytes.len() > MAX_ARTIFACT_CHUNK_BYTES {
        return Err(ProtocolError::InvalidArtifactChunk);
    }
    match availability {
        Availability::Available => {
            let expected_next = offset
                .checked_add(bytes.len() as u64)
                .ok_or(ProtocolError::InvalidArtifactChunk)?;
            if artifact_version.is_none()
                || offset > total_bytes
                || *next_offset != expected_next
                || next_offset > total_bytes
                || *end_of_file != (next_offset == total_bytes)
                || (bytes.is_empty() && !*end_of_file)
                || artifact_version
                    .as_ref()
                    .is_some_and(|version| ArtifactVersion::new(version.as_str()).is_err())
            {
                return Err(ProtocolError::InvalidArtifactChunk);
            }
        }
        Availability::Unknown | Availability::Expired => {
            if *total_bytes != 0
                || next_offset != offset
                || *end_of_file
                || artifact_version.is_some()
                || !bytes.is_empty()
            {
                return Err(ProtocolError::InvalidArtifactChunk);
            }
        }
    }
    Ok(())
}

/// Validate that one v3 response is correlated to the exact query, including
/// the caller's byte budget and version token. Unavailable and expired replies
/// remain explicit values and contain no bytes or version token.
pub(crate) fn validate_artifact_reply(
    request_id: &str,
    artifact: &MediaArtifact,
    offset: u64,
    byte_budget: u32,
    expected_version: Option<&ArtifactVersion>,
    response: &ResponseMessage,
) -> Result<(), ProtocolError> {
    validate_request_id(request_id)?;
    validate_artifact_reference(artifact)?;
    validate_artifact_byte_budget(byte_budget)?;
    if offset > 0 && expected_version.is_none() {
        return Err(ProtocolError::InvalidArtifactVersion);
    }
    let ResponseMessage::Artifact {
        request_id: response_id,
        artifact: response_artifact,
        availability,
        offset: response_offset,
        artifact_version,
        bytes,
        ..
    } = response
    else {
        return Err(ProtocolError::InvalidRequest);
    };
    if response_id != request_id || response_artifact != artifact || *response_offset != offset {
        return Err(ProtocolError::InvalidArtifactChunk);
    }
    validate_artifact_response_fields(response)?;
    if bytes.len() > byte_budget as usize
        || (matches!(availability, Availability::Available)
            && expected_version.is_some_and(|expected| artifact_version.as_ref() != Some(expected)))
    {
        return Err(ProtocolError::InvalidArtifactChunk);
    }
    Ok(())
}

pub(crate) fn validate_result_page(
    after_cursor: u64,
    byte_budget: u32,
    page: &ResultPage,
) -> Result<(), ProtocolError> {
    validate_page_budget(byte_budget)?;
    let page_bytes = serde_json::to_vec(page).map_err(|_| ProtocolError::InvalidRequest)?;
    if page_bytes.len() > byte_budget as usize {
        return Err(ProtocolError::InvalidRequest);
    }
    let cursor_validation = match page.availability {
        Availability::Unknown => Ok(()),
        Availability::Expired => {
            if after_cursor <= page.latest_cursor && page.next_cursor == page.latest_cursor {
                Ok(())
            } else {
                Err(ProtocolError::InvalidRequest)
            }
        }
        Availability::Available => {
            if after_cursor > page.latest_cursor {
                Err(ProtocolError::InvalidRequest)
            } else if page.events.is_empty() {
                if page.next_cursor == after_cursor && page.latest_cursor == after_cursor {
                    Ok(())
                } else {
                    Err(ProtocolError::InvalidRequest)
                }
            } else {
                let mut expected = after_cursor
                    .checked_add(1)
                    .ok_or(ProtocolError::InvalidRequest)?;
                for (index, event) in page.events.iter().enumerate() {
                    if event.cursor != expected {
                        return Err(ProtocolError::InvalidRequest);
                    }
                    if index + 1 < page.events.len() {
                        expected = expected
                            .checked_add(1)
                            .ok_or(ProtocolError::InvalidRequest)?;
                    }
                }
                if page.events.last().map(|event| event.cursor) == Some(page.next_cursor) {
                    Ok(())
                } else {
                    Err(ProtocolError::InvalidRequest)
                }
            }
        }
    };
    cursor_validation?;
    // Validate lane-specific event details only after query-relative cursor
    // checks. This also keeps malformed max-u64 sequences from overflowing
    // the page validator's adjacent-cursor check.
    page.validate().map_err(map_result_page_error)
}

fn map_result_page_error(_: ResultBufferError) -> ProtocolError {
    ProtocolError::InvalidRequest
}

pub(crate) fn validate_request_id(value: &str) -> Result<(), ProtocolError> {
    // The frame cap bounds this clone before the shared history validator
    // applies its canonical identifier length, character, and secret rules.
    if value.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::InvalidRequestId);
    }
    let metadata = JobMetadata {
        request_id: Some(value.to_owned()),
        ..JobMetadata::default()
    };
    if validate_job_metadata(&metadata).is_empty() {
        Ok(())
    } else {
        Err(ProtocolError::InvalidRequestId)
    }
}

fn read_payload<R: Read>(reader: &mut R) -> Result<Vec<u8>, ProtocolError> {
    let mut prefix = [0_u8; 4];
    reader.read_exact(&mut prefix).map_err(map_read_error)?;
    let length = u32::from_be_bytes(prefix) as usize;
    if length == 0 {
        return Err(ProtocolError::EmptyFrame);
    }
    if length > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload).map_err(map_read_error)?;
    Ok(payload)
}

fn write_payload<W: Write>(writer: &mut W, payload: &[u8]) -> Result<(), ProtocolError> {
    if payload.is_empty() {
        return Err(ProtocolError::EmptyFrame);
    }
    if payload.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    let length = u32::try_from(payload.len()).map_err(|_| ProtocolError::FrameTooLarge)?;
    writer
        .write_all(&length.to_be_bytes())
        .and_then(|()| writer.write_all(payload))
        .map_err(|_| ProtocolError::Io)
}

fn encode_json<T: Serialize>(value: &T) -> Result<Vec<u8>, ProtocolError> {
    let mut writer = BoundedBuffer::default();
    if serde_json::to_writer(&mut writer, value).is_err() {
        return Err(if writer.exceeded_limit {
            ProtocolError::FrameTooLarge
        } else {
            ProtocolError::MalformedFrame
        });
    }
    if writer.bytes.is_empty() {
        return Err(ProtocolError::EmptyFrame);
    }
    Ok(writer.bytes)
}

#[derive(Default)]
struct BoundedBuffer {
    bytes: Vec<u8>,
    exceeded_limit: bool,
}

impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_FRAME_BYTES.saturating_sub(self.bytes.len()) {
            self.exceeded_limit = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IPC frame exceeds the size limit",
            ));
        }
        if self.bytes.capacity().saturating_sub(self.bytes.len()) < bytes.len() {
            self.bytes
                .try_reserve_exact(bytes.len())
                .map_err(io::Error::other)?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn decode_json<T: for<'de> Deserialize<'de>>(payload: &[u8]) -> Result<T, ProtocolError> {
    serde_json::from_slice(payload).map_err(|_| ProtocolError::MalformedFrame)
}

fn map_read_error(error: io::Error) -> ProtocolError {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        ProtocolError::TruncatedFrame
    } else {
        ProtocolError::Io
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn frame(payload: &[u8]) -> Vec<u8> {
        let length = u32::try_from(payload.len()).unwrap();
        let mut framed = length.to_be_bytes().to_vec();
        framed.extend_from_slice(payload);
        framed
    }

    fn status_envelope(version: u16) -> RequestEnvelope {
        RequestEnvelope {
            version,
            message: WireRequestMessage::Status {
                request_id: "caller:item-17".into(),
            },
        }
    }

    fn result_page(request_id: &str) -> ResultPage {
        use super::super::result_buffer::{ResultBuffer, ResultEvent, ResultTerminal};
        let mut buffer = ResultBuffer::new();
        buffer.begin(request_id, Lane::Language).unwrap();
        buffer
            .append(
                request_id,
                ResultEvent::ChatContent("PRIVATE_RESULT_BODY".into()),
            )
            .unwrap();
        buffer
            .terminal(request_id, ResultTerminal::Succeeded)
            .unwrap();
        buffer.page(request_id, 0, MAX_PAGE_BYTES).unwrap()
    }

    fn artifact_reference() -> MediaArtifact {
        MediaArtifact::new(
            super::super::result_buffer::ArtifactKind::Backend,
            "history/42/output.png",
        )
        .unwrap()
    }

    #[test]
    fn v2_results_round_trip_and_v1_wire_contract_stays_operation_scoped() {
        let request = RequestEnvelope::v2_results("caller:result-1", 0, MIN_PAGE_BYTES as u32);
        let mut request_bytes = Vec::new();
        write_request(&mut request_bytes, &request).unwrap();
        assert!(matches!(
            read_incoming_request(&mut Cursor::new(request_bytes)).unwrap(),
            IncomingRequest::Results(ResultsQuery {
                request_id,
                after_cursor: 0,
                byte_budget,
            }) if request_id == "caller:result-1" && byte_budget == MIN_PAGE_BYTES as u32
        ));
        assert_eq!(
            read_request(&mut Cursor::new({
                let mut bytes = Vec::new();
                write_request(&mut bytes, &request).unwrap();
                bytes
            }))
            .err(),
            Some(ProtocolError::UnsupportedVersion(RESULTS_PROTOCOL_VERSION))
        );

        let mut v1_results = RequestEnvelope::v1(WireRequestMessage::Results {
            request_id: "caller:result-1".into(),
            after_cursor: 0,
            byte_budget: MIN_PAGE_BYTES as u32,
        });
        assert_eq!(
            write_request(&mut Vec::new(), &v1_results),
            Err(ProtocolError::InvalidRequest)
        );
        v1_results.version = RESULTS_PROTOCOL_VERSION;
        v1_results.message = WireRequestMessage::Status {
            request_id: "caller:result-1".into(),
        };
        assert_eq!(
            write_request(&mut Vec::new(), &v1_results),
            Err(ProtocolError::InvalidRequest)
        );
    }

    #[test]
    fn v3_artifact_query_and_hex_response_round_trip_without_changing_older_versions() {
        let artifact = artifact_reference();
        let version_value = "0123456789abcdef".repeat(4);
        let version = ArtifactVersion::new(&version_value).unwrap();
        let request = RequestEnvelope::v3_artifact(
            "caller:artifact-1",
            artifact.clone(),
            7,
            3,
            Some(version.clone()),
        );
        let mut request_bytes = Vec::new();
        write_request(&mut request_bytes, &request).unwrap();
        let IncomingRequest::Artifact(query) =
            read_incoming_request(&mut Cursor::new(request_bytes)).unwrap()
        else {
            panic!("v3 artifact envelope must decode as a read-only artifact query")
        };
        assert_eq!(query.request_id, "caller:artifact-1");
        assert_eq!(query.artifact, artifact);
        assert_eq!(query.offset, 7);
        assert_eq!(query.byte_budget, 3);
        assert_eq!(query.artifact_version, Some(version.clone()));
        let query_debug = format!("{query:?}");
        assert!(!query_debug.contains("caller:artifact-1"));
        assert!(!query_debug.contains("history/42/output.png"));
        assert!(!query_debug.contains(&version_value));
        assert_eq!(
            read_request(&mut Cursor::new({
                let mut bytes = Vec::new();
                write_request(
                    &mut bytes,
                    &RequestEnvelope::v3_artifact(
                        "caller:artifact-1",
                        artifact.clone(),
                        7,
                        3,
                        Some(version.clone()),
                    ),
                )
                .unwrap();
                bytes
            }))
            .err()
            .unwrap(),
            ProtocolError::UnsupportedVersion(ARTIFACT_PROTOCOL_VERSION)
        );

        let response = ResponseEnvelope::v3_artifact(ResponseMessage::Artifact {
            request_id: "caller:artifact-1".into(),
            artifact: artifact.clone(),
            availability: Availability::Available,
            offset: 7,
            total_bytes: 12,
            next_offset: 10,
            end_of_file: false,
            artifact_version: Some(version.clone()),
            bytes: vec![0x00, 0xab, 0xff],
        });
        assert_eq!(
            serde_json::to_value(&response).unwrap()["message"]["bytes_hex"],
            "00abff"
        );
        let mut response_bytes = Vec::new();
        write_response(&mut response_bytes, &response).unwrap();
        let decoded = read_response(&mut Cursor::new(response_bytes)).unwrap();
        assert_eq!(decoded.version, response.version);
        assert_eq!(decoded.message, response.message);
        let debug = format!("{:?}", decoded.message);
        assert!(!debug.contains("caller:artifact-1"));
        assert!(!debug.contains(version.as_str()));
        assert!(!debug.contains("history/42/output.png"));
        assert!(!debug.contains("00abff"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn v3_artifact_queries_and_responses_reject_bad_bounds_and_shapes() {
        let artifact = artifact_reference();
        for byte_budget in [0, MAX_ARTIFACT_CHUNK_BYTES as u32 + 1] {
            assert_eq!(
                write_request(
                    &mut Vec::new(),
                    &RequestEnvelope::v3_artifact(
                        "caller:artifact-1",
                        artifact.clone(),
                        0,
                        byte_budget,
                        None,
                    ),
                ),
                Err(ProtocolError::InvalidRequest)
            );
        }
        assert_eq!(
            write_request(
                &mut Vec::new(),
                &RequestEnvelope::v3_artifact("caller:artifact-1", artifact.clone(), 1, 1, None,),
            ),
            Err(ProtocolError::InvalidArtifactVersion)
        );
        let invalid_artifact = MediaArtifact {
            kind: super::super::result_buffer::ArtifactKind::LocalJob,
            relative_path: "/outside/root.png".into(),
        };
        assert_eq!(
            write_request(
                &mut Vec::new(),
                &RequestEnvelope::v3_artifact("caller:artifact-1", invalid_artifact, 0, 1, None,),
            ),
            Err(ProtocolError::InvalidArtifactReference)
        );
        assert!(ArtifactVersion::new("../private/path").is_err());

        let version = ArtifactVersion::new(&"a".repeat(64)).unwrap();
        for response in [
            ResponseMessage::Artifact {
                request_id: "caller:artifact-1".into(),
                artifact: artifact.clone(),
                availability: Availability::Available,
                offset: 0,
                total_bytes: 2,
                next_offset: 2,
                end_of_file: true,
                artifact_version: None,
                bytes: b"ok".to_vec(),
            },
            ResponseMessage::Artifact {
                request_id: "caller:artifact-1".into(),
                artifact: artifact.clone(),
                availability: Availability::Available,
                offset: 0,
                total_bytes: 2,
                next_offset: 1,
                end_of_file: false,
                artifact_version: Some(version.clone()),
                bytes: b"ok".to_vec(),
            },
            ResponseMessage::Artifact {
                request_id: "caller:artifact-1".into(),
                artifact: artifact.clone(),
                availability: Availability::Expired,
                offset: 0,
                total_bytes: 1,
                next_offset: 1,
                end_of_file: false,
                artifact_version: None,
                bytes: Vec::new(),
            },
        ] {
            assert_eq!(
                write_response(&mut Vec::new(), &ResponseEnvelope::v3_artifact(response)),
                Err(ProtocolError::InvalidArtifactChunk)
            );
        }

        let version = ArtifactVersion::new(&"b".repeat(64)).unwrap();
        let maximum_chunk = ResponseEnvelope::v3_artifact(ResponseMessage::Artifact {
            request_id: "caller:artifact-1".into(),
            artifact: artifact.clone(),
            availability: Availability::Available,
            offset: 0,
            total_bytes: MAX_ARTIFACT_CHUNK_BYTES as u64,
            next_offset: MAX_ARTIFACT_CHUNK_BYTES as u64,
            end_of_file: true,
            artifact_version: Some(version.clone()),
            bytes: vec![0; MAX_ARTIFACT_CHUNK_BYTES],
        });
        let mut encoded_maximum = Vec::new();
        write_response(&mut encoded_maximum, &maximum_chunk).unwrap();
        assert!(encoded_maximum.len() < MAX_FRAME_BYTES);
        assert_eq!(
            write_response(
                &mut Vec::new(),
                &ResponseEnvelope::v3_artifact(ResponseMessage::Artifact {
                    request_id: "caller:artifact-1".into(),
                    artifact,
                    availability: Availability::Available,
                    offset: 0,
                    total_bytes: (MAX_ARTIFACT_CHUNK_BYTES + 1) as u64,
                    next_offset: (MAX_ARTIFACT_CHUNK_BYTES + 1) as u64,
                    end_of_file: true,
                    artifact_version: Some(version),
                    bytes: vec![0; MAX_ARTIFACT_CHUNK_BYTES + 1],
                }),
            ),
            Err(ProtocolError::InvalidArtifactChunk)
        );

        let invalid_hex = frame(
            br#"{"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png"},"availability":"available","offset":0,"total_bytes":2,"next_offset":2,"end_of_file":true,"artifact_version":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","bytes_hex":"0G"}}"#,
        );
        assert_eq!(
            read_response(&mut Cursor::new(invalid_hex)).err().unwrap(),
            ProtocolError::MalformedFrame
        );
    }

    #[test]
    fn v3_artifact_wire_objects_reject_duplicate_and_unknown_fields() {
        let invalid = [
            br#"{"version":3,"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png"},"offset":0,"byte_budget":1,"artifact_version":null}}"#.as_slice(),
            br#"{"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png"},"offset":0,"offset":0,"byte_budget":1,"artifact_version":null}}"#.as_slice(),
            br#"{"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png","extra":true},"offset":0,"byte_budget":1,"artifact_version":null}}"#.as_slice(),
            br#"{"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png"},"offset":0,"byte_budget":1,"artifact_version":null,"url":"file:///private"}}"#.as_slice(),
            br#"{"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png"},"offset":0,"byte_budget":1,"artifact_version":"../private"}}"#.as_slice(),
        ];
        for bytes in invalid {
            assert_eq!(
                read_incoming_request(&mut Cursor::new(frame(bytes)))
                    .err()
                    .unwrap(),
                ProtocolError::MalformedFrame
            );
        }

        let invalid_responses = [
            br#"{"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png"},"availability":"available","offset":0,"offset":0,"total_bytes":1,"next_offset":1,"end_of_file":true,"artifact_version":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","bytes_hex":"61"}}"#.as_slice(),
            br#"{"version":3,"message":{"type":"artifact","request_id":"caller:artifact-1","artifact":{"kind":"backend","relative_path":"history/42/output.png"},"availability":"available","offset":0,"total_bytes":1,"next_offset":1,"end_of_file":true,"artifact_version":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","bytes_hex":"61","local_path":"/private"}}"#.as_slice(),
        ];
        for bytes in invalid_responses {
            assert_eq!(
                read_response(&mut Cursor::new(frame(bytes))).err().unwrap(),
                ProtocolError::MalformedFrame
            );
        }
    }

    #[test]
    fn results_budget_cursor_and_response_version_pairing_are_strict() {
        for byte_budget in [(MIN_PAGE_BYTES - 1) as u32, (MAX_PAGE_BYTES + 1) as u32] {
            let request = RequestEnvelope::v2_results("caller:result-1", 0, byte_budget);
            assert_eq!(
                write_request(&mut Vec::new(), &request),
                Err(ProtocolError::InvalidRequest)
            );
        }

        let page = result_page("caller:result-1");
        let response = ResponseEnvelope::v2_results(ResponseMessage::Results {
            request_id: "caller:result-1".into(),
            after_cursor: 0,
            byte_budget: MIN_PAGE_BYTES as u32,
            state: RequestStatus::Succeeded,
            page: page.clone(),
        });
        let mut bytes = Vec::new();
        write_response(&mut bytes, &response).unwrap();
        let decoded = read_response(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(decoded.version, response.version);
        assert_eq!(decoded.message, response.message);

        let mut gap = page.clone();
        gap.events[0].cursor = 2;
        gap.next_cursor = 2;
        gap.latest_cursor = 2;
        assert_eq!(
            write_response(
                &mut Vec::new(),
                &ResponseEnvelope::v2_results(ResponseMessage::Results {
                    request_id: "caller:result-1".into(),
                    after_cursor: 0,
                    byte_budget: MIN_PAGE_BYTES as u32,
                    state: RequestStatus::Succeeded,
                    page: gap,
                }),
            ),
            Err(ProtocolError::InvalidRequest)
        );

        let wrong_version = ResponseEnvelope::v1(ResponseMessage::Results {
            request_id: "caller:result-1".into(),
            after_cursor: 0,
            byte_budget: MIN_PAGE_BYTES as u32,
            state: RequestStatus::Succeeded,
            page,
        });
        assert_eq!(
            write_response(&mut Vec::new(), &wrong_version),
            Err(ProtocolError::InvalidRequest)
        );

        use super::super::result_buffer::{ResultBuffer, ResultEvent, ResultTerminal};
        let mut large_buffer = ResultBuffer::new();
        large_buffer
            .begin("caller:large-page", Lane::Language)
            .unwrap();
        for _ in 0..20 {
            large_buffer
                .append(
                    "caller:large-page",
                    ResultEvent::ChatContent("x".repeat(64)),
                )
                .unwrap();
        }
        large_buffer
            .terminal("caller:large-page", ResultTerminal::Succeeded)
            .unwrap();
        let large_page = large_buffer
            .page("caller:large-page", 0, MAX_PAGE_BYTES)
            .unwrap();
        assert!(serde_json::to_vec(&large_page).unwrap().len() > MIN_PAGE_BYTES);
        assert_eq!(
            write_response(
                &mut Vec::new(),
                &ResponseEnvelope::v2_results(ResponseMessage::Results {
                    request_id: "caller:large-page".into(),
                    after_cursor: 0,
                    byte_budget: MIN_PAGE_BYTES as u32,
                    state: RequestStatus::Succeeded,
                    page: large_page,
                }),
            ),
            Err(ProtocolError::InvalidRequest)
        );

        let mut expiring = ResultBuffer::new();
        expiring.begin("caller:expired", Lane::Language).unwrap();
        expiring
            .append("caller:expired", ResultEvent::ChatContent("gone".into()))
            .unwrap();
        expiring
            .terminal("caller:expired", ResultTerminal::Succeeded)
            .unwrap();
        expiring.evict_terminal("caller:expired").unwrap();
        let expired = expiring.page("caller:expired", 0, MIN_PAGE_BYTES).unwrap();
        assert_eq!(expired.availability, Availability::Expired);
        assert_eq!(expired.next_cursor, 1); // Tombstones can safely skip lost bodies.
        let expired_response = ResponseEnvelope::v2_results(ResponseMessage::Results {
            request_id: "caller:expired".into(),
            after_cursor: 0,
            byte_budget: MIN_PAGE_BYTES as u32,
            state: RequestStatus::Succeeded,
            page: expired,
        });
        write_response(&mut Vec::new(), &expired_response).unwrap();
    }

    #[test]
    fn decoded_envelopes_reject_version_operation_and_budget_mismatches() {
        let invalid_requests = [
            RequestEnvelope::v1(WireRequestMessage::Results {
                request_id: "caller:result-1".into(),
                after_cursor: 0,
                byte_budget: MIN_PAGE_BYTES as u32,
            }),
            RequestEnvelope {
                version: RESULTS_PROTOCOL_VERSION,
                message: WireRequestMessage::Status {
                    request_id: "caller:result-1".into(),
                },
            },
            RequestEnvelope::v2_results("caller:result-1", 0, 0),
        ];
        for request in invalid_requests {
            let bytes = frame(&serde_json::to_vec(&request).unwrap());
            assert_eq!(
                read_incoming_request(&mut Cursor::new(bytes)).err(),
                Some(ProtocolError::InvalidRequest)
            );
        }

        let wrong_version_response = serde_json::json!({
            "version": PROTOCOL_VERSION,
            "message": {
                "type": "results",
                "request_id": "caller:result-1",
                "after_cursor": 0,
                "byte_budget": MIN_PAGE_BYTES,
                "state": "succeeded",
                "page": result_page("caller:result-1")
            }
        });
        let bytes = frame(&serde_json::to_vec(&wrong_version_response).unwrap());
        assert_eq!(
            read_response(&mut Cursor::new(bytes)).err(),
            Some(ProtocolError::InvalidRequest)
        );
    }

    #[test]
    fn result_cursor_overflow_is_rejected_without_panicking() {
        use super::super::result_buffer::{CursorEvent, PageEvent};
        let page = ResultPage {
            request_id: "caller:cursor-overflow".into(),
            availability: Availability::Available,
            events: vec![
                CursorEvent {
                    cursor: u64::MAX,
                    event: PageEvent::ChatContent {
                        text: "last".into(),
                    },
                },
                CursorEvent {
                    cursor: 1,
                    event: PageEvent::ChatContent {
                        text: "invalid".into(),
                    },
                },
            ],
            next_cursor: 1,
            latest_cursor: u64::MAX,
            terminal: None,
        };
        assert_eq!(
            validate_result_page(u64::MAX - 1, MIN_PAGE_BYTES as u32, &page),
            Err(ProtocolError::InvalidRequest)
        );
    }

    #[test]
    fn results_debug_redacts_text_and_artifact_paths() {
        use super::super::result_buffer::{ArtifactKind, CursorEvent, MediaArtifact, PageEvent};
        let page = ResultPage {
            request_id: "caller:private".into(),
            availability: Availability::Available,
            events: vec![
                CursorEvent {
                    cursor: 1,
                    event: PageEvent::ChatContent {
                        text: "PRIVATE_RESULT_BODY".into(),
                    },
                },
                CursorEvent {
                    cursor: 2,
                    event: PageEvent::MediaArtifactBatch {
                        result_id: 2,
                        batch_index: 0,
                        batch_count: 1,
                        artifacts: vec![MediaArtifact::new(
                            ArtifactKind::LocalJob,
                            "private/output-file.wav",
                        )
                        .unwrap()],
                    },
                },
            ],
            next_cursor: 2,
            latest_cursor: 2,
            terminal: None,
        };
        let response = ResponseMessage::Results {
            request_id: "caller:private".into(),
            after_cursor: 0,
            byte_budget: MAX_PAGE_BYTES as u32,
            state: RequestStatus::Running,
            page,
        };
        let debug = format!("{response:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("PRIVATE_RESULT_BODY"));
        assert!(!debug.contains("private/output-file.wav"));
        assert!(!debug.contains("caller:private"));
    }

    #[test]
    fn request_frame_round_trips_and_uses_bounded_big_endian_length() {
        let envelope = RequestEnvelope::v1(WireRequestMessage::Status {
            request_id: "caller:item-17".into(),
        });
        let mut bytes = Vec::new();
        write_request(&mut bytes, &envelope).unwrap();
        let length = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert_eq!(length, bytes.len() - 4);
        assert!(length <= MAX_FRAME_BYTES);
        assert!(matches!(
            read_request(&mut Cursor::new(bytes)).unwrap(),
            ClientRequest::Status { request_id } if request_id == "caller:item-17"
        ));
    }

    #[test]
    fn rejects_empty_oversized_and_truncated_frames_before_unbounded_reading() {
        assert_eq!(
            read_request(&mut Cursor::new(0_u32.to_be_bytes())).err(),
            Some(ProtocolError::EmptyFrame)
        );
        let oversized = u32::try_from(MAX_FRAME_BYTES + 1).unwrap().to_be_bytes();
        assert_eq!(
            read_request(&mut Cursor::new(oversized)).err(),
            Some(ProtocolError::FrameTooLarge)
        );
        assert_eq!(
            read_request(&mut Cursor::new([0_u8, 0_u8])).err(),
            Some(ProtocolError::TruncatedFrame)
        );
        assert_eq!(
            read_request(&mut Cursor::new(frame(b"{}"))).err(),
            Some(ProtocolError::MalformedFrame)
        );
        assert_eq!(
            read_request(&mut Cursor::new(frame(b"{\"version\":1"))).err(),
            Some(ProtocolError::MalformedFrame)
        );
    }

    #[test]
    fn serialization_stops_at_frame_limit_without_writing_a_partial_frame() {
        let oversized = "x".repeat(MAX_FRAME_BYTES + 1);
        assert_eq!(
            encode_json(&oversized).err(),
            Some(ProtocolError::FrameTooLarge)
        );

        let envelope = RequestEnvelope::v1(WireRequestMessage::Submit {
            request_id: "large:request".into(),
            timeout_ms: 1_000,
            request: WireInferenceRequest::Language {
                profile_id: "model".into(),
                turns: vec![WireChatTurn {
                    role: WireChatRole::User,
                    content: oversized,
                }],
                max_tokens: 1,
            },
        });
        let mut output = Vec::new();
        assert_eq!(
            write_request(&mut output, &envelope),
            Err(ProtocolError::FrameTooLarge)
        );
        assert!(output.is_empty());
    }

    #[test]
    fn rejects_truncated_payload_unknown_fields_and_unsupported_versions() {
        let mut truncated = 12_u32.to_be_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        assert_eq!(
            read_request(&mut Cursor::new(truncated)).err(),
            Some(ProtocolError::TruncatedFrame)
        );

        let unknown = frame(
            br#"{"version":1,"message":{"type":"status","request_id":"req-1","extra":true}}"#,
        );
        assert_eq!(
            read_request(&mut Cursor::new(unknown)).err(),
            Some(ProtocolError::MalformedFrame)
        );

        let mut bytes = Vec::new();
        write_payload(
            &mut bytes,
            &serde_json::to_vec(&status_envelope(99)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            read_request(&mut Cursor::new(bytes)).err(),
            Some(ProtocolError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn rejects_unsafe_unbounded_and_secret_like_request_ids() {
        for request_id in [
            "",
            "has spaces",
            "sk-test-secret",
            "akia-test",
            &"x".repeat(129), // One byte beyond the persisted 128-byte identifier cap.
        ] {
            let envelope = RequestEnvelope::v1(WireRequestMessage::Status {
                request_id: request_id.into(),
            });
            let mut bytes = Vec::new();
            assert_eq!(
                write_request(&mut bytes, &envelope),
                Err(ProtocolError::InvalidRequestId)
            );
            assert!(bytes.is_empty());
        }
    }

    #[test]
    fn maps_language_roles_and_preserves_native_turns() {
        let envelope = RequestEnvelope::v1(WireRequestMessage::Submit {
            request_id: "chat:1".into(),
            timeout_ms: 5_000,
            request: WireInferenceRequest::Language {
                profile_id: "gemma-4-26b-qat-mtp".into(),
                turns: vec![
                    WireChatTurn {
                        role: WireChatRole::System,
                        content: "system prompt".into(),
                    },
                    WireChatTurn {
                        role: WireChatRole::User,
                        content: "user prompt".into(),
                    },
                    WireChatTurn {
                        role: WireChatRole::Assistant,
                        content: "prior answer".into(),
                    },
                ],
                max_tokens: 64,
            },
        });
        let ClientRequest::Submit {
            request, timeout, ..
        } = envelope.into_client_request().unwrap()
        else {
            panic!("expected submit")
        };
        assert_eq!(timeout, Duration::from_secs(5));
        let InferenceRequest::Chat {
            profile,
            turns,
            max_tokens,
        } = request
        else {
            panic!("expected chat")
        };
        assert_eq!(profile.to_string(), "language:gemma-4-26b-qat-mtp");
        assert_eq!(max_tokens, 64);
        assert!(matches!(turns[0].role, ChatRole::System));
        assert!(matches!(turns[1].role, ChatRole::User));
        assert!(matches!(turns[2].role, ChatRole::Assistant));
        assert_eq!(turns[0].content, "system prompt");
    }

    #[test]
    fn legacy_chat_wire_bytes_are_unchanged_and_embeddings_use_explicit_task() {
        let chat = WireInferenceRequest::Language {
            profile_id: "model".into(),
            turns: vec![WireChatTurn {
                role: WireChatRole::User,
                content: "hello".into(),
            }],
            max_tokens: 8,
        };
        assert_eq!(
            serde_json::to_string(&chat).unwrap(),
            r#"{"lane":"language","profile_id":"model","turns":[{"role":"user","content":"hello"}],"max_tokens":8}"#
        );

        let embedding = WireInferenceRequest::Embeddings {
            profile_id: "nomic-fixture".into(),
            input: vec!["first".into(), "second".into()],
        };
        assert_eq!(
            serde_json::to_string(&embedding).unwrap(),
            r#"{"lane":"language","task":"embeddings","profile_id":"nomic-fixture","input":["first","second"]}"#
        );
        assert!(matches!(
            embedding.into_inference_request().unwrap(),
            InferenceRequest::Embeddings { profile, input }
                if profile.to_string() == "language:nomic-fixture"
                    && input == vec!["first".to_owned(), "second".to_owned()]
        ));
    }

    #[test]
    fn embeddings_wire_requires_one_strict_explicit_shape() {
        let valid: WireInferenceRequest = serde_json::from_str(
            r#"{"lane":"language","task":"embeddings","profile_id":"nomic-fixture","input":["hello"]}"#,
        )
        .unwrap();
        assert!(matches!(valid, WireInferenceRequest::Embeddings { .. }));

        for invalid in [
            r#"{"lane":"language","task":null,"profile_id":"model","input":["x"]}"#,
            r#"{"lane":"language","task":"embeddings","profile_id":"model"}"#,
            r#"{"lane":"language","task":"embeddings","profile_id":"model","input":null}"#,
            r#"{"lane":"language","task":"embeddings","profile_id":"model","input":{"kind":"workflow","task":"image","graph":{}}}"#,
            r#"{"lane":"language","task":"embeddings","profile_id":"model","input":["x"],"turns":[],"max_tokens":1}"#,
            r#"{"lane":"language","task":"chat","profile_id":"model","turns":[],"max_tokens":1}"#,
            r#"{"lane":"language","task":"embeddings","profile_id":"model","input":["x"],"extra":true}"#,
            r#"{"lane":"media","profile_id":"model","input":["x"]}"#,
            r#"{"lane":"language","profile_id":"model","turns":[],"max_tokens":1,"input":["x"]}"#,
            r#"{"lane":"language","profile_id":"model","profile_id":"other","turns":[],"max_tokens":1}"#,
            r#"{"lane":"language","lane":"media","profile_id":"model","turns":[],"max_tokens":1}"#,
            r#"{"lane":"language","task":"embeddings","task":"chat","profile_id":"model","input":["x"]}"#,
        ] {
            assert!(
                serde_json::from_str::<WireInferenceRequest>(invalid).is_err(),
                "accepted malformed wire request: {invalid}"
            );
        }
    }

    #[test]
    fn maps_workflow_script_and_every_speech_mode_without_flattening_inputs() {
        let workflow = WireInferenceRequest::Media {
            profile_id: "comfyui".into(),
            input: WireMediaInput::Workflow {
                task: "image".into(),
                graph: serde_json::json!({"nodes":{"7":{"class_type":"LoadImage"}}}),
            },
        }
        .into_inference_request()
        .unwrap();
        assert!(matches!(
            workflow,
            InferenceRequest::Media {
                input: MediaInput::Workflow { ref task, ref graph },
                ..
            } if task.as_str() == "image" && graph["nodes"]["7"]["class_type"] == "LoadImage"
        ));

        let script = WireInferenceRequest::Media {
            profile_id: "breeze-tts-2".into(),
            input: WireMediaInput::Script {
                task: "tts".into(),
                args: vec!["--stream".into(), "true".into()],
            },
        }
        .into_inference_request()
        .unwrap();
        assert!(matches!(
            script,
            InferenceRequest::Media {
                input: MediaInput::Script { ref args, .. },
                ..
            } if args == &["--stream", "true"]
        ));

        for (wire_mode, native_mode) in [
            (WireSpeechMode::Standard, SpeechMode::Standard),
            (WireSpeechMode::VoiceDesign, SpeechMode::VoiceDesign),
            (WireSpeechMode::VoiceClone, SpeechMode::VoiceClone),
            (WireSpeechMode::Streaming, SpeechMode::Streaming),
        ] {
            let speech = WireInferenceRequest::Media {
                profile_id: "breeze-tts-2".into(),
                input: WireMediaInput::Speech {
                    text: "spoken text".into(),
                    mode: wire_mode,
                    instruction: Some("warm voice".into()),
                    voice_reference: Some("voices/reference.wav".into()),
                    reference_text: Some("reference transcript".into()),
                },
            }
            .into_inference_request()
            .unwrap();
            assert!(matches!(
                speech,
                InferenceRequest::Media {
                    input: MediaInput::Speech(SpeechInput {
                        text,
                        mode,
                        instruction: Some(instruction),
                        voice_reference: Some(path),
                        reference_text: Some(reference_text),
                    }),
                    ..
                } if text == "spoken text"
                    && mode == native_mode
                    && instruction == "warm voice"
                    && path == std::path::Path::new("voices/reference.wav")
                    && reference_text == "reference transcript"
            ));
        }
    }

    #[test]
    fn maps_native_decision_state_and_question_definitions() {
        let request = WireInferenceRequest::Decisions {
            profile_id: "nagi-smol".into(),
            state: serde_json::json!({"position":{"x":2,"y":4}}),
            questions: serde_json::json!({
                "move": {"type":"choice", "criteria":{"left":"Go left", "stay":"Wait"}},
                "urgency": {"type":"score", "range":[0,100]}
            })
            .as_object()
            .unwrap()
            .clone(),
        }
        .into_inference_request()
        .unwrap();
        let InferenceRequest::Decision {
            profile,
            state,
            questions,
            images,
        } = request
        else {
            panic!("expected decision request")
        };
        assert_eq!(profile.to_string(), "decisions:nagi-smol");
        assert_eq!(state["position"]["x"], 2);
        assert_eq!(questions["move"]["criteria"]["left"], "Go left");
        assert_eq!(questions["urgency"]["range"][1], 100);
        assert!(images.is_empty());
    }

    #[test]
    fn response_status_is_keyed_by_request_id_and_has_no_job_number() {
        let envelope = ResponseEnvelope::v1(ResponseMessage::Status {
            request_id: "caller:item-17".into(),
            state: RequestStatus::Interrupted,
        });
        let mut bytes = Vec::new();
        write_response(&mut bytes, &envelope).unwrap();
        let decoded = read_response(&mut Cursor::new(bytes)).unwrap();
        assert!(matches!(
            decoded.message,
            ResponseMessage::Status {
                request_id,
                state: RequestStatus::Interrupted,
            } if request_id == "caller:item-17"
        ));
    }

    #[test]
    fn cancel_and_acceptance_error_messages_round_trip_by_caller_id() {
        let cancel = RequestEnvelope::v1(WireRequestMessage::Cancel {
            request_id: "caller:item-17".into(),
        })
        .into_client_request()
        .unwrap();
        assert!(matches!(
            cancel,
            ClientRequest::Cancel { request_id } if request_id == "caller:item-17"
        ));

        for message in [
            ResponseMessage::Accepted {
                request_id: "caller:item-17".into(),
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued,
            },
            ResponseMessage::Error {
                request_id: Some("caller:item-17".into()),
                code: ResponseErrorCode::DuplicateRequest,
            },
        ] {
            let mut bytes = Vec::new();
            write_response(&mut bytes, &ResponseEnvelope::v1(message.clone())).unwrap();
            assert_eq!(
                read_response(&mut Cursor::new(bytes)).unwrap().message,
                message
            );
        }
    }

    #[test]
    fn response_unknown_fields_are_rejected() {
        let bytes = frame(
            br#"{"version":1,"message":{"type":"status","request_id":"caller:item-17","state":"running","job_id":42}}"#,
        );
        assert_eq!(
            read_response(&mut Cursor::new(bytes)).err(),
            Some(ProtocolError::MalformedFrame)
        );
    }

    #[test]
    fn malformed_error_text_never_echoes_request_content() {
        let secret_prompt = "private prompt content";
        let malformed = frame(format!("{{\"version\":1,\"message\":{secret_prompt}").as_bytes());
        let error = read_request(&mut Cursor::new(malformed)).err().unwrap();
        assert_eq!(error, ProtocolError::MalformedFrame);
        assert!(!error.to_string().contains(secret_prompt));
    }
}
