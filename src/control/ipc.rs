//! Versioned, single-frame IPC protocol contracts for the future coordinator.
//!
//! This module only defines the wire format and bounded framing. It does not
//! open sockets, authenticate peers, acquire a host-wide lock, dispatch work,
//! or perform lifecycle operations. A future daemon must supply those
//! guarantees separately. The generic `Read`/`Write` helpers are blocking and
//! do not enforce IO deadlines; a future transport must bound peer IO outside
//! this module. Requests intentionally have no `Debug` output so prompts,
//! workflow graphs, speech text, and native arguments are not dumped
//! accidentally by diagnostic logging. Version 1 returns acceptance and
//! lifecycle status only; result bodies, artifact retrieval, and streaming
//! require an explicit versioned protocol extension.

use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::catalog::{Lane, ProfileKey, Task};
use super::requests::{ChatRole, ChatTurn, InferenceRequest, MediaInput, SpeechInput, SpeechMode};
use crate::state_store::{validate_job_metadata, JobMetadata};

pub const PROTOCOL_VERSION: u16 = 1;
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
        }
    }
}

impl Error for ProtocolError {}

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
}

/// Native inference inputs remain explicitly lane-qualified on the wire.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "lane", rename_all = "snake_case", deny_unknown_fields)]
pub enum WireInferenceRequest {
    Language {
        profile_id: String,
        turns: Vec<WireChatTurn>,
        max_tokens: u32,
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
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
        code: ResponseErrorCode,
    },
}

/// Read exactly one bounded length-prefixed JSON frame and decode a request.
/// The 4-byte big-endian length is checked before allocating the frame buffer.
pub fn read_request<R: Read>(reader: &mut R) -> Result<ClientRequest, ProtocolError> {
    let payload = read_payload(reader)?;
    let envelope: RequestEnvelope = decode_json(&payload)?;
    envelope.into_client_request()
}

/// Serialize and write one request frame. Version, request ID, and timeout are
/// checked before any bytes are written; full catalog validation belongs to
/// the coordinator after decoding.
pub fn write_request<W: Write>(
    writer: &mut W,
    envelope: &RequestEnvelope,
) -> Result<(), ProtocolError> {
    validate_version(envelope.version)?;
    validate_wire_request_headers(&envelope.message)?;
    let payload = encode_json(envelope)?;
    write_payload(writer, &payload)
}

pub fn read_response<R: Read>(reader: &mut R) -> Result<ResponseEnvelope, ProtocolError> {
    let payload = read_payload(reader)?;
    let envelope: ResponseEnvelope = decode_json(&payload)?;
    validate_version(envelope.version)?;
    validate_response_ids(&envelope.message)?;
    Ok(envelope)
}

pub fn write_response<W: Write>(
    writer: &mut W,
    envelope: &ResponseEnvelope,
) -> Result<(), ProtocolError> {
    validate_version(envelope.version)?;
    validate_response_ids(&envelope.message)?;
    let payload = encode_json(envelope)?;
    write_payload(writer, &payload)
}

fn validate_version(version: u16) -> Result<(), ProtocolError> {
    if version == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(ProtocolError::UnsupportedVersion(version))
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
    }
    Ok(())
}

fn validate_response_ids(message: &ResponseMessage) -> Result<(), ProtocolError> {
    match message {
        ResponseMessage::Accepted { request_id, .. }
        | ResponseMessage::Status { request_id, .. } => validate_request_id(request_id),
        ResponseMessage::Error { request_id, .. } => {
            if let Some(request_id) = request_id {
                validate_request_id(request_id)?;
            }
            Ok(())
        }
    }
}

fn validate_request_id(value: &str) -> Result<(), ProtocolError> {
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
        } = request
        else {
            panic!("expected decision request")
        };
        assert_eq!(profile.to_string(), "decisions:nagi-smol");
        assert_eq!(state["position"]["x"], 2);
        assert_eq!(questions["move"]["criteria"]["left"], "Go left");
        assert_eq!(questions["urgency"]["range"][1], 100);
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
