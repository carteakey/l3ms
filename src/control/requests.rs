//! Modality-specific in-memory request and result contracts.
//!
//! These types describe work before a runtime adapter or scheduler accepts it.
//! Validation checks manifest ownership and basic input invariants only: it is
//! not an installation check, resource lease, admission barrier, or permission
//! to launch an engine. Native payloads and prompts must not be written to job
//! history; the persistent record stores bounded metadata instead.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{bail, ensure, Result};
use serde_json::{Map, Value};

use super::catalog::{Catalog, CatalogProfile, Lane, NativeProfileSource, ProfileKey, Task};

const MAX_TURNS: usize = 128;
const MAX_QUESTIONS: usize = 128;
const MAX_SCRIPT_ARGS: usize = 128;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_DECISION_IMAGES: usize = 8;
pub const MAX_EMBEDDING_INPUTS: usize = 128;
pub const MAX_EMBEDDING_DIMENSIONS: usize = 65_536;
pub const MAX_EMBEDDING_JSON_BYTES: usize = 4 * 1024 * 1024;

fn ensure_text_budget<'a>(values: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let total = values
        .into_iter()
        .fold(0usize, |bytes, value| bytes.saturating_add(value.len()));
    ensure!(total <= MAX_REQUEST_BYTES, "request text exceeds 1 MiB");
    Ok(())
}

fn ensure_json_budget(value: &Value) -> Result<()> {
    ensure!(
        serde_json::to_vec(value)?.len() <= MAX_REQUEST_BYTES,
        "request JSON exceeds 1 MiB"
    );
    Ok(())
}

pub(crate) fn decision_profile_supports_images(profile: &CatalogProfile) -> bool {
    !profile.disabled
        && profile.key.lane == Lane::Decisions
        && profile.runtime.lane == Lane::Decisions
        && profile.runtime.as_str() == "llamacpp"
        && profile.tasks.iter().any(|task| task.as_str() == "decision")
        && matches!(
            &profile.native_source,
            NativeProfileSource::Decisions {
                model,
                profile_type,
                ..
            } if profile_type == "local"
                && matches!(model.as_str(), "vllm-d3-flash" | "cloudflare-clef-flash")
        )
}

pub(crate) fn validate_decision_images(images: &[String]) -> Result<()> {
    ensure!(
        images.len() <= MAX_DECISION_IMAGES,
        "decision image count exceeds the native limit"
    );
    ensure!(
        images.iter().all(|image| valid_image_data_url(image)),
        "decision images must be base64 data:image URLs"
    );
    Ok(())
}

fn valid_image_data_url(value: &str) -> bool {
    let Some((metadata, encoded)) = value.split_once(',') else {
        return false;
    };
    let Some(media_type) = metadata
        .strip_prefix("data:")
        .and_then(|value| value.strip_suffix(";base64"))
    else {
        return false;
    };
    let Some(subtype) = media_type.strip_prefix("image/") else {
        return false;
    };
    if subtype.is_empty()
        || !subtype.as_bytes()[0].is_ascii_alphanumeric()
        || !subtype
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
        || encoded.is_empty()
        || encoded.len() % 4 != 0
    {
        return false;
    }
    let bytes = encoded.as_bytes();
    let padding_start = bytes
        .iter()
        .position(|byte| *byte == b'=')
        .unwrap_or(bytes.len());
    padding_start > 0
        && bytes.len().saturating_sub(padding_start) <= 2
        && padding_start >= bytes.len().saturating_sub(2)
        && bytes[..padding_start]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
        && bytes[padding_start..].iter().all(|byte| *byte == b'=')
}

/// A language turn. A chat adapter translates this to its native wire shape.
#[derive(Clone)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// A speech operation remains distinct from image/video workflow payloads.
#[derive(Clone)]
pub struct SpeechInput {
    pub text: String,
    pub mode: SpeechMode,
    pub instruction: Option<String>,
    pub voice_reference: Option<PathBuf>,
    pub reference_text: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechMode {
    Standard,
    VoiceDesign,
    VoiceClone,
    Streaming,
}

impl SpeechMode {
    fn native_task(self) -> &'static str {
        match self {
            Self::Standard | Self::Streaming => "tts",
            Self::VoiceDesign => "design",
            Self::VoiceClone => "clon",
        }
    }
}

/// Native media dispatch shapes; a media adapter owns task-specific parsing.
#[derive(Clone)]
pub enum MediaInput {
    /// A native ComfyUI graph, kept as a graph rather than a universal prompt.
    Workflow { task: Task, graph: Value },
    /// Arguments for the native L2M2 profile script. Never persist these.
    Script { task: Task, args: Vec<String> },
    /// Breeze-style speech parameters; the adapter chooses native flags.
    Speech(SpeechInput),
}

impl MediaInput {
    fn task(&self) -> &str {
        match self {
            Self::Workflow { task, .. } | Self::Script { task, .. } => task.as_str(),
            Self::Speech(input) => input.mode.native_task(),
        }
    }
}

/// One lane-specific request. The qualified profile key cannot be silently
/// resolved to an identically named profile in another toolkit.
#[derive(Clone)]
pub enum InferenceRequest {
    Chat {
        profile: ProfileKey,
        turns: Vec<ChatTurn>,
        max_tokens: u32,
    },
    /// Bounded text batch for the native OpenAI-compatible embeddings endpoint.
    Embeddings {
        profile: ProfileKey,
        input: Vec<String>,
    },
    Media {
        profile: ProfileKey,
        input: MediaInput,
    },
    Decision {
        profile: ProfileKey,
        state: Value,
        /// Native `system_one` question definitions keyed by question ID.
        questions: Map<String, Value>,
        /// Optional local llama.cpp image inputs as bounded base64 data URLs.
        images: Vec<String>,
    },
}

impl InferenceRequest {
    pub fn profile(&self) -> &ProfileKey {
        match self {
            Self::Chat { profile, .. }
            | Self::Embeddings { profile, .. }
            | Self::Media { profile, .. }
            | Self::Decision { profile, .. } => profile,
        }
    }

    pub fn lane(&self) -> Lane {
        match self {
            Self::Chat { .. } | Self::Embeddings { .. } => Lane::Language,
            Self::Media { .. } => Lane::Media,
            Self::Decision { .. } => Lane::Decisions,
        }
    }

    pub fn task(&self) -> &str {
        match self {
            Self::Chat { .. } => "chat",
            Self::Embeddings { .. } => "embeddings",
            Self::Media { input, .. } => input.task(),
            Self::Decision { .. } => "decision",
        }
    }

    /// Validate catalog ownership and bounded, lane-specific input shape.
    /// The returned profile is still only `manifest-present` unless separately
    /// checked by a runtime adapter.
    pub fn validate<'a>(&self, catalog: &'a Catalog) -> Result<&'a CatalogProfile> {
        let key = self.profile();
        ensure!(
            key.lane == self.lane(),
            "request lane does not match profile {key}"
        );
        let profile = catalog
            .inspect(key)
            .ok_or_else(|| anyhow::anyhow!("unknown profile {key}"))?;
        ensure!(!profile.disabled, "profile {key} is disabled");
        ensure!(
            profile
                .tasks
                .iter()
                .any(|task| task.as_str() == self.task()),
            "profile {key} does not declare task {}",
            self.task()
        );

        match self {
            Self::Chat {
                turns, max_tokens, ..
            } => {
                ensure!(
                    (1..=MAX_TURNS).contains(&turns.len()),
                    "chat turn count is out of range"
                );
                ensure!(*max_tokens > 0, "max_tokens must be positive");
                ensure!(
                    turns.iter().all(|turn| !turn.content.trim().is_empty()),
                    "chat turns must not be empty"
                );
                ensure_text_budget(turns.iter().map(|turn| turn.content.as_str()))?;
            }
            Self::Embeddings { input, .. } => {
                ensure!(
                    (1..=MAX_EMBEDDING_INPUTS).contains(&input.len()),
                    "embedding input count is out of range"
                );
                ensure!(
                    input.iter().all(|text| !text.trim().is_empty()),
                    "embedding inputs must not be empty"
                );
                ensure!(
                    input.iter().all(|text| !text.contains('\0')),
                    "embedding input contains NUL"
                );
                ensure_text_budget(input.iter().map(String::as_str))?;
            }
            Self::Media { input, .. } => match input {
                MediaInput::Workflow { graph, .. } => {
                    ensure!(graph.is_object(), "media workflow must be an object");
                    ensure_json_budget(graph)?;
                }
                MediaInput::Script { args, .. } => {
                    ensure!(
                        args.len() <= MAX_SCRIPT_ARGS,
                        "too many native script arguments"
                    );
                    ensure!(
                        args.iter().all(|arg| !arg.contains('\0')),
                        "native script argument contains NUL"
                    );
                    ensure_text_budget(args.iter().map(String::as_str))?;
                }
                MediaInput::Speech(input) => {
                    ensure!(
                        !input.text.trim().is_empty(),
                        "speech text must not be empty"
                    );
                    ensure_text_budget(
                        [
                            Some(input.text.as_str()),
                            input.instruction.as_deref(),
                            input.reference_text.as_deref(),
                        ]
                        .into_iter()
                        .flatten(),
                    )?;
                    match input.mode {
                        SpeechMode::VoiceClone => {
                            ensure!(
                                input
                                    .voice_reference
                                    .as_ref()
                                    .is_some_and(|path| !path.as_os_str().is_empty()),
                                "voice cloning requires reference audio"
                            );
                        }
                        SpeechMode::VoiceDesign => {
                            ensure!(
                                input
                                    .instruction
                                    .as_deref()
                                    .is_some_and(|text| !text.trim().is_empty()),
                                "voice design requires an instruction"
                            );
                        }
                        SpeechMode::Standard | SpeechMode::Streaming => {}
                    }
                }
            },
            Self::Decision {
                state,
                questions,
                images,
                ..
            } => {
                ensure!(state.is_object(), "decision state must be an object");
                ensure!(
                    (1..=MAX_QUESTIONS).contains(&questions.len())
                        && questions
                            .iter()
                            .all(|(id, definition)| !id.trim().is_empty()
                                && id.len() <= 128
                                && definition
                                    .as_object()
                                    .is_some_and(|fields| !fields.is_empty())),
                    "decision questions must be named, non-empty objects and bounded"
                );
                for (id, question) in questions {
                    if matches!(&profile.native_source,
                        NativeProfileSource::Decisions { model, .. }
                        if profile.runtime.as_str() == "llamacpp" && model == "cloudflare-clef-flash")
                    {
                        ensure!(
                            question
                                .get("instructions")
                                .is_some_and(|value| !value.is_null()),
                            "Clef question {id} requires instructions"
                        );
                    }
                    if question.get("type").and_then(Value::as_str) == Some("choice") {
                        let valid_criteria = question
                            .get("criteria")
                            .and_then(Value::as_object)
                            .is_some_and(|criteria| {
                                !criteria.is_empty()
                                    && criteria.keys().all(|key| !key.trim().is_empty())
                            });
                        let valid_options = question
                            .get("options")
                            .and_then(Value::as_array)
                            .is_some_and(|options| {
                                !options.is_empty()
                                    && options.iter().all(|option| {
                                        option.as_str().is_some_and(|text| !text.trim().is_empty())
                                    })
                            });
                        ensure!(
                            valid_criteria || valid_options,
                            "choice question {id} requires declared options"
                        );
                    }
                }
                validate_decision_images(images)?;
                if !images.is_empty() {
                    ensure!(
                        decision_profile_supports_images(profile),
                        "selected decision profile does not support image input"
                    );
                }
                let mut payload = serde_json::json!({
                    "state": state,
                    "questions": questions,
                });
                if profile.runtime.as_str() == "llamacpp" {
                    if let NativeProfileSource::Decisions { model, .. } = &profile.native_source {
                        payload["model"] = Value::String(model.clone());
                    }
                }
                if !images.is_empty() {
                    payload["images"] = serde_json::to_value(images)?;
                }
                ensure_json_budget(&payload)?;
            }
        }
        Ok(profile)
    }
}

/// Results remain lane-specific. A pending media job is not reported as a
/// completed artifact, and a chat stream is not collapsed into a media job.
#[derive(Clone)]
pub enum InferenceResult {
    Chat {
        content: String,
        /// Reasoning is distinct from assistant content, including when a
        /// bounded token budget produces only reasoning.
        reasoning: String,
        completion_tokens: Option<u64>,
    },
    Embeddings {
        vectors: Vec<Vec<f32>>,
        usage: EmbeddingUsage,
    },
    MediaPending {
        backend_job_id: String,
    },
    MediaComplete {
        artifacts: Vec<PathBuf>,
    },
    Decision {
        /// Native choice/score/probability objects keyed by question ID.
        answers: Map<String, Value>,
    },
}

impl InferenceResult {
    pub fn validate_for(&self, request: &InferenceRequest) -> Result<()> {
        let matching_lane = matches!(
            (request, self),
            (InferenceRequest::Chat { .. }, Self::Chat { .. })
                | (InferenceRequest::Embeddings { .. }, Self::Embeddings { .. })
                | (InferenceRequest::Media { .. }, Self::MediaPending { .. })
                | (InferenceRequest::Media { .. }, Self::MediaComplete { .. })
                | (InferenceRequest::Decision { .. }, Self::Decision { .. })
        );
        ensure!(matching_lane, "result lane does not match request lane");
        match self {
            Self::MediaPending { backend_job_id } => {
                ensure!(!backend_job_id.trim().is_empty(), "backend job ID is empty");
            }
            Self::Decision { answers } => {
                let InferenceRequest::Decision { questions, .. } = request else {
                    bail!("decision result requires a decision request");
                };
                ensure!(
                    answers.len() == questions.len(),
                    "decision answer count mismatch"
                );
                ensure!(
                    serde_json::to_vec(answers)?.len() <= MAX_REQUEST_BYTES,
                    "decision result JSON exceeds 1 MiB"
                );
                for (id, question) in questions {
                    let answer = answers
                        .get(id)
                        .ok_or_else(|| anyhow::anyhow!("missing decision answer for {id}"))?;
                    if let Some(confidence) = answer.get("confidence") {
                        ensure!(
                            confidence.as_f64().is_some_and(
                                |value| value.is_finite() && (0.0..=1.0).contains(&value)
                            ),
                            "decision confidence must be in [0, 1]"
                        );
                    }
                    if question.get("type").and_then(Value::as_str) == Some("choice") {
                        let choice = answer
                            .get("choice")
                            .and_then(Value::as_str)
                            .or_else(|| answer.as_str())
                            .ok_or_else(|| {
                                anyhow::anyhow!("decision answer for {id} has no choice")
                            })?;
                        let declared = question
                            .get("criteria")
                            .and_then(Value::as_object)
                            .is_some_and(|criteria| criteria.contains_key(choice))
                            || question
                                .get("options")
                                .and_then(Value::as_array)
                                .is_some_and(|options| {
                                    options.iter().any(|option| option.as_str() == Some(choice))
                                });
                        ensure!(
                            declared,
                            "decision choice for {id} is not a declared option"
                        );
                    }
                }
            }
            Self::Chat {
                content, reasoning, ..
            } => {
                ensure!(
                    !content.is_empty() || !reasoning.is_empty(),
                    "chat result has no output"
                );
                ensure_text_budget([content.as_str(), reasoning.as_str()])?;
            }
            Self::Embeddings { vectors, usage } => {
                let InferenceRequest::Embeddings { input, .. } = request else {
                    bail!("embedding result requires an embedding request");
                };
                ensure!(
                    vectors.len() == input.len()
                        && (1..=MAX_EMBEDDING_INPUTS).contains(&vectors.len()),
                    "embedding vector count mismatch"
                );
                serialize_embedding_payload(vectors, usage)?;
            }
            Self::MediaComplete { .. } => {}
        }
        Ok(())
    }
}

/// Native usage for `/v1/embeddings`. llama.cpp reports total and prompt token
/// counts as the same quantity for this endpoint; zero is valid and is not
/// replaced with an estimate.
#[derive(Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}

impl std::fmt::Debug for EmbeddingUsage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EmbeddingUsage")
            .field("prompt_tokens", &self.prompt_tokens)
            .field("total_tokens", &self.total_tokens)
            .finish()
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmbeddingPayload {
    pub vectors: Vec<Vec<f32>>,
    pub usage: EmbeddingUsage,
}

#[derive(serde::Serialize)]
struct EmbeddingPayloadRef<'a> {
    vectors: &'a [Vec<f32>],
    usage: &'a EmbeddingUsage,
}

pub(crate) fn serialize_embedding_payload(
    vectors: &[Vec<f32>],
    usage: &EmbeddingUsage,
) -> Result<Vec<u8>> {
    ensure!(
        (1..=MAX_EMBEDDING_INPUTS).contains(&vectors.len()),
        "embedding vector count is out of range"
    );
    let dimensions = vectors.first().map(Vec::len).unwrap_or_default();
    ensure!(
        (1..=MAX_EMBEDDING_DIMENSIONS).contains(&dimensions),
        "embedding dimensions are out of range"
    );
    ensure!(
        vectors.iter().all(|vector| {
            vector.len() == dimensions && vector.iter().all(|value| value.is_finite())
        }),
        "embedding vectors must have consistent finite dimensions"
    );
    ensure!(
        usage.total_tokens == usage.prompt_tokens,
        "embedding usage token totals are inconsistent"
    );
    let mut writer = LimitedEmbeddingWriter::default();
    if serde_json::to_writer(&mut writer, &EmbeddingPayloadRef { vectors, usage }).is_err() {
        if writer.exceeded_limit {
            bail!("embedding result JSON exceeds 4 MiB");
        }
        bail!("embedding result serialization failed");
    }
    Ok(writer.bytes)
}

#[derive(Default)]
struct LimitedEmbeddingWriter {
    bytes: Vec<u8>,
    exceeded_limit: bool,
}

impl Write for LimitedEmbeddingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_EMBEDDING_JSON_BYTES.saturating_sub(self.bytes.len()) {
            self.exceeded_limit = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "bounded embedding JSON exceeded",
            ));
        }
        let required = self.bytes.len() + bytes.len();
        if self.bytes.capacity() < required {
            let target_capacity = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .max(4096)
                .max(required)
                .min(MAX_EMBEDDING_JSON_BYTES);
            self.bytes
                .try_reserve_exact(target_capacity.saturating_sub(self.bytes.len()))
                .map_err(io::Error::other)?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn catalog() -> Catalog {
        Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap()
    }

    fn embedding_catalog() -> (tempfile::TempDir, Catalog) {
        let language_root = tempfile::tempdir().unwrap();
        fs::write(
            language_root.path().join("llama-swap.yaml"),
            "models:\n  embedding-fixture:\n    cmd: |\n      llama-server --embedding\n  chat-fixture:\n    cmd: |\n      llama-server -m chat.gguf\n",
        )
        .unwrap();
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let catalog = Catalog::load_from_roots(language_root.path(), repo, repo).unwrap();
        (language_root, catalog)
    }

    fn key(lane: Lane, id: &str) -> ProfileKey {
        ProfileKey::new(lane, id).unwrap()
    }

    #[test]
    fn validates_each_native_lane_without_starting_a_runtime() {
        let catalog = catalog();
        let chat = InferenceRequest::Chat {
            profile: key(Lane::Language, "gemma-4-26b-qat-mtp"),
            turns: vec![ChatTurn {
                role: ChatRole::User,
                content: "hello".into(),
            }],
            max_tokens: 8,
        };
        let speech = InferenceRequest::Media {
            profile: key(Lane::Media, "breeze-tts-2"),
            input: MediaInput::Speech(SpeechInput {
                text: "hello".into(),
                mode: SpeechMode::Streaming,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            }),
        };
        let decision = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: serde_json::json!({"position": 1}),
            questions: serde_json::json!({"move": {"type":"choice", "criteria":{"yes":"Move", "no":"Stay"}}}).as_object().unwrap().clone(),
            images: Vec::new(),
        };
        assert_eq!(chat.validate(&catalog).unwrap().key, *chat.profile());
        assert_eq!(speech.task(), "tts");
        assert_eq!(speech.validate(&catalog).unwrap().key, *speech.profile());
        assert_eq!(
            decision.validate(&catalog).unwrap().key,
            *decision.profile()
        );
    }

    #[test]
    fn rejects_cross_lane_and_undeclared_media_tasks() {
        let catalog = catalog();
        let wrong_lane = InferenceRequest::Chat {
            profile: key(Lane::Media, "breeze-tts-2"),
            turns: vec![ChatTurn {
                role: ChatRole::User,
                content: "hello".into(),
            }],
            max_tokens: 8,
        };
        assert!(wrong_lane.validate(&catalog).is_err());
        let unsupported = InferenceRequest::Media {
            profile: key(Lane::Media, "heartmula-music"),
            input: MediaInput::Speech(SpeechInput {
                text: "hello".into(),
                mode: SpeechMode::VoiceClone,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            }),
        };
        assert!(unsupported.validate(&catalog).is_err());
    }

    #[test]
    fn voice_cloning_requires_a_nonempty_reference_path() {
        let catalog = catalog();
        let mut speech = InferenceRequest::Media {
            profile: key(Lane::Media, "breeze-tts-2"),
            input: MediaInput::Speech(SpeechInput {
                text: "hello".into(),
                mode: SpeechMode::VoiceClone,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            }),
        };
        assert!(speech.validate(&catalog).is_err());
        if let InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } = &mut speech
        {
            input.voice_reference = Some(PathBuf::new());
        }
        assert!(speech.validate(&catalog).is_err());
        if let InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } = &mut speech
        {
            input.voice_reference = Some(PathBuf::from("reference.wav"));
        }
        assert!(speech.validate(&catalog).is_ok());
    }

    #[test]
    fn decision_result_must_match_questions_and_confidence_bounds() {
        let request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: serde_json::json!({}),
            questions: serde_json::json!({"move": {"type":"choice", "criteria":{"yes":"Move", "no":"Stay"}}}).as_object().unwrap().clone(),
            images: Vec::new(),
        };
        let valid = InferenceResult::Decision {
            answers: serde_json::json!({"move":{"choice":"yes","confidence":0.8,"probabilities":{"yes":0.8,"no":0.2}}}).as_object().unwrap().clone(),
        };
        assert!(valid.validate_for(&request).is_ok());
        let wrong_lane = InferenceResult::Chat {
            reasoning: String::new(),
            content: "yes".into(),
            completion_tokens: None,
        };
        assert!(wrong_lane.validate_for(&request).is_err());
        let invalid = InferenceResult::Decision {
            answers: serde_json::json!({"move":{"choice":"yes","confidence":1.1}})
                .as_object()
                .unwrap()
                .clone(),
        };
        assert!(invalid.validate_for(&request).is_err());
        for answers in [
            serde_json::json!({"other":{"choice":"yes"}}),
            serde_json::json!({"move":{"choice":"undeclared"}}),
        ] {
            assert!(InferenceResult::Decision {
                answers: answers.as_object().unwrap().clone()
            }
            .validate_for(&request)
            .is_err());
        }
    }

    #[test]
    fn chat_results_preserve_reasoning_only_output_with_combined_size_bound() {
        let request = InferenceRequest::Chat {
            profile: key(Lane::Language, "gemma-4-26b-qat-mtp"),
            turns: vec![ChatTurn {
                role: ChatRole::User,
                content: "hi".into(),
            }],
            max_tokens: 64,
        };
        assert!(InferenceResult::Chat {
            content: String::new(),
            reasoning: "thinking".into(),
            completion_tokens: Some(8)
        }
        .validate_for(&request)
        .is_ok());
        assert!(InferenceResult::Chat {
            content: String::new(),
            reasoning: String::new(),
            completion_tokens: None
        }
        .validate_for(&request)
        .is_err());
        assert!(InferenceResult::Chat {
            content: "x".repeat(MAX_REQUEST_BYTES),
            reasoning: "y".into(),
            completion_tokens: None
        }
        .validate_for(&request)
        .is_err());
    }

    #[test]
    fn decision_contract_preserves_native_definitions_and_score_results() {
        let catalog = catalog();
        let questions = serde_json::json!({"urgency":{"type":"score","instructions":"Assess urgency","range":[0,100]}});
        let request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: serde_json::json!({"alert":"test"}),
            questions: questions.as_object().unwrap().clone(),
            images: Vec::new(),
        };
        assert!(request.validate(&catalog).is_ok());
        let result = InferenceResult::Decision {
            answers: serde_json::json!({"urgency":{"score":75,"confidence":0.9}})
                .as_object()
                .unwrap()
                .clone(),
        };
        assert!(result.validate_for(&request).is_ok());
        for question in [
            serde_json::json!("flattened prompt"),
            serde_json::json!({"type":"choice"}),
        ] {
            let invalid = InferenceRequest::Decision {
                profile: key(Lane::Decisions, "nagi-smol"),
                state: serde_json::json!({}),
                questions: Map::from_iter([("question".into(), question)]),
                images: Vec::new(),
            };
            assert!(invalid.validate(&catalog).is_err());
        }
    }

    #[test]
    fn decision_images_are_bounded_base64_urls_and_profile_gated() {
        let catalog = catalog();
        let questions = serde_json::json!({
            "route": {"type":"choice", "criteria":{"yes":"Proceed", "no":"Stop"}}
        })
        .as_object()
        .unwrap()
        .clone();
        let image = "data:image/png;base64,iVBORw0KGgo=".to_owned();
        let request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "vllm-d3-flash"),
            state: serde_json::json!({"message":"inspect this"}),
            questions: questions.clone(),
            images: vec![image.clone()],
        };
        assert!(request.validate(&catalog).is_ok());

        let unsupported = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: serde_json::json!({"message":"inspect this"}),
            questions: questions.clone(),
            images: vec![image.clone()],
        };
        assert!(unsupported.validate(&catalog).is_err());

        for invalid_image in [
            "https://example.test/image.png".to_owned(),
            "file:///tmp/image.png".to_owned(),
            "data:image/png;base64,".to_owned(),
            "data:image/png;base64,AAAA_=".to_owned(),
            "data:audio/wav;base64,UklGRg==".to_owned(),
            "data:image/png,AAAA".to_owned(),
        ] {
            let invalid = InferenceRequest::Decision {
                profile: key(Lane::Decisions, "vllm-d3-flash"),
                state: serde_json::json!({}),
                questions: questions.clone(),
                images: vec![invalid_image],
            };
            assert!(invalid.validate(&catalog).is_err());
        }

        let too_many = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "cloudflare-clef-flash"),
            state: serde_json::json!({}),
            questions: questions.clone(),
            images: vec![image; MAX_DECISION_IMAGES + 1],
        };
        assert!(too_many.validate(&catalog).is_err());

        let oversized = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "vllm-d3-flash"),
            state: serde_json::json!({}),
            questions,
            images: vec![format!(
                "data:image/png;base64,{}",
                "A".repeat(MAX_REQUEST_BYTES)
            )],
        };
        assert!(oversized.validate(&catalog).is_err());
    }

    #[test]
    fn clef_requires_question_instructions_before_dispatch() {
        let catalog = Catalog::load(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
        let mut request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "cloudflare-clef-flash"),
            state: serde_json::json!({}),
            questions: serde_json::json!({"color": {
                "type":"choice", "criteria":{"green":"Green", "red":"Red"}
            }})
            .as_object()
            .unwrap()
            .clone(),
            images: Vec::new(),
        };
        assert!(request.validate(&catalog).is_err());
        let InferenceRequest::Decision { questions, .. } = &mut request else {
            unreachable!()
        };
        questions.get_mut("color").unwrap()["instructions"] =
            Value::String("Which color is shown?".into());
        assert!(request.validate(&catalog).is_ok());
    }

    #[test]
    fn request_payloads_have_a_bounded_memory_contract() {
        let catalog = catalog();
        let oversized = InferenceRequest::Chat {
            profile: key(Lane::Language, "gemma-4-26b-qat-mtp"),
            turns: vec![ChatTurn {
                role: ChatRole::User,
                content: "x".repeat(MAX_REQUEST_BYTES + 1),
            }],
            max_tokens: 8,
        };
        assert!(oversized.validate(&catalog).is_err());

        let oversized = InferenceRequest::Media {
            profile: key(Lane::Media, "qwen-image-2.1"),
            input: MediaInput::Workflow {
                task: Task::new("image").unwrap(),
                graph: serde_json::json!({"prompt": "x".repeat(MAX_REQUEST_BYTES)}),
            },
        };
        assert!(oversized.validate(&catalog).is_err());
    }

    #[test]
    fn embedding_requests_are_explicit_language_tasks_and_bounded_batches() {
        let (_root, catalog) = embedding_catalog();
        let request = InferenceRequest::Embeddings {
            profile: key(Lane::Language, "embedding-fixture"),
            input: vec!["first text".into(), "second text".into()],
        };
        assert_eq!(request.lane(), Lane::Language);
        assert_eq!(request.task(), "embeddings");
        assert!(request.validate(&catalog).is_ok());

        let mislabeled_profile = InferenceRequest::Embeddings {
            profile: key(Lane::Language, "chat-fixture"),
            input: vec!["text".into()],
        };
        assert!(mislabeled_profile.validate(&catalog).is_err());

        for input in [
            Vec::new(),
            vec![String::new()],
            vec![" \t".to_owned()],
            vec!["bad\0text".to_owned()],
            vec!["x".repeat(MAX_REQUEST_BYTES + 1)],
            vec!["x".to_owned(); MAX_EMBEDDING_INPUTS + 1],
        ] {
            assert!(InferenceRequest::Embeddings {
                profile: key(Lane::Language, "embedding-fixture"),
                input,
            }
            .validate(&catalog)
            .is_err());
        }
    }

    #[test]
    fn embedding_results_require_matching_finite_vectors_consistent_usage_and_size() {
        let (_root, catalog) = embedding_catalog();
        let request = InferenceRequest::Embeddings {
            profile: key(Lane::Language, "embedding-fixture"),
            input: vec!["first".into(), "second".into()],
        };
        assert!(request.validate(&catalog).is_ok());
        let valid = InferenceResult::Embeddings {
            vectors: vec![vec![0.0, 0.25, -1.5], vec![1.0, 0.5, 2.0]],
            usage: EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };
        assert!(valid.validate_for(&request).is_ok());

        for invalid in [
            InferenceResult::Embeddings {
                vectors: vec![vec![1.0, 2.0]],
                usage: EmbeddingUsage {
                    prompt_tokens: 1,
                    total_tokens: 1,
                },
            },
            InferenceResult::Embeddings {
                vectors: vec![vec![], vec![]],
                usage: EmbeddingUsage {
                    prompt_tokens: 1,
                    total_tokens: 1,
                },
            },
            InferenceResult::Embeddings {
                vectors: vec![vec![1.0], vec![1.0, 2.0]],
                usage: EmbeddingUsage {
                    prompt_tokens: 1,
                    total_tokens: 1,
                },
            },
            InferenceResult::Embeddings {
                vectors: vec![vec![f32::NAN], vec![f32::INFINITY]],
                usage: EmbeddingUsage {
                    prompt_tokens: 1,
                    total_tokens: 1,
                },
            },
            InferenceResult::Embeddings {
                vectors: vec![vec![1.0; MAX_EMBEDDING_DIMENSIONS + 1]; 2],
                usage: EmbeddingUsage {
                    prompt_tokens: 1,
                    total_tokens: 1,
                },
            },
            InferenceResult::Embeddings {
                vectors: vec![vec![1.0], vec![1.0]],
                usage: EmbeddingUsage {
                    prompt_tokens: 1,
                    total_tokens: 2,
                },
            },
        ] {
            assert!(invalid.validate_for(&request).is_err());
        }

        let oversized = InferenceResult::Embeddings {
            vectors: vec![vec![-f32::MAX; 4096]; MAX_EMBEDDING_INPUTS],
            usage: EmbeddingUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
        };
        assert!(oversized
            .validate_for(&InferenceRequest::Embeddings {
                profile: key(Lane::Language, "embedding-fixture"),
                input: vec!["x".into(); MAX_EMBEDDING_INPUTS],
            })
            .is_err());
    }
}
