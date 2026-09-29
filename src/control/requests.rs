//! Modality-specific in-memory request and result contracts.
//!
//! These types describe work before a runtime adapter or scheduler accepts it.
//! Validation checks manifest ownership and basic input invariants only: it is
//! not an installation check, resource lease, admission barrier, or permission
//! to launch an engine. Native payloads and prompts must not be written to job
//! history; the persistent record stores bounded metadata instead.

use std::path::PathBuf;

use anyhow::{bail, ensure, Result};
use serde_json::Value;

use super::catalog::{Catalog, CatalogProfile, Lane, ProfileKey, Task};

const MAX_TURNS: usize = 128;
const MAX_QUESTIONS: usize = 128;
const MAX_SCRIPT_ARGS: usize = 128;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;

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
    Media {
        profile: ProfileKey,
        input: MediaInput,
    },
    Decision {
        profile: ProfileKey,
        state: Value,
        questions: Vec<String>,
    },
}

impl InferenceRequest {
    pub fn profile(&self) -> &ProfileKey {
        match self {
            Self::Chat { profile, .. }
            | Self::Media { profile, .. }
            | Self::Decision { profile, .. } => profile,
        }
    }

    pub fn lane(&self) -> Lane {
        match self {
            Self::Chat { .. } => Lane::Language,
            Self::Media { .. } => Lane::Media,
            Self::Decision { .. } => Lane::Decisions,
        }
    }

    pub fn task(&self) -> &str {
        match self {
            Self::Chat { .. } => "chat",
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
                                input.voice_reference.is_some()
                                    && input
                                        .reference_text
                                        .as_deref()
                                        .is_some_and(|text| !text.trim().is_empty()),
                                "voice cloning requires reference audio and transcript"
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
                state, questions, ..
            } => {
                ensure!(state.is_object(), "decision state must be an object");
                ensure_json_budget(state)?;
                ensure!(
                    (1..=MAX_QUESTIONS).contains(&questions.len())
                        && questions.iter().all(|question| !question.trim().is_empty()),
                    "decision questions must be non-empty and bounded"
                );
                ensure_text_budget(questions.iter().map(String::as_str))?;
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
        completion_tokens: Option<u64>,
    },
    MediaPending {
        backend_job_id: String,
    },
    MediaComplete {
        artifacts: Vec<PathBuf>,
    },
    Decision {
        answers: Vec<DecisionAnswer>,
    },
}

#[derive(Clone)]
pub struct DecisionAnswer {
    pub question: String,
    pub answer: String,
    pub confidence: Option<f64>,
}

impl InferenceResult {
    pub fn validate_for(&self, request: &InferenceRequest) -> Result<()> {
        let matching_lane = matches!(
            (request, self),
            (InferenceRequest::Chat { .. }, Self::Chat { .. })
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
                for (answer, question) in answers.iter().zip(questions) {
                    ensure!(
                        answer.question == *question,
                        "decision question order mismatch"
                    );
                    ensure!(
                        answer
                            .confidence
                            .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value)),
                        "decision confidence must be in [0, 1]"
                    );
                }
            }
            Self::Chat { .. } | Self::MediaComplete { .. } => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Catalog {
        Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap()
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
            questions: vec!["move?".into()],
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
    fn voice_cloning_requires_reference_and_transcript() {
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
            input.voice_reference = Some(PathBuf::from("reference.wav"));
            input.reference_text = Some("reference words".into());
        }
        assert!(speech.validate(&catalog).is_ok());
    }

    #[test]
    fn decision_result_must_match_questions_and_confidence_bounds() {
        let request = InferenceRequest::Decision {
            profile: key(Lane::Decisions, "nagi-smol"),
            state: serde_json::json!({}),
            questions: vec!["move?".into()],
        };
        let valid = InferenceResult::Decision {
            answers: vec![DecisionAnswer {
                question: "move?".into(),
                answer: "yes".into(),
                confidence: Some(0.8),
            }],
        };
        assert!(valid.validate_for(&request).is_ok());
        let wrong_lane = InferenceResult::Chat {
            content: "yes".into(),
            completion_tokens: None,
        };
        assert!(wrong_lane.validate_for(&request).is_err());
        let invalid = InferenceResult::Decision {
            answers: vec![DecisionAnswer {
                question: "move?".into(),
                answer: "yes".into(),
                confidence: Some(1.1),
            }],
        };
        assert!(invalid.validate_for(&request).is_err());
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
}
