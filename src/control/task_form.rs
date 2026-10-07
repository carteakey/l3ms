//! Typed, in-memory construction of the existing lane-native Submit envelope.
//!
//! A draft names its catalog profile and declared task explicitly. Its fields
//! stay native to the selected lane; this module supplies no generic prompt,
//! model default, runtime start, artifact access, or persistence.

use std::fmt;
use std::io::{self, Write};

use serde::Serialize;
use serde_json::{Map, Value};

use super::catalog::{Catalog, ProfileKey, Task};
use super::ipc::{
    self, ClientRequest, RequestEnvelope, WireChatTurn, WireInferenceRequest, WireMediaInput,
    WireRequestMessage, WireSpeechMode, MAX_FRAME_BYTES,
};
use super::requests::SpeechMode;

const ENVELOPE_OVERHEAD_BUDGET: usize = 2048;
/// Maximum UTF-8 bytes accepted into one editable native field.
pub const MAX_TASK_FORM_FIELD_BYTES: usize = 512 * 1024;
/// Aggregate typed-form input leaves room for the native IPC envelope.
pub const MAX_TASK_FORM_TOTAL_BYTES: usize = 900 * 1024;

/// Lane-native keyboard fields. Script arguments are already-separated argv
/// items: they are never parsed as a shell command or split on whitespace.
pub enum TaskFields {
    Language {
        turns: Vec<WireChatTurn>,
        max_tokens: u32,
    },
    LanguageEmbeddings {
        input: Vec<String>,
    },
    MediaWorkflow {
        graph: Value,
    },
    MediaScript {
        args: Vec<String>,
    },
    Speech {
        text: String,
        mode: SpeechMode,
        instruction: Option<String>,
        voice_reference: Option<String>,
        reference_text: Option<String>,
    },
    Decisions {
        state: Value,
        questions: Map<String, Value>,
    },
}

/// An owned, immutable draft with every routing/control choice explicit.
/// Debug intentionally excludes caller IDs and every native field body.
pub struct TaskFormDraft {
    selected_profile: ProfileKey,
    declared_task: Task,
    request_id: String,
    timeout_ms: u64,
    fields: TaskFields,
}

impl TaskFormDraft {
    pub fn new(
        selected_profile: ProfileKey,
        declared_task: Task,
        request_id: String,
        timeout_ms: u64,
        fields: TaskFields,
    ) -> Self {
        Self {
            selected_profile,
            declared_task,
            request_id,
            timeout_ms,
            fields,
        }
    }
}

impl fmt::Debug for TaskFormDraft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TaskFormDraft(<redacted>)")
    }
}

/// Fixed, payload-free validation outcomes for task-form construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskFormError {
    InvalidRequest,
    CatalogUnavailable,
    ProfileTaskMismatch,
    InvalidNativeFields,
    FrameTooLarge,
}

impl fmt::Display for TaskFormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidRequest => "caller request ID, timeout, or envelope is invalid",
            Self::CatalogUnavailable => "the selected catalog could not be refreshed",
            Self::ProfileTaskMismatch => {
                "selected profile or declared task does not match the refreshed catalog"
            }
            Self::InvalidNativeFields => "native task fields are invalid for the selected profile",
            Self::FrameTooLarge => "native request exceeds the bounded IPC frame",
        })
    }
}

impl std::error::Error for TaskFormError {}

/// Convert one draft into a validated v1 native Submit envelope. The method
/// refreshes only the already-loaded manifest roots; it never probes or starts
/// a runtime. Payload size is counted without allocation before the bounded
/// envelope clone used by the canonical IPC decoder.
pub fn build_submit(
    catalog: &Catalog,
    draft: TaskFormDraft,
) -> Result<RequestEnvelope, TaskFormError> {
    preflight_draft(&draft)?;

    let TaskFormDraft {
        selected_profile,
        declared_task,
        request_id,
        timeout_ms,
        fields,
    } = draft;
    let profile_id = selected_profile.profile_id.as_str().to_owned();
    let declared_task_text = declared_task.as_str();
    let request = match fields {
        TaskFields::Language { turns, max_tokens } => WireInferenceRequest::Language {
            profile_id,
            turns,
            max_tokens,
        },
        TaskFields::LanguageEmbeddings { input } => {
            WireInferenceRequest::Embeddings { profile_id, input }
        }
        TaskFields::MediaWorkflow { graph } => WireInferenceRequest::Media {
            profile_id,
            input: WireMediaInput::Workflow {
                task: declared_task_text.to_owned(),
                graph,
            },
        },
        TaskFields::MediaScript { args } => WireInferenceRequest::Media {
            profile_id,
            input: WireMediaInput::Script {
                task: declared_task_text.to_owned(),
                args,
            },
        },
        TaskFields::Speech {
            text,
            mode,
            instruction,
            voice_reference,
            reference_text,
        } => WireInferenceRequest::Media {
            profile_id,
            input: WireMediaInput::Speech {
                text,
                mode: wire_speech_mode(mode),
                instruction,
                voice_reference,
                reference_text,
            },
        },
        TaskFields::Decisions { state, questions } => WireInferenceRequest::Decisions {
            profile_id,
            state,
            questions,
        },
    };
    let envelope = RequestEnvelope::v1(WireRequestMessage::Submit {
        request_id,
        timeout_ms,
        request,
    });

    // Count the complete encoded form without allocating or cloning its body.
    // This also accounts for JSON escaping beyond the individual-field pass.
    bounded_json_len(&envelope)?;
    ipc::validate_request_envelope(&envelope).map_err(|_| TaskFormError::InvalidRequest)?;

    let refreshed = catalog
        .refresh()
        .map_err(|_| TaskFormError::CatalogUnavailable)?;
    let original_profile = catalog
        .inspect(&selected_profile)
        .ok_or(TaskFormError::ProfileTaskMismatch)?;
    let refreshed_profile = refreshed
        .inspect(&selected_profile)
        .ok_or(TaskFormError::ProfileTaskMismatch)?;
    if refreshed_profile != original_profile
        || refreshed_profile.disabled
        || !refreshed_profile
            .tasks
            .iter()
            .any(|task| task.as_str() == declared_task.as_str())
    {
        return Err(TaskFormError::ProfileTaskMismatch);
    }
    let decoded = envelope
        .clone()
        .into_client_request()
        .map_err(|_| TaskFormError::InvalidNativeFields)?;
    let ClientRequest::Submit { request, .. } = decoded else {
        return Err(TaskFormError::InvalidRequest);
    };
    if request.profile() != &selected_profile || request.task() != declared_task.as_str() {
        return Err(TaskFormError::ProfileTaskMismatch);
    }
    request
        .validate(&refreshed)
        .map_err(|_| TaskFormError::InvalidNativeFields)?;
    Ok(envelope)
}

fn wire_speech_mode(mode: SpeechMode) -> WireSpeechMode {
    match mode {
        SpeechMode::Standard => WireSpeechMode::Standard,
        SpeechMode::VoiceDesign => WireSpeechMode::VoiceDesign,
        SpeechMode::VoiceClone => WireSpeechMode::VoiceClone,
        SpeechMode::Streaming => WireSpeechMode::Streaming,
    }
}

fn preflight_draft(draft: &TaskFormDraft) -> Result<(), TaskFormError> {
    let mut total = ENVELOPE_OVERHEAD_BUDGET;
    add_json_size(&mut total, &draft.request_id)?;
    add_json_size(&mut total, &draft.selected_profile.profile_id.as_str())?;
    add_json_size(&mut total, &draft.declared_task.as_str())?;
    add_json_size(&mut total, &draft.timeout_ms)?;

    match &draft.fields {
        TaskFields::Language { turns, max_tokens } => {
            add_json_size(&mut total, turns)?;
            add_json_size(&mut total, max_tokens)?;
        }
        TaskFields::LanguageEmbeddings { input } => add_json_size(&mut total, input)?,
        TaskFields::MediaWorkflow { graph } => add_json_size(&mut total, graph)?,
        TaskFields::MediaScript { args } => add_json_size(&mut total, args)?,
        TaskFields::Speech {
            text,
            mode,
            instruction,
            voice_reference,
            reference_text,
        } => {
            add_json_size(&mut total, text)?;
            add_json_size(&mut total, &speech_mode_name(*mode))?;
            if let Some(instruction) = instruction {
                add_json_size(&mut total, instruction)?;
            }
            if let Some(voice_reference) = voice_reference {
                add_json_size(&mut total, voice_reference)?;
            }
            if let Some(reference_text) = reference_text {
                add_json_size(&mut total, reference_text)?;
            }
        }
        TaskFields::Decisions { state, questions } => {
            add_json_size(&mut total, state)?;
            add_json_size(&mut total, questions)?;
        }
    }
    if total > MAX_FRAME_BYTES {
        return Err(TaskFormError::FrameTooLarge);
    }
    Ok(())
}

fn speech_mode_name(mode: SpeechMode) -> &'static str {
    match mode {
        SpeechMode::Standard => "standard",
        SpeechMode::VoiceDesign => "voice_design",
        SpeechMode::VoiceClone => "voice_clone",
        SpeechMode::Streaming => "streaming",
    }
}

fn add_json_size<T: Serialize>(total: &mut usize, value: &T) -> Result<(), TaskFormError> {
    let size = bounded_json_len(value)?;
    *total = total.saturating_add(size);
    if *total > MAX_FRAME_BYTES {
        return Err(TaskFormError::FrameTooLarge);
    }
    Ok(())
}

fn bounded_json_len<T: Serialize>(value: &T) -> Result<usize, TaskFormError> {
    let mut counter = BoundedCounter {
        bytes: 0,
        exceeded: false,
    };
    if serde_json::to_writer(&mut counter, value).is_err() {
        return Err(if counter.exceeded {
            TaskFormError::FrameTooLarge
        } else {
            TaskFormError::InvalidRequest
        });
    }
    Ok(counter.bytes)
}

struct BoundedCounter {
    bytes: usize,
    exceeded: bool,
}

impl Write for BoundedCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(total) = self.bytes.checked_add(bytes.len()) else {
            self.exceeded = true;
            return Err(io::Error::other("bounded JSON size counter"));
        };
        if total > MAX_FRAME_BYTES {
            self.exceeded = true;
            return Err(io::Error::other("bounded JSON size counter"));
        }
        self.bytes = total;
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

    use serde_json::json;
    use tempfile::TempDir;

    use super::super::requests::{ChatRole, SpeechInput};

    fn catalog() -> Catalog {
        Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap()
    }

    fn profile(lane: super::super::catalog::Lane, id: &str) -> ProfileKey {
        ProfileKey::new(lane, id).unwrap()
    }

    fn task(name: &str) -> Task {
        Task::new(name).unwrap()
    }

    fn draft(profile: ProfileKey, task_name: &str, fields: TaskFields) -> TaskFormDraft {
        TaskFormDraft::new(
            profile,
            task(task_name),
            "caller:form".into(),
            30_000,
            fields,
        )
    }

    fn error(result: Result<RequestEnvelope, TaskFormError>) -> TaskFormError {
        match result {
            Err(error) => error,
            Ok(_) => panic!("expected a fixed task-form error"),
        }
    }

    fn language_fields() -> TaskFields {
        TaskFields::Language {
            turns: vec![
                WireChatTurn {
                    role: super::super::ipc::WireChatRole::System,
                    content: "private system instruction".into(),
                },
                WireChatTurn {
                    role: super::super::ipc::WireChatRole::User,
                    content: "private user input".into(),
                },
                WireChatTurn {
                    role: super::super::ipc::WireChatRole::Assistant,
                    content: "earlier assistant output".into(),
                },
            ],
            max_tokens: 96,
        }
    }

    fn speech_fields(mode: SpeechMode) -> TaskFields {
        let input = match mode {
            SpeechMode::Standard => SpeechInput {
                text: "hello there".into(),
                mode,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            },
            SpeechMode::VoiceDesign => SpeechInput {
                text: "hello there".into(),
                mode,
                instruction: Some("warm alto voice".into()),
                voice_reference: None,
                reference_text: None,
            },
            SpeechMode::VoiceClone => SpeechInput {
                text: "hello there".into(),
                mode,
                instruction: None,
                voice_reference: Some("/private/reference.wav".into()),
                reference_text: Some("spoken words".into()),
            },
            SpeechMode::Streaming => SpeechInput {
                text: "hello there".into(),
                mode,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            },
        };
        TaskFields::Speech {
            text: input.text,
            mode: input.mode,
            instruction: input.instruction,
            voice_reference: input
                .voice_reference
                .map(|path| path.to_string_lossy().into_owned()),
            reference_text: input.reference_text,
        }
    }

    #[test]
    fn language_form_preserves_roles_and_builds_explicit_submit() {
        let selected = profile(super::super::catalog::Lane::Language, "gemma-4-26b-qat-mtp");
        let envelope = build_submit(
            &catalog(),
            draft(selected.clone(), "chat", language_fields()),
        )
        .unwrap();
        assert_eq!(envelope.version, ipc::PROTOCOL_VERSION);
        let ClientRequest::Submit {
            request_id,
            timeout,
            request,
        } = envelope.clone().into_client_request().unwrap()
        else {
            panic!("task form builds Submit")
        };
        assert_eq!(request_id, "caller:form");
        assert_eq!(timeout.as_millis(), 30_000);
        assert_eq!(request.profile(), &selected);
        let super::super::requests::InferenceRequest::Chat {
            turns, max_tokens, ..
        } = request
        else {
            panic!("language fields stay language-native")
        };
        assert_eq!(max_tokens, 96);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].role, ChatRole::System);
        assert_eq!(turns[1].role, ChatRole::User);
        assert_eq!(turns[2].role, ChatRole::Assistant);
        assert_eq!(turns[0].content, "private system instruction");
        assert_eq!(turns[1].content, "private user input");
        assert_eq!(turns[2].content, "earlier assistant output");
    }

    #[test]
    fn media_script_keeps_exact_argv_items_without_shell_splitting() {
        let fields = TaskFields::MediaScript {
            args: vec![
                "--prompt".into(),
                "two words stay together".into(),
                "x; y".into(),
            ],
        };
        let envelope = build_submit(
            &catalog(),
            draft(
                profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                "tts",
                fields,
            ),
        )
        .unwrap();
        let ClientRequest::Submit { request, .. } = envelope.into_client_request().unwrap() else {
            panic!("media script is a Submit")
        };
        let super::super::requests::InferenceRequest::Media {
            input: super::super::requests::MediaInput::Script { args, task },
            ..
        } = request
        else {
            panic!("script args remain separate argv items")
        };
        assert_eq!(task.as_str(), "tts");
        assert_eq!(args, ["--prompt", "two words stay together", "x; y"]);
    }

    #[test]
    fn workflow_and_decisions_keep_their_native_json_fields() {
        let workflow = build_submit(
            &catalog(),
            draft(
                profile(super::super::catalog::Lane::Media, "ltx-2.5"),
                "video",
                TaskFields::MediaWorkflow {
                    graph: json!({"nodes": [{"class_type": "KSampler"}]}),
                },
            ),
        )
        .unwrap();
        assert!(matches!(
            workflow.message,
            WireRequestMessage::Submit {
                request: WireInferenceRequest::Media {
                    input: WireMediaInput::Workflow { .. },
                    ..
                },
                ..
            }
        ));

        let mut questions = Map::new();
        questions.insert(
            "route".into(),
            json!({"type": "choice", "criteria": {"yes": "Proceed", "no": "Stop"}}),
        );
        let decisions = build_submit(
            &catalog(),
            draft(
                profile(super::super::catalog::Lane::Decisions, "nagi-smol"),
                "decision",
                TaskFields::Decisions {
                    state: json!({"ready": true}),
                    questions,
                },
            ),
        )
        .unwrap();
        assert!(matches!(
            decisions.message,
            WireRequestMessage::Submit {
                request: WireInferenceRequest::Decisions { .. },
                ..
            }
        ));
    }

    #[test]
    fn speech_modes_preserve_task_aliases_and_require_the_typed_mode_fields() {
        let catalog = catalog();
        for (mode, task_name) in [
            (SpeechMode::Standard, "tts"),
            (SpeechMode::VoiceDesign, "design"),
            (SpeechMode::VoiceClone, "clon"),
            (SpeechMode::Streaming, "tts"),
        ] {
            let envelope = build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                    task_name,
                    speech_fields(mode),
                ),
            )
            .unwrap();
            let ClientRequest::Submit { request, .. } = envelope.into_client_request().unwrap()
            else {
                panic!("speech fields build a Submit")
            };
            assert_eq!(request.task(), task_name);
            let super::super::requests::InferenceRequest::Media {
                input: super::super::requests::MediaInput::Speech(input),
                ..
            } = request
            else {
                panic!("speech remains a typed media operation")
            };
            assert_eq!(input.mode, mode);
        }

        let missing_clone_reference = TaskFields::Speech {
            text: "spoken content".into(),
            mode: SpeechMode::VoiceClone,
            instruction: None,
            voice_reference: None,
            reference_text: None,
        };
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                    "clon",
                    missing_clone_reference,
                ),
            )),
            TaskFormError::InvalidNativeFields
        );

        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                    "design",
                    TaskFields::Speech {
                        text: "spoken content".into(),
                        mode: SpeechMode::VoiceDesign,
                        instruction: None,
                        voice_reference: None,
                        reference_text: None,
                    },
                ),
            )),
            TaskFormError::InvalidNativeFields
        );
    }

    #[test]
    fn rejects_lane_profile_task_timeout_and_identifier_mismatch() {
        let catalog = catalog();
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                    "chat",
                    language_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Language, "gemma-4-26b-qat-mtp"),
                    "not-declared",
                    language_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Language, "unknown-profile"),
                    "chat",
                    language_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );

        let invalid_id = TaskFormDraft::new(
            profile(super::super::catalog::Lane::Language, "gemma-4-26b-qat-mtp"),
            task("chat"),
            "sk-example-fixture".into(),
            30_000,
            language_fields(),
        );
        assert_eq!(
            error(build_submit(&catalog, invalid_id)),
            TaskFormError::InvalidRequest
        );

        let invalid_timeout = TaskFormDraft::new(
            profile(super::super::catalog::Lane::Language, "gemma-4-26b-qat-mtp"),
            task("chat"),
            "caller:bad-timeout".into(),
            0,
            language_fields(),
        );
        assert_eq!(
            error(build_submit(&catalog, invalid_timeout)),
            TaskFormError::InvalidRequest
        );
    }

    #[test]
    fn bounds_payload_before_clone_and_redacts_draft_debug() {
        let secret = "private-native-prompt";
        let draft = TaskFormDraft::new(
            profile(super::super::catalog::Lane::Language, "gemma-4-26b-qat-mtp"),
            task("chat"),
            "caller:private-id".into(),
            30_000,
            TaskFields::Language {
                turns: vec![WireChatTurn {
                    role: super::super::ipc::WireChatRole::User,
                    content: secret.into(),
                }],
                max_tokens: 64,
            },
        );
        let debug = format!("{draft:?}");
        assert!(!debug.contains(secret));
        assert!(!debug.contains("caller:private-id"));

        let oversized = TaskFormDraft::new(
            profile(super::super::catalog::Lane::Media, "ltx-2.5"),
            task("video"),
            "caller:oversize".into(),
            30_000,
            TaskFields::MediaWorkflow {
                graph: json!({"prompt": "x".repeat(MAX_FRAME_BYTES)}),
            },
        );
        assert_eq!(
            error(build_submit(&catalog(), oversized)),
            TaskFormError::FrameTooLarge
        );
    }

    #[test]
    fn builder_rejects_a_stale_catalog_task_or_profile_route() {
        let temporary = TempDir::new().unwrap();
        let root = temporary.path();
        fs::create_dir_all(root.join("toolkits/l2m2")).unwrap();
        fs::create_dir_all(root.join("toolkits/l3ds")).unwrap();
        fs::write(
            root.join("llama-swap.yaml"),
            "models:\n  fixture-chat:\n    cmd: server\n",
        )
        .unwrap();
        fs::write(
            root.join("toolkits/l2m2/media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"fixture-media","name":"Fixture","runtime":"fixture","tasks":["image"],"script":"run.sh"}]}"#,
        )
        .unwrap();
        fs::write(
            root.join("toolkits/l3ds/decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[{"id":"fixture-decision","name":"Fixture","engine":"nagi","type":"local","model":"fixture","entrypoint":"run.py"}]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root).unwrap();
        fs::write(
            root.join("toolkits/l2m2/media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"fixture-media","name":"Fixture","runtime":"fixture","tasks":["audio"],"script":"run.sh"}]}"#,
        )
        .unwrap();
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Media, "fixture-media"),
                    "image",
                    TaskFields::MediaWorkflow {
                        graph: json!({"nodes": []}),
                    },
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );

        // The tag still matches, but the profile's native route changed after
        // this form was opened. Requiring an exact refreshed profile snapshot
        // prevents a silent reroute under the same public key.
        fs::write(
            root.join("toolkits/l2m2/media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"fixture-media","name":"Fixture","runtime":"fixture-v2","tasks":["image"],"script":"run-v2.sh"}]}"#,
        )
        .unwrap();
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Media, "fixture-media"),
                    "image",
                    TaskFields::MediaWorkflow {
                        graph: json!({"nodes": []}),
                    },
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
    }
}
