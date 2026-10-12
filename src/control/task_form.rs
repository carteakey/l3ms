//! Typed, in-memory construction for lane-native requests and Submit envelopes.
//!
//! A draft names its catalog profile and declared task explicitly. Its fields
//! stay native to the selected lane; this module supplies no generic prompt,
//! model default, runtime start, artifact access, or persistence.

use std::fmt;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use serde_json::{Map, Value};

use super::catalog::{Catalog, Lane, ProfileKey, Task};
use super::ipc::{
    self, ClientRequest, RequestEnvelope, WireInferenceRequest, WireMediaInput, WireRequestMessage,
    WireSpeechMode, MAX_FRAME_BYTES,
};
use super::native_commands::{is_supported_cpu_breeze_speech, CPU_BREEZE_PROFILE_ID};
use super::requests::{InferenceRequest, SpeechInput, SpeechMode};

const ENVELOPE_OVERHEAD_BUDGET: usize = 2048;
const MAX_DIRECT_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
/// Maximum UTF-8 bytes accepted into one editable native field.
pub const MAX_TASK_FORM_FIELD_BYTES: usize = 512 * 1024;
/// Aggregate typed-form input leaves room for the native IPC envelope.
pub const MAX_TASK_FORM_TOTAL_BYTES: usize = 900 * 1024;

/// Lane-native keyboard fields. Script arguments are already-separated argv
/// items: they are never parsed as a shell command or split on whitespace.
pub enum TaskFields {
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
            Self::InvalidRequest => "task form request or timeout is invalid",
            Self::CatalogUnavailable => "the selected catalog could not be refreshed",
            Self::ProfileTaskMismatch => {
                "selected profile or declared task does not match the refreshed catalog"
            }
            Self::InvalidNativeFields => "native task fields are invalid for the selected profile",
            Self::FrameTooLarge => "native request exceeds the bounded input limit",
        })
    }
}

impl std::error::Error for TaskFormError {}

/// Build a typed request for the supported direct CPU execution slice.
///
/// The legacy caller ID remains available to the IPC submit builder but is
/// ignored here. Decision fields and supported speech stay typed in memory;
/// this function constructs no wire envelope or output-directory path.
pub fn build_direct_native(
    catalog: &Catalog,
    draft: TaskFormDraft,
) -> Result<(InferenceRequest, Duration), TaskFormError> {
    let TaskFormDraft {
        selected_profile,
        declared_task,
        request_id: _,
        timeout_ms,
        fields,
    } = draft;

    if timeout_ms == 0 || timeout_ms > MAX_DIRECT_TIMEOUT_MS {
        return Err(TaskFormError::InvalidRequest);
    }
    let request = match fields {
        TaskFields::Decisions { state, questions } => {
            if selected_profile.lane != Lane::Decisions || declared_task.as_str() != "decision" {
                return Err(TaskFormError::ProfileTaskMismatch);
            }
            preflight_direct_decision_fields(&state, &questions)?;
            InferenceRequest::Decision {
                profile: selected_profile.clone(),
                state,
                questions,
            }
        }
        TaskFields::LanguageEmbeddings { input } => {
            if selected_profile.lane != Lane::Language || declared_task.as_str() != "embeddings" {
                return Err(TaskFormError::ProfileTaskMismatch);
            }
            preflight_direct_embedding_fields(&input)?;
            InferenceRequest::Embeddings {
                profile: selected_profile.clone(),
                input,
            }
        }
        TaskFields::MediaWorkflow { graph } => {
            let Some(profile) = catalog.inspect(&selected_profile) else {
                return Err(TaskFormError::ProfileTaskMismatch);
            };
            if selected_profile.lane != Lane::Media
                || profile.runtime.lane != Lane::Media
                || profile.runtime.as_str() != "ComfyUI"
            {
                return Err(TaskFormError::ProfileTaskMismatch);
            }
            if graph.as_object().is_none_or(|object| object.is_empty()) {
                return Err(TaskFormError::InvalidNativeFields);
            }
            preflight_direct_workflow_fields(&graph)?;
            InferenceRequest::Media {
                profile: selected_profile.clone(),
                input: super::requests::MediaInput::Workflow {
                    task: declared_task.clone(),
                    graph,
                },
            }
        }
        TaskFields::Speech {
            text,
            mode,
            instruction,
            voice_reference,
            reference_text,
        } => {
            let supported_speech_profile =
                catalog.inspect(&selected_profile).is_some_and(|profile| {
                    selected_profile.lane == Lane::Media
                        && profile.runtime.as_str() == "audio.cpp"
                        && matches!(
                            (
                                selected_profile.profile_id.as_str(),
                                profile.device.as_ref()
                            ),
                            (CPU_BREEZE_PROFILE_ID, Some(super::catalog::Device::Cpu))
                                | ("breeze-tts-2", Some(super::catalog::Device::Cuda))
                        )
                });
            if !supported_speech_profile {
                return Err(TaskFormError::ProfileTaskMismatch);
            }
            preflight_direct_speech_fields(&text, &instruction, &voice_reference, &reference_text)?;
            let request = InferenceRequest::Media {
                profile: selected_profile.clone(),
                input: super::requests::MediaInput::Speech(SpeechInput {
                    text,
                    mode,
                    instruction,
                    voice_reference: voice_reference.map(PathBuf::from),
                    reference_text,
                }),
            };
            if request.task() != declared_task.as_str() {
                return Err(TaskFormError::ProfileTaskMismatch);
            }
            let InferenceRequest::Media {
                input: super::requests::MediaInput::Speech(input),
                ..
            } = &request
            else {
                unreachable!("speech fields construct a speech request")
            };
            if !is_supported_cpu_breeze_speech(input) {
                return Err(TaskFormError::InvalidNativeFields);
            }
            request
        }
        _ => return Err(TaskFormError::InvalidNativeFields),
    };

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

    request
        .validate(&refreshed)
        .map_err(|_| TaskFormError::InvalidNativeFields)?;

    Ok((request, Duration::from_millis(timeout_ms)))
}

fn preflight_direct_speech_fields(
    text: &str,
    instruction: &Option<String>,
    voice_reference: &Option<String>,
    reference_text: &Option<String>,
) -> Result<(), TaskFormError> {
    let mut total = 0usize;
    for size in [
        bounded_json_len(&text)?,
        bounded_json_len(instruction)?,
        bounded_json_len(voice_reference)?,
        bounded_json_len(reference_text)?,
    ] {
        if size > MAX_TASK_FORM_FIELD_BYTES {
            return Err(TaskFormError::FrameTooLarge);
        }
        total = total.saturating_add(size);
    }
    if total > MAX_TASK_FORM_TOTAL_BYTES {
        return Err(TaskFormError::FrameTooLarge);
    }
    Ok(())
}

fn preflight_direct_decision_fields(
    state: &Value,
    questions: &Map<String, Value>,
) -> Result<(), TaskFormError> {
    let state_bytes = bounded_json_len(state)?;
    let question_bytes = bounded_json_len(questions)?;
    if state_bytes > MAX_TASK_FORM_FIELD_BYTES
        || question_bytes > MAX_TASK_FORM_FIELD_BYTES
        || state_bytes.saturating_add(question_bytes) > MAX_TASK_FORM_TOTAL_BYTES
    {
        return Err(TaskFormError::FrameTooLarge);
    }
    Ok(())
}

fn preflight_direct_embedding_fields(input: &[String]) -> Result<(), TaskFormError> {
    let mut total = 0usize;
    for text in input {
        let size = bounded_json_len(text)?;
        if size > MAX_TASK_FORM_FIELD_BYTES {
            return Err(TaskFormError::FrameTooLarge);
        }
        total = total.saturating_add(size);
        if total > MAX_TASK_FORM_TOTAL_BYTES {
            return Err(TaskFormError::FrameTooLarge);
        }
    }
    Ok(())
}

fn preflight_direct_workflow_fields(graph: &Value) -> Result<(), TaskFormError> {
    let size = bounded_json_len(graph)?;
    if size > MAX_TASK_FORM_FIELD_BYTES || size > MAX_TASK_FORM_TOTAL_BYTES {
        return Err(TaskFormError::FrameTooLarge);
    }
    Ok(())
}

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

    use super::super::requests::SpeechInput;

    fn catalog() -> Catalog {
        Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap()
    }

    fn cpu_breeze_catalog() -> (TempDir, Catalog) {
        let temporary = tempfile::tempdir().unwrap();
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
            serde_json::json!({
                "schema_version": 1,
                "profiles": [{
                    "id": CPU_BREEZE_PROFILE_ID,
                    "name": "CPU Breeze fixture",
                    "runtime": "audio.cpp",
                    "device": "cpu",
                    "tasks": ["tts", "clon", "design"],
                    "script": "media-models/generate-breeze-tts-2.sh"
                }, {
                    "id": "breeze-tts-2",
                    "name": "CUDA Breeze fixture",
                    "runtime": "audio.cpp",
                    "device": "cuda",
                    "tasks": ["tts", "clon", "design"],
                    "script": "media-models/generate-breeze-tts-2.sh"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            root.join("toolkits/l3ds/decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root).unwrap();
        (temporary, catalog)
    }

    fn embedding_catalog() -> (TempDir, Catalog) {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        fs::create_dir_all(root.join("toolkits/l2m2")).unwrap();
        fs::create_dir_all(root.join("toolkits/l3ds")).unwrap();
        fs::write(
            root.join("llama-swap.yaml"),
            "models:\n  fixture-embeddings:\n    cmd: |\n      llama-server --embedding\n",
        )
        .unwrap();
        fs::write(
            root.join("toolkits/l2m2/media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        fs::write(
            root.join("toolkits/l3ds/decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root).unwrap();
        (temporary, catalog)
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

    fn direct_error(
        result: Result<
            (
                super::super::requests::InferenceRequest,
                std::time::Duration,
            ),
            TaskFormError,
        >,
    ) -> TaskFormError {
        match result {
            Err(error) => error,
            Ok(_) => panic!("expected a fixed direct task-form error"),
        }
    }

    fn language_embedding_fields() -> TaskFields {
        TaskFields::LanguageEmbeddings {
            input: vec![
                "private system instruction".into(),
                "private user input".into(),
                "earlier assistant output".into(),
            ],
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
    fn language_embedding_form_builds_explicit_submit() {
        let (_temporary, catalog) = embedding_catalog();
        let selected = profile(super::super::catalog::Lane::Language, "fixture-embeddings");
        let envelope = build_submit(
            &catalog,
            draft(selected.clone(), "embeddings", language_embedding_fields()),
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
        assert_eq!(request.task(), "embeddings");
        let super::super::requests::InferenceRequest::Embeddings { input, .. } = request else {
            panic!("language embedding fields stay lane-native")
        };
        assert_eq!(
            input,
            vec![
                String::from("private system instruction"),
                String::from("private user input"),
                String::from("earlier assistant output"),
            ]
        );
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
    fn direct_native_builder_ignores_legacy_id_and_returns_typed_decision() {
        let mut questions = Map::new();
        questions.insert(
            "route".into(),
            json!({"type": "choice", "criteria": {"yes": "Proceed", "no": "Stop"}}),
        );
        let draft = TaskFormDraft::new(
            profile(super::super::catalog::Lane::Decisions, "nagi-smol"),
            task("decision"),
            "not a wire request ID".into(),
            120_000,
            TaskFields::Decisions {
                state: json!({"ready": true}),
                questions,
            },
        );

        let (request, timeout) = build_direct_native(&catalog(), draft).unwrap();
        assert_eq!(timeout, std::time::Duration::from_secs(120));
        let super::super::requests::InferenceRequest::Decision {
            profile: selected,
            state,
            questions,
        } = request
        else {
            panic!("direct decision fields build an inference request")
        };
        assert_eq!(
            selected,
            profile(super::super::catalog::Lane::Decisions, "nagi-smol")
        );
        assert_eq!(state, json!({"ready": true}));
        assert_eq!(questions["route"]["type"], "choice");
    }

    #[test]
    fn direct_native_builder_accepts_bounded_language_embeddings() {
        let (_temporary, catalog) = embedding_catalog();
        let (request, timeout) = build_direct_native(
            &catalog,
            TaskFormDraft::new(
                profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
                task("embeddings"),
                "ignored legacy id".into(),
                45_000,
                language_embedding_fields(),
            ),
        )
        .unwrap();
        assert_eq!(timeout, std::time::Duration::from_secs(45));
        let super::super::requests::InferenceRequest::Embeddings {
            profile: selected,
            input,
        } = request
        else {
            panic!("direct embedding fields build a typed embeddings request")
        };
        assert_eq!(
            selected,
            profile(super::super::catalog::Lane::Language, "fixture-embeddings")
        );
        assert_eq!(input.len(), 3);
        assert_eq!(input[0], "private system instruction");

        for (profile_lane, task_name, input) in [
            (
                super::super::catalog::Lane::Decisions,
                "embeddings",
                vec!["alpha".into()],
            ),
            (
                super::super::catalog::Lane::Language,
                "chat",
                vec!["alpha".into()],
            ),
            (
                super::super::catalog::Lane::Language,
                "embeddings",
                Vec::new(),
            ),
            (
                super::super::catalog::Lane::Language,
                "embeddings",
                vec!["\0".into()],
            ),
        ] {
            let error = direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(profile_lane, "fixture-embeddings"),
                    task(task_name),
                    String::new(),
                    30_000,
                    TaskFields::LanguageEmbeddings { input },
                ),
            ));
            assert_eq!(
                error,
                if profile_lane != super::super::catalog::Lane::Language
                    || task_name != "embeddings"
                {
                    TaskFormError::ProfileTaskMismatch
                } else {
                    TaskFormError::InvalidNativeFields
                }
            );
        }

        let too_large = vec!["x".repeat(MAX_TASK_FORM_FIELD_BYTES + 1)];
        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
                    task("embeddings"),
                    String::new(),
                    30_000,
                    TaskFields::LanguageEmbeddings { input: too_large },
                ),
            )),
            TaskFormError::FrameTooLarge
        );

        let too_much_total = vec!["x".repeat(470_000), "y".repeat(470_000)];
        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
                    task("embeddings"),
                    String::new(),
                    30_000,
                    TaskFields::LanguageEmbeddings {
                        input: too_much_total,
                    },
                ),
            )),
            TaskFormError::FrameTooLarge
        );
    }

    #[test]
    fn direct_native_builder_accepts_cpu_breeze_modes_and_rejects_field_conflicts() {
        let (_temporary, catalog) = cpu_breeze_catalog();
        for (mode, declared_task) in [
            (SpeechMode::Standard, "tts"),
            (SpeechMode::VoiceDesign, "design"),
            (SpeechMode::VoiceClone, "clon"),
            (SpeechMode::Streaming, "tts"),
        ] {
            let draft = TaskFormDraft::new(
                profile(super::super::catalog::Lane::Media, CPU_BREEZE_PROFILE_ID),
                task(declared_task),
                "ignored legacy id".into(),
                120_000,
                speech_fields(mode),
            );

            let (request, timeout) = build_direct_native(&catalog, draft).unwrap();
            assert_eq!(timeout, std::time::Duration::from_secs(120));
            let super::super::requests::InferenceRequest::Media {
                profile: selected,
                input: super::super::requests::MediaInput::Speech(input),
            } = request
            else {
                panic!("direct speech fields build a typed media request")
            };
            assert_eq!(selected.profile_id.as_str(), CPU_BREEZE_PROFILE_ID);
            assert_eq!(input.mode, mode);
            assert_eq!(input.text, "hello there");
            assert_eq!(
                input.instruction.as_deref(),
                (mode == SpeechMode::VoiceDesign).then_some("warm alto voice")
            );
            if mode == SpeechMode::VoiceClone {
                assert!(input.voice_reference.is_some());
                assert_eq!(input.reference_text.as_deref(), Some("spoken words"));
            }
        }

        for (mode, declared_task, instruction, voice_reference, reference_text) in [
            (SpeechMode::Standard, "tts", Some("extra"), None, None),
            (SpeechMode::VoiceDesign, "design", None, None, None),
            (SpeechMode::VoiceDesign, "design", Some("   "), None, None),
            (
                SpeechMode::VoiceClone,
                "clon",
                None,
                None,
                Some("spoken words"),
            ),
            (
                SpeechMode::VoiceClone,
                "clon",
                None,
                Some("/private/reference.wav"),
                None,
            ),
            (
                SpeechMode::VoiceClone,
                "clon",
                Some("not allowed"),
                Some("/private/reference.wav"),
                Some("spoken words"),
            ),
            (
                SpeechMode::Streaming,
                "tts",
                None,
                None,
                Some("unexpected reference"),
            ),
        ] {
            let result = build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Media, CPU_BREEZE_PROFILE_ID),
                    task(declared_task),
                    String::new(),
                    30_000,
                    TaskFields::Speech {
                        text: "hello there".into(),
                        mode,
                        instruction: instruction.map(str::to_owned),
                        voice_reference: voice_reference.map(str::to_owned),
                        reference_text: reference_text.map(str::to_owned),
                    },
                ),
            );
            assert_eq!(direct_error(result), TaskFormError::InvalidNativeFields);
        }

        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Media, CPU_BREEZE_PROFILE_ID),
                    task("tts"),
                    String::new(),
                    30_000,
                    speech_fields(SpeechMode::VoiceDesign),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );

        for (mode, declared_task) in [
            (SpeechMode::Standard, "tts"),
            (SpeechMode::VoiceDesign, "design"),
            (SpeechMode::VoiceClone, "clon"),
            (SpeechMode::Streaming, "tts"),
        ] {
            // Building a declared CUDA request does not grant GPU admission.
            // The shared runner still requires an explicit idle operator window.
            let (request, _) = build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                    task(declared_task),
                    String::new(),
                    30_000,
                    speech_fields(mode),
                ),
            )
            .unwrap();
            assert_eq!(request.task(), declared_task);
        }
    }

    #[test]
    fn direct_native_builder_rejects_wrong_profile_fields_and_timeout() {
        let catalog = catalog();
        let fields = || TaskFields::Decisions {
            state: json!({"ready": true}),
            questions: Map::from_iter([("route".into(), json!({"type": "rank"}))]),
        };

        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                    task("decision"),
                    String::new(),
                    30_000,
                    fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Decisions, "nagi-smol"),
                    task("not-decision"),
                    String::new(),
                    30_000,
                    fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Decisions, "nagi-smol"),
                    task("decision"),
                    String::new(),
                    30_000,
                    language_embedding_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
        for timeout_ms in [0, MAX_DIRECT_TIMEOUT_MS + 1] {
            assert_eq!(
                direct_error(build_direct_native(
                    &catalog,
                    TaskFormDraft::new(
                        profile(super::super::catalog::Lane::Decisions, "nagi-smol"),
                        task("decision"),
                        String::new(),
                        timeout_ms,
                        fields(),
                    ),
                )),
                TaskFormError::InvalidRequest
            );
        }
        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Decisions, "nagi-smol"),
                    task("decision"),
                    String::new(),
                    30_000,
                    TaskFields::Decisions {
                        state: json!([]),
                        questions: Map::from_iter([("route".into(), json!({"type": "rank"}))]),
                    },
                ),
            )),
            TaskFormError::InvalidNativeFields
        );
    }

    #[test]
    fn direct_native_builder_rejects_catalog_route_drift() {
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
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        let decision_manifest = root.join("toolkits/l3ds/decision-runtimes.json");
        fs::write(
            &decision_manifest,
            r#"{"version":"1.0","profiles":[{"id":"fixture-decision","name":"Fixture","engine":"nagi","type":"local","device":"cpu","model":"fixture","entrypoint":"run.py"}]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root).unwrap();
        fs::write(
            &decision_manifest,
            r#"{"version":"1.0","profiles":[{"id":"fixture-decision","name":"Fixture","engine":"laya","type":"local","device":"cpu","model":"fixture","entrypoint":"run.py"}]}"#,
        )
        .unwrap();

        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Decisions, "fixture-decision"),
                    task("decision"),
                    String::new(),
                    30_000,
                    TaskFields::Decisions {
                        state: json!({"ready": true}),
                        questions: Map::from_iter([("route".into(), json!({"type": "rank"}))]),
                    },
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
    }

    #[test]
    fn direct_native_embedding_builder_rejects_catalog_route_drift() {
        let (root, catalog) = embedding_catalog();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-embeddings:\n    cmd: llama-server --port 8080\n",
        )
        .unwrap();

        assert_eq!(
            direct_error(build_direct_native(
                &catalog,
                TaskFormDraft::new(
                    profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
                    task("embeddings"),
                    String::new(),
                    30_000,
                    language_embedding_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
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
        let base_catalog = catalog();
        let (_temporary, catalog) = embedding_catalog();
        assert_eq!(
            error(build_submit(
                &base_catalog,
                draft(
                    profile(super::super::catalog::Lane::Media, "breeze-tts-2"),
                    "tts",
                    language_embedding_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
                    "not-declared",
                    language_embedding_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );
        assert_eq!(
            error(build_submit(
                &catalog,
                draft(
                    profile(super::super::catalog::Lane::Language, "unknown-profile"),
                    "embeddings",
                    language_embedding_fields(),
                ),
            )),
            TaskFormError::ProfileTaskMismatch
        );

        let invalid_id = TaskFormDraft::new(
            profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
            task("embeddings"),
            "sk-example-fixture".into(),
            30_000,
            language_embedding_fields(),
        );
        assert_eq!(
            error(build_submit(&catalog, invalid_id)),
            TaskFormError::InvalidRequest
        );

        let invalid_timeout = TaskFormDraft::new(
            profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
            task("embeddings"),
            "caller:bad-timeout".into(),
            0,
            language_embedding_fields(),
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
            profile(super::super::catalog::Lane::Language, "fixture-embeddings"),
            task("embeddings"),
            "caller:private-id".into(),
            30_000,
            TaskFields::LanguageEmbeddings {
                input: vec![secret.into()],
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
