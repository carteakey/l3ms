use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    ffi::OsString,
    fs,
    io::{self, BufRead, BufReader, Stdout},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::control::requests::SpeechMode;
#[cfg(target_os = "linux")]
use crate::control::{
    comfy::{ComfyAdapter, ComfyContract},
    direct_run::{self, DirectRunError, DirectRunResult},
    language::LanguageAdapter,
    runtime::{Cancellation, OperationContext},
    task_form::build_direct_native,
};
use crate::{
    chat::{
        detect_chat_server, detected_servers, terminate_detected_server, ChatClient,
        ChatCompletion, ChatModelInventory, ChatRequest,
    },
    chat_history::ChatHistory,
    commands::{command, search_all_commands, visible_commands, CommandContext, CommandId},
    config_store::{load_config, DownloadConfig},
    control::task_form::{
        TaskFields, TaskFormDraft, TaskFormError, MAX_TASK_FORM_FIELD_BYTES,
        MAX_TASK_FORM_TOTAL_BYTES,
    },
    control::{
        catalog::{Catalog, CatalogProfile, Device, Lane},
        routes::{resolve_profile_route, NativeRoute},
    },
    download_editor::DownloadEditor,
    download_preflight::{
        probe_disk_space, run_download_preflight_cancellable, DiskSpace, DownloadPreflight,
    },
    download_ui::{DownloadFocus, DownloadUiState, ModelField},
    downloader_command::downloader_command_prefix,
    gguf::{self, GgufFile},
    job_history::{CrossToolkitJobStart, CrossToolkitJobUpdate, JobHistory},
    llama_swap::{SwapClient, SwapModel, DEFAULT_BASE_URL},
    prompt_store::{list_prompts, load_prompt, prompts_path},
    script_editor::ScriptEditorState,
    script_lint::lint_shell_script,
    script_store::{collect_scripts_in, command_for_script, pretty_name, ScriptMode},
    state_store::{self, ChatSession, ChatSessionList, ChatSessionSummary, SavedChatSession},
    telemetry::{
        cpu_percent_between, find_process_named, query_gpu_telemetry, query_host_telemetry,
        query_server_io, query_system_memory, snapshot_descendants, snapshot_process_group,
        GpuTelemetry, HostTelemetry, ServerIoTelemetry, SystemMemoryTelemetry,
        SystemTelemetrySnapshot,
    },
    text_buffer::TextBuffer,
    theme,
    ui::{
        model_view, shell,
        sidebar::{self, JobRow, ModelRow, Section as SidebarSection, SidebarModel},
        statusbar::{self, StatusInputs},
        toolbar,
    },
};
use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Style, Stylize},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Cell, Clear, Gauge, List, ListItem, ListState, Paragraph, Row, Sparkline,
        Table, TableState, Wrap,
    },
    Frame, Terminal,
};
use unicode_width::UnicodeWidthStr;

type Tui = Terminal<CrosstermBackend<Stdout>>;

const BACKGROUND_QUEUE_CAPACITY: usize = 512;
const EVENTS_PER_TICK: usize = 128;
const MAX_PROCESS_LINE_BYTES: usize = 16 * 1024;
const MAX_CHAT_PREVIEW_BYTES: usize = 64 * 1024;
const TELEMETRY_INTERVAL: Duration = Duration::from_secs(1);
/// Router state refresh. `GET /v1/models` never loads a model, so polling it is
/// safe; other clients swap models, and the sidebar and the load warning must
/// not go stale.
const MODELS_REFRESH_INTERVAL: Duration = Duration::from_secs(5);
const DOWNLOAD_PAGE_SIZE: usize = 50;
const DOWNLOAD_ESTIMATE_TIMEOUT: Duration = Duration::from_secs(180);
const DOWNLOAD_DISK_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_STACK_OUTPUT_PREVIEW_CHARS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq)]
struct DownloadProgressSnapshot {
    downloaded_bytes: u64,
    total_bytes: u64,
    speed_bytes_per_second: Option<f64>,
}

/// What the pane shows. The sidebar is always there; views replace the old
/// fixed tabs and keep their F-key order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Model,
    Chat,
    Jobs,
    Bench,
    Download,
    Browser,
    Maintenance,
    System,
    Log,
}

impl View {
    const ALL: [Self; 9] = [
        Self::Model,
        Self::Chat,
        Self::Jobs,
        Self::Bench,
        Self::Download,
        Self::Browser,
        Self::Maintenance,
        Self::System,
        Self::Log,
    ];

    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|view| *view == self)
            .unwrap_or_default()
    }

    fn from_index(index: usize) -> Self {
        Self::ALL[index % Self::ALL.len()]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Model => "Model",
            Self::Chat => "Chat",
            Self::Jobs => "Jobs",
            Self::Bench => "Bench",
            Self::Download => "Download",
            Self::Browser => "Model files",
            Self::Maintenance => "Maintenance",
            Self::System => "System",
            Self::Log => "Log",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptEditorTarget {
    Bench,
    Maintenance,
}

impl ScriptEditorTarget {
    fn label(self) -> &'static str {
        match self {
            Self::Bench => "Bench",
            Self::Maintenance => "Maintenance",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EditorViewport {
    cursor_byte: usize,
    scroll_y: usize,
    scroll_x: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InputMode {
    Normal,
    ModelFilter,
    BenchFilter,
    BrowserPath,
    BrowserFilter,
    ChatMessage,
    ChatEndpoint,
    ChatSystemPrompt,
    TaskForm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskFormField {
    Timeout,
    DeclaredTask,
    MediaOperation,
    EmbeddingInputs,
    MediaWorkflowGraph,
    MediaScriptArgv,
    SpeechText,
    SpeechMode,
    SpeechInstruction,
    SpeechVoiceReference,
    SpeechReferenceText,
    DecisionState,
    DecisionQuestions,
    DecisionImages,
    Run,
}

impl TaskFormField {
    fn label(self) -> &'static str {
        match self {
            Self::Timeout => "Timeout (ms)",
            Self::DeclaredTask => "Declared task",
            Self::MediaOperation => "Media input kind",
            Self::EmbeddingInputs => "Embedding texts JSON array",
            Self::MediaWorkflowGraph => "Workflow graph JSON",
            Self::MediaScriptArgv => "Script argv JSON array",
            Self::SpeechText => "Speech text",
            Self::SpeechMode => "Speech mode",
            Self::SpeechInstruction => "Design instruction",
            Self::SpeechVoiceReference => "Voice reference path",
            Self::SpeechReferenceText => "Reference transcript",
            Self::DecisionState => "Decision state JSON",
            Self::DecisionQuestions => "Decision questions JSON",
            Self::DecisionImages => "Decision images data URL JSON array",
            Self::Run => "Run once",
        }
    }

    fn is_selector(self) -> bool {
        matches!(
            self,
            Self::DeclaredTask | Self::MediaOperation | Self::SpeechMode
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskFormMediaOperation {
    Workflow,
    Script,
    Speech,
}

impl TaskFormMediaOperation {
    const ALL: [Self; 3] = [Self::Workflow, Self::Script, Self::Speech];

    fn label(self) -> &'static str {
        match self {
            Self::Workflow => "workflow graph",
            Self::Script => "native script argv",
            Self::Speech => "typed speech",
        }
    }
}

impl SpeechMode {
    const BREEZE_TTS_FORM_MODES: [Self; 2] = [Self::Standard, Self::Streaming];
    const TASK_FORM_MODES: [Self; 4] = [
        Self::Standard,
        Self::VoiceDesign,
        Self::VoiceClone,
        Self::Streaming,
    ];

    fn task_form_label(self) -> &'static str {
        match self {
            Self::Standard => "standard (tts)",
            Self::VoiceDesign => "design",
            Self::VoiceClone => "clone (clon)",
            Self::Streaming => "streaming (tts)",
        }
    }
}

#[derive(Default)]
struct TaskFormValues {
    timeout: String,
    embedding_inputs: String,
    workflow_graph: String,
    script_argv: String,
    speech_text: String,
    speech_instruction: String,
    speech_voice_reference: String,
    speech_reference_text: String,
    decision_state: String,
    decision_questions: String,
    decision_images: String,
}

impl TaskFormValues {
    fn total_bytes(&self) -> usize {
        [
            self.timeout.len(),
            self.embedding_inputs.len(),
            self.workflow_graph.len(),
            self.script_argv.len(),
            self.speech_text.len(),
            self.speech_instruction.len(),
            self.speech_voice_reference.len(),
            self.speech_reference_text.len(),
            self.decision_state.len(),
            self.decision_questions.len(),
            self.decision_images.len(),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add)
    }

    fn clear_media_inputs(&mut self) {
        self.workflow_graph.clear();
        self.script_argv.clear();
        self.speech_text.clear();
        self.speech_instruction.clear();
        self.speech_voice_reference.clear();
        self.speech_reference_text.clear();
    }
}

/// Sensitive form contents are kept only in this transient editor and never
/// formatted through Debug, status text, or activity logs.
struct TaskFormEditor {
    profile: crate::control::catalog::ProfileKey,
    profile_snapshot: CatalogProfile,
    direct_native_decision: bool,
    direct_native_decision_images: bool,
    direct_breeze_speech: bool,
    direct_native_music: bool,
    direct_native_h3: bool,
    direct_blender_headless: bool,
    direct_comfy_workflow: bool,
    direct_language_embeddings: bool,
    declared_tasks: Vec<String>,
    selected_task: Option<usize>,
    media_operation: Option<TaskFormMediaOperation>,
    speech_mode: Option<SpeechMode>,
    fields: Vec<TaskFormField>,
    active: usize,
    cursor_byte: usize,
    values: TaskFormValues,
}

impl TaskFormEditor {
    fn new_with_direct_embeddings(
        profile: &CatalogProfile,
        direct_language_embeddings: bool,
    ) -> Self {
        let direct_native_decision = is_direct_native_decision_profile(profile);
        let direct_native_decision_images = direct_native_decision
            && crate::control::requests::decision_profile_supports_images(profile);
        let direct_breeze_speech = is_direct_breeze_speech_profile(profile);
        let direct_native_music = is_direct_cuda_music_profile(profile);
        let direct_native_h3 = is_direct_cuda_h3_profile(profile);
        let direct_blender_headless = is_direct_blender_headless_profile(profile);
        #[cfg(target_os = "linux")]
        let direct_comfy_workflow = is_direct_comfy_workflow_profile(profile);
        #[cfg(not(target_os = "linux"))]
        let direct_comfy_workflow = false;
        let direct_language_embeddings = direct_language_embeddings
            && profile.key.lane == Lane::Language
            && profile
                .tasks
                .iter()
                .any(|task| task.as_str() == "embeddings");
        let mut declared_tasks = profile
            .tasks
            .iter()
            .map(|task| task.as_str().to_owned())
            .collect::<Vec<_>>();
        if direct_breeze_speech {
            declared_tasks.retain(|task| matches!(task.as_str(), "tts" | "clon" | "design"));
        }
        if direct_native_music {
            declared_tasks.retain(|task| task == "music");
        }
        if direct_native_h3 {
            declared_tasks.retain(|task| matches!(task.as_str(), "music" | "tts"));
        }
        if direct_blender_headless {
            declared_tasks.retain(|task| matches!(task.as_str(), "3d" | "render" | "image"));
        }
        if profile.key.lane == Lane::Language
            && profile
                .tasks
                .iter()
                .any(|task| task.as_str() == "embeddings")
        {
            declared_tasks.retain(|task| task == "embeddings");
        }
        let mut editor = Self {
            profile: profile.key.clone(),
            profile_snapshot: profile.clone(),
            direct_native_decision,
            direct_native_decision_images,
            direct_breeze_speech,
            direct_native_music,
            direct_native_h3,
            direct_blender_headless,
            direct_comfy_workflow,
            direct_language_embeddings,
            declared_tasks,
            selected_task: None,
            media_operation: None,
            speech_mode: None,
            fields: Vec::new(),
            active: 0,
            cursor_byte: 0,
            values: TaskFormValues::default(),
        };
        if direct_native_decision
            || direct_breeze_speech
            || editor.is_direct_script_media()
            || direct_comfy_workflow
            || direct_language_embeddings
        {
            // Foreground native work has no caller ID to retain. A useful
            // bounded timeout keeps the direct path keyboard-first.
            editor.values.timeout = "30000".into();
        }
        if direct_native_decision {
            editor.values.decision_state = "{}".into();
            editor.selected_task = editor
                .declared_tasks
                .iter()
                .position(|task| task == "decision");
        }
        if direct_native_decision_images {
            editor.values.decision_images = "[]".into();
        }
        if direct_breeze_speech {
            editor.selected_task = editor
                .declared_tasks
                .iter()
                .position(|task| task == "tts")
                .or_else(|| (!editor.declared_tasks.is_empty()).then_some(0));
            editor.media_operation = Some(TaskFormMediaOperation::Speech);
            editor.speech_mode = Some(SpeechMode::Standard);
        }
        if direct_native_music {
            editor.selected_task = editor
                .declared_tasks
                .iter()
                .position(|task| task == "music");
            editor.media_operation = Some(TaskFormMediaOperation::Script);
        }
        if direct_native_h3 {
            editor.selected_task = editor
                .declared_tasks
                .iter()
                .position(|task| task == "music")
                .or_else(|| (!editor.declared_tasks.is_empty()).then_some(0));
            editor.media_operation = Some(TaskFormMediaOperation::Script);
        }
        if direct_blender_headless {
            editor.selected_task = editor
                .declared_tasks
                .iter()
                .position(|task| task == "3d")
                .or_else(|| (!editor.declared_tasks.is_empty()).then_some(0));
            editor.media_operation = Some(TaskFormMediaOperation::Script);
        }
        if direct_comfy_workflow {
            editor.media_operation = Some(TaskFormMediaOperation::Workflow);
        }
        if direct_language_embeddings {
            editor.selected_task = editor
                .declared_tasks
                .iter()
                .position(|task| task == "embeddings");
        }
        editor.refresh_fields();
        editor
    }

    fn is_direct_native(&self) -> bool {
        self.direct_native_decision
            || self.direct_breeze_speech
            || self.is_direct_script_media()
            || self.direct_comfy_workflow
            || self.direct_language_embeddings
    }

    fn is_direct_script_media(&self) -> bool {
        self.direct_native_music || self.direct_native_h3 || self.direct_blender_headless
    }

    fn refresh_fields(&mut self) {
        let mut fields = Vec::with_capacity(8);
        fields.push(TaskFormField::Timeout);
        if !self.direct_native_decision
            && (!self.direct_breeze_speech || self.declared_tasks.len() > 1)
            && (!self.is_direct_script_media() || self.declared_tasks.len() > 1)
            && !self.direct_language_embeddings
        {
            fields.push(TaskFormField::DeclaredTask);
        }
        match self.profile.lane {
            Lane::Language => {
                if self
                    .selected_task
                    .and_then(|index| self.declared_tasks.get(index))
                    .is_some_and(|task| task == "embeddings")
                {
                    fields.push(TaskFormField::EmbeddingInputs);
                }
            }
            Lane::Decisions => {
                fields.extend([
                    TaskFormField::DecisionState,
                    TaskFormField::DecisionQuestions,
                ]);
                if self.direct_native_decision_images {
                    fields.push(TaskFormField::DecisionImages);
                }
            }
            Lane::Media => {
                if self.direct_breeze_speech {
                    match self
                        .selected_task
                        .and_then(|index| self.declared_tasks.get(index))
                        .map(String::as_str)
                    {
                        Some("tts") => {
                            fields.extend([TaskFormField::SpeechText, TaskFormField::SpeechMode])
                        }
                        Some("design") => fields
                            .extend([TaskFormField::SpeechText, TaskFormField::SpeechInstruction]),
                        Some("clon") => fields.extend([
                            TaskFormField::SpeechText,
                            TaskFormField::SpeechVoiceReference,
                            TaskFormField::SpeechReferenceText,
                        ]),
                        _ => fields.push(TaskFormField::SpeechText),
                    }
                } else if self.direct_comfy_workflow {
                    fields.push(TaskFormField::MediaWorkflowGraph);
                } else if self.is_direct_script_media() {
                    fields.push(TaskFormField::MediaScriptArgv);
                } else {
                    fields.push(TaskFormField::MediaOperation);
                    match self.media_operation {
                        Some(TaskFormMediaOperation::Workflow) => {
                            fields.push(TaskFormField::MediaWorkflowGraph);
                        }
                        Some(TaskFormMediaOperation::Script) => {
                            fields.push(TaskFormField::MediaScriptArgv);
                        }
                        Some(TaskFormMediaOperation::Speech) => fields.extend([
                            TaskFormField::SpeechText,
                            TaskFormField::SpeechMode,
                            TaskFormField::SpeechInstruction,
                            TaskFormField::SpeechVoiceReference,
                            TaskFormField::SpeechReferenceText,
                        ]),
                        None => {}
                    }
                }
            }
        }
        fields.push(TaskFormField::Run);
        self.fields = fields;
        self.active = self.active.min(self.fields.len().saturating_sub(1));
        self.sync_cursor();
    }

    fn active_field(&self) -> TaskFormField {
        self.fields[self.active]
    }

    fn current_text(&self) -> Option<&str> {
        self.value(self.active_field()).map(String::as_str)
    }

    fn current_text_mut(&mut self) -> Option<&mut String> {
        self.value_mut(self.active_field())
    }

    fn value(&self, field: TaskFormField) -> Option<&String> {
        match field {
            TaskFormField::Timeout => Some(&self.values.timeout),
            TaskFormField::EmbeddingInputs => Some(&self.values.embedding_inputs),
            TaskFormField::MediaWorkflowGraph => Some(&self.values.workflow_graph),
            TaskFormField::MediaScriptArgv => Some(&self.values.script_argv),
            TaskFormField::SpeechText => Some(&self.values.speech_text),
            TaskFormField::SpeechInstruction => Some(&self.values.speech_instruction),
            TaskFormField::SpeechVoiceReference => Some(&self.values.speech_voice_reference),
            TaskFormField::SpeechReferenceText => Some(&self.values.speech_reference_text),
            TaskFormField::DecisionState => Some(&self.values.decision_state),
            TaskFormField::DecisionQuestions => Some(&self.values.decision_questions),
            TaskFormField::DecisionImages => Some(&self.values.decision_images),
            TaskFormField::DeclaredTask
            | TaskFormField::MediaOperation
            | TaskFormField::SpeechMode
            | TaskFormField::Run => None,
        }
    }

    fn value_mut(&mut self, field: TaskFormField) -> Option<&mut String> {
        match field {
            TaskFormField::Timeout => Some(&mut self.values.timeout),
            TaskFormField::EmbeddingInputs => Some(&mut self.values.embedding_inputs),
            TaskFormField::MediaWorkflowGraph => Some(&mut self.values.workflow_graph),
            TaskFormField::MediaScriptArgv => Some(&mut self.values.script_argv),
            TaskFormField::SpeechText => Some(&mut self.values.speech_text),
            TaskFormField::SpeechInstruction => Some(&mut self.values.speech_instruction),
            TaskFormField::SpeechVoiceReference => Some(&mut self.values.speech_voice_reference),
            TaskFormField::SpeechReferenceText => Some(&mut self.values.speech_reference_text),
            TaskFormField::DecisionState => Some(&mut self.values.decision_state),
            TaskFormField::DecisionQuestions => Some(&mut self.values.decision_questions),
            TaskFormField::DecisionImages => Some(&mut self.values.decision_images),
            TaskFormField::DeclaredTask
            | TaskFormField::MediaOperation
            | TaskFormField::SpeechMode
            | TaskFormField::Run => None,
        }
    }

    fn field_display(&self, field: TaskFormField) -> String {
        match field {
            TaskFormField::DeclaredTask => self
                .selected_task
                .and_then(|index| self.declared_tasks.get(index))
                .cloned()
                .unwrap_or_else(|| "(choose with ↑/↓)".into()),
            TaskFormField::MediaOperation => self
                .media_operation
                .map(TaskFormMediaOperation::label)
                .unwrap_or("(choose with ↑/↓)")
                .to_owned(),
            TaskFormField::SpeechMode => self
                .speech_mode
                .map(SpeechMode::task_form_label)
                .unwrap_or("(choose with ↑/↓)")
                .to_owned(),
            TaskFormField::Run => "Ctrl+Enter or focus this row and press Enter".into(),
            text_field => self
                .value(text_field)
                .filter(|value| !value.is_empty())
                .map(|value| safe_form_preview(value, 88))
                .unwrap_or_else(|| "(empty)".into()),
        }
    }

    fn move_focus(&mut self, direction: isize) {
        let count = self.fields.len();
        if count == 0 {
            return;
        }
        self.active = if direction < 0 {
            (self.active + count - 1) % count
        } else {
            (self.active + 1) % count
        };
        self.sync_cursor();
    }

    fn sync_cursor(&mut self) {
        self.cursor_byte = self
            .value(self.active_field())
            .map_or(0, |value| value.len());
    }

    fn cycle_selector(&mut self, direction: isize) -> bool {
        match self.active_field() {
            TaskFormField::DeclaredTask => {
                let Some(index) =
                    next_selection(self.selected_task, self.declared_tasks.len(), direction)
                else {
                    return false;
                };
                self.selected_task = Some(index);
                self.refresh_fields();
                true
            }
            TaskFormField::MediaOperation => {
                let Some(index) = next_selection(
                    self.media_operation.map(|operation| {
                        TaskFormMediaOperation::ALL
                            .iter()
                            .position(|candidate| *candidate == operation)
                            .unwrap_or(0)
                    }),
                    TaskFormMediaOperation::ALL.len(),
                    direction,
                ) else {
                    return false;
                };
                let operation = TaskFormMediaOperation::ALL[index];
                if self.media_operation != Some(operation) {
                    self.values.clear_media_inputs();
                    self.speech_mode = None;
                    self.media_operation = Some(operation);
                    self.refresh_fields();
                }
                true
            }
            TaskFormField::SpeechMode => {
                let modes: &[SpeechMode] = if self.direct_breeze_speech {
                    &SpeechMode::BREEZE_TTS_FORM_MODES
                } else {
                    &SpeechMode::TASK_FORM_MODES
                };
                let Some(index) = next_selection(
                    self.speech_mode
                        .and_then(|mode| modes.iter().position(|candidate| *candidate == mode)),
                    modes.len(),
                    direction,
                ) else {
                    return false;
                };
                self.speech_mode = Some(modes[index]);
                true
            }
            _ => false,
        }
    }

    fn insert_char(&mut self, character: char) -> bool {
        if character.is_control() {
            return false;
        }
        self.insert_text(&character.to_string())
    }

    fn insert_newline(&mut self) -> bool {
        self.insert_text("\n")
    }

    fn insert_text(&mut self, input: &str) -> bool {
        let Some(field) = self.current_text() else {
            return false;
        };
        let field_limit = match self.active_field() {
            TaskFormField::Timeout => 32,
            _ => MAX_TASK_FORM_FIELD_BYTES,
        };
        if field.len().saturating_add(input.len()) > field_limit
            || self.values.total_bytes().saturating_add(input.len()) > MAX_TASK_FORM_TOTAL_BYTES
        {
            return false;
        }
        let cursor = self.cursor_byte.min(field.len());
        {
            let Some(field) = self.current_text_mut() else {
                return false;
            };
            field.insert_str(cursor, input);
        }
        self.cursor_byte = cursor + input.len();
        true
    }

    fn backspace(&mut self) {
        let cursor_byte = self.cursor_byte;
        let previous = {
            let Some(field) = self.current_text_mut() else {
                return;
            };
            let Some((previous, _)) = field[..cursor_byte].char_indices().next_back() else {
                return;
            };
            field.remove(previous);
            previous
        };
        self.cursor_byte = previous;
    }

    fn move_cursor(&mut self, direction: isize) {
        let Some(field) = self.current_text() else {
            return;
        };
        let cursor_byte = if direction < 0 {
            field[..self.cursor_byte]
                .char_indices()
                .next_back()
                .map_or(0, |(index, _)| index)
        } else {
            self.cursor_byte
                + field[self.cursor_byte..]
                    .chars()
                    .next()
                    .map_or(0, char::len_utf8)
        };
        self.cursor_byte = cursor_byte;
    }

    fn move_cursor_edge(&mut self, to_end: bool) {
        if let Some(cursor_byte) = self
            .current_text()
            .map(|field| if to_end { field.len() } else { 0 })
        {
            self.cursor_byte = cursor_byte;
        }
    }

    fn to_draft(&self) -> Result<TaskFormDraft, TaskFormError> {
        let timeout_ms = self
            .values
            .timeout
            .parse::<u64>()
            .map_err(|_| TaskFormError::InvalidRequest)?;
        let selected_task = self
            .selected_task
            .and_then(|index| self.declared_tasks.get(index))
            .ok_or(TaskFormError::ProfileTaskMismatch)?;
        let declared_task = crate::control::catalog::Task::new(selected_task.clone())
            .map_err(|_| TaskFormError::ProfileTaskMismatch)?;
        let fields = match self.profile.lane {
            Lane::Language if selected_task == "embeddings" => TaskFields::LanguageEmbeddings {
                input: serde_json::from_str(&self.values.embedding_inputs)
                    .map_err(|_| TaskFormError::InvalidNativeFields)?,
            },
            Lane::Language => return Err(TaskFormError::ProfileTaskMismatch),
            Lane::Decisions => TaskFields::Decisions {
                state: serde_json::from_str(&self.values.decision_state)
                    .map_err(|_| TaskFormError::InvalidNativeFields)?,
                questions: serde_json::from_str(&self.values.decision_questions)
                    .map_err(|_| TaskFormError::InvalidNativeFields)?,
                images: if self.direct_native_decision_images {
                    serde_json::from_str(&self.values.decision_images)
                        .map_err(|_| TaskFormError::InvalidNativeFields)?
                } else {
                    Vec::new()
                },
            },
            Lane::Media => match self
                .media_operation
                .ok_or(TaskFormError::InvalidNativeFields)?
            {
                TaskFormMediaOperation::Workflow => TaskFields::MediaWorkflow {
                    graph: serde_json::from_str(&self.values.workflow_graph)
                        .map_err(|_| TaskFormError::InvalidNativeFields)?,
                },
                TaskFormMediaOperation::Script => TaskFields::MediaScript {
                    args: serde_json::from_str(&self.values.script_argv)
                        .map_err(|_| TaskFormError::InvalidNativeFields)?,
                },
                TaskFormMediaOperation::Speech => TaskFields::Speech {
                    text: self.values.speech_text.clone(),
                    mode: if self.direct_breeze_speech {
                        match selected_task.as_str() {
                            "tts" => match self.speech_mode {
                                Some(mode @ (SpeechMode::Standard | SpeechMode::Streaming)) => mode,
                                _ => return Err(TaskFormError::InvalidNativeFields),
                            },
                            "design" => SpeechMode::VoiceDesign,
                            "clon" => SpeechMode::VoiceClone,
                            _ => return Err(TaskFormError::InvalidNativeFields),
                        }
                    } else {
                        self.speech_mode.ok_or(TaskFormError::InvalidNativeFields)?
                    },
                    instruction: if !self.direct_breeze_speech || selected_task == "design" {
                        nonempty_form_field(&self.values.speech_instruction)
                    } else {
                        None
                    },
                    voice_reference: if !self.direct_breeze_speech || selected_task == "clon" {
                        nonempty_form_field(&self.values.speech_voice_reference)
                    } else {
                        None
                    },
                    reference_text: if !self.direct_breeze_speech || selected_task == "clon" {
                        nonempty_form_field(&self.values.speech_reference_text)
                    } else {
                        None
                    },
                },
            },
        };
        Ok(TaskFormDraft::new(
            self.profile.clone(),
            declared_task,
            timeout_ms,
            fields,
        ))
    }
}

fn next_selection(current: Option<usize>, count: usize, direction: isize) -> Option<usize> {
    if count == 0 {
        return None;
    }
    Some(match current {
        Some(index) if direction < 0 => (index + count - 1) % count,
        Some(index) => (index + 1) % count,
        None if direction < 0 => count - 1,
        None => 0,
    })
}

fn nonempty_form_field(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

fn safe_form_preview(value: &str, max_chars: usize) -> String {
    let mut rendered = String::new();
    let mut characters = value.chars();
    for character in characters.by_ref().take(max_chars) {
        if character == '\n' {
            rendered.push('↵');
        } else if character.is_control() {
            rendered.push(' ');
        } else {
            rendered.push(character);
        }
    }
    if characters.next().is_some() {
        rendered.push('…');
    }
    rendered
}

/// A bounded, control-safe single-line window around the byte cursor.
/// Native JSON/newlines stay unchanged in memory; only their display is escaped.
fn task_form_text_window(value: &str, cursor: usize, columns: usize) -> (String, usize) {
    let Some(before) = value.get(..cursor) else {
        return (String::new(), 0);
    };
    if columns == 0 {
        return (String::new(), 0);
    }
    let display_char = |character: char| {
        if character == '\n' {
            '↵'
        } else if character.is_control() {
            ' '
        } else {
            character
        }
    };
    let mut start = cursor;
    let mut before_width = 0usize;
    for (index, character) in before.char_indices().rev().take(512) {
        let width = unicode_width::UnicodeWidthChar::width(display_char(character)).unwrap_or(0);
        if before_width.saturating_add(width) > columns.saturating_sub(2) {
            break;
        }
        start = index;
        before_width += width;
    }
    let mut visible = String::new();
    let mut width = 0usize;
    if start > 0 {
        visible.push('…');
        width = 1;
    }
    let cursor_column = (width + before_width).min(columns.saturating_sub(1));
    for character in value[start..].chars().take(512) {
        let character = display_char(character);
        let character_width = unicode_width::UnicodeWidthChar::width(character).unwrap_or(0);
        if width.saturating_add(character_width) > columns {
            break;
        }
        visible.push(character);
        width += character_width;
    }
    (visible, cursor_column)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BrowserSort {
    NameAsc,
    SizeDesc,
    SizeAsc,
    ModifiedDesc,
    ModifiedAsc,
    QuantizationAsc,
}

impl BrowserSort {
    fn next(self) -> Self {
        match self {
            Self::NameAsc => Self::SizeDesc,
            Self::SizeDesc => Self::SizeAsc,
            Self::SizeAsc => Self::ModifiedDesc,
            Self::ModifiedDesc => Self::ModifiedAsc,
            Self::ModifiedAsc => Self::QuantizationAsc,
            Self::QuantizationAsc => Self::NameAsc,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::NameAsc => "name ↑",
            Self::SizeDesc => "size ↓",
            Self::SizeAsc => "size ↑",
            Self::ModifiedDesc => "modified ↓",
            Self::ModifiedAsc => "modified ↑",
            Self::QuantizationAsc => "quant ↑",
        }
    }
}

struct RunningProcess {
    job_id: u64,
    process_group: u32,
    child: Arc<Mutex<Option<Child>>>,
}

struct PendingDownloadPreflight {
    request_id: u64,
    name: String,
    parts: Vec<OsString>,
    target: PathBuf,
    cancellation: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

#[cfg(target_os = "linux")]
struct DirectNativeJob {
    job_id: u64,
    cancellation: Arc<AtomicBool>,
    receiver: mpsc::Receiver<DirectNativeCompletion>,
    worker: thread::JoinHandle<()>,
}

#[cfg(target_os = "linux")]
struct DirectNativeCompletion {
    job_id: u64,
    outcome: Result<DirectRunResult, DirectRunError>,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
enum DirectNativeState {
    Running,
    CancelRequested,
    Succeeded,
    Cancelled,
    Failed,
    Interrupted,
}

#[cfg(target_os = "linux")]
impl DirectNativeState {
    fn label(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::CancelRequested => "cancellation requested · waiting for operation outcome",
            Self::Succeeded => "succeeded",
            Self::Cancelled => "cancelled · dispatch or terminal outcome confirmed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted · completion unconfirmed",
        }
    }
}

#[cfg(target_os = "linux")]
struct DirectNativeSummary {
    job_id: u64,
    lane: Lane,
    profile_id: String,
    task: String,
    requires_idle_gpu_window: bool,
    native_id: Option<String>,
    state: DirectNativeState,
    result: Option<DirectRunResult>,
}

#[cfg(target_os = "linux")]
struct DirectNativeCancellation(Arc<AtomicBool>);

#[cfg(target_os = "linux")]
impl Cancellation for DirectNativeCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

enum BackgroundEvent {
    Models(Result<Vec<SwapModel>, String>),
    ModelAction {
        job_id: u64,
        model_id: String,
        load: bool,
        result: Result<String, String>,
    },
    BrowserScan(Result<Vec<GgufFile>, String>),
    ChatConnected {
        request_id: u64,
        endpoint: String,
        client: ChatClient,
        inventory: ChatModelInventory,
    },
    ChatConnectionFailed {
        request_id: u64,
        operation: String,
        error: String,
    },
    ChatDelta {
        request_id: u64,
        delta: String,
    },
    ChatFinished {
        request_id: u64,
        result: Result<ChatCompletion, String>,
    },
    ChatSessionsListed(Result<ChatSessionList, String>),
    ChatSessionLoaded(Result<ChatSession, String>),
    ChatSessionSaved(Result<SavedChatSession, String>),
    ProcessLine {
        job_id: u64,
        line: String,
    },
    ProcessFinished {
        job_id: u64,
        exit_code: i32,
    },
    DownloadDiskSpace {
        request_id: u64,
        target: PathBuf,
        result: Result<DiskSpace, String>,
    },
    DownloadPreflight {
        request_id: u64,
        result: Result<DownloadPreflight, String>,
    },
    Telemetry(Result<SystemTelemetrySnapshot, String>),
}

struct App {
    root: PathBuf,
    data_root: PathBuf,
    view: View,
    input_mode: InputMode,
    should_quit: bool,
    show_quit_confirmation: bool,
    show_help: bool,
    show_palette: bool,
    palette_query: String,
    palette_state: TableState,
    status: String,
    log: VecDeque<String>,
    sender: mpsc::SyncSender<BackgroundEvent>,
    receiver: mpsc::Receiver<BackgroundEvent>,

    models: Vec<SwapModel>,
    /// Sidebar model filter (`/`).
    model_filter: String,
    loading_models: bool,
    models_last_started: Option<Instant>,
    /// The pending refresh was asked for (startup, `r`), so its result is
    /// reported on the message line.
    models_refresh_announced: bool,
    models_refresh_failing: bool,
    /// The first router listing moves the selection to the resident model.
    selected_resident_once: bool,
    model_action_pending: bool,
    loaded_model_id: Option<String>,

    bench_scripts: Vec<PathBuf>,
    bench_filter: String,
    bench_state: ListState,
    bench_editor: ScriptEditorState,

    chat_input: String,
    chat_history: ChatHistory,
    chat_streaming: String,
    chat_pending: bool,
    chat_endpoint_draft: String,
    chat_endpoint_committed: Option<String>,
    chat_client: Option<ChatClient>,
    chat_models: Vec<SwapModel>,
    chat_non_chat_model_ids: Vec<String>,
    chat_model_state: TableState,
    chat_connection_pending: bool,
    chat_connection_request_id: u64,
    chat_stream_request_id: u64,
    chat_stream_cancellation: Option<Arc<AtomicBool>>,
    chat_system_prompt: String,
    chat_temperature: f64,
    chat_max_tokens: u32,
    chat_thinking: bool,
    chat_sessions: Vec<ChatSessionSummary>,
    chat_sessions_state: TableState,
    show_chat_sessions: bool,
    chat_session_pending: bool,
    chat_prompts: Vec<String>,
    chat_prompts_state: ListState,
    show_chat_prompts: bool,

    browser_path: String,
    browser_filter: String,
    browser_recursive: bool,
    browser_sort: BrowserSort,
    browser_scanned_root: PathBuf,
    browser_files: Vec<GgufFile>,
    browser_state: TableState,
    browser_scanning: bool,

    stack_catalog: Option<Catalog>,
    stack_catalog_error: Option<String>,
    stack_catalog_state: TableState,
    task_form_editor: Option<TaskFormEditor>,
    #[cfg(target_os = "linux")]
    direct_native_job: Option<DirectNativeJob>,
    #[cfg(target_os = "linux")]
    direct_native_admission_closed: bool,
    #[cfg(target_os = "linux")]
    direct_native_summary: Option<DirectNativeSummary>,

    download: DownloadUiState,
    download_page: usize,
    download_state: TableState,
    download_log: VecDeque<String>,
    download_load_error: Option<String>,
    download_disk_space: String,
    download_disk_request_id: u64,
    download_preflight_request_id: u64,
    download_preflight_pending: Option<PendingDownloadPreflight>,
    download_estimate_total: Option<u64>,
    download_progress: Option<DownloadProgressSnapshot>,
    download_started_at: Option<Instant>,
    download_reload_armed: bool,
    download_restore_armed: bool,

    maintenance_scripts: Vec<PathBuf>,
    maintenance_state: ListState,
    maintenance_editor: ScriptEditorState,
    script_input_target: Option<ScriptEditorTarget>,
    script_buffer: TextBuffer,
    bench_editor_view: EditorViewport,
    maintenance_editor_view: EditorViewport,
    script_versions_target: Option<ScriptEditorTarget>,
    script_version_state: ListState,
    script_reload_armed: Option<ScriptEditorTarget>,

    job_history: JobHistory,
    jobs_state: TableState,
    running_process: Option<RunningProcess>,
    telemetry: String,
    telemetry_pending: bool,
    telemetry_last_started: Option<Instant>,
    /// Focus is on the sidebar (true) or the pane.
    sidebar_focused: bool,
    sidebar_section: SidebarSection,
    models_collapsed: bool,
    jobs_collapsed: bool,
    /// Profile id awaiting a second `L` to confirm a model load.
    load_armed: Option<String>,
    /// Catalog index whose Chat opens once the pending Chat connection
    /// (a model listing, never a load) succeeds.
    chat_handoff_pending: Option<usize>,
    /// Last steady decode rate per llama-swap model id, from bench results
    /// whose GGUF path matches the entry's `-m` path.
    bench_rates: HashMap<String, f64>,
    /// llama-swap entries marked `unlisted: true` (aliases); the sidebar
    /// hides them like the router does.
    unlisted_models: HashSet<String>,
    /// Parsed `llama-swap.yaml` entries by id, for the Model view.
    yaml_entries: HashMap<String, crate::swap_yaml::SwapYamlEntry>,
    /// Matching bench runs per entry, newest first (at most five).
    bench_runs: HashMap<String, Vec<crate::bench_results::BenchResult>>,
    /// On-disk size of each entry's GGUF (shard sets summed).
    model_sizes: HashMap<String, u64>,
    /// `llama-swap.yaml` macros, to name the build behind `${…_server}`.
    yaml_macros: HashMap<String, String>,
    host: HostTelemetry,
    cpu_percent: Option<f64>,
    gpu_telemetry: Option<GpuTelemetry>,
    sys_memory: Option<SystemMemoryTelemetry>,
    server_io: Option<ServerIoTelemetry>,
    gpu_util_history: VecDeque<u64>,
    vram_history: VecDeque<u64>,
    last_server_read_bytes: Option<u64>,
    last_server_read_time: Option<Instant>,
}

impl App {
    fn new(root: PathBuf) -> Result<Self> {
        let data_root = state_store::data_root()?;
        Self::new_in(root, data_root)
    }

    fn new_in(root: PathBuf, data_root: PathBuf) -> Result<Self> {
        Self::new_in_with_model_refresh(root, data_root, true)
    }

    #[cfg(test)]
    fn new_in_without_model_refresh(root: PathBuf, data_root: PathBuf) -> Result<Self> {
        Self::new_in_with_model_refresh(root, data_root, false)
    }

    fn new_in_with_model_refresh(
        root: PathBuf,
        data_root: PathBuf,
        refresh_models_on_start: bool,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(BACKGROUND_QUEUE_CAPACITY);
        let bench_editor =
            ScriptEditorState::for_repository(&root).context("initialize bench script editor")?;
        let maintenance_editor = ScriptEditorState::for_repository(&root)
            .context("initialize maintenance script editor")?;
        let config_path = root.join("model_downloader/models_config.json");
        let versions_root = root.join(".toolkit/download_config_versions");
        let (download, config_warning) = match DownloadUiState::open(&config_path, &versions_root) {
            Ok(download) => (download, None),
            Err(error) => {
                let warning =
                    format!("Download config is blocked until a strict reload succeeds: {error:#}");
                let fallback = DownloadEditor::from_config(
                    &config_path,
                    &versions_root,
                    load_config(&config_path),
                );
                (DownloadUiState::new(fallback), Some(warning))
            }
        };
        let browser_path = if download.config().base_models_dir.trim().is_empty() {
            root.join("models").display().to_string()
        } else {
            download.config().base_models_dir.clone()
        };
        let (job_history, job_warning) = match JobHistory::load_in(&data_root, &root) {
            Ok((history, notice)) => {
                let warning =
                    (!notice.is_empty()).then(|| format!("Job history: {}", notice.summary()));
                (history, warning)
            }
            Err(error) => (
                JobHistory::unavailable(),
                Some(format!(
                    "Job history unavailable (clear Jobs to recover): {error:#}"
                )),
            ),
        };
        let mut app = Self {
            root,
            data_root,
            view: View::Model,
            input_mode: InputMode::Normal,
            should_quit: false,
            show_quit_confirmation: false,
            show_help: false,
            show_palette: false,
            palette_query: String::new(),
            palette_state: TableState::default().with_selected(Some(0)),
            status: "Starting Rust workbench…".into(),
            log: VecDeque::new(),
            sender,
            receiver,
            models: Vec::new(),
            model_filter: String::new(),
            loading_models: false,
            models_last_started: None,
            models_refresh_announced: false,
            models_refresh_failing: false,
            selected_resident_once: false,
            model_action_pending: false,
            loaded_model_id: None,
            bench_scripts: Vec::new(),
            bench_filter: String::new(),
            bench_state: ListState::default().with_selected(Some(0)),
            bench_editor,
            chat_input: String::new(),
            chat_history: ChatHistory::default(),
            chat_streaming: String::new(),
            chat_pending: false,
            chat_endpoint_draft: env::var("LLAMA_SWAP_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.into()),
            chat_endpoint_committed: None,
            chat_client: None,
            chat_models: Vec::new(),
            chat_non_chat_model_ids: Vec::new(),
            chat_model_state: TableState::default().with_selected(Some(0)),
            chat_connection_pending: false,
            chat_connection_request_id: 0,
            chat_stream_request_id: 0,
            chat_stream_cancellation: None,
            chat_system_prompt: String::new(),
            chat_temperature: 0.8,
            chat_max_tokens: 2048,
            chat_thinking: false,
            chat_sessions: Vec::new(),
            chat_sessions_state: TableState::default().with_selected(Some(0)),
            show_chat_sessions: false,
            chat_session_pending: false,
            chat_prompts: Vec::new(),
            chat_prompts_state: ListState::default().with_selected(Some(0)),
            show_chat_prompts: false,
            browser_path,
            browser_filter: String::new(),
            browser_recursive: true,
            browser_sort: BrowserSort::SizeDesc,
            browser_scanned_root: PathBuf::new(),
            browser_files: Vec::new(),
            browser_state: TableState::default().with_selected(Some(0)),
            browser_scanning: false,
            stack_catalog: None,
            stack_catalog_error: None,
            stack_catalog_state: TableState::default().with_selected(Some(0)),
            task_form_editor: None,
            #[cfg(target_os = "linux")]
            direct_native_job: None,
            #[cfg(target_os = "linux")]
            direct_native_admission_closed: false,
            #[cfg(target_os = "linux")]
            direct_native_summary: None,
            download,
            download_page: 0,
            download_state: TableState::default().with_selected(Some(0)),
            download_log: VecDeque::new(),
            download_load_error: config_warning.clone(),
            download_disk_space: "Disk: —".into(),
            download_disk_request_id: 0,
            download_preflight_request_id: 0,
            download_preflight_pending: None,
            download_estimate_total: None,
            download_progress: None,
            download_started_at: None,
            download_reload_armed: false,
            download_restore_armed: false,
            maintenance_scripts: Vec::new(),
            maintenance_state: ListState::default().with_selected(Some(0)),
            maintenance_editor,
            script_input_target: None,
            script_buffer: TextBuffer::new(),
            bench_editor_view: EditorViewport::default(),
            maintenance_editor_view: EditorViewport::default(),
            script_versions_target: None,
            script_version_state: ListState::default().with_selected(Some(0)),
            script_reload_armed: None,
            job_history,
            jobs_state: TableState::default().with_selected(Some(0)),
            running_process: None,
            telemetry: "Resources: idle".into(),
            telemetry_pending: false,
            telemetry_last_started: None,
            sidebar_focused: true,
            sidebar_section: SidebarSection::Models,
            models_collapsed: false,
            jobs_collapsed: false,
            load_armed: None,
            chat_handoff_pending: None,
            bench_rates: HashMap::new(),
            unlisted_models: HashSet::new(),
            yaml_entries: HashMap::new(),
            bench_runs: HashMap::new(),
            model_sizes: HashMap::new(),
            yaml_macros: HashMap::new(),
            host: HostTelemetry::default(),
            cpu_percent: None,
            gpu_telemetry: None,
            sys_memory: None,
            server_io: None,
            gpu_util_history: VecDeque::with_capacity(60),
            vram_history: VecDeque::with_capacity(60),
            last_server_read_bytes: None,
            last_server_read_time: None,
        };
        if let Some(warning) = config_warning {
            app.status = "Download config needs attention".into();
            app.record_download_message(warning);
        }
        if let Some(warning) = app.download.take_history_warning() {
            app.status = "Download config loaded; snapshot history needs attention".into();
            app.record_download_message(warning);
        }
        if let Some(warning) = job_warning {
            app.push_log(warning);
        }
        clamp_table_selection(&mut app.jobs_state, app.job_history.records().len());
        app.sync_download_table_state();
        app.refresh_download_disk_space();
        app.refresh_local_inventories();
        app.refresh_stack_catalog();
        app.refresh_bench_rates();
        app.ensure_visible_model_selection();
        if app.stack_catalog.is_some() {
            app.status = format!(
                "{} models · Enter open · L load · ? help",
                app.sidebar_model_indices().len()
            );
        }
        if refresh_models_on_start {
            app.refresh_models_quietly();
        }
        Ok(app)
    }

    fn run(&mut self, terminal: &mut Tui) -> Result<()> {
        while !self.should_quit {
            self.drain_background_events();
            self.refresh_telemetry_if_due();
            if self
                .models_last_started
                .is_none_or(|started| started.elapsed() >= MODELS_REFRESH_INTERVAL)
            {
                self.refresh_models_quietly();
            }
            terminal.draw(|frame| self.draw(frame))?;
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == event::KeyEventKind::Press {
                        self.handle_key(key)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn refresh_local_inventories(&mut self) {
        let selected_bench = self.selected_script_path(ScriptEditorTarget::Bench);
        match collect_scripts_in(&self.root, ScriptMode::Bench) {
            Ok(scripts) => self.bench_scripts = scripts,
            Err(error) => self.push_log(format!("Bench inventory failed: {error:#}")),
        }
        self.maintenance_scripts = collect_shell_scripts(&self.root.join("maintenance"), "");
        self.restore_bench_selection(selected_bench.as_deref());
        clamp_list_selection(&mut self.maintenance_state, self.maintenance_scripts.len());
        self.sync_script_editor_selection(ScriptEditorTarget::Bench);
        self.sync_script_editor_selection(ScriptEditorTarget::Maintenance);
    }

    fn refresh_models(&mut self) {
        if self.loading_models {
            return;
        }
        self.status = "Refreshing llama-swap models…".into();
        self.models_refresh_announced = true;
        self.refresh_models_quietly();
    }

    /// Refresh router state without touching the status line.
    fn refresh_models_quietly(&mut self) {
        if self.loading_models {
            return;
        }
        self.loading_models = true;
        self.models_last_started = Some(Instant::now());
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = SwapClient::from_env()
                .and_then(|client| client.list_models())
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::Models(result));
        });
    }

    fn start_chat_connect(&mut self) {
        if self.chat_connection_pending {
            self.status = "A Chat connection probe is already running".into();
            return;
        }
        let endpoint = self.chat_endpoint_draft.trim().to_owned();
        if endpoint.is_empty() {
            self.status = "Enter a Chat endpoint before connecting".into();
            self.input_mode = InputMode::ChatEndpoint;
            return;
        }
        self.chat_connection_request_id = self.chat_connection_request_id.wrapping_add(1);
        let request_id = self.chat_connection_request_id;
        self.chat_connection_pending = true;
        self.status = format!("Connecting to {endpoint}…");
        let api_key = env::var("LLAMA_SWAP_API_KEY").ok();
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = ChatClient::new(&endpoint, api_key.as_deref()).and_then(|client| {
                client
                    .list_model_inventory()
                    .map(|inventory| (client.base_url().to_owned(), client, inventory))
            });
            match result {
                Ok((endpoint, client, inventory)) => {
                    let _ = sender.send(BackgroundEvent::ChatConnected {
                        request_id,
                        endpoint,
                        client,
                        inventory,
                    });
                }
                Err(error) => {
                    let _ = sender.send(BackgroundEvent::ChatConnectionFailed {
                        request_id,
                        operation: "connect".into(),
                        error: format!("{error:#}"),
                    });
                }
            }
        });
    }

    fn start_chat_detect(&mut self) {
        if self.chat_connection_pending {
            self.status = "A Chat connection probe is already running".into();
            return;
        }
        self.chat_connection_request_id = self.chat_connection_request_id.wrapping_add(1);
        let request_id = self.chat_connection_request_id;
        self.chat_connection_pending = true;
        self.status = "Detecting a local Chat server…".into();
        let api_key = env::var("LLAMA_SWAP_API_KEY").ok();
        let sender = self.sender.clone();
        thread::spawn(move || match detect_chat_server(api_key.as_deref()) {
            Ok((client, inventory)) => {
                let _ = sender.send(BackgroundEvent::ChatConnected {
                    request_id,
                    endpoint: client.base_url().to_owned(),
                    client,
                    inventory,
                });
            }
            Err(error) => {
                let _ = sender.send(BackgroundEvent::ChatConnectionFailed {
                    request_id,
                    operation: "detect".into(),
                    error: format!("{error:#}"),
                });
            }
        });
    }

    fn kill_detected_chat_server(&mut self) {
        let servers = detected_servers();
        let Some(server) = servers.first() else {
            self.status = "No externally-started llama-server was detected".into();
            return;
        };
        let pid = server.pid;
        match terminate_detected_server(pid) {
            Ok(()) => {
                self.status = format!("Termination requested for detected llama-server PID {pid}");
                self.push_log(format!("Stopped externally-started llama-server PID {pid}"));
                self.chat_client = None;
                self.chat_models.clear();
                self.chat_non_chat_model_ids.clear();
            }
            Err(error) => {
                self.status = format!("Could not stop detected server PID {pid}");
                self.push_log(format!("{error:#}"));
            }
        }
    }

    fn refresh_chat_models(&mut self) {
        let Some(client) = self.chat_client.clone() else {
            self.start_chat_connect();
            return;
        };
        if self.chat_connection_pending {
            self.status = "A Chat connection probe is already running".into();
            return;
        }
        self.chat_connection_request_id = self.chat_connection_request_id.wrapping_add(1);
        let request_id = self.chat_connection_request_id;
        self.chat_connection_pending = true;
        self.status = format!("Refreshing Chat models from {}…", client.base_url());
        let sender = self.sender.clone();
        thread::spawn(move || match client.list_model_inventory() {
            Ok(inventory) => {
                let _ = sender.send(BackgroundEvent::ChatConnected {
                    request_id,
                    endpoint: client.base_url().to_owned(),
                    client,
                    inventory,
                });
            }
            Err(error) => {
                let _ = sender.send(BackgroundEvent::ChatConnectionFailed {
                    request_id,
                    operation: "refresh".into(),
                    error: format!("{error:#}"),
                });
            }
        });
    }

    fn initialize_chat_model_selection(&mut self) {
        let selectable = self.chat_picker_model_indices();
        let selected = selectable
            .iter()
            .position(|&model_index| self.chat_models[model_index].state == "loaded")
            .or_else(|| (!selectable.is_empty()).then_some(0));
        self.chat_model_state.select(selected);
    }

    fn chat_picker_model_indices(&self) -> Vec<usize> {
        self.chat_models
            .iter()
            .enumerate()
            .filter(|(_, model)| {
                !self
                    .chat_non_chat_model_ids
                    .iter()
                    .any(|non_chat_id| non_chat_id == &model.id)
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn drain_background_events(&mut self) {
        for _ in 0..EVENTS_PER_TICK {
            let Ok(event) = self.receiver.try_recv() else {
                break;
            };
            match event {
                BackgroundEvent::Models(result) => {
                    self.loading_models = false;
                    match result {
                        Ok(models) => {
                            self.models = models;
                            // The router is authoritative: other clients load
                            // and unload models too.
                            self.loaded_model_id = self
                                .models
                                .iter()
                                .find(|model| model.state == "loaded")
                                .map(|model| model.id.clone());
                            if !self.selected_resident_once {
                                self.selected_resident_once = true;
                                self.select_resident_model();
                            }
                            if std::mem::take(&mut self.models_refresh_announced) {
                                self.status = format!("llama-swap: {} model(s)", self.models.len());
                            }
                            self.models_refresh_failing = false;
                        }
                        Err(error) => {
                            // Background polls report a failure once, not every tick.
                            let announced = std::mem::take(&mut self.models_refresh_announced);
                            if announced {
                                self.status = "llama-swap unavailable".into();
                            }
                            if announced || !self.models_refresh_failing {
                                self.push_log(format!("Model refresh failed: {error}"));
                            }
                            self.models_refresh_failing = true;
                        }
                    }
                }
                BackgroundEvent::ModelAction {
                    job_id,
                    model_id,
                    load,
                    result,
                } => {
                    self.model_action_pending = false;
                    match result {
                        Ok(message) => {
                            if load {
                                self.loaded_model_id = Some(model_id.clone());
                            } else if self.loaded_model_id.as_deref() == Some(model_id.as_str()) {
                                self.loaded_model_id = None;
                            }
                            self.finish_job(job_id, 0);
                            self.status = message.clone();
                            self.push_log(message);
                            self.refresh_models();
                        }
                        Err(error) => {
                            self.finish_job(job_id, 1);
                            self.status = format!(
                                "Could not {} {model_id}",
                                if load { "load" } else { "unload" }
                            );
                            self.push_log(error);
                        }
                    }
                }
                BackgroundEvent::BrowserScan(result) => {
                    self.browser_scanning = false;
                    match result {
                        Ok(files) => {
                            self.browser_files = files;
                            let visible_count = self.visible_browser_files().len();
                            clamp_table_selection(&mut self.browser_state, visible_count);
                            let warnings = self
                                .browser_files
                                .iter()
                                .filter(|file| file.parse_error.is_some())
                                .count();
                            let total_size: u64 =
                                self.browser_files.iter().map(|file| file.size).sum();
                            self.status = format!(
                                "Found {} GGUF file(s), {} total, {} warning(s)",
                                self.browser_files.len(),
                                format_bytes(total_size),
                                warnings
                            );
                        }
                        Err(error) => {
                            self.status = "GGUF scan failed".into();
                            self.push_log(error);
                        }
                    }
                }
                BackgroundEvent::ChatConnected {
                    request_id,
                    endpoint,
                    client,
                    inventory,
                } => {
                    if request_id != self.chat_connection_request_id {
                        continue;
                    }
                    self.chat_connection_pending = false;
                    self.chat_endpoint_draft = endpoint.clone();
                    self.chat_endpoint_committed = Some(endpoint.clone());
                    self.chat_client = Some(client);
                    self.chat_models = inventory.models;
                    self.chat_non_chat_model_ids = inventory.non_chat_model_ids;
                    self.initialize_chat_model_selection();
                    self.status = format!(
                        "Connected to {endpoint} · {} model(s)",
                        self.chat_picker_model_indices().len()
                    );
                    if let Some(index) = self.chat_handoff_pending.take() {
                        if self.stack_catalog_state.selected() == Some(index) {
                            self.open_sidebar_selection();
                        }
                    }
                }
                BackgroundEvent::ChatConnectionFailed {
                    request_id,
                    operation,
                    error,
                } => {
                    if request_id != self.chat_connection_request_id {
                        continue;
                    }
                    self.chat_connection_pending = false;
                    self.chat_handoff_pending = None;
                    self.status = format!("Chat {operation} failed");
                    self.push_log(error);
                }
                BackgroundEvent::ChatDelta { request_id, delta } => {
                    if self.chat_stream_request_id != request_id {
                        continue;
                    }
                    append_bounded_text(&mut self.chat_streaming, &delta, MAX_CHAT_PREVIEW_BYTES);
                }
                BackgroundEvent::ChatFinished { request_id, result } => {
                    if self.chat_stream_request_id != request_id {
                        continue;
                    }
                    self.chat_pending = false;
                    self.chat_stream_cancellation = None;
                    match result {
                        Ok(completion) => {
                            let status = match (
                                completion.completion_tokens,
                                completion.tokens_per_second(),
                            ) {
                                (Some(tokens), Some(rate)) => {
                                    format!("Chat response: {tokens} tokens · {rate:.1} tok/s")
                                }
                                _ => format!(
                                    "Chat response received in {:.1}s",
                                    completion.elapsed.as_secs_f64()
                                ),
                            };
                            self.chat_history.push("assistant", completion.content);
                            self.status = status;
                        }
                        Err(error) => {
                            self.status = "Chat request failed".into();
                            self.push_log(error);
                        }
                    }
                    self.chat_streaming.clear();
                }
                BackgroundEvent::ChatSessionsListed(result) => {
                    self.chat_session_pending = false;
                    match result {
                        Ok(list) => {
                            self.chat_sessions = list.sessions;
                            clamp_table_selection(
                                &mut self.chat_sessions_state,
                                self.chat_sessions.len(),
                            );
                            self.status =
                                format!("{} saved chat session(s)", self.chat_sessions.len());
                            if !list.issues.is_empty() {
                                self.push_log(format!(
                                    "Ignored {} malformed chat session(s)",
                                    list.issues.len()
                                ));
                            }
                            if list.truncated_entries > 0 {
                                self.push_log(format!(
                                    "Chat session list omitted {} older file(s)",
                                    list.truncated_entries
                                ));
                            }
                        }
                        Err(error) => {
                            self.status = "Could not list chat sessions".into();
                            self.push_log(error);
                        }
                    }
                }
                BackgroundEvent::ChatSessionLoaded(result) => {
                    self.chat_session_pending = false;
                    match result {
                        Ok(session) => {
                            let count = session.history.len();
                            let saved = session.saved.clone();
                            self.chat_history.replace_with_session(session);
                            self.chat_streaming.clear();
                            self.show_chat_sessions = false;
                            self.status = format!("Loaded chat {saved} · {count} message(s)");
                        }
                        Err(error) => {
                            self.status = "Could not load chat session".into();
                            self.push_log(error);
                        }
                    }
                }
                BackgroundEvent::ChatSessionSaved(result) => {
                    self.chat_session_pending = false;
                    match result {
                        Ok(saved) => {
                            let file_name = saved
                                .json_path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy();
                            self.status = format!("Saved chat session {file_name}");
                        }
                        Err(error) => {
                            self.status = "Could not save chat session".into();
                            self.push_log(error);
                        }
                    }
                }
                BackgroundEvent::ProcessLine { job_id, line } => {
                    let is_download = self
                        .job_history
                        .records()
                        .iter()
                        .find(|job| job.id == job_id)
                        .map(|job| job.kind == "download");
                    if let Some(is_download) = is_download {
                        if is_download {
                            self.update_download_progress(&line);
                            self.push_download_log(line.clone());
                        }
                        self.push_log(line);
                    }
                }
                BackgroundEvent::ProcessFinished { job_id, exit_code } => {
                    let is_download = self
                        .job_history
                        .records()
                        .iter()
                        .any(|job| job.id == job_id && job.kind == "download");
                    self.finish_job(job_id, exit_code);
                    if is_download {
                        self.push_download_log(format!("Download exited with code {exit_code}"));
                        self.refresh_download_disk_space();
                        self.download_started_at = None;
                    }
                    if self
                        .running_process
                        .as_ref()
                        .is_some_and(|running| running.job_id == job_id)
                    {
                        self.running_process = None;
                    }
                }
                BackgroundEvent::DownloadDiskSpace {
                    request_id,
                    target,
                    result,
                } => {
                    if request_id != self.download_disk_request_id {
                        continue;
                    }
                    self.download_disk_space = match result {
                        Ok(disk_space) => format_disk_space(&target, disk_space),
                        Err(error) => {
                            self.push_download_log(format!(
                                "Disk-space probe unavailable for {}: {error}",
                                target.display()
                            ));
                            format!("Disk: unavailable [{}]", target.display())
                        }
                    };
                }
                BackgroundEvent::DownloadPreflight { request_id, result } => {
                    let matches_pending = self
                        .download_preflight_pending
                        .as_ref()
                        .is_some_and(|pending| pending.request_id == request_id);
                    if !matches_pending {
                        continue;
                    }
                    let mut pending = self
                        .download_preflight_pending
                        .take()
                        .expect("matching download preflight is present");
                    if let Some(worker) = pending.worker.take() {
                        let _ = worker.join();
                    }
                    if pending.cancellation.load(Ordering::Acquire) {
                        self.status = "Download preflight cancelled".into();
                        self.push_download_log(format!(
                            "Download preflight cancelled for {}",
                            pending.name
                        ));
                        continue;
                    }

                    match result {
                        Ok(preflight) => {
                            self.record_download_preflight(&pending.target, &preflight);
                        }
                        Err(error) => {
                            self.push_download_log(format!(
                                "Download preflight unavailable for {}: {error}",
                                pending.name
                            ));
                        }
                    }
                    if self.running_process.is_some() {
                        self.status = format!(
                            "Did not start {}; another process started during preflight",
                            pending.name
                        );
                        self.push_download_log(self.status.clone());
                        continue;
                    }
                    self.status = format!("Preflight complete; starting {}", pending.name);
                    self.start_process_from_parts(pending.name, "download", pending.parts);
                }
                BackgroundEvent::Telemetry(result) => {
                    self.telemetry_pending = false;
                    match result {
                        Ok(snapshot) => {
                            self.telemetry = snapshot.resource_summary;
                            if let Some(gpu) = &snapshot.gpu {
                                self.gpu_util_history.push_back(gpu.gpu_util_percent as u64);
                                if self.gpu_util_history.len() > 60 {
                                    self.gpu_util_history.pop_front();
                                }
                                self.vram_history.push_back(gpu.used_vram_mib);
                                if self.vram_history.len() > 60 {
                                    self.vram_history.pop_front();
                                }
                            }
                            if let Some(io) = &snapshot.server_io {
                                self.last_server_read_bytes = Some(io.read_bytes);
                                self.last_server_read_time = Some(Instant::now());
                            }
                            self.gpu_telemetry = snapshot.gpu;
                            self.sys_memory = snapshot.memory;
                            self.server_io = snapshot.server_io;
                            if let (Some(previous), Some(current)) =
                                (self.host.cpu_jiffies, snapshot.host.cpu_jiffies)
                            {
                                self.cpu_percent = cpu_percent_between(previous, current);
                            }
                            self.host = snapshot.host;
                        }
                        Err(error) => {
                            self.telemetry = format!("Resources: unavailable ({error})");
                        }
                    }
                }
            }
        }
        #[cfg(target_os = "linux")]
        self.drain_direct_native_result();
    }

    /// Filesystems whose free space the status bar watches: home, the
    /// development volume and the model store.
    fn watched_disks(&self) -> Vec<String> {
        let mut paths = vec!["/home".to_owned(), "/mnt/dev".to_owned()];
        let models = expand_tilde(&self.download.config().base_models_dir);
        if !models.trim().is_empty() {
            paths.push(models);
        }
        paths
    }

    fn refresh_telemetry_if_due(&mut self) {
        if self.telemetry_pending
            || self
                .telemetry_last_started
                .is_some_and(|started| started.elapsed() < TELEMETRY_INTERVAL)
        {
            return;
        }

        let running_target = self.running_process.as_ref().map(|running| {
            let elapsed = self
                .job_history
                .records()
                .iter()
                .find(|job| job.id == running.job_id)
                .and_then(|job| job.elapsed_seconds())
                .unwrap_or_default();
            (running.process_group, elapsed)
        });
        let monitor_swap = self.loaded_model_id.is_some();
        let watched_disks = self.watched_disks();
        let last_server_read = self.last_server_read_bytes;
        let last_server_time = self.last_server_read_time;

        self.telemetry_pending = true;
        self.telemetry_last_started = Some(Instant::now());
        let sender = self.sender.clone();
        thread::spawn(move || {
            let res_summary = if let Some((process_group, elapsed)) = running_target {
                snapshot_process_group(process_group)
                    .map(|snapshot| snapshot.render("procs", Some(elapsed)))
                    .unwrap_or_else(|error| format!("procs: unavailable ({error:#})"))
            } else if monitor_swap {
                find_process_named("llama-swap")
                    .ok()
                    .flatten()
                    .and_then(|process| {
                        snapshot_descendants(process)
                            .map(|snapshot| snapshot.render("upstreams", None))
                            .ok()
                    })
                    .unwrap_or_else(|| "Resources: idle".into())
            } else {
                "Resources: idle".into()
            };

            let gpu = query_gpu_telemetry();
            let memory = query_system_memory();
            let server_io = find_process_named("llama-server")
                .ok()
                .flatten()
                .and_then(|pid| {
                    let elapsed = last_server_time.map(|t| t.elapsed().as_secs_f64());
                    query_server_io(pid, last_server_read, elapsed)
                });

            let snapshot = SystemTelemetrySnapshot {
                resource_summary: res_summary,
                gpu,
                memory,
                server_io,
                host: query_host_telemetry(&watched_disks),
            };
            let _ = sender.send(BackgroundEvent::Telemetry(Ok(snapshot)));
        });
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<()> {
        if self.load_armed.is_some() && !(key.code == KeyCode::Char('L') && self.sidebar_focused) {
            self.load_armed = None;
            self.status = "Load cancelled".into();
        }
        if !self.is_script_reload_shortcut(key) {
            self.script_reload_armed = None;
        }
        if !self.is_download_reload_trigger(key) {
            self.download_reload_armed = false;
        }
        if !self.is_download_restore_trigger(key) {
            self.download_restore_armed = false;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.request_quit();
            return Ok(());
        }
        if self.show_palette {
            return self.handle_palette_key(key);
        }
        if self.show_quit_confirmation {
            self.handle_quit_confirmation_key(key);
            return Ok(());
        }
        if key.code == KeyCode::Char('p') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.open_palette();
            return Ok(());
        }
        if self.handle_global_navigation_key(key) {
            return Ok(());
        }
        if self.show_help {
            self.show_help = false;
            return Ok(());
        }
        if self.show_chat_sessions {
            if key.code == KeyCode::Char('s') && key.modifiers.contains(KeyModifiers::ALT) {
                self.save_chat_session();
                return Ok(());
            }
            self.handle_chat_sessions_key(key);
            return Ok(());
        }
        if self.show_chat_prompts {
            self.handle_chat_prompts_key(key);
            return Ok(());
        }
        if self.script_versions_target.is_some() {
            self.handle_script_versions_key(key);
            return Ok(());
        }
        if self.script_input_target.is_some() {
            if let Some(command_id) = self.modified_key_command(key) {
                return self.execute_command(command_id);
            }
            self.handle_script_input_key(key);
            return Ok(());
        }
        if let Some(command_id) = self.modified_key_command(key) {
            return self.execute_command(command_id);
        }
        if !self.sidebar_focused
            && self.view == View::Download
            && self.download.focus() != DownloadFocus::Table
        {
            self.handle_download_input_key(key);
            return Ok(());
        }
        // Download uses Tab for its own fields; Esc on its table reaches the
        // sidebar instead.
        if matches!(key.code, KeyCode::Tab | KeyCode::BackTab)
            && matches!(self.input_mode, InputMode::Normal | InputMode::ChatMessage)
            && (self.sidebar_focused || self.view != View::Download)
        {
            self.toggle_focus();
            return Ok(());
        }
        if self.input_mode == InputMode::Normal
            && key.code == KeyCode::Char('?')
            && key.modifiers == KeyModifiers::NONE
        {
            self.show_help = true;
            return Ok(());
        }
        if self.input_mode != InputMode::Normal {
            return self.handle_input_key(key);
        }
        if self.sidebar_focused {
            return self.handle_sidebar_key(key);
        }
        match key.code {
            KeyCode::Char('q') => self.request_quit(),
            KeyCode::Char('?') => self.show_help = true,
            _ => self.handle_tab_key(key)?,
        }
        Ok(())
    }

    fn is_script_reload_shortcut(&self, key: KeyEvent) -> bool {
        key.code == KeyCode::Char('o')
            && key.modifiers.contains(KeyModifiers::ALT)
            && matches!(self.view, View::Bench | View::Maintenance)
    }

    fn is_download_reload_trigger(&self, key: KeyEvent) -> bool {
        self.view == View::Download
            && ((key.code == KeyCode::Char('o') && key.modifiers.contains(KeyModifiers::ALT))
                || (self.download.focus() == DownloadFocus::ConfigPath
                    && key.code == KeyCode::Enter
                    && key.modifiers == KeyModifiers::NONE))
    }

    fn is_download_restore_trigger(&self, key: KeyEvent) -> bool {
        self.view == View::Download
            && ((key.code == KeyCode::Char('r') && key.modifiers.contains(KeyModifiers::ALT))
                || (self.download.focus() == DownloadFocus::Versions
                    && key.code == KeyCode::Enter
                    && key.modifiers == KeyModifiers::NONE))
    }

    fn handle_global_navigation_key(&mut self, key: KeyEvent) -> bool {
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let destination = match key.code {
            KeyCode::F(number @ 1..=9) => Some(View::from_index(number as usize - 1)),
            KeyCode::Char(digit @ '1'..='9') if alt => {
                Some(View::from_index(digit as usize - '1' as usize))
            }
            KeyCode::Left if alt => Some(View::from_index(self.view.index() + View::ALL.len() - 1)),
            KeyCode::Right if alt => Some(View::from_index(self.view.index() + 1)),
            _ => None,
        };
        let Some(destination) = destination else {
            return false;
        };
        self.open_view(destination);
        true
    }

    /// Show `view` in the pane and move focus there.
    fn open_view(&mut self, view: View) {
        self.leave_tab_inputs();
        self.view = view;
        self.sidebar_focused = false;
        match view {
            View::Chat => {
                self.input_mode = InputMode::ChatMessage;
                if self.chat_client.is_none() && !self.chat_connection_pending {
                    self.start_chat_connect();
                }
            }
            View::Browser => self.ensure_browser_scan(),
            View::Model if self.stack_catalog.is_none() => self.refresh_stack_catalog(),
            _ => {}
        }
    }

    /// Tab: move focus between the sidebar and the pane. Leaving the pane
    /// drops a half-typed chat line's input mode but keeps its text.
    fn toggle_focus(&mut self) {
        if self.sidebar_focused {
            self.sidebar_focused = false;
            if self.view == View::Chat {
                self.input_mode = InputMode::ChatMessage;
            }
        } else {
            self.leave_tab_inputs();
            self.sidebar_focused = true;
        }
    }

    /// Sidebar group of a profile: its lane, or `engines` for llama-swap
    /// entries that proxy a non-OpenAI engine (ComfyUI, audio.cpp).
    fn sidebar_group(&self, profile: &CatalogProfile) -> (&'static str, u8, Style) {
        let engine = profile.key.lane == Lane::Language
            && self
                .yaml_entries
                .get(profile.key.profile_id.as_str())
                .is_some_and(|entry| entry.proxy);
        match profile.key.lane {
            Lane::Language if engine => ("engines", 3, theme::dim()),
            Lane::Language => ("language", 0, theme::lane_language()),
            Lane::Media => ("media", 1, theme::lane_media()),
            Lane::Decisions => ("decisions", 2, theme::lane_decisions()),
        }
    }

    /// Catalog indices of the models the sidebar shows, in display order.
    fn sidebar_model_indices(&self) -> Vec<usize> {
        let Some(catalog) = self.stack_catalog.as_ref() else {
            return Vec::new();
        };
        let filter = self.model_filter.trim().to_ascii_lowercase();
        let mut indices = catalog
            .list()
            .iter()
            .enumerate()
            .filter(|(_, profile)| {
                !(profile.key.lane == Lane::Language
                    && self
                        .unlisted_models
                        .contains(profile.key.profile_id.as_str()))
            })
            .filter(|(_, profile)| {
                filter.is_empty()
                    || profile
                        .key
                        .profile_id
                        .as_str()
                        .to_ascii_lowercase()
                        .contains(&filter)
                    || profile.name.to_ascii_lowercase().contains(&filter)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        indices.sort_by_key(|index| self.sidebar_group(&catalog.list()[*index]).1);
        indices
    }

    fn select_first_visible_model(&mut self) {
        let first = self.sidebar_model_indices().first().copied();
        self.stack_catalog_state.select(first);
    }

    /// Select the resident model when the sidebar shows it.
    fn select_resident_model(&mut self) {
        let Some(resident) = self.loaded_model_id.clone() else {
            return;
        };
        let Some(catalog) = self.stack_catalog.as_ref() else {
            return;
        };
        let profiles = catalog.list();
        if let Some(index) = self.sidebar_model_indices().into_iter().find(|index| {
            profiles[*index].key.lane == Lane::Language
                && profiles[*index].key.profile_id.as_str() == resident
        }) {
            self.stack_catalog_state.select(Some(index));
        }
    }

    /// Keep the selection on a row the sidebar shows.
    fn ensure_visible_model_selection(&mut self) {
        let visible = self.sidebar_model_indices();
        if !self
            .stack_catalog_state
            .selected()
            .is_some_and(|selected| visible.contains(&selected))
        {
            self.stack_catalog_state.select(visible.first().copied());
        }
    }

    fn move_sidebar_selection(&mut self, direction: isize) {
        self.load_armed = None;
        match self.sidebar_section {
            SidebarSection::Models => {
                let visible = self.sidebar_model_indices();
                if visible.is_empty() {
                    return;
                }
                let next = match self
                    .stack_catalog_state
                    .selected()
                    .and_then(|selected| visible.iter().position(|index| *index == selected))
                {
                    Some(position) => (position as isize + direction)
                        .clamp(0, visible.len() as isize - 1)
                        as usize,
                    None => 0,
                };
                self.stack_catalog_state.select(Some(visible[next]));
            }
            SidebarSection::Jobs => {
                let count = self.job_history.records().len();
                if direction < 0 {
                    select_previous_table(&mut self.jobs_state, count);
                } else {
                    select_next_table(&mut self.jobs_state, count);
                }
            }
        }
    }

    /// Enter in the sidebar. A model opens its default view: Chat for a
    /// language chat profile, the task form otherwise. It never loads a model;
    /// that is `L`. A job opens the Jobs view on that job.
    fn open_sidebar_selection(&mut self) {
        match self.sidebar_section {
            SidebarSection::Models => {
                if self.stack_catalog_state.selected().is_none() {
                    self.status = "No model selected".into();
                    return;
                }
                self.view = View::Model;
                let wants_chat = self.selected_stack_profile().is_some_and(|profile| {
                    profile.key.lane == Lane::Language
                        && profile.tasks.len() == 1
                        && profile.tasks.iter().any(|task| task.as_str() == "chat")
                });
                if wants_chat && self.chat_client.is_none() {
                    self.chat_handoff_pending = self.stack_catalog_state.selected();
                    if !self.chat_connection_pending {
                        // Lists models only (GET /v1/models); never loads one.
                        self.start_chat_connect();
                    }
                    self.status = "Connecting to llama-swap to open Chat…".into();
                    return;
                }
                self.begin_task_form();
                if self.view == View::Chat || self.input_mode == InputMode::TaskForm {
                    self.sidebar_focused = false;
                }
            }
            SidebarSection::Jobs => {
                if self.jobs_state.selected().is_none() {
                    self.status = "No job selected".into();
                    return;
                }
                let selected = self.jobs_state.selected();
                self.open_view(View::Jobs);
                self.jobs_state.select(selected);
            }
        }
    }

    fn handle_sidebar_key(&mut self, key: KeyEvent) -> Result<()> {
        let command = match key.code {
            KeyCode::Up | KeyCode::Char('k') => Some(CommandId::SidebarSelectPrevious),
            KeyCode::Down | KeyCode::Char('j') => Some(CommandId::SidebarSelectNext),
            KeyCode::Char('[') => Some(CommandId::SidebarPreviousSection),
            KeyCode::Char(']') => Some(CommandId::SidebarNextSection),
            KeyCode::Char(' ') => Some(CommandId::SidebarToggleSection),
            KeyCode::Enter | KeyCode::Char('f') => Some(CommandId::SidebarOpen),
            KeyCode::Char('/') => Some(CommandId::SidebarFilter),
            KeyCode::Char('r') => Some(CommandId::StackCatalogRefresh),
            KeyCode::Char('L') => Some(CommandId::ModelLoad),
            KeyCode::Char('U') => Some(CommandId::ModelUnload),
            KeyCode::Char('c') => Some(CommandId::StackCatalogCancel),
            KeyCode::Char('q') => Some(CommandId::Quit),
            KeyCode::Esc if !self.model_filter.is_empty() => {
                self.model_filter.clear();
                self.status = "Model filter cleared".into();
                None
            }
            KeyCode::Char(character) => toolbar::command_for_key(character),
            _ => None,
        };
        if let Some(command) = command {
            self.execute_command(command)?;
        }
        Ok(())
    }

    fn leave_tab_inputs(&mut self) {
        self.store_active_script_view();
        if self.input_mode == InputMode::TaskForm {
            self.task_form_editor = None;
        }
        self.input_mode = InputMode::Normal;
        self.script_input_target = None;
        self.script_versions_target = None;
        self.show_chat_sessions = false;
        self.show_chat_prompts = false;
        self.show_help = false;
    }

    fn request_quit(&mut self) {
        self.show_palette = false;
        if self.bench_editor.is_dirty()
            || self.maintenance_editor.is_dirty()
            || self.download.is_dirty()
        {
            self.show_quit_confirmation = true;
            self.status = "Unsaved edits: S save and quit · D discard and quit · Esc cancel".into();
        } else {
            self.should_quit = true;
        }
    }

    fn handle_quit_confirmation_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.show_quit_confirmation = false;
                self.status = "Quit cancelled; unsaved edits retained".into();
            }
            KeyCode::Char('s' | 'S') => {
                if self.bench_editor.is_dirty() {
                    self.save_script_editor(ScriptEditorTarget::Bench);
                }
                if self.maintenance_editor.is_dirty() {
                    self.save_script_editor(ScriptEditorTarget::Maintenance);
                }
                if self.download.is_dirty() {
                    self.save_download_config();
                }
                if !self.bench_editor.is_dirty()
                    && !self.maintenance_editor.is_dirty()
                    && !self.download.is_dirty()
                {
                    self.show_quit_confirmation = false;
                    self.should_quit = true;
                } else {
                    self.status = "Could not save every dirty editor; quit cancelled".into();
                }
            }
            KeyCode::Char('d' | 'D') => {
                self.show_quit_confirmation = false;
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn modified_key_command(&self, key: KeyEvent) -> Option<CommandId> {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match (self.view, key.code, control, alt) {
            (View::Bench, KeyCode::Char('r'), true, _) => Some(CommandId::BenchRun),
            (View::Bench, KeyCode::Char('s'), true, _) => Some(CommandId::BenchStop),
            (View::Bench, KeyCode::Char('f'), true, _) => Some(CommandId::BenchFocusFilter),
            (View::Bench, KeyCode::Char('j'), true, _) => Some(CommandId::BenchFocusList),
            (View::Bench, KeyCode::Char('u'), true, _) => Some(CommandId::BenchFocusEditor),
            (View::Bench, KeyCode::Char('p'), _, true) => Some(CommandId::BenchSaveScript),
            (View::Bench, KeyCode::Char('o'), _, true) => Some(CommandId::BenchReloadScript),
            (View::Bench, KeyCode::Char('v'), _, true) => Some(CommandId::BenchRestoreScript),
            (View::Chat, KeyCode::Char('g'), true, _) => Some(CommandId::ChatConnect),
            (View::Chat, KeyCode::Char('b'), true, _) => Some(CommandId::ChatDetect),
            (View::Chat, KeyCode::Char('x'), true, _) => Some(CommandId::ChatClear),
            (View::Chat, KeyCode::Char('s'), _, true) => Some(CommandId::ChatSave),
            (View::Browser, KeyCode::Char('r'), _, true) => Some(CommandId::BrowserScan),
            (View::Browser, KeyCode::Char('g'), _, true) => Some(CommandId::BrowserFocusPath),
            (View::Browser, KeyCode::Char('j'), _, true) => Some(CommandId::BrowserFocusTable),
            (View::Download, KeyCode::Char('d'), _, true) => Some(CommandId::DownloadSelected),
            (View::Download, KeyCode::Char('e'), _, true) => Some(CommandId::DownloadEnabled),
            (View::Download, KeyCode::Char('w'), _, true) => Some(CommandId::DownloadSaveConfig),
            (View::Download, KeyCode::Char('v'), _, true) => {
                Some(CommandId::DownloadValidateConfig)
            }
            (View::Download, KeyCode::Char('t'), _, true) => Some(CommandId::DownloadFocusTable),
            (View::Download, KeyCode::Char('i'), _, true) => Some(CommandId::DownloadFocusEditor),
            (View::Download, KeyCode::Char('o'), _, true) => Some(CommandId::DownloadLoadConfig),
            (View::Download, KeyCode::Char('r'), _, true) => Some(CommandId::DownloadRestoreConfig),
            (View::Download, KeyCode::Char('n'), _, true) => Some(CommandId::DownloadAddModel),
            (View::Download, KeyCode::Char('a'), _, true) => Some(CommandId::DownloadApplyEdit),
            (View::Download, KeyCode::Char('k'), _, true) => Some(CommandId::DownloadDeleteModel),
            (View::Download, KeyCode::Char('y'), _, true) => Some(CommandId::DownloadClearLog),
            (View::Maintenance, KeyCode::Char('r'), true, _) => Some(CommandId::MaintenanceRun),
            (View::Maintenance, KeyCode::Char('s'), true, _) => Some(CommandId::MaintenanceStop),
            (View::Maintenance, KeyCode::Char('u'), true, _) => {
                Some(CommandId::MaintenanceFocusEditor)
            }
            (View::Maintenance, KeyCode::Char('p'), _, true) => {
                Some(CommandId::MaintenanceSaveScript)
            }
            (View::Maintenance, KeyCode::Char('o'), _, true) => {
                Some(CommandId::MaintenanceReloadScript)
            }
            (View::Maintenance, KeyCode::Char('v'), _, true) => {
                Some(CommandId::MaintenanceRestoreScript)
            }
            (View::Maintenance, KeyCode::Char('l'), true, _) => {
                Some(CommandId::MaintenanceClearLog)
            }
            _ => None,
        }
    }

    fn open_palette(&mut self) {
        self.show_help = false;
        self.show_palette = true;
        self.palette_query.clear();
        self.palette_state.select(Some(0));
    }

    fn handle_palette_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => self.show_palette = false,
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.show_palette = false;
            }
            KeyCode::Up => {
                let count = self.palette_commands().len();
                select_previous_table(&mut self.palette_state, count)
            }
            KeyCode::Down => {
                let count = self.palette_commands().len();
                select_next_table(&mut self.palette_state, count)
            }
            KeyCode::Backspace => {
                self.palette_query.pop();
                self.palette_state.select(Some(0));
            }
            KeyCode::Enter => {
                let command_id = self
                    .palette_state
                    .selected()
                    .and_then(|index| self.palette_commands().get(index).map(|spec| spec.id));
                self.show_palette = false;
                if let Some(command_id) = command_id {
                    self.execute_command(command_id)?;
                }
            }
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.palette_query.push(character);
                self.palette_state.select(Some(0));
            }
            _ => {}
        }
        let count = self.palette_commands().len();
        clamp_table_selection(&mut self.palette_state, count);
        Ok(())
    }

    /// Every visible command matching the query. Commands for the current
    /// screen (and global ones) come first; the rest stay reachable by search.
    fn palette_commands(&self) -> Vec<&'static crate::commands::CommandSpec> {
        let context = self.command_context();
        let mut commands = search_all_commands(&self.palette_query);
        commands.sort_by_key(|spec| {
            !spec
                .contexts
                .iter()
                .any(|owner| *owner == CommandContext::Global || *owner == context)
        });
        commands
    }

    fn handle_chat_sessions_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('o') => self.show_chat_sessions = false,
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.chat_sessions_state, self.chat_sessions.len())
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.chat_sessions_state, self.chat_sessions.len())
            }
            KeyCode::Enter => self.load_selected_chat_session(),
            KeyCode::Char('r') => self.refresh_chat_sessions(),
            _ => {}
        }
    }

    fn handle_chat_prompts_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('l') => self.show_chat_prompts = false,
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_list(&mut self.chat_prompts_state, self.chat_prompts.len())
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_list(&mut self.chat_prompts_state, self.chat_prompts.len())
            }
            KeyCode::Enter => self.load_selected_chat_prompt(),
            _ => {}
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => {
                if self.view == View::Chat && self.cancel_chat_request() {
                    return Ok(());
                }
                if self.input_mode == InputMode::TaskForm {
                    self.task_form_editor = None;
                }
                self.input_mode = InputMode::Normal;
            }
            KeyCode::Enter => match self.input_mode {
                InputMode::TaskForm => {
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        || self
                            .task_form_editor
                            .as_ref()
                            .is_some_and(|editor| editor.active_field() == TaskFormField::Run)
                    {
                        self.run_task_form();
                    } else if let Some(editor) = self.task_form_editor.as_mut() {
                        if !editor.insert_newline() {
                            self.status = "Task form field or aggregate input limit reached".into();
                        }
                    }
                }
                InputMode::ChatMessage => self.send_chat_message(),
                InputMode::ChatEndpoint => self.start_chat_connect(),
                InputMode::BrowserPath => {
                    self.input_mode = InputMode::Normal;
                    self.scan_browser();
                }
                InputMode::BrowserFilter => self.input_mode = InputMode::Normal,
                InputMode::BenchFilter => self.input_mode = InputMode::Normal,
                InputMode::ChatSystemPrompt => self.input_mode = InputMode::Normal,
                _ => self.input_mode = InputMode::Normal,
            },
            KeyCode::Tab | KeyCode::BackTab if self.input_mode == InputMode::TaskForm => {
                if let Some(editor) = self.task_form_editor.as_mut() {
                    let direction = if key.code == KeyCode::BackTab
                        || key.modifiers.contains(KeyModifiers::SHIFT)
                    {
                        -1
                    } else {
                        1
                    };
                    editor.move_focus(direction);
                }
            }
            KeyCode::Up | KeyCode::Down if self.input_mode == InputMode::TaskForm => {
                if let Some(editor) = self.task_form_editor.as_mut() {
                    let direction = if key.code == KeyCode::Up { -1 } else { 1 };
                    if !editor.cycle_selector(direction) {
                        editor.move_focus(direction);
                    }
                }
            }
            KeyCode::Left | KeyCode::Right if self.input_mode == InputMode::TaskForm => {
                if let Some(editor) = self.task_form_editor.as_mut() {
                    editor.move_cursor(if key.code == KeyCode::Left { -1 } else { 1 });
                }
            }
            KeyCode::Home | KeyCode::End if self.input_mode == InputMode::TaskForm => {
                if let Some(editor) = self.task_form_editor.as_mut() {
                    editor.move_cursor_edge(key.code == KeyCode::End);
                }
            }
            KeyCode::Backspace => match self.input_mode {
                InputMode::ModelFilter => {
                    self.model_filter.pop();
                    self.select_first_visible_model();
                }
                InputMode::BenchFilter => {
                    let mut filter = self.bench_filter.clone();
                    filter.pop();
                    self.set_bench_filter(filter);
                }
                InputMode::BrowserPath => {
                    self.browser_path.pop();
                }
                InputMode::BrowserFilter => {
                    self.browser_filter.pop();
                    let count = self.visible_browser_files().len();
                    clamp_table_selection(&mut self.browser_state, count);
                }
                InputMode::ChatMessage => {
                    self.chat_input.pop();
                }
                InputMode::ChatSystemPrompt => {
                    self.chat_system_prompt.pop();
                }
                InputMode::ChatEndpoint => {
                    self.chat_endpoint_draft.pop();
                }
                InputMode::TaskForm => {
                    if let Some(editor) = self.task_form_editor.as_mut() {
                        editor.backspace();
                    }
                }
                InputMode::Normal => {}
            },
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                match self.input_mode {
                    InputMode::ModelFilter => {
                        self.model_filter.push(character);
                        self.select_first_visible_model();
                    }
                    InputMode::BenchFilter => {
                        let mut filter = self.bench_filter.clone();
                        filter.push(character);
                        self.set_bench_filter(filter);
                    }
                    InputMode::BrowserPath => self.browser_path.push(character),
                    InputMode::BrowserFilter => {
                        self.browser_filter.push(character);
                        let count = self.visible_browser_files().len();
                        clamp_table_selection(&mut self.browser_state, count);
                    }
                    InputMode::ChatMessage => self.chat_input.push(character),
                    InputMode::ChatSystemPrompt => self.chat_system_prompt.push(character),
                    InputMode::ChatEndpoint => self.chat_endpoint_draft.push(character),
                    InputMode::TaskForm => {
                        if let Some(editor) = self.task_form_editor.as_mut() {
                            if !editor.insert_char(character) && !character.is_control() {
                                self.status =
                                    "Task form field or aggregate input limit reached".into();
                            }
                        }
                    }
                    InputMode::Normal => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn script_editor(&self, target: ScriptEditorTarget) -> &ScriptEditorState {
        match target {
            ScriptEditorTarget::Bench => &self.bench_editor,
            ScriptEditorTarget::Maintenance => &self.maintenance_editor,
        }
    }

    fn script_editor_mut(&mut self, target: ScriptEditorTarget) -> &mut ScriptEditorState {
        match target {
            ScriptEditorTarget::Bench => &mut self.bench_editor,
            ScriptEditorTarget::Maintenance => &mut self.maintenance_editor,
        }
    }

    fn script_editor_view(&self, target: ScriptEditorTarget) -> EditorViewport {
        match target {
            ScriptEditorTarget::Bench => self.bench_editor_view,
            ScriptEditorTarget::Maintenance => self.maintenance_editor_view,
        }
    }

    fn script_editor_view_mut(&mut self, target: ScriptEditorTarget) -> &mut EditorViewport {
        match target {
            ScriptEditorTarget::Bench => &mut self.bench_editor_view,
            ScriptEditorTarget::Maintenance => &mut self.maintenance_editor_view,
        }
    }

    fn store_active_script_view(&mut self) {
        let Some(target) = self.script_input_target else {
            return;
        };
        let cursor_byte = self.script_buffer.cursor_byte();
        self.script_editor_view_mut(target).cursor_byte = cursor_byte;
    }

    fn visible_bench_scripts(&self) -> Vec<&PathBuf> {
        filter_bench_scripts(&self.bench_scripts, &self.root, &self.bench_filter)
    }

    fn restore_bench_selection(&mut self, preferred: Option<&Path>) {
        let visible = self.visible_bench_scripts();
        let selected = preferred
            .and_then(|path| {
                visible
                    .iter()
                    .position(|candidate| candidate.as_path() == path)
            })
            .or_else(|| (!visible.is_empty()).then_some(0));
        self.bench_state.select(selected);
    }

    fn set_bench_filter(&mut self, filter: String) {
        let selected = self.selected_script_path(ScriptEditorTarget::Bench);
        let visible = filter_bench_scripts(&self.bench_scripts, &self.root, &filter);
        let dirty_path = self.bench_editor.selected_path();
        if self.bench_editor.is_dirty()
            && dirty_path
                .is_some_and(|path| !visible.iter().any(|candidate| candidate.as_path() == path))
        {
            self.status = "Save or reload the edited bench script before filtering it out".into();
            return;
        }
        self.bench_filter = filter;
        self.restore_bench_selection(selected.as_deref());
        self.sync_script_editor_selection(ScriptEditorTarget::Bench);
    }

    fn selected_script_path(&self, target: ScriptEditorTarget) -> Option<PathBuf> {
        match target {
            ScriptEditorTarget::Bench => self.bench_state.selected().and_then(|index| {
                self.visible_bench_scripts()
                    .into_iter()
                    .nth(index)
                    .map(|path| path.as_path().to_path_buf())
            }),
            ScriptEditorTarget::Maintenance => self
                .maintenance_state
                .selected()
                .and_then(|index| self.maintenance_scripts.get(index))
                .cloned(),
        }
    }

    fn sync_script_editor_selection(&mut self, target: ScriptEditorTarget) -> bool {
        let Some(path) = self.selected_script_path(target) else {
            if !self.script_editor(target).is_dirty() {
                self.script_editor_mut(target).clear_selection();
            }
            return false;
        };
        if self.script_editor(target).selected_path() == Some(path.as_path()) {
            return true;
        }
        if self.script_editor(target).is_dirty() {
            self.status = format!(
                "{} editor has unsaved changes; save or reload before changing scripts",
                target.label()
            );
            return false;
        }
        match self.script_editor_mut(target).select(&path) {
            Ok(()) => {
                *self.script_editor_view_mut(target) = EditorViewport::default();
                true
            }
            Err(error) => {
                self.status = format!("Could not load {} script", target.label());
                self.push_log(format!("{error:#}"));
                false
            }
        }
    }

    fn move_script_selection(&mut self, target: ScriptEditorTarget, previous: bool) {
        if self.script_editor(target).is_dirty() {
            self.status = format!(
                "{} editor has unsaved changes; save or reload before changing scripts",
                target.label()
            );
            return;
        }
        let current = match target {
            ScriptEditorTarget::Bench => self.bench_state.selected(),
            ScriptEditorTarget::Maintenance => self.maintenance_state.selected(),
        };
        let scripts = match target {
            ScriptEditorTarget::Bench => {
                self.visible_bench_scripts().into_iter().cloned().collect()
            }
            ScriptEditorTarget::Maintenance => self.maintenance_scripts.clone(),
        };
        if scripts.is_empty() {
            return;
        }
        let next = if previous {
            match current {
                Some(0) | None => scripts.len() - 1,
                Some(index) => index - 1,
            }
        } else {
            current.map_or(0, |index| (index + 1) % scripts.len())
        };
        let path = scripts[next].clone();
        if let Err(error) = self.script_editor_mut(target).select(&path) {
            self.status = format!("Could not load {} script", target.label());
            self.push_log(format!("{error:#}"));
            return;
        }
        match target {
            ScriptEditorTarget::Bench => self.bench_state.select(Some(next)),
            ScriptEditorTarget::Maintenance => self.maintenance_state.select(Some(next)),
        }
        *self.script_editor_view_mut(target) = EditorViewport::default();
        self.script_reload_armed = None;
    }

    fn focus_script_editor(&mut self, target: ScriptEditorTarget) {
        if !self.sync_script_editor_selection(target) {
            self.status = format!("No {} script selected", target.label().to_ascii_lowercase());
            return;
        }
        let content = self.script_editor(target).content().to_owned();
        self.script_buffer.set_content(content);
        let viewport = self.script_editor_view(target);
        if !self.script_buffer.set_cursor_byte(viewport.cursor_byte) {
            let end = self.script_buffer.content().len();
            self.script_buffer.set_cursor_byte(end);
        }
        self.script_input_target = Some(target);
        self.script_reload_armed = None;
        self.input_mode = InputMode::Normal;
        self.status = format!("Editing {} script", target.label().to_ascii_lowercase());
    }

    fn leave_script_editor(&mut self) {
        let Some(target) = self.script_input_target else {
            return;
        };
        self.sync_script_buffer(target);
        self.store_active_script_view();
        self.script_input_target = None;
        self.script_reload_armed = None;
        self.status = if self.script_editor(target).is_dirty() {
            format!("{} editor retains unsaved changes", target.label())
        } else {
            format!(
                "{} editor focus returned to the script list",
                target.label()
            )
        };
    }

    fn sync_script_buffer(&mut self, target: ScriptEditorTarget) -> bool {
        let content = self.script_buffer.content().to_owned();
        match self.script_editor_mut(target).set_content(content) {
            Ok(()) => true,
            Err(error) => {
                self.status = format!("Could not update {} editor", target.label());
                self.push_log(format!("{error:#}"));
                false
            }
        }
    }

    fn handle_script_input_key(&mut self, key: KeyEvent) {
        let Some(target) = self.script_input_target else {
            return;
        };
        let would_mutate = matches!(
            key.code,
            KeyCode::Enter | KeyCode::Tab | KeyCode::Backspace | KeyCode::Delete
        ) || matches!(key.code, KeyCode::Char(_))
            && !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        if would_mutate && !self.script_editor(target).content_synchronized() {
            self.status = format!(
                "{} editor is out of sync; press Alt+O to reload before editing",
                target.label()
            );
            return;
        }
        let mut changed = false;
        match key.code {
            KeyCode::Esc => {
                self.leave_script_editor();
                return;
            }
            KeyCode::Enter => {
                self.script_buffer.insert_newline();
                changed = true;
            }
            KeyCode::Tab => {
                for _ in 0..4 {
                    self.script_buffer.insert_char(' ');
                }
                changed = true;
            }
            KeyCode::Backspace => changed = self.script_buffer.backspace(),
            KeyCode::Delete => changed = self.script_buffer.delete_forward(),
            KeyCode::Left => {
                self.script_buffer.move_left();
            }
            KeyCode::Right => {
                self.script_buffer.move_right();
            }
            KeyCode::Up => {
                self.script_buffer.move_up();
            }
            KeyCode::Down => {
                self.script_buffer.move_down();
            }
            KeyCode::Home => {
                self.script_buffer.move_home();
            }
            KeyCode::End => {
                self.script_buffer.move_end();
            }
            KeyCode::PageUp => {
                self.script_buffer.move_page_up(10);
            }
            KeyCode::PageDown => {
                self.script_buffer.move_page_down(10);
            }
            KeyCode::Char(character)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.script_buffer.insert_char(character);
                changed = true;
            }
            _ => {}
        }
        if changed {
            self.script_reload_armed = None;
            self.sync_script_buffer(target);
        }
        self.store_active_script_view();
    }

    fn save_script_editor(&mut self, target: ScriptEditorTarget) {
        if !self.sync_script_editor_selection(target) {
            return;
        }
        if self.script_input_target == Some(target) && !self.sync_script_buffer(target) {
            return;
        }
        let saved_path = self
            .script_editor(target)
            .selected_path()
            .map(PathBuf::from);
        match self.script_editor_mut(target).save("manual-save") {
            Ok(outcome) => {
                self.script_reload_armed = None;
                let versions = self.script_editor(target).versions().len();
                if let Some(warning) = outcome.warning_message() {
                    self.status = format!(
                        "Saved {} script; snapshot refresh needs attention",
                        target.label().to_ascii_lowercase()
                    );
                    self.push_log(warning.to_owned());
                } else {
                    self.status = format!(
                        "Saved {} script with snapshot · {versions} version(s)",
                        target.label().to_ascii_lowercase()
                    );
                }
                if let Some(path) = saved_path {
                    match lint_shell_script(&path) {
                        Ok(report) if report.available && !report.diagnostics.is_empty() => {
                            self.push_log(format!("{}: {}", path.display(), report.summary()));
                            for diagnostic in report.diagnostics.iter().take(8) {
                                self.push_log(format!(
                                    "  {}:{}:{} SC{} {:?}: {}",
                                    path.display(),
                                    diagnostic.line,
                                    diagnostic.column,
                                    diagnostic.code,
                                    diagnostic.level,
                                    diagnostic.message
                                ));
                            }
                        }
                        Ok(_) => {}
                        Err(error) => self.push_log(format!(
                            "shellcheck could not inspect {}: {error:#}",
                            path.display()
                        )),
                    }
                }
            }
            Err(error) => {
                self.status = format!("Could not save {} script", target.label());
                self.push_log(format!("{error:#}"));
            }
        }
    }

    fn reload_script_editor(&mut self, target: ScriptEditorTarget) {
        if !self.sync_script_editor_selection(target) {
            return;
        }
        if self.script_editor(target).is_dirty()
            && self.script_editor(target).content_synchronized()
            && self.script_reload_armed != Some(target)
        {
            self.script_reload_armed = Some(target);
            self.status = format!(
                "{} editor has unsaved changes; press Alt+O again to discard and reload",
                target.label()
            );
            return;
        }
        match self.script_editor_mut(target).reload() {
            Ok(()) => {
                self.script_reload_armed = None;
                *self.script_editor_view_mut(target) = EditorViewport::default();
                if self.script_input_target == Some(target) {
                    let content = self.script_editor(target).content().to_owned();
                    self.script_buffer.set_content(content);
                }
                self.status = format!(
                    "Reloaded {} script from disk",
                    target.label().to_ascii_lowercase()
                );
            }
            Err(error) => {
                self.status = format!("Could not reload {} script", target.label());
                self.push_log(format!("{error:#}"));
            }
        }
    }

    fn open_script_versions(&mut self, target: ScriptEditorTarget) {
        if !self.sync_script_editor_selection(target) {
            return;
        }
        if self.script_editor(target).is_dirty() {
            self.status = format!(
                "Save {} changes or press Alt+O twice to discard before restoring a version",
                target.label().to_ascii_lowercase()
            );
            return;
        }
        if let Err(error) = self.script_editor_mut(target).refresh_versions() {
            self.status = format!("Could not list {} script versions", target.label());
            self.push_log(format!("{error:#}"));
            return;
        }
        let count = self.script_editor(target).versions().len();
        if count == 0 {
            self.status = format!(
                "No snapshots exist for this {} script",
                target.label().to_ascii_lowercase()
            );
            return;
        }
        self.script_versions_target = Some(target);
        self.script_version_state.select(Some(0));
        self.status = format!(
            "Select one of {count} {} script snapshot(s)",
            target.label().to_ascii_lowercase()
        );
    }

    fn handle_script_versions_key(&mut self, key: KeyEvent) {
        let Some(target) = self.script_versions_target else {
            return;
        };
        let count = self.script_editor(target).versions().len();
        match key.code {
            KeyCode::Esc => self.script_versions_target = None,
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_list(&mut self.script_version_state, count)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_list(&mut self.script_version_state, count)
            }
            KeyCode::Enter => self.restore_selected_script_version(target),
            KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.script_versions_target = None
            }
            _ => {}
        }
    }

    fn restore_selected_script_version(&mut self, target: ScriptEditorTarget) {
        let Some(index) = self.script_version_state.selected() else {
            return;
        };
        let Some(version) = self.script_editor(target).versions().get(index).cloned() else {
            return;
        };
        match self.script_editor_mut(target).restore(&version) {
            Ok(outcome) => {
                *self.script_editor_view_mut(target) = EditorViewport::default();
                if outcome.content_synchronized() && self.script_input_target == Some(target) {
                    let content = self.script_editor(target).content().to_owned();
                    self.script_buffer.set_content(content);
                }
                self.script_versions_target = None;
                self.script_reload_armed = None;
                if let Some(warning) = outcome.warning_message() {
                    self.status = format!(
                        "Restored {} script from {version}; refresh needs attention",
                        target.label().to_ascii_lowercase()
                    );
                    self.push_log(warning.to_owned());
                } else {
                    self.status = format!(
                        "Restored {} script from {version}",
                        target.label().to_ascii_lowercase()
                    );
                }
            }
            Err(error) => {
                self.status = format!("Could not restore {} script", target.label());
                self.push_log(format!("{error:#}"));
            }
        }
    }

    fn handle_tab_key(&mut self, key: KeyEvent) -> Result<()> {
        match self.view {
            View::Model => self.handle_stack_catalog_key(key),
            View::Chat => self.handle_chat_key(key),
            View::Jobs => self.handle_jobs_key(key),
            View::Bench => self.handle_bench_key(key),
            View::Download => self.handle_download_key(key)?,
            View::Browser => self.handle_browser_key(key),
            View::Maintenance => self.handle_maintenance_key(key),
            View::System => {}
            View::Log => {
                if key.code == KeyCode::Char('c') {
                    self.execute_command(CommandId::ClearLog)?;
                }
            }
        }
        Ok(())
    }

    fn handle_bench_key(&mut self, key: KeyEvent) {
        let command = match key.code {
            KeyCode::Up | KeyCode::Char('k') => Some(CommandId::BenchSelectPrevious),
            KeyCode::Down | KeyCode::Char('j') => Some(CommandId::BenchSelectNext),
            KeyCode::Char('/') => Some(CommandId::BenchFocusFilter),
            KeyCode::Enter | KeyCode::Char('r') => Some(CommandId::BenchRun),
            KeyCode::Char('s') => Some(CommandId::BenchStop),
            _ => None,
        };
        if let Some(command) = command {
            if let Err(error) = self.execute_command(command) {
                self.status = format!("Bench: {error:#}");
            }
        }
    }

    fn command_context(&self) -> CommandContext {
        match self.view {
            View::Model => CommandContext::StackCatalog,
            View::Chat => CommandContext::Chat,
            View::Jobs => CommandContext::Jobs,
            View::Bench => CommandContext::Bench,
            View::Download => CommandContext::Download,
            View::Browser => CommandContext::Browser,
            View::Maintenance => CommandContext::Maintenance,
            View::System => CommandContext::System,
            View::Log => CommandContext::Log,
        }
    }

    /// Whether a command can run now: global and sidebar commands always,
    /// Model-view commands while the sidebar is focused, and everything else
    /// in its own view.
    fn command_available(&self, spec: &crate::commands::CommandSpec) -> bool {
        spec.is_available_in(self.command_context())
            || spec.contexts.contains(&CommandContext::Sidebar)
            || (self.sidebar_focused && spec.contexts.contains(&CommandContext::StackCatalog))
    }

    fn execute_command(&mut self, command_id: CommandId) -> Result<()> {
        use CommandId::*;

        if let Some(spec) = command(command_id) {
            if !spec.is_visible() {
                self.status = format!("{} is not built yet (workbench plan step 2)", spec.label);
                return Ok(());
            }
            if !self.command_available(spec) {
                self.status = format!(
                    "{} is only available in its {} view",
                    spec.palette_label,
                    spec.contexts
                        .first()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "declared".into())
                );
                return Ok(());
            }
        }

        match command_id {
            Quit => self.request_quit(),
            ShowHelp => self.show_help = true,
            ShowPalette => self.open_palette(),
            OpenStackCatalog => self.open_view(View::Model),
            OpenChat => self.open_view(View::Chat),
            OpenJobs => self.open_view(View::Jobs),
            OpenBench => self.open_view(View::Bench),
            OpenDownload => self.open_view(View::Download),
            OpenBrowser => self.open_view(View::Browser),
            OpenMaintenance => self.open_view(View::Maintenance),
            OpenSystem => self.open_view(View::System),
            OpenLog => self.open_view(View::Log),
            PreviousTab => {
                let count = View::ALL.len();
                self.open_view(View::from_index(self.view.index() + count - 1));
            }
            NextTab => self.open_view(View::from_index(self.view.index() + 1)),
            ToggleFocus => self.toggle_focus(),
            ReloadRouter | Tune => {}
            SidebarSelectPrevious => self.move_sidebar_selection(-1),
            SidebarSelectNext => self.move_sidebar_selection(1),
            SidebarPreviousSection | SidebarNextSection => {
                self.sidebar_section = self.sidebar_section.other();
            }
            SidebarToggleSection => match self.sidebar_section {
                SidebarSection::Models => self.models_collapsed = !self.models_collapsed,
                SidebarSection::Jobs => self.jobs_collapsed = !self.jobs_collapsed,
            },
            SidebarOpen => self.open_sidebar_selection(),
            SidebarFilter => {
                self.sidebar_section = SidebarSection::Models;
                self.models_collapsed = false;
                self.sidebar_focused = true;
                self.input_mode = InputMode::ModelFilter;
            }
            StackCatalogRefresh => {
                // Router state in the sidebar comes from the model list.
                self.refresh_models();
                self.refresh_stack_catalog();
                self.refresh_bench_rates();
                self.ensure_visible_model_selection();
                if self.stack_catalog.is_some() {
                    self.status =
                        format!("Refreshed · {} models", self.sidebar_model_indices().len());
                }
            }
            ModelLoad => self.model_action(true),
            ModelUnload => self.model_action(false),
            StackCatalogForm => self.begin_task_form(),
            StackCatalogCancel => {
                #[cfg(target_os = "linux")]
                self.begin_direct_native_cancel();
            }
            BenchSelectPrevious => self.move_script_selection(ScriptEditorTarget::Bench, true),
            BenchSelectNext => self.move_script_selection(ScriptEditorTarget::Bench, false),
            BenchRun => self.run_selected_bench(),
            BenchStop => self.stop_running_process(),
            BenchFocusFilter => self.input_mode = InputMode::BenchFilter,
            BenchFocusList => {
                if self.script_input_target == Some(ScriptEditorTarget::Bench) {
                    self.leave_script_editor();
                }
                self.input_mode = InputMode::Normal;
            }
            BenchFocusEditor => {
                if self.script_input_target == Some(ScriptEditorTarget::Bench) {
                    self.leave_script_editor();
                } else {
                    self.focus_script_editor(ScriptEditorTarget::Bench);
                }
            }
            BenchSaveScript => self.save_script_editor(ScriptEditorTarget::Bench),
            BenchReloadScript => self.reload_script_editor(ScriptEditorTarget::Bench),
            BenchRestoreScript => self.open_script_versions(ScriptEditorTarget::Bench),
            ClearLog => self.clear_activity_log("Log"),
            ChatCompose => self.input_mode = InputMode::ChatMessage,
            ChatSend => {
                if self.chat_input.trim().is_empty() {
                    self.input_mode = InputMode::ChatMessage;
                } else {
                    self.send_chat_message();
                }
            }
            ChatRefreshModels => self.refresh_chat_models(),
            ChatConnect => self.start_chat_connect(),
            ChatDetect => self.start_chat_detect(),
            ChatKillDetected => self.kill_detected_chat_server(),
            ChatPrompts => self.open_chat_prompts(),
            ChatClear => {
                self.cancel_chat_request();
                self.chat_history.clear();
                self.chat_streaming.clear();
                self.status = "Chat cleared".into();
            }
            ChatSave => self.save_chat_session(),
            ChatSessions => self.open_chat_sessions(),
            ChatEditSystemPrompt => self.input_mode = InputMode::ChatSystemPrompt,
            ChatToggleThinking => self.chat_thinking = !self.chat_thinking,
            ChatDecreaseTemperature => {
                self.chat_temperature = (self.chat_temperature - 0.1).max(0.0)
            }
            ChatIncreaseTemperature => {
                self.chat_temperature = (self.chat_temperature + 0.1).min(10.0)
            }
            ChatDecreaseMaxTokens => self.chat_max_tokens = (self.chat_max_tokens / 2).max(128),
            ChatIncreaseMaxTokens => {
                self.chat_max_tokens = self.chat_max_tokens.saturating_mul(2).min(65_536)
            }
            BrowserSelectPrevious => {
                let count = self.visible_browser_files().len();
                select_previous_table(&mut self.browser_state, count)
            }
            BrowserSelectNext => {
                let count = self.visible_browser_files().len();
                select_next_table(&mut self.browser_state, count)
            }
            BrowserScan => self.scan_browser(),
            BrowserFocusPath => self.input_mode = InputMode::BrowserPath,
            BrowserFocusTable => self.input_mode = InputMode::Normal,
            BrowserFocusFilter => self.input_mode = InputMode::BrowserFilter,
            BrowserChangeSort => {
                self.browser_sort = self.browser_sort.next();
                self.browser_state.select(Some(0));
            }
            BrowserToggleRecursive => {
                self.browser_recursive = !self.browser_recursive;
                self.status = format!(
                    "GGUF scan mode: {}",
                    if self.browser_recursive {
                        "recursive"
                    } else {
                        "top-level"
                    }
                );
            }
            DownloadSelectPrevious => self.select_download_previous(),
            DownloadSelectNext => self.select_download_next(),
            DownloadFocusTable => self.focus_download_table(),
            DownloadFocusEditor => self.focus_download_editor(),
            DownloadToggleEnabled => self.toggle_download_enabled(),
            DownloadLoadConfig => self.reload_download_config(),
            DownloadSaveConfig => self.save_download_config(),
            DownloadValidateConfig => self.validate_download_config(),
            DownloadRestoreConfig => self.restore_download_config(),
            DownloadAddModel => self.add_download_model(),
            DownloadApplyEdit => self.apply_download_edit(),
            DownloadDeleteModel => self.delete_download_model(),
            DownloadSelected => self.download_selected(),
            DownloadEnabled => self.download_enabled(),
            DownloadClearLog => self.clear_download_log(),
            JobsSelectPrevious => {
                select_previous_table(&mut self.jobs_state, self.job_history.records().len())
            }
            JobsSelectNext => {
                select_next_table(&mut self.jobs_state, self.job_history.records().len())
            }
            JobsStop => self.stop_active_job(),
            JobsRetry => self.retry_selected_job(),
            JobsClear => {
                if self.can_clear_job_history() {
                    self.clear_job_history();
                } else {
                    self.status = "Cannot clear job history while an operation is running".into();
                }
            }
            MaintenanceSelectPrevious => {
                self.move_script_selection(ScriptEditorTarget::Maintenance, true)
            }
            MaintenanceSelectNext => {
                self.move_script_selection(ScriptEditorTarget::Maintenance, false)
            }
            MaintenanceRun => self.run_selected_maintenance(),
            MaintenanceStop => self.stop_running_process(),
            MaintenanceFocusEditor => {
                if self.script_input_target == Some(ScriptEditorTarget::Maintenance) {
                    self.leave_script_editor();
                } else {
                    self.focus_script_editor(ScriptEditorTarget::Maintenance)
                }
            }
            MaintenanceSaveScript => self.save_script_editor(ScriptEditorTarget::Maintenance),
            MaintenanceReloadScript => self.reload_script_editor(ScriptEditorTarget::Maintenance),
            MaintenanceRestoreScript => self.open_script_versions(ScriptEditorTarget::Maintenance),
            MaintenanceClearLog => self.clear_activity_log("Maintenance"),
            OpenParamBuilder
            | ParamBuilderToggleFocus
            | ParamBuilderNextParam
            | ParamBuilderPrevParam
            | ParamBuilderCycleValue
            | ParamBuilderSaveToYaml
            | ParamBuilderImportEntry
            | ParamBuilderOpenPresets
            | ParamBuilderClearAll => {}
        }
        Ok(())
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('i') | KeyCode::Enter => self.input_mode = InputMode::ChatMessage,
            KeyCode::Esc => {
                if !self.cancel_chat_request() {
                    self.input_mode = InputMode::Normal;
                }
            }
            KeyCode::Char('x') => {
                self.cancel_chat_request();
                self.chat_history.clear();
                self.chat_streaming.clear();
                self.status = "Chat cleared".into();
            }
            KeyCode::Char('r') => self.refresh_chat_models(),
            KeyCode::Char('e') => self.input_mode = InputMode::ChatEndpoint,
            KeyCode::Up | KeyCode::Char('k') => self.select_previous_chat_model(),
            KeyCode::Down | KeyCode::Char('j') => self.select_next_chat_model(),
            KeyCode::Char('o') => self.open_chat_sessions(),
            KeyCode::Char('p') => self.input_mode = InputMode::ChatSystemPrompt,
            KeyCode::Char('l') => self.open_chat_prompts(),
            KeyCode::Char('K') => self.kill_detected_chat_server(),
            KeyCode::Char('t') => {
                self.chat_thinking = !self.chat_thinking;
                self.status = format!(
                    "Thinking prefix {}",
                    if self.chat_thinking {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
            }
            KeyCode::Char('[') => {
                self.chat_temperature = (self.chat_temperature - 0.1).max(0.0);
            }
            KeyCode::Char(']') => {
                self.chat_temperature = (self.chat_temperature + 0.1).min(10.0);
            }
            KeyCode::Char('-') => {
                self.chat_max_tokens = (self.chat_max_tokens / 2).max(128);
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.chat_max_tokens = self.chat_max_tokens.saturating_mul(2).min(65_536);
            }
            _ => {}
        }
    }

    fn handle_browser_key(&mut self, key: KeyEvent) {
        let visible_count = self.visible_browser_files().len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.browser_state, visible_count)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.browser_state, visible_count)
            }
            KeyCode::Char('g') => self.input_mode = InputMode::BrowserPath,
            KeyCode::Char('/') => self.input_mode = InputMode::BrowserFilter,
            KeyCode::Char('c') => {
                self.browser_sort = self.browser_sort.next();
                self.browser_state.select(Some(0));
            }
            KeyCode::Char('t') => {
                self.browser_recursive = !self.browser_recursive;
                self.status = format!(
                    "GGUF scan mode: {}",
                    if self.browser_recursive {
                        "recursive"
                    } else {
                        "top-level"
                    }
                );
            }
            KeyCode::Char('r') | KeyCode::Enter => self.scan_browser(),
            _ => {}
        }
    }

    fn handle_download_key(&mut self, key: KeyEvent) -> Result<()> {
        if key.code == KeyCode::Esc && self.cancel_download_preflight() {
            return Ok(());
        }
        if key.code == KeyCode::Esc {
            self.toggle_focus();
            return Ok(());
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.select_download_previous(),
            KeyCode::Down | KeyCode::Char('j') => self.select_download_next(),
            KeyCode::PageUp | KeyCode::Char('[') => self.previous_download_page(),
            KeyCode::PageDown | KeyCode::Char(']') => self.next_download_page(),
            KeyCode::Tab => {
                self.download.focus_next();
                self.describe_download_focus();
            }
            KeyCode::BackTab => {
                self.download.focus_previous();
                self.describe_download_focus();
            }
            KeyCode::Enter => self.focus_download_editor(),
            KeyCode::Char(' ') => self.toggle_download_enabled(),
            KeyCode::Char('v') => self.validate_download_config(),
            KeyCode::Char('w') => self.save_download_config(),
            KeyCode::Char('d') => self.download_selected(),
            KeyCode::Char('e') => self.download_enabled(),
            _ => {}
        }
        Ok(())
    }

    fn handle_download_input_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Esc && self.cancel_download_preflight() {
            return;
        }
        match key.code {
            KeyCode::Esc => {
                self.focus_download_table();
                return;
            }
            KeyCode::Tab => {
                self.download.focus_next();
                self.describe_download_focus();
                return;
            }
            KeyCode::BackTab => {
                self.download.focus_previous();
                self.describe_download_focus();
                return;
            }
            _ => {}
        }

        match self.download.focus() {
            DownloadFocus::Table => {}
            DownloadFocus::ConfigPath => {
                if key.code == KeyCode::Enter {
                    self.reload_download_config();
                } else if let Some(buffer) = self.download.focused_buffer_mut() {
                    edit_single_line_buffer(buffer, key);
                }
            }
            DownloadFocus::Versions => match key.code {
                KeyCode::Up | KeyCode::Left | KeyCode::Char('k') => {
                    self.download.select_previous_version();
                    self.describe_download_focus();
                }
                KeyCode::Down | KeyCode::Right | KeyCode::Char('j') => {
                    self.download.select_next_version();
                    self.describe_download_focus();
                }
                KeyCode::Enter => self.restore_download_config(),
                _ => {}
            },
            DownloadFocus::SlowPreset => match key.code {
                KeyCode::Char(' ') | KeyCode::Enter => {
                    let enabled = self.download.toggle_slow();
                    self.status = format!(
                        "Slow download preset {}",
                        if enabled { "enabled" } else { "disabled" }
                    );
                }
                _ => {}
            },
            DownloadFocus::Model(field) if field.is_boolean() => match key.code {
                KeyCode::Char(' ') | KeyCode::Enter => {
                    if !self.download_config_is_usable() {
                        return;
                    }
                    match self.download.draft_mut().toggle_boolean(field) {
                        Ok(value) => {
                            self.status = format!(
                                "{} {} in the model draft",
                                field.label(),
                                if value { "enabled" } else { "disabled" }
                            )
                        }
                        Err(error) => self.record_download_message(format!("{error:#}")),
                    }
                }
                _ => {}
            },
            DownloadFocus::BaseModelsDir | DownloadFocus::Model(_) => {
                if !self.download_config_is_usable() {
                    return;
                }
                if key.code == KeyCode::Enter {
                    self.download.focus_next();
                    self.describe_download_focus();
                } else if let Some(buffer) = self.download.focused_buffer_mut() {
                    edit_single_line_buffer(buffer, key);
                }
            }
            DownloadFocus::GlobalWorkers | DownloadFocus::SaveNote => {
                if key.code == KeyCode::Enter {
                    self.download.focus_next();
                    self.describe_download_focus();
                } else if let Some(buffer) = self.download.focused_buffer_mut() {
                    edit_single_line_buffer(buffer, key);
                }
            }
        }
    }

    fn sync_download_table_state(&mut self) {
        let Some(index) = self.download.selected_index() else {
            self.download_page = 0;
            self.download_state.select(None);
            return;
        };
        self.download_page =
            (index / DOWNLOAD_PAGE_SIZE).min(self.download_page_count().saturating_sub(1));
        self.download_state.select(Some(
            index.saturating_sub(self.download_page * DOWNLOAD_PAGE_SIZE),
        ));
    }

    fn select_download_previous(&mut self) {
        self.download.select_previous();
        self.sync_download_table_state();
        self.describe_download_selection();
    }

    fn select_download_next(&mut self) {
        self.download.select_next();
        self.sync_download_table_state();
        self.describe_download_selection();
    }

    fn download_page_count(&self) -> usize {
        self.download
            .models()
            .len()
            .div_ceil(DOWNLOAD_PAGE_SIZE)
            .max(1)
    }

    fn previous_download_page(&mut self) {
        self.download_page = self.download_page.saturating_sub(1);
        let index = self.download_page * DOWNLOAD_PAGE_SIZE;
        self.download.select(index);
        self.sync_download_table_state();
    }

    fn next_download_page(&mut self) {
        self.download_page =
            (self.download_page + 1).min(self.download_page_count().saturating_sub(1));
        let index = self.download_page * DOWNLOAD_PAGE_SIZE;
        self.download.select(index);
        self.sync_download_table_state();
    }

    fn describe_download_selection(&mut self) {
        self.status = self.download.selected_model().map_or_else(
            || "No download model selected".into(),
            |model| format!("Selected download model {}", value_or_dash(&model.repo_id)),
        );
    }

    fn focus_download_table(&mut self) {
        self.download.set_focus(DownloadFocus::Table);
        self.sync_download_table_state();
        self.status = "Download model table focused".into();
    }

    fn focus_download_editor(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        if self.download.selected_model().is_none() {
            self.status = "No download model selected".into();
            return;
        }
        self.download
            .set_focus(DownloadFocus::Model(ModelField::RepoId));
        self.status = "Download model editor focused · Tab moves through fields".into();
    }

    fn describe_download_focus(&mut self) {
        let label = match self.download.focus() {
            DownloadFocus::Table => "model table".to_owned(),
            DownloadFocus::ConfigPath => "config path".to_owned(),
            DownloadFocus::Versions => "config snapshots".to_owned(),
            DownloadFocus::BaseModelsDir => "base models directory".to_owned(),
            DownloadFocus::SlowPreset => "slow preset".to_owned(),
            DownloadFocus::GlobalWorkers => "global worker override".to_owned(),
            DownloadFocus::SaveNote => "save note".to_owned(),
            DownloadFocus::Model(field) => format!("model {}", field.label()),
        };
        self.status = format!("Download focus: {label} · Tab/Shift+Tab move · Esc table");
    }

    fn add_download_model(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        let index = self.download.add_model();
        self.sync_download_table_state();
        self.status = format!("Added download model row {} · enter repo_id", index + 1);
    }

    fn apply_download_edit(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        match self.download.apply_selected() {
            Ok(index) => {
                self.sync_download_table_state();
                self.status = format!("Applied editor changes to model row {}", index + 1);
            }
            Err(error) => {
                self.status = "Could not apply download model edit".into();
                self.record_download_message(format!("Apply failed: {error:#}"));
            }
        }
    }

    fn delete_download_model(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        match self.download.delete_selected() {
            Some(model) => {
                self.sync_download_table_state();
                self.status = format!("Deleted download model {}", value_or_dash(&model.repo_id));
            }
            None => self.status = "No download model selected".into(),
        }
    }

    fn toggle_download_enabled(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        match self.download.toggle_selected_enabled() {
            Ok(enabled) => {
                let repo_id = self
                    .download
                    .selected_model()
                    .map(|model| value_or_dash(&model.repo_id).to_owned())
                    .unwrap_or_else(|| "model".into());
                self.status = format!("{} {repo_id}", if enabled { "Enabled" } else { "Disabled" });
            }
            Err(error) => {
                self.status = "Could not toggle download model".into();
                self.record_download_message(format!("{error:#}"));
            }
        }
    }

    fn validate_download_config(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        match self.download.validate() {
            Ok(errors) if errors.is_empty() => {
                self.sync_download_table_state();
                self.refresh_download_disk_space();
                self.status = "Download config validation passed".into();
            }
            Ok(errors) => {
                self.sync_download_table_state();
                self.status = format!("Download config has {} validation error(s)", errors.len());
                for error in errors {
                    self.record_download_message(format!("Validation: {error}"));
                }
            }
            Err(error) => {
                self.status = "Download config validation failed".into();
                self.record_download_message(format!("Validation: {error:#}"));
            }
        }
    }

    fn save_download_config(&mut self) {
        if !self.download_config_is_usable() {
            return;
        }
        match self.download.save() {
            Ok(()) => {
                self.download_reload_armed = false;
                self.download_restore_armed = false;
                self.sync_download_table_state();
                self.refresh_download_disk_space();
                self.status = format!(
                    "Saved download config with snapshot: {}",
                    self.download.editor().config_path().display()
                );
                if let Some(warning) = self.download.take_history_warning() {
                    self.status = "Saved download config; snapshot history is unavailable".into();
                    self.record_download_message(warning);
                }
            }
            Err(error) => {
                self.status = "Could not save download config".into();
                self.record_download_message(format!("Save failed: {error:#}"));
            }
        }
    }

    fn reload_download_config(&mut self) {
        if self.download.is_dirty() && !self.download_reload_armed {
            self.download_reload_armed = true;
            self.download_restore_armed = false;
            self.status =
                "Unsaved Download edits: trigger reload again to discard them, or press any other key"
                    .into();
            return;
        }
        self.download_reload_armed = false;
        self.download_restore_armed = false;
        let was_blocked = self.download_load_error.is_some();
        match self.download.reload_path_in(&self.root) {
            Ok(()) => {
                self.download_load_error = None;
                self.sync_download_table_state();
                self.refresh_download_disk_space();
                self.status = format!(
                    "Loaded download config: {}",
                    self.download.editor().config_path().display()
                );
                if let Some(warning) = self.download.take_history_warning() {
                    self.status = "Loaded download config; snapshot history is unavailable".into();
                    self.record_download_message(warning);
                }
            }
            Err(error) => {
                let message = format!("Load failed: {error:#}");
                if was_blocked {
                    self.download_load_error = Some(message.clone());
                }
                self.status = "Could not load download config; current state retained".into();
                self.record_download_message(message);
            }
        }
    }

    fn restore_download_config(&mut self) {
        if self.download.versions().is_empty() {
            if let Err(error) = self.download.refresh_versions() {
                self.status = "Could not list download config snapshots".into();
                self.record_download_message(format!("Snapshot list failed: {error:#}"));
                return;
            }
        }
        if self.download.versions().is_empty() {
            self.status = "No download config snapshots available".into();
            return;
        }
        if self.download.selected_version().is_none() {
            self.download.select_version(0);
            self.download.set_focus(DownloadFocus::Versions);
            self.download_restore_armed = false;
            self.status = "Choose a download config snapshot and press Enter to restore".into();
            return;
        }
        if self.download.is_dirty() && !self.download_restore_armed {
            self.download_restore_armed = true;
            self.download_reload_armed = false;
            self.status =
                "Unsaved Download edits: trigger restore again to discard them, or press any other key"
                    .into();
            return;
        }
        self.download_restore_armed = false;
        self.download_reload_armed = false;
        match self.download.restore_selected_version() {
            Ok(version) => {
                self.download_load_error = None;
                self.sync_download_table_state();
                self.refresh_download_disk_space();
                self.status = format!("Restored download config snapshot {version}");
                if let Some(warning) = self.download.take_history_warning() {
                    self.status =
                        format!("Restored snapshot {version}; snapshot history is unavailable");
                    self.record_download_message(warning);
                }
            }
            Err(error) => {
                self.status = "Could not restore download config snapshot".into();
                self.record_download_message(format!("Restore failed: {error:#}"));
            }
        }
    }

    fn download_config_is_usable(&mut self) -> bool {
        let Some(error) = self.download_load_error.as_deref() else {
            return true;
        };
        self.status =
            "Download config is blocked; edit its path and use Alt+O, or restore a snapshot".into();
        if self.download_log.back().map(String::as_str) != Some(error) {
            self.push_download_log(error.to_owned());
        }
        false
    }

    fn clear_download_log(&mut self) {
        self.download_log.clear();
        self.status = "Download activity log cleared".into();
    }

    fn refresh_download_disk_space(&mut self) {
        let raw = self.download.base_models_dir_buffer().content().trim();
        self.download_disk_request_id = next_request_id(self.download_disk_request_id);
        if raw.is_empty() {
            self.download_disk_space = "Disk: —".into();
            return;
        }
        let path = PathBuf::from(raw);
        let path = if path.is_absolute() {
            path
        } else {
            self.root.join(path)
        };
        let request_id = self.download_disk_request_id;
        self.download_disk_space = format!("Disk: checking… [{}]", path.display());
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = probe_disk_space(&path, DOWNLOAD_DISK_TIMEOUT)
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::DownloadDiskSpace {
                request_id,
                target: path,
                result,
            });
        });
    }

    fn download_launch_is_busy(&mut self) -> bool {
        if self.download_preflight_pending.is_some() {
            self.status = "A download preflight is already running; Esc cancels it".into();
            return true;
        }
        if self.running_process.is_some() {
            self.status = "A process is already running; stop it first".into();
            return true;
        }
        false
    }

    fn begin_download_preflight(&mut self, name: String, parts: Vec<OsString>, target: PathBuf) {
        if self.download_launch_is_busy() {
            return;
        }
        if parts.is_empty() {
            self.status = "Could not build download command".into();
            self.push_download_log("Download command is empty");
            return;
        }

        self.download_preflight_request_id = next_request_id(self.download_preflight_request_id);
        let request_id = self.download_preflight_request_id;
        let mut estimate_parts = parts.clone();
        estimate_parts.push("--estimate-json".into());
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation);
        let worker_target = target.clone();
        let sender = self.sender.clone();
        let worker = thread::spawn(move || {
            let result = run_download_preflight_cancellable(
                estimate_parts,
                &worker_target,
                DOWNLOAD_ESTIMATE_TIMEOUT,
                DOWNLOAD_DISK_TIMEOUT,
                &worker_cancellation,
            )
            .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::DownloadPreflight { request_id, result });
        });

        self.status = format!("Checking remote sizes for {name}… · Esc cancels");
        self.push_download_log(format!(
            "Checking remote sizes and disk space for {name} [{}]",
            target.display()
        ));
        self.download_preflight_pending = Some(PendingDownloadPreflight {
            request_id,
            name,
            parts,
            target,
            cancellation,
            worker: Some(worker),
        });
    }

    fn cancel_download_preflight(&mut self) -> bool {
        let Some(pending) = self.download_preflight_pending.as_ref() else {
            return false;
        };
        if !pending.cancellation.swap(true, Ordering::AcqRel) {
            self.status = format!("Cancelling download preflight for {}…", pending.name);
        }
        true
    }

    fn record_download_preflight(&mut self, target: &Path, preflight: &DownloadPreflight) {
        self.download_disk_request_id = next_request_id(self.download_disk_request_id);
        self.download_disk_space = preflight.disk_space.map_or_else(
            || format!("Disk: unavailable [{}]", target.display()),
            |disk_space| format_disk_space(target, disk_space),
        );
        let totals = &preflight.estimate.totals;
        self.download_estimate_total = (totals.download_bytes > 0).then_some(totals.download_bytes);
        self.download_progress = None;
        self.push_download_log(format!(
            "Estimate: {} to download / {} matched · {} file(s) · {} cached · {} model(s)",
            format_bytes(totals.download_bytes),
            format_bytes(totals.total_bytes),
            totals.matched_files,
            format_bytes(totals.cached_bytes),
            totals.models,
        ));
        if let Some(warning) = &preflight.warning {
            self.push_download_log(format!("Preflight warning: {warning}"));
        }
    }

    fn update_download_progress(&mut self, line: &str) {
        let Some(mut progress) = parse_download_progress_line(line) else {
            return;
        };
        if let Some(total) = self.download_estimate_total.filter(|total| *total > 0) {
            progress.total_bytes = total;
            if progress.downloaded_bytes <= 100 {
                progress.downloaded_bytes = total.saturating_mul(progress.downloaded_bytes) / 100;
            }
        }
        self.download_progress = Some(progress);
    }

    fn download_eta_label(&self) -> Option<String> {
        let progress = self.download_progress?;
        let total = self
            .download_estimate_total
            .filter(|total| *total > 0)
            .unwrap_or(progress.total_bytes);
        let downloaded = progress.downloaded_bytes.min(total);
        let remaining = total.saturating_sub(downloaded);
        if remaining == 0 {
            return Some("ETA done".into());
        }
        let speed = progress
            .speed_bytes_per_second
            .filter(|speed| *speed > 0.0)
            .or_else(|| {
                self.download_started_at.and_then(|started| {
                    let seconds = started.elapsed().as_secs_f64();
                    (seconds > 0.0).then(|| downloaded as f64 / seconds)
                })
            })?;
        let seconds = remaining as f64 / speed;
        if !seconds.is_finite() || seconds < 0.0 {
            return None;
        }
        Some(format!(
            "ETA {}",
            format_duration(Duration::from_secs_f64(seconds))
        ))
    }

    fn handle_jobs_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.jobs_state, self.job_history.records().len())
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.jobs_state, self.job_history.records().len())
            }
            KeyCode::Char('s') => self.stop_active_job(),
            KeyCode::Char('r') => self.retry_selected_job(),
            KeyCode::Char('c') | KeyCode::Delete => {
                if self.can_clear_job_history() {
                    self.clear_job_history();
                } else {
                    self.status = "Cannot clear job history while an operation is running".into();
                }
            }
            _ => {}
        }
    }

    fn handle_maintenance_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_script_selection(ScriptEditorTarget::Maintenance, true)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_script_selection(ScriptEditorTarget::Maintenance, false)
            }
            KeyCode::Enter | KeyCode::Char('r') => self.run_selected_maintenance(),
            KeyCode::Char('s') => self.stop_running_process(),
            _ => {}
        }
    }

    fn selected_chat_model(&self) -> Option<SwapModel> {
        let selectable = self.chat_picker_model_indices();
        let model_index = *selectable.get(self.chat_model_state.selected()?)?;
        self.chat_models.get(model_index).cloned()
    }

    fn select_previous_chat_model(&mut self) {
        let count = self.chat_picker_model_indices().len();
        select_previous_table(&mut self.chat_model_state, count);
        if let Some(model) = self.selected_chat_model() {
            self.status = format!("Chat model: {}", model.id);
        }
    }

    fn select_next_chat_model(&mut self) {
        let count = self.chat_picker_model_indices().len();
        select_next_table(&mut self.chat_model_state, count);
        if let Some(model) = self.selected_chat_model() {
            self.status = format!("Chat model: {}", model.id);
        }
    }

    fn visible_browser_files(&self) -> Vec<GgufFile> {
        let filter = self.browser_filter.trim().to_ascii_lowercase();
        let mut files = self
            .browser_files
            .iter()
            .filter(|file| {
                if filter.is_empty() {
                    return true;
                }
                let metadata = file.metadata.as_ref();
                self.browser_relative_path(file)
                    .to_ascii_lowercase()
                    .contains(&filter)
                    || file.quantization.to_ascii_lowercase().contains(&filter)
                    || metadata
                        .and_then(|metadata| metadata.architecture.as_deref())
                        .is_some_and(|value| value.to_ascii_lowercase().contains(&filter))
                    || metadata
                        .and_then(|metadata| metadata.name.as_deref())
                        .is_some_and(|value| value.to_ascii_lowercase().contains(&filter))
            })
            .cloned()
            .collect::<Vec<_>>();
        files.sort_by(|left, right| {
            let ordering = match self.browser_sort {
                BrowserSort::NameAsc => self
                    .browser_relative_path(left)
                    .to_ascii_lowercase()
                    .cmp(&self.browser_relative_path(right).to_ascii_lowercase()),
                BrowserSort::SizeDesc => right.size.cmp(&left.size),
                BrowserSort::SizeAsc => left.size.cmp(&right.size),
                BrowserSort::ModifiedDesc => right.modified.cmp(&left.modified),
                BrowserSort::ModifiedAsc => left.modified.cmp(&right.modified),
                BrowserSort::QuantizationAsc => left
                    .quantization
                    .to_ascii_lowercase()
                    .cmp(&right.quantization.to_ascii_lowercase()),
            };
            ordering.then_with(|| left.path.cmp(&right.path))
        });
        files
    }

    fn browser_relative_path(&self, file: &GgufFile) -> String {
        file.path
            .strip_prefix(&self.browser_scanned_root)
            .unwrap_or(&file.path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn selected_browser_file(&self) -> Option<GgufFile> {
        let index = self.browser_state.selected()?;
        self.visible_browser_files().get(index).cloned()
    }

    /// Load the selected sidebar model (`load`) or unload the resident one.
    /// A load asks for a second `L` first, naming the model it will evict.
    fn model_action(&mut self, load: bool) {
        if self.model_action_pending {
            self.status = "A llama-swap action is already in progress".into();
            return;
        }
        if !load {
            let Some(model_id) = self.loaded_model_id.clone() else {
                self.status = "No loaded model is known".into();
                return;
            };
            self.begin_model_action(model_id, false);
            return;
        }
        let Some(model_id) = self.selected_swap_model_id() else {
            self.status = "The selected profile is not a llama-swap model".into();
            return;
        };
        if self.load_armed.as_deref() != Some(model_id.as_str()) {
            let evicts = match self.loaded_model_id.as_deref() {
                Some(resident) if resident == model_id => {
                    format!("{model_id} is already resident")
                }
                Some(resident) => format!("this evicts {resident}"),
                None => "nothing is resident".into(),
            };
            self.status = format!("Load {model_id}? {evicts} · press L again to confirm");
            self.load_armed = Some(model_id);
            return;
        }
        self.load_armed = None;
        self.begin_model_action(model_id, true);
    }

    /// The llama-swap model id of the selected sidebar profile, when the
    /// router lists it.
    fn selected_swap_model_id(&self) -> Option<String> {
        let profile = self.selected_stack_profile()?;
        if profile.key.lane != Lane::Language {
            return None;
        }
        let id = profile.key.profile_id.as_str();
        self.models
            .iter()
            .any(|model| model.id == id)
            .then(|| id.to_owned())
    }
    fn begin_model_action(&mut self, model_id: String, load: bool) {
        #[cfg(target_os = "linux")]
        if self.direct_native_job.is_some() || self.direct_native_admission_closed {
            self.status =
                "Finish or recover foreground native work before changing loaded models".into();
            return;
        }
        if self.model_action_pending {
            self.status = "A llama-swap action is already in progress".into();
            return;
        }
        self.model_action_pending = true;
        self.status = format!(
            "{} {}…",
            if load { "Loading" } else { "Unloading" },
            model_id
        );
        let verb = if load { "load" } else { "unload" };
        let job_id = self.job_history.begin(
            model_id.clone(),
            format!("model-{verb}"),
            vec!["llama-swap".into(), verb.into(), model_id.clone()],
            "run".into(),
            if load {
                model_id.clone()
            } else {
                String::new()
            },
        );
        self.jobs_state.select(Some(0));
        self.persist_jobs("model action start");
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = SwapClient::from_env()
                .and_then(|client| {
                    if load {
                        client.load_model(&model_id)
                    } else {
                        client.unload_model(&model_id)
                    }
                })
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::ModelAction {
                job_id,
                model_id,
                load,
                result,
            });
        });
    }

    /// Scan the GGUF browser path when it has never been scanned or changed
    /// since the last scan.
    fn ensure_browser_scan(&mut self) {
        let path = PathBuf::from(expand_tilde(&self.browser_path));
        if !self.browser_scanning && self.browser_scanned_root != path {
            self.scan_browser();
        }
    }

    fn refresh_stack_catalog(&mut self) {
        match Catalog::load(&self.root) {
            Ok(catalog) => {
                let count = catalog.list().len();
                self.stack_catalog = Some(catalog);
                self.stack_catalog_error = None;
                clamp_table_selection(&mut self.stack_catalog_state, count);
                self.status = format!("Manifests loaded · {count} profile(s)");
            }
            Err(error) => {
                self.stack_catalog = None;
                self.stack_catalog_error = Some(format!("{error:#}"));
                self.stack_catalog_state.select(None);
                self.status = "Could not read Stack Catalog manifests".into();
                if let Some(error) = &self.stack_catalog_error {
                    self.push_log(format!("Stack Catalog: {error}"));
                }
            }
        }
    }

    fn handle_stack_catalog_key(&mut self, key: KeyEvent) {
        let command = match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.sidebar_section = SidebarSection::Models;
                Some(CommandId::SidebarSelectPrevious)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.sidebar_section = SidebarSection::Models;
                Some(CommandId::SidebarSelectNext)
            }
            KeyCode::Char('r') => Some(CommandId::StackCatalogRefresh),
            KeyCode::Char('f') => Some(CommandId::StackCatalogForm),
            KeyCode::Char('c') => Some(CommandId::StackCatalogCancel),
            _ => None,
        };
        if let Some(command) = command {
            if let Err(error) = self.execute_command(command) {
                self.status = format!("Model: {error:#}");
            }
        }
    }

    fn begin_task_form(&mut self) {
        let Some(profile) = self.selected_stack_profile().cloned() else {
            self.status = "Select a manifest profile before opening the native task form".into();
            return;
        };

        #[cfg(target_os = "linux")]
        let direct_language_embeddings = self
            .stack_catalog
            .as_ref()
            .is_some_and(|catalog| is_direct_native_language_embeddings_profile(&profile, catalog));
        #[cfg(target_os = "linux")]
        let direct_native_profile = self
            .stack_catalog
            .as_ref()
            .is_some_and(|catalog| is_direct_native_profile(&profile, catalog));

        #[cfg(target_os = "linux")]
        if direct_native_profile {
            if self.direct_native_admission_closed {
                self.status = "Foreground native admission is closed after an unconfirmed completion; inspect Jobs and recover the backend before retrying".into();
                return;
            }
            if self.direct_native_job.is_some() {
                self.status = "A foreground native task is still running; press c to cancel and wait for its outcome".into();
                return;
            }
        }
        #[cfg(target_os = "linux")]
        {
            if !direct_native_profile
                && (self.direct_native_job.is_some() || self.direct_native_admission_closed)
            {
                self.status = "A foreground native task is active or interrupted; finish or recover it before other work".into();
                return;
            }
        }

        if self.handoff_chat_profile_to_chat(&profile) {
            return;
        }

        #[cfg(target_os = "linux")]
        {
            if profile.tasks.is_empty() {
                self.status = "Selected profile declares no task forms".into();
                return;
            }
            if !direct_native_profile {
                self.status = "No direct native workflow is available for this profile".into();
                return;
            }
            self.task_form_editor = Some(TaskFormEditor::new_with_direct_embeddings(
                &profile,
                direct_language_embeddings,
            ));
            self.input_mode = InputMode::TaskForm;
            self.status = if is_direct_blender_headless_profile(&profile) {
                "Foreground Blender headless form open · Python runs with your user permissions and is not sandboxed · shared GPU must be idle before starting · external GUI/MCP remains separate · run once with Ctrl+Enter".into()
            } else if is_direct_cuda_h3_profile(&profile) {
                "MiniMax H3 form open · music/tts use llama-swap; connect Chat first · Ctrl+Enter runs once".into()
            } else if is_direct_cuda_music_profile(&profile) {
                "HeartMuLa music form open · served by llama-swap (/audiocpp/) · connect Chat first · Ctrl+Enter runs once".into()
            } else if direct_run::is_supported_cuda_breeze_profile(&profile) {
                "Breeze speech form open · served by llama-swap (/audiocpp/) · connect Chat first · run once with Ctrl+Enter".into()
            } else if is_direct_native_decision_profile(&profile) {
                if profile.runtime.as_str() == "llamacpp" {
                    "Router decision may evict the loaded language model · quiet window required · Ctrl+Enter runs once".into()
                } else {
                    "Foreground native decision form open · request stays in memory; run once with Ctrl+Enter".into()
                }
            } else if is_direct_breeze_speech_profile(&profile) {
                "Foreground CPU task form open · request stays in memory; run once with Ctrl+Enter"
                    .into()
            } else if is_direct_comfy_workflow_profile(&profile) {
                "Comfy workflow form open · llama-swap shared engine (connect Chat first) · API-format workflow JSON · Ctrl+Enter runs once".into()
            } else if direct_language_embeddings {
                "Foreground language embeddings form open; run once with Ctrl+Enter".into()
            } else {
                "Native task form open · choose the declared task explicitly".into()
            };
        }
        #[cfg(not(target_os = "linux"))]
        {
            if profile.tasks.is_empty() {
                self.status = "Selected profile declares no task forms".into();
            } else {
                self.status =
                    "No direct native workflow is available for this profile on this platform"
                        .into();
            }
        }
    }

    /// Hand a chat-only language profile to the Chat view, without
    /// creating a request or changing any conversation state. `true` means
    /// this profile owns Chat and the user should not fall through to a form.
    fn handoff_chat_profile_to_chat(&mut self, captured_profile: &CatalogProfile) -> bool {
        if captured_profile.key.lane != Lane::Language
            || captured_profile.tasks.len() != 1
            || !captured_profile
                .tasks
                .iter()
                .any(|task| task.as_str() == "chat")
        {
            return false;
        }

        let refreshed_catalog = match self
            .stack_catalog
            .as_ref()
            .and_then(|catalog| catalog.refresh().ok())
        {
            Some(catalog) => catalog,
            None => {
                self.status = "Could not revalidate the selected Chat profile; refresh Stack Catalog before retrying".into();
                return true;
            }
        };
        let refreshed_index = refreshed_catalog
            .list()
            .iter()
            .position(|profile| profile.key == captured_profile.key);
        let refreshed_profile = refreshed_catalog.inspect(&captured_profile.key).cloned();
        self.stack_catalog = Some(refreshed_catalog);
        self.stack_catalog_state.select(refreshed_index);
        let Some(refreshed_profile) = refreshed_profile else {
            self.status =
                "Selected Chat profile disappeared; refresh Stack Catalog and select it again"
                    .into();
            return true;
        };
        if refreshed_profile.disabled {
            self.status =
                "Selected llama-swap profile is disabled; enable it before opening Chat".into();
            return true;
        }
        if &refreshed_profile != captured_profile {
            self.status = "Selected Chat profile changed since it was displayed; confirm the refreshed route before retrying".into();
            return true;
        }
        let route = match self
            .stack_catalog
            .as_ref()
            .and_then(|catalog| resolve_profile_route(catalog, &captured_profile.key).ok())
        {
            Some(route) => route,
            None => {
                self.status = "Selected Chat profile has no valid native llama-swap route".into();
                return true;
            }
        };
        let NativeRoute::LlamaSwapModel { model_id, .. } = route.native_route else {
            self.status = "Selected Chat profile does not resolve to a llama-swap model".into();
            return true;
        };
        if self.chat_connection_pending {
            self.status =
                "Wait for the current Chat connection or model refresh before handing off".into();
            return true;
        }
        if self.chat_pending || self.chat_stream_cancellation.is_some() || self.chat_session_pending
        {
            self.status =
                "Finish the active Chat response or session action before handing off".into();
            return true;
        }
        let Some(client) = self.chat_client.as_ref() else {
            self.status = "Connect Chat before opening this llama-swap profile".into();
            return true;
        };
        if self.chat_endpoint_committed.as_deref() != Some(client.base_url()) {
            self.status = "The current Chat endpoint is not a confirmed connection; reconnect before handing off".into();
            return true;
        }
        let Some((model_index, model)) = self
            .chat_models
            .iter()
            .enumerate()
            .find(|(_, model)| model.id == model_id.as_str())
        else {
            self.status = format!(
                "Route model {} is not advertised by the connected Chat endpoint; refresh Chat models",
                model_id
            );
            return true;
        };
        if model.disabled {
            self.status = format!("Route model {} is disabled in Chat", model_id);
            return true;
        }

        let Some(picker_index) = self
            .chat_picker_model_indices()
            .iter()
            .position(|&index| index == model_index)
        else {
            self.status = format!(
                "Route model {} advertises non-chat output and cannot be selected for Chat",
                model_id
            );
            return true;
        };

        self.chat_model_state.select(Some(picker_index));
        self.view = View::Chat;
        self.input_mode = InputMode::ChatMessage;
        self.status = format!(
            "Chat selected {} · no message sent; existing draft and session preserved",
            model_id
        );
        true
    }

    fn run_task_form(&mut self) {
        #[cfg(not(target_os = "linux"))]
        {
            self.status = "Direct native task execution is available on Linux only".into();
        }
        #[cfg(target_os = "linux")]
        {
            if self
                .task_form_editor
                .as_ref()
                .is_some_and(TaskFormEditor::is_direct_native)
            {
                self.run_direct_native();
                return;
            }
            self.status = "Only allowlisted direct native forms can run".into();
        }
    }

    #[cfg(target_os = "linux")]
    fn run_direct_native(&mut self) {
        if self.chat_connection_pending
            || self.chat_pending
            || self.chat_stream_cancellation.is_some()
            || self.chat_session_pending
            || self.model_action_pending
        {
            self.status =
                "Wait for active Chat or model operations before starting foreground native work"
                    .into();
            return;
        }
        if self.direct_native_admission_closed {
            self.status = "Foreground native admission is closed after an unconfirmed completion; inspect Jobs and recover the backend before retrying".into();
            return;
        }
        if self.direct_native_job.is_some() {
            self.status = "A foreground native task is still running; press c to cancel and wait for its outcome".into();
            return;
        }
        let Some(catalog) = self.stack_catalog.as_ref() else {
            self.status =
                "Refresh the Stack Catalog before running a foreground native task".into();
            return;
        };
        let Some(editor) = self.task_form_editor.as_ref() else {
            self.status = "Native task form is not open".into();
            return;
        };
        if !editor.is_direct_native()
            || catalog.inspect(&editor.profile) != Some(&editor.profile_snapshot)
        {
            self.status = "Selected native profile changed; refresh and reopen the form".into();
            return;
        }
        let lane = editor.profile.lane;
        let profile_id = editor.profile.profile_id.as_str().to_owned();
        let draft = match editor.to_draft() {
            Ok(draft) => draft,
            Err(error) => {
                self.status = error.to_string();
                return;
            }
        };
        let catalog = catalog.clone();
        let (request, timeout) = match build_direct_native(&catalog, draft) {
            Ok(request) => request,
            Err(error) => {
                self.status = error.to_string();
                return;
            }
        };
        let task = request.task().to_owned();
        let comfy_request = is_direct_comfy_workflow_request(&catalog, &request);
        let hosted_jev_decision = lane == Lane::Decisions
            && catalog
                .inspect(request.profile())
                .is_some_and(is_hosted_direct_decision_profile);
        let llamacpp_decision = lane == Lane::Decisions
            && catalog
                .inspect(request.profile())
                .is_some_and(direct_run::is_llamacpp_decision_profile);
        let remote_decision = hosted_jev_decision || llamacpp_decision;
        let routed_audio = is_routed_speech_request(&catalog, &request);
        let requires_idle_gpu_window = !routed_audio
            && catalog
                .inspect(request.profile())
                .is_some_and(direct_run::requires_operator_gpu_window);
        let blender_headless = catalog
            .inspect(request.profile())
            .is_some_and(is_direct_blender_headless_profile);
        let script_media_request = is_direct_native_script_request(&catalog, &request);
        let task_label = if comfy_request {
            "Comfy workflow"
        } else if script_media_request {
            direct_script_media_task_label(&profile_id, &task)
        } else {
            direct_native_task_label(&task)
        };
        let language_adapter = if routed_audio {
            match self.prepare_direct_routed_speech_adapter(&catalog, &request) {
                Ok(adapter) => Some(adapter),
                Err(message) => {
                    self.status = message;
                    return;
                }
            }
        } else if task == "embeddings" {
            match self.prepare_direct_embedding_adapter(&catalog, &request) {
                Ok(adapter) => Some(adapter),
                Err(message) => {
                    self.status = message;
                    return;
                }
            }
        } else if llamacpp_decision {
            match self.prepare_direct_llamacpp_decision_adapter(&catalog, &request) {
                Ok(adapter) => Some(adapter),
                Err(message) => {
                    self.status = message;
                    return;
                }
            }
        } else {
            None
        };
        let comfy_adapter = if comfy_request {
            match self.prepare_direct_comfy_adapter(&catalog, &request) {
                Ok(adapter) => Some(adapter),
                Err(message) => {
                    self.status = message.into();
                    return;
                }
            }
        } else {
            None
        };
        let cancellation = Arc::new(AtomicBool::new(false));
        let native_id = if comfy_request {
            let cancellation_view = DirectNativeCancellation(cancellation.clone());
            let candidate_context =
                OperationContext::new(Instant::now() + timeout, &cancellation_view);
            match crate::control::comfy::fresh_comfy_candidate(&candidate_context) {
                Ok(native_id) => Some(native_id),
                Err(_) => {
                    self.status =
                        "Could not generate a fresh Comfy job ID; nothing was started".into();
                    return;
                }
            }
        } else {
            None
        };

        // History remains metadata-only: only a native Comfy job ID is added
        // as request_id; no workflow graph, prompt, command, or native result
        // is persisted. Output references are added only after validation.
        let toolkit = match lane {
            Lane::Language => "l3ms",
            Lane::Decisions => "l3ds",
            Lane::Media => "l2m2",
        };
        let job_id = match self.job_history.begin_cross_toolkit(CrossToolkitJobStart {
            toolkit: toolkit.into(),
            profile: Some(profile_id.clone()),
            task: Some(task.clone()),
            request_id: native_id.clone(),
        }) {
            Ok(job_id) => job_id,
            Err(_) => {
                self.status = "Could not create safe foreground native job history".into();
                return;
            }
        };
        self.jobs_state.select(Some(0));
        if self
            .job_history
            .update_cross_toolkit(
                job_id,
                CrossToolkitJobUpdate {
                    result_state: Some("running".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .is_err()
        {
            self.finish_direct_native_history(job_id, "failed", None);
            self.status = "Could not mark foreground native task as running".into();
            return;
        }
        if self.job_history.persist_in(&self.data_root).is_err() {
            let _ = self.job_history.update_cross_toolkit(
                job_id,
                CrossToolkitJobUpdate {
                    result_state: Some("failed".into()),
                    ..CrossToolkitJobUpdate::default()
                },
            );
            self.status =
                "Safe foreground native job history could not be saved; nothing was started".into();
            return;
        }

        let output_directory = if is_direct_breeze_speech_request(&request)
            || script_media_request
            || comfy_request
        {
            match create_private_direct_output_directory(
                &self.data_root,
                job_id,
                if comfy_request {
                    "comfy-job"
                } else if script_media_request {
                    if blender_headless {
                        "blender-job"
                    } else {
                        "cuda-script-job"
                    }
                } else {
                    "cpu-job"
                },
            ) {
                Ok(directory) => Some(directory),
                Err(()) => {
                    self.finish_direct_native_history(job_id, "failed", None);
                    self.status = "Could not allocate a fresh private native output directory; nothing was started".into();
                    return;
                }
            }
        } else {
            None
        };

        let worker_cancellation = cancellation.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        let deadline = Instant::now() + timeout;
        let worker_task = task.clone();
        let worker_comfy = comfy_adapter;
        let worker_native_id = native_id.clone();
        let worker = match thread::Builder::new()
            .name(format!("l3ms-native-{toolkit}-{job_id}"))
            .spawn(move || {
                let cancellation = DirectNativeCancellation(worker_cancellation);
                let context = OperationContext::new(deadline, &cancellation);
                let connection = match (
                    worker_comfy.as_ref(),
                    worker_native_id.as_deref(),
                    language_adapter.as_ref(),
                ) {
                    (Some(adapter), Some(native_id), _) => {
                        Some(direct_run::DirectConnection::Comfy { adapter, native_id })
                    }
                    (None, _, Some(adapter)) => {
                        Some(direct_run::DirectConnection::Language(adapter))
                    }
                    _ => None,
                };
                let outcome = direct_run::run_native_in_operator_window(
                    &catalog,
                    &request,
                    output_directory.as_deref(),
                    connection,
                    &context,
                );
                let _ = sender.send(DirectNativeCompletion { job_id, outcome });
            }) {
            Ok(worker) => worker,
            Err(_) => {
                self.finish_direct_native_history(job_id, "failed", None);
                self.direct_native_summary = Some(DirectNativeSummary {
                    job_id,
                    lane,
                    profile_id,
                    task,
                    requires_idle_gpu_window,
                    native_id,
                    state: DirectNativeState::Failed,
                    result: None,
                });
                self.status = "Could not start the foreground native task worker".into();
                return;
            }
        };

        self.direct_native_summary = Some(DirectNativeSummary {
            job_id,
            lane,
            profile_id: profile_id.clone(),
            task: worker_task.clone(),
            requires_idle_gpu_window,
            native_id: native_id.clone(),
            state: DirectNativeState::Running,
            result: None,
        });
        self.direct_native_job = Some(DirectNativeJob {
            job_id,
            cancellation,
            receiver,
            worker,
        });
        self.task_form_editor = None;
        self.input_mode = InputMode::Normal;
        self.status = if native_id.is_some() {
            format!(
                "{task_label} #{job_id} · Comfy ID {} saved before dispatch · press c to cancel and wait for terminal state",
                native_id.as_deref().unwrap_or_default()
            )
        } else if requires_idle_gpu_window {
            format!(
                "{task_label} #{job_id} running · GPU release is checked after runner cleanup; press c to cancel"
            )
        } else if lane == Lane::Language || remote_decision {
            format!(
                "{task_label} #{job_id} running · press c to cancel; remote completion may be uncertain"
            )
        } else if lane == Lane::Decisions {
            format!(
                "{task_label} #{job_id} running · press c to cancel and wait for the confirmed decision outcome"
            )
        } else {
            format!("{task_label} #{job_id} running · press c to cancel")
        };
        self.push_log(format!("{task_label} #{job_id} started"));
    }

    #[cfg(target_os = "linux")]
    fn prepare_direct_embedding_adapter(
        &self,
        catalog: &Catalog,
        request: &crate::control::requests::InferenceRequest,
    ) -> Result<LanguageAdapter, String> {
        if request.lane() != Lane::Language || request.task() != "embeddings" {
            return Err("Direct language work is limited to the declared embeddings task".into());
        }
        let route = resolve_profile_route(catalog, request.profile())
            .map_err(|_| "The selected embeddings route could not be revalidated".to_owned())?;
        let NativeRoute::LlamaSwapModel { model_id, .. } = route.native_route else {
            return Err("The selected embeddings task has no native llama-swap model route".into());
        };
        self.prepare_confirmed_chat_adapter(Some(model_id.as_str()), "embeddings")
    }

    #[cfg(target_os = "linux")]
    fn prepare_direct_llamacpp_decision_adapter(
        &self,
        catalog: &Catalog,
        request: &crate::control::requests::InferenceRequest,
    ) -> Result<LanguageAdapter, String> {
        if request.lane() != Lane::Decisions
            || request.task() != "decision"
            || !catalog
                .inspect(request.profile())
                .is_some_and(direct_run::is_llamacpp_decision_profile)
        {
            return Err(
                "Direct HTTP work is limited to the declared llama.cpp decision task".into(),
            );
        }
        let route = resolve_profile_route(catalog, request.profile()).map_err(|_| {
            "The selected llama.cpp decision route could not be revalidated".to_owned()
        })?;
        let NativeRoute::DecisionRunner { model, .. } = route.native_route else {
            return Err("The selected llama.cpp decision task has no router model ID".into());
        };
        self.prepare_confirmed_chat_adapter(Some(&model), "decision")
    }

    #[cfg(target_os = "linux")]
    fn prepare_direct_routed_speech_adapter(
        &self,
        catalog: &Catalog,
        request: &crate::control::requests::InferenceRequest,
    ) -> Result<LanguageAdapter, String> {
        if !is_routed_speech_request(catalog, request) {
            return Err("Routed audio requires a declared audio.cpp audio task".into());
        }
        self.prepare_confirmed_chat_adapter(None, "speech")
    }

    #[cfg(target_os = "linux")]
    fn prepare_confirmed_chat_adapter(
        &self,
        model_id: Option<&str>,
        task_label: &str,
    ) -> Result<LanguageAdapter, String> {
        if self.chat_connection_pending
            || self.chat_pending
            || self.chat_stream_cancellation.is_some()
            || self.chat_session_pending
        {
            return Err(format!(
                "Wait for the current Chat connection, response, or session action before running {task_label}"
            ));
        }
        if self.model_action_pending {
            return Err(format!(
                "Wait for the active model action to finish before running {task_label}"
            ));
        }
        let client = self
            .chat_client
            .as_ref()
            .ok_or_else(|| format!("Connect Chat before running native {task_label}"))?;
        if self.chat_endpoint_committed.as_deref() != Some(client.base_url()) {
            return Err(format!(
                "The current Chat endpoint is not a confirmed connection; reconnect before running {task_label}"
            ));
        }
        // Reserved engine routes (for example audiocpp_auto) are not Chat models.
        if let Some(model_id) = model_id {
            let Some(model) = self.chat_models.iter().find(|model| model.id == model_id) else {
                return Err(format!(
                    "The exact {task_label} route model is not advertised by the connected Chat endpoint; refresh Chat models"
                ));
            };
            if model.disabled {
                return Err(format!(
                    "The exact {task_label} route model is disabled in Chat"
                ));
            }
        }
        client.embedding_adapter().map_err(|_| {
            format!("Could not configure the confirmed Chat endpoint for {task_label}")
        })
    }

    #[cfg(target_os = "linux")]
    fn prepare_direct_comfy_adapter(
        &self,
        catalog: &Catalog,
        request: &crate::control::requests::InferenceRequest,
    ) -> Result<ComfyAdapter, &'static str> {
        if !is_direct_comfy_workflow_request(catalog, request) {
            return Err("Direct Comfy work requires a declared ComfyUI workflow task");
        }
        if self.chat_connection_pending
            || self.chat_pending
            || self.chat_stream_cancellation.is_some()
            || self.chat_session_pending
            || self.model_action_pending
        {
            return Err("Wait for active Chat or model operations before running a Comfy workflow");
        }
        if env::var("L3MS_COMFY_SCOPED_JOBS").as_deref() != Ok("1") {
            return Err(
                "Set L3MS_COMFY_SCOPED_JOBS=1 only after confirming canonical Comfy job IDs and atomic scoped cancellation",
            );
        }
        let endpoint = match env::var("L2M2_COMFY_URL") {
            Ok(endpoint) => endpoint,
            Err(env::VarError::NotPresent) => {
                let client = self
                    .chat_client
                    .as_ref()
                    .ok_or("Connect Chat before running a routed Comfy workflow")?;
                if self.chat_endpoint_committed.as_deref() != Some(client.base_url()) {
                    return Err("Reconnect Chat to confirm the Comfy router endpoint");
                }
                return client
                    .comfy_adapter()
                    .map_err(|_| "The confirmed Comfy router connection is invalid");
            }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("The configured Comfy endpoint is invalid");
            }
        };
        let api_key = match env::var("L3MS_COMFY_API_KEY") {
            Ok(api_key) => Some(api_key),
            Err(env::VarError::NotPresent) => None,
            Err(env::VarError::NotUnicode(_)) => {
                return Err("The configured Comfy API key is invalid");
            }
        };
        ComfyAdapter::new(
            &endpoint,
            api_key.as_deref(),
            ComfyContract::CanonicalIdsAndAtomicScopedCancellation,
        )
        .map_err(|_| "The configured Comfy endpoint or API key is invalid")
    }

    #[cfg(target_os = "linux")]
    fn begin_direct_native_cancel(&mut self) -> bool {
        let Some(job) = self.direct_native_job.as_ref() else {
            return false;
        };
        job.cancellation.store(true, Ordering::Release);
        if let Some(summary) = self
            .direct_native_summary
            .as_mut()
            .filter(|summary| summary.job_id == job.job_id)
        {
            summary.state = DirectNativeState::CancelRequested;
        }
        let label = self
            .direct_native_summary
            .as_ref()
            .filter(|summary| summary.job_id == job.job_id)
            .map_or("foreground native task", direct_native_summary_label);
        self.status = format!(
            "{label} #{} cancellation requested · waiting for operation outcome",
            job.job_id
        );
        true
    }

    #[cfg(target_os = "linux")]
    fn finish_direct_native_history(
        &mut self,
        job_id: u64,
        result_state: &str,
        artifact_refs: Option<Vec<String>>,
    ) {
        let updated = self
            .job_history
            .update_cross_toolkit(
                job_id,
                CrossToolkitJobUpdate {
                    result_state: Some(result_state.to_owned()),
                    artifact_refs,
                    ..CrossToolkitJobUpdate::default()
                },
            )
            .is_ok();
        if updated {
            self.persist_jobs("foreground native task finish");
        } else {
            self.push_log("Foreground native job history update failed");
        }
    }

    #[cfg(target_os = "linux")]
    fn apply_direct_native_completion(
        &mut self,
        expected_job_id: u64,
        completion: Option<DirectNativeCompletion>,
    ) {
        if self
            .direct_native_summary
            .as_ref()
            .is_none_or(|summary| summary.job_id != expected_job_id)
        {
            return;
        }
        let Some(completion) = completion.filter(|completion| completion.job_id == expected_job_id)
        else {
            self.direct_native_admission_closed = true;
            if let Some(summary) = self.direct_native_summary.as_mut() {
                summary.state = DirectNativeState::Interrupted;
            }
            self.finish_direct_native_history(expected_job_id, "interrupted", None);
            let label = self
                .direct_native_summary
                .as_ref()
                .map_or("foreground native task", direct_native_summary_label);
            let requires_idle_gpu_window = self
                .direct_native_summary
                .as_ref()
                .is_some_and(|summary| summary.requires_idle_gpu_window);
            self.status = if requires_idle_gpu_window {
                format!(
                    "{label} #{expected_job_id} interrupted without a confirmed worker result; shared GPU idle is unconfirmed, keep other clients quiesced and verify before retrying"
                )
            } else {
                format!(
                    "{label} #{expected_job_id} interrupted without a confirmed worker result; inspect Jobs and verify the backend before retrying"
                )
            };
            return;
        };

        match completion.outcome {
            Ok(result) if self.direct_native_result_matches_task(expected_job_id, &result) => {
                let artifact_refs = direct_native_artifact_refs(&self.data_root, &result);
                let latency = direct_native_latency(&result);
                self.finish_direct_native_history(expected_job_id, "succeeded", artifact_refs);
                if let Some(summary) = self
                    .direct_native_summary
                    .as_mut()
                    .filter(|summary| summary.job_id == expected_job_id)
                {
                    summary.state = DirectNativeState::Succeeded;
                    summary.result = Some(result);
                }
                let label = self
                    .direct_native_summary
                    .as_ref()
                    .map_or("foreground native task", direct_native_summary_label);
                let latency = latency.map_or_else(String::new, |latency| {
                    format!(" · native latency {:.2} ms", latency.as_secs_f64() * 1000.0)
                });
                self.status = format!("{label} #{expected_job_id} succeeded{latency}");
                self.push_log(format!("{label} #{expected_job_id} completed"));
            }
            Ok(_) => {
                self.finish_direct_native_history(expected_job_id, "failed", None);
                if let Some(summary) = self
                    .direct_native_summary
                    .as_mut()
                    .filter(|summary| summary.job_id == expected_job_id)
                {
                    summary.state = DirectNativeState::Failed;
                    summary.result = None;
                }
                self.status = format!(
                    "Foreground native job #{expected_job_id} returned an unexpected result lane"
                );
            }
            Err(error) if error.blocks_following_work() => {
                self.direct_native_admission_closed = true;
                self.finish_direct_native_history(expected_job_id, "interrupted", None);
                if let Some(summary) = self.direct_native_summary.as_mut() {
                    summary.state = DirectNativeState::Interrupted;
                }
                let label = self
                    .direct_native_summary
                    .as_ref()
                    .map_or("foreground native task", direct_native_summary_label);
                let remote = error.is_remote();
                let requires_idle_gpu_window = self
                    .direct_native_summary
                    .as_ref()
                    .is_some_and(|summary| summary.requires_idle_gpu_window);
                self.status = if remote {
                    format!(
                        "{label} #{expected_job_id} interrupted: {error}; foreground admission is closed; verify the endpoint before retrying"
                    )
                } else if requires_idle_gpu_window {
                    format!(
                        "{label} #{expected_job_id} interrupted: {error}; foreground admission is closed, keep other clients quiesced and verify the shared GPU is idle before retrying"
                    )
                } else {
                    format!(
                        "{label} #{expected_job_id} interrupted: {error}; inspect Jobs and recover the runner before retrying"
                    )
                };
                self.push_log(if remote {
                    format!("{label} #{expected_job_id} remote completion unconfirmed")
                } else {
                    format!("{label} #{expected_job_id} requires runner recovery")
                });
            }
            Err(error) if error.remote_terminal_confirmed() => {
                let cancellation_confirmed = error.cancellation_confirmed();
                let result_state = if cancellation_confirmed {
                    "cancelled"
                } else {
                    "failed"
                };
                self.finish_direct_native_history(expected_job_id, result_state, None);
                if let Some(summary) = self.direct_native_summary.as_mut() {
                    summary.state = if cancellation_confirmed {
                        DirectNativeState::Cancelled
                    } else {
                        DirectNativeState::Failed
                    };
                }
                let label = self
                    .direct_native_summary
                    .as_ref()
                    .map_or("foreground native task", direct_native_summary_label);
                self.status = if cancellation_confirmed {
                    format!(
                        "{label} #{expected_job_id} cancelled · remote terminal cancellation confirmed"
                    )
                } else {
                    format!(
                        "{label} #{expected_job_id} reached a terminal remote state, but cancellation was not confirmed; no result was retained"
                    )
                };
                self.push_log(if cancellation_confirmed {
                    format!("{label} #{expected_job_id} cancelled")
                } else {
                    format!("{label} #{expected_job_id} reached a terminal remote state")
                });
            }
            Err(error) if error.cancellation_confirmed() => {
                self.finish_direct_native_history(expected_job_id, "cancelled", None);
                if let Some(summary) = self.direct_native_summary.as_mut() {
                    summary.state = DirectNativeState::Cancelled;
                }
                let label = self
                    .direct_native_summary
                    .as_ref()
                    .map_or("foreground native task", direct_native_summary_label);
                let remote = error.is_remote();
                self.status = if remote {
                    let backend = if self
                        .direct_native_summary
                        .as_ref()
                        .is_some_and(|summary| summary.native_id.is_some())
                    {
                        "Comfy"
                    } else {
                        "remote"
                    };
                    format!("{label} #{expected_job_id} cancelled before {backend} dispatch")
                } else if self
                    .direct_native_summary
                    .as_ref()
                    .is_some_and(|summary| summary.requires_idle_gpu_window)
                {
                    format!(
                        "{label} #{expected_job_id} cancelled · runner cleanup and shared GPU idle verification confirmed"
                    )
                } else {
                    format!(
                        "{label} #{expected_job_id} cancelled · owned process cleanup confirmed"
                    )
                };
                self.push_log(format!("{label} #{expected_job_id} cancelled"));
            }
            Err(error) => {
                self.finish_direct_native_history(expected_job_id, "failed", None);
                if let Some(summary) = self.direct_native_summary.as_mut() {
                    summary.state = DirectNativeState::Failed;
                }
                let label = self
                    .direct_native_summary
                    .as_ref()
                    .map_or("foreground native task", direct_native_summary_label);
                self.status = format!("{label} #{expected_job_id} failed: {error}");
                self.push_log(format!("{label} #{expected_job_id} failed: {error}"));
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn direct_native_result_matches_task(&self, job_id: u64, result: &DirectRunResult) -> bool {
        self.direct_native_summary
            .as_ref()
            .filter(|summary| summary.job_id == job_id)
            .is_some_and(|summary| match (summary.task.as_str(), result) {
                ("decision", DirectRunResult::Decision(native)) => {
                    summary.lane == Lane::Decisions
                        && matches!(
                            &native.result,
                            crate::control::requests::InferenceResult::Decision { .. }
                        )
                }
                ("tts", DirectRunResult::Speech(native)) => {
                    matches!(native.mode(), SpeechMode::Standard | SpeechMode::Streaming)
                        && summary.lane == Lane::Media
                        && is_direct_breeze_profile_id(&summary.profile_id)
                }
                ("clon", DirectRunResult::Speech(native)) => {
                    native.mode() == SpeechMode::VoiceClone
                        && summary.lane == Lane::Media
                        && is_direct_breeze_profile_id(&summary.profile_id)
                }
                ("design", DirectRunResult::Speech(native)) => {
                    native.mode() == SpeechMode::VoiceDesign
                        && summary.lane == Lane::Media
                        && is_direct_breeze_profile_id(&summary.profile_id)
                }
                (_, DirectRunResult::ScriptMedia(native)) => {
                    summary.lane == Lane::Media
                        && self.stack_catalog.as_ref().is_some_and(|catalog| {
                            catalog.list().iter().any(|profile| {
                                let blender = is_direct_blender_headless_profile(profile);
                                profile.key.lane == Lane::Media
                                    && profile.key.profile_id.as_str() == summary.profile_id
                                    && blender
                                    && profile
                                        .tasks
                                        .iter()
                                        .any(|task| task.as_str() == summary.task)
                                    && is_direct_script_media_task(
                                        &summary.profile_id,
                                        &summary.task,
                                    )
                                    && (!native.artifacts().is_empty()
                                        || (blender && summary.task == "3d"))
                            })
                        })
                }
                ("embeddings", DirectRunResult::Embeddings(_)) => summary.lane == Lane::Language,
                (_, DirectRunResult::Comfy(native)) => {
                    summary.lane == Lane::Media
                        && summary.native_id.as_deref() == Some(native.native_id.as_str())
                        && self.stack_catalog.as_ref().is_some_and(|catalog| {
                            catalog.list().iter().any(|profile| {
                                profile.key.lane == Lane::Media
                                    && profile.key.profile_id.as_str() == summary.profile_id
                                    && is_direct_comfy_workflow_profile(profile)
                                    && profile
                                        .tasks
                                        .iter()
                                        .any(|task| task.as_str() == summary.task)
                            })
                        })
                }
                _ => false,
            })
    }

    #[cfg(target_os = "linux")]
    fn drain_direct_native_result(&mut self) {
        let Some(job) = self.direct_native_job.as_ref() else {
            return;
        };
        let job_id = job.job_id;
        match job.receiver.try_recv() {
            Ok(completion) => {
                let Some(job) = self.direct_native_job.take() else {
                    return;
                };
                if job.job_id != completion.job_id {
                    self.direct_native_job = Some(job);
                    return;
                }
                if job.worker.join().is_err() {
                    self.apply_direct_native_completion(job_id, None);
                } else {
                    self.apply_direct_native_completion(job_id, Some(completion));
                }
            }
            Err(mpsc::TryRecvError::Disconnected) if job.worker.is_finished() => {
                let Some(job) = self.direct_native_job.take() else {
                    return;
                };
                let _ = job.worker.join();
                self.apply_direct_native_completion(job_id, None);
            }
            Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => {}
        }
    }

    fn selected_stack_profile(&self) -> Option<&CatalogProfile> {
        self.stack_catalog
            .as_ref()?
            .list()
            .get(self.stack_catalog_state.selected()?)
    }

    fn scan_browser(&mut self) {
        if self.browser_scanning {
            return;
        }
        let path = PathBuf::from(expand_tilde(&self.browser_path));
        let recursive = self.browser_recursive;
        self.browser_scanned_root = path.clone();
        self.browser_scanning = true;
        self.status = format!(
            "Scanning {} ({})…",
            path.display(),
            if recursive { "recursive" } else { "top-level" }
        );
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result =
                gguf::scan_directory(&path, recursive).map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::BrowserScan(result));
        });
    }

    fn cancel_chat_request(&mut self) -> bool {
        if !self.chat_pending {
            return false;
        }
        if let Some(cancellation) = self.chat_stream_cancellation.take() {
            cancellation.store(true, Ordering::Release);
        }
        self.chat_stream_request_id = self.chat_stream_request_id.wrapping_add(1);
        self.chat_pending = false;
        self.chat_streaming.clear();
        self.status = "Chat response stopped".into();
        true
    }

    fn send_chat_message(&mut self) {
        #[cfg(target_os = "linux")]
        if self.direct_native_job.is_some() || self.direct_native_admission_closed {
            self.status = "A foreground native task is active or unconfirmed; resolve its outcome before sending Chat".into();
            return;
        }
        self.input_mode = InputMode::Normal;
        let prompt = self.chat_input.trim().to_owned();
        if prompt.is_empty() || self.chat_pending {
            return;
        }
        let Some(client) = self.chat_client.clone() else {
            self.status = "Connect to a Chat endpoint before chatting".into();
            return;
        };
        let Some(model) = self.selected_chat_model() else {
            self.status = "Connect a Chat endpoint with at least one model".into();
            return;
        };
        self.chat_input.clear();
        self.chat_history.push("user", prompt);
        self.chat_streaming.clear();
        self.chat_pending = true;
        self.chat_stream_request_id = self.chat_stream_request_id.wrapping_add(1);
        let request_id = self.chat_stream_request_id;
        let cancellation = Arc::new(AtomicBool::new(false));
        self.chat_stream_cancellation = Some(Arc::clone(&cancellation));
        self.status = format!("Waiting for {}…", model.id);
        let history = self.chat_history.request_messages();
        let system_prompt = self.chat_system_prompt.clone();
        let temperature = self.chat_temperature;
        let max_tokens = self.chat_max_tokens;
        let thinking = self.chat_thinking;
        let sender = self.sender.clone();
        thread::spawn(move || {
            let mut request = ChatRequest::new(model.id, history);
            request.system_prompt = system_prompt;
            request.temperature = temperature;
            request.max_tokens = max_tokens;
            request.thinking = thinking;
            let result = client
                .stream_completion_cancellable(&request, &cancellation, |delta| {
                    sender
                        .send(BackgroundEvent::ChatDelta {
                            request_id,
                            delta: delta.to_owned(),
                        })
                        .is_ok()
                })
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::ChatFinished { request_id, result });
        });
    }

    fn save_chat_session(&mut self) {
        if self.chat_pending {
            self.status = "Stop the active Chat response before saving".into();
            return;
        }
        if self.chat_history.is_empty() {
            self.status = "Nothing to save; chat is empty".into();
            return;
        }
        if self.chat_session_pending {
            self.status = "A chat session operation is already in progress".into();
            return;
        }
        self.chat_session_pending = true;
        self.status = "Saving chat session…".into();
        let history = self.chat_history.clone();
        let data_root = self.data_root.clone();
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = history
                .save_in(&data_root)
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::ChatSessionSaved(result));
        });
    }

    fn open_chat_sessions(&mut self) {
        if self.chat_pending {
            self.status = "Stop the active Chat response before loading a session".into();
            return;
        }
        self.show_chat_sessions = true;
        self.refresh_chat_sessions();
    }

    fn refresh_chat_sessions(&mut self) {
        if self.chat_session_pending {
            return;
        }
        self.chat_session_pending = true;
        self.status = "Loading chat sessions…".into();
        let data_root = self.data_root.clone();
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result =
                ChatHistory::list_sessions_in(&data_root).map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::ChatSessionsListed(result));
        });
    }

    fn load_selected_chat_session(&mut self) {
        if self.chat_pending {
            self.status = "Stop the active Chat response before loading a session".into();
            return;
        }
        if self.chat_session_pending {
            return;
        }
        let Some(index) = self.chat_sessions_state.selected() else {
            return;
        };
        let Some(file_name) = self
            .chat_sessions
            .get(index)
            .map(|session| session.file_name.clone())
        else {
            return;
        };
        self.chat_session_pending = true;
        self.status = format!("Loading chat session {file_name}…");
        let data_root = self.data_root.clone();
        let sender = self.sender.clone();
        thread::spawn(move || {
            let result = ChatHistory::load_session_in(&data_root, file_name)
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(BackgroundEvent::ChatSessionLoaded(result));
        });
    }

    fn open_chat_prompts(&mut self) {
        match list_prompts(&self.data_root) {
            Ok(prompts) if prompts.is_empty() => {
                self.show_chat_prompts = false;
                self.status = format!(
                    "No saved prompts; add Markdown files under {}",
                    prompts_path(&self.data_root).display()
                );
            }
            Ok(prompts) => {
                self.chat_prompts = prompts;
                self.chat_prompts_state.select(Some(0));
                self.show_chat_prompts = true;
                self.status = "Select a saved system prompt · Enter loads · Esc closes".into();
            }
            Err(error) => {
                self.show_chat_prompts = false;
                self.status = "Could not list saved prompts".into();
                self.push_log(format!("{error:#}"));
            }
        }
    }

    fn load_selected_chat_prompt(&mut self) {
        let Some(index) = self.chat_prompts_state.selected() else {
            return;
        };
        let Some(name) = self.chat_prompts.get(index).cloned() else {
            return;
        };
        match load_prompt(&self.data_root, &name) {
            Ok(prompt) => {
                self.chat_system_prompt = prompt.content;
                self.show_chat_prompts = false;
                self.input_mode = InputMode::ChatSystemPrompt;
                self.status = format!("Loaded system prompt {name}; edit or press Enter to keep");
            }
            Err(error) => {
                self.status = format!("Could not load system prompt {name}");
                self.push_log(format!("{error:#}"));
            }
        }
    }

    fn run_selected_bench(&mut self) {
        let Some(path) = self.selected_script_path(ScriptEditorTarget::Bench) else {
            self.status = if self.bench_scripts.is_empty() {
                "No bench scripts are available".into()
            } else {
                "No bench script matches the current filter".into()
            };
            return;
        };
        if self.bench_editor.selected_path() == Some(path.as_path()) && self.bench_editor.is_dirty()
        {
            self.status = "Save or reload the edited bench script before running it".into();
            return;
        }
        let command = command_for_script(&path, &[]);
        self.start_process_from_parts(
            display_name(&path),
            "bench",
            command.into_iter().map(OsString::from).collect(),
        );
    }

    fn run_selected_maintenance(&mut self) {
        let Some(index) = self.maintenance_state.selected() else {
            return;
        };
        let Some(path) = self.maintenance_scripts.get(index).cloned() else {
            return;
        };
        if self.maintenance_editor.selected_path() == Some(path.as_path())
            && self.maintenance_editor.is_dirty()
        {
            self.status = "Save or reload the edited maintenance script before running it".into();
            return;
        }
        let command = command_for_script(&path, &[]);
        self.start_process_from_parts(
            display_name(&path),
            "maintenance",
            command.into_iter().map(OsString::from).collect(),
        );
    }

    fn download_selected(&mut self) {
        if self.download_launch_is_busy() {
            return;
        }
        if !self.download_config_is_usable() {
            return;
        }
        let validation_errors = match self.download.validate() {
            Ok(errors) => errors,
            Err(error) => {
                self.status = "Could not prepare selected download".into();
                self.record_download_message(format!("Download validation: {error:#}"));
                return;
            }
        };
        self.sync_download_table_state();
        if !validation_errors.is_empty() {
            self.status = format!(
                "Selected download blocked by {} validation error(s)",
                validation_errors.len()
            );
            for error in validation_errors {
                self.record_download_message(format!("Validation: {error}"));
            }
            return;
        }
        let Some(index) = self.download.selected_index() else {
            self.status = "No download model selected".into();
            return;
        };
        let speed_args = match self.selected_download_speed_args(index) {
            Ok(args) => args,
            Err(error) => {
                self.status = "Invalid download speed settings".into();
                self.record_download_message(format!("{error:#}"));
                return;
            }
        };
        match build_selected_download_command(
            &self.root,
            self.download.config(),
            index,
            &speed_args,
        ) {
            Ok((name, parts)) => {
                match selected_download_target(&self.root, self.download.config(), index) {
                    Ok(target) => self.begin_download_preflight(name, parts, target),
                    Err(error) => {
                        self.status = "Could not resolve download target".into();
                        self.record_download_message(format!("{error:#}"));
                    }
                }
            }
            Err(error) => {
                self.status = "Could not build download command".into();
                self.record_download_message(format!("{error:#}"));
            }
        }
    }

    fn download_enabled(&mut self) {
        if self.download_launch_is_busy() {
            return;
        }
        if !self.download_config_is_usable() {
            return;
        }
        if self.download.is_dirty() {
            self.status = "Save the edited config before downloading enabled models".into();
            return;
        }
        let speed_args = match self.download.speed_args() {
            Ok(args) => args,
            Err(error) => {
                self.status = "Invalid download speed settings".into();
                self.record_download_message(format!("{error:#}"));
                return;
            }
        };
        let mut parts = downloader_command_prefix(&self.root);
        parts.push("--config".into());
        parts.push(self.download.editor().config_path().as_os_str().to_owned());
        parts.extend(speed_args.into_iter().map(OsString::from));
        let target = config_download_target(&self.root, self.download.config());
        self.begin_download_preflight("enabled downloads".into(), parts, target);
    }

    fn selected_download_speed_args(&self, index: usize) -> Result<Vec<String>> {
        if let Some(workers) = self.download.global_max_workers()? {
            return Ok(vec!["--max-workers".into(), workers.to_string()]);
        }
        if let Some(workers) = self
            .download
            .models()
            .get(index)
            .and_then(|model| model.max_workers)
        {
            return Ok(vec!["--max-workers".into(), workers.to_string()]);
        }
        if self.download.slow() {
            Ok(vec!["--slow".into()])
        } else {
            Ok(Vec::new())
        }
    }

    fn start_process_from_parts(&mut self, name: String, kind: &str, parts: Vec<OsString>) {
        if self.download_preflight_pending.is_some() {
            self.status = "A download preflight is running; Esc cancels it".into();
            return;
        }
        if self.running_process.is_some() {
            self.status = "A process is already running; stop it first".into();
            return;
        }
        let Some((program, arguments)) = parts.split_first() else {
            return;
        };
        let command_text: Vec<String> = parts
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect();
        let mut command = Command::new(program);
        command
            .args(arguments)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);

        let spawned = {
            #[cfg(test)]
            let _fixture_guard = crate::control::process::executable_fixture_guard();
            command.spawn()
        };
        let mut child = match spawned {
            Ok(child) => child,
            Err(error) => {
                self.status = format!("Could not start {name}");
                if kind == "download" {
                    self.record_download_message(format!("{error:#}"));
                } else {
                    self.push_log(format!("{error:#}"));
                }
                return;
            }
        };

        let script_path = if matches!(kind, "bench" | "maintenance") {
            command_text.get(1).cloned().unwrap_or_default()
        } else {
            String::new()
        };
        let job_id = self.job_history.begin(
            name.clone(),
            kind.into(),
            command_text.clone(),
            kind.into(),
            script_path,
        );
        let process_group = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let child = Arc::new(Mutex::new(Some(child)));
        self.running_process = Some(RunningProcess {
            job_id,
            process_group,
            child: Arc::clone(&child),
        });
        if kind == "download" {
            self.download_started_at = Some(Instant::now());
            self.download_progress = None;
        }
        self.jobs_state.select(Some(0));
        self.persist_jobs("process start");
        self.status = format!("Running {name}");
        let command_line = format!("$ {}", command_text.join(" "));
        if kind == "download" {
            self.record_download_message(command_line);
        } else {
            self.push_log(command_line);
        }

        if let Some(stdout) = stdout {
            stream_reader(stdout, job_id, self.sender.clone());
        }
        if let Some(stderr) = stderr {
            stream_reader(stderr, job_id, self.sender.clone());
        }
        let sender = self.sender.clone();
        thread::spawn(move || loop {
            let result = {
                let mut guard = match child.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                match guard.as_mut() {
                    Some(process) => process.try_wait(),
                    None => return,
                }
            };
            match result {
                Ok(Some(status)) => {
                    if let Ok(mut guard) = child.lock() {
                        *guard = None;
                    }
                    let _ = sender.send(BackgroundEvent::ProcessFinished {
                        job_id,
                        exit_code: status.code().unwrap_or(1),
                    });
                    return;
                }
                Ok(None) => thread::sleep(Duration::from_millis(100)),
                Err(error) => {
                    #[cfg(unix)]
                    {
                        let _ = signal_process_group(process_group, "TERM");
                        thread::sleep(Duration::from_millis(250));
                        let _ = signal_process_group(process_group, "KILL");
                    }
                    if let Ok(mut guard) = child.lock() {
                        if let Some(mut process) = guard.take() {
                            let _ = process.kill();
                            let _ = process.wait();
                        }
                    }
                    let _ = sender.send(BackgroundEvent::ProcessLine {
                        job_id,
                        line: format!("process monitor failed: {error}"),
                    });
                    let _ = sender.send(BackgroundEvent::ProcessFinished {
                        job_id,
                        exit_code: 1,
                    });
                    return;
                }
            }
        });
    }

    fn stop_running_process(&mut self) {
        let Some(running) = &self.running_process else {
            self.status = "No process is running".into();
            return;
        };
        #[cfg(unix)]
        let result = signal_process_group(running.process_group, "TERM").or_else(|group_error| {
            running
                .child
                .lock()
                .map_err(|_| "process lock is poisoned".to_owned())
                .and_then(|mut child| match child.as_mut() {
                    Some(child) => child
                        .kill()
                        .map_err(|error| format!("{group_error}; fallback kill failed: {error}")),
                    None => Ok(()),
                })
        });
        #[cfg(not(unix))]
        let result = running
            .child
            .lock()
            .map_err(|_| "process lock is poisoned".to_owned())
            .and_then(|mut child| match child.as_mut() {
                Some(child) => child.kill().map_err(|error| error.to_string()),
                None => Ok(()),
            });

        #[cfg(unix)]
        if result.is_ok() {
            let process_group = running.process_group;
            let child = Arc::clone(&running.child);
            thread::spawn(move || {
                thread::sleep(Duration::from_secs(3));
                let still_running = child
                    .lock()
                    .ok()
                    .and_then(|mut child| child.as_mut().and_then(|child| child.try_wait().ok()))
                    .is_some_and(|status| status.is_none());
                if still_running {
                    let _ = signal_process_group(process_group, "KILL");
                }
            });
        }
        self.status = match result {
            Ok(()) => "Stop requested".into(),
            Err(error) => format!("Stop failed: {error}"),
        };
    }

    fn retry_selected_job(&mut self) {
        let Some(index) = self.jobs_state.selected() else {
            return;
        };
        let Some(job) = self.job_history.get(index).cloned() else {
            return;
        };
        if job.is_running() {
            self.status = "The selected job is still running".into();
            return;
        }
        if job.command.is_empty() {
            self.status = "The selected historical job cannot be retried safely".into();
            return;
        }
        if matches!(job.kind.as_str(), "model-load" | "model-unload") {
            let Some(model_id) = job.command.get(2).cloned() else {
                self.status = "Model job has no model identifier".into();
                return;
            };
            self.begin_model_action(model_id, job.kind == "model-load");
            return;
        }
        self.start_process_from_parts(
            format!("{} (retry)", job.name),
            &job.kind,
            job.command.into_iter().map(OsString::from).collect(),
        );
    }

    fn stop_active_job(&mut self) {
        if self.running_process.is_some() {
            self.stop_running_process();
        } else if self.download_preflight_pending.is_some() {
            self.cancel_download_preflight();
        } else if self.model_action_pending {
            self.status = "The current llama-swap request cannot be cancelled safely".into();
        } else {
            self.status = "No active job to stop (U in the sidebar unloads the model)".into();
        }
    }

    fn can_clear_job_history(&self) -> bool {
        self.running_process.is_none()
            && !self.model_action_pending
            && !self
                .job_history
                .records()
                .iter()
                .any(|job| job.is_running())
    }

    fn clear_job_history(&mut self) {
        self.job_history.clear_for_recovery();
        self.jobs_state.select(None);
        self.status = "Job history cleared".into();
        self.persist_jobs("clear");
    }

    fn persist_jobs(&mut self, context: &str) {
        if let Err(error) = self.job_history.persist_in(&self.data_root) {
            self.push_log(format!("Job history {context} save failed: {error:#}"));
        }
    }

    fn push_log(&mut self, message: impl Into<String>) {
        self.log.push_back(message.into());
        while self.log.len() > 300 {
            self.log.pop_front();
        }
    }

    fn push_download_log(&mut self, message: impl Into<String>) {
        self.download_log.push_back(message.into());
        while self.download_log.len() > 200 {
            self.download_log.pop_front();
        }
    }

    fn record_download_message(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.push_download_log(message.clone());
        self.push_log(message);
    }

    fn clear_activity_log(&mut self, context: &str) {
        self.log.clear();
        self.status = format!("{context} activity log cleared");
    }

    fn finish_job(&mut self, job_id: u64, exit_code: i32) {
        let summary = self.job_history.finish(job_id, exit_code).map(|job| {
            (
                job.name.clone(),
                job.elapsed_label.clone(),
                job.exit_label.clone(),
            )
        });
        if let Some((name, elapsed, exit)) = summary {
            self.status = format!("{name} exited with {exit} after {elapsed}");
            self.persist_jobs("finish");
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        frame.render_widget(Block::default().style(theme::surface()), frame.area());
        let areas = shell::layout(frame.area(), self.sidebar_focused);
        frame.render_widget(
            Paragraph::new(toolbar::toolbar_line(self.view.name(), areas.toolbar.width))
                .style(theme::text()),
            areas.toolbar,
        );
        if let Some(area) = areas.sidebar {
            self.draw_sidebar(frame, area);
        }
        if let Some(area) = areas.pane {
            match self.view {
                View::Model => self.draw_stack_catalog(frame, area),
                View::Chat => self.draw_chat(frame, area),
                View::Jobs => self.draw_jobs(frame, area),
                View::Bench => self.draw_bench(frame, area),
                View::Download => self.draw_download(frame, area),
                View::Browser => self.draw_browser(frame, area),
                View::Maintenance => self.draw_maintenance(frame, area),
                View::System => self.draw_system(frame, area),
                View::Log => self.draw_log(frame, area),
            }
        }
        self.draw_footer(frame, areas.message);
        frame.render_widget(
            Paragraph::new(statusbar::status_line(
                &self.status_inputs(),
                areas.status.width,
            ))
            .style(theme::surface()),
            areas.status,
        );
        if self.show_palette {
            self.draw_palette(frame);
        } else if self.show_quit_confirmation {
            self.draw_quit_confirmation(frame);
        } else if self.show_help {
            self.draw_help(frame);
        } else if self.show_chat_sessions {
            self.draw_chat_sessions(frame);
        } else if self.show_chat_prompts {
            self.draw_chat_prompts(frame);
        } else if self.script_versions_target.is_some() {
            self.draw_script_versions(frame);
        } else if self.input_mode == InputMode::TaskForm {
            self.draw_task_form(frame);
        }
    }

    fn status_inputs(&self) -> StatusInputs {
        let endpoint = env::var("LLAMA_SWAP_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        let endpoint = endpoint
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_start_matches("localhost")
            .trim_start_matches("127.0.0.1")
            .trim_end_matches('/')
            .to_owned();
        StatusInputs {
            resident_model: self
                .models
                .iter()
                .find(|model| model.state == "loaded")
                .map(|model| model.id.clone()),
            vram_mib: self
                .gpu_telemetry
                .as_ref()
                .map(|gpu| (gpu.used_vram_mib, gpu.total_vram_mib)),
            ram_mib: self
                .sys_memory
                .as_ref()
                .map(|memory| (memory.used_ram_mib, memory.total_ram_mib)),
            cpu_percent: self.cpu_percent,
            governor: self.host.governor.clone(),
            fullest_disk: self
                .host
                .disks
                .iter()
                .max_by_key(|disk| disk.used_percent())
                .map(|disk| (disk.path.clone(), disk.used_percent())),
            endpoint,
        }
    }

    /// Sidebar model rows: lane headings, then each profile with its router
    /// state dot and a detail line (tasks or runtime, last steady t/s).
    fn sidebar_model_rows(&self) -> Vec<ModelRow> {
        let Some(catalog) = self.stack_catalog.as_ref() else {
            return Vec::new();
        };
        let profiles = catalog.list();
        let mut rows = Vec::new();
        let mut group = None;
        for index in self.sidebar_model_indices() {
            let profile = &profiles[index];
            let (name, _, style) = self.sidebar_group(profile);
            if group != Some(name) {
                group = Some(name);
                rows.push(ModelRow::Lane { name, style });
            }
            let id = profile.key.profile_id.as_str();
            let (dot, dot_style) = sidebar::model_dot(self.profile_router_state(profile));
            let tasks = profile
                .tasks
                .iter()
                .map(|task| task.as_str())
                .collect::<Vec<_>>()
                .join(",");
            let entry = self.yaml_entries.get(id);
            let note = if profile.disabled {
                "off".to_owned()
            } else {
                self.bench_rates
                    .get(id)
                    .map(|rate| format!("{rate:.0} t/s"))
                    .unwrap_or_default()
            };
            let detail = match (profile.key.lane, entry) {
                (Lane::Language, Some(entry)) if entry.proxy => {
                    "engine · used by media profiles".to_owned()
                }
                (Lane::Language, Some(entry)) => {
                    let (server, _) = model_view::cmd_facts(&entry.cmd);
                    match server {
                        Some(server) => format!("{tasks} · {}", self.server_display(&server)),
                        None => tasks,
                    }
                }
                _ => format!("{} · {tasks}", profile.runtime.as_str()),
            };
            rows.push(ModelRow::Model {
                catalog_index: index,
                id: id.to_owned(),
                dot,
                dot_style,
                note,
                muted: profile.disabled,
                detail,
            });
        }
        rows
    }

    fn sidebar_job_rows(&self) -> Vec<JobRow> {
        self.job_history
            .records()
            .iter()
            .map(|job| {
                let (dot, dot_style) = sidebar::job_dot(&job.status);
                let subject = if job.name.is_empty() {
                    job.kind.clone()
                } else {
                    format!("{} · {}", job.kind, job.name)
                };
                JobRow {
                    dot,
                    dot_style,
                    label: subject,
                    age: job.elapsed_label.clone(),
                }
            })
            .collect()
    }

    fn draw_sidebar(&self, frame: &mut Frame, area: Rect) {
        let models = self.sidebar_model_rows();
        let jobs = self.sidebar_job_rows();
        let model = SidebarModel {
            focused: self.sidebar_focused,
            section: self.sidebar_section,
            models_collapsed: self.models_collapsed,
            jobs_collapsed: self.jobs_collapsed,
            models: &models,
            selected_catalog_index: self.stack_catalog_state.selected(),
            filter: &self.model_filter,
            filter_editing: self.input_mode == InputMode::ModelFilter,
            jobs: &jobs,
            selected_job: self.jobs_state.selected(),
        };
        sidebar::draw(frame, area, &model);
    }

    /// Bench view: bench scripts beside the selected script.
    fn draw_bench(&mut self, frame: &mut Frame, area: Rect) {
        let panes = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(area);
        self.draw_bench_list_widget(frame, panes[0]);
        self.draw_script_editor(frame, panes[1], ScriptEditorTarget::Bench);
    }

    /// Read `llama-swap.yaml` and bench results once: note unlisted alias
    /// entries, entry facts, macros, GGUF sizes and each model's matching
    /// bench runs (see [`bench_runs_by_entry`]). A result belongs to a model only when its GGUF path is exactly
    /// the `-m` path of that model's yaml entry; anything else stays
    /// "not benched" rather than being guessed.
    fn refresh_bench_rates(&mut self) {
        self.bench_rates.clear();
        self.unlisted_models.clear();
        self.yaml_entries.clear();
        self.bench_runs.clear();
        self.model_sizes.clear();
        let Ok(yaml) = fs::read_to_string(self.root.join("llama-swap.yaml")) else {
            return;
        };
        let entries = crate::swap_yaml::parse_entries(&yaml);
        self.yaml_macros = crate::swap_yaml::parse_macros(&yaml);
        self.unlisted_models = entries
            .iter()
            .filter(|entry| entry.unlisted)
            .map(|entry| entry.id.clone())
            .collect();
        self.model_sizes = entries
            .iter()
            .filter_map(|entry| {
                let path = expand_tilde(entry.model_path.as_deref()?);
                Some((entry.id.clone(), crate::swap_yaml::disk_size(&path)?))
            })
            .collect();
        self.yaml_entries = entries
            .iter()
            .map(|entry| (entry.id.clone(), entry.clone()))
            .collect();
        let Ok(load) = crate::bench_results::load_results(&self.root) else {
            return;
        };
        self.bench_runs = bench_runs_by_entry(&entries, &load.results);
        self.bench_rates = self
            .bench_runs
            .iter()
            .filter_map(|(id, runs)| Some((id.clone(), runs.iter().find_map(|run| run.tg_ts)?)))
            .collect();
    }

    /// System view: GPU, memory and SSD spill in detail.
    fn draw_system(&self, frame: &mut Frame, area: Rect) {
        let title = "System · GPU, memory and spill · nvidia-smi and /proc";
        let block = theme::block().title(title).borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(inner);

        let left_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6),
                Constraint::Length(3),
                Constraint::Length(5),
                Constraint::Min(2),
            ])
            .split(cols[0]);

        let mut status_lines = Vec::new();
        if let Some(gpu) = &self.gpu_telemetry {
            status_lines.push(Line::from(vec![
                Span::styled("Device:  ", theme::title()),
                Span::styled(&gpu.name, theme::text()),
            ]));
            status_lines.push(Line::from(vec![
                Span::styled("Temp:    ", theme::title()),
                Span::styled(format!("{} °C", gpu.temperature_c), theme::text()),
                Span::raw("   Power: "),
                Span::styled(format!("{:.1} W", gpu.power_watts), theme::text()),
            ]));
            status_lines.push(Line::from(vec![
                Span::styled("GPU Core Util:  ", theme::title()),
                Span::styled(format!("{}%", gpu.gpu_util_percent), theme::text()),
                Span::raw("  Memory Controller: "),
                Span::styled(format!("{}%", gpu.mem_util_percent), theme::text()),
            ]));
        } else {
            status_lines.push(Line::from(Span::styled(
                "No NVIDIA GPU detected via nvidia-smi",
                theme::warning(),
            )));
        }
        frame.render_widget(
            Paragraph::new(status_lines).style(theme::text()),
            left_rows[0],
        );

        if let Some(gpu) = &self.gpu_telemetry {
            let vram_pct = (gpu.used_vram_mib * 100)
                .checked_div(gpu.total_vram_mib)
                .unwrap_or(0) as u16;
            let vram_label = format!(
                "VRAM: {:.2} / {:.2} GiB ({}%)",
                gpu.used_vram_mib as f64 / 1024.0,
                gpu.total_vram_mib as f64 / 1024.0,
                vram_pct
            );
            let gauge = Gauge::default()
                .block(
                    Block::default()
                        .title("VRAM Allocation")
                        .borders(Borders::NONE),
                )
                .gauge_style(theme::selected())
                .percent(vram_pct)
                .label(vram_label);
            frame.render_widget(gauge, left_rows[1]);
        }

        let mut mem_lines = Vec::new();
        if let Some(mem) = &self.sys_memory {
            let ram_pct = (mem.used_ram_mib * 100)
                .checked_div(mem.total_ram_mib)
                .unwrap_or(0);
            mem_lines.push(Line::from(vec![
                Span::styled("Host RAM: ", theme::title()),
                Span::raw(format!(
                    "{:.1}/{:.1} GiB ({}%) · Free: {:.1} GiB",
                    mem.used_ram_mib as f64 / 1024.0,
                    mem.total_ram_mib as f64 / 1024.0,
                    ram_pct,
                    mem.available_ram_mib as f64 / 1024.0
                )),
            ]));
            mem_lines.push(Line::from(vec![
                Span::styled("Swap/zram:", theme::title()),
                Span::raw(format!(
                    " {:.1}/{:.1} GiB (used: {:.1} GiB)",
                    mem.used_swap_mib as f64 / 1024.0,
                    mem.total_swap_mib as f64 / 1024.0,
                    mem.used_swap_mib as f64 / 1024.0
                )),
            ]));
        }
        if let Some(io) = &self.server_io {
            mem_lines.push(Line::from(vec![
                Span::styled("Server Disk Read: ", theme::title()),
                Span::raw(format!(
                    "{:.2} GiB cumulative",
                    io.read_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )),
            ]));
            if let Some(rate) = io.read_rate_bytes_per_sec {
                let status_str = if rate < 1024.0 * 1024.0 {
                    format!("{:.1} KB/s (Normal PLE traffic)", rate / 1024.0)
                } else {
                    format!(
                        "{:.1} MB/s (HOT EXPERT SPILL / NVMe THRASHING!)",
                        rate / (1024.0 * 1024.0)
                    )
                };
                mem_lines.push(Line::from(vec![
                    Span::styled("Spill Rate:       ", theme::title()),
                    Span::styled(status_str, theme::warning()),
                ]));
            }
        }
        frame.render_widget(Paragraph::new(mem_lines).style(theme::text()), left_rows[2]);

        let right_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(cols[1]);

        let gpu_data: Vec<u64> = self.gpu_util_history.iter().copied().collect();
        let spark_gpu = Sparkline::default()
            .block(
                Block::default()
                    .title("GPU Utilization Trend (Last 60s)")
                    .borders(Borders::ALL)
                    .title_style(theme::title()),
            )
            .data(&gpu_data)
            .max(100)
            .style(theme::title());
        frame.render_widget(spark_gpu, right_rows[0]);

        let vram_data: Vec<u64> = self.vram_history.iter().copied().collect();
        let vram_max = self
            .gpu_telemetry
            .as_ref()
            .map_or(12288, |g| g.total_vram_mib);
        let spark_vram = Sparkline::default()
            .block(
                Block::default()
                    .title("VRAM History (Last 60s)")
                    .borders(Borders::ALL)
                    .title_style(theme::warning()),
            )
            .data(&vram_data)
            .max(vram_max)
            .style(theme::warning());
        frame.render_widget(spark_vram, right_rows[1]);
    }

    fn draw_bench_list_widget(&mut self, frame: &mut Frame, area: Rect) {
        let visible = self
            .visible_bench_scripts()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        if visible.is_empty() {
            self.bench_state.select(None);
        }
        let items = if visible.is_empty() {
            vec![ListItem::new("No matching scripts")]
        } else {
            visible
                .iter()
                .map(|path| {
                    let shown = relative_display(&self.root, path);
                    let shown = shown
                        .strip_prefix("bench-models/")
                        .map(str::to_owned)
                        .unwrap_or(shown);
                    ListItem::new(shown)
                })
                .collect::<Vec<_>>()
        };
        let filter = if self.bench_filter.is_empty() {
            String::new()
        } else {
            format!(" · filter: {}", self.bench_filter)
        };
        let list = List::new(items)
            .style(theme::text())
            .block(
                theme::block()
                    .title(format!(
                        "Bench scripts{filter} · / filter · ↑/↓ select · Ctrl+U editor"
                    ))
                    .borders(Borders::ALL),
            )
            .highlight_symbol("▶ ")
            .highlight_style(theme::title());
        frame.render_stateful_widget(list, area, &mut self.bench_state);
    }

    fn draw_script_editor(&mut self, frame: &mut Frame, area: Rect, target: ScriptEditorTarget) {
        let editor = self.script_editor(target);
        let path = editor
            .selected_relative_path()
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|| "No script selected".into());
        let persisted_content = editor.content().to_owned();
        let dirty = editor.is_dirty();
        let synchronized = editor.content_synchronized();
        let versions = editor.versions().len();
        let focused = self.script_input_target == Some(target);
        let content = if focused {
            self.script_buffer.content().to_owned()
        } else {
            persisted_content
        };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(1)])
            .split(area);
        let content_area = chunks[0];
        let mut viewport = self.script_editor_view(target);

        if focused {
            let cursor = self.script_buffer.cursor_position();
            let cursor_display_column = text_buffer_cursor_display_width(&self.script_buffer);
            let visible_height = content_area.height.saturating_sub(2) as usize;
            let visible_width = content_area.width.saturating_sub(2) as usize;
            if visible_height > 0 {
                if cursor.line < viewport.scroll_y {
                    viewport.scroll_y = cursor.line;
                } else if cursor.line >= viewport.scroll_y + visible_height {
                    viewport.scroll_y = cursor.line + 1 - visible_height;
                }
            }
            if visible_width > 0 {
                if cursor_display_column < viewport.scroll_x {
                    viewport.scroll_x = cursor_display_column;
                } else if cursor_display_column >= viewport.scroll_x + visible_width {
                    viewport.scroll_x = cursor_display_column + 1 - visible_width;
                }
            }
            *self.script_editor_view_mut(target) = viewport;
        }

        let state = if !synchronized {
            " · OUT OF SYNC"
        } else if dirty {
            " · UNSAVED"
        } else {
            ""
        };
        let border = if focused {
            theme::border_focused()
        } else {
            Style::default()
        };
        let scroll = if focused {
            (
                viewport.scroll_y.min(u16::MAX as usize) as u16,
                viewport.scroll_x.min(u16::MAX as usize) as u16,
            )
        } else {
            (0, 0)
        };
        frame.render_widget(
            Paragraph::new(content)
                .style(theme::text())
                .block(
                    theme::block()
                        .title(format!(
                            "{} script · {path}{state} · {versions} snapshot(s)",
                            target.label()
                        ))
                        .borders(Borders::ALL)
                        .border_style(border),
                )
                .scroll(scroll),
            content_area,
        );
        frame.render_widget(
            Paragraph::new("Ctrl+U edit/Esc list · Alt+P save · Alt+O reload · Alt+V versions")
                .style(theme::dim()),
            chunks[1],
        );

        if focused && content_area.width > 2 && content_area.height > 2 {
            let cursor = self.script_buffer.cursor_position();
            let cursor_display_column = text_buffer_cursor_display_width(&self.script_buffer);
            let visible_width = content_area.width.saturating_sub(2);
            let visible_height = content_area.height.saturating_sub(2);
            if let (Some(x_offset), Some(y_offset)) = (
                visible_cursor_offset(cursor_display_column, scroll.1, visible_width),
                visible_cursor_offset(cursor.line, scroll.0, visible_height),
            ) {
                frame.set_cursor_position((
                    content_area.x + 1 + x_offset,
                    content_area.y + 1 + y_offset,
                ));
            }
        }
    }

    fn draw_chat(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Min(4),
                Constraint::Length(3),
            ])
            .split(area);
        let connection = if self.chat_connection_pending {
            "connecting…".to_owned()
        } else if self.chat_client.is_some() {
            format!(
                "connected · {} model(s)",
                self.chat_picker_model_indices().len()
            )
        } else {
            "disconnected".into()
        };
        let selected_model = self
            .selected_chat_model()
            .map(|model| model.id)
            .unwrap_or_else(|| "—".into());
        frame.render_widget(
            Paragraph::new(format!(
                "connection: {connection}\nendpoint: {}\nmodel: {selected_model} │ system: {} │ temp {:.1} │ max {} │ thinking {}",
                self.chat_endpoint_draft,
                if self.chat_system_prompt.trim().is_empty() {
                    "—"
                } else {
                    self.chat_system_prompt.as_str()
                },
                self.chat_temperature,
                self.chat_max_tokens,
                if self.chat_thinking { "on" } else { "off" }
            ))
            .style(theme::text())
            .block(
                theme::block()
                    .title("Chat · e endpoint · Ctrl+G connect · Ctrl+B detect · K stop detected · ↑/↓ model")
                    .borders(Borders::ALL),
            ),
            chunks[0],
        );
        if self.input_mode == InputMode::ChatSystemPrompt {
            frame.set_cursor_position((
                chunks[0].x + 10 + self.chat_system_prompt.chars().count() as u16,
                chunks[0].y + 3,
            ));
        }
        if self.input_mode == InputMode::ChatEndpoint {
            frame.set_cursor_position((
                chunks[0].x + 11 + self.chat_endpoint_draft.chars().count() as u16,
                chunks[0].y + 2,
            ));
        }
        let mut lines = Vec::new();
        for message in self.chat_history.records().iter().rev().take(14).rev() {
            let color = if message.role == "user" {
                theme::AMBER
            } else {
                theme::PHOSPHOR
            };
            lines.push(Line::from(Span::styled(
                format!("{}:", message.role),
                Style::default().fg(color).bold(),
            )));
            lines.push(Line::from(message.content.as_str()));
            lines.push(Line::from(""));
        }
        if !self.chat_streaming.is_empty() {
            lines.push(Line::from(Span::styled(
                "assistant (streaming):",
                Style::default().fg(theme::PHOSPHOR).bold(),
            )));
            lines.push(Line::from(self.chat_streaming.as_str()));
        }
        if lines.is_empty() {
            lines.push(Line::from("Press i or Enter to compose a message."));
            lines.push(Line::from(
                "Connect with Ctrl+G, detect with Ctrl+B, then choose a model.",
            ));
        }
        frame.render_widget(
            Paragraph::new(lines)
                .style(theme::text())
                .block(
                    theme::block()
                        .title("Conversation · Alt+S save · o sessions")
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: false }),
            chunks[1],
        );
        let marker = if self.chat_pending { " waiting…" } else { "" };
        frame.render_widget(
            Paragraph::new(format!("> {}{marker}", self.chat_input))
                .style(theme::text())
                .block(
                    theme::block()
                        .title("Message · Enter send · Esc stop/exit")
                        .borders(Borders::ALL),
                ),
            chunks[2],
        );
        if self.input_mode == InputMode::ChatMessage {
            frame.set_cursor_position((
                chunks[2].x + 3 + self.chat_input.chars().count() as u16,
                chunks[2].y + 1,
            ));
        }
    }

    fn draw_browser(&mut self, frame: &mut Frame, area: Rect) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Min(4),
            ])
            .split(area);
        frame.render_widget(
            Paragraph::new(self.browser_path.as_str())
                .style(theme::text())
                .block(
                    theme::block()
                        .title(format!(
                            "GGUF path · g edit · r scan · t {}",
                            if self.browser_recursive {
                                "recursive"
                            } else {
                                "top-level"
                            }
                        ))
                        .borders(Borders::ALL),
                ),
            outer[0],
        );
        if self.input_mode == InputMode::BrowserPath {
            frame.set_cursor_position((
                outer[0].x + 1 + self.browser_path.chars().count() as u16,
                outer[0].y + 1,
            ));
        }

        let files = self.visible_browser_files();
        let shown_size: u64 = files.iter().map(|file| file.size).sum();
        let warning_count = self
            .browser_files
            .iter()
            .filter(|file| file.parse_error.is_some())
            .count();
        frame.render_widget(
            Paragraph::new(format!(
                " / filter: {} │ sort: {} │ {} shown / {} total │ {} shown │ {} warning(s)",
                self.browser_filter,
                self.browser_sort.label(),
                files.len(),
                self.browser_files.len(),
                format_bytes(shown_size),
                warning_count,
            ))
            .style(theme::warning()),
            outer[1],
        );
        if self.input_mode == InputMode::BrowserFilter {
            frame.set_cursor_position((
                outer[1].x + 11 + self.browser_filter.chars().count() as u16,
                outer[1].y,
            ));
        }

        let main = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(outer[2]);
        let rows = files.iter().enumerate().map(|(index, file)| {
            let metadata = file.metadata.as_ref();
            Row::new(vec![
                Cell::from(index.to_string()),
                Cell::from(self.browser_relative_path(file)),
                Cell::from(file.quantization.clone()),
                Cell::from(format_bytes(file.size)),
                Cell::from(format_parameter_count(
                    metadata.and_then(|metadata| metadata.parameter_count),
                )),
                Cell::from(
                    metadata
                        .and_then(|metadata| metadata.architecture.clone())
                        .unwrap_or_else(|| "—".into()),
                ),
                Cell::from(format_system_time(file.modified)),
            ])
        });
        let table = Table::new(
            rows,
            [
                Constraint::Length(4),
                Constraint::Percentage(42),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(9),
                Constraint::Length(10),
                Constraint::Length(17),
            ],
        )
        .style(theme::text())
        .header(
            Row::new(["#", "GGUF", "Quant", "Size", "Params", "Arch", "Modified"])
                .style(theme::title()),
        )
        .block(
            theme::block()
                .title("Inventory · / filter · c cycle sort")
                .borders(Borders::ALL),
        )
        .row_highlight_style(theme::selected())
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, main[0], &mut self.browser_state);

        let details = self.selected_browser_file().map_or_else(
            || Text::from("No GGUF file selected."),
            |file| gguf_details(&file),
        );
        frame.render_widget(
            Paragraph::new(details)
                .style(theme::text())
                .block(theme::block().title("Selected GGUF").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            main[1],
        );
    }

    /// Model view: what the profile selected in the sidebar is, how it is
    /// served and benched, and (for native task profiles) the foreground task
    /// and its last result. The model list itself is the sidebar.
    fn draw_stack_catalog(&mut self, frame: &mut Frame, area: Rect) {
        if self.stack_catalog.is_none() {
            let message = self.stack_catalog_error.as_deref().unwrap_or(
                "Model manifests have not been loaded. Press r in the sidebar to refresh.",
            );
            frame.render_widget(
                Paragraph::new(message)
                    .style(theme::warning())
                    .block(theme::block().title("Model").borders(Borders::ALL))
                    .wrap(Wrap { trim: false }),
                area,
            );
            return;
        }
        let Some(profile) = self.selected_stack_profile().cloned() else {
            frame.render_widget(
                Paragraph::new("No model selected. Pick one in the sidebar.")
                    .style(theme::dim())
                    .block(theme::block().title("Model").borders(Borders::ALL)),
                area,
            );
            return;
        };
        #[cfg(target_os = "linux")]
        let show_direct_native = {
            let native_busy_or_uncertain =
                self.direct_native_job.is_some() || self.direct_native_admission_closed;
            let retained_native_summary = self.direct_native_summary.is_some();
            let selected_native_profile = self.selected_stack_profile().is_some_and(|profile| {
                self.stack_catalog
                    .as_ref()
                    .is_some_and(|catalog| is_direct_native_profile(profile, catalog))
            });
            native_busy_or_uncertain || retained_native_summary || selected_native_profile
        };
        #[cfg(not(target_os = "linux"))]
        let show_direct_native = false;
        #[cfg(target_os = "linux")]
        let direct_result_visible = self.direct_native_summary.as_ref().is_some_and(|summary| {
            matches!(summary.state, DirectNativeState::Succeeded)
                && self.selected_stack_profile().is_some_and(|profile| {
                    profile.key.lane == summary.lane
                        && profile.key.profile_id.as_str() == summary.profile_id
                        && self
                            .stack_catalog
                            .as_ref()
                            .is_some_and(|catalog| is_direct_native_profile(profile, catalog))
                })
        });
        #[cfg(not(target_os = "linux"))]
        let direct_result_visible = false;
        #[cfg(target_os = "linux")]
        let native_chat_guidance = "f opens Chat/task\nLanguage chat profiles open in the Chat view.\nSupported native tasks run foreground; unsupported profiles are refused.";
        #[cfg(not(target_os = "linux"))]
        let native_chat_guidance = "f opens Chat\nLanguage chat profiles open in the Chat view.\nNative task forms are unavailable here; unsupported profiles are refused.";
        #[cfg(target_os = "linux")]
        let (action_text, action_title) = if show_direct_native {
            let text = self.direct_native_summary.as_ref().map_or_else(
                || "One task runs at a time, in the foreground, through llama-swap.\nEnter opens the task form · Ctrl+Enter runs it · c cancels".to_owned(),
                |summary| {
                    let operator_note = if summary.native_id.is_some() {
                        "\nComfyUI shared engine · llama-swap by default"
                    } else if summary.requires_idle_gpu_window {
                        "\nBlender headless · shared GPU idle required before admission"
                    } else {
                        ""
                    };
                    let latency = summary.result.as_ref().and_then(direct_native_latency).map_or_else(
                        String::new,
                        |latency| format!("\nnative latency: {:.2} ms", latency.as_secs_f64() * 1000.0),
                    );
                    format!(
                        "Foreground job #{} · {}:{} · task {}\nstate: {}{}\n{}{}",
                        summary.job_id,
                        summary.lane.as_str(),
                        summary.profile_id,
                        summary.task,
                        summary.state.label(),
                        latency,
                        if self.direct_native_admission_closed && summary.native_id.is_some() {
                            "admission closed · Comfy completion unconfirmed; inspect endpoint before retrying"
                        } else if self.direct_native_admission_closed
                            && summary.requires_idle_gpu_window
                        {
                            "admission closed · shared GPU idle is unconfirmed; keep other clients quiesced and verify before retrying"
                        } else if self.direct_native_admission_closed && summary.lane == Lane::Language {
                            "admission closed · remote completion unconfirmed; verify endpoint before retrying"
                        } else if self.direct_native_admission_closed && summary.lane == Lane::Decisions {
                            "admission closed · decision completion unconfirmed; inspect Jobs and the declared backend before retrying"
                        } else if self.direct_native_admission_closed {
                            "admission closed · inspect Jobs and recover runner before retrying"
                        } else if self.direct_native_job.is_some() && summary.native_id.is_some() {
                            "c cancel · waits for Comfy terminal state; server remains externally managed"
                        } else if self.direct_native_job.is_some()
                            && summary.requires_idle_gpu_window
                        {
                            "c cancel · waits for runner cleanup and GPU release checks"
                        } else if self.direct_native_job.is_some() && summary.lane == Lane::Language {
                            "c cancel · dispatched remote work may continue"
                        } else if self.direct_native_job.is_some() && summary.lane == Lane::Decisions {
                            "c cancel · wait for the local runner or hosted decision outcome"
                        } else if self.direct_native_job.is_some() {
                            "c cancel · waits for local process cleanup"
                        } else {
                            "f run again · c only applies while active"
                        },
                        operator_note
                    )
                },
            );
            (text, "Foreground native task")
        } else {
            (native_chat_guidance.to_owned(), "Native / Chat guidance")
        };
        #[cfg(not(target_os = "linux"))]
        let (action_text, action_title) =
            (native_chat_guidance.to_owned(), "Native / Chat guidance");
        let facts_area = if show_direct_native {
            // Facts, then the task, then the result at full width (result
            // paths are long).
            let sections = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(40),
                    Constraint::Percentage(25),
                    Constraint::Percentage(35),
                ])
                .split(area);
            frame.render_widget(
                Paragraph::new(action_text)
                    .style(theme::text())
                    .block(theme::block().title(action_title).borders(Borders::ALL))
                    .wrap(Wrap { trim: false }),
                sections[1],
            );
            let output_text = if show_direct_native {
                if direct_result_visible {
                    self.direct_native_summary
                        .as_ref()
                        .and_then(|summary| summary.result.as_ref())
                        .map(direct_native_result_preview)
                        .unwrap_or_else(|| "Foreground native result is not available.".into())
                } else {
                    "Foreground native result is hidden until its completed profile is selected."
                        .into()
                }
            } else {
                "Foreground native results appear here after a supported task completes.".to_owned()
            };
            frame.render_widget(
                Paragraph::new(output_text)
                    .style(theme::text())
                    .block(
                        theme::block()
                            .title(if direct_result_visible {
                                "Foreground native result"
                            } else {
                                "Native task results"
                            })
                            .borders(Borders::ALL),
                    )
                    .wrap(Wrap { trim: false }),
                sections[2],
            );
            sections[0]
        } else {
            area
        };
        self.draw_model_facts(frame, facts_area, &profile);
    }

    /// `qwen38_mtp` → `llama.cpp-exp-stack-v2` when the `${qwen38_mtp_server}`
    /// macro points into `vendor/`; otherwise the stem itself.
    fn server_display(&self, stem: &str) -> String {
        self.yaml_macros
            .get(&format!("{stem}_server"))
            .map(|value| model_view::macro_display(value))
            .unwrap_or_else(|| stem.to_owned())
    }

    /// The llama-swap engine entry serving a media profile, by runtime.
    fn media_engine(&self, profile: &CatalogProfile) -> Option<&'static str> {
        if profile.key.lane != Lane::Media {
            return None;
        }
        let runtime = profile.runtime.as_str().to_ascii_lowercase();
        if runtime.contains("comfy") {
            Some("comfyui_auto")
        } else if runtime.contains("audio.cpp") {
            Some("audiocpp_auto")
        } else {
            None
        }
    }

    /// Router state of a profile: its own for language entries, its engine's
    /// for media profiles served through one.
    fn profile_router_state(&self, profile: &CatalogProfile) -> Option<&str> {
        let id = match profile.key.lane {
            Lane::Language => profile.key.profile_id.as_str(),
            _ => self.media_engine(profile)?,
        };
        self.models
            .iter()
            .find(|model| model.id == id)
            .map(|model| model.state.as_str())
    }

    fn draw_model_facts(&self, frame: &mut Frame, area: Rect, profile: &CatalogProfile) {
        let id = profile.key.profile_id.as_str();
        let entry = (profile.key.lane == Lane::Language)
            .then(|| self.yaml_entries.get(id))
            .flatten();
        let state = self.profile_router_state(profile);
        let tasks = profile
            .tasks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let home = env::var("HOME").unwrap_or_default();
        let short = |path: &str| -> String {
            if !home.is_empty() {
                if let Some(rest) = path.strip_prefix(&home) {
                    return format!("~{rest}");
                }
            }
            path.to_owned()
        };
        let mut facts: Vec<(&'static str, String)> = Vec::new();
        match entry {
            Some(entry) if entry.proxy => {
                facts.push((
                    "engine",
                    "proxied by llama-swap; media profiles route to it".into(),
                ));
            }
            Some(entry) => {
                let (server, flags) = model_view::cmd_facts(&entry.cmd);
                if let Some(server) = server {
                    facts.push(("server", self.server_display(&server)));
                }
                if let Some(path) = entry.model_path.as_deref() {
                    let size = self
                        .model_sizes
                        .get(id)
                        .map(|bytes| format!(" · {}", model_view::human_bytes(*bytes)))
                        .unwrap_or_else(|| " · file not found".into());
                    let file = path.rsplit('/').next().unwrap_or(path);
                    facts.push(("model", format!("{file}{size}")));
                }
                let mut config = None;
                let mut shown = Vec::new();
                for (label, value) in flags {
                    match label {
                        "config" => config = Some(value),
                        "override" => shown.push("-ot …".to_owned()),
                        "engine" => {}
                        _ => shown.push(format!("{label} {value}")),
                    }
                }
                if let Some(config) = config {
                    let shown = self
                        .yaml_macros
                        .get(&config)
                        .map(|value| model_view::macro_display(value))
                        .unwrap_or(config);
                    facts.push(("config", shown));
                }
                if !shown.is_empty() {
                    facts.push(("flags", shown.join(" · ")));
                }
                facts.push((
                    "idle",
                    entry
                        .ttl
                        .map(|ttl| format!("unloads after {}", model_view::human_ttl(ttl)))
                        .unwrap_or_else(|| "router default ttl".into()),
                ));
            }
            None => {
                if let Some(engine) = self.media_engine(profile) {
                    facts.push(("engine", format!("{engine} (state shown is the engine's)")));
                }
                facts.push(("runtime", profile.runtime.as_str().to_owned()));
                let device = match &profile.device {
                    Some(Device::Cpu) => "cpu",
                    Some(Device::Cuda) => "cuda",
                    Some(Device::Remote) => "remote",
                    Some(Device::Other(device)) => device.as_str(),
                    None => "not declared",
                };
                facts.push(("device", device.to_owned()));
                facts.push((
                    "manifest",
                    short(&profile.source_path.display().to_string()),
                ));
            }
        }
        if !tasks.is_empty() && !entry.is_some_and(|entry| entry.proxy) {
            facts.push(("tasks", tasks));
        }
        if profile.disabled {
            facts.push(("status", "disabled in llama-swap.yaml".into()));
        }
        let bench: Vec<model_view::BenchRow> = self
            .bench_runs
            .get(id)
            .map(|runs| {
                runs.iter()
                    .map(|run| {
                        let mut setup = Vec::new();
                        if !run.strategy.is_empty() {
                            setup.push(run.strategy.clone());
                        }
                        if !run.ctx.is_empty() {
                            setup.push(format!("ctx {}", run.ctx));
                        }
                        if !run.ctk.is_empty() || !run.ctv.is_empty() {
                            setup.push(format!("kv {}/{}", run.ctk, run.ctv));
                        }
                        model_view::BenchRow {
                            date: run.ts.chars().take(10).collect(),
                            tg: run.tg_ts,
                            pp: run.pp_ts,
                            setup: setup.join(" · "),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        if bench.is_empty() && entry.is_some_and(|entry| !entry.proxy) {
            facts.push((
                "bench",
                "no matching runs in bench-models/logs/results".into(),
            ));
        }
        let chat = profile.key.lane == Lane::Language
            && profile.tasks.len() == 1
            && profile.tasks.iter().any(|task| task.as_str() == "chat");
        let keys = match (profile.key.lane, entry) {
            (Lane::Language, Some(entry)) if entry.proxy => "L load · U unload".to_owned(),
            (Lane::Language, _) if chat => "Enter chat · L load · U unload · b bench".to_owned(),
            (Lane::Language, _) => "Enter task form · L load · U unload".to_owned(),
            _ => "Enter task form · c cancel a running task".to_owned(),
        };
        let data = model_view::ModelViewData {
            id,
            lane_style: lane_style(profile.key.lane),
            state,
            routed: profile.key.lane == Lane::Language || self.media_engine(profile).is_some(),
            title: entry
                .and_then(|entry| entry.name.as_deref())
                .or(Some(profile.name.as_str()).filter(|name| *name != id)),
            description: entry.and_then(|entry| entry.description.as_deref()),
            facts,
            bench: &bench,
            keys,
            focused: !self.sidebar_focused,
        };
        model_view::draw(frame, area, &data);
    }

    fn draw_task_form(&self, frame: &mut Frame) {
        let Some(editor) = self.task_form_editor.as_ref() else {
            return;
        };
        let area = centered_rect(92, 90, frame.area());
        frame.render_widget(Clear, area);
        let panes = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(6),
                Constraint::Length(3),
            ])
            .split(area);
        let form_hint = if editor.direct_blender_headless {
            "Blender headless · Python runs with your user permissions (no sandbox) · shared GPU must be idle before starting · GUI/MCP is external and not launched here"
        } else if editor.direct_native_h3 {
            "MiniMax H3 · music/tts via llama-swap (connect Chat); video needs idle GPU · JSON argv"
        } else if editor.direct_native_music {
            "HeartMuLa · llama-swap (connect Chat first) · JSON argv · prompt/lyrics stay in memory"
        } else if editor.direct_breeze_speech && is_routed_breeze_profile(&editor.profile_snapshot)
        {
            "Breeze via llama-swap · needs a confirmed Chat connection · tts/clon/design fields stay task-specific"
        } else if editor.direct_breeze_speech {
            "CPU Breeze · tts standard/streaming, clon reference, design instruction"
        } else if editor.direct_native_decision {
            if editor.profile_snapshot.runtime.as_str() == "llamacpp" {
                "Router-owned CUDA decision · normal router swapping may evict the language model · use a quiet window · no automatic restoration"
            } else {
                "Native decision · local CPU or hosted route"
            }
        } else if editor.direct_comfy_workflow {
            "ComfyUI shared engine · API-format workflow JSON · llama-swap by default"
        } else if editor.direct_language_embeddings {
            "language embedding batch · connected Chat endpoint"
        } else {
            "choose the task explicitly from its declared list"
        };
        frame.render_widget(
            Paragraph::new(format!("Profile: {} · {form_hint}", editor.profile))
                .style(theme::text())
                .block(
                    theme::block()
                        .title(
                            if editor.direct_native_decision
                                && editor.profile_snapshot.runtime.as_str() == "llamacpp"
                            {
                                "Router swap: may evict language model · quiet window"
                            } else {
                                "Native task form · in memory only"
                            },
                        )
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: false }),
            panes[0],
        );

        let lines = editor
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                let selected = index == editor.active;
                let marker = if selected { "▶ " } else { "  " };
                let label = format!("{}: ", field.label());
                let value_display = if selected && !field.is_selector() {
                    let value = editor.value(*field).map(String::as_str).unwrap_or("");
                    if value.is_empty() {
                        editor.field_display(*field)
                    } else {
                        let columns = (panes[1].width as usize)
                            .saturating_sub(4 + UnicodeWidthStr::width(label.as_str()));
                        task_form_text_window(value, editor.cursor_byte, columns).0
                    }
                } else {
                    editor.field_display(*field)
                };
                let mut spans = vec![
                    Span::styled(
                        marker,
                        if selected {
                            theme::border_focused()
                        } else {
                            Style::default()
                        },
                    ),
                    Span::styled(
                        label,
                        if selected {
                            theme::title()
                        } else {
                            theme::text()
                        },
                    ),
                    Span::raw(value_display),
                ];
                if selected && field.is_selector() {
                    spans.push(Span::styled("  ↑/↓ choose", theme::warning()));
                }
                Line::from(spans)
            })
            .collect::<Vec<_>>();
        let visible_rows = panes[1].height.saturating_sub(2) as usize;
        let scroll = if visible_rows == 0 {
            0
        } else {
            editor.active.saturating_sub(visible_rows.saturating_sub(1))
        };
        frame.render_widget(
            Paragraph::new(lines)
                .style(theme::text())
                .block(
                    theme::block()
                        .title("Fields · payload stays out of logs/history")
                        .borders(Borders::ALL),
                )
                .scroll((scroll.min(u16::MAX as usize) as u16, 0)),
            panes[1],
        );
        frame.render_widget(
            Paragraph::new(
                "Tab/Shift+Tab move fields · ↑/↓ choose task/kind/mode · Enter inserts newline · Ctrl+Enter runs once · Esc clears draft · c cancels active run",
            )
                .style(theme::warning())
                .block(theme::block().borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            panes[2],
        );

        let field = editor.active_field();
        if !field.is_selector()
            && field != TaskFormField::Run
            && panes[1].width > 2
            && panes[1].height > 2
        {
            let label_width = field.label().chars().count() as u16 + 4;
            let cursor_chars = editor
                .current_text()
                .map(|value| {
                    task_form_text_window(
                        value,
                        editor.cursor_byte,
                        (panes[1].width as usize).saturating_sub(2 + label_width as usize),
                    )
                    .1
                    .min(u16::MAX as usize) as u16
                })
                .unwrap_or(0);
            let body_visible_row = editor.active.saturating_sub(scroll);
            let cursor_x = panes[1]
                .x
                .saturating_add(1)
                .saturating_add(label_width)
                .saturating_add(cursor_chars)
                .min(panes[1].x.saturating_add(panes[1].width.saturating_sub(2)));
            let cursor_y = panes[1]
                .y
                .saturating_add(1)
                .saturating_add(body_visible_row.min(u16::MAX as usize) as u16)
                .min(panes[1].y.saturating_add(panes[1].height.saturating_sub(2)));
            frame.set_cursor_position((cursor_x, cursor_y));
        }
    }

    fn draw_download(&mut self, frame: &mut Frame, area: Rect) {
        let compact = area.width < 120 || area.height < 24;
        let section_constraints = if compact && area.height < 12 {
            [
                Constraint::Length(3),
                Constraint::Min(3),
                Constraint::Length(2),
            ]
        } else if compact {
            [
                Constraint::Length(6),
                Constraint::Min(3),
                Constraint::Length(3),
            ]
        } else {
            [
                Constraint::Length(6),
                Constraint::Min(6),
                Constraint::Length(4),
            ]
        };
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints(section_constraints)
            .split(area);

        if compact {
            render_download_compact_settings(frame, sections[0], &self.download);
        } else {
            let setting_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Length(3)])
                .split(sections[0]);
            let top = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
                .split(setting_rows[0]);
            let bottom = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(36),
                    Constraint::Percentage(16),
                    Constraint::Percentage(22),
                    Constraint::Percentage(26),
                ])
                .split(setting_rows[1]);

            render_download_text_field(
                frame,
                top[0],
                "Config path · Alt+O load",
                self.download.config_path_buffer(),
                self.download.focus() == DownloadFocus::ConfigPath,
            );
            let version = self.download.selected_version().unwrap_or("—");
            render_download_value_field(
                frame,
                top[1],
                "Snapshot · Alt+R restore",
                version,
                self.download.focus() == DownloadFocus::Versions,
            );
            render_download_text_field(
                frame,
                bottom[0],
                "Base models dir",
                self.download.base_models_dir_buffer(),
                self.download.focus() == DownloadFocus::BaseModelsDir,
            );
            render_download_value_field(
                frame,
                bottom[1],
                "Speed preset",
                if self.download.slow() {
                    "[x] slow"
                } else {
                    "[ ] slow"
                },
                self.download.focus() == DownloadFocus::SlowPreset,
            );
            render_download_text_field(
                frame,
                bottom[2],
                "Global workers",
                self.download.global_workers_buffer(),
                self.download.focus() == DownloadFocus::GlobalWorkers,
            );
            render_download_text_field(
                frame,
                bottom[3],
                "Save note",
                self.download.save_note_buffer(),
                self.download.focus() == DownloadFocus::SaveNote,
            );
        }

        let (table_area, editor_area) = if !compact || sections[1].width >= 72 {
            let panes = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(54), Constraint::Percentage(46)])
                .split(sections[1]);
            (Some(panes[0]), Some(panes[1]))
        } else if matches!(self.download.focus(), DownloadFocus::Model(_)) {
            (None, Some(sections[1]))
        } else {
            (Some(sections[1]), None)
        };

        let page_start = self.download_page.saturating_mul(DOWNLOAD_PAGE_SIZE);
        let rows = self
            .download
            .models()
            .iter()
            .enumerate()
            .skip(page_start)
            .take(DOWNLOAD_PAGE_SIZE)
            .map(|(index, model)| {
                if compact {
                    return Row::new(vec![
                        Cell::from((index + 1).to_string()),
                        Cell::from(if model.enabled { "yes" } else { "no" }),
                        Cell::from(model.repo_id.clone()),
                    ]);
                }
                let pattern = model
                    .allow_patterns
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "*".into());
                Row::new(vec![
                    Cell::from((index + 1).to_string()),
                    Cell::from(if model.enabled { "yes" } else { "no" }),
                    Cell::from(model.repo_id.clone()),
                    Cell::from(pattern),
                    Cell::from(model.local_dir.clone()),
                ])
            })
            .collect::<Vec<_>>();
        let state = if self.download_load_error.is_some() {
            " · BLOCKED"
        } else if self.download.is_dirty() {
            " · UNSAVED"
        } else {
            ""
        };
        let table_border = if self.download.focus() == DownloadFocus::Table {
            theme::border_focused()
        } else {
            Style::default()
        };
        let table_widths = if compact {
            vec![
                Constraint::Length(3),
                Constraint::Length(4),
                Constraint::Min(10),
            ]
        } else {
            vec![
                Constraint::Length(4),
                Constraint::Length(8),
                Constraint::Percentage(35),
                Constraint::Percentage(28),
                Constraint::Percentage(37),
            ]
        };
        let table_header = if compact {
            Row::new(["#", "On", "Repository"])
        } else {
            Row::new(["#", "Enabled", "Repository", "Pattern", "Local dir"])
        }
        .style(theme::title());
        let table_title = if compact {
            format!(
                "Models{state} · page {}/{} · [/ ] page · Space toggle · Alt+D/E download",
                self.download_page + 1,
                self.download_page_count()
            )
        } else {
            format!(
                "Models{state} · page {}/{} · [/ ] page · Space toggle · Alt+N/A/K CRUD · Alt+D/E download",
                self.download_page + 1,
                self.download_page_count()
            )
        };
        let table = Table::new(rows, table_widths)
            .style(theme::text())
            .header(table_header)
            .block(
                theme::block()
                    .title(table_title)
                    .borders(Borders::ALL)
                    .border_style(table_border),
            )
            .row_highlight_style(theme::selected())
            .highlight_symbol("▶ ");
        if let Some(table_area) = table_area {
            frame.render_stateful_widget(table, table_area, &mut self.download_state);
        }
        if let Some(editor_area) = editor_area {
            if compact {
                render_download_compact_model_editor(frame, editor_area, &self.download);
            } else {
                render_download_model_editor(frame, editor_area, &self.download);
            }
        }

        let log_lines = self
            .download_log
            .iter()
            .rev()
            .take(sections[2].height.saturating_sub(2) as usize)
            .rev()
            .map(|line| Line::from(line.as_str()))
            .collect::<Vec<_>>();
        let eta = self
            .download_eta_label()
            .map_or_else(String::new, |label| format!(" · {label}"));
        frame.render_widget(
            Paragraph::new(log_lines)
                .style(theme::text())
                .block(
                    theme::block()
                        .title(format!(
                            "Download activity · Alt+Y clear · {}{}",
                            self.download_disk_space, eta
                        ))
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: false }),
            sections[2],
        );
    }

    fn draw_jobs(&mut self, frame: &mut Frame, area: Rect) {
        let rows = self.job_history.records().iter().map(|job| {
            Row::new(vec![
                Cell::from(job.id.to_string()),
                Cell::from(job.kind.clone()),
                Cell::from(job.name.clone()),
                Cell::from(job.status.clone()),
                Cell::from(job.started_label.clone()),
                Cell::from(job.elapsed_label.clone()),
                Cell::from(job.exit_label.clone()),
            ])
        });
        let table = Table::new(
            rows,
            [
                Constraint::Length(5),
                Constraint::Length(12),
                Constraint::Percentage(45),
                Constraint::Length(11),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(8),
            ],
        )
        .style(theme::text())
        .header(
            Row::new(["ID", "Kind", "Name", "Status", "Started", "Elapsed", "Exit"])
                .style(theme::title()),
        )
        .block(
            theme::block()
                .title("Jobs · s stop · r retry · c clear")
                .borders(Borders::ALL),
        )
        .row_highlight_style(theme::selected())
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, area, &mut self.jobs_state);
    }

    fn draw_maintenance(&mut self, frame: &mut Frame, area: Rect) {
        let panes = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(area);
        let items = self
            .maintenance_scripts
            .iter()
            .map(|path| ListItem::new(relative_display(&self.root, path)))
            .collect::<Vec<_>>();
        let list = List::new(items)
            .style(theme::text())
            .block(
                theme::block()
                    .title("Maintenance · r/Enter run · s stop · Ctrl+U editor")
                    .borders(Borders::ALL),
            )
            .highlight_symbol("▶ ")
            .highlight_style(theme::title());
        // The editor reserves its bottom row for key hints; match that height so
        // both boxes end on the same row.
        let list_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(1)])
            .split(panes[0])[0];
        frame.render_stateful_widget(list, list_area, &mut self.maintenance_state);
        self.draw_script_editor(frame, panes[1], ScriptEditorTarget::Maintenance);
    }

    fn draw_log(&self, frame: &mut Frame, area: Rect) {
        let height = area.height.saturating_sub(2) as usize;
        let lines = self
            .log
            .iter()
            .rev()
            .take(height)
            .rev()
            .map(|line| Line::from(line.as_str()))
            .collect::<Vec<_>>();
        let lines = if lines.is_empty() {
            vec![Line::from(Span::styled("No activity yet.", theme::dim()))]
        } else {
            lines
        };
        frame.render_widget(
            Paragraph::new(lines)
                .style(theme::text())
                .block(theme::block().title("Log · c clear").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    /// Message line: the latest status, the active input mode and the keys
    /// for where focus is.
    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let input = if let Some(target) = self.script_input_target {
            match target {
                ScriptEditorTarget::Bench => "BENCH SCRIPT INPUT",
                ScriptEditorTarget::Maintenance => "MAINTENANCE SCRIPT INPUT",
            }
        } else if !self.sidebar_focused
            && self.view == View::Download
            && self.download.focus() != DownloadFocus::Table
        {
            "DOWNLOAD INPUT"
        } else {
            match self.input_mode {
                InputMode::Normal => "",
                InputMode::ModelFilter => "FILTER",
                InputMode::BenchFilter => "BENCH FILTER",
                InputMode::BrowserPath => "PATH INPUT",
                InputMode::BrowserFilter => "GGUF FILTER",
                InputMode::ChatMessage => "CHAT INPUT",
                InputMode::ChatEndpoint => "CHAT ENDPOINT",
                InputMode::ChatSystemPrompt => "SYSTEM PROMPT INPUT",
                InputMode::TaskForm => "TASK FORM",
            }
        };
        let keys = if self.script_input_target.is_some() {
            "Esc list · Ctrl+P palette"
        } else if self.input_mode == InputMode::TaskForm {
            "Tab fields · Ctrl+Enter run · Esc discard"
        } else if self.input_mode == InputMode::ModelFilter {
            "type to filter · Enter keep · Esc done"
        } else if self.sidebar_focused {
            "j/k · [/] · Enter open · L load · Tab pane"
        } else if self.view == View::Download {
            "Tab fields · Esc sidebar · Ctrl+P"
        } else {
            "Tab sidebar · F1–F9 · Ctrl+P · ? help"
        };
        let mut spans = Vec::new();
        if !input.is_empty() {
            spans.push(Span::styled(format!(" {input} "), theme::badge()));
        }
        // Key hints keep their place when there is room for them and at least
        // 24 columns of status; a long status is clipped instead.
        let width = area.width as usize;
        let badge: usize = spans.iter().map(|span| span.width()).sum();
        let keys_width = keys.width() + 2;
        let show_keys = width >= badge + keys_width + 24;
        let status_room = if show_keys {
            width - badge - keys_width
        } else {
            width.saturating_sub(badge)
        };
        let status = format!(" {}", self.status);
        let status = if status.width() > status_room {
            let mut clipped = String::new();
            for character in status.chars() {
                if clipped.width() + 2 > status_room {
                    break;
                }
                clipped.push(character);
            }
            clipped.push('…');
            clipped
        } else {
            status
        };
        let used = badge + status.width();
        spans.push(Span::styled(status, theme::text()));
        if show_keys {
            spans.push(Span::raw(
                " ".repeat(width.saturating_sub(used + keys_width - 1)),
            ));
            spans.push(Span::styled(format!("{keys} "), theme::dim()));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_help(&self, frame: &mut Frame) {
        let area = centered_rect(82, 82, frame.area());
        frame.render_widget(Clear, area);
        let context = self.command_context();
        let mut help = vec![
            Line::from(Span::styled("L3MS key bindings", theme::title())),
            Line::from(""),
            Line::from("Tab sidebar ↔ pane · F1–F9/Alt+1–9 views · Alt+←/→ cycle · Ctrl+P palette · q quit"),
            Line::from(""),
        ];
        let section = |help: &mut Vec<Line<'static>>,
                       title: String,
                       specs: Vec<&'static crate::commands::CommandSpec>| {
            help.push(Line::from(Span::styled(title, theme::title())));
            for spec in specs {
                help.push(Line::from(vec![
                    Span::styled(format!("  {:<20}", spec.shortcut), theme::border_focused()),
                    Span::raw(spec.label),
                ]));
            }
            help.push(Line::from(""));
        };
        section(
            &mut help,
            format!("{} view", self.view.name()),
            visible_commands(context)
                .into_iter()
                .filter(|spec| !spec.contexts.contains(&CommandContext::Global))
                .collect(),
        );
        section(
            &mut help,
            "Sidebar".into(),
            visible_commands(CommandContext::Sidebar)
                .into_iter()
                .filter(|spec| spec.contexts.contains(&CommandContext::Sidebar))
                .collect(),
        );
        let toolbar = toolbar::BUTTONS
            .iter()
            .map(|button| {
                let enabled = command(button.id).is_some_and(|spec| spec.is_visible());
                let label = command(button.id).map_or("", |spec| spec.label);
                Line::from(vec![
                    Span::styled(format!("  {:<20}", button.key), theme::border_focused()),
                    Span::styled(
                        format!(
                            "{} {label}{}",
                            button.glyph,
                            if enabled { "" } else { " (coming)" }
                        ),
                        if enabled { theme::text() } else { theme::dim() },
                    ),
                ])
            })
            .collect::<Vec<_>>();
        help.push(Line::from(Span::styled(
            "Toolbar (from the sidebar)",
            theme::title(),
        )));
        help.extend(toolbar);
        help.push(Line::from(""));
        help.push(Line::from("Press any key to close."));
        frame.render_widget(
            Paragraph::new(help)
                .style(theme::text())
                .block(theme::block().title(" Help ").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_palette(&mut self, frame: &mut Frame) {
        let area = centered_rect(94, 84, frame.area());
        frame.render_widget(Clear, area);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(4)])
            .split(area);
        frame.render_widget(
            Paragraph::new(self.palette_query.as_str())
                .style(theme::text())
                .block(
                    theme::block()
                        .title(" Command palette · type to search · Esc close ")
                        .borders(Borders::ALL),
                ),
            chunks[0],
        );
        let commands = self.palette_commands();
        let rows = commands
            .iter()
            .map(|spec| Row::new([Cell::from(spec.shortcut), Cell::from(spec.palette_label)]));
        let table = Table::new(rows, [Constraint::Length(22), Constraint::Percentage(100)])
            .block(
                theme::block()
                    .title(format!(" {} command(s) · Enter run ", commands.len()))
                    .borders(Borders::ALL),
            )
            .row_highlight_style(theme::selected())
            .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, chunks[1], &mut self.palette_state);
        frame.set_cursor_position((
            chunks[0].x + 1 + self.palette_query.chars().count() as u16,
            chunks[0].y + 1,
        ));
    }

    fn draw_chat_sessions(&mut self, frame: &mut Frame) {
        let area = centered_rect(78, 72, frame.area());
        frame.render_widget(Clear, area);
        let rows = self.chat_sessions.iter().map(|session| {
            Row::new([
                Cell::from(session.saved.clone()),
                Cell::from(session.message_count.to_string()),
                Cell::from(session.file_name.clone()),
            ])
        });
        let title = if self.chat_session_pending {
            " Chat sessions · loading… "
        } else {
            " Chat sessions · Enter load · r refresh · o/Esc close "
        };
        let table = Table::new(
            rows,
            [
                Constraint::Length(20),
                Constraint::Length(10),
                Constraint::Percentage(100),
            ],
        )
        .style(theme::text())
        .header(Row::new(["Saved", "Messages", "File"]).yellow().bold())
        .block(theme::block().title(title).borders(Borders::ALL))
        .row_highlight_style(theme::selected())
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, area, &mut self.chat_sessions_state);
    }

    fn draw_chat_prompts(&mut self, frame: &mut Frame) {
        let area = centered_rect(66, 58, frame.area());
        frame.render_widget(Clear, area);
        let items = self
            .chat_prompts
            .iter()
            .cloned()
            .map(ListItem::new)
            .collect::<Vec<_>>();
        let list = List::new(items)
            .style(theme::text())
            .block(theme::block().title(" System prompts · Enter load · l/Esc close "))
            .highlight_symbol("▶ ")
            .highlight_style(theme::title());
        frame.render_stateful_widget(list, area, &mut self.chat_prompts_state);
    }

    fn draw_script_versions(&mut self, frame: &mut Frame) {
        let Some(target) = self.script_versions_target else {
            return;
        };
        let area = centered_rect(72, 70, frame.area());
        frame.render_widget(Clear, area);
        let items = self
            .script_editor(target)
            .versions()
            .iter()
            .cloned()
            .map(ListItem::new)
            .collect::<Vec<_>>();
        let list = List::new(items)
            .style(theme::text())
            .block(
                theme::block()
                    .title(format!(
                        " {} snapshots · Enter restore · Esc/Alt+V close ",
                        target.label()
                    ))
                    .borders(Borders::ALL),
            )
            .highlight_symbol("▶ ")
            .highlight_style(theme::title());
        frame.render_stateful_widget(list, area, &mut self.script_version_state);
    }

    fn draw_quit_confirmation(&self, frame: &mut Frame) {
        let area = centered_rect(66, 24, frame.area());
        frame.render_widget(Clear, area);
        let dirty = [
            self.bench_editor.is_dirty().then_some("Bench"),
            self.maintenance_editor.is_dirty().then_some("Maintenance"),
            self.download.is_dirty().then_some("Download"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" and ");
        frame.render_widget(
            Paragraph::new(format!(
                "{dirty} edits are not saved.\n\nS  save snapshots, then quit\nD  discard edits and quit\nEsc  return to L3MS"
            ))
            .style(theme::text())
            .block(
                theme::block()
                    .title(" Unsaved changes ")
                    .borders(Borders::ALL)
                    .border_style(theme::warning()),
            )
            .wrap(Wrap { trim: false }),
            area,
        );
    }
}

/// Lane signature colour: marks where a profile lives, never its runtime state.
/// Bench runs per llama-swap entry, newest first. A result belongs to an entry only
/// when its GGUF path is exactly the entry's `-m` path; when several entries
/// share that GGUF (different flags), the result's `model_key` must also name
/// the entry. Anything else stays "not benched" rather than being guessed.
fn bench_runs_by_entry(
    entries: &[crate::swap_yaml::SwapYamlEntry],
    results: &[crate::bench_results::BenchResult],
) -> HashMap<String, Vec<crate::bench_results::BenchResult>> {
    let path_of =
        |entry: &crate::swap_yaml::SwapYamlEntry| entry.model_path.as_deref().map(expand_tilde);
    let mut rates = HashMap::new();
    for entry in entries {
        let Some(path) = path_of(entry) else {
            continue;
        };
        let shared = entries
            .iter()
            .filter(|other| path_of(other).as_deref() == Some(path.as_str()))
            .count()
            > 1;
        // Results are sorted newest first.
        let runs: Vec<_> = results
            .iter()
            .filter(|result| expand_tilde(&result.model) == path)
            .filter(|result| !shared || result.model_key == entry.id)
            .take(5)
            .cloned()
            .collect();
        if !runs.is_empty() {
            rates.insert(entry.id.clone(), runs);
        }
    }
    rates
}

fn lane_style(lane: Lane) -> Style {
    match lane {
        Lane::Language => theme::lane_language(),
        Lane::Media => theme::lane_media(),
        Lane::Decisions => theme::lane_decisions(),
    }
}

fn is_direct_native_decision_profile(profile: &CatalogProfile) -> bool {
    #[cfg(target_os = "linux")]
    {
        direct_run::is_supported_direct_decision_profile(profile)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = profile;
        false
    }
}

#[cfg(target_os = "linux")]
fn is_hosted_direct_decision_profile(profile: &CatalogProfile) -> bool {
    profile.runtime.as_str() == "jev" && direct_run::is_supported_direct_decision_profile(profile)
}

/// Native audio tasks run through llama-swap's /audiocpp/ route.
#[cfg(target_os = "linux")]
fn is_routed_speech_request(
    catalog: &Catalog,
    request: &crate::control::requests::InferenceRequest,
) -> bool {
    direct_run::is_routed_audio_script_request(catalog, request)
        || matches!(
            request,
            crate::control::requests::InferenceRequest::Media {
                input: crate::control::requests::MediaInput::Speech(_),
                ..
            }
        ) && catalog
            .inspect(request.profile())
            .is_some_and(direct_run::is_supported_cuda_breeze_profile)
}

#[cfg(target_os = "linux")]
fn is_direct_blender_headless_profile(profile: &CatalogProfile) -> bool {
    direct_run::is_supported_blender_headless_profile(profile)
        && profile.tasks.len() == 3
        && ["3d", "render", "image"]
            .iter()
            .all(|expected| profile.tasks.iter().any(|task| task.as_str() == *expected))
}

#[cfg(not(target_os = "linux"))]
fn is_direct_blender_headless_profile(_profile: &CatalogProfile) -> bool {
    false
}

#[cfg(target_os = "linux")]
fn is_direct_cuda_music_profile(profile: &CatalogProfile) -> bool {
    direct_run::is_supported_direct_music_profile(profile)
}

#[cfg(not(target_os = "linux"))]
fn is_direct_cuda_music_profile(_profile: &CatalogProfile) -> bool {
    false
}

#[cfg(target_os = "linux")]
fn is_direct_cuda_h3_profile(profile: &CatalogProfile) -> bool {
    direct_run::is_supported_direct_h3_profile(profile)
}

#[cfg(not(target_os = "linux"))]
fn is_direct_cuda_h3_profile(_profile: &CatalogProfile) -> bool {
    false
}

fn is_direct_cpu_speech_profile(profile: &CatalogProfile) -> bool {
    profile.key.lane == Lane::Media
        && profile.key.profile_id.as_str() == "breeze-tts-2-cpu"
        && profile.runtime.as_str() == "audio.cpp"
        && profile.device.as_ref() == Some(&Device::Cpu)
        && profile.tasks.iter().any(|task| task.as_str() == "tts")
        && profile.tasks.iter().any(|task| task.as_str() == "clon")
        && profile.tasks.iter().any(|task| task.as_str() == "design")
}

fn is_direct_breeze_speech_profile(profile: &CatalogProfile) -> bool {
    if is_direct_cpu_speech_profile(profile) {
        return true;
    }
    is_routed_breeze_profile(profile)
}

/// CUDA Breeze speech is served by llama-swap's `/audiocpp/` route.
#[cfg(target_os = "linux")]
fn is_routed_breeze_profile(profile: &CatalogProfile) -> bool {
    direct_run::is_supported_cuda_breeze_profile(profile)
}

#[cfg(not(target_os = "linux"))]
fn is_routed_breeze_profile(_profile: &CatalogProfile) -> bool {
    false
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "linux")]
fn is_direct_comfy_workflow_profile(profile: &CatalogProfile) -> bool {
    profile.key.lane == Lane::Media
        && !profile.disabled
        && profile.runtime.as_str() == "ComfyUI"
        && !profile.tasks.is_empty()
}

#[cfg(target_os = "linux")]
fn is_direct_native_language_embeddings_profile(
    profile: &CatalogProfile,
    catalog: &Catalog,
) -> bool {
    profile.key.lane == Lane::Language
        && !profile.disabled
        && profile
            .tasks
            .iter()
            .any(|task| task.as_str() == "embeddings")
        && resolve_profile_route(catalog, &profile.key)
            .is_ok_and(|route| matches!(route.native_route, NativeRoute::LlamaSwapModel { .. }))
}

#[cfg(target_os = "linux")]
fn is_direct_native_profile(profile: &CatalogProfile, catalog: &Catalog) -> bool {
    is_direct_native_decision_profile(profile)
        || is_direct_breeze_speech_profile(profile)
        || is_direct_cuda_music_profile(profile)
        || is_direct_cuda_h3_profile(profile)
        || is_direct_blender_headless_profile(profile)
        || is_direct_comfy_workflow_profile(profile)
        || is_direct_native_language_embeddings_profile(profile, catalog)
}

#[cfg(target_os = "linux")]
fn direct_native_task_label(task: &str) -> &'static str {
    match task {
        "decision" => "Native decision",
        "tts" => "CPU speech",
        "clon" => "CPU voice clone",
        "design" => "CPU voice design",
        "music" => "Music",
        "embeddings" => "Language embeddings",
        _ => "Foreground native task",
    }
}

#[cfg(target_os = "linux")]
fn direct_script_media_task_label(profile_id: &str, task: &str) -> &'static str {
    if profile_id == "blender-3d" {
        return match task {
            "3d" => "Blender headless Python",
            "render" | "image" => "Blender headless render",
            _ => "Blender headless task",
        };
    }
    if profile_id == "minimax-h3" {
        return match task {
            "music" => "CUDA H3 music",
            "tts" => "CUDA H3 tts",
            "video" => "CUDA H3 video",
            _ => "CUDA H3 media",
        };
    }
    if profile_id == "heartmula-music" && task == "music" {
        "HeartMuLa music"
    } else {
        "CUDA script media"
    }
}

#[cfg(target_os = "linux")]
fn direct_native_summary_label(summary: &DirectNativeSummary) -> &'static str {
    if summary.native_id.is_some() {
        "Comfy workflow"
    } else if summary.profile_id == "blender-3d" {
        direct_script_media_task_label(&summary.profile_id, &summary.task)
    } else {
        direct_native_task_label(&summary.task)
    }
}

#[cfg(target_os = "linux")]
fn is_direct_breeze_profile_id(profile_id: &str) -> bool {
    matches!(profile_id, "breeze-tts-2-cpu" | "breeze-tts-2")
}

#[cfg(target_os = "linux")]
fn is_direct_breeze_speech_request(request: &crate::control::requests::InferenceRequest) -> bool {
    matches!(request,
        crate::control::requests::InferenceRequest::Media {
            profile,
            input: crate::control::requests::MediaInput::Speech(input),
        } if profile.lane == Lane::Media
            && matches!(profile.profile_id.as_str(), "breeze-tts-2-cpu" | "breeze-tts-2")
            && matches!(
                (input.mode, request.task()),
                (SpeechMode::Standard | SpeechMode::Streaming, "tts")
                    | (SpeechMode::VoiceDesign, "design")
                    | (SpeechMode::VoiceClone, "clon")
            )
    )
}

#[cfg(target_os = "linux")]
fn is_direct_native_script_request(
    catalog: &Catalog,
    request: &crate::control::requests::InferenceRequest,
) -> bool {
    let crate::control::requests::InferenceRequest::Media {
        profile,
        input: crate::control::requests::MediaInput::Script { task, .. },
    } = request
    else {
        return false;
    };
    task.as_str() == request.task()
        && profile.lane == Lane::Media
        && catalog.inspect(profile).is_some_and(|profile| {
            (direct_run::is_supported_direct_h3_profile(profile)
                || direct_run::is_supported_direct_music_profile(profile)
                || is_direct_blender_headless_profile(profile))
                && profile
                    .tasks
                    .iter()
                    .any(|declared| declared.as_str() == task.as_str())
                && is_direct_script_media_task(profile.key.profile_id.as_str(), task.as_str())
        })
        && request.validate(catalog).is_ok()
}

#[cfg(target_os = "linux")]
fn is_direct_script_media_task(profile_id: &str, task: &str) -> bool {
    match profile_id {
        "heartmula-music" => task == "music",
        "minimax-h3" => matches!(task, "music" | "tts"),
        "blender-3d" => matches!(task, "3d" | "render" | "image"),
        _ => false,
    }
}

#[cfg(target_os = "linux")]
fn is_direct_comfy_workflow_request(
    catalog: &Catalog,
    request: &crate::control::requests::InferenceRequest,
) -> bool {
    let crate::control::requests::InferenceRequest::Media {
        profile: key,
        input: crate::control::requests::MediaInput::Workflow { task, .. },
    } = request
    else {
        return false;
    };
    let Some(profile) = catalog.inspect(key) else {
        return false;
    };
    is_direct_comfy_workflow_profile(profile)
        && request.validate(catalog).is_ok()
        && profile
            .tasks
            .iter()
            .any(|declared| declared.as_str() == task.as_str())
        && request.task() == task.as_str()
        && resolve_profile_route(catalog, key)
            .is_ok_and(|route| route.runtime.as_str() == "ComfyUI")
}

#[cfg(target_os = "linux")]
fn direct_native_latency(result: &DirectRunResult) -> Option<Duration> {
    match result {
        DirectRunResult::Decision(result) => result.native_latency,
        DirectRunResult::Speech(_) => None,
        DirectRunResult::ScriptMedia(_) => None,
        DirectRunResult::Embeddings(_) => None,
        DirectRunResult::Comfy(_) => None,
    }
}

#[cfg(target_os = "linux")]
fn direct_native_artifact_refs(data_root: &Path, result: &DirectRunResult) -> Option<Vec<String>> {
    let (output_root, artifact_paths) = match result {
        DirectRunResult::Speech(result) | DirectRunResult::ScriptMedia(result) => (
            result.output_root(),
            result
                .artifacts()
                .iter()
                .map(|artifact| artifact.relative_path())
                .collect::<Vec<_>>(),
        ),
        DirectRunResult::Comfy(result) => (
            result.output_root.as_path(),
            result
                .artifacts
                .iter()
                .map(|artifact| artifact.relative_path.as_path())
                .collect::<Vec<_>>(),
        ),
        DirectRunResult::Decision(_) | DirectRunResult::Embeddings(_) => return None,
    };
    let canonical_data_root = fs::canonicalize(data_root).ok()?;
    let canonical_output_root = fs::canonicalize(output_root).ok()?;
    if !canonical_output_root.starts_with(&canonical_data_root) {
        return None;
    }
    let prefix = canonical_output_root
        .strip_prefix(&canonical_data_root)
        .ok()?;
    if prefix.as_os_str().is_empty()
        || !prefix
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }
    artifact_paths
        .iter()
        .map(|relative| {
            if relative.as_os_str().is_empty()
                || relative.is_absolute()
                || !relative
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
            {
                return None;
            }
            prefix.join(*relative).to_str().map(str::to_owned)
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn direct_native_result_preview(result: &DirectRunResult) -> String {
    let rendered = match result {
        DirectRunResult::Decision(result) => match &result.result {
            crate::control::requests::InferenceResult::Decision { answers } => {
                serde_json::to_string(answers)
                    .unwrap_or_else(|_| "Decision output could not be rendered".into())
            }
            _ => "Unexpected native result lane".into(),
        },
        DirectRunResult::Speech(result) => result
            .artifacts()
            .iter()
            .map(|artifact| {
                let path = result.output_root().join(artifact.relative_path());
                let filename = artifact
                    .relative_path()
                    .file_name()
                    .unwrap_or(artifact.relative_path().as_os_str());
                format!(
                    "{:?}: {} · {}",
                    artifact.kind(),
                    filename.to_string_lossy(),
                    path.display()
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        DirectRunResult::ScriptMedia(result) => {
            if result.artifacts().is_empty() {
                "Script completed · no declared artifact".into()
            } else {
                result
                    .artifacts()
                    .iter()
                    .map(|artifact| {
                        format!(
                            "Script media artifact: {}",
                            result
                                .output_root()
                                .join(artifact.relative_path())
                                .display()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        DirectRunResult::Embeddings(payload) => {
            let dimensions = payload.vectors.first().map_or(0, Vec::len);
            let mut samples = payload
                .vectors
                .iter()
                .take(3)
                .enumerate()
                .map(|(index, vector)| {
                    let values = vector
                        .iter()
                        .take(8)
                        .map(|value| format!("{value:.4}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let suffix = if vector.len() > 8 { ", …" } else { "" };
                    format!("vector[{index}] [{values}{suffix}]")
                })
                .collect::<Vec<_>>();
            if payload.vectors.len() > 3 {
                samples.push(format!("… {} more vector(s)", payload.vectors.len() - 3));
            }
            format!(
                "Embedding batch · {} vector(s) × {dimensions} dimension(s)\nusage: prompt_tokens={} · total_tokens={}\n{}",
                payload.vectors.len(),
                payload.usage.prompt_tokens,
                payload.usage.total_tokens,
                samples.join("\n")
            )
        }
        DirectRunResult::Comfy(result) => result
            .artifacts
            .iter()
            .map(|artifact| {
                format!(
                    "{} · node {}: {}",
                    artifact.media_type,
                    artifact.node_id,
                    result.output_root.join(&artifact.relative_path).display()
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let rendered = rendered
        .chars()
        .map(|character| {
            if character.is_control() && character != '\n' && character != '\t' {
                '�'
            } else {
                character
            }
        })
        .collect::<String>();
    let mut preview = rendered
        .chars()
        .take(MAX_STACK_OUTPUT_PREVIEW_CHARS)
        .collect::<String>();
    if rendered.chars().count() > MAX_STACK_OUTPUT_PREVIEW_CHARS {
        preview.push_str("\n… output truncated");
    }
    preview
}

#[cfg(target_os = "linux")]
fn create_private_direct_output_directory(
    data_root: &Path,
    job_id: u64,
    prefix: &str,
) -> Result<PathBuf, ()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};

    static NEXT_DIRECTORY_NONCE: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    let data_metadata = fs::symlink_metadata(data_root).map_err(|_| ())?;
    let uid = unsafe { libc::geteuid() };
    if data_metadata.file_type().is_symlink()
        || !data_metadata.is_dir()
        || data_metadata.uid() != uid
        || data_metadata.mode() & 0o022 != 0
    {
        return Err(());
    }
    let canonical_data_root = fs::canonicalize(data_root).map_err(|_| ())?;
    let artifact_root = canonical_data_root.join("native-media");
    match fs::DirBuilder::new().mode(0o700).create(&artifact_root) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(_) => return Err(()),
    }
    let artifact_metadata = fs::symlink_metadata(&artifact_root).map_err(|_| ())?;
    if artifact_metadata.file_type().is_symlink()
        || !artifact_metadata.is_dir()
        || artifact_metadata.uid() != uid
        || artifact_metadata.permissions().mode() & 0o7777 != 0o700
    {
        return Err(());
    }
    let artifact_root = fs::canonicalize(&artifact_root).map_err(|_| ())?;
    if !artifact_root.starts_with(&canonical_data_root) {
        return Err(());
    }

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| ())?
        .as_nanos();
    if prefix.is_empty()
        || !prefix
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(());
    }
    for _ in 0..32 {
        let nonce = NEXT_DIRECTORY_NONCE.fetch_add(1, Ordering::Relaxed);
        let output_directory = artifact_root.join(format!("{prefix}-{job_id}-{timestamp}-{nonce}"));
        match fs::DirBuilder::new().mode(0o700).create(&output_directory) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(()),
        }
        let output_metadata = fs::symlink_metadata(&output_directory).map_err(|_| ())?;
        if output_metadata.file_type().is_symlink()
            || !output_metadata.is_dir()
            || output_metadata.uid() != uid
            || output_metadata.permissions().mode() & 0o7777 != 0o700
        {
            return Err(());
        }
        let output_directory = fs::canonicalize(&output_directory).map_err(|_| ())?;
        if !output_directory.starts_with(&artifact_root) {
            return Err(());
        }
        return Ok(output_directory);
    }
    Err(())
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(mut pending) = self.download_preflight_pending.take() {
            pending.cancellation.store(true, Ordering::Release);
            if let Some(worker) = pending.worker.take() {
                let deadline = Instant::now() + Duration::from_secs(2);
                while !worker.is_finished() && Instant::now() < deadline {
                    while self.receiver.try_recv().is_ok() {}
                    thread::sleep(Duration::from_millis(10));
                }
                if worker.is_finished() {
                    let _ = worker.join();
                }
            }
        }
        #[cfg(target_os = "linux")]
        if let Some(job) = self.direct_native_job.take() {
            // Once a child exists, the direct runner bounds process-group
            // cleanup with its five-second recovery policy. Join (rather than
            // detach) before dropping App so a model child cannot outlive this
            // foreground job owner. This is not a wall-clock bound on local
            // catalog preflight I/O.
            job.cancellation.store(true, Ordering::Release);
            let job_id = job.job_id;
            let joined = job.worker.join().is_ok();
            let completion = if joined {
                job.receiver.try_recv().ok()
            } else {
                None
            };
            self.apply_direct_native_completion(job_id, completion);
        }
        if self.running_process.is_some() {
            self.stop_running_process();
        }
    }
}

pub fn run_tui() -> Result<()> {
    let root = repository_root()?;
    let mut session = TerminalSession::new()?;
    App::new(root)?.run(session.terminal_mut())
}

struct TerminalSession {
    terminal: Tui,
}

impl TerminalSession {
    fn new() -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error).context("enter alternate screen");
        }
        let backend = CrosstermBackend::new(stdout);
        match Terminal::new(backend) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(error) => {
                let _ = execute!(io::stdout(), LeaveAlternateScreen);
                let _ = disable_raw_mode();
                Err(error).context("initialize terminal")
            }
        }
    }

    fn terminal_mut(&mut self) -> &mut Tui {
        &mut self.terminal
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

pub(crate) fn repository_root() -> Result<PathBuf> {
    if let Some(root) = env::var_os("L3MS_ROOT").filter(|value| !value.is_empty()) {
        return validate_root(PathBuf::from(root));
    }
    if let Ok(current) = env::current_dir() {
        for candidate in current.ancestors() {
            if is_repository_root(candidate) {
                return Ok(candidate.to_path_buf());
            }
        }
    }
    if let Ok(executable) = env::current_exe() {
        for candidate in executable.ancestors() {
            if is_repository_root(candidate) {
                return Ok(candidate.to_path_buf());
            }
        }
    }
    let development_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    validate_root(development_root)
}

fn validate_root(path: PathBuf) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("resolve L3MS root {}", path.display()))?;
    anyhow::ensure!(
        is_repository_root(&path),
        "{} is not an L3MS repository root",
        path.display()
    );
    Ok(path)
}

fn is_repository_root(path: &Path) -> bool {
    path.join("llama-swap.yaml").is_file() && path.join("bench-models").is_dir()
}

fn collect_shell_scripts(directory: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut scripts = fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().is_some_and(|extension| extension == "sh")
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with(prefix))
        })
        .collect::<Vec<_>>();
    scripts.sort();
    scripts
}

fn filter_bench_scripts<'a>(scripts: &'a [PathBuf], root: &Path, filter: &str) -> Vec<&'a PathBuf> {
    let filter = filter.trim().to_ascii_lowercase();
    scripts
        .iter()
        .filter(|script| {
            filter.is_empty()
                || script
                    .strip_prefix(root)
                    .unwrap_or(script)
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .contains(&filter)
                || pretty_name(script).to_ascii_lowercase().contains(&filter)
        })
        .collect()
}

fn stream_reader(
    reader: impl io::Read + Send + 'static,
    job_id: u64,
    sender: mpsc::SyncSender<BackgroundEvent>,
) {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = Vec::with_capacity(1024);
        let mut truncated = false;
        loop {
            let available = match reader.fill_buf() {
                Ok([]) => break,
                Ok(available) => available,
                Err(_) => return,
            };
            let newline = available.iter().position(|byte| *byte == b'\n');
            let consumed = newline.map_or(available.len(), |index| index + 1);
            let content = &available[..consumed];
            if line.len() < MAX_PROCESS_LINE_BYTES {
                let remaining = MAX_PROCESS_LINE_BYTES - line.len();
                let content_without_newline = content.strip_suffix(b"\n").unwrap_or(content);
                let keep = content_without_newline.len().min(remaining);
                line.extend_from_slice(&content_without_newline[..keep]);
                truncated |= keep < content_without_newline.len();
            } else {
                truncated = true;
            }
            reader.consume(consumed);

            if newline.is_some() {
                let mut rendered = String::from_utf8_lossy(&line).into_owned();
                if truncated {
                    rendered.push_str(" … [line truncated]");
                }
                if sender
                    .send(BackgroundEvent::ProcessLine {
                        job_id,
                        line: rendered,
                    })
                    .is_err()
                {
                    return;
                }
                line.clear();
                truncated = false;
            }
        }
        if !line.is_empty() || truncated {
            let mut rendered = String::from_utf8_lossy(&line).into_owned();
            if truncated {
                rendered.push_str(" … [line truncated]");
            }
            let _ = sender.send(BackgroundEvent::ProcessLine {
                job_id,
                line: rendered,
            });
        }
    });
}

fn append_bounded_text(target: &mut String, addition: &str, max_bytes: usize) {
    target.push_str(addition);
    if target.len() <= max_bytes {
        return;
    }
    if max_bytes == 0 {
        target.clear();
        return;
    }
    let mut remove = target.len() - max_bytes;
    while remove < target.len() && !target.is_char_boundary(remove) {
        remove += 1;
    }
    target.drain(..remove);
}

#[cfg(unix)]
fn signal_process_group(process_group: u32, signal: &str) -> std::result::Result<(), String> {
    let status = {
        #[cfg(test)]
        let _fixture_guard = crate::control::process::executable_fixture_guard();
        Command::new("/bin/kill")
            .args([
                format!("-{signal}"),
                "--".into(),
                format!("-{process_group}"),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
    }
    .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "kill -{signal} for process group {process_group} exited with {status}"
        ))
    }
}

fn build_selected_download_command(
    root: &Path,
    config: &DownloadConfig,
    index: usize,
    speed_args: &[String],
) -> Result<(String, Vec<OsString>)> {
    let model = config
        .models
        .get(index)
        .with_context(|| format!("download model index {index} is out of range"))?;
    anyhow::ensure!(
        !model.repo_id.trim().is_empty(),
        "selected model has no repo_id"
    );

    let mut parts = downloader_command_prefix(root);
    parts.push("--repo-id".into());
    parts.push(model.repo_id.clone().into());
    if !model.allow_patterns.is_empty() {
        parts.push("--allow-patterns".into());
        parts.extend(model.allow_patterns.iter().map(OsString::from));
    }
    if !model.ignore_patterns.is_empty() {
        parts.push("--ignore-patterns".into());
        parts.extend(model.ignore_patterns.iter().map(OsString::from));
    }
    if !model.local_dir.is_empty() {
        parts.push("--local-dir".into());
        parts.push(model.local_dir.clone().into());
    } else if !config.base_models_dir.trim().is_empty() {
        parts.push("--base-models-dir".into());
        parts.push(config.base_models_dir.trim().into());
    }
    if !model.revision.is_empty() {
        parts.push("--revision".into());
        parts.push(model.revision.clone().into());
    }
    if model.force_download {
        parts.push("--force-download".into());
    }
    parts.extend(speed_args.iter().map(OsString::from));
    Ok((model.repo_id.clone(), parts))
}

fn selected_download_target(root: &Path, config: &DownloadConfig, index: usize) -> Result<PathBuf> {
    let model = config
        .models
        .get(index)
        .with_context(|| format!("download model index {index} is out of range"))?;
    if !model.local_dir.trim().is_empty() {
        return Ok(resolve_runtime_download_path(root, model.local_dir.trim()));
    }
    Ok(config_download_target(root, config))
}

fn config_download_target(root: &Path, config: &DownloadConfig) -> PathBuf {
    let base = config.base_models_dir.trim();
    if base.is_empty() {
        root.join("model_downloader/models")
    } else {
        resolve_runtime_download_path(root, base)
    }
}

fn resolve_runtime_download_path(root: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn display_name(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn value_or_dash(value: &str) -> &str {
    if value.trim().is_empty() {
        "—"
    } else {
        value
    }
}

fn text_buffer_cursor_display_width(buffer: &TextBuffer) -> usize {
    let before_cursor = &buffer.content()[..buffer.cursor_byte()];
    let current_line = before_cursor
        .rsplit_once('\n')
        .map_or(before_cursor, |(_, line)| line);
    UnicodeWidthStr::width(current_line)
}

fn visible_cursor_offset(cursor: usize, rendered_scroll: u16, visible_extent: u16) -> Option<u16> {
    let offset = cursor.checked_sub(rendered_scroll as usize)?;
    (offset < visible_extent as usize).then_some(offset as u16)
}

fn download_compact_focus_style(focused: bool) -> Style {
    if focused {
        theme::badge()
    } else {
        Style::default()
    }
}

fn download_compact_field_areas(area: Rect, label: &str) -> (Rect, Rect, String) {
    let prefix = format!("{label}: ");
    let label_width = UnicodeWidthStr::width(prefix.as_str()).min(area.width as usize) as u16;
    let label_area = Rect::new(area.x, area.y, label_width, area.height);
    let value_area = Rect::new(
        area.x.saturating_add(label_width),
        area.y,
        area.width.saturating_sub(label_width),
        area.height,
    );
    (label_area, value_area, prefix)
}

fn render_download_compact_value_field(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    value: &str,
    focused: bool,
) {
    if area.is_empty() {
        return;
    }
    let style = download_compact_focus_style(focused);
    let (label_area, value_area, prefix) = download_compact_field_areas(area, label);
    frame.render_widget(Paragraph::new(prefix).style(style), label_area);
    if !value_area.is_empty() {
        frame.render_widget(Paragraph::new(value).style(style), value_area);
    }
}

fn render_download_compact_text_field(
    frame: &mut Frame,
    area: Rect,
    label: &str,
    buffer: &TextBuffer,
    focused: bool,
) {
    if area.is_empty() {
        return;
    }
    let style = download_compact_focus_style(focused);
    let (label_area, value_area, prefix) = download_compact_field_areas(area, label);
    frame.render_widget(Paragraph::new(prefix).style(style), label_area);
    if value_area.is_empty() {
        return;
    }

    let visible_width = value_area.width as usize;
    let cursor_column = text_buffer_cursor_display_width(buffer);
    let logical_scroll = if focused {
        cursor_column.saturating_sub(visible_width.saturating_sub(1))
    } else {
        0
    };
    let rendered_scroll = logical_scroll.min(u16::MAX as usize) as u16;
    frame.render_widget(
        Paragraph::new(buffer.content())
            .style(style)
            .scroll((0, rendered_scroll)),
        value_area,
    );
    if focused {
        let visible_column = cursor_column
            .saturating_sub(rendered_scroll as usize)
            .min(visible_width.saturating_sub(1));
        frame.set_cursor_position((value_area.x + visible_column as u16, value_area.y));
    }
}

fn render_download_compact_settings(frame: &mut Frame, area: Rect, download: &DownloadUiState) {
    let settings_focused = matches!(
        download.focus(),
        DownloadFocus::ConfigPath
            | DownloadFocus::Versions
            | DownloadFocus::BaseModelsDir
            | DownloadFocus::SlowPreset
            | DownloadFocus::GlobalWorkers
            | DownloadFocus::SaveNote
    );
    let block = theme::block()
        .title("Settings · Tab/Shift+Tab")
        .borders(Borders::ALL)
        .border_style(if settings_focused {
            theme::border_focused()
        } else {
            Style::default()
        });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }

    if inner.height < 4 {
        match download.focus() {
            DownloadFocus::Versions => render_download_compact_value_field(
                frame,
                inner,
                "Snapshot",
                download.selected_version().unwrap_or("—"),
                true,
            ),
            DownloadFocus::BaseModelsDir => render_download_compact_text_field(
                frame,
                inner,
                "Base",
                download.base_models_dir_buffer(),
                true,
            ),
            DownloadFocus::SlowPreset => render_download_compact_value_field(
                frame,
                inner,
                "Slow",
                if download.slow() {
                    "[x] slow"
                } else {
                    "[ ] slow"
                },
                true,
            ),
            DownloadFocus::GlobalWorkers => render_download_compact_text_field(
                frame,
                inner,
                "Workers",
                download.global_workers_buffer(),
                true,
            ),
            DownloadFocus::SaveNote => render_download_compact_text_field(
                frame,
                inner,
                "Note",
                download.save_note_buffer(),
                true,
            ),
            DownloadFocus::Table | DownloadFocus::ConfigPath | DownloadFocus::Model(_) => {
                render_download_compact_text_field(
                    frame,
                    inner,
                    "Config",
                    download.config_path_buffer(),
                    download.focus() == DownloadFocus::ConfigPath,
                );
            }
        }
        return;
    }

    let row = |offset| Rect::new(inner.x, inner.y + offset, inner.width, 1);
    render_download_compact_text_field(
        frame,
        row(0),
        "Config",
        download.config_path_buffer(),
        download.focus() == DownloadFocus::ConfigPath,
    );
    render_download_compact_value_field(
        frame,
        row(1),
        "Snapshot",
        download.selected_version().unwrap_or("—"),
        download.focus() == DownloadFocus::Versions,
    );
    render_download_compact_text_field(
        frame,
        row(2),
        "Base",
        download.base_models_dir_buffer(),
        download.focus() == DownloadFocus::BaseModelsDir,
    );
    let runtime = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(24),
            Constraint::Percentage(34),
            Constraint::Percentage(42),
        ])
        .split(row(3));
    render_download_compact_value_field(
        frame,
        runtime[0],
        "Slow",
        if download.slow() {
            "[x] slow"
        } else {
            "[ ] slow"
        },
        download.focus() == DownloadFocus::SlowPreset,
    );
    render_download_compact_text_field(
        frame,
        runtime[1],
        "Workers",
        download.global_workers_buffer(),
        download.focus() == DownloadFocus::GlobalWorkers,
    );
    render_download_compact_text_field(
        frame,
        runtime[2],
        "Note",
        download.save_note_buffer(),
        download.focus() == DownloadFocus::SaveNote,
    );
}

fn render_download_model_editor(frame: &mut Frame, area: Rect, download: &DownloadUiState) {
    let editor_lines = ModelField::ALL
        .iter()
        .map(|field| {
            let value = match field {
                ModelField::Enabled | ModelField::ForceDownload => download
                    .draft()
                    .boolean(*field)
                    .map(|value| if value { "[x]" } else { "[ ]" })
                    .unwrap_or("[ ]")
                    .to_owned(),
                _ => download
                    .draft()
                    .buffer(*field)
                    .map(|buffer| buffer.content().to_owned())
                    .unwrap_or_default(),
            };
            let line = format!("{}: {value}", field.label());
            if download.focus() == DownloadFocus::Model(*field) {
                Line::styled(line, download_compact_focus_style(true))
            } else {
                Line::from(line)
            }
        })
        .collect::<Vec<_>>();
    let editor_border = if matches!(download.focus(), DownloadFocus::Model(_)) {
        theme::border_focused()
    } else {
        Style::default()
    };
    frame.render_widget(
        Paragraph::new(editor_lines)
            .style(theme::text())
            .block(
                theme::block()
                    .title("Model editor · Alt+I focus · Tab/Shift+Tab fields")
                    .borders(Borders::ALL)
                    .border_style(editor_border),
            )
            .wrap(Wrap { trim: false }),
        area,
    );

    if let DownloadFocus::Model(field) = download.focus() {
        if let Some(buffer) = download.draft().buffer(field) {
            let line = ModelField::ALL
                .iter()
                .position(|candidate| *candidate == field)
                .unwrap_or_default();
            let inner_height = area.height.saturating_sub(2) as usize;
            let inner_width = area.width.saturating_sub(2) as usize;
            if line < inner_height && inner_width > 0 {
                let prefix = UnicodeWidthStr::width(format!("{}: ", field.label()).as_str());
                let column = prefix + text_buffer_cursor_display_width(buffer);
                let x = area.x + 1 + column.min(inner_width.saturating_sub(1)) as u16;
                let y = area.y + 1 + line as u16;
                frame.set_cursor_position((x, y));
            }
        }
    }
}

fn render_download_compact_model_editor(frame: &mut Frame, area: Rect, download: &DownloadUiState) {
    let editor_focused = matches!(download.focus(), DownloadFocus::Model(_));
    let block = theme::block()
        .title("Model editor · Tab/Shift+Tab")
        .borders(Borders::ALL)
        .border_style(if editor_focused {
            theme::border_focused()
        } else {
            Style::default()
        });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let visible_rows = (inner.height as usize).min(ModelField::ALL.len());
    if visible_rows == 0 {
        return;
    }

    let focused_index = match download.focus() {
        DownloadFocus::Model(field) => ModelField::ALL
            .iter()
            .position(|candidate| *candidate == field),
        _ => None,
    };
    let max_start = ModelField::ALL.len().saturating_sub(visible_rows);
    let start = focused_index
        .map(|index| index.saturating_sub(visible_rows.saturating_sub(1)))
        .unwrap_or_default()
        .min(max_start);

    for (offset, field) in ModelField::ALL
        .iter()
        .skip(start)
        .take(visible_rows)
        .enumerate()
    {
        let row = Rect::new(inner.x, inner.y + offset as u16, inner.width, 1);
        let focused = download.focus() == DownloadFocus::Model(*field);
        if field.is_boolean() {
            let value = download
                .draft()
                .boolean(*field)
                .map(|value| if value { "[x]" } else { "[ ]" })
                .unwrap_or("[ ]");
            render_download_compact_value_field(frame, row, field.label(), value, focused);
        } else if let Some(buffer) = download.draft().buffer(*field) {
            render_download_compact_text_field(frame, row, field.label(), buffer, focused);
        }
    }
}

fn render_download_value_field(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    value: &str,
    focused: bool,
) {
    let border = if focused {
        theme::border_focused()
    } else {
        Style::default()
    };
    frame.render_widget(
        Paragraph::new(value)
            .block(
                theme::block()
                    .title(title)
                    .borders(Borders::ALL)
                    .border_style(border),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_download_text_field(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    buffer: &TextBuffer,
    focused: bool,
) {
    let border = if focused {
        theme::border_focused()
    } else {
        Style::default()
    };
    let inner_width = area.width.saturating_sub(2) as usize;
    let cursor_column = text_buffer_cursor_display_width(buffer);
    let logical_scroll = if focused && inner_width > 0 {
        cursor_column.saturating_sub(inner_width.saturating_sub(1))
    } else {
        0
    };
    let rendered_scroll = logical_scroll.min(u16::MAX as usize) as u16;
    frame.render_widget(
        Paragraph::new(buffer.content())
            .block(
                theme::block()
                    .title(title)
                    .borders(Borders::ALL)
                    .border_style(border),
            )
            .scroll((0, rendered_scroll)),
        area,
    );
    if focused && area.height > 2 && inner_width > 0 {
        let visible_column = cursor_column
            .saturating_sub(rendered_scroll as usize)
            .min(inner_width.saturating_sub(1));
        frame.set_cursor_position((area.x + 1 + visible_column as u16, area.y + 1));
    }
}

fn edit_single_line_buffer(buffer: &mut TextBuffer, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char(character)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            buffer.insert_char(character);
            true
        }
        KeyCode::Backspace => {
            buffer.backspace();
            true
        }
        KeyCode::Delete => {
            buffer.delete_forward();
            true
        }
        KeyCode::Left => {
            buffer.move_left();
            true
        }
        KeyCode::Right => {
            buffer.move_right();
            true
        }
        KeyCode::Home => {
            buffer.move_home();
            true
        }
        KeyCode::End => {
            buffer.move_end();
            true
        }
        _ => false,
    }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3600 {
        format!("{}h {:02}m", seconds / 3600, (seconds / 60) % 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn parse_download_progress_line(line: &str) -> Option<DownloadProgressSnapshot> {
    let percent_index = line.find('%')?;
    let percent_start = line[..percent_index]
        .char_indices()
        .rev()
        .take_while(|(_, character)| character.is_ascii_digit())
        .last()
        .map_or(percent_index, |(index, _)| index);
    let percent = line[percent_start..percent_index].parse::<u64>().ok()?;
    if percent > 100 {
        return None;
    }
    let progress_part = line[percent_index + 1..].split('[').next()?;
    let size_token = progress_part
        .split_whitespace()
        .find(|token| token.contains('/'))?;
    let (downloaded, total) = size_token.split_once('/')?;
    let downloaded_bytes = parse_progress_size(downloaded)?;
    let total_bytes = parse_progress_size(total)?;
    let speed_bytes_per_second = line
        .split_whitespace()
        .find_map(|token| {
            token
                .trim_matches(|character: char| matches!(character, ',' | ']' | ')'))
                .strip_suffix("/s")
        })
        .and_then(parse_progress_size)
        .map(|bytes| bytes as f64);
    Some(DownloadProgressSnapshot {
        downloaded_bytes,
        total_bytes,
        speed_bytes_per_second,
    })
}

fn parse_progress_size(value: &str) -> Option<u64> {
    let value = value
        .trim_matches(|character: char| matches!(character, '|' | ',' | ')' | ']' | '\r' | '\n'));
    if value.is_empty() {
        return None;
    }
    let split_at = value
        .char_indices()
        .find(|(_, character)| !character.is_ascii_digit() && *character != '.')
        .map_or(value.len(), |(index, _)| index);
    let number = value[..split_at].parse::<f64>().ok()?;
    if !number.is_finite() || number < 0.0 {
        return None;
    }
    let suffix = value[split_at..].trim().to_ascii_lowercase();
    let multiplier = match suffix.as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0_f64.powi(2),
        "g" | "gb" | "gib" => 1024.0_f64.powi(3),
        "t" | "tb" | "tib" => 1024.0_f64.powi(4),
        _ => return None,
    };
    let bytes = number * multiplier;
    (bytes <= u64::MAX as f64).then(|| bytes.round() as u64)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_disk_space(target: &Path, disk_space: DiskSpace) -> String {
    format!(
        "Disk: {} / {} free [{}]",
        format_bytes(disk_space.free_bytes),
        format_bytes(disk_space.total_bytes),
        target.display()
    )
}

fn next_request_id(current: u64) -> u64 {
    let next = current.wrapping_add(1);
    if next == 0 {
        1
    } else {
        next
    }
}

fn format_parameter_count(parameters: Option<u64>) -> String {
    let Some(parameters) = parameters.filter(|parameters| *parameters > 0) else {
        return "—".into();
    };
    if parameters >= 1_000_000_000 {
        format!("{:.1}B", parameters as f64 / 1_000_000_000.0)
    } else if parameters >= 1_000_000 {
        format!("{:.1}M", parameters as f64 / 1_000_000.0)
    } else if parameters >= 1_000 {
        format!("{:.1}K", parameters as f64 / 1_000.0)
    } else {
        parameters.to_string()
    }
}

fn format_system_time(time: Option<SystemTime>) -> String {
    let Some(seconds) = time
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
    else {
        return "—".into();
    };
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
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
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}Z")
}

fn gguf_details(file: &GgufFile) -> Text<'static> {
    let metadata = file.metadata.as_ref();
    let value = |value: Option<&str>| {
        value
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("—")
            .to_owned()
    };
    let mut lines = vec![
        Line::from(file.path.display().to_string()),
        Line::from(""),
        Line::from(format!(
            "model: {}",
            value(metadata.and_then(|metadata| metadata.name.as_deref()))
        )),
        Line::from(format!("quantization: {}", file.quantization)),
        Line::from(format!(
            "size: {} ({} bytes)",
            format_bytes(file.size),
            file.size
        )),
        Line::from(format!(
            "architecture: {}",
            value(metadata.and_then(|metadata| metadata.architecture.as_deref()))
        )),
        Line::from(format!(
            "parameters: {}",
            format_parameter_count(metadata.and_then(|metadata| metadata.parameter_count))
        )),
        Line::from(format!("modified: {}", format_system_time(file.modified))),
        Line::from(format!(
            "gguf version: {}",
            metadata
                .map(|metadata| metadata.version.to_string())
                .unwrap_or_else(|| "—".into())
        )),
        Line::from(format!(
            "tensor count: {}",
            metadata
                .map(|metadata| metadata.tensor_count.to_string())
                .unwrap_or_else(|| "—".into())
        )),
        Line::from(format!(
            "file type id: {}",
            metadata
                .and_then(|metadata| metadata.file_type)
                .map(|file_type| file_type.to_string())
                .unwrap_or_else(|| "—".into())
        )),
        Line::from(format!(
            "tokenizer: {}",
            value(metadata.and_then(|metadata| metadata.tokenizer_model.as_deref()))
        )),
    ];
    if let Some(error) = &file.parse_error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("parse warning: {error}"),
            theme::warning(),
        )));
    }
    Text::from(lines)
}

fn expand_tilde(value: &str) -> String {
    if value == "~" {
        return env::var("HOME").unwrap_or_else(|_| value.into());
    }
    if let Some(rest) = value.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return Path::new(&home).join(rest).display().to_string();
        }
    }
    value.into()
}

fn clamp_table_selection(state: &mut TableState, count: usize) {
    if count == 0 {
        state.select(None);
    } else if state.selected().is_none_or(|index| index >= count) {
        state.select(Some(0));
    }
}

fn clamp_list_selection(state: &mut ListState, count: usize) {
    if count == 0 {
        state.select(None);
    } else if state.selected().is_none_or(|index| index >= count) {
        state.select(Some(0));
    }
}

fn select_previous_table(state: &mut TableState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    let next = match state.selected() {
        Some(0) | None => count - 1,
        Some(index) => index - 1,
    };
    state.select(Some(next));
}

fn select_next_table(state: &mut TableState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    state.select(Some(
        state.selected().map_or(0, |index| (index + 1) % count),
    ));
}

fn select_previous_list(state: &mut ListState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    let next = match state.selected() {
        Some(0) | None => count - 1,
        Some(index) => index - 1,
    };
    state.select(Some(next));
}

fn select_next_list(state: &mut ListState, count: usize) {
    if count == 0 {
        state.select(None);
        return;
    }
    state.select(Some(
        state.selected().map_or(0, |index| (index + 1) % count),
    ));
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_store::{load_config_strict, ModelConfig};
    use crate::control::catalog::Readiness;
    use ratatui::backend::TestBackend;
    use tempfile::TempDir;

    #[test]
    fn parses_huggingface_progress_and_speed() {
        let progress = parse_download_progress_line(
            "weights.safetensors:  25%|██▌       | 1.0GiB/4.0GiB [00:10<00:30, 32.0MiB/s]",
        )
        .unwrap();
        assert_eq!(progress.downloaded_bytes, 1 << 30);
        assert_eq!(progress.total_bytes, 4 << 30);
        assert_eq!(
            progress.speed_bytes_per_second,
            Some(32.0 * (1 << 20) as f64)
        );
    }

    struct AppFixture {
        _repository: TempDir,
        data: TempDir,
        root: PathBuf,
        bench_first: PathBuf,
        bench_second: PathBuf,
    }

    impl AppFixture {
        fn new() -> Self {
            let repository = TempDir::new().unwrap();
            let data = TempDir::new().unwrap();
            let root = repository.path().join("repo");
            let bench = root.join("bench-models");
            let maintenance = root.join("maintenance");
            let downloader = root.join("model_downloader");
            fs::create_dir_all(&bench).unwrap();
            fs::create_dir_all(&maintenance).unwrap();
            fs::create_dir_all(&downloader).unwrap();
            fs::write(root.join("llama-swap.yaml"), "models: {}\n").unwrap();
            let bench_first = bench.join("bench-a.sh");
            let bench_second = bench.join("bench-b.sh");
            fs::write(&bench_first, "#!/bin/sh\necho a\n").unwrap();
            fs::write(&bench_second, "#!/bin/sh\necho b\n").unwrap();
            fs::write(maintenance.join("cleanup.sh"), "#!/bin/sh\necho cleanup\n").unwrap();
            fs::write(
                downloader.join("models_config.json"),
                concat!(
                    "{\"base_models_dir\":\"/models\",\"models\":[",
                    "{\"enabled\":true,\"repo_id\":\"org/one\",\"description\":\"first\"},",
                    "{\"enabled\":true,\"repo_id\":\"org/two\",\"description\":\"second\"}",
                    "]}\n"
                ),
            )
            .unwrap();
            let downloader_script = downloader.join("download_hf_model.py");
            fs::write(
                &downloader_script,
                concat!(
                    "#!/usr/bin/env python3\n",
                    "import json\n",
                    "import sys\n",
                    "if '--estimate-json' in sys.argv:\n",
                    "    print(json.dumps({",
                    "'schema_version': 1, ",
                    "'models': [{'repo_id': 'org/one', 'revision': 'main', ",
                    "'matched_files': 2, 'total_bytes': 300, ",
                    "'download_bytes': 200, 'cached_bytes': 100}], ",
                    "'totals': {'models': 1, 'matched_files': 2, ",
                    "'total_bytes': 300, 'download_bytes': 200, 'cached_bytes': 100}}))\n",
                    "else:\n",
                    "    for argument in sys.argv[1:]:\n",
                    "        print(argument)\n",
                ),
            )
            .unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut permissions = fs::metadata(&downloader_script).unwrap().permissions();
                permissions.set_mode(0o755);
                fs::set_permissions(&downloader_script, permissions).unwrap();
            }
            let root = fs::canonicalize(root).unwrap();
            Self {
                _repository: repository,
                data,
                bench_first: root.join("bench-models/bench-a.sh"),
                bench_second: root.join("bench-models/bench-b.sh"),
                root,
            }
        }

        /// An app with the pane focused, so view tests see their view at any
        /// width. Sidebar tests focus the sidebar themselves.
        fn app(&self) -> App {
            let mut app = App::new_in(self.root.clone(), self.data.path().to_path_buf()).unwrap();
            app.sidebar_focused = false;
            app
        }

        fn app_without_model_refresh(&self) -> App {
            App::new_in_without_model_refresh(self.root.clone(), self.data.path().to_path_buf())
                .unwrap()
        }
    }

    fn wait_for_download(app: &mut App) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while (app.download_preflight_pending.is_some() || app.running_process.is_some())
            && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        app.drain_background_events();
        assert!(
            app.download_preflight_pending.is_none(),
            "test download preflight did not finish"
        );
        assert!(
            app.running_process.is_none(),
            "test download did not finish"
        );
    }

    fn terminal_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn terminal_row(terminal: &Terminal<TestBackend>, y: u16) -> String {
        let buffer = terminal.backend().buffer();
        let width = buffer.area.width as usize;
        let start = y as usize * width;
        buffer.content()[start..start + width]
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn write_stack_fixture(fixture: &AppFixture) {
        fs::write(
            fixture.root.join("llama-swap.yaml"),
            "models:\n  local-chat:\n    cmd: server\n",
        )
        .unwrap();
        let media = fixture.root.to_path_buf();
        let decisions = fixture.root.to_path_buf();
        fs::create_dir_all(&media).unwrap();
        fs::create_dir_all(&decisions).unwrap();
        fs::write(media.join("media-runtimes.json"), r#"{"schema_version":1,"profiles":[{"id":"breeze","name":"Speech fixture","runtime":"audio.cpp","tasks":["tts","clon","design"],"script":"media-models/speech.sh"}]}"#).unwrap();
        fs::write(decisions.join("decision-runtimes.json"), r#"{"version":"1.0","profiles":[{"id":"nagi","name":"Decision fixture","engine":"nagi","type":"local","model":"org/nagi","device":"cpu","entrypoint":"decision-models/runner.py"}]}"#).unwrap();
    }

    fn write_hosted_jev_decision_stack_fixture(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[{"id":"jev-api","name":"Hosted Jev SystemOne fixture","engine":"jev","tier":"hosted","type":"api","model":"typesafe/jev-latest","device":"remote","entrypoint":"decision-models/runner.py"}]}"#,
        )
        .unwrap();
    }

    #[cfg(target_os = "linux")]
    fn write_llamacpp_decision_stack_fixture(fixture: &AppFixture, model_id: &str) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("llama-swap.yaml"),
            format!(
                "models:\n  {model_id}:\n    metadata:\n      task: decision\n    cmd: server\n"
            ),
        )
        .unwrap();
        fs::write(
            fixture.root.join("decision-runtimes.json"),
            format!(
                r#"{{"version":"1.0","profiles":[{{"id":"{model_id}","name":"llama.cpp decision fixture","engine":"llamacpp","type":"local","model":"{model_id}","device":"cuda","tasks":["decision"],"entrypoint":"decision-models/runner.py"}}]}}"#
            ),
        )
        .unwrap();
    }

    fn write_embedding_stack_fixture(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("llama-swap.yaml"),
            "models:\n  local-embeddings:\n    cmd: server --embedding\n",
        )
        .unwrap();
    }

    fn write_cuda_decision_stack_fixture(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[{"id":"nagi-cuda","name":"CUDA decision fixture","engine":"nagi","type":"local","model":"org/nagi","device":"cuda","entrypoint":"decision-models/runner.py"}]}"#,
        )
        .unwrap();
    }

    fn write_coordinator_decision_stack_fixture(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[{"id":"semif-fixture","name":"Coordinator decision fixture","engine":"semif","type":"local","model":"org/semif","device":"cpu","entrypoint":"decision-models/runner.py"}]}"#,
        )
        .unwrap();
    }

    #[cfg(target_os = "linux")]
    fn write_cuda_breeze_media_manifest(fixture: &AppFixture) {
        fs::write(
            fixture.root.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"breeze-tts-2","name":"CUDA Breeze fixture","runtime":"audio.cpp","device":"cuda","tasks":["tts","clon","design"],"script":"media-models/generate-breeze-tts-2.sh"}]}"#,
        )
        .unwrap();
    }

    #[cfg(target_os = "linux")]
    fn write_heartmula_music_media_manifest(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"heartmula-music","name":"HeartMuLa music fixture","runtime":"audio.cpp","device":"cuda","tasks":["music"],"script":"media-models/generate-heartmula-music.sh"}]}"#,
        )
        .unwrap();
    }

    #[cfg(target_os = "linux")]
    fn write_minimax_h3_media_manifest(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"minimax-h3","name":"MiniMax H3 fixture","runtime":"audio.cpp","device":"cuda","tasks":["music","video","tts"],"script":"media-models/generate-minimax-h3.sh"}]}"#,
        )
        .unwrap();
    }

    #[cfg(target_os = "linux")]
    fn write_blender_headless_media_manifest(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"blender-3d","name":"Blender fixture","runtime":"Blender 5.2 / blender-mcp","tasks":["3d","render","image"],"script":"media-models/generate-blender-3d.sh"}]}"#,
        )
        .unwrap();
    }

    #[cfg(target_os = "linux")]
    fn write_comfy_stack_fixture(fixture: &AppFixture) {
        write_stack_fixture(fixture);
        fs::write(
            fixture.root.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"comfy-workflow","name":"Comfy workflow fixture","runtime":"ComfyUI","tasks":["image"],"script":"media-models/workflow.json"}]}"#,
        )
        .unwrap();
    }

    fn focus_stack_profile_by_id(app: &mut App, profile_id: &str) {
        app.handle_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE))
            .unwrap();
        let index = app
            .stack_catalog
            .as_ref()
            .unwrap()
            .list()
            .iter()
            .position(|profile| profile.key.profile_id.as_str() == profile_id)
            .unwrap();
        app.stack_catalog_state.select(Some(index));
    }

    fn connect_test_chat_at(
        app: &mut App,
        endpoint: &str,
        api_key: Option<&str>,
        models: Vec<SwapModel>,
    ) {
        app.chat_endpoint_draft = endpoint.into();
        app.chat_endpoint_committed = Some(endpoint.into());
        app.chat_client = Some(ChatClient::new(endpoint, api_key).unwrap());
        app.chat_models = models;
        app.chat_non_chat_model_ids.clear();
        app.chat_model_state.select(Some(0));
    }

    fn connect_test_chat(app: &mut App, models: Vec<SwapModel>) {
        connect_test_chat_at(app, "http://connected.example", None, models);
    }

    fn test_chat_model(id: &str, state: &str, disabled: bool) -> SwapModel {
        SwapModel {
            id: id.into(),
            state: state.into(),
            name: String::new(),
            description: String::new(),
            created: None,
            disabled,
            size_bytes: None,
        }
    }

    #[cfg(target_os = "linux")]
    fn write_direct_decision_runner(fixture: &AppFixture, marker: &Path, wait: bool) {
        use std::os::unix::fs::PermissionsExt;

        let decisions = fixture.root.to_path_buf();
        fs::create_dir_all(decisions.join(".venv-decisions/bin")).unwrap();
        fs::create_dir_all(decisions.join("decision-models")).unwrap();
        let guard = crate::control::process::executable_fixture_guard();
        let interpreter = decisions.join(".venv-decisions/bin/python");
        fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
        let response = serde_json::json!({
            "engine": "nagi-native",
            "answers": {"route": {"choice": "yes"}},
            "latency_ms": 1.25,
            "usage": {"fixture_count": 1}
        })
        .to_string();
        let body = if wait {
            let late_marker = marker.with_extension("late");
            format!(
                "#!/bin/sh\ncat >/dev/null\nprintf started > '{}'\nsleep 30\nprintf late > '{}'\nprintf '%s\\n' '{}'\n",
                marker.display(),
                late_marker.display(),
                response.replace('\'', "'\\''")
            )
        } else {
            format!(
                "#!/bin/sh\ncat >/dev/null\nprintf started > '{}'\nprintf '%s\\n' '{}'\n",
                marker.display(),
                response.replace('\'', "'\\''")
            )
        };
        let runner = decisions.join("decision-models/runner.py");
        fs::write(&runner, body).unwrap();
        fs::set_permissions(&runner, fs::Permissions::from_mode(0o700)).unwrap();
        drop(guard);
    }

    #[cfg(target_os = "linux")]
    fn write_direct_cpu_tts_runner(fixture: &AppFixture, marker: &Path, wait: bool) {
        use std::os::unix::fs::PermissionsExt;

        let media = fixture.root.to_path_buf();
        fs::create_dir_all(media.join("media-models")).unwrap();
        fs::write(
            media.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[{"id":"breeze-tts-2-cpu","name":"CPU Breeze fixture","runtime":"audio.cpp","device":"cpu","tasks":["tts","clon","design"],"script":"media-models/generate-breeze-tts-2.sh"}]}"#,
        )
        .unwrap();
        let late_marker = marker.with_extension("late");
        let marker_command = format!("printf started > '{}'\n", marker.display());
        let capture_args_command = format!("printf '%s\\n' \"$@\" > '{}.args'\n", marker.display());
        let wait_command = if wait {
            format!("sleep 30\nprintf late > '{}'\n", late_marker.display())
        } else {
            String::new()
        };
        let wrapper = format!(
            "#!/bin/bash\nset -eu\noutput=\nmode=offline\n{capture_args_command}while (($#)); do\n  case $1 in\n    --output) output=$2; shift 2 ;;\n    --mode) mode=$2; shift 2 ;;\n    *) shift ;;\n  esac\ndone\n[ -n \"$output\" ]\n{marker_command}{wait_command}printf '\\x52\\x49\\x46\\x46\\x26\\x00\\x00\\x00\\x57\\x41\\x56\\x45\\x66\\x6d\\x74\\x20\\x10\\x00\\x00\\x00\\x01\\x00\\x01\\x00\\x40\\x1f\\x00\\x00\\x80\\x3e\\x00\\x00\\x02\\x00\\x10\\x00\\x64\\x61\\x74\\x61\\x02\\x00\\x00\\x00\\x00\\x00' > \"$output\"\nif [[ $mode == streaming ]]; then\n  mkdir \"$output.chunks\"\n  printf '\\x52\\x49\\x46\\x46\\x26\\x00\\x00\\x00\\x57\\x41\\x56\\x45\\x66\\x6d\\x74\\x20\\x10\\x00\\x00\\x00\\x01\\x00\\x01\\x00\\x40\\x1f\\x00\\x00\\x80\\x3e\\x00\\x00\\x02\\x00\\x10\\x00\\x64\\x61\\x74\\x61\\x02\\x00\\x00\\x00\\x00\\x00' > \"$output.chunks/chunk-0001.wav\"\nfi\n"
        );
        let script = media.join("media-models/generate-breeze-tts-2.sh");
        fs::write(&script, wrapper).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[cfg(target_os = "linux")]
    fn open_direct_cpu_speech_form(app: &mut App, text: &str) {
        app.handle_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE))
            .unwrap();
        let index = app
            .stack_catalog
            .as_ref()
            .unwrap()
            .list()
            .iter()
            .position(|profile| profile.key.profile_id.as_str() == "breeze-tts-2-cpu")
            .unwrap();
        app.stack_catalog_state.select(Some(index));
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.direct_breeze_speech);
        assert!(!editor.fields.contains(&TaskFormField::MediaOperation));
        assert!(editor.fields.contains(&TaskFormField::SpeechMode));
        assert!(!editor.fields.contains(&TaskFormField::SpeechInstruction));
        assert!(!editor.fields.contains(&TaskFormField::SpeechVoiceReference));
        assert!(!editor.fields.contains(&TaskFormField::SpeechReferenceText));
        assert_eq!(editor.values.timeout, "30000");
        assert_eq!(editor.declared_tasks, ["tts", "clon", "design"]);
        assert_eq!(editor.selected_task, Some(0));
        assert!(editor.fields.contains(&TaskFormField::DeclaredTask));
        editor.values.speech_text = text.into();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn breeze_form_is_routed_and_cuda_decisions_are_no_longer_offered() {
        let fixture = AppFixture::new();
        write_cuda_decision_stack_fixture(&fixture);
        write_cuda_breeze_media_manifest(&fixture);
        let mut app = fixture.app();

        // CUDA Nagi/Laya are no longer run from the root workbench.
        focus_stack_profile_by_id(&mut app, "nagi-cuda");
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.task_form_editor.is_none());
        assert!(
            app.status
                .contains("No direct native workflow is available for this profile"),
            "{}",
            app.status
        );
        assert!(app.job_history.records().is_empty());

        app.task_form_editor = None;
        app.input_mode = InputMode::Normal;
        focus_stack_profile_by_id(&mut app, "breeze-tts-2");
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.direct_breeze_speech);
        assert_eq!(editor.declared_tasks, ["tts", "clon", "design"]);
        assert_eq!(editor.selected_task, Some(0));
        assert!(!editor.fields.contains(&TaskFormField::MediaOperation));
        assert!(editor.fields.contains(&TaskFormField::SpeechMode));
        assert_eq!(editor.speech_mode, Some(SpeechMode::Standard));
        editor.values.speech_text = "cuda speech text".into();
        editor.active = editor
            .fields
            .iter()
            .position(|field| *field == TaskFormField::SpeechMode)
            .unwrap();
        assert!(editor.cycle_selector(1));
        assert_eq!(editor.speech_mode, Some(SpeechMode::Streaming));
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let streaming = build_direct_native(&catalog, editor.to_draft().unwrap())
            .unwrap()
            .0;
        assert!(matches!(
            streaming,
            crate::control::requests::InferenceRequest::Media {
                input: crate::control::requests::MediaInput::Speech(
                    crate::control::requests::SpeechInput {
                        mode: SpeechMode::Streaming,
                        ..
                    }
                ),
                ..
            }
        ));

        editor.selected_task = Some(2);
        editor.refresh_fields();
        editor.values.speech_instruction = "private CUDA design instruction".into();
        editor.values.speech_voice_reference = "hidden-clone-reference.wav".into();
        assert!(editor.fields.contains(&TaskFormField::SpeechInstruction));
        assert!(!editor.fields.contains(&TaskFormField::SpeechVoiceReference));
        let design = build_direct_native(&catalog, editor.to_draft().unwrap())
            .unwrap()
            .0;
        let crate::control::requests::InferenceRequest::Media {
            input: crate::control::requests::MediaInput::Speech(input),
            ..
        } = design
        else {
            panic!("CUDA Breeze design selection builds a native speech request")
        };
        assert_eq!(input.mode, SpeechMode::VoiceDesign);
        assert_eq!(
            input.instruction.as_deref(),
            Some("private CUDA design instruction")
        );
        assert!(input.voice_reference.is_none());
        assert!(!app.status.contains("private CUDA design instruction"));
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("private CUDA design instruction")));
        assert!(app.direct_native_job.is_none());

        // Admission gating still applies to the routed form.
        app.chat_pending = true;
        app.run_direct_native();
        assert!(app.status.contains("Wait for"), "{}", app.status);
        assert!(app.direct_native_job.is_none());
        assert!(app.job_history.records().is_empty());
        app.chat_pending = false;
        app.direct_native_admission_closed = true;
        app.begin_model_action("fixture-chat".into(), true);
        assert!(!app.model_action_pending);
        assert!(app.status.contains("Finish or recover foreground"));
        app.direct_native_admission_closed = false;
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn routed_breeze_refuses_to_run_without_a_confirmed_router_connection() {
        let fixture = AppFixture::new();
        write_cuda_decision_stack_fixture(&fixture);
        write_cuda_breeze_media_manifest(&fixture);
        let mut app = fixture.app();
        focus_stack_profile_by_id(&mut app, "breeze-tts-2");
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.status.contains("llama-swap"), "{}", app.status);
        let editor = app.task_form_editor.as_mut().unwrap();
        editor.values.speech_text = "routed speech".into();
        app.run_direct_native();
        assert!(
            app.status
                .contains("Connect Chat before running native speech"),
            "{}",
            app.status
        );
        assert!(app.direct_native_job.is_none());
        assert!(app.job_history.records().is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn heartmula_form_keeps_argv_ephemeral_and_uses_no_coordinator() {
        let fixture = AppFixture::new();
        write_heartmula_music_media_manifest(&fixture);
        let mut app = fixture.app();

        focus_stack_profile_by_id(&mut app, "heartmula-music");
        let profile = app.selected_stack_profile().unwrap();
        assert!(is_direct_cuda_music_profile(profile));
        assert!(
            !direct_run::requires_operator_gpu_window(profile),
            "HeartMuLa music is routed through llama-swap, not an operator GPU window"
        );
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();

        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.direct_native_music);
        assert!(editor.is_direct_native());
        assert_eq!(editor.declared_tasks, ["music"]);
        assert_eq!(editor.selected_task, Some(0));
        assert!(!editor.fields.contains(&TaskFormField::DeclaredTask));
        assert!(!editor.fields.contains(&TaskFormField::MediaOperation));
        assert!(editor.fields.contains(&TaskFormField::MediaScriptArgv));
        editor.values.script_argv =
            r#"["--prompt","private music prompt","--lyrics","private lyrics"]"#.into();
        let draft = editor.to_draft().unwrap();
        assert!(!format!("{draft:?}").contains("private music prompt"));
        assert!(!format!("{draft:?}").contains("private lyrics"));
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let (request, _) = build_direct_native(&catalog, draft).unwrap();
        let crate::control::requests::InferenceRequest::Media {
            profile,
            input: crate::control::requests::MediaInput::Script { task, args },
        } = request
        else {
            panic!("HeartMuLa form builds a native media argv request")
        };
        assert_eq!(profile.profile_id.as_str(), "heartmula-music");
        assert_eq!(task.as_str(), "music");
        assert_eq!(
            args,
            [
                "--prompt",
                "private music prompt",
                "--lyrics",
                "private lyrics"
            ]
        );

        assert!(app.status.contains("llama-swap"));
        assert!(!app.status.contains("private music prompt"));
        assert!(!app.status.contains("private lyrics"));
        assert!(
            app.log
                .iter()
                .all(|line| !line.contains("private music prompt")
                    && !line.contains("private lyrics"))
        );
        assert!(app.job_history.records().is_empty());
        assert!(app.direct_native_job.is_none());
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap_or_default();
        assert!(!persisted.contains("private music prompt"));
        assert!(!persisted.contains("private lyrics"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn minimax_h3_form_keeps_all_script_tasks_ephemeral_and_native() {
        let fixture = AppFixture::new();
        write_minimax_h3_media_manifest(&fixture);
        let mut app = fixture.app();

        focus_stack_profile_by_id(&mut app, "minimax-h3");
        let profile = app.selected_stack_profile().unwrap();
        assert!(is_direct_cuda_h3_profile(profile));
        assert!(
            !direct_run::requires_operator_gpu_window(profile),
            "H3 music/tts are routed; H3 video is no longer offered"
        );
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();

        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        {
            let editor = app.task_form_editor.as_ref().unwrap();
            assert!(editor.direct_native_h3);
            assert!(!editor.direct_native_music);
            assert!(editor.is_direct_native());
            assert_eq!(editor.declared_tasks, ["music", "tts"]);
            assert!(editor.fields.contains(&TaskFormField::DeclaredTask));
            assert!(!editor.fields.contains(&TaskFormField::MediaOperation));
            assert!(editor.fields.contains(&TaskFormField::MediaScriptArgv));
            assert!(!editor.fields.contains(&TaskFormField::SpeechText));
            assert!(!editor.fields.contains(&TaskFormField::SpeechMode));
            assert!(!editor.fields.contains(&TaskFormField::SpeechInstruction));
            assert!(!editor.fields.contains(&TaskFormField::SpeechVoiceReference));
            assert!(!editor.fields.contains(&TaskFormField::SpeechReferenceText));
            assert!(!editor.fields.contains(&TaskFormField::MediaWorkflowGraph));
        }

        for task in ["music", "tts"] {
            select_task_form_declared_task(&mut app, task);
            let editor = app.task_form_editor.as_mut().unwrap();
            editor.values.script_argv = r#"["--prompt","private H3 input","--seed","42"]"#.into();
            let draft = editor.to_draft().unwrap();
            assert!(!format!("{draft:?}").contains("private H3 input"));
            let (request, _) = build_direct_native(&catalog, draft).unwrap();
            assert!(is_direct_native_script_request(&catalog, &request));
            let crate::control::requests::InferenceRequest::Media {
                profile,
                input:
                    crate::control::requests::MediaInput::Script {
                        task: built_task,
                        args,
                    },
            } = request
            else {
                panic!("MiniMax H3 form builds a native script argv request")
            };
            assert_eq!(profile.profile_id.as_str(), "minimax-h3");
            assert_eq!(built_task.as_str(), task);
            assert_eq!(args, ["--prompt", "private H3 input", "--seed", "42"]);
            assert!(editor.fields.contains(&TaskFormField::DeclaredTask));
            assert!(editor.fields.contains(&TaskFormField::MediaScriptArgv));
        }

        assert!(app.status.contains("music/tts use llama-swap"));
        assert!(!app.status.contains("private H3 input"));
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("private H3 input")));
        assert!(app.job_history.records().is_empty());
        assert!(app.direct_native_job.is_none());
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap_or_default();
        assert!(!persisted.contains("private H3 input"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn blender_headless_form_is_non_cuda_and_keeps_python_argv_private() {
        let fixture = AppFixture::new();
        write_blender_headless_media_manifest(&fixture);
        let mut app = fixture.app();

        focus_stack_profile_by_id(&mut app, "blender-3d");
        let profile = app.selected_stack_profile().unwrap();
        assert!(is_direct_blender_headless_profile(profile));
        assert!(direct_run::requires_operator_gpu_window(profile));
        let summary = DirectNativeSummary {
            job_id: 1,
            lane: Lane::Media,
            profile_id: "blender-3d".into(),
            task: "3d".into(),
            requires_idle_gpu_window: true,
            native_id: None,
            state: DirectNativeState::Running,
            result: None,
        };
        assert_eq!(
            direct_native_summary_label(&summary),
            "Blender headless Python"
        );
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();

        assert!(app.status.contains("not sandboxed"));
        assert!(app.status.contains("shared GPU must be idle"));
        assert!(app.status.contains("GUI/MCP remains separate"));
        assert!(!app.status.contains("CUDA"));
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.direct_blender_headless);
        assert!(editor.is_direct_native());
        assert_eq!(editor.declared_tasks, ["3d", "render", "image"]);
        assert_eq!(editor.selected_task, Some(0));
        assert_eq!(editor.media_operation, Some(TaskFormMediaOperation::Script));
        assert!(editor.fields.contains(&TaskFormField::DeclaredTask));
        assert!(editor.fields.contains(&TaskFormField::MediaScriptArgv));
        assert!(!editor.fields.contains(&TaskFormField::MediaWorkflowGraph));
        assert!(!editor.fields.contains(&TaskFormField::SpeechText));

        for (task, args) in [
            (
                "3d",
                r#"["--code","import bpy; bpy.ops.mesh.primitive_cube_add()"]"#,
            ),
            ("render", r#"["--render","/tmp/private-scene.blend"]"#),
            ("image", r#"["--render","/tmp/private-scene.blend"]"#),
        ] {
            select_task_form_declared_task(&mut app, task);
            let editor = app.task_form_editor.as_mut().unwrap();
            editor.values.script_argv = args.into();
            let draft = editor.to_draft().unwrap();
            assert!(!format!("{draft:?}").contains("private-scene"));
            assert!(!format!("{draft:?}").contains("primitive_cube"));
            let (request, _) = build_direct_native(&catalog, draft).unwrap();
            assert!(is_direct_native_script_request(&catalog, &request));
            let expected_args: Vec<String> = serde_json::from_str(args).unwrap();
            let crate::control::requests::InferenceRequest::Media {
                profile,
                input:
                    crate::control::requests::MediaInput::Script {
                        task: built_task,
                        args: built_args,
                    },
            } = request
            else {
                panic!("Blender form builds the typed native script request")
            };
            assert_eq!(profile.profile_id.as_str(), "blender-3d");
            assert!(catalog.inspect(&profile).unwrap().device.is_none());
            assert_eq!(built_task.as_str(), task);
            assert_eq!(built_args, expected_args);
        }

        assert!(!app.status.contains("primitive_cube"));
        assert!(app.log.iter().all(|line| !line.contains("primitive_cube")));
        assert!(app.job_history.records().is_empty());
        assert!(app.direct_native_job.is_none());
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap_or_default();
        assert!(!persisted.contains("primitive_cube"));
        assert!(!persisted.contains("private-scene"));
    }

    #[cfg(target_os = "linux")]
    fn select_task_form_declared_task(app: &mut App, task: &str) {
        focus_task_form_field(app, TaskFormField::DeclaredTask);
        let task_count = app.task_form_editor.as_ref().unwrap().declared_tasks.len();
        for _ in 0..task_count {
            let editor = app.task_form_editor.as_ref().unwrap();
            if editor
                .selected_task
                .and_then(|index| editor.declared_tasks.get(index))
                .is_some_and(|selected| selected == task)
            {
                return;
            }
            app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
                .unwrap();
        }
        panic!("task form does not declare task: {task}");
    }

    #[cfg(target_os = "linux")]
    fn focus_task_form_field(app: &mut App, field: TaskFormField) {
        let max_steps = app.task_form_editor.as_ref().unwrap().fields.len();
        for _ in 0..max_steps {
            if app.task_form_editor.as_ref().unwrap().active_field() == field {
                return;
            }
            app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
                .unwrap();
        }
        panic!("task form field is not focusable: {field:?}");
    }

    #[cfg(target_os = "linux")]
    fn type_task_form_text(app: &mut App, text: &str) {
        for character in text.chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE))
                .unwrap();
        }
    }

    #[cfg(target_os = "linux")]
    fn open_direct_decision_form(app: &mut App) {
        open_direct_decision_form_for_profile(app, "nagi");
    }

    #[cfg(target_os = "linux")]
    fn open_direct_decision_form_for_profile(app: &mut App, profile_id: &str) {
        app.handle_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE))
            .unwrap();
        let index = app
            .stack_catalog
            .as_ref()
            .unwrap()
            .list()
            .iter()
            .position(|profile| {
                profile.key.lane == Lane::Decisions && profile.key.profile_id.as_str() == profile_id
            })
            .unwrap();
        app.stack_catalog_state.select(Some(index));
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.direct_native_decision);
        assert!(!editor.fields.contains(&TaskFormField::DeclaredTask));
        assert_eq!(editor.values.timeout, "30000");
        assert_eq!(editor.declared_tasks, ["decision"]);
        assert_eq!(editor.selected_task, Some(0));
        editor.values.decision_state = r#"{"private_prompt":"decision-private-secret"}"#.into();
        editor.values.decision_questions =
            r#"{"route":{"type":"choice","criteria":{"yes":"Proceed","no":"Stop"}}}"#.into();
    }

    #[cfg(target_os = "linux")]
    fn open_direct_embedding_form(app: &mut App, input: &str) {
        focus_stack_profile_by_id(app, "local-embeddings");
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.direct_language_embeddings);
        assert!(editor.is_direct_native());
        assert!(!editor.fields.contains(&TaskFormField::DeclaredTask));
        assert!(editor.fields.contains(&TaskFormField::EmbeddingInputs));
        assert_eq!(editor.selected_task, Some(0));
        assert_eq!(editor.declared_tasks, ["embeddings"]);
        assert_eq!(editor.values.timeout, "30000");
        editor.values.embedding_inputs = input.into();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ltx_and_yue2_use_the_shared_comfy_form_not_gpu_scripts() {
        for (profile_id, task) in [("ltx-2.5", "video"), ("yue2-music", "music")] {
            let fixture = AppFixture::new();
            write_comfy_stack_fixture(&fixture);
            let manifest = fixture.root.join("media-runtimes.json");
            let text = fs::read_to_string(&manifest)
                .unwrap()
                .replace("comfy-workflow", profile_id)
                .replace("\"tasks\":[\"image\"]", &format!("\"tasks\":[\"{task}\"]"));
            fs::write(manifest, text).unwrap();
            let mut app = fixture.app();
            focus_stack_profile_by_id(&mut app, profile_id);
            let profile = app.selected_stack_profile().unwrap();
            assert!(!direct_run::requires_operator_gpu_window(profile));
            app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
                .unwrap();
            let editor = app.task_form_editor.as_mut().unwrap();
            assert!(editor.direct_comfy_workflow);
            assert!(!editor.is_direct_script_media());
            editor.selected_task = Some(0);
            editor.values.workflow_graph = r#"{"1":{"class_type":"SaveAudio","inputs":{}}}"#.into();
            let draft = editor.to_draft().unwrap();
            let catalog = app.stack_catalog.as_ref().unwrap();
            let (request, _) = build_direct_native(catalog, draft).unwrap();
            assert!(is_direct_comfy_workflow_request(catalog, &request));
            assert!(!is_direct_native_script_request(catalog, &request));
            assert!(app.direct_native_job.is_none());
            assert!(app.job_history.records().is_empty());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn comfy_workflow_form_is_direct_and_keeps_raw_graph_ephemeral() {
        let fixture = AppFixture::new();
        write_comfy_stack_fixture(&fixture);
        let mut app = fixture.app();
        focus_stack_profile_by_id(&mut app, "comfy-workflow");

        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.input_mode, InputMode::TaskForm);
        assert!(app.status.contains("llama-swap shared engine"));
        let draft = {
            let editor = app.task_form_editor.as_mut().unwrap();
            assert!(editor.direct_comfy_workflow);
            assert!(editor.is_direct_native());
            assert!(!editor.fields.contains(&TaskFormField::MediaOperation));
            assert!(editor.fields.contains(&TaskFormField::DeclaredTask));
            assert!(editor.fields.contains(&TaskFormField::MediaWorkflowGraph));
            editor.selected_task = Some(0);
            editor.values.workflow_graph = r#"{"1":{"class_type":"SaveImage"}}"#.into();
            editor.to_draft().unwrap()
        };
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let (request, _) = build_direct_native(&catalog, draft).unwrap();
        assert!(is_direct_comfy_workflow_request(&catalog, &request));
        assert!(matches!(
            request,
            crate::control::requests::InferenceRequest::Media {
                input: crate::control::requests::MediaInput::Workflow { task, .. },
                ..
            } if task.as_str() == "image"
        ));
        assert!(app.job_history.records().is_empty());
        assert!(app.direct_native_job.is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn comfy_history_artifact_refs_are_private_relative_paths_only() {
        let data = TempDir::new().unwrap();
        let output_root = data.path().join("native-media/comfy-job-1-123-0");
        fs::create_dir_all(output_root.join("node-9")).unwrap();
        fs::write(output_root.join("node-9/image.png"), b"fixture").unwrap();
        let result = DirectRunResult::Comfy(crate::control::comfy::NativeComfyResult {
            native_id: "12345678-1234-4234-9234-123456789abc".into(),
            output_root,
            artifacts: vec![crate::control::comfy::ComfyLocalArtifact {
                relative_path: PathBuf::from("node-9/image.png"),
                node_id: "9".into(),
                media_type: "image".into(),
            }],
        });
        assert_eq!(
            direct_native_artifact_refs(data.path(), &result),
            Some(vec![
                "native-media/comfy-job-1-123-0/node-9/image.png".into()
            ])
        );
        let preview = direct_native_result_preview(&result);
        assert!(preview.contains("image · node 9"));
        assert!(preview.contains("image.png"));
        assert!(!preview.contains("12345678"));
    }

    #[cfg(target_os = "linux")]
    fn start_direct_embedding_job(fixture: &AppFixture, endpoint: &str, input: &str) -> (App, u64) {
        let mut app = fixture.app();
        open_direct_embedding_form(&mut app, input);
        connect_test_chat_at(
            &mut app,
            endpoint,
            Some("fixture-key"),
            vec![test_chat_model("local-embeddings", "unloaded", false)],
        );
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL))
            .unwrap();
        let job_id = app.direct_native_job.as_ref().unwrap().job_id;
        (app, job_id)
    }

    #[cfg(target_os = "linux")]
    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        use std::io::Read;

        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0, "HTTP client closed before completing its request");
            bytes.extend_from_slice(&chunk[..read]);
            let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if bytes.len() >= header_end + 4 + content_length {
                return String::from_utf8(bytes).unwrap();
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn serve_direct_embeddings(
        listener: std::net::TcpListener,
        expected_input: Vec<String>,
        expected_authorization: Option<&'static str>,
        release: Option<mpsc::Receiver<()>>,
        request_received: Option<mpsc::Sender<()>>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            use std::io::Write;

            listener.set_nonblocking(true).unwrap();
            let accept_deadline = Instant::now() + Duration::from_secs(3);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < accept_deadline =>
                    {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("embedding endpoint did not receive request: {error}"),
                }
            };
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let request = read_http_request(&mut stream);
            assert!(request.starts_with("POST /v1/embeddings "));
            let (headers, body) = request.split_once("\r\n\r\n").unwrap();
            if let Some(expected) = expected_authorization {
                assert!(headers.lines().any(|line| {
                    line.to_ascii_lowercase()
                        .starts_with(&format!("authorization: bearer {expected}"))
                }));
            }
            let request: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(request["model"], "local-embeddings");
            assert_eq!(request["input"], serde_json::json!(expected_input.clone()));
            assert_eq!(request["encoding_format"], "float");
            if let Some(request_received) = request_received {
                request_received.send(()).unwrap();
            }
            if let Some(release) = release {
                release.recv_timeout(Duration::from_secs(3)).unwrap();
            }
            let data = expected_input
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    serde_json::json!({
                        "object": "embedding",
                        "index": index,
                        "embedding": [0.25 + index as f32, -0.5]
                    })
                })
                .collect::<Vec<_>>();
            let body = serde_json::json!({
                "object": "list",
                "model": "server-model-name",
                "data": data,
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            })
            .to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
        })
    }

    #[cfg(target_os = "linux")]
    fn serve_llamacpp_decision(
        listener: std::net::TcpListener,
        expected_request: serde_json::Value,
        expected_authorization: Option<&'static str>,
        release: Option<mpsc::Receiver<()>>,
        request_received: Option<mpsc::Sender<()>>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            use std::io::Write;

            listener.set_nonblocking(true).unwrap();
            let accept_deadline = Instant::now() + Duration::from_secs(3);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < accept_deadline =>
                    {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("decision endpoint did not receive request: {error}"),
                }
            };
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let request = read_http_request(&mut stream);
            assert!(request.starts_with("POST /gateway/v1/systemone "));
            let (headers, body) = request.split_once("\r\n\r\n").unwrap();
            if let Some(expected) = expected_authorization {
                assert!(headers.lines().any(|line| {
                    line.to_ascii_lowercase()
                        .starts_with(&format!("authorization: bearer {expected}"))
                }));
            }
            let request: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(request, expected_request);
            if let Some(request_received) = request_received {
                request_received.send(()).unwrap();
            }
            if let Some(release) = release {
                release.recv_timeout(Duration::from_secs(3)).unwrap();
            }
            let body = serde_json::json!({
                "answers": {
                    "route": {"choice": "yes", "confidence": 0.9}
                },
                "usage": {"fixture_units": 1}
            })
            .to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
        })
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn connected_embedding_batch_reuses_committed_auth_and_keeps_results_ephemeral() {
        let fixture = AppFixture::new();
        write_embedding_stack_fixture(&fixture);
        let input = vec![
            "embedding-private-secret".to_owned(),
            "unicode π".to_owned(),
        ];
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server =
            serve_direct_embeddings(listener, input.clone(), Some("fixture-key"), None, None);
        let (mut app, _) = start_direct_embedding_job(
            &fixture,
            &endpoint,
            r#"["embedding-private-secret","unicode π"]"#,
        );
        assert!(app.direct_native_job.is_some());
        assert!(app.task_form_editor.is_none());
        wait_for_direct_native(&mut app);
        server.join().unwrap();

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Succeeded));
        let Some(DirectRunResult::Embeddings(payload)) = summary.result.as_ref() else {
            panic!("direct language embeddings were not retained in memory")
        };
        assert_eq!(payload.vectors, vec![vec![0.25, -0.5], vec![1.25, -0.5]]);
        assert_eq!(payload.usage.prompt_tokens, 5);
        assert_eq!(payload.usage.total_tokens, 5);
        assert!(app.status.contains("Language embeddings"));
        assert!(!app.status.contains("embedding-private-secret"));
        assert!(app.log.iter().all(|line| {
            !line.contains("embedding-private-secret") && !line.contains("fixture-key")
        }));

        let job = &app.job_history.records()[0];
        assert!(job.command.is_empty());
        assert_eq!(job.metadata.toolkit.as_deref(), Some("l3ms"));
        assert_eq!(job.metadata.task.as_deref(), Some("embeddings"));
        assert!(job.metadata.request_id.is_none());
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("embedding-private-secret"));
        assert!(!persisted.contains("fixture-key"));
        assert!(!persisted.contains("vectors"));

        let mut terminal = Terminal::new(TestBackend::new(140, 45)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let rendered = terminal_text(&terminal);
        assert!(rendered.contains("Embedding batch"));
        assert!(rendered.contains("vector[0]"));
        assert!(rendered.contains("prompt_tokens=5"));
        assert!(rendered.contains("Foreground native result"));
        assert!(!rendered.contains("results are shown here only after pressing o"));
        assert!(!rendered.contains("embedding-private-secret"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn llamacpp_decision_form_posts_once_through_committed_chat_and_keeps_results_ephemeral() {
        let fixture = AppFixture::new();
        let router_model = "vllm-d3-flash";
        write_llamacpp_decision_stack_fixture(&fixture, router_model);
        let state = serde_json::json!({"private_prompt": "decision-private-secret"});
        let questions = serde_json::json!({
            "route": {
                "type": "choice",
                "criteria": {"yes": "Proceed", "no": "Stop"}
            }
        });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/gateway", listener.local_addr().unwrap());
        let (request_received_tx, request_received_rx) = mpsc::channel();
        let server = serve_llamacpp_decision(
            listener,
            serde_json::json!({
                "model": router_model, "state": state, "questions": questions,
                "images": ["data:image/png;base64,iVBORw0KGgo="]
            }),
            Some("fixture-key"),
            None,
            Some(request_received_tx),
        );
        let mut app = fixture.app_without_model_refresh();
        connect_test_chat_at(
            &mut app,
            &endpoint,
            Some("fixture-key"),
            vec![
                test_chat_model("another-model", "loaded", false),
                test_chat_model(router_model, "unloaded", false),
            ],
        );
        app.chat_non_chat_model_ids = vec![router_model.into()];
        assert_eq!(app.selected_chat_model().unwrap().id, "another-model");
        assert!(app.chat_models.iter().any(|model| model.id == router_model));
        open_direct_decision_form_for_profile(&mut app, router_model);
        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.fields.contains(&TaskFormField::DecisionImages));
        assert_eq!(editor.values.decision_images, "[]");
        editor.values.decision_images = r#"["data:image/png;base64,iVBORw0KGgo="]"#.into();
        assert!(app.status.contains("may evict the loaded language model"));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        assert!(terminal_text(&terminal).contains("quiet window"));
        app.run_task_form();
        request_received_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("decision request did not reach the local fixture endpoint");
        wait_for_direct_native(&mut app);
        server.join().unwrap();

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Succeeded));
        assert!(!summary.requires_idle_gpu_window);
        assert!(!app.direct_native_admission_closed);
        let Some(DirectRunResult::Decision(native)) = summary.result.as_ref() else {
            panic!("llama.cpp decision answer was not retained in memory");
        };
        let crate::control::requests::InferenceResult::Decision { answers } = &native.result else {
            panic!("llama.cpp decision returned a non-decision result");
        };
        assert_eq!(answers["route"]["choice"], "yes");
        assert_eq!(native.native_usage.as_ref().unwrap()["fixture_units"], 1);
        assert!(app.status.contains("succeeded"));
        assert!(!app.status.contains("decision-private-secret"));
        assert!(app.log.iter().all(|line| {
            !line.contains("decision-private-secret") && !line.contains("fixture-key")
        }));

        let job = &app.job_history.records()[0];
        assert!(job.command.is_empty());
        assert_eq!(job.metadata.toolkit.as_deref(), Some("l3ds"));
        assert_eq!(job.metadata.profile.as_deref(), Some(router_model));
        assert_eq!(job.metadata.task.as_deref(), Some("decision"));
        assert!(job.metadata.request_id.is_none());
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("decision-private-secret"));
        assert!(!persisted.contains("fixture-key"));
        assert!(!persisted.contains("fixture_units"));
        assert!(!persisted.contains("data:image"));
        assert!(!persisted.contains("\"answers\""));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cancelling_dispatched_llamacpp_decision_closes_chat_and_foreground_admission() {
        let fixture = AppFixture::new();
        let router_model = "cloudflare-clef-flash";
        write_llamacpp_decision_stack_fixture(&fixture, router_model);
        let state = serde_json::json!({"private_prompt": "decision-cancel-secret"});
        let questions = serde_json::json!({
            "route": {
                "type": "choice",
                "instructions": "Choose whether to proceed.",
                "criteria": {"yes": "Proceed", "no": "Stop"}
            }
        });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/gateway", listener.local_addr().unwrap());
        let (request_received_tx, request_received_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let server = serve_llamacpp_decision(
            listener,
            serde_json::json!({"model": router_model, "state": state, "questions": questions}),
            Some("fixture-key"),
            Some(release_rx),
            Some(request_received_tx),
        );
        let mut app = fixture.app_without_model_refresh();
        connect_test_chat_at(
            &mut app,
            &endpoint,
            Some("fixture-key"),
            vec![test_chat_model(router_model, "unloaded", false)],
        );
        open_direct_decision_form_for_profile(&mut app, router_model);
        app.task_form_editor.as_mut().unwrap().values.decision_state =
            r#"{"private_prompt":"decision-cancel-secret"}"#.into();
        app.task_form_editor
            .as_mut()
            .unwrap()
            .values
            .decision_questions = questions.to_string();
        app.run_task_form();
        request_received_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("decision request did not reach the local fixture endpoint");

        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.status.contains("cancellation requested"));
        wait_for_direct_native(&mut app);
        release_tx.send(()).unwrap();
        server.join().unwrap();

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Interrupted));
        assert!(summary.result.is_none());
        assert!(app.direct_native_admission_closed);
        assert!(app.status.contains("remote completion is unconfirmed"));
        assert!(app.status.contains("verify the endpoint before retrying"));
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("interrupted")
        );

        app.chat_input = "unsent chat draft".into();
        app.send_chat_message();
        assert!(app
            .status
            .contains("resolve its outcome before sending Chat"));
        assert_eq!(app.chat_input, "unsent chat draft");
        assert!(!app.chat_pending);
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("decision-cancel-secret"));
        assert!(!persisted.contains("fixture-key"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cancelling_dispatched_embedding_keeps_admission_closed_on_uncertain_completion() {
        let fixture = AppFixture::new();
        write_embedding_stack_fixture(&fixture);
        let input = vec!["cancel-private-secret".to_owned()];
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (request_received_tx, request_received_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let server = serve_direct_embeddings(
            listener,
            input,
            Some("fixture-key"),
            Some(release_rx),
            Some(request_received_tx),
        );
        let (mut app, job_id) =
            start_direct_embedding_job(&fixture, &endpoint, r#"["cancel-private-secret"]"#);
        request_received_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("embedding request did not reach the connected endpoint");
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("running")
        );

        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.direct_native_job.as_ref().unwrap().job_id, job_id);
        assert!(app.status.contains("cancellation requested"));
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("running")
        );
        wait_for_direct_native(&mut app);
        release_tx.send(()).unwrap();
        server.join().unwrap();

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Interrupted));
        assert!(summary.result.is_none());
        assert!(app.direct_native_admission_closed);
        assert!(app.status.contains("remote completion is unconfirmed"));
        assert!(app.status.contains("verify the endpoint before retrying"));
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("interrupted")
        );
        assert!(app.log.iter().all(|line| {
            !line.contains("cancel-private-secret") && !line.contains("fixture-key")
        }));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("cancel-private-secret"));
        assert!(!persisted.contains("fixture-key"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_embedding_requires_the_exact_enabled_model_and_idle_confirmed_chat() {
        let fixture = AppFixture::new();
        write_embedding_stack_fixture(&fixture);
        let mut app = fixture.app();
        focus_stack_profile_by_id(&mut app, "local-embeddings");
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let request = crate::control::requests::InferenceRequest::Embeddings {
            profile: crate::control::catalog::ProfileKey::new(Lane::Language, "local-embeddings")
                .unwrap(),
            input: vec!["private batch input".into()],
        };
        connect_test_chat(
            &mut app,
            vec![
                test_chat_model("another-model", "loaded", false),
                test_chat_model("local-embeddings", "unloaded", false),
            ],
        );
        assert!(app
            .prepare_direct_embedding_adapter(&catalog, &request)
            .is_ok());

        for (case, expected) in [
            ("missing-client", "Connect Chat"),
            ("uncommitted-endpoint", "not a confirmed connection"),
            ("connection-pending", "Wait for the current Chat connection"),
            ("chat-pending", "Wait for the current Chat connection"),
            ("session-pending", "Wait for the current Chat connection"),
            (
                "missing-route-model",
                "exact embeddings route model is not advertised",
            ),
            (
                "disabled-route-model",
                "exact embeddings route model is disabled",
            ),
        ] {
            let mut app = fixture.app();
            connect_test_chat(
                &mut app,
                vec![
                    test_chat_model("another-model", "loaded", false),
                    test_chat_model("local-embeddings", "unloaded", false),
                ],
            );
            match case {
                "missing-client" => {
                    app.chat_client = None;
                    app.chat_endpoint_committed = None;
                }
                "uncommitted-endpoint" => {
                    app.chat_endpoint_committed = Some("http://different.example".into());
                }
                "connection-pending" => app.chat_connection_pending = true,
                "chat-pending" => app.chat_pending = true,
                "session-pending" => app.chat_session_pending = true,
                "missing-route-model" => {
                    app.chat_models = vec![test_chat_model("another-model", "loaded", false)];
                }
                "disabled-route-model" => {
                    app.chat_models = vec![test_chat_model("local-embeddings", "unloaded", true)];
                }
                _ => unreachable!(),
            }
            let error = match app.prepare_direct_embedding_adapter(&catalog, &request) {
                Ok(_) => panic!("{case} must refuse a direct embedding run"),
                Err(error) => error,
            };
            assert!(error.contains(expected), "{case}: {error}");
            assert!(app.direct_native_job.is_none());
            assert!(app.job_history.records().is_empty());
        }
    }

    #[cfg(target_os = "linux")]
    fn wait_for_direct_native(app: &mut App) {
        let deadline = Instant::now() + Duration::from_secs(8);
        while app.direct_native_job.is_some() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        app.drain_background_events();
        assert!(
            app.direct_native_job.is_none(),
            "direct native task did not finish"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_decision_form_runs_without_socket_and_persists_only_redacted_metadata() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("runner-started");
        write_direct_decision_runner(&fixture, &marker, false);
        let mut app = fixture.app();
        open_direct_decision_form(&mut app);
        app.run_task_form();
        assert!(app.direct_native_job.is_some());
        assert!(app.task_form_editor.is_none());
        wait_for_direct_native(&mut app);

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Succeeded));
        let Some(DirectRunResult::Decision(native)) = summary.result.as_ref() else {
            panic!("direct CPU decision result was not retained");
        };
        assert_eq!(native.native_latency, Some(Duration::from_nanos(1_250_000)));
        assert!(app.status.contains("succeeded"));
        assert!(!app.status.contains("decision-private-secret"));
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("decision-private-secret")));
        let job = &app.job_history.records()[0];
        assert!(job.command.is_empty());
        assert_eq!(job.metadata.toolkit.as_deref(), Some("l3ds"));
        assert_eq!(job.metadata.task.as_deref(), Some("decision"));
        assert!(job.metadata.request_id.is_none());
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("decision-private-secret"));
        assert!(!persisted.contains("argv"));
        assert!(marker.exists());

        // A short terminal must show the answer, not only JSON opening brackets.
        let mut narrow = Terminal::new(TestBackend::new(80, 24)).unwrap();
        narrow.draw(|frame| app.draw(frame)).unwrap();
        let visible = terminal_text(&narrow);
        assert!(visible.contains("\"choice\":\"yes\""));
        assert!(visible.contains("\"route\""));

        let language_index = app
            .stack_catalog
            .as_ref()
            .unwrap()
            .list()
            .iter()
            .position(|profile| profile.key.lane == Lane::Language)
            .unwrap();
        app.stack_catalog_state.select(Some(language_index));
        let mut terminal = Terminal::new(TestBackend::new(140, 45)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        assert!(terminal_text(&terminal).contains("result is hidden"));
        app.direct_native_summary = None;
        assert!(app.direct_native_summary.is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hosted_jev_decision_reuses_native_form_without_dispatching() {
        let fixture = AppFixture::new();
        write_hosted_jev_decision_stack_fixture(&fixture);
        let mut app = fixture.app();
        focus_stack_profile_by_id(&mut app, "jev-api");
        let profile = app
            .stack_catalog
            .as_ref()
            .unwrap()
            .list()
            .iter()
            .find(|profile| profile.key.profile_id.as_str() == "jev-api")
            .unwrap();
        assert!(is_direct_native_decision_profile(profile));
        assert_eq!(profile.runtime.as_str(), "jev");
        assert_eq!(profile.device, Some(Device::Remote));

        let mut terminal = Terminal::new(TestBackend::new(140, 45)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        assert!(terminal_text(&terminal).contains("device    remote"));

        open_direct_decision_form_for_profile(&mut app, "jev-api");
        assert!(app.status.contains("Foreground native decision form open"));
        assert!(app.direct_native_job.is_none());
        assert!(app.job_history.records().is_empty());
        let editor = app.task_form_editor.as_mut().unwrap();
        assert!(editor.direct_native_decision);
        assert!(editor.is_direct_native());
        assert!(!editor.fields.contains(&TaskFormField::DeclaredTask));
        assert!(!editor.fields.contains(&TaskFormField::DecisionImages));
        assert_eq!(editor.profile.profile_id.as_str(), "jev-api");

        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let (request, timeout) = build_direct_native(&catalog, editor.to_draft().unwrap()).unwrap();
        assert_eq!(timeout, Duration::from_secs(30));
        let crate::control::requests::InferenceRequest::Decision {
            profile,
            state,
            questions,
            images,
        } = request
        else {
            panic!("hosted Jev task form must build the standard decision request");
        };
        assert_eq!(profile.profile_id.as_str(), "jev-api");
        assert_eq!(state["private_prompt"], "decision-private-secret");
        assert!(questions.get("route").is_some());
        assert!(images.is_empty());
        assert_eq!(app.job_history.records().len(), 0);
        assert!(!app.status.contains("decision-private-secret"));
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("decision-private-secret")));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap_or_default();
        assert!(!persisted.contains("decision-private-secret"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_standard_tts_runs_without_socket_and_keeps_only_path_metadata() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("tts-runner-started");
        write_direct_cpu_tts_runner(&fixture, &marker, false);
        let mut app = fixture.app();
        open_direct_cpu_speech_form(&mut app, "tts-private-secret");
        app.run_task_form();
        assert!(app.direct_native_job.is_some());
        assert!(app.task_form_editor.is_none());
        wait_for_direct_native(&mut app);

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Succeeded));
        assert_eq!(summary.task, "tts");
        let Some(DirectRunResult::Speech(result)) = summary.result.as_ref() else {
            panic!("direct CPU speech artifact was not retained");
        };
        assert_eq!(result.artifacts().len(), 1);
        assert_eq!(
            result.artifacts()[0].relative_path(),
            Path::new("result.wav")
        );
        assert!(result.output_root().join("result.wav").is_file());
        let retry_directory =
            create_private_direct_output_directory(fixture.data.path(), 1, "cpu-job").unwrap();
        assert_ne!(retry_directory, result.output_root());
        assert!(result.output_root().join("result.wav").is_file());
        assert!(app.status.contains("CPU speech"));
        assert!(!app.status.contains("tts-private-secret"));
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("tts-private-secret")));

        let job = &app.job_history.records()[0];
        assert!(job.command.is_empty());
        assert_eq!(job.metadata.toolkit.as_deref(), Some("l2m2"));
        assert_eq!(job.metadata.task.as_deref(), Some("tts"));
        assert!(job.metadata.request_id.is_none());
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        let artifact_ref = job.metadata.artifact_refs.first().unwrap();
        assert!(artifact_ref.starts_with("native-media/cpu-job-1-"));
        assert!(artifact_ref.ends_with("/result.wav"));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("tts-private-secret"));
        assert!(!persisted.contains(result.output_root().to_str().unwrap()));
        assert!(persisted.contains(artifact_ref));
        assert!(marker.exists());

        let displayed_path = result
            .output_root()
            .join("result.wav")
            .to_string_lossy()
            .into_owned();
        let preview = direct_native_result_preview(summary.result.as_ref().unwrap());
        assert!(preview.find("result.wav").unwrap() < preview.find(&displayed_path).unwrap());

        let mut compact_terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        compact_terminal.draw(|frame| app.draw(frame)).unwrap();
        assert!(terminal_text(&compact_terminal).contains("result.wav"));

        let mut terminal = Terminal::new(TestBackend::new(140, 45)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        assert!(terminal_text(&terminal).contains(&displayed_path));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_voice_design_uses_declared_task_and_keeps_instruction_private() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("voice-design-runner-started");
        write_direct_cpu_tts_runner(&fixture, &marker, false);
        let mut app = fixture.app();
        open_direct_cpu_speech_form(&mut app, "voice-design-private-text");

        select_task_form_declared_task(&mut app, "design");
        let editor = app.task_form_editor.as_ref().unwrap();
        assert_eq!(editor.active_field(), TaskFormField::DeclaredTask);
        assert_eq!(
            editor.declared_tasks[editor.selected_task.unwrap()],
            "design"
        );
        assert!(editor.fields.contains(&TaskFormField::SpeechInstruction));
        assert!(!editor.fields.contains(&TaskFormField::SpeechMode));
        assert!(!editor.fields.contains(&TaskFormField::SpeechVoiceReference));
        assert!(!editor.fields.contains(&TaskFormField::SpeechReferenceText));

        focus_task_form_field(&mut app, TaskFormField::SpeechInstruction);
        type_task_form_text(&mut app, "voice-design-private-instruction");
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let draft = app.task_form_editor.as_ref().unwrap().to_draft().unwrap();
        let (design_request, _) = build_direct_native(&catalog, draft).unwrap();
        let crate::control::requests::InferenceRequest::Media {
            input: crate::control::requests::MediaInput::Speech(input),
            ..
        } = design_request
        else {
            panic!("design task form must build a speech request");
        };
        assert_eq!(input.mode, SpeechMode::VoiceDesign);
        assert_eq!(
            input.instruction.as_deref(),
            Some("voice-design-private-instruction")
        );

        select_task_form_declared_task(&mut app, "tts");
        let draft = app.task_form_editor.as_ref().unwrap().to_draft().unwrap();
        let (standard_request, _) = build_direct_native(&catalog, draft).unwrap();
        let crate::control::requests::InferenceRequest::Media {
            input: crate::control::requests::MediaInput::Speech(input),
            ..
        } = standard_request
        else {
            panic!("standard task form must build a speech request");
        };
        assert_eq!(input.mode, SpeechMode::Standard);
        assert!(input.instruction.is_none());
        assert_eq!(
            app.task_form_editor
                .as_ref()
                .unwrap()
                .values
                .speech_instruction,
            "voice-design-private-instruction"
        );
        select_task_form_declared_task(&mut app, "design");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL))
            .unwrap();
        assert!(app.direct_native_job.is_some());
        wait_for_direct_native(&mut app);

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Succeeded));
        assert_eq!(summary.task, "design");
        let Some(DirectRunResult::Speech(result)) = summary.result.as_ref() else {
            panic!("direct CPU voice-design artifact was not retained");
        };
        assert_eq!(result.mode(), SpeechMode::VoiceDesign);
        assert_eq!(result.artifacts().len(), 1);
        assert!(result.output_root().join("result.wav").is_file());
        assert!(app.status.contains("CPU voice design"));
        assert!(!app.status.contains("voice-design-private"));
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("voice-design-private")));

        let job = &app.job_history.records()[0];
        assert!(job.command.is_empty());
        assert_eq!(job.metadata.task.as_deref(), Some("design"));
        assert!(job.metadata.request_id.is_none());
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("voice-design-private"));
        assert!(marker.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_voice_clone_uses_reference_fields_without_persisting_payloads() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("voice-clone-runner-started");
        write_direct_cpu_tts_runner(&fixture, &marker, false);
        let reference = fixture.root.join("private-reference.wav");
        fs::write(&reference, b"fixture voice sample").unwrap();
        let mut app = fixture.app();
        open_direct_cpu_speech_form(&mut app, "clone-private-text");
        {
            let editor = app.task_form_editor.as_mut().unwrap();
            editor.values.speech_instruction = "hidden-design-instruction".into();
            editor.values.speech_voice_reference = reference.to_string_lossy().into_owned();
            editor.values.speech_reference_text = "clone-private-transcript".into();
        }
        select_task_form_declared_task(&mut app, "clon");
        let editor = app.task_form_editor.as_ref().unwrap();
        assert!(editor.fields.contains(&TaskFormField::SpeechVoiceReference));
        assert!(editor.fields.contains(&TaskFormField::SpeechReferenceText));
        assert!(!editor.fields.contains(&TaskFormField::SpeechMode));
        assert!(!editor.fields.contains(&TaskFormField::SpeechInstruction));
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let (request, _) = build_direct_native(&catalog, editor.to_draft().unwrap()).unwrap();
        let crate::control::requests::InferenceRequest::Media {
            input: crate::control::requests::MediaInput::Speech(input),
            ..
        } = request
        else {
            panic!("clone form must build a speech request");
        };
        assert_eq!(input.mode, SpeechMode::VoiceClone);
        assert!(input.instruction.is_none());
        assert_eq!(input.voice_reference.as_deref(), Some(reference.as_path()));
        assert_eq!(
            input.reference_text.as_deref(),
            Some("clone-private-transcript")
        );

        app.run_task_form();
        assert!(app.direct_native_job.is_some());
        wait_for_direct_native(&mut app);

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Succeeded));
        assert_eq!(summary.task, "clon");
        let Some(DirectRunResult::Speech(result)) = summary.result.as_ref() else {
            panic!("direct CPU clone artifact was not retained");
        };
        assert_eq!(result.mode(), SpeechMode::VoiceClone);
        assert!(result.output_root().join("result.wav").is_file());
        assert!(app.status.contains("CPU voice clone"));

        let args = fs::read_to_string(marker.with_extension("args")).unwrap();
        assert!(args.contains("--task\nclon"));
        assert!(args.contains("--voice-ref"));
        assert!(args.contains("--reference-text\nclone-private-transcript"));
        assert!(args.contains(reference.to_str().unwrap()));
        assert!(!args.contains("hidden-design-instruction"));
        assert!(!app.status.contains("clone-private"));
        assert!(app.log.iter().all(|line| {
            !line.contains("clone-private")
                && !line.contains("private-reference.wav")
                && !line.contains("hidden-design-instruction")
        }));
        let job = &app.job_history.records()[0];
        assert!(job.command.is_empty());
        assert_eq!(job.metadata.task.as_deref(), Some("clon"));
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        assert!(job.metadata.request_id.is_none());
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("clone-private"));
        assert!(!persisted.contains("private-reference.wav"));
        assert!(!persisted.contains(reference.to_str().unwrap()));
        assert!(marker.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_streaming_tts_retains_completed_chunks_and_omits_hidden_fields() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("streaming-tts-runner-started");
        write_direct_cpu_tts_runner(&fixture, &marker, false);
        let mut app = fixture.app();
        open_direct_cpu_speech_form(&mut app, "streaming-private-text");
        {
            let editor = app.task_form_editor.as_mut().unwrap();
            editor.values.speech_instruction = "hidden-instruction-private".into();
            editor.values.speech_voice_reference = "hidden-reference-private.wav".into();
            editor.values.speech_reference_text = "hidden-transcript-private".into();
        }
        focus_task_form_field(&mut app, TaskFormField::SpeechMode);
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            .unwrap();
        let editor = app.task_form_editor.as_ref().unwrap();
        assert_eq!(editor.speech_mode, Some(SpeechMode::Streaming));
        assert!(!editor.fields.contains(&TaskFormField::SpeechInstruction));
        assert!(!editor.fields.contains(&TaskFormField::SpeechVoiceReference));
        assert!(!editor.fields.contains(&TaskFormField::SpeechReferenceText));
        let catalog = app.stack_catalog.as_ref().unwrap().clone();
        let (request, _) = build_direct_native(&catalog, editor.to_draft().unwrap()).unwrap();
        let crate::control::requests::InferenceRequest::Media {
            input: crate::control::requests::MediaInput::Speech(input),
            ..
        } = request
        else {
            panic!("streaming form must build a speech request");
        };
        assert_eq!(input.mode, SpeechMode::Streaming);
        assert!(input.instruction.is_none());
        assert!(input.voice_reference.is_none());
        assert!(input.reference_text.is_none());

        app.run_task_form();
        assert!(app.direct_native_job.is_some());
        wait_for_direct_native(&mut app);

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Succeeded));
        assert_eq!(summary.task, "tts");
        let Some(DirectRunResult::Speech(result)) = summary.result.as_ref() else {
            panic!("direct CPU streaming artifacts were not retained");
        };
        assert_eq!(result.mode(), SpeechMode::Streaming);
        assert_eq!(result.artifacts().len(), 2);
        assert!(result.artifacts().iter().any(|artifact| {
            artifact.kind() == crate::control::media_execution::MediaArtifactKind::StreamingChunk
                && result
                    .output_root()
                    .join(artifact.relative_path())
                    .is_file()
        }));
        let args = fs::read_to_string(marker.with_extension("args")).unwrap();
        assert!(args.contains("--task\ntts"));
        assert!(args.contains("--mode\nstreaming"));
        assert!(!args.contains("hidden-instruction-private"));
        assert!(!args.contains("hidden-reference-private.wav"));
        assert!(!args.contains("hidden-transcript-private"));
        assert!(!app.status.contains("streaming-private"));
        assert!(app.log.iter().all(|line| {
            !line.contains("streaming-private")
                && !line.contains("hidden-instruction-private")
                && !line.contains("hidden-reference-private.wav")
                && !line.contains("hidden-transcript-private")
        }));
        let job = &app.job_history.records()[0];
        assert!(job.command.is_empty());
        assert_eq!(job.metadata.task.as_deref(), Some("tts"));
        assert_eq!(job.metadata.result_state.as_deref(), Some("succeeded"));
        assert!(job.metadata.request_id.is_none());
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("streaming-private"));
        assert!(!persisted.contains("hidden-instruction-private"));
        assert!(!persisted.contains("hidden-reference-private.wav"));
        assert!(!persisted.contains("hidden-transcript-private"));
        assert!(marker.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_standard_tts_cancel_keeps_single_foreground_owner_until_cleanup() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("tts-runner-started");
        write_direct_cpu_tts_runner(&fixture, &marker, true);
        let mut app = fixture.app();
        open_direct_cpu_speech_form(&mut app, "tts-private-secret");
        app.run_task_form();
        let job_id = app.direct_native_job.as_ref().unwrap().job_id;
        let start_deadline = Instant::now() + Duration::from_secs(3);
        while !marker.exists() && Instant::now() < start_deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        assert!(marker.exists(), "fake CPU speech runner did not start");

        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.direct_native_job.as_ref().unwrap().job_id, job_id);
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("running")
        );
        wait_for_direct_native(&mut app);

        let summary = app.direct_native_summary.as_ref().unwrap();
        assert!(matches!(summary.state, DirectNativeState::Cancelled));
        assert!(summary.result.is_none());
        assert!(!app.direct_native_admission_closed);
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("cancelled")
        );
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("tts-private-secret")));
        let native_media = fixture.data.path().join("native-media");
        let output_directory = fs::read_dir(native_media)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("cpu-job-1-"))
            })
            .unwrap();
        assert!(output_directory.is_dir());
        assert!(!output_directory.join("result.wav").exists());
        assert!(!marker.with_extension("late").exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_decision_history_save_failure_prevents_launch() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("runner-started");
        write_direct_decision_runner(&fixture, &marker, false);
        let mut app = fixture.app();
        // A directory where the history file belongs makes persistence fail
        // deterministically, without relying on the test user's privileges.
        fs::create_dir(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        open_direct_decision_form(&mut app);
        app.run_task_form();
        assert!(app.direct_native_job.is_none());
        assert!(!marker.exists());
        assert!(app.status.contains("nothing was started"));
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("failed")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_decision_cancel_keeps_job_busy_until_cleanup_and_rejects_a_second_form() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("runner-started");
        write_direct_decision_runner(&fixture, &marker, true);
        let mut app = fixture.app();
        open_direct_decision_form(&mut app);
        app.run_task_form();
        let job_id = app.direct_native_job.as_ref().unwrap().job_id;
        let start_deadline = Instant::now() + Duration::from_secs(3);
        while !marker.exists() && Instant::now() < start_deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        assert!(marker.exists(), "fake native runner did not start");

        let language_index = app
            .stack_catalog
            .as_ref()
            .unwrap()
            .list()
            .iter()
            .position(|profile| profile.key.lane == Lane::Language)
            .unwrap();
        app.stack_catalog_state.select(Some(language_index));
        app.begin_task_form();
        assert!(app.task_form_editor.is_none());
        assert!(app
            .status
            .contains("foreground native task is active or interrupted"));
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.direct_native_job.as_ref().unwrap().job_id, job_id);
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("running")
        );
        assert!(app.status.contains("cancellation requested"));

        wait_for_direct_native(&mut app);
        assert!(!app.direct_native_admission_closed);
        assert!(matches!(
            app.direct_native_summary.as_ref().unwrap().state,
            DirectNativeState::Cancelled
        ));
        assert_eq!(
            app.job_history.records()[0]
                .metadata
                .result_state
                .as_deref(),
            Some("cancelled")
        );
        assert!(app
            .log
            .iter()
            .all(|line| !line.contains("decision-private-secret")));
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(!persisted.contains("decision-private-secret"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dropping_app_cancels_and_joins_the_direct_cpu_child() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let marker = fixture.root.join("runner-started");
        let late_marker = marker.with_extension("late");
        write_direct_decision_runner(&fixture, &marker, true);
        let mut app = fixture.app();
        open_direct_decision_form(&mut app);
        app.run_task_form();
        let start_deadline = Instant::now() + Duration::from_secs(3);
        while !marker.exists() && Instant::now() < start_deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        assert!(marker.exists(), "fake native runner did not start");

        drop(app);
        assert!(!late_marker.exists(), "native runner outlived App shutdown");
        let persisted =
            fs::read_to_string(crate::state_store::jobs_path_in(&fixture.data)).unwrap();
        assert!(persisted.contains("cancelled"));
        assert!(!persisted.contains("decision-private-secret"));
    }

    #[test]
    fn stack_catalog_keyboard_and_palette_navigation_are_read_only() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.handle_key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.view, View::Model);
        assert_eq!(app.stack_catalog.as_ref().unwrap().list().len(), 3);
        assert!(app
            .stack_catalog
            .as_ref()
            .unwrap()
            .list()
            .iter()
            .all(|p| p.readiness == Readiness::ManifestPresent));
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.stack_catalog_state.selected(), Some(1));
        for size in [(140, 45), (80, 24), (40, 12)] {
            let mut terminal = Terminal::new(TestBackend::new(size.0, size.1)).unwrap();
            terminal.draw(|frame| app.draw(frame)).unwrap();
            if size.0 == 140 {
                let text = terminal_text(&terminal);
                assert!(text.contains(" breeze "), "{text}");
                assert!(text.contains("runtime   audio.cpp"), "{text}");
                assert!(text.contains("tts, clon, design"), "{text}");
            }
        }
        app.execute_command(CommandId::OpenLog).unwrap();
        app.execute_command(CommandId::NextTab).unwrap();
        assert_eq!(app.view, View::Model);
        app.execute_command(CommandId::OpenStackCatalog).unwrap();
        app.execute_command(CommandId::OpenMaintenance).unwrap();
        app.focus_script_editor(ScriptEditorTarget::Maintenance);
        let before = app.maintenance_editor.content().to_owned();
        app.execute_command(CommandId::OpenStackCatalog).unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.maintenance_editor.content(), before);
        assert!(app.script_input_target.is_none());
        assert!(app.running_process.is_none());
        assert!(!app.model_action_pending);
        assert!(!app.chat_connection_pending);
    }

    #[test]
    fn chat_profiles_show_serving_facts_and_keys_without_native_panels() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        focus_stack_profile_by_id(&mut app, "local-chat");
        for (width, height) in [(80, 24), (120, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| app.draw(frame)).unwrap();
            let text = terminal_text(&terminal);
            assert!(text.contains("local-chat"), "{text}");
            assert!(text.contains("Enter chat · L load · U unload"), "{text}");
            assert!(text.contains("tasks     chat"), "{text}");
            assert!(!text.contains("Native task results"), "{text}");
            assert!(!text.contains("Coordinator · observation/cancel only"));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_embedding_form_needs_no_socket_or_caller_id_and_requires_connected_chat() {
        let fixture = AppFixture::new();
        write_embedding_stack_fixture(&fixture);
        let mut app = fixture.app();
        open_direct_embedding_form(&mut app, r#"["private text π"]"#);
        assert_eq!(app.input_mode, InputMode::TaskForm);
        let editor = app.task_form_editor.as_ref().unwrap();
        assert_eq!(editor.active_field(), TaskFormField::Timeout);
        assert!(!editor.fields.contains(&TaskFormField::DeclaredTask));
        assert_eq!(editor.values.embedding_inputs, r#"["private text π"]"#);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL))
            .unwrap();
        assert!(app.status.contains("Connect Chat"));
        assert!(app.task_form_editor.is_some());
        assert!(app.direct_native_job.is_none());
        assert!(app.job_history.records().is_empty());
        assert!(!app.status.contains("private text π"));
        assert!(!app.log.iter().any(|line| line.contains("private text π")));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn embedding_form_uses_native_text_batch_and_never_chat_turns() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        fs::write(
            fixture.root.join("llama-swap.yaml"),
            "models:\n  vector-fixture:\n    cmd: server --embedding\n",
        )
        .unwrap();
        let catalog = Catalog::load(&fixture.root).unwrap();
        let profile = catalog
            .list()
            .iter()
            .find(|profile| profile.key.profile_id.as_str() == "vector-fixture")
            .unwrap();
        let mut editor = TaskFormEditor::new_with_direct_embeddings(profile, true);
        editor.values.timeout = "1000".into();
        editor.selected_task = Some(0);
        assert_eq!(editor.declared_tasks, ["embeddings"]);
        assert!(editor.fields.contains(&TaskFormField::EmbeddingInputs));
        editor.values.embedding_inputs = r#"["first private text","second private text"]"#.into();
        let (request, timeout) = build_direct_native(&catalog, editor.to_draft().unwrap()).unwrap();
        let crate::control::requests::InferenceRequest::Embeddings { input, .. } = request else {
            panic!("embedding form must retain its native task")
        };
        assert_eq!(input, ["first private text", "second private text"]);
        assert_eq!(timeout, Duration::from_secs(1));
        editor.values.embedding_inputs = "[]".into();
        assert!(build_direct_native(&catalog, editor.to_draft().unwrap()).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_task_form_render_keeps_unicode_cursor_and_small_terminals_bounded() {
        let fixture = AppFixture::new();
        write_embedding_stack_fixture(&fixture);
        let catalog = Catalog::load(&fixture.root).unwrap();
        let profile = catalog
            .list()
            .iter()
            .find(|profile| profile.key.profile_id.as_str() == "local-embeddings")
            .unwrap();
        let mut app = fixture.app();
        let mut editor = TaskFormEditor::new_with_direct_embeddings(profile, true);
        editor.values.embedding_inputs = r#"["界e\u{301}"]"#.into();
        editor.active = 1;
        editor.sync_cursor();
        app.task_form_editor = Some(editor);
        app.input_mode = InputMode::TaskForm;
        for (width, height) in [(100, 30), (80, 24), (40, 12), (10, 5)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| app.draw_task_form(frame)).unwrap();
            let cursor = terminal.get_cursor_position().unwrap();
            assert!(
                cursor.x < width && cursor.y < height,
                "cursor outside {width}x{height}"
            );
        }
        assert!(!app.log.iter().any(|line| line.contains('界')));
        assert!(!app.status.contains('界'));
    }

    #[test]
    fn native_form_window_tracks_long_input_without_rendering_controls() {
        let value = format!("{}界e\u{301}\n\u{1b}TAIL", "prefix".repeat(100));
        let (visible, cursor) = task_form_text_window(&value, value.len(), 20);
        assert!(visible.starts_with('…'));
        assert!(visible.ends_with("TAIL"));
        assert!(visible.contains('↵'));
        assert!(!visible.contains('\n') && !visible.contains('\u{1b}'));
        assert!(UnicodeWidthStr::width(visible.as_str()) <= 20);
        assert!(cursor < 20);
        let (beginning, cursor) = task_form_text_window(&value, 0, 20);
        assert!(beginning.starts_with("prefix"));
        assert_eq!(cursor, 0);
        assert_eq!(task_form_text_window("界", 1, 20), (String::new(), 0));
        assert_eq!(task_form_text_window("abc", 3, 0), (String::new(), 0));
        let (bounded, _) = task_form_text_window(&"\u{301}".repeat(10_000), 20_000, 20);
        assert!(bounded.chars().count() <= 513);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn task_form_bounds_each_field_and_aggregate_before_typing_more() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let catalog = Catalog::load(&fixture.root).unwrap();
        let profile = catalog
            .list()
            .iter()
            .find(|profile| profile.key.profile_id.as_str() == "nagi")
            .unwrap();
        let mut editor = TaskFormEditor::new_with_direct_embeddings(profile, false);
        assert!(editor.direct_native_decision);
        editor.move_focus(1);
        assert_eq!(editor.active_field(), TaskFormField::DecisionState);
        editor.values.decision_state = "c".repeat(MAX_TASK_FORM_FIELD_BYTES);
        editor.sync_cursor();
        assert!(!editor.insert_char('x'));

        editor.values.timeout = "3".repeat(32);
        editor.values.decision_state = "c".repeat(400 * 1024);
        let remaining = MAX_TASK_FORM_TOTAL_BYTES
            - editor.values.timeout.len()
            - editor.values.decision_state.len();
        editor.values.decision_questions = "4".repeat(remaining);
        assert_eq!(editor.active_field(), TaskFormField::DecisionState);
        assert_eq!(editor.values.total_bytes(), MAX_TASK_FORM_TOTAL_BYTES);
        editor.sync_cursor();
        assert!(!editor.insert_char('x'));
    }

    #[test]
    fn unsupported_task_form_refuses_with_native_guidance() {
        let fixture = AppFixture::new();
        write_coordinator_decision_stack_fixture(&fixture);
        let mut app = fixture.app();
        focus_stack_profile_by_id(&mut app, "semif-fixture");
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.input_mode, InputMode::Normal);
        assert!(app
            .status
            .contains("No direct native workflow is available for this profile"));
        assert!(app.task_form_editor.is_none());
        assert!(app.direct_native_job.is_none());
        assert!(app.job_history.records().is_empty());
    }

    #[test]
    fn signal_theme_fills_the_frame_and_marks_the_three_lanes_at_80x24() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.sidebar_focused = true;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        let toolbar = terminal_row(&terminal, 0);
        assert!(toolbar.contains("L3MS"));
        assert_eq!(buffer[(79, 12)].bg, theme::BG);
        // Lane headings in the sidebar carry the lane colour.
        let lane_fg = |lane: &str| {
            (0..24).find_map(|y| {
                let row = terminal_row(&terminal, y);
                (row.trim_end_matches([' ', '│']).trim() == lane).then(|| {
                    let x = row.find(lane).unwrap() as u16;
                    buffer[(x, y)].fg
                })
            })
        };
        assert_eq!(lane_fg("language"), Some(theme::PHOSPHOR));
        assert_eq!(lane_fg("media"), Some(theme::CYAN));
        assert_eq!(lane_fg("decisions"), Some(theme::AMBER));
    }

    #[test]
    fn stack_catalog_marks_lanes_and_advertises_only_live_keys() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.view = View::Model;
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let text = terminal_text(&terminal);
        assert!(!text.contains("observe"), "removed key still advertised");
        assert!(text.contains("Enter chat · L load · U unload"), "{text}");
        let buffer = terminal.backend().buffer();
        let fg_of = |word: &str| {
            for y in 0..buffer.area.height {
                let row: Vec<String> = (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect();
                let line = row.concat();
                // Sidebar lane headings start in the first columns.
                if let Some(x) = line.find(word).filter(|x| *x < 12) {
                    return Some(buffer[(x as u16, y)].fg);
                }
            }
            None
        };
        assert_eq!(fg_of("media"), Some(theme::CYAN));
        assert_eq!(fg_of("decisions"), Some(theme::AMBER));
    }

    #[test]
    fn palette_lists_current_screen_commands_first_but_keeps_every_command_searchable() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Model;
        let owned_here = |spec: &crate::commands::CommandSpec| {
            spec.contexts
                .iter()
                .any(|c| *c == CommandContext::Global || *c == CommandContext::StackCatalog)
        };
        let all = app.palette_commands();
        let first_foreign = all.iter().position(|spec| !owned_here(spec)).unwrap();
        assert!(
            all[..first_foreign].iter().all(|spec| owned_here(spec)),
            "screen and global commands must precede the rest"
        );
        assert!(all[first_foreign..].iter().all(|spec| !owned_here(spec)));
        assert_eq!(
            all.len(),
            search_all_commands("").len(),
            "nothing is hidden"
        );

        // A command that belongs to another screen is still found by searching.
        app.palette_query = "native task".into();
        assert!(app
            .palette_commands()
            .iter()
            .any(|spec| spec.id == CommandId::StackCatalogForm));
    }

    #[test]
    fn stack_catalog_shows_live_router_state_and_dashes_for_everything_else() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.view = View::Model;
        focus_stack_profile_by_id(&mut app, "local-chat");
        app.models = vec![test_chat_model("local-chat", "loaded", false)];
        let mut terminal = Terminal::new(TestBackend::new(110, 30)).unwrap();
        let rows = |terminal: &Terminal<TestBackend>| -> Vec<String> {
            (0..30).map(|y| terminal_row(terminal, y)).collect()
        };
        // The sidebar is the first 31 columns of each row.
        let row_of = |screen: &[String], id: &str| {
            screen
                .iter()
                .map(|row| row.chars().take(31).collect::<String>())
                .find(|row| row.contains(&format!(" {id} ")))
                .unwrap_or_else(|| panic!("{id} missing from sidebar: {screen:#?}"))
                .clone()
        };
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let screen = rows(&terminal);
        assert!(row_of(&screen, "local-chat").contains("● local-chat"));
        assert!(screen.iter().any(|row| row.contains("local-chat ● loaded")));
        // Profiles the router does not list are unknown, not guessed.
        assert!(row_of(&screen, "breeze").contains("· breeze"));
        assert!(row_of(&screen, "nagi").contains("· nagi"));

        app.models = vec![test_chat_model("local-chat", "unloaded", false)];
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let screen = rows(&terminal);
        assert!(row_of(&screen, "local-chat").contains("○ local-chat"));
        assert!(screen
            .iter()
            .any(|row| row.contains("local-chat ○ unloaded")));
    }

    #[test]
    fn maintenance_panes_end_on_the_same_row_and_empty_chat_values_show_a_dash() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Maintenance;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        // The first box bottom below the tab bar must close both panes at once:
        // the list's and the editor's bottom-left corners share a row.
        let corners_in_row = |y: u16| {
            (0..buffer.area.width)
                .filter(|&x| buffer[(x, y)].symbol() == "└")
                .count()
        };
        let first_bottom = (4..buffer.area.height)
            .find(|&y| corners_in_row(y) > 0)
            .expect("a pane bottom exists");
        assert_eq!(
            corners_in_row(first_bottom),
            2,
            "list and editor boxes end on different rows"
        );

        app.view = View::Chat;
        app.chat_system_prompt.clear();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let text = terminal_text(&terminal);
        assert!(text.contains("system: —"), "{text}");
    }

    #[test]
    fn every_tab_and_overlay_renders_at_awkward_terminal_sizes_without_panicking() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.models = vec![
            test_chat_model(
                "a-model-with-a-rather-long-identifier-for-wrapping",
                "loaded",
                false,
            ),
            test_chat_model("short", "unloaded", false),
        ];
        let tabs = View::ALL;
        for (width, height) in [
            (1, 1),
            (10, 3),
            (20, 8),
            (40, 10),
            (60, 18),
            (80, 24),
            (200, 60),
        ] {
            for tab in tabs {
                for overlay in [0, 1, 2] {
                    app.view = tab;
                    app.show_palette = overlay == 1;
                    app.show_help = overlay == 2;
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal
                        .draw(|frame| app.draw(frame))
                        .unwrap_or_else(|_| panic!("draw failed at {width}x{height}"));
                }
            }
        }
    }

    #[test]
    fn byte_formatter_is_readable() {
        assert_eq!(format_bytes(10), "10 B");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_parameter_count(Some(7_200_000_000)), "7.2B");
        assert_eq!(format_parameter_count(None), "—");
        assert_eq!(
            format_system_time(Some(SystemTime::UNIX_EPOCH)),
            "1970-01-01 00:00Z"
        );
    }

    #[test]
    fn editor_cursor_width_uses_terminal_cells_not_utf8_bytes() {
        let mut buffer = TextBuffer::from_content("a界🙂e\u{301}");
        buffer.set_cursor_byte(buffer.content().len());
        assert_eq!(buffer.cursor_position().column, 5);
        assert_eq!(text_buffer_cursor_display_width(&buffer), 6);
    }

    #[test]
    fn cursor_visibility_uses_the_rendered_scroll_window() {
        assert_eq!(visible_cursor_offset(5, 2, 4), Some(3));
        assert_eq!(visible_cursor_offset(1, 2, 4), None);
        assert_eq!(visible_cursor_offset(6, 2, 4), None);
        assert_eq!(visible_cursor_offset(70_000, u16::MAX, 80), None);
    }

    #[test]
    fn shell_inventory_is_sorted_and_filtered() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("bench-z.sh"), "").unwrap();
        fs::write(directory.path().join("bench-a.sh"), "").unwrap();
        fs::write(directory.path().join("other.sh"), "").unwrap();
        let paths = collect_shell_scripts(directory.path(), "bench-");
        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("bench-a.sh"));
    }

    #[test]
    fn selected_download_keeps_all_patterns_and_config_base_directory() {
        let config = DownloadConfig {
            base_models_dir: "/models".into(),
            models: vec![ModelConfig {
                repo_id: "org/model".into(),
                allow_patterns: vec!["*Q4*".into(), "*mmproj*".into()],
                ignore_patterns: vec!["*old*".into(), "*debug*".into()],
                max_workers: Some(3),
                ..ModelConfig::default()
            }],
        };
        let speed_args = vec!["--max-workers".to_owned(), "7".to_owned()];
        let (_, command) =
            build_selected_download_command(Path::new("/repo"), &config, 0, &speed_args)
                .expect("download command");
        let command = command
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(!command[0].is_empty());
        assert_eq!(
            &command[1..],
            [
                "/repo/model_downloader/download_hf_model.py",
                "--repo-id",
                "org/model",
                "--allow-patterns",
                "*Q4*",
                "*mmproj*",
                "--ignore-patterns",
                "*old*",
                "*debug*",
                "--base-models-dir",
                "/models",
                "--max-workers",
                "7",
            ]
        );
    }

    #[test]
    fn download_keyboard_editor_crud_and_snapshot_actions_stay_synchronized() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;

        app.handle_key(KeyEvent::new(KeyCode::Char('i'), KeyModifiers::ALT))
            .unwrap();
        assert_eq!(
            app.download.focus(),
            DownloadFocus::Model(ModelField::RepoId)
        );
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE))
            .unwrap();
        assert!(!app.should_quit);
        assert_eq!(
            app.download
                .draft()
                .buffer(ModelField::RepoId)
                .unwrap()
                .content(),
            "org/oneq"
        );

        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(
            app.download.focus(),
            DownloadFocus::Model(ModelField::Description)
        );
        app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE))
            .unwrap();
        assert!(!app.show_help);
        app.execute_command(CommandId::DownloadApplyEdit).unwrap();
        assert_eq!(app.download.selected_model().unwrap().repo_id, "org/oneq");
        assert_eq!(app.download_state.selected(), Some(0));

        app.execute_command(CommandId::DownloadSaveConfig).unwrap();
        assert!(!app.download.is_dirty());
        assert_eq!(app.download.versions().len(), 1);
        assert_eq!(
            load_config_strict(app.download.editor().config_path())
                .unwrap()
                .models[0]
                .repo_id,
            "org/oneq"
        );

        app.execute_command(CommandId::DownloadAddModel).unwrap();
        assert_eq!(app.download.selected_index(), Some(2));
        assert_eq!(app.download_state.selected(), Some(2));
        app.download
            .draft_mut()
            .buffer_mut(ModelField::RepoId)
            .unwrap()
            .set_content("org/three");
        app.execute_command(CommandId::DownloadApplyEdit).unwrap();
        assert_eq!(app.download.selected_model().unwrap().repo_id, "org/three");
        app.execute_command(CommandId::DownloadDeleteModel).unwrap();
        assert_eq!(app.download.models().len(), 2);
        assert_eq!(app.download.selected_index(), Some(1));
        assert_eq!(app.download_state.selected(), Some(1));
    }

    #[test]
    fn failed_runtime_download_reload_retains_valid_dirty_state_and_allows_recovery() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        app.download
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("changed");
        app.save_download_config();
        assert_eq!(app.download.versions().len(), 1);
        let original_version = app.download.versions()[0].clone();

        app.download
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("unsaved recovery");
        assert!(app.download.is_dirty());

        let config_path = app.download.editor().config_path().to_path_buf();
        fs::write(&config_path, "{not json").unwrap();
        let before = app.download.config().clone();
        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::ALT))
            .unwrap();
        assert!(app.download_reload_armed);
        assert!(app.download.is_dirty());
        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::ALT))
            .unwrap();
        assert!(!app.download_reload_armed);
        assert!(app.download_load_error.is_none());
        assert_eq!(app.download.config(), &before);
        assert_eq!(
            app.download
                .draft()
                .buffer(ModelField::Description)
                .unwrap()
                .content(),
            "unsaved recovery"
        );
        assert!(app.download.is_dirty());

        app.save_download_config();
        assert!(!app.download.is_dirty());
        assert_eq!(
            load_config_strict(&config_path).unwrap().models[0].description,
            "unsaved recovery"
        );

        let original_index = app
            .download
            .versions()
            .iter()
            .position(|version| version == &original_version)
            .unwrap();
        app.download.select_version(original_index);
        app.restore_download_config();
        assert!(app.download_load_error.is_none());
        assert_eq!(
            load_config_strict(&config_path).unwrap().models[0].description,
            "first"
        );
        assert_eq!(app.download.config().models[0].description, "first");
        assert_eq!(
            app.download.config_path_buffer().content(),
            config_path.display().to_string()
        );
    }

    #[test]
    fn dirty_download_restore_requires_two_consecutive_triggers() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        app.download
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("saved state");
        app.save_download_config();
        let original_version = app.download.versions()[0].clone();
        let original_index = app
            .download
            .versions()
            .iter()
            .position(|version| version == &original_version)
            .unwrap();
        app.download.select_version(original_index);
        app.download.set_focus(DownloadFocus::Versions);
        app.download
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("dirty state");

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .unwrap();
        assert!(app.download_restore_armed);
        assert!(app.download.is_dirty());
        assert_eq!(
            load_config_strict(app.download.editor().config_path())
                .unwrap()
                .models[0]
                .description,
            "saved state"
        );

        app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
            .unwrap();
        assert!(!app.download_restore_armed);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .unwrap();
        assert!(app.download_restore_armed);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .unwrap();
        assert!(!app.download_restore_armed);
        assert!(!app.download.is_dirty());
        assert_eq!(app.download.config().models[0].description, "first");
    }

    #[test]
    fn snapshot_history_failure_never_becomes_false_save_success() {
        let fixture = AppFixture::new();
        let history_root = fixture.root.join(".toolkit/download_config_versions");
        fs::create_dir_all(history_root.parent().unwrap()).unwrap();
        fs::write(&history_root, "not a directory").unwrap();
        let config_path = fixture.root.join("model_downloader/models_config.json");
        let before = fs::read(&config_path).unwrap();

        let mut app = fixture.app();
        assert!(app.download_load_error.is_none());
        assert!(app
            .download_log
            .iter()
            .any(|line| line.contains("snapshots could not be listed")));
        app.download
            .draft_mut()
            .buffer_mut(ModelField::Description)
            .unwrap()
            .set_content("must not be reported saved");
        app.save_download_config();

        assert_eq!(app.status, "Could not save download config");
        assert!(app.download.is_dirty());
        assert_eq!(fs::read(&config_path).unwrap(), before);
    }

    #[test]
    fn download_speed_precedence_is_global_then_model_then_slow() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.download
            .draft_mut()
            .buffer_mut(ModelField::MaxWorkers)
            .unwrap()
            .set_content("3");
        app.download.apply_selected().unwrap();
        assert_eq!(
            app.selected_download_speed_args(0).unwrap(),
            ["--max-workers", "3"]
        );
        app.download.global_workers_buffer_mut().set_content("7");
        assert_eq!(
            app.selected_download_speed_args(0).unwrap(),
            ["--max-workers", "7"]
        );
        app.download.global_workers_buffer_mut().set_content("");
        app.download
            .draft_mut()
            .buffer_mut(ModelField::MaxWorkers)
            .unwrap()
            .set_content("");
        app.download.apply_selected().unwrap();
        assert_eq!(app.selected_download_speed_args(0).unwrap(), ["--slow"]);
    }

    #[cfg(unix)]
    #[test]
    fn selected_and_enabled_downloads_use_supervision_speed_args_and_dedicated_logs() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        app.download.global_workers_buffer_mut().set_content("7");

        app.download_selected();
        assert!(app.download_preflight_pending.is_some());
        app.download.global_workers_buffer_mut().set_content("99");
        wait_for_download(&mut app);
        let selected = &app.job_history.records()[0];
        assert_eq!(selected.kind, "download");
        assert!(selected
            .command
            .windows(2)
            .any(|parts| parts == ["--repo-id", "org/one"]));
        assert!(selected
            .command
            .windows(2)
            .any(|parts| parts == ["--max-workers", "7"]));
        assert!(!selected.command.iter().any(|part| part == "--slow"));
        assert!(!selected
            .command
            .iter()
            .any(|part| part == "--estimate-json"));
        assert!(app
            .download_log
            .iter()
            .any(|line| line.contains("Estimate: 200 B to download / 300 B matched")));

        let job_id = selected.id;
        app.sender
            .send(BackgroundEvent::ProcessLine {
                job_id,
                line: "download-log-marker".into(),
            })
            .unwrap();
        app.drain_background_events();
        assert!(app
            .download_log
            .iter()
            .any(|line| line == "download-log-marker"));

        app.download.global_workers_buffer_mut().set_content("");
        app.download_enabled();
        assert!(app.download_preflight_pending.is_some());
        wait_for_download(&mut app);
        let enabled = &app.job_history.records()[0];
        assert_eq!(enabled.kind, "download");
        assert!(enabled.command.iter().any(|part| part == "--config"));
        assert!(enabled.command.iter().any(|part| part == "--slow"));
    }

    #[cfg(unix)]
    #[test]
    fn failed_download_preflight_is_visible_and_preserves_the_actual_launch() {
        let fixture = AppFixture::new();
        fs::write(
            fixture.root.join("model_downloader/download_hf_model.py"),
            concat!(
                "#!/usr/bin/env python3\n",
                "import sys\n",
                "if '--estimate-json' in sys.argv:\n",
                "    print('estimate exploded', file=sys.stderr)\n",
                "    raise SystemExit(7)\n",
                "print('actual-download-ran')\n",
            ),
        )
        .unwrap();
        let mut app = fixture.app();
        app.view = View::Download;

        app.download_selected();
        wait_for_download(&mut app);

        let job = &app.job_history.records()[0];
        assert_eq!(job.kind, "download");
        assert!(!job.command.iter().any(|part| part == "--estimate-json"));
        assert!(app.download_log.iter().any(
            |line| line.contains("preflight unavailable") && line.contains("estimate exploded")
        ));
        assert!(app
            .download_log
            .iter()
            .any(|line| line == "actual-download-ran"));
    }

    #[cfg(unix)]
    #[test]
    fn escape_cancels_download_preflight_without_starting_the_download() {
        let fixture = AppFixture::new();
        fs::write(
            fixture.root.join("model_downloader/download_hf_model.py"),
            concat!(
                "#!/usr/bin/env python3\n",
                "import sys\n",
                "import time\n",
                "if '--estimate-json' in sys.argv:\n",
                "    time.sleep(5)\n",
                "    raise SystemExit(0)\n",
                "print('actual-download-must-not-run')\n",
            ),
        )
        .unwrap();
        let mut app = fixture.app();
        app.view = View::Download;
        app.download_selected();
        assert!(app.download_preflight_pending.is_some());

        let started = Instant::now();
        app.handle_download_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while app.download_preflight_pending.is_some() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }

        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(app.download_preflight_pending.is_none());
        assert!(app.running_process.is_none());
        assert!(app.job_history.records().is_empty());
        assert_eq!(app.status, "Download preflight cancelled");
        assert!(!app
            .download_log
            .iter()
            .any(|line| line.contains("actual-download-must-not-run")));
    }

    #[test]
    fn stale_download_disk_results_are_ignored() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.download_disk_request_id = 10;
        app.download_disk_space = "current disk state".into();
        app.sender
            .send(BackgroundEvent::DownloadDiskSpace {
                request_id: 9,
                target: PathBuf::from("/stale"),
                result: Ok(DiskSpace {
                    total_bytes: 200,
                    free_bytes: 100,
                }),
            })
            .unwrap();

        app.drain_background_events();

        assert_eq!(app.download_disk_space, "current disk state");
    }

    #[test]
    fn stale_chat_events_are_ignored_after_a_new_request() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.chat_pending = true;
        app.chat_stream_request_id = 8;
        app.chat_streaming = "current".into();
        app.sender
            .send(BackgroundEvent::ChatDelta {
                request_id: 7,
                delta: "stale".into(),
            })
            .unwrap();
        app.sender
            .send(BackgroundEvent::ChatFinished {
                request_id: 7,
                result: Ok(ChatCompletion {
                    content: "stale completion".into(),
                    completion_tokens: Some(1),
                    elapsed: Duration::from_secs(1),
                }),
            })
            .unwrap();

        app.drain_background_events();

        assert!(app.chat_pending);
        assert_eq!(app.chat_streaming, "current");
        assert!(app
            .chat_history
            .records()
            .iter()
            .all(|message| message.content != "stale completion"));
    }

    #[test]
    fn chat_endpoint_commits_only_after_the_current_probe_succeeds() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.chat_endpoint_draft = "http://draft.example".into();
        app.chat_connection_request_id = 2;
        let client = ChatClient::new("http://connected.example", None).unwrap();
        app.sender
            .send(BackgroundEvent::ChatConnected {
                request_id: 1,
                endpoint: client.base_url().into(),
                client: client.clone(),
                inventory: ChatModelInventory::default(),
            })
            .unwrap();
        app.drain_background_events();
        assert!(app.chat_client.is_none());
        assert_eq!(app.chat_endpoint_committed, None);
        assert_eq!(app.chat_endpoint_draft, "http://draft.example");

        app.sender
            .send(BackgroundEvent::ChatConnected {
                request_id: 2,
                endpoint: client.base_url().into(),
                client,
                inventory: ChatModelInventory {
                    models: vec![SwapModel {
                        id: "chat-model".into(),
                        state: "loaded".into(),
                        name: String::new(),
                        description: String::new(),
                        created: None,
                        disabled: false,
                        size_bytes: None,
                    }],
                    non_chat_model_ids: Vec::new(),
                },
            })
            .unwrap();
        app.drain_background_events();
        assert_eq!(
            app.chat_endpoint_committed.as_deref(),
            Some("http://connected.example")
        );
        assert_eq!(app.selected_chat_model().unwrap().id, "chat-model");
    }

    #[test]
    fn chat_keyboard_and_persistence_guards_prioritize_stop() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Chat;
        app.chat_pending = true;
        app.chat_history.push("user", "in progress");
        app.chat_stream_cancellation = Some(Arc::new(AtomicBool::new(false)));

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        assert!(!app.chat_pending);
        assert_eq!(app.status, "Chat response stopped");

        app.chat_pending = true;
        app.save_chat_session();
        assert!(app.status.contains("before saving"));
        app.open_chat_sessions();
        assert!(app.status.contains("before loading"));
        assert!(!app.show_chat_sessions);
    }

    #[test]
    fn stack_chat_handoff_selects_exact_route_and_preserves_chat_state() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        focus_stack_profile_by_id(&mut app, "local-chat");
        connect_test_chat(
            &mut app,
            vec![
                test_chat_model("decision-engine", "loaded", false),
                test_chat_model("image-engine", "loaded", false),
                test_chat_model("audio-engine", "loaded", false),
                test_chat_model("embeddings-engine", "loaded", false),
                test_chat_model("other-loaded-model", "loaded", false),
                test_chat_model("local-chat", "unloaded", false),
            ],
        );
        app.chat_endpoint_draft = "http://uncommitted-draft.example".into();
        app.chat_input = "unsent chat draft".into();
        app.chat_system_prompt = "preserved system prompt".into();
        app.chat_history.push("user", "earlier conversation");
        app.chat_sessions = vec![ChatSessionSummary {
            file_name: "saved-session.json".into(),
            saved: "2026-10-10T00:00:00Z".into(),
            message_count: 2,
            json_path: PathBuf::from("saved-session.json"),
            markdown_path: None,
        }];
        let previous_history = app.chat_history.clone();
        let previous_sessions = app.chat_sessions.clone();
        let previous_stream_id = app.chat_stream_request_id;

        // Non-chat models before the Chat route must not shift the filtered
        // picker index when the exact route is selected.
        app.chat_non_chat_model_ids = vec![
            "decision-engine".into(),
            "image-engine".into(),
            "audio-engine".into(),
            "embeddings-engine".into(),
        ];
        app.initialize_chat_model_selection();
        assert_eq!(app.selected_chat_model().unwrap().id, "other-loaded-model");
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
            .unwrap();

        assert_eq!(app.view, View::Chat);
        assert_eq!(app.input_mode, InputMode::ChatMessage);
        assert_eq!(app.selected_chat_model().unwrap().id, "local-chat");
        assert_eq!(app.chat_endpoint_draft, "http://uncommitted-draft.example");
        assert_eq!(
            app.chat_endpoint_committed.as_deref(),
            Some("http://connected.example")
        );
        assert_eq!(
            app.chat_client.as_ref().unwrap().base_url(),
            "http://connected.example"
        );
        assert_eq!(app.chat_input, "unsent chat draft");
        assert_eq!(app.chat_system_prompt, "preserved system prompt");
        assert_eq!(app.chat_history, previous_history);
        assert_eq!(app.chat_sessions, previous_sessions);
        assert_eq!(app.chat_stream_request_id, previous_stream_id);
        assert!(!app.chat_pending);
        assert!(!app.chat_connection_pending);
        assert!(app.task_form_editor.is_none());
        assert!(app.status.contains("no message sent"));
    }

    #[test]
    fn stack_chat_handoff_refuses_missing_disabled_disconnected_and_active_routes() {
        #[derive(Clone, Copy)]
        enum Refusal {
            MissingModel,
            DisabledProfile,
            DisabledModel,
            NonChatModel,
            Disconnected,
            EndpointMismatch,
            ActiveRequest,
            SessionPending,
            ConnectionPending,
            ChangedProfile,
        }

        let cases = [
            (Refusal::MissingModel, "not advertised"),
            (Refusal::DisabledProfile, "profile is disabled"),
            (Refusal::DisabledModel, "disabled in Chat"),
            (Refusal::NonChatModel, "advertises non-chat output"),
            (Refusal::Disconnected, "Connect Chat"),
            (Refusal::EndpointMismatch, "not a confirmed connection"),
            (Refusal::ActiveRequest, "active Chat response"),
            (Refusal::SessionPending, "active Chat response or session"),
            (Refusal::ConnectionPending, "current Chat connection"),
            (Refusal::ChangedProfile, "profile changed"),
        ];

        for (case, expected_status) in cases {
            let fixture = AppFixture::new();
            write_stack_fixture(&fixture);
            if matches!(case, Refusal::DisabledProfile) {
                fs::write(
                    fixture.root.join("llama-swap.yaml"),
                    "models:\n  local-chat:\n    disabled: true\n    cmd: server\n",
                )
                .unwrap();
            }
            let mut app = fixture.app();
            focus_stack_profile_by_id(&mut app, "local-chat");
            let advertised_models = if matches!(case, Refusal::MissingModel) {
                vec![test_chat_model("other-model", "loaded", false)]
            } else if matches!(case, Refusal::DisabledModel) {
                vec![test_chat_model("local-chat", "loaded", true)]
            } else {
                vec![test_chat_model("local-chat", "loaded", false)]
            };
            connect_test_chat(&mut app, advertised_models);
            if matches!(case, Refusal::NonChatModel) {
                app.chat_non_chat_model_ids = vec!["local-chat".into()];
            }
            match case {
                Refusal::Disconnected => {
                    app.chat_client = None;
                    app.chat_endpoint_committed = None;
                }
                Refusal::ActiveRequest => app.chat_pending = true,
                Refusal::EndpointMismatch => {
                    app.chat_endpoint_committed = Some("http://different.example".into());
                }
                Refusal::SessionPending => app.chat_session_pending = true,
                Refusal::ConnectionPending => app.chat_connection_pending = true,
                Refusal::ChangedProfile => fs::write(
                    fixture.root.join("llama-swap.yaml"),
                    "models:\n  local-chat:\n    cmd: server --embedding\n",
                )
                .unwrap(),
                Refusal::MissingModel
                | Refusal::DisabledProfile
                | Refusal::DisabledModel
                | Refusal::NonChatModel => {}
            }
            let input_before = app.chat_input.clone();
            app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE))
                .unwrap();

            assert_eq!(app.view, View::Model, "case: {expected_status}");
            assert_eq!(app.input_mode, InputMode::Normal, "case: {expected_status}");
            assert!(app.task_form_editor.is_none(), "case: {expected_status}");
            assert!(app.status.contains(expected_status), "{}", app.status);
            assert_eq!(app.chat_input, input_before);
            assert!(!app.chat_connection_pending || matches!(case, Refusal::ConnectionPending));
            assert!(!app.chat_pending || matches!(case, Refusal::ActiveRequest));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn active_or_unconfirmed_native_work_blocks_chat_without_losing_the_draft() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        let previous_history = app.chat_history.clone();
        app.chat_input = "unsent draft".into();
        app.input_mode = InputMode::ChatMessage;
        app.direct_native_admission_closed = true;
        app.direct_native_summary = Some(DirectNativeSummary {
            job_id: 1,
            lane: Lane::Decisions,
            profile_id: "jev-api".into(),
            task: "decision".into(),
            requires_idle_gpu_window: false,
            native_id: None,
            state: DirectNativeState::Interrupted,
            result: None,
        });
        app.send_chat_message();
        assert_eq!(app.chat_input, "unsent draft");
        assert_eq!(app.input_mode, InputMode::ChatMessage);
        assert!(!app.chat_pending);
        assert_eq!(app.chat_history, previous_history);
        assert!(app
            .status
            .contains("foreground native task is active or unconfirmed"));

        app.direct_native_admission_closed = false;
        app.direct_native_summary = Some(DirectNativeSummary {
            job_id: 2,
            lane: Lane::Decisions,
            profile_id: "nagi".into(),
            task: "decision".into(),
            requires_idle_gpu_window: false,
            native_id: None,
            state: DirectNativeState::Running,
            result: None,
        });
        let (_sender, receiver) = mpsc::sync_channel(1);
        app.direct_native_job = Some(DirectNativeJob {
            job_id: 2,
            cancellation: Arc::new(AtomicBool::new(false)),
            receiver,
            worker: thread::spawn(|| {}),
        });
        app.chat_input = "unsent active-run draft".into();
        app.input_mode = InputMode::ChatMessage;
        app.send_chat_message();
        assert_eq!(app.chat_input, "unsent active-run draft");
        assert_eq!(app.input_mode, InputMode::ChatMessage);
        assert!(!app.chat_pending);
        assert_eq!(app.chat_history, previous_history);
        assert!(app
            .status
            .contains("foreground native task is active or unconfirmed"));
        let job = app.direct_native_job.take().unwrap();
        assert!(job.worker.join().is_ok());
    }

    #[test]
    fn chat_model_selection_is_independent_from_workbench_selection() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.loaded_model_id = Some("workbench-model".into());
        app.chat_models = vec![
            SwapModel {
                id: "chat-loaded".into(),
                state: "loaded".into(),
                name: String::new(),
                description: String::new(),
                created: None,
                disabled: false,
                size_bytes: None,
            },
            SwapModel {
                id: "chat-other".into(),
                state: "unloaded".into(),
                name: String::new(),
                description: String::new(),
                created: None,
                disabled: false,
                size_bytes: None,
            },
        ];
        app.initialize_chat_model_selection();

        assert_eq!(app.selected_chat_model().unwrap().id, "chat-loaded");
        assert_eq!(app.loaded_model_id.as_deref(), Some("workbench-model"));
    }

    #[test]
    fn chat_picker_skips_known_non_chat_models_but_preserves_unknown_and_text_models() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.chat_models = vec![
            test_chat_model("loaded-decision", "loaded", false),
            test_chat_model("image-output", "loaded", false),
            test_chat_model("audio-output", "loaded", false),
            test_chat_model("embedding-output", "loaded", false),
            // This model accepts image input but returns text, so it remains a
            // valid Chat choice.
            test_chat_model("text-image", "loaded", false),
            // Unknown output metadata and older metadata-less entries fail open.
            test_chat_model("custom-output", "loaded", false),
            test_chat_model("legacy-model", "unloaded", false),
            test_chat_model("chat-loaded", "loaded", false),
            test_chat_model("chat-unloaded", "unloaded", false),
        ];
        app.chat_non_chat_model_ids = vec![
            "loaded-decision".into(),
            "image-output".into(),
            "audio-output".into(),
            "embedding-output".into(),
        ];

        app.initialize_chat_model_selection();
        assert_eq!(app.selected_chat_model().unwrap().id, "text-image");
        assert_eq!(app.chat_models.len(), 9);
        assert_eq!(app.chat_picker_model_indices(), [4, 5, 6, 7, 8]);

        app.select_previous_chat_model();
        assert_eq!(app.selected_chat_model().unwrap().id, "chat-unloaded");
        app.select_next_chat_model();
        assert_eq!(app.selected_chat_model().unwrap().id, "text-image");

        app.chat_models.truncate(4);
        app.initialize_chat_model_selection();
        app.select_next_chat_model();
        assert!(app.selected_chat_model().is_none());
        assert_eq!(app.chat_models.len(), 4);
    }

    #[test]
    fn chat_shortcuts_keep_f2_composer_focus_and_modified_key_precedence() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.chat_client = Some(ChatClient::new("http://chat.example", None).unwrap());

        app.handle_key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.view, View::Chat);
        assert_eq!(app.input_mode, InputMode::ChatMessage);
        assert_eq!(
            app.modified_key_command(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL,)),
            Some(CommandId::ChatConnect)
        );
        assert_eq!(
            app.modified_key_command(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL,)),
            Some(CommandId::ChatDetect)
        );
    }

    #[test]
    fn download_targets_follow_runtime_root_and_custom_directories() {
        let root = Path::new("/runtime/repository");
        let mut config = DownloadConfig {
            base_models_dir: "relative/models".into(),
            models: vec![ModelConfig {
                repo_id: "org/model".into(),
                ..ModelConfig::default()
            }],
        };
        assert_eq!(
            selected_download_target(root, &config, 0).unwrap(),
            root.join("relative/models")
        );
        config.models[0].local_dir = "custom/model".into();
        assert_eq!(
            selected_download_target(root, &config, 0).unwrap(),
            root.join("custom/model")
        );
        config.models[0].local_dir = "/mounted/model".into();
        assert_eq!(
            selected_download_target(root, &config, 0).unwrap(),
            PathBuf::from("/mounted/model")
        );
        config.base_models_dir.clear();
        assert_eq!(
            config_download_target(root, &config),
            root.join("model_downloader/models")
        );
    }

    #[test]
    fn dirty_download_quit_confirmation_closes_palette_and_consumes_choice() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        app.focus_download_editor();
        app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.download.is_dirty());
        app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
            .unwrap();
        assert!(app.show_palette);
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .unwrap();
        assert!(!app.show_palette);
        assert!(app.show_quit_confirmation);
        assert!(!app.should_quit);
        app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.should_quit);
    }

    #[test]
    fn download_view_renders_controls_editor_table_and_dedicated_log() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        app.push_download_log("download-only-marker");
        // A 150-column pane beside the sidebar.
        let backend = TestBackend::new(150 + shell::SIDEBAR_WIDTH, 48);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();

        let rendered = terminal_text(&terminal);
        for expected in [
            "Config path",
            "Base models dir",
            "Global workers",
            "Model editor",
            "force_download",
            "max_workers",
            "Pattern",
            "Download activity",
            "download-only-marker",
        ] {
            assert!(
                rendered.contains(expected),
                "missing rendered text: {expected}"
            );
        }
    }

    #[test]
    fn compact_download_view_keeps_late_model_field_and_cursor_visible_at_80x24() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        app.push_download_log("compact-download-marker");
        app.download
            .draft_mut()
            .buffer_mut(ModelField::MaxWorkers)
            .unwrap()
            .set_content("workers-prefix-1234567890-TAIL80");
        app.download
            .set_focus(DownloadFocus::Model(ModelField::MaxWorkers));

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();

        let rendered = terminal_text(&terminal);
        for expected in [
            "Settings",
            "Config:",
            "Snapshot:",
            "Base:",
            "Slow:",
            "Workers:",
            "Note:",
            "Models",
            "Repository",
            "Model editor",
            "force_download:",
            "max_workers:",
            "TAIL80",
            "Download activity",
            "compact-download-marker",
        ] {
            assert!(
                rendered.contains(expected),
                "missing compact rendered text: {expected}"
            );
        }

        let cursor = terminal.get_cursor_position().unwrap();
        assert!(cursor.x < 80 && cursor.y < 24, "cursor out of bounds");
        let cursor_row = terminal_row(&terminal, cursor.y);
        assert!(cursor_row.contains("max_workers:"));
        assert!(cursor_row.contains("TAIL80"));
    }

    #[test]
    fn compact_download_view_scrolls_focused_setting_at_100x30() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        let long_base = format!("/models/{}/TAIL100", "nested-directory/".repeat(10));
        app.download
            .base_models_dir_buffer_mut()
            .set_content(long_base);
        app.download.set_focus(DownloadFocus::BaseModelsDir);
        app.push_download_log("compact-settings-marker");

        // A 100-column pane beside the sidebar.
        let backend = TestBackend::new(100 + shell::SIDEBAR_WIDTH, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();

        let rendered = terminal_text(&terminal);
        for expected in [
            "Settings",
            "Base:",
            "TAIL100",
            "Models",
            "Repository",
            "Model editor",
            "max_workers:",
            "Download activity",
            "compact-settings-marker",
        ] {
            assert!(
                rendered.contains(expected),
                "missing compact rendered text: {expected}"
            );
        }

        let cursor = terminal.get_cursor_position().unwrap();
        assert!(
            cursor.x < 100 + shell::SIDEBAR_WIDTH && cursor.y < 30,
            "cursor out of bounds"
        );
        let cursor_row = terminal_row(&terminal, cursor.y);
        assert!(cursor_row.contains("Base:"));
        assert!(cursor_row.contains("TAIL100"));
    }

    #[test]
    fn compact_download_view_uses_focused_editor_when_too_narrow_for_both_panes() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        app.download
            .draft_mut()
            .buffer_mut(ModelField::MaxWorkers)
            .unwrap()
            .set_content("fallback-TAIL60");
        app.download
            .set_focus(DownloadFocus::Model(ModelField::MaxWorkers));

        let backend = TestBackend::new(60, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();

        let rendered = terminal_text(&terminal);
        assert!(rendered.contains("Model editor"));
        assert!(rendered.contains("max_workers:"));
        assert!(rendered.contains("TAIL60"));
        assert!(!rendered.contains("Repository"));
        let cursor = terminal.get_cursor_position().unwrap();
        assert!(terminal_row(&terminal, cursor.y).contains("max_workers:"));
    }

    #[test]
    fn process_output_lines_are_bounded() {
        let mut bytes = vec![b'x'; MAX_PROCESS_LINE_BYTES * 2];
        bytes.push(b'\n');
        let (sender, receiver) = mpsc::sync_channel(2);
        stream_reader(io::Cursor::new(bytes), 42, sender);
        let event = receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("bounded output event");
        let BackgroundEvent::ProcessLine { job_id, line } = event else {
            panic!("unexpected background event");
        };
        assert_eq!(job_id, 42);
        assert!(line.len() < MAX_PROCESS_LINE_BYTES + 64);
        assert!(line.ends_with("[line truncated]"));
    }

    #[test]
    fn streaming_preview_is_bounded_at_utf8_boundaries() {
        let mut preview = "old".to_owned();
        append_bounded_text(&mut preview, "éclair", 6);
        assert_eq!(preview, "clair");
        append_bounded_text(&mut preview, "ignored", 0);
        assert!(preview.is_empty());
    }

    #[test]
    fn script_editor_input_navigation_and_snapshot_save_are_consistent() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Bench;
        app.execute_command(CommandId::BenchFocusEditor).unwrap();

        let before = app.bench_editor.content().to_owned();
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE))
            .unwrap();
        assert!(!app.should_quit);
        assert_ne!(app.bench_editor.content(), before);
        let edited = app.bench_editor.content().to_owned();

        app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE))
            .unwrap();
        assert!(!app.show_help);
        assert_ne!(app.bench_editor.content(), edited);
        let saved = app.bench_editor.content().to_owned();

        app.handle_key(KeyEvent::new(KeyCode::F(7), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.view, View::Maintenance);
        assert!(app.bench_editor.is_dirty());
        app.handle_key(KeyEvent::new(KeyCode::F(4), KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::ALT))
            .unwrap();

        assert!(!app.bench_editor.is_dirty());
        assert_eq!(fs::read_to_string(&fixture.bench_first).unwrap(), saved);
        assert!(!app.bench_editor.versions().is_empty());
    }

    #[test]
    fn dirty_script_blocks_selection_until_explicit_reload() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Bench;
        app.focus_script_editor(ScriptEditorTarget::Bench);
        app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();

        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.bench_state.selected(), Some(0));
        assert_eq!(
            app.bench_editor.selected_path(),
            Some(fixture.bench_first.as_path())
        );

        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::ALT))
            .unwrap();
        assert!(app.bench_editor.is_dirty());
        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::ALT))
            .unwrap();
        assert!(!app.bench_editor.is_dirty());
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.bench_state.selected(), Some(1));
        assert_eq!(
            app.bench_editor.selected_path(),
            Some(fixture.bench_second.as_path())
        );
    }

    #[test]
    fn bench_filter_tracks_keyboard_selection_editor_and_no_results() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Bench;

        app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.input_mode, InputMode::BenchFilter);
        for character in "bench-b".chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE))
                .unwrap();
        }
        app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.bench_filter, "bench-");
        assert_eq!(
            app.selected_script_path(ScriptEditorTarget::Bench),
            Some(fixture.bench_second.clone())
        );
        app.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.bench_filter, "bench-b");
        assert_eq!(app.bench_state.selected(), Some(0));
        assert_eq!(
            app.selected_script_path(ScriptEditorTarget::Bench),
            Some(fixture.bench_second.clone())
        );
        assert_eq!(
            app.bench_editor.selected_path(),
            Some(fixture.bench_second.as_path())
        );

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL))
            .unwrap();
        assert_eq!(app.script_input_target, Some(ScriptEditorTarget::Bench));
        app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        assert!(app.bench_editor.is_dirty());

        app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.bench_filter, "bench-b");
        assert!(app.status.contains("Save or reload"));
        assert_eq!(
            app.bench_editor.selected_path(),
            Some(fixture.bench_second.as_path())
        );

        app.reload_script_editor(ScriptEditorTarget::Bench);
        app.reload_script_editor(ScriptEditorTarget::Bench);
        app.set_bench_filter("missing".into());
        assert!(app.visible_bench_scripts().is_empty());
        assert_eq!(app.bench_state.selected(), None);
        assert_eq!(app.bench_editor.selected_path(), None);

        let backend = TestBackend::new(120, 36);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        assert!(terminal_text(&terminal).contains("No matching scripts"));
    }

    #[test]
    fn bench_inventory_order_is_stable_when_files_are_created_in_reverse_order() {
        let fixture = AppFixture::new();
        fs::remove_file(&fixture.bench_first).unwrap();
        fs::remove_file(&fixture.bench_second).unwrap();
        fs::write(&fixture.bench_second, "#!/bin/sh\necho b\n").unwrap();
        fs::write(&fixture.bench_first, "#!/bin/sh\necho a\n").unwrap();

        let app = fixture.app();
        assert_eq!(
            app.bench_scripts,
            vec![fixture.bench_first.clone(), fixture.bench_second.clone()]
        );
        assert_eq!(
            app.visible_bench_scripts()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![fixture.bench_first.clone(), fixture.bench_second.clone()]
        );
        assert_eq!(
            app.selected_script_path(ScriptEditorTarget::Bench),
            Some(fixture.bench_first.clone())
        );
    }

    #[test]
    fn bench_filter_runs_the_filtered_script() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Bench;
        app.set_bench_filter("bench-b".into());

        app.run_selected_bench();
        let job = app.job_history.records().first().expect("bench job");
        assert_eq!(job.kind, "bench");
        assert_eq!(job.command.first(), Some(&"bash".to_owned()));
        assert_eq!(
            job.command.get(1),
            Some(&fixture.bench_second.to_string_lossy().into_owned())
        );
        wait_for_download(&mut app);
        assert_eq!(app.job_history.records()[0].exit_code, Some(0));
    }

    #[test]
    fn bench_filter_is_available_in_palette_and_help() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Bench;

        assert!(visible_commands(CommandContext::Bench)
            .iter()
            .any(|spec| spec.id == CommandId::BenchFocusFilter));
        app.open_palette();
        app.palette_query = "script filter".into();
        assert!(app
            .palette_commands()
            .iter()
            .any(|spec| spec.id == CommandId::BenchFocusFilter));

        app.show_palette = false;
        app.execute_command(CommandId::BenchFocusFilter).unwrap();
        assert_eq!(app.input_mode, InputMode::BenchFilter);
        app.input_mode = InputMode::Normal;
        app.show_help = true;
        let backend = TestBackend::new(120, 36);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let rendered = terminal_text(&terminal);
        assert!(rendered.contains("Ctrl+F / /"));
        assert!(rendered.contains("Filter scripts"));
    }

    #[test]
    fn dirty_quit_requires_an_explicit_choice() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Bench;
        app.focus_script_editor(ScriptEditorTarget::Bench);
        app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.show_quit_confirmation);
        assert!(!app.should_quit);
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        assert!(!app.show_quit_confirmation);
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.should_quit);
    }

    #[test]
    fn contextual_palette_actions_cannot_create_hidden_editors() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Model;
        app.execute_command(CommandId::MaintenanceFocusEditor)
            .unwrap();
        assert_eq!(app.script_input_target, None);
        assert!(app.status.contains("maintenance view"));
    }

    #[test]
    fn jobs_and_chat_persist_to_the_explicit_app_data_root() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.job_history.begin(
            "test".into(),
            "maintenance".into(),
            vec!["bash".into(), "maintenance/cleanup.sh".into()],
            "maintenance".into(),
            "maintenance/cleanup.sh".into(),
        );
        app.persist_jobs("test");
        assert!(fixture.data.path().join("jobs.json").is_file());

        app.chat_history.push("user", "alternate root");
        app.save_chat_session();
        let deadline = Instant::now() + Duration::from_secs(2);
        while app.chat_session_pending && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        assert!(!app.chat_session_pending);
        assert!(fixture.data.path().join("chats").is_dir());
        assert!(
            fs::read_dir(fixture.data.path().join("chats"))
                .unwrap()
                .count()
                >= 2
        );
    }

    #[cfg(unix)]
    #[test]
    fn process_group_signal_stops_and_reaps_shell() {
        let mut command = Command::new("sh");
        command
            .args(["-c", "sleep 30 & wait"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        let mut child = {
            let _fixture_guard = crate::control::process::executable_fixture_guard();
            command.spawn()
        }
        .expect("spawn isolated shell");
        let process_group = child.id();
        thread::sleep(Duration::from_millis(50));
        signal_process_group(process_group, "TERM").expect("signal process group");

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if child.try_wait().expect("poll child").is_some() {
                break;
            }
            if Instant::now() >= deadline {
                let _ = signal_process_group(process_group, "KILL");
                let _ = child.wait();
                panic!("process group did not stop after TERM");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn system_view_renders_gpu_memory_and_spill() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::System;

        // Populate sample telemetry
        app.gpu_telemetry = Some(GpuTelemetry {
            name: "NVIDIA GeForce RTX 4070".into(),
            temperature_c: 44,
            power_watts: 9.5,
            used_vram_mib: 11678,
            free_vram_mib: 604,
            total_vram_mib: 12282,
            gpu_util_percent: 0,
            mem_util_percent: 0,
        });
        app.sys_memory = Some(SystemMemoryTelemetry {
            total_ram_mib: 64000,
            available_ram_mib: 32000,
            used_ram_mib: 32000,
            total_swap_mib: 20000,
            used_swap_mib: 5000,
            free_swap_mib: 15000,
        });
        app.server_io = Some(ServerIoTelemetry {
            pid: 12345,
            read_bytes: 5_000_000,
            read_rate_bytes_per_sec: Some(5120.0),
        });
        for i in 0..30 {
            app.gpu_util_history.push_back(i * 3);
            app.vram_history.push_back(11600 + i);
        }

        let backend = TestBackend::new(140, 45);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| app.draw(frame)).unwrap();
        let text = terminal_text(&terminal);
        assert!(text.contains("RTX 4070"), "{text}");
        assert!(text.contains("PLE"), "{text}");
        // The status bar carries the same readings in compact form.
        assert!(text.contains("11.4/12G"), "{text}");
        assert!(text.contains("RAM"), "{text}");
    }

    fn screen_rows(terminal: &Terminal<TestBackend>) -> Vec<String> {
        let height = terminal.backend().buffer().area.height;
        (0..height).map(|y| terminal_row(terminal, y)).collect()
    }

    #[test]
    fn shell_and_every_view_render_readably_at_80x24_and_60x18() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.models = vec![test_chat_model("local-chat", "loaded", false)];
        for (width, height) in [(80, 24), (60, 18)] {
            // Sidebar focused: below 100 columns the sidebar takes the body.
            app.sidebar_focused = true;
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| app.draw(frame)).unwrap();
            let rows = screen_rows(&terminal);
            assert!(rows[0].contains("L3MS"), "{width}x{height}: {rows:#?}");
            assert!(rows.iter().any(|row| row.contains("Models 3")), "{rows:#?}");
            assert!(
                rows.iter().any(|row| row.contains("● local-chat")),
                "{rows:#?}"
            );
            assert!(rows.iter().any(|row| row.contains("Jobs 0")), "{rows:#?}");
            assert!(
                rows[height as usize - 1].contains("● local-chat"),
                "status bar: {rows:#?}"
            );

            // Pane focused: every view shows its name and some content.
            app.sidebar_focused = false;
            for view in View::ALL {
                app.view = view;
                terminal.draw(|frame| app.draw(frame)).unwrap();
                let rows = screen_rows(&terminal);
                assert!(
                    rows[0].contains(view.name()),
                    "{view:?} at {width}x{height}: {rows:#?}"
                );
                let body: String = rows[1..height as usize - 2].concat();
                let visible = body.chars().filter(|c| c.is_alphanumeric()).count();
                assert!(
                    visible > 20,
                    "{view:?} at {width}x{height} is blank: {rows:#?}"
                );
                assert!(rows.iter().all(|row| row.chars().count() == width as usize));
            }
        }
    }

    #[test]
    fn every_former_workbench_and_model_ops_command_has_a_new_home() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        // (former command, its replacement, the view the replacement lives in)
        let mapping = [
            (
                "workbench.select-previous",
                CommandId::SidebarSelectPrevious,
                View::Model,
            ),
            (
                "workbench.select-next",
                CommandId::SidebarSelectNext,
                View::Model,
            ),
            (
                "workbench.focus-filter",
                CommandId::SidebarFilter,
                View::Model,
            ),
            ("workbench.focus-table", CommandId::ToggleFocus, View::Model),
            (
                "workbench.refresh-models",
                CommandId::StackCatalogRefresh,
                View::Model,
            ),
            ("workbench.load-model", CommandId::ModelLoad, View::Model),
            (
                "workbench.unload-model",
                CommandId::ModelUnload,
                View::Model,
            ),
            ("workbench.clear-log", CommandId::ClearLog, View::Log),
            ("workbench.cycle-layout", CommandId::OpenSystem, View::Model),
            ("model-ops.toggle-mode", CommandId::OpenBench, View::Model),
            (
                "model-ops.select-previous",
                CommandId::BenchSelectPrevious,
                View::Bench,
            ),
            (
                "model-ops.select-next",
                CommandId::BenchSelectNext,
                View::Bench,
            ),
            ("model-ops.start", CommandId::BenchRun, View::Bench),
            ("model-ops.stop", CommandId::BenchStop, View::Bench),
            (
                "model-ops.refresh-models",
                CommandId::StackCatalogRefresh,
                View::Bench,
            ),
            (
                "model-ops.focus-filter",
                CommandId::BenchFocusFilter,
                View::Bench,
            ),
            (
                "model-ops.focus-table",
                CommandId::BenchFocusList,
                View::Bench,
            ),
            (
                "model-ops.focus-editor",
                CommandId::BenchFocusEditor,
                View::Bench,
            ),
            (
                "model-ops.save-script",
                CommandId::BenchSaveScript,
                View::Bench,
            ),
            (
                "model-ops.reload-script",
                CommandId::BenchReloadScript,
                View::Bench,
            ),
            (
                "model-ops.restore-script",
                CommandId::BenchRestoreScript,
                View::Bench,
            ),
            ("model-ops.clear-log", CommandId::ClearLog, View::Log),
        ];
        for (former, replacement, view) in mapping {
            let spec = command(replacement).unwrap();
            assert!(spec.is_visible(), "{former} → {replacement} is hidden");
            app.view = view;
            assert!(
                app.command_available(spec),
                "{former} → {replacement} is unavailable in {view:?}"
            );
            app.palette_query = spec.label.to_owned();
            assert!(
                app.palette_commands()
                    .iter()
                    .any(|found| found.id == replacement),
                "{former} → {replacement} is not in the palette"
            );
        }
        assert!(!CommandId::ALL.iter().any(
            |id| id.as_str().starts_with("workbench.") || id.as_str().starts_with("model-ops.")
        ));
    }

    #[test]
    fn sidebar_keys_move_sections_collapse_filter_and_open_views() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.sidebar_focused = true;
        let key = |app: &mut App, code: KeyCode| {
            app.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
                .unwrap();
        };
        app.stack_catalog_state.select(Some(0));
        key(&mut app, KeyCode::Char('j'));
        assert_eq!(app.stack_catalog_state.selected(), Some(1));
        key(&mut app, KeyCode::Char('k'));
        key(&mut app, KeyCode::Char('k'));
        assert_eq!(app.stack_catalog_state.selected(), Some(0));

        // The filter narrows navigation to matching profiles.
        key(&mut app, KeyCode::Char('/'));
        assert_eq!(app.input_mode, InputMode::ModelFilter);
        for character in "nag".chars() {
            key(&mut app, KeyCode::Char(character));
        }
        key(&mut app, KeyCode::Enter);
        let selected = app.selected_stack_profile().unwrap();
        assert_eq!(selected.key.profile_id.as_str(), "nagi");
        key(&mut app, KeyCode::Esc);
        assert!(app.model_filter.is_empty());

        key(&mut app, KeyCode::Char(']'));
        assert_eq!(app.sidebar_section, SidebarSection::Jobs);
        key(&mut app, KeyCode::Char(' '));
        assert!(app.jobs_collapsed);
        key(&mut app, KeyCode::Char('['));
        assert_eq!(app.sidebar_section, SidebarSection::Models);

        // Toolbar keys work from the sidebar; unbuilt utilities say so.
        key(&mut app, KeyCode::Char('R'));
        assert!(app.status.contains("not built yet"), "{}", app.status);
        key(&mut app, KeyCode::Char('h'));
        assert_eq!(app.view, View::System);
        assert!(!app.sidebar_focused);

        // Tab returns to the sidebar; q there still quits.
        key(&mut app, KeyCode::Tab);
        assert!(app.sidebar_focused);
        key(&mut app, KeyCode::Char('q'));
        assert!(app.should_quit);
    }

    #[test]
    fn enter_on_a_chat_model_opens_chat_without_loading_it() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        connect_test_chat(
            &mut app,
            vec![test_chat_model("local-chat", "unloaded", false)],
        );
        app.models = vec![test_chat_model("local-chat", "unloaded", false)];
        focus_stack_profile_by_id(&mut app, "local-chat");
        app.sidebar_focused = true;
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.view, View::Chat, "{}", app.status);
        assert!(!app.sidebar_focused);
        assert!(!app.model_action_pending);
        assert!(app.job_history.records().is_empty());
    }

    #[test]
    fn loading_a_model_names_what_it_evicts_and_needs_a_second_press() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.models = vec![
            test_chat_model("local-chat", "unloaded", false),
            test_chat_model("resident", "loaded", false),
        ];
        app.loaded_model_id = Some("resident".into());
        focus_stack_profile_by_id(&mut app, "local-chat");
        app.sidebar_focused = true;
        app.handle_key(KeyEvent::new(KeyCode::Char('L'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.status.contains("evicts resident"), "{}", app.status);
        assert!(app.status.contains("press L again"));
        assert!(!app.model_action_pending);
        // Any other key cancels.
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.load_armed, None);
        assert_eq!(app.status, "Load cancelled");

        // Media and decision profiles are not llama-swap models.
        focus_stack_profile_by_id(&mut app, "breeze");
        app.sidebar_focused = true;
        app.handle_key(KeyEvent::new(KeyCode::Char('L'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.status.contains("not a llama-swap model"));
        assert!(!app.model_action_pending);
    }

    #[test]
    fn background_router_refresh_tracks_other_clients_without_touching_status() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        // Let the fixture's own startup refresh settle first.
        let deadline = Instant::now() + Duration::from_secs(5);
        while app.loading_models && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        app.loaded_model_id = Some("ours".into());
        app.status = "press L again to confirm".into();
        app.loading_models = true;
        app.sender
            .send(BackgroundEvent::Models(Ok(vec![
                test_chat_model("ours", "unloaded", false),
                test_chat_model("someone-elses", "loaded", false),
            ])))
            .unwrap();
        app.drain_background_events();
        assert_eq!(app.loaded_model_id.as_deref(), Some("someone-elses"));
        assert_eq!(app.status, "press L again to confirm");

        app.loading_models = true;
        for _ in 0..3 {
            app.loading_models = true;
            app.sender
                .send(BackgroundEvent::Models(Err("router down".into())))
                .unwrap();
            app.drain_background_events();
        }
        assert_eq!(app.status, "press L again to confirm");
        assert_eq!(
            app.log
                .iter()
                .filter(|line| line.contains("router down"))
                .count(),
            1
        );
    }

    #[test]
    fn bench_rates_need_an_exact_path_and_a_named_entry_when_paths_are_shared() {
        let entry = |id: &str, path: &str| crate::swap_yaml::SwapYamlEntry {
            id: id.into(),
            disabled: false,
            unlisted: false,
            model_path: Some(path.into()),
            name: None,
            description: None,
            ttl: None,
            proxy: false,
            cmd: String::new(),
            tasks: vec!["chat".into()],
        };
        let result = |key: &str, path: &str, rate: f64| crate::bench_results::BenchResult {
            model_key: key.into(),
            model: path.into(),
            tg_ts: Some(rate),
            ..Default::default()
        };
        let entries = [
            entry("solo", "/m/solo.gguf"),
            entry("shared-a", "/m/shared.gguf"),
            entry("shared-b", "/m/shared.gguf"),
        ];
        let results = [
            result("solo_fit", "/m/solo.gguf", 40.0),
            result("shared-b", "/m/shared.gguf", 77.0),
            result("some-script", "/m/shared.gguf", 99.0),
            result("other", "/m/other.gguf", 10.0),
        ];
        let runs = bench_runs_by_entry(&entries, &results);
        let rate = |id: &str| runs.get(id).and_then(|runs| runs[0].tg_ts);
        assert_eq!(rate("solo"), Some(40.0));
        assert_eq!(rate("shared-b"), Some(77.0));
        assert_eq!(rate("shared-a"), None);
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn download_keeps_tab_for_fields_and_esc_reaches_the_sidebar() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.view = View::Download;
        assert_eq!(app.download.focus(), DownloadFocus::Table);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .unwrap();
        assert!(!app.sidebar_focused);
        assert_ne!(app.download.focus(), DownloadFocus::Table);
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.download.focus(), DownloadFocus::Table);
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        assert!(app.sidebar_focused);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .unwrap();
        assert!(!app.sidebar_focused);
        assert_eq!(app.view, View::Download);
    }

    #[test]
    fn a_pending_load_is_cancelled_by_leaving_the_sidebar() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        app.models = vec![test_chat_model("local-chat", "unloaded", false)];
        focus_stack_profile_by_id(&mut app, "local-chat");
        app.sidebar_focused = true;
        app.handle_key(KeyEvent::new(KeyCode::Char('L'), KeyModifiers::NONE))
            .unwrap();
        assert!(app.load_armed.is_some());
        app.handle_key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.load_armed, None);
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .unwrap();
        assert!(app.sidebar_focused);
        app.handle_key(KeyEvent::new(KeyCode::Char('L'), KeyModifiers::NONE))
            .unwrap();
        // The next L after leaving only re-arms, showing the warning again.
        assert!(app.status.contains("press L again"), "{}", app.status);
        assert!(!app.model_action_pending);
    }

    #[test]
    fn the_first_router_listing_selects_the_resident_model() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        let deadline = Instant::now() + Duration::from_secs(5);
        while app.loading_models && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
            app.drain_background_events();
        }
        app.refresh_stack_catalog();
        app.selected_resident_once = false;
        let listing =
            || BackgroundEvent::Models(Ok(vec![test_chat_model("local-chat", "loaded", false)]));
        focus_stack_profile_by_id(&mut app, "breeze");
        app.loading_models = true;
        app.sender.send(listing()).unwrap();
        app.drain_background_events();
        assert_eq!(
            app.selected_stack_profile()
                .map(|profile| profile.key.profile_id.as_str()),
            Some("local-chat")
        );
        // Later listings leave the user's selection alone.
        focus_stack_profile_by_id(&mut app, "breeze");
        app.loading_models = true;
        app.sender.send(listing()).unwrap();
        app.drain_background_events();
        assert_eq!(
            app.selected_stack_profile()
                .map(|profile| profile.key.profile_id.as_str()),
            Some("breeze")
        );
    }

    #[test]
    fn message_line_clips_a_long_status_before_dropping_key_hints() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.sidebar_focused = true;
        app.status = "x".repeat(300);
        for width in [80_u16, 110] {
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            terminal.draw(|frame| app.draw(frame)).unwrap();
            let row = terminal_row(&terminal, 22);
            assert!(row.contains("Enter open"), "{width}: {row:?}");
            assert!(row.contains('…'), "{width}: {row:?}");
            assert_eq!(row.chars().count(), width as usize);
        }
    }

    #[test]
    fn engines_get_their_own_group_and_media_shows_its_engine_state() {
        let fixture = AppFixture::new();
        write_stack_fixture(&fixture);
        let mut app = fixture.app();
        app.refresh_stack_catalog();
        let mut engine = crate::swap_yaml::parse_entries(
            "models:\n  \"audiocpp_auto\":\n    cmd: audiocpp-server --port ${PORT}\n    proxy: http://127.0.0.1:${PORT}\n",
        );
        app.yaml_entries.insert("local-chat".into(), {
            let mut entry = engine.remove(0);
            entry.id = "local-chat".into();
            entry
        });
        let catalog = app.stack_catalog.as_ref().unwrap();
        let chat = catalog
            .list()
            .iter()
            .find(|profile| profile.key.profile_id.as_str() == "local-chat")
            .unwrap();
        assert_eq!(app.sidebar_group(chat).0, "engines");
        let breeze = catalog
            .list()
            .iter()
            .find(|profile| profile.key.profile_id.as_str() == "breeze")
            .unwrap()
            .clone();
        assert_eq!(app.media_engine(&breeze), Some("audiocpp_auto"));
        app.models = vec![test_chat_model("audiocpp_auto", "loaded", false)];
        assert_eq!(app.profile_router_state(&breeze), Some("loaded"));
        // Engines sort after every lane.
        let order: Vec<&str> = app
            .sidebar_model_indices()
            .into_iter()
            .map(|index| {
                app.stack_catalog.as_ref().unwrap().list()[index]
                    .key
                    .profile_id
                    .as_str()
            })
            .collect();
        assert_eq!(order.last(), Some(&"local-chat"));
    }
}
