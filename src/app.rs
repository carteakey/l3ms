use std::{
    collections::VecDeque,
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

use crate::{
    chat::{
        detect_chat_server, detected_servers, terminate_detected_server, ChatClient,
        ChatCompletion, ChatRequest,
    },
    chat_history::ChatHistory,
    commands::{command, search_all_commands, visible_commands, CommandContext, CommandId},
    config_store::{load_config, DownloadConfig},
    download_editor::DownloadEditor,
    download_preflight::{
        probe_disk_space, run_download_preflight_cancellable, DiskSpace, DownloadPreflight,
    },
    download_ui::{DownloadFocus, DownloadUiState, ModelField},
    downloader_command::downloader_command_prefix,
    gguf::{self, GgufFile},
    job_history::JobHistory,
    llama_swap::{SwapClient, SwapModel, DEFAULT_BASE_URL},
    prompt_store::{list_prompts, load_prompt, prompts_path},
    script_editor::ScriptEditorState,
    script_lint::lint_shell_script,
    script_store::{collect_scripts_in, command_for_script, pretty_name, ScriptMode},
    state_store::{self, ChatSession, ChatSessionList, ChatSessionSummary, SavedChatSession},
    telemetry::{
        find_process_named, query_gpu_telemetry, query_server_io, query_system_memory,
        snapshot_descendants, snapshot_process_group, GpuTelemetry, ServerIoTelemetry,
        SystemMemoryTelemetry, SystemTelemetrySnapshot,
    },
    text_buffer::TextBuffer,
    theme,
};
use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Style, Stylize},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Cell, Clear, Gauge, List, ListItem, ListState, Paragraph, Row, Sparkline,
        Table, TableState, Tabs, Wrap,
    },
    Frame, Terminal,
};
use unicode_width::UnicodeWidthStr;

type Tui = Terminal<CrosstermBackend<Stdout>>;

const TAB_NAMES: [&str; 7] = [
    "Workbench",
    "Model Ops",
    "Chat",
    "Model Browser",
    "Download",
    "Jobs",
    "Maintenance",
];
const BACKGROUND_QUEUE_CAPACITY: usize = 512;
const EVENTS_PER_TICK: usize = 128;
const MAX_PROCESS_LINE_BYTES: usize = 16 * 1024;
const MAX_CHAT_PREVIEW_BYTES: usize = 64 * 1024;
const TELEMETRY_INTERVAL: Duration = Duration::from_secs(1);
const MODEL_PAGE_SIZE: usize = 50;
const DOWNLOAD_PAGE_SIZE: usize = 50;
const DOWNLOAD_ESTIMATE_TIMEOUT: Duration = Duration::from_secs(180);
const DOWNLOAD_DISK_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy, PartialEq)]
struct DownloadProgressSnapshot {
    downloaded_bytes: u64,
    total_bytes: u64,
    speed_bytes_per_second: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Workbench,
    ModelOps,
    Chat,
    Browser,
    Download,
    Jobs,
    Maintenance,
}

impl Tab {
    fn index(self) -> usize {
        match self {
            Self::Workbench => 0,
            Self::ModelOps => 1,
            Self::Chat => 2,
            Self::Browser => 3,
            Self::Download => 4,
            Self::Jobs => 5,
            Self::Maintenance => 6,
        }
    }

    fn from_index(index: usize) -> Self {
        match index % TAB_NAMES.len() {
            0 => Self::Workbench,
            1 => Self::ModelOps,
            2 => Self::Chat,
            3 => Self::Browser,
            4 => Self::Download,
            5 => Self::Jobs,
            _ => Self::Maintenance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpsMode {
    Run,
    Bench,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkbenchLayout {
    #[default]
    Dashboard,
    SplitVertical,
    Classic,
    FocusBench,
    FocusJobs,
    FocusGpu,
}

impl WorkbenchLayout {
    pub fn next(self) -> Self {
        match self {
            Self::Dashboard => Self::SplitVertical,
            Self::SplitVertical => Self::Classic,
            Self::Classic => Self::FocusBench,
            Self::FocusBench => Self::FocusJobs,
            Self::FocusJobs => Self::FocusGpu,
            Self::FocusGpu => Self::Dashboard,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Dashboard => "Dashboard (4-pane)",
            Self::SplitVertical => "Split Vertical",
            Self::Classic => "Classic (2-pane)",
            Self::FocusBench => "Focus: Benchmarks",
            Self::FocusJobs => "Focus: Jobs",
            Self::FocusGpu => "Focus: GPU Monitor",
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
        models: Vec<SwapModel>,
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
    tab: Tab,
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
    model_filter: String,
    model_page: usize,
    model_state: TableState,
    loading_models: bool,
    model_action_pending: bool,
    loaded_model_id: Option<String>,

    ops_mode: OpsMode,
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
    workbench_layout: WorkbenchLayout,
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
            tab: Tab::Workbench,
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
            model_page: 0,
            model_state: TableState::default().with_selected(Some(0)),
            loading_models: false,
            model_action_pending: false,
            loaded_model_id: None,
            ops_mode: OpsMode::Run,
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
            workbench_layout: WorkbenchLayout::default(),
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
        app.refresh_models();
        Ok(app)
    }

    fn run(&mut self, terminal: &mut Tui) -> Result<()> {
        while !self.should_quit {
            self.drain_background_events();
            self.refresh_telemetry_if_due();
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
        self.loading_models = true;
        self.status = "Refreshing llama-swap models…".into();
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
                    .list_models()
                    .map(|models| (client.base_url().to_owned(), client, models))
            });
            match result {
                Ok((endpoint, client, models)) => {
                    let _ = sender.send(BackgroundEvent::ChatConnected {
                        request_id,
                        endpoint,
                        client,
                        models,
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
            Ok((client, models)) => {
                let _ = sender.send(BackgroundEvent::ChatConnected {
                    request_id,
                    endpoint: client.base_url().to_owned(),
                    client,
                    models,
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
        thread::spawn(move || match client.list_models() {
            Ok(models) => {
                let _ = sender.send(BackgroundEvent::ChatConnected {
                    request_id,
                    endpoint: client.base_url().to_owned(),
                    client,
                    models,
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
        let selected = self
            .chat_models
            .iter()
            .position(|model| model.state == "loaded")
            .or_else(|| (!self.chat_models.is_empty()).then_some(0));
        self.chat_model_state.select(selected);
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
                            self.model_page = self
                                .model_page
                                .min(self.model_page_count().saturating_sub(1));
                            if self.loaded_model_id.is_none() {
                                self.loaded_model_id = self
                                    .models
                                    .iter()
                                    .find(|model| model.state == "loaded")
                                    .map(|model| model.id.clone());
                            }
                            let visible_count = self.visible_model_count();
                            clamp_table_selection(&mut self.model_state, visible_count);
                            if matches!(self.tab, Tab::Workbench | Tab::ModelOps) {
                                self.status = format!("llama-swap: {} model(s)", self.models.len());
                            }
                        }
                        Err(error) => {
                            if matches!(self.tab, Tab::Workbench | Tab::ModelOps) {
                                self.status = "llama-swap unavailable".into();
                            }
                            self.push_log(format!("Model refresh failed: {error}"));
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
                    models,
                } => {
                    if request_id != self.chat_connection_request_id {
                        continue;
                    }
                    self.chat_connection_pending = false;
                    self.chat_endpoint_draft = endpoint.clone();
                    self.chat_endpoint_committed = Some(endpoint.clone());
                    self.chat_client = Some(client);
                    self.chat_models = models;
                    self.initialize_chat_model_selection();
                    self.status = format!(
                        "Connected to {endpoint} · {} model(s)",
                        self.chat_models.len()
                    );
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
                        }
                        Err(error) => {
                            self.telemetry = format!("Resources: unavailable ({error})");
                        }
                    }
                }
            }
        }
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
            };
            let _ = sender.send(BackgroundEvent::Telemetry(Ok(snapshot)));
        });
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<()> {
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
        if self.tab == Tab::Download && self.download.focus() != DownloadFocus::Table {
            self.handle_download_input_key(key);
            return Ok(());
        }
        if key.code == KeyCode::Char('?') && key.modifiers == KeyModifiers::NONE {
            self.show_help = true;
            return Ok(());
        }
        if self.input_mode != InputMode::Normal {
            return self.handle_input_key(key);
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
            && matches!(self.tab, Tab::ModelOps | Tab::Maintenance)
    }

    fn is_download_reload_trigger(&self, key: KeyEvent) -> bool {
        self.tab == Tab::Download
            && ((key.code == KeyCode::Char('o') && key.modifiers.contains(KeyModifiers::ALT))
                || (self.download.focus() == DownloadFocus::ConfigPath
                    && key.code == KeyCode::Enter
                    && key.modifiers == KeyModifiers::NONE))
    }

    fn is_download_restore_trigger(&self, key: KeyEvent) -> bool {
        self.tab == Tab::Download
            && ((key.code == KeyCode::Char('r') && key.modifiers.contains(KeyModifiers::ALT))
                || (self.download.focus() == DownloadFocus::Versions
                    && key.code == KeyCode::Enter
                    && key.modifiers == KeyModifiers::NONE))
    }

    fn handle_global_navigation_key(&mut self, key: KeyEvent) -> bool {
        let destination = match key.code {
            KeyCode::F(number @ 1..=7) => Some(number as usize - 1),
            KeyCode::Char('1') if key.modifiers.contains(KeyModifiers::ALT) => Some(0),
            KeyCode::Char('2') if key.modifiers.contains(KeyModifiers::ALT) => Some(1),
            KeyCode::Char('3') if key.modifiers.contains(KeyModifiers::ALT) => Some(2),
            KeyCode::Char('4') if key.modifiers.contains(KeyModifiers::ALT) => Some(3),
            KeyCode::Char('5') if key.modifiers.contains(KeyModifiers::ALT) => Some(4),
            KeyCode::Char('6') if key.modifiers.contains(KeyModifiers::ALT) => Some(5),
            KeyCode::Char('7') if key.modifiers.contains(KeyModifiers::ALT) => Some(6),
            KeyCode::Left if key.modifiers.contains(KeyModifiers::ALT) => {
                Some((self.tab.index() + TAB_NAMES.len() - 1) % TAB_NAMES.len())
            }
            KeyCode::Right if key.modifiers.contains(KeyModifiers::ALT) => {
                Some(self.tab.index() + 1)
            }
            _ => None,
        };
        let Some(destination) = destination else {
            return false;
        };
        self.store_active_script_view();
        self.tab = Tab::from_index(destination);
        self.input_mode = InputMode::Normal;
        self.script_input_target = None;
        self.script_versions_target = None;
        self.show_chat_sessions = false;
        self.show_chat_prompts = false;
        self.show_help = false;
        if self.tab == Tab::Chat {
            self.input_mode = InputMode::ChatMessage;
            if self.chat_client.is_none() && !self.chat_connection_pending {
                self.start_chat_connect();
            }
        } else if self.tab == Tab::Browser {
            self.ensure_browser_scan();
        }
        true
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
        match (self.tab, key.code, control, alt) {
            (Tab::Workbench, KeyCode::Char('r'), true, _) => Some(CommandId::WorkbenchLoadModel),
            (Tab::Workbench, KeyCode::Char('s'), true, _) => Some(CommandId::WorkbenchUnloadModel),
            (Tab::Workbench, KeyCode::Char('f'), true, _) => Some(CommandId::WorkbenchFocusFilter),
            (Tab::Workbench, KeyCode::Char('j'), true, _) => Some(CommandId::WorkbenchFocusTable),
            (Tab::Workbench, KeyCode::Char('l'), true, _) => Some(CommandId::WorkbenchClearLog),
            (Tab::ModelOps, KeyCode::Char('r'), true, _) => Some(CommandId::ModelOpsStart),
            (Tab::ModelOps, KeyCode::Char('s'), true, _) => Some(CommandId::ModelOpsStop),
            (Tab::ModelOps, KeyCode::Char('m'), true, _) => Some(CommandId::ModelOpsToggleMode),
            (Tab::ModelOps, KeyCode::Char('f'), true, _) => Some(CommandId::ModelOpsFocusFilter),
            (Tab::ModelOps, KeyCode::Char('j'), true, _) => Some(CommandId::ModelOpsFocusTable),
            (Tab::ModelOps, KeyCode::Char('u'), true, _) => Some(CommandId::ModelOpsFocusEditor),
            (Tab::ModelOps, KeyCode::Char('p'), _, true) => Some(CommandId::ModelOpsSaveScript),
            (Tab::ModelOps, KeyCode::Char('o'), _, true) => Some(CommandId::ModelOpsReloadScript),
            (Tab::ModelOps, KeyCode::Char('v'), _, true) => Some(CommandId::ModelOpsRestoreScript),
            (Tab::ModelOps, KeyCode::Char('l'), true, _) => Some(CommandId::ModelOpsClearLog),
            (Tab::Chat, KeyCode::Char('g'), true, _) => Some(CommandId::ChatConnect),
            (Tab::Chat, KeyCode::Char('b'), true, _) => Some(CommandId::ChatDetect),
            (Tab::Chat, KeyCode::Char('x'), true, _) => Some(CommandId::ChatClear),
            (Tab::Chat, KeyCode::Char('s'), _, true) => Some(CommandId::ChatSave),
            (Tab::Browser, KeyCode::Char('r'), _, true) => Some(CommandId::BrowserScan),
            (Tab::Browser, KeyCode::Char('g'), _, true) => Some(CommandId::BrowserFocusPath),
            (Tab::Browser, KeyCode::Char('j'), _, true) => Some(CommandId::BrowserFocusTable),
            (Tab::Download, KeyCode::Char('d'), _, true) => Some(CommandId::DownloadSelected),
            (Tab::Download, KeyCode::Char('e'), _, true) => Some(CommandId::DownloadEnabled),
            (Tab::Download, KeyCode::Char('w'), _, true) => Some(CommandId::DownloadSaveConfig),
            (Tab::Download, KeyCode::Char('v'), _, true) => Some(CommandId::DownloadValidateConfig),
            (Tab::Download, KeyCode::Char('t'), _, true) => Some(CommandId::DownloadFocusTable),
            (Tab::Download, KeyCode::Char('i'), _, true) => Some(CommandId::DownloadFocusEditor),
            (Tab::Download, KeyCode::Char('o'), _, true) => Some(CommandId::DownloadLoadConfig),
            (Tab::Download, KeyCode::Char('r'), _, true) => Some(CommandId::DownloadRestoreConfig),
            (Tab::Download, KeyCode::Char('n'), _, true) => Some(CommandId::DownloadAddModel),
            (Tab::Download, KeyCode::Char('a'), _, true) => Some(CommandId::DownloadApplyEdit),
            (Tab::Download, KeyCode::Char('k'), _, true) => Some(CommandId::DownloadDeleteModel),
            (Tab::Download, KeyCode::Char('y'), _, true) => Some(CommandId::DownloadClearLog),
            (Tab::Maintenance, KeyCode::Char('r'), true, _) => Some(CommandId::MaintenanceRun),
            (Tab::Maintenance, KeyCode::Char('s'), true, _) => Some(CommandId::MaintenanceStop),
            (Tab::Maintenance, KeyCode::Char('u'), true, _) => {
                Some(CommandId::MaintenanceFocusEditor)
            }
            (Tab::Maintenance, KeyCode::Char('p'), _, true) => {
                Some(CommandId::MaintenanceSaveScript)
            }
            (Tab::Maintenance, KeyCode::Char('o'), _, true) => {
                Some(CommandId::MaintenanceReloadScript)
            }
            (Tab::Maintenance, KeyCode::Char('v'), _, true) => {
                Some(CommandId::MaintenanceRestoreScript)
            }
            (Tab::Maintenance, KeyCode::Char('l'), true, _) => Some(CommandId::MaintenanceClearLog),
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

    fn palette_commands(&self) -> Vec<&'static crate::commands::CommandSpec> {
        search_all_commands(&self.palette_query)
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
                if self.tab == Tab::Chat && self.cancel_chat_request() {
                    return Ok(());
                }
                self.input_mode = InputMode::Normal;
            }
            KeyCode::Enter => match self.input_mode {
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
            KeyCode::Backspace => match self.input_mode {
                InputMode::ModelFilter => {
                    self.model_filter.pop();
                    self.model_page = 0;
                    self.model_state.select(Some(0));
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
                InputMode::Normal => {}
            },
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                match self.input_mode {
                    InputMode::ModelFilter => {
                        self.model_filter.push(character);
                        self.model_page = 0;
                        self.model_state.select(Some(0));
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
        if target == ScriptEditorTarget::Bench && self.ops_mode != OpsMode::Bench {
            self.status = "Run mode is read-only; switch to Bench to edit scripts".into();
            return;
        }
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
        match self.tab {
            Tab::Workbench => self.handle_workbench_key(key),
            Tab::ModelOps => self.handle_ops_key(key),
            Tab::Chat => self.handle_chat_key(key),
            Tab::Browser => self.handle_browser_key(key),
            Tab::Download => self.handle_download_key(key)?,
            Tab::Jobs => self.handle_jobs_key(key),
            Tab::Maintenance => self.handle_maintenance_key(key),
        }
        Ok(())
    }

    fn command_context(&self) -> CommandContext {
        match self.tab {
            Tab::Workbench => CommandContext::Workbench,
            Tab::ModelOps => CommandContext::ModelOps,
            Tab::Chat => CommandContext::Chat,
            Tab::Browser => CommandContext::Browser,
            Tab::Download => CommandContext::Download,
            Tab::Jobs => CommandContext::Jobs,
            Tab::Maintenance => CommandContext::Maintenance,
        }
    }

    fn execute_command(&mut self, command_id: CommandId) -> Result<()> {
        use CommandId::*;

        if let Some(spec) = command(command_id) {
            let context = self.command_context();
            if !spec.is_available_in(context) {
                self.status = format!(
                    "{} is only available in its {} context",
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
            OpenWorkbench => self.tab = Tab::Workbench,
            OpenModelOps => self.tab = Tab::ModelOps,
            OpenChat => self.tab = Tab::Chat,
            OpenBrowser => {
                self.tab = Tab::Browser;
                self.ensure_browser_scan();
            }
            OpenDownload => self.tab = Tab::Download,
            OpenJobs => self.tab = Tab::Jobs,
            OpenMaintenance => self.tab = Tab::Maintenance,
            PreviousTab => {
                self.tab =
                    Tab::from_index((self.tab.index() + TAB_NAMES.len() - 1) % TAB_NAMES.len());
                self.ensure_browser_scan();
            }
            NextTab => {
                self.tab = Tab::from_index(self.tab.index() + 1);
                self.ensure_browser_scan();
            }
            WorkbenchSelectPrevious => {
                let count = self.visible_model_count();
                select_previous_table(&mut self.model_state, count);
            }
            WorkbenchSelectNext => {
                let count = self.visible_model_count();
                select_next_table(&mut self.model_state, count);
            }
            WorkbenchFocusFilter => self.input_mode = InputMode::ModelFilter,
            WorkbenchFocusTable => self.input_mode = InputMode::Normal,
            WorkbenchRefreshModels => self.refresh_models(),
            WorkbenchLoadModel => self.model_action(true),
            WorkbenchUnloadModel => self.model_action(false),
            WorkbenchClearLog => self.clear_activity_log("Workbench"),
            WorkbenchCycleLayout => {
                self.workbench_layout = self.workbench_layout.next();
                self.status = format!("Workbench layout: {}", self.workbench_layout.label());
            }
            ModelOpsToggleMode => {
                if self.script_input_target == Some(ScriptEditorTarget::Bench) {
                    self.leave_script_editor();
                }
                self.ops_mode = match self.ops_mode {
                    OpsMode::Run => OpsMode::Bench,
                    OpsMode::Bench => OpsMode::Run,
                };
                self.input_mode = InputMode::Normal;
            }
            ModelOpsSelectPrevious => match self.ops_mode {
                OpsMode::Run => {
                    let count = self.visible_model_count();
                    select_previous_table(&mut self.model_state, count);
                }
                OpsMode::Bench => self.move_script_selection(ScriptEditorTarget::Bench, true),
            },
            ModelOpsSelectNext => match self.ops_mode {
                OpsMode::Run => {
                    let count = self.visible_model_count();
                    select_next_table(&mut self.model_state, count);
                }
                OpsMode::Bench => self.move_script_selection(ScriptEditorTarget::Bench, false),
            },
            ModelOpsStart => match self.ops_mode {
                OpsMode::Run => self.model_action(true),
                OpsMode::Bench => self.run_selected_bench(),
            },
            ModelOpsStop => match self.ops_mode {
                OpsMode::Run => self.model_action(false),
                OpsMode::Bench => self.stop_running_process(),
            },
            ModelOpsRefreshModels => self.refresh_models(),
            ModelOpsFocusFilter => {
                if self.ops_mode == OpsMode::Run {
                    self.input_mode = InputMode::ModelFilter;
                } else {
                    self.input_mode = InputMode::BenchFilter;
                }
            }
            ModelOpsFocusTable => {
                if self.script_input_target == Some(ScriptEditorTarget::Bench) {
                    self.leave_script_editor();
                }
                self.input_mode = InputMode::Normal;
            }
            ModelOpsFocusEditor => {
                if self.script_input_target == Some(ScriptEditorTarget::Bench) {
                    self.leave_script_editor();
                } else {
                    self.focus_script_editor(ScriptEditorTarget::Bench);
                }
            }
            ModelOpsSaveScript => self.save_script_editor(ScriptEditorTarget::Bench),
            ModelOpsReloadScript => self.reload_script_editor(ScriptEditorTarget::Bench),
            ModelOpsRestoreScript => self.open_script_versions(ScriptEditorTarget::Bench),
            ModelOpsClearLog => self.clear_activity_log("Model Ops"),
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
        if matches!(
            command_id,
            OpenWorkbench
                | OpenModelOps
                | OpenChat
                | OpenBrowser
                | OpenDownload
                | OpenJobs
                | OpenMaintenance
                | PreviousTab
                | NextTab
        ) {
            self.store_active_script_view();
            self.input_mode = InputMode::Normal;
            self.script_input_target = None;
            self.script_versions_target = None;
            self.show_help = false;
        }
        Ok(())
    }

    fn handle_workbench_key(&mut self, key: KeyEvent) {
        let visible_count = self.visible_model_count();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                select_previous_table(&mut self.model_state, visible_count)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                select_next_table(&mut self.model_state, visible_count)
            }
            KeyCode::Char('/') => self.input_mode = InputMode::ModelFilter,
            KeyCode::PageUp | KeyCode::Char('[') => self.previous_model_page(),
            KeyCode::PageDown | KeyCode::Char(']') => self.next_model_page(),
            KeyCode::Char('r') => self.refresh_models(),
            KeyCode::Enter | KeyCode::Char('l') => self.model_action(true),
            KeyCode::Char('s') => self.model_action(false),
            KeyCode::Char('v') => {
                let _ = self.execute_command(CommandId::WorkbenchCycleLayout);
            }
            KeyCode::Char('1') => {
                self.workbench_layout = WorkbenchLayout::Dashboard;
                self.status = format!("Workbench layout: {}", self.workbench_layout.label());
            }
            KeyCode::Char('2') => {
                self.workbench_layout = WorkbenchLayout::SplitVertical;
                self.status = format!("Workbench layout: {}", self.workbench_layout.label());
            }
            KeyCode::Char('3') => {
                self.workbench_layout = WorkbenchLayout::Classic;
                self.status = format!("Workbench layout: {}", self.workbench_layout.label());
            }
            KeyCode::Char('4') => {
                self.workbench_layout = WorkbenchLayout::FocusBench;
                self.status = format!("Workbench layout: {}", self.workbench_layout.label());
            }
            KeyCode::Char('5') => {
                self.workbench_layout = WorkbenchLayout::FocusJobs;
                self.status = format!("Workbench layout: {}", self.workbench_layout.label());
            }
            KeyCode::Char('6') => {
                self.workbench_layout = WorkbenchLayout::FocusGpu;
                self.status = format!("Workbench layout: {}", self.workbench_layout.label());
            }
            KeyCode::Char('b') => self.run_selected_bench(),
            _ => {}
        }
    }

    fn handle_ops_key(&mut self, key: KeyEvent) {
        if matches!(key.code, KeyCode::Char('m')) {
            self.ops_mode = match self.ops_mode {
                OpsMode::Run => OpsMode::Bench,
                OpsMode::Bench => OpsMode::Run,
            };
            return;
        }
        match self.ops_mode {
            OpsMode::Run => self.handle_workbench_key(key),
            OpsMode::Bench => match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.move_script_selection(ScriptEditorTarget::Bench, true)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.move_script_selection(ScriptEditorTarget::Bench, false)
                }
                KeyCode::Char('/') => self.input_mode = InputMode::BenchFilter,
                KeyCode::Enter | KeyCode::Char('r') => self.run_selected_bench(),
                KeyCode::Char('s') => self.stop_running_process(),
                _ => {}
            },
        }
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

    fn filtered_models(&self) -> Vec<&SwapModel> {
        let filter = self.model_filter.to_ascii_lowercase();
        self.models
            .iter()
            .filter(|model| filter.is_empty() || model.id.to_ascii_lowercase().contains(&filter))
            .collect()
    }

    fn visible_models(&self) -> Vec<&SwapModel> {
        let filtered = self.filtered_models();
        let start = self.model_page.saturating_mul(MODEL_PAGE_SIZE);
        filtered
            .into_iter()
            .skip(start)
            .take(MODEL_PAGE_SIZE)
            .collect()
    }

    fn visible_model_count(&self) -> usize {
        self.visible_models().len()
    }

    fn model_page_count(&self) -> usize {
        self.filtered_models()
            .len()
            .div_ceil(MODEL_PAGE_SIZE)
            .max(1)
    }

    fn previous_model_page(&mut self) {
        self.model_page = self.model_page.saturating_sub(1);
        self.model_state.select(Some(0));
    }

    fn next_model_page(&mut self) {
        self.model_page = (self.model_page + 1).min(self.model_page_count().saturating_sub(1));
        self.model_state.select(Some(0));
    }

    fn selected_model(&self) -> Option<SwapModel> {
        let index = self.model_state.selected()?;
        self.visible_models()
            .get(index)
            .map(|model| (*model).clone())
    }

    fn selected_chat_model(&self) -> Option<SwapModel> {
        self.chat_model_state
            .selected()
            .and_then(|index| self.chat_models.get(index))
            .cloned()
    }

    fn select_previous_chat_model(&mut self) {
        select_previous_table(&mut self.chat_model_state, self.chat_models.len());
        if let Some(model) = self.selected_chat_model() {
            self.status = format!("Chat model: {}", model.id);
        }
    }

    fn select_next_chat_model(&mut self) {
        select_next_table(&mut self.chat_model_state, self.chat_models.len());
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

    fn model_action(&mut self, load: bool) {
        if self.model_action_pending {
            self.status = "A llama-swap action is already in progress".into();
            return;
        }
        let model_id = if load {
            self.selected_model().map(|model| model.id)
        } else {
            self.loaded_model_id
                .clone()
                .or_else(|| self.selected_model().map(|model| model.id))
        };
        let Some(model_id) = model_id else {
            self.status = if load {
                "No model selected".into()
            } else {
                "No loaded model is known".into()
            };
            return;
        };
        self.begin_model_action(model_id, load);
    }

    fn begin_model_action(&mut self, model_id: String, load: bool) {
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

        let mut child = match command.spawn() {
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
        } else if self.loaded_model_id.is_some() {
            self.model_action(false);
        } else {
            self.status = "No active job to stop".into();
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
        let areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(3),
                Constraint::Min(8),
                Constraint::Length(5),
                Constraint::Length(1),
            ])
            .split(frame.area());

        self.draw_header(frame, areas[0]);
        self.draw_tabs(frame, areas[1]);
        match self.tab {
            Tab::Workbench => self.draw_workbench(frame, areas[2]),
            Tab::ModelOps => self.draw_model_ops(frame, areas[2]),
            Tab::Chat => self.draw_chat(frame, areas[2]),
            Tab::Browser => self.draw_browser(frame, areas[2]),
            Tab::Download => self.draw_download(frame, areas[2]),
            Tab::Jobs => self.draw_jobs(frame, areas[2]),
            Tab::Maintenance => self.draw_maintenance(frame, areas[2]),
        }
        self.draw_log(frame, areas[3]);
        self.draw_footer(frame, areas[4]);
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
        }
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let endpoint =
            env::var("LLAMA_SWAP_URL").unwrap_or_else(|_| "http://localhost:8080".into());
        let line = Line::from(vec![
            Span::styled(" L3MS ", theme::badge()),
            Span::raw(format!(" Rust v{}  ", env!("CARGO_PKG_VERSION"))),
            Span::styled(endpoint, theme::dim()),
        ]);
        frame.render_widget(Paragraph::new(line).style(theme::text()), area);
    }

    fn draw_tabs(&self, frame: &mut Frame, area: Rect) {
        let titles = TAB_NAMES
            .iter()
            .map(|name| Line::from(*name))
            .collect::<Vec<_>>();
        let tabs = Tabs::new(titles)
            .block(theme::block().borders(Borders::ALL))
            .style(theme::text())
            .select(self.tab.index())
            .highlight_style(theme::title())
            .divider("│");
        frame.render_widget(tabs, area);
    }

    fn draw_workbench(&mut self, frame: &mut Frame, area: Rect) {
        match self.workbench_layout {
            WorkbenchLayout::Dashboard => {
                let rows = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
                    .split(area);
                let top_cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
                    .split(rows[0]);
                let bot_cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(rows[1]);

                self.draw_models_table(frame, top_cols[0], "Models");
                self.draw_gpu_monitor(frame, top_cols[1]);
                self.draw_workbench_fast_actions(frame, bot_cols[0]);
                self.draw_workbench_ops(frame, bot_cols[1]);
            }
            WorkbenchLayout::SplitVertical => {
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
                    .split(area);
                self.draw_models_table(frame, cols[0], "Models");
                let right_rows = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
                    .split(cols[1]);
                self.draw_gpu_monitor(frame, right_rows[0]);
                self.draw_workbench_fast_actions(frame, right_rows[1]);
            }
            WorkbenchLayout::Classic => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
                    .split(area);
                self.draw_models_table(frame, chunks[0], "Models");
                self.draw_workbench_fast_actions(frame, chunks[1]);
            }
            WorkbenchLayout::FocusBench => {
                let panes = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
                    .split(area);
                self.draw_bench_list_widget(frame, panes[0]);
                self.draw_script_editor(frame, panes[1], ScriptEditorTarget::Bench);
            }
            WorkbenchLayout::FocusJobs => {
                self.draw_jobs(frame, area);
            }
            WorkbenchLayout::FocusGpu => {
                self.draw_gpu_monitor_expanded(frame, area);
            }
        }
    }

    fn draw_gpu_monitor(&self, frame: &mut Frame, area: Rect) {
        let title = format!(
            "Hardware & GPU (nvidia-smi) · v layout ({})",
            self.workbench_layout.label()
        );
        let block = theme::block().title(title).borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height < 2 || inner.width < 10 {
            return;
        }

        let mut lines = Vec::new();

        if let Some(gpu) = &self.gpu_telemetry {
            lines.push(Line::from(vec![
                Span::styled("GPU:  ", theme::title()),
                Span::styled(
                    format!(
                        "{} · {}°C · {:.1}W",
                        gpu.name, gpu.temperature_c, gpu.power_watts
                    ),
                    theme::text(),
                ),
            ]));

            let vram_used_gib = gpu.used_vram_mib as f64 / 1024.0;
            let vram_total_gib = gpu.total_vram_mib as f64 / 1024.0;
            let vram_pct = (gpu.used_vram_mib * 100)
                .checked_div(gpu.total_vram_mib)
                .unwrap_or(0);
            let vram_style = if vram_pct > 90 {
                theme::warning()
            } else {
                theme::text()
            };
            lines.push(Line::from(vec![
                Span::styled("VRAM: ", theme::title()),
                Span::styled(
                    format!("{vram_used_gib:.2}/{vram_total_gib:.2} GiB ({vram_pct}%)"),
                    vram_style,
                ),
                Span::raw("  Load: "),
                Span::styled(
                    format!("{}%", gpu.gpu_util_percent),
                    if gpu.gpu_util_percent > 80 {
                        theme::title()
                    } else {
                        theme::dim()
                    },
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                "GPU: nvidia-smi polling / idle",
                theme::dim(),
            )));
        }

        if let Some(mem) = &self.sys_memory {
            let ram_used = mem.used_ram_mib as f64 / 1024.0;
            let ram_total = mem.total_ram_mib as f64 / 1024.0;
            let ram_pct = (mem.used_ram_mib * 100)
                .checked_div(mem.total_ram_mib)
                .unwrap_or(0);
            let swap_used = mem.used_swap_mib as f64 / 1024.0;
            lines.push(Line::from(vec![
                Span::styled("RAM:  ", theme::title()),
                Span::raw(format!(
                    "{ram_used:.1}/{ram_total:.1} GiB ({ram_pct}%) · zram: {swap_used:.1} GiB"
                )),
            ]));
        }

        if let Some(io) = &self.server_io {
            let io_text = io.read_rate_bytes_per_sec.map_or_else(
                || format!("read: {:.1} MB", io.read_bytes as f64 / (1024.0 * 1024.0)),
                |rate| {
                    if rate < 1024.0 * 1024.0 {
                        format!("read: {:.1} KB/s (PLE normal)", rate / 1024.0)
                    } else {
                        format!(
                            "read: {:.1} MB/s (HOT EXPERT SPILL!)",
                            rate / (1024.0 * 1024.0)
                        )
                    }
                },
            );
            lines.push(Line::from(vec![
                Span::styled("IO:   ", theme::title()),
                Span::styled(io_text, theme::dim()),
            ]));
        }

        let text_height = (lines.len() as u16).min(inner.height);
        if inner.height > text_height + 1 && !self.gpu_util_history.is_empty() {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(text_height), Constraint::Min(1)])
                .split(inner);

            frame.render_widget(Paragraph::new(lines).style(theme::text()), chunks[0]);

            let spark_data: Vec<u64> = self.gpu_util_history.iter().copied().collect();
            let sparkline = Sparkline::default()
                .block(
                    Block::default()
                        .title("GPU Util Trend (60s)")
                        .title_style(theme::dim())
                        .borders(Borders::NONE),
                )
                .data(&spark_data)
                .max(100)
                .style(theme::title());
            frame.render_widget(sparkline, chunks[1]);
        } else {
            frame.render_widget(Paragraph::new(lines).style(theme::text()), inner);
        }
    }

    fn draw_workbench_fast_actions(&self, frame: &mut Frame, area: Rect) {
        let model = self.selected_model();
        let text = if let Some(model) = model {
            Text::from(vec![
                Line::from(Span::styled(model.id, theme::title())),
                Line::from(format!("state: {}", model.state)),
                Line::from(format!("name: {}", value_or_dash(&model.name))),
                Line::from(""),
                Line::from(model.description),
                Line::from(""),
                Line::from("Enter/l load · s unload · r refresh · b run bench"),
                Line::from("v cycle layout · 1-6 layout · / filter · F3 chat"),
            ])
        } else {
            Text::from(
                "No model selected\n\nr refreshes llama-swap\nv cycles workbench layout\n1-6 jumps between views",
            )
        };
        frame.render_widget(
            Paragraph::new(text)
                .style(theme::text())
                .block(
                    theme::block()
                        .title("Fast actions & profile")
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_workbench_ops(&mut self, frame: &mut Frame, area: Rect) {
        let running_label = self.running_process.as_ref().map_or_else(
            || "idle".to_string(),
            |p| format!("PID {} active", p.process_group),
        );
        let title = format!("Benchmarks & Quick Ops ({running_label})");
        let visible = self
            .visible_bench_scripts()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let items = if visible.is_empty() {
            vec![ListItem::new("No bench scripts match the filter")]
        } else {
            visible
                .iter()
                .take(10)
                .map(|path| ListItem::new(relative_display(&self.root, path)))
                .collect::<Vec<_>>()
        };
        let list = List::new(items)
            .style(theme::text())
            .block(
                theme::block()
                    .title(format!("{title} · b run bench · 4 bench · 5 jobs"))
                    .borders(Borders::ALL),
            )
            .highlight_symbol("▶ ")
            .highlight_style(theme::title());
        frame.render_stateful_widget(list, area, &mut self.bench_state);
    }

    fn draw_gpu_monitor_expanded(&self, frame: &mut Frame, area: Rect) {
        let title = format!(
            "Hardware & GPU Performance Analyzer · nvidia-smi · v layout ({})",
            self.workbench_layout.label()
        );
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
            vec![ListItem::new("No bench scripts match the current filter")]
        } else {
            visible
                .iter()
                .map(|path| ListItem::new(relative_display(&self.root, path)))
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

    fn draw_model_ops(&mut self, frame: &mut Frame, area: Rect) {
        let mode = match self.ops_mode {
            OpsMode::Run => "RUN · llama-swap",
            OpsMode::Bench => "BENCH · scripts",
        };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(4)])
            .split(area);
        frame.render_widget(
            Paragraph::new(format!(
                " {mode}   m toggle mode · r/Enter start · s stop\n {}",
                self.telemetry
            ))
            .style(theme::warning()),
            chunks[0],
        );
        match self.ops_mode {
            OpsMode::Run => self.draw_models_table(frame, chunks[1], "Servable models"),
            OpsMode::Bench => {
                let panes = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
                    .split(chunks[1]);
                self.draw_bench_list_widget(frame, panes[0]);
                self.draw_script_editor(frame, panes[1], ScriptEditorTarget::Bench);
            }
        }
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

    fn draw_models_table(&mut self, frame: &mut Frame, area: Rect, title: &str) {
        let filter = if self.model_filter.is_empty() {
            String::new()
        } else {
            format!(" · filter: {}", self.model_filter)
        };
        let page = format!(
            " · page {}/{} · [/] page",
            self.model_page + 1,
            self.model_page_count()
        );
        let rows = self
            .visible_models()
            .into_iter()
            .map(|model| {
                Row::new(vec![
                    Cell::from(model.id.clone()),
                    Cell::from(model.state.clone()),
                ])
            })
            .collect::<Vec<_>>();
        let table = Table::new(
            rows,
            [Constraint::Percentage(76), Constraint::Percentage(24)],
        )
        .style(theme::text())
        .header(
            Row::new(["Model", "State"])
                .style(theme::title())
                .bottom_margin(1),
        )
        .block(
            theme::block()
                .title(format!("{title}{filter}{page}"))
                .borders(Borders::ALL),
        )
        .row_highlight_style(theme::selected())
        .highlight_symbol("▶ ");
        frame.render_stateful_widget(table, area, &mut self.model_state);
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
            format!("connected · {} model(s)", self.chat_models.len())
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
                self.chat_system_prompt,
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
                theme::GREEN
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
                Style::default().fg(theme::GREEN).bold(),
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
        frame.render_stateful_widget(list, panes[0], &mut self.maintenance_state);
        self.draw_script_editor(frame, panes[1], ScriptEditorTarget::Maintenance);
    }

    fn draw_log(&self, frame: &mut Frame, area: Rect) {
        let lines = self
            .log
            .iter()
            .rev()
            .take(area.height.saturating_sub(2) as usize)
            .rev()
            .map(|line| Line::from(line.as_str()))
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(lines)
                .style(theme::text())
                .block(theme::block().title("Activity").borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_footer(&self, frame: &mut Frame, area: Rect) {
        let input = if let Some(target) = self.script_input_target {
            match target {
                ScriptEditorTarget::Bench => "  BENCH SCRIPT INPUT",
                ScriptEditorTarget::Maintenance => "  MAINTENANCE SCRIPT INPUT",
            }
        } else if self.tab == Tab::Download && self.download.focus() != DownloadFocus::Table {
            "  DOWNLOAD INPUT"
        } else {
            match self.input_mode {
                InputMode::Normal => "",
                InputMode::ModelFilter => "  FILTER",
                InputMode::BenchFilter => "  BENCH FILTER",
                InputMode::BrowserPath => "  PATH INPUT",
                InputMode::BrowserFilter => "  GGUF FILTER",
                InputMode::ChatMessage => "  CHAT INPUT",
                InputMode::ChatEndpoint => "  CHAT ENDPOINT",
                InputMode::ChatSystemPrompt => "  SYSTEM PROMPT INPUT",
            }
        };
        let controls = if self.script_input_target.is_some() {
            "Esc list │ F1–F7 tabs │ Ctrl+P palette │ Ctrl+C quit"
        } else if self.tab == Tab::Download && self.download.focus() != DownloadFocus::Table {
            "Tab/Shift+Tab fields │ Esc table │ Ctrl+P palette │ Ctrl+C quit"
        } else {
            "F1–F7 tabs │ Alt+←/→ cycle │ Ctrl+P palette │ ? help │ q quit"
        };
        let footer = format!(" {}{}  │ {controls} ", self.status, input);
        frame.render_widget(
            Paragraph::new(footer)
                .style(theme::badge())
                .alignment(Alignment::Left),
            area,
        );
    }

    fn draw_help(&self, frame: &mut Frame) {
        let area = centered_rect(82, 82, frame.area());
        frame.render_widget(Clear, area);
        let context = self.command_context();
        let mut help = vec![
            Line::from(Span::styled("L3MS Rust key bindings", theme::title())),
            Line::from(""),
            Line::from("Global  F1–F7/Alt+1–7 tabs · Alt+←/→ cycle · Ctrl+P palette · q quit"),
            Line::from(""),
        ];
        help.push(Line::from(Span::styled(
            format!("{} commands", context),
            theme::title(),
        )));
        for spec in visible_commands(context)
            .into_iter()
            .filter(|spec| !spec.contexts.contains(&CommandContext::Global))
        {
            help.push(Line::from(vec![
                Span::styled(format!("  {:<20}", spec.shortcut), theme::border_focused()),
                Span::raw(spec.label),
            ]));
        }
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
        let area = centered_rect(84, 84, frame.area());
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
    let status = Command::new("/bin/kill")
        .args([
            format!("-{signal}"),
            "--".into(),
            format!("-{process_group}"),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
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

        fn app(&self) -> App {
            App::new_in(self.root.clone(), self.data.path().to_path_buf()).unwrap()
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
        app.tab = Tab::Download;

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
        app.tab = Tab::Download;
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
        app.tab = Tab::Download;
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
        app.tab = Tab::Download;
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
        app.tab = Tab::Download;

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
        app.tab = Tab::Download;
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
                models: Vec::new(),
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
                models: vec![SwapModel {
                    id: "chat-model".into(),
                    state: "loaded".into(),
                    name: String::new(),
                    description: String::new(),
                    created: None,
                    disabled: false,
                    size_bytes: None,
                }],
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
        app.tab = Tab::Chat;
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
    fn chat_shortcuts_keep_f3_composer_focus_and_modified_key_precedence() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.chat_client = Some(ChatClient::new("http://chat.example", None).unwrap());

        app.handle_key(KeyEvent::new(KeyCode::F(3), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.tab, Tab::Chat);
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
        app.tab = Tab::Download;
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
        app.tab = Tab::Download;
        app.push_download_log("download-only-marker");
        let backend = TestBackend::new(150, 48);
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
        app.tab = Tab::Download;
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
        app.tab = Tab::Download;
        let long_base = format!("/models/{}/TAIL100", "nested-directory/".repeat(10));
        app.download
            .base_models_dir_buffer_mut()
            .set_content(long_base);
        app.download.set_focus(DownloadFocus::BaseModelsDir);
        app.push_download_log("compact-settings-marker");

        let backend = TestBackend::new(100, 30);
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
        assert!(cursor.x < 100 && cursor.y < 30, "cursor out of bounds");
        let cursor_row = terminal_row(&terminal, cursor.y);
        assert!(cursor_row.contains("Base:"));
        assert!(cursor_row.contains("TAIL100"));
    }

    #[test]
    fn compact_download_view_uses_focused_editor_when_too_narrow_for_both_panes() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.tab = Tab::Download;
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
        app.tab = Tab::ModelOps;
        app.ops_mode = OpsMode::Bench;
        app.execute_command(CommandId::ModelOpsFocusEditor).unwrap();

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
        assert_eq!(app.tab, Tab::Maintenance);
        assert!(app.bench_editor.is_dirty());
        app.handle_key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE))
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
        app.tab = Tab::ModelOps;
        app.ops_mode = OpsMode::Bench;
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
        app.tab = Tab::ModelOps;
        app.ops_mode = OpsMode::Bench;

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
        assert!(terminal_text(&terminal).contains("No bench scripts match the current filter"));
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
        app.tab = Tab::ModelOps;
        app.ops_mode = OpsMode::Bench;
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
        app.tab = Tab::ModelOps;
        app.ops_mode = OpsMode::Bench;

        assert!(visible_commands(CommandContext::ModelOps)
            .iter()
            .any(|spec| spec.id == CommandId::ModelOpsFocusFilter));
        app.open_palette();
        app.palette_query = "script filter".into();
        assert!(app
            .palette_commands()
            .iter()
            .any(|spec| spec.id == CommandId::ModelOpsFocusFilter));

        app.show_palette = false;
        app.execute_command(CommandId::ModelOpsFocusFilter).unwrap();
        assert_eq!(app.input_mode, InputMode::BenchFilter);
        app.input_mode = InputMode::Normal;
        app.show_help = true;
        let backend = TestBackend::new(120, 36);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let rendered = terminal_text(&terminal);
        assert!(rendered.contains("Ctrl+F / /"));
        assert!(rendered.contains("Filter items"));
    }

    #[test]
    fn dirty_quit_requires_an_explicit_choice() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.tab = Tab::ModelOps;
        app.ops_mode = OpsMode::Bench;
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
        assert_eq!(app.tab, Tab::Workbench);
        app.execute_command(CommandId::MaintenanceFocusEditor)
            .unwrap();
        assert_eq!(app.script_input_target, None);
        assert!(app.status.contains("maintenance context"));
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
        let mut child = command.spawn().expect("spawn isolated shell");
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
    fn workbench_layout_cycles_and_shortcuts_work() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.tab = Tab::Workbench;
        assert_eq!(app.workbench_layout, WorkbenchLayout::Dashboard);

        // Press 'v' to cycle
        app.handle_key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.workbench_layout, WorkbenchLayout::SplitVertical);

        app.handle_key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.workbench_layout, WorkbenchLayout::Classic);

        // Direct jumps via 1-6
        app.handle_key(KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.workbench_layout, WorkbenchLayout::FocusBench);

        app.handle_key(KeyEvent::new(KeyCode::Char('5'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.workbench_layout, WorkbenchLayout::FocusJobs);

        app.handle_key(KeyEvent::new(KeyCode::Char('6'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.workbench_layout, WorkbenchLayout::FocusGpu);

        app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE))
            .unwrap();
        assert_eq!(app.workbench_layout, WorkbenchLayout::Dashboard);
    }

    #[test]
    fn workbench_renders_all_layouts_without_panics() {
        let fixture = AppFixture::new();
        let mut app = fixture.app();
        app.tab = Tab::Workbench;

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

        for layout in [
            WorkbenchLayout::Dashboard,
            WorkbenchLayout::SplitVertical,
            WorkbenchLayout::Classic,
            WorkbenchLayout::FocusBench,
            WorkbenchLayout::FocusJobs,
            WorkbenchLayout::FocusGpu,
        ] {
            app.workbench_layout = layout;
            terminal.draw(|frame| app.draw(frame)).unwrap();
        }
    }
}
