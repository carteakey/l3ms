//! Typed command metadata shared by the TUI help, command bar, and palette.
//!
//! This module deliberately describes shortcuts as display text instead of
//! depending on a terminal event library. The application remains responsible
//! for mapping concrete key events to [`CommandId`] values and for applying
//! runtime predicates such as "a model is selected" or "a process is running".

use std::fmt;

/// A scope in which a command can be offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandContext {
    Global,
    Workbench,
    ModelOps,
    Chat,
    Browser,
    Download,
    Jobs,
    Maintenance,
    ParamBuilder,
}

impl CommandContext {
    /// The eight top-level TUI tabs, in navigation order.
    pub const TABS: [Self; 8] = [
        Self::Workbench,
        Self::ModelOps,
        Self::ParamBuilder,
        Self::Chat,
        Self::Browser,
        Self::Download,
        Self::Jobs,
        Self::Maintenance,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Workbench => "workbench",
            Self::ModelOps => "model-ops",
            Self::Chat => "chat",
            Self::Browser => "browser",
            Self::Download => "download",
            Self::Jobs => "jobs",
            Self::Maintenance => "maintenance",
            Self::ParamBuilder => "param-builder",
        }
    }
}

impl fmt::Display for CommandContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Stable, typed identifiers used by key handling and palette dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandId {
    Quit,
    ShowHelp,
    ShowPalette,
    OpenWorkbench,
    OpenModelOps,
    OpenParamBuilder,
    OpenChat,
    OpenBrowser,
    OpenDownload,
    OpenJobs,
    OpenMaintenance,
    ParamBuilderToggleFocus,
    ParamBuilderNextParam,
    ParamBuilderPrevParam,
    ParamBuilderCycleValue,
    ParamBuilderSaveToYaml,
    ParamBuilderImportEntry,
    ParamBuilderOpenPresets,
    ParamBuilderClearAll,
    PreviousTab,
    NextTab,

    WorkbenchSelectPrevious,
    WorkbenchSelectNext,
    WorkbenchFocusFilter,
    WorkbenchFocusTable,
    WorkbenchRefreshModels,
    WorkbenchLoadModel,
    WorkbenchUnloadModel,
    WorkbenchClearLog,
    WorkbenchCycleLayout,

    ModelOpsToggleMode,
    ModelOpsSelectPrevious,
    ModelOpsSelectNext,
    ModelOpsStart,
    ModelOpsStop,
    ModelOpsRefreshModels,
    ModelOpsFocusFilter,
    ModelOpsFocusTable,
    ModelOpsFocusEditor,
    ModelOpsSaveScript,
    ModelOpsReloadScript,
    ModelOpsRestoreScript,
    ModelOpsClearLog,

    ChatCompose,
    ChatSend,
    ChatRefreshModels,
    ChatConnect,
    ChatDetect,
    ChatKillDetected,
    ChatPrompts,
    ChatClear,
    ChatSave,
    ChatSessions,
    ChatEditSystemPrompt,
    ChatToggleThinking,
    ChatDecreaseTemperature,
    ChatIncreaseTemperature,
    ChatDecreaseMaxTokens,
    ChatIncreaseMaxTokens,

    BrowserSelectPrevious,
    BrowserSelectNext,
    BrowserScan,
    BrowserFocusPath,
    BrowserFocusTable,
    BrowserFocusFilter,
    BrowserChangeSort,
    BrowserToggleRecursive,

    DownloadSelectPrevious,
    DownloadSelectNext,
    DownloadFocusTable,
    DownloadFocusEditor,
    DownloadToggleEnabled,
    DownloadLoadConfig,
    DownloadSaveConfig,
    DownloadValidateConfig,
    DownloadRestoreConfig,
    DownloadAddModel,
    DownloadApplyEdit,
    DownloadDeleteModel,
    DownloadSelected,
    DownloadEnabled,
    DownloadClearLog,

    JobsSelectPrevious,
    JobsSelectNext,
    JobsStop,
    JobsRetry,
    JobsClear,

    MaintenanceSelectPrevious,
    MaintenanceSelectNext,
    MaintenanceRun,
    MaintenanceStop,
    MaintenanceFocusEditor,
    MaintenanceSaveScript,
    MaintenanceReloadScript,
    MaintenanceRestoreScript,
    MaintenanceClearLog,
}

impl CommandId {
    /// Every command identifier. Kept in declaration order for completeness
    /// checks and integrations which want to validate their dispatch table.
    pub const ALL: [Self; 96] = [
        Self::Quit,
        Self::ShowHelp,
        Self::ShowPalette,
        Self::OpenWorkbench,
        Self::OpenModelOps,
        Self::OpenParamBuilder,
        Self::OpenChat,
        Self::OpenBrowser,
        Self::OpenDownload,
        Self::OpenJobs,
        Self::OpenMaintenance,
        Self::ParamBuilderToggleFocus,
        Self::ParamBuilderNextParam,
        Self::ParamBuilderPrevParam,
        Self::ParamBuilderCycleValue,
        Self::ParamBuilderSaveToYaml,
        Self::ParamBuilderImportEntry,
        Self::ParamBuilderOpenPresets,
        Self::ParamBuilderClearAll,
        Self::PreviousTab,
        Self::NextTab,
        Self::WorkbenchSelectPrevious,
        Self::WorkbenchSelectNext,
        Self::WorkbenchFocusFilter,
        Self::WorkbenchFocusTable,
        Self::WorkbenchRefreshModels,
        Self::WorkbenchLoadModel,
        Self::WorkbenchUnloadModel,
        Self::WorkbenchClearLog,
        Self::WorkbenchCycleLayout,
        Self::ModelOpsToggleMode,
        Self::ModelOpsSelectPrevious,
        Self::ModelOpsSelectNext,
        Self::ModelOpsStart,
        Self::ModelOpsStop,
        Self::ModelOpsRefreshModels,
        Self::ModelOpsFocusFilter,
        Self::ModelOpsFocusTable,
        Self::ModelOpsFocusEditor,
        Self::ModelOpsSaveScript,
        Self::ModelOpsReloadScript,
        Self::ModelOpsRestoreScript,
        Self::ModelOpsClearLog,
        Self::ChatCompose,
        Self::ChatSend,
        Self::ChatRefreshModels,
        Self::ChatConnect,
        Self::ChatDetect,
        Self::ChatKillDetected,
        Self::ChatPrompts,
        Self::ChatClear,
        Self::ChatSave,
        Self::ChatSessions,
        Self::ChatEditSystemPrompt,
        Self::ChatToggleThinking,
        Self::ChatDecreaseTemperature,
        Self::ChatIncreaseTemperature,
        Self::ChatDecreaseMaxTokens,
        Self::ChatIncreaseMaxTokens,
        Self::BrowserSelectPrevious,
        Self::BrowserSelectNext,
        Self::BrowserScan,
        Self::BrowserFocusPath,
        Self::BrowserFocusTable,
        Self::BrowserFocusFilter,
        Self::BrowserChangeSort,
        Self::BrowserToggleRecursive,
        Self::DownloadSelectPrevious,
        Self::DownloadSelectNext,
        Self::DownloadFocusTable,
        Self::DownloadFocusEditor,
        Self::DownloadToggleEnabled,
        Self::DownloadLoadConfig,
        Self::DownloadSaveConfig,
        Self::DownloadValidateConfig,
        Self::DownloadRestoreConfig,
        Self::DownloadAddModel,
        Self::DownloadApplyEdit,
        Self::DownloadDeleteModel,
        Self::DownloadSelected,
        Self::DownloadEnabled,
        Self::DownloadClearLog,
        Self::JobsSelectPrevious,
        Self::JobsSelectNext,
        Self::JobsStop,
        Self::JobsRetry,
        Self::JobsClear,
        Self::MaintenanceSelectPrevious,
        Self::MaintenanceSelectNext,
        Self::MaintenanceRun,
        Self::MaintenanceStop,
        Self::MaintenanceFocusEditor,
        Self::MaintenanceSaveScript,
        Self::MaintenanceReloadScript,
        Self::MaintenanceRestoreScript,
        Self::MaintenanceClearLog,
    ];

    /// A durable string suitable for logs, tests, and non-Rust integrations.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quit => "global.quit",
            Self::ShowHelp => "global.show-help",
            Self::ShowPalette => "global.show-palette",
            Self::OpenWorkbench => "global.open-workbench",
            Self::OpenModelOps => "global.open-model-ops",
            Self::OpenParamBuilder => "global.open-param-builder",
            Self::OpenChat => "global.open-chat",
            Self::OpenBrowser => "global.open-browser",
            Self::OpenDownload => "global.open-download",
            Self::OpenJobs => "global.open-jobs",
            Self::OpenMaintenance => "global.open-maintenance",
            Self::ParamBuilderToggleFocus => "param-builder.toggle-focus",
            Self::ParamBuilderNextParam => "param-builder.next-param",
            Self::ParamBuilderPrevParam => "param-builder.prev-param",
            Self::ParamBuilderCycleValue => "param-builder.cycle-value",
            Self::ParamBuilderSaveToYaml => "param-builder.save-to-yaml",
            Self::ParamBuilderImportEntry => "param-builder.import-entry",
            Self::ParamBuilderOpenPresets => "param-builder.open-presets",
            Self::ParamBuilderClearAll => "param-builder.clear-all",
            Self::PreviousTab => "global.previous-tab",
            Self::NextTab => "global.next-tab",
            Self::WorkbenchSelectPrevious => "workbench.select-previous",
            Self::WorkbenchSelectNext => "workbench.select-next",
            Self::WorkbenchFocusFilter => "workbench.focus-filter",
            Self::WorkbenchFocusTable => "workbench.focus-table",
            Self::WorkbenchRefreshModels => "workbench.refresh-models",
            Self::WorkbenchLoadModel => "workbench.load-model",
            Self::WorkbenchUnloadModel => "workbench.unload-model",
            Self::WorkbenchClearLog => "workbench.clear-log",
            Self::WorkbenchCycleLayout => "workbench.cycle-layout",
            Self::ModelOpsToggleMode => "model-ops.toggle-mode",
            Self::ModelOpsSelectPrevious => "model-ops.select-previous",
            Self::ModelOpsSelectNext => "model-ops.select-next",
            Self::ModelOpsStart => "model-ops.start",
            Self::ModelOpsStop => "model-ops.stop",
            Self::ModelOpsRefreshModels => "model-ops.refresh-models",
            Self::ModelOpsFocusFilter => "model-ops.focus-filter",
            Self::ModelOpsFocusTable => "model-ops.focus-table",
            Self::ModelOpsFocusEditor => "model-ops.focus-editor",
            Self::ModelOpsSaveScript => "model-ops.save-script",
            Self::ModelOpsReloadScript => "model-ops.reload-script",
            Self::ModelOpsRestoreScript => "model-ops.restore-script",
            Self::ModelOpsClearLog => "model-ops.clear-log",
            Self::ChatCompose => "chat.compose",
            Self::ChatSend => "chat.send",
            Self::ChatRefreshModels => "chat.refresh-models",
            Self::ChatConnect => "chat.connect",
            Self::ChatDetect => "chat.detect",
            Self::ChatKillDetected => "chat.kill-detected",
            Self::ChatPrompts => "chat.prompts",
            Self::ChatClear => "chat.clear",
            Self::ChatSave => "chat.save",
            Self::ChatSessions => "chat.sessions",
            Self::ChatEditSystemPrompt => "chat.edit-system-prompt",
            Self::ChatToggleThinking => "chat.toggle-thinking",
            Self::ChatDecreaseTemperature => "chat.decrease-temperature",
            Self::ChatIncreaseTemperature => "chat.increase-temperature",
            Self::ChatDecreaseMaxTokens => "chat.decrease-max-tokens",
            Self::ChatIncreaseMaxTokens => "chat.increase-max-tokens",
            Self::BrowserSelectPrevious => "browser.select-previous",
            Self::BrowserSelectNext => "browser.select-next",
            Self::BrowserScan => "browser.scan",
            Self::BrowserFocusPath => "browser.focus-path",
            Self::BrowserFocusTable => "browser.focus-table",
            Self::BrowserFocusFilter => "browser.focus-filter",
            Self::BrowserChangeSort => "browser.change-sort",
            Self::BrowserToggleRecursive => "browser.toggle-recursive",
            Self::DownloadSelectPrevious => "download.select-previous",
            Self::DownloadSelectNext => "download.select-next",
            Self::DownloadFocusTable => "download.focus-table",
            Self::DownloadFocusEditor => "download.focus-editor",
            Self::DownloadToggleEnabled => "download.toggle-enabled",
            Self::DownloadLoadConfig => "download.load-config",
            Self::DownloadSaveConfig => "download.save-config",
            Self::DownloadValidateConfig => "download.validate-config",
            Self::DownloadRestoreConfig => "download.restore-config",
            Self::DownloadAddModel => "download.add-model",
            Self::DownloadApplyEdit => "download.apply-edit",
            Self::DownloadDeleteModel => "download.delete-model",
            Self::DownloadSelected => "download.selected",
            Self::DownloadEnabled => "download.enabled",
            Self::DownloadClearLog => "download.clear-log",
            Self::JobsSelectPrevious => "jobs.select-previous",
            Self::JobsSelectNext => "jobs.select-next",
            Self::JobsStop => "jobs.stop",
            Self::JobsRetry => "jobs.retry",
            Self::JobsClear => "jobs.clear",
            Self::MaintenanceSelectPrevious => "maintenance.select-previous",
            Self::MaintenanceSelectNext => "maintenance.select-next",
            Self::MaintenanceRun => "maintenance.run",
            Self::MaintenanceStop => "maintenance.stop",
            Self::MaintenanceFocusEditor => "maintenance.focus-editor",
            Self::MaintenanceSaveScript => "maintenance.save-script",
            Self::MaintenanceReloadScript => "maintenance.reload-script",
            Self::MaintenanceRestoreScript => "maintenance.restore-script",
            Self::MaintenanceClearLog => "maintenance.clear-log",
        }
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Immutable display and availability metadata for a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandSpec {
    pub id: CommandId,
    /// Compact label suitable for help and command bars.
    pub label: &'static str,
    /// Fully qualified label suitable for an all-action palette.
    pub palette_label: &'static str,
    /// Canonical user-facing shortcut text. `—` means no direct key binding.
    pub shortcut: &'static str,
    /// Contexts which own this command. Global commands are visible in every
    /// tab, but a global-only query contains only global commands.
    pub contexts: &'static [CommandContext],
    /// Whether the Rust application currently has behavior for the action.
    pub implemented: bool,
    /// Whether the action should currently be offered by normal UI surfaces.
    /// Runtime state can further disable an otherwise enabled action.
    pub enabled: bool,
    /// Additional palette terms which are useful but awkward in a label.
    pub keywords: &'static [&'static str],
}

impl CommandSpec {
    pub fn is_available_in(&self, context: CommandContext) -> bool {
        self.contexts.iter().any(|candidate| {
            *candidate == context
                || (*candidate == CommandContext::Global && context != CommandContext::Global)
        })
    }

    pub const fn is_visible(&self) -> bool {
        self.implemented && self.enabled
    }
}

macro_rules! command {
    (
        $id:ident, $context:ident, $shortcut:literal,
        $label:literal, $palette_label:literal,
        $implemented:literal, [$($keyword:literal),* $(,)?]
    ) => {
        CommandSpec {
            id: CommandId::$id,
            label: $label,
            palette_label: $palette_label,
            shortcut: $shortcut,
            contexts: &[CommandContext::$context],
            implemented: $implemented,
            enabled: $implemented,
            keywords: &[$($keyword),*],
        }
    };
}

/// The authoritative command catalog. Declaration order is intentional: it is
/// also the stable tie-break order used by palette search.
pub static COMMANDS: &[CommandSpec] = &[
    command!(
        Quit,
        Global,
        "q / Ctrl+C",
        "Quit",
        "Quit L3MS",
        true,
        ["exit", "close"]
    ),
    command!(
        ShowHelp,
        Global,
        "?",
        "Help",
        "Show keyboard shortcuts",
        true,
        ["bindings", "keys"]
    ),
    command!(
        ShowPalette,
        Global,
        "Ctrl+P",
        "Command palette",
        "Open command palette",
        true,
        ["commands", "search"]
    ),
    command!(
        OpenWorkbench,
        Global,
        "F1 / Alt+1",
        "Workbench",
        "Open Workbench tab",
        true,
        ["navigate", "switch"]
    ),
    command!(
        OpenModelOps,
        Global,
        "F2 / Alt+2",
        "Model Ops",
        "Open Model Ops tab",
        true,
        ["run", "bench", "navigate", "switch"]
    ),
    command!(
        OpenParamBuilder,
        Global,
        "F3 / Alt+3",
        "Param Builder",
        "Open Param Builder tab",
        true,
        ["flags", "llama-server", "params", "navigate", "switch"]
    ),
    command!(
        ParamBuilderToggleFocus,
        ParamBuilder,
        "Tab / Ctrl+L",
        "Toggle focus",
        "Toggle focus between param list and cmd preview",
        true,
        ["focus", "pane"]
    ),
    command!(
        ParamBuilderNextParam,
        ParamBuilder,
        "j / Down",
        "Next param",
        "Select the next parameter",
        true,
        ["navigate"]
    ),
    command!(
        ParamBuilderPrevParam,
        ParamBuilder,
        "k / Up",
        "Previous param",
        "Select the previous parameter",
        true,
        ["navigate"]
    ),
    command!(
        ParamBuilderCycleValue,
        ParamBuilder,
        "Enter / Space",
        "Change value",
        "Cycle, toggle, or edit the selected parameter",
        true,
        ["edit", "toggle"]
    ),
    command!(
        ParamBuilderSaveToYaml,
        ParamBuilder,
        "Ctrl+S",
        "Save to llama-swap.yaml",
        "Write the built flags into a model entry's cmd block (snapshot + SIGHUP)",
        true,
        ["write", "apply", "save"]
    ),
    command!(
        ParamBuilderImportEntry,
        ParamBuilder,
        "Ctrl+I",
        "Import from entry",
        "Load an existing llama-swap.yaml entry's flags into the builder",
        true,
        ["load", "read"]
    ),
    command!(
        ParamBuilderOpenPresets,
        ParamBuilder,
        "Alt+P",
        "Presets",
        "Apply a built-in parameter preset",
        true,
        ["preset", "template"]
    ),
    command!(
        ParamBuilderClearAll,
        ParamBuilder,
        "Ctrl+X",
        "Clear all params",
        "Reset every parameter and custom flag to unset",
        true,
        ["reset", "wipe"]
    ),
    command!(
        OpenChat,
        Global,
        "F4 / Alt+4",
        "Chat",
        "Open Chat tab",
        true,
        ["navigate", "switch"]
    ),
    command!(
        OpenBrowser,
        Global,
        "F5 / Alt+5",
        "Model Browser",
        "Open Model Browser tab",
        true,
        ["gguf", "inventory", "navigate", "switch"]
    ),
    command!(
        OpenDownload,
        Global,
        "F6 / Alt+6",
        "Download",
        "Open Download tab",
        true,
        ["hugging face", "navigate", "switch"]
    ),
    command!(
        OpenJobs,
        Global,
        "F7 / Alt+7",
        "Jobs",
        "Open Jobs tab",
        true,
        ["history", "processes", "navigate", "switch"]
    ),
    command!(
        OpenMaintenance,
        Global,
        "F8 / Alt+8",
        "Maintenance",
        "Open Maintenance tab",
        true,
        ["scripts", "navigate", "switch"]
    ),
    command!(
        PreviousTab,
        Global,
        "Alt+←",
        "Previous tab",
        "Navigate to previous tab",
        true,
        ["left", "back"]
    ),
    command!(
        NextTab,
        Global,
        "Alt+→",
        "Next tab",
        "Navigate to next tab",
        true,
        ["right", "forward"]
    ),
    command!(
        WorkbenchSelectPrevious,
        Workbench,
        "↑ / k",
        "Previous model",
        "Workbench: select previous model",
        true,
        ["up", "navigate"]
    ),
    command!(
        WorkbenchSelectNext,
        Workbench,
        "↓ / j",
        "Next model",
        "Workbench: select next model",
        true,
        ["down", "navigate"]
    ),
    command!(
        WorkbenchFocusFilter,
        Workbench,
        "Ctrl+F / /",
        "Filter models",
        "Workbench: focus model filter",
        true,
        ["find", "search"]
    ),
    command!(
        WorkbenchFocusTable,
        Workbench,
        "Ctrl+J",
        "Model table",
        "Workbench: focus model table",
        true,
        ["list", "models"]
    ),
    command!(
        WorkbenchRefreshModels,
        Workbench,
        "r",
        "Refresh models",
        "Workbench: refresh llama-swap models",
        true,
        ["reload", "sync"]
    ),
    command!(
        WorkbenchLoadModel,
        Workbench,
        "Ctrl+R / Enter / l",
        "Load model",
        "Workbench: load selected llama-swap model",
        true,
        ["start", "serve"]
    ),
    command!(
        WorkbenchUnloadModel,
        Workbench,
        "Ctrl+S / s",
        "Unload model",
        "Workbench: unload active llama-swap model",
        true,
        ["stop", "release"]
    ),
    command!(
        WorkbenchClearLog,
        Workbench,
        "Ctrl+L",
        "Clear log",
        "Workbench: clear activity log",
        true,
        ["output"]
    ),
    command!(
        WorkbenchCycleLayout,
        Workbench,
        "v",
        "Cycle layout",
        "Workbench: cycle view layout (dashboard, split, classic, bench, jobs, gpu)",
        true,
        ["view", "layout", "monitor", "telemetry", "gpu"]
    ),
    command!(
        ModelOpsToggleMode,
        ModelOps,
        "Ctrl+M / m",
        "Toggle mode",
        "Model Ops: toggle run and bench mode",
        true,
        ["switch"]
    ),
    command!(
        ModelOpsSelectPrevious,
        ModelOps,
        "↑ / k",
        "Previous item",
        "Model Ops: select previous model or script",
        true,
        ["up", "navigate"]
    ),
    command!(
        ModelOpsSelectNext,
        ModelOps,
        "↓ / j",
        "Next item",
        "Model Ops: select next model or script",
        true,
        ["down", "navigate"]
    ),
    command!(
        ModelOpsStart,
        ModelOps,
        "Ctrl+R / Enter / r",
        "Start selected",
        "Model Ops: load model or run bench script",
        true,
        ["serve", "execute", "launch"]
    ),
    command!(
        ModelOpsStop,
        ModelOps,
        "Ctrl+S / s",
        "Stop active",
        "Model Ops: unload model or stop bench script",
        true,
        ["terminate", "kill"]
    ),
    command!(
        ModelOpsRefreshModels,
        ModelOps,
        "r (run mode)",
        "Refresh models",
        "Model Ops: refresh llama-swap models",
        true,
        ["reload", "sync"]
    ),
    command!(
        ModelOpsFocusFilter,
        ModelOps,
        "Ctrl+F / /",
        "Filter items",
        "Model Ops: focus model or script filter",
        true,
        ["find", "search"]
    ),
    command!(
        ModelOpsFocusTable,
        ModelOps,
        "Ctrl+J",
        "Item table",
        "Model Ops: focus model or script table",
        true,
        ["list"]
    ),
    command!(
        ModelOpsFocusEditor,
        ModelOps,
        "Ctrl+U",
        "Script editor",
        "Model Ops: focus script editor",
        true,
        ["edit", "bench"]
    ),
    command!(
        ModelOpsSaveScript,
        ModelOps,
        "Alt+P",
        "Save script",
        "Model Ops: save edited script snapshot",
        true,
        ["version", "bench"]
    ),
    command!(
        ModelOpsReloadScript,
        ModelOps,
        "Alt+O",
        "Reload script",
        "Model Ops: reload script from disk",
        true,
        ["discard", "bench"]
    ),
    command!(
        ModelOpsRestoreScript,
        ModelOps,
        "Alt+V",
        "Restore script",
        "Model Ops: restore a script snapshot",
        true,
        ["version", "history", "bench"]
    ),
    command!(
        ModelOpsClearLog,
        ModelOps,
        "Ctrl+L",
        "Clear log",
        "Model Ops: clear run log",
        true,
        ["output"]
    ),
    command!(
        ChatCompose,
        Chat,
        "i / Enter",
        "Compose message",
        "Chat: focus message composer",
        true,
        ["input", "prompt", "write"]
    ),
    command!(
        ChatSend,
        Chat,
        "Enter",
        "Send message",
        "Chat: send composed message",
        true,
        ["submit", "prompt"]
    ),
    command!(
        ChatRefreshModels,
        Chat,
        "r",
        "Refresh models",
        "Chat: refresh llama-swap models",
        true,
        ["reload", "sync"]
    ),
    command!(
        ChatConnect,
        Chat,
        "Ctrl+G",
        "Connect",
        "Chat: connect to server",
        true,
        ["endpoint", "probe"]
    ),
    command!(
        ChatDetect,
        Chat,
        "Ctrl+B",
        "Detect server",
        "Chat: auto-detect a local server",
        true,
        ["port", "endpoint", "probe"]
    ),
    command!(
        ChatKillDetected,
        Chat,
        "K",
        "Kill detected server",
        "Chat: terminate an explicitly detected external llama-server",
        true,
        ["stop", "terminate", "external", "server"]
    ),
    command!(
        ChatPrompts,
        Chat,
        "l",
        "Prompt library",
        "Chat: load a saved system prompt",
        true,
        ["system", "instructions", "persona", "library"]
    ),
    command!(
        ChatClear,
        Chat,
        "Ctrl+X / x",
        "Clear chat",
        "Chat: clear current conversation",
        true,
        ["history", "messages"]
    ),
    command!(
        ChatSave,
        Chat,
        "Alt+S",
        "Save session",
        "Chat: save session as Markdown and JSON",
        true,
        ["persist", "history"]
    ),
    command!(
        ChatSessions,
        Chat,
        "o",
        "Saved sessions",
        "Chat: browse and load saved sessions",
        true,
        ["restore", "history", "open"]
    ),
    command!(
        ChatEditSystemPrompt,
        Chat,
        "p",
        "Edit system prompt",
        "Chat: edit system prompt",
        true,
        ["instructions", "persona"]
    ),
    command!(
        ChatToggleThinking,
        Chat,
        "t",
        "Toggle thinking",
        "Chat: toggle /think prefix",
        true,
        ["reasoning"]
    ),
    command!(
        ChatDecreaseTemperature,
        Chat,
        "[",
        "Lower temperature",
        "Chat: decrease temperature",
        true,
        ["sampling", "less creative"]
    ),
    command!(
        ChatIncreaseTemperature,
        Chat,
        "]",
        "Raise temperature",
        "Chat: increase temperature",
        true,
        ["sampling", "more creative"]
    ),
    command!(
        ChatDecreaseMaxTokens,
        Chat,
        "-",
        "Fewer max tokens",
        "Chat: decrease maximum response tokens",
        true,
        ["length", "shorter"]
    ),
    command!(
        ChatIncreaseMaxTokens,
        Chat,
        "+",
        "More max tokens",
        "Chat: increase maximum response tokens",
        true,
        ["length", "longer"]
    ),
    command!(
        BrowserSelectPrevious,
        Browser,
        "↑ / k",
        "Previous file",
        "Model Browser: select previous GGUF file",
        true,
        ["up", "navigate"]
    ),
    command!(
        BrowserSelectNext,
        Browser,
        "↓ / j",
        "Next file",
        "Model Browser: select next GGUF file",
        true,
        ["down", "navigate"]
    ),
    command!(
        BrowserScan,
        Browser,
        "Alt+R / r / Enter",
        "Scan GGUF files",
        "Model Browser: scan GGUF directory",
        true,
        ["refresh", "inventory", "models"]
    ),
    command!(
        BrowserFocusPath,
        Browser,
        "Alt+G / g",
        "Root path",
        "Model Browser: focus root path input",
        true,
        ["directory", "folder", "edit"]
    ),
    command!(
        BrowserFocusTable,
        Browser,
        "Alt+J",
        "GGUF table",
        "Model Browser: focus GGUF table",
        true,
        ["files", "list"]
    ),
    command!(
        BrowserFocusFilter,
        Browser,
        "/",
        "Filter files",
        "Model Browser: focus path and metadata filter",
        true,
        ["find", "search", "quantization", "architecture"]
    ),
    command!(
        BrowserChangeSort,
        Browser,
        "c",
        "Sort files",
        "Model Browser: change GGUF sort order",
        true,
        ["size", "modified", "path", "quantization"]
    ),
    command!(
        BrowserToggleRecursive,
        Browser,
        "t",
        "Toggle recursive scan",
        "Model Browser: toggle recursive or top-level scan",
        true,
        ["directory", "tree", "depth"]
    ),
    command!(
        DownloadSelectPrevious,
        Download,
        "↑ / k",
        "Previous model",
        "Download: select previous model",
        true,
        ["up", "navigate"]
    ),
    command!(
        DownloadSelectNext,
        Download,
        "↓ / j",
        "Next model",
        "Download: select next model",
        true,
        ["down", "navigate"]
    ),
    command!(
        DownloadFocusTable,
        Download,
        "Alt+T",
        "Model table",
        "Download: focus model table",
        true,
        ["list"]
    ),
    command!(
        DownloadFocusEditor,
        Download,
        "Alt+I",
        "Model editor",
        "Download: focus model editor",
        true,
        ["edit", "fields"]
    ),
    command!(
        DownloadToggleEnabled,
        Download,
        "Space",
        "Toggle enabled",
        "Download: toggle selected model enabled",
        true,
        ["disable", "checkbox"]
    ),
    command!(
        DownloadLoadConfig,
        Download,
        "Alt+O",
        "Load config",
        "Download: reload config from disk",
        true,
        ["open", "refresh"]
    ),
    command!(
        DownloadSaveConfig,
        Download,
        "Alt+W / w",
        "Save config",
        "Download: save config and create snapshot",
        true,
        ["write", "version"]
    ),
    command!(
        DownloadValidateConfig,
        Download,
        "Alt+V / v",
        "Validate config",
        "Download: validate config",
        true,
        ["check", "errors"]
    ),
    command!(
        DownloadRestoreConfig,
        Download,
        "Alt+R",
        "Restore config",
        "Download: restore a config snapshot",
        true,
        ["version", "history"]
    ),
    command!(
        DownloadAddModel,
        Download,
        "Alt+N",
        "Add model",
        "Download: add a model entry",
        true,
        ["new", "create"]
    ),
    command!(
        DownloadApplyEdit,
        Download,
        "Alt+A",
        "Apply edit",
        "Download: apply editor fields to config",
        true,
        ["update", "model"]
    ),
    command!(
        DownloadDeleteModel,
        Download,
        "Alt+K",
        "Delete model",
        "Download: delete selected model entry",
        true,
        ["remove"]
    ),
    command!(
        DownloadSelected,
        Download,
        "Alt+D / d",
        "Download selected",
        "Download: download selected model",
        true,
        ["fetch", "hugging face"]
    ),
    command!(
        DownloadEnabled,
        Download,
        "Alt+E / e",
        "Download enabled",
        "Download: download all enabled models",
        true,
        ["fetch", "batch", "hugging face"]
    ),
    command!(
        DownloadClearLog,
        Download,
        "Alt+Y",
        "Clear log",
        "Download: clear download log",
        true,
        ["output"]
    ),
    command!(
        JobsSelectPrevious,
        Jobs,
        "↑ / k",
        "Previous job",
        "Jobs: select previous job",
        true,
        ["up", "navigate"]
    ),
    command!(
        JobsSelectNext,
        Jobs,
        "↓ / j",
        "Next job",
        "Jobs: select next job",
        true,
        ["down", "navigate"]
    ),
    command!(
        JobsStop,
        Jobs,
        "s",
        "Stop job",
        "Jobs: stop running job",
        true,
        ["terminate", "kill"]
    ),
    command!(
        JobsRetry,
        Jobs,
        "r",
        "Retry job",
        "Jobs: retry selected job",
        true,
        ["rerun", "restart"]
    ),
    command!(
        JobsClear,
        Jobs,
        "Del / c",
        "Clear jobs",
        "Jobs: clear job history",
        true,
        ["delete", "remove"]
    ),
    command!(
        MaintenanceSelectPrevious,
        Maintenance,
        "↑ / k",
        "Previous script",
        "Maintenance: select previous script",
        true,
        ["up", "navigate"]
    ),
    command!(
        MaintenanceSelectNext,
        Maintenance,
        "↓ / j",
        "Next script",
        "Maintenance: select next script",
        true,
        ["down", "navigate"]
    ),
    command!(
        MaintenanceRun,
        Maintenance,
        "Ctrl+R / Enter / r",
        "Run script",
        "Maintenance: run selected script",
        true,
        ["start", "execute", "launch"]
    ),
    command!(
        MaintenanceStop,
        Maintenance,
        "Ctrl+S / s",
        "Stop script",
        "Maintenance: stop running script",
        true,
        ["terminate", "kill"]
    ),
    command!(
        MaintenanceFocusEditor,
        Maintenance,
        "Ctrl+U",
        "Script editor",
        "Maintenance: focus script editor",
        true,
        ["edit"]
    ),
    command!(
        MaintenanceSaveScript,
        Maintenance,
        "Alt+P",
        "Save script",
        "Maintenance: save edited script snapshot",
        true,
        ["write", "version"]
    ),
    command!(
        MaintenanceReloadScript,
        Maintenance,
        "Alt+O",
        "Reload script",
        "Maintenance: reload script from disk",
        true,
        ["refresh", "discard"]
    ),
    command!(
        MaintenanceRestoreScript,
        Maintenance,
        "Alt+V",
        "Restore script",
        "Maintenance: restore a script snapshot",
        true,
        ["version", "history"]
    ),
    command!(
        MaintenanceClearLog,
        Maintenance,
        "Ctrl+L",
        "Clear log",
        "Maintenance: clear script log",
        true,
        ["output"]
    ),
];

/// Look up metadata for a typed command identifier.
pub fn command(id: CommandId) -> Option<&'static CommandSpec> {
    COMMANDS.iter().find(|spec| spec.id == id)
}

/// Return all registered commands for a context, including parity commands
/// which are not implemented or enabled yet. Global commands are included for
/// a tab context.
pub fn commands_for_context(context: CommandContext) -> Vec<&'static CommandSpec> {
    COMMANDS
        .iter()
        .filter(|spec| spec.is_available_in(context))
        .collect()
}

/// Return commands which can currently be shown and dispatched in a context.
/// Callers can apply additional runtime state predicates after this static
/// filter.
pub fn visible_commands(context: CommandContext) -> Vec<&'static CommandSpec> {
    COMMANDS
        .iter()
        .filter(|spec| spec.is_available_in(context) && spec.is_visible())
        .collect()
}

/// Search visible global and contextual commands. Every whitespace-delimited
/// query token must match an identifier, label, shortcut, or keyword. More
/// exact matches rank first and catalog declaration order breaks all ties.
pub fn search_commands(context: CommandContext, query: &str) -> Vec<&'static CommandSpec> {
    search_specs(visible_commands(context), query)
}

/// Search every visible command, regardless of its owning tab. This is useful
/// for an all-actions palette while [`search_commands`] supports a contextual
/// palette or command bar.
pub fn search_all_commands(query: &str) -> Vec<&'static CommandSpec> {
    search_specs(COMMANDS.iter().filter(|spec| spec.is_visible()), query)
}

fn search_specs(
    specs: impl IntoIterator<Item = &'static CommandSpec>,
    query: &str,
) -> Vec<&'static CommandSpec> {
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|token| token.to_lowercase())
        .collect();
    let specs: Vec<&'static CommandSpec> = specs.into_iter().collect();
    if tokens.is_empty() {
        return specs;
    }

    let mut ranked: Vec<(u32, usize, &'static CommandSpec)> = specs
        .into_iter()
        .enumerate()
        .filter_map(|(declaration_index, spec)| {
            command_rank(spec, &tokens).map(|rank| (rank, declaration_index, spec))
        })
        .collect();
    ranked.sort_by_key(|(rank, declaration_index, _)| (*rank, *declaration_index));
    ranked.into_iter().map(|(_, _, spec)| spec).collect()
}

fn command_rank(spec: &CommandSpec, tokens: &[String]) -> Option<u32> {
    let id = spec.id.as_str().to_lowercase();
    let label = spec.label.to_lowercase();
    let palette_label = spec.palette_label.to_lowercase();
    let shortcut = spec.shortcut.to_lowercase();
    let keywords: Vec<String> = spec
        .keywords
        .iter()
        .map(|keyword| keyword.to_lowercase())
        .collect();

    tokens.iter().try_fold(0_u32, |total, token| {
        token_rank(token, &id, &label, &palette_label, &shortcut, &keywords)
            .map(|rank| total + rank)
    })
}

fn token_rank(
    token: &str,
    id: &str,
    label: &str,
    palette_label: &str,
    shortcut: &str,
    keywords: &[String],
) -> Option<u32> {
    if token == label || token == palette_label || token == id || token == shortcut {
        return Some(0);
    }
    if label.starts_with(token) || palette_label.starts_with(token) {
        return Some(1);
    }

    if words(label)
        .chain(words(palette_label))
        .chain(keywords.iter().flat_map(|keyword| words(keyword)))
        .any(|word| word == token)
    {
        return Some(2);
    }
    if words(label)
        .chain(words(palette_label))
        .chain(keywords.iter().flat_map(|keyword| words(keyword)))
        .any(|word| word.starts_with(token))
    {
        return Some(3);
    }
    if palette_label.contains(token) || label.contains(token) {
        return Some(4);
    }
    if keywords.iter().any(|keyword| keyword.contains(token)) {
        return Some(5);
    }
    if id.contains(token) {
        return Some(6);
    }
    shortcut.contains(token).then_some(7)
}

fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|character: char| {
        !(character.is_alphanumeric() || matches!(character, '+' | '?' | '←' | '→'))
    })
    .filter(|word| !word.is_empty())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn command_ids_are_unique_and_complete() {
        let ids: HashSet<_> = COMMANDS.iter().map(|spec| spec.id).collect();
        let names: HashSet<_> = COMMANDS.iter().map(|spec| spec.id.as_str()).collect();

        assert_eq!(COMMANDS.len(), CommandId::ALL.len());
        assert_eq!(ids.len(), COMMANDS.len());
        assert_eq!(names.len(), COMMANDS.len());
        assert!(CommandId::ALL.iter().all(|id| ids.contains(id)));
    }

    #[test]
    fn every_command_has_shortcut_and_label_metadata() {
        for spec in COMMANDS {
            assert!(
                !spec.label.trim().is_empty(),
                "missing label for {}",
                spec.id
            );
            assert!(
                !spec.palette_label.trim().is_empty(),
                "missing palette label for {}",
                spec.id
            );
            assert!(
                !spec.shortcut.trim().is_empty(),
                "missing shortcut display for {}",
                spec.id
            );
            assert!(!spec.contexts.is_empty(), "missing context for {}", spec.id);
            assert!(
                !spec.enabled || spec.implemented,
                "unimplemented command {} cannot be enabled",
                spec.id
            );
        }
    }

    #[test]
    fn every_tab_owns_commands() {
        for context in CommandContext::TABS {
            assert!(
                COMMANDS.iter().any(|spec| spec.contexts.contains(&context)),
                "{context} has no commands"
            );
        }
    }

    #[test]
    fn contextual_commands_include_global_but_not_other_tabs() {
        let browser = commands_for_context(CommandContext::Browser);
        assert!(browser.iter().any(|spec| spec.id == CommandId::Quit));
        assert!(browser.iter().any(|spec| spec.id == CommandId::BrowserScan));
        assert!(!browser
            .iter()
            .any(|spec| spec.id == CommandId::DownloadSaveConfig));

        let global = commands_for_context(CommandContext::Global);
        assert!(global.iter().any(|spec| spec.id == CommandId::OpenChat));
        assert!(!global.iter().any(|spec| spec.id == CommandId::ChatCompose));

        let visible_browser = visible_commands(CommandContext::Browser);
        assert!(visible_browser
            .iter()
            .any(|spec| spec.id == CommandId::BrowserScan));
        assert!(visible_browser
            .iter()
            .any(|spec| spec.id == CommandId::BrowserFocusFilter));

        let visible_chat = visible_commands(CommandContext::Chat);
        assert!(visible_chat
            .iter()
            .any(|spec| spec.id == CommandId::ChatConnect));
        assert!(visible_chat
            .iter()
            .any(|spec| spec.id == CommandId::ChatDetect));
    }

    #[test]
    fn palette_search_is_case_insensitive_and_matches_all_tokens() {
        let results = search_commands(CommandContext::Workbench, "MODEL LoAd");
        assert_eq!(
            results.first().map(|spec| spec.id),
            Some(CommandId::WorkbenchLoadModel)
        );
        assert!(results.iter().all(|spec| {
            let searchable = format!(
                "{} {} {} {} {}",
                spec.id,
                spec.label,
                spec.palette_label,
                spec.shortcut,
                spec.keywords.join(" ")
            )
            .to_lowercase();
            searchable.contains("model") && searchable.contains("load")
        }));

        let shortcut = search_commands(CommandContext::Download, "ALT+W");
        assert_eq!(
            shortcut.first().map(|spec| spec.id),
            Some(CommandId::DownloadSaveConfig)
        );
    }

    #[test]
    fn palette_search_uses_stable_catalog_order_for_ties() {
        let results = search_all_commands("open tab");
        let navigation: Vec<_> = results
            .iter()
            .filter(|spec| {
                matches!(
                    spec.id,
                    CommandId::OpenWorkbench
                        | CommandId::OpenModelOps
                        | CommandId::OpenChat
                        | CommandId::OpenBrowser
                        | CommandId::OpenDownload
                        | CommandId::OpenJobs
                        | CommandId::OpenMaintenance
                )
            })
            .map(|spec| spec.id)
            .collect();
        assert_eq!(
            navigation,
            vec![
                CommandId::OpenWorkbench,
                CommandId::OpenModelOps,
                CommandId::OpenChat,
                CommandId::OpenBrowser,
                CommandId::OpenDownload,
                CommandId::OpenJobs,
                CommandId::OpenMaintenance,
            ]
        );

        let first = search_commands(CommandContext::Jobs, "");
        let second = search_commands(CommandContext::Jobs, "   ");
        assert_eq!(first, second);
    }
}
