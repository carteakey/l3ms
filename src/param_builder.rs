//! Param Builder: interactive construction of a `llama-server` flag set.
//!
//! State and pure logic for the Param Builder tab. The curated flag table
//! lives in [`crate::param_registry`]; this module holds the editable
//! values, free-form custom flags, import of an existing yaml entry's cmd
//! block, deterministic cmd-block generation, and the text-level yaml edit
//! that replaces an entry's `cmd:` body (snapshot + SIGHUP are handled by
//! the caller, matching house conventions).

use anyhow::{Context, Result};
use crossterm::event::{KeyCode, KeyEvent};

use crate::param_registry::{find_by_flag, ParamValue, PRESETS, REGISTRY};
use crate::swap_yaml;

/// Maximum length of one editable value / custom flag line.
const MAX_EDIT_LEN: usize = 240;

/// Full editable parameter state: one value per registry entry plus
/// free-form custom flag lines.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamValues {
    pub values: Vec<ParamValue>,
    pub custom: Vec<String>,
}

impl ParamValues {
    pub fn new() -> Self {
        Self {
            values: vec![ParamValue::Unset; REGISTRY.len()],
            custom: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.values
            .iter_mut()
            .for_each(|value| *value = ParamValue::Unset);
        self.custom.clear();
    }

    /// Flags in registry order (only set values), then custom lines.
    /// Does not include the binary, `-m`, or `--port` (callers add those).
    pub fn generate_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        for (index, def) in REGISTRY.iter().enumerate() {
            match &self.values[index] {
                ParamValue::Unset => {}
                ParamValue::On => {
                    // TriState `on` is an explicit value; a plain toggle is
                    // just the flag itself.
                    if matches!(def.kind, crate::param_registry::ParamKind::TriState) {
                        flags.push(format!("{} on", def.flag));
                    } else {
                        flags.push(def.flag.to_owned());
                    }
                }
                ParamValue::Integer(value) => {
                    flags.push(format!("{} {}", def.flag, value));
                }
                ParamValue::Off => flags.push(format!("{} off", def.flag)),
                ParamValue::Choice(option) => {
                    if let Some(options) = def.kind_options() {
                        if let Some(option) = options.get(*option) {
                            flags.push(format!("{} {}", def.flag, option));
                        }
                    }
                }
                ParamValue::Text(text) => {
                    let text = text.trim();
                    if text.is_empty() {
                        continue;
                    }
                    // Re-quote values with spaces so the shell sees one token.
                    if text.contains(char::is_whitespace) {
                        let quoted = if text.contains('"') && !text.contains('\'') {
                            format!("'{text}'")
                        } else {
                            format!("\"{text}\"")
                        };
                        flags.push(format!("{} {quoted}", def.flag));
                    } else {
                        flags.push(format!("{} {text}", def.flag));
                    }
                }
            }
        }
        for line in &self.custom {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                flags.push(trimmed.to_owned());
            }
        }
        flags
    }
}

/// Which pane inside the Param Builder holds keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamFocus {
    List,
    Preview,
}

/// One row of the param list: a category header, a registry param, a
/// custom flag line, or the add-custom placeholder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRow {
    Header(&'static str),
    Param(usize),
    Custom(usize),
    AddCustom,
}

impl ParamRow {
    pub fn is_selectable(&self) -> bool {
        !matches!(self, Self::Header(_))
    }
}

/// Target of an in-progress single-line edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditTarget {
    Param(usize),
    Custom(usize),
    CustomNew,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditState {
    pub target: EditTarget,
    pub buffer: String,
}

/// Overlay shown above the tab: preset picker or yaml entry picker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamOverlay {
    Presets,
    /// `true` = import from the picked entry, `false` = save into it.
    Entries {
        import: bool,
    },
}

#[derive(Debug, Clone)]
pub struct ParamBuilderState {
    pub values: ParamValues,
    /// Binary invocation line preserved from the imported entry
    /// (e.g. `taskset -c ${cpu_range} ${qwen38_master_server}`).
    pub binary: String,
    /// Model path imported from (or destined for) an entry's `-m`.
    pub model_path: Option<String>,
    pub focus: ParamFocus,
    pub selected: usize,
    pub scroll: usize,
    pub preview_scroll: usize,
    pub editing: Option<EditState>,
    pub overlay: Option<ParamOverlay>,
    pub overlay_selected: usize,
}

impl Default for ParamBuilderState {
    fn default() -> Self {
        Self {
            values: ParamValues::new(),
            binary: "llama-server".to_owned(),
            model_path: None,
            focus: ParamFocus::List,
            selected: 0,
            scroll: 0,
            preview_scroll: 0,
            editing: None,
            overlay: None,
            overlay_selected: 0,
        }
    }
}

impl ParamBuilderState {
    /// Flattened display rows: category headers with their params, then the
    /// custom-flag section.
    pub fn rows(&self) -> Vec<ParamRow> {
        let mut rows = Vec::with_capacity(REGISTRY.len() + 16);
        let mut last_category: Option<&'static str> = None;
        for (index, def) in REGISTRY.iter().enumerate() {
            let title = def.category.title();
            if last_category != Some(title) {
                rows.push(ParamRow::Header(title));
                last_category = Some(title);
            }
            rows.push(ParamRow::Param(index));
        }
        rows.push(ParamRow::Header("Custom flags"));
        for index in 0..self.values.custom.len() {
            rows.push(ParamRow::Custom(index));
        }
        rows.push(ParamRow::AddCustom);
        rows
    }

    fn selectable_count(rows: &[ParamRow]) -> usize {
        rows.iter().filter(|row| row.is_selectable()).count()
    }

    /// Index of the `selected`-th selectable row in `rows`.
    fn selectable_index(rows: &[ParamRow], selected: usize) -> Option<usize> {
        let mut count = 0;
        for (index, row) in rows.iter().enumerate() {
            if row.is_selectable() {
                if count == selected {
                    return Some(index);
                }
                count += 1;
            }
        }
        None
    }

    /// Move selection by `delta` selectable rows, skipping headers.
    pub fn move_selection(&mut self, delta: isize) {
        let rows = self.rows();
        let count = Self::selectable_count(&rows);
        if count == 0 {
            return;
        }
        let total = count as isize;
        let next = (self.selected as isize + delta).rem_euclid(total);
        self.selected = next as usize;
    }

    pub fn selected_row(&self) -> ParamRow {
        let rows = self.rows();
        let index = Self::selectable_index(&rows, self.selected)
            .unwrap_or_else(|| Self::selectable_index(&rows, 0).unwrap_or(0));
        rows[index].clone()
    }

    /// `(row index into `rows()`, row)` of the current selection.
    pub fn selected_row_position(&self) -> (usize, ParamRow) {
        let rows = self.rows();
        let index = Self::selectable_index(&rows, self.selected)
            .unwrap_or_else(|| Self::selectable_index(&rows, 0).unwrap_or(0));
        (index, rows[index].clone())
    }

    /// Jump the selection to a selectable row by its position in `rows()`.
    pub fn select_row_position(&mut self, position: usize) {
        let rows = self.rows();
        let Some(index) = rows.get(position) else {
            return;
        };
        if !index.is_selectable() {
            return;
        }
        let selected = rows[..position]
            .iter()
            .filter(|row| row.is_selectable())
            .count();
        self.selected = selected;
    }

    /// Clamp `selected` into range (e.g. after custom rows were removed).
    pub fn clamp_selection(&mut self) {
        let count = Self::selectable_count(&self.rows());
        if count == 0 {
            self.selected = 0;
        } else if self.selected >= count {
            self.selected = count - 1;
        }
    }

    /// Cycle or begin editing the selected row's value. Returns a status
    /// message when the action is worth reporting.
    pub fn activate(&mut self) -> Option<String> {
        match self.selected_row() {
            ParamRow::Header(_) => None,
            ParamRow::Param(index) => {
                let def = &REGISTRY[index];
                match def.kind {
                    crate::param_registry::ParamKind::Toggle => {
                        let value = &mut self.values.values[index];
                        *value = match value {
                            ParamValue::On => ParamValue::Unset,
                            _ => ParamValue::On,
                        };
                        Some(format!("{} toggled", def.label))
                    }
                    crate::param_registry::ParamKind::TriState => {
                        let value = &mut self.values.values[index];
                        *value = match value {
                            ParamValue::Unset => ParamValue::On,
                            ParamValue::On => ParamValue::Off,
                            _ => ParamValue::Unset,
                        };
                        Some(format!("{} = {}", def.label, value.display(def)))
                    }
                    crate::param_registry::ParamKind::Choice(options) => {
                        let value = &mut self.values.values[index];
                        let next = match value {
                            ParamValue::Unset => 0,
                            ParamValue::Choice(current) if *current + 1 < options.len() => {
                                *current + 1
                            }
                            _ => usize::MAX, // wraps back to Unset
                        };
                        *value = if next == usize::MAX {
                            ParamValue::Unset
                        } else {
                            ParamValue::Choice(next)
                        };
                        Some(format!("{} = {}", def.label, value.display(def)))
                    }
                    crate::param_registry::ParamKind::Integer { .. }
                    | crate::param_registry::ParamKind::Text => {
                        self.start_edit_at(EditTarget::Param(index));
                        Some(String::new())
                    }
                }
            }
            ParamRow::Custom(index) => {
                self.start_edit_at(EditTarget::Custom(index));
                Some(String::new())
            }
            ParamRow::AddCustom => {
                self.start_edit_at(EditTarget::CustomNew);
                Some(String::new())
            }
        }
    }

    /// Adjust the selected numeric value by signed `steps` of its step size.
    pub fn adjust(&mut self, steps: isize) -> Option<String> {
        let ParamRow::Param(index) = self.selected_row() else {
            return None;
        };
        let def = &REGISTRY[index];
        let crate::param_registry::ParamKind::Integer { min, max, step } = def.kind else {
            return None;
        };
        let value = &mut self.values.values[index];
        let current = match value {
            ParamValue::Integer(value) => *value,
            _ => min.max(0).min(max),
        };
        let delta = step * steps as i64;
        let next = (current + delta).clamp(min, max);
        *value = ParamValue::Integer(next);
        Some(format!("{} = {}", def.label, value.display(def)))
    }

    fn start_edit_at(&mut self, target: EditTarget) {
        let buffer = match &target {
            EditTarget::Param(index) => match &self.values.values[*index] {
                ParamValue::Text(text) => text.clone(),
                ParamValue::Integer(value) => value.to_string(),
                _ => String::new(),
            },
            EditTarget::Custom(index) => self.values.custom[*index].clone(),
            EditTarget::CustomNew => String::new(),
        };
        self.editing = Some(EditState { target, buffer });
    }

    pub fn editing(&self) -> bool {
        self.editing.is_some()
    }

    /// Feed a key event to the active editor. Returns a status message on
    /// commit (empty string = silent success) and `None` while still editing.
    pub fn edit_key(&mut self, key: KeyEvent) -> Option<String> {
        let edit = self.editing.as_mut()?;
        match key.code {
            KeyCode::Esc => {
                self.editing = None;
                Some("edit cancelled".to_owned())
            }
            KeyCode::Enter => self.commit_edit(),
            KeyCode::Backspace => {
                edit.buffer.pop();
                None
            }
            KeyCode::Char(character) => {
                if edit.buffer.len() < MAX_EDIT_LEN {
                    edit.buffer.push(character);
                }
                None
            }
            _ => None,
        }
    }

    fn commit_edit(&mut self) -> Option<String> {
        let edit = self.editing.clone()?;
        let text = edit.buffer.trim().to_owned();
        match edit.target {
            EditTarget::Param(index) => {
                let def = &REGISTRY[index];
                if text.is_empty() {
                    self.values.values[index] = ParamValue::Unset;
                    self.editing = None;
                    return Some(format!("{} cleared", def.label));
                }
                match def.kind {
                    crate::param_registry::ParamKind::Integer { min, max, .. } => {
                        let Ok(value) = text.parse::<i64>() else {
                            return Some(format!(
                                "{}: {text:?} is not an integer ({min}..={max})",
                                def.label
                            ));
                        };
                        self.values.values[index] = ParamValue::Integer(value.clamp(min, max));
                    }
                    _ => {
                        self.values.values[index] = ParamValue::Text(text);
                    }
                }
            }
            EditTarget::Custom(index) => {
                if text.is_empty() {
                    self.values.custom.remove(index);
                } else {
                    self.values.custom[index] = text;
                }
            }
            EditTarget::CustomNew => {
                if !text.is_empty() {
                    self.values.custom.push(text);
                }
            }
        }
        self.editing = None;
        self.clamp_selection();
        Some(String::new())
    }

    /// Remove the selected custom flag line (`x` key). Returns a status
    /// message when a line was removed.
    pub fn delete_custom(&mut self) -> Option<String> {
        if let ParamRow::Custom(index) = self.selected_row() {
            self.values.custom.remove(index);
            self.clamp_selection();
            return Some("custom flag removed".to_owned());
        }
        None
    }

    /// Apply a built-in preset: clears all values first.
    pub fn apply_preset(&mut self, index: usize) -> Option<String> {
        let preset = PRESETS.get(index)?;
        self.values.clear();
        for (flag, value) in preset.values {
            let Some(registry_index) = find_by_flag(flag) else {
                continue;
            };
            self.values.values[registry_index] = match value {
                crate::param_registry::PresetValue::Int(value) => ParamValue::Integer(*value),
                crate::param_registry::PresetValue::On => ParamValue::On,
                crate::param_registry::PresetValue::Off => ParamValue::Off,
                crate::param_registry::PresetValue::Opt(option) => {
                    match REGISTRY[registry_index].kind_options() {
                        Some(options) => options
                            .iter()
                            .position(|candidate| candidate == option)
                            .map_or(ParamValue::Unset, ParamValue::Choice),
                        None => ParamValue::Text((*option).to_owned()),
                    }
                }
                crate::param_registry::PresetValue::Text(text) => {
                    ParamValue::Text((*text).to_owned())
                }
            };
        }
        Some(format!("preset applied: {}", preset.name))
    }

    /// Import a parsed cmd block: `binary` is the invocation prefix,
    /// `token_lines` the remaining CLI tokens grouped per original line,
    /// `model_path` the entry's `-m`.
    pub fn import(&mut self, binary: &str, token_lines: &[Vec<String>], model_path: Option<&str>) {
        self.values.clear();
        self.binary = binary.trim().to_owned();
        if self.binary.is_empty() {
            self.binary = "llama-server".to_owned();
        }
        self.model_path = model_path.map(|path| path.to_owned());

        for tokens in token_lines {
            // --port is managed by llama-swap (${PORT} macro); never import it.
            let mut tokens = tokens.clone();
            let mut port_index = 0;
            while port_index < tokens.len() {
                if tokens[port_index] == "--port" {
                    let end = (port_index + 2).min(tokens.len());
                    tokens.drain(port_index..end);
                } else if tokens[port_index].starts_with("--port=") {
                    tokens.remove(port_index);
                } else {
                    port_index += 1;
                }
            }

            let mut index = 0;
            while index < tokens.len() {
                let token = tokens[index].as_str();
                index += 1;
                if !token.starts_with('-') || token == "-" {
                    // Orphan token: keep the remainder of the line together.
                    let mut line = token.to_owned();
                    while index < tokens.len() {
                        line.push(' ');
                        line.push_str(&tokens[index]);
                        index += 1;
                    }
                    self.values.custom.push(line);
                    continue;
                }
                // --name=value form
                if let Some((name, value)) = token.split_once('=') {
                    match find_by_flag(name).map(|index| &REGISTRY[index]) {
                        Some(def) if def.takes_value() => {
                            let value = strip_quotes(value);
                            if def.parse_value(value).is_none() {
                                self.values.custom.push(token.to_owned());
                            } else {
                                self.set_imported(def, value);
                            }
                        }
                        _ => self.values.custom.push(token.to_owned()),
                    }
                    continue;
                }
                let Some(def_index) = find_by_flag(token) else {
                    // Unknown flag: consume its value tokens up to the next
                    // flag within the same line, preserving the grouping.
                    let mut line = token.to_owned();
                    while index < tokens.len() {
                        line.push(' ');
                        line.push_str(&tokens[index]);
                        index += 1;
                    }
                    self.values.custom.push(line);
                    continue;
                };
                let def = &REGISTRY[def_index];
                if !def.takes_value() {
                    self.values.values[def_index] = ParamValue::On;
                    continue;
                }
                let Some(value_token) = tokens.get(index) else {
                    self.values.custom.push(token.to_owned());
                    continue;
                };
                index += 1;
                let value = strip_quotes(value_token);
                if def.parse_value(value).is_none() {
                    self.values.custom.push(format!("{token} {value_token}"));
                } else {
                    self.set_imported(def, value);
                }
            }
        }
        self.clamp_selection();
    }

    fn set_imported(&mut self, def: &crate::param_registry::ParamDef, value: &str) {
        let index = REGISTRY
            .iter()
            .position(|candidate| candidate.flag == def.flag)
            .unwrap_or(0);
        if let Some(parsed) = def.parse_value(value) {
            self.values.values[index] = parsed;
        }
    }

    /// Full cmd-block body: binary line, `-m`, generated flags, and the
    /// llama-swap-managed `--port ${PORT}`.
    pub fn cmd_body(&self) -> Vec<String> {
        let mut body = vec![self.binary.clone()];
        if let Some(path) = self.model_path.as_deref() {
            body.push(format!("-m {path}"));
        }
        body.extend(self.values.generate_flags());
        body.push("--port ${PORT}".to_owned());
        body
    }

    /// Preview lines for the right pane (same content as `cmd_body`).
    pub fn preview_lines(&self) -> Vec<String> {
        self.cmd_body()
    }
}

/// Split an entry cmd block into `(binary line, token lines, model path)`.
/// The binary line is everything up to and including the server binary
/// token; `-m`/`--model` pairs are lifted out; each remaining cmd-block
/// line becomes one `Vec<String>` of quote-aware tokens so line grouping
/// survives import/export round trips. Returns `None` when the entry has
/// no cmd block.
pub fn parse_cmd_block(body: &[&str]) -> Option<(String, Vec<Vec<String>>, Option<String>)> {
    let cmd_index = body.iter().position(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("cmd:") && trimmed[4..].trim().starts_with('|')
    })?;
    let cmd_indent = body[cmd_index].len() - body[cmd_index].trim_start().len();
    let mut content: Vec<&str> = Vec::new();
    for line in body[cmd_index + 1..].iter() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            content.push("");
            continue;
        }
        let indent = trimmed.len() - trimmed.trim_start().len();
        if indent <= cmd_indent {
            break;
        }
        content.push(trimmed.trim_start());
    }
    while content.last() == Some(&"") {
        content.pop();
    }
    if content.is_empty() {
        return None;
    }

    let mut token_lines: Vec<Vec<String>> = Vec::new();
    let mut binary = String::new();
    let mut binary_done = false;
    for line in &content {
        let mut line_tokens: Vec<String> = Vec::new();
        for token in tokenize_line(line) {
            if !binary_done {
                if binary.is_empty() {
                    binary.push_str(&token);
                } else {
                    binary.push(' ');
                    binary.push_str(&token);
                }
                // The invocation prefix ends at the server binary token:
                // plain (`llama-server`, `llama-server-exp`) or a macro
                // (`${llama_server}`, `${qwen38_master_server}`) — house
                // macros use underscores and a closing brace, so strip
                // trailing `}` and match on the `server` suffix.
                let bare = token.trim_end_matches('}');
                if bare.ends_with("llama-server") || bare.ends_with("server") {
                    binary_done = true;
                }
                continue;
            }
            line_tokens.push(token);
        }
        if !line_tokens.is_empty() {
            token_lines.push(line_tokens);
        }
    }
    if !binary_done {
        // No recognizable server binary: treat the whole first line as
        // flags and use the bare binary from PATH.
        if let Some(first_line) = token_lines.first_mut() {
            let mut first: Vec<String> = tokenize_line(content.first().unwrap_or(&""));
            first.append(first_line);
            *first_line = first;
        }
        binary = "llama-server".to_owned();
    }

    // Lift -m/--model out of the flag tokens.
    let mut model_path = None;
    let mut filtered: Vec<Vec<String>> = Vec::with_capacity(token_lines.len());
    for line_tokens in token_lines {
        let mut kept: Vec<String> = Vec::with_capacity(line_tokens.len());
        let mut index = 0;
        while index < line_tokens.len() {
            if (line_tokens[index] == "-m" || line_tokens[index] == "--model")
                && index + 1 < line_tokens.len()
            {
                model_path = Some(strip_quotes(&line_tokens[index + 1]).to_owned());
                index += 2;
                continue;
            }
            kept.push(line_tokens[index].clone());
            index += 1;
        }
        if !kept.is_empty() {
            filtered.push(kept);
        }
    }
    Some((binary, filtered, model_path))
}

/// Whitespace tokenizer that keeps double- and single-quoted spans as one
/// token, preserving the quote characters so shell semantics survive.
fn tokenize_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for character in line.chars() {
        match quote {
            Some(active) => {
                current.push(character);
                if character == active {
                    quote = None;
                }
            }
            None => {
                if character == '"' || character == '\'' {
                    quote = Some(character);
                    current.push(character);
                } else if character.is_whitespace() {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                } else {
                    current.push(character);
                }
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Strip one layer of matching surrounding quotes from a token.
fn strip_quotes(token: &str) -> &str {
    let bytes = token.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        &token[1..token.len() - 1]
    } else {
        token
    }
}

/// Replace the `cmd:` block body of entry `id` with `new_body` (indented to
/// match the existing block). Text-level and deterministic; everything else
/// in the entry (name, description, env, ttl) is preserved byte-for-byte.
pub fn replace_entry_cmd(text: &str, id: &str, new_body: &[String]) -> Result<String> {
    let trailing_newline = text.ends_with('\n');
    let lines: Vec<&str> = text.lines().collect();
    let (entry_start, entry_end) = swap_yaml::entry_span(&lines, id)
        .with_context(|| format!("model {id:?} not found in llama-swap.yaml"))?;
    let entry = &lines[entry_start..entry_end];

    let cmd_index = entry
        .iter()
        .position(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("cmd:")
        })
        .with_context(|| format!("entry {id:?} has no cmd block"))?;
    let cmd_indent = entry[cmd_index].len() - entry[cmd_index].trim_start().len();

    // Extent of the literal-block content: advance past every non-blank
    // line indented deeper than the `cmd:` key; blank lines between the
    // block and the next key are preserved as separators.
    let mut block_end = entry_start + cmd_index + 1;
    for (offset, line) in entry[cmd_index + 1..].iter().enumerate() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let indent = trimmed.len() - trimmed.trim_start().len();
        if indent <= cmd_indent {
            break;
        }
        block_end = entry_start + cmd_index + 1 + offset + 1;
    }

    let content_indent = " ".repeat(cmd_indent + 2);
    let mut updated: Vec<String> = Vec::with_capacity(lines.len() + new_body.len());
    for line in &lines[..entry_start + cmd_index + 1] {
        updated.push((*line).to_owned());
    }
    for line in new_body {
        updated.push(format!("{content_indent}{line}"));
    }
    // Everything from the first line after the literal-block content:
    // sibling keys of cmd inside the entry, and the rest of the file.
    for line in &lines[block_end.min(entry_end)..] {
        updated.push((*line).to_owned());
    }

    let mut joined = updated.join("\n");
    if trailing_newline {
        joined.push('\n');
    }
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_registry::find_by_flag;
    use crossterm::event::KeyModifiers;

    const SAMPLE: &str = r#"apiKeys: []

models:
  "alpha":
    name: "Alpha"
    description: "test entry"
    env:
      - "GGML_CUDA_GRAPH_OPT=1"
    cmd: |
      taskset -c ${cpu_range} ${llama_server}
      -m /models/alpha/model.gguf
      --alias alpha
      --fit on --fit-target 512
      -c 65536 --parallel 1
      -fa on --jinja
      --lazy-mode on
      --host 0.0.0.0 --port ${PORT}

  "beta":
    cmd: |
      llama-server -m /models/beta/beta.gguf --port ${PORT}
"#;

    fn set(values: &mut ParamValues, flag: &str, value: ParamValue) {
        values.values[find_by_flag(flag).unwrap()] = value;
    }

    #[test]
    fn rows_cover_registry_and_custom_section() {
        let state = ParamBuilderState::default();
        let rows = state.rows();
        assert_eq!(
            rows.iter()
                .filter(|row| matches!(row, ParamRow::Header(_)))
                .count(),
            crate::param_registry::ParamCategory::ALL.len() + 1
        );
        assert!(rows.contains(&ParamRow::AddCustom));
        assert_eq!(
            rows.iter()
                .filter(|row| matches!(row, ParamRow::Param(_)))
                .count(),
            REGISTRY.len()
        );
    }

    #[test]
    fn navigation_skips_headers_and_wraps() {
        let mut state = ParamBuilderState {
            selected: 0,
            ..Default::default()
        };
        state.move_selection(-1);
        let last = state.selected;
        assert!(last > 0, "backward wrap should land on the last row");
        // The AddCustom row is selectable and last.
        assert_eq!(state.selected_row(), ParamRow::AddCustom);
        state.move_selection(1);
        assert_eq!(state.selected, 0);
        state.move_selection(2);
        assert_eq!(state.selected, 2);
    }

    #[test]
    fn generate_flags_respects_kinds_and_order() {
        let mut values = ParamValues::new();
        set(&mut values, "--ctx-size", ParamValue::Integer(65_536));
        set(&mut values, "--flash-attn", ParamValue::On);
        set(&mut values, "--fit", ParamValue::Off);
        set(&mut values, "--cache-type-k", ParamValue::Choice(3));
        set(&mut values, "--jinja", ParamValue::On);
        values.custom.push("--lazy-mode on".into());
        values.custom.push("   ".into());
        let flags = values.generate_flags();
        assert_eq!(
            flags,
            vec![
                "--ctx-size 65536",
                "--flash-attn on",
                "--fit off",
                "--cache-type-k q8_0",
                "--jinja",
                "--lazy-mode on",
            ]
        );
    }

    #[test]
    fn cmd_body_ends_with_port_and_starts_with_binary() {
        let mut state = ParamBuilderState {
            binary: "taskset -c 0-3 ${llama_server}".into(),
            model_path: Some("/m/x.gguf".into()),
            ..Default::default()
        };
        set(&mut state.values, "--ctx-size", ParamValue::Integer(8192));
        let body = state.cmd_body();
        assert_eq!(body.first().unwrap(), "taskset -c 0-3 ${llama_server}");
        assert_eq!(body[1], "-m /m/x.gguf");
        assert!(body.contains(&"--ctx-size 8192".to_owned()));
        assert_eq!(body.last().unwrap(), "--port ${PORT}");
    }

    #[test]
    fn adjust_clamps_to_bounds() {
        let mut state = ParamBuilderState::default();
        let ctx = find_by_flag("--ctx-size").unwrap();
        // Row index counts headers; walk until the ctx-size param row.
        while state.selected_row() != ParamRow::Param(ctx) {
            state.move_selection(1);
        }
        state.adjust(1).unwrap();
        assert_eq!(state.values.values[ctx], ParamValue::Integer(1024));
        state.adjust(-10).unwrap();
        assert_eq!(state.values.values[ctx], ParamValue::Integer(0));
    }

    #[test]
    fn edit_integer_commit_clamps_and_rejects_garbage() {
        let mut state = ParamBuilderState::default();
        let ctx = find_by_flag("--ctx-size").unwrap();
        state.start_edit_at(EditTarget::Param(ctx));
        state.editing.as_mut().unwrap().buffer = "999999999".into();
        state.commit_edit();
        assert_eq!(state.values.values[ctx], ParamValue::Integer(2_097_152));

        state.start_edit_at(EditTarget::Param(ctx));
        state.editing.as_mut().unwrap().buffer = "nope".into();
        let message = state.commit_edit().unwrap();
        assert!(message.contains("not an integer"));
        // Invalid input keeps the editor open so it can be fixed or Esc'd.
        assert!(state.editing());
        state.edit_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!state.editing());

        state.start_edit_at(EditTarget::Param(ctx));
        state.editing.as_mut().unwrap().buffer = String::new();
        state.commit_edit();
        assert_eq!(state.values.values[ctx], ParamValue::Unset);
    }

    #[test]
    fn preset_applies_cleared_values() {
        let mut state = ParamBuilderState::default();
        set(&mut state.values, "--ctx-size", ParamValue::Integer(1));
        let message = state.apply_preset(0).unwrap();
        assert!(message.contains("full-VRAM"));
        let ctx = find_by_flag("--ctx-size").unwrap();
        assert_eq!(state.values.values[ctx], ParamValue::Integer(65_536));
        let fa = find_by_flag("--flash-attn").unwrap();
        assert_eq!(state.values.values[fa], ParamValue::On);
        let ctk = find_by_flag("--cache-type-k").unwrap();
        assert_eq!(state.values.values[ctk], ParamValue::Choice(3));
    }

    #[test]
    fn import_parses_known_flags_and_preserves_unknown() {
        let (binary, token_lines, model_path) =
            parse_cmd_block(&SAMPLE.lines().collect::<Vec<_>>()[6..19]).unwrap();
        assert!(binary.ends_with("${llama_server}"), "binary: {binary}");
        assert_eq!(model_path.as_deref(), Some("/models/alpha/model.gguf"));

        let mut state = ParamBuilderState::default();
        state.import(&binary, &token_lines, model_path.as_deref());
        let ctx = find_by_flag("--ctx-size").unwrap();
        assert_eq!(state.values.values[ctx], ParamValue::Integer(65_536));
        let fa = find_by_flag("--flash-attn").unwrap();
        assert_eq!(state.values.values[fa], ParamValue::On);
        let fit = find_by_flag("--fit").unwrap();
        assert_eq!(state.values.values[fit], ParamValue::On);
        let alias = find_by_flag("--alias").unwrap();
        assert_eq!(state.values.values[alias], ParamValue::Text("alpha".into()));
        let host = find_by_flag("--host").unwrap();
        assert_eq!(
            state.values.values[host],
            ParamValue::Text("0.0.0.0".into())
        );
        // Unknown flag preserved verbatim.
        assert!(state
            .values
            .custom
            .iter()
            .any(|line| line == "--lazy-mode on"));
        // --fit-target survived as a paired value.
        let fitt = find_by_flag("--fit-target").unwrap();
        assert_eq!(state.values.values[fitt], ParamValue::Text("512".into()));
    }

    #[test]
    fn parse_cmd_block_handles_one_line_invocation() {
        let lines: Vec<&str> = SAMPLE.lines().collect();
        let beta_start = lines
            .iter()
            .position(|line| line.contains("\"beta\""))
            .unwrap();
        let (binary, token_lines, model_path) = parse_cmd_block(&lines[beta_start..]).unwrap();
        assert_eq!(binary, "llama-server");
        assert_eq!(model_path.as_deref(), Some("/models/beta/beta.gguf"));
        // --port pair is dropped by import (save appends it).
        assert!(token_lines.iter().flatten().any(|token| token == "--port"));
    }

    #[test]
    fn replace_entry_cmd_preserves_surroundings() {
        let mut state = ParamBuilderState {
            binary: "llama-server".into(),
            model_path: Some("/models/beta/beta.gguf".into()),
            ..Default::default()
        };
        set(&mut state.values, "--ctx-size", ParamValue::Integer(4096));
        let body = state.cmd_body();

        let updated = replace_entry_cmd(SAMPLE, "beta", &body).unwrap();
        assert!(updated.contains("  \"beta\":\n    cmd: |\n"));
        assert!(updated.contains("      -m /models/beta/beta.gguf\n"));
        assert!(updated.contains("      --port ${PORT}\n"));
        assert!(updated.contains("--ctx-size 4096"));
        // Alpha untouched, trailing top-level structure intact.
        assert!(updated.contains("--lazy-mode on"));
        assert!(updated.contains(r#"- "GGML_CUDA_GRAPH_OPT=1""#));

        let reparsed = crate::swap_yaml::parse_entries(&updated);
        assert_eq!(reparsed.len(), 2);
        assert_eq!(reparsed[1].id, "beta");
        assert_eq!(
            reparsed[1].model_path.as_deref(),
            Some("/models/beta/beta.gguf")
        );
    }

    #[test]
    fn replace_entry_cmd_errors_for_unknown_or_blockless_entries() {
        let body = vec!["llama-server".to_owned()];
        assert!(replace_entry_cmd(SAMPLE, "missing", &body).is_err());
        let minimal = "models:\n  \"gamma\":\n    ttl: 60\n";
        assert!(replace_entry_cmd(minimal, "gamma", &body).is_err());
    }

    #[test]
    fn edit_key_handles_typing_and_escape() {
        let mut state = ParamBuilderState::default();
        state.start_edit_at(EditTarget::CustomNew);
        for character in "abc ".chars() {
            state.edit_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }
        state.edit_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        let status = state
            .edit_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .unwrap();
        assert!(status.is_empty());
        assert_eq!(state.values.custom, vec!["abc".to_owned()]);
        assert!(!state.editing());

        state.start_edit_at(EditTarget::CustomNew);
        let status = state
            .edit_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .unwrap();
        assert_eq!(status, "edit cancelled");
        assert_eq!(state.values.custom, vec!["abc".to_owned()]);
    }

    #[test]
    fn delete_custom_removes_and_clamps() {
        let mut state = ParamBuilderState::default();
        state.values.custom = vec!["--a".into(), "--b".into()];
        // Wrap to the last selectable row (AddCustom), then back one.
        state.move_selection(-1);
        state.move_selection(-1);
        assert!(matches!(state.selected_row(), ParamRow::Custom(1)));
        state.delete_custom().unwrap();
        assert_eq!(state.values.custom, vec!["--a".to_owned()]);
    }

    /// Round-trip every real entry: import its cmd block, regenerate it,
    /// re-import the regenerated block — the value state must be identical.
    /// Depends on the host's llama-swap.yaml, so it is ignored by default.
    #[test]
    #[ignore = "requires the host's llama-swap.yaml"]
    fn import_real_yaml_entries_round_trip() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("llama-swap.yaml");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let lines: Vec<&str> = text.lines().collect();
        for entry in crate::swap_yaml::parse_entries(&text) {
            let Some((start, end)) = crate::swap_yaml::entry_span(&lines, &entry.id) else {
                continue;
            };
            let Some((binary, token_lines, model_path)) = parse_cmd_block(&lines[start..end])
            else {
                continue;
            };
            let mut state = ParamBuilderState::default();
            state.import(&binary, &token_lines, model_path.as_deref());
            let body = state.cmd_body();

            let updated = replace_entry_cmd(&text, &entry.id, &body).unwrap();
            let lines2: Vec<&str> = updated.lines().collect();
            let (start2, end2) = crate::swap_yaml::entry_span(&lines2, &entry.id).unwrap();
            let Some((binary2, token_lines2, model2)) = parse_cmd_block(&lines2[start2..end2])
            else {
                panic!("entry {}: regenerated cmd block did not parse", entry.id);
            };
            assert_eq!(model2, model_path, "entry {}: -m changed", entry.id);
            let mut state2 = ParamBuilderState::default();
            state2.import(&binary2, &token_lines2, model2.as_deref());
            assert_eq!(
                state2.values, state.values,
                "entry {}: round trip diverged",
                entry.id
            );
            let _ = binary2; // macros may differ in form but must be preserved
            assert_eq!(state.binary, binary);
        }
    }
}
