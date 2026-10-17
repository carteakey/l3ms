//! Model view: what the selected profile is, how it is served, and how it
//! benched. `app.rs` gathers the facts; this module reads serving flags out
//! of a llama-swap `cmd` and draws the view.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
    Frame,
};

use crate::theme;

/// Flags worth showing, with their label. Long and short spellings share a
/// label; the first occurrence wins.
const FLAGS: &[(&[&str], &str)] = &[
    (&["-c", "--ctx-size"], "ctx"),
    (&["-ngl", "--n-gpu-layers", "--gpu-layers"], "ngl"),
    (&["-ncmoe", "--n-cpu-moe"], "ncmoe"),
    (&["-ctk", "--cache-type-k"], "kv-k"),
    (&["-ctv", "--cache-type-v"], "kv-v"),
    (&["-fa", "--flash-attn"], "fa"),
    (&["-np", "--parallel"], "parallel"),
    (&["-md", "--model-draft"], "draft"),
    (&["--spec-draft-n-max", "--draft-max"], "draft-max"),
    (&["--mmproj"], "mmproj"),
    (&["--engine"], "engine"),
    (&["--config"], "config"),
    (&["-ot", "--override-tensor"], "override"),
];

/// The server binary or macro (`${qwen38_master_server}` → `qwen38_master`)
/// and the notable flags of a llama-swap `cmd`.
pub fn cmd_facts(cmd: &str) -> (Option<String>, Vec<(&'static str, String)>) {
    let tokens = shell_words::split(cmd)
        .unwrap_or_else(|_| cmd.split_whitespace().map(str::to_owned).collect());
    // The server is the first `${…_server}` macro or known binary; flags
    // before it (such as `taskset -c`) belong to wrappers.
    let server_at = tokens.iter().position(|token| {
        token
            .strip_prefix("${")
            .and_then(|rest| rest.strip_suffix('}'))
            .is_some_and(|name| name.ends_with("_server"))
            || matches!(
                token.rsplit('/').next(),
                Some("llama-server" | "audiocpp-server")
            )
    });
    let server = server_at.map(|at| {
        let token = &tokens[at];
        token
            .strip_prefix("${")
            .and_then(|rest| rest.strip_suffix('}'))
            .and_then(|name| name.strip_suffix("_server"))
            .map(str::to_owned)
            .unwrap_or_else(|| token.rsplit('/').next().unwrap_or(token).to_owned())
    });
    let mut facts: Vec<(&'static str, String)> = Vec::new();
    let mut index = server_at.map_or(0, |at| at + 1);
    while index < tokens.len() {
        let token = tokens[index].as_str();
        let (flag, inline) = match token.split_once('=') {
            Some((flag, value)) if flag.starts_with('-') => (flag, Some(value.to_owned())),
            _ => (token, None),
        };
        if let Some((_, label)) = FLAGS.iter().find(|(names, _)| names.contains(&flag)) {
            let value = inline.or_else(|| {
                tokens
                    .get(index + 1)
                    .filter(|next| !next.starts_with('-') || next.parse::<i64>().is_ok())
                    .cloned()
            });
            if let Some(value) = value {
                if !facts.iter().any(|(existing, _)| existing == label) {
                    let value = value
                        .strip_prefix("${")
                        .and_then(|rest| rest.strip_suffix('}'))
                        .map(str::to_owned)
                        .unwrap_or(value);
                    facts.push((label, value));
                }
                index += 1;
            }
        }
        index += 1;
    }
    (server, facts)
}

/// A readable name for a macro value: the `vendor/<checkout>` directory for
/// binaries under `vendor/`, otherwise the file name.
pub fn macro_display(value: &str) -> String {
    if let Some((_, rest)) = value.split_once("/vendor/") {
        if let Some(checkout) = rest.split('/').next().filter(|part| !part.is_empty()) {
            let file = value.rsplit('/').next().unwrap_or_default();
            return if file.ends_with(".json") {
                format!("{checkout}/{file}")
            } else {
                checkout.to_owned()
            };
        }
    }
    value.rsplit('/').next().unwrap_or(value).to_owned()
}

/// Human size such as `46.1 GB`.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// `3600` → `1 h`, `300` → `5 min`.
pub fn human_ttl(seconds: u64) -> String {
    match seconds {
        0 => "never unloads".into(),
        s if s % 3600 == 0 => format!("{} h", s / 3600),
        s if s % 60 == 0 => format!("{} min", s / 60),
        s => format!("{s} s"),
    }
}

/// One bench run as shown in the view.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchRow {
    pub date: String,
    pub tg: Option<f64>,
    pub pp: Option<f64>,
    pub setup: String,
}

pub struct ModelViewData<'a> {
    pub id: &'a str,
    pub lane_style: Style,
    pub state: Option<&'a str>,
    /// Whether router state applies (language entries and media profiles
    /// served by an engine). Others show no state at all.
    pub routed: bool,
    pub title: Option<&'a str>,
    pub description: Option<&'a str>,
    /// Label/value rows: serving, model file, flags, ttl, tasks, runtime.
    pub facts: Vec<(&'static str, String)>,
    pub bench: &'a [BenchRow],
    /// Keys that apply to this profile, e.g. `Enter chat · L load`.
    pub keys: String,
    pub focused: bool,
}

fn state_span(state: Option<&str>, routed: bool) -> Span<'static> {
    if !routed {
        return Span::raw("");
    }
    match state {
        Some("loaded") => Span::styled("● loaded", theme::title()),
        Some("loading") => Span::styled("◐ loading", theme::warning()),
        Some("unloaded") => Span::styled("○ unloaded", theme::dim()),
        Some(other) => Span::styled(other.to_owned(), theme::dim()),
        None => Span::styled("· not listed by the router", theme::dim()),
    }
}

/// Draw the facts block in `area`; returns nothing. The caller draws any
/// task or result panel below it.
pub fn draw(frame: &mut Frame, area: Rect, data: &ModelViewData) {
    let border = if data.focused {
        theme::border_focused()
    } else {
        data.lane_style
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border)
        .title(Line::from(vec![
            Span::styled(format!(" {} ", data.id), theme::title()),
            state_span(data.state, data.routed),
            Span::raw(" "),
        ]));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let mut lines = Vec::new();
    if let Some(title) = data.title {
        lines.push(Line::from(Span::styled(title.to_owned(), theme::text())));
    }
    if let Some(description) = data.description {
        lines.push(Line::from(Span::styled(
            description.to_owned(),
            theme::dim(),
        )));
    }
    if !lines.is_empty() {
        lines.push(Line::from(""));
    }
    for (label, value) in &data.facts {
        lines.push(Line::from(vec![
            Span::styled(format!("{label:<9} "), theme::dim()),
            Span::styled(value.clone(), theme::text()),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(data.keys.clone(), theme::accent())));

    // Bench history takes the bottom when there is any and room for it.
    let bench_height = if data.bench.is_empty() {
        0
    } else {
        (data.bench.len() as u16 + 2).min(inner.height / 2)
    };
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(bench_height)])
        .split(inner);
    frame.render_widget(
        Paragraph::new(lines)
            .style(theme::text())
            .wrap(Wrap { trim: false }),
        parts[0],
    );
    if bench_height > 0 {
        let rows = data.bench.iter().map(|run| {
            let rate = |value: Option<f64>| value.map_or("—".to_owned(), |v| format!("{v:.1}"));
            Row::new(vec![
                Cell::from(run.date.clone()),
                Cell::from(rate(run.tg)).style(theme::title()),
                Cell::from(rate(run.pp)),
                Cell::from(run.setup.clone()).style(theme::dim()),
            ])
        });
        let table = Table::new(
            rows,
            [
                Constraint::Length(11),
                Constraint::Length(8),
                Constraint::Length(9),
                Constraint::Min(10),
            ],
        )
        .header(Row::new(["bench", "tg t/s", "pp t/s", "setup"]).style(theme::dim()))
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(theme::border()),
        );
        frame.render_widget(table, parts[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facts_read_macros_and_both_flag_spellings() {
        let cmd = "taskset -c ${cpu_range} ${qwen38_master_server}\n-m /m/x.gguf -c 98304 -ngl 99 --n-cpu-moe 45\n-ctk q8_0 -ctv q4_0 -fa on --port ${PORT}\n";
        let (server, facts) = cmd_facts(cmd);
        assert_eq!(server.as_deref(), Some("qwen38_master"));
        let get = |label: &str| {
            facts
                .iter()
                .find(|(l, _)| *l == label)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get("ctx"), Some("98304"));
        assert_eq!(get("ngl"), Some("99"));
        assert_eq!(get("ncmoe"), Some("45"));
        assert_eq!(get("kv-k"), Some("q8_0"));
        assert_eq!(get("kv-v"), Some("q4_0"));
        assert_eq!(get("fa"), Some("on"));
        // `-c` in `taskset -c` precedes the server; it is the cpu range macro.
        assert_eq!(facts.iter().filter(|(l, _)| *l == "ctx").count(), 1);
    }

    #[test]
    fn strata_and_engine_commands_yield_their_engine_and_config() {
        let (server, facts) = cmd_facts(
            "taskset -c ${cpu_range} ${strata_python}\n${strata_server}\n--engine strata\n--config ${strata_iq3_config}\n--fit-max-tokens\n",
        );
        assert_eq!(server.as_deref(), Some("strata"));
        assert!(facts.contains(&("engine", "strata".into())));
        assert!(facts.contains(&("config", "strata_iq3_config".into())));
        let (server, _) = cmd_facts(
            "/usr/bin/bash -c 'cd x && exec python /v/ComfyUI/main.py --listen 127.0.0.1'",
        );
        assert_eq!(server, None);
    }

    #[test]
    fn macro_values_show_their_vendor_checkout() {
        assert_eq!(
            macro_display("${env.L3MS_ROOT}/vendor/llama.cpp-exp-stack-v2/build/bin/llama-server"),
            "llama.cpp-exp-stack-v2"
        );
        assert_eq!(
            macro_display("${env.L3MS_ROOT}/vendor/strata-v0.1.44.1/strata-iq3_xxs.json"),
            "strata-v0.1.44.1/strata-iq3_xxs.json"
        );
        assert_eq!(macro_display("/usr/bin/llama-server"), "llama-server");
    }

    #[test]
    fn human_units() {
        assert_eq!(human_bytes(46_100_000_000), "46.1 GB");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_ttl(3600), "1 h");
        assert_eq!(human_ttl(300), "5 min");
        assert_eq!(human_ttl(45), "45 s");
    }
}
