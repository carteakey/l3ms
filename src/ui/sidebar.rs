//! Sidebar: a Models section (profiles grouped by lane, with live router
//! state) and a Jobs section (newest first). `app.rs` builds the rows; this
//! module draws them and keeps the selected row in view.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState},
    Frame,
};

use crate::theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Models,
    Jobs,
}

impl Section {
    pub fn other(self) -> Self {
        match self {
            Self::Models => Self::Jobs,
            Self::Jobs => Self::Models,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelRow {
    /// A lane heading such as `language`.
    Lane { name: &'static str, style: Style },
    Model {
        /// Index into the catalog's profile list.
        catalog_index: usize,
        id: String,
        dot: &'static str,
        dot_style: Style,
        /// Short right-aligned note, e.g. `61 t/s` or `off`.
        note: String,
        /// Disabled entries are drawn dim.
        muted: bool,
        /// Second line, shown only under the selected row, e.g.
        /// `chat · llama.cpp master`.
        detail: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct JobRow {
    pub dot: &'static str,
    pub dot_style: Style,
    pub label: String,
    pub age: String,
}

pub struct SidebarModel<'a> {
    pub focused: bool,
    pub section: Section,
    pub models_collapsed: bool,
    pub jobs_collapsed: bool,
    pub models: &'a [ModelRow],
    pub selected_catalog_index: Option<usize>,
    pub filter: &'a str,
    pub filter_editing: bool,
    pub jobs: &'a [JobRow],
    pub selected_job: Option<usize>,
}

/// Job status → dot and colour. The status word stays available in the
/// Jobs view; the dot only summarises it.
pub fn job_dot(status: &str) -> (&'static str, Style) {
    match status {
        "running" | "queued" => ("●", theme::warning()),
        "done" | "succeeded" => ("✓", theme::title()),
        "failed" | "interrupted" => ("✗", theme::error()),
        "cancelled" => ("○", theme::dim()),
        _ => ("·", theme::dim()),
    }
}

/// Router state → dot and colour for a model row.
pub fn model_dot(state: Option<&str>) -> (&'static str, Style) {
    match state {
        Some("loaded") => ("●", theme::title()),
        Some("loading") => ("◐", theme::warning()),
        Some("unloaded") => ("○", theme::dim()),
        Some(_) => ("?", theme::dim()),
        None => ("·", theme::dim()),
    }
}

fn clip(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    let mut clipped: String = text.chars().take(width.saturating_sub(1)).collect();
    clipped.push('…');
    clipped
}

fn header(label: String, active: bool, collapsed: bool, focused: bool) -> Line<'static> {
    let marker = if collapsed { "▸" } else { "▾" };
    let style = if active && focused {
        theme::title()
    } else {
        theme::dim()
    };
    Line::from(Span::styled(format!("{marker} {label}"), style))
}

pub fn draw(frame: &mut Frame, area: Rect, model: &SidebarModel) {
    let border = if model.focused {
        theme::border_focused()
    } else {
        theme::border()
    };
    let block = Block::default()
        .borders(Borders::RIGHT)
        .border_style(border);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width < 4 {
        return;
    }
    let width = inner.width as usize;

    // Collapsed sections take one row; an expanded Jobs section takes up to a
    // third of the height so the model list stays the main thing.
    let jobs_height = if model.jobs_collapsed {
        1
    } else {
        let wanted = model.jobs.len().max(1) as u16 + 1;
        wanted.min((inner.height / 3).max(2))
    };
    let models_height = if model.models_collapsed {
        1
    } else {
        inner.height.saturating_sub(jobs_height)
    };
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(models_height),
            Constraint::Length(jobs_height),
            Constraint::Min(0),
        ])
        .split(inner);

    let model_count = model
        .models
        .iter()
        .filter(|row| matches!(row, ModelRow::Model { .. }))
        .count();
    let mut models_label = format!("Models {model_count}");
    if !model.filter.is_empty() || model.filter_editing {
        let cursor = if model.filter_editing { "▏" } else { "" };
        models_label = format!("Models /{}{cursor}", model.filter);
    }
    let models_active = model.section == Section::Models;
    frame.render_widget(
        header(
            models_label,
            models_active,
            model.models_collapsed,
            model.focused,
        ),
        Rect::new(areas[0].x, areas[0].y, areas[0].width, 1),
    );
    if !model.models_collapsed && areas[0].height > 1 {
        let list_area = Rect::new(
            areas[0].x,
            areas[0].y + 1,
            areas[0].width,
            areas[0].height - 1,
        );
        let mut selected = None;
        let items: Vec<ListItem> = model
            .models
            .iter()
            .enumerate()
            .map(|(index, row)| match row {
                ModelRow::Lane { name, style } => {
                    ListItem::new(Line::from(Span::styled(format!(" {name}"), *style)))
                }
                ModelRow::Model {
                    catalog_index,
                    id,
                    dot,
                    dot_style,
                    note,
                    muted,
                    detail,
                } => {
                    let is_selected = Some(*catalog_index) == model.selected_catalog_index;
                    if is_selected {
                        selected = Some(index);
                    }
                    // One line per model: dot, id, then the note right-aligned.
                    let note_width = if note.is_empty() {
                        0
                    } else {
                        note.chars().count() + 1
                    };
                    let id_width = width.saturating_sub(3 + note_width);
                    let shown = clip(id, id_width);
                    let pad = id_width.saturating_sub(shown.chars().count());
                    let id_style = if *muted { theme::dim() } else { theme::text() };
                    let mut lines = vec![Line::from(vec![
                        Span::styled(format!(" {dot} "), *dot_style),
                        Span::styled(shown, id_style),
                        Span::raw(" ".repeat(pad)),
                        Span::styled(
                            if note.is_empty() {
                                String::new()
                            } else {
                                format!(" {note}")
                            },
                            theme::dim(),
                        ),
                    ])];
                    if is_selected && !detail.is_empty() {
                        lines.push(Line::from(Span::styled(
                            format!("   {}", clip(detail, width.saturating_sub(3))),
                            theme::dim(),
                        )));
                    }
                    ListItem::new(lines)
                }
            })
            .collect();
        let highlight = if models_active && model.focused {
            theme::selected()
        } else if models_active {
            theme::title()
        } else {
            Style::default()
        };
        let list = if items.is_empty() {
            List::new(vec![ListItem::new(Span::styled(
                "  no models match",
                theme::dim(),
            ))])
        } else {
            List::new(items)
        };
        let mut state = ListState::default().with_selected(selected);
        frame.render_stateful_widget(list.highlight_style(highlight), list_area, &mut state);
    }

    let jobs_active = model.section == Section::Jobs;
    frame.render_widget(
        header(
            format!("Jobs {}", model.jobs.len()),
            jobs_active,
            model.jobs_collapsed,
            model.focused,
        ),
        Rect::new(areas[1].x, areas[1].y, areas[1].width, 1),
    );
    if !model.jobs_collapsed && areas[1].height > 1 {
        let list_area = Rect::new(
            areas[1].x,
            areas[1].y + 1,
            areas[1].width,
            areas[1].height - 1,
        );
        let items: Vec<ListItem> = if model.jobs.is_empty() {
            vec![ListItem::new(Span::styled("  no jobs yet", theme::dim()))]
        } else {
            model
                .jobs
                .iter()
                .map(|job| {
                    let age_width = job.age.chars().count() + 1;
                    let label_width = width.saturating_sub(3 + age_width);
                    let label = clip(&job.label, label_width);
                    let pad = label_width.saturating_sub(label.chars().count());
                    ListItem::new(Line::from(vec![
                        Span::styled(format!(" {} ", job.dot), job.dot_style),
                        Span::styled(label, theme::text()),
                        Span::raw(" ".repeat(pad + 1)),
                        Span::styled(job.age.clone(), theme::dim()),
                    ]))
                })
                .collect()
        };
        let highlight = if jobs_active && model.focused {
            theme::selected()
        } else {
            Style::default()
        };
        let selected = if jobs_active && !model.jobs.is_empty() {
            model.selected_job
        } else {
            None
        };
        let mut state = ListState::default().with_selected(selected);
        frame.render_stateful_widget(
            List::new(items).highlight_style(highlight),
            list_area,
            &mut state,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    fn rows() -> Vec<ModelRow> {
        let mut rows = vec![ModelRow::Lane {
            name: "language",
            style: theme::lane_language(),
        }];
        for index in 0..30 {
            rows.push(ModelRow::Model {
                catalog_index: index,
                id: format!("model-{index}-with-a-rather-long-identifier"),
                dot: "○",
                dot_style: theme::dim(),
                note: if index % 2 == 0 {
                    "61 t/s".into()
                } else {
                    String::new()
                },
                muted: false,
                detail: "chat · llama.cpp master".into(),
            });
        }
        rows
    }

    fn screen(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        let mut text = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                text.push_str(buffer[(x, y)].symbol());
            }
            text.push('\n');
        }
        text
    }

    #[test]
    fn selected_model_stays_visible_and_long_ids_are_clipped() {
        let rows = rows();
        let jobs = vec![JobRow {
            dot: "✓",
            dot_style: theme::title(),
            label: "model-load · plat".into(),
            age: "5m".into(),
        }];
        let model = SidebarModel {
            focused: true,
            section: Section::Models,
            models_collapsed: false,
            jobs_collapsed: false,
            models: &rows,
            selected_catalog_index: Some(25),
            filter: "",
            filter_editing: false,
            jobs: &jobs,
            selected_job: Some(0),
        };
        let mut terminal = Terminal::new(TestBackend::new(32, 18)).unwrap();
        terminal
            .draw(|frame| draw(frame, frame.area(), &model))
            .unwrap();
        let text = screen(&terminal);
        assert!(text.contains("model-25-"), "{text}");
        assert!(text.contains('…'), "{text}");
        // Only the selected row carries the detail line; rates stay visible.
        assert_eq!(text.matches("chat · llama.cpp master").count(), 1, "{text}");
        assert!(text.contains("61 t/s"), "{text}");
        assert!(text.contains("Jobs 1"), "{text}");
        assert!(text.contains("model-load · plat"), "{text}");
    }

    #[test]
    fn collapsed_sections_take_one_row() {
        let rows = rows();
        let model = SidebarModel {
            focused: false,
            section: Section::Jobs,
            models_collapsed: true,
            jobs_collapsed: false,
            models: &rows,
            selected_catalog_index: None,
            filter: "plat",
            filter_editing: false,
            jobs: &[],
            selected_job: None,
        };
        let mut terminal = Terminal::new(TestBackend::new(30, 10)).unwrap();
        terminal
            .draw(|frame| draw(frame, frame.area(), &model))
            .unwrap();
        let text = screen(&terminal);
        assert!(text.contains("▸ Models /plat"), "{text}");
        assert!(text.contains("▾ Jobs 0"), "{text}");
        assert!(text.contains("no jobs yet"), "{text}");
    }
}
