//! Winamp-classic theme: green-on-black, dark-blue selection bars, amber accents.

use ratatui::style::{Color, Modifier, Style};

/// Playlist green — primary text.
pub const GREEN: Color = Color::Rgb(0x00, 0xFF, 0x00);
/// Dim green — borders and secondary text.
pub const DIM_GREEN: Color = Color::Rgb(0x00, 0xB4, 0x00);
/// Playlist selection background (classic dark blue).
pub const SELECTION_BG: Color = Color::Rgb(0x00, 0x00, 0xC6);
/// Amber — accents and status values.
pub const AMBER: Color = Color::Rgb(0xFF, 0xB0, 0x00);
/// Red — errors and destructive state.
pub const RED: Color = Color::Rgb(0xFF, 0x30, 0x30);

/// Default body text.
pub fn text() -> Style {
    Style::default().fg(GREEN)
}

/// Secondary text.
pub fn dim() -> Style {
    Style::default().fg(DIM_GREEN)
}

/// Titles and emphasized labels.
pub fn title() -> Style {
    Style::default().fg(GREEN).add_modifier(Modifier::BOLD)
}

/// Unfocused block borders.
pub fn border() -> Style {
    Style::default().fg(DIM_GREEN)
}

/// Focused block borders.
pub fn border_focused() -> Style {
    Style::default().fg(GREEN)
}

/// Selected row (playlist highlight).
pub fn selected() -> Style {
    Style::default()
        .bg(SELECTION_BG)
        .fg(Color::White)
        .add_modifier(Modifier::BOLD)
}

/// Inverted LED look — badges, footer bar, focused input fields.
pub fn badge() -> Style {
    Style::default()
        .fg(Color::Black)
        .bg(GREEN)
        .add_modifier(Modifier::BOLD)
}

/// Warnings.
pub fn warning() -> Style {
    Style::default().fg(AMBER)
}

/// Errors.
pub fn error() -> Style {
    Style::default().fg(RED)
}

/// Amber accent values.
pub fn accent() -> Style {
    Style::default().fg(AMBER)
}

/// Bordered block with themed border and title.
pub fn block<'a>() -> ratatui::widgets::Block<'a> {
    ratatui::widgets::Block::default()
        .border_style(border())
        .title_style(title())
}
