//! Signal / 03 theme: ink-blue graphite surfaces, phosphor-mint focus, and one
//! accent per lane (language = phosphor, media = mineral cyan, decisions = amber).
//! See `docs/interface-direction.md`. Accents mark location, never runtime state;
//! state always carries a text label as well.

use ratatui::style::{Color, Modifier, Style};

/// Base surface.
pub const BG: Color = Color::Rgb(0x0B, 0x0F, 0x14);
/// Primary text.
pub const TEXT: Color = Color::Rgb(0xD4, 0xDE, 0xD8);
/// Secondary text and unfocused borders; stays at 4.5:1 on `BG`.
pub const DIM: Color = Color::Rgb(0x8A, 0x98, 0x98);
/// Phosphor mint: focus, selection and the language lane.
pub const PHOSPHOR: Color = Color::Rgb(0x7E, 0xE7, 0xAA);
/// Mineral cyan: the media lane.
pub const CYAN: Color = Color::Rgb(0x58, 0xC4, 0xD0);
/// Amber: the decisions lane, warnings and status values.
pub const AMBER: Color = Color::Rgb(0xE8, 0xB0, 0x48);
/// Errors and destructive state.
pub const RED: Color = Color::Rgb(0xF0, 0x70, 0x64);
/// Selected-row background.
pub const SELECTION_BG: Color = Color::Rgb(0x1C, 0x34, 0x48);
/// Selected-row text.
pub const SELECTION_FG: Color = Color::Rgb(0xF0, 0xF6, 0xF2);

/// Fills the frame so the palette does not depend on the terminal theme.
pub fn surface() -> Style {
    Style::default().fg(TEXT).bg(BG)
}

/// Default body text.
pub fn text() -> Style {
    Style::default().fg(TEXT)
}

/// Secondary text.
pub fn dim() -> Style {
    Style::default().fg(DIM)
}

/// Titles and emphasized labels.
pub fn title() -> Style {
    Style::default().fg(PHOSPHOR).add_modifier(Modifier::BOLD)
}

/// Unfocused block borders.
pub fn border() -> Style {
    Style::default().fg(DIM)
}

/// Focused block borders.
pub fn border_focused() -> Style {
    Style::default().fg(PHOSPHOR)
}

/// Selected row.
pub fn selected() -> Style {
    Style::default()
        .bg(SELECTION_BG)
        .fg(SELECTION_FG)
        .add_modifier(Modifier::BOLD)
}

/// Inverted badge: brand mark, footer bar, focused input fields.
pub fn badge() -> Style {
    Style::default()
        .fg(BG)
        .bg(PHOSPHOR)
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

/// Accent values.
pub fn accent() -> Style {
    Style::default().fg(AMBER)
}

/// Lane signature for the language lane.
pub fn lane_language() -> Style {
    Style::default().fg(PHOSPHOR)
}

/// Lane signature for the media lane.
pub fn lane_media() -> Style {
    Style::default().fg(CYAN)
}

/// Lane signature for the decisions lane.
pub fn lane_decisions() -> Style {
    Style::default().fg(AMBER)
}

/// Bordered block with themed border and title.
pub fn block<'a>() -> ratatui::widgets::Block<'a> {
    ratatui::widgets::Block::default()
        .border_style(border())
        .title_style(title())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(value: u8) -> f64 {
        let v = f64::from(value) / 255.0;
        if v <= 0.03928 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    fn luminance(color: Color) -> f64 {
        let Color::Rgb(r, g, b) = color else {
            panic!("theme colors are RGB")
        };
        0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    }

    fn contrast(a: Color, b: Color) -> f64 {
        let (hi, lo) = (
            luminance(a).max(luminance(b)),
            luminance(a).min(luminance(b)),
        );
        (hi + 0.05) / (lo + 0.05)
    }

    #[test]
    fn text_and_accents_meet_four_point_five_to_one_on_the_surface() {
        for (name, color) in [
            ("text", TEXT),
            ("dim", DIM),
            ("phosphor", PHOSPHOR),
            ("cyan", CYAN),
            ("amber", AMBER),
            ("red", RED),
        ] {
            assert!(contrast(color, BG) >= 4.5, "{name} contrast too low");
        }
    }

    #[test]
    fn selection_and_badge_text_are_readable() {
        assert!(contrast(SELECTION_FG, SELECTION_BG) >= 4.5);
        assert!(contrast(BG, PHOSPHOR) >= 4.5);
        assert!(contrast(PHOSPHOR, SELECTION_BG) >= 4.5);
    }
}
