//! Toolbar: the workbench utilities as a row of buttons.
//!
//! A button is a `commands.rs` entry plus a glyph; its key and enabled state
//! come from the registry, so the toolbar, palette and help never disagree.
//! Disabled entries (utilities not built yet) render dim.

use ratatui::text::{Line, Span};

use crate::commands::{command, CommandId};
use crate::theme;

pub struct Button {
    pub id: CommandId,
    pub glyph: &'static str,
    /// The sidebar key that triggers the button.
    pub key: char,
}

pub const BUTTONS: &[Button] = &[
    Button {
        id: CommandId::ReloadRouter,
        glyph: "⟳",
        key: 'R',
    },
    Button {
        id: CommandId::OpenBench,
        glyph: "▷",
        key: 'b',
    },
    Button {
        id: CommandId::Tune,
        glyph: "⚙",
        key: 't',
    },
    Button {
        id: CommandId::OpenDownload,
        glyph: "⇩",
        key: 'd',
    },
    Button {
        id: CommandId::OpenSystem,
        glyph: "♥",
        key: 'h',
    },
];

/// The command bound to a toolbar key, if the key is a button's key.
pub fn command_for_key(key: char) -> Option<CommandId> {
    BUTTONS
        .iter()
        .find(|button| button.key == key)
        .map(|button| button.id)
}

/// Brand mark, the active view's name and the buttons, clipped to `width`.
/// Buttons drop their labels before they drop entirely.
pub fn toolbar_line(view: &str, width: u16) -> Line<'static> {
    let width = width as usize;
    let mut spans = vec![
        Span::styled(" L3MS ", theme::badge()),
        Span::styled(format!(" {view} "), theme::title()),
    ];
    let head: usize = spans.iter().map(|span| span.width()).sum();
    let buttons = |labels: bool| -> Vec<Span<'static>> {
        BUTTONS
            .iter()
            .filter_map(|button| {
                let spec = command(button.id)?;
                let text = if labels {
                    format!(
                        " {} {} {} ",
                        button.glyph,
                        spec.label.to_lowercase(),
                        button.key
                    )
                } else {
                    format!(" {}{} ", button.glyph, button.key)
                };
                let style = if spec.is_visible() {
                    theme::text()
                } else {
                    theme::dim()
                };
                Some(Span::styled(text, style))
            })
            .collect()
    };
    for labels in [true, false] {
        let row = buttons(labels);
        let row_width: usize = row.iter().map(|span| span.width()).sum();
        if head + row_width <= width {
            spans.extend(row);
            return Line::from(spans);
        }
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_button_is_a_registered_command_with_a_unique_key() {
        let mut keys = std::collections::HashSet::new();
        for button in BUTTONS {
            assert!(command(button.id).is_some(), "{}", button.id);
            assert!(keys.insert(button.key), "duplicate key {}", button.key);
        }
        assert_eq!(command_for_key('b'), Some(CommandId::OpenBench));
        assert_eq!(command_for_key('x'), None);
    }

    #[test]
    fn toolbar_fits_and_shortens_labels_before_dropping_buttons() {
        let text = |line: Line| -> String {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect()
        };
        let wide = toolbar_line("Model", 120);
        assert!(wide.width() <= 120);
        assert!(text(wide).contains("reload router R"));
        let narrow = toolbar_line("Model", 60);
        assert!(narrow.width() <= 60);
        assert!(text(narrow).contains("⟳R"));
    }
}
