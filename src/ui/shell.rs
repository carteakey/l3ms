//! Frame layout for the shell.

use ratatui::layout::Rect;

/// Sidebar width when it sits beside the pane.
pub const SIDEBAR_WIDTH: u16 = 32;
/// Narrower terminals show the sidebar or the pane, whichever has focus.
pub const SPLIT_MIN_WIDTH: u16 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellAreas {
    pub toolbar: Rect,
    pub sidebar: Option<Rect>,
    pub pane: Option<Rect>,
    pub message: Rect,
    pub status: Rect,
}

/// Split the frame. The toolbar is the top row, the message line and status
/// bar the bottom two; the body holds the sidebar and the pane side by side,
/// or only the focused one when the terminal is narrower than
/// [`SPLIT_MIN_WIDTH`].
pub fn layout(area: Rect, sidebar_focused: bool) -> ShellAreas {
    let row = |y: u16| Rect::new(area.x, y, area.width, 1.min(area.height));
    let toolbar = row(area.y);
    let status = row(area.bottom().saturating_sub(1));
    let message = row(area.bottom().saturating_sub(2));
    let body_height = area.height.saturating_sub(3);
    let body = Rect::new(area.x, area.y + 1.min(area.height), area.width, body_height);
    let (sidebar, pane) = if body.height == 0 {
        (None, None)
    } else if area.width >= SPLIT_MIN_WIDTH {
        let sidebar = Rect::new(body.x, body.y, SIDEBAR_WIDTH, body.height);
        let pane = Rect::new(
            body.x + SIDEBAR_WIDTH,
            body.y,
            body.width - SIDEBAR_WIDTH,
            body.height,
        );
        (Some(sidebar), Some(pane))
    } else if sidebar_focused {
        (Some(body), None)
    } else {
        (None, Some(body))
    };
    ShellAreas {
        toolbar,
        sidebar,
        pane,
        message,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_terminals_show_sidebar_and_pane_together() {
        let areas = layout(Rect::new(0, 0, 120, 30), false);
        let sidebar = areas.sidebar.unwrap();
        let pane = areas.pane.unwrap();
        assert_eq!(sidebar.width, SIDEBAR_WIDTH);
        assert_eq!(pane.x, SIDEBAR_WIDTH);
        assert_eq!(sidebar.width + pane.width, 120);
        assert_eq!(sidebar.height, 27);
        assert_eq!(areas.toolbar.y, 0);
        assert_eq!(areas.message.y, 28);
        assert_eq!(areas.status.y, 29);
    }

    #[test]
    fn narrow_terminals_show_only_the_focused_side() {
        let sidebar_only = layout(Rect::new(0, 0, 80, 24), true);
        assert_eq!(sidebar_only.sidebar.unwrap().width, 80);
        assert!(sidebar_only.pane.is_none());
        let pane_only = layout(Rect::new(0, 0, 60, 18), false);
        assert!(pane_only.sidebar.is_none());
        assert_eq!(pane_only.pane.unwrap().height, 15);
    }

    #[test]
    fn tiny_terminals_do_not_underflow() {
        let areas = layout(Rect::new(0, 0, 10, 2), true);
        assert!(areas.sidebar.is_none() && areas.pane.is_none());
    }
}
