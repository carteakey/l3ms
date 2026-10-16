//! The workbench shell: a toolbar row, a sidebar with model and job sections,
//! one pane for the current view, a message line and a status bar.
//!
//! These modules only lay out and draw plain data; `app.rs` owns state and
//! keys. See `docs/workbench-design.md` and `docs/workbench-plan.md`.

pub mod shell;
pub mod sidebar;
pub mod statusbar;
pub mod toolbar;
