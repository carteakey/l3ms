//! Shared control-plane discovery and gated execution primitives.
//! Native execution is not yet connected to public CLI/UI actions.

pub mod catalog;
pub mod comfy;
pub mod decision;
pub mod inference;
pub mod ipc;
pub mod language;
pub mod native_commands;
pub mod native_paths;
pub mod ownership;
pub mod process;
pub mod requests;
pub mod routes;
pub mod runtime;
