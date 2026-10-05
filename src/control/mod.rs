//! Shared control-plane discovery and gated execution primitives.
//! Native execution is not yet connected to public CLI/UI actions.

pub mod catalog;
#[cfg(target_os = "linux")]
pub mod cgroup_release;
#[cfg(target_os = "linux")]
pub mod client;
pub mod comfy;
#[cfg(target_os = "linux")]
pub mod daemon;
pub mod decision;
#[cfg(target_os = "linux")]
pub mod gpu_probe;
pub mod inference;
pub mod ipc;
pub mod language;
#[cfg(target_os = "linux")]
pub mod local_ipc;
#[cfg(target_os = "linux")]
pub mod managed_service;
#[cfg(target_os = "linux")]
pub mod media_execution;
pub mod native_commands;
pub mod native_paths;
pub mod ownership;
pub mod process;
pub mod requests;
pub mod routes;
pub mod runtime;
#[cfg(target_os = "linux")]
pub mod service_intents;
#[cfg(target_os = "linux")]
pub mod service_owner;
