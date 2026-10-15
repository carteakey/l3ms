//! Shared native discovery and explicit CLI/UI execution primitives.
//! CUDA-native execution requires an explicit operator-owned GPU window;
//! resident llama-swap pause/restore is opt-in. Live acceptance remains gated.

pub mod catalog;
#[cfg(target_os = "linux")]
pub mod client;
pub mod comfy;
#[cfg(target_os = "linux")]
pub mod coordinator_workbench;
pub mod cuda_device;
pub mod decision;
#[cfg(target_os = "linux")]
pub(crate) mod direct_run;
#[cfg(target_os = "linux")]
pub(crate) mod foreground_shutdown;
#[cfg(target_os = "linux")]
pub mod gpu_probe;
pub mod ipc;
pub mod language;
#[cfg(target_os = "linux")]
pub mod local_ipc;
#[cfg(target_os = "linux")]
pub mod media_execution;
pub mod native_commands;
pub mod native_paths;
pub mod ownership;
pub mod process;
pub mod requests;
#[cfg(target_os = "linux")]
pub(crate) mod resident_unit;
pub mod result_buffer;
pub mod result_consumer;
pub mod routes;
pub mod runtime;
pub mod task_form;
