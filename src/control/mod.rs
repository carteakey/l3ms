//! Shared control-plane discovery and gated execution primitives.
//! Opt-in CPU-local execution is connected to explicit CLI/UI actions; general
//! managed-engine and GPU/media admission remain gated.

pub mod catalog;
#[cfg(target_os = "linux")]
pub mod cgroup_release;
#[cfg(target_os = "linux")]
pub mod client;
pub mod comfy;
#[cfg(target_os = "linux")]
pub mod coordinator_launch;
#[cfg(target_os = "linux")]
pub mod coordinator_workbench;
#[cfg(target_os = "linux")]
pub mod daemon;
pub mod decision;
#[cfg(target_os = "linux")]
pub mod decision_worker;
#[cfg(target_os = "linux")]
pub mod engine_transport;
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
#[cfg(target_os = "linux")]
pub(crate) mod native_http;
pub mod native_paths;
pub mod ownership;
pub mod process;
pub mod requests;
pub mod result_buffer;
pub mod result_consumer;
pub mod routes;
pub mod runtime;
#[cfg(target_os = "linux")]
pub mod service_intents;
#[cfg(target_os = "linux")]
pub mod service_owner;
pub mod submission_file;
