//! Direct, bounded execution for explicitly supported native work.
//!
//! This module accepts explicitly selected CPU/CUDA-local Nagi/Laya and hosted
//! Jev decisions, Breeze speech, embeddings and Comfy workflows through
//! existing adapters. GPU windows only check idleness for scripts that bypass
//! the router. They never install runtimes, select a
//! fallback, or retry inference. Native payloads stay out of errors.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::catalog::{Catalog, CatalogProfile, Device, Lane, NativeProfileSource};
use super::comfy::{
    ComfyAdapter, ComfyErrorKind, ComfyFailure, ComfyJobState, ComfyOutputRoot, NativeComfyResult,
};
use super::cuda_device::CudaDevice;
use super::decision::{
    execute_accepted_cpu_decision, execute_accepted_cuda_decision,
    execute_accepted_hosted_decision, DecisionExecutionError, NativeDecisionResult,
};
use super::desktop_gpu::DesktopGpuBaseline;
use super::gpu_probe::{GpuProbeConfig, NvidiaSmiGpuProbe};
use super::language::{DispatchDisposition, LanguageAdapter, LanguageErrorKind, LanguageFailure};
use super::media_execution::{
    execute_breeze_speech_on_cpu, execute_heartmula_music_on_device, is_supported_cpu_breeze_route,
    MediaExecutionErrorKind, MediaExecutionFailure, MediaExecutionResult,
};
use super::ownership::CoordinatorLock;
use super::process::{ProcessFailureDisposition, ProcessFailureKind};
use super::requests::{
    EmbeddingPayload, InferenceRequest, InferenceResult, MediaInput, SpeechMode,
};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::{NeverCancel, OperationContext, RecoveryPolicy};

const PROCESS_RECOVERY_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of one direct native operation. The payloads intentionally do not
/// implement `Debug`: native answers and filesystem paths are not log data.
pub enum DirectRunResult {
    Decision(NativeDecisionResult),
    Speech(MediaExecutionResult),
    ScriptMedia(MediaExecutionResult),
    Embeddings(EmbeddingPayload),
    Comfy(NativeComfyResult),
}

/// Explicit endpoint connection for one direct lane-native request.
pub enum DirectConnection<'a> {
    Language(&'a LanguageAdapter),
    /// Exact child device selection only. The caller must establish exclusive
    /// GPU admission before dispatch and separately prove release afterward.
    Cuda(&'a CudaDevice),
    Comfy {
        adapter: &'a ComfyAdapter,
        native_id: &'a str,
    },
}

/// Fixed failure categories with no request body, path, runner output, or OS
/// error retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectRunErrorKind {
    InvalidRequest,
    CatalogUnavailable,
    UnsupportedRoute,
    Cancelled,
    DeadlineExceeded,
    ExecutionFailed,
    InvalidOutput,
    OwnershipUncertain,
    RemoteCompletionUncertain,
    GpuConfiguration,
    GpuUnavailable,
    GpuReleaseUncertain,
    GpuRecoveryRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectRunDisposition {
    Process(ProcessFailureDisposition),
    Remote(DispatchDisposition),
    RemoteTerminal { cancellation_confirmed: bool },
}

/// A redacted failure plus whether local ownership or remote completion is uncertain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectRunError {
    kind: DirectRunErrorKind,
    disposition: DirectRunDisposition,
}

impl DirectRunError {
    fn new(kind: DirectRunErrorKind, disposition: ProcessFailureDisposition) -> Self {
        Self {
            kind,
            disposition: DirectRunDisposition::Process(disposition),
        }
    }

    fn remote(kind: DirectRunErrorKind, disposition: DispatchDisposition) -> Self {
        let kind = if disposition == DispatchDisposition::Uncertain {
            DirectRunErrorKind::RemoteCompletionUncertain
        } else {
            kind
        };
        Self {
            kind,
            disposition: DirectRunDisposition::Remote(disposition),
        }
    }

    fn remote_terminal(kind: DirectRunErrorKind, cancellation_confirmed: bool) -> Self {
        Self {
            kind,
            disposition: DirectRunDisposition::RemoteTerminal {
                cancellation_confirmed,
            },
        }
    }

    #[cfg(test)]
    pub fn kind(self) -> DirectRunErrorKind {
        self.kind
    }

    #[cfg(test)]
    fn disposition(self) -> DirectRunDisposition {
        self.disposition
    }

    /// True only when no child was spawned, the owned child group was reaped,
    /// a remote request was never dispatched, or Comfy reported Cancelled.
    pub fn cancellation_confirmed(self) -> bool {
        self.kind == DirectRunErrorKind::Cancelled
            && matches!(
                self.disposition,
                DirectRunDisposition::Process(
                    ProcessFailureDisposition::NoChildSpawned
                        | ProcessFailureDisposition::ChildGroupReaped,
                ) | DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
                    | DirectRunDisposition::RemoteTerminal {
                        cancellation_confirmed: true,
                    }
            )
    }

    pub fn is_remote(self) -> bool {
        matches!(
            self.disposition,
            DirectRunDisposition::Remote(_) | DirectRunDisposition::RemoteTerminal { .. }
        )
    }

    /// True when a client cancellation request was followed by a
    /// confirmed terminal Comfy state. This does not say the backend honored
    /// cancellation: `cancellation_confirmed` is true only for native Cancelled.
    pub fn remote_terminal_confirmed(self) -> bool {
        self.kind == DirectRunErrorKind::Cancelled
            && matches!(
                self.disposition,
                DirectRunDisposition::RemoteTerminal { .. }
            )
    }

    /// Callers must keep execution serialized when a child group or remote
    /// request may still be active.
    pub fn blocks_following_work(self) -> bool {
        matches!(
            self.disposition,
            DirectRunDisposition::Process(ProcessFailureDisposition::OwnershipUncertain)
                | DirectRunDisposition::Remote(DispatchDisposition::Uncertain)
        )
    }
}

impl fmt::Display for DirectRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.disposition == DirectRunDisposition::Remote(DispatchDisposition::Uncertain) {
            return formatter.write_str("remote completion is unconfirmed; stop before continuing");
        }
        formatter.write_str(match self.kind {
            DirectRunErrorKind::InvalidRequest => "direct native request is invalid",
            DirectRunErrorKind::CatalogUnavailable => "native catalog is unavailable",
            DirectRunErrorKind::UnsupportedRoute => "direct native route is unsupported",
            DirectRunErrorKind::Cancelled => "native operation was cancelled",
            DirectRunErrorKind::DeadlineExceeded => "native operation timed out",
            DirectRunErrorKind::ExecutionFailed => "native execution failed",
            DirectRunErrorKind::InvalidOutput => "native output is invalid",
            DirectRunErrorKind::OwnershipUncertain => {
                "native process cleanup is uncertain; stop before continuing"
            }
            DirectRunErrorKind::RemoteCompletionUncertain => {
                "remote completion is unconfirmed; stop before continuing"
            }
            DirectRunErrorKind::GpuConfiguration => "GPU-window work requires explicit L3MS_NATIVE_GPU configuration and quiesced clients",
            DirectRunErrorKind::GpuUnavailable => "configured GPU is not idle or the shared native lock is unavailable; no job started",
            DirectRunErrorKind::GpuReleaseUncertain => "GPU release is unconfirmed; stop before continuing",
            DirectRunErrorKind::GpuRecoveryRequired => "native-gpu-recovery-required marker exists; verify interrupted work and GPU release before manual recovery",
        })
    }
}

impl Error for DirectRunError {}

/// One operator-managed GPU window for direct scripts that bypass the router.
/// All L3MS processes must use the same private lock directory;
/// its advisory locks do not fence unrelated clients. The operator explicitly
/// keeps those clients quiesced for the whole operation.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGpuWindow {
    uuid: String,
    index: u32,
    idle_mib: u64,
    lock_dir: PathBuf,
    clients_quiesced: bool,
    #[serde(default)]
    kde_desktop_baseline: bool,
}

impl NativeGpuWindow {
    fn parse(raw: &str) -> Result<Self, DirectRunError> {
        let invalid = || {
            DirectRunError::new(
                DirectRunErrorKind::GpuConfiguration,
                ProcessFailureDisposition::NoChildSpawned,
            )
        };
        if raw.len() > 4096 {
            return Err(invalid());
        }
        let window: Self = serde_json::from_str(raw).map_err(|_| invalid())?;
        if !window.clients_quiesced
            || window.index > 1023
            || !window.lock_dir.is_absolute()
            || CudaDevice::new(&window.uuid).is_err()
        {
            return Err(invalid());
        }
        Ok(window)
    }
}

/// Shared CLI/UI entry: CUDA-native and Blender headless work require explicit
/// idle-GPU admission. Only declared CUDA profiles receive a CUDA device;
/// Blender retains its native scene backend. Both require explicit device
/// configuration, one advisory L3MS lock and idle observations before/after
/// the existing native process runner. CPU/hosted/HTTP behavior is unchanged.
/// llama-swap owns routed engine swapping; this window only guards direct scripts.
pub(crate) fn run_native_in_operator_window(
    catalog: &Catalog,
    request: &InferenceRequest,
    output_directory: Option<&Path>,
    connection: Option<DirectConnection<'_>>,
    context: &OperationContext<'_>,
) -> Result<DirectRunResult, DirectRunError> {
    if is_routed_audio_script_request(catalog, request)
        || !catalog
            .inspect(request.profile())
            .is_some_and(requires_operator_gpu_window)
    {
        return run_native(catalog, request, output_directory, connection, context);
    }
    if connection.is_some() {
        return Err(DirectRunError::new(
            DirectRunErrorKind::InvalidRequest,
            ProcessFailureDisposition::NoChildSpawned,
        ));
    }
    let raw = std::env::var("L3MS_NATIVE_GPU").map_err(|_| {
        DirectRunError::new(
            DirectRunErrorKind::GpuConfiguration,
            ProcessFailureDisposition::NoChildSpawned,
        )
    })?;
    let window = NativeGpuWindow::parse(&raw)?;
    let config =
        GpuProbeConfig::new(window.uuid.clone(), window.index, window.idle_mib).map_err(|_| {
            DirectRunError::new(
                DirectRunErrorKind::GpuConfiguration,
                ProcessFailureDisposition::NoChildSpawned,
            )
        })?;
    let probe = NvidiaSmiGpuProbe::new(config);
    let desktop = window
        .kde_desktop_baseline
        .then(|| DesktopGpuBaseline::capture(context))
        .transpose()
        .map_err(|_| {
            DirectRunError::new(
                DirectRunErrorKind::GpuUnavailable,
                ProcessFailureDisposition::NoChildSpawned,
            )
        })?;
    let cuda_profile = catalog
        .inspect(request.profile())
        .is_some_and(is_supported_direct_cuda_profile);
    let recovery = RecoveryPolicy::new(PROCESS_RECOVERY_TIMEOUT).expect("fixed recovery timeout");
    with_idle_gpu(
        &window,
        context,
        |context| {
            if let Some(desktop) = &desktop {
                desktop.verify(context).map_err(|_| ())?;
            }
            let is_desktop = |pid| desktop.as_ref().is_some_and(|d| d.accepts_claimant(pid));
            let observation = probe
                .observe_idle_with_baseline(context, recovery, is_desktop)
                .map(|_| ());
            if let Some(desktop) = &desktop {
                desktop.verify(context).map_err(|_| ())?;
            }
            observation.map_err(|_| ())
        },
        |device| {
            run_native(
                catalog,
                request,
                output_directory,
                cuda_profile.then_some(DirectConnection::Cuda(device)),
                context,
            )
        },
    )
}

fn with_idle_gpu<T>(
    window: &NativeGpuWindow,
    context: &OperationContext<'_>,
    mut observe: impl FnMut(&OperationContext<'_>) -> Result<(), ()>,
    execute: impl FnOnce(&CudaDevice) -> Result<T, DirectRunError>,
) -> Result<T, DirectRunError> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let unavailable = || {
        DirectRunError::new(
            DirectRunErrorKind::GpuUnavailable,
            ProcessFailureDisposition::NoChildSpawned,
        )
    };
    if context.check_active("native GPU window").is_err() {
        return Err(inactive_error(context));
    }
    let metadata = std::fs::metadata(&window.lock_dir).map_err(|_| unavailable())?;
    if metadata.uid() != unsafe { libc::geteuid() } {
        return Err(unavailable());
    }
    let _lock =
        CoordinatorLock::acquire(&window.lock_dir, [window.index]).map_err(|_| unavailable())?;
    let marker_path = window.lock_dir.join("native-gpu-recovery-required");
    match std::fs::symlink_metadata(&marker_path) {
        Ok(_) => {
            return Err(DirectRunError::new(
                DirectRunErrorKind::GpuRecoveryRequired,
                ProcessFailureDisposition::OwnershipUncertain,
            ))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(unavailable()),
    }
    let device = CudaDevice::new(&window.uuid).map_err(|_| unavailable())?;
    if observe(context).is_err() {
        return Err(if context.check_active("native GPU window").is_err() {
            inactive_error(context)
        } else {
            unavailable()
        });
    }
    if context.check_active("native GPU dispatch").is_err() {
        return Err(inactive_error(context));
    }
    // A single payload-free marker blocks later cooperating invocations after
    // interruption/uncertain release. It is never replayed or auto-recovered.
    let marker = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&marker_path)
        .map_err(|_| unavailable())?;
    let marker_identity = marker.metadata().map_err(|_| unavailable())?;
    let result = if context.check_active("native GPU dispatch").is_err() {
        Err(inactive_error(context))
    } else {
        execute(&device)
    };
    if result
        .as_ref()
        .is_err_and(|error| error.blocks_following_work())
    {
        return result;
    }
    // Cancellation/deadline cannot skip release observation. No inference is
    // retried, and native outputs remain unpublished if release is uncertain.
    let recovery_context = OperationContext::new(
        std::time::Instant::now() + PROCESS_RECOVERY_TIMEOUT,
        &NeverCancel,
    );
    if observe(&recovery_context).is_err() {
        return Err(DirectRunError::new(
            DirectRunErrorKind::GpuReleaseUncertain,
            ProcessFailureDisposition::OwnershipUncertain,
        ));
    }
    let retained = std::fs::symlink_metadata(&marker_path);
    if !retained.is_ok_and(|metadata| {
        metadata.is_file()
            && metadata.nlink() == 1
            && (metadata.dev(), metadata.ino()) == (marker_identity.dev(), marker_identity.ino())
    }) || std::fs::remove_file(&marker_path).is_err()
    {
        return Err(DirectRunError::new(
            DirectRunErrorKind::GpuReleaseUncertain,
            ProcessFailureDisposition::OwnershipUncertain,
        ));
    }
    result
}

/// Any of the four Breeze speech modes as a typed request.
fn is_breeze_speech_request(request: &InferenceRequest) -> bool {
    matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Speech(input),
            ..
        } if matches!(
            input.mode,
            SpeechMode::Standard
                | SpeechMode::VoiceDesign
                | SpeechMode::VoiceClone
                | SpeechMode::Streaming
        )
    )
}

/// Execute one already-built native request on an explicitly selected route.
/// Comfy workflow IDs must be generated and persisted by the caller before this
/// function; this runner never chooses a fallback, starts a runtime, or retries
/// a dispatched request. Comfy cancellation uses one bounded scoped recovery
/// cancel and terminal status is not GPU-release proof. A `Cuda` connection
/// restricts the child to one GPU; it does not provide reservation or release
/// proof. The caller must hold exclusive GPU admission across dispatch and its
/// cleanup/release checks.
fn run_native(
    catalog: &Catalog,
    request: &InferenceRequest,
    output_directory: Option<&Path>,
    connection: Option<DirectConnection<'_>>,
    context: &OperationContext<'_>,
) -> Result<DirectRunResult, DirectRunError> {
    let is_decision = matches!(request, InferenceRequest::Decision { .. });
    let is_embeddings = matches!(request, InferenceRequest::Embeddings { .. });
    let is_comfy_workflow = matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Workflow { .. },
            ..
        }
    );
    let is_hosted_jev_decision = is_decision
        && catalog
            .inspect(request.profile())
            .is_some_and(is_jev_api_decision_profile);
    let is_llamacpp_decision = is_decision
        && catalog
            .inspect(request.profile())
            .is_some_and(is_llamacpp_decision_profile);
    // CUDA Breeze speech is served by llama-swap's /audiocpp/ route (no GPU window).
    let is_routed_speech = is_breeze_speech_request(request)
        && catalog
            .inspect(request.profile())
            .is_some_and(is_supported_cuda_breeze_profile);
    let is_remote_operation = is_embeddings
        || is_comfy_workflow
        || is_hosted_jev_decision
        || is_llamacpp_decision
        || is_routed_speech
        || is_routed_audio_script_request(catalog, request);
    let is_breeze_speech = is_breeze_speech_request(request);
    let is_script_media = matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Script { .. },
            ..
        }
    );
    let (language, comfy, cuda) = match connection {
        Some(DirectConnection::Language(adapter)) => (Some(adapter), None, None),
        Some(DirectConnection::Cuda(device)) => (None, None, Some(device)),
        Some(DirectConnection::Comfy { adapter, native_id }) => {
            (None, Some((adapter, native_id)), None)
        }
        None => (None, None, None),
    };
    let valid_lane_arguments = match request {
        InferenceRequest::Decision { .. } => {
            output_directory.is_none()
                && language.is_some() == is_llamacpp_decision
                && comfy.is_none()
                && (!is_llamacpp_decision || cuda.is_none())
        }
        InferenceRequest::Media { .. }
            if is_routed_speech || is_routed_audio_script_request(catalog, request) =>
        {
            output_directory.is_some() && language.is_some() && comfy.is_none() && cuda.is_none()
        }
        InferenceRequest::Media { .. } if is_breeze_speech || is_script_media => {
            output_directory.is_some() && language.is_none() && comfy.is_none()
        }
        InferenceRequest::Media { .. } if is_comfy_workflow => {
            output_directory.is_some() && language.is_none() && comfy.is_some() && cuda.is_none()
        }
        InferenceRequest::Embeddings { .. } => {
            output_directory.is_none() && language.is_some() && comfy.is_none() && cuda.is_none()
        }
        _ => false,
    };
    if !valid_lane_arguments {
        return Err(unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_remote_operation,
        ));
    }
    if context.check_active("direct native operation").is_err() {
        return Err(if is_remote_operation {
            inactive_remote_error(context)
        } else {
            inactive_error(context)
        });
    }

    let original_profile = catalog
        .inspect(request.profile())
        .ok_or_else(|| unstarted_error(DirectRunErrorKind::InvalidRequest, is_remote_operation))?;
    let catalog = catalog.refresh().map_err(|_| {
        unstarted_error(DirectRunErrorKind::CatalogUnavailable, is_remote_operation)
    })?;
    if catalog.inspect(request.profile()) != Some(original_profile) {
        return Err(unstarted_error(
            DirectRunErrorKind::InvalidRequest,
            is_remote_operation,
        ));
    }
    request
        .validate(&catalog)
        .map_err(|_| unstarted_error(DirectRunErrorKind::InvalidRequest, is_remote_operation))?;
    let route = resolve_profile_route(&catalog, request.profile())
        .map_err(|_| unstarted_error(DirectRunErrorKind::InvalidRequest, is_remote_operation))?;
    let recovery = RecoveryPolicy::new(PROCESS_RECOVERY_TIMEOUT)
        .expect("fixed process recovery timeout is valid");
    if is_comfy_workflow {
        if !is_comfy_workflow_route(request, &route) {
            return Err(DirectRunError::remote(
                DirectRunErrorKind::UnsupportedRoute,
                DispatchDisposition::NoDispatch,
            ));
        }
        let (adapter, native_id) = comfy.expect("validated Comfy connection");
        let output_directory = output_directory.expect("validated Comfy output directory");
        execute_comfy_workflow(
            &catalog,
            request,
            &route,
            output_directory,
            adapter,
            native_id,
            context,
        )
        .map(DirectRunResult::Comfy)
    } else if is_decision {
        let profile = catalog
            .inspect(request.profile())
            .expect("validated request profile exists in refreshed catalog");
        if is_llamacpp_decision_profile(profile) {
            return language
                .expect("validated decision connection")
                .execute_decisions(&catalog, request, context)
                .map(DirectRunResult::Decision)
                .map_err(remote_language_error);
        }
        if is_supported_cuda_direct_decision_profile(profile) {
            let Some(device) = cuda else {
                return Err(DirectRunError::new(
                    DirectRunErrorKind::UnsupportedRoute,
                    ProcessFailureDisposition::NoChildSpawned,
                ));
            };
            return execute_accepted_cuda_decision(
                &catalog, request, &route, device, context, recovery,
            )
            .map(DirectRunResult::Decision)
            .map_err(|error| execution_error(error, context));
        } else if !is_supported_direct_decision_profile(profile) || cuda.is_some() {
            return Err(if is_jev_api_decision_profile(profile) {
                DirectRunError::remote(
                    DirectRunErrorKind::UnsupportedRoute,
                    DispatchDisposition::NoDispatch,
                )
            } else {
                DirectRunError::new(
                    DirectRunErrorKind::UnsupportedRoute,
                    ProcessFailureDisposition::NoChildSpawned,
                )
            });
        }
        if is_jev_api_decision_profile(profile) {
            execute_accepted_hosted_decision(&catalog, request, &route, context, recovery)
                .map(DirectRunResult::Decision)
                .map_err(|error| hosted_decision_error(error, context))
        } else {
            execute_accepted_cpu_decision(&catalog, request, &route, context, recovery)
                .map(DirectRunResult::Decision)
                .map_err(|error| execution_error(error, context))
        }
    } else if is_script_media {
        if is_routed_audio_script_request(&catalog, request) {
            let output = output_directory.expect("validated audio output directory");
            super::media_execution::preflight_routed_audio_output(output, false).map_err(|_| {
                DirectRunError::remote(
                    DirectRunErrorKind::InvalidRequest,
                    DispatchDisposition::NoDispatch,
                )
            })?;
            let audio = language
                .expect("validated routed audio connection")
                .execute_routed_audio_script(request, context)
                .map_err(remote_language_error)?;
            return super::media_execution::publish_routed_audio(request, &audio, output, context)
                .map(DirectRunResult::ScriptMedia)
                .map_err(|_| remote_completion_uncertain());
        }
        let profile = catalog
            .inspect(request.profile())
            .expect("validated refreshed profile");
        if is_supported_blender_headless_profile(profile) {
            if cuda.is_some() {
                return Err(DirectRunError::new(
                    DirectRunErrorKind::UnsupportedRoute,
                    ProcessFailureDisposition::NoChildSpawned,
                ));
            }
            return super::media_execution::execute_blender_headless(
                &catalog,
                request,
                &route,
                output_directory.expect("validated script output directory"),
                context,
                recovery,
            )
            .map(DirectRunResult::ScriptMedia)
            .map_err(|error| media_execution_error(error, context));
        }
        if !is_supported_direct_script_profile(profile) || cuda.is_none() {
            return Err(DirectRunError::new(
                DirectRunErrorKind::UnsupportedRoute,
                ProcessFailureDisposition::NoChildSpawned,
            ));
        }
        let execute = if is_supported_direct_h3_profile(profile) {
            super::media_execution::execute_h3_on_device
        } else if is_supported_direct_ltx_profile(profile) {
            super::media_execution::execute_ltx_on_device
        } else if is_supported_direct_yue2_profile(profile) {
            super::media_execution::execute_yue2_on_device
        } else {
            execute_heartmula_music_on_device
        };
        execute(
            &catalog,
            request,
            &route,
            cuda.expect("validated CUDA device"),
            output_directory.expect("validated music output directory"),
            context,
            recovery,
        )
        .map(DirectRunResult::ScriptMedia)
        .map_err(|error| media_execution_error(error, context))
    } else if is_breeze_speech {
        if is_supported_cpu_breeze_route(request, &route) {
            if cuda.is_some() {
                return Err(DirectRunError::new(
                    DirectRunErrorKind::UnsupportedRoute,
                    ProcessFailureDisposition::NoChildSpawned,
                ));
            }
            let output_directory = output_directory.expect("validated speech output directory");
            execute_breeze_speech_on_cpu(
                &catalog,
                request,
                &route,
                output_directory,
                context,
                recovery,
            )
            .map(DirectRunResult::Speech)
            .map_err(|error| media_execution_error(error, context))
        } else if is_supported_cuda_breeze_route(request, &route) {
            let InferenceRequest::Media {
                input: MediaInput::Speech(input),
                ..
            } = request
            else {
                unreachable!("validated speech request")
            };
            let output_directory = output_directory.expect("validated speech output directory");
            super::media_execution::preflight_routed_audio_output(
                output_directory,
                input.mode == super::requests::SpeechMode::Streaming,
            )
            .map_err(|_| {
                DirectRunError::remote(
                    DirectRunErrorKind::InvalidRequest,
                    DispatchDisposition::NoDispatch,
                )
            })?;
            let audio = language
                .expect("validated routed speech connection")
                .execute_speech(input, context)
                .map_err(remote_language_error)?;
            super::media_execution::publish_routed_speech(
                request,
                &audio,
                output_directory,
                context,
            )
            .map(DirectRunResult::Speech)
            .map_err(|_| remote_completion_uncertain())
        } else {
            Err(DirectRunError::new(
                DirectRunErrorKind::UnsupportedRoute,
                ProcessFailureDisposition::NoChildSpawned,
            ))
        }
    } else if is_embeddings {
        if !is_language_embeddings_route(request, &route) {
            return Err(DirectRunError::remote(
                DirectRunErrorKind::UnsupportedRoute,
                DispatchDisposition::NoDispatch,
            ));
        }
        let adapter = language.expect("validated embeddings adapter");
        let result = adapter
            .execute_embeddings_for_route(&catalog, request, &route, context)
            .map_err(remote_language_error)?;
        match result {
            InferenceResult::Embeddings { vectors, usage } => {
                Ok(DirectRunResult::Embeddings(EmbeddingPayload {
                    vectors,
                    usage,
                }))
            }
            _ => Err(DirectRunError::remote(
                DirectRunErrorKind::InvalidOutput,
                DispatchDisposition::Uncertain,
            )),
        }
    } else {
        Err(unstarted_error(DirectRunErrorKind::InvalidRequest, false))
    }
}

fn is_comfy_workflow_route(request: &InferenceRequest, route: &ResolvedProfileRoute) -> bool {
    let InferenceRequest::Media {
        profile,
        input: MediaInput::Workflow { task, graph },
    } = request
    else {
        return false;
    };
    profile == &route.profile
        && route.profile.lane == Lane::Media
        && route.runtime.lane == Lane::Media
        && route.runtime.as_str() == "ComfyUI"
        && route.tasks.iter().any(|declared| declared == task)
        && matches!(&route.native_route, NativeRoute::MediaScript { .. })
        && graph.as_object().is_some_and(|object| !object.is_empty())
}

fn execute_comfy_workflow(
    catalog: &Catalog,
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
    output_directory: &Path,
    adapter: &ComfyAdapter,
    native_id: &str,
    context: &OperationContext<'_>,
) -> Result<NativeComfyResult, DirectRunError> {
    let output_root = ComfyOutputRoot::open_fresh(output_directory).map_err(comfy_failure_error)?;
    let mut job = adapter
        .prepare_for_route(catalog, request, route, native_id)
        .map_err(comfy_failure_error)?;
    if let Err(error) = adapter.submit_for_route(&mut job, route, context) {
        if job.submission_attempted() && context_inactive(context) {
            return recover_comfy_cancellation(adapter, &job, context);
        }
        return Err(comfy_failure_error(error));
    }

    loop {
        if context_inactive(context) {
            return recover_comfy_cancellation(adapter, &job, context);
        }
        match adapter.status(&job, context) {
            Ok(snapshot) => {
                if let Some(result) = finish_comfy_snapshot(
                    adapter,
                    snapshot,
                    &output_root,
                    output_directory,
                    native_id,
                    context,
                )? {
                    return Ok(result);
                }
            }
            Err(_) if context_inactive(context) => {
                return recover_comfy_cancellation(adapter, &job, context)
            }
            Err(error) => return Err(comfy_failure_error(error)),
        }
        wait_for_poll(context);
    }
}

fn recover_comfy_cancellation(
    adapter: &ComfyAdapter,
    job: &super::comfy::PreparedComfyJob,
    caller_context: &OperationContext<'_>,
) -> Result<NativeComfyResult, DirectRunError> {
    let deadline = std::time::Instant::now()
        .checked_add(PROCESS_RECOVERY_TIMEOUT)
        .ok_or_else(remote_completion_uncertain)?;
    let recovery = OperationContext::new(deadline, &NeverCancel);

    // A successful cancel response is only an acknowledgement. The exact
    // native ID is polled until terminal; this never proves engine/GPU release.
    let _ = adapter.cancel(job, &recovery);
    loop {
        if recovery.remaining().is_zero() {
            return Err(remote_completion_uncertain());
        }
        match adapter.status(job, &recovery) {
            Ok(snapshot) => match snapshot.state {
                ComfyJobState::Completed | ComfyJobState::Failed => {
                    return Err(comfy_recovery_terminal_error(caller_context))
                }
                ComfyJobState::Cancelled => {
                    return Err(DirectRunError::remote_terminal(
                        DirectRunErrorKind::Cancelled,
                        true,
                    ))
                }
                ComfyJobState::Unknown | ComfyJobState::Pending | ComfyJobState::Running => {}
            },
            Err(_) => return Err(remote_completion_uncertain()),
        }
        wait_for_poll(&recovery);
    }
}

fn comfy_recovery_terminal_error(context: &OperationContext<'_>) -> DirectRunError {
    let kind = if context.is_cancelled() {
        DirectRunErrorKind::Cancelled
    } else {
        DirectRunErrorKind::DeadlineExceeded
    };
    DirectRunError::remote_terminal(kind, false)
}

fn finish_comfy_snapshot(
    adapter: &ComfyAdapter,
    snapshot: super::comfy::ComfySnapshot,
    output_root: &ComfyOutputRoot,
    output_directory: &Path,
    native_id: &str,
    context: &OperationContext<'_>,
) -> Result<Option<NativeComfyResult>, DirectRunError> {
    match snapshot.state {
        ComfyJobState::Completed => {
            let artifacts = adapter
                .collect_terminal_outputs(output_root, &snapshot, context)
                .map_err(|error| {
                    DirectRunError::remote_terminal(comfy_direct_error_kind(error.kind), false)
                })?;
            Ok(Some(NativeComfyResult {
                native_id: native_id.to_owned(),
                output_root: output_directory.to_path_buf(),
                artifacts,
            }))
        }
        ComfyJobState::Failed => Err(DirectRunError::remote_terminal(
            DirectRunErrorKind::ExecutionFailed,
            false,
        )),
        ComfyJobState::Cancelled => Err(DirectRunError::remote_terminal(
            DirectRunErrorKind::Cancelled,
            true,
        )),
        ComfyJobState::Unknown | ComfyJobState::Pending | ComfyJobState::Running => Ok(None),
    }
}

fn wait_for_poll(context: &OperationContext<'_>) {
    let remaining = context.remaining();
    if !remaining.is_zero() {
        std::thread::sleep(remaining.min(Duration::from_millis(25)));
    }
}

fn context_inactive(context: &OperationContext<'_>) -> bool {
    context.is_cancelled() || context.remaining().is_zero()
}

fn comfy_failure_error(error: ComfyFailure) -> DirectRunError {
    DirectRunError::remote(comfy_direct_error_kind(error.kind), error.disposition)
}

fn comfy_direct_error_kind(kind: ComfyErrorKind) -> DirectRunErrorKind {
    match kind {
        ComfyErrorKind::Cancelled => DirectRunErrorKind::Cancelled,
        ComfyErrorKind::Deadline => DirectRunErrorKind::DeadlineExceeded,
        ComfyErrorKind::Configuration
        | ComfyErrorKind::InvalidRequest
        | ComfyErrorKind::InvalidOwnership => DirectRunErrorKind::InvalidRequest,
        ComfyErrorKind::Protocol | ComfyErrorKind::TooLarge => DirectRunErrorKind::InvalidOutput,
        ComfyErrorKind::AlreadyAttempted | ComfyErrorKind::Transport | ComfyErrorKind::Rejected => {
            DirectRunErrorKind::ExecutionFailed
        }
    }
}

fn remote_completion_uncertain() -> DirectRunError {
    DirectRunError::remote(
        DirectRunErrorKind::RemoteCompletionUncertain,
        DispatchDisposition::Uncertain,
    )
}

/// CPU, hosted and router-owned HTTP decision routes exposed to the UI.
/// Child-owned CUDA remains a separate explicit selection.
pub(crate) fn is_supported_direct_decision_profile(profile: &CatalogProfile) -> bool {
    if is_llamacpp_decision_profile(profile) {
        return true;
    }
    if profile.disabled
        || profile.key.lane != Lane::Decisions
        || profile.runtime.lane != Lane::Decisions
        || !profile.tasks.iter().any(|task| task.as_str() == "decision")
    {
        return false;
    }
    let NativeProfileSource::Decisions { profile_type, .. } = &profile.native_source else {
        return false;
    };
    matches!(
        (
            profile.runtime.as_str(),
            profile.device.as_ref(),
            profile_type.as_str(),
        ),
        ("nagi" | "laya", Some(Device::Cpu), "local") | ("jev", Some(Device::Remote), "api")
    )
}

/// The router owns this GPU model. HTTP dispatch must not stop its service or
/// launch a Python/CUDA child, despite the model's declared CUDA device.
pub(crate) fn is_llamacpp_decision_profile(profile: &CatalogProfile) -> bool {
    !profile.disabled
        && profile.key.lane == Lane::Decisions
        && profile.runtime.lane == Lane::Decisions
        && profile.runtime.as_str() == "llamacpp"
        && profile.device == Some(Device::Cuda)
        && profile.tasks.iter().any(|task| task.as_str() == "decision")
        && matches!(&profile.native_source, NativeProfileSource::Decisions {profile_type, ..} if profile_type == "local")
}

/// Profile-level CUDA-local Nagi/Laya decision routes. Callers still need an
/// explicit `CudaDevice`; this predicate does not establish GPU ownership.
pub(crate) fn is_supported_cuda_direct_decision_profile(profile: &CatalogProfile) -> bool {
    if profile.disabled
        || profile.key.lane != Lane::Decisions
        || profile.runtime.lane != Lane::Decisions
        || !profile.tasks.iter().any(|task| task.as_str() == "decision")
    {
        return false;
    }
    let NativeProfileSource::Decisions { profile_type, .. } = &profile.native_source else {
        return false;
    };
    matches!(
        (
            profile.runtime.as_str(),
            profile.device.as_ref(),
            profile_type.as_str(),
        ),
        ("nagi" | "laya", Some(Device::Cuda), "local")
    )
}

/// Profile-level CUDA Breeze route selectable by direct media forms. The
/// route-level check below additionally binds the request and refreshed task.
pub(crate) fn is_supported_cuda_breeze_profile(profile: &CatalogProfile) -> bool {
    const BREEZE_PROFILE_ID: &str = "breeze-tts-2";
    const BREEZE_SCRIPT: &str = "media-models/generate-breeze-tts-2.sh";

    !profile.disabled
        && profile.key.lane == Lane::Media
        && profile.key.profile_id.as_str() == BREEZE_PROFILE_ID
        && profile.runtime.lane == Lane::Media
        && profile.runtime.as_str() == "audio.cpp"
        && profile.device.as_ref() == Some(&Device::Cuda)
        && profile
            .tasks
            .iter()
            .any(|task| matches!(task.as_str(), "tts" | "design" | "clon"))
        && matches!(
            &profile.native_source,
            NativeProfileSource::Media { script } if script == BREEZE_SCRIPT
        )
}

/// Whether a profile has one of the direct CUDA implementations. The caller
/// must still apply its operator-selected device and admission checks.
pub(crate) fn is_supported_direct_cuda_profile(profile: &CatalogProfile) -> bool {
    is_supported_cuda_direct_decision_profile(profile)
        || is_supported_direct_script_profile(profile)
}

pub(crate) fn is_routed_audio_script_request(
    catalog: &Catalog,
    request: &InferenceRequest,
) -> bool {
    matches!(
        request,
        InferenceRequest::Media {
            input: MediaInput::Script { .. },
            ..
        }
    ) && catalog.inspect(request.profile()).is_some_and(|profile| {
        (is_supported_direct_music_profile(profile) && request.task() == "music")
            || (is_supported_direct_h3_profile(profile)
                && matches!(request.task(), "music" | "tts"))
    })
}

pub(crate) fn is_supported_direct_script_profile(profile: &CatalogProfile) -> bool {
    is_supported_direct_music_profile(profile)
        || is_supported_direct_h3_profile(profile)
        || is_supported_direct_ltx_profile(profile)
        || is_supported_direct_yue2_profile(profile)
}

pub(crate) fn is_supported_direct_yue2_profile(profile: &CatalogProfile) -> bool {
    !profile.disabled
        && profile.key.lane == Lane::Media
        && profile.key.profile_id.as_str() == "yue2-music"
        && profile.runtime.lane == Lane::Media
        && profile.runtime.as_str() == "YuE2 Python"
        && profile.device == Some(Device::Cuda)
        && profile.tasks.iter().any(|task| task.as_str() == "music")
        && matches!(&profile.native_source, NativeProfileSource::Media { script }
            if script == "media-models/generate-yue2.sh")
}

/// GPU admission is separate from selecting a CUDA inference device. Blender
/// keeps its scene-configured backend, but must not overlap other GPU work.
pub(crate) fn requires_operator_gpu_window(profile: &CatalogProfile) -> bool {
    is_supported_direct_cuda_profile(profile) || is_supported_blender_headless_profile(profile)
}

pub(crate) fn is_supported_blender_headless_profile(profile: &CatalogProfile) -> bool {
    !profile.disabled
        && profile.key.lane == Lane::Media
        && profile.key.profile_id.as_str() == "blender-3d"
        && profile.runtime.lane == Lane::Media
        && profile.runtime.as_str() == "Blender 5.2 / blender-mcp"
        && profile.device.is_none()
        && profile
            .tasks
            .iter()
            .any(|task| matches!(task.as_str(), "3d" | "render" | "image"))
        && matches!(&profile.native_source, NativeProfileSource::Media { script }
            if script == "media-models/generate-blender-3d.sh")
}

pub(crate) fn is_supported_direct_ltx_profile(profile: &CatalogProfile) -> bool {
    !profile.disabled
        && profile.key.lane == Lane::Media
        && profile.key.profile_id.as_str() == "ltx-2.5"
        && profile.runtime.lane == Lane::Media
        && profile.runtime.as_str() == "LTX-2 Python"
        && profile.device.as_ref() == Some(&Device::Cuda)
        && profile
            .tasks
            .iter()
            .any(|task| matches!(task.as_str(), "video" | "music"))
        && matches!(&profile.native_source, NativeProfileSource::Media { script }
            if script == "media-models/generate-ltx-2.5.sh")
}

pub(crate) fn is_supported_direct_h3_profile(profile: &CatalogProfile) -> bool {
    !profile.disabled
        && profile.key.lane == Lane::Media
        && profile.key.profile_id.as_str() == "minimax-h3"
        && profile.runtime.lane == Lane::Media
        && profile.runtime.as_str() == "audio.cpp"
        && profile.device.as_ref() == Some(&Device::Cuda)
        && profile
            .tasks
            .iter()
            .any(|task| matches!(task.as_str(), "music" | "tts" | "video"))
        && matches!(&profile.native_source, NativeProfileSource::Media { script }
            if script == "media-models/generate-minimax-h3.sh")
}

pub(crate) fn is_supported_direct_music_profile(profile: &CatalogProfile) -> bool {
    !profile.disabled
        && profile.key.lane == Lane::Media
        && profile.key.profile_id.as_str() == "heartmula-music"
        && profile.runtime.lane == Lane::Media
        && profile.runtime.as_str() == "audio.cpp"
        && profile.device.as_ref() == Some(&Device::Cuda)
        && profile.tasks.iter().any(|task| task.as_str() == "music")
        && matches!(&profile.native_source, NativeProfileSource::Media { script }
            if script == "media-models/generate-heartmula-music.sh")
}

fn is_supported_cuda_breeze_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
) -> bool {
    let InferenceRequest::Media {
        profile,
        input: MediaInput::Speech(_),
    } = request
    else {
        return false;
    };
    profile == &route.profile
        && route.profile.lane == Lane::Media
        && route.profile.profile_id.as_str() == "breeze-tts-2"
        && route.runtime.lane == Lane::Media
        && route.runtime.as_str() == "audio.cpp"
        && route.device == Some(Device::Cuda)
        && route
            .tasks
            .iter()
            .any(|task| task.as_str() == request.task())
        && matches!(
            &route.native_route,
            NativeRoute::MediaScript { path }
                if path.file_name().and_then(|name| name.to_str()) == Some("generate-breeze-tts-2.sh")
        )
}

fn is_jev_api_decision_profile(profile: &CatalogProfile) -> bool {
    profile.key.lane == Lane::Decisions
        && profile.runtime.lane == Lane::Decisions
        && profile.runtime.as_str() == "jev"
        && matches!(
            &profile.native_source,
            NativeProfileSource::Decisions { profile_type, .. } if profile_type == "api"
        )
}

fn is_language_embeddings_route(request: &InferenceRequest, route: &ResolvedProfileRoute) -> bool {
    let InferenceRequest::Embeddings { profile, .. } = request else {
        return false;
    };
    let NativeRoute::LlamaSwapModel { model_id, .. } = &route.native_route else {
        return false;
    };
    profile == &route.profile
        && route.profile.lane == Lane::Language
        && route.runtime.lane == Lane::Language
        && model_id == &route.profile.profile_id
        && route.tasks.iter().any(|task| task.as_str() == "embeddings")
}

fn inactive_error(context: &OperationContext<'_>) -> DirectRunError {
    let kind = if context.is_cancelled() {
        DirectRunErrorKind::Cancelled
    } else {
        DirectRunErrorKind::DeadlineExceeded
    };
    DirectRunError::new(kind, ProcessFailureDisposition::NoChildSpawned)
}

fn inactive_remote_error(context: &OperationContext<'_>) -> DirectRunError {
    let kind = if context.is_cancelled() {
        DirectRunErrorKind::Cancelled
    } else {
        DirectRunErrorKind::DeadlineExceeded
    };
    DirectRunError::remote(kind, DispatchDisposition::NoDispatch)
}

fn unstarted_error(kind: DirectRunErrorKind, remote: bool) -> DirectRunError {
    if remote {
        DirectRunError::remote(kind, DispatchDisposition::NoDispatch)
    } else {
        DirectRunError::new(kind, ProcessFailureDisposition::NoChildSpawned)
    }
}

fn remote_language_error(error: LanguageFailure) -> DirectRunError {
    let kind = match error.kind {
        LanguageErrorKind::Configuration | LanguageErrorKind::InvalidRequest => {
            DirectRunErrorKind::InvalidRequest
        }
        LanguageErrorKind::Cancelled => DirectRunErrorKind::Cancelled,
        LanguageErrorKind::Deadline => DirectRunErrorKind::DeadlineExceeded,
        LanguageErrorKind::Protocol | LanguageErrorKind::TooLarge => {
            DirectRunErrorKind::InvalidOutput
        }
        LanguageErrorKind::Transport | LanguageErrorKind::Rejected => {
            DirectRunErrorKind::ExecutionFailed
        }
    };
    DirectRunError::remote(kind, error.disposition)
}

fn execution_error(
    error: DecisionExecutionError,
    context: &OperationContext<'_>,
) -> DirectRunError {
    match error {
        DecisionExecutionError::NotDispatched => {
            if context.check_active("direct native operation").is_err() {
                inactive_error(context)
            } else {
                DirectRunError::new(
                    DirectRunErrorKind::ExecutionFailed,
                    ProcessFailureDisposition::NoChildSpawned,
                )
            }
        }
        DecisionExecutionError::InvalidResult(_) => DirectRunError::new(
            DirectRunErrorKind::InvalidOutput,
            ProcessFailureDisposition::ChildGroupReaped,
        ),
        DecisionExecutionError::Uncertain { process } => {
            let kind = match process {
                ProcessFailureDisposition::OwnershipUncertain => {
                    DirectRunErrorKind::OwnershipUncertain
                }
                ProcessFailureDisposition::ChildGroupReaped if context.is_cancelled() => {
                    DirectRunErrorKind::Cancelled
                }
                ProcessFailureDisposition::ChildGroupReaped if context.remaining().is_zero() => {
                    DirectRunErrorKind::DeadlineExceeded
                }
                _ => DirectRunErrorKind::ExecutionFailed,
            };
            DirectRunError::new(kind, process)
        }
    }
}

fn hosted_decision_error(
    error: DecisionExecutionError,
    context: &OperationContext<'_>,
) -> DirectRunError {
    match error {
        DecisionExecutionError::NotDispatched => {
            if context.check_active("hosted direct decision").is_err() {
                inactive_remote_error(context)
            } else {
                DirectRunError::remote(
                    DirectRunErrorKind::ExecutionFailed,
                    DispatchDisposition::NoDispatch,
                )
            }
        }
        DecisionExecutionError::InvalidResult(_) | DecisionExecutionError::Uncertain { .. } => {
            remote_completion_uncertain()
        }
    }
}

fn media_execution_error(
    error: MediaExecutionFailure,
    context: &OperationContext<'_>,
) -> DirectRunError {
    let disposition = error.disposition();
    if disposition == ProcessFailureDisposition::NoChildSpawned
        && context.check_active("direct native operation").is_err()
    {
        return inactive_error(context);
    }
    let kind = if disposition == ProcessFailureDisposition::OwnershipUncertain {
        DirectRunErrorKind::OwnershipUncertain
    } else {
        match error.kind() {
            MediaExecutionErrorKind::InvalidArtifact
            | MediaExecutionErrorKind::ArtifactTooLarge
            | MediaExecutionErrorKind::TooManyArtifacts => DirectRunErrorKind::InvalidOutput,
            MediaExecutionErrorKind::InvalidRequest => DirectRunErrorKind::InvalidRequest,
            MediaExecutionErrorKind::Process(ProcessFailureKind::Cancelled) => {
                DirectRunErrorKind::Cancelled
            }
            MediaExecutionErrorKind::Process(ProcessFailureKind::DeadlineExceeded) => {
                DirectRunErrorKind::DeadlineExceeded
            }
            MediaExecutionErrorKind::Process(_) => {
                if context.is_cancelled()
                    && disposition != ProcessFailureDisposition::NoChildSpawned
                {
                    DirectRunErrorKind::Cancelled
                } else if context.remaining().is_zero()
                    && disposition != ProcessFailureDisposition::NoChildSpawned
                {
                    DirectRunErrorKind::DeadlineExceeded
                } else {
                    DirectRunErrorKind::ExecutionFailed
                }
            }
            MediaExecutionErrorKind::InvalidOutputDirectory
            | MediaExecutionErrorKind::OutputAlreadyExists => DirectRunErrorKind::InvalidRequest,
        }
    };
    DirectRunError::new(kind, disposition)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    use serde_json::{json, Map};
    use tempfile::TempDir;

    use super::super::catalog::ProfileKey;
    use super::super::native_commands::CPU_BREEZE_PROFILE_ID;
    use super::super::requests::InferenceResult;
    use super::super::runtime::NeverCancel;

    struct AlwaysCancel;

    impl super::super::runtime::Cancellation for AlwaysCancel {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    struct SharedCancel(Arc<AtomicBool>);

    impl super::super::runtime::Cancellation for SharedCancel {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    fn gpu_window(directory: &Path) -> NativeGpuWindow {
        NativeGpuWindow::parse(
            &json!({
                "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "index": 0, "idle_mib": 512, "lock_dir": directory,
                "clients_quiesced": true
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn operator_gpu_configuration_is_explicit_and_redacted() {
        let valid = json!({"uuid":"GPU-12345678-1234-1234-1234-123456789abc",
            "index":0,"idle_mib":512,"lock_dir":"/private/native-locks","clients_quiesced":true});
        assert!(
            !NativeGpuWindow::parse(&valid.to_string())
                .unwrap()
                .kde_desktop_baseline
        );
        let mut desktop = valid.clone();
        desktop["kde_desktop_baseline"] = json!(true);
        assert!(
            NativeGpuWindow::parse(&desktop.to_string())
                .unwrap()
                .kde_desktop_baseline
        );
        for (field, value) in [
            ("uuid", json!("0")),
            ("index", json!(1024)),
            ("lock_dir", json!("relative")),
            ("clients_quiesced", json!(false)),
            ("kde_desktop_baseline", json!("unverified")),
            ("extra", json!(true)),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            let error = NativeGpuWindow::parse(&invalid.to_string()).err().unwrap();
            assert_eq!(error.kind(), DirectRunErrorKind::GpuConfiguration);
            assert!(!error.to_string().contains("/private"));
        }
        assert!(NativeGpuWindow::parse(&"x".repeat(4097)).is_err());
    }

    #[test]
    fn blender_requires_idle_gpu_admission_without_a_cuda_profile() {
        let catalog = Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap();
        let key = super::super::catalog::ProfileKey::new(Lane::Media, "blender-3d").unwrap();
        let profile = catalog.inspect(&key).unwrap();
        assert!(is_supported_blender_headless_profile(profile));
        assert!(requires_operator_gpu_window(profile));
        assert!(!is_supported_direct_cuda_profile(profile));
        assert!(!is_supported_direct_script_profile(profile));
        assert!(profile.device.is_none());
        let mut changed = profile.clone();
        changed.device = Some(Device::Cuda);
        assert!(!is_supported_blender_headless_profile(&changed));
        changed.device = None;
        changed.disabled = true;
        assert!(!is_supported_blender_headless_profile(&changed));
    }

    #[test]
    fn idle_gpu_window_serializes_and_checks_release_without_retrying() {
        use std::cell::Cell;
        for failing_observation in [0, 1, 2] {
            let directory = tempfile::tempdir().unwrap();
            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
            let window = gpu_window(directory.path());
            let calls = Cell::new(0);
            let executions = Cell::new(0);
            let context =
                OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel);
            let result = with_idle_gpu(
                &window,
                &context,
                |_| {
                    calls.set(calls.get() + 1);
                    if calls.get() == failing_observation {
                        Err(())
                    } else {
                        Ok(())
                    }
                },
                |device| {
                    executions.set(executions.get() + 1);
                    assert_eq!(device.gpu_uuid(), window.uuid);
                    assert!(CoordinatorLock::acquire(directory.path(), [0]).is_err());
                    Ok(42)
                },
            );
            assert_eq!(executions.get(), usize::from(failing_observation != 1));
            assert_eq!(calls.get(), if failing_observation == 1 { 1 } else { 2 });
            match failing_observation {
                0 => assert_eq!(result.unwrap(), 42),
                1 => assert!(!result.unwrap_err().blocks_following_work()),
                2 => assert!(result.unwrap_err().blocks_following_work()),
                _ => unreachable!(),
            }
            // Empty lock inodes remain; release never authorizes skipping the
            // next invocation's idle probe or bypassing external quiescence.
            assert!(CoordinatorLock::acquire(directory.path(), [0]).is_ok());
            if failing_observation == 2 {
                assert!(directory
                    .path()
                    .join("native-gpu-recovery-required")
                    .is_file());
                let retry: Result<(), _> = with_idle_gpu(
                    &window,
                    &context,
                    |_| Ok(()),
                    |_| panic!("uncertain work must not be restarted"),
                );
                assert_eq!(
                    retry.unwrap_err().kind(),
                    DirectRunErrorKind::GpuRecoveryRequired
                );
            } else {
                assert!(!directory
                    .path()
                    .join("native-gpu-recovery-required")
                    .exists());
            }
        }
    }

    #[test]
    fn cancelled_gpu_job_uses_fresh_release_context_and_preserves_uncertainty() {
        use std::cell::Cell;
        for uncertain_child in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
            let window = gpu_window(directory.path());
            let cancelled = Arc::new(AtomicBool::new(false));
            let cancellation = SharedCancel(cancelled.clone());
            let context =
                OperationContext::new(Instant::now() + Duration::from_secs(2), &cancellation);
            let calls = Cell::new(0);
            let result: Result<(), _> = with_idle_gpu(
                &window,
                &context,
                |observation_context| {
                    calls.set(calls.get() + 1);
                    assert!(!observation_context.is_cancelled());
                    assert!(!observation_context.remaining().is_zero());
                    Ok(())
                },
                |_| {
                    cancelled.store(true, Ordering::Release);
                    Err(DirectRunError::new(
                        DirectRunErrorKind::Cancelled,
                        if uncertain_child {
                            ProcessFailureDisposition::OwnershipUncertain
                        } else {
                            ProcessFailureDisposition::ChildGroupReaped
                        },
                    ))
                },
            );
            let error = result.unwrap_err();
            assert_eq!(calls.get(), if uncertain_child { 1 } else { 2 });
            assert_eq!(error.blocks_following_work(), uncertain_child);
            assert_eq!(error.cancellation_confirmed(), !uncertain_child);
            assert_eq!(
                directory
                    .path()
                    .join("native-gpu-recovery-required")
                    .exists(),
                uncertain_child
            );
        }
    }

    fn fixture(
        engine: &str,
        profile_type: &str,
        device: &str,
        response: &str,
    ) -> (TempDir, Catalog, InferenceRequest, std::path::PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("toolkits/l2m2");
        let decisions = root.path().join("toolkits/l3ds");
        fs::create_dir_all(&media).unwrap();
        fs::create_dir_all(decisions.join("decision-models")).unwrap();
        fs::create_dir_all(decisions.join(".venv/bin")).unwrap();
        fs::create_dir_all(decisions.join(".venv-hosted/bin")).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-chat:\n    cmd: server\n",
        )
        .unwrap();
        fs::write(
            media.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        let manifest = json!({
            "version": "1.0",
            "profiles": [{
                "id": "fixture-decision",
                "name": "Fixture decision",
                "engine": engine,
                "type": profile_type,
                "device": device,
                "model": "fixture-model",
                "entrypoint": "decision-models/runner.py"
            }]
        });
        fs::write(
            decisions.join("decision-runtimes.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let interpreter = decisions.join(".venv/bin/python");
        let hosted_interpreter = decisions.join(".venv-hosted/bin/python");
        let fixture_guard = super::super::process::executable_fixture_guard();
        fs::write(&interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        fs::write(&hosted_interpreter, "#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o700)).unwrap();
        fs::set_permissions(&hosted_interpreter, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(
            decisions.join("decision-models/runner.py"),
            format!(
                "#!/bin/sh\nprintf x >> \"$0.calls\"\n/bin/cat >/dev/null\nprintf '%s\\n' '{}'\n",
                response.replace('\'', "'\\''")
            ),
        )
        .unwrap();
        drop(fixture_guard);

        let catalog = Catalog::load(root.path()).unwrap();
        let profile =
            ProfileKey::new(Lane::Decisions, "fixture-decision").expect("valid fixture profile");
        let mut questions = Map::new();
        questions.insert(
            "route".into(),
            json!({"type": "choice", "criteria": {"yes": "Proceed", "no": "Stop"}}),
        );
        let request = InferenceRequest::Decision {
            profile,
            state: json!({"ready": true}),
            questions,
            images: Vec::new(),
        };
        let launch_marker = decisions
            .join("decision-models/runner.py")
            .with_extension("py.calls");
        (root, catalog, request, launch_marker)
    }

    fn context<'a>(
        cancellation: &'a dyn super::super::runtime::Cancellation,
    ) -> OperationContext<'a> {
        OperationContext::new(Instant::now() + Duration::from_secs(5), cancellation)
    }

    fn breeze_fixture(profile_id: &str, device: &str) -> (TempDir, Catalog) {
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("toolkits/l2m2");
        let decisions = root.path().join("toolkits/l3ds");
        fs::create_dir_all(media.join("media-models")).unwrap();
        fs::create_dir_all(&decisions).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-chat:\n    cmd: server\n",
        )
        .unwrap();
        fs::write(
            media.join("media-runtimes.json"),
            json!({
                "schema_version": 1,
                "profiles": [{
                    "id": profile_id,
                    "name": "Breeze fixture",
                    "runtime": "audio.cpp",
                    "device": device,
                    "tasks": ["tts", "clon", "design"],
                    "script": "media-models/generate-breeze-tts-2.sh"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            media.join("media-models/generate-breeze-tts-2.sh"),
            r#"#!/usr/bin/env bash
set -euo pipefail
output=""
out_dir=""
backend=""
task=""
instruction=""
mode="offline"
voice_ref=""
reference_text=""
while (($#)); do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --instruction) instruction="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --voice-ref) voice_ref="$2"; shift 2 ;;
    --reference-text) reference_text="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ "${CUDA_VISIBLE_DEVICES-unset}" == "" ]]
[[ "$backend" == "cpu" ]]
[[ "$output" == "$out_dir/result.wav" ]]
if [[ "$task" == "design" ]]; then
  [[ "$instruction" == "fixture voice direction" ]]
elif [[ "$task" == "clon" ]]; then
  [[ -n "$voice_ref" && -f "$voice_ref" && "$reference_text" == "fixture transcript" ]]
else
  [[ "$task" == "tts" && -z "$instruction" ]]
fi
printf 'RIFF\046\000\000\000WAVEfmt \020\000\000\000\001\000\001\000\300\135\000\000\200\273\000\000\002\000\020\000data\002\000\000\000\000\000' > "$output"
if [[ "$mode" == "streaming" ]]; then
  mkdir "$output.chunks"
  cp "$output" "$output.chunks/chunk_0001.wav"
fi
"#,
        )
        .unwrap();
        fs::set_permissions(
            media.join("media-models/generate-breeze-tts-2.sh"),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        fs::write(
            decisions.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        (root, catalog)
    }

    fn cpu_breeze_fixture() -> (TempDir, Catalog) {
        breeze_fixture(CPU_BREEZE_PROFILE_ID, "cpu")
    }

    fn cpu_breeze_request(mode: SpeechMode, root: &std::path::Path) -> InferenceRequest {
        let voice_reference = if mode == SpeechMode::VoiceClone {
            let path = root.join("speaker.wav");
            fs::write(&path, b"fixture reference").unwrap();
            Some(path)
        } else {
            None
        };
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, CPU_BREEZE_PROFILE_ID).unwrap(),
            input: super::super::requests::MediaInput::Speech(
                super::super::requests::SpeechInput {
                    text: "fixture speech".into(),
                    mode,
                    instruction: (mode == SpeechMode::VoiceDesign)
                        .then(|| "fixture voice direction".into()),
                    voice_reference,
                    reference_text: (mode == SpeechMode::VoiceClone)
                        .then(|| "fixture transcript".into()),
                },
            ),
        }
    }

    fn cpu_breeze_output(root: &std::path::Path, name: &str) -> std::path::PathBuf {
        let output = root.join(name);
        fs::create_dir(&output).unwrap();
        fs::set_permissions(&output, fs::Permissions::from_mode(0o700)).unwrap();
        output
    }

    fn embedding_fixture() -> (TempDir, Catalog, InferenceRequest) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("toolkits/l2m2")).unwrap();
        fs::create_dir_all(root.path().join("toolkits/l3ds")).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-embeddings:\n    cmd: |\n      llama-server --embedding\n",
        )
        .unwrap();
        fs::write(
            root.path().join("toolkits/l2m2/media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        fs::write(
            root.path().join("toolkits/l3ds/decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        let request = InferenceRequest::Embeddings {
            profile: ProfileKey::new(Lane::Language, "fixture-embeddings").unwrap(),
            input: vec!["alpha".into(), "beta".into()],
        };
        (root, catalog, request)
    }

    fn accept_bounded(listener: &TcpListener) -> TcpStream {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "embedding request not received");
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("fixture listener failed: {error}"),
            }
        }
    }

    fn read_embedding_request(stream: &mut TcpStream) -> serde_json::Value {
        read_json_request(stream, "POST /v1/embeddings HTTP/1.1")
    }

    fn read_json_request(stream: &mut TcpStream, request_line: &str) -> serde_json::Value {
        let mut header = Vec::new();
        let mut byte = [0];
        while !header.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            header.push(byte[0]);
            assert!(header.len() < 16_384, "request headers are bounded");
        }
        let header = String::from_utf8(header).unwrap();
        assert!(
            header.contains(request_line),
            "unexpected request: {header}"
        );
        let length = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        assert!(length <= 1024 * 1024 + 4096);
        let mut body = vec![0; length];
        stream.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn write_http_response(stream: &mut TcpStream, status: &str, body: &str) {
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
    }

    fn assert_no_connection(listener: &TcpListener) {
        match listener.accept() {
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Ok(_) => panic!("direct embeddings unexpectedly dispatched HTTP"),
            Err(error) => panic!("fixture listener failed: {error}"),
        }
    }

    fn native_response(engine: &str) -> String {
        serde_json::to_string(&json!({
            "engine": engine,
            "answers": {"route": {"choice": "yes"}},
            "latency_ms": 17.25,
            "usage": {"fixture_count": 1}
        }))
        .unwrap()
    }

    #[test]
    fn routed_speech_invalid_output_never_dispatches() {
        let (root, catalog) = breeze_fixture("breeze-tts-2", "cuda");
        let mut request = cpu_breeze_request(SpeechMode::Standard, root.path());
        let InferenceRequest::Media { profile, .. } = &mut request else {
            unreachable!()
        };
        *profile = ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let adapter =
            LanguageAdapter::new(&format!("http://{}", listener.local_addr().unwrap()), None)
                .unwrap();
        let occupied = cpu_breeze_output(root.path(), "occupied");
        fs::write(occupied.join("result.wav"), b"keep").unwrap();
        for output in [root.path().join("missing"), occupied.clone()] {
            let error = run_native(
                &catalog,
                &request,
                Some(&output),
                Some(DirectConnection::Language(&adapter)),
                &context(&NeverCancel),
            )
            .err()
            .unwrap();
            assert!(!error.blocks_following_work());
            assert_no_connection(&listener);
        }
        assert_eq!(fs::read(occupied.join("result.wav")).unwrap(), b"keep");
    }

    #[test]
    fn direct_runner_accepts_supported_cpu_breeze_modes() {
        let (root, catalog) = cpu_breeze_fixture();
        for (mode, name) in [
            (SpeechMode::Standard, "cpu-standard"),
            (SpeechMode::VoiceDesign, "cpu-design"),
            (SpeechMode::VoiceClone, "cpu-clone"),
            (SpeechMode::Streaming, "cpu-streaming"),
        ] {
            let request = cpu_breeze_request(mode, root.path());
            let output = cpu_breeze_output(root.path(), name);
            let DirectRunResult::Speech(result) = run_native(
                &catalog,
                &request,
                Some(&output),
                None,
                &context(&NeverCancel),
            )
            .unwrap() else {
                panic!("CPU Breeze direct mode returns validated speech artifacts")
            };
            assert_eq!(result.mode(), mode);
            assert_eq!(
                result.artifacts().len(),
                if mode == SpeechMode::Streaming { 2 } else { 1 }
            );
            assert!(output.join("result.wav").is_file());
            if mode == SpeechMode::Streaming {
                assert_eq!(
                    result.artifacts()[1].relative_path(),
                    std::path::Path::new("result.wav.chunks/chunk_0001.wav")
                );
            }
        }
    }

    #[test]
    fn direct_runner_returns_native_output_once_without_replay() {
        let (_root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let DirectRunResult::Decision(result) =
            run_native(&catalog, &request, None, None, &context(&NeverCancel)).unwrap()
        else {
            panic!("direct decision returns native decision result")
        };

        let InferenceResult::Decision { answers } = result.result else {
            panic!("decision runner returns native answers")
        };
        assert_eq!(answers["route"]["choice"], "yes");
        assert_eq!(
            result.native_latency,
            Some(Duration::from_nanos(17_250_000))
        );
        assert_eq!(result.native_usage.unwrap()["fixture_count"], 1);
        assert_eq!(fs::read(launch_marker).unwrap(), b"x");
    }

    #[test]
    fn direct_decision_profile_allowlist_is_exact() {
        let (_root, hosted, request, _) = fixture(
            "jev",
            "api",
            "remote",
            &native_response("systemone-official-api"),
        );
        let profile = hosted.inspect(request.profile()).unwrap();
        assert!(is_supported_direct_decision_profile(profile));

        let (_root, cpu, request, _) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        assert!(is_supported_direct_decision_profile(
            cpu.inspect(request.profile()).unwrap()
        ));

        for (engine, profile_type, device) in [
            ("jev", "api", "cuda"),
            ("nagi", "local", "cuda"),
            ("semif", "local", "cpu"),
        ] {
            let (_root, catalog, request, _) =
                fixture(engine, profile_type, device, &native_response("unused"));
            let profile = catalog.inspect(request.profile()).unwrap();
            assert!(!is_supported_direct_decision_profile(profile));
            assert_eq!(
                is_supported_cuda_direct_decision_profile(profile),
                engine == "nagi" && profile_type == "local" && device == "cuda"
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn direct_hosted_jev_requires_one_attempt_and_returns_a_valid_native_result() {
        if std::env::var_os(super::super::native_commands::HOSTED_DECISION_PYTHON_ENV).is_some() {
            return;
        }
        let response = native_response("systemone-official-api");
        let (root, catalog, request, launch_marker) = fixture("jev", "api", "remote", &response);
        let runner = root.path().join("toolkits/l3ds/decision-models/runner.py");
        let script = format!(
            "#!/bin/sh\nprintf x >> \"$0.calls\"\n[ \"$PYTHONDONTWRITEBYTECODE\" = 1 ] || exit 8\n[ -z \"${{TYPESAFE_LOG_LEVEL+x}}\" ] || exit 9\ncase \" $* \" in *\" --single-attempt \"*) ;; *) exit 10;; esac\ninput=$(/bin/cat)\ncase \"$input\" in *'\"state\"'*'\"questions\"'*) ;; *) exit 11;; esac\nprintf '%s\\n' '{}'\n",
            response.replace('\'', "'\\''")
        );
        fs::write(runner, script).unwrap();
        let DirectRunResult::Decision(result) =
            run_native(&catalog, &request, None, None, &context(&NeverCancel)).unwrap()
        else {
            panic!("hosted Jev returns the native decision result")
        };
        let InferenceResult::Decision { answers } = result.result else {
            panic!("hosted Jev returns decision answers")
        };
        assert_eq!(answers["route"]["choice"], "yes");
        assert_eq!(fs::read(launch_marker).unwrap(), b"x");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn direct_hosted_cancellation_after_spawn_is_remote_uncertain() {
        if std::env::var_os(super::super::native_commands::HOSTED_DECISION_PYTHON_ENV).is_some() {
            return;
        }
        let (root, catalog, request, launch_marker) = fixture(
            "jev",
            "api",
            "remote",
            &native_response("systemone-official-api"),
        );
        let runner = root.path().join("toolkits/l3ds/decision-models/runner.py");
        fs::write(
            runner,
            "#!/bin/sh\nprintf x >> \"$0.calls\"\n/bin/cat >/dev/null\nsleep 30\n",
        )
        .unwrap();

        let cancelled = Arc::new(AtomicBool::new(false));
        let watcher_cancelled = Arc::clone(&cancelled);
        let watcher_marker = launch_marker.clone();
        let watcher = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            while !watcher_marker.exists() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            watcher_cancelled.store(true, Ordering::Release);
        });
        let error = run_native(
            &catalog,
            &request,
            None,
            None,
            &context(&SharedCancel(cancelled)),
        )
        .err()
        .expect("hosted failure is not a successful result");
        watcher.join().unwrap();
        assert!(launch_marker.exists());
        assert_eq!(error.kind(), DirectRunErrorKind::RemoteCompletionUncertain);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::Uncertain)
        );
        assert!(error.blocks_following_work());
        assert!(!error.cancellation_confirmed());
    }

    #[test]
    fn direct_runner_rejects_cuda_before_launch() {
        let (_root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cuda", &native_response("nagi-native"));
        let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
            .err()
            .expect("CUDA route without an explicit device is rejected");
        assert_eq!(error.kind(), DirectRunErrorKind::UnsupportedRoute);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!error.blocks_following_work());
        assert!(!launch_marker.exists());
    }

    #[test]
    fn direct_runner_rejects_cuda_connection_for_cpu_and_hosted_decisions() {
        let device = CudaDevice::new("GPU-01234567-89ab-cdef-0123-456789abcdef").unwrap();

        let (_root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Cuda(&device)),
            &context(&NeverCancel),
        )
        .err()
        .expect("a CPU route must reject an explicit CUDA device");
        assert_eq!(error.kind(), DirectRunErrorKind::UnsupportedRoute);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!launch_marker.exists());

        let (_root, catalog, request, launch_marker) = fixture(
            "jev",
            "api",
            "remote",
            &native_response("systemone-official-api"),
        );
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Cuda(&device)),
            &context(&NeverCancel),
        )
        .err()
        .expect("a hosted route must reject a local CUDA device");
        assert_eq!(error.kind(), DirectRunErrorKind::UnsupportedRoute);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
        );
        assert!(!launch_marker.exists());
    }

    fn cuda_breeze_request(mode: SpeechMode) -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "breeze-tts-2").unwrap(),
            input: MediaInput::Speech(super::super::requests::SpeechInput {
                text: "fixture speech".into(),
                mode,
                instruction: None,
                voice_reference: None,
                reference_text: None,
            }),
        }
    }

    #[test]
    fn cuda_breeze_runs_through_the_router_with_no_device_or_gpu_window() {
        let (root, catalog) = breeze_fixture("breeze-tts-2", "cuda");
        let request = cuda_breeze_request(SpeechMode::Standard);
        let profile = catalog.inspect(request.profile()).unwrap();
        assert!(is_supported_cuda_breeze_profile(profile));
        assert!(
            !requires_operator_gpu_window(profile),
            "llama-swap owns the engine; no operator GPU window"
        );
        let route = resolve_profile_route(&catalog, request.profile()).unwrap();
        assert!(is_supported_cuda_breeze_route(&request, &route));

        // Without a router connection the request is refused before any script.
        let output = cpu_breeze_output(root.path(), "routed-missing-router");
        let error = run_native(
            &catalog,
            &request,
            Some(&output),
            None,
            &context(&NeverCancel),
        )
        .err()
        .expect("CUDA Breeze needs the router connection");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert!(error.is_remote());
        assert!(!error.blocks_following_work());

        // A pinned CUDA device is no longer a valid connection for this lane.
        let device = CudaDevice::new("GPU-01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let error = run_native(
            &catalog,
            &request,
            Some(&output),
            Some(DirectConnection::Cuda(&device)),
            &context(&NeverCancel),
        )
        .err()
        .expect("a device connection is refused");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert!(!root
            .path()
            .join("toolkits/l2m2/media-models/generate-breeze-tts-2.sh.calls")
            .exists());
        assert!(!output.join("result.wav").exists());

        let (cpu_root, cpu_catalog) = cpu_breeze_fixture();
        let cpu_request = cpu_breeze_request(SpeechMode::Standard, cpu_root.path());
        let cpu_output = cpu_breeze_output(cpu_root.path(), "cpu-with-cuda-device");
        let error = run_native(
            &cpu_catalog,
            &cpu_request,
            Some(&cpu_output),
            Some(DirectConnection::Cuda(&device)),
            &context(&NeverCancel),
        )
        .err()
        .expect("a CPU Breeze route must reject an explicit CUDA device");
        assert_eq!(error.kind(), DirectRunErrorKind::UnsupportedRoute);
        assert!(!cpu_output.join("result.wav").exists());
    }

    #[test]
    fn direct_runner_rejects_catalog_drift_before_launch() {
        let (root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let manifest_path = root.path().join("toolkits/l3ds/decision-runtimes.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["profiles"][0]["device"] = json!("cuda");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
            .err()
            .expect("catalog drift is rejected before dispatch");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!launch_marker.exists());
    }

    #[test]
    fn embedding_request_without_a_connection_does_not_launch_a_local_engine() {
        let (_root, catalog, _request, launch_marker) =
            fixture("nagi", "local", "cpu", &native_response("nagi-native"));
        let request = InferenceRequest::Embeddings {
            profile: ProfileKey::new(Lane::Language, "fixture-chat").unwrap(),
            input: vec!["private input".into()],
        };
        let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
            .err()
            .expect("embedding request without a connection is rejected");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
        );
        assert!(!error.blocks_following_work());
        assert!(!launch_marker.exists());
    }

    #[test]
    fn routed_cuda_breeze_publishes_validated_files_from_the_router_response() {
        let (root, catalog) = breeze_fixture("breeze-tts-2", "cuda");
        let request = cuda_breeze_request(SpeechMode::Standard);
        let output = cpu_breeze_output(root.path(), "routed-ok");
        let pcm: Vec<u8> = (0..2400)
            .map(|index| (((index as f32) * 0.3).sin() * 8000.0) as i16)
            .flat_map(i16::to_le_bytes)
            .collect();
        let wave = super::super::language::wav_from_pcm16(&pcm);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let served = wave.clone();
        let server = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_json_request(&mut stream, "POST /audiocpp/v1/audio/speech HTTP/1.1");
            assert_eq!(payload["model"], "breeze-tts");
            assert_eq!(payload["input"], "fixture speech");
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            )
            .unwrap();
            stream.write_all(&served).unwrap();
        });
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let result = run_native(
            &catalog,
            &request,
            Some(&output),
            Some(DirectConnection::Language(&adapter)),
            &context(&NeverCancel),
        )
        .unwrap();
        let DirectRunResult::Speech(result) = result else {
            panic!("routed Breeze returns the same typed speech result")
        };
        assert_eq!(result.mode(), SpeechMode::Standard);
        assert_eq!(result.artifacts().len(), 1);
        assert_eq!(fs::read(output.join("result.wav")).unwrap(), wave);
        // No script ran and nothing else was written.
        assert!(!root
            .path()
            .join("toolkits/l2m2/media-models/generate-breeze-tts-2.sh.calls")
            .exists());
        server.join().unwrap();
    }

    #[test]
    fn invalid_native_output_is_not_replayed() {
        let (_root, catalog, request, launch_marker) =
            fixture("nagi", "local", "cpu", "{\"engine\":\"wrong\"}");
        let error = run_native(&catalog, &request, None, None, &context(&NeverCancel))
            .err()
            .expect("invalid native output is rejected");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidOutput);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::ChildGroupReaped)
        );
        assert!(!error.blocks_following_work());
        assert_eq!(fs::read(launch_marker).unwrap(), b"x");
        assert!(!error.to_string().contains("wrong"));
    }

    #[test]
    fn router_decisions_do_not_stop_the_router_or_launch_a_native_child() {
        let (_root, catalog, request, marker) = fixture("llamacpp", "local", "cuda", "unused");
        let profile = catalog.inspect(request.profile()).unwrap();
        assert!(is_llamacpp_decision_profile(profile));
        assert!(is_supported_direct_decision_profile(profile));
        assert!(!requires_operator_gpu_window(profile));
        let error =
            run_native_in_operator_window(&catalog, &request, None, None, &context(&NeverCancel))
                .err()
                .expect("HTTP connection is required before dispatch");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert!(error.is_remote());
        assert!(!error.blocks_following_work());
        assert!(!marker.exists());
    }

    #[test]
    fn direct_runner_executes_typed_embeddings_through_existing_adapter() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_embedding_request(&mut stream);
            assert_eq!(payload["model"], "fixture-embeddings");
            assert_eq!(payload["input"], json!(["alpha", "beta"]));
            assert_eq!(payload["encoding_format"], "float");
            let body = r#"{"object":"list","model":"fixture-embeddings","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]},{"object":"embedding","index":1,"embedding":[0.3,0.4]}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#;
            write_http_response(&mut stream, "200 OK", body);
        });
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let result = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Language(&adapter)),
            &context(&NeverCancel),
        )
        .unwrap();
        let DirectRunResult::Embeddings(result) = result else {
            panic!("language embeddings return their typed payload")
        };
        assert_eq!(result.vectors, [vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert_eq!(result.usage.prompt_tokens, 2);
        assert_eq!(result.usage.total_tokens, 2);
        server.join().unwrap();
    }

    #[test]
    fn direct_runner_confirms_embedding_cancellation_only_before_dispatch() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Language(&adapter)),
            &context(&AlwaysCancel),
        )
        .err()
        .expect("cancelled embedding request is not dispatched");
        assert_eq!(error.kind(), DirectRunErrorKind::Cancelled);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
        );
        assert!(error.cancellation_confirmed());
        assert!(!error.blocks_following_work());
        assert_no_connection(&listener);
    }

    #[test]
    fn dispatched_embedding_cancellation_is_remote_uncertainty() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let cancelled = Arc::new(AtomicBool::new(false));
        let server_cancelled = cancelled.clone();
        let server = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_embedding_request(&mut stream);
            assert_eq!(payload["input"], json!(["alpha", "beta"]));
            server_cancelled.store(true, Ordering::Release);
            std::thread::sleep(Duration::from_millis(100));
        });
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let cancellation = SharedCancel(cancelled);
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &cancellation);
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Language(&adapter)),
            &context,
        )
        .err()
        .expect("dispatched cancellation remains uncertain");
        assert_eq!(error.kind(), DirectRunErrorKind::RemoteCompletionUncertain);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::Uncertain)
        );
        assert!(!error.cancellation_confirmed());
        assert!(error.blocks_following_work());
        assert_eq!(
            error.to_string(),
            "remote completion is unconfirmed; stop before continuing"
        );
        assert!(!error.to_string().contains("process cleanup"));
        server.join().unwrap();
    }

    #[test]
    fn uncertain_embedding_response_is_not_retried() {
        let (_root, catalog, request) = embedding_fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let no_retry_probe = listener.try_clone().unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut stream = accept_bounded(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_embedding_request(&mut stream);
            assert_eq!(payload["model"], "fixture-embeddings");
            write_http_response(&mut stream, "503 Service Unavailable", "private body");
        });
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Language(&adapter)),
            &context(&NeverCancel),
        )
        .err()
        .expect("server rejection after dispatch is uncertain");
        assert_eq!(error.kind(), DirectRunErrorKind::RemoteCompletionUncertain);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::Uncertain)
        );
        assert!(error.blocks_following_work());
        assert!(!error.cancellation_confirmed());
        assert_eq!(
            error.to_string(),
            "remote completion is unconfirmed; stop before continuing"
        );
        server.join().unwrap();
        assert_no_connection(&no_retry_probe);
    }

    #[test]
    fn direct_runner_rejects_embedding_catalog_drift_before_http_dispatch() {
        let (root, catalog, request) = embedding_fixture();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-embeddings:\n    cmd: llama-server --port 8080\n",
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let error = run_native(
            &catalog,
            &request,
            None,
            Some(DirectConnection::Language(&adapter)),
            &context(&NeverCancel),
        )
        .err()
        .expect("catalog drift is rejected before HTTP dispatch");
        assert_eq!(error.kind(), DirectRunErrorKind::InvalidRequest);
        assert_eq!(
            error.disposition(),
            DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
        );
        assert!(!error.blocks_following_work());
        assert_no_connection(&listener);
    }

    #[test]
    fn uncertain_cleanup_blocks_following_work() {
        let error = DirectRunError::new(
            DirectRunErrorKind::OwnershipUncertain,
            ProcessFailureDisposition::OwnershipUncertain,
        );
        assert!(error.blocks_following_work());
        assert!(!error.cancellation_confirmed());
        assert_eq!(
            error.to_string(),
            "native process cleanup is uncertain; stop before continuing"
        );
    }

    #[test]
    fn undispatched_errors_classify_cancellation_deadline_and_setup_failure() {
        let cancelled = execution_error(
            DecisionExecutionError::NotDispatched,
            &context(&AlwaysCancel),
        );
        assert_eq!(cancelled.kind(), DirectRunErrorKind::Cancelled);
        assert_eq!(
            cancelled.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(cancelled.cancellation_confirmed());

        let expired_context = OperationContext::new(Instant::now(), &NeverCancel);
        let expired = execution_error(DecisionExecutionError::NotDispatched, &expired_context);
        assert_eq!(expired.kind(), DirectRunErrorKind::DeadlineExceeded);
        assert_eq!(
            expired.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!expired.cancellation_confirmed());

        let setup_failure = execution_error(
            DecisionExecutionError::NotDispatched,
            &context(&NeverCancel),
        );
        assert_eq!(setup_failure.kind(), DirectRunErrorKind::ExecutionFailed);
        assert_eq!(
            setup_failure.disposition(),
            DirectRunDisposition::Process(ProcessFailureDisposition::NoChildSpawned)
        );
        assert!(!setup_failure.cancellation_confirmed());
    }

    fn routed_audio_script_fixture(
        profile: &str,
        task: &str,
    ) -> (TempDir, Catalog, InferenceRequest) {
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("toolkits/l2m2");
        fs::create_dir_all(media.join("media-models")).unwrap();
        fs::create_dir_all(root.path().join("toolkits/l3ds")).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  fixture-chat:\n    cmd: server\n",
        )
        .unwrap();
        fs::write(
            root.path().join("toolkits/l3ds/decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[]}"#,
        )
        .unwrap();
        let script = if profile == "heartmula-music" {
            "media-models/generate-heartmula-music.sh"
        } else {
            "media-models/generate-minimax-h3.sh"
        };
        fs::write(media.join(script), "#!/bin/sh\nexit 99\n").unwrap();
        fs::write(media.join("media-runtimes.json"), json!({
            "schema_version": 1,
            "profiles": [{"id": profile, "name": "Audio routing fixture", "runtime": "audio.cpp",
                "device": "cuda", "tasks": if profile == "heartmula-music" { vec!["music"] } else { vec!["music", "tts", "video"] },
                "script": script}]
        }).to_string()).unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        let request = InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, profile).unwrap(),
            input: MediaInput::Script {
                task: super::super::catalog::Task::new(task).unwrap(),
                args: vec!["--prompt".into(), "fixture audio".into()],
            },
        };
        (root, catalog, request)
    }

    #[test]
    fn routed_audio_scripts_publish_through_the_operator_entry_without_native_gpu_admission() {
        for (profile, task) in [
            ("heartmula-music", "music"),
            ("minimax-h3", "music"),
            ("minimax-h3", "tts"),
        ] {
            let (root, catalog, request) = routed_audio_script_fixture(profile, task);
            let output = cpu_breeze_output(root.path(), "routed-audio");
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let probe = listener.try_clone().unwrap();
            let endpoint = format!("http://{}/prefix", listener.local_addr().unwrap());
            let server = std::thread::spawn(move || {
                let mut stream = accept_bounded(&listener);
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut header = Vec::new();
                let mut byte = [0];
                while !header.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    header.push(byte[0]);
                    assert!(header.len() < 16_384);
                }
                let header = String::from_utf8(header).unwrap();
                assert!(header.starts_with("POST /prefix/audiocpp/v1/tasks/run HTTP/1.1\r\n"));
                assert!(header
                    .lines()
                    .any(|line| line
                        .eq_ignore_ascii_case("authorization: Bearer fixture-audio-secret")));
                let length = header
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                assert!(length < 16_384);
                let mut body = vec![0; length];
                stream.read_exact(&mut body).unwrap();
                let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(payload["model"], profile);
                assert_eq!(payload["request"]["text"], "fixture audio");
                let response = json!({"audio":"UklGRigAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQQAAAAAAAEA",
                    "sample_rate":24000,"channels":1}).to_string();
                write_http_response(&mut stream, "200 OK", &response);
            });
            let adapter = LanguageAdapter::new(&endpoint, Some("fixture-audio-secret")).unwrap();
            let result = run_native_in_operator_window(
                &catalog,
                &request,
                Some(&output),
                Some(DirectConnection::Language(&adapter)),
                &context(&NeverCancel),
            )
            .unwrap();
            let DirectRunResult::ScriptMedia(result) = result else {
                panic!("expected script media result")
            };
            assert_eq!(result.artifacts().len(), 1);
            assert_eq!(
                fs::read(output.join("result.wav")).unwrap(),
                super::super::language::wav_from_pcm16(&[0, 0, 1, 0])
            );
            server.join().unwrap();
            assert_no_connection(&probe);
        }
    }

    #[test]
    fn routed_audio_scripts_reject_existing_output_before_http_and_keep_h3_video_direct() {
        for profile in ["heartmula-music", "minimax-h3"] {
            let (root, catalog, request) = routed_audio_script_fixture(profile, "music");
            let output = cpu_breeze_output(root.path(), "occupied-audio");
            fs::write(output.join("result.wav"), b"existing artifact").unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let adapter =
                LanguageAdapter::new(&format!("http://{}", listener.local_addr().unwrap()), None)
                    .unwrap();
            let error = run_native_in_operator_window(
                &catalog,
                &request,
                Some(&output),
                Some(DirectConnection::Language(&adapter)),
                &context(&NeverCancel),
            )
            .err()
            .unwrap();
            assert_eq!(
                error.disposition(),
                DirectRunDisposition::Remote(DispatchDisposition::NoDispatch)
            );
            assert_no_connection(&listener);
            assert_eq!(
                fs::read(output.join("result.wav")).unwrap(),
                b"existing artifact"
            );
        }
        let (_root, catalog, request) = routed_audio_script_fixture("minimax-h3", "video");
        assert!(!is_routed_audio_script_request(&catalog, &request));
        assert!(requires_operator_gpu_window(
            catalog.inspect(request.profile()).unwrap()
        ));
    }
}
