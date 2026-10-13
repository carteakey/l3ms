//! Private, write-once request and native-result packets for one owned local
//! Breeze or local-decision job.
//!
//! This module does not start or stop services, admit GPU work, or establish
//! cgroup/GPU release. A result packet means only that native execution has a
//! typed result candidate. The caller must still stop the exact owned service
//! and complete its independent cleanup proofs.
//!
//! A caller provides a fresh existing per-job directory. This module opens and
//! pins that directory without following any path-component symlinks, requires
//! exact current-UID mode 0700, and writes bounded mode-0600 packets with
//! create-exclusive semantics. Native input is kept only in the private
//! request packet and in process memory; errors and Debug output are redacted.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::{CStr, CString, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};

use super::catalog::{Catalog, Device, Lane, ProfileKey, Readiness, RuntimeIdentity, Task};
use super::cuda_device::CudaDevice;
use super::decision::NativeDecisionResult;
use super::gpu_probe::GpuProbeConfig;
use super::media_execution::{MediaArtifactKind, MediaExecutionResult};
use super::native_commands::{
    prepare_breeze_speech_recipe_for_route, prepare_native_command_for_route,
    NativeOutputExpectation,
};
use super::requests::{InferenceRequest, MediaInput, SpeechInput, SpeechMode};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::RecoveryPolicy;

const REQUEST_PACKET_NAME: &str = "l3ms-job-request.json";
const RESULT_PACKET_NAME: &str = "l3ms-job-result.json";
const EXECUTION_CLAIM_NAME: &str = "l3ms-native-started.json";
const SCHEMA_VERSION: u32 = 1;
const DIRECTORY_MODE: u32 = 0o700;
const PACKET_MODE: u32 = 0o600;

/// Request content remains capped at the request module's one-MiB budget. The
/// packet ceiling adds only bounded route and envelope metadata.
pub(crate) const MAX_REQUEST_PACKET_BYTES: usize = 2 * 1024 * 1024;
/// Native decision stdout is capped at one MiB (with a separate bounded
/// stderr stream); two MiB leaves room for the typed result envelope.
pub(crate) const MAX_RESULT_PACKET_BYTES: usize = 2 * 1024 * 1024;
const MAX_TASKS: usize = 256;
const MAX_ARTIFACTS: usize = 64;
const MAX_ARTIFACT_PATH_BYTES: usize = 2048;
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOTAL_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_EXECUTION_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_NATIVE_LATENCY: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_USAGE_BYTES: usize = 1024 * 1024;

/// Fixed failures. No path, request content, route path, or OS error is kept.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnedJobStoreError {
    UnsafeDirectory,
    UnexpectedContents,
    InvalidIdentity,
    InvalidRequest,
    UnsupportedRequest,
    InvalidRoute,
    RouteChanged,
    InvalidDeviceSelection,
    InvalidBounds,
    InvalidPacket,
    InvalidResult,
    PacketTooLarge,
    RequestUnavailable,
    ResultNotReady,
    DuplicatePacket,
    WriteUncertain,
    ReadFailed,
    Io,
}

impl fmt::Display for OwnedJobStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsafeDirectory => "owned job directory is unavailable or unsafe",
            Self::UnexpectedContents => "owned job directory has unexpected contents",
            Self::InvalidIdentity => "owned job identity is invalid",
            Self::InvalidRequest => "owned job request is invalid",
            Self::UnsupportedRequest => "owned job request is not a supported local operation",
            Self::InvalidRoute => "owned job route is invalid",
            Self::RouteChanged => "owned job route changed after acceptance",
            Self::InvalidDeviceSelection => "owned job device selection is invalid",
            Self::InvalidBounds => "owned job execution bounds are invalid",
            Self::InvalidPacket => "owned job packet is invalid",
            Self::InvalidResult => "owned native result is invalid",
            Self::PacketTooLarge => "owned job packet exceeds its size limit",
            Self::RequestUnavailable => "owned job request packet is unavailable",
            Self::ResultNotReady => "owned native result packet is not ready",
            Self::DuplicatePacket => "owned job packet already exists",
            Self::WriteUncertain => "owned job packet publication is uncertain",
            Self::ReadFailed => "owned job packet could not be read",
            Self::Io => "owned job storage operation failed",
        })
    }
}

impl Error for OwnedJobStoreError {}

/// Device choice is explicit and child-ready. CUDA selection always names one
/// validated physical GPU UUID; it never means “whatever CUDA sees”.
pub enum DeviceSelection {
    Cpu,
    Nvidia {
        device: CudaDevice,
        expected_index: u32,
        max_idle_mib: u64,
    },
}

impl DeviceSelection {
    pub fn nvidia(
        physical_uuid: &str,
        expected_index: u32,
        max_idle_mib: u64,
    ) -> Result<Self, OwnedJobStoreError> {
        let device = CudaDevice::new(physical_uuid)
            .map_err(|_| OwnedJobStoreError::InvalidDeviceSelection)?;
        let probe = GpuProbeConfig::new(physical_uuid.to_owned(), expected_index, max_idle_mib)
            .map_err(|_| OwnedJobStoreError::InvalidDeviceSelection)?;
        if device.gpu_uuid() != probe.gpu_uuid() {
            return Err(OwnedJobStoreError::InvalidDeviceSelection);
        }
        Ok(Self::Nvidia {
            device,
            expected_index,
            max_idle_mib,
        })
    }

    pub(crate) fn gpu_probe_config(&self) -> Result<Option<GpuProbeConfig>, OwnedJobStoreError> {
        match self {
            Self::Cpu => Ok(None),
            Self::Nvidia {
                device,
                expected_index,
                max_idle_mib,
            } => GpuProbeConfig::new(device.gpu_uuid().to_owned(), *expected_index, *max_idle_mib)
                .map(Some)
                .map_err(|_| OwnedJobStoreError::InvalidDeviceSelection),
        }
    }
}

impl fmt::Debug for DeviceSelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("DeviceSelection::Cpu"),
            Self::Nvidia { .. } => {
                formatter.write_str("DeviceSelection::Nvidia(<physical GPU selected>)")
            }
        }
    }
}

/// Bounded native-operation timing. This is packet metadata, not a service
/// stop timeout; cleanup receives an independent recovery context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobExecutionBounds {
    timeout: Duration,
    recovery: RecoveryPolicy,
}

impl JobExecutionBounds {
    pub fn new(timeout: Duration, recovery: RecoveryPolicy) -> Result<Self, OwnedJobStoreError> {
        if timeout.is_zero() || timeout > MAX_EXECUTION_TIMEOUT {
            return Err(OwnedJobStoreError::InvalidBounds);
        }
        Ok(Self { timeout, recovery })
    }

    pub(crate) fn timeout(self) -> Duration {
        self.timeout
    }

    pub(crate) fn recovery(self) -> RecoveryPolicy {
        self.recovery
    }
}

/// Input to the single write-once request operation. No Debug implementation
/// is provided because request values contain user payload.
pub struct OwnedJobRequestSpec<'a> {
    pub job_id: u64,
    pub request_id: &'a str,
    pub repository_root: &'a Path,
    pub request: &'a InferenceRequest,
    pub accepted_route: &'a ResolvedProfileRoute,
    pub device: DeviceSelection,
    pub bounds: JobExecutionBounds,
}

#[derive(Clone, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    length: u64,
    modified_seconds: i64,
    modified_nanos: i64,
    changed_seconds: i64,
    changed_nanos: i64,
}

#[derive(Clone, Eq, PartialEq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    mode: u32,
}

/// A pinned, existing, unique per-job directory. It never creates, removes,
/// or replaces the directory.
pub(crate) struct OwnedJobStore {
    directory: File,
    path: PathBuf,
    identity: DirectoryIdentity,
    uid: u32,
}

impl fmt::Debug for OwnedJobStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OwnedJobStore(<pinned private directory>)")
    }
}

/// A request packet that has passed request, route, entrypoint, and device
/// checks. It retains the packet bytes and file identity in memory so later
/// result publication can reject in-place edits or file replacement. Debug is
/// intentionally redacted.
pub(crate) struct ValidatedOwnedJob {
    job_id: u64,
    request_id: String,
    repository_root: PathBuf,
    request: InferenceRequest,
    route: ResolvedProfileRoute,
    device: DeviceSelection,
    bounds: JobExecutionBounds,
    directory_identity: DirectoryIdentity,
    request_identity: FileIdentity,
    request_bytes: Vec<u8>,
}

impl ValidatedOwnedJob {
    pub(crate) fn job_id(&self) -> u64 {
        self.job_id
    }

    pub(crate) fn request_id(&self) -> &str {
        &self.request_id
    }

    pub(crate) fn repository_root(&self) -> &Path {
        &self.repository_root
    }

    pub(crate) fn request(&self) -> &InferenceRequest {
        &self.request
    }

    pub(crate) fn route(&self) -> &ResolvedProfileRoute {
        &self.route
    }

    pub(crate) fn device(&self) -> &DeviceSelection {
        &self.device
    }

    pub(crate) fn bounds(&self) -> JobExecutionBounds {
        self.bounds
    }
}

impl fmt::Debug for ValidatedOwnedJob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedOwnedJob")
            .field("job_id", &self.job_id)
            .field("request_id", &"<redacted>")
            .field("request", &"<redacted>")
            .field("route", &"<redacted>")
            .field("device", &self.device)
            .field("bounds", &self.bounds)
            .finish()
    }
}

/// Typed native outcome persisted by the wrapper. A present result is not a
/// stop receipt, cgroup observation, GPU-idle observation, or cleanup proof.
pub(crate) enum NativeJobResult {
    Decision(NativeDecisionResult),
    Breeze(BreezeArtifactReceipt),
    Failed(NativeJobFailure),
}

impl NativeJobResult {
    pub(crate) fn from_breeze(result: &MediaExecutionResult) -> Result<Self, OwnedJobStoreError> {
        let artifacts = result
            .artifacts()
            .iter()
            .map(|artifact| {
                let path = artifact
                    .relative_path()
                    .to_str()
                    .ok_or(OwnedJobStoreError::InvalidResult)?;
                let kind = match artifact.kind() {
                    MediaArtifactKind::FinalWave => BreezeArtifactKind::FinalWave,
                    MediaArtifactKind::StreamingChunk => BreezeArtifactKind::StreamingChunk,
                    MediaArtifactKind::RawRgb24Video
                    | MediaArtifactKind::Video
                    | MediaArtifactKind::Image => return Err(OwnedJobStoreError::InvalidResult),
                };
                Ok(BreezeArtifactRef {
                    kind,
                    relative_path: path.to_owned(),
                })
            })
            .collect::<Result<Vec<_>, OwnedJobStoreError>>()?;
        Ok(Self::Breeze(BreezeArtifactReceipt {
            mode: result.mode(),
            artifacts,
        }))
    }

    pub(crate) fn validate_for(&self, job: &ValidatedOwnedJob) -> Result<(), OwnedJobStoreError> {
        match (&job.request, self) {
            (InferenceRequest::Decision { .. }, Self::Decision(result)) => {
                result
                    .result
                    .validate_for(&job.request)
                    .map_err(|_| OwnedJobStoreError::InvalidResult)?;
                validate_decision_metadata(result)?;
                Ok(())
            }
            (
                InferenceRequest::Media {
                    input: MediaInput::Speech(speech),
                    ..
                },
                Self::Breeze(receipt),
            ) => {
                if receipt.mode != speech.mode {
                    return Err(OwnedJobStoreError::InvalidResult);
                }
                receipt.validate()?;
                Ok(())
            }
            (_, Self::Failed(_)) => Ok(()),
            _ => Err(OwnedJobStoreError::InvalidResult),
        }
    }
}

impl fmt::Debug for NativeJobResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decision(_) => formatter.write_str("NativeJobResult::Decision(<redacted>)"),
            Self::Breeze(_) => {
                formatter.write_str("NativeJobResult::Breeze(<artifact refs redacted>)")
            }
            Self::Failed(code) => formatter
                .debug_tuple("NativeJobResult::Failed")
                .field(code)
                .finish(),
        }
    }
}

/// Stable redacted failure categories for a native attempt. Cleanup is
/// deliberately not represented here.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NativeJobFailure {
    NotDispatched,
    Unavailable,
    TimedOut,
    Cancelled,
    InvalidOutput,
    Uncertain,
    Internal,
}

/// Validated Breeze artifacts are relative references only. Their root remains
/// the caller-owned job directory and must be checked again after cgroup stop.
pub(crate) struct BreezeArtifactReceipt {
    mode: SpeechMode,
    artifacts: Vec<BreezeArtifactRef>,
}

impl BreezeArtifactReceipt {
    pub(crate) fn mode(&self) -> SpeechMode {
        self.mode
    }

    pub(crate) fn artifacts(&self) -> &[BreezeArtifactRef] {
        &self.artifacts
    }

    fn validate(&self) -> Result<(), OwnedJobStoreError> {
        if self.artifacts.is_empty() || self.artifacts.len() > MAX_ARTIFACTS {
            return Err(OwnedJobStoreError::InvalidResult);
        }
        let mut paths = BTreeSet::new();
        let mut final_wave_count = 0usize;
        let mut chunk_count = 0usize;
        for artifact in &self.artifacts {
            validate_relative_artifact_path(&artifact.relative_path)?;
            if !paths.insert(artifact.relative_path.as_str()) {
                return Err(OwnedJobStoreError::InvalidResult);
            }
            match artifact.kind {
                BreezeArtifactKind::FinalWave => {
                    if artifact.relative_path != "result.wav" {
                        return Err(OwnedJobStoreError::InvalidResult);
                    }
                    final_wave_count += 1;
                }
                BreezeArtifactKind::StreamingChunk => {
                    let Some(name) = artifact.relative_path.strip_prefix("result.wav.chunks/")
                    else {
                        return Err(OwnedJobStoreError::InvalidResult);
                    };
                    if !safe_chunk_name(name) {
                        return Err(OwnedJobStoreError::InvalidResult);
                    }
                    chunk_count += 1;
                }
            }
        }
        if final_wave_count != 1
            || (self.mode == SpeechMode::Streaming && chunk_count == 0)
            || (self.mode != SpeechMode::Streaming && chunk_count != 0)
            || chunk_count > 63
        {
            return Err(OwnedJobStoreError::InvalidResult);
        }
        Ok(())
    }
}

impl fmt::Debug for BreezeArtifactReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BreezeArtifactReceipt")
            .field("mode", &self.mode)
            .field("artifact_count", &self.artifacts.len())
            .field("artifacts", &"<redacted>")
            .finish()
    }
}

pub(crate) struct BreezeArtifactRef {
    kind: BreezeArtifactKind,
    relative_path: String,
}

impl BreezeArtifactRef {
    pub(crate) fn kind(&self) -> BreezeArtifactKind {
        self.kind
    }

    pub(crate) fn relative_path(&self) -> &str {
        &self.relative_path
    }
}

impl fmt::Debug for BreezeArtifactRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BreezeArtifactRef")
            .field("kind", &self.kind)
            .field("relative_path", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BreezeArtifactKind {
    FinalWave,
    StreamingChunk,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireLane {
    Language,
    Media,
    Decisions,
}

impl WireLane {
    fn from_lane(lane: Lane) -> Self {
        match lane {
            Lane::Language => Self::Language,
            Lane::Media => Self::Media,
            Lane::Decisions => Self::Decisions,
        }
    }

    fn into_lane(self) -> Lane {
        match self {
            Self::Language => Lane::Language,
            Self::Media => Lane::Media,
            Self::Decisions => Lane::Decisions,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProfileKey {
    lane: WireLane,
    profile_id: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRuntimeIdentity {
    lane: WireLane,
    id: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDeclaredDevice {
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    other: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WireNativeRoute {
    MediaScript {
        path: String,
    },
    DecisionRunner {
        entrypoint: String,
        model: String,
        profile_type: String,
        tier: Option<String>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireReadiness {
    ManifestPresent,
    Installed,
    Healthy,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRouteSnapshot {
    profile: WireProfileKey,
    runtime: WireRuntimeIdentity,
    device: Option<WireDeclaredDevice>,
    native_route: WireNativeRoute,
    tasks: Vec<String>,
    readiness: WireReadiness,
}

impl WireRouteSnapshot {
    fn from_route(route: &ResolvedProfileRoute) -> Result<Self, OwnedJobStoreError> {
        if route.tasks.len() > MAX_TASKS {
            return Err(OwnedJobStoreError::InvalidRoute);
        }
        let native_route = match &route.native_route {
            NativeRoute::MediaScript { path } => WireNativeRoute::MediaScript {
                path: path
                    .to_str()
                    .ok_or(OwnedJobStoreError::InvalidRoute)?
                    .to_owned(),
            },
            NativeRoute::DecisionRunner {
                entrypoint,
                model,
                profile_type,
                tier,
            } => WireNativeRoute::DecisionRunner {
                entrypoint: entrypoint
                    .to_str()
                    .ok_or(OwnedJobStoreError::InvalidRoute)?
                    .to_owned(),
                model: model.clone(),
                profile_type: profile_type.clone(),
                tier: tier.clone(),
            },
            NativeRoute::LlamaSwapModel { .. } => {
                return Err(OwnedJobStoreError::UnsupportedRequest);
            }
        };
        let device = route.device.as_ref().map(|device| match device {
            Device::Cpu => WireDeclaredDevice {
                kind: "cpu".to_owned(),
                other: None,
            },
            Device::Cuda => WireDeclaredDevice {
                kind: "cuda".to_owned(),
                other: None,
            },
            Device::Remote => WireDeclaredDevice {
                kind: "remote".to_owned(),
                other: None,
            },
            Device::Other(value) => WireDeclaredDevice {
                kind: "other".to_owned(),
                other: Some(value.clone()),
            },
        });
        Ok(Self {
            profile: WireProfileKey {
                lane: WireLane::from_lane(route.profile.lane),
                profile_id: route.profile.profile_id.as_str().to_owned(),
            },
            runtime: WireRuntimeIdentity {
                lane: WireLane::from_lane(route.runtime.lane),
                id: route.runtime.as_str().to_owned(),
            },
            device,
            native_route,
            tasks: route
                .tasks
                .iter()
                .map(|task| task.as_str().to_owned())
                .collect(),
            readiness: match route.readiness {
                Readiness::ManifestPresent => WireReadiness::ManifestPresent,
                Readiness::Installed => WireReadiness::Installed,
                Readiness::Healthy => WireReadiness::Healthy,
            },
        })
    }

    fn into_route(self) -> Result<ResolvedProfileRoute, OwnedJobStoreError> {
        if self.tasks.len() > MAX_TASKS {
            return Err(OwnedJobStoreError::InvalidRoute);
        }
        let profile = ProfileKey::new(self.profile.lane.into_lane(), self.profile.profile_id)
            .map_err(|_| OwnedJobStoreError::InvalidRoute)?;
        let runtime = RuntimeIdentity::new(self.runtime.lane.into_lane(), self.runtime.id)
            .map_err(|_| OwnedJobStoreError::InvalidRoute)?;
        let device = self
            .device
            .map(|device| match (device.kind.as_str(), device.other) {
                ("cpu", None) => Ok(Device::Cpu),
                ("cuda", None) => Ok(Device::Cuda),
                ("remote", None) => Ok(Device::Remote),
                ("other", Some(value)) if !value.trim().is_empty() => Ok(Device::Other(value)),
                _ => Err(OwnedJobStoreError::InvalidRoute),
            })
            .transpose()?;
        let native_route = match self.native_route {
            WireNativeRoute::MediaScript { path } => {
                let path = PathBuf::from(path);
                if !path.is_absolute() {
                    return Err(OwnedJobStoreError::InvalidRoute);
                }
                NativeRoute::MediaScript { path }
            }
            WireNativeRoute::DecisionRunner {
                entrypoint,
                model,
                profile_type,
                tier,
            } => {
                let entrypoint = PathBuf::from(entrypoint);
                if !entrypoint.is_absolute() {
                    return Err(OwnedJobStoreError::InvalidRoute);
                }
                NativeRoute::DecisionRunner {
                    entrypoint,
                    model,
                    profile_type,
                    tier,
                }
            }
        };
        let tasks = self
            .tasks
            .into_iter()
            .map(|task| Task::new(task).map_err(|_| OwnedJobStoreError::InvalidRoute))
            .collect::<Result<Vec<_>, _>>()?;
        let readiness = match self.readiness {
            WireReadiness::ManifestPresent => Readiness::ManifestPresent,
            WireReadiness::Installed => Readiness::Installed,
            WireReadiness::Healthy => Readiness::Healthy,
        };
        Ok(ResolvedProfileRoute {
            profile,
            runtime,
            device,
            native_route,
            tasks,
            readiness,
        })
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireSpeechMode {
    Standard,
    VoiceDesign,
    VoiceClone,
    Streaming,
}

impl From<SpeechMode> for WireSpeechMode {
    fn from(mode: SpeechMode) -> Self {
        match mode {
            SpeechMode::Standard => Self::Standard,
            SpeechMode::VoiceDesign => Self::VoiceDesign,
            SpeechMode::VoiceClone => Self::VoiceClone,
            SpeechMode::Streaming => Self::Streaming,
        }
    }
}

impl From<WireSpeechMode> for SpeechMode {
    fn from(mode: WireSpeechMode) -> Self {
        match mode {
            WireSpeechMode::Standard => Self::Standard,
            WireSpeechMode::VoiceDesign => Self::VoiceDesign,
            WireSpeechMode::VoiceClone => Self::VoiceClone,
            WireSpeechMode::Streaming => Self::Streaming,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpeechInput {
    text: String,
    mode: WireSpeechMode,
    instruction: Option<String>,
    voice_reference: Option<String>,
    reference_text: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WireOperation {
    BreezeSpeech {
        speech: WireSpeechInput,
    },
    LocalDecision {
        state: Value,
        questions: Map<String, Value>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInferenceRequest {
    profile: String,
    operation: WireOperation,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDeviceSelection {
    kind: String,
    physical_uuid: Option<String>,
    expected_index: Option<u32>,
    max_idle_mib: Option<u64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExecutionBounds {
    timeout_nanos: u64,
    recovery_timeout_nanos: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRequestPacket {
    schema_version: u32,
    job_id: u64,
    request_id: String,
    repository_root: String,
    route: WireRouteSnapshot,
    request: WireInferenceRequest,
    device: WireDeviceSelection,
    bounds: WireExecutionBounds,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireArtifactRef {
    kind: BreezeArtifactKind,
    relative_path: String,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WireNativeResult {
    Decision {
        answers: Map<String, Value>,
        native_latency_nanos: Option<u64>,
        native_usage: Option<Map<String, Value>>,
    },
    Breeze {
        mode: WireSpeechMode,
        artifacts: Vec<WireArtifactRef>,
    },
    Failed {
        code: NativeJobFailure,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResultPacket {
    schema_version: u32,
    job_id: u64,
    request_id: String,
    result: WireNativeResult,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExecutionClaim {
    schema_version: u32,
    job_id: u64,
    request_id: String,
}

impl WireInferenceRequest {
    fn from_request(request: &InferenceRequest) -> Result<Self, OwnedJobStoreError> {
        let operation = match request {
            InferenceRequest::Media {
                input: MediaInput::Speech(speech),
                ..
            } => WireOperation::BreezeSpeech {
                speech: WireSpeechInput {
                    text: speech.text.clone(),
                    mode: speech.mode.into(),
                    instruction: speech.instruction.clone(),
                    voice_reference: speech
                        .voice_reference
                        .as_ref()
                        .map(|path| {
                            path.to_str()
                                .map(str::to_owned)
                                .ok_or(OwnedJobStoreError::InvalidRequest)
                        })
                        .transpose()?,
                    reference_text: speech.reference_text.clone(),
                },
            },
            InferenceRequest::Decision {
                state, questions, ..
            } => WireOperation::LocalDecision {
                state: state.clone(),
                questions: questions.clone(),
            },
            _ => return Err(OwnedJobStoreError::UnsupportedRequest),
        };
        Ok(Self {
            profile: request.profile().to_string(),
            operation,
        })
    }

    fn into_request(self) -> Result<InferenceRequest, OwnedJobStoreError> {
        let profile =
            ProfileKey::from_str(&self.profile).map_err(|_| OwnedJobStoreError::InvalidRequest)?;
        match self.operation {
            WireOperation::BreezeSpeech { speech } => {
                if profile.lane != Lane::Media {
                    return Err(OwnedJobStoreError::InvalidRequest);
                }
                Ok(InferenceRequest::Media {
                    profile,
                    input: MediaInput::Speech(SpeechInput {
                        text: speech.text,
                        mode: speech.mode.into(),
                        instruction: speech.instruction,
                        voice_reference: speech.voice_reference.map(PathBuf::from),
                        reference_text: speech.reference_text,
                    }),
                })
            }
            WireOperation::LocalDecision { state, questions } => {
                if profile.lane != Lane::Decisions {
                    return Err(OwnedJobStoreError::InvalidRequest);
                }
                Ok(InferenceRequest::Decision {
                    profile,
                    state,
                    questions,
                })
            }
        }
    }
}

impl WireDeviceSelection {
    fn from_device(device: &DeviceSelection) -> Self {
        match device {
            DeviceSelection::Cpu => Self {
                kind: "cpu".to_owned(),
                physical_uuid: None,
                expected_index: None,
                max_idle_mib: None,
            },
            DeviceSelection::Nvidia {
                device,
                expected_index,
                max_idle_mib,
            } => Self {
                kind: "nvidia".to_owned(),
                physical_uuid: Some(device.gpu_uuid().to_owned()),
                expected_index: Some(*expected_index),
                max_idle_mib: Some(*max_idle_mib),
            },
        }
    }

    fn into_device(self) -> Result<DeviceSelection, OwnedJobStoreError> {
        match (
            self.kind.as_str(),
            self.physical_uuid,
            self.expected_index,
            self.max_idle_mib,
        ) {
            ("cpu", None, None, None) => Ok(DeviceSelection::Cpu),
            ("nvidia", Some(uuid), Some(index), Some(max_idle_mib)) => {
                DeviceSelection::nvidia(&uuid, index, max_idle_mib)
            }
            _ => Err(OwnedJobStoreError::InvalidDeviceSelection),
        }
    }
}

impl NativeJobResult {
    fn to_wire(&self) -> Result<WireNativeResult, OwnedJobStoreError> {
        match self {
            Self::Decision(result) => {
                validate_decision_metadata(result)?;
                let super::requests::InferenceResult::Decision { answers } = &result.result else {
                    return Err(OwnedJobStoreError::InvalidResult);
                };
                Ok(WireNativeResult::Decision {
                    answers: answers.clone(),
                    native_latency_nanos: result
                        .native_latency
                        .map(|latency| latency.as_nanos() as u64),
                    native_usage: result.native_usage.clone(),
                })
            }
            Self::Breeze(receipt) => {
                receipt.validate()?;
                Ok(WireNativeResult::Breeze {
                    mode: receipt.mode.into(),
                    artifacts: receipt
                        .artifacts
                        .iter()
                        .map(|artifact| WireArtifactRef {
                            kind: artifact.kind,
                            relative_path: artifact.relative_path.clone(),
                        })
                        .collect(),
                })
            }
            Self::Failed(code) => Ok(WireNativeResult::Failed { code: *code }),
        }
    }

    fn from_wire(wire: WireNativeResult) -> Result<Self, OwnedJobStoreError> {
        match wire {
            WireNativeResult::Decision {
                answers,
                native_latency_nanos,
                native_usage,
            } => {
                let result = NativeDecisionResult {
                    result: super::requests::InferenceResult::Decision { answers },
                    native_latency: native_latency_nanos.map(Duration::from_nanos),
                    native_usage,
                };
                validate_decision_metadata(&result)?;
                Ok(Self::Decision(result))
            }
            WireNativeResult::Breeze { mode, artifacts } => {
                if artifacts.len() > MAX_ARTIFACTS {
                    return Err(OwnedJobStoreError::InvalidResult);
                }
                let receipt = BreezeArtifactReceipt {
                    mode: mode.into(),
                    artifacts: artifacts
                        .into_iter()
                        .map(|artifact| BreezeArtifactRef {
                            kind: artifact.kind,
                            relative_path: artifact.relative_path,
                        })
                        .collect(),
                };
                receipt.validate()?;
                Ok(Self::Breeze(receipt))
            }
            WireNativeResult::Failed { code } => Ok(Self::Failed(code)),
        }
    }
}

fn validate_decision_metadata(result: &NativeDecisionResult) -> Result<(), OwnedJobStoreError> {
    if result
        .native_latency
        .is_some_and(|latency| latency > MAX_NATIVE_LATENCY)
    {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    if let Some(usage) = &result.native_usage {
        let bytes = serde_json::to_vec(usage).map_err(|_| OwnedJobStoreError::InvalidResult)?;
        if bytes.len() > MAX_USAGE_BYTES {
            return Err(OwnedJobStoreError::InvalidResult);
        }
    }
    Ok(())
}

fn validate_supported_request_route(
    request: &InferenceRequest,
    route: &ResolvedProfileRoute,
    device: &DeviceSelection,
) -> Result<(), OwnedJobStoreError> {
    if request.profile() != &route.profile {
        return Err(OwnedJobStoreError::InvalidRoute);
    }
    match request {
        InferenceRequest::Media {
            input: MediaInput::Speech(_),
            ..
        } => {
            let is_breeze_route = route.profile.lane == Lane::Media
                && route.profile.profile_id.as_str() == "breeze-tts-2"
                && route.runtime.lane == Lane::Media
                && route.runtime.as_str() == "audio.cpp"
                && matches!(route.device.as_ref(), Some(Device::Cuda))
                && matches!(&route.native_route, NativeRoute::MediaScript { .. });
            if !is_breeze_route || !matches!(device, DeviceSelection::Nvidia { .. }) {
                return Err(OwnedJobStoreError::UnsupportedRequest);
            }
        }
        InferenceRequest::Decision { .. } => {
            let NativeRoute::DecisionRunner { profile_type, .. } = &route.native_route else {
                return Err(OwnedJobStoreError::UnsupportedRequest);
            };
            if route.profile.lane != Lane::Decisions
                || route.runtime.lane != Lane::Decisions
                || !matches!(route.runtime.as_str(), "nagi" | "laya")
                || profile_type != "local"
            {
                return Err(OwnedJobStoreError::UnsupportedRequest);
            }
            match (route.device.as_ref(), device) {
                (Some(Device::Cpu), DeviceSelection::Cpu)
                | (Some(Device::Cuda), DeviceSelection::Nvidia { .. }) => {}
                _ => return Err(OwnedJobStoreError::InvalidDeviceSelection),
            }
        }
        _ => return Err(OwnedJobStoreError::UnsupportedRequest),
    }
    if let DeviceSelection::Nvidia { .. } = device {
        device.gpu_probe_config()?;
    }
    Ok(())
}

impl OwnedJobStore {
    /// Pin an existing current-user mode-0700 job directory. Every path
    /// component is opened with `O_NOFOLLOW`; the final descriptor is retained
    /// and its name is revalidated on every operation.
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self, OwnedJobStoreError> {
        let path = path.as_ref();
        let directory = open_directory_nofollow(path)?;
        let metadata = directory
            .metadata()
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        let uid = effective_uid();
        let mode = metadata.mode() & 0o7777;
        if !metadata.is_dir() || metadata.uid() != uid || mode != DIRECTORY_MODE {
            return Err(OwnedJobStoreError::UnsafeDirectory);
        }
        let identity = DirectoryIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
            uid,
            mode,
        };
        let store = Self {
            directory,
            path: path.to_path_buf(),
            identity,
            uid,
        };
        store.verify_directory()?;
        Ok(store)
    }

    pub(crate) fn directory_device(&self) -> u64 {
        self.identity.device
    }

    pub(crate) fn directory_inode(&self) -> u64 {
        self.identity.inode
    }

    /// Validate the request and accepted route against a fresh catalog and
    /// native entrypoint, then publish exactly one immutable request packet.
    pub(crate) fn write_request(
        &self,
        spec: OwnedJobRequestSpec<'_>,
    ) -> Result<ValidatedOwnedJob, OwnedJobStoreError> {
        self.verify_directory()?;
        self.require_empty_directory()?;
        if spec.job_id == 0 {
            return Err(OwnedJobStoreError::InvalidIdentity);
        }
        super::ipc::validate_request_id(spec.request_id)
            .map_err(|_| OwnedJobStoreError::InvalidIdentity)?;
        let repository_root = canonical_repository_root(spec.repository_root)?;
        let catalog =
            Catalog::load(&repository_root).map_err(|_| OwnedJobStoreError::InvalidRoute)?;
        validate_request_against_catalog(
            &catalog,
            spec.request,
            spec.accepted_route,
            &spec.device,
            spec.bounds,
            &self.path,
        )?;

        let route = WireRouteSnapshot::from_route(spec.accepted_route)?;
        let request = WireInferenceRequest::from_request(spec.request)?;
        let root_text = repository_root
            .to_str()
            .ok_or(OwnedJobStoreError::InvalidRoute)?
            .to_owned();
        let packet = WireRequestPacket {
            schema_version: SCHEMA_VERSION,
            job_id: spec.job_id,
            request_id: spec.request_id.to_owned(),
            repository_root: root_text,
            route,
            request,
            device: WireDeviceSelection::from_device(&spec.device),
            bounds: WireExecutionBounds {
                timeout_nanos: u64::try_from(spec.bounds.timeout.as_nanos())
                    .map_err(|_| OwnedJobStoreError::InvalidBounds)?,
                recovery_timeout_nanos: u64::try_from(spec.bounds.recovery.timeout().as_nanos())
                    .map_err(|_| OwnedJobStoreError::InvalidBounds)?,
            },
        };
        let bytes = serde_json::to_vec(&packet).map_err(|_| OwnedJobStoreError::InvalidPacket)?;
        if bytes.len() > MAX_REQUEST_PACKET_BYTES {
            return Err(OwnedJobStoreError::PacketTooLarge);
        }
        let request_identity = publish_exclusive(
            &self.directory,
            REQUEST_PACKET_NAME,
            &bytes,
            MAX_REQUEST_PACKET_BYTES,
        )?;
        if self.verify_directory().is_err() {
            return Err(OwnedJobStoreError::WriteUncertain);
        }
        let (published_bytes, published_identity) = self
            .read_packet(REQUEST_PACKET_NAME, MAX_REQUEST_PACKET_BYTES)
            .map_err(|_| OwnedJobStoreError::WriteUncertain)?;
        if published_bytes != bytes || published_identity != request_identity {
            return Err(OwnedJobStoreError::WriteUncertain);
        }
        Ok(ValidatedOwnedJob {
            job_id: spec.job_id,
            request_id: spec.request_id.to_owned(),
            repository_root,
            request: spec.request.clone(),
            route: spec.accepted_route.clone(),
            device: spec.device,
            bounds: spec.bounds,
            directory_identity: self.identity.clone(),
            request_identity,
            request_bytes: bytes,
        })
    }

    /// Re-read a request in a short-lived helper and revalidate the current
    /// catalog, route, native entrypoint, request shape and device. No packet
    /// state is considered a live runtime or cleanup acknowledgment.
    pub(crate) fn read_request(&self) -> Result<ValidatedOwnedJob, OwnedJobStoreError> {
        self.verify_directory()?;
        let (bytes, request_identity) = self
            .read_packet(REQUEST_PACKET_NAME, MAX_REQUEST_PACKET_BYTES)
            .map_err(|_| OwnedJobStoreError::RequestUnavailable)?;
        let packet: WireRequestPacket = decode_strict_packet(&bytes, PacketKind::Request)?;
        if packet.schema_version != SCHEMA_VERSION || packet.job_id == 0 {
            return Err(OwnedJobStoreError::InvalidPacket);
        }
        super::ipc::validate_request_id(&packet.request_id)
            .map_err(|_| OwnedJobStoreError::InvalidIdentity)?;
        let repository_root = canonical_repository_root(Path::new(&packet.repository_root))?;
        if repository_root.to_str() != Some(packet.repository_root.as_str()) {
            return Err(OwnedJobStoreError::InvalidPacket);
        }
        let route = packet.route.into_route()?;
        let request = packet.request.into_request()?;
        let device = packet.device.into_device()?;
        let bounds = JobExecutionBounds::new(
            Duration::from_nanos(packet.bounds.timeout_nanos),
            RecoveryPolicy::new(Duration::from_nanos(packet.bounds.recovery_timeout_nanos))
                .map_err(|_| OwnedJobStoreError::InvalidBounds)?,
        )?;
        let catalog =
            Catalog::load(&repository_root).map_err(|_| OwnedJobStoreError::InvalidRoute)?;
        validate_request_against_catalog(&catalog, &request, &route, &device, bounds, &self.path)?;
        let job = ValidatedOwnedJob {
            job_id: packet.job_id,
            request_id: packet.request_id,
            repository_root,
            request,
            route,
            device,
            bounds,
            directory_identity: self.identity.clone(),
            request_identity,
            request_bytes: bytes,
        };
        self.validate_request_directory_entries(&job)?;
        Ok(job)
    }

    /// Claim the request packet for its single native execution attempt.
    /// The exclusive, synced claim is permanent: an existing or uncertain
    /// claim must be reconciled, never retried or removed.
    pub(crate) fn claim_execution(
        &self,
        job: &ValidatedOwnedJob,
    ) -> Result<(), OwnedJobStoreError> {
        self.verify_job(job)?;
        let entries = directory_entry_names(&self.directory)
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        if entries
            .iter()
            .any(|entry| entry.to_str() == Some(EXECUTION_CLAIM_NAME))
        {
            self.validate_execution_claim(job)?;
            return Err(OwnedJobStoreError::DuplicatePacket);
        }
        if entries.len() != 1 || entries[0].to_str() != Some(REQUEST_PACKET_NAME) {
            return if entries
                .iter()
                .any(|entry| entry.to_str() == Some(RESULT_PACKET_NAME))
            {
                Err(OwnedJobStoreError::DuplicatePacket)
            } else {
                Err(OwnedJobStoreError::UnexpectedContents)
            };
        }
        let packet = WireExecutionClaim {
            schema_version: SCHEMA_VERSION,
            job_id: job.job_id,
            request_id: job.request_id.clone(),
        };
        let bytes = serde_json::to_vec(&packet).map_err(|_| OwnedJobStoreError::InvalidPacket)?;
        let identity = publish_exclusive(&self.directory, EXECUTION_CLAIM_NAME, &bytes, 4096)?;
        let (published, published_identity) = self
            .read_packet(EXECUTION_CLAIM_NAME, 4096)
            .map_err(|_| OwnedJobStoreError::WriteUncertain)?;
        if published != bytes || published_identity != identity {
            return Err(OwnedJobStoreError::WriteUncertain);
        }
        let entries = directory_entry_names(&self.directory)
            .map_err(|_| OwnedJobStoreError::WriteUncertain)?;
        if entries.len() != 2
            || !entries
                .iter()
                .any(|entry| entry.to_str() == Some(REQUEST_PACKET_NAME))
            || !entries
                .iter()
                .any(|entry| entry.to_str() == Some(EXECUTION_CLAIM_NAME))
        {
            return Err(OwnedJobStoreError::WriteUncertain);
        }
        self.verify_directory()
            .map_err(|_| OwnedJobStoreError::WriteUncertain)?;
        Ok(())
    }

    /// Persist a typed native outcome once. Success means this result file was
    /// synchronously published, not that its wrapper/cgroup/GPU was released.
    pub(crate) fn write_result(
        &self,
        job: &ValidatedOwnedJob,
        result: &NativeJobResult,
    ) -> Result<(), OwnedJobStoreError> {
        self.verify_job(job)?;
        if directory_entry_names(&self.directory)
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?
            .iter()
            .any(|entry| entry.to_str() == Some(RESULT_PACKET_NAME))
        {
            return Err(OwnedJobStoreError::DuplicatePacket);
        }
        result.validate_for(job)?;
        self.validate_result_directory(job, result, false)?;
        let packet = WireResultPacket {
            schema_version: SCHEMA_VERSION,
            job_id: job.job_id,
            request_id: job.request_id.clone(),
            result: result.to_wire()?,
        };
        let bytes = serde_json::to_vec(&packet).map_err(|_| OwnedJobStoreError::InvalidResult)?;
        if bytes.len() > MAX_RESULT_PACKET_BYTES {
            return Err(OwnedJobStoreError::PacketTooLarge);
        }
        publish_exclusive(
            &self.directory,
            RESULT_PACKET_NAME,
            &bytes,
            MAX_RESULT_PACKET_BYTES,
        )?;
        self.verify_directory()
            .map_err(|_| OwnedJobStoreError::WriteUncertain)?;
        self.read_packet(RESULT_PACKET_NAME, MAX_RESULT_PACKET_BYTES)
            .and_then(|(published, _)| {
                if published == bytes {
                    Ok(())
                } else {
                    Err(OwnedJobStoreError::WriteUncertain)
                }
            })
            .map_err(|_| OwnedJobStoreError::WriteUncertain)?;
        self.validate_result_directory(job, result, true)
            .map_err(|_| OwnedJobStoreError::WriteUncertain)?;
        Ok(())
    }

    /// Read and validate the native result packet. The result stays distinct
    /// from the owner-stop receipt and cgroup/GPU release proofs.
    pub(crate) fn read_result(
        &self,
        job: &ValidatedOwnedJob,
    ) -> Result<NativeJobResult, OwnedJobStoreError> {
        self.verify_job(job)?;
        let (bytes, _) = self
            .read_packet(RESULT_PACKET_NAME, MAX_RESULT_PACKET_BYTES)
            .map_err(|_| OwnedJobStoreError::ResultNotReady)?;
        let packet: WireResultPacket = decode_strict_packet(&bytes, PacketKind::Result)?;
        if packet.schema_version != SCHEMA_VERSION
            || packet.job_id != job.job_id
            || packet.request_id != job.request_id
        {
            return Err(OwnedJobStoreError::InvalidResult);
        }
        let result = NativeJobResult::from_wire(packet.result)?;
        result.validate_for(job)?;
        self.validate_result_directory(job, &result, true)?;
        Ok(result)
    }

    fn verify_job(&self, job: &ValidatedOwnedJob) -> Result<(), OwnedJobStoreError> {
        self.verify_directory()?;
        if job.directory_identity != self.identity {
            return Err(OwnedJobStoreError::InvalidIdentity);
        }
        let (bytes, identity) = self
            .read_packet(REQUEST_PACKET_NAME, MAX_REQUEST_PACKET_BYTES)
            .map_err(|_| OwnedJobStoreError::RequestUnavailable)?;
        if bytes != job.request_bytes || identity != job.request_identity {
            return Err(OwnedJobStoreError::InvalidPacket);
        }
        self.validate_execution_claim(job)?;
        Ok(())
    }

    fn validate_execution_claim(
        &self,
        job: &ValidatedOwnedJob,
    ) -> Result<bool, OwnedJobStoreError> {
        if !directory_entry_names(&self.directory)
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?
            .iter()
            .any(|entry| entry.to_str() == Some(EXECUTION_CLAIM_NAME))
        {
            return Ok(false);
        }
        let (bytes, _) = self
            .read_packet(EXECUTION_CLAIM_NAME, 4096)
            .map_err(|_| OwnedJobStoreError::InvalidPacket)?;
        let packet: WireExecutionClaim = decode_strict_packet(&bytes, PacketKind::Claim)?;
        if packet.schema_version != SCHEMA_VERSION
            || packet.job_id != job.job_id
            || packet.request_id != job.request_id
        {
            return Err(OwnedJobStoreError::InvalidIdentity);
        }
        Ok(true)
    }

    fn verify_directory(&self) -> Result<(), OwnedJobStoreError> {
        let pinned = self
            .directory
            .metadata()
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        if !pinned.is_dir()
            || pinned.dev() != self.identity.device
            || pinned.ino() != self.identity.inode
            || pinned.uid() != self.uid
            || (pinned.mode() & 0o7777) != self.identity.mode
        {
            return Err(OwnedJobStoreError::UnsafeDirectory);
        }
        let named =
            open_directory_nofollow(&self.path).map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        let named = named
            .metadata()
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        if named.dev() != self.identity.device
            || named.ino() != self.identity.inode
            || named.uid() != self.uid
            || (named.mode() & 0o7777) != self.identity.mode
        {
            return Err(OwnedJobStoreError::UnsafeDirectory);
        }
        Ok(())
    }

    fn require_empty_directory(&self) -> Result<(), OwnedJobStoreError> {
        let entries = directory_entry_names(&self.directory)
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        if entries.is_empty() {
            return Ok(());
        }
        if entries.iter().any(|entry| {
            matches!(
                entry.to_str(),
                Some(REQUEST_PACKET_NAME | RESULT_PACKET_NAME)
            )
        }) {
            return Err(OwnedJobStoreError::DuplicatePacket);
        }
        Err(OwnedJobStoreError::UnexpectedContents)
    }

    fn validate_request_directory_entries(
        &self,
        job: &ValidatedOwnedJob,
    ) -> Result<(), OwnedJobStoreError> {
        let entries = directory_entry_names(&self.directory)
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        let allowed = [
            REQUEST_PACKET_NAME,
            RESULT_PACKET_NAME,
            EXECUTION_CLAIM_NAME,
            "result.wav",
            "result.wav.chunks",
        ];
        if entries
            .iter()
            .any(|entry| entry.to_str().is_none_or(|name| !allowed.contains(&name)))
        {
            return Err(OwnedJobStoreError::UnexpectedContents);
        }
        self.validate_execution_claim(job)?;
        Ok(())
    }

    fn validate_result_directory(
        &self,
        job: &ValidatedOwnedJob,
        result: &NativeJobResult,
        has_result_packet: bool,
    ) -> Result<(), OwnedJobStoreError> {
        let entries = directory_entry_names(&self.directory)
            .map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        let mut allowed = BTreeSet::from([REQUEST_PACKET_NAME]);
        if has_result_packet {
            allowed.insert(RESULT_PACKET_NAME);
        }
        if self.validate_execution_claim(job)? {
            allowed.insert(EXECUTION_CLAIM_NAME);
        }
        let breeze = matches!(
            &job.request,
            InferenceRequest::Media {
                input: MediaInput::Speech(_),
                ..
            }
        );
        if breeze {
            let expected_success_artifacts = match result {
                NativeJobResult::Breeze(receipt) => Some(&receipt.artifacts),
                NativeJobResult::Decision(_) => return Err(OwnedJobStoreError::InvalidResult),
                NativeJobResult::Failed(_) => None,
            };
            let (needs_wave, expected_chunks) = if let Some(artifacts) = expected_success_artifacts
            {
                let wave = artifacts.iter().any(|artifact| {
                    artifact.kind == BreezeArtifactKind::FinalWave
                        && artifact.relative_path == "result.wav"
                });
                let chunks = artifacts
                    .iter()
                    .filter_map(|artifact| {
                        artifact
                            .relative_path
                            .strip_prefix("result.wav.chunks/")
                            .map(str::to_owned)
                    })
                    .collect::<BTreeSet<_>>();
                (wave, Some(chunks))
            } else {
                (false, None)
            };
            let mut artifact_bytes = 0u64;
            if needs_wave
                || entries
                    .iter()
                    .any(|entry| entry.to_str() == Some("result.wav"))
            {
                allowed.insert("result.wav");
                artifact_bytes = validate_regular_child(&self.directory, "result.wav", self.uid)?;
            }
            let chunks_present = entries
                .iter()
                .any(|entry| entry.to_str() == Some("result.wav.chunks"));
            if chunks_present
                || expected_chunks
                    .as_ref()
                    .is_some_and(|chunks| !chunks.is_empty())
            {
                allowed.insert("result.wav.chunks");
                let allowed_chunks = expected_chunks.as_ref();
                artifact_bytes = artifact_bytes.saturating_add(validate_chunk_directory(
                    &self.directory,
                    self.uid,
                    allowed_chunks,
                )?);
            }
            if artifact_bytes > MAX_TOTAL_ARTIFACT_BYTES {
                return Err(OwnedJobStoreError::InvalidResult);
            }
        } else if matches!(result, NativeJobResult::Breeze(_))
            || entries
                .iter()
                .any(|entry| matches!(entry.to_str(), Some("result.wav" | "result.wav.chunks")))
        {
            return Err(OwnedJobStoreError::UnexpectedContents);
        }
        if entries
            .iter()
            .any(|entry| entry.to_str().is_none_or(|name| !allowed.contains(name)))
        {
            return Err(OwnedJobStoreError::UnexpectedContents);
        }
        Ok(())
    }

    fn read_packet(
        &self,
        name: &str,
        limit: usize,
    ) -> Result<(Vec<u8>, FileIdentity), OwnedJobStoreError> {
        let c_name = CString::new(name).map_err(|_| OwnedJobStoreError::InvalidPacket)?;
        let fd = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                c_name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            )
        };
        if fd < 0 {
            return Err(OwnedJobStoreError::ReadFailed);
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let metadata = file
            .metadata()
            .map_err(|_| OwnedJobStoreError::ReadFailed)?;
        validate_packet_metadata(&metadata, self.uid, limit)?;
        let before = file_identity(&metadata);
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        Read::by_ref(&mut file)
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| OwnedJobStoreError::ReadFailed)?;
        let after_metadata = file
            .metadata()
            .map_err(|_| OwnedJobStoreError::ReadFailed)?;
        let after = file_identity(&after_metadata);
        if bytes.len() > limit || before != after {
            return Err(OwnedJobStoreError::ReadFailed);
        }
        let c_name = CString::new(name).map_err(|_| OwnedJobStoreError::InvalidPacket)?;
        let mut named_stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        let stat_result = unsafe {
            libc::fstatat(
                self.directory.as_raw_fd(),
                c_name.as_ptr(),
                named_stat.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if stat_result != 0 {
            return Err(OwnedJobStoreError::ReadFailed);
        }
        let named_stat = unsafe { named_stat.assume_init() };
        if named_stat.st_dev != after.device
            || named_stat.st_ino != after.inode
            || named_stat.st_size as u64 != after.length
            || named_stat.st_mtime != after.modified_seconds
            || named_stat.st_mtime_nsec != after.modified_nanos
            || named_stat.st_ctime != after.changed_seconds
            || named_stat.st_ctime_nsec != after.changed_nanos
        {
            return Err(OwnedJobStoreError::ReadFailed);
        }
        Ok((bytes, after))
    }
}

fn validate_request_against_catalog(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: &ResolvedProfileRoute,
    device: &DeviceSelection,
    bounds: JobExecutionBounds,
    job_directory: &Path,
) -> Result<(), OwnedJobStoreError> {
    if bounds.timeout.is_zero() || bounds.timeout > MAX_EXECUTION_TIMEOUT {
        return Err(OwnedJobStoreError::InvalidBounds);
    }
    request
        .validate(catalog)
        .map_err(|_| OwnedJobStoreError::InvalidRequest)?;
    validate_supported_request_route(request, accepted_route, device)?;
    let current_catalog = catalog
        .refresh()
        .map_err(|_| OwnedJobStoreError::InvalidRoute)?;
    let current_route = resolve_profile_route(&current_catalog, request.profile())
        .map_err(|_| OwnedJobStoreError::InvalidRoute)?;
    if current_route != *accepted_route {
        return Err(OwnedJobStoreError::RouteChanged);
    }
    match request {
        InferenceRequest::Media {
            input: MediaInput::Speech(_),
            ..
        } => {
            let recipe = prepare_breeze_speech_recipe_for_route(
                &current_catalog,
                request,
                accepted_route,
                job_directory,
            )
            .map_err(|_| OwnedJobStoreError::InvalidRoute)?;
            if !matches!(
                recipe.expectation,
                NativeOutputExpectation::BreezeSpeech { .. }
            ) {
                return Err(OwnedJobStoreError::InvalidRoute);
            }
        }
        InferenceRequest::Decision { .. } => {
            let recipe =
                prepare_native_command_for_route(&current_catalog, request, accepted_route)
                    .map_err(|_| OwnedJobStoreError::InvalidRoute)?;
            if !matches!(
                recipe.expectation,
                NativeOutputExpectation::DecisionJsonStdout {
                    ref engine,
                    ref profile_type,
                    ..
                } if matches!(engine.as_str(), "nagi" | "laya") && profile_type == "local"
            ) {
                return Err(OwnedJobStoreError::InvalidRoute);
            }
        }
        _ => return Err(OwnedJobStoreError::UnsupportedRequest),
    }
    Ok(())
}

fn canonical_repository_root(path: &Path) -> Result<PathBuf, OwnedJobStoreError> {
    let canonical = fs::canonicalize(path).map_err(|_| OwnedJobStoreError::InvalidRoute)?;
    if !canonical.is_dir() {
        return Err(OwnedJobStoreError::InvalidRoute);
    }
    Ok(canonical)
}

#[derive(Clone, Copy)]
enum PacketKind {
    Request,
    Result,
    Claim,
}

fn decode_strict_packet<T: for<'de> Deserialize<'de>>(
    bytes: &[u8],
    kind: PacketKind,
) -> Result<T, OwnedJobStoreError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let strict = StrictJson::deserialize(&mut deserializer)
        .map_err(|_| OwnedJobStoreError::InvalidPacket)?;
    deserializer
        .end()
        .map_err(|_| OwnedJobStoreError::InvalidPacket)?;
    let value = strict.into_value();
    validate_packet_object_shapes(&value, kind)?;
    serde_json::from_value(value).map_err(|_| OwnedJobStoreError::InvalidPacket)
}

/// A JSON tree that rejects duplicate object members at every depth before a
/// serde_json::Value can collapse them into a map.
enum StrictJson {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<StrictJson>),
    Object(BTreeMap<String, StrictJson>),
}

impl StrictJson {
    fn into_value(self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(value),
            Self::Number(value) => Value::Number(value),
            Self::String(value) => Value::String(value),
            Self::Array(values) => Value::Array(values.into_iter().map(Self::into_value).collect()),
            Self::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, value.into_value()))
                    .collect(),
            ),
        }
    }
}

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value with unique object keys")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(StrictJson::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJson>()? {
            values.push(value);
        }
        Ok(StrictJson::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, StrictJson>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON object key"));
            }
            values.insert(key, value);
        }
        Ok(StrictJson::Object(values))
    }
}

fn validate_packet_object_shapes(
    value: &Value,
    kind: PacketKind,
) -> Result<(), OwnedJobStoreError> {
    let root = value.as_object().ok_or(OwnedJobStoreError::InvalidPacket)?;
    match kind {
        PacketKind::Request => {
            let route = root
                .get("route")
                .and_then(Value::as_object)
                .ok_or(OwnedJobStoreError::InvalidPacket)?;
            require_object_member(route, "profile")?;
            require_object_member(route, "runtime")?;
            if route.get("device").is_some_and(|device| !device.is_null()) {
                require_object_member(route, "device")?;
            }
            require_object_member(route, "native_route")?;
            if !route.get("tasks").is_some_and(Value::is_array) {
                return Err(OwnedJobStoreError::InvalidPacket);
            }
            let request = root
                .get("request")
                .and_then(Value::as_object)
                .ok_or(OwnedJobStoreError::InvalidPacket)?;
            let operation = request
                .get("operation")
                .and_then(Value::as_object)
                .ok_or(OwnedJobStoreError::InvalidPacket)?;
            match operation.get("kind").and_then(Value::as_str) {
                Some("breeze_speech") => require_object_member(operation, "speech")?,
                Some("local_decision") => {
                    if !operation.get("questions").is_some_and(Value::is_object) {
                        return Err(OwnedJobStoreError::InvalidPacket);
                    }
                }
                _ => return Err(OwnedJobStoreError::InvalidPacket),
            }
            require_object_member(root, "device")?;
            require_object_member(root, "bounds")?;
        }
        PacketKind::Result => {
            let result = root
                .get("result")
                .and_then(Value::as_object)
                .ok_or(OwnedJobStoreError::InvalidPacket)?;
            match result.get("kind").and_then(Value::as_str) {
                Some("decision") => {
                    if !result.get("answers").is_some_and(Value::is_object) {
                        return Err(OwnedJobStoreError::InvalidPacket);
                    }
                    if result
                        .get("native_usage")
                        .is_some_and(|usage| !usage.is_null() && !usage.is_object())
                    {
                        return Err(OwnedJobStoreError::InvalidPacket);
                    }
                }
                Some("breeze") => {
                    let artifacts = result
                        .get("artifacts")
                        .and_then(Value::as_array)
                        .ok_or(OwnedJobStoreError::InvalidPacket)?;
                    if artifacts.iter().any(|artifact| !artifact.is_object()) {
                        return Err(OwnedJobStoreError::InvalidPacket);
                    }
                }
                Some("failed") => {}
                _ => return Err(OwnedJobStoreError::InvalidPacket),
            }
        }
        PacketKind::Claim => {}
    }
    Ok(())
}

fn require_object_member(object: &Map<String, Value>, key: &str) -> Result<(), OwnedJobStoreError> {
    if object.get(key).is_some_and(Value::is_object) {
        Ok(())
    } else {
        Err(OwnedJobStoreError::InvalidPacket)
    }
}

fn effective_uid() -> u32 {
    unsafe { libc::geteuid() as u32 }
}

fn open_directory_nofollow(path: &Path) -> Result<File, OwnedJobStoreError> {
    let raw = path.as_os_str().as_bytes();
    if raw.first() != Some(&b'/') || raw.contains(&0) {
        return Err(OwnedJobStoreError::UnsafeDirectory);
    }
    let components = raw[1..].split(|byte| *byte == b'/').collect::<Vec<_>>();
    if components
        .iter()
        .any(|component| component.is_empty() || *component == b"." || *component == b"..")
    {
        return Err(OwnedJobStoreError::UnsafeDirectory);
    }
    let root = CString::new("/").map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
    let root_fd = unsafe {
        libc::open(
            root.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if root_fd < 0 {
        return Err(OwnedJobStoreError::UnsafeDirectory);
    }
    let mut directory = unsafe { File::from_raw_fd(root_fd) };
    for component in components {
        let name = CString::new(component).map_err(|_| OwnedJobStoreError::UnsafeDirectory)?;
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(OwnedJobStoreError::UnsafeDirectory);
        }
        directory = unsafe { File::from_raw_fd(fd) };
    }
    Ok(directory)
}

fn directory_entry_names(directory: &File) -> std::io::Result<Vec<OsString>> {
    let dot = CString::new(".").expect("literal has no NUL");
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            dot.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let stream = unsafe { libc::fdopendir(fd) };
    if stream.is_null() {
        unsafe { libc::close(fd) };
        return Err(std::io::Error::last_os_error());
    }
    let mut names = Vec::new();
    unsafe { *libc::__errno_location() = 0 };
    loop {
        let entry = unsafe { libc::readdir(stream) };
        if entry.is_null() {
            break;
        }
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
        if name != b"." && name != b".." {
            names.push(OsString::from_vec(name.to_vec()));
        }
    }
    let read_error = unsafe { *libc::__errno_location() };
    unsafe { libc::closedir(stream) };
    if read_error != 0 {
        return Err(std::io::Error::from_raw_os_error(read_error));
    }
    Ok(names)
}

fn file_identity(metadata: &fs::Metadata) -> FileIdentity {
    FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        length: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanos: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanos: metadata.ctime_nsec(),
    }
}

fn validate_packet_metadata(
    metadata: &fs::Metadata,
    uid: u32,
    limit: usize,
) -> Result<(), OwnedJobStoreError> {
    if !metadata.is_file()
        || metadata.uid() != uid
        || (metadata.mode() & 0o7777) != PACKET_MODE
        || metadata.nlink() != 1
        || metadata.len() == 0
        || metadata.len() > limit as u64
    {
        return Err(OwnedJobStoreError::ReadFailed);
    }
    Ok(())
}

fn publish_exclusive(
    directory: &File,
    name: &str,
    bytes: &[u8],
    limit: usize,
) -> Result<FileIdentity, OwnedJobStoreError> {
    if bytes.is_empty() || bytes.len() > limit {
        return Err(OwnedJobStoreError::PacketTooLarge);
    }
    let name = CString::new(name).map_err(|_| OwnedJobStoreError::InvalidPacket)?;
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            PACKET_MODE as libc::mode_t,
        )
    };
    if fd < 0 {
        return match std::io::Error::last_os_error().raw_os_error() {
            Some(code) if code == libc::EEXIST || code == libc::ELOOP => {
                Err(OwnedJobStoreError::DuplicatePacket)
            }
            _ => Err(OwnedJobStoreError::Io),
        };
    }
    let mut file = unsafe { File::from_raw_fd(fd) };
    let publish = (|| -> std::io::Result<FileIdentity> {
        if unsafe { libc::fchmod(file.as_raw_fd(), PACKET_MODE as libc::mode_t) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let before = file.metadata()?;
        if !before.is_file()
            || before.uid() != effective_uid()
            || (before.mode() & 0o7777) != PACKET_MODE
            || before.nlink() != 1
        {
            return Err(std::io::Error::other("unsafe packet file"));
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        directory.sync_all()?;
        let after = file.metadata()?;
        if !after.is_file()
            || after.uid() != effective_uid()
            || (after.mode() & 0o7777) != PACKET_MODE
            || after.nlink() != 1
            || after.len() != bytes.len() as u64
        {
            return Err(std::io::Error::other("packet identity changed"));
        }
        Ok(file_identity(&after))
    })();
    publish.map_err(|_| OwnedJobStoreError::WriteUncertain)
}

fn validate_regular_child(
    directory: &File,
    name: &str,
    uid: u32,
) -> Result<u64, OwnedJobStoreError> {
    let name = CString::new(name).map_err(|_| OwnedJobStoreError::InvalidResult)?;
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if fd < 0 {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file
        .metadata()
        .map_err(|_| OwnedJobStoreError::InvalidResult)?;
    if !metadata.is_file()
        || metadata.uid() != uid
        || metadata.nlink() != 1
        || metadata.len() > MAX_ARTIFACT_BYTES
    {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    Ok(metadata.len())
}

fn validate_chunk_directory(
    directory: &File,
    uid: u32,
    expected_chunks: Option<&BTreeSet<String>>,
) -> Result<u64, OwnedJobStoreError> {
    let name = CString::new("result.wav.chunks").expect("literal has no NUL");
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    let chunks_directory = unsafe { File::from_raw_fd(fd) };
    let metadata = chunks_directory
        .metadata()
        .map_err(|_| OwnedJobStoreError::InvalidResult)?;
    if !metadata.is_dir() || metadata.uid() != uid {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    let names =
        directory_entry_names(&chunks_directory).map_err(|_| OwnedJobStoreError::InvalidResult)?;
    if (expected_chunks.is_some() && names.is_empty()) || names.len() > MAX_ARTIFACTS - 1 {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    let mut actual = BTreeSet::new();
    let mut total_bytes = 0u64;
    for entry in &names {
        let Some(name) = entry.to_str() else {
            return Err(OwnedJobStoreError::InvalidResult);
        };
        if !safe_chunk_name(name) || !actual.insert(name.to_owned()) {
            return Err(OwnedJobStoreError::InvalidResult);
        }
        total_bytes =
            total_bytes.saturating_add(validate_regular_child(&chunks_directory, name, uid)?);
        if total_bytes > MAX_TOTAL_ARTIFACT_BYTES {
            return Err(OwnedJobStoreError::InvalidResult);
        }
    }
    if expected_chunks.is_some_and(|expected| expected != &actual) {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    Ok(total_bytes)
}

fn safe_chunk_name(name: &str) -> bool {
    name.len() <= 128
        && name.ends_with(".wav")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        && name != ".wav"
}

fn validate_relative_artifact_path(path: &str) -> Result<(), OwnedJobStoreError> {
    if path.is_empty()
        || path.len() > MAX_ARTIFACT_PATH_BYTES
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains(':')
        || path.chars().any(char::is_control)
    {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    if path
        .split('/')
        .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(OwnedJobStoreError::InvalidResult);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    use tempfile::TempDir;

    const TEST_GPU: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";

    struct Fixture {
        _root: TempDir,
        repo: PathBuf,
        job_dir: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let root = tempfile::tempdir().expect("fixture root");
            let repo = root.path().join("repo");
            let media_root = repo.join("toolkits/l2m2");
            let decisions_root = repo.join("toolkits/l3ds");
            fs::create_dir_all(media_root.join("media-models")).unwrap();
            fs::create_dir_all(decisions_root.join("decision-models")).unwrap();
            fs::create_dir_all(decisions_root.join(".venv/bin")).unwrap();
            fs::write(
                repo.join("llama-swap.yaml"),
                "models:\n  \"fixture-language\":\n    cmd: |\n      llama-server -m /not-loaded.gguf\n",
            )
            .unwrap();
            fs::write(
                media_root.join("media-runtimes.json"),
                serde_json::to_vec(&serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "breeze-tts-2",
                        "name": "Fixture Breeze",
                        "runtime": "audio.cpp",
                        "device": "cuda",
                        "tasks": ["tts", "clon", "design"],
                        "script": "media-models/generate-breeze-tts-2.sh"
                    }]
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                media_root.join("media-models/generate-breeze-tts-2.sh"),
                "#!/bin/bash\nexit 0\n",
            )
            .unwrap();
            fs::write(
                decisions_root.join("decision-runtimes.json"),
                serde_json::to_vec(&serde_json::json!({
                    "version": "test",
                    "profiles": [{
                        "id": "nagi-fixture",
                        "name": "Fixture Nagi",
                        "engine": "nagi",
                        "tier": "test",
                        "type": "local",
                        "model": "fixture-model",
                        "device": "cpu",
                        "entrypoint": "decision-models/runner.py"
                    }]
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                decisions_root.join("decision-models/runner.py"),
                "# fixture only\n",
            )
            .unwrap();
            let python = decisions_root.join(".venv/bin/python");
            fs::write(&python, "#!/bin/sh\nexit 0\n").unwrap();
            fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
            let job_dir = root.path().join("job");
            fs::create_dir(&job_dir).unwrap();
            fs::set_permissions(&job_dir, fs::Permissions::from_mode(DIRECTORY_MODE)).unwrap();
            Self {
                _root: root,
                repo,
                job_dir,
            }
        }

        fn catalog(&self) -> Catalog {
            Catalog::load(&self.repo).expect("fixture catalog")
        }

        fn route(&self, profile: &str) -> ResolvedProfileRoute {
            let key = ProfileKey::from_str(profile).unwrap();
            resolve_profile_route(&self.catalog(), &key).unwrap()
        }

        fn decision_request(&self) -> InferenceRequest {
            let mut questions = Map::new();
            questions.insert(
                "choice".to_owned(),
                serde_json::json!({"type":"choice", "options":["yes", "no"]}),
            );
            InferenceRequest::Decision {
                profile: ProfileKey::from_str("decisions:nagi-fixture").unwrap(),
                state: serde_json::json!({"private_state":"private-state-marker"}),
                questions,
            }
        }

        fn breeze_request(&self) -> InferenceRequest {
            InferenceRequest::Media {
                profile: ProfileKey::from_str("media:breeze-tts-2").unwrap(),
                input: MediaInput::Speech(SpeechInput {
                    text: "private-prompt-marker".to_owned(),
                    mode: SpeechMode::Standard,
                    instruction: None,
                    voice_reference: None,
                    reference_text: None,
                }),
            }
        }

        fn bounds() -> JobExecutionBounds {
            JobExecutionBounds::new(
                Duration::from_secs(30),
                RecoveryPolicy::new(Duration::from_millis(25)).unwrap(),
            )
            .unwrap()
        }

        fn write_decision_request(&self, store: &OwnedJobStore) -> ValidatedOwnedJob {
            let request = self.decision_request();
            let route = self.route("decisions:nagi-fixture");
            store
                .write_request(OwnedJobRequestSpec {
                    job_id: 17,
                    request_id: "caller:owned-job-fixture",
                    repository_root: &self.repo,
                    request: &request,
                    accepted_route: &route,
                    device: DeviceSelection::Cpu,
                    bounds: Self::bounds(),
                })
                .unwrap()
        }
    }

    #[test]
    fn private_directory_requires_exact_owner_mode_and_nofollow_components() {
        let fixture = Fixture::new();
        let store = OwnedJobStore::open(&fixture.job_dir).unwrap();
        assert_eq!(store.uid, effective_uid());
        assert_eq!(fixture.job_dir.metadata().unwrap().mode() & 0o777, 0o700);

        let alias = fixture._root.path().join("job-alias");
        std::os::unix::fs::symlink(&fixture.job_dir, &alias).unwrap();
        assert_eq!(
            OwnedJobStore::open(alias).unwrap_err(),
            OwnedJobStoreError::UnsafeDirectory
        );

        fs::set_permissions(&fixture.job_dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(
            OwnedJobStore::open(&fixture.job_dir).unwrap_err(),
            OwnedJobStoreError::UnsafeDirectory
        );
    }

    #[test]
    fn wrong_directory_owner_is_rejected_when_test_process_can_change_it() {
        if effective_uid() != 0 {
            return;
        }
        let fixture = Fixture::new();
        let path = CString::new(fixture.job_dir.as_os_str().as_bytes()).unwrap();
        let changed = unsafe { libc::chown(path.as_ptr(), 1, 1) };
        if changed == 0 {
            assert_eq!(
                OwnedJobStore::open(&fixture.job_dir).unwrap_err(),
                OwnedJobStoreError::UnsafeDirectory
            );
        }
    }

    #[test]
    fn breeze_request_is_private_write_once_and_roundtrips_with_full_route() {
        let fixture = Fixture::new();
        let store = OwnedJobStore::open(&fixture.job_dir).unwrap();
        let request = fixture.breeze_request();
        let route = fixture.route("media:breeze-tts-2");
        let job = store
            .write_request(OwnedJobRequestSpec {
                job_id: 18,
                request_id: "caller:breeze-fixture",
                repository_root: &fixture.repo,
                request: &request,
                accepted_route: &route,
                device: DeviceSelection::nvidia(TEST_GPU, 2, 256).unwrap(),
                bounds: Fixture::bounds(),
            })
            .unwrap();
        let packet_path = fixture.job_dir.join(REQUEST_PACKET_NAME);
        let packet = fs::read(&packet_path).unwrap();
        assert_eq!(packet_path.metadata().unwrap().mode() & 0o777, 0o600);
        assert!(String::from_utf8_lossy(&packet).contains("private-prompt-marker"));
        assert!(String::from_utf8_lossy(&packet).contains("audio.cpp"));
        assert!(String::from_utf8_lossy(&packet).contains("media-models/generate-breeze-tts-2.sh"));
        assert!(!String::from_utf8_lossy(&packet).contains("fingerprint"));
        assert!(!format!("{job:?}").contains("private-prompt-marker"));
        assert!(!format!("{store:?}").contains(fixture.job_dir.to_str().unwrap()));

        let reopened = store.read_request().unwrap();
        assert_eq!(reopened.job_id(), 18);
        assert_eq!(reopened.route(), &route);
        assert_eq!(
            reopened
                .device()
                .gpu_probe_config()
                .unwrap()
                .unwrap()
                .gpu_uuid(),
            TEST_GPU
        );
        assert_eq!(
            reopened.bounds().recovery().timeout(),
            Duration::from_millis(25)
        );
        store.claim_execution(&reopened).unwrap();
        assert_eq!(
            store.claim_execution(&reopened),
            Err(OwnedJobStoreError::DuplicatePacket)
        );
        let claim_bytes = fs::read(fixture.job_dir.join(EXECUTION_CLAIM_NAME)).unwrap();
        let claim_text = String::from_utf8_lossy(&claim_bytes);
        assert!(claim_text.contains("caller:breeze-fixture"));
        assert!(!claim_text.contains("private-prompt-marker"));
        assert_eq!(
            fixture
                .job_dir
                .join(EXECUTION_CLAIM_NAME)
                .metadata()
                .unwrap()
                .mode()
                & 0o777,
            0o600
        );
        assert_eq!(
            store
                .write_request(OwnedJobRequestSpec {
                    job_id: 18,
                    request_id: "caller:breeze-fixture",
                    repository_root: &fixture.repo,
                    request: &request,
                    accepted_route: &route,
                    device: DeviceSelection::nvidia(TEST_GPU, 2, 256).unwrap(),
                    bounds: Fixture::bounds(),
                })
                .unwrap_err(),
            OwnedJobStoreError::DuplicatePacket
        );
    }

    #[test]
    fn route_drift_after_acceptance_rejects_helper_revalidation() {
        let fixture = Fixture::new();
        let store = OwnedJobStore::open(&fixture.job_dir).unwrap();
        let request = fixture.decision_request();
        let route = fixture.route("decisions:nagi-fixture");
        store
            .write_request(OwnedJobRequestSpec {
                job_id: 19,
                request_id: "caller:route-drift",
                repository_root: &fixture.repo,
                request: &request,
                accepted_route: &route,
                device: DeviceSelection::Cpu,
                bounds: Fixture::bounds(),
            })
            .unwrap();
        let manifest = fixture.repo.join("toolkits/l3ds/decision-runtimes.json");
        let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        value["profiles"][0]["model"] = Value::String("changed-model".to_owned());
        fs::write(manifest, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(store.read_request().is_err());
    }

    #[test]
    fn decision_result_preserves_actual_latency_and_usage_without_cleanup_claims() {
        let fixture = Fixture::new();
        let store = OwnedJobStore::open(&fixture.job_dir).unwrap();
        let job = fixture.write_decision_request(&store);
        store.claim_execution(&job).unwrap();
        let mut answers = Map::new();
        answers.insert(
            "choice".to_owned(),
            serde_json::json!({"choice":"yes", "confidence":0.8}),
        );
        let mut usage = Map::new();
        usage.insert("input_tokens".to_owned(), Value::from(7));
        let native = NativeJobResult::Decision(NativeDecisionResult {
            result: super::super::requests::InferenceResult::Decision { answers },
            native_latency: Some(Duration::from_micros(23)),
            native_usage: Some(usage.clone()),
        });
        assert!(!format!("{native:?}").contains("private-state-marker"));
        store.write_result(&job, &native).unwrap();
        let result_bytes = fs::read(fixture.job_dir.join(RESULT_PACKET_NAME)).unwrap();
        let result_text = String::from_utf8_lossy(&result_bytes);
        assert!(result_text.contains("native_latency_nanos"));
        assert!(result_text.contains("native_usage"));
        assert!(!result_text.contains("cleanup_proven"));
        assert!(!result_text.contains("cgroup_released"));
        assert!(!result_text.contains("gpu_released"));

        let result = store.read_result(&job).unwrap();
        let NativeJobResult::Decision(result) = result else {
            panic!("decision result expected")
        };
        assert_eq!(result.native_latency, Some(Duration::from_micros(23)));
        assert_eq!(result.native_usage, Some(usage));
        assert!(matches!(
            store.write_result(&job, &NativeJobResult::Failed(NativeJobFailure::Internal)),
            Err(OwnedJobStoreError::DuplicatePacket)
        ));
    }

    #[test]
    fn duplicate_keys_unknown_cleanup_fields_and_tuple_structs_are_rejected() {
        let duplicate_root = br#"{"schema_version":1,"schema_version":1}"#;
        assert!(matches!(
            decode_strict_packet::<WireRequestPacket>(duplicate_root, PacketKind::Request),
            Err(OwnedJobStoreError::InvalidPacket)
        ));
        let duplicate_nested = br#"{"state":{"secret":1,"secret":2}}"#;
        assert!(matches!(
            decode_strict_packet::<WireRequestPacket>(duplicate_nested, PacketKind::Request),
            Err(OwnedJobStoreError::InvalidPacket)
        ));
        let tuple_route = br#"{"route":[],"request":{},"device":{},"bounds":{}}"#;
        assert!(matches!(
            decode_strict_packet::<WireRequestPacket>(tuple_route, PacketKind::Request),
            Err(OwnedJobStoreError::InvalidPacket)
        ));
        let unexpected_cleanup = br#"{"schema_version":1,"job_id":1,"request_id":"caller:x","result":{"kind":"failed","code":"internal","cleanup_proven":true}}"#;
        assert!(matches!(
            decode_strict_packet::<WireResultPacket>(unexpected_cleanup, PacketKind::Result),
            Err(OwnedJobStoreError::InvalidPacket)
        ));
    }

    #[test]
    fn unexpected_contents_and_oversized_packets_fail_closed() {
        let fixture = Fixture::new();
        let store = OwnedJobStore::open(&fixture.job_dir).unwrap();
        fs::write(fixture.job_dir.join("surprise"), b"unexpected").unwrap();
        let request = fixture.decision_request();
        let route = fixture.route("decisions:nagi-fixture");
        assert_eq!(
            store
                .write_request(OwnedJobRequestSpec {
                    job_id: 20,
                    request_id: "caller:unexpected",
                    repository_root: &fixture.repo,
                    request: &request,
                    accepted_route: &route,
                    device: DeviceSelection::Cpu,
                    bounds: Fixture::bounds(),
                })
                .unwrap_err(),
            OwnedJobStoreError::UnexpectedContents
        );

        fs::remove_file(fixture.job_dir.join("surprise")).unwrap();
        fs::write(
            fixture.job_dir.join(REQUEST_PACKET_NAME),
            vec![b'x'; MAX_REQUEST_PACKET_BYTES + 1],
        )
        .unwrap();
        fs::set_permissions(
            fixture.job_dir.join(REQUEST_PACKET_NAME),
            fs::Permissions::from_mode(PACKET_MODE),
        )
        .unwrap();
        assert!(store.read_request().is_err());
    }

    #[test]
    fn artifact_receipt_rejects_absolute_escape_and_oversized_paths() {
        for path in [
            "/tmp/result.wav",
            "../result.wav",
            "result.wav.chunks/../../x.wav",
            &"x".repeat(MAX_ARTIFACT_PATH_BYTES + 1),
        ] {
            assert_eq!(
                validate_relative_artifact_path(path),
                Err(OwnedJobStoreError::InvalidResult)
            );
        }
    }
}
