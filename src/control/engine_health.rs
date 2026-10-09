//! Bounded native health and model-readiness probes for engines owned by an
//! isolated supervisor.
//!
//! A successful probe means only that the expected supervisor answered its
//! fixed native route at one point in time. For `LlamaServer`, that route's
//! documented response means the server is ready; ComfyUI's `/system_stats`
//! is only a valid inspection snapshot, not model readiness. llama-swap model
//! readiness uses its router liveness endpoint plus its read-only `/running`
//! listing. The OpenAI `/v1/models` status is not sufficient: upstream maps
//! both starting and ready models to `loaded`. None of these results is an
//! admission, drain, cgroup-release, GPU-idle, or lease proof. Inference
//! methods are not called or replayed here. Response bodies are parsed and
//! discarded; ComfyUI and llama-swap listings can include command-line args.

use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::de::value::MapAccessDeserializer;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

use super::catalog::ProfileId;
use super::managed_service::{
    ManagedServiceErrorKind, OwnedService, OwnedServiceGenerationIdentity,
};
use super::native_http::{PinnedEngineSocket, VerifiedConnectError};
use super::runtime::OperationContext;
use super::service_owner::ManagedRuntimeOwner;

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 512 * 1024;
const MAX_CHUNK_LINE_BYTES: usize = 16;
const MAX_CHUNK_COUNT: usize = 4096;
const MAX_CHUNKED_OVERHEAD_BYTES: usize = 64 * 1024;
const READ_CHUNK_BYTES: usize = 4096;
const IO_POLL: Duration = Duration::from_millis(25);

#[derive(Clone, Copy)]
enum NativeContentType {
    Json,
    PlainText,
}

#[derive(Deserialize)]
struct LlamaServerHealthResponse {
    status: String,
}

#[derive(Deserialize)]
struct ComfySystemStatsResponse {
    system: Map<String, Value>,
    devices: Vec<Value>,
}

#[derive(Deserialize)]
struct LlamaSwapRunningResponse {
    running: Vec<JsonObject<LlamaSwapRunningModel>>,
}

#[derive(Deserialize)]
struct LlamaSwapRunningModel {
    model: String,
    state: String,
}

struct JsonObject<T>(T);

impl<'de, T> Deserialize<'de> for JsonObject<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ObjectVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> Visitor<'de> for ObjectVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = JsonObject<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                T::deserialize(MapAccessDeserializer::new(map)).map(JsonObject)
            }
        }

        deserializer.deserialize_map(ObjectVisitor(std::marker::PhantomData))
    }
}

fn deserialize_json_object<T>(bytes: &[u8]) -> Result<T, serde_json::Error>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_slice::<JsonObject<T>>(bytes).map(|object| object.0)
}

/// Native backends with a bounded health endpoint in this slice. The enum is
/// also the explicit backend gate for the one-shot llama-swap activation seam;
/// that router is separate from a direct `LlamaServer` probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeEngineKind {
    LlamaServer,
    Comfy,
}

impl NativeEngineKind {
    fn path(self) -> &'static str {
        match self {
            Self::LlamaServer => "/health",
            Self::Comfy => "/system_stats",
        }
    }
}

/// Fixed, redacted failure categories. No native response, header, path,
/// process identity, or operating-system error is retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeHealthFailureKind {
    Cancelled,
    DeadlineExceeded,
    OwnerUnverified,
    NotRunning,
    WrongServiceRole,
    WrongNativeBackend,
    SocketUnavailable,
    PeerMismatch,
    Io,
    MalformedResponse,
    TruncatedResponse,
    ResponseTooLarge,
    Unhealthy,
}

/// A redacted health failure that carries only a fixed category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeHealthFailure {
    kind: NativeHealthFailureKind,
}

impl NativeHealthFailure {
    fn new(kind: NativeHealthFailureKind) -> Self {
        Self { kind }
    }

    pub fn kind(&self) -> NativeHealthFailureKind {
        self.kind
    }
}

impl fmt::Display for NativeHealthFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            NativeHealthFailureKind::Cancelled => "native health probe was cancelled",
            NativeHealthFailureKind::DeadlineExceeded => "native health probe deadline expired",
            NativeHealthFailureKind::OwnerUnverified => {
                "owned engine supervisor could not be verified"
            }
            NativeHealthFailureKind::NotRunning => "owned engine supervisor is not running",
            NativeHealthFailureKind::WrongServiceRole => {
                "owned service is not an isolated engine supervisor"
            }
            NativeHealthFailureKind::WrongNativeBackend => {
                "native engine backend is unsupported for this operation"
            }
            NativeHealthFailureKind::SocketUnavailable => "pinned engine socket is unavailable",
            NativeHealthFailureKind::PeerMismatch => {
                "engine socket peer is not the owned supervisor"
            }
            NativeHealthFailureKind::Io => "native health transport failed",
            NativeHealthFailureKind::MalformedResponse => "native health response is malformed",
            NativeHealthFailureKind::TruncatedResponse => "native health response is truncated",
            NativeHealthFailureKind::ResponseTooLarge => "native health response exceeds the limit",
            NativeHealthFailureKind::Unhealthy => "native engine reports unhealthy",
        })
    }
}

impl Error for NativeHealthFailure {}

/// Point-in-time native endpoint evidence. It is deliberately not an owner,
/// admission token, job-completion receipt, or resource-release observation.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct NativeHealthObservation {
    kind: NativeEngineKind,
    observed_at: Instant,
}

impl NativeHealthObservation {
    pub fn kind(&self) -> NativeEngineKind {
        self.kind
    }

    pub fn observed_at(&self) -> Instant {
        self.observed_at
    }
}

impl fmt::Debug for NativeHealthObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeHealthObservation")
            .field("kind", &self.kind)
            .field("observed_at", &"<point-in-time>")
            .finish()
    }
}

/// Point-in-time evidence that llama-swap answered its liveness endpoint and
/// listed the selected model with exact process state `ready`. This does not
/// prove admission, job completion, resource release, GPU idleness, or a
/// lifecycle lease.
#[derive(Eq, PartialEq)]
pub struct RouterModelReadinessObservation {
    selected_model: ProfileId,
    observed_at: Instant,
}

impl RouterModelReadinessObservation {
    pub fn observed_at(&self) -> Instant {
        self.observed_at
    }

    pub fn matches_model(&self, model_id: &str) -> bool {
        self.selected_model.as_str() == model_id
    }

    #[cfg(test)]
    pub(crate) fn test_for_model(model_id: &str) -> Self {
        Self {
            selected_model: validated_selected_model(model_id)
                .expect("test model ID must be valid"),
            observed_at: Instant::now(),
        }
    }
}

impl fmt::Debug for RouterModelReadinessObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouterModelReadinessObservation")
            .field("observed_at", &"<point-in-time>")
            .finish()
    }
}

/// Whether a one-shot activation request is known not to have crossed the
/// transport boundary or may have reached llama-swap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationDisposition {
    NoDispatch,
    Unconfirmed,
}

/// A fixed, redacted result category for an activation attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterModelActivationFailure {
    disposition: ActivationDisposition,
    kind: NativeHealthFailureKind,
}

impl RouterModelActivationFailure {
    fn new(disposition: ActivationDisposition, kind: NativeHealthFailureKind) -> Self {
        Self { disposition, kind }
    }

    pub fn disposition(&self) -> ActivationDisposition {
        self.disposition
    }

    pub fn kind(&self) -> NativeHealthFailureKind {
        self.kind
    }
}

impl fmt::Display for RouterModelActivationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.disposition {
            ActivationDisposition::NoDispatch => "llama-swap model activation was not dispatched",
            ActivationDisposition::Unconfirmed => {
                "llama-swap model activation outcome is unconfirmed"
            }
        })
    }
}

impl Error for RouterModelActivationFailure {}

/// A bounded acknowledgment from one forwarded llama-server health request.
/// It is not a readiness, admission, manager-generation, or resource-release
/// proof; the caller must reconcile through the read-only `/running` probe.
#[derive(Eq, PartialEq)]
pub struct RouterModelActivationAck {
    selected_model: ProfileId,
    acknowledged_at: Instant,
}

impl RouterModelActivationAck {
    pub fn acknowledged_at(&self) -> Instant {
        self.acknowledged_at
    }

    pub fn matches_model(&self, model_id: &str) -> bool {
        self.selected_model.as_str() == model_id
    }
}

impl fmt::Debug for RouterModelActivationAck {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouterModelActivationAck")
            .field("acknowledged_at", &"<point-in-time>")
            .finish()
    }
}

/// Probe a fixed native health route through the private pinned socket.
///
/// The exact write-ahead intent and live manager generation/namespace are
/// verified before connecting and again after a successful response. The
/// actual connected peer must also match the owned supervisor MainPID and its
/// captured coordinator UID via `SO_PEERCRED`. No TCP/proxy/DNS fallback or
/// request retry is available. The whole operation shares `context`'s
/// deadline and cancellation token.
pub fn probe(
    owner: &ManagedRuntimeOwner<'_>,
    service: &OwnedService,
    socket_directory: &Path,
    kind: NativeEngineKind,
    context: &OperationContext<'_>,
) -> Result<NativeHealthObservation, NativeHealthFailure> {
    let mut stream = connect_owned_supervisor(owner, service, socket_directory, context)?;
    request_get(&mut stream, kind.path(), "application/json", context)?;
    let response = read_response(&mut stream, context, NativeContentType::Json, false)?;
    validate_native_response(kind, response.status, &response.body)?;
    check_active(context)?;
    require_running(owner, service, context)?;
    check_active(context)?;
    Ok(NativeHealthObservation {
        kind,
        observed_at: Instant::now(),
    })
}

/// Verify router liveness and exact selected-model process readiness using
/// only read-only llama-swap endpoints. `/v1/models` is deliberately not used
/// as readiness evidence: its `loaded` state includes `starting` processes.
/// The selected ID is compared to the bounded `/running` response and is never
/// interpolated into a path, header, error, or diagnostic.
pub fn probe_llama_swap_router_model_readiness(
    owner: &ManagedRuntimeOwner<'_>,
    service: &OwnedService,
    socket_directory: &Path,
    selected_model: &str,
    context: &OperationContext<'_>,
) -> Result<RouterModelReadinessObservation, NativeHealthFailure> {
    check_active(context)?;
    let selected_model = validated_selected_model(selected_model)?;

    let mut health_stream = connect_owned_supervisor(owner, service, socket_directory, context)?;
    request_get(&mut health_stream, "/health", "text/plain", context)?;
    let health_response = read_response(
        &mut health_stream,
        context,
        NativeContentType::PlainText,
        false,
    )?;
    validate_router_liveness(health_response.status, &health_response.body)?;

    // Use a second peer-verified connection. The first response may have
    // crossed a supervisor restart or ownership transition; reconfirming the
    // owner immediately before this request preserves the same generation
    // boundary for both observations.
    check_active(context)?;
    let mut running_stream = connect_owned_supervisor(owner, service, socket_directory, context)?;
    request_get(&mut running_stream, "/running", "application/json", context)?;
    let running_response =
        read_response(&mut running_stream, context, NativeContentType::Json, true)?;
    validate_router_model_readiness(
        running_response.status,
        &running_response.body,
        selected_model.as_str(),
    )?;

    check_active(context)?;
    require_running(owner, service, context)?;
    check_active(context)?;
    Ok(RouterModelReadinessObservation {
        selected_model,
        observed_at: Instant::now(),
    })
}

/// Make exactly one side-effectful llama-swap request for a model whose
/// accepted backend contract is llama-server. The routed `/health` request may
/// cause llama-swap to start the model; a successful response is only an
/// acknowledgment from that forwarded native endpoint. It is not router
/// readiness or evidence that the model remains running. Callers must not
/// retry an `Unconfirmed` outcome and may reconcile only through the
/// read-only `/running` probe.
pub fn activate_llama_swap_model_once(
    owner: &ManagedRuntimeOwner<'_>,
    service: &OwnedService,
    socket_directory: &Path,
    backend: NativeEngineKind,
    selected_model: &str,
    context: &OperationContext<'_>,
) -> Result<RouterModelActivationAck, RouterModelActivationFailure> {
    let no_dispatch = |error: NativeHealthFailure| {
        RouterModelActivationFailure::new(ActivationDisposition::NoDispatch, error.kind())
    };
    let unconfirmed = |error: NativeHealthFailure| {
        RouterModelActivationFailure::new(ActivationDisposition::Unconfirmed, error.kind())
    };

    check_active(context).map_err(no_dispatch)?;
    if backend != NativeEngineKind::LlamaServer {
        return Err(RouterModelActivationFailure::new(
            ActivationDisposition::NoDispatch,
            NativeHealthFailureKind::WrongNativeBackend,
        ));
    }
    let selected_model = validated_selected_model(selected_model).map_err(no_dispatch)?;
    let mut stream =
        connect_owned_supervisor(owner, service, socket_directory, context).map_err(no_dispatch)?;
    require_running(owner, service, context).map_err(no_dispatch)?;
    check_active(context).map_err(no_dispatch)?;

    if let Err((error, wrote_any_bytes)) =
        request_llama_swap_model_health_once(&mut stream, &selected_model, context)
    {
        return Err(activation_write_failure(error, wrote_any_bytes));
    }

    let response =
        read_response(&mut stream, context, NativeContentType::Json, false).map_err(unconfirmed)?;
    validate_native_response(
        NativeEngineKind::LlamaServer,
        response.status,
        &response.body,
    )
    .map_err(unconfirmed)?;
    check_active(context).map_err(unconfirmed)?;
    require_running(owner, service, context).map_err(unconfirmed)?;
    check_active(context).map_err(unconfirmed)?;
    Ok(RouterModelActivationAck {
        selected_model,
        acknowledged_at: Instant::now(),
    })
}

fn request_llama_swap_model_health_once(
    stream: &mut UnixStream,
    selected_model: &ProfileId,
    context: &OperationContext<'_>,
) -> Result<(), (NativeHealthFailure, bool)> {
    let mut request = Vec::with_capacity(384);
    request.extend_from_slice(b"GET /upstream/");
    append_percent_encoded_path_segment(&mut request, selected_model.as_str());
    request.extend_from_slice(
        b"/health HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\nConnection: close\r\n\r\n",
    );
    write_all_bounded_tracking(stream, &request, context)
}

fn activation_write_failure(
    error: NativeHealthFailure,
    wrote_any_bytes: bool,
) -> RouterModelActivationFailure {
    let disposition = if wrote_any_bytes {
        ActivationDisposition::Unconfirmed
    } else {
        ActivationDisposition::NoDispatch
    };
    RouterModelActivationFailure::new(disposition, error.kind())
}

fn append_percent_encoded_path_segment(output: &mut Vec<u8>, value: &str) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            output.push(byte);
        } else {
            output.push(b'%');
            output.push(HEX[usize::from(byte >> 4)]);
            output.push(HEX[usize::from(byte & 0x0f)]);
        }
    }
}

fn validated_selected_model(value: &str) -> Result<ProfileId, NativeHealthFailure> {
    // Match catalog::ProfileId's 128-byte limit before copying so callers
    // cannot force an unbounded allocation with an invalid identifier.
    if value.len() > 128 {
        return Err(failure(NativeHealthFailureKind::Unhealthy));
    }
    ProfileId::new(value.to_owned()).map_err(|_| failure(NativeHealthFailureKind::Unhealthy))
}

fn connect_owned_supervisor(
    owner: &ManagedRuntimeOwner<'_>,
    service: &OwnedService,
    socket_directory: &Path,
    context: &OperationContext<'_>,
) -> Result<UnixStream, NativeHealthFailure> {
    check_active(context)?;
    let generation = running_generation(owner, service, context)?;
    let (expected_pid, expected_uid) = service
        .supervisor_peer_identity()
        .ok_or_else(|| failure(NativeHealthFailureKind::WrongServiceRole))?;
    check_active(context)?;
    let socket = match PinnedEngineSocket::open(socket_directory) {
        Ok(socket) => socket,
        Err(_) => {
            check_active(context)?;
            return Err(failure(NativeHealthFailureKind::SocketUnavailable));
        }
    };
    check_active(context)?;
    let stream = socket
        .connect_verified(expected_pid, expected_uid, context)
        .map_err(map_connect_error)?;
    // The actual socket peer and the retained owner generation must agree
    // after connection too. A stale pre-connect observation is not dispatch
    // authority if the manager/WAL changes during connect.
    if running_generation(owner, service, context)? != generation {
        return Err(failure(NativeHealthFailureKind::OwnerUnverified));
    }
    Ok(stream)
}

fn require_running(
    owner: &ManagedRuntimeOwner<'_>,
    service: &OwnedService,
    context: &OperationContext<'_>,
) -> Result<(), NativeHealthFailure> {
    running_generation(owner, service, context).map(|_| ())
}

fn running_generation(
    owner: &ManagedRuntimeOwner<'_>,
    service: &OwnedService,
    context: &OperationContext<'_>,
) -> Result<OwnedServiceGenerationIdentity, NativeHealthFailure> {
    match owner.running_generation_identity(service, context) {
        Ok(identity) => Ok(identity),
        Err(ManagedServiceErrorKind::ServiceNotActive) => {
            Err(failure(NativeHealthFailureKind::NotRunning))
        }
        Err(ManagedServiceErrorKind::Cancelled) => Err(failure(NativeHealthFailureKind::Cancelled)),
        Err(ManagedServiceErrorKind::DeadlineExceeded) => {
            Err(failure(NativeHealthFailureKind::DeadlineExceeded))
        }
        Err(_) => Err(failure(NativeHealthFailureKind::OwnerUnverified)),
    }
}

fn request_get(
    stream: &mut UnixStream,
    path: &str,
    accept: &str,
    context: &OperationContext<'_>,
) -> Result<(), NativeHealthFailure> {
    let mut request = Vec::with_capacity(160);
    request.extend_from_slice(b"GET ");
    request.extend_from_slice(path.as_bytes());
    request.extend_from_slice(b" HTTP/1.1\r\nHost: localhost\r\nAccept: ");
    request.extend_from_slice(accept.as_bytes());
    request.extend_from_slice(b"\r\nConnection: close\r\n\r\n");
    write_all_bounded(stream, &request, context)
}

#[derive(Debug)]
struct NativeResponse {
    status: u16,
    body: Vec<u8>,
}

#[derive(Debug)]
enum NativeBodyFraming {
    ContentLength(usize),
    Chunked,
}

fn read_response(
    stream: &mut UnixStream,
    context: &OperationContext<'_>,
    expected_content_type: NativeContentType,
    allow_chunked: bool,
) -> Result<NativeResponse, NativeHealthFailure> {
    let mut buffered = Vec::with_capacity(READ_CHUNK_BYTES);
    let header_end = loop {
        if let Some(offset) = find_subslice(&buffered, b"\r\n\r\n") {
            if offset + 4 > MAX_HEADER_BYTES {
                return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
            }
            break offset;
        }
        if buffered.len() > MAX_HEADER_BYTES {
            return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
        }
        let mut chunk = [0u8; READ_CHUNK_BYTES];
        let count = read_some(stream, &mut chunk, context)?;
        if count == 0 {
            return Err(failure(NativeHealthFailureKind::TruncatedResponse));
        }
        buffered.extend_from_slice(&chunk[..count]);
    };

    let (status, framing) = parse_response_head(
        &buffered[..header_end],
        expected_content_type,
        allow_chunked,
    )?;
    let body_start = header_end + 4;
    let body_prefix = &buffered[body_start..];
    let body = match framing {
        NativeBodyFraming::ContentLength(content_length) => {
            if body_prefix.len() > content_length {
                return Err(failure(NativeHealthFailureKind::MalformedResponse));
            }
            let mut body = Vec::with_capacity(content_length);
            body.extend_from_slice(body_prefix);
            while body.len() < content_length {
                let remaining = content_length - body.len();
                let mut chunk = [0u8; READ_CHUNK_BYTES];
                let capacity = remaining.min(chunk.len());
                let count = read_some(stream, &mut chunk[..capacity], context)?;
                if count == 0 {
                    return Err(failure(NativeHealthFailureKind::TruncatedResponse));
                }
                body.extend_from_slice(&chunk[..count]);
            }
            body
        }
        NativeBodyFraming::Chunked => read_chunked_body(stream, body_prefix, context)?,
    };
    Ok(NativeResponse { status, body })
}

fn parse_response_head(
    bytes: &[u8],
    expected_content_type: NativeContentType,
    allow_chunked: bool,
) -> Result<(u16, NativeBodyFraming), NativeHealthFailure> {
    let lines = split_crlf_lines(bytes)?;
    let Some(status_line) = lines.first() else {
        return Err(failure(NativeHealthFailureKind::MalformedResponse));
    };
    if status_line.len() < 12
        || &status_line[..9] != b"HTTP/1.1 "
        || !status_line[9..12].iter().all(u8::is_ascii_digit)
        || (status_line.len() > 12 && status_line[12] != b' ')
        || !status_line.iter().all(|byte| (0x20..=0x7e).contains(byte))
    {
        return Err(failure(NativeHealthFailureKind::MalformedResponse));
    }
    let status = std::str::from_utf8(&status_line[9..12])
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| failure(NativeHealthFailureKind::MalformedResponse))?;

    let mut content_length = None;
    let mut content_type = None;
    let mut transfer_encoding = false;
    for line in lines.iter().skip(1) {
        if line.is_empty()
            || !line.iter().all(|byte| (0x20..=0x7e).contains(byte))
            || line
                .first()
                .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
        {
            return Err(failure(NativeHealthFailureKind::MalformedResponse));
        }
        let Some(colon) = line.iter().position(|byte| *byte == b':') else {
            return Err(failure(NativeHealthFailureKind::MalformedResponse));
        };
        let name = &line[..colon];
        if name.is_empty() || !name.iter().copied().all(is_header_token) {
            return Err(failure(NativeHealthFailureKind::MalformedResponse));
        }
        let name = name.iter().map(u8::to_ascii_lowercase).collect::<Vec<_>>();
        let value = trim_ows(&line[colon + 1..]);
        if name == b"content-encoding" {
            // Do not add decompression to this narrow native protocol.
            return Err(failure(NativeHealthFailureKind::MalformedResponse));
        }
        if name == b"transfer-encoding" {
            if transfer_encoding || !allow_chunked || !value.eq_ignore_ascii_case(b"chunked") {
                return Err(failure(NativeHealthFailureKind::MalformedResponse));
            }
            transfer_encoding = true;
        }
        if name == b"content-type" {
            if content_type.is_some() {
                return Err(failure(NativeHealthFailureKind::MalformedResponse));
            }
            content_type = Some(value);
        }
        if name == b"content-length" {
            if content_length.is_some() || value.is_empty() || !value.iter().all(u8::is_ascii_digit)
            {
                return Err(failure(NativeHealthFailureKind::MalformedResponse));
            }
            let parsed = std::str::from_utf8(value)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| failure(NativeHealthFailureKind::MalformedResponse))?;
            if parsed > MAX_BODY_BYTES {
                return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
            }
            content_length = Some(parsed);
        }
    }
    let framing = match (content_length, transfer_encoding) {
        (Some(_), true) => return Err(failure(NativeHealthFailureKind::MalformedResponse)),
        (Some(content_length), false) => NativeBodyFraming::ContentLength(content_length),
        (None, true) => NativeBodyFraming::Chunked,
        (None, false) => return Err(failure(NativeHealthFailureKind::MalformedResponse)),
    };
    let content_type =
        content_type.ok_or_else(|| failure(NativeHealthFailureKind::MalformedResponse))?;
    let valid_content_type = match expected_content_type {
        NativeContentType::Json => is_json_content_type(content_type),
        NativeContentType::PlainText => is_plain_text_content_type(content_type),
    };
    if !valid_content_type {
        return Err(failure(NativeHealthFailureKind::MalformedResponse));
    }
    Ok((status, framing))
}

fn read_chunked_body(
    stream: &mut UnixStream,
    prefix: &[u8],
    context: &OperationContext<'_>,
) -> Result<Vec<u8>, NativeHealthFailure> {
    let mut reader = ChunkedBodyReader::new(stream, prefix, context);
    let mut body = Vec::new();
    let mut chunk_count = 0usize;
    let mut framing_bytes = 0usize;

    loop {
        check_active(context)?;
        let line = reader.read_line(MAX_CHUNK_LINE_BYTES)?;
        framing_bytes = framing_bytes
            .checked_add(line.len() + 2)
            .ok_or_else(|| failure(NativeHealthFailureKind::ResponseTooLarge))?;
        if framing_bytes > MAX_CHUNKED_OVERHEAD_BYTES {
            return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
        }
        let size = parse_chunk_size(&line)?;
        if size == 0 {
            if reader.consume_empty_trailer_line()? {
                framing_bytes = framing_bytes
                    .checked_add(2)
                    .ok_or_else(|| failure(NativeHealthFailureKind::ResponseTooLarge))?;
                if framing_bytes > MAX_CHUNKED_OVERHEAD_BYTES {
                    return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
                }
            } else {
                return Err(failure(NativeHealthFailureKind::MalformedResponse));
            }
            if reader.has_buffered_bytes() {
                return Err(failure(NativeHealthFailureKind::MalformedResponse));
            }
            return Ok(body);
        }

        chunk_count = chunk_count
            .checked_add(1)
            .ok_or_else(|| failure(NativeHealthFailureKind::ResponseTooLarge))?;
        if chunk_count > MAX_CHUNK_COUNT {
            return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
        }
        let total_body_bytes = body
            .len()
            .checked_add(size)
            .ok_or_else(|| failure(NativeHealthFailureKind::ResponseTooLarge))?;
        if total_body_bytes > MAX_BODY_BYTES {
            return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
        }
        framing_bytes = framing_bytes
            .checked_add(2)
            .ok_or_else(|| failure(NativeHealthFailureKind::ResponseTooLarge))?;
        if framing_bytes > MAX_CHUNKED_OVERHEAD_BYTES {
            return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
        }

        body.try_reserve(size)
            .map_err(|_| failure(NativeHealthFailureKind::ResponseTooLarge))?;
        let previous_len = body.len();
        body.resize(total_body_bytes, 0);
        reader.read_exact(&mut body[previous_len..])?;
        let mut chunk_terminator = [0u8; 2];
        reader.read_exact(&mut chunk_terminator)?;
        if chunk_terminator != *b"\r\n" {
            return Err(failure(NativeHealthFailureKind::MalformedResponse));
        }
    }
}

fn parse_chunk_size(line: &[u8]) -> Result<usize, NativeHealthFailure> {
    if line.is_empty() || !line.iter().all(u8::is_ascii_hexdigit) {
        // This also rejects chunk extensions; this narrow parser accepts only
        // the exact hexadecimal size token emitted by Go's standard writer.
        return Err(failure(NativeHealthFailureKind::MalformedResponse));
    }
    line.iter().try_fold(0usize, |size, byte| {
        let digit = match *byte {
            b'0'..=b'9' => usize::from(*byte - b'0'),
            b'a'..=b'f' => usize::from(*byte - b'a' + 10),
            b'A'..=b'F' => usize::from(*byte - b'A' + 10),
            _ => return Err(failure(NativeHealthFailureKind::MalformedResponse)),
        };
        size.checked_mul(16)
            .and_then(|size| size.checked_add(digit))
            .ok_or_else(|| failure(NativeHealthFailureKind::ResponseTooLarge))
    })
}

struct ChunkedBodyReader<'a, 'context> {
    stream: &'a mut UnixStream,
    prefix: &'a [u8],
    prefix_offset: usize,
    read_buffer: [u8; READ_CHUNK_BYTES],
    read_offset: usize,
    read_len: usize,
    context: &'a OperationContext<'context>,
}

impl<'a, 'context> ChunkedBodyReader<'a, 'context> {
    fn new(
        stream: &'a mut UnixStream,
        prefix: &'a [u8],
        context: &'a OperationContext<'context>,
    ) -> Self {
        Self {
            stream,
            prefix,
            prefix_offset: 0,
            read_buffer: [0; READ_CHUNK_BYTES],
            read_offset: 0,
            read_len: 0,
            context,
        }
    }

    fn read_exact(&mut self, output: &mut [u8]) -> Result<(), NativeHealthFailure> {
        let mut offset = 0;
        while offset < output.len() {
            check_active(self.context)?;
            if self.prefix_offset < self.prefix.len() {
                let count = (self.prefix.len() - self.prefix_offset).min(output.len() - offset);
                output[offset..offset + count]
                    .copy_from_slice(&self.prefix[self.prefix_offset..self.prefix_offset + count]);
                self.prefix_offset += count;
                offset += count;
                continue;
            }
            if self.read_offset == self.read_len {
                self.read_len = read_some(self.stream, &mut self.read_buffer, self.context)?;
                self.read_offset = 0;
                if self.read_len == 0 {
                    return Err(failure(NativeHealthFailureKind::TruncatedResponse));
                }
            }
            let count = (self.read_len - self.read_offset).min(output.len() - offset);
            output[offset..offset + count]
                .copy_from_slice(&self.read_buffer[self.read_offset..self.read_offset + count]);
            self.read_offset += count;
            offset += count;
        }
        Ok(())
    }

    fn read_line(&mut self, maximum: usize) -> Result<Vec<u8>, NativeHealthFailure> {
        let mut line = Vec::with_capacity(maximum.min(32));
        loop {
            let mut byte = [0u8; 1];
            self.read_exact(&mut byte)?;
            if byte[0] == b'\r' {
                let mut terminator = [0u8; 1];
                self.read_exact(&mut terminator)?;
                if terminator[0] != b'\n' {
                    return Err(failure(NativeHealthFailureKind::MalformedResponse));
                }
                return Ok(line);
            }
            if byte[0] == b'\n' {
                return Err(failure(NativeHealthFailureKind::MalformedResponse));
            }
            if line.len() >= maximum {
                return Err(failure(NativeHealthFailureKind::ResponseTooLarge));
            }
            line.push(byte[0]);
        }
    }

    fn consume_empty_trailer_line(&mut self) -> Result<bool, NativeHealthFailure> {
        let mut first = [0u8; 1];
        self.read_exact(&mut first)?;
        if first[0] != b'\r' {
            // Trailers are deliberately unsupported. Reject without reading or
            // retaining arbitrary upstream metadata.
            return Ok(false);
        }
        let mut second = [0u8; 1];
        self.read_exact(&mut second)?;
        Ok(second[0] == b'\n')
    }

    fn has_buffered_bytes(&self) -> bool {
        self.prefix_offset < self.prefix.len() || self.read_offset < self.read_len
    }
}

fn is_json_content_type(value: &[u8]) -> bool {
    let Some(separator) = value.iter().position(|byte| *byte == b';') else {
        return trim_ows(value).eq_ignore_ascii_case(b"application/json");
    };
    if !trim_ows(&value[..separator]).eq_ignore_ascii_case(b"application/json") {
        return false;
    }
    let parameter = trim_ows(&value[separator + 1..]);
    let Some(equals) = parameter.iter().position(|byte| *byte == b'=') else {
        return false;
    };
    if !trim_ows(&parameter[..equals]).eq_ignore_ascii_case(b"charset") {
        return false;
    }
    let mut charset = trim_ows(&parameter[equals + 1..]);
    if charset.len() >= 2 && charset.first() == Some(&b'"') && charset.last() == Some(&b'"') {
        charset = &charset[1..charset.len() - 1];
    }
    charset.eq_ignore_ascii_case(b"utf-8") || charset.eq_ignore_ascii_case(b"utf8")
}

fn is_plain_text_content_type(value: &[u8]) -> bool {
    let Some(separator) = value.iter().position(|byte| *byte == b';') else {
        return trim_ows(value).eq_ignore_ascii_case(b"text/plain");
    };
    if !trim_ows(&value[..separator]).eq_ignore_ascii_case(b"text/plain") {
        return false;
    }
    let parameter = trim_ows(&value[separator + 1..]);
    let Some(equals) = parameter.iter().position(|byte| *byte == b'=') else {
        return false;
    };
    if !trim_ows(&parameter[..equals]).eq_ignore_ascii_case(b"charset") {
        return false;
    }
    let mut charset = trim_ows(&parameter[equals + 1..]);
    if charset.len() >= 2 && charset.first() == Some(&b'"') && charset.last() == Some(&b'"') {
        charset = &charset[1..charset.len() - 1];
    }
    charset.eq_ignore_ascii_case(b"utf-8") || charset.eq_ignore_ascii_case(b"utf8")
}

fn split_crlf_lines(bytes: &[u8]) -> Result<Vec<&[u8]>, NativeHealthFailure> {
    let mut lines = Vec::new();
    let mut remaining = bytes;
    loop {
        match find_subslice(remaining, b"\r\n") {
            Some(offset) => {
                lines.push(&remaining[..offset]);
                remaining = &remaining[offset + 2..];
            }
            None => {
                if remaining.contains(&b'\n') || remaining.contains(&b'\r') {
                    return Err(failure(NativeHealthFailureKind::MalformedResponse));
                }
                lines.push(remaining);
                return Ok(lines);
            }
        }
    }
}

fn is_header_token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn trim_ows(mut value: &[u8]) -> &[u8] {
    while value.first() == Some(&b' ') {
        value = &value[1..];
    }
    while value.last() == Some(&b' ') {
        value = &value[..value.len() - 1];
    }
    value
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn validate_native_response(
    kind: NativeEngineKind,
    status: u16,
    bytes: &[u8],
) -> Result<(), NativeHealthFailure> {
    if status != 200 {
        return Err(failure(NativeHealthFailureKind::Unhealthy));
    }
    let healthy = match kind {
        NativeEngineKind::LlamaServer => {
            let body: LlamaServerHealthResponse = deserialize_json_object(bytes)
                .map_err(|_| failure(NativeHealthFailureKind::MalformedResponse))?;
            body.status == "ok"
        }
        NativeEngineKind::Comfy => {
            let body: ComfySystemStatsResponse = deserialize_json_object(bytes)
                .map_err(|_| failure(NativeHealthFailureKind::MalformedResponse))?;
            let _discarded_details = (body.system, body.devices);
            true
        }
    };
    if healthy {
        Ok(())
    } else {
        Err(failure(NativeHealthFailureKind::Unhealthy))
    }
}

fn validate_router_liveness(status: u16, body: &[u8]) -> Result<(), NativeHealthFailure> {
    if status != 200 {
        return Err(failure(NativeHealthFailureKind::Unhealthy));
    }
    if body == b"OK" {
        Ok(())
    } else {
        Err(failure(NativeHealthFailureKind::Unhealthy))
    }
}

fn validate_router_model_readiness(
    status: u16,
    bytes: &[u8],
    selected_model: &str,
) -> Result<(), NativeHealthFailure> {
    if status != 200 {
        return Err(failure(NativeHealthFailureKind::Unhealthy));
    }
    let response: LlamaSwapRunningResponse = deserialize_json_object(bytes)
        .map_err(|_| failure(NativeHealthFailureKind::MalformedResponse))?;
    let mut selected = response
        .running
        .iter()
        .filter(|model| model.0.model == selected_model);
    let Some(model) = selected.next() else {
        return Err(failure(NativeHealthFailureKind::Unhealthy));
    };
    if selected.next().is_some() {
        return Err(failure(NativeHealthFailureKind::MalformedResponse));
    }
    if model.0.state == "ready" {
        Ok(())
    } else {
        Err(failure(NativeHealthFailureKind::Unhealthy))
    }
}

fn write_all_bounded(
    stream: &mut UnixStream,
    bytes: &[u8],
    context: &OperationContext<'_>,
) -> Result<(), NativeHealthFailure> {
    write_all_bounded_tracking(stream, bytes, context).map_err(|(error, _)| error)
}

fn write_all_bounded_tracking(
    stream: &mut UnixStream,
    mut bytes: &[u8],
    context: &OperationContext<'_>,
) -> Result<(), (NativeHealthFailure, bool)> {
    let mut wrote_any_bytes = false;
    while !bytes.is_empty() {
        check_active(context).map_err(|error| (error, wrote_any_bytes))?;
        match stream.write(bytes) {
            Ok(0) => {
                return Err((failure(NativeHealthFailureKind::Io), wrote_any_bytes));
            }
            Ok(count) => {
                wrote_any_bytes = true;
                bytes = &bytes[count..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                wait_for_io(stream, libc::POLLOUT, context)
                    .map_err(|error| (error, wrote_any_bytes))?;
            }
            Err(_) => {
                return Err((failure(NativeHealthFailureKind::Io), wrote_any_bytes));
            }
        }
    }
    check_active(context).map_err(|error| (error, wrote_any_bytes))
}

fn read_some(
    stream: &mut UnixStream,
    bytes: &mut [u8],
    context: &OperationContext<'_>,
) -> Result<usize, NativeHealthFailure> {
    loop {
        check_active(context)?;
        match stream.read(bytes) {
            Ok(count) => return Ok(count),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                wait_for_io(stream, libc::POLLIN, context)?
            }
            Err(_) => return Err(failure(NativeHealthFailureKind::Io)),
        }
    }
}

fn wait_for_io(
    stream: &UnixStream,
    events: libc::c_short,
    context: &OperationContext<'_>,
) -> Result<(), NativeHealthFailure> {
    loop {
        check_active(context)?;
        let timeout = context
            .remaining()
            .min(IO_POLL)
            .as_millis()
            .clamp(1, i32::MAX as u128) as i32;
        let mut descriptor = libc::pollfd {
            fd: stream.as_raw_fd(),
            events,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, timeout) };
        if ready < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(failure(NativeHealthFailureKind::Io));
        }
        if ready == 0 {
            continue;
        }
        if descriptor.revents & libc::POLLNVAL != 0 {
            return Err(failure(NativeHealthFailureKind::Io));
        }
        check_active(context)?;
        return Ok(());
    }
}

fn check_active(context: &OperationContext<'_>) -> Result<(), NativeHealthFailure> {
    if context.is_cancelled() {
        Err(failure(NativeHealthFailureKind::Cancelled))
    } else if context.remaining().is_zero() {
        Err(failure(NativeHealthFailureKind::DeadlineExceeded))
    } else {
        Ok(())
    }
}

fn map_connect_error(error: VerifiedConnectError) -> NativeHealthFailure {
    let kind = match error {
        VerifiedConnectError::InvalidPeer => NativeHealthFailureKind::WrongServiceRole,
        VerifiedConnectError::SocketUnavailable => NativeHealthFailureKind::SocketUnavailable,
        VerifiedConnectError::PeerMismatch => NativeHealthFailureKind::PeerMismatch,
        VerifiedConnectError::Cancelled => NativeHealthFailureKind::Cancelled,
        VerifiedConnectError::DeadlineExceeded => NativeHealthFailureKind::DeadlineExceeded,
        VerifiedConnectError::Io => NativeHealthFailureKind::Io,
    };
    failure(kind)
}

fn failure(kind: NativeHealthFailureKind) -> NativeHealthFailure {
    NativeHealthFailure::new(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    use super::super::runtime::{Cancellation, NeverCancel};

    struct Fixture {
        directory: tempfile::TempDir,
        listener: UnixListener,
        socket: PinnedEngineSocket,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
            let path = directory
                .path()
                .join(super::super::engine_transport::ENGINE_SOCKET_NAME);
            let listener = UnixListener::bind(&path).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
            let socket = PinnedEngineSocket::open(directory.path()).unwrap();
            Self {
                directory,
                listener,
                socket,
            }
        }

        fn connect(
            &self,
            context: &OperationContext<'_>,
        ) -> Result<UnixStream, NativeHealthFailure> {
            let (pid, uid) = (std::process::id(), unsafe { libc::geteuid() });
            self.socket
                .connect_verified(pid, uid, context)
                .map_err(map_connect_error)
        }

        fn spawn_server(
            &self,
            path: &'static str,
            response: Option<Vec<u8>>,
            delay: Duration,
        ) -> thread::JoinHandle<()> {
            let listener = self.listener.try_clone().unwrap();
            thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                serve(stream, path, response, delay);
            })
        }

        fn spawn_server_sequence(
            &self,
            responses: Vec<(&'static str, Vec<u8>, Duration)>,
        ) -> thread::JoinHandle<()> {
            let listener = self.listener.try_clone().unwrap();
            thread::spawn(move || {
                for (path, response, delay) in responses {
                    let (stream, _) = listener.accept().unwrap();
                    serve(stream, path, Some(response), delay);
                }
            })
        }

        fn spawn_server_fragments(
            &self,
            path: &'static str,
            fragments: Vec<(Vec<u8>, Duration)>,
        ) -> thread::JoinHandle<()> {
            let listener = self.listener.try_clone().unwrap();
            thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let mut byte = [0u8; 1];
                    if stream.read_exact(&mut byte).is_err() {
                        return;
                    }
                    request.push(byte[0]);
                    assert!(request.len() <= 1024);
                }
                let expected_request_line = format!("GET {path} HTTP/1.1\r\n");
                assert!(request.starts_with(expected_request_line.as_bytes()));
                for (fragment, delay) in fragments {
                    thread::sleep(delay);
                    if stream.write_all(&fragment).is_err() {
                        return;
                    }
                }
            })
        }
    }

    struct TestCancellation(Arc<AtomicBool>);

    impl TestCancellation {
        fn cancel(&self) {
            self.0.store(true, Ordering::Release);
        }
    }

    impl Cancellation for TestCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    fn context(timeout: Duration) -> OperationContext<'static> {
        static NEVER: NeverCancel = NeverCancel;
        OperationContext::new(Instant::now() + timeout, &NEVER)
    }

    fn serve(mut stream: UnixStream, path: &str, response: Option<Vec<u8>>, delay: Duration) {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            let mut byte = [0u8; 1];
            if stream.read_exact(&mut byte).is_err() {
                return;
            }
            request.push(byte[0]);
            assert!(request.len() <= 1024);
        }
        let expected_request_line = format!("GET {path} HTTP/1.1\r\n");
        assert!(request.starts_with(expected_request_line.as_bytes()));
        thread::sleep(delay);
        if let Some(response) = response {
            let _ = stream.write_all(&response);
        }
    }

    fn response(status: u16, body: &[u8]) -> Vec<u8> {
        response_with_content_type(status, "application/json", body)
    }

    fn response_with_content_type(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\nX-Native-Extension: accepted\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(body);
        response
    }

    fn chunked_response(
        status: u16,
        content_type: &str,
        body: &[u8],
        chunk_size: usize,
    ) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        .into_bytes();
        for chunk in body.chunks(chunk_size.max(1)) {
            response.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            response.extend_from_slice(chunk);
            response.extend_from_slice(b"\r\n");
        }
        response.extend_from_slice(b"0\r\n\r\n");
        response
    }

    fn connected_probe(
        fixture: &Fixture,
        kind: NativeEngineKind,
        context: &OperationContext<'_>,
    ) -> Result<(), NativeHealthFailure> {
        let mut stream = fixture.connect(context)?;
        request_get(&mut stream, kind.path(), "application/json", context)?;
        let response = read_response(&mut stream, context, NativeContentType::Json, false)?;
        validate_native_response(kind, response.status, &response.body)
    }

    fn connected_router_readiness_probe(
        fixture: &Fixture,
        selected_model: &str,
        context: &OperationContext<'_>,
    ) -> Result<(), NativeHealthFailure> {
        let mut health_stream = fixture.connect(context)?;
        request_get(&mut health_stream, "/health", "text/plain", context)?;
        let health_response = read_response(
            &mut health_stream,
            context,
            NativeContentType::PlainText,
            false,
        )?;
        validate_router_liveness(health_response.status, &health_response.body)?;

        let mut running_stream = fixture.connect(context)?;
        request_get(&mut running_stream, "/running", "application/json", context)?;
        let running_response =
            read_response(&mut running_stream, context, NativeContentType::Json, true)?;
        validate_router_model_readiness(
            running_response.status,
            &running_response.body,
            selected_model,
        )
    }

    fn connected_running_response(
        fixture: &Fixture,
        context: &OperationContext<'_>,
    ) -> Result<NativeResponse, NativeHealthFailure> {
        let mut stream = fixture.connect(context)?;
        request_get(&mut stream, "/running", "application/json", context)?;
        read_response(&mut stream, context, NativeContentType::Json, true)
    }

    #[test]
    fn llama_health_requires_native_ready_response() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(response_with_content_type(
                200,
                "application/json; charset=UTF-8",
                br#"{"status":"ok"}"#,
            )),
            Duration::ZERO,
        );
        connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap();
        server.join().unwrap();
        assert!(fixture.directory.path().exists());
    }

    #[test]
    fn llama_swap_router_readiness_uses_plaintext_health_and_exact_ready_state() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server_sequence(vec![
            (
                "/health",
                response_with_content_type(200, "text/plain; charset=utf-8", b"OK"),
                Duration::ZERO,
            ),
            (
                "/running",
                chunked_response(
                    200,
                    "application/json",
                    br#"{"running":[{"model":"model.with-characters","state":"ready","cmd":"/private/model/path"}]}"#,
                    11,
                ),
                Duration::ZERO,
            ),
        ]);
        connected_router_readiness_probe(
            &fixture,
            "model.with-characters",
            &context(Duration::from_secs(2)),
        )
        .unwrap();
        server.join().unwrap();
    }

    #[test]
    fn chunked_framing_is_allowed_only_for_router_and_rejects_ambiguity() {
        let valid =
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked";
        assert!(parse_response_head(valid, NativeContentType::Json, true).is_ok());
        assert_eq!(
            parse_response_head(valid, NativeContentType::Json, false)
                .unwrap_err()
                .kind(),
            NativeHealthFailureKind::MalformedResponse
        );

        let invalid_heads = [
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nContent-Length: 0".as_slice(),
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: chunked".as_slice(),
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: gzip".as_slice(),
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked, gzip".as_slice(),
        ];
        for head in invalid_heads {
            let error = parse_response_head(head, NativeContentType::Json, true).unwrap_err();
            assert_eq!(error.kind(), NativeHealthFailureKind::MalformedResponse);
        }
    }

    #[test]
    fn chunked_running_response_rejects_extensions_trailers_truncation_and_oversize() {
        let invalid_bodies = [
            (
                b"1;extension\r\nx\r\n0\r\n\r\n".as_slice(),
                NativeHealthFailureKind::MalformedResponse,
            ),
            (
                b"0\r\nX-Secret: value\r\n\r\n".as_slice(),
                NativeHealthFailureKind::MalformedResponse,
            ),
            (
                b"4\r\nab".as_slice(),
                NativeHealthFailureKind::TruncatedResponse,
            ),
            (
                b"80001\r\n".as_slice(),
                NativeHealthFailureKind::ResponseTooLarge,
            ),
        ];
        for (encoded_body, expected_kind) in invalid_bodies {
            let fixture = Fixture::new();
            let mut wire = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
            wire.extend_from_slice(encoded_body);
            let server = fixture.spawn_server("/running", Some(wire), Duration::ZERO);
            let error =
                connected_running_response(&fixture, &context(Duration::from_secs(2))).unwrap_err();
            assert_eq!(error.kind(), expected_kind);
            server.join().unwrap();
        }

        let mut many_chunks = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for _ in 0..=MAX_CHUNK_COUNT {
            many_chunks.extend_from_slice(b"1\r\nx\r\n");
        }
        many_chunks.extend_from_slice(b"0\r\n\r\n");
        let fixture = Fixture::new();
        let server = fixture.spawn_server("/running", Some(many_chunks), Duration::ZERO);
        let error =
            connected_running_response(&fixture, &context(Duration::from_secs(2))).unwrap_err();
        assert_eq!(error.kind(), NativeHealthFailureKind::ResponseTooLarge);
        server.join().unwrap();

        let mut excessive_overhead = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for _ in 0..3300 {
            excessive_overhead.extend_from_slice(b"0000000000000001\r\nx\r\n");
        }
        excessive_overhead.extend_from_slice(b"0\r\n\r\n");
        let fixture = Fixture::new();
        let server = fixture.spawn_server("/running", Some(excessive_overhead), Duration::ZERO);
        let error =
            connected_running_response(&fixture, &context(Duration::from_secs(2))).unwrap_err();
        assert_eq!(error.kind(), NativeHealthFailureKind::ResponseTooLarge);
        server.join().unwrap();

        let mut oversized_header = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nX-Fill: ".to_vec();
        let padded_length = oversized_header.len() + MAX_HEADER_BYTES;
        oversized_header.resize(padded_length, b'a');
        oversized_header.extend_from_slice(b"\r\n\r\n");
        let fixture = Fixture::new();
        let server = fixture.spawn_server("/running", Some(oversized_header), Duration::ZERO);
        let error =
            connected_running_response(&fixture, &context(Duration::from_secs(2))).unwrap_err();
        assert_eq!(error.kind(), NativeHealthFailureKind::ResponseTooLarge);
        server.join().unwrap();
    }

    #[test]
    fn chunked_running_body_read_honors_deadline_and_cancellation_mid_chunk() {
        fn fragments() -> Vec<(Vec<u8>, Duration)> {
            let mut first = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nab".to_vec();
            let second = b"cd\r\n0\r\n\r\n".to_vec();
            first.shrink_to_fit();
            vec![
                (first, Duration::ZERO),
                (second, Duration::from_millis(180)),
            ]
        }

        let fixture = Fixture::new();
        let server = fixture.spawn_server_fragments("/running", fragments());
        let error =
            connected_running_response(&fixture, &context(Duration::from_millis(70))).unwrap_err();
        assert_eq!(error.kind(), NativeHealthFailureKind::DeadlineExceeded);
        server.join().unwrap();

        let fixture = Fixture::new();
        let server = fixture.spawn_server_fragments("/running", fragments());
        let cancellation = Arc::new(TestCancellation(Arc::new(AtomicBool::new(false))));
        let cancel_signal = Arc::clone(&cancellation);
        let signal = thread::spawn(move || {
            thread::sleep(Duration::from_millis(60));
            cancel_signal.cancel();
        });
        let context = OperationContext::new(
            Instant::now() + Duration::from_secs(2),
            cancellation.as_ref(),
        );
        let error = connected_running_response(&fixture, &context).unwrap_err();
        assert_eq!(error.kind(), NativeHealthFailureKind::Cancelled);
        signal.join().unwrap();
        server.join().unwrap();
    }

    #[test]
    fn activation_request_is_one_encoded_private_socket_get_and_ack_is_redacted() {
        let fixture = Fixture::new();
        let listener = fixture.listener.try_clone().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0u8; 1];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
                assert!(request.len() <= 1024);
            }
            request
        });
        let mut stream = fixture.connect(&context(Duration::from_secs(2))).unwrap();
        let selected_model = validated_selected_model("private.model-id").unwrap();
        request_llama_swap_model_health_once(
            &mut stream,
            &selected_model,
            &context(Duration::from_secs(2)),
        )
        .unwrap();
        let request = server.join().unwrap();
        assert_eq!(
            request,
            b"GET /upstream/private.model-id/health HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
        );

        let mut encoded = Vec::new();
        append_percent_encoded_path_segment(&mut encoded, "model/with?reserved%chars");
        assert_eq!(encoded, b"model%2Fwith%3Freserved%25chars");

        let acknowledgment = RouterModelActivationAck {
            selected_model: validated_selected_model("private.model-id").unwrap(),
            acknowledged_at: Instant::now(),
        };
        assert!(acknowledgment.matches_model("private.model-id"));
        assert!(!acknowledgment.matches_model("another-model"));
        assert!(!format!("{acknowledgment:?}").contains("private.model-id"));

        let failure = RouterModelActivationFailure::new(
            ActivationDisposition::Unconfirmed,
            NativeHealthFailureKind::Io,
        );
        assert_eq!(
            failure.to_string(),
            "llama-swap model activation outcome is unconfirmed"
        );
        assert_eq!(failure.disposition(), ActivationDisposition::Unconfirmed);
        assert_eq!(failure.kind(), NativeHealthFailureKind::Io);
        assert!(!format!("{failure:?}").contains("private.model-id"));
    }

    #[test]
    fn activation_write_progress_distinguishes_no_dispatch_from_uncertain() {
        let (mut stream, _peer) = UnixStream::pair().unwrap();
        let cancellation = TestCancellation(Arc::new(AtomicBool::new(true)));
        let cancelled_context =
            OperationContext::new(Instant::now() + Duration::from_secs(2), &cancellation);
        let (error, wrote_any_bytes) =
            write_all_bounded_tracking(&mut stream, b"GET /health", &cancelled_context)
                .unwrap_err();
        assert!(!wrote_any_bytes);
        let failure = activation_write_failure(error, wrote_any_bytes);
        assert_eq!(failure.disposition(), ActivationDisposition::NoDispatch);
        assert_eq!(failure.kind(), NativeHealthFailureKind::Cancelled);

        let (mut stream, _peer) = UnixStream::pair().unwrap();
        stream.set_nonblocking(true).unwrap();
        let request = vec![b'x'; 1024 * 1024];
        let (error, wrote_any_bytes) =
            write_all_bounded_tracking(&mut stream, &request, &context(Duration::from_millis(50)))
                .unwrap_err();
        assert!(wrote_any_bytes);
        let failure = activation_write_failure(error, wrote_any_bytes);
        assert_eq!(failure.disposition(), ActivationDisposition::Unconfirmed);
        assert!(matches!(
            failure.kind(),
            NativeHealthFailureKind::DeadlineExceeded | NativeHealthFailureKind::Io
        ));
    }

    #[test]
    fn router_readiness_observation_is_bound_to_its_profile_id_and_redacted() {
        let observation = RouterModelReadinessObservation {
            selected_model: validated_selected_model("private-model").unwrap(),
            observed_at: Instant::now(),
        };
        assert!(observation.matches_model("private-model"));
        assert!(!observation.matches_model("other-model"));
        assert!(!format!("{observation:?}").contains("private-model"));
    }

    #[test]
    fn invalid_router_profile_ids_are_fixed_redacted_failures() {
        let too_long = "x".repeat(129);
        for input in ["", "../private-model", "model?query", too_long.as_str()] {
            let failure = validated_selected_model(input).unwrap_err();
            assert_eq!(failure.kind(), NativeHealthFailureKind::Unhealthy);
            assert_eq!(failure.to_string(), "native engine reports unhealthy");
            assert!(!format!("{failure:?}").contains("private-model"));
            if !input.is_empty() {
                assert!(!failure.to_string().contains(input));
                assert!(!format!("{failure:?}").contains(input));
            }
        }
    }

    #[test]
    fn llama_swap_loaded_presence_is_not_ready_and_errors_do_not_echo_listing() {
        // Upstream /v1/models maps both process StateStarting and StateReady
        // into `status.value = "loaded"`. Only /running's exact `ready` state
        // can pass this probe.
        for state in ["starting", "stopping", "unloaded", "unknown", "READY"] {
            let fixture = Fixture::new();
            let running_body = format!(
                r#"{{"running":[{{"model":"secret-model-id","state":"{state}","cmd":"/secret/path"}}]}}"#
            );
            let server = fixture.spawn_server_sequence(vec![
                (
                    "/health",
                    response_with_content_type(200, "text/plain", b"OK"),
                    Duration::ZERO,
                ),
                (
                    "/running",
                    response(200, running_body.as_bytes()),
                    Duration::ZERO,
                ),
            ]);
            let failure = connected_router_readiness_probe(
                &fixture,
                "secret-model-id",
                &context(Duration::from_secs(2)),
            )
            .unwrap_err();
            assert_eq!(failure.kind(), NativeHealthFailureKind::Unhealthy);
            assert!(!format!("{failure:?}").contains("secret-model-id"));
            assert!(!failure.to_string().contains("/secret/path"));
            server.join().unwrap();
        }
    }

    #[test]
    fn llama_swap_readiness_rejects_ambiguous_or_malformed_running_state() {
        for body in [
            br#"{"running":[]}"#.as_slice(),
            br#"{"running":[{"model":"selected","state":"ready"},{"model":"selected","state":"ready"}]}"#.as_slice(),
            br#"{"running":[{"model":"selected","state":"ready","state":"starting"}]}"#.as_slice(),
            br#"{"running":[{"model":"selected"}]}"#.as_slice(),
            br#"{"running":[],"running":[]}"#.as_slice(),
            br#"[[["selected","ready"]]]"#.as_slice(),
            br#"{"running":[["selected","ready"]]}"#.as_slice(),
        ] {
            let failure = validate_router_model_readiness(200, body, "selected").unwrap_err();
            assert!(matches!(
                failure.kind(),
                NativeHealthFailureKind::MalformedResponse | NativeHealthFailureKind::Unhealthy
            ));
        }

        let failure = validate_router_liveness(200, b"{\"status\":\"ok\"}").unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::Unhealthy);
        let failure = validate_router_liveness(503, b"OK").unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::Unhealthy);
        assert_eq!(
            validate_native_response(NativeEngineKind::LlamaServer, 200, br#"["ok"]"#)
                .unwrap_err()
                .kind(),
            NativeHealthFailureKind::MalformedResponse
        );
        assert_eq!(
            validate_native_response(NativeEngineKind::Comfy, 200, br#"[{},{ }]"#)
                .unwrap_err()
                .kind(),
            NativeHealthFailureKind::MalformedResponse
        );
    }

    #[test]
    fn llama_swap_router_probe_uses_one_deadline_for_both_routes() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server_sequence(vec![
            (
                "/health",
                response_with_content_type(200, "text/plain", b"OK"),
                Duration::from_millis(40),
            ),
            (
                "/running",
                response(
                    200,
                    br#"{"running":[{"model":"selected","state":"ready"}]}"#,
                ),
                Duration::from_millis(40),
            ),
        ]);
        let failure = connected_router_readiness_probe(
            &fixture,
            "selected",
            &context(Duration::from_millis(60)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::DeadlineExceeded);
        server.join().unwrap();
    }

    #[test]
    fn llama_swap_router_probe_honors_cancellation_while_reading_running_state() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server_sequence(vec![
            (
                "/health",
                response_with_content_type(200, "text/plain", b"OK"),
                Duration::from_millis(20),
            ),
            (
                "/running",
                response(
                    200,
                    br#"{"running":[{"model":"selected","state":"ready"}]}"#,
                ),
                Duration::from_millis(120),
            ),
        ]);
        let cancellation = Arc::new(TestCancellation(Arc::new(AtomicBool::new(false))));
        let cancel_signal = Arc::clone(&cancellation);
        let signal = thread::spawn(move || {
            thread::sleep(Duration::from_millis(60));
            cancel_signal.cancel();
        });
        let context = OperationContext::new(
            Instant::now() + Duration::from_secs(2),
            cancellation.as_ref(),
        );
        let failure = connected_router_readiness_probe(&fixture, "selected", &context).unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::Cancelled);
        signal.join().unwrap();
        server.join().unwrap();
    }

    #[test]
    fn llama_swap_health_requires_plaintext_content_type() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server_sequence(vec![(
            "/health",
            response_with_content_type(200, "application/json", b"OK"),
            Duration::ZERO,
        )]);
        let failure = connected_router_readiness_probe(
            &fixture,
            "selected",
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
        server.join().unwrap();
    }

    #[test]
    fn comfy_system_stats_requires_only_documented_top_level_shape() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::Comfy.path(),
            Some(response(
                200,
                br#"{"system":{"argv":["sensitive"]},"devices":[]}"#,
            )),
            Duration::ZERO,
        );
        let observation = connected_probe(
            &fixture,
            NativeEngineKind::Comfy,
            &context(Duration::from_secs(2)),
        );
        observation.unwrap();
        server.join().unwrap();
    }

    #[test]
    fn unhealthy_status_and_wrong_native_shape_are_fixed_failures() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(response(503, br#"{"loading":true}"#)),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::Unhealthy);
        assert_eq!(failure.to_string(), "native engine reports unhealthy");
        server.join().unwrap();

        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(response(200, br#"{"status":"loading"}"#)),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::Unhealthy);
        server.join().unwrap();
    }

    #[test]
    fn malformed_truncated_and_oversize_responses_fail_closed() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}"
                    .to_vec(),
            ),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
        server.join().unwrap();

        for forbidden_header in [
            b"Transfer-Encoding: chunked\r\n".as_slice(),
            b"Content-Encoding: gzip\r\n".as_slice(),
        ] {
            let fixture = Fixture::new();
            let mut raw_response =
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n".to_vec();
            raw_response.extend_from_slice(forbidden_header);
            raw_response.extend_from_slice(b"Content-Length: 2\r\n\r\n{}");
            let server = fixture.spawn_server(
                NativeEngineKind::LlamaServer.path(),
                Some(raw_response),
                Duration::ZERO,
            );
            let failure = connected_probe(
                &fixture,
                NativeEngineKind::LlamaServer,
                &context(Duration::from_secs(2)),
            )
            .unwrap_err();
            assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
            server.join().unwrap();
        }

        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 40\r\n\r\n{}"
                    .to_vec(),
            ),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::TruncatedResponse);
        server.join().unwrap();

        let fixture = Fixture::new();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        )
        .into_bytes();
        let server = fixture.spawn_server(
            NativeEngineKind::Comfy.path(),
            Some(response),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::Comfy,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::ResponseTooLarge);
        server.join().unwrap();
    }

    #[test]
    fn duplicate_known_json_fields_and_wrong_mime_fail_closed() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(response(200, br#"{"status":"loading","status":"ok"}"#)),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
        server.join().unwrap();

        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::Comfy.path(),
            Some(response(200, br#"{"system":{},"system":{},"devices":[]}"#)),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::Comfy,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
        server.join().unwrap();

        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::Comfy.path(),
            Some(response(200, br#"{"system":{},"devices":[],"devices":[]}"#)),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::Comfy,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
        server.join().unwrap();

        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}"
                    .to_vec(),
            ),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
        server.join().unwrap();

        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            Some(response_with_content_type(
                200,
                "text/plain",
                br#"{"status":"ok"}"#,
            )),
            Duration::ZERO,
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::MalformedResponse);
        server.join().unwrap();
    }

    #[test]
    fn response_wait_honors_deadline_and_cancellation() {
        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            None,
            Duration::from_millis(150),
        );
        let failure = connected_probe(
            &fixture,
            NativeEngineKind::LlamaServer,
            &context(Duration::from_millis(60)),
        )
        .unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::DeadlineExceeded);
        server.join().unwrap();

        let fixture = Fixture::new();
        let server = fixture.spawn_server(
            NativeEngineKind::LlamaServer.path(),
            None,
            Duration::from_millis(100),
        );
        let cancellation = Arc::new(TestCancellation(Arc::new(AtomicBool::new(false))));
        let cancel_signal = Arc::clone(&cancellation);
        let signal = thread::spawn(move || {
            thread::sleep(Duration::from_millis(40));
            cancel_signal.cancel();
        });
        let context = OperationContext::new(
            Instant::now() + Duration::from_secs(1),
            cancellation.as_ref(),
        );
        let failure =
            connected_probe(&fixture, NativeEngineKind::LlamaServer, &context).unwrap_err();
        assert_eq!(failure.kind(), NativeHealthFailureKind::Cancelled);
        signal.join().unwrap();
        server.join().unwrap();
    }

    #[test]
    fn verified_connect_rejects_wrong_supervisor_pid_even_for_same_uid() {
        let fixture = Fixture::new();
        let listener = fixture.listener.try_clone().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });
        let expected_pid = std::process::id().saturating_add(1);
        let error = fixture
            .socket
            .connect_verified(
                expected_pid,
                unsafe { libc::geteuid() },
                &context(Duration::from_secs(2)),
            )
            .unwrap_err();
        assert_eq!(error, VerifiedConnectError::PeerMismatch);
        server.join().unwrap();
    }
}
