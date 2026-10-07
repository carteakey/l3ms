//! Bounded native health probes for engines owned by an isolated supervisor.
//!
//! A successful probe means only that the expected supervisor answered its
//! fixed native route at one point in time. For `LlamaServer`, that route's
//! documented response means the server is ready; ComfyUI's `/system_stats`
//! is only a valid inspection snapshot, not model readiness. Neither result
//! is an admission, drain, cgroup-release, GPU-idle, or lease proof. Inference
//! methods are not called or replayed here. The response body is parsed and
//! discarded; ComfyUI's route can include command-line arguments.

use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Map, Value};

use super::managed_service::{ManagedServiceErrorKind, ManagedServiceState, OwnedService};
use super::native_http::{PinnedEngineSocket, VerifiedConnectError};
use super::runtime::OperationContext;
use super::service_owner::ManagedRuntimeOwner;

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 512 * 1024;
const READ_CHUNK_BYTES: usize = 4096;
const IO_POLL: Duration = Duration::from_millis(25);

#[derive(Deserialize)]
struct LlamaServerHealthResponse {
    status: String,
}

#[derive(Deserialize)]
struct ComfySystemStatsResponse {
    system: Map<String, Value>,
    devices: Vec<Value>,
}

/// Only the two native endpoints supported by this production slice.
/// `LlamaServer` is intentionally not the llama-swap router: llama-swap's
/// `/v1/models` and per-model load semantics are a separate lifecycle contract.
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
    check_active(context)?;
    require_running(owner, service, context)?;
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
    let mut stream = socket
        .connect_verified(expected_pid, expected_uid, context)
        .map_err(map_connect_error)?;
    request_health(&mut stream, kind.path(), context)?;
    let response = read_response(&mut stream, context)?;
    validate_native_response(kind, response.status, &response.body)?;
    check_active(context)?;
    require_running(owner, service, context)?;
    check_active(context)?;
    Ok(NativeHealthObservation {
        kind,
        observed_at: Instant::now(),
    })
}

fn require_running(
    owner: &ManagedRuntimeOwner<'_>,
    service: &OwnedService,
    context: &OperationContext<'_>,
) -> Result<(), NativeHealthFailure> {
    match owner.state(service, context) {
        Ok(ManagedServiceState::Running) => Ok(()),
        Ok(ManagedServiceState::Stopped) => Err(failure(NativeHealthFailureKind::NotRunning)),
        Ok(ManagedServiceState::Unknown) => Err(failure(NativeHealthFailureKind::OwnerUnverified)),
        Err(ManagedServiceErrorKind::Cancelled) => Err(failure(NativeHealthFailureKind::Cancelled)),
        Err(ManagedServiceErrorKind::DeadlineExceeded) => {
            Err(failure(NativeHealthFailureKind::DeadlineExceeded))
        }
        Err(_) => Err(failure(NativeHealthFailureKind::OwnerUnverified)),
    }
}

fn request_health(
    stream: &mut UnixStream,
    path: &str,
    context: &OperationContext<'_>,
) -> Result<(), NativeHealthFailure> {
    let mut request = Vec::with_capacity(128);
    request.extend_from_slice(b"GET ");
    request.extend_from_slice(path.as_bytes());
    request.extend_from_slice(
        b" HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\nConnection: close\r\n\r\n",
    );
    write_all_bounded(stream, &request, context)
}

struct NativeResponse {
    status: u16,
    body: Vec<u8>,
}

fn read_response(
    stream: &mut UnixStream,
    context: &OperationContext<'_>,
) -> Result<NativeResponse, NativeHealthFailure> {
    let mut buffered = Vec::with_capacity(READ_CHUNK_BYTES);
    let header_end = loop {
        if let Some(offset) = find_subslice(&buffered, b"\r\n\r\n") {
            if offset > MAX_HEADER_BYTES {
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

    let (status, content_length) = parse_response_head(&buffered[..header_end])?;
    let body_start = header_end + 4;
    let body_prefix = &buffered[body_start..];
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
    Ok(NativeResponse { status, body })
}

fn parse_response_head(bytes: &[u8]) -> Result<(u16, usize), NativeHealthFailure> {
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
        if name == b"transfer-encoding" || name == b"content-encoding" {
            // The native endpoints return bounded JSON directly. Do not add
            // decompression or chunk framing to this narrow health protocol.
            return Err(failure(NativeHealthFailureKind::MalformedResponse));
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
    let content_length =
        content_length.ok_or_else(|| failure(NativeHealthFailureKind::MalformedResponse))?;
    let content_type =
        content_type.ok_or_else(|| failure(NativeHealthFailureKind::MalformedResponse))?;
    if !is_json_content_type(content_type) {
        return Err(failure(NativeHealthFailureKind::MalformedResponse));
    }
    Ok((status, content_length))
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
            let body: LlamaServerHealthResponse = serde_json::from_slice(bytes)
                .map_err(|_| failure(NativeHealthFailureKind::MalformedResponse))?;
            body.status == "ok"
        }
        NativeEngineKind::Comfy => {
            let body: ComfySystemStatsResponse = serde_json::from_slice(bytes)
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

fn write_all_bounded(
    stream: &mut UnixStream,
    mut bytes: &[u8],
    context: &OperationContext<'_>,
) -> Result<(), NativeHealthFailure> {
    while !bytes.is_empty() {
        check_active(context)?;
        match stream.write(bytes) {
            Ok(0) => return Err(failure(NativeHealthFailureKind::Io)),
            Ok(count) => bytes = &bytes[count..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                wait_for_io(stream, libc::POLLOUT, context)?
            }
            Err(_) => return Err(failure(NativeHealthFailureKind::Io)),
        }
    }
    check_active(context)
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

    fn connected_probe(
        fixture: &Fixture,
        kind: NativeEngineKind,
        context: &OperationContext<'_>,
    ) -> Result<(), NativeHealthFailure> {
        let mut stream = fixture.connect(context)?;
        request_health(&mut stream, kind.path(), context)?;
        let response = read_response(&mut stream, context)?;
        validate_native_response(kind, response.status, &response.body)
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
