//! Explicit, bounded HTTP ingress to the existing private coordinator.
//!
//! The gateway is opt-in: callers must supply a loopback bind, the private
//! coordinator directory, and an opaque bearer token. It accepts one native
//! request on each HTTP/1.1 connection and performs exactly one IPC exchange.
//! It never starts an engine, retries inference, or falls back to an engine
//! endpoint. All socket IO is nonblocking and observes the caller's session
//! cancellation/deadline plus a per-connection absolute deadline.
//!
//! Each `serve` call is a bounded foreground session, not a long-running
//! installed gateway. The caller supplies its outer deadline; default limits
//! cap the session at 64 accepted TCP attempts and at 16 attempts from one peer
//! IP. Loopback clients from the same address share that peer allowance. A
//! caller must treat any lost HTTP acknowledgment for a validated Submit as
//! unconfirmed and query status using the same request ID; the response header
//! below is an additional signal when the HTTP error reaches the caller.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use super::client::{self, ClientError};
use super::ipc::{
    self, RequestEnvelope, ResponseEnvelope, ResponseErrorCode, ResponseMessage,
    WireRequestMessage, MAX_FRAME_BYTES, RESULTS_PROTOCOL_VERSION,
};
use super::runtime::{Cancellation, OperationContext};

const MIN_TOKEN_BYTES: usize = 32;
const MAX_TOKEN_BYTES: usize = 256;
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_HEADER_FIELDS: usize = 32;
const HTTP_READ_CHUNK: usize = 1024;
const IO_POLL_INTERVAL: Duration = Duration::from_millis(5);
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_SESSION_ATTEMPTS: usize = 64;
const MAX_SESSION_ATTEMPTS: usize = 256;
const DEFAULT_PEER_ATTEMPTS: usize = 16;
const MAX_PEER_ATTEMPTS: usize = 64;
const ROUTE: &str = "/v1/coordinator";
const UNCERTAIN_SUBMIT_HEADER: &str =
    "unconfirmed; query-status-with-same-request-id; do-not-resubmit";

/// An explicitly supplied opaque bearer credential. Its contents never appear
/// in its Debug representation and are never logged by this module.
pub struct BearerToken(String);

impl BearerToken {
    /// Create the required opaque token from explicitly supplied visible ASCII
    /// with a length from 32 through 256 bytes.
    pub fn new(value: impl Into<String>) -> Result<Self, GatewayConfigError> {
        let value = value.into();
        if !(MIN_TOKEN_BYTES..=MAX_TOKEN_BYTES).contains(&value.len())
            || !value.bytes().all(|byte| (0x21..=0x7e).contains(&byte))
        {
            return Err(GatewayConfigError::InvalidToken);
        }
        Ok(Self(value))
    }
}

impl fmt::Debug for BearerToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("BearerToken")
            .field(&"<redacted>")
            .finish()
    }
}

/// Required construction data and validated resource limits for one gateway
/// session. The token has no default; limits may be lowered for short sessions
/// but cannot exceed the module's fixed upper bounds.
pub struct HttpGatewayConfig {
    bind: SocketAddr,
    coordinator_directory: PathBuf,
    token: BearerToken,
    request_timeout: Duration,
    session_attempts: usize,
    peer_attempts: usize,
}

impl HttpGatewayConfig {
    pub fn new(
        bind: SocketAddr,
        coordinator_directory: impl Into<PathBuf>,
        token: BearerToken,
    ) -> Result<Self, GatewayConfigError> {
        if !bind.ip().is_loopback() {
            return Err(GatewayConfigError::NonLoopbackBind);
        }
        Ok(Self {
            bind,
            coordinator_directory: coordinator_directory.into(),
            token,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            session_attempts: DEFAULT_SESSION_ATTEMPTS,
            peer_attempts: DEFAULT_PEER_ATTEMPTS,
        })
    }

    /// Set bounded limits for callers that need a shorter foreground session.
    /// A zero limit, a timeout above 30 seconds, or limits above the fixed
    /// maxima is rejected.
    pub fn with_limits(
        mut self,
        request_timeout: Duration,
        session_attempts: usize,
        peer_attempts: usize,
    ) -> Result<Self, GatewayConfigError> {
        if request_timeout.is_zero() || request_timeout > MAX_REQUEST_TIMEOUT {
            return Err(GatewayConfigError::InvalidRequestTimeout);
        }
        if session_attempts == 0
            || session_attempts > MAX_SESSION_ATTEMPTS
            || peer_attempts == 0
            || peer_attempts > MAX_PEER_ATTEMPTS
            || peer_attempts > session_attempts
        {
            return Err(GatewayConfigError::InvalidConnectionLimits);
        }
        self.request_timeout = request_timeout;
        self.session_attempts = session_attempts;
        self.peer_attempts = peer_attempts;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatewayConfigError {
    InvalidToken,
    NonLoopbackBind,
    InvalidRequestTimeout,
    InvalidConnectionLimits,
}

impl fmt::Display for GatewayConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidToken => "gateway bearer token is invalid",
            Self::NonLoopbackBind => "HTTP gateway bind address must be loopback",
            Self::InvalidRequestTimeout => "HTTP gateway request timeout is invalid",
            Self::InvalidConnectionLimits => "HTTP gateway connection limits are invalid",
        })
    }
}

impl std::error::Error for GatewayConfigError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatewayError {
    BindUnavailable,
}

impl fmt::Display for GatewayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HTTP gateway could not bind its loopback listener")
    }
}

impl std::error::Error for GatewayError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionStop {
    Cancelled,
    Deadline,
    AttemptLimit,
    ListenerUnavailable,
}

/// Counts contain no peer addresses, paths, headers, tokens, or request data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServeReport {
    pub accepted_attempts: usize,
    pub completed_http_connections: usize,
    pub denied_non_loopback_peers: usize,
    pub denied_peer_limit: usize,
    pub stop: SessionStop,
}

/// A sequential HTTP adapter to the private coordinator. At most one TCP
/// connection is active, and every accepted socket is closed after one reply.
pub struct HttpGateway {
    listener: TcpListener,
    local_addr: SocketAddr,
    coordinator_directory: PathBuf,
    token: BearerToken,
    request_timeout: Duration,
    session_attempts: usize,
    peer_attempts: usize,
}

impl HttpGateway {
    pub fn bind(config: HttpGatewayConfig) -> Result<Self, GatewayError> {
        if !config.bind.ip().is_loopback() {
            return Err(GatewayError::BindUnavailable);
        }
        let listener = TcpListener::bind(config.bind).map_err(|_| GatewayError::BindUnavailable)?;
        listener
            .set_nonblocking(true)
            .map_err(|_| GatewayError::BindUnavailable)?;
        let local_addr = listener
            .local_addr()
            .map_err(|_| GatewayError::BindUnavailable)?;
        if !local_addr.ip().is_loopback() {
            return Err(GatewayError::BindUnavailable);
        }
        Ok(Self {
            listener,
            local_addr,
            coordinator_directory: config.coordinator_directory,
            token: config.token,
            request_timeout: config.request_timeout,
            session_attempts: config.session_attempts,
            peer_attempts: config.peer_attempts,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Serve one bounded session. Only `POST /v1/coordinator` with
    /// `Content-Type: application/json` and one `Content-Length` is accepted;
    /// transfer coding and pipelined bytes are rejected. Header blocks are
    /// capped at 16 KiB and native request bodies at `MAX_FRAME_BYTES`. Every
    /// response is a compact validated native envelope with `Connection:
    /// close`. The caller's context cancels the listener and bounds the whole
    /// session; each connection also has the configured absolute timeout.
    ///
    /// For a validated Submit, a lost HTTP acknowledgment is unconfirmed even
    /// if its error response/header could not be delivered. Callers must query
    /// status with that same request ID rather than resubmitting.
    pub fn serve(&self, context: &OperationContext<'_>) -> ServeReport {
        let mut report = ServeReport {
            accepted_attempts: 0,
            completed_http_connections: 0,
            denied_non_loopback_peers: 0,
            denied_peer_limit: 0,
            stop: SessionStop::Deadline,
        };
        let mut peer_attempts: HashMap<IpAddr, usize> = HashMap::new();

        loop {
            if let Some(stop) = session_stop(context) {
                report.stop = stop;
                return report;
            }
            if report.accepted_attempts >= self.session_attempts {
                report.stop = SessionStop::AttemptLimit;
                return report;
            }

            match self.listener.accept() {
                Ok((mut stream, peer)) => {
                    report.accepted_attempts += 1;
                    if !peer.ip().is_loopback() {
                        report.denied_non_loopback_peers += 1;
                        continue;
                    }
                    let attempts = peer_attempts.entry(peer.ip()).or_default();
                    if *attempts >= self.peer_attempts {
                        report.denied_peer_limit += 1;
                        continue;
                    }
                    *attempts += 1;
                    if stream.set_nonblocking(true).is_err() {
                        continue;
                    }
                    let cancellation = ParentCancellation(context);
                    let deadline = Instant::now()
                        .checked_add(self.request_timeout)
                        .unwrap_or(context.deadline())
                        .min(context.deadline());
                    let request_context = OperationContext::new(deadline, &cancellation);
                    handle_connection(
                        &mut stream,
                        peer,
                        &self.coordinator_directory,
                        &self.token,
                        &request_context,
                    );
                    report.completed_http_connections += 1;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    pause(context);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    // No sensitive operating-system text escapes.
                    report.stop = SessionStop::ListenerUnavailable;
                    return report;
                }
            }
        }
    }
}

struct ParentCancellation<'a, 'b>(&'a OperationContext<'b>);

impl Cancellation for ParentCancellation<'_, '_> {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

fn session_stop(context: &OperationContext<'_>) -> Option<SessionStop> {
    if context.is_cancelled() {
        Some(SessionStop::Cancelled)
    } else if Instant::now() >= context.deadline() {
        Some(SessionStop::Deadline)
    } else {
        None
    }
}

fn pause(context: &OperationContext<'_>) {
    let remaining = context.remaining();
    if !remaining.is_zero() && !context.is_cancelled() {
        thread::sleep(IO_POLL_INTERVAL.min(remaining));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HttpFailure {
    BadRequest,
    Unauthorized,
    MethodNotAllowed,
    NotFound,
    PayloadTooLarge,
    Internal,
}

struct ParsedHeaders {
    method: String,
    target: String,
    authorization: Option<String>,
    content_length: usize,
}

fn handle_connection(
    stream: &mut TcpStream,
    _peer: SocketAddr,
    coordinator_directory: &Path,
    token: &BearerToken,
    context: &OperationContext<'_>,
) {
    let result = (|| {
        let (header_bytes, mut body_prefix) = read_header(stream, context)?;
        active(context)?;
        let headers = parse_headers(&header_bytes)?;
        active(context)?;
        if headers.method != "POST" {
            return Err(HttpFailure::MethodNotAllowed);
        }
        if !headers.target.starts_with('/') || headers.target.starts_with("//") {
            return Err(HttpFailure::BadRequest);
        }
        if headers.target != ROUTE {
            return Err(HttpFailure::NotFound);
        }
        if !authorization_matches(headers.authorization.as_deref(), token) {
            return Err(HttpFailure::Unauthorized);
        }
        active(context)?;
        if body_prefix.len() > headers.content_length {
            return Err(HttpFailure::BadRequest);
        }
        read_body(stream, &mut body_prefix, headers.content_length, context)?;
        reject_pipelined_bytes(stream, context)?;
        active(context)?;
        let request: RequestEnvelope =
            serde_json::from_slice(&body_prefix).map_err(|_| HttpFailure::BadRequest)?;
        active(context)?;
        ipc::validate_request_envelope(&request).map_err(|_| HttpFailure::BadRequest)?;
        active(context)?;
        request
            .clone()
            .into_incoming_request()
            .map_err(|_| HttpFailure::BadRequest)?;
        active(context)?;
        Ok(request)
    })();

    let request = match result {
        Ok(request) => request,
        Err(failure) => {
            write_gateway_failure(stream, failure, context);
            return;
        }
    };

    let request_id = match &request.message {
        WireRequestMessage::Submit { request_id, .. }
        | WireRequestMessage::Cancel { request_id }
        | WireRequestMessage::Status { request_id }
        | WireRequestMessage::Results { request_id, .. } => request_id.clone(),
    };
    let is_submit = matches!(&request.message, WireRequestMessage::Submit { .. });
    let version = request.version;

    match client::exchange(coordinator_directory, request, context) {
        Ok(message) => {
            let response = response_envelope(version, message);
            write_http_response(stream, 200, &response, false, context);
        }
        Err(error) => {
            let status = match error {
                ClientError::TimedOut => 504,
                ClientError::Unavailable
                | ClientError::UnsafePath
                | ClientError::EndpointUnavailable => 503,
                _ => 502,
            };
            let code = match error {
                ClientError::InvalidRequest | ClientError::InvalidRequestId => {
                    ResponseErrorCode::InvalidRequest
                }
                ClientError::Unavailable
                | ClientError::TimedOut
                | ClientError::UnsafePath
                | ClientError::EndpointUnavailable => ResponseErrorCode::Unavailable,
                _ => ResponseErrorCode::Internal,
            };
            let response = response_envelope(
                version,
                ResponseMessage::Error {
                    request_id: Some(request_id),
                    code,
                },
            );
            write_http_response(stream, status, &response, is_submit, context);
        }
    }
}

fn read_header(
    stream: &mut TcpStream,
    context: &OperationContext<'_>,
) -> Result<(Vec<u8>, Vec<u8>), HttpFailure> {
    let mut bytes = Vec::with_capacity(MAX_HEADER_BYTES + HTTP_READ_CHUNK);
    loop {
        active(context)?;
        if let Some(end) = header_end(&bytes) {
            if end > MAX_HEADER_BYTES {
                return Err(HttpFailure::BadRequest);
            }
            let extra = bytes.split_off(end);
            validate_header_line_endings(&bytes)?;
            return Ok((bytes, extra));
        }
        if bytes.len() > MAX_HEADER_BYTES {
            return Err(HttpFailure::BadRequest);
        }
        validate_header_line_endings(&bytes)?;
        let mut chunk = [0_u8; HTTP_READ_CHUNK];
        match stream.read(&mut chunk) {
            Ok(0) => return Err(HttpFailure::BadRequest),
            Ok(count) => bytes.extend_from_slice(&chunk[..count]),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => pause(context),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(HttpFailure::BadRequest),
        }
    }
}

fn header_end(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| position + 4)
}

fn validate_header_line_endings(bytes: &[u8]) -> Result<(), HttpFailure> {
    for (index, byte) in bytes.iter().copied().enumerate() {
        match byte {
            b'\n' if index == 0 || bytes[index - 1] != b'\r' => {
                return Err(HttpFailure::BadRequest)
            }
            b'\r' if index + 1 < bytes.len() && bytes[index + 1] != b'\n' => {
                return Err(HttpFailure::BadRequest)
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_headers(bytes: &[u8]) -> Result<ParsedHeaders, HttpFailure> {
    if bytes.len() > MAX_HEADER_BYTES
        || !bytes.ends_with(b"\r\n\r\n")
        || validate_header_line_endings(bytes).is_err()
    {
        return Err(HttpFailure::BadRequest);
    }
    for byte in bytes.iter().copied() {
        if byte == b'\r' || byte == b'\n' {
            continue;
        }
        if !(0x20..=0x7e).contains(&byte) {
            return Err(HttpFailure::BadRequest);
        }
    }

    let text =
        std::str::from_utf8(&bytes[..bytes.len() - 4]).map_err(|_| HttpFailure::BadRequest)?;
    let mut lines = text.split("\r\n");
    let request_line = lines.next().ok_or(HttpFailure::BadRequest)?;
    let mut request_parts = request_line.split(' ');
    let method = request_parts.next().ok_or(HttpFailure::BadRequest)?;
    let target = request_parts.next().ok_or(HttpFailure::BadRequest)?;
    let version = request_parts.next().ok_or(HttpFailure::BadRequest)?;
    if request_parts.next().is_some() || method.is_empty() || target.is_empty() {
        return Err(HttpFailure::BadRequest);
    }
    if version != "HTTP/1.1" {
        return Err(HttpFailure::BadRequest);
    }

    let mut seen = std::collections::HashSet::new();
    let mut host_seen = false;
    let mut content_type_seen = false;
    let mut authorization = None;
    let mut content_length = None;
    let mut count = 0;
    for line in lines {
        if line.is_empty() {
            return Err(HttpFailure::BadRequest);
        }
        count += 1;
        if count > MAX_HEADER_FIELDS || line.starts_with(' ') || line.starts_with('\t') {
            return Err(HttpFailure::BadRequest);
        }
        let (name, raw_value) = line.split_once(':').ok_or(HttpFailure::BadRequest)?;
        if name.is_empty() || !name.bytes().all(is_header_name_byte) {
            return Err(HttpFailure::BadRequest);
        }
        let name_lower = name.to_ascii_lowercase();
        if !seen.insert(name_lower.clone()) {
            return Err(HttpFailure::BadRequest);
        }
        let value = raw_value.trim_matches(' ');
        if value.is_empty() && name_lower != "authorization" {
            return Err(HttpFailure::BadRequest);
        }
        match name_lower.as_str() {
            "host" => {
                if value.is_empty()
                    || !value.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric()
                            || matches!(byte, b'.' | b'-' | b':' | b'[' | b']')
                    })
                {
                    return Err(HttpFailure::BadRequest);
                }
                host_seen = true;
            }
            "authorization" => authorization = Some(value.to_owned()),
            "content-length" => {
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(HttpFailure::BadRequest);
                }
                let length = value
                    .parse::<usize>()
                    .map_err(|_| HttpFailure::BadRequest)?;
                if length > MAX_FRAME_BYTES {
                    return Err(HttpFailure::PayloadTooLarge);
                }
                content_length = Some(length);
            }
            "content-type" => {
                if !value.eq_ignore_ascii_case("application/json") {
                    return Err(HttpFailure::BadRequest);
                }
                content_type_seen = true;
            }
            "transfer-encoding" | "expect" | "upgrade" => return Err(HttpFailure::BadRequest),
            _ => {}
        }
    }
    if !host_seen || !content_type_seen {
        return Err(HttpFailure::BadRequest);
    }
    Ok(ParsedHeaders {
        method: method.to_owned(),
        target: target.to_owned(),
        authorization,
        content_length: content_length.ok_or(HttpFailure::BadRequest)?,
    })
}

fn is_header_name_byte(byte: u8) -> bool {
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

fn authorization_matches(value: Option<&str>, token: &BearerToken) -> bool {
    let Some(value) = value else {
        return false;
    };
    let Some((scheme, presented)) = value.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer") || presented.is_empty() {
        return false;
    }
    constant_time_token_eq(presented.as_bytes(), token.0.as_bytes())
}

fn constant_time_token_eq(presented: &[u8], expected: &[u8]) -> bool {
    let mut difference = presented.len() ^ expected.len();
    for index in 0..MAX_TOKEN_BYTES {
        let left = presented.get(index).copied().unwrap_or(0);
        let right = expected.get(index).copied().unwrap_or(0);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

fn read_body(
    stream: &mut TcpStream,
    body: &mut Vec<u8>,
    content_length: usize,
    context: &OperationContext<'_>,
) -> Result<(), HttpFailure> {
    active(context)?;
    let additional = content_length.saturating_sub(body.len());
    body.try_reserve_exact(additional)
        .map_err(|_| HttpFailure::Internal)?;
    active(context)?;
    while body.len() < content_length {
        active(context)?;
        let mut chunk = [0_u8; HTTP_READ_CHUNK];
        let limit = (content_length - body.len()).min(chunk.len());
        match stream.read(&mut chunk[..limit]) {
            Ok(0) => return Err(HttpFailure::BadRequest),
            Ok(count) => body.extend_from_slice(&chunk[..count]),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => pause(context),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(HttpFailure::BadRequest),
        }
    }
    Ok(())
}

fn reject_pipelined_bytes(
    stream: &mut TcpStream,
    context: &OperationContext<'_>,
) -> Result<(), HttpFailure> {
    loop {
        active(context)?;
        let mut extra = [0_u8; 1];
        match stream.read(&mut extra) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(HttpFailure::BadRequest),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(HttpFailure::BadRequest),
        }
    }
}

fn active(context: &OperationContext<'_>) -> Result<(), HttpFailure> {
    context
        .check_active("HTTP coordinator gateway")
        .map_err(|_| HttpFailure::Internal)
}

fn response_envelope(version: u16, message: ResponseMessage) -> ResponseEnvelope {
    if version == RESULTS_PROTOCOL_VERSION {
        ResponseEnvelope::v2_results(message)
    } else {
        ResponseEnvelope::v1(message)
    }
}

fn write_gateway_failure(
    stream: &mut TcpStream,
    failure: HttpFailure,
    context: &OperationContext<'_>,
) {
    let (status, code) = match failure {
        HttpFailure::BadRequest => (400, ResponseErrorCode::InvalidRequest),
        HttpFailure::Unauthorized => (401, ResponseErrorCode::InvalidRequest),
        HttpFailure::MethodNotAllowed => (405, ResponseErrorCode::InvalidRequest),
        HttpFailure::NotFound => (404, ResponseErrorCode::InvalidRequest),
        HttpFailure::PayloadTooLarge => (413, ResponseErrorCode::InvalidFrame),
        HttpFailure::Internal => (500, ResponseErrorCode::Internal),
    };
    let response = ResponseEnvelope::v1(ResponseMessage::Error {
        request_id: None,
        code,
    });
    write_http_response(stream, status, &response, false, context);
}

fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    response: &ResponseEnvelope,
    uncertain_submit: bool,
    context: &OperationContext<'_>,
) {
    if context
        .check_active("HTTP coordinator gateway response")
        .is_err()
    {
        return;
    }
    let body = match encode_response_body(response) {
        Some(body) => body,
        None => match encode_response_body(&ResponseEnvelope::v1(ResponseMessage::Error {
            request_id: None,
            code: ResponseErrorCode::Internal,
        })) {
            Some(body) => body,
            None => return,
        },
    };
    if context
        .check_active("HTTP coordinator gateway response")
        .is_err()
    {
        return;
    }
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    };
    let mut headers = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n",
        body.len()
    );
    if uncertain_submit {
        headers.push_str("X-L3MS-Submit-Outcome: ");
        headers.push_str(UNCERTAIN_SUBMIT_HEADER);
        headers.push_str("\r\n");
    }
    headers.push_str("\r\n");
    if write_bounded(stream, headers.as_bytes(), context).is_ok() {
        let _ = write_bounded(stream, &body, context);
    }
}

fn encode_response_body(response: &ResponseEnvelope) -> Option<Vec<u8>> {
    let mut framed = Vec::new();
    ipc::write_response(&mut framed, response).ok()?;
    if framed.len() < 4 {
        return None;
    }
    Some(framed.split_off(4))
}

fn write_bounded(
    stream: &mut TcpStream,
    bytes: &[u8],
    context: &OperationContext<'_>,
) -> io::Result<()> {
    let mut written = 0;
    while written < bytes.len() {
        context
            .check_active("HTTP coordinator gateway write")
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "deadline"))?;
        match stream.write(&bytes[written..]) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(count) => written += count,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => pause(context),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::net::{Shutdown, TcpStream};
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::control::catalog::Lane;
    use crate::control::ipc::{
        AcceptedOperation, ClientRequest, IncomingRequest, RequestStatus, ResponseMessage,
        WireChatRole, WireChatTurn, WireInferenceRequest,
    };
    use crate::control::result_buffer::{
        ResultBuffer, ResultEvent, ResultTerminal, MAX_PAGE_BYTES,
    };
    use crate::control::runtime::NeverCancel;

    const TEST_TOKEN: &str = "test-bearer-token-with-more-than-32-bytes";
    type MalformedHttpCase<'a> = (&'a str, &'a [(&'a str, &'a str)], Vec<u8>, u16);

    struct MockCoordinator {
        _directory: tempfile::TempDir,
        socket_directory: PathBuf,
        listener: UnixListener,
        accepts: std::sync::Arc<AtomicUsize>,
    }

    impl MockCoordinator {
        fn bind() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let socket_directory = directory.path().join("socket");
            std::fs::create_dir(&socket_directory).unwrap();
            std::fs::set_permissions(&socket_directory, std::fs::Permissions::from_mode(0o700))
                .unwrap();
            let listener = UnixListener::bind(socket_directory.join("coordinator.sock")).unwrap();
            std::fs::set_permissions(
                socket_directory.join("coordinator.sock"),
                std::fs::Permissions::from_mode(0o600),
            )
            .unwrap();
            listener.set_nonblocking(true).unwrap();
            Self {
                _directory: directory,
                socket_directory,
                listener,
                accepts: std::sync::Arc::new(AtomicUsize::new(0)),
            }
        }

        fn respond_once(
            &self,
            handler: impl FnOnce(IncomingRequest) -> Option<ResponseEnvelope> + Send + 'static,
        ) -> thread::JoinHandle<()> {
            let listener = self.listener.try_clone().unwrap();
            let accepts = self.accepts.clone();
            thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(3);
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(connection) => break connection,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            assert!(
                                Instant::now() < deadline,
                                "coordinator fixture accept timed out"
                            );
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        Err(error) => panic!("fixture accept failed: {error}"),
                    }
                };
                accepts.fetch_add(1, Ordering::SeqCst);
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let incoming = ipc::read_incoming_request(&mut stream).unwrap();
                if let Some(response) = handler(incoming) {
                    ipc::write_response(&mut stream, &response).unwrap();
                }
            })
        }

        fn accept_count(&self) -> usize {
            self.accepts.load(Ordering::SeqCst)
        }
    }

    fn submit(request_id: &str) -> RequestEnvelope {
        RequestEnvelope::v1(WireRequestMessage::Submit {
            request_id: request_id.into(),
            timeout_ms: 5_000,
            request: WireInferenceRequest::Language {
                profile_id: "fixture-model".into(),
                turns: vec![WireChatTurn {
                    role: WireChatRole::User,
                    content: "fixture payload".into(),
                }],
                max_tokens: 32,
            },
        })
    }

    fn gateway(
        coordinator_directory: &Path,
        timeout: Duration,
        session_attempts: usize,
        peer_attempts: usize,
    ) -> HttpGateway {
        let config = HttpGatewayConfig::new(
            SocketAddr::from(([127, 0, 0, 1], 0)),
            coordinator_directory,
            BearerToken::new(TEST_TOKEN).unwrap(),
        )
        .unwrap()
        .with_limits(timeout, session_attempts, peer_attempts)
        .unwrap();
        HttpGateway::bind(config).unwrap()
    }

    fn run_gateway(gateway: HttpGateway) -> thread::JoinHandle<ServeReport> {
        thread::spawn(move || {
            let cancellation = NeverCancel;
            let context =
                OperationContext::new(Instant::now() + Duration::from_secs(3), &cancellation);
            gateway.serve(&context)
        })
    }

    fn http_request(target: &str, body: &[u8], extra_headers: &[(&str, &str)]) -> Vec<u8> {
        let mut request = format!(
            "POST {target} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TEST_TOKEN}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in extra_headers {
            request.push_str(name);
            request.push_str(": ");
            request.push_str(value);
            request.push_str("\r\n");
        }
        request.push_str("\r\n");
        let mut bytes = request.into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }

    fn send_http(addr: SocketAddr, bytes: &[u8]) -> Vec<u8> {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(bytes).unwrap();
        stream.shutdown(Shutdown::Write).unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        response
    }

    fn response_parts(bytes: &[u8]) -> (u16, HashMap<String, String>, Vec<u8>) {
        let split = bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap();
        let head = std::str::from_utf8(&bytes[..split]).unwrap();
        let mut lines = head.split("\r\n");
        let status = lines
            .next()
            .unwrap()
            .split(' ')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .map(|line| {
                let (name, value) = line.split_once(':').unwrap();
                (name.to_ascii_lowercase(), value.trim().to_owned())
            })
            .collect();
        (status, headers, bytes[split + 4..].to_vec())
    }

    fn decode_envelope(body: &[u8]) -> ResponseEnvelope {
        serde_json::from_slice(body).unwrap()
    }

    #[test]
    fn token_is_explicit_bounded_and_redacted() {
        assert!(matches!(
            BearerToken::new("short"),
            Err(GatewayConfigError::InvalidToken)
        ));
        let token = BearerToken::new(TEST_TOKEN).unwrap();
        assert!(!format!("{token:?}").contains(TEST_TOKEN));
        assert_eq!(
            HttpGatewayConfig::new(
                SocketAddr::from(([0, 0, 0, 0], 0)),
                "/tmp/private-coordinator",
                BearerToken::new(TEST_TOKEN).unwrap(),
            )
            .err(),
            Some(GatewayConfigError::NonLoopbackBind)
        );
    }

    #[test]
    fn parses_one_minimal_valid_http11_header_block() {
        let bytes = b"POST /v1/coordinator HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-bearer-token-with-more-than-32-bytes\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n";
        let parsed = parse_headers(bytes).unwrap();
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.target, ROUTE);
        assert_eq!(
            parsed.authorization.as_deref(),
            Some("Bearer test-bearer-token-with-more-than-32-bytes")
        );
        assert_eq!(parsed.content_length, 2);
    }

    #[test]
    fn submit_is_one_correlated_v1_exchange() {
        let coordinator = MockCoordinator::bind();
        let server = coordinator.respond_once(|incoming| {
            let IncomingRequest::Control(ClientRequest::Submit { request_id, .. }) = incoming
            else {
                panic!("expected native Submit")
            };
            assert_eq!(request_id, "caller:http-submit-1");
            Some(ResponseEnvelope::v1(ResponseMessage::Accepted {
                request_id,
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued,
            }))
        });
        let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let body = serde_json::to_vec(&submit("caller:http-submit-1")).unwrap();
        let response = send_http(address, &http_request(ROUTE, &body, &[]));
        let (status, headers, body) = response_parts(&response);
        assert_eq!(status, 200);
        assert_eq!(headers.get("connection").map(String::as_str), Some("close"));
        let envelope = decode_envelope(&body);
        assert_eq!(envelope.version, ipc::PROTOCOL_VERSION);
        assert!(matches!(
            envelope.message,
            ResponseMessage::Accepted {
                request_id,
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued,
            } if request_id == "caller:http-submit-1"
        ));
        assert_eq!(serving.join().unwrap().completed_http_connections, 1);
        server.join().unwrap();
        assert_eq!(coordinator.accept_count(), 1);
    }

    #[test]
    fn status_and_cancel_keep_the_v1_request_id_and_operation() {
        for cancel in [false, true] {
            let coordinator = MockCoordinator::bind();
            let server = coordinator.respond_once(move |incoming| {
                if cancel {
                    let IncomingRequest::Control(ClientRequest::Cancel { request_id }) = incoming
                    else {
                        panic!("expected native Cancel")
                    };
                    assert_eq!(request_id, "caller:http-control-2");
                    Some(ResponseEnvelope::v1(ResponseMessage::Accepted {
                        request_id,
                        operation: AcceptedOperation::Cancel,
                        state: RequestStatus::Running,
                    }))
                } else {
                    let IncomingRequest::Control(ClientRequest::Status { request_id }) = incoming
                    else {
                        panic!("expected native Status")
                    };
                    assert_eq!(request_id, "caller:http-control-2");
                    Some(ResponseEnvelope::v1(ResponseMessage::Status {
                        request_id,
                        state: RequestStatus::Succeeded,
                    }))
                }
            });
            let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
            let address = gateway.local_addr();
            let serving = run_gateway(gateway);
            let request = if cancel {
                RequestEnvelope::v1(WireRequestMessage::Cancel {
                    request_id: "caller:http-control-2".into(),
                })
            } else {
                RequestEnvelope::v1(WireRequestMessage::Status {
                    request_id: "caller:http-control-2".into(),
                })
            };
            let body = serde_json::to_vec(&request).unwrap();
            let response = send_http(address, &http_request(ROUTE, &body, &[]));
            let (status, _, body) = response_parts(&response);
            assert_eq!(status, 200);
            let envelope = decode_envelope(&body);
            match (cancel, envelope.message) {
                (
                    true,
                    ResponseMessage::Accepted {
                        request_id,
                        operation: AcceptedOperation::Cancel,
                        state: RequestStatus::Running,
                    },
                ) => assert_eq!(request_id, "caller:http-control-2"),
                (
                    false,
                    ResponseMessage::Status {
                        request_id,
                        state: RequestStatus::Succeeded,
                    },
                ) => assert_eq!(request_id, "caller:http-control-2"),
                _ => panic!("unexpected native response"),
            }
            assert_eq!(serving.join().unwrap().completed_http_connections, 1);
            server.join().unwrap();
            assert_eq!(coordinator.accept_count(), 1);
        }
    }

    #[test]
    fn v2_results_round_trip_is_correlated_and_validated() {
        let coordinator = MockCoordinator::bind();
        let server = coordinator.respond_once(|incoming| {
            let IncomingRequest::Results(query) = incoming else {
                panic!("expected native v2 Results")
            };
            assert_eq!(query.request_id, "caller:http-result-3");
            let mut result = ResultBuffer::new();
            result.begin(&query.request_id, Lane::Language).unwrap();
            result
                .append(
                    &query.request_id,
                    ResultEvent::ChatContent("fixture result".into()),
                )
                .unwrap();
            result
                .terminal(&query.request_id, ResultTerminal::Succeeded)
                .unwrap();
            let page = result
                .page(&query.request_id, query.after_cursor, MAX_PAGE_BYTES)
                .unwrap();
            Some(ResponseEnvelope::v2_results(ResponseMessage::Results {
                request_id: query.request_id,
                after_cursor: query.after_cursor,
                byte_budget: query.byte_budget,
                state: RequestStatus::Succeeded,
                page,
            }))
        });
        let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let request = RequestEnvelope::v2_results("caller:http-result-3", 0, MAX_PAGE_BYTES as u32);
        let body = serde_json::to_vec(&request).unwrap();
        let response = send_http(address, &http_request(ROUTE, &body, &[]));
        let (status, _, body) = response_parts(&response);
        assert_eq!(status, 200);
        let envelope = decode_envelope(&body);
        assert_eq!(envelope.version, RESULTS_PROTOCOL_VERSION);
        assert!(matches!(
            envelope.message,
            ResponseMessage::Results {
                request_id,
                after_cursor: 0,
                byte_budget,
                ..
            } if request_id == "caller:http-result-3" && byte_budget == MAX_PAGE_BYTES as u32
        ));
        assert_eq!(serving.join().unwrap().completed_http_connections, 1);
        server.join().unwrap();
        assert_eq!(coordinator.accept_count(), 1);
    }

    #[test]
    fn authentication_and_duplicate_framing_fail_before_ipc() {
        let cases: Vec<(Vec<u8>, u16)> = {
            let body = serde_json::to_vec(&submit("caller:http-auth-4")).unwrap();
            let wrong_token = String::from_utf8(http_request(ROUTE, &body, &[]))
                .unwrap()
                .replace(
                    &format!("Authorization: Bearer {TEST_TOKEN}"),
                    "Authorization: Bearer wrong-token-with-more-than-32-bytes",
                )
                .into_bytes();
            vec![
                (wrong_token, 401),
                (http_request(ROUTE, &body, &[("Content-Length", "2")]), 400),
                (
                    http_request(
                        ROUTE,
                        &body,
                        &[(
                            "Authorization",
                            "Bearer test-bearer-token-with-more-than-32-bytes",
                        )],
                    ),
                    400,
                ),
            ]
        };
        for (request, expected_status) in cases {
            let coordinator = MockCoordinator::bind();
            let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
            let address = gateway.local_addr();
            let serving = run_gateway(gateway);
            let response = send_http(address, &request);
            let (status, _, _) = response_parts(&response);
            assert_eq!(status, expected_status);
            assert_eq!(serving.join().unwrap().accepted_attempts, 1);
            assert_eq!(coordinator.accept_count(), 0);
        }
    }

    #[test]
    fn transfer_encoding_unknown_fields_and_non_origin_targets_are_rejected() {
        let cases: &[MalformedHttpCase<'_>] = &[
            (ROUTE, &[("Transfer-Encoding", "chunked")], Vec::new(), 400),
            (
                ROUTE,
                &[],
                br#"{"version":1,"message":{"type":"status","request_id":"caller:x","extra":true}}"#.to_vec(),
                400,
            ),
            (
                "http://localhost/v1/coordinator",
                &[],
                serde_json::to_vec(&submit("caller:http-route-5")).unwrap(),
                400,
            ),
        ];
        for (target, extra, body, expected_status) in cases {
            let coordinator = MockCoordinator::bind();
            let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
            let address = gateway.local_addr();
            let serving = run_gateway(gateway);
            let response = send_http(address, &http_request(target, body, extra));
            let (status, _, _) = response_parts(&response);
            assert_eq!(status, *expected_status);
            assert_eq!(serving.join().unwrap().accepted_attempts, 1);
            assert_eq!(coordinator.accept_count(), 0);
        }
    }

    #[test]
    fn obs_fold_and_control_characters_are_rejected_before_ipc() {
        let body = serde_json::to_vec(&RequestEnvelope::v1(WireRequestMessage::Status {
            request_id: "caller:header-characters-5".into(),
        }))
        .unwrap();
        let malformed = [
            http_request(ROUTE, &body, &[("X-Test", "safe\r\n folded: bad")]),
            http_request(ROUTE, &body, &[("X-Test", "safe\tbad")]),
        ];
        for request in malformed {
            let coordinator = MockCoordinator::bind();
            let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
            let address = gateway.local_addr();
            let serving = run_gateway(gateway);
            let response = send_http(address, &request);
            assert_eq!(response_parts(&response).0, 400);
            assert_eq!(serving.join().unwrap().accepted_attempts, 1);
            assert_eq!(coordinator.accept_count(), 0);
        }
    }

    #[test]
    fn pipelined_bytes_after_the_native_body_are_rejected_before_ipc() {
        let coordinator = MockCoordinator::bind();
        let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let body = serde_json::to_vec(&RequestEnvelope::v1(WireRequestMessage::Status {
            request_id: "caller:pipeline-5".into(),
        }))
        .unwrap();
        let mut bytes = http_request(ROUTE, &body, &[]);
        bytes.extend_from_slice(b"GET /v1/coordinator HTTP/1.1\r\nHost: localhost\r\n\r\n");
        let response = send_http(address, &bytes);
        let (status, _, response_body) = response_parts(&response);
        assert_eq!(status, 400);
        assert!(!String::from_utf8_lossy(&response_body).contains("pipeline-5"));
        assert_eq!(serving.join().unwrap().accepted_attempts, 1);
        assert_eq!(coordinator.accept_count(), 0);
    }

    #[test]
    fn oversized_body_is_rejected_without_reading_or_ipc() {
        let coordinator = MockCoordinator::bind();
        let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let headers = format!(
            "POST {ROUTE} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TEST_TOKEN}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            MAX_FRAME_BYTES + 1
        );
        let response = send_http(address, headers.as_bytes());
        let (status, _, _) = response_parts(&response);
        assert_eq!(status, 413);
        assert_eq!(serving.join().unwrap().accepted_attempts, 1);
        assert_eq!(coordinator.accept_count(), 0);
    }

    #[test]
    fn slow_header_is_closed_by_the_absolute_connection_deadline() {
        let coordinator = MockCoordinator::bind();
        let gateway = gateway(
            &coordinator.socket_directory,
            Duration::from_millis(100),
            1,
            1,
        );
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let started = Instant::now();
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .write_all(b"POST /v1/coordinator HTTP/1.1\r\nHost: local\r\n")
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        assert!(response.is_empty());
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(serving.join().unwrap().accepted_attempts, 1);
        assert_eq!(coordinator.accept_count(), 0);
    }

    #[test]
    fn lost_submit_reply_is_explicitly_unconfirmed_and_never_retried() {
        let coordinator = MockCoordinator::bind();
        let server = coordinator.respond_once(|incoming| {
            let IncomingRequest::Control(ClientRequest::Submit { request_id, .. }) = incoming
            else {
                panic!("expected exactly one native Submit")
            };
            assert_eq!(request_id, "caller:uncertain-6");
            None // Drop the private connection without an acknowledgement.
        });
        let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let body = serde_json::to_vec(&submit("caller:uncertain-6")).unwrap();
        let response = send_http(address, &http_request(ROUTE, &body, &[]));
        let (status, headers, body) = response_parts(&response);
        assert!(matches!(status, 502..=504));
        assert_eq!(
            headers.get("x-l3ms-submit-outcome").map(String::as_str),
            Some(UNCERTAIN_SUBMIT_HEADER)
        );
        let envelope = decode_envelope(&body);
        assert!(matches!(
            envelope.message,
            ResponseMessage::Error {
                request_id: Some(request_id),
                ..
            } if request_id == "caller:uncertain-6"
        ));
        assert_eq!(serving.join().unwrap().completed_http_connections, 1);
        server.join().unwrap();
        assert_eq!(coordinator.accept_count(), 1);
    }

    #[test]
    fn one_session_connection_limit_is_enforced() {
        let coordinator = MockCoordinator::bind();
        let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 1, 1);
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let body = serde_json::to_vec(&submit("caller:session-limit-7")).unwrap();
        let request = String::from_utf8(http_request(ROUTE, &body, &[]))
            .unwrap()
            .replace(
                &format!("Authorization: Bearer {TEST_TOKEN}"),
                "Authorization: Bearer wrong-token-with-more-than-32-bytes",
            )
            .into_bytes();
        let response = send_http(address, &request);
        assert_eq!(response_parts(&response).0, 401);
        let report = serving.join().unwrap();
        assert_eq!(report.accepted_attempts, 1);
        assert_eq!(report.completed_http_connections, 1);
        assert_eq!(report.stop, SessionStop::AttemptLimit);
        assert_eq!(coordinator.accept_count(), 0);
    }

    #[test]
    fn loopback_peer_attempt_limit_drops_later_connections() {
        let coordinator = MockCoordinator::bind();
        let gateway = gateway(&coordinator.socket_directory, Duration::from_secs(1), 2, 1);
        let address = gateway.local_addr();
        let serving = run_gateway(gateway);
        let body = serde_json::to_vec(&submit("caller:peer-limit-8")).unwrap();
        let first = String::from_utf8(http_request(ROUTE, &body, &[]))
            .unwrap()
            .replace(
                &format!("Authorization: Bearer {TEST_TOKEN}"),
                "Authorization: Bearer wrong-token-with-more-than-32-bytes",
            )
            .into_bytes();
        assert_eq!(response_parts(&send_http(address, &first)).0, 401);

        let mut second = TcpStream::connect(address).unwrap();
        second
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        second.write_all(&http_request(ROUTE, &body, &[])).unwrap();
        second.shutdown(Shutdown::Write).unwrap();
        let mut second_reply = Vec::new();
        match second.read_to_end(&mut second_reply) {
            Ok(_) => assert!(second_reply.is_empty()),
            Err(error) => {
                assert_eq!(error.kind(), io::ErrorKind::ConnectionReset);
                assert!(second_reply.is_empty());
            }
        }
        let report = serving.join().unwrap();
        assert_eq!(report.accepted_attempts, 2);
        assert_eq!(report.completed_http_connections, 1);
        assert_eq!(report.denied_peer_limit, 1);
        assert_eq!(report.stop, SessionStop::AttemptLimit);
        assert_eq!(coordinator.accept_count(), 0);
    }
}
