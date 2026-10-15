//! One-shot client for the private coordinator socket.
//!
//! The caller chooses the socket directory and supplies the operation
//! deadline. This module performs exactly one request/reply exchange, validates
//! request correlation and operation-specific replies, and exposes only fixed
//! errors. It never retries a request or falls back to starting a runtime.

use std::error::Error;
use std::fmt;
use std::path::Path;

use super::{
    ipc::{
        self, AcceptedOperation, ArtifactVersion, RequestEnvelope, RequestStatus, ResponseMessage,
        WireRequestMessage,
    },
    local_ipc::{LocalConnection, LocalIpcError},
    runtime::OperationContext,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClientError {
    InvalidRequestId,
    InvalidRequest,
    UnsafePath,
    Unavailable,
    TimedOut,
    InvalidResponse,
    PeerUnauthorized,
    ExchangeFailed,
}

impl fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequestId => "invalid coordinator request ID",
            Self::InvalidRequest => "coordinator request is invalid",
            Self::UnsafePath => "coordinator directory or socket is unsafe",
            Self::Unavailable => "coordinator is unavailable",
            Self::TimedOut => "coordinator operation timed out",
            Self::InvalidResponse => "coordinator response is invalid",
            Self::PeerUnauthorized => "coordinator peer is not authorized",
            Self::ExchangeFailed => "coordinator exchange failed",
        })
    }
}

impl Error for ClientError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeReadError {
    Exchange(ClientError),
    Assembly(super::result_consumer::ConsumeError),
    Rejected,
    PageLimit,
}

impl fmt::Display for NativeReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "native coordinator results: {self:?}")
    }
}
impl Error for NativeReadError {}

/// One validated, read-only artifact response. The chunk bytes are capped at
/// 64 KiB and omitted from Debug output; artifact paths and version tokens are
/// redacted as well.
pub struct ArtifactChunk {
    pub request_id: String,
    pub artifact: super::result_buffer::MediaArtifact,
    pub availability: super::result_buffer::Availability,
    pub offset: u64,
    pub total_bytes: u64,
    pub next_offset: u64,
    pub end_of_file: bool,
    pub artifact_version: Option<ArtifactVersion>,
    pub bytes: Vec<u8>,
}

impl fmt::Debug for ArtifactChunk {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactChunk")
            .field("request_id", &"<redacted>")
            .field("artifact", &self.artifact)
            .field("availability", &self.availability)
            .field("offset", &self.offset)
            .field("total_bytes", &self.total_bytes)
            .field("next_offset", &self.next_offset)
            .field("end_of_file", &self.end_of_file)
            .field("artifact_version", &"<redacted>")
            .field("bytes", &"<redacted>")
            .finish()
    }
}

/// Fetch exactly one bounded chunk from a previously published media artifact.
/// The reference is relative and bound to `request_id`; this method performs
/// one v3 exchange only, never submits inference or retries a failed read.
pub fn artifact_chunk(
    socket_directory: &Path,
    request_id: &str,
    artifact: &super::result_buffer::MediaArtifact,
    offset: u64,
    byte_budget: u32,
    artifact_version: Option<&ArtifactVersion>,
    context: &OperationContext<'_>,
) -> Result<ArtifactChunk, ClientError> {
    let request = RequestEnvelope::v3_artifact(
        request_id,
        artifact.clone(),
        offset,
        byte_budget,
        artifact_version.cloned(),
    );
    let response = exchange(socket_directory, request, context)?;
    let ResponseMessage::Artifact {
        request_id,
        artifact,
        availability,
        offset,
        total_bytes,
        next_offset,
        end_of_file,
        artifact_version,
        bytes,
    } = response
    else {
        return Err(ClientError::InvalidResponse);
    };
    Ok(ArtifactChunk {
        request_id,
        artifact,
        availability,
        offset,
        total_bytes,
        next_offset,
        end_of_file,
        artifact_version,
        bytes,
    })
}

/// Read and reassemble currently available output from cursor zero. Each page
/// is a fresh read-only query, never an inference retry. The same absolute
/// context bounds every exchange and all pages. Stops when caught up, even if
/// the job is still running; callers explicitly poll again if desired. A
/// hostile server cannot force more than 128 exchanges in this invocation.
/// Unknown/expired, malformed groups, or a failure on a later page returns a
/// fixed error, never a misleading partial success or a new Submit.
pub fn read_native_results(
    socket_directory: &Path,
    request_id: &str,
    lane: super::catalog::Lane,
    context: &OperationContext<'_>,
) -> Result<super::result_consumer::ResultConsumer, NativeReadError> {
    let mut consumer = super::result_consumer::ResultConsumer::new(request_id, lane)
        .map_err(NativeReadError::Assembly)?;
    read_native_results_into(socket_directory, &mut consumer, context)?;
    Ok(consumer)
}

/// Resume an explicit read-only observation from the consumer's last verified
/// cursor. The whole poll is transactional: a failed later page leaves the
/// caller's prior output/cursor unchanged. Native output remains bounded by the
/// consumer, and at most 128 fresh read-only exchanges share one context. This
/// does not submit, retry inference, automatically poll or reclaim an endpoint.
pub fn read_native_results_into(
    socket_directory: &Path,
    consumer: &mut super::result_consumer::ResultConsumer,
    context: &OperationContext<'_>,
) -> Result<(), NativeReadError> {
    let mut staged = consumer.clone();
    for _ in 0..128 {
        let request = RequestEnvelope::v2_results(
            staged.request_id(),
            staged.cursor(),
            super::result_buffer::MAX_PAGE_BYTES as u32,
        );
        let response =
            exchange(socket_directory, request, context).map_err(NativeReadError::Exchange)?;
        let ResponseMessage::Results { page, .. } = response else {
            return Err(NativeReadError::Rejected);
        };
        staged.consume(&page).map_err(NativeReadError::Assembly)?;
        if staged.cursor() == staged.latest_cursor() {
            *consumer = staged;
            return Ok(());
        }
    }
    Err(NativeReadError::PageLimit)
}

#[derive(Clone, Copy)]
enum Phase {
    Connect,
    Write,
    Read,
}

/// Sends one v1 control request, v2 results query, or v3 artifact query over the caller-selected
/// private socket directory.
///
/// `context` bounds the entire connect/write/read sequence. Request IDs are
/// checked before connecting; the response must correlate to the request and
/// use its versioned operation-specific shape. The function does not retry
/// after a write or start any inference/runtime work itself.
pub fn exchange(
    socket_directory: &Path,
    request: RequestEnvelope,
    context: &OperationContext<'_>,
) -> Result<ResponseMessage, ClientError> {
    let request_id = request_id(&request.message);
    ipc::validate_request_id(request_id).map_err(|_| ClientError::InvalidRequestId)?;
    ipc::validate_request_envelope(&request).map_err(|error| match error {
        ipc::ProtocolError::InvalidRequestId => ClientError::InvalidRequestId,
        _ => ClientError::InvalidRequest,
    })?;

    let mut connection = LocalConnection::connect(socket_directory, context)
        .map_err(|error| fixed_transport_error(error, Phase::Connect))?;
    connection
        .write_request(&request, context)
        .map_err(|error| fixed_transport_error(error, Phase::Write))?;
    let response_envelope = connection
        .read_response(context)
        .map_err(|error| fixed_transport_error(error, Phase::Read))?;
    if response_envelope.version != request.version {
        return Err(ClientError::InvalidResponse);
    }

    validate_response(&request.message, request_id, response_envelope.message)
}

fn request_id(message: &WireRequestMessage) -> &str {
    match message {
        WireRequestMessage::Submit { request_id, .. }
        | WireRequestMessage::Cancel { request_id }
        | WireRequestMessage::Status { request_id }
        | WireRequestMessage::Results { request_id, .. }
        | WireRequestMessage::Artifact { request_id, .. } => request_id,
    }
}

fn validate_response(
    request: &WireRequestMessage,
    request_id: &str,
    response: ResponseMessage,
) -> Result<ResponseMessage, ClientError> {
    let response_id = match &response {
        ResponseMessage::Accepted { request_id, .. }
        | ResponseMessage::Status { request_id, .. } => Some(request_id.as_str()),
        ResponseMessage::Results { request_id, .. } => Some(request_id.as_str()),
        ResponseMessage::Artifact { request_id, .. } => Some(request_id.as_str()),
        ResponseMessage::Error { request_id, .. } => request_id.as_deref(),
    };
    if response_id.is_some_and(|response_id| response_id != request_id) {
        return Err(ClientError::InvalidResponse);
    }
    if matches!(request, WireRequestMessage::Results { .. })
        && matches!(
            &response,
            ResponseMessage::Error {
                request_id: None,
                ..
            }
        )
    {
        return Err(ClientError::InvalidResponse);
    }
    if matches!(request, WireRequestMessage::Artifact { .. })
        && matches!(
            &response,
            ResponseMessage::Error {
                request_id: None,
                ..
            }
        )
    {
        return Err(ClientError::InvalidResponse);
    }

    let matches_request = match (request, &response) {
        (_, ResponseMessage::Error { .. }) => true,
        (
            WireRequestMessage::Submit { .. },
            ResponseMessage::Accepted {
                operation: AcceptedOperation::Submit,
                state: RequestStatus::Queued,
                ..
            },
        ) => true,
        (WireRequestMessage::Status { .. }, ResponseMessage::Status { .. }) => true,
        (
            WireRequestMessage::Cancel { .. },
            ResponseMessage::Accepted {
                operation: AcceptedOperation::Cancel,
                state: RequestStatus::Queued | RequestStatus::Running | RequestStatus::Cancelled,
                ..
            },
        ) => true,
        (
            WireRequestMessage::Results {
                after_cursor,
                byte_budget,
                ..
            },
            ResponseMessage::Results {
                request_id: page_request_id,
                after_cursor: response_cursor,
                byte_budget: response_budget,
                page,
                ..
            },
        ) => {
            page_request_id == request_id
                && response_cursor == after_cursor
                && response_budget == byte_budget
                && page.request_id == request_id
                && ipc::validate_result_page(*after_cursor, *byte_budget, page).is_ok()
        }
        (
            WireRequestMessage::Artifact {
                request_id,
                artifact,
                offset,
                byte_budget,
                artifact_version,
            },
            response @ ResponseMessage::Artifact { .. },
        ) => ipc::validate_artifact_reply(
            request_id,
            artifact,
            *offset,
            *byte_budget,
            artifact_version.as_ref(),
            response,
        )
        .is_ok(),
        _ => false,
    };
    if matches_request {
        Ok(response)
    } else {
        Err(ClientError::InvalidResponse)
    }
}

fn fixed_transport_error(error: LocalIpcError, phase: Phase) -> ClientError {
    match error {
        LocalIpcError::UnsafePath => ClientError::UnsafePath,
        LocalIpcError::Unavailable if matches!(phase, Phase::Connect) => ClientError::Unavailable,
        LocalIpcError::Unavailable => ClientError::ExchangeFailed,
        LocalIpcError::Deadline | LocalIpcError::Cancelled => ClientError::TimedOut,
        LocalIpcError::Protocol(ipc::ProtocolError::InvalidRequestId)
            if matches!(phase, Phase::Write) =>
        {
            ClientError::InvalidRequestId
        }
        LocalIpcError::Protocol(_) | LocalIpcError::ResponseMismatch
            if matches!(phase, Phase::Read) =>
        {
            ClientError::InvalidResponse
        }
        LocalIpcError::Protocol(_) => ClientError::InvalidRequest,
        LocalIpcError::PeerDenied => ClientError::PeerUnauthorized,
        LocalIpcError::Io | LocalIpcError::OutOfOrder | LocalIpcError::ResponseMismatch => {
            ClientError::ExchangeFailed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::control::{
        catalog::Lane,
        ipc::{
            ArtifactVersion, ClientRequest, RequestEnvelope, ResponseEnvelope, ResponseErrorCode,
            WireChatRole, WireChatTurn, WireInferenceRequest,
        },
        result_buffer::{
            ArtifactKind, Availability, MediaArtifact, ResultBuffer, ResultEvent, ResultTerminal,
            MAX_PAGE_BYTES, MIN_PAGE_BYTES,
        },
        runtime::NeverCancel,
    };

    fn private_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }

    fn listener(directory: &Path) -> UnixListener {
        let socket_path = directory.join("coordinator.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        listener
    }

    fn context(cancellation: &NeverCancel) -> OperationContext<'_> {
        OperationContext::new(Instant::now() + Duration::from_secs(3), cancellation)
    }

    fn result_page(request_id: &str) -> super::super::result_buffer::ResultPage {
        let mut buffer = ResultBuffer::new();
        buffer.begin(request_id, Lane::Language).unwrap();
        buffer
            .append(request_id, ResultEvent::ChatContent("result body".into()))
            .unwrap();
        buffer
            .terminal(request_id, ResultTerminal::Succeeded)
            .unwrap();
        buffer.page(request_id, 0, MAX_PAGE_BYTES).unwrap()
    }

    fn artifact_reference() -> MediaArtifact {
        MediaArtifact::new(ArtifactKind::LocalJob, "job/41/output.png").unwrap()
    }

    fn one_artifact_response(
        response: ResponseMessage,
        requested_offset: u64,
        requested_version: Option<ArtifactVersion>,
    ) -> Result<ArtifactChunk, ClientError> {
        let directory = private_directory();
        let listener = listener(directory.path());
        let server_requested_version = requested_version.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let ipc::IncomingRequest::Artifact(query) =
                ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("artifact reader must send one artifact query")
            };
            assert_eq!(query.request_id, "request:artifact");
            assert_eq!(query.artifact, artifact_reference());
            assert_eq!(query.offset, requested_offset);
            assert_eq!(query.byte_budget, 2);
            assert_eq!(query.artifact_version, server_requested_version);
            ipc::write_response(&mut stream, &ResponseEnvelope::v3_artifact(response)).unwrap();
        });
        let cancellation = NeverCancel;
        let result = artifact_chunk(
            directory.path(),
            "request:artifact",
            &artifact_reference(),
            requested_offset,
            2,
            requested_version.as_ref(),
            &context(&cancellation),
        );
        server.join().unwrap();
        if let Ok(chunk) = result.as_ref() {
            assert_eq!(chunk.request_id, "request:artifact");
            let debug = format!("{chunk:?}");
            assert!(!debug.contains("job/41/output.png"));
            assert!(!debug.contains("xy"));
        }
        result
    }

    #[test]
    fn native_reader_resumes_verified_cursor_and_rolls_back_a_failed_later_page() {
        let directory = private_directory();
        let listener = listener(directory.path());
        let mut initial = ResultBuffer::new();
        initial.begin("resume-read", Lane::Language).unwrap();
        initial
            .append("resume-read", ResultEvent::ChatContent("first".into()))
            .unwrap();
        let mut consumer =
            super::super::result_consumer::ResultConsumer::new("resume-read", Lane::Language)
                .unwrap();
        consumer
            .consume(&initial.page("resume-read", 0, MAX_PAGE_BYTES).unwrap())
            .unwrap();
        let prior_cursor = consumer.cursor();
        let server = thread::spawn(move || {
            let mut buffer = initial;
            buffer
                .append("resume-read", ResultEvent::ChatContent("second".into()))
                .unwrap();
            // Force another page after the first valid response, then send an
            // inconsistent cursor. Neither output nor cursor may be committed.
            for step in 0..2 {
                let mut stream = listener.accept().unwrap().0;
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let ipc::IncomingRequest::Results(query) =
                    ipc::read_incoming_request(&mut stream).unwrap()
                else {
                    panic!("resume must only query results")
                };
                assert_eq!(query.after_cursor, prior_cursor + step);
                let mut page = buffer
                    .page("resume-read", query.after_cursor, MAX_PAGE_BYTES)
                    .unwrap();
                if step == 0 {
                    page.latest_cursor += 1;
                } else {
                    page.latest_cursor = 0;
                }
                let response = ResponseEnvelope::v2_results(ResponseMessage::Results {
                    request_id: query.request_id,
                    state: RequestStatus::Running,
                    after_cursor: query.after_cursor,
                    byte_budget: query.byte_budget,
                    page,
                });
                // Bypass the server-side validator deliberately: this fixture
                // tests a malformed peer response, not the writer's refusal.
                let payload = serde_json::to_vec(&response).unwrap();
                stream
                    .write_all(&(payload.len() as u32).to_be_bytes())
                    .unwrap();
                stream.write_all(&payload).unwrap();
            }
        });
        assert!(
            read_native_results_into(directory.path(), &mut consumer, &context(&NeverCancel))
                .is_err()
        );
        assert_eq!(consumer.cursor(), prior_cursor);
        let super::super::result_consumer::NativeOutput::Language { content, .. } =
            consumer.output()
        else {
            panic!("wrong lane")
        };
        assert_eq!(content, "first");
        server.join().unwrap();
    }

    #[test]
    fn native_reader_reassembles_multiple_private_ipc_pages_without_submit() {
        let directory = private_directory();
        let listener = listener(directory.path());
        listener.set_nonblocking(true).unwrap();
        let text = "secret-native-output".repeat(10_000);
        let expected = text.clone();
        let server = thread::spawn(move || {
            let mut buffer = ResultBuffer::new();
            buffer.begin("native-read", Lane::Language).unwrap();
            buffer
                .append("native-read", ResultEvent::ChatContent(text))
                .unwrap();
            buffer
                .terminal("native-read", ResultTerminal::Succeeded)
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut after = 0;
            let mut count = 0;
            loop {
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "result query not received");
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(_) => panic!("fixture accept failed"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let ipc::IncomingRequest::Results(query) =
                    ipc::read_incoming_request(&mut stream).unwrap()
                else {
                    panic!("native reader attempted a mutating request")
                };
                assert_eq!(query.request_id, "native-read");
                assert_eq!(query.after_cursor, after);
                assert_eq!(query.byte_budget, MAX_PAGE_BYTES as u32);
                let page = buffer
                    .page(&query.request_id, after, MAX_PAGE_BYTES)
                    .unwrap();
                after = page.next_cursor;
                let done = after == page.latest_cursor;
                ipc::write_response(
                    &mut stream,
                    &ResponseEnvelope::v2_results(ResponseMessage::Results {
                        request_id: query.request_id,
                        state: RequestStatus::Succeeded,
                        after_cursor: query.after_cursor,
                        byte_budget: query.byte_budget,
                        page,
                    }),
                )
                .unwrap();
                count += 1;
                if done {
                    return count;
                }
            }
        });
        let output = read_native_results(
            directory.path(),
            "native-read",
            Lane::Language,
            &context(&NeverCancel),
        )
        .unwrap();
        assert!(output.is_complete());
        let super::super::result_consumer::NativeOutput::Language { content, reasoning } =
            output.output()
        else {
            panic!("wrong native lane")
        };
        assert_eq!(content, &expected);
        assert!(reasoning.is_empty());
        assert!(server.join().unwrap() > 1);
    }

    #[test]
    fn native_reader_unknown_response_is_not_replayed_or_reported_as_empty_success() {
        let directory = private_directory();
        let listener = listener(directory.path());
        listener.set_nonblocking(true).unwrap();
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline);
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => panic!("fixture accept failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let ipc::IncomingRequest::Results(query) =
                ipc::read_incoming_request(&mut stream).unwrap()
            else {
                panic!("unexpected mutation")
            };
            let page = ResultBuffer::new()
                .page(&query.request_id, 0, MAX_PAGE_BYTES)
                .unwrap();
            ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v2_results(ResponseMessage::Results {
                    request_id: query.request_id,
                    state: RequestStatus::NotFound,
                    after_cursor: query.after_cursor,
                    byte_budget: query.byte_budget,
                    page,
                }),
            )
            .unwrap();
            // No retry follows this read-only but unavailable response.
            drop(stream);
            thread::sleep(Duration::from_millis(25));
            assert!(
                matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
            );
        });
        assert!(matches!(
            read_native_results(
                directory.path(),
                "unknown",
                Lane::Decisions,
                &context(&NeverCancel)
            ),
            Err(NativeReadError::Assembly(
                super::super::result_consumer::ConsumeError::Unknown
            ))
        ));
        server.join().unwrap();
    }

    fn request(operation: ClientRequestKind) -> RequestEnvelope {
        RequestEnvelope::v1(match operation {
            ClientRequestKind::Status => WireRequestMessage::Status {
                request_id: "request-1".into(),
            },
            ClientRequestKind::Cancel => WireRequestMessage::Cancel {
                request_id: "request-1".into(),
            },
            ClientRequestKind::Submit => WireRequestMessage::Submit {
                request_id: "request-1".into(),
                timeout_ms: 10_000,
                request: WireInferenceRequest::Language {
                    profile_id: "model".into(),
                    turns: vec![WireChatTurn {
                        role: WireChatRole::User,
                        content: "hello".into(),
                    }],
                    max_tokens: 8,
                },
            },
        })
    }

    #[derive(Clone, Copy)]
    enum ClientRequestKind {
        Submit,
        Status,
        Cancel,
    }

    #[test]
    fn one_shot_exchange_covers_submit_status_and_cancel_shapes() {
        for (kind, response) in [
            (
                ClientRequestKind::Submit,
                ResponseMessage::Accepted {
                    request_id: "request-1".into(),
                    operation: AcceptedOperation::Submit,
                    state: RequestStatus::Queued,
                },
            ),
            (
                ClientRequestKind::Status,
                ResponseMessage::Status {
                    request_id: "request-1".into(),
                    state: RequestStatus::Running,
                },
            ),
            (
                ClientRequestKind::Cancel,
                ResponseMessage::Accepted {
                    request_id: "request-1".into(),
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Cancelled,
                },
            ),
        ] {
            let directory = private_directory();
            let listener = listener(directory.path());
            let expected_response = response.clone();
            let server = thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let request = ipc::read_request(&mut stream).unwrap();
                match (kind, request) {
                    (ClientRequestKind::Submit, ClientRequest::Submit { request_id, .. })
                    | (ClientRequestKind::Status, ClientRequest::Status { request_id })
                    | (ClientRequestKind::Cancel, ClientRequest::Cancel { request_id }) => {
                        assert_eq!(request_id, "request-1");
                    }
                    _ => panic!("coordinator client sent the wrong operation"),
                }
                ipc::write_response(&mut stream, &ResponseEnvelope::v1(response)).unwrap();
            });
            let never_cancel = NeverCancel;
            let actual =
                exchange(directory.path(), request(kind), &context(&never_cancel)).unwrap();
            server.join().unwrap();
            assert_eq!(actual, expected_response);
        }
    }

    #[test]
    fn one_shot_results_exchange_validates_query_echo_and_bounded_page() {
        let directory = private_directory();
        let listener = listener(directory.path());
        let expected_page = result_page("request:results");
        let server_page = expected_page.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = ipc::read_incoming_request(&mut stream).unwrap();
            assert!(matches!(
                request,
                ipc::IncomingRequest::Results(ref query)
                    if query.request_id == "request:results"
                        && query.after_cursor == 0
                        && query.byte_budget == MIN_PAGE_BYTES as u32
            ));
            ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v2_results(ResponseMessage::Results {
                    request_id: "request:results".into(),
                    after_cursor: 0,
                    byte_budget: MIN_PAGE_BYTES as u32,
                    state: RequestStatus::Succeeded,
                    page: server_page,
                }),
            )
            .unwrap();
        });
        let never_cancel = NeverCancel;
        let response = exchange(
            directory.path(),
            RequestEnvelope::v2_results("request:results", 0, MIN_PAGE_BYTES as u32),
            &context(&never_cancel),
        )
        .unwrap();
        server.join().unwrap();
        assert!(matches!(
            response,
            ResponseMessage::Results {
                state: RequestStatus::Succeeded,
                page,
                ..
            } if page == expected_page
        ));
    }

    #[test]
    fn one_shot_artifact_chunk_decodes_data_and_preserves_first_version_token() {
        let artifact = artifact_reference();
        let version = ArtifactVersion::new(&"a".repeat(64)).unwrap();
        let chunk = one_artifact_response(
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact: artifact.clone(),
                availability: Availability::Available,
                offset: 0,
                total_bytes: 9,
                next_offset: 2,
                end_of_file: false,
                artifact_version: Some(version.clone()),
                bytes: b"xy".to_vec(),
            },
            0,
            None,
        )
        .unwrap();
        assert_eq!(chunk.artifact, artifact);
        assert_eq!(chunk.offset, 0);
        assert_eq!(chunk.total_bytes, 9);
        assert_eq!(chunk.next_offset, 2);
        assert!(!chunk.end_of_file);
        assert_eq!(chunk.artifact_version, Some(version.clone()));
        assert_eq!(chunk.bytes.as_slice(), b"xy");
        assert!(!format!("{chunk:?}").contains(version.as_str()));

        let resumed = one_artifact_response(
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact: artifact_reference(),
                availability: Availability::Available,
                offset: 2,
                total_bytes: 4,
                next_offset: 4,
                end_of_file: true,
                artifact_version: Some(version.clone()),
                bytes: b"xy".to_vec(),
            },
            2,
            Some(version.clone()),
        )
        .unwrap();
        assert!(resumed.end_of_file);
        assert_eq!(resumed.artifact_version, Some(version));
    }

    #[test]
    fn artifact_chunk_surfaces_unknown_availability_and_rejects_mismatched_replies() {
        let artifact = artifact_reference();
        let version = ArtifactVersion::new(&"a".repeat(64)).unwrap();
        let unknown = one_artifact_response(
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact: artifact.clone(),
                availability: Availability::Unknown,
                offset: 2,
                total_bytes: 0,
                next_offset: 2,
                end_of_file: false,
                artifact_version: None,
                bytes: Vec::new(),
            },
            2,
            Some(version.clone()),
        )
        .unwrap();
        assert_eq!(unknown.availability, Availability::Unknown);
        assert!(unknown.bytes.is_empty());
        assert!(unknown.artifact_version.is_none());
        let expired = one_artifact_response(
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact: artifact.clone(),
                availability: Availability::Expired,
                offset: 2,
                total_bytes: 0,
                next_offset: 2,
                end_of_file: false,
                artifact_version: None,
                bytes: Vec::new(),
            },
            2,
            Some(version.clone()),
        )
        .unwrap();
        assert_eq!(expired.availability, Availability::Expired);
        assert!(expired.bytes.is_empty());
        assert!(expired.artifact_version.is_none());

        let wrong_artifact =
            MediaArtifact::new(ArtifactKind::Backend, "history/42/image.png").unwrap();
        let wrong_version = ArtifactVersion::new(&"b".repeat(64)).unwrap();
        let responses = [
            ResponseMessage::Artifact {
                request_id: "request:other".into(),
                artifact: artifact.clone(),
                availability: Availability::Available,
                offset: 2,
                total_bytes: 4,
                next_offset: 4,
                end_of_file: true,
                artifact_version: Some(version.clone()),
                bytes: b"xy".to_vec(),
            },
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact: wrong_artifact,
                availability: Availability::Available,
                offset: 2,
                total_bytes: 4,
                next_offset: 4,
                end_of_file: true,
                artifact_version: Some(version.clone()),
                bytes: b"xy".to_vec(),
            },
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact: artifact.clone(),
                availability: Availability::Available,
                offset: 3,
                total_bytes: 5,
                next_offset: 5,
                end_of_file: true,
                artifact_version: Some(version.clone()),
                bytes: b"xy".to_vec(),
            },
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact: artifact.clone(),
                availability: Availability::Available,
                offset: 2,
                total_bytes: 5,
                next_offset: 5,
                end_of_file: true,
                artifact_version: Some(version.clone()),
                bytes: b"xyz".to_vec(),
            },
            ResponseMessage::Artifact {
                request_id: "request:artifact".into(),
                artifact,
                availability: Availability::Available,
                offset: 2,
                total_bytes: 4,
                next_offset: 4,
                end_of_file: true,
                artifact_version: Some(wrong_version),
                bytes: b"xy".to_vec(),
            },
        ];
        for response in responses {
            assert_eq!(
                one_artifact_response(response, 2, Some(version.clone())).err(),
                Some(ClientError::InvalidResponse)
            );
        }
    }

    #[test]
    fn results_exchange_rejects_mismatched_cursor_or_budget_echo() {
        for mismatch_budget in [false, true] {
            let directory = private_directory();
            let listener = listener(directory.path());
            let server = thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let _request = ipc::read_incoming_request(&mut stream).unwrap();
                let mut buffer = ResultBuffer::new();
                buffer.begin("request:results", Lane::Language).unwrap();
                buffer
                    .append(
                        "request:results",
                        ResultEvent::ChatContent("result body".into()),
                    )
                    .unwrap();
                buffer
                    .terminal("request:results", ResultTerminal::Succeeded)
                    .unwrap();
                let page = buffer
                    .page(
                        "request:results",
                        if mismatch_budget { 0 } else { 1 },
                        MAX_PAGE_BYTES,
                    )
                    .unwrap();
                ipc::write_response(
                    &mut stream,
                    &ResponseEnvelope::v2_results(ResponseMessage::Results {
                        request_id: "request:results".into(),
                        after_cursor: if mismatch_budget { 0 } else { 1 },
                        byte_budget: if mismatch_budget {
                            (MIN_PAGE_BYTES + 1) as u32
                        } else {
                            MIN_PAGE_BYTES as u32
                        },
                        state: RequestStatus::Succeeded,
                        page,
                    }),
                )
                .unwrap();
            });
            let never_cancel = NeverCancel;
            let error = exchange(
                directory.path(),
                RequestEnvelope::v2_results("request:results", 0, MIN_PAGE_BYTES as u32),
                &context(&never_cancel),
            )
            .unwrap_err();
            server.join().unwrap();
            assert_eq!(error, ClientError::InvalidResponse);
        }
    }

    #[test]
    fn operation_and_state_validation_rejects_misrouted_replies() {
        let request_submit = request(ClientRequestKind::Submit);
        for state in [
            RequestStatus::Queued,
            RequestStatus::Running,
            RequestStatus::Succeeded,
            RequestStatus::Failed,
            RequestStatus::Cancelled,
            RequestStatus::Interrupted,
            RequestStatus::NotFound,
        ] {
            let response = ResponseMessage::Accepted {
                request_id: "request-1".into(),
                operation: AcceptedOperation::Submit,
                state,
            };
            assert_eq!(
                validate_response(&request_submit.message, "request-1", response).is_ok(),
                state == RequestStatus::Queued
            );
        }

        let request_cancel = request(ClientRequestKind::Cancel);
        for state in [
            RequestStatus::Queued,
            RequestStatus::Running,
            RequestStatus::Cancelled,
        ] {
            let response = ResponseMessage::Accepted {
                request_id: "request-1".into(),
                operation: AcceptedOperation::Cancel,
                state,
            };
            assert!(validate_response(&request_cancel.message, "request-1", response).is_ok());
        }
        for state in [
            RequestStatus::Succeeded,
            RequestStatus::Failed,
            RequestStatus::Interrupted,
            RequestStatus::NotFound,
        ] {
            let response = ResponseMessage::Accepted {
                request_id: "request-1".into(),
                operation: AcceptedOperation::Cancel,
                state,
            };
            assert_eq!(
                validate_response(&request_cancel.message, "request-1", response),
                Err(ClientError::InvalidResponse)
            );
        }

        let request_status = request(ClientRequestKind::Status);
        let submit_response = ResponseMessage::Accepted {
            request_id: "request-1".into(),
            operation: AcceptedOperation::Submit,
            state: RequestStatus::Queued,
        };
        assert_eq!(
            validate_response(&request_status.message, "request-1", submit_response),
            Err(ClientError::InvalidResponse)
        );
        let mismatched_id = ResponseMessage::Status {
            request_id: "other-request".into(),
            state: RequestStatus::Queued,
        };
        assert_eq!(
            validate_response(&request_status.message, "request-1", mismatched_id),
            Err(ClientError::InvalidResponse)
        );
    }

    #[test]
    fn response_errors_are_accepted_and_remain_caller_correlated() {
        let request = request(ClientRequestKind::Status);
        let with_id = ResponseMessage::Error {
            request_id: Some("request-1".into()),
            code: ResponseErrorCode::Unavailable,
        };
        assert!(validate_response(&request.message, "request-1", with_id).is_ok());
        let without_id = ResponseMessage::Error {
            request_id: None,
            code: ResponseErrorCode::InvalidFrame,
        };
        assert!(validate_response(&request.message, "request-1", without_id).is_ok());
        let unrelated_id = ResponseMessage::Error {
            request_id: Some("other-request".into()),
            code: ResponseErrorCode::Unavailable,
        };
        assert_eq!(
            validate_response(&request.message, "request-1", unrelated_id),
            Err(ClientError::InvalidResponse)
        );
    }

    #[test]
    fn transport_failures_are_fixed_and_never_fall_back() {
        let missing = private_directory();
        let never_cancel = NeverCancel;
        let unavailable = exchange(
            missing.path(),
            request(ClientRequestKind::Status),
            &context(&never_cancel),
        )
        .unwrap_err();
        assert_eq!(unavailable.to_string(), "coordinator is unavailable");
        assert!(!missing.path().join("coordinator.sock").exists());

        let unsafe_directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(
            unsafe_directory.path(),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        let unsafe_path = exchange(
            unsafe_directory.path(),
            request(ClientRequestKind::Cancel),
            &context(&never_cancel),
        )
        .unwrap_err();
        assert_eq!(
            unsafe_path.to_string(),
            "coordinator directory or socket is unsafe"
        );
        assert!(!unsafe_path
            .to_string()
            .contains(unsafe_directory.path().to_str().unwrap()));
    }

    #[test]
    fn invalid_request_id_is_rejected_before_connecting() {
        let missing = private_directory();
        let invalid = RequestEnvelope::v1(WireRequestMessage::Status {
            request_id: "Bearer secret-value".into(),
        });
        let never_cancel = NeverCancel;
        let error = exchange(missing.path(), invalid, &context(&never_cancel)).unwrap_err();
        assert_eq!(error, ClientError::InvalidRequestId);
        assert_eq!(error.to_string(), "invalid coordinator request ID");
        assert!(!missing.path().join("coordinator.sock").exists());
    }

    #[test]
    fn invalid_transport_reply_is_fixed() {
        let directory = private_directory();
        let listener = listener(directory.path());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request = ipc::read_request(&mut stream).unwrap();
            ipc::write_response(
                &mut stream,
                &ResponseEnvelope::v1(ResponseMessage::Accepted {
                    request_id: "request-1".into(),
                    operation: AcceptedOperation::Submit,
                    state: RequestStatus::Queued,
                }),
            )
            .unwrap();
        });
        let never_cancel = NeverCancel;
        let error = exchange(
            directory.path(),
            request(ClientRequestKind::Status),
            &context(&never_cancel),
        )
        .unwrap_err();
        server.join().unwrap();
        assert_eq!(error.to_string(), "coordinator response is invalid");
    }
}
