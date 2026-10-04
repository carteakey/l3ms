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
        self, AcceptedOperation, RequestEnvelope, RequestStatus, ResponseMessage,
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
    OwnershipUnavailable,
    EndpointUnavailable,
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
            Self::OwnershipUnavailable => "coordinator ownership is unavailable",
            Self::EndpointUnavailable => "coordinator endpoint is unavailable",
            Self::ExchangeFailed => "coordinator exchange failed",
        })
    }
}

impl Error for ClientError {}

#[derive(Clone, Copy)]
enum Phase {
    Connect,
    Write,
    Read,
}

/// Sends one v1 request over the caller-selected private socket directory.
///
/// `context` bounds the entire connect/write/read sequence. Request IDs are
/// checked before connecting; the response must correlate to that ID and use
/// the operation-specific v1 response shape. The function does not retry after
/// a write or start any inference/runtime work itself.
pub fn exchange(
    socket_directory: &Path,
    request: RequestEnvelope,
    context: &OperationContext<'_>,
) -> Result<ResponseMessage, ClientError> {
    let request_id = request_id(&request.message);
    ipc::validate_request_id(request_id).map_err(|_| ClientError::InvalidRequestId)?;

    let mut connection = LocalConnection::connect(socket_directory, context)
        .map_err(|error| fixed_transport_error(error, Phase::Connect))?;
    connection
        .write_request(&request, context)
        .map_err(|error| fixed_transport_error(error, Phase::Write))?;
    let response = connection
        .read_response(context)
        .map_err(|error| fixed_transport_error(error, Phase::Read))?
        .message;

    validate_response(&request.message, request_id, response)
}

fn request_id(message: &WireRequestMessage) -> &str {
    match message {
        WireRequestMessage::Submit { request_id, .. }
        | WireRequestMessage::Cancel { request_id }
        | WireRequestMessage::Status { request_id } => request_id,
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
        ResponseMessage::Error { request_id, .. } => request_id.as_deref(),
    };
    if response_id.is_some_and(|response_id| response_id != request_id) {
        return Err(ClientError::InvalidResponse);
    }

    let matches_request = matches!(
        (request, &response),
        (_, ResponseMessage::Error { .. })
            | (
                WireRequestMessage::Submit { .. },
                ResponseMessage::Accepted {
                    operation: AcceptedOperation::Submit,
                    state: RequestStatus::Queued,
                    ..
                }
            )
            | (
                WireRequestMessage::Status { .. },
                ResponseMessage::Status { .. }
            )
            | (
                WireRequestMessage::Cancel { .. },
                ResponseMessage::Accepted {
                    operation: AcceptedOperation::Cancel,
                    state: RequestStatus::Queued
                        | RequestStatus::Running
                        | RequestStatus::Cancelled,
                    ..
                }
            )
    );
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
        LocalIpcError::OwnershipUnavailable => ClientError::OwnershipUnavailable,
        LocalIpcError::ExistingSocket => ClientError::EndpointUnavailable,
        LocalIpcError::Io | LocalIpcError::OutOfOrder | LocalIpcError::ResponseMismatch => {
            ClientError::ExchangeFailed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::control::{
        ipc::{
            ClientRequest, RequestEnvelope, ResponseEnvelope, ResponseErrorCode, WireChatRole,
            WireChatTurn, WireInferenceRequest,
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
