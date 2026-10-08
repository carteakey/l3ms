//! Concrete bounded language execution, not lifecycle management.
//!
//! The future executor must hold the host/resource lease and prove admission
//! before calling this adapter. It never acquires/relinquishes GPU ownership,
//! stops a runtime, changes configured endpoints, or retries a request. Once
//! HTTP dispatch starts, any failure is conservatively uncertain: dropping a
//! connection does not acknowledge server-side cancellation or GPU release.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::catalog::Catalog;
use super::requests::{
    ChatRole, EmbeddingUsage, InferenceRequest, InferenceResult, MAX_EMBEDDING_INPUTS,
    MAX_EMBEDDING_JSON_BYTES,
};
use super::result_buffer::{ResultBuffer, ResultEvent};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::OperationContext;
use crate::chat::ChatRequest;
use crate::llama_swap::{auth_headers, normalize_base_url};

const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_WIRE_BYTES: usize = 8 * 1024 * 1024;
/// Canonical embedding output is capped at 4 MiB; this wire allowance covers
/// the bounded OpenAI-compatible response envelope and per-row metadata.
const MAX_EMBEDDING_WIRE_BYTES: usize = MAX_EMBEDDING_JSON_BYTES + 32 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchDisposition {
    NoDispatch,
    Uncertain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LanguageErrorKind {
    Configuration,
    InvalidRequest,
    Cancelled,
    Deadline,
    Transport,
    Rejected,
    Protocol,
    TooLarge,
    ResultPublication,
}

/// Deliberately omits endpoints, native payloads, response bodies and source
/// errors, which may contain prompts or credentials.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LanguageFailure {
    pub disposition: DispatchDisposition,
    pub kind: LanguageErrorKind,
}

impl fmt::Display for LanguageFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "language execution {:?}: {:?}",
            self.disposition, self.kind
        )
    }
}
impl Error for LanguageFailure {}

pub enum ChatDelta<'a> {
    Content(&'a str),
    Reasoning(&'a str),
}

/// Incremental native chat information produced before the full response is
/// available. Stream completion is a backend marker only; it is not the
/// coordinator job's terminal state.
pub enum ChatStreamEvent<'a> {
    Delta(ChatDelta<'a>),
    Usage { completion_tokens: u64 },
    StreamComplete,
}

/// A streaming consumer may stop for cancellation or because publication to
/// the already-reserved result slot failed. Either outcome after dispatch is
/// uncertain and never triggers an automatic retry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatEventCallbackError {
    Cancelled,
    Publication,
}

/// Caller-configured connection; neither endpoint discovery nor credential
/// lookup happens here. Does not implement Debug, to avoid leaking credentials.
pub struct LanguageAdapter {
    endpoint: String,
    api_key: Option<String>,
    #[cfg(target_os = "linux")]
    private_engine: Option<super::native_http::PinnedEngineSocket>,
}

impl LanguageAdapter {
    pub fn new(endpoint: &str, api_key: Option<&str>) -> Result<Self, LanguageFailure> {
        let failure = failure(
            DispatchDisposition::NoDispatch,
            LanguageErrorKind::Configuration,
        );
        let endpoint = normalize_base_url(endpoint).map_err(|_| failure)?;
        let parsed = reqwest::Url::parse(&endpoint).map_err(|_| failure)?;
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(failure);
        }
        auth_headers(api_key).map_err(|_| failure)?;
        Ok(Self {
            endpoint,
            api_key: api_key.map(str::to_owned),
            #[cfg(target_os = "linux")]
            private_engine: None,
        })
    }

    /// Explicit owned-engine transport. The lifecycle owner must attest the
    /// helper and admission before use; socket validation alone is not proof.
    #[cfg(target_os = "linux")]
    pub fn new_private_engine(
        directory: &std::path::Path,
        api_key: Option<&str>,
    ) -> Result<Self, LanguageFailure> {
        let mut adapter = Self::new("http://127.0.0.1", api_key)?;
        adapter.private_engine = Some(
            super::native_http::PinnedEngineSocket::open(directory).map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?,
        );
        Ok(adapter)
    }

    /// Run on a blocking executor thread, not inside an entered Tokio runtime.
    /// Delta callbacks must return promptly; synchronous user callbacks cannot
    /// be preempted by this adapter's deadline/cancellation checks.
    pub fn execute(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        context: &OperationContext<'_>,
        mut on_delta: impl FnMut(ChatDelta<'_>) -> bool,
    ) -> Result<InferenceResult, LanguageFailure> {
        self.execute_with_events(catalog, request, context, |event| match event {
            ChatStreamEvent::Delta(delta) => {
                if on_delta(delta) {
                    Ok(())
                } else {
                    Err(ChatEventCallbackError::Cancelled)
                }
            }
            ChatStreamEvent::Usage { .. } | ChatStreamEvent::StreamComplete => Ok(()),
        })
    }

    /// Run one bounded chat request and expose deltas, optional token usage,
    /// and the native stream-complete marker as the response arrives. Callbacks
    /// run synchronously on the adapter thread and must return promptly.
    pub fn execute_with_events(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        context: &OperationContext<'_>,
        on_event: impl FnMut(ChatStreamEvent<'_>) -> Result<(), ChatEventCallbackError>,
    ) -> Result<InferenceResult, LanguageFailure> {
        self.execute_with_events_inner(catalog, request, None, context, on_event)
    }

    /// Execute against the complete route retained at coordinator acceptance.
    /// The refreshed profile, runtime, device, native route, task set and
    /// readiness must all still match before any HTTP request is constructed.
    pub fn execute_with_events_for_route(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        accepted_route: &ResolvedProfileRoute,
        context: &OperationContext<'_>,
        on_event: impl FnMut(ChatStreamEvent<'_>) -> Result<(), ChatEventCallbackError>,
    ) -> Result<InferenceResult, LanguageFailure> {
        self.execute_with_events_inner(catalog, request, Some(accepted_route), context, on_event)
    }

    fn execute_with_events_inner(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        accepted_route: Option<&ResolvedProfileRoute>,
        context: &OperationContext<'_>,
        mut on_event: impl FnMut(ChatStreamEvent<'_>) -> Result<(), ChatEventCallbackError>,
    ) -> Result<InferenceResult, LanguageFailure> {
        check_context(context, DispatchDisposition::NoDispatch)?;
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Configuration,
            ));
        }
        let catalog = catalog.refresh().map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            )
        })?;
        let native = native_request_for_route(&catalog, request, accepted_route)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?;
        runtime.block_on(self.execute_async(&native, context, &mut on_event))
    }

    /// Execute chat once while publishing native deltas into an already-made
    /// full-reservation result slot. Preflight is read-only; it neither creates
    /// a result nor admits work. This method never marks the outer result
    /// terminal: its coordinator must durably record terminal state separately.
    /// Any failure after HTTP dispatch is uncertain and must not be retried.
    pub fn execute_chat_into_result_buffer(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        request_id: &str,
        result_buffer: &mut ResultBuffer,
        context: &OperationContext<'_>,
    ) -> Result<InferenceResult, LanguageFailure> {
        if !matches!(request, InferenceRequest::Chat { .. }) {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            ));
        }
        result_buffer
            .can_append_reserved_chat_stream(request_id)
            .map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::ResultPublication,
                )
            })?;
        self.execute_with_events(catalog, request, context, |event| {
            let result_event = match event {
                ChatStreamEvent::Delta(ChatDelta::Content(text)) => {
                    ResultEvent::ChatContent(text.to_owned())
                }
                ChatStreamEvent::Delta(ChatDelta::Reasoning(text)) => {
                    ResultEvent::ChatReasoning(text.to_owned())
                }
                ChatStreamEvent::Usage { completion_tokens } => {
                    ResultEvent::ChatUsage { completion_tokens }
                }
                ChatStreamEvent::StreamComplete => ResultEvent::ChatStreamComplete,
            };
            result_buffer
                .append(request_id, result_event)
                .map_err(|_| ChatEventCallbackError::Publication)
        })
    }

    /// Run one bounded OpenAI-compatible embeddings request. This is an
    /// adapter operation only: callers must already hold lifecycle admission.
    /// It performs exactly one non-streaming POST and never retries or guesses
    /// an endpoint/model.
    pub fn execute_embeddings(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        context: &OperationContext<'_>,
    ) -> Result<InferenceResult, LanguageFailure> {
        self.execute_embeddings_inner(catalog, request, None, context)
    }

    /// Execute embeddings only if the complete route retained at acceptance
    /// remains identical in the refreshed catalog. Drift fails before HTTP.
    pub fn execute_embeddings_for_route(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        accepted_route: &ResolvedProfileRoute,
        context: &OperationContext<'_>,
    ) -> Result<InferenceResult, LanguageFailure> {
        self.execute_embeddings_inner(catalog, request, Some(accepted_route), context)
    }

    fn execute_embeddings_inner(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        accepted_route: Option<&ResolvedProfileRoute>,
        context: &OperationContext<'_>,
    ) -> Result<InferenceResult, LanguageFailure> {
        check_context(context, DispatchDisposition::NoDispatch)?;
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Configuration,
            ));
        }
        let catalog = catalog.refresh().map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            )
        })?;
        let native = native_embedding_request_for_route(&catalog, request, accepted_route)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?;
        runtime.block_on(self.execute_embeddings_async(&native, request, context))
    }

    async fn execute_embeddings_async(
        &self,
        native: &NativeEmbeddingsRequest,
        request: &InferenceRequest,
        context: &OperationContext<'_>,
    ) -> Result<InferenceResult, LanguageFailure> {
        check_context(context, DispatchDisposition::NoDispatch)?;
        let client = self.http_client(context)?;
        let http = client
            .post(format!("{}/v1/embeddings", self.endpoint))
            .header(reqwest::header::ACCEPT, "application/json")
            .json(native);
        check_context(context, DispatchDisposition::NoDispatch)?;
        let send = http.send();
        tokio::pin!(send);
        let response = loop {
            check_context(context, DispatchDisposition::Uncertain)?;
            match tokio::time::timeout(context.remaining().min(POLL_INTERVAL), &mut send).await {
                Err(_) => continue,
                Ok(result) => break result.map_err(transport_failure)?,
            }
        };
        check_context(context, DispatchDisposition::Uncertain)?;
        if !response.status().is_success() {
            return Err(failure(
                DispatchDisposition::Uncertain,
                LanguageErrorKind::Rejected,
            ));
        }
        if !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
            })
        {
            return Err(failure(
                DispatchDisposition::Uncertain,
                LanguageErrorKind::Protocol,
            ));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_EMBEDDING_WIRE_BYTES as u64)
        {
            return Err(failure(
                DispatchDisposition::Uncertain,
                LanguageErrorKind::TooLarge,
            ));
        }
        let mut response = response;
        let mut body = Vec::new();
        loop {
            check_context(context, DispatchDisposition::Uncertain)?;
            let chunk = response.chunk();
            tokio::pin!(chunk);
            let bytes = loop {
                check_context(context, DispatchDisposition::Uncertain)?;
                match tokio::time::timeout(context.remaining().min(POLL_INTERVAL), &mut chunk).await
                {
                    Err(_) => continue,
                    Ok(result) => break result.map_err(transport_failure)?,
                }
            };
            let Some(bytes) = bytes else {
                break;
            };
            if bytes.len() > MAX_EMBEDDING_WIRE_BYTES.saturating_sub(body.len()) {
                return Err(failure(
                    DispatchDisposition::Uncertain,
                    LanguageErrorKind::TooLarge,
                ));
            }
            body.extend_from_slice(&bytes);
        }
        check_context(context, DispatchDisposition::Uncertain)?;
        let response: EmbeddingsResponse = serde_json::from_slice(&body)
            .map_err(|_| failure(DispatchDisposition::Uncertain, LanguageErrorKind::Protocol))?;
        let valid_data = response.object == "list"
            && !response.model.is_empty()
            && response.data.len() == native.input.len()
            && response
                .data
                .iter()
                .enumerate()
                .all(|(index, row)| row.object == "embedding" && row.index == index as u32);
        if !valid_data {
            return Err(failure(
                DispatchDisposition::Uncertain,
                LanguageErrorKind::Protocol,
            ));
        }
        let result = InferenceResult::Embeddings {
            vectors: response.data.into_iter().map(|row| row.embedding).collect(),
            usage: response.usage,
        };
        result
            .validate_for(request)
            .map_err(|_| failure(DispatchDisposition::Uncertain, LanguageErrorKind::Protocol))?;
        Ok(result)
    }

    async fn execute_async(
        &self,
        request: &ChatRequest,
        context: &OperationContext<'_>,
        on_event: &mut impl FnMut(ChatStreamEvent<'_>) -> Result<(), ChatEventCallbackError>,
    ) -> Result<InferenceResult, LanguageFailure> {
        check_context(context, DispatchDisposition::NoDispatch)?;
        let client = self.http_client(context)?;
        let http = client
            .post(format!("{}/v1/chat/completions", self.endpoint))
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(&request.payload());
        check_context(context, DispatchDisposition::NoDispatch)?;
        // No retries. Even a send failure may follow partial transmission.
        let send = http.send();
        tokio::pin!(send);
        let mut response = loop {
            check_context(context, DispatchDisposition::Uncertain)?;
            match tokio::time::timeout(context.remaining().min(POLL_INTERVAL), &mut send).await {
                Err(_) => continue,
                Ok(result) => break result.map_err(transport_failure)?,
            }
        };
        check_context(context, DispatchDisposition::Uncertain)?;
        if !response.status().is_success() {
            return Err(failure(
                DispatchDisposition::Uncertain,
                LanguageErrorKind::Rejected,
            ));
        }
        if !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
            })
        {
            return Err(failure(
                DispatchDisposition::Uncertain,
                LanguageErrorKind::Protocol,
            ));
        }
        let mut output = Output::default();
        let mut line = Vec::new();
        let mut wire_bytes = 0usize;
        loop {
            check_context(context, DispatchDisposition::Uncertain)?;
            let chunk = response.chunk();
            tokio::pin!(chunk);
            let bytes = loop {
                check_context(context, DispatchDisposition::Uncertain)?;
                match tokio::time::timeout(context.remaining().min(POLL_INTERVAL), &mut chunk).await
                {
                    Err(_) => continue,
                    Ok(result) => break result.map_err(transport_failure)?,
                }
            };
            let Some(bytes) = bytes else {
                // EOF without [DONE], or a partial final line, is not proof of
                // a completed backend stream. We defer publishing StreamComplete
                // until EOF so late malformed/data records cannot expose it.
                if !line.is_empty()
                    || !output.done_marker_seen
                    || (output.content.is_empty() && output.reasoning.is_empty())
                {
                    return Err(failure(
                        DispatchDisposition::Uncertain,
                        LanguageErrorKind::Protocol,
                    ));
                }
                check_context(context, DispatchDisposition::Uncertain)?;
                notify_chat_event(on_event, ChatStreamEvent::StreamComplete)?;
                return Ok(InferenceResult::Chat {
                    content: output.content,
                    reasoning: output.reasoning,
                    completion_tokens: output.completion_tokens,
                });
            };
            wire_bytes = wire_bytes.saturating_add(bytes.len());
            if wire_bytes > MAX_WIRE_BYTES {
                return Err(failure(
                    DispatchDisposition::Uncertain,
                    LanguageErrorKind::TooLarge,
                ));
            }
            for byte in bytes {
                check_context(context, DispatchDisposition::Uncertain)?;
                if line.len() >= MAX_LINE_BYTES {
                    return Err(failure(
                        DispatchDisposition::Uncertain,
                        LanguageErrorKind::TooLarge,
                    ));
                }
                line.push(byte);
                if byte == b'\n' {
                    output.consume_line(&line, on_event)?;
                    line.clear();
                }
            }
        }
    }

    fn http_client(
        &self,
        context: &OperationContext<'_>,
    ) -> Result<reqwest::Client, LanguageFailure> {
        let builder = reqwest::Client::builder()
            .retry(reqwest::retry::never())
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(context.remaining().min(Duration::from_secs(2)))
            .timeout(context.remaining())
            .default_headers(auth_headers(self.api_key.as_deref()).map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?);
        #[cfg(target_os = "linux")]
        let builder = match &self.private_engine {
            Some(socket) => socket.configure(builder).map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?,
            None => builder,
        };
        builder.build().map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Configuration,
            )
        })
    }
}

fn failure(disposition: DispatchDisposition, kind: LanguageErrorKind) -> LanguageFailure {
    LanguageFailure { disposition, kind }
}

fn notify_chat_event(
    on_event: &mut impl FnMut(ChatStreamEvent<'_>) -> Result<(), ChatEventCallbackError>,
    event: ChatStreamEvent<'_>,
) -> Result<(), LanguageFailure> {
    on_event(event).map_err(|error| {
        failure(
            DispatchDisposition::Uncertain,
            match error {
                ChatEventCallbackError::Cancelled => LanguageErrorKind::Cancelled,
                ChatEventCallbackError::Publication => LanguageErrorKind::ResultPublication,
            },
        )
    })
}

fn transport_failure(error: reqwest::Error) -> LanguageFailure {
    failure(
        DispatchDisposition::Uncertain,
        if error.is_timeout() {
            LanguageErrorKind::Deadline
        } else {
            LanguageErrorKind::Transport
        },
    )
}

fn check_context(
    context: &OperationContext<'_>,
    disposition: DispatchDisposition,
) -> Result<(), LanguageFailure> {
    if context.is_cancelled() {
        return Err(failure(disposition, LanguageErrorKind::Cancelled));
    }
    if context.remaining().is_zero() {
        return Err(failure(disposition, LanguageErrorKind::Deadline));
    }
    Ok(())
}

#[cfg(test)]
fn native_request(
    catalog: &Catalog,
    request: &InferenceRequest,
) -> Result<ChatRequest, LanguageFailure> {
    native_request_for_route(catalog, request, None)
}

fn native_request_for_route(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: Option<&ResolvedProfileRoute>,
) -> Result<ChatRequest, LanguageFailure> {
    let invalid = failure(
        DispatchDisposition::NoDispatch,
        LanguageErrorKind::InvalidRequest,
    );
    request.validate(catalog).map_err(|_| invalid)?;
    let route = resolve_profile_route(catalog, request.profile()).map_err(|_| invalid)?;
    if accepted_route.is_some_and(|accepted| accepted != &route) {
        return Err(invalid);
    }
    let NativeRoute::LlamaSwapModel { model_id, .. } = route.native_route else {
        return Err(invalid);
    };
    let InferenceRequest::Chat {
        turns, max_tokens, ..
    } = request
    else {
        return Err(invalid);
    };
    let mut native = ChatRequest::new(
        model_id.to_string(),
        turns
            .iter()
            .map(|turn| {
                let role = match turn.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                };
                (role.to_owned(), turn.content.clone())
            })
            .collect(),
    );
    native.max_tokens = *max_tokens;
    Ok(native)
}

fn native_embedding_request_for_route(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: Option<&ResolvedProfileRoute>,
) -> Result<NativeEmbeddingsRequest, LanguageFailure> {
    let invalid = failure(
        DispatchDisposition::NoDispatch,
        LanguageErrorKind::InvalidRequest,
    );
    request.validate(catalog).map_err(|_| invalid)?;
    let route = resolve_profile_route(catalog, request.profile()).map_err(|_| invalid)?;
    if accepted_route.is_some_and(|accepted| accepted != &route) {
        return Err(invalid);
    }
    let NativeRoute::LlamaSwapModel { model_id, .. } = route.native_route else {
        return Err(invalid);
    };
    let InferenceRequest::Embeddings { input, .. } = request else {
        return Err(invalid);
    };
    if input.len() > MAX_EMBEDDING_INPUTS {
        return Err(invalid);
    }
    Ok(NativeEmbeddingsRequest {
        model: model_id.to_string(),
        input: input.clone(),
        encoding_format: "float",
    })
}

#[derive(Serialize)]
struct NativeEmbeddingsRequest {
    model: String,
    input: Vec<String>,
    encoding_format: &'static str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingsResponse {
    object: String,
    model: String,
    data: Vec<EmbeddingDatum>,
    usage: EmbeddingUsage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingDatum {
    object: String,
    index: u32,
    embedding: Vec<f32>,
}

/// Strictly types the known chat SSE envelope while leaving unknown provider
/// metadata (including tool-call extensions we do not execute) ignored. Derive
/// deserialization rejects duplicate known fields instead of accepting the
/// last value selected by `serde_json::Value`.
#[derive(Deserialize)]
struct NativeChatChunk {
    #[serde(default)]
    choices: Vec<NativeChatChoice>,
    #[serde(default)]
    usage: Option<NativeChatUsage>,
    #[serde(default, deserialize_with = "deserialize_present")]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct NativeChatChoice {
    #[serde(default)]
    delta: Option<NativeChatDelta>,
}

#[derive(Deserialize)]
struct NativeChatDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct NativeChatUsage {
    #[serde(default, deserialize_with = "deserialize_present")]
    completion_tokens: Option<u64>,
}

fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Default)]
struct Output {
    content: String,
    reasoning: String,
    completion_tokens: Option<u64>,
    done_marker_seen: bool,
}

impl Output {
    fn consume_line(
        &mut self,
        line: &[u8],
        on_event: &mut impl FnMut(ChatStreamEvent<'_>) -> Result<(), ChatEventCallbackError>,
    ) -> Result<(), LanguageFailure> {
        let protocol = failure(DispatchDisposition::Uncertain, LanguageErrorKind::Protocol);
        let line = std::str::from_utf8(line)
            .map_err(|_| protocol)?
            .trim_end_matches(['\r', '\n']);
        if self.done_marker_seen {
            return if line.is_empty() {
                Ok(())
            } else {
                Err(protocol)
            };
        }
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(());
        };
        let data = data.trim();
        if data == "[DONE]" {
            self.done_marker_seen = true;
            return Ok(());
        }
        let shape: Value = serde_json::from_str(data).map_err(|_| protocol)?;
        let Some(object) = shape.as_object() else {
            return Err(protocol);
        };
        if let Some(choices) = object.get("choices") {
            let Some(choices) = choices.as_array() else {
                return Err(protocol);
            };
            for choice in choices {
                let Some(choice) = choice.as_object() else {
                    return Err(protocol);
                };
                if let Some(delta) = choice.get("delta") {
                    if !delta.is_null() && !delta.is_object() {
                        return Err(protocol);
                    }
                }
            }
        }
        if let Some(usage) = object.get("usage") {
            if !usage.is_null() && !usage.is_object() {
                return Err(protocol);
            }
        }
        let event: NativeChatChunk = serde_json::from_str(data).map_err(|_| protocol)?;
        if event.error.is_some() {
            return Err(protocol);
        }
        // Validate the complete native SSE record before publishing anything
        // from it. A valid usage field must not leak from a record whose later
        // delta field is malformed.
        let completion_tokens = event
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens);
        if completion_tokens.is_some() && self.completion_tokens.is_some() {
            return Err(protocol);
        }
        let delta = event
            .choices
            .first()
            .and_then(|choice| choice.delta.as_ref());
        let content_delta = delta.and_then(|delta| delta.content.as_deref());
        let reasoning_delta = delta.and_then(|delta| delta.reasoning_content.as_deref());
        let mut staged_output_bytes = self.content.len().saturating_add(self.reasoning.len());
        for delta in [content_delta, reasoning_delta].into_iter().flatten() {
            if !delta.is_empty() {
                staged_output_bytes = staged_output_bytes.saturating_add(delta.len());
            }
            if staged_output_bytes > MAX_OUTPUT_BYTES {
                return Err(failure(
                    DispatchDisposition::Uncertain,
                    LanguageErrorKind::TooLarge,
                ));
            }
        }
        if let Some(completion_tokens) = completion_tokens {
            self.completion_tokens = Some(completion_tokens);
            notify_chat_event(on_event, ChatStreamEvent::Usage { completion_tokens })?;
        }
        if let Some(delta) = content_delta {
            self.content.push_str(delta);
            notify_chat_event(on_event, ChatStreamEvent::Delta(ChatDelta::Content(delta)))?;
        }
        if let Some(delta) = reasoning_delta {
            self.reasoning.push_str(delta);
            notify_chat_event(
                on_event,
                ChatStreamEvent::Delta(ChatDelta::Reasoning(delta)),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{
        Device, Lane, ProfileId, ProfileKey, Readiness, RuntimeIdentity, Task,
    };
    use super::super::requests::ChatTurn;
    use super::super::routes::{resolve_profile_route, NativeRoute};
    use super::super::runtime::{Cancellation, NeverCancel};
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use std::time::Instant;

    fn catalog() -> Catalog {
        Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap()
    }

    fn embedding_catalog() -> (tempfile::TempDir, Catalog) {
        let language_root = tempfile::tempdir().unwrap();
        fs::write(
            language_root.path().join("llama-swap.yaml"),
            "models:\n  embedding-fixture:\n    cmd: |\n      llama-server --embedding\n",
        )
        .unwrap();
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let catalog = Catalog::load_from_roots(
            language_root.path(),
            repo.join("toolkits/l2m2"),
            repo.join("toolkits/l3ds"),
        )
        .unwrap();
        (language_root, catalog)
    }

    fn embedding_request() -> InferenceRequest {
        InferenceRequest::Embeddings {
            profile: ProfileKey::new(Lane::Language, "embedding-fixture").unwrap(),
            input: vec!["alpha".into(), "beta".into()],
        }
    }
    fn request() -> InferenceRequest {
        InferenceRequest::Chat {
            profile: ProfileKey::new(Lane::Language, "gemma-4-26b-qat-mtp").unwrap(),
            turns: vec![
                ChatTurn {
                    role: ChatRole::System,
                    content: "  system exactly  ".into(),
                },
                ChatTurn {
                    role: ChatRole::User,
                    content: "hello".into(),
                },
                ChatTurn {
                    role: ChatRole::Assistant,
                    content: "prior".into(),
                },
                ChatTurn {
                    role: ChatRole::User,
                    content: "next".into(),
                },
            ],
            max_tokens: 64,
        }
    }

    fn route_drift_cases(route: &ResolvedProfileRoute) -> Vec<ResolvedProfileRoute> {
        let mut cases = Vec::with_capacity(6);

        let mut drifted = route.clone();
        drifted.profile.profile_id = ProfileId::new("drifted-profile").unwrap();
        cases.push(drifted);

        let mut drifted = route.clone();
        drifted.runtime = RuntimeIdentity::new(Lane::Language, "drifted-runtime").unwrap();
        cases.push(drifted);

        let mut drifted = route.clone();
        drifted.device = Some(match route.device.as_ref() {
            Some(Device::Cpu) => Device::Cuda,
            Some(Device::Cuda | Device::Remote) | None => Device::Cpu,
            Some(Device::Other(value)) => Device::Other(format!("{value}-drifted")),
        });
        cases.push(drifted);

        let mut drifted = route.clone();
        match &mut drifted.native_route {
            NativeRoute::LlamaSwapModel { model_id, .. } => {
                *model_id = ProfileId::new("drifted-model").unwrap();
            }
            _ => panic!("language fixture must resolve to llama-swap"),
        }
        cases.push(drifted);

        let mut drifted = route.clone();
        drifted.tasks.push(Task::new("drifted-task").unwrap());
        cases.push(drifted);

        let mut drifted = route.clone();
        drifted.readiness = match route.readiness {
            Readiness::ManifestPresent => Readiness::Installed,
            Readiness::Installed => Readiness::Healthy,
            Readiness::Healthy => Readiness::ManifestPresent,
        };
        cases.push(drifted);

        cases
    }

    fn assert_listener_has_no_connection(listener: &TcpListener) {
        match listener.accept() {
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Ok(_) => panic!("route drift dispatched an HTTP request"),
            Err(error) => panic!("fixture listener failed: {error}"),
        }
    }

    #[test]
    fn native_translation_preserves_role_order_and_text_without_extra_system_prompt() {
        let native = native_request(&catalog(), &request()).unwrap();
        let payload = native.payload();
        assert_eq!(payload["model"], "gemma-4-26b-qat-mtp");
        assert_eq!(payload["max_tokens"], 64);
        assert_eq!(
            payload["messages"],
            serde_json::json!([
                {"role":"system", "content":"  system exactly  "},
                {"role":"user", "content":"hello"},
                {"role":"assistant", "content":"prior"},
                {"role":"user", "content":"next"}
            ])
        );
    }

    #[test]
    fn chat_route_drift_is_rejected_before_http_dispatch() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let catalog = catalog();
        let request = request();
        let accepted = resolve_profile_route(&catalog, request.profile()).unwrap();
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let context = OperationContext::new(Instant::now() + Duration::from_secs(5), &NeverCancel);

        for drifted in route_drift_cases(&accepted) {
            let error = adapter
                .execute_with_events_for_route(&catalog, &request, &drifted, &context, |_| {
                    panic!("callback must not run when the route is stale")
                })
                .err()
                .expect("stale accepted route must fail");
            assert_eq!(
                error,
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::InvalidRequest,
                )
            );
            assert_listener_has_no_connection(&listener);
        }
    }

    #[test]
    fn embeddings_route_drift_is_rejected_before_http_dispatch() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (_root, catalog) = embedding_catalog();
        let request = embedding_request();
        let accepted = resolve_profile_route(&catalog, request.profile()).unwrap();
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let context = OperationContext::new(Instant::now() + Duration::from_secs(5), &NeverCancel);

        for drifted in route_drift_cases(&accepted) {
            let error = adapter
                .execute_embeddings_for_route(&catalog, &request, &drifted, &context)
                .err()
                .expect("stale accepted route must fail");
            assert_eq!(
                error,
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::InvalidRequest,
                )
            );
            assert_listener_has_no_connection(&listener);
        }
    }

    #[test]
    fn route_bound_embeddings_refreshes_manifest_before_any_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (root, catalog) = embedding_catalog();
        let request = embedding_request();
        let accepted = resolve_profile_route(&catalog, request.profile()).unwrap();
        // The cached catalog still declares embeddings. Only an actual refresh
        // observes that the native manifest now declares a chat-only model.
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  embedding-fixture:\n    cmd: llama-server\n",
        )
        .unwrap();
        assert!(request.validate(&catalog).is_ok());
        let error = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_embeddings_for_route(
                &catalog,
                &request,
                &accepted,
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
            )
            .err()
            .expect("changed native manifest must not dispatch");
        assert_eq!(
            error,
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest
            )
        );
        assert_listener_has_no_connection(&listener);
    }

    #[test]
    fn stream_parser_keeps_reasoning_separate_and_bounds_combined_output() {
        let mut output = Output::default();
        let mut deltas = Vec::new();
        let mut usage = Vec::new();
        let mut callback = |event: ChatStreamEvent<'_>| {
            match event {
                ChatStreamEvent::Delta(ChatDelta::Content(text)) => {
                    deltas.push(("content", text.to_owned()))
                }
                ChatStreamEvent::Delta(ChatDelta::Reasoning(text)) => {
                    deltas.push(("reasoning", text.to_owned()))
                }
                ChatStreamEvent::Usage { completion_tokens } => usage.push(completion_tokens),
                ChatStreamEvent::StreamComplete => unreachable!("completion is deferred to EOF"),
            }
            Ok(())
        };
        output
            .consume_line(
                b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n",
                &mut callback,
            )
            .unwrap();
        output.consume_line(b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}],\"usage\":{\"completion_tokens\":7}}\r\n", &mut callback).unwrap();
        output
            .consume_line(b"data: [DONE]\n", &mut callback)
            .unwrap();
        assert_eq!(output.content, "answer");
        assert_eq!(output.reasoning, "think");
        assert_eq!(output.completion_tokens, Some(7));
        assert_eq!(usage, [7]);
        assert!(output.done_marker_seen);
        assert_eq!(
            deltas,
            [("reasoning", "think".into()), ("content", "answer".into())]
        );
        let mut bounded = Output {
            content: "x".repeat(MAX_OUTPUT_BYTES),
            ..Output::default()
        };
        assert_eq!(
            bounded
                .consume_line(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n",
                    &mut |_| Ok(())
                )
                .unwrap_err()
                .kind,
            LanguageErrorKind::TooLarge
        );
    }

    #[test]
    fn malformed_events_and_cancelled_callbacks_fail_closed_without_body_echo() {
        for line in [
            b"data: secret-malformed\n".as_slice(),
            b"data: []\n",
            b"data: {\"error\":\"sk-secret-response\"}\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":42}}]}\n",
        ] {
            let error = Output::default()
                .consume_line(line, &mut |_| Ok(()))
                .unwrap_err();
            assert_eq!(error.kind, LanguageErrorKind::Protocol);
            assert!(!error.to_string().contains("secret"));
        }
        let error = Output::default()
            .consume_line(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n",
                &mut |_| Err(ChatEventCallbackError::Cancelled),
            )
            .err()
            .unwrap();
        assert_eq!(error.disposition, DispatchDisposition::Uncertain);
        assert_eq!(error.kind, LanguageErrorKind::Cancelled);
    }

    #[test]
    fn one_malformed_sse_record_does_not_publish_earlier_usage_fields() {
        let mut output = Output::default();
        let mut published = Vec::new();
        let error = output
            .consume_line(
                b"data: {\"usage\":{\"completion_tokens\":9},\"choices\":[{\"delta\":{\"content\":42}}]}\n",
                &mut |event| {
                    published.push(match event {
                        ChatStreamEvent::Usage { .. } => "usage",
                        ChatStreamEvent::Delta(_) => "delta",
                        ChatStreamEvent::StreamComplete => "complete",
                    });
                    Ok(())
                },
            )
            .err()
            .unwrap();
        assert_eq!(error.kind, LanguageErrorKind::Protocol);
        assert!(published.is_empty());
        assert_eq!(output.completion_tokens, None);
        assert!(output.content.is_empty());
    }

    #[test]
    fn typed_chat_sse_rejects_wrong_type_and_duplicate_known_fields() {
        for line in [
            br#"data: []"#.as_slice(),
            br#"data: {"usage":42,"choices":[]}"#.as_slice(),
            br#"data: {"usage":[],"choices":[]}"#,
            br#"data: {"choices":42}"#,
            br#"data: {"choices":[[]]}"#,
            br#"data: {"choices":[{"delta":false}]}"#,
            br#"data: {"choices":[{"delta":[]}] }"#,
            br#"data: {"choices":[{"delta":{"content":42}}]}"#,
            br#"data: {"usage":{"completion_tokens":"many"},"choices":[]}"#,
            br#"data: {"choices":[],"choices":[]}"#,
            br#"data: {"usage":{"completion_tokens":1},"usage":{"completion_tokens":2},"choices":[]}"#,
            br#"data: {"usage":{"completion_tokens":1,"completion_tokens":2},"choices":[]}"#,
            br#"data: {"choices":[{"delta":{"content":"a","content":"b"}}]}"#,
        ] {
            let mut output = Output::default();
            let mut published = 0;
            let input = [line, b"\n"].concat();
            let error = output
                .consume_line(&input, &mut |_| {
                    published += 1;
                    Ok(())
                })
                .err()
                .unwrap();
            assert_eq!(error.disposition, DispatchDisposition::Uncertain);
            assert_eq!(error.kind, LanguageErrorKind::Protocol);
            assert_eq!(published, 0);
            assert!(output.content.is_empty());
            assert!(output.reasoning.is_empty());
            assert_eq!(output.completion_tokens, None);
        }

        let mut usage_only = Output::default();
        let mut usage_seen = None;
        usage_only
            .consume_line(
                b"data: {\"usage\":{\"completion_tokens\":5},\"choices\":[]}\n",
                &mut |event| {
                    if let ChatStreamEvent::Usage { completion_tokens } = event {
                        usage_seen = Some(completion_tokens);
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(usage_seen, Some(5));

        let mut extension = Output::default();
        extension
            .consume_line(
                br#"data: {"provider_metadata":{"x":1},"choices":[{"delta":{"tool_calls":[{"id":"fixture"}],"content":"kept"}}]}"#,
                &mut |_| Ok(()),
            )
            .unwrap();
        assert_eq!(extension.content, "kept");
    }

    #[test]
    fn malformed_native_container_after_a_delta_is_uncertain_and_not_complete() {
        let mut output = Output::default();
        let mut delta_count = 0;
        let mut complete_count = 0;
        let mut callback = |event: ChatStreamEvent<'_>| {
            match event {
                ChatStreamEvent::Delta(_) => delta_count += 1,
                ChatStreamEvent::Usage { .. } => {}
                ChatStreamEvent::StreamComplete => complete_count += 1,
            }
            Ok(())
        };
        output
            .consume_line(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n",
                &mut callback,
            )
            .unwrap();
        let error = output
            .consume_line(b"data: {\"choices\":[[]]}\n", &mut callback)
            .err()
            .unwrap();
        assert_eq!(error.disposition, DispatchDisposition::Uncertain);
        assert_eq!(error.kind, LanguageErrorKind::Protocol);
        assert_eq!(delta_count, 1);
        assert_eq!(complete_count, 0);
        assert_eq!(output.content, "first");
        assert!(!output.done_marker_seen);
    }

    #[test]
    fn credentials_and_invalid_configuration_do_not_enter_errors() {
        for endpoint in [
            "https://user:password@example.com",
            "http://example.com?token=secret",
            "secret-invalid-url",
        ] {
            let error = LanguageAdapter::new(endpoint, Some("sk-secret"))
                .err()
                .unwrap();
            assert_eq!(error.kind, LanguageErrorKind::Configuration);
            assert!(!error.to_string().contains("secret"));
            assert!(!error.to_string().contains("password"));
        }
        let error = LanguageAdapter::new("http://localhost:1", Some("secret\ninvalid"))
            .err()
            .unwrap();
        assert_eq!(error.disposition, DispatchDisposition::NoDispatch);
    }

    struct Flag(Arc<AtomicBool>);
    impl Cancellation for Flag {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[test]
    fn cancelled_or_expired_work_never_dispatches() {
        let adapter = LanguageAdapter::new("http://127.0.0.1:1", None).unwrap();
        let cancelled = Flag(Arc::new(AtomicBool::new(true)));
        let error = adapter
            .execute(
                &catalog(),
                &request(),
                &OperationContext::new(Instant::now() + Duration::from_secs(1), &cancelled),
                |_| true,
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Cancelled
            )
        );
        let error = adapter
            .execute(
                &catalog(),
                &request(),
                &OperationContext::new(Instant::now(), &NeverCancel),
                |_| true,
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            failure(DispatchDisposition::NoDispatch, LanguageErrorKind::Deadline)
        );
    }

    fn read_request(stream: &mut impl Read) -> Value {
        read_http_json(stream, "/v1/chat/completions", 1024 * 1024)
    }

    fn read_embeddings_request(stream: &mut impl Read) -> Value {
        read_http_json(stream, "/v1/embeddings", 1024 * 1024 + 4096)
    }

    fn read_http_json(stream: &mut impl Read, path: &str, max_bytes: usize) -> Value {
        let mut header = Vec::new();
        let mut byte = [0];
        while !header.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            header.push(byte[0]);
            assert!(header.len() < 16384);
        }
        let header = String::from_utf8(header).unwrap();
        assert!(header.contains(&format!("POST {path} HTTP/1.1")));
        let length = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        assert!(length <= max_bytes);
        let mut body = vec![0; length];
        stream.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn run_embedding_fixture(
        body: String,
        content_type: &str,
        status: &str,
    ) -> Result<InferenceResult, LanguageFailure> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let content_type = content_type.to_owned();
        let status = status.to_owned();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_embeddings_request(&mut stream);
            assert_eq!(payload["model"], "embedding-fixture");
            assert_eq!(payload["input"], serde_json::json!(["alpha", "beta"]));
            assert_eq!(payload["encoding_format"], "float");
            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
        });
        let (_root, catalog) = embedding_catalog();
        let result = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_embeddings(
                &catalog,
                &embedding_request(),
                &OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel),
            );
        server.join().unwrap();
        result
    }

    #[test]
    fn embeddings_posts_one_exact_batch_and_accepts_zero_native_usage() {
        let body = serde_json::json!({
            "object": "list",
            "model": "server-model-name",
            "data": [
                {"object":"embedding", "index":0, "embedding":[0.25, -0.5]},
                {"object":"embedding", "index":1, "embedding":[1.0, 0.0]}
            ],
            "usage": {"prompt_tokens":0, "total_tokens":0}
        })
        .to_string();
        let result =
            run_embedding_fixture(body, "application/json; charset=utf-8", "200 OK").unwrap();
        let InferenceResult::Embeddings { vectors, usage } = result else {
            panic!("wrong result shape")
        };
        assert_eq!(vectors, [vec![0.25, -0.5], vec![1.0, 0.0]]);
        assert_eq!(
            usage,
            EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            }
        );
    }

    #[test]
    fn malformed_embedding_responses_fail_closed_without_echo() {
        let valid_rows = r#""data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}]"#;
        let malformed = [
            format!(r#"{{"object":"list","object":"list","model":"m",{valid_rows},"usage":{{"prompt_tokens":1,"total_tokens":1}}}}"#),
            r#"{"object":"list","model":"m","data":[{"object":"embedding","index":1,"embedding":[0.1]},{"object":"embedding","index":0,"embedding":[0.2]}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#.into(),
            r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2,0.3]}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#.into(),
            r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}]}"#.into(),
            r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}],"usage":{"prompt_tokens":1,"total_tokens":2}}"#.into(),
            r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[NaN]},{"object":"embedding","index":1,"embedding":[0.2]}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#.into(),
            r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}],"usage":{"prompt_tokens":1,"total_tokens":1},"private":"secret-body"}"#.into(),
        ];
        for body in malformed {
            let error = run_embedding_fixture(body, "application/json", "200 OK")
                .err()
                .unwrap();
            assert_eq!(error.disposition, DispatchDisposition::Uncertain);
            assert_eq!(error.kind, LanguageErrorKind::Protocol);
            assert!(!error.to_string().contains("secret-body"));
        }
        let error = run_embedding_fixture(
            r#"{"object":"list","model":"m","data":[],"usage":{"prompt_tokens":0,"total_tokens":0}}"#.into(),
            "text/plain",
            "200 OK",
        )
        .err()
        .unwrap();
        assert_eq!(error.kind, LanguageErrorKind::Protocol);
        let error = run_embedding_fixture(
            r#"{"error":"secret-body"}"#.into(),
            "application/json",
            "400 Bad Request",
        )
        .err()
        .unwrap();
        assert_eq!(error.kind, LanguageErrorKind::Rejected);
        assert!(!error.to_string().contains("secret-body"));
    }

    #[test]
    fn embedding_response_limit_is_checked_before_unbounded_body_growth() {
        let oversized = "x".repeat(MAX_EMBEDDING_WIRE_BYTES + 1);
        let error = run_embedding_fixture(oversized, "application/json", "200 OK")
            .err()
            .unwrap();
        assert_eq!(error.disposition, DispatchDisposition::Uncertain);
        assert_eq!(error.kind, LanguageErrorKind::TooLarge);
    }

    #[test]
    fn embedding_transport_does_not_retry_after_connection_loss() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "embedding request not received");
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture accept failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let _ = read_embeddings_request(&mut stream);
            drop(stream);
            let retry_deadline = Instant::now() + Duration::from_millis(100);
            while Instant::now() < retry_deadline {
                match listener.accept() {
                    Ok(_) => panic!("embedding adapter retried a dispatched request"),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture accept failed"),
                }
            }
        });
        let (_root, catalog) = embedding_catalog();
        let error = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_embeddings(
                &catalog,
                &embedding_request(),
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
            )
            .err()
            .unwrap();
        assert_eq!(error.disposition, DispatchDisposition::Uncertain);
        server.join().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn private_socket_adapter_preserves_native_http_and_streams_content() {
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory
            .path()
            .join(super::super::engine_transport::ENGINE_SOCKET_NAME);
        let listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let adapter = LanguageAdapter::new_private_engine(directory.path(), None).unwrap();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "private request not received");
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => panic!("private fixture accept failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let payload = read_request(&mut stream);
            assert_eq!(payload["messages"][0]["content"], "  system exactly  ");
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"socket-answer\"}}]}\n\ndata: [DONE]\n\n";
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        });
        let result = adapter
            .execute(
                &catalog(),
                &request(),
                &OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel),
                |_| true,
            )
            .unwrap();
        let InferenceResult::Chat {
            content, reasoning, ..
        } = result
        else {
            panic!("wrong result lane")
        };
        assert_eq!(content, "socket-answer");
        assert!(reasoning.is_empty());
        server.join().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn private_socket_embeddings_use_the_same_native_endpoint_contract() {
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::net::UnixListener;

        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory
            .path()
            .join(super::super::engine_transport::ENGINE_SOCKET_NAME);
        let listener = UnixListener::bind(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let adapter = LanguageAdapter::new_private_engine(directory.path(), None).unwrap();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "private embeddings request not received"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => panic!("private fixture accept failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let payload = read_embeddings_request(&mut stream);
            assert_eq!(payload["input"], serde_json::json!(["alpha", "beta"]));
            let body = r#"{"object":"list","model":"fixture","data":[{"object":"embedding","index":0,"embedding":[0.1]},{"object":"embedding","index":1,"embedding":[0.2]}],"usage":{"prompt_tokens":0,"total_tokens":0}}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        });
        let (_root, catalog) = embedding_catalog();
        let request = embedding_request();
        let accepted_route = resolve_profile_route(&catalog, request.profile()).unwrap();
        let result = adapter
            .execute_embeddings_for_route(
                &catalog,
                &request,
                &accepted_route,
                &OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel),
            )
            .unwrap();
        assert!(matches!(result, InferenceResult::Embeddings { .. }));
        server.join().unwrap();
    }

    #[test]
    fn actual_http_adapter_preserves_native_request_and_accepts_reasoning_only_completion() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let catalog = catalog();
        let request = request();
        let accepted_route = resolve_profile_route(&catalog, request.profile()).unwrap();
        let server = std::thread::spawn(move || {
            let accept_deadline = Instant::now() + Duration::from_secs(2);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < accept_deadline, "request not received");
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture listener failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let payload = read_request(&mut stream);
            assert_eq!(payload["messages"][0]["content"], "  system exactly  ");
            let body = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"reason only\"}}]}\n\ndata: {\"usage\":{\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
        });
        let result = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_with_events_for_route(
                &catalog,
                &request,
                &accepted_route,
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
                |_| Ok(()),
            )
            .unwrap();
        let InferenceResult::Chat {
            content,
            reasoning,
            completion_tokens,
        } = result
        else {
            panic!("wrong result lane")
        };
        assert!(content.is_empty());
        assert_eq!(reasoning, "reason only");
        assert_eq!(completion_tokens, Some(9));
        server.join().unwrap();
    }

    #[test]
    fn chat_events_are_observable_before_server_sends_done() {
        use std::sync::mpsc;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (release_tx, release_rx) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let _ = read_request(&mut stream);
            let prefix = "data: {\"choices\":[{\"delta\":{\"content\":\"early\"}}]}\n\n";
            let suffix = "data: {\"usage\":{\"completion_tokens\":4}}\n\ndata: [DONE]\n\n";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                prefix.len() + suffix.len(),
                prefix
            )
            .unwrap();
            stream.flush().unwrap();
            release_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("test must observe a delta before releasing [DONE]");
            stream.write_all(suffix.as_bytes()).unwrap();
        });

        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let (event_tx, event_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            adapter.execute_with_events(
                &catalog(),
                &request(),
                &OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel),
                |event| {
                    let label = match event {
                        ChatStreamEvent::Delta(ChatDelta::Content(text)) => {
                            format!("delta:{text}")
                        }
                        ChatStreamEvent::Delta(ChatDelta::Reasoning(_)) => "reasoning".into(),
                        ChatStreamEvent::Usage { .. } => "usage".into(),
                        ChatStreamEvent::StreamComplete => "stream_complete".into(),
                    };
                    event_tx.send(label).unwrap();
                    Ok(())
                },
            )
        });
        assert_eq!(
            event_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            "delta:early"
        );
        release_tx.send(()).unwrap();
        assert!(matches!(
            worker.join().unwrap(),
            Ok(InferenceResult::Chat { .. })
        ));
        assert_eq!(
            event_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "usage"
        );
        assert_eq!(
            event_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "stream_complete"
        );
        server.join().unwrap();
    }

    #[test]
    fn reserved_chat_bridge_publishes_incrementally_without_terminalizing_job() {
        use super::super::catalog::Lane;
        use super::super::result_buffer::{Availability, ResultBuffer, MAX_PAGE_BYTES};
        use super::super::result_consumer::{NativeOutput, ResultConsumer};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let accept_deadline = Instant::now() + Duration::from_secs(2);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < accept_deadline, "request not received");
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture listener failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let _ = read_request(&mut stream);
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thought\"}}],\"usage\":{\"completion_tokens\":11}}\n\ndata: [DONE]\n\n";
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
        });
        let id = "stream-bridge";
        let mut buffer = ResultBuffer::new();
        buffer.begin_reserved(id, Lane::Language).unwrap();
        let result = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_chat_into_result_buffer(
                &catalog(),
                &request(),
                id,
                &mut buffer,
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
            )
            .unwrap();
        assert!(matches!(
            result,
            InferenceResult::Chat {
                completion_tokens: Some(11),
                ..
            }
        ));
        let page = buffer.page(id, 0, MAX_PAGE_BYTES).unwrap();
        assert_eq!(page.availability, Availability::Available);
        assert_eq!(page.terminal, None);
        let mut consumer = ResultConsumer::new(id, Lane::Language).unwrap();
        consumer.consume(&page).unwrap();
        assert_eq!(consumer.chat_completion_tokens(), Some(11));
        assert!(consumer.chat_stream_completed());
        assert_eq!(consumer.terminal(), None);
        assert!(!consumer.is_complete());
        assert_eq!(
            serde_json::to_string(consumer.output()).unwrap(),
            r#"{"lane":"language","content":"answer","reasoning":"thought"}"#
        );
        assert!(matches!(
            consumer.output(),
            NativeOutput::Language { content, reasoning }
                if content == "answer" && reasoning == "thought"
        ));
        server.join().unwrap();
    }

    #[test]
    fn failure_after_delta_preserves_partial_result_without_done_or_retry() {
        use super::super::catalog::Lane;
        use super::super::result_buffer::{PageEvent, ResultBuffer, MAX_PAGE_BYTES};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let accept_deadline = Instant::now() + Duration::from_secs(2);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < accept_deadline, "request not received");
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture listener failed"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let _ = read_request(&mut stream);
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: [DONE]\n\ndata: malformed-secret\n\n";
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
            stream.flush().unwrap();
            drop(stream);
            let retry_deadline = Instant::now() + Duration::from_millis(100);
            while Instant::now() < retry_deadline {
                match listener.accept() {
                    Ok(_) => panic!("chat adapter retried a dispatched request"),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture listener failed"),
                }
            }
        });
        let id = "late-sse-failure";
        let mut buffer = ResultBuffer::new();
        buffer.begin_reserved(id, Lane::Language).unwrap();
        let error = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_chat_into_result_buffer(
                &catalog(),
                &request(),
                id,
                &mut buffer,
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
            )
            .err()
            .unwrap();
        assert_eq!(error.disposition, DispatchDisposition::Uncertain);
        assert_eq!(error.kind, LanguageErrorKind::Protocol);
        assert!(!error.to_string().contains("malformed-secret"));
        let page = buffer.page(id, 0, MAX_PAGE_BYTES).unwrap();
        assert_eq!(page.terminal, None);
        assert!(page.events.iter().any(|item| matches!(
            &item.event,
            PageEvent::ChatContent { text } if text == "partial"
        )));
        assert!(!page
            .events
            .iter()
            .any(|item| matches!(&item.event, PageEvent::ChatStreamComplete)));
        server.join().unwrap();
    }

    #[test]
    fn cancellation_and_deadline_after_delta_keep_partial_chat_unterminalized() {
        use super::super::catalog::Lane;
        use super::super::result_buffer::{PageEvent, ResultBuffer, MAX_PAGE_BYTES};

        for cancel in [true, false] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let flag = Arc::new(AtomicBool::new(false));
            let server_flag = Arc::clone(&flag);
            let server = std::thread::spawn(move || {
                let accept_deadline = Instant::now() + Duration::from_secs(2);
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(accepted) => break accepted,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < accept_deadline, "request not received");
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(_) => panic!("fixture listener failed"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let _ = read_request(&mut stream);
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {{\"choices\":[{{\"delta\":{{\"content\":\"partial\"}}}}]}}\n\n").unwrap();
                stream.flush().unwrap();
                if cancel {
                    std::thread::sleep(Duration::from_millis(200));
                    server_flag.store(true, Ordering::Release);
                }
                std::thread::sleep(Duration::from_millis(450));
                drop(stream);
                let retry_deadline = Instant::now() + Duration::from_millis(100);
                while Instant::now() < retry_deadline {
                    match listener.accept() {
                        Ok(_) => panic!("chat adapter retried a dispatched request"),
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(_) => panic!("fixture listener failed"),
                    }
                }
            });
            let id = if cancel {
                "cancel-stream"
            } else {
                "deadline-stream"
            };
            let mut buffer = ResultBuffer::new();
            buffer.begin_reserved(id, Lane::Language).unwrap();
            let cancellation = Flag(flag);
            let deadline = Instant::now()
                + if cancel {
                    Duration::from_secs(2)
                } else {
                    Duration::from_millis(300)
                };
            let error = LanguageAdapter::new(&endpoint, None)
                .unwrap()
                .execute_chat_into_result_buffer(
                    &catalog(),
                    &request(),
                    id,
                    &mut buffer,
                    &OperationContext::new(deadline, &cancellation),
                )
                .err()
                .unwrap();
            assert_eq!(error.disposition, DispatchDisposition::Uncertain);
            assert_eq!(
                error.kind,
                if cancel {
                    LanguageErrorKind::Cancelled
                } else {
                    LanguageErrorKind::Deadline
                }
            );
            let page = buffer.page(id, 0, MAX_PAGE_BYTES).unwrap();
            assert_eq!(page.terminal, None);
            assert!(page.events.iter().any(|item| matches!(
                &item.event,
                PageEvent::ChatContent { text } if text == "partial"
            )));
            assert!(!page
                .events
                .iter()
                .any(|item| matches!(&item.event, PageEvent::ChatStreamComplete)));
            server.join().unwrap();
        }
    }

    #[test]
    fn cancellation_while_waiting_for_headers_is_bounded_and_uncertain() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let flag = Arc::new(AtomicBool::new(false));
        let server_flag = Arc::clone(&flag);
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let _ = read_request(&mut stream);
            server_flag.store(true, Ordering::Release);
            std::thread::sleep(Duration::from_millis(150));
        });
        let cancellation = Flag(flag);
        let started = Instant::now();
        let error = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute(
                &catalog(),
                &request(),
                &OperationContext::new(started + Duration::from_secs(2), &cancellation),
                |_| true,
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            failure(DispatchDisposition::Uncertain, LanguageErrorKind::Cancelled)
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        server.join().unwrap();
    }

    #[test]
    fn idle_stream_deadline_and_premature_eof_do_not_report_success_or_retry() {
        for idle in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let _ = read_request(&mut stream);
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {{\"choices\":[{{\"delta\":{{\"content\":\"partial\"}}}}]}}\n\n").unwrap();
                stream.flush().unwrap();
                if idle {
                    std::thread::sleep(Duration::from_millis(250));
                }
                listener.set_nonblocking(true).unwrap();
                assert!(listener.accept().is_err(), "adapter must not retry");
            });
            let error = LanguageAdapter::new(&endpoint, None)
                .unwrap()
                .execute(
                    &catalog(),
                    &request(),
                    &OperationContext::new(
                        Instant::now() + Duration::from_millis(150),
                        &NeverCancel,
                    ),
                    |_| true,
                )
                .err()
                .unwrap();
            assert_eq!(error.disposition, DispatchDisposition::Uncertain);
            assert_eq!(
                error.kind,
                if idle {
                    LanguageErrorKind::Deadline
                } else {
                    LanguageErrorKind::Protocol
                }
            );
            server.join().unwrap();
        }
    }
}
