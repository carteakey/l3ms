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
use super::routes::{resolve_profile_route, NativeRoute};
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
        let native = native_request(&catalog, request)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?;
        runtime.block_on(self.execute_async(&native, context, &mut on_delta))
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
        let native = native_embedding_request(&catalog, request)?;
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
        on_delta: &mut impl FnMut(ChatDelta<'_>) -> bool,
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
                // EOF without [DONE] is not proof of a completed backend job.
                return Err(failure(
                    DispatchDisposition::Uncertain,
                    LanguageErrorKind::Protocol,
                ));
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
                    let done = output.consume_line(&line, on_delta)?;
                    line.clear();
                    if done {
                        check_context(context, DispatchDisposition::Uncertain)?;
                        if output.content.is_empty() && output.reasoning.is_empty() {
                            return Err(failure(
                                DispatchDisposition::Uncertain,
                                LanguageErrorKind::Protocol,
                            ));
                        }
                        return Ok(InferenceResult::Chat {
                            content: output.content,
                            reasoning: output.reasoning,
                            completion_tokens: output.completion_tokens,
                        });
                    }
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

fn native_request(
    catalog: &Catalog,
    request: &InferenceRequest,
) -> Result<ChatRequest, LanguageFailure> {
    let invalid = failure(
        DispatchDisposition::NoDispatch,
        LanguageErrorKind::InvalidRequest,
    );
    request.validate(catalog).map_err(|_| invalid)?;
    let route = resolve_profile_route(catalog, request.profile()).map_err(|_| invalid)?;
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

fn native_embedding_request(
    catalog: &Catalog,
    request: &InferenceRequest,
) -> Result<NativeEmbeddingsRequest, LanguageFailure> {
    let invalid = failure(
        DispatchDisposition::NoDispatch,
        LanguageErrorKind::InvalidRequest,
    );
    request.validate(catalog).map_err(|_| invalid)?;
    let route = resolve_profile_route(catalog, request.profile()).map_err(|_| invalid)?;
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

#[derive(Default)]
struct Output {
    content: String,
    reasoning: String,
    completion_tokens: Option<u64>,
}

impl Output {
    fn consume_line(
        &mut self,
        line: &[u8],
        on_delta: &mut impl FnMut(ChatDelta<'_>) -> bool,
    ) -> Result<bool, LanguageFailure> {
        let protocol = failure(DispatchDisposition::Uncertain, LanguageErrorKind::Protocol);
        let line = std::str::from_utf8(line)
            .map_err(|_| protocol)?
            .trim_end_matches(['\r', '\n']);
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(false);
        };
        let data = data.trim();
        if data == "[DONE]" {
            return Ok(true);
        }
        let event: Value = serde_json::from_str(data).map_err(|_| protocol)?;
        if !event.is_object() {
            return Err(protocol);
        }
        if event.get("error").is_some() {
            return Err(protocol);
        }
        if let Some(tokens) = event.pointer("/usage/completion_tokens") {
            self.completion_tokens = Some(tokens.as_u64().ok_or(protocol)?);
        }
        for (field, reasoning) in [("content", false), ("reasoning_content", true)] {
            let Some(value) = event.pointer(&format!("/choices/0/delta/{field}")) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let delta = value.as_str().ok_or(protocol)?;
            if self
                .content
                .len()
                .saturating_add(self.reasoning.len())
                .saturating_add(delta.len())
                > MAX_OUTPUT_BYTES
            {
                return Err(failure(
                    DispatchDisposition::Uncertain,
                    LanguageErrorKind::TooLarge,
                ));
            }
            if !delta.is_empty() {
                if reasoning {
                    self.reasoning.push_str(delta);
                } else {
                    self.content.push_str(delta);
                }
                if !on_delta(if reasoning {
                    ChatDelta::Reasoning(delta)
                } else {
                    ChatDelta::Content(delta)
                }) {
                    return Err(failure(
                        DispatchDisposition::Uncertain,
                        LanguageErrorKind::Cancelled,
                    ));
                }
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{Lane, ProfileKey};
    use super::super::requests::ChatTurn;
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
    fn stream_parser_keeps_reasoning_separate_and_bounds_combined_output() {
        let mut output = Output::default();
        let mut deltas = Vec::new();
        let mut callback = |delta: ChatDelta<'_>| {
            match delta {
                ChatDelta::Content(text) => deltas.push(("content", text.to_owned())),
                ChatDelta::Reasoning(text) => deltas.push(("reasoning", text.to_owned())),
            };
            true
        };
        assert!(!output
            .consume_line(
                b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n",
                &mut callback
            )
            .unwrap());
        assert!(!output.consume_line(b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}],\"usage\":{\"completion_tokens\":7}}\r\n", &mut callback).unwrap());
        assert!(output
            .consume_line(b"data: [DONE]\n", &mut callback)
            .unwrap());
        assert_eq!(output.content, "answer");
        assert_eq!(output.reasoning, "think");
        assert_eq!(output.completion_tokens, Some(7));
        assert_eq!(
            deltas,
            [("reasoning", "think".into()), ("content", "answer".into())]
        );
        output.content = "x".repeat(MAX_OUTPUT_BYTES);
        assert_eq!(
            output
                .consume_line(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n",
                    &mut |_| true
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
                .consume_line(line, &mut |_| true)
                .unwrap_err();
            assert_eq!(error.kind, LanguageErrorKind::Protocol);
            assert!(!error.to_string().contains("secret"));
        }
        let error = Output::default()
            .consume_line(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n",
                &mut |_| false,
            )
            .unwrap_err();
        assert_eq!(error.disposition, DispatchDisposition::Uncertain);
        assert_eq!(error.kind, LanguageErrorKind::Cancelled);
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
        let result = adapter
            .execute_embeddings(
                &catalog,
                &embedding_request(),
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
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
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
            .execute(
                &catalog(),
                &request(),
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
                |_| true,
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
