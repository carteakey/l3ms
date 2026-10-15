//! Bounded embedding and decision requests over existing llama-swap HTTP.
//!
//! The caller supplies the authorized endpoint and request. Model lifecycle
//! belongs to the router. This adapter never acquires/relinquishes GPU ownership,
//! stops a runtime, changes configured endpoints, or retries a request. Once
//! HTTP dispatch starts, any failure is conservatively uncertain: dropping a
//! connection does not acknowledge server-side cancellation or GPU release.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::catalog::{Catalog, Lane, ProfileKey};
use super::decision::{decode_systemone_response, NativeDecisionResult};
use super::requests::{
    EmbeddingUsage, InferenceRequest, InferenceResult, SpeechInput, SpeechMode,
    MAX_EMBEDDING_INPUTS, MAX_EMBEDDING_JSON_BYTES,
};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::runtime::OperationContext;
use crate::llama_swap::{auth_headers, normalize_base_url};

const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
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

/// Caller-configured connection; neither endpoint discovery nor credential
/// lookup happens here. Does not implement Debug, to avoid leaking credentials.
pub struct LanguageAdapter {
    endpoint: String,
    api_key: Option<String>,
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

    /// Run one bounded local llama.cpp SystemOne decision request. The caller
    /// owns lifecycle admission; this method only revalidates the catalog,
    /// posts once to the configured loopback llama-swap route, and validates
    /// the native answer. A failed dispatched attempt is never retried.
    pub fn execute_decisions(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        context: &OperationContext<'_>,
    ) -> Result<NativeDecisionResult, LanguageFailure> {
        check_context(context, DispatchDisposition::NoDispatch)?;
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Configuration,
            ));
        }
        let accepted_model_id = native_systemone_model(catalog, request)?;
        let accepted_route = resolve_profile_route(catalog, request.profile()).map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            )
        })?;
        let catalog = catalog.refresh().map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            )
        })?;
        let model_id = native_systemone_model(&catalog, request)?;
        let refreshed_route = resolve_profile_route(&catalog, request.profile()).map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            )
        })?;
        if accepted_route != refreshed_route || accepted_model_id != model_id {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            ));
        }
        let url = self.systemone_url()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?;
        runtime.block_on(self.execute_decisions_async(&url, &model_id, request, context))
    }

    async fn execute_decisions_async(
        &self,
        url: &reqwest::Url,
        model_id: &str,
        request: &InferenceRequest,
        context: &OperationContext<'_>,
    ) -> Result<NativeDecisionResult, LanguageFailure> {
        let InferenceRequest::Decision {
            state,
            questions,
            images,
            ..
        } = request
        else {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            ));
        };
        check_context(context, DispatchDisposition::NoDispatch)?;
        let native_request = NativeSystemOneRequest {
            model: model_id,
            state,
            questions,
            images: (!images.is_empty()).then_some(images.as_slice()),
        };
        let body = serde_json::to_vec(&native_request).map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            )
        })?;
        if body.len() > MAX_OUTPUT_BYTES {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            ));
        }
        let client = self.http_client_without_proxy(context)?;
        let http = client
            .post(url.clone())
            .header(reqwest::header::ACCEPT, "application/json")
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body);
        let body = send_json_body(http, context, MAX_OUTPUT_BYTES).await?;
        decode_systemone_response(request, &body)
            .map_err(|_| failure(DispatchDisposition::Uncertain, LanguageErrorKind::Protocol))
    }

    /// Run one Breeze speech request through llama-swap's `/audiocpp/` route
    /// (reserved model `audiocpp_auto`). llama-swap owns starting, swapping and
    /// stopping the engine; this only posts once and returns the audio. A
    /// dispatched request is never retried.
    pub fn execute_speech(
        &self,
        input: &SpeechInput,
        context: &OperationContext<'_>,
    ) -> Result<SpeechAudio, LanguageFailure> {
        check_context(context, DispatchDisposition::NoDispatch)?;
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Configuration,
            ));
        }
        let url = self.audiocpp_speech_url()?;
        let body = speech_request_body(input)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| {
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::Configuration,
                )
            })?;
        runtime.block_on(async {
            let client = self.http_client_without_proxy(context)?;
            let streaming = input.mode == SpeechMode::Streaming;
            let http = client
                .post(url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .header(
                    reqwest::header::ACCEPT,
                    if streaming {
                        "text/event-stream"
                    } else {
                        "audio/wav"
                    },
                )
                .body(body);
            let mime = if streaming {
                "text/event-stream"
            } else {
                "audio/wav"
            };
            let bytes = send_body(http, context, MAX_SPEECH_BYTES, mime).await?;
            if streaming {
                parse_speech_stream(&bytes)
            } else {
                Ok(SpeechAudio {
                    wave: bytes,
                    chunks: Vec::new(),
                })
            }
        })
    }

    fn audiocpp_speech_url(&self) -> Result<reqwest::Url, LanguageFailure> {
        let configuration = failure(
            DispatchDisposition::NoDispatch,
            LanguageErrorKind::Configuration,
        );
        let mut url = reqwest::Url::parse(&self.endpoint).map_err(|_| configuration)?;
        if !is_loopback_url(&url) {
            return Err(configuration);
        }
        let mut segments = url.path_segments_mut().map_err(|_| configuration)?;
        segments
            .pop_if_empty()
            .extend(["audiocpp", "v1", "audio", "speech"]);
        drop(segments);
        Ok(url)
    }

    fn systemone_url(&self) -> Result<reqwest::Url, LanguageFailure> {
        let configuration = failure(
            DispatchDisposition::NoDispatch,
            LanguageErrorKind::Configuration,
        );
        let mut url = reqwest::Url::parse(&self.endpoint).map_err(|_| configuration)?;
        if !is_loopback_url(&url) {
            return Err(configuration);
        }
        let mut segments = url.path_segments_mut().map_err(|_| configuration)?;
        segments.pop_if_empty().push("v1").push("systemone");
        drop(segments);
        Ok(url)
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
        let body = send_json_body(http, context, MAX_EMBEDDING_WIRE_BYTES).await?;
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

    fn http_client(
        &self,
        context: &OperationContext<'_>,
    ) -> Result<reqwest::Client, LanguageFailure> {
        self.build_http_client(context, false)
    }

    fn http_client_without_proxy(
        &self,
        context: &OperationContext<'_>,
    ) -> Result<reqwest::Client, LanguageFailure> {
        self.build_http_client(context, true)
    }

    fn build_http_client(
        &self,
        context: &OperationContext<'_>,
        no_proxy: bool,
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
        let builder = if no_proxy {
            builder.no_proxy()
        } else {
            builder
        };
        builder.build().map_err(|_| {
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Configuration,
            )
        })
    }
}

async fn send_json_body(
    http: reqwest::RequestBuilder,
    context: &OperationContext<'_>,
    maximum_bytes: usize,
) -> Result<Vec<u8>, LanguageFailure> {
    send_body(http, context, maximum_bytes, "application/json").await
}

/// One bounded request/response exchange that requires an exact response MIME
/// type. Never retries; cancellation and the deadline are polled throughout.
async fn send_body(
    http: reqwest::RequestBuilder,
    context: &OperationContext<'_>,
    maximum_bytes: usize,
    expected_mime: &str,
) -> Result<Vec<u8>, LanguageFailure> {
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
                .is_some_and(|mime| mime.trim().eq_ignore_ascii_case(expected_mime))
        })
    {
        return Err(failure(
            DispatchDisposition::Uncertain,
            LanguageErrorKind::Protocol,
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
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
            match tokio::time::timeout(context.remaining().min(POLL_INTERVAL), &mut chunk).await {
                Err(_) => continue,
                Ok(result) => break result.map_err(transport_failure)?,
            }
        };
        let Some(bytes) = bytes else {
            break;
        };
        if bytes.len() > maximum_bytes.saturating_sub(body.len()) {
            return Err(failure(
                DispatchDisposition::Uncertain,
                LanguageErrorKind::TooLarge,
            ));
        }
        body.extend_from_slice(&bytes);
    }
    check_context(context, DispatchDisposition::Uncertain)?;
    Ok(body)
}

fn native_systemone_model(
    catalog: &Catalog,
    request: &InferenceRequest,
) -> Result<String, LanguageFailure> {
    let invalid = failure(
        DispatchDisposition::NoDispatch,
        LanguageErrorKind::InvalidRequest,
    );
    if !matches!(request, InferenceRequest::Decision { .. }) {
        return Err(invalid);
    }
    request.validate(catalog).map_err(|_| invalid)?;
    let route = resolve_profile_route(catalog, request.profile()).map_err(|_| invalid)?;
    let NativeRoute::DecisionRunner {
        model,
        profile_type,
        ..
    } = &route.native_route
    else {
        return Err(invalid);
    };
    if route.profile.lane != Lane::Decisions
        || route.runtime.lane != Lane::Decisions
        || route.runtime.as_str() != "llamacpp"
        || profile_type != "local"
    {
        return Err(invalid);
    }
    let language_key = ProfileKey::new(Lane::Language, model.clone()).map_err(|_| invalid)?;
    let router_model = catalog
        .inspect(&language_key)
        .filter(|profile| {
            !profile.disabled && profile.tasks.iter().any(|task| task.as_str() == "decision")
        })
        .ok_or(invalid)?;
    let NativeRoute::LlamaSwapModel { model_id, .. } =
        resolve_profile_route(catalog, &router_model.key)
            .map_err(|_| invalid)?
            .native_route
    else {
        return Err(invalid);
    };
    Ok(model_id.to_string())
}

/// Audio returned by one routed speech request. `wave` is a complete WAV file;
/// `chunks` holds one WAV per streamed event (empty unless streaming).
pub struct SpeechAudio {
    pub wave: Vec<u8>,
    pub chunks: Vec<Vec<u8>>,
}

const MAX_SPEECH_BYTES: usize = 128 * 1024 * 1024;
/// Breeze-TTS-2 streams raw signed 16-bit little-endian mono PCM at 24 kHz.
const SPEECH_SAMPLE_RATE: u32 = 24_000;

/// Server model ids registered in `toolkits/l2m2/audiocpp-server.json`.
fn speech_model_id(mode: SpeechMode) -> &'static str {
    match mode {
        SpeechMode::Standard | SpeechMode::VoiceDesign => "breeze-tts",
        SpeechMode::VoiceClone => "breeze-clone",
        SpeechMode::Streaming => "breeze-stream",
    }
}

fn speech_request_body(input: &SpeechInput) -> Result<Vec<u8>, LanguageFailure> {
    let invalid = failure(
        DispatchDisposition::NoDispatch,
        LanguageErrorKind::InvalidRequest,
    );
    let mut body = serde_json::json!({
        "model": speech_model_id(input.mode),
        "input": input.text,
    });
    let object = body.as_object_mut().ok_or(invalid)?;
    let mut options = serde_json::Map::new();
    if let Some(instruction) = input.instruction.as_deref().filter(|text| !text.is_empty()) {
        options.insert("instruction".into(), instruction.into());
    }
    if input.mode == SpeechMode::Streaming {
        object.insert("stream".into(), true.into());
        object.insert("stream_format".into(), "sse".into());
        object.insert("response_format".into(), "pcm".into());
        options.insert("stream_frames_per_event".into(), "16".into());
        options.insert("stream_lookahead_margin".into(), "12".into());
    } else {
        object.insert("response_format".into(), "wav".into());
    }
    if input.mode == SpeechMode::VoiceClone {
        // The server reads the reference from the same host's filesystem.
        let reference = input.voice_reference.as_deref().ok_or(invalid)?;
        let reference = std::fs::canonicalize(reference).map_err(|_| invalid)?;
        let transcript = input
            .reference_text
            .as_deref()
            .filter(|text| !text.is_empty())
            .ok_or(invalid)?;
        object.insert(
            "voice_ref".into(),
            reference.to_str().ok_or(invalid)?.into(),
        );
        object.insert("reference_text".into(), transcript.into());
    }
    if !options.is_empty() {
        object.insert("options".into(), options.into());
    }
    serde_json::to_vec(&body).map_err(|_| invalid)
}

/// Parse a complete server-sent-event body: `speech.audio.delta` events carry
/// base64 PCM, `speech.audio.done` ends the stream, `[DONE]` is a sentinel.
fn parse_speech_stream(body: &[u8]) -> Result<SpeechAudio, LanguageFailure> {
    let protocol = failure(DispatchDisposition::Uncertain, LanguageErrorKind::Protocol);
    let text = std::str::from_utf8(body).map_err(|_| protocol)?;
    let mut pcm_chunks: Vec<Vec<u8>> = Vec::new();
    let mut done = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let event: Value = serde_json::from_str(data).map_err(|_| protocol)?;
        match event.get("type").and_then(Value::as_str) {
            Some("speech.audio.delta") => {
                let encoded = event.get("audio").and_then(Value::as_str).ok_or(protocol)?;
                let pcm = decode_base64(encoded).ok_or(protocol)?;
                if pcm.is_empty() || pcm.len() % 2 != 0 {
                    return Err(protocol);
                }
                pcm_chunks.push(pcm);
            }
            Some("speech.audio.done") => done = true,
            _ => {
                return Err(failure(
                    DispatchDisposition::Uncertain,
                    LanguageErrorKind::Rejected,
                ))
            }
        }
    }
    if !done || pcm_chunks.is_empty() {
        return Err(protocol);
    }
    let all: Vec<u8> = pcm_chunks.iter().flatten().copied().collect();
    Ok(SpeechAudio {
        wave: wav_from_pcm16(&all),
        chunks: pcm_chunks.iter().map(|pcm| wav_from_pcm16(pcm)).collect(),
    })
}

/// Canonical 44-byte-header PCM16 mono WAV.
pub(crate) fn wav_from_pcm16(pcm: &[u8]) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&SPEECH_SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SPEECH_SAMPLE_RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

/// Standard-alphabet base64 (padding optional); `None` on any invalid byte.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut buffer, mut bits) = (0u32, 0u32);
    for byte in text.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return None,
        };
        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    Some(out)
}

fn is_loopback_url(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
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

#[derive(Serialize)]
struct NativeSystemOneRequest<'a> {
    model: &'a str,
    state: &'a Value,
    questions: &'a serde_json::Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<&'a [String]>,
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

#[cfg(test)]
mod tests {
    use super::super::catalog::{
        Device, Lane, ProfileId, ProfileKey, Readiness, RuntimeIdentity, Task,
    };
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

    fn systemone_catalog() -> (tempfile::TempDir, Catalog, InferenceRequest) {
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("toolkits/l2m2");
        let decisions = root.path().join("toolkits/l3ds");
        fs::create_dir_all(&media).unwrap();
        fs::create_dir_all(&decisions).unwrap();
        fs::write(
            root.path().join("llama-swap.yaml"),
            "models:\n  router-decision_2.0:\n    cmd: llama-server\n    metadata:\n      task: decision\n",
        )
        .unwrap();
        fs::write(
            media.join("media-runtimes.json"),
            r#"{"schema_version":1,"profiles":[]}"#,
        )
        .unwrap();
        fs::write(
            decisions.join("decision-runtimes.json"),
            r#"{"version":"1.0","profiles":[{"id":"flash-local","name":"Flash fixture","engine":"llamacpp","type":"local","model":"router-decision_2.0","device":"cuda","entrypoint":"decision-models/runner.py"}]}"#,
        )
        .unwrap();
        let catalog = Catalog::load(root.path()).unwrap();
        let request = InferenceRequest::Decision {
            profile: ProfileKey::new(Lane::Decisions, "flash-local").unwrap(),
            state: serde_json::json!({"severity": "high"}),
            questions: serde_json::json!({
                "route": {"type": "choice", "criteria": {"proceed": "Proceed", "stop": "Stop"}}
            })
            .as_object()
            .unwrap()
            .clone(),
            images: Vec::new(),
        };
        (root, catalog, request)
    }

    fn embedding_request() -> InferenceRequest {
        InferenceRequest::Embeddings {
            profile: ProfileKey::new(Lane::Language, "embedding-fixture").unwrap(),
            input: vec!["alpha".into(), "beta".into()],
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
    fn credentials_and_invalid_configuration_do_not_enter_errors() {
        for endpoint in [
            "https://user:password@example.com",
            "http://example.com?token=secret",
            "http://127.0.0.1/base#secret",
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

    #[test]
    fn ipv6_loopback_preserves_base_path_without_network_access() {
        let adapter = LanguageAdapter::new("http://[::1]:8080/base/path", None).unwrap();
        let url = adapter.systemone_url().unwrap();
        assert_eq!(url.path(), "/base/path/v1/systemone");
        assert!(is_loopback_url(&url));
    }

    struct Flag(Arc<AtomicBool>);
    impl Cancellation for Flag {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[test]
    fn cancelled_or_expired_embeddings_never_dispatch() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (_root, catalog) = embedding_catalog();
        let request = embedding_request();
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let cancelled = Flag(Arc::new(AtomicBool::new(true)));
        let error = adapter
            .execute_embeddings(
                &catalog,
                &request,
                &OperationContext::new(Instant::now() + Duration::from_secs(1), &cancelled),
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
        assert_listener_has_no_connection(&listener);
        let error = adapter
            .execute_embeddings(
                &catalog,
                &request,
                &OperationContext::new(Instant::now(), &NeverCancel),
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            failure(DispatchDisposition::NoDispatch, LanguageErrorKind::Deadline)
        );
        assert_listener_has_no_connection(&listener);
    }

    fn read_embeddings_request(stream: &mut impl Read) -> Value {
        read_http_json(stream, "/v1/embeddings", 1024 * 1024 + 4096)
    }

    fn read_http_json(stream: &mut impl Read, path: &str, max_bytes: usize) -> Value {
        read_http_json_with_headers(stream, path, max_bytes).1
    }

    fn read_http_json_with_headers(
        stream: &mut impl Read,
        path: &str,
        max_bytes: usize,
    ) -> (String, Value) {
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
        (header, serde_json::from_slice(&body).unwrap())
    }

    fn read_systemone_request(stream: &mut impl Read) -> (String, Value) {
        read_http_json_with_headers(stream, "/v1/systemone", MAX_OUTPUT_BYTES)
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

    fn run_systemone_fixture(
        body: String,
        content_type: &str,
        status: &str,
    ) -> Result<super::NativeDecisionResult, LanguageFailure> {
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
            let (_, payload) = read_systemone_request(&mut stream);
            assert_eq!(payload["model"], "router-decision_2.0");
            assert_eq!(payload["state"], serde_json::json!({"severity":"high"}));
            assert_eq!(payload["questions"]["route"]["type"], "choice");
            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nLocation: /redirect-target\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.flush();
            listener.set_nonblocking(true).unwrap();
            let deadline = Instant::now() + Duration::from_millis(100);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok(_) => panic!("SystemOne adapter retried or followed a redirect"),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture accept failed"),
                }
            }
        });
        let (_root, catalog, request) = systemone_catalog();
        let result = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_decisions(
                &catalog,
                &request,
                &OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel),
            );
        server.join().unwrap();
        result
    }

    #[test]
    fn systemone_uses_one_exact_loopback_route_body_and_auth_header() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/base/path/", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let (headers, payload) = read_http_json_with_headers(
                &mut stream,
                "/base/path/v1/systemone",
                MAX_OUTPUT_BYTES,
            );
            assert!(headers
                .lines()
                .any(|line| { line.eq_ignore_ascii_case("authorization: Bearer fixture-secret") }));
            assert!(headers
                .lines()
                .any(|line| { line.eq_ignore_ascii_case("accept: application/json") }));
            assert_eq!(
                payload,
                serde_json::json!({
                    "model": "router-decision_2.0",
                    "state": {"severity":"high"},
                    "questions": {
                        "route": {"type":"choice", "criteria":{"proceed":"Proceed", "stop":"Stop"}}
                    }
                })
            );
            let response = serde_json::json!({
                "answers": {"route": {"choice":"proceed", "confidence":0.9}},
                "usage": {"total_tokens": 12},
                "latency_ms": 23.5,
                "engine": "llama.cpp"
            })
            .to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            )
            .unwrap();
            stream.flush().unwrap();
            listener.set_nonblocking(true).unwrap();
            let deadline = Instant::now() + Duration::from_millis(100);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok(_) => panic!("SystemOne adapter issued more than one POST"),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => panic!("fixture accept failed"),
                }
            }
        });
        let (_root, catalog, request) = systemone_catalog();
        let result = LanguageAdapter::new(&endpoint, Some("fixture-secret"))
            .unwrap()
            .execute_decisions(
                &catalog,
                &request,
                &OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel),
            )
            .unwrap();
        server.join().unwrap();
        assert_eq!(result.native_latency, None);
        assert_eq!(
            result.native_usage,
            Some(
                serde_json::json!({"total_tokens":12})
                    .as_object()
                    .unwrap()
                    .clone()
            )
        );
        let InferenceResult::Decision { answers } = result.result else {
            panic!("wrong result lane")
        };
        assert_eq!(answers["route"]["choice"], "proceed");
    }

    #[test]
    fn systemone_forwards_bounded_images_as_top_level_data_urls() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let (_, payload) = read_systemone_request(&mut stream);
            assert_eq!(payload["model"], "vllm-d3-flash");
            assert_eq!(payload["state"], serde_json::json!({"severity":"high"}));
            assert_eq!(
                payload["images"],
                serde_json::json!(["data:image/png;base64,iVBORw0KGgo="])
            );
            assert_eq!(payload["questions"]["route"]["type"], "choice");
            let response = serde_json::json!({
                "answers": {"route": {"choice":"proceed", "confidence":0.9}},
                "usage": {"total_tokens": 12},
                "latency_ms": 23.5,
                "engine": "llama.cpp"
            })
            .to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            )
            .unwrap();
            stream.flush().unwrap();
        });
        let catalog = catalog();
        let request = InferenceRequest::Decision {
            profile: ProfileKey::new(Lane::Decisions, "vllm-d3-flash").unwrap(),
            state: serde_json::json!({"severity":"high"}),
            questions: serde_json::json!({
                "route": {"type":"choice", "criteria":{"proceed":"Proceed", "stop":"Stop"}}
            })
            .as_object()
            .unwrap()
            .clone(),
            images: vec!["data:image/png;base64,iVBORw0KGgo=".into()],
        };
        let result = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_decisions(
                &catalog,
                &request,
                &OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel),
            )
            .unwrap();
        server.join().unwrap();
        let InferenceResult::Decision { answers } = result.result else {
            panic!("wrong result lane")
        };
        assert_eq!(answers["route"]["choice"], "proceed");
    }

    #[test]
    fn invalid_systemone_routes_and_non_loopback_endpoint_never_dispatch() {
        let (root, catalog, request) = systemone_catalog();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let adapter = LanguageAdapter::new(&endpoint, None).unwrap();
        let manifest_path = root.path().join("toolkits/l3ds/decision-runtimes.json");
        let yaml_path = root.path().join("llama-swap.yaml");
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        let yaml = fs::read_to_string(&yaml_path).unwrap();
        let cases = [
            (manifest.replace("llamacpp", "nagi"), yaml.clone()),
            (
                manifest.replace("\"type\":\"local\"", "\"type\":\"api\""),
                yaml.clone(),
            ),
            (
                manifest.replace("router-decision_2.0", "missing-router-model"),
                yaml.clone(),
            ),
            (
                manifest.clone(),
                yaml.replace("      task: decision", "      task: chat"),
            ),
            (
                manifest.clone(),
                yaml.replace(
                    "  router-decision_2.0:\n",
                    "  router-decision_2.0:\n    disabled: true\n",
                ),
            ),
        ];
        for (bad_manifest, bad_yaml) in cases {
            fs::write(&manifest_path, bad_manifest).unwrap();
            fs::write(&yaml_path, bad_yaml).unwrap();
            let error = adapter
                .execute_decisions(
                    &catalog,
                    &request,
                    &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
                )
                .err()
                .expect("changed or invalid route must fail before dispatch");
            assert_eq!(
                error,
                failure(
                    DispatchDisposition::NoDispatch,
                    LanguageErrorKind::InvalidRequest,
                )
            );
            assert_listener_has_no_connection(&listener);
        }

        fs::write(&manifest_path, manifest.clone()).unwrap();
        fs::write(&yaml_path, yaml.clone()).unwrap();
        let drifted_manifest = fs::read_to_string(&manifest_path)
            .unwrap()
            .replace("router-decision_2.0", "router-other_2.0");
        let drifted_yaml = fs::read_to_string(&yaml_path)
            .unwrap()
            .replace(
                "models:\n",
                "models:\n  router-other_2.0:\n    cmd: llama-server\n    metadata:\n      task: decision\n",
            );
        fs::write(&manifest_path, drifted_manifest).unwrap();
        fs::write(&yaml_path, drifted_yaml).unwrap();
        let error = adapter
            .execute_decisions(
                &catalog,
                &request,
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::InvalidRequest,
            )
        );
        assert_listener_has_no_connection(&listener);

        fs::write(&manifest_path, manifest).unwrap();
        fs::write(&yaml_path, yaml).unwrap();
        let remote = LanguageAdapter::new("http://example.com/base", None).unwrap();
        let error = remote
            .execute_decisions(
                &catalog,
                &request,
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel),
            )
            .err()
            .unwrap();
        assert_eq!(
            error,
            failure(
                DispatchDisposition::NoDispatch,
                LanguageErrorKind::Configuration,
            )
        );
        assert_listener_has_no_connection(&listener);
    }

    #[test]
    fn systemone_protocol_rejection_question_validation_redirect_and_size_are_uncertain() {
        for (body, content_type, status, kind) in [
            (
                "not-json".to_owned(),
                "application/json",
                "200 OK",
                LanguageErrorKind::Protocol,
            ),
            (
                r#"{"answers":{"route":{"choice":"undeclared"}}}"#.to_owned(),
                "application/json",
                "200 OK",
                LanguageErrorKind::Protocol,
            ),
            (
                r#"{"answers":{"route":{"choice":"proceed"}}}"#.to_owned(),
                "text/plain",
                "200 OK",
                LanguageErrorKind::Protocol,
            ),
            (
                r#"{"answers":{"route":{"choice":"proceed"}}}"#.to_owned(),
                "application/json",
                "302 Found",
                LanguageErrorKind::Rejected,
            ),
            (
                "x".repeat(MAX_OUTPUT_BYTES + 1),
                "application/json",
                "200 OK",
                LanguageErrorKind::TooLarge,
            ),
        ] {
            let error = run_systemone_fixture(body, content_type, status)
                .err()
                .expect("invalid response must fail");
            assert_eq!(error.disposition, DispatchDisposition::Uncertain);
            assert_eq!(error.kind, kind);
        }
    }

    #[test]
    fn cancellation_after_systemone_dispatch_is_uncertain_and_not_retried() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let flag = Arc::new(AtomicBool::new(false));
        let server_flag = Arc::clone(&flag);
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let _ = read_systemone_request(&mut stream);
            server_flag.store(true, Ordering::Release);
            std::thread::sleep(Duration::from_millis(150));
            listener.set_nonblocking(true).unwrap();
            assert!(matches!(
                listener.accept(),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
            ));
        });
        let (_root, catalog, request) = systemone_catalog();
        let cancellation = Flag(flag);
        let error = LanguageAdapter::new(&endpoint, None)
            .unwrap()
            .execute_decisions(
                &catalog,
                &request,
                &OperationContext::new(Instant::now() + Duration::from_secs(2), &cancellation),
            )
            .err()
            .unwrap();
        server.join().unwrap();
        assert_eq!(
            error,
            failure(DispatchDisposition::Uncertain, LanguageErrorKind::Cancelled)
        );
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

    fn speech_input(mode: SpeechMode) -> SpeechInput {
        SpeechInput {
            text: "Hello there.".into(),
            mode,
            instruction: None,
            voice_reference: None,
            reference_text: None,
        }
    }

    fn json_body(input: &SpeechInput) -> Value {
        serde_json::from_slice(&speech_request_body(input).unwrap()).unwrap()
    }

    #[test]
    fn base64_decoder_handles_padding_and_rejects_garbage() {
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbG8").unwrap(), b"hello");
        assert_eq!(decode_base64("AAECAw==").unwrap(), [0, 1, 2, 3]);
        assert_eq!(decode_base64("").unwrap(), Vec::<u8>::new());
        assert!(decode_base64("not base64!").is_none());
    }

    #[test]
    fn wav_header_describes_24khz_mono_pcm16() {
        let wav = wav_from_pcm16(&[1, 0, 2, 0, 3, 0]);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 6);
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(&wav[44..], [1, 0, 2, 0, 3, 0]);
    }

    #[test]
    fn speech_stream_builds_chunks_and_a_merged_wave() {
        // The real audio.cpp grammar: delta events, a done event, then [DONE].
        let body = concat!(
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AQACAA==\"}\n\n",
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AwAEAA==\"}\n\n",
            "data: {\"type\":\"speech.audio.done\",\"timing\":{\"ttft_ms\":1.5}}\n\n",
            "data: [DONE]\n\n",
        );
        let audio = parse_speech_stream(body.as_bytes()).unwrap();
        assert_eq!(audio.chunks.len(), 2);
        assert_eq!(&audio.chunks[0][44..], [1, 0, 2, 0]);
        assert_eq!(&audio.wave[44..], [1, 0, 2, 0, 3, 0, 4, 0]);
    }

    #[test]
    fn malformed_or_incomplete_streams_are_protocol_errors_not_audio() {
        for body in [
            "",
            "data: [DONE]\n",
            // No done event: the engine may have been cut off.
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AQAC\"}\n",
            // Odd byte count cannot be PCM16.
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AQ==\"}\ndata: {\"type\":\"speech.audio.done\"}\n",
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"%%%\"}\n",
            "data: {\"type\":\"error\"}\n",
        ] {
            assert!(parse_speech_stream(body.as_bytes()).is_err(), "{body:?}");
        }
    }

    #[test]
    fn speech_requests_select_the_declared_server_model_per_mode() {
        let standard = json_body(&speech_input(SpeechMode::Standard));
        assert_eq!(standard["model"], "breeze-tts");
        assert_eq!(standard["response_format"], "wav");
        assert!(standard.get("options").is_none() && standard.get("stream").is_none());

        let mut design = speech_input(SpeechMode::VoiceDesign);
        design.instruction = Some("Warm narrator.".into());
        let design = json_body(&design);
        assert_eq!(design["model"], "breeze-tts");
        assert_eq!(design["options"]["instruction"], "Warm narrator.");

        let streaming = json_body(&speech_input(SpeechMode::Streaming));
        assert_eq!(streaming["model"], "breeze-stream");
        assert_eq!(streaming["stream"], true);
        assert_eq!(streaming["stream_format"], "sse");
        assert_eq!(streaming["response_format"], "pcm");
        assert_eq!(streaming["options"]["stream_frames_per_event"], "16");
    }

    #[test]
    fn voice_clone_needs_an_existing_reference_and_a_transcript() {
        let directory = tempfile::tempdir().unwrap();
        let reference = directory.path().join("ref.wav");
        std::fs::write(&reference, b"RIFF").unwrap();
        let mut clone = speech_input(SpeechMode::VoiceClone);
        assert!(speech_request_body(&clone).is_err(), "missing reference");
        clone.voice_reference = Some(reference.clone());
        assert!(speech_request_body(&clone).is_err(), "missing transcript");
        clone.reference_text = Some("Transcript.".into());
        let body = json_body(&clone);
        assert_eq!(body["model"], "breeze-clone");
        assert_eq!(body["reference_text"], "Transcript.");
        assert_eq!(
            body["voice_ref"],
            reference.canonicalize().unwrap().to_str().unwrap()
        );
        clone.voice_reference = Some(directory.path().join("missing.wav"));
        assert!(
            speech_request_body(&clone).is_err(),
            "nonexistent reference"
        );
    }

    #[test]
    fn speech_url_keeps_a_reverse_proxy_prefix_and_requires_loopback() {
        let prefixed = LanguageAdapter::new("http://127.0.0.1:8080/swap", None).unwrap();
        assert_eq!(
            prefixed.audiocpp_speech_url().unwrap().as_str(),
            "http://127.0.0.1:8080/swap/audiocpp/v1/audio/speech"
        );
        let remote = LanguageAdapter::new("http://example.com", None).unwrap();
        assert!(remote.audiocpp_speech_url().is_err());
    }

    /// One-shot loopback router: checks the exact path, bearer token and Accept
    /// header, then answers with the given status line, content type and body.
    fn speech_router(
        status: &'static str,
        content_type: &'static str,
        body: Vec<u8>,
        accept: &'static str,
    ) -> (String, std::thread::JoinHandle<Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let (headers, payload) = read_http_json_with_headers(
                &mut stream,
                "/audiocpp/v1/audio/speech",
                MAX_OUTPUT_BYTES,
            );
            assert!(headers
                .lines()
                .any(|line| line.eq_ignore_ascii_case("authorization: Bearer speech-secret")));
            assert!(headers
                .lines()
                .any(|line| line.eq_ignore_ascii_case(&format!("accept: {accept}"))));
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
            stream.flush().unwrap();
            payload
        });
        (endpoint, server)
    }

    fn speech_context() -> OperationContext<'static> {
        OperationContext::new(Instant::now() + Duration::from_secs(5), &NeverCancel)
    }

    #[test]
    fn routed_speech_posts_once_and_returns_the_wave() {
        let wave = wav_from_pcm16(&[0, 0, 1, 0]);
        let (endpoint, server) = speech_router("200 OK", "audio/wav", wave.clone(), "audio/wav");
        let adapter = LanguageAdapter::new(&endpoint, Some("speech-secret")).unwrap();
        let audio = adapter
            .execute_speech(&speech_input(SpeechMode::Standard), &speech_context())
            .unwrap();
        assert_eq!(audio.wave, wave);
        assert!(audio.chunks.is_empty());
        let payload = server.join().unwrap();
        assert_eq!(payload["model"], "breeze-tts");
        assert_eq!(payload["input"], "Hello there.");
    }

    #[test]
    fn routed_streaming_speech_returns_chunks_from_the_event_stream() {
        let body = concat!(
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AQACAA==\"}\n\n",
            "data: {\"type\":\"speech.audio.done\"}\n\n",
            "data: [DONE]\n\n",
        );
        let (endpoint, server) = speech_router(
            "200 OK",
            "text/event-stream; charset=utf-8",
            body.as_bytes().to_vec(),
            "text/event-stream",
        );
        let adapter = LanguageAdapter::new(&endpoint, Some("speech-secret")).unwrap();
        let audio = adapter
            .execute_speech(&speech_input(SpeechMode::Streaming), &speech_context())
            .unwrap();
        assert_eq!(audio.chunks.len(), 1);
        assert_eq!(&audio.wave[44..], [1, 0, 2, 0]);
        assert_eq!(server.join().unwrap()["model"], "breeze-stream");
    }

    #[test]
    fn routed_speech_rejections_and_wrong_content_types_never_become_audio() {
        for (status, content_type, kind) in [
            (
                "503 Service Unavailable",
                "application/json",
                LanguageErrorKind::Rejected,
            ),
            ("200 OK", "application/json", LanguageErrorKind::Protocol),
            ("200 OK", "text/html", LanguageErrorKind::Protocol),
        ] {
            let (endpoint, server) = speech_router(
                status,
                content_type,
                b"{\"error\":\"busy\"}".to_vec(),
                "audio/wav",
            );
            let adapter = LanguageAdapter::new(&endpoint, Some("speech-secret")).unwrap();
            let error = adapter
                .execute_speech(&speech_input(SpeechMode::Standard), &speech_context())
                .err()
                .unwrap();
            assert_eq!(error.kind, kind, "{status} {content_type}");
            // The request was sent, so completion is uncertain: never retried.
            assert_eq!(error.disposition, DispatchDisposition::Uncertain);
            server.join().unwrap();
        }
    }
}
