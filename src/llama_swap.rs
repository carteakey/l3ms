//! Blocking client helpers for the llama-swap HTTP API.
//!
//! llama-swap remains the runtime source of truth for servable models. The
//! public client is deliberately synchronous so both the launcher and the TUI
//! can decide where blocking work should run.

use std::time::Duration;
use std::{collections::HashSet, env, fmt, io::Read};

use anyhow::{bail, Context, Result};
use reqwest::blocking::{Client, RequestBuilder};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Url;
use serde::Deserialize;
use serde_json::Value;
/// Lifecycle state of a model as reported by llama-swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelState {
    Loaded,
    Loading,
    Unloaded,
    Unknown,
}

impl ModelState {
    /// Parse from a normalized lowercase state string without failing.
    pub fn parse_lossy(s: &str) -> Self {
        match s {
            "loaded" => Self::Loaded,
            "loading" => Self::Loading,
            "unloaded" => Self::Unloaded,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loaded => "loaded",
            Self::Loading => "loading",
            Self::Unloaded => "unloaded",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_loaded(self) -> bool {
        matches!(self, Self::Loaded)
    }

    pub fn is_loading(self) -> bool {
        matches!(self, Self::Loading)
    }
}

impl std::str::FromStr for ModelState {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::parse_lossy(s))
    }
}

impl std::fmt::Display for ModelState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub const DEFAULT_BASE_URL: &str = "http://localhost:8080";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const LOAD_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const RESPONSE_PREVIEW_CHARS: usize = 200;
const MAX_RUNNING_RESPONSE_BYTES: usize = 256 * 1024;
const MAX_RUNNING_MODELS: usize = 128;
const MAX_LIFECYCLE_MODEL_ID_BYTES: usize = 256;
const MAX_ACTIVATION_RESPONSE_BYTES: usize = 16 * 1024;

/// A model advertised by llama-swap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapModel {
    pub id: String,
    /// One of `loaded`, `loading`, `unloaded`, or `unknown`.
    pub state: String,
    pub name: String,
    pub description: String,
    /// Unix timestamp from the `/v1/models` `created` field, when present.
    pub created: Option<u64>,
    /// Config-level disable flag. Filled from `llama-swap.yaml` by the app
    /// layer because llama-swap v247 does not expose it over `/v1/models`.
    pub disabled: bool,
    /// Size in bytes of the GGUF files backing the entry's `-m` path,
    /// resolved on disk by the app layer. `None` when unresolvable.
    pub size_bytes: Option<u64>,
}

impl SwapModel {
    /// Parse `self.state` into a typed [`ModelState`].
    pub fn model_state(&self) -> ModelState {
        ModelState::parse_lossy(&self.state)
    }
}

/// Blocking llama-swap API client.
#[derive(Clone)]
pub struct SwapClient {
    base_url: String,
    client: Client,
}

impl fmt::Debug for SwapClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SwapClient")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl SwapClient {
    /// Build a client from `LLAMA_SWAP_URL` and `LLAMA_SWAP_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let base_url = env::var("LLAMA_SWAP_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let api_key = env::var("LLAMA_SWAP_API_KEY").ok();
        Self::new(&base_url, api_key.as_deref())
    }

    /// Build a client with an explicit endpoint and optional bearer token.
    pub fn new(base_url: impl AsRef<str>, api_key: Option<&str>) -> Result<Self> {
        Self::build(base_url.as_ref(), api_key, false)
    }

    /// Resident handoff must contact its configured local router directly,
    /// never an environment-selected HTTP proxy.
    pub(crate) fn local_resident(base_url: &str, api_key: Option<&str>) -> Result<Self> {
        Self::build(base_url, api_key, true)
    }

    fn build(base_url: &str, api_key: Option<&str>, direct_local: bool) -> Result<Self> {
        let base_url = normalize_base_url(base_url)?;
        let headers = auth_headers(api_key)?;
        let mut builder = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never());
        if direct_local {
            builder = builder.no_proxy();
        }
        let client = builder
            .build()
            .context("failed to build llama-swap HTTP client")?;

        Ok(Self { base_url, client })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Return all models advertised by the OpenAI-compatible models endpoint.
    pub fn list_models(&self) -> Result<Vec<SwapModel>> {
        let url = self.endpoint("/v1/models");
        let response = self
            .with_timeout(self.client.get(&url), REQUEST_TIMEOUT)
            .send()
            .with_context(|| format!("failed to query llama-swap at {url}"))?
            .error_for_status()
            .with_context(|| format!("llama-swap rejected GET {url}"))?;
        let body = response
            .text()
            .context("failed to read llama-swap model response")?;
        parse_models(&body).context("failed to parse llama-swap model response")
    }

    /// Return the IDs whose exact `/running` process state is `ready`.
    ///
    /// This is a bounded listing, not evidence that the model is request-idle
    /// or that its GPU is quiescent. Empty listings are valid.
    pub fn ready_model_ids(&self, timeout: Duration) -> Result<Vec<String>> {
        if timeout.is_zero() {
            bail!("llama-swap readiness timeout must be nonzero");
        }
        let response = self
            .with_timeout(self.client.get(self.endpoint("/running")), timeout)
            .send()
            .map_err(|_| anyhow::anyhow!("llama-swap running request failed"))?;
        if response.status().as_u16() != 200 {
            bail!("llama-swap running request returned an unsuccessful status");
        }
        let body = read_limited_body(
            response,
            MAX_RUNNING_RESPONSE_BYTES,
            "llama-swap running response is invalid or too large",
        )?;
        parse_ready_model_ids(&body)
    }

    /// Make one side-effectful routed health GET to request model activation.
    ///
    /// This operation is never retried. A 2xx response is only an
    /// acknowledgment; it does not establish readiness or request-idle state.
    pub fn activate_model_once(&self, model_id: &str, timeout: Duration) -> Result<()> {
        if timeout.is_zero() {
            bail!("llama-swap activation timeout must be nonzero");
        }
        if !valid_lifecycle_model_id(model_id) {
            bail!("llama-swap activation model ID is invalid");
        }
        let url = model_endpoint(&self.base_url, "upstream", model_id, Some("health"))
            .map_err(|_| anyhow::anyhow!("llama-swap activation route is invalid"))?;
        let response = self
            .with_timeout(self.client.get(url), timeout.min(LOAD_TIMEOUT))
            .send()
            .map_err(|_| anyhow::anyhow!("llama-swap activation request failed"))?;
        if !response.status().is_success() {
            bail!("llama-swap activation was not acknowledged");
        }
        read_limited_body(
            response,
            MAX_ACTIVATION_RESPONSE_BYTES,
            "llama-swap activation response is invalid or too large",
        )?;
        Ok(())
    }

    /// Make one side-effectful request to unload exactly one selected model.
    ///
    /// This uses llama-swap's model-scoped route, never the unsuffixed route
    /// that unloads every model. A 2xx response is only an acknowledgment.
    pub fn deactivate_model_once(&self, model_id: &str, timeout: Duration) -> Result<()> {
        if timeout.is_zero() {
            bail!("llama-swap deactivation timeout must be nonzero");
        }
        if !valid_lifecycle_model_id(model_id) {
            bail!("llama-swap deactivation model ID is invalid");
        }
        let url = model_endpoint(&self.base_url, "api/models/unload", model_id, None)
            .map_err(|_| anyhow::anyhow!("llama-swap deactivation route is invalid"))?;
        let response = self
            .with_timeout(self.client.post(url), timeout.min(REQUEST_TIMEOUT))
            .send()
            .map_err(|_| anyhow::anyhow!("llama-swap deactivation request failed"))?;
        if !response.status().is_success() {
            bail!("llama-swap deactivation was not acknowledged");
        }
        read_limited_body(
            response,
            MAX_ACTIVATION_RESPONSE_BYTES,
            "llama-swap deactivation response is invalid or too large",
        )?;
        Ok(())
    }

    /// Ask llama-swap to load `model_id` and return a compact HTTP summary.
    ///
    /// One routed native `/health` request triggers on-demand startup for a
    /// llama-server-compatible backend. This is side-effectful, not an idle
    /// probe; failure must not be retried automatically. An acknowledgment
    /// does not establish request-idle or continued model readiness.
    pub fn load_model(&self, model_id: &str) -> Result<String> {
        let url = model_endpoint(&self.base_url, "upstream", model_id, Some("health"))?;
        let response = self
            .with_timeout(self.client.get(url.clone()), LOAD_TIMEOUT)
            .send()
            .with_context(|| format!("failed to GET {url}"))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .with_context(|| format!("failed to read response from {url}"))?;
        action_result(status, &body)
    }

    /// Ask llama-swap to unload `model_id` and return a compact HTTP summary.
    pub fn unload_model(&self, model_id: &str) -> Result<String> {
        // The unsuffixed endpoint unloads ALL models and ignores a JSON model
        // body. Always address the selected model in the native route instead.
        let url = model_endpoint(&self.base_url, "api/models/unload", model_id, None)?;
        let response = self
            .with_timeout(self.client.post(url.clone()), REQUEST_TIMEOUT)
            .send()
            .with_context(|| format!("failed to POST {url}"))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .with_context(|| format!("failed to read response from {url}"))?;
        action_result(status, &body)
    }

    /// Check whether the configured llama-swap endpoint is reachable and healthy.
    pub fn probe(&self) -> Result<()> {
        let url = self.endpoint("/v1/models");
        self.with_timeout(self.client.get(&url), REQUEST_TIMEOUT)
            .send()
            .with_context(|| format!("failed to reach llama-swap at {url}"))?
            .error_for_status()
            .with_context(|| format!("llama-swap health probe failed at {url}"))?;
        Ok(())
    }

    fn endpoint(&self, path: &str) -> String {
        endpoint_url(&self.base_url, path)
    }

    fn with_timeout(&self, request: RequestBuilder, timeout: Duration) -> RequestBuilder {
        request.timeout(timeout)
    }
}

fn model_endpoint(
    base_url: &str,
    route: &str,
    model_id: &str,
    suffix: Option<&str>,
) -> Result<Url> {
    let model_id = model_id.trim();
    if model_id.is_empty() || matches!(model_id, "." | "..") {
        bail!("model id is invalid");
    }
    let mut url = Url::parse(&endpoint_url(base_url, route))?;
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| anyhow::anyhow!("model endpoint is invalid"))?;
    segments.push(model_id);
    if let Some(suffix) = suffix {
        segments.push(suffix);
    }
    drop(segments);
    Ok(url)
}

#[derive(Deserialize)]
struct ReadyModelsResponse {
    running: Vec<ReadyModel>,
}

#[derive(Deserialize)]
struct ReadyModel {
    model: String,
    state: String,
}

fn parse_ready_model_ids(body: &[u8]) -> Result<Vec<String>> {
    if body.len() > MAX_RUNNING_RESPONSE_BYTES {
        bail!("llama-swap running response is invalid or too large");
    }
    let response: ReadyModelsResponse = serde_json::from_slice(body)
        .map_err(|_| anyhow::anyhow!("llama-swap running response is malformed"))?;
    if response.running.len() > MAX_RUNNING_MODELS {
        bail!("llama-swap running response has too many models");
    }

    let mut ids = Vec::with_capacity(response.running.len());
    let mut unique = HashSet::with_capacity(response.running.len());
    for model in response.running {
        if model.state != "ready" {
            bail!("llama-swap model readiness is not confirmed");
        }
        if !valid_lifecycle_model_id(&model.model) {
            bail!("llama-swap running response contains an invalid model ID");
        }
        if !unique.insert(model.model.clone()) {
            bail!("llama-swap running response contains duplicate model IDs");
        }
        ids.push(model.model);
    }
    ids.sort();
    Ok(ids)
}

fn valid_lifecycle_model_id(model_id: &str) -> bool {
    model_id.len() <= MAX_LIFECYCLE_MODEL_ID_BYTES
        && !model_id.trim().is_empty()
        && model_id.trim() == model_id
        && model_endpoint(DEFAULT_BASE_URL, "upstream", model_id, Some("health")).is_ok()
}

fn read_limited_body(
    response: reqwest::blocking::Response,
    maximum_bytes: usize,
    failure_message: &'static str,
) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    response
        .take(maximum_bytes as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|_| anyhow::anyhow!("{}", failure_message))?;
    if body.len() > maximum_bytes {
        bail!("{}", failure_message);
    }
    Ok(body)
}

pub(crate) fn normalize_base_url(raw: &str) -> Result<String> {
    let raw = raw.trim();
    let raw = if raw.is_empty() {
        DEFAULT_BASE_URL
    } else {
        raw
    };
    let normalized = raw.trim_end_matches('/');
    let parsed =
        Url::parse(normalized).with_context(|| format!("invalid llama-swap URL {normalized:?}"))?;

    if !matches!(parsed.scheme(), "http" | "https") {
        bail!("llama-swap URL must use http or https");
    }
    if parsed.cannot_be_a_base() || parsed.host_str().is_none() {
        bail!("llama-swap URL must be an absolute HTTP URL");
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        bail!("llama-swap URL cannot contain a query or fragment");
    }

    Ok(normalized.to_owned())
}

fn endpoint_url(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

pub(crate) fn auth_headers(api_key: Option<&str>) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let Some(api_key) = api_key.map(str::trim).filter(|key| !key.is_empty()) else {
        return Ok(headers);
    };

    let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
        .context("LLAMA_SWAP_API_KEY contains invalid header characters")?;
    headers.insert(AUTHORIZATION, value);
    Ok(headers)
}

pub(crate) fn parse_models(body: &str) -> Result<Vec<SwapModel>> {
    let payload: Value = serde_json::from_str(body)?;
    let entries = payload
        .get("data")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();

    let mut models = Vec::new();
    for entry in entries {
        let Some(entry) = entry.as_object() else {
            continue;
        };
        let Some(id) = entry
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            continue;
        };

        models.push(SwapModel {
            id: id.to_owned(),
            state: normalize_state(entry).to_owned(),
            name: text_field(entry, "name"),
            description: text_field(entry, "description"),
            created: entry.get("created").and_then(Value::as_u64),
            disabled: false,
            size_bytes: None,
        });
    }

    models.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(models)
}

fn text_field(entry: &serde_json::Map<String, Value>, key: &str) -> String {
    match entry.get(key) {
        Some(Value::String(value)) => value.clone(),
        Some(value) if !value.is_null() => value.to_string(),
        _ => String::new(),
    }
}

fn normalize_state(entry: &serde_json::Map<String, Value>) -> &'static str {
    let raw = entry
        .get("status")
        .and_then(|status| status.get("value"))
        .and_then(Value::as_str)
        .or_else(|| entry.get("state").and_then(Value::as_str))
        .or_else(|| entry.get("status").and_then(Value::as_str))
        .map(str::trim)
        .filter(|state| !state.is_empty());

    if let Some(raw) = raw {
        let lower = raw.to_ascii_lowercase();
        return match lower.as_str() {
            "loaded" | "running" | "ready" | "active" | "online" => "loaded",
            "loading" | "starting" | "pending" | "unloading" => "loading",
            "unloaded" | "stopped" | "idle" | "inactive" | "offline" => "unloaded",
            _ => "unknown",
        };
    }

    match entry.get("loaded").and_then(Value::as_bool) {
        Some(true) => "loaded",
        Some(false) => "unloaded",
        None => "unknown",
    }
}

fn response_summary(status: u16, body: &str) -> String {
    let preview: String = body.trim().chars().take(RESPONSE_PREVIEW_CHARS).collect();
    if preview.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status} {preview}")
    }
}

fn action_result(status: u16, body: &str) -> Result<String> {
    let summary = response_summary(status, body);
    if !(200..300).contains(&status) {
        bail!(summary);
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::thread::JoinHandle;

    fn lifecycle_http_fixture(
        status: &str,
        headers: &[&str],
        body: Vec<u8>,
        watch_for_extra_request: bool,
    ) -> (SwapClient, JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .expect("loopback TCP binding is required for lifecycle HTTP tests");
        listener
            .set_nonblocking(true)
            .expect("set lifecycle listener nonblocking");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let status = status.to_owned();
        let headers = headers
            .iter()
            .map(|header| (*header).to_owned())
            .collect::<Vec<_>>();
        let worker = std::thread::spawn(move || {
            let mut stream = accept_lifecycle_connection(&listener);
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("set lifecycle request timeout");
            let mut requests = vec![read_http_request(&mut stream)];

            let mut response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
                body.len()
            );
            for header in headers {
                response.push_str(&header);
                response.push_str("\r\n");
            }
            response.push_str("\r\n");
            stream
                .write_all(response.as_bytes())
                .and_then(|()| stream.write_all(&body))
                .expect("write lifecycle HTTP response");

            if watch_for_extra_request {
                listener
                    .set_nonblocking(true)
                    .expect("watch for retried lifecycle request");
                let deadline = std::time::Instant::now() + Duration::from_millis(100);
                while std::time::Instant::now() < deadline {
                    match listener.accept() {
                        Ok((mut retry, _)) => {
                            retry
                                .set_read_timeout(Some(Duration::from_secs(2)))
                                .expect("set retried request timeout");
                            requests.push(read_http_request(&mut retry));
                        }
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                || error.kind() == std::io::ErrorKind::Interrupted =>
                        {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("accept retried request: {error}"),
                    }
                }
            }
            requests
        });
        let client = SwapClient::local_resident(&endpoint, Some("fixture-token")).unwrap();
        (client, worker)
    }

    fn accept_lifecycle_connection(listener: &TcpListener) -> TcpStream {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return stream,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        || error.kind() == std::io::ErrorKind::Interrupted =>
                {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "timed out waiting for lifecycle HTTP request"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept lifecycle HTTP request: {error}"),
            }
        }
    }

    fn read_http_request(stream: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut chunk = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = std::io::Read::read(stream, &mut chunk).expect("read HTTP request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
        }
        String::from_utf8_lossy(&request).into_owned()
    }

    fn request_line(request: &str) -> &str {
        request.lines().next().expect("HTTP request line")
    }

    #[test]
    fn lifecycle_readiness_uses_authenticated_running_route_and_sorts_ids() {
        let body =
            br#"{"running":[{"model":"zeta","state":"ready"},{"model":"alpha","state":"ready"}]}"#
                .to_vec();
        let (client, server) = lifecycle_http_fixture("200 OK", &[], body, false);

        assert_eq!(
            client.ready_model_ids(Duration::from_secs(1)).unwrap(),
            ["alpha".to_owned(), "zeta".to_owned()]
        );
        let request = server.join().unwrap();
        assert_eq!(request.len(), 1);
        assert_eq!(request_line(&request[0]), "GET /running HTTP/1.1");
        assert!(request[0]
            .lines()
            .any(|header| header.eq_ignore_ascii_case("authorization: Bearer fixture-token")));
    }

    #[test]
    fn lifecycle_model_actions_use_encoded_scoped_routes_once() {
        let id = "model/with?reserved%chars#";
        let (client, server) = lifecycle_http_fixture("204 No Content", &[], Vec::new(), false);
        client
            .deactivate_model_once(id, Duration::from_secs(1))
            .unwrap();
        let request = server.join().unwrap();
        assert_eq!(request.len(), 1);
        assert_eq!(
            request_line(&request[0]),
            "POST /api/models/unload/model%2Fwith%3Freserved%25chars%23 HTTP/1.1"
        );
        assert_ne!(
            request_line(&request[0]),
            "POST /api/models/unload HTTP/1.1"
        );

        let (client, server) = lifecycle_http_fixture("200 OK", &[], Vec::new(), false);
        client
            .activate_model_once(id, Duration::from_secs(1))
            .unwrap();
        let request = server.join().unwrap();
        assert_eq!(request.len(), 1);
        assert_eq!(
            request_line(&request[0]),
            "GET /upstream/model%2Fwith%3Freserved%25chars%23/health HTTP/1.1"
        );
    }

    #[test]
    fn lifecycle_activation_does_not_retry_errors_or_follow_redirects() {
        for (status, headers) in [
            ("500 Internal Server Error", &[][..]),
            ("302 Found", &["Location: /redirect-target"][..]),
        ] {
            let (client, server) =
                lifecycle_http_fixture(status, headers, b"fixture response".to_vec(), true);
            let error = client
                .activate_model_once("model", Duration::from_secs(1))
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "llama-swap activation was not acknowledged"
            );
            let requests = server.join().unwrap();
            assert_eq!(requests.len(), 1, "status={status}");
            assert_eq!(
                request_line(&requests[0]),
                "GET /upstream/model/health HTTP/1.1"
            );
        }
    }

    #[test]
    fn lifecycle_activation_rejects_oversized_response_body() {
        let body = vec![b'x'; MAX_ACTIVATION_RESPONSE_BYTES + 1];
        let (client, server) = lifecycle_http_fixture("200 OK", &[], body, false);
        let error = client
            .activate_model_once("model", Duration::from_secs(1))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "llama-swap activation response is invalid or too large"
        );
        assert_eq!(server.join().unwrap().len(), 1);
    }

    #[test]
    fn native_model_routes_preserve_prefix_and_encode_selected_id() {
        let base = "http://example.test:8080/router";
        let model = " model/with?reserved%chars# ";
        let load = model_endpoint(base, "upstream", model, Some("health")).unwrap();
        assert_eq!(
            load.as_str(),
            "http://example.test:8080/router/upstream/model%2Fwith%3Freserved%25chars%23/health"
        );
        let unload = model_endpoint(base, "api/models/unload", model, None).unwrap();
        assert_eq!(
            unload.as_str(),
            "http://example.test:8080/router/api/models/unload/model%2Fwith%3Freserved%25chars%23"
        );
        assert!(load.query().is_none() && load.fragment().is_none());
        assert!(unload.query().is_none() && unload.fragment().is_none());
    }

    #[test]
    fn model_actions_cannot_fall_back_to_all_models_or_dot_routes() {
        for model in ["", " ", ".", "..", " .. "] {
            assert!(model_endpoint(DEFAULT_BASE_URL, "api/models/unload", model, None).is_err());
            assert!(model_endpoint(DEFAULT_BASE_URL, "upstream", model, Some("health")).is_err());
        }
    }

    #[test]
    fn normalizes_base_urls_and_builds_endpoints() {
        assert_eq!(
            normalize_base_url("  http://example.test:8080/// ").unwrap(),
            "http://example.test:8080"
        );
        assert_eq!(normalize_base_url(" ").unwrap(), DEFAULT_BASE_URL);
        assert_eq!(
            endpoint_url("http://example.test:8080/", "/v1/models"),
            "http://example.test:8080/v1/models"
        );
        assert!(normalize_base_url("file:///tmp/socket").is_err());
        assert!(normalize_base_url("localhost:8080").is_err());
    }

    #[test]
    fn builds_optional_bearer_header() {
        assert!(auth_headers(None).unwrap().is_empty());
        assert!(auth_headers(Some("   ")).unwrap().is_empty());

        let headers = auth_headers(Some("  secret-token  ")).unwrap();
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer secret-token");
        assert!(auth_headers(Some("bad\nkey")).is_err());
    }

    #[test]
    fn parses_sorts_and_normalizes_models() {
        let body = r#"{
            "data": [
                {"id":"zeta", "status":"RUNNING", "name":"Zeta"},
                {"id":"alpha", "state":"starting", "description":"warming"},
                {"id":"idle", "loaded":false},
                {"id":"mystery", "state":"surprising"},
                {"id":"nested", "status":{"value":"loaded"}},
                {"id":"nested-idle", "status":{"value":"unloaded"}},
                {"id":"timed", "created":1788318354},
                {"id":"", "loaded":true},
                null
            ]
        }"#;

        let models = parse_models(body).unwrap();
        assert_eq!(
            models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            [
                "alpha",
                "idle",
                "mystery",
                "nested",
                "nested-idle",
                "timed",
                "zeta"
            ]
        );
        assert_eq!(models[0].state, "loading");
        assert_eq!(models[1].state, "unloaded");
        assert_eq!(models[2].state, "unknown");
        assert_eq!(models[3].state, "loaded");
        assert_eq!(models[4].state, "unloaded");
        assert_eq!(models[5].created, Some(1_788_318_354));
        assert_eq!(models[6].name, "Zeta");
    }

    #[test]
    fn tolerates_missing_or_non_array_model_data() {
        assert!(parse_models("{}").unwrap().is_empty());
        assert!(parse_models(r#"{"data":"not-an-array"}"#)
            .unwrap()
            .is_empty());
        assert!(parse_models("not json").is_err());
    }

    #[test]
    fn ready_model_parser_requires_exact_ready_state_and_sorts_ids() {
        let body = br#"{
            "running": [
                {"model":"zeta", "state":"ready", "cmd":"/private/path"},
                {"model":"alpha", "state":"ready", "args":["--secret"]}
            ],
            "router_extra": true
        }"#;
        assert_eq!(
            parse_ready_model_ids(body).unwrap(),
            ["alpha".to_owned(), "zeta".to_owned()]
        );
        assert!(parse_ready_model_ids(br#"{"running":[]}"#)
            .unwrap()
            .is_empty());

        for state in ["starting", "stopping", "unloaded", "unknown", "READY"] {
            let body = format!(r#"{{"running":[{{"model":"model","state":"{state}"}}]}}"#);
            assert!(
                parse_ready_model_ids(body.as_bytes()).is_err(),
                "state={state}"
            );
        }
    }

    #[test]
    fn ready_model_parser_rejects_malformed_duplicate_and_unsafe_rows() {
        for body in [
            br#"{}"#.as_slice(),
            br#"{"running":"not-an-array"}"#.as_slice(),
            br#"{"running":[{"model":"missing-state"}]}"#.as_slice(),
            br#"{"running":[{"model":"x","model":"y","state":"ready"}]}"#.as_slice(),
            br#"{"running":[{"model":"x","state":"ready","state":"starting"}]}"#.as_slice(),
            br#"{"running":[],"running":[]}"#.as_slice(),
            b"not json".as_slice(),
        ] {
            assert!(parse_ready_model_ids(body).is_err());
        }

        assert!(parse_ready_model_ids(
            br#"{"running":[{"model":"same","state":"ready"},{"model":"same","state":"ready"}]}"#
        )
        .is_err());
        for id in ["", " ", ".", "..", " padded "] {
            let body = format!(r#"{{"running":[{{"model":{id:?},"state":"ready"}}]}}"#);
            assert!(parse_ready_model_ids(body.as_bytes()).is_err(), "id={id:?}");
        }
        let too_long = "m".repeat(MAX_LIFECYCLE_MODEL_ID_BYTES + 1);
        let body = format!(r#"{{"running":[{{"model":{too_long:?},"state":"ready"}}]}}"#);
        assert!(parse_ready_model_ids(body.as_bytes()).is_err());

        let maximum = "m".repeat(MAX_LIFECYCLE_MODEL_ID_BYTES);
        let body = format!(r#"{{"running":[{{"model":{maximum:?},"state":"ready"}}]}}"#);
        assert_eq!(parse_ready_model_ids(body.as_bytes()).unwrap(), [maximum]);
    }

    #[test]
    fn ready_model_parser_bounds_model_count_and_response_size() {
        let rows = (0..=MAX_RUNNING_MODELS)
            .map(|index| format!(r#"{{"model":"model-{index:03}","state":"ready"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let body = format!(r#"{{"running":[{rows}]}}"#);
        assert!(parse_ready_model_ids(body.as_bytes()).is_err());

        let large_unknown = "x".repeat(MAX_RUNNING_RESPONSE_BYTES);
        let body = format!(r#"{{"running":[],"extra":{large_unknown:?}}}"#);
        assert!(parse_ready_model_ids(body.as_bytes()).is_err());
    }

    #[test]
    fn lifecycle_calls_reject_zero_timeout_and_invalid_ids_without_echoing_input() {
        let client = SwapClient::new("http://127.0.0.1:1", None).unwrap();
        let readiness = client.ready_model_ids(Duration::ZERO).unwrap_err();
        assert_eq!(
            readiness.to_string(),
            "llama-swap readiness timeout must be nonzero"
        );
        let activation = client
            .activate_model_once("private-model-id", Duration::ZERO)
            .unwrap_err();
        assert_eq!(
            activation.to_string(),
            "llama-swap activation timeout must be nonzero"
        );
        let deactivation = client
            .deactivate_model_once("private-model-id", Duration::ZERO)
            .unwrap_err();
        assert_eq!(
            deactivation.to_string(),
            "llama-swap deactivation timeout must be nonzero"
        );
        let invalid_id = client
            .activate_model_once("..", Duration::from_secs(1))
            .unwrap_err();
        assert_eq!(
            invalid_id.to_string(),
            "llama-swap activation model ID is invalid"
        );
        assert!(!invalid_id.to_string().contains(".."));
        let invalid_deactivation = client
            .deactivate_model_once("..", Duration::from_secs(1))
            .unwrap_err();
        assert_eq!(
            invalid_deactivation.to_string(),
            "llama-swap deactivation model ID is invalid"
        );
        assert!(!invalid_deactivation.to_string().contains(".."));
    }

    #[test]
    fn summarizes_utf8_response_safely() {
        let body = "🦀".repeat(RESPONSE_PREVIEW_CHARS + 10);
        let summary = response_summary(200, &body);
        assert_eq!(
            summary
                .chars()
                .filter(|character| *character == '🦀')
                .count(),
            200
        );
    }

    #[test]
    fn rejects_unsuccessful_model_actions_with_bounded_body() {
        assert_eq!(action_result(204, "").unwrap(), "HTTP 204");

        let body = "denied ".repeat(100);
        let error = action_result(503, &body).unwrap_err().to_string();
        assert!(error.starts_with("HTTP 503 denied"));
        assert_eq!(error.chars().count(), "HTTP 503 ".chars().count() + 200);
    }
}
