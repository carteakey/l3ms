//! Blocking client helpers for the llama-swap HTTP API.
//!
//! llama-swap remains the runtime source of truth for servable models. The
//! public client is deliberately synchronous so both the launcher and the TUI
//! can decide where blocking work should run.

use std::env;
use std::fmt;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use reqwest::blocking::{Client, RequestBuilder};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Url;
use serde::Serialize;
use serde_json::Value;

pub const DEFAULT_BASE_URL: &str = "http://localhost:8080";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const LOAD_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const RESPONSE_PREVIEW_CHARS: usize = 200;

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
        let base_url = normalize_base_url(base_url.as_ref())?;
        let headers = auth_headers(api_key)?;
        let client = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .default_headers(headers)
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

    /// Ask llama-swap to load `model_id` and return a compact HTTP summary.
    ///
    /// `GET /upstream/{model}/load` triggers the on-demand startup path and
    /// holds the connection until the model passes its health check.
    pub fn load_model(&self, model_id: &str) -> Result<String> {
        let model_id = model_id.trim();
        if model_id.is_empty() {
            bail!("model id cannot be empty");
        }
        let url = self.endpoint(&format!("/upstream/{model_id}/load"));
        let response = self
            .with_timeout(self.client.get(&url), LOAD_TIMEOUT)
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
        self.model_action("/api/models/unload", model_id, REQUEST_TIMEOUT)
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

    fn model_action(&self, path: &str, model_id: &str, timeout: Duration) -> Result<String> {
        let model_id = model_id.trim();
        if model_id.is_empty() {
            bail!("model id cannot be empty");
        }

        let url = self.endpoint(path);
        let response = self
            .with_timeout(
                self.client
                    .post(&url)
                    .json(&ModelAction { model: model_id }),
                timeout,
            )
            .send()
            .with_context(|| format!("failed to POST {url}"))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .with_context(|| format!("failed to read response from {url}"))?;
        action_result(status, &body)
    }
}

#[derive(Serialize)]
struct ModelAction<'a> {
    model: &'a str,
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
