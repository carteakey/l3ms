use std::{
    env,
    io::{BufRead, BufReader, Read},
    path::Path,
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use reqwest::{
    blocking::{Client, Response},
    header::{ACCEPT, CONTENT_TYPE},
    StatusCode,
};
use serde_json::{json, Value};

use crate::llama_swap::{
    auth_headers, normalize_base_url, parse_models_with_non_chat_ids, SwapModel,
};

const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;
const MAX_MODEL_BODY_BYTES: usize = 1024 * 1024;
const MAX_ERROR_BODY_BYTES: u64 = 16 * 1024;
const CHAT_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const CHAT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const CHAT_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const CHAT_SSE_READ_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<(String, String)>,
    pub system_prompt: String,
    pub temperature: f64,
    pub max_tokens: u32,
    pub thinking: bool,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<(String, String)>) -> Self {
        Self {
            model: model.into(),
            messages,
            system_prompt: String::new(),
            temperature: 0.8,
            max_tokens: 2048,
            thinking: false,
        }
    }

    pub(crate) fn payload(&self) -> Value {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        if !self.system_prompt.trim().is_empty() {
            messages.push(json!({
                "role": "system",
                "content": self.system_prompt.trim(),
            }));
        }
        for (index, (role, content)) in self.messages.iter().enumerate() {
            let is_last = index + 1 == self.messages.len();
            let content = if self.thinking && is_last && role == "user" {
                format!("/think\n{content}")
            } else {
                content.clone()
            };
            messages.push(json!({"role": role, "content": content}));
        }
        json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_options": {"include_usage": true},
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    pub content: String,
    pub completion_tokens: Option<u64>,
    pub elapsed: Duration,
}

/// Models returned by `/v1/models`, retaining known non-chat output IDs so
/// native tasks can still see the full inventory while Chat filters its picker.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatModelInventory {
    pub models: Vec<SwapModel>,
    pub non_chat_model_ids: Vec<String>,
}

impl ChatCompletion {
    pub fn tokens_per_second(&self) -> Option<f64> {
        let seconds = self.elapsed.as_secs_f64();
        (seconds > 0.0)
            .then(|| self.completion_tokens.map(|tokens| tokens as f64 / seconds))
            .flatten()
    }
}

/// An authenticated OpenAI-compatible client owned by the Chat workflow.
///
/// The app keeps this client separate from the Workbench client. A draft
/// endpoint never changes this client; it is replaced only after a successful
/// probe of the requested endpoint.
#[derive(Clone)]
pub struct ChatClient {
    base_url: String,
    client: Client,
    api_key: Option<String>,
}

impl std::fmt::Debug for ChatClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatClient")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl ChatClient {
    pub fn from_env() -> Result<Self> {
        let endpoint = env::var("LLAMA_SWAP_URL")
            .unwrap_or_else(|_| crate::llama_swap::DEFAULT_BASE_URL.into());
        let api_key = env::var("LLAMA_SWAP_API_KEY").ok();
        Self::new(endpoint, api_key.as_deref())
    }

    pub fn new(endpoint: impl AsRef<str>, api_key: Option<&str>) -> Result<Self> {
        let base_url = normalize_base_url(endpoint.as_ref())?;
        let api_key = api_key
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(ToOwned::to_owned);
        let client = Client::builder()
            .connect_timeout(CHAT_CONNECT_TIMEOUT)
            .timeout(CHAT_REQUEST_TIMEOUT)
            .default_headers(auth_headers(api_key.as_deref())?)
            .build()
            .context("build chat HTTP client")?;
        Ok(Self {
            base_url,
            client,
            api_key,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Reuse this committed connection, including its captured credentials,
    /// for explicit embedding requests. Construction performs no network I/O.
    #[cfg(target_os = "linux")]
    pub(crate) fn embedding_adapter(
        &self,
    ) -> Result<crate::control::language::LanguageAdapter, crate::control::language::LanguageFailure>
    {
        crate::control::language::LanguageAdapter::new(&self.base_url, self.api_key.as_deref())
    }

    pub fn list_models(&self) -> Result<Vec<SwapModel>> {
        Ok(self.list_model_inventory()?.models)
    }

    pub fn list_model_inventory(&self) -> Result<ChatModelInventory> {
        let url = endpoint_url(&self.base_url, "/v1/models");
        let response = self
            .client
            .get(&url)
            .timeout(CHAT_PROBE_TIMEOUT)
            .send()
            .with_context(|| format!("failed to query chat server at {url}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(response_error(status, response));
        }
        let mut body = Vec::new();
        response
            .take((MAX_MODEL_BODY_BYTES + 1) as u64)
            .read_to_end(&mut body)
            .context("failed to read chat model response")?;
        anyhow::ensure!(
            body.len() <= MAX_MODEL_BODY_BYTES,
            "chat model response exceeds {MAX_MODEL_BODY_BYTES} bytes"
        );
        let body = String::from_utf8(body).context("chat model response is not UTF-8")?;
        let (models, non_chat_model_ids) =
            parse_models_with_non_chat_ids(&body).context("failed to parse chat model response")?;
        Ok(ChatModelInventory {
            models,
            non_chat_model_ids,
        })
    }

    pub fn stream_completion(
        &self,
        request: &ChatRequest,
        mut on_delta: impl FnMut(&str) -> bool,
    ) -> Result<ChatCompletion> {
        stream_completion_inner(self, request, None, &mut on_delta)
    }

    pub fn stream_completion_cancellable(
        &self,
        request: &ChatRequest,
        cancellation: &AtomicBool,
        mut on_delta: impl FnMut(&str) -> bool,
    ) -> Result<ChatCompletion> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build cancellable chat runtime")?;
        runtime.block_on(stream_completion_async(
            &self.base_url,
            self.api_key.as_deref(),
            request,
            cancellation,
            &mut on_delta,
        ))
    }
}

/// Probe running llama-server instances and common local ports.
pub fn detect_chat_server(api_key: Option<&str>) -> Result<(ChatClient, ChatModelInventory)> {
    let endpoints = detected_endpoints();
    let mut failures = Vec::new();
    for endpoint in endpoints {
        match ChatClient::new(&endpoint, api_key).and_then(|client| {
            client
                .list_model_inventory()
                .map(|inventory| (client, inventory))
        }) {
            Ok(connection) => return Ok(connection),
            Err(error) => failures.push(format!("{endpoint}: {error}")),
        }
    }
    if failures.is_empty() {
        anyhow::bail!("no chat server candidates were found")
    }
    anyhow::bail!("no reachable chat server found ({})", failures.join("; "))
}

/// A llama-server process discovered from the local process table. Discovery
/// is read-only; callers must explicitly pass a returned PID to
/// [`terminate_detected_server`] before any signal is sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedServer {
    pub pid: u32,
    pub command: String,
    pub port: Option<u16>,
}

/// Return externally-started llama-server processes visible to this user.
pub fn detected_servers() -> Vec<DetectedServer> {
    #[cfg(unix)]
    {
        let output = {
            #[cfg(test)]
            let _fixture_guard = crate::control::process::executable_fixture_guard();
            Command::new("pgrep").args(["-fa", "llama-server"]).output()
        };
        if let Ok(output) = output {
            return parse_detected_servers(&String::from_utf8_lossy(&output.stdout));
        }
    }
    Vec::new()
}

/// Terminate one explicitly selected externally-started llama-server.
///
/// The process must still appear in a fresh `pgrep -fa llama-server` result,
/// its command line must identify the llama-server binary, and PID 1 is never
/// accepted. This prevents a stale or unrelated PID from being signalled.
pub fn terminate_detected_server(pid: u32) -> Result<()> {
    anyhow::ensure!(pid > 1, "refusing to terminate reserved PID {pid}");
    let server = detected_servers()
        .into_iter()
        .find(|server| server.pid == pid)
        .ok_or_else(|| anyhow::anyhow!("PID {pid} is not a detected llama-server"))?;
    anyhow::ensure!(
        is_llama_server_command(&server.command),
        "PID {pid} command is not llama-server"
    );
    #[cfg(unix)]
    {
        let status = {
            #[cfg(test)]
            let _fixture_guard = crate::control::process::executable_fixture_guard();
            Command::new("kill").arg(pid.to_string()).status()
        }
        .with_context(|| format!("signal llama-server PID {pid}"))?;
        anyhow::ensure!(status.success(), "kill returned {status}");
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = server;
        anyhow::bail!("external server termination is not supported on this platform")
    }
}

fn detected_endpoints() -> Vec<String> {
    let mut endpoints = Vec::new();
    for server in detected_servers() {
        if let Some(port) = server.port {
            push_endpoint(&mut endpoints, port);
        }
    }
    for port in [8080, 8001, 8000, 8888] {
        push_endpoint(&mut endpoints, port);
    }
    endpoints
}

fn parse_detected_servers(text: &str) -> Vec<DetectedServer> {
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields.next()?.parse::<u32>().ok()?;
            let command = fields.collect::<Vec<_>>().join(" ");
            if !is_llama_server_command(&command) {
                return None;
            }
            let port = ports_from_process_list(line).into_iter().next();
            Some(DetectedServer { pid, command, port })
        })
        .collect()
}

fn is_llama_server_command(command: &str) -> bool {
    command.split_whitespace().any(|token| {
        let token = token.trim_matches(|character: char| "'\"".contains(character));
        Path::new(token)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "llama-server" || name.starts_with("llama-server-"))
    })
}

fn ports_from_process_list(text: &str) -> Vec<u16> {
    let mut ports = Vec::new();
    for (index, token) in text.split_whitespace().enumerate() {
        let candidate = token.strip_prefix("--port=").or_else(|| {
            (token == "--port")
                .then(|| text.split_whitespace().nth(index + 1))
                .flatten()
        });
        let Some(candidate) = candidate else {
            continue;
        };
        if let Ok(port) = candidate.parse::<u16>() {
            if port > 0 && !ports.contains(&port) {
                ports.push(port);
            }
        }
    }
    ports
}

fn push_endpoint(endpoints: &mut Vec<String>, port: u16) {
    let endpoint = format!("http://localhost:{port}");
    if !endpoints.contains(&endpoint) {
        endpoints.push(endpoint);
    }
}

pub fn stream_completion(
    request: &ChatRequest,
    mut on_delta: impl FnMut(&str) -> bool,
) -> Result<ChatCompletion> {
    validate_request(request)?;
    let client = ChatClient::from_env()?;
    client.stream_completion(request, &mut on_delta)
}

fn stream_completion_inner(
    client: &ChatClient,
    request: &ChatRequest,
    cancellation: Option<&AtomicBool>,
    on_delta: &mut impl FnMut(&str) -> bool,
) -> Result<ChatCompletion> {
    validate_request(request)?;
    if is_cancelled(cancellation) {
        anyhow::bail!("chat stream cancelled");
    }
    let http_request = client
        .client
        .post(endpoint_url(&client.base_url, "/v1/chat/completions"))
        .header(ACCEPT, "text/event-stream")
        .header(CONTENT_TYPE, "application/json")
        .json(&request.payload());

    let started = Instant::now();
    let response = http_request.send().context("send streaming chat request")?;
    let status = response.status();
    if !status.is_success() {
        return Err(response_error(status, response));
    }
    let mut completion =
        consume_sse_with_cancellation(BufReader::new(response), on_delta, cancellation)?;
    completion.elapsed = started.elapsed();
    anyhow::ensure!(
        !completion.content.is_empty(),
        "chat stream completed without assistant content"
    );
    Ok(completion)
}

async fn stream_completion_async(
    base_url: &str,
    api_key: Option<&str>,
    request: &ChatRequest,
    cancellation: &AtomicBool,
    on_delta: &mut impl FnMut(&str) -> bool,
) -> Result<ChatCompletion> {
    validate_request(request)?;
    if cancellation.load(Ordering::Acquire) {
        anyhow::bail!("chat stream cancelled");
    }
    let client = reqwest::Client::builder()
        .connect_timeout(CHAT_CONNECT_TIMEOUT)
        .read_timeout(CHAT_SSE_READ_TIMEOUT)
        .timeout(CHAT_REQUEST_TIMEOUT)
        .default_headers(auth_headers(api_key)?)
        .build()
        .context("build cancellable chat HTTP client")?;
    let started = Instant::now();
    let response = client
        .post(endpoint_url(base_url, "/v1/chat/completions"))
        .header(ACCEPT, "text/event-stream")
        .header(CONTENT_TYPE, "application/json")
        .json(&request.payload())
        .send()
        .await
        .context("send streaming chat request")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_default();
        return Err(async_response_error(status, &body));
    }

    let mut response = response;
    let mut line = Vec::with_capacity(1024);
    let mut content = String::new();
    let mut completion_tokens = None;
    let mut done = false;
    while !done {
        if cancellation.load(Ordering::Acquire) {
            anyhow::bail!("chat stream cancelled");
        }
        let chunk = tokio::time::timeout(CHAT_SSE_READ_TIMEOUT, response.chunk()).await;
        let bytes = match chunk {
            Err(_) => continue,
            Ok(Err(error)) if error.is_timeout() => continue,
            Ok(Err(error)) => return Err(error).context("read chat event stream"),
            Ok(Ok(Some(bytes))) => bytes,
            Ok(Ok(None)) => break,
        };
        for byte in bytes {
            line.push(byte);
            anyhow::ensure!(
                line.len() <= MAX_SSE_LINE_BYTES,
                "chat event line exceeds {MAX_SSE_LINE_BYTES} bytes"
            );
            if byte == b'\n' {
                done = consume_sse_line(
                    &line,
                    &mut content,
                    &mut completion_tokens,
                    cancellation,
                    on_delta,
                )?;
                line.clear();
                if done {
                    break;
                }
            }
        }
    }
    if !line.is_empty() && !done {
        let _ = consume_sse_line(
            &line,
            &mut content,
            &mut completion_tokens,
            cancellation,
            on_delta,
        )?;
    }
    anyhow::ensure!(
        !content.is_empty(),
        "chat stream completed without assistant content"
    );
    Ok(ChatCompletion {
        content,
        completion_tokens,
        elapsed: started.elapsed(),
    })
}

fn consume_sse_line(
    line: &[u8],
    content: &mut String,
    completion_tokens: &mut Option<u64>,
    cancellation: &AtomicBool,
    on_delta: &mut impl FnMut(&str) -> bool,
) -> Result<bool> {
    let line = String::from_utf8_lossy(line);
    let Some(data) = line.trim_end().strip_prefix("data:") else {
        return Ok(false);
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(true);
    }
    let event: Value = match serde_json::from_str(data) {
        Ok(event) => event,
        Err(_) => return Ok(false),
    };
    if let Some(tokens) = event
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
    {
        *completion_tokens = Some(tokens);
    }
    let Some(delta) = event
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
    else {
        return Ok(false);
    };
    if delta.is_empty() {
        return Ok(false);
    }
    content.push_str(delta);
    if cancellation.load(Ordering::Acquire) || !on_delta(delta) {
        anyhow::bail!("chat stream cancelled");
    }
    Ok(false)
}

fn async_response_error(status: StatusCode, body: &[u8]) -> anyhow::Error {
    let preview = String::from_utf8_lossy(&body[..body.len().min(MAX_ERROR_BODY_BYTES as usize)]);
    let summary = preview.split_whitespace().collect::<Vec<_>>().join(" ");
    if summary.is_empty() {
        anyhow::anyhow!("chat server returned HTTP {status}")
    } else {
        anyhow::anyhow!("chat server returned HTTP {status}: {summary}")
    }
}

fn validate_request(request: &ChatRequest) -> Result<()> {
    anyhow::ensure!(
        !request.model.trim().is_empty(),
        "chat model cannot be empty"
    );
    anyhow::ensure!(
        request.temperature.is_finite() && request.temperature >= 0.0,
        "temperature must be a non-negative finite number"
    );
    anyhow::ensure!(
        request.max_tokens > 0,
        "max_tokens must be greater than zero"
    );
    anyhow::ensure!(!request.messages.is_empty(), "chat history cannot be empty");
    Ok(())
}

#[cfg(test)]
fn consume_sse(
    mut reader: impl BufRead,
    on_delta: &mut impl FnMut(&str) -> bool,
) -> Result<ChatCompletion> {
    consume_sse_with_cancellation(&mut reader, on_delta, None)
}

fn consume_sse_with_cancellation(
    mut reader: impl BufRead,
    on_delta: &mut impl FnMut(&str) -> bool,
    cancellation: Option<&AtomicBool>,
) -> Result<ChatCompletion> {
    let mut content = String::new();
    let mut completion_tokens = None;
    let mut line = Vec::with_capacity(1024);
    loop {
        if is_cancelled(cancellation) {
            anyhow::bail!("chat stream cancelled");
        }
        line.clear();
        let bytes = match reader.read_until(b'\n', &mut line) {
            Ok(bytes) => bytes,
            Err(error) if is_read_timeout(&error) => continue,
            Err(error) => return Err(error).context("read chat event stream"),
        };
        if bytes == 0 {
            break;
        }
        anyhow::ensure!(
            line.len() <= MAX_SSE_LINE_BYTES,
            "chat event line exceeds {MAX_SSE_LINE_BYTES} bytes"
        );
        let line = String::from_utf8_lossy(&line);
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let event: Value = match serde_json::from_str(data) {
            Ok(event) => event,
            Err(_) => continue,
        };
        if let Some(tokens) = event
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
        {
            completion_tokens = Some(tokens);
        }
        let Some(delta) = event
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        else {
            continue;
        };
        if delta.is_empty() {
            continue;
        }
        content.push_str(delta);
        if is_cancelled(cancellation) || !on_delta(delta) {
            anyhow::bail!("chat stream cancelled");
        }
    }
    Ok(ChatCompletion {
        content,
        completion_tokens,
        elapsed: Duration::ZERO,
    })
}

fn is_cancelled(cancellation: Option<&AtomicBool>) -> bool {
    cancellation.is_some_and(|flag| flag.load(Ordering::Acquire))
}

fn is_read_timeout(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    )
}

fn endpoint_url(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn response_error(status: StatusCode, response: Response) -> anyhow::Error {
    let mut body = String::new();
    let _ = response
        .take(MAX_ERROR_BODY_BYTES)
        .read_to_string(&mut body);
    let summary = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if summary.is_empty() {
        anyhow::anyhow!("chat server returned HTTP {status}")
    } else {
        anyhow::anyhow!("chat server returned HTTP {status}: {summary}")
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Cursor, Read, Write},
        net::TcpListener,
        sync::{atomic::AtomicBool, Arc},
        thread,
        time::{Duration, Instant},
    };

    use super::*;

    #[test]
    fn payload_includes_parameters_system_prompt_and_thinking_prefix() {
        let mut request = ChatRequest::new("model-a", vec![("user".into(), "explain this".into())]);
        request.system_prompt = " Be concise. ".into();
        request.temperature = 0.25;
        request.max_tokens = 512;
        request.thinking = true;
        let payload = request.payload();

        assert_eq!(payload["model"], "model-a");
        assert_eq!(payload["temperature"], 0.25);
        assert_eq!(payload["max_tokens"], 512);
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][0]["content"], "Be concise.");
        assert_eq!(payload["messages"][1]["content"], "/think\nexplain this");
    }

    #[test]
    fn consumes_tokens_and_usage_from_sse() {
        let input = concat!(
            ": keepalive\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n",
            "data: not-json\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut deltas = Vec::new();
        let completion = consume_sse(Cursor::new(input), &mut |delta| {
            deltas.push(delta.to_owned());
            true
        })
        .unwrap();

        assert_eq!(deltas, ["hello ", "world"]);
        assert_eq!(completion.content, "hello world");
        assert_eq!(completion.completion_tokens, Some(2));
    }

    #[test]
    fn callback_can_cancel_a_stream() {
        let input = "data: {\"choices\":[{\"delta\":{\"content\":\"stop\"}}]}\n";
        let error = consume_sse(Cursor::new(input), &mut |_| false).unwrap_err();
        assert!(error.to_string().contains("cancelled"), "{error:#}");
    }

    #[test]
    fn validates_user_controlled_parameters() {
        let mut request = ChatRequest::new("model", vec![("user".into(), "hi".into())]);
        request.temperature = f64::NAN;
        assert!(validate_request(&request).is_err());
        request.temperature = 0.8;
        request.max_tokens = 0;
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn extracts_and_orders_process_ports() {
        let ports = ports_from_process_list(
            "123 llama-server --port 8001 --host 127.0.0.1\n456 llama-server --port=8080",
        );
        assert_eq!(ports, [8001, 8080]);
    }

    #[test]
    fn detects_only_llama_server_processes_and_ports() {
        let servers = parse_detected_servers(
            "123 /opt/llama-server --port 8001 --host 127.0.0.1\n456 unrelated --port 9999\n789 llama-server --port=8080",
        );
        assert_eq!(
            servers,
            [
                DetectedServer {
                    pid: 123,
                    command: "/opt/llama-server --port 8001 --host 127.0.0.1".into(),
                    port: Some(8001),
                },
                DetectedServer {
                    pid: 789,
                    command: "llama-server --port=8080".into(),
                    port: Some(8080),
                },
            ]
        );
    }

    #[test]
    fn connected_client_lists_inventory_with_bearer_auth() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            assert!(request.contains("GET /v1/models HTTP/1.1"));
            assert!(request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-token"));
            write_http_response(
                &mut stream,
                "application/json",
                r#"{"data":[{"id":"decision-a","state":"loaded","architecture":{"output_modalities":["decisions"]}},{"id":"model-a","state":"loaded"}]}"#,
            );
        });

        let client = ChatClient::new(format!("http://{address}"), Some("test-token")).unwrap();
        let inventory = client.list_model_inventory().unwrap();
        server.join().unwrap();
        assert_eq!(inventory.non_chat_model_ids, ["decision-a"]);
        assert_eq!(inventory.models.len(), 2);
        let model = inventory
            .models
            .iter()
            .find(|model| model.id == "model-a")
            .unwrap();
        assert_eq!(model.state, "loaded");
    }

    #[test]
    fn connected_client_bounds_model_response_size() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request = read_http_request(&mut stream);
            let body = "x".repeat(MAX_MODEL_BODY_BYTES + 1);
            write_http_response(&mut stream, "application/json", &body);
        });

        let client = ChatClient::new(format!("http://{address}"), None).unwrap();
        let error = client.list_models().unwrap_err();
        server.join().unwrap();
        assert!(error.to_string().contains("exceeds"));
    }

    #[test]
    fn cancellable_stream_stops_during_an_idle_response_read() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            assert!(request.contains("POST /v1/chat/completions HTTP/1.1"));
            write_stream_prefix(&mut stream);
            thread::sleep(Duration::from_secs(2));
        });

        let client = ChatClient::new(format!("http://{address}"), None).unwrap();
        let request = ChatRequest::new("model-a", vec![("user".into(), "hello".into())]);
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation);
        let worker = thread::spawn(move || {
            client.stream_completion_cancellable(&request, &worker_cancellation, |_| true)
        });
        thread::sleep(Duration::from_millis(150));
        cancellation.store(true, std::sync::atomic::Ordering::Release);
        let started = Instant::now();
        let error = worker.join().unwrap().unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(error.to_string().contains("cancelled"), "{error:#}");
        server.join().unwrap();
    }

    fn read_http_request(stream: &mut impl Read) -> String {
        let mut request = Vec::new();
        let mut byte = [0_u8; 1];
        while !request.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
        }
        String::from_utf8(request).unwrap()
    }

    fn write_http_response(stream: &mut impl Write, content_type: &str, body: &str) {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        stream.flush().unwrap();
    }

    fn write_stream_prefix(stream: &mut impl Write) {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\ndata: {{\"choices\":[{{\"delta\":{{\"content\":\"hello\"}}}}]}}\n\n"
        )
        .unwrap();
        stream.flush().unwrap();
    }
}
