//! Bounded ComfyUI jobs, not runtime lifecycle or GPU-release proof.
//!
//! Callers must assert support for canonical client prompt IDs and atomic
//! scoped jobs cancellation. No legacy/global interrupt fallback is allowed.
//! Allocate a fresh UUID and persist it before submission; native ComfyUI does
//! not deduplicate IDs. Tokens are single-use and adapter-local, not a fence
//! against direct clients or a durable ownership registry after restart.

use std::{error::Error, fmt, sync::Arc, time::Duration};

use serde_json::Value;

use super::catalog::{Catalog, Lane};
use super::language::DispatchDisposition;
use super::requests::{InferenceRequest, MediaInput};
use super::runtime::OperationContext;
use crate::llama_swap::{auth_headers, normalize_base_url};

const MAX_RESPONSE: usize = 2 * 1024 * 1024;
const POLL: Duration = Duration::from_millis(25);

/// Explicit caller assertion, based on the installed server implementation.
/// This does not discover capabilities or enable switching/admission.
pub enum ComfyContract {
    CanonicalIdsAndAtomicScopedCancellation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComfyErrorKind {
    Configuration,
    InvalidRequest,
    InvalidOwnership,
    AlreadyAttempted,
    Cancelled,
    Deadline,
    Transport,
    Rejected,
    Protocol,
    TooLarge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComfyFailure {
    pub disposition: DispatchDisposition,
    pub kind: ComfyErrorKind,
}
impl fmt::Display for ComfyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Comfy operation {:?}: {:?}", self.disposition, self.kind)
    }
}
impl Error for ComfyFailure {}

/// Not Clone/Debug: private affinity and one-shot state protect cancellation
/// from arbitrary UUIDs, and workflow payloads must not enter logs/history.
pub struct PreparedComfyJob {
    owner: Arc<()>,
    id: String,
    catalog: Catalog,
    request: InferenceRequest,
    attempted: bool,
}
impl PreparedComfyJob {
    /// Save this ID in durable job history before calling submit.
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn submission_attempted(&self) -> bool {
        self.attempted
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComfyJobState {
    Unknown,
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Backend-relative file descriptors, never trusted local paths. No file is
/// opened/downloaded here; names can be sensitive and are omitted from Debug.
#[derive(Clone, Eq, PartialEq)]
pub struct ComfyArtifact {
    pub node_id: String,
    pub media_type: String,
    pub filename: String,
    pub subfolder: String,
    pub storage_type: String,
}
impl fmt::Debug for ComfyArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ComfyArtifact(<redacted>)")
    }
}

#[derive(Debug)]
pub struct ComfySnapshot {
    pub state: ComfyJobState,
    pub artifacts: Vec<ComfyArtifact>,
}

pub struct ComfyAdapter {
    endpoint: String,
    api_key: Option<String>,
    owner: Arc<()>,
}
impl ComfyAdapter {
    pub fn new(
        endpoint: &str,
        api_key: Option<&str>,
        _contract: ComfyContract,
    ) -> Result<Self, ComfyFailure> {
        let error = no_dispatch(ComfyErrorKind::Configuration);
        if endpoint.trim().is_empty() {
            return Err(error);
        }
        let endpoint = normalize_base_url(endpoint).map_err(|_| error)?;
        let url = reqwest::Url::parse(&endpoint).map_err(|_| error)?;
        if !url.username().is_empty() || url.password().is_some() {
            return Err(error);
        }
        auth_headers(api_key).map_err(|_| error)?;
        Ok(Self {
            endpoint,
            api_key: api_key.map(str::to_owned),
            owner: Arc::new(()),
        })
    }

    pub fn prepare(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        fresh_uuid: &str,
    ) -> Result<PreparedComfyJob, ComfyFailure> {
        if !canonical_uuid(fresh_uuid) {
            return Err(no_dispatch(ComfyErrorKind::InvalidRequest));
        }
        validate_workflow(catalog, request)?;
        Ok(PreparedComfyJob {
            owner: self.owner.clone(),
            id: fresh_uuid.to_owned(),
            catalog: catalog.clone(),
            request: request.clone(),
            attempted: false,
        })
    }

    /// Blocking-executor thread only. A token becomes used before HTTP work
    /// starts and stays used on every failure. There is no automatic replay.
    pub fn submit(
        &self,
        job: &mut PreparedComfyJob,
        context: &OperationContext<'_>,
    ) -> Result<(), ComfyFailure> {
        self.check_owner(job, false)?;
        if job.attempted {
            return Err(no_dispatch(ComfyErrorKind::AlreadyAttempted));
        }
        active(context, DispatchDisposition::NoDispatch)?;
        validate_workflow(&job.catalog, &job.request)?;
        let InferenceRequest::Media {
            input: MediaInput::Workflow { graph, .. },
            ..
        } = &job.request
        else {
            return Err(no_dispatch(ComfyErrorKind::InvalidRequest));
        };
        let payload = serde_json::json!({"prompt_id":job.id, "prompt":graph});
        job.attempted = true;
        let (status, body) = self.http("/prompt", Some(&payload), context)?;
        if !status.is_success() {
            return Err(uncertain(ComfyErrorKind::Rejected));
        }
        if body.get("prompt_id").and_then(Value::as_str) != Some(job.id()) {
            return Err(uncertain(ComfyErrorKind::Protocol));
        }
        Ok(())
    }

    pub fn status(
        &self,
        job: &PreparedComfyJob,
        context: &OperationContext<'_>,
    ) -> Result<ComfySnapshot, ComfyFailure> {
        self.check_owner(job, true)?;
        let (status, body) = self.http(&format!("/api/jobs/{}", job.id), None, context)?;
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(ComfySnapshot {
                state: ComfyJobState::Unknown,
                artifacts: Vec::new(),
            });
        }
        if !status.is_success() {
            return Err(uncertain(ComfyErrorKind::Rejected));
        }
        parse_snapshot(&body, job.id())
    }

    /// True means cancellation was dispatched, NOT acknowledged completion.
    /// False/404/unknown never proves cleanup. Use a fresh recovery context if
    /// the original execution context is already cancelled or expired.
    pub fn cancel(
        &self,
        job: &PreparedComfyJob,
        context: &OperationContext<'_>,
    ) -> Result<bool, ComfyFailure> {
        self.check_owner(job, true)?;
        let (status, body) = self.http(
            &format!("/api/jobs/{}/cancel", job.id),
            Some(&serde_json::json!({})),
            context,
        )?;
        if !status.is_success() {
            return Err(uncertain(ComfyErrorKind::Rejected));
        }
        body.get("cancelled")
            .and_then(Value::as_bool)
            .ok_or_else(|| uncertain(ComfyErrorKind::Protocol))
    }

    fn check_owner(&self, job: &PreparedComfyJob, attempted: bool) -> Result<(), ComfyFailure> {
        if !Arc::ptr_eq(&self.owner, &job.owner) || (attempted && !job.attempted) {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        Ok(())
    }

    fn http(
        &self,
        path: &str,
        payload: Option<&Value>,
        context: &OperationContext<'_>,
    ) -> Result<(reqwest::StatusCode, Value), ComfyFailure> {
        active(context, DispatchDisposition::NoDispatch)?;
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(no_dispatch(ComfyErrorKind::Configuration));
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?;
        runtime.block_on(async {
            let client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(context.remaining().min(Duration::from_secs(2)))
                .timeout(context.remaining())
                .default_headers(
                    auth_headers(self.api_key.as_deref())
                        .map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?,
                )
                .build()
                .map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?;
            let url = format!("{}{path}", self.endpoint);
            let request = match payload {
                Some(body) => client.post(url).json(body),
                None => client.get(url),
            };
            active(context, DispatchDisposition::NoDispatch)?;
            let send = request.send();
            tokio::pin!(send);
            let mut response = loop {
                active(context, DispatchDisposition::Uncertain)?;
                match tokio::time::timeout(context.remaining().min(POLL), &mut send).await {
                    Err(_) => continue,
                    Ok(result) => break result.map_err(transport)?,
                }
            };
            let status = response.status();
            if response
                .content_length()
                .is_some_and(|n| n > MAX_RESPONSE as u64)
            {
                return Err(uncertain(ComfyErrorKind::TooLarge));
            }
            let mut bytes = Vec::new();
            loop {
                active(context, DispatchDisposition::Uncertain)?;
                let chunk = response.chunk();
                tokio::pin!(chunk);
                let chunk = loop {
                    active(context, DispatchDisposition::Uncertain)?;
                    match tokio::time::timeout(context.remaining().min(POLL), &mut chunk).await {
                        Err(_) => continue,
                        Ok(result) => break result.map_err(transport)?,
                    }
                };
                let Some(chunk) = chunk else { break };
                if chunk.len() > MAX_RESPONSE.saturating_sub(bytes.len()) {
                    return Err(uncertain(ComfyErrorKind::TooLarge));
                }
                bytes.extend_from_slice(&chunk);
            }
            active(context, DispatchDisposition::Uncertain)?;
            // Non-success bodies can be HTML or contain prompts/credentials.
            let body = if status.is_success() {
                serde_json::from_slice(&bytes).map_err(|_| uncertain(ComfyErrorKind::Protocol))?
            } else {
                Value::Null
            };
            Ok((status, body))
        })
    }
}

fn validate_workflow(catalog: &Catalog, request: &InferenceRequest) -> Result<(), ComfyFailure> {
    let error = no_dispatch(ComfyErrorKind::InvalidRequest);
    let refreshed = catalog.refresh().map_err(|_| error)?;
    let original = catalog.inspect(request.profile()).ok_or(error)?;
    let current = request.validate(&refreshed).map_err(|_| error)?;
    if current.runtime != original.runtime
        || current.runtime.lane != Lane::Media
        || current.runtime.as_str() != "ComfyUI"
        || !matches!(request, InferenceRequest::Media { input: MediaInput::Workflow { graph, .. }, .. } if graph.as_object().is_some_and(|o| !o.is_empty()))
    {
        return Err(error);
    }
    Ok(())
}

fn canonical_uuid(id: &str) -> bool {
    id.len() == 36
        && id.bytes().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                b == b'-'
            } else {
                b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
            }
        })
}

fn safe_relative(path: &str, empty: bool) -> bool {
    (empty && path.is_empty())
        || (!path.is_empty()
            && path.len() <= 512
            && !path.contains(['\\', ':', '%', '?', '#'])
            && !path.chars().any(char::is_control)
            && path
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != ".." && part.len() <= 128))
}

fn parse_snapshot(body: &Value, id: &str) -> Result<ComfySnapshot, ComfyFailure> {
    let error = uncertain(ComfyErrorKind::Protocol);
    if body.get("id").and_then(Value::as_str) != Some(id) {
        return Err(error);
    }
    let state = match body.get("status").and_then(Value::as_str) {
        Some("pending") => ComfyJobState::Pending,
        Some("in_progress") => ComfyJobState::Running,
        Some("completed") => ComfyJobState::Completed,
        Some("failed") => ComfyJobState::Failed,
        Some("cancelled") => ComfyJobState::Cancelled,
        _ => return Err(error),
    };
    if matches!(
        state,
        ComfyJobState::Completed | ComfyJobState::Failed | ComfyJobState::Cancelled
    ) {
        let native = body.get("execution_status").ok_or(error)?;
        let success = state == ComfyJobState::Completed;
        if native.get("completed").and_then(Value::as_bool) != Some(success)
            || native.get("status_str").and_then(Value::as_str)
                != Some(if success { "success" } else { "error" })
        {
            return Err(error);
        }
        let messages = native
            .get("messages")
            .and_then(Value::as_array)
            .ok_or(error)?;
        let interrupted = messages
            .iter()
            .any(|m| m.get(0).and_then(Value::as_str) == Some("execution_interrupted"));
        if interrupted != (state == ComfyJobState::Cancelled) {
            return Err(error);
        }
    }
    let mut artifacts = Vec::new();
    if let Some(outputs) = body.get("outputs") {
        for (node_id, node) in outputs.as_object().ok_or(error)? {
            if node_id.is_empty() || node_id.len() > 128 || node_id.chars().any(char::is_control) {
                return Err(error);
            }
            for (media_type, items) in node.as_object().ok_or(error)? {
                if media_type.is_empty()
                    || media_type.len() > 128
                    || media_type.chars().any(char::is_control)
                {
                    return Err(error);
                }
                let Some(items) = items.as_array() else {
                    continue;
                };
                for item in items {
                    let Some(filename) = item.get("filename") else {
                        continue;
                    };
                    let filename = filename.as_str().ok_or(error)?;
                    let subfolder = item.get("subfolder").and_then(Value::as_str).ok_or(error)?;
                    let storage_type = item.get("type").and_then(Value::as_str).ok_or(error)?;
                    if !safe_relative(filename, false)
                        || filename.contains('/')
                        || !safe_relative(subfolder, true)
                        || !matches!(storage_type, "output" | "temp" | "input")
                    {
                        return Err(error);
                    }
                    if artifacts.len() >= 128 {
                        return Err(uncertain(ComfyErrorKind::TooLarge));
                    }
                    artifacts.push(ComfyArtifact {
                        node_id: node_id.clone(),
                        media_type: media_type.clone(),
                        filename: filename.into(),
                        subfolder: subfolder.into(),
                        storage_type: storage_type.into(),
                    });
                }
            }
        }
    }
    Ok(ComfySnapshot { state, artifacts })
}

fn no_dispatch(kind: ComfyErrorKind) -> ComfyFailure {
    ComfyFailure {
        disposition: DispatchDisposition::NoDispatch,
        kind,
    }
}
fn uncertain(kind: ComfyErrorKind) -> ComfyFailure {
    ComfyFailure {
        disposition: DispatchDisposition::Uncertain,
        kind,
    }
}
fn active(
    context: &OperationContext<'_>,
    disposition: DispatchDisposition,
) -> Result<(), ComfyFailure> {
    let kind = if context.is_cancelled() {
        ComfyErrorKind::Cancelled
    } else if context.remaining().is_zero() {
        ComfyErrorKind::Deadline
    } else {
        return Ok(());
    };
    Err(ComfyFailure { disposition, kind })
}
fn transport(error: reqwest::Error) -> ComfyFailure {
    uncertain(if error.is_timeout() {
        ComfyErrorKind::Deadline
    } else {
        ComfyErrorKind::Transport
    })
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{ProfileKey, Task};
    use super::super::runtime::{Cancellation, NeverCancel};
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    const ID: &str = "12345678-1234-1234-1234-123456789abc";
    fn adapter(endpoint: &str) -> ComfyAdapter {
        ComfyAdapter::new(
            endpoint,
            Some("fixture"),
            ComfyContract::CanonicalIdsAndAtomicScopedCancellation,
        )
        .unwrap()
    }
    fn request() -> InferenceRequest {
        InferenceRequest::Media {
            profile: ProfileKey::new(Lane::Media, "qwen-image-2.1").unwrap(),
            input: MediaInput::Workflow {
                task: Task::new("image").unwrap(),
                graph: serde_json::json!({"1":{"class_type":"Fixture","inputs":{"text":"SENSITIVE_PROMPT"}}}),
            },
        }
    }
    fn prepared(adapter: &ComfyAdapter) -> PreparedComfyJob {
        adapter
            .prepare(
                &Catalog::load(env!("CARGO_MANIFEST_DIR")).unwrap(),
                &request(),
                ID,
            )
            .unwrap()
    }
    fn context() -> OperationContext<'static> {
        OperationContext::new(Instant::now() + Duration::from_secs(2), &NeverCancel)
    }
    fn terminal(state: &str) -> Value {
        let success = state == "completed";
        let event = match state {
            "cancelled" => "execution_interrupted",
            "failed" => "execution_error",
            _ => "execution_success",
        };
        serde_json::json!({"id":ID,"status":state,"execution_status":{"status_str":if success {"success"} else {"error"},"completed":success,"messages":[[event,{"prompt_id":ID}]]},"outputs":{},"workflow":{"prompt":{"secret":"MUST_NOT_LEAK"}}})
    }
    type Recorded = Vec<(String, Value)>;
    fn fixture(
        replies: Vec<(u16, String, Duration)>,
    ) -> (String, std::thread::JoinHandle<Recorded>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let server = std::thread::spawn(move || {
            let mut recorded = Vec::new();
            for (status, body, delay) in replies {
                let deadline = Instant::now() + Duration::from_secs(3);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "fixture accept deadline");
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("fixture accept: {error}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut header = Vec::new();
                let mut byte = [0];
                while !header.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    header.push(byte[0]);
                    assert!(header.len() < 16384);
                }
                let header = String::from_utf8(header).unwrap();
                let length = header
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                assert!(length <= 1024 * 1024 + 256);
                let mut request = vec![0; length];
                stream.read_exact(&mut request).unwrap();
                recorded.push((
                    header,
                    if request.is_empty() {
                        Value::Null
                    } else {
                        serde_json::from_slice(&request).unwrap()
                    },
                ));
                std::thread::sleep(delay);
                let _ = write!(stream,"HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());
            }
            recorded
        });
        (endpoint, server)
    }
    fn reply(body: Value) -> (u16, String, Duration) {
        (200, body.to_string(), Duration::ZERO)
    }

    #[test]
    fn native_http_routes_preserve_graph_and_scope_cancel_to_owned_job() {
        let mut output = terminal("completed");
        output["outputs"] = serde_json::json!({"9":{"images":[{"filename":"out.png","subfolder":"run","type":"output"}],"audio":[{"filename":"out.wav","subfolder":"","type":"output"}],"video":[{"filename":"out.mp4","subfolder":"","type":"temp"}]}});
        let (endpoint, server) = fixture(vec![
            reply(serde_json::json!({"prompt_id":ID})),
            reply(output),
            reply(serde_json::json!({"cancelled":true})),
        ]);
        let adapter = adapter(&endpoint);
        let mut job = prepared(&adapter);
        assert_eq!(job.id(), ID);
        assert!(!job.submission_attempted());
        adapter.submit(&mut job, &context()).unwrap();
        assert_eq!(
            adapter.submit(&mut job, &context()).unwrap_err().kind,
            ComfyErrorKind::AlreadyAttempted
        );
        let snapshot = adapter.status(&job, &context()).unwrap();
        assert_eq!(snapshot.state, ComfyJobState::Completed);
        assert_eq!(snapshot.artifacts.len(), 3);
        assert!(!format!("{snapshot:?}").contains("out.png"));
        assert!(!format!("{snapshot:?}").contains("MUST_NOT_LEAK"));
        assert!(adapter.cancel(&job, &context()).unwrap());
        let recorded = server.join().unwrap();
        assert!(recorded[0].0.starts_with("POST /prompt HTTP/1.1"));
        assert!(recorded[0]
            .0
            .to_lowercase()
            .contains("authorization: bearer fixture"));
        assert_eq!(recorded[0].1["prompt_id"], ID);
        assert_eq!(
            recorded[0].1["prompt"]["1"]["inputs"]["text"],
            "SENSITIVE_PROMPT"
        );
        assert!(recorded[1]
            .0
            .starts_with(&format!("GET /api/jobs/{ID} HTTP/1.1")));
        assert!(recorded[2]
            .0
            .starts_with(&format!("POST /api/jobs/{ID}/cancel HTTP/1.1")));
        assert!(recorded
            .iter()
            .all(|(header, _)| !header.contains("/interrupt")));
    }

    #[test]
    fn unknown_and_cancel_noop_are_not_terminal_proof() {
        let (endpoint, server) = fixture(vec![
            reply(serde_json::json!({"prompt_id":ID})),
            (404, "secret failure".into(), Duration::ZERO),
            reply(serde_json::json!({"cancelled":false})),
        ]);
        let adapter = adapter(&endpoint);
        let mut job = prepared(&adapter);
        adapter.submit(&mut job, &context()).unwrap();
        assert_eq!(
            adapter.status(&job, &context()).unwrap().state,
            ComfyJobState::Unknown
        );
        assert!(!adapter.cancel(&job, &context()).unwrap());
        assert!(job.submission_attempted());
        server.join().unwrap();
    }

    #[test]
    fn malformed_submission_keeps_id_and_never_replays() {
        let (endpoint, server) =
            fixture(vec![(200, "malformed-secret-body".into(), Duration::ZERO)]);
        let adapter = adapter(&endpoint);
        let mut job = prepared(&adapter);
        let error = adapter.submit(&mut job, &context()).unwrap_err();
        assert_eq!(error, uncertain(ComfyErrorKind::Protocol));
        assert!(!error.to_string().contains("secret"));
        assert_eq!(job.id(), ID);
        assert!(job.submission_attempted());
        assert_eq!(
            adapter.submit(&mut job, &context()).unwrap_err().kind,
            ComfyErrorKind::AlreadyAttempted
        );
        assert_eq!(server.join().unwrap().len(), 1);
    }

    struct Flag(Arc<AtomicBool>);
    impl Cancellation for Flag {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }
    #[test]
    fn idle_headers_observe_cancellation_and_deadline_without_replay() {
        for cancel in [false, true] {
            let (endpoint, server) = fixture(vec![(
                200,
                serde_json::json!({"prompt_id":ID}).to_string(),
                Duration::from_millis(200),
            )]);
            let adapter = adapter(&endpoint);
            let mut job = prepared(&adapter);
            let flag = Arc::new(AtomicBool::new(false));
            let cancellation = Flag(flag.clone());
            let signal = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(60));
                if cancel {
                    flag.store(true, Ordering::Release);
                }
            });
            let start = Instant::now();
            let ctx = OperationContext::new(
                start
                    + if cancel {
                        Duration::from_secs(2)
                    } else {
                        Duration::from_millis(80)
                    },
                &cancellation,
            );
            let failure = adapter.submit(&mut job, &ctx).unwrap_err();
            assert_eq!(
                failure,
                uncertain(if cancel {
                    ComfyErrorKind::Cancelled
                } else {
                    ComfyErrorKind::Deadline
                })
            );
            assert!(start.elapsed() < Duration::from_secs(1));
            assert!(job.submission_attempted());
            signal.join().unwrap();
            server.join().unwrap();
        }
    }

    #[test]
    fn ownership_and_configuration_fail_before_http() {
        let first = adapter("http://127.0.0.1:9");
        let second = adapter("http://127.0.0.1:9");
        let mut job = prepared(&first);
        assert_eq!(
            first.cancel(&job, &context()).unwrap_err(),
            no_dispatch(ComfyErrorKind::InvalidOwnership)
        );
        assert_eq!(
            second.submit(&mut job, &context()).unwrap_err(),
            no_dispatch(ComfyErrorKind::InvalidOwnership)
        );
        assert!(!job.submission_attempted());
        for id in ["../escape", "12345678-1234-1234-1234-123456789ABC", ""] {
            assert!(!canonical_uuid(id));
        }
        for url in [
            "",
            "http://user:secret@localhost",
            "http://localhost?secret=x",
        ] {
            assert_eq!(
                ComfyAdapter::new(
                    url,
                    None,
                    ComfyContract::CanonicalIdsAndAtomicScopedCancellation
                )
                .err()
                .unwrap()
                .kind,
                ComfyErrorKind::Configuration
            );
        }
        assert_eq!(
            first
                .submit(
                    &mut job,
                    &OperationContext::new(Instant::now(), &NeverCancel)
                )
                .unwrap_err(),
            no_dispatch(ComfyErrorKind::Deadline)
        );
        assert!(!job.submission_attempted());
    }

    #[test]
    fn terminal_history_and_artifact_validation_fail_closed() {
        for (label, state) in [
            ("completed", ComfyJobState::Completed),
            ("failed", ComfyJobState::Failed),
            ("cancelled", ComfyJobState::Cancelled),
        ] {
            assert_eq!(parse_snapshot(&terminal(label), ID).unwrap().state, state);
        }
        for label in ["pending", "in_progress"] {
            assert!(parse_snapshot(&serde_json::json!({"id":ID,"status":label}), ID).is_ok());
        }
        assert!(parse_snapshot(&serde_json::json!({"id":ID,"status":"completed"}), ID).is_err());
        let mut bad = terminal("completed");
        bad["id"] = Value::String("wrong".into());
        assert!(parse_snapshot(&bad, ID).is_err());
        for name in [
            "../out.png",
            "/out.png",
            "a/b.png",
            "a\\b.png",
            "C:out.png",
            "%2e%2e%2fsecret.png",
            "out.png?path=secret",
            "out.png#fragment",
            "\u{0}",
        ] {
            let mut bad = terminal("completed");
            bad["outputs"] = serde_json::json!({"9":{"images":[{"filename":name,"subfolder":"","type":"output"}]}});
            assert!(parse_snapshot(&bad, ID).is_err());
        }
        for folder in [
            "..",
            "a/../b",
            "/absolute",
            "a//b",
            "%2fescape",
            "a?b",
            "a#b",
        ] {
            assert!(!safe_relative(folder, true));
        }
        let mut bad = terminal("cancelled");
        bad["execution_status"]["messages"] = serde_json::json!([]);
        assert!(parse_snapshot(&bad, ID).is_err());
        let mut many = terminal("completed");
        many["outputs"] = serde_json::json!({"9":{"images":vec![serde_json::json!({"filename":"a.png","subfolder":"","type":"output"});129]}});
        assert_eq!(
            parse_snapshot(&many, ID).unwrap_err().kind,
            ComfyErrorKind::TooLarge
        );
    }

    #[test]
    fn manifest_runtime_changes_are_rejected_before_dispatch() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let media = tempfile::tempdir().unwrap();
        let manifest = media.path().join("media-runtimes.json");
        let mut native: Value = serde_json::from_slice(
            &std::fs::read(root.join("toolkits/l2m2/media-runtimes.json")).unwrap(),
        )
        .unwrap();
        std::fs::write(&manifest, native.to_string()).unwrap();
        let catalog =
            Catalog::load_from_roots(root, media.path(), root.join("toolkits/l3ds")).unwrap();
        let adapter = adapter("http://127.0.0.1:9");
        let mut job = adapter.prepare(&catalog, &request(), ID).unwrap();
        let profile = native["profiles"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|profile| profile["id"] == "qwen-image-2.1")
            .unwrap();
        profile["runtime"] = Value::String("changed-runtime".into());
        std::fs::write(&manifest, native.to_string()).unwrap();
        assert_eq!(
            adapter.submit(&mut job, &context()).unwrap_err(),
            no_dispatch(ComfyErrorKind::InvalidRequest)
        );
        assert!(!job.submission_attempted());
    }

    #[test]
    fn unsupported_scoped_cancel_never_falls_back_to_global_interrupt() {
        let (endpoint, server) = fixture(vec![
            reply(serde_json::json!({"prompt_id":ID})),
            (
                404,
                "unsupported scoped cancellation".into(),
                Duration::ZERO,
            ),
        ]);
        let adapter = adapter(&endpoint);
        let mut job = prepared(&adapter);
        adapter.submit(&mut job, &context()).unwrap();
        assert_eq!(
            adapter.cancel(&job, &context()).unwrap_err(),
            uncertain(ComfyErrorKind::Rejected)
        );
        let recorded = server.join().unwrap();
        assert_eq!(recorded.len(), 2);
        assert!(recorded[1]
            .0
            .starts_with(&format!("POST /api/jobs/{ID}/cancel HTTP/1.1")));
        assert!(job.submission_attempted());
    }

    #[test]
    fn oversized_http_response_is_bounded_and_uncertain() {
        let (endpoint, server) = fixture(vec![(200, "x".repeat(MAX_RESPONSE + 1), Duration::ZERO)]);
        let adapter = adapter(&endpoint);
        let mut job = prepared(&adapter);
        assert_eq!(
            adapter.submit(&mut job, &context()).unwrap_err(),
            uncertain(ComfyErrorKind::TooLarge)
        );
        assert!(job.submission_attempted());
        server.join().unwrap();
    }
}
