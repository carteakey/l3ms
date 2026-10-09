//! Bounded ComfyUI jobs, not runtime lifecycle or GPU-release proof.
//!
//! Callers must assert support for canonical client prompt IDs and atomic
//! scoped jobs cancellation. No legacy/global interrupt fallback is allowed.
//! Allocate a fresh UUID and persist it before submission; native ComfyUI does
//! not deduplicate IDs. Tokens are single-use and adapter-local, not a fence
//! against direct clients or a durable ownership registry after restart.

use std::{collections::BTreeMap, error::Error, fmt, sync::Arc, time::Duration};

use serde::de::value::MapAccessDeserializer;
use serde::de::{Error as DeError, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use super::catalog::{Catalog, Lane};
use super::language::DispatchDisposition;
use super::requests::{InferenceRequest, MediaInput};
use super::routes::{resolve_profile_route, ResolvedProfileRoute};
use super::runtime::OperationContext;
use crate::llama_swap::{auth_headers, normalize_base_url};

const MAX_RESPONSE: usize = 2 * 1024 * 1024;
const MAX_NATIVE_OBJECT_ENTRIES: usize = 4096;
const MAX_NATIVE_OUTPUT_ITEMS: usize = 4096;
const MAX_NATIVE_MESSAGES: usize = 4096;
const POLL: Duration = Duration::from_millis(25);

type ComfyHttpCall<'a> = dyn Fn(
        &str,
        Option<&Value>,
        &OperationContext<'_>,
    ) -> Result<(reqwest::StatusCode, Vec<u8>), ComfyFailure>
    + 'a;

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
    accepted_route: ResolvedProfileRoute,
    attempted: bool,
}
impl PreparedComfyJob {
    /// Candidate native prompt ID. Persist it as a dispatch intent before
    /// submit; it is not an acknowledged backend ID until native acceptance.
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn submission_attempted(&self) -> bool {
        self.attempted
    }
}

/// A managed job cannot be handed to the legacy HTTP adapter: its inner
/// preparation is private and its exact owned generation stays attached.
#[cfg(target_os = "linux")]
pub struct PreparedManagedComfyJob {
    inner: PreparedComfyJob,
    generation: super::managed_service::OwnedServiceGenerationIdentity,
    cancel_attempted: bool,
}

#[cfg(target_os = "linux")]
impl PreparedManagedComfyJob {
    pub fn id(&self) -> &str {
        self.inner.id()
    }

    pub fn submission_attempted(&self) -> bool {
        self.inner.submission_attempted()
    }
}

/// Exact-generation native operations, not lifecycle/admission or release
/// proof. Every operation revalidates the owner and actual socket peer. No
/// legacy reqwest/TCP/auth fallback can receive this session's managed jobs.
#[cfg(target_os = "linux")]
pub struct OwnedComfySession<'session, 'host> {
    adapter: &'session ComfyAdapter,
    owner: &'session super::service_owner::ManagedRuntimeOwner<'host>,
    service: &'session super::managed_service::OwnedService,
    socket_directory: &'session std::path::Path,
    generation: super::managed_service::OwnedServiceGenerationIdentity,
}

#[cfg(target_os = "linux")]
impl<'session, 'host> OwnedComfySession<'session, 'host> {
    pub fn new(
        adapter: &'session ComfyAdapter,
        owner: &'session super::service_owner::ManagedRuntimeOwner<'host>,
        service: &'session super::managed_service::OwnedService,
        socket_directory: &'session std::path::Path,
        context: &OperationContext<'_>,
    ) -> Result<Self, ComfyFailure> {
        active(context, DispatchDisposition::NoDispatch)?;
        if adapter.api_key.is_some()
            || !service
                .intent_record()
                .runtime_identity()
                .is_some_and(|runtime| runtime.lane == Lane::Media && runtime.as_str() == "ComfyUI")
            || service.supervisor_peer_identity().is_none()
        {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        let generation = owner
            .running_generation_identity(service, context)
            .map_err(owned_generation_failure)?;
        Ok(Self {
            adapter,
            owner,
            service,
            socket_directory,
            generation,
        })
    }

    pub fn prepare_for_route(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        accepted_route: &ResolvedProfileRoute,
        fresh_uuid: &str,
    ) -> Result<PreparedManagedComfyJob, ComfyFailure> {
        if self.service.intent_record().runtime_identity() != Some(&accepted_route.runtime) {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        Ok(PreparedManagedComfyJob {
            inner: self
                .adapter
                .prepare_for_route(catalog, request, accepted_route, fresh_uuid)?,
            generation: self.generation.clone(),
            cancel_attempted: false,
        })
    }

    pub fn status(
        &self,
        job: &PreparedManagedComfyJob,
        context: &OperationContext<'_>,
    ) -> Result<ComfySnapshot, ComfyFailure> {
        self.check_job(job)?;
        self.adapter
            .status_using_http(&job.inner, context, &|path, payload, context| {
                self.http(path, payload, context)
            })
    }

    /// Main-thread handoff: prepare one job and atomically persist its fresh
    /// candidate plus exact generation before returning a one-shot permit.
    /// Losing this handoff never authorizes a second candidate or submission.
    pub fn prepare_dispatch(
        &self,
        coordinator: &mut super::inference::CoordinatorCore,
        catalog: &Catalog,
        ticket: &super::inference::DispatchTicket,
        context: &OperationContext<'_>,
    ) -> Result<
        (
            PreparedManagedComfyJob,
            super::inference::ManagedComfyDispatchPermit,
        ),
        ComfyFailure,
    > {
        active(context, DispatchDisposition::NoDispatch)?;
        let current = self
            .owner
            .running_generation_identity(self.service, context)
            .map_err(owned_generation_failure)?;
        if current != self.generation {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        let candidate = fresh_comfy_candidate(context)?;
        let job = self.prepare_for_route(catalog, &ticket.request, &ticket.route, &candidate)?;
        let permit = coordinator
            .record_managed_comfy_dispatch_candidate(
                ticket,
                candidate,
                current,
                std::time::Instant::now(),
            )
            .map_err(|_| no_dispatch(ComfyErrorKind::InvalidOwnership))?;
        if !coordinator.accepts_managed_comfy_dispatch_permit(&permit) {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        Ok((job, permit))
    }

    pub fn submit_with_dispatch_permit(
        &self,
        job: &mut PreparedManagedComfyJob,
        permit: super::inference::ManagedComfyDispatchPermit,
        context: &OperationContext<'_>,
    ) -> Result<(), ComfyFailure> {
        self.check_job(job)?;
        if !permit.matches_generation(&self.generation)
            || !permit.matches_candidate(job.id(), &job.inner.accepted_route)
            || !permit.matches_request(
                permit.request_id(),
                &job.inner.request,
                &job.inner.accepted_route,
            )
        {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        let route = job.inner.accepted_route.clone();
        self.adapter.submit_using_http(
            &mut job.inner,
            &route,
            context,
            &|path, payload, context| self.http(path, payload, context),
        )
    }

    /// A single scoped cancel attempt. Its acknowledgment is not terminal
    /// evidence; uncertainty never authorizes another cancellation or POST.
    pub fn cancel(
        &self,
        job: &mut PreparedManagedComfyJob,
        context: &OperationContext<'_>,
    ) -> Result<bool, ComfyFailure> {
        self.check_job(job)?;
        self.adapter.check_owner(&job.inner, true)?;
        if job.cancel_attempted {
            return Err(no_dispatch(ComfyErrorKind::AlreadyAttempted));
        }
        job.cancel_attempted = true;
        self.adapter
            .cancel_using_http(&job.inner, context, &|path, payload, context| {
                self.http(path, payload, context)
            })
    }

    fn check_job(&self, job: &PreparedManagedComfyJob) -> Result<(), ComfyFailure> {
        if job.generation != self.generation {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        self.adapter.check_owner(&job.inner, false)
    }

    fn http(
        &self,
        path: &str,
        payload: Option<&Value>,
        context: &OperationContext<'_>,
    ) -> Result<(reqwest::StatusCode, Vec<u8>), ComfyFailure> {
        use super::engine_health::{OwnedComfyFailureKind, OwnedComfyOperation};
        let invalid = || no_dispatch(ComfyErrorKind::InvalidRequest);
        let operation = if path == "/prompt" {
            let payload = payload.ok_or_else(invalid)?;
            OwnedComfyOperation::Submit {
                prompt_id: payload
                    .get("prompt_id")
                    .and_then(Value::as_str)
                    .ok_or_else(invalid)?,
                graph: payload.get("prompt").ok_or_else(invalid)?,
            }
        } else if let Some(prompt_id) = path.strip_prefix("/api/jobs/") {
            match prompt_id.strip_suffix("/cancel") {
                Some(prompt_id) if payload == Some(&serde_json::json!({})) => {
                    OwnedComfyOperation::Cancel { prompt_id }
                }
                None if payload.is_none() => OwnedComfyOperation::Status { prompt_id },
                _ => return Err(invalid()),
            }
        } else {
            return Err(invalid());
        };
        let response = super::engine_health::request_owned_comfy_json(
            self.owner,
            self.service,
            self.socket_directory,
            &self.generation,
            operation,
            context,
        )
        .map_err(|failure| ComfyFailure {
            disposition: failure.disposition,
            kind: match failure.kind {
                OwnedComfyFailureKind::Configuration => ComfyErrorKind::Configuration,
                OwnedComfyFailureKind::Ownership => ComfyErrorKind::InvalidOwnership,
                OwnedComfyFailureKind::Cancelled => ComfyErrorKind::Cancelled,
                OwnedComfyFailureKind::Deadline => ComfyErrorKind::Deadline,
                OwnedComfyFailureKind::Transport => ComfyErrorKind::Transport,
                OwnedComfyFailureKind::Protocol => ComfyErrorKind::Protocol,
                OwnedComfyFailureKind::TooLarge => ComfyErrorKind::TooLarge,
            },
        })?;
        let status = reqwest::StatusCode::from_u16(response.status)
            .map_err(|_| uncertain(ComfyErrorKind::Protocol))?;
        Ok((status, response.body))
    }
}

#[cfg(target_os = "linux")]
fn fresh_comfy_candidate(context: &OperationContext<'_>) -> Result<String, ComfyFailure> {
    let mut bytes = [0u8; 16];
    let mut offset = 0;
    while offset < bytes.len() {
        active(context, DispatchDisposition::NoDispatch)?;
        let count = unsafe {
            libc::getrandom(
                bytes[offset..].as_mut_ptr().cast(),
                bytes.len() - offset,
                libc::GRND_NONBLOCK,
            )
        };
        if count < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(no_dispatch(ComfyErrorKind::Configuration));
        }
        if count == 0 {
            return Err(no_dispatch(ComfyErrorKind::Configuration));
        }
        offset += count as usize;
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let mut id = String::with_capacity(36);
    for (index, byte) in bytes.iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            id.push('-');
        }
        use std::fmt::Write;
        write!(id, "{byte:02x}").map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?;
    }
    Ok(id)
}

#[cfg(target_os = "linux")]
fn owned_generation_failure(
    error: super::managed_service::ManagedServiceErrorKind,
) -> ComfyFailure {
    use super::managed_service::ManagedServiceErrorKind;
    no_dispatch(match error {
        ManagedServiceErrorKind::Cancelled => ComfyErrorKind::Cancelled,
        ManagedServiceErrorKind::DeadlineExceeded => ComfyErrorKind::Deadline,
        _ => ComfyErrorKind::InvalidOwnership,
    })
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
    #[cfg(target_os = "linux")]
    private_engine: Option<super::native_http::PinnedEngineSocket>,
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
            #[cfg(target_os = "linux")]
            private_engine: None,
        })
    }

    /// Explicit private helper selection, not lifecycle/admission proof.
    #[cfg(target_os = "linux")]
    pub fn new_private_engine(
        directory: &std::path::Path,
        api_key: Option<&str>,
        contract: ComfyContract,
    ) -> Result<Self, ComfyFailure> {
        let mut adapter = Self::new("http://127.0.0.1", api_key, contract)?;
        adapter.private_engine = Some(
            super::native_http::PinnedEngineSocket::open(directory)
                .map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?,
        );
        Ok(adapter)
    }

    pub fn prepare(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        fresh_uuid: &str,
    ) -> Result<PreparedComfyJob, ComfyFailure> {
        self.prepare_inner(catalog, request, None, fresh_uuid)
    }

    /// Prepare against the exact route already accepted by the coordinator.
    /// The full route is retained and checked again immediately before POST.
    pub fn prepare_for_route(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        accepted_route: &ResolvedProfileRoute,
        fresh_uuid: &str,
    ) -> Result<PreparedComfyJob, ComfyFailure> {
        self.prepare_inner(catalog, request, Some(accepted_route), fresh_uuid)
    }

    fn prepare_inner(
        &self,
        catalog: &Catalog,
        request: &InferenceRequest,
        accepted_route: Option<&ResolvedProfileRoute>,
        fresh_uuid: &str,
    ) -> Result<PreparedComfyJob, ComfyFailure> {
        if !canonical_uuid(fresh_uuid) {
            return Err(no_dispatch(ComfyErrorKind::InvalidRequest));
        }
        let route = validate_workflow(catalog, request, accepted_route)?;
        Ok(PreparedComfyJob {
            owner: self.owner.clone(),
            id: fresh_uuid.to_owned(),
            catalog: catalog.clone(),
            request: request.clone(),
            accepted_route: route,
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
        let accepted_route = job.accepted_route.clone();
        self.submit_inner(job, &accepted_route, context)
    }

    /// Submit only if the caller's accepted route and the freshly resolved
    /// route still equal the route retained during preparation.
    pub fn submit_for_route(
        &self,
        job: &mut PreparedComfyJob,
        accepted_route: &ResolvedProfileRoute,
        context: &OperationContext<'_>,
    ) -> Result<(), ComfyFailure> {
        self.submit_inner(job, accepted_route, context)
    }

    /// Coordinator dispatch: consume the one-shot permit returned only after
    /// the candidate UUID was durably recorded for the active accepted route.
    /// Consuming a permit is not a native acknowledgment. Route/transport
    /// failures never manufacture another permit or retry the POST.
    pub fn submit_with_dispatch_permit(
        &self,
        job: &mut PreparedComfyJob,
        permit: super::inference::ComfyDispatchPermit,
        context: &OperationContext<'_>,
    ) -> Result<(), ComfyFailure> {
        if !permit.matches_candidate(job.id(), &job.accepted_route)
            || !permit.matches_request(permit.request_id(), &job.request, &job.accepted_route)
        {
            return Err(no_dispatch(ComfyErrorKind::InvalidOwnership));
        }
        let route = job.accepted_route.clone();
        self.submit_inner(job, &route, context)
    }

    fn submit_inner(
        &self,
        job: &mut PreparedComfyJob,
        accepted_route: &ResolvedProfileRoute,
        context: &OperationContext<'_>,
    ) -> Result<(), ComfyFailure> {
        self.submit_using_http(job, accepted_route, context, &|path, payload, context| {
            self.http(path, payload, context)
        })
    }

    fn submit_using_http(
        &self,
        job: &mut PreparedComfyJob,
        accepted_route: &ResolvedProfileRoute,
        context: &OperationContext<'_>,
        http: &ComfyHttpCall<'_>,
    ) -> Result<(), ComfyFailure> {
        self.check_owner(job, false)?;
        if job.attempted {
            return Err(no_dispatch(ComfyErrorKind::AlreadyAttempted));
        }
        active(context, DispatchDisposition::NoDispatch)?;
        if accepted_route != &job.accepted_route {
            return Err(no_dispatch(ComfyErrorKind::InvalidRequest));
        }
        validate_workflow(&job.catalog, &job.request, Some(&job.accepted_route))?;
        active(context, DispatchDisposition::NoDispatch)?;
        let InferenceRequest::Media {
            input: MediaInput::Workflow { graph, .. },
            ..
        } = &job.request
        else {
            return Err(no_dispatch(ComfyErrorKind::InvalidRequest));
        };
        let payload = serde_json::json!({"prompt_id":job.id, "prompt":graph});
        job.attempted = true;
        let (status, body) = http("/prompt", Some(&payload), context)?;
        if !status.is_success() {
            return Err(uncertain(ComfyErrorKind::Rejected));
        }
        let ack = serde_json::from_slice::<NativeJsonObject<NativeSubmitAck>>(&body)
            .map_err(|_| uncertain(ComfyErrorKind::Protocol))?
            .0;
        if ack.prompt_id.len() != 36 || ack.prompt_id != job.id() {
            return Err(uncertain(ComfyErrorKind::Protocol));
        }
        Ok(())
    }

    pub fn status(
        &self,
        job: &PreparedComfyJob,
        context: &OperationContext<'_>,
    ) -> Result<ComfySnapshot, ComfyFailure> {
        self.status_using_http(job, context, &|path, payload, context| {
            self.http(path, payload, context)
        })
    }

    fn status_using_http(
        &self,
        job: &PreparedComfyJob,
        context: &OperationContext<'_>,
        http: &ComfyHttpCall<'_>,
    ) -> Result<ComfySnapshot, ComfyFailure> {
        self.check_owner(job, true)?;
        let (status, body) = http(&format!("/api/jobs/{}", job.id), None, context)?;
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
        self.cancel_using_http(job, context, &|path, payload, context| {
            self.http(path, payload, context)
        })
    }

    fn cancel_using_http(
        &self,
        job: &PreparedComfyJob,
        context: &OperationContext<'_>,
        http: &ComfyHttpCall<'_>,
    ) -> Result<bool, ComfyFailure> {
        self.check_owner(job, true)?;
        let (status, body) = http(
            &format!("/api/jobs/{}/cancel", job.id),
            Some(&serde_json::json!({})),
            context,
        )?;
        if !status.is_success() {
            return Err(uncertain(ComfyErrorKind::Rejected));
        }
        let response = serde_json::from_slice::<NativeJsonObject<NativeCancelAck>>(&body)
            .map_err(|_| uncertain(ComfyErrorKind::Protocol))?
            .0;
        Ok(response.cancelled)
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
    ) -> Result<(reqwest::StatusCode, Vec<u8>), ComfyFailure> {
        active(context, DispatchDisposition::NoDispatch)?;
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(no_dispatch(ComfyErrorKind::Configuration));
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?;
        runtime.block_on(async {
            let builder = reqwest::Client::builder()
                .retry(reqwest::retry::never())
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(context.remaining().min(Duration::from_secs(2)))
                .timeout(context.remaining())
                .default_headers(
                    auth_headers(self.api_key.as_deref())
                        .map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?,
                );
            #[cfg(target_os = "linux")]
            let builder = match &self.private_engine {
                Some(socket) => socket
                    .configure(builder)
                    .map_err(|_| no_dispatch(ComfyErrorKind::Configuration))?,
                None => builder,
            };
            let client = builder
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
            Ok((status, bytes))
        })
    }
}

fn validate_workflow(
    catalog: &Catalog,
    request: &InferenceRequest,
    accepted_route: Option<&ResolvedProfileRoute>,
) -> Result<ResolvedProfileRoute, ComfyFailure> {
    let error = no_dispatch(ComfyErrorKind::InvalidRequest);
    let refreshed = catalog.refresh().map_err(|_| error)?;
    request.validate(&refreshed).map_err(|_| error)?;
    let current = resolve_profile_route(&refreshed, request.profile()).map_err(|_| error)?;
    if current.runtime.lane != Lane::Media
        || current.runtime.as_str() != "ComfyUI"
        || accepted_route.is_some_and(|accepted| accepted != &current)
        || !matches!(request, InferenceRequest::Media { input: MediaInput::Workflow { graph, .. }, .. } if graph.as_object().is_some_and(|o| !o.is_empty()))
    {
        return Err(error);
    }
    Ok(current)
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

#[derive(Deserialize)]
struct NativeSubmitAck {
    prompt_id: String,
}

#[derive(Deserialize)]
struct NativeCancelAck {
    cancelled: bool,
}

/// Deserialize a typed DTO only from a JSON object, never from a sequence.
/// The wrapped struct still performs field-level duplicate/type validation.
struct NativeJsonObject<T>(T);

impl<'de, T> Deserialize<'de> for NativeJsonObject<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ObjectVisitor<T>(std::marker::PhantomData<fn() -> T>);

        impl<'de, T> Visitor<'de> for ObjectVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = NativeJsonObject<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                T::deserialize(MapAccessDeserializer::new(map)).map(NativeJsonObject)
            }
        }

        deserializer.deserialize_map(ObjectVisitor(std::marker::PhantomData))
    }
}

/// Typed representation of the installed `/api/jobs/{id}` response. Unknown
/// source metadata such as priority, timestamps, workflow, and preview fields
/// is intentionally ignored; semantic fields remain typed so duplicates and
/// wrong JSON shapes fail closed.
#[derive(Deserialize)]
struct NativeJobStatus {
    id: String,
    status: String,
    #[serde(default)]
    execution_status: Option<NativeJsonObject<NativeExecutionStatus>>,
    #[serde(default)]
    outputs: Option<NativeObject<NativeObject<NativeOutputField>>>,
}

#[derive(Deserialize)]
struct NativeExecutionStatus {
    #[serde(default)]
    status_str: Option<String>,
    #[serde(default)]
    completed: Option<bool>,
    #[serde(default)]
    messages: Option<NativeMessages>,
}

struct NativeMessages(Vec<NativeStatusMessage>);

impl<'de> Deserialize<'de> for NativeMessages {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NativeMessagesVisitor;

        impl<'de> Visitor<'de> for NativeMessagesVisitor {
            type Value = NativeMessages;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a bounded Comfy execution message list")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut messages = Vec::new();
                while let Some(message) = sequence.next_element::<NativeStatusMessage>()? {
                    if messages.len() >= MAX_NATIVE_MESSAGES {
                        return Err(A::Error::custom("too many native execution messages"));
                    }
                    messages.push(message);
                }
                Ok(NativeMessages(messages))
            }
        }

        deserializer.deserialize_seq(NativeMessagesVisitor)
    }
}

struct NativeStatusMessage(String);

impl<'de> Deserialize<'de> for NativeStatusMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NativeStatusMessageVisitor;

        impl<'de> Visitor<'de> for NativeStatusMessageVisitor {
            type Value = NativeStatusMessage;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a two-element Comfy execution message")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let event = sequence
                    .next_element::<String>()?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                sequence
                    .next_element::<NativeObject<IgnoredAny>>()?
                    .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                if sequence.next_element::<IgnoredAny>()?.is_some() {
                    return Err(A::Error::invalid_length(3, &self));
                }
                Ok(NativeStatusMessage(event))
            }
        }

        deserializer.deserialize_seq(NativeStatusMessageVisitor)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum NativeOutputItem {
    File(NativeJsonObject<NativeOutputFile>),
    Text(String),
}

#[derive(Deserialize)]
struct NativeOutputFile {
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    subfolder: Option<String>,
    #[serde(default, rename = "type")]
    storage_type: Option<String>,
}

enum NativeOutputField {
    Items(Vec<NativeOutputItem>),
    Other,
}

impl<'de> Deserialize<'de> for NativeOutputField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NativeOutputFieldVisitor;

        impl<'de> Visitor<'de> for NativeOutputFieldVisitor {
            type Value = NativeOutputField;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a Comfy output list or non-list output metadata")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items = Vec::new();
                while let Some(item) = sequence.next_element::<NativeOutputItem>()? {
                    if items.len() >= MAX_NATIVE_OUTPUT_ITEMS {
                        return Err(A::Error::custom("too many native output items"));
                    }
                    items.push(item);
                }
                Ok(NativeOutputField::Items(items))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                while map.next_entry::<String, IgnoredAny>()?.is_some() {}
                Ok(NativeOutputField::Other)
            }

            fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }

            fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }

            fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }

            fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }

            fn visit_str<E>(self, _: &str) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }

            fn visit_string<E>(self, _: String) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(NativeOutputField::Other)
            }
        }

        deserializer.deserialize_any(NativeOutputFieldVisitor)
    }
}

/// A JSON object that rejects duplicate keys and has a bounded entry count.
/// Used for dynamic node/media maps and message payload objects.
struct NativeObject<T>(BTreeMap<String, T>);

impl<'de, T> Deserialize<'de> for NativeObject<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NativeObjectVisitor<T>(std::marker::PhantomData<fn() -> T>);

        impl<'de, T> Visitor<'de> for NativeObjectVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = NativeObject<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object with unique keys")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, T>()? {
                    if entries.len() >= MAX_NATIVE_OBJECT_ENTRIES {
                        return Err(A::Error::custom("too many native object entries"));
                    }
                    if entries.insert(key, value).is_some() {
                        return Err(A::Error::custom("duplicate native object key"));
                    }
                }
                Ok(NativeObject(entries))
            }
        }

        deserializer.deserialize_map(NativeObjectVisitor(std::marker::PhantomData))
    }
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

fn parse_snapshot(body: &[u8], id: &str) -> Result<ComfySnapshot, ComfyFailure> {
    let error = uncertain(ComfyErrorKind::Protocol);
    let body = serde_json::from_slice::<NativeJsonObject<NativeJobStatus>>(body)
        .map_err(|_| uncertain(ComfyErrorKind::Protocol))?
        .0;
    if body.id.len() != 36 || !canonical_uuid(&body.id) || body.id != id || body.status.len() > 32 {
        return Err(error);
    }
    let state = match body.status.as_str() {
        "pending" => ComfyJobState::Pending,
        "in_progress" => ComfyJobState::Running,
        "completed" => ComfyJobState::Completed,
        "failed" => ComfyJobState::Failed,
        "cancelled" => ComfyJobState::Cancelled,
        _ => return Err(error),
    };
    if matches!(
        state,
        ComfyJobState::Completed | ComfyJobState::Failed | ComfyJobState::Cancelled
    ) {
        let native = &body.execution_status.as_ref().ok_or(error)?.0;
        let success = state == ComfyJobState::Completed;
        let status_str = native.status_str.as_deref().ok_or(error)?;
        let completed = native.completed.ok_or(error)?;
        let messages = &native.messages.as_ref().ok_or(error)?.0;
        if status_str.len() > 32
            || completed != success
            || status_str != if success { "success" } else { "error" }
        {
            return Err(error);
        }
        if messages.iter().any(|message| {
            message.0.is_empty() || message.0.len() > 128 || message.0.chars().any(char::is_control)
        }) {
            return Err(error);
        }
        let interrupted = messages
            .iter()
            .any(|message| message.0 == "execution_interrupted");
        if interrupted != (state == ComfyJobState::Cancelled) {
            return Err(error);
        }
    }
    let mut artifacts = Vec::new();
    if let Some(outputs) = body.outputs {
        for (node_id, node) in outputs.0 {
            if node_id.is_empty() || node_id.len() > 128 || node_id.chars().any(char::is_control) {
                return Err(error);
            }
            for (media_type, output) in node.0 {
                if media_type.is_empty()
                    || media_type.len() > 128
                    || media_type.chars().any(char::is_control)
                {
                    return Err(error);
                }
                let NativeOutputField::Items(items) = output else {
                    continue;
                };
                for item in items {
                    let item = match item {
                        NativeOutputItem::File(item) => item,
                        // Text outputs are valid native metadata, but are not
                        // returned as downloadable artifacts. Keep them
                        // bounded by the same response limit and discard them.
                        NativeOutputItem::Text(text) => {
                            if text.len() > MAX_RESPONSE {
                                return Err(uncertain(ComfyErrorKind::TooLarge));
                            }
                            continue;
                        }
                    };
                    let Some(filename) = item.0.filename.as_deref() else {
                        continue;
                    };
                    let subfolder = item.0.subfolder.as_deref().ok_or(error)?;
                    let storage_type = item.0.storage_type.as_deref().ok_or(error)?;
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
                        filename: filename.to_owned(),
                        subfolder: subfolder.to_owned(),
                        storage_type: storage_type.to_owned(),
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

    #[cfg(target_os = "linux")]
    #[test]
    fn managed_candidate_is_canonical_v4_and_respects_cancelled_context() {
        let id = fresh_comfy_candidate(&context()).unwrap();
        assert!(canonical_uuid(&id));
        assert_eq!(id.as_bytes()[14], b'4');
        assert!(matches!(id.as_bytes()[19], b'8' | b'9' | b'a' | b'b'));
        struct Cancelled;
        impl Cancellation for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let cancelled = OperationContext::new(Instant::now() + Duration::from_secs(2), &Cancelled);
        assert_eq!(
            fresh_comfy_candidate(&cancelled),
            Err(no_dispatch(ComfyErrorKind::Cancelled))
        );
    }

    #[test]
    fn shared_submit_logic_never_retries_an_uncertain_transport() {
        let adapter = adapter("http://127.0.0.1:1");
        let mut job = prepared(&adapter);
        let route = job.accepted_route.clone();
        let calls = std::cell::Cell::new(0);
        let http = |path: &str, payload: Option<&Value>, _: &OperationContext<'_>| {
            calls.set(calls.get() + 1);
            assert_eq!(path, "/prompt");
            assert_eq!(payload.unwrap()["prompt_id"], ID);
            Err(uncertain(ComfyErrorKind::Transport))
        };
        assert_eq!(
            adapter.submit_using_http(&mut job, &route, &context(), &http),
            Err(uncertain(ComfyErrorKind::Transport))
        );
        assert_eq!(
            adapter.submit_using_http(&mut job, &route, &context(), &http),
            Err(no_dispatch(ComfyErrorKind::AlreadyAttempted))
        );
        assert_eq!(calls.get(), 1);
    }
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
    fn snapshot(value: &Value) -> Result<ComfySnapshot, ComfyFailure> {
        parse_snapshot(&serde_json::to_vec(value).unwrap(), ID)
    }
    type Recorded = Vec<(String, Value)>;
    fn fixture(
        replies: Vec<(u16, String, Duration)>,
    ) -> (String, std::thread::JoinHandle<Recorded>) {
        fixture_with_history(replies, None)
    }
    fn fixture_with_history(
        replies: Vec<(u16, String, Duration)>,
        history: Option<std::path::PathBuf>,
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
                if header.starts_with("POST /prompt ") {
                    if let Some(history) = &history {
                        let rows: Value =
                            serde_json::from_slice(&std::fs::read(history).unwrap()).unwrap();
                        let row = rows
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|row| row["request_id"] == "durable-comfy")
                            .unwrap();
                        assert_eq!(row["comfy_dispatch_candidate"], ID);
                        assert!(row.get("backend_id").is_none());
                        assert!(!rows.to_string().contains("SENSITIVE_PROMPT"));
                    }
                }
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
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
            recorded
        });
        (endpoint, server)
    }
    fn reply(body: Value) -> (u16, String, Duration) {
        (200, body.to_string(), Duration::ZERO)
    }

    #[test]
    fn coordinator_candidate_is_durable_before_post_and_lost_ack_is_not_replayed() {
        use super::super::catalog::RuntimeIdentity;
        use super::super::inference::CoordinatorCore;
        use super::super::runtime::ResourceClaim;

        for lost_ack in [false, true] {
            let data = tempfile::tempdir().unwrap();
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let catalog = Catalog::load(root).unwrap();
            let mut core = CoordinatorCore::load(
                catalog.clone(),
                BTreeMap::from([(
                    RuntimeIdentity::new(Lane::Media, "ComfyUI").unwrap(),
                    ResourceClaim::ExclusiveGpu(0),
                )]),
                data.path(),
                root,
                8,
            )
            .unwrap();
            // Fixture reconciliation is not a native owner/readiness proof.
            core.confirm_reconciled().unwrap();
            let job_id = core
                .submit(
                    "durable-comfy".into(),
                    request(),
                    Duration::from_secs(10),
                    Instant::now(),
                )
                .unwrap();
            let ticket = core.reserve_next(Instant::now()).unwrap().unwrap();
            core.begin_dispatch(&ticket, Instant::now()).unwrap();
            let permit = core
                .record_comfy_dispatch_candidate(&ticket, ID.into(), Instant::now())
                .unwrap();
            let replies = if lost_ack {
                vec![
                    reply(serde_json::json!({"prompt_id":"wrong-ack"})),
                    reply(serde_json::json!({"id":ID,"status":"pending"})),
                ]
            } else {
                vec![reply(serde_json::json!({"prompt_id":ID}))]
            };
            let (endpoint, server) =
                fixture_with_history(replies, Some(data.path().join("jobs.json")));
            let adapter = adapter(&endpoint);
            let mut prepared = adapter
                .prepare_for_route(&catalog, &ticket.request, &ticket.route, ID)
                .unwrap();
            let outcome = adapter.submit_with_dispatch_permit(&mut prepared, permit, &context());
            if lost_ack {
                assert_eq!(
                    outcome.unwrap_err().disposition,
                    DispatchDisposition::Uncertain
                );
                let observation = adapter.status(&prepared, &context()).unwrap();
                assert_eq!(observation.state, ComfyJobState::Pending);
            } else {
                outcome.unwrap();
            }
            // Candidate persistence alone never sets an acknowledged backend ID.
            let rows: Value =
                serde_json::from_slice(&std::fs::read(data.path().join("jobs.json")).unwrap())
                    .unwrap();
            assert!(rows.as_array().unwrap()[0].get("backend_id").is_none());
            core.attach_backend(job_id, ID.into()).unwrap();
            assert!(core
                .record_comfy_dispatch_candidate(&ticket, ID.into(), Instant::now())
                .is_err());
            assert_eq!(
                adapter.submit(&mut prepared, &context()).unwrap_err().kind,
                ComfyErrorKind::AlreadyAttempted
            );
            let recorded = server.join().unwrap();
            assert_eq!(
                recorded
                    .iter()
                    .filter(|(header, _)| header.starts_with("POST /prompt "))
                    .count(),
                1
            );
            assert_eq!(recorded.len(), if lost_ack { 2 } else { 1 });
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn coordinator_permit_cannot_submit_another_workflow_or_candidate() {
        use super::super::catalog::RuntimeIdentity;
        use super::super::inference::CoordinatorCore;
        use super::super::runtime::ResourceClaim;

        for changed_graph in [false, true] {
            let data = tempfile::tempdir().unwrap();
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let catalog = Catalog::load(root).unwrap();
            let mut core = CoordinatorCore::load(
                catalog.clone(),
                BTreeMap::from([(
                    RuntimeIdentity::new(Lane::Media, "ComfyUI").unwrap(),
                    ResourceClaim::ExclusiveGpu(0),
                )]),
                data.path(),
                root,
                8,
            )
            .unwrap();
            core.confirm_reconciled().unwrap();
            core.submit(
                "durable-comfy".into(),
                request(),
                Duration::from_secs(10),
                Instant::now(),
            )
            .unwrap();
            let ticket = core.reserve_next(Instant::now()).unwrap().unwrap();
            core.begin_dispatch(&ticket, Instant::now()).unwrap();
            let permit = core
                .record_comfy_dispatch_candidate(&ticket, ID.into(), Instant::now())
                .unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let adapter = adapter(&format!("http://{}", listener.local_addr().unwrap()));
            let mut other_request = request();
            if changed_graph {
                let InferenceRequest::Media {
                    input: MediaInput::Workflow { graph, .. },
                    ..
                } = &mut other_request
                else {
                    panic!("fixture must be a workflow")
                };
                *graph = serde_json::json!({"2":{"class_type":"Other","inputs":{}}});
            }
            let other_id = if changed_graph {
                ID
            } else {
                "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            };
            let mut prepared = adapter
                .prepare_for_route(&catalog, &other_request, &ticket.route, other_id)
                .unwrap();
            let error = adapter
                .submit_with_dispatch_permit(&mut prepared, permit, &context())
                .unwrap_err();
            assert_eq!(error.disposition, DispatchDisposition::NoDispatch);
            assert_eq!(error.kind, ComfyErrorKind::InvalidOwnership);
            assert!(!prepared.submission_attempted());
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
            assert!(core
                .record_comfy_dispatch_candidate(&ticket, ID.into(), Instant::now())
                .is_err());
            assert_eq!(core.active_count(), 1);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn coordinator_permit_is_revoked_before_post_when_its_owner_or_reservation_ends() {
        use super::super::catalog::RuntimeIdentity;
        use super::super::inference::{Completion, CoordinatorCore};
        use super::super::runtime::ResourceClaim;

        for action in ["drop", "quarantine", "cancel", "complete", "deadline"] {
            let data = tempfile::tempdir().unwrap();
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let catalog = Catalog::load(root).unwrap();
            let mut core = CoordinatorCore::load(
                catalog.clone(),
                BTreeMap::from([(
                    RuntimeIdentity::new(Lane::Media, "ComfyUI").unwrap(),
                    ResourceClaim::ExclusiveGpu(0),
                )]),
                data.path(),
                root,
                8,
            )
            .unwrap();
            core.confirm_reconciled().unwrap();
            let job_id = core
                .submit(
                    "durable-comfy".into(),
                    request(),
                    Duration::from_secs(1),
                    Instant::now(),
                )
                .unwrap();
            let ticket = core.reserve_next(Instant::now()).unwrap().unwrap();
            core.begin_dispatch(&ticket, Instant::now()).unwrap();
            let permit = core
                .record_comfy_dispatch_candidate(&ticket, ID.into(), Instant::now())
                .unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let adapter = adapter(&format!("http://{}", listener.local_addr().unwrap()));
            let mut prepared = adapter
                .prepare_for_route(&catalog, &ticket.request, &ticket.route, ID)
                .unwrap();
            match action {
                "drop" => drop(core),
                "quarantine" => core.quarantine(),
                "cancel" => core.cancel_request("durable-comfy").unwrap(),
                "complete" => {
                    core.complete(job_id, Completion::Failed, Vec::new(), Instant::now())
                        .unwrap();
                }
                "deadline" => std::thread::sleep(
                    ticket.deadline.saturating_duration_since(Instant::now())
                        + Duration::from_millis(5),
                ),
                _ => unreachable!(),
            }
            let error = adapter
                .submit_with_dispatch_permit(&mut prepared, permit, &context())
                .unwrap_err();
            assert_eq!(
                error.disposition,
                DispatchDisposition::NoDispatch,
                "{action}"
            );
            assert!(!prepared.submission_attempted(), "{action}");
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn private_socket_submission_keeps_owned_native_id_and_one_shot_semantics() {
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
        let adapter = ComfyAdapter::new_private_engine(
            directory.path(),
            None,
            ComfyContract::CanonicalIdsAndAtomicScopedCancellation,
        )
        .unwrap();
        let mut job = prepared(&adapter);
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "private submission not received");
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
            let mut header = Vec::new();
            let mut byte = [0];
            while !header.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                header.push(byte[0]);
                assert!(header.len() <= 16 * 1024);
            }
            let header = String::from_utf8(header).unwrap();
            assert!(header.starts_with("POST /prompt HTTP/1.1\r\n"));
            let length = header
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            assert!(length <= 1024 * 1024 + 256);
            let mut body = vec![0; length];
            stream.read_exact(&mut body).unwrap();
            let payload: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(payload["prompt_id"], ID);
            assert_eq!(payload["prompt"]["1"]["inputs"]["text"], "SENSITIVE_PROMPT");
            let response = serde_json::json!({"prompt_id":ID}).to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).unwrap();
        });
        adapter.submit(&mut job, &context()).unwrap();
        assert!(job.submission_attempted());
        assert_eq!(
            adapter.submit(&mut job, &context()).unwrap_err(),
            no_dispatch(ComfyErrorKind::AlreadyAttempted)
        );
        server.join().unwrap();
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

    #[test]
    fn duplicate_native_ack_fields_are_rejected_without_replay() {
        let duplicate_prompt_id = format!(r#"{{"prompt_id":"{ID}","prompt_id":"{ID}"}}"#);
        let (endpoint, server) = fixture(vec![
            (200, duplicate_prompt_id, Duration::ZERO),
            (
                200,
                r#"{"cancelled":true,"cancelled":false}"#.into(),
                Duration::ZERO,
            ),
        ]);
        let adapter = adapter(&endpoint);
        let mut job = prepared(&adapter);
        assert_eq!(
            adapter.submit(&mut job, &context()).unwrap_err(),
            uncertain(ComfyErrorKind::Protocol)
        );
        assert!(job.submission_attempted());
        assert_eq!(
            adapter.submit(&mut job, &context()).unwrap_err().kind,
            ComfyErrorKind::AlreadyAttempted
        );
        assert_eq!(
            adapter.cancel(&job, &context()).unwrap_err(),
            uncertain(ComfyErrorKind::Protocol)
        );
        assert_eq!(server.join().unwrap().len(), 2);
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
            assert_eq!(snapshot(&terminal(label)).unwrap().state, state);
        }
        for label in ["pending", "in_progress"] {
            assert!(snapshot(&serde_json::json!({"id":ID,"status":label})).is_ok());
        }
        assert!(snapshot(&serde_json::json!({"id":ID,"status":"completed"})).is_err());
        let mut bad = terminal("completed");
        bad["id"] = Value::String("wrong".into());
        assert!(snapshot(&bad).is_err());
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
            assert!(snapshot(&bad).is_err());
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
        assert!(snapshot(&bad).is_err());
        let mut many = terminal("completed");
        many["outputs"] = serde_json::json!({"9":{"images":vec![serde_json::json!({"filename":"a.png","subfolder":"","type":"output"});129]}});
        assert_eq!(snapshot(&many).unwrap_err().kind, ComfyErrorKind::TooLarge);
    }

    #[test]
    fn native_status_dtos_reject_duplicate_semantics_and_wrong_shapes() {
        for body in [
            format!(r#"{{"id":"{ID}","id":"{ID}","status":"pending"}}"#),
            format!(r#"{{"id":"{ID}","status":"pending","status":"in_progress"}}"#),
            format!(r#"{{"id":"{ID}","status":"pending","outputs":[]}}"#),
            format!(r#"{{"id":"{ID}","status":"pending","execution_status":[]}}"#),
            format!(r#"{{"id":["{ID}"],"status":"pending"}}"#),
            format!(
                r#"{{"id":"{ID}","status":"completed","execution_status":{{"status_str":"success","completed":true,"completed":false,"messages":[]}}}}"#
            ),
            format!(
                r#"{{"id":"{ID}","status":"completed","execution_status":["success",true,[]]}}"#
            ),
            format!(
                r#"{{"id":"{ID}","status":"completed","execution_status":{{"status_str":"success","completed":true,"messages":[["execution_success",[]]]}}}}"#
            ),
            format!(
                r#"{{"id":"{ID}","status":"completed","execution_status":{{"status_str":"success","completed":true,"messages":[["execution_success",{{}}]]}},"outputs":{{"9":{{"images":[["a.png","","output"]]}}}}}}"#
            ),
        ] {
            assert_eq!(
                parse_snapshot(body.as_bytes(), ID).unwrap_err(),
                uncertain(ComfyErrorKind::Protocol)
            );
        }
        assert_eq!(
            parse_snapshot(format!(r#"["{ID}","pending"]"#).as_bytes(), ID).unwrap_err(),
            uncertain(ComfyErrorKind::Protocol)
        );

        assert!(
            serde_json::from_str::<NativeJsonObject<NativeSubmitAck>>(&format!(r#"["{ID}"]"#))
                .is_err()
        );
        assert!(serde_json::from_str::<NativeJsonObject<NativeCancelAck>>("[true]").is_err());

        let duplicate_artifact = format!(
            r#"{{"id":"{ID}","status":"completed","execution_status":{{"status_str":"success","completed":true,"messages":[["execution_success",{{}}]]}},"outputs":{{"9":{{"images":[{{"filename":"first.png","filename":"second.png","subfolder":"","type":"output"}}]}}}}}}"#
        );
        assert_eq!(
            parse_snapshot(duplicate_artifact.as_bytes(), ID).unwrap_err(),
            uncertain(ComfyErrorKind::Protocol)
        );

        let duplicate_execution = format!(
            r#"{{"id":"{ID}","status":"completed","execution_status":{{"status_str":"success","status_str":"error","completed":true,"messages":[]}}}}"#
        );
        assert_eq!(
            parse_snapshot(duplicate_execution.as_bytes(), ID).unwrap_err(),
            uncertain(ComfyErrorKind::Protocol)
        );

        let mut installed = terminal("completed");
        installed["priority"] = serde_json::json!([0.5, ID]);
        installed["outputs"] = serde_json::json!({
            "9": {
                "images": [{
                    "filename": "out.png",
                    "subfolder": "run",
                    "type": "output",
                    "mediaType": "images",
                    "extension_metadata": {"retained_by_native": true}
                }],
                "animated": true
            }
        });
        let parsed = snapshot(&installed).unwrap();
        assert_eq!(parsed.state, ComfyJobState::Completed);
        assert_eq!(parsed.artifacts.len(), 1);
        assert_eq!(parsed.artifacts[0].filename, "out.png");
    }

    #[test]
    fn caller_and_refreshed_full_route_must_match_before_submit() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let media = tempfile::tempdir().unwrap();
        let manifest = media.path().join("media-runtimes.json");
        let original = std::fs::read(root.join("toolkits/l2m2/media-runtimes.json")).unwrap();
        std::fs::write(&manifest, &original).unwrap();
        let catalog =
            Catalog::load_from_roots(root, media.path(), root.join("toolkits/l3ds")).unwrap();
        let request = request();
        let route = resolve_profile_route(&catalog, request.profile()).unwrap();
        let adapter = adapter("http://127.0.0.1:9");
        let mut job = adapter
            .prepare_for_route(&catalog, &request, &route, ID)
            .unwrap();

        let mut caller_route_mismatch = route.clone();
        caller_route_mismatch.tasks.clear();
        assert_eq!(
            adapter
                .submit_for_route(&mut job, &caller_route_mismatch, &context())
                .unwrap_err(),
            no_dispatch(ComfyErrorKind::InvalidRequest)
        );
        assert!(!job.submission_attempted());

        let mut native: Value = serde_json::from_slice(&original).unwrap();
        let profile = native["profiles"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|profile| profile["id"] == "qwen-image-2.1")
            .unwrap();
        profile["script"] = Value::String("media-models/changed-route.sh".into());
        std::fs::write(&manifest, native.to_string()).unwrap();
        assert_eq!(
            adapter
                .submit_for_route(&mut job, &route, &context())
                .unwrap_err(),
            no_dispatch(ComfyErrorKind::InvalidRequest)
        );
        assert!(!job.submission_attempted());
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
