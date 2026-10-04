//! Conservative ownership of one coordinator-created Linux transient service.
//!
//! This is intentionally not a RuntimeAdapter and is not wired into the
//! daemon. Callers persist the prepared intent before dispatch. A local
//! systemd-run timeout can happen after the manager accepted the start, so all
//! post-spawn failures retain that intent for explicit reconciliation. No
//! failure path stops a unit by name.
//!
//! The owner is a uniquely named transient user service started with
//! KillMode=control-group and Restart=no. A handle is issued only after
//! systemd reports the expected transient marker, active state, InvocationID,
//! and cgroup. This is not proof of GPU release, ingress closure, admission,
//! or runtime health.
//!
//! Callers serialize coordinator operations with the shared host lock. That
//! lock does not fence another same-user systemd client; no external actor may
//! create, restart, or replace these transient units concurrently.
//!
//! Stop is serialized with other coordinator operations by the host lock.
//! systemctl addresses a unit name, not an InvocationID, so the generation
//! check immediately before stop is not atomic against an external same-user
//! restart or replacement. This primitive assumes no such concurrent manager
//! operation. It does not independently prove that a cgroup is empty.

use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use super::process::{self, ProcessLimits};
use super::runtime::{OperationContext, RecoveryPolicy};

const SYSTEMD_RUN: &str = "/usr/bin/systemd-run";
const SYSTEMCTL: &str = "/usr/bin/systemctl";
const MAX_COORDINATOR_ID_BYTES: usize = 32;
const MAX_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_ARGUMENTS: usize = 4096;
const MAX_AGGREGATE_ARGUMENT_BYTES: usize = 256 * 1024;
const MAX_MANAGER_OUTPUT_BYTES: usize = 32 * 1024;
const UNIT_RANDOM_BYTES: usize = 16;
const MAX_STOP_TIMEOUT: Duration = Duration::from_secs(300);
const POLL_QUANTUM: Duration = Duration::from_millis(25);

/// Fixed, redacted failure categories. Command output, argv, and OS errors are
/// never retained in this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedServiceErrorKind {
    InvalidCoordinatorId,
    InvalidRecipe,
    InvalidStopTimeout,
    RandomnessUnavailable,
    ExecutableUnavailable,
    ManagerCommandFailed,
    UnitAlreadyExists,
    UnitStateUnverifiable,
    OwnershipMismatch,
    ServiceNotActive,
    StopUnconfirmed,
    DeadlineExceeded,
    Cancelled,
}

impl fmt::Display for ManagedServiceErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidCoordinatorId => "invalid coordinator identity",
            Self::InvalidRecipe => "invalid service launch recipe",
            Self::InvalidStopTimeout => "invalid service stop timeout",
            Self::RandomnessUnavailable => "unique service identity is unavailable",
            Self::ExecutableUnavailable => "required service executable is unavailable",
            Self::ManagerCommandFailed => "service manager operation failed",
            Self::UnitAlreadyExists => "generated service name is already in use",
            Self::UnitStateUnverifiable => "service ownership state could not be verified",
            Self::OwnershipMismatch => "service ownership proof does not match",
            Self::ServiceNotActive => "owned service is not confirmed active",
            Self::StopUnconfirmed => "service stop could not be confirmed",
            Self::DeadlineExceeded => "service operation deadline expired",
            Self::Cancelled => "service operation was cancelled",
        })
    }
}

impl Error for ManagedServiceErrorKind {}

/// Bounded immutable argv. The executable is resolved and checked when the
/// recipe is created; no shell command string or environment override exists.
pub struct LaunchRecipe {
    executable: PathBuf,
    arguments: Vec<OsString>,
}

impl LaunchRecipe {
    pub fn new(
        executable: impl AsRef<Path>,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let path = executable.as_ref();
        if !path.is_absolute() {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }
        let executable = path
            .canonicalize()
            .map_err(|_| ManagedServiceErrorKind::InvalidRecipe)?;
        let metadata = executable
            .metadata()
            .map_err(|_| ManagedServiceErrorKind::InvalidRecipe)?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
            return Err(ManagedServiceErrorKind::InvalidRecipe);
        }

        let mut collected = Vec::new();
        let mut total = executable.as_os_str().as_bytes().len();
        for argument in arguments {
            let bytes = argument.as_os_str().as_bytes();
            if bytes.len() > MAX_ARGUMENT_BYTES
                || bytes.contains(&0)
                || collected.len() >= MAX_ARGUMENTS
            {
                return Err(ManagedServiceErrorKind::InvalidRecipe);
            }
            total = total.saturating_add(bytes.len());
            if total > MAX_AGGREGATE_ARGUMENT_BYTES {
                return Err(ManagedServiceErrorKind::InvalidRecipe);
            }
            collected.push(argument);
        }
        Ok(Self {
            executable,
            arguments: collected,
        })
    }

    /// Stable opaque identity for the immutable executable and argv, suitable
    /// for comparing a persisted intent with a refreshed recipe. It does not
    /// expose the command or its arguments.
    pub fn launch_identity(&self) -> String {
        launch_fingerprint(self)
    }
}

impl fmt::Debug for LaunchRecipe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LaunchRecipe(<redacted>)")
    }
}

/// Safe durable description of a start intent. This record is not authority
/// to stop anything; reconciliation still requires a live generation and
/// cgroup match.
#[derive(Clone, Eq, PartialEq)]
pub struct ServiceIntentRecord {
    coordinator_id: String,
    unit_name: String,
    launch_identity: String,
    stop_timeout_millis: u64,
}

impl ServiceIntentRecord {
    /// Rebuild a durable pending-intent record after restart. This restores
    /// data only; it does not create an owned handle or authorize any action.
    pub fn from_persisted_parts(
        coordinator_id: String,
        unit_name: String,
        launch_identity: String,
        stop_timeout_millis: u64,
    ) -> Result<Self, ManagedServiceErrorKind> {
        if coordinator_id.len() > MAX_COORDINATOR_ID_BYTES
            || unit_name.len() > 96
            || launch_identity.len() > 64
            || stop_timeout_millis == 0
            || stop_timeout_millis > MAX_STOP_TIMEOUT.as_millis() as u64
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        let record = Self {
            coordinator_id,
            unit_name,
            launch_identity,
            stop_timeout_millis,
        };
        if !valid_intent(&record) {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        Ok(record)
    }

    pub fn coordinator_id(&self) -> &str {
        &self.coordinator_id
    }

    pub fn unit_name(&self) -> &str {
        &self.unit_name
    }

    /// Stable opaque fingerprint for distinguishing immutable recipes in
    /// persisted intent records. This is metadata, not a secret or ownership
    /// proof.
    pub fn launch_identity(&self) -> &str {
        &self.launch_identity
    }

    pub fn stop_timeout_millis(&self) -> u64 {
        self.stop_timeout_millis
    }
}

impl fmt::Debug for ServiceIntentRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServiceIntentRecord")
            .field("coordinator_id", &self.coordinator_id)
            .field("unit_name", &self.unit_name)
            .field("launch_identity", &"<redacted>")
            .field("stop_timeout_millis", &self.stop_timeout_millis)
            .finish()
    }
}

/// A single-use prepared start. Persist intent_record before moving this token
/// into dispatch. The token is intentionally not Clone.
pub struct PreparedServiceStart {
    intent: ServiceIntentRecord,
    recipe: LaunchRecipe,
}

impl PreparedServiceStart {
    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }
}

impl fmt::Debug for PreparedServiceStart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedServiceStart")
            .field("intent", &self.intent)
            .field("recipe", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StartDisposition {
    /// The manager command was not attempted, or preflight rejected the name.
    NotDispatched,
    /// systemd-run may have submitted a start, so the intent must be retained.
    OwnershipUncertain,
}

/// Start failures preserve their generated intent, including timeout and
/// partial-start uncertainty.
pub struct StartFailure {
    intent: ServiceIntentRecord,
    disposition: StartDisposition,
    kind: ManagedServiceErrorKind,
}

impl StartFailure {
    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }

    pub fn disposition(&self) -> StartDisposition {
        self.disposition
    }

    pub fn kind(&self) -> ManagedServiceErrorKind {
        self.kind
    }
}

impl fmt::Debug for StartFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StartFailure")
            .field("intent", &self.intent)
            .field("disposition", &self.disposition)
            .field("kind", &self.kind)
            .finish()
    }
}

impl fmt::Display for StartFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({:?})", self.kind, self.disposition)
    }
}

impl Error for StartFailure {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedServiceState {
    Running,
    Stopped,
    Unknown,
}

/// Reconciliation describes the manager query at this instant. AbsentNow
/// does not erase the durable intent or establish resource release.
pub enum ReconcileResult {
    Owned(OwnedService),
    AbsentNow,
    Unknown,
}

impl fmt::Debug for ReconcileResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned(handle) => formatter.debug_tuple("Owned").field(handle).finish(),
            Self::AbsentNow => formatter.write_str("AbsentNow"),
            Self::Unknown => formatter.write_str("Unknown"),
        }
    }
}

/// Private, non-Clone ownership capability. The unit name alone is never
/// accepted by state or stop operations.
pub struct OwnedService {
    intent: ServiceIntentRecord,
    invocation_id: String,
    cgroup: String,
}

impl OwnedService {
    pub fn unit_name(&self) -> &str {
        &self.intent.unit_name
    }

    pub fn intent_record(&self) -> &ServiceIntentRecord {
        &self.intent
    }
}

impl fmt::Debug for OwnedService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedService")
            .field("unit_name", &self.intent.unit_name)
            .field("invocation_id", &"<redacted>")
            .field("cgroup", &"<redacted>")
            .finish()
    }
}

/// Coordinates only transient user services created from validated recipes.
pub struct ManagedServiceController {
    coordinator_id: String,
    systemd_run: PathBuf,
    systemctl: PathBuf,
}

impl ManagedServiceController {
    /// Use fixed, root-owned systemd client binaries. No PATH lookup is used.
    pub fn production(coordinator_id: impl Into<String>) -> Result<Self, ManagedServiceErrorKind> {
        let coordinator_id = coordinator_id.into();
        validate_coordinator_id(&coordinator_id)?;
        let systemd_run = PathBuf::from(SYSTEMD_RUN);
        let systemctl = PathBuf::from(SYSTEMCTL);
        verify_production_binary(&systemd_run)?;
        verify_production_binary(&systemctl)?;
        Ok(Self {
            coordinator_id,
            systemd_run,
            systemctl,
        })
    }

    /// Generate and expose the durable identifier before any manager command.
    pub fn prepare_start(
        &self,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
    ) -> Result<PreparedServiceStart, ManagedServiceErrorKind> {
        if stop_timeout.is_zero() || stop_timeout > MAX_STOP_TIMEOUT {
            return Err(ManagedServiceErrorKind::InvalidStopTimeout);
        }
        let suffix = random_unit_suffix()?;
        let unit_name = format!("l3ms-{}-{suffix}.service", self.coordinator_id);
        let launch_identity = recipe.launch_identity();
        let stop_timeout_millis = duration_millis_ceil(stop_timeout) as u64;
        Ok(PreparedServiceStart {
            intent: ServiceIntentRecord {
                coordinator_id: self.coordinator_id.clone(),
                unit_name,
                launch_identity,
                stop_timeout_millis,
            },
            recipe,
        })
    }

    /// Consume a prepared start exactly once. The caller persists the intent
    /// record first. Post-spawn failures never trigger name-only cleanup.
    pub fn dispatch(
        &self,
        prepared: PreparedServiceStart,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, StartFailure> {
        let intent = prepared.intent;
        let fail = |disposition, kind| StartFailure {
            intent: intent.clone(),
            disposition,
            kind,
        };
        if intent.coordinator_id != self.coordinator_id || !valid_intent(&intent) {
            return Err(fail(
                StartDisposition::NotDispatched,
                ManagedServiceErrorKind::OwnershipMismatch,
            ));
        }
        if let Err(kind) = context_error(context) {
            return Err(fail(StartDisposition::NotDispatched, kind));
        }
        match self.preflight_unit(&intent, context) {
            Ok(true) => {}
            Ok(false) => {
                return Err(fail(
                    StartDisposition::NotDispatched,
                    ManagedServiceErrorKind::UnitAlreadyExists,
                ))
            }
            Err(kind) => return Err(fail(StartDisposition::NotDispatched, kind)),
        }

        let description = description_for(&intent);
        let timeout = format!("{}ms", intent.stop_timeout_millis);
        let mut command = Command::new(&self.systemd_run);
        command
            .arg("--user")
            .arg("--no-ask-password")
            .arg("--quiet")
            .arg("--no-block")
            .arg("--service-type=exec")
            .arg(format!("--unit={}", intent.unit_name))
            .arg(format!("--description={description}"))
            .arg("--property=KillMode=control-group")
            .arg("--property=Restart=no")
            .arg(format!("--property=TimeoutStopSec={timeout}"))
            .arg("--")
            .arg(&prepared.recipe.executable)
            .args(&prepared.recipe.arguments);
        match execute(command, context) {
            Ok(_) => {}
            Err((kind, false)) => {
                return Err(fail(StartDisposition::NotDispatched, kind));
            }
            Err((kind, true)) => {
                return Err(fail(StartDisposition::OwnershipUncertain, kind));
            }
        }

        loop {
            if let Err(kind) = context_error(context) {
                return Err(fail(StartDisposition::OwnershipUncertain, kind));
            }
            if let Ok(Some(snapshot)) = self.show_unit(&intent, context) {
                match owned_snapshot(&intent, &snapshot, None) {
                    Ok(ManagedServiceState::Running) => {
                        let invocation_id = snapshot.invocation_id.clone();
                        let cgroup = snapshot.control_group.clone();
                        return Ok(OwnedService {
                            intent: intent.clone(),
                            invocation_id,
                            cgroup,
                        });
                    }
                    Ok(ManagedServiceState::Unknown) => {}
                    Ok(ManagedServiceState::Stopped) | Err(_) => {
                        return Err(fail(
                            StartDisposition::OwnershipUncertain,
                            ManagedServiceErrorKind::OwnershipMismatch,
                        ))
                    }
                }
            }
            if context.remaining().is_zero() {
                return Err(fail(
                    StartDisposition::OwnershipUncertain,
                    ManagedServiceErrorKind::DeadlineExceeded,
                ));
            }
            thread::sleep(POLL_QUANTUM.min(context.remaining()));
        }
    }

    /// Reconcile a pending durable intent. Only a fully matching active
    /// transient service produces a handle; this method never sends stop.
    pub fn reconcile(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<ReconcileResult, ManagedServiceErrorKind> {
        if intent.coordinator_id != self.coordinator_id || !valid_intent(intent) {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        context_error(context)?;
        let Some(snapshot) = self.show_unit(intent, context)? else {
            return Ok(ReconcileResult::AbsentNow);
        };
        match owned_snapshot(intent, &snapshot, None) {
            Ok(ManagedServiceState::Running) => Ok(ReconcileResult::Owned(OwnedService {
                intent: intent.clone(),
                invocation_id: snapshot.invocation_id,
                cgroup: snapshot.control_group,
            })),
            Ok(ManagedServiceState::Stopped | ManagedServiceState::Unknown) | Err(_) => {
                Ok(ReconcileResult::Unknown)
            }
        }
    }

    /// Query state only when the current unit still matches the handle's
    /// InvocationID and cgroup. A disappeared unit is unknown to this query.
    pub fn state(
        &self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<ManagedServiceState, ManagedServiceErrorKind> {
        self.validate_handle(service)?;
        context_error(context)?;
        let Some(snapshot) = self.show_unit(&service.intent, context)? else {
            return Ok(ManagedServiceState::Unknown);
        };
        Ok(owned_snapshot(
            &service.intent,
            &snapshot,
            Some((&service.invocation_id, &service.cgroup)),
        )
        .unwrap_or(ManagedServiceState::Unknown))
    }

    /// Stop only after the active generation matches InvocationID and cgroup.
    /// systemctl stop is name-targeted, so the coordinator host lock must
    /// serialize its callers, and external same-user restarts/replacements
    /// must not race this method. The generation check is not atomic. Success
    /// is a manager stop acknowledgment, not proof that the cgroup is empty or
    /// that a GPU/resource was released.
    pub fn stop(
        &self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<(), ManagedServiceErrorKind> {
        self.validate_handle(service)?;
        context_error(context)?;
        let Some(snapshot) = self.show_unit(&service.intent, context)? else {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        };
        match owned_snapshot(
            &service.intent,
            &snapshot,
            Some((&service.invocation_id, &service.cgroup)),
        )? {
            ManagedServiceState::Running => {}
            ManagedServiceState::Stopped | ManagedServiceState::Unknown => {
                return Err(ManagedServiceErrorKind::ServiceNotActive)
            }
        }

        let mut command = Command::new(&self.systemctl);
        command
            .arg("--user")
            .arg("--no-ask-password")
            .arg("stop")
            .arg(&service.intent.unit_name);
        execute(command, context).map_err(|(kind, _)| kind)?;

        let Some(after) = self.show_unit(&service.intent, context)? else {
            // A successful stop followed by transient-unit unloading confirms
            // that no unit is currently addressable under this generated name.
            return Ok(());
        };
        match owned_snapshot(
            &service.intent,
            &after,
            Some((&service.invocation_id, &service.cgroup)),
        )
        .unwrap_or(ManagedServiceState::Unknown)
        {
            ManagedServiceState::Stopped => Ok(()),
            ManagedServiceState::Running | ManagedServiceState::Unknown => {
                Err(ManagedServiceErrorKind::StopUnconfirmed)
            }
        }
    }

    fn validate_handle(&self, service: &OwnedService) -> Result<(), ManagedServiceErrorKind> {
        if service.intent.coordinator_id != self.coordinator_id
            || !valid_intent(&service.intent)
            || !valid_invocation_id(&service.invocation_id)
            || !valid_cgroup(&service.cgroup, &service.intent.unit_name)
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        Ok(())
    }

    fn preflight_unit(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<bool, ManagedServiceErrorKind> {
        // Check loaded units and installed unit files. Static or pre-existing
        // services are never claimed, including inactive ones.
        let loaded = self.list_unit("list-units", intent, context)?;
        if !loaded.stdout().iter().all(u8::is_ascii_whitespace) {
            return Ok(false);
        }
        let installed = self.list_unit("list-unit-files", intent, context)?;
        Ok(installed.stdout().iter().all(u8::is_ascii_whitespace))
    }

    fn list_unit(
        &self,
        verb: &str,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<process::ProcessOutput, ManagedServiceErrorKind> {
        let mut command = Command::new(&self.systemctl);
        command
            .arg("--user")
            .arg("--no-pager")
            .arg("--no-legend")
            .arg(verb)
            .arg("--all")
            .arg(&intent.unit_name);
        execute(command, context).map_err(|(kind, _)| kind)
    }

    fn show_unit(
        &self,
        intent: &ServiceIntentRecord,
        context: &OperationContext<'_>,
    ) -> Result<Option<UnitSnapshot>, ManagedServiceErrorKind> {
        let mut command = Command::new(&self.systemctl);
        command
            .arg("--user")
            .arg("--no-pager")
            .arg("show")
            .arg("--all");
        for property in [
            "LoadState",
            "ActiveState",
            "Transient",
            "InvocationID",
            "ControlGroup",
            "FragmentPath",
            "Description",
            "Restart",
            "KillMode",
        ] {
            command.arg(format!("--property={property}"));
        }
        command.arg(&intent.unit_name);
        let output = execute(command, context).map_err(|(kind, _)| kind)?;
        parse_snapshot(output.stdout())
    }
}

#[cfg(test)]
impl ManagedServiceController {
    fn with_test_executables(
        coordinator_id: impl Into<String>,
        systemd_run: PathBuf,
        systemctl: PathBuf,
    ) -> Result<Self, ManagedServiceErrorKind> {
        let coordinator_id = coordinator_id.into();
        validate_coordinator_id(&coordinator_id)?;
        verify_executable(&systemd_run)?;
        verify_executable(&systemctl)?;
        Ok(Self {
            coordinator_id,
            systemd_run,
            systemctl,
        })
    }
}

struct UnitSnapshot {
    load_state: String,
    active_state: String,
    transient: String,
    invocation_id: String,
    control_group: String,
    _fragment_path: String,
    description: String,
    restart: String,
    kill_mode: String,
}

fn parse_snapshot(stdout: &[u8]) -> Result<Option<UnitSnapshot>, ManagedServiceErrorKind> {
    let text =
        std::str::from_utf8(stdout).map_err(|_| ManagedServiceErrorKind::UnitStateUnverifiable)?;
    let mut fields = std::collections::BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            return Err(ManagedServiceErrorKind::UnitStateUnverifiable);
        };
        if fields.insert(key, value).is_some() {
            return Err(ManagedServiceErrorKind::UnitStateUnverifiable);
        }
    }
    let get = |name| {
        fields
            .get(name)
            .copied()
            .ok_or(ManagedServiceErrorKind::UnitStateUnverifiable)
    };
    let load_state = get("LoadState")?.to_owned();
    if load_state == "not-found" {
        return Ok(None);
    }
    let snapshot = UnitSnapshot {
        load_state,
        active_state: get("ActiveState")?.to_owned(),
        transient: get("Transient")?.to_owned(),
        invocation_id: get("InvocationID")?.to_owned(),
        control_group: get("ControlGroup")?.to_owned(),
        _fragment_path: get("FragmentPath")?.to_owned(),
        description: get("Description")?.to_owned(),
        restart: get("Restart")?.to_owned(),
        kill_mode: get("KillMode")?.to_owned(),
    };
    Ok(Some(snapshot))
}

fn owned_snapshot(
    intent: &ServiceIntentRecord,
    snapshot: &UnitSnapshot,
    generation: Option<(&str, &str)>,
) -> Result<ManagedServiceState, ManagedServiceErrorKind> {
    if snapshot.load_state != "loaded"
        || snapshot.transient != "yes"
        || snapshot.description != description_for(intent)
        || snapshot.restart != "no"
        || snapshot.kill_mode != "control-group"
    {
        return Err(ManagedServiceErrorKind::OwnershipMismatch);
    }
    let expected = generation;
    match snapshot.active_state.as_str() {
        // The transient job may be visible before systemd has assigned an
        // invocation ID or cgroup. Keep polling instead of failing ownership.
        "activating" | "deactivating" | "reloading" => {
            if let Some((invocation_id, cgroup)) = expected {
                if (!snapshot.invocation_id.is_empty() && snapshot.invocation_id != invocation_id)
                    || (!snapshot.control_group.is_empty() && snapshot.control_group != cgroup)
                {
                    return Err(ManagedServiceErrorKind::OwnershipMismatch);
                }
            }
            Ok(ManagedServiceState::Unknown)
        }
        "active" | "inactive" => {
            if !valid_invocation_id(&snapshot.invocation_id)
                || !valid_cgroup(&snapshot.control_group, &intent.unit_name)
            {
                return Err(ManagedServiceErrorKind::OwnershipMismatch);
            }
            if let Some((invocation_id, cgroup)) = expected {
                if snapshot.invocation_id != invocation_id || snapshot.control_group != cgroup {
                    return Err(ManagedServiceErrorKind::OwnershipMismatch);
                }
            }
            if snapshot.active_state == "active" {
                Ok(ManagedServiceState::Running)
            } else {
                Ok(ManagedServiceState::Stopped)
            }
        }
        _ => Ok(ManagedServiceState::Unknown),
    }
}

fn execute(
    command: Command,
    context: &OperationContext<'_>,
) -> Result<process::ProcessOutput, (ManagedServiceErrorKind, bool)> {
    let limits = ProcessLimits::new(0, MAX_MANAGER_OUTPUT_BYTES)
        .map_err(|_| (ManagedServiceErrorKind::ManagerCommandFailed, false))?;
    process::execute(
        command,
        None,
        limits,
        context,
        RecoveryPolicy::new(Duration::from_secs(5))
            .expect("fixed process recovery timeout is valid"),
    )
    .map_err(|failure| {
        use process::ProcessFailureDisposition as Disposition;
        let may_have_spawned = !matches!(failure.disposition(), Disposition::NoChildSpawned);
        let kind = match failure.kind() {
            process::ProcessFailureKind::Cancelled => ManagedServiceErrorKind::Cancelled,
            process::ProcessFailureKind::DeadlineExceeded => {
                ManagedServiceErrorKind::DeadlineExceeded
            }
            _ if context.is_cancelled() => ManagedServiceErrorKind::Cancelled,
            _ if context.remaining().is_zero() => ManagedServiceErrorKind::DeadlineExceeded,
            _ => ManagedServiceErrorKind::ManagerCommandFailed,
        };
        (kind, may_have_spawned)
    })
}

fn context_error(context: &OperationContext<'_>) -> Result<(), ManagedServiceErrorKind> {
    if context.is_cancelled() {
        Err(ManagedServiceErrorKind::Cancelled)
    } else if context.remaining().is_zero() {
        Err(ManagedServiceErrorKind::DeadlineExceeded)
    } else {
        Ok(())
    }
}

fn validate_coordinator_id(value: &str) -> Result<(), ManagedServiceErrorKind> {
    if value.is_empty()
        || value.len() > MAX_COORDINATOR_ID_BYTES
        || !value.as_bytes()[0].is_ascii_lowercase()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || value.ends_with('-')
    {
        return Err(ManagedServiceErrorKind::InvalidCoordinatorId);
    }
    Ok(())
}

fn valid_intent(intent: &ServiceIntentRecord) -> bool {
    if validate_coordinator_id(&intent.coordinator_id).is_err()
        || intent.launch_identity.len() != 64
        || intent.stop_timeout_millis == 0
        || intent.stop_timeout_millis > MAX_STOP_TIMEOUT.as_millis() as u64
        || !intent
            .launch_identity
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return false;
    }
    let prefix = format!("l3ms-{}-", intent.coordinator_id);
    let Some(random) = intent
        .unit_name
        .strip_prefix(&prefix)
        .and_then(|name| name.strip_suffix(".service"))
    else {
        return false;
    };
    random.len() == UNIT_RANDOM_BYTES * 2
        && random
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn description_for(intent: &ServiceIntentRecord) -> String {
    format!(
        "l3ms-managed:{}:{}:{}:{}",
        intent.coordinator_id, intent.unit_name, intent.launch_identity, intent.stop_timeout_millis
    )
}

fn valid_invocation_id(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn valid_cgroup(value: &str, unit_name: &str) -> bool {
    value.starts_with('/')
        && value.len() <= 4096
        && value.ends_with(unit_name)
        && value.split('/').skip(1).all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_.:@-".contains(&byte))
        })
}

fn launch_fingerprint(recipe: &LaunchRecipe) -> String {
    // Four independently seeded FNV-1a streams produce a stable recipe
    // fingerprint. It distinguishes persisted intents; random unit names and
    // systemd generation fields remain the ownership proof.
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut states = [
        0xcbf2_9ce4_8422_2325,
        0x8422_2325_cbf2_9ce4,
        0x9e37_79b9_7f4a_7c15,
        0xd6e8_feb8_6659_fd93,
    ];
    let mut feed = |bytes: &[u8]| {
        for state in &mut states {
            for byte in (bytes.len() as u64)
                .to_le_bytes()
                .into_iter()
                .chain(bytes.iter().copied())
            {
                *state = (*state ^ u64::from(byte)).wrapping_mul(PRIME);
            }
        }
    };
    feed(b"l3ms-launch-recipe-v1");
    feed(recipe.executable.as_os_str().as_bytes());
    for argument in &recipe.arguments {
        feed(argument.as_os_str().as_bytes());
    }
    format!(
        "{:016x}{:016x}{:016x}{:016x}",
        states[0], states[1], states[2], states[3]
    )
}

fn random_unit_suffix() -> Result<String, ManagedServiceErrorKind> {
    let mut bytes = [0u8; UNIT_RANDOM_BYTES];
    let mut offset = 0;
    while offset < bytes.len() {
        let result = unsafe {
            libc::getrandom(
                bytes[offset..].as_mut_ptr().cast(),
                bytes.len() - offset,
                libc::GRND_NONBLOCK,
            )
        };
        if result > 0 {
            offset += result as usize;
            continue;
        }
        if result < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(ManagedServiceErrorKind::RandomnessUnavailable);
    }
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use fmt::Write;
        let _ = write!(encoded, "{byte:02x}");
    }
    Ok(encoded)
}

fn duration_millis_ceil(value: Duration) -> u128 {
    value.as_millis()
        + if !value.subsec_nanos().is_multiple_of(1_000_000) {
            1
        } else {
            0
        }
}

fn verify_production_binary(path: &Path) -> Result<(), ManagedServiceErrorKind> {
    verify_executable(path)?;
    let metadata =
        fs::symlink_metadata(path).map_err(|_| ManagedServiceErrorKind::ExecutableUnavailable)?;
    if metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
        return Err(ManagedServiceErrorKind::ExecutableUnavailable);
    }
    Ok(())
}

fn verify_executable(path: &Path) -> Result<(), ManagedServiceErrorKind> {
    let metadata =
        fs::metadata(path).map_err(|_| ManagedServiceErrorKind::ExecutableUnavailable)?;
    if !path.is_absolute() || !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
        return Err(ManagedServiceErrorKind::ExecutableUnavailable);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Instant;
    use tempfile::TempDir;

    static NEVER_CANCEL: super::super::runtime::NeverCancel = super::super::runtime::NeverCancel;
    const INVOCATION: &str = "0123456789abcdef0123456789abcdef";

    struct Fixture {
        directory: TempDir,
        state: PathBuf,
        recipe_executable: PathBuf,
        controller: ManagedServiceController,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
            let state = directory.path().join("state");
            let runner = directory.path().join("systemd-run");
            let systemctl = directory.path().join("systemctl");
            let recipe_executable = directory.path().join("service");
            write_executable(&runner, "#!/bin/sh\nexit 0\n");
            write_executable(&systemctl, "#!/bin/sh\nexit 0\n");
            write_executable(&recipe_executable, "#!/bin/sh\nexit 0\n");
            let controller = ManagedServiceController::with_test_executables(
                "test-coordinator",
                runner,
                systemctl,
            )
            .unwrap();
            Self {
                directory,
                state,
                recipe_executable,
                controller,
            }
        }

        fn recipe(&self) -> LaunchRecipe {
            LaunchRecipe::new(&self.recipe_executable, [OsString::from("fixture-arg")]).unwrap()
        }

        fn install_scripts(&self, intent: &ServiceIntentRecord, launch_result: &str) {
            self.install_scripts_with_cgroup(intent, launch_result, false);
        }

        fn install_scripts_with_cgroup(
            &self,
            intent: &ServiceIntentRecord,
            launch_result: &str,
            clear_cgroup_when_inactive: bool,
        ) {
            let runner = self.directory.path().join("systemd-run");
            let runner_script = format!(
                "#!/bin/sh\nprintf active > '{}'\nexit {}\n",
                self.state.display(),
                launch_result
            );
            write_executable(&runner, &runner_script);
            let systemctl = self.directory.path().join("systemctl");
            let cgroup_output = if clear_cgroup_when_inactive {
                format!(
                    "if [ \"$active\" = inactive ]; then printf 'ControlGroup=\\n'; else printf 'ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/{}\\n'; fi",
                    intent.unit_name
                )
            } else {
                format!(
                    "printf 'ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/{}\\n'",
                    intent.unit_name
                )
            };
            let script = format!(
                r#"#!/bin/sh
verb=
for arg in "$@"; do
  case "$arg" in
    show|stop|list-units|list-unit-files) verb="$arg" ;;
  esac
done
case "$verb" in
  list-units|list-unit-files) exit 0 ;;
  stop) printf inactive > '{state}'; exit 0 ;;
  show)
    active=$(cat '{state}' 2>/dev/null || printf active)
    printf 'LoadState=loaded\n'
    printf 'ActiveState=%s\n' "$active"
    printf 'InvocationID={invocation}\n'
    {cgroup_output}
    printf 'FragmentPath=/run/user/1000/systemd/transient/{unit_name}\n'
    printf 'Transient=yes\n'
    printf 'Description=l3ms-managed:{coordinator}:{unit_name}:{identity}:{timeout}\n'
    printf 'Restart=no\nKillMode=control-group\n'
    exit 0
    ;;
esac
exit 1
"#,
                state = self.state.display(),
                invocation = INVOCATION,
                coordinator = intent.coordinator_id,
                unit_name = intent.unit_name,
                identity = intent.launch_identity,
                timeout = intent.stop_timeout_millis,
                cgroup_output = cgroup_output,
            );
            write_executable(&systemctl, &script);
        }

        fn context(&self, duration: Duration) -> OperationContext<'static> {
            OperationContext::new(Instant::now() + duration, &NEVER_CANCEL)
        }
    }

    fn write_executable(path: &Path, contents: &str) {
        fs::write(path, contents).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[test]
    fn prepared_intent_is_unique_and_available_before_dispatch() {
        let fixture = Fixture::new();
        let first = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let second = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        assert_ne!(
            first.intent_record().unit_name(),
            second.intent_record().unit_name()
        );
        assert!(first
            .intent_record()
            .unit_name()
            .starts_with("l3ms-test-coordinator-"));
        assert_eq!(first.intent_record().launch_identity().len(), 64);
        assert_eq!(
            first.intent_record().launch_identity(),
            fixture.recipe().launch_identity()
        );
        assert!(!fixture.state.exists());
    }

    #[test]
    fn dispatch_returns_handle_only_after_generation_and_cgroup_proof() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let context = fixture.context(Duration::from_secs(2));
        let handle = fixture.controller.dispatch(prepared, &context).unwrap();
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Running
        );
        fixture.controller.stop(&handle, &context).unwrap();
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Stopped
        );
    }

    #[test]
    fn inactive_snapshot_without_cgroup_does_not_claim_stop_cleanup() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts_with_cgroup(&intent, "0", true);
        let context = fixture.context(Duration::from_secs(2));
        let handle = fixture.controller.dispatch(prepared, &context).unwrap();
        assert_eq!(
            fixture.controller.stop(&handle, &context),
            Err(ManagedServiceErrorKind::StopUnconfirmed)
        );
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Unknown
        );
    }

    #[test]
    fn non_transient_unit_never_yields_an_owned_handle() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let systemctl = fixture.directory.path().join("systemctl");
        let changed = fs::read_to_string(&systemctl)
            .unwrap()
            .replace("Transient=yes", "Transient=no");
        write_executable(&systemctl, &changed);
        let context = fixture.context(Duration::from_secs(2));
        let failure = fixture.controller.dispatch(prepared, &context).unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::OwnershipUncertain);
        assert_eq!(failure.kind(), ManagedServiceErrorKind::OwnershipMismatch);
        assert_eq!(failure.intent_record(), &intent);
    }

    #[test]
    fn failed_launcher_keeps_uncertain_intent() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "1");
        let context = fixture.context(Duration::from_secs(2));
        let failure = fixture.controller.dispatch(prepared, &context).unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::OwnershipUncertain);
        assert_eq!(failure.intent_record(), &intent);
        assert_eq!(
            failure.kind(),
            ManagedServiceErrorKind::ManagerCommandFailed
        );
    }

    #[test]
    fn mismatched_generation_never_reaches_stop() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        let context = fixture.context(Duration::from_secs(2));
        let handle = fixture.controller.dispatch(prepared, &context).unwrap();
        let systemctl = fixture.directory.path().join("systemctl");
        let changed = fs::read_to_string(&systemctl)
            .unwrap()
            .replace(INVOCATION, "ffffffffffffffffffffffffffffffff");
        write_executable(&systemctl, &changed);
        assert_eq!(
            fixture.controller.stop(&handle, &context),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        );
        assert_eq!(fs::read_to_string(&fixture.state).unwrap(), "active");
    }

    #[test]
    fn refuses_preexisting_unit_file_before_dispatch() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        let systemctl = fixture.directory.path().join("systemctl");
        let script = format!(
            "#!/bin/sh\nfor arg in \"$@\"; do\n  if [ \"$arg\" = list-unit-files ]; then printf '{} disabled\\n'; fi\ndone\nexit 0\n",
            intent.unit_name
        );
        write_executable(&systemctl, &script);
        let context = fixture.context(Duration::from_secs(2));
        let failure = fixture.controller.dispatch(prepared, &context).unwrap_err();
        assert_eq!(failure.disposition(), StartDisposition::NotDispatched);
        assert_eq!(failure.kind(), ManagedServiceErrorKind::UnitAlreadyExists);
        assert!(!fixture.state.exists());
    }

    #[test]
    fn restored_intent_reconciles_to_handle_only_with_live_generation_proof() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record().clone();
        fixture.install_scripts(&intent, "0");
        fs::write(&fixture.state, "active").unwrap();
        let restored = ServiceIntentRecord::from_persisted_parts(
            intent.coordinator_id().to_owned(),
            intent.unit_name().to_owned(),
            intent.launch_identity().to_owned(),
            intent.stop_timeout_millis(),
        )
        .unwrap();
        let context = fixture.context(Duration::from_secs(2));
        let ReconcileResult::Owned(handle) =
            fixture.controller.reconcile(&restored, &context).unwrap()
        else {
            panic!("fixture should prove the matching active generation");
        };
        assert_eq!(
            fixture.controller.state(&handle, &context).unwrap(),
            ManagedServiceState::Running
        );
    }

    #[test]
    fn rejects_invalid_names_and_stop_timeouts() {
        assert_eq!(
            ManagedServiceController::with_test_executables(
                "../bad",
                PathBuf::from("/bin/true"),
                PathBuf::from("/bin/true")
            )
            .err(),
            Some(ManagedServiceErrorKind::InvalidCoordinatorId)
        );
        let fixture = Fixture::new();
        assert_eq!(
            fixture
                .controller
                .prepare_start(fixture.recipe(), Duration::ZERO)
                .err(),
            Some(ManagedServiceErrorKind::InvalidStopTimeout)
        );
    }

    #[test]
    fn persisted_intent_restore_is_bounded_and_not_a_handle() {
        let fixture = Fixture::new();
        let prepared = fixture
            .controller
            .prepare_start(fixture.recipe(), Duration::from_secs(10))
            .unwrap();
        let intent = prepared.intent_record();
        assert_eq!(
            ServiceIntentRecord::from_persisted_parts(
                intent.coordinator_id().to_owned(),
                intent.unit_name().to_owned(),
                intent.launch_identity().to_owned(),
                intent.stop_timeout_millis(),
            )
            .unwrap(),
            *intent
        );
        assert_eq!(
            ServiceIntentRecord::from_persisted_parts(
                intent.coordinator_id().to_owned(),
                "llama-swap.service".to_owned(),
                intent.launch_identity().to_owned(),
                intent.stop_timeout_millis(),
            )
            .err(),
            Some(ManagedServiceErrorKind::OwnershipMismatch)
        );
        assert_eq!(
            ServiceIntentRecord::from_persisted_parts(
                intent.coordinator_id().to_owned(),
                intent.unit_name().to_owned(),
                "x".repeat(100_000),
                intent.stop_timeout_millis(),
            )
            .err(),
            Some(ManagedServiceErrorKind::OwnershipMismatch)
        );
    }
}
