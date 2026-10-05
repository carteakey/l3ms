//! Persist-before-dispatch ownership of coordinator-created services.
//!
//! This composes the real intent store and controller under a retained host
//! guard. It is not an admission barrier or RuntimeAdapter. Restored intents
//! are observed, never replayed; neither stop acknowledgment nor an absent
//! unit retires its record or proves resource release.
//! A lifecycle executor must reconcile retained intents and separately own
//! admission/resource reservations before calling start. Host exclusion alone
//! does not authorize engine activation or resolve GPU conflicts.

use std::fmt;
use std::time::Duration;

use super::catalog::RuntimeIdentity;
use super::cgroup_release::{CgroupObservation, CgroupReleaseErrorKind, CgroupReleaseProbe};
use super::managed_service::{
    LaunchRecipe, ManagedServiceController, ManagedServiceErrorKind, OwnedService, ReconcileResult,
    ServiceIntentRecord, StartDisposition, StartFailure, StoppedService,
};
use super::ownership::CoordinatorLock;
use super::runtime::OperationContext;
use super::service_intents::{ServiceIntentStore, ServiceIntentStoreError};

pub enum OwnerStartError {
    Preparation(ManagedServiceErrorKind),
    Persistence {
        intent: ServiceIntentRecord,
        kind: ServiceIntentStoreError,
    },
    NotDispatched {
        intent: ServiceIntentRecord,
        kind: ManagedServiceErrorKind,
    },
    Dispatch(StartFailure),
}

impl OwnerStartError {
    pub fn intent_record(&self) -> Option<&ServiceIntentRecord> {
        match self {
            Self::Preparation(_) => None,
            Self::Persistence { intent, .. } | Self::NotDispatched { intent, .. } => Some(intent),
            Self::Dispatch(failure) => Some(failure.intent_record()),
        }
    }

    pub fn disposition(&self) -> StartDisposition {
        match self {
            Self::Dispatch(failure) => failure.disposition(),
            _ => StartDisposition::NotDispatched,
        }
    }
}

impl fmt::Debug for OwnerStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnerStartError")
            .field("disposition", &self.disposition())
            .field("intent", &"<retained privately>")
            .finish()
    }
}
impl fmt::Display for OwnerStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Preparation(_) => "service preparation failed",
            Self::Persistence { .. } => "service intent persistence failed; dispatch refused",
            Self::NotDispatched { .. } => "service operation ended before dispatch",
            Self::Dispatch(_) => "service dispatch failed; intent retained",
        })
    }
}
impl std::error::Error for OwnerStartError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnerObservationError {
    StoreUnavailable,
    ManagerUnavailable,
    Cancelled,
    DeadlineExceeded,
}
impl fmt::Display for OwnerObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("retained service observation failed")
    }
}
impl std::error::Error for OwnerObservationError {}

pub struct RetainedServiceObservation {
    pub intent: ServiceIntentRecord,
    pub observation: ReconcileResult,
}

/// Fixed failure phase and category for the stop-plus-observation operation.
/// The contained kinds are themselves fixed and redacted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopAndObserveErrorKind {
    Stop(ManagedServiceErrorKind),
    Observation(CgroupReleaseErrorKind),
}

/// The generation-bound cgroup observation returned together with the exact
/// stop receipt it inspected. Neither item proves GPU release or admission
/// safety.
#[derive(Eq, PartialEq)]
pub struct StoppedServiceObservation {
    receipt: StoppedService,
    observation: CgroupObservation,
}

impl StoppedServiceObservation {
    pub fn receipt(&self) -> &StoppedService {
        &self.receipt
    }

    pub fn observation(&self) -> &CgroupObservation {
        &self.observation
    }
}

impl fmt::Debug for StoppedServiceObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoppedServiceObservation")
            .field("receipt", &"<redacted>")
            .field("observation", &self.observation)
            .finish()
    }
}

/// Stop errors carry no receipt. If stop succeeded but cgroup observation
/// failed, the receipt is retained so callers can re-observe without issuing
/// another stop command.
pub struct StopAndObserveError {
    kind: StopAndObserveErrorKind,
    receipt: Option<StoppedService>,
}

impl StopAndObserveError {
    pub fn kind(&self) -> StopAndObserveErrorKind {
        self.kind
    }

    pub fn receipt(&self) -> Option<&StoppedService> {
        self.receipt.as_ref()
    }

    pub fn into_receipt(self) -> Option<StoppedService> {
        self.receipt
    }
}

impl fmt::Debug for StopAndObserveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StopAndObserveError")
            .field("kind", &self.kind)
            .field("receipt", &"<retained privately>")
            .finish()
    }
}

impl fmt::Display for StopAndObserveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            StopAndObserveErrorKind::Stop(_) => "service stop failed",
            StopAndObserveErrorKind::Observation(_) => "stopped service observation failed",
        })
    }
}

impl std::error::Error for StopAndObserveError {}

/// Borrowing the guard prevents normal callers from dropping host exclusion
/// while this owner is in use. The caller must use the same configured host
/// lock directory as all other coordinated entry points; advisory locks do not
/// exclude standalone clients or concurrent same-user manager changes.
/// Create one owner per dedicated store/controller under that guard. Mutable
/// operation methods serialize use of this instance; constructing other owners
/// or direct controller/store users does not bypass the caller precondition.
pub struct ManagedRuntimeOwner<'host> {
    controller: ManagedServiceController,
    store: ServiceIntentStore,
    _host: &'host CoordinatorLock,
}

impl<'host> ManagedRuntimeOwner<'host> {
    pub fn new(
        controller: ManagedServiceController,
        store: ServiceIntentStore,
        host: &'host CoordinatorLock,
    ) -> Self {
        Self {
            controller,
            store,
            _host: host,
        }
    }

    /// No manager command is attempted unless the write-ahead record has been
    /// acknowledged durable. Persistence errors can leave a partial record;
    /// retain it and quarantine rather than delete/retry. Context is checked
    /// before preparation and again after synchronous non-preemptible fsync.
    /// The caller must already have reconciled runtime ownership and obtained
    /// admission/resource authority; this method does not check GPU availability.
    pub fn start(
        &mut self,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        self.start_inner(None, recipe, stop_timeout, context)
    }

    /// Persist and dispatch a start intent bound to the selected catalog
    /// runtime. This shares the exact write-ahead path with generic starts.
    pub fn start_for_runtime(
        &mut self,
        runtime_identity: RuntimeIdentity,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        self.start_inner(Some(runtime_identity), recipe, stop_timeout, context)
    }

    fn start_inner(
        &mut self,
        runtime_identity: Option<RuntimeIdentity>,
        recipe: LaunchRecipe,
        stop_timeout: Duration,
        context: &OperationContext<'_>,
    ) -> Result<OwnedService, OwnerStartError> {
        check_context(context).map_err(OwnerStartError::Preparation)?;
        let prepared = match runtime_identity {
            Some(identity) => {
                self.controller
                    .prepare_start_for_runtime(identity, recipe, stop_timeout)
            }
            None => self.controller.prepare_start(recipe, stop_timeout),
        }
        .map_err(OwnerStartError::Preparation)?;
        let intent = prepared.intent_record().clone();
        self.store
            .record(&intent)
            .map_err(|kind| OwnerStartError::Persistence {
                intent: intent.clone(),
                kind,
            })?;
        check_context(context).map_err(|kind| OwnerStartError::NotDispatched { intent, kind })?;
        self.controller
            .dispatch(prepared, context)
            .map_err(OwnerStartError::Dispatch)
    }

    /// Observe every retained intent; never recreate a recipe or replay start.
    /// Any partial/failed observation is an error, not startup admission.
    pub fn reconcile(
        &mut self,
        context: &OperationContext<'_>,
    ) -> Result<Vec<RetainedServiceObservation>, OwnerObservationError> {
        observation_context(context)?;
        let records = self
            .store
            .list()
            .map_err(|_| OwnerObservationError::StoreUnavailable)?;
        let mut observations = Vec::with_capacity(records.len());
        for intent in records {
            observation_context(context)?;
            let observation =
                self.controller
                    .reconcile(&intent, context)
                    .map_err(|kind| match kind {
                        ManagedServiceErrorKind::Cancelled => OwnerObservationError::Cancelled,
                        ManagedServiceErrorKind::DeadlineExceeded => {
                            OwnerObservationError::DeadlineExceeded
                        }
                        _ => OwnerObservationError::ManagerUnavailable,
                    })?;
            observations.push(RetainedServiceObservation {
                intent,
                observation,
            });
        }
        observation_context(context)?;
        Ok(observations)
    }

    /// Retains the write-ahead record on every outcome. Success acknowledges
    /// the manager operation only; a separate release/admission protocol is
    /// needed before conflicting inference can be accepted.
    pub fn stop(
        &mut self,
        service: &OwnedService,
        context: &OperationContext<'_>,
    ) -> Result<StoppedService, ManagedServiceErrorKind> {
        check_context(context)?;
        let records = self
            .store
            .list()
            .map_err(|_| ManagedServiceErrorKind::UnitStateUnverifiable)?;
        if !records
            .iter()
            .any(|record| record == service.intent_record())
        {
            return Err(ManagedServiceErrorKind::OwnershipMismatch);
        }
        check_context(context)?;
        self.controller.stop(service, context)
    }

    /// Stop the exact owned generation, then inspect only the cgroup bound to
    /// the returned stop receipt. The durable intent remains retained on all
    /// outcomes. If observation fails after stop acknowledgment, the error
    /// carries the receipt so the caller can retry only the read-only probe.
    /// This does not release GPU ownership or open runtime admission.
    pub fn stop_and_observe(
        &mut self,
        service: &OwnedService,
        probe: &CgroupReleaseProbe,
        context: &OperationContext<'_>,
    ) -> Result<StoppedServiceObservation, StopAndObserveError> {
        let receipt = self
            .stop(service, context)
            .map_err(|kind| StopAndObserveError {
                kind: StopAndObserveErrorKind::Stop(kind),
                receipt: None,
            })?;
        let observation = match probe.observe(&receipt, context) {
            Ok(observation) => observation,
            Err(kind) => {
                return Err(StopAndObserveError {
                    kind: StopAndObserveErrorKind::Observation(kind),
                    receipt: Some(receipt),
                });
            }
        };
        if !observation.matches(&receipt) {
            return Err(StopAndObserveError {
                kind: StopAndObserveErrorKind::Observation(CgroupReleaseErrorKind::InvalidReceipt),
                receipt: Some(receipt),
            });
        }
        Ok(StoppedServiceObservation {
            receipt,
            observation,
        })
    }
}

fn check_context(context: &OperationContext<'_>) -> Result<(), ManagedServiceErrorKind> {
    if context.is_cancelled() {
        Err(ManagedServiceErrorKind::Cancelled)
    } else if context.remaining().is_zero() {
        Err(ManagedServiceErrorKind::DeadlineExceeded)
    } else {
        Ok(())
    }
}
fn observation_context(context: &OperationContext<'_>) -> Result<(), OwnerObservationError> {
    check_context(context).map_err(|kind| match kind {
        ManagedServiceErrorKind::Cancelled => OwnerObservationError::Cancelled,
        _ => OwnerObservationError::DeadlineExceeded,
    })
}

#[cfg(test)]
mod tests {
    use super::super::catalog::{Lane, RuntimeIdentity};
    use super::super::cgroup_release::CgroupObservationKind;
    use super::super::runtime::{Cancellation, NeverCancel};
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::time::Instant;

    struct Fixture {
        root: tempfile::TempDir,
        intents: tempfile::TempDir,
        locks: tempfile::TempDir,
        cgroups: tempfile::TempDir,
    }
    impl Fixture {
        fn new() -> Self {
            let result = Self {
                root: tempfile::tempdir().unwrap(),
                intents: tempfile::tempdir().unwrap(),
                locks: tempfile::tempdir().unwrap(),
                cgroups: tempfile::tempdir().unwrap(),
            };
            for directory in [
                result.root.path(),
                result.intents.path(),
                result.locks.path(),
                result.cgroups.path(),
            ] {
                fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
            }
            let root_json = serde_json::to_string(result.root.path().to_str().unwrap()).unwrap();
            let intents_json =
                serde_json::to_string(result.intents.path().to_str().unwrap()).unwrap();
            let script = format!(
                r#"#!/usr/bin/python3
import json, os, pathlib, sys
root = pathlib.Path({root_json})
intents = pathlib.Path({intents_json})
state = root / 'manager-state.json'
args = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == 'fake-run':
    unit = next(a.split('=', 1)[1] for a in args if a.startswith('--unit='))
    description = next(a.split('=', 1)[1] for a in args if a.startswith('--description='))
    record = json.loads((intents / (unit + '.json')).read_text())
    assert record['unit_name'] == unit
    if (root / 'require-runtime').exists():
        assert record['schema'] == 2
        assert record['runtime_lane'] == 'media'
        assert record['runtime_name'] == 'comfyui'
    with (root / 'starts').open('a') as log:
        log.write('start\n')
    state.write_text(json.dumps(dict(unit=unit, description=description)))
    sys.exit(9 if (root / 'fail-launcher').exists() else 0)
if 'list-units' in args or 'list-unit-files' in args:
    sys.exit(0)
if 'stop' in args:
    with (root / 'stops').open('a') as log:
        log.write('stop\n')
    state.unlink()
    sys.exit(0)
if 'show' in args:
    if not state.exists():
        print('LoadState=not-found')
        sys.exit(0)
    data = json.loads(state.read_text())
    if args[-1] != data['unit']:
        print('LoadState=not-found')
        sys.exit(0)
    print('LoadState=loaded\nActiveState=active\nTransient=yes')
    print('InvocationID=' + 'a' * 32)
    print('ControlGroup=/user.slice/' + data['unit'])
    print('FragmentPath=/run/user/1000/systemd/transient/' + data['unit'])
    print('Description=' + data['description'])
    print('Restart=no\nKillMode=control-group')
    sys.exit(0)
sys.exit(23)
"#
            );
            for name in ["fake-run", "fake-ctl"] {
                let _guard = super::super::process::executable_fixture_guard();
                let path = result.root.path().join(name);
                fs::write(&path, &script).unwrap();
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
            }
            result
        }
        fn controller(&self) -> ManagedServiceController {
            ManagedServiceController::with_test_executables(
                "fixture",
                self.root.path().join("fake-run"),
                self.root.path().join("fake-ctl"),
            )
            .unwrap()
        }
        fn owner<'a>(&self, guard: &'a CoordinatorLock) -> ManagedRuntimeOwner<'a> {
            ManagedRuntimeOwner::new(
                self.controller(),
                ServiceIntentStore::open(self.intents.path()).unwrap(),
                guard,
            )
        }
        fn recipe(&self) -> LaunchRecipe {
            LaunchRecipe::new(Path::new("/usr/bin/true"), self.root.path(), Vec::new()).unwrap()
        }
        fn context(&self) -> OperationContext<'static> {
            OperationContext::new(Instant::now() + Duration::from_secs(3), &NeverCancel)
        }

        fn cgroup_probe(&self, service: &OwnedService, events: Option<&str>) -> CgroupReleaseProbe {
            let leaf = self
                .cgroups
                .path()
                .join("user.slice")
                .join(service.unit_name());
            fs::create_dir_all(&leaf).unwrap();
            if let Some(events) = events {
                fs::write(leaf.join("cgroup.events"), events).unwrap();
            }
            CgroupReleaseProbe::with_test_root(self.cgroups.path()).unwrap()
        }
    }

    #[test]
    fn durable_record_precedes_dispatch_and_restart_observes_without_replay() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        assert_eq!(
            ServiceIntentStore::open(fixture.intents.path())
                .unwrap()
                .list()
                .unwrap()
                .len(),
            1
        );
        drop(owner);
        let mut owner = fixture.owner(&guard);
        let observations = owner.reconcile(&fixture.context()).unwrap();
        assert_eq!(observations.len(), 1);
        assert!(matches!(
            &observations[0].observation,
            ReconcileResult::Owned(_)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
        owner.stop(&handle, &fixture.context()).unwrap();
        let observations = owner.reconcile(&fixture.context()).unwrap();
        assert!(matches!(
            &observations[0].observation,
            ReconcileResult::AbsentNow
        ));
        assert_eq!(
            ServiceIntentStore::open(fixture.intents.path())
                .unwrap()
                .list()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn runtime_bound_start_is_persisted_before_dispatch_and_restored_without_replay() {
        let fixture = Fixture::new();
        fs::write(fixture.root.path().join("require-runtime"), "").unwrap();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let runtime = RuntimeIdentity::new(Lane::Media, "comfyui").unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start_for_runtime(
                runtime.clone(),
                fixture.recipe(),
                Duration::from_secs(1),
                &fixture.context(),
            )
            .unwrap();

        let stored = ServiceIntentStore::open(fixture.intents.path())
            .unwrap()
            .list()
            .unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].runtime_identity(), Some(&runtime));
        assert_eq!(handle.intent_record().runtime_identity(), Some(&runtime));

        drop(owner);
        let mut restarted_owner = fixture.owner(&guard);
        let observations = restarted_owner.reconcile(&fixture.context()).unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].intent.runtime_identity(), Some(&runtime));
        assert!(matches!(
            &observations[0].observation,
            ReconcileResult::Owned(_)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
    }

    #[test]
    fn uncertain_launcher_retains_durable_intent_for_observation() {
        let fixture = Fixture::new();
        fs::write(fixture.root.path().join("fail-launcher"), "").unwrap();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap_err();
        assert_eq!(error.disposition(), StartDisposition::OwnershipUncertain);
        let records = ServiceIntentStore::open(fixture.intents.path())
            .unwrap()
            .list()
            .unwrap();
        assert_eq!(records.len(), 1);
        assert!(error.intent_record() == Some(&records[0]));
        assert!(matches!(
            &owner.reconcile(&fixture.context()).unwrap()[0].observation,
            ReconcileResult::Owned(_)
        ));
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("starts")).unwrap(),
            "start\n"
        );
    }

    #[test]
    fn persistence_failure_refuses_manager_dispatch_and_preserves_files() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        fs::write(fixture.intents.path().join("unexpected"), "retain me").unwrap();
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap_err();
        assert!(matches!(error, OwnerStartError::Persistence { .. }));
        assert_eq!(error.disposition(), StartDisposition::NotDispatched);
        assert!(!fixture.root.path().join("starts").exists());
        assert_eq!(
            fs::read_to_string(fixture.intents.path().join("unexpected")).unwrap(),
            "retain me"
        );
    }

    #[test]
    fn expired_context_does_not_create_intent_or_launch() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let context = OperationContext::new(Instant::now(), &NeverCancel);
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &context)
            .unwrap_err();
        assert!(error.intent_record().is_none());
        assert!(owner.store.list().unwrap().is_empty());
        assert!(!fixture.root.path().join("starts").exists());
    }

    #[test]
    fn corrupt_store_refuses_stop_without_removing_ownership_record() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        fs::write(fixture.intents.path().join("unexpected"), "preserve me").unwrap();
        assert_eq!(
            owner.stop(&handle, &fixture.context()),
            Err(ManagedServiceErrorKind::UnitStateUnverifiable)
        );
        assert!(!fixture.root.path().join("stops").exists());
        assert!(fixture
            .intents
            .path()
            .join(format!("{}.json", handle.unit_name()))
            .exists());
        assert!(fixture.root.path().join("manager-state.json").exists());
    }

    #[test]
    fn cancellation_after_persistence_retains_record_without_dispatch() {
        struct CancelWhenRecorded<'a>(&'a Path);
        impl super::super::runtime::Cancellation for CancelWhenRecorded<'_> {
            fn is_cancelled(&self) -> bool {
                fs::read_dir(self.0).unwrap().next().is_some()
            }
        }
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let cancellation = CancelWhenRecorded(fixture.intents.path());
        let context = OperationContext::new(Instant::now() + Duration::from_secs(3), &cancellation);
        let error = owner
            .start(fixture.recipe(), Duration::from_secs(1), &context)
            .unwrap_err();
        assert!(matches!(
            error,
            OwnerStartError::NotDispatched {
                kind: ManagedServiceErrorKind::Cancelled,
                ..
            }
        ));
        let records = owner.store.list().unwrap();
        assert_eq!(records.len(), 1);
        assert!(error.intent_record() == Some(&records[0]));
        assert!(!fixture.root.path().join("starts").exists());
        assert!(!fixture.root.path().join("manager-state.json").exists());
    }

    #[test]
    fn valid_nonmember_handle_is_refused_before_manager_stop() {
        let fixture = Fixture::new();
        let foreign = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let foreign_guard = CoordinatorLock::acquire(foreign.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let mut foreign_owner = foreign.owner(&foreign_guard);
        let handle = foreign_owner
            .start(foreign.recipe(), Duration::from_secs(1), &foreign.context())
            .unwrap();
        assert_eq!(
            owner.stop(&handle, &fixture.context()),
            Err(ManagedServiceErrorKind::OwnershipMismatch)
        );
        assert!(!fixture.root.path().join("stops").exists());
        assert!(!foreign.root.path().join("stops").exists());
        assert!(foreign.root.path().join("manager-state.json").exists());
    }

    #[test]
    fn stop_and_observe_returns_receipt_bound_empty_observation_and_retains_ledger() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, Some("populated 0\nfrozen 0\n"));

        let result = owner
            .stop_and_observe(&handle, &probe, &fixture.context())
            .unwrap();
        assert_eq!(result.observation().kind(), CgroupObservationKind::Empty);
        assert!(result.observation().matches(result.receipt()));
        assert!(fixture.root.path().join("stops").exists());
        assert_eq!(owner.store.list().unwrap().len(), 1);
        assert!(!format!("{result:?}").contains(handle.unit_name()));
    }

    #[test]
    fn observation_failure_retains_transferable_receipt_without_repeating_stop() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, Some("populated 1\nfrozen 0\n"));

        let error = owner
            .stop_and_observe(&handle, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Observation(CgroupReleaseErrorKind::CgroupNotEmpty)
        );
        assert!(error.receipt().is_some());
        assert!(!format!("{error:?}").contains(handle.unit_name()));
        let receipt = error.into_receipt().unwrap();
        assert_eq!(
            probe.observe(&receipt, &fixture.context()).unwrap_err(),
            CgroupReleaseErrorKind::CgroupNotEmpty
        );
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }

    #[test]
    fn missing_events_is_observation_failure_with_receipt_but_not_removal() {
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, None);

        let error = owner
            .stop_and_observe(&handle, &probe, &fixture.context())
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Observation(CgroupReleaseErrorKind::MissingEvents)
        );
        assert!(error.receipt().is_some());
        assert_eq!(owner.store.list().unwrap().len(), 1);
        assert_eq!(
            fs::read_to_string(fixture.root.path().join("stops")).unwrap(),
            "stop\n"
        );
    }

    #[test]
    fn cancelled_stop_has_no_receipt_and_does_not_dispatch_manager_stop() {
        struct Cancel;
        impl Cancellation for Cancel {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let fixture = Fixture::new();
        let guard = CoordinatorLock::acquire(fixture.locks.path(), []).unwrap();
        let mut owner = fixture.owner(&guard);
        let handle = owner
            .start(fixture.recipe(), Duration::from_secs(1), &fixture.context())
            .unwrap();
        let probe = fixture.cgroup_probe(&handle, Some("populated 0\n"));
        let cancelled = OperationContext::new(Instant::now() + Duration::from_secs(2), &Cancel);

        let error = owner
            .stop_and_observe(&handle, &probe, &cancelled)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            StopAndObserveErrorKind::Stop(ManagedServiceErrorKind::Cancelled)
        );
        assert!(error.receipt().is_none());
        assert!(!fixture.root.path().join("stops").exists());
        assert_eq!(owner.store.list().unwrap().len(), 1);
    }
}
