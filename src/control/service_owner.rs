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

use super::managed_service::{
    LaunchRecipe, ManagedServiceController, ManagedServiceErrorKind, OwnedService, ReconcileResult,
    ServiceIntentRecord, StartDisposition, StartFailure,
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
        check_context(context).map_err(OwnerStartError::Preparation)?;
        let prepared = self
            .controller
            .prepare_start(recipe, stop_timeout)
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
    ) -> Result<(), ManagedServiceErrorKind> {
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
    use super::super::runtime::NeverCancel;
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::time::Instant;

    struct Fixture {
        root: tempfile::TempDir,
        intents: tempfile::TempDir,
        locks: tempfile::TempDir,
    }
    impl Fixture {
        fn new() -> Self {
            let result = Self {
                root: tempfile::tempdir().unwrap(),
                intents: tempfile::tempdir().unwrap(),
                locks: tempfile::tempdir().unwrap(),
            };
            for directory in [
                result.root.path(),
                result.intents.path(),
                result.locks.path(),
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
}
