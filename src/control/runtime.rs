//! In-process lifecycle contract for exclusive-resource runtime transitions.
//!
//! This module is intentionally not connected to service commands yet. A
//! backend adapter must *prove* that it can stop new direct ingress and drain
//! accepted work; a one-time idle check is not an admission barrier. Unknown
//! resource ownership or an external GPU runtime makes switching fail closed.
//! Cross-process locking and a host-local coordinator are separate release
//! gates before this can back a user-facing switch command.

use std::time::{Duration, Instant};
use std::{error::Error, fmt};

use anyhow::{anyhow, ensure, Context, Result};

use super::catalog::RuntimeIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleOwner {
    SystemdUser,
    ChildProcess,
    /// A process or endpoint this stack did not start and must never stop.
    External,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceClaim {
    None,
    ExclusiveGpu(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeState {
    Stopped,
    Running,
    /// Inspection was incomplete or the observed state cannot be trusted.
    Unknown,
}

/// Bounds and cancellation supplied to one adapter operation.
///
/// Adapters should pass `remaining()` to native timeout controls and check
/// `is_cancelled()` while waiting or polling. The coordinator checks this
/// context before and after non-commit operations, but cannot preempt a
/// synchronous adapter method that ignores the context. Admission-open
/// operations are commit points: their success is final if cancellation races
/// after the open.
pub struct OperationContext<'a> {
    deadline: Instant,
    cancellation: &'a dyn Cancellation,
}

impl<'a> OperationContext<'a> {
    pub fn new(deadline: Instant, cancellation: &'a dyn Cancellation) -> Self {
        Self {
            deadline,
            cancellation,
        }
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }

    pub fn check_active(&self, operation: &str) -> Result<()> {
        ensure!(!self.is_cancelled(), "operation cancelled: {operation}");
        ensure!(
            Instant::now() < self.deadline,
            "operation deadline expired: {operation}"
        );
        Ok(())
    }
}

/// Independent time allowance for cleanup/restoration after primary failure.
/// Recovery ignores caller cancellation so a cancelled request cannot strand
/// exclusive-resource ownership, but every compensation remains bounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecoveryPolicy {
    timeout: Duration,
}

impl RecoveryPolicy {
    pub const MAX_TIMEOUT: Duration = Duration::from_secs(300);

    pub fn new(timeout: Duration) -> Result<Self> {
        ensure!(
            timeout > Duration::ZERO,
            "recovery timeout must be positive"
        );
        ensure!(
            timeout <= Self::MAX_TIMEOUT,
            "recovery timeout exceeds {} seconds",
            Self::MAX_TIMEOUT.as_secs()
        );
        Ok(Self { timeout })
    }

    pub fn timeout(self) -> Duration {
        self.timeout
    }
}

// A recovery context must not inherit a cancelled request's token.
static NEVER_CANCEL: NeverCancel = NeverCancel;

fn recovery_context(policy: RecoveryPolicy) -> Result<OperationContext<'static>> {
    let deadline = Instant::now()
        .checked_add(policy.timeout())
        .context("recovery deadline is not representable")?;
    Ok(OperationContext::new(deadline, &NEVER_CANCEL))
}

/// A native adapter owns backend-specific admission, service, and health
/// details. `close_admission` must prevent *new direct work* until
/// `open_admission` is called; an adapter unable to do that returns an error.
/// `drain` must acknowledge that every previously accepted job has finished.
/// Every potentially blocking operation receives the transition or recovery
/// context that bounds it.
/// `open_admission` must be atomic: an error leaves admission closed, while a
/// successful open is the relevant transition/recovery commit point.
pub trait RuntimeAdapter {
    fn identity(&self) -> &RuntimeIdentity;
    fn ownership(&self) -> LifecycleOwner;
    fn resource(&self) -> ResourceClaim;
    fn state(&self, context: &OperationContext<'_>) -> Result<RuntimeState>;
    fn close_admission(&mut self, context: &OperationContext<'_>) -> Result<()>;
    fn drain(&mut self, context: &OperationContext<'_>) -> Result<()>;
    /// Must be safe to call after partial startup, including failed activate.
    fn deactivate(&mut self, context: &OperationContext<'_>) -> Result<()>;
    fn verify_released(&self, context: &OperationContext<'_>) -> Result<()>;
    fn activate(&mut self, context: &OperationContext<'_>) -> Result<()>;
    fn verify_healthy(&self, context: &OperationContext<'_>) -> Result<()>;
    fn open_admission(&mut self, context: &OperationContext<'_>) -> Result<()>;
}

/// A cancellation source supplied by the caller. Cancellation is checked
/// before and after primary adapter operations; a cancellation after the old
/// runtime stops triggers bounded cleanup/restoration with cancellation masked.
pub trait Cancellation {
    fn is_cancelled(&self) -> bool;
}

pub struct NeverCancel;

impl Cancellation for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// What a coordinator may infer after a failed transition. None of these
/// states licenses replaying an inference request with an uncertain result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureDisposition {
    /// No runtime stop/start was attempted; existing ownership was not changed.
    NoTransition,
    /// The previous runtime is healthy with admission reopened.
    PreviousRestored,
    /// Runtime or GPU ownership cannot be verified; quarantine conflicting work.
    OwnershipUncertain,
}

#[derive(Debug)]
pub struct SwitchFailure {
    pub disposition: FailureDisposition,
    source: anyhow::Error,
}

impl SwitchFailure {
    fn new(disposition: FailureDisposition, source: anyhow::Error) -> Self {
        Self {
            disposition,
            source,
        }
    }
}

impl fmt::Display for SwitchFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:#}", self.source)
    }
}

impl Error for SwitchFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

pub type SwitchResult<T> = std::result::Result<T, SwitchFailure>;

fn call_adapter<T>(
    context: &OperationContext<'_>,
    operation: &str,
    call: impl FnOnce() -> Result<T>,
) -> Result<T> {
    context.check_active(&format!("before {operation}"))?;
    let result = call().with_context(|| operation.to_owned())?;
    context.check_active(&format!("after {operation}"))?;
    Ok(result)
}

fn recovery_failure(primary: anyhow::Error, recovery: anyhow::Error) -> SwitchFailure {
    SwitchFailure::new(
        FailureDisposition::OwnershipUncertain,
        anyhow!(
            "transition failed: {primary:#}; recovery failed: {recovery:#}; ownership is uncertain"
        ),
    )
}

fn restore(previous: &mut dyn RuntimeAdapter, recovery: &OperationContext<'_>) -> Result<()> {
    call_adapter(recovery, "reactivate previous runtime", || {
        previous.activate(recovery)
    })?;
    call_adapter(recovery, "verify restored runtime health", || {
        previous.verify_healthy(recovery)
    })?;
    open_admission_commit(previous, recovery, "reopen previous runtime admission")
}

/// An admission open is a commit point. Check before it; a successful adapter
/// return commits even if cancellation/deadline expiration races immediately
/// after the open. The adapter must honor the passed context before returning
/// success; the coordinator cannot safely roll back an already-open ingress.
fn open_admission_commit(
    runtime: &mut dyn RuntimeAdapter,
    context: &OperationContext<'_>,
    operation: &str,
) -> Result<()> {
    context.check_active(&format!("before {operation}"))?;
    runtime
        .open_admission(context)
        .with_context(|| operation.to_owned())
}

fn failed_with_restore(
    previous: &mut dyn RuntimeAdapter,
    primary: anyhow::Error,
    recovery: &OperationContext<'_>,
) -> SwitchFailure {
    match restore(previous, recovery) {
        Ok(()) => SwitchFailure::new(
            FailureDisposition::PreviousRestored,
            primary.context("previous runtime restored"),
        ),
        Err(recovery_error) => recovery_failure(primary, recovery_error),
    }
}

fn failed_with_new_recovery(
    previous: &mut dyn RuntimeAdapter,
    primary: anyhow::Error,
    policy: RecoveryPolicy,
) -> SwitchFailure {
    match recovery_context(policy) {
        Ok(recovery) => failed_with_restore(previous, primary, &recovery),
        Err(recovery_error) => recovery_failure(primary, recovery_error),
    }
}

fn reopen_previous_after_failure(
    previous: &mut dyn RuntimeAdapter,
    primary: anyhow::Error,
    policy: RecoveryPolicy,
) -> SwitchFailure {
    let recovery = match recovery_context(policy) {
        Ok(recovery) => recovery,
        Err(recovery_error) => return recovery_failure(primary, recovery_error),
    };
    match open_admission_commit(previous, &recovery, "reopen previous runtime admission") {
        Ok(()) => SwitchFailure::new(FailureDisposition::NoTransition, primary),
        Err(recovery_error) => recovery_failure(primary, recovery_error),
    }
}

fn failed_during_deactivation(
    previous: &mut dyn RuntimeAdapter,
    primary: anyhow::Error,
    policy: RecoveryPolicy,
) -> SwitchFailure {
    let recovery = match recovery_context(policy) {
        Ok(recovery) => recovery,
        Err(recovery_error) => return recovery_failure(primary, recovery_error),
    };
    let observed = call_adapter(
        &recovery,
        "inspect previous runtime during recovery",
        || previous.state(&recovery),
    );
    match observed {
        Ok(RuntimeState::Running) => {
            let restored = call_adapter(
                &recovery,
                "verify previous runtime after failed stop",
                || previous.verify_healthy(&recovery),
            )
            .and_then(|()| {
                open_admission_commit(
                    previous,
                    &recovery,
                    "reopen previous runtime admission",
                )
            });
            match restored {
                Ok(()) => SwitchFailure::new(
                    FailureDisposition::PreviousRestored,
                    primary.context("previous runtime remained healthy and admission reopened"),
                ),
                Err(error) => recovery_failure(primary, error),
            }
        }
        Ok(RuntimeState::Stopped) => {
            match call_adapter(&recovery, "verify previous GPU release", || {
                previous.verify_released(&recovery)
            }) {
                Ok(()) => failed_with_restore(previous, primary, &recovery),
                Err(error) => recovery_failure(primary, error),
            }
        }
        Ok(RuntimeState::Unknown) => SwitchFailure::new(
            FailureDisposition::OwnershipUncertain,
            anyhow!("transition failed: {primary:#}; previous runtime state is unknown; refusing restart"),
        ),
        Err(error) => recovery_failure(primary, error),
    }
}

fn failed_after_stop_before_target(
    previous: &mut dyn RuntimeAdapter,
    primary: anyhow::Error,
    policy: RecoveryPolicy,
) -> SwitchFailure {
    let recovery = match recovery_context(policy) {
        Ok(recovery) => recovery,
        Err(recovery_error) => return recovery_failure(primary, recovery_error),
    };
    match call_adapter(
        &recovery,
        "verify previous GPU release during recovery",
        || previous.verify_released(&recovery),
    ) {
        Ok(()) => failed_with_restore(previous, primary, &recovery),
        Err(error) => recovery_failure(primary, error),
    }
}

fn preflight(
    previous: &dyn RuntimeAdapter,
    target: &dyn RuntimeAdapter,
    context: &OperationContext<'_>,
) -> Result<()> {
    context.check_active("before transition preflight")?;
    ensure!(
        previous.identity() != target.identity(),
        "cannot switch a runtime to itself"
    );
    ensure!(
        previous.ownership() != LifecycleOwner::External
            && target.ownership() != LifecycleOwner::External,
        "external runtimes are not lifecycle-managed"
    );
    ensure!(
        matches!(previous.resource(), ResourceClaim::ExclusiveGpu(_))
            && previous.resource() == target.resource(),
        "runtimes do not declare the same exclusive GPU"
    );
    ensure!(
        call_adapter(context, "inspect previous runtime", || {
            previous.state(context)
        })? == RuntimeState::Running,
        "previous runtime is not known running"
    );
    ensure!(
        call_adapter(context, "inspect target runtime", || target.state(context))?
            == RuntimeState::Stopped,
        "target runtime is not known stopped"
    );
    context.check_active("after transition preflight")?;
    Ok(())
}

/// Switch one exclusively owned GPU resource between two managed runtimes.
/// This is an in-process primitive, not an installed host coordinator. Every
/// adapter call receives `context`; adapters must honor its cancellation and
/// deadline. The coordinator checks around non-commit calls but cannot preempt
/// a blocking synchronous implementation. A successful admission open is the
/// commit point. On primary failure, target cleanup and any
/// restoration share one fresh `recovery_policy` budget that ignores caller
/// cancellation. Successful target `open_admission` is the commit point.
pub fn switch_exclusive(
    previous: &mut dyn RuntimeAdapter,
    target: &mut dyn RuntimeAdapter,
    context: &OperationContext<'_>,
    recovery_policy: RecoveryPolicy,
) -> SwitchResult<()> {
    preflight(previous, target, context)
        .map_err(|error| SwitchFailure::new(FailureDisposition::NoTransition, error))?;

    if let Err(error) = call_adapter(context, "close previous runtime admission", || {
        previous.close_admission(context)
    }) {
        return Err(reopen_previous_after_failure(
            previous,
            error,
            recovery_policy,
        ));
    }
    if let Err(error) = call_adapter(context, "drain previous runtime", || {
        previous.drain(context)
    }) {
        return Err(reopen_previous_after_failure(
            previous,
            error,
            recovery_policy,
        ));
    }

    if let Err(reason) = context.check_active("after draining previous runtime") {
        return Err(reopen_previous_after_failure(
            previous,
            reason,
            recovery_policy,
        ));
    }

    if let Err(error) = call_adapter(context, "deactivate previous runtime", || {
        previous.deactivate(context)
    }) {
        return Err(failed_during_deactivation(previous, error, recovery_policy));
    }
    if let Err(error) = call_adapter(context, "verify exclusive GPU release", || {
        previous.verify_released(context)
    }) {
        return Err(failed_after_stop_before_target(
            previous,
            error,
            recovery_policy,
        ));
    }
    if let Err(reason) = context.check_active("after exclusive GPU release") {
        return Err(failed_with_new_recovery(previous, reason, recovery_policy));
    }

    let activation = call_adapter(context, "activate target runtime", || {
        target.activate(context)
    })
    .and_then(|()| {
        call_adapter(context, "verify target runtime health", || {
            target.verify_healthy(context)
        })
    })
    .and_then(|()| {
        context.check_active("before target admission")?;
        open_admission_commit(target, context, "open target runtime admission")
    });
    if let Err(primary) = activation {
        let recovery = match recovery_context(recovery_policy) {
            Ok(recovery) => recovery,
            Err(recovery_error) => return Err(recovery_failure(primary, recovery_error)),
        };
        let cleanup = call_adapter(&recovery, "stop partially activated target", || {
            target.deactivate(&recovery)
        })
        .and_then(|()| {
            call_adapter(&recovery, "verify target GPU release", || {
                target.verify_released(&recovery)
            })
        });
        if let Err(cleanup_error) = cleanup {
            return Err(recovery_failure(
                primary,
                anyhow!("target cleanup failed: {cleanup_error:#}; previous runtime was not restarted because GPU ownership is uncertain"),
            ));
        }
        return Err(failed_with_restore(previous, primary, &recovery));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Duration;

    use super::*;
    use crate::control::catalog::Lane;

    struct FakeAdapter {
        id: RuntimeIdentity,
        owner: LifecycleOwner,
        resource: ResourceClaim,
        state: RuntimeState,
        admission_open: bool,
        fail: Option<&'static str>,
        delay_on: Option<(&'static str, Duration)>,
        events: Rc<RefCell<Vec<String>>>,
        observed_contexts: RefCell<Vec<(String, Instant)>>,
    }

    impl FakeAdapter {
        fn new(
            lane: Lane,
            name: &str,
            state: RuntimeState,
            events: Rc<RefCell<Vec<String>>>,
        ) -> Self {
            Self {
                id: RuntimeIdentity::new(lane, name).unwrap(),
                owner: LifecycleOwner::SystemdUser,
                resource: ResourceClaim::ExclusiveGpu(0),
                state,
                admission_open: state == RuntimeState::Running,
                fail: None,
                delay_on: None,
                events,
                observed_contexts: RefCell::new(Vec::new()),
            }
        }

        fn event(&self, step: &str, context: &OperationContext<'_>) -> Result<()> {
            let event = format!("{}:{step}", self.id.as_str());
            self.events.borrow_mut().push(event.clone());
            self.observed_contexts
                .borrow_mut()
                .push((event, context.deadline()));
            if let Some((delay_step, duration)) = self.delay_on {
                if delay_step == step {
                    std::thread::sleep(duration);
                }
            }
            if self.fail == Some(step) {
                anyhow::bail!("injected {step} failure")
            }
            Ok(())
        }
    }

    impl RuntimeAdapter for FakeAdapter {
        fn identity(&self) -> &RuntimeIdentity {
            &self.id
        }

        fn ownership(&self) -> LifecycleOwner {
            self.owner
        }

        fn resource(&self) -> ResourceClaim {
            self.resource
        }

        fn state(&self, context: &OperationContext<'_>) -> Result<RuntimeState> {
            self.event("state", context)?;
            Ok(self.state)
        }

        fn close_admission(&mut self, context: &OperationContext<'_>) -> Result<()> {
            self.event("close", context)?;
            self.admission_open = false;
            Ok(())
        }

        fn drain(&mut self, context: &OperationContext<'_>) -> Result<()> {
            self.event("drain", context)
        }

        fn deactivate(&mut self, context: &OperationContext<'_>) -> Result<()> {
            self.event("stop", context)?;
            self.state = RuntimeState::Stopped;
            Ok(())
        }

        fn verify_released(&self, context: &OperationContext<'_>) -> Result<()> {
            self.event("released", context)?;
            ensure!(self.state == RuntimeState::Stopped, "still running");
            Ok(())
        }

        fn activate(&mut self, context: &OperationContext<'_>) -> Result<()> {
            self.event("start", context)?;
            self.state = RuntimeState::Running;
            Ok(())
        }

        fn verify_healthy(&self, context: &OperationContext<'_>) -> Result<()> {
            self.event("healthy", context)?;
            ensure!(self.state == RuntimeState::Running, "not running");
            Ok(())
        }

        fn open_admission(&mut self, context: &OperationContext<'_>) -> Result<()> {
            self.event("open", context)?;
            self.admission_open = true;
            Ok(())
        }
    }

    fn pair() -> (FakeAdapter, FakeAdapter, Rc<RefCell<Vec<String>>>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let previous = FakeAdapter::new(
            Lane::Language,
            "language",
            RuntimeState::Running,
            events.clone(),
        );
        let target = FakeAdapter::new(Lane::Media, "media", RuntimeState::Stopped, events.clone());
        (previous, target, events)
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(3)
    }

    fn recovery_policy() -> RecoveryPolicy {
        RecoveryPolicy::new(Duration::from_secs(1)).unwrap()
    }

    fn run_switch_with_policy(
        previous: &mut dyn RuntimeAdapter,
        target: &mut dyn RuntimeAdapter,
        deadline: Instant,
        cancellation: &dyn Cancellation,
        recovery: RecoveryPolicy,
    ) -> SwitchResult<()> {
        let context = OperationContext::new(deadline, cancellation);
        switch_exclusive(previous, target, &context, recovery)
    }

    fn run_switch(
        previous: &mut dyn RuntimeAdapter,
        target: &mut dyn RuntimeAdapter,
        deadline: Instant,
        cancellation: &dyn Cancellation,
    ) -> SwitchResult<()> {
        run_switch_with_policy(previous, target, deadline, cancellation, recovery_policy())
    }

    fn observed_deadline(adapter: &FakeAdapter, event: &str) -> Instant {
        adapter
            .observed_contexts
            .borrow()
            .iter()
            .find_map(|(observed, deadline)| (observed == event).then_some(*deadline))
            .unwrap_or_else(|| panic!("adapter did not observe context for {event}"))
    }

    #[test]
    fn happy_path_drain_precedes_stop_and_verified_release_precedes_start() {
        let (mut previous, mut target, events) = pair();
        let primary_deadline = deadline();
        run_switch(&mut previous, &mut target, primary_deadline, &NeverCancel).unwrap();
        assert_eq!(previous.state, RuntimeState::Stopped);
        assert!(!previous.admission_open);
        assert_eq!(target.state, RuntimeState::Running);
        assert!(target.admission_open);
        assert_eq!(
            events.borrow().as_slice(),
            [
                "language:state",
                "media:state",
                "language:close",
                "language:drain",
                "language:stop",
                "language:released",
                "media:start",
                "media:healthy",
                "media:open",
            ]
        );
        for event in [
            "language:state",
            "language:close",
            "language:drain",
            "language:stop",
            "language:released",
        ] {
            assert_eq!(observed_deadline(&previous, event), primary_deadline);
        }
        for event in ["media:state", "media:start", "media:healthy", "media:open"] {
            assert_eq!(observed_deadline(&target, event), primary_deadline);
        }
    }

    #[test]
    fn failed_drain_reopens_admission_without_stopping_previous() {
        let (mut previous, mut target, events) = pair();
        previous.fail = Some("drain");
        let error = run_switch(&mut previous, &mut target, deadline(), &NeverCancel).unwrap_err();
        assert_eq!(error.disposition, FailureDisposition::NoTransition);
        assert_eq!(previous.state, RuntimeState::Running);
        assert!(previous.admission_open);
        assert_eq!(target.state, RuntimeState::Stopped);
        assert!(!events.borrow().iter().any(|event| event == "media:start"));
    }

    #[test]
    fn failed_admission_close_attempts_to_reopen_without_stopping() {
        let (mut previous, mut target, events) = pair();
        previous.fail = Some("close");
        let error = run_switch(&mut previous, &mut target, deadline(), &NeverCancel).unwrap_err();
        assert_eq!(error.disposition, FailureDisposition::NoTransition);
        assert_eq!(previous.state, RuntimeState::Running);
        assert!(previous.admission_open);
        assert!(events.borrow().iter().any(|event| event == "language:open"));
        assert!(!events.borrow().iter().any(|event| event == "language:stop"));
    }

    #[test]
    fn failed_target_health_cleans_target_then_restores_previous() {
        let (mut previous, mut target, events) = pair();
        target.fail = Some("healthy");
        let error = run_switch(&mut previous, &mut target, deadline(), &NeverCancel).unwrap_err();
        assert_eq!(error.disposition, FailureDisposition::PreviousRestored);
        assert!(format!("{error:#}").contains("previous runtime restored"));
        assert_eq!(previous.state, RuntimeState::Running);
        assert!(previous.admission_open);
        assert_eq!(target.state, RuntimeState::Stopped);
        let events = events.borrow();
        let target_stop = events
            .iter()
            .position(|event| event == "media:stop")
            .unwrap();
        let previous_start = events
            .iter()
            .position(|event| event == "language:start")
            .unwrap();
        assert!(target_stop < previous_start);
    }

    #[test]
    fn unknown_release_and_external_ownership_fail_closed() {
        let (mut previous, mut target, events) = pair();
        previous.fail = Some("released");
        let error = run_switch(&mut previous, &mut target, deadline(), &NeverCancel).unwrap_err();
        assert_eq!(error.disposition, FailureDisposition::OwnershipUncertain);
        assert!(format!("{error:#}").contains("ownership is uncertain"));
        assert!(!events.borrow().iter().any(|event| event == "media:start"));
        assert!(!events
            .borrow()
            .iter()
            .any(|event| event == "language:start"));

        let (mut previous, mut target, events) = pair();
        previous.owner = LifecycleOwner::External;
        let error = run_switch(&mut previous, &mut target, deadline(), &NeverCancel).unwrap_err();
        assert_eq!(error.disposition, FailureDisposition::NoTransition);
        assert!(events.borrow().is_empty());
    }

    #[test]
    fn cancellation_before_admission_has_no_side_effects() {
        struct Cancel;
        impl Cancellation for Cancel {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let (mut previous, mut target, events) = pair();
        assert!(run_switch(&mut previous, &mut target, deadline(), &Cancel).is_err());
        assert!(events.borrow().is_empty());
    }

    struct CancelAfterEvent {
        events: Rc<RefCell<Vec<String>>>,
        event: &'static str,
    }

    impl Cancellation for CancelAfterEvent {
        fn is_cancelled(&self) -> bool {
            self.events.borrow().iter().any(|event| event == self.event)
        }
    }

    #[test]
    fn cancellation_during_target_activation_or_health_prevents_admission() {
        for event in ["media:start", "media:healthy"] {
            let (mut previous, mut target, events) = pair();
            let primary_deadline = deadline();
            let cancellation = CancelAfterEvent {
                events: events.clone(),
                event,
            };
            let failure = run_switch(&mut previous, &mut target, primary_deadline, &cancellation)
                .expect_err("cancellation during target start must abort before admission");

            assert_eq!(failure.disposition, FailureDisposition::PreviousRestored);
            assert!(!events.borrow().iter().any(|entry| entry == "media:open"));
            assert_eq!(target.state, RuntimeState::Stopped);
            assert!(previous.admission_open);
            assert_eq!(observed_deadline(&target, event), primary_deadline);
            let cleanup_deadline = observed_deadline(&target, "media:stop");
            assert_eq!(
                cleanup_deadline,
                observed_deadline(&previous, "language:start"),
                "target cleanup and previous restoration share one recovery budget"
            );
            assert_ne!(cleanup_deadline, primary_deadline);
        }
    }

    #[test]
    fn successful_target_admission_is_the_cancellation_commit_point() {
        let (mut previous, mut target, events) = pair();
        let cancellation = CancelAfterEvent {
            events: events.clone(),
            event: "media:open",
        };
        run_switch(&mut previous, &mut target, deadline(), &cancellation)
            .expect("cancellation after successful admission cannot roll back accepted work");

        assert_eq!(target.state, RuntimeState::Running);
        assert!(target.admission_open);
        assert_eq!(previous.state, RuntimeState::Stopped);
        assert!(events.borrow().iter().any(|event| event == "media:open"));
        assert!(!events.borrow().iter().any(|event| event == "media:stop"));
    }

    #[test]
    fn deadline_expiring_during_preflight_probe_stops_before_admission_close() {
        let (mut previous, mut target, events) = pair();
        previous.delay_on = Some(("state", Duration::from_millis(150)));
        let error = run_switch(
            &mut previous,
            &mut target,
            Instant::now() + Duration::from_millis(100),
            &NeverCancel,
        )
        .expect_err("late state probe must not allow the transition to continue");

        assert_eq!(error.disposition, FailureDisposition::NoTransition);
        assert_eq!(events.borrow().as_slice(), ["language:state"]);
        assert_eq!(target.state, RuntimeState::Stopped);
    }

    #[test]
    fn recovery_policy_requires_a_positive_timeout_with_a_finite_cap() {
        assert!(RecoveryPolicy::new(Duration::ZERO).is_err());
        assert!(RecoveryPolicy::new(RecoveryPolicy::MAX_TIMEOUT).is_ok());
        assert!(RecoveryPolicy::new(RecoveryPolicy::MAX_TIMEOUT + Duration::from_secs(1)).is_err());
    }

    #[test]
    fn recovery_deadline_exhaustion_during_target_cleanup_is_uncertain() {
        let (mut previous, mut target, events) = pair();
        target.delay_on = Some(("stop", Duration::from_millis(150)));
        let cancellation = CancelAfterEvent {
            events: events.clone(),
            event: "media:healthy",
        };
        let error = run_switch_with_policy(
            &mut previous,
            &mut target,
            deadline(),
            &cancellation,
            RecoveryPolicy::new(Duration::from_millis(100)).unwrap(),
        )
        .expect_err("cleanup must stop when its independent recovery budget expires");

        assert_eq!(error.disposition, FailureDisposition::OwnershipUncertain);
        assert!(events.borrow().iter().any(|event| event == "media:stop"));
        assert!(!events
            .borrow()
            .iter()
            .any(|event| event == "media:released"));
        assert!(!events
            .borrow()
            .iter()
            .any(|event| event == "language:start"));
        assert!(!previous.admission_open);
    }

    #[test]
    fn recovery_deadline_exhaustion_during_restore_is_uncertain() {
        let (mut previous, mut target, events) = pair();
        previous.delay_on = Some(("start", Duration::from_millis(150)));
        let cancellation = CancelAfterEvent {
            events: events.clone(),
            event: "media:healthy",
        };
        let error = run_switch_with_policy(
            &mut previous,
            &mut target,
            deadline(),
            &cancellation,
            RecoveryPolicy::new(Duration::from_millis(100)).unwrap(),
        )
        .expect_err("restoration must stop when the shared recovery budget expires");

        assert_eq!(error.disposition, FailureDisposition::OwnershipUncertain);
        assert!(events
            .borrow()
            .iter()
            .any(|event| event == "media:released"));
        assert!(events
            .borrow()
            .iter()
            .any(|event| event == "language:start"));
        assert!(!events
            .borrow()
            .iter()
            .any(|event| event == "language:healthy"));
        assert!(!events.borrow().iter().any(|event| event == "language:open"));
        assert_eq!(target.state, RuntimeState::Stopped);
    }
}
