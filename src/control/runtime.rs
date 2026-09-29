//! In-process lifecycle contract for exclusive-resource runtime transitions.
//!
//! This module is intentionally not connected to service commands yet. A
//! backend adapter must *prove* that it can stop new direct ingress and drain
//! accepted work; a one-time idle check is not an admission barrier. Unknown
//! resource ownership or an external GPU runtime makes switching fail closed.
//! Cross-process locking and a host-local coordinator are separate release
//! gates before this can back a user-facing switch command.

use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};

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

/// A native adapter owns backend-specific admission, service, and health
/// details. `close_admission` must prevent *new direct work* until
/// `open_admission` is called; an adapter unable to do that returns an error.
/// `drain` must acknowledge that every previously accepted job has finished.
pub trait RuntimeAdapter {
    fn identity(&self) -> &RuntimeIdentity;
    fn ownership(&self) -> LifecycleOwner;
    fn resource(&self) -> ResourceClaim;
    fn state(&self) -> Result<RuntimeState>;
    fn close_admission(&mut self) -> Result<()>;
    fn drain(&mut self, deadline: Instant) -> Result<()>;
    /// Must be safe to call after partial startup, including failed activate.
    fn deactivate(&mut self) -> Result<()>;
    fn verify_released(&self) -> Result<()>;
    fn activate(&mut self) -> Result<()>;
    fn verify_healthy(&self) -> Result<()>;
    fn open_admission(&mut self) -> Result<()>;
}

/// A cancellation source supplied by the caller. Cancellation is checked
/// before accepting a transition and before activating a new runtime; a
/// cancellation after the old runtime stops triggers restoration.
pub trait Cancellation {
    fn is_cancelled(&self) -> bool;
}

pub struct NeverCancel;

impl Cancellation for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

fn restore(previous: &mut dyn RuntimeAdapter) -> Result<()> {
    previous.activate().context("reactivate previous runtime")?;
    previous
        .verify_healthy()
        .context("verify restored runtime health")?;
    previous
        .open_admission()
        .context("reopen previous runtime admission")
}

fn failed_with_restore(previous: &mut dyn RuntimeAdapter, primary: anyhow::Error) -> anyhow::Error {
    match restore(previous) {
        Ok(()) => primary.context("previous runtime restored"),
        Err(recovery) => {
            anyhow!("transition failed: {primary:#}; restoration also failed: {recovery:#}")
        }
    }
}

fn failed_during_deactivation(
    previous: &mut dyn RuntimeAdapter,
    primary: anyhow::Error,
) -> anyhow::Error {
    let observed = previous.state();
    match observed {
        Ok(RuntimeState::Running) => {
            let recovery = previous
                .verify_healthy()
                .context("verify previous runtime after failed stop")
                .and_then(|()| previous.open_admission().context("reopen previous admission"));
            match recovery {
                Ok(()) => primary.context("previous runtime remained healthy and admission reopened"),
                Err(error) => anyhow!("transition failed: {primary:#}; recovery also failed: {error:#}"),
            }
        }
        Ok(RuntimeState::Stopped) => match previous.verify_released() {
            Ok(()) => failed_with_restore(previous, primary),
            Err(error) => anyhow!(
                "transition failed: {primary:#}; GPU release is uncertain: {error:#}; refusing restart"
            ),
        },
        Ok(RuntimeState::Unknown) => anyhow!(
            "transition failed: {primary:#}; previous runtime state is unknown; refusing restart"
        ),
        Err(error) => anyhow!(
            "transition failed: {primary:#}; previous runtime inspection failed: {error:#}; refusing restart"
        ),
    }
}

/// Switch one exclusively owned GPU resource between two managed runtimes.
/// This is an in-process primitive, not an installed host coordinator.
/// Returns only after target health and admission are verified. On a failed
/// activation it attempts to stop the target and restore the previous runtime.
/// Both primary and recovery failures are surfaced together.
pub fn switch_exclusive(
    previous: &mut dyn RuntimeAdapter,
    target: &mut dyn RuntimeAdapter,
    deadline: Instant,
    cancellation: &dyn Cancellation,
) -> Result<()> {
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
        !cancellation.is_cancelled(),
        "transition cancelled before admission close"
    );
    ensure!(Instant::now() < deadline, "transition deadline expired");
    ensure!(
        previous.state()? == RuntimeState::Running,
        "previous runtime is not known running"
    );
    ensure!(
        target.state()? == RuntimeState::Stopped,
        "target runtime is not known stopped"
    );

    if let Err(error) = previous
        .close_admission()
        .context("close previous runtime admission")
    {
        return match previous.open_admission() {
            Ok(()) => Err(error),
            Err(recovery) => Err(anyhow!(
                "admission close failed: {error:#}; reopening admission also failed: {recovery:#}"
            )),
        };
    }
    if let Err(error) = previous.drain(deadline).context("drain previous runtime") {
        return match previous.open_admission() {
            Ok(()) => Err(error),
            Err(recovery) => Err(anyhow!(
                "drain failed: {error:#}; reopening admission also failed: {recovery:#}"
            )),
        };
    }
    if cancellation.is_cancelled() || Instant::now() >= deadline {
        let reason = if cancellation.is_cancelled() {
            "transition cancelled after drain"
        } else {
            "transition deadline expired after drain"
        };
        return match previous.open_admission() {
            Ok(()) => Err(anyhow!(reason)),
            Err(recovery) => Err(anyhow!(
                "{reason}; reopening admission also failed: {recovery:#}"
            )),
        };
    }

    if let Err(error) = previous.deactivate().context("deactivate previous runtime") {
        return Err(failed_during_deactivation(previous, error));
    }
    if let Err(error) = previous
        .verify_released()
        .context("verify exclusive GPU release")
    {
        bail!("{error:#}; GPU ownership is uncertain, so neither runtime was activated");
    }
    if cancellation.is_cancelled() || Instant::now() >= deadline {
        let reason = if cancellation.is_cancelled() {
            anyhow!("transition cancelled after release")
        } else {
            anyhow!("transition deadline expired after release")
        };
        return Err(failed_with_restore(previous, reason));
    }

    let activation = target
        .activate()
        .context("activate target runtime")
        .and_then(|()| {
            target
                .verify_healthy()
                .context("verify target runtime health")
        })
        .and_then(|()| {
            ensure!(
                Instant::now() < deadline,
                "transition deadline expired before target admission"
            );
            Ok(())
        })
        .and_then(|()| {
            target
                .open_admission()
                .context("open target runtime admission")
        });
    if let Err(primary) = activation {
        let cleanup = target
            .deactivate()
            .context("stop partially activated target")
            .and_then(|()| {
                target
                    .verify_released()
                    .context("verify target GPU release")
            });
        if let Err(cleanup_error) = cleanup {
            bail!(
                "transition failed: {primary:#}; target cleanup failed: {cleanup_error:#}; previous runtime was not restarted because GPU ownership is uncertain"
            );
        }
        return Err(failed_with_restore(previous, primary));
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
        events: Rc<RefCell<Vec<String>>>,
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
                events,
            }
        }

        fn event(&self, step: &str) -> Result<()> {
            self.events
                .borrow_mut()
                .push(format!("{}:{step}", self.id.as_str()));
            if self.fail == Some(step) {
                bail!("injected {step} failure")
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

        fn state(&self) -> Result<RuntimeState> {
            self.event("state")?;
            Ok(self.state)
        }

        fn close_admission(&mut self) -> Result<()> {
            self.event("close")?;
            self.admission_open = false;
            Ok(())
        }

        fn drain(&mut self, _deadline: Instant) -> Result<()> {
            self.event("drain")
        }

        fn deactivate(&mut self) -> Result<()> {
            self.event("stop")?;
            self.state = RuntimeState::Stopped;
            Ok(())
        }

        fn verify_released(&self) -> Result<()> {
            self.event("released")?;
            ensure!(self.state == RuntimeState::Stopped, "still running");
            Ok(())
        }

        fn activate(&mut self) -> Result<()> {
            self.event("start")?;
            self.state = RuntimeState::Running;
            Ok(())
        }

        fn verify_healthy(&self) -> Result<()> {
            self.event("healthy")?;
            ensure!(self.state == RuntimeState::Running, "not running");
            Ok(())
        }

        fn open_admission(&mut self) -> Result<()> {
            self.event("open")?;
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

    #[test]
    fn happy_path_drain_precedes_stop_and_verified_release_precedes_start() {
        let (mut previous, mut target, events) = pair();
        switch_exclusive(&mut previous, &mut target, deadline(), &NeverCancel).unwrap();
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
    }

    #[test]
    fn failed_drain_reopens_admission_without_stopping_previous() {
        let (mut previous, mut target, events) = pair();
        previous.fail = Some("drain");
        assert!(switch_exclusive(&mut previous, &mut target, deadline(), &NeverCancel).is_err());
        assert_eq!(previous.state, RuntimeState::Running);
        assert!(previous.admission_open);
        assert_eq!(target.state, RuntimeState::Stopped);
        assert!(!events.borrow().iter().any(|event| event == "media:start"));
    }

    #[test]
    fn failed_admission_close_attempts_to_reopen_without_stopping() {
        let (mut previous, mut target, events) = pair();
        previous.fail = Some("close");
        assert!(switch_exclusive(&mut previous, &mut target, deadline(), &NeverCancel).is_err());
        assert_eq!(previous.state, RuntimeState::Running);
        assert!(previous.admission_open);
        assert!(events.borrow().iter().any(|event| event == "language:open"));
        assert!(!events.borrow().iter().any(|event| event == "language:stop"));
    }

    #[test]
    fn failed_target_health_cleans_target_then_restores_previous() {
        let (mut previous, mut target, events) = pair();
        target.fail = Some("healthy");
        let error =
            switch_exclusive(&mut previous, &mut target, deadline(), &NeverCancel).unwrap_err();
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
        let error =
            switch_exclusive(&mut previous, &mut target, deadline(), &NeverCancel).unwrap_err();
        assert!(format!("{error:#}").contains("ownership is uncertain"));
        assert!(!events.borrow().iter().any(|event| event == "media:start"));
        assert!(!events
            .borrow()
            .iter()
            .any(|event| event == "language:start"));

        let (mut previous, mut target, events) = pair();
        previous.owner = LifecycleOwner::External;
        assert!(switch_exclusive(&mut previous, &mut target, deadline(), &NeverCancel).is_err());
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
        assert!(switch_exclusive(&mut previous, &mut target, deadline(), &Cancel).is_err());
        assert!(events.borrow().is_empty());
    }
}
