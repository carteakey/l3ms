//! Shared cancellation and time bounds for foreground control operations.

use std::time::{Duration, Instant};

use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceClaim {
    None,
    ExclusiveGpu(u32),
}

/// Bounds and cancellation supplied to one operation.
///
/// Callers should pass `remaining()` to native timeout controls and check
/// `is_cancelled()` while waiting or polling. `check_active` rejects work that
/// is already cancelled or past its deadline.
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
/// Recovery ignores caller cancellation, but every compensation remains
/// bounded.
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

/// A cancellation source supplied by the caller.
pub trait Cancellation {
    fn is_cancelled(&self) -> bool;
}

pub struct NeverCancel;

impl Cancellation for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cancel;

    impl Cancellation for Cancel {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    #[test]
    fn operation_context_rejects_cancellation_and_expired_deadlines() {
        let cancellation = Cancel;
        let cancelled =
            OperationContext::new(Instant::now() + Duration::from_secs(1), &cancellation);
        assert!(cancelled.check_active("cancelled operation").is_err());

        let never_cancel = NeverCancel;
        assert!(!never_cancel.is_cancelled());
        let expired = OperationContext::new(Instant::now(), &never_cancel);
        assert_eq!(expired.remaining(), Duration::ZERO);
        assert!(expired.check_active("expired operation").is_err());
    }

    #[test]
    fn recovery_policy_requires_a_positive_timeout_with_a_finite_cap() {
        assert!(RecoveryPolicy::new(Duration::ZERO).is_err());
        assert!(RecoveryPolicy::new(RecoveryPolicy::MAX_TIMEOUT).is_ok());
        assert!(RecoveryPolicy::new(RecoveryPolicy::MAX_TIMEOUT + Duration::from_secs(1)).is_err());
    }
}
