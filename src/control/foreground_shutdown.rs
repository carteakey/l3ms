//! Exclusive, scoped signal ownership for dedicated foreground operations.
//! Handlers only set a lock-free flag; callers perform bounded cleanup.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use super::runtime::Cancellation;

static OWNER: Mutex<()> = Mutex::new(());
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn request_stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

pub(crate) struct ShutdownSignals {
    _owner: MutexGuard<'static, ()>,
    interrupt: libc::sigaction,
    terminate: libc::sigaction,
}

impl ShutdownSignals {
    /// Only call from a process dedicated to a foreground operation, not from
    /// the interactive UI or a library worker. Original handlers are restored.
    pub(crate) fn install() -> Result<Self, ()> {
        let owner = OWNER.try_lock().map_err(|_| ())?;
        STOP.store(false, Ordering::Relaxed);
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut interrupt: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut terminate: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = request_stop as *const () as usize;
        unsafe {
            if libc::sigemptyset(&mut action.sa_mask) != 0
                || libc::sigaction(libc::SIGINT, &action, &mut interrupt) != 0
            {
                return Err(());
            }
            if libc::sigaction(libc::SIGTERM, &action, &mut terminate) != 0 {
                libc::sigaction(libc::SIGINT, &interrupt, std::ptr::null_mut());
                return Err(());
            }
        }
        Ok(Self {
            _owner: owner,
            interrupt,
            terminate,
        })
    }
}

impl Cancellation for ShutdownSignals {
    fn is_cancelled(&self) -> bool {
        STOP.load(Ordering::Relaxed)
    }
}

impl Drop for ShutdownSignals {
    fn drop(&mut self) {
        unsafe {
            libc::sigaction(libc::SIGINT, &self.interrupt, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &self.terminate, std::ptr::null_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_is_exclusive_and_requests_cooperative_stop() {
        let signals = ShutdownSignals::install().unwrap();
        assert!(!signals.is_cancelled());
        assert!(ShutdownSignals::install().is_err());
        request_stop(libc::SIGTERM);
        assert!(signals.is_cancelled());
        drop(signals);
        let signals = ShutdownSignals::install().unwrap();
        assert!(!signals.is_cancelled());
    }
}
