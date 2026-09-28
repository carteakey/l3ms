use anyhow::{bail, Result};
use std::path::Path;
use std::process::Command;

use crate::manifest::Profile;

pub fn run_arena(
    root: &Path,
    _profiles: &[Profile],
    competitors: &[String],
    ticks: usize,
    tick_ms: u64,
) -> Result<()> {
    println!("\n  l3ds — Real-Time Arena Simulation");
    println!("  Simulation: Bomber Grid Arena");
    println!("  Ticks: {}, Deadline per tick: {}ms", ticks, tick_ms);
    println!("  ================================================================");

    let venv_python = root.join(".venv/bin/python");
    let python_bin = if venv_python.exists() {
        venv_python
    } else {
        std::path::PathBuf::from("python3")
    };

    let arena_script = root.join("decision-models/arena_sim.py");

    let mut cmd = Command::new(&python_bin);
    cmd.arg(&arena_script)
        .arg("--ticks")
        .arg(ticks.to_string())
        .arg("--tick-ms")
        .arg(tick_ms.to_string());

    if !competitors.is_empty() {
        cmd.arg("--competitors").args(competitors);
    }

    let status = cmd.status()?;
    if !status.success() {
        bail!("Arena simulation exited with non-zero status");
    }

    Ok(())
}
