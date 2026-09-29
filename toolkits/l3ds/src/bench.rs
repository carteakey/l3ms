use anyhow::{bail, Result};
use std::path::Path;
use std::process::Command;

use crate::manifest::Profile;

pub fn run_benchmark(
    root: &Path,
    profiles: &[Profile],
    profile_filter: Option<&str>,
    suite_name: &str,
    iterations: usize,
    simulate: bool,
) -> Result<()> {
    let suite_path = root.join("suites").join(format!("{}.json", suite_name));
    if !suite_path.exists() {
        bail!("Benchmark suite not found at {}", suite_path.display());
    }

    println!("\n  l3ds — Benchmark Suite: {}", suite_name);
    println!("  Iterations: {}", iterations);
    if simulate {
        println!("  SYNTHETIC DEMO — not native inference or benchmark evidence");
    }
    println!("  ================================================================");

    let venv_python = root.join(".venv/bin/python");
    let python_bin = if venv_python.exists() {
        venv_python
    } else {
        std::path::PathBuf::from("python3")
    };

    let target_profiles: Vec<&Profile> = profiles
        .iter()
        .filter(|p| {
            if let Some(filter) = profile_filter {
                p.id == filter || p.engine == filter
            } else {
                true
            }
        })
        .collect();

    if target_profiles.is_empty() {
        bail!("No profiles matched filter: {:?}", profile_filter);
    }

    println!(
        "  {:<16} {:<10} {:<10} {:<10} {:<12}",
        "PROFILE", "P50 (ms)", "P95 (ms)", "FLIPS (%)", "STATUS"
    );
    println!("  ----------------------------------------------------------------");

    let mut failures = 0usize;
    for profile in target_profiles {
        let entrypoint = root.join(&profile.entrypoint);

        // Run python benchmark harness
        let mut command = Command::new(&python_bin);
        command
            .arg(&entrypoint)
            .arg("--engine")
            .arg(&profile.engine)
            .arg("--model")
            .arg(&profile.model)
            .arg("--device")
            .arg(&profile.device)
            .arg("--bench-suite")
            .arg(&suite_path)
            .arg("--iterations")
            .arg(iterations.to_string());
        if simulate {
            command.arg("--simulate");
        }
        let output = command.output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    let stdout_str = String::from_utf8_lossy(&out.stdout);
                    // Parse JSON output line from runner
                    if let Some(json_line) =
                        stdout_str.lines().find(|l| l.starts_with("BENCH_RESULT:"))
                    {
                        let json_data = &json_line["BENCH_RESULT:".len()..];
                        if let Ok(res) = serde_json::from_str::<serde_json::Value>(json_data) {
                            let p50 = res["p50_ms"].as_f64().unwrap_or(0.0);
                            let p95 = res["p95_ms"].as_f64().unwrap_or(0.0);
                            let flips = res["flip_rate"].as_f64().unwrap_or(0.0);
                            println!(
                                "  {:<16} {:<10.2} {:<10.2} {:<10.2}% {:<12}",
                                profile.id,
                                p50,
                                p95,
                                flips * 100.0,
                                "PASSED"
                            );
                            continue;
                        }
                    }
                    println!(
                        "  {:<16} {:<10} {:<10} {:<10} {:<12}",
                        profile.id, "-", "-", "-", "ERROR"
                    );
                    println!("    Raw output:\n{}", stdout_str);
                    failures += 1;
                } else {
                    let stderr_str = String::from_utf8_lossy(&out.stderr);
                    let preview = stderr_str.lines().next().unwrap_or("Failed");
                    println!(
                        "  {:<16} {:<10} {:<10} {:<10} {:<12}",
                        profile.id, "-", "-", "-", "ERROR"
                    );
                    eprintln!("    Error: {}", preview);
                    failures += 1;
                }
            }
            Err(e) => {
                println!(
                    "  {:<16} {:<10} {:<10} {:<10} {:<12}",
                    profile.id, "-", "-", "-", "SPAWN_ERR"
                );
                eprintln!("    Process error: {:#}", e);
                failures += 1;
            }
        }
    }
    println!("\n  Benchmark complete.\n");
    if failures > 0 {
        bail!("{failures} decision profile(s) failed; no synthetic fallback was counted")
    }
    Ok(())
}
