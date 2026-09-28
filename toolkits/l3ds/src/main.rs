//! l3ds: Local Low-latency Decision System CLI
mod arena;
mod bench;
mod manifest;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

#[derive(Parser)]
#[command(
    name = "l3ds",
    version = "0.1.0",
    about = "Local Low-latency Decision System: Homelab Decision Model Tooling & Benchmarks"
)]
struct Cli {
    /// List available decision model profiles (default action if no flags provided).
    #[arg(long)]
    list: bool,

    /// Profile ID to run or test (e.g., nagi-smol, nagi-big, laya-en, semif-openjev, jev-api).
    #[arg(long)]
    profile: Option<String>,

    /// JSON state dictionary to evaluate.
    #[arg(long)]
    state: Option<String>,

    /// Question definition or name.
    #[arg(long)]
    question: Option<String>,

    /// Run the standardized benchmark suite across profiles.
    #[arg(long)]
    bench: bool,

    /// Benchmark suite name (default: "routing-gate").
    #[arg(long, default_value = "routing-gate")]
    suite: String,

    /// Number of iterations for benchmarking.
    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// Run real-time tick-based arena simulation.
    #[arg(long)]
    arena: bool,

    /// Competitor profile IDs for arena simulation.
    #[arg(long, value_delimiter = ',')]
    competitors: Vec<String>,

    /// Tick deadline in milliseconds for arena mode.
    #[arg(long, default_value_t = 100)]
    tick_ms: u64,

    /// API key for hosted System One / TypeSafe / OpenRouter API.
    #[arg(long, env = "TYPESAFE_API_KEY")]
    api_key: Option<String>,

    /// Base URL override for System One compatible API endpoint.
    #[arg(long, env = "TYPESAFE_BASE_URL")]
    base_url: Option<String>,

    /// Root directory containing decision-runtimes.json.
    #[arg(long, default_value = env!("CARGO_MANIFEST_DIR"))]
    root: PathBuf,
}

fn run(cli: Cli) -> Result<u8> {
    let manifest_data = manifest::load_manifest(&cli.root)?;

    if cli.bench {
        bench::run_benchmark(
            &cli.root,
            &manifest_data.profiles,
            cli.profile.as_deref(),
            &cli.suite,
            cli.iterations,
        )?;
        return Ok(0);
    }

    if cli.arena {
        let comps = if cli.competitors.is_empty() {
            vec![
                "nagi-big".to_string(),
                "laya-en".to_string(),
                "semif-openjev".to_string(),
                "jev-api".to_string(),
            ]
        } else {
            cli.competitors
        };
        arena::run_arena(&cli.root, &manifest_data.profiles, &comps, 20, cli.tick_ms)?;
        return Ok(0);
    }

    if let Some(profile_id) = cli.profile {
        let profile = manifest_data
            .profiles
            .iter()
            .find(|p| p.id == profile_id)
            .ok_or_else(|| anyhow::anyhow!("Profile '{}' not found in manifest", profile_id))?;

        let entrypoint = cli.root.join(&profile.entrypoint);
        let venv_python = cli.root.join(".venv/bin/python");
        let python_bin = if venv_python.exists() {
            venv_python
        } else {
            PathBuf::from("python3")
        };

        let mut cmd = Command::new(&python_bin);
        cmd.arg(&entrypoint)
            .arg("--engine")
            .arg(&profile.engine)
            .arg("--model")
            .arg(&profile.model)
            .arg("--device")
            .arg(&profile.device);

        if let Some(ref st) = cli.state {
            cmd.arg("--state").arg(st);
        }
        if let Some(ref q) = cli.question {
            cmd.arg("--question").arg(q);
        }
        if let Some(ref key) = cli.api_key {
            cmd.arg("--api-key").arg(key);
        }
        if let Some(ref base) = cli.base_url {
            cmd.arg("--base-url").arg(base);
        }

        let status = cmd.status()?;
        return Ok(if status.success() { 0 } else { 1 });
    }

    // Default: list profiles
    manifest::print_profile_list(&manifest_data.profiles);
    Ok(0)
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn default_root_loads_child_manifest_and_assets() {
        let cli = Cli::try_parse_from(["l3ds"]).expect("default CLI arguments should parse");
        let expected_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        assert_eq!(cli.root, expected_root);
        let manifest = crate::manifest::load_manifest(&cli.root)
            .expect("default root should contain decision-runtimes.json");
        assert!(!manifest.profiles.is_empty());
        assert!(cli.root.join("suites/routing-gate.json").is_file());
        assert!(cli.root.join("decision-models/arena_sim.py").is_file());
        for profile in manifest.profiles {
            assert!(
                cli.root.join(profile.entrypoint).is_file(),
                "profile entrypoint should resolve under the default root"
            );
        }
    }

    #[test]
    fn explicit_root_loads_its_manifest() {
        let root = tempfile::tempdir().expect("temporary root should be created");
        std::fs::write(
            root.path().join("decision-runtimes.json"),
            r#"{"version":"test","profiles":[]}"#,
        )
        .expect("test manifest should be written");
        let root_arg = root.path().to_str().expect("temp path should be UTF-8");
        let cli = Cli::try_parse_from(["l3ds", "--root", root_arg])
            .expect("explicit root argument should parse");

        assert_eq!(cli.root, root.path());
        let manifest = crate::manifest::load_manifest(&cli.root)
            .expect("explicit root should select its manifest");
        assert_eq!(manifest.version, "test");
        assert!(manifest.profiles.is_empty());
    }
}
