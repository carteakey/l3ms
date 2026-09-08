//! Minimal launcher for editable media runtime scripts.
mod media;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// List declared profiles (also the default action).
    #[arg(long, conflicts_with = "media")]
    list: bool,
    /// Select a profile by name, runtime, or task; an empty filter opens the picker.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    media: Option<String>,
    /// Quoted arguments passed to the script without shell evaluation.
    #[arg(
        long,
        default_value = "",
        allow_hyphen_values = true,
        requires = "media"
    )]
    extra: String,
    /// Repository containing media-runtimes.json and scripts.
    #[arg(long, default_value = env!("CARGO_MANIFEST_DIR"))]
    root: PathBuf,
}

fn run(cli: Cli) -> Result<u8> {
    if let Some(filter) = cli.media {
        media::interactive_media(&cli.root, &filter, &cli.extra)
    } else {
        media::print_profile_list(&media::load_manifest(&cli.root)?.profiles);
        Ok(0)
    }
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
