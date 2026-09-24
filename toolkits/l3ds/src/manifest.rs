use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub tier: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub model: String,
    pub backbone: String,
    pub device: String,
    #[serde(default = "default_cpu")]
    pub fallback_device: String,
    pub vram_gb: f32,
    pub expected_latency_ms: u32,
    pub description: String,
    pub entrypoint: String,
}

fn default_cpu() -> String {
    "cpu".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub profiles: Vec<Profile>,
}

pub fn load_manifest(root: &Path) -> Result<Manifest> {
    let manifest_path = root.join("decision-runtimes.json");
    let content = fs::read_to_string(&manifest_path)
        .with_context(|| format!("Failed to read manifest at {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON in {}", manifest_path.display()))?;
    Ok(manifest)
}

pub fn print_profile_list(profiles: &[Profile]) {
    println!("\n  l3ds — Local Low-latency Decision System profiles");
    println!("  ================================================================");
    println!(
        "  {:<16} {:<8} {:<7} {:<8} {:<10} {:<6}",
        "PROFILE ID", "ENGINE", "TIER", "TYPE", "LATENCY", "VRAM"
    );
    println!("  ----------------------------------------------------------------");
    for p in profiles {
        println!(
            "  {:<16} {:<8} {:<7} {:<8} {:<10} {:<6.1} GB",
            p.id,
            p.engine,
            p.tier,
            p.kind,
            format!("~{}ms", p.expected_latency_ms),
            p.vram_gb
        );
        println!("    ├─ Model: {}", p.model);
        println!("    ├─ Backbone: {}", p.backbone);
        println!("    └─ Description: {}\n", p.description);
    }
}
