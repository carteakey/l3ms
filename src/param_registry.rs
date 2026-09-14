//! Curated registry of `llama-server` flags surfaced by the Param Builder.
//!
//! Static and deterministic by design: every entry declares its CLI flag,
//! aliases, category, control kind, and a human description. Anything the
//! registry does not model (build-specific flags like `--lazy-mode on`,
//! `--spec-*` knobs) flows through the free-form "custom flags" section so
//! import/save never silently drops text. Verify flags against the target
//! binary with `llama-server --help` before adding new entries — flag sets
//! drift between builds (see AGENTS.md).

/// Category header shown in the Param Builder list and used to group flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamCategory {
    Context,
    Gpu,
    Cache,
    Cpu,
    Load,
    Sampling,
    Server,
}

impl ParamCategory {
    pub const ALL: [Self; 7] = [
        Self::Context,
        Self::Gpu,
        Self::Cache,
        Self::Cpu,
        Self::Load,
        Self::Sampling,
        Self::Server,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Context => "Context & batching",
            Self::Gpu => "GPU & offload",
            Self::Cache => "KV cache",
            Self::Cpu => "CPU & threads",
            Self::Load => "Load & memory",
            Self::Sampling => "Sampling",
            Self::Server => "Server",
        }
    }
}

/// Control kind for one flag: drives the editor UI and flag emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Integer spinner with inclusive bounds and a +/- step.
    Integer { min: i64, max: i64, step: i64 },
    /// Boolean flag: emitted only when switched on.
    Toggle,
    /// on/off/auto: emitted as `--flag on` / `--flag off`, unset = auto.
    TriState,
    /// Pick one of the listed literals, or leave unset.
    Choice(&'static [&'static str]),
    /// Free-form single-line text (paths, tensor overrides, floats).
    Text,
}

/// One curated `llama-server` flag.
#[derive(Debug, Clone, Copy)]
pub struct ParamDef {
    /// Long flag text exactly as emitted (starts with `--`).
    pub flag: &'static str,
    /// Alternative CLI spellings accepted on import (start with `-`).
    pub aliases: &'static [&'static str],
    pub label: &'static str,
    pub category: ParamCategory,
    pub kind: ParamKind,
    pub description: &'static str,
}

impl ParamDef {
    /// True when the flag consumes a following CLI token as its value.
    pub fn takes_value(&self) -> bool {
        !matches!(self.kind, ParamKind::Toggle)
    }

    /// Choice literals for this flag, when its kind is `Choice`.
    pub fn kind_options(&self) -> Option<&'static [&'static str]> {
        match self.kind {
            ParamKind::Choice(options) => Some(options),
            _ => None,
        }
    }

    /// Map a CLI value token onto a stored value for this flag's kind.
    /// Returns `None` when the token cannot be represented (caller should
    /// fall back to a custom flag line).
    pub fn parse_value(&self, token: &str) -> Option<ParamValue> {
        match self.kind {
            ParamKind::Integer { min, max, .. } => {
                let value: i64 = token.parse().ok()?;
                (value >= min && value <= max).then_some(ParamValue::Integer(value))
            }
            ParamKind::Toggle => None,
            ParamKind::TriState => match token {
                "on" => Some(ParamValue::On),
                "off" => Some(ParamValue::Off),
                "auto" => Some(ParamValue::Unset),
                _ => None,
            },
            ParamKind::Choice(options) => options
                .iter()
                .position(|option| *option == token)
                .map(ParamValue::Choice),
            ParamKind::Text => Some(ParamValue::Text(token.to_owned())),
        }
    }
}

/// Stored value for one registry flag.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// Not set: the flag is not emitted and the binary default applies.
    Unset,
    Integer(i64),
    /// Toggle switched on.
    On,
    /// TriState explicit off (`--flag off`).
    Off,
    /// Index into the flag's `Choice` options.
    Choice(usize),
    Text(String),
}

impl ParamValue {
    pub fn is_set(&self) -> bool {
        !matches!(self, Self::Unset)
    }

    /// Compact representation for the value column of the param list.
    pub fn display(&self, def: &ParamDef) -> String {
        match self {
            Self::Unset => match def.kind {
                ParamKind::Toggle => "off".to_owned(),
                ParamKind::TriState => "auto".to_owned(),
                _ => "·".to_owned(),
            },
            Self::Integer(value) => value.to_string(),
            Self::On => "ON".to_owned(),
            Self::Off => "off".to_owned(),
            Self::Choice(index) => def
                .kind_options()
                .and_then(|options| options.get(*index))
                .map(|option| (*option).to_owned())
                .unwrap_or_else(|| "·".to_owned()),
            Self::Text(text) => text.clone(),
        }
    }
}

/// The curated flag table, in display order (grouped by category).
pub static REGISTRY: &[ParamDef] = &[
    // --- Context & batching ---
    ParamDef {
        flag: "--ctx-size",
        aliases: &["-c"],
        label: "ctx-size",
        category: ParamCategory::Context,
        kind: ParamKind::Integer {
            min: 0,
            max: 2_097_152,
            step: 1024,
        },
        description: "Prompt context size; 0 = model default",
    },
    ParamDef {
        flag: "--predict",
        aliases: &["-n", "--n-predict"],
        label: "predict",
        category: ParamCategory::Context,
        kind: ParamKind::Integer {
            min: -1,
            max: 2_097_152,
            step: 64,
        },
        description: "Tokens to predict; -1 = infinity",
    },
    ParamDef {
        flag: "--batch-size",
        aliases: &["-b"],
        label: "batch-size",
        category: ParamCategory::Context,
        kind: ParamKind::Integer {
            min: 256,
            max: 65_536,
            step: 256,
        },
        description: "Logical maximum batch size",
    },
    ParamDef {
        flag: "--ubatch-size",
        aliases: &["-ub"],
        label: "ubatch-size",
        category: ParamCategory::Context,
        kind: ParamKind::Integer {
            min: 64,
            max: 16_384,
            step: 64,
        },
        description: "Physical maximum batch size",
    },
    ParamDef {
        flag: "--keep",
        aliases: &[],
        label: "keep",
        category: ParamCategory::Context,
        kind: ParamKind::Integer {
            min: 0,
            max: 131_072,
            step: 1,
        },
        description: "Tokens to keep from the initial prompt",
    },
    ParamDef {
        flag: "--swa-full",
        aliases: &[],
        label: "swa-full",
        category: ParamCategory::Context,
        kind: ParamKind::Toggle,
        description: "Use full-size SWA cache",
    },
    ParamDef {
        flag: "--flash-attn",
        aliases: &["-fa"],
        label: "flash-attn",
        category: ParamCategory::Context,
        kind: ParamKind::TriState,
        description: "Flash Attention on/off/auto",
    },
    ParamDef {
        flag: "--rope-scaling",
        aliases: &[],
        label: "rope-scaling",
        category: ParamCategory::Context,
        kind: ParamKind::Choice(&["none", "linear", "yarn"]),
        description: "RoPE frequency scaling method",
    },
    ParamDef {
        flag: "--rope-scale",
        aliases: &[],
        label: "rope-scale",
        category: ParamCategory::Context,
        kind: ParamKind::Text,
        description: "RoPE context scaling factor",
    },
    ParamDef {
        flag: "--rope-freq-base",
        aliases: &[],
        label: "rope-freq-base",
        category: ParamCategory::Context,
        kind: ParamKind::Text,
        description: "RoPE base frequency (NTK-aware scaling)",
    },
    // --- GPU & offload ---
    ParamDef {
        flag: "--gpu-layers",
        aliases: &["-ngl", "--n-gpu-layers"],
        label: "gpu-layers",
        category: ParamCategory::Gpu,
        kind: ParamKind::Integer {
            min: 0,
            max: 999,
            step: 1,
        },
        description: "Layers in VRAM; 999 = all",
    },
    ParamDef {
        flag: "--split-mode",
        aliases: &["-sm"],
        label: "split-mode",
        category: ParamCategory::Gpu,
        kind: ParamKind::Choice(&["none", "layer", "row", "tensor"]),
        description: "Multi-GPU split strategy",
    },
    ParamDef {
        flag: "--tensor-split",
        aliases: &["-ts"],
        label: "tensor-split",
        category: ParamCategory::Gpu,
        kind: ParamKind::Text,
        description: "Per-GPU proportions, e.g. 3,1",
    },
    ParamDef {
        flag: "--main-gpu",
        aliases: &["-mg"],
        label: "main-gpu",
        category: ParamCategory::Gpu,
        kind: ParamKind::Integer {
            min: 0,
            max: 63,
            step: 1,
        },
        description: "GPU index for model or intermediate results",
    },
    ParamDef {
        flag: "--fit",
        aliases: &[],
        label: "fit",
        category: ParamCategory::Gpu,
        kind: ParamKind::TriState,
        description: "Adjust unset args to fit device memory",
    },
    ParamDef {
        flag: "--fit-target",
        aliases: &["-fitt"],
        label: "fit-target",
        category: ParamCategory::Gpu,
        kind: ParamKind::Text,
        description: "MiB of VRAM to leave FREE per device",
    },
    ParamDef {
        flag: "--fit-ctx",
        aliases: &["-fitc"],
        label: "fit-ctx",
        category: ParamCategory::Gpu,
        kind: ParamKind::Integer {
            min: 0,
            max: 2_097_152,
            step: 1024,
        },
        description: "Minimum ctx --fit may set",
    },
    ParamDef {
        flag: "--override-tensor",
        aliases: &["-ot"],
        label: "override-tensor",
        category: ParamCategory::Gpu,
        kind: ParamKind::Text,
        description: "Tensor pattern=buffer override, e.g. PLE offload",
    },
    ParamDef {
        flag: "--cpu-moe",
        aliases: &["-cmoe"],
        label: "cpu-moe",
        category: ParamCategory::Gpu,
        kind: ParamKind::Toggle,
        description: "Keep all MoE weights on CPU",
    },
    ParamDef {
        flag: "--n-cpu-moe",
        aliases: &["-ncmoe"],
        label: "n-cpu-moe",
        category: ParamCategory::Gpu,
        kind: ParamKind::Integer {
            min: 0,
            max: 999,
            step: 1,
        },
        description: "Keep first N layers of MoE weights on CPU",
    },
    ParamDef {
        flag: "--device",
        aliases: &["-dev"],
        label: "device",
        category: ParamCategory::Gpu,
        kind: ParamKind::Text,
        description: "Comma-separated devices for offload",
    },
    ParamDef {
        flag: "--no-kv-offload",
        aliases: &["-nkvo"],
        label: "no-kv-offload",
        category: ParamCategory::Gpu,
        kind: ParamKind::Toggle,
        description: "Disable KV cache offloading to GPU",
    },
    // --- KV cache ---
    ParamDef {
        flag: "--cache-type-k",
        aliases: &["-ctk"],
        label: "cache-type-k",
        category: ParamCategory::Cache,
        kind: ParamKind::Choice(&[
            "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
        ]),
        description: "KV cache data type for K",
    },
    ParamDef {
        flag: "--cache-type-v",
        aliases: &["-ctv"],
        label: "cache-type-v",
        category: ParamCategory::Cache,
        kind: ParamKind::Choice(&[
            "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
        ]),
        description: "KV cache data type for V",
    },
    // --- CPU & threads ---
    ParamDef {
        flag: "--threads",
        aliases: &["-t"],
        label: "threads",
        category: ParamCategory::Cpu,
        kind: ParamKind::Integer {
            min: 1,
            max: 256,
            step: 1,
        },
        description: "CPU threads for generation",
    },
    ParamDef {
        flag: "--threads-batch",
        aliases: &["-tb"],
        label: "threads-batch",
        category: ParamCategory::Cpu,
        kind: ParamKind::Integer {
            min: 1,
            max: 256,
            step: 1,
        },
        description: "CPU threads for batch/prompt processing",
    },
    ParamDef {
        flag: "--prio",
        aliases: &[],
        label: "prio",
        category: ParamCategory::Cpu,
        kind: ParamKind::Integer {
            min: -1,
            max: 3,
            step: 1,
        },
        description: "Process priority: -1 low … 3 realtime",
    },
    ParamDef {
        flag: "--poll",
        aliases: &[],
        label: "poll",
        category: ParamCategory::Cpu,
        kind: ParamKind::Integer {
            min: 0,
            max: 100,
            step: 1,
        },
        description: "Polling level to wait for work; 0 = off",
    },
    ParamDef {
        flag: "--cpu-range",
        aliases: &["-Cr"],
        label: "cpu-range",
        category: ParamCategory::Cpu,
        kind: ParamKind::Text,
        description: "CPU affinity range (often handled by taskset)",
    },
    ParamDef {
        flag: "--numa",
        aliases: &[],
        label: "numa",
        category: ParamCategory::Cpu,
        kind: ParamKind::Choice(&["distribute", "isolate", "numactl"]),
        description: "NUMA optimization strategy",
    },
    // --- Load & memory ---
    ParamDef {
        flag: "--load-mode",
        aliases: &["-lm"],
        label: "load-mode",
        category: ParamCategory::Load,
        kind: ParamKind::Choice(&["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]),
        description: "Model loading mode (mmap keeps PLE reads lazy)",
    },
    ParamDef {
        flag: "--check-tensors",
        aliases: &[],
        label: "check-tensors",
        category: ParamCategory::Load,
        kind: ParamKind::Toggle,
        description: "Validate model tensor data on load",
    },
    ParamDef {
        flag: "--override-kv",
        aliases: &[],
        label: "override-kv",
        category: ParamCategory::Load,
        kind: ParamKind::Text,
        description: "Model metadata overrides, KEY=TYPE:VALUE,...",
    },
    // --- Sampling ---
    ParamDef {
        flag: "--temp",
        aliases: &[],
        label: "temp",
        category: ParamCategory::Sampling,
        kind: ParamKind::Text,
        description: "Temperature (float, e.g. 1.0)",
    },
    ParamDef {
        flag: "--top-k",
        aliases: &[],
        label: "top-k",
        category: ParamCategory::Sampling,
        kind: ParamKind::Integer {
            min: 0,
            max: 200,
            step: 1,
        },
        description: "Top-K sampling",
    },
    ParamDef {
        flag: "--top-p",
        aliases: &[],
        label: "top-p",
        category: ParamCategory::Sampling,
        kind: ParamKind::Text,
        description: "Top-P sampling (float, e.g. 0.95)",
    },
    ParamDef {
        flag: "--min-p",
        aliases: &[],
        label: "min-p",
        category: ParamCategory::Sampling,
        kind: ParamKind::Text,
        description: "Min-P sampling (float)",
    },
    ParamDef {
        flag: "--repeat-penalty",
        aliases: &[],
        label: "repeat-penalty",
        category: ParamCategory::Sampling,
        kind: ParamKind::Text,
        description: "Repeat penalty (float)",
    },
    // --- Server ---
    ParamDef {
        flag: "--alias",
        aliases: &[],
        label: "alias",
        category: ParamCategory::Server,
        kind: ParamKind::Text,
        description: "Model alias for the API",
    },
    ParamDef {
        flag: "--host",
        aliases: &[],
        label: "host",
        category: ParamCategory::Server,
        kind: ParamKind::Text,
        description: "Bind address (--port is managed by llama-swap)",
    },
    ParamDef {
        flag: "--parallel",
        aliases: &["-np"],
        label: "parallel",
        category: ParamCategory::Server,
        kind: ParamKind::Integer {
            min: 1,
            max: 64,
            step: 1,
        },
        description: "Number of server slots",
    },
    ParamDef {
        flag: "--no-cont-batching",
        aliases: &[],
        label: "no-cont-batching",
        category: ParamCategory::Server,
        kind: ParamKind::Toggle,
        description: "Disable continuous batching",
    },
    ParamDef {
        flag: "--jinja",
        aliases: &[],
        label: "jinja",
        category: ParamCategory::Server,
        kind: ParamKind::Toggle,
        description: "Use the model's Jinja chat template",
    },
    ParamDef {
        flag: "--no-warmup",
        aliases: &[],
        label: "no-warmup",
        category: ParamCategory::Server,
        kind: ParamKind::Toggle,
        description: "Skip warmup pass at load",
    },
    ParamDef {
        flag: "--metrics",
        aliases: &[],
        label: "metrics",
        category: ParamCategory::Server,
        kind: ParamKind::Toggle,
        description: "Enable Prometheus /metrics endpoint",
    },
    ParamDef {
        flag: "--slots",
        aliases: &[],
        label: "slots",
        category: ParamCategory::Server,
        kind: ParamKind::Toggle,
        description: "Enable /slots endpoint",
    },
    ParamDef {
        flag: "--reasoning-effort",
        aliases: &[],
        label: "reasoning-effort",
        category: ParamCategory::Server,
        kind: ParamKind::Choice(&["minimal", "low", "medium", "high"]),
        description: "Default reasoning effort",
    },
    ParamDef {
        flag: "--reasoning-budget",
        aliases: &[],
        label: "reasoning-budget",
        category: ParamCategory::Server,
        kind: ParamKind::Integer {
            min: 0,
            max: 131_072,
            step: 1,
        },
        description: "Reasoning token budget; 0 = unlimited",
    },
];

/// Look up a registry flag by long flag or alias (exact match).
pub fn find_by_flag(token: &str) -> Option<usize> {
    REGISTRY
        .iter()
        .position(|def| def.flag == token || def.aliases.contains(&token))
}

/// First registry index in each category, for header rendering.
pub fn category_start(category: ParamCategory) -> Option<usize> {
    REGISTRY.iter().position(|def| def.category == category)
}

/// A built-in parameter preset: a name plus a sparse list of values applied
/// on top of a cleared state.
#[derive(Debug, Clone)]
pub struct Preset {
    pub name: &'static str,
    pub description: &'static str,
    /// `(long flag, value)` pairs; every flag must exist in REGISTRY.
    pub values: &'static [(&'static str, PresetValue)],
}

/// One value inside a preset definition.
#[derive(Debug, Clone, Copy)]
pub enum PresetValue {
    Int(i64),
    On,
    Off,
    Opt(&'static str),
    Text(&'static str),
}

pub static PRESETS: &[Preset] = &[
    Preset {
        name: "12 GB full-VRAM (house)",
        description: "qwen4exp house baseline: 64k ctx, q8 KV, flash-attn, fit on",
        values: &[
            ("--ctx-size", PresetValue::Int(65_536)),
            ("--gpu-layers", PresetValue::Int(999)),
            ("--flash-attn", PresetValue::On),
            ("--fit", PresetValue::On),
            ("--fit-ctx", PresetValue::Int(4_096)),
            ("--cache-type-k", PresetValue::Opt("q8_0")),
            ("--cache-type-v", PresetValue::Opt("q8_0")),
            ("--batch-size", PresetValue::Int(4_096)),
            ("--ubatch-size", PresetValue::Int(1_024)),
            ("--parallel", PresetValue::Int(1)),
            ("--jinja", PresetValue::On),
        ],
    },
    Preset {
        name: "PLE on SSD (unsloth quant)",
        description: "Token-embd table offloaded to CPU with lazy mmap reads",
        values: &[
            (
                "--override-tensor",
                PresetValue::Text(r"per_layer_token_embd\.weight=CPU"),
            ),
            ("--load-mode", PresetValue::Opt("mmap")),
            ("--fit", PresetValue::On),
            ("--flash-attn", PresetValue::On),
        ],
    },
    Preset {
        name: "MoE hybrid (n-cpu-moe ladder)",
        description: "Experts on CPU, attention on GPU; tune n-cpu-moe to fit",
        values: &[
            ("--gpu-layers", PresetValue::Int(999)),
            ("--n-cpu-moe", PresetValue::Int(8)),
            ("--flash-attn", PresetValue::On),
            ("--fit", PresetValue::On),
        ],
    },
    Preset {
        name: "Bench (max VRAM, cold)",
        description: "Bench ladder start: fit off, all layers pinned, no warmup",
        values: &[
            ("--ctx-size", PresetValue::Int(65_536)),
            ("--gpu-layers", PresetValue::Int(999)),
            ("--fit", PresetValue::Off),
            ("--flash-attn", PresetValue::On),
            ("--no-warmup", PresetValue::On),
            ("--metrics", PresetValue::On),
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_flags_are_unique_and_wellformed() {
        let mut seen = std::collections::HashSet::new();
        for def in REGISTRY {
            assert!(
                def.flag.starts_with("--"),
                "{} must be a long flag",
                def.flag
            );
            assert!(seen.insert(def.flag), "duplicate flag {}", def.flag);
            for alias in def.aliases {
                assert!(alias.starts_with('-'), "alias {alias} malformed");
                assert!(seen.insert(alias), "duplicate alias {alias}");
            }
            assert!(!def.label.is_empty());
            assert!(!def.description.is_empty());
        }
    }

    #[test]
    fn registry_covers_every_category() {
        for category in ParamCategory::ALL {
            assert!(
                category_start(category).is_some(),
                "no flags for {}",
                category.title()
            );
        }
    }

    #[test]
    fn find_by_flag_matches_aliases() {
        assert_eq!(find_by_flag("--ctx-size"), find_by_flag("-c"));
        assert_eq!(find_by_flag("--gpu-layers"), find_by_flag("-ngl"));
        assert!(find_by_flag("--nope").is_none());
    }

    #[test]
    fn parse_value_respects_bounds_and_options() {
        let ctx = find_by_flag("--ctx-size").unwrap();
        assert_eq!(
            REGISTRY[ctx].parse_value("65536"),
            Some(ParamValue::Integer(65_536))
        );
        assert_eq!(REGISTRY[ctx].parse_value("-5"), None);

        let fa = find_by_flag("--flash-attn").unwrap();
        assert_eq!(REGISTRY[fa].parse_value("on"), Some(ParamValue::On));
        assert_eq!(REGISTRY[fa].parse_value("auto"), Some(ParamValue::Unset));
        assert_eq!(REGISTRY[fa].parse_value("sure"), None);

        let temp = find_by_flag("--temp").unwrap();
        assert_eq!(
            REGISTRY[temp].parse_value("0.95"),
            Some(ParamValue::Text("0.95".into()))
        );

        let jinja = find_by_flag("--jinja").unwrap();
        assert_eq!(REGISTRY[jinja].parse_value("on"), None);
    }

    #[test]
    fn presets_reference_known_flags() {
        for preset in PRESETS {
            for (flag, _) in preset.values {
                assert!(
                    find_by_flag(flag).is_some(),
                    "preset {} references unknown flag {}",
                    preset.name,
                    flag
                );
            }
        }
    }
}
