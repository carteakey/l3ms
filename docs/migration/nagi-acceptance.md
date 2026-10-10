# Nagi local acceptance — 2026-10-07

Historical evidence: generic coordinator envelope-file submission is now retired.
The old probe's `--run` refuses before starting a daemon or loading models;
its bounded helper functions remain usable by direct-run acceptance probes.
Use [direct CPU acceptance](direct-decision-acceptance.md) for the current path.

The owner approved the separate environment and required model acquisition.
Original decision environments, CUDA manifests and live services were not changed.
This is CPU native execution evidence, not a GPU-latency or routing-quality claim.

## Reproducible acquisition

`toolkits/l3ds/maintenance/prepare-nagi-smol-env.sh --install` installed its
existing 52-package pinned closure with CPython 3.12.13, Nagi 0.4.1 at source
commit `80f8c5e71832e227c8bf14cc52bbb448d3771c7a`, Torch 2.6.0+cu124 and
Transformers 5.17.0. The initial staged rename left activation/console commands
pointing at the vanished staging directory. The installer now requests a
relocatable environment; the installed environment was repaired with
`uv venv --allow-existing --relocatable` and offline exact-pin reinstallation.
Activation, the direct interpreter and `hf --help` then passed. No model or
original environment was deleted. The staged-move regression fails on the old
installer and passes after the fix.

The explicit acquisition script fetched only seven required files, not historical
training checkpoints. It verified upstream main against the pinned revisions,
verified the large-file SHA-256 values, and seeded matching offline cache refs.
The shared cache remains operator-managed; this evidence is not a permanent
runtime guarantee against later cache/reference changes.

| Repository / revision | Weight | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `nagisanzeninz/nagi-smol-v0` / `6d842eed3cf2ba619a7ecade77e00eed16f22c57` | `model.pt` | 1,920,127,806 | `ca9c530f1d4de61485d0537806254b74bb35ae1d673a7b88a47857cdea599c51` |
| `answerdotai/ModernBERT-large` / `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | `model.safetensors` | 1,583,544,840 | `44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f` |

Configuration/tokenizer hashes:

```text
nagi config.snapshot.yaml  12e89bdadf179f2a8e6531e48489751a03c3f3f25b4427245ef702cf248cfb6e
ModernBERT config.json     55dca68059e98f0debf8eb63ab355c451c65b1f5d90fdca4754113fdaadd2cca
tokenizer.json             9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1
tokenizer_config.json      3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2
special_tokens_map.json    ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049
```

Six offline acquisition safety tests, fourteen runner safety tests and the
installer relocation regression pass without model/network work.

## Native one-shot

The imported native runner was invoked with `--engine nagi`, the declared
`nagisanzeninz/nagi-smol-v0` model, `--device cpu` and bounded `--input-stdin`.
All three HF/Transformers/dataset offline flags were set, CUDA visibility was
empty, and OMP/MKL threads were limited to two. A 120-second outer deadline
bounded this test; `--simulate` was not used.

For the public synthetic state “A network switch is unreachable in the
datacenter,” with infrastructure/billing choices, the native result was:

```json
{
  "engine": "nagi-native",
  "latency_ms": 1650.0770569982706,
  "answers": {
    "route": {
      "choice": "billing",
      "probabilities": {"infra": 0.4763623774051666, "billing": 0.523637592792511},
      "confidence": 0.523637592792511
    }
  }
}
```

Execution and typed native output passed. This example's route is not semantically
correct; do not describe it as an accuracy pass or compare its CPU latency with
the manifest's unvalidated 17 ms GPU target. The base masked-LM load reports unused
prediction-head keys; the subsequent Nagi checkpoint loads strictly.

The ten-iteration native routing benchmark initially exceeded a 120-second outer
budget before emitting results. That run is not a benchmark pass. A separate
240-second bounded run completed with the same offline CPU/thread limits:

```json
{"p50_ms":1608.26,"p95_ms":1639.35,"flip_rate":0.0,"samples":40,"flips":0,"total_pairs":40}
```

This is forty standard forward passes plus forty option-order-reversed comparisons
over the four native routing cases. The harness does not evaluate correctness
against labeled answers. Zero flips is stability evidence, not accuracy or
100 ms tick-deadline compliance.

Host context: Intel i5-12600K, sixteen online CPUs, Linux `7.2.8-1-cachyos`,
CPU0 governor `powersave` and energy-performance preference `performance` at the
read-only context check. Power policy was not changed. This is an acceptance
measurement on a shared active workstation, not a controlled GPU performance
comparison.

## Private foreground coordinator

The opt-in `maintenance/probe-nagi-coordinator.py --run` passed against the
branch CLI built at `2c7d837`. The harness copied a private CPU-only profile,
linked the installed environment, forced offline model access and empty CUDA
visibility, and submitted one synthetic request with one unique caller ID.
Original manifests, services and environments were unchanged.

The submit acknowledgment matched; ten read-only status queries observed
`succeeded`. One result page contained two events, and typed output contained
one valid native answer. The private `jobs.json` contained exactly one matching
successful terminal row for `l3ds` / `nagi-smol` / `decision`, exit zero.
The daemon stopped, captured process birth identities disappeared, process
groups were empty, and the exact temporary fixture was removed. No resubmission
or simulation occurred. Eight offline harness safety regressions also pass.

This closes the CPU coordinator execution gate only. The synthetic question
does not measure routing accuracy; GPU execution, managed engine switching and
cross-client/live cutover acceptance remain open.

## Native metadata rerun — 2026-10-09

The current branch CLI and updated CPU harness completed one real offline Nagi
request with a 120-second bound. There was exactly one submission and confirmed
caller acknowledgment; fourteen read-only status queries observed success. One
result page contained three events, and CLI native output retained one answer
plus the exact model-reported latency **1,638,821,625 ns**. This is native
forward-pass timing, not coordinator execution/cold-start timing. The private
history had exactly one matching successful terminal row.

All offline flags were enabled, CUDA visibility was empty, and library thread
limits were two. No weights were downloaded, service manager invoked, GPU work
started, or original manifest/environment changed. The foreground daemon stopped,
captured owned process groups were empty, and the exact temporary fixture was
removed. Nine offline harness tests reject absent, floating-point, boolean,
overflowing or coordinator-substituted metadata. Scripts were snapshotted under
the branch's private `.toolkit/model_config_versions` before the harness update.

This closes real CPU native-metadata delivery through coordinator pages and CLI
export. It is not a GPU latency target, routing-accuracy, TUI live interaction,
mixed managed-runtime switching or crash-recovery acceptance claim.
