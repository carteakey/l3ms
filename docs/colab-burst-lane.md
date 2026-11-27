# Colab burst lane (plan)

Status: **Phase 0 in progress** (2026-11-26). Owner: L3MS.

An on-demand Google Colab lane alongside the local RTX 4070, for three jobs the
12 GB card can't do well: **(a) fine-tuning**, **(b) bigger-than-VRAM models**,
and **(c) a backup endpoint** that runs without evicting the local tier. Colab is
one VM per job, not a cluster: think "burst lane", not "fleet".

## Accounts

There are two Google AI subscriptions on separate accounts. Their pools never merge.

| | `ultra` (AI Ultra 5x) | `pro` (AI Pro) |
|---|---|---|
| Colab CCU / month | 1,000 **or** 2,000 (Google's pages disagree; verify with `colab usage`) | 200 |
| CCU rollover | 90 days, oldest burned first | 90 days |
| Runs on after you disconnect (24 h max) | yes | no |
| GDP Cloud credits | $40/mo (12-month expiry) | $10/mo |
| Role | long/background jobs, A100/H100, backup endpoint | short L4 smoke tests, kernel checks |

Free trials receive 0 CCU. Every VM is wiped at 24 h (`/content` is lost).

## Why the L4 matters

The Colab L4 is Ada `sm_89`, the same architecture as the 4070. `vendor/strata-lean`
and `vendor/polystrata` builds run unmodified, so the Anti-Bloat Policy is
untouched. The L4 has 24 GB but only ~300 GB/s, against ~504 GB/s on the 4070.

- **Good for:** correctness, testing with the model fully in VRAM (e.g. Inkling 2 Mini 16.4 GiB), kernel cherry-pick smoke tests.
- **Not comparable for decode t/s.** Tag all Colab results with host/GPU via
  `bench-models/bench-env.sh` and keep them in `docs/community-runs.json`, never
  the 4070 tables in `docs/bench-results.md`.

## Cost model (unverified, from third-party probing; confirm in Phase 0)

1 CU ≈ $0.10 (pay-as-you-go price). CLI `--gpu` choices (v0.8.0): `T4, L4, G4, H100, A100`.
Availability depends on tier.

| Shape | Arch | VRAM | ~CU/hr | `pro` @200/mo | `ultra` @1,000/mo |
|---|---|---|---|---|---|
| T4 | Turing `sm_75` | 16 GB | ~1.2-1.8 | ~110-160 h | ~550-830 h |
| **L4** | **Ada `sm_89`** (= 4070) | 24 GB | ~1.94 | **~100 h (~3.3 h/day)** | ~500 h (~17 h/day) |
| **G4** | **Blackwell** (believed RTX PRO 6000, `sm_120`, unconfirmed) | 96 GB | ~10 | ~20 h | ~100 h |
| A100 | Ampere `sm_80` | 40/80 GB | ~5-13 | ~15-40 h | ~75-200 h |
| H100 | Hopper `sm_90` | 80 GB | ~16-20 | ~10 h | ~50-60 h |

G4 is the most interesting option: 96 GB holds Inkling 2 Mini, high-precision Qwen3.8 quants,
or dense 70B fully in VRAM, and it has native NVFP4/FP4 tensor cores (the 4070 only
emulates them). It needs a separate `sm_120` Strata/PolyStrata build. Whether `pro`
can get a G4 is unknown; a short allocation test will tell.

Colab bills while a VM is allocated, even idle. Use `colab run` (auto-teardown)
by default. Always-on L4 ≈ 1,400 CU/month: that only fits `ultra` if it deposits 2,000.

## Phases

### Phase 0: CLI + accounts (now)
- [x] `uv tool install google-colab-cli` (v0.8.0).
- [x] `maintenance/colab/colab-acct.sh`: per-account creds (`~/.config/colab-cli/accounts/<p>.json`, 0600) + per-account session files, driven via `--auth adc`. The upstream CLI holds only one keyring token, so without this the accounts clobber each other.
- [x] Two-step sign-in: `colab-acct.sh <p> login` prints the URL, then `login-code <code>` finishes. This replaces the CLI's `input()` prompt, which fails in scripts and agent shells. Both profiles signed in 2026-11-26.
- [ ] Record real `colab usage` balance per account (settles the 1,000 vs 2,000 question).
  - `pro`: **400.00 CU** (2026-11-26; looks like 2 × 200 rolled over).
  - `ultra`: **0.00 CU** (2026-11-26). Open question: is the account correct, has Colab been linked, is it a trial, or is the deposit stuck in sync?
- [x] Auth works end to end: a `pro` T4 session allocated and terminated cleanly.
- [x] One `pro` L4 smoke test (`maintenance/colab/smoke-l4.sh pro`): `nvidia-smi` + `torch.cuda.get_device_capability()` == `(8, 9)`. First attempt got `503 Service Unavailable` on the L4 assign (no capacity; T4 fine), so the script retries with backoff.
  - 2026-11-26 run: L4 assigned after retries (332 s including backoff). **Measured rate: 2.33 CU/hr** (the guide said ~1.94), which puts `pro` at **~86 L4-h/month**. `colab exec` then failed with `RuntimeError: Connection was lost` and **hung with the VM allocated for ~1.5 h (3.65 CU lost)** until it was killed by hand. Fix: `exec` now runs under a hard `timeout`. The capability check is still unconfirmed.
  - 2026-11-27 re-run: **PASS.** `NVIDIA L4, 23034 MiB, compute_cap 8.9`, torch `2.14.1+cu130` reports capability `(8, 9)`, so it matches the 4070's `sm_89`. 12 vCPU, 52 GiB RAM. Boot 45 s, 53 s wall, under 0.05 CU used.
  - **CUDA mismatch:** the Colab VM runs driver `580.82.07` with nvcc `13.0`; the local machine has CUDA `13.5`. Binaries built locally against 13.5 will probably fail on Colab (driver too old). Build Strata/PolyStrata on the VM with its nvcc 13.0, or locally against CUDA 13.0.
- [ ] G4 (Blackwell) on `pro`: `503 Service Unavailable` twice (2026-11-27). This can't tell no capacity apart from not included in the tier. Retry at an off-peak hour, then try `ultra` once it has a balance.
  - Lesson for Phase 2: every blocking `colab` call that runs while a VM is allocated needs a timeout. A CLI hang is a fourth way a VM can stay running.
- [ ] Measure real CU/hr for L4 from the `colab usage` rate.

### Phase 1: job runner (fine-tuning)
- [ ] `maintenance/colab/run-job.sh <profile> <gpu> <script>` wrapping `colab run`, with resumable checkpoints and `.tar.gz` Drive transfer (a Drive mount with >10k loose files hangs with `OSError: [Errno 5]`).
- [ ] First target: QLoRA (≤14B) or a media-lane LoRA on `ultra` with background execution.

### Phase 2: on-demand remote machine via llama-swap
Treat Colab as a second machine that spins up and down like any llama-swap profile,
so it spends CU only while a model is loaded:

```
request → llama-swap :8080 → profile "colab-l4-<model>"
            cmd: maintenance/colab/colab-serve.sh <profile> <gpu> <model>
              1. colab new --gpu L4             (VM boot, ~1 min)
              2. fetch GGUF from HF             (fast inside Google's network)
              3. start llama-server on the VM
              4. tunnel to localhost:<port>     (colab ssh port forward)
            ttl → llama-swap kills cmd → trap runs `colab stop`
```

**Never leave a VM running (3 layers):** (1) the wrapper's cleanup trap runs `colab stop`;
(2) a VM-side idle timer shuts the VM down if it gets no requests; (3) a cron sweep runs
`colab sessions` on both profiles and stops any leftover `l3ms-*` session. The local
process dying does not stop the VM; only `colab stop` does.

- [ ] Prototype `colab-serve.sh` with a small GGUF; manual cycle: spin up → chat request → spin down → `colab sessions` empty.
- [ ] Capacity fallback (L4 often returns 503): L4 → T4, with A100/G4 only when explicitly allowed.
- [ ] Experimental `colab-l4-*` profile in llama-swap once the manual cycle passes.
- [ ] `ttl` auto-stops the VM (stops CU burn), plus a raised health-check timeout for the cold start (VM boot + weight fetch).
- [ ] Non-exclusive `groups:` entry so waking it does **not** evict the local `:8089` tier.
- [ ] Guardrails: snapshot `llama-swap.yaml`, then `python3 maintenance/check-serving-configs.py --local-paths`.
- [ ] Personal use only: SSH/custom UIs are allowed with a paid balance, but reselling Colab as an API is banned.

### Phase 3: bigger-model evals
- [ ] A100/H100 eval harness on `ultra` (dense 70B, full-precision baselines for quant-quality A/B), results tagged into `docs/community-runs.json`.

### Optional
- Remote ComfyUI on `ultra` + `COMFYUI_URL` would let media batches run without evicting the LLM (lifts the "one blocking call" rule in AGENTS.md). Watch the ~246 s cold start plus weight download.
- GDP credits ($50/mo combined) for a paid Gemini API key as a decision-lane `suites/` judge. Keep NSFW grading in `l2m2-nsfw`.

## Operating rules
- Default to `colab run`. If you use `colab new`, always `colab stop`, and check `colab sessions` on both profiles at the end of the day.
- Never print or commit `~/.config/colab-cli/accounts/*`.
- Check the monthly deposit on each billing date; the Google One → Colab sync reportedly stalls sometimes.
