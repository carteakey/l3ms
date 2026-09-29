# Publication audit

**Decision: NO-GO for a public GitHub push of this branch.** The imported history contains an explicit public-tunnel hostname and instructions for exposing local ComfyUI and llama-swap services. Several imported and newly added migration documents also record host-specific paths, backup locations, and machine state. A high-entropy API-key-like value appears in an unchanged configuration-reference example; the owner confirmed on 2026-09-29 that it was only an example. No large/bulk blob was found. Resolve the endpoint and path disclosures before publication, and replace the sample value with an unmistakable placeholder in the public export.

## Scope and snapshot

This recheck is pinned to committed snapshot `2287d9d822e1e0325271108d8589799218ede1ab` (`feature/l3ms-monorepo`), based on `origin/main` at `68fd7bc48f803aecb5ae5d25a3f30967b46da3eb`. There are 30 commits reachable from the branch but not `origin/main`, including the imported L2M2 and L3DS histories. At that snapshot, the worktree also had uncommitted edits in the L3DS `.gitignore`, README, setup script, Python test, and a new dependency lockfile; those files were separately checked for host-path, Funnel/host, credential, and credential-bearing-URL patterns, with no matches. The audit covers the pinned branch-only reachable objects, final committed tree, and relevant baseline files. It assumes the destination already has the `origin/main` history; if publishing to a new/different public repository, scan the full history reachable from the export ref as well. The branch advanced after this snapshot (subsequent commits include `7ccaf93` and `59800e3`); those descendants and the current worktree are not audited here and require a final recheck.

No network or service status was checked. Whether the documented Funnel endpoint is currently live is unverified.

## Findings

### Public endpoint and service exposure — blocker

[`toolkits/l2m2/docs/gpu-inference-switching.md`](../../toolkits/l2m2/docs/gpu-inference-switching.md) contains a literal Tailscale Funnel hostname at lines 3, 7, 34, and 54, and commands that configure Funnel at lines 31 and 51. The surrounding text identifies the public-facing ComfyUI REST/UI and llama-swap OpenAI-compatible API/UI, with local service ports at lines 10–12. This is operational endpoint disclosure, not a compatibility path. The document does not state an access-control posture; whether that endpoint is live now is unverified. The file is part of the reachable imported L2M2 history, so deleting it only from the latest tree would not remove the exposure from a pushed branch.

Other private/local endpoint-like strings found in root `configuration.md:529` and `docs/qwen38-flash-next-internal.md:61` are unchanged from `origin/main`; they are baseline content, not introduced by this branch. The latter documents bearer-key access via an environment variable. `toolkits/l2m2/runtime-locks/comfyui-py313-linux.txt:60` was only a version-number regex false positive, not an IP address.

### Host paths and machine-specific records — cleanup required, classify selectively

The final tracked tree at the pinned snapshot has 392 line locations matching broad absolute-home/media-path patterns; many are in files inherited unchanged from `origin/main` (133 matched files were unchanged at path level). This includes root model/service settings and older benchmark logs, so do not mistake every path match for a new import leak. The branch-added L2M2 and migration records below contain genuine host-specific strings. The generic path-pattern notation in this audit is not itself a host path.

Imported L2M2 paths appear in these tracked files (line numbers are in the audited snapshot):

| File | Matching lines | Assessment |
| --- | --- | --- |
| `toolkits/l2m2/DETAILED_SESSION.md` | 5–7, 48–49, 165–169, 177, 199, 221, 239, 313 | Historical session transcript includes local checkouts, model locations, output paths, and local auth/cache context. Prefer excluding or substantially sanitizing this transcript for public release. |
| `toolkits/l2m2/HANDOFF.md` | 5, 18, 20, 67, 90 | Handoff records live checkout, model/output, and backup locations; sanitize or keep internal. |
| `toolkits/l2m2/docs/blender-mcp-setup.md` | 42, 60 | User-specific executable path in setup examples; parameterize (for example, use a generic `uvx` lookup) if the setup contract allows it. |
| `toolkits/l2m2/docs/experiments/breeze-tts-2-baseline.md` | 38, 54, 72, 91, 101, 112 | Experiment outputs and a local audio.cpp checkout; replace with generic artifact paths or omit local run records. |
| `toolkits/l2m2/docs/experiments/h3-batman.md` | 5, 15, 37 | Local output and checkout paths; generalize or omit. |
| `toolkits/l2m2/docs/experiments/h3-reference-baseline.md` | 35 | Local output path; generalize or omit. |
| `toolkits/l2m2/docs/experiments/qwen-vs-krea2-comparison.md` | 17, 26, 40, 50 | Local output paths; generalize or omit. |
| `toolkits/l2m2/docs/extraction.md` | 3, 7 | Historical source checkout and snapshot location; use repo-relative/generic wording or keep internal. |
| `toolkits/l2m2/docs/flux-2-klein-9b-setup.md` | 24 | User/mount-specific model root in setup documentation; parameterize if this is a supported deployment location. |
| `toolkits/l2m2/docs/z-image-turbo-setup.md` | 21 | Same: user/mount-specific storage root; parameterize if supported. |
| `toolkits/l2m2/maintenance/setup-flux-2-klein-9b-comfyui.sh` | 14–15 | Script detects and defaults to a user/mount-specific model root. This may be part of local behavior; preserve its effective default only if intentional, preferably through a documented configurable variable. |
| `toolkits/l2m2/maintenance/setup-z-image-turbo-comfyui.sh` | 14–15 | Same as the FLUX setup script. |
| `toolkits/l2m2/model_downloader/MODEL_DOWNLOADER.md` | 285 | Example download destination exposes a user-specific checkout path; use a placeholder. This file's blob is byte-identical to the root `model_downloader` baseline. |
| `toolkits/l2m2/model_downloader/basic.py` | 9, 15, 21 | Local model destinations, including a live default; preserve behavior only if it is an intentional compatibility contract, otherwise make configurable. This file's blob is byte-identical to the root baseline. |
| `toolkits/l2m2/model_downloader/models_config.json` | 2 | Absolute model-root default; this affects runtime behavior and should not be blindly rewritten. Confirm the intended contract, then use a configurable root/default while retaining expected local behavior. |

Paths that identify a documented checkout/runtime contract are not secrets by themselves. In particular, genericizing an intended L3MS root, model root, or output-root default must not silently break compatibility. The session/handoff transcripts, experiment outputs, machine username/mount label, and setup-history details are not needed to express that contract and should be removed or generalized. Script/config defaults should be changed only with an equivalent configurable behavior.

The migration evidence itself is also host-specific and is not required as public product documentation. In pinned snapshot `2287d9d`, `docs/migration/status.md` has path matches at lines 4, 64, 67–68, 72, 75, and 82; those lines record local checkouts/recovery archives, host service and hardware state, and pilot execution details. `docs/migration/l2m2-prep.md` path matches are at lines 9, 14, 64, 110–114, and 135; `docs/migration/l3ds-prep.md` matches are at lines 9, 43, and 47. Keep these internal or sanitize them in the export history.

| File | Relevant lines | Finding |
| --- | --- | --- |
| `docs/migration/l2m2-prep.md` | 9, 14, 64, 110–114, 135 | Local source checkout, private backup bundle, model/audio.cpp locations, and a machine-specific invocation. |
| `docs/migration/l3ds-prep.md` | 9, 43, 47 | Local checkout, interpreter, and cache paths. |
| `docs/migration/status.md` | 4, 64, 67–68, 72, 75, 82 | Local checkout and recovery archive paths, host hardware/service state, and detailed pilot/rollback operations. |

These migration/recovery details should be sanitized or excluded from the public history, rather than treated as runtime compatibility requirements. Avoid publishing backup bundle/archive names and local deployment state. Root `CHANGELOG.md:126,548` also matches the host-path scan, but the matching lines were already present in `origin/main`; the branch’s added changelog hunks did not add a host path. Likewise, many path matches in root runtime/bench files are inherited baseline content. A new public repository would publish that ancestor content too, so do not assume “unchanged from origin/main” means safe for a first public release.

### Credential scan — no new branch-only credential found; baseline example confirmed

`gitleaks`, `trufflehog`, and `detect-secrets` were not installed in `PATH`. A high-signal text scan over all 189 unique blobs reachable from the pinned snapshot but not `origin/main`, plus its final tree, checked common provider-token prefixes (Hugging Face, GitHub, AWS, OpenAI-style, and Google API keys), private-key PEM markers, long bearer values, credential-bearing URLs, and likely secret assignments. The only branch-only match was a deliberately invalid `user:password` URL fixture in `src/state_store.rs:1188`; no likely live credential was found in the branch-only blobs.

`configuration.md:211` in the unchanged `origin/main` blob contains a high-entropy `sk-…` string in an `apiKeys` example. The surrounding reference section includes a toy key and an explicit command for generating a sample key; the active root `llama-swap.yaml:13` uses `${env.LLAMA_SWAP_API_KEY}`, not that literal. The owner confirmed on 2026-09-29 that this value was only an example and never a live credential. It was not introduced by this branch, but a public export should replace it with a clear placeholder to avoid confusion and scanner noise. A dedicated scanner remains required; this pattern scan is not a secret-free guarantee.

The generic credential-assignment scan also flagged only placeholder/environment-variable examples in `docs/vibe_configuration.md:43,47`. A broad `hf_` search hit an environment-variable name, not a Hugging Face token.

### Bulk/binary content — no newly introduced large blob found

The pinned branch delta contains 189 unique blobs totaling 2,037,245 bytes; the largest is 69,831 bytes, with none over 1 MiB. `git diff --numstat origin/main...2287d9d` reported no binary deltas, and no newly added model/archive/media artifact was found. The final imported child trees contain 104 tracked files; the largest is a 69,831-byte workflow JSON. The ZIP-classified `codex-skills/model-inventory-manager.skill` and HPGL-classified `windows/run.log` are already present in `origin/main` and unchanged by this branch. No weights or other bulk model artifacts were found in the branch-only blobs.

### Explicitly excluded source documents

The two requested exclusions are absent both from the audited tree and from the branch’s full path history:

- `toolkits/l2m2/docs/2026-09-25-comfyui-mcp-tailscale.md`
- `toolkits/l2m2/docs/sources/minimax-h3-batman-reddit-20260903.txt`

They were treated as absent from safe history; this audit did not inspect or reproduce their contents.

## Checks performed

- Established snapshot/base and branch-only commit count with `git rev-parse HEAD`, `git merge-base HEAD origin/main`, and `git rev-list --count HEAD ^origin/main`.
- Enumerated unique objects reachable from `HEAD` but not `origin/main` with `git rev-list --objects HEAD ^origin/main`; ran text-pattern checks once per unique new blob and separately checked the final tree for credential indicators, credential-bearing URLs, host-specific path forms, public Funnel commands/hostnames, and private IPv4 patterns. Reviewed reported matches to separate actual findings from examples, environment variables, test fixtures, and version-number false positives.
- Checked excluded paths in both the final tree (`git ls-tree`) and complete branch path history (`git log --full-history --name-only`).
- Counted branch/imported blobs with `git rev-list --objects` plus `git cat-file --batch-check`; checked text/binary changes with `git diff --numstat` and full-history numstat; inspected MIME classifications for the two non-text baseline files.
- Did not contact any endpoint, inspect live services, or verify whether documented host/service state remains current.

## Non-destructive export plan and pre-push gate

Do not rewrite this working branch or delete the source copies. Preserve it as the private canonical/original history. For a public release, create a separate sanitized export:

1. Save and verify a private recovery bundle of the intended source refs. Clone only the intended branch into a fresh, isolated export directory; do not point the export at a public destination yet.
2. In that export clone only, use a history-filtering pass to remove the Funnel-specific runbook (or replace it with a generic, authenticated-by-default example) from every historical commit. Exclude or sanitize the L2M2 session/handoff/experiment/extraction records and the migration preparation/status files. This preserves the L2M2/L3DS code and manifests while changing only targeted content; affected commits and descendants receive new IDs, and empty commits may be dropped. Keep the audit and recovery evidence private.
3. Sanitize user/mount paths in docs and changelog entries. Treat code/config defaults separately: retain intended L3MS/model-root compatibility behavior, but express machine roots through supported settings/environment variables with the same effective default. Do not blindly substitute paths in executable files.
4. Replace the confirmed documentation-only, high-entropy configuration-reference sample key with an unmistakable placeholder in the export history. Keep active service credentials environment-backed.
5. Scan the export clone’s complete reachable history and exact final tree with a dedicated secret scanner plus the path/endpoint checks above. Confirm the two excluded L2M2 docs remain absent, verify no large/model blobs, and run the relevant offline checks. Inspect the export ref and its remotes manually before any push.

If the public destination is a new repository, the export will include the base ancestry too; filter/scan that full reachable history and coordinate whether preserving the original merge base is acceptable. If the destination already has `origin/main`, confirm its visibility and whether it already contains the baseline candidate key before choosing a history-rewritten branch strategy. Do not force-push or change remotes without an explicit release decision. This audit records **NO-GO**, not approval to publish.
