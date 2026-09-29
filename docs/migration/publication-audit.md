# Publication audit

**Decision: NO-GO for a public GitHub push of this branch.** The imported history contains an explicit public-tunnel hostname and instructions for exposing local ComfyUI and llama-swap services. Several imported and newly added migration documents also record host-specific paths, backup locations, and machine state. No likely live credential was found, and no large/bulk blob was found, but the endpoint and path findings must be resolved across reachable history before publication.

## Scope and snapshot

This audit examined committed snapshot `5e405afb46ad02b2dfcb2bfde147d828b6ccb806` (`feature/l3ms-monorepo`), based on `origin/main` at `68fd7bc48f803aecb5ae5d25a3f30967b46da3eb`. There are 16 commits reachable from the branch but not `origin/main`, including the imported L2M2 and L3DS histories. The audit covered the branch delta, the final imported child trees, and relevant root files in the snapshot.

The worktree was already dirty during this check: `docs/migration/status.md`, four `src/*.rs` files, and untracked `src/control/`. Those uncommitted changes are not covered by the commit-history findings below; re-audit the exact pre-push tree after they settle. No network or service status was checked.

## Findings

### Public endpoint and service exposure — blocker

[`toolkits/l2m2/docs/gpu-inference-switching.md`](../../toolkits/l2m2/docs/gpu-inference-switching.md) contains a literal Tailscale Funnel hostname at lines 3, 7, 34, and 54, and commands that configure Funnel at lines 31 and 51. The surrounding text identifies the public-facing ComfyUI REST/UI and llama-swap OpenAI-compatible API/UI, with local service ports at lines 10–12. This is operational endpoint disclosure, not a compatibility path. The document does not state an access-control posture; whether that endpoint is live now is unverified. The file is part of the reachable imported L2M2 history, so deleting it only from the latest tree would not remove the exposure from a pushed branch.

Other private/local endpoint-like strings found in root `configuration.md:529` and `docs/qwen38-flash-next-internal.md:61` are unchanged from `origin/main`; they are baseline content, not introduced by this branch. The latter documents bearer-key access via an environment variable. `toolkits/l2m2/runtime-locks/comfyui-py313-linux.txt:60` was only a version-number regex false positive, not an IP address.

### Host paths and machine-specific records — cleanup required, classify selectively

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
| `toolkits/l2m2/model_downloader/MODEL_DOWNLOADER.md` | 285 | Example download destination exposes a user-specific checkout path; use a placeholder. |
| `toolkits/l2m2/model_downloader/basic.py` | 9, 15, 21 | Local model destinations, including a live default; preserve behavior only if it is an intentional compatibility contract, otherwise make configurable. |
| `toolkits/l2m2/model_downloader/models_config.json` | 2 | Absolute model-root default; this affects runtime behavior and should not be blindly rewritten. Confirm the intended contract, then use a configurable root/default while retaining expected local behavior. |

Paths that identify a documented checkout/runtime contract are not secrets by themselves. In particular, genericizing an intended L3MS root, model root, or output-root default must not silently break compatibility. The session/handoff transcripts, experiment outputs, machine username/mount label, and setup-history details are not needed to express that contract and should be removed or generalized. Script/config defaults should be changed only with an equivalent configurable behavior.

The migration evidence itself is also host-specific and is not required as public product documentation:

| File | Relevant lines | Finding |
| --- | --- | --- |
| `docs/migration/l2m2-prep.md` | 9, 14, 64, 110–114, 135 | Local source checkout, private backup bundle, model/audio.cpp locations, and a machine-specific invocation. |
| `docs/migration/l3ds-prep.md` | 9, 43, 47 | Local checkout, interpreter, and cache paths. |
| `docs/migration/status.md` | 4, 59, 62–63, 67, 70 | Local checkout and recovery archive paths, host hardware/service state, and detailed pilot/rollback operations. The current worktree version may contain additional uncommitted edits. |

These migration/recovery details should be sanitized or excluded from the public history, rather than treated as runtime compatibility requirements. Avoid publishing backup bundle/archive names and local deployment state.

### Credential scan — no likely live secret found; scanner gate remains

`gitleaks`, `trufflehog`, and `detect-secrets` were not installed in `PATH`. A high-signal text scan across the 16 branch-only commits and final tree checked common provider-token prefixes (including Hugging Face, GitHub, AWS, OpenAI-style, and Google API keys), private-key PEM markers, long bearer values, credential-bearing URLs, and likely secret assignments. It found no likely live credential. The generic assignment check flagged only placeholder/environment-variable examples in `docs/vibe_configuration.md:43,47`; no literal credential was present. A broad `hf_` match at `configuration.md:211` was an environment-variable name, not a token. This is not equivalent to a dedicated secret scanner and does not certify the history secret-free.

### Bulk/binary content — no newly introduced large blob found

The branch delta contains 138 unique blobs totaling 1,275,895 bytes; the largest is 69,831 bytes, with none over 1 MiB. The imported child histories contain no binary numstat deltas and no blob over 1 MiB. The final imported child tree has 102 tracked files; its largest is a 69,831-byte workflow JSON. The ZIP-classified `codex-skills/model-inventory-manager.skill` and HPGL-classified `windows/run.log` are already present in `origin/main` and unchanged by this branch. No weights or other bulk model artifacts were found in the imported history.

### Explicitly excluded source documents

The two requested exclusions are absent both from the audited tree and from the branch’s full path history:

- `toolkits/l2m2/docs/2026-09-25-comfyui-mcp-tailscale.md`
- `toolkits/l2m2/docs/sources/minimax-h3-batman-reddit-20260903.txt`

They were treated as absent from safe history; this audit did not inspect or reproduce their contents.

## Checks performed

- Established snapshot/base and branch-only commit count with `git rev-parse HEAD`, `git merge-base HEAD origin/main`, and `git rev-list --count HEAD ^origin/main`.
- Searched text in every branch-only commit and `HEAD` for credential indicators, credential-bearing URLs, host-specific path forms (`/home/...`, `/run/media/...`, `/Users/...`), public Funnel commands/hostnames, and private IPv4 patterns. Reviewed reported matches to separate actual findings from placeholders, environment variables, and version-number false positives.
- Checked excluded paths in both the final tree (`git ls-tree`) and complete branch path history (`git log --full-history --name-only`).
- Counted branch/imported blobs with `git rev-list --objects` plus `git cat-file --batch-check`; checked text/binary changes with `git diff --numstat` and full-history numstat; inspected MIME classifications for the two non-text baseline files.
- Did not contact any endpoint, inspect live services, or verify whether documented host/service state remains current.

## Pre-push gate

Keep the branch private until all of these are complete:

1. Remove or replace the Funnel hostname and exposure-specific instructions, then ensure they are absent from every commit that would be pushed (a final-tree deletion alone is insufficient).
2. Sanitize or exclude machine-specific session, handoff, experiment, and migration/recovery records across reachable history. Review script/config defaults separately and preserve their documented behavior when converting private literals to configurable paths.
3. Scan the exact push ref set—including the settled worktree and any additional refs that will be published—for secret patterns and host/endpoint disclosures. Run a dedicated secret scanner when available and resolve all findings.
4. Recheck the exact outgoing commit/blob set and confirm no unintended binary/model artifacts or newly introduced large files.

Only after those checks pass should the publication decision be revisited. This note is an audit record, not approval to publish.
