# L3DS phase 0/1 preparation

Recorded 2026-09-28 for the [consolidation plan](../monorepo-consolidation-plan.md).
This is L3DS-only preparation evidence; it does not complete the repository-wide
Phase 0 gate.

## Source baseline

- Checkout: `/home/kchauhan/repos/l3ds`
- Branch and HEAD: `main`, `4716ddf3508d224645416f3ba70d5e34c138cedd`
- Initial and final tracked status: clean; no remotes or tags are configured.
- History: five commits, from `f35db387f2b9cbb864cde0c283a08877b9637ed9`
  through `4716ddf3508d224645416f3ba70d5e34c138cedd` (all dated 2026-09-24).

## Bounded baseline checks

| Check | Result |
| --- | --- |
| `cargo build --locked --offline` | Passed. |
| `./target/debug/l3ds --list` | Passed; listed six profiles, including the official System One profile. |
| `cargo test --locked --offline` | Could not start because locked dev dependency `bitflags 2.13.2` was not cached. |
| `cargo test --locked` | Passed after Cargo fetched its Rust dependencies; 0 tests are defined. |

No model inference, model download, hosted API request, GPU action, or service
switch was performed. The checks did update ignored build artifacts under
`l3ds/target/`; tracked source remained clean. The existing ignored `target/`
was 135 MB before these checks and is rebuildable.

## Paths and ignored runtime state

- The CLI default root is compile-time `CARGO_MANIFEST_DIR`; `--root` overrides
  it. Manifest, Python entrypoint, `.venv/bin/python`, suites, and arena script
  are resolved beneath that root. Building from `toolkits/l3ds/` therefore
  preserves the child-relative default.
- `maintenance/setup-env.sh` derives its root from its own location. README
  build/run commands assume the current directory is the L3DS crate, so Phase 2
  should show commands from `toolkits/l3ds/`.
- `decision-runtimes.json` declares `./docs/decision-runtimes.schema.json`, but
  that schema file is absent from the source checkout. The CLI currently parses
  the manifest without it.
- Ignored `.venv/` is 5.1 GB. Its Python 3.12.13 executable links to the
  host's uv-managed interpreter at
  `/home/kchauhan/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`.
  It contains Torch 2.6.0+cu124 (about 1.5 GB), NVIDIA libraries (about 2.7 GB),
  and the L3DS Python dependencies. Rebuild a fresh venv at the imported child
  path; do not copy this venv.
- The default shared Hugging Face cache is `/home/kchauhan/.cache/huggingface`
  (36 MB; no `HF_HOME` or related override is set). It contains lock/Xet cache
  state and no model `snapshots/` directories. Keep this host-level cache in
  place; it is not part of the Git import. No symlinked model or output data was
  found in the L3DS checkout.

## Filtered history snapshot

`git-filter-repo` was unavailable. A disposable clone at
`/tmp/l3ds-filtered.eogoNF` was rewritten with `git filter-branch`; the rewrite
was confined to that clone, and its `origin` remote was removed afterward.
Fetchable ref: `migration/l3ds-to-toolkits-l3ds`, HEAD
`16eb62a3db94cd7244bcf5231f25c0e98d3647de`. All five commits remain in order.
The final tree has the same 15 tracked paths, modes, and blob IDs as the source,
prefixed by `toolkits/l3ds/`; `git fsck --no-reflogs --connectivity-only` passed.

| Source commit | Filtered commit |
| --- | --- |
| `f35db387f2b9cbb864cde0c283a08877b9637ed9` | `15b015b017a1fb1220a5ea5df1e691b2a4526d78` |
| `82647b1778bc239c4ea5ab838ba40020e948b4a7` | `7e1a35757d51ac556312963f4cc86cc07798b297` |
| `9912135a0c0ef9baf1b7b9b9da655a0468b4df39` | `8ad888a4acca906cae1e5e601b56194ad431e7a4` |
| `6398cf81f5cad748b2a867b0d0f4057d308bd5ff` | `baeee72cca12ba85f28fe27cd049c517d32c839a` |
| `4716ddf3508d224645416f3ba70d5e34c138cedd` | `16eb62a3db94cd7244bcf5231f25c0e98d3647de` |

The temporary ref is ready to merge with unrelated histories into the isolated
L3MS integration branch. The following audit covers all five commits on this
ref.

### History audit and import decision

Scanned all five commits reachable from the filtered ref for common provider
token formats, private-key headers, long bearer tokens, and quoted credential
assignments; no matches were found. No credential, environment, model-weight,
database, or binary-artifact filenames were found in any historical tree. The
history contains 21 unique blobs totaling 95,324 bytes; the largest is 14,462
bytes, with none over 1 MiB. `gitleaks`, `trufflehog`, and `detect-secrets` are
not installed, so this is a high-signal pattern and artifact audit, not a
dedicated secret-scanner result.

**Decision: GO to import this ref into the isolated integration branch.** Keep
the normal pre-push secret-scan gate if a dedicated scanner becomes available.
