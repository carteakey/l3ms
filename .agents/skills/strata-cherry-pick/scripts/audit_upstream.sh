#!/usr/bin/env bash
# Audit upstream Strata changes and what the lean fork still owes it.
#
# v2 (2026-10-28): compares against the NEWEST tag instead of the tag the fork was cut
# from, shows the commits that are still ours, and checks whether a switch is already
# upstream before it becomes a cherry-pick candidate. That last part matters: on 2026-10-28
# two of the fork's seven commits (the sm_89 MMVQ-IL table and STRATA_GR_NORM_FAST) had
# already been absorbed by v0.2.7, and the same check showed that the two levers I most
# wanted to port (trimmed MTP draft head, cache-conditioned routing) already ship as
# --mtp-draft-vocab and STRATA_ROUTE_PRIOR.
set -eu   # no pipefail here: '| head' closes a pipe, and a closed pipe is not an error in an audit script

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/vendor/strata-lean" ]; do ROOT="$(dirname "$ROOT")"; done
if [ ! -d "$ROOT/vendor/strata-lean" ]; then echo "could not find a tree containing vendor/strata-lean above $0"; exit 1; fi
cd "$ROOT/vendor/strata-lean"
git fetch --tags -q origin

NEWEST="${NEWEST:-$(git tag --sort=-v:refname | head -n1)}"
BASE="$(git merge-base HEAD "$NEWEST")"

echo "## Tags"
git tag --sort=-v:refname | head -8 || true
echo
echo "## Head"
git log -1 --oneline
echo "Branch: $(git branch --show-current 2>/dev/null || echo detached) | fork cut from: $(git describe --tags --abbrev=0 "$BASE" 2>/dev/null || echo '?')"
echo "Newest upstream tag: $NEWEST | merge base: $(git log -1 --format='%h %ad %s' --date=short "$BASE")"

echo
echo "## Upstream commits since the fork's merge base"
echo "all:                $(git rev-list --count "$BASE..$NEWEST")"
echo "core + cuda + spec: $(git rev-list --count "$BASE..$NEWEST" -- src/core src/kernels/cuda src/spec)"
echo "kernels/cuda only:  $(git rev-list --count "$BASE..$NEWEST" -- src/kernels/cuda)"
echo "src/spec only:      $(git rev-list --count "$BASE..$NEWEST" -- src/spec)"

echo
echo "## Commits that are still ours (a re-slice onto $NEWEST has to re-apply these)"
git log "$NEWEST..HEAD" --oneline

echo
echo "## Is it already upstream? (check before porting)"
for sym in STRATA_GR_NORM_FAST STRATA_ROUTE_PRIOR STRATA_SPEC_COST STRATA_KV_GROW STRATA_PROMPT_ATTN_IMMA STRATA_KVA route_prior native_mmvq fused_gr; do
  hits=$(git grep -l -e "$sym" "$NEWEST" -- src 2>/dev/null | wc -l || echo 0)
  lean=$(git grep -l -e "$sym" HEAD -- src 2>/dev/null | wc -l || echo 0)
  if [ "$hits" -gt 0 ]; then
    printf '%-26s upstream: %s file(s)%s\n' "$sym" "$hits" "$([ "$lean" -gt 0 ] && echo ' | also in lean')"
  else
    printf '%-26s not in %s\n' "$sym" "$NEWEST"
  fi
done

echo
echo "## New STRATA_* switches between the merge base and $NEWEST (what a re-slice unlocks)"
git grep -ho "STRATA_[A-Z0-9_]*" "$BASE" -- src | sort -u > /tmp/strata-old-env.txt || true
git grep -ho "STRATA_[A-Z0-9_]*" "$NEWEST" -- src | sort -u > /tmp/strata-new-env.txt || true
echo "merge base $(git log -1 --format=%h "$BASE"): $(wc -l < /tmp/strata-old-env.txt) | $NEWEST: $(wc -l < /tmp/strata-new-env.txt) | new: $(comm -13 /tmp/strata-old-env.txt /tmp/strata-new-env.txt | wc -l)"
comm -13 /tmp/strata-old-env.txt /tmp/strata-new-env.txt | head -30 || true
echo "(full list: comm -13 /tmp/strata-old-env.txt /tmp/strata-new-env.txt)"

echo
echo "## Core-path commits worth reading ($BASE..$NEWEST)"
git log "$BASE..$NEWEST" --oneline --no-merges -- src/core src/kernels/cuda src/spec | head -40 || true

echo
echo "## Re-slice recipe (preferred over a long cherry-pick chain)"
cat <<TXT
git worktree add -b 4070-main-lite-${NEWEST#v} ../strata-${NEWEST#v} $NEWEST
cd ../strata-${NEWEST#v}
git cherry-pick -x <the commits listed above as still ours>
# expect the conflict in src/core/gr.cpp and src/kernels/cuda/fused_gr.cu: 0.2.x reworked
# the hyper-connection path the lean inject/gate-mean fold (8645328d) patches.
TXT
