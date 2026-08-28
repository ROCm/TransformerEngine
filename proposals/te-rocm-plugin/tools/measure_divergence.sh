#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Measure ROCm-fork divergence against upstream NVIDIA TransformerEngine.
#
# Implements the normative methodology of the plugin proposal, Appendix A:
#
#   * BASE SELECTION - the upstream base is the SECOND PARENT of the fork's IFU
#     merge commit, and it must equal merge-base(upstream/main, fork/dev).
#     Never a release-branch tip, never a tag. Measuring against a release tip
#     charges upstream's post-branch-point work to ROCm (this happened in
#     manifest v2.3: 15 commits / 1,339 in-scope lines, five phantom entries,
#     and an inverted divergence trend).
#
#   * DIFF TOOL - GNU diff. added = count of '^>' lines, removed = '^<'.
#     The "lines" figure in summary tables is added+removed. git's default
#     Myers algorithm disagrees on some files and MUST NOT be substituted,
#     because M1 burn-down and the divergence-regression alarm are line-count
#     based.
#
# Usage:
#   tools/measure_divergence.sh [--base <sha>] [--upstream-remote <url>] [--per-file]
#
#   --base            Override base detection (skips the merge-base assertion).
#   --upstream-remote Upstream URL to fetch main from for the assertion.
#                     Default: https://github.com/NVIDIA/TransformerEngine.git
#   --per-file        Also emit the per-file table for Python layers.
#   --no-verify       Skip the merge-base assertion (offline use).
#
# Exit codes: 0 ok; 1 usage/repo error; 2 base assertion FAILED.

set -euo pipefail

UPSTREAM_URL="https://github.com/NVIDIA/TransformerEngine.git"
BASE=""
PER_FILE=0
VERIFY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --base)            BASE="$2"; VERIFY=0; shift 2 ;;
    --upstream-remote) UPSTREAM_URL="$2"; shift 2 ;;
    --per-file)        PER_FILE=1; shift ;;
    --no-verify)       VERIFY=0; shift ;;
    -h|--help)         sed -n '7,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "not a git repo" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"

# ---------------------------------------------------------------- base -------
if [ -z "$BASE" ]; then
  # Newest commit whose subject announces an upstream merge; take its 2nd parent.
  merge=$(git log --merges --format='%H %s' -n 200 \
          | grep -iE 'merge upstream' | head -1 | cut -d' ' -f1 || true)
  [ -n "$merge" ] || { echo "could not locate an 'Merge upstream ...' commit; pass --base" >&2; exit 1; }
  BASE=$(git log -1 --format='%P' "$merge" | awk '{print $2}')
  [ -n "$BASE" ] || { echo "merge $merge has no second parent" >&2; exit 1; }
  echo "IFU merge commit : $merge"
  echo "                   $(git log -1 --format='%s' "$merge")"
fi

echo "upstream base    : $BASE"
git log -1 --format='                   %s [%an, %ad]' --date=short "$BASE" 2>/dev/null || true
echo "fork HEAD        : $(git rev-parse HEAD)  ($(git rev-parse --abbrev-ref HEAD))"

if [ "$VERIFY" = "1" ]; then
  echo
  echo "-- asserting base == merge-base(upstream/main, HEAD) --"
  git fetch -q --no-tags "$UPSTREAM_URL" main:refs/nvidia/_verify_main 2>/dev/null || {
    echo "   WARNING: could not fetch upstream main; assertion SKIPPED" >&2; VERIFY=0; }
  if [ "$VERIFY" = "1" ]; then
    mb=$(git merge-base refs/nvidia/_verify_main HEAD)
    if [ "$mb" = "$BASE" ]; then
      echo "   OK: $mb"
    else
      echo "   FAILED" >&2
      echo "     merge-base(main, HEAD) = $mb" >&2
      echo "     detected base          = $BASE" >&2
      echo "   The fork has diverged from main other than through the IFU merge," >&2
      echo "   or the base was taken from a release branch. Do not publish these" >&2
      echo "   numbers until resolved." >&2
      exit 2
    fi
  fi
fi

# ------------------------------------------------------------- measure -------
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
set -f   # pathspecs must reach git unglobbed

measure () {  # -> "files added removed"; args: pathspec...
  local ta=0 tr=0 n=0 a r f
  while read -r f; do
    git show "$BASE:$f" > "$TMP/u" 2>/dev/null || continue
    a=$(diff "$TMP/u" "$f" | grep -c '^>' || true)
    r=$(diff "$TMP/u" "$f" | grep -c '^<' || true)
    ta=$((ta+a)); tr=$((tr+r)); n=$((n+1))
  done < <(git diff --name-only --diff-filter=M "$BASE" HEAD -- "$@")
  echo "$n $ta $tr"
}

added_only () { git diff --numstat --diff-filter=A "$BASE" HEAD -- "$@" \
                | awk '{a+=$1;n++} END{printf "%d %d\n", n+0, a+0}'; }

row () {  # label, pathspec...
  local label="$1"; shift
  read -r n a r <<<"$(measure "$@")"
  printf "  %-26s %5d %8d %9d %8d\n" "$label" "$n" "$a" "$r" "$((a+r))"
  TOTAL_SUM=$((TOTAL_SUM + a + r))
}

echo
echo "=============================================================================="
echo " MODIFIED (GNU diff, vs $(echo "$BASE" | cut -c1-12))"
echo "=============================================================================="
printf "  %-26s %5s %8s %9s %8s\n" "LAYER" "files" "added" "removed" "sum"

TOTAL_SUM=0
row "common/ C++"        'transformer_engine/common/' \
                         ':(exclude)transformer_engine/common/include/' \
                         ':(exclude)transformer_engine/common/*.py'
row "common/ Python"     'transformer_engine/common/*.py'
row "NVTE headers"       'transformer_engine/common/include/'
row "pytorch/csrc"       'transformer_engine/pytorch/csrc/'
row "jax/csrc"           'transformer_engine/jax/csrc/'
row "pytorch Python"     'transformer_engine/pytorch/*.py'
row "jax Python"         'transformer_engine/jax/*.py'
row "package root"       'transformer_engine/__init__.py'
NON_TEST=$TOTAL_SUM
row "tests/pytorch"      'tests/pytorch/'
row "tests/cpp"          'tests/cpp/'
row "tests/jax"          'tests/jax/'
row "tests/cpp_distributed" 'tests/cpp_distributed/'
echo "  ----------------------------------------------------------------------------"
printf "  %-26s %5s %8s %9s %8d\n" "non-test total" "" "" "" "$NON_TEST"
printf "  %-26s %5s %8s %9s %8d\n" "all in-scope total" "" "" "" "$TOTAL_SUM"

echo
echo "=============================================================================="
echo " ROCm-ONLY ADDITIONS (migration volume, NOT overlap debt)"
echo "=============================================================================="
printf "  %-26s %5s %8s\n" "AREA" "files" "lines"
for spec in "common/:transformer_engine/common/" \
            "pytorch Python:transformer_engine/pytorch/*.py" \
            "jax Python:transformer_engine/jax/*.py" \
            "pytorch/csrc:transformer_engine/pytorch/csrc/" \
            "jax/csrc:transformer_engine/jax/csrc/" \
            "tests/pytorch:tests/pytorch/" \
            "tests/cpp:tests/cpp/" \
            "tests/jax:tests/jax/"; do
  read -r n a <<<"$(added_only "${spec#*:}")"
  printf "  %-26s %5d %8d\n" "${spec%%:*}" "$n" "$a"
done

echo
echo "=============================================================================="
echo " SANITY"
echo "=============================================================================="
d=$(git diff --name-only --diff-filter=D "$BASE" HEAD -- \
      transformer_engine/ tests/pytorch/ tests/cpp/ tests/cpp_distributed/ tests/jax/ | wc -l)
echo "  upstream files absent from fork (in scope): $d   (expect 0)"

echo
echo "  common/ composition at base (Python CANNOT move to a C++ backend repo):"
git ls-tree -r --name-only "$BASE" transformer_engine/common \
  | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -6 | sed 's/^/    /'

if [ "$PER_FILE" = "1" ]; then
  echo
  echo "=============================================================================="
  echo " PER-FILE (Python layers)"
  echo "=============================================================================="
  while read -r f; do
    git show "$BASE:$f" > "$TMP/u" 2>/dev/null || continue
    a=$(diff "$TMP/u" "$f" | grep -c '^>' || true)
    r=$(diff "$TMP/u" "$f" | grep -c '^<' || true)
    echo "$((a+r))|$a|$r|$f"
  done < <(git diff --name-only --diff-filter=M "$BASE" HEAD -- \
             'transformer_engine/pytorch/*.py' 'transformer_engine/jax/*.py' \
             'transformer_engine/common/*.py' 'transformer_engine/__init__.py') \
  | sort -t'|' -k1,1rn \
  | while IFS='|' read -r _ a r f; do printf "  %5d /%-5d  %s\n" "$a" "$r" "$f"; done
fi

echo
echo "Regenerate the manifest from these figures; per-entry delta GROWTH is the"
echo "divergence-regression signal. added_class must be re-classified whenever the"
echo "base moves - hunk composition is base-dependent."
