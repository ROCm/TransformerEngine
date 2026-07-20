#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Redeploy the static dashboard front-end to the GitHub Pages demo checkout.
#
# Copies the front-end files (app.js, index.html, styles.css, vendor/) from this
# repo into the gh-pages checkout, then commits and pushes. Data under data/ is
# left untouched by default -- this is a front-end-only redeploy (no benchmark
# rerun). Pass --with-data to also sync the local dashboard/data/ shards.
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: dashboard_redeploy.sh [options]

  --dst DIR        gh-pages checkout to deploy into (default: $TE_DASH_DST or /tmp/te-dash)
  --with-data      also sync local dashboard/data/ shards (default: front-end only)
  --bundle         also emit a self-contained single-file dashboard.html (from DST/data)
  --no-push        commit locally but do not push
  -m, --message M  commit message (default: "dashboard: redeploy front-end (<date>)")
  -h, --help       show this help

The gh-pages checkout must already exist and be on the 'gh-pages' branch. Point it
at your own GitHub Pages repo (set $TE_DASH_DST or pass --dst), e.g.:
  git clone -b gh-pages <your-gh-pages-repo-url> /tmp/te-dash
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/dashboard"
DST="${TE_DASH_DST:-/tmp/te-dash}"
WITH_DATA=0
BUNDLE=0
PUSH=1
MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dst) DST="$2"; shift 2 ;;
    --with-data) WITH_DATA=1; shift ;;
    --bundle) BUNDLE=1; shift ;;
    --no-push) PUSH=0; shift ;;
    -m|--message) MSG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

# --- sanity checks --------------------------------------------------------
[[ -f "$SRC/app.js" ]] || { echo "error: front-end not found at $SRC" >&2; exit 1; }
[[ -d "$DST/.git" ]]   || { echo "error: '$DST' is not a git checkout -- clone the gh-pages demo there first (see --help)" >&2; exit 1; }

branch="$(git -C "$DST" rev-parse --abbrev-ref HEAD)"
[[ "$branch" == "gh-pages" ]] || { echo "error: '$DST' is on branch '$branch', expected 'gh-pages'" >&2; exit 1; }

# --- copy front-end -------------------------------------------------------
echo "front-end: $SRC -> $DST"
cp "$SRC"/app.js "$SRC"/index.html "$SRC"/styles.css "$DST"/
mkdir -p "$DST/vendor"
cp "$SRC"/vendor/* "$DST"/vendor/

if [[ "$WITH_DATA" == 1 ]]; then
  echo "data:      $SRC/data -> $DST/data"
  mkdir -p "$DST/data"
  cp "$SRC"/data/index.csv "$SRC"/data/perf-*.csv "$DST"/data/
fi

if [[ "$BUNDLE" == 1 ]]; then
  echo "bundle:    $DST/dashboard.html (single-file, from $DST/data)"
  python3 "$SCRIPT_DIR/build_bundle.py" --data-dir "$DST/data" --out "$DST/dashboard.html"
fi

# --- commit + push --------------------------------------------------------
git -C "$DST" add -A
if git -C "$DST" diff --cached --quiet; then
  echo "no changes to deploy."
  exit 0
fi

[[ -n "$MSG" ]] || MSG="dashboard: redeploy front-end ($(date -u +%Y-%m-%d))"
git -C "$DST" commit -m "$MSG"

if [[ "$PUSH" == 1 ]]; then
  git -C "$DST" push origin gh-pages
  echo "pushed. Hard-refresh the demo (Ctrl/Cmd+Shift+R) -- Pages caches app.js/styles.css."
else
  echo "committed (not pushed). Push with: git -C \"$DST\" push origin gh-pages"
fi
