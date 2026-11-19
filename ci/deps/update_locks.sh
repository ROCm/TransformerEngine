#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Regenerate ci/deps/locks from ci/deps/requirements: per GPU arch in config.env's
# GPU_ARCHS, <arch>-all.txt for the venv with both frameworks, and one lock per framework's
# own venv. The framework locks are constrained by the combined one, so the packages they
# share get the same version and are downloaded once. Also refreshes AITER_CK_COMMIT in
# config.env.
#
# Usage: ci/deps/update_locks.sh [--upgrade]
#   Existing locks are kept where the requirements allow; --upgrade re-resolves
#   every package to its newest allowed version.
# Uses `uv` from PATH when it is the pinned version, else downloads that version.

set -euo pipefail

DEPS_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$DEPS_DIR/requirements/config.env"

UPGRADE=()
case "${1:-}" in
    "") ;;
    --upgrade) UPGRADE=(--upgrade) ;;
    *) echo "Usage: $0 [--upgrade]" >&2; exit 1 ;;
esac

if command -v uv > /dev/null && [[ "$(uv --version)" == "uv ${UV_VERSION} "* ]]; then
    UV=uv
else
    UV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/te-ci/uv-${UV_VERSION}"
    UV="$UV_DIR/uv"
    if [[ ! -x "$UV" ]]; then
        mkdir -p "$UV_DIR"
        curl -sSfL -o "$UV_DIR/uv.tar.gz" \
            "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
        echo "${UV_SHA256}  $UV_DIR/uv.tar.gz" | sha256sum -c --quiet
        tar xzf "$UV_DIR/uv.tar.gz" -C "$UV_DIR" --strip-components=1
    fi
fi
# The package sources from config.env, and no uv config files, so a personal one
# can't change what is installed
export UV_EXTRA_INDEX_URL UV_INDEX_STRATEGY UV_NO_CONFIG=1

# The CK commit aiter pins, read from GitHub since the archives aiter is installed
# from carry no submodules
ck=$(curl -sSf "https://api.github.com/repos/ROCm/aiter/contents/3rdparty/composable_kernel?ref=${AITER_COMMIT}" \
     | grep -o '"sha": *"[0-9a-f]\{40\}"' | head -1 | grep -o '[0-9a-f]\{40\}')
sed -i "s/^AITER_CK_COMMIT=.*/AITER_CK_COMMIT=${ck}/" "$DEPS_DIR/requirements/config.env"
echo "aiter ${AITER_COMMIT:0:8}: CK ${ck:0:8}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Linux x86_64 with Ubuntu 24.04's glibc, the base image's
compile=( "$UV" pip compile --quiet --python-version "$PYTHON_VERSION"
          --python-platform x86_64-manylinux_2_39 --generate-hashes --emit-index-annotation
          --custom-compile-command "ci/deps/update_locks.sh" "${UPGRADE[@]}" )
# uv runs inside $tmp so the locks' "via" comments name the inputs without a temp path
cd "$tmp"
for arch in $GPU_ARCHS; do
    for req in common torch jax; do
        sed "s/\${GPU_ARCH}/${arch}/g" "$DEPS_DIR/requirements/${req}.in" > "${req}.in"
    done
    printf -- '-r torch.in\n-r jax.in\n' > all.in
    "${compile[@]}" all.in -o "$DEPS_DIR/locks/${arch}-all.txt"
    ln -sf "$DEPS_DIR/locks/${arch}-all.txt" all.txt
    for fw in torch jax; do
        "${compile[@]}" "${fw}.in" -c all.txt -o "$DEPS_DIR/locks/${arch}-${fw}.txt"
    done
    echo "${arch}: $(grep -c '^[a-z0-9]' "$DEPS_DIR/locks/${arch}-torch.txt") torch," \
         "$(grep -c '^[a-z0-9]' "$DEPS_DIR/locks/${arch}-jax.txt") jax packages"
done
# Run as root in a container, keep the files owned by the repo's owner
if [[ $EUID -eq 0 ]]; then
    chown --reference="$DEPS_DIR/requirements/common.in" "$DEPS_DIR"/locks/*.txt "$DEPS_DIR/requirements/config.env"
fi
