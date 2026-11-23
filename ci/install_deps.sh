#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Install the CI test environment for one GPU arch into a plain Ubuntu 24.04 container,
# from ci/deps/uv.lock (see "CI test environment" in ci/README.md). By default torch and
# JAX go into one venv, /opt/venv. A single framework gets its own, /opt/venv-torch or
# /opt/venv-jax; CI installs each framework that way. Downloads run at once; installs
# follow.
#
# Usage: ci/install_deps.sh [gfx942|gfx950] [options]
#   The arch defaults to the GPUs', or without GPUs and the check, to the first in pyproject.toml.
#   --frameworks torch|jax|both   what to install (default both, into one venv)
#   --update-locks                regenerate ci/deps/uv.lock from ci/deps/pyproject.toml first
#   --no-gpu-check                skip the GPU arch and device checks (GPU-less builders)
#   --no-flash-attn               leave Flash Attention out. Otherwise its wheel is built here
#                                 if it isn't published yet.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$DIR/deps"
PYPROJECT="$DEPS_DIR/pyproject.toml"
# The [tool.te-ci] settings and the GPU archs (one dependency group each) of pyproject.toml
setting() { sed -n "s/^$1 *= *\"\(.*\)\"$/\1/p" "$PYPROJECT"; }
UV_VERSION=$(setting required-version)
UV_VERSION=${UV_VERSION#==}
UV_SHA256=$(setting uv-sha256)
AITER_COMMIT=$(setting aiter-commit)
AITER_CK_COMMIT=$(setting aiter-ck-commit)
GPU_ARCHS=$(sed -n 's/^\(gfx[0-9a-f]*\) *=.*/\1/p' "$PYPROJECT" | xargs)

ARCH=""
FRAMEWORKS=both
UPDATE_LOCKS=""
GPU_CHECK=1
FLASH_ATTN=1
USAGE="Usage: $0 [gfx942|gfx950] [--frameworks torch|jax|both] [--update-locks] [--no-gpu-check] [--no-flash-attn]"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frameworks) FRAMEWORKS="$2"; shift 2 ;;
        --update-locks) UPDATE_LOCKS=1; shift ;;
        --no-gpu-check) GPU_CHECK=""; shift ;;
        --no-flash-attn) FLASH_ATTN=""; shift ;;
        gfx*) ARCH="$1"; shift ;;
        *) echo "$USAGE" >&2; exit 1 ;;
    esac
done
case "$FRAMEWORKS" in
    torch|jax) VENV="/opt/venv-${FRAMEWORKS}" ;;
    both) VENV=/opt/venv ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
has() { [[ "$FRAMEWORKS" == both || "$FRAMEWORKS" == "$1" ]]; }

# GPU arch of the visible devices, from KFD's gfx_target_version (e.g. 90402 = gfx942)
gpu_archs() {
    local v
    for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        v=$(awk '$1 == "gfx_target_version" {print $2}' "$f")
        [[ -n "$v" && "$v" != 0 ]] || continue
        printf 'gfx%d%d%x\n' $(( v / 10000 )) $(( v / 100 % 100 )) $(( v % 100 ))
    done | sort -u
}
found=$(gpu_archs 2>/dev/null | xargs)
if [[ -z "$ARCH" ]]; then
    if [[ -z "$found" && -z "$GPU_CHECK" ]]; then
        # A GPU-less builder: any arch will do
        ARCH=${GPU_ARCHS%% *}
    elif [[ -z "$found" || "$found" == *" "* ]]; then
        echo "Pass the GPU arch; the GPUs here are: ${found:-none}" >&2
        exit 1
    else
        ARCH=$found
    fi
fi
if [[ -n "$GPU_CHECK" && " $found " != *" $ARCH "* ]]; then
    echo "Requested ${ARCH}, but the GPUs here are: ${found:-none}" >&2
    exit 1
fi
if [[ " $GPU_ARCHS " != *" $ARCH "* ]]; then
    echo "ci/deps/pyproject.toml has no ${ARCH} group; its archs: ${GPU_ARCHS}" >&2
    exit 1
fi
# The dependency groups of ci/deps/pyproject.toml the venv gets
UV_GROUPS=( --group "$ARCH" )
if has torch; then
    UV_GROUPS+=( --group torch --group "torch-${ARCH}" )
fi
if has jax; then
    UV_GROUPS+=( --group jax )
fi

WORK=/tmp/te-ci-deps
rm -rf "$WORK"
mkdir -p "$WORK"
export DEBIAN_FRONTEND=noninteractive
export UV_PYTHON_INSTALL_DIR=/opt/python
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export UV_PYTHON_PREFERENCE=only-managed

# Step timing: steps that run concurrently are timed on their own and reported as such
T0=$(date +%s%N)
STEP_START=$T0
secs() { printf '%d.%d' $(( $1 / 1000 )) $(( $1 % 1000 / 100 )); }
step_done() {
    local now ms
    now=$(date +%s%N)
    ms=$(( (now - STEP_START) / 1000000 ))
    printf '== %-44s %6s s\n' "$1" "$(secs $ms)"
    STEP_START=$now
}
# Usage: timed <name> <command>... -- runs the command, records its own duration
timed() {
    local name=$1 s rc=0
    shift
    s=$(date +%s%N)
    "$@" > "$WORK/${name// /_}.log" 2>&1 || rc=$?
    echo "$(( ($(date +%s%N) - s) / 1000000 ))" > "$WORK/${name// /_}.ms"
    return $rc
}
report_bg() {
    local ms
    ms=$(cat "$WORK/${1// /_}.ms")
    printf '==   %-42s %6s s (concurrent)\n' "$1" "$(secs $ms)"
}
fail_log() {
    echo "$1 failed:" >&2
    tail -30 "$WORK/${1// /_}.log" >&2
    exit 1
}

echo "== Installing ${FRAMEWORKS/both/torch and JAX} for ${ARCH} into ${VENV}"

# The base image has no curl; apt can't run twice at once, so this goes first
if ! command -v curl > /dev/null; then
    apt-get update -qq > /dev/null
    apt-get install -y -qq --no-install-recommends curl ca-certificates > /dev/null
fi
step_done "apt: curl"

if [[ "$(uv --version 2>/dev/null)" != "uv ${UV_VERSION} "* ]]; then
    curl -sSfL --retry 3 -o "$WORK/uv.tar.gz" \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
    echo "${UV_SHA256}  $WORK/uv.tar.gz" | sha256sum -c --quiet
    tar xzf "$WORK/uv.tar.gz" -C /usr/local/bin --strip-components=1
fi
step_done "uv ${UV_VERSION}"

if [[ -n "$UPDATE_LOCKS" ]]; then
    # The CK commit aiter pins, read from GitHub since the archives aiter is installed
    # from carry no submodules
    AITER_CK_COMMIT=$(curl -sSf "https://api.github.com/repos/ROCm/aiter/contents/3rdparty/composable_kernel?ref=${AITER_COMMIT}" \
        | grep -o '"sha": *"[0-9a-f]\{40\}"' | head -1 | grep -o '[0-9a-f]\{40\}')
    sed -i "s/^aiter-ck-commit = \".*\"/aiter-ck-commit = \"${AITER_CK_COMMIT}\"/" "$PYPROJECT"
    uv lock --project "$DEPS_DIR"
    # Run as root in a container, keep the files owned by the repo's owner
    if [[ $EUID -eq 0 ]]; then
        chown --reference="$DEPS_DIR" "$PYPROJECT" "$DEPS_DIR/uv.lock"
    fi
    step_done "update locks"
fi

# --- Downloads, all at once ---
# libgomp1 is listed although build-essential brings it: `import torch` needs it
os_packages() {
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends build-essential cmake ninja-build pkg-config \
        git liblzma-dev libgomp1 libnuma1 libnuma-dev
}
timed "apt: OS packages" os_packages & APT_PID=$!

# aiter has no wheel: GitHub's archives of the pinned commits are its source
fetch_aiter() {
    mkdir -p "$WORK/aiter/3rdparty/composable_kernel"
    curl -sSfL --retry 3 -o "$WORK/aiter.tar.gz" "https://github.com/ROCm/aiter/archive/${AITER_COMMIT}.tar.gz"
    curl -sSfL --retry 3 -o "$WORK/ck.tar.gz" \
        "https://github.com/ROCm/composable_kernel/archive/${AITER_CK_COMMIT}.tar.gz"
    tar xzf "$WORK/aiter.tar.gz" -C "$WORK/aiter" --strip-components=1
    tar xzf "$WORK/ck.tar.gz" -C "$WORK/aiter/3rdparty/composable_kernel" --strip-components=1
}
AITER_PID=""
if has torch; then
    timed "aiter + CK source" fetch_aiter & AITER_PID=$!
fi

# Also installs the Python. A second run, e.g. CI's other framework, takes the packages
# both share from uv's cache.
if ! UV_PROJECT_ENVIRONMENT="$VENV" uv sync --project "$DEPS_DIR" --locked --no-default-groups \
        "${UV_GROUPS[@]}" --compile-bytecode > "$WORK/sync.log" 2>&1; then
    if grep -q "needs to be updated" "$WORK/sync.log"; then
        echo "ci/deps/uv.lock doesn't match ci/deps/pyproject.toml: run ci/install_deps.sh --update-locks" >&2
    else
        tail -30 "$WORK/sync.log" >&2
    fi
    exit 1
fi
sed -n 's/^\(Prepared\|Installed\)/   &/p' "$WORK/sync.log"
step_done "venv: download + install"

wait "$APT_PID" || fail_log "apt: OS packages"
[[ -z "$AITER_PID" ]] || wait "$AITER_PID" || fail_log "aiter + CK source"
step_done "wait for the concurrent downloads"
report_bg "apt: OS packages"
[[ -z "$AITER_PID" ]] || report_bg "aiter + CK source"

# --- Installs that need the downloads ---
"$VENV/bin/rocm-sdk" path --root > /dev/null
step_done "ROCm devel tree (rocm-sdk path --root)"

if has torch; then
    if ! ( cd "$WORK/aiter"
           ROCM_PATH="$("$VENV/bin/rocm-sdk" path --root)" \
               "$VENV/bin/python" -m pip install --no-build-isolation --no-deps --no-cache-dir . ) \
            > "$WORK/aiter-install.log" 2>&1; then
        tail -30 "$WORK/aiter-install.log" >&2
        exit 1
    fi
    step_done "aiter ${AITER_COMMIT:0:8} (from source)"
fi

# Flash Attention's wheel for this venv's torch, ROCm and Python: one built or fetched here
# earlier, the published one, or else built now
if has torch && [[ -n "$FLASH_ATTN" ]]; then
    fa_wheel=$("$DEPS_DIR/build_fa_wheel.sh" --name)
    fa=${fa_wheel#flash_attn-}
    fa=${fa%%-*}
    how=""
    if [[ ! -f "/opt/fa-wheels/${fa_wheel}" ]]; then
        rc=0
        "$DEPS_DIR/build_fa_wheel.sh" --fetch > "$WORK/fa.log" 2>&1 || rc=$?
        if [[ $rc == 1 ]]; then
            echo "== flash-attn ${fa} isn't published yet: building it, which takes a while"
            if [[ -z "${GHCR_TOKEN:-}" ]]; then
                echo "   (if its ghcr.io package is still private, GHCR_TOKEN with read:packages fetches it instead)"
            fi
            if ! VIRTUAL_ENV="$VENV" "$DEPS_DIR/build_fa_wheel.sh" > "$WORK/fa-build.log" 2>&1; then
                tail -30 "$WORK/fa-build.log" >&2
                exit 1
            fi
            how=" (built here)"
        elif [[ $rc != 0 ]]; then
            tail -30 "$WORK/fa.log" >&2
            exit 1
        fi
    fi
    uv pip install --quiet --python "$VENV/bin/python" --no-deps "/opt/fa-wheels/${fa_wheel}"
    step_done "flash-attn ${fa}${how}"
fi

# --- Checks ---
# A single framework's venv has none of the other's packages
if [[ "$FRAMEWORKS" != both ]]; then
    declare -A NOT_IN=( [torch]="jax flax" [jax]="torch flash_attn aiter flydsl torchvision" )
    for mod in ${NOT_IN[$FRAMEWORKS]}; do
        if "$VENV/bin/python" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('$mod') is None)"; then
            echo "The ${FRAMEWORKS} venv has ${mod}, which belongs to the other framework" >&2
            exit 1
        fi
    done
fi
if has torch; then
    ARCH="$ARCH" GPU_CHECK="$GPU_CHECK" FLASH_ATTN="$FLASH_ATTN" "$VENV/bin/python" - <<'EOF'
import importlib.util
import os
from importlib.metadata import version
import torch
fa = "not installed"
if os.environ["FLASH_ATTN"]:
    import flash_attn_2_cuda  # noqa: F401
    fa = version("flash-attn")
assert importlib.util.find_spec("aiter"), "aiter is not installed"
if os.environ["GPU_CHECK"]:
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch == os.environ["ARCH"], f"torch sees {arch}"
print(f"   torch {torch.__version__}, flash-attn {fa}, aiter OK")
EOF
fi
if has jax; then
    # Without this, initializing the devices reserves 75% of every GPU's memory
    XLA_PYTHON_CLIENT_PREALLOCATE=false GPU_CHECK="$GPU_CHECK" "$VENV/bin/python" - <<'EOF'
import os
import jax
if os.environ["GPU_CHECK"]:
    devices = jax.devices()
    assert devices and devices[0].platform == "gpu", f"JAX sees {devices}"
    print(f"   jax {jax.__version__}: {len(devices)} {devices[0].device_kind} devices")
else:
    print(f"   jax {jax.__version__}")
EOF
fi
step_done "checks"

total=$(( ($(date +%s%N) - T0) / 1000000 ))
echo "== Ready in $(secs $total) s: ${VENV}"
rm -rf "$WORK"
