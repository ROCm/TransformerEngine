#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Install the CI test environment for one GPU arch into a plain Ubuntu 24.04 container,
# from the hash-pinned locks in ci/deps/locks (see "CI test environment" in ci/README.md).
# By default torch and JAX go into one venv, /opt/venv. A single framework gets its own,
# /opt/venv-torch or /opt/venv-jax; CI installs each framework that way. Downloads run at
# once; installs follow.
#
# Usage: ci/install_deps.sh [gfx942|gfx950] [options]
#   The arch defaults to the GPUs'.
#   --frameworks torch|jax|both   what to install (default both, into one venv)
#   --update-locks                regenerate the locks from ci/deps/requirements first
#   --no-gpu-check                skip the GPU arch and device checks (GPU-less builders)
#   --no-flash-attn               leave Flash Attention out, e.g. to build its wheel
# Writes a per-step timing table to $TE_CI_DEPS_SUMMARY when that is set.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$DIR/deps"
. "$DEPS_DIR/requirements/config.env"

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
# The venv and the lock it installs
case "$FRAMEWORKS" in
    torch|jax) VENV="/opt/venv-${FRAMEWORKS}"; LOCK=$FRAMEWORKS ;;
    both) VENV=/opt/venv; LOCK=all ;;
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
    if [[ -z "$found" || "$found" == *" "* ]]; then
        echo "Pass the GPU arch; the GPUs here are: ${found:-none}" >&2
        exit 1
    fi
    ARCH=$found
fi
if [[ -n "$GPU_CHECK" && " $found " != *" $ARCH "* ]]; then
    echo "Requested ${ARCH}, but the GPUs here are: ${found:-none}" >&2
    exit 1
fi

WORK=/tmp/te-ci-deps
rm -rf "$WORK"
mkdir -p "$WORK"
export DEBIAN_FRONTEND=noninteractive
# The package sources from config.env, and no uv config files, so a personal one
# can't change what is installed
export UV_EXTRA_INDEX_URL UV_INDEX_STRATEGY UV_NO_CONFIG=1
export UV_PYTHON_INSTALL_DIR=/opt/python
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export UV_PYTHON_PREFERENCE=only-managed

# Step timing: steps that run concurrently are timed on their own and reported as such
T0=$(date +%s%N)
STEP_START=$T0
STEPS=()
secs() { printf '%d.%d' $(( $1 / 1000 )) $(( $1 % 1000 / 100 )); }
step_done() {
    local now ms
    now=$(date +%s%N)
    ms=$(( (now - STEP_START) / 1000000 ))
    STEPS+=( "$1|$ms" )
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
    STEPS+=( "  $1 (concurrent)|$ms" )
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
    "$DEPS_DIR/update_locks.sh"
    # It may have refreshed AITER_CK_COMMIT
    . "$DEPS_DIR/requirements/config.env"
    step_done "update locks"
fi
if [[ ! -f "$DEPS_DIR/locks/${ARCH}-${LOCK}.txt" ]]; then
    echo "No lock for ${ARCH}: add it to GPU_ARCHS in ci/deps/requirements/config.env and run ci/deps/update_locks.sh" >&2
    exit 1
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

uv python install --quiet "$PYTHON_VERSION"
step_done "Python ${PYTHON_VERSION}"

# A second run, e.g. CI's other framework, takes the packages both share from uv's cache
uv venv --quiet --allow-existing --python "$PYTHON_VERSION" "$VENV"
if ! uv pip sync --python "$VENV/bin/python" --require-hashes --compile-bytecode \
        "$DEPS_DIR/locks/${ARCH}-${LOCK}.txt" > "$WORK/sync.log" 2>&1; then
    tail -30 "$WORK/sync.log" >&2
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

# Flash Attention's wheel for this venv's torch, ROCm and Python, labelled as
# ci/deps/build_fa_wheel.sh labels it
if has torch && [[ -n "$FLASH_ATTN" ]]; then
    torch_version=$("$VENV/bin/python" -c 'import torch; print(torch.__version__)')
    fa="${FLASH_ATTN_VERSION}+rocm${torch_version#*+rocm}.torch$(cut -d. -f1,2 <<< "${torch_version%%+*}").${GPU_ARCHS// /.}"
    mkdir -p "$FLASH_ATTN_LOCAL_WHEELS"
    if ! uv pip install --python "$VENV/bin/python" --no-deps --no-index --find-links "$FLASH_ATTN_WHEELS" \
            --find-links "$FLASH_ATTN_LOCAL_WHEELS" "flash-attn==${fa}" > "$WORK/fa.log" 2>&1; then
        if grep -q "no version of flash-attn" "$WORK/fa.log"; then
            echo "Flash Attention ${FLASH_ATTN_VERSION} isn't built for torch ${torch_version} and Python ${PYTHON_VERSION} yet." >&2
            echo "Build it with the \"Publish Flash Attention wheel\" workflow or ci/deps/build_fa_wheel.sh;" \
                 "see \"Flash Attention\" in ci/README.md." >&2
        else
            tail -30 "$WORK/fa.log" >&2
        fi
        exit 1
    fi
    step_done "flash-attn ${fa}"
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
if [[ -n "${TE_CI_DEPS_SUMMARY:-}" ]]; then
    {
        echo "### Test environment: ${FRAMEWORKS/both/torch and JAX} for ${ARCH}, installed at runtime"
        echo
        echo "| Step | Seconds |"
        echo "|---|---:|"
        for s in "${STEPS[@]}"; do
            echo "| ${s%|*} | $(secs "${s##*|}") |"
        done
        echo "| **Total** | **$(secs $total)** |"
    } > "$TE_CI_DEPS_SUMMARY"
fi
rm -rf "$WORK"
