#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Build the Flash Attention wheel ci/install_deps.sh looks for: FLASH_ATTN_VERSION from
# config.env, compiled against the venv's torch for all GPU_ARCHS, its version labelled with
# them, e.g. 2.8.3.post2+rocm10.1.0.torch2.13.gfx942.gfx950. Run it in the container
# ci/install_deps.sh set up. The wheel goes to FLASH_ATTN_LOCAL_WHEELS, where later installs
# find it, and into the venv.
#
# Usage: ci/deps/build_fa_wheel.sh [--upload]
#   --upload   also publish it to FLASH_ATTN_WHEELS with $ARTIFACTORY_TOKEN, a token that can
#              deploy there. A published wheel is never replaced.

set -euo pipefail

DEPS_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$DEPS_DIR/requirements/config.env"

UPLOAD=""
case "${1:-}" in
    "") ;;
    --upload) UPLOAD=1 ;;
    *) echo "Usage: $0 [--upload]" >&2; exit 1 ;;
esac
if [[ -n "$UPLOAD" && -z "${ARTIFACTORY_TOKEN:-}" ]]; then
    echo "--upload needs ARTIFACTORY_TOKEN, a token that can deploy to ${FLASH_ATTN_WHEELS}" >&2
    exit 1
fi

# The active venv, else the one install_deps.sh made, if it has torch
VENV=""
for v in "${VIRTUAL_ENV:-}" /opt/venv /opt/venv-torch; do
    if [[ -n "$v" && -x "$v/bin/python" ]] && "$v/bin/python" -c "import torch" 2> /dev/null; then
        VENV=$v
        break
    fi
done
if [[ -z "$VENV" ]]; then
    echo "No venv with torch: run ci/install_deps.sh first" >&2
    exit 1
fi
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
ROCM_PATH="$(rocm-sdk path --root)"
export ROCM_PATH ROCM_HOME="$ROCM_PATH"

# The wheel install_deps.sh looks for
torch_version=$(python -c 'import torch; print(torch.__version__)')
version="${FLASH_ATTN_VERSION}+rocm${torch_version#*+rocm}.torch$(cut -d. -f1,2 <<< "${torch_version%%+*}").${GPU_ARCHS// /.}"
py=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
name="flash_attn-${version}-${py}-${py}-linux_x86_64.whl"
url="${FLASH_ATTN_WHEELS%/}/${name//+/%2B}"
whl="${FLASH_ATTN_LOCAL_WHEELS}/${name}"
published() { curl -sSfI "$url" > /dev/null 2>&1; }

if published; then
    echo "${name} is published already; ci/install_deps.sh installs it"
    exit 0
fi
if [[ -f "$whl" ]]; then
    echo "Using ${whl}, built earlier"
else
    src=$(mktemp -d)
    git -C "$src" init -q
    git -C "$src" fetch -q --depth 1 https://github.com/Dao-AILab/flash-attention.git "v${FLASH_ATTN_VERSION}"
    git -C "$src" checkout -q FETCH_HEAD
    # ROCm needs only the CK submodule. Without .git, setup.py skips its own submodule
    # update, which would also clone cutlass.
    git -C "$src" submodule update -q --init csrc/composable_kernel
    rm -rf "$src/.git"
    echo "Building flash-attn ${version}"
    mkdir -p "$FLASH_ATTN_LOCAL_WHEELS"
    ( cd "$src"
      FLASH_ATTENTION_FORCE_BUILD=TRUE GPU_ARCHS="${GPU_ARCHS// /;}" FLASH_ATTN_LOCAL_VERSION="${version#*+}" \
          python setup.py bdist_wheel -d "$FLASH_ATTN_LOCAL_WHEELS" )
    rm -rf "$src"
    if [[ ! -f "$whl" ]]; then
        echo "Expected ${whl}; the build made: $(ls "$FLASH_ATTN_LOCAL_WHEELS")" >&2
        exit 1
    fi
fi
python -m pip install --quiet --no-deps "$whl"
echo "Installed ${name} into ${VENV}"

if [[ -n "$UPLOAD" ]]; then
    if published; then
        echo "${name} was published meanwhile; not replacing it" >&2
        exit 1
    fi
    sha=$(sha256sum "$whl" | cut -d' ' -f1)
    # The token goes through stdin, not the command line
    curl -sSf -X PUT -H @- -H "X-Checksum-Sha256: ${sha}" -T "$whl" "$url" > /dev/null \
        <<< "Authorization: Bearer ${ARTIFACTORY_TOKEN}"
    echo "Published ${name} (sha256 ${sha})"
fi
