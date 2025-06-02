#!/usr/bin/env bash
set -euo pipefail

# Default configuration (can be overridden by env or CLI flags)
: "${NVTE_FRAMEWORK:=jax}"
: "${NVTE_ROCM_ARCH:=gfx942}"
: "${NVTE_USE_ROCM:=1}"
: "${NVTE_USE_HIPBLASLT:=1}"
: "${NVTE_FUSED_ATTN_AOTRITON:=0}"

usage() {
    cat <<EOF
Usage: $0 [options]
Installs TransformerEngine. Options can be set via env vars or flags.

Options:
   -f FRAMEWORK    Framework (default: $NVTE_FRAMEWORK) e.g., pytorch
   -a ROCM_ARCH    ROCm architecture (default: $NVTE_ROCM_ARCH) e.g., gfx942
   -r <0|1>        Use ROCm (default: $NVTE_USE_ROCM)
   -b <0|1>        Use HIPBLASLT (default: $NVTE_USE_HIPBLASLT)
   -t <0|1>        Use Fused Attention AOTriton (default: $NVTE_FUSED_ATTN_AOTRITON)
   -h              Show this help and exit.
EOF
    exit 1
}

while getopts "f:a:r:b:t:h" opt; do
    case $opt in
        f) NVTE_FRAMEWORK=$OPTARG ;;
        a) NVTE_ROCM_ARCH=$OPTARG ;;
        r) NVTE_USE_ROCM=$OPTARG ;;
        b) NVTE_USE_HIPBLASLT=$OPTARG ;;
        t) NVTE_FUSED_ATTN_AOTRITON=$OPTARG ;;
        h | *) usage ;;
    esac
done

export NVTE_FRAMEWORK NVTE_ROCM_ARCH NVTE_USE_ROCM NVTE_USE_HIPBLASLT NVTE_FUSED_ATTN_AOTRITON

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Patch CMake requirement if present
CMAKE_FILE="$SCRIPT_DIR/3rdparty/hipify_torch/cmake/Hipify.cmake"
if [[ -f "$CMAKE_FILE" ]]; then
    sed -i 's/cmake_minimum_required(VERSION [0-9.]\+)/cmake_minimum_required(VERSION 3.5)/' "$CMAKE_FILE"
fi

# Uninstall any existing installation of transformer_engine
pip uninstall -y transformer_engine || true

# Install
python setup.py install

