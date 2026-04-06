#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

# Helper to build aiter libs
# Options:
#   --aiter-dir <path>            Path to aiter root (required)
#   --aiter-test-dir <path>       Path to aiter test dir containing compile.py, default: <aiter-dir>/op_tests/cpp/mha
#   --install-dir <path>          Path to install dir for built libs
#   --gpu-archs <list>            GPU arches (required)
#   --ck-tile-bf16 <val>          CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT, default: 3
#   --mode <mha|f4gemm|all>       Build mode: mha (default), f4gemm, or all

set -euo pipefail

AITER_DIR=""
AITER_TEST_DIR=""
GPU_ARCHS_VAL=""
CK_TILE_BF16_DEFAULT="${CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT:-3}"
INSTALL_DIR=""
BUILD_MODE="mha"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aiter-dir)
      AITER_DIR="$2"; shift 2;;
    --aiter-test-dir)
      AITER_TEST_DIR="$2"; shift 2;;
    --install-dir)
      INSTALL_DIR="$2"; shift 2;;
    --gpu-archs)
      GPU_ARCHS_VAL="$2"; shift 2;;
    --ck-tile-bf16)
      CK_TILE_BF16_DEFAULT="$2"; shift 2;;
    --mode)
      BUILD_MODE="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

if [ -z "${AITER_DIR}" -o -z "${GPU_ARCHS_VAL}" ]; then
  echo "[AITER-BUILD] --aiter-dir, --aiter-test-dir, and --gpu-archs are required." >&2
  exit 1
fi

AITER_TEST_DIR="${AITER_TEST_DIR:-${AITER_DIR}/op_tests/cpp/mha}"

# Resolve the TE root (3 levels up from ck_fused_attn/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

build_mha() {
  echo "[AITER-BUILD] Building MHA libs..."
  echo "[AITER-BUILD] AITER_DIR: ${AITER_DIR} TEST_DIR: ${AITER_TEST_DIR} GPU_ARCHS: ${GPU_ARCHS_VAL} CK_TILE_BF16_DEFAULT: ${CK_TILE_BF16_DEFAULT} INSTALL_DIR: ${INSTALL_DIR}"

  rm -rf "${AITER_DIR}/aiter/jit/build"
  CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT="${CK_TILE_BF16_DEFAULT}" \
  GPU_ARCHS="${GPU_ARCHS_VAL}" \
  python3 "${AITER_TEST_DIR}/compile.py"

  if [ -n "${INSTALL_DIR}" ]; then
    mkdir -p "${INSTALL_DIR}"
    cp "${AITER_TEST_DIR}/libmha_fwd.so" "${AITER_TEST_DIR}/libmha_bwd.so" "${INSTALL_DIR}/"
  fi
}

build_f4gemm() {
  # Check if gfx950 is in the arch list — f4gemm only supports gfx950
  if [[ "${GPU_ARCHS_VAL}" != *"gfx950"* ]]; then
    echo "[AITER-BUILD] Skipping f4gemm build: gfx950 not in GPU_ARCHS (${GPU_ARCHS_VAL})"
    return 0
  fi

  echo "[AITER-BUILD] Building f4gemm libs for gfx950..."
  local F4GEMM_COMPILE="${TE_ROOT}/transformer_engine/common/ck_f4gemm/compile_f4gemm.py"

  if [ ! -f "${F4GEMM_COMPILE}" ]; then
    echo "[AITER-BUILD] ERROR: compile_f4gemm.py not found at ${F4GEMM_COMPILE}" >&2
    exit 1
  fi

  GPU_ARCHS="gfx950" \
  AITER_GPU_ARCHS="gfx950" \
  AITER_REBUILD=1 \
  python3 "${F4GEMM_COMPILE}"

  if [ -n "${INSTALL_DIR}" ]; then
    # Install Python extension modules + tuned config CSV
    local F4GEMM_INSTALL="${INSTALL_DIR}/f4gemm"
    mkdir -p "${F4GEMM_INSTALL}"

    # Find and copy the built .so modules from AITER JIT build dir.
    # get_user_jit_dir() returns AITER_JIT_DIR if set, else <aiter_root>/aiter/jit/
    # if writable, else ~/.aiter/jit/. Search all candidate locations.
    local JIT_SEARCH_DIRS="${AITER_JIT_DIR:-} ${AITER_DIR}/aiter/jit ${HOME}/.aiter/jit"
    for module in module_gemm_a4w4_asm module_gemm_a4w4_blockscale; do
      local so_file=""
      for jit_dir in ${JIT_SEARCH_DIRS}; do
        so_file=$(find "${jit_dir}" -maxdepth 1 -name "${module}*.so" -type f 2>/dev/null | head -1)
        [ -n "${so_file}" ] && break
      done
      if [ -n "${so_file}" ]; then
        cp "${so_file}" "${F4GEMM_INSTALL}/"
        echo "[AITER-BUILD] Copied ${so_file} -> ${F4GEMM_INSTALL}/"
      else
        echo "[AITER-BUILD] WARNING: ${module}.so not found in ${JIT_DIR}"
      fi
    done

    # Copy tuned GEMM config CSV
    local TUNED_CSV="${AITER_DIR}/aiter/configs/a4w4_blockscale_tuned_gemm.csv"
    if [ -f "${TUNED_CSV}" ]; then
      cp "${TUNED_CSV}" "${F4GEMM_INSTALL}/"
    fi

    # Install .co blobs into unified aiter/ tree
    local ASM_SRC="${AITER_DIR}/hsa/gfx950/f4gemm"
    if [ -d "${ASM_SRC}" ]; then
      local ASM_DEST="${INSTALL_DIR}/aiter/gfx950/f4gemm"
      mkdir -p "${ASM_DEST}"
      cp "${ASM_SRC}"/*.co "${ASM_DEST}/" 2>/dev/null || true
      cp "${ASM_SRC}"/*.csv "${ASM_DEST}/" 2>/dev/null || true
      echo "[AITER-BUILD] Copied f4gemm .co blobs to ${ASM_DEST}/"
    fi
  fi
}

case "${BUILD_MODE}" in
  mha)
    build_mha
    ;;
  f4gemm)
    build_f4gemm
    ;;
  all)
    build_mha
    build_f4gemm
    ;;
  *)
    echo "[AITER-BUILD] Unknown --mode: ${BUILD_MODE}. Use 'mha', 'f4gemm', or 'all'." >&2
    exit 1
    ;;
esac
