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

set -euo pipefail

AITER_DIR=""
AITER_TEST_DIR=""
GPU_ARCHS_VAL=""
CK_TILE_BF16_DEFAULT="${CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT:-3}"
INSTALL_DIR=""

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
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

if [ -z "${AITER_DIR}" -o -z "${GPU_ARCHS_VAL}" ]; then
  echo "[AITER-BUILD] --aiter-dir, --aiter-test-dir, and --gpu-archs are required." >&2
  exit 1
fi

AITER_TEST_DIR="${AITER_TEST_DIR:-${AITER_DIR}/op_tests/cpp/mha}"

echo "[AITER-BUILD] AITER_DIR: ${AITER_DIR} TEST_DIR: ${AITER_TEST_DIR} GPU_ARCHS: ${GPU_ARCHS_VAL} CK_TILE_BF16_DEFAULT: ${CK_TILE_BF16_DEFAULT} INSTALL_DIR: ${INSTALL_DIR}"

rm -rf "${AITER_DIR}/aiter/jit/build"
CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT="${CK_TILE_BF16_DEFAULT}" \
GPU_ARCHS="${GPU_ARCHS_VAL}" \
python3 "${AITER_TEST_DIR}/compile.py"

if [ -n "${INSTALL_DIR}" ]; then
  mkdir -p "${INSTALL_DIR}"
  cp "${AITER_TEST_DIR}/libmha_fwd.so" "${AITER_TEST_DIR}/libmha_bwd.so" "${INSTALL_DIR}/"
fi
