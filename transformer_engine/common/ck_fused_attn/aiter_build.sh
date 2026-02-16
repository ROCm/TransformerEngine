#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

# Helper to build aiter libs
# Options:
#   --aiter-dir <path>            Path to aiter root (required)
#   --aiter-test-dir <path>       Path to aiter test dir containing compile.py (required)
#   --gpu-archs <list>            GPU arches (required)
#   --ck-tile-bf16 <val>          CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT, default: 3
set -euo pipefail

AITER_DIR=""
AITER_TEST_DIR=""
GPU_ARCHS_VAL=""
CK_TILE_BF16_DEFAULT="${CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT:-3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aiter-dir)
      AITER_DIR="$2"; shift 2;;
    --aiter-test-dir)
      AITER_TEST_DIR="$2"; shift 2;;
    --gpu-archs)
      GPU_ARCHS_VAL="$2"; shift 2;;
    --ck-tile-bf16)
      CK_TILE_BF16_DEFAULT="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

if [[ -z "${AITER_DIR}" || -z "${AITER_TEST_DIR}" || -z "${GPU_ARCHS_VAL}" ]]; then
  echo "[AITER-PREBUILT] --aiter-dir, --aiter-test-dir, and --gpu-archs are required." >&2
  exit 1
fi

rm -rf "${AITER_DIR}/aiter/jit/build"
AITER_LOG_MORE=0 \
CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT="${CK_TILE_BF16_DEFAULT}" \
GPU_ARCHS="${GPU_ARCHS_VAL}" \
python3 "${AITER_TEST_DIR}/compile.py"

