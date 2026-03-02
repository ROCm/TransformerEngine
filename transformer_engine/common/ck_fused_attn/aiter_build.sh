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
  echo "[AITER-BUILD] --aiter-dir, --aiter-test-dir, and --gpu-archs are required." >&2
  exit 1
fi

rm -rf "${AITER_DIR}/aiter/jit/build"
AITER_LOG_MORE=1 \
CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT="${CK_TILE_BF16_DEFAULT}" \
GPU_ARCHS="${GPU_ARCHS_VAL}" \
python3 "${AITER_TEST_DIR}/compile.py"

# Generate static archives from the built object files only if NVTE_AITER_STATIC_LINK=1
if [[ "${NVTE_AITER_STATIC_LINK:-0}" -ne 1 ]]; then
  exit 0
fi

# Check for ar and ranlib
AR_BIN="${AR:-$(command -v ar || true)}"
RANLIB_BIN="${RANLIB:-$(command -v ranlib || true)}"
if [[ -z "${AR_BIN}" ]]; then
  echo "[AITER-BUILD] Could not find ar for static archive generation." >&2
  exit 1
fi
if [[ -z "${RANLIB_BIN}" ]]; then
  echo "[AITER-BUILD] Could not find ranlib for static archive generation." >&2
  exit 1
fi

# Create static archives for both forward and backward passes
for lib in fwd bwd; do
  src_obj_dir="${AITER_DIR}/aiter/jit/build/libmha_${lib}/build"
  out_archive="${AITER_TEST_DIR}/libmha_${lib}.a"

  if [[ ! -d "${src_obj_dir}" ]]; then
    echo "[AITER-BUILD] Missing object directory: ${src_obj_dir}" >&2
    exit 1
  fi

  mapfile -d '' obj_files < <(find "${src_obj_dir}" -type f -name '*.o' -print0)
  if [[ ${#obj_files[@]} -eq 0 ]]; then
    echo "[AITER-BUILD] No object files found under ${src_obj_dir}" >&2
    exit 1
  fi

  rm -f "${out_archive}"
  for obj in "${obj_files[@]}"; do
    "${AR_BIN}" q "${out_archive}" "${obj}"
  done
  if [[ -n "${RANLIB_BIN}" ]]; then
    "${RANLIB_BIN}" "${out_archive}"
  fi

  echo "[AITER-BUILD] Created static archive: ${out_archive} (${#obj_files[@]} objects)"
done

