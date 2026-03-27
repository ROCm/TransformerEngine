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

# Generate static archives from the built object files only if NVTE_AITER_STATIC_LINK=1
if [[ "${NVTE_AITER_STATIC_LINK:-1}" -ne 1 ]]; then
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

# Create a single unified static archive from both forward and backward object files
out_archive="${AITER_TEST_DIR}/libmha.a"
obj_list=$(mktemp)
rm -f "${obj_list}"

for lib in fwd bwd; do
  src_obj_dir="${AITER_DIR}/aiter/jit/build/libmha_${lib}/build"
  if [[ ! -d "${src_obj_dir}" ]]; then
    echo "[AITER-BUILD] Missing object directory: ${src_obj_dir}" >&2
    rm -f "${obj_list}"
    exit 1
  fi
  find "${src_obj_dir}" -type f -name '*.o' >> "${obj_list}"
done

total_objs=$(wc -l < "${obj_list}")
if [[ "${total_objs}" -eq 0 ]]; then
  echo "[AITER-BUILD] No object files found for fwd/bwd" >&2
  rm -f "${obj_list}"
  exit 1
fi

rm -f "${out_archive}"
# Use a file list to avoid ARG_MAX limits with thousands of object files
"${AR_BIN}" qc "${out_archive}" @"${obj_list}"

if [[ -n "${RANLIB_BIN}" ]]; then
  "${RANLIB_BIN}" "${out_archive}"
fi

echo "[AITER-BUILD] Created static archive: ${out_archive} (${total_objs} objects)"
rm -f "${obj_list}"

if [ -n "${INSTALL_DIR}" ]; then
  mkdir -p "${INSTALL_DIR}"
  cp "${out_archive}" "${INSTALL_DIR}/"
fi
