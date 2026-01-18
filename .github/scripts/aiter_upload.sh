#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
set -euo pipefail

# Inputs for upload (optional):
#   NVTE_AITER_PREBUILT_BASE_URL - base URL for prebuilts
#   NVTE_AITER_PREBUILT_UPLOAD_TOKEN - bearer token for Artifactory
# Optional flag:
#   --build : build aiter libs before packaging/uploading; default is package-only.

# Derive ROCm version and aiter commit -> cache key
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ROCM_VER="$(head -n1 "${ROCM_PATH}/.info/version" | sed -n 's/^\([0-9]\+\.[0-9]\+\).*/\1/p')"

AITER_DIR="${ROOT_DIR}/3rdparty/aiter"
GIT_CONFIG_GLOBAL="$(mktemp /tmp/gitconfig.XXXXXX)"
trap 'rm -f "${GIT_CONFIG_GLOBAL}"' EXIT
git config --global --add safe.directory "${AITER_DIR}" --file "${GIT_CONFIG_GLOBAL}" >/dev/null 2>&1 || true
AITER_SHA="$(GIT_CONFIG_GLOBAL=${GIT_CONFIG_GLOBAL} git -C "${AITER_DIR}" rev-parse HEAD)"

KEY="rocm-${ROCM_VER}_aiter-${AITER_SHA}"
CACHE_ROOT="${ROOT_DIR}/build/aiter-prebuilts"
EXTRACT_DIR="${CACHE_ROOT}/${KEY}"
OUTPUT_TGZ="/tmp/${KEY}.tar.gz"
OUTPUT_SHA="/tmp/${KEY}.tar.gz.sha256"

HAS_UPLOAD=0
if [[ -n "${NVTE_AITER_PREBUILT_BASE_URL:-}" && -n "${NVTE_AITER_PREBUILT_UPLOAD_TOKEN:-}" ]]; then
  HAS_UPLOAD=1
fi

# Skip early when remote prebuilt already exists
REMOTE_URL=""
if [[ ${HAS_UPLOAD} -eq 1 ]]; then
  REMOTE_URL="${NVTE_AITER_PREBUILT_BASE_URL}/${KEY}.tar.gz"
  if curl -sIf "${REMOTE_URL}" >/dev/null; then
    echo "[aiter-upload] Remote prebuilt already present at ${REMOTE_URL}; nothing to do."
    exit 0
  fi
fi

# Optional build stage (uses GPU_ARCHS if set, else gfx942;gfx950)
if [[ "${1:-}" == "--build" ]]; then
  shift
  ARCHS="${GPU_ARCHS:-gfx942;gfx950}"
  echo "[AITER-PREBUILT] Building aiter libs for ${ARCHS} ..."
  rm -rf "${AITER_DIR}/aiter/jit/build"
  AITER_LOG_MORE=1 \
  CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3 \
  GPU_ARCHS="${ARCHS}" \
  python3 "${ROOT_DIR}/3rdparty/aiter/op_tests/cpp/mha/compile.py"
  mkdir -p "${EXTRACT_DIR}"
  cp "${ROOT_DIR}/3rdparty/aiter/op_tests/cpp/mha/libmha_fwd.so" "${EXTRACT_DIR}/"
  cp "${ROOT_DIR}/3rdparty/aiter/op_tests/cpp/mha/libmha_bwd.so" "${EXTRACT_DIR}/"
fi

# Ensure built libs exist
if [[ ! -f "${EXTRACT_DIR}/libmha_fwd.so" ]]; then
  echo "[AITER-PREBUILT] Missing libmha_fwd.so in ${EXTRACT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${EXTRACT_DIR}/libmha_bwd.so" ]]; then
  echo "[AITER-PREBUILT] Missing libmha_bwd.so in ${EXTRACT_DIR}" >&2
  exit 1
fi

echo "[AITER-PREBUILT] Packaging ${EXTRACT_DIR} -> ${OUTPUT_TGZ}"
tar -C "${CACHE_ROOT}" -czf "${OUTPUT_TGZ}" "${KEY}"
sha256sum "${OUTPUT_TGZ}" | awk '{print $1}' > "${OUTPUT_SHA}"

if [[ ${HAS_UPLOAD} -eq 1 ]]; then
  echo "[AITER-PREBUILT] Uploading..."
  COLUMNS=50 curl --progress-bar --fail -X PUT \
    -H "Authorization: Bearer ${NVTE_AITER_PREBUILT_UPLOAD_TOKEN}" \
    -T "${OUTPUT_TGZ}" \
    "${REMOTE_URL}" \
    -o /dev/null
  echo "[AITER-PREBUILT] Uploaded tgz to ${REMOTE_URL}"

  # Verify remote SHA256 matches local
  REMOTE_SHA_TMP="$(mktemp /tmp/aiter_remote_sha.XXXXXX)"
  trap 'rm -f "${REMOTE_SHA_TMP}"' EXIT
  if curl -fsSL "${REMOTE_URL}.sha256" -o "${REMOTE_SHA_TMP}"; then
    REMOTE_SHA_VAL="$(awk '{print $1}' "${REMOTE_SHA_TMP}")"
    LOCAL_SHA_VAL="$(cat "${OUTPUT_SHA}")"
    if [[ "${REMOTE_SHA_VAL}" != "${LOCAL_SHA_VAL}" ]]; then
      echo "[AITER-PREBUILT] Remote SHA256 mismatch!"
      exit 1
    else
      echo "[AITER-PREBUILT] Remote SHA256 verified."
    fi
  else
    echo "[AITER-PREBUILT] Warning: failed to download remote .sha256 for verification." >&2
  fi
fi

echo "[AITER-PREBUILT] Artifacts:"
echo "  tgz: ${OUTPUT_TGZ}"
echo "  sha: ${OUTPUT_SHA}"
if [[ ${HAS_UPLOAD} -eq 0 ]]; then
  echo "[AITER-PREBUILT] To upload, set NVTE_AITER_PREBUILT_BASE_URL and NVTE_AITER_PREBUILT_UPLOAD_TOKEN."
fi

