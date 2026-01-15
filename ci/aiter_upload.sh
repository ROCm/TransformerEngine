#!/usr/bin/env bash
set -euo pipefail

# Inputs for upload (optional):
#   NVTE_AITER_PREBUILT_BASE_URL - base URL for prebuilts
#   NVTE_AITER_PREBUILT_UPLOAD_TOKEN - bearer token for Artifactory
# Optional flag:
#   --build : build aiter libs before packaging/uploading; default is package-only.

# Derive ROCm version and aiter commit -> cache key
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ROCM_VER="$(head -n1 "${ROCM_PATH}/.info/version" | sed -n 's/^\([0-9]\+\.[0-9]\+\).*/\1/p')"

AITER_DIR="${ROOT_DIR}/3rdparty/aiter"
git -C "${AITER_DIR}" config --global --add safe.directory "${AITER_DIR}" >/dev/null
AITER_SHA="$(git -C "${AITER_DIR}" rev-parse HEAD)"

KEY="rocm-${ROCM_VER}_aiter-${AITER_SHA}"
CACHE_ROOT="${ROOT_DIR}/build/aiter-prebuilts"
EXTRACT_DIR="${CACHE_ROOT}/${KEY}"
OUTPUT_TGZ="/tmp/${KEY}.tar.gz"

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

if [[ ${HAS_UPLOAD} -eq 1 ]]; then
  echo "[AITER-PREBUILT] Uploading..."
  COLUMNS=50 curl --progress-bar --fail -X PUT \
    -H "Authorization: Bearer ${NVTE_AITER_PREBUILT_UPLOAD_TOKEN}" \
    -T "${OUTPUT_TGZ}" \
    "${REMOTE_URL}" \
    -o /dev/null
  echo "[AITER-PREBUILT] Uploaded tgz to ${REMOTE_URL}"
fi

echo "[AITER-PREBUILT] Artifacts:"
echo "  tgz: ${OUTPUT_TGZ}"
if [[ ${HAS_UPLOAD} -eq 0 ]]; then
  echo "[AITER-PREBUILT] To upload, set NVTE_AITER_PREBUILT_BASE_URL and NVTE_AITER_PREBUILT_UPLOAD_TOKEN."
fi

