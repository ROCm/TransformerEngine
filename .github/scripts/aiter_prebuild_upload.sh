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
DIR="${ROOT_DIR}/ci"
. "${ROOT_DIR}/ci/_utils.sh"
ROCM_PATH="$(resolve_rocm_path)"
export ROCM_PATH
ROCM_VER=`head -n1 "${ROCM_PATH}/.info/version" | cut -d. -f1`

QOLA_DIR="${ROOT_DIR}/3rdparty/QoLA"
AITER_DIR="${QOLA_DIR}/3rdparty/aiter"
QOLA_MANIFEST="${ROOT_DIR}/transformer_engine/common/ck_fused_attn/qola_manifest.toml"
GIT_CONFIG_GLOBAL="$(mktemp /tmp/gitconfig.XXXXXX)"
trap 'rm -f "${GIT_CONFIG_GLOBAL}"' EXIT
git config --file "${GIT_CONFIG_GLOBAL}" --add safe.directory "${AITER_DIR}"
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
    echo "[aiter-upload] Remote prebuilt already exists at ${REMOTE_URL}; nothing to do."
    exit 0
  fi
fi

# Optional build stage
if [[ "${1:-}" == "--build" ]]; then
  shift
  GPU_ARCHS=("gfx942" "gfx950")
  echo "[AITER-PREBUILT] Building aiter libs via QoLA for ${GPU_ARCHS[*]} ..."
  QOLA_BUILD_DIR="${QOLA_DIR}/build"
  arch_args=()
  for a in "${GPU_ARCHS[@]}"; do arch_args+=(--arch "${a}"); done
  PYTHONPATH="${QOLA_DIR}:${PYTHONPATH:-}" \
    python3 -m qola.cli build \
      --manifest "${QOLA_MANIFEST}" \
      --aiter-root "${AITER_DIR}" \
      --output-dir "${QOLA_BUILD_DIR}" \
      "${arch_args[@]}"

  # Stage QoLA outputs into the cache layout expected by aiter_prebuilt.cmake.
  mkdir -p "${EXTRACT_DIR}/lib" "${EXTRACT_DIR}/include"
  cp "${QOLA_BUILD_DIR}/lib/"*.so "${EXTRACT_DIR}/lib/"
  cp "${QOLA_BUILD_DIR}/include/"*.h "${EXTRACT_DIR}/include/"
fi

# Ensure built libs exist (matches aiter_prebuilt.cmake::is_aiter_cache_valid).
for lib in te_libmha_fwd.so te_libmha_bwd.so; do
  if [[ ! -f "${EXTRACT_DIR}/lib/${lib}" ]]; then
    echo "[AITER-PREBUILT] Missing ${lib} in ${EXTRACT_DIR}/lib" >&2
    exit 1
  fi
done

# qola_config.h is the namespace-baked header; without it consumer compiles fail.
if [[ ! -f "${EXTRACT_DIR}/include/qola_config.h" ]]; then
  echo "[AITER-PREBUILT] Missing qola_config.h in ${EXTRACT_DIR}/include" >&2
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

