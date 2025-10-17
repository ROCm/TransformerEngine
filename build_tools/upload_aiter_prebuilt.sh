#!/bin/sh
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information

if [ -z "${NVTE_AITER_PREBUILT_BASE_URL}" ]; then
  echo "[AITER-PREBUILT] [Error] Set NVTE_AITER_PREBUILT_BASE_URL to specify upload destination." >&2
  exit 1
fi
if [ -z "${NVTE_ARTIFACTORY_USER}" ] || [ -z "${NVTE_ARTIFACTORY_PASSWORD}" ]; then
  echo "[AITER-PREBUILT] [Error] To fetch .so, set NVTE_ARTIFACTORY_USER & NVTE_ARTIFACTORY_PASSWORD" >&2
  exit 1
fi
CURL_AUTH="-u ${NVTE_ARTIFACTORY_USER}:${NVTE_ARTIFACTORY_PASSWORD}"

# Derive keys for cache/artifact naming
TE_ROOT="$(git rev-parse --show-toplevel)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ROCM_VER="$(cut -d . -f1,2,3 < "${ROCM_PATH}/.info/version")"
AITER_SHA="$(git -C "${TE_ROOT}/3rdparty/aiter" rev-parse HEAD)"

# Artifact key + remote path
KEY="rocm-${ROCM_VER}_aiter-${AITER_SHA}"
REMOTE_PATH="aiter-prebuilts/${KEY}.tar.gz"
UPLOAD_URL="${NVTE_PREBUILT_BASE_URL%/}/${REMOTE_PATH}"

# Local cache
CACHE_ROOT="/tmp/aiter-prebuilts"
KEY_DIR="${CACHE_ROOT}/${KEY}"
TAR_PATH="${CACHE_ROOT}/${KEY}.tar.gz"

[ -d "${KEY_DIR}" ] && rm -rf -- "${KEY_DIR}"
[ -f "${TAR_PATH}" ] && rm -f -- "${TAR_PATH}"
mkdir -p "${KEY_DIR}"

# --- Stage built artifacts ---
FWD="$(echo "${TE_ROOT}"/build/lib.*-cpython-*/transformer_engine/lib/libmha_fwd.so | head -n 1)"
BWD="$(echo "${TE_ROOT}"/build/lib.*-cpython-*/transformer_engine/lib/libmha_bwd.so | head -n 1)"

[ -f "${FWD}" ] || { echo "[AITER-PREBUILT][ERROR] Missing libmha_fwd.so" >&2; exit 2; }
[ -f "${BWD}" ] || { echo "[AITER-PREBUILT][ERROR] Missing libmha_bwd.so" >&2; exit 2; }

cp -f "${FWD}" "${KEY_DIR}/"
cp -f "${BWD}" "${KEY_DIR}/"

# Zip and upload to artifactory
tar -czf "${TAR_PATH}" -C "${CACHE_ROOT}" "${KEY}"
echo "[AITER-PREBUILT] Started uploading ${TAR_PATH} ..."
curl -fSL ${CURL_AUTH} -T "${TAR_PATH}" "${ARTIFACT_URL}" -w '\n'
echo "[AITER-PREBUILT] Upload complete."
rm -rf -- "${CACHE_ROOT}";