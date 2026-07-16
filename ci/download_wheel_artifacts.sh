#!/usr/bin/env bash
# Download the te-rocm-wheels artifact, verify expected files, and retry twice if needed.
set -euo pipefail

artifact_dir="${1:-dist}"
artifact_name="${2:-te-rocm-wheels}"
repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
artifact_id="${ARTIFACT_ID:-${TE_WHEEL_ARTIFACT_ID:-}}"
expected_digest="${ARTIFACT_DIGEST:-${TE_WHEEL_ARTIFACT_DIGEST:-}}"
api_url="${GITHUB_API_URL:-https://api.github.com}"

required_patterns=(
  'transformer_engine_rocm[0-9]*.whl'
  'transformer_engine_rocm_torch*.tar.gz'
  'transformer_engine_rocm_jax*.tar.gz'
)

if [[ -z "${GITHUB_TOKEN:-}" && -z "${GH_TOKEN:-}" ]]; then
  echo "::error::GITHUB_TOKEN or GH_TOKEN is required to download GitHub Actions artifacts"
  exit 1
fi

token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if [[ -z "${artifact_id}" ]]; then
  echo "::error::ARTIFACT_ID/TE_WHEEL_ARTIFACT_ID is required to download ${artifact_name}"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "::error::curl is required to download artifacts"
  exit 1
fi

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "::error::sha256sum is required to verify artifact digests"
  exit 1
fi

extract_zip() {
  local zip_file="$1"
  local dest_dir="$2"

  if command -v unzip >/dev/null 2>&1; then
    unzip -q "${zip_file}" -d "${dest_dir}"
  elif command -v busybox >/dev/null 2>&1 && busybox unzip --help >/dev/null 2>&1; then
    busybox unzip -q "${zip_file}" -d "${dest_dir}"
  elif command -v bsdtar >/dev/null 2>&1; then
    (cd "${dest_dir}" && bsdtar -xf "${zip_file}")
  else
    echo "::error::unzip, busybox unzip, or bsdtar is required to extract artifacts"
    return 1
  fi
}

print_manifest_group() {
  echo "::group::Build artifact manifest"
  echo "=== artifact download context ==="
  echo "repository: ${repo}"
  echo "artifact_name: ${artifact_name}"
  echo "artifact_id: ${artifact_id}"
  echo "artifact_dir: ${artifact_dir}"
  echo "expected_digest: ${expected_digest:-unknown}"
  echo ""
  echo "=== downloaded files ==="
  if [[ -d "${artifact_dir}" ]]; then
    find "${artifact_dir}" -maxdepth 2 -type f -printf '%p\t%s bytes\n' | sort
  else
    echo "${artifact_dir} directory is missing"
  fi
  echo "::endgroup::"
}

verify_artifacts() {
  local missing=0
  print_manifest_group
  for pattern in "${required_patterns[@]}"; do
    if ! find "${artifact_dir}" -maxdepth 2 -type f -name "${pattern}" | grep -q .; then
      echo "::error::Missing required build artifact matching ${pattern} under ${artifact_dir}"
      missing=1
    fi
  done
  return "${missing}"
}

download_once() {
  local attempt="$1"
  local tmp_dir artifact_zip curl_config expected_sha actual_sha download_url
  tmp_dir="$(mktemp -d)"
  artifact_zip="${tmp_dir}/artifact.zip"
  curl_config="${tmp_dir}/curl.conf"
  download_url="${api_url}/repos/${repo}/actions/artifacts/${artifact_id}/zip"

  cat > "${curl_config}" <<EOF
header = "Accept: application/vnd.github+json"
header = "Authorization: Bearer ${token}"
header = "X-GitHub-Api-Version: 2022-11-28"
EOF

  echo "Preparing to download artifact ${artifact_name} (ID: ${artifact_id}, Expected Digest: ${expected_digest:-unknown})"
  echo "Downloading artifact attempt ${attempt} to ${artifact_zip}"

  if ! curl --fail --silent --show-error --location \
    --connect-timeout 30 \
    --max-time 1800 \
    --config "${curl_config}" \
    --output "${artifact_zip}" \
    "${download_url}"; then
    rm -rf "${tmp_dir}"
    return 1
  fi

  if [[ -n "${expected_digest}" ]]; then
    expected_sha="${expected_digest#sha256:}"
    actual_sha="$(sha256sum "${artifact_zip}" | awk '{print $1}')"
    echo "SHA256 digest of downloaded artifact is ${actual_sha}"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
      echo "::error::Artifact digest mismatch: expected ${expected_sha}, got ${actual_sha}"
      rm -rf "${tmp_dir}"
      return 1
    fi
  else
    echo "::warning::No expected artifact digest was provided; validating file contents only."
  fi

  rm -rf "${artifact_dir}"
  mkdir -p "${artifact_dir}"
  if ! extract_zip "${artifact_zip}" "${artifact_dir}"; then
    rm -rf "${tmp_dir}"
    return 1
  fi

  rm -rf "${tmp_dir}"
  return 0
}

max_attempts=3
for attempt in $(seq 1 "${max_attempts}"); do
  if download_once "${attempt}" && verify_artifacts; then
    echo "Build artifact download and verification succeeded on attempt ${attempt}."
    exit 0
  fi
  if [[ "${attempt}" -lt "${max_attempts}" ]]; then
    echo "::warning::Build artifact download/verification failed; retrying ($((max_attempts - attempt)) retries left)."
  fi
done

echo "::error::Build artifact download/verification failed after 2 retries."
exit 1
