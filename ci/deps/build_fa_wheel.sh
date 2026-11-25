#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# The Flash Attention wheel the CI test environment installs: the flash-attn release in
# pyproject.toml, compiled for uv.lock's torch, ROCm and Python and every GPU arch in
# pyproject.toml, its version labelled with them, e.g.
# 2.8.3.post2+rocm10.1.0.torch2.13.gfx942.gfx950. CI publishes it to the flash-attn-wheels
# package on ghcr.io, tagged with that version; ci/install_deps.sh fetches it from there, or
# runs this to build it. Wheels go to /opt/fa-wheels, where later installs find them.
#
# Usage: ci/deps/build_fa_wheel.sh [--upload] [--override] | --name | --ref | --check | --fetch
#   (default)   build the wheel unless it is published, and install it into the venv
#   --upload    also publish it; needs $GHCR_TOKEN, a GitHub token that can write packages
#   --override  build and publish it even if it is published already, replacing it
#   --name      print the wheel's file name
#   --ref       print where it is published: the package and its tag
#   --check     exit 0 if the wheel is published, 1 if not, 2 if that can't be told
#   --fetch     download the published wheel to /opt/fa-wheels; exit 1 if there is none
# $GHCR_TOKEN, when set, also lets it read the package while that is private.

set -euo pipefail

DEPS_DIR="$(cd "$(dirname "$0")" && pwd)"
setting() { sed -n "s/^$1 *= *\"\(.*\)\"$/\1/p" "$DEPS_DIR/pyproject.toml"; }
FLASH_ATTN_VERSION=$(setting flash-attn)
FLASH_ATTN_WHEELS=$(setting flash-attn-wheels)
REGISTRY=${FLASH_ATTN_WHEELS%%/*}
REPO=${FLASH_ATTN_WHEELS#*/}
GPU_ARCHS=$(sed -n 's/^\(gfx[0-9a-f]*\) *=.*/\1/p' "$DEPS_DIR/pyproject.toml" | xargs)
WHEELS_DIR=/opt/fa-wheels
# The repository ghcr.io links the package to
SOURCE=https://github.com/ROCm/TransformerEngine
MANIFEST_TYPE=application/vnd.oci.image.manifest.v1+json

UPLOAD=""
OVERRIDE=""
USAGE="Usage: $0 [--upload] [--override] | --name | --ref | --check | --fetch"
case "${1:-}" in
    --name|--ref|--check|--fetch) MODE=$1; shift ;;
    *) MODE=build ;;
esac
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload) UPLOAD=1; shift ;;
        --override) OVERRIDE=1; shift ;;
        *) echo "$USAGE" >&2; exit 1 ;;
    esac
done
if [[ "$MODE" != build && -n "${UPLOAD}${OVERRIDE}" ]]; then
    echo "$USAGE" >&2
    exit 1
fi

# The wheel's name, from uv.lock's torch and Python, and its tag in the
# package: the version, which can't hold "+" there, and the Python
torch_version=$(awk '$0 == "name = \"torch\"" { getline; gsub(/^version = "|"$/, ""); print; exit }' "$DEPS_DIR/uv.lock")
py=$(sed -n 's/^requires-python = "==\([0-9]*\)\.\([0-9]*\).*/cp\1\2/p' "$DEPS_DIR/uv.lock")
version="${FLASH_ATTN_VERSION}+rocm${torch_version#*+rocm}.torch$(cut -d. -f1,2 <<< "${torch_version%%+*}").${GPU_ARCHS// /.}"
name="flash_attn-${version}-${py}-${py}-linux_x86_64.whl"
tag="${version/+/-}-${py}"
whl="${WHEELS_DIR}/${name}"

# --- The registry API ---
# The registry's token for $1 (pull or pull,push) on the package, if it wants one:
# anonymous, or with $GHCR_TOKEN when that is set. Fails when access is denied, which is
# also how a package that doesn't exist, or is private, looks.
TOKEN=""
login() {
    local challenge realm service
    TOKEN=""
    challenge=$(curl -sS -D - -o /dev/null "https://${REGISTRY}/v2/" | tr -d '\r' \
        | sed -n 's/^[Ww][Ww][Ww]-[Aa]uthenticate: *Bearer //p') || return 0
    [[ -n "$challenge" ]] || return 0
    realm=$(sed -n 's/.*realm="\([^"]*\)".*/\1/p' <<< "$challenge")
    service=$(sed -n 's/.*service="\([^"]*\)".*/\1/p' <<< "$challenge")
    # Credentials and tokens go through stdin, not the command line
    TOKEN=$(curl -sS -K - "${realm}?service=${service}&scope=repository:${REPO}:$1" \
        <<< "${GHCR_TOKEN:+user = \"${GITHUB_ACTOR:-token}:${GHCR_TOKEN}\"}" \
        | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
    [[ -n "$TOKEN" ]]
}
api() { curl -K - "$@" <<< "${TOKEN:+header = \"Authorization: Bearer ${TOKEN}\"}"; }

# Exit status 0 if the wheel is published, 1 if not, 2 if that can't be told
published() {
    local code
    login pull || return 1
    code=$(api -s -o /dev/null -w '%{http_code}' -I -H "Accept: ${MANIFEST_TYPE}" \
        "https://${REGISTRY}/v2/${REPO}/manifests/${tag}")
    case "$code" in
        200) return 0 ;;
        404) return 1 ;;
        *) echo "Can't tell whether ${FLASH_ATTN_WHEELS}:${tag} is published: HTTP ${code}" >&2; return 2 ;;
    esac
}

fetch() {
    local rc=0 digest
    published || rc=$?
    [[ $rc == 0 ]] || return $rc
    digest=$(api -sSf -H "Accept: ${MANIFEST_TYPE}" "https://${REGISTRY}/v2/${REPO}/manifests/${tag}" \
        | tr -d ' \n\t' | sed -n 's/.*"layers":\[{[^}]*"digest":"\(sha256:[0-9a-f]*\)".*/\1/p')
    if [[ -z "$digest" ]]; then
        echo "${FLASH_ATTN_WHEELS}:${tag} has no wheel" >&2
        return 2
    fi
    mkdir -p "$WHEELS_DIR"
    api -sSfL -o "${whl}.part" "https://${REGISTRY}/v2/${REPO}/blobs/${digest}"
    echo "${digest#sha256:}  ${whl}.part" | sha256sum -c --quiet
    mv "${whl}.part" "$whl"
    echo "Fetched ${name} from ${FLASH_ATTN_WHEELS}:${tag}"
}

push_blob() {  # <file> <sha256>
    local location
    if [[ "$(api -s -o /dev/null -w '%{http_code}' -I "https://${REGISTRY}/v2/${REPO}/blobs/sha256:$2")" == 200 ]]; then
        return 0
    fi
    location=$(api -sSf -X POST -D - -o /dev/null "https://${REGISTRY}/v2/${REPO}/blobs/uploads/" \
        | tr -d '\r' | sed -n 's/^[Ll]ocation: *//p')
    [[ "$location" == http* ]] || location="https://${REGISTRY}${location}"
    [[ "$location" == *\?* ]] && location+="&" || location+="?"
    api -sSf -T "$1" -H "Content-Type: application/octet-stream" "${location}digest=sha256:$2" > /dev/null
}

# The wheel as an OCI artifact: one layer, under the tag
upload() {
    local sha size empty manifest
    if ! login pull,push; then
        echo "Can't publish to ${FLASH_ATTN_WHEELS} with the token in GHCR_TOKEN" >&2
        return 1
    fi
    sha=$(sha256sum "$whl" | cut -d' ' -f1)
    size=$(stat -c %s "$whl")
    empty=$(mktemp)
    printf '{}' > "$empty"
    push_blob "$whl" "$sha"
    push_blob "$empty" 44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a
    # From a file: api() takes stdin for the token
    manifest=$(mktemp)
    cat > "$manifest" <<EOF
{"schemaVersion":2,"mediaType":"${MANIFEST_TYPE}","artifactType":"application/vnd.unknown.artifact.v1","config":{"mediaType":"application/vnd.oci.empty.v1+json","digest":"sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a","size":2},"layers":[{"mediaType":"application/vnd.oci.image.layer.v1.tar","digest":"sha256:${sha}","size":${size},"annotations":{"org.opencontainers.image.title":"${name}"}}],"annotations":{"org.opencontainers.image.source":"${SOURCE}"}}
EOF
    api -sSf -X PUT -H "Content-Type: ${MANIFEST_TYPE}" --data-binary @"$manifest" \
        "https://${REGISTRY}/v2/${REPO}/manifests/${tag}" > /dev/null
    rm -f "$empty" "$manifest"
    echo "Published the wheel at ${FLASH_ATTN_WHEELS}:${tag}"
}

rc=0
case "$MODE" in
    --name) echo "$name"; exit 0 ;;
    --ref) echo "${FLASH_ATTN_WHEELS}:${tag}"; exit 0 ;;
    --check) published || rc=$?; exit $rc ;;
    --fetch) fetch || rc=$?; exit $rc ;;
esac

# --- Build ---
if [[ -n "$UPLOAD" && -z "${GHCR_TOKEN:-}" ]]; then
    echo "--upload needs GHCR_TOKEN, a GitHub token that can write packages to ${FLASH_ATTN_WHEELS}" >&2
    exit 1
fi
if [[ -z "$OVERRIDE" ]]; then
    published || rc=$?
    if [[ $rc == 0 ]]; then
        echo "${FLASH_ATTN_WHEELS}:${tag} is published already"
        exit 0
    elif [[ $rc == 2 ]]; then
        exit 2
    fi
fi

# The active venv, else the one install_deps.sh made, if it has torch
VENV=""
for v in "${VIRTUAL_ENV:-}" /opt/venv /opt/venv-torch; do
    if [[ -n "$v" && -x "$v/bin/python" ]] && "$v/bin/python" -c "import torch" 2> /dev/null; then
        VENV=$v
        break
    fi
done
if [[ -z "$VENV" ]]; then
    echo "No venv with torch: run ci/install_deps.sh first" >&2
    exit 1
fi
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
if [[ "$(python -c 'import torch; print(torch.__version__)')" != "$torch_version" ]]; then
    echo "${VENV} doesn't have uv.lock's torch ${torch_version}: run ci/install_deps.sh" >&2
    exit 1
fi
ROCM_PATH="$(rocm-sdk path --root)"
export ROCM_PATH ROCM_HOME="$ROCM_PATH"

if [[ -f "$whl" && -z "$OVERRIDE" ]]; then
    echo "Using ${whl}, built earlier"
else
    src=$(mktemp -d)
    git -C "$src" init -q
    git -C "$src" fetch -q --depth 1 https://github.com/Dao-AILab/flash-attention.git "v${FLASH_ATTN_VERSION}"
    git -C "$src" checkout -q FETCH_HEAD
    # ROCm needs only the CK submodule. Without .git, setup.py skips its own submodule
    # update, which would also clone cutlass.
    git -C "$src" submodule update -q --init csrc/composable_kernel
    rm -rf "$src/.git"
    echo "Building flash-attn ${version}"
    mkdir -p "$WHEELS_DIR"
    rm -f "$whl"
    ( cd "$src"
      FLASH_ATTENTION_FORCE_BUILD=TRUE GPU_ARCHS="${GPU_ARCHS// /;}" FLASH_ATTN_LOCAL_VERSION="${version#*+}" \
          python setup.py bdist_wheel -d "$WHEELS_DIR" )
    rm -rf "$src"
    if [[ ! -f "$whl" ]]; then
        echo "Expected ${whl}; the build made: $(ls "$WHEELS_DIR")" >&2
        exit 1
    fi
fi
python -m pip install --quiet --no-deps --force-reinstall "$whl"
echo "Installed ${name} into ${VENV}"

if [[ -n "$UPLOAD" ]]; then
    # Without --override, a wheel another run published meanwhile stays
    if [[ -z "$OVERRIDE" ]] && published; then
        echo "${FLASH_ATTN_WHEELS}:${tag} was published meanwhile; keeping that one"
        exit 0
    fi
    upload
fi
