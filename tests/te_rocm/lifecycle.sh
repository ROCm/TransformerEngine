#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Wheel lifecycle tests (plugin plan S3.4 / proposal sec 4.4): install / re-install(upgrade path) /
# uninstall / full-stack, in a CLEAN venv, from a wheelhouse. The mixed-index property is asserted
# as T2: the pure-Python `transformer_engine` wheel carries no hard dependency on a core wheel, so
# an install that resolved only the pure wheel (e.g. from an index without the ROCm wheels) must
# fail LOUDLY at import - never import silently without a backend.
#
# usage: lifecycle.sh <wheelhouse-dir> [workdir]
set -u
WHEELHOUSE=$(readlink -f "${1:?usage: lifecycle.sh <wheelhouse-dir> [workdir]}")
WORK=${2:-$(mktemp -d)}
PURE=$(ls "$WHEELHOUSE"/transformer_engine-*-py3-none-any.whl 2>/dev/null | sort | tail -1)
CORE=$(ls "$WHEELHOUSE"/transformer_engine_rocm[0-9]*-*.whl 2>/dev/null | sort | tail -1)
TORCHW=$(ls "$WHEELHOUSE"/transformer_engine_rocm_torch-*.whl 2>/dev/null | sort | tail -1)
[ -n "$PURE" ] || { echo "FATAL: no pure transformer_engine wheel in $WHEELHOUSE"; exit 2; }

PASS=0; FAIL=0; SKIP=0
report() { echo "[$1] $2"; case $1 in PASS) PASS=$((PASS+1));; FAIL) FAIL=$((FAIL+1));; SKIP) SKIP=$((SKIP+1));; esac; }

# All imports MUST run from a neutral cwd: a checkout on cwd shadows site-packages and the
# test would exercise the repo tree, not the installed wheels (same trap as overlay testing).
cd "$WORK"

python3 -m venv "$WORK/venv" || { echo "FATAL: venv creation failed"; exit 2; }
PIP="$WORK/venv/bin/pip"; PY="$WORK/venv/bin/python"
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

# T1 install: pure wheel installs standalone (no index, no deps - deps are runtime-only)
if "$PIP" install -q --no-index --no-deps "$PURE"; then
  "$PIP" show transformer_engine >/dev/null && report PASS "T1 install: pure wheel installs standalone" \
    || report FAIL "T1 install: installed but pip show fails"
else
  report FAIL "T1 install: pip install failed"
fi

# T2 mixed-index / loud failure: import WITHOUT a core wheel must fail with a clear TE error
OUT=$("$PY" -c "import transformer_engine" 2>&1); RC=$?
if [ $RC -eq 0 ]; then
  report FAIL "T2 loud-failure: import SUCCEEDED without a core library (silent-wrong hazard)"
elif echo "$OUT" | grep -qiE "transformer_engine|libtransformer_engine|shared object|TE_ROCM"; then
  report PASS "T2 loud-failure: import refused loudly ($(echo "$OUT" | grep -m1 -oiE '[^\"]*(shared object not found|No module named[^\"]*|TE_ROCM[^\"]*)' | head -c 80))"
else
  report FAIL "T2 loud-failure: import failed but message names nothing recognizable: $(echo "$OUT" | tail -1)"
fi

# T3 re-install / upgrade path: force-reinstall over an existing install leaves one clean copy
if "$PIP" install -q --no-index --no-deps --force-reinstall "$PURE" \
   && [ "$("$PIP" list 2>/dev/null | grep -c '^transformer.engine ')" -le 1 ]; then
  report PASS "T3 upgrade: force-reinstall over existing install is clean"
else
  report FAIL "T3 upgrade: re-install failed or duplicated"
fi

# T4 uninstall: removes the tree completely (manifest copy included - it is in RECORD)
"$PIP" uninstall -q -y transformer_engine
if [ ! -e "$SITE/transformer_engine" ]; then
  report PASS "T4 uninstall: site-packages left clean"
else
  report FAIL "T4 uninstall: residue at $SITE/transformer_engine: $(ls "$SITE/transformer_engine" | head -3 | tr '\n' ' ')"
fi

# T5 full stack: pure + core -> import must resolve the seam and the core ABI. Runs in a
# SECOND venv with --system-site-packages: the core library needs a provisioned ROCm runtime
# (rocm-sdk wheels; in production pulled by install_requires, here supplied by the host env),
# while the TE wheels under test still come from this venv and take precedence.
ROCM_SITE=$(python3 -c "import rocm_sdk, pathlib; print(pathlib.Path(rocm_sdk.__file__).parent.parent)" 2>/dev/null)
if [ -z "$ROCM_SITE" ]; then
  report SKIP "T5 full-stack: invoking python has no rocm_sdk to provision the runtime"
elif [ -n "$CORE" ]; then
  python3 -m venv "$WORK/venv2"
  # Expose ONLY the host env's site (for the rocm-sdk runtime wheels; in production these come
  # from install_requires). A plain --system-site-packages is wrong here: when the invoking
  # python is itself a venv (as in our containers), venv creation chains to the BASE python and
  # would expose the base site, not the one holding rocm_sdk. The venv's own site-packages
  # precedes .pth entries, so the TE wheels under test still win.
  echo "$ROCM_SITE" > "$("$WORK/venv2/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/zz_rocm_provider.pth"
  "$WORK/venv2/bin/pip" install -q --no-index --no-deps "$PURE" "$CORE"
  PY2="$WORK/venv2/bin/python"
  # NVTE_FRAMEWORK=pytorch: the host env also provides jax, and without the scoping TE
  # correctly demands the rocm_jax extension too. A torch-only stack is a supported install.
  IMPORT_SNIPPET="import transformer_engine; import transformer_engine.te_rocm_diagnostics as d; s=d.snapshot(); print(s['core_abi_version'])"
  export NVTE_FRAMEWORK=pytorch

  # T5a framework loud-failure: if torch is importable but the extension wheel is absent, the
  # upstream sanity check must refuse with the exact remedy - never a partial import.
  if "$PY2" -c "import torch" 2>/dev/null; then
    OUT=$("$PY2" -c "$IMPORT_SNIPPET" 2>&1); RC=$?
    if [ -z "$TORCHW" ]; then
      if [ $RC -ne 0 ] && echo "$OUT" | grep -q "transformer_engine_rocm_torch"; then
        report PASS "T5a framework loud-failure: refused, remedy names transformer_engine_rocm_torch"
      elif [ $RC -eq 0 ]; then
        report FAIL "T5a framework loud-failure: import SUCCEEDED without the torch extension wheel"
      else
        report FAIL "T5a framework loud-failure: refused but message unhelpful: $(echo "$OUT" | grep -vE '^\s*(\^+\s*)?$' | tail -1)"
      fi
    fi
  fi

  # T5b full stack: pure + core + torch extension -> import, seam, core ABI all resolve.
  if [ -n "$TORCHW" ]; then
    "$WORK/venv2/bin/pip" install -q --no-index --no-deps "$TORCHW"
    OUT=$("$PY2" -c "$IMPORT_SNIPPET" 2>&1); RC=$?
    if [ $RC -eq 0 ]; then
      report PASS "T5b full-stack: import + diagnostics OK (core_abi=$(echo "$OUT" | tail -1))"
    else
      report FAIL "T5b full-stack: import failed with all wheels present: $(echo "$OUT" | grep -vE '^\s*(\^+\s*)?$' | tail -1)"
    fi
  else
    report SKIP "T5b full-stack: no transformer_engine_rocm_torch wheel in wheelhouse"
  fi
else
  report SKIP "T5 full-stack: no transformer_engine_rocm* wheel in wheelhouse"
fi

echo "lifecycle: $PASS passed, $FAIL failed, $SKIP skipped  (wheelhouse=$WHEELHOUSE)"
[ $FAIL -eq 0 ]
