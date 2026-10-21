#!/usr/bin/bash
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Run TE examples on a single GPU.
# HIP_VISIBLE_DEVICES must be set by the caller (run_parallel_sgpu.sh).

set -e

# Autodetect repo root from this script's location (.github/scripts/ -> ../..)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"

JAX_CONSTRAINTS=/tmp/jax-constraints.txt
pip freeze | grep -iE '^(jax|jaxlib|jax[_-]rocm|jax[_-]plugins)[=@]' > "$JAX_CONSTRAINTS" || true

# Examples run as plain scripts (not pytest), so they emit no JUnit XML and their
# failures never reach the job summary that ci/junit_report.py builds. When the
# parallel runner provides JUNITXML_PREFIX (e.g. test-results/examples/), wrap each
# invocation to drop a one-testcase JUnit XML there so example failures surface in
# the summary alongside the pytest suites. Reporting is best-effort: it never masks
# or alters the example's own exit status.
emit_junit() {
    # $1=xml-path  $2=testcase-name  $3=exit-code  $4=duration-seconds  $5=command
    python3 - "$@" <<'PY'
import sys, xml.sax.saxutils as su
path, name, rc, dur, cmd = sys.argv[1:6]
rc, dur = int(rc), float(dur)
failed = int(rc != 0)
with open(path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="utf-8"?>\n')
    f.write(f'<testsuite name="examples" tests="1" failures="{failed}" '
            f'errors="0" skipped="0" time="{dur:.1f}">\n')
    f.write(f'  <testcase classname="examples" name={su.quoteattr(name)} time="{dur:.1f}">\n')
    if failed:
        f.write('    <failure message='
                + su.quoteattr(f"{cmd} exited with code {rc}")
                + '>see the examples.log artifact for the full output</failure>\n')
    f.write('  </testcase>\n')
    f.write('</testsuite>\n')
PY
}

run_example() {
    # Usage: run_example <testcase-name> <command> [args...]
    local name="$1"; shift
    local start rc end dur
    start=$(date +%s)
    set +e
    "$@"
    rc=$?
    set -e
    end=$(date +%s)
    dur=$((end - start))
    if [ -n "${JUNITXML_PREFIX}" ]; then
        mkdir -p "${JUNITXML_PREFIX}" 2>/dev/null || true
        emit_junit "${JUNITXML_PREFIX}${name}.xml" "$name" "$rc" "$dur" "$*" || true
    fi
    return $rc
}

cd "${REPO_ROOT}/examples/pytorch/mnist"
run_example pytorch_mnist      python main.py
run_example pytorch_mnist_te   python main.py --use-te
run_example pytorch_mnist_fp8  python main.py --use-fp8

cd "${REPO_ROOT}/examples/jax/mnist"
pip3 install -c "$JAX_CONSTRAINTS" -r requirements.txt
run_example jax_mnist      python test_single_gpu_mnist.py
run_example jax_mnist_te   python test_single_gpu_mnist.py --use-te
run_example jax_mnist_fp8  python test_single_gpu_mnist.py --use-fp8

cd "${REPO_ROOT}/examples/jax/encoder"
pip3 install -c "$JAX_CONSTRAINTS" -r requirements.txt
run_example jax_encoder      python test_single_gpu_encoder.py
run_example jax_encoder_fp8  python test_single_gpu_encoder.py --use-fp8
