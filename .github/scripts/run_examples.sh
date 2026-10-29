#!/usr/bin/bash
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Run TE examples on a single GPU.
# HIP_VISIBLE_DEVICES must be set by the caller (ci/run_queue.sh).

set -e

# Autodetect repo root from this script's location (.github/scripts/ -> ../..)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"

JAX_CONSTRAINTS=/tmp/jax-constraints.txt
pip freeze | grep -iE '^(jax|jaxlib|jax[_-]rocm|jax[_-]plugins)[=@]' > "$JAX_CONSTRAINTS" || true

# Every step runs through run_example.py, which records it as one JUnit test case
# so the examples get a section of their own in the job summary. A failing step is
# recorded and the rest still run, so the summary names every broken example; the
# script then fails at the end.
EXAMPLES_RC=0
run_example() {
    python "${SCRIPT_DIR}/run_example.py" "$@" || EXAMPLES_RC=1
}

cd "${REPO_ROOT}/examples/pytorch/mnist"
run_example pytorch.mnist python main.py
run_example pytorch.mnist.use-te python main.py --use-te
run_example pytorch.mnist.use-fp8 python main.py --use-fp8

cd "${REPO_ROOT}/examples/jax/mnist"
run_example jax.mnist.requirements pip3 install -c "$JAX_CONSTRAINTS" -r requirements.txt
run_example jax.mnist python test_single_gpu_mnist.py
run_example jax.mnist.use-te python test_single_gpu_mnist.py --use-te
run_example jax.mnist.use-fp8 python test_single_gpu_mnist.py --use-fp8

cd "${REPO_ROOT}/examples/jax/encoder"
run_example jax.encoder.requirements pip3 install -c "$JAX_CONSTRAINTS" -r requirements.txt
# These scripts import their sibling common.py, and ci/_utils.sh exports
# PYTHONSAFEPATH=1, which drops the script's own directory from sys.path. Name the
# directory explicitly rather than opting out: the flag is what keeps the source tree
# from shadowing the installed package, and it is still wanted here.
ENCODER_PYTHONPATH="${REPO_ROOT}/examples/jax/encoder${PYTHONPATH:+:${PYTHONPATH}}"
PYTHONPATH="$ENCODER_PYTHONPATH" run_example jax.encoder python test_single_gpu_encoder.py
PYTHONPATH="$ENCODER_PYTHONPATH" run_example jax.encoder.use-fp8 \
    python test_single_gpu_encoder.py --use-fp8

exit $EXAMPLES_RC
