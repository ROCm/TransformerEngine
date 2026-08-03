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

cd "${REPO_ROOT}/examples/pytorch/mnist"
python main.py
python main.py --use-te
python main.py --use-fp8

cd "${REPO_ROOT}/examples/jax/mnist"
pip3 install -c "$JAX_CONSTRAINTS" -r requirements.txt
python test_single_gpu_mnist.py
python test_single_gpu_mnist.py --use-te
python test_single_gpu_mnist.py --use-fp8

cd "${REPO_ROOT}/examples/jax/encoder"
pip3 install -c "$JAX_CONSTRAINTS" -r requirements.txt
python test_single_gpu_encoder.py
python test_single_gpu_encoder.py --use-fp8
