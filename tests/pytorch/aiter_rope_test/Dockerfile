# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Dockerfile for locally testing AITER fused RoPE integration.
#
# Build:
#   docker build -t te-aiter-rope-test -f tests/pytorch/aiter_rope_test/Dockerfile .
#
# Run (requires ROCm GPU):
#   docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
#       te-aiter-rope-test

ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

ARG TE_BRANCH=feat/aiter-fused-rope
ARG GPU_ARCHS="gfx942;gfx950"

ENV NVTE_ROCM_ARCH=${GPU_ARCHS}

WORKDIR /workspace

RUN pip install setuptools wheel pybind11 ninja pandas pytest psutil

# Install AITER from PyPI
RUN pip install amd-aiter

# Clone and install TransformerEngine from the PR branch
RUN git clone --branch ${TE_BRANCH} --recursive \
        https://github.com/ROCm/TransformerEngine.git /workspace/TransformerEngine

WORKDIR /workspace/TransformerEngine
RUN pip install -e .

# Install the PyTorch extension
WORKDIR /workspace/TransformerEngine/transformer_engine/pytorch
RUN python setup.py develop

WORKDIR /workspace/TransformerEngine

ENV NVTE_USE_AITER_ROPE=1

CMD ["pytest", "tests/pytorch/test_fused_rope.py::test_aiter_rope_matches_te_fused", \
     "tests/pytorch/test_fused_rope.py::test_aiter_rope_can_use_guard", \
     "-v", "--tb=short"]
