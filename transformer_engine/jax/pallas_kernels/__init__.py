# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Pallas-backed kernels for Transformer Engine JAX.

Pallas is the JAX-native kernel-authoring API. Its Triton lowering uses the
new ``__gpu$xla.gpu.triton`` custom call, which works on both NVIDIA and
AMD/ROCm. The kernel body looks essentially like a Triton kernel - same
program_id / block-pointer model - but uses ``pl.*`` primitives instead of
``tl.*`` so it composes cleanly with JAX (jit, sharding, dtype rules).
"""

from .indexer import indexer_fused

__all__ = ["indexer_fused"]
