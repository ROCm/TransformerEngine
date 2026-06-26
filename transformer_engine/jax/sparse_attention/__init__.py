# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Deep Sparse Attention (DSA) family.

Bundles the lightning indexer and the attention module built on top of it:

  * :mod:`~transformer_engine.jax.sparse_attention.indexer` — the lightning
    indexer op (``indexer`` / ``indexer_topk``).
  * :mod:`~transformer_engine.jax.sparse_attention.dsa` — Deep Sparse
    Attention, which composes the indexer with dense attention.

The Triton kernel backends live in
:mod:`transformer_engine.jax.triton_extensions` alongside the other Triton
kernels.
"""

from . import indexer
from . import dsa

from .indexer import LightningIndexer
from .dsa import DeepSparseAttention, deep_sparse_attention_core

__all__ = [
    "indexer",
    "dsa",
    "LightningIndexer",
    "DeepSparseAttention",
    "deep_sparse_attention_core",
]
