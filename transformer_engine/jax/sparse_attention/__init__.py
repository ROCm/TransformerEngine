# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Sparse attention primitives.

Currently exposes the lightning indexer:

  * :mod:`~transformer_engine.jax.sparse_attention.indexer` — the lightning
    indexer op (``indexer`` / ``indexer_topk``) and the :class:`LightningIndexer`
    Flax module.

The Triton kernel backends live in
:mod:`transformer_engine.jax.triton_extensions` alongside the other Triton
kernels.
"""

from . import indexer

from .indexer import LightningIndexer

__all__ = [
    "indexer",
    "LightningIndexer",
]
