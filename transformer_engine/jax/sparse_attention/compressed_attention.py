# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Heavily Compressed Attention (HCA) — design-deferred scaffold.

This module stakes out the API surface for a future MLA-style (DeepSeek-V2/V3
Multi-head Latent Attention) implementation. The Flax module and functional
entry point both raise :class:`NotImplementedError` so downstream code can
write against the eventual signature today while the design is finalized.

The intended math (for context, not yet implemented)::

    C_q = LayerNorm(X) @ W_dq        # (..., T, q_lora_rank)
    Q   = C_q @ W_uq                  # (..., T, H, qk_nope_head_dim + qk_rope_head_dim)
    C_kv = X @ W_dkv                  # (..., S, kv_lora_rank)        <-- KV cache stores this
    K   = C_kv @ W_uk                 # (..., S, H, qk_nope_head_dim + qk_rope_head_dim)
    V   = C_kv @ W_uv                 # (..., S, H, v_head_dim)
    K_rope, K_nope = split(K, ...)
    apply RoPE to (Q_rope, K_rope)
    O = softmax(Q @ K^T / sqrt(d)) @ V

See ``transformer_engine.jax.sparse_attention.dsa`` for the sibling DSA module
that is implemented today.
"""

from typing import Optional

from flax import linen as nn

from . import indexer as _indexer  # noqa: F401  — surface to assert package layout


_HCA_DEFER_MESSAGE = (
    "HeavilyCompressedAttention is a phase-1 scaffold (design deferred).\n"
    "Open design questions to resolve before implementing:\n"
    "  1. RoPE applied on compressed (C_q/C_kv) or decompressed (Q/K) tensors?\n"
    "     - DeepSeek-V2 applies RoPE on a separate sub-head; we should match.\n"
    "  2. KV cache layout: latent-only (memory-optimal) vs latent+RoPE-sub-head?\n"
    "  3. Backward through decompression: recompute (memory) vs store (bandwidth)?\n"
    "  4. Should this share projection plumbing with MultiHeadAttention's "
    "LayerNormDenseGeneral, or use bespoke low-rank projections?\n"
    "  5. Interaction with TE's existing fused-attn backends — does any of "
    "CK/AITER/cuDNN support split (RoPE/no-RoPE) head dims natively?\n"
    "Pin these before filling in. See "
    "transformer_engine.jax.sparse_attention.dsa for the working DSA module."
)


class HeavilyCompressedAttention(nn.Module):  # pylint: disable=too-few-public-methods
    """MLA-style heavily compressed attention — **DESIGN DEFERRED**.

    Parameters
    ----------
    head_dim : int
        Per-head dimension of the dense (decompressed) attention.
    num_attention_heads : int
        Number of attention heads.
    q_lora_rank : int
        Rank of the query low-rank compression (``d_c`` in indexer notation).
    kv_lora_rank : int
        Rank of the key/value low-rank compression. The KV cache stores
        only this latent (``kv_lora_rank``-dimensional) representation.
    qk_nope_head_dim : int
        Per-head dimension for the non-RoPE component of Q/K.
    qk_rope_head_dim : int
        Per-head dimension for the RoPE component of Q/K. Total Q/K head
        dim is ``qk_nope_head_dim + qk_rope_head_dim``.
    v_head_dim : int
        Per-head dimension of V (may differ from Q/K head dim).
    attn_mask_type : str, default = ``"causal"``
        Mask type. Plumbed to the eventual dense attention call.
    attention_dropout : float, default = ``0.0``
    qkv_layout : str, default = ``"bshd_bshd_bshd"``
    scale_factor : Optional[float], default = ``None``
        Defaults to ``1/sqrt(qk_nope_head_dim + qk_rope_head_dim)`` when implemented.
    """

    head_dim: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    attn_mask_type: str = "causal"
    attention_dropout: float = 0.0
    qkv_layout: str = "bshd_bshd_bshd"
    scale_factor: Optional[float] = None

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, *, deterministic: bool = False):  # noqa: D401
        del inputs_q, inputs_kv, deterministic
        raise NotImplementedError(_HCA_DEFER_MESSAGE)


def heavily_compressed_attention(
    inputs_q,
    inputs_kv,
    *,
    head_dim: int,
    num_attention_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    attn_mask_type: str = "causal",
    scale_factor: Optional[float] = None,
):
    """Functional HCA — **DESIGN DEFERRED** (raises NotImplementedError).

    Mirrors the planned :class:`HeavilyCompressedAttention` surface as a
    stateless function for callers that prefer functional composition.
    """
    del (
        inputs_q,
        inputs_kv,
        head_dim,
        num_attention_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        attn_mask_type,
        scale_factor,
    )
    raise NotImplementedError(_HCA_DEFER_MESSAGE)
