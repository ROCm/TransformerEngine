# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Deep Sparse Attention (DSA) — composes the lightning indexer with dense attention.

The pipeline composes existing pieces:

    1. Per-attention-head Q/K/V projection (DenseGeneral)
    2. Lightning-indexer scoring via the hybrid Triton backend
    3. Causal mask + jax.lax.top_k on each per-head score row
    4. Scatter top-k indices into a per-head sparse attention mask
    5. Call transformer_engine.jax.flax.DotProductAttention with that mask

**Shape contract — all DSA tensors are rank-4 with the outer-head dim
explicit:**

    inputs_q  : [B, oH, T_t, hidden]
    inputs_kv : [B, oH, T_s, hidden]
    output    : [B, oH, T_t, head_dim]

``oH ≡ num_attention_heads``. Each attention head has its own indexer
score row, its own top-k pattern, and its own attention output. The
indexer projection *weights* are shared across attention heads — the
per-head divergence comes from the per-head input slice (the caller is
expected to have already produced per-head hidden states upstream).

This shape contract aligns with the lightning-indexer benchmark's
``[B, oH, T, d]`` convention (see ``benchmarks/profile_indexer_topk.py``)
and lets us call the Triton hybrid backend directly without rank
adjustment.

Zero modifications are made to upstream-tracked TE files; DSA composes
:class:`DotProductAttention` from the outside via its public ``mask=``
argument.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from transformer_engine.jax.flax.module import DenseGeneral
from transformer_engine.jax.flax.transformer import DotProductAttention
from .indexer import indexer as _indexer_fn


# -----------------------------------------------------------------------------
# Mask construction helpers
# -----------------------------------------------------------------------------


def _causal_keep_mask(T_t: int, T_s: int, dtype=jnp.bool_):
    """Lower-triangular keep mask aligned to the bottom-right corner.

    For self-attention (T_t == T_s) this is the standard ``jnp.tril(ones)``.
    For cross-attention with T_t < T_s, query position ``t`` attends to key
    positions ``[0, T_s - T_t + t]``. This matches the convention used by
    causal cross-attention with prefix context.
    """
    q_pos = jnp.arange(T_t)[:, None]                       # [T_t, 1]
    k_pos = jnp.arange(T_s)[None, :]                       # [1, T_s]
    keep = k_pos <= (q_pos + (T_s - T_t))                  # [T_t, T_s]
    return keep.astype(dtype)


def _topk_indices_to_attn_mask(
    indices: jax.Array,
    T_s: int,
    *,
    causal: bool,
) -> jax.Array:
    """Convert per-(B, oH, T_t) top-k indices into a DPA-style mask.

    Args:
        indices: ``[B, oH, T_t, k]`` int32 — top-k key positions per (B, oH, T_t).
        T_s: number of key positions.
        causal: if True, AND the keep-mask with a causal keep-mask before
            inverting.

    Returns:
        ``[B, oH, T_t, T_s]`` uint8 — ``1`` means *mask out*. The caller
        reshapes to ``[B*oH, 1, T_t, T_s]`` for DPA dispatch.
    """
    B, oH, T_t, _k = indices.shape

    # Scatter True at every (b, h, t, indices[b, h, t, :]) position.
    keep = jnp.zeros((B, oH, T_t, T_s), dtype=jnp.bool_)
    b_idx = jnp.arange(B)[:, None, None, None]            # [B, 1, 1, 1]
    h_idx = jnp.arange(oH)[None, :, None, None]           # [1, oH, 1, 1]
    t_idx = jnp.arange(T_t)[None, None, :, None]          # [1, 1, T_t, 1]
    # Duplicates from .at[].set(True) are idempotent — safe when k > finite scores.
    keep = keep.at[b_idx, h_idx, t_idx, indices].set(True)  # [B, oH, T_t, T_s]

    if causal:
        keep = keep & _causal_keep_mask(T_t, T_s)[None, None, :, :]

    mask_out = jnp.logical_not(keep)
    # TE's ScaledMaskedSoftmax expects uint8 mask (cpp_extensions/softmax.py:483).
    return mask_out.astype(jnp.uint8)                      # [B, oH, T_t, T_s]


# -----------------------------------------------------------------------------
# Functional API
# -----------------------------------------------------------------------------


def deep_sparse_attention_core(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    indexer_inputs_q: jax.Array,
    indexer_inputs_kv: jax.Array,
    indexer_W_uq: jax.Array,
    indexer_W_dq: jax.Array,
    indexer_W_k: jax.Array,
    indexer_W_w: jax.Array,
    *,
    k: int,
    attn_mask_type: str = "causal",
    scale_factor: Optional[float] = None,
    attention_dropout: float = 0.0,
    deterministic: bool = True,
    dropout_rng_name: str = "dropout",
    indexer_backend: str = "hybrid",
) -> jax.Array:
    """Functional DSA: indexer-top-k + per-head sparse attention.

    Args:
        query, key, value: ``[B, oH, T, head_dim]`` — post-projection per-head
            attention tensors. ``oH ≡ num_attention_heads``; each outer-head
            slice owns a single attention head of dimension ``head_dim``.
        indexer_inputs_q:  ``[B, oH, T_t, hidden]`` — per-head hidden states
            fed to the indexer's query side.
        indexer_inputs_kv: ``[B, oH, T_s, hidden]`` — per-head hidden states
            fed to the indexer's key side.
        indexer_W_uq: ``[H_idx, d_c, d_i]`` indexer up-projection (shared).
        indexer_W_dq: ``[hidden, d_c]`` indexer down-projection (shared).
        indexer_W_k:  ``[hidden, d_i]`` indexer key projection (shared).
        indexer_W_w:  ``[hidden, H_idx]`` indexer output-weight projection (shared).
        k: number of top key positions to retain per (B, oH, T_t).
        attn_mask_type: ``"causal"`` or ``"no_mask"`` (phase 1 only).
        scale_factor: passed through to DPA. ``None`` → ``1/sqrt(head_dim)``.
        attention_dropout, deterministic, dropout_rng_name: passed through to DPA.
        indexer_backend: which indexer implementation to use. ``"hybrid"``
            (default, fast Triton) or ``"reference"`` (pure einsum).

    Returns:
        Attention output of the same shape as ``query``: ``[B, oH, T_t, head_dim]``.
    """
    if attn_mask_type not in ("causal", "no_mask"):
        raise NotImplementedError(
            f"deep_sparse_attention_core: attn_mask_type={attn_mask_type!r} "
            "not supported in phase 1. Supported: 'causal', 'no_mask'. "
            "(Padding / segment-id mask types are tracked as a follow-up.)"
        )

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"DSA expects rank-4 query/key/value [B, oH, T, head_dim]; got "
            f"shapes query={query.shape} key={key.shape} value={value.shape}"
        )
    if indexer_inputs_q.ndim != 4 or indexer_inputs_kv.ndim != 4:
        raise ValueError(
            f"DSA expects rank-4 indexer inputs [B, oH, T, hidden]; got "
            f"shapes indexer_inputs_q={indexer_inputs_q.shape} "
            f"indexer_inputs_kv={indexer_inputs_kv.shape}"
        )

    B, oH, T_t, head_dim = query.shape
    T_s = key.shape[2]
    if key.shape != (B, oH, T_s, head_dim) or value.shape != (B, oH, T_s, head_dim):
        raise ValueError(
            f"DSA shape mismatch: query={query.shape} key={key.shape} value={value.shape}"
        )

    # 1. Indexer produces a per-head score row [B, oH, T_t, T_s].
    scores = _indexer_fn(
        indexer_inputs_q,
        indexer_inputs_kv,
        indexer_W_uq,
        indexer_W_dq,
        indexer_W_k,
        indexer_W_w,
        backend=indexer_backend,
        out_dtype=jnp.float32,
    )                                                       # [B, oH, T_t, T_s] fp32

    # 2. Causal mask BEFORE top-k so non-causal positions are excluded.
    causal = (attn_mask_type == "causal")
    if causal:
        ckeep = _causal_keep_mask(T_t, T_s)[None, None, :, :]    # [1, 1, T_t, T_s]
        scores = jnp.where(ckeep, scores, jnp.asarray(-jnp.inf, dtype=scores.dtype))

    # 3. Per-(B, oH, T_t) top-k.
    k_eff = min(k, T_s)
    _, topk_idx = jax.lax.top_k(scores, k_eff)              # [B, oH, T_t, k_eff]

    # 4. Scatter into [B, oH, T_t, T_s] uint8 DPA mask (1 = mask out).
    sparse_mask = _topk_indices_to_attn_mask(
        topk_idx, T_s, causal=causal,
    )                                                       # [B, oH, T_t, T_s] uint8

    # 5. Dense attention with the sparse mask. We collapse (B, oH) into the
    # batch dim of DPA so each attention head gets its own mask. attn_mask_type
    # 'padding' tells DPA to honor the provided mask as-is (causal is baked in).
    BH = B * oH
    q_r = query.reshape(BH, T_t, 1, head_dim)              # [BH, T_t, 1, D]
    k_r = key.reshape(BH, T_s, 1, head_dim)
    v_r = value.reshape(BH, T_s, 1, head_dim)
    mask_r = sparse_mask.reshape(BH, 1, T_t, T_s)          # [BH, 1, T_t, T_s]

    dpa = DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=1,
        num_gqa_groups=1,    # one head per oH slice; must be int (probe rejects None)
        attention_dropout=attention_dropout,
        attn_mask_type="padding",
        qkv_layout="bshd_bshd_bshd",
        scale_factor=scale_factor,
        dropout_rng_name=dropout_rng_name,
    )
    out = dpa(
        q_r, k_r, v_r,
        sequence_descriptor=mask_r,
        deterministic=deterministic,
    )                                                       # [BH, T_t, head_dim] (flattened H=1)
    # DPA flattens the H=1 axis on output. Reshape back to [B, oH, T_t, head_dim].
    return out.reshape(B, oH, T_t, head_dim)


# -----------------------------------------------------------------------------
# Flax module
# -----------------------------------------------------------------------------


class DeepSparseAttention(nn.Module):  # pylint: disable=too-few-public-methods
    """Deep Sparse Attention (DSA) Flax module — rank-4, per-attention-head.

    Composes the lightning indexer with TE's :class:`DotProductAttention`.
    Each attention head (``oH``) has its own indexer score row, top-k
    pattern, and dense-attention output. Indexer projection weights are
    shared across heads.

    Parameters
    ----------
    head_dim : int
        Per-attention-head dimension.
    num_attention_heads : int
        Number of attention heads (``oH``).
    indexer_num_heads : int
        Number of indexer-internal heads (``H`` in the indexer notation).
    indexer_d_c : int
        Indexer down-projection rank (``d_c``).
    indexer_d_i : int
        Indexer inner head dimension (``d_i``).
    topk : int
        Number of top key positions to retain per query.
    attn_mask_type : str, default ``"causal"``
        ``"causal"`` or ``"no_mask"`` (phase 1).
    attention_dropout : float, default ``0.0``
    scale_factor : Optional[float]
        Defaults to ``1/sqrt(head_dim)`` inside DPA.
    indexer_backend : str, default ``"hybrid"``
        ``"hybrid"`` (fast Triton) or ``"reference"`` (pure einsum).
    dtype : Optional[jnp.dtype]
        Parameter dtype. Defaults to the input dtype.
    """

    head_dim: int
    num_attention_heads: int
    indexer_num_heads: int
    indexer_d_c: int
    indexer_d_i: int
    topk: int
    attn_mask_type: str = "causal"
    attention_dropout: float = 0.0
    scale_factor: Optional[float] = None
    indexer_backend: str = "hybrid"
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(
        self,
        inputs_q: jax.Array,
        inputs_kv: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Run DSA on rank-4 per-head inputs.

        Args:
            inputs_q:  ``[B, oH, T_t, hidden]`` — per-head query-side hidden state.
            inputs_kv: ``[B, oH, T_s, hidden]`` — per-head key-side hidden state.
            deterministic: forwarded to DPA.

        Returns:
            ``[B, oH, T_t, head_dim]`` — per-head attention output.
        """
        if inputs_q.ndim != 4 or inputs_kv.ndim != 4:
            raise ValueError(
                f"DeepSparseAttention expects rank-4 inputs [B, oH, T, hidden]; "
                f"got inputs_q.shape={inputs_q.shape}, inputs_kv.shape={inputs_kv.shape}"
            )
        B, oH, T_t, hidden = inputs_q.shape
        if oH != self.num_attention_heads:
            raise ValueError(
                f"DeepSparseAttention: inputs_q.shape[1]={oH} must equal "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if inputs_kv.shape[0] != B or inputs_kv.shape[1] != oH or inputs_kv.shape[3] != hidden:
            raise ValueError(
                f"DeepSparseAttention: inputs_kv.shape={inputs_kv.shape} must match "
                f"(B={B}, oH={oH}, T_s, hidden={hidden})"
            )

        param_dtype = self.dtype if self.dtype is not None else inputs_q.dtype

        # ---- per-head Q/K/V projections ----
        # DenseGeneral with features=head_dim and axis=-1 maps [..., hidden] →
        # [..., head_dim], preserving the (B, oH, T) leading dims. Each attention
        # head (oH slice) shares the projection kernel — divergence comes from the
        # per-head input slice the caller provides.
        query = DenseGeneral(
            features=self.head_dim,
            use_bias=False,
            dtype=param_dtype,
            name="query",
        )(inputs_q)                                         # [B, oH, T_t, head_dim]
        key = DenseGeneral(
            features=self.head_dim,
            use_bias=False,
            dtype=param_dtype,
            name="key",
        )(inputs_kv)                                        # [B, oH, T_s, head_dim]
        value = DenseGeneral(
            features=self.head_dim,
            use_bias=False,
            dtype=param_dtype,
            name="value",
        )(inputs_kv)                                        # [B, oH, T_s, head_dim]

        # ---- indexer projections (shared across oH) ----
        # Shapes mirror transformer_engine.jax.sparse_attention.indexer.
        init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
        W_dq = self.param("indexer_W_dq", init, (hidden, self.indexer_d_c), param_dtype)
        W_uq = self.param(
            "indexer_W_uq", init,
            (self.indexer_num_heads, self.indexer_d_c, self.indexer_d_i), param_dtype,
        )
        W_k_idx = self.param("indexer_W_k", init, (hidden, self.indexer_d_i), param_dtype)
        W_w = self.param("indexer_W_w", init, (hidden, self.indexer_num_heads), param_dtype)

        return deep_sparse_attention_core(
            query, key, value,
            inputs_q, inputs_kv,
            W_uq, W_dq, W_k_idx, W_w,
            k=self.topk,
            attn_mask_type=self.attn_mask_type,
            scale_factor=self.scale_factor,
            attention_dropout=self.attention_dropout,
            deterministic=deterministic,
            indexer_backend=self.indexer_backend,
        )                                                   # [B, oH, T_t, head_dim]
