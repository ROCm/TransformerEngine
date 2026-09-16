# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Indexer op, bf16 inputs with an fp8 score matmul.

The op runs a hybrid backend: einsum projections (C_q, H_q, H_k, W_o) —
which lower to hipBLASLt bf16 GEMMs — followed by a fused Triton kernel that
does score+relu+H-reduction in registers. This avoids materializing the
(B, oH, T, H, S) pre-relu score tensor in HBM.

The score matmul itself runs in e4m3 by default (``fp8=True``), matching the
DeepSeek lightning indexer. Because these entry points own the projections,
they also own the quantization: H_q and H_k are quantized along ``d_i`` here
and handed to the Triton op together with their row scales.

The Triton op itself never quantizes. A caller that already holds
quantized index-q / index-k — from an fp8 GEMM epilogue, a TE quantizer, or a
previous step — should skip this module and call
``triton_extensions.indexer.score_reduce_triton`` directly, passing the fp8
tensors and their scales; no requantization needed.

Functional entry point: ``indexer(Q, K, W_uq, W_dq, W_k, W_w)``.
User-facing Flax module: :class:`LightningIndexer`, which owns the projection
weights and delegates to ``indexer`` / ``indexer_topk``.

Variable-length batches are supported through ``attn_mask_type`` plus the
TE-JAX ``segment_ids`` / ``segment_pos`` metadata, the same arguments fused
attention takes. :func:`_mask_keys` reduces the mask type to the ``(seg, key)``
pairs the score kernel applies; masking happens inside that kernel, so a fully
masked score tile costs no matmul at all.

Math (low-rank form: Q is hidden state; query heads are produced by a
down-projection (d -> d_c) followed by an up-projection (d_c -> H * d_i);
output weights are produced from Q via a learnable d -> H projection):

    C_q = Q @ W_dq                                           # (..., T, d_c)
    H_q = einsum("...tc,hci->...thi", C_q, W_uq)             # (..., T, H, d_i)
    H_k = K @ W_k                                             # (..., S, d_i)
    W_o = Q @ W_w                                             # (..., T, H)
    H   = relu(einsum("...thi,...si->...ths", H_q, H_k))      # (..., T, H, S)
    O   = einsum("...ths,...th->...ts", H, W_o)               # (..., T, S)
"""

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


def _indexer_projections(Q, K, W_uq, W_dq, W_k, W_w):
    """Low-rank indexer projections shared by the score and top-k paths.

    Returns (H_q, H_k, W_o) with shapes
    (..., T, H, d_i), (..., S, d_i), (..., T, H).
    """
    C_q = jnp.einsum("...td,dc->...tc", Q, W_dq)
    H_q = jnp.einsum("...tc,hci->...thi", C_q, W_uq)
    H_k = jnp.einsum("...sd,di->...si", K, W_k)
    W_o = jnp.einsum("...td,dh->...th", Q, W_w)
    return H_q, H_k, W_o


def _mask_keys(
    B,
    T_t,
    T_s,
    attn_mask_type,
    segment_ids_q=None,
    segment_pos_q=None,
    segment_ids_kv=None,
    segment_pos_kv=None,
):
    """Reduce an attention mask type to the kernel's ``(seg, key)`` pairs.

    The score kernels apply one predicate, ``seg_q == seg_k and key_q >= key_k``
    (see ``triton_extensions.indexer``), so each mask type is just a choice of
    key:

    ==================  =====================  ==========================
    ``attn_mask_type``  seg                    key
    ==================  =====================  ==========================
    ``no_mask``         -- (returns ``None``)  --
    ``causal``          all-equal              absolute position
    ``padding``         segment id             constant
    ``padding_causal``  segment id             ``segment_pos``
    ==================  =====================  ==========================

    Padding is ``segment_id == 0`` by TE-JAX convention. Those tokens are
    remapped to two *distinct* sentinels (-1 query side, -2 key side) so padding
    matches neither real tokens nor other padding; real ids are >= 1, so the
    sentinels cannot collide. Fully-padded regions then form fully-masked tiles,
    which the forward kernel skips outright.

    Returns ``(seg_q, key_q, seg_k, key_k)`` int32 arrays of shapes
    (B, T_t), (B, T_t), (B, T_s), (B, T_s) -- or ``None`` for ``no_mask``.
    """
    from transformer_engine.jax.attention import canonicalize_attn_mask_type

    mask_type = canonicalize_attn_mask_type(attn_mask_type)
    if mask_type.is_bottom_right():
        raise NotImplementedError(
            f"attn_mask_type={attn_mask_type!r} is not supported by the indexer yet; "
            "supported: 'no_mask', 'padding', 'causal', 'padding_causal'"
        )
    if not mask_type.is_causal() and not mask_type.is_padding():
        return None  # no_mask

    if mask_type.is_padding():
        if segment_ids_q is None:
            raise ValueError(
                f"attn_mask_type={attn_mask_type!r} requires segment_ids_q"
            )
        # Self-attention: reuse the query metadata when the kv side is omitted.
        if segment_ids_kv is None:
            if T_t != T_s:
                raise ValueError(
                    "segment_ids_kv is required when T_t != T_s "
                    f"(got T_t={T_t}, T_s={T_s})"
                )
            segment_ids_kv = segment_ids_q
            segment_pos_kv = segment_pos_q
        if segment_ids_q.shape != (B, T_t):
            raise ValueError(
                f"segment_ids_q has shape {segment_ids_q.shape}, expected {(B, T_t)}"
            )
        if segment_ids_kv.shape != (B, T_s):
            raise ValueError(
                f"segment_ids_kv has shape {segment_ids_kv.shape}, expected {(B, T_s)}"
            )
        seg_q = jnp.where(segment_ids_q == 0, -1, segment_ids_q).astype(jnp.int32)
        seg_k = jnp.where(segment_ids_kv == 0, -2, segment_ids_kv).astype(jnp.int32)
    else:
        # Causal without padding: one implicit segment covering every token. The
        # metadata is batch-invariant here, but the kernel indexes it by b, so
        # it is materialized at full B rather than broadcast.
        seg_q = jnp.zeros((B, T_t), jnp.int32)
        seg_k = jnp.zeros((B, T_s), jnp.int32)

    if not mask_type.is_causal():
        # Padding only -- segment identity alone decides, so the order test must
        # always pass.
        key_q = jnp.zeros((B, T_t), jnp.int32)
        key_k = jnp.zeros((B, T_s), jnp.int32)
    elif mask_type.is_padding():
        if segment_pos_q is None or segment_pos_kv is None:
            raise ValueError(
                f"attn_mask_type={attn_mask_type!r} requires segment_pos_q and "
                "segment_pos_kv (or a square self-attention shape, where "
                "segment_pos_q is reused)"
            )
        key_q = segment_pos_q.astype(jnp.int32)
        key_k = segment_pos_kv.astype(jnp.int32)
    else:
        # Top-left causal over absolute positions: query i attends key j <= i.
        key_q = jnp.broadcast_to(jnp.arange(T_t, dtype=jnp.int32), (B, T_t))
        key_k = jnp.broadcast_to(jnp.arange(T_s, dtype=jnp.int32), (B, T_s))

    return seg_q, key_q, seg_k, key_k


def _maybe_quantize(H_q, H_k, fp8):
    """Quantize the score-matmul operands, or pass them through in bf16.

    Returns ``(H_q, H_k, Sq, Ks)`` with ``Sq`` / ``Ks`` set to ``None`` (unit
    scales) on the bf16 path. Quantization is skipped when ``d_i`` is too small
    for an e4m3 MFMA tile, where padding K would cost more than fp8 saves.
    """
    from transformer_engine.jax.triton_extensions.indexer import (
        fp8_dot_supported,
        quantize_e4m3,
    )

    if not (fp8 and fp8_dot_supported(H_q.shape[-1])):
        return H_q, H_k, None, None
    H_q_q, Sq = quantize_e4m3(H_q)
    H_k_q, Ks = quantize_e4m3(H_k)
    return H_q_q, H_k_q, Sq, Ks


def _indexer_impl_hybrid(Q, K, W_uq, W_dq, W_k, W_w, out_dtype=None, fp8=True,
                         attn_mask_type="no_mask", segment_ids_q=None,
                         segment_pos_q=None, segment_ids_kv=None,
                         segment_pos_kv=None):
    """Einsum projections + Triton score-relu-reduce.

    Runs the four projections (which lower to hipBLASLt bf16 GEMMs), then
    hands Hq / Hk / W_o to a fused Triton kernel that does
    score+relu+H-reduction in registers — eliminating the
    (B, oH, T, H, S) pre-relu-score HBM round-trip a pure-einsum path pays.
    """
    from transformer_engine.jax.attention import canonicalize_attn_mask_type
    from transformer_engine.jax.triton_extensions.indexer import score_reduce_triton

    mask = _mask_keys(
        Q.shape[0], Q.shape[-2], K.shape[-2], attn_mask_type,
        segment_ids_q, segment_pos_q, segment_ids_kv, segment_pos_kv,
    )
    # The causal family keys on position (``causal``) or segment_pos
    # (``padding_causal``), so a valid pair always has t >= s and the kernel can
    # take its triangular grid. ``padding`` keys on a constant -- its valid set
    # is block-diagonal, not triangular -- so it must not claim this.
    mask_is_causal = canonicalize_attn_mask_type(attn_mask_type).is_causal()
    H_q, H_k, W_o = _indexer_projections(Q, K, W_uq, W_dq, W_k, W_w)
    H_q, H_k, Sq, Ks = _maybe_quantize(H_q, H_k, fp8)
    return score_reduce_triton(H_q, H_k, W_o, Sq=Sq, Ks=Ks, mask=mask,
                               out_dtype=out_dtype if out_dtype else Q.dtype,
                               mask_is_causal=mask_is_causal)


@functools.partial(jax.jit, static_argnames=("k", "fp8", "attn_mask_type"))
def indexer_topk(Q, K, W_uq, W_dq, W_k, weights, *, k, fp8=True,
                 attn_mask_type="no_mask", segment_ids_q=None,
                 segment_pos_q=None, segment_ids_kv=None, segment_pos_kv=None):
    """Lightning-indexer logits followed by a top-k selection.

    Runs ``indexer()`` to produce the (..., T_t, T_s) logits, then selects the
    top ``k`` per row with ``jax.lax.top_k``. The selection is a separate pass
    over the materialized logits, not fused into the score kernel.

    Args:
        Q, K, W_uq, W_dq, W_k, weights: same as ``indexer()``.
        k: number of top scores to return per (B, oH, T_t) row. Must be <= S.
        fp8: run the score matmul in e4m3 (default).
        attn_mask_type, segment_ids_q, segment_pos_q, segment_ids_kv,
        segment_pos_kv: same as ``indexer()``.

    Returns:
        Topk_idx: (..., T_t, k) int32 — top-k indices into the S axis,
        in descending score order.

    Under a mask, masked slots hold ``finfo(dtype).min`` and so rank below every
    valid logit. When ``k`` exceeds a row's valid-key count -- a short segment,
    or an early query under a causal mask -- the surplus entries are those
    masked slots, and their indices are meaningless. That is inherent to asking
    for more keys than a row has; the caller must bound ``k`` or ignore the
    tail.
    """
    scores = _indexer_impl_hybrid(
        Q, K, W_uq, W_dq, W_k, weights, fp8=fp8,
        attn_mask_type=attn_mask_type, segment_ids_q=segment_ids_q,
        segment_pos_q=segment_pos_q, segment_ids_kv=segment_ids_kv,
        segment_pos_kv=segment_pos_kv,
    )
    return jax.lax.top_k(scores, k)[1]


@functools.partial(jax.jit, static_argnames=("out_dtype", "fp8", "attn_mask_type"))
def indexer(Q, K, W_uq, W_dq, W_k, weights, *, out_dtype=None, fp8=True,
            attn_mask_type="no_mask", segment_ids_q=None, segment_pos_q=None,
            segment_ids_kv=None, segment_pos_kv=None):
    """Low-rank lightning-indexer (bf16 I/O, fp8 score matmul).

    Args:
        Q:       (B, oH, T, d)          hidden state (per token)
        K:       (B, oH, S, d)          key hidden state
        W_uq:    (H, d_c, d_i)          up-projection: d_c -> d_i (per head)
        W_dq:    (d, d_c)               down-projection: d -> d_c
        W_k:     (d, d_i)               key projection
        weights: (d, H)                 learnable output-weight projection
                                        (W_o = Q @ weights inside the impl)
        out_dtype: output dtype override (defaults to Q.dtype).
        fp8: run the score matmul in e4m3 (default). Quantization happens
             inside the Triton op, so inputs and gradients stay bf16.
        attn_mask_type: one of ``'no_mask'`` (default), ``'padding'``,
            ``'causal'``, ``'padding_causal'``. Bottom-right-causal and sliding
            window are not supported yet.
        segment_ids_q, segment_pos_q: (B, T) int32 variable-length metadata, as
            used by TE-JAX fused attention — ``segment_ids`` labels which packed
            segment each token belongs to (0 = padding) and ``segment_pos`` is
            its position within that segment. Required by the ``padding*`` mask
            types.
        segment_ids_kv, segment_pos_kv: (B, S) key-side equivalents. Omit for
            self-attention (T == S), where the query metadata is reused.

    Returns:
        O of shape (B, oH, T, S). Masked positions hold ``finfo(out_dtype).min``
        — the logits are signed, so a zero fill would outrank valid negative
        logits under a downstream top-k.
    """
    return _indexer_impl_hybrid(
        Q, K, W_uq, W_dq, W_k, weights, out_dtype=out_dtype, fp8=fp8,
        attn_mask_type=attn_mask_type, segment_ids_q=segment_ids_q,
        segment_pos_q=segment_pos_q, segment_ids_kv=segment_ids_kv,
        segment_pos_kv=segment_pos_kv,
    )


class LightningIndexer(nn.Module):  # pylint: disable=too-few-public-methods
    """Lightning-indexer Flax module — the user-facing indexer API.

    Owns the low-rank indexer projection weights (``W_dq``, ``W_uq``, ``W_k``,
    ``W_w``) and delegates to the functional :func:`indexer` / :func:`indexer_topk`
    ops. Weight shapes mirror :func:`indexer`'s ``Args`` and are inferred from the
    trailing hidden dimension ``d`` of ``Q`` at call time.

    Parameters
    ----------
    num_heads : int
        Number of indexer-internal heads (``H``).
    d_c : int
        Down-projection rank (``d -> d_c``).
    d_i : int
        Inner head dimension (``d_i``).
    topk : Optional[int], default ``None``
        If set, :meth:`__call__` returns the top-``k`` indices
        (``(..., T, k)`` int32) via :func:`indexer_topk`, and ``out_dtype`` is
        ignored (the logits are selected over in their native dtype).
        If ``None``, :meth:`__call__` returns the full score tensor
        ``(..., T, S)`` (hybrid Triton backend).
    out_dtype : Optional[jnp.dtype]
        Output dtype override; defaults to ``Q.dtype``. Unused when ``topk`` is set.
    dtype : Optional[jnp.dtype]
        Parameter dtype. Defaults to the input dtype.
    fp8 : bool, default ``True``
        Run the score matmul in e4m3. Quantization is internal to the Triton
        op, so parameters, activations and gradients remain ``dtype``.
    attn_mask_type : str, default ``'no_mask'``
        One of ``'no_mask'``, ``'padding'``, ``'causal'``, ``'padding_causal'``.
        The ``padding*`` types need the segment metadata passed to
        :meth:`__call__`.
    """

    num_heads: int
    d_c: int
    d_i: int
    topk: Optional[int] = None
    out_dtype: Optional[jnp.dtype] = None
    dtype: Optional[jnp.dtype] = None
    fp8: bool = True
    attn_mask_type: str = "no_mask"

    @nn.compact
    def __call__(
        self,
        Q: jax.Array,
        K: jax.Array,
        segment_ids_q: Optional[jax.Array] = None,
        segment_pos_q: Optional[jax.Array] = None,
        segment_ids_kv: Optional[jax.Array] = None,
        segment_pos_kv: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Run the indexer on ``Q`` / ``K``.

        Args:
            Q: ``(B, oH, T, d)`` query-side hidden state.
            K: ``(B, oH, S, d)`` key-side hidden state.
            segment_ids_q, segment_pos_q, segment_ids_kv, segment_pos_kv:
                variable-length (THD) metadata; see :func:`indexer`. Required by
                the ``padding*`` mask types, ignored otherwise.

        Returns:
            ``(B, oH, T, S)`` scores if ``topk is None``, else
            ``(B, oH, T, k)`` int32 top-k indices (in descending score order).
        """
        d = Q.shape[-1]
        param_dtype = self.dtype if self.dtype is not None else Q.dtype
        init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")

        W_dq = self.param("W_dq", init, (d, self.d_c), param_dtype)
        W_uq = self.param("W_uq", init, (self.num_heads, self.d_c, self.d_i), param_dtype)
        W_k = self.param("W_k", init, (d, self.d_i), param_dtype)
        W_w = self.param("W_w", init, (d, self.num_heads), param_dtype)

        mask_kwargs = {
            "attn_mask_type": self.attn_mask_type,
            "segment_ids_q": segment_ids_q,
            "segment_pos_q": segment_pos_q,
            "segment_ids_kv": segment_ids_kv,
            "segment_pos_kv": segment_pos_kv,
        }
        if self.topk is not None:
            return indexer_topk(
                Q, K, W_uq, W_dq, W_k, W_w, k=self.topk, fp8=self.fp8, **mask_kwargs
            )
        return indexer(
            Q, K, W_uq, W_dq, W_k, W_w, out_dtype=self.out_dtype, fp8=self.fp8,
            **mask_kwargs
        )
