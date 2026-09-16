# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Triton score-relu-reduce kernel for the lightning-indexer hybrid backend.

FP8 score matmul
----------------
These ops never quantize. ``Hq`` / ``Hk`` are consumed in whatever precision
they arrive in -- e4m3 or bf16 -- alongside the per-row scales ``Sq`` / ``Ks``,
so a caller holding already-quantized index-q / index-k can hand them straight
in without a dequantize/requantize round trip. ``tl.dot`` specializes on the
operand dtype, so one kernel serves both precisions; omitting the scales means
"already unit-scaled" and passes all-ones.

The expected quantization (see :func:`quantize_e4m3`, which callers may use but
are not required to) is per-row along the contracted ``d_i`` axis: ``Sq`` per
(b, oH, t, h) and ``Ks`` per (b, oH, s), i.e. the row scale of each matmul
operand. Because ReLU is positive-homogeneous and both scales are strictly
positive, dequantization commutes with it and factors out of the H-reduction:

    O[t, s] = sum_h relu(sq[t,h] * ks[s] * (q_hat . k_hat)) * W_o[t,h]
            = ks[s] * sum_h relu(q_hat . k_hat) * (W_o[t,h] * sq[t,h])

So ``Sq`` folds into the per-head weight and ``Ks`` is a single per-column
post-multiply -- no separate dequantization pass over the operands.

Gradients follow the operand dtype: with e4m3 ``Hq`` / ``Hk`` the cotangents
come back as e4m3 too. They are the chain-rule cotangents w.r.t. the *quantized*
operands (``dHq_true * sq``), which renormalizes them into the operand's own
range rather than leaving a raw unscaled fp8 gradient. ``Sq`` / ``Ks`` are
quantization metadata and are treated as constants (zero cotangent).

Masking
-------
An optional ``mask`` restricts which (t, s) pairs are scored, which is what lets
the indexer run on packed variable-length (THD) batches without scoring across
segment boundaries. It is a ``(seg_q, key_q, seg_k, key_k)`` tuple of int32
arrays and the kernels apply exactly one predicate::

    valid[t, s] = (seg_q[t] == seg_k[s]) and (key_q[t] >= key_k[s])

Every supported attention mask reduces to a choice of key, made by the caller,
so the kernels carry no mask-type switch. Masked slots in the forward output get
``finfo(out_dtype).min`` rather than zero: the ReLU sits *inside* the
H-reduction, ahead of the signed ``W_o`` weighting, so the logits themselves are
signed and a zero fill would outrank a valid negative logit under a downstream
top-k. Backward needs no fill -- the mask is ANDed into the ReLU mask, which
zeroes both ``dscores`` and the ``dW_o`` reduction at a masked pair.

A tile with no valid pair skips its score matmul entirely; under a causal mask
that is roughly half of them.
"""

import copy
import functools
import os

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from jax import core
from jax.extend import core as extend_core
from jax.interpreters import mlir, xla

from transformer_engine.jax.util import get_jnp_float8_e4m3_type

from .utils import triton_call_lowering


@functools.lru_cache(maxsize=None)
def _tile_skip_enabled():
    """False when ``NVTE_INDEXER_TILE_SKIP=0`` — A/B knob for the tile skip."""
    return os.environ.get("NVTE_INDEXER_TILE_SKIP", "1") != "0"


@functools.lru_cache(maxsize=None)
def _autotune_disabled():
    """True when ``NVTE_INDEXER_DISABLE_AUTOTUNE=1``.

    When set, each kernel's lowering collapses its autotune sweep to the first
    (still prune-valid) config, so no time is spent compiling and benchmarking
    every candidate. Intended for the test suite — a full sweep at large k/T_s
    costs many minutes and only picks the fastest config, not a more correct
    one. Read once at first lowering and cached."""
    return os.environ.get("NVTE_INDEXER_DISABLE_AUTOTUNE", "0") == "1"


# Below this T_t the triangular grid loses: the kernel is latency-bound rather
# than compute-bound, so the tile work it skips was not costing full price,
# while the MASK_FILL prefill it requires is a real extra pass over the output.
# Measured crossover on gfx950 (fp8 operands, bf16 out, H=32, d_i=128): 0.94x at
# T=2048, 1.30x at 4096, 1.60x at 8192. Revisit if the fill stops being a
# separate pass -- see the note in _score_reduce_kernel.
_TRI_GRID_MIN_T = 4096


@functools.lru_cache(maxsize=None)
def _tri_grid_override():
    """``NVTE_INDEXER_TRI_GRID`` — A/B knob for the causal grid.

    Returns True to force the triangular grid on (used by the tests, which run
    shapes far below _TRI_GRID_MIN_T and would otherwise never exercise it),
    False to force the rectangular grid, or None to use the size heuristic."""
    val = os.environ.get("NVTE_INDEXER_TRI_GRID")
    if val is None:
        return None
    return val != "0"


# --- Causal triangular grid --------------------------------------------------
#
# Under a causal-family mask every valid (t, s) pair has t >= s, so roughly half
# the rectangular grid's tiles are fully masked. Those tiles are cheap
# individually but are not free: they interleave with working tiles along the
# fast-dispatching axis, and measurement shows the resulting imbalance costs
# nearly the whole benefit of masking (a causal mask runs at ~1.0x the cost of
# no mask at all, against an available 0.5x). The fix is to not dispatch them:
# enumerate only the tiles that meet the causal region, on a 1-D grid.
#
# Both the host (to size the grid) and the kernel (to invert a linear tile id)
# must agree exactly on that enumeration, so both derive it from the same four
# numbers -- T_t, T_s, BLOCK_T, BLOCK_S -- using the identities below.
#
# Per t-tile i, the number of s-tiles meeting the causal region is
#
#     c(i) = min(n_s, min(i * BT + BT - 1, T_t - 1) // BS + 1)
#
# c is nondecreasing and saturates at n_s, which splits the grid into a
# triangular head (c still growing) and a rectangular tail (c == n_s). Carrying
# the tail explicitly is what keeps this exact when BS > T_s or when the shapes
# do not divide evenly.


def _tri_grid_params(T_t, T_s, BLOCK_T, BLOCK_S):
    """Derive the causal tile-grid constants for one (shape, config) pair.

    Mirrored verbatim by ``_score_reduce_kernel`` from its own constexprs.
    Returns a dict with the launch size in ``n_tiles``.
    """
    n_t = -(-T_t // BLOCK_T)
    n_s = -(-T_s // BLOCK_S)

    if BLOCK_S >= BLOCK_T:
        # c(i) = i // ratio + 1 -- `ratio` consecutive t-tiles share a count.
        ratio = BLOCK_S // BLOCK_T
        case_a = True
        i_star = min(n_t, (n_s - 1) * ratio)
        if i_star == n_t:  # saturation never reached within the grid
            tri_count = sum(i // ratio + 1 for i in range(n_t))
        else:
            tri_count = ratio * (n_s - 1) * n_s // 2
    else:
        # c(i) = (i + 1) * ratio
        ratio = BLOCK_T // BLOCK_S
        case_a = False
        i_star = min(n_t, max(0, -(-n_s // ratio) - 1))
        tri_count = ratio * i_star * (i_star + 1) // 2

    return {
        "n_t": n_t,
        "n_s": n_s,
        "ratio": ratio,
        "case_a": case_a,
        "i_star": i_star,
        "tri_count": tri_count,
        "n_tiles": tri_count + (n_t - i_star) * n_s,
    }


# --- FP8 operand quantization ------------------------------------------------
#
# e4m3 is the score-matmul operand format. On gfx942 the hardware format is the
# "fnuz" variant (exponent bias 8); everywhere else it is OCP e4m3. Both are
# already mapped to Triton type strings in ``utils.get_triton_dtype``, so the
# kernels below are dtype-polymorphic and need no per-variant handling.


def fp8_dtype():
    """e4m3 dtype matching the current device's hardware format."""
    return jnp.dtype(get_jnp_float8_e4m3_type())


def is_fp8(dtype):
    """Whether ``dtype`` is one of the 8-bit float formats."""
    dtype = jnp.dtype(dtype)
    return jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize == 1


def fp8_dot_supported(d_i):
    """Whether the score matmul is worth running in fp8 for this inner dim.

    The narrowest e4m3 MFMA/WGMMA tile contracts 32 elements, so a ``d_i``
    below that would have Triton pad the K axis with zeros -- costing more than
    the fp8 speedup buys. Callers should stay in bf16 there.
    """
    return d_i >= 32


def quantize_e4m3(x, *, axis=-1):
    """Symmetric per-row e4m3 quantization along ``axis``.

    Offered for callers that hold unquantized operands; the score ops do not
    call it themselves. Returns ``(x_q, scale)`` such that
    ``x ~= x_q * expand_dims(scale, axis)``. ``scale`` is fp32 and strictly
    positive, which is what lets it commute with the ReLU downstream (see the
    module docstring).

    Differentiable as a straight-through estimator: the scale is held constant
    (``stop_gradient``), so a cotangent w.r.t. ``x_q`` flows back to ``x`` as
    ``g / scale`` without the spurious term the amax reduction would otherwise
    contribute, and the e4m3 rounding is not modelled.
    """
    dtype = fp8_dtype()
    fp8_max = float(jnp.finfo(dtype).max)
    x_f32 = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x_f32), axis=axis)
    # An all-zero row has no scale worth choosing; 1.0 keeps it strictly
    # positive (so the ReLU factoring stays valid) and round-trips to zero.
    scale = jax.lax.stop_gradient(jnp.where(amax > 0, amax / fp8_max, 1.0))
    x_q = jnp.clip(x_f32 / jnp.expand_dims(scale, axis), -fp8_max, fp8_max)
    return x_q.astype(dtype), scale


def _validate_scales(Hq, Hk, Sq, Ks):
    """Fill in unit scales for any omitted, and check the supplied ones.

    The scales index the non-contracted axes of their operand: ``Sq`` is
    ``Hq.shape[:-1]`` and ``Ks`` is ``Hk.shape[:-1]``.
    """
    if Sq is None:
        Sq = jnp.ones(Hq.shape[:-1], jnp.float32)
    elif Sq.shape != Hq.shape[:-1]:
        raise ValueError(
            f"Sq shape {Sq.shape} does not match Hq rows {Hq.shape[:-1]}"
        )
    if Ks is None:
        Ks = jnp.ones(Hk.shape[:-1], jnp.float32)
    elif Ks.shape != Hk.shape[:-1]:
        raise ValueError(
            f"Ks shape {Ks.shape} does not match Hk rows {Hk.shape[:-1]}"
        )
    return Sq.astype(jnp.float32), Ks.astype(jnp.float32)


def _score_reduce_autotune_configs():
    # The kernel is dominated by Hq reads (one (BLOCK_T, d_i) load per H
    # iteration). Bigger BLOCK_T ⇒ fewer T tiles ⇒ less total Hq traffic.
    # Bigger BLOCK_S ⇒ more Hk reuse but bigger per-CTA footprint.
    cfgs = [
        triton.Config({"BLOCK_T": 32,  "BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32,  "BLOCK_S": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 256, "BLOCK_S": 32},  num_warps=8, num_stages=2),
    ]
    cfgs += [
        triton.Config({"BLOCK_T": bt, "BLOCK_S": bs, "matrix_instr_nonkdim": nk_dim, "waves_per_eu": wpe}, num_warps=nw, num_stages=ns)
        for bt in (32, 64)
        for bs in (256, 512)
        for nk_dim in (16,)
        for wpe in (0, 2)   # 0 means let backend compiler decide
        for nw in (4,)
        for ns in (2,)
    ]
    return cfgs


@triton.autotune(configs=_score_reduce_autotune_configs(), key=["H", "d_i"])
@triton.jit
def _score_reduce_kernel(
    Hq_ptr,       # (B, oH, T_t, H, d_i) — produced by einsum("...tc,hci->...thi")
    Hk_ptr,       # (B, oH, T_s, d_i)
    W_o_ptr,      # (B, oH, T_t, H)
    Sq_ptr,       # (B, oH, T_t, H)  fp32 — Hq row scales (all-ones when bf16)
    Ks_ptr,       # (B, oH, T_s)     fp32 — Hk row scales (all-ones when bf16)
    SegQ_ptr,     # (B, T_t) int32 — query segment id   (unread when HAS_MASK=0)
    KeyQ_ptr,     # (B, T_t) int32 — query order key    (unread when HAS_MASK=0)
    SegK_ptr,     # (B, T_s) int32 — key segment id     (unread when HAS_MASK=0)
    KeyK_ptr,     # (B, T_s) int32 — key order key      (unread when HAS_MASK=0)
    OInit_ptr,    # (B, oH, T_t, T_s) MASK_FILL-prefilled, aliased to O_ptr when
                  # TRI_GRID=1; a (1, 1) dummy otherwise. Never dereferenced.
    O_ptr,        # (B, oH, T_t, T_s)
    B: tl.constexpr,
    oH: tl.constexpr,
    T_t: tl.constexpr,
    T_s: tl.constexpr,
    H: tl.constexpr,
    d_i: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_FILL: tl.constexpr,
    SKIP_TILES: tl.constexpr,
    TRI_GRID: tl.constexpr,
    # Triangular-grid decode constants. Computed on the host by
    # _tri_grid_params and injected per autotune config, so the launch size and
    # this decode cannot drift apart. All 1 (unused) when TRI_GRID=0.
    TRI_N_S: tl.constexpr,
    TRI_RATIO: tl.constexpr,
    TRI_CASE_A: tl.constexpr,
    TRI_I_STAR: tl.constexpr,
    TRI_COUNT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Compute one (BLOCK_T, BLOCK_S) tile of O for one (b, h_outer) slice.

    Grid order: (cdiv(T_s, BLOCK_S), cdiv(T_t, BLOCK_T), B * oH).

    S is the fastest-dispatching axis so consecutive CTAs share (B*oH, T)
    and vary only in S — they all read the same per-head Hq slab, hitting
    L2 instead of HBM. Hq layout is the natural einsum output
    (..., T, H, d_i); per-head loads are strided in T (stride H*d_i).

    Masking (``HAS_MASK``): a (t, s) pair is scored iff it shares a segment and
    the query's order key is at or after the key's. Every supported mask type is
    a choice of key made by the caller, so the kernel has one predicate and no
    mask-type branching. Masked slots get ``MASK_FILL`` -- a large negative
    value, not zero, because the logits are signed (the ReLU sits inside the
    H-reduction, ahead of the signed W_o weighting) and a downstream top-k must
    not prefer a masked slot to a valid negative one.

    A tile whose every pair is masked skips the Hk load and the whole H loop; it
    still stores, since the output slots are not otherwise written.

    How much that buys depends on the mask's *shape*, not just how much it masks
    (gfx950, BLOCK_T=32/BLOCK_S=128, T=4096, measured against the unmasked
    kernel):

      - masking at all costs ~14%: the metadata loads, the predicate, the select
      - a block-structured mask (padding) at 49% skipped tiles: 0.90x
      - a causal mask at the same 48% skipped tiles: 1.12x

    Same tile-skip fraction, very different payoff. With a block mask whole rows
    of the grid skip together, so slots free up in bulk; a causal mask
    interleaves skipped and working tiles along ``pid_s`` (the fast-dispatching
    axis), so co-resident CTAs finish at the working ones' pace and the freed
    slots are not reclaimed -- which is why the skip alone leaves a causal mask
    costing about as much as no mask at all.
    Set ``NVTE_INDEXER_TILE_SKIP=0`` to A/B the skip.

    ``TRI_GRID`` is the answer to that, for the causal family specifically: the
    grid collapses to 1-D over only the tiles that meet the causal region, so a
    fully-masked tile is never dispatched and there is nothing to interleave.
    The tile body below is unchanged -- only the decode differs:

        TRI_GRID=0:  (cdiv(T_s, BLOCK_S), cdiv(T_t, BLOCK_T), B * oH)
        TRI_GRID=1:  (n_tiles, 1, B * oH), pid inverted to (pid_t, pid_s)

    Because the skipped region is then never visited, ``O`` must already hold
    ``MASK_FILL`` there; the caller supplies that via ``OInit_ptr``, aliased to
    ``O_ptr``. Enabled only for causal / padding_causal at T_t == T_s, where
    every valid pair satisfies t >= s. Set ``NVTE_INDEXER_TRI_GRID=0`` to A/B it.
    """
    pid_bh = tl.program_id(2)

    if TRI_GRID:
        pid = tl.program_id(0)
        if pid >= TRI_COUNT:
            # Rectangular tail: these t-tiles meet all TRI_N_S s-tiles.
            rem = pid - TRI_COUNT
            pid_t = TRI_I_STAR + rem // TRI_N_S
            pid_s = rem % TRI_N_S
        else:
            # Triangular head. Largest n with TRI_RATIO*n*(n+1)/2 <= pid, via an
            # f32 sqrt plus a correction step -- the estimate only has to land
            # within one, and pid stays well inside f32's exact-integer range.
            n = ((tl.sqrt(8.0 * pid / TRI_RATIO + 1.0) - 1.0) * 0.5).to(tl.int32)
            n = tl.maximum(n, 0)
            while TRI_RATIO * (n + 1) * (n + 2) // 2 <= pid:
                n += 1
            while n > 0 and TRI_RATIO * n * (n + 1) // 2 > pid:
                n -= 1
            rem = pid - TRI_RATIO * n * (n + 1) // 2
            if TRI_CASE_A:
                # n indexes a group of TRI_RATIO t-tiles sharing a count of n+1.
                pid_t = n * TRI_RATIO + rem // (n + 1)
                pid_s = rem % (n + 1)
            else:
                # One t-tile per n, meeting (n+1)*TRI_RATIO s-tiles.
                pid_t = n
                pid_s = rem
    else:
        pid_s = tl.program_id(0)
        pid_t = tl.program_id(1)

    # int64 indexing — Hq alone has B*oH*T*H*d_i = 4.3 B elements at T=S=4096,
    # exceeds int32 range.
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rdi = tl.arange(0, d_i)

    rt_mask = rt < T_t
    rs_mask = rs < T_s

    hq_base = b * (oH * T_t * H * d_i) + h_outer * (T_t * H * d_i)
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    wo_base = b * (oH * T_t * H) + h_outer * (T_t * H)
    ks_base = b * (oH * T_s) + h_outer * T_s
    o_base = b * (oH * T_t * T_s) + h_outer * (T_t * T_s)

    # Segment metadata is (B, T) and broadcasts over the outer-head axis, so it
    # is indexed by b alone -- no h_outer term. The out-of-range fills are
    # distinct sentinels (-1 query, -2 key) that match neither each other nor a
    # real segment id, so a masked-off lane can never read as valid.
    if HAS_MASK:
        seg_q = tl.load(SegQ_ptr + b * T_t + rt, mask=rt_mask, other=-1)
        key_q = tl.load(KeyQ_ptr + b * T_t + rt, mask=rt_mask, other=0)
        seg_k = tl.load(SegK_ptr + b * T_s + rs, mask=rs_mask, other=-2)
        key_k = tl.load(KeyK_ptr + b * T_s + rs, mask=rs_mask, other=0)
        valid = (seg_q[:, None] == seg_k[None, :]) & (key_q[:, None] >= key_k[None, :])
        if SKIP_TILES:
            any_valid = tl.max(valid.to(tl.int32)) > 0
        else:
            any_valid = True
    else:
        any_valid = True

    acc = tl.zeros((BLOCK_T, BLOCK_S), dtype=tl.float32)

    if any_valid:
        # Load the (BLOCK_S, d_i) Hk slab once — it is loop-invariant over H.
        hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
        Hk_tile = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)
        Hk_T = tl.trans(Hk_tile)  # (d_i, BLOCK_S)

        # Hk row scales — also loop-invariant over H, applied once after the loop.
        ks = tl.load(Ks_ptr + ks_base + rs, mask=rs_mask, other=0.0)

        for h in range(H):
            hq_ptrs = (Hq_ptr + hq_base
                       + rt[:, None] * (H * d_i) + h * d_i + rdi[None, :])
            Hq_h = tl.load(hq_ptrs, mask=rt_mask[:, None], other=0.0)

            wo_ptrs = W_o_ptr + wo_base + rt * H + h
            w_h = tl.load(wo_ptrs, mask=rt_mask, other=0.0).to(tl.float32)
            # Fold the Hq row scale into the output weight: relu is
            # positive-homogeneous, so sq can be pulled out through it.
            sq_h = tl.load(Sq_ptr + wo_base + rt * H + h, mask=rt_mask, other=0.0)

            score = tl.dot(Hq_h, Hk_T)
            score = tl.maximum(score, 0.0)
            acc += score * (w_h * sq_h)[:, None]

        acc = acc * ks[None, :]

    if HAS_MASK:
        acc = tl.where(valid, acc, MASK_FILL)

    o_ptrs = O_ptr + o_base + rt[:, None] * T_s + rs[None, :]
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty),
             mask=rt_mask[:, None] & rs_mask[None, :])


_score_reduce_p = extend_core.Primitive("te_indexer_score_reduce_triton")
_score_reduce_p.multiple_results = True


@_score_reduce_p.def_abstract_eval
def _score_reduce_abstract(Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK, OInit, *,
                           out_dtype, has_mask, tri_grid):
    del W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK, has_mask, tri_grid
    del OInit  # carried for input_output_aliases only
    # Hq layout: (B, oH, T_t, H, d_i)
    B, oH, T_t, _H, _d_i = Hq.shape
    T_s = Hk.shape[2]
    return [core.ShapedArray((B, oH, T_t, T_s), out_dtype)]


_score_reduce_p.def_impl(functools.partial(xla.apply_primitive, _score_reduce_p))


def _score_reduce_lowering(ctx, Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK, OInit, *,
                           out_dtype, has_mask, tri_grid):
    del out_dtype
    Hq_aval = ctx.avals_in[0]
    Hk_aval = ctx.avals_in[1]
    B, oH, T_t, H, d_i = Hq_aval.shape
    T_s = Hk_aval.shape[2]

    # Fill for masked slots, in the *output* dtype's range. Finite rather than
    # -inf so a consumer that does arithmetic on the logits cannot turn a masked
    # slot into a NaN; still below every representable real logit, which is what
    # a downstream top-k needs.
    mask_fill = float(jnp.finfo(ctx.avals_out[0].dtype).min)

    def grid_fn(merged_kwargs):
        # Not .get() with a default: under TRI_GRID a wrong block size would
        # silently under-cover the output rather than merely over-launch. Every
        # config sets both keys.
        bt = merged_kwargs["BLOCK_T"]
        bs = merged_kwargs["BLOCK_S"]
        if tri_grid:
            return (_tri_grid_params(T_t, T_s, bt, bs)["n_tiles"], 1, B * oH)
        # S as grid_x (fastest-dispatching) so per-(B*oH, T-tile) S workgroups
        # cluster in time and hit L2 on the shared Hq slab.
        return (triton.cdiv(T_s, bs), triton.cdiv(T_t, bt), B * oH)

    # Under TRI_GRID the masked region is never visited, so the prefilled buffer
    # *is* the output. Otherwise OInit is an unused (1, 1) dummy.
    aliases = {9: 0} if tri_grid else None

    saved_configs = _score_reduce_kernel.configs
    configs = saved_configs
    if _autotune_disabled():
        configs = configs[:1]

    # The decode constants depend on BLOCK_T/BLOCK_S, so they must ride along in
    # each config's kwargs rather than in the shared constexprs dict. They are
    # attached unconditionally (trivially when not tri_grid) because the two
    # launch paths inside triton_call_lowering merge the two dicts in OPPOSITE
    # orders -- the autotuned path lets config kwargs win, the non-autotuned
    # fallback lets constexprs win -- so a value that appears in both would mean
    # something different depending on the JAX version. Deriving them from the
    # same helper that sizes the grid is what keeps launch and decode in step.
    tri_configs = []
    for cfg in configs:
        if tri_grid:
            p = _tri_grid_params(T_t, T_s, cfg.kwargs["BLOCK_T"], cfg.kwargs["BLOCK_S"])
        else:
            p = {"n_s": 1, "ratio": 1, "case_a": True, "i_star": 0, "tri_count": 0}
        tri_cfg = copy.copy(cfg)
        tri_cfg.kwargs = {
            **cfg.kwargs,
            "TRI_N_S": p["n_s"],
            "TRI_RATIO": p["ratio"],
            "TRI_CASE_A": p["case_a"],
            "TRI_I_STAR": p["i_star"],
            "TRI_COUNT": p["tri_count"],
        }
        tri_configs.append(tri_cfg)
    _score_reduce_kernel.configs = tri_configs
    try:
        return triton_call_lowering(
            ctx,
            _score_reduce_kernel,
            Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK, OInit,
            grid=grid_fn,
            input_output_aliases=aliases,
            constexprs={
                "B": B,
                "oH": oH,
                "T_t": T_t,
                "T_s": T_s,
                "H": H,
                "d_i": d_i,
                "HAS_MASK": has_mask,
                "MASK_FILL": mask_fill,
                "SKIP_TILES": _tile_skip_enabled(),
                "TRI_GRID": tri_grid,
                # TRI_N_S / TRI_RATIO / TRI_CASE_A / TRI_I_STAR / TRI_COUNT are
                # deliberately absent here -- they are per-config, see above.
            },
        )
    finally:
        _score_reduce_kernel.configs = saved_configs


mlir.register_lowering(_score_reduce_p, _score_reduce_lowering, platform="rocm")
mlir.register_lowering(_score_reduce_p, _score_reduce_lowering, platform="cuda")


# --- Chunked score-tile kernel for hybrid bwd --------------------------------
#
# Produces dscores_chunk[B, oH, T, H_CHUNK, T_s] and dW_o_chunk[B, oH, T, H_CHUNK]
# for ONE h-chunk. Caller loops over H/H_CHUNK chunks and feeds dscores_chunk
# to hipBLASLt einsums for dHq/dHk reductions. Bounds peak materialization to
# H/H_CHUNK fraction of the full (B, oH, T, H, T_s) score tensor.
#
# Fuses score recompute + relu + mask + dO*W_o broadcast in registers --
# nothing of size (B, oH, T, H, T_s) ever lands in HBM at full size. dW_o is
# reduced inline (sum_s of h_relu * dO) so h_relu also never materializes.


_HBWD_BLOCK_T = 64


def _score_dscores_chunk_autotune_configs():
    # matrix_instr_nonkdim is pinned to 16. On Triton 3.7.0 / gfx950 the
    # H_CHUNK-unrolled score matmul crashes the compiler (uncatchable
    # std::bad_alloc / heap corruption the autotuner cannot skip) for both
    # nonkdim=32 and the backend-default nonkdim (0); only 16x16 MFMA tiles
    # compile. num_stages is pinned to 1: pipelining the s-loop crashes LLVM
    # codegen on the same toolchain.
    cfgs = []
    cfgs += [
        triton.Config(
            {"BLOCK_T": bt, "BLOCK_S": bs,
             "matrix_instr_nonkdim": 16, "waves_per_eu": wpe},
            num_warps=nw, num_stages=1)
        for bt in (32, 64, 128)
        for bs in (128, 256)
        for wpe in (0, 2)
        for nw in (4, 8)
    ]
    # larger BLOCK_S for long T_s
    cfgs += [
        triton.Config({"BLOCK_T": bt, "BLOCK_S": 512, "matrix_instr_nonkdim": 16},
                      num_warps=4, num_stages=1)
        for bt in (32, 64)
    ]
    return cfgs


@triton.autotune(configs=_score_dscores_chunk_autotune_configs(),
                 key=["T", "T_s", "H_CHUNK", "d_i"])
@triton.jit
def _score_dscores_chunk_kernel(
    Hq_chunk_ptr,        # input  (B, oH, T,   H_CHUNK, d_i) bf16 or e4m3
    Hk_ptr,              # input  (B, oH, T_s, d_i)         bf16 or e4m3
    W_o_chunk_ptr,       # input  (B, oH, T,   H_CHUNK)     bf16
    dO_ptr,              # input  (B, oH, T,   T_s)         fp32
    Sq_chunk_ptr,        # input  (B, oH, T,   H_CHUNK)     fp32 — Hq row scales
    Ks_ptr,              # input  (B, oH, T_s)              fp32 — Hk row scales
    SegQ_ptr,            # input  (B, T)   int32 (unread when HAS_MASK=0)
    KeyQ_ptr,            # input  (B, T)   int32 (unread when HAS_MASK=0)
    SegK_ptr,            # input  (B, T_s) int32 (unread when HAS_MASK=0)
    KeyK_ptr,            # input  (B, T_s) int32 (unread when HAS_MASK=0)
    dscores_chunk_ptr,   # output (B, oH, T,   H_CHUNK, T_s) bf16
    dWo_chunk_ptr,       # output (B, oH, T,   H_CHUNK)     bf16
    B: tl.constexpr,
    oH: tl.constexpr,
    T: tl.constexpr,
    T_s: tl.constexpr,
    H_CHUNK: tl.constexpr,
    d_i: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """One CTA handles (T_tile, all H_CHUNK heads) for one (b, h_outer).

    Grid: (cdiv(T, BLOCK_T), B * oH). For each s-chunk we load dO_chunk and
    Hk_chunk ONCE and reuse them across every head in the chunk -- the key
    saving vs the original (which spun a separate CTA per head, each re-reading
    dO/Hk). dW_o is reduced in registers (sum over s) per head, so h_relu never
    lands in HBM.

    The mask (same predicate as the forward kernel) is ANDed into the ReLU mask,
    which zeroes both outputs at a masked pair in one term: dscores directly,
    and dW_o because the h_relu it reduces is already zero there. No fill value
    is involved -- these are cotangents, and a masked pair contributes nothing.
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // oH).to(tl.int64)
    h_outer = (pid_bh % oH).to(tl.int64)

    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rdi = tl.arange(0, d_i)
    rhc = tl.arange(0, H_CHUNK)
    rt_mask = rt < T

    hq_base = b * (oH * T * H_CHUNK * d_i) + h_outer * (T * H_CHUNK * d_i)
    hk_base = b * (oH * T_s * d_i) + h_outer * (T_s * d_i)
    wo_base = b * (oH * T * H_CHUNK) + h_outer * (T * H_CHUNK)
    do_base = b * (oH * T * T_s) + h_outer * (T * T_s)
    ks_base = b * (oH * T_s) + h_outer * T_s
    ds_base = b * (oH * T * H_CHUNK * T_s) + h_outer * (T * H_CHUNK * T_s)

    # Query-side mask metadata depends only on rt, so it is loaded once here
    # rather than per s-chunk. Sentinels match the forward kernel's.
    if HAS_MASK:
        seg_q = tl.load(SegQ_ptr + b * T + rt, mask=rt_mask, other=-1)
        key_q = tl.load(KeyQ_ptr + b * T + rt, mask=rt_mask, other=0)

    # Per-head dW_o accumulators packed as (BLOCK_T, H_CHUNK), reduced over s.
    dWo_acc = tl.zeros((BLOCK_T, H_CHUNK), dtype=tl.float32)

    for s_start in range(0, T_s, BLOCK_S):
        rs = s_start + tl.arange(0, BLOCK_S)
        rs_mask = rs < T_s

        # Shared across all H_CHUNK heads below, like Hk_chunk / dO_chunk.
        if HAS_MASK:
            seg_k = tl.load(SegK_ptr + b * T_s + rs, mask=rs_mask, other=-2)
            key_k = tl.load(KeyK_ptr + b * T_s + rs, mask=rs_mask, other=0)
            valid = (seg_q[:, None] == seg_k[None, :]) & (key_q[:, None] >= key_k[None, :])

        # Load Hk[..., s_chunk, :] and dO[..., t_tile, s_chunk] ONCE per s-chunk
        # -- shared across all H_CHUNK heads below.
        hk_ptrs = Hk_ptr + hk_base + rs[:, None] * d_i + rdi[None, :]
        Hk_chunk = tl.load(hk_ptrs, mask=rs_mask[:, None], other=0.0)
        Hk_T = tl.trans(Hk_chunk)  # (d_i, BLOCK_S)

        ks_chunk = tl.load(Ks_ptr + ks_base + rs, mask=rs_mask, other=0.0)

        do_ptrs = dO_ptr + do_base + rt[:, None] * T_s + rs[None, :]
        dO_chunk = tl.load(
            do_ptrs, mask=rt_mask[:, None] & rs_mask[None, :], other=0.0,
        )

        for h in tl.static_range(H_CHUNK):
            # Hq/w for head h (small, L2-resident across s-chunks).
            Hq_h = tl.load(
                Hq_chunk_ptr + hq_base + rt[:, None] * (H_CHUNK * d_i)
                + h * d_i + rdi[None, :],
                mask=rt_mask[:, None], other=0.0,
            )
            w_h = tl.load(
                W_o_chunk_ptr + wo_base + rt * H_CHUNK + h,
                mask=rt_mask, other=0.0,
            ).to(tl.float32)
            sq_h = tl.load(
                Sq_chunk_ptr + wo_base + rt * H_CHUNK + h,
                mask=rt_mask, other=0.0,
            )

            # Scores here are the *quantized* dot q_hat.k_hat; the true score is
            # sq_h * ks * scores. Both scales are positive, so the relu mask is
            # the same either way and only the magnitude below needs rescaling.
            scores = tl.dot(Hq_h, Hk_T)  # (BLOCK_T, BLOCK_S)
            relu_mask = scores > 0
            if HAS_MASK:
                relu_mask = relu_mask & valid
            h_relu = tl.where(relu_mask, scores, 0.0) * ks_chunk[None, :]

            # dW_o[..., h] += sum_s (h_relu * dO); accumulate into column h.
            dwo_h = tl.sum(h_relu * dO_chunk, axis=1) * sq_h  # (BLOCK_T,)
            dWo_acc += tl.where(rhc[None, :] == h, dwo_h[:, None], 0.0)

            # dscores[..., h, s] = relu_mask * (dO * W_o)
            dscores = tl.where(relu_mask, dO_chunk * w_h[:, None], 0.0)
            ds_ptrs = (dscores_chunk_ptr + ds_base
                       + rt[:, None] * (H_CHUNK * T_s) + h * T_s + rs[None, :])
            tl.store(
                ds_ptrs, dscores.to(dscores_chunk_ptr.dtype.element_ty),
                mask=rt_mask[:, None] & rs_mask[None, :],
            )

    # Store dW_o[..., t_tile, :] for all heads.
    dwo_out_ptrs = dWo_chunk_ptr + wo_base + rt[:, None] * H_CHUNK + rhc[None, :]
    tl.store(
        dwo_out_ptrs, dWo_acc.to(dWo_chunk_ptr.dtype.element_ty),
        mask=rt_mask[:, None],
    )


_score_dscores_chunk_p = extend_core.Primitive("te_indexer_score_dscores_chunk")
_score_dscores_chunk_p.multiple_results = True


@_score_dscores_chunk_p.def_abstract_eval
def _score_dscores_chunk_abstract(Hq_chunk, Hk, W_o_chunk, dO, Sq_chunk, Ks,
                                  SegQ, KeyQ, SegK, KeyK, *, has_mask):
    del Hk, Sq_chunk, Ks, SegQ, KeyQ, SegK, KeyK, has_mask
    B, oH, T, H_CHUNK, _ = Hq_chunk.shape
    T_s = dO.shape[-1]
    # Both outputs are cotangents, so they follow W_o's (bf16) dtype -- Hq_chunk
    # is e4m3 on the fp8 path and would silently narrow them.
    return [
        core.ShapedArray((B, oH, T, H_CHUNK, T_s), W_o_chunk.dtype),  # dscores
        core.ShapedArray((B, oH, T, H_CHUNK), W_o_chunk.dtype),       # dW_o
    ]


_score_dscores_chunk_p.def_impl(
    functools.partial(xla.apply_primitive, _score_dscores_chunk_p)
)


def _score_dscores_chunk_lowering(ctx, Hq_chunk, Hk, W_o_chunk, dO, Sq_chunk, Ks,
                                  SegQ, KeyQ, SegK, KeyK, *, has_mask):
    Hq_aval = ctx.avals_in[0]
    dO_aval = ctx.avals_in[3]
    B, oH, T, H_CHUNK, d_i = Hq_aval.shape
    T_s = dO_aval.shape[-1]

    # Grid: (T-tiles, B*oH) -- one CTA per (T_tile, b, h_outer) covers all
    # H_CHUNK heads (dO/Hk shared across heads). Depends on autotuned BLOCK_T.
    def grid_fn(merged_kwargs):
        bt = merged_kwargs.get("BLOCK_T", _HBWD_BLOCK_T)
        return ((T + bt - 1) // bt, B * oH)

    saved_configs = _score_dscores_chunk_kernel.configs
    if _autotune_disabled():
        _score_dscores_chunk_kernel.configs = saved_configs[:1]
    try:
        return triton_call_lowering(
            ctx,
            _score_dscores_chunk_kernel,
            Hq_chunk, Hk, W_o_chunk, dO, Sq_chunk, Ks, SegQ, KeyQ, SegK, KeyK,
            grid=grid_fn,
            constexprs={
                "B": B, "oH": oH, "T": T, "T_s": T_s,
                "H_CHUNK": H_CHUNK, "d_i": d_i,
                "HAS_MASK": has_mask,
            },
        )
    finally:
        _score_dscores_chunk_kernel.configs = saved_configs


mlir.register_lowering(_score_dscores_chunk_p, _score_dscores_chunk_lowering, platform="rocm")
mlir.register_lowering(_score_dscores_chunk_p, _score_dscores_chunk_lowering, platform="cuda")


# --- Public score_reduce_triton with custom_vjp ------------------------------


def _score_reduce_bind(Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK,
                       out_dtype, has_mask, tri_grid):
    """Bind the forward primitive, supplying the aliased output buffer.

    Under ``tri_grid`` the kernel never visits the masked region, so the buffer
    must arrive already holding MASK_FILL and is aliased to the output. The
    rectangular path writes every slot itself and gets a (1, 1) dummy, matching
    how the mask operands are handled when ``has_mask`` is False.
    """
    if tri_grid:
        B, oH, T_t = Hq.shape[0], Hq.shape[1], Hq.shape[2]
        T_s = Hk.shape[2]
        OInit = jnp.full((B, oH, T_t, T_s), jnp.finfo(out_dtype).min, out_dtype)
    else:
        OInit = jnp.zeros((1, 1), out_dtype)
    return _score_reduce_p.bind(Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK, OInit,
                                out_dtype=out_dtype, has_mask=has_mask,
                                tri_grid=tri_grid)[0]


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _score_reduce_with_vjp(Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK,
                           out_dtype, has_mask, tri_grid):
    return _score_reduce_bind(Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK,
                              out_dtype, has_mask, tri_grid)


def _score_reduce_fwd(Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK,
                      out_dtype, has_mask, tri_grid):
    out = _score_reduce_bind(Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK,
                             out_dtype, has_mask, tri_grid)
    return out, (Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK)


_BWD_H_CHUNK = 8  # peak (B, oH, T, H_CHUNK, T_s) tile -- bounds materialization


def _score_reduce_bwd(out_dtype, has_mask, tri_grid, residuals, dO):
    # tri_grid is a forward-only launch detail: the backward kernel walks S as an
    # in-kernel loop rather than a grid axis, so it has no tile grid to remap.
    del out_dtype, tri_grid
    Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK = residuals
    B, oH, T, H, d_i = Hq.shape
    # The score recompute inside the Triton kernel consumes Hq / Hk in their
    # stored precision (with Sq / Ks), but the two dscores reductions below are
    # hipBLASLt GEMMs and need dequantized operands. W_o is the bf16 anchor for
    # that intermediate precision -- Hq / Hk may be e4m3.
    compute_dtype = W_o.dtype
    fp8_operands = is_fp8(Hq.dtype)

    def _dequant(x, scale):
        if not fp8_operands:
            return x
        return (x.astype(jnp.float32) * scale[..., None]).astype(compute_dtype)

    Hk_deq = _dequant(Hk, Ks)

    # Hybrid scheme with bounded materialization:
    #   For each h-chunk of size H_CHUNK (driven by lax.scan, NOT Python
    #   unroll, so intermediates are freed between iterations):
    #     1. Triton kernel fuses (score recompute + relu + mask + dO*W_o
    #        broadcast) and writes dscores_chunk[B,oH,T,H_CHUNK,T_s] to HBM.
    #        h_relu is consumed in-register to also produce dWo_chunk
    #        without ever materializing the (B,oH,T,H,T_s) h_relu tensor.
    #     2. hipBLASLt einsums on dscores_chunk give dHq_chunk and a partial
    #        dHk contribution.
    # Peak HBM intermediate stays at H_CHUNK/H fraction of the full score.
    if H % _BWD_H_CHUNK == 0:
        H_CHUNK = _BWD_H_CHUNK
    else:
        H_CHUNK = 1
        for c in (4, 2):
            if H % c == 0:
                H_CHUNK = c
                break
    n_chunks = H // H_CHUNK

    Hq_r = Hq.reshape(B, oH, T, n_chunks, H_CHUNK, d_i)
    Wo_r = W_o.reshape(B, oH, T, n_chunks, H_CHUNK)
    Sq_r = Sq.reshape(B, oH, T, n_chunks, H_CHUNK)
    # Move chunk axis to leading for scan over axis 0.
    Hq_s = jnp.moveaxis(Hq_r, -3, 0)   # (n_chunks, B, oH, T, H_CHUNK, d_i)
    Wo_s = jnp.moveaxis(Wo_r, -2, 0)   # (n_chunks, B, oH, T, H_CHUNK)
    Sq_s = jnp.moveaxis(Sq_r, -2, 0)   # (n_chunks, B, oH, T, H_CHUNK)

    def step(dHk_acc, chunk):
        Hq_c, Wo_c, Sq_c = chunk
        # Triton: dscores_chunk + dWo_chunk; no full (B,oH,T,H,T_s) tensor
        # ever exists in HBM.
        dscores_c, dWo_c = _score_dscores_chunk_p.bind(
            Hq_c, Hk, Wo_c, dO, Sq_c, Ks, SegQ, KeyQ, SegK, KeyK,
            has_mask=has_mask,
        )
        # Dequantize only the current chunk, so peak stays at H_CHUNK/H of Hq.
        Hq_c_deq = _dequant(Hq_c, Sq_c)
        dHq_c = jnp.einsum("...ths,...si->...thi", dscores_c, Hk_deq)
        dHk_c = jnp.einsum("...ths,...thi->...si", dscores_c, Hq_c_deq)
        new_dHk_acc = dHk_acc + dHk_c.astype(jnp.float32)
        return new_dHk_acc, (dHq_c, dWo_c)

    init = jnp.zeros(Hk.shape, dtype=jnp.float32)
    dHk_acc, (dHq_chunks, dWo_chunks) = jax.lax.scan(
        step, init, (Hq_s, Wo_s, Sq_s),
    )
    # dHq_chunks: (n_chunks, B, oH, T, H_CHUNK, d_i)
    # dWo_chunks: (n_chunks, B, oH, T, H_CHUNK)
    dHq = jnp.moveaxis(dHq_chunks, 0, -3).reshape(B, oH, T, H, d_i)
    dWo = jnp.moveaxis(dWo_chunks, 0, -2).reshape(B, oH, T, H)
    dHk = dHk_acc

    # The einsums above produce cotangents w.r.t. the *dequantized* operands.
    # Chain through Hq = Hq_q * sq to get the cotangent w.r.t. the operand we
    # were actually handed. When the operands are e4m3 this also renormalizes
    # the gradient into their range, instead of storing a raw unscaled fp8
    # value. With unit scales it is an exact no-op.
    dHq = (dHq.astype(jnp.float32) * Sq[..., None]).astype(Hq.dtype)
    dHk = (dHk.astype(jnp.float32) * Ks[..., None]).astype(Hk.dtype)

    # Sq / Ks are quantization metadata, held constant by every scaling recipe
    # (and by quantize_e4m3's stop_gradient), so they take a zero cotangent. The
    # four mask operands are integer arrays; JAX takes None as their cotangent
    # (same convention as permutation.py's row_id_map / pad_offsets).
    return (
        dHq,
        dHk,
        dWo.astype(W_o.dtype),
        jnp.zeros_like(Sq),
        jnp.zeros_like(Ks),
        None,
        None,
        None,
        None,
    )


_score_reduce_with_vjp.defvjp(_score_reduce_fwd, _score_reduce_bwd)


def _validate_mask(mask, B, T_t, T_s):
    """Check a ``(seg_q, key_q, seg_k, key_k)`` mask, or build unused dummies.

    Returns ``(arrays, has_mask)``. When no mask is given the four operands are
    one-element int32 placeholders: the kernel's ``HAS_MASK`` constexpr gates
    every dereference, so nothing reads them, and passing full-size arrays would
    cost HBM for data the kernel never touches.
    """
    if mask is None:
        dummy = jnp.zeros((1, 1), jnp.int32)
        return (dummy, dummy, dummy, dummy), False

    if len(mask) != 4:
        raise ValueError(
            "mask must be a 4-tuple (seg_q, key_q, seg_k, key_k); "
            f"got {len(mask)} entries"
        )
    names = ("seg_q", "key_q", "seg_k", "key_k")
    expected = ((B, T_t), (B, T_t), (B, T_s), (B, T_s))
    out = []
    for name, arr, shape in zip(names, mask, expected):
        if arr is None:
            raise ValueError(f"mask entry {name} is None; pass mask=None to disable masking")
        if arr.shape != shape:
            raise ValueError(f"mask entry {name} has shape {arr.shape}, expected {shape}")
        out.append(arr.astype(jnp.int32))
    return tuple(out), True


def score_reduce_triton(Hq, Hk, W_o, *, Sq=None, Ks=None, mask=None, out_dtype=None,
                        mask_is_causal=False):
    """Triton fused score-matmul + relu + per-(t, h) weighted H-reduction.

    Replaces the pattern:

        scores = relu(jnp.einsum("...thi,...si->...ths", Hq, Hk))   # never write
        O      = jnp.einsum("...ths,...th->...ts", scores, W_o)

    with a single kernel that holds the per-head score tile in registers,
    avoiding the (B, oH, T, H, S) HBM round-trip an einsum+XLA chain pays.

    Differentiable via two backward kernels (FlashAttention-style: residuals
    are just (Hq, Hk, W_o); the (T, H, S) score tensor is recomputed inside
    backward, never materialized).

    Args:
        Hq:  (B, oH, T_t, H, d_i) — e4m3 or bf16, consumed as given.
        Hk:  (B, oH, T_s, d_i)    — same dtype as Hq.
        W_o: (B, oH, T_t, H)
        Sq:  (B, oH, T_t, H) fp32 row scales for Hq, or None for unit scales.
        Ks:  (B, oH, T_s) fp32 row scales for Hk, or None for unit scales.
        mask: optional ``(seg_q, key_q, seg_k, key_k)`` of int32 arrays shaped
            (B, T_t), (B, T_t), (B, T_s), (B, T_s). A (t, s) pair is scored iff
            ``seg_q[t] == seg_k[s] and key_q[t] >= key_k[s]``; every other slot
            gets ``finfo(out_dtype).min``. The metadata has no outer-head axis —
            it broadcasts over ``oH``. Each supported attention mask is a choice
            of key (causal: position; padding: a constant), so the kernel needs
            no mask-type switch; see
            ``sparse_attention.indexer._mask_keys`` for the mapping.
        out_dtype: defaults to W_o.dtype.
        mask_is_causal: caller's promise that every valid pair satisfies
            ``t >= s`` -- true for the causal family (``causal``,
            ``padding_causal``), false for ``padding``, whose keys are constant
            and whose valid set is block-diagonal rather than triangular. Purely
            a launch hint: it selects the triangular grid, which skips
            dispatching the fully-masked half instead of dispatching and
            discarding it. Ignored unless a mask is supplied and T_t == T_s.

    Returns:
        O: (B, oH, T_t, T_s)
    """
    if Hq.ndim != 5:
        raise ValueError(
            f"Hq must be rank-5 (B, oH, T_t, H, d_i); got shape {Hq.shape}"
        )
    if Hk.ndim != 4:
        raise ValueError(
            f"Hk must be rank-4 (B, oH, T_s, d_i); got shape {Hk.shape}"
        )
    if W_o.ndim != 4:
        raise ValueError(
            f"W_o must be rank-4 (B, oH, T_t, H); got shape {W_o.shape}"
        )

    B, oH, T_t, H, d_i = Hq.shape
    Bk, oHk, T_s, d_i_k = Hk.shape
    Bw, oHw, T_t_w, H_w = W_o.shape
    if (Bk, oHk) != (B, oH):
        raise ValueError(
            f"(B, oH) mismatch: Hq has {(B, oH)}, Hk has {(Bk, oHk)}"
        )
    if d_i != d_i_k:
        raise ValueError(f"d_i mismatch: Hq has {d_i}, Hk has {d_i_k}")
    if (Bw, oHw, T_t_w, H_w) != (B, oH, T_t, H):
        raise ValueError(
            f"W_o shape {W_o.shape} does not match expected "
            f"(B={B}, oH={oH}, T_t={T_t}, H={H})"
        )
    # Both operands feed one tl.dot, which specializes on a single input dtype.
    if Hq.dtype != Hk.dtype:
        raise ValueError(
            f"Hq and Hk must share a dtype; got {Hq.dtype} and {Hk.dtype}"
        )
    Sq, Ks = _validate_scales(Hq, Hk, Sq, Ks)
    (SegQ, KeyQ, SegK, KeyK), has_mask = _validate_mask(mask, B, T_t, T_s)

    if out_dtype is None:
        # Not Hq.dtype: that would make an e4m3 operand yield an e4m3 output.
        out_dtype = W_o.dtype

    # T_t == T_s is what lets the tile count saturate cleanly at n_s, which the
    # kernel's closed-form tile-id inverse relies on.
    tri_eligible = bool(mask_is_causal) and has_mask and T_t == T_s
    override = _tri_grid_override()
    tri_grid = tri_eligible and (T_t >= _TRI_GRID_MIN_T if override is None else override)

    return _score_reduce_with_vjp(
        Hq, Hk, W_o, Sq, Ks, SegQ, KeyQ, SegK, KeyK,
        jnp.dtype(out_dtype), has_mask, tri_grid,
    )
