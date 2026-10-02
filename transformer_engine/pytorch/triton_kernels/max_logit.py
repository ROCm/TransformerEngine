# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Per-head max attention logit (for MuonClip / QK-clip) as a standalone Triton kernel.

max_logit[h] = max over (b, i, unmasked j) of scale * q[b, i, h] . k[b, j, h // (H / H_kv)]

Only Q.K^T is recomputed: no softmax, no V, no score matrix is stored. Each program writes
one value per (b, h, q_block, k_split), which is then reduced with torch.amax (deterministic,
no atomics). Intended to run next to a fused attention forward that does not emit the max.
"""

import contextlib
import os
from typing import Optional

import torch
import triton
import triton.language as tl

SUPPORTED_MASK_TYPES = ("no_mask", "causal", "causal_bottom_right")
SUPPORTED_QKV_FORMATS = ("bshd", "sbhd")

# Fixed launch configs, tuned on MI355X (gfx950); no autotune: training shapes change often and
# tuning stalls a step. The default puts 128 q rows on each wave (BLOCK_M / num_warps): twice the
# MFMA work per K tile read from LDS compared to 64 rows.
# QSPLIT: q is loaded as QSPLIT head-dim slices (one dot each per K block), which divides the LDS
#   used to stage q into registers, so two programs still fit per CU.
# ASYNC_COPY: compile with direct-to-LDS K loads into a ring of num_stages LDS buffers, one
#   barrier per K block. This is the double buffering of K, done by Triton's stream pipeliner.
# GQA_STACK: max number of q heads sharing a K head that one program stacks along M (causal only:
#   fewer rows per head means less wasted work on the masked diagonal).
# SPLIT: number of K chunks per q block; None picks it from the grid size.
_DEFAULT_CONFIG = dict(
    BLOCK_M=512,
    BLOCK_N=32,
    QSPLIT=2,
    num_warps=4,
    num_stages=3,
    waves_per_eu=2,
    matrix_instr_nonkdim=16,
    ASYNC_COPY=True,
    GQA_STACK=4,
    SPLIT=None,
)
# head_dim > 128: 64 q rows per wave, or q no longer fits in registers.
_LARGE_D_CONFIG = dict(BLOCK_M=256)
# Causal without GQA stacking, moderate sizes: half the rows per program (same rows per wave)
# wastes less work on the masked diagonal and balances the uneven q blocks better.
_CAUSAL_CONFIG = dict(BLOCK_M=256, num_warps=2)
# Few query rows in total (a few programs per CU, the longest one sets the runtime): fewer rows
# per wave.
_FEW_ROWS_CONFIG = dict(BLOCK_M=256, BLOCK_N=64, QSPLIT=1)
_TINY_CONFIG = dict(BLOCK_M=128, BLOCK_N=64, QSPLIT=1)
# Other GPUs (not tuned): the original config, without async copies.
_FALLBACK_CONFIG = dict(
    _DEFAULT_CONFIG, BLOCK_M=256, BLOCK_N=64, QSPLIT=1, num_stages=2, ASYNC_COPY=False
)
# Split K while the grid has fewer waves than this per CU (2 per SIMD are resident).
_MIN_WAVES_PER_CU = 8
_MIN_SPLIT_KEYS = 512
_ASYNC_COPY_ENV = "TRITON_HIP_USE_ASYNC_COPY"
_UNSET = object()
_DEVICE_INFO = {}


def _device_info(device: torch.device) -> tuple:
    """(number of CUs, whether the device is gfx950), cached per device."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _DEVICE_INFO:
        props = torch.cuda.get_device_properties(idx)
        arch = getattr(props, "gcnArchName", "")
        _DEVICE_INFO[idx] = (props.multi_processor_count, arch.split(":")[0] == "gfx950")
    return _DEVICE_INFO[idx]


def _pick_config(
    head_dim: int, rows: int, seqlen_k: int, is_causal: bool, stacked: bool, gfx950: bool
) -> dict:
    """rows: query rows over all batches and heads; stacked: GQA heads are stacked per program."""
    if not gfx950:
        return dict(_FALLBACK_CONFIG)
    cfg = dict(_DEFAULT_CONFIG)
    if head_dim > 128:
        cfg.update(_LARGE_D_CONFIG)
    elif rows * seqlen_k <= 2**25 or (is_causal and not stacked and rows <= 2**17):
        cfg.update(_TINY_CONFIG)
    elif is_causal and rows <= 2**18:
        cfg.update(_FEW_ROWS_CONFIG)
    elif is_causal and not stacked and rows * head_dim <= 2**27 and seqlen_k * head_dim <= 2**19:
        cfg.update(_CAUSAL_CONFIG)
    if head_dim <= 32:
        cfg["QSPLIT"] = 1
    return cfg


@contextlib.contextmanager
def _async_copy(enable: bool):
    """Set triton.knobs.amd.use_async_copy around the launch (it is only read when compiling).

    The knob is mirrored into an env var that is part of the on-disk cache key, and the kernel has
    an ASYNC_COPY constexpr so the in-memory cache is keyed on it too. Both are restored on exit,
    so other kernels are not affected.
    """
    knobs = triton.knobs.amd
    saved = (knobs.__dict__.get("use_async_copy", _UNSET), os.environ.get(_ASYNC_COPY_ENV))
    knobs.use_async_copy = enable
    try:
        yield
    finally:
        if saved[0] is _UNSET:
            del knobs.use_async_copy
        else:
            knobs.use_async_copy = saved[0]
        if saved[1] is None:
            os.environ.pop(_ASYNC_COPY_ENV, None)
        else:
            os.environ[_ASYNC_COPY_ENV] = saved[1]


@triton.jit
def _load_kt(k_ptrs, kmask, offs_d, D_LO: tl.constexpr, CHUNK: tl.constexpr, D: tl.constexpr):
    """Load the [CHUNK, BLOCK_N] slice of K^T covering head dims [D_LO, D_LO + CHUNK)."""
    if D_LO + CHUNK > D:
        kmask = kmask & (offs_d[:, None] + D_LO < D)
    return tl.load(k_ptrs + D_LO, mask=kmask, other=0.0)


@triton.jit
def _max_logit_k_blocks(
    q0,
    q1,
    k_base,
    stride_ks,
    nb_lo,
    nb_flo,
    nb_fhi,
    nb_hi,
    hi_row,
    lo_row,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    QSPLIT: tl.constexpr,
    GS: tl.constexpr,
    HAS_LEFT: tl.constexpr,
):
    """Max of q.k^T over K blocks [nb_lo, nb_hi); blocks outside [nb_flo, nb_fhi) are masked.

    One loop over all blocks, so the pipelined K prefetch also runs across the masked and
    unmasked blocks. Key j is allowed for row r iff lo_row[r] <= j <= hi_row[r].
    Returns the max over all rows (GS == 1) or over each head's BLOCK_M // GS rows ([GS]).
    The running max is kept elementwise and only reduced across threads after the loop.
    """
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, CHUNK)
    k_rel = offs_n0[None, :] * stride_ks + offs_d[:, None]
    if GS == 1:
        m_acc = tl.full([BLOCK_M * BLOCK_N // 16], float("-inf"), tl.float32)
    else:
        m_acc = tl.full([BLOCK_M, BLOCK_N // 4], float("-inf"), tl.float32)
    for nb in range(nb_lo, nb_hi):
        offs_n = nb * BLOCK_N + offs_n0
        k_ptrs = k_base + (nb * BLOCK_N).to(tl.int64) * stride_ks + k_rel
        kmask = offs_n[None, :] < seqlen_k
        s = tl.dot(q0, _load_kt(k_ptrs, kmask, offs_d, 0, CHUNK, D))  # fp32 [BLOCK_M, BLOCK_N]
        if QSPLIT == 2:
            s = tl.dot(q1, _load_kt(k_ptrs, kmask, offs_d, CHUNK, CHUNK, D), s)
        if (nb < nb_flo) | (nb >= nb_fhi):
            ok = offs_n[None, :] <= hi_row[:, None]
            if HAS_LEFT:
                ok = ok & (offs_n[None, :] >= lo_row[:, None])
            s = tl.where(ok, s, float("-inf"))
        if GS == 1:
            # Only the overall max is needed, so the elements may be reduced in any order: this
            # lets the compiler keep the reduction within each thread.
            s = tl.reshape(s, [BLOCK_M * BLOCK_N // 16, 16], can_reorder=True)
            m_acc = tl.maximum(m_acc, tl.max(s, axis=1))
        else:
            # Rows of different heads must stay apart: reduce groups of 4 adjacent columns, which
            # the 16x16 MFMA result layout keeps within each thread.
            s = tl.reshape(s, [BLOCK_M, BLOCK_N // 4, 4])
            m_acc = tl.maximum(m_acc, tl.max(s, axis=2))
    if GS == 1:
        m = tl.max(m_acc, axis=0)
    else:
        m = tl.max(tl.reshape(tl.max(m_acc, axis=1), [GS, BLOCK_M // GS]), axis=1)
    return m


@triton.jit
def _max_logit_fwd_kernel(
    Q,
    K,
    Out,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_ob,
    stride_oh,
    num_heads,
    seqlen_q,
    seqlen_k,
    win_left,
    win_right,
    scale,
    GQA_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QSPLIT: tl.constexpr,
    GS: tl.constexpr,
    BOTTOM_RIGHT: tl.constexpr,
    HAS_LEFT: tl.constexpr,
    HAS_RIGHT: tl.constexpr,
    INT64_OFFS: tl.constexpr,
    ASYNC_COPY: tl.constexpr,  # not used in the body: keys the in-memory cache on the knob
):
    # Grid is (B * H / GS, n_q_blocks, n_split). A program covers ROWS query rows of each of GS
    # consecutive heads (sharing one K head), stacked along M. q blocks are launched last-to-first:
    # under a causal mask the last q blocks see the most keys, so starting them first avoids a
    # long tail.
    ROWS: tl.constexpr = BLOCK_M // GS
    pid_g = tl.program_id(0)
    pid_m = tl.num_programs(1) - 1 - tl.program_id(1)
    pid_s = tl.program_id(2)
    n_split = tl.num_programs(2)
    pid_h = (pid_g % (num_heads // GS)) * GS
    pid_b = pid_g // (num_heads // GS)
    out_ptr = Out + pid_b.to(tl.int64) * stride_ob + pid_h * stride_oh + pid_m * n_split + pid_s
    if INT64_OFFS:
        # in-tile offsets (up to BLOCK_M * stride_qs, BLOCK_N * stride_ks) would overflow int32
        stride_qs = stride_qs.to(tl.int64)
        stride_qh = stride_qh.to(tl.int64)
        stride_ks = stride_ks.to(tl.int64)

    m0 = pid_m * ROWS
    if BOTTOM_RIGHT:
        shift = seqlen_k - seqlen_q
    else:
        shift = 0

    # Keys touched by any row of this q block: [lo, hi). Blocks outside are skipped entirely.
    lo = 0
    hi = seqlen_k
    if HAS_LEFT:
        lo = tl.maximum(lo, m0 + shift - win_left)
    if HAS_RIGHT:
        hi = tl.minimum(hi, tl.minimum(m0 + ROWS, seqlen_q) + shift + win_right)
    nb_lo = tl.maximum(lo, 0) // BLOCK_N
    nb_hi = tl.maximum((hi + BLOCK_N - 1) // BLOCK_N, nb_lo)
    nb_hi = tl.where(hi <= lo, nb_lo, nb_hi)

    # Keys unmasked for every row of this q block: [flo, fhi). Those blocks skip the mask.
    flo = 0
    fhi = seqlen_k
    if HAS_LEFT:
        flo = tl.maximum(flo, m0 + ROWS - 1 + shift - win_left)
    if HAS_RIGHT:
        fhi = tl.minimum(fhi, m0 + shift + win_right + 1)
    fhi = tl.maximum(fhi, 0)
    nb_flo = tl.minimum(tl.maximum((flo + BLOCK_N - 1) // BLOCK_N, nb_lo), nb_hi)
    nb_fhi = tl.maximum(tl.minimum(fhi // BLOCK_N, nb_hi), nb_flo)
    if m0 + ROWS > seqlen_q:
        # rows past seqlen_q are masked below, so a partial q block has no mask-free K block
        nb_fhi = nb_flo

    # Split-K (small grids): this program only visits its share of the touched K blocks.
    chunk = tl.cdiv(nb_hi - nb_lo, n_split)
    nb_lo = tl.minimum(nb_lo + pid_s * chunk, nb_hi)
    nb_hi = tl.minimum(nb_lo + chunk, nb_hi)

    # Row r is query offs_m[r] of head pid_h + r // ROWS. Per-row inclusive key range; rows past
    # seqlen_q (q loaded as 0) allow no key.
    offs_r = tl.arange(0, BLOCK_M)
    offs_m = m0 + offs_r % ROWS
    hi_row = tl.full([BLOCK_M], seqlen_k - 1, tl.int32)
    if HAS_RIGHT:
        hi_row = tl.minimum(hi_row, offs_m + shift + win_right)
    hi_row = tl.where(offs_m < seqlen_q, hi_row, -1)
    lo_row = offs_m + shift - win_left

    CHUNK: tl.constexpr = BLOCK_D // QSPLIT
    offs_d = tl.arange(0, CHUNK)
    h_kv = pid_h // GQA_GROUP
    q_base = Q + pid_b.to(tl.int64) * stride_qb + pid_h.to(tl.int64) * stride_qh
    k_base = K + pid_b.to(tl.int64) * stride_kb + h_kv.to(tl.int64) * stride_kh
    q_rel = (offs_m - m0) * stride_qs + (offs_r // ROWS) * stride_qh
    q_ptrs = q_base + m0.to(tl.int64) * stride_qs + q_rel[:, None] + offs_d[None, :]
    qmask = offs_m[:, None] < seqlen_q
    q0 = tl.load(q_ptrs, mask=qmask & (offs_d[None, :] < D), other=0.0)
    if QSPLIT == 2:
        q1 = tl.load(q_ptrs + CHUNK, mask=qmask & (offs_d[None, :] + CHUNK < D), other=0.0)
    else:
        q1 = q0

    m = _max_logit_k_blocks(
        q0,
        q1,
        k_base,
        stride_ks,
        nb_lo,
        nb_flo,
        nb_fhi,
        nb_hi,
        hi_row,
        lo_row,
        seqlen_k,
        BLOCK_M,
        BLOCK_N,
        D,
        CHUNK,
        QSPLIT,
        GS,
        HAS_LEFT,
    )
    # scale > 0 commutes with max, and -inf * scale stays -inf. Rounding to q.dtype is monotonic,
    # so it commutes with the final max over programs too.
    m = (m * scale).to(Out.dtype.element_ty)
    if GS == 1:
        tl.store(out_ptr, m)
    else:
        tl.store(out_ptr + tl.arange(0, GS) * stride_oh, m)


def max_logit_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    qkv_format: str = "bshd",
    attn_mask_type: str = "no_mask",
    softmax_scale: Optional[float] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Compute max_logit[h] = max over b, i and unmasked j of softmax_scale * q_i . k_j.

    q: [B, S_q, H, D] (bshd) or [S_q, B, H, D] (sbhd); k likewise with H_kv heads, H % H_kv == 0.
    attn_mask_type: "no_mask", "causal" (top-left) or "causal_bottom_right".
    Returns a [H] tensor in q.dtype. Heads whose rows are all masked get -inf.
    """
    if qkv_format not in SUPPORTED_QKV_FORMATS:
        raise NotImplementedError(f"max_logit_fwd: qkv_format={qkv_format} is not supported yet.")
    if attn_mask_type not in SUPPORTED_MASK_TYPES:
        raise NotImplementedError(
            f"max_logit_fwd: attn_mask_type={attn_mask_type} is not supported yet."
        )
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise TypeError(f"max_logit_fwd: expected bf16/fp16 q and k, got {q.dtype} and {k.dtype}.")
    if q.dim() != 4 or k.dim() != 4 or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"max_logit_fwd: bad shapes q={tuple(q.shape)}, k={tuple(k.shape)}.")

    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if softmax_scale <= 0:
        raise ValueError("max_logit_fwd: softmax_scale must be > 0.")

    if qkv_format == "bshd":
        batch, seqlen_q, num_heads, _ = q.shape
        batch_k, seqlen_k, num_heads_kv = k.shape[0], k.shape[1], k.shape[2]
        stride_qb, stride_qs = q.stride(0), q.stride(1)
        stride_kb, stride_ks = k.stride(0), k.stride(1)
    else:
        seqlen_q, batch, num_heads, _ = q.shape
        seqlen_k, batch_k, num_heads_kv = k.shape[0], k.shape[1], k.shape[2]
        stride_qb, stride_qs = q.stride(1), q.stride(0)
        stride_kb, stride_ks = k.stride(1), k.stride(0)
    if batch_k != batch:
        raise ValueError(f"max_logit_fwd: batch mismatch q={tuple(q.shape)}, k={tuple(k.shape)}.")
    if num_heads % num_heads_kv != 0:
        raise ValueError(f"max_logit_fwd: H={num_heads} is not a multiple of H_kv={num_heads_kv}.")

    if batch == 0 or seqlen_q == 0 or seqlen_k == 0:
        return torch.full((num_heads,), float("-inf"), dtype=q.dtype, device=q.device)

    # (left, right) window; -1 means unlimited. Causal is (-1, 0).
    is_causal = attn_mask_type != "no_mask"
    group = num_heads // num_heads_kv
    num_cus, gfx950 = _device_info(q.device)
    rows = batch * num_heads * seqlen_q
    cfg = _pick_config(head_dim, rows, seqlen_k, is_causal, group % 2 == 0, gfx950)
    if config:
        cfg.update(config)
    block_m, block_n = cfg["BLOCK_M"], cfg["BLOCK_N"]
    gs = 1
    if is_causal:
        while 2 * gs <= cfg["GQA_STACK"] and group % (2 * gs) == 0 and block_m // (2 * gs) >= 16:
            gs *= 2
    n_q_blocks = triton.cdiv(seqlen_q, block_m // gs)
    n_waves = batch * (num_heads // gs) * n_q_blocks * cfg["num_warps"]
    split = cfg["SPLIT"]
    if split is None:
        # Too few programs leave SIMDs idle: split the K range of each q block, keeping at least
        # _MIN_SPLIT_KEYS keys per program. Causal q blocks are uneven, so aim for twice as many.
        split = 1
        target = _MIN_WAVES_PER_CU * num_cus * (2 if is_causal else 1)
        while n_waves * split < target and seqlen_k // (2 * split) >= _MIN_SPLIT_KEYS:
            split *= 2
    # Partials are stored in q.dtype: rounding is monotonic, so max(round(x)) == round(max(x)).
    partial = torch.empty((batch, num_heads, n_q_blocks * split), dtype=q.dtype, device=q.device)
    q_extent = block_m * stride_qs + gs * q.stride(2)
    int64_offs = max(q_extent, block_n * stride_ks) + head_dim >= 2**31
    with _async_copy(bool(cfg["ASYNC_COPY"])):
        _max_logit_fwd_kernel[(batch * (num_heads // gs), n_q_blocks, split)](
            q,
            k,
            partial,
            stride_qb,
            stride_qs,
            q.stride(2),
            stride_kb,
            stride_ks,
            k.stride(2),
            partial.stride(0),
            partial.stride(1),
            num_heads,
            seqlen_q,
            seqlen_k,
            0,
            0,
            float(softmax_scale),
            GQA_GROUP=group,
            D=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            QSPLIT=cfg["QSPLIT"],
            GS=gs,
            BOTTOM_RIGHT=attn_mask_type == "causal_bottom_right",
            HAS_LEFT=False,
            HAS_RIGHT=is_causal,
            INT64_OFFS=int64_offs,
            ASYNC_COPY=bool(cfg["ASYNC_COPY"]),
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            waves_per_eu=cfg["waves_per_eu"],
            matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
        )
    return torch.amax(partial, dim=(0, 2))
