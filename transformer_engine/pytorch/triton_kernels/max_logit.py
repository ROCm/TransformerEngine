# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Per-head max attention logit (for MuonClip / QK-clip) as a standalone Triton kernel.

max_logit[h] = max over (b, i, unmasked j) of scale * q[b, i, h] . k[b, j, h // (H / H_kv)]

Only Q.K^T is recomputed: no softmax, no V, no score matrix is stored. Each program writes
one fp32 value per (b, h, q_block), which is then reduced with torch.amax (deterministic,
no atomics). Intended to run next to a fused attention forward that does not emit the max.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

SUPPORTED_MASK_TYPES = ("no_mask", "causal", "causal_bottom_right")
SUPPORTED_QKV_FORMATS = ("bshd", "sbhd")

# Fixed launch config (no autotune: training shapes change often and tuning stalls a step).
_DEFAULT_CONFIG = dict(
    BLOCK_M=256, BLOCK_N=64, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16
)


@triton.jit
def _max_logit_k_blocks(
    m_acc,
    q,
    k_base,
    stride_ks,
    offs_m,
    offs_n0,
    offs_d,
    nb_start,
    nb_end,
    seqlen_k,
    shift,
    win_left,
    win_right,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MASKED: tl.constexpr,
    HAS_LEFT: tl.constexpr,
    HAS_RIGHT: tl.constexpr,
):
    """Update the running row max of q.k^T over K blocks [nb_start, nb_end)."""
    k_rel = offs_n0[None, :] * stride_ks + offs_d[:, None]
    k_blk = k_base + (nb_start * BLOCK_N).to(tl.int64) * stride_ks
    for nb in range(nb_start, nb_end):
        offs_n = nb * BLOCK_N + offs_n0
        k_ptrs = k_blk + k_rel
        k_blk += BLOCK_N * stride_ks
        if MASKED:
            kmask = offs_n[None, :] < seqlen_k
            if BLOCK_D != D:
                kmask = kmask & (offs_d[:, None] < D)
            k = tl.load(k_ptrs, mask=kmask, other=0.0)
        else:
            if BLOCK_D != D:
                k = tl.load(k_ptrs, mask=offs_d[:, None] < D, other=0.0)
            else:
                k = tl.load(k_ptrs)
        s = tl.dot(q, k)  # fp32 [BLOCK_M, BLOCK_N]
        if MASKED:
            # key j is allowed for query i iff -left <= j - (i + shift) <= right and j < seqlen_k
            diag = offs_n[None, :] - (offs_m[:, None] + shift)
            ok = offs_n[None, :] < seqlen_k
            if HAS_LEFT:
                ok = ok & (diag >= -win_left)
            if HAS_RIGHT:
                ok = ok & (diag <= win_right)
            s = tl.where(ok, s, float("-inf"))
        m_acc = tl.maximum(m_acc, tl.max(s, axis=1))
    return m_acc


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
    BOTTOM_RIGHT: tl.constexpr,
    HAS_LEFT: tl.constexpr,
    HAS_RIGHT: tl.constexpr,
):
    # Grid is (B * H, n_q_blocks) with q blocks launched last-to-first: under a causal mask the
    # last q blocks see the most keys, so starting them first avoids a long tail.
    pid_bh = tl.program_id(0)
    pid_m = tl.num_programs(1) - 1 - tl.program_id(1)
    pid_h = pid_bh % num_heads
    pid_b = pid_bh // num_heads
    out_ptr = Out + pid_b.to(tl.int64) * stride_ob + pid_h * stride_oh + pid_m

    m0 = pid_m * BLOCK_M
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
        hi = tl.minimum(hi, tl.minimum(m0 + BLOCK_M, seqlen_q) + shift + win_right)
    nb_lo = tl.maximum(lo, 0) // BLOCK_N
    nb_hi = tl.maximum((hi + BLOCK_N - 1) // BLOCK_N, nb_lo)
    nb_hi = tl.where(hi <= lo, nb_lo, nb_hi)

    # Keys unmasked for every row of this q block: [flo, fhi). Those blocks skip the mask.
    flo = 0
    fhi = seqlen_k
    if HAS_LEFT:
        flo = tl.maximum(flo, m0 + BLOCK_M - 1 + shift - win_left)
    if HAS_RIGHT:
        fhi = tl.minimum(fhi, m0 + shift + win_right + 1)
    fhi = tl.maximum(fhi, 0)
    nb_flo = tl.minimum(tl.maximum((flo + BLOCK_N - 1) // BLOCK_N, nb_lo), nb_hi)
    nb_fhi = tl.maximum(tl.minimum(fhi // BLOCK_N, nb_hi), nb_flo)

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    h_kv = pid_h // GQA_GROUP
    q_base = Q + pid_b.to(tl.int64) * stride_qb + pid_h.to(tl.int64) * stride_qh
    k_base = K + pid_b.to(tl.int64) * stride_kb + h_kv.to(tl.int64) * stride_kh
    q_ptrs = q_base + m0.to(tl.int64) * stride_qs + (offs_m[:, None] - m0) * stride_qs + offs_d[None, :]
    qmask = offs_m[:, None] < seqlen_q
    if BLOCK_D != D:
        qmask = qmask & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=qmask, other=0.0)
    m_acc = tl.full([BLOCK_M], float("-inf"), tl.float32)

    # masked prefix, unmasked interior, masked suffix
    m_acc = _max_logit_k_blocks(
        m_acc, q, k_base, stride_ks, offs_m, offs_n0, offs_d, nb_lo, nb_flo, seqlen_k, shift,
        win_left, win_right, BLOCK_N, D, BLOCK_D, True, HAS_LEFT, HAS_RIGHT,
    )
    m_acc = _max_logit_k_blocks(
        m_acc, q, k_base, stride_ks, offs_m, offs_n0, offs_d, nb_flo, nb_fhi, seqlen_k, shift,
        win_left, win_right, BLOCK_N, D, BLOCK_D, False, HAS_LEFT, HAS_RIGHT,
    )
    m_acc = _max_logit_k_blocks(
        m_acc, q, k_base, stride_ks, offs_m, offs_n0, offs_d, nb_fhi, nb_hi, seqlen_k, shift,
        win_left, win_right, BLOCK_N, D, BLOCK_D, True, HAS_LEFT, HAS_RIGHT,
    )

    # Rows past seqlen_q were loaded as zeros and must not contribute.
    m_acc = tl.where(offs_m < seqlen_q, m_acc, float("-inf"))
    # scale > 0 commutes with max, and -inf * scale stays -inf.
    tl.store(out_ptr, tl.max(m_acc, axis=0) * scale)


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
        raise NotImplementedError(f"max_logit_fwd: attn_mask_type={attn_mask_type} is not supported yet.")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise TypeError(f"max_logit_fwd: expected bf16/fp16 q and k, got {q.dtype} and {k.dtype}.")
    if q.dim() != 4 or k.dim() != 4 or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"max_logit_fwd: bad shapes q={tuple(q.shape)}, k={tuple(k.shape)}.")

    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

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
    block_m, block_n = cfg["BLOCK_M"], cfg["BLOCK_N"]
    n_q_blocks = triton.cdiv(seqlen_q, block_m)
    partial = torch.empty((batch, num_heads, n_q_blocks), dtype=torch.float32, device=q.device)
    _max_logit_fwd_kernel[(batch * num_heads, n_q_blocks)](
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
        GQA_GROUP=num_heads // num_heads_kv,
        D=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BOTTOM_RIGHT=attn_mask_type == "causal_bottom_right",
        HAS_LEFT=False,
        HAS_RIGHT=is_causal,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
    )
    return torch.amax(partial, dim=(0, 2)).to(q.dtype)
