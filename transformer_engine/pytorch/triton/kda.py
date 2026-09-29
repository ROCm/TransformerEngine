# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
#
# Adapted from AITER (ROCm/aiter @ 7d2f6a51a, aiter/ops/triton/kimi_delta_attn and
# _triton_kernels/chunk_delta_attn/{chunk_fwd,flash_kda}.py), MIT licensed.

"""PyTorch driver for the Kimi Delta Attention (KDA) Triton kernels.

``kimi_delta_attn`` mirrors the signature of ``fla.ops.kda.chunk_kda`` /
AITER's ``chunk_kimi_delta_attn``. Its forward routes a call to FlashKDA when
the call fits its envelope (``K == V == 128``, bf16, no GVA, ``safe_gate``,
the fused sigmoid gate with in-kernel l2norm and beta sigmoid) and to the
general pipeline otherwise. The backward saves only the inputs and recomputes
the general pipeline's intermediates (fla's default, ``disable_recompute=False``).

Variable-length inputs are planned on the device: chunk and segment tables are
built with tensor ops and padded to static upper bounds, so a call never reads
``cu_seqlens`` back to the host and can be captured in a CUDA graph.

Environment variables:
    NVTE_KDA_FLASH: ``0`` pins the general pipeline (default ``1``).
    NVTE_KDA_USE_GLUON: ``1``/``all`` (default), ``k1``, ``k2`` or ``0`` --
        which FlashKDA kernels use their Gluon versions on gfx950.
    NVTE_KDA_FLASH_CHUNKS_PER_SEG: override FlashKDA's segment length in
        chunks; ``<= 0`` disables segmentation.
"""

import functools
import os
from typing import Optional, Tuple

import torch
import triton

from transformer_engine.common.triton.kda import (
    FLASH_KDA_CHUNK,
    FLASH_KDA_INV_BLOCK,
    KDA_SUB_CHUNK,
    KDA_VARLEN_PLAN_BLOCK,
    PADDED_CHUNK,
    RCP_LN2,
    _flash_kda_prepare_kernel,
    _flash_kda_seg_scan_kernel,
    _flash_kda_segment_kernel,
    _kda_beta_sigmoid_kernel,
    _kda_fwd_h_kernel,
    _kda_gate_cumsum_kernel,
    _kda_gla_fwd_o_kernel,
    _kda_inter_solve_kernel,
    _kda_intra_sub_chunk_kernel,
    _kda_intra_token_parallel_kernel,
    _kda_l2norm_kernel,
    _kda_local_cumsum_kernel,
    _kda_recompute_w_u_kernel,
    _kda_varlen_plan_kernel,
    flash_kda_choose_chunks_per_seg,
    flash_kda_fixed_segments,
    flash_kda_gluon_k2_schedule,
    flash_kda_scan_bv,
    flash_kda_supported,
    kda_bwd_recurrences_config,
    kda_device_arch,
    kda_launch_config,
    kda_num_cus,
    kda_num_xcds,
    kda_recurrence_config,
    kda_varlen_max_chunks,
)
from transformer_engine.common.triton.kda_bwd import (
    _kda_bwd_beta_sigmoid_kernel,
    _kda_bwd_dav_kernel,
    _kda_bwd_dhu_kernel,
    _kda_bwd_gate_cumsum_kernel,
    _kda_bwd_intra_kernel,
    _kda_bwd_l2norm_kernel,
    _kda_bwd_wy_dqkg_kernel,
    _kda_fwd_h_bwd_dhu_kernel,
)
from transformer_engine.pytorch.triton.fast_launch import fast_launch

__all__ = ["kimi_delta_attn", "kda_fwd", "kda_bwd"]

_DEFAULT_CHUNK_SIZE = 64

_l2norm_k = fast_launch(_kda_l2norm_kernel)
_beta_sigmoid_k = fast_launch(_kda_beta_sigmoid_kernel)
_gate_cumsum_k = fast_launch(_kda_gate_cumsum_kernel)
_local_cumsum_k = fast_launch(_kda_local_cumsum_kernel)
_intra_token_parallel_k = fast_launch(_kda_intra_token_parallel_kernel)
_intra_sub_chunk_k = fast_launch(_kda_intra_sub_chunk_kernel)
_inter_solve_k = fast_launch(_kda_inter_solve_kernel)
_recompute_w_u_k = fast_launch(_kda_recompute_w_u_kernel)
_fwd_h_k = fast_launch(_kda_fwd_h_kernel)
_gla_fwd_o_k = fast_launch(_kda_gla_fwd_o_kernel)
_flash_prepare_k = fast_launch(_flash_kda_prepare_kernel)
_flash_segment_k = fast_launch(_flash_kda_segment_kernel)
_flash_seg_scan_k = fast_launch(_flash_kda_seg_scan_kernel)
_varlen_plan_k = fast_launch(_kda_varlen_plan_kernel)
_bwd_dav_k = fast_launch(_kda_bwd_dav_kernel)
_bwd_dhu_k = fast_launch(_kda_bwd_dhu_kernel)
_fwd_h_bwd_dhu_k = fast_launch(_kda_fwd_h_bwd_dhu_kernel)
_bwd_wy_dqkg_k = fast_launch(_kda_bwd_wy_dqkg_kernel)
_bwd_intra_k = fast_launch(_kda_bwd_intra_kernel)
_bwd_gate_cumsum_k = fast_launch(_kda_bwd_gate_cumsum_kernel)
_bwd_l2norm_k = fast_launch(_kda_bwd_l2norm_kernel)
_bwd_beta_sigmoid_k = fast_launch(_kda_bwd_beta_sigmoid_kernel)


def _env_flag(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


def _use_flash() -> bool:
    return _env_flag("NVTE_KDA_FLASH", "1") in ("1", "true", "yes", "on")


def _use_gluon(which: str, arch: str) -> bool:
    if arch != "gfx950":
        return False
    sel = _env_flag("NVTE_KDA_USE_GLUON", "1")
    if sel not in ("1", "true", "yes", "on", "all", which):
        return False
    return _gluon_module() is not None


@functools.lru_cache(maxsize=None)
def _gluon_module():
    try:
        from transformer_engine.common.triton import kda_gluon
    except ImportError:
        return None
    return kda_gluon


@functools.lru_cache(maxsize=None)
def _gluon_launchers():
    kg = _gluon_module()
    return fast_launch(kg.flash_kda_k1_prepare_gluon), fast_launch(kg.flash_kda_k2_ab_fused_gluon)


# ---------------------------------------------------------------------------
# Varlen planning (device-side, padded to static bounds)
# ---------------------------------------------------------------------------


def _varlen_plan(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    max_chunks: int,
    chunks_per_seg: Optional[int] = None,
    max_segs: int = 0,
):
    """Chunk (and, with ``chunks_per_seg``, FlashKDA segment) tables in one launch.

    Returns ``(chunk_indices [max_chunks, 2], chunk_offsets [N + 1], desc
    [6, max_segs] or None, seq_seg_off [N + 1] or None)``. Rows past the real
    counts are padding that every consumer skips; nothing is read back to the
    host, so the call is graph-capturable.
    """
    n = cu_seqlens.numel() - 1
    dev = cu_seqlens.device
    with_segments = chunks_per_seg is not None
    chunk_indices = torch.empty(max_chunks, 2, dtype=cu_seqlens.dtype, device=dev)
    chunk_offsets = torch.empty(n + 1, dtype=cu_seqlens.dtype, device=dev)
    desc = torch.empty(6, max_segs, dtype=torch.int32, device=dev) if with_segments else None
    seq_seg_off = torch.empty(n + 1, dtype=torch.int32, device=dev) if with_segments else None
    block = KDA_VARLEN_PLAN_BLOCK
    _varlen_plan_k[(triton.cdiv(max(max_chunks, max_segs, 1), block),)](
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seg_desc=desc,
        seq_seg_off=seq_seg_off,
        N=n,
        MAX_CHUNKS=max_chunks,
        MAX_SEGS=max_segs,
        BT=chunk_size,
        CHUNKS_PER_SEG=chunks_per_seg if with_segments else 1,
        BLOCK=block,
        WITH_SEGMENTS=with_segments,
        num_warps=4,
    )
    return chunk_indices, chunk_offsets, desc, seq_seg_off


@functools.lru_cache(maxsize=64)
def _fixed_segments(B: int, T: int, C: int, chunks_per_seg: int, device: torch.device):
    rows, seq_seg_off, num_segs = flash_kda_fixed_segments(B, T, C, chunks_per_seg)
    desc = torch.tensor(rows, dtype=torch.int32, device=device)
    off = torch.tensor(seq_seg_off, dtype=torch.int32, device=device)
    return desc, off, num_segs


# ---------------------------------------------------------------------------
# General pipeline
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor, arch: str) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    y = torch.empty_like(x)
    T, D = x.shape
    BD = triton.next_power_of_2(D)
    if D > 512:
        raise ValueError(f"KDA l2norm supports head dims up to 512, got {D}.")
    cfg = kda_launch_config("l2norm", arch)
    BT = cfg.kwargs["BT"]
    _l2norm_k[(triton.cdiv(T, BT),)](
        X=x,
        Y=y,
        eps=1e-6,
        T=T,
        D=D,
        BD=BD,
        BT=BT,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return y.view(shape)


def _beta_sigmoid(x: torch.Tensor, arch: str) -> torch.Tensor:
    y = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    cfg = kda_launch_config("beta_sigmoid", arch)
    bs = cfg.kwargs["BLOCK_SIZE"]
    _beta_sigmoid_k[(triton.cdiv(n, bs),)](
        x=x,
        y=y,
        n_elements=n,
        BLOCK_SIZE=bs,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return y


def _general_states(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    chunk_size,
    safe_gate,
    lower_bound,
    use_gate_in_kernel,
    A_log,
    dt_bias,
    use_qk_l2norm_in_kernel,
    use_beta_sigmoid_in_kernel,
    state_v_first,
    arch,
    run_fwd_h=True,
    store_qg=False,
):
    """The general pipeline up to (not including) the output kernel.

    Returns a dict of the intermediates the output kernel and the backward
    consume: the activated ``q``/``k``/``beta``, ``g_cumsum``, ``Aqk``,
    ``Akk``, ``w``, ``kg``, ``h``, ``v_new``, ``final_state`` and the chunk tables,
    plus ``qg = q * exp2(g_cumsum)`` per value head with ``store_qg``.
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    BC = KDA_SUB_CHUNK
    NC = triton.cdiv(BT, BC)
    is_varlen = cu_seqlens is not None
    N = cu_seqlens.numel() - 1 if is_varlen else B
    if is_varlen:
        NT = kda_varlen_max_chunks(T, N, BT)
        chunk_indices, chunk_offsets, _, _ = _varlen_plan(cu_seqlens, BT, NT)
    else:
        NT = triton.cdiv(T, BT)
        chunk_indices = chunk_offsets = None

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, arch)
        k = _l2norm(k, arch)
    if use_beta_sigmoid_in_kernel:
        beta = _beta_sigmoid(beta, arch)

    # Gate cumsum, in log2 space.
    g_cumsum = torch.empty_like(g, dtype=torch.float32)
    if use_gate_in_kernel:
        cfg = kda_launch_config("gate_cumsum", arch)
        BS = cfg.kwargs["BS"]
        _gate_cumsum_k[(triton.cdiv(K, BS), NT, B * HV)](
            s=g,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            o=g_cumsum,
            scale=RCP_LN2,
            lower_bound=0.0 if lower_bound is None else float(lower_bound),
            T=T,
            H=HV,
            S=K,
            BT=BT,
            BS=BS,
            HAS_BIAS=dt_bias is not None,
            HAS_SCALE=True,
            IS_VARLEN=is_varlen,
            USE_LOWER_BOUND=lower_bound is not None,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    else:
        cfg = kda_launch_config("local_cumsum", arch)
        BS = cfg.kwargs["BS"]
        _local_cumsum_k[(triton.cdiv(K, BS), NT, B * HV)](
            s=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            o=g_cumsum,
            scale=RCP_LN2,
            T=T,
            H=HV,
            S=K,
            BT=BT,
            BS=BS,
            HAS_SCALE=True,
            IS_VARLEN=is_varlen,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    # Intra-chunk: diagonal blocks, then off-diagonal blocks + inverse.
    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)
    if safe_gate:
        cfg = kda_launch_config("intra_sub_chunk", arch)
        _intra_sub_chunk_k[(NT, NC, B * HV)](
            q=q,
            k=k,
            g=g_cumsum,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=min(64, triton.next_power_of_2(K)),
            IS_VARLEN=is_varlen,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    else:
        cfg = kda_launch_config("intra_token_parallel", arch)
        BH = cfg.kwargs["BH"]
        _intra_token_parallel_k[(B * T, triton.cdiv(HV, BH))](
            q=q,
            k=k,
            g=g_cumsum,
            beta=beta,
            cu_seqlens=cu_seqlens,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            N=N,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BH=BH,
            BK=triton.next_power_of_2(K),
            IS_VARLEN=is_varlen,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    cfg = kda_launch_config("inter_solve", arch)
    _inter_solve_k[(NT, B * HV)](
        q=q,
        k=k,
        g=g_cumsum,
        beta=beta,
        Akkd=Akkd,
        Aqk_diag=Aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        BK=cfg.kwargs["BK"],
        IS_VARLEN=is_varlen,
        USE_SAFE_GATE=safe_gate,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    # W/U recompute.
    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    qg = torch.empty_like(kg) if store_qg else None
    cfg = kda_launch_config("recompute_w_u", arch)
    _recompute_w_u_k[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=g_cumsum,
        q=q if store_qg else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        w=w,
        u=u,
        kg=kg,
        qg=qg,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=cfg.kwargs["BK"],
        BV=cfg.kwargs["BV"],
        IS_VARLEN=is_varlen,
        STORE_QG=store_qg,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    # Inter-chunk recurrence.
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    if initial_state is not None and tuple(initial_state.shape) != state_shape:
        raise ValueError(
            f"`initial_state` must have shape {state_shape} for "
            f"state_v_first={state_v_first}, got {tuple(initial_state.shape)}."
        )
    h = k.new_empty(B, NT, HV, K, V)
    v_new = torch.empty_like(u)
    final_state = k.new_empty(*state_shape, dtype=torch.float32) if output_final_state else None
    cfg = kda_recurrence_config("fwd_h", N * HV, V, arch, kda_num_cus(k.device.index or 0))
    BV = cfg.kwargs["BV"]
    # kda_bwd may run fwd_h itself, alongside dhu (_kda_fwd_h_bwd_dhu_kernel).
    if run_fwd_h:
        _fwd_h_k[(triton.cdiv(V, BV), N * HV)](
            k=kg,
            v=u,
            w=w,
            gk=g_cumsum,
            h0=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            h=h,
            v_new=v_new,
            ht=final_state,
            T=T,
            H=HV,
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            IS_VARLEN=is_varlen,
            TRANSPOSE_STATE=state_v_first,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    return {
        "q": q,
        "k": k,
        "beta": beta,
        "g_cumsum": g_cumsum,
        "Aqk": Aqk,
        "Akk": Akk,
        "w": w,
        "kg": kg,
        "qg": qg,
        "u": u,
        "h": h,
        "v_new": v_new,
        "final_state": final_state,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "N": N,
        "NT": NT,
    }


def _general_fwd(q, v, scale, cu_seqlens, chunk_size, arch, **kwargs):
    st = _general_states(
        q=q, v=v, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size, arch=arch, **kwargs
    )
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    NT = st["NT"]
    is_varlen = cu_seqlens is not None
    o = torch.zeros_like(v)
    cfg = kda_launch_config("gla_fwd_o", arch)
    BV = cfg.kwargs["BV"]
    _gla_fwd_o_k[(triton.cdiv(V, BV), NT, B * HV)](
        q=st["q"],
        v=st["v_new"],
        g=st["g_cumsum"],
        h=st["h"],
        A=st["Aqk"],
        cu_seqlens=cu_seqlens,
        chunk_indices=st["chunk_indices"],
        o=o,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=cfg.kwargs["BK"],
        BV=BV,
        IS_VARLEN=is_varlen,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return o, st["final_state"]


# ---------------------------------------------------------------------------
# FlashKDA
# ---------------------------------------------------------------------------


def _flash_fwd(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    initial_state,
    output_final_state,
    state_v_first,
    cu_seqlens,
    max_seqlen,
    arch,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = FLASH_KDA_CHUNK
    BC = min(FLASH_KDA_INV_BLOCK, C)
    dev = q.device
    num_cus = kda_num_cus(dev.index or 0)
    is_varlen = cu_seqlens is not None

    if is_varlen:
        N = cu_seqlens.numel() - 1
        NT = 0
        total_tiles = kda_varlen_max_chunks(T, N, C)
        n_chunks_max = triton.cdiv(max_seqlen if max_seqlen is not None else T, C)
    else:
        N = B
        NT = triton.cdiv(T, C)
        total_tiles = B * NT
        n_chunks_max = NT

    override = os.getenv("NVTE_KDA_FLASH_CHUNKS_PER_SEG", "").strip()
    if override:
        chunks_per_seg = int(override)
        if chunks_per_seg <= 0:
            chunks_per_seg = n_chunks_max
    else:
        chunks_per_seg = flash_kda_choose_chunks_per_seg(n_chunks_max, N, H, V, num_cus)
    segmented = n_chunks_max > chunks_per_seg

    if is_varlen:
        if segmented:
            num_segs = triton.cdiv(total_tiles, chunks_per_seg) + N
            seg_len = chunks_per_seg
        else:
            # One segment per sequence however long it is.
            num_segs = N
            seg_len = PADDED_CHUNK
        chunk_indices, _, desc, seq_seg_off = _varlen_plan(
            cu_seqlens, C, total_tiles, chunks_per_seg=seg_len, max_segs=num_segs
        )
    else:
        chunk_indices = None
        desc, seq_seg_off, num_segs = _fixed_segments(
            B, T, C, chunks_per_seg if segmented else NT, dev
        )
    seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end, seg_seq, seg_is_last = desc

    CM_LOAD = ".cg"
    CM_STORE = "" if segmented else ".wt"
    CM_OUT_STORE = "" if segmented else ".cs"

    ws_shape = (H * total_tiles, C, K)
    ws_kd = torch.empty(ws_shape, dtype=torch.bfloat16, device=dev)
    ws_qd = torch.empty(ws_shape, dtype=torch.bfloat16, device=dev)
    ws_kr = torch.empty(ws_shape, dtype=torch.bfloat16, device=dev)
    ws_gt = torch.empty(H * total_tiles, K, dtype=torch.float32, device=dev)
    # fp16, not bf16, for the mantissa: the inverse is bounded by 1 and K2 walks
    # it through a sequence-length recurrence.
    ws_inv_mqk = torch.empty(H * total_tiles, 2 * C, C, dtype=torch.float16, device=dev)

    k1_grid = (total_tiles if is_varlen else NT, B * H)
    if _use_gluon("k1", arch):
        k1_gluon, _ = _gluon_launchers()
        k1_gluon[k1_grid](
            q=q,
            k=k,
            g_raw=g,
            beta_raw=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            ws_kd=ws_kd,
            ws_qd=ws_qd,
            ws_kr=ws_kr,
            ws_gt=ws_gt,
            ws_inv_mqk=ws_inv_mqk,
            scale=scale,
            lower_bound=lower_bound,
            T=T,
            NT=NT,
            TOTAL_TILES=total_tiles,
            H=H,
            K=K,
            C=C,
            BC=BC,
            IS_VARLEN=is_varlen,
            HAS_BIAS=dt_bias is not None,
            CM_WS=CM_STORE,
            CM_LOAD=CM_LOAD,
            num_warps=_gluon_module().K1_NUM_WARPS,
            num_stages=kda_launch_config("flash_gluon_k1", arch).num_stages,
        )
    else:
        cfg = kda_launch_config("flash_prepare", arch)
        _flash_prepare_k[k1_grid](
            q=q,
            k=k,
            g_raw=g,
            beta_raw=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            ws_kd=ws_kd,
            ws_qd=ws_qd,
            ws_kr=ws_kr,
            ws_gt=ws_gt,
            ws_inv_mqk=ws_inv_mqk,
            scale=scale,
            lower_bound=lower_bound,
            T=T,
            NT=NT,
            TOTAL_TILES=total_tiles,
            H=H,
            K=K,
            C=C,
            BC=BC,
            NUM_DOUBLING=BC.bit_length() - 2,
            NUM_MERGE=(C // BC).bit_length() - 1,
            IS_VARLEN=is_varlen,
            HAS_BIAS=dt_bias is not None,
            CM_QKG=CM_LOAD,
            CM_WS=CM_STORE,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    # K2 indexes h_in by segment, so normalize a V-first initial state once.
    h0 = initial_state
    if h0 is not None:
        if state_v_first:
            h0 = h0.transpose(-1, -2)
        h0 = h0.contiguous()

    o = torch.empty_like(v)
    final_state = None
    if output_final_state:
        shape = (N, H, V, K) if state_v_first else (N, H, K, V)
        state_dtype = h0.dtype if h0 is not None else torch.float32
        final_state = torch.empty(shape, dtype=state_dtype, device=dev)

    seg_cfg = kda_launch_config("flash_segment", arch)
    common = {
        "ws_kd": ws_kd,
        "ws_qd": ws_qd,
        "ws_kr": ws_kr,
        "ws_gt": ws_gt,
        "ws_inv_mqk": ws_inv_mqk,
        "beta_raw": beta,
        "seg_chunk_base": seg_chunk_base,
        "seg_nchunks": seg_nchunks,
        "seg_tok_base": seg_tok_base,
        "seg_tok_end": seg_tok_end,
        "seg_seq": seg_seq,
        "seg_is_last": seg_is_last,
        "TOTAL_TILES": total_tiles,
        "H": H,
        "K": K,
        "V": V,
        "C": C,
        "BW": seg_cfg.kwargs["BW"],
        "STATE_V_FIRST": state_v_first,
        "CM_OUT": CM_OUT_STORE,
        "NUM_XCDS": kda_num_xcds(arch),
        "num_warps": seg_cfg.num_warps,
        "num_stages": seg_cfg.num_stages,
    }

    def _launch_k2(*, W, **kw):
        grid = (triton.cdiv(W, seg_cfg.kwargs["BW"]), num_segs * H)
        _flash_segment_k[grid](W=W, **common, **kw)

    if segmented:
        # Pass A: b_seg (affine part) and A_seg (linear part), parallel over segments.
        b_seg = torch.empty(num_segs, H, K, V, dtype=torch.float32, device=dev)
        A_seg = torch.empty(num_segs, H, K, K, dtype=torch.bfloat16, device=dev)
        if _use_gluon("k2", arch):
            _, k2_gluon = _gluon_launchers()
            bw, nw, ns, wpe = flash_kda_gluon_k2_schedule(V, num_segs, H, arch, num_cus)
            k2_gluon[(triton.cdiv(V, bw), num_segs * H)](
                ws_kd=ws_kd,
                ws_kr=ws_kr,
                ws_gt=ws_gt,
                ws_inv_mqk=ws_inv_mqk,
                v_input=v,
                beta_raw=beta,
                seg_chunk_base=seg_chunk_base,
                seg_nchunks=seg_nchunks,
                seg_tok_base=seg_tok_base,
                seg_tok_end=seg_tok_end,
                h_out_b=b_seg,
                h_out_a=A_seg,
                TOTAL_TILES=total_tiles,
                H=H,
                K=K,
                V=V,
                C=C,
                BW=bw,
                **_gluon_module().flash_kda_k2_layouts(nw),
                NUM_XCDS=kda_num_xcds(arch),
                num_warps=nw,
                num_stages=ns,
                waves_per_eu=wpe,
            )
        else:
            for buf, width, identity, has_v in (
                (b_seg, V, False, True),
                (A_seg, K, True, False),
            ):
                _launch_k2(
                    v_input=v,
                    out=None,
                    h_in=None,
                    h_out=buf,
                    final_state=None,
                    W=width,
                    INIT_IDENTITY=identity,
                    HAS_H_IN=False,
                    HAS_V=has_v,
                    COMPUTE_OUTPUT=False,
                    STORE_H_OUT=True,
                    STORE_FINAL=False,
                )

        # Pass B: propagate across segments; depth is the segment count.
        h_in = torch.empty(num_segs, H, K, V, dtype=torch.float32, device=dev)
        BV_SCAN, SCAN_WARPS = flash_kda_scan_bv(N, H, V, num_cus)
        _flash_seg_scan_k[(triton.cdiv(V, BV_SCAN), N * H)](
            A_seg=A_seg,
            b_seg=b_seg,
            h0=h0,
            seq_seg_off=seq_seg_off,
            h_in=h_in,
            H=H,
            K=K,
            V=V,
            BV=BV_SCAN,
            HAS_H0=h0 is not None,
            num_warps=SCAN_WARPS,
            num_stages=kda_launch_config("flash_seg_scan", arch).num_stages,
        )
    else:
        h_in = h0

    # Pass C: re-run each segment from its true incoming state, writing outputs.
    _launch_k2(
        v_input=v,
        out=o,
        h_in=h_in,
        h_out=None,
        final_state=final_state,
        W=V,
        INIT_IDENTITY=False,
        HAS_H_IN=h_in is not None,
        HAS_V=True,
        COMPUTE_OUTPUT=True,
        STORE_H_OUT=False,
        STORE_FINAL=output_final_state,
    )
    return o, final_state


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    chunk_size: Optional[int] = None,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    state_v_first: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KDA forward on contiguous inputs, choosing between FlashKDA and the general pipeline.

    An unset ``chunk_size`` resolves to 32 when that lets FlashKDA serve the
    call and to 64 otherwise (64 is the faster of the two inside the general
    pipeline).
    """
    H, K = q.shape[2], q.shape[3]
    HV, V = v.shape[2], v.shape[-1]
    use_flash = _use_flash() and flash_kda_supported(
        K=K,
        V=V,
        H=H,
        HV=HV,
        qv_bf16=q.dtype == torch.bfloat16 and v.dtype == torch.bfloat16,
        chunk_size=FLASH_KDA_CHUNK if chunk_size is None else chunk_size,
        safe_gate=safe_gate,
        use_gate_in_kernel=use_gate_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        has_lower_bound=lower_bound is not None,
        has_A_log=A_log is not None,
    )
    if chunk_size is None:
        chunk_size = FLASH_KDA_CHUNK if use_flash else _DEFAULT_CHUNK_SIZE
    if chunk_size not in (32, 64):
        raise ValueError(f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}.")

    with torch.cuda.device(q.device):
        arch = kda_device_arch(q.device.index or 0)
        if use_flash:
            return _flash_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                lower_bound=float(lower_bound),
                initial_state=initial_state,
                output_final_state=output_final_state,
                state_v_first=state_v_first,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                arch=arch,
            )
        return _general_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            state_v_first=state_v_first,
            arch=arch,
        )


def kda_bwd(
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    state_v_first: bool = False,
    **_,
) -> Tuple[torch.Tensor, ...]:
    """KDA backward on contiguous inputs. Recomputes the general pipeline's intermediates.

    The gradient is that of the general pipeline, whichever implementation ran
    the forward (FlashKDA computes the same function up to rounding).
    Returns ``(dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0)``; entries for
    absent inputs are ``None``.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None
    with torch.cuda.device(q.device):
        arch = kda_device_arch(q.device.index or 0)
        num_cus = kda_num_cus(q.device.index or 0)
        N = cu_seqlens.numel() - 1 if is_varlen else B
        rec_cfg = kda_bwd_recurrences_config(N * HV, V, arch, num_cus)
        st = _general_states(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            state_v_first=state_v_first,
            arch=arch,
            run_fwd_h=rec_cfg is None,
            store_qg=True,
        )
        qn, kn, bn, gc = st["q"], st["k"], st["beta"], st["g_cumsum"]
        chunk_indices, NT = st["chunk_indices"], st["NT"]
        do = do.contiguous()
        dht = dht.contiguous() if dht is not None else None

        # dAqk = do @ v_new^T, dv = Aqk^T @ do.
        dAqk = torch.empty(B, T, HV, BT, device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        cfg_dav = kda_launch_config("bwd_dav", arch)

        def dav(compute_da, compute_dv):
            _bwd_dav_k[(NT, B * HV)](
                v=st["v_new"],
                A=st["Aqk"],
                do=do,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                dA=dAqk,
                dv=dv,
                scale=scale,
                T=T,
                HV=HV,
                V=V,
                BT=BT,
                BV=cfg_dav.kwargs["BV"],
                IS_VARLEN=is_varlen,
                COMPUTE_DA=compute_da,
                COMPUTE_DV=compute_dv,
                num_warps=cfg_dav.num_warps,
                num_stages=cfg_dav.num_stages,
            )

        # Reverse-time state gradient.
        dh = q.new_empty(B, NT, HV, K, V)
        dh0 = (
            torch.empty_like(initial_state, dtype=torch.float32)
            if initial_state is not None
            else None
        )
        dv2 = torch.empty_like(dv)
        dhu_args = {
            "qg": st["qg"],
            "g": gc,
            "k": st["kg"],
            "w": st["w"],
            "dht": dht,
            "do": do,
            "dv": dv,
            "cu_seqlens": cu_seqlens,
            "chunk_offsets": st["chunk_offsets"],
            "dh": dh,
            "dh0": dh0,
            "dv2": dv2,
            "scale": scale,
            "T": T,
            "HV": HV,
            "K": K,
            "V": V,
            "BT": BT,
            "USE_INITIAL_STATE": initial_state is not None,
            "USE_FINAL_STATE_GRADIENT": dht is not None,
            "IS_VARLEN": is_varlen,
            "TRANSPOSE_STATE": state_v_first,
            "NUM_XCDS": kda_num_xcds(arch),
        }
        if rec_cfg is None:
            dav(True, True)
            cfg = kda_recurrence_config("bwd_dhu", N * HV, V, arch, num_cus)
            BV = cfg.kwargs["BV"]
            _bwd_dhu_k[(triton.cdiv(V, BV), N * HV)](
                **dhu_args, BV=BV, num_warps=cfg.num_warps, num_stages=cfg.num_stages
            )
        else:
            # dhu needs dv but not v_new, so it runs alongside the recomputed
            # fwd_h; dAqk, which does need v_new, follows.
            dav(False, True)
            BV = rec_cfg.kwargs["BV"]
            _fwd_h_bwd_dhu_k[(2 * triton.cdiv(V, BV), N * HV)](
                **dhu_args,
                v=st["u"],
                h0=initial_state,
                h=st["h"],
                v_new=st["v_new"],
                BV=BV,
                num_warps=rec_cfg.num_warps,
                num_stages=rec_cfg.num_stages,
            )
            dav(True, False)

        # Inter-chunk and WY-representation gradients.
        # Varlen tokens past cu_seqlens[-1] are never written: zero those outputs.
        alloc = torch.zeros if is_varlen else torch.empty
        dq = torch.empty(B, T, HV, K, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dg = torch.empty_like(dq)
        dv = alloc(v.shape, device=v.device, dtype=v.dtype)
        db = alloc(B, T, HV, device=q.device, dtype=torch.float32)
        dAkk = torch.empty_like(dAqk)
        cfg = kda_launch_config("bwd_wy_dqkg", arch)
        _bwd_wy_dqkg_k[(NT, B * HV)](
            q=qn,
            k=kn,
            v=v,
            v_new=st["v_new"],
            g=gc,
            beta=bn,
            A=st["Akk"],
            h=st["h"],
            do=do,
            dh=dh,
            dv=dv2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            dq=dq,
            dk=dk,
            dv2=dv,
            dg=dg,
            db=db,
            dA=dAkk,
            scale=scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK=cfg.kwargs["BK"],
            BV=cfg.kwargs["BV"],
            IS_VARLEN=is_varlen,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

        # Intra-chunk gradients.
        # With the in-kernel l2norm and no GVA (no head-group sum pending), one
        # program per whole K row (K <= 128) finishes dq/dk through the l2norm
        # and dbeta. Without the l2norm to fold in, the wider tile costs more
        # than the small dbeta kernels it would save.
        cfg = kda_launch_config("bwd_intra", arch)
        BC = min(KDA_SUB_CHUNK, BT)
        fuse = K <= 128 and use_qk_l2norm_in_kernel and HV == H
        sigmoid = fuse and use_beta_sigmoid_in_kernel
        BK = triton.next_power_of_2(K)
        if not fuse:
            BK = min(cfg.kwargs["BK"], BK)
        NC = triton.cdiv(BT, BC)
        NK = triton.cdiv(K, BK)
        dq2 = alloc(dq.shape, device=q.device, dtype=q.dtype if fuse else torch.float32)
        dk2 = alloc(dk.shape, device=q.device, dtype=k.dtype if fuse else torch.float32)
        dg2 = torch.empty_like(dg)
        if fuse:
            db2 = alloc(B, T, HV, device=q.device, dtype=beta.dtype)
        else:
            db2 = alloc(NK, B, T, HV, device=q.device, dtype=torch.float32)
        _bwd_intra_k[(NK * NC, NT, B * HV)](
            q=qn,
            k=kn,
            g=gc,
            beta=bn,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq,
            dk=dk,
            dg=dg,
            q_raw=q if fuse else None,
            k_raw=k if fuse else None,
            beta_raw=beta if sigmoid else None,
            db=db if fuse else None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            dq2=dq2,
            dk2=dk2,
            dg2=dg2,
            db2=db2,
            eps=1e-6,
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            NC=NC,
            IS_VARLEN=is_varlen,
            SAFE_GATE=safe_gate,
            NUM_XCDS=kda_num_xcds(arch),
            L2NORM_QK=fuse,
            FINISH_DB=fuse,
            BETA_SIGMOID=sigmoid,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        dq, dk, db = dq2, dk2, db2 if fuse else db2.sum(0).add_(db)
        if HV > H:
            dq = dq.view(B, T, H, HV // H, K).sum(3)
            dk = dk.view(B, T, H, HV // H, K).sum(3)

        # Gradient w.r.t. the per-token gate, then through its activation, in one
        # pass that also leaves per-chunk partial sums of dA_log and ddt_bias.
        cfg = kda_launch_config("bwd_gate_cumsum", arch)
        BS = cfg.kwargs["BS"]
        NS = triton.cdiv(K, BS)
        use_bias = use_gate_in_kernel and dt_bias is not None
        f32 = {"device": q.device, "dtype": torch.float32}
        dA_part = torch.empty(NT, B, HV, NS, **f32) if use_gate_in_kernel else None
        db_part = torch.empty(NT * B, HV * K, **f32) if use_bias else None
        dg = alloc(dg2.shape, device=q.device, dtype=g.dtype)
        _bwd_gate_cumsum_k[(NS, NT, B * HV)](
            s=dg2,
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            o=dg,
            dA=dA_part,
            dbias=db_part,
            lower_bound=0.0 if lower_bound is None else float(lower_bound),
            T=T,
            H=HV,
            S=K,
            BT=BT,
            BS=BS,
            IS_VARLEN=is_varlen,
            USE_GATE=use_gate_in_kernel,
            HAS_BIAS=use_bias,
            USE_LOWER_BOUND=lower_bound is not None,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        dA_log = dA_part.sum((0, 1, 3)).to(A_log.dtype) if use_gate_in_kernel else None
        ddt_bias = db_part.sum(0).to(dt_bias.dtype) if use_bias else None

        if use_qk_l2norm_in_kernel and not fuse:
            dq = _l2norm_bwd(q, dq, arch)
            dk = _l2norm_bwd(k, dk, arch)
        if use_beta_sigmoid_in_kernel and not sigmoid:
            db = _beta_sigmoid_bwd(beta, db, arch)
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv,
        dg.to(g.dtype),
        db.to(beta.dtype),
        dA_log,
        ddt_bias,
        dh0.to(initial_state.dtype) if dh0 is not None else None,
    )


def _l2norm_bwd(x: torch.Tensor, dy: torch.Tensor, arch: str) -> torch.Tensor:
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    dx = torch.empty_like(x2)
    T = x2.shape[0]
    cfg = kda_launch_config("bwd_l2norm", arch)
    BT = cfg.kwargs["BT"]
    _bwd_l2norm_k[(triton.cdiv(T, BT),)](
        X=x2,
        DY=dy.reshape(-1, D),
        DX=dx,
        eps=1e-6,
        T=T,
        D=D,
        BD=triton.next_power_of_2(D),
        BT=BT,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return dx.view(x.shape)


def _beta_sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, arch: str) -> torch.Tensor:
    dx = torch.empty_like(x)
    n = x.numel()
    cfg = kda_launch_config("bwd_beta_sigmoid", arch)
    bs = cfg.kwargs["BLOCK_SIZE"]
    _bwd_beta_sigmoid_k[(triton.cdiv(n, bs),)](
        x=x,
        dy=dy,
        dx=dx,
        n_elements=n,
        BLOCK_SIZE=bs,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return dx


class _KimiDeltaAttnFunction(torch.autograd.Function):
    """Autograd wrapper: the forward runs ``kda_fwd`` and the backward ``kda_bwd``."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, initial_state, kwargs):
        """Run ``kda_fwd`` and save its inputs; the backward recomputes the rest."""
        o, final_state = kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            **kwargs,
        )
        ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias, initial_state, kwargs["cu_seqlens"])
        ctx.kwargs = {key: val for key, val in kwargs.items() if key != "cu_seqlens"}
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        """Run ``kda_bwd``."""
        q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens = ctx.saved_tensors
        grads = kda_bwd(
            do=do,
            dht=dht,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            **ctx.kwargs,
        )
        return (*grads, None)


def kimi_delta_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
    chunk_size: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Chunked Kimi Delta Attention, differentiable w.r.t. every float input.

    Args:
        q: queries ``[B, T, H, K]``.
        k: keys ``[B, T, H, K]``.
        v: values ``[B, T, HV, V]``; GVA when ``HV > H`` (``HV % H == 0``).
        g: forget gate ``[B, T, HV, K]``: the log-space decay, or the raw
            pre-activation when ``use_gate_in_kernel``.
        beta: ``[B, T, HV]``, raw logits when ``use_beta_sigmoid_in_kernel``.
            Pass fp32 to keep the write strength from rounding.
        A_log: per-head log-scale ``[HV]``; required with ``use_gate_in_kernel``.
        dt_bias: per-channel bias ``[HV * K]`` added before the gate activation.
        scale: attention scale, default ``K ** -0.5``.
        initial_state: ``[N, HV, K, V]`` (``[N, HV, V, K]`` when
            ``state_v_first``) for ``N`` sequences.
        output_final_state: also return the final state. Its dtype is
            ``initial_state``'s on the FlashKDA path, fp32 otherwise.
        use_qk_l2norm_in_kernel: L2-normalize ``q`` and ``k``.
        use_gate_in_kernel: compute the decay from ``g`` with ``A_log``/``dt_bias``.
        use_beta_sigmoid_in_kernel: apply ``sigmoid`` to ``beta``.
        safe_gate: bound the intra-chunk decay (requires ``lower_bound``).
        lower_bound: gate lower bound in log space; the activation becomes
            ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` instead of
            ``-exp(A_log) * softplus(g + dt_bias)``. Kimi uses ``-5.0``.
        state_v_first: store states V-first.
        chunk_size: 32 or 64; ``None`` lets the implementation choose.
        cu_seqlens: ``[N + 1]`` cumulative sequence lengths (``B`` must be 1).
        max_seqlen: optional upper bound on the longest sequence with
            ``cu_seqlens``; tightens FlashKDA's scheduling. Must not be exceeded.

    Returns:
        ``(o, final_state)``: ``o`` is ``[B, T, HV, V]`` in ``q``'s dtype.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} vs {k.shape}.")
    if K > 256:
        raise ValueError(f"Only key head dim <= 256 is supported, got {K}.")
    if HV % H != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={HV}) must be divisible by num_qk_heads (H={H})."
        )
    if tuple(g.shape) != (B, T, HV, K):
        raise ValueError(f"g must have shape {(B, T, HV, K)}, got {tuple(g.shape)}.")
    if tuple(beta.shape) != (B, T, HV):
        raise ValueError(f"beta must have shape {(B, T, HV)}, got {tuple(beta.shape)}.")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`."
            )
        if initial_state is not None and initial_state.shape[0] != cu_seqlens.numel() - 1:
            raise ValueError(
                "The number of initial states is expected to be equal to the number of "
                f"input sequences, {cu_seqlens.numel() - 1}, not {initial_state.shape[0]}."
            )
    if initial_state is not None and not initial_state.is_floating_point():
        raise ValueError(f"`initial_state` must be a float tensor, got {initial_state.dtype}.")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError(
                "`lower_bound` must be specified when `safe_gate=True` and "
                "`use_gate_in_kernel=True`."
            )
        if not -5 <= lower_bound < 0:
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")
    if scale is None:
        scale = K**-0.5
    elif scale <= 0:
        raise ValueError(f"`scale` must be positive, got {scale}.")

    # Several kernels index their operands with contiguous strides.
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    A_log = A_log.contiguous() if A_log is not None else None
    dt_bias = dt_bias.contiguous() if dt_bias is not None else None
    initial_state = initial_state.contiguous() if initial_state is not None else None
    cu_seqlens = cu_seqlens.contiguous() if cu_seqlens is not None else None

    kwargs = {
        "scale": scale,
        "output_final_state": output_final_state,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "chunk_size": chunk_size,
        "safe_gate": safe_gate,
        "lower_bound": lower_bound,
        "use_gate_in_kernel": use_gate_in_kernel,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
        "state_v_first": state_v_first,
    }
    tensors = (q, k, v, g, beta, A_log, dt_bias, initial_state)
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        o, final_state = _KimiDeltaAttnFunction.apply(*tensors, kwargs)
    else:
        o, final_state = kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            **kwargs,
        )
    return o.to(q.dtype), final_state
