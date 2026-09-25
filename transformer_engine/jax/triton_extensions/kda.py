# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""JAX driver for the Kimi Delta Attention (KDA) Triton kernels. Forward only.

Mirrors ``transformer_engine.pytorch.triton.kda``: the same kernels from
``transformer_engine.common.triton.kda`` (and ``kda_gluon`` on gfx950), the
same launch configs and the same FlashKDA / general-pipeline routing, so the
two frameworks produce the same results.

Every launch goes through one generic primitive, ``te_kda_triton_call``, whose
lowering is ``triton_call_lowering``. Kernels take their tensor inputs first
and outputs last, which is the order that lowering binds them in. Scalars are
compile-time constants, and optional tensors are passed as ``None`` constexprs.

Variable-length inputs are planned with ``jnp`` ops and padded to static upper
bounds, so the whole op is traceable under ``jax.jit``.
"""

import functools
import os
from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import core as jax_core
from jax.extend import core
from jax.interpreters import mlir, xla
from jax._src import dispatch
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
    kda_device_arch,
    kda_launch_config,
    kda_num_cus,
    kda_varlen_max_chunks,
)
from ..util import is_hip_extension
from .utils import triton_call_lowering

__all__ = ["kda_fwd"]

_DEFAULT_CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# Generic Triton-call primitive
# ---------------------------------------------------------------------------

_kda_call_p = core.Primitive("te_kda_triton_call")
_kda_call_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(_kda_call_p)
_kda_call_p.def_impl(functools.partial(xla.apply_primitive, _kda_call_p))


def _kda_call_abstract(*avals, out_shapes, **kwargs):
    del avals, kwargs
    return [jax_core.ShapedArray(shape, jnp.dtype(dtype)) for shape, dtype in out_shapes]


_kda_call_p.def_abstract_eval(_kda_call_abstract)


def _kda_call_lowering(
    ctx,
    *args,
    kernel,
    out_shapes,
    grid,
    constexprs,
    num_warps,
    num_stages,
    aliases,
    gluon_k2_warps,
):
    del out_shapes
    consts = dict(constexprs)
    if gluon_k2_warps:
        consts.update(_gluon_module().flash_kda_k2_layouts(gluon_k2_warps))
    # FMA contraction on, as Triton's default for the same kernels launched
    # from PyTorch, so the two frameworks agree bit for bit.
    return triton_call_lowering(
        ctx,
        kernel,
        *args,
        grid=grid,
        input_output_aliases=dict(aliases) if aliases else None,
        constexprs=consts,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=True,
    )


mlir.register_lowering(
    _kda_call_p, _kda_call_lowering, platform="rocm" if is_hip_extension() else "cuda"
)


def _call(
    kernel,
    inputs,
    outputs,
    grid,
    constexprs,
    num_warps=4,
    num_stages=3,
    aliases=None,
    gluon_k2_warps=0,
):
    """Launch ``kernel`` once.

    ``inputs`` is a list of ``(name, array_or_None)`` and ``outputs`` a list of
    ``(name, shape, dtype)`` or ``(name, None)``, both in kernel parameter
    order. ``None`` entries become ``None`` constexprs. ``aliases`` maps an
    input name to an output name that must reuse its buffer.
    """
    consts = dict(constexprs)
    arrays, in_names = [], []
    for name, arr in inputs:
        if arr is None:
            consts[name] = None
        else:
            arrays.append(arr)
            in_names.append(name)
    out_shapes, out_names = [], []
    for entry in outputs:
        if entry[1] is None:
            consts[entry[0]] = None
        else:
            out_names.append(entry[0])
            out_shapes.append((tuple(int(d) for d in entry[1]), jnp.dtype(entry[2]).name))
    alias_pairs = tuple((in_names.index(i), out_names.index(o)) for i, o in (aliases or {}).items())
    grid = tuple(int(x) for x in grid) + (1,) * (3 - len(grid))
    outs = _kda_call_p.bind(
        *arrays,
        kernel=kernel,
        out_shapes=tuple(out_shapes),
        grid=grid,
        constexprs=tuple(sorted(consts.items())),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        aliases=alias_pairs,
        gluon_k2_warps=int(gluon_k2_warps),
    )
    return dict(zip(out_names, outs))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


def _use_flash() -> bool:
    return _env_flag("NVTE_KDA_FLASH", "1") in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=None)
def _gluon_module():
    try:
        from transformer_engine.common.triton import kda_gluon
    except ImportError:
        return None
    return kda_gluon


def _use_gluon(which: str, arch: str) -> bool:
    if arch != "gfx950":
        return False
    sel = _env_flag("NVTE_KDA_USE_GLUON", "1")
    if sel not in ("1", "true", "yes", "on", "all", which):
        return False
    return _gluon_module() is not None


# ---------------------------------------------------------------------------
# Varlen planning
# ---------------------------------------------------------------------------


def _varlen_plan(cu_seqlens, chunk_size: int, max_chunks: int, chunks_per_seg=None, max_segs=0):
    """Chunk (and FlashKDA segment) tables in one launch; see ``_kda_varlen_plan_kernel``.

    Returns ``(chunk_indices, chunk_offsets, desc or None, seq_seg_off or None)``.
    """
    n = cu_seqlens.shape[0] - 1
    with_segments = chunks_per_seg is not None
    block = KDA_VARLEN_PLAN_BLOCK
    r = _call(
        _kda_varlen_plan_kernel,
        [("cu_seqlens", cu_seqlens)],
        [
            ("chunk_indices", (max_chunks, 2), cu_seqlens.dtype),
            ("chunk_offsets", (n + 1,), cu_seqlens.dtype),
            ("seg_desc", (6, max_segs) if with_segments else None, jnp.int32),
            ("seq_seg_off", (n + 1,) if with_segments else None, jnp.int32),
        ],
        (triton.cdiv(max(max_chunks, max_segs, 1), block),),
        {
            "N": n,
            "MAX_CHUNKS": max_chunks,
            "MAX_SEGS": max_segs,
            "BT": chunk_size,
            "CHUNKS_PER_SEG": chunks_per_seg if with_segments else 1,
            "BLOCK": block,
            "WITH_SEGMENTS": with_segments,
        },
        num_warps=4,
        num_stages=2,
    )
    return r["chunk_indices"], r["chunk_offsets"], r.get("seg_desc"), r.get("seq_seg_off")


def _fixed_segments(B: int, T: int, C: int, chunks_per_seg: int):
    rows, seq_seg_off, num_segs = flash_kda_fixed_segments(B, T, C, chunks_per_seg)
    return (
        jnp.asarray(np.asarray(rows, dtype=np.int32)),
        jnp.asarray(np.asarray(seq_seg_off, dtype=np.int32)),
        num_segs,
    )


def _mask_tail(o, cu_seqlens):
    """Zero tokens past ``cu_seqlens[-1]``, which no kernel writes."""
    tok = jnp.arange(o.shape[1], dtype=jnp.int32)
    return jnp.where((tok < cu_seqlens[-1])[None, :, None, None], o, jnp.zeros_like(o))


# ---------------------------------------------------------------------------
# General pipeline
# ---------------------------------------------------------------------------


def _l2norm(x, arch):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    T, D = x2.shape
    if D > 512:
        raise ValueError(f"KDA l2norm supports head dims up to 512, got {D}.")
    cfg = kda_launch_config("l2norm", arch)
    BT = cfg.kwargs["BT"]
    y = _call(
        _kda_l2norm_kernel,
        [("X", x2)],
        [("Y", x2.shape, x2.dtype)],
        (triton.cdiv(T, BT),),
        {"eps": 1e-6, "T": T, "D": D, "BD": triton.next_power_of_2(D), "BT": BT},
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )["Y"]
    return y.reshape(shape)


def _beta_sigmoid(x, arch):
    n = x.size
    cfg = kda_launch_config("beta_sigmoid", arch)
    bs = cfg.kwargs["BLOCK_SIZE"]
    return _call(
        _kda_beta_sigmoid_kernel,
        [("x", x)],
        [("y", x.shape, jnp.float32)],
        (triton.cdiv(n, bs),),
        {"n_elements": n, "BLOCK_SIZE": bs},
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )["y"]


def _general_fwd(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    initial_state,
    cu_seqlens,
    *,
    scale,
    output_final_state,
    chunk_size,
    safe_gate,
    lower_bound,
    use_gate_in_kernel,
    use_qk_l2norm_in_kernel,
    use_beta_sigmoid_in_kernel,
    state_v_first,
    arch,
):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    BC = KDA_SUB_CHUNK
    NC = triton.cdiv(BT, BC)
    is_varlen = cu_seqlens is not None
    N = cu_seqlens.shape[0] - 1 if is_varlen else B
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

    if use_gate_in_kernel:
        cfg = kda_launch_config("gate_cumsum", arch)
        BS = cfg.kwargs["BS"]
        g_cumsum = _call(
            _kda_gate_cumsum_kernel,
            [
                ("s", g),
                ("A_log", A_log),
                ("dt_bias", dt_bias),
                ("cu_seqlens", cu_seqlens),
                ("chunk_indices", chunk_indices),
            ],
            [("o", g.shape, jnp.float32)],
            (triton.cdiv(K, BS), NT, B * HV),
            {
                "scale": RCP_LN2,
                "lower_bound": 0.0 if lower_bound is None else float(lower_bound),
                "T": T,
                "H": HV,
                "S": K,
                "BT": BT,
                "BS": BS,
                "HAS_BIAS": dt_bias is not None,
                "HAS_SCALE": True,
                "IS_VARLEN": is_varlen,
                "USE_LOWER_BOUND": lower_bound is not None,
            },
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )["o"]
    else:
        cfg = kda_launch_config("local_cumsum", arch)
        BS = cfg.kwargs["BS"]
        g_cumsum = _call(
            _kda_local_cumsum_kernel,
            [("s", g), ("cu_seqlens", cu_seqlens), ("chunk_indices", chunk_indices)],
            [("o", g.shape, jnp.float32)],
            (triton.cdiv(K, BS), NT, B * HV),
            {
                "scale": RCP_LN2,
                "T": T,
                "H": HV,
                "S": K,
                "BT": BT,
                "BS": BS,
                "HAS_SCALE": True,
                "IS_VARLEN": is_varlen,
            },
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )["o"]

    a_shape = (B, T, HV, BT)
    common_qk = [("q", q), ("k", k), ("g", g_cumsum), ("beta", beta)]
    if safe_gate:
        cfg = kda_launch_config("intra_sub_chunk", arch)
        r = _call(
            _kda_intra_sub_chunk_kernel,
            common_qk + [("cu_seqlens", cu_seqlens), ("chunk_indices", chunk_indices)],
            [("Aqk", a_shape, k.dtype), ("Akk", (B, T, HV, BC), jnp.float32)],
            (NT, NC, B * HV),
            {
                "scale": scale,
                "T": T,
                "H": H,
                "HV": HV,
                "K": K,
                "BT": BT,
                "BC": BC,
                "BK": min(64, triton.next_power_of_2(K)),
                "IS_VARLEN": is_varlen,
            },
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    else:
        cfg = kda_launch_config("intra_token_parallel", arch)
        BH = cfg.kwargs["BH"]
        r = _call(
            _kda_intra_token_parallel_kernel,
            common_qk + [("cu_seqlens", cu_seqlens)],
            [("Aqk", a_shape, k.dtype), ("Akk", (B, T, HV, BC), jnp.float32)],
            (B * T, triton.cdiv(HV, BH)),
            {
                "scale": scale,
                "N": N,
                "T": T,
                "H": H,
                "HV": HV,
                "K": K,
                "BT": BT,
                "BC": BC,
                "BH": BH,
                "BK": cfg.kwargs["BK"],
                "IS_VARLEN": is_varlen,
            },
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    Aqk_diag, Akkd = r["Aqk"], r["Akk"]

    cfg = kda_launch_config("inter_solve", arch)
    r = _call(
        _kda_inter_solve_kernel,
        common_qk
        + [
            ("Akkd", Akkd),
            ("Aqk_diag", Aqk_diag),
            ("cu_seqlens", cu_seqlens),
            ("chunk_indices", chunk_indices),
        ],
        [("Aqk", a_shape, k.dtype), ("Akk", a_shape, k.dtype)],
        (NT, B * HV),
        {
            "scale": scale,
            "T": T,
            "H": H,
            "HV": HV,
            "K": K,
            "BT": BT,
            "BC": BC,
            "NC": NC,
            "BK": cfg.kwargs["BK"],
            "IS_VARLEN": is_varlen,
            "USE_SAFE_GATE": safe_gate,
        },
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        aliases={"Aqk_diag": "Aqk"},
    )
    Aqk, Akk = r["Aqk"], r["Akk"]

    cfg = kda_launch_config("recompute_w_u", arch)
    r = _call(
        _kda_recompute_w_u_kernel,
        [
            ("k", k),
            ("v", v),
            ("beta", beta),
            ("A", Akk),
            ("gk", g_cumsum),
            ("cu_seqlens", cu_seqlens),
            ("chunk_indices", chunk_indices),
        ],
        [
            ("w", (B, T, HV, K), k.dtype),
            ("u", v.shape, v.dtype),
            ("kg", (B, T, HV, K), k.dtype),
        ],
        (NT, B * HV),
        {
            "T": T,
            "H": H,
            "HV": HV,
            "K": K,
            "V": V,
            "BT": BT,
            "BK": cfg.kwargs["BK"],
            "BV": cfg.kwargs["BV"],
            "IS_VARLEN": is_varlen,
        },
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    w, u, kg = r["w"], r["u"], r["kg"]

    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    if initial_state is not None and tuple(initial_state.shape) != state_shape:
        raise ValueError(
            f"`initial_state` must have shape {state_shape} for "
            f"state_v_first={state_v_first}, got {tuple(initial_state.shape)}."
        )
    cfg = kda_launch_config("fwd_h", arch)
    BV = cfg.kwargs["BV"]
    r = _call(
        _kda_fwd_h_kernel,
        [
            ("k", kg),
            ("v", u),
            ("w", w),
            ("gk", g_cumsum),
            ("h0", initial_state),
            ("cu_seqlens", cu_seqlens),
            ("chunk_offsets", chunk_offsets),
        ],
        [
            ("h", (B, NT, HV, K, V), k.dtype),
            ("v_new", u.shape, u.dtype),
            ("ht", state_shape if output_final_state else None, jnp.float32),
        ],
        (triton.cdiv(V, BV), N * HV),
        {
            "T": T,
            "H": HV,
            "K": K,
            "V": V,
            "BT": BT,
            "BV": BV,
            "USE_INITIAL_STATE": initial_state is not None,
            "STORE_FINAL_STATE": output_final_state,
            "IS_VARLEN": is_varlen,
            "TRANSPOSE_STATE": state_v_first,
        },
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    h, v_new, final_state = r["h"], r["v_new"], r.get("ht")

    cfg = kda_launch_config("gla_fwd_o", arch)
    BV = cfg.kwargs["BV"]
    o = _call(
        _kda_gla_fwd_o_kernel,
        [
            ("q", q),
            ("v", v_new),
            ("g", g_cumsum),
            ("h", h),
            ("A", Aqk),
            ("cu_seqlens", cu_seqlens),
            ("chunk_indices", chunk_indices),
        ],
        [("o", v.shape, v.dtype)],
        (triton.cdiv(V, BV), NT, B * HV),
        {
            "scale": scale,
            "T": T,
            "H": H,
            "HV": HV,
            "K": K,
            "V": V,
            "BT": BT,
            "BK": cfg.kwargs["BK"],
            "BV": BV,
            "IS_VARLEN": is_varlen,
        },
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )["o"]
    return o, final_state


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
    initial_state,
    cu_seqlens,
    *,
    scale,
    lower_bound,
    output_final_state,
    state_v_first,
    max_seqlen,
    arch,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = FLASH_KDA_CHUNK
    BC = min(FLASH_KDA_INV_BLOCK, C)
    num_cus = kda_num_cus(0)
    is_varlen = cu_seqlens is not None

    if is_varlen:
        N = cu_seqlens.shape[0] - 1
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
        desc, seq_seg_off, num_segs = _fixed_segments(B, T, C, chunks_per_seg if segmented else NT)
    seg_rows = {
        "seg_chunk_base": desc[0],
        "seg_nchunks": desc[1],
        "seg_tok_base": desc[2],
        "seg_tok_end": desc[3],
        "seg_seq": desc[4],
        "seg_is_last": desc[5],
    }

    CM_LOAD = ".cg"
    CM_STORE = "" if segmented else ".wt"
    CM_OUT_STORE = "" if segmented else ".cs"

    ws_shape = (H * total_tiles, C, K)
    ws_outputs = [
        ("ws_kd", ws_shape, jnp.bfloat16),
        ("ws_qd", ws_shape, jnp.bfloat16),
        ("ws_kr", ws_shape, jnp.bfloat16),
        ("ws_gt", (H * total_tiles, K), jnp.float32),
        ("ws_inv_mqk", (H * total_tiles, 2 * C, C), jnp.float16),
    ]
    k1_inputs = [
        ("q", q),
        ("k", k),
        ("g_raw", g),
        ("beta_raw", beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("cu_seqlens", cu_seqlens),
        ("chunk_indices", chunk_indices),
    ]
    k1_consts = {
        "scale": scale,
        "lower_bound": lower_bound,
        "T": T,
        "NT": NT,
        "TOTAL_TILES": total_tiles,
        "H": H,
        "K": K,
        "C": C,
        "BC": BC,
        "IS_VARLEN": is_varlen,
        "HAS_BIAS": dt_bias is not None,
    }
    k1_grid = (total_tiles if is_varlen else NT, B * H)
    if _use_gluon("k1", arch):
        kg = _gluon_module()
        ws = _call(
            kg.flash_kda_k1_prepare_gluon,
            k1_inputs,
            ws_outputs,
            k1_grid,
            dict(k1_consts, CM_WS=CM_STORE, CM_LOAD=CM_LOAD),
            num_warps=kg.K1_NUM_WARPS,
            num_stages=kda_launch_config("flash_gluon_k1", arch).num_stages,
        )
    else:
        cfg = kda_launch_config("flash_prepare", arch)
        ws = _call(
            _flash_kda_prepare_kernel,
            k1_inputs,
            ws_outputs,
            k1_grid,
            dict(
                k1_consts,
                NUM_DOUBLING=BC.bit_length() - 2,
                NUM_MERGE=(C // BC).bit_length() - 1,
                CM_QKG=CM_LOAD,
                CM_WS=CM_STORE,
            ),
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    h0 = initial_state
    if h0 is not None and state_v_first:
        h0 = jnp.swapaxes(h0, -1, -2)

    state_shape = (N, H, V, K) if state_v_first else (N, H, K, V)
    state_dtype = h0.dtype if h0 is not None else jnp.float32

    seg_cfg = kda_launch_config("flash_segment", arch)
    BW = seg_cfg.kwargs["BW"]
    ws_inputs = [(n, ws[n]) for n in ("ws_kd", "ws_qd", "ws_kr", "ws_gt", "ws_inv_mqk")]
    seg_inputs = list(seg_rows.items())

    def _k2(*, W, h_in, outputs, **flags):
        return _call(
            _flash_kda_segment_kernel,
            ws_inputs + [("v_input", v), ("beta_raw", beta), ("h_in", h_in)] + seg_inputs,
            outputs,
            (triton.cdiv(W, BW), num_segs * H),
            {
                "TOTAL_TILES": total_tiles,
                "H": H,
                "K": K,
                "V": V,
                "W": W,
                "C": C,
                "BW": BW,
                "STATE_V_FIRST": state_v_first,
                "CM_OUT": CM_OUT_STORE,
                **flags,
            },
            num_warps=seg_cfg.num_warps,
            num_stages=seg_cfg.num_stages,
        )

    if segmented:
        seg_state = (num_segs, H, K, V)
        seg_op = (num_segs, H, K, K)
        if _use_gluon("k2", arch):
            bw, nw, ns = flash_kda_gluon_k2_schedule(V, num_segs, H, arch, num_cus)
            r = _call(
                _gluon_module().flash_kda_k2_ab_fused_gluon,
                [
                    ("ws_kd", ws["ws_kd"]),
                    ("ws_kr", ws["ws_kr"]),
                    ("ws_gt", ws["ws_gt"]),
                    ("ws_inv_mqk", ws["ws_inv_mqk"]),
                    ("v_input", v),
                    ("beta_raw", beta),
                ]
                + [
                    (n, seg_rows[n])
                    for n in ("seg_chunk_base", "seg_nchunks", "seg_tok_base", "seg_tok_end")
                ],
                [("h_out_b", seg_state, jnp.float32), ("h_out_a", seg_op, jnp.bfloat16)],
                (triton.cdiv(V, bw), num_segs * H),
                {"TOTAL_TILES": total_tiles, "H": H, "K": K, "V": V, "C": C, "BW": bw},
                num_warps=nw,
                num_stages=ns,
                gluon_k2_warps=nw,
            )
            b_seg, A_seg = r["h_out_b"], r["h_out_a"]
        else:
            flags = {
                "HAS_H_IN": False,
                "COMPUTE_OUTPUT": False,
                "STORE_H_OUT": True,
                "STORE_FINAL": False,
            }
            b_seg = _k2(
                W=V,
                h_in=None,
                outputs=[("out", None), ("h_out", seg_state, jnp.float32), ("final_state", None)],
                INIT_IDENTITY=False,
                HAS_V=True,
                **flags,
            )["h_out"]
            A_seg = _k2(
                W=K,
                h_in=None,
                outputs=[("out", None), ("h_out", seg_op, jnp.bfloat16), ("final_state", None)],
                INIT_IDENTITY=True,
                HAS_V=False,
                **flags,
            )["h_out"]

        BV_SCAN, SCAN_WARPS = flash_kda_scan_bv(N, H, V, num_cus)
        h_in = _call(
            _flash_kda_seg_scan_kernel,
            [("A_seg", A_seg), ("b_seg", b_seg), ("h0", h0), ("seq_seg_off", seq_seg_off)],
            [("h_in", seg_state, jnp.float32)],
            (triton.cdiv(V, BV_SCAN), N * H),
            {"H": H, "K": K, "V": V, "BV": BV_SCAN, "HAS_H0": h0 is not None},
            num_warps=SCAN_WARPS,
            num_stages=kda_launch_config("flash_seg_scan", arch).num_stages,
        )["h_in"]
    else:
        h_in = h0

    r = _k2(
        W=V,
        h_in=h_in,
        outputs=[
            ("out", v.shape, v.dtype),
            ("h_out", None),
            ("final_state", state_shape if output_final_state else None, state_dtype),
        ],
        INIT_IDENTITY=False,
        HAS_H_IN=h_in is not None,
        HAS_V=True,
        COMPUTE_OUTPUT=True,
        STORE_H_OUT=False,
        STORE_FINAL=output_final_state,
    )
    return r["out"], r.get("final_state")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,
    dt_bias=None,
    initial_state=None,
    cu_seqlens=None,
    *,
    scale: float,
    output_final_state: bool = False,
    max_seqlen: Optional[int] = None,
    chunk_size: Optional[int] = None,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    state_v_first: bool = False,
):
    """KDA forward, choosing between FlashKDA and the general pipeline.

    Takes and returns ``jax.Array``s; see ``transformer_engine.jax.kda`` for the
    public, validated entry point.
    """
    H, K = q.shape[2], q.shape[3]
    HV, V = v.shape[2], v.shape[-1]
    arch = kda_device_arch(0)
    use_flash = _use_flash() and flash_kda_supported(
        K=K,
        V=V,
        H=H,
        HV=HV,
        qv_bf16=q.dtype == jnp.bfloat16 and v.dtype == jnp.bfloat16,
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

    if use_flash:
        o, final_state = _flash_fwd(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            scale=scale,
            lower_bound=float(lower_bound),
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            max_seqlen=max_seqlen,
            arch=arch,
        )
    else:
        o, final_state = _general_fwd(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            scale=scale,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            state_v_first=state_v_first,
            arch=arch,
        )
    if cu_seqlens is not None:
        o = _mask_tail(o, cu_seqlens)
    return o, final_state
