# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""TE entry points for the FlyDSL sparse-MLA attention backend."""

import os

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch.utils import get_device_compute_capability

from ..exceptions import FlyDSLUnsupportedError
from ..version import _check_flydsl_version

# Kernel contract constants. Mirrored here so the whole request can be validated
# before importing the kernel module (which imports FlyDSL at import time).
# Keep in sync with ``sparse_mla_fwd.py``.
D_LATENT = 512  # kv_lora_rank; also the V width and the output width
D_QK = 576  # Q / KV row stride
TOPK_GRANULARITY = 32  # topk must be a multiple of this (pad with -1)

# The kernel is written against the gfx950 (MI355X) ISA: ds_read_tr16_b64,
# permlane16/32_swap and the K=32 bf16 MFMAs are all CDNA4-only.
_REQUIRED_CAPABILITY = (9, 5)


def _env_flag(name: str, default: str) -> bool:
    """Parse a boolean env var using the same spelling TE's GEMM path accepts."""
    return os.environ.get(name, default).lower() not in ("", "0", "false", "no", "off")


def select_kernel_config(total_tokens, num_heads, num_kv, topk):
    """Choose the kernel specialization knobs for one problem shape.

    Ported verbatim from the upstream Primus-Turbo host dispatcher. This is a
    baked-in autotuner: the branches encode an offline search over the
    ``{flash,pro} x {cr0,cr4,cr128}`` shape families, so the ordering and the
    exact thresholds are load-bearing -- do not "simplify" them.

    Pure Python, no FlyDSL import, so callers can validate against ``bh`` before
    committing to the kernel import.
    """
    # cr=0 (pure SWA): num_kv==total_tokens and topk==W=128 -> closed-form banded
    # KV (token i attends the contiguous window [i-127..i]). Skips the topk HBM
    # load + scatter and loads KV contiguously. Bit-exact (kv=i-127+rank).
    banded = (num_kv == total_tokens) and (topk == 128)

    # cr128 HCA: num_kv = S + P pool rows (closed-form causal pool, topk[p]
    # visible iff token >= cr*p+(cr-1), cr = S/P). Folds the whole topk into a
    # closed-form SWA[128]+pool[P] band. Gate cr = S/P >= 64 to exclude cr4's
    # random pool.
    pool_P, pool_cr = 0, 0
    P = num_kv - total_tokens
    if (
        not banded
        and P > 0
        and total_tokens % P == 0
        and (total_tokens // P) >= 64
        and topk >= 128 + P
    ):
        banded, pool_P, pool_cr = True, P, total_tokens // P

    # QK 2-accumulator: splits the 16-deep QK MFMA RAW chain into two 8-deep
    # chains. Only the few-tile banded groups (topk<=256); cr4 hides the bubble
    # across tiles.
    qk2 = topk <= 256

    # PV tr16 prefetch depth: flash cr4 depth 7 (pstore frees VGPR), pro cr4
    # depth 2 (LDS-port bound), banded few-tile depth 5 (under the occ-2 VGPR
    # cliff).
    if topk > 256 and num_heads == 64:
        pf_pv = 7
    elif topk > 256:
        pf_pv = 2
    else:
        pf_pv = 5

    # single_buffer: pro many-tile gather only (extra WAR barrier is free there).
    single_buffer = topk > 256 and num_heads >= 128

    # XCD-aware token remap: contiguous token block per XCD so overlapping banded
    # SWA windows share L2. pro uniform-work groups (pool_P==0) only; T % 8 == 0.
    xcd_remap = 1 if (total_tokens % 8 == 0 and num_heads >= 128 and pool_P == 0) else 0

    # peel_last: unroll the last KV tile after the scf.for so the epilogue scalar
    # chain interleaves with its PV MFMA drain. Banded groups only.
    peel = banded

    # hoist_sink: issue the loop-invariant sink VMEM load at kernel top. Only
    # flash cr0 (register-tighter H=128/pool groups don't amortize holding it).
    hoist_sink = banded and pool_P == 0 and num_heads < 128

    # fused_store: spread the O stores across the last tile's PV MFMA drain.
    fused_store = peel

    # pv_early: hoist the PV tr16 reads ahead of the softmax permlane bridge.
    pv_early = banded and (pool_P > 0 or num_heads >= 128)

    # pf_qk: QK bv prefetch depth. flash cr4 5 (VGPR-free under pstore),
    # flash cr128 4, pro cr0 2 (grid/LDS-contention bound), rest 3.
    if not banded and topk > 256 and num_heads == 64:
        pf_qk = 5
    elif pool_P > 0 and num_heads < 128:
        pf_qk = 4
    elif banded and pool_P == 0 and num_heads >= 128:
        pf_qk = 2
    else:
        pf_qk = 3

    # dyn_pool: skip fully-masked cr128 pool tiles (bit-exact no-op).
    dyn_pool = pool_P > 0

    # bh=128: 1 WG/token (8 waves) for non-banded pro (H%128==0), collapsing the
    # two bh=64 WGs that redundantly gather+store the shared MLA latent.
    bh = 128 if (not banded and num_heads >= 128 and num_heads % 128 == 0) else 64

    # pstore: 4-buffer parity prestore pipeline overlapping the exposed KV
    # reg->LDS store with the PV MFMA drain. flash cr4 only.
    pstore = not banded and bh == 64

    return {
        "banded": banded,
        "qk2": qk2,
        "pool_P": pool_P,
        "pool_cr": pool_cr,
        "pf_pv": pf_pv,
        "single_buffer": single_buffer,
        "xcd_remap": xcd_remap,
        "peel": peel,
        "hoist_sink": hoist_sink,
        "fused_store": fused_store,
        "pv_early": pv_early,
        "pf_qk": pf_qk,
        "dyn_pool": dyn_pool,
        "bh": bh,
        "pstore": pstore,
    }


def _check_backend_available():
    """Raise unless this build/GPU can run the FlyDSL sparse-MLA kernel."""
    if not IS_HIP_EXTENSION:
        raise FlyDSLUnsupportedError("FlyDSL sparse-MLA attention is only supported on ROCm")

    capability = get_device_compute_capability()
    if capability != _REQUIRED_CAPABILITY:
        raise FlyDSLUnsupportedError(
            "FlyDSL sparse-MLA attention requires gfx950 (compute capability "
            f"{_REQUIRED_CAPABILITY}); this device reports {capability}"
        )

    if not bool(int(os.environ.get("NVTE_USE_FLYDSL", "0"))):
        raise FlyDSLUnsupportedError(
            "FlyDSL sparse-MLA attention is disabled; set NVTE_USE_FLYDSL=1 to enable it"
        )


def _validate(q, kv, topk_indices, attn_sink, kv_lora_rank):
    """Validate one request and return ``(total_tokens, num_heads, num_kv, topk, kv)``."""
    if q.dim() != 3:
        raise FlyDSLUnsupportedError(
            f"FlyDSL sparse-MLA expects q of rank 3 [tokens, heads, d_qk], got {tuple(q.shape)}"
        )
    if topk_indices.dim() != 2:
        raise FlyDSLUnsupportedError(
            "FlyDSL sparse-MLA expects topk_indices of rank 2 [tokens, topk], "
            f"got {tuple(topk_indices.shape)}"
        )

    total_tokens, num_heads, d_qk = q.shape
    topk = topk_indices.shape[1]

    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    if kv.dim() != 3:
        raise FlyDSLUnsupportedError(
            f"FlyDSL sparse-MLA expects kv of rank 2 or 3, got {tuple(kv.shape)}"
        )
    num_kv = kv.shape[0]

    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise FlyDSLUnsupportedError(
            "FlyDSL sparse-MLA is a bf16 kernel; got "
            f"q={q.dtype}, kv={kv.dtype}"
        )
    if topk_indices.dtype != torch.int32:
        raise FlyDSLUnsupportedError(
            f"FlyDSL sparse-MLA expects int32 topk_indices, got {topk_indices.dtype}"
        )
    if not (q.is_contiguous() and kv.is_contiguous() and topk_indices.is_contiguous()):
        raise FlyDSLUnsupportedError(
            "FlyDSL sparse-MLA requires contiguous q/kv/topk_indices, got strides "
            f"q={tuple(q.stride())}, kv={tuple(kv.stride())}, "
            f"topk_indices={tuple(topk_indices.stride())}"
        )
    if not (q.device == kv.device == topk_indices.device):
        raise FlyDSLUnsupportedError(
            "FlyDSL sparse-MLA requires q/kv/topk_indices on one device, got "
            f"q={q.device}, kv={kv.device}, topk_indices={topk_indices.device}"
        )

    if kv_lora_rank != D_LATENT or d_qk != D_QK:
        raise FlyDSLUnsupportedError(
            f"FlyDSL sparse-MLA is fixed to kv_lora_rank={D_LATENT} and d_qk={D_QK}, "
            f"got kv_lora_rank={kv_lora_rank}, d_qk={d_qk}"
        )
    if kv.shape[-1] != D_QK:
        raise FlyDSLUnsupportedError(
            f"FlyDSL sparse-MLA expects kv last dim {D_QK}, got {kv.shape[-1]}"
        )
    if topk % TOPK_GRANULARITY != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL sparse-MLA requires topk to be a multiple of {TOPK_GRANULARITY} "
            f"(pad with -1), got topk={topk}"
        )

    # Upstream asserts ``num_heads % 32 == 0``, which is too weak: the launch
    # computes ``grid.y = num_heads // bh`` with bh in {64,128}, so num_heads=32
    # yields a zero-size grid dimension and hipErrorInvalidValue at launch.
    bh = select_kernel_config(total_tokens, num_heads, num_kv, topk)["bh"]
    if num_heads % bh != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL sparse-MLA selected head-block bh={bh} for this shape, so num_heads "
            f"must be a multiple of {bh}; got num_heads={num_heads}"
        )

    if attn_sink is not None:
        if attn_sink.dtype != torch.float32:
            raise FlyDSLUnsupportedError(
                f"FlyDSL sparse-MLA expects a float32 attn_sink, got {attn_sink.dtype}"
            )
        if tuple(attn_sink.shape) != (num_heads,):
            raise FlyDSLUnsupportedError(
                f"FlyDSL sparse-MLA expects attn_sink of shape ({num_heads},), "
                f"got {tuple(attn_sink.shape)}"
            )

    return total_tokens, num_heads, num_kv, topk, kv


def sparse_mla_attn_fwd(
    q,
    kv,
    topk_indices,
    attn_sink=None,
    kv_lora_rank=D_LATENT,
    scale=None,
):
    """Sparse multi-head latent attention forward on gfx950.

    Args:
        q: ``[total_tokens, num_heads, 576]`` bf16, contiguous.
        kv: ``[num_kv, 576]`` or ``[num_kv, 1, 576]`` bf16, contiguous. Serves as
            both K (scored dims) and V (first 512 dims) -- MLA shares one latent.
        topk_indices: ``[total_tokens, topk]`` int32, contiguous. ``topk`` must be
            a multiple of 32; pad unused slots with ``-1``. Ignored entirely by
            the closed-form banded path (see below).
        attn_sink: optional ``[num_heads]`` float32 per-head attention sink.
        kv_lora_rank: must be 512.
        scale: softmax scale; defaults to ``1/sqrt(576)``.

    Returns:
        ``(output, lse)`` with ``output`` ``[total_tokens, num_heads, 512]`` in
        q's dtype and ``lse`` ``[total_tokens, num_heads]`` float32. With
        ``attn_sink``, ``lse`` is sink-inclusive (the sink term is part of the
        softmax denominator it logs).

    .. warning::

        **Scores cover only dims [0, 512).** ``d_qk == 576`` is a storage stride;
        the trailing 64 dims -- the decoupled RoPE sub-head in standard MLA --
        are read by neither Q nor KV. The computed score is
        ``q[..., :512] @ kv[..., :512].T``. Callers must fold RoPE into the
        512-wide latent themselves; this is **not** a drop-in MLA forward.

    Three sparsity modes are selected automatically from the shapes:

    - **gather**: general topk, reads indices from HBM.
    - **banded**: ``num_kv == total_tokens and topk == 128`` -> closed-form
      sliding window ``[t-127, t]``. **``topk_indices`` is not read at all.**
    - **pool (cr128 HCA)**: ``num_kv = S + P`` with compression ratio
      ``S/P >= 64`` -> closed-form ``SWA[128] + pool[P]`` band.

    Raises:
        FlyDSLUnsupportedError: the build, GPU, flydsl version or request shape
            is unsupported.
            There is no C++ sparse-MLA backend to fall back to, so this is raised
            rather than swallowed.
    """
    _check_backend_available()
    total_tokens, num_heads, num_kv, topk, kv = _validate(
        q, kv, topk_indices, attn_sink, kv_lora_rank
    )

    if scale is None:
        scale = 1.0 / (D_QK**0.5)

    # Checked before the kernel import: an unsupported flydsl otherwise fails
    # there with an opaque missing-symbol ImportError.
    try:
        _check_flydsl_version("sparse-MLA attention")
    except ImportError as exc:
        raise FlyDSLUnsupportedError(str(exc)) from exc

    # Lazy import keeps FlyDSL off the normal Transformer Engine import path.
    from .sparse_mla_fwd import SparseMLASpec, get_compiled_fwd

    config = select_kernel_config(total_tokens, num_heads, num_kv, topk)
    spec = SparseMLASpec(
        topk_len=topk,
        scale=float(scale),
        has_sink=attn_sink is not None,
        # fast_path (first-pair fixed-max softmax) is ON upstream. It is exact
        # under shift-invariance but can overflow exp2 if later scores greatly
        # exceed the first pair's max, so expose an escape hatch.
        fast_path=_env_flag("NVTE_FLYDSL_SPARSE_MLA_FAST_PATH", "1"),
        **config,
    )

    if attn_sink is not None:
        sink = attn_sink.contiguous()
    else:
        sink = torch.empty(1, dtype=torch.float32, device=q.device)

    out = torch.empty(total_tokens, num_heads, D_LATENT, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    args = (
        q,
        kv,
        topk_indices,
        sink,
        out,
        lse,
        int(total_tokens),
        int(num_heads),
        int(num_kv),
        torch.cuda.current_stream(),
    )
    get_compiled_fwd(spec, args)(*args)
    return out, lse
