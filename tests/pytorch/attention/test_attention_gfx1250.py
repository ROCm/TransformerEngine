# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Gfx1250-targeted attention tests.

Configs mirror the smoke tests in:
  3rdparty/aiter/op_tests/cpp/mha/smoke_test_fwd_v3_gfx1250.sh  (fwd V3)

FWD V3 notes (fmha_fwd_gfx1250_batched / fmha_fwd_with_sink_asm):
  - Both D64 and D128 require a non-null sink_addr (fixed kernarg layout).
    TE supplies a static [256] fp32 buffer initialized to -1e30f so that
    exp(-1e30f) ≈ 0.0f adds no effective weight for non-fully-masked rows.
  - D64  (ENABLE_SINK=1): kernel reads and uses the sink values as a logit
    floor. Top-left causal; sq ≤ sk (rectangular) safe because even with
    sink≈0 the real attention weights dominate.
  - D128 (ENABLE_SINK=0): kernel ignores the sink values entirely.
    Causal (top-left or bottom-right); sq == sk only — rectangular shapes
    risk NaN on fully-masked KV tiles because the sink floor is disabled.
  - No SWA (window_size_left must be -1).

Each test forces CK V3 and compares against a pure-PyTorch scaled dot product
attention reference (torch.nn.functional.scaled_dot_product_attention) computed
in float32 for numerical stability.
"""
import os
import sys
import pathlib

import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch import DotProductAttention, get_device_compute_capability
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import (
    reset_rng_states,
    ModelConfig,
    get_available_attention_backends,
)

# Whole file is ROCm + gfx1250 only.
pytestmark = [
    pytest.mark.skipif(not IS_HIP_EXTENSION, reason="ROCm TE specific test."),
    pytest.mark.skipif(
        get_device_compute_capability() != (12, 5),
        reason="gfx1250 (compute capability 12.5) required.",
    ),
]

_SEED = 1234


def _make_rng_tracker():
    tracker = CudaRNGStatesTracker()
    tracker.add("model-parallel-rng", _SEED)
    return tracker


def _run_dpa(
    config: ModelConfig,
    backend_env: dict,
    dtype: torch.dtype,
    is_training: bool,
    q: torch.Tensor = None,
    k: torch.Tensor = None,
    v: torch.Tensor = None,
):
    """Run one forward (and optional backward) pass, return (out, grads).

    If q/k/v are provided they are used directly (requires_grad set appropriately);
    otherwise fresh random tensors are created.
    """
    reset_rng_states()

    for key in [
        "NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN",
        "NVTE_FUSED_ATTN_CK", "NVTE_FUSED_ATTN_AOTRITON",
        "NVTE_CK_USES_FWD_V3", "NVTE_CK_USES_BWD_V3",
    ]:
        os.environ.pop(key, None)
    for key, val in backend_env.items():
        os.environ[key] = str(val)
    _attention_backends["backend_selection_requires_update"] = True

    device = "cuda"
    b   = config.batch_size
    sq  = config.max_seqlen_q
    sk  = config.max_seqlen_kv
    hq  = config.num_heads
    hk  = config.num_gqa_groups
    dqk = config.head_dim_qk

    if q is None:
        q = torch.randn(b, sq, hq, dqk, dtype=dtype, device=device)
        k = torch.randn(b, sk, hk, dqk, dtype=dtype, device=device)
        v = torch.randn(b, sk, hk, dqk, dtype=dtype, device=device)

    q = q.detach().requires_grad_(is_training)
    k = k.detach().requires_grad_(is_training)
    v = v.detach().requires_grad_(is_training)

    block = DotProductAttention(
        num_attention_heads=hq,
        kv_channels=dqk,
        num_gqa_groups=hk,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type=config.attn_mask_type,
        tp_size=1,
        tp_group=None,
        get_rng_state_tracker=_make_rng_tracker,
    ).to(dtype=dtype, device=device)
    if not is_training:
        block.eval()

    out = block(
        q, k, v,
        qkv_format="bshd",
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
        max_seqlen_q=sq,
        max_seqlen_kv=sk,
    )

    grads = None
    if is_training:
        out.sum().backward()
        grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    return out.detach(), grads


def _pytorch_ref(
    config: ModelConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_training: bool,
) -> tuple:
    """Pure-PyTorch SDPA reference computed in float32.

    Inputs are BSHD; GQA heads are expanded before calling SDPA.
    Returns (out_bf16, (dq, dk, dv)) where dq/dk/dv are None when not training.
    The reference runs in float32 for numerical stability so that the comparison
    is against a correct high-precision result rather than a TE-specific backend
    that may itself have issues on gfx1250.
    """
    b, sq, hq, d = q.shape
    hk = k.shape[2]

    # Upcast to float32 for reference stability.
    qf = q.float()
    kf = k.float()
    vf = v.float()

    if is_training:
        qf = qf.detach().requires_grad_(True)
        kf = kf.detach().requires_grad_(True)
        vf = vf.detach().requires_grad_(True)

    # SDPA expects [B, H, S, D].
    qf_t = qf.permute(0, 2, 1, 3)    # [b, hq, sq, d]
    kf_t = kf.permute(0, 2, 1, 3)    # [b, hk, sk, d]
    vf_t = vf.permute(0, 2, 1, 3)

    # Expand KV heads for GQA so SDPA sees [b, hq, sk, d].
    # Use a separate leaf for the expanded tensor so grad flows back to hk heads.
    gqa = hk != hq
    if gqa:
        kf_t_exp = kf_t.repeat_interleave(hq // hk, dim=1).detach().requires_grad_(is_training)
        vf_t_exp = vf_t.repeat_interleave(hq // hk, dim=1).detach().requires_grad_(is_training)
    else:
        kf_t_exp = kf_t
        vf_t_exp = vf_t

    attn_mask_type = config.attn_mask_type
    window_size = config.window_size
    sk_len = kf_t.shape[2]

    # Use explicit mask for bottom-right causal or SWA; otherwise let SDPA handle it.
    has_swa = window_size is not None and window_size not in ((-1, -1), (-1, 0))
    needs_explicit_mask = attn_mask_type == "causal_bottom_right" or has_swa

    if needs_explicit_mask:
        # Start with all-attend, then apply causal + SWA constraints.
        rows = torch.arange(sq, device=q.device).unsqueeze(1)      # [sq, 1]
        cols = torch.arange(sk_len, device=q.device).unsqueeze(0)  # [1, sk]
        mask = torch.ones(sq, sk_len, dtype=torch.bool, device=q.device)
        if attn_mask_type in ("causal", "causal_bottom_right"):
            offset = sk_len - sq if attn_mask_type == "causal_bottom_right" else 0
            mask = mask & (cols <= rows + offset)
        if has_swa:
            left, right = window_size
            lo = (rows - left).clamp(min=0) if left >= 0 else torch.zeros_like(rows)
            hi = (rows + right) if right >= 0 else torch.full_like(rows, sk_len - 1)
            mask = mask & (cols >= lo) & (cols <= hi)
        float_mask = torch.zeros(sq, sk_len, dtype=torch.float32, device=q.device)
        float_mask[~mask] = float("-inf")
        out_t = F.scaled_dot_product_attention(qf_t, kf_t_exp, vf_t_exp, attn_mask=float_mask)
    elif attn_mask_type == "no_mask":
        out_t = F.scaled_dot_product_attention(qf_t, kf_t_exp, vf_t_exp, is_causal=False)
    else:
        # causal top-left: PyTorch is_causal=True is exactly this
        out_t = F.scaled_dot_product_attention(qf_t, kf_t_exp, vf_t_exp, is_causal=True)

    out_bf16 = out_t.permute(0, 2, 1, 3).to(dtype=q.dtype).detach()  # [b, sq, hq, d]

    dq = dk = dv = None
    if is_training:
        out_t.sum().backward()
        dq = qf.grad.to(dtype=q.dtype).detach()  # [b, sq, hq, d]
        if gqa:
            # kf_t_exp.grad: [b, hq, sk, d] — reduce over the hq/hk groups back to hk heads
            group = hq // hk
            dk_exp = kf_t_exp.grad.view(b, hk, group, sk_len, d).sum(dim=2)  # [b, hk, sk, d]
            dv_exp = vf_t_exp.grad.view(b, hk, group, sk_len, d).sum(dim=2)
            dk = dk_exp.permute(0, 2, 1, 3).to(dtype=k.dtype).detach()  # [b, sk, hk, d]
            dv = dv_exp.permute(0, 2, 1, 3).to(dtype=v.dtype).detach()
        else:
            dk = kf.grad.to(dtype=k.dtype).detach()  # [b, sk, hk, d]
            dv = vf.grad.to(dtype=v.dtype).detach()

    return out_bf16, (dq, dk, dv)


def _compare(config: ModelConfig, dtype: torch.dtype = torch.bfloat16, is_training: bool = True):
    """Run CK V3 and compare against a float32 PyTorch SDPA reference."""
    tols = dict(atol=2e-2, rtol=2e-2)

    _, _, fused_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout="bshd_bshd_bshd",
        is_training=is_training,
    )
    if FusedAttnBackend["CK"] not in fused_backends:
        pytest.skip("CK backend not available for this config")

    reset_rng_states()
    for _evar in [
        "NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN",
        "NVTE_FUSED_ATTN_CK", "NVTE_FUSED_ATTN_AOTRITON",
        "NVTE_CK_USES_FWD_V3", "NVTE_CK_USES_BWD_V3",
    ]:
        os.environ.pop(_evar, None)

    device = "cuda"
    b   = config.batch_size
    sq  = config.max_seqlen_q
    sk  = config.max_seqlen_kv
    hq  = config.num_heads
    hk  = config.num_gqa_groups
    dqk = config.head_dim_qk

    q = torch.randn(b, sq, hq, dqk, dtype=dtype, device=device)
    k = torch.randn(b, sk, hk, dqk, dtype=dtype, device=device)
    v = torch.randn(b, sk, hk, dqk, dtype=dtype, device=device)

    ref_out, ref_grads = _pytorch_ref(config, q, k, v, is_training)

    ck_v3_env = {
        "NVTE_FUSED_ATTN": "1",
        "NVTE_FLASH_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
        "NVTE_FUSED_ATTN_CK": "1",
        "NVTE_FUSED_ATTN_AOTRITON": "0",
        "NVTE_CK_USES_FWD_V3": "1",
        "NVTE_CK_USES_BWD_V3": "1",
    }
    ck_out, ck_grads = _run_dpa(config, ck_v3_env, dtype, is_training, q=q, k=k, v=v)

    # TE DotProductAttention returns [b, sq, hq*d] for bshd format; reshape to [b, sq, hq, d].
    ck_out = ck_out.view(b, sq, hq, dqk)

    torch.testing.assert_close(ck_out, ref_out, **tols)
    if is_training and ref_grads is not None:
        dq_ref, dk_ref, dv_ref = ref_grads
        dq_ck, dk_ck, dv_ck = ck_grads
        # grads from _run_dpa are already [b, s, h, d] since q/k/v were passed as BSHD leaves
        torch.testing.assert_close(dq_ck, dq_ref, **tols)
        torch.testing.assert_close(dk_ck, dk_ref, **tols)
        torch.testing.assert_close(dv_ck, dv_ref, **tols)


# ---------------------------------------------------------------------------
# FWD V3 D64 configs (smoke_test_fwd_sink.sh — run_d64)
#
# Kernel: fmha_fwd_with_sink_asm, ENABLE_SINK=1, bottom-right causal.
# Sink floor prevents div-by-zero, so sq < sk (rectangular) is safe.
# h=8, h_k∈{1,2,4}, b∈{1,2}.  FWD-only (no backward for this path).
#
# NOTE: the smoke test labels some cases "mha" but actually uses h_k=1
# (GQA-1 / MQA).  The TE test mirrors that: "mha" suffix = num_gqa_groups=1.
# The square cases (sq==sk) additionally exercise num_gqa_groups=8 (true MHA).
# ---------------------------------------------------------------------------

_fwd_v3_d64: dict = {}
# The gfx1250 ASM kernel implements bottom-right causal (mask_type 1 and 2 both
# map to is_causal=1 inside fmha_fwd_with_sink_asm, which follows bottom-right
# causal semantics).  Square shapes (sq==sk) are equivalent for both variants;
# rectangular shapes (sq < sk) must use "causal_bottom_right" to match.
for _s in (512, 1024, 2048):
    # h_k=1 matches smoke test "mha" label; also add true MHA (h_k=h=8) for square.
    _fwd_v3_d64[f"d64_sq{_s}_sk{_s}_b1_mqa"] = ModelConfig(
        1, _s, 8, 64, max_seqlen_kv=_s, num_gqa_groups=1, attn_mask_type="causal_bottom_right")
    _fwd_v3_d64[f"d64_sq{_s}_sk{_s}_b1_mha"] = ModelConfig(
        1, _s, 8, 64, max_seqlen_kv=_s, num_gqa_groups=8, attn_mask_type="causal_bottom_right")
    _fwd_v3_d64[f"d64_sq{_s}_sk{_s}_b2_gqa2"] = ModelConfig(
        2, _s, 8, 64, max_seqlen_kv=_s, num_gqa_groups=2, attn_mask_type="causal_bottom_right")
for _sq in (128, 256, 512):
    for _sk in (512, 2048):
        for _b in (1, 2):
            for _hq, _hk in ((8, 1), (8, 2), (4, 4)):
                _key = f"d64_sq{_sq}_sk{_sk}_b{_b}_h{_hq}k{_hk}"
                _fwd_v3_d64[_key] = ModelConfig(
                    _b, _sq, _hq, 64, max_seqlen_kv=_sk, num_gqa_groups=_hk,
                    attn_mask_type="causal_bottom_right")
# Rectangular tail configs: smoke test uses h_k=1 for these non-standard shapes.
for _sq in (130, 300):
    _fwd_v3_d64[f"d64_sq{_sq}_sk2048_b1_mha"] = ModelConfig(
        1, _sq, 8, 64, max_seqlen_kv=2048, num_gqa_groups=1, attn_mask_type="causal_bottom_right")
for _sk in (768, 2300):
    _fwd_v3_d64[f"d64_sq128_sk{_sk}_b1_mha"] = ModelConfig(
        1, 128, 8, 64, max_seqlen_kv=_sk, num_gqa_groups=1, attn_mask_type="causal_bottom_right")


@pytest.mark.parametrize("model", sorted(_fwd_v3_d64.keys()))
def test_gfx1250_fwd_v3_d64(model):
    """FWD V3 D64 correctness — run_d64 configs from smoke_test_fwd_v3_gfx1250.sh."""
    _compare(_fwd_v3_d64[model], dtype=torch.bfloat16, is_training=False)


# ---------------------------------------------------------------------------
# FWD V3 D128 configs (smoke_test_fwd_sink.sh — run_d128)
#
# Kernel: fmha_fwd_with_sink_asm, ENABLE_SINK=0 (sink_ptr ignored/nullptr).
# D128 uses bottom-right causal (kernel is_causal=1, same path as D64).
# h=8, h_k∈{1,2,4}, b∈{1,2}.  FWD-only.
#
# NOTE: smoke test "mha" cases use h_k=1 (MQA); true MHA (h_k=8) is added
# for square shapes only.
# ---------------------------------------------------------------------------

_fwd_v3_d128: dict = {}
for _s in (512, 1024, 2048):
    # h_k=1 matches smoke test; also add true MHA for square shapes.
    _fwd_v3_d128[f"d128_sq{_s}_sk{_s}_b1_mqa"] = ModelConfig(
        1, _s, 8, 128, max_seqlen_kv=_s, num_gqa_groups=1,
        attn_mask_type="causal_bottom_right")
    _fwd_v3_d128[f"d128_sq{_s}_sk{_s}_b1_mha"] = ModelConfig(
        1, _s, 8, 128, max_seqlen_kv=_s, num_gqa_groups=8,
        attn_mask_type="causal_bottom_right")
    _fwd_v3_d128[f"d128_sq{_s}_sk{_s}_b2_gqa2"] = ModelConfig(
        2, _s, 8, 128, max_seqlen_kv=_s, num_gqa_groups=2,
        attn_mask_type="causal_bottom_right")
# Rectangular configs from smoke test run_d128 (bottom-right causal, sq < sk safe).
for _sq in (128, 256):
    for _hq, _hk in ((8, 1), (8, 2), (4, 4)):
        _fwd_v3_d128[f"d128_sq{_sq}_sk2048_h{_hq}k{_hk}"] = ModelConfig(
            1, _sq, _hq, 128, max_seqlen_kv=2048, num_gqa_groups=_hk,
            attn_mask_type="causal_bottom_right")
# Unaligned tail configs: h_k=1 matching smoke test.
_fwd_v3_d128["d128_sq130_sk2048_b1_mha"] = ModelConfig(
    1, 130, 8, 128, max_seqlen_kv=2048, num_gqa_groups=1,
    attn_mask_type="causal_bottom_right")
_fwd_v3_d128["d128_sq128_sk2300_b1_mha"] = ModelConfig(
    1, 128, 8, 128, max_seqlen_kv=2300, num_gqa_groups=1,
    attn_mask_type="causal_bottom_right")


@pytest.mark.parametrize("model", sorted(_fwd_v3_d128.keys()))
def test_gfx1250_fwd_v3_d128(model):
    """FWD V3 D128 correctness — run_d128 configs from smoke_test_fwd_v3_gfx1250.sh."""
    _compare(_fwd_v3_d128[model], dtype=torch.bfloat16, is_training=False)
