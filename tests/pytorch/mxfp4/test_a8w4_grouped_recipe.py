# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Grouped MXFP4/a8w4 recipe path: reference bytes -> persistent Triton kernel.

Covers both the standalone pre-quantized entry
(:func:`grouped_gemm_mxfp4_fprop_prequantized`) and its end-to-end use inside
``te.GroupedLinear`` under ``CustomRecipe(a8w4_quantizer_factory)``.
"""

import gc

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from transformer_engine.pytorch.triton_kernels.grouped_gemm_mxfp4_impl import (
    _row_operand,
    _row_operand_mxfp8,
    _row_operand_mxfp8_torch,
    grouped_gemm_mxfp4_dgrad,
    grouped_gemm_mxfp4_fprop_prequantized,
)
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4 import (
    MXFP4_BLOCK_SIZE,
    e8m0_to_f32,
    mxfp4_to_f32,
)
from transformer_engine.pytorch.custom_recipes.quantization_mxfp4_grouped import (
    MXFP8E4M3QuantizerRef,
    _deq_ref_operand,
    _e2m1_ref,
    a8w4_quantizer_factory,
    mxfp4_grouped_quantizer_factory,
)

_recipe_ok, _reason = te.is_mxfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(not _recipe_ok, reason=_reason)

_REL_TOL = 5e-2


def _isolate():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _reference_grouped_a8w4(aq, weight_refs, m_splits, N):
    """Independent reference: dequant the same reference bytes, per-group fp32 matmul."""
    total_M = aq.data.shape[0]
    a_hp = _deq_ref_operand(aq.data, aq.scale, aq, default_e4m3=True)
    ref = torch.zeros(total_M, N, device=aq.data.device, dtype=torch.float32)
    start = 0
    for g, m in enumerate(m_splits):
        if m > 0:
            w_hp = _deq_ref_operand(
                weight_refs[g].data, weight_refs[g].scale, weight_refs[g], default_e4m3=False
            )
            ref[start : start + m] = a_hp[start : start + m] @ w_hp.t()
        start += m
    return ref


def test_grouped_prequantized_a8w4_matches_reference():
    """The kernel on reference bytes agrees with the reference qgemm."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    K, N = 256, 256  # K % 128 == 0 -> loop_k > 1
    m_splits = [128, 64, 0, 192]
    total_M = sum(m_splits)
    torch.manual_seed(0)

    a = torch.randn(total_M, K, device=device, dtype=dtype)
    weights = [torch.randn(N, K, device=device, dtype=dtype) for _ in m_splits]

    aq = MXFP8E4M3QuantizerRef(rowwise=True, columnwise=False).quantize(a)
    weight_refs = [_e2m1_ref(rowwise=True, columnwise=False).quantize(w) for w in weights]
    b_data = torch.stack([w.data for w in weight_refs], dim=0)
    b_scale = torch.stack([w.scale for w in weight_refs], dim=0)

    out = grouped_gemm_mxfp4_fprop_prequantized(
        aq.data, aq.scale, b_data, b_scale, m_splits, a_is_mxfp8=True, out_dtype=dtype
    )
    ref = _reference_grouped_a8w4(aq, weight_refs, m_splits, N)

    rel = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert out.shape == (total_M, N)
    assert rel < _REL_TOL, f"prequantized grouped a8w4 disagrees with reference: rel={rel:.4f}"


def test_grouped_linear_a8w4_recipe_forward():
    """te.GroupedLinear forward under CustomRecipe(a8w4) routes to the grouped Triton kernel."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    num_gemms, K, N = 4, 256, 256
    m_splits = [128, 64, 0, 192]
    total_M = sum(m_splits)
    torch.manual_seed(0)

    model = te.GroupedLinear(num_gemms, K, N, bias=False, params_dtype=dtype).cuda()
    inp = torch.randn(total_M, K, device=device, dtype=dtype)

    rec = recipe.CustomRecipe(qfactory=a8w4_quantizer_factory)
    with torch.no_grad(), te.autocast(enabled=True, recipe=rec):
        out = model(inp, m_splits)

    # Independent reference on the module's own weights, quantized by the same
    # reference quantizers the recipe factory produces.
    aq = MXFP8E4M3QuantizerRef(rowwise=True, columnwise=False).quantize(inp)
    weight_refs = [
        _e2m1_ref(rowwise=True, columnwise=False).quantize(getattr(model, f"weight{g}").detach())
        for g in range(num_gemms)
    ]
    ref = _reference_grouped_a8w4(aq, weight_refs, m_splits, N)

    rel = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert out.shape == (total_M, N)
    assert rel < _REL_TOL, f"GroupedLinear a8w4 forward disagrees with reference: rel={rel:.4f}"


def test_grouped_prequantized_a4w4_dgrad_matches_reference():
    """dgrad kernel on reference col-weight bytes agrees with the dequant reference."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    K, N = 256, 256
    m_splits = [128, 64, 0, 192]
    total_M = sum(m_splits)
    torch.manual_seed(0)
    grad_out = torch.randn(total_M, N, device=device, dtype=dtype)
    weights = [torch.randn(N, K, device=device, dtype=dtype) for _ in m_splits]

    # Reference col weight (data_t/scale_t)
    wrefs = [_e2m1_ref(rowwise=False, columnwise=True).quantize(w) for w in weights]
    w_col_data = torch.stack([w.data_t for w in wrefs], dim=0)  # [G, K, N/2]
    w_col_scale = torch.stack([w.scale_t for w in wrefs], dim=0)  # [G, K, N/32]

    grad_a = grouped_gemm_mxfp4_dgrad(
        grad_out, None, m_splits, out_dtype=dtype, weight_col=(w_col_data, w_col_scale)
    )

    # Reference: dequant the same operands (the kernel's native mxfp4 grad + the
    # reference col weight) and matmul in fp32. dX[g] = dY[g] @ W[g].
    gd, gs = _row_operand(grad_out)  # native mxfp4 grad bytes the kernel casts to
    g_hp = mxfp4_to_f32(gd) * e8m0_to_f32(gs).repeat_interleave(MXFP4_BLOCK_SIZE, dim=1)
    ref = torch.zeros(total_M, K, device=device, dtype=torch.float32)
    start = 0
    for g, m in enumerate(m_splits):
        if m > 0:
            w_t_hp = _deq_ref_operand(
                wrefs[g].data_t, wrefs[g].scale_t, wrefs[g], default_e4m3=False
            )  # [K, N] == W^T
            ref[start : start + m] = g_hp[start : start + m] @ w_t_hp.t()
        start += m

    rel = (grad_a.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert grad_a.shape == (total_M, K)
    assert rel < _REL_TOL, f"a4w4 dgrad on reference col weight disagrees: rel={rel:.4f}"


def test_a4w4_activation_native_matches_reference():
    """
    The a4w4 module activation uses the native GPU MXFP4 quantizer; its dequantized values must match the reference CPU quantizer.
    """
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    M, K = 384, 256
    torch.manual_seed(0)
    x = torch.randn(M, K, device=device, dtype=dtype)

    d_nat, s_nat = _row_operand(x)  # native GPU path
    hp_nat = mxfp4_to_f32(d_nat) * e8m0_to_f32(s_nat).repeat_interleave(MXFP4_BLOCK_SIZE, dim=1)

    ref = _e2m1_ref(rowwise=True, columnwise=False).quantize(x)  # pure-NumPy reference
    hp_ref = _deq_ref_operand(ref.data, ref.scale, ref, default_e4m3=False)

    rel = (hp_nat - hp_ref).norm() / hp_ref.norm().clamp_min(1e-12)
    assert rel < _REL_TOL, f"native a4w4 activation disagrees with reference: rel={rel:.4f}"


def test_grouped_linear_a4w4_recipe_backward():
    """te.GroupedLinear a4w4 fwd+bwd routes to the low-precision dgrad/wgrad kernels."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    num_gemms, K, N = 4, 256, 256
    m_splits = [128, 64, 0, 192]
    total_M = sum(m_splits)
    torch.manual_seed(0)

    model = te.GroupedLinear(num_gemms, K, N, bias=False, params_dtype=dtype).cuda()
    inp = torch.randn(total_M, K, device=device, dtype=dtype, requires_grad=True)

    rec = recipe.CustomRecipe(qfactory=mxfp4_grouped_quantizer_factory)
    with te.autocast(enabled=True, recipe=rec):
        out = model(inp, m_splits)
    out.backward(torch.randn_like(out))

    assert out.shape == (total_M, N)
    assert inp.grad is not None and inp.grad.shape == inp.shape
    assert torch.isfinite(inp.grad).all()
    for g in range(num_gemms):
        w = getattr(model, f"weight{g}")
        assert w.grad is not None and w.grad.shape == w.shape
        assert torch.isfinite(w.grad).all()


def test_grouped_linear_a8w4_backward_raises():
    """a8w4 is forward-only QAT: the grouped-path backward must raise clearly."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    num_gemms, K, N = 2, 256, 256
    m_splits = [128, 128]
    total_M = sum(m_splits)
    torch.manual_seed(0)

    model = te.GroupedLinear(num_gemms, K, N, bias=False, params_dtype=dtype).cuda()
    inp = torch.randn(total_M, K, device=device, dtype=dtype, requires_grad=True)

    rec = recipe.CustomRecipe(qfactory=a8w4_quantizer_factory)
    with te.autocast(enabled=True, recipe=rec):
        out = model(inp, m_splits)
    with pytest.raises(NotImplementedError, match="forward-only QAT"):
        out.backward(torch.randn_like(out))


def test_grouped_linear_a8w4_weight_cache_is_consistent():
    """is_first_microbatch weight caching must not change the a8w4 forward output."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    num_gemms, K, N = 4, 256, 256
    m_splits = [128, 64, 0, 192]
    total_M = sum(m_splits)
    torch.manual_seed(0)

    model = te.GroupedLinear(num_gemms, K, N, bias=False, params_dtype=dtype).cuda()
    inp = torch.randn(total_M, K, device=device, dtype=dtype)

    rec = recipe.CustomRecipe(qfactory=a8w4_quantizer_factory)
    with torch.no_grad(), te.autocast(enabled=True, recipe=rec):
        out_nocache = model(inp, m_splits)  # is_first_microbatch=None -> no caching
        out_first = model(inp, m_splits, is_first_microbatch=True)  # quantize + cache
        out_reuse = model(inp, m_splits, is_first_microbatch=False)  # reuse the cache

    assert torch.equal(out_first, out_reuse), "cached weights diverged from the first microbatch"
    assert torch.equal(out_nocache, out_first), "weight caching changed the forward output"


def test_row_operand_mxfp8_fused_matches_torch():
    """The fused MXFP8 e4m3 downcast is bit-identical to the torch reference."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    for M, K in [(384, 256), (130, 256), (293, 3584)]:
        x = torch.randn(M, K, device=device, dtype=dtype)
        d_ref, s_ref = _row_operand_mxfp8_torch(x)
        d_fused, s_fused = _row_operand_mxfp8(x)  # fused on gfx950
        assert torch.equal(s_fused, s_ref), f"E8M0 scales differ at {(M, K)}"
        assert torch.equal(
            d_fused.view(torch.uint8), d_ref.view(torch.uint8)
        ), f"e4m3 data bytes differ at {(M, K)}"
        # transposed-scale (a8w4 swizzle) path: [K/32, M], must equal the plain scale.T
        _, s_fused_t = _row_operand_mxfp8(x, transpose_scale=True)
        _, s_ref_t = _row_operand_mxfp8_torch(x, transpose_scale=True)
        assert torch.equal(s_fused_t, s_ref_t), f"transposed E8M0 scales differ at {(M, K)}"
        assert torch.equal(
            s_fused_t, s_ref.t().contiguous()
        ), f"transposed scale != plain scale.T at {(M, K)}"


def test_grouped_prequantized_empty_batch():
    """Empty MoE routing batch (total_M=0): the grouped path returns [0, N] with no
    zero-sized grid launch (the fused downcast and the GEMM both short-circuit)."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    G, K, N = 4, 256, 256
    m_splits = [0] * G
    weights = [torch.randn(N, K, device=device, dtype=dtype) for _ in range(G)]
    b_ops = [_row_operand(w) for w in weights]
    b_data = torch.stack([d for d, _ in b_ops])
    b_scale = torch.stack([s for _, s in b_ops])

    a = torch.empty(0, K, device=device, dtype=dtype)
    a_data, a_scale = _row_operand_mxfp8(a)  # fused downcast must skip the zero-sized grid
    assert a_data.shape == (0, K) and a_scale.numel() == 0

    out = grouped_gemm_mxfp4_fprop_prequantized(
        a_data, a_scale, b_data, b_scale, m_splits, a_is_mxfp8=True, out_dtype=dtype
    )
    assert out.shape == (0, N)


def test_grouped_prequantized_out_buffer():
    """grouped_gemm_mxfp4_fprop_prequantized writes into a caller-provided out buffer."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    K, N = 256, 256
    m_splits = [128, 64, 0, 192]
    total_M = sum(m_splits)
    torch.manual_seed(0)
    a = torch.randn(total_M, K, device=device, dtype=dtype)
    weights = [torch.randn(N, K, device=device, dtype=dtype) for _ in m_splits]

    aq = MXFP8E4M3QuantizerRef(rowwise=True, columnwise=False).quantize(a)
    weight_refs = [_e2m1_ref(rowwise=True, columnwise=False).quantize(w) for w in weights]
    b_data = torch.stack([w.data for w in weight_refs], dim=0)
    b_scale = torch.stack([w.scale for w in weight_refs], dim=0)

    ref = grouped_gemm_mxfp4_fprop_prequantized(
        aq.data, aq.scale, b_data, b_scale, m_splits, a_is_mxfp8=True, out_dtype=dtype
    )
    buf = torch.empty(total_M, N, device=device, dtype=dtype)
    got = grouped_gemm_mxfp4_fprop_prequantized(
        aq.data, aq.scale, b_data, b_scale, m_splits, a_is_mxfp8=True, out_dtype=dtype, out=buf
    )
    assert got.data_ptr() == buf.data_ptr(), "the caller's out buffer was not used"
    assert torch.equal(buf, ref), "out-buffer result differs from the allocating path"


def test_grouped_linear_a8w4_out_buffer():
    """te.GroupedLinear a8w4 honors a caller-provided out buffer in the fast path."""
    _isolate()
    device, dtype = "cuda", torch.bfloat16
    num_gemms, K, N = 4, 256, 256
    m_splits = [128, 64, 0, 192]
    total_M = sum(m_splits)
    torch.manual_seed(0)
    model = te.GroupedLinear(num_gemms, K, N, bias=False, params_dtype=dtype).cuda()
    inp = torch.randn(total_M, K, device=device, dtype=dtype)

    rec = recipe.CustomRecipe(qfactory=a8w4_quantizer_factory)
    with torch.no_grad(), te.autocast(enabled=True, recipe=rec):
        ref = model(inp, m_splits)
        buf = torch.empty(total_M, N, device=device, dtype=dtype)
        got = model(inp, m_splits, out=buf)

    assert torch.equal(buf, ref), "out buffer does not hold the forward result"
    assert torch.equal(got, ref), "returned output differs when an out buffer is given"
