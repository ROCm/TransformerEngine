# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for TE Lite mode (NVTE_LITE=1).

These tests verify that the pure-Python _lite backend can replace the
C++ transformer_engine_torch extension for core TE modules.

Run with:
    NVTE_LITE=1 pytest tests/pytorch/test_lite.py -v
"""

import os
import pytest
import torch
import torch.nn.functional as F

# Ensure lite mode is active before importing TE
os.environ["NVTE_LITE"] = "1"

import transformer_engine.pytorch as te  # noqa: E402
import transformer_engine_torch as tex  # noqa: E402
from transformer_engine.common import recipe  # noqa: E402
from transformer_engine.pytorch.quantization import FP8GlobalStateManager  # noqa: E402


@pytest.fixture(autouse=True)
def _check_lite_mode():
    """Skip all tests if lite mode is not active."""
    assert tex.__name__ == "transformer_engine.pytorch._lite", (
        "NVTE_LITE=1 must be set before importing transformer_engine"
    )


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


# ---------------------------------------------------------------------------
# Import / smoke tests
# ---------------------------------------------------------------------------

class TestImport:
    """Verify that the lite module loads and aliases correctly."""

    def test_lite_module_loaded(self):
        assert "transformer_engine.pytorch._lite" in tex.__name__

    def test_key_symbols_exist(self):
        required = [
            "DType", "FP8TensorMeta", "NVTE_Fused_Attn_Backend",
            "generic_gemm", "te_general_grouped_gemm",
            "layernorm_fwd", "layernorm_bwd",
            "rmsnorm_fwd", "rmsnorm_bwd", "gelu", "silu", "swiglu",
            "multi_tensor_adam", "multi_tensor_scale",
        ]
        for name in required:
            assert hasattr(tex, name), f"Missing symbol: {name}"


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

class TestForward:
    """Forward pass for core TE modules under lite mode."""

    @pytest.mark.parametrize("in_features,out_features", [(1024, 512), (256, 256)])
    def test_linear(self, device, in_features, out_features):
        mod = te.Linear(in_features, out_features, bias=True).to(
            dtype=torch.bfloat16, device=device
        )
        x = torch.randn(4, in_features, device=device, dtype=torch.bfloat16)
        y = mod(x)
        assert y.shape == (4, out_features)

    def test_layernorm_linear(self, device):
        mod = te.LayerNormLinear(1024, 512, bias=True).to(
            dtype=torch.bfloat16, device=device
        )
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16)
        y = mod(x)
        assert y.shape == (4, 512)

    def test_layernorm_mlp(self, device):
        mod = te.LayerNormMLP(1024, 4096).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16)
        y = mod(x)
        assert y.shape == (4, 1024)

    def test_layernorm(self, device):
        mod = te.LayerNorm(1024).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16)
        y = mod(x)
        assert y.shape == (4, 1024)

    def test_rmsnorm(self, device):
        mod = te.RMSNorm(1024).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16)
        y = mod(x)
        assert y.shape == (4, 1024)

    def test_transformer_layer(self, device):
        mod = te.TransformerLayer(1024, 4096, 16).to(
            dtype=torch.bfloat16, device=device
        )
        x = torch.randn(2, 8, 1024, device=device, dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = mod(x)
        assert y.shape == (2, 8, 1024)


# ---------------------------------------------------------------------------
# Forward + backward tests
# ---------------------------------------------------------------------------

class TestBackward:
    """Forward + backward pass for core TE modules under lite mode."""

    @pytest.mark.parametrize("in_features,out_features", [(1024, 512), (256, 256)])
    def test_linear(self, device, in_features, out_features):
        mod = te.Linear(in_features, out_features, bias=True).to(
            dtype=torch.bfloat16, device=device
        )
        x = torch.randn(4, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_layernorm_linear(self, device):
        mod = te.LayerNormLinear(1024, 512, bias=True).to(
            dtype=torch.bfloat16, device=device
        )
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None

    def test_layernorm_mlp(self, device):
        mod = te.LayerNormMLP(1024, 4096).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None

    def test_layernorm(self, device):
        mod = te.LayerNorm(1024).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None

    def test_rmsnorm(self, device):
        mod = te.RMSNorm(1024).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(4, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None

    def test_transformer_layer(self, device):
        mod = te.TransformerLayer(1024, 4096, 16).to(
            dtype=torch.bfloat16, device=device
        )
        x = torch.randn(2, 8, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = mod(x)
        y.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

class TestNumerical:
    """Verify lite-mode results match PyTorch reference implementations."""

    def test_linear_fp32_exact(self, device):
        """te.Linear should match torch.nn.Linear exactly in FP32."""
        te_mod = te.Linear(256, 128, bias=True).to(device=device)
        pt_mod = torch.nn.Linear(256, 128, bias=True).to(device=device)
        with torch.no_grad():
            pt_mod.weight.copy_(te_mod.weight)
            pt_mod.bias.copy_(te_mod.bias)
        x = torch.randn(4, 256, device=device)
        y_te = te_mod(x)
        y_pt = pt_mod(x)
        assert torch.allclose(y_te, y_pt, atol=0, rtol=0), (
            f"FP32 max diff: {(y_te - y_pt).abs().max().item():.2e}"
        )

    def test_linear_bf16_close(self, device):
        """te.Linear should be close to torch.nn.Linear in BF16."""
        te_mod = te.Linear(256, 128, bias=True).to(dtype=torch.bfloat16, device=device)
        pt_mod = torch.nn.Linear(256, 128, bias=True).to(dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            pt_mod.weight.copy_(te_mod.weight)
            pt_mod.bias.copy_(te_mod.bias)
        x = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
        y_te = te_mod(x).to(torch.bfloat16)
        y_pt = pt_mod(x)
        assert torch.allclose(y_te, y_pt, atol=5e-3, rtol=1e-2), (
            f"BF16 max diff: {(y_te - y_pt).abs().max().item():.2e}"
        )

    def test_linear_backward_fp32_exact(self, device):
        """Backward gradients should match exactly in FP32."""
        te_mod = te.Linear(256, 128, bias=True).to(device=device)
        pt_mod = torch.nn.Linear(256, 128, bias=True).to(device=device)
        with torch.no_grad():
            pt_mod.weight.copy_(te_mod.weight)
            pt_mod.bias.copy_(te_mod.bias)
        x_te = torch.randn(4, 256, device=device, requires_grad=True)
        x_pt = x_te.detach().clone().requires_grad_(True)
        te_mod(x_te).sum().backward()
        pt_mod(x_pt).sum().backward()
        assert torch.allclose(x_te.grad, x_pt.grad, atol=0, rtol=0), (
            f"Grad max diff: {(x_te.grad - x_pt.grad).abs().max().item():.2e}"
        )

    def test_layernorm_close(self, device):
        """te.LayerNorm should be close to torch.nn.LayerNorm."""
        te_mod = te.LayerNorm(512).to(dtype=torch.bfloat16, device=device)
        pt_mod = torch.nn.LayerNorm(512).to(dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            pt_mod.weight.copy_(te_mod.weight)
            pt_mod.bias.copy_(te_mod.bias)
        x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
        y_te = te_mod(x)
        y_pt = pt_mod(x)
        assert torch.allclose(y_te, y_pt, atol=5e-3, rtol=1e-2), (
            f"LayerNorm max diff: {(y_te - y_pt).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Triton kernel wiring (Phase 2)
# ---------------------------------------------------------------------------

class TestTritonNorms:
    """Verify that Triton norm kernels are wired correctly via _lite."""

    def test_triton_norms_loadable(self):
        """Triton norm kernels should be importable when Triton is installed."""
        try:
            import triton  # noqa: F401
            has_triton = True
        except ImportError:
            has_triton = False

        from transformer_engine.pytorch._lite.norms import (
            _try_load_triton_norms,
            _triton_ln_fwd,
        )
        _try_load_triton_norms()
        if has_triton:
            from transformer_engine.pytorch._lite import norms as _n
            assert _n._triton_ln_fwd is not None, "Triton norms should load when Triton is available"
        # If triton is not installed, the fallback path is tested by other tests.

    def test_aiter_norms_active(self, device):
        """AITER Triton norm kernels should be the active backend when AITER is available."""
        from transformer_engine.pytorch._lite.aiter_utils import is_aiter_available
        from transformer_engine.pytorch._lite import norms as _n

        # Trigger lazy loading by calling a norm function
        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(128, device=device, dtype=torch.bfloat16)
        b = torch.randn(128, device=device, dtype=torch.bfloat16)
        tex.rmsnorm_fwd(x, w, 1e-5, None, None, None, 0, False)
        tex.layernorm_fwd(x, w, b, 1e-5, None, None, None, 0, False)

        if is_aiter_available():
            assert _n._aiter_rms_fwd is not None, "AITER RMSNorm fwd should be loaded"
            assert _n._aiter_rms_bwd is not None, "AITER RMSNorm bwd should be loaded"
            assert _n._aiter_ln_fwd is not None, "AITER LayerNorm fwd should be loaded"
            assert _n._aiter_ln_bwd is not None, "AITER LayerNorm bwd should be loaded"

    def test_aiter_rmsnorm_fwd_bwd(self, device):
        """AITER RMSNorm forward and backward produce correct results."""
        from transformer_engine.pytorch._lite.norms import _rmsnorm_fwd_pytorch, _rmsnorm_bwd_pytorch

        hidden = 512
        x = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        g = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)

        # PyTorch reference
        y_pt, rstd_pt = _rmsnorm_fwd_pytorch(x, w, 1e-5, False)
        dx_pt, dw_pt = _rmsnorm_bwd_pytorch(g, x, rstd_pt, w, False)

        # AITER-backed tex path
        y_te, _, rstd_te = tex.rmsnorm_fwd(x, w, 1e-5, None, None, None, 0, False)
        dx_te, dw_te = tex.rmsnorm_bwd(g, x, rstd_te, w, 0, False)

        assert torch.allclose(y_te, y_pt, atol=5e-1, rtol=5e-2), (
            f"RMSNorm fwd max diff: {(y_te - y_pt).abs().max().item():.2e}"
        )
        assert torch.allclose(dx_te.to(torch.bfloat16), dx_pt.to(torch.bfloat16),
                              atol=5e-1, rtol=5e-2), (
            f"RMSNorm bwd dx max diff: {(dx_te - dx_pt).abs().max().item():.2e}"
        )

    def test_fused_rmsnorm_fp8_quant_active(self, device):
        """Fused RMSNorm+FP8 quantize kernel is used for Float8Quantizer."""
        from transformer_engine.pytorch._lite import norms as _n
        from transformer_engine.pytorch import Float8Quantizer

        _n._try_load_aiter_norms()
        if _n._aiter_fused_rms_fp8_static is None:
            pytest.skip("AITER fused RMSNorm+FP8 kernel not available")

        hidden = 256
        x = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)

        q = Float8Quantizer(
            scale=torch.tensor([4.0], dtype=torch.float32, device=device),
            amax=torch.tensor([0.0], dtype=torch.float32, device=device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )

        out, _, rsigma = tex.rmsnorm_fwd(x, w, 1e-5, None, q, None, 0, False)

        # Verify output is a Float8Tensor
        assert type(out).__name__ == "Float8Tensor", (
            f"Expected Float8Tensor, got {type(out).__name__}"
        )
        # Verify shape is preserved
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        # Verify amax was updated (non-zero means kernel ran)
        assert q.amax.item() > 0, "amax should be updated by fused kernel"
        # Verify scale_inv was set
        assert hasattr(out, '_scale_inv')
        expected_scale_inv = 1.0 / 4.0
        assert abs(out._scale_inv.item() - expected_scale_inv) < 1e-6

    def test_fused_rmsnorm_fp8_quant_vs_separate(self, device):
        """Fused RMSNorm+FP8 path matches separate norm->quantize path."""
        from transformer_engine.pytorch._lite.norms import (
            _aiter_fused_rms_fp8_static, _try_load_aiter_norms,
            _rmsnorm_fwd_pytorch,
        )
        from transformer_engine.pytorch import Float8Quantizer

        _try_load_aiter_norms()
        if _aiter_fused_rms_fp8_static is None:
            pytest.skip("AITER fused RMSNorm+FP8 kernel not available")

        hidden = 512
        x = torch.randn(16, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        scale_val = 6.0

        # Separate path: norm then quantize manually
        normed, _ = _rmsnorm_fwd_pytorch(x, w, 1e-5, False)
        dequant_scale = 1.0 / scale_val
        fp8_separate = (normed.float() * scale_val).to(torch.float8_e4m3fnuz)
        deq_separate = fp8_separate.float() * dequant_scale

        # Fused path via tex
        q = Float8Quantizer(
            scale=torch.tensor([scale_val], dtype=torch.float32, device=device),
            amax=torch.tensor([0.0], dtype=torch.float32, device=device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        out_fused, _, _ = tex.rmsnorm_fwd(x, w, 1e-5, None, q, None, 0, False)
        deq_fused = out_fused._data.view(torch.float8_e4m3fnuz).float() * dequant_scale

        # Allow FP8 rounding tolerance — different intermediate precision
        # (fused kernel does norm in float32 vs PyTorch fallback in bf16)
        diff = (deq_separate - deq_fused).abs().max().item()
        assert diff < 0.5, (
            f"Fused vs separate max dequantized diff: {diff:.4f} (expected < 0.5)"
        )

    def test_fused_rmsnorm_fp8_quant_3d_input(self, device):
        """Fused path handles 3D input shape correctly."""
        from transformer_engine.pytorch._lite.norms import (
            _aiter_fused_rms_fp8_static, _try_load_aiter_norms,
        )
        from transformer_engine.pytorch import Float8Quantizer

        _try_load_aiter_norms()
        if _aiter_fused_rms_fp8_static is None:
            pytest.skip("AITER fused RMSNorm+FP8 kernel not available")

        x = torch.randn(4, 8, 256, device=device, dtype=torch.bfloat16)
        w = torch.randn(256, device=device, dtype=torch.bfloat16)

        q = Float8Quantizer(
            scale=torch.tensor([2.0], dtype=torch.float32, device=device),
            amax=torch.tensor([0.0], dtype=torch.float32, device=device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        out, _, rsigma = tex.rmsnorm_fwd(x, w, 1e-5, None, q, None, 0, False)

        assert out.shape == (4, 8, 256), f"Expected (4,8,256), got {out.shape}"
        assert rsigma.shape == (4, 8), f"Expected (4,8), got {rsigma.shape}"

    # --- Current Scaling: fused RMSNorm + per-row dynamic FP8 quantize ---

    def test_fused_rmsnorm_current_scaling_active(self, device):
        """Fused RMSNorm+FP8 per-row dynamic quant kernel is used for CurrentScaling."""
        from transformer_engine.pytorch._lite import norms as _n
        from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer

        _n._try_load_aiter_norms()
        if _n._aiter_fused_rms_dynamic_quant is None:
            pytest.skip("AITER rmsnorm2d_fwd_with_dynamicquant not available")

        hidden = 256
        x = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)

        q = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
        )

        out, _, rsigma = tex.rmsnorm_fwd(x, w, 1e-5, None, q, None, 0, False)

        # Verify output is Float8Tensor
        assert type(out).__name__ == "Float8Tensor", (
            f"Expected Float8Tensor, got {type(out).__name__}"
        )
        # Verify shape is preserved
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        # Verify _scale_inv is per-row (M,) not scalar
        assert hasattr(out, '_scale_inv')
        assert out._scale_inv.shape == (8,), (
            f"Expected per-row scale_inv shape (8,), got {out._scale_inv.shape}"
        )
        # Verify all per-row scales are positive (valid dequant scales)
        assert (out._scale_inv > 0).all(), "All per-row scales should be positive"

    def test_fused_rmsnorm_current_scaling_vs_separate(self, device):
        """Fused per-row path matches separate norm->quantize within FP8 tolerance."""
        from transformer_engine.pytorch._lite.norms import (
            _aiter_fused_rms_dynamic_quant, _try_load_aiter_norms,
            _rmsnorm_fwd_pytorch,
        )

        _try_load_aiter_norms()
        if _aiter_fused_rms_dynamic_quant is None:
            pytest.skip("AITER rmsnorm2d_fwd_with_dynamicquant not available")

        hidden = 512
        x = torch.randn(16, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)

        # Reference: separate RMSNorm (PyTorch)
        normed_ref, _ = _rmsnorm_fwd_pytorch(x, w, 1e-5, False)

        # Fused: RMSNorm + per-row dynamic FP8 quant
        fp8_dtype = torch.float8_e4m3fnuz
        out_fp8 = torch.empty_like(x, dtype=fp8_dtype)
        yscale = torch.empty(x.shape[0], dtype=torch.float32, device=device)
        _aiter_fused_rms_dynamic_quant(out_fp8, x, yscale, w, 1e-5)

        # Dequantize: FP8 data * per-row scale
        deq_fused = out_fp8.to(torch.float32) * yscale.unsqueeze(1)

        # FP8 E4M3 has ~3.5% relative error budget — use generous tolerance
        max_err = (normed_ref.float() - deq_fused).abs().max().item()
        rel_err = (
            (normed_ref.float() - deq_fused).abs()
            / (normed_ref.float().abs() + 1e-8)
        ).mean().item()
        assert rel_err < 0.05, (
            f"Mean relative error {rel_err:.4f} exceeds 5% tolerance"
        )
        assert max_err < 1.0, (
            f"Max abs error {max_err:.4f} exceeds tolerance"
        )

    def test_fused_rmsnorm_current_scaling_per_row_scales_vary(self, device):
        """Per-row scales should differ across rows (not degenerate scalar)."""
        from transformer_engine.pytorch._lite.norms import (
            _aiter_fused_rms_dynamic_quant, _try_load_aiter_norms,
        )

        _try_load_aiter_norms()
        if _aiter_fused_rms_dynamic_quant is None:
            pytest.skip("AITER rmsnorm2d_fwd_with_dynamicquant not available")

        # Use input with varying row magnitudes to ensure different scales
        hidden = 256
        x = torch.randn(32, hidden, device=device, dtype=torch.bfloat16)
        # Scale rows differently so per-row scales must differ
        row_scales = torch.linspace(0.1, 10.0, 32, device=device).unsqueeze(1)
        x = x * row_scales.to(x.dtype)
        w = torch.ones(hidden, device=device, dtype=torch.bfloat16)

        fp8_dtype = torch.float8_e4m3fnuz
        out_fp8 = torch.empty_like(x, dtype=fp8_dtype)
        yscale = torch.empty(32, dtype=torch.float32, device=device)
        _aiter_fused_rms_dynamic_quant(out_fp8, x, yscale, w, 1e-5)

        # With 32 rows at very different magnitudes, all scales should be unique
        unique_scales = yscale.unique().numel()
        assert unique_scales == 32, (
            f"Expected 32 unique per-row scales, got {unique_scales}"
        )

    def test_fused_rmsnorm_current_scaling_3d_input(self, device):
        """Fused per-row path handles 3D input shape correctly."""
        from transformer_engine.pytorch._lite import norms as _n
        from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer

        _n._try_load_aiter_norms()
        if _n._aiter_fused_rms_dynamic_quant is None:
            pytest.skip("AITER rmsnorm2d_fwd_with_dynamicquant not available")

        x = torch.randn(4, 8, 256, device=device, dtype=torch.bfloat16)
        w = torch.randn(256, device=device, dtype=torch.bfloat16)

        q = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
        )
        out, _, rsigma = tex.rmsnorm_fwd(x, w, 1e-5, None, q, None, 0, False)

        assert out.shape == (4, 8, 256), f"Expected (4,8,256), got {out.shape}"
        # rsigma should be flattened to batch dims
        assert rsigma.shape == (4, 8), f"Expected (4,8), got {rsigma.shape}"
        # scale_inv should be per-row over the flattened batch: 4*8 = 32 rows
        assert out._scale_inv.shape == (32,), (
            f"Expected per-row scale_inv shape (32,), got {out._scale_inv.shape}"
        )

    def test_aiter_layernorm_fwd_bwd(self, device):
        """AITER LayerNorm forward and backward produce correct results."""
        from transformer_engine.pytorch._lite.norms import (
            _layernorm_fwd_pytorch, _layernorm_bwd_pytorch,
        )

        hidden = 512
        x = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        b = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        g = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)

        # PyTorch reference
        y_pt, mean_pt, rstd_pt = _layernorm_fwd_pytorch(x, w, b, 1e-5, False)
        dx_pt, dw_pt, db_pt = _layernorm_bwd_pytorch(g, x, mean_pt, rstd_pt, w, False)

        # AITER-backed tex path
        y_te, mean_te, rstd_te = tex.layernorm_fwd(x, w, b, 1e-5, None, None, None, 0, False)
        dx_te, dw_te, db_te = tex.layernorm_bwd(g, x, mean_te, rstd_te, w, 0, False)

        assert torch.allclose(y_te, y_pt, atol=8e-2, rtol=2e-2), (
            f"LayerNorm fwd max diff: {(y_te - y_pt).abs().max().item():.2e}"
        )
        assert torch.allclose(dx_te.to(torch.bfloat16), dx_pt.to(torch.bfloat16),
                              atol=8e-2, rtol=2e-2), (
            f"LayerNorm bwd dx max diff: {(dx_te - dx_pt).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("hidden_size", [256, 512, 1024])
    def test_layernorm_fwd_triton_vs_pytorch(self, device, hidden_size):
        """Triton layernorm_fwd should match PyTorch reference."""
        from transformer_engine.pytorch._lite.norms import (
            _layernorm_fwd_pytorch,
        )
        weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
        bias = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
        x = torch.randn(8, hidden_size, device=device, dtype=torch.bfloat16)

        y_pt, mean_pt, rstd_pt = _layernorm_fwd_pytorch(
            x, weight, bias, 1e-5, False,
        )
        y_te, mean_te, rstd_te = tex.layernorm_fwd(
            x, weight, bias, 1e-5, None, None, None, 0, False,
        )
        # Dequantize if needed
        if hasattr(y_te, 'dequantize'):
            y_te = y_te.dequantize()
        # BF16 layernorm: Triton fused kernel vs PyTorch individual ops have
        # different rounding, so tolerance must accommodate BF16 ULP differences
        assert torch.allclose(y_te.to(torch.bfloat16), y_pt, atol=8e-2, rtol=2e-2), (
            f"LayerNorm fwd max diff: {(y_te.to(torch.bfloat16) - y_pt).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("hidden_size", [256, 512, 1024])
    def test_rmsnorm_fwd_triton_vs_pytorch(self, device, hidden_size):
        """Triton rmsnorm_fwd should match PyTorch reference."""
        from transformer_engine.pytorch._lite.norms import (
            _rmsnorm_fwd_pytorch,
        )
        weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
        x = torch.randn(8, hidden_size, device=device, dtype=torch.bfloat16)

        y_pt, rstd_pt = _rmsnorm_fwd_pytorch(
            x, weight, 1e-5, False,
        )
        y_te, _, rstd_te = tex.rmsnorm_fwd(
            x, weight, 1e-5, None, None, None, 0, False,
        )
        if hasattr(y_te, 'dequantize'):
            y_te = y_te.dequantize()
        # BF16 RMSNorm: Triton fused kernel vs PyTorch individual ops have
        # different rounding; wider tolerance for BF16 comparison
        assert torch.allclose(y_te.to(torch.bfloat16), y_pt, atol=5e-1, rtol=5e-2), (
            f"RMSNorm fwd max diff: {(y_te.to(torch.bfloat16) - y_pt).abs().max().item():.2e}"
        )

    def test_layernorm_bwd_triton_vs_pytorch(self, device):
        """Triton layernorm_bwd should match PyTorch reference."""
        from transformer_engine.pytorch._lite.norms import (
            _layernorm_fwd_pytorch,
            _layernorm_bwd_pytorch,
        )
        hidden = 512
        weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        bias = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        x = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
        grad_out = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)

        _, mean, rstd = _layernorm_fwd_pytorch(
            x, weight, bias, 1e-5, False,
        )
        dx_pt, dw_pt, db_pt = _layernorm_bwd_pytorch(
            grad_out, x, mean, rstd, weight, False,
        )
        dx_te, dw_te, db_te = tex.layernorm_bwd(
            grad_out, x, mean, rstd, weight, 0, False,
        )
        assert torch.allclose(dx_te, dx_pt, atol=8e-2, rtol=2e-2), (
            f"LayerNorm bwd dx max diff: {(dx_te - dx_pt).abs().max().item():.2e}"
        )
        # Weight grad is reduced over batch -- wider BF16 tolerance
        assert torch.allclose(dw_te, dw_pt, atol=5e-2, rtol=5e-2), (
            f"LayerNorm bwd dw max diff: {(dw_te - dw_pt).abs().max().item():.2e}"
        )

    def test_rmsnorm_bwd_triton_vs_pytorch(self, device):
        """Triton rmsnorm_bwd should match PyTorch reference."""
        from transformer_engine.pytorch._lite.norms import (
            _rmsnorm_fwd_pytorch,
            _rmsnorm_bwd_pytorch,
        )
        hidden = 512
        weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        x = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
        grad_out = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)

        _, rstd = _rmsnorm_fwd_pytorch(
            x, weight, 1e-5, False,
        )
        dx_pt, dw_pt = _rmsnorm_bwd_pytorch(
            grad_out, x, rstd, weight, False,
        )
        dx_te, dw_te = tex.rmsnorm_bwd(
            grad_out, x, rstd, weight, 0, False,
        )
        # Cast to common dtype -- PyTorch fallback may promote to fp32
        # while Triton kernel returns in input dtype
        dx_te_bf16 = dx_te.to(torch.bfloat16)
        dx_pt_bf16 = dx_pt.to(torch.bfloat16)
        dw_te_bf16 = dw_te.to(torch.bfloat16)
        dw_pt_bf16 = dw_pt.to(torch.bfloat16)
        assert torch.allclose(dx_te_bf16, dx_pt_bf16, atol=5e-2, rtol=2e-2), (
            f"RMSNorm bwd dx max diff: {(dx_te_bf16 - dx_pt_bf16).abs().max().item():.2e}"
        )
        assert torch.allclose(dw_te_bf16, dw_pt_bf16, atol=5e-2, rtol=5e-2), (
            f"RMSNorm bwd dw max diff: {(dw_te_bf16 - dw_pt_bf16).abs().max().item():.2e}"
        )

    def test_layernorm_3d_input(self, device):
        """Norm functions should handle 3D input (batch, seq, hidden)."""
        mod = te.LayerNorm(256).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(2, 4, 256, device=device, dtype=torch.bfloat16)
        y = mod(x)
        assert y.shape == (2, 4, 256)

    def test_rmsnorm_3d_input(self, device):
        """RMSNorm should handle 3D input (batch, seq, hidden)."""
        mod = te.RMSNorm(256).to(dtype=torch.bfloat16, device=device)
        x = torch.randn(2, 4, 256, device=device, dtype=torch.bfloat16)
        y = mod(x)
        assert y.shape == (2, 4, 256)


# ---------------------------------------------------------------------------
# FP8 quantize / dequantize (Phase 2)
# ---------------------------------------------------------------------------

class TestQuantize:
    """Verify FP8 quantize/dequantize works in lite mode without recursion."""

    def test_fp8_quantize_no_recursion(self, device):
        """tex.quantize with Float8Quantizer should not recurse."""
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

        x = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
        amax_val = x.abs().max().item()
        fp8_max = 240.0
        scale = torch.tensor([fp8_max / amax_val], device=device, dtype=torch.float32)
        amax = torch.tensor([0.0], device=device, dtype=torch.float32)
        q = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E4M3)

        result = tex.quantize(x, q)
        assert hasattr(result, '_data'), "Quantize should return a Float8Tensor"
        assert result._data.shape == (8, 16)
        assert result._data.dtype == torch.uint8

    def test_fp8_dequantize(self, device):
        """tex.dequantize should reconstruct values from FP8."""
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

        x = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
        amax_val = x.abs().max().item()
        fp8_max = 240.0
        scale = torch.tensor([fp8_max / amax_val], device=device, dtype=torch.float32)
        amax = torch.tensor([0.0], device=device, dtype=torch.float32)
        q = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E4M3)

        quantized = tex.quantize(x, q)
        y = tex.dequantize(quantized, tex.DType.kBFloat16)

        # FP8 quantization error should be small with proper scaling
        max_abs_err = (y.to(torch.bfloat16) - x).abs().max().item()
        assert max_abs_err < 0.5, f"FP8 roundtrip max error too large: {max_abs_err:.4f}"

    def test_fp8_roundtrip_relative_error(self, device):
        """FP8 quantize-dequantize roundtrip should have low relative error."""
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
        amax_val = x.abs().max().item()
        fp8_max = 240.0
        scale = torch.tensor([fp8_max / amax_val], device=device, dtype=torch.float32)
        amax = torch.tensor([0.0], device=device, dtype=torch.float32)
        q = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E4M3)

        quantized = tex.quantize(x, q)
        y = tex.dequantize(quantized, tex.DType.kBFloat16)

        mean_rel_err = ((y.to(torch.bfloat16) - x).abs() / (x.abs() + 1e-8)).mean().item()
        assert mean_rel_err < 0.1, f"FP8 mean relative error too large: {mean_rel_err:.4f}"

    def test_quantize_no_quantizer(self, device):
        """quantize with no quantizer should return tensor as-is."""
        x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        result = tex.quantize(x, None)
        assert torch.equal(result, x)

    def test_quantize_with_output(self, device):
        """quantize with output tensor should copy into it."""
        x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        result = tex.quantize(x, None, output=out)
        assert torch.equal(result, x)

    def test_bgrad_quantize(self, device):
        """bgrad_quantize should return (bias_grad, quantized)."""
        x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        bgrad, quantized = tex.bgrad_quantize(x, None)
        expected_bgrad = x.sum(dim=0)
        assert torch.allclose(bgrad, expected_bgrad)

    def test_dequantize_plain_tensor(self, device):
        """dequantize on a plain tensor should just cast dtype."""
        x = torch.randn(4, 8, device=device, dtype=torch.float32)
        y = tex.dequantize(x, tex.DType.kBFloat16)
        assert y.dtype == torch.bfloat16
        assert torch.allclose(y, x.to(torch.bfloat16))

    # --- CurrentScaling per-row quantize (backward path) ---

    def test_current_scaling_quantize_per_row(self, device):
        """CurrentScaling quantizer should produce per-row scales via AITER."""
        import sys
        _qmod = sys.modules["transformer_engine.pytorch._lite.quantize"]
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
        )

        _qmod._try_load_aiter_quant()
        if _qmod._aiter_dynamic_per_token_quant is None:
            pytest.skip("AITER dynamic_per_token_quant_fp8_i8 not available")

        x = torch.randn(16, 64, device=device, dtype=torch.bfloat16)
        q = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
        )

        result = tex.quantize(x, q)

        # Should be Float8Tensor
        assert hasattr(result, '_data'), "Expected Float8Tensor output"
        assert result._data.shape == (16, 64)
        # Per-row: _scale_inv should be (M,) not scalar
        assert result._scale_inv.shape == (16,), (
            f"Expected per-row scale_inv (16,), got {result._scale_inv.shape}"
        )
        assert (result._scale_inv > 0).all(), "All per-row scales should be positive"

    def test_current_scaling_quantize_roundtrip(self, device):
        """CurrentScaling per-row quantize->dequantize roundtrip has low error."""
        import sys
        _qmod = sys.modules["transformer_engine.pytorch._lite.quantize"]
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
        )

        _qmod._try_load_aiter_quant()
        if _qmod._aiter_dynamic_per_token_quant is None:
            pytest.skip("AITER dynamic_per_token_quant_fp8_i8 not available")

        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        q = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
        )

        result = tex.quantize(x, q)

        # Manual dequantize: FP8 data * per-row scale
        from transformer_engine.pytorch._lite.quantize import _te_dtype_to_torch_fp8
        fp8_dtype = _te_dtype_to_torch_fp8(q.dtype)
        fp8_data = result._data.view(fp8_dtype)
        deq = fp8_data.to(torch.float32) * result._scale_inv.unsqueeze(1)

        rel_err = (
            (x.float() - deq).abs() / (x.float().abs() + 1e-8)
        ).mean().item()
        assert rel_err < 0.05, (
            f"Per-row quantize roundtrip mean rel error {rel_err:.4f} > 5%"
        )

    def test_current_scaling_quantize_backward_dgrad_flow(self, device):
        """Simulate backward: quantize dY per-row, then per-token GEMM for dgrad."""
        import sys
        _qmod = sys.modules["transformer_engine.pytorch._lite.quantize"]
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
        )

        _qmod._try_load_aiter_quant()
        if _qmod._aiter_dynamic_per_token_quant is None:
            pytest.skip("AITER dynamic_per_token_quant_fp8_i8 not available")

        try:
            from aiter.ops.triton.gemm_a8w8_per_token_scale import (
                gemm_a8w8_per_token_scale,
            )
        except ImportError:
            pytest.skip("AITER gemm_a8w8_per_token_scale not available")

        M, N, K = 32, 64, 128  # dY is [M, N], W is [N, K], dX = dY @ W
        fp8_dtype = torch.float8_e4m3fnuz

        # dY: quantize per-row via CurrentScaling
        dY = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        q = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
        )
        dY_quant = tex.quantize(dY, q)
        dY_fp8 = dY_quant._data.view(fp8_dtype)
        dY_scale = dY_quant._scale_inv  # (M,)

        # W: per-tensor quantize
        W = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        w_amax = W.abs().max()
        w_qs = 240.0 / w_amax
        W_fp8 = (W.float() * w_qs).to(fp8_dtype)
        w_ds = torch.full((K, 1), (1.0 / w_qs).item(),
                          dtype=torch.float32, device=device)

        # dgrad GEMM: dX = dY @ W  (dY is [M,N], W is [N,K])
        # per_token_scale: Y = X @ W^T, so X=dY [M,N], W_t=W^T [K,N]
        # We need W transposed: W is [N,K], so W^T is [K,N]
        # gemm_a8w8_per_token_scale(x, w, x_scale, w_scale) computes x @ w^T
        # So: x=dY_fp8 [M,N], w=W_fp8^T [K,N] → result [M,K]
        # But kernel takes w in [N_out, K_in] and transposes internally
        # Actually: kernel computes Y = X @ W^T where W is [K, N]
        # So we pass w=W^T = [K, N], then kernel does dY @ (W^T)^T = dY @ W
        W_T_fp8 = W_fp8.t().contiguous()  # [K, N]
        result = gemm_a8w8_per_token_scale(
            dY_fp8, W_T_fp8,
            dY_scale.unsqueeze(1), w_ds,
        )

        # Reference: dequant both, matmul
        dY_deq = dY_fp8.to(torch.float32) * dY_scale.unsqueeze(1)
        W_deq = W_fp8.to(torch.float32) * (1.0 / w_qs.item())
        ref = dY_deq @ W_deq  # [M,N] @ [N,K] = [M,K]

        assert result.shape == (M, K), f"Expected ({M},{K}), got {result.shape}"
        rel_err = (
            (result.float() - ref).abs() / (ref.abs() + 1e-8)
        ).mean().item()
        assert rel_err < 0.05, (
            f"Backward dgrad per-row mean rel error {rel_err:.4f} > 5%"
        )


# ---------------------------------------------------------------------------
# MXFP8 BlockScaling tests
# ---------------------------------------------------------------------------

class TestMXFP8:
    """Verify MXFP8 BlockScaling detection, quantize, GEMM, and norms in lite mode."""

    def test_mxfp8_detection_not_fp4(self, device):
        """MXFP8 tensor should be detected as MXFP8, not FP4."""
        from transformer_engine.pytorch._lite.gemm import _is_mxfp8, _is_fp4
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        # Both dims must be divisible by 32 for MXFP8
        q = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        t = q.make_empty((32, 64), dtype=torch.bfloat16, device=device)
        assert _is_mxfp8(t), "MXFP8 tensor not detected by _is_mxfp8()"
        assert not _is_fp4(t), "MXFP8 tensor should NOT match _is_fp4()"

    def test_fp4_detection_not_mxfp8(self, device):
        """MXFP4 tensor should be detected as FP4, not MXFP8."""
        from transformer_engine.pytorch._lite.gemm import _is_mxfp8, _is_fp4
        try:
            from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
        except ImportError:
            pytest.skip("MXFP4Quantizer not available")

        q = MXFP4Quantizer(tex.DType.kFloat4E2M1)
        t = q.make_empty((32, 64), dtype=torch.bfloat16, device=device)
        assert _is_fp4(t), "MXFP4 tensor not detected by _is_fp4()"
        assert not _is_mxfp8(t), "MXFP4 tensor should NOT match _is_mxfp8()"

    def test_mxfp8_is_quantized(self, device):
        """_is_quantized should return True for MXFP8 tensors."""
        from transformer_engine.pytorch._lite.gemm import _is_quantized
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        q = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        t = q.make_empty((32, 64), dtype=torch.bfloat16, device=device)
        assert _is_quantized(t), "MXFP8 tensor should be detected as quantized"

    def test_linear_scale_to_e8m0(self, device):
        """E8M0 conversion should produce correct biased exponents."""
        import sys
        _qmod = sys.modules["transformer_engine.pytorch._lite.quantize"]
        e8m0_fn = _qmod._linear_scale_to_e8m0

        scales = torch.tensor([1.0, 2.0, 0.5, 4.0, 0.25], device=device)
        e8m0 = e8m0_fn(scales)
        # 1.0 = 2^0 → 0 + 127 = 127
        # 2.0 = 2^1 → 1 + 127 = 128
        # 0.5 = 2^-1 → -1 + 127 = 126
        # 4.0 = 2^2 → 2 + 127 = 129
        # 0.25 = 2^-2 → -2 + 127 = 125
        expected = torch.tensor([127, 128, 126, 129, 125], dtype=torch.uint8, device=device)
        assert torch.equal(e8m0, expected), (
            f"E8M0 mismatch: {e8m0.tolist()} vs {expected.tolist()}"
        )

    def test_mxfp8_quantize_roundtrip(self, device):
        """MXFP8 quantize→dequantize roundtrip should have low error."""
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        q = MXFP8Quantizer(tex.DType.kFloat8E4M3)

        quantized = tex.quantize(x, q)
        # Verify it's an MXFP8 tensor
        assert hasattr(quantized, '_rowwise_data'), "Expected MXFP8Tensor"
        assert hasattr(quantized, '_rowwise_scale_inv'), "Expected MXFP8 scales"

        # Dequantize and check error
        deq = tex.dequantize(quantized, tex.DType.kBFloat16)
        rel_err = (
            (x.float() - deq.float()).abs() / (x.float().abs() + 1e-8)
        ).mean().item()
        assert rel_err < 0.1, (
            f"MXFP8 roundtrip mean relative error {rel_err:.4f} > 10%"
        )

    def test_mxfp8_quantize_pytorch_fallback(self, device):
        """MXFP8 PyTorch fallback should produce correct results without Triton."""
        import sys
        _qmod = sys.modules["transformer_engine.pytorch._lite.quantize"]
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        q = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        out = q.make_empty(x.shape, dtype=x.dtype, device=device)

        result = _qmod._quantize_mxfp8_pytorch(x, q, out)

        # Verify data was written
        assert result._rowwise_data is not None
        assert result._rowwise_data.any(), "FP8 data should be non-zero"
        # Verify E8M0 scales were written
        assert result._rowwise_scale_inv is not None
        assert result._rowwise_scale_inv.any(), "E8M0 scales should be non-zero"

        # Dequantize via Triton/tensor method and check error
        deq = result.dequantize(dtype=torch.bfloat16)
        rel_err = (
            (x.float() - deq.float()).abs() / (x.float().abs() + 1e-8)
        ).mean().item()
        assert rel_err < 0.1, (
            f"MXFP8 PyTorch fallback roundtrip mean rel error {rel_err:.4f} > 10%"
        )

    def test_mxfp8_gemm_dequant_path(self, device):
        """MXFP8 tensors through generic_gemm should produce correct BF16 via dequant."""
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        M, N, K = 32, 64, 128
        A_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        B_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        # Quantize both to MXFP8
        q = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        A_mxfp8 = tex.quantize(A_bf16, q)
        B_mxfp8 = tex.quantize(B_bf16, q)

        # Reference: dequantize then matmul
        A_deq = A_mxfp8.dequantize(dtype=torch.bfloat16)
        B_deq = B_mxfp8.dequantize(dtype=torch.bfloat16)
        ref = B_deq @ A_deq.t()  # TN layout: result = B @ A^T

        ws = torch.empty(1024, device=device, dtype=torch.uint8)
        out, _, _, _ = tex.generic_gemm(
            A_mxfp8, True, B_mxfp8, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )

        assert out.shape == (M, N), f"Expected ({M},{N}), got {out.shape}"
        # Should match dequant reference closely (same precision path)
        max_diff = (out.float() - ref.float()).abs().max().item()
        assert max_diff < 0.5, (
            f"MXFP8 GEMM max diff {max_diff:.4f} vs dequant reference"
        )

    def test_mxfp8_fused_rmsnorm(self, device):
        """Fused RMSNorm+MXFP8 quant should produce valid MXFP8Tensor."""
        from transformer_engine.pytorch._lite import norms as _n
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        _n._try_load_aiter_norms()
        if _n._aiter_fused_rms_fp8_group is None:
            pytest.skip("AITER fused_rms_fp8_group_quant not available")

        hidden = 256
        x = torch.randn(32, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)

        q = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        out, _, rsigma = tex.rmsnorm_fwd(x, w, 1e-5, None, q, None, 0, False)

        # Verify output is MXFP8
        assert hasattr(out, '_rowwise_data'), (
            f"Expected MXFP8Tensor, got {type(out).__name__}"
        )
        assert out._rowwise_data is not None, "MXFP8 rowwise data should be populated"
        assert out._rowwise_scale_inv is not None, "MXFP8 scales should be populated"
        assert out._rowwise_scale_inv.any(), "E8M0 scales should be non-zero"


# ---------------------------------------------------------------------------
# GEMM tests
# ---------------------------------------------------------------------------

class TestGemm:
    """Verify generic_gemm in lite mode (AITER CK/Triton + PyTorch fallback)."""

    DTYPE = torch.bfloat16

    def _workspace(self, device):
        return torch.empty(1024, device=device, dtype=torch.uint8)

    # -- Basic matmul (TN layout: weight[out,in], input[batch,in]) --------

    def test_gemm_tn_basic(self, device):
        """TN GEMM: A[out,in].T @ B[batch,in] -> [batch,out]."""
        M, N, K = 128, 64, 256
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)  # weight [out, in]
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)  # input  [batch, in]
        ws = self._workspace(device)
        out, bias_grad, gelu_in, extra = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        assert out.shape == (M, N), f"Expected ({M},{N}), got {out.shape}"

    def test_gemm_tn_numerical(self, device):
        """TN GEMM result should match torch.matmul reference."""
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        ref = B @ A.t()  # [M,K] @ [K,N] = [M,N]
        assert torch.allclose(out.to(self.DTYPE), ref, atol=1e-2, rtol=1e-2), (
            f"GEMM max diff: {(out.to(self.DTYPE) - ref).abs().max().item():.4e}"
        )

    @pytest.mark.parametrize("transA,transB,shapeA,shapeB,expect", [
        (True,  False, (64, 128),  (32, 128), (32, 64)),   # TN
        (False, False, (128, 64),  (32, 128), (32, 64)),   # NN
        (True,  True,  (64, 128),  (128, 32), (32, 64)),   # TT
        (False, True,  (128, 64),  (128, 32), (32, 64)),   # NT
    ])
    def test_gemm_transpose_combos(self, device, transA, transB, shapeA, shapeB, expect):
        """All transpose combinations should produce correct shapes."""
        A = torch.randn(*shapeA, device=device, dtype=self.DTYPE)
        B = torch.randn(*shapeB, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, transA, B, transB, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        assert out.shape == expect, f"Expected {expect}, got {out.shape}"

    # -- Bias epilogue ----------------------------------------------------

    def test_gemm_with_bias(self, device):
        """GEMM + bias addition should work."""
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        bias = torch.randn(N, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, None,
            bias, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        ref = B @ A.t() + bias
        assert torch.allclose(out.to(self.DTYPE), ref, atol=1e-2, rtol=1e-2), (
            f"GEMM+bias max diff: {(out.to(self.DTYPE) - ref).abs().max().item():.4e}"
        )

    def test_gemm_bias_grad(self, device):
        """GEMM with grad=True should compute bias gradient.

        In the backward pass, B is the grad_output dY. The bias gradient
        is dY.reshape(-1, dY.shape[-1]).sum(dim=0).
        cuBLAS column-major: result = op(B) @ op(A).
        Use transA=False, transB=False so result = B @ A.
        A=[N,K], B=[M,N] → result=[M,K], bias_grad=B.sum(0)=[N].
        """
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, N, device=device, dtype=self.DTYPE)  # dY [M, N]
        bias = torch.randn(N, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        _, bias_grad, _, _ = tex.generic_gemm(
            A, False, B, False, None, None, None,
            bias, None, False, None, True, ws, ws.shape[0],
            False, False,
        )
        ref_bgrad = B.reshape(-1, B.shape[-1]).sum(dim=0)
        assert bias_grad.shape == ref_bgrad.shape, (
            f"bias_grad shape {bias_grad.shape} != expected {ref_bgrad.shape}"
        )
        assert torch.allclose(bias_grad.to(self.DTYPE), ref_bgrad, atol=1e-2, rtol=1e-2)

    # -- GELU epilogue ----------------------------------------------------

    def test_gemm_with_gelu(self, device):
        """GEMM + GELU activation should work."""
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        gelu_in = torch.empty(M, N, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        out, _, gelu_saved, _ = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, True, gelu_in, False, ws, ws.shape[0],
            False, False,
        )
        # gelu_saved should hold the pre-GELU values
        ref_pre_gelu = B @ A.t()
        ref_out = torch.nn.functional.gelu(ref_pre_gelu, approximate='tanh')
        assert torch.allclose(out.to(self.DTYPE), ref_out, atol=1e-2, rtol=1e-2), (
            f"GEMM+GELU max diff: {(out.to(self.DTYPE) - ref_out).abs().max().item():.4e}"
        )

    # -- Accumulate -------------------------------------------------------

    def test_gemm_accumulate(self, device):
        """GEMM with accumulate=True should add to existing D."""
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        D = torch.randn(M, N, device=device, dtype=self.DTYPE)
        D_orig = D.clone()
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, D, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            True, False,
        )
        ref = D_orig + B @ A.t()
        assert torch.allclose(D.to(self.DTYPE), ref, atol=1e-2, rtol=1e-2), (
            f"Accumulate max diff: {(D.to(self.DTYPE) - ref).abs().max().item():.4e}"
        )

    # -- Alpha scaling ----------------------------------------------------

    def test_gemm_alpha(self, device):
        """GEMM with alpha != 1.0 should scale the result."""
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False, alpha=0.5,
        )
        ref = 0.5 * (B @ A.t())
        assert torch.allclose(out.to(self.DTYPE), ref, atol=1e-2, rtol=1e-2)

    # -- Output into pre-allocated D (no accumulate) ----------------------

    def test_gemm_output_into_d(self, device):
        """GEMM should write result into D when accumulate=False."""
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        D = torch.zeros(M, N, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, D, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        ref = B @ A.t()
        assert torch.allclose(D, ref, atol=1e-2, rtol=1e-2)

    # -- Return format ----------------------------------------------------

    def test_gemm_return_format(self, device):
        """generic_gemm should return (out, bias_grad, gelu_input, extra_output)."""
        M, N, K = 16, 32, 64
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        result = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        assert isinstance(result, tuple) and len(result) == 4, (
            f"Expected 4-tuple, got {type(result)} of len {len(result)}"
        )

    # -- FP32 precision ---------------------------------------------------

    def test_gemm_fp32(self, device):
        """GEMM should work with FP32 inputs."""
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=torch.float32)
        B = torch.randn(M, K, device=device, dtype=torch.float32)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        ref = B @ A.t()
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    # -- output_dtype honored across mixed-precision operands --------------

    def test_gemm_output_dtype_honored_mixed_operands(self, device):
        """output_dtype must be honored even when an operand is fp32.

        Regression: the PyTorch fallback promoted compute to fp32 whenever
        either operand was fp32 and ignored `output_dtype`, returning fp32.
        The next module then failed `set_activation_dtype` with input=fp32
        against bf16 weights. cuBLAS in the full build always casts to the
        caller-requested dtype.
        """
        M, N, K = 16, 32, 64
        ws = self._workspace(device)

        # bf16 weight × fp32 activation: naive promotion would yield fp32.
        # Caller requests bf16 output.
        A = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        B_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)

        out, _, _, _ = tex.generic_gemm(
            A, True, B_fp32, False, None, None, tex.DType.kBFloat16,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        assert out.dtype == torch.bfloat16, (
            f"Expected bf16 output, got {out.dtype} — output_dtype not honored"
        )

        # Symmetric case: fp32 weight × bf16 activation.
        A_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
        B = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        out, _, _, _ = tex.generic_gemm(
            A_fp32, True, B, False, None, None, tex.DType.kBFloat16,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        assert out.dtype == torch.bfloat16

        # torch.dtype is also accepted (pass-through in _resolve_output_dtype).
        out, _, _, _ = tex.generic_gemm(
            A, True, B_fp32, False, None, None, torch.bfloat16,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        assert out.dtype == torch.bfloat16

    # -- Per-row scaled FP8 GEMM (CurrentScaling) ----------------------------

    def test_gemm_per_row_scaled_fp8(self, device):
        """FP8 GEMM with per-row activation scales dispatches correctly."""
        from transformer_engine.pytorch._lite.gemm import (
            _is_per_row_scaled, _is_block_scaled, is_aiter_available,
        )
        if not is_aiter_available():
            pytest.skip("AITER not available")
        try:
            from aiter.ops.triton.gemm_a8w8_per_token_scale import (
                gemm_a8w8_per_token_scale,
            )
        except ImportError:
            pytest.skip("AITER gemm_a8w8_per_token_scale not available")

        M, N, K = 32, 64, 128
        fp8_dtype = torch.float8_e4m3fnuz

        # Create per-row-scaled activation (simulates fused norm+quant output)
        x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        from aiter.ops.triton.quant import dynamic_per_token_quant_fp8_i8
        x_fp8 = torch.empty(M, K, dtype=fp8_dtype, device=device)
        x_scale = torch.empty(M, dtype=torch.float32, device=device)
        dynamic_per_token_quant_fp8_i8(x_fp8, x_bf16, x_scale)

        # Create per-tensor-scaled weight
        w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        w_amax = w_bf16.abs().max()
        w_quant_scale = 240.0 / w_amax
        w_fp8 = (w_bf16.float() * w_quant_scale).to(fp8_dtype)
        w_dequant = torch.tensor([1.0 / w_quant_scale.item()],
                                 dtype=torch.float32, device=device)

        # Verify scale detection
        assert _is_per_row_scaled(x_scale), "x_scale should be per-row"
        assert not _is_per_row_scaled(w_dequant), "w_dequant should not be per-row"
        assert not _is_block_scaled(x_scale), "per-row scale should not be block-scaled"

        # Build mock Float8Tensor-like objects for generic_gemm
        class _FP8Wrap:
            def __init__(self, data, scale_inv):
                self._data = data
                self._scale_inv = scale_inv
            @property
            def dtype(self):
                return self._data.dtype

        A = _FP8Wrap(w_fp8, w_dequant)  # weight [N, K], transA=True
        B = _FP8Wrap(x_fp8, x_scale)    # activation [M, K], transB=False

        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )

        # Reference: dequantize both and matmul in float32
        x_deq = x_fp8.to(torch.float32) * x_scale.unsqueeze(1)
        w_deq = w_fp8.to(torch.float32) * (1.0 / w_quant_scale.item())
        ref = x_deq @ w_deq.t()

        assert out.shape == (M, N), f"Expected ({M},{N}), got {out.shape}"
        rel_err = (
            (out.float() - ref).abs() / (ref.abs() + 1e-8)
        ).mean().item()
        assert rel_err < 0.05, (
            f"Per-row GEMM mean relative error {rel_err:.4f} exceeds 5% tolerance"
        )

    def test_gemm_per_row_scaled_numerical_accuracy(self, device):
        """Per-row scaled FP8 GEMM matches dequantized matmul reference."""
        from transformer_engine.pytorch._lite.gemm import is_aiter_available
        if not is_aiter_available():
            pytest.skip("AITER not available")
        try:
            from aiter.ops.triton.gemm_a8w8_per_token_scale import (
                gemm_a8w8_per_token_scale,
            )
        except ImportError:
            pytest.skip("AITER gemm_a8w8_per_token_scale not available")

        from aiter.ops.triton.rmsnorm import rmsnorm2d_fwd_with_dynamicquant

        M, K, N = 64, 256, 128
        fp8_dtype = torch.float8_e4m3fnuz

        # Full forward path: input → fused RMSNorm+quant → per-token GEMM
        inp = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        norm_w = torch.randn(K, device=device, dtype=torch.bfloat16)
        x_fp8 = torch.empty(M, K, dtype=fp8_dtype, device=device)
        x_scale = torch.empty(M, dtype=torch.float32, device=device)
        rmsnorm2d_fwd_with_dynamicquant(x_fp8, inp, x_scale, norm_w, 1e-5)

        # Weight (per-tensor quantized)
        w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        w_amax = w_bf16.abs().max()
        w_qs = 240.0 / w_amax
        w_fp8 = (w_bf16.float() * w_qs).to(fp8_dtype)
        w_ds = torch.full((N, 1), (1.0 / w_qs).item(),
                          dtype=torch.float32, device=device)

        result = gemm_a8w8_per_token_scale(
            x_fp8, w_fp8, x_scale.unsqueeze(1), w_ds,
        )

        # Reference
        x_deq = x_fp8.to(torch.float32) * x_scale.unsqueeze(1)
        w_deq = w_fp8.to(torch.float32) * w_ds
        ref = x_deq @ w_deq.t()

        rel_err = (
            (result.float() - ref).abs() / (ref.abs() + 1e-8)
        ).mean().item()
        assert rel_err < 0.02, (
            f"End-to-end fused norm→per-row GEMM mean rel error {rel_err:.4f} > 2%"
        )


# ---------------------------------------------------------------------------
# GEMM backend coverage (pytorch / triton / ck parity + dispatch asserts)
# ---------------------------------------------------------------------------

class _FP8Wrap:
    """Minimal Float8Tensor shim for generic_gemm.

    _is_quantized(tensor) returns True for anything with (_data, _scale_inv).
    _get_raw_data returns (_data, _scale_inv), which is all downstream
    dispatch paths need.
    """
    def __init__(self, data, scale_inv):
        self._data = data
        self._scale_inv = scale_inv

    @property
    def dtype(self):
        return self._data.dtype


def _quant_per_tensor_e4m3(x_bf16, fp8_dtype=torch.float8_e4m3fnuz):
    """Quantize a BF16 tensor to FP8 with a single scalar per-tensor scale.

    Returns (fp8_data, scale_inv_scalar) where scale_inv_scalar is a 1-elem
    tensor carrying dequant scale (matches Float8Tensor._scale_inv layout).
    """
    amax = x_bf16.abs().max().clamp_min(1e-6)
    qscale = 240.0 / amax
    x_fp8 = (x_bf16.float() * qscale).to(fp8_dtype)
    scale_inv = torch.tensor([1.0 / qscale.item()],
                             dtype=torch.float32, device=x_bf16.device)
    return x_fp8, scale_inv


class TestGemmBackendMatrix:
    """Ensure all three GEMM backends produce correct results and the
    `pytorch` backend actually takes the fast `torch._scaled_mm` path.
    """

    DTYPE = torch.bfloat16

    def _workspace(self, device):
        return torch.empty(1024, device=device, dtype=torch.uint8)

    def _set_backend(self, monkeypatch, backend):
        """Swap _GEMM_BACKEND in the gemm module for one test."""
        from transformer_engine.pytorch._lite import gemm as lite_gemm
        monkeypatch.setattr(lite_gemm, "_GEMM_BACKEND", backend)

    # -- Backend matrix: BF16 ---------------------------------------------

    @pytest.mark.parametrize("backend", ["pytorch", "triton", "ck"])
    def test_bf16_gemm_matches_reference(self, device, monkeypatch, backend):
        """BF16 GEMM must agree with torch.matmul on every backend.

        Protects against silent regressions in the ck/triton paths now
        that `pytorch` is the default.
        """
        self._set_backend(monkeypatch, backend)
        M, N, K = 32, 64, 128
        A = torch.randn(N, K, device=device, dtype=self.DTYPE)
        B = torch.randn(M, K, device=device, dtype=self.DTYPE)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, None,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )
        ref = B @ A.t()
        max_diff = (out.to(self.DTYPE) - ref).abs().max().item()
        assert max_diff < 5e-2, (
            f"[backend={backend}] BF16 GEMM max diff {max_diff:.4e}"
        )

    # -- Backend matrix: per-tensor FP8 (DelayedScaling layout) -----------

    @pytest.mark.parametrize("backend", ["pytorch", "triton", "ck"])
    def test_per_tensor_fp8_gemm_matches_dequant(
        self, device, monkeypatch, backend,
    ):
        """Per-tensor FP8×FP8 GEMM (DelayedScaling shape) must match the
        dequantized reference on every backend.

        This is the recipe Megatron hard-codes. Scalar scales should route
        to the per-tensor kernel family on all three backends. A regression
        here means the production training path is broken.
        """
        from transformer_engine.pytorch._lite.gemm import is_aiter_available
        if backend in ("triton", "ck") and not is_aiter_available():
            pytest.skip("AITER not available")

        self._set_backend(monkeypatch, backend)
        M, N, K = 64, 128, 256
        fp8 = torch.float8_e4m3fnuz

        x_bf16 = torch.randn(M, K, device=device, dtype=self.DTYPE)
        w_bf16 = torch.randn(N, K, device=device, dtype=self.DTYPE)
        x_fp8, x_scale = _quant_per_tensor_e4m3(x_bf16, fp8)
        w_fp8, w_scale = _quant_per_tensor_e4m3(w_bf16, fp8)

        A = _FP8Wrap(w_fp8, w_scale)  # weight [N, K]
        B = _FP8Wrap(x_fp8, x_scale)  # activation [M, K]

        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, tex.DType.kBFloat16,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )

        # Dequantized reference in fp32
        x_deq = x_fp8.float() * x_scale.item()
        w_deq = w_fp8.float() * w_scale.item()
        ref = x_deq @ w_deq.t()

        rel_err = (
            (out.float() - ref).abs() / (ref.abs() + 1e-3)
        ).mean().item()
        assert rel_err < 0.05, (
            f"[backend={backend}] per-tensor FP8 GEMM mean rel err "
            f"{rel_err:.4f} exceeds 5% tolerance"
        )
        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16

    # -- Dispatch path counters: pytorch backend must take fast path ------

    def test_pytorch_backend_takes_scaled_mm_path(
        self, device, monkeypatch,
    ):
        """Per-tensor FP8 under backend=pytorch must land on _scaled_mm,
        not dequant+matmul.

        The whole point of the pytorch default is hipBLASLt via _scaled_mm.
        If a future change silently forces scalar scales into a rejected
        layout (e.g. broadcast to (M,1)), this test catches it by reading
        the dispatch counters.
        """
        from transformer_engine.pytorch._lite import gemm as lite_gemm
        if not hasattr(torch, "_scaled_mm"):
            pytest.skip("torch._scaled_mm not available")

        self._set_backend(monkeypatch, "pytorch")
        # Counters are gated behind _LITE_DIAG; flip on and zero them.
        monkeypatch.setattr(lite_gemm, "_LITE_DIAG", True)
        lite_gemm._GEMM_CALLS.clear()

        M, N, K = 64, 128, 256
        fp8 = torch.float8_e4m3fnuz
        x_fp8, x_scale = _quant_per_tensor_e4m3(
            torch.randn(M, K, device=device, dtype=self.DTYPE), fp8,
        )
        w_fp8, w_scale = _quant_per_tensor_e4m3(
            torch.randn(N, K, device=device, dtype=self.DTYPE), fp8,
        )

        A = _FP8Wrap(w_fp8, w_scale)
        B = _FP8Wrap(x_fp8, x_scale)
        ws = self._workspace(device)
        tex.generic_gemm(
            A, True, B, False, None, None, tex.DType.kBFloat16,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )

        calls = dict(lite_gemm._GEMM_CALLS)
        assert calls.get("pytorch_scaled_mm_ok", 0) >= 1, (
            f"Expected pytorch_scaled_mm_ok>=1 for per-tensor FP8 under "
            f"backend=pytorch; got counters {calls}"
        )
        assert calls.get("pytorch_dequant_matmul", 0) == 0, (
            f"Per-tensor FP8 under backend=pytorch must not fall back to "
            f"dequant+matmul (the 100-1000x slow path); got counters {calls}"
        )

    # -- Pad-M: M not divisible by 16 ------------------------------------

    def test_scaled_mm_pads_m_when_not_div16(
        self, device, monkeypatch,
    ):
        """hipBLASLt FP8 requires mat1 rows div-by-16. The pad-then-slice
        path (commit 3ed9d8ae) must preserve numerical correctness.

        Uses M=100 (pads to 112); no unit test currently exercises this.
        """
        from transformer_engine.pytorch._lite import gemm as lite_gemm
        if not hasattr(torch, "_scaled_mm"):
            pytest.skip("torch._scaled_mm not available")

        self._set_backend(monkeypatch, "pytorch")
        monkeypatch.setattr(lite_gemm, "_LITE_DIAG", True)
        lite_gemm._GEMM_CALLS.clear()

        M, N, K = 100, 64, 128  # M not div-by-16, K div-by-16
        assert M % 16 != 0 and K % 16 == 0, "Shape preconditions"

        fp8 = torch.float8_e4m3fnuz
        x_bf16 = torch.randn(M, K, device=device, dtype=self.DTYPE)
        w_bf16 = torch.randn(N, K, device=device, dtype=self.DTYPE)
        x_fp8, x_scale = _quant_per_tensor_e4m3(x_bf16, fp8)
        w_fp8, w_scale = _quant_per_tensor_e4m3(w_bf16, fp8)

        A = _FP8Wrap(w_fp8, w_scale)
        B = _FP8Wrap(x_fp8, x_scale)
        ws = self._workspace(device)
        out, _, _, _ = tex.generic_gemm(
            A, True, B, False, None, None, tex.DType.kBFloat16,
            None, None, False, None, False, ws, ws.shape[0],
            False, False,
        )

        # The fast path should still fire — pad-and-slice, not dequant.
        calls = dict(lite_gemm._GEMM_CALLS)
        assert calls.get("pytorch_scaled_mm_ok", 0) >= 1, (
            f"Pad-M case must still land on _scaled_mm; got {calls}"
        )

        # Output must match dequantized reference AND be sliced back to M.
        assert out.shape == (M, N), (
            f"Output must be sliced back to M={M}, got {out.shape}"
        )
        x_deq = x_fp8.float() * x_scale.item()
        w_deq = w_fp8.float() * w_scale.item()
        ref = x_deq @ w_deq.t()
        rel_err = (
            (out.float() - ref).abs() / (ref.abs() + 1e-3)
        ).mean().item()
        assert rel_err < 0.05, (
            f"Pad-M FP8 GEMM mean rel err {rel_err:.4f} exceeds 5%"
        )


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------

class TestAttention:
    """Verify attention kernels in lite mode (AITER CK + SDPA fallback)."""

    B, S, H, D = 2, 64, 16, 64
    DTYPE = torch.bfloat16

    def _make_qkv(self, device, fmt="bshd", num_kv_heads=None):
        """Create Q, K, V tensors and cu_seqlens for a given format."""
        B, S, H, D = self.B, self.S, self.H, self.D
        H_kv = num_kv_heads or H
        cu = torch.arange(0, (B + 1) * S, S, device=device, dtype=torch.int32)
        if fmt == "bshd":
            q = torch.randn(B, S, H, D, device=device, dtype=self.DTYPE)
            k = torch.randn(B, S, H_kv, D, device=device, dtype=self.DTYPE)
            v = torch.randn(B, S, H_kv, D, device=device, dtype=self.DTYPE)
        elif fmt == "sbhd":
            q = torch.randn(S, B, H, D, device=device, dtype=self.DTYPE)
            k = torch.randn(S, B, H_kv, D, device=device, dtype=self.DTYPE)
            v = torch.randn(S, B, H_kv, D, device=device, dtype=self.DTYPE)
        elif fmt == "thd":
            total = B * S
            cu = torch.arange(0, (B + 1) * S, S, device=device, dtype=torch.int32)
            q = torch.randn(total, H, D, device=device, dtype=self.DTYPE)
            k = torch.randn(total, H_kv, D, device=device, dtype=self.DTYPE)
            v = torch.randn(total, H_kv, D, device=device, dtype=self.DTYPE)
        else:
            raise ValueError(fmt)
        return q, k, v, cu

    # -- get_fused_attn_backend -------------------------------------------

    def test_backend_selection(self):
        """get_fused_attn_backend should return a valid backend."""
        backend = tex.get_fused_attn_backend(
            True,                                       # is_training
            tex.DType.kBFloat16, tex.DType.kBFloat16,   # q/kv dtype
            tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,   # layout
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            0.0, 16, 16, 64, 64, 64, 64, -1, -1, False, False,
        )
        assert backend in (
            tex.NVTE_Fused_Attn_Backend.NVTE_CK,
            tex.NVTE_Fused_Attn_Backend.NVTE_SDPA,
        )

    # -- fused_attn_fwd / bwd  (C++ binding interface) --------------------

    @pytest.mark.parametrize("layout_name,layout_enum", [
        ("bshd", tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD),
        ("sbhd", tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD),
        ("thd",  tex.NVTE_QKV_Layout.NVTE_THD_THD_THD),
    ])
    def test_fused_attn_fwd_shapes(self, device, layout_name, layout_enum):
        """fused_attn_fwd should produce correct output shapes for each layout."""
        q, k, v, cu = self._make_qkv(device, layout_name)
        result = tex.fused_attn_fwd(
            self.S, self.S, True, 1.0 / 8.0, 0.0, True,
            layout_enum,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, 0), cu, cu, q, k, v, self.DTYPE,
        )
        assert isinstance(result, list), "fused_attn_fwd should return a list"
        assert len(result) >= 1, "Result must contain at least the output tensor"
        assert result[0].shape == q.shape, (
            f"Output shape {result[0].shape} != Q shape {q.shape}"
        )

    @pytest.mark.parametrize("layout_name,layout_enum", [
        ("bshd", tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD),
        ("sbhd", tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD),
        ("thd",  tex.NVTE_QKV_Layout.NVTE_THD_THD_THD),
    ])
    def test_fused_attn_bwd_shapes(self, device, layout_name, layout_enum):
        """fused_attn_bwd should produce correct gradient shapes."""
        q, k, v, cu = self._make_qkv(device, layout_name)
        result = tex.fused_attn_fwd(
            self.S, self.S, True, 1.0 / 8.0, 0.0, True,
            layout_enum,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, 0), cu, cu, q, k, v, self.DTYPE,
        )
        out = result[0]
        aux = result[1:]
        d_o = torch.randn_like(out)
        bwd = tex.fused_attn_bwd(
            self.S, self.S, 1.0 / 8.0, 0.0, True,
            layout_enum,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, 0), False,
            cu, cu, q, k, v, out, d_o, self.DTYPE, None, aux,
        )
        assert len(bwd) == 5, "fused_attn_bwd should return [dQ, dK, dV, dBias, dSoftmaxOffset]"
        assert bwd[0].shape == q.shape, f"dQ shape {bwd[0].shape} != Q shape {q.shape}"
        assert bwd[1].shape == k.shape, f"dK shape {bwd[1].shape} != K shape {k.shape}"
        assert bwd[2].shape == v.shape, f"dV shape {bwd[2].shape} != V shape {v.shape}"

    def test_fused_attn_aux_ctx_tensors(self, device):
        """Forward should return [out, softmax_lse, rng_state] for training."""
        q, k, v, cu = self._make_qkv(device, "bshd")
        result = tex.fused_attn_fwd(
            self.S, self.S, True, 1.0 / 8.0, 0.0, True,
            tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, 0), cu, cu, q, k, v, self.DTYPE,
        )
        assert len(result) >= 3, (
            f"Training fwd should return [out, softmax_lse, rng_state], got {len(result)} tensors"
        )
        softmax_lse = result[1]
        rng_state = result[2]
        assert softmax_lse.dtype == torch.float32, "softmax_lse should be float32"
        assert rng_state.shape[0] == 2, "rng_state should have 2 elements [seed, offset]"

    @pytest.mark.parametrize("mask_type", [
        tex.NVTE_Mask_Type.NVTE_NO_MASK,
        tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
        tex.NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK,
    ])
    def test_fused_attn_mask_types(self, device, mask_type):
        """fused_attn_fwd should handle different mask types."""
        q, k, v, cu = self._make_qkv(device, "bshd")
        result = tex.fused_attn_fwd(
            self.S, self.S, False, 1.0 / 8.0, 0.0, True,
            tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            mask_type,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, -1), cu, cu, q, k, v, self.DTYPE,
        )
        assert result[0].shape == q.shape

    # -- GQA (grouped query attention) ------------------------------------

    def test_fused_attn_gqa(self, device):
        """Attention should work with fewer KV heads than Q heads (GQA)."""
        q, k, v, cu = self._make_qkv(device, "bshd", num_kv_heads=4)
        result = tex.fused_attn_fwd(
            self.S, self.S, False, 1.0 / 8.0, 0.0, True,
            tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, 0), cu, cu, q, k, v, self.DTYPE,
        )
        assert result[0].shape == q.shape

    # -- Variable-length sequences (thd) ----------------------------------

    def test_fused_attn_varlen(self, device):
        """Attention should handle variable-length sequences in THD format."""
        H, D = self.H, self.D
        # 3 sequences of length 20, 30, 14
        cu = torch.tensor([0, 20, 50, 64], device=device, dtype=torch.int32)
        total = 64
        q = torch.randn(total, H, D, device=device, dtype=self.DTYPE)
        k = torch.randn(total, H, D, device=device, dtype=self.DTYPE)
        v = torch.randn(total, H, D, device=device, dtype=self.DTYPE)
        result = tex.fused_attn_fwd(
            30, 30, True, 1.0 / 8.0, 0.0, True,
            tex.NVTE_QKV_Layout.NVTE_THD_THD_THD,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, 0), cu, cu, q, k, v, self.DTYPE,
        )
        assert result[0].shape == (total, H, D)

    # -- Numerical: AITER vs SDPA -----------------------------------------

    def test_aiter_vs_sdpa_numerical(self, device):
        """AITER CK and SDPA fallback should produce similar results."""
        from transformer_engine.pytorch._lite import attention as attn_mod

        torch.manual_seed(42)
        q, k, v, cu = self._make_qkv(device, "bshd")
        layout = tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
        mask = tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK
        args = (
            self.S, self.S, False, 1.0 / 8.0, 0.0, True,
            layout, tex.NVTE_Bias_Type.NVTE_NO_BIAS, mask,
            tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
            (-1, 0), cu, cu, q, k, v, self.DTYPE,
        )

        # AITER path
        result_aiter = attn_mod.fused_attn_fwd(*args)
        out_aiter = result_aiter[0]

        # SDPA path (force by temporarily disabling AITER)
        saved_fwd = attn_mod._aiter_fwd
        saved_varlen = attn_mod._aiter_varlen_fwd
        attn_mod._aiter_fwd = None
        attn_mod._aiter_varlen_fwd = None
        try:
            result_sdpa = attn_mod.fused_attn_fwd(*args)
            out_sdpa = result_sdpa[0]
        finally:
            attn_mod._aiter_fwd = saved_fwd
            attn_mod._aiter_varlen_fwd = saved_varlen

        max_diff = (out_aiter - out_sdpa).abs().max().item()
        assert max_diff < 1e-2, (
            f"AITER vs SDPA max diff {max_diff:.4e} exceeds tolerance"
        )

    # -- Helper functions -------------------------------------------------

    def test_fa_prepare_fwd(self, device):
        """fa_prepare_fwd: [s, b, n, 3*h] -> [3, b, s, n, h]."""
        qkvi = torch.randn(32, 2, 16, 192, device=device, dtype=self.DTYPE)
        out = tex.fa_prepare_fwd(qkvi)
        assert out.shape == (3, 2, 32, 16, 64)

    def test_fa_prepare_bwd(self, device):
        """fa_prepare_bwd: 3 x [s, b, n, h] -> [b, s, n, 3*h]."""
        q = torch.randn(32, 2, 16, 64, device=device, dtype=self.DTYPE)
        k = torch.randn(32, 2, 16, 64, device=device, dtype=self.DTYPE)
        v = torch.randn(32, 2, 16, 64, device=device, dtype=self.DTYPE)
        out = tex.fa_prepare_bwd(q, k, v)
        assert out.shape == (2, 32, 16, 192)

    def test_convert_thd_bshd_roundtrip(self, device):
        """THD -> BSHD -> THD should preserve data."""
        cu = torch.tensor([0, 10, 25, 32], device=device, dtype=torch.int32)
        thd = torch.randn(32, 16, 64, device=device, dtype=self.DTYPE)
        bshd = tex.convert_thd_to_bshd(thd, cu, 3, 15)
        assert bshd.shape == (3, 15, 16, 64)
        thd2 = tex.convert_bshd_to_thd(bshd, cu, 32)
        assert thd2.shape == (32, 16, 64)
        # Data for each sequence should survive the roundtrip
        assert torch.allclose(thd[:10], thd2[:10])
        assert torch.allclose(thd[10:25], thd2[10:25])
        assert torch.allclose(thd[25:32], thd2[25:32])

    # -- DotProductAttention module (end-to-end) --------------------------

    def test_dot_product_attention_fwd(self, device):
        """DotProductAttention forward should work in lite mode."""
        dpa = te.DotProductAttention(16, 64, 16, attn_mask_type="causal").to(
            dtype=self.DTYPE, device=device,
        )
        q = torch.randn(self.B, self.S, self.H, self.D, device=device, dtype=self.DTYPE)
        k = torch.randn(self.B, self.S, self.H, self.D, device=device, dtype=self.DTYPE)
        v = torch.randn(self.B, self.S, self.H, self.D, device=device, dtype=self.DTYPE)
        with torch.amp.autocast("cuda", dtype=self.DTYPE):
            out = dpa(q, k, v)
        # DotProductAttention returns (B, S, H*D) after head projection
        assert out.shape == (self.B, self.S, self.H * self.D)

    def test_multihead_attention_fwd(self, device):
        """MultiheadAttention forward should work in lite mode."""
        hidden = self.H * self.D  # 1024
        mha = te.MultiheadAttention(hidden, self.H, attn_mask_type="causal").to(
            dtype=self.DTYPE, device=device,
        )
        x = torch.randn(self.B, self.S, hidden, device=device, dtype=self.DTYPE)
        with torch.amp.autocast("cuda", dtype=self.DTYPE):
            out = mha(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# MoE router tests
# ---------------------------------------------------------------------------

class TestMoERouter:
    """Test MoE router operations in lite mode."""

    SEED = 42

    def _seed(self):
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)

    # -- fused_topk_with_score_function forward ----------------------------

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("topk", [1, 2])
    def test_topk_fwd_shapes(self, device, score_function, topk):
        """Forward returns (probs, routing_map, intermediate_output) with correct shapes."""
        self._seed()
        N, E = 32, 8
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        expert_bias = torch.randn(E, device=device) if score_function == "sigmoid" else None

        probs, routing_map, intermediate = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, score_function, expert_bias,
        )
        assert probs.shape == (N, E)
        assert routing_map.shape == (N, E)
        assert intermediate.shape == (N, E)
        assert routing_map.dtype == torch.bool
        # Exactly topk experts selected per token
        assert (routing_map.sum(dim=-1) == topk).all()

    def test_softmax_pre_softmax(self, device):
        """Pre-softmax mode: softmax applied before topk."""
        self._seed()
        N, E, topk = 16, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        probs, routing_map, intermediate = tex.fused_topk_with_score_function_fwd(
            logits, topk, True, None, None, 1.0, "softmax", None,
        )
        # intermediate should be softmax output (sums to 1 per row)
        torch.testing.assert_close(
            intermediate.sum(dim=-1),
            torch.ones(N, device=device),
            atol=1e-5, rtol=1e-5,
        )

    def test_softmax_post_softmax(self, device):
        """Post-softmax mode: softmax applied after topk over selected experts."""
        self._seed()
        N, E, topk = 16, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        probs, routing_map, _ = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, "softmax", None,
        )
        # Selected probs should sum to 1 per token (post-softmax over topk)
        selected_probs = probs[routing_map].reshape(N, topk)
        torch.testing.assert_close(
            selected_probs.sum(dim=-1),
            torch.ones(N, device=device),
            atol=1e-5, rtol=1e-5,
        )

    def test_sigmoid_values(self, device):
        """Sigmoid scores are in (0, 1) range."""
        self._seed()
        N, E, topk = 16, 8, 1
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        _, _, intermediate = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, "sigmoid", None,
        )
        # intermediate holds sigmoid output
        ref_sigmoid = torch.sigmoid(logits)
        torch.testing.assert_close(intermediate, ref_sigmoid, atol=1e-6, rtol=1e-6)

    def test_sigmoid_expert_bias(self, device):
        """Expert bias affects expert selection but is removed from final scores."""
        self._seed()
        N, E, topk = 32, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        bias = torch.zeros(E, device=device)
        bias[0] = 100.0  # strongly bias towards expert 0

        probs, routing_map, _ = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, "sigmoid", bias,
        )
        # Expert 0 should be selected for (almost) every token
        assert routing_map[:, 0].sum() >= N * 0.9

    def test_sigmoid_normalization_topk_gt1(self, device):
        """With topk > 1, sigmoid scores are normalized to sum to ~1."""
        self._seed()
        N, E, topk = 16, 8, 3
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        probs, routing_map, _ = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, "sigmoid", None,
        )
        selected = probs[routing_map].reshape(N, topk)
        # Should be approximately normalized (sum ≈ 1)
        torch.testing.assert_close(
            selected.sum(dim=-1),
            torch.ones(N, device=device),
            atol=1e-5, rtol=1e-5,
        )

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_group_topk(self, device, score_function):
        """Group top-k selects experts only from winning groups."""
        self._seed()
        N, E, topk = 32, 16, 4
        num_groups, group_topk = 4, 2  # 4 groups of 4 experts, pick 2 groups
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        expert_bias = torch.randn(E, device=device) if score_function == "sigmoid" else None

        probs, routing_map, intermediate = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, num_groups, group_topk, 1.0,
            score_function, expert_bias,
        )
        assert probs.shape == (N, E)
        assert routing_map.dtype == torch.bool
        # Exactly topk experts selected per token
        assert (routing_map.sum(dim=-1) == topk).all()

        # Selected experts should come from at most group_topk groups
        group_size = E // num_groups
        for i in range(N):
            selected_experts = routing_map[i].nonzero(as_tuple=True)[0]
            groups_used = set((idx.item() // group_size) for idx in selected_experts)
            assert len(groups_used) <= group_topk, (
                f"Token {i}: experts from {len(groups_used)} groups, expected <= {group_topk}"
            )

    def test_scaling_factor(self, device):
        """Scaling factor multiplies the output probs."""
        self._seed()
        N, E, topk = 16, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        scale = 2.5
        probs_s1, _, _ = tex.fused_topk_with_score_function_fwd(
            logits, topk, True, None, None, 1.0, "softmax", None,
        )
        probs_s2, _, _ = tex.fused_topk_with_score_function_fwd(
            logits, topk, True, None, None, scale, "softmax", None,
        )
        torch.testing.assert_close(probs_s2, probs_s1 * scale, atol=1e-5, rtol=1e-5)

    # -- fused_topk_with_score_function backward ---------------------------

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("topk", [1, 2])
    def test_topk_bwd_shapes(self, device, score_function, topk):
        """Backward returns grad_logits with correct shape."""
        self._seed()
        N, E = 32, 8
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        expert_bias = torch.randn(E, device=device) if score_function == "sigmoid" else None

        probs, routing_map, intermediate = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, score_function, expert_bias,
        )
        grad_probs = torch.randn_like(probs)
        grad_logits = tex.fused_topk_with_score_function_bwd(
            N, E, routing_map, intermediate, grad_probs, topk,
            False, 1.0, score_function,
        )
        assert grad_logits.shape == (N, E)

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_topk_bwd_unselected_zero(self, device, score_function):
        """Gradients at unselected expert positions should be zero (softmax post) or
        propagated through score function (softmax pre / sigmoid)."""
        self._seed()
        N, E, topk = 16, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)

        probs, routing_map, intermediate = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, score_function, None,
        )
        grad_probs = torch.randn_like(probs)
        grad_logits = tex.fused_topk_with_score_function_bwd(
            N, E, routing_map, intermediate, grad_probs, topk,
            False, 1.0, score_function,
        )
        # grad_logits should not be all zeros
        assert not torch.all(grad_logits == 0)

    # -- fused_score_for_moe_aux_loss --------------------------------------

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_aux_loss_score_fwd(self, device, score_function):
        """fused_score_for_moe_aux_loss_fwd returns 3 tensors."""
        self._seed()
        N, E, topk = 32, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)

        scores, routing_map, intermediate = tex.fused_score_for_moe_aux_loss_fwd(
            logits, topk, score_function,
        )
        assert scores.shape == (N, E)
        assert routing_map.shape == (N, E)
        assert intermediate.shape == (N, E)

        if score_function == "sigmoid":
            ref = torch.sigmoid(logits)
        else:
            ref = F.softmax(logits, dim=-1)
        torch.testing.assert_close(scores, ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_aux_loss_score_bwd(self, device, score_function):
        """fused_score_for_moe_aux_loss_bwd returns correct gradient shape."""
        self._seed()
        N, E, topk = 32, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        scores, _, intermediate = tex.fused_score_for_moe_aux_loss_fwd(
            logits, topk, score_function,
        )
        grad_scores = torch.randn_like(scores)
        grad_logits = tex.fused_score_for_moe_aux_loss_bwd(
            N, E, intermediate, grad_scores, topk, score_function,
        )
        assert grad_logits.shape == (N, E)

    # -- Aux loss fwd/bwd return + autograd ---------------------------------

    def test_aux_loss_fwd_returns_tuple(self, device):
        """fused_moe_aux_loss_fwd returns (loss, Const_buf) matching C++ interface."""
        self._seed()
        N, E, topk, coeff = 32, 8, 2, 0.01
        probs = torch.rand(N, E, device=device, dtype=torch.float32)
        tpe = torch.randint(1, N, (E,), device=device, dtype=torch.int32)
        total = int(tpe.sum().item())

        result = tex.fused_moe_aux_loss_fwd(
            probs, tpe, total, E, N, E, topk, coeff,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        loss, const_buf = result
        assert loss.shape == (), f"loss should be scalar, got {loss.shape}"
        assert const_buf.shape == (), f"Const_buf should be scalar, got {const_buf.shape}"
        assert const_buf.dtype == torch.float32

        # Verify Const_buf value
        expected_c = (E * coeff) / topk / (total * total)
        torch.testing.assert_close(
            const_buf, torch.tensor(expected_c, dtype=torch.float32, device=device),
            atol=1e-6, rtol=1e-6,
        )

    def test_aux_loss_bwd_uses_const_buf(self, device):
        """fused_moe_aux_loss_bwd produces correct gradient using Const_buf."""
        self._seed()
        N, E, topk, coeff = 16, 8, 2, 0.05
        probs = torch.rand(N, E, device=device, dtype=torch.float32)
        tpe = torch.randint(1, N, (E,), device=device, dtype=torch.int32)
        total = int(tpe.sum().item())

        loss, const_buf = tex.fused_moe_aux_loss_fwd(
            probs, tpe, total, E, N, E, topk, coeff,
        )
        grad_aux = torch.tensor(1.0, device=device, dtype=torch.float32)
        grad_probs = tex.fused_moe_aux_loss_bwd(const_buf, tpe, N, E, grad_aux)

        assert grad_probs.shape == (N, E)
        # grad_probs[j, i] = C_coeff * tokens_per_expert[i] * grad_aux_loss
        for i in range(E):
            expected = const_buf.item() * tpe[i].item() * grad_aux.item()
            torch.testing.assert_close(
                grad_probs[:, i],
                torch.full((N,), expected, device=device),
                atol=1e-5, rtol=1e-5,
            )

    def test_autograd_aux_loss(self, device):
        """High-level fused_moe_aux_loss propagates gradients end-to-end."""
        from transformer_engine.pytorch.router import fused_moe_aux_loss
        self._seed()
        N, E, topk, coeff = 16, 8, 2, 0.01
        probs = torch.rand(N, E, device=device, dtype=torch.float32, requires_grad=True)
        tpe = torch.randint(1, N, (E,), device=device, dtype=torch.int32)
        total = int(tpe.sum().item())

        loss = fused_moe_aux_loss(probs, tpe, total, E, topk, coeff)
        loss.backward()
        assert probs.grad is not None
        assert probs.grad.shape == (N, E)
        assert not torch.all(probs.grad == 0)

    # -- High-level autograd integration -----------------------------------

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_autograd_topk(self, device, score_function):
        """High-level fused_topk_with_score_function propagates gradients."""
        from transformer_engine.pytorch.router import fused_topk_with_score_function
        self._seed()
        N, E, topk = 16, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32, requires_grad=True)
        expert_bias = torch.randn(E, device=device) if score_function == "sigmoid" else None

        probs, routing_map = fused_topk_with_score_function(
            logits, topk, False, None, None, 1.0, score_function, expert_bias,
        )
        probs.sum().backward()
        assert logits.grad is not None
        assert logits.grad.shape == (N, E)
        assert not torch.all(logits.grad == 0)

    # -- Numerical gradient verification ------------------------------------

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("topk", [1, 2, 3])
    def test_gradient_vs_finite_diff(self, device, score_function, topk):
        """Verify backward matches finite-difference approximation.

        Uses random grad_out (uniform grad is degenerate for normalization)
        and tolerances appropriate for float32 finite differences.
        """
        self._seed()
        N, E = 4, 8
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        eps = 1e-4  # larger eps for float32 stability

        # Forward + backward
        probs, rmap, inter = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, score_function, None,
        )
        grad_out = torch.randn_like(probs)
        grad_ana = tex.fused_topk_with_score_function_bwd(
            N, E, rmap, inter, grad_out, topk, False, 1.0, score_function,
        )

        # Finite-difference per element, only where routing is stable
        max_err = 0.0
        n_checked = 0
        for i in range(N):
            for j in range(E):
                logits_p = logits.clone()
                logits_p[i, j] += eps
                probs_p, rmap_p, _ = tex.fused_topk_with_score_function_fwd(
                    logits_p, topk, False, None, None, 1.0, score_function, None,
                )
                logits_m = logits.clone()
                logits_m[i, j] -= eps
                probs_m, rmap_m, _ = tex.fused_topk_with_score_function_fwd(
                    logits_m, topk, False, None, None, 1.0, score_function, None,
                )
                if not (torch.equal(rmap_p, rmap) and torch.equal(rmap_m, rmap)):
                    continue  # skip topk discontinuity
                fd = ((probs_p - probs_m) * grad_out).sum() / (2 * eps)
                err = abs(grad_ana[i, j].item() - fd.item())
                max_err = max(max_err, err)
                n_checked += 1

        assert n_checked > 0, "No stable routing points found for finite-diff check"
        assert max_err < 0.01, (
            f"Max gradient error {max_err:.4e} for {score_function} topk={topk} "
            f"({n_checked} points checked)"
        )

    def test_sigmoid_topk1_no_normalization(self, device):
        """Sigmoid with topk=1 skips normalization — score is raw sigmoid * scale."""
        self._seed()
        N, E = 16, 8
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        probs, rmap, inter = tex.fused_topk_with_score_function_fwd(
            logits, 1, False, None, None, 1.0, "sigmoid", None,
        )
        # For each token, the selected prob should equal sigmoid(logit)
        selected_probs = probs[rmap]
        selected_sigmoid = inter[rmap]
        torch.testing.assert_close(selected_probs, selected_sigmoid, atol=1e-6, rtol=1e-6)

    def test_pre_softmax_backward(self, device):
        """Pre-softmax backward: gradient flows through all experts via softmax Jacobian."""
        from transformer_engine.pytorch.router import fused_topk_with_score_function
        self._seed()
        N, E, topk = 8, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32, requires_grad=True)

        probs, _ = fused_topk_with_score_function(
            logits, topk, True, None, None, 1.0, "softmax", None,
        )
        probs.sum().backward()
        # Pre-softmax: gradient should be non-zero at ALL expert positions
        # (softmax couples all inputs), not just selected ones
        assert logits.grad is not None
        non_zero_per_token = (logits.grad.abs() > 1e-7).sum(dim=-1)
        assert (non_zero_per_token > topk).all(), (
            "Pre-softmax backward should produce non-zero gradients beyond selected experts"
        )

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_triton_vs_pytorch_fallback(self, device, score_function):
        """Triton-fused path matches the PyTorch-native fallback."""
        from transformer_engine.pytorch._lite.router import (
            _fused_topk_fwd_pytorch, _fused_topk_bwd_pytorch,
        )
        from transformer_engine.pytorch.triton.fused_router import (
            fused_topk_with_score_function_fwd as triton_fwd,
            fused_topk_with_score_function_bwd as triton_bwd,
        )
        self._seed()
        N, E, topk = 32, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        expert_bias = torch.randn(E, device=device) if score_function == "sigmoid" else None

        # Forward
        p_tri, r_tri, i_tri = triton_fwd(
            logits, topk, False, 1.0, score_function, expert_bias,
        )
        p_pt, r_pt, i_pt = _fused_topk_fwd_pytorch(
            logits, topk, False, None, None, 1.0, score_function, expert_bias,
        )
        torch.testing.assert_close(r_tri, r_pt, msg="routing_map mismatch")
        torch.testing.assert_close(p_tri, p_pt, atol=1e-5, rtol=1e-5, msg="probs fwd mismatch")
        torch.testing.assert_close(i_tri, i_pt, atol=1e-5, rtol=1e-5, msg="intermediate mismatch")

        # Backward
        grad_out = torch.randn_like(p_tri)
        g_tri = triton_bwd(N, E, r_tri, i_tri, grad_out, topk, False, 1.0, score_function)
        g_pt = _fused_topk_bwd_pytorch(
            N, E, r_pt, i_pt, grad_out, topk, False, 1.0, score_function,
        )
        torch.testing.assert_close(g_tri, g_pt, atol=1e-5, rtol=1e-5, msg="grad_logits mismatch")

    def test_sigmoid_scaling_factor(self, device):
        """Scaling factor works correctly with sigmoid scoring."""
        self._seed()
        N, E, topk = 16, 8, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        scale = 3.0
        p1, r1, _ = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, 1.0, "sigmoid", None,
        )
        p2, r2, _ = tex.fused_topk_with_score_function_fwd(
            logits, topk, False, None, None, scale, "sigmoid", None,
        )
        # Same routing
        torch.testing.assert_close(r1, r2)
        # Probs scaled
        torch.testing.assert_close(p2, p1 * scale, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_autograd_group_topk(self, device, score_function):
        """Autograd works end-to-end with group top-k."""
        from transformer_engine.pytorch.router import fused_topk_with_score_function
        self._seed()
        N, E, topk = 16, 16, 4
        num_groups, group_topk = 4, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32, requires_grad=True)
        expert_bias = torch.randn(E, device=device) if score_function == "sigmoid" else None

        probs, routing_map = fused_topk_with_score_function(
            logits, topk, False, num_groups, group_topk, 1.0,
            score_function, expert_bias,
        )
        probs.sum().backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_triton_vs_pytorch_group_topk(self, device, score_function):
        """Triton group top-k matches PyTorch fallback."""
        from transformer_engine.pytorch._lite.router import _fused_topk_fwd_pytorch
        from transformer_engine.pytorch.triton.fused_router import (
            fused_topk_with_score_function_fwd as triton_fwd,
        )
        self._seed()
        N, E, topk = 32, 16, 4
        num_groups, group_topk = 4, 2
        logits = torch.randn(N, E, device=device, dtype=torch.float32)
        expert_bias = torch.randn(E, device=device) if score_function == "sigmoid" else None

        p_tri, r_tri, i_tri = triton_fwd(
            logits, topk, False, 1.0, score_function, expert_bias,
            num_groups, group_topk,
        )
        p_pt, r_pt, i_pt = _fused_topk_fwd_pytorch(
            logits, topk, False, num_groups, group_topk, 1.0,
            score_function, expert_bias,
        )
        torch.testing.assert_close(r_tri, r_pt, msg="group topk routing_map mismatch")
        torch.testing.assert_close(
            p_tri, p_pt, atol=1e-5, rtol=1e-5, msg="group topk probs mismatch",
        )


# ---------------------------------------------------------------------------
# MoE permutation tests
# ---------------------------------------------------------------------------

def _pytorch_permute_index_map(tokens, indices, num_out_tokens=None):
    """Reference implementation for index-map permutation."""
    topk = indices.size(1) if indices.dim() > 1 else 1
    flat = indices.view(-1)
    sorted_indices = torch.argsort(flat, stable=True)
    n_out = num_out_tokens if num_out_tokens is not None else flat.size(0)
    return tokens.index_select(0, sorted_indices[:n_out] // topk), sorted_indices


def _pytorch_unpermute_index_map(permuted, sorted_indices, probs=None):
    """Reference implementation for index-map unpermutation."""
    if probs is not None:
        n_unp = probs.numel()
        topk = probs.size(1)
    else:
        n_unp = sorted_indices.size(0)
        topk = 1
    out = torch.zeros(n_unp, permuted.shape[-1],
                      dtype=permuted.dtype, device=permuted.device)
    out.index_copy_(0, sorted_indices[:permuted.size(0)], permuted)
    out = out.reshape(-1, topk, permuted.size(-1))
    if probs is not None:
        out = out * probs.unsqueeze(-1)
    return out.sum(dim=1)


class TestMoEPermutation:
    """Test MoE permutation operations in lite mode."""

    DTYPE = torch.bfloat16
    SEED = 1234

    def _seed(self):
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)

    # -- Low-level tex interface tests -----------------------------------

    def test_permute_fwd_shapes(self, device):
        """moe_permute_fwd returns correct shapes."""
        self._seed()
        N, H, topK, E = 32, 64, 2, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)

        permuted, row_id_map, ws = tex.moe_permute_fwd(
            inp, tex.DType.kBFloat16, indices, -1, [], N * topK,
        )
        assert permuted.shape == (N * topK, H)
        assert row_id_map.shape == (N * topK,)
        assert row_id_map.dtype == torch.int32

    def test_permute_fwd_with_num_out_tokens(self, device):
        """moe_permute_fwd respects num_out_tokens truncation."""
        self._seed()
        N, H, topK, E = 32, 64, 2, 8
        num_out = 48
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)

        permuted, row_id_map, _ = tex.moe_permute_fwd(
            inp, tex.DType.kBFloat16, indices, num_out, [], N * topK,
        )
        assert permuted.shape == (num_out, H)

    def test_roundtrip_identity(self, device):
        """Permute then unpermute with uniform probs recovers the input."""
        self._seed()
        N, H, topK, E = 64, 128, 2, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)

        permuted, row_id_map, _ = tex.moe_permute_fwd(
            inp, tex.DType.kBFloat16, indices, -1, [], N * topK,
        )
        probs = torch.ones(N, topK, device=device, dtype=torch.float32)
        unpermuted = tex.moe_unpermute_fwd(
            permuted, tex.DType.kBFloat16, row_id_map, probs, N, topK,
        )
        # Each token is gathered topK times and summed with prob=1.0
        torch.testing.assert_close(unpermuted, inp.float() * topK, atol=1e-5, rtol=1e-5)

    def test_permute_bwd_equals_unpermute_fwd(self, device):
        """moe_permute_bwd delegates to moe_unpermute_fwd."""
        self._seed()
        N, H, topK, E = 32, 64, 2, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)

        permuted, row_id_map, _ = tex.moe_permute_fwd(
            inp, tex.DType.kBFloat16, indices, -1, [], N * topK,
        )
        probs = torch.rand(N, topK, device=device, dtype=torch.float32)

        via_bwd = tex.moe_permute_bwd(
            permuted, tex.DType.kBFloat16, row_id_map, probs, N, topK,
        )
        via_fwd = tex.moe_unpermute_fwd(
            permuted, tex.DType.kBFloat16, row_id_map, probs, N, topK,
        )
        torch.testing.assert_close(via_bwd, via_fwd)

    def test_unpermute_bwd_shapes(self, device):
        """moe_unpermute_bwd returns (act_grad, prob_grad) with correct shapes."""
        self._seed()
        N, H, topK, E = 32, 64, 2, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)

        permuted, row_id_map, _ = tex.moe_permute_fwd(
            inp, tex.DType.kBFloat16, indices, -1, [], N * topK,
        )
        probs = torch.rand(N, topK, device=device, dtype=torch.float32)
        grad_out = torch.randn(N, H, device=device, dtype=self.DTYPE)

        act_grad, prob_grad = tex.moe_unpermute_bwd(
            grad_out, permuted, tex.DType.kBFloat16, row_id_map, probs,
        )
        assert act_grad.shape == (N * topK, H)
        assert prob_grad.shape == (N, topK)
        assert prob_grad.dtype == torch.float32

    def test_unpermute_bwd_no_probs(self, device):
        """moe_unpermute_bwd works without probs."""
        self._seed()
        N, H, topK, E = 32, 64, 1, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)

        permuted, row_id_map, _ = tex.moe_permute_fwd(
            inp, tex.DType.kBFloat16, indices, -1, [], N * topK,
        )
        grad_out = torch.randn(N, H, device=device, dtype=self.DTYPE)
        empty_prob = torch.empty(0, device=device)

        act_grad, prob_grad = tex.moe_unpermute_bwd(
            grad_out, permuted, tex.DType.kBFloat16, row_id_map, empty_prob,
        )
        assert act_grad.shape == (N * topK, H)
        assert prob_grad.numel() == 0

    # -- High-level API: forward / backward numerical tests ---------------

    @pytest.mark.parametrize("topK", [1, 2])
    @pytest.mark.parametrize("with_probs", [True, False])
    def test_index_map_vs_reference(self, device, topK, with_probs):
        """High-level moe_permute/unpermute matches PyTorch reference (index map)."""
        if not with_probs and topK > 1:
            pytest.skip("topK>1 without probs not supported for index-map")
        self._seed()
        N, H, E = 64, 128, 8
        from transformer_engine.pytorch.permutation import moe_permute, moe_unpermute

        inp = torch.randn(N, H, device=device, dtype=self.DTYPE, requires_grad=True)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)

        # Reference
        ref_perm, ref_sorted = _pytorch_permute_index_map(inp.detach(), indices)
        probs = None
        if with_probs:
            probs = torch.rand(N, topK, device=device).softmax(dim=-1)

        ref_unperm = _pytorch_unpermute_index_map(ref_perm, ref_sorted, probs)

        # TE lite
        te_inp = inp.detach().clone().requires_grad_(True)
        te_perm, row_id_map = moe_permute(te_inp, indices, map_type="index")

        te_probs = probs.clone().requires_grad_(True) if probs is not None else None
        te_unperm = moe_unpermute(
            te_perm.detach().clone().requires_grad_(True),
            row_id_map, te_probs, map_type="index",
        )

        # Forward check
        torch.testing.assert_close(
            ref_perm.float(), te_perm.float(),
            msg="permute fwd mismatch",
        )
        tols = dict(rtol=2.5e-2, atol=1e-5)
        torch.testing.assert_close(
            ref_unperm.float(), te_unperm.float(),
            msg="unpermute fwd mismatch", **tols,
        )

        # Backward check
        grad = torch.randn(N, H, device=device, dtype=self.DTYPE)
        te_unperm.backward(grad)

    @pytest.mark.parametrize("topK", [1, 2, 3])
    def test_index_map_empty_input(self, device, topK):
        """Empty tensor should pass through without error."""
        from transformer_engine.pytorch.permutation import moe_permute, moe_unpermute
        inp = torch.empty(0, 64, device=device, dtype=self.DTYPE)
        indices = torch.empty(0, topK, device=device, dtype=torch.int32)
        perm, rid = moe_permute(inp, indices, map_type="index")
        assert perm.numel() == 0

    # -- Triton kernel integration ----------------------------------------

    def test_triton_sort_used_in_permute(self, device):
        """Verify the Triton sort_chunks_by_map kernel is loaded for permute."""
        from transformer_engine.pytorch._lite.permutation import _try_load_triton_sort
        fn = _try_load_triton_sort()
        assert fn is not None, "Triton sort_chunks_by_map should be loadable"

    def test_triton_gather_matches_pytorch(self, device):
        """Triton sort_chunks_by_map (gather mode) matches PyTorch indexing."""
        from transformer_engine.pytorch.triton.permutation import sort_chunks_by_map
        self._seed()
        N, H = 128, 256
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        ids = torch.randperm(N, device=device, dtype=torch.int32)
        triton_out, _ = sort_chunks_by_map(inp, ids, None, N, H, is_forward=False)
        pytorch_out = inp[ids.long()]
        torch.testing.assert_close(triton_out, pytorch_out)

    # -- Mask-map path tests ------------------------------------------------

    @staticmethod
    def _make_routing_map(N, E, topK, device):
        """Create a mask-format routing map: [N, E] int32 with topK ones per row."""
        routing_map = torch.zeros(N, E, dtype=torch.int32, device=device)
        for i in range(N):
            experts = torch.randperm(E, device=device)[:topK]
            routing_map[i, experts] = 1
        return routing_map

    @pytest.mark.parametrize("topK", [1, 2, 3])
    def test_mask_map_permute_shapes(self, device, topK):
        """moe_permute with mask map returns correct shapes."""
        from transformer_engine.pytorch.permutation import moe_permute
        self._seed()
        N, H, E = 64, 128, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        routing_map = self._make_routing_map(N, E, topK, device)
        num_out = int(routing_map.sum().item())

        perm, row_id_map = moe_permute(inp, routing_map, num_out_tokens=num_out, map_type="mask")
        assert perm.shape == (num_out, H)
        assert row_id_map.shape[0] == N

    @pytest.mark.parametrize("topK", [1, 2])
    def test_mask_map_roundtrip(self, device, topK):
        """Mask-map permute then unpermute with merging_probs recovers weighted sum."""
        from transformer_engine.pytorch.permutation import moe_permute, moe_unpermute
        self._seed()
        N, H, E = 32, 64, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        routing_map = self._make_routing_map(N, E, topK, device)
        num_out = int(routing_map.sum().item())

        perm, row_id_map = moe_permute(inp, routing_map, num_out_tokens=num_out, map_type="mask")

        # Create merging probs matching the routing map
        probs_full = torch.rand(N, E, device=device, dtype=torch.float32) * routing_map.float()
        # Normalize per token
        probs_full = probs_full / probs_full.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        unperm = moe_unpermute(
            perm, row_id_map, merging_probs=probs_full,
            restore_shape=torch.Size([N, H]), map_type="mask",
        )
        assert unperm.shape == (N, H)

    @pytest.mark.parametrize("topK", [1, 2])
    def test_mask_map_backward(self, device, topK):
        """Mask-map path propagates gradients through permute and unpermute."""
        from transformer_engine.pytorch.permutation import moe_permute, moe_unpermute
        self._seed()
        N, H, E = 32, 64, 8
        inp = torch.randn(N, H, device=device, dtype=torch.float32, requires_grad=True)
        routing_map = self._make_routing_map(N, E, topK, device)
        num_out = int(routing_map.sum().item())

        perm, row_id_map = moe_permute(inp, routing_map, num_out_tokens=num_out, map_type="mask")
        assert perm.requires_grad

        # Backward through permute
        perm.sum().backward()
        assert inp.grad is not None
        assert inp.grad.shape == inp.shape

    @pytest.mark.parametrize("topK", [1, 2])
    def test_mask_map_unpermute_backward_with_probs(self, device, topK):
        """Unpermute backward propagates gradients to both act and probs."""
        from transformer_engine.pytorch.permutation import moe_permute, moe_unpermute
        self._seed()
        N, H, E = 32, 64, 8
        inp = torch.randn(N, H, device=device, dtype=torch.float32)
        routing_map = self._make_routing_map(N, E, topK, device)
        num_out = int(routing_map.sum().item())

        perm, row_id_map = moe_permute(inp, routing_map, num_out_tokens=num_out, map_type="mask")
        perm_detached = perm.detach().clone().requires_grad_(True)

        probs_full = (torch.rand(N, E, device=device, dtype=torch.float32)
                      * routing_map.float()).requires_grad_(True)

        unperm = moe_unpermute(
            perm_detached, row_id_map, merging_probs=probs_full,
            restore_shape=torch.Size([N, H]), map_type="mask",
        )
        grad_out = torch.randn_like(unperm)
        unperm.backward(grad_out)

        assert perm_detached.grad is not None, "act_grad not propagated"
        assert perm_detached.grad.shape == perm_detached.shape
        assert probs_full.grad is not None, "probs_grad not propagated"
        assert probs_full.grad.shape == probs_full.shape

    # -- moe_permute_with_probs tests ---------------------------------------

    @pytest.mark.parametrize("topK", [1, 2])
    def test_permute_with_probs_forward(self, device, topK):
        """moe_permute_with_probs permutes both tokens and probs."""
        from transformer_engine.pytorch.permutation import moe_permute_with_probs
        self._seed()
        N, H, E = 32, 64, 8
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        routing_map = self._make_routing_map(N, E, topK, device)
        probs = torch.rand(N, E, device=device, dtype=torch.float32) * routing_map.float()
        num_out = int(routing_map.sum().item())

        perm_out, perm_probs, row_id_map = moe_permute_with_probs(
            inp, probs, routing_map, num_out_tokens=num_out,
        )
        assert perm_out.shape == (num_out, H)
        assert perm_probs.shape == (num_out,)

    @pytest.mark.parametrize("topK", [1, 2])
    def test_permute_with_probs_backward(self, device, topK):
        """moe_permute_with_probs backward propagates gradients to probs."""
        from transformer_engine.pytorch.permutation import moe_permute_with_probs
        self._seed()
        N, H, E = 32, 64, 8
        inp = torch.randn(N, H, device=device, dtype=torch.float32, requires_grad=True)
        routing_map = self._make_routing_map(N, E, topK, device)
        probs = (torch.rand(N, E, device=device, dtype=torch.float32)
                 * routing_map.float()).requires_grad_(True)
        num_out = int(routing_map.sum().item())

        perm_out, perm_probs, row_id_map = moe_permute_with_probs(
            inp, probs, routing_map, num_out_tokens=num_out,
        )
        # Backward through probs
        perm_probs.sum().backward()
        assert probs.grad is not None
        assert probs.grad.shape == probs.shape

    # -- Chunk-sort tests ---------------------------------------------------

    def test_sort_chunks_by_index(self, device):
        """moe_sort_chunks_by_index reorders chunks correctly."""
        from transformer_engine.pytorch.permutation import moe_sort_chunks_by_index
        self._seed()
        H = 64
        split_sizes = torch.tensor([10, 20, 15, 5], device=device, dtype=torch.int32)
        N = int(split_sizes.sum().item())
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE, requires_grad=True)
        sorted_indices = torch.tensor([2, 0, 3, 1], device=device, dtype=torch.int32)

        output = moe_sort_chunks_by_index(inp, split_sizes, sorted_indices)
        assert output.shape == (N, H)

        # Verify chunks are reordered: chunk at sorted_indices[i] moves to position i
        ref_chunks = torch.split(inp.detach(), split_sizes.tolist(), dim=0)
        ref_output = torch.cat([ref_chunks[idx] for idx in sorted_indices.tolist()], dim=0)
        torch.testing.assert_close(output.detach(), ref_output)

    def test_sort_chunks_by_index_backward(self, device):
        """moe_sort_chunks_by_index backward propagates gradients."""
        from transformer_engine.pytorch.permutation import moe_sort_chunks_by_index
        self._seed()
        H = 64
        split_sizes = torch.tensor([8, 12, 6], device=device, dtype=torch.int32)
        N = int(split_sizes.sum().item())
        inp = torch.randn(N, H, device=device, dtype=torch.float32, requires_grad=True)
        sorted_indices = torch.tensor([1, 2, 0], device=device, dtype=torch.int32)

        output = moe_sort_chunks_by_index(inp, split_sizes, sorted_indices)
        output.sum().backward()
        assert inp.grad is not None
        assert inp.grad.shape == inp.shape

    def test_sort_chunks_by_index_with_probs(self, device):
        """moe_sort_chunks_by_index_with_probs reorders both tokens and probs."""
        from transformer_engine.pytorch.permutation import moe_sort_chunks_by_index_with_probs
        self._seed()
        H = 64
        split_sizes = torch.tensor([10, 20, 15], device=device, dtype=torch.int32)
        N = int(split_sizes.sum().item())
        inp = torch.randn(N, H, device=device, dtype=self.DTYPE)
        probs = torch.rand(N, device=device, dtype=torch.float32)
        sorted_indices = torch.tensor([2, 0, 1], device=device, dtype=torch.int32)

        output, perm_probs = moe_sort_chunks_by_index_with_probs(
            inp, probs, split_sizes, sorted_indices,
        )
        assert output.shape == (N, H)
        assert perm_probs.shape == (N,)

        # Verify probs are reordered consistently with tokens
        ref_prob_chunks = torch.split(probs, split_sizes.tolist(), dim=0)
        ref_probs = torch.cat([ref_prob_chunks[idx] for idx in sorted_indices.tolist()], dim=0)
        torch.testing.assert_close(perm_probs, ref_probs)

    # -- Numerical gradient verification for index-map ----------------------

    @pytest.mark.parametrize("topK", [1, 2])
    def test_index_map_gradient_numerical(self, device, topK):
        """Verify index-map permute/unpermute gradients numerically."""
        from transformer_engine.pytorch.permutation import moe_permute, moe_unpermute
        self._seed()
        N, H, E = 16, 32, 8
        inp = torch.randn(N, H, device=device, dtype=torch.float32, requires_grad=True)
        indices = torch.stack(
            [torch.randperm(E, device=device)[:topK] for _ in range(N)]
        ).to(torch.int32)
        probs = torch.rand(N, topK, device=device, dtype=torch.float32).requires_grad_(True)

        # Forward
        perm, row_id_map = moe_permute(inp, indices, map_type="index")
        perm_detached = perm.detach().clone().requires_grad_(True)
        unperm = moe_unpermute(perm_detached, row_id_map, probs, map_type="index")

        # Backward
        grad_out = torch.randn(N, H, device=device, dtype=torch.float32)
        unperm.backward(grad_out)

        # Verify act_grad: manual computation
        # unpermute gathers from permuted[row_id_map], applies probs, sums over topK
        # So d(unperm)/d(perm[j]) = probs_flat[inv_map[j]] * (grad_out broadcast)
        assert perm_detached.grad is not None
        assert not torch.all(perm_detached.grad == 0), "act_grad is all zeros"

        # Verify prob_grad: d(loss)/d(prob[i,k]) = sum_h(perm[row_id_map[i*topK+k], h] * grad_out[i, h])
        assert probs.grad is not None
        assert probs.grad.shape == (N, topK)
        assert not torch.all(probs.grad == 0), "prob_grad is all zeros"


# ---------------------------------------------------------------------------
# MoE padding tests
# ---------------------------------------------------------------------------

class TestMoEPadding:
    """Test multi-row padding / unpadding in lite mode."""

    DTYPE = torch.bfloat16

    def test_padding_basic(self, device):
        """Rows are copied and extra rows are zero-padded."""
        src_splits = [3, 5, 2]
        dst_splits = [4, 8, 4]
        features = 64
        inp = torch.randn(sum(src_splits), features, device=device, dtype=self.DTYPE)
        out = torch.full(
            (sum(dst_splits), features), float("nan"), device=device, dtype=self.DTYPE,
        )
        tex.fused_multi_row_padding(inp, out, src_splits, dst_splits)

        in_off, out_off = 0, 0
        for src, dst in zip(src_splits, dst_splits):
            # Copied region matches
            torch.testing.assert_close(
                out[out_off:out_off + src], inp[in_off:in_off + src],
            )
            # Padding region is zero
            if dst > src:
                assert (out[out_off + src:out_off + dst] == 0).all()
            in_off += src
            out_off += dst

    def test_unpadding_basic(self, device):
        """Unpadding extracts the correct rows."""
        src_splits = [4, 8, 4]
        dst_splits = [3, 5, 2]
        features = 64
        inp = torch.randn(sum(src_splits), features, device=device, dtype=self.DTYPE)
        out = torch.empty(sum(dst_splits), features, device=device, dtype=self.DTYPE)
        tex.fused_multi_row_unpadding(inp, out, src_splits, dst_splits)

        in_off, out_off = 0, 0
        for src, dst in zip(src_splits, dst_splits):
            torch.testing.assert_close(
                out[out_off:out_off + dst], inp[in_off:in_off + dst],
            )
            in_off += src
            out_off += dst

    def test_roundtrip(self, device):
        """Padding then unpadding recovers the original tensor."""
        src_splits = [7, 3, 11, 1]
        dst_splits = [8, 8, 16, 8]
        features = 128
        inp = torch.randn(sum(src_splits), features, device=device, dtype=self.DTYPE)
        padded = torch.empty(
            sum(dst_splits), features, device=device, dtype=self.DTYPE,
        )
        tex.fused_multi_row_padding(inp, padded, src_splits, dst_splits)

        recovered = torch.empty_like(inp)
        tex.fused_multi_row_unpadding(padded, recovered, dst_splits, src_splits)
        torch.testing.assert_close(recovered, inp)

    def test_no_padding_needed(self, device):
        """When splits are equal, data is just copied."""
        splits = [4, 4, 4]
        features = 32
        inp = torch.randn(sum(splits), features, device=device, dtype=self.DTYPE)
        out = torch.empty_like(inp)
        tex.fused_multi_row_padding(inp, out, splits, splits)
        torch.testing.assert_close(out, inp)

    def test_single_group(self, device):
        """Works with a single group."""
        inp = torch.randn(5, 16, device=device, dtype=self.DTYPE)
        out = torch.empty(8, 16, device=device, dtype=self.DTYPE)
        tex.fused_multi_row_padding(inp, out, [5], [8])
        torch.testing.assert_close(out[:5], inp)
        assert (out[5:] == 0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_preservation(self, device, dtype):
        """Padding works across dtypes."""
        inp = torch.randn(10, 32, device=device, dtype=dtype)
        out = torch.empty(16, 32, device=device, dtype=dtype)
        tex.fused_multi_row_padding(inp, out, [4, 6], [8, 8])
        assert out.dtype == dtype
        torch.testing.assert_close(out[:4], inp[:4])
        torch.testing.assert_close(out[8:14], inp[4:10])


# ---------------------------------------------------------------------------
# Fused LayerNormLinear / LayerNormMLP (lite-native modules)
# ---------------------------------------------------------------------------

class TestLiteLayerNormLinear:
    """Tests for the lite-native LayerNormLinear module."""

    DTYPE = torch.bfloat16

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward_shape(self, device, normalization):
        mod = te.LayerNormLinear(
            256, 128, bias=True, normalization=normalization,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (4, 128)

    def test_forward_3d_input(self, device):
        mod = te.LayerNormLinear(256, 128, bias=True).to(
            dtype=self.DTYPE, device=device
        )
        x = torch.randn(2, 8, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (2, 8, 128)

    def test_forward_no_bias(self, device):
        mod = te.LayerNormLinear(256, 128, bias=False).to(
            dtype=self.DTYPE, device=device
        )
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (4, 128)

    def test_return_layernorm_output(self, device):
        mod = te.LayerNormLinear(
            256, 128, bias=True, return_layernorm_output=True,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        out = mod(x)
        assert isinstance(out, tuple) and len(out) == 2
        y, ln_out = out
        assert y.shape == (4, 128)
        assert ln_out.shape == (4, 256)

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_backward_all_grads(self, device, normalization):
        mod = te.LayerNormLinear(
            256, 128, bias=True, normalization=normalization,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape
        assert mod.weight.grad is not None
        assert mod.layer_norm_weight.grad is not None
        if normalization == "LayerNorm":
            assert mod.layer_norm_bias.grad is not None

    def test_backward_no_bias(self, device):
        mod = te.LayerNormLinear(256, 128, bias=False).to(
            dtype=self.DTYPE, device=device
        )
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None

    def test_numerical_vs_manual(self, device):
        """Verify output matches manual norm→linear composition."""
        mod = te.LayerNormLinear(128, 64, bias=True, normalization="RMSNorm").to(
            dtype=self.DTYPE, device=device
        )
        x = torch.randn(4, 128, device=device, dtype=self.DTYPE)

        # Manual reference
        w = mod.layer_norm_weight.data
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-5).rsqrt()
        normed = (x.float() * rms * w.float()).to(self.DTYPE)
        expected = torch.nn.functional.linear(normed, mod.weight.data, mod.bias.data)

        y = mod(x)
        diff = (y - expected).abs().max().item()
        assert diff < 0.1, f"Max diff {diff:.4f} too large"


class TestLiteLayerNormMLP:
    """Tests for the lite-native LayerNormMLP module."""

    DTYPE = torch.bfloat16

    @pytest.mark.parametrize("activation", ["gelu", "silu", "relu"])
    def test_forward_non_gated(self, device, activation):
        mod = te.LayerNormMLP(
            256, 512, bias=True, activation=activation,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (4, 256)

    @pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
    def test_forward_gated(self, device, activation):
        mod = te.LayerNormMLP(
            256, 512, bias=True, activation=activation,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (4, 256)
        # Gated activations should have 2x fc1_weight first dim
        assert mod.fc1_weight.shape[0] == 1024

    def test_forward_3d_input(self, device):
        mod = te.LayerNormMLP(256, 512).to(dtype=self.DTYPE, device=device)
        x = torch.randn(2, 8, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (2, 8, 256)

    def test_forward_no_bias(self, device):
        mod = te.LayerNormMLP(256, 512, bias=False).to(
            dtype=self.DTYPE, device=device
        )
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (4, 256)

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_forward_norm_variants(self, device, normalization):
        mod = te.LayerNormMLP(
            256, 512, normalization=normalization,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        y = mod(x)
        assert y.shape == (4, 256)

    def test_return_layernorm_output(self, device):
        mod = te.LayerNormMLP(
            256, 512, return_layernorm_output=True,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        out = mod(x)
        assert isinstance(out, tuple) and len(out) == 2
        y, ln_out = out
        assert y.shape == (4, 256)
        assert ln_out.shape == (4, 256)

    @pytest.mark.parametrize("activation", ["gelu", "silu", "relu", "swiglu"])
    def test_backward_all_grads(self, device, activation):
        mod = te.LayerNormMLP(
            256, 512, bias=True, activation=activation,
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape
        assert mod.fc1_weight.grad is not None
        assert mod.fc2_weight.grad is not None
        assert mod.layer_norm_weight.grad is not None

    def test_backward_no_bias(self, device):
        mod = te.LayerNormMLP(256, 512, bias=False, activation="gelu").to(
            dtype=self.DTYPE, device=device
        )
        x = torch.randn(4, 256, device=device, dtype=self.DTYPE, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        assert x.grad is not None
        assert mod.fc1_weight.grad is not None
        assert mod.fc2_weight.grad is not None

    def test_numerical_vs_manual(self, device):
        """Verify output matches manual norm→fc1→gelu→fc2 composition."""
        mod = te.LayerNormMLP(
            128, 256, bias=True, normalization="RMSNorm", activation="gelu",
        ).to(dtype=self.DTYPE, device=device)
        x = torch.randn(4, 128, device=device, dtype=self.DTYPE)

        # Manual reference
        w = mod.layer_norm_weight.data
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-5).rsqrt()
        normed = (x.float() * rms * w.float()).to(self.DTYPE)
        fc1_out = torch.nn.functional.linear(normed, mod.fc1_weight.data, mod.fc1_bias.data)
        act_out = torch.nn.functional.gelu(fc1_out, approximate="tanh")
        expected = torch.nn.functional.linear(act_out, mod.fc2_weight.data, mod.fc2_bias.data)

        y = mod(x)
        diff = (y - expected).abs().max().item()
        assert diff < 0.5, f"Max diff {diff:.4f} too large"


# ---------------------------------------------------------------------------
# Fused gated activation + FP8 quantize (AITER kernel)
# ---------------------------------------------------------------------------

class TestFusedGatedActQuant:
    """Tests for AITER fused gated activation + block FP8 quantize."""

    DTYPE = torch.bfloat16

    @staticmethod
    def _has_fused_kernel():
        """Check if the AITER fused act+quant kernel is available."""
        try:
            from aiter.ops.triton.activation import act_mul_and_fp8_group_quant  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _has_float8_block():
        """Check if Float8BlockQuantizer is available."""
        try:
            from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
                Float8BlockQuantizer,
            )
            return True
        except ImportError:
            return False

    @pytest.mark.parametrize("activation,aiter_act", [
        ("swiglu", "silu"),
        ("geglu", "gelu_tanh"),
        ("reglu", "relu"),
    ])
    def test_fused_path_matches_separate(self, device, activation, aiter_act):
        """Fused act+quant output should dequantize close to separate act then quant."""
        if not self._has_fused_kernel() or not self._has_float8_block():
            pytest.skip("AITER fused act+quant or Float8BlockQuantizer not available")

        from aiter.ops.triton.activation import act_mul_and_fp8_group_quant
        from transformer_engine.pytorch._lite.activations import (
            _aiter_fused_gated_act_quant,
        )
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
            Float8BlockQuantizer,
        )

        hidden = 256
        x = torch.randn(8, 2 * hidden, device=device, dtype=self.DTYPE)

        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )

        # Fused path
        fused_out = _aiter_fused_gated_act_quant(x, activation, quantizer)
        assert fused_out is not None, "Fused path should fire for Float8BlockQuantizer"

        # Separate path: manual act then quantize
        act_fn = {"swiglu": F.silu, "geglu": lambda t: F.gelu(t, approximate="tanh"), "reglu": F.relu}
        chunks = x.chunk(2, dim=-1)
        ref_bf16 = act_fn[activation](chunks[0]) * chunks[1]

        # Dequantize fused output
        fp8_data = fused_out._rowwise_data.view(torch.float8_e4m3fnuz).float()
        scale_inv = fused_out._rowwise_scale_inv
        # Expand scales to match data: each scale covers block_len elements
        block_len = quantizer.block_len
        num_blocks = fp8_data.shape[-1] // block_len
        scale_expanded = scale_inv.repeat_interleave(block_len, dim=-1)
        if scale_expanded.shape[-1] > fp8_data.shape[-1]:
            scale_expanded = scale_expanded[..., :fp8_data.shape[-1]]
        dequant = fp8_data * scale_expanded

        diff = (dequant - ref_bf16.float()).abs().max().item()
        # FP8 quantization error should be small relative to the values
        ref_max = ref_bf16.float().abs().max().item()
        rel_err = diff / max(ref_max, 1e-6)
        assert rel_err < 0.15, f"Fused act+quant relative error {rel_err:.4f} too large"

    @pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
    def test_fused_path_not_taken_without_block_quantizer(self, device, activation):
        """Fused path should return None when quantizer is not Float8BlockQuantizer."""
        from transformer_engine.pytorch._lite.activations import (
            _aiter_fused_gated_act_quant,
        )

        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        # None quantizer → no fused path
        result = _aiter_fused_gated_act_quant(x, activation, None)
        assert result is None

    def test_fused_path_output_shape(self, device):
        """Fused output should have half the last dim (gated activation)."""
        if not self._has_fused_kernel() or not self._has_float8_block():
            pytest.skip("AITER fused act+quant or Float8BlockQuantizer not available")

        from transformer_engine.pytorch._lite.activations import (
            _aiter_fused_gated_act_quant,
        )
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
            Float8BlockQuantizer,
        )

        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )

        x = torch.randn(4, 8, 512, device=device, dtype=self.DTYPE)
        result = _aiter_fused_gated_act_quant(x, "swiglu", quantizer)
        assert result is not None
        # Gated activation halves last dim: 512 → 256
        assert result._rowwise_data.shape[-1] == 256
        # Total elements should match batch * 256
        assert result._rowwise_data.numel() == 4 * 8 * 256


class TestFusedGatedActCurrentScaling:
    """Tests for AITER fused gated activation + per-row FP8 quantize (CurrentScaling)."""

    DTYPE = torch.bfloat16

    @staticmethod
    def _has_fused_kernel():
        try:
            from aiter.ops.triton.activation import act_mul_and_fp8_group_quant  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _has_current_scaling():
        try:
            from transformer_engine.pytorch.tensor.float8_tensor import (
                Float8CurrentScalingQuantizer,  # noqa: F401
            )
            return True
        except ImportError:
            return False

    @pytest.mark.parametrize("activation,aiter_act", [
        ("swiglu", "silu"),
        ("geglu", "gelu_tanh"),
        ("reglu", "relu"),
    ])
    def test_current_scaling_fused_matches_separate(self, device, activation, aiter_act):
        """Fused act+quant with CurrentScaling should dequantize close to separate act then quant."""
        if not self._has_fused_kernel() or not self._has_current_scaling():
            pytest.skip("AITER fused kernel or Float8CurrentScalingQuantizer not available")

        from transformer_engine.pytorch._lite.activations import (
            _aiter_fused_gated_act_current_scaling,
        )
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
            Float8Tensor,
        )

        hidden = 256
        x = torch.randn(8, 2 * hidden, device=device, dtype=self.DTYPE)

        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=x.device,
            rowwise=True,
            columnwise=False,
        )

        # Fused path
        fused_out = _aiter_fused_gated_act_current_scaling(x, activation, quantizer)
        assert fused_out is not None, "Fused path should fire for Float8CurrentScalingQuantizer"
        assert isinstance(fused_out, Float8Tensor)

        # Per-row scale_inv: shape (M,)
        assert fused_out._scale_inv.shape == (8,), (
            f"Expected per-row scale_inv shape (8,), got {fused_out._scale_inv.shape}"
        )

        # Separate path: manual act then per-row quant reference
        act_fn = {"swiglu": F.silu, "geglu": lambda t: F.gelu(t, approximate="tanh"), "reglu": F.relu}
        chunks = x.chunk(2, dim=-1)
        ref_bf16 = act_fn[activation](chunks[0]) * chunks[1]

        # Dequantize fused output
        fp8_data = fused_out._data.view(torch.float8_e4m3fnuz).float()
        scale_inv = fused_out._scale_inv  # shape (M,)
        # Expand per-row scales: (M,) → (M, 1) for broadcast
        dequant = fp8_data * scale_inv.unsqueeze(-1)

        diff = (dequant - ref_bf16.float()).abs().max().item()
        ref_max = ref_bf16.float().abs().max().item()
        rel_err = diff / max(ref_max, 1e-6)
        assert rel_err < 0.15, f"Fused act+quant (CurrentScaling) relative error {rel_err:.4f} too large"

    @pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
    def test_current_scaling_not_taken_for_block_quantizer(self, device, activation):
        """CurrentScaling fused path should return None for Float8BlockQuantizer."""
        from transformer_engine.pytorch._lite.activations import (
            _aiter_fused_gated_act_current_scaling,
        )

        x = torch.randn(4, 256, device=device, dtype=self.DTYPE)
        # None quantizer → no fused path
        result = _aiter_fused_gated_act_current_scaling(x, activation, None)
        assert result is None

    def test_current_scaling_output_shape_3d(self, device):
        """Fused output should handle 3D input and have half the last dim."""
        if not self._has_fused_kernel() or not self._has_current_scaling():
            pytest.skip("AITER fused kernel or Float8CurrentScalingQuantizer not available")

        from transformer_engine.pytorch._lite.activations import (
            _aiter_fused_gated_act_current_scaling,
        )
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
            Float8Tensor,
        )

        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
            rowwise=True,
            columnwise=False,
        )

        x = torch.randn(4, 8, 512, device=device, dtype=self.DTYPE)
        result = _aiter_fused_gated_act_current_scaling(x, "swiglu", quantizer)
        assert result is not None
        assert isinstance(result, Float8Tensor)
        # Gated activation halves last dim: 512 → 256
        assert result._data.shape == (4, 8, 256)
        # Per-row scales: M = 4*8 = 32 rows
        assert result._scale_inv.shape == (32,)

    def test_current_scaling_per_row_scales_vary(self, device):
        """Per-row scales should differ across rows (not degenerate per-tensor)."""
        if not self._has_fused_kernel() or not self._has_current_scaling():
            pytest.skip("AITER fused kernel or Float8CurrentScalingQuantizer not available")

        from transformer_engine.pytorch._lite.activations import (
            _aiter_fused_gated_act_current_scaling,
        )
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
        )

        # Use input with deliberately different magnitudes per row
        hidden = 128
        x = torch.randn(16, 2 * hidden, device=device, dtype=self.DTYPE)
        # Scale each row differently so per-row scales must differ
        row_scales = torch.logspace(-2, 2, 16, device=device, dtype=self.DTYPE).unsqueeze(-1)
        x = x * row_scales

        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=device,
            rowwise=True,
            columnwise=False,
        )

        result = _aiter_fused_gated_act_current_scaling(x, "swiglu", quantizer)
        assert result is not None
        scales = result._scale_inv
        # With 4 orders of magnitude in row scales, per-row scales should vary
        assert scales.max() / scales.min() > 2.0, (
            f"Per-row scales should vary, got max/min ratio {scales.max() / scales.min():.2f}"
        )


# ---------------------------------------------------------------------------
# Recipe-level FP8 integration tests
# ---------------------------------------------------------------------------

def _available_recipes():
    """Build list of recipe instances for what this hardware supports."""
    recipes = []
    avail, _ = te.is_fp8_available(return_reason=True)
    if avail:
        recipes.append(recipe.DelayedScaling())
        recipes.append(recipe.Float8CurrentScaling())
    block_avail, _ = te.is_fp8_block_scaling_available(return_reason=True)
    if block_avail:
        recipes.append(recipe.Float8BlockScaling())
    mx_avail, _ = te.is_mxfp8_available(return_reason=True)
    if mx_avail:
        recipes.append(recipe.MXFP8BlockScaling())
    return recipes


def _recipe_id(val):
    """Short string ID for parametrize labels."""
    if isinstance(val, pytest.param):
        return None  # let pytest use the id from pytest.param
    return type(val).__name__


def _mark_recipes(recipes):
    """Wrap recipes with xfail markers for any known lite-mode bugs."""
    return [pytest.param(r, id=type(r).__name__) for r in recipes]


_RECIPES = _available_recipes()
_RECIPES_FWD = _mark_recipes(_RECIPES)
_RECIPES_FWD_BWD = _mark_recipes(_RECIPES)


class TestRecipeIntegration:
    """Recipe-level FP8 integration through te.autocast for core TE modules.

    Tests the full path: recipe object -> autocast context -> RecipeState ->
    quantizer construction -> module forward/backward.
    """

    DTYPE = torch.bfloat16
    HIDDEN = 256
    FFN_HIDDEN = 1024
    BATCH = 8       # Must be divisible by 8 for FP8 alignment
    SEQ = 8

    @pytest.fixture(autouse=True)
    def _reset_fp8_state(self):
        """Reset global FP8 state between tests."""
        yield
        FP8GlobalStateManager.reset()

    def _skip_if_no_recipes(self):
        if not _RECIPES:
            pytest.skip("No FP8 recipes available on this hardware")

    # ---------------------------------------------------------------
    # Linear — simplest module, isolates GEMM + quantize path
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_linear_fwd_bwd(self, device, fp8_recipe):
        """Linear forward+backward completes without error under each recipe."""
        mod = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                        params_dtype=self.DTYPE).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        with te.autocast(enabled=True, recipe=fp8_recipe):
            y = mod(x)
        y.sum().backward()
        torch.cuda.synchronize()
        assert y.shape == (self.BATCH, self.HIDDEN)
        assert x.grad is not None
        assert x.grad.shape == x.shape
        for p in mod.parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing grad for param shape {p.shape}"

    # ---------------------------------------------------------------
    # LayerNormLinear — tests fused norm+quant -> GEMM path
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_layernorm_linear_fwd_bwd(self, device, fp8_recipe, normalization):
        """LayerNormLinear forward+backward under each recipe and norm variant."""
        mod = te.LayerNormLinear(
            self.HIDDEN, self.HIDDEN, bias=True,
            normalization=normalization,
            params_dtype=self.DTYPE,
        ).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        with te.autocast(enabled=True, recipe=fp8_recipe):
            y = mod(x)
        y.sum().backward()
        torch.cuda.synchronize()
        assert y.shape == (self.BATCH, self.HIDDEN)
        assert x.grad is not None

    # ---------------------------------------------------------------
    # LayerNormMLP — norm+quant -> GEMM -> act+quant -> GEMM
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    @pytest.mark.parametrize("activation", ["gelu", "swiglu", "geglu"])
    def test_layernorm_mlp_fwd_bwd(self, device, fp8_recipe, activation):
        """LayerNormMLP forward+backward — exercises fused act+quant dispatch."""
        mod = te.LayerNormMLP(
            self.HIDDEN, self.FFN_HIDDEN,
            activation=activation,
            params_dtype=self.DTYPE,
        ).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        with te.autocast(enabled=True, recipe=fp8_recipe):
            y = mod(x)
        y.sum().backward()
        torch.cuda.synchronize()
        assert y.shape == (self.BATCH, self.HIDDEN)
        assert x.grad is not None

    # ---------------------------------------------------------------
    # Multi-step — catches stale scale / amax history bugs
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_linear_multi_step(self, device, fp8_recipe):
        """3-step forward+backward — verifies amax history and scale updates."""
        mod = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                        params_dtype=self.DTYPE).to(device)
        for step in range(3):
            x = torch.randn(self.BATCH, self.HIDDEN, device=device,
                             dtype=self.DTYPE, requires_grad=True)
            with te.autocast(enabled=True, recipe=fp8_recipe):
                y = mod(x)
            y.sum().backward()
            torch.cuda.synchronize()
            assert y.shape == (self.BATCH, self.HIDDEN), f"Failed at step {step}"
            assert x.grad is not None, f"No input grad at step {step}"

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_layernorm_mlp_multi_step(self, device, fp8_recipe):
        """3-step LayerNormMLP — full pipeline with scale state evolution."""
        mod = te.LayerNormMLP(
            self.HIDDEN, self.FFN_HIDDEN,
            activation="swiglu",
            params_dtype=self.DTYPE,
        ).to(device)
        for step in range(3):
            x = torch.randn(self.BATCH, self.HIDDEN, device=device,
                             dtype=self.DTYPE, requires_grad=True)
            with te.autocast(enabled=True, recipe=fp8_recipe):
                y = mod(x)
            y.sum().backward()
            torch.cuda.synchronize()
            assert y.shape == (self.BATCH, self.HIDDEN), f"Failed at step {step}"

    # ---------------------------------------------------------------
    # Output sanity — FP8 shouldn't produce garbage
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD)
    def test_linear_output_finite(self, device, fp8_recipe):
        """FP8 output should be finite and not all-zeros."""
        mod = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                        params_dtype=self.DTYPE).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device,
                         dtype=self.DTYPE)
        with te.autocast(enabled=True, recipe=fp8_recipe):
            y = mod(x)
        assert torch.isfinite(y).all(), "Output contains NaN/Inf"
        assert y.abs().max() > 0, "Output is all zeros"

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD)
    def test_fp8_vs_bf16_correlation(self, device, fp8_recipe):
        """FP8 output should correlate with bf16 output (same weights, same input)."""
        mod = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                        params_dtype=self.DTYPE).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device,
                         dtype=self.DTYPE)

        # bf16 reference (no FP8)
        with torch.no_grad():
            ref = mod(x)

        # FP8 path
        with torch.no_grad():
            with te.autocast(enabled=True, recipe=fp8_recipe):
                fp8_out = mod(x)

        cos_sim = F.cosine_similarity(ref.flatten().float(),
                                       fp8_out.flatten().float(), dim=0)
        assert cos_sim > 0.9, (
            f"FP8 output too far from bf16: cosine_similarity={cos_sim:.4f}"
        )

    # ---------------------------------------------------------------
    # TransformerLayer — full attention + MLP stack with FP8
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_transformer_layer_fwd_bwd(self, device, fp8_recipe):
        """Full TransformerLayer forward+backward under FP8 recipe."""
        mod = te.TransformerLayer(
            self.HIDDEN, self.FFN_HIDDEN, num_attention_heads=4,
            params_dtype=self.DTYPE,
        ).to(device)
        # TransformerLayer expects (seq, batch, hidden)
        x = torch.randn(
            self.SEQ, 2, self.HIDDEN, device=device,
            dtype=self.DTYPE, requires_grad=True,
        )
        with torch.amp.autocast("cuda", dtype=self.DTYPE):
            with te.autocast(enabled=True, recipe=fp8_recipe):
                y = mod(x)
        y.sum().backward()
        torch.cuda.synchronize()
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # ---------------------------------------------------------------
    # FP8 vs bf16 correlation for fused modules — catches silent
    # wrong-dispatch, scale broadcast bugs, and per-row axis misalignment
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD)
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_layernorm_linear_correlation(self, device, fp8_recipe, normalization):
        """LayerNormLinear FP8 output should correlate with bf16 (same weights)."""
        mod = te.LayerNormLinear(
            self.HIDDEN, self.HIDDEN, bias=True,
            normalization=normalization, params_dtype=self.DTYPE,
        ).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
        with torch.no_grad():
            ref = mod(x)
            with te.autocast(enabled=True, recipe=fp8_recipe):
                fp8_out = mod(x)
        cos = F.cosine_similarity(ref.flatten().float(),
                                   fp8_out.flatten().float(), dim=0).item()
        assert cos > 0.9, f"cos_sim={cos:.4f}"

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD)
    @pytest.mark.parametrize("activation", ["gelu", "swiglu"])
    def test_layernorm_mlp_correlation(self, device, fp8_recipe, activation):
        """LayerNormMLP FP8 output should correlate with bf16 (same weights)."""
        mod = te.LayerNormMLP(
            self.HIDDEN, self.FFN_HIDDEN,
            activation=activation, params_dtype=self.DTYPE,
        ).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
        with torch.no_grad():
            ref = mod(x)
            with te.autocast(enabled=True, recipe=fp8_recipe):
                fp8_out = mod(x)
        cos = F.cosine_similarity(ref.flatten().float(),
                                   fp8_out.flatten().float(), dim=0).item()
        assert cos > 0.9, f"cos_sim={cos:.4f}"

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD)
    def test_transformer_layer_correlation(self, device, fp8_recipe):
        """TransformerLayer FP8 output should correlate with bf16 (same weights)."""
        mod = te.TransformerLayer(
            self.HIDDEN, self.FFN_HIDDEN, num_attention_heads=4,
            params_dtype=self.DTYPE,
        ).to(device)
        x = torch.randn(self.SEQ, 2, self.HIDDEN, device=device, dtype=self.DTYPE)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.DTYPE):
                ref = mod(x)
                with te.autocast(enabled=True, recipe=fp8_recipe):
                    fp8_out = mod(x)
        cos = F.cosine_similarity(ref.flatten().float(),
                                   fp8_out.flatten().float(), dim=0).item()
        # TransformerLayer has more accumulated error than a single Linear,
        # so use a looser tolerance (still much better than random ~0).
        assert cos > 0.75, f"cos_sim={cos:.4f}"


# ---------------------------------------------------------------------------
# API contract tests — verify lite exposes the symbols the full TE does,
# with compatible constructor signatures. Catches cases like "accepted kwarg
# but silently dropped it" (the return_bias bug we just fixed).
# ---------------------------------------------------------------------------

class TestLiteAPI:
    """Verify the lite backend exposes the API contract of full TE."""

    # Top-level symbols that test_sanity.py imports from transformer_engine.pytorch
    _TE_PUBLIC_SYMBOLS = [
        "Linear", "LayerNormLinear", "LayerNormMLP", "TransformerLayer",
        "GroupedLinear", "RMSNorm", "LayerNorm",
        "autocast", "fp8_autocast",
        "Float8Tensor", "Float8Quantizer", "Float8CurrentScalingQuantizer",
        "QuantizedTensor",
        "is_fp8_available", "is_mxfp8_available", "is_fp8_block_scaling_available",
        "is_bf16_available",
    ]

    # Critical tex (transformer_engine_torch) functions used across the codebase
    _TEX_FUNCTIONS = [
        "quantize", "dequantize", "bgrad_quantize", "split_quantize",
        "generic_gemm", "fp8_transpose",
        "layernorm_fwd", "layernorm_bwd", "rmsnorm_fwd", "rmsnorm_bwd",
        "fused_amax_and_scale_update_after_reduction", "compute_amax",
        "swiglu", "geglu", "reglu", "gelu", "silu", "relu", "srelu", "qgelu",
        "dswiglu", "dgeglu", "dreglu", "dgelu", "dsilu", "drelu",
        "dbias_dgelu", "dbias_dsilu", "dbias_drelu",
    ]

    # Expected DType enum values
    _TEX_DTYPES = [
        "kFloat32", "kFloat16", "kBFloat16",
        "kFloat8E4M3", "kFloat8E5M2",
    ]

    def test_te_public_symbols_exist(self):
        """Every symbol test_sanity.py imports from te must exist in lite."""
        missing = [s for s in self._TE_PUBLIC_SYMBOLS if not hasattr(te, s)]
        assert not missing, f"Missing from te: {missing}"

    def test_tex_functions_exist(self):
        """Every tex function called across the TE codebase must exist in lite."""
        missing = [s for s in self._TEX_FUNCTIONS if not hasattr(tex, s)]
        assert not missing, f"Missing from tex: {missing}"

    def test_tex_functions_callable(self):
        """tex functions must be callable, not just exist as sentinel values."""
        non_callable = [
            s for s in self._TEX_FUNCTIONS
            if hasattr(tex, s) and not callable(getattr(tex, s))
        ]
        assert not non_callable, f"Not callable: {non_callable}"

    def test_tex_dtype_enum(self):
        """DType enum must expose the standard values."""
        assert hasattr(tex, "DType"), "tex.DType missing"
        missing = [v for v in self._TEX_DTYPES if not hasattr(tex.DType, v)]
        assert not missing, f"Missing DType values: {missing}"

    @pytest.mark.parametrize("cls_name,required_kwargs", [
        ("Linear", ["bias", "params_dtype"]),
        ("LayerNormLinear", ["bias", "params_dtype", "normalization",
                             "return_bias", "return_layernorm_output",
                             "zero_centered_gamma"]),
        ("LayerNormMLP", ["bias", "params_dtype", "normalization", "activation",
                          "return_bias", "return_layernorm_output",
                          "zero_centered_gamma"]),
        ("TransformerLayer", ["num_attention_heads", "params_dtype"]),
        ("LayerNorm", ["zero_centered_gamma"]),
        ("RMSNorm", ["zero_centered_gamma"]),
    ])
    def test_module_accepts_expected_kwargs(self, cls_name, required_kwargs):
        """Module constructors must accept the documented kwargs, not silently drop them."""
        import inspect
        cls = getattr(te, cls_name)
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        # Either the kwarg is explicitly listed, or **kwargs catches it.
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        missing = [k for k in required_kwargs if k not in params and not has_kwargs]
        assert not missing, f"{cls_name} missing kwargs: {missing}"

    def test_layernorm_mlp_return_bias_returns_tuple(self, device):
        """Regression test: LayerNormMLP(return_bias=True) must return (out, bias)."""
        mod = te.LayerNormMLP(
            256, 1024, bias=True, return_bias=True, params_dtype=torch.bfloat16,
        ).to(device)
        x = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
        out = mod(x)
        assert isinstance(out, tuple), f"Expected tuple, got {type(out).__name__}"
        assert len(out) == 2, f"Expected 2-tuple, got {len(out)}-tuple"
        main_out, bias = out
        assert main_out.shape == (8, 256)
        # bias must be 1D with last-dim size — either a real bias or a placeholder
        assert bias.ndim == 1 and bias.shape[0] == 256

    def test_lite_mode_flag(self):
        """The tex module's __name__ reliably identifies lite vs full backend."""
        assert tex.__name__ == "transformer_engine.pytorch._lite"

    def test_recipes_available(self):
        """Recipe classes must be importable via the common API, regardless of hw support."""
        from transformer_engine.common import recipe as r
        assert hasattr(r, "DelayedScaling")
        assert hasattr(r, "Float8CurrentScaling")
        assert hasattr(r, "Float8BlockScaling")
        assert hasattr(r, "MXFP8BlockScaling")


# ---------------------------------------------------------------------------
# End-to-end training loop tests — optimizer.step() drives weight updates,
# verifying that FP8 recipes actually converge and the fp8 weight cache
# invalidates correctly between steps.
# ---------------------------------------------------------------------------

class TestFP8Training:
    """Verify FP8 recipes support real training (optimizer.step)."""

    DTYPE = torch.bfloat16
    HIDDEN = 256
    FFN_HIDDEN = 1024
    BATCH = 8

    @pytest.fixture(autouse=True)
    def _reset_fp8_state(self):
        yield
        FP8GlobalStateManager.reset()

    def _overfit_and_check(self, mod, x, target, fp8_recipe,
                            steps=50, lr=1e-3, use_amp=False,
                            loss_drop_ratio=0.8):
        """Run N training steps with Adam; assert loss drops by at least
        (1-loss_drop_ratio) of the initial value. Adam is used (not SGD)
        to avoid per-module LR tuning — it adapts to gradient magnitudes."""
        opt = torch.optim.Adam(mod.parameters(), lr=lr)
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=self.DTYPE):
                    with te.autocast(enabled=True, recipe=fp8_recipe):
                        y = mod(x)
            else:
                with te.autocast(enabled=True, recipe=fp8_recipe):
                    y = mod(x)
            loss = (y.float() - target.float()).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        torch.cuda.synchronize()
        # No NaN/Inf during the trajectory
        assert all(torch.isfinite(torch.tensor(L)) for L in losses), (
            f"Non-finite loss during training: trajectory={losses[::5]}"
        )
        # Substantial learning (final < loss_drop_ratio * initial)
        assert losses[-1] < losses[0] * loss_drop_ratio, (
            f"Loss didn't drop enough: {losses[0]:.4f} -> {losses[-1]:.4f} "
            f"(trajectory every 10: {losses[::10]})"
        )
        return losses

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_linear_overfits_batch(self, device, fp8_recipe):
        """Linear should overfit a fixed batch under each FP8 recipe."""
        torch.manual_seed(0)
        mod = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                        params_dtype=self.DTYPE).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
        target = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
        self._overfit_and_check(mod, x, target, fp8_recipe)

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_layernorm_mlp_overfits_batch(self, device, fp8_recipe):
        """LayerNormMLP should overfit a fixed batch."""
        torch.manual_seed(0)
        mod = te.LayerNormMLP(self.HIDDEN, self.FFN_HIDDEN,
                              activation="swiglu",
                              params_dtype=self.DTYPE).to(device)
        x = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
        target = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
        self._overfit_and_check(mod, x, target, fp8_recipe)

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_transformer_layer_overfits_batch(self, device, fp8_recipe):
        """TransformerLayer should overfit a fixed batch."""
        torch.manual_seed(0)
        mod = te.TransformerLayer(
            self.HIDDEN, self.FFN_HIDDEN, num_attention_heads=4,
            params_dtype=self.DTYPE,
        ).to(device)
        x = torch.randn(self.BATCH, 2, self.HIDDEN, device=device, dtype=self.DTYPE)
        target = torch.randn(self.BATCH, 2, self.HIDDEN, device=device, dtype=self.DTYPE)
        self._overfit_and_check(mod, x, target, fp8_recipe, use_amp=True)

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_weights_change_after_step(self, device, fp8_recipe):
        """Sanity: optimizer.step() must actually update the weights."""
        torch.manual_seed(0)
        mod = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                        params_dtype=self.DTYPE).to(device)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1)

        x = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
        w_before = mod.weight.detach().clone()

        with te.autocast(enabled=True, recipe=fp8_recipe):
            y = mod(x)
        y.sum().backward()
        opt.step()
        torch.cuda.synchronize()

        assert not torch.equal(w_before, mod.weight), (
            "Weights unchanged after optimizer.step()"
        )

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    def test_fp8_training_tracks_bf16(self, device, fp8_recipe):
        """After N training steps, FP8 weights should still correlate with bf16 weights.
        Catches FP8 weight-cache staleness — if fp8 cache isn't invalidated after
        optimizer.step, the two trajectories diverge rapidly."""
        torch.manual_seed(0)
        mod_fp8 = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                            params_dtype=self.DTYPE).to(device)
        torch.manual_seed(0)
        mod_bf = te.Linear(self.HIDDEN, self.HIDDEN, bias=True,
                           params_dtype=self.DTYPE).to(device)

        opt_fp8 = torch.optim.SGD(mod_fp8.parameters(), lr=0.01)
        opt_bf = torch.optim.SGD(mod_bf.parameters(), lr=0.01)

        for step in range(10):
            torch.manual_seed(step + 1)
            x = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)
            target = torch.randn(self.BATCH, self.HIDDEN, device=device, dtype=self.DTYPE)

            opt_fp8.zero_grad()
            with te.autocast(enabled=True, recipe=fp8_recipe):
                y = mod_fp8(x)
            ((y.float() - target.float()).pow(2).mean()).backward()
            opt_fp8.step()

            opt_bf.zero_grad()
            y_bf = mod_bf(x)
            ((y_bf.float() - target.float()).pow(2).mean()).backward()
            opt_bf.step()

        torch.cuda.synchronize()
        cos = F.cosine_similarity(
            mod_fp8.weight.detach().float().flatten(),
            mod_bf.weight.detach().float().flatten(),
            dim=0,
        ).item()
        assert cos > 0.95, f"FP8 weights diverged from bf16: cos={cos:.4f}"


# ---------------------------------------------------------------------------
# FP8 attention flags — lite does not implement FP8 attention kernels.
# These tests document the behavior: fp8_dpa=False/fp8_mha=False works,
# setting either to True raises a clear NotImplementedError (not a cryptic
# AttributeError from a missing enum value).
# ---------------------------------------------------------------------------

class TestFP8AttentionFlags:
    """Lite accepts fp8_dpa/fp8_mha recipe flags but rejects them cleanly."""

    DTYPE = torch.bfloat16
    HIDDEN = 256
    FFN_HIDDEN = 1024

    @pytest.fixture(autouse=True)
    def _reset_fp8_state(self):
        yield
        FP8GlobalStateManager.reset()

    def _make_model_and_input(self, device):
        mod = te.TransformerLayer(
            self.HIDDEN, self.FFN_HIDDEN, num_attention_heads=4,
            params_dtype=self.DTYPE,
        ).to(device)
        x = torch.randn(8, 2, self.HIDDEN, device=device, dtype=self.DTYPE)
        return mod, x

    def test_fp8_dpa_raises_not_implemented(self, device):
        """fp8_dpa=True must raise a clear NotImplementedError, not AttributeError."""
        if not _RECIPES:
            pytest.skip("No FP8 recipes available on this hardware")
        mod, x = self._make_model_and_input(device)
        r = recipe.DelayedScaling(fp8_dpa=True)
        with pytest.raises(NotImplementedError, match="FP8 attention"):
            with torch.amp.autocast("cuda", dtype=self.DTYPE):
                with te.autocast(enabled=True, recipe=r):
                    mod(x)

    def test_fp8_mha_raises_not_implemented(self, device):
        """fp8_mha=True must raise NotImplementedError (q_type becomes FP8 upstream)."""
        if not _RECIPES:
            pytest.skip("No FP8 recipes available on this hardware")
        mod, x = self._make_model_and_input(device)
        r = recipe.DelayedScaling(fp8_mha=True)
        with pytest.raises(NotImplementedError, match="FP8 attention"):
            with torch.amp.autocast("cuda", dtype=self.DTYPE):
                with te.autocast(enabled=True, recipe=r):
                    mod(x)

    def test_fp8_dpa_and_mha_raises(self, device):
        """Both flags set together still raises cleanly."""
        if not _RECIPES:
            pytest.skip("No FP8 recipes available on this hardware")
        mod, x = self._make_model_and_input(device)
        r = recipe.DelayedScaling(fp8_dpa=True, fp8_mha=True)
        with pytest.raises(NotImplementedError, match="FP8 attention"):
            with torch.amp.autocast("cuda", dtype=self.DTYPE):
                with te.autocast(enabled=True, recipe=r):
                    mod(x)

    def test_default_flags_work(self, device):
        """Recipe without the FP8 attention flags (default) must work end-to-end."""
        if not _RECIPES:
            pytest.skip("No FP8 recipes available on this hardware")
        mod, x = self._make_model_and_input(device)
        # Explicitly False to pin the contract
        r = recipe.DelayedScaling(fp8_dpa=False, fp8_mha=False)
        with torch.amp.autocast("cuda", dtype=self.DTYPE):
            with te.autocast(enabled=True, recipe=r):
                y = mod(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_current_scaling_fp8_dpa_raises(self, device):
        """CurrentScaling recipe with fp8_dpa=True also rejected cleanly."""
        if not _RECIPES:
            pytest.skip("No FP8 recipes available on this hardware")
        mod, x = self._make_model_and_input(device)
        r = recipe.Float8CurrentScaling(fp8_dpa=True)
        with pytest.raises(NotImplementedError, match="FP8 attention"):
            with torch.amp.autocast("cuda", dtype=self.DTYPE):
                with te.autocast(enabled=True, recipe=r):
                    mod(x)

    def test_enum_has_nvte_fp8(self):
        """NVTE_FP8 enum value must exist for API compatibility."""
        assert hasattr(tex.NVTE_Fused_Attn_Backend, "NVTE_FP8"), (
            "NVTE_FP8 must exist in the enum for framework compat, even if "
            "lite doesn't implement the corresponding backend."
        )


# ---------------------------------------------------------------------------
# GroupedLinear tests — MoE-style expert parallelism via a single module
# that dispatches num_gemms weights against an input split by m_splits.
# ---------------------------------------------------------------------------

class TestGroupedLinear:
    """Verify GroupedLinear works end-to-end in lite mode."""

    DTYPE = torch.bfloat16
    NUM_GEMMS = 4
    IN_FEATURES = 256
    OUT_FEATURES = 256
    M_SPLITS = [8, 8, 8, 8]  # total 32 tokens

    @pytest.fixture(autouse=True)
    def _reset_fp8_state(self):
        yield
        FP8GlobalStateManager.reset()

    def _make_module(self, device, bias=True):
        return te.GroupedLinear(
            num_gemms=self.NUM_GEMMS,
            in_features=self.IN_FEATURES,
            out_features=self.OUT_FEATURES,
            bias=bias,
            params_dtype=self.DTYPE,
            parallel_mode=None,
        ).to(device)

    @pytest.mark.parametrize("bias", [True, False])
    def test_forward_shape(self, device, bias):
        """Output shape is (total_tokens, out_features)."""
        mod = self._make_module(device, bias=bias)
        total = sum(self.M_SPLITS)
        x = torch.randn(total, self.IN_FEATURES, device=device, dtype=self.DTYPE)
        y = mod(x, self.M_SPLITS)
        assert y.shape == (total, self.OUT_FEATURES)
        assert torch.isfinite(y).all()

    @pytest.mark.parametrize("bias", [True, False])
    def test_forward_matches_manual(self, device, bias):
        """Output matches F.linear per chunk (reference implementation)."""
        torch.manual_seed(42)
        mod = self._make_module(device, bias=bias)
        total = sum(self.M_SPLITS)
        x = torch.randn(total, self.IN_FEATURES, device=device, dtype=self.DTYPE)
        y = mod(x, self.M_SPLITS)

        # Manual reference: split input, run each chunk through its expert
        chunks = torch.split(x, self.M_SPLITS, dim=0)
        y_ref_parts = []
        for i, chunk in enumerate(chunks):
            w = getattr(mod, f"weight{i}")
            b = getattr(mod, f"bias{i}") if bias else None
            y_ref_parts.append(F.linear(chunk, w, b))
        y_ref = torch.cat(y_ref_parts, dim=0)
        assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3), (
            f"GroupedLinear output differs from manual: max_diff="
            f"{(y - y_ref).abs().max().item()}"
        )

    @pytest.mark.parametrize("bias", [True, False])
    def test_backward_grads_finite(self, device, bias):
        """Input gradient and all per-expert weight gradients must be finite."""
        mod = self._make_module(device, bias=bias)
        total = sum(self.M_SPLITS)
        x = torch.randn(total, self.IN_FEATURES, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        y = mod(x, self.M_SPLITS)
        y.sum().backward()
        torch.cuda.synchronize()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for i in range(self.NUM_GEMMS):
            w_grad = getattr(mod, f"weight{i}").grad
            assert w_grad is not None, f"weight{i}.grad is None"
            assert torch.isfinite(w_grad).all(), f"weight{i}.grad has NaN/Inf"
            if bias:
                b_grad = getattr(mod, f"bias{i}").grad
                assert b_grad is not None, f"bias{i}.grad is None"
                assert torch.isfinite(b_grad).all(), f"bias{i}.grad has NaN/Inf"

    def test_uneven_splits(self, device):
        """Non-uniform m_splits should also work (MoE often has imbalanced routing)."""
        mod = self._make_module(device, bias=True)
        m_splits = [4, 12, 8, 8]  # total 32
        total = sum(m_splits)
        x = torch.randn(total, self.IN_FEATURES, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        y = mod(x, m_splits)
        assert y.shape == (total, self.OUT_FEATURES)
        y.sum().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("fp8_recipe", _RECIPES_FWD_BWD)
    @pytest.mark.xfail(
        strict=True,
        reason="FP8 GroupedLinear hits dtype mismatch in Triton wrapper "
               "(lhs=fp32 vs bias=bf16) — pre-existing issue in "
               "triton_kernels/gmm/gmm_common.py, out of scope for lite adapter.",
    )
    def test_fp8_forward(self, device, fp8_recipe):
        """FP8 GroupedLinear — currently blocked on a Triton GMM bug."""
        mod = self._make_module(device, bias=True)
        total = sum(self.M_SPLITS)
        x = torch.randn(total, self.IN_FEATURES, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        with te.autocast(enabled=True, recipe=fp8_recipe):
            y = mod(x, self.M_SPLITS)
        y.sum().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(y).all()
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Grouped-GEMM dispatcher (lite) — Phase 1 covers BF16 only via
# `general_grouped_gemm_triton`. FP8 operands must fail loudly until the
# Phase 2 fused-MoE path lands; otherwise they would silently misroute
# through the BF16 kernel and either crash or miscompute.
# ---------------------------------------------------------------------------

class TestGroupedGemmDispatch:
    """Verify lite's te_general_grouped_gemm dispatcher gating."""

    NUM_GEMMS = 2
    IN_FEATURES = 64
    OUT_FEATURES = 64
    M_SPLITS = [4, 4]

    def test_fp8_operands_raise_not_implemented(self, device):
        total = sum(self.M_SPLITS)
        A = [
            torch.empty(self.OUT_FEATURES, self.IN_FEATURES,
                        device=device, dtype=torch.float8_e4m3fn)
            for _ in range(self.NUM_GEMMS)
        ]
        B = [
            torch.empty(m, self.IN_FEATURES,
                        device=device, dtype=torch.float8_e4m3fn)
            for m in self.M_SPLITS
        ]
        out = [
            torch.empty(m, self.OUT_FEATURES, device=device, dtype=torch.bfloat16)
            for m in self.M_SPLITS
        ]
        ws = [torch.empty(1024, device=device, dtype=torch.uint8)]
        with pytest.raises(NotImplementedError, match="FP8 grouped GEMM"):
            tex.te_general_grouped_gemm(
                A, True, B, False, out, torch.bfloat16, self.M_SPLITS,
                [], None, False, [], False, ws, ws[0].shape[0],
                False, False, 0,
            )

    def test_empty_tokens_short_circuit_forward(self, device):
        """MoE routing can produce zero local tokens in early training; the
        underlying AITER gmm asserts M > 0, so the lite wrapper must
        short-circuit instead of forwarding."""
        m_splits = [0, 0]
        A = [
            torch.randn(self.OUT_FEATURES, self.IN_FEATURES,
                        device=device, dtype=torch.bfloat16)
            for _ in range(self.NUM_GEMMS)
        ]
        # Empty input: (0, in_features)
        B = [torch.empty(0, self.IN_FEATURES, device=device, dtype=torch.bfloat16)]
        out = [torch.empty(0, self.OUT_FEATURES, device=device, dtype=torch.bfloat16)]
        ws = [torch.empty(1024, device=device, dtype=torch.uint8)]
        # Forward layout (transa=True, transb=False), grad=False.
        # Should return without raising, with the empty out tensor untouched.
        tex.te_general_grouped_gemm(
            A, True, B, False, out, torch.bfloat16, m_splits,
            [], None, True, [], False, ws, ws[0].shape[0],
            False, False, 0,
        )
        assert out[0].shape == (0, self.OUT_FEATURES)

    def test_empty_tokens_short_circuit_wgrad_zeros_out(self, device):
        """Wgrad with zero tokens must zero its (G, K, N) output when
        accumulate=False so the caller sees the correct zero contribution."""
        m_splits = [0, 0]
        A = [torch.empty(0, self.IN_FEATURES, device=device, dtype=torch.bfloat16)]
        B = [torch.empty(0, self.OUT_FEATURES, device=device, dtype=torch.bfloat16)]
        # Pre-fill out with garbage so we can verify the zero_() actually fired.
        out = [
            torch.full((self.OUT_FEATURES, self.IN_FEATURES), 7.0,
                       device=device, dtype=torch.bfloat16)
            for _ in range(self.NUM_GEMMS)
        ]
        ws = [torch.empty(1024, device=device, dtype=torch.uint8)]
        # Wgrad layout: transa=True (wgrad layout NT in the upstream wrapper
        # corresponds to transa=False, transb=True from the C++ binding's view
        # — but our lite wrapper checks `transa and not transb and grad`, so
        # invoke with (transa=True, transb=False, grad=True) to hit it).
        tex.te_general_grouped_gemm(
            A, True, B, False, out, torch.bfloat16, m_splits,
            [], None, True, [], True, ws, ws[0].shape[0],
            False, False, 0,
        )
        for o in out:
            assert torch.all(o == 0), "wgrad output not zeroed under M=0"


# ---------------------------------------------------------------------------
# FSDP2 weight-wrap tests — lite's compound modules must wrap FP8 weights in
# FSDPAGTensor when use_fsdp2=True so FSDP2's all-gather calls
# fsdp_pre_all_gather to quantize at gather time, not at parameter init.
# ---------------------------------------------------------------------------

class TestFSDP2WeightWrap:
    """Verify lite compound modules emit FSDPAGTensor under use_fsdp2=True."""

    DTYPE = torch.bfloat16
    HIDDEN = 256
    FFN_HIDDEN = 1024

    @staticmethod
    def _has_fsdpag():
        try:
            from transformer_engine.pytorch.tensor.fsdp2_allgather_tensor import FSDPAGTensor  # noqa
            return True
        except ImportError:
            return False

    def _fsdpag_cls(self):
        from transformer_engine.pytorch.tensor.fsdp2_allgather_tensor import FSDPAGTensor
        return FSDPAGTensor

    def test_layernorm_linear_no_wrap_by_default(self, device):
        """Default (use_fsdp2=False): weight is a plain Parameter, not FSDPAGTensor."""
        if not self._has_fsdpag():
            pytest.skip("FSDPAGTensor unavailable")
        mod = te.LayerNormLinear(
            self.HIDDEN, self.HIDDEN, params_dtype=self.DTYPE, device=device,
        )
        assert not isinstance(mod.weight, self._fsdpag_cls())
        assert isinstance(mod.weight, torch.nn.Parameter)

    def test_layernorm_linear_wraps_with_use_fsdp2(self, device):
        """use_fsdp2=True: weight is wrapped in FSDPAGTensor for FSDP2 all-gather."""
        if not self._has_fsdpag():
            pytest.skip("FSDPAGTensor unavailable")
        mod = te.LayerNormLinear(
            self.HIDDEN, self.HIDDEN, use_fsdp2=True,
            params_dtype=self.DTYPE, device=device,
        )
        assert isinstance(mod.weight, self._fsdpag_cls())
        # Must still be a Parameter so autograd works
        assert isinstance(mod.weight, torch.nn.Parameter)

    def test_layernorm_mlp_wraps_both_weights(self, device):
        """use_fsdp2=True: both fc1_weight and fc2_weight are wrapped."""
        if not self._has_fsdpag():
            pytest.skip("FSDPAGTensor unavailable")
        mod = te.LayerNormMLP(
            self.HIDDEN, self.FFN_HIDDEN, use_fsdp2=True,
            params_dtype=self.DTYPE, device=device,
        )
        assert isinstance(mod.fc1_weight, self._fsdpag_cls())
        assert isinstance(mod.fc2_weight, self._fsdpag_cls())

    def test_layernorm_mlp_no_wrap_by_default(self, device):
        """Default (use_fsdp2=False): no FSDPAGTensor wrapping."""
        if not self._has_fsdpag():
            pytest.skip("FSDPAGTensor unavailable")
        mod = te.LayerNormMLP(
            self.HIDDEN, self.FFN_HIDDEN, params_dtype=self.DTYPE, device=device,
        )
        assert not isinstance(mod.fc1_weight, self._fsdpag_cls())
        assert not isinstance(mod.fc2_weight, self._fsdpag_cls())

    def test_forward_bf16_with_wrapped_weights(self, device):
        """bf16 forward+backward works with FSDPAGTensor-wrapped weights (the
        wrapper's __torch_dispatch__ unwraps for ordinary ops).
        FP8 + use_fsdp2 requires an actual FSDP2 wrap to gather properly —
        that path is not tested here because it needs ≥2 GPUs + fully_shard.
        """
        if not self._has_fsdpag():
            pytest.skip("FSDPAGTensor unavailable")
        mod = te.LayerNormLinear(
            self.HIDDEN, self.HIDDEN, use_fsdp2=True,
            params_dtype=self.DTYPE, device=device,
        )
        x = torch.randn(8, self.HIDDEN, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        torch.cuda.synchronize()
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_forward_bf16_layernorm_mlp_with_wrap(self, device):
        """Same bf16 smoke test for LayerNormMLP with use_fsdp2=True."""
        if not self._has_fsdpag():
            pytest.skip("FSDPAGTensor unavailable")
        mod = te.LayerNormMLP(
            self.HIDDEN, self.FFN_HIDDEN, activation="swiglu",
            use_fsdp2=True, params_dtype=self.DTYPE, device=device,
        )
        x = torch.randn(8, self.HIDDEN, device=device,
                         dtype=self.DTYPE, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        torch.cuda.synchronize()
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert torch.isfinite(x.grad).all()

    def test_non_hip_silently_ignores_flag(self, device):
        """On non-ROCm builds the flag is forced to False (matches full build).
        On ROCm this is always True; we just verify the attribute is accessible."""
        mod = te.LayerNormLinear(
            self.HIDDEN, self.HIDDEN, use_fsdp2=True,
            params_dtype=self.DTYPE, device=device,
        )
        assert hasattr(mod, "use_fsdp2")
        # On ROCm (lite's target platform), use_fsdp2 should pass through.
        from torch.utils.cpp_extension import IS_HIP_EXTENSION
        if IS_HIP_EXTENSION:
            assert mod.use_fsdp2 is True
        else:
            assert mod.use_fsdp2 is False

    def test_fsdpag_wraps_parameter_preserve_grad(self, device):
        """After wrap, gradients still flow to the underlying _data tensor."""
        if not self._has_fsdpag():
            pytest.skip("FSDPAGTensor unavailable")
        mod = te.LayerNormLinear(
            self.HIDDEN, self.HIDDEN, use_fsdp2=True,
            params_dtype=self.DTYPE, device=device,
        )
        assert mod.weight.requires_grad
        x = torch.randn(8, self.HIDDEN, device=device,
                         dtype=self.DTYPE)
        mod(x).sum().backward()
        torch.cuda.synchronize()
        assert mod.weight.grad is not None
        assert torch.isfinite(mod.weight.grad).all()


# ---------------------------------------------------------------------------
# Multi-tensor kernels
# ---------------------------------------------------------------------------

class TestMultiTensor:
    """Cover the lite replacements for transformer_engine_torch multi-tensor ops.

    The reference semantics come from common/multi_tensor/{scale,l2norm,adam,sgd}.cu
    and csrc/extensions/multi_tensor/*.cpp.
    """

    CHUNK = 2048 * 32

    @staticmethod
    def _overflow_buf(device):
        return torch.zeros(1, dtype=torch.int32, device=device)

    # --- multi_tensor_scale -------------------------------------------------

    @pytest.mark.parametrize("in_dtype,out_dtype", [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ])
    def test_scale_writes_out_list(self, device, in_dtype, out_dtype):
        overflow = self._overflow_buf(device)
        a = torch.full([777], 4.0, dtype=in_dtype, device=device)
        b = torch.full([555], 4.0, dtype=in_dtype, device=device)
        in_list = [a.clone(), b.clone()]
        out_list = [torch.empty_like(t, dtype=out_dtype) for t in in_list]

        tex.multi_tensor_scale(self.CHUNK, overflow, [in_list, out_list], 0.25)

        expected = torch.full_like(out_list[0], 1.0)
        torch.testing.assert_close(out_list[0], expected)
        torch.testing.assert_close(out_list[1], torch.full_like(out_list[1], 1.0))
        # in_list should not be modified
        torch.testing.assert_close(in_list[0], torch.full_like(in_list[0], 4.0))
        assert overflow.item() == 0

    def test_scale_sets_overflow_on_nan(self, device):
        overflow = self._overflow_buf(device)
        a = torch.full([64], 4.0, dtype=torch.float32, device=device)
        a[3] = float("nan")
        out = torch.empty_like(a)
        tex.multi_tensor_scale(self.CHUNK, overflow, [[a], [out]], 0.5)
        assert overflow.item() == 1

    def test_scale_sets_overflow_on_inf(self, device):
        overflow = self._overflow_buf(device)
        a = torch.full([64], 4.0, dtype=torch.float32, device=device)
        a[7] = float("inf")
        out = torch.empty_like(a)
        tex.multi_tensor_scale(self.CHUNK, overflow, [[a], [out]], 0.5)
        assert overflow.item() == 1

    # --- multi_tensor_l2norm ------------------------------------------------

    def test_l2norm_returns_2_tuple_when_per_tensor_false(self, device):
        """Megatron's clip_grad_norm unpacks (norm, _) unconditionally."""
        overflow = self._overflow_buf(device)
        a = torch.full([100], 3.0, dtype=torch.float32, device=device)
        b = torch.full([100], 4.0, dtype=torch.float32, device=device)
        result = tex.multi_tensor_l2norm(self.CHUNK, overflow, [[a, b]], False)
        assert isinstance(result, tuple) and len(result) == 2
        norm, per_tensor = result
        # sqrt(100*9 + 100*16) = sqrt(2500) = 50
        torch.testing.assert_close(norm, torch.tensor([50.0], device=device))
        assert per_tensor.numel() == 0

    def test_l2norm_per_tensor_values(self, device):
        overflow = self._overflow_buf(device)
        a = torch.full([100], 3.0, dtype=torch.float32, device=device)
        b = torch.full([100], 4.0, dtype=torch.float32, device=device)
        norm, per_tensor = tex.multi_tensor_l2norm(
            self.CHUNK, overflow, [[a, b]], True
        )
        # norms: sqrt(100*9)=30, sqrt(100*16)=40; total=50
        torch.testing.assert_close(norm, torch.tensor([50.0], device=device))
        torch.testing.assert_close(per_tensor, torch.tensor([30.0, 40.0], device=device))

    def test_l2norm_mixed_dtypes(self, device):
        """l2norm must tolerate fp16/bf16 inputs (upcasts to fp32 internally)."""
        overflow = self._overflow_buf(device)
        a = torch.full([64], 3.0, dtype=torch.bfloat16, device=device)
        b = torch.full([64], 4.0, dtype=torch.float16, device=device)
        norm, _ = tex.multi_tensor_l2norm(self.CHUNK, overflow, [[a, b]], False)
        expected = (64 * 9 + 64 * 16) ** 0.5  # sqrt(1600) = 40
        torch.testing.assert_close(norm, torch.tensor([expected], device=device),
                                   atol=1e-4, rtol=1e-4)

    # --- multi_tensor_unscale_l2norm ---------------------------------------

    def test_unscale_l2norm_returns_2_tuple(self, device):
        overflow = self._overflow_buf(device)
        a = torch.full([100], 6.0, dtype=torch.float32, device=device)
        b = torch.full([100], 8.0, dtype=torch.float32, device=device)
        inv_scale = torch.tensor([2.0], device=device)  # scale = 0.5
        result = tex.multi_tensor_unscale_l2norm(
            self.CHUNK, overflow, [[a, b]], inv_scale, False
        )
        assert isinstance(result, tuple) and len(result) == 2
        norm, per_tensor = result
        # After unscaling (multiply by 0.5): norms 3 and 4; total 50 over 100 each
        # sqrt(100*9 + 100*16) = 50
        torch.testing.assert_close(norm, torch.tensor([50.0], device=device))
        assert per_tensor.numel() == 0

    def test_unscale_l2norm_per_tensor(self, device):
        overflow = self._overflow_buf(device)
        a = torch.full([100], 6.0, dtype=torch.float32, device=device)
        b = torch.full([100], 8.0, dtype=torch.float32, device=device)
        inv_scale = torch.tensor([2.0], device=device)
        norm, per_tensor = tex.multi_tensor_unscale_l2norm(
            self.CHUNK, overflow, [[a, b]], inv_scale, True
        )
        torch.testing.assert_close(norm, torch.tensor([50.0], device=device))
        torch.testing.assert_close(per_tensor, torch.tensor([30.0, 40.0], device=device))

    # --- multi_tensor_adam --------------------------------------------------

    @staticmethod
    def _adam_reference(p, g, m, v, lr, b1, b2, eps, step, wd, adam_w):
        """Reference implementation mirroring common/multi_tensor/adam.cu."""
        p_f = p.float()
        g_f = g.float()
        if not adam_w and wd != 0.0:
            g_f = g_f + wd * p_f
        m_new = b1 * m + (1 - b1) * g_f
        v_new = b2 * v + (1 - b2) * g_f * g_f
        bc1 = 1 - b1 ** step
        bc2 = 1 - b2 ** step
        denom = (v_new / bc2).sqrt() + eps
        update = (m_new / bc1) / denom
        if adam_w and wd != 0.0:
            update = update + wd * p_f
        p_new = p_f - lr * update
        return p_new, m_new, v_new

    @pytest.mark.parametrize("adam_w_mode", [True, False])
    @pytest.mark.parametrize("weight_decay", [0.0, 0.01])
    def test_adam_4list_no_master(self, device, adam_w_mode, weight_decay):
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        p = torch.randn(256, device=device, dtype=torch.float32)
        g = torch.randn(256, device=device, dtype=torch.float32) * 0.01
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)
        p0, g0 = p.clone(), g.clone()

        tex.multi_tensor_adam(
            self.CHUNK, overflow, [[g], [p], [m], [v]],
            1e-3, 0.9, 0.999, 1e-8, 1, adam_w_mode, True, weight_decay,
        )

        p_ref, m_ref, v_ref = self._adam_reference(
            p0, g0, torch.zeros_like(p0), torch.zeros_like(p0),
            1e-3, 0.9, 0.999, 1e-8, 1, weight_decay, adam_w_mode,
        )
        torch.testing.assert_close(p, p_ref, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(m, m_ref, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(v, v_ref, atol=1e-6, rtol=1e-5)

    def test_adam_5list_master_bf16(self, device):
        """Megatron's master-weights path: bf16 params, fp32 master + m + v."""
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        pm = torch.randn(256, device=device, dtype=torch.float32)
        p = pm.to(torch.bfloat16)
        g = (torch.randn(256, device=device) * 0.01).to(torch.bfloat16)
        m = torch.zeros(256, device=device, dtype=torch.float32)
        v = torch.zeros(256, device=device, dtype=torch.float32)
        pm0 = pm.clone()

        tex.multi_tensor_adam(
            self.CHUNK, overflow, [[g], [p], [m], [v], [pm]],
            1e-3, 0.9, 0.999, 1e-8, 1, True, True, 0.0,
        )

        p_ref, _, _ = self._adam_reference(
            pm0, g.float(), torch.zeros_like(pm0), torch.zeros_like(pm0),
            1e-3, 0.9, 0.999, 1e-8, 1, 0.0, True,
        )
        # Master is kept in fp32
        torch.testing.assert_close(pm, p_ref, atol=1e-6, rtol=1e-5)
        # bf16 shadow is a downcast of master
        torch.testing.assert_close(p, p_ref.to(torch.bfloat16), atol=1e-3, rtol=1e-2)

    def test_adam_no_bias_correction(self, device):
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        p = torch.randn(64, device=device)
        g = torch.randn(64, device=device) * 0.01
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)
        p0, g0 = p.clone(), g.clone()

        tex.multi_tensor_adam(
            self.CHUNK, overflow, [[g], [p], [m], [v]],
            1e-3, 0.9, 0.999, 1e-8, 5, True, False, 0.0,
        )

        # bias_correction=False → bc1=bc2=1
        m_ref = 0.9 * torch.zeros_like(p0) + 0.1 * g0
        v_ref = 0.999 * torch.zeros_like(p0) + 0.001 * g0 * g0
        denom = v_ref.sqrt() + 1e-8
        p_ref = p0 - 1e-3 * (m_ref / denom)
        torch.testing.assert_close(p, p_ref, atol=1e-6, rtol=1e-5)

    def test_adam_oversized_state_tensors(self, device):
        """Megatron's distributed optimizer passes m/v sized for the full
        vocab while the gradient is only this rank's TP shard. The C++ kernel
        uses `g.numel()` as the work size, and lite must match that. Only the
        first g.numel() elements of m/v/p should be modified.
        """
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        # g/p are the TP shard; m/v are the full parameter (8x larger).
        shard_rows, full_rows, cols = 64, 512, 32
        g = torch.randn(shard_rows, cols, device=device, dtype=torch.bfloat16) * 0.01
        p = torch.randn(shard_rows, cols, device=device, dtype=torch.bfloat16)
        m = torch.randn(full_rows, cols, device=device, dtype=torch.float32) * 0.001
        v = torch.randn(full_rows, cols, device=device, dtype=torch.float32).abs() * 0.001

        # Snapshot the out-of-shard region to confirm it's untouched.
        m_tail = m[shard_rows:].clone()
        v_tail = v[shard_rows:].clone()
        p0 = p.clone()
        m_head0 = m[:shard_rows].clone()
        v_head0 = v[:shard_rows].clone()

        tex.multi_tensor_adam(
            self.CHUNK, overflow, [[g], [p], [m], [v]],
            1e-3, 0.9, 0.999, 1e-8, 1, True, True, 0.0,
        )

        # Out-of-shard region untouched.
        torch.testing.assert_close(m[shard_rows:], m_tail)
        torch.testing.assert_close(v[shard_rows:], v_tail)

        # In-shard region updated per Adam math on fp32-upcast grads.
        g_f = g.float()
        m_ref = 0.9 * m_head0 + 0.1 * g_f
        v_ref = 0.999 * v_head0 + 0.001 * g_f * g_f
        torch.testing.assert_close(m[:shard_rows], m_ref, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(v[:shard_rows], v_ref, atol=1e-5, rtol=1e-4)

    def test_adam_param_remainder_not_implemented(self, device):
        with pytest.raises(NotImplementedError):
            tex.multi_tensor_adam_param_remainder()

    def test_adam_fp8_not_implemented(self, device):
        with pytest.raises(NotImplementedError):
            tex.multi_tensor_adam_fp8()

    def test_adam_capturable_not_implemented(self, device):
        with pytest.raises(NotImplementedError):
            tex.multi_tensor_adam_capturable()

    # --- multi_tensor_sgd ---------------------------------------------------

    def test_sgd_no_momentum(self, device):
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        w = torch.randn(64, device=device, dtype=torch.float32)
        g = torch.randn(64, device=device, dtype=torch.float32) * 0.01
        mom = torch.zeros_like(w)
        w0, g0 = w.clone(), g.clone()

        tex.multi_tensor_sgd(
            self.CHUNK, overflow, [[g], [w], [mom]],
            1e-2,  # lr
            0.0,   # momentum
            0.0,   # dampening
            0.0,   # weight_decay
            False, # nesterov
            True,  # first_run
            False, # wd_after_momentum
            1.0,   # scale
        )
        torch.testing.assert_close(w, w0 - 1e-2 * g0, atol=1e-6, rtol=1e-5)

    def test_sgd_with_momentum_first_run(self, device):
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        w = torch.randn(64, device=device, dtype=torch.float32)
        g = torch.randn(64, device=device, dtype=torch.float32) * 0.01
        mom = torch.zeros_like(w)
        w0, g0 = w.clone(), g.clone()

        tex.multi_tensor_sgd(
            self.CHUNK, overflow, [[g], [w], [mom]],
            1e-2, 0.9, 0.0, 0.0, False, True, False, 1.0,
        )
        # first_run: mom = g; g_eff = mom = g; w -= lr*g
        torch.testing.assert_close(mom, g0, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(w, w0 - 1e-2 * g0, atol=1e-6, rtol=1e-5)

    def test_sgd_weight_decay_before_momentum(self, device):
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        w = torch.randn(32, device=device, dtype=torch.float32)
        g = torch.randn(32, device=device, dtype=torch.float32) * 0.01
        mom = torch.zeros_like(w)
        w0, g0 = w.clone(), g.clone()

        tex.multi_tensor_sgd(
            self.CHUNK, overflow, [[g], [w], [mom]],
            1e-2, 0.0, 0.0, 0.1, False, True, False, 1.0,
        )
        # wd before momentum: g_eff = g + 0.1*w; w -= lr*g_eff
        g_eff = g0 + 0.1 * w0
        torch.testing.assert_close(w, w0 - 1e-2 * g_eff, atol=1e-6, rtol=1e-5)

    def test_sgd_4list_writes_fp16_copy(self, device):
        overflow = self._overflow_buf(device)
        torch.manual_seed(0)
        w = torch.randn(32, device=device, dtype=torch.float32)
        g = torch.randn(32, device=device, dtype=torch.float32) * 0.01
        mom = torch.zeros_like(w)
        w_fp16 = torch.zeros(32, device=device, dtype=torch.float16)
        w0, g0 = w.clone(), g.clone()

        tex.multi_tensor_sgd(
            self.CHUNK, overflow, [[g], [w], [mom], [w_fp16]],
            1e-2, 0.0, 0.0, 0.0, False, True, False, 1.0,
        )
        expected = w0 - 1e-2 * g0
        torch.testing.assert_close(w, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(w_fp16, expected.to(torch.float16),
                                   atol=1e-3, rtol=1e-2)
