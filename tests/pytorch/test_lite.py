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

# Ensure lite mode is active before importing TE
os.environ["NVTE_LITE"] = "1"

import transformer_engine.pytorch as te  # noqa: E402
import transformer_engine_torch as tex  # noqa: E402


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
            "generic_gemm", "layernorm_fwd", "layernorm_bwd",
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

    @pytest.mark.xfail(reason="TransformerLayer backward has autograd Variable issue with fc2_bias (Phase 1)")
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
            x, weight, bias, 1e-5, None, None, None, 0, False,
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

        y_pt, _, rstd_pt = _rmsnorm_fwd_pytorch(
            x, weight, 1e-5, None, None, None, 0, False,
        )
        y_te, _, rstd_te = tex.rmsnorm_fwd(
            x, weight, 1e-5, None, None, None, 0, False,
        )
        if hasattr(y_te, 'dequantize'):
            y_te = y_te.dequantize()
        assert torch.allclose(y_te.to(torch.bfloat16), y_pt, atol=5e-3, rtol=1e-2), (
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
            x, weight, bias, 1e-5, None, None, None, 0, False,
        )
        dx_pt, dw_pt, db_pt = _layernorm_bwd_pytorch(
            grad_out, x, mean, rstd, weight, 0, False,
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

        _, _, rstd = _rmsnorm_fwd_pytorch(
            x, weight, 1e-5, None, None, None, 0, False,
        )
        dx_pt, dw_pt = _rmsnorm_bwd_pytorch(
            grad_out, x, rstd, weight, 0, False,
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
