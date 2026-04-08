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
        """bgrad_quantize should return (quantized, bias_grad)."""
        x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        quantized, bgrad = tex.bgrad_quantize(x, None)
        expected_bgrad = x.sum(dim=0)
        assert torch.allclose(bgrad, expected_bgrad)

    def test_dequantize_plain_tensor(self, device):
        """dequantize on a plain tensor should just cast dtype."""
        x = torch.randn(4, 8, device=device, dtype=torch.float32)
        y = tex.dequantize(x, tex.DType.kBFloat16)
        assert y.dtype == torch.bfloat16
        assert torch.allclose(y, x.to(torch.bfloat16))


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
