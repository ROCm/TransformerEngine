# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

"""
Consolidated test for te_generic_gemm_triton() via general_gemm().

Tests regular, FP8, and MXFP8 tensor types with two reference approaches:
  1. Triton vs PyTorch torch.matmul reference
  2. Triton vs C++ tex.generic_gemm reference
"""

import os
import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch import Float8Tensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
import transformer_engine_torch as tex

# --- Feature detection --------------------------------------------------------

major, minor = torch.cuda.get_device_capability()
is_gfx950 = (major == 9 and minor >= 5)

from transformer_engine.pytorch import torch_version
_torch_ver = torch_version()

requires_gfx950 = pytest.mark.skipif(
    not is_gfx950,
    reason="MXFP8 requires gfx950 (compute capability >= 9.5)",
)

# --- Test parameters ----------------------------------------------------------

REGULAR_FP8_SHAPES = [
    (2304, 768, 4096),
    (768, 768, 4096),
    (768, 3072, 4096),
    (229, 541, 541),
    (71, 71, 3571),
    (29, 29, 17389),
]

MXFP8_SHAPES = [
    (128, 256, 512),
    (768, 768, 4096),
    (224, 544, 544),
]

LAYOUTS = ["TN", "NN", "NT"]

FP8_FORMAT_COMBOS = [
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E5M2),
]

# Mixed FP8 formats are disabled due to a Triton compiler bug on gfx950:
# when the MFMA layout is transposed, operand B is packed using A's element type,
# and the instruction format encoding doesn't account for the operand swap.
# This affects both v_mfma_f32_32x32x16_{fp8|bf8} and v_mfma_f32_32x32x64_f8f6f4.
# Fixed upstream in triton-lang/triton PR #9567 (commit eaaa75cf5, 2026-02-27).
# Not yet in any pytorch-triton-rocm release as of PyTorch 2.11.
# TODO: Re-enable once pytorch-triton-rocm includes the fix (expected PyTorch 2.12+).
FP8_MIXED_FORMAT_COMBOS = [
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E4M3),
]

REGULAR_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# --- Fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_env():
    """Save/restore NVTE_USE_GEMM_TRITON env var between tests."""
    old_val = os.environ.get('NVTE_USE_GEMM_TRITON', None)
    yield
    if old_val is None:
        os.environ.pop('NVTE_USE_GEMM_TRITON', None)
    else:
        os.environ['NVTE_USE_GEMM_TRITON'] = old_val


# --- Helpers ------------------------------------------------------------------

def get_shapes(layout, M, K, N):
    """Returns (A_shape, B_shape) based on layout."""
    if layout == "TN":
        return (M, K), (N, K)
    elif layout == "NN":
        return (M, K), (K, M)
    elif layout == "NT":
        return (M, K), (M, K)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def compute_pytorch_reference(A_ref, B_ref, layout):
    """torch.matmul with correct transpose for layout."""
    if layout == "TN":
        return torch.matmul(B_ref, A_ref.T)
    elif layout == "NN":
        return torch.matmul(B_ref, A_ref)
    elif layout == "NT":
        return torch.matmul(B_ref.T, A_ref)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def create_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b):
    """Create Float8Tensor inputs and dequantized references."""
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A_f32 = torch.randn(A_shape, dtype=torch.float32, device='cuda') * 0.5
    B_f32 = torch.randn(B_shape, dtype=torch.float32, device='cuda') * 0.5

    A_fp8 = Float8Quantizer(
        scale=torch.full([1], 1.0, dtype=torch.float32, device='cuda'),
        amax=torch.empty([1], dtype=torch.float32, device='cuda'),
        fp8_dtype=fp8_dtype_a,
    )(A_f32)
    B_fp8 = Float8Quantizer(
        scale=torch.full([1], 1.0, dtype=torch.float32, device='cuda'),
        amax=torch.empty([1], dtype=torch.float32, device='cuda'),
        fp8_dtype=fp8_dtype_b,
    )(B_f32)

    return A_fp8, B_fp8, A_fp8.dequantize(), B_fp8.dequantize()


def create_mxfp8_tensors(M, K, N, layout):
    """Create MXFP8Tensor inputs and dequantized references."""
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A_f32 = torch.randn(A_shape, dtype=torch.float32, device='cuda') * 0.5
    B_f32 = torch.randn(B_shape, dtype=torch.float32, device='cuda') * 0.5

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    A_mxfp8 = quantizer(A_f32)
    B_mxfp8 = quantizer(B_f32)

    return A_mxfp8, B_mxfp8, A_mxfp8.dequantize(), B_mxfp8.dequantize()


def call_gemm(A, B, layout, out_dtype, use_triton=True):
    """Call general_gemm() with appropriate env var setting."""
    os.environ['NVTE_USE_GEMM_TRITON'] = '1' if use_triton else '0'
    output, _, _, _ = general_gemm(
        A=A,
        B=B,
        out_dtype=out_dtype,
        layout=layout,
    )
    return output


def call_gemm_with_bias(A, B, layout, out_dtype, bias, grad, use_triton=True):
    """Call general_gemm() with a bias argument.

    Returns (output, bias_grad). When grad=True the GEMM uses the BGRADB
    epilogue and bias_grad contains the reduced bias gradient; otherwise
    it uses the BIAS epilogue and bias is fused into the output.
    """
    os.environ['NVTE_USE_GEMM_TRITON'] = '1' if use_triton else '0'
    output, bias_grad, _, _ = general_gemm(
        A=A,
        B=B,
        out_dtype=out_dtype,
        layout=layout,
        bias=bias,
        grad=grad,
    )
    return output, bias_grad


# ==============================================================================
# Approach 1: Triton vs PyTorch torch.matmul reference
# ==============================================================================

@pytest.mark.parametrize("M, K, N", REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_triton_vs_pytorch_regular(M, K, N, layout, dtype):
    """Test Triton GEMM vs torch.matmul for regular tensors."""
    torch.manual_seed(42)
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda') * 0.5
    B = torch.randn(B_shape, dtype=dtype, device='cuda') * 0.5

    # Triton result
    output = call_gemm(A, B, layout, out_dtype=dtype, use_triton=True)

    # PyTorch reference on fp32 copies
    expected = compute_pytorch_reference(A.float(), B.float(), layout)

    torch.testing.assert_close(
        output.float(), expected.float(),
        atol=1e-3, rtol=1e-2,
    )


@pytest.mark.parametrize("M, K, N", REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS,
                         ids=["e4m3_e4m3", "e5m2_e5m2"])
def test_triton_vs_pytorch_fp8(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs torch.matmul for Float8Tensor inputs."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    output = call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    torch.testing.assert_close(
        output.float(), expected.float(),
        atol=5e-3, rtol=1e-2,
    )


@pytest.mark.skip(reason="Triton compiler bug with mixed FP8 formats (triton-lang/triton#9567)")
@pytest.mark.parametrize("M, K, N", REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_MIXED_FORMAT_COMBOS,
                         ids=["e4m3_e5m2", "e5m2_e4m3"])
def test_triton_vs_pytorch_fp8_mixed(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs torch.matmul for mixed Float8Tensor formats."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    output = call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    torch.testing.assert_close(
        output.float(), expected.float(),
        atol=5e-3, rtol=1e-2,
    )


@requires_gfx950
@pytest.mark.skipif(
    _torch_ver < (2, 10),
    reason=(
        "Triton tl.dot_scaled() RHS scale bug fixed in PyTorch 2.10 "
        f"(found {_torch_ver}). The TE kernel uses the new dot_scaled API "
        "(rhs_scale in [N, K//32] layout) which requires PyTorch >= 2.10."
    ),
)
@pytest.mark.parametrize("M, K, N", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_triton_vs_pytorch_mxfp8(M, K, N, layout):
    """Test Triton GEMM vs torch.matmul for MXFP8Tensor inputs."""
    torch.manual_seed(42)
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_mxfp8_tensors(M, K, N, layout)

    output = call_gemm(A_mxfp8, B_mxfp8, layout, out_dtype=torch.bfloat16, use_triton=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    torch.testing.assert_close(
        output.float(), expected.float(),
        atol=5e-3, rtol=1e-2,
    )


# ==============================================================================
# Approach 2: Triton vs C++ tex.generic_gemm reference
# ==============================================================================

@pytest.mark.parametrize("M, K, N", REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_triton_vs_cpp_regular(M, K, N, layout, dtype):
    """Test Triton GEMM vs C++ generic_gemm for regular tensors."""
    torch.manual_seed(42)
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda') * 0.5
    B = torch.randn(B_shape, dtype=dtype, device='cuda') * 0.5

    triton_out = call_gemm(A, B, layout, out_dtype=dtype, use_triton=True)
    cpp_out = call_gemm(A, B, layout, out_dtype=dtype, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(), cpp_out.float(),
        atol=1e-3, rtol=1e-2,
    )


@pytest.mark.parametrize("M, K, N", REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS,
                         ids=["e4m3_e4m3", "e5m2_e5m2"])
def test_triton_vs_cpp_fp8(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs C++ generic_gemm for Float8Tensor inputs."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, _, _ = create_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    triton_out = call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    cpp_out = call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(), cpp_out.float(),
        atol=5e-3, rtol=1e-2,
    )


@pytest.mark.skip(reason="Triton compiler bug with mixed FP8 formats (triton-lang/triton#9567)")
@pytest.mark.parametrize("M, K, N", REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_MIXED_FORMAT_COMBOS,
                         ids=["e4m3_e5m2", "e5m2_e4m3"])
def test_triton_vs_cpp_fp8_mixed(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs C++ generic_gemm for mixed Float8Tensor formats."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, _, _ = create_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    triton_out = call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    cpp_out = call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(), cpp_out.float(),
        atol=5e-3, rtol=1e-2,
    )


@requires_gfx950
@pytest.mark.skipif(
    _torch_ver < (2, 10),
    reason=(
        "Triton tl.dot_scaled() RHS scale bug fixed in PyTorch 2.10 "
        f"(found {_torch_ver}). The TE kernel uses the new dot_scaled API "
        "(rhs_scale in [N, K//32] layout) which requires PyTorch >= 2.10."
    ),
)
@pytest.mark.parametrize("M, K, N", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_triton_vs_cpp_mxfp8(M, K, N, layout):
    """Test Triton GEMM vs C++ generic_gemm for MXFP8Tensor inputs."""
    torch.manual_seed(42)
    A_mxfp8, B_mxfp8, _, _ = create_mxfp8_tensors(M, K, N, layout)

    triton_out = call_gemm(A_mxfp8, B_mxfp8, layout, out_dtype=torch.bfloat16, use_triton=True)
    cpp_out = call_gemm(A_mxfp8, B_mxfp8, layout, out_dtype=torch.bfloat16, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(), cpp_out.float(),
        atol=5e-3, rtol=1e-2,
    )


# ==============================================================================
# Bias epilogue coverage (regression guard for gemm_triton.py bias wiring)
#
# The Triton wrapper must honor the `bias` + `grad` arguments to general_gemm:
#   - grad=False + bias present → BIAS epilogue, bias added to output
#   - grad=True  + bias present → BGRADB epilogue, bias gradient returned as
#                                  the second element of general_gemm's tuple
# Layout TN matches TE Linear's forward convention: A=weight[M,K],
# B=input[N,K], output[N,M]; BIAS reads bias[M], BGRADB reduces to shape [N].
# ==============================================================================

BIAS_SHAPES = [(128, 256, 512), (229, 541, 541), (71, 71, 3571)]


@pytest.mark.parametrize("M, K, N", BIAS_SHAPES)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_triton_vs_cpp_bias_forward(M, K, N, dtype):
    """Forward with BIAS epilogue: Triton must match C++ when bias is fused."""
    torch.manual_seed(42)
    A_shape, B_shape = get_shapes("TN", M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device='cuda') * 0.5
    B = torch.randn(B_shape, dtype=dtype, device='cuda') * 0.5
    bias = torch.randn((M,), dtype=dtype, device='cuda')

    triton_out, _ = call_gemm_with_bias(A, B, "TN", dtype, bias, grad=False, use_triton=True)
    cpp_out, _ = call_gemm_with_bias(A, B, "TN", dtype, bias, grad=False, use_triton=False)

    # Bias must actually change the result vs. no-bias path; otherwise BIAS
    # silently reverted to DEFAULT would pass a simple Triton-vs-C++ check.
    no_bias_out = call_gemm(A, B, "TN", out_dtype=dtype, use_triton=True)
    assert not torch.allclose(triton_out.float(), no_bias_out.float(), atol=1e-4), (
        "Triton output matches no-bias output; BIAS epilogue appears inactive."
    )

    torch.testing.assert_close(
        triton_out.float(), cpp_out.float(),
        atol=5e-3, rtol=1e-2,
    )


WGRAD_SHAPES = [
    # (batch*seq, in_features, out_features) — TE Linear wgrad pattern
    (256, 128, 512),
    (512, 541, 229),
    (128, 3571, 71),
]


@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_triton_vs_cpp_bias_grad(batch, in_features, out_features, dtype):
    """Backward with BGRADB epilogue: Triton must produce the correct bias gradient.

    Exercises the same call shape TE Linear uses for weight-grad:
      general_gemm(x, dy, layout="NT", bias=<weight.bias>, grad=True)
    A=x[batch, in_features], B=dy[batch, out_features].
    The reduced bias gradient is expected to equal dy.sum(dim=0).

    Regression guard for the wrapper bug where the epilogue was hardcoded to
    DEFAULT, which silently zeroed the returned bias gradient.
    """
    torch.manual_seed(42)
    A = torch.randn((batch, in_features), dtype=dtype, device='cuda') * 0.5
    B = torch.randn((batch, out_features), dtype=dtype, device='cuda') * 0.5
    bias = torch.zeros((out_features,), dtype=dtype, device='cuda')

    _, triton_bias_grad = call_gemm_with_bias(A, B, "NT", dtype, bias, grad=True, use_triton=True)
    _, cpp_bias_grad = call_gemm_with_bias(A, B, "NT", dtype, bias, grad=True, use_triton=False)

    assert triton_bias_grad is not None, "Triton did not return a bias gradient tensor."
    assert cpp_bias_grad is not None, "C++ did not return a bias gradient tensor."
    # A correct BGRADB must not produce an all-zero gradient for non-trivial B.
    assert triton_bias_grad.abs().sum().item() > 0, (
        "Triton bias gradient is all zeros — BGRADB epilogue appears inactive."
    )

    # Cross-check against the analytical reduction.
    expected = B.float().sum(dim=0)
    torch.testing.assert_close(
        triton_bias_grad.float(), expected,
        atol=5e-2, rtol=1e-2,
    )
    torch.testing.assert_close(
        triton_bias_grad.float(), cpp_bias_grad.float(),
        atol=5e-3, rtol=1e-2,
    )


if __name__ == "__main__":
    # Quick smoke test
    os.environ['NVTE_USE_GEMM_TRITON'] = '1'
    test_triton_vs_pytorch_regular(128, 256, 512, "TN", torch.float16)
    test_triton_vs_pytorch_fp8(128, 256, 512, "TN",
                               (tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3))
    test_triton_vs_cpp_regular(128, 256, 512, "TN", torch.float16)
    print("All smoke tests passed!")
