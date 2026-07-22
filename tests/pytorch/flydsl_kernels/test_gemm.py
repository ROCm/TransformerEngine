# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

"""User-facing FlyDSL GEMM tests -- ``general_gemm()`` under ``NVTE_USE_FLYDSL=1``.

Exercises the same public entry point used by TE ``Linear`` /
``LayerNormLinear``. Coverage mirrors the Triton user-facing GEMM tests for the
currently supported FlyDSL surface:

- fp32 / fp16 / bf16 regular tensors
- same-format and mixed-format tensor-wise FP8
- same-format and mixed-format MXFP8
- TN / NN / NT layouts
- batched multidimensional FP8 flattening

Fused BIAS and BGRADB epilogues are intentionally not included yet because the
FlyDSL GEMM path does not currently support them.

Each test compares the FlyDSL path against two independent references:

1. ``torch.matmul`` on dequantized inputs, independent of hipBLASLt behavior.
2. The native C++ ``tex.generic_gemm`` backend through the same
   ``general_gemm`` public surface.

FlyDSL kernels currently require tile-aligned launch dimensions, so the test
shapes are aligned to the 256x256x128 kernel contract rather than reusing the
odd-sized Triton edge-mask cases.
"""

import os

import pytest
import torch

from transformer_engine.pytorch import Float8Tensor
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import (
    MXFP8Quantizer,
    MXFP8Tensor,
)
import transformer_engine_torch as tex


# --- Feature detection --------------------------------------------------------

major, minor = torch.cuda.get_device_capability()

# The current FlyDSL MXFP8 implementation uses the gfx950 fp8-scaled MFMA.
has_mxfp8_support = major == 9 and minor >= 5

requires_mxfp8_support = pytest.mark.skipif(
    not has_mxfp8_support,
    reason="FlyDSL MXFP8 requires gfx950+ fp8-scaled MFMA support",
)


# --- Test parameters ----------------------------------------------------------

# The current FlyDSL kernels have no M/N edge masks and specialize K in K128
# tiles. Keep all dimensions aligned to exercise the supported production path.
FLYDSL_SHAPES = [
    (512, 512, 512),
    (512, 1024, 512),
    (1024, 512, 1024),
]

MXFP8_SHAPES = [
    (512, 512, 512),
    (512, 1024, 512),
]

LAYOUTS = ["TN", "NN", "NT"]

FP8_FORMAT_COMBOS = [
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3),
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E4M3),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E5M2),
]

FP8_FORMAT_IDS = [
    "e4m3_e4m3",
    "e4m3_e5m2",
    "e5m2_e4m3",
    "e5m2_e5m2",
]

REGULAR_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# --- Fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_env():
    """Save and restore FlyDSL-related environment variables between tests."""
    old_flydsl = os.environ.get("NVTE_USE_FLYDSL")
    old_mxfp8 = os.environ.get("NVTE_ROCM_ENABLE_MXFP8")

    yield

    if old_flydsl is None:
        os.environ.pop("NVTE_USE_FLYDSL", None)
    else:
        os.environ["NVTE_USE_FLYDSL"] = old_flydsl

    if old_mxfp8 is None:
        os.environ.pop("NVTE_ROCM_ENABLE_MXFP8", None)
    else:
        os.environ["NVTE_ROCM_ENABLE_MXFP8"] = old_mxfp8


# --- Helpers ------------------------------------------------------------------

def get_shapes(layout, M, K, N):
    """Return the A/B storage shapes used by TE's public GEMM tests."""
    if layout == "TN":
        return (M, K), (N, K)
    if layout == "NN":
        return (M, K), (K, M)
    if layout == "NT":
        return (M, K), (M, K)
    raise ValueError(f"Unsupported layout: {layout}")


def compute_pytorch_reference(A_ref, B_ref, layout):
    """Compute the equivalent public-layout GEMM with ``torch.matmul``."""
    if layout == "TN":
        return torch.matmul(B_ref, A_ref.T)
    if layout == "NN":
        return torch.matmul(B_ref, A_ref)
    if layout == "NT":
        return torch.matmul(B_ref.T, A_ref)
    raise ValueError(f"Unsupported layout: {layout}")


def create_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b):
    """Create independently typed Float8Tensor inputs and references."""
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A_f32 = torch.randn(A_shape, dtype=torch.float32, device="cuda") * 0.5
    B_f32 = torch.randn(B_shape, dtype=torch.float32, device="cuda") * 0.5

    A_fp8 = Float8Quantizer(
        scale=torch.full((1,), 1.0, dtype=torch.float32, device="cuda"),
        amax=torch.empty((1,), dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype_a,
    )(A_f32)
    B_fp8 = Float8Quantizer(
        scale=torch.full((1,), 1.0, dtype=torch.float32, device="cuda"),
        amax=torch.empty((1,), dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype_b,
    )(B_f32)

    return A_fp8, B_fp8, A_fp8.dequantize(), B_fp8.dequantize()


def _make_mxfp8_quantizer(fp8_dtype):
    """Create one independently typed MXFP8 quantizer with both orientations."""
    quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype)
    quantizer.set_usage(rowwise=True, columnwise=True)
    return quantizer


def create_mxfp8_tensors(
    M,
    K,
    N,
    layout,
    fp8_dtype_a,
    fp8_dtype_b,
):
    """Create independently typed MXFP8Tensor inputs and references."""
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A_f32 = torch.randn(A_shape, dtype=torch.float32, device="cuda") * 0.5
    B_f32 = torch.randn(B_shape, dtype=torch.float32, device="cuda") * 0.5

    A_mxfp8 = _make_mxfp8_quantizer(fp8_dtype_a)(A_f32)
    B_mxfp8 = _make_mxfp8_quantizer(fp8_dtype_b)(B_f32)

    return (
        A_mxfp8,
        B_mxfp8,
        A_mxfp8.dequantize(),
        B_mxfp8.dequantize(),
    )


def call_gemm(A, B, layout, out_dtype, use_flydsl=True):
    """Call ``general_gemm`` through either FlyDSL or the native C++ path."""
    os.environ["NVTE_USE_FLYDSL"] = "1" if use_flydsl else "0"

    output, bias_grad, gelu_input, extra_output = general_gemm(
        A=A,
        B=B,
        out_dtype=out_dtype,
        layout=layout,
        bias=None,
        quantization_params=None,
        gelu=False,
        grad=False,
        accumulate=False,
    )

    assert bias_grad is None
    assert gelu_input is None
    assert extra_output is None
    return output


def assert_gemm_close(actual, expected, *, atol, rtol):
    """Compare through FP32 so output narrowing does not hide diagnostics."""
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )


# ==============================================================================
# Approach 1: FlyDSL vs PyTorch torch.matmul reference
# ==============================================================================

@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "dtype",
    REGULAR_DTYPES,
    ids=["fp32", "fp16", "bf16"],
)
def test_flydsl_vs_pytorch_regular(M, K, N, layout, dtype):
    """Test regular FlyDSL GEMM against an FP32 PyTorch reference."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    output = call_gemm(
        A,
        B,
        layout,
        out_dtype=dtype,
        use_flydsl=True,
    )
    expected = compute_pytorch_reference(A.float(), B.float(), layout)

    assert_gemm_close(output, expected, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_pytorch_fp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format tensor-wise FP8 FlyDSL GEMMs."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_fp8_tensors(
        M,
        K,
        N,
        layout,
        fp8_dtype_a,
        fp8_dtype_b,
    )

    output = call_gemm(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    expected = compute_pytorch_reference(
        A_deq.float(),
        B_deq.float(),
        layout,
    )

    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_pytorch_mxfp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format MXFP8 FlyDSL GEMMs."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_mxfp8_tensors(
        M,
        K,
        N,
        layout,
        fp8_dtype_a,
        fp8_dtype_b,
    )

    output = call_gemm(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    expected = compute_pytorch_reference(
        A_deq.float(),
        B_deq.float(),
        layout,
    )

    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


# ==============================================================================
# Approach 2: FlyDSL vs native C++ ``generic_gemm`` reference
# ==============================================================================

@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "dtype",
    REGULAR_DTYPES,
    ids=["fp32", "fp16", "bf16"],
)
def test_flydsl_vs_cpp_regular(M, K, N, layout, dtype):
    """Test regular FlyDSL GEMM against the native C++ backend."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    flydsl_out = call_gemm(
        A,
        B,
        layout,
        out_dtype=dtype,
        use_flydsl=True,
    )
    cpp_out = call_gemm(
        A,
        B,
        layout,
        out_dtype=dtype,
        use_flydsl=False,
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_cpp_fp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format FP8 against native C++."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, _, _ = create_fp8_tensors(
        M,
        K,
        N,
        layout,
        fp8_dtype_a,
        fp8_dtype_b,
    )

    flydsl_out = call_gemm(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    cpp_out = call_gemm(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=False,
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=5e-3, rtol=1e-2)


@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_cpp_mxfp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format MXFP8 against native C++."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, _, _ = create_mxfp8_tensors(
        M,
        K,
        N,
        layout,
        fp8_dtype_a,
        fp8_dtype_b,
    )

    flydsl_out = call_gemm(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    cpp_out = call_gemm(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=False,
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=5e-3, rtol=1e-2)


# ==============================================================================
# Batched multidimensional FP8 coverage
# ==============================================================================

@pytest.mark.parametrize(
    "batch_size, M, K, N",
    [
        (2, 256, 512, 256),
        (4, 256, 512, 256),
    ],
)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_pytorch_fp8_multidim(
    batch_size,
    M,
    K,
    N,
    fp8_format,
):
    """Exercise flatten-leading-dim semantics for multidimensional FP8."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format

    # TN layout: the wrapper flattens all leading dimensions into rows.
    A_f32 = (
        torch.randn(
            batch_size,
            M,
            K,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.5
    )
    B_f32 = (
        torch.randn(
            batch_size,
            N,
            K,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.5
    )

    A_fp8 = Float8Quantizer(
        scale=torch.full((1,), 1.0, dtype=torch.float32, device="cuda"),
        amax=torch.empty((1,), dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype_a,
    )(A_f32)
    B_fp8 = Float8Quantizer(
        scale=torch.full((1,), 1.0, dtype=torch.float32, device="cuda"),
        amax=torch.empty((1,), dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype_b,
    )(B_f32)

    output = call_gemm(
        A_fp8,
        B_fp8,
        layout="TN",
        out_dtype=torch.float32,
        use_flydsl=True,
    )

    A_flat = A_fp8.dequantize().reshape(-1, K)
    B_flat = B_fp8.dequantize().reshape(-1, K)
    expected = torch.matmul(B_flat, A_flat.T)

    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


if __name__ == "__main__":
    # Quick smoke tests using one case from each supported input family.
    os.environ["NVTE_USE_FLYDSL"] = "1"
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"

    test_flydsl_vs_pytorch_regular(
        256,
        512,
        256,
        "TN",
        torch.float16,
    )
    test_flydsl_vs_pytorch_fp8(
        256,
        512,
        256,
        "TN",
        (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2),
    )

    if has_mxfp8_support:
        test_flydsl_vs_pytorch_mxfp8(
            256,
            512,
            256,
            "TN",
            (tex.DType.kFloat8E5M2, tex.DType.kFloat8E4M3),
        )

    print("All FlyDSL GEMM smoke tests passed!")
