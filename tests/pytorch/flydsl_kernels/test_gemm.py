# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
- fused BIAS epilogue across all backends (regular, tensor-wise FP8, MXFP8)
- fused GELU_AUX / GELU_AUX_BIAS epilogue across all backends

The fused forward BIAS and GELU (GELU_AUX, saving the pre-activation aux)
epilogues are implemented for every FlyDSL GEMM backend. BGRADB (fused
bias-gradient) and DGELU (fused GELU gradient, grad=True) are not implemented
on any FlyDSL path yet.

Each test compares the FlyDSL path against two independent references:

1. ``torch.matmul`` on dequantized inputs, independent of hipBLASLt behavior.
2. The native C++ ``tex.generic_gemm`` backend through the same
   ``general_gemm`` public surface.

FlyDSL kernels currently require tile-aligned launch dimensions, so the test
shapes are aligned to the 256x256x128 kernel contract rather than reusing the
odd-sized Triton edge-mask cases.
"""

import os
import warnings

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


def _device_capability():
    """Compute capability as (major, minor), or None on a CPU-only box.

    Deferred behind a helper so importing this module never initialises CUDA at
    collection time (which would error the whole module on a CPU-only runner
    instead of skipping it).
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


# All FlyDSL GEMM dispatch is gated on gfx950 in cpp_extensions/gemm.py, not
# just MXFP8. On any other arch general_gemm silently runs the C++ backend, so
# every test here would either exercise hipBLASLt or (for the vs-cpp cases) be
# a tautology. Skip the whole module unless we are on gfx950 with the
# user-installed flydsl package present (importorskip below).
_CAP = _device_capability()
has_flydsl_support = _CAP == (9, 5)
pytestmark = pytest.mark.skipif(
    not has_flydsl_support,
    reason="FlyDSL GEMM dispatch requires gfx950",
)
if has_flydsl_support:
    pytest.importorskip("flydsl", reason="FlyDSL package is not installed")

# The current FlyDSL MXFP8 implementation uses the gfx950 fp8-scaled MFMA.
has_mxfp8_support = has_flydsl_support

requires_mxfp8_support = pytest.mark.skipif(
    not has_mxfp8_support,
    reason="FlyDSL MXFP8 requires gfx950+ fp8-scaled MFMA support",
)


# --- FlyDSL fallback detection ------------------------------------------------

_FLYDSL_FALLBACK_TAG = "[FLYDSL WARNING]"


def _run_capturing_fallback(fn):
    """Run ``fn`` with FlyDSL fallback warnings enabled and captured.

    Returns ``(result, fell_back)`` where ``fell_back`` is True if the dispatch
    emitted a ``[FLYDSL WARNING]`` fallback for any GEMM in ``fn``. The env var
    makes cpp_extensions/gemm.py warn (instead of silently) when a FlyDSL GEMM
    is unsupported and it falls back to the default backend.
    """
    old = os.environ.get("NVTE_FLYDSL_GEMM_WARN_FALLBACK")
    os.environ["NVTE_FLYDSL_GEMM_WARN_FALLBACK"] = "1"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn()
        fell_back = any(_FLYDSL_FALLBACK_TAG in str(w.message) for w in caught)
        return result, fell_back
    finally:
        if old is None:
            os.environ.pop("NVTE_FLYDSL_GEMM_WARN_FALLBACK", None)
        else:
            os.environ["NVTE_FLYDSL_GEMM_WARN_FALLBACK"] = old


# --- Test parameters ----------------------------------------------------------

# The current FlyDSL kernels have no M/N edge masks and specialize K in K128
# tiles. Keep all dimensions aligned to exercise the supported production path.
FLYDSL_SHAPES = [
    (512, 512, 512),
    (512, 1024, 512),
    (1024, 512, 1024),
    (512, 512, 1024),  # M != N: exercises the operand-swap contract asymmetrically
]

MXFP8_SHAPES = [
    (512, 512, 512),
    (512, 1024, 512),
    (512, 512, 1024),  # M != N
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
    """Return the (A, B) storage shapes for a given TE ``general_gemm`` layout.

    Every layout produces a logical ``(N, M)`` output (see
    ``compute_pytorch_reference``), so each must reference both M and N --
    otherwise NN/NT silently collapse to square outputs and never exercise the
    M/N operand-swap contract asymmetrically.
    """
    if layout == "TN":  # A transposed, B not transposed
        return (M, K), (N, K)
    if layout == "NN":  # neither transposed
        return (K, M), (N, K)
    if layout == "NT":  # A not transposed, B transposed
        return (K, M), (K, N)
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


def _assert_flydsl_ran(use_flydsl, fell_back):
    """Fail if FlyDSL was requested for a supported config but silently fell back.

    Every shape in this suite is tile-aligned and a config the PR claims to
    support, so a fallback means FlyDSL did not actually run and the comparison
    below would be vacuous (native vs native).
    """
    if use_flydsl and fell_back:
        pytest.fail(
            "FlyDSL GEMM unexpectedly fell back to the native backend; "
            "the FlyDSL path was not exercised."
        )


def call_gemm(A, B, layout, out_dtype, use_flydsl=True):
    """Call ``general_gemm`` through either FlyDSL or the native C++ path."""
    os.environ["NVTE_USE_FLYDSL"] = "1" if use_flydsl else "0"

    (output, bias_grad, gelu_input, extra_output), fell_back = _run_capturing_fallback(
        lambda: general_gemm(
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
    )
    _assert_flydsl_ran(use_flydsl, fell_back)

    assert bias_grad is None
    assert gelu_input is None
    assert extra_output is None
    return output


def call_gemm_with_bias(A, B, layout, out_dtype, bias, use_flydsl=True):
    """Call ``general_gemm`` with a fused forward BIAS epilogue.

    Bias is a 1-D vector along the output feature axis (the last dim of the
    returned ``(*, out_features)`` tensor) and is added to the matmul result.
    """
    os.environ["NVTE_USE_FLYDSL"] = "1" if use_flydsl else "0"

    (output, bias_grad, gelu_input, extra_output), fell_back = _run_capturing_fallback(
        lambda: general_gemm(
            A=A,
            B=B,
            out_dtype=out_dtype,
            layout=layout,
            bias=bias,
            quantization_params=None,
            gelu=False,
            grad=False,
            accumulate=False,
        )
    )
    _assert_flydsl_ran(use_flydsl, fell_back)

    assert gelu_input is None
    assert extra_output is None
    return output, bias_grad


def call_gemm_with_gelu(A, B, layout, out_dtype, bias=None, use_flydsl=True):
    """Call ``general_gemm`` with a fused forward GELU (GELU_AUX) epilogue.

    Returns ``(output, gelu_input)`` where ``output`` is ``gelu(A@B[+bias])``
    and ``gelu_input`` is the saved pre-activation (``A@B[+bias]``) that the
    backward pass consumes.
    """
    os.environ["NVTE_USE_FLYDSL"] = "1" if use_flydsl else "0"

    (output, bias_grad, gelu_input, extra_output), fell_back = _run_capturing_fallback(
        lambda: general_gemm(
            A=A,
            B=B,
            out_dtype=out_dtype,
            layout=layout,
            bias=bias,
            quantization_params=None,
            gelu=True,
            grad=False,
            accumulate=False,
        )
    )
    _assert_flydsl_ran(use_flydsl, fell_back)

    assert bias_grad is None
    assert extra_output is None
    return output, gelu_input


def gelu_tanh_ref(x):
    """tanh-approx GELU reference (matches the kernel and torch approximate='tanh')."""
    return torch.nn.functional.gelu(x, approximate="tanh")


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


def test_flydsl_mxfp8_unsupported_shape_falls_back():
    """An MXFP8 shape FlyDSL cannot tile must fall back, not crash.

    M=128 is not a multiple of the 256-wide kernel M tile, so FlyDSL rejects it
    with ``FlyDSLUnsupportedError`` -- the only type ``general_gemm`` catches.
    Dispatch must degrade to the C++ backend rather than propagating an
    uncatchable ValueError/RuntimeError, and the fallen-back result must still
    be correct.

    An M/N-tiling mismatch is used rather than an untileable K: the C++ MXFP8
    backend shares FlyDSL's K%128 constraint, so a K-based rejection would also
    fail C++ and could never produce a comparable result. K=512 here keeps the
    scale packing and K-tile count valid, isolating the M-tiling rejection.
    """
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    os.environ["NVTE_USE_FLYDSL"] = "1"
    torch.manual_seed(42)

    M, K, N = 128, 512, 256
    fp8_dtype = tex.DType.kFloat8E4M3
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_mxfp8_tensors(
        M,
        K,
        N,
        "TN",
        fp8_dtype,
        fp8_dtype,
    )

    old_warn = os.environ.get("NVTE_FLYDSL_GEMM_WARN_FALLBACK")
    os.environ["NVTE_FLYDSL_GEMM_WARN_FALLBACK"] = "1"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output, _, _, _ = general_gemm(
                A=A_mxfp8,
                B=B_mxfp8,
                out_dtype=torch.float32,
                layout="TN",
                bias=None,
                quantization_params=None,
                gelu=False,
                grad=False,
                accumulate=False,
            )
    finally:
        if old_warn is None:
            os.environ.pop("NVTE_FLYDSL_GEMM_WARN_FALLBACK", None)
        else:
            os.environ["NVTE_FLYDSL_GEMM_WARN_FALLBACK"] = old_warn

    fallback_msgs = [str(w.message) for w in caught if _FLYDSL_FALLBACK_TAG in str(w.message)]
    for msg in fallback_msgs:
        print(msg)  # visible with ``pytest -s``
    assert fallback_msgs, (
        "an untileable MXFP8 shape should fall back to the C++ backend, "
        "but no [FLYDSL WARNING] fallback was emitted"
    )
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), "TN")
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
# Fused BIAS epilogue coverage
#
# Every FlyDSL GEMM backend (regular fp32/fp16/bf16, tensor-wise FP8, MXFP8)
# adds bias along the output-feature (N) axis -- the last dim of the returned
# ``(M, N)`` tensor -- broadcast across the M/token rows. These guard the full
# public plumbing: general_gemm(bias=...) -> te_generic_gemm_flydsl ->
# _run_<backend> -> <backend>_matmul(epilogue="BIAS").
#
# Each backend has a vs-pytorch test (with a guard that bias actually changes
# the output, catching a silent decay to DEFAULT) and a vs-cpp cross-check.
# ==============================================================================


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_flydsl_vs_pytorch_regular_bias(M, K, N, layout, dtype):
    """Regular fp32/fp16/bf16 GEMM with a fused BIAS epilogue vs PyTorch."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    expected_ab = compute_pytorch_reference(A.float(), B.float(), layout)
    out_features = expected_ab.shape[-1]
    bias = torch.randn(out_features, dtype=dtype, device="cuda")

    output, bias_grad = call_gemm_with_bias(
        A,
        B,
        layout,
        out_dtype=dtype,
        bias=bias,
        use_flydsl=True,
    )
    assert bias_grad is None

    no_bias_out = call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_bias_out.float(), atol=1e-4
    ), "FlyDSL output matches the no-bias output; BIAS epilogue appears inactive."

    expected = expected_ab + bias.float()
    assert_gemm_close(output, expected, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_flydsl_vs_cpp_regular_bias(M, K, N, layout, dtype):
    """Regular BIAS epilogue: FlyDSL must match the native C++ backend."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    out_features = compute_pytorch_reference(A.float(), B.float(), layout).shape[-1]
    bias = torch.randn(out_features, dtype=dtype, device="cuda")

    flydsl_out, _ = call_gemm_with_bias(
        A,
        B,
        layout,
        out_dtype=dtype,
        bias=bias,
        use_flydsl=True,
    )
    cpp_out, _ = call_gemm_with_bias(
        A,
        B,
        layout,
        out_dtype=dtype,
        bias=bias,
        use_flydsl=False,
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS, ids=FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8_bias(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 GEMM with a fused BIAS epilogue vs PyTorch."""
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

    expected_ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    out_features = expected_ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    output, bias_grad = call_gemm_with_bias(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
        use_flydsl=True,
    )
    assert bias_grad is None

    no_bias_out = call_gemm(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert not torch.allclose(
        output.float(), no_bias_out.float(), atol=1e-4
    ), "FlyDSL FP8 output matches the no-bias output; BIAS epilogue appears inactive."

    expected = expected_ab + bias.float()
    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS, ids=FP8_FORMAT_IDS)
def test_flydsl_vs_cpp_fp8_bias(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 BIAS epilogue: FlyDSL must match the native C++ backend."""
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

    out_features = compute_pytorch_reference(
        A_deq.float(),
        B_deq.float(),
        layout,
    ).shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    flydsl_out, _ = call_gemm_with_bias(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
        use_flydsl=True,
    )
    cpp_out, _ = call_gemm_with_bias(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
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
def test_flydsl_vs_pytorch_mxfp8_bias(M, K, N, layout, fp8_format):
    """MXFP8 forward GEMM with a fused BIAS epilogue vs a PyTorch reference."""
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

    expected_ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    # Bias is a vector along the output-feature axis (last dim of the output).
    out_features = expected_ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    output, bias_grad = call_gemm_with_bias(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
        use_flydsl=True,
    )
    assert bias_grad is None

    # Bias must actually change the result -- guards against the BIAS epilogue
    # silently decaying to DEFAULT and the test passing vacuously.
    no_bias_out = call_gemm(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert not torch.allclose(
        output.float(), no_bias_out.float(), atol=1e-4
    ), "FlyDSL MXFP8 output matches the no-bias output; the BIAS epilogue appears inactive."

    expected = expected_ab + bias.float()
    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_cpp_mxfp8_bias(M, K, N, layout, fp8_format):
    """MXFP8 forward BIAS epilogue: FlyDSL must match the native C++ backend."""
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

    out_features = compute_pytorch_reference(
        A_deq.float(),
        B_deq.float(),
        layout,
    ).shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    flydsl_out, _ = call_gemm_with_bias(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
        use_flydsl=True,
    )
    cpp_out, _ = call_gemm_with_bias(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
        use_flydsl=False,
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=5e-3, rtol=1e-2)


# ==============================================================================
# Fused GELU epilogue coverage
#
# GELU_AUX applies tanh-approx GELU to the C output while saving the
# pre-activation (A@B, or A@B+bias for GELU_AUX_BIAS) to a second aux output
# for the backward pass. general_gemm returns the aux in the third tuple slot
# (gelu_input). Implemented across all FlyDSL backends (regular fp32/fp16/bf16,
# tensor-wise FP8, MXFP8). Each test checks output == gelu(pre_act) and
# aux == pre_act, with a guard that GELU actually changes the output.
# ==============================================================================


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_flydsl_vs_pytorch_regular_gelu(M, K, N, layout, dtype):
    """Regular fp32/fp16/bf16 forward GELU_AUX vs PyTorch."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    pre_act = compute_pytorch_reference(A.float(), B.float(), layout)

    output, gelu_input = call_gemm_with_gelu(
        A,
        B,
        layout,
        out_dtype=dtype,
        use_flydsl=True,
    )
    assert gelu_input is not None, "GELU_AUX did not return the pre-activation aux."

    no_gelu_out = call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL output matches the no-GELU output; GELU epilogue appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=1e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_flydsl_vs_pytorch_regular_gelu_bias(M, K, N, layout, dtype):
    """Regular fp32/fp16/bf16 forward GELU_AUX_BIAS vs PyTorch: bias folded before GELU, aux saved."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    ab = compute_pytorch_reference(A.float(), B.float(), layout)
    out_features = ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")
    pre_act = ab + bias.float()

    output, gelu_input = call_gemm_with_gelu(
        A,
        B,
        layout,
        out_dtype=dtype,
        bias=bias,
        use_flydsl=True,
    )
    assert gelu_input is not None, "GELU_AUX_BIAS did not return the pre-activation aux."

    no_gelu_out = call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL output matches the no-epilogue output; GELU_AUX_BIAS appears inactive."

    # Aux is the post-bias pre-activation (A@B + bias); output is gelu of it.
    assert_gemm_close(gelu_input, pre_act, atol=1e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS, ids=FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8_gelu(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 forward GELU_AUX vs PyTorch."""
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

    pre_act = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    output, gelu_input = call_gemm_with_gelu(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert gelu_input is not None, "GELU_AUX did not return the pre-activation aux."

    no_gelu_out = call_gemm(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL FP8 output matches the no-GELU output; GELU epilogue appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS, ids=FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8_gelu_bias(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 forward GELU_AUX_BIAS vs PyTorch: bias folded before GELU, aux saved."""
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

    ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    out_features = ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")
    pre_act = ab + bias.float()

    output, gelu_input = call_gemm_with_gelu(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
        use_flydsl=True,
    )
    assert gelu_input is not None, "GELU_AUX_BIAS did not return the pre-activation aux."

    no_gelu_out = call_gemm(
        A_fp8,
        B_fp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL FP8 output matches the no-epilogue output; GELU_AUX_BIAS appears inactive."

    # Aux is the post-bias pre-activation (A@B + bias); output is gelu of it.
    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_pytorch_mxfp8_gelu(M, K, N, layout, fp8_format):
    """MXFP8 forward GELU_AUX vs PyTorch: check both output and saved aux."""
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

    pre_act = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    output, gelu_input = call_gemm_with_gelu(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert gelu_input is not None, "GELU_AUX did not return the pre-activation aux."

    # GELU must actually change the result vs the plain matmul output.
    no_gelu_out = call_gemm(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL MXFP8 output matches the no-GELU output; the GELU epilogue appears inactive."

    # Aux is the pre-activation (A@B); output is gelu(A@B).
    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format",
    FP8_FORMAT_COMBOS,
    ids=FP8_FORMAT_IDS,
)
def test_flydsl_vs_pytorch_mxfp8_gelu_bias(M, K, N, layout, fp8_format):
    """MXFP8 forward GELU_AUX_BIAS vs PyTorch: bias folded before GELU, aux saved."""
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

    ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    out_features = ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")
    pre_act = ab + bias.float()

    output, gelu_input = call_gemm_with_gelu(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        bias=bias,
        use_flydsl=True,
    )
    assert gelu_input is not None, "GELU_AUX_BIAS did not return the pre-activation aux."

    no_gelu_out = call_gemm(
        A_mxfp8,
        B_mxfp8,
        layout,
        out_dtype=torch.float32,
        use_flydsl=True,
    )
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL MXFP8 output matches the no-epilogue output; GELU_AUX_BIAS appears inactive."

    # Aux is the post-bias pre-activation (A@B + bias); output is gelu of it.
    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


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

    # general_gemm keeps B's leading dims for a non-transposed B, so the output
    # is (batch, N, batch*M) while the flattened reference is (batch*N, batch*M).
    # The values are the same fully-flattened matmul; reshape to match rank.
    A_flat = A_fp8.dequantize().reshape(-1, K)
    B_flat = B_fp8.dequantize().reshape(-1, K)
    expected = torch.matmul(B_flat, A_flat.T).reshape(output.shape)

    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)
