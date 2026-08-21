# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

"""User-facing GEMM tests for the ROCm Triton and FlyDSL backends.

Both backends are exercised through the same public entry point that TE
``Linear`` / ``LayerNormLinear`` use (``general_gemm()``), selected at runtime
by the ``NVTE_GEMM_BACKEND`` env var:

- ``NVTE_GEMM_BACKEND=TRITON`` runs the ``test_triton_*`` tests.
- ``NVTE_GEMM_BACKEND=FLYDSL`` runs the ``test_flydsl_*`` tests.

Each family self-gates: with a single collection, only the tests matching the
active backend run, so ``ci/pytorch.sh`` invokes this one file twice (once per
backend). Everything else collects-and-skips.

Triton coverage: fp32 / fp16 / bf16 / same-format FP8 / mixed FP8 (skipped for a
compiler bug) / MXFP8 across TN / NN / NT, plus bias / bias-grad epilogues and a
batched-fp8 multidim case. Each Triton test compares against two references:

  1. ``torch.matmul`` on dequantized inputs -- catches functional bugs
     independent of any hipBLASLt behavior.
  2. The C++ ``tex.generic_gemm`` backend under the same TE ``general_gemm``
     surface -- catches divergence from the production path.

Complementary file: ``triton_kernels/test_gemm_kernel.py`` -- low-level
``te_gemm_triton()`` kernel-direct correctness (unchanged, stays put).

FlyDSL coverage mirrors the Triton surface for the currently supported FlyDSL
config: fp32 / fp16 / bf16, same- and mixed-format tensor-wise FP8, same- and
mixed-format MXFP8, TN / NN / NT, batched multidim FP8, the fused forward BIAS
epilogue, and the fused GELU_AUX / GELU_AUX_BIAS epilogues. BGRADB and DGELU are
not implemented on any FlyDSL path yet. FlyDSL kernels require tile-aligned
launch dimensions, so its shapes are aligned to the 256x256x128 kernel contract
rather than reusing the odd-sized Triton edge-mask cases.
"""

import os
import warnings

import pytest
import torch

from transformer_engine.pytorch import Float8Tensor
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from transformer_engine.pytorch import torch_version
import transformer_engine_torch as tex


# --- Backend selection --------------------------------------------------------

_GEMM_BACKEND = os.environ.get("NVTE_GEMM_BACKEND")


def _device_capability():
    """Compute capability as (major, minor), or None on a CPU-only box.

    Deferred behind a helper so importing this module never initialises CUDA at
    collection time (which would error the whole module on a CPU-only runner
    instead of skipping it).
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


_CAP = _device_capability()

# The Triton family runs only under NVTE_GEMM_BACKEND=TRITON; the FlyDSL family
# only under NVTE_GEMM_BACKEND=FLYDSL (and, like the C++ dispatch in
# cpp_extensions/gemm.py, only on gfx950). Outside the selected backend the
# tests would either exercise the wrong path or compare native-vs-native, so
# skip rather than pass vacuously.
requires_triton_backend = pytest.mark.skipif(
    _GEMM_BACKEND != "TRITON",
    reason="Triton GEMM tests require NVTE_GEMM_BACKEND=TRITON",
)

_flydsl_backend_selected = _GEMM_BACKEND == "FLYDSL" and _CAP == (9, 5)
requires_flydsl_backend = pytest.mark.skipif(
    not _flydsl_backend_selected,
    reason="FlyDSL GEMM tests require NVTE_GEMM_BACKEND=FLYDSL on gfx950",
)

# flydsl is only installed when the FlyDSL backend is built in; import it lazily
# so the Triton shard (and CPU-only runners) never force the dependency.
if _flydsl_backend_selected:
    pytest.importorskip("flydsl", reason="FlyDSL package is not installed")

# --- Feature detection --------------------------------------------------------

# MXFP8 requires the fp8-scaled MFMA instructions currently available only on
# gfx950+ (compute capability >= 9.5). Name reflects the capability, not the
# specific arch, so it stays meaningful if future archs also support it.
_has_mxfp8_support = _CAP is not None and _CAP[0] == 9 and _CAP[1] >= 5

requires_mxfp8_support = pytest.mark.skipif(
    not _has_mxfp8_support,
    reason="MXFP8 requires hardware with fp8-scaled MFMA (gfx950+, cc >= 9.5)",
)

# tl.dot_scaled()'s current API (RHS scale in [N, K//32] layout) is only
# available from PyTorch 2.10 onwards (Triton 3.6+).
_torch_ver = torch_version()
requires_torch210 = pytest.mark.skipif(
    _torch_ver < (2, 10),
    reason=(
        "Triton tl.dot_scaled() RHS scale bug fixed in PyTorch 2.10 "
        f"(found {_torch_ver}). The TE kernel uses the new dot_scaled API "
        "(rhs_scale in [N, K//32] layout) which requires PyTorch >= 2.10."
    ),
)


# --- Test parameters ----------------------------------------------------------

# Triton shapes include odd sizes that exercise the kernel's M/N/K edge masks.
TRITON_REGULAR_FP8_SHAPES = [
    (2304, 768, 4096),
    (768, 768, 4096),
    (768, 3072, 4096),
    (229, 541, 541),
    (71, 71, 3571),
    (29, 29, 17389),
]

TRITON_MXFP8_SHAPES = [
    (128, 256, 512),
    (768, 768, 4096),
]

# FlyDSL kernels have no M/N edge masks and specialize K in K128 tiles, so every
# dimension is tile-aligned to exercise the supported production path.
FLYDSL_SHAPES = [
    (512, 512, 512),
    (512, 1024, 512),
    (1024, 512, 1024),
    (512, 512, 1024),  # M != N: exercises the operand-swap contract asymmetrically
]

FLYDSL_MXFP8_SHAPES = [
    (512, 512, 512),
    (512, 1024, 512),
    (512, 512, 1024),  # M != N
]

LAYOUTS = ["TN", "NN", "NT"]

# Same-format FP8 combos, shared by both backends.
FP8_FORMAT_COMBOS = [
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E5M2),
]

# Mixed FP8 formats are disabled on the Triton path due to a Triton compiler bug
# on gfx950: when the MFMA layout is transposed, operand B is packed using A's
# element type, and the instruction format encoding doesn't account for the
# operand swap. This affects both v_mfma_f32_32x32x16_{fp8|bf8} and
# v_mfma_f32_32x32x64_f8f6f4. Fixed upstream in triton-lang/triton PR #9567
# (commit eaaa75cf5, 2026-02-27). Not yet in any pytorch-triton-rocm release as
# of PyTorch 2.11.
# TODO: Re-enable once pytorch-triton-rocm includes the fix (expected 2.12+).
TRITON_FP8_MIXED_FORMAT_COMBOS = [
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E4M3),
]

# FlyDSL supports mixed FP8 formats directly, so it runs the full combo matrix.
FLYDSL_FP8_FORMAT_COMBOS = [
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3),
    (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E4M3),
    (tex.DType.kFloat8E5M2, tex.DType.kFloat8E5M2),
]

FLYDSL_FP8_FORMAT_IDS = [
    "e4m3_e4m3",
    "e4m3_e5m2",
    "e5m2_e4m3",
    "e5m2_e5m2",
]

REGULAR_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# --- Fixtures -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_env():
    """Save/restore GEMM-related env vars between tests.

    The per-call helpers below rewrite ``NVTE_GEMM_BACKEND`` to flip between the
    backend under test and the native C++ reference, and some MXFP8 tests set
    ``NVTE_ROCM_ENABLE_MXFP8``. Restore both so tests do not leak backend
    selection into one another.
    """
    old_backend = os.environ.get("NVTE_GEMM_BACKEND")
    old_mxfp8 = os.environ.get("NVTE_ROCM_ENABLE_MXFP8")

    yield

    if old_backend is None:
        os.environ.pop("NVTE_GEMM_BACKEND", None)
    else:
        os.environ["NVTE_GEMM_BACKEND"] = old_backend

    if old_mxfp8 is None:
        os.environ.pop("NVTE_ROCM_ENABLE_MXFP8", None)
    else:
        os.environ["NVTE_ROCM_ENABLE_MXFP8"] = old_mxfp8


# --- Shared helpers -----------------------------------------------------------


def get_shapes(layout, M, K, N):
    """Return the (A, B) storage shapes for a given TE ``general_gemm`` layout.

    The first / second letter of ``layout`` is transa / transb (``T`` =
    transposed, ``N`` = not). Every layout produces a logical ``(N, M)`` output
    (see ``compute_pytorch_reference``), so each must reference both M and N --
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
    """torch.matmul reference producing the same ``(N, M)`` output shape as
    ``general_gemm`` for the given ``get_shapes()`` operands."""
    if layout == "TN":
        return torch.matmul(B_ref, A_ref.T)
    if layout == "NN":
        return torch.matmul(B_ref, A_ref)
    if layout == "NT":
        return torch.matmul(B_ref.T, A_ref)
    raise ValueError(f"Unsupported layout: {layout}")


def _set_backend(active):
    """Point NVTE_GEMM_BACKEND at the backend under test or the native default.

    ``active`` True selects the current backend (already validated by the
    module-level gate); False selects the native C++ reference by unsetting the
    var, matching the ``use_*=False`` reference path in the original suites.
    """
    if active:
        os.environ["NVTE_GEMM_BACKEND"] = _GEMM_BACKEND
    else:
        os.environ.pop("NVTE_GEMM_BACKEND", None)


# ==============================================================================
# Triton backend (NVTE_GEMM_BACKEND=TRITON)
# ==============================================================================

# --- Triton helpers -----------------------------------------------------------


def create_regular_tensors(M, K, N, layout, dtype=torch.float32):
    """Random (A, B) regular tensors sized for ``layout`` at ``(M, K, N)``.

    Values are ``randn * 0.5`` -- magnitude ~O(1). Used both as-is (regular GEMM
    tests) and as the fp32 input to the Float8 / MXFP8 quantizers.
    """
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5
    return A, B


def create_triton_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b, a_scale=1.0, b_scale=1.0):
    """Create Float8Tensor inputs and dequantized references.

    ``a_scale`` / ``b_scale`` set the per-tensor quantization scale. Non-unity
    values exercise the kernel's ``scale_inv`` fold-back -- with ``scale=1.0`` a
    bug that dropped ``scale_inv`` would still pass since ``1/scale = 1``.
    """
    A_f32, B_f32 = create_regular_tensors(M, K, N, layout, dtype=torch.float32)

    A_fp8 = Float8Quantizer(
        scale=torch.full([1], a_scale, dtype=torch.float32, device="cuda"),
        amax=torch.empty([1], dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype_a,
    )(A_f32)
    B_fp8 = Float8Quantizer(
        scale=torch.full([1], b_scale, dtype=torch.float32, device="cuda"),
        amax=torch.empty([1], dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype_b,
    )(B_f32)

    return A_fp8, B_fp8, A_fp8.dequantize(), B_fp8.dequantize()


def create_triton_mxfp8_tensors(M, K, N, layout):
    """Create MXFP8Tensor inputs and dequantized references."""
    A_f32, B_f32 = create_regular_tensors(M, K, N, layout, dtype=torch.float32)

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    A_mxfp8 = quantizer(A_f32)
    B_mxfp8 = quantizer(B_f32)

    return A_mxfp8, B_mxfp8, A_mxfp8.dequantize(), B_mxfp8.dequantize()


def triton_call_gemm(A, B, layout, out_dtype, use_triton=True):
    """Call general_gemm() with the Triton backend or the native C++ reference."""
    _set_backend(use_triton)
    output, _, _, _ = general_gemm(
        A=A,
        B=B,
        out_dtype=out_dtype,
        layout=layout,
    )
    return output


def triton_call_gemm_with_bias(A, B, layout, out_dtype, bias, grad, use_triton=True):
    """Call general_gemm() with a bias argument.

    Returns (output, bias_grad). When grad=True the GEMM uses the BGRADB
    epilogue and bias_grad contains the reduced bias gradient; otherwise it uses
    the BIAS epilogue and bias is fused into the output.
    """
    _set_backend(use_triton)
    output, bias_grad, _, _ = general_gemm(
        A=A,
        B=B,
        out_dtype=out_dtype,
        layout=layout,
        bias=bias,
        grad=grad,
    )
    return output, bias_grad


# --- Approach 1: Triton vs PyTorch torch.matmul reference ---------------------


@requires_triton_backend
@pytest.mark.parametrize("M, K, N", TRITON_REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_triton_vs_pytorch_regular(M, K, N, layout, dtype):
    """Test Triton GEMM vs torch.matmul for regular tensors."""
    torch.manual_seed(42)
    A, B = create_regular_tensors(M, K, N, layout, dtype=dtype)

    output = triton_call_gemm(A, B, layout, out_dtype=dtype, use_triton=True)
    expected = compute_pytorch_reference(A.float(), B.float(), layout)

    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=1e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@pytest.mark.parametrize("M, K, N", TRITON_REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS, ids=["e4m3_e4m3", "e5m2_e5m2"])
def test_triton_vs_pytorch_fp8(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs torch.matmul for Float8Tensor inputs."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_triton_fp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    output = triton_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=5e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "a_scale, b_scale",
    [(4.0, 0.25), (16.0, 0.125), (0.5, 2.0)],
    ids=["a=4,b=0.25", "a=16,b=0.125", "a=0.5,b=2"],
)
def test_triton_vs_pytorch_fp8_scales(layout, a_scale, b_scale):
    """FP8 GEMM with non-unity per-tensor scales.

    ``test_triton_vs_pytorch_fp8`` above quantizes both operands with
    ``scale=1.0`` -- ``scale_inv=1.0`` -- so a kernel bug that dropped the
    ``accumulator *= a_scale * b_scale`` fold-back would still pass the assert.
    Use asymmetric non-unity scales (product 1 to keep output magnitude
    comparable) so any missing scale multiply produces a systematically wrong
    result. Uses a single shape to keep the parametrize matrix small.
    """
    torch.manual_seed(42)
    M, K, N = 768, 768, 4096
    A_fp8, B_fp8, A_deq, B_deq = create_triton_fp8_tensors(
        M,
        K,
        N,
        layout,
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E4M3,
        a_scale=a_scale,
        b_scale=b_scale,
    )

    output = triton_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=5e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@pytest.mark.skip(reason="Triton compiler bug with mixed FP8 formats (triton-lang/triton#9567)")
@pytest.mark.parametrize("M, K, N", TRITON_REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format", TRITON_FP8_MIXED_FORMAT_COMBOS, ids=["e4m3_e5m2", "e5m2_e4m3"]
)
def test_triton_vs_pytorch_fp8_mixed(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs torch.matmul for mixed Float8Tensor formats."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_triton_fp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    output = triton_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=5e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@requires_mxfp8_support
@requires_torch210
@pytest.mark.parametrize("M, K, N", TRITON_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_triton_vs_pytorch_mxfp8(M, K, N, layout):
    """Test Triton GEMM vs torch.matmul for MXFP8Tensor inputs."""
    torch.manual_seed(42)
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_triton_mxfp8_tensors(M, K, N, layout)

    output = triton_call_gemm(A_mxfp8, B_mxfp8, layout, out_dtype=torch.bfloat16, use_triton=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=5e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@requires_mxfp8_support
@requires_torch210
@pytest.mark.parametrize("M, K, N", TRITON_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "alpha, beta, accumulate",
    [
        (-2.5, 0.0, False),
        (3.5, 0.0, False),
        (1.0, 1.0, True),
        (-2.5, 1.0, True),
    ],
    ids=["alpha=-2.5", "alpha=3.5", "acc-beta=1", "alpha=-2.5,acc-beta=1"],
)
def test_triton_mxfp8_alpha_beta_accumulate(M, K, N, layout, alpha, beta, accumulate):
    """MXFP8 Triton GEMM with the alpha/beta/accumulate epilogue.

    Exercises the fused-op pipeline entry points where
    ``BasicLinear._functional_forward`` folds a scale/add into the GEMM by
    passing ``alpha=<scale>, accumulate_into_out=True``. A prior version of
    ``mxfp8_matmul`` silently dropped these three parameters, so the kernel
    computed ``C = A @ B`` regardless.
    """
    torch.manual_seed(42)
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_triton_mxfp8_tensors(M, K, N, layout)

    out_shape = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout).shape
    d_init = torch.randn(out_shape, dtype=torch.bfloat16, device="cuda") * 0.5

    ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    expected = alpha * ab
    if accumulate:
        expected = expected + beta * d_init.float()

    d = d_init.clone()
    _set_backend(True)
    out, _, _, _ = general_gemm(
        A=A_mxfp8,
        B=B_mxfp8,
        out_dtype=torch.bfloat16,
        layout=layout,
        out=d,
        alpha=alpha,
        beta=beta,
        accumulate=accumulate,
    )

    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=5e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@requires_mxfp8_support
@requires_torch210
@pytest.mark.parametrize("M, K, N", TRITON_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_triton_mxfp8_bias(M, K, N, layout):
    """MXFP8 forward GEMM with the BIAS epilogue.

    ``mxfp8_matmul_kernel`` implements a BIAS path so ``TransformerLayer`` /
    ``te.ops.Linear(bias=True)`` under MXFP8 flow through Triton instead of
    falling back to hipBLASLt (which asserts on MXFP8+bias). Compares the
    fused-bias output against a plain-matmul reference with the same bias added
    post-hoc.
    """
    torch.manual_seed(42)
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_triton_mxfp8_tensors(M, K, N, layout)

    out_features = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout).shape[-1]
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")

    triton_out, _ = triton_call_gemm_with_bias(
        A_mxfp8, B_mxfp8, layout, torch.bfloat16, bias, grad=False, use_triton=True
    )

    no_bias_out = triton_call_gemm(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.bfloat16, use_triton=True
    )
    assert not torch.allclose(
        triton_out.float(), no_bias_out.float(), atol=1e-4
    ), "Triton MXFP8 output matches no-bias output; BIAS epilogue appears inactive."

    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout) + bias.float()
    torch.testing.assert_close(
        triton_out.float(),
        expected.float(),
        atol=5e-3,
        rtol=1e-2,
    )


# --- Approach 2: Triton vs C++ tex.generic_gemm reference ---------------------


@requires_triton_backend
@pytest.mark.parametrize("M, K, N", TRITON_REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_triton_vs_cpp_regular(M, K, N, layout, dtype):
    """Test Triton GEMM vs C++ generic_gemm for regular tensors."""
    torch.manual_seed(42)
    A, B = create_regular_tensors(M, K, N, layout, dtype=dtype)

    triton_out = triton_call_gemm(A, B, layout, out_dtype=dtype, use_triton=True)
    cpp_out = triton_call_gemm(A, B, layout, out_dtype=dtype, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(),
        cpp_out.float(),
        atol=1e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@pytest.mark.parametrize("M, K, N", TRITON_REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FP8_FORMAT_COMBOS, ids=["e4m3_e4m3", "e5m2_e5m2"])
def test_triton_vs_cpp_fp8(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs C++ generic_gemm for Float8Tensor inputs."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, _, _ = create_triton_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    triton_out = triton_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    cpp_out = triton_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(),
        cpp_out.float(),
        atol=5e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@pytest.mark.skip(reason="Triton compiler bug with mixed FP8 formats (triton-lang/triton#9567)")
@pytest.mark.parametrize("M, K, N", TRITON_REGULAR_FP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "fp8_format", TRITON_FP8_MIXED_FORMAT_COMBOS, ids=["e4m3_e5m2", "e5m2_e4m3"]
)
def test_triton_vs_cpp_fp8_mixed(M, K, N, layout, fp8_format):
    """Test Triton GEMM vs C++ generic_gemm for mixed Float8Tensor formats."""
    torch.manual_seed(42)
    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, _, _ = create_triton_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    triton_out = triton_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=True)
    cpp_out = triton_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(),
        cpp_out.float(),
        atol=5e-3,
        rtol=1e-2,
    )


@requires_triton_backend
@requires_mxfp8_support
@requires_torch210
@pytest.mark.parametrize("M, K, N", TRITON_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_triton_vs_cpp_mxfp8(M, K, N, layout):
    """Test Triton GEMM vs C++ generic_gemm for MXFP8Tensor inputs."""
    torch.manual_seed(42)
    A_mxfp8, B_mxfp8, _, _ = create_triton_mxfp8_tensors(M, K, N, layout)

    triton_out = triton_call_gemm(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.bfloat16, use_triton=True
    )
    cpp_out = triton_call_gemm(A_mxfp8, B_mxfp8, layout, out_dtype=torch.bfloat16, use_triton=False)

    torch.testing.assert_close(
        triton_out.float(),
        cpp_out.float(),
        atol=5e-3,
        rtol=1e-2,
    )


# --- Triton bias epilogue coverage --------------------------------------------
#
# The Triton wrapper must honor the `bias` + `grad` arguments to general_gemm:
#   - grad=False + bias present -> BIAS epilogue, bias added to output
#   - grad=True  + bias present -> BGRADB epilogue, bias gradient returned as
#                                  the second element of general_gemm's tuple
# Layout TN matches TE Linear's forward convention: A=weight[M,K],
# B=input[N,K], output[N,M]; BIAS reads bias[M], BGRADB reduces to shape [N].

TRITON_BIAS_SHAPES = [(128, 256, 512), (229, 541, 541), (71, 71, 3571)]


@requires_triton_backend
@pytest.mark.parametrize("M, K, N", TRITON_BIAS_SHAPES)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_triton_vs_cpp_bias_forward(M, K, N, dtype):
    """Forward with BIAS epilogue: Triton must match C++ when bias is fused."""
    torch.manual_seed(42)
    A, B = create_regular_tensors(M, K, N, "TN", dtype=dtype)
    bias = torch.randn((M,), dtype=dtype, device="cuda")

    triton_out, _ = triton_call_gemm_with_bias(A, B, "TN", dtype, bias, grad=False, use_triton=True)
    cpp_out, _ = triton_call_gemm_with_bias(A, B, "TN", dtype, bias, grad=False, use_triton=False)

    no_bias_out = triton_call_gemm(A, B, "TN", out_dtype=dtype, use_triton=True)
    assert not torch.allclose(
        triton_out.float(), no_bias_out.float(), atol=1e-4
    ), "Triton output matches no-bias output; BIAS epilogue appears inactive."

    torch.testing.assert_close(
        triton_out.float(),
        cpp_out.float(),
        atol=5e-3,
        rtol=1e-2,
    )


TRITON_WGRAD_SHAPES = [
    # (batch*seq, in_features, out_features) -- TE Linear wgrad pattern
    (256, 128, 512),
    (512, 541, 229),
    (128, 3571, 71),
]


@requires_triton_backend
@pytest.mark.parametrize("batch, in_features, out_features", TRITON_WGRAD_SHAPES)
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
    A = torch.randn((batch, in_features), dtype=dtype, device="cuda") * 0.5
    B = torch.randn((batch, out_features), dtype=dtype, device="cuda") * 0.5
    bias = torch.zeros((out_features,), dtype=dtype, device="cuda")

    _, triton_bias_grad = triton_call_gemm_with_bias(
        A, B, "NT", dtype, bias, grad=True, use_triton=True
    )
    _, cpp_bias_grad = triton_call_gemm_with_bias(
        A, B, "NT", dtype, bias, grad=True, use_triton=False
    )

    assert triton_bias_grad is not None, "Triton did not return a bias gradient tensor."
    assert cpp_bias_grad is not None, "C++ did not return a bias gradient tensor."
    assert (
        triton_bias_grad.abs().sum().item() > 0
    ), "Triton bias gradient is all zeros -- BGRADB epilogue appears inactive."

    expected = B.float().sum(dim=0)
    torch.testing.assert_close(
        triton_bias_grad.float(),
        expected,
        atol=5e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        triton_bias_grad.float(),
        cpp_bias_grad.float(),
        atol=5e-3,
        rtol=1e-2,
    )


# Batched (multi-dim) FP8 coverage. The 2D fp8 case is already covered by
# test_triton_vs_pytorch_fp8 above; this exercises the backend's
# flatten-leading-dims semantics for tensors with ndim > 2.
@requires_triton_backend
@pytest.mark.parametrize(
    "batch_size, M, K, N",
    [
        (2, 128, 256, 512),
        (4, 64, 128, 256),
    ],
)
def test_triton_vs_pytorch_fp8_multidim(batch_size, M, K, N):
    torch.manual_seed(42)
    # TN layout: A=[batch, M, K], B=[batch, N, K]. Leading dims flatten.
    A_f32 = torch.randn(batch_size, M, K, dtype=torch.float32, device="cuda") * 0.5
    B_f32 = torch.randn(batch_size, N, K, dtype=torch.float32, device="cuda") * 0.5

    A_fp8 = Float8Quantizer(
        scale=torch.full([1], 1.0, dtype=torch.float32, device="cuda"),
        amax=torch.empty([1], dtype=torch.float32, device="cuda"),
        fp8_dtype=tex.DType.kFloat8E4M3,
    )(A_f32)
    B_fp8 = Float8Quantizer(
        scale=torch.full([1], 1.0, dtype=torch.float32, device="cuda"),
        amax=torch.empty([1], dtype=torch.float32, device="cuda"),
        fp8_dtype=tex.DType.kFloat8E4M3,
    )(B_f32)

    output = triton_call_gemm(A_fp8, B_fp8, layout="TN", out_dtype=torch.float32)

    # Reference: flatten leading dims, then B @ A.T (TN semantics).
    A_flat = A_fp8.dequantize().reshape(-1, K)  # [batch*M, K]
    B_flat = B_fp8.dequantize().reshape(-1, K)  # [batch*N, K]
    expected = torch.matmul(B_flat, A_flat.T).reshape(batch_size, N, batch_size * M)

    torch.testing.assert_close(
        output.to(torch.float32),
        expected.to(torch.float32),
        atol=5e-3,
        rtol=1e-2,
    )


# ==============================================================================
# FlyDSL backend (NVTE_GEMM_BACKEND=FLYDSL)
# ==============================================================================

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


# --- FlyDSL helpers -----------------------------------------------------------


def create_flydsl_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b):
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


def _make_flydsl_mxfp8_quantizer(fp8_dtype):
    """Create one independently typed MXFP8 quantizer with both orientations."""
    quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype)
    quantizer.set_usage(rowwise=True, columnwise=True)
    return quantizer


def create_flydsl_mxfp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b):
    """Create independently typed MXFP8Tensor inputs and references."""
    A_shape, B_shape = get_shapes(layout, M, K, N)
    A_f32 = torch.randn(A_shape, dtype=torch.float32, device="cuda") * 0.5
    B_f32 = torch.randn(B_shape, dtype=torch.float32, device="cuda") * 0.5

    A_mxfp8 = _make_flydsl_mxfp8_quantizer(fp8_dtype_a)(A_f32)
    B_mxfp8 = _make_flydsl_mxfp8_quantizer(fp8_dtype_b)(B_f32)

    return A_mxfp8, B_mxfp8, A_mxfp8.dequantize(), B_mxfp8.dequantize()


def _assert_flydsl_ran(use_flydsl, fell_back):
    """Fail if FlyDSL was requested for a supported config but silently fell back.

    Every shape in this suite is tile-aligned and a config the backend supports,
    so a fallback means FlyDSL did not actually run and the comparison below
    would be vacuous (native vs native).
    """
    if use_flydsl and fell_back:
        pytest.fail(
            "FlyDSL GEMM unexpectedly fell back to the native backend; "
            "the FlyDSL path was not exercised."
        )


def flydsl_call_gemm(A, B, layout, out_dtype, use_flydsl=True):
    """Call ``general_gemm`` through either FlyDSL or the native C++ path."""
    _set_backend(use_flydsl)

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


def flydsl_call_gemm_with_bias(A, B, layout, out_dtype, bias, use_flydsl=True):
    """Call ``general_gemm`` with a fused forward BIAS epilogue.

    Bias is a 1-D vector along the output feature axis (the last dim of the
    returned ``(*, out_features)`` tensor) and is added to the matmul result.
    """
    _set_backend(use_flydsl)

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


def flydsl_call_gemm_with_gelu(A, B, layout, out_dtype, bias=None, use_flydsl=True):
    """Call ``general_gemm`` with a fused forward GELU (GELU_AUX) epilogue.

    Returns ``(output, gelu_input)`` where ``output`` is ``gelu(A@B[+bias])``
    and ``gelu_input`` is the saved pre-activation (``A@B[+bias]``) that the
    backward pass consumes.
    """
    _set_backend(use_flydsl)

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


# --- Approach 1: FlyDSL vs PyTorch torch.matmul reference ---------------------


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_flydsl_vs_pytorch_regular(M, K, N, layout, dtype):
    """Test regular FlyDSL GEMM against an FP32 PyTorch reference."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    output = flydsl_call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    expected = compute_pytorch_reference(A.float(), B.float(), layout)

    assert_gemm_close(output, expected, atol=1e-3, rtol=1e-2)


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format tensor-wise FP8 FlyDSL GEMMs."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_flydsl_fp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    output = flydsl_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_flydsl=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", FLYDSL_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_mxfp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format MXFP8 FlyDSL GEMMs."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_flydsl_mxfp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    output = flydsl_call_gemm(A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, use_flydsl=True)
    expected = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
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
    _set_backend(True)
    torch.manual_seed(42)

    M, K, N = 128, 512, 256
    fp8_dtype = tex.DType.kFloat8E4M3
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_flydsl_mxfp8_tensors(
        M, K, N, "TN", fp8_dtype, fp8_dtype
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


# --- Approach 2: FlyDSL vs native C++ ``generic_gemm`` reference --------------


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", REGULAR_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_flydsl_vs_cpp_regular(M, K, N, layout, dtype):
    """Test regular FlyDSL GEMM against the native C++ backend."""
    torch.manual_seed(42)

    A_shape, B_shape = get_shapes(layout, M, K, N)
    A = torch.randn(A_shape, dtype=dtype, device="cuda") * 0.5
    B = torch.randn(B_shape, dtype=dtype, device="cuda") * 0.5

    flydsl_out = flydsl_call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    cpp_out = flydsl_call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=False)

    assert_gemm_close(flydsl_out, cpp_out, atol=1e-3, rtol=1e-2)


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_cpp_fp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format FP8 against native C++."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, _, _ = create_flydsl_fp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    flydsl_out = flydsl_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_flydsl=True)
    cpp_out = flydsl_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_flydsl=False)

    assert_gemm_close(flydsl_out, cpp_out, atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", FLYDSL_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_cpp_mxfp8(M, K, N, layout, fp8_format):
    """Test same-format and mixed-format MXFP8 against native C++."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, _, _ = create_flydsl_mxfp8_tensors(M, K, N, layout, fp8_dtype_a, fp8_dtype_b)

    flydsl_out = flydsl_call_gemm(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, use_flydsl=True
    )
    cpp_out = flydsl_call_gemm(A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, use_flydsl=False)

    assert_gemm_close(flydsl_out, cpp_out, atol=5e-3, rtol=1e-2)


# --- FlyDSL fused BIAS epilogue coverage --------------------------------------
#
# Every FlyDSL GEMM backend (regular fp32/fp16/bf16, tensor-wise FP8, MXFP8)
# adds bias along the output-feature (N) axis -- the last dim of the returned
# ``(M, N)`` tensor -- broadcast across the M/token rows. These guard the full
# public plumbing: general_gemm(bias=...) -> te_generic_gemm_flydsl ->
# _run_<backend> -> <backend>_matmul(epilogue="BIAS").
#
# Each backend has a vs-pytorch test (with a guard that bias actually changes
# the output, catching a silent decay to DEFAULT) and a vs-cpp cross-check.


@requires_flydsl_backend
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

    output, bias_grad = flydsl_call_gemm_with_bias(
        A, B, layout, out_dtype=dtype, bias=bias, use_flydsl=True
    )
    assert bias_grad is None

    no_bias_out = flydsl_call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_bias_out.float(), atol=1e-4
    ), "FlyDSL output matches the no-bias output; BIAS epilogue appears inactive."

    expected = expected_ab + bias.float()
    assert_gemm_close(output, expected, atol=1e-3, rtol=1e-2)


@requires_flydsl_backend
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

    flydsl_out, _ = flydsl_call_gemm_with_bias(
        A, B, layout, out_dtype=dtype, bias=bias, use_flydsl=True
    )
    cpp_out, _ = flydsl_call_gemm_with_bias(
        A, B, layout, out_dtype=dtype, bias=bias, use_flydsl=False
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=1e-3, rtol=1e-2)


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8_bias(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 GEMM with a fused BIAS epilogue vs PyTorch."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_flydsl_fp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    expected_ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    out_features = expected_ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    output, bias_grad = flydsl_call_gemm_with_bias(
        A_fp8, B_fp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=True
    )
    assert bias_grad is None

    no_bias_out = flydsl_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_bias_out.float(), atol=1e-4
    ), "FlyDSL FP8 output matches the no-bias output; BIAS epilogue appears inactive."

    expected = expected_ab + bias.float()
    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_cpp_fp8_bias(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 BIAS epilogue: FlyDSL must match the native C++ backend."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_flydsl_fp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    out_features = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout).shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    flydsl_out, _ = flydsl_call_gemm_with_bias(
        A_fp8, B_fp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=True
    )
    cpp_out, _ = flydsl_call_gemm_with_bias(
        A_fp8, B_fp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=False
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", FLYDSL_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_mxfp8_bias(M, K, N, layout, fp8_format):
    """MXFP8 forward GEMM with a fused BIAS epilogue vs a PyTorch reference."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_flydsl_mxfp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    expected_ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    out_features = expected_ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    output, bias_grad = flydsl_call_gemm_with_bias(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=True
    )
    assert bias_grad is None

    no_bias_out = flydsl_call_gemm(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, use_flydsl=True
    )
    assert not torch.allclose(
        output.float(), no_bias_out.float(), atol=1e-4
    ), "FlyDSL MXFP8 output matches the no-bias output; the BIAS epilogue appears inactive."

    expected = expected_ab + bias.float()
    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", FLYDSL_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_cpp_mxfp8_bias(M, K, N, layout, fp8_format):
    """MXFP8 forward BIAS epilogue: FlyDSL must match the native C++ backend."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_flydsl_mxfp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    out_features = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout).shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")

    flydsl_out, _ = flydsl_call_gemm_with_bias(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=True
    )
    cpp_out, _ = flydsl_call_gemm_with_bias(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=False
    )

    assert_gemm_close(flydsl_out, cpp_out, atol=5e-3, rtol=1e-2)


# --- FlyDSL fused GELU epilogue coverage --------------------------------------
#
# GELU_AUX applies tanh-approx GELU to the C output while saving the
# pre-activation (A@B, or A@B+bias for GELU_AUX_BIAS) to a second aux output
# for the backward pass. general_gemm returns the aux in the third tuple slot
# (gelu_input). Implemented across all FlyDSL backends (regular fp32/fp16/bf16,
# tensor-wise FP8, MXFP8). Each test checks output == gelu(pre_act) and
# aux == pre_act, with a guard that GELU actually changes the output.


@requires_flydsl_backend
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

    output, gelu_input = flydsl_call_gemm_with_gelu(A, B, layout, out_dtype=dtype, use_flydsl=True)
    assert gelu_input is not None, "GELU_AUX did not return the pre-activation aux."

    no_gelu_out = flydsl_call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL output matches the no-GELU output; GELU epilogue appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=1e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=1e-3, rtol=1e-2)


@requires_flydsl_backend
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

    output, gelu_input = flydsl_call_gemm_with_gelu(
        A, B, layout, out_dtype=dtype, bias=bias, use_flydsl=True
    )
    assert gelu_input is not None, "GELU_AUX_BIAS did not return the pre-activation aux."

    no_gelu_out = flydsl_call_gemm(A, B, layout, out_dtype=dtype, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL output matches the no-epilogue output; GELU_AUX_BIAS appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=1e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=1e-3, rtol=1e-2)


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8_gelu(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 forward GELU_AUX vs PyTorch."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_flydsl_fp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    pre_act = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    output, gelu_input = flydsl_call_gemm_with_gelu(
        A_fp8, B_fp8, layout, out_dtype=torch.float32, use_flydsl=True
    )
    assert gelu_input is not None, "GELU_AUX did not return the pre-activation aux."

    no_gelu_out = flydsl_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL FP8 output matches the no-GELU output; GELU epilogue appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@pytest.mark.parametrize("M, K, N", FLYDSL_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8_gelu_bias(M, K, N, layout, fp8_format):
    """Tensor-wise FP8 forward GELU_AUX_BIAS vs PyTorch: bias folded before GELU, aux saved."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_fp8, B_fp8, A_deq, B_deq = create_flydsl_fp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    out_features = ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")
    pre_act = ab + bias.float()

    output, gelu_input = flydsl_call_gemm_with_gelu(
        A_fp8, B_fp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=True
    )
    assert gelu_input is not None, "GELU_AUX_BIAS did not return the pre-activation aux."

    no_gelu_out = flydsl_call_gemm(A_fp8, B_fp8, layout, out_dtype=torch.float32, use_flydsl=True)
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL FP8 output matches the no-epilogue output; GELU_AUX_BIAS appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", FLYDSL_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_mxfp8_gelu(M, K, N, layout, fp8_format):
    """MXFP8 forward GELU_AUX vs PyTorch: check both output and saved aux."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_flydsl_mxfp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    pre_act = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)

    output, gelu_input = flydsl_call_gemm_with_gelu(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, use_flydsl=True
    )
    assert gelu_input is not None, "GELU_AUX did not return the pre-activation aux."

    no_gelu_out = flydsl_call_gemm(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, use_flydsl=True
    )
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL MXFP8 output matches the no-GELU output; the GELU epilogue appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


@requires_flydsl_backend
@requires_mxfp8_support
@pytest.mark.parametrize("M, K, N", FLYDSL_MXFP8_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_mxfp8_gelu_bias(M, K, N, layout, fp8_format):
    """MXFP8 forward GELU_AUX_BIAS vs PyTorch: bias folded before GELU, aux saved."""
    os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format
    A_mxfp8, B_mxfp8, A_deq, B_deq = create_flydsl_mxfp8_tensors(
        M, K, N, layout, fp8_dtype_a, fp8_dtype_b
    )

    ab = compute_pytorch_reference(A_deq.float(), B_deq.float(), layout)
    out_features = ab.shape[-1]
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda")
    pre_act = ab + bias.float()

    output, gelu_input = flydsl_call_gemm_with_gelu(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, bias=bias, use_flydsl=True
    )
    assert gelu_input is not None, "GELU_AUX_BIAS did not return the pre-activation aux."

    no_gelu_out = flydsl_call_gemm(
        A_mxfp8, B_mxfp8, layout, out_dtype=torch.float32, use_flydsl=True
    )
    assert not torch.allclose(
        output.float(), no_gelu_out.float(), atol=1e-4
    ), "FlyDSL MXFP8 output matches the no-epilogue output; GELU_AUX_BIAS appears inactive."

    assert_gemm_close(gelu_input, pre_act, atol=5e-3, rtol=1e-2)
    assert_gemm_close(output, gelu_tanh_ref(pre_act), atol=5e-3, rtol=1e-2)


# --- FlyDSL batched multidimensional FP8 coverage -----------------------------


@requires_flydsl_backend
@pytest.mark.parametrize(
    "batch_size, M, K, N",
    [
        (2, 256, 512, 256),
        (4, 256, 512, 256),
    ],
)
@pytest.mark.parametrize("fp8_format", FLYDSL_FP8_FORMAT_COMBOS, ids=FLYDSL_FP8_FORMAT_IDS)
def test_flydsl_vs_pytorch_fp8_multidim(batch_size, M, K, N, fp8_format):
    """Exercise flatten-leading-dim semantics for multidimensional FP8."""
    torch.manual_seed(42)

    fp8_dtype_a, fp8_dtype_b = fp8_format

    # TN layout: the wrapper flattens all leading dimensions into rows.
    A_f32 = torch.randn(batch_size, M, K, dtype=torch.float32, device="cuda") * 0.5
    B_f32 = torch.randn(batch_size, N, K, dtype=torch.float32, device="cuda") * 0.5

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

    output = flydsl_call_gemm(A_fp8, B_fp8, layout="TN", out_dtype=torch.float32, use_flydsl=True)

    # general_gemm keeps B's leading dims for a non-transposed B, so the output
    # is (batch, N, batch*M) while the flattened reference is (batch*N, batch*M).
    # The values are the same fully-flattened matmul; reshape to match rank.
    A_flat = A_fp8.dequantize().reshape(-1, K)
    B_flat = B_fp8.dequantize().reshape(-1, K)
    expected = torch.matmul(B_flat, A_flat.T).reshape(output.shape)

    assert_gemm_close(output, expected, atol=5e-3, rtol=1e-2)
