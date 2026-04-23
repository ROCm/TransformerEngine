# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GEMM operations -- multi-backend with AITER, Triton, and PyTorch fallback.

Backend priority (configurable via NVTE_LITE_GEMM_BACKEND env var):
1. AITER CK GEMM (default) -- CK/ASM kernels for FP8 precisions
2. AITER Triton GEMM -- dedicated Triton kernels for FP8 and BF16/FP16
3. PyTorch fallback -- torch._scaled_mm for FP8 (hipBLASLt-backed on ROCm),
   dequantize + torch.matmul otherwise

Set NVTE_LITE_GEMM_BACKEND to override:
  "ck"      -- prefer AITER CK kernels (default)
  "triton"  -- prefer AITER Triton GEMM kernels
  "pytorch" -- prefer torch._scaled_mm for FP8 (hipBLASLt-backed on ROCm);
               fall back to AITER for FP8 cases _scaled_mm can't serve
               (wgrad with per-row scale on reduction axis, block scaled,
               unsupported dtype combos); dequantize + torch.matmul as a
               last resort for non-FP8 or when AITER is unavailable.
"""

import os
import torch
import torch.nn.functional as F

from .aiter_utils import is_aiter_available, get_aiter

# FP8 dtypes for detection
_FP8_DTYPES = (
    torch.float8_e4m3fn, torch.float8_e5m2,
    torch.float8_e4m3fnuz, torch.float8_e5m2fnuz,
)

_GEMM_BACKEND = os.environ.get("NVTE_LITE_GEMM_BACKEND", "ck").lower()

_LITE_DIAG = os.environ.get("NVTE_LITE_DIAG", "0") != "0"

from collections import Counter as _GemmCounter
_GEMM_CALLS = _GemmCounter()
_GEMM_BACKEND_PRINTED = False
_CK_FAIL_DIAG_PRINTS = 0
_SCALED_MM_FAIL_DIAG_PRINTS = 0
_SCALED_MM_FAIL_DIAG_MAX = 5


def _log_scaled_mm_fail(reason, A, transA, B, transB, x=None, w=None,
                        x_scale=None, w_scale=None, M=None, N=None,
                        effective_transA=None, effective_transB=None, err=None):
    """Log the first _SCALED_MM_FAIL_DIAG_MAX rejections from _try_scaled_mm.

    Gated by NVTE_LITE_DIAG. Captures shapes, dtypes, scale layout, and the
    transpose-only state of the operands so we can classify the fallthrough
    pattern (per-row on reduction axis vs shape mismatch vs library reject).
    """
    if not _LITE_DIAG:
        return
    global _SCALED_MM_FAIL_DIAG_PRINTS
    if _SCALED_MM_FAIL_DIAG_PRINTS >= _SCALED_MM_FAIL_DIAG_MAX:
        return
    _SCALED_MM_FAIL_DIAG_PRINTS += 1

    def _fmt_scale(s):
        if s is None:
            return "None"
        return f"shape={tuple(s.shape)} numel={s.numel()} dtype={s.dtype}"

    def _fmt_operand(t, name):
        if t is None:
            return f"{name}=None"
        trans_only = _is_transpose_only(t)
        return (f"{name}: shape={tuple(t.shape)} "
                f"dtype={getattr(t, 'dtype', '?')} "
                f"transpose_only={trans_only}")

    bits = [
        f"[LITE-SCALED-MM-FAIL #{_SCALED_MM_FAIL_DIAG_PRINTS}] reason={reason}",
        _fmt_operand(A, "A") + f" transA={transA}",
        _fmt_operand(B, "B") + f" transB={transB}",
    ]
    if effective_transA is not None or effective_transB is not None:
        bits.append(
            f"eff_transA={effective_transA} eff_transB={effective_transB}"
        )
    if x is not None:
        bits.append(
            f"x: shape={tuple(x.shape)} dtype={x.dtype} "
            f"stride_last={x.stride(-1)}"
        )
    if w is not None:
        bits.append(
            f"w: shape={tuple(w.shape)} dtype={w.dtype} "
            f"stride_last={w.stride(-1)}"
        )
    bits.append(f"x_scale: {_fmt_scale(x_scale)}")
    bits.append(f"w_scale: {_fmt_scale(w_scale)}")
    if M is not None or N is not None:
        bits.append(f"M={M} N={N}")
    if err is not None:
        msg = str(err)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        bits.append(f"err={type(err).__name__}: {msg}")
    print(" | ".join(bits), flush=True)

def _gemm_bump(tag):
    if not _LITE_DIAG:
        return
    global _GEMM_BACKEND_PRINTED
    if not _GEMM_BACKEND_PRINTED:
        _GEMM_BACKEND_PRINTED = True
        print(f"[LITE-GEMM-BACKEND] {_GEMM_BACKEND}", flush=True)
    _GEMM_CALLS[tag] += 1
    if sum(_GEMM_CALLS.values()) % 500 == 0:
        print(f"[LITE-GEMM] {dict(_GEMM_CALLS)}", flush=True)


def _resolve_output_dtype(output_dtype):
    """Normalize output_dtype (TE_DType | torch.dtype | None) to torch.dtype.

    `cpp_extensions/gemm.py` forwards the user-provided `out_dtype` as a
    `TE_DType` enum. The pure-Python path needs a `torch.dtype` to cast the
    result; the full build resolves this inside cuBLAS. Returns None when the
    caller did not specify an output dtype (the full build uses the D operand
    dtype in that case).
    """
    if output_dtype is None or isinstance(output_dtype, torch.dtype):
        return output_dtype
    try:
        from transformer_engine.pytorch.triton_kernels.common import (
            te_dtype_to_torch_dtype,
        )
        return te_dtype_to_torch_dtype(output_dtype)
    except (ImportError, KeyError):
        return None


def _dequantize_from_transpose(tensor):
    """Dequantize a Float8Tensor when only its _transpose is available.

    Columnwise-only tensors (wgrad path) have _data=None; the standard
    dequantize() raises NotImplementedError. We dequantize from _transpose
    manually: reinterpret uint8 as FP8 dtype, transpose back to logical
    shape, and multiply by the per-row or per-tensor scale.

    Transpose is done on the uint8 view (fast byte-level copy) before
    reinterpreting as FP8, so the materialization doesn't go through the
    slower float8_copy_kernel_cuda path.
    """
    t = tensor._transpose
    u8 = t if t.dtype == torch.uint8 else t.view(torch.uint8)
    # _transpose is [K, d0, d1, ...] (last dim moved to front); invert to
    # the logical [d0, d1, ..., K] on uint8, then view as FP8 and cast once.
    if u8.ndim == 2:
        u8_logical = u8.t().contiguous()
    else:
        inv_perm = list(range(1, u8.ndim)) + [0]
        u8_logical = u8.permute(*inv_perm).contiguous()
    if hasattr(tensor, '_fp8_dtype'):
        from transformer_engine.pytorch._lite.quantize import _te_dtype_to_torch_fp8
        fp8_logical = u8_logical.view(_te_dtype_to_torch_fp8(tensor._fp8_dtype))
    else:
        fp8_logical = u8_logical
    logical = fp8_logical.to(torch.bfloat16)
    scale_inv = tensor._scale_inv
    if scale_inv.numel() == 1:
        return logical * scale_inv
    # Per-row scale shape (M_flat,); reshape to match logical's leading dims
    leading_numel = 1
    for d in logical.shape[:-1]:
        leading_numel *= d
    if scale_inv.numel() == leading_numel:
        return logical * scale_inv.reshape(*logical.shape[:-1], 1)
    return logical * scale_inv.reshape(-1, 1)


def _dequantize_if_needed(tensor):
    """Dequantize FP8/quantized tensor to BF16 for matmul."""
    if _is_mxfp8(tensor):
        return tensor.dequantize(dtype=torch.bfloat16)
    if _is_blockwise_fp8(tensor):
        return tensor.dequantize(dtype=torch.bfloat16)
    # Columnwise-only Float8Tensor: _data deleted, must dequantize from _transpose
    if (hasattr(tensor, '_data') and tensor._data is None
            and hasattr(tensor, '_transpose') and tensor._transpose is not None):
        return _dequantize_from_transpose(tensor)
    if hasattr(tensor, 'dequantize'):
        return tensor.dequantize()
    if isinstance(tensor, torch.Tensor) and tensor.dtype in _FP8_DTYPES:
        return tensor.to(torch.bfloat16)
    return tensor


def _is_quantized(tensor):
    """Check if tensor is a quantized type with FP8 data."""
    if hasattr(tensor, '_data') and hasattr(tensor, '_scale_inv'):
        return True
    if _is_mxfp8(tensor):
        return True
    return False


def _get_raw_data(tensor):
    """Extract raw data and scale from a quantized tensor, or return tensor as-is."""
    if _is_blockwise_fp8(tensor):
        data, scale = _get_blockwise_data(tensor, need_rowwise=True)
        return data, scale
    if _is_mxfp8(tensor):
        # MXFP8 scales are E8M0 uint8 — not directly usable as float scales
        # for AITER GEMM dispatch. Return data only; GEMM will dequantize.
        return tensor._rowwise_data, None
    if hasattr(tensor, '_data') and hasattr(tensor, '_scale_inv'):
        data = tensor._data
        if data is None:
            # Columnwise-only tensor: _data was deleted by update_usage.
            # Use transpose if available, otherwise dequantize.
            if hasattr(tensor, '_transpose') and tensor._transpose is not None:
                data = tensor._transpose
            else:
                return tensor, None
        # Float8Tensor stores FP8 bit patterns as uint8 — reinterpret as the
        # actual FP8 dtype so downstream Triton kernels see the correct type.
        if data.dtype == torch.uint8 and hasattr(tensor, '_fp8_dtype'):
            from transformer_engine.pytorch._lite.quantize import _te_dtype_to_torch_fp8
            data = data.view(_te_dtype_to_torch_fp8(tensor._fp8_dtype))
        return data, tensor._scale_inv
    return tensor, None


def _is_transpose_only(tensor):
    """Return True if tensor has _data=None but _transpose set (columnwise-only)."""
    return (hasattr(tensor, '_data') and tensor._data is None
            and hasattr(tensor, '_transpose') and tensor._transpose is not None)


def _fp8_transposed_operand(tensor, data_2d):
    """Return the transposed 2D FP8 operand for a GEMM, preferring the tensor's
    _transpose cache to avoid materializing a fresh transposed copy.

    data_2d is the (already-flattened) rowwise [M, K] FP8 view of tensor._data.
    If tensor._transpose is populated and valid, we reshape it to [K, M] and
    return — the byte layout already matches what data_2d.t().contiguous()
    would produce, at zero copy cost.

    When no transpose cache is available, we transpose via the uint8 view
    instead of the fp8 view. Same number of bytes copied, but dispatches to
    the plain uint8 copy kernel rather than the slow float8_copy_kernel_cuda
    that .t().contiguous() on an FP8-dtype tensor hits.
    """
    data_buf = getattr(tensor, '_data', None)
    trans = getattr(tensor, '_transpose', None)
    trans_invalid = getattr(tensor, '_transpose_invalid', True)
    # Only use the cache when data_2d came from _data (i.e. the tensor has
    # both buffers). If _data is None, data_2d came from _transpose already
    # and we actually need to undo that layout via an explicit copy.
    can_use_cache = (
        data_buf is not None and trans is not None and not trans_invalid
    )
    fp8_dtype = data_2d.dtype
    if can_use_cache:
        t = trans
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.dtype == torch.uint8:
            t = t.view(fp8_dtype)
        return t
    # Fallback: uint8-level transpose to avoid float8_copy_kernel_cuda.
    d = data_2d
    if d.dtype != torch.uint8:
        d = d.view(torch.uint8)
    return d.t().contiguous().view(fp8_dtype)


# ---------------------------------------------------------------------------
# AITER CK GEMM dispatch
# ---------------------------------------------------------------------------

def _is_per_row_scaled(scale):
    """Check if scale tensor is per-row (one scale per token/row).

    Per-row scales have shape (M,) or (M, 1) — 1D with numel > 1.
    Block scales are 2D with shape (ceil(M/block), ceil(N/block)).
    """
    return (scale is not None
            and scale.numel() > 1
            and scale.ndim == 1)


def _is_block_scaled(scale):
    """Check if scale tensor indicates block scaling (2D multi-element scale).

    Excludes per-row scales (1D) — those use gemm_a8w8_per_token_scale.
    """
    return (scale is not None
            and scale.numel() > 1
            and not _is_per_row_scaled(scale))


def _is_fp4(tensor):
    """Check if tensor is MXFP4 quantized.

    Discriminates from MXFP8 via _fp4_dtype (MXFP4) vs _fp8_dtype (MXFP8).
    """
    return (hasattr(tensor, '_rowwise_data') and
            hasattr(tensor, '_fp4_dtype') and
            not hasattr(tensor, '_is_2D_scaled') and  # exclude Float8Blockwise
            tensor._rowwise_data is not None)


def _is_mxfp8(tensor):
    """Check if tensor is MXFP8 quantized (block-scaled FP8, group_size=32).

    MXFP8 uses _rowwise_data/_rowwise_scale_inv (shared attribute names with
    MXFP4), distinguished by _fp8_dtype. No AITER GEMM kernel exists on MI300X;
    future MI350 kernel hook is in _aiter_ck_gemm/_aiter_triton_gemm.
    """
    return (hasattr(tensor, '_rowwise_data') and
            hasattr(tensor, '_fp8_dtype') and
            not hasattr(tensor, '_is_2D_scaled') and  # exclude Float8Blockwise
            not hasattr(tensor, '_data') and           # exclude Float8Tensor
            tensor._rowwise_data is not None)


def _get_fp4_data(tensor):
    """Extract FP4 data and scale from MXFP4 tensor."""
    return tensor._rowwise_data, tensor._rowwise_scale_inv


def _is_blockwise_fp8(tensor):
    """Check if tensor is Float8BlockwiseQTensorStorage (2D block-scaled FP8)."""
    return hasattr(tensor, '_is_2D_scaled') and hasattr(tensor, '_data_format')


def _get_blockwise_data(tensor, need_rowwise=True):
    """Extract data and scale from Float8BlockwiseQTensorStorage.

    Returns (data, scale_inv) for the requested orientation.
    For GEMM: A (weight) typically needs columnwise, B (activation) needs rowwise.
    """
    if need_rowwise and tensor._rowwise_data is not None:
        return tensor._rowwise_data, tensor._rowwise_scale_inv
    if not need_rowwise and tensor._columnwise_data is not None:
        return tensor._columnwise_data, tensor._columnwise_scale_inv
    # Fall back to whatever is available
    if tensor._rowwise_data is not None:
        return tensor._rowwise_data, tensor._rowwise_scale_inv
    return tensor._columnwise_data, tensor._columnwise_scale_inv


def _reshape_scale_for_scaled_mm(scale, dim, is_row):
    """Reshape a Float8 _scale_inv for torch._scaled_mm.

    - Per-tensor scalar (numel==1): return a 0-dim tensor. hipBLASLt's
      per-tensor FP8 kernels (same family full TE uses for DelayedScaling)
      are selected by scalar scale shape. Broadcasting a scalar to (dim, 1)
      would force the rowwise kernel family, which isn't tuned for
      mixed-dtype (E4M3×E5M2) on ROCm — that's the "could not find valid
      hipblaslt solution" error for dgrad calls.
    - Per-row (numel==dim): return `(dim, 1)` (is_row=True) or `(1, dim)`
      (is_row=False), the rowwise convention.
    - Anything else: None (caller falls through).
    """
    if scale is None:
        return None
    scale = scale.to(torch.float32) if scale.dtype != torch.float32 else scale
    if scale.numel() == 1:
        return scale.reshape(())
    if scale.numel() == dim:
        flat = scale.reshape(dim).contiguous()
        return flat.unsqueeze(1) if is_row else flat.unsqueeze(0)
    return None


def _try_scaled_mm(A, transA, B, transB, output_dtype):
    """FP8×FP8 GEMM via torch._scaled_mm (hipBLASLt-backed on ROCm).

    Matches AITER's NT convention: x=[M,K] (rowwise), w=[N,K] (rowwise),
    compute x @ w.T. Uses the same `_fp8_transposed_operand` path that
    feeds AITER Triton, so operands are K-innermost by construction.

    Returns the result tensor (with original leading B dims restored), or
    None when torch._scaled_mm is unavailable, block-scaled (not supported
    here), or rejects the inputs (any RuntimeError falls through).
    """
    if not hasattr(torch, '_scaled_mm'):
        return None

    # Block-scaled uses a different scale layout — fall through.
    if _is_blockwise_fp8(A) or _is_blockwise_fp8(B):
        _log_scaled_mm_fail("blockwise_fp8", A, transA, B, transB)
        return None

    a_data, a_scale = _get_raw_data(A)
    b_data, b_scale = _get_raw_data(B)

    # Resolve NT operand form, same logic as _aiter_triton_gemm.
    a_transpose_only = _is_transpose_only(A)
    b_transpose_only = _is_transpose_only(B)
    effective_transA = transA ^ a_transpose_only
    effective_transB = transB ^ b_transpose_only

    x_leading = b_data.shape[:-1] if not b_transpose_only else b_data.shape[1:]
    if b_data.ndim > 2:
        if b_transpose_only:
            b_data = b_data.reshape(b_data.shape[0], -1)
        else:
            b_data = b_data.reshape(-1, b_data.shape[-1])
    if a_data.ndim > 2:
        if a_transpose_only:
            a_data = a_data.reshape(a_data.shape[0], -1)
        else:
            a_data = a_data.reshape(-1, a_data.shape[-1])

    x = b_data if not effective_transB else _fp8_transposed_operand(B, b_data)
    w = a_data if effective_transA else _fp8_transposed_operand(A, a_data)
    x_scale = b_scale
    w_scale = a_scale

    # Per-row on the REDUCTION axis (wgrad corner) is not supported by
    # per-row scaled GEMM kernels — fall through to dequant path.
    M = x.shape[0]
    N = w.shape[0]
    if _is_per_row_scaled(x_scale) and x_scale.numel() != M:
        _log_scaled_mm_fail("per_row_on_reduction_x", A, transA, B, transB,
                            x=x, w=w, x_scale=x_scale, w_scale=w_scale,
                            M=M, N=N,
                            effective_transA=effective_transA,
                            effective_transB=effective_transB)
        return None
    if _is_per_row_scaled(w_scale) and w_scale.numel() != N:
        _log_scaled_mm_fail("per_row_on_reduction_w", A, transA, B, transB,
                            x=x, w=w, x_scale=x_scale, w_scale=w_scale,
                            M=M, N=N,
                            effective_transA=effective_transA,
                            effective_transB=effective_transB)
        return None

    x_scale_2d = _reshape_scale_for_scaled_mm(x_scale, M, is_row=True)
    w_scale_2d = _reshape_scale_for_scaled_mm(w_scale, N, is_row=False)
    if x_scale_2d is None or w_scale_2d is None:
        _log_scaled_mm_fail("scale_shape_mismatch", A, transA, B, transB,
                            x=x, w=w, x_scale=x_scale, w_scale=w_scale,
                            M=M, N=N,
                            effective_transA=effective_transA,
                            effective_transB=effective_transB)
        return None

    # hipBLASLt FP8 kernels require mat1's dims divisible by 16. When the
    # tokens count isn't a clean multiple (e.g. 8184 = 2046×4), we pad the
    # M axis of x (and its per-row scale) with zeros/ones up to the next
    # multiple, run the GEMM, and slice the result back. K-dim misalignment
    # hits in the wgrad corner (K = tokens after transpose) — that case also
    # has separate per-row-on-reduction issues, so we skip it here and let
    # the caller fall through to AITER.
    K = x.shape[1]
    if K % 16 != 0:
        _log_scaled_mm_fail("k_not_div16", A, transA, B, transB,
                            x=x, w=w, x_scale=x_scale_2d, w_scale=w_scale_2d,
                            M=M, N=N,
                            effective_transA=effective_transA,
                            effective_transB=effective_transB)
        return None

    pad_rows = (-M) % 16
    if pad_rows:
        x = F.pad(x, (0, 0, 0, pad_rows))  # zero-pad new rows
        # Only pad x_scale_2d if it's per-row (shape (M, 1)); a 0-dim
        # scalar per-tensor scale applies to every row automatically.
        if x_scale_2d.ndim == 2 and x_scale_2d.shape[0] == M:
            # Value irrelevant (scale × 0 = 0), just non-NaN/Inf.
            x_scale_2d = F.pad(x_scale_2d, (0, 0, 0, pad_rows), value=1.0)

    out_dtype = output_dtype if output_dtype is not None else torch.bfloat16

    try:
        result = torch._scaled_mm(
            x, w.t(),
            scale_a=x_scale_2d, scale_b=w_scale_2d,
            out_dtype=out_dtype,
        )
    except (RuntimeError, TypeError) as _sm_err:
        _log_scaled_mm_fail("torch._scaled_mm_raised", A, transA, B, transB,
                            x=x, w=w, x_scale=x_scale_2d, w_scale=w_scale_2d,
                            M=M, N=N,
                            effective_transA=effective_transA,
                            effective_transB=effective_transB,
                            err=_sm_err)
        return None

    if pad_rows:
        result = result[:M]

    if len(x_leading) > 1:
        result = result.reshape(*x_leading, result.shape[-1])
    return result


def _aiter_ck_gemm(aiter, a_data, a_scale, b_data, b_scale,
                   a_is_fp8, b_is_fp8, transA, transB,
                   A, B):
    """Dispatch to AITER CK/ASM kernels. Returns result tensor or None."""
    _gemm_bump("ck_enter")
    try:
        # MXFP8: No hardware GEMM on MI300X. Fall through to dequant path.
        # TODO(MI350): Add aiter.gemm_mxfp8() dispatch when available.
        if _is_mxfp8(A) or _is_mxfp8(B):
            _gemm_bump("ck_skip_mxfp8")
            return None

        # FP4 × FP4
        if _is_fp4(A) and _is_fp4(B):
            if hasattr(aiter, 'gemm_a4w4'):
                a4_data, a4_scale = _get_fp4_data(A)
                b4_data, b4_scale = _get_fp4_data(B)
                M, _ = a4_data.shape
                N, _ = b4_data.shape
                out = torch.empty(M, N, dtype=torch.bfloat16, device=a4_data.device)
                return aiter.gemm_a4w4(a4_data, b4_data, a4_scale, b4_scale, out)

        # Float8Blockwise (2D block-scaled, 128×128 blocks) — always block-scaled
        a_is_blockwise = _is_blockwise_fp8(A)
        b_is_blockwise = _is_blockwise_fp8(B)

        # FP8 × FP8
        if a_is_fp8 and b_is_fp8:
            # Determine layout: Y = X @ W^T
            # TE: result = B @ A. transA=True means A is (N,K) weight layout.
            # When _get_raw_data returned the transpose (columnwise-only
            # tensor, e.g. wgrad inputmat), orientation is already flipped.
            a_transpose_only = _is_transpose_only(A)
            b_transpose_only = _is_transpose_only(B)
            effective_transA = transA ^ a_transpose_only
            effective_transB = transB ^ b_transpose_only

            # CK FP8 kernels require 2D; flatten N-D leading dims first so
            # subsequent .t() works and scales (which are per flattened row)
            # stay aligned. _data is [d0, ..., K]; _transpose is [K, d0, ...].
            x_leading_shape = b_data.shape[:-1] if not b_transpose_only else b_data.shape[1:]
            if b_data.ndim > 2:
                if b_transpose_only:
                    b_data = b_data.reshape(b_data.shape[0], -1)
                else:
                    b_data = b_data.reshape(-1, b_data.shape[-1])
            if a_data.ndim > 2:
                if a_transpose_only:
                    a_data = a_data.reshape(a_data.shape[0], -1)
                else:
                    a_data = a_data.reshape(-1, a_data.shape[-1])

            if b_is_blockwise:
                x, x_scale = _get_blockwise_data(B, need_rowwise=not transB)
            else:
                x = b_data if not effective_transB else _fp8_transposed_operand(B, b_data)
                x_scale = b_scale

            if a_is_blockwise:
                w, w_scale = _get_blockwise_data(A, need_rowwise=transA)
            else:
                w = a_data if effective_transA else _fp8_transposed_operand(A, a_data)
                w_scale = a_scale

            if (_is_block_scaled(x_scale) or _is_block_scaled(w_scale)
                    or a_is_blockwise or b_is_blockwise):
                # Block-scale FP8 (includes Float8Blockwise)
                if hasattr(aiter, 'gemm_a8w8_blockscale'):
                    _gemm_bump("ck_blockscale")
                    result = aiter.gemm_a8w8_blockscale(x, w, x_scale, w_scale)
                    if len(x_leading_shape) > 1:
                        result = result.reshape(*x_leading_shape, result.shape[-1])
                    return result
            else:
                # Per-tensor or per-row FP8. CK's RowwiseScale kernel accepts
                # x_scale (M, 1) and w_scale (1, N) — a scalar broadcasts to
                # fill, a per-row vector reshapes in place. Per-row scales on
                # the reduction axis (wgrad edge case — scales came from the
                # non-transposed tensor) can't use CK; fall through to Triton.
                M = x.shape[0]
                N = w.shape[0]
                x_per_row = x_scale.numel() > 1
                w_per_row = w_scale.numel() > 1
                x_ok = (not x_per_row) or (x_scale.numel() == M)
                w_ok = (not w_per_row) or (w_scale.numel() == N)
                if x_ok and w_ok and hasattr(aiter, 'gemm_a8w8_CK'):
                    x_scale_ck = (
                        x_scale.expand(M).unsqueeze(1).contiguous()
                        if not x_per_row
                        else x_scale.reshape(M, 1).contiguous()
                    )
                    w_scale_ck = (
                        w_scale.expand(N).unsqueeze(0).contiguous()
                        if not w_per_row
                        else w_scale.reshape(1, N).contiguous()
                    )
                    if x_per_row or w_per_row:
                        _gemm_bump("ck_per_row")
                    else:
                        _gemm_bump("ck_per_tensor")
                    try:
                        result = aiter.gemm_a8w8_CK(x, w, x_scale_ck, w_scale_ck)
                    except RuntimeError as _ck_err:
                        if _LITE_DIAG:
                            global _CK_FAIL_DIAG_PRINTS
                            if _CK_FAIL_DIAG_PRINTS < 5:
                                _CK_FAIL_DIAG_PRINTS += 1
                                print(
                                    f"[LITE-GEMM-CK-FAIL #{_CK_FAIL_DIAG_PRINTS}] "
                                    f"x={tuple(x.shape)}/{x.dtype}/contig={x.is_contiguous()} "
                                    f"w={tuple(w.shape)}/{w.dtype}/contig={w.is_contiguous()} "
                                    f"x_scale_ck={tuple(x_scale_ck.shape)} "
                                    f"w_scale_ck={tuple(w_scale_ck.shape)} "
                                    f"err={type(_ck_err).__name__}: {_ck_err}",
                                    flush=True,
                                )
                        raise
                    if len(x_leading_shape) > 1:
                        result = result.reshape(*x_leading_shape, result.shape[-1])
                    return result
                else:
                    # Per-row scale on reduction axis — CK can't serve.
                    _gemm_bump("ck_reject_per_row_reduction_axis")

        elif not a_is_fp8 and b_is_fp8:
            if hasattr(aiter, 'gemm_a16w8'):
                _gemm_bump("ck_a16w8")
                a_mat = _dequantize_if_needed(A)
                if transA:
                    a_mat = a_mat.t()
                b_mat = b_data.t() if transB else b_data
                return aiter.gemm_a16w8(a_mat, b_mat, b_scale)
            _gemm_bump("ck_skip_bf16_fp8_no_kernel")

        elif not a_is_fp8 and not b_is_fp8:
            _gemm_bump("ck_skip_bf16_bf16")

        else:
            # a_is_fp8 and not b_is_fp8 — no CK branch for this combo
            _gemm_bump("ck_skip_fp8_bf16")

    except (RuntimeError, TypeError, AttributeError) as _e:
        _gemm_bump(f"ck_exception_{type(_e).__name__}")
    return None


# ---------------------------------------------------------------------------
# AITER Triton GEMM dispatch (dedicated Triton kernels per precision)
# ---------------------------------------------------------------------------

def _aiter_triton_gemm(A, transA, B, transB, a_data, a_scale, b_data, b_scale,
                       a_is_fp8, b_is_fp8):
    """Dispatch to AITER's dedicated Triton GEMM kernels.

    Per precision:
      FP4×FP4: aiter.ops.triton.gemm_afp4wfp4
      FP8×FP8 per-row:     aiter.ops.triton.gemm_a8w8_per_token_scale
      FP8×FP8 block-scale: aiter.ops.triton.gemm_a8w8_blockscale
      FP8×FP8 per-tensor:  aiter.ops.triton.gemm_a8w8
      BF16/FP16: aiter.ops.triton.gemm_a16w16
    All kernels compute Y = X @ W^T (weight is internally transposed).
    Returns result tensor or None.
    """
    try:
        # MXFP8: No Triton MXFP8 GEMM on MI300X. Fall through to dequant path.
        # TODO(MI350): Add Triton MXFP8 GEMM kernel dispatch when available.
        if _is_mxfp8(A) or _is_mxfp8(B):
            return None

        # FP4 × FP4
        if _is_fp4(A) and _is_fp4(B):
            from aiter.ops.triton.gemm_afp4wfp4 import (
                gemm_afp4wfp4 as triton_gemm_fp4,
            )
            a4_data, a4_scale = _get_fp4_data(A)
            b4_data, b4_scale = _get_fp4_data(B)
            return triton_gemm_fp4(a4_data, b4_data, a4_scale, b4_scale)

        # Float8Blockwise and standard FP8 layout mapping for Y = X @ W^T
        a_is_blockwise = _is_blockwise_fp8(A)
        b_is_blockwise = _is_blockwise_fp8(B)

        # When _get_raw_data returned the transpose (_data was None), the
        # orientation is already flipped — invert the transpose flag so the
        # dispatch logic below picks the right direction. This happens for
        # columnwise-only tensors in the wgrad path.
        a_transpose_only = _is_transpose_only(A)
        b_transpose_only = _is_transpose_only(B)
        effective_transA = transA ^ a_transpose_only
        effective_transB = transB ^ b_transpose_only

        # Triton FP8 kernels require 2D; flatten N-D leading dims of raw data
        # before the transpose dispatch (.t() only works on 2D).
        # _data has shape [d0, ..., K] → flatten to [prod(d), K].
        # _transpose has shape [K, d0, ...] → flatten to [K, prod(d)].
        x_leading = b_data.shape[:-1] if not b_transpose_only else b_data.shape[1:]
        if b_data.ndim > 2:
            if b_transpose_only:
                b_data = b_data.reshape(b_data.shape[0], -1)
            else:
                b_data = b_data.reshape(-1, b_data.shape[-1])
        if a_data.ndim > 2:
            if a_transpose_only:
                a_data = a_data.reshape(a_data.shape[0], -1)
            else:
                a_data = a_data.reshape(-1, a_data.shape[-1])

        if b_is_blockwise:
            x, x_scale = _get_blockwise_data(B, need_rowwise=not transB)
        else:
            x = b_data if not effective_transB else _fp8_transposed_operand(B, b_data)
            x_scale = b_scale

        if a_is_blockwise:
            w, w_scale = _get_blockwise_data(A, need_rowwise=transA)
        else:
            w = a_data if effective_transA else _fp8_transposed_operand(A, a_data)
            w_scale = a_scale

        if a_is_fp8 and b_is_fp8:

            # AITER Triton a8w8 kernels assume K-innermost on both operands
            # (stride[-1] == 1). Non-K-innermost operands are numerically
            # correct but ~10-100× slower with no diagnostic. Our
            # _fp8_transposed_operand path and the raw _data views should
            # both be K-innermost; assert to catch any future drift in the
            # _transpose_invalid flag or the cast_transpose output layout.
            assert x.stride(-1) == 1, (
                f"lite→AITER Triton a8w8: x must be K-innermost, got strides "
                f"{tuple(x.stride())}; shape={tuple(x.shape)}"
            )
            assert w.stride(-1) == 1, (
                f"lite→AITER Triton a8w8: w must be K-innermost, got strides "
                f"{tuple(w.stride())}; shape={tuple(w.shape)}"
            )

            if _is_per_row_scaled(x_scale) or _is_per_row_scaled(w_scale):
                # Per-row (per-token) FP8 — from CurrentScaling fused norm+quant.
                # Per-row scales are valid only when they index the kernel's
                # non-reduction axis (first dim of x and w). This holds for
                # forward (X @ W^T) and dgrad (dY @ W), but NOT wgrad
                # (dY^T @ X) where the transposes put per-row scales along
                # the reduction axis. Verify scale-axis alignment before
                # dispatching to the per-token kernel.
                x_scale_valid = (x_scale is None or x_scale.numel() == 1
                                 or x_scale.numel() == x.shape[0])
                w_scale_valid = (w_scale is None or w_scale.numel() == 1
                                 or w_scale.numel() == w.shape[0])
                if not (x_scale_valid and w_scale_valid):
                    _gemm_bump("triton_reject_per_row_reduction_axis")
                    return None  # Let caller fall back to dequantize + bf16 GEMM
                from aiter.ops.triton.gemm_a8w8_per_token_scale import (
                    gemm_a8w8_per_token_scale as triton_a8w8_pt,
                )
                # Kernel expects (M, 1) and (N, 1) shaped scales
                if x_scale is not None and x_scale.ndim == 1:
                    x_scale = x_scale.unsqueeze(1)
                if w_scale is not None and w_scale.numel() == 1:
                    w_scale = w_scale.expand(w.shape[0]).unsqueeze(1)
                elif w_scale is not None and w_scale.ndim == 1:
                    w_scale = w_scale.unsqueeze(1)
                _gemm_bump("triton_per_row")
                result = triton_a8w8_pt(x, w, x_scale, w_scale)
            elif (_is_block_scaled(x_scale) or _is_block_scaled(w_scale)
                    or a_is_blockwise or b_is_blockwise):
                from aiter.ops.triton.gemm_a8w8_blockscale import (
                    gemm_a8w8_blockscale as triton_a8w8_bs,
                )
                _gemm_bump("triton_blockscale")
                result = triton_a8w8_bs(x, w, x_scale, w_scale)
            else:
                # Per-tensor FP8. gemm_a8w8 indexes the scale pointer by row
                # (A) / col (B), so a scalar (1,) scale reads out of bounds
                # and produces garbage. Expand to (M,) and (N,) so every
                # row/col sees the same per-tensor scale.
                from aiter.ops.triton.gemm_a8w8 import (
                    gemm_a8w8 as triton_a8w8,
                )
                x_scale_exp = x_scale.expand(x.shape[0]).contiguous()
                w_scale_exp = w_scale.expand(w.shape[0]).contiguous()
                _gemm_bump("triton_per_tensor")
                result = triton_a8w8(x, w, x_scale_exp, w_scale_exp)

            # Restore the leading N-D shape from x (B operand) on the result
            if len(x_leading) > 1:
                result = result.reshape(*x_leading, result.shape[-1])
            return result

        elif not a_is_fp8 and b_is_fp8:
            try:
                from aiter.ops.triton.gemm_a16w8_blockscale import (
                    gemm_a16w8_blockscale as triton_a16w8,
                )
                x_hp = _dequantize_if_needed(B)
                if transB:
                    x_hp = x_hp.t().contiguous()
                return triton_a16w8(x_hp, w, w_scale)
            except ImportError:
                pass

        elif not a_is_fp8 and not b_is_fp8:
            # Skip FP32 — Triton GEMM only supports BF16/FP16
            a_mat = _dequantize_if_needed(A)
            if a_mat.dtype == torch.float32:
                return None

            from aiter.ops.triton.gemm_a16w16 import (
                gemm_a16w16 as triton_a16w16,
            )
            b_mat = _dequantize_if_needed(B)
            x = b_mat if not transB else b_mat.t().contiguous()
            w = a_mat if transA else a_mat.t().contiguous()
            return triton_a16w16(x, w)

    except (RuntimeError, TypeError, AttributeError, ImportError):
        pass
    return None


# ---------------------------------------------------------------------------
# Unified AITER dispatch with backend selection
# ---------------------------------------------------------------------------

def _aiter_gemm(A, transA, B, transB, D, quantizer, output_dtype,
                bias, bias_type, gelu, gelu_in, grad,
                accumulate, alpha):
    """Dispatch GEMM to AITER backend selected by NVTE_LITE_GEMM_BACKEND.

    Falls back through: preferred backend -> other backends -> None (PyTorch).
    """
    aiter = get_aiter()
    if aiter is None:
        return None

    a_data, a_scale = _get_raw_data(A)
    b_data, b_scale = _get_raw_data(B)

    a_is_fp8 = _is_quantized(A) or (isinstance(a_data, torch.Tensor) and a_data.dtype in _FP8_DTYPES)
    b_is_fp8 = _is_quantized(B) or (isinstance(b_data, torch.Tensor) and b_data.dtype in _FP8_DTYPES)

    result = None

    triton_args = (A, transA, B, transB, a_data, a_scale, b_data, b_scale,
                   a_is_fp8, b_is_fp8)

    if _GEMM_BACKEND == "triton":
        result = _aiter_triton_gemm(*triton_args)
        if result is None:
            result = _aiter_ck_gemm(
                aiter, a_data, a_scale, b_data, b_scale,
                a_is_fp8, b_is_fp8, transA, transB, A, B,
            )

    else:
        # Default "ck" path
        result = _aiter_ck_gemm(
            aiter, a_data, a_scale, b_data, b_scale,
            a_is_fp8, b_is_fp8, transA, transB, A, B,
        )
        if result is None:
            result = _aiter_triton_gemm(*triton_args)

    if result is None:
        _gemm_bump("pytorch_fallback")
        return None  # Signal caller to use PyTorch fallback

    # --- Post-GEMM epilogues ---
    if alpha != 1.0:
        result = result * alpha

    bias_grad = torch.Tensor()
    if bias is not None and bias.numel() > 0:
        if grad:
            grad_out = _dequantize_if_needed(B)
            bias_grad = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0)
        else:
            result = result + bias

    gelu_input = torch.Tensor()
    if gelu and gelu_in is not None:
        gelu_in.copy_(result)
        gelu_input = gelu_in
        result = torch.nn.functional.gelu(result, approximate='tanh')

    if accumulate and D is not None:
        D.add_(result)
    elif D is not None:
        D.copy_(result)
    else:
        D = result

    if quantizer is not None and hasattr(quantizer, 'quantize'):
        D = quantizer.quantize(D)

    return D, bias_grad, gelu_input, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generic_gemm(A, transA, B, transB, D, quantizer, output_dtype,
                 bias, bias_type, gelu, gelu_in, grad, workspace, workspace_size,
                 accumulate, use_split_accumulator,
                 comm_overlap=None, comm_type=None, extra_output=None,
                 bulk_overlap=False, alpha=1.0, beta=None):
    """General matrix-matrix multiply with optional bias, GELU, and accumulation.

    This is the primary GEMM entry point, replacing tex.generic_gemm.
    Dispatches to AITER CK/Triton kernels when available, falls back to torch.matmul.

    Backend selection via NVTE_LITE_GEMM_BACKEND env var:
      "ck" (default), "triton", "pytorch"
    """
    # --- AITER dispatch (all precisions) ---
    if _GEMM_BACKEND != "pytorch" and is_aiter_available():
        result = _aiter_gemm(
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad, accumulate, alpha,
        )
        if result is not None:
            return result[0], result[1], result[2], extra_output

    # --- PyTorch fallback ---
    #
    # For FP8×FP8, prefer torch._scaled_mm (hipBLASLt-backed on ROCm) — that's
    # the same path full TE takes and skips the dequant+matmul round trip. We
    # fall through to dequantize + torch.matmul on any exception (unsupported
    # scale/layout/dtype combo on this ROCm build).
    result = None
    if _is_quantized(A) and _is_quantized(B):
        _gemm_bump("pytorch_scaled_mm_attempt")
        result = _try_scaled_mm(
            A, transA, B, transB, _resolve_output_dtype(output_dtype),
        )
        if result is not None:
            _gemm_bump("pytorch_scaled_mm_ok")

    if result is not None:
        # torch._scaled_mm already handled compute and leading-dim restoration;
        # skip the dequantize+matmul block and go straight to epilogues.
        if alpha != 1.0:
            result = result * alpha

        bias_grad = torch.Tensor()
        if bias is not None and bias.numel() > 0:
            if grad:
                grad_out = _dequantize_if_needed(B)
                bias_grad = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0)
            else:
                result = result + bias

        gelu_input = torch.Tensor()
        if gelu and gelu_in is not None:
            gelu_in.copy_(result)
            gelu_input = gelu_in
            result = torch.nn.functional.gelu(result, approximate='tanh')

        if accumulate and D is not None:
            D.add_(result)
        elif D is not None:
            D.copy_(result)
        else:
            D = result

        if quantizer is not None and hasattr(quantizer, 'quantize'):
            D = quantizer.quantize(D)

        return D, bias_grad, gelu_input, extra_output

    # When backend=="pytorch" and _scaled_mm rejected the call (wgrad with
    # per-row scale on the reduction axis, block-scaled, unsupported dtype
    # combo, etc.), fall back to AITER before the catastrophically-slow
    # dequantize+matmul path. Dequant+matmul on FP8 operands runs 100-1000x
    # slower than AITER Triton and turns a few rejected calls into
    # multi-minute iterations.
    if (_GEMM_BACKEND == "pytorch" and is_aiter_available()
            and _is_quantized(A) and _is_quantized(B)):
        _gemm_bump("pytorch_aiter_fallback_attempt")
        aiter_result = _aiter_gemm(
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad, accumulate, alpha,
        )
        if aiter_result is not None:
            _gemm_bump("pytorch_aiter_fallback_ok")
            return aiter_result[0], aiter_result[1], aiter_result[2], extra_output

    _gemm_bump("pytorch_dequant_matmul")
    a = _dequantize_if_needed(A)
    b = _dequantize_if_needed(B)

    # cuBLAS column-major: C = op(A) @ op(B)
    # In row-major (PyTorch): C_row = B_row @ A_row (reversed operand order)
    # Typical "TN" layout: transA=True, transB=False
    #   A=[out,in] weight -> a.t()=[in,out], B=[batch,in] -> b as-is
    #   result = b @ a.t() = [batch,in] @ [in,out] = [batch,out]

    # cuBLAS GEMM treats N-D tensors as batched 2D: leading dims of B are
    # preserved in the output.  torch.matmul with 2D operands doesn't do
    # this, so we flatten to 2D, matmul, then restore B's leading dims.
    b_leading = b.shape[:-1]  # leading dims of B (before transpose)
    if a.dim() > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.dim() > 2:
        b = b.reshape(-1, b.shape[-1])

    if transA:
        a = a.t()
    if transB:
        b = b.t()

    compute_dtype = torch.bfloat16
    if a.dtype == torch.float32 or b.dtype == torch.float32:
        compute_dtype = torch.float32
    elif a.dtype == torch.float16 or b.dtype == torch.float16:
        compute_dtype = torch.float16

    a = a.to(compute_dtype)
    b = b.to(compute_dtype)

    result = torch.matmul(b, a)

    # Restore B's leading dimensions in the output (cuBLAS convention)
    if len(b_leading) > 1:
        result = result.view(*b_leading, result.shape[-1])

    if alpha != 1.0:
        result = result * alpha

    bias_grad = torch.Tensor()
    if bias is not None and bias.numel() > 0:
        if grad:
            grad_out = _dequantize_if_needed(B)
            bias_grad = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0)
        else:
            result = result + bias

    gelu_input = torch.Tensor()
    if gelu and gelu_in is not None:
        gelu_in.copy_(result)
        gelu_input = gelu_in
        result = torch.nn.functional.gelu(result, approximate='tanh')

    # Honor the caller-requested output dtype. cuBLAS casts to out_dtype in the
    # full build; without this cast, an fp32 operand promotes the whole result
    # to fp32 and the next module fails set_activation_dtype.
    out_torch_dtype = _resolve_output_dtype(output_dtype)
    if out_torch_dtype is not None and result.dtype != out_torch_dtype:
        result = result.to(out_torch_dtype)

    if accumulate and D is not None:
        D.add_(result)
    elif D is not None:
        D.copy_(result)
    else:
        D = result

    if quantizer is not None and hasattr(quantizer, 'quantize'):
        D = quantizer.quantize(D)

    return D, bias_grad, gelu_input, extra_output


def te_general_grouped_gemm(
    A, transa, B, transb, out, out_dtype, m_splits, bias, bias_dtype,
    single_output, pre_gelu_out, grad, workspaces, workspace_size,
    accumulate, use_split_accumulator, sm_count, **kwargs,
):
    """Grouped GEMM for MoE-style expert parallelism.

    Signature matches the C++ tex.te_general_grouped_gemm binding that
    general_grouped_gemm calls from cpp_extensions/gemm.py. Adapts to
    general_grouped_gemm_triton's keyword interface by:
    - deriving the "TN"/"NN"/"NT" layout string from transa/transb flags;
    - converting bias_dtype (TE_DType) into a use_bias flag;
    - treating a non-empty pre_gelu_out as gelu=True.
    """
    try:
        from transformer_engine.pytorch.triton_kernels.grouped_gemm import (
            general_grouped_gemm_triton,
        )
    except (ImportError, ModuleNotFoundError):
        raise NotImplementedError(
            "Grouped GEMM in lite mode requires AITER or Triton GMM. "
            "Install AITER (pip install amd-aiter) or use the standard GEMM path."
        )

    # Layout: T/N for each operand (C++ passes transA, transB booleans)
    layout = ("T" if transa else "N") + ("T" if transb else "N")

    # use_bias: C++ side passes an empty tensor list when no bias needed
    use_bias = bias is not None and len(bias) > 0 and bias[0].numel() > 0

    # gelu: C++ side allocates pre_gelu_out iff gelu was requested
    gelu = pre_gelu_out is not None and len(pre_gelu_out) > 0 \
        and pre_gelu_out[0].numel() > 0

    # out_dtype arrives as TE_DType (general_grouped_gemm reassigns it via
    # TE_DType[out[0].dtype]); convert back to torch.dtype for the Triton
    # wrapper, which compares directly against tensor.dtype.
    if not isinstance(out_dtype, torch.dtype):
        try:
            from transformer_engine.pytorch.triton_kernels.common import (
                te_dtype_to_torch_dtype,
            )
            out_dtype = te_dtype_to_torch_dtype(out_dtype)
        except (ImportError, KeyError):
            out_dtype = out[0].dtype

    # general_grouped_gemm_triton returns (out, bias_or_grad_bias, gelu_input).
    # The C++ tex.te_general_grouped_gemm returns ONLY the bias/grad_bias —
    # `out` and `pre_gelu_out` are mutated in place. Match that contract.
    _, bias_or_grad_bias, _ = general_grouped_gemm_triton(
        A, B, out, out_dtype, workspaces,
        layout=layout,
        m_splits=m_splits,
        gelu=gelu,
        grad=grad,
        accumulate=accumulate,
        bias=bias if use_bias else None,
        use_bias=use_bias,
        use_split_accumulator=use_split_accumulator,
        single_output=single_output,
    )
    return bias_or_grad_bias
