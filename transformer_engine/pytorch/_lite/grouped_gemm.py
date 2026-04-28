# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped GEMM operations for MoE-style expert parallelism.

This module replaces the C++ ``tex.te_general_grouped_gemm`` binding when
TE-lite is active, so Megatron's ``GroupedLinear`` / ``GroupedMLP`` can call
into AITER's Triton GMM kernels without the full TE C++ extension.

Supported today:

* BF16 / FP16 grouped GEMM for forward, dgrad, and wgrad — routed through
  ``transformer_engine.pytorch.triton_kernels.grouped_gemm.general_grouped_gemm_triton``,
  which wraps AITER's ``gmm`` / ``ptgmm`` Triton kernels.

Not yet supported:

* FP8 grouped GEMM. AITER's generic GMM family (``gmm``, ``ptgmm``,
  ``nptgmm``) is BF16/FP16 only — the ``p``/``np`` prefix is persistent vs
  non-persistent kernel, not per-tensor scaling. FP8 grouped expert
  compute lives in AITER as a fused MoE op (``aiter.fused_moe``,
  ``moe_op_gemm_a8w8_blockscale``) with a different API shape, so a
  separate dispatcher will land in Phase 2.
"""

import torch

from .gemm import _FP8_DTYPES, _is_quantized


def _is_fp8_operand(tensor):
    """True if `tensor` is FP8 raw bytes or a TE Float8Tensor wrapper."""
    if tensor is None:
        return False
    if _is_quantized(tensor):
        return True
    return getattr(tensor, "dtype", None) in _FP8_DTYPES


def _any_fp8(tensor_list):
    return tensor_list is not None and any(_is_fp8_operand(t) for t in tensor_list)


def te_general_grouped_gemm(
    A, transa, B, transb, out, out_dtype, m_splits, bias, bias_dtype,
    single_output, pre_gelu_out, grad, workspaces, workspace_size,
    accumulate, use_split_accumulator, sm_count, **kwargs,
):
    """Grouped GEMM for MoE expert parallelism (lite-mode replacement for
    ``tex.te_general_grouped_gemm``).

    Signature matches the C++ binding that ``general_grouped_gemm`` in
    ``cpp_extensions/gemm.py`` calls. Adapts to ``general_grouped_gemm_triton``'s
    keyword interface by:

    * deriving the ``"TN"``/``"NN"``/``"NT"`` layout string from
      ``transa``/``transb`` flags;
    * converting ``bias_dtype`` (``TE_DType``) into a ``use_bias`` flag;
    * treating a non-empty ``pre_gelu_out`` as ``gelu=True``.

    Mutation contract matches the C++ binding: ``out`` and ``pre_gelu_out``
    are filled in place; only the bias / grad-bias list is returned.
    """
    if _any_fp8(A) or _any_fp8(B):
        raise NotImplementedError(
            "FP8 grouped GEMM is not yet supported in TE-lite. "
            "AITER's generic GMM kernels (gmm/ptgmm/nptgmm) are BF16/FP16 only; "
            "FP8 expert compute requires the fused-MoE path "
            "(aiter.fused_moe / moe_op_gemm_a8w8_blockscale). "
            "Run with TE_FP8=0 for now, or wait for Phase 2 of the lite "
            "grouped-GEMM dispatcher."
        )

    # Empty-token short-circuit: when MoE token routing sends zero tokens to
    # this rank's local expert(s) (common in early training before the
    # auxiliary load-balancing loss kicks in), AITER's gmm asserts M > 0.
    # That's a legal MoE state from Megatron's side, so handle it here.
    # Forward / dgrad outputs are (M, ...) so already empty; wgrad output
    # is (G, K, N) and represents zero contribution from this microbatch:
    #   accumulate=True  -> leave existing_out alone (no-op contribution),
    #   accumulate=False -> zero existing_out so caller sees sane state.
    if m_splits is not None and sum(m_splits) == 0:
        is_wgrad = transa and not transb and grad
        if is_wgrad and not accumulate:
            for o in out:
                o.zero_()
        # bias / grad-bias: forward path returns the input bias list as-is;
        # wgrad path would normally return per-group grad-bias tensors, which
        # are also zero contribution under M=0. Match the empty-bias case.
        return [None] * len(m_splits) if (bias is None or len(bias) == 0) else bias

    try:
        from transformer_engine.pytorch.triton_kernels.grouped_gemm import (
            general_grouped_gemm_triton,
        )
    except (ImportError, ModuleNotFoundError):
        raise NotImplementedError(
            "Grouped GEMM in lite mode requires AITER. "
            "Install AITER (pip install amd-aiter) or use the standard "
            "GEMM path."
        )

    layout = ("T" if transa else "N") + ("T" if transb else "N")

    use_bias = bias is not None and len(bias) > 0 and bias[0].numel() > 0

    gelu = (
        pre_gelu_out is not None
        and len(pre_gelu_out) > 0
        and pre_gelu_out[0].numel() > 0
    )

    # out_dtype arrives as TE_DType (general_grouped_gemm reassigns via
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
