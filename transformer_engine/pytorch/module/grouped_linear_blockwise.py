# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# ROCm-only self-contained autograd Function for DeepSeek-style blockwise FP8
# MoE grouped GEMM. Mirrors Primus-Turbo's FP8GroupedGemmBlockFunc using the
# ported Triton kernels; delegated to from GroupedLinear.forward behind a guard.

import torch

from ..triton_kernels.common import te_dtype_to_torch_dtype


def _fp8_dtype(quantizers):
    if quantizers and quantizers[0] is not None:
        return te_dtype_to_torch_dtype(quantizers[0].dtype)
    return torch.float8_e4m3fn


def _group_offsets(m_splits, device):
    group_lens = torch.tensor(m_splits, dtype=torch.int64, device=device)
    group_offs = torch.zeros(len(m_splits) + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, 0)
    return group_lens, group_offs


class _GroupedLinearBlockwiseFP8(torch.autograd.Function):
    """Blockwise FP8 grouped GEMM (fwd/dgrad/wgrad) via ported Triton kernels.

    First-cut scope: no bias, no fuse_wgrad_accumulation, no cpu_offload, no
    delay-wgrad, no save_original_input (each rejected, not silently ignored).
    Uses dev's GroupedLinear calling convention: inputs are ``(inp,
    non_tensor_args, *weights_and_biases)`` with the orchestration flags packed
    into ``non_tensor_args``.
    """

    @staticmethod
    def forward(ctx, inp, non_tensor_args, *weights_and_biases):
        # pylint: disable=missing-function-docstring
        (
            m_splits,
            use_bias,
            is_first_microbatch,
            fp8,
            fp8_calibration,
            wgrad_store,
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
            fuse_wgrad_accumulation,
            cpu_offloading,
            sequence_parallel,
            activation_dtype,
            is_grad_enabled,
            module,
            skip_fp8_weight_update,
            save_original_input,
            debug,
            m_splits_tensor,
            actual_m_splits,
            unpad_output,
        ) = non_tensor_args

        from ..triton_kernels.blockwise_fp8_quantize import (
            quantize_fp8_blockwise,
            quantize_fp8_blockwise_weight,
            quantize_fp8_blockwise_segment_m,
        )
        from ..triton_kernels.blockwise_fp8_grouped_gemm import (
            grouped_gemm_fp8_blockwise_triton_kernel,
        )

        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]

        # First-cut: unsupported orchestration features -> loud error, not silent wrong.
        # fp8 is an internal invariant (assert); the rest are user-reachable -> raise.
        assert fp8, "blockwise grouped FP8 path requires fp8=True"
        if use_bias:
            raise NotImplementedError("bias is not supported in the blockwise grouped FP8 path yet")
        if fuse_wgrad_accumulation:
            raise NotImplementedError(
                "fuse_wgrad_accumulation (gradient_accumulation_fusion) is not yet supported in "
                "the ROCm blockwise grouped FP8 path. Pass --no-gradient-accumulation-fusion to the "
                "training script: wgrad is then returned as a plain gradient and Megatron's DDP "
                "post-hook accumulates it into the fp32 main_grad (numerically equivalent for bring-up)."
            )
        if cpu_offloading:
            raise NotImplementedError("cpu_offloading is not supported in the blockwise grouped FP8 path yet")
        if save_original_input:
            raise NotImplementedError("save_original_input is not supported in the blockwise grouped FP8 path yet")
        if wgrad_store is not None and wgrad_store.delay_wgrad_compute():
            raise NotImplementedError("delayed wgrad is not supported in the blockwise grouped FP8 path yet")
        # dev-only GroupedLinear features absent from the v2.8 bring-up path.
        if debug:
            raise NotImplementedError("debug quantization is not supported in the blockwise grouped FP8 path yet")
        if unpad_output or (actual_m_splits is not None and list(actual_m_splits) != list(m_splits)):
            raise NotImplementedError(
                "the ROCm fused-pad / unpad_output path is not supported in the blockwise grouped FP8 path yet"
            )

        dt = _fp8_dtype(weight_quantizers)
        in_features = weights[0].size(-1)
        out_features = weights[0].size(0)
        device = inp.device

        inp_shape = inp.shape
        a = inp.reshape(-1, in_features).to(activation_dtype).contiguous()  # [M_total, K] bf16
        group_lens, group_offs = _group_offsets(m_splits, device)

        # Stack per-expert weights [G, N, K] (bf16).
        w = torch.stack([wt.to(activation_dtype).contiguous() for wt in weights], 0).contiguous()

        # Quantize: activation rowwise (1x128 along K), weights 128x128.
        a_row, a_srow = quantize_fp8_blockwise(a, dt, axis=1, block_size=128)
        b_fp8, b_scale = quantize_fp8_blockwise_weight(w, dt, block_size=128)

        # Forward GEMM: out[seg] = A[seg] @ W[g]^T  (trans_b=True).
        out = grouped_gemm_fp8_blockwise_triton_kernel(
            a_row, b_fp8, a_srow, b_scale, group_offs, trans_b=True, out_dtype=activation_dtype
        )

        if is_grad_enabled:
            # Segment-padded columnwise activation (from bf16) for the variable-K wgrad.
            a_col, a_scol, _vk_lens, vk_offs = quantize_fp8_blockwise_segment_m(
                a, dt, 128, group_lens, group_offs
            )
            ctx.save_for_backward(a_col, a_scol, b_fp8, b_scale, group_lens, group_offs, vk_offs)
            ctx.dt = dt
            ctx.activation_dtype = activation_dtype
            ctx.num_gemms = num_gemms
            ctx.inp_shape = inp_shape
            ctx.out_features = out_features
            ctx.requires_dgrad = inp.requires_grad
            ctx.weight_requires_grad = weights[0].requires_grad

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        from ..triton_kernels.blockwise_fp8_quantize import (
            quantize_fp8_blockwise,
            quantize_fp8_blockwise_segment_m,
        )
        from ..triton_kernels.blockwise_fp8_grouped_gemm import (
            grouped_gemm_fp8_blockwise_triton_kernel,
            grouped_gemm_fp8_blockwise_variable_k_triton_kernel,
        )

        a_col, a_scol, b_fp8, b_scale, group_lens, group_offs, vk_offs = ctx.saved_tensors
        dt = ctx.dt
        g_out = grad_output.reshape(-1, ctx.out_features).contiguous()  # [M_total, N]

        # dgrad: dX[seg] = dY[seg] @ W[g]  (trans_b=False) -> [M_total, K].
        dgrad = None
        if ctx.requires_dgrad:
            go_row, go_srow = quantize_fp8_blockwise(g_out, dt, axis=1, block_size=128)
            dgrad = grouped_gemm_fp8_blockwise_triton_kernel(
                go_row, b_fp8, go_srow, b_scale, group_offs, trans_b=False,
                out_dtype=ctx.activation_dtype,
            )

        # wgrad: dW[g] = dY[g]^T @ X[g]  (variable-K) -> [G, N, K].
        wgrad_list = [None] * ctx.num_gemms
        if ctx.weight_requires_grad:
            go_col, go_scol, _l, _o = quantize_fp8_blockwise_segment_m(
                g_out, dt, 128, group_lens, group_offs
            )
            dW = grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
                go_col, a_col, go_scol, a_scol, vk_offs, out_dtype=ctx.activation_dtype
            )  # [G, N, K]
            wgrad_list = [dW[g].contiguous() for g in range(ctx.num_gemms)]

        grad_biases = [None] * ctx.num_gemms  # bias rejected in forward

        # Grads match forward inputs: (inp, non_tensor_args, *weights, *biases).
        return (
            dgrad.view(ctx.inp_shape) if dgrad is not None else None,
            None,  # non_tensor_args
            *wgrad_list,
            *grad_biases,
        )
