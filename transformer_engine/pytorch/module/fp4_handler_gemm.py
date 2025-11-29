
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight
from ..tensor.quantized_tensor import QuantizedTensor
from ..tensor._internal.float8_tensor_base import Float8TensorBase
from ..tensor._internal.mxfp4_tensor_base import MXFP4TensorBase
from ..utils import cast_if_needed


def _select_kernel(op_type: str) -> str:
    if "LayerNormLinear" in op_type:
        if op_type.endswith("_wgrad"):
            return "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
        else:
            return "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
    elif "Linear" in op_type:
        if op_type.endswith("_wgrad"):
            return "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
        else:
            return "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
    return ""


def _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=torch.bfloat16, out_buffer=None, kernel_name="", b_pre_shuffled=True):
    A_scales_uint8 = A_scales.view(torch.uint8)
    B_scales_uint8 = B_scales.view(torch.uint8)

    # Shuffle B if not pre-shuffled (e.g., input in wgrad)
    if b_pre_shuffled:
        B_shuffled = B_fp4
    else:
        B_shuffled = shuffle_weight(B_fp4, layout=(16, 16))

    M = A_fp4.shape[0]
    N = B_fp4.shape[0]

    if out_buffer is not None:
        out_hp = out_buffer
        padded_M = out_buffer.shape[0]
    else:
        padded_M = (M + 31) // 32 * 32
        out_hp = torch.empty((padded_M, N), dtype=out_dtype, device=A_fp4.device)

    result = aiter.gemm_a4w4_asm(
        A_fp4,
        B_shuffled,
        A_scales_uint8,
        B_scales_uint8,
        out_hp,
        kernel_name,
        None,
        bpreshuffle=True,
        log2_k_split=0
    )

    if result.shape[0] > M:
        result = result[:M, :]

    return result


def fp4_gemm(
    pass_type,
    weight=None,
    input_tensor=None,
    grad_output=None,
    output_quantizer=None,
    grad_input_quantizer=None,
    bias=None,
    out_dtype=torch.bfloat16,
    dgrad_bulk=None,
    main_grad=None,
    fuse_wgrad_accumulation=False,
    accumulate_wgrad_into_param_main_grad=False,
    op_type="",
):
    with torch._C._DisableTorchDispatch():
        if pass_type == 'fwd':
            A_fp4 = input_tensor._rowwise_data
            A_scales = input_tensor._rowwise_scale
            B_fp4 = weight._rowwise_data  # Weight is pre-shuffled
            B_scales = weight._rowwise_scale

            kernel_name = _select_kernel(op_type + "_fprop") if op_type else ""
            result = _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=out_dtype, kernel_name=kernel_name, b_pre_shuffled=True)

            if bias is not None:
                bias_casted = cast_if_needed(bias, out_dtype)
                result = result + bias_casted

            return result

        elif pass_type == 'dgrad':
            A_fp4 = grad_output._rowwise_data
            A_scales = grad_output._rowwise_scale
            B_fp4 = weight._columnwise_data  # Weight is pre-shuffled
            B_scales = weight._columnwise_scale

            kernel_name = _select_kernel(op_type + "_dgrad") if op_type else ""
            result = _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=out_dtype, out_buffer=dgrad_bulk, kernel_name=kernel_name, b_pre_shuffled=True)

            return result

        elif pass_type == 'wgrad':
            A_fp4 = grad_output._columnwise_data
            A_scales = grad_output._columnwise_scale
            B_fp4 = input_tensor._columnwise_data  # Input is NOT pre-shuffled
            B_scales = input_tensor._columnwise_scale

            kernel_name = _select_kernel(op_type + "_wgrad") if op_type else ""

            if fuse_wgrad_accumulation and main_grad is not None:
                if accumulate_wgrad_into_param_main_grad:
                    result = _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=main_grad.dtype, kernel_name=kernel_name, b_pre_shuffled=False)
                    main_grad.add_(result)
                else:
                    _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=main_grad.dtype, out_buffer=main_grad, kernel_name=kernel_name, b_pre_shuffled=False)
                return None

            result = _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, out_dtype=out_dtype, kernel_name=kernel_name, b_pre_shuffled=False)
            return result

