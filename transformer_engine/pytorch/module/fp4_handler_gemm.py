
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight
from ..utils import cast_if_needed


def _select_kernel(layout: str, grad: bool) -> str:
    """Select kernel based on GEMM layout.
    
    Args:
        layout: GEMM layout (TN=fprop, NN=dgrad, NT=wgrad)
        grad: Whether this is a gradient computation
    """
    if layout == "NT" and grad:
        return "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
    else:
        return "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"


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


def fp4_gemm_layout(
    A,
    B,
    layout: str = "TN",
    out_dtype: torch.dtype = torch.bfloat16,
    bias=None,
    out=None,
    grad: bool = False,
    accumulate: bool = False,
):
    """FP4 GEMM using layout notation (TN/NN/NT).
    
    Layout mapping:
        TN: A=weight, B=input       → fprop: input @ weight^T
        NN: A=weight, B=grad_output → dgrad: grad_output @ weight  
        NT: A=input, B=grad_output  → wgrad: grad_output^T @ input
    """
    with torch._C._DisableTorchDispatch():
        kernel_name = _select_kernel(layout, grad)
        
        if layout == "TN":
            # Forward: input @ weight^T
            A_fp4 = B._rowwise_data
            A_scales = B._rowwise_scale
            B_fp4 = A._rowwise_data
            B_scales = A._rowwise_scale
            b_pre_shuffled = True
            
        elif layout == "NN":
            # Dgrad: grad_output @ weight
            A_fp4 = B._rowwise_data
            A_scales = B._rowwise_scale
            B_fp4 = A._columnwise_data
            B_scales = A._columnwise_scale
            b_pre_shuffled = True
            
        elif layout == "NT":
            # Wgrad: grad_output^T @ input
            A_fp4 = B._columnwise_data
            A_scales = B._columnwise_scale
            B_fp4 = A._columnwise_data
            B_scales = A._columnwise_scale
            b_pre_shuffled = False
            
        else:
            raise ValueError(f"Unsupported layout for FP4 GEMM: {layout}")
        
        # Execute GEMM with optional accumulation
        if accumulate and out is not None:
            result = _fp4_gemm_core(
                A_fp4, A_scales, B_fp4, B_scales,
                out_dtype=out.dtype if out is not None else out_dtype,
                out_buffer=None,
                kernel_name=kernel_name,
                b_pre_shuffled=b_pre_shuffled
            )
            out.add_(result)
            result = None
        else:
            result = _fp4_gemm_core(
                A_fp4, A_scales, B_fp4, B_scales,
                out_dtype=out_dtype,
                out_buffer=out,
                kernel_name=kernel_name,
                b_pre_shuffled=b_pre_shuffled
            )
        
        # Add bias for forward pass only
        if bias is not None and layout == "TN" and not grad:
            bias_casted = cast_if_needed(bias, out_dtype)
            if result is not None:
                result = result + bias_casted
        
        return result
