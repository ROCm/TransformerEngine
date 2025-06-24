import torch
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.triton_kernels.common import get_torch_dtype_to_triton_dtype, te_dtype_to_triton_dtype
import triton
import triton.language as tl

@triton.jit
def _fp8_dequantize_kernel(
    in_ptr,
    scale_inv_ptr,
    out_ptr,
    N_ELEMENTS,
    FP8_TYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    in_val_raw = tl.load(in_ptr + offsets, mask=mask)
    scale_inv = tl.load(scale_inv_ptr)
    dequantized_val = (in_val_raw.to(tl.float32) * scale_inv).to(OUT_DTYPE)
    tl.store(out_ptr + offsets, dequantized_val, mask=mask)

def dequantize_triton_wrapper(input_qtensor: QuantizedTensor, output_dtype: torch.dtype):
    if not isinstance(input_qtensor, QuantizedTensor): # Using abstract QuantizedTensor
        raise TypeError("Input must be a QuantizedTensor for Triton dequantize.")

    in_data = input_qtensor._data
    scale_inv = input_qtensor._scale_inv

    if in_data.numel() == 0:
        return torch.empty_like(in_data, dtype=output_dtype, device=in_data.device)

    assert in_data.is_cuda and scale_inv.is_cuda, "Inputs must be on Same device."

    N_ELEMENTS = in_data.numel()

    tl_dtype = te_dtype_to_triton_dtype(input_qtensor._fp8_dtype)

    triton_out_dtype = get_torch_dtype_to_triton_dtype(output_dtype)
    if triton_out_dtype is None:
        raise ValueError(f"Unsupported output dtype for Triton: {output_dtype}")

    out = torch.empty_like(in_data, dtype=output_dtype, device=in_data.device)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta['BLOCK_SIZE']),)

    _fp8_dequantize_kernel[grid](
        triton.reinterpret(in_data, tl_dtype),
        scale_inv,
        out,
        N_ELEMENTS,
        FP8_TYPE=tl_dtype,
        OUT_DTYPE=triton_out_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out