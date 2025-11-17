 
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight

from .tensor.quantized_tensor import QuantizedTensor

# CUSTOM PATHS HERE
import sys
sys.path.insert(0, '/root/llama3_code/quant_han')
from mxfp4_quantization_original import convert_to_mxfp4
from gemm import blockwise_mxfp4_gemm
# CUSTOM PATHS HERE

def _dequantize_tensor(tensor):
 
    if isinstance(tensor, QuantizedTensor):
        return tensor.dequantize()
    return tensor


def _quantize_to_fp4(tensor, aiter_or_han, use_sr=False, use_2dblock=False):
    if aiter_or_han == 0:
        quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        fp4_data, scales = quant_func(tensor, shuffle=True)
        return fp4_data, scales
    else:
        fp4_data, scales = convert_to_mxfp4(tensor, block_size=32, is_2d_block=use_2dblock, use_sr=use_sr, use_asm=False)
        return fp4_data, scales


def _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales, aiter_or_han, use_2dblock=False, out_dtype=torch.bfloat16, out_buffer=None):
 
    if aiter_or_han == 0:
        # Step 1: Convert scales to uint8
        A_scales_uint8 = A_scales.view(torch.uint8)
        B_scales_uint8 = B_scales.view(torch.uint8)

        # Step 2: Shuffle B (the "weight" operand in gemm_a4w4_asm)
        weight_layout = (16, 16)
        B_shuffled = shuffle_weight(B_fp4, layout=weight_layout)

        # Step 3: Allocate output buffer with padding
        M, K = A_fp4.shape
        N, _ = B_fp4.shape

        if out_buffer is not None:
            out_hp = out_buffer
            padded_M = out_buffer.shape[0]
        else:
            padded_M = (M + 31) // 32 * 32
            out_hp = torch.empty((padded_M, N), dtype=out_dtype, device=A_fp4.device)

        # Step 4: Perform FP4 GEMM: result = A @ B^T
        result = aiter.gemm_a4w4_asm(
            A_fp4,
            B_shuffled,
            A_scales_uint8,
            B_scales_uint8,
            out_hp,
            "",
            None,
            bpreshuffle=True,
            log2_k_split=0
        )

        # Step 5: Trim padding if necessary
        if result.shape[0] > M:
            result = result[:M, :]

        return result

    else:
        result = blockwise_mxfp4_gemm(
            A_fp4,
            A_scales,
            B_fp4,
            B_scales,
            use_2dblock_a=False,
            use_2dblock_b=use_2dblock,
            k_pack_a=True,
            k_pack_b=True,
            trans_a=False,
            trans_b=True,
            block_size=32,
            output_dtype=out_dtype,
        )
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
 
):
 
    #TODO: # define False if type is float8 tensor
    fp4_tensor = False
    aiter_or_han= 1 # 0: uses aiter funcs 1: uses han"s funcs

    #if han enabled
    use_2dblock=True
    use_sr=False


    if pass_type == 'fwd':

        if not fp4_tensor:
            # Forward pass: output = input @ weight^T + bias
            input_hp = _dequantize_tensor(input_tensor)
            weight_hp = _dequantize_tensor(weight)

            # Quantize to FP4
            input_fp4, input_scales = _quantize_to_fp4(input_hp, aiter_or_han)
            weight_fp4, weight_scales = _quantize_to_fp4(weight_hp, aiter_or_han, use_2dblock=use_2dblock)

            # Perform FP4 GEMM
            result = _fp4_gemm_core(input_fp4, input_scales, weight_fp4, weight_scales, aiter_or_han, use_2dblock=use_2dblock, out_dtype=out_dtype)

        else:
            pass
            #TODO: handle the native fp4 data and scale  and call fp4_gemm core with that variables
            #result = _fp4_gemm_core()

        # Add bias if needed
        if bias is not None:
            from .utils import cast_if_needed
            bias_casted = cast_if_needed(bias, out_dtype)
            result = result + bias_casted

        # Handle output quantization
        if output_quantizer is not None:
            return output_quantizer(result)
        return result

    elif pass_type == 'dgrad':
        
        if not fp4_tensor:
            # Dgrad pass: dgrad = grad_output @ weight
            weight_hp = _dequantize_tensor(weight)
            weight_hp_t = weight_hp.T
            grad_output_hp = _dequantize_tensor(grad_output)

            # Quantize to FP4
            grad_fp4, grad_scales = _quantize_to_fp4(grad_output_hp, aiter_or_han, use_sr=use_sr)
            weight_fp4, weight_scales = _quantize_to_fp4(weight_hp_t, aiter_or_han,  use_2dblock=use_2dblock)

            # Perform FP4 GEMM
            result = _fp4_gemm_core(grad_fp4, grad_scales, weight_fp4, weight_scales, aiter_or_han, use_2dblock=use_2dblock, out_dtype=out_dtype, out_buffer=dgrad_bulk)

        else:
            pass
            #TODO: handle the native fp4 data and scale  and call fp4_gemm core with that variables
            #result = _fp4_gemm_core()


        # Handle output quantization
        if grad_input_quantizer is not None:
            return grad_input_quantizer(result)
        return result

    elif pass_type == 'wgrad':
        # Wgrad pass: wgrad = grad_output^T @ input

        #TODO: make this more efficient
        # Dequantize and transpose input
        if not fp4_tensor:
            if input_tensor._data is not None:
                input_hp_t = _dequantize_tensor(input_tensor).T
            elif hasattr(input_tensor, '_transpose') and input_tensor._transpose is not None:
                # Swap to dequantize transpose directly
                saved_data = input_tensor._data
                input_tensor._data = input_tensor._transpose
                input_hp_t = input_tensor.dequantize()
                input_tensor._data = saved_data
    
            # Dequantize and transpose grad_output
            grad_output_hp = _dequantize_tensor(grad_output)
            grad_output_hp_t = grad_output_hp.T

            # Quantize to FP4
            grad_fp4, grad_scales = _quantize_to_fp4(grad_output_hp_t, aiter_or_han, use_sr=use_sr)
            input_fp4, input_scales = _quantize_to_fp4(input_hp_t, aiter_or_han)

            # Perform FP4 GEMM
            result = _fp4_gemm_core(grad_fp4, grad_scales, input_fp4, input_scales, aiter_or_han, use_2dblock=False, out_dtype=out_dtype)

        else:
            pass
            #TODO: handle the native fp4 data and scale  and call fp4_gemm core with that variables
            #result = _fp4_gemm_core()



        # Handle output - accumulate into main_grad if needed
        if fuse_wgrad_accumulation and main_grad is not None:
            if accumulate_wgrad_into_param_main_grad:
                main_grad.add_(result.to(main_grad.dtype))
            else:
                main_grad.copy_(result.to(main_grad.dtype))
            return None
        return result

    
