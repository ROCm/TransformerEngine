
import torch
import aiter
from aiter.ops.shuffle import shuffle_weight
 
from .tensor.quantized_tensor import QuantizedTensor
from .tensor._internal.float8_tensor_base import Float8TensorBase
from .hadamard import HadamardFactory, HadamardTransform
# Configure Hadamard for MXFP (block_size=32, deterministic)
HadamardFactory.configure(block_size=32, randomized=False)
 

def _dequantize_tensor(tensor):

    if isinstance(tensor, QuantizedTensor):
        return tensor.dequantize()
    elif isinstance(tensor, Float8TensorBase):
        return tensor.dequantize()
    return tensor


def _quantize_to_fp4(tensor ):
   
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    fp4_data, scales = quant_func(tensor, shuffle=True )
    return fp4_data, scales



def _fp4_gemm_core(A_fp4, A_scales, B_fp4, B_scales , out_dtype=torch.bfloat16, out_buffer=None):


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
    # New parameters for pre-computed FP4 weights, inputs and grad_output
    weight_fp4_t=None,
    weight_scales_t=None,
    input_fp4_t=None,
    input_scales_t=None,
    grad_fp4_t=None,
    grad_scales_t=None,
):
 
    #TODO: # define False if type is float8 tensor
    fp4_tensor = False

    #use_hadamard=True


    if pass_type == 'fwd':

        if not fp4_tensor:
            # Forward pass: output = input @ weight^T + bias
            input_hp = _dequantize_tensor(input_tensor)
            weight_hp = _dequantize_tensor(weight)
           
            # Quantize to FP4
            input_fp4, input_scales = _quantize_to_fp4(input_hp   )
            input_hp_t=input_hp.T
            input_fp4_t, input_scales_t = _quantize_to_fp4(input_hp_t)


            weight_fp4, weight_scales = _quantize_to_fp4(weight_hp   )
            weight_hp_t = weight_hp.T
            weight_fp4_t, weight_scales_t = _quantize_to_fp4(weight_hp_t)

            # Perform FP4 GEMM
            result = _fp4_gemm_core(input_fp4, input_scales, weight_fp4, weight_scales, out_dtype=out_dtype)

      
   

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
        return result,weight_fp4_t,weight_scales_t,input_fp4_t,input_scales_t

    elif pass_type == 'dgrad':

        if not fp4_tensor:
            # Dgrad pass: grad_input = grad_output @ weight


            grad_output_hp = _dequantize_tensor(grad_output)
            grad_output_hp_t = grad_output_hp.T

            # Quantize to FP4 - use mode 1 for grad_output with same min/max
            grad_fp4_t, grad_scales_t = _quantize_to_fp4(grad_output_hp_t )
            # Quantize to FP4 - use mode 1 for grad_output with same min/max
            grad_fp4, grad_scales = _quantize_to_fp4(grad_output_hp )
    

            # Perform FP4 GEMM
            result = _fp4_gemm_core(grad_fp4, grad_scales, weight_fp4_t, weight_scales_t, out_dtype=out_dtype, out_buffer=dgrad_bulk)

 
             
             

        else:
            pass
            #TODO: handle the native fp4 data and scale  and call fp4_gemm core with that variables
            #result = _fp4_gemm_core()


        # Handle output quantization
        if grad_input_quantizer is not None:
            return grad_input_quantizer(result)
        return result,grad_fp4_t,grad_scales_t

    elif pass_type == 'wgrad':
        # Wgrad pass: wgrad = grad_output^T @ input

        #TODO: make this more efficient
        # Dequantize and transpose input
        if not fp4_tensor:
 
            # Perform FP4 GEMM
            result = _fp4_gemm_core(grad_fp4_t, grad_scales_t, input_fp4_t, input_scales_t, out_dtype=out_dtype)

            
                
   

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

    
