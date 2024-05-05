# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

import pytest
import torch
import triton
import triton.language as tl

from transformer_engine.pytorch.gemm_triton import te_gemm_triton, torch_to_te_dtype


TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8e4': tl.float8e4b8,
    'fp8e5': tl.float8e5b16,
}


def gen_input(M, N, ty_name, needTrans, seed, device='cuda'):
    d_type = name_to_tl_types[ty_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    if needTrans:
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T
    else:
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    # avoid type conversion rounding errors of subnormal values
    raw_data += 0.1
    if d_type == tl.float8e4b8:
        raw_data += torch.sign(raw_data)

    if (d_type == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
            (d_type == tl.float8e5b16 and TORCH_HAS_FP8E5B16) or not d_type.is_fp8():
        input = raw_data.to(tl_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)
    else:
        f8_tensor = raw_data.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, d_type)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16

def get_in_dtypes(in_type):
    types = in_type.split('-')
    if len(types) == 2:
        return types[0], types[1]
    else:
        return types[0], types[0]

def is_fp8(type_name):
    a_type, b_type = get_in_dtypes(type_name)
    return ( a_type in ('fp8e4', 'fp8e5') ) and ( b_type in ('fp8e4', 'fp8e5') )

@pytest.mark.parametrize("M, K, N, in_dtype, out_dtype, col_a, col_b, use_bias, bias_dtype, grad",
[ (*shape, in_dtype, out_dtype, col_a, col_b, use_bias, bias_dtype, grad)
    for shape in [(2304, 768, 4096),
                  (768, 768, 4096),
                  (768, 3072, 4096),
                  (229, 541, 541),
                  (71, 71, 3571),
                  (29, 29, 17389)
                  ]
    for in_dtype, out_dtype in [('fp16', 'fp16'),
                                ('bf16', 'bf16'),
                                ('fp16', 'fp32'),
                                ('fp32', 'fp32'),
                                ('fp8e4', 'fp32'),
                                ('fp8e4', 'bf16'),
                                ('fp8e4', 'fp16'),
                                #('fp8e4', 'fp8e4'),
                                # TODO: d_amax compute seems to have some accuracy issues
                                ('fp8e5-fp8e4', 'fp32'),
                                ('fp8e4-fp8e5', 'fp32'),
                                ('fp8e5-fp8e4', 'bf16'),
                                ('fp8e4-fp8e5', 'bf16'),
                                ]
    for col_a in [False, True]
    for col_b in [False, True]
    for use_bias in [True, False]
    for bias_dtype in ['bf16']
    for grad in [True, False]
    ]
)
def test_correctness(M, N, K, col_a, col_b, in_dtype, out_dtype, use_bias, bias_dtype, grad):
    a_in_dtype, b_in_dtype = get_in_dtypes(in_dtype)
    if is_fp8(in_dtype) and use_bias and grad:
        pytest.skip('Skip tests for fp8 GEMM with BGRADB.')

    if col_a and col_b:
        pytest.skip('Skip tests for TT layout')
    empty_tensor = torch.Tensor()

    a, a_fp16 = gen_input(K, M, a_in_dtype, col_a, 1, device='cuda')
    b, b_fp16 = gen_input(N, K, b_in_dtype, col_b, 2, device='cuda')
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)
    # Allocates output.
    tl_out_dtype = name_to_tl_types[out_dtype]
    torch_out_dtype = tl_to_torch_types[tl_out_dtype]
    tl_bias_dtype = name_to_tl_types[bias_dtype]
    torch_bias_dtype = tl_to_torch_types[tl_bias_dtype]
    c = torch.empty((N, M), device=a.device, dtype=torch_out_dtype)
    if is_fp8(in_dtype):
        A_scale_inverse = torch.randn((3,), dtype=torch.float32, device='cuda')
        B_scale_inverse = torch.randn((3,), dtype=torch.float32, device='cuda')
    else:
        A_scale_inverse = empty_tensor
        B_scale_inverse = empty_tensor

    D_scale = torch.empty((), dtype=torch.float32, device='cuda')
    transa = col_a
    transb = col_b

    A_type = torch_to_te_dtype( tl_to_torch_types[ name_to_tl_types[a_in_dtype] ] )
    B_type = torch_to_te_dtype( tl_to_torch_types[ name_to_tl_types[b_in_dtype] ] )
    D_type = torch_to_te_dtype( tl_to_torch_types[ name_to_tl_types[out_dtype] ] )
    if out_dtype in ('fp8e4', 'fp8e5'):
        D_amax = torch.empty((), dtype=torch.float32, device='cuda')
    else:
        D_amax = empty_tensor
    A_fp8_tensor = 0
    B_fp8_tensor = 0
    if use_bias:
        if grad:
            bias = torch.empty((N,), dtype=torch_bias_dtype, device='cuda')
        else:
            bias = torch.randn((M,), dtype=torch_bias_dtype, device='cuda')
    else:
        bias = empty_tensor
    bias_type = torch_to_te_dtype(torch_bias_dtype)
    pre_gelu_out = empty_tensor
    workspace = empty_tensor
    workspaceSize = 0
    accumulate = False
    use_split_accumulator = False

    output_fp8 = is_fp8(out_dtype)

    ## b is (N, K) in row major and a is (K, M) in row major, 
    ## when doing GEMM in BLAS
    ## a and b are swapped, so gemm_a is (M, K) in column major
    ## and gemm_b is (K, N) is column major
    torch_output = torch.matmul(b_fp32, a_fp32)

    if is_fp8(in_dtype):
        # For f8 and inputs, multiplied by the scales
        torch_output *= A_scale_inverse[A_fp8_tensor] * B_scale_inverse[B_fp8_tensor]

    if use_bias:
        if grad:
            torch_bias_gradient = b.sum(axis=1).to(torch_bias_dtype)
        else:
            torch_output += bias

    if output_fp8:
        torch_output_amax = torch.max(torch.abs(torch_output))
        fp8_amax = 240.0 if out_dtype == 'fp8e4' else 57344.0
        D_scale = fp8_amax * 0.5 / torch_output_amax
        torch_output *= D_scale

    torch_output = torch_output.to(torch_out_dtype)

    # Shape is different based on trans value
    # for example, if transa == True, a is (K, M),
    # we want the shape to be fed to te_gemm_triton
    # to be (M, K) as te_gemm_triton will apply the
    # the transpose.
    a_col_major = a.T if transa else a
    b_col_major = b.T if transb else b
    te_gemm_triton(a_col_major,
                   A_scale_inverse,
                   A_fp8_tensor,
                   A_type,
                   transa,
                   b_col_major,
                   B_scale_inverse,
                   B_fp8_tensor,
                   B_type,
                   transb,
                   c,
                   D_scale,
                   D_type,
                   D_amax,
                   bias,
                   bias_type,
                   pre_gelu_out,
                   grad,
                   workspace,
                   workspaceSize,
                   accumulate,
                   use_split_accumulator)

    atol = 5e-3
    if torch_out_dtype == 'bf16' and use_bias:
        atol = 8e-3
    rtol = 0 if torch.version.hip is None else 1e-2
    if output_fp8:
        def check_if_adjacent_fp8(a, b, out_dtype):
            m = 3 if out_dtype == 'fp8e4' else 2
            a = a.to(torch.float32)
            b = b.to(torch.float32)
            check = torch.abs(a-b) <= torch.exp(torch.floor(torch.log(torch.abs(a))))
            return torch.all(check)
        assert check_if_adjacent_fp8(c, torch_output, out_dtype)
    else:
        torch.testing.assert_close(c.to(torch.float32), torch_output.to(torch.float32), atol=atol, rtol=rtol)

    if use_bias and grad:
        torch.testing.assert_close(bias, torch_bias_gradient, atol=5e-3, rtol=rtol)

    if output_fp8:
        torch.testing.assert_close(D_amax, torch_output_amax, atol=5e-3, rtol=rtol)
