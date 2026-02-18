import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper
import transformer_engine_torch as tex

device = torch.device("cuda")

# Simple case: NN layout
M, N, K = 128, 256, 512
transa, transb = False, False

a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

# Simulate what te_generic_gemm_triton does
A_wrapper = MXFP8TensorWrapper(a_mxfp8)
B_wrapper = MXFP8TensorWrapper(b_mxfp8)

print("="*60)
print("STEP 1: Create wrappers")
print("="*60)
print(f"A_wrapper.size() = {A_wrapper.size()}")
print(f"B_wrapper.size() = {B_wrapper.size()}")

print("\n" + "="*60)
print("STEP 2: Extract data with transpose flags")
print("="*60)
print(f"Calling A_wrapper.get_data_and_scale_for_gemm(will_transpose={transa})")
A_data, a_scale_inv = A_wrapper.get_data_and_scale_for_gemm(will_transpose=transa)
print(f"  A_data: {A_data.shape}")
print(f"  a_scale_inv: {a_scale_inv.shape}")

print(f"\nCalling B_wrapper.get_data_and_scale_for_gemm(will_transpose={transb})")
B_data, b_scale_inv = B_wrapper.get_data_and_scale_for_gemm(will_transpose=transb)
print(f"  B_data: {B_data.shape}")
print(f"  b_scale_inv: {b_scale_inv.shape}")

print("\n" + "="*60)
print("STEP 3: Compute dimensions")
print("="*60)

def product(shape):
    ret = 1
    for i in shape:
        ret *= i
    return ret

A0 = product(A_wrapper.size()[:-1])
A1 = product(A_wrapper.size()[-1:])
B0 = product(B_wrapper.size()[:-1])
B1 = product(B_wrapper.size()[-1:])

m = A0 if transa else A1
k = A1 if transa else A0
n = B1 if transb else B0

print(f"A_wrapper.size() = {A_wrapper.size()}")
print(f"  A0 (all but last) = {A0}")
print(f"  A1 (last) = {A1}")
print(f"B_wrapper.size() = {B_wrapper.size()}")
print(f"  B0 (all but last) = {B0}")
print(f"  B1 (last) = {B1}")
print(f"\nWith transa={transa}, transb={transb}:")
print(f"  m = {'A0' if transa else 'A1'} = {m}")
print(f"  k = {'A1' if transa else 'A0'} = {k}")
print(f"  n = {'B1' if transb else 'B0'} = {n}")
print(f"\nExpected: m={M}, k={K}, n={N}")
print(f"Got:      m={m}, k={k}, n={n}")

print("\n" + "="*60)
print("STEP 4: Flatten and swap")
print("="*60)
A_flat = A_data.reshape(-1, A_data.shape[-1])
B_flat = B_data.reshape(-1, B_data.shape[-1])
print(f"A_flat: {A_flat.shape}")
print(f"B_flat: {B_flat.shape}")

# With the new fix (no additional transpose for MXFP8)
a_row_major = B_flat
b_row_major = A_flat
print(f"\nAfter swap (MXFP8 path, no additional transpose):")
print(f"  a_row_major = B_flat: {a_row_major.shape}")
print(f"  b_row_major = A_flat: {b_row_major.shape}")

print("\n" + "="*60)
print("STEP 5: Actual dimensions for kernel")
print("="*60)
actual_m = a_row_major.shape[0]
actual_k = a_row_major.shape[1]
actual_n = b_row_major.shape[1]
print(f"actual_m = a_row_major.shape[0] = {actual_m}")
print(f"actual_k = a_row_major.shape[1] = {actual_k}")
print(f"actual_n = b_row_major.shape[1] = {actual_n}")
print(f"\nKernel will compute: [{actual_m}, {actual_n}]")
print(f"Expected:            [{M}, {N}]")
