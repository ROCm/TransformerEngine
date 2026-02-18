import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper

device = torch.device("cuda")

M, N, K = 32, 32, 32
transa, transb = False, False

a_fp32 = torch.ones((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.ones((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("Debugging what gets passed to kernel")
print("=" * 60)

# Use the wrappers
A_wrapper = MXFP8TensorWrapper(a_mxfp8)
B_wrapper = MXFP8TensorWrapper(b_mxfp8)

# Following the logic in gemm_triton.py
# For A: use columnwise when transa=False (will_transpose=True gives columnwise)
A_data, a_scale_inv = A_wrapper.get_data_and_scale_for_gemm(will_transpose=(not transa))
# For B: use rowwise when transb=False (will_transpose=False gives rowwise)
B_data, b_scale_inv = B_wrapper.get_data_and_scale_for_gemm(will_transpose=transb)

print(f"\nAfter wrapper selection (transa={transa}, transb={transb}):")
print(f"A data shape: {A_data.shape}")
print(f"A scale shape: {a_scale_inv.shape if a_scale_inv is not None else None}")
print(f"B data shape: {B_data.shape}")
print(f"B scale shape: {b_scale_inv.shape if b_scale_inv is not None else None}")

# After flattening (in this case, no change)
A_flat = A_data.reshape(-1, A_data.shape[-1])
B_flat = B_data.reshape(-1, B_data.shape[-1])

print(f"\nAfter flattening:")
print(f"A_flat: {A_flat.shape}")
print(f"B_flat: {B_flat.shape}")

# Scale slicing
VEC_SIZE = 32
if not transa:
    # Columnwise quantization: scales are [rows//32, cols]
    expected_a_scale_shape = (A_flat.shape[0] // VEC_SIZE, A_flat.shape[1])
else:
    # Rowwise quantization: scales are [rows, cols//32]
    expected_a_scale_shape = (A_flat.shape[0], A_flat.shape[1] // VEC_SIZE)

if not transb:
    # Rowwise quantization: scales are [rows, cols//32]
    expected_b_scale_shape = (B_flat.shape[0], B_flat.shape[1] // VEC_SIZE)
else:
    # Columnwise quantization: scales are [rows//32, cols]
    expected_b_scale_shape = (B_flat.shape[0] // VEC_SIZE, B_flat.shape[1])

print(f"\nExpected scale shapes after slicing:")
print(f"A scale: {expected_a_scale_shape}")
print(f"B scale: {expected_b_scale_shape}")

# Slice scales
if a_scale_inv.shape != expected_a_scale_shape:
    a_scale_sliced = a_scale_inv[:expected_a_scale_shape[0], :expected_a_scale_shape[1]].contiguous()
else:
    a_scale_sliced = a_scale_inv

if b_scale_inv.shape != expected_b_scale_shape:
    b_scale_sliced = b_scale_inv[:expected_b_scale_shape[0], :expected_b_scale_shape[1]].contiguous()
else:
    b_scale_sliced = b_scale_inv

print(f"\nActual scale shapes after slicing:")
print(f"A scale: {a_scale_sliced.shape}")
print(f"B scale: {b_scale_sliced.shape}")

# Now the BLAS swap
print(f"\n" + "=" * 60)
print("BLAS column-major to row-major swap")
print("=" * 60)

# For MXFP8, we swap A and B
a_row_major = B_flat.T if transb else B_flat
b_row_major = A_flat.T if transa else A_flat

# Scales are swapped to match operand swap (B→a, A→b in row-major)
a_scale_triton = b_scale_sliced
b_scale_triton = a_scale_sliced

print(f"\nAfter swap:")
print(f"a_row_major (from B): {a_row_major.shape}")
print(f"a_scale_triton (from B): {a_scale_triton.shape}")
print(f"b_row_major (from A): {b_row_major.shape}")
print(f"b_scale_triton (from A): {b_scale_triton.shape}")

print(f"\nKernel expects:")
print(f"First operand: [{M}, {K}] with scales [{M}, {K//32}]")
print(f"Second operand: [{K}, {N}] with scales [{K//32}, {N}]")

print(f"\nWe have:")
print(f"First operand: {a_row_major.shape} with scales {a_scale_triton.shape}")
print(f"Second operand: {b_row_major.shape} with scales {b_scale_triton.shape}")

# Check scale values
print(f"\nScale values (E8M0):")
print(f"a_scale_triton: {a_scale_triton}")
print(f"b_scale_triton: {b_scale_triton}")