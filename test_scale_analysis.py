import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper

device = torch.device("cuda")

# NT case: transA=False, transB=True
M, N, K = 128, 128, 256
transa, transb = False, True

# Create input tensors
if transa:
    a_shape = (K, M)
else:
    a_shape = (M, K)

if transb:
    b_shape = (N, K)
else:
    b_shape = (K, N)

print(f"Testing NT case: transA={transa}, transB={transb}")
print(f"A shape: {a_shape}")
print(f"B shape: {b_shape}")
print()

torch.manual_seed(42)
a_fp32 = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

# Create wrappers
A_wrapper = MXFP8TensorWrapper(a_mxfp8)
B_wrapper = MXFP8TensorWrapper(b_mxfp8)

# C++ logic selection
print("C++ selection logic:")
print(f"  A with transA={transa}: {'rowwise' if transa else 'columnwise'}")
print(f"  B with transB={transb}: {'columnwise' if transb else 'rowwise'}")
print()

# Get data and scales with C++ logic
A_data, a_scale_inv = A_wrapper.get_data_and_scale_for_gemm(will_transpose=(not transa))
B_data, b_scale_inv = B_wrapper.get_data_and_scale_for_gemm(will_transpose=transb)

print("After C++ selection:")
print(f"  A data: {A_data.shape}, scale: {a_scale_inv.shape if a_scale_inv is not None else None}")
print(f"  B data: {B_data.shape}, scale: {b_scale_inv.shape if b_scale_inv is not None else None}")
print()

# After BLAS swap for row-major
print("After BLAS swap (A,B → B,A):")
a_row_major = B_data.T if transb else B_data
b_row_major = A_data.T if transa else A_data
a_scale_triton = b_scale_inv
b_scale_triton = a_scale_inv

print(f"  a_row_major (from B): {a_row_major.shape}")
print(f"  a_scale (from B): {a_scale_triton.shape if a_scale_triton is not None else None}")
print(f"  b_row_major (from A): {b_row_major.shape}")
print(f"  b_scale (from A): {b_scale_triton.shape if b_scale_triton is not None else None}")
print()

# What tl.dot_scaled expects
print("What tl.dot_scaled expects:")
print(f"  First operand: [{a_row_major.shape[0]}, {a_row_major.shape[1]}]")
print(f"  First scale: [{a_row_major.shape[0]}, {a_row_major.shape[1]//32}]")
print(f"  Second operand: [{b_row_major.shape[0]}, {b_row_major.shape[1]}]")
print(f"  Second scale: [{b_row_major.shape[0]//32}, {b_row_major.shape[1]}]")
print()

print("Scale mismatch analysis:")
if a_scale_triton is not None:
    expected_a_scale = (a_row_major.shape[0], a_row_major.shape[1]//32)
    print(f"  First scale: have {a_scale_triton.shape}, need {expected_a_scale}")
    if a_scale_triton.shape != expected_a_scale:
        print(f"    ✗ MISMATCH!")

if b_scale_triton is not None:
    expected_b_scale = (b_row_major.shape[0]//32, b_row_major.shape[1])
    print(f"  Second scale: have {b_scale_triton.shape}, need {expected_b_scale}")
    if b_scale_triton.shape != expected_b_scale:
        print(f"    ✗ MISMATCH!")