import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Simple case
M, N, K = 128, 256, 512

a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("="*60)
print("ORIGINAL TENSORS (logical/mathematical view)")
print("="*60)
print(f"A: {a_mxfp8.shape} (M={M}, K={K})")
print(f"B: {b_mxfp8.shape} (K={K}, N={N})")
print(f"Expected C: ({M}, {N}) = A @ B")

print("\n" + "="*60)
print("ROWWISE DATA (PyTorch row-major storage)")
print("="*60)
print(f"A._rowwise_data: {a_mxfp8._rowwise_data.shape}")
print(f"A._rowwise_scale_inv: {a_mxfp8._rowwise_scale_inv.shape}")
print(f"B._rowwise_data: {b_mxfp8._rowwise_data.shape}")
print(f"B._rowwise_scale_inv: {b_mxfp8._rowwise_scale_inv.shape}")

print("\n" + "="*60)
print("COLUMNWISE DATA (transposed for column-major)")
print("="*60)
if hasattr(a_mxfp8, '_columnwise_data') and a_mxfp8._columnwise_data is not None:
    print(f"A._columnwise_data: {a_mxfp8._columnwise_data.shape}")
    print(f"A._columnwise_scale_inv: {a_mxfp8._columnwise_scale_inv.shape}")
else:
    print("A has no columnwise data")

if hasattr(b_mxfp8, '_columnwise_data') and b_mxfp8._columnwise_data is not None:
    print(f"B._columnwise_data: {b_mxfp8._columnwise_data.shape}")
    print(f"B._columnwise_scale_inv: {b_mxfp8._columnwise_scale_inv.shape}")
else:
    print("B has no columnwise data")

print("\n" + "="*60)
print("BLAS COLUMN-MAJOR INTERPRETATION (NN layout)")
print("="*60)
print("In BLAS column-major with NN layout:")
print("  - Tensors are logically column-major")
print("  - C = A @ B (no transposes)")
print("  - A is (M, K), B is (K, N), C is (M, N)")

print("\n" + "="*60)
print("CONVERSION TO TRITON ROW-MAJOR")
print("="*60)
print("Triton requires row-major layout, so we:")
print("  1. Swap operands: compute B^T @ A^T = (A @ B)^T")
print("  2. Transpose result back (implicitly via output shape)")
print("")
print("After swap (for NN layout, transa=False, transb=False):")
print("  a_row_major = B (no transpose)")
print("  b_row_major = A (no transpose)")
print(f"  Expected a_row_major shape: {b_mxfp8._rowwise_data.shape} = ({K}, {N})")
print(f"  Expected b_row_major shape: {a_mxfp8._rowwise_data.shape} = ({M}, {K})")
print(f"  Kernel computes: a @ b = B @ A^T")
print(f"  Wait, that's not right...")

print("\n" + "="*60)
print("LET ME RECALCULATE THE CONVERSION CORRECTLY")
print("="*60)
print("Goal: Compute C = A @ B in row-major")
print("  A: (M, K), B: (K, N), C: (M, N)")
print("")
print("Triton matmul(a, b) computes: c = a @ b in row-major")
print("  a: (m, k), b: (k, n), c: (m, n)")
print("")
print("So for NN layout (no transposes in column-major):")
print("  We want: C = A @ B")
print("  In row-major: c_rowmaj = a_rowmaj @ b_rowmaj")
print("  Where a_rowmaj = A (M, K) and b_rowmaj = B (K, N)")
print(f"  So kernel should see: a=[{M}, {K}], b=[{K}, {N}], c=[{M}, {N}]")
