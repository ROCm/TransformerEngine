import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

M, N, K = 128, 256, 512
transa, transb = False, False

print("=" * 60)
print(f"Testing MXFP8 with C++ logic: M={M}, N={N}, K={K}")
print(f"transa={transa}, transb={transb}")
print("=" * 60)

# Create tensors
torch.manual_seed(42)
if transa:
    a_shape = (K, M)
else:
    a_shape = (M, K)

if transb:
    b_shape = (N, K)
else:
    b_shape = (K, N)

a_fp32 = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn(b_shape, dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print(f"\nA shape: {a_shape}")
print(f"B shape: {b_shape}")

# According to C++ logic:
# For A with transa=False: Use columnwise
# For B with transb=False: Use rowwise

print(f"\nC++ logic would select:")
print(f"  A: {'columnwise' if not transa else 'rowwise'}")
print(f"  B: {'rowwise' if not transb else 'columnwise'}")

# Check what we have
print(f"\nA quantization available:")
print(f"  Rowwise: {a_mxfp8._rowwise_data is not None}")
print(f"  Columnwise: {a_mxfp8._columnwise_data is not None}")
if a_mxfp8._rowwise_data is not None:
    print(f"    Rowwise data: {a_mxfp8._rowwise_data.shape}")
    print(f"    Rowwise scale: {a_mxfp8._rowwise_scale_inv.shape}")
if a_mxfp8._columnwise_data is not None:
    print(f"    Columnwise data: {a_mxfp8._columnwise_data.shape}")
    print(f"    Columnwise scale: {a_mxfp8._columnwise_scale_inv.shape}")

print(f"\nB quantization available:")
print(f"  Rowwise: {b_mxfp8._rowwise_data is not None}")
print(f"  Columnwise: {b_mxfp8._columnwise_data is not None}")
if b_mxfp8._rowwise_data is not None:
    print(f"    Rowwise data: {b_mxfp8._rowwise_data.shape}")
    print(f"    Rowwise scale: {b_mxfp8._rowwise_scale_inv.shape}")
if b_mxfp8._columnwise_data is not None:
    print(f"    Columnwise data: {b_mxfp8._columnwise_data.shape}")
    print(f"    Columnwise scale: {b_mxfp8._columnwise_scale_inv.shape}")

# After column-major to row-major swap
print(f"\nAfter BLAS swap (A,B → B,A):")
print(f"  First operand (a_row_major) comes from B")
print(f"  Second operand (b_row_major) comes from A")

# For the kernel, we need:
# First operand: [M, K] with scales [M, K//32]
# Second operand: [K, N] with scales [K//32, N]

print(f"\nFor Triton kernel:")
print(f"  First operand needs: [{M}, {K}] with scales [{M}, {K//32}]")
print(f"  Second operand needs: [{K}, {N}] with scales [{K//32}, {N}]")

# Check what we'd get:
# First operand comes from B (rowwise)
print(f"\nB rowwise → first operand:")
print(f"  Data: {b_mxfp8._rowwise_data.shape if b_mxfp8._rowwise_data is not None else None}")
print(f"  Scale: {b_mxfp8._rowwise_scale_inv.shape if b_mxfp8._rowwise_scale_inv is not None else None}")

# Second operand comes from A (columnwise)
print(f"\nA columnwise → second operand:")
print(f"  Data: {a_mxfp8._columnwise_data.shape if a_mxfp8._columnwise_data is not None else None}")
print(f"  Scale: {a_mxfp8._columnwise_scale_inv.shape if a_mxfp8._columnwise_scale_inv is not None else None}")

print(f"\nProblem: Data shapes don't match kernel needs after swap!")
print(f"  B is [{K}, {N}] but kernel needs [{M}, {K}]")
print(f"  A is [{M}, {K}] but kernel needs [{K}, {N}]")