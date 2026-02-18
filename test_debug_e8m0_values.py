import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import mxfp8_matmul, reinterpret_as_fp8_tensor

device = torch.device("cuda")

M, N, K = 128, 256, 512

# Test with random data
torch.manual_seed(42)
a_fp32 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.randn((K, N), dtype=torch.bfloat16, device=device)

quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

print("=" * 60)
print("E8M0 Scale Analysis")
print("=" * 60)

# Check E8M0 scales for invalid values
print(f"\nA has rowwise: {a_mxfp8._rowwise_data is not None}")
print(f"A has columnwise: {a_mxfp8._columnwise_data is not None}")
print(f"B has rowwise: {b_mxfp8._rowwise_data is not None}")
print(f"B has columnwise: {b_mxfp8._columnwise_data is not None}")

a_rowwise_scale = a_mxfp8._rowwise_scale_inv if a_mxfp8._rowwise_scale_inv is not None else None
a_columnwise_scale = a_mxfp8._columnwise_scale_inv if a_mxfp8._columnwise_scale_inv is not None else None
b_rowwise_scale = b_mxfp8._rowwise_scale_inv if b_mxfp8._rowwise_scale_inv is not None else None
b_columnwise_scale = b_mxfp8._columnwise_scale_inv if b_mxfp8._columnwise_scale_inv is not None else None

print(f"\nA shapes:")
print(f"  Data (rowwise): {a_mxfp8._rowwise_data.shape if a_mxfp8._rowwise_data is not None else None}")
print(f"  Scale (rowwise): {a_rowwise_scale.shape if a_rowwise_scale is not None else None}")
print(f"  Data (columnwise): {a_mxfp8._columnwise_data.shape if a_mxfp8._columnwise_data is not None else None}")
print(f"  Scale (columnwise): {a_columnwise_scale.shape if a_columnwise_scale is not None else None}")

print(f"\nB shapes:")
print(f"  Data (rowwise): {b_mxfp8._rowwise_data.shape if b_mxfp8._rowwise_data is not None else None}")
print(f"  Scale (rowwise): {b_rowwise_scale.shape if b_rowwise_scale is not None else None}")
print(f"  Data (columnwise): {b_mxfp8._columnwise_data.shape if b_mxfp8._columnwise_data is not None else None}")
print(f"  Scale (columnwise): {b_columnwise_scale.shape if b_columnwise_scale is not None else None}")

# For the test, we're doing non-transposed, so:
# A should use rowwise, B should use columnwise
print(f"\nFor transa=False, transb=False:")
print(f"  A should use rowwise (will_transpose=False)")
print(f"  B should use columnwise (will_transpose=False)")

# NOTE: Actually I need to check the wrapper logic again...
# The wrapper gets will_transpose=transb for B
# If transb=False, will_transpose=False, so it uses rowwise
# But that seems wrong?

a_scale = a_rowwise_scale if a_rowwise_scale is not None else a_columnwise_scale
b_scale = b_rowwise_scale if b_rowwise_scale is not None else b_columnwise_scale

print(f"\nA scale shape: {a_scale.shape if a_scale is not None else None}")
print(f"B scale shape: {b_scale.shape if b_scale is not None else None}")

# Convert E8M0 to actual scale values to check for issues
a_scale_fp32 = 2.0 ** (a_scale.float() - 127.0)
b_scale_fp32 = 2.0 ** (b_scale.float() - 127.0)

print(f"\nA scale statistics:")
print(f"  E8M0 min/max: {a_scale.min().item()}, {a_scale.max().item()}")
print(f"  Scale min/max: {a_scale_fp32.min().item():.2e}, {a_scale_fp32.max().item():.2e}")
print(f"  Scales with value 0 (E8M0): {(a_scale == 0).sum().item()}")
print(f"  Scales with value 255 (E8M0): {(a_scale == 255).sum().item()}")
print(f"  Scale range (orders of magnitude): {torch.log2(a_scale_fp32.max() / a_scale_fp32.min()).item():.1f}")

print(f"\nB scale statistics:")
print(f"  E8M0 min/max: {b_scale.min().item()}, {b_scale.max().item()}")
print(f"  Scale min/max: {b_scale_fp32.min().item():.2e}, {b_scale_fp32.max().item():.2e}")
print(f"  Scales with value 0 (E8M0): {(b_scale == 0).sum().item()}")
print(f"  Scales with value 255 (E8M0): {(b_scale == 255).sum().item()}")
print(f"  Scale range (orders of magnitude): {torch.log2(b_scale_fp32.max() / b_scale_fp32.min()).item():.1f}")

# Check if scales are within reasonable range
# E8M0 should be in range [0, 255] with 127 being neutral (scale = 1.0)
# Extreme values could cause issues

# B should already have columnwise scale
b_scale_for_kernel = b_columnwise_scale

print(f"\nB columnwise scale (should match kernel expectation): {b_scale_for_kernel.shape if b_scale_for_kernel is not None else None}")
print(f"  Expected: [{K//32}, {N}] = [{K//32}, {N}]")
if b_scale_for_kernel is not None:
    print(f"  Match: {b_scale_for_kernel.shape == (K//32, N)}")

# Now check if the data has any NaN or Inf BEFORE kernel
# Use rowwise for A, columnwise for B (matching the new wrapper logic)
a_fp8 = reinterpret_as_fp8_tensor(a_mxfp8._rowwise_data, tex.DType.kFloat8E4M3)
b_fp8 = reinterpret_as_fp8_tensor(b_mxfp8._columnwise_data, tex.DType.kFloat8E4M3)

# Check FP8 data
print(f"\nFP8 data before kernel:")
a_fp32_check = a_fp8.to(torch.float32)
b_fp32_check = b_fp8.to(torch.float32)
print(f"  A has NaN: {torch.isnan(a_fp32_check).any().item()}")
print(f"  A has Inf: {torch.isinf(a_fp32_check).any().item()}")
print(f"  B has NaN: {torch.isnan(b_fp32_check).any().item()}")
print(f"  B has Inf: {torch.isinf(b_fp32_check).any().item()}")

# Run kernel
c_kernel = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

print(f"\nRunning kernel with:")
print(f"  A: rowwise data + rowwise scale")
print(f"  B: columnwise data + columnwise scale")
mxfp8_matmul(
    a_fp8, a_rowwise_scale,
    b_fp8, b_scale_for_kernel,
    c_kernel,
    M, N, K,
    tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3
)

# Check output
print(f"\nKernel output:")
print(f"  Contains NaN: {torch.isnan(c_kernel).any().item()}")
print(f"  Contains Inf: {torch.isinf(c_kernel).any().item()}")

if torch.isnan(c_kernel).any():
    nan_count = torch.isnan(c_kernel).sum().item()
    total = c_kernel.numel()
    print(f"  NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")

    # Find first NaN position
    nan_positions = torch.nonzero(torch.isnan(c_kernel))
    if len(nan_positions) > 0:
        first_nan = nan_positions[0]
        print(f"  First NaN at position: [{first_nan[0].item()}, {first_nan[1].item()}]")

# Compare with reference
ref = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
print(f"\nReference output:")
print(f"  Contains NaN: {torch.isnan(ref).any().item()}")
print(f"  Contains Inf: {torch.isinf(ref).any().item()}")

if not torch.isnan(c_kernel).any() and not torch.isnan(ref).any():
    max_diff = torch.max(torch.abs(c_kernel - ref)).item()
    print(f"\nMax difference: {max_diff:.4f}")
