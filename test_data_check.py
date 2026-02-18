import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper

device = torch.device("cuda")

M, N, K = 128, 128, 256
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

print("Checking data selection for tl.dot_scaled")
print("=" * 60)

# Use wrappers
A_wrapper = MXFP8TensorWrapper(a_mxfp8)
B_wrapper = MXFP8TensorWrapper(b_mxfp8)

# Check what we're selecting with reversed logic
# For A: use rowwise when transa=False (will_transpose=False gives rowwise)
A_data, a_scale_inv = A_wrapper.get_data_and_scale_for_gemm(will_transpose=False)
# For B: use columnwise when transb=False (will_transpose=True gives columnwise)
B_data, b_scale_inv = B_wrapper.get_data_and_scale_for_gemm(will_transpose=True)

print(f"A data shape: {A_data.shape}, scale shape: {a_scale_inv.shape}")
print(f"B data shape: {B_data.shape}, scale shape: {b_scale_inv.shape}")

print(f"\nExpected for tl.dot_scaled:")
print(f"  A scales: [M={M}, K//32={K//32}] = [{M}, {K//32}]")
print(f"  B scales: [K//32={K//32}, N={N}] = [{K//32}, {N}]")

print(f"\nActual:")
print(f"  A scales: {a_scale_inv.shape}")
print(f"  B scales: {b_scale_inv.shape}")

# Check data values
print(f"\nSample data values:")
print(f"A data [0, :5] = {A_data[0, :5].view(torch.float8_e4m3fn).to(torch.float32)}")
print(f"B data [0, :5] = {B_data[0, :5].view(torch.float8_e4m3fn).to(torch.float32)}")

print(f"\nSample scale values (E8M0):")
print(f"A scale [0, 0] = {a_scale_inv[0, 0].item()} -> 2^{a_scale_inv[0, 0].item() - 127}")
print(f"B scale [0, 0] = {b_scale_inv[0, 0].item()} -> 2^{b_scale_inv[0, 0].item() - 127}")

# Verify the data is the same as the original tensors
print(f"\nVerifying data matches original tensors:")
print(f"A rowwise data matches: {torch.equal(A_data, a_mxfp8._rowwise_data)}")
print(f"B columnwise data matches: {torch.equal(B_data, a_mxfp8._columnwise_data)}")  # Bug check
print(f"B columnwise data matches: {torch.equal(B_data, b_mxfp8._columnwise_data)}")  # Correct