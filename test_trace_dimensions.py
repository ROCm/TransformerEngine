import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")

# Create specific test pattern
M, N, K = 128, 256, 512

# Create matrices with specific patterns to trace
a_fp32 = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
b_fp32 = torch.zeros((K, N), dtype=torch.bfloat16, device=device)

# Fill A with row index, B with column index
for i in range(M):
    a_fp32[i, :] = float(i % 10)  # Row pattern
for j in range(N):
    b_fp32[:, j] = float(j % 10)  # Column pattern

print("Original patterns:")
print(f"A[0, :5] = {a_fp32[0, :5]}")  # Should be all 0s
print(f"A[1, :5] = {a_fp32[1, :5]}")  # Should be all 1s
print(f"B[:5, 0] = {b_fp32[:5, 0]}")  # Should be all 0s
print(f"B[:5, 1] = {b_fp32[:5, 1]}")  # Should be all 1s

# Expected result: C[i,j] = i * j * K (since we're summing K copies of i*j)
expected = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
for i in range(M):
    for j in range(N):
        expected[i, j] = float((i % 10) * (j % 10) * K)

print(f"\nExpected C[0, 0] = {expected[0, 0].item()}")  # 0*0*512 = 0
print(f"Expected C[1, 1] = {expected[1, 1].item()}")  # 1*1*512 = 512
print(f"Expected C[2, 3] = {expected[2, 3].item()}")  # 2*3*512 = 3072

# Now test with actual matmul
ref = torch.matmul(a_fp32, b_fp32)
print(f"\nReference C[0, 0] = {ref[0, 0].item()}")
print(f"Reference C[1, 1] = {ref[1, 1].item()}")
print(f"Reference C[2, 3] = {ref[2, 3].item()}")

# Check if reference matches expected
print(f"\nReference matches expected: {torch.allclose(ref, expected, rtol=0.01)}")

# Now test with MXFP8
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

a_mxfp8 = quantizer.quantize(a_fp32)
b_mxfp8 = quantizer.quantize(b_fp32)

# Test with dequantize
dequant_result = torch.matmul(a_mxfp8.dequantize(), b_mxfp8.dequantize())
print(f"\nDequant C[0, 0] = {dequant_result[0, 0].item()}")
print(f"Dequant C[1, 1] = {dequant_result[1, 1].item()}")
print(f"Dequant C[2, 3] = {dequant_result[2, 3].item()}")

# Test what the kernel would compute
# After BLAS swap, first operand comes from B, second from A
# So kernel computes: B @ A in row-major
# But B is [K, N] and A is [M, K] in original shapes
# After BLAS logic and swap, what dimensions does the kernel see?