import torch
import triton
import triton.language as tl

device = torch.device("cuda")

# Simple test to understand what tl.dot_scaled expects
M, N, K = 64, 64, 64
VEC_SIZE = 32

# Detect FP8 dtype
major, minor = torch.cuda.get_device_capability()
fp8_dtype = torch.float8_e4m3fn if (major == 9 and minor >= 5) else torch.float8_e4m3fnuz

# Create simple FP8 data (all same value for testing)
a_fp8 = (torch.ones((M, K), dtype=torch.float32, device=device) * 0.5).to(fp8_dtype)
b_fp8 = (torch.ones((K, N), dtype=torch.float32, device=device) * 0.5).to(fp8_dtype)
c = torch.zeros((M, N), dtype=torch.float32, device=device)

# Test 1: E8M0 scales (uint8)
a_scale_e8m0 = torch.full((M, K // VEC_SIZE), 118, dtype=torch.uint8, device=device)
b_scale_e8m0 = torch.full((K // VEC_SIZE, N), 118, dtype=torch.uint8, device=device)

# Test 2: FP32 scales (converted)
# scale = 2^(118 - 127) = 2^(-9)
scale_value = 2.0 ** (118 - 127)
a_scale_fp32 = torch.full((M, K // VEC_SIZE), scale_value, dtype=torch.float32, device=device)
b_scale_fp32 = torch.full((K // VEC_SIZE, N), scale_value, dtype=torch.float32, device=device)

print(f"E8M0 scale value: 118")
print(f"Converted FP32 scale: {scale_value}")
print(f"Expected output (0.5 * 0.5 * 64): {0.5 * 0.5 * K}")

@triton.jit
def test_e8m0_kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # Single block
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])

    offs_k_scale = tl.arange(0, K // 32)
    a_scale = tl.load(a_scale_ptr + offs_m[:, None] * (K // 32) + offs_k_scale[None, :])
    b_scale = tl.load(b_scale_ptr + offs_k_scale[:, None] * N + offs_n[None, :])

    acc = tl.zeros((M, N), dtype=tl.float32)
    result = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, result)

# Test with E8M0
print("\nTest 1: E8M0 scales (uint8)")
c.zero_()
try:
    test_e8m0_kernel[(1,)](a_fp8, b_fp8, c, a_scale_e8m0, b_scale_e8m0, M, N, K)
    print(f"  Result: {c[0, 0].item():.4f}")
    print(f"  Full range: [{c.min().item():.4f}, {c.max().item():.4f}]")
except Exception as e:
    print(f"  Error: {e}")

# Test with FP32
print("\nTest 2: FP32 scales")
c.zero_()
try:
    test_e8m0_kernel[(1,)](a_fp8, b_fp8, c, a_scale_fp32, b_scale_fp32, M, N, K)
    print(f"  Result: {c[0, 0].item():.4f}")
    print(f"  Full range: [{c.min().item():.4f}, {c.max().item():.4f}]")
except Exception as e:
    print(f"  Error: {e}")

# Compute reference
a_fp32_ref = a_fp8.to(torch.float32)
b_fp32_ref = b_fp8.to(torch.float32)
ref = torch.matmul(a_fp32_ref, b_fp32_ref)
print(f"\nReference (FP8→FP32, no scaling): {ref[0, 0].item():.4f}")
print(f"  Range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")

# Reference with manual scaling
ref_scaled = ref * scale_value * scale_value
print(f"Reference with manual scale application: {ref_scaled[0, 0].item():.4f}")
