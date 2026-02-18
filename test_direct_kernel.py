import torch
import triton
import triton.language as tl

@triton.jit
def simple_mxfp8_kernel(
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_k, stride_b_scale_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Simple test kernel - just compute one block
    pid = 0

    # Load data
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    # Load scales
    VEC_SIZE = 32
    offs_scale_k = tl.arange(0, BLOCK_K // VEC_SIZE)

    a_scale_ptrs = a_scale_ptr + offs_m[:, None] * stride_a_scale_m + offs_scale_k[None, :] * stride_a_scale_k
    b_scale_ptrs = b_scale_ptr + offs_scale_k[:, None] * stride_b_scale_k + offs_n[None, :] * stride_b_scale_n

    a_scale_e8m0 = tl.load(a_scale_ptrs)
    b_scale_e8m0 = tl.load(b_scale_ptrs)

    # Compute
    acc = tl.dot_scaled(
        a, a_scale_e8m0, "e4m3",
        b, b_scale_e8m0, "e4m3",
        tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    )

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16))

device = torch.device("cuda")

# Simple test case
M = N = K = 64

# Create simple data - all ones
a_data = torch.ones((M, K), dtype=torch.float8_e4m3fn, device=device)
b_data = torch.ones((K, N), dtype=torch.float8_e4m3fn, device=device)

# Create scales - all 127 (scale = 1.0 in E8M0 format)
a_scale = torch.full((M, K//32), 127, dtype=torch.uint8, device=device)
b_scale = torch.full((K//32, N), 127, dtype=torch.uint8, device=device)

c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

print("Simple test: all ones with scale=1.0")
print(f"A: {a_data.shape}, scale: {a_scale.shape}")
print(f"B: {b_data.shape}, scale: {b_scale.shape}")
print(f"Expected: all values should be {K}")

# Run kernel
simple_mxfp8_kernel[(1,)](
    a_data, a_scale,
    b_data, b_scale,
    c,
    M, N, K,
    a_data.stride(0), a_data.stride(1),
    b_data.stride(0), b_data.stride(1),
    c.stride(0), c.stride(1),
    a_scale.stride(0), a_scale.stride(1),
    b_scale.stride(0), b_scale.stride(1),
    BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,
)

print(f"\nResult:")
print(f"c[0,0] = {c[0,0].item()}")
print(f"c[0,1] = {c[0,1].item()}")
print(f"c[31,31] = {c[31,31].item()}")
print(f"Min = {c.min().item()}, Max = {c.max().item()}")

if abs(c[0,0].item() - K) < 0.1:
    print("✓ Kernel works correctly!")
else:
    print(f"✗ Kernel failed! Expected {K}, got {c[0,0].item()}")