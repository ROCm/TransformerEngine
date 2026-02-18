import torch
import triton
import triton.language as tl

# Minimal test kernel to understand tl.dot_scaled behavior
@triton.jit
def test_dot_scaled_kernel(
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Load one block
    pid_m = 0
    pid_n = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load data
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    # Load scales
    VEC_SIZE = 32
    offs_scale_k = tl.arange(0, BLOCK_K // VEC_SIZE)

    a_scale_ptrs = a_scale_ptr + offs_m[:, None] * (K // VEC_SIZE) + offs_scale_k[None, :]
    b_scale_ptrs = b_scale_ptr + offs_scale_k[:, None] * N + offs_n[None, :]

    a_scale = tl.load(a_scale_ptrs)
    b_scale = tl.load(b_scale_ptrs)

    # Compute with tl.dot_scaled
    c = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)

# Test
device = torch.device("cuda")
BLOCK_SIZE = 64
M = N = K = BLOCK_SIZE

# Create simple test data
a = torch.ones((M, K), dtype=torch.float8_e4m3fn, device=device)
b = torch.ones((K, N), dtype=torch.float8_e4m3fn, device=device)

# Create scales (E8M0 format, 127 = scale of 1.0)
a_scale = torch.full((M, K//32), 127, dtype=torch.uint8, device=device)
b_scale = torch.full((K//32, N), 127, dtype=torch.uint8, device=device)

c = torch.zeros((M, N), dtype=torch.float32, device=device)

print("Testing tl.dot_scaled with simple data")
print(f"A: all ones, shape {a.shape}")
print(f"B: all ones, shape {b.shape}")
print(f"A scale: all 127 (scale=1.0), shape {a_scale.shape}")
print(f"B scale: all 127 (scale=1.0), shape {b_scale.shape}")
print(f"Expected result: all {K} (since 1*1*K = K)")

# Run kernel
grid = (1,)
test_dot_scaled_kernel[grid](
    a, a_scale,
    b, b_scale,
    c,
    M, N, K,
    BLOCK_M=M, BLOCK_N=N, BLOCK_K=K
)

print(f"\nActual result:")
print(f"c[0,0] = {c[0,0].item()}")
print(f"c[0,1] = {c[0,1].item()}")
print(f"c min = {c.min().item()}, max = {c.max().item()}")