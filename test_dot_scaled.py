import torch
import triton
import triton.language as tl

@triton.jit
def test_dot_scaled_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    FP8_FORMAT: tl.constexpr,  # Format string for tl.dot_scaled
):
    pid = tl.program_id(0)

    # Simple single block test
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load data
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    # Load scales (E8M0 format - uint8)
    offs_k_scale = tl.arange(0, BLOCK_K // VEC_SIZE)
    a_scale_ptrs = a_scale_ptr + offs_m[:, None] * (K // VEC_SIZE) + offs_k_scale[None, :]
    b_scale_ptrs = b_scale_ptr + offs_k_scale[:, None] * N + offs_n[None, :]
    a_scale_e8m0 = tl.load(a_scale_ptrs)  # uint8 E8M0 scales
    b_scale_e8m0 = tl.load(b_scale_ptrs)  # uint8 E8M0 scales

    # Try dot_scaled with E8M0 scales (uint8)
    # tl.dot_scaled should handle E8M0 -> FP32 conversion internally
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    result = tl.dot_scaled(a, a_scale_e8m0, FP8_FORMAT, b, b_scale_e8m0, FP8_FORMAT, acc)

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, result)


def test_dot_scaled():
    device = torch.device("cuda")
    M, N, K = 128, 128, 128
    VEC_SIZE = 32

    # Detect FP8 dtype for this architecture
    major, minor = torch.cuda.get_device_capability()
    if major == 9 and minor >= 5:
        fp8_dtype = torch.float8_e4m3fn  # OCP format for gfx950
    else:
        fp8_dtype = torch.float8_e4m3fnuz  # NANOO format for gfx942

    # Try just "e4m3" as format string - Triton might abstract away the fn/fnuz difference
    fp8_format_str = "e4m3"

    # Create test data
    a = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device).to(fp8_dtype)
    b = torch.randint(0, 255, (K, N), dtype=torch.uint8, device=device).to(fp8_dtype)
    c = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Create scales in E8M0 format (uint8 biased exponents)
    # E8M0: scale = 2^(biased_exp - 127), biased_exp in [0, 255]
    # Use range [120, 135] for reasonable scale values around 1.0
    a_scale = torch.randint(120, 135, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(120, 135, (K // VEC_SIZE, N), dtype=torch.uint8, device=device)

    print(f"Testing dot_scaled on {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print(f"a: {a.shape}, {a.dtype}")
    print(f"b: {b.shape}, {b.dtype}")
    print(f"a_scale (E8M0): {a_scale.shape}, {a_scale.dtype}")
    print(f"b_scale (E8M0): {b_scale.shape}, {b_scale.dtype}")

    grid = (1,)
    try:
        test_dot_scaled_kernel[grid](
            a, b, c,
            a_scale, b_scale,
            M, N, K,
            BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,
            VEC_SIZE=VEC_SIZE,
            FP8_FORMAT=fp8_format_str,
        )
        print(f"✓ dot_scaled succeeded with format '{fp8_format_str}'!")
        return True
    except Exception as e:
        print(f"✗ dot_scaled failed with format '{fp8_format_str}': {e}")
        return False


if __name__ == "__main__":
    success = test_dot_scaled()
    exit(0 if success else 1)
