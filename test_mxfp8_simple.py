import torch
import triton
import triton.language as tl
from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

@triton.jit
def simple_mxfp8_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_k, stride_b_scale_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Just try one block
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load data
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

    # Load scales
    offs_k_scale = tl.arange(0, BLOCK_K // VEC_SIZE)
    a_scale_ptrs = a_scale_ptr + offs_m[:, None] * stride_a_scale_m + offs_k_scale[None, :] * stride_a_scale_k
    b_scale_ptrs = b_scale_ptr + offs_k_scale[:, None] * stride_b_scale_k + offs_n[None, :] * stride_b_scale_n

    a_scale = tl.load(a_scale_ptrs, mask=(offs_m[:, None] < M) & (offs_k_scale[None, :] < (K // VEC_SIZE)), other=127)
    b_scale = tl.load(b_scale_ptrs, mask=(offs_k_scale[:, None] < (K // VEC_SIZE)) & (offs_n[None, :] < N), other=127)

    # dot_scaled
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    result = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def test_mxfp8():
    device = torch.device("cuda")
    M, N, K = 128, 256, 512
    VEC_SIZE = MXFP8_BLOCK_SCALING_SIZE  # 32

    # Detect FP8 dtype
    major, minor = torch.cuda.get_device_capability()
    fp8_dtype = torch.float8_e4m3fn if (major == 9 and minor >= 5) else torch.float8_e4m3fnuz

    # Create test data
    a = torch.rand((M, K), dtype=torch.float32, device=device).to(fp8_dtype)
    b = torch.rand((K, N), dtype=torch.float32, device=device).to(fp8_dtype)
    c = torch.zeros((M, N), dtype=torch.float32, device=device)

    # Create E8M0 scales
    a_scale = torch.randint(120, 135, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(120, 135, (K // VEC_SIZE, N), dtype=torch.uint8, device=device)

    print(f"a: {a.shape}, {a.dtype}, stride={a.stride()}")
    print(f"b: {b.shape}, {b.dtype}, stride={b.stride()}")
    print(f"a_scale: {a_scale.shape}, {a_scale.dtype}, stride={a_scale.stride()}")
    print(f"b_scale: {b_scale.shape}, {b_scale.dtype}, stride={b_scale.stride()}")

    grid = (1,)
    try:
        simple_mxfp8_kernel[grid](
            a, b, c,
            a_scale, b_scale,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            b_scale.stride(0), b_scale.stride(1),
            BLOCK_M=128, BLOCK_N=256, BLOCK_K=512,
            VEC_SIZE=VEC_SIZE,
        )
        print("✓ MXFP8 kernel succeeded!")
        return True
    except Exception as e:
        print(f"✗ MXFP8 kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mxfp8()
    exit(0 if success else 1)
