import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.gemm_triton import te_gemm_triton, torch_to_te_dtype

# Use test dimensions from test_gemm_triton.py
K, M = 768, 768  # a is (K, M) in PyTorch
N, K2 = 4096, 768  # b is (N, K) in PyTorch
assert K == K2

device = torch.device("cuda")

# Create regular tensors following the test pattern
a_fp32 = torch.randn((K, M), dtype=torch.float32, device=device)
b_fp32 = torch.randn((N, K), dtype=torch.float32, device=device)

# Convert to bf16
a = a_fp32.to(torch.bfloat16)
b = b_fp32.to(torch.bfloat16)

# Reference output (from test line 176)
torch_output = torch.matmul(b_fp32, a_fp32)  # (N, K) @ (K, M) = (N, M)
print(f"Reference output shape: {torch_output.shape}")
print(f"Expected: (N, M) = ({N}, {M}) = (4096, 768)")

# Now call te_gemm_triton with NN layout (col_a=False, col_b=False)
transa = False
transb = False

c = torch.empty((N, M), device=device, dtype=torch.bfloat16)

# Call the low-level function
te_gemm_triton(
    A=a,  # (K, M) = (768, 768)
    A_scale_inverse=torch.Tensor(),
    A_fp8_tensor=0,
    A_type=torch_to_te_dtype(torch.bfloat16),
    transa=transa,
    B=b,  # (N, K) = (4096, 768)
    B_scale_inverse=torch.Tensor(),
    B_fp8_tensor=0,
    B_type=torch_to_te_dtype(torch.bfloat16),
    transb=transb,
    D=c,  # (N, M) = (4096, 768)
    D_scale=torch.Tensor(),
    D_type=torch_to_te_dtype(torch.bfloat16),
    D_amax=torch.Tensor(),
    bias=torch.Tensor(),
    bias_type=torch_to_te_dtype(torch.bfloat16),
    pre_gelu_out=torch.Tensor(),
    grad=False,
    workspace=torch.Tensor(),
    workspaceSize=0,
    accumulate=False,
    use_split_accumulator=False
)

print(f"\nte_gemm_triton output shape: {c.shape}")
print(f"Output matches reference shape: {c.shape == torch_output.shape}")

# Check numerical correctness
if c.shape == torch_output.shape:
    max_diff = torch.max(torch.abs(c.float() - torch_output)).item()
    print(f"Max difference: {max_diff}")
else:
    print("Shapes don't match!")
