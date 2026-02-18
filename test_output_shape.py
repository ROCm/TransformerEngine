import torch
from transformer_engine.pytorch.gemm_triton import getGemmOutputShape

# For NN layout
A_shape = torch.Size([128, 512])
B_shape = torch.Size([512, 256])
transa = False
transb = False

D_shape = getGemmOutputShape(A_shape, transa, B_shape, transb)
print(f"A: {A_shape}, B: {B_shape}, transa={transa}, transb={transb}")
print(f"getGemmOutputShape returned: {D_shape}")
print(f"Expected for row-major A@B: [128, 256]")
print(f"Expected for BLAS column-major: [512, 512]...?")
