"""
Test to understand the wgrad shape issue in Megatron-LM.
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

device = torch.device("cuda")
torch.manual_seed(42)

print("=" * 80)
print("Understanding wgrad shape issue from Megatron-LM")
print("=" * 80)

# From the error message:
# "got [4096, 2048] but expected shape compatible with [4096, 14336]"
# This means weight should be [4096, 14336] = [out_features, in_features]
out_features = 4096
in_features = 14336

print(f"\nExpected weight shape: [{out_features}, {in_features}]")

# From the debug output, we saw:
# A: [14336, 1, 2048]
# B: [2048, 4096]
# Layout: NT

# The shapes seem wrong. Let me check what makes sense:
# For wgrad with NT layout, we compute: dW = X @ dY^T (in row-major)
# where:
# - X (input): [batch*seq, in_features]
# - dY (grad_output): [batch*seq, out_features]
# - dW (weight_grad): [out_features, in_features]

# But wait, the general_gemm call is: general_gemm(inputmat_total, grad_output, layout="NT")
# So A = inputmat_total, B = grad_output

# If A = [14336, 1, 2048], this looks like it could be:
# - 14336 could be in_features (matches expected)
# - 2048 could be batch*seq
# - The middle 1 dimension is strange

# If B = [2048, 4096], this looks like:
# - 2048 could be batch*seq (matches A's last dim)
# - 4096 could be out_features (matches expected)

# So it seems like A might be a transposed and reshaped version of input

print("\nHypothesis:")
print("A[14336, 1, 2048] might be input reshaped/transposed")
print("B[2048, 4096] might be grad_output [batch*seq, out_features]")

# Let's test with the correct shapes:
batch_seq = 2048
input_correct = torch.randn(batch_seq, in_features, dtype=torch.bfloat16, device=device)
grad_output_correct = torch.randn(batch_seq, out_features, dtype=torch.bfloat16, device=device)

print(f"\nCorrect shapes for wgrad:")
print(f"Input: {input_correct.shape} = [batch*seq, in_features]")
print(f"Grad output: {grad_output_correct.shape} = [batch*seq, out_features]")

# What wgrad should compute (NT layout):
# In BLAS column-major: A @ B^T where A=input, B=grad_output
# This gives us: input @ grad_output^T
# = [batch*seq, in_features] @ [out_features, batch*seq]
# = Won't work! Dimension mismatch

# Wait, that's not right. Let me reconsider...
# Actually for wgrad, we want: dW = grad_output^T @ input
# = [out_features, batch*seq] @ [batch*seq, in_features]
# = [out_features, in_features] ✓

print("\nActual computation needed:")
print("dW = grad_output^T @ input")
print(f"   = [{out_features}, {batch_seq}] @ [{batch_seq}, {in_features}]")
print(f"   = [{out_features}, {in_features}]")

# But the BLAS call is general_gemm(input, grad_output, layout="NT")
# Which means: A=input, B=grad_output, transA=N, transB=T
# BLAS computes (column-major): A @ B^T = input @ grad_output^T
# But we want grad_output^T @ input!

print("\nThe problem:")
print("BLAS NT layout with A=input, B=grad_output computes: input @ grad_output^T")
print("But we want: grad_output^T @ input")
print("These are different!")

# Actually, let's look at the row-major perspective:
# In row-major, NT means: A^T @ B
# So with A=input, B=grad_output, we get: input^T @ grad_output
# = [in_features, batch*seq] @ [batch*seq, out_features]
# = [in_features, out_features]
# But we want [out_features, in_features], so we'd need to transpose the result

print("\nRow-major perspective:")
print("NT layout computes: input^T @ grad_output")
print(f"   = [{in_features}, {batch_seq}] @ [{batch_seq}, {out_features}]")
print(f"   = [{in_features}, {out_features}]")
print("But we want [out_features, in_features], so result needs transpose")

# The weird shapes we're seeing might be related to this confusion
# Let me check what happens if input is somehow transposed before being passed
input_weird = input_correct.T.unsqueeze(1)  # [in_features, 1, batch*seq]
print(f"\nWeird input shape like we saw: {input_weird.shape}")

# This is [14336, 1, 2048] which matches what we saw!
# So it seems like the input is being transposed before wgrad