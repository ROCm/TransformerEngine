"""
Debug the wgrad shape issue by tracing through our logic.
"""

import torch

def product(shape):
    ret = 1
    for i in shape:
        ret *= i
    return ret

def getGemmOutputShape(A, transa, B, transb):
    """Test our getGemmOutputShape logic."""
    # Handle both tensors and torch.Size objects
    A_shape = A if isinstance(A, torch.Size) else A.shape
    B_shape = B if isinstance(B, torch.Size) else B.shape

    # Calculate flattened dimensions (product of all leading dims)
    A0 = product(A_shape[:-1])  # Product of all leading dims
    A1 = A_shape[-1]
    B0 = product(B_shape[:-1])
    B1 = B_shape[-1]

    ret = []

    # First part: from B
    if transb:
        ret.append(B1)
    else:
        # Preserve B's batch structure (all dims except last)
        for i in range(len(B_shape) - 1):
            ret.append(B_shape[i])

    # Second part: from A
    if transa:
        ret.append(A0)  # Flattened A
    else:
        ret.append(A1)  # A's last dim

    return torch.Size(ret)

# Test case 1: What we expect for wgrad
print("=" * 80)
print("Test 1: Expected wgrad with 3D tensors")
print("=" * 80)

# Input: [batch, seq_len, in_features]
input_shape = torch.Size([2, 2048, 14336])
# Grad output: [batch, seq_len, out_features]
grad_output_shape = torch.Size([2, 2048, 4096])

# For wgrad with NT layout (transa=False, transb=True)
# After swapping for row-major:
# - First operand (was B): grad_output
# - Second operand (was A): input

print(f"Input (A): {input_shape}")
print(f"Grad output (B): {grad_output_shape}")
print(f"Layout: NT (transa=False, transb=True)")
print(f"After swap: B becomes first, A becomes second")

result = getGemmOutputShape(input_shape, False, grad_output_shape, True)
print(f"Result shape: {result}")
print(f"Expected: [4096, 14336]")

# Test case 2: What if input had weird shape
print("\n" + "=" * 80)
print("Test 2: Weird input shape like we saw in error")
print("=" * 80)

# What if input was somehow [in_features, batch, seq_len]?
weird_input_shape = torch.Size([14336, 2, 2048])
# And grad_output was [seq_len, out_features]?
weird_grad_shape = torch.Size([2048, 4096])

print(f"Weird input (A): {weird_input_shape}")
print(f"Weird grad (B): {weird_grad_shape}")
print(f"Layout: NT (transa=False, transb=True)")

result2 = getGemmOutputShape(weird_input_shape, False, weird_grad_shape, True)
print(f"Result shape: {result2}")
print(f"We got: [4096, 2048] - THIS MATCHES THE ERROR!")

# So the problem is that the inputs have wrong shapes!
# Input is [14336, batch, seq_len] instead of [batch, seq_len, 14336]
# Grad output is [seq_len, out_features] instead of [batch, seq_len, out_features]

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)
print("The issue is that the tensors are being passed with incorrect shapes:")
print("- Input should be [batch, seq_len, in_features] = [2, 2048, 14336]")
print("  but appears to be [in_features, batch, seq_len] = [14336, 2, 2048]")
print("- Grad output should be [batch, seq_len, out_features] = [2, 2048, 4096]")
print("  but appears to be [seq_len, out_features] = [2048, 4096]")
print("\nThis suggests that:")
print("1. Input might be getting transposed before being passed")
print("2. Grad output might be missing the batch dimension")

# Test case 3: What about the actual shapes from debug output?
print("\n" + "=" * 80)
print("Test 3: Actual shapes from debug output (batch=1)")
print("=" * 80)

# From debug output we saw:
# A: [14336, 1, 2048]
# B: [2048, 4096]

actual_A = torch.Size([14336, 1, 2048])
actual_B = torch.Size([2048, 4096])

print(f"Actual A: {actual_A}")
print(f"Actual B: {actual_B}")
print(f"Layout: NT (transa=False, transb=True)")

result3 = getGemmOutputShape(actual_A, False, actual_B, True)
print(f"Result shape: {result3}")
print(f"Error reported: [4096, 2048]")
print(f"Match: {result3 == torch.Size([4096, 2048])}")