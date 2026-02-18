"""
Test how columnwise MXFP8 tensors are handled.
"""

import torch
import os
os.environ["NVTE_USE_GEMM_TRITON"] = "1"
os.environ["NVTE_ROCM_ENABLE_MXFP8"] = "1"
os.environ["DEBUG_MXFP8_SELECT"] = "1"

from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

device = torch.device("cuda")
torch.manual_seed(42)

print("=" * 80)
print("Testing columnwise MXFP8 tensor shapes")
print("=" * 80)

# Create a 3D tensor
batch = 2
seq_len = 2048
in_features = 14336

tensor_3d = torch.randn(batch, seq_len, in_features, dtype=torch.bfloat16, device=device)
print(f"\nOriginal tensor shape: {tensor_3d.shape}")

# Quantize to MXFP8
quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)

mxfp8_tensor = quantizer.quantize(tensor_3d)

print(f"\nMXFP8 storage:")
print(f"  Rowwise data shape: {mxfp8_tensor._rowwise_data.shape}")
print(f"  Columnwise data shape: {mxfp8_tensor._columnwise_data.shape}")
print(f"  Same shape? {mxfp8_tensor._rowwise_data.shape == mxfp8_tensor._columnwise_data.shape}")

# Now let's see what happens when we use columnwise data
print(f"\n" + "=" * 80)
print("Testing MXFP8TensorWrapper behavior")
print("=" * 80)

from transformer_engine.pytorch.gemm_triton import MXFP8TensorWrapper

wrapper = MXFP8TensorWrapper(mxfp8_tensor)
print(f"Wrapper size(): {wrapper.size()}")
print(f"Wrapper._size: {wrapper._size}")

# Check what data is selected for different transpose flags
print(f"\n" + "-" * 40)
print("Data selection for different transpose flags:")

# For wgrad, input uses transA=False (needs columnwise)
data_no_trans, scale_no_trans = wrapper.get_data_and_scale_for_gemm(will_transpose=False)
print(f"\nwill_transpose=False (rowwise):")
print(f"  Data shape: {data_no_trans.shape if data_no_trans is not None else 'None'}")
print(f"  Scale shape: {scale_no_trans.shape if scale_no_trans is not None else 'None'}")

data_trans, scale_trans = wrapper.get_data_and_scale_for_gemm(will_transpose=True)
print(f"\nwill_transpose=True (columnwise):")
print(f"  Data shape: {data_trans.shape if data_trans is not None else 'None'}")
print(f"  Scale shape: {scale_trans.shape if scale_trans is not None else 'None'}")

# Now test what happens when only columnwise is available
print(f"\n" + "=" * 80)
print("Testing with only columnwise data")
print("=" * 80)

# Set columnwise usage only
mxfp8_tensor._rowwise_data = None
mxfp8_tensor._rowwise_scale_inv = None

wrapper_col_only = MXFP8TensorWrapper(mxfp8_tensor)
print(f"Wrapper size() with only columnwise: {wrapper_col_only.size()}")

# The issue might be in how the wrapper determines size from columnwise data
# Let's check the logic in lines 366-375 of gemm_triton.py
print(f"\nColumnwise data dimensions:")
print(f"  ndim: {mxfp8_tensor._columnwise_data.dim()}")
if mxfp8_tensor._columnwise_data.dim() == 3:
    print(f"  Shape: {mxfp8_tensor._columnwise_data.shape}")
    # The logic for 3D: batch_dims + [m_dim, k_dim]
    batch_dims = list(mxfp8_tensor._columnwise_data.size()[2:])
    m_dim = mxfp8_tensor._columnwise_data.size(1)
    k_dim = mxfp8_tensor._columnwise_data.size(0)
    reconstructed_size = torch.Size(batch_dims + [m_dim, k_dim])
    print(f"  Reconstructed size (current logic): {reconstructed_size}")
    print(f"  This is WRONG! Should be: {tensor_3d.shape}")

    # The correct logic should be different for MXFP8
    # Since columnwise is NOT transposed for MXFP8
    print(f"\n  Correct interpretation for MXFP8:")
    print(f"  Columnwise shape is same as rowwise: {mxfp8_tensor._columnwise_data.shape}")
    print(f"  So size should just be: {mxfp8_tensor._columnwise_data.shape}")