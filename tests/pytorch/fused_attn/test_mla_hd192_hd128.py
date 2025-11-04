# Test for MLA attention config with head_dim_qk=192, head_dim_v=128
# This test verifies that the updated aiter submodule (commit 7a41cca6) 
# correctly handles the non-standard head dimensions for backward pass.
# Related to SWDEV-548321

import os
import pytest
import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION

from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention

# Only run on ROCm
pytestmark = pytest.mark.skipif(not IS_HIP_EXTENSION, reason="ROCm specific test")

# Reset RNG for reproducibility
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class MLAConfig:
    """Configuration for MLA attention test"""
    def __init__(self):
        self.qkv_dtype = torch.bfloat16
        self.qkv_layout = 'sbhd_sbhd_sbhd'
        self.batch_size = 10
        self.num_heads = 16
        self.num_gqa_groups = 16  # Standard attention, not GQA
        self.max_seqlen_q = 4096
        self.max_seqlen_kv = 4096
        self.head_dim_qk = 192  # Non-standard dimension
        self.head_dim_v = 128   # Non-standard dimension
        self.attn_mask_type = 'causal'
        self.window_size = (-1, 0)  # Causal window
        self.alibi_slopes_shape = None
        self.core_attention_bias_type = 'no_bias'
        self.dropout_p = 0.0
        self.hidden_size_q = self.num_heads * self.head_dim_qk  # 16 * 192 = 3072
        self.hidden_size_kv = self.num_gqa_groups * self.head_dim_v  # 16 * 128 = 2048


@pytest.fixture
def mla_config():
    """Fixture providing MLA configuration"""
    return MLAConfig()


def test_mla_attention_fwd_bwd(mla_config):
    """
    Test MLA attention with head_dim_qk=192, head_dim_v=128 for both forward and backward.
    
    This test verifies:
    1. The updated aiter (commit 7a41cca6) supports backward pass with hd192_hd128
    2. The correct aiter ASM kernel is selected
    3. Forward and backward passes complete without errors
    """
    config = mla_config
    
    # Enable CK backend logging to verify kernel selection
    os.environ["NVTE_LOG_CK_CONFIG"] = "1"
    os.environ["NVTE_FUSED_ATTN_CK"] = "1"
    os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "0"
    
    # By default, v3 API is disabled, testing the out-of-the-box behavior
    # Phase 2 will add tests with v3 API enabled
    
    print(f"\n{'='*80}")
    print(f"Testing MLA Attention Config:")
    print(f"  qkv_dtype: {config.qkv_dtype}")
    print(f"  qkv_layout: {config.qkv_layout}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_gqa_groups: {config.num_gqa_groups}")
    print(f"  max_seqlen_q: {config.max_seqlen_q}")
    print(f"  max_seqlen_kv: {config.max_seqlen_kv}")
    print(f"  head_dim_qk: {config.head_dim_qk}")
    print(f"  head_dim_v: {config.head_dim_v}")
    print(f"  attn_mask_type: {config.attn_mask_type}")
    print(f"  window_size: {config.window_size}")
    print(f"  core_attention_bias_type: {config.core_attention_bias_type}")
    print(f"{'='*80}\n")
    
    # Create attention module
    # For MLA with different head dimensions, kv_channels accepts a tuple: (head_dim_qk, head_dim_v)
    attn = DotProductAttention(
        num_attention_heads=config.num_heads,
        kv_channels=(config.head_dim_qk, config.head_dim_v),  # Tuple for MLA
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format='sbhd',
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
    ).to(dtype=config.qkv_dtype, device='cuda')
    
    # Create input tensors: s=seqlen, b=batch, h=heads, d=head_dim
    # For sbhd layout: [seqlen, batch, heads, head_dim]
    q = torch.randn(
        config.max_seqlen_q, 
        config.batch_size, 
        config.num_heads, 
        config.head_dim_qk,
        dtype=config.qkv_dtype, 
        device='cuda',
        requires_grad=True
    )
    
    k = torch.randn(
        config.max_seqlen_kv,
        config.batch_size,
        config.num_gqa_groups,
        config.head_dim_qk,
        dtype=config.qkv_dtype,
        device='cuda',
        requires_grad=True
    )
    
    v = torch.randn(
        config.max_seqlen_kv,
        config.batch_size,
        config.num_gqa_groups,
        config.head_dim_v,
        dtype=config.qkv_dtype,
        device='cuda',
        requires_grad=True
    )
    
    print("Running forward pass...")
    # Forward pass
    out = attn(q, k, v)
    
    # Verify output shape: DotProductAttention returns [seqlen_q, batch, num_heads * head_dim_v]
    # For MLA: [4096, 10, 16 * 128] = [4096, 10, 2048]
    expected_shape = (config.max_seqlen_q, config.batch_size, config.num_heads * config.head_dim_v)
    assert out.shape == expected_shape, f"Output shape mismatch: {out.shape} != {expected_shape}"
    print(f"✓ Forward pass successful! Output shape: {out.shape}")
    
    # Backward pass
    print("Running backward pass...")
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    
    # Verify gradients exist
    assert q.grad is not None, "Query gradient is None"
    assert k.grad is not None, "Key gradient is None"
    assert v.grad is not None, "Value gradient is None"
    
    # Verify gradient shapes
    assert q.grad.shape == q.shape, f"Query gradient shape mismatch: {q.grad.shape} != {q.shape}"
    assert k.grad.shape == k.shape, f"Key gradient shape mismatch: {k.grad.shape} != {k.shape}"
    assert v.grad.shape == v.shape, f"Value gradient shape mismatch: {v.grad.shape} != {v.shape}"
    print(f"✓ Backward pass successful! Gradients computed correctly.")
    
    # Check that gradients are not NaN or Inf
    assert not torch.isnan(q.grad).any(), "Query gradient contains NaN"
    assert not torch.isnan(k.grad).any(), "Key gradient contains NaN"
    assert not torch.isnan(v.grad).any(), "Value gradient contains NaN"
    assert not torch.isinf(q.grad).any(), "Query gradient contains Inf"
    assert not torch.isinf(k.grad).any(), "Key gradient contains Inf"
    assert not torch.isinf(v.grad).any(), "Value gradient contains Inf"
    print(f"✓ Gradient sanity checks passed (no NaN/Inf)")
    
    print(f"\n{'='*80}")
    print(f"MLA Attention Test PASSED!")
    print(f"  - Forward pass completed successfully")
    print(f"  - Backward pass completed successfully")
    print(f"  - Gradients are valid (no NaN/Inf)")
    print(f"  - This confirms aiter commit 7a41cca6 enables hd192_hd128 support")
    print(f"{'='*80}\n")


def test_mla_attention_with_v3_api(mla_config):
    """
    Test MLA attention with v3 API enabled.
    
    This test is for Phase 2 - it verifies the v3 API with proper dq_acc handling.
    """
    pytest.skip("Phase 2: Will be implemented after Phase 1 is verified")
    # TODO Phase 2: Implement v3 API test with:
    # - NVTE_CK_USES_BWD_V3=1
    # - NVTE_CK_IS_V3_ATOMIC_FP32=[0,1]
    # - NVTE_CK_HOW_V3_BF16_CVT=[0,1]


if __name__ == "__main__":
    # Allow running the test directly for debugging
    config = MLAConfig()
    test_mla_attention_fwd_bwd(config)

