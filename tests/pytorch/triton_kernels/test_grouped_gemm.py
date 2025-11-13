# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import pytest
import torch
from torch.nn.parameter import Parameter

from transformer_engine.pytorch.module import GroupedLinear, Linear
from test_common import te_compare_results, fill_uniform, get_tolerances, dtype_tols


@pytest.mark.parametrize("num_gemms", [2, 3, 4, 6])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 512, 2048])
@pytest.mark.parametrize("hidden_size", [256, 768])
@pytest.mark.parametrize("out_features_multiplier", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("equal_splits", [True, False])
def test_grouped_gemm_triton_forward(
    num_gemms, bs, seq_len, hidden_size, out_features_multiplier, dtype, equal_splits
):
    """Test Triton grouped GEMM kernel against hip for forward pass."""
    out_features = hidden_size * out_features_multiplier
    total_tokens = seq_len * bs
    
    # Create m_splits
    if equal_splits:
        base_split = total_tokens // num_gemms
        m_splits = [base_split] * num_gemms
        # Adjust last split to handle remainders
        m_splits[-1] = total_tokens - sum(m_splits[:-1])
    else:
        # Random unequal splits (like the actual test does)
        torch.manual_seed(42)  # For reproducibility
        dist = torch.sort(torch.randint(0, total_tokens, (num_gemms - 2,))).values.tolist()
        dist.append(dist[-1] if dist else 0)
        m_splits = (torch.tensor(dist + [total_tokens]) - torch.tensor([0] + dist)).tolist()
    
    # Create modules
    grouped_linear_triton = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    grouped_linear_hip = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    # Share weights
    with torch.no_grad():
        for i in range(num_gemms):
            weight_name = f"weight{i}"
            src_weight = getattr(grouped_linear_triton, weight_name)
            setattr(grouped_linear_hip, weight_name, Parameter(src_weight.clone()))
    
    # Create input
    inp = fill_uniform((total_tokens, hidden_size), dtype=dtype)
    
    # Run with Triton (enable via environment variable)
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "1"
    out_triton = grouped_linear_triton(inp, m_splits)
    
    # Run with hip (disable Triton)
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "0"
    out_hip = grouped_linear_hip(inp, m_splits)
    
    # Compare results
    atol, rtol = get_tolerances(dtype)
    te_compare_results(
        out_triton,
        out_hip,
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"Triton grouped GEMM does not match hip\n"
                       f"Config: num_gemms={num_gemms}, bs={bs}, seq_len={seq_len}, "
                       f"hidden={hidden_size}, out={out_features}, dtype={dtype}, "
                       f"equal_splits={equal_splits}\n"
                       f"m_splits={m_splits}\n\n{msg}\n",
        use_torch_semantics=True,
    )


@pytest.mark.parametrize("num_gemms", [3, 6])
@pytest.mark.parametrize("total_tokens", [512, 2048])
@pytest.mark.parametrize("hidden_size", [768])
@pytest.mark.parametrize("out_features", [3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_grouped_gemm_triton_with_zero_splits(
    num_gemms, total_tokens, hidden_size, out_features, dtype
):
    """Test Triton grouped GEMM kernel handling of zero-length splits."""
    
    # Skip if not on ROCm
    if not torch.version.hip:
        pytest.skip("Triton grouped GEMM kernel is only for ROCm/HIP")
    
    # Create m_splits with at least one zero
    torch.manual_seed(123)
    m_splits = torch.randint(0, total_tokens // num_gemms, (num_gemms,)).tolist()
    # Force at least one zero split
    m_splits[num_gemms // 2] = 0
    # Adjust to sum to total_tokens
    current_sum = sum(m_splits)
    if current_sum != total_tokens:
        diff = total_tokens - current_sum
        # Add diff to the last non-zero split
        for i in range(num_gemms - 1, -1, -1):
            if m_splits[i] != 0:
                m_splits[i] += diff
                break
    
    assert sum(m_splits) == total_tokens, f"m_splits sum mismatch: {sum(m_splits)} != {total_tokens}"
    assert 0 in m_splits, "Test must include at least one zero split"
    
    # Create modules
    grouped_linear_triton = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    grouped_linear_hip = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    # Share weights
    with torch.no_grad():
        for i in range(num_gemms):
            weight_name = f"weight{i}"
            src_weight = getattr(grouped_linear_triton, weight_name)
            setattr(grouped_linear_hip, weight_name, Parameter(src_weight.clone()))
    
    # Create input
    inp = fill_uniform((total_tokens, hidden_size), dtype=dtype)
    
    # Run with Triton
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "1"
    out_triton = grouped_linear_triton(inp, m_splits)
    
    # Run with hip
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "0"
    out_hip = grouped_linear_hip(inp, m_splits)
    
    # Compare results
    atol, rtol = get_tolerances(dtype)
    te_compare_results(
        out_triton,
        out_hip,
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"Triton grouped GEMM with zero splits does not match hip\n"
                       f"m_splits={m_splits} (contains {m_splits.count(0)} zeros)\n\n{msg}\n",
        use_torch_semantics=True,
    )


@pytest.mark.parametrize("num_gemms", [3])
@pytest.mark.parametrize("total_tokens", [2048])
@pytest.mark.parametrize("hidden_size", [768])
@pytest.mark.parametrize("out_features", [3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_grouped_gemm_triton_all_zero_splits(
    num_gemms, total_tokens, hidden_size, out_features, dtype
):
    """Test Triton grouped GEMM kernel with all zero splits (edge case)."""
    
    # Skip if not on ROCm
    if not torch.version.hip:
        pytest.skip("Triton grouped GEMM kernel is only for ROCm/HIP")
    
    # All zeros - extreme edge case
    m_splits = [0] * num_gemms
    
    # Create modules
    grouped_linear_triton = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    # Create input (will be empty after split)
    inp = torch.zeros((0, hidden_size), dtype=dtype, device="cuda")
    
    # Run with Triton - should not crash
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "1"
    try:
        out_triton = grouped_linear_triton(inp, m_splits)
        assert out_triton.shape == (0, out_features), f"Expected shape (0, {out_features}), got {out_triton.shape}"
    except Exception as e:
        pytest.fail(f"Triton kernel crashed with all-zero splits: {e}")


@pytest.mark.parametrize("hidden_size", [256, 768])
@pytest.mark.parametrize("out_features", [512, 3072])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_gemm_triton_single_expert(hidden_size, out_features, dtype):
    """Test Triton grouped GEMM kernel with single expert (num_gemms=1)."""
    
    # Skip if not on ROCm
    if not torch.version.hip:
        pytest.skip("Triton grouped GEMM kernel is only for ROCm/HIP")
    
    num_gemms = 1
    total_tokens = 1024
    m_splits = [total_tokens]
    
    # Create modules
    grouped_linear_triton = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    grouped_linear_hip = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    # Share weights
    with torch.no_grad():
        grouped_linear_hip.weight0 = Parameter(grouped_linear_triton.weight0.clone())
    
    # Create input
    inp = fill_uniform((total_tokens, hidden_size), dtype=dtype)
    
    # Run with Triton
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "1"
    out_triton = grouped_linear_triton(inp, m_splits)
    
    # Run with hip
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "0"
    out_hip = grouped_linear_hip(inp, m_splits)
    
    # Compare results
    atol, rtol = get_tolerances(dtype)
    te_compare_results(
        out_triton,
        out_hip,
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"Triton grouped GEMM (single expert) does not match hip\n\n{msg}\n",
        use_torch_semantics=True,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_gemm_triton_small_splits(dtype):
    """Test Triton grouped GEMM kernel with very small splits."""
    
    # Skip if not on ROCm
    if not torch.version.hip:
        pytest.skip("Triton grouped GEMM kernel is only for ROCm/HIP")
    
    num_gemms = 4
    hidden_size = 128
    out_features = 256
    # Very small splits (smaller than typical block sizes)
    m_splits = [8, 16, 32, 8]
    total_tokens = sum(m_splits)
    
    # Create modules
    grouped_linear_triton = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    grouped_linear_hip = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).eval()
    
    # Share weights
    with torch.no_grad():
        for i in range(num_gemms):
            weight_name = f"weight{i}"
            src_weight = getattr(grouped_linear_triton, weight_name)
            setattr(grouped_linear_hip, weight_name, Parameter(src_weight.clone()))
    
    # Create input
    inp = fill_uniform((total_tokens, hidden_size), dtype=dtype)
    
    # Run with Triton
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "1"
    out_triton = grouped_linear_triton(inp, m_splits)
    
    # Run with hip
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "0"
    out_hip = grouped_linear_hip(inp, m_splits)
    
    # Compare results
    atol, rtol = get_tolerances(dtype)
    te_compare_results(
        out_triton,
        out_hip,
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"Triton grouped GEMM (small splits) does not match hip\n"
                       f"m_splits={m_splits}\n\n{msg}\n",
        use_torch_semantics=True,
    )


@pytest.mark.parametrize("hidden_size", [256, 768])
@pytest.mark.parametrize("out_features", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_gemms", [1, 4])
def test_grouped_gemm_triton_backward(hidden_size, out_features, dtype, num_gemms):
    """Test Triton grouped GEMM kernel backward pass (dgrad)."""
    
    total_tokens = 512
    # Create equal splits for simplicity
    tokens_per_expert = total_tokens // num_gemms
    m_splits = [tokens_per_expert] * num_gemms
    
    # Create modules
    grouped_linear_triton = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    )
    
    grouped_linear_cublas = GroupedLinear(
        num_gemms,
        hidden_size,
        out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    )
    
    # Share weights
    with torch.no_grad():
        for i in range(num_gemms):
            weight_name = f"weight{i}"
            src_weight = getattr(grouped_linear_triton, weight_name)
            setattr(grouped_linear_cublas, weight_name, Parameter(src_weight.clone()))
    
    # Create input with gradients enabled
    torch.manual_seed(42)
    inp_triton = torch.randn(total_tokens, hidden_size, dtype=dtype, device="cuda", requires_grad=True)
    inp_cublas = inp_triton.clone().detach().requires_grad_(True)
    
    # Forward pass with Triton
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "1"
    out_triton = grouped_linear_triton(inp_triton, m_splits)
    
    # Forward pass with cuBLAS
    os.environ["NVTE_USE_GROUPED_GEMM_TRITON"] = "0"
    out_cublas = grouped_linear_cublas(inp_cublas, m_splits)
    
    # Check forward outputs match
    atol, rtol = get_tolerances(dtype)
    te_compare_results(
        out_triton,
        out_cublas,
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"Triton grouped GEMM forward does not match cuBLAS\n\n{msg}\n",
        use_torch_semantics=True,
    )
    
    # Backward pass with same grad_output
    torch.manual_seed(42)
    grad_output = torch.randn_like(out_triton)
    
    out_triton.backward(grad_output)
    out_cublas.backward(grad_output)
    
    # Check dgrad (input gradients) match
    te_compare_results(
        inp_triton.grad,
        inp_cublas.grad,
        atol=atol,
        rtol=rtol,
        msg=lambda msg: f"Triton grouped GEMM dgrad does not match cuBLAS\n\n{msg}\n",
        use_torch_semantics=True,
    )

