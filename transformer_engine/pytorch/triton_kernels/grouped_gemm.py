import triton
import triton.language as tl
import torch
from typing import Iterable, Optional, Tuple, Union, List
import functools
import json
import os.path
from aiter.ops.triton.gmm import gmm, ptgmm, nptgmm
from torch import Tensor
import transformer_engine_torch as tex

def general_grouped_gemm_triton(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output=False,
    **kwargs,
) -> list:
    """
    Drop-in replacement for general_grouped_gemm using AITER's Triton kernels.
    
    Supports:
    - Forward pass (layout="TN"): C = B @ A^T (where A=weights, B=inputs, C=outputs)
    - Backward pass dgrad (layout="NN", grad=True): C = B @ A (where A=weights, B=grad_output, C=dgrad)
    - Backward pass wgrad (layout="NT", grad=True): C = B^T @ A (where A=inputs, B=grad_output, C=wgrad)
    
    Args:
        A: Left-hand side matrices (weights for forward/dgrad, inputs for wgrad)
        B: Right-hand side matrices (inputs for forward, grad_outputs for backward)
        out: Output matrices (pre-allocated)
        out_dtype: Output dtype
        workspace: Workspace tensor (unused, for compatibility)
        single_output: Whether to produce single concatenated output
        m_splits: List of token counts per expert
        bias: List of bias tensors (optional)
        use_bias: Whether to apply bias
        use_split_accumulator: Unused, for compatibility
        layout: "TN" for forward pass, "NN" for dgrad backward pass, "NT" for wgrad backward pass
        grad: True for backward pass
        accumulate: Whether to accumulate into C (for wgrad only)
        
    Returns:
        Tuple of (outputs, bias_or_grad_bias, gelu_input) to match C++ backend signature
        - bias_or_grad_bias: List of bias/grad_bias tensors (or list of bias if passed in)
    """
    assert m_splits is not None, "m_splits required for Triton kernel"
    assert len(out) > 0, "Output tensor(s) must be pre-allocated and passed in C list"
    
    # Determine operation type
    is_dgrad = (layout == "NN" and grad)
    is_wgrad = (layout == "NT" and grad)
    
    
    if is_wgrad:
        A_tensor = torch.cat(A, dim=0)
        B_tensor = torch.cat(B, dim=0)
        out_tensor = torch.stack(out, dim=0)
        # Check if bias exists and contains non-empty tensors
        if bias is not None and len(bias) > 0 and bias[0].numel() > 0:
            bias_tensor = torch.stack(bias, dim=0)  # Use stack for 3D (G, N)
        else:
            bias_tensor = None
        group_sizes = torch.tensor(m_splits, dtype=torch.int32, device="cuda")
        print("B_tensor.shape", B_tensor.shape)
        print("A_tensor.shape", A_tensor.shape)
        print("group_sizes", group_sizes)
        print("out_tensor.shape", out_tensor.shape)
        # Backward pass: C = B^T @ A (wgrad = grad_output^T @ input)
        # A=inputs, B=grad_outputs, C=wgrad
        ptgmm(
            lhs=B_tensor.transpose(0, 1),  # grad_outputs
            rhs=A_tensor,  # inputs
            group_sizes=group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_tensor,  # wgrad
            config=None,
            bias_grad=bias_tensor,
            accumulate=accumulate,
        )

    elif is_dgrad:
        A_tensor = torch.stack(A, dim=0)
        B_tensor = torch.cat(B, dim=0)
        out_tensor = torch.cat(out, dim=0)
        # Check if bias exists and contains non-empty tensors
        if bias is not None and len(bias) > 0 and bias[0].numel() > 0:
            bias_tensor = torch.stack(bias, dim=0)  # Use stack for 3D (G, N)
        else:
            bias_tensor = None
        group_sizes = torch.tensor(m_splits, dtype=torch.int32, device="cuda")
        # Backward pass: C = B @ A (dgrad = grad_output @ weight)
        # A=weights, B=grad_outputs, C=dgrad
        gmm(
            lhs=B_tensor,  # grad_outputs
            rhs=A_tensor,  # weights
            group_sizes=group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_tensor,  # dgrad
            config=None,
            bias=bias_tensor,
        )
    else:
        # Forward pass: C = B @ A^T (output = input @ weight^T + bias)
        # A=weights, B=inputs, C=outputs
        A_tensor = torch.stack(A, dim=0).transpose(1, 2)
        B_tensor = torch.cat(B, dim=0)
        out_tensor = torch.cat(out, dim=0)
        # Check if bias exists and contains non-empty tensors
        if bias is not None and len(bias) > 0 and bias[0].numel() > 0:
            bias_tensor = torch.stack(bias, dim=0)  # Use stack for 3D (G, N)
        else:
            bias_tensor = None
        group_sizes = torch.tensor(m_splits, dtype=torch.int32, device="cuda")
        gmm(
            lhs=B_tensor,  # inputs
            rhs=A_tensor,  # weights
            group_sizes=group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_tensor,  # output
            config=None,
            bias=bias_tensor,
        )
    
    # Return outputs, grad_biases, and None for gelu_input (to match C++ backend signature)
    return out_tensor, bias, None
