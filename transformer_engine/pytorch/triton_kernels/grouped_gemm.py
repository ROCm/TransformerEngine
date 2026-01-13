import triton
import triton.language as tl
import torch
from typing import Iterable, Optional, Tuple, Union, List
import functools
import json
import os.path
import sys
from pathlib import Path

# Add local 3rdparty/aiter to path to import from local version instead of installed package
_AITER_PATH = Path(__file__).parents[3] / "3rdparty" / "aiter"
if str(_AITER_PATH) not in sys.path:
    sys.path.insert(0, str(_AITER_PATH))

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
    m_splits: torch.Tensor = None,
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
        # WGRAD: ptgmm expects lhs=(K,M), rhs=(M,N), out=(G,K,N)
        # A=inputs (list of (m_i, in_features)), B=grad_outputs (list of (m_i, out_features))
        A_tensor = A[0] if len(A) == 1 else torch.cat(A, dim=0)  # (M, in_features)
        B_tensor = B[0] if len(B) == 1 else torch.cat(B, dim=0)  # (M, out_features)
        out_tensor_3d = out  # (G, out_features, in_features)
        
        # Allocate bias_grad OUTPUT buffer if needed (kernel writes to this)
        bias_grad_tensor = None
        if use_bias:
            G = m_splits.shape[0]
            K = B_tensor.shape[1]  # out_features
            bias_grad_tensor = torch.zeros(G, K, dtype=torch.float32, device=B_tensor.device)
        
        # Backward pass: C = B^T @ A (wgrad = grad_output^T @ input)
        # ptgmm expects lhs shape (K, M), so we need to transpose
        ptgmm(
            lhs=B_tensor.t(),  # (out_features, M) - transpose to get correct shape
            rhs=A_tensor,      # (M, in_features)
            group_sizes=m_splits,
            preferred_element_type=out_dtype,
            existing_out=out_tensor_3d,  # (G, out_features, in_features)
            config=None,
            bias_grad=bias_grad_tensor,  # OUTPUT: (G, out_features) or None
            accumulate=accumulate,
        )
        
        # Convert bias_grad to list to match C++ backend signature
        if use_bias and bias_grad_tensor is not None:
            grad_biases = list(torch.unbind(bias_grad_tensor, dim=0))
        else:
            grad_biases = [None] * len(out) if bias is None else bias
        
        # Return appropriate output format
        return_out = out_tensor_3d.view(-1, out_tensor_3d.shape[-1]) if single_output else out
        return return_out, grad_biases, None

    elif is_dgrad:
        # DGRAD: gmm expects lhs=(M,K), rhs=(G,K,N), out=(M,N)
        # A=weights (list of (out_features, in_features)), B=grad_outputs (list of (m_i, out_features))
        A_tensor_3d = torch.stack(A, dim=0)  # (G, out_features, in_features)
        B_tensor = B[0] if len(B) == 1 else torch.cat(B, dim=0)  # (M, out_features)
        out_tensor = out[0] if len(out) == 1 else torch.cat(out, dim=0)  # (M, in_features)
        
        # Stack bias into 3D if provided
        bias_tensor = None
        if bias is not None and len(bias) > 0 and bias[0].numel() > 0:
            bias_tensor = torch.stack(bias, dim=0)  # (G, in_features)
        
        # Backward pass: C = B @ A (dgrad = grad_output @ weight)
        gmm(
            lhs=B_tensor,      # (M, out_features)
            rhs=A_tensor_3d,   # (G, out_features, in_features)
            group_sizes=m_splits,
            preferred_element_type=out_dtype,
            existing_out=out_tensor,  # (M, in_features)
            config=None,
            bias=bias_tensor,
            group_sizes_list=kwargs.get("m_splits_list", []),
        )
        
        grad_biases = [None] * len(m_splits) if bias is None else bias
        return_out = out_tensor if single_output else out
        return return_out, grad_biases, None
        
    else:
        # FORWARD: gmm expects lhs=(M,K), rhs=(G,K,N), out=(M,N)
        # Forward pass: C = B @ A^T (output = input @ weight^T + bias)
        # A=weights (list of (out_features, in_features)), B=inputs (list of (m_i, in_features))
        A_tensor_3d = torch.stack(A, dim=0)  # (G, out_features, in_features)
        A_tensor_3d = A_tensor_3d.transpose(1, 2)  # (G, in_features, out_features) for TN layout
        B_tensor = B[0] if len(B) == 1 else torch.cat(B, dim=0)  # (M, in_features)
        out_tensor = out[0] if len(out) == 1 else torch.cat(out, dim=0)  # (M, out_features)
        
        # Stack bias into 3D if provided
        bias_tensor = None
        if bias is not None and len(bias) > 0 and bias[0].numel() > 0:
            bias_tensor = torch.stack(bias, dim=0)  # (G, out_features)
        
        gmm(
            lhs=B_tensor,      # (M, in_features)
            rhs=A_tensor_3d,   # (G, in_features, out_features)
            group_sizes=m_splits,
            preferred_element_type=out_dtype,
            existing_out=out_tensor,  # (M, out_features)
            config=None,
            bias=bias_tensor,
            group_sizes_list=kwargs.get("m_splits_list", []),
        )
        
        grad_biases = [None] * len(m_splits) if bias is None else bias
        return_out = out_tensor if single_output else out
        return return_out, grad_biases, None
