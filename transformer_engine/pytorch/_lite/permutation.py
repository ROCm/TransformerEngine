# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MOE permutation operations -- tex-compatible interface.

Index-map path: Uses Triton sort_chunks_by_map kernel for gather operations
when available, falling back to PyTorch-native.

Mask-map path: The higher-level permutation.py imports
transformer_engine.pytorch.triton.permutation directly for mask-map
operations -- those Triton kernels work in lite mode without changes here.
"""

import torch

# ---------------------------------------------------------------------------
# Lazy Triton import for sort_chunks_by_map (gather/scatter kernel)
# ---------------------------------------------------------------------------
_triton_sort = None
_triton_attempted = False


def _try_load_triton_sort():
    global _triton_sort, _triton_attempted
    if _triton_attempted:
        return _triton_sort
    _triton_attempted = True
    try:
        from transformer_engine.pytorch.triton.permutation import sort_chunks_by_map
        _triton_sort = sort_chunks_by_map
    except (ImportError, RuntimeError):
        pass
    return _triton_sort


# ---------------------------------------------------------------------------
# tex-compatible API
# ---------------------------------------------------------------------------

def moe_permute_fwd(input, dtype, indices, num_out_tokens, workspace, max_expanded_token_num):
    """MOE permute forward: sort tokens by expert assignment.

    Matches the ``tex.moe_permute_fwd`` C++ interface used by
    ``_moe_permute_index_map`` in ``permutation.py``.

    Returns
    -------
    (permuted_output, row_id_map, workspace)
    """
    num_tokens = input.size(0)
    num_cols = input.size(1)
    topK = indices.size(1)

    # Flatten expert indices and sort to group tokens by expert
    flat_indices = indices.reshape(-1).to(torch.int32)
    _, sorted_row_id = torch.sort(flat_indices, stable=True)

    num_out = num_out_tokens if num_out_tokens > 0 else num_tokens * topK

    # Map each permuted position to its source token row
    source_token_ids = (sorted_row_id[:num_out] // topK).to(torch.int32)

    # Gather rows -- prefer Triton kernel when available
    sort_fn = _try_load_triton_sort()
    if sort_fn is not None and input.is_cuda:
        permuted_output, _ = sort_fn(
            input, source_token_ids, None, num_out, num_cols, is_forward=False,
        )
    else:
        permuted_output = input[source_token_ids.long()]

    # Build inverse map: flat position j → permuted position
    row_id_map = torch.zeros(
        num_tokens * topK, dtype=torch.int32, device=input.device,
    )
    row_id_map[sorted_row_id[:num_out]] = torch.arange(
        num_out, dtype=torch.int32, device=input.device,
    )

    return permuted_output, row_id_map, workspace


def moe_permute_bwd(input, dtype, row_id_map, prob, num_tokens, topK):
    """MOE permute backward -- identical to ``moe_unpermute_fwd``.

    Matches ``tex.moe_permute_bwd`` (the C++ implementation delegates
    to ``moe_unpermute_fwd`` as well).
    """
    return moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK)


def moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK):
    """MOE unpermute forward: scatter-add from permuted to original order.

    Matches ``tex.moe_unpermute_fwd``.
    """
    num_cols = input.size(1)

    # Gather permuted data back to flat (num_tokens*topK) order
    gathered = input[row_id_map.long()]  # [num_tokens * topK, num_cols]

    if prob.numel() > 0:
        gathered = gathered * prob.reshape(-1, 1)

    # Sum over the topK dimension to merge expert contributions
    output = gathered.reshape(num_tokens, topK, num_cols).sum(dim=1)
    return output


def moe_unpermute_bwd(input_bwd, input_fwd, dtype, row_id_map, prob):
    """MOE unpermute backward.

    Matches ``tex.moe_unpermute_bwd``.

    Returns
    -------
    (act_grad, prob_grad)
    """
    topK = prob.size(1) if prob.numel() > 0 else 1
    num_tokens = prob.size(0) if prob.numel() > 0 else row_id_map.size(0)
    num_cols = input_bwd.size(1)

    # Expand grad from [num_tokens, num_cols] to [num_tokens * topK, num_cols]
    token_ids = torch.arange(num_tokens, device=input_bwd.device).repeat_interleave(topK)

    act_grad = torch.zeros(
        input_fwd.size(0), num_cols,
        device=input_bwd.device, dtype=input_bwd.dtype,
    )

    if prob.numel() > 0:
        weights = prob.reshape(-1, 1).to(input_bwd.dtype)
        act_grad[row_id_map.long()] = input_bwd[token_ids] * weights

        # prob_grad = dot(d_output[token], fwd_input[permuted_pos])
        fwd_gathered = input_fwd[row_id_map.long()].float()
        prob_grad = (
            (fwd_gathered * input_bwd[token_ids].float())
            .sum(dim=-1).reshape(num_tokens, topK)
        )
    else:
        act_grad[row_id_map.long()] = input_bwd[token_ids]
        prob_grad = torch.empty(0, device=input_bwd.device, dtype=torch.float32)

    return act_grad, prob_grad
