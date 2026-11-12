# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""ROCm-specific Triton permutation kernels."""

import triton
import triton.language as tl


# The row_id_map from token-major passes over the routing map: block counts per expert, their
# prefix sums, then each block's rows. Triton 3.8 on gfx1250 miscompiles multi-warp kernels that
# both scan and reduce a block, as _row_id_map_pass_1_kernel does, so none of these does both.
@triton.jit
def _row_id_map_block_counts_kernel(
    routing_map_ptr,
    block_counts_ptr,
    row_id_map_ptr,
    num_tokens,
    num_blocks,
    stride_routing_map_token,
    stride_routing_map_expert,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    num_experts: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """block_counts[e, b]: the tokens of block b routed to expert e; each token's n_routed."""
    pid = tl.program_id(0)
    token = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    expert = tl.arange(0, BLOCK_E)
    routed = tl.load(
        routing_map_ptr
        + token[:, None] * stride_routing_map_token
        + expert[None, :] * stride_routing_map_expert,
        mask=(token < num_tokens)[:, None] & (expert < num_experts)[None, :],
        other=0,
    ).to(tl.int32)
    tl.store(
        block_counts_ptr + expert * num_blocks + pid,
        tl.sum(routed, axis=0),
        mask=expert < num_experts,
    )
    tl.store(
        row_id_map_ptr
        + token * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert,
        tl.sum(routed, axis=1),
        mask=token < num_tokens,
    )


@triton.jit
def _row_id_map_block_scan_kernel(block_counts_ptr, num_blocks, BLOCK_B: tl.constexpr):
    """block_counts[e, b] -> the tokens of blocks 0..b routed to expert e."""
    off = tl.arange(0, BLOCK_B)
    ptrs = block_counts_ptr + tl.program_id(0) * num_blocks + off
    counts = tl.load(ptrs, mask=off < num_blocks, other=0)
    tl.store(ptrs, tl.cumsum(counts, axis=0), mask=off < num_blocks)


@triton.jit
def _row_id_map_block_rows_kernel(
    routing_map_ptr,
    block_counts_ptr,
    row_id_map_ptr,
    num_tokens,
    num_blocks,
    stride_routing_map_token,
    stride_routing_map_expert,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    num_experts: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """The row_id_map entries of block pid, as _row_id_map_pass_3_kernel leaves them."""
    pid = tl.program_id(0)
    token = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    expert = tl.arange(0, BLOCK_E)
    valid_expert = expert < num_experts
    routed = tl.load(
        routing_map_ptr
        + token[:, None] * stride_routing_map_token
        + expert[None, :] * stride_routing_map_expert,
        mask=(token < num_tokens)[:, None] & valid_expert[None, :],
        other=0,
    ).to(tl.int32)
    # Rows are expert-major: an expert's rows follow those of the experts before it.
    expert_tokens = tl.load(
        block_counts_ptr + expert * num_blocks + num_blocks - 1, mask=valid_expert, other=0
    )
    tokens_before_block = tl.load(
        block_counts_ptr + expert * num_blocks + pid - 1, mask=valid_expert & (pid > 0), other=0
    )
    first_row = tl.cumsum(expert_tokens, axis=0) - expert_tokens + tokens_before_block
    dst_row = first_row[None, :] + tl.cumsum(routed, axis=0) - routed
    map_row = row_id_map_ptr + token[:, None] * stride_row_id_map_token
    n_routed = tl.load(
        map_row + num_experts * 2 * stride_row_id_map_expert,
        mask=(token < num_tokens)[:, None],
        other=0,
    )
    # Rows grow with the expert, so the experts in descending order give the rows sorted descending.
    idx = n_routed - tl.cumsum(routed, axis=1)
    tl.store(map_row + idx * stride_row_id_map_expert, dst_row, mask=routed != 0)
    tl.store(
        map_row + (num_experts + idx) * stride_row_id_map_expert,
        expert[None, :] + tl.zeros_like(routed),
        mask=routed != 0,
    )


@triton.jit
def _unpermute_probs_kernel(
    row_id_map_ptr,
    permuted_probs_ptr,
    pad_offsets_ptr,
    unpermuted_probs_ptr,
    num_tokens,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_permuted_probs_token,
    stride_unpermuted_probs_token,
    stride_unpermuted_probs_expert,
    num_experts: tl.constexpr,
    FUSION_UNPAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """The unpermuted probs of _unpermute_kernel with PERMUTE_PROBS, over blocks of tokens."""
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    idx = tl.arange(0, BLOCK_E)
    valid_token = token < num_tokens
    probs_row = unpermuted_probs_ptr + token[:, None] * stride_unpermuted_probs_token
    tl.store(
        probs_row + idx[None, :] * stride_unpermuted_probs_expert,
        0.0,
        mask=valid_token[:, None] & (idx < num_experts)[None, :],
    )
    # The routed probs overwrite zeros of the same program.
    tl.debug_barrier()
    map_row = row_id_map_ptr + token[:, None] * stride_row_id_map_token
    n_routed = tl.load(
        row_id_map_ptr
        + token * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert,
        mask=valid_token,
        other=0,
    )
    routed = valid_token[:, None] & (idx[None, :] < n_routed[:, None])
    src_row = tl.load(map_row + idx[None, :] * stride_row_id_map_expert, mask=routed, other=0)
    expert = tl.load(
        map_row + (num_experts + idx[None, :]) * stride_row_id_map_expert, mask=routed, other=0
    )
    if FUSION_UNPAD:
        src_row += tl.load(pad_offsets_ptr + expert, mask=routed, other=0)
    prob = tl.load(
        permuted_probs_ptr + src_row.to(tl.int64) * stride_permuted_probs_token, mask=routed
    )
    tl.store(probs_row + expert * stride_unpermuted_probs_expert, prob, mask=routed)
