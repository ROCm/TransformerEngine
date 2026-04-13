# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-process worker for testing context parallelism in lite mode.

This script is launched via torch.distributed.launch with >= 2 GPUs.
It runs DotProductAttention with and without CP, then compares outputs
and gradients.

Only BSHD and SBHD formats are tested (THD requires C++ thd_* helpers
that are not yet implemented in lite mode).
"""

import logging
import os
import pathlib
import sys

os.environ["NVTE_LITE"] = "1"

# Ensure repo root is on sys.path for dev-tree runs (no pip install)
_repo_root = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.distributed as dist

from transformer_engine.pytorch import DotProductAttention


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

class CPTestConfig:
    """Minimal model config for CP tests."""

    def __init__(
        self,
        batch_size,
        max_seqlen,
        num_heads,
        head_dim,
        num_gqa_groups=None,
        attn_mask_type="causal",
    ):
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_gqa_groups = num_heads if num_gqa_groups is None else num_gqa_groups


TEST_CONFIGS = {
    "mha_causal": CPTestConfig(2, 1024, 8, 64, attn_mask_type="causal"),
    "gqa_causal": CPTestConfig(2, 1024, 8, 64, num_gqa_groups=2, attn_mask_type="causal"),
    "mha_no_mask": CPTestConfig(2, 1024, 8, 64, attn_mask_type="no_mask"),
    "gqa_no_mask": CPTestConfig(2, 1024, 8, 64, num_gqa_groups=2, attn_mask_type="no_mask"),
}


# ---------------------------------------------------------------------------
# DualChunkSwap partitioning for BSHD / SBHD
# ---------------------------------------------------------------------------

def partition_for_cp(tensor, qkv_format, rank, world_size):
    """Partition a tensor along the sequence dimension using DualChunkSwap.

    Each rank gets 2 chunks: [rank] and [2*world_size - rank - 1].
    """
    seq_dim = qkv_format.index("s")
    shape = list(tensor.shape)
    chunk_size = shape[seq_dim] // (2 * world_size)
    new_shape = shape[:seq_dim] + [2 * world_size, chunk_size] + shape[seq_dim + 1:]
    tensor = tensor.view(*new_shape)
    seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=tensor.device)
    tensor = tensor.index_select(seq_dim, seq_idx)
    final_shape = shape[:seq_dim] + [2 * chunk_size] + shape[seq_dim + 1:]
    return tensor.reshape(*final_shape).contiguous()


def partition_dout(dout, qkv_format, rank, world_size):
    """Partition dout (output gradient) for CP comparison.

    dout shape from DPA is (b, s, h*d) for bshd or (s, b, h*d) for sbhd.
    """
    seq_dim = 0 if qkv_format == "sbhd" else 1
    shape = list(dout.shape)
    chunk_size = shape[seq_dim] // (2 * world_size)
    new_shape = shape[:seq_dim] + [2 * world_size, chunk_size] + shape[seq_dim + 1:]
    dout = dout.view(*new_shape)
    seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=dout.device)
    dout = dout.index_select(seq_dim, seq_idx)
    final_shape = shape[:seq_dim] + [2 * chunk_size] + shape[seq_dim + 1:]
    return dout.reshape(*final_shape).contiguous()


# ---------------------------------------------------------------------------
# Core test logic
# ---------------------------------------------------------------------------

def run_test(
    config_name,
    qkv_format,
    cp_comm_type,
    attn_mask_type,
    dtype_str="bf16",
):
    """Run a single CP vs no-CP comparison test using DotProductAttention."""
    # Initialize distributed process group
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    config = TEST_CONFIGS[config_name]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_str]

    b = config.batch_size
    s = config.max_seqlen
    h_q = config.num_heads
    h_kv = config.num_gqa_groups
    d = config.head_dim

    assert s % (2 * world_size) == 0, (
        f"seqlen ({s}) must be divisible by 2*cp_size ({2 * world_size})"
    )

    # Generate full inputs -- same across all ranks (seeded)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if qkv_format == "bshd":
        q_shape = (b, s, h_q, d)
        k_shape = (b, s, h_kv, d)
        v_shape = (b, s, h_kv, d)
    elif qkv_format == "sbhd":
        q_shape = (s, b, h_q, d)
        k_shape = (s, b, h_kv, d)
        v_shape = (s, b, h_kv, d)
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    q_orig = torch.randn(q_shape, dtype=dtype, device=device)
    k_orig = torch.randn(k_shape, dtype=dtype, device=device)
    v_orig = torch.randn(v_shape, dtype=dtype, device=device)

    # DPA output shape is (b, s, h*d) for bshd, (s, b, h*d) for sbhd
    if qkv_format == "bshd":
        dout_shape = (b, s, h_q * d)
    else:
        dout_shape = (s, b, h_q * d)
    dout_orig = torch.randn(dout_shape, dtype=dtype, device=device)

    # ============== Run WITHOUT CP ==============
    core_attn = DotProductAttention(
        h_q, d, num_gqa_groups=h_kv, attention_dropout=0.0,
        qkv_format=qkv_format, attn_mask_type=attn_mask_type,
    ).cuda()

    q, k, v = [x.clone().detach().requires_grad_(True) for x in [q_orig, k_orig, v_orig]]
    dout = dout_orig.clone().detach()

    out = core_attn(q, k, v)
    out.backward(dout)
    dq, dk, dv = q.grad, k.grad, v.grad

    # ============== Run WITH CP ==============
    # Set up communication group
    cp_comm_ranks = list(range(world_size))
    cp_group = dist.new_group(cp_comm_ranks, backend="nccl")
    cp_stream = torch.cuda.Stream(device=device)

    # Partition inputs for this rank using DualChunkSwap
    q_, k_, v_ = [
        partition_for_cp(x, qkv_format, rank, world_size).clone().detach().requires_grad_(True)
        for x in [q_orig, k_orig, v_orig]
    ]
    dout_ = partition_dout(dout_orig, qkv_format, rank, world_size)

    # Configure CP on the attention module
    core_attn.set_context_parallel_group(cp_group, cp_comm_ranks, cp_stream, cp_comm_type)

    out_ = core_attn(q_, k_, v_)
    out_.backward(dout_)
    dq_, dk_, dv_ = q_.grad, k_.grad, v_.grad

    # ============== Validate ==============
    # Check no NaN/Inf
    for name, t in [("out_cp", out_), ("dq_cp", dq_), ("dk_cp", dk_), ("dv_cp", dv_)]:
        assert torch.all(torch.isfinite(t)), f"Rank {rank}: {name} contains NaN or Inf!"

    # Slice reference to match this rank's CP partition
    seq_dim = qkv_format.index("s")

    # For Q-side tensors (out, dq): partition ref the same as Q was partitioned
    # DPA output is (b, s, h*d) / (s, b, h*d) -- seq_dim is 1 / 0
    out_seq_dim = 1 if qkv_format == "bshd" else 0

    def slice_ref(ref_tensor, local_tensor, s_dim):
        """Slice full reference tensor to match this rank's DualChunkSwap partition."""
        shape = list(ref_tensor.shape)
        chunk_size = shape[s_dim] // (2 * world_size)
        new_shape = shape[:s_dim] + [2 * world_size, chunk_size] + shape[s_dim + 1:]
        ref_chunked = ref_tensor.view(*new_shape)
        seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=ref_tensor.device)
        ref_sliced = ref_chunked.index_select(s_dim, seq_idx)
        local_reshaped = local_tensor.view(*ref_sliced.shape)
        return ref_sliced, local_reshaped

    # Tolerances
    if dtype_str == "bf16":
        if h_q == h_kv:
            atol, rtol = 2.5e-2, 2.5e-2
        else:
            atol, rtol = 3.5e-2, 3.5e-2
    else:
        atol, rtol = 5e-3, 5e-3

    # Compare output and Q-side grads (use output seq_dim since DPA reshapes)
    for name, ref_full, cp_local in [("out", out, out_), ("dq", dq, dq_)]:
        s_dim = out_seq_dim if name == "out" else seq_dim
        ref_s, cp_s = slice_ref(ref_full, cp_local, s_dim)

        for ci in range(2):
            if s_dim == 1:  # bshd
                rc = ref_s[:, ci]
                cc = cp_s[:, ci]
            else:  # sbhd
                rc = ref_s[ci]
                cc = cp_s[ci]

            try:
                torch.testing.assert_close(rc, cc, atol=atol, rtol=rtol)
            except AssertionError:
                diff = (rc.float() - cc.float()).abs()
                rmse = diff.pow(2).mean().sqrt().item()
                val_range = max(rc.abs().max().item(), cc.abs().max().item(), 1e-6)
                assert rmse < 0.02 * val_range, (
                    f"Rank {rank}: {name} chunk {ci} RMSE {rmse:.6f} > "
                    f"tol {0.02 * val_range:.6f}"
                )

    # Compare K/V-side grads
    for name, ref_full, cp_local in [("dk", dk, dk_), ("dv", dv, dv_)]:
        ref_s, cp_s = slice_ref(ref_full, cp_local, seq_dim)

        for ci in range(2):
            if seq_dim == 1:
                rc = ref_s[:, ci]
                cc = cp_s[:, ci]
            else:
                rc = ref_s[ci]
                cc = cp_s[ci]

            try:
                torch.testing.assert_close(rc, cc, atol=atol, rtol=rtol)
            except AssertionError:
                diff = (rc.float() - cc.float()).abs()
                rmse = diff.pow(2).mean().sqrt().item()
                val_range = max(rc.abs().max().item(), cc.abs().max().item(), 1e-6)
                assert rmse < 0.02 * val_range, (
                    f"Rank {rank}: {name} chunk {ci} RMSE {rmse:.6f} > "
                    f"tol {0.02 * val_range:.6f}"
                )

    logging.info(
        f"Rank {rank}: PASSED -- config={config_name} fmt={qkv_format} "
        f"comm={cp_comm_type} mask={attn_mask_type} dtype={dtype_str}"
    )

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[2:])
    run_test(**kwargs)
