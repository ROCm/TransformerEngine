# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""MORI Expert Parallelism integration for tealite.

Wraps MORI's EpDispatchCombineOp to provide distributed expert parallelism
for the MoE pipeline. MORI handles high-performance token dispatch/combine
across GPUs using XGMI (intra-node) and RDMA (inter-node).

Requires: ``pip install mori`` (or MORI built from source with ROCm 6.4+).

Inference usage::

    from transformer_engine.pytorch._lite.mori_ep import (
        mori_ep_available,
        init_mori_ep,
        MoriExpertParallel,
    )

    init_mori_ep()
    ep = MoriExpertParallel(rank=rank, world_size=8, ...)

    recv, recv_w, recv_idx, n = ep.dispatch(tokens, weights, indices)
    expert_out = run_experts(recv[:n])
    output, _ = ep.combine(expert_out, recv_w, recv_idx)
    ep.reset()

Training usage (with autograd, Phase 3 deferred-sync API)::

    state = ep.new_cycle()
    # Returns padded buffers + a device-resident total_recv scalar tensor.
    # No host sync at the dispatch site.
    recv, recv_w, recv_idx, total_recv = MoriEPDispatch.apply(
        tokens, weights, indices, state,
    )

    # Process on device using total_recv as a row mask. For example:
    local_idx = rebase_global_to_local_indices(
        recv_idx, total_recv, rank, num_local_experts,
    )
    m_splits = compute_tokens_per_expert_device(local_idx, num_local_experts)
    # ... permute & grouped GEMM with m_splits as a device tensor ...

    # Combine accepts padded inputs; the single host .item() is deferred to
    # here, where dispatch (on the comm stream) is already done.
    output, _ = MoriEPCombine.apply(weighted_padded, recv_w, recv_idx, state)
    loss = loss_fn(output[:num_tokens])
    loss.backward()  # gradients flow through combine → expert → dispatch
"""

import warnings
from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Lazy MORI import
# ---------------------------------------------------------------------------
_mori = None
_mori_available: Optional[bool] = None
_mori_shmem_initialized = False


def _try_import_mori():
    global _mori, _mori_available
    if _mori_available is not None:
        return _mori_available
    try:
        import mori
        _mori = mori
        _mori_available = True
    except ImportError:
        _mori_available = False
    return _mori_available


def mori_ep_available() -> bool:
    """Check whether MORI is installed and available for expert parallelism."""
    return _try_import_mori()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_mori_ep(process_group_name: str = "default") -> None:
    """Initialize MORI shmem from a PyTorch distributed process group.

    Must be called once per process after ``torch.distributed.init_process_group()``.
    Safe to call multiple times -- subsequent calls are no-ops.

    Args:
        process_group_name: Name of the PyTorch process group to use for
            bootstrapping MORI's symmetric memory. Defaults to ``"default"``
            (the WORLD group).
    """
    global _mori_shmem_initialized
    if _mori_shmem_initialized:
        return

    if not _try_import_mori():
        raise RuntimeError(
            "MORI is not installed. Install with: pip install mori "
            "(or build from source at https://github.com/ROCm/mori)"
        )

    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before init_mori_ep(). "
            "Call torch.distributed.init_process_group() first."
        )

    _mori.shmem.shmem_torch_process_group_init(process_group_name)
    _mori_shmem_initialized = True


def finalize_mori_ep() -> None:
    """Finalize MORI shmem. Call during process cleanup."""
    global _mori_shmem_initialized
    if not _mori_shmem_initialized:
        return
    _mori.shmem.shmem_finalize()
    _mori_shmem_initialized = False


def is_mori_ep_initialized() -> bool:
    """Return whether MORI shmem has been initialized."""
    return _mori_shmem_initialized


# ---------------------------------------------------------------------------
# Comm-stream / async helpers
# ---------------------------------------------------------------------------
# Process-wide high-priority CUDA stream dedicated to MORI dispatch/combine
# launches. Mirrors the MCore-side ``_mori_comm_stream`` pattern so MORI's
# comm kernels can overlap with non-dependent compute on the default stream.
# A host sync (``.item()`` on ``total_recv``) still serializes dispatch when
# the caller needs the receive count to slice the output buffer — eliminating
# that is the Phase 3 sync-free refactor, not this phase.
_mori_comm_stream = None


def _get_mori_comm_stream():
    """Return a process-wide high-priority CUDA stream for MORI launches.

    Lazily allocated on first use; cached for the lifetime of the process.
    Priority ``-1`` keeps MORI's communication kernels from being preempted
    by lower-priority compute on the default stream.
    """
    global _mori_comm_stream
    if _mori_comm_stream is None and torch.cuda.is_available():
        _mori_comm_stream = torch.cuda.Stream(
            device=torch.cuda.current_device(), priority=-1
        )
    return _mori_comm_stream


def _run_mori_op_on_stream(fn, async_finish: bool, allocate_on_comm_stream: bool):
    """Run a MORI op on the dedicated comm stream when both flags are True.

    When enabled, the kernel launch is placed on a high-priority comm stream
    bracketed by ``wait_stream()`` calls so the comm stream sees the inputs
    and the compute stream sees the outputs. Otherwise ``fn()`` runs on the
    current stream (legacy synchronous behavior).

    MORI's ``op.dispatch()`` / ``op.combine()`` take no async kwargs of their
    own; their kernel launches are already CUDA-async. This helper just
    places them on a non-default stream so they can overlap with unrelated
    compute on the default stream.
    """
    if not (async_finish and allocate_on_comm_stream and torch.cuda.is_available()):
        return fn()
    comm_stream = _get_mori_comm_stream()
    if comm_stream is None:
        return fn()
    current_stream = torch.cuda.current_stream()
    comm_stream.wait_stream(current_stream)
    with torch.cuda.stream(comm_stream):
        result = fn()
    current_stream.wait_stream(comm_stream)
    return result


# ---------------------------------------------------------------------------
# Routing map conversion
# ---------------------------------------------------------------------------

def mask_to_index(
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a mask-map routing tensor to the index-map format used by MORI.

    TE's MoE layer supports two routing map formats:

    - **mask**: ``[num_tokens, num_experts]`` binary int32 tensor where 1 means
      the token is routed to that expert.
    - **index**: ``[num_tokens, topk]`` int32 tensor of selected expert IDs.

    MORI only accepts the index format. This function converts mask → index
    and gathers the corresponding routing probabilities.

    Args:
        routing_map: Binary mask tensor, shape ``[num_tokens, num_experts]``,
            dtype int32. Each row has exactly ``topk`` ones.
        probs: Optional probability tensor, shape ``[num_tokens, num_experts]``,
            dtype float32. Contains the routing probability for each
            token-expert pair.  Only the entries where ``routing_map == 1``
            are meaningful.

    Returns:
        Tuple of ``(indices, weights)``:

        - ``indices``: Expert indices, shape ``[num_tokens, topk]``, int32.
        - ``weights``: Routing weights gathered from *probs* at the selected
          positions, shape ``[num_tokens, topk]``, float32.  If *probs* is
          ``None``, returns uniform weights of 1.0.

    Example::

        # mask: token 0 → experts 1,3; token 1 → experts 0,2
        mask = torch.tensor([[0,1,0,1],[1,0,1,0]], dtype=torch.int32, device="cuda")
        probs = torch.tensor([[0,.3,0,.7],[.5,0,.5,0]], dtype=torch.float32, device="cuda")
        indices, weights = mask_to_index(mask, probs)
        # indices: [[1,3],[0,2]]  weights: [[.3,.7],[.5,.5]]
    """
    # nonzero gives sorted (row, col) pairs — rows are in-order, columns
    # within each row are ascending, which matches TE's mask-map convention.
    nz = routing_map.nonzero(as_tuple=False)  # [nnz, 2]
    expert_ids = nz[:, 1].to(torch.int32)

    # Determine topk from the mask (number of ones per row, assumed uniform)
    num_tokens = routing_map.shape[0]
    topk = nz.shape[0] // num_tokens if num_tokens > 0 else 0

    indices = expert_ids.reshape(num_tokens, topk)

    if probs is not None:
        # Gather probabilities at selected positions
        weights = probs[nz[:, 0], nz[:, 1]].to(torch.float32).reshape(num_tokens, topk)
    else:
        weights = torch.ones(
            num_tokens, topk, dtype=torch.float32, device=routing_map.device,
        )

    return indices, weights


def index_to_mask(
    indices: torch.Tensor,
    num_experts: int,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convert an index-map routing tensor to the mask-map format used by TE.

    Inverse of :func:`mask_to_index`.

    Args:
        indices: Expert indices, shape ``[num_tokens, topk]``, int32.
        num_experts: Total number of experts.
        weights: Optional routing weights, shape ``[num_tokens, topk]``, float32.

    Returns:
        Tuple of ``(routing_map, probs)``:

        - ``routing_map``: Binary mask, shape ``[num_tokens, num_experts]``, int32.
        - ``probs``: Probability tensor with weights scattered to expert
          positions, shape ``[num_tokens, num_experts]``, float32.
          ``None`` if *weights* is ``None``.
    """
    num_tokens, topk = indices.shape
    routing_map = torch.zeros(
        num_tokens, num_experts, dtype=torch.int32, device=indices.device,
    )
    row_idx = torch.arange(num_tokens, device=indices.device).unsqueeze(1).expand_as(indices)
    routing_map[row_idx, indices.long()] = 1

    probs = None
    if weights is not None:
        probs = torch.zeros(
            num_tokens, num_experts, dtype=torch.float32, device=indices.device,
        )
        probs[row_idx, indices.long()] = weights

    return routing_map, probs


# ---------------------------------------------------------------------------
# Device-side helpers for sync-free dispatch consumers
# ---------------------------------------------------------------------------
# These let a caller process MORI's padded dispatch output entirely on the
# GPU — no .item() / .tolist() / bincount required. The host sync moves from
# "immediately after dispatch" to "when grouped GEMM needs m_splits as a
# Python list" (inside AITER's gmm wrapper), naturally overlapping with the
# dispatch kernel that runs on the comm stream.


def compute_tokens_per_expert_device(
    local_idx: torch.Tensor,
    num_local_experts: int,
) -> torch.Tensor:
    """Count tokens routed to each local expert, on device, with no host sync.

    Equivalent to ``torch.bincount(local_idx.flatten(), minlength=num_local_experts)``
    but avoids ``bincount``'s internal ``.max().item()`` (which forces a host
    sync to size the output buffer). Entries with value ``-1`` or out of the
    ``[0, num_local_experts)`` range are treated as invalid (e.g. tokens
    routed to a different rank, or rows beyond ``total_recv``).

    Args:
        local_idx: Local expert IDs, shape ``[N, topk]`` or ``[N*topk]``,
            integer dtype. ``-1`` marks invalid entries.
        num_local_experts: Number of experts owned by this rank.

    Returns:
        ``[num_local_experts]`` int32 tensor of token counts.
    """
    flat = local_idx.flatten().long()
    # Map invalid entries to an "overflow bucket" at index num_local_experts
    # rather than using boolean-mask indexing (which would force a host sync
    # to determine the masked output's shape).
    in_range = (flat >= 0) & (flat < num_local_experts)
    safe = torch.where(
        in_range, flat, torch.full_like(flat, num_local_experts),
    )
    counts = torch.zeros(
        num_local_experts + 1, dtype=torch.int64, device=flat.device,
    )
    counts.scatter_add_(0, safe, torch.ones_like(flat, dtype=torch.int64))
    return counts[:num_local_experts].to(torch.int32)


def rebase_global_to_local_indices(
    global_idx: torch.Tensor,
    total_recv: torch.Tensor,
    rank: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Convert MORI's global expert IDs to local IDs with ``-1`` sentinels.

    MORI's dispatch returns expert IDs in the global ``[0, num_total_experts)``
    range. The expert-MLP layer needs them in the local ``[0, num_local_experts)``
    range, with non-local tokens and padding rows (beyond ``total_recv``)
    marked as invalid so downstream permute/grouped GEMM can skip them.

    This runs entirely on device — no host sync.

    Args:
        global_idx: Global expert IDs from MORI dispatch, shape
            ``[max_recv, topk]``, integer dtype.
        total_recv: Device-resident scalar tensor (shape ``[1]`` or ``[]``)
            giving the count of valid rows. Rows ``>= total_recv`` are padding.
        rank: This rank's number in the expert-parallel group.
        num_local_experts: Number of experts owned by each rank.

    Returns:
        ``[max_recv, topk]`` tensor of the same dtype as ``global_idx``.
        Entries are local IDs in ``[0, num_local_experts)`` for tokens routed
        to this rank, ``-1`` otherwise (other ranks' experts, or padding rows).
    """
    lo = rank * num_local_experts
    hi = lo + num_local_experts

    local_id = global_idx - lo
    in_local_range = (global_idx >= lo) & (global_idx < hi)

    # Row-index mask: rows >= total_recv are padding. Broadcast total_recv
    # (which may be shape [1] or [] depending on MORI) against the row axis.
    max_recv = global_idx.shape[0]
    row_idx = torch.arange(
        max_recv, device=global_idx.device, dtype=global_idx.dtype,
    ).unsqueeze(1)
    total_recv_scalar = total_recv.flatten()[0]
    in_valid_rows = row_idx < total_recv_scalar.to(global_idx.dtype)

    return torch.where(
        in_local_range & in_valid_rows,
        local_id,
        torch.full_like(global_idx, -1),
    )


# ---------------------------------------------------------------------------
# Expert Parallel operator
# ---------------------------------------------------------------------------

class MoriExpertParallel:
    """High-level wrapper around MORI's EP dispatch/combine for tealite MoE.

    This replaces the local permute/unpermute steps with distributed
    dispatch/combine when running with expert parallelism across multiple GPUs.

    Args:
        rank: Rank of this process in the expert-parallel group.
        world_size: Total number of ranks in the expert-parallel group.
        hidden_dim: Hidden dimension of token embeddings.
        num_experts_per_rank: Number of local experts hosted on each rank.
        num_experts_per_token: Number of experts selected per token (top-k).
        max_num_inp_token_per_rank: Maximum number of input tokens per rank.
        dtype: Data type for dispatch/combine buffers.
        kernel_type: MORI kernel type. One of ``"intra_node"`` (default),
            ``"inter_node"``, ``"inter_node_v1"``, ``"inter_node_v1_ll"``,
            ``"async_ll"``.
        block_num: Number of GPU blocks for kernel launch.
        warp_num_per_block: Number of warps per GPU block.
        gpu_per_node: Number of GPUs per node (for topology).
        rdma_block_num: Number of RDMA blocks (inter-node kernels).
        quant_type: Quantization mode. ``"none"`` or ``"fp8_direct_cast"``.
    """

    # Map user-friendly names to MORI kernel type enum values
    _KERNEL_TYPE_MAP = {
        "intra_node": "IntraNode",
        "inter_node": "InterNode",
        "inter_node_v1": "InterNodeV1",
        "inter_node_v1_ll": "InterNodeV1LL",
        "async_ll": "AsyncLL",
    }

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden_dim: int,
        num_experts_per_rank: int,
        num_experts_per_token: int,
        max_num_inp_token_per_rank: int,
        dtype: torch.dtype = torch.bfloat16,
        kernel_type: str = "intra_node",
        block_num: int = 80,
        warp_num_per_block: int = 8,
        gpu_per_node: int = 8,
        rdma_block_num: int = 0,
        quant_type: str = "none",
    ):
        if not mori_ep_available():
            raise RuntimeError(
                "MORI is not installed. Install with: pip install mori"
            )
        if not is_mori_ep_initialized():
            raise RuntimeError(
                "MORI shmem not initialized. Call init_mori_ep() first."
            )

        self.rank = rank
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        self.num_experts_per_rank = num_experts_per_rank
        self.num_experts_per_token = num_experts_per_token
        self.max_num_inp_token_per_rank = max_num_inp_token_per_rank
        self.dtype = dtype

        # Resolve kernel type
        kt_name = self._KERNEL_TYPE_MAP.get(kernel_type)
        if kt_name is None:
            raise ValueError(
                f"Unknown kernel_type '{kernel_type}'. "
                f"Expected one of: {list(self._KERNEL_TYPE_MAP.keys())}"
            )
        kt_enum = getattr(_mori.ops.EpDispatchCombineKernelType, kt_name)

        use_external_inp_buf = quant_type == "fp8_direct_cast"

        self._config = _mori.ops.EpDispatchCombineConfig(
            data_type=dtype,
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            scale_dim=0,
            scale_type_size=1,
            max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
            max_num_inp_token_per_rank=max_num_inp_token_per_rank,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            use_external_inp_buf=use_external_inp_buf,
            kernel_type=kt_enum,
            gpu_per_node=gpu_per_node,
            rdma_block_num=rdma_block_num,
            quant_type=quant_type,
        )

        self._op = _mori.ops.EpDispatchCombineOp(self._config)

    @property
    def num_experts(self) -> int:
        """Total number of experts across all ranks."""
        return self.num_experts_per_rank * self.world_size

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        block_num: int = -1,
        warp_per_block: int = -1,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Dispatch tokens to expert-owning ranks.

        Args:
            input: Token embeddings, shape ``[num_tokens, hidden_dim]``.
            weights: Routing weights from the router, shape
                ``[num_tokens, num_experts_per_token]``.
            indices: Expert indices from the router, shape
                ``[num_tokens, num_experts_per_token]``, dtype int32.
            scales: Optional per-token scales for quantized paths.
            block_num: Override GPU block count for this launch.
            warp_per_block: Override warps-per-block for this launch.
            async_finish: When True (with ``allocate_on_comm_stream=True``),
                run the MORI dispatch on the high-priority comm stream so it
                can overlap with unrelated compute on the default stream.
                The host ``.item()`` on the receive count still serializes.
            allocate_on_comm_stream: See ``async_finish``.

        Returns:
            Tuple of ``(recv_tokens, recv_weights, recv_indices, num_recv_tokens)``:

            - ``recv_tokens``: Received token embeddings, shape
              ``[max_recv, hidden_dim]``. Only the first ``num_recv_tokens``
              rows are valid.
            - ``recv_weights``: Routing weights for received tokens.
            - ``recv_indices``: Expert indices for received tokens.
            - ``num_recv_tokens``: Number of valid received tokens.
        """
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        if scales is None:
            scales = torch.empty(
                input.size(0), 0, dtype=torch.float32, device=input.device,
            )

        out, out_weights, _out_scales, out_indices, total_recv = _run_mori_op_on_stream(
            lambda: self._op.dispatch(
                input, weights, scales, indices,
                block_num=block_num,
                warp_per_block=warp_per_block,
            ),
            async_finish,
            allocate_on_comm_stream,
        )

        # .item() is the host sync here; wait_stream() inside the helper
        # already made total_recv visible on the current stream.
        num_recv = total_recv[0].item()

        return out, out_weights, out_indices, num_recv

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Combine expert outputs back to original ranks.

        Args:
            input: Expert output embeddings for tokens this rank processed,
                shape ``[num_recv_tokens, hidden_dim]``.
            weights: Routing weights returned from :meth:`dispatch`.
            indices: Expert indices returned from :meth:`dispatch`.
            block_num: Override GPU block count for this launch.
            warp_per_block: Override warps-per-block for this launch.
            async_finish: When True (with ``allocate_on_comm_stream=True``),
                run the MORI combine on the high-priority comm stream so it
                can overlap with unrelated compute on the default stream.
                No host sync is needed; the compute stream's wait_stream()
                handles ordering for downstream consumers.
            allocate_on_comm_stream: See ``async_finish``.

        Returns:
            Tuple of ``(output, output_weights)``:

            - ``output``: Combined token embeddings, shape
              ``[max_num_inp_token_per_rank, hidden_dim]``. Only the first
              ``num_input_tokens`` rows (matching the original input count)
              are valid.
            - ``output_weights``: Combined routing weights, or ``None``.
        """
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        output, output_weights = _run_mori_op_on_stream(
            lambda: self._op.combine(
                input, weights, indices,
                block_num=block_num,
                warp_per_block=warp_per_block,
            ),
            async_finish,
            allocate_on_comm_stream,
        )

        return output, output_weights

    # ------------------------------------------------------------------
    # Standard MoE layout (per-expert grouped output)
    # ------------------------------------------------------------------

    _STDMOE_KERNEL_TYPES = {"intra_node", "inter_node_v1_ll"}

    def dispatch_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch tokens and output in per-expert layout for grouped GEMM.

        Unlike :meth:`dispatch` which returns tokens in a flat buffer,
        this method arranges received tokens by their destination expert,
        producing a layout directly consumable by grouped GEMM::

            [num_local_experts, max_tokens_per_expert, hidden_dim]

        Each expert's slice contains only the tokens routed to it, with
        ``recv_count[e]`` valid rows for expert ``e``.

        Requires MORI built with ``ENABLE_STANDARD_MOE_ADAPT=ON``.
        Only supported for ``intra_node`` and ``inter_node_v1_ll`` kernels.

        Args:
            input: Token embeddings, shape ``[num_tokens, hidden_dim]``.
            weights: Routing weights, shape ``[num_tokens, topk]``.
            indices: Expert indices, shape ``[num_tokens, topk]``, int32.
            scales: Optional per-token scales for quantized paths.
            block_num: Override GPU block count.
            rdma_block_num: Override RDMA block count.
            warp_per_block: Override warps-per-block.

        Returns:
            Tuple of ``(packed_tokens, recv_count, src_info)``:

            - ``packed_tokens``: Per-expert token tensor, shape
              ``[num_local_experts, max_tokens_per_expert, hidden_dim]``.
              Expert ``e`` has ``recv_count[e]`` valid rows.
            - ``recv_count``: Number of valid tokens per expert, shape
              ``[num_local_experts]``, int32.
            - ``src_info``: Source token provenance metadata, shape
              ``[num_local_experts, max_tokens_per_expert]``, int32.
              Used internally by :meth:`combine_standard_moe`.
        """
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        if scales is None:
            scales = torch.empty(
                input.size(0), 0, dtype=torch.float32, device=input.device,
            )

        packed_tokens, recv_count, src_info, _ = _run_mori_op_on_stream(
            lambda: self._op.dispatch_standard_moe(
                input, weights, scales, indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            ),
            async_finish,
            allocate_on_comm_stream,
        )

        return packed_tokens, recv_count, src_info

    def combine_standard_moe(
        self,
        expert_output: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Combine expert outputs from per-expert layout back to original ranks.

        Accepts expert output in the same per-expert layout produced by
        :meth:`dispatch_standard_moe`::

            [num_local_experts, max_tokens_per_expert, hidden_dim]

        Requires MORI built with ``ENABLE_STANDARD_MOE_ADAPT=ON``.
        Only supported for ``intra_node`` and ``inter_node_v1_ll`` kernels.

        Args:
            expert_output: Per-expert output tensor, shape
                ``[num_local_experts, max_tokens_per_expert, hidden_dim]``.
            weights: Routing weights from the original dispatch, shape
                ``[num_tokens, topk]``.
            indices: Expert indices from the original dispatch, shape
                ``[num_tokens, topk]``, int32.
            block_num: Override GPU block count.
            rdma_block_num: Override RDMA block count.
            warp_per_block: Override warps-per-block.

        Returns:
            Tuple of ``(output, output_weights)``:

            - ``output``: Combined token embeddings, shape
              ``[max_num_inp_token_per_rank, hidden_dim]``.
            - ``output_weights``: ``None`` (standard MoE combine does not
              return accumulated weights).
        """
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        output, output_weights = _run_mori_op_on_stream(
            lambda: self._op.combine_standard_moe(
                expert_output, weights, indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            ),
            async_finish,
            allocate_on_comm_stream,
        )

        return output, output_weights

    def convert_dispatch_to_standard(
        self,
        dispatch_tokens: torch.Tensor,
        dispatch_indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert flat dispatch output to per-expert layout.

        Takes the flat output from :meth:`dispatch` and rearranges it into
        the per-expert layout expected by grouped GEMM. Useful when you want
        to use the regular :meth:`dispatch` (which supports all kernel types)
        but still need per-expert layout for the expert computation.

        Requires MORI built with ``ENABLE_STANDARD_MOE_ADAPT=ON``.

        Args:
            dispatch_tokens: Flat dispatch output, shape ``[max_recv, hidden_dim]``.
            dispatch_indices: Expert indices from dispatch output, shape
                ``[max_recv, topk]``, int32.
            block_num: Override GPU block count.
            warp_per_block: Override warps-per-block.

        Returns:
            Tuple of ``(packed_tokens, recv_count, src_info)`` — same as
            :meth:`dispatch_standard_moe`.
        """
        if dispatch_indices.dtype != torch.int32:
            dispatch_indices = dispatch_indices.to(torch.int32)

        packed_tokens, recv_count, src_info, _ = _run_mori_op_on_stream(
            lambda: self._op.convert_dispatch_output(
                dispatch_tokens, dispatch_indices,
                block_num=block_num,
                warp_per_block=warp_per_block,
            ),
            async_finish,
            allocate_on_comm_stream,
        )

        return packed_tokens, recv_count, src_info

    def convert_standard_to_combine_input(
        self,
        packed_tokens: torch.Tensor,
        src_info: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        """Convert per-expert layout back to flat layout for :meth:`combine`.

        Takes expert output in per-expert layout and converts it to the flat
        layout expected by :meth:`combine`. Useful when you used
        :meth:`convert_dispatch_to_standard` for expert computation but
        want to use the regular :meth:`combine` (which supports all kernel
        types).

        Requires MORI built with ``ENABLE_STANDARD_MOE_ADAPT=ON``.

        Args:
            packed_tokens: Per-expert output, shape
                ``[num_local_experts, max_tokens_per_expert, hidden_dim]``.
            src_info: Source info from :meth:`dispatch_standard_moe` or
                :meth:`convert_dispatch_to_standard`.
            block_num: Override GPU block count.
            warp_per_block: Override warps-per-block.

        Returns:
            Flat combine input, shape ``[max_recv, hidden_dim]``.
        """
        layout_range = torch.empty(0, dtype=torch.int64, device=packed_tokens.device)

        flat_input = _run_mori_op_on_stream(
            lambda: self._op.convert_combine_input(
                packed_tokens, src_info, layout_range,
                block_num=block_num,
                warp_per_block=warp_per_block,
            ),
            async_finish,
            allocate_on_comm_stream,
        )

        return flat_input

    def reset(self) -> None:
        """Reset internal state for the next dispatch/combine cycle.

        Must be called after each complete dispatch + combine round.
        """
        self._op.reset()

    def new_cycle(self) -> "_MoriEPCycleState":
        """Create a new cycle state for use with the flat autograd functions.

        Each forward + backward pass requires one cycle state. The state
        coordinates the paired dispatch/combine calls on the underlying MORI
        operator across the forward and backward passes.

        Returns:
            A :class:`_MoriEPCycleState` to pass to :class:`MoriEPDispatch`
            and :class:`MoriEPCombine`.
        """
        return _MoriEPCycleState(self)

    def new_std_moe_cycle(self) -> "_MoriStdMoECycleState":
        """Create a cycle state for the standard MoE layout autograd functions.

        Like :meth:`new_cycle` but for use with
        :class:`MoriEPDispatchStdMoE` and :class:`MoriEPCombineStdMoE`.

        Returns:
            A :class:`_MoriStdMoECycleState`.
        """
        return _MoriStdMoECycleState(self)

    def dispatch_and_combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_fn,
        scales: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a full dispatch -> expert compute -> combine cycle.

        Convenience method that chains :meth:`dispatch`, expert computation,
        and :meth:`combine` together. Not differentiable -- use
        :class:`MoriEPDispatch` and :class:`MoriEPCombine` for training.

        Args:
            input: Token embeddings, shape ``[num_tokens, hidden_dim]``.
            weights: Routing weights, shape ``[num_tokens, num_experts_per_token]``.
            indices: Expert indices, shape ``[num_tokens, num_experts_per_token]``.
            expert_fn: Callable that takes ``(tokens, indices, num_tokens)`` and
                returns expert output of shape ``[num_tokens, hidden_dim]``.
            scales: Optional per-token scales.

        Returns:
            Tuple of ``(output, output_weights)`` from :meth:`combine`.
        """
        recv_tokens, recv_weights, recv_indices, num_recv = self.dispatch(
            input, weights, indices, scales=scales,
        )

        expert_output = expert_fn(recv_tokens, recv_indices, num_recv)

        output, output_weights = self.combine(
            expert_output, recv_weights, recv_indices,
        )

        self.reset()
        return output, output_weights


# ---------------------------------------------------------------------------
# Autograd support for training
# ---------------------------------------------------------------------------

class _MoriEPCycleState:
    """Shared state between dispatch and combine within one forward+backward pass.

    MORI's dispatch and combine are stateful and paired -- you must call
    dispatch before combine on the same operator, then reset. This state
    object coordinates that pairing across the forward pass and again across
    the backward pass (where the roles are reversed).

    Lifecycle::

        Forward:  dispatch(fwd)  →  expert_fn  →  combine(fwd)  →  reset
        Backward: dispatch(bwd)  →  expert.bwd →  combine(bwd)  →  reset
                  ↑ in combine.backward         ↑ in dispatch.backward

    Sync model (Phase 3 deferred sync): dispatch returns ``total_recv`` as a
    device-resident scalar tensor and saves it here. The single host sync per
    half-cycle (``.item()``) is deferred to the start of the *next* MORI call
    that needs the count to slice MORI's exact-sized input buffer. By then
    the dispatch on the comm stream has long completed, so the ``.item()``
    returns immediately without stalling the host.
    """

    def __init__(self, ep: MoriExpertParallel):
        self.ep = ep
        # Saved from forward dispatch for backward combine
        self.fwd_weights: Optional[torch.Tensor] = None
        self.fwd_indices: Optional[torch.Tensor] = None
        # Number of input tokens on this rank — Python int from ``input.shape[0]``,
        # no GPU sync required.
        self.fwd_num_input: int = 0
        # Device-resident count of valid received rows from the forward dispatch.
        # Shape ``[1]`` int32. ``.item()``-ed only inside MoriEPCombine.forward
        # when MORI needs the count to size its combine input.
        self.fwd_total_recv: Optional[torch.Tensor] = None
        # Saved from backward dispatch (in combine.backward) for backward combine
        # (in dispatch.backward) — padded buffers, plus the device-side count
        # for the backward cycle.
        self.bwd_recv_weights: Optional[torch.Tensor] = None
        self.bwd_recv_indices: Optional[torch.Tensor] = None
        self.bwd_total_recv: Optional[torch.Tensor] = None


class MoriEPDispatch(torch.autograd.Function):
    """Autograd-aware MORI EP dispatch (Phase 3 — deferred sync).

    Forward: dispatches tokens to expert-owning ranks. Returns the *full
    padded* buffers (shape ``[max_recv, ...]``) plus a device-resident
    ``total_recv`` scalar tensor. The caller is responsible for masking /
    slicing using ``total_recv`` if they need only the valid rows. Crucially,
    no ``.item()`` is performed here — the dispatch kernel can complete on
    the comm stream concurrently with the host preparing the next op.

    Backward: combines gradients back from expert ranks (completing the
    backward MORI cycle started by :class:`MoriEPCombine`'s backward).

    Usage::

        state = ep.new_cycle()
        recv_padded, recv_w_padded, recv_idx_padded, total_recv = MoriEPDispatch.apply(
            input, weights, indices, state,
        )
        # Compute on device using total_recv to mask invalid rows, e.g.:
        #   local_idx = rebase_global_to_local_indices(recv_idx_padded, total_recv,
        #                                              rank, num_local_experts)
        #   m_splits = compute_tokens_per_expert_device(local_idx, num_local_experts)
        # ... pass results to grouped GEMM ...
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        state: _MoriEPCycleState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        scales = torch.empty(
            input.size(0), 0, dtype=torch.float32, device=input.device,
        )

        out, out_w, _out_s, out_idx, total_recv = _run_mori_op_on_stream(
            lambda: state.ep._op.dispatch(input, weights, scales, indices),
            async_finish=True,
            allocate_on_comm_stream=True,
        )

        # Save routing info for backward. fwd_num_input is a Python int from
        # input.shape[0] — no GPU sync. fwd_total_recv is a *device tensor*;
        # the .item() that used to happen here is deferred to MoriEPCombine.
        state.fwd_weights = weights.detach()
        state.fwd_indices = indices.detach()
        state.fwd_num_input = input.shape[0]
        state.fwd_total_recv = total_recv.detach().clone()

        ctx.state = state

        # Return the full padded buffers — caller masks via total_recv on device.
        # Clone to decouple from MORI's internal buffers (reset on next cycle).
        return (
            out.clone(),
            out_w.clone(),
            out_idx.clone(),
            total_recv.detach().clone(),
        )

    @staticmethod
    def backward(
        ctx,
        grad_recv_tokens: torch.Tensor,
        grad_recv_weights: torch.Tensor,
        grad_recv_indices: torch.Tensor,
        grad_total_recv: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
        """Complete the backward MORI cycle: combine gradients back to source ranks.

        The backward dispatch (which sent grad_output to expert ranks) was
        already initiated by :meth:`MoriEPCombine.backward` and saved padded
        buffers in ``state``. We now slice them to MORI's exact-size combine
        input using the device-resident ``bwd_total_recv``. By this point the
        backward dispatch has had ample time to complete on the comm stream,
        so the ``.item()`` returns immediately.

        ``grad_total_recv`` is the upstream gradient w.r.t. our forward's
        ``total_recv`` output and is always ``None`` — total_recv is an
        integer count, not a differentiable quantity.
        """
        del grad_recv_weights, grad_recv_indices, grad_total_recv  # unused
        state = ctx.state

        # Deferred sync — backward dispatch is long done by now.
        bwd_n = state.bwd_total_recv.flatten()[0].item()

        output, _ = _run_mori_op_on_stream(
            lambda: state.ep._op.combine(
                grad_recv_tokens[:bwd_n],
                state.bwd_recv_weights[:bwd_n],
                state.bwd_recv_indices[:bwd_n],
            ),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        state.ep._op.reset()

        grad_input = output[:state.fwd_num_input]
        return grad_input, None, None, None


class MoriEPCombine(torch.autograd.Function):
    """Autograd-aware MORI EP combine (Phase 3 — deferred sync).

    Forward: combines expert outputs back to original ranks. Accepts the
    *padded* expert output ``[max_recv, ...]`` produced downstream of
    :class:`MoriEPDispatch`. The single ``.item()`` on ``state.fwd_total_recv``
    happens here, deferred from the original dispatch site — by this point
    dispatch has had time to complete on the comm stream and the sync is
    effectively free.

    Backward: dispatches gradients to expert-owning ranks (starting the
    backward MORI cycle that :class:`MoriEPDispatch`'s backward completes).
    Saves the padded backward buffers plus a device-resident ``bwd_total_recv``.

    Usage::

        output, output_weights = MoriEPCombine.apply(
            expert_output_padded, recv_weights_padded, recv_indices_padded, state,
        )
    """

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        recv_weights: torch.Tensor,
        recv_indices: torch.Tensor,
        state: _MoriEPCycleState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if recv_indices.dtype != torch.int32:
            recv_indices = recv_indices.to(torch.int32)

        # Deferred host sync — dispatch ran on the comm stream while host was
        # busy with permute/grouped-gemm/etc., so this .item() resolves
        # immediately. MORI's combine requires its input slice to be exactly
        # total_recv rows; we slice the padded buffers here.
        n_recv = state.fwd_total_recv.flatten()[0].item()

        output, output_w = _run_mori_op_on_stream(
            lambda: state.ep._op.combine(
                expert_output[:n_recv],
                recv_weights[:n_recv],
                recv_indices[:n_recv],
            ),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        state.ep._op.reset()  # forward cycle complete

        ctx.state = state
        num_input = state.fwd_num_input

        # Clone the valid portion to decouple from MORI buffers
        return (
            output[:num_input].clone(),
            output_w[:num_input].clone() if output_w is not None else None,
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_output_weights: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
        """Start the backward MORI cycle: dispatch gradients to expert ranks.

        This sends ``grad_output`` (on the token-originating ranks) to the
        expert-owning ranks, using the same routing as the forward dispatch.
        The dispatched gradients then flow through ``expert.backward`` via
        normal autograd, and finally :meth:`MoriEPDispatch.backward` combines
        them back. No ``.item()`` here — backward stays on the device and the
        sync is deferred to :meth:`MoriEPDispatch.backward`.
        """
        del grad_output_weights  # unused — accumulated weights have no upstream grad
        state = ctx.state

        # Dispatch gradients using the same routing as forward
        scales = torch.empty(
            grad_output.size(0), 0, dtype=torch.float32, device=grad_output.device,
        )
        out, out_w, _out_s, out_idx, total_recv = _run_mori_op_on_stream(
            lambda: state.ep._op.dispatch(
                grad_output,
                state.fwd_weights,
                scales,
                state.fwd_indices,
            ),
            async_finish=True,
            allocate_on_comm_stream=True,
        )

        # Save padded buffers + device-resident count for MoriEPDispatch.backward.
        state.bwd_recv_weights = out_w.detach().clone()
        state.bwd_recv_indices = out_idx.detach().clone()
        state.bwd_total_recv = total_recv.detach().clone()

        grad_expert_output = out.clone()
        return grad_expert_output, None, None, None


# ---------------------------------------------------------------------------
# Standard MoE autograd (per-expert layout)
# ---------------------------------------------------------------------------

class _MoriStdMoECycleState:
    """Shared state for standard MoE dispatch/combine autograd cycle.

    Like :class:`_MoriEPCycleState` but for the standard MoE layout path
    where tokens are arranged per-expert.

    Lifecycle::

        Forward:  dispatch_standard_moe → expert_fn → combine_standard_moe → reset
        Backward: dispatch_standard_moe → expert.bwd → combine_standard_moe → reset
    """

    def __init__(self, ep: MoriExpertParallel):
        self.ep = ep
        self.fwd_weights: Optional[torch.Tensor] = None
        self.fwd_indices: Optional[torch.Tensor] = None
        self.fwd_num_input: int = 0
        self.fwd_recv_count: Optional[torch.Tensor] = None
        self.fwd_src_info: Optional[torch.Tensor] = None
        # Backward state
        self.bwd_recv_count: Optional[torch.Tensor] = None
        self.bwd_src_info: Optional[torch.Tensor] = None


class MoriEPDispatchStdMoE(torch.autograd.Function):
    """Autograd-aware MORI EP dispatch with standard MoE per-expert layout.

    Forward: dispatches tokens and arranges output as
    ``[num_local_experts, max_tokens_per_expert, hidden_dim]``.
    Backward: combines gradients using :meth:`combine_standard_moe`.

    Usage::

        state = ep.new_std_moe_cycle()
        packed, recv_count, src_info = MoriEPDispatchStdMoE.apply(
            input, weights, indices, state,
        )
        # packed: [num_local_experts, max_tokens_per_expert, hidden_dim]
        # recv_count: [num_local_experts] -- valid tokens per expert
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        state: _MoriStdMoECycleState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        scales = torch.empty(
            input.size(0), 0, dtype=torch.float32, device=input.device,
        )

        packed, recv_count, src_info, _ = _run_mori_op_on_stream(
            lambda: state.ep._op.dispatch_standard_moe(
                input, weights, scales, indices,
            ),
            async_finish=True,
            allocate_on_comm_stream=True,
        )

        state.fwd_weights = weights.detach()
        state.fwd_indices = indices.detach()
        state.fwd_num_input = input.shape[0]
        state.fwd_recv_count = recv_count.clone()
        state.fwd_src_info = src_info.clone()

        ctx.state = state

        return packed.clone(), recv_count.clone(), src_info.clone()

    @staticmethod
    def backward(
        ctx,
        grad_packed: torch.Tensor,
        grad_recv_count: torch.Tensor,
        grad_src_info: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
        state = ctx.state

        output, _ = _run_mori_op_on_stream(
            lambda: state.ep._op.combine_standard_moe(
                grad_packed,
                state.bwd_recv_count,  # dummy -- combine uses internal state
                state.bwd_src_info,    # not directly used but kept for consistency
            ),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        state.ep._op.reset()

        grad_input = output[:state.fwd_num_input]
        return grad_input, None, None, None


class MoriEPCombineStdMoE(torch.autograd.Function):
    """Autograd-aware MORI EP combine with standard MoE per-expert layout.

    Forward: combines expert outputs from per-expert layout back to
    original ranks and resets the cycle.
    Backward: dispatches gradients using :meth:`dispatch_standard_moe`.

    Usage::

        output, _ = MoriEPCombineStdMoE.apply(
            expert_output, weights, indices, state,
        )
    """

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        state: _MoriStdMoECycleState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        output, output_w = _run_mori_op_on_stream(
            lambda: state.ep._op.combine_standard_moe(
                expert_output, weights, indices,
            ),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        state.ep._op.reset()

        ctx.state = state
        num_input = state.fwd_num_input

        return (
            output[:num_input].clone(),
            output_w[:num_input].clone() if output_w is not None else None,
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_output_weights: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
        state = ctx.state

        scales = torch.empty(
            grad_output.size(0), 0, dtype=torch.float32, device=grad_output.device,
        )
        packed, recv_count, src_info, _ = _run_mori_op_on_stream(
            lambda: state.ep._op.dispatch_standard_moe(
                grad_output,
                state.fwd_weights,
                scales,
                state.fwd_indices,
            ),
            async_finish=True,
            allocate_on_comm_stream=True,
        )

        state.bwd_recv_count = recv_count
        state.bwd_src_info = src_info

        return packed.clone(), None, None, None


# ---------------------------------------------------------------------------
# Megatron-compatible adapters
# ---------------------------------------------------------------------------
# These wrap tealite's MoriEPDispatch / MoriEPCombine so they can be dropped
# into Megatron's ``_MoriManager`` via the ``MOE_MORI_BACKEND=tealite`` env var
# (handled in megatron/core/transformer/moe/fused_a2a.py). The adapters slice
# padded outputs back to total_recv at this boundary to match Megatron's
# contract — _MoriManager's downstream code (``_indices_to_multihot``, the
# permute step, GroupedLinear's m_splits list) all consume sliced buffers.
#
# Accepting the ``.item()`` at the adapter boundary means we lose tealite's
# deferred-sync benefit on the dispatch side. The intent here is to validate
# correctness of the tealite primitives under Megatron's real workload before
# committing to a deeper _MoriManager refactor that consumes padded buffers
# end-to-end.

_megatron_adapter_ep: Optional["MoriExpertParallel"] = None
_megatron_adapter_state: Optional["_MoriEPCycleState"] = None


def _adapter_bootstrap_shmem_flag() -> None:
    """Tell tealite that MORI shmem is already initialized externally.

    Megatron's ``pretrain_gpt.py`` calls ``mori.shmem.shmem_torch_process_group_init()``
    directly during entry, before any tealite code runs. That bypasses
    :func:`init_mori_ep`, so :data:`_mori_shmem_initialized` is still False
    when the adapter first creates a :class:`MoriExpertParallel`. We flip the
    flag here to allow construction without re-initializing MORI shmem
    (which would error).
    """
    global _mori_shmem_initialized
    _mori_shmem_initialized = True


def _get_megatron_adapter_ep(
    group: "torch.distributed.ProcessGroup",
    hidden_dim: int,
    num_local_experts: int,
    router_topk: int,
    max_num_tokens_per_rank: int,
    dtype: torch.dtype,
) -> "MoriExpertParallel":
    """Lazily create the process-wide ``MoriExpertParallel`` for the adapter.

    Mirrors Megatron's :func:`get_mori_op` singleton pattern. The wrapper is
    created on the first dispatch call and reused for the lifetime of the
    process. Megatron's training loop holds (hidden_dim, world_size,
    num_local_experts, router_topk, max_num_tokens_per_rank) constant across
    layers and steps, so a single instance suffices.
    """
    global _megatron_adapter_ep
    if _megatron_adapter_ep is not None:
        return _megatron_adapter_ep

    if not is_mori_ep_initialized():
        _adapter_bootstrap_shmem_flag()

    rank = torch.distributed.get_rank(group)
    world_size = group.size()
    # Match Megatron's kernel-type heuristic (fused_a2a.py:665-669).
    kernel_type = "intra_node" if world_size <= 8 else "inter_node_v1"

    _megatron_adapter_ep = MoriExpertParallel(
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        max_num_inp_token_per_rank=max_num_tokens_per_rank,
        dtype=dtype,
        kernel_type=kernel_type,
    )
    return _megatron_adapter_ep


def _reset_megatron_adapter() -> None:
    """Drop the cached EP wrapper and any pending cycle state.

    Call when EP layout changes (e.g., between test parametrizations) so the
    next dispatch reconfigures the underlying MORI op. The state object
    survives via autograd ctx references regardless of clearing here.
    """
    global _megatron_adapter_ep, _megatron_adapter_state
    if _megatron_adapter_ep is not None:
        _megatron_adapter_ep.reset()
    _megatron_adapter_ep = None
    _megatron_adapter_state = None


def tealite_mori_dispatch_for_megatron(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_experts: int,
    group: "torch.distributed.ProcessGroup",
    num_local_experts: int,
    router_topk: int,
    max_num_tokens_per_rank: int,
    fp8_dispatch: bool = False,
    async_finish: bool = True,
    allocate_on_comm_stream: bool = True,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Megatron-shape dispatch adapter, internals routed through tealite.

    Drop-in for Megatron's ``mori_dispatch``: same call signature, same
    5-tuple return, all tensors sliced to ``total_recv``. The autograd
    Functions invoked are :class:`MoriEPDispatch` from tealite.

    Returns
    -------
    (recv_x, recv_token_indices_local, recv_token_probs, tokens_per_expert,
     recv_token_indices_global)

    Notes
    -----
    The host ``.item()`` for slicing happens here at the adapter boundary,
    matching Megatron's contract. Tealite's deferred-sync benefit is
    intentionally not realized in this adapter; that requires _MoriManager
    to consume padded buffers, which is a separate refactor. The
    ``tokens_per_expert`` count uses the sync-free :func:`compute_tokens_per_expert_device`
    helper instead of ``torch.bincount`` — a small win that's free here
    because the upstream ``.item()`` already serialized the host.
    """
    if fp8_dispatch:
        raise NotImplementedError(
            "FP8 dispatch is not yet supported in the tealite MORI adapter. "
            "Run with MOE_MORI_BACKEND=megatron or TE_FP8=0 for now."
        )
    del num_experts, async_finish, allocate_on_comm_stream  # not threaded yet

    global _megatron_adapter_state

    ep = _get_megatron_adapter_ep(
        group=group,
        hidden_dim=x.shape[1],
        num_local_experts=num_local_experts,
        router_topk=router_topk,
        max_num_tokens_per_rank=max_num_tokens_per_rank,
        dtype=x.dtype,
    )

    state = ep.new_cycle()
    _megatron_adapter_state = state

    # Tealite dispatch — returns padded buffers + device-side total_recv.
    out_padded, out_w_padded, out_idx_padded, total_recv_dev = MoriEPDispatch.apply(
        x,
        token_probs.float() if token_probs.dtype != torch.float32 else token_probs,
        token_indices,
        state,
    )

    # Boundary slice — Megatron's _MoriManager expects exact-size buffers.
    total_recv = total_recv_dev.flatten()[0].item()
    recv_x = out_padded[:total_recv]
    recv_token_probs = out_w_padded[:total_recv]
    recv_token_indices_global = out_idx_padded[:total_recv]

    # Rebase global expert IDs to local space with -1 sentinels (matches
    # the contract _indices_to_multihot expects). Same logic as Megatron's
    # MoriDispatch.forward:779-790.
    my_rank = torch.distributed.get_rank(group)
    local_id_start = my_rank * num_local_experts
    local_id_end = local_id_start + num_local_experts
    is_local = (recv_token_indices_global >= local_id_start) & (
        recv_token_indices_global < local_id_end
    )
    recv_token_indices = (recv_token_indices_global - local_id_start).to(torch.int64)
    recv_token_indices = torch.where(
        is_local,
        recv_token_indices,
        torch.full_like(recv_token_indices, -1),
    )

    # Sync-free per-local-expert count via the tealite helper. Equivalent to
    # Megatron's ``torch.bincount(recv_token_indices[is_local], ...)`` but
    # avoids both the boolean-mask indexing sync and bincount's internal
    # ``.max().item()``.
    tokens_per_expert = compute_tokens_per_expert_device(
        recv_token_indices, num_local_experts,
    )

    return (
        recv_x,
        recv_token_indices,
        recv_token_probs,
        tokens_per_expert,
        recv_token_indices_global,
    )


def tealite_mori_combine_for_megatron(
    x: torch.Tensor,
    group: "torch.distributed.ProcessGroup",
    token_indices: torch.Tensor,
    recv_token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_local_experts: int,
    router_topk: int,
    max_num_tokens_per_rank: int,
    fp8_dispatch: bool = False,
    async_finish: bool = True,
    allocate_on_comm_stream: bool = True,
) -> torch.Tensor:
    """Megatron-shape combine adapter, internals routed through tealite.

    Drop-in for Megatron's ``mori_combine``. Reads the cycle state set by
    :func:`tealite_mori_dispatch_for_megatron` and resets it afterwards
    (the state survives into backward via autograd ctx references).

    Parameters mirror Megatron's mori_combine — ``recv_token_indices`` is the
    receiver-side GLOBAL indices (size [total_recv, topk]) and ``token_probs``
    is the receiver-side probs. Both have already been sliced to total_recv
    by the dispatch adapter.
    """
    if fp8_dispatch:
        raise NotImplementedError(
            "FP8 combine is not yet supported in the tealite MORI adapter."
        )
    del group, num_local_experts, router_topk, max_num_tokens_per_rank  # unused
    del async_finish, allocate_on_comm_stream

    global _megatron_adapter_state
    state = _megatron_adapter_state
    if state is None:
        raise RuntimeError(
            "tealite_mori_combine_for_megatron called without a paired "
            "dispatch; ensure tealite_mori_dispatch_for_megatron ran first."
        )

    num_tokens = token_indices.shape[0]

    # MoriEPCombine.forward slices its padded inputs to state.fwd_total_recv.
    # Since the dispatch adapter already sliced down to total_recv, x is
    # already exact-size — fwd_total_recv equals x.shape[0] and the internal
    # slice is a no-op. The deferred .item() on state.fwd_total_recv here is
    # essentially free (the value was already materialized).
    output, _ = MoriEPCombine.apply(
        x,
        token_probs.float() if token_probs.dtype != torch.float32 else token_probs,
        recv_token_indices,
        state,
    )

    _megatron_adapter_state = None

    # MoriEPCombine returned output sliced to state.fwd_num_input (= sender-side
    # num_tokens captured at dispatch). Defensive re-slice in case the consumer
    # passed a wrapped tensor whose shape doesn't match (shouldn't happen).
    if output.shape[0] != num_tokens:
        output = output[:num_tokens]

    return output
