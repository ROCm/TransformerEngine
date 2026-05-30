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

Training usage (with autograd)::

    state = ep.new_cycle()
    recv, recv_w, recv_idx = MoriEPDispatch.apply(tokens, weights, indices, state)
    expert_out = run_experts(recv)           # normal autograd
    weighted = expert_out * recv_w[..., None]  # apply routing weights
    output, _ = MoriEPCombine.apply(weighted, recv_w, recv_idx, state)
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

        out, out_weights, _out_scales, out_indices, total_recv = self._op.dispatch(
            input, weights, scales, indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

        torch.cuda.synchronize()
        num_recv = total_recv[0].item()

        return out, out_weights, out_indices, num_recv

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Combine expert outputs back to original ranks.

        Args:
            input: Expert output embeddings for tokens this rank processed,
                shape ``[num_recv_tokens, hidden_dim]``.
            weights: Routing weights returned from :meth:`dispatch`.
            indices: Expert indices returned from :meth:`dispatch`.
            block_num: Override GPU block count for this launch.
            warp_per_block: Override warps-per-block for this launch.

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

        output, output_weights = self._op.combine(
            input, weights, indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

        torch.cuda.synchronize()
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

        packed_tokens, recv_count, src_info, _ = self._op.dispatch_standard_moe(
            input, weights, scales, indices,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )

        torch.cuda.synchronize()
        return packed_tokens, recv_count, src_info

    def combine_standard_moe(
        self,
        expert_output: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
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

        output, output_weights = self._op.combine_standard_moe(
            expert_output, weights, indices,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )

        torch.cuda.synchronize()
        return output, output_weights

    def convert_dispatch_to_standard(
        self,
        dispatch_tokens: torch.Tensor,
        dispatch_indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
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

        packed_tokens, recv_count, src_info, _ = self._op.convert_dispatch_output(
            dispatch_tokens, dispatch_indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

        torch.cuda.synchronize()
        return packed_tokens, recv_count, src_info

    def convert_standard_to_combine_input(
        self,
        packed_tokens: torch.Tensor,
        src_info: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
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

        flat_input = self._op.convert_combine_input(
            packed_tokens, src_info, layout_range,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

        torch.cuda.synchronize()
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
    """

    def __init__(self, ep: MoriExpertParallel):
        self.ep = ep
        # Saved from forward dispatch for backward combine
        self.fwd_weights: Optional[torch.Tensor] = None
        self.fwd_indices: Optional[torch.Tensor] = None
        self.fwd_num_input: int = 0
        self.fwd_num_recv: int = 0
        # Saved from backward dispatch (in combine.backward) for backward combine
        # (in dispatch.backward)
        self.bwd_recv_weights: Optional[torch.Tensor] = None
        self.bwd_recv_indices: Optional[torch.Tensor] = None
        self.bwd_num_recv: int = 0


class MoriEPDispatch(torch.autograd.Function):
    """Autograd-aware MORI EP dispatch.

    Forward: dispatches tokens to expert-owning ranks.
    Backward: combines gradients back from expert ranks (completing the
    backward MORI cycle started by :class:`MoriEPCombine`'s backward).

    Usage::

        state = ep.new_cycle()
        recv_tokens, recv_weights, recv_indices = MoriEPDispatch.apply(
            input, weights, indices, state,
        )
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        state: _MoriEPCycleState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        scales = torch.empty(
            input.size(0), 0, dtype=torch.float32, device=input.device,
        )

        out, out_w, _out_s, out_idx, total_recv = state.ep._op.dispatch(
            input, weights, scales, indices,
        )
        torch.cuda.synchronize()
        num_recv = total_recv[0].item()

        # Save routing info for backward
        state.fwd_weights = weights.detach()
        state.fwd_indices = indices.detach()
        state.fwd_num_input = input.shape[0]
        state.fwd_num_recv = num_recv

        ctx.state = state

        # Return only valid rows -- clone to decouple from MORI's internal buffers
        return (
            out[:num_recv].clone(),
            out_w[:num_recv].clone(),
            out_idx[:num_recv].clone(),
        )

    @staticmethod
    def backward(
        ctx,
        grad_recv_tokens: torch.Tensor,
        grad_recv_weights: torch.Tensor,
        grad_recv_indices: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], None, None, None]:
        """Complete the backward MORI cycle: combine gradients back to source ranks.

        The backward dispatch (which sent grad_output to expert ranks) was
        already initiated by :meth:`MoriEPCombine.backward`. Now we combine
        the gradients that flowed through ``expert.backward`` back to the
        original token-owning ranks.
        """
        state = ctx.state

        output, _ = state.ep._op.combine(
            grad_recv_tokens,
            state.bwd_recv_weights,
            state.bwd_recv_indices,
        )
        torch.cuda.synchronize()
        state.ep._op.reset()

        grad_input = output[:state.fwd_num_input]
        return grad_input, None, None, None


class MoriEPCombine(torch.autograd.Function):
    """Autograd-aware MORI EP combine.

    Forward: combines expert outputs back to original ranks and resets
    the forward MORI cycle.
    Backward: dispatches gradients to expert-owning ranks (starting the
    backward MORI cycle that :class:`MoriEPDispatch`'s backward completes).

    Usage::

        output, output_weights = MoriEPCombine.apply(
            expert_output, recv_weights, recv_indices, state,
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

        output, output_w = state.ep._op.combine(
            expert_output, recv_weights, recv_indices,
        )
        torch.cuda.synchronize()
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
        them back.
        """
        state = ctx.state

        # Dispatch gradients using the same routing as forward
        scales = torch.empty(
            grad_output.size(0), 0, dtype=torch.float32, device=grad_output.device,
        )
        out, out_w, _out_s, out_idx, total_recv = state.ep._op.dispatch(
            grad_output,
            state.fwd_weights,
            scales,
            state.fwd_indices,
        )
        torch.cuda.synchronize()
        bwd_num_recv = total_recv[0].item()

        # Save for dispatch.backward to complete the backward cycle
        state.bwd_recv_weights = out_w[:bwd_num_recv]
        state.bwd_recv_indices = out_idx[:bwd_num_recv]
        state.bwd_num_recv = bwd_num_recv

        grad_expert_output = out[:bwd_num_recv].clone()
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

        packed, recv_count, src_info, _ = state.ep._op.dispatch_standard_moe(
            input, weights, scales, indices,
        )
        torch.cuda.synchronize()

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

        output, _ = state.ep._op.combine_standard_moe(
            grad_packed,
            state.bwd_recv_count,  # dummy -- combine uses internal state
            state.bwd_src_info,    # not directly used but kept for consistency
        )
        torch.cuda.synchronize()
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

        output, output_w = state.ep._op.combine_standard_moe(
            expert_output, weights, indices,
        )
        torch.cuda.synchronize()
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
        packed, recv_count, src_info, _ = state.ep._op.dispatch_standard_moe(
            grad_output,
            state.fwd_weights,
            scales,
            state.fwd_indices,
        )
        torch.cuda.synchronize()

        state.bwd_recv_count = recv_count
        state.bwd_src_info = src_info

        return packed.clone(), None, None, None
