# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Fused expert-parallel MoE layer (KOSMOS): the fused router, the routed experts and the shared expert, with the
token dispatch and combine, in exactly one kernel launch for the forward and one for the backward."""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

import transformer_engine_torch as tex

from .base import TransformerEngineBaseModule, get_dummy_wgrad
from ..constants import dist_group_type
from ..cpu_offload import mark_not_offload
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..utils import get_default_init_method

__all__ = ["MoE", "MoEWorkspace", "fc1_from_gate_up", "fc1_to_gate_up"]

_ACTIVATIONS = {"swiglu": 0, "swiglu_clamp10": 1}
# The backward's wgrad_accumulate by main_grad dtype: added into fp32 (1) or bf16 (2).
_ACC_OF_DTYPE = {torch.float32: 1, torch.bfloat16: 2}
_RELEASE_RECLAIMED = 3
_ROUTER_SCORES = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2, "softmax_pre": 3}


def _make_allgather(group: dist_group_type) -> Callable[[bytes], bytes]:
    """Host all-gather of bytes over ``group`` (the EP group), for the fused MoE's collective setup calls."""
    n = dist.get_world_size(group)
    if dist.get_backend(group) == "gloo":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", torch.cuda.current_device())

    def allgather(send: bytes) -> bytes:
        mine = torch.frombuffer(bytearray(send), dtype=torch.uint8).to(device)
        out = torch.empty(n * mine.numel(), dtype=torch.uint8, device=device)
        dist.all_gather_into_tensor(out, mine, group=group)
        return out.cpu().numpy().tobytes()

    return allgather


class _IpcPool:
    """The buffers the EP ranks write into each other (shared by IPC handle and offset): PyTorch tensors in a
    dedicated torch.cuda.MemPool of plain hipMalloc segments. IPC handles work on those whatever the caching
    allocator's expandable_segments setting, PyTorch counts them, and their segments hold nothing else."""

    _pools: Dict[int, "torch.cuda.MemPool"] = {}

    def __init__(self, device: torch.device) -> None:
        self.device = device
        idx = device.index
        if idx not in _IpcPool._pools:
            alloc = torch.cuda.memory.CUDAPluggableAllocator(
                tex.__file__, "nvte_moe_ipc_malloc", "nvte_moe_ipc_free"
            )
            _IpcPool._pools[idx] = torch.cuda.MemPool(alloc.allocator())
        self.pool = _IpcPool._pools[idx]
        self.live: Dict[int, torch.Tensor] = {}

    def alloc(self, nbytes: int) -> int:
        with torch.cuda.use_mem_pool(self.pool, device=self.device):
            t = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        self.live[t.data_ptr()] = t
        return t.data_ptr()

    def free(self, ptr: int) -> bool:
        return self.live.pop(ptr, None) is not None


class MoEWorkspace:
    """What a fused-MoE call needs while it runs, plus the training saved-state pool shared by every
    :class:`MoE` module built on it (one per configuration and EP group).

    Collective lifetime: every EP rank creates it, its modules' forwards and backwards run in the same order
    on every rank, and :meth:`finalize` must be called on every rank before the EP process group is
    destroyed. It is not run at interpreter exit: a process that ends without it just releases its memory
    with the process, so one rank failing never waits on the others.

    Parameters
    ----------
    hidden_size, ffn_hidden_size : int
        Hidden size H and per-expert (and shared-expert) intermediate size I, multiples of 256 (MXFP8: H a
        multiple of 1024).
    num_experts : int
        Global experts, 8 times the experts per rank.
    topk : int
        Experts per token, in {1, 2, 4, 6, 8, 10}.
    num_tokens : int
        Tokens per rank T of every call, a multiple of 256 (512 with the shared expert).
    ep_group : ProcessGroup
        The expert-parallel group (8 ranks, one node).
    capacity_rows : int
        Receive rows per rank to reserve; a call that needs more runs in rounds inside its launch.
    router : str
        The fused router's score function: ``"softmax"``, ``"sigmoid"``, ``"sqrtsoftplus"`` or
        ``"softmax_pre"`` (Megatron semantics).
    router_groups, router_group_topk : int, default = 0
        Group-limited routing: the top k within the ``router_group_topk`` best of ``router_groups`` groups.
    aux_loss_groups : int, default = 0
        0: no aux loss; 1: over the batch; n > 1: per sequence, token t in sequence t % n.
    aux_loss_coeff : float, default = 0.0
    router_scale : float, default = 1.0
        Routing-weight scale.
    shared_expert : bool, default = False
        The shared expert (intermediate size I), in the same launches.
    wgcap : int, default = 240
        Workgroups per launch (at most one per CU), in [9, 256]. Keep it below the CU count when other
        kernels (e.g. collectives on side streams) must make progress concurrently.
    training : bool, default = True
        Allocate the backward's state (needed with MXFP8).
    pool_bytes : int, default = 0
        Saved-state pool bytes per rank (0: one call's block); see :meth:`pool_resize`.
    activation : str, default = "swiglu"
        ``"swiglu"`` or ``"swiglu_clamp10"`` (DeepSeek-V4's clamped SwiGLU).
    mxfp8 : bool, default = False
        MXFP8 GEMM operands; the kernels quantize the bf16 activations.
    fp8_combine : bool, default = False
        MXFP8: MXFP8 combines.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        topk: int,
        num_tokens: int,
        ep_group: dist_group_type,
        capacity_rows: int,
        router: str,
        router_groups: int = 0,
        router_group_topk: int = 0,
        aux_loss_groups: int = 0,
        aux_loss_coeff: float = 0.0,
        router_scale: float = 1.0,
        shared_expert: bool = False,
        wgcap: int = 240,
        training: bool = True,
        pool_bytes: int = 0,
        activation: str = "swiglu",
        mxfp8: bool = False,
        fp8_combine: bool = False,
    ) -> None:
        ep = tex.MOE_EP_SIZE
        if not tex.moe_is_supported():
            raise RuntimeError("The fused MoE needs a TransformerEngine build with it and a gfx950 device.")
        if dist.get_world_size(ep_group) != ep:
            raise ValueError(f"The fused MoE needs an EP group of {ep} ranks.")
        if num_experts % ep:
            raise ValueError(f"num_experts ({num_experts}) must be a multiple of {ep}.")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}.")
        if router not in _ROUTER_SCORES:
            raise ValueError(f"router must be one of {sorted(_ROUTER_SCORES)}.")
        if mxfp8 and not training:
            raise ValueError("MXFP8 with the fused router needs training=True (inference calls do not save).")
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep
        self.topk = topk
        self.num_tokens = num_tokens
        self.training = training
        self.router = router
        self.aux_loss_groups = aux_loss_groups
        self.shared_expert = shared_expert
        self.mxfp8 = mxfp8
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        # Every layer handle created on the workspace, in creation order (the same on every rank). The workspace
        # owns them: a module dropped by Python leaves its handles here, destroyed by finalize(). They are not
        # handed to other modules, as Python drops objects at different points on different ranks while every
        # rank must use the same handle for a call.
        self._layers: List["tex.MoeLayer"] = []
        self._ipc = _IpcPool(torch.device("cuda", torch.cuda.current_device()))
        self._ws = tex.MoeWorkspace(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_local_experts=self.num_local_experts,
            topk=topk,
            num_tokens=num_tokens,
            rank=self.ep_rank,
            allgather=_make_allgather(ep_group),
            ipc_alloc=self._ipc.alloc,
            ipc_free=self._ipc.free,
            capacity_rows=capacity_rows,
            wgcap=wgcap,
            training=training,
            pool_bytes=pool_bytes,
            activation=_ACTIVATIONS[activation],
            mxfp8=mxfp8,
            fp8_combine=fp8_combine,
            shared_expert=shared_expert,
            router=True,
            router_score=_ROUTER_SCORES[router],
            router_groups=router_groups,
            router_group_topk=router_group_topk,
            router_aux_groups=aux_loss_groups,
            router_aux_coeff=aux_loss_coeff,
            router_scale=router_scale,
        )

    def _new_layer(self) -> "tex.MoeLayer":
        if self._ws is None:
            raise RuntimeError("MoEWorkspace used after finalize()")
        layer = tex.MoeLayer(self._ws)
        self._layers.append(layer)
        return layer

    def _forget_layer(self, layer: "tex.MoeLayer") -> None:
        self._layers.remove(layer)

    def finalize(self) -> None:
        """Destroy every layer handle (of live and dropped modules) and the workspace. Collective: every EP rank
        calls it at the same point, before the EP process group is destroyed. Idempotent. Not run at interpreter
        exit (the process's memory goes with it), so that one rank's crash never waits on the others."""
        if self._ws is None:
            return
        layers, self._layers = self._layers, []
        for layer in reversed(layers):
            layer.destroy()
        ws, self._ws = self._ws, None
        ws.destroy()

    def pool_resize(self, nbytes: int) -> int:
        """Re-allocate the saved-state pool at the ranks' least ``nbytes`` (collective, between steps).
        Returns 0, or 1 / 2 when refused (a saved forward awaits its backward / inside a graph capture)."""
        return self._ws.pool_resize(nbytes)

    def pool_info(self) -> Dict[str, int]:
        """The pool's bytes per rank, one call's block and the Xp bytes reserved per layer still to save."""
        return self._ws.pool_info()

    def set_capacity(self, rows: int) -> int:
        """Set the receive capacity (collective, between steps). Returns as :meth:`pool_resize`, or 3 when an
        allocation failed on some rank and the old capacity is kept."""
        return self._ws.set_capacity(rows)

    def capacity_query(self) -> Dict[str, int]:
        """Read and clear the capacity statistics of every layer handle (collective)."""
        return self._ws.capacity_query()

    def capacity_bytes(self, rows: int) -> int:
        """Device bytes :meth:`set_capacity` would allocate at ``rows``."""
        return self._ws.capacity_bytes(rows)


class _Ticket:
    """A saving forward's claim on its layer handle, held by autograd with the forward's other saved state. Dropped
    without its backward (a discarded graph, the extra forward of activation checkpointing), it releases the handle:
    KOSMOS frees its pool block once every rank released it, and the module reuses the handle after that."""

    __slots__ = ("layer", "done")

    def __init__(self, layer: "tex.MoeLayer") -> None:
        self.layer = layer
        self.done = False

    def __del__(self) -> None:
        if not self.done:
            try:
                self.layer.release()
            except Exception:  # pylint: disable=broad-except
                pass


def _main_grads(params, needs) -> Tuple[int, List[Optional[torch.Tensor]]]:
    """fuse_wgrad_accumulation: the kernel adds every weight gradient into ``main_grad`` inside its one
    launch, so every parameter that needs a gradient must have a contiguous, 16-byte aligned ``main_grad``
    of its shape, all of one dtype (fp32 or bf16). Returns the backward's wgrad_accumulate and the buffers
    (a parameter needing no gradient gets a scratch buffer: an allocation, no device work; None if absent)."""
    grads = [getattr(p, "main_grad", None) if p is not None and n else None for p, n in zip(params, needs)]
    for p, n, g in zip(params, needs, grads):
        if not n:
            continue
        if getattr(p, "overwrite_main_grad", False):
            raise NotImplementedError("The fused MoE adds into main_grad; overwrite_main_grad is not supported.")
        if g is None:
            raise ValueError("fuse_wgrad_accumulation: every MoE parameter that needs a gradient needs main_grad.")
        if (
            tuple(g.shape) != tuple(p.shape)
            or not g.is_contiguous()
            or g.device != p.device
            or g.data_ptr() % 16
        ):
            raise ValueError(
                "fuse_wgrad_accumulation: main_grad must be contiguous, 16-byte aligned, of the parameter's shape"
                " and device."
            )
    dtypes = {g.dtype for g in grads if g is not None}
    if not dtypes:
        return 0, grads
    if len(dtypes) != 1 or next(iter(dtypes)) not in _ACC_OF_DTYPE:
        raise ValueError(
            f"fuse_wgrad_accumulation: the MoE's main_grad buffers must all be fp32 or all bf16, got {dtypes}."
        )
    dtype = next(iter(dtypes))
    out = []
    for p, n, g in zip(params, needs, grads):
        if g is None and p is not None:
            # Weights only (the router bias's gradient is optional); written by the kernel, never read.
            g = torch.empty(p.shape, dtype=dtype, device=p.device) if p.dim() > 1 else None
        out.append(g)
    return _ACC_OF_DTYPE[dtype], out


def _after_main_grad(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """The gradient autograd gets for a parameter whose gradient is in ``main_grad``: as TE's Linear (Megatron's
    ``grad_added_to_main_grad`` protocol), a dummy, or None."""
    if not hasattr(param, "grad_added_to_main_grad"):
        return None
    param.grad_added_to_main_grad = True
    if getattr(param, "zero_out_wgrad", False):
        return get_dummy_wgrad(list(param.shape), param.dtype, zero=True)
    return get_dummy_wgrad(
        list(param.shape), param.dtype, zero_tensor=getattr(param, "zero_tensor_dummy_wgrad", False)
    )


class _MoEFunction(torch.autograd.Function):
    """One KOSMOS launch for the forward and one for the backward; nothing else runs on the device."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        router_weight: torch.Tensor,
        router_bias: Optional[torch.Tensor],
        expert_bias: Optional[torch.Tensor],
        shared_fc1_weight: Optional[torch.Tensor],
        shared_fc2_weight: Optional[torch.Tensor],
        shared_gate: Optional[torch.Tensor],
        module: "MoE",
        save: bool,
    ):
        # save is decided by the caller: grad mode is always off inside an autograd.Function's forward.
        ws = module.workspace
        T, H, I, E = x.shape[0], module.hidden_size, module.ffn_hidden_size, module.num_experts
        dev = x.device
        shared = ws.shared_expert
        w1, w2, s1, s2 = fc1_weight, fc2_weight, shared_fc1_weight, shared_fc2_weight
        if ws.mxfp8:
            w1, w2, s1, s2 = module._quantized_weights()
        # Every output below is written by the kernel: allocations only, no device work.
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        aux = torch.empty(1, dtype=torch.float32, device=dev) if ws.aux_loss_groups else None
        tpe = torch.empty(E, dtype=torch.int32, device=dev)
        gs = None
        if shared and ws.training and (save or ws.mxfp8):
            gs = torch.empty(T, 2 * I, dtype=torch.bfloat16, device=dev)
        layer = module._acquire()
        try:
            layer.forward(
                x,
                None,
                None,
                w1,
                w2,
                out,
                save,
                router_weight=router_weight,
                expert_bias=expert_bias,
                router_bias=router_bias,
                aux_loss=aux,
                tokens_per_expert=tpe,
                shared_fc1_weight=s1,
                shared_fc2_weight=s2,
                shared_gate=shared_gate,
                shared_pre_act=gs,
            )
        except Exception:
            module._release(layer)
            raise
        ctx.mark_non_differentiable(tpe)
        # Gradients of outputs nothing differentiates arrive as None, not as zero tensors autograd would fill.
        ctx.set_materialize_grads(False)
        if not save:
            module._release(layer)
            return out, aux, tpe
        ctx.layer = layer
        ctx.ticket = _Ticket(layer)
        ctx.module = module
        ctx.needs = (
            x.requires_grad,
            fc1_weight.requires_grad,
            fc2_weight.requires_grad,
            router_weight.requires_grad,
            router_bias is not None and router_bias.requires_grad,
            shared_fc1_weight is not None and shared_fc1_weight.requires_grad,
            shared_fc2_weight is not None and shared_fc2_weight.requires_grad,
            shared_gate is not None and shared_gate.requires_grad,
        )
        # The backward re-reads the forward's x (and gate) through raw pointers: keep them on the device.
        mark_not_offload(x)
        ctx.save_for_backward(x, shared_gate, gs)
        ctx.weights = (w1, w2, router_weight, s1, s2)
        # The parameters' versions: the backward refuses weights changed in place since this forward, as autograd
        # does for saved tensors (a host-side check, no saved copy).
        ctx.params = tuple(
            p for p in (fc1_weight, fc2_weight, router_weight, router_bias, shared_fc1_weight, shared_fc2_weight)
        )
        ctx.versions = tuple(None if p is None else p._version for p in ctx.params)
        return out, aux, tpe

    @staticmethod
    def backward(ctx, dout, daux, _dtpe):
        if getattr(ctx, "layer", None) is None:
            raise RuntimeError(
                "The fused MoE runs one backward per forward: its saved state is freed by the first backward"
                " (retain_graph=True with a second backward is not supported)."
            )
        for p, v in zip(ctx.params, ctx.versions):
            if p is not None and p._version != v:
                raise RuntimeError(
                    "The fused MoE backward needs the weights of its forward: a parameter was modified in place"
                    f" since (version {v} -> {p._version})."
                )
        x, sgate, gs = ctx.saved_tensors
        w1, w2, wr, s1, s2 = ctx.weights
        module, layer = ctx.module, ctx.layer
        shared = module.workspace.shared_expert
        nx, n1, n2, nr, nrb, ns1, ns2, nsg = ctx.needs
        T, dev = x.shape[0], x.device
        if dout is None or not dout.is_contiguous() or dout.dtype != torch.bfloat16:
            raise ValueError("The fused MoE backward takes a contiguous bf16 output gradient.")
        if daux is None:
            aux_scale = module._zero_scalar
        elif daux.dtype == torch.float32 and daux.is_contiguous():
            aux_scale = daux.reshape(1)
        else:
            raise ValueError("The fused MoE backward takes a contiguous fp32 aux-loss gradient.")
        # Every gradient below is written (overwritten, or with fuse_wgrad_accumulation added into main_grad) by
        # the kernel: allocations only, no device work.
        dx = torch.empty_like(x)
        dsg = torch.empty(T, dtype=torch.float32, device=dev) if shared and nsg else None
        params = (
            module.fc1_weight,
            module.fc2_weight,
            module.router_weight,
            module.router_bias,
            module.shared_fc1_weight if shared else None,
            module.shared_fc2_weight if shared else None,
        )
        wneeds = (n1, n2, nr, nrb, ns1, ns2)
        acc = 0
        if module.fuse_wgrad_accumulation:
            acc, (dW1, dW2, dWr, dlb, dWs1, dWs2) = _main_grads(params, wneeds)
        if not acc:
            p1, p2, pr, _, ps1, ps2 = params
            dW1 = torch.empty(p1.shape, dtype=torch.bfloat16, device=dev)
            dW2 = torch.empty(p2.shape, dtype=torch.bfloat16, device=dev)
            dWr = torch.empty(pr.shape, dtype=torch.bfloat16, device=dev)
            dlb = torch.empty(module.num_experts, dtype=torch.float32, device=dev) if nrb else None
            dWs1 = dWs2 = None
            if shared:
                dWs1 = torch.empty(ps1.shape, dtype=torch.bfloat16, device=dev)
                dWs2 = torch.empty(ps2.shape, dtype=torch.bfloat16, device=dev)
        # The operands are checked: from here the handle is the backward's (a refusal above leaves it to the ticket,
        # which releases the saved state when autograd drops the graph).
        ctx.layer = None
        ctx.ticket.done = True
        try:
            layer.backward(
                dout,
                x,
                w1,
                w2,
                dx,
                None,
                dW1,
                dW2,
                router_weight=wr,
                aux_loss_scale=aux_scale,
                router_wgrad=dWr,
                router_bias_grad=dlb,
                shared_fc1_weight=s1,
                shared_fc2_weight=s2,
                shared_gate=sgate,
                shared_pre_act=gs,
                shared_fc1_wgrad=dWs1,
                shared_fc2_wgrad=dWs2,
                shared_gate_grad=dsg,
                wgrad_accumulate=acc,
            )
        finally:
            module._release(layer)
        if acc:
            dW1, dW2, dWr, dlb, dWs1, dWs2 = (
                _after_main_grad(p) if n else None for p, n in zip(params, wneeds)
            )
        return (
            dx if nx else None,
            dW1 if n1 else None,
            dW2 if n2 else None,
            dWr if nr else None,
            dlb if nrb else None,
            None,
            dWs1 if ns1 else None,
            dWs2 if ns2 else None,
            dsg if nsg else None,
            None,
            None,
        )


class MoE(TransformerEngineBaseModule):
    """Expert-parallel MoE layer ported from KOSMOS: the fused router, the routed SwiGLU experts and the
    (optional) shared expert, with the token dispatch and combine, in exactly one kernel launch for the
    forward and one for the backward (ROCm gfx950, EP = 8 in one node). The module adds no device work of its
    own around those launches.

    ``forward(hidden_states, shared_gate=None)`` returns ``(out, aux_loss, tokens_per_expert)``:

    * ``hidden_states``: bf16, contiguous, exactly the workspace's ``num_tokens`` tokens (``[..., H]``).
    * ``aux_loss``: fp32 ``[1]``, differentiable (None when the workspace has no aux loss).
    * ``tokens_per_expert``: int32 ``[E]``, this rank's routed tokens per expert (for the expert-bias update).
    * ``shared_gate``: with the shared expert, an optional fp32 ``[tokens]`` per-token gate, contiguous and
      16-byte aligned (differentiable; k in {6, 8, 10}): ``out = routed + shared_gate * shared``.

    Parameters
    ----------
    workspace : MoEWorkspace
        Shared workspace; it fixes H, I, the experts, top k, the tokens per rank and the features.
    init_method, output_layer_init_method : Callable, optional
        Initializers of ``router_weight`` and the fc1 weights, and of the fc2 weights.
    add_router_bias : bool, default = False
        A learned fp32 logit bias ``router_bias`` ``[E]``.
    use_expert_bias : bool, default = False
        Sigmoid / sqrtsoftplus router: an fp32 ``expert_bias`` buffer ``[E]`` added to the scores for the
        selection only (aux-loss-free balancing; the caller updates it from ``tokens_per_expert``).

    Notes
    -----
    Parameters: ``router_weight`` ``[E, H]``; ``fc1_weight`` ``[E_loc, 2I, H]`` with each expert's rows in
    256-row blocks of 128 gate rows followed by the matching 128 up rows (see :func:`fc1_from_gate_up`);
    ``fc2_weight`` ``[E_loc, H, I]``; with the shared expert ``shared_fc1_weight`` ``[2I, H]`` (rows
    ``[gate; up]``) and ``shared_fc2_weight`` ``[H, I]``.

    Expert parallelism: ``fc1_weight`` and ``fc2_weight`` hold this rank's experts and carry
    ``allreduce = False`` (Megatron's marker: reduce them over the expert-data-parallel group, never over the EP
    group). The router, its bias and the shared expert are replicated over the EP group: construction (and
    ``reset_parameters``) broadcasts EP rank 0's values, and their gradients are this rank's partial sums, to be
    reduced over the data-parallel group that includes the EP ranks, as any replicated parameter.

    Inputs are used as given, with no copies or casts: ``hidden_states`` must be bf16, contiguous, with exactly
    the workspace's ``num_tokens`` tokens (another token count needs another workspace), and the output gradient
    must be a contiguous bf16 tensor.

    Every EP rank must make the same calls in the same order: a rank that does not make a call (an exception
    before it, a skipped layer) leaves the others waiting inside their kernel. Such a wait is bounded: after 10
    minutes the kernel traps and the process fails with a GPU exception (which may write a GPU core dump; ``ulimit
    -c 0`` disables it) instead of hanging.

    ``fuse_wgrad_accumulation``: the backward kernel adds every weight gradient (and the router bias's) into
    the parameters' ``main_grad`` itself, inside its one launch: all fp32 or all bf16 (each element rounded
    once), contiguous and 16-byte aligned (Megatron's ``grad_added_to_main_grad`` protocol, as TE's Linear;
    ``overwrite_main_grad`` is not supported).

    MXFP8: the kernels read MXFP8 copies of the weights made by :meth:`update_weights`, which the caller
    runs after every change of the weights (e.g. the optimizer step), outside the forward; a forward with
    weights changed since then raises.

    Each forward that saves state for a backward holds a layer handle until that backward runs, so several
    forwards may be in flight (pipeline parallelism). A saving forward whose backward never runs (its graph
    discarded, or the first forward under activation checkpointing) releases its handle when autograd drops it:
    its saved block in the shared pool is freed once every EP rank dropped it, agreed inside the next forwards,
    and the handle is reused a few forwards later.
    """

    def __init__(
        self,
        workspace: MoEWorkspace,
        *,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        add_router_bias: bool = False,
        use_expert_bias: bool = False,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        params_dtype = torch.bfloat16 if params_dtype is None else params_dtype
        if params_dtype != torch.bfloat16:
            raise ValueError("The fused MoE supports bf16 parameters only.")
        if use_expert_bias and workspace.router not in ("sigmoid", "sqrtsoftplus"):
            raise ValueError("expert_bias needs the sigmoid or sqrtsoftplus router")
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        self.workspace = workspace
        self.hidden_size = workspace.hidden_size
        self.ffn_hidden_size = workspace.ffn_hidden_size
        self.num_experts = workspace.num_experts
        self.num_local_experts = workspace.num_local_experts
        self.topk = workspace.topk
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        E, El, H, I = self.num_experts, self.num_local_experts, self.hidden_size, self.ffn_hidden_size
        init_method = get_default_init_method() if init_method is None else init_method
        output_layer_init_method = (
            get_default_init_method() if output_layer_init_method is None else output_layer_init_method
        )

        def param(name, shape, init, dtype=params_dtype):
            self.register_parameter(
                name,
                torch.nn.Parameter(torch.empty(*shape, dtype=dtype, device=device)),
                init_fn=init,
            )

        param("router_weight", (E, H), init_method)
        if add_router_bias:
            param("router_bias", (E,), torch.nn.init.zeros_, torch.float32)
        else:
            self.register_parameter("router_bias", None)
        param("fc1_weight", (El, 2 * I, H), init_method)
        param("fc2_weight", (El, H, I), output_layer_init_method)
        if workspace.shared_expert:
            param("shared_fc1_weight", (2 * I, H), init_method)
            param("shared_fc2_weight", (H, I), output_layer_init_method)
        expert_bias = torch.zeros(E, dtype=torch.float32, device=device) if use_expert_bias else None
        self.register_buffer("expert_bias", expert_bias, persistent=True)
        self.register_buffer(
            "_zero_scalar", torch.zeros(1, dtype=torch.float32, device=device), persistent=False
        )
        self.reset_parameters(defer_init=device.type == "meta")
        # Expert parallelism: fc1 / fc2 hold this rank's experts, so their gradients must not be averaged over the
        # EP group (Megatron's allreduce=False marks expert parameters for its expert-data-parallel reduction).
        # The router and the shared expert are replicated: every EP rank starts from EP rank 0's values, and a
        # data-parallel reduction over the EP ranks keeps them equal (their gradients are per-rank partial sums).
        for name in ("fc1_weight", "fc2_weight"):
            getattr(self, name).allreduce = False
        self._free: List["tex.MoeLayer"] = []
        self._busy: List["tex.MoeLayer"] = []
        self._qweights: Optional[Tuple[Tuple, Tuple]] = None
        if workspace.mxfp8 and device.type != "meta":
            self.update_weights()
        # One handle set up now (collective, as construction already is), so a forward and its backward are one
        # kernel launch each from the first call; prepare(n) sets up n for n forwards in flight.
        if device.type != "meta":
            self.prepare(1)

    def reset_parameters(self, defer_init: Optional[bool] = False) -> None:
        super().reset_parameters(defer_init=defer_init)
        if not defer_init:
            self._broadcast_replicated()

    @torch.no_grad()
    def _broadcast_replicated(self) -> None:
        """Make the replicated parameters (router, router bias, shared expert) EP rank 0's on every EP rank."""
        src = dist.get_global_rank(self.workspace.ep_group, 0)
        for name in ("router_weight", "router_bias", "shared_fc1_weight", "shared_fc2_weight"):
            p = getattr(self, name, None)
            if p is not None:
                dist.broadcast(p.data, src, group=self.workspace.ep_group)

    def _get_weight_tensors(self) -> List[torch.Tensor]:
        return [self.fc1_weight, self.fc2_weight]

    def _get_weight_quantizers(self) -> List[None]:
        return [None, None]

    def _expert_weights(self) -> List[Optional[torch.Tensor]]:
        return [
            self.fc1_weight,
            self.fc2_weight,
            getattr(self, "shared_fc1_weight", None),
            getattr(self, "shared_fc2_weight", None),
        ]

    @staticmethod
    def _tags(weights) -> Tuple:
        return tuple(None if w is None else (w.data_ptr(), w._version) for w in weights)

    @torch.no_grad()
    def update_weights(self) -> None:
        """MXFP8: quantize the expert weights (row- and column-wise) for the kernels. Run it after every change
        of the weights (e.g. after the optimizer step), outside the forward."""
        if not self.workspace.mxfp8:
            return
        weights = self._expert_weights()
        quantized = []
        for w in weights:
            if w is None:
                quantized.append(None)
                continue
            q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
            q.optimize_for_gemm = False
            quantized.append(q(w.detach()))
        self._qweights = (self._tags(weights), tuple(quantized))

    def _quantized_weights(self):
        if self._qweights is None or self._qweights[0] != self._tags(self._expert_weights()):
            raise RuntimeError(
                "The expert weights changed since MoE.update_weights(): call it after every weight update."
            )
        return self._qweights[1]

    def _acquire(self) -> "tex.MoeLayer":
        # Handles whose forward every rank released come back first (in acquisition order); then LIFO, created on
        # demand: every EP rank pairs the same handles, since calls and release agreements run in the same order.
        for layer in [l for l in self._busy if l.release_state() == _RELEASE_RECLAIMED]:
            self._busy.remove(layer)
            self._free.append(layer)
        if self._free:
            layer = self._free.pop()
        else:
            # A handle beyond the prepared ones (more forwards in flight than before): its one-time setup runs here.
            layer = self.workspace._new_layer()
            layer.prepare()
        self._busy.append(layer)
        return layer

    def _release(self, layer: "tex.MoeLayer") -> None:
        self._busy.remove(layer)
        self._free.append(layer)

    def reclaim(self) -> int:
        """Destroy every handle still held by a saving forward (collective, between steps): for a caller that
        keeps graphs it will never run backward on. Returns how many were destroyed."""
        n = len(self._busy)
        for layer in reversed(self._busy):
            layer.destroy()
            self.workspace._forget_layer(layer)
        self._busy.clear()
        return n

    def prepare(self, num_handles: int = 1) -> None:
        """Create and map ``num_handles`` layer handles now (collective), e.g. before a CUDA graph capture
        or to allocate the in-flight handles of a pipeline schedule up front."""
        while len(self._free) + len(self._busy) < num_handles:
            layer = self.workspace._new_layer()
            layer.prepare()
            self._free.insert(0, layer)

    def status(self) -> int:
        """OR of the status bits of every handle's calls since the last query (host-blocking): 8 a call ran
        in rounds, 16 / 32 a forward kept Xp only / nothing (its backward recomputes)."""
        s = 0
        for layer in self._free + self._busy:
            s |= layer.status()
        return s

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Apply the MoE layer; see the class documentation."""
        ws = self.workspace
        shape = hidden_states.shape
        H, T = self.hidden_size, ws.num_tokens
        if shape[-1] != H or hidden_states.numel() != T * H:
            raise ValueError(f"hidden_states must hold exactly {T} tokens of {H}, got {tuple(shape)}")
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise ValueError("The fused MoE takes contiguous bf16 hidden_states.")
        if shared_gate is not None:
            if not ws.shared_expert:
                raise ValueError("shared_gate needs a workspace with the shared expert")
            if (
                shared_gate.dtype != torch.float32
                or shared_gate.numel() != T
                or not shared_gate.is_contiguous()
                or shared_gate.data_ptr() % 16
            ):
                raise ValueError("shared_gate must be a contiguous, 16-byte aligned fp32 tensor of one value per token")
        with self.prepare_forward_ctx(
            hidden_states,
            num_gemms=2,
            allow_non_contiguous=True,
            allow_different_data_and_param_types=True,
        ) as inp:
            if self.fp8 and not ws.mxfp8:
                raise NotImplementedError("Under an FP8 recipe the fused MoE needs a workspace with mxfp8=True.")
            x = inp.view(T, H)
            sg = None if shared_gate is None else shared_gate.view(T)
            s1 = getattr(self, "shared_fc1_weight", None)
            s2 = getattr(self, "shared_fc2_weight", None)
            grads = [x, self.fc1_weight, self.fc2_weight, self.router_weight, self.router_bias, s1, s2, sg]
            save = (
                torch.is_grad_enabled()
                and ws.training
                and any(t is not None and t.requires_grad for t in grads)
            )
            out, aux, tpe = _MoEFunction.apply(
                x,
                self.fc1_weight,
                self.fc2_weight,
                self.router_weight,
                self.router_bias,
                self.expert_bias,
                s1,
                s2,
                sg,
                self,
                save,
            )
            return out.view(shape), aux, tpe


def fc1_from_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """``[..., 2I, H]`` with rows ``[gate; up]`` (Megatron / TE layout) to the fused MoE's fc1 layout:
    256-row blocks of 128 gate rows then the matching 128 up rows."""
    *lead, n, h = weight.shape
    i = n // 2
    g, u = weight[..., :i, :], weight[..., i:, :]
    blocks = torch.stack([g.reshape(*lead, i // 128, 128, h), u.reshape(*lead, i // 128, 128, h)], dim=-3)
    return blocks.reshape(*lead, n, h)


def fc1_to_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`fc1_from_gate_up`."""
    *lead, n, h = weight.shape
    i = n // 2
    blocks = weight.reshape(*lead, i // 128, 2, 128, h)
    g = blocks[..., 0, :, :].reshape(*lead, i, h)
    u = blocks[..., 1, :, :].reshape(*lead, i, h)
    return torch.cat([g, u], dim=-2)
