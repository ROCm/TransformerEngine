# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""``kosmos_moe``: the KOSMOS fused expert-parallel MoE layer as one differentiable PyTorch op."""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from . import _ext
from ._instance import EP_SIZE, KosmosInstance, SavedState, resolve_allocator

__all__ = ["kosmos_moe", "status", "memory_report", "destroy", "is_available"]

_TOPK_SUPPORTED = (1, 2, 4, 6, 8, 10)
_MAX_LOCAL_EXPERTS = 64
_ROW = 256
_SAVE = ("all", "xp", "none")
# acc argument of KOSMOS's training backward by main_grad dtype: 1 adds into fp32, 2 into bf16
_ACC_OF_DTYPE = {torch.float32: 1, torch.bfloat16: 2}

# Status bits, OR-ed over the calls since the last query (src/moe/moe.h). Errors: 1 an expert id
# outside [0, 8 E_loc) (-1 included: such a route is dropped), 2 a token routed twice to one expert,
# 64 a call that could not run. Notes: 8 a call ran in rounds (over capacity), 32 a forward kept
# nothing, less than its save tier allowed (it ran in rounds, or its rows did not fit the saved
# rows); the backward then recomputes what is missing. A training call keeps its tier or nothing.
STATUS_BAD_ID = 1
STATUS_DUPLICATE = 2
STATUS_ROUNDS = 8
STATUS_KEPT_NOTHING = 32
STATUS_ERROR = 64


def is_available() -> bool:
    """True if the KOSMOS binding can be imported and the current device is gfx950."""
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", "")
    return arch.startswith("gfx950") and _ext.module_or_none() is not None


def _default_capacity_rows(tokens: int, topk: int, local_experts: int) -> int:
    # 1.5x the balanced routed rows plus a row block of padding per expert, rounded to 256: a
    # busier routing still runs, in rounds (status 8)
    rows = (3 * tokens * topk) // 2 + _ROW * local_experts
    return -(-rows // _ROW) * _ROW


class _KosmosInstanceCache:
    """One KOSMOS instance per configuration key, shared by every MoE layer of that shape.

    The key is (EP group ranks, device, H, I, E_loc, k, T, capacity rows, activation, wgcap,
    symmetric-memory allocator). Creating an instance is collective: every EP rank reaches the first
    call of a configuration at the same point."""

    def __init__(self):
        self._instances: Dict[tuple, KosmosInstance] = {}

    def get(
        self,
        group: dist.ProcessGroup,
        *,
        tokens: int,
        hidden: int,
        inter: int,
        local_experts: int,
        topk: int,
        capacity_rows: Optional[int],
        wgcap: int,
        allocator: Optional[str] = None,
    ) -> KosmosInstance:
        """The instance for this configuration, created (collectively) on first use. ``allocator``
        None: from ``NVTE_KOSMOS_MOE_SYMMETRIC_MEMORY`` (see ``_instance.resolve_allocator``)."""
        mod = _ext.load()
        act = int(mod.ACT_SWIGLU)
        if capacity_rows is None:
            capacity_rows = _default_capacity_rows(tokens, topk, local_experts)
        else:
            capacity_rows = -(-int(capacity_rows) // _ROW) * _ROW
        if allocator is None:
            allocator = resolve_allocator()
        key = (
            tuple(dist.get_process_group_ranks(group)),
            torch.cuda.current_device(),
            hidden,
            inter,
            local_experts,
            topk,
            tokens,
            capacity_rows,
            act,
            int(wgcap),
            allocator,
        )
        inst = self._instances.get(key)
        if inst is None:
            inst = KosmosInstance(
                mod,
                group,
                hidden=hidden,
                inter=inter,
                local_experts=local_experts,
                topk=topk,
                tokens=tokens,
                capacity_rows=capacity_rows,
                wgcap=int(wgcap),
                act=act,
                allocator=allocator,
            )
            self._instances[key] = inst
        return inst

    def matching(self, group: Optional[dist.ProcessGroup]) -> List[Tuple[tuple, KosmosInstance]]:
        """(key, instance) of every instance on ``group``'s ranks (all if None)."""
        if group is None:
            return list(self._instances.items())
        ranks = tuple(dist.get_process_group_ranks(group))
        return [(k, v) for k, v in self._instances.items() if k[0] == ranks]

    def destroy(self, group: Optional[dist.ProcessGroup]) -> None:
        """Collective: destroys and forgets the instances of ``group`` (all if None)."""
        for key, inst in self.matching(group):
            inst.destroy()
            del self._instances[key]

    def destroy_instance(self, inst: KosmosInstance) -> None:
        """Collective: destroys and forgets one instance."""
        for key, val in list(self._instances.items()):
            if val is inst:
                val.destroy()
                del self._instances[key]


_INSTANCES = _KosmosInstanceCache()


def _check(group, x, topk_idx, topk_weights, w1, w2) -> None:
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError(f"group must be a torch.distributed ProcessGroup, got {type(group)}")
    if group.size() != EP_SIZE:
        raise ValueError(
            f"KOSMOS MoE needs an EP group of {EP_SIZE} ranks in one node, got {group.size()}"
        )
    named = (
        ("x", x),
        ("topk_idx", topk_idx),
        ("topk_weights", topk_weights),
        ("w1", w1),
        ("w2", w2),
    )
    device = torch.device("cuda", torch.cuda.current_device())
    for name, t in named:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if t.device != device:
            raise ValueError(
                f"{name} is on {t.device}; every tensor must be on the current device {device}"
            )
    for name, t in (("x", x), ("w1", w1), ("w2", w2)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {t.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}")
    if x.dim() != 2 or w1.dim() != 3 or w2.dim() != 3 or topk_idx.dim() != 2:
        raise ValueError(
            "expected x [T, H], topk_idx and topk_weights [T, k], w1 [E_loc, 2I, H], w2 [E_loc, H,"
            f" I]; got {tuple(x.shape)}, {tuple(topk_idx.shape)}, {tuple(w1.shape)},"
            f" {tuple(w2.shape)}"
        )
    tokens, hidden = x.shape
    local_experts, two_inter, _ = w1.shape
    inter = two_inter // 2
    if (
        tuple(w1.shape) != (local_experts, 2 * inter, hidden)
        or tuple(w2.shape) != (local_experts, hidden, inter)
        or topk_idx.shape[0] != tokens
        or tuple(topk_weights.shape) != tuple(topk_idx.shape)
    ):
        raise ValueError(
            "inconsistent shapes: expected x [T, H], topk_idx and topk_weights [T, k], "
            f"w1 [E_loc, 2I, H], w2 [E_loc, H, I]; got {tuple(x.shape)}, {tuple(topk_idx.shape)}, "
            f"{tuple(topk_weights.shape)}, {tuple(w1.shape)}, {tuple(w2.shape)}"
        )
    topk = topk_idx.shape[1]
    if topk not in _TOPK_SUPPORTED:
        raise ValueError(f"KOSMOS MoE supports k in {_TOPK_SUPPORTED}, got {topk}")
    if not 1 <= local_experts <= _MAX_LOCAL_EXPERTS:
        raise ValueError(
            f"KOSMOS MoE supports 1 to {_MAX_LOCAL_EXPERTS} experts per rank, got {local_experts}"
        )
    if hidden % _ROW or inter % _ROW:
        raise ValueError(f"H and I must be multiples of {_ROW}, got H={hidden} I={inter}")
    # A weight is never copied: a per-call copy costs a weight's traffic, and its gradient must
    # reach the parameter.
    for name, t in (("w1", w1), ("w2", w2)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _int32(topk_idx: torch.Tensor) -> torch.Tensor:
    # a temporary is safe to free after the call: the caching allocator reuses its block only for
    # work on the same stream, which runs after the KOSMOS kernel that reads it
    return topk_idx.to(torch.int32).contiguous()


def _dtopk_weights(dprobs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    """KOSMOS's dprobs is dense [T, 8 E_loc] (zero at experts a token did not use); the op's input
    is [T, k]. Routes KOSMOS dropped (ids outside [0, 8 E_loc), -1 included) get zero."""
    num_experts = dprobs.shape[1]
    idx = topk_idx.long()
    dropped = (idx < 0) | (idx >= num_experts)
    return dprobs.gather(1, idx.clamp(0, num_experts - 1)).masked_fill_(dropped, 0.0)


def _after_main_grad(param: torch.Tensor):
    """Megatron's protocol once ``dW`` is in ``param.main_grad``, as TE's own modules follow it
    (module/linear.py): with ``grad_added_to_main_grad``, set it and hand autograd a dummy."""
    if not hasattr(param, "grad_added_to_main_grad"):
        return None
    from ..module.base import get_dummy_wgrad  # pylint: disable=import-outside-toplevel

    param.grad_added_to_main_grad = True
    return get_dummy_wgrad(
        list(param.shape),
        param.dtype,
        zero=getattr(param, "zero_out_wgrad", False),
        zero_tensor=getattr(param, "zero_tensor_dummy_wgrad", False),
    )


def _weight_grad(param: torch.Tensor, dw: torch.Tensor, fuse_wgrad_accumulation: bool):
    """A bf16 ``dW`` from the kernel: returned as the gradient, or, with
    ``fuse_wgrad_accumulation`` and a ``main_grad``, added into it."""
    if not (fuse_wgrad_accumulation and hasattr(param, "main_grad")):
        return dw
    param.main_grad.add_(dw)
    return _after_main_grad(param)


def _in_kernel_acc(w1: torch.Tensor, w2: torch.Tensor, needs: tuple, fuse: bool) -> int:
    """KOSMOS's ``acc`` when the kernel can add both weight gradients into ``main_grad`` itself:
    both weights need a gradient and have contiguous ``main_grad`` of their shape, both fp32 (1) or
    both bf16 (2). 0 otherwise: the kernel writes bf16 ``dW`` and the op adds it afterwards."""
    if not (fuse and needs[2] and needs[3]):
        return 0
    grads = [getattr(w, "main_grad", None) for w in (w1, w2)]
    if any(g is None for g in grads) or grads[0].dtype != grads[1].dtype:
        return 0
    for w, g in zip((w1, w2), grads):
        if tuple(g.shape) != tuple(w.shape) or not g.is_contiguous() or g.device != w.device:
            return 0
    return _ACC_OF_DTYPE.get(grads[0].dtype, 0)


class _KosmosMoeFn(torch.autograd.Function):
    """One KOSMOS forward launch and one backward launch; the saved state is autograd's."""

    @staticmethod
    def forward(ctx, x, topk_weights, w1, w2, topk_idx32, inst, save, saved_rows, fuse_acc):
        # pylint: disable=missing-function-docstring
        y = torch.empty_like(x)
        saved = inst.empty_saved(save, saved_rows)
        inst.forward(x, w1, w2, topk_idx32, topk_weights, y, saved)
        ctx.inst = inst
        ctx.tier = saved.tier
        ctx.saved_rows = saved.rows
        ctx.fuse_wgrad_accumulation = fuse_acc
        # The parameter objects themselves, for main_grad (a saved-tensor hook may hand back a
        # copy).
        ctx.w1_param = w1
        ctx.w2_param = w2
        # Everything the backward reads, so autograd owns its lifetime (freed after the backward,
        # or with the graph if no backward runs) and its version counters make an in-place change
        # of x, topk_weights, w1 or w2 before the backward an error.
        ctx.save_for_backward(x, topk_weights, w1, w2, topk_idx32, *saved.tensors())
        return y

    @staticmethod
    def backward(ctx, dy):
        # pylint: disable=missing-function-docstring
        x, topk_weights, w1, w2, topk_idx32, *buffers = ctx.saved_tensors
        inst = ctx.inst
        saved = SavedState.from_tensors(ctx.tier, ctx.saved_rows, buffers, inst.device)
        x = x.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_idx32 = topk_idx32.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        dy = dy.to(torch.bfloat16).contiguous()
        dx = torch.empty_like(dy)
        dprobs = torch.empty(
            (dy.shape[0], EP_SIZE * inst.local_experts), dtype=torch.float32, device=dy.device
        )
        needs = ctx.needs_input_grad
        fuse = ctx.fuse_wgrad_accumulation
        w1p = ctx.w1_param
        w2p = ctx.w2_param
        acc = _in_kernel_acc(w1p, w2p, needs, fuse)
        if acc:
            dw1 = w1p.main_grad
            dw2 = w2p.main_grad
        else:
            dw1 = torch.empty_like(w1)
            dw2 = torch.empty_like(w2)
        inst.backward(saved, dy, x, w1, w2, topk_idx32, topk_weights, dx, dprobs, dw1, dw2, acc)
        if acc:
            g1 = _after_main_grad(w1p)
            g2 = _after_main_grad(w2p)
        else:
            g1 = _weight_grad(w1p, dw1, fuse) if needs[2] else None
            g2 = _weight_grad(w2p, dw2, fuse) if needs[3] else None
        return (
            dx if needs[0] else None,
            _dtopk_weights(dprobs, topk_idx32) if needs[1] else None,
            g1,
            g2,
            None,
            None,
            None,
            None,
            None,
        )


def kosmos_moe(
    group: dist.ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    save: str = "all",
    saved_rows: Optional[int] = None,
    fuse_wgrad_accumulation: bool = False,
    capacity_rows: Optional[int] = None,
    wgcap: int = 256,
) -> torch.Tensor:
    """An expert-parallel MoE layer (routed experts, SwiGLU, bf16) as one KOSMOS kernel per pass.

    ``y[t] = sum_j topk_weights[t, j] * W2[e] @ swiglu(W1[e] @ x[t])`` with ``e = topk_idx[t, j]``,
    the experts spread over the 8 ranks of ``group``, ``E_loc`` per rank (global expert ``e`` lives
    on rank ``e // E_loc``). Router weights are applied in fp32 when the results are folded.

    Parameters
    ----------
    group : ProcessGroup
        The EP group: exactly 8 ranks, one GPU each, in one node (gfx950).
    x : torch.Tensor
        ``[T, H]`` bfloat16, this rank's tokens.
    topk_idx : torch.Tensor
        ``[T, k]`` int32 or int64 global expert ids, ``k`` in {1, 2, 4, 6, 8, 10}; -1 drops a route
        (padding tokens, a variable token count padded to ``T``). No expert twice per token.
    topk_weights : torch.Tensor
        ``[T, k]`` float32 router weights.
    w1 : torch.Tensor
        ``[E_loc, 2I, H]`` bfloat16, rows in KOSMOS's GLU-interleave-128 order
        (:func:`to_kosmos_w1`).
    w2 : torch.Tensor
        ``[E_loc, H, I]`` bfloat16.
    save : {"all", "xp", "none"}
        The most a training call keeps for its backward, in ordinary tensors freed after the
        backward: ``"all"`` the dispatched input rows (``Xp``), GEMM1's output (``G``) and the
        activation (``A``); ``"xp"`` only ``Xp`` (the backward recomputes GEMM1); ``"none"``
        neither (the backward sends ``x`` to the experts again and recomputes). ``Y`` (one row per
        routed token, for the router-weight gradient) and a small routing record are kept always.
    saved_rows : int, optional
        Rows ``Xp``, ``G`` and ``A`` are sized for, at most the capacity (default: the capacity). A
        call whose received rows do not fit keeps nothing (status bit 32) and its backward
        recomputes; it stays correct.
    fuse_wgrad_accumulation : bool
        Add the weight gradients into ``w.main_grad`` (Megatron's protocol) instead of returning
        them. With fp32 or bf16 ``main_grad`` on both weights the kernel adds into it directly.
    capacity_rows : int, optional
        Receive rows per rank the instance is sized for (default ``1.5 T k + 256 E_loc``, rounded up
        to 256). A routing that needs more runs correctly, in rounds, and slower (status bit 8).
    wgcap : int
        Workgroups per KOSMOS launch, in [9, 256]; fewer leave CUs to other kernels.

    Returns
    -------
    torch.Tensor
        ``[T, H]`` bfloat16; differentiable in ``x``, ``topk_weights``, ``w1`` and ``w2``.

    Every rank of ``group`` must make the same KOSMOS calls (forward and backward, with the same
    ``save`` and ``saved_rows``) in the same order on one stream: the kernels wait on their peers,
    and a rank that skips a call leaves the others waiting forever. The first call of a
    configuration is collective and host-blocking (it creates the instance).
    """
    _check(group, x, topk_idx, topk_weights, w1, w2)
    if save not in _SAVE:
        raise ValueError(f"save must be one of {_SAVE}, got {save!r}")
    tokens, hidden = x.shape
    local_experts, two_inter, _ = w1.shape
    inst = _INSTANCES.get(
        group,
        tokens=tokens,
        hidden=hidden,
        inter=two_inter // 2,
        local_experts=local_experts,
        topk=topk_idx.shape[1],
        capacity_rows=capacity_rows,
        wgcap=wgcap,
    )
    rows = inst.capacity_rows if saved_rows is None else int(saved_rows)
    if not 1 <= rows <= inst.capacity_rows:
        raise ValueError(
            f"saved_rows must be in [1, capacity_rows={inst.capacity_rows}], got {rows}"
        )
    x = x.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_idx32 = _int32(topk_idx)
    training = torch.is_grad_enabled() and any(t.requires_grad for t in (x, topk_weights, w1, w2))
    if training:
        return _KosmosMoeFn.apply(
            x, topk_weights, w1, w2, topk_idx32, inst, save, rows, fuse_wgrad_accumulation
        )
    # No backward will follow: nothing is kept.
    y = torch.empty_like(x)
    inst.forward(x, w1, w2, topk_idx32, topk_weights, y, None)
    return y


def status(
    group: Optional[dist.ProcessGroup] = None,
    raise_on: Iterable[int] = (STATUS_BAD_ID, STATUS_DUPLICATE, STATUS_ERROR),
) -> int:
    """Host-blocking: OR of KOSMOS's status bits over every call since the last query, for the
    instances of ``group`` (all if None); clears them. Raises RuntimeError if a bit in ``raise_on``
    is set. A caller that pads with -1 ids passes ``raise_on=(2, 64)``: KOSMOS reports a dropped
    route with bit 1, the same bit as an out-of-range id."""
    st = 0
    for _, inst in _INSTANCES.matching(group):
        st |= inst.status()
    bad = st & sum(set(raise_on))
    if bad:
        raise RuntimeError(f"KOSMOS MoE status {st} (error bits {bad}; see kosmos.functional)")
    return st


def memory_report(group: Optional[dist.ProcessGroup] = None) -> List[dict]:
    """One dict per instance of ``group`` (all if None): its configuration; the device memory it
    holds outside PyTorch's caching allocator, fixed per instance (``measured_bytes``: its symmetric
    buffer, ``symmetric_bytes``, plus KOSMOS's scratch); and the caching-allocator bytes one training
    call keeps until its backward, per save tier (``saved_bytes_per_call``)."""
    return [inst.memory_report() for _, inst in _INSTANCES.matching(group)]


def destroy(group: Optional[dist.ProcessGroup] = None) -> None:
    """Collective: frees the instances of ``group`` (all if None). A forward whose backward has not
    run yet cannot be differentiated afterwards. Instances are otherwise kept until the process
    exits."""
    _INSTANCES.destroy(group)
