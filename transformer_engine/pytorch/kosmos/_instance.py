# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""One KOSMOS training instance per configuration, over KOSMOS's caller-owned training API.

The instance owns KOSMOS's compiled kernels, its settings and the memory of the call in progress,
fixed in size and reused by every call after an in-kernel handshake of all EP ranks: one symmetric
buffer (the arenas peers write into) and KOSMOS's own scratch (e.g. GEMM2's output at the expert
owner and the gradient of GEMM1's output in the backward).
Everything a backward reads from its forward (the saved state) lives in ordinary tensors the caller
allocates for each training call with :meth:`KosmosInstance.empty_saved`, passes to the forward,
and passes again to that forward's backward. So any number of forwards may be in flight, backwards
may come in any order, and a forward whose backward never runs leaves nothing behind in KOSMOS:
its saved tensors are freed with its autograd graph.

The symmetric buffer is a torch symmetric-memory tensor (``symm_mem.empty`` and ``rendezvous`` over
the EP group) handed to KOSMOS through ``kosmos_comm_wrap`` from torch 2.13, or else KOSMOS's own
allocation (see :func:`resolve_allocator`).
"""

import contextlib
import importlib.util
import os
import re
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

__all__ = ["KosmosInstance", "SavedState", "resolve_allocator", "SYMMETRIC_MEMORY_ENV"]

EP_SIZE = 8
SYMMETRIC_MEMORY_ENV = "NVTE_KOSMOS_MOE_SYMMETRIC_MEMORY"
_TORCH_SYMM_MEM_MIN = (2, 12)
# auto picks torch's allocation only from 2.13: 2.10 cannot allocate it on ROCm, and 2.12
# (ROCm 7.14) never frees it, so every instance destroy leaks the buffer.
_TORCH_SYMM_MEM_AUTO_MIN = (2, 13)
_ALIGN = 256  # every saved-buffer pointer (src/moe/moe.h, KosmosMoeSaved)
_SAVED_NAMES = ("xp", "g", "a", "y", "route")


def _log_enabled() -> bool:
    return os.getenv("NVTE_LOG_KOSMOS_MOE_CONFIG", "0") == "1"


def _torch_version() -> Tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def torch_symmetric_memory_supported(minimum: Tuple[int, int] = _TORCH_SYMM_MEM_MIN) -> bool:
    """torch ``minimum`` or newer (default 2.12, the first that can allocate symmetric memory on
    ROCm) with ``torch.distributed._symmetric_memory``."""
    if _torch_version() < minimum:
        return False
    try:
        return importlib.util.find_spec("torch.distributed._symmetric_memory") is not None
    except ImportError:
        return False


def resolve_allocator() -> str:
    """Where the instance's symmetric buffer comes from: ``"torch"`` or ``"kosmos"``.

    ``NVTE_KOSMOS_MOE_SYMMETRIC_MEMORY``: ``auto`` (default: torch from torch 2.13, else KOSMOS's
    own), ``torch`` (required; raises below torch 2.12; on 2.12 every instance destroy leaks the
    buffer), ``kosmos`` (KOSMOS's own allocation)."""
    mode = os.getenv(SYMMETRIC_MEMORY_ENV, "auto").strip().lower()
    if mode not in ("auto", "torch", "kosmos"):
        raise ValueError(f"{SYMMETRIC_MEMORY_ENV}={mode!r}; expected auto, torch or kosmos")
    if mode == "auto":
        return "torch" if torch_symmetric_memory_supported(_TORCH_SYMM_MEM_AUTO_MIN) else "kosmos"
    if mode == "torch" and not torch_symmetric_memory_supported():
        raise RuntimeError(
            f"{SYMMETRIC_MEMORY_ENV}=torch needs torch >= 2.12 with torch.distributed."
            f"_symmetric_memory; this is torch {torch.__version__}"
        )
    return mode


def _host_allgather(group: dist.ProcessGroup):
    """KOSMOS's host all-gather (``KosmosCollective``) over the EP group itself.

    It runs only inside KOSMOS's collective, host-blocking calls (instance creation). The EP group
    is used rather than a new gloo group because ``dist.new_group`` must be entered by every rank of
    the default group, which a lazily created EP subgroup cannot arrange.
    """
    world = group.size()
    if dist.get_backend(group) == "gloo":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", torch.cuda.current_device())

    def allgather(send) -> bytes:
        n = len(send)
        if n == 0:
            return b""
        src = torch.frombuffer(bytearray(send), dtype=torch.uint8).to(device)
        dst = [torch.empty(n, dtype=torch.uint8, device=device) for _ in range(world)]
        dist.all_gather(dst, src, group=group)
        return torch.cat(dst).cpu().numpy().tobytes()

    return allgather


class SavedState:
    """The saved state of one training forward: caller-owned tensors, plus the tier the forward may
    keep at most and the rows ``xp``, ``g`` and ``a`` were sized for. ``xp`` is None below tier
    ``xp``, ``g`` and ``a`` below tier ``all``. The layout inside the buffers is KOSMOS's."""

    __slots__ = ("tier", "rows") + _SAVED_NAMES

    def __init__(self, tier: int, rows: int, xp, g, a, y, route):
        self.tier = tier
        self.rows = rows
        self.xp = xp
        self.g = g
        self.a = a
        self.y = y
        self.route = route

    def tensors(self) -> tuple:
        """``(xp, g, a, y, route)``, for ``save_for_backward``."""
        return tuple(getattr(self, n) for n in _SAVED_NAMES)

    @classmethod
    def from_tensors(cls, tier: int, rows: int, tensors, device) -> "SavedState":
        """Rebuilds the state from what ``saved_tensors`` returned. A saved-tensor hook may have
        moved a buffer and brought it back as a new tensor; KOSMOS needs it contiguous, on the
        device and 256-byte aligned, so anything else is copied into a fresh buffer."""
        fixed = {}
        for name, t in zip(_SAVED_NAMES, tensors):
            if t is not None:
                if t.device != device:
                    raise RuntimeError(
                        f"KOSMOS MoE: saved buffer {name} came back on {t.device}, expected"
                        f" {device} (a saved_tensors_hooks unpack must restore the device)"
                    )
                if not t.is_contiguous() or t.data_ptr() % _ALIGN:
                    t = torch.empty(t.numel(), dtype=t.dtype, device=device).copy_(t.reshape(-1))
            fixed[name] = t
        return cls(tier, rows, **fixed)

    def pointers(self) -> tuple:
        """``(xp, g, a, y, route)`` device pointers, 0 for an absent buffer."""
        ptrs = []
        for name in _SAVED_NAMES:
            t = getattr(self, name)
            p = 0 if t is None else t.data_ptr()
            if p % _ALIGN:
                raise ValueError(f"KOSMOS MoE: saved buffer {name} is not {_ALIGN}-byte aligned")
            ptrs.append(p)
        return tuple(ptrs)

    def nbytes(self) -> int:
        """Bytes of the buffers present."""
        return sum(t.numel() * t.element_size() for t in self.tensors() if t is not None)


class KosmosInstance:
    """One KOSMOS training instance for one configuration key.

    Collective to construct and to destroy: every rank of ``group`` must do it at the same point.
    """

    def __init__(
        self,
        mod,
        group: dist.ProcessGroup,
        *,
        hidden: int,
        inter: int,
        local_experts: int,
        topk: int,
        tokens: int,
        capacity_rows: int,
        wgcap: int,
        act: int,
        allocator: str,
    ):
        if allocator not in ("torch", "kosmos"):
            raise ValueError(f"allocator must be 'torch' or 'kosmos', got {allocator!r}")
        self.mod = mod
        self.group = group
        self.rank = group.rank()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden = hidden
        self.inter = inter
        self.local_experts = local_experts
        self.topk = topk
        self.tokens = tokens
        self.capacity_rows = capacity_rows
        self.wgcap = wgcap
        self.act = act
        self.allocator = allocator
        self.tiers = {"none": mod.SAVE_NONE, "xp": mod.SAVE_XP, "all": mod.SAVE_ALL}
        self.measured_bytes = 0
        self.symmetric_bytes = 0
        self._saved_bytes: Dict[int, Dict[str, int]] = {}
        self._symm = None  # (tensor, rendezvous handle, peer pointer array) for allocator "torch"
        self._inst = None
        self._allgather = _host_allgather(group)

        self._check_same_config()
        # The bytes to allocate for MoeInstance(ext_*): the instance's symmetric bytes plus the peer
        # flags kosmos_comm_wrap carves from the end of a buffer given without a signal pad.
        need = int(
            mod.instance_bytes(
                hidden,
                inter,
                local_experts,
                topk,
                tokens,
                capacity_rows,
                self.rank,
                self._allgather,
                wgcap=wgcap,
                act=act,
            )
        )
        self.instance_bytes = need
        with self._measure():
            ext = {}
            if allocator == "torch":
                ext = self._torch_symmetric_buffer(need)
                self.symmetric_bytes = ext["ext_bytes"]
            self._inst = mod.MoeInstance(
                hidden,
                inter,
                local_experts,
                topk,
                tokens,
                self.rank,
                self._allgather,
                wgcap=wgcap,
                capacity_rows=capacity_rows,
                act=act,
                **ext,
            )
        if allocator == "kosmos":
            self.symmetric_bytes = need
        self._log(f"created: {self.describe()}")

    # ------------------------------------------------------------------ #
    # creation
    # ------------------------------------------------------------------ #
    def _allgather_ints(self, values) -> list:
        mine = torch.tensor(list(values), dtype=torch.int64)
        if dist.get_backend(self.group) != "gloo":
            mine = mine.to(self.device)
        got = [torch.empty_like(mine) for _ in range(self.group.size())]
        dist.all_gather(got, mine, group=self.group)
        return [tuple(int(v) for v in g.cpu().tolist()) for g in got]

    def _check_same_config(self) -> None:
        rows = self._allgather_ints(
            [
                self.hidden,
                self.inter,
                self.local_experts,
                self.topk,
                self.tokens,
                self.capacity_rows,
                self.wgcap,
                self.act,
                int(self.allocator == "torch"),
            ]
        )
        if len(set(rows)) != 1:
            raise ValueError(
                "KOSMOS MoE: the EP ranks called with different configurations (H, I, E_loc, k, T,"
                f" capacity_rows, wgcap, act, torch symmetric memory per rank: {rows}); every rank"
                " must make the same calls with the same shapes and the same"
                f" {SYMMETRIC_MEMORY_ENV}"
            )

    def _torch_symmetric_buffer(self, nbytes: int) -> dict:
        """Collective: one torch symmetric-memory buffer on every rank of the EP group, as the
        ``ext_*`` arguments of ``MoeInstance``."""
        # pylint: disable-next=import-outside-toplevel
        import torch.distributed._symmetric_memory as symm_mem

        buf = None
        err = ""
        try:
            buf = symm_mem.empty(nbytes, dtype=torch.uint8, device=self.device)
        except RuntimeError as e:
            err = str(e).splitlines()[0] if str(e) else type(e).__name__
        # Agree before the rendezvous, which is collective: a rank whose allocation failed must not
        # leave the others waiting in it.
        ok = [r[0] for r in self._allgather_ints([int(buf is not None)])]
        if not all(ok):
            failed = [r for r, v in enumerate(ok) if not v]
            raise RuntimeError(
                f"KOSMOS MoE: torch symm_mem.empty({nbytes}) failed on EP ranks {failed}"
                f"{': ' + err if err else ''}. Set {SYMMETRIC_MEMORY_ENV}=kosmos to let KOSMOS"
                " allocate its symmetric buffer itself."
            )
        hdl = symm_mem.rendezvous(buf, self.group.group_name)
        if hdl.world_size != self.group.size() or hdl.rank != self.rank:
            raise RuntimeError(
                f"KOSMOS MoE: symmetric-memory rendezvous gave rank {hdl.rank} of"
                f" {hdl.world_size}, expected {self.rank} of {self.group.size()}"
            )
        # buffer_ptrs are the bases of each rank's allocation; the tensor may start at an offset
        # into it, the same on every rank. KOSMOS requires peers[rank] == local.
        base = [int(p) for p in hdl.buffer_ptrs]
        offset = buf.data_ptr() - base[self.rank]
        peers = torch.tensor([p + offset for p in base], dtype=torch.int64, device=self.device)
        self._symm = (buf, hdl, peers)
        return {"ext_local": buf.data_ptr(), "ext_peers": peers.data_ptr(), "ext_bytes": nbytes}

    @contextlib.contextmanager
    def _measure(self):
        """Adds the device memory taken outside the caching allocator by the enclosed calls."""
        torch.cuda.synchronize(self.device)
        free0 = torch.cuda.mem_get_info(self.device)[0]
        resv0 = torch.cuda.memory_reserved(self.device)
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device)
            free1 = torch.cuda.mem_get_info(self.device)[0]
            resv1 = torch.cuda.memory_reserved(self.device)
            self.measured_bytes += (free0 - free1) - (resv1 - resv0)

    # ------------------------------------------------------------------ #
    # bookkeeping
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """One line: the configuration and the memory the instance holds."""
        return (
            f"H={self.hidden} I={self.inter} E_loc={self.local_experts} k={self.topk}"
            f" T={self.tokens} capacity_rows={self.capacity_rows} wgcap={self.wgcap} act={self.act}"
            f" module={self.mod.__name__} symmetric_memory={self.allocator}"
            f" instance_bytes={self.instance_bytes} symmetric_bytes={self.symmetric_bytes}"
            f" measured_bytes={self.measured_bytes}"
        )

    def _log(self, msg: str) -> None:
        if _log_enabled() and self.rank == 0:
            print(f"[TE KOSMOS MoE] {msg}", flush=True)

    def _live(self):
        if self._inst is None:
            raise RuntimeError("KOSMOS MoE instance used after destroy()")
        return self._inst

    # ------------------------------------------------------------------ #
    # saved state
    # ------------------------------------------------------------------ #
    def saved_bytes(self, rows: int) -> Dict[str, int]:
        """Bytes of each saved buffer (``xp``, ``g``, ``a``, ``y``, ``route``) for ``rows`` saved
        rows."""
        got = self._saved_bytes.get(rows)
        if got is None:
            raw = self._live().saved_bytes(rows)
            got = {n: int(raw[n]) for n in _SAVED_NAMES}
            self._saved_bytes[rows] = got
        return got

    def tier(self, save: str) -> int:
        """KOSMOS's tier constant for ``save`` (``"all"``, ``"xp"``, ``"none"``)."""
        try:
            return self.tiers[save]
        except KeyError:
            raise ValueError(f"save must be one of {sorted(self.tiers)}, got {save!r}") from None

    def empty_saved(self, save: str, rows: int) -> SavedState:
        """Uninitialized saved buffers for one training call from the caching allocator: ``xp`` for
        tier ``xp`` and up, ``g`` and ``a`` for tier ``all``, ``y`` and ``route`` always."""
        tier = self.tier(save)
        nbytes = self.saved_bytes(rows)
        keep = {
            "xp": tier >= self.tiers["xp"],
            "g": tier >= self.tiers["all"],
            "a": tier >= self.tiers["all"],
            "y": True,
            "route": True,
        }
        bufs = [
            torch.empty(nbytes[n], dtype=torch.uint8, device=self.device) if keep[n] else None
            for n in _SAVED_NAMES
        ]
        return SavedState(tier, rows, *bufs)

    def saved_bytes_per_call(self, rows: int) -> Dict[str, int]:
        """Caching-allocator bytes one training call keeps until its backward, per save tier."""
        nb = self.saved_bytes(rows)
        return {
            "all": sum(nb.values()),
            "xp": nb["xp"] + nb["y"] + nb["route"],
            "none": nb["y"] + nb["route"],
        }

    # ------------------------------------------------------------------ #
    # calls (each enqueues one KOSMOS kernel on the current stream)
    # ------------------------------------------------------------------ #
    def forward(self, x, w1, w2, topk_idx32, topk_w, out, saved: Optional[SavedState]) -> None:
        """A training forward (``saved`` given) or an inference forward (``saved`` None: nothing
        is kept)."""
        inst = self._live()
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if saved is None:
            tier, rows, ptrs = -1, 0, (0, 0, 0, 0, 0)
        else:
            tier, rows, ptrs = saved.tier, saved.rows, saved.pointers()
        inst.forward(
            x.data_ptr(),
            w1.data_ptr(),
            w2.data_ptr(),
            topk_idx32.data_ptr(),
            topk_w.data_ptr(),
            out.data_ptr(),
            tier,
            rows,
            *ptrs,
            stream,
        )

    def backward(
        self, saved: SavedState, dout, x, w1, w2, topk_idx32, topk_w, dx, dprobs, dw1, dw2, acc
    ) -> None:
        """The backward of the forward that filled ``saved``; ``acc`` 0 writes bf16 ``dw1`` /
        ``dw2``, 1 adds into fp32 buffers, 2 adds into bf16 buffers."""
        inst = self._live()
        stream = torch.cuda.current_stream(self.device).cuda_stream
        inst.backward(
            dout.data_ptr(),
            x.data_ptr(),
            w1.data_ptr(),
            w2.data_ptr(),
            topk_idx32.data_ptr(),
            topk_w.data_ptr(),
            saved.tier,
            saved.rows,
            *saved.pointers(),
            dx.data_ptr(),
            dprobs.data_ptr(),
            dw1.data_ptr(),
            dw2.data_ptr(),
            acc,
            stream,
        )

    # ------------------------------------------------------------------ #
    # queries and teardown
    # ------------------------------------------------------------------ #
    def status(self) -> int:
        """Host-blocking: OR of the status bits of every call since the last query; clears them."""
        return int(self._live().status())

    def memory_report(self) -> dict:
        """The configuration, the instance's memory and the saved bytes per call (see
        :func:`transformer_engine.pytorch.kosmos.memory_report`)."""
        info = self._live().info()
        return {
            "hidden": self.hidden,
            "inter": self.inter,
            "local_experts": self.local_experts,
            "topk": self.topk,
            "tokens": self.tokens,
            "capacity_rows": self.capacity_rows,
            "wgcap": self.wgcap,
            "act": self.act,
            "symmetric_memory": self.allocator,
            # the symmetric bytes KOSMOS asks for; the symmetric buffer given to it (the torch
            # symmetric-memory tensor, with room for KOSMOS's peer flags) or asked of KOSMOS's own
            # allocator; and the change in free device memory, net of the caching allocator, around
            # the instance's creation: everything the instance holds outside the caching allocator,
            # its scratch included
            "instance_bytes": self.instance_bytes,
            "symmetric_bytes": self.symmetric_bytes,
            "measured_bytes": self.measured_bytes,
            # caching-allocator bytes one training call keeps until its backward, per save tier, at
            # the default saved rows (the capacity)
            "saved_bytes_per_call": self.saved_bytes_per_call(self.capacity_rows),
            "kosmos_info": {str(k): v for k, v in dict(info).items()},
        }

    def destroy(self) -> None:
        """Collective: frees the instance once no peer can still touch its symmetric buffer."""
        if self._inst is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._inst.destroy()
        self._inst = None
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._symm = None
        self._log("destroyed")
