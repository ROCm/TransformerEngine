#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""GPU interference detection (AMD/ROCm): warn when another process shares the
benchmark GPU(s), so timings aren't silently skewed by a neighbor.

This module is deliberately free of a top-level ``torch`` import. On ROCm the
first CUDA call in a process (e.g. importing ``torch.utils.benchmark`` or
``torch.cuda.get_device_properties``) opens every visible device and registers
this process on all of them under its *host* PID -- which, inside a container,
can't be matched to our container PID. So the neighbor snapshot must be taken
*before* torch touches the GPU; conftest calls :func:`snapshot_gpu_neighbors`
at import time, ahead of ``from utils import ...`` (which pulls torch). With no
CUDA context yet we are absent from the process list, so every entry is a
genuine neighbor and there is no self to exclude across the PID namespace.
:func:`detect_gpu_interference` then maps torch's ``cuda:0`` to its amdsmi
handle by PCI address and reports the pre-existing processes on that GPU.
"""

import os


def _proc_cmdline(pid):
    """Full argv for *pid* from /proc, or None if unreadable (e.g. another pid namespace)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\x00") if p]
    if parts:
        return " ".join(parts)
    try:  # kernel thread / hidden argv -> fall back to comm
        with open(f"/proc/{pid}/comm") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _own_pids():
    """Our PID plus all descendant PIDs (so worker processes aren't flagged)."""
    me = os.getpid()
    children = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return {me}
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                data = fh.read()
            # comm (field 2) may contain spaces/parens; ppid is field 4, after the last ')'.
            ppid = int(data[data.rfind(")") + 2:].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    own, stack = set(), [me]
    while stack:
        pid = stack.pop()
        if pid in own:
            continue
        own.add(pid)
        stack.extend(children.get(pid, []))
    return own


# Monitoring tools open every GPU but aren't real compute interference.
_IGNORED_PROC_NAMES = ("nvtop", "amdsmi", "rocmsmi")


def _is_ignored_proc(name):
    base = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
    return bool(base) and any(base.startswith(n) for n in _IGNORED_PROC_NAMES)


def _amdsmi_attr(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _amdsmi_gpu_processes(amdsmi, handles):
    """{handle_index: {pid: name}} of compute processes across *handles*."""
    out = {}
    for idx, handle in enumerate(handles):
        procs = {}
        try:
            plist = amdsmi.amdsmi_get_gpu_process_list(handle)
        except Exception:
            plist = []
        for p in plist:
            pid = _amdsmi_attr(p, "pid")
            name = _amdsmi_attr(p, "name") or _amdsmi_attr(p, "process_name")
            if pid is None:  # older amdsmi returns opaque process handles -> fetch details
                try:
                    info = amdsmi.amdsmi_get_gpu_process_info(handle, p)
                    pid = _amdsmi_attr(info, "pid")
                    name = name or _amdsmi_attr(info, "name")
                except Exception:
                    pass
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            procs[pid] = str(name) if name else ""
        out[idx] = procs
    return out


def _amdsmi_bdf(amdsmi, handle):
    """amdsmi BDF for *handle* normalized to ``'domain:bus:device'`` (function dropped), or None."""
    try:
        bdf = str(amdsmi.amdsmi_get_gpu_device_bdf(handle)).strip().lower()
    except Exception:
        return None
    return bdf.split(".")[0] or None


def snapshot_gpu_neighbors():
    """Snapshot GPU processes + PCI addresses via amdsmi, before any CUDA init.

    Must be called before this process makes its first CUDA call, so the process
    list is free of our own (soon-to-be-registered) host PID. Returns a dict with
    ``status`` (``"ok"`` / ``"unavailable"``), ``neighbors`` (``{idx: {pid:
    name}}``) and ``bdfs`` (``{idx: "domain:bus:device"}``). amdsmi is shut down
    before returning, so nothing GPU-related is held for the rest of the session.
    """
    try:
        import amdsmi
    except Exception:
        return {"status": "unavailable", "neighbors": {}, "bdfs": {}}
    try:
        amdsmi.amdsmi_init()
    except Exception:
        return {"status": "unavailable", "neighbors": {}, "bdfs": {}}
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        neighbors = _amdsmi_gpu_processes(amdsmi, handles)
        bdfs = {idx: _amdsmi_bdf(amdsmi, h) for idx, h in enumerate(handles)}
        return {"status": "ok", "neighbors": neighbors, "bdfs": bdfs}
    except Exception:
        return {"status": "unavailable", "neighbors": {}, "bdfs": {}}
    finally:
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass


def _torch_cuda_bdf(index=0):
    """PCI address of torch's ``cuda:index`` as ``'domain:bus:device'`` (lowercase hex), or None.

    Reading device properties lazily initializes CUDA, so call this only *after*
    the neighbor snapshot. It respects HIP_VISIBLE_DEVICES, so cuda:0 is the GPU
    the benchmark actually uses.
    """
    try:
        import torch
        p = torch.cuda.get_device_properties(index)
        return f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}"
    except Exception:
        return None


def detect_gpu_interference(snapshot):
    """AMD/ROCm only: find foreign compute processes on *our* GPU.

    Uses the pre-CUDA *snapshot* from :func:`snapshot_gpu_neighbors` (genuine
    neighbors only, no self), then maps torch's ``cuda:0`` to its amdsmi handle
    by matching PCI addresses -- deterministic and immune to both the
    amdsmi<->HIP index mismatch and VRAM races -- and reports the pre-existing
    processes on that GPU, dropping monitoring tools (nvtop/amd-smi/rocm-smi). If
    we can't locate our GPU we report nothing rather than risk a false positive.
    Returns ``(status, foreign)`` with *status* in ``"ok"`` / ``"unavailable"`` /
    ``"not_amd"`` and *foreign* a list of ``(gpu, pid, cmdline)``.
    """
    import torch

    if not getattr(torch.version, "hip", None):
        return "not_amd", []
    if not snapshot or snapshot.get("status") != "ok":
        return "unavailable", []

    neighbors = snapshot.get("neighbors", {})
    bdfs = snapshot.get("bdfs", {})
    target = _torch_cuda_bdf(0)
    our_idx = None
    if target is not None:
        for idx, bdf in bdfs.items():
            if bdf == target:
                our_idx = idx
                break

    if our_idx is None:  # couldn't locate our GPU -> stay silent, never false-positive
        return "ok", []
    own = _own_pids()
    foreign = []
    for pid, name in sorted(neighbors.get(our_idx, {}).items()):
        if pid in own or _is_ignored_proc(name):
            continue
        foreign.append((our_idx, pid, _proc_cmdline(pid) or name or f"pid {pid}"))
    return "ok", foreign


def format_gpu_interference(foreign):
    """Render the foreign-process list as a warning banner."""
    bar = "=" * 74
    lines = ["", bar,
             f"GPU INTERFERENCE: {len(foreign)} other process(es) on the benchmark GPU(s)",
             "-" * 74]
    for gpu, pid, cmd in foreign:
        if len(cmd) > 150:
            cmd = cmd[:147] + "..."
        lines.append(f"  GPU {gpu}  PID {pid:<8}  {cmd}")
    lines.append(bar)
    return "\n".join(lines)
