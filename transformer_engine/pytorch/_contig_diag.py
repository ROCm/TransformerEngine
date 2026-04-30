# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Materialize-cost instrumentation for full vs lite TE comparisons.

Counts and times every `.contiguous()` call site that we wrap with
`record()`. Designed to attribute the BSH-strided materialize cost we see
under Megatron + DelayedScaling, so we can diff full vs lite to find sites
that fire in lite but not in full (or fire more often).

Activation:
  NVTE_CONTIG_DIAG=1                  enable instrumentation
  NVTE_CONTIG_DIAG_DUMP_STEP=N        auto-dump after step N tick

`tick_step()` is invoked once per training step from
`FP8GlobalStateManager.autocast_exit` (forward path under DelayedScaling),
so the user does not need to patch their training loop.

Timing is `time.perf_counter_ns` around the materialize call — CPU launch
latency only, no cuda.synchronize. The counter answers "where and how
often"; rocprof answers "how long on device". Synchronizing here would
distort the very gap we are trying to measure.
"""
from __future__ import annotations

import os
import time
import traceback
from collections import Counter
from typing import Dict, Tuple

import torch

ENABLED = os.environ.get("NVTE_CONTIG_DIAG", "0") != "0"
_DUMP_STEP_ENV = os.environ.get("NVTE_CONTIG_DIAG_DUMP_STEP", "")
DUMP_AT_STEP = int(_DUMP_STEP_ENV) if _DUMP_STEP_ENV else None

# Frame filenames to skip when locating the user-code caller — these are
# either TE plumbing or context-manager glue that does not identify the
# producer of the non-contiguous tensor.
_SKIP_FRAME_FRAGMENTS = (
    "transformer_engine/pytorch/module/base.py",
    "transformer_engine/pytorch/module/linear.py",
    "transformer_engine/pytorch/module/layernorm_linear.py",
    "transformer_engine/pytorch/module/layernorm_mlp.py",
    "transformer_engine/pytorch/_lite/",
    "transformer_engine/pytorch/_contig_diag.py",
    "/contextlib.py",
    "torch/_dynamo/",
    "torch/nn/modules/module.py",
)

Signature = Tuple[str, Tuple[int, ...], Tuple[int, ...], str]
_counts: Counter = Counter()
_total_ns: Dict[Signature, int] = {}
_step: int = 0
_dumped: bool = False


def _caller_top() -> str:
    """Top user-code frame, skipping TE/contextlib/dynamo plumbing."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if any(s in fr.filename for s in _SKIP_FRAME_FRAGMENTS):
            continue
        return f"{fr.filename}:{fr.lineno}({fr.name})"
    return "<unknown>"


def record(module_class: str, inp: torch.Tensor, copy_time_ns: int) -> None:
    """Record one materialize event. Cheap when ENABLED is False."""
    if not ENABLED:
        return
    sig: Signature = (
        module_class,
        tuple(inp.shape),
        tuple(inp.stride()),
        _caller_top(),
    )
    _counts[sig] += 1
    _total_ns[sig] = _total_ns.get(sig, 0) + copy_time_ns


def tick_step() -> None:
    """Bump the step counter; auto-dump on the configured step."""
    global _step, _dumped
    if not ENABLED:
        return
    _step += 1
    if DUMP_AT_STEP is not None and _step >= DUMP_AT_STEP and not _dumped:
        dump(reason="auto")
        _dumped = True


def dump(reason: str = "explicit") -> None:
    """Print accumulated counts and CPU launch ns per signature."""
    if not ENABLED:
        return
    print(
        f"[CONTIG-DIAG] dump reason={reason} step={_step} "
        f"unique_sites={len(_counts)} total_calls={sum(_counts.values())}",
        flush=True,
    )
    rows = sorted(_counts.items(), key=lambda kv: -_total_ns.get(kv[0], 0))
    for sig, n in rows:
        total_ns = _total_ns.get(sig, 0)
        mean_us = (total_ns / n) / 1000.0 if n else 0.0
        module_class, shape, stride, caller = sig
        print(
            f"[CONTIG-DIAG] module={module_class} "
            f"shape={shape} stride={stride} "
            f"calls={n} total_ms={total_ns/1e6:.2f} mean_us={mean_us:.1f} "
            f"caller={caller}",
            flush=True,
        )


def time_contiguous(module_class: str, inp: torch.Tensor) -> torch.Tensor:
    """Materialize `inp` and record the event. Used at hook sites."""
    if not ENABLED:
        return inp.contiguous()
    t0 = time.perf_counter_ns()
    out = inp.contiguous()
    t1 = time.perf_counter_ns()
    record(module_class, inp, t1 - t0)
    return out
