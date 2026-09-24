#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared utilities for microbenchmarks: model configs, timing, throughput, CSV output."""

import functools
import importlib.util
import itertools
import math
import mmap
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.utils.benchmark as benchmark

# ---------------------------------------------------------------------------
# Sequence / batch-token sizes
# ---------------------------------------------------------------------------
M_SIZE_LIST = [1024, 2048, 4096, 8192]

# Shared dtype sweep for TE activation benchmarks. Extend this list to add
# additional precisions such as torch.float16.
DTYPE_LIST = [torch.bfloat16]

DEFAULT_MIN_RUN_TIME_SECONDS = 0.2

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
# (name, hidden, intermediate, num_q_heads, num_kv_heads, head_dim, tp)
#
# Sources:
# - Llama 3.1 8B   https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
# - Llama 3.1 70B  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
# - Llama 3.1 405B https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
# - Qwen 2.5 7B  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
# - Qwen 2.5 72B https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

MODEL_CONFIGS = [
    ("Llama3.1-8B/TP1",   4096,  14336,  32,  8, 128,  1),
    ("Llama3.1-8B/TP8",   4096,  14336,  32,  8, 128,  8),
    ("Llama3.1-70B/TP8",  8192,  28672,  64,  8, 128,  8),
    ("Llama3.1-405B/TP8", 16384, 53248, 128,  8, 128,  8),
    ("Qwen2.5-7B/TP1",  3584, 18944,  28,  4, 128,  1),
    ("Qwen2.5-72B/TP8", 8192, 29568,  64,  8, 128,  8),
]

# Unique (model_name, hidden_size) pairs for element-wise benchmarks
MODEL_HIDDEN_SIZES = [
    ("Llama3.1-8B",   4096),
    ("Llama3.1-70B",  8192),
    ("Llama3.1-405B", 16384),
    ("Qwen2.5-7B",  3584),
    ("Qwen2.5-72B", 8192),
]


def gemm_shapes(configs=None):
    """Generate {case_name: (N, K)} dict from MODEL_CONFIGS.

    Each model contributes up to four GEMM shapes:
      QKV, AttnOut, GateUp (SwiGLU), Down.
    """
    shapes = {}
    for (name, hidden, intermediate, n_q, n_kv, hd, tp) in (configs or MODEL_CONFIGS):
        shapes[f"{name}-QKV"]     = ((n_q * hd + 2 * n_kv * hd) // tp, hidden)
        shapes[f"{name}-AttnOut"] = (hidden, (n_q * hd) // tp)
        shapes[f"{name}-GateUp"]  = ((2 * intermediate) // tp, hidden)
        shapes[f"{name}-Down"]    = (hidden, intermediate // tp)
    return shapes


def generate_gemm_test_cases(configs=None, m_sizes=None, dtypes=None):
    """Generate dense GEMM benchmark cases shared by BF16 and FP8 GEMM."""
    test_cases = []
    active_shapes = gemm_shapes(configs)
    for m_value in (m_sizes or M_SIZE_LIST):
        for case_name, (n_value, k_value) in active_shapes.items():
            for dtype in (dtypes or DTYPE_LIST):
                test_cases.append({
                    "Case": case_name,
                    "M": m_value,
                    "N": n_value,
                    "K": k_value,
                    "dtype": dtype,
                })
    return test_cases


# ---------------------------------------------------------------------------
# Low-precision recipe sweep (shared by the dense and grouped GEMM benchmarks)
# ---------------------------------------------------------------------------
# Transformer Engine imports are deferred into the helpers so importing
# utils.py stays free of a GPU / built TE (keeps the non-GEMM benchmarks and
# offline tooling importable).


def _check_mxfp4_support_with_aiter():
    """MXFP4 gate: device support plus the aiter FP4 GEMM backend.

    The MXFP4 GEMM path calls into aiter's a4w4 kernels, so a missing aiter
    package would crash at benchmark time even on supported hardware.
    """
    from transformer_engine.pytorch.quantization import check_mxfp4_support

    supported, reason = check_mxfp4_support()
    if not supported:
        return supported, reason
    if importlib.util.find_spec("aiter") is None:
        return False, "aiter is not installed (required for the MXFP4 GEMM backend)."
    return True, ""


def _precision_specs():
    """Ordered sweep of (name, recipe factory | None, support check | None).

    A ``None`` factory is the bf16 baseline (no autocast). The fp8 entry uses
    HYBRID delayed scaling and is shared by the dense and grouped GEMM
    benchmarks so their fp8 numbers stay comparable.
    """
    from transformer_engine.common.recipe import (
        DelayedScaling,
        Format,
        MXFP4BlockScaling,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )
    from transformer_engine.pytorch.quantization import (
        check_fp8_support,
        check_mxfp8_support,
        check_nvfp4_support,
    )

    return (
        ("bf16", None, None),
        (
            "fp8",
            lambda: DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
            ),
            check_fp8_support,
        ),
        ("mxfp8", MXFP8BlockScaling, check_mxfp8_support),
        ("mxfp4", MXFP4BlockScaling, _check_mxfp4_support_with_aiter),
        ("nvfp4", NVFP4BlockScaling, check_nvfp4_support),
    )


def build_recipes(names=None):
    """Build an ordered ``{name: recipe_or_None}`` sweep of supported precisions.

    ``bf16`` maps to ``None`` (no autocast). Each low-precision entry is
    included only when its support check passes on the current device;
    unsupported ones are dropped with a short notice. Pass *names* to restrict
    and order the sweep, e.g. ``("bf16", "fp8", "mxfp8", "nvfp4")`` for grouped
    GEMM, which has no MXFP4 grouped kernel.
    """
    specs = _precision_specs()
    if names is not None:
        by_name = {spec[0]: spec for spec in specs}
        specs = tuple(by_name[name] for name in names)
    recipes = {}
    for name, factory, support_check in specs:
        if support_check is not None:
            supported, reason = support_check()
            if not supported:
                print(f"Skipping {name} precision: {reason}")
                continue
        recipes[name] = factory() if factory is not None else None
    return recipes


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_func(fn, method="adaptive", min_run_time=DEFAULT_MIN_RUN_TIME_SECONDS):
    """Time *fn* and return ``(mean_ms, measurement)``.

    The ``Measurement`` object carries per-sample times accessible via
    ``measurement.times`` (total wall time per run) and
    ``measurement.number_per_run``.

    method: "adaptive" uses adaptive_autorange (good for compute-bound),
            "blocked"  uses blocked_autorange  (good for memory-bound).
    """
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    if method == "blocked":
        m = timer.blocked_autorange(min_run_time=min_run_time)
    else:
        m = timer.adaptive_autorange(min_run_time=min_run_time)
    return m.mean * 1e3, m


# Dual wall + GPU-kernel timing. Off by default; enabled per run by
# --kernel-profile (set via configure_kernel_profile from conftest).
_KERNEL_PROFILE = False


def configure_kernel_profile(enabled):
    """Enable/disable dual wall+kernel timing (set from --kernel-profile)."""
    global _KERNEL_PROFILE
    _KERNEL_PROFILE = bool(enabled)


def _kernel_time_profiler_ms(fn, warmup=100, iters=100):
    """Mean GPU kernel (device) time per call, in ms, via torch.profiler.

    Sums the self device time of every kernel launched per call, so it excludes
    host launch overhead and host-side timing noise. Accurate for single-kernel
    ops; unreliable for concurrent multi-stream ops (use the "event" method there).
    """
    from torch.profiler import profile, ProfilerActivity
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    device_us = sum(e.self_device_time_total for e in events)
    return (device_us / iters) / 1e3


def _kernel_time_event_ms(fn, warmup=100, iters=100):
    """Mean elapsed GPU device time per call, in ms, via a CUDA-event makespan.

    Brackets a saturated ``iters`` loop with events on the current stream, so
    concurrent multi-stream kernels are measured by their overlapped span rather
    than a per-kernel sum. Handles the multi-stream grouped-GEMM path (where the
    profiler under-counts); converges to the wall time for device-bound ops.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# Kernel-time method: "profiler" (default) or "event". A benchmark opts into the
# event makespan for concurrent multi-stream ops (grouped GEMM); setting
# NVTE_MICROBENCH_KERNEL_TIME overrides the method globally for A/B testing.
_KERNEL_TIME_METHOD_ENV = "NVTE_MICROBENCH_KERNEL_TIME"


def _resolve_kernel_method(method):
    env = os.environ.get(_KERNEL_TIME_METHOD_ENV, "").strip().lower()
    if env in ("profiler", "event"):
        return env
    return method or "profiler"


def _kernel_time_ms(fn, warmup=100, iters=100, method=None):
    """Mean GPU device time per call (ms) using the resolved kernel-time method."""
    if _resolve_kernel_method(method) == "event":
        return _kernel_time_event_ms(fn, warmup, iters)
    return _kernel_time_profiler_ms(fn, warmup, iters)


def time_func_dual(fn, method="adaptive", min_run_time=DEFAULT_MIN_RUN_TIME_SECONDS,
                   kernel_method=None):
    """Time *fn* and return ``(wall_ms, measurement, kernel_ms)``.

    ``wall_ms`` / ``measurement`` are the host wall-clock timing from
    :func:`time_func`. ``kernel_ms`` is the mean GPU kernel (device) time per
    call from :func:`_kernel_time_ms` when kernel profiling is enabled
    (``--kernel-profile``); otherwise it is ``None`` and no profiler pass runs.
    *kernel_method* selects the measurement ("profiler" default, or "event" for
    concurrent multi-stream ops such as grouped GEMM).
    """
    wall_ms, measurement = time_func(fn, method=method, min_run_time=min_run_time)
    kernel_ms = _kernel_time_ms(fn, method=kernel_method) if _KERNEL_PROFILE else None
    return wall_ms, measurement, kernel_ms


# ---------------------------------------------------------------------------
# Rotating input buffers (on by default; disable via --no-rotating)
# ---------------------------------------------------------------------------
# Benchmark inputs are cycled through a ring of buffers so that back-to-back
# kernel launches read different input memory and don't benefit from artificial
# cache residency.  Configured from the CLI options via conftest.
_ROTATE_BUFFERS = True
_ROTATE_MB = 0  # rotation memory budget in MB; 0 => auto-size to exceed the LLC
# Ceiling on the rotation ring size. hipBLASLt-bench caps its rotating block
# count at the iteration count (max(cold_iters, iters)) so it never allocates a
# buffer it won't revisit. torch.utils.benchmark picks the iteration count
# adaptively, so there is no fixed value to cap against; we instead bound the
# ring at a fixed maximum (mirroring hipBLASLt's default cold_iters of 1000).
# With the auto budget this ceiling is never reached; it only guards a very
# large explicit --rotating budget on a small buffer, which would otherwise
# allocate a copy per few MB up to the whole budget.
_ROTATE_MAX_BUFFERS = 1000


def _last_level_cache_bytes():
    """Bytes of the last-level cache that buffer rotation must exceed.

    HIP reports ``L2_cache_size`` as the small per-XCD L2 (e.g. 4 MB on gfx950),
    but the real last-level cache is the much larger AMD Infinity Cache.

    Actual last-level/Infinity Cache sizes:
      - gfx942 / gfx950: 256 MB
      - gfx1250:         192 MB

    We use 256 MB for all devices: a slightly oversized ring is harmless (it
    only allocates a little more memory) and avoids per-arch probing.
    """
    return 256 * 1024 * 1024


def _rotation_count(bytes_per_buffer, cache_mult=2.0, min_buffers=2):
    """Number of buffers so the rotation ring spans the requested memory budget.

    With an explicit ``--rotating MB`` the budget is that many megabytes; when
    omitted it is *cache_mult* x the last-level cache (the ~256 MB AMD Infinity
    Cache), so a buffer is evicted before it is reused.  The ring is floored at
    *min_buffers* (so enabling rotation always rotates) and capped at
    ``_ROTATE_MAX_BUFFERS`` (the adaptive-timer analog of hipBLASLt-bench capping
    its block count at the iteration count, so a huge budget on a small buffer
    can't allocate an unbounded ring).
    """
    if bytes_per_buffer <= 0:
        return min_buffers
    if _ROTATE_MB and _ROTATE_MB > 0:
        budget = _ROTATE_MB * 1024 * 1024
    else:
        cache = _last_level_cache_bytes()
        if not cache:
            return min_buffers
        budget = cache_mult * cache
    count = math.ceil(budget / bytes_per_buffer)
    if _ROTATE_MAX_BUFFERS and _ROTATE_MAX_BUFFERS > 0:
        count = min(count, _ROTATE_MAX_BUFFERS)
    return max(min_buffers, count)


def _tensor_nbytes(t):
    """Byte size of a torch tensor, or 0 if it can't be determined."""
    numel = getattr(t, "numel", None)
    element_size = getattr(t, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    return 0


def rotating(build, *, bytes_per_buffer=None):
    """Return a zero-arg callable yielding an input buffer to time.

    Rotation is on by default: it builds a ring of ``build()`` buffers (spanning
    the ``--rotating MB`` budget, or ~2x the last-level cache when the size is
    omitted) and returns the next one on each call.  With ``--no-rotating`` it
    returns a single cached buffer from ``build()`` on every call, matching the
    original single-buffer behavior.

    ``build`` is a zero-arg callable returning one fresh buffer.
    ``bytes_per_buffer`` overrides the auto-sizing hint for buffers whose byte
    size can't be inferred (e.g. FP8 tensors).
    """
    first = build()
    if not _ROTATE_BUFFERS:
        return lambda: first
    nbytes = bytes_per_buffer if bytes_per_buffer is not None else _tensor_nbytes(first)
    count = _rotation_count(nbytes)
    buffers = [first] + [build() for _ in range(max(0, count - 1))]
    ring = itertools.cycle(buffers)
    return lambda: next(ring)


def make_input(shape, dtype, *, device="cuda", requires_grad=False):
    """Rotation-aware input: a zero-arg callable returning a ``randn`` tensor.

    Honors ``--rotating`` (see :func:`rotating`); on by default, so it returns
    the next tensor in the ring each call (``--no-rotating`` for a single one).
    """
    return rotating(
        lambda: torch.randn(
            *shape, dtype=dtype, device=device, requires_grad=requires_grad
        )
    )


# ---------------------------------------------------------------------------
# Throughput helpers
# ---------------------------------------------------------------------------

def compute_tflops(flops, ms):
    """TFLOPS from operation count and milliseconds."""
    return flops / (ms * 1e-3) / 1e12


def compute_gbps(nbytes, ms):
    """GB/s from byte count and milliseconds."""
    return nbytes / (ms * 1e-3) / 1e9


def make_metric_record(label, ms, unit, throughput, derived=False,
                       ms_precision=3, throughput_precision=2,
                       measurement=None, samples_only=False,
                       kernel_ms=None, kernel_throughput=None):
    """Create a structured metric record for stdout and CSV generation.

    Each record describes one benchmark line item such as "GEMM Forward".
    The harness formats these records for stdout and expands them into
    ``<label> Wall Time (ms)`` / ``<label> Wall <unit>`` columns, plus
    ``<label> Kernel Time (ms)`` / ``<label> Kernel <unit>`` when kernel timing
    is enabled (``kernel_ms`` is not None).

    If *measurement* is provided (a ``torch.utils.benchmark.Measurement``),
    the per-sample times are available for the ``--csv-samples`` output.
    Records with *samples_only=True* are excluded from stdout and the main
    CSV but their samples are still written to the samples CSV.
    """
    return {
        "label": label,
        "ms": ms,
        "unit": unit,
        "throughput": throughput,
        "derived": derived,
        "ms_precision": ms_precision,
        "throughput_precision": throughput_precision,
        "measurement": measurement,
        "samples_only": samples_only,
        "kernel_ms": kernel_ms,
        "kernel_throughput": kernel_throughput,
    }


def make_forward_backward_metric_records(label_prefix, unit,
                                         forward_ms, forward_throughput,
                                         backward_ms, backward_throughput,
                                         backward_derived=False,
                                         ms_precision=3,
                                         throughput_precision=2,
                                         fwd_measurement=None,
                                         bwd_measurement=None,
                                         fwd_bwd_measurement=None,
                                         forward_kernel_ms=None,
                                         forward_kernel_throughput=None,
                                         backward_kernel_ms=None,
                                         backward_kernel_throughput=None):
    """Create standard forward/backward metric records for a benchmark.

    When *backward_derived* is True and *fwd_bwd_measurement* is provided,
    an extra samples-only record for "Forward+Backward" is emitted so that
    the raw timing samples are preserved in the ``--csv-samples`` output.
    """
    records = [
        make_metric_record(
            f"{label_prefix} Forward",
            forward_ms,
            unit,
            forward_throughput,
            ms_precision=ms_precision,
            throughput_precision=throughput_precision,
            measurement=fwd_measurement,
            kernel_ms=forward_kernel_ms,
            kernel_throughput=forward_kernel_throughput,
        ),
        make_metric_record(
            f"{label_prefix} Backward",
            backward_ms,
            unit,
            backward_throughput,
            derived=backward_derived,
            ms_precision=ms_precision,
            throughput_precision=throughput_precision,
            measurement=bwd_measurement,
            kernel_ms=backward_kernel_ms,
            kernel_throughput=backward_kernel_throughput,
        ),
    ]
    if fwd_bwd_measurement is not None:
        records.append(make_metric_record(
            f"{label_prefix} Forward+Backward",
            forward_ms + backward_ms,
            unit,
            0,
            samples_only=True,
            measurement=fwd_bwd_measurement,
        ))
    return records


def direction_records(direction, label, unit, throughput,
                      fwd_func, fwd_bwd_func, fwd_work, bwd_work, kernel_method=None):
    """Metric records for a forward-only or a derived-backward timing.

    *direction* is ``"fwd"`` or ``"bwd"``. *throughput* is ``compute_tflops`` or
    ``compute_gbps`` and *fwd_work* / *bwd_work* the matching flops / bytes.
    Backward is ``(fwd+bwd) - fwd``; its per-sample distribution is each fwd+bwd
    sample shifted by the fwd mean (fwd and fwd+bwd are timed separately, so the
    spread is inherited from fwd+bwd). *kernel_method* is forwarded to
    :func:`time_func_dual` ("event" for concurrent multi-stream ops).
    """
    if direction == "fwd":
        fwd_ms, fwd_measurement, fwd_kernel_ms = time_func_dual(fwd_func, kernel_method=kernel_method)
        return [make_metric_record(
            label, fwd_ms, unit, throughput(fwd_work, fwd_ms), measurement=fwd_measurement,
            kernel_ms=fwd_kernel_ms,
            kernel_throughput=throughput(fwd_work, fwd_kernel_ms) if fwd_kernel_ms else None,
        )]
    fwd_bwd_func()  # warm the backward graph
    fwd_ms, fwd_measurement, fwd_kernel_ms = time_func_dual(fwd_func, kernel_method=kernel_method)
    fwd_bwd_ms, fwd_bwd_measurement, fwd_bwd_kernel_ms = time_func_dual(
        fwd_bwd_func, kernel_method=kernel_method)
    bwd_ms = fwd_bwd_ms - fwd_ms
    bwd_kernel_ms = (fwd_bwd_kernel_ms - fwd_kernel_ms
                     if fwd_kernel_ms is not None and fwd_bwd_kernel_ms is not None else None)
    fwd_mean_s = fwd_measurement.mean
    bwd_measurement = SimpleNamespace(
        times=[t - fwd_mean_s for t in fwd_bwd_measurement.times]
    )
    return [make_metric_record(
        label, bwd_ms, unit, throughput(bwd_work, bwd_ms),
        derived=True, measurement=bwd_measurement,
        kernel_ms=bwd_kernel_ms,
        kernel_throughput=throughput(bwd_work, bwd_kernel_ms) if bwd_kernel_ms else None,
    )]


def _metric_time_key(metric):
    return f"{metric['label']} Wall Time (ms)"


def _metric_throughput_key(metric):
    return f"{metric['label']} Wall {metric['unit']}"


def _metric_kernel_time_key(metric):
    return f"{metric['label']} Kernel Time (ms)"


def _metric_kernel_throughput_key(metric):
    return f"{metric['label']} Kernel {metric['unit']}"


def _format_metric_number(value, precision):
    return f"{value:.{precision}f}"


def _metric_row_from_records(metric_records):
    row = {}
    for metric in metric_records:
        if metric.get("samples_only"):
            continue
        row[_metric_time_key(metric)] = _format_metric_number(
            metric["ms"], metric.get("ms_precision", 3)
        )
        row[_metric_throughput_key(metric)] = _format_metric_number(
            metric["throughput"], metric.get("throughput_precision", 2)
        )
        # Kernel columns appear only under --kernel-profile (kernel_ms present).
        if metric.get("kernel_ms") is not None:
            row[_metric_kernel_time_key(metric)] = _format_metric_number(
                metric["kernel_ms"], metric.get("ms_precision", 3)
            )
            kt = metric.get("kernel_throughput")
            row[_metric_kernel_throughput_key(metric)] = (
                _format_metric_number(kt, metric.get("throughput_precision", 2))
                if kt is not None else ""
            )
    return row


def _print_metric_records(metric_records):
    printable = [m for m in metric_records if not m.get("samples_only")]
    if not printable:
        return
    label_width = max(24, *(len(metric["label"]) for metric in printable))
    for metric in printable:
        ms_str = _format_metric_number(metric["ms"], metric.get("ms_precision", 3))
        throughput_str = _format_metric_number(
            metric["throughput"], metric.get("throughput_precision", 2)
        )
        derived_suffix = " (derived)" if metric.get("derived", False) else ""
        line = (
            f"  {metric['label']:<{label_width}} {ms_str} ms | "
            f"{throughput_str} {metric['unit']}{derived_suffix}"
        )
        kernel_ms = metric.get("kernel_ms")
        if kernel_ms is not None:
            k_ms = _format_metric_number(kernel_ms, metric.get("ms_precision", 3))
            kt = metric.get("kernel_throughput")
            k_thr = (_format_metric_number(kt, metric.get("throughput_precision", 2))
                     if kt is not None else "-")
            line += f"  ||  kernel {k_ms} ms | {k_thr} {metric['unit']}"
        print(line)


# ---------------------------------------------------------------------------
# pytest-based execution support
# ---------------------------------------------------------------------------
# conftest.py drives the microbenchmarks under pytest via the framework-agnostic
# helpers below (no pytest import here, so importing utils.py never requires it).
# Results are collected per family (test module) and written with the CSV / samples
# schema the dashboard ingest consumes.

def configure_rotating(rotating, no_rotating):
    """Set module-level input-rotation state from parsed options."""
    global _ROTATE_BUFFERS, _ROTATE_MB
    if rotating is not None and rotating < 0:
        raise ValueError("--rotating expects a non-negative size in MB")
    _ROTATE_BUFFERS = not no_rotating
    _ROTATE_MB = rotating or 0


def apply_backend_env(monkeypatch, env):
    """Force a kernel backend for one test by setting/unsetting env vars.

    A ``None`` value unsets the var, so forcing one backend cleanly clears the
    toggles that would select a competing one; pytest restores them afterwards.
    """
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


@functools.lru_cache(maxsize=1)
def _te_install_root():
    import transformer_engine as _te
    return Path(_te.__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def _te_py_source():
    chunks = []
    for p in _te_install_root().rglob("*.py"):
        try:
            chunks.append(p.read_text(errors="ignore"))
        except OSError:
            pass
    return "\n".join(chunks)


@functools.lru_cache(maxsize=None)
def te_honors_env(varname):
    """True if the installed TE build reads *varname* (python dispatch or compiled .so).

    Lets a benchmark skip a forced backend on builds that predate its dispatch,
    instead of silently measuring the default (which shows up as a phantom
    regression in a long-running history sweep).
    """
    try:
        root = _te_install_root()
    except Exception:
        return False
    if varname in _te_py_source():
        return True
    needle = varname.encode()
    for so in root.rglob("*.so"):
        try:
            with open(so, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if mm.find(needle) != -1:
                    return True
        except (OSError, ValueError):
            pass
    return False


class _FamilyResults:
    """Accumulated rows / samples for one benchmark family."""

    def __init__(self):
        self.param_columns = None
        self.metric_columns = None
        self.rows = []
        self.case_metrics = []


def _stringify_params(case_params):
    return {k: (str(v) if isinstance(v, torch.dtype) else v) for k, v in case_params.items()}


def record_bench(store, family, case_params, metric_records, node_name=""):
    """Record one benchmark case into *store* (a dict keyed by *family*)."""
    fam = store.setdefault(family, _FamilyResults())
    metric_row = _metric_row_from_records(metric_records)
    metric_columns = list(metric_row.keys())
    if fam.param_columns is None:
        fam.param_columns = list(case_params.keys())
        fam.metric_columns = metric_columns
    elif metric_columns != fam.metric_columns:
        raise ValueError(
            f"Inconsistent metric columns for {family}: "
            f"expected {fam.metric_columns}, got {metric_columns}"
        )
    row = _stringify_params(case_params)
    row.update(metric_row)
    fam.rows.append(row)
    fam.case_metrics.append((_stringify_params(case_params), metric_records, node_name))


def print_case(case_params, metric_records):
    """Print a case header and its metric lines (reused stdout format)."""
    label = "  ".join(f"{k}={v}" for k, v in case_params.items())
    print(f"\n{'='*60}\nTesting: {label}\n{'='*60}")
    _print_metric_records(metric_records)


def write_bench_outputs(store, *, csv=None, csv_samples=None):
    """Write per-family CSV / samples outputs; return paths written."""
    import pandas as pd
    from pathlib import Path

    # When an explicit filename is given but several families run in one session
    # (e.g. `pytest .`), insert the family name so they don't overwrite each other.
    multi = sum(1 for fam in store.values() if fam.rows) > 1

    def _dest(explicit, family, default_name):
        if not isinstance(explicit, str):
            return default_name
        if not multi:
            return explicit
        p = Path(explicit)
        return str(p.with_name(f"{p.stem}-{family}{p.suffix}"))

    written = []
    for family, fam in store.items():
        if not fam.rows:
            continue
        if csv is not None:
            out = _dest(csv, family, f"{family}.csv")
            pd.DataFrame(fam.rows, columns=fam.param_columns + fam.metric_columns).to_csv(
                out, index=False
            )
            written.append(out)
        if csv_samples is not None:
            sout = _dest(csv_samples, family, f"{family}_samples.csv")
            sample_rows = _sample_rows(fam)
            if sample_rows:
                pd.DataFrame(
                    sample_rows,
                    columns=fam.param_columns + ["label", "sample_idx", "time_ms"],
                ).to_csv(sout, index=False)
                written.append(sout)
    return written


def _sample_rows(fam):
    """Flatten a family's per-iteration timing measurements into long rows."""
    rows = []
    for case_params, records, _node in fam.case_metrics:
        for metric in records:
            m = metric.get("measurement")
            if m is None:
                continue
            for i, t in enumerate(m.times):
                sr = dict(case_params)
                sr["label"] = metric["label"]
                sr["sample_idx"] = i
                sr["time_ms"] = t * 1e3
                rows.append(sr)
    return rows


# ---------------------------------------------------------------------------
# Dashboard run: one pytest session -> one run dir of born-tagged CSVs + samples/
# + run_info.txt, ready for the TE-dashboard ingest. Replaces the shell runner.
# ---------------------------------------------------------------------------

def _detect_gpu_model():
    """Short GPU token (e.g. ``MI355X``) from torch's device name, else UNKNOWN."""
    import re
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return "UNKNOWN"
    m = re.search(r"MI\s?\d{3,4}[A-Za-z]*", name)
    return m.group(0).replace(" ", "").upper() if m else "UNKNOWN"


def _gpu_pci():
    """PCI BDF of the active GPU (torch device 0), e.g. ``0000:75:00.0``; '' if unknown."""
    try:
        p = torch.cuda.get_device_properties(0)
        return "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)
    except Exception:
        return ""


def _dashboard_run_meta():
    """Run metadata: run timestamp, git sha/date, GPU model, visible GPU id, host."""
    import datetime
    import subprocess

    here = Path(__file__).resolve().parent

    def git(*args):
        # safe.directory=* bypasses git's dubious-ownership guard when the repo
        # is bind-mounted into a container under a different uid.
        try:
            return subprocess.check_output(
                ["git", "-c", "safe.directory=*", "-C", str(here), *args],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    sha = git("rev-parse", "HEAD")
    gpu = (os.environ.get("HIP_VISIBLE_DEVICES", "") or "0").split(",")[0].strip() or "0"
    now = datetime.datetime.now().astimezone()
    return {
        "run_ts": now.isoformat(timespec="seconds"),
        "stamp": now.strftime("%Y-%m-%dT%H-%M-%S"),
        "sha": sha,
        "short": sha[:12] or "unknown",
        "cdate": git("show", "-s", "--format=%cI", "HEAD"),
        "model": _detect_gpu_model(),
        "gpu": gpu,
        "gpu_bdf": _gpu_pci(),
        "host": os.uname().nodename.split(".")[0],
    }


def _write_run_info(path, meta):
    """Write a small version/machine manifest alongside a run's CSVs."""
    import datetime
    import importlib

    def ver(mod):
        try:
            return getattr(importlib.import_module(mod), "__version__", "?")
        except Exception:
            return "?"

    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return ""

    rocm = read("/opt/rocm/.info/version") or getattr(torch.version, "hip", "") or "?"
    info = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": meta["host"],
        "gpu_id": meta["gpu"],
        "gpu_bdf": meta.get("gpu_bdf") or "?",
        "arch": meta["model"],
        "te_commit": meta["sha"],
        "te": ver("transformer_engine"),
        "pytorch": ver("torch"),
        "triton": ver("triton"),
        "jax": ver("jax"),
        "rocm": rocm,
        "amdgpu_drv": read("/sys/module/amdgpu/version") or "?",
        "kernel": os.uname().release,
    }
    path.write_text("".join(f"{k + ':':12}{v}\n" for k, v in info.items()))


def _dashboard_run_dir(out_base, meta):
    """Run dir path from run metadata: ``<out_base>/<node>/<datetime>_<commit>/``."""
    node = f"{meta['model']}_{meta['host']}_gpu{meta['gpu']}"
    if meta["gpu_bdf"]:
        node += "_" + meta["gpu_bdf"].replace(":", "-").replace(".", "-")
    return Path(out_base) / node / f"{meta['stamp']}_{meta['short']}"


def dashboard_run_plan(out_base="results"):
    """Resolve (meta, run_dir) for a dashboard run without writing anything."""
    meta = _dashboard_run_meta()
    return SimpleNamespace(meta=meta, run_dir=_dashboard_run_dir(out_base, meta))


def write_dashboard_run(store, *, out_base="results", csv_samples=None, meta=None):
    """Write one dashboard run and return the run dir.

    Layout: ``<out_base>/<arch>_<host>_gpu<id>_<pci>/<datetime>_<commit>/``. Per-family
    CSVs are born-tagged with run_week/commit_sha/commit_date (so dashboard_ingest.py
    consumes them directly); per-iteration samples land untagged under ``samples/``;
    ``run_info.txt`` records versions + machine. Pass *meta* from
    :func:`dashboard_run_plan` to reuse the metadata resolved at session start.
    """
    import pandas as pd

    if meta is None:
        meta = _dashboard_run_meta()
    run_dir = _dashboard_run_dir(out_base, meta)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    _write_run_info(run_dir / "run_info.txt", meta)

    tag = {"run_ts": meta["run_ts"], "commit_sha": meta["sha"], "commit_date": meta["cdate"]}
    tag_cols = list(tag)
    for family, fam in store.items():
        if not fam.rows:
            continue
        rows = [{**tag, **row} for row in fam.rows]
        pd.DataFrame(rows, columns=tag_cols + fam.param_columns + fam.metric_columns).to_csv(
            run_dir / f"{family}.csv", index=False
        )
        if csv_samples is not None:
            sample_rows = _sample_rows(fam)
            if sample_rows:
                pd.DataFrame(
                    sample_rows, columns=fam.param_columns + ["label", "sample_idx", "time_ms"]
                ).to_csv(run_dir / "samples" / f"{family}_samples.csv", index=False)
    return run_dir


def _times_ms(measurement):
    if measurement is None:
        return []
    return [float(t) * 1e3 for t in getattr(measurement, "times", [])]


def _result_rows(store):
    """Flatten *store* into (suite, name, wall_ms, wall_thr, kernel_ms, kernel_thr, unit) rows."""
    import numpy as np

    rows = []
    for family, fam in store.items():
        suite = family[len("benchmark_"):] if family.startswith("benchmark_") else family
        for _case_params, records, node_name in fam.case_metrics:
            base = node_name
            if base.endswith("]") and "[" in base:
                base = base[base.index("[") + 1 : -1]  # keep the parametrize id
            visible = [m for m in records if not m.get("samples_only")]
            for m in visible:
                name = base if len(visible) == 1 else f"{base} {m['label']}"
                times = _times_ms(m.get("measurement"))
                wall_ms = float(np.median(np.asarray(times))) if times else m["ms"]
                rows.append((
                    suite, name, wall_ms, m["throughput"],
                    m.get("kernel_ms"), m.get("kernel_throughput"), m["unit"],
                ))
    return rows


def format_results_table(store):
    """Render the results as a Markdown table (times in ms) with a caption line."""
    rows = _result_rows(store)
    if not rows:
        return ""
    has_kernel = any(r[4] is not None for r in rows)

    def ms_cell(v):
        return "-" if v is None else f"{v:.4f}"

    def thr_cell(v, unit):
        return "-" if v is None else f"{v:.2f} {unit}"

    headers = ["Benchmark", "Config", "Wall Median (ms)", "Wall Throughput"]
    if has_kernel:
        headers += ["Kernel Median (ms)", "Kernel Throughput"]
    body = []
    for suite, name, wall_ms, wall_thr, kernel_ms, kernel_thr, unit in sorted(
        rows, key=lambda r: (r[0], r[1])
    ):
        cells = [suite, name, ms_cell(wall_ms), thr_cell(wall_thr, unit)]
        if has_kernel:
            cells += [ms_cell(kernel_ms), thr_cell(kernel_thr, unit)]
        body.append(cells)
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))]

    def row(cells):
        # Left-align the text columns (Benchmark, Config); right-align the numerics.
        padded = [
            c.ljust(widths[i]) if i <= 1 else c.rjust(widths[i]) for i, c in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    align = [
        ":" + "-" * (widths[i] - 1) if i <= 1 else "-" * (widths[i] - 1) + ":"
        for i in range(len(headers))
    ]
    caption = f"benchmark: {len(body)} tests"
    return "\n".join(
        [caption, "", row(headers), "| " + " | ".join(align) + " |", *(row(r) for r in body)]
    )


_THROUGHPUT_UNITS = frozenset({"TFLOPS", "GB/s"})
_AGG_COLLAPSE_ALWAYS = frozenset({"Case", "dtype"})


def _aggregate_rows(store):
    """Work-weighted harmonic-mean throughput per (suite, group), collapsing shape/size axes."""
    import numpy as np

    agg = {}
    order = []
    for family, fam in store.items():
        suite = family[len("benchmark_"):] if family.startswith("benchmark_") else family
        params = fam.param_columns or []
        for case_params, records, _node in fam.case_metrics:
            group_cols = [
                c for c in params
                if c not in _AGG_COLLAPSE_ALWAYS and isinstance(case_params.get(c), str)
            ]
            group = tuple((c, case_params[c]) for c in group_cols)
            for m in records:
                if m.get("samples_only") or m["unit"] not in _THROUGHPUT_UNITS:
                    continue
                thr = m["throughput"]
                times = _times_ms(m.get("measurement"))
                wall_ms = float(np.median(np.asarray(times))) if times else m["ms"]
                if thr is None or not (thr > 0) or not (wall_ms > 0):
                    continue
                key = (suite, m["label"], group, m["unit"])
                a = agg.get(key)
                if a is None:
                    a = agg[key] = {"wall_work": 0.0, "wall_time": 0.0,
                                    "kern_work": 0.0, "kern_time": 0.0, "n": 0}
                    order.append(key)
                a["wall_work"] += thr * wall_ms
                a["wall_time"] += wall_ms
                a["n"] += 1
                k_ms, k_thr = m.get("kernel_ms"), m.get("kernel_throughput")
                if k_ms and k_thr and k_ms > 0:
                    a["kern_work"] += k_thr * k_ms
                    a["kern_time"] += k_ms
    return agg, order


def format_aggregate_table(store):
    """Render per-group aggregate throughput (work-weighted harmonic mean) below the table."""
    agg, order = _aggregate_rows(store)
    if not agg:
        return ""
    # Include the metric label in the group text only when a suite emits more than one.
    labels_per_suite = {}
    for suite, label, _group, _unit in order:
        labels_per_suite.setdefault(suite, set()).add(label)

    has_kernel = any(a["kern_time"] > 0 for a in agg.values())

    def thr_cell(work, time, unit):
        return f"{work / time:.2f} {unit}" if time > 0 else "-"

    headers = ["Benchmark", "Group", "n", "Wall Throughput"]
    if has_kernel:
        headers += ["Kernel Throughput"]
    body = []
    for key in order:
        suite, label, group, unit = key
        a = agg[key]
        parts = [f"{c}={v}" for c, v in group]
        if len(labels_per_suite[suite]) > 1:
            parts.insert(0, label)
        group_text = "  ".join(parts) or "all"
        cells = [suite, group_text, str(a["n"]), thr_cell(a["wall_work"], a["wall_time"], unit)]
        if has_kernel:
            cells.append(thr_cell(a["kern_work"], a["kern_time"], unit))
        body.append(cells)
    body.sort(key=lambda r: (r[0], r[1]))
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))]

    def row(cells):
        # Left-align the text columns (Benchmark, Group); right-align n / throughputs.
        padded = [
            c.ljust(widths[i]) if i <= 1 else c.rjust(widths[i]) for i, c in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    align = [
        ":" + "-" * (widths[i] - 1) if i <= 1 else "-" * (widths[i] - 1) + ":"
        for i in range(len(headers))
    ]
    caption = (f"aggregate throughput (work-weighted harmonic mean over shapes): "
               f"{len(body)} groups")
    return "\n".join(
        [caption, "", row(headers), "| " + " | ".join(align) + " |", *(row(r) for r in body)]
    )
