#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""pytest glue for the microbenchmarks.

Thin shim: the option/CSV/timing logic lives in utils.py; this file only wires
those helpers into pytest hooks and exposes the ``microbench`` fixture.  Each
benchmark is run canonically with ``python benchmark_gemm.py`` (the file
forwards to pytest), e.g.::

    python benchmark_gemm.py --csv                 # write benchmark_gemm.csv
    python benchmark_gemm.py -k "bf16 and QKV"     # select by parametrize id
    python benchmark_gemm.py -k triton             # select the triton backend
"""

import os

# MXFP8 is gated off by default on ROCm (NVTE_ROCM_ENABLE_MXFP8=0). Enable it here,
# before any benchmark module imports Transformer Engine, because check_mxfp8_support()
# caches its result on the first call (at build_recipes() import time).
os.environ.setdefault("NVTE_ROCM_ENABLE_MXFP8", "1")

# Snapshot GPU neighbors BEFORE importing torch: on ROCm the first CUDA call (which
# `from utils import` can trigger via `import torch.utils.benchmark`) registers this
# process on every visible GPU, which would otherwise look like a neighbor. Taken
# here, the snapshot is free of our own PID.
import gpu_neighbors

_GPU_SNAPSHOT = gpu_neighbors.snapshot_gpu_neighbors()

from pathlib import Path

import pytest

from utils import (
    configure_kernel_profile,
    configure_rotating,
    dashboard_run_plan,
    format_aggregate_table,
    format_results_table,
    print_case,
    record_bench,
    write_bench_outputs,
    write_dashboard_run,
)


def pytest_addoption(parser):
    group = parser.getgroup("microbench", "TE GPU microbenchmarks")
    group.addoption(
        "--csv", nargs="?", const=True, default=None, metavar="FILE",
        help="Write results to CSV (one per family; default name from the module).",
    )
    group.addoption(
        "--csv-samples", nargs="?", const=True, default=None, metavar="FILE",
        help="Write per-sample timing data to a CSV.",
    )
    group.addoption(
        "--kernel-profile", action="store_true", default=False,
        help="Also measure GPU kernel (device) time alongside wall time, adding "
             "Kernel Time / Kernel <unit> columns to the CSV.",
    )
    group.addoption(
        "--rotating", nargs="?", type=int, const=0, default=None, metavar="MB",
        help="Rotate inputs through a ring of buffers (optional MB budget). On by default.",
    )
    group.addoption(
        "--no-rotating", action="store_true", default=False,
        help="Disable input buffer rotation.",
    )
    group.addoption(
        "--abort-on-gpu-interference", action="store_true", default=False,
        help="Abort the run if another process is using the benchmark GPU(s).",
    )
    group.addoption(
        "--no-gpu-interference-check", action="store_true", default=False,
        help="Disable the shared-GPU interference check.",
    )
    group.addoption(
        "--run-flydsl", action="store_true", default=False,
        help="Run FlyDSL-backed cases (marked @pytest.mark.flydsl); skipped by default due to long compile times.",
    )
    group.addoption(
        "--dashboard-run", action="store_true", default=False,
        help="Collect all families into one run dir (born-tagged CSVs + samples/ + "
             "run_info.txt) under --dashboard-out, for the TE dashboard.",
    )
    group.addoption(
        "--dashboard-out", default="results", metavar="DIR",
        help="Base output dir for --dashboard-run; a "
             "<arch>_<host>_gpu<id>_<pci>/<date>_<commit>/ run dir is created under it "
             "(default: results).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: TE GPU microbenchmark")
    config.addinivalue_line("markers", "flydsl: FlyDSL-backed case; opt-in via --run-flydsl")
    configure_rotating(config.getoption("--rotating"), config.getoption("--no-rotating"))
    configure_kernel_profile(config.getoption("--kernel-profile"))
    config._microbench_store = {}
    if not config.getoption("--no-gpu-interference-check"):
        status, foreign = gpu_neighbors.detect_gpu_interference(_GPU_SNAPSHOT)
        if foreign:
            print(gpu_neighbors.format_gpu_interference(foreign))
            if config.getoption("--abort-on-gpu-interference"):
                pytest.exit(
                    "GPU interference detected; aborting (--abort-on-gpu-interference).",
                    returncode=3,
                )
        elif status == "unavailable":
            print("WARNING: GPU interference check skipped -- amdsmi package not available "
                  "(pip install amdsmi, or pass --no-gpu-interference-check to silence).")

    # Resolve the dashboard run dir up front so pytest_report_header can show it
    # and write_dashboard_run can reuse the same metadata at session end.
    if config.getoption("--dashboard-run"):
        config._dashboard_plan = dashboard_run_plan(config.getoption("--dashboard-out"))


def pytest_report_header(config):
    plan = getattr(config, "_dashboard_plan", None)
    if plan is None:
        return None
    extras = [name for name, opt in (
        ("kernel-profile", "--kernel-profile"),
        ("flydsl", "--run-flydsl"),
        ("samples", "--csv-samples"),
    ) if config.getoption(opt)]
    m = plan.meta
    return [
        f"dashboard-run -> {plan.run_dir}",
        f"  {m['model']} on {m['host']} gpu{m['gpu']} "
        f"({m['gpu_bdf'] or 'no pci'}) @ {m['short']}; "
        f"extras: {', '.join(extras) or 'none (wall only)'}",
    ]


def pytest_collection_modifyitems(config, items):
    # FlyDSL cases are opt-in: skip anything marked @pytest.mark.flydsl unless
    # --run-flydsl is passed.
    if config.getoption("--run-flydsl"):
        return
    skip_flydsl = pytest.mark.skip(reason="FlyDSL is opt-in; pass --run-flydsl")
    for item in items:
        if "flydsl" in item.keywords:
            item.add_marker(skip_flydsl)


def pytest_collect_file(parent, file_path):
    # Collect benchmark_*.py like test files so `pytest .` finds them without a
    # rename; skip init paths so an explicitly-passed file isn't double-collected.
    if (
        file_path.suffix == ".py"
        and file_path.name.startswith("benchmark_")
        and not parent.session.isinitpath(file_path)
    ):
        return pytest.Module.from_parent(parent, path=file_path)
    return None


class _MicroBench:
    """Handed to each test as the ``microbench`` fixture."""

    def __init__(self, request):
        self._request = request
        self._config = request.config

    def run(self, case, bench_callable):
        """Time *bench_callable* (returns metric records) and record it under *case*."""
        records = bench_callable()
        print_case(case, records)
        family = Path(self._request.module.__file__).stem
        record_bench(
            self._config._microbench_store, family, case, records,
            self._request.node.name,
        )
        return records


@pytest.fixture
def microbench(request):
    return _MicroBench(request)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    store = getattr(config, "_microbench_store", None)
    if not store:
        return
    table = format_results_table(store)
    if table:
        terminalreporter.write_line("")
        for row in table.splitlines():
            terminalreporter.write_line(row)
    aggregate = format_aggregate_table(store)
    if aggregate:
        terminalreporter.write_line("")
        for row in aggregate.splitlines():
            terminalreporter.write_line(row)


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    store = getattr(config, "_microbench_store", None)
    if not store:
        return
    if config.getoption("--dashboard-run"):
        plan = getattr(config, "_dashboard_plan", None)
        run_dir = write_dashboard_run(
            store,
            out_base=config.getoption("--dashboard-out"),
            csv_samples=config.getoption("--csv-samples"),
            meta=plan.meta if plan else None,
        )
        print(f"microbench: dashboard run -> {run_dir}")
        return
    written = write_bench_outputs(
        store,
        csv=config.getoption("--csv"),
        csv_samples=config.getoption("--csv-samples"),
    )
    for path in written:
        print(f"microbench: wrote {path}")
