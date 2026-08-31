#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""pytest glue for the microbenchmarks.

Thin shim: the option/CSV/timing logic lives in utils.py; this file only wires
those helpers into pytest hooks and exposes the ``microbench`` fixture.  Run a
family with, e.g.::

    pytest benchmark_gemm.py --csv                 # write benchmark_gemm.csv
    pytest benchmark_gemm.py -k "bf16 and QKV"     # select by parametrize id
    pytest benchmark_gemm.py -k triton             # select the triton backend
"""

from pathlib import Path

import pytest

from utils import (
    collect_kernel_rows,
    configure_rotating,
    format_results_table,
    print_case,
    record_bench,
    write_bench_outputs,
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
        help="Also profile GPU kernels via torch.profiler and write a _kernel_profile CSV.",
    )
    group.addoption(
        "--rotating", nargs="?", type=int, const=0, default=None, metavar="MB",
        help="Rotate inputs through a ring of buffers (optional MB budget). On by default.",
    )
    group.addoption(
        "--no-rotating", action="store_true", default=False,
        help="Disable input buffer rotation.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: TE GPU microbenchmark")
    configure_rotating(config.getoption("--rotating"), config.getoption("--no-rotating"))
    config._microbench_store = {}


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
        kernel_rows = None
        if self._config.getoption("--kernel-profile"):
            kernel_rows = collect_kernel_rows(bench_callable, case)
        record_bench(
            self._config._microbench_store, family, case, records,
            kernel_rows, self._request.node.name,
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


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    store = getattr(config, "_microbench_store", None)
    if not store:
        return
    written = write_bench_outputs(
        store,
        csv=config.getoption("--csv"),
        csv_samples=config.getoption("--csv-samples"),
        kernel_profile=config.getoption("--kernel-profile"),
    )
    for path in written:
        print(f"microbench: wrote {path}")
