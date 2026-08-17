# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Work around a ROCm HSA-runtime teardown segfault after the triton_kernels tests.

On ROCm 7.14 the HIP runtime's atexit handler calls ``hsa_shut_down()``, which
destroys the GPU agents and their AQL queues; ``AqlQueue::~AqlQueue()`` then
writes to an already-freed HSA doorbell signal and the process dies with SIGSEGV.
Observed backtrace (``libamdhip64.so.7`` / ``libhsa-runtime64.so.1``)::

    exit() -> <hip atexit> -> rocr::HSA::hsa_shut_down()
           -> Runtime::Release() -> Unload() -> DestroyAgents()
           -> GpuAgent::~GpuAgent() -> AqlQueue::~AqlQueue()
           -> hsa_signal_store_screlease   <-- SIGSEGV

The crash happens *after* every test has passed and pytest has already written
its JUnit XML report, so the run is functionally green, but the process exits
139 and ``ci/_utils.sh``'s exit-code gate records a suite error. ``test_norms.py``
reliably trips it because it allocates the most Triton streams/queues of the
triton_kernels suites; the fault itself is purely in the ROCm libraries' shutdown
path, not in TE.

Bypass the buggy C-level atexit handler by hard-exiting with pytest's real exit
status once the session (and its report writing) has finished. Real failures
still propagate through ``exitstatus``, and a crash *during* the run (before
``pytest_sessionfinish``) is unaffected and still surfaces as an error. Remove
once the ROCm HSA shutdown crash is fixed.
"""

import os
import sys

import pytest
import torch


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    # Only ROCm hits the hsa_shut_down teardown segfault; leave CUDA/CPU exit
    # semantics (and their normal atexit cleanup) untouched.
    if getattr(torch.version, "hip", None) is None:
        return
    # trylast ensures the junitxml plugin and te_ci_result_sink have already
    # written their reports in this same hook before we hard-exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if exitstatus == 0 else int(exitstatus))
