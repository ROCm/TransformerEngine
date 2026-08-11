# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""ROCm CI hardening for the distributed PyTorch launchers.

The tests in this directory spawn ``torchrun``/``mpirun`` children via
``subprocess``. In CI the outer per-test bound is pytest-timeout using the
``thread`` method (see ``ci/_utils.sh``); on expiry it calls ``os._exit()``,
which kills the pytest process but *orphans* those children -- they keep holding
GPU memory and can poison every subsequent test in the job.

To get a clean, attributable failure instead, this conftest wraps each launch
with the ``timeout`` coreutil: a hung child is sent SIGTERM (letting torchrun
tear down its workers), then SIGKILL after a grace period, *before* the outer
pytest-timeout fires. The inner bound defaults to ``PYTEST_TIMEOUT`` minus a
grace so the child is always reaped first.
"""

from functools import cache
import os
import shutil
import subprocess

try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
except Exception:  # torch missing/broken -> let the tests themselves report it
    IS_HIP_EXTENSION = False

# Commands treated as distributed launchers worth bounding.
_LAUNCHERS = ("torchrun", "mpirun")

@cache
def _terminate_timeout_seconds():
    """Grace between SIGTERM and the SIGKILL backstop."""
    try:
        return str(max(1, int(os.environ.get("TE_DIST_LAUNCH_KILL_AFTER", "60"))))
    except ValueError:
        return "60"


def _launch_timeout_seconds():
    """Inner per-launch bound, in seconds, as a string for the coreutil."""
    explicit = os.environ.get("TE_DIST_LAUNCH_TIMEOUT")
    if explicit:
        return explicit
    # Fire a bit before the outer pytest-timeout so the child is reaped cleanly
    # rather than orphaned when the watchdog calls os._exit().
    try:
        outer = int(os.environ.get("PYTEST_TIMEOUT", "1200"))
    except ValueError:
        return "1200"
    return str(max(60, outer - 60 - int(_terminate_timeout_seconds())))


def _is_launcher(cmd):
    if not isinstance(cmd, (list, tuple)) or not cmd:
        return False
    head = os.path.basename(str(cmd[0]))
    if head == "timeout":  # already bounded by the caller
        return False
    return head in _LAUNCHERS


def _wrap(cmd):
    return ["timeout", "-k", _terminate_timeout_seconds(), "-v", _launch_timeout_seconds(), *cmd]


# Patch at import (collection) time so it is active for every test in this dir.
if IS_HIP_EXTENSION and shutil.which("timeout") is not None:
    _orig_run = subprocess.run

    def _run_with_timeout(cmd, *args, **kwargs):
        if _is_launcher(cmd):
            cmd = _wrap(cmd)
        return _orig_run(cmd, *args, **kwargs)

    subprocess.run = _run_with_timeout
