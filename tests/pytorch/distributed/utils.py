# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

import os, signal, subprocess


def run_proctree_with_timeout(cmd, timeout, **kwargs):
    """Run a command in a subprocess and check for errors."""

    if timeout is None:
        return subprocess.run(cmd, **kwargs)

    if "timeout" in kwargs:
        raise ValueError("Timeout should be passed as a separate argument, not in kwargs")

    stdout, stderr = None, None
    capture_output = kwargs.pop("capture_output", False)
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    check = kwargs.pop("check", False)

    kwargs["start_new_session"] = True  # To use killpg as termination fallback
    p = subprocess.Popen(cmd, **kwargs)
    try:
        if capture_output:
            stdout, stderr = p.communicate(timeout=timeout)
        else:
            p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.terminate()
        try:
            # Give the process time to terminate gracefully
            timeout = 60
            if capture_output:
                stdout, stderr = p.communicate(timeout=timeout)
            else:
                p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            if capture_output:
                stdout, stderr = p.communicate()
        if check:
            raise

    # Handle check=True
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(
            p.returncode,
            cmd,
            output=stdout,
            stderr=stderr
        )

    return subprocess.CompletedProcess(
        cmd,
        p.returncode,
        stdout,
        stderr
    )
