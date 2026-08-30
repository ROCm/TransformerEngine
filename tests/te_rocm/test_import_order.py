# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Import-order conformance (plan P3): the seam must behave the same in every order a consumer
can import, and must fail LOUDLY - never split-brain - when the extension is imported before
the package on a clean install. Each case is a fresh interpreter."""
import os
import subprocess
import sys

import pytest

SEAM = "transformer_engine_torch"
COMPILED = "transformer_engine_rocm_torch"


def run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "NVTE_FRAMEWORK": "pytorch"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd="/tmp", timeout=300)


def test_a_package_first():
    r = run(f"import sys, transformer_engine.pytorch, {SEAM} as tex; assert tex is sys.modules['{SEAM}'] is sys.modules['{COMPILED}']; print('OK')")
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


def test_b_extension_first_fails_loudly():
    """On a clean install nothing answers to upstream's name until the package installs the seam.
    A silent success here would mean an NVIDIA wheel or stale .so is answering - split-brain."""
    r = run(f"import {SEAM}")
    assert r.returncode != 0, "importing the extension first must fail; something answered under upstream's name"
    assert "ModuleNotFoundError" in r.stderr, r.stderr[-800:]


def test_c_consumer_deep_submodule_first():
    r = run(f"import sys, torch, transformer_engine.pytorch.module.linear as L; assert L.tex is sys.modules['{SEAM}']; print('OK')")
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


def test_d_reload_cpp_extensions_is_stable():
    r = run("import importlib, transformer_engine.pytorch, transformer_engine.pytorch.cpp_extensions as ce\n"
            "b=set(dir(ce)); importlib.reload(ce); a=set(dir(ce)); assert a==b, (a^b); print('OK')")
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


def test_e_fork_child_sees_the_seam():
    r = run(f"import os, sys, transformer_engine.pytorch\n"
            f"pid=os.fork()\n"
            f"if pid==0:\n    import {SEAM} as t; os._exit(0 if t is sys.modules['{COMPILED}'] else 3)\n"
            f"_, st=os.waitpid(pid,0); assert os.waitstatus_to_exitcode(st)==0; print('OK')")
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]
