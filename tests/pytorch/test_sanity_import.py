# This file was modified for portability to AMDGPU
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine.pytorch

if __name__ == "__main__":
    print("OK")

AMDSMI_SRC = "/opt/rocm/share/amd_smi"


def _install_amdsmi():
    """Install amdsmi for the duration of a test, returning True if we installed it.

    torch counts devices through amdsmi whenever it is importable, reading
    HIP_VISIBLE_DEVICES at call time rather than asking an already initialized HIP
    runtime -- which is what makes a visible-devices change after import observable.
    """
    import importlib.util, os, subprocess, sys
    if importlib.util.find_spec("amdsmi") is not None or not os.path.isdir(AMDSMI_SRC):
        return False
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", AMDSMI_SRC]
        ).returncode == 0


def _remove_amdsmi():
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "amdsmi"])


def test_lazy_init():
    import torch, os, pytest, subprocess, sys
    if not torch.utils.cpp_extension.IS_HIP_EXTENSION:
        pytest.skip("This is ROCm test")
    os.environ["NVTE_FRAMEWORK"] = "pytorch"
    ret = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path[:] = [p for p in sys.path if p not in ['', '.']]; "+
         "import os; os.environ['HIP_VISIBLE_DEVICES']=''; import transformer_engine"]
         ).returncode
    assert ret == 0, "Failed to import TE without visible devices"
    amdsmi_installed_here = _install_amdsmi()
    try:
        dev_count = torch.cuda.device_count()
        if dev_count >= 2:
            print("dev_count:",dev_count)
            dev_count_test = subprocess.run(
                [sys.executable, '-c',
                 "import sys; sys.path[:] = [p for p in sys.path if p not in ['', '.']]; "+
                 "import transformer_engine, torch, os; os.environ['HIP_VISIBLE_DEVICES']='0'; "+
                 "exit(torch.cuda.device_count())"]
                 ).returncode
            assert dev_count_test==1, (
                "Changing visible devices after import did not affect reported devices count")
    finally:
        if amdsmi_installed_here:
            _remove_amdsmi()
