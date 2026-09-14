# This file was modified for portability to AMDGPU
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine.pytorch

if __name__ == "__main__":
    print("OK")

def _amdsmi_pythonpath():
    """Return ROCm amdsmi source dir if the package is not already installed.

    torch counts devices through amdsmi whenever it is importable, reading
    HIP_VISIBLE_DEVICES at call time rather than asking an already initialized HIP
    runtime -- which is what makes a visible-devices change after import observable.

    With TheRock pip wheels, bindings live under the rocm-sdk-core tree
    (``<core-root>/share/amd_smi``).
    """
    import importlib.util

    if importlib.util.find_spec("amdsmi") is not None:
        return None
    try:
        from rocm_sdk_core._cli import _get_core_module_path

        candidate = _get_core_module_path() / "share/amd_smi"
        if candidate.is_dir():
            return str(candidate)
    except (ImportError, ModuleNotFoundError, OSError):
        pass
    return None


def _env_with_amdsmi_pythonpath(env, amdsmi_path):
    if amdsmi_path is None:
        return env
    env = dict(env)
    prefix = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = amdsmi_path + (":" + prefix if prefix else "")
    return env


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
    amdsmi_path = _amdsmi_pythonpath()
    if amdsmi_path is not None:
        sys.path.insert(0, amdsmi_path)
    test_env = _env_with_amdsmi_pythonpath(os.environ, amdsmi_path)
    amdsmi_import = "import amdsmi; " if amdsmi_path is not None else ""
    dev_count = torch.cuda.device_count()
    if dev_count >= 2:
        dev_count_test = subprocess.run(
            [sys.executable, '-c',
             "import sys; sys.path[:] = [p for p in sys.path if p not in ['', '.']]; "+
             amdsmi_import +
             "import transformer_engine, torch, os; os.environ['HIP_VISIBLE_DEVICES']='0'; "+
             "exit(torch.cuda.device_count())"],
            env=test_env,
        ).returncode
        assert dev_count_test==1, (
            "Changing visible devices after import did not affect reported devices count")
