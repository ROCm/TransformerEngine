# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine.jax

if __name__ == "__main__":
    print("OK")

def test_lazy_init():
    import jax, os, pytest, subprocess, sys
    if not transformer_engine.jax.is_hip_extension():
        pytest.skip("This is ROCm test")
    os.environ["NVTE_FRAMEWORK"] = "jax"
    ret = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['HIP_VISIBLE_DEVICES']=''; import transformer_engine"]
         ).returncode
    assert ret == 0, "Failed to import TE witout visible devices"
    dev_count = jax.local_device_count()
    if dev_count >= 2:
        dev_count_test = subprocess.run([sys.executable, '-c',
                                        "import transformer_engine, jax, os; " +
                                        "os.environ['HIP_VISIBLE_DEVICES']='0'; "+
                                        "exit(jax.local_device_count())"]
                                        ).returncode
        assert dev_count_test==1, (
            "Changing visible devices after import did not affect reported devices count")
