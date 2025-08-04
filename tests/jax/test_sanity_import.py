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
    dev_count = jax.local_device_count()
    if dev_count < 2:
        pytest.skip("Test requires several visible devices")
    os.environ["NVTE_FRAMEWORK"] = "jax"
    dev_count_test = subprocess.run([sys.executable, '-c', 
                                     "import transformer_engine, jax, os; " +
                                     "os.environ['HIP_VISIBLE_DEVICES']='0'; "+
                                     "exit(jax.local_device_count())"]
                                     ).returncode
    assert dev_count_test==1, "Changing visible devices after import did not affect device count"
