# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
from functools import cache
import importlib.metadata
import re
import subprocess, sys
import jax.numpy as jnp

# check whether ROCm is supported by JAX
@cache
def is_hip_extension() -> bool:
  if any(re.match(r'jax-rocm\d+-plugin', d.metadata['Name'])
         for d in importlib.metadata.distributions()):
    return True
  return False

if is_hip_extension():
  @cache
  def is_mi200():
    import jax
    """check whether this machine is mi200/210/250"""
    return (re.search('AMD Instinct MI2.0', jax.devices()[0].device_kind) is not None)
  
@cache
def is_fp8_fnuz():
  if not is_hip_extension():
    return False
  ret = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path[:] = [p for p in sys.path if p not in ['', '.']]; "+
     "import os; os.environ['NVTE_FRAMEWORK']='none'; "+
     "import transformer_engine as te; exit(not te.common.is_fp8_fnuz())"]
     ).returncode
  return ret == 0

get_jnp_float8_e4m3_type = lambda: jnp.float8_e4m3fnuz if is_fp8_fnuz() else jnp.float8_e4m3fn
get_jnp_float8_e5m2_type = lambda: jnp.float8_e5m2fnuz if is_fp8_fnuz() else jnp.float8_e5m2
