# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
from functools import cache
import importlib.metadata
import jax
import re
from transformer_engine.transformer_engine_jax import get_device_compute_capability

# check whether ROCm is supported by JAX
@cache
def is_hip_extension() -> bool:
  if any(re.match(r'jax-rocm\d+-plugin', d.metadata['Name'])
             for d in importlib.metadata.distributions()):
    return True
  try:
    import jaxlib.rocm #pre JAX 0.4.30 way
    return True
  except ImportError:
    pass
  return False

if is_hip_extension():
  @cache
  def is_mi200():
      """check whether this machine is mi200/210/250"""
      import re
      return (re.search('AMD Instinct MI2.0', jax.devices()[0].device_kind) is not None)
  
@cache
def is_fp8_fnuz():
  return is_hip_extension() and get_device_compute_capability(0) == 94
