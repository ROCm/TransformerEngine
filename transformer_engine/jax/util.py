# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
from functools import cache
import importlib.metadata
import jax, jax.numpy as jnp
import re

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


if not is_hip_extension():
  jnp_float8_e4m3_type = jnp.float8_e4m3fn
  jnp_float8_e5m2_type = jnp.float8_e5m2
else:
  jnp_float8_e4m3_type = jnp.float8_e4m3fnuz
  jnp_float8_e5m2_type = jnp.float8_e5m2fnuz
