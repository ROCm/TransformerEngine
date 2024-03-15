# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
import jax
# check whether AMD GPU is loaded
def is_hip_extension() -> bool:
  has_rocm = False
  #iterate through all devices in jax
  for dev in jax.devices():
    if "rocm" in dev.client.platform_version:
      has_rocm=True
      break
  return has_rocm

