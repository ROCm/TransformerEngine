# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import contextlib
import os
import torch
import triton
import triton.language as tl
import transformer_engine_torch as tex
from functools import cache

@cache
def get_arch():
    return triton.runtime.driver.active.get_current_target().arch

def is_cdna3():
    return get_arch() == "gfx942"

# ── AMD compiler knobs (scoped) ──
# The AMD codegen toggles live on ``triton.knobs.amd`` and are process-global: they
# are read when a kernel is compiled, so leaving them set would silently change the
# codegen of every unrelated Triton kernel compiled afterwards (cast, gmm, MXFP8,
# ...). We therefore apply them only around our own launches and restore the prior
# state on exit. Triton's setter keeps the backing env var in sync with the
# attribute, so we snapshot/restore both.
_AMD_KNOB_ENV = {
    "use_async_copy": "TRITON_HIP_USE_ASYNC_COPY",
    "scalarize_packed_fops": "AMDGCN_SCALARIZE_PACKED_FOPS",
    "use_block_pingpong": "TRITON_HIP_USE_BLOCK_PINGPONG",
}


def _amd_knob_overrides(*, is_tn: bool) -> dict:
    """Knob values to apply for the enclosed GEMM launch.

    gfx950 (CDNA4): force async_copy / scalarize / block_pingpong on.
    gfx942 (CDNA3): async_copy / scalarize help NT/NN but regress TN/wgrad
    ~5-8%, so gate them on the GEMM layout (block_pingpong left at its default).
    """
    if is_cdna3():
        enable = not is_tn
        return {"use_async_copy": enable, "scalarize_packed_fops": enable}
    return {
        "use_async_copy": True,
        "scalarize_packed_fops": True,
        "use_block_pingpong": True,
    }


@contextlib.contextmanager
def _amd_compiler_knobs(*, is_tn: bool, overrides: dict | None = None):
    """Temporarily apply AMD Triton compiler knobs, restoring prior state on exit.

    Scoped rather than set once because the knobs are process-global and read at
    kernel compile time; leaving them set would leak into other kernels' codegen.
    ``overrides`` replaces the default :func:`_amd_knob_overrides` set.
    """
    amd = getattr(getattr(triton, "knobs", None), "amd", None)
    if amd is None:
        yield
        return
    if overrides is None:
        overrides = _amd_knob_overrides(is_tn=is_tn)
    saved = {
        name: (name in amd.__dict__, amd.__dict__.get(name), os.environ.get(env_key))
        for name, env_key in _AMD_KNOB_ENV.items()
        if name in overrides
    }
    try:
        for name, value in overrides.items():
            setattr(amd, name, value)
        yield
    finally:
        for name, (had_override, override_val, env_val) in saved.items():
            if had_override:
                amd.__dict__[name] = override_val
            else:
                amd.__dict__.pop(name, None)
            env_key = _AMD_KNOB_ENV[name]
            if env_val is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = env_val

get_torch_e4m3_type = lambda: torch.float8_e4m3fn if not is_cdna3() else torch.float8_e4m3fnuz
get_torch_e5m2_type = lambda: torch.float8_e5m2 if not is_cdna3() else torch.float8_e5m2fnuz

# Convert te dtype to torch type.
def te_dtype_to_torch_dtype(te_dtype):
    return {
        tex.DType.kByte: torch.uint8, 
        tex.DType.kInt32: torch.int32,
        tex.DType.kFloat32: torch.float32, 
        tex.DType.kFloat16: torch.float16, 
        tex.DType.kBFloat16: torch.bfloat16, 
        tex.DType.kFloat8E4M3: get_torch_e4m3_type(), 
        tex.DType.kFloat8E5M2: get_torch_e5m2_type(), 
    }[te_dtype]

# Convert PyTorch type to TE type.
def torch_dtype_to_te_dtype(dtype):
    return {
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
        torch.float8_e4m3fnuz: tex.DType.kFloat8E4M3,
        torch.float8_e4m3fn: tex.DType.kFloat8E4M3,
        torch.float8_e5m2fnuz: tex.DType.kFloat8E5M2,
        torch.float8_e5m2: tex.DType.kFloat8E5M2,
    }[dtype]

# Convert te dtype to pytorch aten type.
# used in ln_out allocation
def te_dtype_to_aten_dtype(te_dtype):
    return {
        tex.DType.kInt32: torch.int32,
        tex.DType.kFloat32: torch.float32, 
        tex.DType.kFloat16: torch.float16, 
        tex.DType.kBFloat16: torch.bfloat16, 
        tex.DType.kByte: torch.uint8, 
        tex.DType.kFloat8E4M3: torch.uint8, 
        tex.DType.kFloat8E5M2: torch.uint8, 
    }[te_dtype]

# convert te_dtype to their enum value in class DType
def te_dtype_to_enum_value(te_dtype):
    return {
      tex.DType.kByte: 0, 
      tex.DType.kInt32: 1,
      #kInt64 not exported to transformer_engine_pytorch
      #tex.DType.kInt64: 2,
      tex.DType.kFloat32: 3,
      tex.DType.kFloat16: 4,
      tex.DType.kBFloat16: 5,
      tex.DType.kFloat8E4M3: 6,
      tex.DType.kFloat8E5M2: 7,
    }[te_dtype]

# convert te_dtype to their enum value in class DType
def enum_value_to_te_dtype(te_dtype_enum):
    return {
      0: tex.DType.kByte, 
      1: tex.DType.kInt32,
      #kInt64 not exported to transformer_engine_pytorch
      #2: tex.DType.kInt64,
      3: tex.DType.kFloat32,
      4: tex.DType.kFloat16,
      5: tex.DType.kBFloat16,
      6: tex.DType.kFloat8E4M3,
      7: tex.DType.kFloat8E5M2,
    }[te_dtype_enum]

def is_fp8_torch_dtype(dtype):
    return dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)

def te_dtype_to_triton_dtype(dtype: tex.DType):
    if dtype == tex.DType.kFloat8E4M3:
        return tl.float8e4b8 if is_cdna3() else tl.float8e4nv
    if dtype == tex.DType.kFloat8E5M2:
        return tl.float8e5b16 if is_cdna3() else tl.float8e5
    if dtype == tex.DType.kFloat32:
        return tl.float32
    if dtype == tex.DType.kFloat16:
        return tl.float16
    if dtype == tex.DType.kBFloat16:
        return tl.bfloat16

def get_fp8_max(dtype: tex.DType):
    if dtype == tex.DType.kFloat8E4M3:
        return 240.0 if is_cdna3() else 448.0
    if dtype == tex.DType.kFloat8E5M2:
        return 57344.0
