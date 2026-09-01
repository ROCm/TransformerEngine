# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""MXFP4 recipe (plugin plan S5.4 sidecar burn-down): the class body lives fork-side.

`transformer_engine.common.recipe` re-exports it, so the public name and every existing import
path are unchanged. Imported at the BOTTOM of recipe/__init__.py - by then CustomRecipe, Format
and _BACKWARD_OVERRIDES are bound, so the circular import resolves in the standard way.
"""
import os
from typing import Any, Callable, Optional

from pydantic.dataclasses import dataclass

from transformer_engine.common.recipe import _BACKWARD_OVERRIDES, CustomRecipe, Format


@dataclass()
class MXFP4BlockScaling(CustomRecipe):
    """
    Use the MXFP4 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 (same as MXFP8) consecutive values is scaled together using
    their own scaling factor. The type of the scaling factor is E8M0
    (8 bits of exponent, 0 bits of mantissa), equivalent to scaling
    by a power of 2. FP4 (E2M1) values are stored two per byte,
    with a single uint8 holding two 4-bit elements.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), in this recipe the quantized tensor and its transpose
    are not numerically equivalent. Due to this, when Transformer Engine
    needs both the MXFP4 tensor and its transpose (e.g. to calculate both
    forward and backward pass), during the quantization both versions are
    computed from the high precision input to avoid double quantization
    errors.

    Unlike MXFP8, the columnwise (transpose) FP4 data is stored in transposed
    layout: buffer shape is (K, M/2) for logical (M, K), i.e. N×M rather
    than M×N. Rowwise remains (M, K/2).

    Parameters
    ----------
    fp4_format : {Format.E2M1}, default = Format.E2M1
             FP4 data format.
    """

    margin: int = 0
    fp4_format: Format = Format.E2M1
    # Must remain set: Recipe paths expect a valid `fp8_format` even
    # though the MXFP4 code path is FP4-only. Changing it can break compatibility.
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = os.getenv("NVTE_BACKWARD_OVERRIDE", None)
    use_hadamard: bool = os.getenv("NVTE_MXFP4_USE_HADAMARD", "0") == "1"
    # Self-wired in __post_init__ (plugin plan S5.1): dispatch flows through upstream's own
    # CustomRecipe -> CustomRecipeState path; the ROCm-side factory lives in
    # te_rocm.recipes.adapter_2_18 (one adapter module per certified upstream version).
    qfactory: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        if self.qfactory is None:
            from transformer_engine.te_rocm.recipes.adapter_2_18 import make_mxfp4_qfactory

            self.qfactory = make_mxfp4_qfactory(self)
        assert self.fp4_format == Format.E2M1, "Only E2M1 is supported for MXFP4 scaling."
        assert (
            self.backward_override in _BACKWARD_OVERRIDES
        ), "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."

    def __repr__(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"fp4_format={str(self.fp4_format).split('.')[1]}"
        )


# Pickle/extra-state pinning: the checkpoint extra-state policy is keyed by
# (module, class name) = ("transformer_engine.common.recipe", "MXFP4BlockScaling").
MXFP4BlockScaling.__module__ = "transformer_engine.common.recipe"
