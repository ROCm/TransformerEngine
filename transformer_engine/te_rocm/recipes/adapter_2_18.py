# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""CustomRecipe adapter for upstream 2.18 (pin 868d8d92) - plugin plan S5.1.

Maps upstream's ``qfactory(role: QuantizerRole) -> Quantizer`` protocol onto the ROCm MXFP4
quantizer, reproducing the retired ``MXFP4BlockScalingRecipeState`` behavior exactly:

  forward   input        -> shuffle (rowwise=False, columnwise=True)
            weight       -> shuffle (rowwise=True,  columnwise=True)
            other/output -> no shuffle
  backward  grad_output  -> no shuffle

All four core modules (linear, layernorm_linear, layernorm_mlp, grouped_linear) supply
role metadata at this pin, so the tensor_type dispatch is total for TE's own modules. A bare
role (custom module without ``get_quantizer_roles``) gets the no-shuffle configuration -
numerically correct, potentially slower; upstream already warns in that path.
"""
from __future__ import annotations

from typing import Any, Callable

_SHUFFLE_BY_TENSOR_TYPE = {
    "input": (False, True),
    "weight": (True, True),
}


def make_mxfp4_qfactory(recipe: Any) -> Callable[[Any], Any]:
    """Build the qfactory for one MXFP4BlockScaling recipe instance.

    Torch-side imports happen at call time: this module is imported from
    ``transformer_engine.common.recipe`` (framework-neutral), the factory runs inside
    the PyTorch quantization machinery.
    """

    def qfactory(role: Any) -> Any:
        from transformer_engine.pytorch.quantization import get_fp4_te_dtype
        from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

        shuffle_rowwise, shuffle_columnwise = _SHUFFLE_BY_TENSOR_TYPE.get(
            getattr(role, "tensor_type", ""), (False, False)
        )
        return MXFP4Quantizer(
            fp4_dtype=get_fp4_te_dtype(recipe),
            rowwise=True,
            columnwise=True,
            shuffle_rowwise_data=shuffle_rowwise,
            shuffle_columnwise_data=shuffle_columnwise,
            with_gemm_swizzled_scales=True,
            use_hadamard=recipe.use_hadamard,
        )

    return qfactory
