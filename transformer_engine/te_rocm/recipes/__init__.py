# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""ROCm recipes over upstream's CustomRecipe protocol (plugin plan S5.1).

The stable internal contract: a *quantizer factory builder* per recipe -
``make_<recipe>_qfactory(recipe) -> Callable[[QuantizerRole], Quantizer]`` - implemented once
per certified upstream version in ``adapter_<ver>`` modules, so upstream protocol drift is
absorbed in exactly one file per version.
"""
