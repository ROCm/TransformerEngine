# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Amax bookkeeping helpers for lite backend.

These helpers let call sites update a quantizer's amax tensor without
materializing an FP8 tensor. Used by the "skip FP8 dgrad round-trip"
optimization on the norm backward path: when the only downstream consumer
of an FP8 tensor is going to dequantize it immediately, we can skip the
cast entirely and still preserve DelayedScaling amax history by running
a standalone reduction on the BF16 source.
"""

import torch


def update_amax_from_bf16(quantizer, bf16_tensor):
    """Update quantizer.amax from a BF16 tensor's abs-max.

    Only meaningful for DelayedScaling (Float8Quantizer), which uses
    amax history to pick next step's scalar scale. CurrentScaling computes
    per-row scales in-kernel on each use (no cross-step bookkeeping), and
    MXFP8 uses per-block scales computed at quantize time — for both, this
    call is a no-op.

    Matches the amax update the Triton cast-transpose kernel performs as a
    side effect of the FP8 cast (see quantize.py: amax_out=q.amax), so the
    stored amax value is identical whether we take the cast path or skip it.
    """
    if quantizer is None or not hasattr(quantizer, "amax"):
        return
    if type(quantizer).__name__ != "Float8Quantizer":
        return
    if bf16_tensor is None or bf16_tensor.numel() == 0:
        return
    quantizer.amax.copy_(bf16_tensor.abs().amax())
