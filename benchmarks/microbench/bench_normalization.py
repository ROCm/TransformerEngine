#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
RMSNorm and LayerNorm benchmarks on activation-sized tensors.

Modern models predominantly use RMSNorm, but we benchmark both since TE
supports both and they share the same kernel infrastructure.

The M dimension (batch * seq_len) is swept across typical training sizes;
hidden sizes are derived from the shared dense-model configs.
"""

import torch
import transformer_engine.pytorch as te

from driver import time_func
from shapes import M_SIZES, hidden_sizes

NORMS = {"RMSNorm": te.RMSNorm, "LayerNorm": te.LayerNorm}

# Sweep unique hidden sizes from the shared dense-model configs; replace or
# extend (e.g. HIDDEN_SIZES.append(2048)) to add custom shapes.
HIDDEN_SIZES = sorted(set(hidden_sizes().values()))


class BenchNormalization:
    params = [M_SIZES, HIDDEN_SIZES, list(NORMS)]
    param_names = ["M", "hidden", "norm_type"]
    timeout = 120

    def setup(self, M, hidden, norm_type):
        dtype = torch.bfloat16
        self.norm = NORMS[norm_type](hidden).to(device="cuda", dtype=dtype)
        self.x = torch.randn(M, hidden, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn_like(self.norm(self.x))

    def work_forward(self, M, hidden, norm_type):
        # Read input (2B) + write output (2B) = 4 bytes per element
        return {"bytes": M * hidden * 4}

    def work_forward_backward(self, M, hidden, norm_type):
        # Fwd: read+write (4B), Bwd: read input+grad_out+write grad_in (6B) = 10B
        return {"bytes": M * hidden * 10}

    def time_forward(self, M, hidden, norm_type):
        return time_func(lambda: self.norm(self.x))

    def time_forward_backward(self, M, hidden, norm_type):
        def fn():
            out = self.norm(self.x)
            out.backward(self.grad_out)
        return time_func(fn)


if __name__ == "__main__":
    from driver import main
    main(__file__)
