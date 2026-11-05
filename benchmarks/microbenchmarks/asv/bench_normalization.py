#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""RMSNorm and LayerNorm benchmarks on activation-sized tensors.

Memory-bound; we report GB/s. The hidden dimension is swept over the distinct
model hidden sizes and M (batch * seq_len) over typical training sizes.
"""

import torch
import transformer_engine.pytorch as te

from driver import BenchBase, run_as_main
from models import M_SIZES, unique_hidden_sizes

NORMS = {"RMSNorm": te.RMSNorm, "LayerNorm": te.LayerNorm}


class BenchNormalization(BenchBase):
    params = [M_SIZES, unique_hidden_sizes(), list(NORMS)]
    param_names = ["M", "hidden", "norm_type"]

    def setup(self, M, hidden, norm_type):
        dtype = torch.bfloat16
        self.norm = NORMS[norm_type](hidden).to(device="cuda", dtype=dtype)
        self.x = torch.randn(M, hidden, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn_like(self.norm(self.x))

    def work_forward(self, M, hidden, norm_type):
        # read input (2B) + write output (2B)
        return {"bytes": M * hidden * 4}

    def work_forward_backward(self, M, hidden, norm_type):
        # fwd read+write (4B) + bwd read input+grad_out, write grad_in (6B)
        return {"bytes": M * hidden * 10}

    def time_forward(self, M, hidden, norm_type):
        return self._time(lambda: self.norm(self.x))

    def time_forward_backward(self, M, hidden, norm_type):
        t = self._time(lambda: self.norm(self.x).backward(self.grad_out))
        self.x.grad = None
        for p in self.norm.parameters():
            p.grad = None
        return t


if __name__ == "__main__":
    run_as_main(__file__)
