#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""RMSNorm and LayerNorm benchmarks on activation-sized tensors."""

import torch
import transformer_engine.pytorch as te

NORMS = {"RMSNorm": te.RMSNorm, "LayerNorm": te.LayerNorm}
HIDDEN_SIZES = [3584, 4096, 8192, 16384]


class BenchNormalization:
    params = [[1024, 2048, 4096, 8192], HIDDEN_SIZES, list(NORMS)]
    param_names = ["M", "hidden", "norm_type"]
    timeout = 120

    def setup(self, M, hidden, norm_type):
        dtype = torch.bfloat16
        self.norm = NORMS[norm_type](hidden).to(device="cuda", dtype=dtype)
        self.x = torch.randn(M, hidden, dtype=dtype, device="cuda", requires_grad=True)
        self.grad_out = torch.randn_like(self.norm(self.x))
        self._evt = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def work_forward(self, M, hidden, norm_type):
        # Read input (2B) + write output (2B) = 4 bytes per element
        return {"bytes": M * hidden * 4}

    def work_forward_backward(self, M, hidden, norm_type):
        # Fwd: read+write (4B), Bwd: read input+grad_out+write grad_in (6B) = 10B
        return {"bytes": M * hidden * 10}

    def time_forward(self, M, hidden, norm_type):
        self._evt[0].record()
        self.norm(self.x)
        self._evt[1].record()
        torch.cuda.synchronize()
        return self._evt[0].elapsed_time(self._evt[1]) / 1000

    def time_forward_backward(self, M, hidden, norm_type):
        self._evt[0].record()
        out = self.norm(self.x)
        out.backward(self.grad_out)
        self._evt[1].record()
        torch.cuda.synchronize()
        self.x.grad = None
        for p in self.norm.parameters():
            p.grad = None
        return self._evt[0].elapsed_time(self._evt[1]) / 1000

if __name__ == "__main__":
    from driver import run_as_main
    run_as_main(__file__)
