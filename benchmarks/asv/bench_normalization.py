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

    def time_forward(self, M, hidden, norm_type):
        self.norm(self.x)
        torch.cuda.synchronize()

    def time_forward_backward(self, M, hidden, norm_type):
        out = self.norm(self.x)
        out.backward(self.grad_out)
        self.x.grad = None
        for p in self.norm.parameters():
            p.grad = None
        torch.cuda.synchronize()

if __name__ == "__main__":
    from direct_run import run_as_main
    run_as_main(__file__)
