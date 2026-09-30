# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Time the Triton max_logit kernel against a CK attention forward (without max_logit).

The CK reference is flash_attn.flash_attn_func (the ROCm flash_attn package runs CK kernels).
It is a proxy for TE's own CK forward, which it will be replaced with once TE is installed.

    python benchmarks/attention/benchmark_max_logit.py
"""

import argparse
import json

import torch
from flash_attn import flash_attn_func

from transformer_engine.pytorch.triton_kernels.max_logit import max_logit_fwd

# (batch, seqlen, num_heads, num_heads_kv)
SHAPES = [
    (8, 4096, 32, 32),
    (4, 8192, 32, 32),
    (1, 32768, 32, 32),
    (4, 8192, 32, 8),  # GQA
]


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--config", type=json.loads, default=None,
                        help='max_logit launch config override, e.g. \'{"BLOCK_M": 128, "BLOCK_N": 128}\'')
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    d = args.head_dim

    print(f"{torch.cuda.get_device_name()}  dtype={args.dtype}  d={d}  config={args.config}")
    header = f"{'B':>3} {'S':>6} {'H':>3} {'Hkv':>3} {'mask':>7} | {'CK fwd ms':>9} {'max_logit ms':>12} {'% of CK':>8} {'TFLOPS':>7}"
    print(header)
    print("-" * len(header))
    for batch, seqlen, num_heads, num_heads_kv in SHAPES:
        q = torch.randn(batch, seqlen, num_heads, d, dtype=dtype, device="cuda")
        k = torch.randn(batch, seqlen, num_heads_kv, d, dtype=dtype, device="cuda")
        v = torch.randn_like(k)
        for causal in (False, True):
            mask = "causal" if causal else "no_mask"
            ck_ms = _time_ms(lambda: flash_attn_func(q, k, v, causal=causal), args.warmup, args.iters)
            ml_ms = _time_ms(
                lambda: max_logit_fwd(q, k, qkv_format="bshd", attn_mask_type=mask, config=args.config),
                args.warmup,
                args.iters,
            )
            # Q.K^T only: 2 * B * H * S_q * S_k * d, halved for causal
            flops = 2 * batch * num_heads * seqlen * seqlen * d * (0.5 if causal else 1.0)
            print(
                f"{batch:>3} {seqlen:>6} {num_heads:>3} {num_heads_kv:>3} {mask:>7} | {ck_ms:>9.3f}"
                f" {ml_ms:>12.3f} {100 * ml_ms / ck_ms:>7.1f}% {flops / ml_ms / 1e9:>7.0f}"
            )


if __name__ == "__main__":
    main()
