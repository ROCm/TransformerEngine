#!/usr/bin/env python
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Attention micro-benchmark using te.DotProductAttention.

Benchmarks fused multi-head attention (with flash attention backend) for
model configurations with grouped-query attention (GQA).

Models:
  - Llama 3   8B (TP=1, TP=8), 70B (TP=8), 405B (TP=8)
  - Qwen 2.5  7B (TP=1), 72B (TP=8)

Forward FLOPs = 4 * batch * num_q_heads * seq_len^2 * head_dim
  (two matmuls: Q@K^T and attn@V, each contributing 2*b*h*s^2*d)
Backward FLOPs = 2 * Forward FLOPs (approximately)

Sources for model configs:
  https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
  https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json

Output: benchmark_attention.csv (written to cwd)
"""

import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te

# Sweep parameters
BATCH_SIZE = 2
SEQ_LEN_LIST = [1024, 2048, 4096, 8192]

# (name, num_q_heads, num_kv_heads, head_dim, tp)
MODEL_CONFIGS = [
    ("Llama3-8B/TP1",   32,  8, 128,  1),
    ("Llama3-8B/TP8",   32,  8, 128,  8),
    ("Llama3-70B/TP8",  64,  8, 128,  8),
    ("Llama3-405B/TP8", 128, 8, 128,  8),
    ("Qwen2.5-7B/TP1",  28,  4, 128,  1),
    ("Qwen2.5-72B/TP8", 64,  8, 128,  8),
]


def _generate_attn_test_cases():
    test_cases = []
    for (name, n_q, n_kv, hd, tp) in MODEL_CONFIGS:
        q_per_gpu = n_q // tp
        kv_per_gpu = n_kv // tp
        if q_per_gpu < 1 or kv_per_gpu < 1:
            continue
        for seq_len in SEQ_LEN_LIST:
            test_cases.append({
                "Case": name,
                "batch": BATCH_SIZE,
                "seq_len": seq_len,
                "num_q_heads": q_per_gpu,
                "num_kv_heads": kv_per_gpu,
                "head_dim": hd,
            })
    return test_cases


def bench_attention(batch, seq_len, num_q_heads, num_kv_heads, head_dim):
    device = "cuda"
    dtype = torch.bfloat16

    attn = te.DotProductAttention(
        num_attention_heads=num_q_heads,
        kv_channels=head_dim,
        num_gqa_groups=num_kv_heads,
        attn_mask_type="causal",
    ).to(device=device, dtype=dtype)

    q = torch.randn(seq_len, batch, num_q_heads, head_dim,
                     dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(seq_len, batch, num_kv_heads, head_dim,
                     dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(seq_len, batch, num_kv_heads, head_dim,
                     dtype=dtype, device=device, requires_grad=True)

    fwd_func = lambda: attn(q, k, v)
    out = fwd_func()
    grad_out = torch.randn_like(out)

    def fwd_bwd_func():
        out = attn(q, k, v)
        out.backward(grad_out)
        q.grad = None
        k.grad = None
        v.grad = None

    fwd_bwd_func()

    # FLOPs: two matmuls (Q@K^T and attn@V), each 2*b*h*s^2*d
    fwd_flops = 4 * batch * num_q_heads * seq_len * seq_len * head_dim
    bwd_flops = 2 * fwd_flops

    # Warmup
    for _ in range(20):
        fwd_func()
        fwd_bwd_func()
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 100

    fwd_ms = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func}).timeit(n_iters).mean * 1e3
    fwd_bwd_ms = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func}).timeit(n_iters).mean * 1e3

    bwd_ms = max(fwd_bwd_ms - fwd_ms, 0.0)

    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) / 1e12 if bwd_ms > 0 else 0.0

    print(f"  Forward      {fwd_ms:.3f} ms | {fwd_tflops:.2f} TFLOPS")
    print(f"  Backward     {bwd_ms:.3f} ms | {bwd_tflops:.2f} TFLOPS (derived)")

    return {
        "TE Forward Time (ms)": f"{fwd_ms:.2f}",
        "TE Forward TFLOPS": f"{fwd_tflops:.2f}",
        "TE Backward Time (ms)": f"{bwd_ms:.2f}",
        "TE Backward TFLOPS": f"{bwd_tflops:.2f}",
    }


if __name__ == "__main__":
    import pandas as pd

    test_cases = _generate_attn_test_cases()

    columns = [
        "Case", "batch", "seq_len", "num_q_heads", "num_kv_heads", "head_dim",
        "TE Forward Time (ms)",
        "TE Forward TFLOPS",
        "TE Backward Time (ms)",
        "TE Backward TFLOPS",
    ]
    rows = []

    # Warmup run
    c = test_cases[0]
    print(f"\n{'='*60}")
    print(f"WARMUP: {c['Case']} b={c['batch']} s={c['seq_len']} "
          f"qh={c['num_q_heads']} kvh={c['num_kv_heads']} hd={c['head_dim']}")
    print(f"{'='*60}")
    bench_attention(batch=c["batch"], seq_len=c["seq_len"],
                    num_q_heads=c["num_q_heads"], num_kv_heads=c["num_kv_heads"],
                    head_dim=c["head_dim"])

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case['Case']} b={case['batch']} s={case['seq_len']} "
              f"qh={case['num_q_heads']} kvh={case['num_kv_heads']} hd={case['head_dim']}")
        print(f"{'='*60}")
        try:
            metrics = bench_attention(
                batch=case["batch"],
                seq_len=case["seq_len"],
                num_q_heads=case["num_q_heads"],
                num_kv_heads=case["num_kv_heads"],
                head_dim=case["head_dim"],
            )
            row = {
                "Case": case["Case"],
                "batch": case["batch"],
                "seq_len": case["seq_len"],
                "num_q_heads": case["num_q_heads"],
                "num_kv_heads": case["num_kv_heads"],
                "head_dim": case["head_dim"],
                **metrics,
            }
            rows.append(row)
        except Exception as e:
            print(f"FAILED: {case['Case']}: {e}")
            raise

    results = pd.DataFrame(rows, columns=columns)

    out_csv = "benchmark_attention.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
