#!/usr/bin/env python3
"""
Profile TE FP8, TE BF16, or PyTorch BF16 linear with torch.profiler.
Choose which backend to profile via --profile argument.
"""

import argparse
import sys
import torch

def check_environment():
    """Verify environment and return TE module."""
    if not torch.cuda.is_available():
        print("ERROR: No GPU available", file=sys.stderr)
        sys.exit(1)
    try:
        import transformer_engine.pytorch as te
        return te
    except ImportError as e:
        print(f"ERROR: Transformer Engine not available: {e}", file=sys.stderr)
        sys.exit(1)


def _profile_kwargs(with_stack=True, profile_memory=True):
    """Common profiler options for memory, stack, and shapes."""
    return dict(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=with_stack,
        profile_memory=profile_memory,
    )


def run_profiled_pytorch_bf16(pt_linear, A_bf16, warmup=5, profile_steps=10, with_stack=True, profile_memory=True):
    """Run PyTorch BF16 linear under the profiler."""
    for _ in range(warmup):
        _ = pt_linear(A_bf16)
    torch.cuda.synchronize()

    with torch.profiler.profile(**_profile_kwargs(with_stack=with_stack, profile_memory=profile_memory)) as prof:
        for _ in range(profile_steps):
            _ = pt_linear(A_bf16)
            torch.cuda.synchronize()

    return prof


def run_profiled_te_bf16(te, linear, A_bf16, warmup=5, profile_steps=10, with_stack=True, profile_memory=True):
    """Run TE BF16 linear under the profiler."""
    for _ in range(warmup):
        _ = linear(A_bf16)
    torch.cuda.synchronize()

    with torch.profiler.profile(**_profile_kwargs(with_stack=with_stack, profile_memory=profile_memory)) as prof:
        for _ in range(profile_steps):
            _ = linear(A_bf16)
            torch.cuda.synchronize()

    return prof


def run_profiled_te_fp8(te, linear, A_bf16, warmup=5, profile_steps=10, with_stack=True, profile_memory=True):
    """Run TE FP8 linear under the profiler."""
    with te.fp8_autocast(enabled=True):
        for _ in range(warmup):
            _ = linear(A_bf16)
    torch.cuda.synchronize()

    with torch.profiler.profile(**_profile_kwargs(with_stack=with_stack, profile_memory=profile_memory)) as prof:
        for _ in range(profile_steps):
            with te.fp8_autocast(enabled=True):
                _ = linear(A_bf16)
            torch.cuda.synchronize()

    return prof


def main():
    parser = argparse.ArgumentParser(
        description="Profile TE FP8, TE BF16, or PyTorch BF16 linear with torch.profiler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile", "-p",
        choices=["te_fp8", "te_bf16", "pytorch_bf16"],
        required=True,
        help="Which backend to profile: te_fp8, te_bf16, or pytorch_bf16",
    )
    parser.add_argument(
        "--M", type=int, default=2048,
        help="Batch size (rows of A)",
    )
    parser.add_argument(
        "--K", type=int, default=2048,
        help="Input features (cols of A, rows of weight)",
    )
    parser.add_argument(
        "--N", type=int, default=8192,
        help="Output features (cols of weight)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations before profiling",
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Iterations to capture in the profile",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for Chrome trace JSON (e.g. trace_te_fp8.json). If not set, prints summary only.",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Do not print profiler key_averages() summary",
    )
    parser.add_argument(
        "--no-stack", action="store_true",
        help="Do not record Python stack traces (faster profiling)",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Do not profile memory allocation",
    )
    args = parser.parse_args()
    args.print_summary = not args.no_summary
    args.with_stack = not args.no_stack
    args.profile_memory = not args.no_memory

    te = check_environment()
    M, K, N = args.M, args.K, args.N

    print(f"Shape: M={M}, K={K}, N={N}")
    print(f"Profiling: {args.profile}")
    print()

    A_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).cuda()
    pt_linear = torch.nn.Linear(K, N, bias=False).to(device="cuda", dtype=torch.bfloat16)

    if args.profile == "pytorch_bf16":
        prof = run_profiled_pytorch_bf16(
            pt_linear, A_bf16, warmup=args.warmup, profile_steps=args.steps,
            with_stack=args.with_stack, profile_memory=args.profile_memory,
        )
    elif args.profile == "te_bf16":
        prof = run_profiled_te_bf16(
            te, linear, A_bf16, warmup=args.warmup, profile_steps=args.steps,
            with_stack=args.with_stack, profile_memory=args.profile_memory,
        )
    else:  # te_fp8
        prof = run_profiled_te_fp8(
            te, linear, A_bf16, warmup=args.warmup, profile_steps=args.steps,
            with_stack=args.with_stack, profile_memory=args.profile_memory,
        )

    if args.print_summary:
        key_averages = prof.key_averages()
        print("=" * 80)
        print("PROFILER SUMMARY (key_averages) - sort by CUDA time total")
        print("=" * 80)
        print(key_averages.table(sort_by="cuda_time_total", row_limit=40))
        print()
        print("=" * 80)
        print("PROFILER SUMMARY (key_averages) - sort by CPU time total")
        print("=" * 80)
        print(key_averages.table(sort_by="cpu_time_total", row_limit=40))
        print()
        if args.profile_memory:
            print("=" * 80)
            print("PROFILER SUMMARY (key_averages) - sort by self CUDA memory")
            print("=" * 80)
            print(key_averages.table(sort_by="self_cuda_memory_usage", row_limit=40))
            print()
        if args.with_stack:
            print("=" * 80)
            print("PROFILER SUMMARY - grouped by stack (top 20)")
            print("=" * 80)
            key_avg_stack = prof.key_averages(group_by_stack_n=5)
            print(key_avg_stack.table(sort_by="cuda_time_total", row_limit=20))
            print()

    if args.output:
        prof.export_chrome_trace(args.output)
        print(f"Chrome trace written to: {args.output}")
        print("Open in Chrome: chrome://tracing")


if __name__ == "__main__":
    main()
