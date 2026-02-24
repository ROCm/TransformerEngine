#!/usr/bin/env python3

import torch
import time
import subprocess
import os
import sys

def check_environment():
    """Verify environment is correctly configured."""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    
    # Check for MI300X/MI355X
    if "MI3" not in gpu_name and "gfx9" not in gpu_name:
        print(f"WARNING: Expected MI300X/MI355X, got {gpu_name}")
    
    # Check PyTorch
    print(f"PyTorch: {torch.__version__}")
    
    # Check Transformer Engine
    try:
        import transformer_engine.pytorch as te
        import transformer_engine_torch as tex
        print(f"Transformer Engine: Available")
    except ImportError as e:
        print(f"ERROR: Transformer Engine not available: {e}")
        sys.exit(1)
    
        
    return te, tex


def benchmark_pytorch_bf16(A_bf16, pt_linear, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        _ = pt_linear(A_bf16)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = pt_linear(A_bf16)
    torch.cuda.synchronize()
    
    return (time.time() - start) / iterations * 1e6  # Return in microseconds

def benchmark_te_bf16(te, linear, A_bf16, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        _ = linear(A_bf16)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = linear(A_bf16)
    torch.cuda.synchronize()
    
    return (time.time() - start) / iterations * 1e6  # Return in microseconds


def benchmark_te_fp8(te, linear, A_bf16, warmup=10, iterations=100):
    # Warmup
    with te.fp8_autocast(enabled=True):
        for _ in range(warmup):
            _ = linear(A_bf16)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        with te.fp8_autocast(enabled=True):
            _ = linear(A_bf16)
    torch.cuda.synchronize()
    
    return (time.time() - start) / iterations * 1e6  # Return in microseconds


def run_single_shape(te, M, K, N, hipblaslt_bench_path=None):
    results = {'M': M, 'K': K, 'N': N}
    
    # Create tensors
    A_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    B_bf16 = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    
    # Create TE Linear
    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).cuda()
    pt_linear = torch.nn.Linear(K, N, bias=False).to(device='cuda', dtype=torch.bfloat16)
    
    # PyTorch BF16
    results['torch_bf16_us'] = benchmark_pytorch_bf16(A_bf16, pt_linear=pt_linear)
    
    # TE BF16   
    results['te_bf16_us'] = benchmark_te_bf16(te, linear, A_bf16)
    
    # TE FP8
    results['te_fp8_us'] = benchmark_te_fp8(te, linear, A_bf16)
    
    return results


def print_results(results_list):
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"{'Shape':<20} {'Pytorch BF16':>14} {'TE BF16':>12} {'TE FP8':>12}")
    print(f"{'(M x K x N)':<20} {'(µs)':>12} {'(µs)':>14} {'(µs)':>12}")
    print("-" * 100)
    
    for r in results_list:
        shape = f"{r['M']} x {r['K']} x {r['N']}"
        torch_bf16 = f"{r['torch_bf16_us']:.1f}"
        te_bf16 = f"{r['te_bf16_us']:.1f}"
        te_fp8 = f"{r['te_fp8_us']:.1f}"
        
        print(f"{shape:<20} {torch_bf16:>14} {te_bf16:>12} {te_fp8:>12}")
    
    print("-" * 100)

def main():
    # Check environment
    te, tex = check_environment()
    
    # Test shapes
    K, N = 2048, 8192
    M_values = [512, 1024, 2048, 4096, 8192, 16384]
    
    print("=" * 70)
    print("RUNNING BENCHMARKS")
    print("=" * 70)
    print(f"Shapes: M ∈ {M_values}, K={K}, N={N}")
    print()
    
    results = []
    for M in M_values:
        print(f"Benchmarking M={M}...", end=" ", flush=True)
        r = run_single_shape(te, M, K, N)
        results.append(r)
        print(f"TE FP8: {r['te_fp8_us']:.1f}µs, TE BF16: {r['te_bf16_us']:.1f}µs")
    
    # Print summary
    print_results(results)


if __name__ == "__main__":
    main()