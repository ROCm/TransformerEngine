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
    
    # Check tuning environment variables
    tuning_count = os.environ.get('TE_HIPBLASLT_TUNING_RUN_COUNT', '0')
    print(f"TE_HIPBLASLT_TUNING_RUN_COUNT: {tuning_count}")
    
    if tuning_count == '0':
        print("NOTE: Tuning disabled. Set TE_HIPBLASLT_TUNING_RUN_COUNT=10 for optimal kernel selection")
    
    print()
    return te, tex


def benchmark_hipblaslt_direct(M, K, N, hipblaslt_bench_path=None):
    # Try to find hipblaslt-bench
    if hipblaslt_bench_path is None:
        candidates = [
            "/work/rocm-libraries/projects/hipblaslt/build/release/clients/hipblaslt-bench",
            "/opt/rocm/bin/hipblaslt-bench",
            "hipblaslt-bench",
        ]
        for path in candidates:
            if os.path.exists(path) or os.system(f"which {path} > /dev/null 2>&1") == 0:
                hipblaslt_bench_path = path
                break
    
    if hipblaslt_bench_path is None:
        return None, None
    
    env = os.environ.copy()
    
    try:
        # FP8 benchmark
        cmd_fp8 = f"{hipblaslt_bench_path} -m {M} -n {N} -k {K} --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --compute_type f32_r --transA T --transB N -i 100 -j 10 2>/dev/null | tail -1"
        result = subprocess.run(cmd_fp8, shell=True, capture_output=True, text=True, env=env)
        fp8_line = result.stdout.strip()
        fp8_us = float(fp8_line.split(',')[-1]) if fp8_line else None
        
        # BF16 benchmark
        cmd_bf16 = f"{hipblaslt_bench_path} -m {M} -n {N} -k {K} --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --transA T --transB N -i 100 -j 10 2>/dev/null | tail -1"
        result = subprocess.run(cmd_bf16, shell=True, capture_output=True, text=True, env=env)
        bf16_line = result.stdout.strip()
        bf16_us = float(bf16_line.split(',')[-1]) if bf16_line else None
        
        return fp8_us, bf16_us
    except Exception as e:
        print(f"hipblaslt-bench failed: {e}")
        return None, None


def benchmark_pytorch_bf16(A_bf16, B_bf16, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        _ = torch.mm(A_bf16, B_bf16)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(A_bf16, B_bf16)
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


def benchmark_te_fp8_graph(te, linear, A_bf16, warmup=10, iterations=100):
    # Warmup and capture graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    
    with torch.cuda.stream(s):
        with te.fp8_autocast(enabled=True):
            for _ in range(3):
                out = linear(A_bf16)
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with te.fp8_autocast(enabled=True):
            out = linear(A_bf16)
    
    # Warmup graph replay
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        g.replay()
    torch.cuda.synchronize()
    
    return (time.time() - start) / iterations * 1e6  # Return in microseconds


def run_single_shape(te, M, K, N, hipblaslt_bench_path=None):
    results = {'M': M, 'K': K, 'N': N}
    
    # Create tensors
    A_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    B_bf16 = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    
    # Create TE Linear
    linear = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).cuda()
    
    # hipblaslt-bench (if available)
    fp8_kernel_us, bf16_kernel_us = benchmark_hipblaslt_direct(M, K, N, hipblaslt_bench_path)
    results['hipblaslt_fp8_us'] = fp8_kernel_us
    results['hipblaslt_bf16_us'] = bf16_kernel_us
    if fp8_kernel_us and bf16_kernel_us:
        results['kernel_speedup'] = bf16_kernel_us / fp8_kernel_us
    
    # PyTorch BF16
    results['torch_bf16_us'] = benchmark_pytorch_bf16(A_bf16, B_bf16)
    
    # TE FP8
    results['te_fp8_us'] = benchmark_te_fp8(te, linear, A_bf16)
    results['te_fp8_speedup'] = results['torch_bf16_us'] / results['te_fp8_us']
    
    # TE FP8 with CUDA graph
    try:
        results['te_fp8_graph_us'] = benchmark_te_fp8_graph(te, linear, A_bf16)
        results['te_fp8_graph_speedup'] = results['torch_bf16_us'] / results['te_fp8_graph_us']
    except Exception as e:
        results['te_fp8_graph_us'] = None
        results['te_fp8_graph_speedup'] = None
    
    # Calculate overhead
    if fp8_kernel_us:
        results['overhead_us'] = results['te_fp8_us'] - fp8_kernel_us
        results['overhead_pct'] = results['overhead_us'] / results['te_fp8_us'] * 100
    
    return results


def print_results(results_list):
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"{'Shape':<20} {'hipblaslt FP8':>12} {'hipblaslt BF16':>14} {'torch.mm BF16':>14} {'TE FP8':>12} {'TE FP8+Graph':>14} {'FP8 vs BF16':>12}")
    print(f"{'(M x K x N)':<20} {'(µs)':>12} {'(µs)':>14} {'(µs)':>14} {'(µs)':>12} {'(µs)':>14} {'(speedup)':>12}")
    print("-" * 100)
    
    for r in results_list:
        shape = f"{r['M']} x {r['K']} x {r['N']}"
        hipblaslt_fp8 = f"{r['hipblaslt_fp8_us']:.1f}" if r.get('hipblaslt_fp8_us') else "N/A"
        hipblaslt_bf16 = f"{r['hipblaslt_bf16_us']:.1f}" if r.get('hipblaslt_bf16_us') else "N/A"
        torch_bf16 = f"{r['torch_bf16_us']:.1f}"
        te_fp8 = f"{r['te_fp8_us']:.1f}"
        te_graph = f"{r['te_fp8_graph_us']:.1f}" if r.get('te_fp8_graph_us') else "N/A"
        speedup = f"{r['te_fp8_speedup']:.2f}x"
        
        print(f"{shape:<20} {hipblaslt_fp8:>12} {hipblaslt_bf16:>14} {torch_bf16:>14} {te_fp8:>12} {te_graph:>14} {speedup:>12}")
    
    print("-" * 100)
    
    # Summary
    print("\nKEY FINDINGS:")
    
    # Find crossover point
    crossover = None
    for r in results_list:
        if r['te_fp8_speedup'] >= 1.0:
            crossover = r['M']
            break
    
    if crossover:
        print(f"  - Crossover point: M ≈ {crossover} (FP8 faster than BF16 above this batch size)")
    
    # Overhead analysis
    overhead_results = [r for r in results_list if r.get('overhead_us')]
    if overhead_results:
        avg_overhead = sum(r['overhead_us'] for r in overhead_results) / len(overhead_results)
        print(f"  - Average TE overhead: {avg_overhead:.0f} µs")
        print(f"  - Overhead is fixed cost, dominates small batch sizes")
    
    # Kernel speedup
    kernel_results = [r for r in results_list if r.get('kernel_speedup')]
    if kernel_results:
        avg_kernel_speedup = sum(r['kernel_speedup'] for r in kernel_results) / len(kernel_results)
        print(f"  - Average kernel speedup (hipblaslt): {avg_kernel_speedup:.2f}x")
        print(f"  - Kernels are fast; overhead is the problem")


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
        print(f"TE FP8: {r['te_fp8_us']:.1f}µs, BF16: {r['torch_bf16_us']:.1f}µs, Speedup: {r['te_fp8_speedup']:.2f}x")
    
    # Print summary
    print_results(results)
    
    # Print diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    small_batch = results[0]  # M=512



if __name__ == "__main__":
    main()
EOF