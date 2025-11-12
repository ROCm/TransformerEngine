import torch
import torch.nn as nn
from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import time

try:
    import grouped_gemm
    ops = grouped_gemm.ops
    GROUPED_GEMM_AVAILABLE = True
except ImportError:
    GROUPED_GEMM_AVAILABLE = False
    print("Warning: grouped_gemm library not available")

def benchmark_te_grouped_linear(m, in_features, out_features, num_experts, dtype, num_iters=100):
    """Benchmark TransformerEngine GroupedLinear"""
    print("\n" + "="*80)
    print("Benchmarking TransformerEngine GroupedLinear")
    print("="*80)
    
    FP8GlobalStateManager.reset()
    
    # Create m_splits
    m_splits = [m // num_experts] * num_experts
    
    # Create GroupedLinear module
    grouped_linear = GroupedLinear(
        num_gemms=num_experts,
        in_features=in_features,
        out_features=out_features,
        bias=False,  # No bias for fair comparison
        params_dtype=dtype,
        device="cuda",
    ).train()
    
    # Create input
    inp = torch.randn(m, in_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Warmup
    for _ in range(10):
        out = grouped_linear(inp, m_splits)
        loss = out.sum()
        loss.backward()
        grouped_linear.zero_grad()
        inp.grad = None
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        out = grouped_linear(inp, m_splits)
        loss = out.sum()
        loss.backward()
        grouped_linear.zero_grad()
        inp.grad = None
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = (end - start) * 1000  # Convert to ms
    avg_time = elapsed / num_iters
    
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Average time per iteration: {avg_time:.3f} ms")
    print(f"  Throughput: {num_iters / (elapsed / 1000):.2f} iter/s")
    
    return avg_time, out

def benchmark_grouped_gemm_ops(m, in_features, out_features, num_experts, dtype, num_iters=100):
    """Benchmark grouped_gemm.ops.gmm"""
    print("\n" + "="*80)
    print("Benchmarking grouped_gemm.ops.gmm")
    print("="*80)
    
    if not GROUPED_GEMM_AVAILABLE:
        print("  SKIPPED: grouped_gemm not available")
        return None, None
    
    # Create tokens_per_expert (must be a tensor on CPU with int64 dtype!)
    tokens_per_expert = torch.tensor([m // num_experts] * num_experts, dtype=torch.int64, device="cpu")
    
    # Create weights for each expert
    # IMPORTANT: grouped_gemm does NOT support transposition (trans_b=True not supported)
    # Must use trans_b=False with weights in [num_experts, in_features, out_features] layout
    weights = torch.randn(num_experts, in_features, out_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Create input
    inp = torch.randn(m, in_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Warmup
    for _ in range(10):
        out = ops.gmm(inp, weights, tokens_per_expert, trans_b=False)
        loss = out.sum()
        loss.backward()
        weights.grad = None
        inp.grad = None
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        out = ops.gmm(inp, weights, tokens_per_expert, trans_b=False)
        loss = out.sum()
        loss.backward()
        weights.grad = None
        inp.grad = None
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = (end - start) * 1000  # Convert to ms
    avg_time = elapsed / num_iters
    
    print(f"  Total time: {elapsed:.2f} ms")
    print(f"  Average time per iteration: {avg_time:.3f} ms")
    print(f"  Throughput: {num_iters / (elapsed / 1000):.2f} iter/s")
    
    return avg_time, out

def main():
    # Configuration
    m = 24576  # Total tokens
    in_features = 2048
    out_features = 2816
    num_experts = 64
    dtype = torch.bfloat16
    num_iters = 100
    
    print("Benchmark Comparison: TransformerEngine vs grouped_gemm")
    print("="*80)
    print(f"Configuration:")
    print(f"  Total tokens: {m}")
    print(f"  Number of experts: {num_experts}")
    print(f"  Tokens per expert: {m // num_experts}")
    print(f"  Input features: {in_features}")
    print(f"  Output features: {out_features}")
    print(f"  dtype: {dtype}")
    print(f"  Iterations: {num_iters}")
    
    # Benchmark TransformerEngine
    te_time, te_out = benchmark_te_grouped_linear(
        m, in_features, out_features, num_experts, dtype, num_iters
    )
    
    # Benchmark grouped_gemm
    gg_time, gg_out = benchmark_grouped_gemm_ops(
        m, in_features, out_features, num_experts, dtype, num_iters
    )
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(f"TransformerEngine GroupedLinear: {te_time:.3f} ms/iter")
    if gg_time is not None:
        print(f"grouped_gemm.ops.gmm:           {gg_time:.3f} ms/iter")
        speedup = te_time / gg_time
        if speedup > 1:
            print(f"\nResult: grouped_gemm is {speedup:.2f}x FASTER")
        else:
            print(f"\nResult: TransformerEngine is {1/speedup:.2f}x FASTER")
    else:
        print("grouped_gemm.ops.gmm:           N/A (not available)")

if __name__ == "__main__":
    main()

