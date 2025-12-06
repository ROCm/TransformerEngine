import torch
import torch.nn as nn
from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

def benchmark_grouped_linear():
    # Reset FP8 state
    FP8GlobalStateManager.reset()
    
    # Configuration based on the provided input dimensions from trace
    # Input Dims: [24576, 2048] - total tokens routed to experts
    # num_gemms (experts): 64
    m = 24576  # Total tokens routed to all experts
    in_features = 2048  # Hidden size (from trace)
    out_features = 2816  # Output features per expert
    num_gemms = 64  # Number of experts (grouped GEMMs)
    dtype = torch.bfloat16
    
    # Create uneven m_splits that sum to 24576
    # Simulate real MoE routing with power law distribution
    # Some experts get more tokens than others (realistic workload)
    import random
    random.seed(42)  # For reproducibility
    
    # Generate uneven distribution using power law
    weights = [1.0 / (i + 1)**0.5 for i in range(num_gemms)]
    total_weight = sum(weights)
    m_splits = []
    for i in range(num_gemms - 1):
        tokens = int(m * weights[i] / total_weight)
        m_splits.append(tokens)
    # Last expert gets remainder to ensure exact sum
    m_splits.append(m - sum(m_splits))
    
    # Shuffle to make it more realistic (not sorted by size)
    random.shuffle(m_splits)
    
    # Verify that m_splits sums to exactly m
    assert sum(m_splits) == m, f"m_splits sum {sum(m_splits)} != {m}"
    
    print(f"Benchmark Configuration:")
    print(f"  Total tokens: {m}")
    print(f"  Number of experts (num_gemms): {num_gemms}")
    print(f"  Token distribution: UNEVEN (power law)")
    print(f"    Min tokens per expert: {min(m_splits)}")
    print(f"    Max tokens per expert: {max(m_splits)}")
    print(f"    Avg tokens per expert: {sum(m_splits) / len(m_splits):.1f}")
    print(f"    Sum of m_splits: {sum(m_splits)} (verified == {m})")
    print(f"  Input shape: [{m}, {in_features}]")
    print(f"  Output features per expert: {out_features}")
    print(f"  Output shape: [{m}, {out_features}]")
    print(f"  m_splits (first 10): {m_splits[:10]}...")
    print(f"  dtype: {dtype}")
    print(f"  Device: cuda")
    print()
    
    # Create GroupedLinear module (train mode for backward pass)
    grouped_linear = GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        params_dtype=dtype,
        device="cuda",
    ).train()  # Changed to train() to enable gradients
    
    # Create input tensor with specified dimensions (requires_grad for backward)
    inp = torch.randn(m, in_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Verify input strides (should be [2048, 1] for contiguous input)
    print(f"Input strides: {inp.stride()}")
    print(f"Input shape: {inp.shape}")
    print()
    
    # Warmup (forward + backward)
    print("Warming up (forward + backward)...")
    for _ in range(10):
        out = grouped_linear(inp, m_splits)
        loss = out.sum()
        loss.backward()
        # Zero gradients for next iteration
        grouped_linear.zero_grad()
        inp.grad = None
    torch.cuda.synchronize()
    print("Warmup complete.\n")
    
    # Profile with PyTorch profiler (forward + backward)
    print("Starting profiling (forward + backward)...")
    num_iters = 100
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_iters):
            out = grouped_linear(inp, m_splits)
            loss = out.sum()
            loss.backward()
            # Zero gradients for next iteration
            grouped_linear.zero_grad()
            inp.grad = None
        torch.cuda.synchronize()
    
    print("Profiling complete.\n")
    
    # Print profiler results
    print("=" * 80)
    print("Profiler Results (sorted by CUDA time):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print("\n" + "=" * 80)
    print("Profiler Results (sorted by CPU time):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Export chrome trace for detailed analysis
    trace_file = "grouped_linear_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace exported to: {trace_file}")
    print("You can view it by opening chrome://tracing in Chrome and loading the file.")
    
    # Summary
    print(f"\nBenchmark Summary:")
    print(f"  Iterations profiled: {num_iters}")
    print(f"  Forward + Backward pass per iteration")
    print(f"  Output shape: {out.shape}")
    print(f"  Output dtype: {out.dtype}")

if __name__ == "__main__":
    benchmark_grouped_linear()

