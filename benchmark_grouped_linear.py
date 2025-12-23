import torch
import torch.nn as nn
from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

def benchmark_grouped_linear():
    # Reset FP8 state
    FP8GlobalStateManager.reset()
    
    m = 24576  # Total tokens routed to all experts
    in_features = 2048  # Hidden size
    out_features = 2816  # Output features per expert
    num_gemms = 64  # Number of experts
    dtype = torch.bfloat16
    
    # Create m_splits that sum to 24576
    # For simplicity, split evenly across experts
    # In practice, this would come from the router
    m_splits = [m // num_gemms] * num_gemms  # Each expert gets 384 tokens
    
    print(f"Benchmark Configuration:")
    print(f"  Total tokens: {m}")
    print(f"  Number of experts (num_gemms): {num_gemms}")
    print(f"  Tokens per expert: {m // num_gemms}")
    print(f"  Input shape: [{m}, {in_features}]")
    print(f"  Output features per expert: {out_features}")
    print(f"  Output shape: [{m}, {out_features}]")
    print(f"  m_splits (first 5): {m_splits[:5]}...")
    print(f"  dtype: {dtype}")
    print(f"  Device: cuda")
    print()
    
    # Create GroupedLinear module
    grouped_linear = GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        params_dtype=dtype,
        device="cuda",
    ).train()
    
    # Create input tensor
    inp = torch.randn(m, in_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Verify input strides
    print(f"Input strides: {inp.stride()}")
    print(f"Input shape: {inp.shape}")
    print()
    
    # Warmup
    print("Warming up (forward + backward)...")
    for _ in range(10):
        out = grouped_linear(inp, m_splits)
        loss = out.sum()
        loss.backward()
        # Zero gradients
        grouped_linear.zero_grad()
        inp.grad = None
    torch.cuda.synchronize()
    print("Warmup complete.\n")
    
    # Profile
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
            # Zero gradients
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
    
    # Summary
    print(f"\nBenchmark Summary:")
    print(f"  Iterations profiled: {num_iters}")
    print(f"  Forward + Backward pass per iteration")
    print(f"  Output shape: {out.shape}")
    print(f"  Output dtype: {out.dtype}")

if __name__ == "__main__":
    benchmark_grouped_linear()
