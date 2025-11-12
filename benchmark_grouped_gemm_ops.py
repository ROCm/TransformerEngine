import torch
import torch.nn as nn

try:
    import grouped_gemm
    ops = grouped_gemm.ops
    GROUPED_GEMM_AVAILABLE = True
except ImportError:
    GROUPED_GEMM_AVAILABLE = False
    print("Warning: grouped_gemm library not available")

def benchmark_grouped_gemm_ops():
    if not GROUPED_GEMM_AVAILABLE:
        print("ERROR: grouped_gemm library is required but not available")
        print("Please install it to run this benchmark")
        return
    
    # Configuration based on the provided input dimensions from trace
    # Input Dims: [24576, 2048] - total tokens routed to experts
    # num_gemms (experts): 64
    m = 24576  # Total tokens routed to all experts
    in_features = 2048  # Hidden size (from trace)
    out_features = 2816  # Output features per expert
    num_experts = 64  # Number of experts
    dtype = torch.bfloat16
    
    # Create tokens_per_expert that sum to 24576
    # For simplicity, split evenly across experts
    # In practice, this would come from the router
    # NOTE: batch_sizes/tokens_per_expert must be on CPU and int64 for grouped_gemm
    tokens_per_expert = torch.tensor([m // num_experts] * num_experts, dtype=torch.int64, device="cpu")  # Each expert gets 384 tokens
    
    print(f"Benchmark Configuration (grouped_gemm.ops.gmm):")
    print(f"  Total tokens: {m}")
    print(f"  Number of experts: {num_experts}")
    print(f"  Tokens per expert: {m // num_experts}")
    print(f"  Input shape: [{m}, {in_features}]")
    print(f"  Output features per expert: {out_features}")
    print(f"  Output shape: [{m}, {out_features}]")
    print(f"  tokens_per_expert device: {tokens_per_expert.device} (must be CPU)")
    print(f"  tokens_per_expert shape: {tokens_per_expert.shape}")
    print(f"  tokens_per_expert sum: {tokens_per_expert.sum().item()}")
    print(f"  dtype: {dtype}")
    print(f"  Device: cuda")
    print()
    
    # Create weight tensors for each expert
    # IMPORTANT: grouped_gemm does NOT support transposition with CUTLASS grouped GEMM
    # (see https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
    # Therefore, we MUST use trans_b=False and store weights in [in_features, out_features] layout
    # - trans_b=False: weights are [num_experts, in_features, out_features] = [64, 2048, 2816]
    # - trans_b=True: NOT SUPPORTED or inefficient!
    weights = torch.randn(num_experts, in_features, out_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Create input tensor with specified dimensions (requires_grad for backward)
    inp = torch.randn(m, in_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Verify input strides (should be [2048, 1] for contiguous input)
    print(f"Input strides: {inp.stride()}")
    print(f"Input shape: {inp.shape}")
    print(f"Weights shape: {weights.shape} (num_experts, in_features, out_features)")
    print(f"Expected output shape: [{m}, {out_features}]")
    print()
    
    # Warmup (forward + backward)
    print("Warming up (forward + backward)...")
    for _ in range(10):
        out = ops.gmm(inp, weights, tokens_per_expert, trans_b=False)
        loss = out.sum()
        loss.backward()
        # Zero gradients for next iteration
        weights.grad = None
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
            out = ops.gmm(inp, weights, tokens_per_expert, trans_b=False)
            loss = out.sum()
            loss.backward()
            # Zero gradients for next iteration
            weights.grad = None
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
    trace_file = "grouped_gemm_ops_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace exported to: {trace_file}")
    print("You can view it by opening chrome://tracing in Chrome and loading the file.")
    
    # Summary
    print(f"\nBenchmark Summary:")
    print(f"  Implementation: grouped_gemm.ops.gmm")
    print(f"  Iterations profiled: {num_iters}")
    print(f"  Forward + Backward pass per iteration")
    print(f"  Output shape: {out.shape}")
    print(f"  Output dtype: {out.dtype}")

if __name__ == "__main__":
    benchmark_grouped_gemm_ops()

