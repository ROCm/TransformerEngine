import torch
import torch.nn as nn
from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast

def check_for_nan_inf(tensor_name: str, tensor: torch.Tensor):
    """Check for NaN and Inf values in tensors during forward pass."""
    if not isinstance(tensor, torch.Tensor):
        print(f"Skipping check for '{tensor_name}' as it's not a PyTorch tensor.")
        return

    # A boolean flag to track if any issue is found
    has_issue = False
   
    if torch.isnan(tensor).any():
        print(f"🚨 ALERT: NaN value detected in tensor '{tensor_name}'!")
        has_issue = True
   
    if torch.isinf(tensor).any():
        print(f"🚨 ALERT: Inf value detected in tensor '{tensor_name}'!")
        has_issue = True

    # The new 'else' part
    max_val = tensor.abs().max().detach().item()
    # print(f"✅ DEBUG: Max absolute value in '{tensor_name}' tensor is: {max_val}")

    if max_val > 1e20:  # Adjust this threshold based on your model's dtype and scale
        print(f"⚠️ WARNING: Very large values detected in the '{tensor_name}' tensor. "
              "This could be a source of numerical instability and lead to overflow.")
    if not has_issue:
        print(f"✅ OK: No NaN or Inf values detected in tensor '{tensor_name}'.")

def print_rank_0(message):
    if torch.cuda.current_device() == 0:
        print(message)

def register_nan_inf_hooks(model):
    """Register forward and backward hooks on all modules to check for NaN/Inf values."""
    def forward_hook(module, input, output):
        module_name = module.__class__.__name__
        
        # Check inputs
        if isinstance(input, tuple):
            for idx, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    check_for_nan_inf(f"[FWD] {module_name}_input_{idx}", inp)
        elif isinstance(input, torch.Tensor):
            check_for_nan_inf(f"[FWD] {module_name}_input", input)
        
        # Check outputs
        if isinstance(output, tuple):
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    check_for_nan_inf(f"[FWD] {module_name}_output_{idx}", out)
        elif isinstance(output, torch.Tensor):
            check_for_nan_inf(f"[FWD] {module_name}_output", output)
    
    def backward_hook(module, grad_input, grad_output):
        module_name = module.__class__.__name__
        
        # Check gradient outputs (gradients w.r.t. outputs)
        if isinstance(grad_output, tuple):
            for idx, grad in enumerate(grad_output):
                if isinstance(grad, torch.Tensor) and grad is not None:
                    check_for_nan_inf(f"[BWD] {module_name}_grad_output_{idx}", grad)
        elif isinstance(grad_output, torch.Tensor):
            check_for_nan_inf(f"[BWD] {module_name}_grad_output", grad_output)
        
        # Check gradient inputs (gradients w.r.t. inputs)
        if isinstance(grad_input, tuple):
            for idx, grad in enumerate(grad_input):
                if isinstance(grad, torch.Tensor) and grad is not None:
                    check_for_nan_inf(f"[BWD] {module_name}_grad_input_{idx}", grad)
        elif isinstance(grad_input, torch.Tensor):
            check_for_nan_inf(f"[BWD] {module_name}_grad_input", grad_input)
    
    # Register hooks on all modules
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)
    
    # Also register hooks on parameters to check their gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, param_name=name: check_for_nan_inf(f"[PARAM_GRAD] {param_name}", grad) or grad)
    
    print_rank_0("✅ NaN/Inf detection hooks registered on model (forward + backward + parameter gradients)")


def benchmark_grouped_linear():
    # Reset FP8 state
    FP8GlobalStateManager.reset()
    
    m = 32 * 24576  # Total tokens routed to all experts
    in_features = 2048  # Hidden size
    out_features = 2816  # Output features per expert
    num_gemms = 64  # Number of experts
    dtype = torch.bfloat16
    
    # Create m_splits that sum to 24576
    # Randomize to simulate real MoE routing distribution
    # In practice, this would come from the router
    torch.manual_seed(42)  # For reproducibility
    random_weights = torch.rand(num_gemms)
    m_splits_float = (random_weights / random_weights.sum()) * m
    m_splits = m_splits_float.int().tolist()
    
    # Adjust for rounding errors to ensure exact sum
    diff = m - sum(m_splits)
    if diff > 0:
        # Add remaining tokens to the first 'diff' experts
        for i in range(diff):
            m_splits[i] += 1
    elif diff < 0:
        # Remove excess tokens from the first 'abs(diff)' experts
        for i in range(abs(diff)):
            if m_splits[i] > 1:  # Ensure we don't go below 1
                m_splits[i] -= 1
    
    print(f"Benchmark Configuration:")
    print(f"  Total tokens: {m}")
    print(f"  Number of experts (num_gemms): {num_gemms}")
    print(f"  Tokens per expert (avg): {m / num_gemms:.1f}")
    print(f"  Tokens per expert (min/max): {min(m_splits)}/{max(m_splits)}")
    print(f"  m_splits sum verification: {sum(m_splits)}")
    print(f"  Input shape: [{m}, {in_features}]")
    print(f"  Output features per expert: {out_features}")
    print(f"  Output shape: [{m}, {out_features}]")
    print(f"  m_splits (first 10): {m_splits[:10]}...")
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

    register_nan_inf_hooks(grouped_linear)
    
    # Create input tensor
    inp = torch.randn(m, in_features, dtype=dtype, device="cuda", requires_grad=True)
    
    # Verify input strides
    print(f"Input strides: {inp.stride()}")
    print(f"Input shape: {inp.shape}")
    print()
    
    # Warmup
    print("Warming up (forward + backward)...")
    for _ in range(10):
        with fp8_autocast(enabled=True):
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

    for _ in range(num_iters):
        with fp8_autocast(enabled=True):
            out = grouped_linear(inp, m_splits)
        loss = out.sum()
        loss.backward()
        # Zero gradients
        grouped_linear.zero_grad()
        inp.grad = None
    torch.cuda.synchronize()
    
    # Summary
    print(f"\nBenchmark Summary:")
    print(f"  Iterations profiled: {num_iters}")
    print(f"  Forward + Backward pass per iteration")
    print(f"  Output shape: {out.shape}")
    print(f"  Output dtype: {out.dtype}")

if __name__ == "__main__":
    benchmark_grouped_linear()
