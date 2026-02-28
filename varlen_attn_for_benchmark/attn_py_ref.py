import numpy as np
from enum import Enum

class CausalMaskType(Enum):
    DISABLE = 0
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2

def attn_forward(Q, K, V, dropout_mask, dropout_p, batch, head_num, q_seq, 
                 max_kv_seq, head_dim, mask_type, cu_seqlens_kv, cu_seqlens_kv_padded):
    """
    Multi-Head Attention Forward Pass (CPU Reference Implementation)
    
    Args:
        Q: Query tensor, shape [batch, q_seq, head_num, head_dim]
        K: Key tensor, shape [total_padded_seq_kv, head_num, head_dim]
        V: Value tensor, shape [total_padded_seq_kv, head_num, head_dim]
        dropout_mask: Dropout mask, shape [batch, head_num, q_seq, max_kv_seq] or None
        dropout_p: Dropout probability
        batch: Batch size
        head_num: Number of attention heads
        q_seq: Query sequence length
        max_kv_seq: Maximum KV sequence length
        head_dim: Dimension of each attention head
        mask_type: CausalMaskType enum value
        cu_seqlens_kv: Cumulative sequence lengths for KV, shape [batch + 1]
        cu_seqlens_kv_padded: Cumulative padded sequence lengths for KV, shape [batch + 1]
    
    Returns:
        O: Output tensor, shape [batch, q_seq, head_num, head_dim]
        attn_weights: Attention weights, shape [batch, head_num, q_seq, max_kv_seq]
    """
    
    scale = 1.0 / np.sqrt(head_dim)
    dropout_scale = (1.0 / (1.0 - dropout_p)) if dropout_p > 0.0 else 1.0
    
    # Initialize output tensors
    O = np.zeros((batch, q_seq, head_num, head_dim), dtype=Q.dtype)
    attn_weights = np.zeros((batch, head_num, q_seq, max_kv_seq), dtype=Q.dtype)
    
    # Process each batch and head
    for b in range(batch):
        # Get actual sequence length for this batch
        kv_seq = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b]
        kv_offset = cu_seqlens_kv_padded[b]
        
        for h in range(head_num):
            # For each query position
            for q_idx in range(q_seq):
                # Get pointers to current Q, O, attention weights, and dropout mask
                Q_ptr = Q[b, q_idx, h, :]  # [head_dim]
                O_ptr = O[b, q_idx, h, :]  # [head_dim]
                attn_ptr = attn_weights[b, h, q_idx, :]  # [max_kv_seq]
                
                if dropout_mask is not None:
                    dropout_ptr = dropout_mask[b, h, q_idx, :]  # [max_kv_seq]
                else:
                    dropout_ptr = None
                
                # Step 1: Compute scores = Q @ K^T / sqrt(d_k)
                # Q: [head_dim], K: [kv_seq, head_dim] -> scores: [kv_seq]
                scores = np.zeros(max_kv_seq, dtype=Q.dtype)
                
                for kv_idx in range(kv_seq):
                    K_ptr = K[kv_offset + kv_idx, h, :]  # [head_dim]
                    scores[kv_idx] = np.sum(Q_ptr * K_ptr) * scale
                
                # Step 2: Apply causal mask
                if mask_type == CausalMaskType.TOP_LEFT:
                    for j in range(kv_seq):
                        if j > q_idx:
                            scores[j] = -1e9
                elif mask_type == CausalMaskType.BOTTOM_RIGHT:
                    for j in range(kv_seq):
                        if j < q_idx:
                            scores[j] = -1e9
                
                # Step 3: Softmax
                # Find max for numerical stability
                max_val = np.max(scores[:kv_seq])
                
                # Compute exp and sum
                attn_probs = np.exp(scores[:kv_seq] - max_val)
                sum_exp = np.sum(attn_probs)
                
                # Normalize
                attn_probs = attn_probs / sum_exp
                
                # Step 4: Apply dropout
                if dropout_p > 0.0 and dropout_ptr is not None:
                    for i in range(kv_seq):
                        attn_probs[i] = attn_probs[i] * dropout_ptr[i] * dropout_scale
                
                # Save attention weights
                attn_ptr[:kv_seq] = attn_probs
                
                # Step 5: Compute output = attn_probs @ V
                # attn_probs: [kv_seq], V: [kv_seq, head_dim] -> O: [head_dim]
                for d in range(head_dim):
                    sum_val = 0.0
                    for kv_idx in range(kv_seq):
                        V_ptr = V[kv_offset + kv_idx, h, d]
                        sum_val += attn_probs[kv_idx] * V_ptr
                    O[b, q_idx, h, d] = sum_val
    
    return O, attn_weights


def matmul(A, B):
    """Matrix multiplication C = A @ B"""
    return A @ B

def transpose(A):
    """Matrix transpose"""
    return A.T

def sum_last_dim(A):
    """Sum along last dimension"""
    return A.sum(axis=-1)

def attn_backward(Q, K, V, grad_O, attn_weights, dropout_mask, dropout_p,
                  batch, head_num, q_seq, max_kv_seq, head_dim, mask_type,
                  cu_seqlens_kv, cu_seqlens_kv_padded, total_padded_kv_seq):
    """
    Multi-Head Attention Backward Pass (Python Reference Implementation)
    
    Args:
        Q: Query tensor [batch * q_seq * head_num * head_dim] (flattened, q_seq=1 typically)
        K: Key tensor [total_padded_kv_seq * head_num * head_dim] (dynamic layout)
        V: Value tensor [total_padded_kv_seq * head_num * head_dim] (dynamic layout)
        grad_O: Gradient of output [batch * q_seq * head_num * head_dim] (flattened, q_seq=1 typically)
        attn_weights: Attention weights [batch * head_num * q_seq * max_kv_seq]
        dropout_mask: Dropout mask [batch * head_num * q_seq * max_kv_seq] or None
        dropout_p: Dropout probability
        batch: Batch size
        head_num: Number of attention heads
        q_seq: Query sequence length
        max_kv_seq: Maximum key/value sequence length
        head_dim: Head dimension
        mask_type: CausalMaskType enum
        cu_seqlens_kv: Cumulative sequence lengths [batch+1]
        cu_seqlens_kv_padded: Cumulative padded sequence lengths [batch+1]
        total_padded_kv_seq: Total padded key/value sequence length
    
    Returns:
        grad_Q: Gradient of Q [batch * q_seq * head_num * head_dim] (flattened, q_seq=1 typically)
        grad_K: Gradient of K [total_padded_kv_seq * head_num * head_dim]
        grad_V: Gradient of V [total_padded_kv_seq * head_num * head_dim]
    """
    
    scale = 1.0 / np.sqrt(head_dim)
    dropout_scale = 1.0 / (1.0 - dropout_p) if dropout_p > 0.0 else 1.0
    
    # Initialize gradients to zero
    grad_Q = np.zeros(batch * q_seq * head_num * head_dim, dtype=Q.dtype)
    grad_K = np.zeros(total_padded_kv_seq * head_num * head_dim, dtype=K.dtype)
    grad_V = np.zeros(total_padded_kv_seq * head_num * head_dim, dtype=V.dtype)
    
    # Process each batch and head
    for b in range(batch):
        # Get actual sequence length for this batch
        kv_seq = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b]
        
        for h in range(head_num):
            # Calculate offsets for [batch, q_seq, head_num, head_dim] layout
            offset_Q = (b * q_seq * head_num + h) * head_dim
            offset_grad_O = (b * q_seq * head_num + h) * head_dim
            offset_attn = (b * head_num + h) * q_seq * max_kv_seq
            offset_dropout = (b * head_num + h) * q_seq * max_kv_seq if dropout_mask is not None else 0
            
            # Get pointers for current batch and head
            Q_bh = Q[offset_Q:offset_Q + q_seq * head_dim].reshape(q_seq, head_dim)
            grad_O_bh = grad_O[offset_grad_O:offset_grad_O + q_seq * head_dim].reshape(q_seq, head_dim)
            attn_bh = attn_weights[offset_attn:offset_attn + q_seq * max_kv_seq].reshape(q_seq, max_kv_seq)
            dropout_bh = None if dropout_mask is None else dropout_mask[offset_dropout:offset_dropout + q_seq * max_kv_seq].reshape(q_seq, max_kv_seq)
            
            # DYNAMIC layout: use cu_seqlens_kv_padded for offset
            offset_K_base = cu_seqlens_kv_padded[b] * head_num * head_dim + h * head_dim
            kv_stride = head_num * head_dim
            
            # Copy K/V data to contiguous buffers for this batch (use actual seq length)
            K_cont = np.zeros((kv_seq, head_dim), dtype=K.dtype)
            V_cont = np.zeros((kv_seq, head_dim), dtype=V.dtype)
            for i in range(kv_seq):
                for j in range(head_dim):
                    K_cont[i, j] = K[offset_K_base + i * kv_stride + j]
                    V_cont[i, j] = V[offset_K_base + i * kv_stride + j]
            
            # Step 1: grad_V = attn_weights^T @ grad_O
            attn_T = attn_bh[:, :kv_seq].T  # Shape: [kv_seq, q_seq]
            grad_V_cont = matmul(attn_T, grad_O_bh)  # Shape: [kv_seq, head_dim]
            
            # Step 2: grad_attn = grad_O @ V^T
            V_T = V_cont.T  # Shape: [head_dim, kv_seq]
            grad_attn = matmul(grad_O_bh, V_T)  # Shape: [q_seq, kv_seq]
            
            # Step 3: Dropout backward
            if dropout_p > 0.0 and dropout_bh is not None:
                grad_attn = grad_attn * dropout_bh[:, :kv_seq] * dropout_scale
            
            # Step 4: Softmax backward
            grad_scores = grad_attn * attn_bh[:, :kv_seq]
            row_sums = sum_last_dim(grad_scores)  # Shape: [q_seq]
            
            for i in range(q_seq):
                for j in range(kv_seq):
                    grad_scores[i, j] = grad_scores[i, j] - attn_bh[i, j] * row_sums[i]
            
            # Step 5: Mask backward
            if mask_type == CausalMaskType.TOP_LEFT:
                for i in range(q_seq):
                    for j in range(kv_seq):
                        if j > i:
                            grad_scores[i, j] = 0.0
            elif mask_type == CausalMaskType.BOTTOM_RIGHT:
                for i in range(q_seq):
                    for j in range(kv_seq):
                        if j < i:
                            grad_scores[i, j] = 0.0
            
            # Step 6: grad_Q = grad_scores @ K * scale
            grad_Q_bh = matmul(grad_scores, K_cont) * scale  # Shape: [q_seq, head_dim]
            
            # Step 7: grad_K = grad_scores^T @ Q * scale
            grad_scores_T = grad_scores.T  # Shape: [kv_seq, q_seq]
            grad_K_cont = matmul(grad_scores_T, Q_bh) * scale  # Shape: [kv_seq, head_dim]
            
            # Copy results back to output arrays
            grad_Q[offset_Q:offset_Q + q_seq * head_dim] = grad_Q_bh.flatten()
            
            # Copy grad_K and grad_V back to dynamic layout
            for i in range(kv_seq):
                for j in range(head_dim):
                    grad_K[offset_K_base + i * kv_stride + j] = grad_K_cont[i, j]
                    grad_V[offset_K_base + i * kv_stride + j] = grad_V_cont[i, j]
    
    return grad_Q, grad_K, grad_V


def test_mha_backward_vs_pytorch():
    """
    Test function to compare custom MHA backward implementation with PyTorch's implementation.
    For simplicity, all kv_seq = max_kv_seq (no variable length sequences).
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("PyTorch is not installed. Please install it to run this test.")
        return
    
    # Test parameters
    batch = 256
    head_num = 8
    q_seq = 1
    max_kv_seq = 16  # For simplicity, kv_seq = max_kv_seq
    head_dim = 128
    dropout_p = 0.0  # No dropout for easier comparison
    mask_type = CausalMaskType.TOP_LEFT  # causal mask
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Testing MHA backward with:")
    print(f"  batch={batch}, head_num={head_num}, q_seq={q_seq}, max_kv_seq={max_kv_seq}, head_dim={head_dim}")
    print(f"  mask_type={mask_type}, dropout_p={dropout_p}")
    print()
    
    # Generate random inputs (NumPy)
    Q_np = np.random.randn(batch, q_seq, head_num, head_dim).astype(np.float32)
    # Since kv_seq = max_kv_seq for all batches, total_padded_kv_seq = batch * max_kv_seq
    total_padded_kv_seq = batch * max_kv_seq
    K_np = np.random.randn(total_padded_kv_seq, head_num, head_dim).astype(np.float32)
    V_np = np.random.randn(total_padded_kv_seq, head_num, head_dim).astype(np.float32)
    grad_O_np = np.random.randn(batch, q_seq, head_num, head_dim).astype(np.float32)
    
    # All sequences have same length = max_kv_seq
    cu_seqlens_kv = np.array([i * max_kv_seq for i in range(batch + 1)], dtype=np.int32)
    cu_seqlens_kv_padded = cu_seqlens_kv.copy()
    
    # Convert to PyTorch tensors
    Q_torch = torch.from_numpy(Q_np).requires_grad_(True)
    # Reshape K and V for PyTorch: [total_padded_kv_seq, head_num, head_dim] -> [batch, max_kv_seq, head_num, head_dim]
    K_reshaped = K_np.reshape(batch, max_kv_seq, head_num, head_dim)
    V_reshaped = V_np.reshape(batch, max_kv_seq, head_num, head_dim)
    K_torch = torch.from_numpy(K_reshaped).requires_grad_(True)
    V_torch = torch.from_numpy(V_reshaped).requires_grad_(True)
    grad_O_torch = torch.from_numpy(grad_O_np)
    
    # ========== Custom Implementation ==========
    print("Running custom implementation...")
    
    # Forward pass
    O_custom, attn_weights = attn_forward(
        Q_np, K_np, V_np,
        dropout_mask=None,
        dropout_p=dropout_p,
        batch=batch,
        head_num=head_num,
        q_seq=q_seq,
        max_kv_seq=max_kv_seq,
        head_dim=head_dim,
        mask_type=mask_type,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded
    )
    
    # Backward pass
    # Q and grad_O are in [batch, q_seq, head_num, head_dim] layout (q_seq=1)
    # attn_weights is [batch, head_num, q_seq, max_kv_seq] from forward
    # No transpose needed as attn_backward expects [batch, q_seq, head_num, head_dim]
    
    Q_flat = Q_np.reshape(-1)
    K_flat = K_np.reshape(-1)
    V_flat = V_np.reshape(-1)
    grad_O_flat = grad_O_np.reshape(-1)
    attn_weights_flat = attn_weights.reshape(-1)
    
    grad_Q_flat, grad_K_flat, grad_V_flat = attn_backward(
        Q_flat, K_flat, V_flat,
        grad_O_flat, attn_weights_flat,
        dropout_mask=None,
        dropout_p=dropout_p,
        batch=batch,
        head_num=head_num,
        q_seq=q_seq,
        max_kv_seq=max_kv_seq,
        head_dim=head_dim,
        mask_type=mask_type,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        total_padded_kv_seq=total_padded_kv_seq
    )
    
    # Reshape to original layout [batch, q_seq, head_num, head_dim]
    grad_Q_custom = grad_Q_flat.reshape(batch, q_seq, head_num, head_dim)
    grad_K_custom = grad_K_flat.reshape(batch, max_kv_seq, head_num, head_dim)
    grad_V_custom = grad_V_flat.reshape(batch, max_kv_seq, head_num, head_dim)
    
    print("Custom implementation completed.")
    
    # ========== PyTorch Implementation ==========
    print("Running PyTorch implementation...")
    
    # Reshape for PyTorch SDPA: [batch, head_num, seq, head_dim]
    Q_pt = Q_torch.permute(0, 2, 1, 3)  # [batch, head_num, q_seq, head_dim]
    K_pt = K_torch.permute(0, 2, 1, 3)  # [batch, head_num, max_kv_seq, head_dim]
    V_pt = V_torch.permute(0, 2, 1, 3)  # [batch, head_num, max_kv_seq, head_dim]
    
    # Create attention mask based on mask_type
    attn_mask_pt = None
    is_causal = False
    
    if mask_type == CausalMaskType.TOP_LEFT:
        # expects float mask with -inf for masked positions, or bool mask
        attn_mask_pt = torch.zeros(q_seq, max_kv_seq, dtype=torch.float32)
        for i in range(q_seq):
            for j in range(max_kv_seq):
                if j > i:
                    attn_mask_pt[i, j] = float('-inf')
                    
    elif mask_type == CausalMaskType.BOTTOM_RIGHT:
        # Bottom-right causal mask: mask positions where j < i
        attn_mask_pt = torch.zeros(q_seq, max_kv_seq, dtype=torch.float32)
        for i in range(q_seq):
            for j in range(max_kv_seq):
                if j < i:
                    attn_mask_pt[i, j] = float('-inf')
    
    # Compute attention
    O_pytorch = F.scaled_dot_product_attention(
        Q_pt, K_pt, V_pt,
        attn_mask=attn_mask_pt,
        dropout_p=dropout_p,
        is_causal=is_causal
    )
    
    # Backward pass
    O_pytorch.backward(grad_O_torch.permute(0, 2, 1, 3))
    
    # Get gradients
    grad_Q_pytorch = Q_torch.grad.numpy()
    grad_K_pytorch = K_torch.grad.numpy()
    grad_V_pytorch = V_torch.grad.numpy()
    
    # Permute PyTorch output back to match custom format
    O_pytorch_np = O_pytorch.permute(0, 2, 1, 3).detach().numpy()
    
    print("PyTorch implementation completed.")
    print()
    
    # ========== Compare Results ==========
    print("="*60)
    print("Comparison Results:")
    print("="*60)
    
    # Compare forward outputs
    O_diff = np.abs(O_custom - O_pytorch_np)
    O_max_diff = np.max(O_diff)
    O_mean_diff = np.mean(O_diff)
    O_rel_error = np.max(O_diff / (np.abs(O_pytorch_np) + 1e-8))
    
    print("\nForward Pass (Output O):")
    print(f"  Max absolute diff: {O_max_diff:.6e}")
    print(f"  Mean absolute diff: {O_mean_diff:.6e}")
    print(f"  Max relative error: {O_rel_error:.6e}")
    print(f"  Are they close? {np.allclose(O_custom, O_pytorch_np, rtol=1e-4, atol=1e-5)}")
    
    # Compare backward gradients
    grad_Q_diff = np.abs(grad_Q_custom - grad_Q_pytorch)
    grad_Q_max_diff = np.max(grad_Q_diff)
    grad_Q_mean_diff = np.mean(grad_Q_diff)
    grad_Q_rel_error = np.max(grad_Q_diff / (np.abs(grad_Q_pytorch) + 1e-8))
    
    print("\nBackward Pass (grad_Q):")
    print(f"  Max absolute diff: {grad_Q_max_diff:.6e}")
    print(f"  Mean absolute diff: {grad_Q_mean_diff:.6e}")
    print(f"  Max relative error: {grad_Q_rel_error:.6e}")
    print(f"  Are they close? {np.allclose(grad_Q_custom, grad_Q_pytorch, rtol=1e-4, atol=1e-5)}")
    
    grad_K_diff = np.abs(grad_K_custom - grad_K_pytorch)
    grad_K_max_diff = np.max(grad_K_diff)
    grad_K_mean_diff = np.mean(grad_K_diff)
    grad_K_rel_error = np.max(grad_K_diff / (np.abs(grad_K_pytorch) + 1e-8))
    
    print("\nBackward Pass (grad_K):")
    print(f"  Max absolute diff: {grad_K_max_diff:.6e}")
    print(f"  Mean absolute diff: {grad_K_mean_diff:.6e}")
    print(f"  Max relative error: {grad_K_rel_error:.6e}")
    print(f"  Are they close? {np.allclose(grad_K_custom, grad_K_pytorch, rtol=1e-4, atol=1e-5)}")
    
    grad_V_diff = np.abs(grad_V_custom - grad_V_pytorch)
    grad_V_max_diff = np.max(grad_V_diff)
    grad_V_mean_diff = np.mean(grad_V_diff)
    grad_V_rel_error = np.max(grad_V_diff / (np.abs(grad_V_pytorch) + 1e-8))
    
    print("\nBackward Pass (grad_V):")
    print(f"  Max absolute diff: {grad_V_max_diff:.6e}")
    print(f"  Mean absolute diff: {grad_V_mean_diff:.6e}")
    print(f"  Max relative error: {grad_V_rel_error:.6e}")
    print(f"  Are they close? {np.allclose(grad_V_custom, grad_V_pytorch, rtol=1e-4, atol=1e-5)}")
    
    # Overall result
    print("\n" + "="*60)
    all_close = (
        np.allclose(O_custom, O_pytorch_np, rtol=1e-4, atol=1e-5) and
        np.allclose(grad_Q_custom, grad_Q_pytorch, rtol=1e-4, atol=1e-5) and
        np.allclose(grad_K_custom, grad_K_pytorch, rtol=1e-4, atol=1e-5) and
        np.allclose(grad_V_custom, grad_V_pytorch, rtol=1e-4, atol=1e-5)
    )
    
    if all_close:
        print("✓ SUCCESS: Custom implementation matches PyTorch!")
    else:
        print("✗ FAILURE: Custom implementation differs from PyTorch.")
    print("="*60)


if __name__ == "__main__":
    test_mha_backward_vs_pytorch()