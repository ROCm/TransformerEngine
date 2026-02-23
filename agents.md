# Specialized Unfused Attention Backend Integration

## Task Overview

### Problem Statement
The customer (SciForum) has a special cross-attention use case where:
- **Q sequence length is always 1** (seq_q = 1)
- **KV sequence length varies from 2 to 16** (max_seq_kv <= 16)
- **Batch size is very large** (~30K)
- **Layout**: THD (ragged/packed format)
- **Data type**: BF16

The current fused attention kernels (Flash Attention, CK fused attention) are optimized for large `seq_q * seq_kv` products. When `seq_q = 1`, these kernels become inefficient because they're designed to read continuous segments in the sequence dimension, which creates unnecessary overhead for this small case.

**Performance Issue**: Current fused attention is ~20x slower than JAX unfused attention for this specific configuration.

**Customer Requirement**: They want to use the same input/output format as the current fused attention (no conversion from cumulative sequence format to square masking), but with performance similar to unfused attention.

### Solution
The CK team has provided HIP kernels that implement unfused attention specifically optimized for this case:
- Forward: 3-step process (compute scores, apply mask+softmax, compute output)
- Backward: 4-step process (standard unfused attention backward)
- These kernels store **attention weights** instead of softmax_LSE (which is what traditional fused attention stores)

## Files in `varlen_attn/` Directory

### `attn_fwd.cpp` - Forward Pass Implementation

**Key Components:**

1. **Template Configuration** (`FmhaKernelConfig`):
   - Template parameters: `BS`, `HEAD_NUM`, `SEQ_Q`, `MAX_SEQ_KV`, `HEAD_DIM`, `STEP2_BLOCK_SIZE`, `ENABLE_DROPOUT_MASK`, `MASK_TYPE`
   - **Constraint**: `SEQ_Q` must be 1 (enforced via `static_assert`)

2. **Three-Step Kernel Process**:
   - **Kernel 1** (`compute_scores_kernel`): Computes `Q @ K^T * scale`
     - Each thread handles one Q token
     - Uses block-based computation (block_k = 64)
     - Supports BF16 and float types
     - Output: scores tensor of shape `[batch * head_num * seq_q, max_seq_kv]`
   
   - **Kernel 2** (`apply_mask_and_softmax_kernel`): Applies mask and softmax
     - Handles causal masks (TOP_LEFT, BOTTOM_RIGHT, DISABLE)
     - Applies padding mask
     - Numerically stable softmax (max reduction, then exp and sum)
     - Applies dropout if enabled
     - **Output**: Attention weights (stored in workspace, reused in backward)
   
   - **Kernel 3** (`compute_output_kernel`): Computes `attn_weights @ V`
     - Uses warp-level parallelism
     - Output: O tensor of shape `[batch, seq_q, head_num, head_dim]`

3. **Host API** (`AttnForwardKernelLauncher`):
   - `calc_workspace_size()`: Returns `bs * head_num * seq_q * max_seq_kv * sizeof(T)`
   - `run_attn_fwd_kernel()`: Launches the three kernels in sequence
   - **Inputs**: Q, K, V, dropout_mask, dropout_p, scale, cu_seqlens_kv, cu_seqlens_kv_padded
   - **Outputs**: O, workspace (contains attention weights)

4. **Key Features**:
   - Supports variable sequence lengths via `cu_seqlens_kv` and `cu_seqlens_kv_padded`
   - Layout: THD format (ragged/packed)
   - Workspace stores attention weights: `[batch, head_num, seq_q, max_seq_kv]`

### `attn_bwd.cpp` - Backward Pass Implementation

**Key Components:**

1. **Four-Step Kernel Process**:
   - **Kernel 1** (`compute_grad_v_kernel`): `grad_V = attn_weights^T @ grad_O`
   - **Kernel 2** (`compute_grad_attn_kernel`): `grad_attn = grad_O @ V^T`
   - **Kernel 3** (`softmax_backward_kernel`): Applies softmax backward and dropout backward
   - **Kernel 4** (`compute_grad_qk_kernel`): Computes `grad_Q` and `grad_K`

2. **Host API** (`AttnBackwardKernelLauncher`):
   - `calc_workspace_size()`: Same as forward (`bs * head_num * seq_q * max_seq_kv * sizeof(T)`)
   - `run_attn_bwd_kernel()`: Launches the four kernels
   - **Inputs**: Q, K, V, grad_O, attn_weights (from forward), dropout_mask, dropout_p, scale, cu_seqlens_kv, cu_seqlens_kv_padded
   - **Outputs**: grad_Q, grad_K, grad_V, workspace (temporary)

3. **Key Features**:
   - Requires attention weights from forward pass (stored in workspace/softmax_LSE buffer)
   - Same variable sequence length support as forward

## Current CK Flow and Requirements

### Backend Selection Flow

1. **Entry Point**: `nvte_fused_attn_fwd()` or `nvte_fused_attn_fwd_kvpacked()` in `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`

2. **Backend Selection** (`nvte_get_fused_attn_backend()`):
   - Checks environment variables: `NVTE_FUSED_ATTN`, `NVTE_FUSED_ATTN_CK`, `NVTE_FUSED_ATTN_AOTRITON`
   - First checks if CK backend is supported via `is_ck_backend_supported()`
   - Then checks AOTriton backend
   - Returns `NVTE_Fused_Attn_Backend::NVTE_CK`, `NVTE_AOTriton`, or `NVTE_No_Backend`

3. **Current Backend Enum** (ROCm):
   ```cpp
   enum NVTE_Fused_Attn_Backend {
       NVTE_No_Backend = -1,
       NVTE_AOTriton = 0,
       NVTE_CK = 1,
   };
   ```

4. **CK Forward Implementation** (`fused_attn_ck_fwd()` in `fused_attn_ck.cpp`):
   - Handles QKV unpacked format (separate Q, K, V tensors)
   - Supports THD layout (ragged format)
   - Uses `cu_seqlens_q` and `cu_seqlens_kv` for variable sequences
   - **Storage**: Stores `softmax_LSE` in `Aux_CTX_Tensors` (for backward pass)
   - Shape: `[max_tokens_q, h_q, 1]` for ragged, `[b, h_q, max_seqlen_q, 1]` for regular

### Key Differences: CK vs. New Unfused Kernel

| Aspect | CK Fused Attention | New Unfused Kernel |
|--------|-------------------|-------------------|
| **Storage** | `softmax_LSE` (float32) | `attention_weights` (same dtype as QKV) |
| **Storage Shape** | `[max_tokens_q, h_q, 1]` or `[b, h_q, max_seqlen_q, 1]` | `[batch, head_num, seq_q, max_seq_kv]` |
| **Workspace** | Used for temporary computation | Stores attention weights |
| **Backward Input** | Requires `softmax_LSE` | Requires `attention_weights` |
| **Optimization** | Large seq_q * seq_kv | Small seq_q=1, seq_kv<16 |

### Runtime Sequence Length Detection

For THD layout, we need to determine `max_seqlen_q` and `max_seqlen_kv` at runtime:
- **Current approach** (`get_runtime_max_seqlen()` in `ck_fused_attn_utils.cpp`):
  - Launches a kernel to compute max sequence length
  - Uses `hipMemcpyAsync` + `hipStreamSynchronize` (host-device sync required)
  - Reads from `cu_seqlens` or `cu_seqlens_padded` device pointers

**For our specialized backend**:
- Need to check: `max_seqlen_q == 1` and `max_seqlen_kv <= 16`
- Must do this check before allocating buffers and selecting backend
- Will require similar host-device synchronization

## Implementation Plan

### Phase 1: Add New Backend Enum

**Files to modify:**
1. `transformer_engine/common/include/transformer_engine/fused_attn.h`
   - Add `NVTE_Unfused_SmallSeq = 2` to the ROCm enum

2. `transformer_engine/common/util/pybind_helper.h`
   - Add `.value("NVTE_Unfused_SmallSeq", NVTE_Fused_Attn_Backend::NVTE_Unfused_SmallSeq)` to the ROCm section

### Phase 2: Create Backend Support Check Function

**New file**: `transformer_engine/common/fused_attn_rocm/fused_attn_unfused_smallseq.cpp` (or integrate into existing file)

**Functions to implement:**
1. `is_unfused_smallseq_backend_supported()`:
   - Check if `max_seqlen_q == 1`
   - Check if `max_seqlen_kv < 16`
   - Check layout is THD
   - Check data type is BF16 or FP16
   - Check other constraints (no bias, specific mask types, etc.)

2. `get_runtime_max_seqlen_q_kv()`:
   - Similar to `get_runtime_max_seqlen()` but returns both Q and KV max lengths
   - Uses host-device synchronization
   - Called before backend selection if layout is THD

### Phase 3: Integrate HIP Kernels

**New file**: `transformer_engine/common/fused_attn_rocm/fused_attn_unfused_smallseq.cpp`

**Key functions:**
1. `fused_attn_unfused_smallseq_fwd()`:
   - Extract Q, K, V from input tensors
   - Calculate workspace size (using formula from `AttnForwardKernelLauncher::calc_workspace_size()`)
   - Allocate workspace (or use provided workspace)
   - Call HIP kernel launcher (adapt from `AttnForwardKernelLauncher::run_attn_fwd_kernel()`)
   - **Store attention weights in softmax_LSE buffer** (hack: reuse the storage)

2. `fused_attn_unfused_smallseq_bwd()`:
   - Extract Q, K, V, grad_O from input tensors
   - Extract attention weights from softmax_LSE buffer
   - Allocate workspace
   - Call backward HIP kernel launcher
   - Output grad_Q, grad_K, grad_V

**HIP Kernel Integration:**
- Copy kernels from `varlen_attn/attn_fwd.cpp` and `varlen_attn/attn_bwd.cpp`
- Adapt template parameters to runtime values (or use template specialization)
- Handle device pointer setup
- Handle stream synchronization

### Phase 4: Update Backend Selection Logic

**File**: `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`

**Modify `nvte_get_fused_attn_backend()`:**
- Add check for unfused_smallseq backend **before** CK backend check
- Priority order: Unfused_SmallSeq > CK > AOTriton
- Only check if layout is THD and we can determine sequence lengths

**Modify forward/backward functions:**
- Add `else if (backend == NVTE_Unfused_SmallSeq)` branches
- Call the new specialized functions

### Phase 5: Handle Workspace and Buffer Allocation

**Key Challenge**: Reusing `softmax_LSE` storage for attention weights

**Solution**:
1. In forward pass:
   - Calculate attention weights size: `batch * head_num * seq_q * max_seq_kv * sizeof(dtype)`
   - Calculate softmax_LSE size: `max_tokens_q * h_q * 1 * sizeof(float32)`
   - Ensure attention weights fit (they should, since seq_q=1 and max_seq_kv<16)
   - Store attention weights in the softmax_LSE buffer (cast/reinterpret as needed)

2. In backward pass:
   - Read attention weights from softmax_LSE buffer
   - Cast back to original dtype

**Files to modify:**
- `fused_attn_ck.cpp`: May need to adjust buffer allocation logic
- New unfused backend file: Implement the storage hack

### Phase 6: Add Tests

**File**: `tests/jax/test_fused_attn.py`

**Add test case:**
- Config: `b=large (e.g., 1000)`, `s_q=1`, `s_kv=8`, `h_q=32`, `h_kv=32`, `d_qk=128`, `d_v=128`, `dtype=bfloat16`
- Layout: THD format
- Verify correctness against reference
- Verify backend selection

## Detailed Todo List

### 1. Backend Enum and Registration
- [ ] Add `NVTE_Unfused_SmallSeq` to enum in `fused_attn.h`
- [ ] Update pybind registration in `pybind_helper.h`
- [ ] Verify enum is accessible from Python/JAX

### 2. Backend Support Check
- [ ] Implement `is_unfused_smallseq_backend_supported()` function
- [ ] Implement `get_runtime_max_seqlen_q_kv()` helper function
- [ ] Add logic to check sequence lengths at runtime for THD layout
- [ ] Add constraints checking (dtype, layout, bias, mask, etc.)

### 3. HIP Kernel Integration
- [ ] Create new file `fused_attn_unfused_smallseq.cpp` and header
- [ ] Copy and adapt forward kernels from `varlen_attn/attn_fwd.cpp`
- [ ] Copy and adapt backward kernels from `varlen_attn/attn_bwd.cpp`
- [ ] Implement template instantiation or runtime kernel selection
- [ ] Handle device memory allocation and pointer setup
- [ ] Implement workspace size calculation
- [ ] Implement forward launcher function
- [ ] Implement backward launcher function

### 4. Storage Hack (Attention Weights in softmax_LSE)
- [ ] Implement attention weights storage in forward pass
- [ ] Store in softmax_LSE buffer (reinterpret cast)
- [ ] Implement reading from softmax_LSE buffer in backward pass
- [ ] Handle dtype conversion (float32 softmax_LSE buffer vs. BF16 attention weights)
- [ ] Verify memory layout compatibility

### 5. Backend Selection Integration
- [ ] Update `nvte_get_fused_attn_backend()` to check unfused_smallseq first
- [ ] Add backend check in forward functions (`nvte_fused_attn_fwd`, `nvte_fused_attn_fwd_kvpacked`)
- [ ] Add backend check in backward functions
- [ ] Ensure proper fallback to CK or AOTriton if conditions not met

### 6. Workspace and Buffer Management
- [ ] Calculate workspace size for forward pass
- [ ] Calculate workspace size for backward pass
- [ ] Integrate with existing workspace allocation logic
- [ ] Handle Aux_CTX_Tensors setup (softmax_LSE buffer)

### 7. Testing
- [ ] Add pytest test case in `test_fused_attn.py`
- [ ] Test with seq_q=1, seq_kv=2-16 16 is also allowed in the  range
- [ ] Test with large batch size (~1000+)
- [ ] Test forward pass correctness
- [ ] Test backward pass correctness
- [ ] Test backend selection logic
- [ ] Verify performance improvement

### 8. Error Handling and Edge Cases
- [ ] Handle cases where sequence lengths don't match constraints
- [ ] Handle memory allocation failures
- [ ] Add proper error messages
- [ ] Handle unsupported configurations gracefully

### 9. Documentation
- [ ] Document the new backend in code comments
- [ ] Document constraints and limitations
- [ ] Update any relevant documentation files

## Implementation Notes

### Template vs. Runtime Kernel Selection

The HIP kernels use template metaprogramming with compile-time constants. We have two options:

1. **Template Specialization**: Create template instantiations for common configurations
2. **Runtime Kernel Selection**: Use runtime values and select appropriate kernel variant

**Recommendation**: Start with runtime kernel selection for flexibility, can optimize later with templates.

### Memory Layout Considerations

- **Attention Weights Shape**: `[batch, head_num, seq_q, max_seq_kv]` = `[b, h, 1, max_seq_kv]`
- **softmax_LSE Shape**: `[max_tokens_q, h_q, 1]` for ragged format
- **Compatibility**: Need to ensure these are compatible for storage hack

### Performance Optimization

- The kernels are already optimized for the specific case (seq_q=1, small seq_kv)
- Focus on integration correctness first
- Performance tuning can be done later if needed

## Questions and Considerations

1. **Template Instantiation**: Should we use template specialization or runtime kernel selection?
2. **Memory Layout**: Verify attention weights and softmax_LSE buffer layouts are compatible
3. **Dtype Handling**: softmax_LSE is float32, attention weights are BF16/FP16 - need proper casting
4. **Backward Compatibility**: Ensure existing CK flow is not affected
5. **Testing Strategy**: How to test with very large batch sizes efficiently?
