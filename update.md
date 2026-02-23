# Unfused_SmallSeq Backend Implementation - Issue Tracking

## Test Configuration
- Test: `tests/jax/test_fused_attn.py::test_unfused_smallseq_backend`
- Configs:
  1. `b=30720, s_q=1, s_kv=2, h_q=16, h_kv=16, d_qk=128, d_v=128, dtype=BF16`
  2. `b=30720, s_q=1, s_kv=16, h_q=16, h_kv=16, d_qk=128, d_v=128, dtype=BF16`

## Issues and Fixes

### Issue #1: BF16 Memory Access Pattern
**Error**: `hipError_t(901)` - Illegal memory access
**Root Cause**: BF16 requires 8 elements per uint4, but code was using 4 elements per uint4
**Fix**: Added `if constexpr(std::is_same<T, hip_bfloat16>::value)` to handle BF16 separately in `compute_scores_kernel_runtime`
**Status**: Fixed in code, pending rebuild/test

### Issue #2: Runtime Sequence Length Synchronization
**Error**: Stream capture issues with `hipStreamSynchronize`
**Root Cause**: JAX uses stream capture, and synchronization breaks it
**Fix**: User reverted to use `hipStreamSynchronize` - need to verify this works
**Status**: User change applied

### Issue #3: Compute Output Kernel Missing Tasks Per Block
**Error**: `hipError_t(901/904)` - Illegal memory access
**Root Cause**: Original template uses `TASKS_PER_BLOCK = 2` but runtime version only processed 1 task per block, causing incorrect grid calculation and incomplete work processing
**Fix**: 
- Added `tasks_per_block = 2` constant to match original
- Added loop `for(int task = 0; task < tasks_per_block; task++)` in kernel
- Fixed grid calculation: `(merge_bs / process_head_per_warp + tasks_per_block - 1) / tasks_per_block`
**Status**: Fixed in code, pending rebuild/test

---

## Iteration Log

### Iteration 1: Initial Debug Setup and Fixes
**Date**: Current
**Actions**:
- Created update.md for tracking
- Added debug logging statements
- Fixed compute_output kernel to process 2 tasks per block (matching original template)
- Fixed grid calculation for kernel 3
- Fixed undeclared variable `nvte_log_unfused_config` in forward function
- Removed build artifacts
- Build: SUCCESS
- Next: Run tests with timeout

### Issue #4: Undeclared Variable in Forward Function
**Error**: `use of undeclared identifier 'nvte_log_unfused_config'`
**Root Cause**: Variable was declared in `is_unfused_smallseq_backend_supported` but used in `fused_attn_unfused_smallseq_fwd`
**Fix**: Added variable declaration in forward function scope
**Status**: Fixed, build successful

### Issue #5: Persistent Illegal Memory Access (hipError_t 901/904)
**Error**: `hipError_t(901)` and `hipError_t(904)` - Illegal memory access
**Root Cause**: Still investigating - likely in kernel memory access patterns
**Observations**:
- BF16 handling fixed in compute_scores kernel
- compute_output kernel uses `dwordx4_load_elt = 16/sizeof(T)` which is 8 for BF16
- `block_k = 8`, so `block_k / dwordx4_load_elt = 1` for BF16
- Need to verify memory access patterns match original template exactly
**Status**: Investigating - checking compute_output kernel implementation

### Issue #6: Runtime Sequence Length Check Returning 0
**Error**: "max_seqlen_q must be 1, got 0" - Runtime sequence length check returns 0
**Root Cause**: Runtime max seqlen kernels may not be executing correctly, or cu_seqlens not set up properly
**Observations**:
- Debug output shows function called 3 times with pointers=0 (workspace size calculation)
- Batch size shows as 61440 instead of 30720 (doubled)
- Runtime check returns max_seqlen_q=0, causing backend check to fail
**Fix**: Added early return for workspace size calculation to avoid executing kernels with null pointers
**Status**: Fixed early return, but runtime seqlen check still needs investigation

---

## Summary of Current Status

### Completed
1. ✅ Added `NVTE_Unfused_SmallSeq` backend enum
2. ✅ Implemented support check function
3. ✅ Integrated backend selection logic
4. ✅ Added forward kernel implementations (3 kernels)
5. ✅ Fixed BF16 handling in compute_scores kernel
6. ✅ Fixed compute_output kernel to process 2 tasks per block
7. ✅ Added workspace size calculations
8. ✅ Integrated with JAX Python bindings
9. ✅ Added test cases
10. ✅ Fixed compilation errors

### Remaining Issues
1. ❌ **Illegal Memory Access (hipError_t 901/904)**: Kernels are crashing with illegal memory access
   - Likely causes:
     - Incorrect tensor indexing for THD layout
     - Incorrect cu_seqlens usage
     - Buffer overruns
     - Misaligned memory access
   - **Recommendation**: Use `rocgdb` or `rocprof` to debug exact memory access causing the issue

### Next Steps
1. Debug kernel memory access patterns using ROCm debugging tools
2. Verify tensor layouts match expected THD format
3. Check cu_seqlens values are correct
4. Verify kernel grid/block dimensions are correct
5. Add more detailed error checking in kernels

### Test Status
- **Build**: ✅ SUCCESS
- **Backend Selection**: ✅ PASSES
- **Forward Pass Execution**: ❌ FAILS (hipError_t 901/904)
