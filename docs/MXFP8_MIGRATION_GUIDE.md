# MXFP8 Implementation - Migration Guide (MI300 → MI350)

This guide explains how to migrate the MXFP8 Triton GEMM implementation from the MI300 development machine to MI350 for testing.

## Quick Summary

All implementation work is complete and committed to git. You just need to:
1. Push the branch from MI300
2. Clone and test on MI350

## Step 1: Push from MI300 (Current Machine)

```bash
cd /workspace/TransformerEngine
git push docs triton_gemm_mxfp8
```

**Current branch:** `triton_gemm_mxfp8`
**Commits:**
- `7f71fe05` - MXFP8 implementation (wrapper + kernel)
- `e65e8cb6` - MXFP8 test suite
- `29d91eb6` - Implementation plan documentation

## Step 2: Clone and Setup on MI350

```bash
# Clone the repository
git clone https://github.com/wenchenvincent/TransformerEngine-private.git
cd TransformerEngine-private

# Checkout the MXFP8 branch
git checkout triton_gemm_mxfp8

# Install in-place (editable mode)
pip install -e .

# Verify installation
python -c "from transformer_engine.pytorch.gemm_triton import mxfp8_matmul; print('✓ MXFP8 imported successfully')"
```

## Step 3: Run Tests on MI350

### Basic import and wrapper tests
```bash
cd /workspace/TransformerEngine-private
python tests/pytorch/mxfp8/test_mxfp8_gemm_basic.py
```

### Direct kernel test with simulated data
```bash
python tests/pytorch/mxfp8/test_mxfp8_kernel_direct.py
```

### Run all tests with pytest
```bash
pytest tests/pytorch/mxfp8/ -v
```

## What Was Implemented

### Code Changes (1 file modified)
- `transformer_engine/pytorch/gemm_triton.py`
  - Added MXFP8TensorWrapper class (~146 lines)
  - Added mxfp8_matmul_kernel() Triton kernel (~150 lines)
  - Added mxfp8_matmul() wrapper function
  - Updated te_generic_gemm_triton() for MXFP8 detection and dispatch
  - Total: ~410 insertions, 19 deletions

### Test Suite (4 new files)
- `tests/pytorch/mxfp8/test_mxfp8_gemm_basic.py` - Import and wrapper tests
- `tests/pytorch/mxfp8/test_mxfp8_kernel_direct.py` - Direct kernel test
- `tests/pytorch/mxfp8/README.md` - Test documentation
- `tests/pytorch/mxfp8/__init__.py` - Package init

### Documentation (1 new file)
- `docs/mxfp8_implementation_plan.md` - Full implementation plan with:
  - C++ verification details
  - Architecture and design decisions
  - Kernel implementation specifications
  - Testing strategy

## Implementation Highlights

### MXFP8TensorWrapper
- Extracts rowwise/columnwise data and E8M0 scales from MXFP8Tensor
- Mimics C++ `NVTETensorFromMXFP8Tensor()` behavior
- Handles data/scale selection based on transpose flags

### mxfp8_matmul_kernel()
- Uses Triton's `tl.dot_scaled()` for block-scaled FP8 matmul
- E8M0 to FP32 conversion: `scale = 2^(biased_exponent - 127)`
- VEC_SIZE = 32 (MXFP8_BLOCK_SCALING_SIZE)
- Autotune configs optimized for MXFP8 block constraints

### Design Decisions
- **Separate kernel**: MXFP8 kernel is separate from regular FP8 (not merged)
  - Rationale: Different operations (`tl.dot()` vs `tl.dot_scaled()`), independent autotuning
- **Dual storage**: MXFP8Tensor keeps both rowwise and columnwise copies
  - Rationale: MXFP8 cannot be transposed after quantization without requantization
- **DEFAULT epilogue only**: Initially supports only DEFAULT epilogue
  - Future: BIAS and BGRADB epilogues can be added later

## Verification Status

### Static Verification ✓ (Completed on MI300)
All 10 checks passed:
- ✓ MXFP8_BLOCK_SCALING_SIZE import
- ✓ MXFP8TensorWrapper class
- ✓ get_data_and_scale_for_gemm method
- ✓ MXFP8 detection in te_generic_gemm_triton
- ✓ input_mxfp8 flag
- ✓ Dispatch to mxfp8_matmul
- ✓ mxfp8_matmul_kernel definition
- ✓ tl.dot_scaled usage
- ✓ E8M0 to FP32 conversion
- ✓ te_dtype_to_triton_format function

### Runtime Testing ⏳ (Pending on MI350)
- Test environment issues on MI300 (CUDA initialization timeouts)
- Requires MI350 hardware for proper MXFP8 workload support

## Expected Test Results on MI350

### test_mxfp8_gemm_basic.py
```
✓ Successfully imported MXFP8 classes
✓ MXFP8TensorWrapper created for regular tensor
✓ Created test tensors: A=(128, 512), B=(512, 256)
✓ Computed FP32 reference: C=(128, 256)
BASIC TESTS PASSED!
```

### test_mxfp8_kernel_direct.py
```
✓ Imports successful
  MXFP8_BLOCK_SCALING_SIZE = 32
✓ Created test tensors:
  A_fp8: torch.Size([128, 512]), dtype=torch.uint8
  A_scale: torch.Size([128, 16]), dtype=torch.uint8
  B_fp8: torch.Size([512, 256]), dtype=torch.uint8
  B_scale: torch.Size([16, 256]), dtype=torch.uint8
  C: torch.Size([128, 256]), dtype=torch.float32
✓ mxfp8_matmul executed without errors!
✓ Output contains non-zero values (kernel produced results)
MXFP8 KERNEL TEST PASSED!
```

## Troubleshooting

### Import errors
```python
# Verify TE installation
python -c "import transformer_engine_torch as tex; print(tex.__version__)"

# Check MXFP8 constant
python -c "from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE; print(MXFP8_BLOCK_SCALING_SIZE)"
```

### CUDA/GPU errors
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify MI350 (gfx95x)
rocm-smi
```

### Test failures
```bash
# Run with verbose output
pytest tests/pytorch/mxfp8/test_mxfp8_kernel_direct.py -v -s

# Run with debugging
python -m pdb tests/pytorch/mxfp8/test_mxfp8_kernel_direct.py
```

## Claude Code Session Context (Optional)

If you want to preserve the full Claude Code conversation context:

### Plan File (Already in repo)
- **Location:** `docs/mxfp8_implementation_plan.md`
- **Content:** Full implementation plan with C++ verification, kernel specs, testing strategy

### Conversation Transcript (Optional)
The full conversation transcript is stored locally on MI300 at:
```
/root/.claude/projects/-workspace-TransformerEngine/58a3da94-8ffe-496a-8b6f-06aa2831d3df.jsonl
```

To preserve it (optional):
```bash
# On MI300
cp /root/.claude/projects/-workspace-TransformerEngine/58a3da94-8ffe-496a-8b6f-06aa2831d3df.jsonl \
   /workspace/TransformerEngine/docs/mxfp8_claude_session.jsonl

# Add to git
git add docs/mxfp8_claude_session.jsonl
git commit -m "Add Claude Code session transcript for MXFP8 implementation"
git push docs triton_gemm_mxfp8
```

**Note:** This is optional - all critical information is already in the plan document.

### Starting a New Claude Code Session on MI350

You don't need to transfer the Claude Code session to continue development. Simply:

1. Start Claude Code on MI350 in the repository directory
2. Reference `docs/mxfp8_implementation_plan.md` for context
3. Run the tests and continue from there

Claude Code will have access to:
- All committed code (implementation + tests)
- Full implementation plan in `docs/mxfp8_implementation_plan.md`
- Git history with detailed commit messages

## Next Steps on MI350

1. **Run basic tests** - Verify kernel execution
2. **Test with real MXFP8Tensor** - Use MXFP8Quantizer for full E2E test
3. **Numerical accuracy** - Compare against dequantized FP32 reference
4. **Performance benchmarking** - Compare vs FP32 and regular FP8
5. **Layout testing** - Test TN, NN, NT configurations
6. **Integration testing** - Test with actual training workloads

## Reference Files

### Implementation
- `transformer_engine/pytorch/gemm_triton.py` - Main implementation

### Tests
- `tests/pytorch/mxfp8/` - Test suite directory

### Documentation
- `docs/mxfp8_implementation_plan.md` - Implementation plan
- `docs/MXFP8_MIGRATION_GUIDE.md` - This file

### C++ Reference (for verification)
- `transformer_engine/pytorch/csrc/type_converters.cpp` - MXFP8Tensor extraction
- `transformer_engine/common/gemm/cublaslt_gemm.cu` - Data/scale selection
- `transformer_engine/common/utils.cuh` - E8M0 conversion formulas

## Success Criteria

- ✓ All static verification checks pass
- ⏳ Runtime tests execute without errors on MI350
- ⏳ Kernel produces non-zero output
- ⏳ Numerical accuracy within tolerance (rtol=1e-2)
- ⏳ Performance competitive with FP32 baseline

---

**Implementation completed on:** 2026-01-27
**Migration to:** MI350
**Branch:** `triton_gemm_mxfp8`
**Remote:** `docs` (wenchenvincent/TransformerEngine-private)
