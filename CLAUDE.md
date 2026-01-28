# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Transformer Engine (TE) for ROCm enables acceleration of Transformer models on AMD GPUs using 8-bit floating point (FP8) precision. This is a fork optimized for AMD hardware (MI300/MI325/MI350 with gfx942/gfx950 architectures), maintaining compatibility with the NVIDIA reference while adding AMD-specific optimizations.

## Build and Installation

### Environment Setup

Key environment variables:
- `NVTE_FRAMEWORK`: Comma-separated list of frameworks to build (`pytorch`, `jax`)
- `NVTE_ROCM_ARCH`: Target architectures (e.g., `gfx942,gfx950` for MI300/MI350)
- `NVTE_USE_ROCM=1`: Force ROCm platform (vs CUDA)
- `NVTE_USE_GEMM_TRITON=1`: Enable experimental Triton GEMM backend (instead of hipBLASLt)

### Installation from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine

# Set target architectures
export NVTE_ROCM_ARCH=gfx942,gfx950  # gfx942=MI300/MI325, gfx950=MI350

# Install (in-place for development)
pip install -e . --no-build-isolation
```

**Note**: If build fails with `--no-build-isolation`, try `setuptools<80.0.0`

### Switching Between Source and Wheel Installations

Always uninstall before switching:
```bash
pip list | grep transformer.engine | xargs pip uninstall -y
```

## Testing

### PyTorch Tests

```bash
# Run all PyTorch tests
cd tests/pytorch
pytest .

# Run specific test file
pytest test_gemm_triton.py -v

# Run specific test function
pytest test_gemm_triton.py::test_matmul -v

# Run with specific marker (if defined)
pytest -m "not slow"
```

### C++ Framework-Agnostic Tests

```bash
cd tests/cpp
cmake -GNinja -Bbuild . && cmake --build build

# Run utility tests
./build/util/test_util

# Run operator tests with threading
OMP_NUM_THREADS=64 ./build/operator/test_operator
```

### JAX Tests

```bash
cd tests/jax
pytest .
```

## Architecture

### Codebase Structure

```
transformer_engine/
├── common/              # Framework-agnostic C++/HIP kernels
│   ├── activation/      # Activation functions (GELU, SwiGLU, etc.)
│   ├── fused_attn/      # Fused attention kernels
│   ├── gemm/            # GEMM backends (hipBLASLt, Triton)
│   └── normalization/   # LayerNorm, RMSNorm
├── pytorch/             # PyTorch bindings and high-level API
│   ├── module.py        # TE layers (Linear, LayerNorm, etc.)
│   ├── fp8.py           # FP8 autocasting context manager
│   ├── gemm_triton.py   # Triton GEMM backend (experimental)
│   ├── tensor/          # FP8 tensor implementations
│   │   ├── float8_tensor.py       # Regular FP8 (per-tensor scaling)
│   │   └── mxfp8_tensor.py        # MXFP8 (block scaling, 32 elements/block)
│   └── cpp_extensions/  # C++ extension wrappers
└── jax/                 # JAX bindings and API
```

### GEMM Backends

TE supports multiple GEMM backends with different precision modes:

1. **hipBLASLt** (default): AMD's optimized BLAS library
   - Controlled via C++ extensions
   - Supports FP32, FP16, BF16, FP8 (E4M3/E5M2)

2. **Triton GEMM** (experimental): Python-based Triton kernels
   - Enable with `NVTE_USE_GEMM_TRITON=1`
   - Location: `transformer_engine/pytorch/gemm_triton.py`
   - Two implementation layers:
     - `te_gemm_triton()`: Low-level kernel interface
     - `te_generic_gemm_triton()`: High-level wrapper with tensor detection
   - Supports: FP32, FP16, BF16, FP8, MXFP8

### FP8 Quantization Modes

**Regular FP8** (`Float8Tensor`):
- Per-tensor scaling (1 scale per tensor)
- Transpose is numerically trivial (transpose after quantization is valid)
- Attributes: `_data` (uint8), `_scale_inv` (float32 scalar), `_fp8_dtype`

**MXFP8** (`MXFP8Tensor`):
- Block scaling (32 elements per scale, VEC_SIZE=32)
- Scale format: E8M0 (uint8 biased exponent), formula: `scale = 2^(biased_exp - 127)`
- **Critical**: Cannot transpose after quantization without requantization (precision loss)
- Dual storage: maintains both rowwise and columnwise pre-quantized copies
- Attributes: `_rowwise_data`, `_rowwise_scale_inv`, `_columnwise_data`, `_columnwise_scale_inv`
- Triton kernel uses `tl.dot_scaled()` for block-scaled matmul

### Triton GEMM Implementation Details

**Float8Tensor Detection** (commit 5ab60234):
- `Float8TensorWrapper` class extracts FP8 components from Float8Tensor
- Converts uint8 → native PyTorch FP8 types (float8_e4m3fnuz/e5m2fnuz)
- Handles columnwise tensors with dimension reordering: [K,M,*batch] → [*batch,M,K]

**MXFP8Tensor Detection** (commit 270da804):
- `MXFP8TensorWrapper` class extracts dual rowwise/columnwise data
- Selects appropriate copy based on transpose flag (avoids requantization)
- Separate kernel `mxfp8_matmul_kernel()` uses `tl.dot_scaled()`
- E8M0 scale conversion happens inside kernel: `tl.exp2(scale_e8m0 - 127.0)`

**Key Files**:
- `transformer_engine/pytorch/gemm_triton.py`: Main implementation
- `tests/pytorch/test_gemm_triton.py`: Low-level kernel tests
- `tests/pytorch/test_gemm_triton_generic_fp8.py`: High-level wrapper tests (FP8)
- `tests/pytorch/mxfp8/`: MXFP8 GEMM test suite

**Implementation Plans**:
- `docs/fp8_triton_gemm_implementation_plan.md`: FP8 design document
- `docs/mxfp8_implementation_plan.md`: MXFP8 design document

## Common Development Patterns

### Adding a New Triton Kernel

1. Add kernel in `transformer_engine/pytorch/gemm_triton.py`
2. Use `@triton.autotune` with appropriate block size configurations
3. Follow row-major layout convention (Triton requirement)
4. Add wrapper function for Python interface
5. Create tests in `tests/pytorch/`

### Working with FP8 Tensors

```python
# Import FP8 utilities
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import fp8_autocast

# Use FP8 autocast context
with fp8_autocast(enabled=True):
    output = model(input)

# Manual FP8 tensor creation (if needed)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
quantizer = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=fp8_dtype)
fp8_tensor = quantizer(regular_tensor)
```

### Testing MXFP8 Implementation

```bash
# Set environment to use Triton backend
export NVTE_USE_GEMM_TRITON=1

# Run MXFP8 tests
python tests/pytorch/mxfp8/test_mxfp8_gemm_basic.py
python tests/pytorch/mxfp8/test_mxfp8_kernel_direct.py

# Or with pytest
pytest tests/pytorch/mxfp8/ -v
```

## Git Workflow

### Current Branch Structure

- `dev`: Main development branch (upstream ROCm)
- `triton_gemm_rebase`: Triton GEMM base implementation (BF16/FP16)
- `triton_gemm_rebase_fp8`: Regular FP8 support added
- `triton_gemm_mxfp8`: MXFP8 support added (most recent work)

### Commit Message Format

Follow existing patterns:
```
[Component] Brief description

Detailed explanation of changes:
- Key change 1
- Key change 2

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Hardware-Specific Notes

### AMD GPU Architectures

- **gfx942**: MI300/MI325 (current testing platform)
- **gfx950**: MI350 (newer platform with improved MXFP8 support)

### Known Issues

**ROCm 6.4 PyTorch**: Test `tests/pytorch/test_permutation.py` fails (SWDEV-534311)
- Workaround: Rebuild PyTorch from commit `f929e0d602a71aa393ca2e6097674b210bdf321c`

**BF16 Performance**: Triton GEMM has suboptimal BF16 performance vs hipBLASLt
- Note in code: "Perf is not as good as hipblasLt" (commit 5095cfd5)

## Migration and Testing Workflow

See `docs/MXFP8_MIGRATION_GUIDE.md` for detailed instructions on migrating MXFP8 implementation between machines (e.g., MI300 → MI350 for testing).

## Key Constants and Environment Variables

**Build-time**:
- `NVTE_PROJECT_BUILDING=1`: Set automatically during build
- `NVTE_FRAMEWORK`: Comma-separated frameworks to build
- `NVTE_ROCM_ARCH`: Target GPU architectures

**Runtime**:
- `NVTE_USE_GEMM_TRITON=1`: Enable Triton GEMM backend
- `OMP_NUM_THREADS`: Control OpenMP threading (for C++ tests)

**MXFP8 Constants**:
- `MXFP8_BLOCK_SCALING_SIZE = 32`: Number of elements per scale block
- E8M0 bias: 127 (scale = 2^(biased_exp - 127))
