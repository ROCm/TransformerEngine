# Agent instructions for TransformerEngine (ROCm fork)

## Using Docker containers
- We generally work in Docker containers for reproducibility.
- For live debugging/investigations, run build/test commands **only** inside the designated container (not on host).
- If container is unspecified, ask for the exact image/tag and launch command **before** running anything expensive.
- Before debugging, record runtime context in notes/logs:
  - container image/tag
  - ROCm version in container
  - GPU architecture visible in container
  - TE commit/submodule state
- If results are suspicious, first verify you are in the expected container and that GPU devices/libs are exposed correctly.

## Big picture
- This repo builds **one core C++/HIP library** plus optional framework bindings:
  - core: `transformer_engine/common` (CMake project producing `libtransformer_engine.so`)
  - PyTorch binding: `transformer_engine/pytorch` + `transformer_engine/pytorch/csrc`
  - JAX binding: `transformer_engine/jax` + `transformer_engine/jax/csrc/extensions`
- Python import flow is split:
  - top-level framework selection in `transformer_engine/__init__.py` (`NVTE_FRAMEWORK` controls `pytorch|jax|all|none`)
  - `.so` discovery/loading logic in `transformer_engine/common/__init__.py` (`load_framework_extension`, wheel/source/editable layouts)
- Build orchestration is in `setup.py` + `build_tools/*.py`, not only in CMake.
  - `build_tools/utils.py::rocm_build()` auto-detects ROCm first, then CUDA, unless `NVTE_USE_ROCM` is set.

## Platform/backends
- ROCm path is first-class in this fork (`README.rst`, `transformer_engine/common/CMakeLists.txt`).
- Fused attention backends are runtime/compile-time gated by env vars:
  - `NVTE_FUSED_ATTN`, `NVTE_FUSED_ATTN_CK`, `NVTE_FUSED_ATTN_AOTRITON`
- ROCm fused-attn implementation is in `transformer_engine/common/fused_attn_rocm/*`; CK and AOTriton integration is wired in `transformer_engine/common/CMakeLists.txt`.
- Build-time validation for CK args runs from `setup.py` via `tools/check_aiter_mha_args_usage.py`.

## Developer workflows you should follow
- Always initialize submodules before debugging build failures: `git submodule update --init --recursive` (required by CMake for 3rdparty deps).
- Typical source install in this repo: `pip install . --no-build-isolation` (see `README.rst`).
- C++ tests: build/run from `tests/cpp` with CMake+Ninja (`qa/L0_cppunittest/test.sh`, `ci/core.sh`).
- CI-style framework test entrypoints are shell scripts, not a single pytest command:
  - PyTorch: `ci/pytorch.sh`
  - JAX: `ci/jax.sh`
  - They use `TEST_LEVEL`, `TEST_SGPU`, `TEST_MGPU`, `TEST_FILTER` from `ci/_utils.sh`.
- Lint/format workflow is repo-specific:
  - local formatting: `qa/format.sh` (pre-commit hooks)
  - cpplint+pylint flows: `qa/L0_pytorch_lint/test.sh`, `qa/L0_jax_lint/test.sh`

## Code conventions and change boundaries
- Prefer edits in `transformer_engine/*`, `build_tools/*`, `tests/*`, `ci/*`; avoid changing `3rdparty/*` unless explicitly required.
- Preserve dual-platform structure when modifying kernels/build logic:
  - shared sources are often `.cu` then hipified for ROCm (`transformer_engine/common/CMakeLists.txt`, `build_tools/pytorch.py`, `build_tools/jax.py`).
  - never edit HIP files directly -- instead, edit the CUDA source and let the build system generate HIP variants.
- Keep environment-variable behavior stable; many tests intentionally toggle flags (examples in `ci/pytorch.sh` and `ci/jax.sh`).
- Respect existing tooling/style:
  - Python formatted by Black (line length 100) via `.pre-commit-config.yaml`
  - C/C++ style checked by cpplint and `.clang-format`

## Practical pointers for AI agents
- If import fails with missing TE extension `.so`, inspect `transformer_engine/common/__init__.py` path resolution before changing packaging.
- If framework extension unexpectedly does not build on ROCm, check framework detection in `build_tools/utils.py::get_frameworks()` (ROCm-capable torch/jax checks).
- For fused-attn regressions, reproduce under multiple backend configs (`auto`, `ck`, `aotriton`, `unfused`) like CI scripts do.
