# AITER Fused RoPE Local Testing

This directory provides a Dockerfile for testing the AITER fused RoPE integration
on a ROCm system. CI cannot test this feature since it depends on the `aiter` package
and ROCm hardware.

## Prerequisites

- Docker with ROCm support
- AMD GPU with ROCm drivers installed (e.g. MI250X, MI300X, MI355X)

## Quick Start

From the **repository root**:

```bash
# Build the test image
docker build -t te-aiter-rope-test \
    -f tests/pytorch/aiter_rope_test/Dockerfile .

# Run the AITER RoPE tests
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    te-aiter-rope-test
```

## Run All RoPE Tests (regression check)

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    te-aiter-rope-test \
    pytest tests/pytorch/test_fused_rope.py -v --tb=short
```

## Interactive Debugging

```bash
docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video \
    te-aiter-rope-test /bin/bash

# Inside the container:
NVTE_USE_AITER_ROPE=1 pytest tests/pytorch/test_fused_rope.py::test_aiter_rope_matches_te_fused -v
NVTE_USE_AITER_ROPE=1 pytest tests/pytorch/test_fused_rope.py::test_aiter_rope_can_use_guard -v
```

## Customization

Override build args as needed:

```bash
docker build -t te-aiter-rope-test \
    --build-arg BASE_DOCKER=rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.5.1 \
    --build-arg TE_BRANCH=feat/aiter-fused-rope \
    --build-arg GPU_ARCHS="gfx942" \
    -f tests/pytorch/aiter_rope_test/Dockerfile .
```
