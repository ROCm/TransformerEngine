# Transformer Engine ROCm CI #

This directory contains scripts to prepare and run TE unit tests on ROCm dockers manually or from CI automation.
There are 3 executable scripts here:
* `core.sh` - build and run tests/cpp unit tests
* `jax.sh` - install prerequisites and run tests/jax framework integration tests
* `pytorch.sh` - install prerequisites and run tests/pytorch framework integration tests

The scripts return 0 in case of test success, and other values for testing errors. Logging is performed on standard output and error streams.

The scripts can be controlled by environment variables:
* `TEST_LEVEL` specifies testing thoroughness. Levels 1 and 3 are currently defined and can be used to run in feature branch and main branch correspondingly. Default=99 (maximal thoroughness)
* `TEST_SGPU` and `TEST_MGPU` instructs to run single-GPU tests or multi-GPU tests only that can be used to run several sGPU tests parallel on mGPU config
* `JUNITXML_PREFIX` and `JUNITXML_SUFFIX` enable JUnit XML logging if set, for both pytest (pytorch and jax) and ctest (core). Each test run generates a JUnit XML log with the full filename `JUNITXML_PREFIX<test_name>.<test_config>JUNITXML_SUFFIX` (for core, `<test_name>.<test_config>` is `core.gemm` / `core.nongemm`).
If JUNITXML_PREFIX contains a path component, it is the caller's responsibility to create necessary directories.
If `JUNITXML_PREFIX` contains only a directory (no filename prefix), it should end with `/`.
Test scripts do not add any extension to the log filename so it is advised to end `JUNITXML_SUFFIX` with `.xml`.
It is the caller's responsibility to clean up generated files.

## Work-queue runner

CI does not run the suite scripts one after another; `run_queue.sh` expands them into one work item per test invocation and packs those across the GPUs of the box.
* `run_queue.sh --queue sgpu` runs the single-GPU tests of the suites in [`ci_sgpu_queue.conf`](ci_sgpu_queue.conf), one item per GPU.
* `run_queue.sh --queue mgpu` runs the multi-GPU tests of the suites in [`ci_mgpu_queue.conf`](ci_mgpu_queue.conf). An item runs on the number of GPUs its call site declares with a `TE_CI_GPUS=N` prefix, e.g. `TE_CI_GPUS=4 run_default_fa 2 distributed/test_numerics.py`, and items that fit side by side share the box. An item that declares nothing gets the whole box.

When adding a multi-GPU test, declare the smallest `N` that still runs every case the whole box does. Leave it undeclared if the test sizes itself to the visible GPU count (e.g. `nproc = torch.cuda.device_count()`): with fewer GPUs such a test does not fail, it silently runs less.

The queue learns per-item durations (`ci-weights/`) to order the next run, and on a re-run of a failed job queues only the items, and where possible the individual tests, that failed (`ci-rerun/`).

## CI Docker images

Default and release-specific TE CI images are listed in [`ci_config.json`](ci_config.json) under `docker_images`.

For `dev` and other branches using the `default` entry, the workflow appends a GPU-arch suffix to the image tag:

| Runner label | GPU arch | Tag suffix |
|--------------|----------|------------|
| `linux-te-mi30x-*` | gfx942 (MI300X) | `_gfx942` |
| `linux-te-mi35x-*` | gfx950 (MI350X) | `_gfx950` |

Example full reference: `registry-sc-harbor.amd.com/framework/te-ci:therock_7.13.0_ubuntu24.04_py3.12_pytorch_2.10.0_triton_3.6.0_jax_0.10.2_fa_2.8.1_gfx942` (see [`ci_config.json`](ci_config.json) for the base tag).

The default image is built from [`.github/scripts/Dockerfile.ci.deps`](../.github/scripts/Dockerfile.ci.deps). It pins [ROCm/aiter](https://github.com/ROCm/aiter) at commit [`77455e3ecf4f0d28756afc452e914940c45b944b`](https://github.com/ROCm/aiter/commit/77455e3ecf4f0d28756afc452e914940c45b944b). That revision was validated in CI for **MXFP4 FP4 GEMM** kernel coverage.
