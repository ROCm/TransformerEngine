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

## CI Docker images

Default and release-specific TE CI images are listed in [`ci_config.json`](ci_config.json) under `docker_images`.

For `dev` and other branches using the `default` entry, images are selected per runner architecture:

| Runner label | GPU arch | Image tag |
|--------------|----------|-----------|
| `linux-te-mi30x-*` | gfx942 (MI300X) | `rocm-7.12.0-ubuntu24.04-py312-gfx942_test` |
| `linux-te-mi35x-*` | gfx950 (MI350X) | `rocm-7.12.0-ubuntu24.04-py312-gfx950_test` |

The default image is built from [`.github/scripts/Dockerfile.ci.deps`](../.github/scripts/Dockerfile.ci.deps). It pins [ROCm/aiter](https://github.com/ROCm/aiter) at commit [`77455e3ecf4f0d28756afc452e914940c45b944b`](https://github.com/ROCm/aiter/commit/77455e3ecf4f0d28756afc452e914940c45b944b). That revision was validated in CI for **MXFP4 FP4 GEMM** kernel coverage.
