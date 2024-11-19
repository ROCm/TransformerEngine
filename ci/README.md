# Transformer Engine ROCm CI #

This directory contains scripts to prepare and run TE unit tests on ROCm dockers manually or from CI automation.
There are 3 executable scripts here:
* `core.sh` - build and run tests/cpp unit tests
* `jax.sh` - install prerequisites and run tests/jax framework integration tests
* `pytorch.sh` - install prerequisites and run tests/pytorch framework integration tests

The scripts return 0 in case of test success, and other values for testing errors. Logging is performed on standard output and error streams.

The scripts can be controlled by environment variables:
* `TEST_LEVEL` specifies testing thoroughness. Levels 1 and 3 are currently defined and can be used to run in feature branch and main branch correspondingly. Default=99 (maximal thoroughness)
* `TEST_JOBS` specifies the maximal number of background parallel test jobs on MGPU system (only pytorch and jax). Default=0 (disable background jobs).
The actual number of jobs is limited to the number of available GPUs and each job is run on its own GPU. Distributed tests that require mGPU are run separately on all available GPUs.
Jobs are distributed based on config: HIPBLASLT/ROCBLAS CK/AOTRITON/UNFUSED, thus the actual max number of jobs is 6 for Pytorch and 3 for JAX tests.
* `JUNITXML_PREFIX` and `JUNITXML_SUFFIX` enable pytest (pytorch and jax) junitxml logging if set. Each test will generate a junitxml log with the full filename `JUNITXML_PREFIX<test_name>.<test_config>JUNITXML_SUFFIX`.
If JUNITXML_PREFIX contains a path component, it is the caller's responsibility to create necessary directories.
If `JUNITXML_PREFIX` contains only a directory (no filename prefix), it should end with `/`.
Test scripts do not add any extension to the log filename so it is advised to end `JUNITXML_SUFFIX` with `.xml`.
It is the caller's responsibility to clean up generated files.
