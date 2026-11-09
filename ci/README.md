# Transformer Engine ROCm CI #

This directory contains scripts to prepare and run TE unit tests on ROCm dockers manually or from CI automation.
There are 5 executable scripts here:
* `core.sh` - build and run tests/cpp unit tests
* `jax.sh` - install prerequisites and run tests/jax framework integration tests
* `pytorch.sh` - install prerequisites and run tests/pytorch framework integration tests
* `rocjitsu.sh` - run a small-shape subset of tests/cpp and tests/pytorch on gfx1250 (MI455X) emulated on the CPU, see [below](#gfx1250-emulation)
* `run_queue.sh` - run the scripts above as one work queue spread across the GPUs of the box; this is what CI runs

The scripts return 0 in case of test success, and other values for testing errors. Logging is performed on standard output and error streams.

The scripts can be controlled by environment variables:
* `TEST_LEVEL` specifies testing thoroughness. Levels 1 and 3 are currently defined and can be used to run in feature branch and main branch correspondingly. Default=99 (maximal thoroughness)
* `TEST_SGPU` and `TEST_MGPU` instructs to run single-GPU tests or multi-GPU tests only that can be used to run several sGPU tests parallel on mGPU config
* `TEST_FILTER` runs only the test invocations whose tag matches one of its space-separated glob patterns, e.g. `TEST_FILTER="test_numerics.*"`. A tag is `<test_name>.<test_config>`, as in the JUnit XML file names below.
* `TEST_LAUNCHER` is a command prefix for every pytest run, e.g. a profiler or an emulator.
* `JUNITXML_PREFIX` and `JUNITXML_SUFFIX` enable JUnit XML logging if set, for both pytest (pytorch and jax) and ctest (core). Each test run generates a JUnit XML log with the full filename `JUNITXML_PREFIX<test_name>.<test_config>JUNITXML_SUFFIX` (for core, the name is `gemm` / `nongemm`).
If JUNITXML_PREFIX contains a path component, it is the caller's responsibility to create necessary directories.
If `JUNITXML_PREFIX` contains only a directory (no filename prefix), it should end with `/`.
Test scripts do not add any extension to the log filename so it is advised to end `JUNITXML_SUFFIX` with `.xml`.
It is the caller's responsibility to clean up generated files.

## Work queue

`run_queue.sh` lists every test invocation of the suites as a work item and runs the items on the GPUs it can see:
* `--queue sgpu` (default) runs the single-GPU tests of the suites in [`ci_sgpu_queue.conf`](ci_sgpu_queue.conf), one item per GPU, longest first.
* `--queue mgpu` runs the multi-GPU tests of the suites in [`ci_mgpu_queue.conf`](ci_mgpu_queue.conf), each item on the whole box, one after another.
* `--queue emu` runs the suites in [`ci_emu_queue.conf`](ci_emu_queue.conf) on GPUs emulated on the CPU, so it needs no GPU: one item per CPU slot of `ROCJITSU_CPU_BUDGET` (4) physical cores, longest first. `TE_CI_EMU_SLOTS` sets the number of slots.

Everything it reads and writes is under `ci-data/` in the repo root (`TE_CI_DATA_DIR` moves it):
* `test-results/` - this run's JUnit XML and `logs/`: one log per item (`logs/items/`), per-suite verdicts and the schedule report (`logs/report/schedule.md`)
* `weights/` - learned item durations, used to start the longest items first (sGPU and emu only)
* `rerun/` - the items, and where possible the single tests, that failed; the next run queues only those

CI keeps `weights/` and `rerun/` across runs with `actions/cache`, so "Re-run failed jobs" re-runs only what failed; "Re-run all jobs" runs everything again.

### Running locally

Run from the repo root inside a TE CI docker with Transformer Engine installed. `run_queue.sh` needs bash 5.1 or newer, `rocminfo`, and `python3` with pandas.

```sh
# Single-GPU tests on every visible GPU
TEST_LEVEL=1 ci/run_queue.sh

# Only some GPUs, or only some suites (labels from ci_sgpu_queue.conf)
HIP_VISIBLE_DEVICES=0,1 TEST_LEVEL=1 ci/run_queue.sh
TEST_LEVEL=1 ci/run_queue.sh --suites "torch core"

# Multi-GPU tests on the whole box
TEST_LEVEL=3 ci/run_queue.sh --queue mgpu

# As CI runs it: with JUnit XML, which also lets a re-run narrow to single tests
JUNITXML_PREFIX=ci-data/test-results/ JUNITXML_SUFFIX=.xml TEST_LEVEL=3 ci/run_queue.sh
```

* After a run with failures, the next run queues only the failed items. Delete `ci-data/rerun/` to run everything again.
* `python3 ci/junit_report.py ci-data/test-results/torch` prints a summary of a suite's JUnit XML.
* To run one item without the queue, pass its tag to the suite script, e.g. `TEST_LEVEL=1 TEST_SGPU=1 TEST_FILTER=test_numerics.auto ci/pytorch.sh`. Add `TE_CI_LIST_ITEMS=1` to print the tags instead of running them; an item's log is `ci-data/test-results/logs/items/<suite label>.<tag>.log`.

## Which suites a pull request runs

Pull requests run only the suites their changed paths can affect, by the rules in [`.github/scripts/change_scope.py`](../.github/scripts/change_scope.py); pushes and manual runs always run every suite. The "Select Test Suites" job summary lists the rule each changed path matched.
* CI uses the PR's own copy of the rules, so a change that alters what a path affects (a new or renamed suite, moved tests) should update `change_scope.py` in the same PR.
* The `ci-skip-scope` label runs every suite at the PR's `ci-level`, from the next run (adding a `ci-level` label or pushing a commit).
* To see what a branch would run: `git diff --name-status --no-renames dev...HEAD | python3 .github/scripts/change_scope.py`

## gfx1250 emulation

`rocjitsu.sh` runs tests on gfx1250 (MI455X) emulated on the CPU by [rocjitsu](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu), so it needs no GPU. Its setup builds rocjitsu from the `ROCm/rocm-systems` commit `ROCJITSU_REF` into `ROCJITSU_HOME` (default `build/rocjitsu`, reused while the commit matches), installs the gfx1250 device wheels matching the installed ROCm SDK and PyTorch, and builds tests/cpp for gfx1250 into `ROCJITSU_CPP_BUILD_DIR` (default `build/rocjitsu_cpp_tests`).
Emulation is three to five orders of magnitude slower than hardware, so test cases with large shapes are excluded (`LARGE_SHAPES` in the script). It cannot reveal timing-dependent hazards or hardware errata and says nothing about performance.
* Each work item runs one emulator process: a shard of a C++ test group (`ROCJITSU_SHARDS` per group, default 4) or a pytest file. Tags are `cpp_<group>.<leg>.<shard>` and `<test_name>.b0`.
* `TEST_LEVEL` 1 runs functional B0 emulation (leg `b0`). Level 3 adds the race detector on the quantization and normalization tests (leg `race`; a reported race fails the item) and A0 emulation with the hsa-hotswap B0-to-A0 translation that A0 parts run (leg `a0`).
* `ROCJITSU_CPU_BUDGET` sets the emulator threads per process (default 4), and `OMP_NUM_THREADS` unless that is set; `ROCJITSU_TIMEOUT` sets the timeout per process in seconds (default 3600). A process can take up to about 4 GB of memory.
* The race detector and A0 configs and race reports go to `rocjitsu.<tag>/` in the item's working directory (`ci-data/test-results/logs/cwd/` in the queue).

```sh
# As CI runs it: every item on as many CPU slots as the box has
TEST_LEVEL=1 ci/run_queue.sh --queue emu

# One item without the queue (the first run also does the setup)
TEST_LEVEL=1 TEST_FILTER=cpp_nvfp4.b0.0 ci/rocjitsu.sh
```

In CI, the `emu_tests` job of [`rocm-ci.yml`](../.github/workflows/rocm-ci.yml) runs the emu queue on the runner named by the repository variable `TE_CPU_TEST_RUNNER` (default `build-only-te`); the repository variable `TE_CI_EMU_SLOTS` overrides its slot count. The job is advisory: its failures do not fail the run.

## CI Docker images

Default and release-specific TE CI images are listed in [`ci_config.json`](ci_config.json) under `docker_images`.

The default image is built from [`.github/scripts/Dockerfile.ci.deps`](../.github/scripts/Dockerfile.ci.deps). It pins [ROCm/aiter](https://github.com/ROCm/aiter) at commit [`77455e3ecf4f0d28756afc452e914940c45b944b`](https://github.com/ROCm/aiter/commit/77455e3ecf4f0d28756afc452e914940c45b944b). That revision was validated in CI for **MXFP4 FP4 GEMM** kernel coverage.
