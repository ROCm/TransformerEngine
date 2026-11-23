# Transformer Engine ROCm CI #

This directory contains scripts to prepare and run TE unit tests on ROCm dockers manually or from CI automation.
There are 5 executable scripts here:
* `core.sh` - build and run tests/cpp unit tests
* `jax.sh` - install prerequisites and run tests/jax framework integration tests
* `pytorch.sh` - install prerequisites and run tests/pytorch framework integration tests
* `rocjitsu.sh` - run a small-shape subset of tests/cpp and tests/pytorch on gfx1250 (MI455X) emulated on the CPU, see [below](#gfx1250-emulation)
* `run_queue.sh` - run the scripts above as one work queue spread across the GPUs of the box; this is what CI runs
* `install_deps.sh` - install the packages the tests need into a plain Ubuntu 24.04 container; see [CI test environment](#ci-test-environment)

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

Run from the repo root in the [CI test environment](#ci-test-environment), with Transformer Engine installed. The suites run in the active venv; with one venv per framework, as in CI, export `TE_CI_TORCH_VENV=/opt/venv-torch` and `TE_CI_JAX_VENV=/opt/venv-jax` and each suite runs in its own. `run_queue.sh` needs bash 5.1 or newer, `rocminfo`, and `python3` with pandas.

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

Pull requests and manual runs run only the suites their changed paths can affect, by the rules in [`.github/scripts/change_scope.py`](../.github/scripts/change_scope.py); a manual run's changes are its branch's against `dev`. Runs on `dev` and release branches run every suite. The "Plan Build and Tests" job summary lists the rule each changed path matched.
* For every suite and a fresh TE build, whatever the branch changed, run "Build and Test Branch" by hand with **Full run** checked.
* CI uses the PR's own copy of the rules, so a change that alters what a path affects (a new or renamed suite, moved tests) should update `change_scope.py` in the same PR.
* To see what a branch would run: `git diff --name-status --no-renames dev...HEAD | python3 .github/scripts/change_scope.py`

### Reusing TE builds

A run builds TE only when a file built into the wheels changed. Otherwise it reuses an earlier build of the same files: from its own PR's or branch's runs, or from runs on the branch the PR targets (`dev` for a manual run), while their artifacts are kept (3 days).
* `BUILD_RULES` in `change_scope.py` lists what isn't built into the wheels: tests, docs, CI scripts. Anything else rebuilds, including the build workflow and `build_tools/`. A new directory rebuilds until it is listed there, so list one that isn't part of the build.
* Runs on `dev` and release branches, and manual runs with **Full run**, always build.
* The "Plan Build and Tests" job summary says which build a run uses.

## gfx1250 emulation

`rocjitsu.sh` runs tests on gfx1250 (MI455X) emulated on the CPU by [rocjitsu](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu), so it needs no GPU. Its setup builds rocjitsu from the `ROCm/rocm-systems` commit `ROCJITSU_REF` into `ROCJITSU_HOME` (default `build/rocjitsu`, reused while the commit matches), installs the gfx1250 device wheels matching the installed ROCm SDK and PyTorch, and builds tests/cpp for gfx1250 into `ROCJITSU_CPP_BUILD_DIR` (default `build/rocjitsu_cpp_tests`).
Emulation is three to five orders of magnitude slower than hardware, so test cases with large shapes are excluded (`LARGE_SHAPES` in the script), as are known gfx1250 bugs (`KNOWN_FAILURES`). It cannot reveal timing-dependent hazards or hardware errata and says nothing about performance.
* Each work item runs one emulator process: a shard of a C++ test group (`ROCJITSU_SHARDS` per group, default 4) or a pytest file. Tags are `cpp_<group>.b0.<shard>` and `<test_name>.b0`, after the emulated B0 stepping. All items are level 1, so every `TEST_LEVEL` runs them.
* `ROCJITSU_CPU_BUDGET` sets the emulator threads per process (default 4), and `OMP_NUM_THREADS` unless that is set; `ROCJITSU_TIMEOUT` sets the timeout per process in seconds (default 3600). A process can take up to about 4 GB of memory.

```sh
# As CI runs it: every item on as many CPU slots as the box has
TEST_LEVEL=1 ci/run_queue.sh --queue emu

# One item without the queue (the first run also does the setup)
TEST_LEVEL=1 TEST_FILTER=cpp_nvfp4.b0.0 ci/rocjitsu.sh
```

In CI, the `emu_tests` job of [`rocm-ci.yml`](../.github/workflows/rocm-ci.yml) runs the emu queue on the runner named by the repository variable `TE_CPU_TEST_RUNNER` (default `build-only-te`) with 8 slots, so the runtime and the learned item durations do not depend on the size of the runner; the repository variable `TE_CI_EMU_SLOTS` overrides the slot count. The job is advisory: its failures do not fail the run.

## CI test environment

The tests run in a plain Ubuntu 24.04 container set up by [`install_deps.sh`](install_deps.sh). Every version is in [`deps/pyproject.toml`](deps/pyproject.toml), and [`deps/uv.lock`](deps/uv.lock) pins each package's exact version and sha256. By default it installs PyTorch and JAX into one venv, `/opt/venv`. `--frameworks torch` or `--frameworks jax` installs one framework into its own venv, `/opt/venv-torch` or `/opt/venv-jax`; CI does that for each framework a job needs, so each suite runs without the other framework installed.

| File | What it is |
|---|---|
| `deps/pyproject.toml` | Every version: the packages of both frameworks, PyTorch's, JAX's and each GPU arch's, then in `[tool.te-ci]` the base image, aiter and Flash Attention |
| `deps/uv.lock` | Generated by `uv lock`: exact versions and hashes, for both GPU archs and frameworks |
| `deps/build_fa_wheel.sh` | Builds the Flash Attention wheel for `uv.lock`'s torch; `--upload` publishes it |

[`ci_config.json`](ci_config.json) names the docker images of the release branches' CI, whose workflows read it from `dev`.

### Set it up locally

1. Start Ubuntu 24.04 with the GPUs and this repo:
   ```sh
   docker run -it --network=host --device=/dev/kfd --device=/dev/dri --shm-size=16G \
     --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
     -v "$PWD:/workspace" -w /workspace ubuntu:24.04
   ```
2. Inside it, install PyTorch and JAX for this machine's GPU, and activate the venv:
   ```sh
   ci/install_deps.sh
   source /opt/venv/bin/activate
   ```

### Update it

1. Edit `deps/pyproject.toml`.
2. Regenerate `deps/uv.lock` and install from it:
   ```sh
   ci/install_deps.sh --update-locks
   ```
3. Commit `deps/pyproject.toml` and `deps/uv.lock` together. CI installs the lock, and fails if it doesn't match `pyproject.toml`.

### Flash Attention

Flash Attention is built for the exact torch, ROCm and Python, so it isn't in `uv.lock`: `install_deps.sh` installs the wheel of the `flash-attn` release in `pyproject.toml` built for `uv.lock`'s, from its ghcr.io package (`flash-attn-wheels`). CI publishes it: when the wheel isn't published yet, the "Flash Attention wheel" job builds and uploads it next to the TE build, and the tests wait for it. Locally, `install_deps.sh` builds a missing wheel itself, which takes a while; while the package is private, `GHCR_TOKEN` set to a GitHub token with `read:packages` fetches it instead.

Only the "Publish Flash Attention wheel" workflow run by hand with "override" on replaces a published wheel.
