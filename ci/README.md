# Transformer Engine ROCm CI #

This directory contains scripts to prepare and run TE unit tests on ROCm dockers manually or from CI automation.
There are 5 executable scripts here:
* `core.sh` - build and run tests/cpp unit tests
* `jax.sh` - install prerequisites and run tests/jax framework integration tests
* `pytorch.sh` - install prerequisites and run tests/pytorch framework integration tests
* `run_queue.sh` - run the scripts above as one work queue spread across the GPUs of the box; this is what CI runs
* `install_deps.sh` - install the packages the tests need into a plain Ubuntu 24.04 container; see [CI test environment](#ci-test-environment)

The scripts return 0 in case of test success, and other values for testing errors. Logging is performed on standard output and error streams.

The scripts can be controlled by environment variables:
* `TEST_LEVEL` specifies testing thoroughness. Levels 1 and 3 are currently defined and can be used to run in feature branch and main branch correspondingly. Default=99 (maximal thoroughness)
* `TEST_SGPU` and `TEST_MGPU` instructs to run single-GPU tests or multi-GPU tests only that can be used to run several sGPU tests parallel on mGPU config
* `TEST_FILTER` runs only the test invocations whose tag matches one of its space-separated glob patterns, e.g. `TEST_FILTER="test_numerics.*"`. A tag is `<test_name>.<test_config>`, as in the JUnit XML file names below.
* `JUNITXML_PREFIX` and `JUNITXML_SUFFIX` enable JUnit XML logging if set, for both pytest (pytorch and jax) and ctest (core). Each test run generates a JUnit XML log with the full filename `JUNITXML_PREFIX<test_name>.<test_config>JUNITXML_SUFFIX` (for core, the name is `gemm` / `nongemm`).
If JUNITXML_PREFIX contains a path component, it is the caller's responsibility to create necessary directories.
If `JUNITXML_PREFIX` contains only a directory (no filename prefix), it should end with `/`.
Test scripts do not add any extension to the log filename so it is advised to end `JUNITXML_SUFFIX` with `.xml`.
It is the caller's responsibility to clean up generated files.

## Work queue

`run_queue.sh` lists every test invocation of the suites as a work item and runs the items on the GPUs it can see:
* `--queue sgpu` (default) runs the single-GPU tests of the suites in [`ci_sgpu_queue.conf`](ci_sgpu_queue.conf), one item per GPU, longest first.
* `--queue mgpu` runs the multi-GPU tests of the suites in [`ci_mgpu_queue.conf`](ci_mgpu_queue.conf), each item on the whole box, one after another.

Everything it reads and writes is under `ci-data/` in the repo root (`TE_CI_DATA_DIR` moves it):
* `test-results/` - this run's JUnit XML and `logs/`: one log per item (`logs/items/`), per-suite verdicts and the schedule report (`logs/report/schedule.md`)
* `weights/` - learned item durations, used to start the longest items first (sGPU only)
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

Pull requests run only the suites their changed paths can affect, by the rules in [`.github/scripts/change_scope.py`](../.github/scripts/change_scope.py); pushes and manual runs always run every suite. The "Select Test Suites" job summary lists the rule each changed path matched.
* CI uses the PR's own copy of the rules, so a change that alters what a path affects (a new or renamed suite, moved tests) should update `change_scope.py` in the same PR.
* The `ci-skip-scope` label runs every suite at the PR's `ci-level`, from the next run (adding a `ci-level` label or pushing a commit).
* To see what a branch would run: `git diff --name-status --no-renames dev...HEAD | python3 .github/scripts/change_scope.py`

## CI test environment

The tests run in a plain Ubuntu 24.04 container set up by [`install_deps.sh`](install_deps.sh), with every package pinned by version and sha256 in [`deps/`](deps). By default it installs PyTorch and JAX into one venv, `/opt/venv`. `--frameworks torch` or `--frameworks jax` installs one framework into its own venv, `/opt/venv-torch` or `/opt/venv-jax`; CI does that for each framework a job needs, so each suite runs without the other framework installed.

| File | What it is |
|---|---|
| `deps/requirements/common.in` | Packages for both frameworks: ROCm SDK, Triton, test and build tools |
| `deps/requirements/torch.in` | PyTorch: torch, torchvision, FlyDSL, aiter's and Flash Attention's needs |
| `deps/requirements/jax.in` | JAX: jax, the JAX ROCm plugin, flax |
| `deps/requirements/config.env` | GPU archs, base image, uv and Python versions, aiter and Flash Attention versions, package sources |
| `deps/locks/` | Generated: exact versions and hashes per GPU arch, `<arch>-all.txt` for both frameworks and `<arch>-{torch,jax}.txt` for one |
| `deps/update_locks.sh` | Regenerates `deps/locks/` from `deps/requirements/` |
| `deps/build_fa_wheel.sh` | Builds the Flash Attention wheel for the venv's torch; `--upload` publishes it |

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

1. Edit `deps/requirements/`.
2. Regenerate the locks and install from them:
   ```sh
   ci/install_deps.sh --update-locks
   ```
3. Commit `deps/requirements/` and `deps/locks/` together; CI installs the committed locks.

### Flash Attention

Flash Attention is built for the exact torch, ROCm and Python, so it isn't in the locks: `install_deps.sh` installs the wheel of `FLASH_ATTN_VERSION` (in `config.env`) built for the venv's. After a change to any of them it stops with a message until that wheel exists. Build it either way:
- On GitHub: run the "Publish Flash Attention wheel" workflow on your branch. It builds the wheel and publishes it.
- Locally, in the container: build it and install it, then publish it for CI:
  ```sh
  ci/deps/build_fa_wheel.sh
  ARTIFACTORY_TOKEN=<token> ci/deps/build_fa_wheel.sh --upload
  ```
