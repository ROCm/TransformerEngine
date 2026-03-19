# ASV Benchmarks for TransformerEngine

Performance benchmarks built on [ASV (Air Speed Velocity)](https://asv.readthedocs.io/),
a framework for benchmarking Python packages over their lifetime.

## Prerequisites

- TransformerEngine must already be built and installed in the current Python environment.
- A ROCm or CUDA GPU must be available.
- Install ASV: `pip install asv`

ASV is configured with `environment_type: "existing"` (in `asv.conf.json` at the repo root),
meaning it uses the current Python environment directly — it does not create virtualenvs or
attempt to build TE itself. The config sets `branches: ["HEAD", "dev"]` so that `asv publish`
accepts results from both the currently checked-out branch and `dev` (for CI history).

## Helper script

A convenience wrapper (`benchmarks/asv/run_benchmarks.sh`) is provided for common tasks.
It can be run from anywhere — it automatically `cd`s to the repo root. Available benchmark
suites are discovered dynamically from `bench_*.py` files.

```bash
bash benchmarks/asv/run_benchmarks.sh <command> [options]
```

| Command | Description |
|---|---|
| `setup [name]` | Register machine with ASV (defaults to `hostname`) |
| `run [suite]` | Run benchmarks for the current commit (optionally a single suite) |
| `quick [suite]` | Smoke test — single iteration, results not saved |
| `compare [ref] [new]` | Compare two commits (defaults to `HEAD~1` vs `HEAD`) |
| `view` | Generate HTML dashboard and serve on `localhost:8080` |
| `list` | List available benchmark suites |

Examples:

```bash
bash benchmarks/asv/run_benchmarks.sh setup mi325
bash benchmarks/asv/run_benchmarks.sh run bench_casting
bash benchmarks/asv/run_benchmarks.sh quick
bash benchmarks/asv/run_benchmarks.sh compare HEAD~3 HEAD
bash benchmarks/asv/run_benchmarks.sh view
```

## Local usage (manual ASV commands)

All commands are run from the **repository root** (where `asv.conf.json` lives).

### Register your machine

```bash
asv machine --yes --machine my-machine-name
```

This creates a machine profile in `benchmarks/.asv/results/my-machine-name/machine.json`.
Use a descriptive name (e.g., `mi325`, `mi300x-dev`) — results are stored per machine, so
the name must be consistent across runs for historical comparison.

### Run all benchmarks

```bash
asv run --python=same --launch-method spawn --set-commit-hash $(git rev-parse HEAD)
```

- `--python=same` — use the current interpreter (required with `environment_type: "existing"`)
- `--launch-method spawn` — required for CUDA (fork causes "Cannot re-initialize CUDA in forked subprocess")
- `--set-commit-hash` — **required** with `environment_type: "existing"`. Without it, ASV
  runs benchmarks but silently discards results. The helper script sets this automatically.

### Run a single suite

```bash
asv run --python=same --launch-method spawn --set-commit-hash $(git rev-parse HEAD) --bench bench_casting
```

The `--bench` argument accepts a regex that matches benchmark file or class names.

### Quick smoke test

```bash
asv run --python=same --launch-method spawn --quick --set-commit-hash $(git rev-parse HEAD) --bench bench_casting
```

`--quick` runs each benchmark only once with no statistical analysis. Useful for verifying
benchmarks work, but note that results are **not saved to disk** in quick mode.

### Compare two commits

```bash
asv continuous --python=same --launch-method spawn HEAD~1 HEAD
```

This checks out each commit, runs benchmarks on both, and reports regressions.
Note: this only works if the benchmark files exist on both commits.

### Generate an HTML dashboard

```bash
asv publish
asv preview
```

`asv publish` generates static HTML from stored results into `benchmarks/.asv/html/`.
`asv preview` serves it locally on `http://localhost:8080`.

## How results are stored

### Local results

ASV stores results as JSON files under `benchmarks/.asv/results/`:

```
benchmarks/.asv/results/
  my-machine-name/
    machine.json           # Hardware/OS metadata
    <commit-hash>.json     # Timing results for that commit
    <commit-hash>.json
    ...
```

Each commit JSON contains the wall-clock timings for every benchmark + parameter combination
run on that machine. The `benchmarks/.asv/` directory is in `.gitignore`.

### CI results (Artifactory)

In CI, benchmarks run **only on pushes to `dev`** (not on PRs). This builds a historical
record of performance on the main branch.

The CI pipeline (`.github/workflows/rocm-ci.yml`) follows this flow:

1. **Restore** — Download `results.tar.gz` from Artifactory for the current runner
2. **Benchmark** — Run `asv run`, which appends a new `{commit}.json` to the results directory
3. **Upload** — Tar up the results directory and upload back to Artifactory

Results are stored per machine at:
```
https://compute-artifactory.amd.com:5000/artifactory/rocm-generic-local/te-ci/asv-results/
  linux-te-mi325-8/results.tar.gz
  linux-te-mi355-8/results.tar.gz
```

Each tarball contains the full ASV results directory for that machine, accumulating
a new commit JSON on every push to `dev`. ASV machine names map to hardware:
`mi325` for MI325X runners, `mi355` for MI355X runners.

### Downloading CI results locally

To inspect CI results on your local machine (requires Artifactory access):

```bash
# Download results for a specific machine
curl -sf -H "X-JFrog-Art-Api:${ARTIFACTORY_API_KEY}" \
  -o results.tar.gz \
  "https://compute-artifactory.amd.com:5000/artifactory/rocm-generic-local/te-ci/asv-results/linux-te-mi325-8/results.tar.gz"

# Extract into your local ASV results directory
mkdir -p benchmarks/.asv/results
tar xzf results.tar.gz -C benchmarks/.asv/results/

# Generate and view the dashboard
asv publish
asv preview
```

This can also be provided statically via github pages.

## Writing new benchmarks

Create a new file in `benchmarks/asv/` following the naming convention `bench_<name>.py`.

```python
import torch
import transformer_engine.pytorch as te

class BenchSomething:
    params = [[1024, 4096], ["config_a", "config_b"]]
    param_names = ["M", "config"]
    timeout = 300  # seconds, per parameter combination

    def setup(self, M, config):
        # Allocate tensors, create modules.
        # This runs before each time_* method but is NOT timed.
        ...

    def time_forward(self, M, config):
        # ASV times this method (adaptive iterations + statistics).
        # MUST call torch.cuda.synchronize() to ensure GPU work completes.
        self.module(self.x)
        torch.cuda.synchronize()
```

Key rules:
- Method names starting with `time_` are automatically timed by ASV.
- Always call `torch.cuda.synchronize()` at the end of `time_*` methods.
- Clear `.grad` attributes in backward benchmarks to prevent memory accumulation.
- ASV runs each `time_*` method in a **separate subprocess** — no shared state between methods.
- The `params` list defines a cross-product; keep the matrix size reasonable.
