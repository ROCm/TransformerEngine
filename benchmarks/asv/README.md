# ASV Benchmarks for TransformerEngine

Performance benchmarks built on [ASV (Air Speed Velocity)](https://asv.readthedocs.io/),
a framework for benchmarking Python packages over their lifetime.

## Prerequisites

- TransformerEngine must already be built and installed in the current Python environment.
- A ROCm or CUDA GPU must be available.
- Install ASV: `pip install asv`

ASV is configured with `environment_type: "existing"` (in `benchmarks/asv/asv.conf.json`),
meaning it uses the current Python environment directly — it does not create virtualenvs or
attempt to build TE itself. The config sets `branches: ["HEAD"]` so that `asv publish` accepts results from
whichever branch is currently checked out — this works for both local development
and CI (where `HEAD` points to `dev`).

## Running benchmarks

### Direct execution (recommended for development)

Each `bench_*.py` file is directly executable. Results are saved in ASV-compatible
format by default.

```bash
cd benchmarks/asv
python driver.py --all                      # run every suite
python driver.py bench_gemm                 # run one suite via driver
python bench_gemm.py                        # run one suite directly
python bench_gemm.py time_forward           # filter to a specific method
python bench_gemm.py -w 5 -n 20             # custom warmup/iteration counts
python bench_casting.py --no-save           # skip saving results
```

### Helper script

`run_benchmarks.sh` wraps common tasks and can be run from anywhere.

```bash
bash benchmarks/asv/run_benchmarks.sh <command> [options]
```

| Command | Description |
|---|---|
| `setup [name]` | Register machine with ASV (defaults to `hostname`) |
| `run [suite] [method]` | Run benchmarks in-process (fast, saves ASV-compatible results) |
| `run --asv [suite]` | Run via ASV subprocess isolation (for CI or statistical rigor) |
| `compare [ref] [new]` | Compare two commits (defaults to `HEAD~1` vs `HEAD`) |
| `view` | Generate HTML dashboard and serve on `localhost:8080` |
| `list` | List available benchmark suites |

### Manual ASV commands

All `asv` commands require `--config` with an **absolute path** and should be run
from the **repo root**. The common flags are:

```bash
ASV="asv --config $(pwd)/benchmarks/asv/asv.conf.json"
COMMON="--python=same --launch-method spawn --set-commit-hash $(git rev-parse HEAD)"
```

- `--python=same` — use the current interpreter (required with `environment_type: "existing"`)
- `--launch-method spawn` — required for CUDA/ROCm (fork causes reinitialization errors)
- `--set-commit-hash` — **required** with `environment_type: "existing"`, otherwise ASV silently discards results

```bash
$ASV machine --yes --machine mi325                  # register machine
$ASV run $COMMON                                    # run all benchmarks
$ASV run $COMMON --bench bench_casting              # single suite (regex match)
$ASV continuous $COMMON HEAD~1 HEAD                 # compare two commits
$ASV publish && $ASV preview                        # HTML dashboard on localhost:8080
```

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

## Writing new benchmarks

Create a new file in `benchmarks/asv/` following the naming convention `bench_<name>.py`.

```python
#!/usr/bin/env python3
import torch
import transformer_engine.pytorch as te

class BenchSomething:
    params = [[1024, 4096], ["config_a", "config_b"]]
    param_names = ["M", "config"]
    timeout = 300  # seconds, per parameter combination

    def setup(self, M, config):
        # Allocate tensors, create modules.
        # This runs before each time_* method but is NOT timed.
        self._evt = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ...

    def time_forward(self, M, config):
        # Use CUDA events for accurate GPU timing.
        # Return elapsed seconds — the driver uses this instead of wall time.
        self._evt[0].record()
        self.module(self.x)
        self._evt[1].record()
        torch.cuda.synchronize()
        return self._evt[0].elapsed_time(self._evt[1]) / 1000

    # Optional: define work_<name> to get throughput columns (TFLOPS / GB/s).
    def work_forward(self, M, config):
        return {"flops": 2 * M * self.N * self.K}   # compute-bound
        # return {"bytes": M * self.hidden * 4}      # memory-bound

if __name__ == "__main__":
    from driver import run_as_main
    run_as_main(__file__)
```

Key rules:
- Method names starting with `time_` are automatically timed.
- Use CUDA events and return elapsed seconds for accurate GPU timing.
- Optionally define `work_<name>` companions to get TFLOPS or GB/s columns.
- Clear `.grad` attributes in backward benchmarks to prevent memory accumulation.
- The `params` list defines a cross-product; keep the matrix size reasonable.
