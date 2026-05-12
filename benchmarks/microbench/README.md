# Microbenchmarks for TransformerEngine

GPU microbenchmarks driven by `driver.py`. Each `bench_*.py` file defines one
or more bench classes following an ASV-style API (`params`, `param_names`,
`time_*` methods, optional `work_*` companions). Timing uses
`torch.utils.benchmark.Timer` under the hood. The driver runs each suite
in-process and writes results as long-format CSV — one row per Timer block —
intended to be consumed by a separate analysis tool (statistical tests,
cross-run comparison).

## Prerequisites

- TransformerEngine must already be built and installed in the current Python environment.
- A ROCm or CUDA GPU must be available.

## Running benchmarks

Each `bench_*.py` file is directly executable, or you can drive them through
`driver.py`. Results are written by default to
`benchmarks/.bench-results/<machine>/<commit-short>.csv`.

```bash
cd benchmarks/microbench
python driver.py --all                      # run every suite
python driver.py bench_gemm                 # run one suite via driver
python bench_gemm.py                        # run one suite directly
python bench_gemm.py time_forward           # filter to method names containing this string
python bench_casting.py --no-csv            # stdout only, don't write CSV
python bench_casting.py --csv out.csv       # custom output path
python bench_casting.py --append            # append to existing CSV
```

## Output format

Long-format CSV — one row per `torch.utils.benchmark` block. Default location
is `benchmarks/.bench-results/<machine>/<commit-short>.csv`; the
`.bench-results` tree is in `.gitignore`. Schema:

| Column | Type | Description |
|---|---|---|
| `suite` | str | Module name (e.g. `bench_gemm`) |
| `class` | str | Bench class name (e.g. `BenchGemm`) |
| `method` | str | Timed method (e.g. `time_forward`) |
| `params` | str | `k1=v1;k2=v2` canonical form for joining across runs |
| `sample_idx` | int | Block index within this Measurement |
| `time_s` | float | Per-call elapsed seconds (Timer normalizes by `number_per_run`) |
| `number_per_run` | int | Kernel invocations averaged into this row's `time_s` |
| `tflops` | float | Per-call throughput, empty if no `work_*` flops |
| `gbps` | float | Per-call bandwidth, empty if no `work_*` bytes |
| `commit` | str | Short git HEAD hash |
| `machine` | str | `platform.node()` |
| `started_at_ms` | int | Unix-ms timestamp when this method's run began |

Per-PR comparison and statistical tests are handled by a separate analysis
tool (TBD) that reads two or more of these CSVs and joins on
`(suite, class, method, params)`. Note that `time_s` is a *block mean* —
the analysis tool should weight by `number_per_run` (or use blocks as
independent samples) when computing significance.

## Writing new benchmarks

Create a new file in `benchmarks/microbench/` following the naming convention `bench_<name>.py`.

```python
#!/usr/bin/env python3
import torch
import transformer_engine.pytorch as te

from driver import time_func


class BenchSomething:
    params = [[1024, 4096], ["config_a", "config_b"]]
    param_names = ["M", "config"]
    timeout = 300  # seconds, per parameter combination

    def setup(self, M, config):
        # Allocate tensors, create modules.
        # Runs once per (combo, method); same instance is reused for warmup
        # and timed Timer blocks.
        self.module = ...
        self.x = ...

    def time_forward(self, M, config):
        return time_func(lambda: self.module(self.x))

    def time_forward_backward(self, M, config):
        def fn():
            out = self.module(self.x)
            out.backward(self.grad_out)
        return time_func(fn)

    # Optional: define work_<name> to get throughput columns (TFLOPS / GB/s).
    def work_forward(self, M, config):
        return {"flops": 2 * M * self.N * self.K}   # compute-bound
        # return {"bytes": M * self.hidden * 4}     # memory-bound


if __name__ == "__main__":
    from driver import run_as_main
    run_as_main(__file__)
```

Key rules:
- Method names starting with `time_` are automatically timed.
- `time_*` methods must return `time_func(fn)` — a `torch.utils.benchmark.Measurement`.
- Inside `fn`, do whatever per-call work you want measured. For backward,
  let gradients accumulate in-place across iterations — Timer's repeated
  invocations don't OOM (grads accumulate into the same tensor) and the
  numerical correctness of accumulated grad doesn't affect timing.
- Optionally define `work_<name>` companions to get TFLOPS or GB/s columns.
  Return per-call work; the driver derives per-sample throughput.
- The `params` list defines a cross-product; keep the matrix size reasonable.

