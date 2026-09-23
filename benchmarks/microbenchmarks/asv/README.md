# TransformerEngine Microbenchmarks

GPU microbenchmarks for TE ops (GEMM, FP8 GEMM, grouped GEMM, attention,
casting, normalization), run in-process by `driver.py`. Each suite is a
`bench_*.py` file with a `Bench*` class; the driver times every `time_*` method,
prints a table with throughput, and saves raw per-call samples to JSON for
statistical comparison.

## Prerequisites

- TransformerEngine built and installed in the current Python environment.
- A ROCm or CUDA GPU.

## Running

```bash
cd benchmarks/microbenchmarks/asv
python driver.py --all                    # run every suite
python driver.py bench_gemm               # run one suite via the driver
python bench_gemm.py                      # run one suite directly
python bench_gemm.py time_forward         # filter to methods containing a string
python bench_gemm.py -w 5 -n 20           # custom warmup / timed iterations
python bench_casting.py --no-save         # don't write a result file
python bench_casting.py --cold-cache      # flush GPU cache before each sample
python bench_gemm.py --inner 50           # fix the inner-loop count to 50
python bench_gemm.py --kernel-profile     # per-kernel CUDA-time breakdown
```

Results are written to `benchmarks/microbenchmarks/asv/results/<commit-hash>.json`
(gitignored), one raw-sample record per benchmark + parameter combination.

## Timing model: inner loop and cache state

Each `time_*` method runs its kernel `_inner` times inside one CUDA-event window
and divides by `_inner`, amortizing kernel-launch and CUDA-event jitter
(`~0.5 µs` on AMD). By default the driver auto-tunes `_inner` per (combo, method)
so each window lasts at least `--target-window-ms` (default `1.0 ms`).

| Flag | Effect |
|---|---|
| `--inner auto` (default) | Probe one invocation, then pick `_inner` so the next window lasts ≥ `--target-window-ms` (capped at 10000). |
| `--inner N` | Force a fixed `_inner = N`. |
| `--target-window-ms T` | Target window duration for `--inner auto` (default `1.0`). |
| `--cold-cache` | Write a `--cache-flush-mb` scratch buffer before each sample to evict L2 + Infinity Cache. Implies `--inner=1` (otherwise later inner iterations refill the cache). |
| `--cache-flush-mb M` | Scratch buffer size for `--cold-cache` (default `256`, sized for the MI300 Infinity Cache). |

- **Warm cache, large `_inner`** (default): steady-state throughput, lowest variance.
- **Cold cache, `_inner=1`**: isolated cold-memory cost — higher variance; bandwidth-bound benches (cast, norm) run ~1.5–3× slower than warm.

## Kernel profiling

`--kernel-profile` runs each benchmark once under `torch.profiler` instead of
collecting timing distributions, and prints the GPU kernels it launched, sorted
by total device time:

```bash
python driver.py bench_gemm --kernel-profile
python bench_attention.py time_forward --kernel-profile   # one method
```

For each `(method, parameter combo)` it reports per-kernel total/avg CUDA time,
launch count, and share of total — useful for spotting which kernel dominates or
whether an op is launch-bound. This bypasses the timing machinery (`--inner`,
`--cold-cache`, interleaving); `--profile-inner N` sets how many invocations are
profiled per run (default `1`). Output is saved to
`results/<commit-hash>-kernelprofile.json` unless `--no-save`.

## Sample scheduling: interleaving

By default the driver does **not** collect a benchmark's samples in one
contiguous block. It samples in round-robin chunks: it sets up a group of
`(method, combo)` benchmarks, then takes one sample from each per round, for `-n`
rounds. Sequential scheduling (all of A, then all of B) makes wall-clock time a
proxy for benchmark identity, so any time-correlated GPU noise (thermal ramp,
DVFS throttle, a neighbor on a shared GPU) becomes a systematic **bias** between
benchmarks rather than noise. Round-robin spreads every benchmark across the same
window, so a transient lands on one sample of each. The per-round visit order is
also randomly permuted (seeded, so runs are reproducible) to remove residual
within-round phase/predecessor bias.

| Flag | Effect |
|---|---|
| `--interleave-group N` (default `8`) | Benchmarks sampled round-robin together. Each keeps a live GPU instance, so **lower this if a group runs out of memory**. |
| `--sequential` | Collect each benchmark's samples contiguously (≡ `--interleave-group 1`). Lowest memory, biased under thermal drift. |
| `--seed S` (default `0`) | Seed for the per-round shuffle. |
| `--no-shuffle` | Fixed round-robin order instead of permuting each round (debugging). |

Interleaving removes *within-run* time-position bias. It does **not** remove a
whole-run thermal offset between two separately produced result files, so for the
comparison below, produce the baseline and candidate files back-to-back under
similar conditions.

## Comparing two checkouts statistically

The driver records raw per-call samples; `compare_results.py` compares two result
files with a Brunner-Munzel test via
[benchstats](https://github.com/Arech/benchstats):

```bash
pip install -r requirements.txt   # benchstats (pulls rich, scipy, numpy)
cd benchmarks/microbenchmarks/asv

python driver.py --all -n 20      # on the baseline checkout -> results/<base>.json
python driver.py --all -n 20      # on the candidate checkout -> results/<cand>.json
python compare_results.py results/<base>.json results/<cand>.json
```

It marks each `(benchmark, parameter combination)` faster (`>`), slower (`<`), or
not significant (`~`), and exits `1` on a significant difference (CI gating).

Two runs on the **same** commit (e.g. a dirty working tree, where `HEAD` is
unchanged) would overwrite each other; pass `--label` to keep them distinct:

```bash
python driver.py --all -n 20 --label base   # -> results/<hash>-base.json
python driver.py --all -n 20 --label cand   # -> results/<hash>-cand.json
python compare_results.py results/<hash>-base.json results/<hash>-cand.json
```

| Flag | Effect |
|---|---|
| `--alpha A` | Significance level (default `0.001`). |
| `--method M` | Statistical test (default `brunnermunzel`). |
| `--filter REGEX` | Only compare benchmarks whose name matches `REGEX`. |
| `--always-show-pvalues` | Show p-values for non-significant rows too. |
| `--export-to FILE` | Save the report to `.txt`/`.svg`/`.html`. |

The rank test needs a reasonable sample count (≥ ~10); the default `-n 20`
satisfies this. Only timing is tested — throughput is a constant-work transform
of time, so a rank test on it is identical.

## Writing a new benchmark

Add `bench_<name>.py` with a `Bench*` class subclassing `BenchBase`. Pull model
shapes from `models.py` so configs stay in one place.

```python
import torch
import transformer_engine.pytorch as te

from driver import BenchBase, run_as_main
from models import M_SIZES

class BenchSomething(BenchBase):
    params = [M_SIZES, ["config_a", "config_b"]]
    param_names = ["M", "config"]

    def setup(self, M, config):
        # Allocate tensors / modules. Runs once per (combo, method); the same
        # instance is reused for warmup and timed iterations.
        self.module = ...
        self.x = ...

    def time_forward(self, M, config):
        # self._time runs the callable _inner times in one CUDA-event window
        # and returns seconds per single invocation (handles --cold-cache).
        return self._time(lambda: self.module(self.x))

    # Optional: work_<name> returns per-call work for throughput columns.
    def work_forward(self, M, config):
        return {"flops": 2 * M * self.N * self.K}   # or {"bytes": ...}

if __name__ == "__main__":
    run_as_main(__file__)
```

Rules:
- `time_*` methods are timed automatically; time through `self._time(fn)`.
- `work_<name>` companions return **per-call** work and yield TFLOPS (`flops`) or GB/s (`bytes`) columns.
- Clear `.grad` attributes in backward benchmarks to prevent accumulation.
- `params` is a cross-product — keep the matrix size reasonable.
