# Benchmarks for TransformerEngine

GPU microbenchmarks driven by `driver.py`. Results are written in
[ASV (Air Speed Velocity)](https://asv.readthedocs.io/) JSON format so they
can be browsed with `asv publish` / `asv preview`, but the `asv` CLI is **not**
used to run benchmarks — `driver.py` runs everything in-process.

## Prerequisites

- TransformerEngine must already be built and installed in the current Python environment.
- A ROCm or CUDA GPU must be available.
- `asv` is only required if you want the HTML dashboard (`pip install asv`).

## Running benchmarks

Each `bench_*.py` file is directly executable, or you can drive them through
`driver.py`. Results are saved to `benchmarks/.asv/results/` in ASV-compatible
format by default.

```bash
cd benchmarks/asv
python driver.py --all                      # run every suite
python driver.py bench_gemm                 # run one suite via driver
python bench_gemm.py                        # run one suite directly
python bench_gemm.py time_forward           # filter to a specific method
python bench_gemm.py -w 5 -n 20             # custom warmup/iteration counts
python bench_casting.py --no-save           # skip saving results
python bench_casting.py --cold-cache        # flush cache before each sample
python bench_gemm.py --inner 50             # fix inner-loop count to 50
python bench_gemm.py --target-window-ms 5   # tune inner so each window >=5 ms
```

### Timing model: inner loop and cache state

Each `time_*` method runs the kernel `_inner` times inside a single CUDA event
window and divides by `_inner`, so kernel-launch and CUDA-event jitter
(`~0.5 µs` resolution on AMD) are amortized. By default the driver
**auto-tunes** `_inner` per (combo, method) so each window lasts at least
`--target-window-ms` (default `1.0 ms`):

| Flag | Effect |
|---|---|
| `--inner auto` (default) | Probe a single invocation, then pick `_inner` so the next timed window lasts ≥ `--target-window-ms`. Capped at 10000. |
| `--inner N` | Force a fixed `_inner = N` (overrides auto-tune). |
| `--target-window-ms T` | Target window duration for `--inner auto` (default `1.0`). |
| `--cold-cache` | Write a `--cache-flush-mb` byte scratch buffer before each sample to evict L2 + Infinity Cache. Implies `--inner=1` (otherwise iterations 2..N would refill the cache and the measurement degenerates back to warm-cache). |
| `--cache-flush-mb M` | Scratch buffer size for `--cold-cache` (default `256`, sized for the MI300 Infinity Cache). |

Choose the regime that matches the question you're asking:
- **Warm cache, large `_inner`** (default): steady-state kernel throughput,
  matches what a hot inner loop in a model sees. Lowest variance.
- **Cold cache, `_inner=1`**: realistic cost of the kernel as an isolated
  call into cold memory — closer to what `rocprofv3 --hip-trace` reports
  on a freshly launched kernel. Higher variance; bandwidth-bound
  benchmarks (cast, normalization) typically run 1.5–3× slower than warm.

Caveat: the inner loop runs in Python, so each iteration carries
~80–200 ns of interpreter overhead. For sub-microsecond kernels this is
not removable without CUDA graph capture; pick `--inner` deliberately
in that regime or use the cold-cache mode.

### Sample scheduling: interleaving

By default the driver does **not** collect a benchmark's samples in one
contiguous block. It samples in round-robin chunks: it sets up a group of
`(method, combo)` benchmarks, then takes one sample from each per round, for
`-n` rounds. This is on by default because *sequential* scheduling (all of A,
then all of B) makes wall-clock time a proxy for benchmark identity — so any
time-correlated GPU noise (thermal warm-up ramp, DVFS throttle, a neighbor
container on a shared GPU) becomes a systematic **bias** between benchmarks
rather than noise. The Monte-Carlo study in `repro/transient_noise_sim.py`
quantifies it: under a 5% thermal ramp a sequential Brunner-Munzel comparison
fires a false positive 86% of the time (α=0.05), and a 20% ramp can flip a real
5% speedup into a reported regression. Round-robin sampling spreads every
benchmark across the same window, so a transient lands on one sample of each
instead of corrupting one benchmark's whole block.

The per-round visit order is also **randomly permuted** each round (a balanced
randomized design, not a global shuffle). Fixed round-robin would still pin each
benchmark to a constant phase within the round — so a monotonic ramp leaves a
small constant per-benchmark offset, and each benchmark always sees the same
predecessor's cache/clock state. Re-permuting each round makes both uniform in
expectation, turning that residual bias into variance. The shuffle is seeded
(`--seed`, default `0`) so runs stay reproducible.

| Flag | Effect |
|---|---|
| `--interleave-group N` (default `8`) | Number of benchmarks sampled round-robin together. Each keeps a live GPU instance for the duration of the chunk, so **lower this if a group runs out of memory**; raise it to share the time window across more benchmarks. |
| `--sequential` | Collect each benchmark's samples contiguously (≡ `--interleave-group 1`). Lowest memory, but biased under thermal drift — use only for quick local runs. |
| `--seed S` (default `0`) | Seed for the per-round shuffle, fixed so runs are reproducible. |
| `--no-shuffle` | Use a fixed round-robin order instead of permuting each round. Leaves a small residual ordering/predecessor bias; mainly for debugging. |

Caveat: interleaving removes *within-run* time-position bias. It does **not**
remove a whole-run thermal offset between two **separately produced** result
files (e.g. a cold baseline run vs. a warm candidate run). For the statistical
comparison below, produce the baseline and candidate result files back-to-back
under similar conditions.

### Helper script

`run_benchmarks.sh` wraps common tasks and can be run from anywhere.

```bash
bash benchmarks/asv/run_benchmarks.sh <command> [options]
```

| Command | Description |
|---|---|
| `run [suite] [method]` | Run benchmarks in-process (saves ASV-compatible results) |
| `view` | Build the ASV HTML dashboard from saved results and serve it on `localhost:8080` |
| `list` | List available benchmark suites |
| `compare BASE CAND` | Statistically compare two result JSONs (exits 1 on a significant regression) |

## How results are stored

ASV-format JSON files under `benchmarks/.asv/results/`:

```
benchmarks/.asv/results/
  my-machine-name/
    machine.json           # Hardware/OS metadata (auto-generated by driver)
    <commit-hash>.json     # Timing results for that commit
    <commit-hash>.json
    ...
```

Each commit JSON contains the wall-clock timings for every benchmark + parameter combination
run on that machine, including the raw per-call samples (the ASV `samples`
column) used by `compare_results.py`. The `benchmarks/.asv/` directory is in
`.gitignore`.

## Viewing results

To browse historical results in a dashboard, point `asv` at the saved JSON:

```bash
bash benchmarks/asv/run_benchmarks.sh view
# or, manually:
asv publish --config benchmarks/asv/asv.conf.json
asv preview --config benchmarks/asv/asv.conf.json
```

`asv.conf.json` exists only to support `publish` / `preview`; benchmarks
themselves are not invoked through `asv`.

## Comparing two checkouts statistically

The dashboard plots point estimates (medians), which cannot tell a real
regression from measurement noise. To test whether timing differences between
two checkouts are statistically significant, the driver records the raw per-call
samples in each result file (the ASV `samples` column), and `compare_results.py`
compares them with a Brunner-Munzel test via the
[benchstats](https://github.com/Arech/benchstats) package:

```bash
pip install -r requirements.txt   # benchstats (pulls rich, scipy, numpy)

cd benchmarks/asv

# baseline checkout — saves <baseline-hash>-<env>.json
python driver.py --all -n 20
# candidate checkout — saves <candidate-hash>-<env>.json
python driver.py --all -n 20

python compare_results.py \
    ../.asv/results/<machine>/<baseline-hash>-<env>.json \
    ../.asv/results/<machine>/<candidate-hash>-<env>.json
```

It prints a table marking each `(benchmark, parameter combination)` as faster
(`<`), slower (`>`), or not significantly different (`~`), and exits `1` when a
significant difference is found, so it can gate CI.

By default the result filename is derived from the commit hash, so two runs on
the **same** commit (e.g. prototyping against a dirty working tree, where `HEAD`
is unchanged) would overwrite each other. Pass `--label` to fold a tag into the
filename and keep them distinct:

```bash
python driver.py --all -n 20 --label base   # -> <hash>-base-<env>.json
# ... edit code (HEAD stays the same) ...
python driver.py --all -n 20 --label cand   # -> <hash>-cand-<env>.json

python compare_results.py \
    ../.asv/results/<machine>/<hash>-base-<env>.json \
    ../.asv/results/<machine>/<hash>-cand-<env>.json
```

| Flag | Effect |
|---|---|
| `--alpha A` | Significance level for the test (default `0.001`). |
| `--method M` | Statistical test to use (default `brunnermunzel`). |
| `--filter REGEX` | Only compare benchmarks whose name matches `REGEX`. |
| `--always-show-pvalues` | Show p-values for non-significant rows too. |
| `--export-to FILE` | Save the report to a `.txt`/`.svg`/`.html` file. |

The test is rank-based and needs a reasonable number of samples per benchmark
(≥ ~10 recommended); the default `-n 20` timed iterations satisfies this. Only
timing is tested — throughput (`TFLOPS`/`GB/s`) is a constant-work transform of
time, so a rank test on it is identical; the driver already prints throughput
columns during a run.

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

    # Driver overrides per (combo, method): _inner controls how many kernel
    # invocations land in one CUDA event window; _scratch (when not None) is
    # written to before each sample to evict the GPU cache.
    _inner = 1
    _scratch = None

    def setup(self, M, config):
        # Allocate tensors, create modules.
        # This runs once per (combo, method); the same instance is reused for
        # warmup and timed iterations.
        self._evt = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ...

    def time_forward(self, M, config):
        # Use CUDA events for accurate GPU timing.
        # Return elapsed seconds per single invocation — the driver uses this
        # instead of wall time. Looping inside the event window amortizes
        # CUDA event resolution and kernel-launch overhead.
        if self._scratch is not None:
            self._scratch.fill_(1.0)        # cold-cache mode
        self._evt[0].record()
        for _ in range(self._inner):
            self.module(self.x)
        self._evt[1].record()
        torch.cuda.synchronize()
        return self._evt[0].elapsed_time(self._evt[1]) / 1000 / self._inner

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
- Use CUDA events and return elapsed seconds **per single invocation** —
  divide the event delta by `self._inner` so the driver and the throughput
  columns get per-call values regardless of inner-loop count.
- Honor `self._inner` (loop the kernel) and `self._scratch` (write before
  recording the start event) so the driver's `--inner` and `--cold-cache`
  flags work for your benchmark.
- Optionally define `work_<name>` companions to get TFLOPS or GB/s columns.
  These return the per-call work, not per-window work.
- Clear `.grad` attributes in backward benchmarks to prevent memory accumulation.
- The `params` list defines a cross-product; keep the matrix size reasonable.
