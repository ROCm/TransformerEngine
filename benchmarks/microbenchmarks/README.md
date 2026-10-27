# Transformer Engine Microbenchmarks

This directory contains lightweight Python microbenchmarks for selected
Transformer Engine kernels and helper scripts for comparing benchmark CSVs.
For how to read the results and why the harness measures the way it does, see
[BENCHMARKING_NOTES.md](BENCHMARKING_NOTES.md).

## Benchmarks

- `benchmark_gemm.py`: dense GEMM benchmark sweeping BF16 plus the supported
  low-precision recipes (FP8, MXFP8, MXFP4, NVFP4) via `autocast`
- `benchmark_grouped_gemm.py`: grouped GEMM benchmark for MoE-style shapes
- `benchmark_casting.py`: quantize / dequantize benchmark across FP8, MXFP8, NVFP4, and MXFP4
  (MXFP4 quantize only)
- `benchmark_normalization.py`: LayerNorm / RMSNorm forward benchmark with BF16 output and each
  supported quantized output (FP8, MXFP8, MXFP4, NVFP4)

## Running

Run a benchmark with `python`:

```bash
python benchmark_gemm.py                 # run, print a results table
```

Each benchmark is a pytest module under the hood (wired up by `conftest.py`),
and the file forwards its arguments to pytest, so any pytest flag (`-k`, `-v`,
...) and the custom options below behave the same whether you launch it with
`python benchmark_gemm.py` or `pytest benchmark_gemm.py`. To run every family in
one go you can use `pytest .`.

### CSV output

Pass `--csv` to write results. With no filename the CSV is named after the
module (`benchmark_gemm.py` -> `benchmark_gemm.csv`):

```bash
python benchmark_gemm.py --csv
python benchmark_grouped_gemm.py --csv grouped_results.csv
```

To also save per-sample timing data for downstream analysis (e.g. histograms,
confidence intervals), pass `--csv-samples`:

```bash
python benchmark_gemm.py --csv --csv-samples
python benchmark_gemm.py --csv --csv-samples gemm_samples.csv
```

The samples CSV (`<module>_samples.csv` by default) contains one row per timing
sample with columns for all benchmark parameters plus `label`, `sample_idx`,
and `time_ms`.

The `benchmark_gemm.py` CSV also has a `Kernels` column with the GEMM kernels the timed
direction launched (`;`-separated, in launch order): hipBLASLt kernel names (Tensile names
are cut after the tile and matrix-instruction sizes and get a `~<crc32>` of the full name)
and, for Triton, the autotuned kernel with the config it picked, e.g.
`matmul_kernel[128x256x32 GROUP_SIZE_M=4 waves_per_eu=0 num_warps=8 num_stages=3]`. It is
empty for the other backends.

### Selecting cases

Use pytest's `-k` to run a subset by parametrize id:

```bash
python benchmark_gemm.py -k "bf16 and QKV"   # select shapes/precisions
python benchmark_gemm.py -k triton           # select the Triton backend
python benchmark_gemm.py -k "not flydsl"     # skip FlyDSL (long compile times)
```

### Other options

- `--kernel-profile`: also measure GPU kernel (device) time (see
  [Kernel profiling](#kernel-profiling)).
- `--no-gpu-interference-check` / `--abort-on-gpu-interference`: control the
  shared-GPU neighbor check, which warns (or aborts) when another process is
  using the benchmark GPU (requires the `amdsmi` package).
- `--no-cpu-pin`: disable CPU pinning. By default the main thread is pinned to 4 physical
  cores on the GPU's NUMA node (offset by GPU id), the HIP/ROCr helper threads to the cores'
  SMT siblings, and the cores are set to the performance governor for the session (needs
  root; restored at the end, also if the process is killed or crashes, by a small watchdog
  process). The header and `run_info.txt` show the cores used; see
  [CPU pinning](BENCHMARKING_NOTES.md#cpu-pinning) for why.
- `--dashboard-run`: write all families of the session into one run directory for the TE
  dashboard instead of the `--csv` outputs:
  `<dashboard-out>/<arch>_<host>_gpu<id>_<pci>/<date>_<commit>/` with
  `benchmark_<family>.csv` (each row tagged with the run timestamp and commit),
  `run_info.txt` (host, GPU, software versions, CPU pinning, HW queues) and, with
  `--csv-samples`, `samples/`. `--dashboard-out DIR` sets the base directory (default
  `results`); `--gpu-model NAME` overrides the detected GPU model (the run is refused if
  no model can be detected and none is given). A full dashboard run:

  ```bash
  pytest . --dashboard-run --kernel-profile --csv-samples
  ```

### Environment

- `GPU_MAX_HW_QUEUES` defaults to `2`, as in Megatron's launch scripts; set it explicitly to
  override (with `4`, HIP's default, the per-expert grouped GEMM path runs up to ~2x slower;
  see [HW queues](BENCHMARKING_NOTES.md#hw-queues)). The header and `run_info.txt` show the
  value used.
- The `hipblaslt` rows of `benchmark_gemm.py` run with TE's hipBLASLt tuning
  (`TE_HIPBLASLT_TUNING_RUN_COUNT=10`, `TE_HIPBLASLT_TUNING_ALGO_COUNT=10`), which Megatron's
  training scripts enable for dense GEMMs. Each shape is tuned at its first call (about a
  minute per MXFP8 shape, whose hipBLASLt kernels are compiled on the fly); the `Kernels`
  CSV column (see [CSV output](#csv-output)) shows the kernels it picked.

### Kernel profiling

Pass `--kernel-profile` to report GPU kernel (device) time alongside the default
host wall-clock time. It adds `<label> Kernel Time (ms)` / `<label> Kernel <unit>`
columns to the console table and the CSV:

```bash
python benchmark_gemm.py --kernel-profile --csv
```

- **Wall time** is the end-to-end host-side time per call over back-to-back calls (median):
  Python dispatch + kernel launch/queue latency + GPU execution. Forward is timed with
  `torch.utils.benchmark.Timer`; backward runs a block of forwards untimed and then times
  their backwards back to back, so no forward time is subtracted.
- **Kernel time** is the GPU device time only (`torch.profiler`, summing the device
  time of each call's kernels, reported as the median over a separate profiling pass),
  so it excludes host and launch overhead. If no call's kernels can be matched, the
  kernel cells are left empty and a warning names the case.
- **Grouped GEMM kernel time is not device-only.** It is the CUDA-event span between
  consecutive calls on the main stream, which includes GPU idle time in launch-bound cases.

See [Wall time and kernel time](BENCHMARKING_NOTES.md#wall-time-and-kernel-time) for how to
interpret the two (launch-bound gaps, profiler effects, grouped GEMM).

Because it runs an extra warmup plus a profiled loop under `torch.profiler`,
`--kernel-profile` makes each benchmark take longer.

### Rotating input buffers

By default each benchmark cycles its inputs through a ring of buffers whose
total footprint exceeds the **last-level cache**, so back-to-back kernel
launches touch different memory (closer to a cold-cache, steady-state workload)
instead of reading data still resident in cache and reporting optimistic
numbers. This matches the `--rotating` option of `hipblaslt-bench`, which
likewise takes a rotating memory budget in MB. Pass `--no-rotating` to instead
time a single cached input buffer:

```bash
python benchmark_gemm.py                     # rotate, auto-size the ring past the LLC
python benchmark_casting.py --rotating 512   # rotate within a 512 MB budget
python benchmark_gemm.py --no-rotating       # single cached input buffer
```

Rotation is **on by default**. Passing `--rotating MB` sets the rotating memory
budget in megabytes (the ring holds enough buffers to span it); omitting the
value auto-sizes the ring to ~2x a conservative 256 MB last-level cache (the AMD
Infinity Cache; see `utils.py::_last_level_cache_bytes`). `--no-rotating`
disables rotation entirely.

## Shared configuration

Common benchmark settings live in `utils.py`.

- `M_SIZE_LIST`: default token-count sweep for dense and elementwise kernels
- `DTYPE_LIST`: shared dtype sweep for TE activation benchmarks
- `MODEL_CONFIGS`: dense GEMM model shapes
- `MODEL_HIDDEN_SIZES`: hidden sizes for elementwise kernels

Grouped GEMM keeps its own smaller M sweep because its working set scales with
expert count `B` in addition to `M`.

## Adding a benchmark

Each benchmark is a pytest module named `benchmark_<family>.py` (`conftest.py` collects that
pattern, and the family and CSV names come from it) with three pieces:

1. A case generator plus `pytest_generate_tests`, which parametrizes a `case`:

   ```python
   def pytest_generate_tests(metafunc):
       if "case" in metafunc.fixturenames:
           cases = generate_cases()
           metafunc.parametrize("case", cases, ids=[_case_id(c) for c in cases])
   ```

2. A `bench_*` function that runs the kernel and returns a list of metric
   records built with `make_metric_record(...)`, or with `direction_records(...)`,
   which times the forward or the backward of a forward/backward callable pair.

3. A `@pytest.mark.benchmark` test that applies any backend env vars and hands
   the work to the `microbench` fixture:

   ```python
   @pytest.mark.benchmark
   def test_gemm(microbench, case, monkeypatch):
       apply_backend_env(monkeypatch, BACKENDS[case["Backend"]])
       microbench.run(case, lambda: bench_gemm(**case))
   ```

Start the file, right after the module docstring and before importing torch, with
the direct-run shim so `python benchmark_x.py` works. Entering pytest before torch
touches the GPU keeps this process out of the GPU-interference snapshot:

```python
if __name__ == "__main__":
    import sys

    import pytest

    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
```

`microbench.run(case, bench_callable)` prints the case, records its metrics, and
`conftest.py` collects them per family and writes the CSV. Each metric record
represents one line such as `Normalization Forward` and expands into two CSV columns:

- `<label> Wall Time (ms)`
- `<label> Wall <unit>`

For example, a `Normalization Forward` metric with unit `GB/s` becomes
`Normalization Forward Wall Time (ms)` and `Normalization Forward Wall GB/s`. Passing
`--kernel-profile` adds matching `<label> Kernel Time (ms)` / `<label> Kernel <unit>` columns.

## Comparing results

Use `compare_results.py` to compare two CSV files from the same benchmark
family:

```bash
python compare_results.py baseline.csv candidate.csv --bench-name GEMM
```

The script auto-detects metric columns, computes speedups for overlapping rows,
and reports rows that exist only in the baseline or only in the candidate. It ignores the
run-tag columns of `--dashboard-run` CSVs (`run_ts`, `commit_sha`, `commit_date`), so two
dashboard runs can be compared directly.
