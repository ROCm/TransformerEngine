# Transformer Engine Microbenchmarks

This directory contains lightweight Python microbenchmarks for selected
Transformer Engine kernels and helper scripts for comparing benchmark CSVs.

## Benchmarks

- `benchmark_gemm.py`: dense GEMM benchmark sweeping BF16 plus the supported
  low-precision recipes (FP8, MXFP8, MXFP4, NVFP4) via `autocast`
- `benchmark_grouped_gemm.py`: grouped GEMM benchmark for MoE-style shapes
- `benchmark_casting.py`: quantize / dequantize benchmark across FP8, MXFP8, NVFP4, and MXFP4
- `benchmark_normalization.py`: LayerNorm / RMSNorm forward benchmark across BF16 and quantized (FP8, MXFP8) output

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

### Selecting cases

Use pytest's `-k` to run a subset by parametrize id:

```bash
python benchmark_gemm.py -k "bf16 and QKV"   # select shapes/precisions
python benchmark_gemm.py -k triton           # select the Triton backend
```

### Other options

- `--kernel-profile`: also measure GPU kernel (device) time (see
  [Kernel profiling](#kernel-profiling)).
- `--run-flydsl`: include FlyDSL-backed cases (skipped by default due to long compile times).
- `--no-gpu-interference-check` / `--abort-on-gpu-interference`: control the
  shared-GPU neighbor check, which warns (or aborts) when another process is
  using the benchmark GPU (requires the `amdsmi` package).

### Kernel profiling

Pass `--kernel-profile` to report GPU kernel (device) time alongside the default
host wall-clock time. It adds `Kernel Time (ms)` / `Kernel <unit>` columns to the
console table and the CSV:

```bash
python benchmark_gemm.py --kernel-profile --csv
```

- **Wall time** is the end-to-end host-side time per call (`torch.utils.benchmark.Timer`,
  reported as the median): Python dispatch + kernel launch/queue latency + GPU
  execution.
- **Kernel time** is the GPU device time only (`torch.profiler`, summing
  `self_device_time_total`, reported as the mean over a separate profiling pass),
  so it excludes host and launch overhead.

Because kernel time drops the launch/dispatch overhead, kernel throughput is
normally at least as high as wall throughput. A large gap means the case is
launch/dispatch-bound rather than compute-bound (common at small sizes or with
higher-overhead backends such as Triton); when the two nearly match, the op is
compute-bound and any sub-percent difference is just noise between the two
independent measurement passes.

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

Each benchmark is a pytest module with three pieces:

1. A case generator plus `pytest_generate_tests`, which parametrizes a `case`:

   ```python
   def pytest_generate_tests(metafunc):
       if "case" in metafunc.fixturenames:
           cases = generate_cases()
           metafunc.parametrize("case", cases, ids=[_case_id(c) for c in cases])
   ```

2. A `bench_*` function that runs the kernel and returns a list of metric
   records built with `make_metric_record(...)` or
   `make_forward_backward_metric_records(...)`.

3. A `@pytest.mark.benchmark` test that applies any backend env vars and hands
   the work to the `microbench` fixture:

   ```python
   @pytest.mark.benchmark
   def test_gemm(microbench, case, monkeypatch):
       apply_backend_env(monkeypatch, BACKENDS[case["Backend"]])
       microbench.run(case, lambda: bench_gemm(**case))
   ```

Finish the file with the direct-run shim so `python benchmark_x.py` works:

```python
if __name__ == "__main__":
    import sys
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
```

`microbench.run(case, bench_callable)` prints the case, records its metrics, and
`conftest.py` collects them per family and writes the CSV. Each metric record
represents one line such as `GEMM Forward` and expands into two CSV columns:

- `<label> Time (ms)`
- `<label> <unit>`

For example, a `GEMM Forward` metric with unit `TFLOPS` becomes
`GEMM Forward Time (ms)` and `GEMM Forward TFLOPS`. Passing `--kernel-profile`
adds matching `Kernel Time (ms)` / `Kernel <unit>` columns.

## Comparing results

Use `compare_results.py` to compare two CSV files from the same benchmark
family:

```bash
python compare_results.py baseline.csv candidate.csv --bench-name GEMM
```

The script auto-detects metric columns, computes speedups for overlapping rows,
and reports rows that exist only in the baseline or only in the candidate.
