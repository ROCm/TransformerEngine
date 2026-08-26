# Transformer Engine Microbenchmarks

This directory contains lightweight Python microbenchmarks for selected
Transformer Engine kernels and helper scripts for comparing benchmark CSVs.

## Benchmarks

- `benchmark_gemm.py`: dense GEMM benchmark sweeping BF16 plus the supported
  low-precision recipes (FP8, MXFP8, MXFP4, NVFP4) via `autocast`
- `benchmark_grouped_gemm.py`: grouped GEMM benchmark for MoE-style shapes
- `benchmark_casting.py`: quantize / dequantize benchmark across FP8, MXFP8, NVFP4, and MXFP4
- `benchmark_normalization.py`: LayerNorm / RMSNorm forward benchmark across BF16 and quantized (FP8, MXFP8) output

Run a benchmark directly from this directory. Pass `--csv` to write results.
When no filename is provided, `run_benchmarks` derives the CSV name from the
benchmark script file name.

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

The samples CSV contains one row per timing sample with columns for all
benchmark parameters plus `label`, `sample_idx`, and `time_ms`.

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

Use `run_benchmarks(test_cases, bench_fn, param_columns)`.

- `test_cases` is a list of dictionaries containing benchmark inputs.
- `param_columns` lists the case fields that should appear in stdout headers
  and CSV output.
- `bench_fn(**case)` must return a list of metric records created by
  `make_metric_record(...)` or `make_forward_backward_metric_records(...)`.

Each metric record represents one benchmark line such as `GEMM Forward`. The
runner prints that line to stdout and expands it into two CSV columns:

- `<label> Time (ms)`
- `<label> <unit>`

For example, a `GEMM Forward` metric with unit `TFLOPS` becomes:

- `GEMM Forward Time (ms)`
- `GEMM Forward TFLOPS`

## Comparing results

Use `compare_results.py` to compare two CSV files from the same benchmark
family:

```bash
python compare_results.py baseline.csv candidate.csv --bench-name GEMM
```

The script auto-detects metric columns, computes speedups for overlapping rows,
and reports rows that exist only in the baseline or only in the candidate.
