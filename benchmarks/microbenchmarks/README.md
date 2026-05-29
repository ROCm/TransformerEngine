# Transformer Engine Microbenchmarks

This directory contains lightweight Python microbenchmarks for selected
Transformer Engine kernels and helper scripts for comparing benchmark CSVs.

## Benchmarks

- `benchmark_gemm.py`: dense BF16 GEMM benchmark
- `benchmark_gemm_fp8.py`: dense FP8 GEMM benchmark using `fp8_autocast`
- `benchmark_grouped_gemm.py`: grouped GEMM benchmark for MoE-style shapes
- `benchmark_casting.py`: BF16 `<->` FP8 casting benchmark
- `benchmark_normalization.py`: LayerNorm and RMSNorm benchmark

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
