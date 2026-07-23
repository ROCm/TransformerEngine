# Transformer Engine Microbenchmarks

This directory contains lightweight Python microbenchmarks for selected
Transformer Engine kernels and helper scripts for comparing benchmark CSVs.
Timing is powered by [`usv`](https://github.com/matthiasdiener/usv) (install it
in your environment first, e.g. `pip install -e /path/to/usv`).

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

## Timing options (usv)

Each benchmark prints **one line per run** and finishes with a terminal-only
min / median / max throughput summary across all cases. Timing is delegated to
`usv`, so the following optional flags are available on every benchmark (all off
by default, so the default behavior is unchanged):

| Flag | Meaning |
| --- | --- |
| `--interleave` | Sample a run's callables (e.g. fwd / fwd+bwd) round-robin, spreading time-correlated noise across them. |
| `--warmup N` | Untimed warmup iterations per benchmark. |
| `--iters N` | Timed samples per benchmark. If unset, sample until ~0.2 s of kernel time elapses (autorange-like). |
| `--cache-flush` | Flush an L2-sized buffer before each sample (cold-cache timing). |
| `--cudagraph` | Capture each callable into a CUDA/HIP graph and time replays (removes launch overhead). |
| `--rotate` | Rotate inputs through an L2-sized ring of buffers (defeats L2 residency). |
| `--cooldown SECONDS` | Idle sleep after each benchmark to let the GPU cool. |
| `--monitor` | Sample `rocm-smi` during timing and warn if the GPU clock drifts (AMD). |
| `--timeout SECONDS` | Abort a benchmark if timing exceeds this many seconds (GPU-hang guard). |

```bash
python benchmark_gemm.py --interleave --rotate
python benchmark_gemm.py --iters 200 --cache-flush --monitor
```

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
runner prints all of a run's metrics on a single line and expands each into two
CSV columns:

- `<label> Time (ms)`
- `<label> <unit>`

To time the callables, `bench_fn` uses `time_funcs({name: callable})` (which
honors the usv flags above and enables `--interleave`); `make_input(shape,
dtype, ...)` returns a rotation-aware input factory that respects `--rotate`.

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
