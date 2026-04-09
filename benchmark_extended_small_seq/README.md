# Small-Sequence Attention Benchmarks

Benchmarking suite for attention kernels at small sequence lengths (sq <= 17, skv <= 17).
Compares **JAX unfused attention** (the reference baseline) against **TE/CK fused attention**
(Transformer Engine with Composable Kernel / aiter backend).

## Background

Sciforium needs optimized attention for small sequence lengths. Four scenarios were identified,
three of which are in scope:

| Scenario | sq | skv | Padding | Self-Attention | Notes |
|----------|----|-----|---------|----------------|-------|
| 1 | <= 16 | <= 16 | Yes (padding + varlen) | Yes | Sweep of sizes; padding deferred |
| 2 | 1 (fixed) | <= 16 | Yes | No | Already done (out of scope here) |
| 3 | 16 | 16 | No padding, fixed len | Yes (sq == skv) | Tile-aligned |
| 4 | 17 | 17 | No padding, fixed len | Yes (sq == skv) | **High priority** -- misaligned tiles |

Scenario 4 is high priority because matrix-multiply instructions use 4x4 or 16x16 tiles,
so 17 requires extra handling in the CK kernel.

Without padding, Scenarios 3 and 4 are just specific points (sq=16 and sq=17) in the
Scenario 1 sweep, so all three run as a **single unified sweep from 1 to 17**.

## File Overview

```
benchmark_extended_small_seq/
├── fa_profiling.py              # Main benchmarking harness (entry point)
├── utils.py                     # Data generation + JAX unfused attention reference
├── run_small_seq_benchmarks.sh  # One-command runner for all scenarios
├── results/                     # CSV output directory (auto-created)
└── README.md                    # This file
```

### `fa_profiling.py`

The main script. Supports two attention backends:

- **`jax`** -- JAX unfused attention: pure JAX implementation using einsum + softmax.
  Known to be competitive at small sequence lengths due to low overhead.
  Backward pass is computed automatically via `jax.vjp`.
- **`te`** -- Transformer Engine fused attention: wraps `transformer_engine.jax.attention.fused_attn`,
  which dispatches to CK/aiter kernels on ROCm.

For each configuration, the script:
1. Generates random Q, K, V, dO tensors
2. Runs all specified kernels (e.g., `jax` and `te`) for that config
3. JIT-compiles each kernel and runs XLA memory analysis
4. Executes warmup iterations, then timed repeats
5. Writes a **single merged CSV row** per config with both JAX and TE timings
   side by side, plus a `speedup_mean` column (TE mean step time / JAX mean step time)

### `utils.py`

Contains three functions:
- `gen_data()` -- generates random input tensors in BSHD layout with segment IDs
- `jax_attention()` -- the JAX unfused attention reference implementation
- `segment_ids_to_cu_seqlens()` -- converts segment IDs to cumulative sequence lengths (for varlen)

### `run_small_seq_benchmarks.sh`

Wrapper script that sweeps sq=skv from 1 to 17. It writes **one CSV per combination**
of layout, batch size, head dimension, and dtype (same names as `fa_profiling.py --dtypes`):

`results/small_seq_sweep_<LAYOUT>_bs_<BATCH>_hd_<DIM>_dt_<DTYPE>_scenario_1_3_4.csv`

Example: `small_seq_sweep_bshd_bs_2048_hd_64_dt_bfloat16_scenario_1_3_4.csv`

Defaults: `BATCH_SIZES=2048 4096`, `DIMS=64 128 256`, `LAYOUTS=bshd`, `DTYPES=bfloat16`
(12 CSV files for a full default run). Add `thd` with `LAYOUTS="bshd thd"` to double the file count;
add more dtypes with `DTYPES="bfloat16 float16"` to multiply by the number of dtypes.

## Quick Start

### Run all scenarios

```bash
cd benchmark_extended_small_seq
bash run_small_seq_benchmarks.sh
```

This sweeps sq=skv from 1 to 17 with default settings (batch sizes 2048 and 4096,
head dimensions 64, 128, and 256, layout `bshd`, dtype `bfloat16`, 32 heads, forward + backward,
25 repeats, 5 warmups) and appends merged rows to one file per `(layout, batch, head_dim, dtype)`

### Run specific sequence lengths only

```bash
# Just Scenarios 3 and 4
bash run_small_seq_benchmarks.sh --seqlens "16 17"

# Just Scenario 4
bash run_small_seq_benchmarks.sh --seqlens "17"
```

### Run with custom parameters

```bash
# Different batch sizes
bash run_small_seq_benchmarks.sh --batch-sizes "512 1024 2048 4096"

# Explicit head dimensions (still one CSV per layout x batch x dim)
bash run_small_seq_benchmarks.sh --dims "64 128 256"

# BSHD and THD layouts (separate CSVs per layout)
bash run_small_seq_benchmarks.sh --layouts "bshd thd"

# More repeats for tighter measurements
bash run_small_seq_benchmarks.sh --repeats 50 --warmups 10

# Forward only
bash run_small_seq_benchmarks.sh --modes "fwd"
```

### Run directly with `fa_profiling.py`

```bash
python fa_profiling.py \
    --kernel-names jax te \
    --seqlens-q 16 --seqlens-kv 16 \
    --batch-sizes 2048 4096 \
    --dtypes bfloat16 \
    --modes fwd bwd \
    --repeats 25 --warmups 5 \
    --csv results/my_test.csv
```

Note: when passing multiple values for `--seqlens-q` and `--seqlens-kv`, the script
generates a Cartesian product. For self-attention (sq == skv), use the shell script
or loop one size at a time.

## Customizing via Environment Variables

```bash
REPEATS=50 BATCH_SIZES="2048 4096" MODES="fwd" bash run_small_seq_benchmarks.sh
KERNELS="jax" bash run_small_seq_benchmarks.sh  # JAX only
SEQLENS="8 16 17" bash run_small_seq_benchmarks.sh  # Subset of sizes
DIMS="128" BATCH_SIZES="2048" bash run_small_seq_benchmarks.sh  # One batch x one dim -> 1 CSV (per layout & dtype)
DTYPES="float16" bash run_small_seq_benchmarks.sh
LAYOUTS="bshd" bash run_small_seq_benchmarks.sh
```

| Flag / Env Var | Default | Description |
|----------------|---------|-------------|
| `--repeats` / `REPEATS` | 25 | Timed iterations per config |
| `--warmups` / `WARMUPS` | 5 | Warmup iterations (excluded from timing) |
| `--batch-sizes` / `BATCH_SIZES` | `2048 4096` | Batch sizes; one CSV per value (with each layout, dim, dtype) |
| `--nheads` / `NHEADS` | 32 | Number of attention heads |
| `--dims` / `DIMS` | `64 128 256` | Head dimension(s); space-separated list |
| `--dtypes` / `DTYPES` | `bfloat16` | Dtype(s): `float16`, `bfloat16`, `float32`, … (see `fa_profiling.py` `DTYPE_MAP`) |
| `--layouts` / `LAYOUTS` | `bshd` | Layout(s); `bshd` and/or `thd` |
| `--modes` / `MODES` | `fwd bwd` | Forward, backward, or both |
| `--kernels` / `KERNELS` | `jax te` | Kernel backends to benchmark |
| `--seqlens` / `SEQLENS` | `1 2 3 ... 16 17` | Sequence lengths to sweep |
