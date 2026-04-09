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

## Prerequisites

- ROCm-capable GPU(s)
- Python 3.11+
- JAX with ROCm support
- Transformer Engine (`transformer_engine`) built from the `dev` branch
- Python packages: `einops`, `numpy`, `tqdm`

## File Overview

```
benchmark_extended_small_seq/
├── fa_profiling.py              # Main benchmarking harness (entry point)
├── utils.py                     # Data generation + JAX unfused attention reference
├── jax_unfused_attn.py          # Standalone copy of JAX attention (reference only)
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
   side by side, plus a `speedup_mean` column (TE mean / JAX mean)

### `utils.py`

Contains three functions:
- `gen_data()` -- generates random input tensors in BSHD layout with segment IDs
- `jax_attention()` -- the JAX unfused attention reference implementation
- `segment_ids_to_cu_seqlens()` -- converts segment IDs to cumulative sequence lengths (for varlen)

### `run_small_seq_benchmarks.sh`

Wrapper script that sweeps sq=skv from 1 to 17, grouped by batch size
(all batch_size=2048 first, then batch_size=4096). Produces a single CSV file:
`results/small_seq_sweep_bshd_scenario_1_3_4.csv`.

## Quick Start

### Run all scenarios

```bash
cd benchmark_extended_small_seq
bash run_small_seq_benchmarks.sh
```

This sweeps sq=skv from 1 to 17 with default settings (batch sizes 2048/4096,
32 heads, dim 128, bfloat16, forward + backward, 25 repeats, 5 warmups) and writes
merged results to `results/small_seq_sweep_bshd_scenario_1_3_4.csv`.

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
```

| Flag / Env Var | Default | Description |
|----------------|---------|-------------|
| `--repeats` / `REPEATS` | 25 | Timed iterations per config |
| `--warmups` / `WARMUPS` | 5 | Warmup iterations (excluded from timing) |
| `--batch-sizes` / `BATCH_SIZES` | `2048 4096` | Batch sizes to benchmark |
| `--nheads` / `NHEADS` | 32 | Number of attention heads |
| `--dims` / `DIMS` | 128 | Head dimension |
| `--modes` / `MODES` | `fwd bwd` | Forward, backward, or both |
| `--kernels` / `KERNELS` | `jax te` | Kernel backends to benchmark |
| `--seqlens` / `SEQLENS` | `1 2 3 ... 16 17` | Sequence lengths to sweep |

## `fa_profiling.py` CLI Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--kernel-names` | str list | *required* | Backends: `jax`, `te` |
| `--seqlens-q` | int list | `[2048]` | Query sequence lengths |
| `--seqlens-kv` | int list | `[2048]` | Key/value sequence lengths |
| `--batch-sizes` | int list | `[1]` | Batch sizes |
| `--nheads` | int list | `[32]` | Number of attention heads |
| `--dims` | int list | `[128]` | Head dimension |
| `--gqa-ratios` | int list | `[1]` | GQA ratios (1 = MHA, >1 = GQA) |
| `--dtypes` | str list | `[bfloat16]` | Data types: `float16`, `bfloat16`, `float32` |
| `--modes` | str list | `[fwd]` | `fwd`, `bwd`, or both |
| `--layouts` | str list | `[bshd]` | Tensor layout: `bshd` or `thd` |
| `--nr-segments` | int list | `[1]` | Segments per sequence (1 = no padding) |
| `--window-sizes` | int list | `[-1]` | Sliding window size (-1 = disabled) |
| `--non-causal` | flag | `False` | Disable causal masking |
| `--repeats` | int | `25` | Timed iterations |
| `--warmups` | int | `3` | Warmup iterations |
| `--csv` | str | `None` | Output CSV path (append mode) |
| `--name-suffix` | str | `""` | Suffix appended to kernel name in output |
| `--tensorboard-logdir` | str | `None` | Directory for TensorBoard trace |

## Output Format

Each CSV row contains results for **both kernels** on the same config, with a speedup column:

| Column | Description |
|--------|-------------|
| `mode` | `fwd` or `bwd` |
| `layout` | `bshd` or `thd` |
| `dtype` | Data type used |
| `batch_size` | Batch size |
| `seqlen_q` / `seqlen_kv` | Sequence lengths |
| `nheads` / `dim` / `gqa_ratio` | Model config |
| `causal` | Whether causal masking was applied |
| `num_segments` | Number of segments (1 = no padding) |
| `sliding_window_size` | Window size (-1 = disabled) |
| `jax_min_steptime_ms` | JAX minimum step time |
| `jax_median_steptime_ms` | JAX median step time |
| `jax_mean_steptime_ms` | JAX mean step time |
| `jax_q1_steptime_ms` / `jax_q3_steptime_ms` | JAX 25th / 75th percentile |
| `jax_memory_total_mib` | JAX total memory |
| `te_min_steptime_ms` | TE minimum step time |
| `te_median_steptime_ms` | TE median step time |
| `te_mean_steptime_ms` | TE mean step time |
| `te_q1_steptime_ms` / `te_q3_steptime_ms` | TE 25th / 75th percentile |
| `te_memory_total_mib` | TE total memory |
| `speedup_mean` | TE mean / JAX mean (how many times faster JAX is) |
