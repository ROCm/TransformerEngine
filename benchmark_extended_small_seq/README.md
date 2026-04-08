# Small-Sequence Attention Benchmarks

Benchmarking suite for attention kernels at small sequence lengths (sq <= 17, skv <= 17).
Compares **JAX unfused attention** (the reference baseline) against **TE/CK fused attention**
(Transformer Engine with Composable Kernel / aiter backend).

## Background

| Scenario | sq | skv | Padding | Self-Attention | Notes |
|----------|----|-----|---------|----------------|-------|
| 1 | <= 16 | <= 16 | Yes (padding + varlen) | Yes | Sweep of sizes; padding deferred |
| 2 | 1 (fixed) | <= 16 | Yes | No | Already done (out of scope here) |
| 3 | 16 | 16 | No padding, fixed len | Yes (sq == skv) | Tile-aligned |
| 4 | 17 | 17 | No padding, fixed len | Yes (sq == skv) | **High priority** -- misaligned tiles |

## File Overview

```
benchmark_extended_small_seq/
├── fa_profiling.py          # Main benchmarking harness (entry point)
├── utils.py                 # Data generation + JAX unfused attention reference
├── run_small_seq_benchmarks.sh  # One-command runner for all 3 scenarios
├── results/                 # CSV output directory (auto-created)
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
2. JIT-compiles the attention function
3. Runs XLA memory analysis
4. Executes warmup iterations, then timed repeats
5. Reports min/median/mean/q1/q3 step times and memory usage

### `utils.py`

Contains three functions:
- `gen_data()` -- generates random input tensors in BSHD layout with segment IDs
- `jax_attention()` -- the JAX unfused attention reference implementation
- `segment_ids_to_cu_seqlens()` -- converts segment IDs to cumulative sequence lengths (for varlen)

### `run_small_seq_benchmarks.sh`

Wrapper script that runs all three scenarios sequentially, producing timestamped CSV files
in the `results/` directory.

## Quick Start

### Run all scenarios at once

```bash
cd benchmark_extended_small_seq
bash run_small_seq_benchmarks.sh
```

This runs Scenarios 3, 4, and 1 with default settings (batch sizes 1/4/8, 32 heads,
dim 128, bfloat16, forward + backward, 25 repeats, 5 warmups) and writes results to
`results/`.

### Run individual scenarios

**Scenario 3** -- 16x16, causal, no padding:

```bash
python fa_profiling.py \
    --kernel-names jax te \
    --seqlens-q 16 --seqlens-kv 16 \
    --batch-sizes 1 4 8 \
    --modes fwd bwd \
    --repeats 25 --warmups 5 \
    --csv results/scenario3.csv
```

**Scenario 4** -- 17x17, causal, no padding:

```bash
python fa_profiling.py \
    --kernel-names jax te \
    --seqlens-q 17 --seqlens-kv 17 \
    --batch-sizes 1 4 8 \
    --modes fwd bwd \
    --repeats 25 --warmups 5 \
    --csv results/scenario4.csv
```

**Scenario 1** -- sweep sq=skv from 1 to 16:

```bash
for SEQ in $(seq 1 16); do
    python fa_profiling.py \
        --kernel-names jax te \
        --seqlens-q $SEQ --seqlens-kv $SEQ \
        --batch-sizes 1 4 8 \
        --modes fwd bwd \
        --repeats 25 --warmups 5 \
        --csv results/scenario1.csv
done
```

Note: Scenario 1 requires a loop because `fa_profiling.py` generates a Cartesian product
of `--seqlens-q` and `--seqlens-kv`. Passing both as lists would benchmark all cross-attention
combinations (e.g., sq=4, skv=12), not just self-attention (sq == skv).

### Run a single kernel only

Benchmark just the JAX baseline:

```bash
python fa_profiling.py \
    --kernel-names jax \
    --seqlens-q 16 --seqlens-kv 16 \
    --batch-sizes 1 \
    --modes fwd \
    --csv results/jax_only.csv
```

Benchmark just the TE/CK kernel:

```bash
python fa_profiling.py \
    --kernel-names te \
    --seqlens-q 17 --seqlens-kv 17 \
    --batch-sizes 1 \
    --modes fwd bwd \
    --csv results/te_only.csv
```