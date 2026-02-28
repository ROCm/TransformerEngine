## JAX Fused-Attention Benchmarking
The benchmarking process is split into two stages: *generating* the timing data, and *visualizing* the timing data. The following steps assume you are located in `TransformerEngine/benchmarks/attention` (i.e. where this README is located). First, ensure that you install requirements via `pip install -r requirements.txt`.

### Generate Timing Data
Run the following command to generate timing data. Please use the `-h` flag for details on the available arguments. The output csv, which will later be parsed to generate the interactive visualizations, is generated in the same directory as the script, since that is where the visualization stage expects it.

```bash
python benchmark_attention_jax.py --bench-bwd --fwd-v3 --bwd-v3 -v
```

Note that you can also specify a target HIP device via `HIP_VISIBLE_DEVICES=<device index>` which may be useful in isolating the benchmarks to an unused GPU on a shared machine.

### Profiling JAX CK smallseq path (rocprof)
Single script: profiles fwd and bwd per config, stores each run in its own folder, and writes one combined kernel-timings CSV.

```bash
cd /path/to/TransformerEngine
NVTE_FUSED_ATTN_CK_SMALLSEQ=1 python benchmarks/attention/profile_smallseq_rocm.py [OPTIONS]
```

By default only **bwd** (fwd+bwd) is profiled per config; the bwd run includes both fwd and bwd kernels. Use **`--fwd`** to also run fwd-only profiling per config.

Output under `profiler_outputs_smallseq/`: **`<config_id>_bwd/`** (and **`<config_id>_fwd/`** if `--fwd`), plus **`smallseq_kernel_timings.csv`**. Default config: `config1` (4000,1,2,16,16,128,128, bf16).

Requires AMD GPU and ROCm; the script sets `NVTE_FUSED_ATTN_CK_SMALLSEQ=1` and `XLA_FLAGS=--xla_gpu_graph_level=0` automatically.

### Comparing TE smallseq vs varlen_attn (CK team kernels)
To compare kernel runtimes from TE (fused_attn_smallseq.cpp) with the standalone varlen_attn CK binaries (`varlen_attn/attn_fwd.cpp`, `varlen_attn/attn_bwd.cpp`) using the same configs:

1. **Run TE profile** (if not already done) to populate `profiler_outputs_smallseq/` and `smallseq_kernel_timings.csv`:
   ```bash
   cd /path/to/TransformerEngine
   NVTE_FUSED_ATTN_CK_SMALLSEQ=1 python benchmarks/attention/profile_smallseq_rocm.py
   ```

2. **Build varlen_attn binaries** (same configs as above; requires ROCm clang):
   ```bash
   cd varlen_attn
   /opt/rocm/llvm/bin/clang++ -O3 -x hip --offload-arch=gfx950 -o attn_fwd attn_fwd.cpp
   /opt/rocm/llvm/bin/clang++ -O3 -x hip --offload-arch=gfx950 -o attn_bwd attn_bwd.cpp
   ```

3. **Run comparison** (from repo root):
   ```bash
   python benchmarks/attention/compare_te_varlen_smallseq.py
   ```
   Optional: `--te-dir`, `--varlen-dir`, `-o` for paths. Output: `profiler_outputs_smallseq/te_varlen_comparison.csv` with columns `te_total_fwd_ms`, `te_total_bwd_ms`, `varlen_fwd_ms`, `varlen_bwd_ms`, and ratios. Configs with `s_kv` outside [2,16] or large seq (e.g. 4096/8192) are skipped for varlen.

### Profiling varlen_attn with rocprof (per-kernel runtime)
To get **actual kernel runtimes** (not just end-to-end) for the standalone varlen_attn binaries, run:

```bash
cd /path/to/TransformerEngine
python benchmarks/attention/profile_varlen_attn_rocm.py
```

This runs **rocprof --stats** on `attn_fwd` and `attn_bwd` for each config (4000, 1, s_kv=2,4,...,16, bf16/fp16), parses `results.stats.csv` for the attention kernels (`compute_scores_kernel`, `apply_mask_and_softmax_kernel`, `compute_output_kernel`, `compute_grad_v_kernel`, etc.), and writes **`profiler_outputs_varlen/varlen_kernel_timings.csv`** with per-kernel ms and `total_fwd_ms` / `total_bwd_ms`. Options: `--varlen-dir`, `-o`, `--configs all|bf16|fp16`.

### Generating Interactive Visualization
Simply run `panel serve panel_app.py`. This will launch a web-service on your localhost which displays an interactive visualization app. If launching on a remote server, VS code users will find that their IDE automatically port-forwards the correct ports, and thus they may directly open the link that is printed after running the command. Other users must ensure that their `ssh` into the remote server includes an appropriate port-forwarding (the default port is `5006`).