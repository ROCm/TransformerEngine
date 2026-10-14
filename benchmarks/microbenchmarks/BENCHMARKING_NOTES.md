# Microbenchmark Notes

How to read the microbenchmark results (including the TE dashboard, which is fed by
`--dashboard-run`), and why the harness measures the way it does. For how to run the
benchmarks and their options, see [README.md](README.md).

- [Reading the results](#reading-the-results) is for anyone looking at the numbers.
- [Measurement setup](#measurement-setup) and [Known limitations](#known-limitations)
  are for people changing the harness or chasing a surprising result.

## Reading the results

### What a row measures

Each row is one case: model shape x precision x backend (x direction where it applies), timed
in one process on one GPU. Times are per call and are medians over many samples.

| Family | What one call is | Throughput |
|---|---|---|
| `gemm` | `te.Linear` forward, or its backward, under `te.autocast` for the quantized recipes | TFLOPS from GEMM FLOPs: `2*M*N*K` forward, twice that backward (dgrad + wgrad) |
| `grouped_gemm` | `te.GroupedLinear` forward or backward with `B` experts of `M` tokens each (balanced) | TFLOPS from `2*B*M*N*K` forward, twice that backward |
| `casting` | quantize one BF16 tensor, or dequantize one quantized tensor back to BF16 | GB/s from BF16 bytes + quantized bytes (data + block scales) |
| `normalization` | LayerNorm / RMSNorm forward, writing its output in the target precision | GB/s from BF16 input read + output write |

- GEMM rows time the whole TE layer call, not just the GEMM kernel: casts TE does inside
  the layer (e.g. quantizing the input under autocast) and its Python overhead are included,
  but the throughput counts GEMM FLOPs only.
- The `fp8` recipe is per-tensor delayed scaling (HYBRID format); `mxfp8`, `mxfp4` and
  `nvfp4` are TE's block-scaling recipes.
- `B` in grouped GEMM is the number of experts per GPU for a given expert-parallel size
  (e.g. 256 routed experts at EP 32 gives `B = 8`).

### Wall time and kernel time

- **Wall time** is measured on the host around back-to-back calls, so it is throughput
  (calls pipeline on the GPU, as in training), not single-call latency. It includes Python
  dispatch and kernel launch overhead.
- **Kernel time** (`--kernel-profile`) is GPU device time per call from `torch.profiler`:
  the summed device time of the call's kernels, without host overhead.
- **Kernel time well below wall time** means the case is launch- or host-bound (common at
  small M and for higher-overhead backends). For GPU-bound cases the two nearly match.
- **Kernel time slightly above wall time** (a few percent) happens for some GPU-bound cases:
  the profiler itself changes how the GPU runs back-to-back kernels, in either direction.
  For GPU-bound cases, wall time is the more faithful number.
- **Grouped GEMM kernel time is not device-only.** It is the event span between consecutive
  calls on the main stream, because the per-expert path overlaps kernels on several streams
  (a per-kernel sum would double-count). The span includes GPU idle time while the host
  submits the next call, so launch-bound grouped cases read at or slightly above wall time.
  Compare it with dense kernel time only for GPU-bound cases.
- **An empty kernel cell** means the profiled kernels could not be matched to calls for that
  case; the run log names it.

### Forward and backward

The backward row times only the backward: a block of forwards runs untimed, then their
backwards run back to back and are timed, so the host can run ahead as in the forward
timing. No forward time is subtracted.

### Inputs

Activations are rotated through a ring of buffers larger than twice the last-level cache, so
consecutive calls read memory that is not cache-resident (close to a steady-state training
workload). Weights are not rotated.

### Backends and skipped cases

- A backend is selected with TE's environment variables; if the installed TE build doesn't
  honor them, the case is skipped rather than silently timing the default path.
- FlyDSL cases are opt-in (`--run-flydsl`). Each one first runs once with FlyDSL's fallback
  warning enabled and is skipped if FlyDSL hands the shape back to the default GEMM, so every
  `flydsl` row is a FlyDSL measurement. CK and HipKittens grouped GEMM are probed the same way
  for fallbacks to the per-expert path.
- Grouped GEMM backends:
  - `hipblaslt`: per-expert hipBLASLt GEMMs on TE's compute streams (TE's default for BF16,
    FP8 and NVFP4).
  - `hipkittens_per_expert`: dense HipKittens GEMM per expert on the compute streams (TE's
    default for MXFP8 on gfx950).
  - `hipkittens_grouped`: one HipKittens grouped kernel on the main stream (opt-in in TE).
  - `ck_tile`, `triton`: CK and Triton grouped kernels.
- HipKittens needs gfx950 and N and K divisible by 256; MXFP8 and MXFP4 GEMMs need
  32-aligned dimensions (hipBLASLt MXFP8 on gfx950 needs K divisible by 128). Unaligned
  shapes are skipped.
- Known backend bugs on specific shapes are marked xfail (failing configs) or skipped
  (configs that hang); the reasons are in the benchmark files.

### Noise and outliers

- GPU-bound cases typically repeat within a few percent between runs. Launch-bound cases
  depend on host speed and are more sensitive, which is why the harness pins CPUs (below).
- Occasionally a stretch of cases runs slower for reasons outside the harness. Two patterns:
  - host-side: wall time up (often ~2x), kernel time unchanged;
  - GPU-side: wall and kernel time both up by the same factor, uniformly for the whole case.
- To judge a change, first check whether kernel time moved with wall time (GPU-side) or not
  (host-side).

## Measurement setup

### Wall timing

- Forward and single-op cases use `torch.utils.benchmark.Timer` (`blocked_autorange` for
  casting, `adaptive_autorange` otherwise) with a 0.2 s minimum run time, and report the
  median.
- Each case first makes one untimed call. The Timer chooses its block size (calls per
  sample) from its first timed calls; a slow one-off first call (lazy kernel load, JIT
  compilation, algorithm selection, autotuning) can make it pick a block of one call, so
  every sample includes a device sync and measures latency instead of throughput. Without
  the one-off, blocks last at least several milliseconds and the per-block sync cost is
  negligible.

### Backward timing

- Per block, N forwards (N up to 20, fewer if their saved state would exceed 16 GB) run
  untimed, then the N backwards run back to back and are timed. The first block is warmup;
  timing continues for at least 15 blocks and 0.2 s. GC is off while timing (N live
  autograd graphs make collections expensive).
- Rejected alternatives:
  - `(forward + backward) - forward`: its median is stable, but individual runs produced
    occasional large outliers.
  - Synchronizing around each backward: repeatable, but measures latency plus a fixed sync
    cost rather than pipelined throughput, and reads high for GPU-bound shapes.
  - Repeating the backward on one graph (`retain_graph`): TE frees saved state in backward,
    so this fails for `Linear` and `GroupedLinear`.
- The short blocks leave a small fixed per-block cost in the result (around 1% for the
  fastest backward cases). It is constant, so trends are unaffected.

### Kernel time

- `--kernel-profile` runs a separate pass after the wall timing: 100 warmup calls, then one
  profiler session of 100 calls (for backward, sessions of one backward block each). The
  kernel list is split per call by matching the kernel sequence of a single profiled call;
  only fully matched calls count, and the median is reported.
- One long session rather than many short ones: short sessions let the GPU clock down
  between them and bias large GEMMs high. A brief busy loop before each session brings the
  clocks back up.
- The profiler on ROCm can drop kernel records that are still in flight when a session
  stops; a short sleep before stopping avoids that. Turning off the profiler's queue
  interposition also avoids it but adds launch overhead to every call, so it isn't used.
- Grouped GEMM uses the event-span method described above. `NVTE_MICROBENCH_KERNEL_TIME`
  (`profiler` or `event`) forces one method for all families, for comparisons.

### Input rotation

The ring is sized to the `--rotating` budget, or by default to twice a 256 MB last-level
cache (the AMD Infinity Cache size on gfx942/gfx950), and capped at 1000 buffers.

### CPU pinning

- Launch-bound cases depend on CPU clocks. With the default `schedutil` governor, the wall
  time of launch-bound backward cases (which run on PyTorch's autograd thread) varied up to
  2x between runs.
- The harness therefore pins the main thread to 4 physical cores on the GPU's NUMA node
  (offset by GPU id, so runs on other GPUs use other cores) and sets them to the
  `performance` governor. Threads the main thread creates later, such as the autograd
  thread, inherit the cores.
- HIP/ROCr helper threads that exist after HIP init go to those cores' SMT siblings. One of
  them polls events continuously, and the multi-stream grouped GEMM path ran about 1.6x
  slower when it landed on another CCD or socket than the main thread (and slower still
  when it shared the main thread's cores).
- The previous governors are restored at the end of the session, or by a small watchdog
  process if the benchmark is killed or crashes. `--no-cpu-pin` disables all of this.

### HW queues

- `GPU_MAX_HW_QUEUES` defaults to 2 (as in Megatron's launch scripts); an explicit value
  overrides it.
- Kernels on the main stream slow down while a cross-queue event wait is pending on a HW
  queue that shares the main queue's hardware pipe. With HIP's default of 4 queues, TE's
  high-priority compute streams get an extra queue that lands on the main queue's pipe, and
  the per-expert grouped GEMM path ran up to about 2x slower and bimodally. With 2 queues it
  is stable.

### Isolation

- TE's FP8 global state is reset after each test: modules from earlier cases otherwise stay
  registered and slow down every later autocast exit.
- At startup the harness warns (or, with `--abort-on-gpu-interference`, aborts) if another
  process is using the benchmark GPU. The direct-run shim enters pytest before torch
  touches the GPU, so the benchmark's own process is not reported.

### Dashboard runs

`--dashboard-run` writes all families of a run into
`results/<arch>_<host>_gpu<id>_<pci>/<date>_<commit>/`:

- `benchmark_<family>.csv`, each row tagged with the run timestamp, commit SHA and commit
  date;
- `run_info.txt` with the host, GPU, software versions, CPU pinning and HW-queue setting;
- `samples/` with per-sample times when `--csv-samples` is given.

## Known limitations

- **Profiler perturbation:** kernel time of GPU-bound cases can differ from the unprofiled
  run by a few percent in either direction.
- **Grouped GEMM kernel time** includes GPU idle time for launch-bound cases (see above). A
  device-busy measurement (the union of kernel intervals per call) would fix this; not
  implemented.
- **No per-case GPU clock or throttle record:** GPU-side transients are found only by
  comparing with other runs.
- **Configuration-specific results:** the numbers hold for pinned CPUs and
  `GPU_MAX_HW_QUEUES=2`. Without them, launch-bound and per-expert grouped GEMM cases can
  be up to about 2x slower.
- **Single measurement per run:** each case is timed once per run, in one process, so
  process-level effects (e.g. memory placement) are not averaged out.
- **Upstream issues:** some results are shaped by issues outside TE that are worked around
  or marked in the harness: the ROCm runtime's cross-queue wait slowdown (HW-queue setting),
  profiler record loss at session stop (sleep before stop), and known backend failures or
  hangs on specific shapes (xfail/skip).
