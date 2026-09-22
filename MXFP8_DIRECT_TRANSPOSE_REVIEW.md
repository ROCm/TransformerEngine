<!-- Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved. -->
<!-- License for AMD contributions = MIT. See LICENSE for more information -->

# Review guide — deriving wgrad's column-scaled dY inside the fused MXFP8 dgrad kernel

Walkthrough of the scale-aware direct transpose: what changed, in the order it makes sense to
read it, and what to push on. The design rationale, the loss characterisation and the measured
numbers live in `MXFP8_DIRECT_TRANSPOSE_SCOPE.md`; this document is the tour, and carries the
headline measurement in "Results" below.

Companions: `MXFP8_WGRAD_AG_HANDOFF.md` (why dY is needed twice at all),
`MXFP8_WGRAD_AG_REVIEW.md` (the second all-gather this work makes optional, `6eb5bff1c`),
`../mxfp8_direct_transpose.py` (the host derivation everything here is built on) and
`../mxfp8_direct_transpose_check.cpp` (the exhaustive check of the shipped bit math).

13 files, +1191 / −139, plus one new header and one new host check.

---

## The short version

Row-parallel backward needs dY twice: row-scaled for dgrad and column-scaled for wgrad. MXFP8
cannot convert between them, so the kernel carried a **second all-gather** to fetch a separately
quantized column-scaled copy.

It no longer has to. TE's MXFP8 column-wise form has the **same `[rows, cols]` data layout** as
the row-wise one — only the scale blocking changes, `[rows, cols/32]` → `[rows/32, cols]` — so the
second copy is derivable from the first by exponent arithmetic alone (arXiv:2511.02302
Algorithm 1 at MXFP8's block size of 32). No bytes move; each element is re-encoded against its
new group's scale.

The derivation now runs **inside the all-gather copy**, on bytes already in registers, so the
gathered tensor is never read twice. It is faster than the all-gather it replaces, and the
all-gather remains selectable.

---

## Why it is simpler than the paper

Three facts, in the order they mattered:

1. **There is no data transpose.** `MXFP8Quantizer.get_columnwise_shape()` returns
   `rowwise_data_shape` verbatim (`pytorch/tensor/mxfp8_tensor.py:267`). The paper's Step 1
   ("move the bytes") disappears; Steps 2–3 — pick a target scale per 32×32 tile, shift each
   element's exponent onto it — are the whole job.
2. **The target scale needs no element scan.** It is the max of the tile's 32 *row scales*, i.e.
   32 bytes out of an array 32× smaller than the data. This is what later makes a streaming
   implementation possible at all.
3. **Overflow is structurally impossible.** The target is a max, so every shift is downward.
   Underflow is the sole failure mode, and it is the price of that.

### The element format is E4M3, not E5M2

Worth stating because it was assumed otherwise at the start. TE does route backward tensors
through the "bwd" dtype slot (`pytorch/quantization.py:1645`), but `MXFP8BlockScaling.fp8_format`
**defaults to `Format.E4M3`** and `__post_init__` asserts pure E5M2 out
(`common/recipe/__init__.py:395,401`). E5M2 grads happen only under `Format.HYBRID`, and every
distributed test builds a bare `MXFP8BlockScaling()`. **CI therefore runs the lossier format.**
Both are implemented — the kernel already carries the gathered operand's format as its `CBSZ`
template parameter — but E4M3 is the one to watch.

Elements preserved bit-exactly, from the host script: 100% / 99.988% / 99.902% / 97.589% for
E4M3 across all-normal, gaussian, ±3-binade and 12-binade row-scale spreads; 100% for E5M2 on all
four.

---

## Reading order

Bottom-up. Each section says what to check, not just what changed.

### 1. The element rule — `cdna4/overlap/mxfp8_direct_transpose.cuh` (new, ~200 lines)

Start here; everything else is plumbing around it.

`Fmt<CODE>` traits (`CODE` matches the kernels' `CBSZ`/`BLGP` codes: 0 = E4M3, 1 = E5M2),
`rne_shift_right`, `mx_rescale<CODE>` (per element) and `mx_rescale_dword<CODE>` /
`mx_rescale_dword_fast<CODE>` (four elements at once).

**Check the two cases the reference script does not handle**, both of which are here and are not
in `mxfp8_direct_transpose.py`:
- *Reserved inf/NaN patterns.* The script shifts them like any other value, so an E5M2 `0x7C`
  would silently become finite and at `k=0` would trip the script's own exponent-range assert.
  `mx_rescale` passes them through. A gathered tensor is not ours to trust.
- *`0xFF` scale bytes.* `ptx::float_to_e8m0` emits `0xFF` for NaN (`util/ptx_hip.cuh:417`). One
  such scale makes a band's `T'` `0xFF` and flushes the band. This propagates rather than being
  clamped — a NaN scale means the layer is already NaN — but it is a failure mode that did not
  exist while the bytes were merely copied.

**Check the packed path's selection condition.** It fires per dword iff **all four** bytes are
normal, stay normal, and are not reserved; then the whole re-encode is `x - (k << MAN_BITS)`,
one 32-bit subtract. Two subtleties worth your eye:
- a 32-bit shift mixes the neighbouring byte's bits into the lane above, but an FP8 byte is
  `1 + EXP_BITS + MAN_BITS == 8`, so the **sign bit sits between** the exponent field and the
  neighbour's bits and `EXP_MASK` stops clear of them in both formats;
- `k` is a difference of two E8M0 bytes so it reaches 255, and is clamped to `EXP_MASK` before
  the lane arithmetic. The clamp changes no answer (`E <= EXP_MASK`, so a larger `k` admits no
  byte) but keeps every lane in range.

**The file compiles on the host** (`HK_MXT_FN` degrades to `inline`), which is what makes the
exhaustive check possible. It carries a commented `hip/hip_runtime.h` marker per the hipify
convention.

### 2. The kernel-side drivers — `cdna4/overlap/overlap_common.cuh` (+473)

Three implementations, all additive. **Check first that they are additive**: `gather_copy_wg`,
`gather_peer_tile`, `gather_all`, `gather_peer_tile_plus_scales`, `gather_all_plus_scales`,
`pack_tile_scales_from` and `store_c_tile` are byte-identical to HEAD, so the bf16 AG, TN and RS
kernels cannot have moved.

- **`mx_stream_band` / `gather_peer_tile_colwise` / `gather_all_colwise` — the in-flight path.**
  Re-encodes inside the gather copy. Rests on: each gather sub-block already owns **exactly one
  32-row band**, because `sub_bytes = align16(ceil(256*K / gath_wg))` is exactly `32*K` at
  `gath_wg == 8` for any K that is a multiple of 32 — and `run_mxfp8` passes the `GATH_WG`
  constant, not the `gath_wg_nn()` heuristic. **Check `mx_can_stream_colwise`**: it re-verifies
  that, the tile layout, `K % 32`, and the LDS budget, and anything failing falls back to the
  pool rather than mis-striding.
- **`MxColwisePool` / `mx_colwise_pool_step` / `mx_colwise_pool_drain` — the deferred path.**
  Bands claimed from a shared counter, drained one per scheduling step between GEMM tiles.
  Always handles our own rank's chunk, which nothing gathers and therefore nothing can stream.
- **`mx_colwise_band` / `mx_colwise_all` — the original serial pass.** Kept as a baseline and a
  safety net.

**Check the store ordering in `mx_stream_band`, and do not tidy it.** The row-wise store is
issued *before* the re-encode touches the register, so the row-wise stream stays ahead of the
fence preceding `AG_PUBLISH`. The arrival flag covers only the row-wise bytes — nothing waits on
the column-wise ones until the kernel retires — which is the whole reason in-flight is cheap
enough to be worth doing.

**Known limit, stated in the code:** the column-wise *stores* could not be moved past the publish.
Holding a band across it needs 64–128 VGPRs/thread from a runtime-trip-count loop. So in-flight
does pay a second store stream inside the publish window, and depends on the gather having spare
write bandwidth. `deferred` exists as the measurable control for exactly that; the measurement
(SCOPE §8) says the bandwidth is there.

### 3. The kernel — `cdna4/overlap/fused_ag_gemm_nn.cuh` (+121)

The post-primary-round block is now a three-way branch: `gather_all` (second all-gather),
`gather_all_colwise` (in-flight), or nothing-here-plus-pool (deferred/phase). `CBSZ` is the A
slot's format and for NN the A slot *is* the gathered tensor, so the derivation gets its element
format for free.

**Check the pool step's placement** (`:790-798`): after the barrier that ends the previous tile,
before the next tile's `G::load`, because it stages through `As[0][0]`, which is dead at exactly
that point. If this is wrong it corrupts the **GEMM output**, not the column-wise bytes — so the
numerics check is what guards it, not the byte check.

### 4. Contract and host — `comm_gemm.h`, `comm_gemm.cpp`, `rocm_comm_gemm_overlap.cpp`

`KittensAuxAgRegion` → `KittensAuxColwiseRegion`, carrying a `mode` plus the gather-only peer and
arrival fields (zero in transpose mode). `hk_fused_ag_gemm` keeps the shard-shape validation —
the band addressing rests on it — and splits the rest: transpose mode checks 16-byte alignment
(the derivation writes whole vectors), gather mode checks the region is registered.

**Check that the gather path is byte-for-byte what it was**: peer table, aux `ag_ready_kernel`,
aux `gather_scales`, and the `_ag_signal_base` advance all still happen, now under `if (gather)`.

### 5. Python — `module/base.py`, `module/linear.py`, `module/layernorm_mlp.py`

`userbuffers_view_for_derived_columnwise()` returns an `MXFP8TensorStorage` view over the region
the kernel fills: no quantize, no copy, no collective. `linear.py` branches on
`ag_wgrad_copy_mode()`; gather mode still stages this rank's shard *before* the dgrad GEMM
launches, exactly as at HEAD.

**Check the ordering requirement is gone in transpose mode.** The previous review flagged the
staging order as "most likely to be subtly wrong"; with nothing staged and no peer reading the
region, that hazard does not exist in the default path.

### 6. Tests — `run_layer_with_overlap.py` (+163), `test_rocm_fused_overlap.py` (+78)

`--check-wgrad-colwise` compares the region the kernel wrote against a torch transpose of the
row-wise region it read: all `tp` chunks, data and scales, byte for byte. `--grad-scale-binades`
exists for the reason in §"The trap" below.

---

## Mode selection

| variable | values | meaning |
|---|---|---|
| `NVTE_AG_WGRAD_COPY` | `transpose` (default), `gather` | how the copy is produced |
| `NVTE_AG_WGRAD_TRANSPOSE` | `inflight` (default), `deferred`, `phase` | where the derivation runs |
| `NVTE_AG_NO_WGRAD_COPY` | set = skip | measurement only; numerically WRONG by construction |

`NVTE_AG_WGRAD_COPY` is read in **two** places — `kittens_aux_colwise_mode()` in
`gemm/kittens/comm_gemm.h` and `ag_wgrad_copy_mode()` in `pytorch/module/base.py`. **Nothing
cross-checks them.** If they disagreed, Python would stage for one mode while the kernel ran the
other: silently wrong, no error. Both latch within microseconds of each other on the first
backward pass of one process, so divergence needs `os.environ` mutated in between — but the
robust fix is to stop coupling through the environment and pass the mode down through
`general_gemm`, or at minimum expose the C++ value through pybind and assert equality once.
Neither is done. **This is the thing in this change most worth arguing about.**

---

## Verification, in four layers

**1. The element rule, exhaustively, on the host.** `mxfp8_direct_transpose_check.cpp` compiles
the shipped header on the host and checks it against a reference encoder sharing no logic with it
(a sorted search over the format's value grid with round-to-nearest-even, as the python script's
`encode` does). All 256 bit patterns × `k ∈ [0,40]` × both formats, plus the paper's invariant
directly (normal-range elements come back bit-identical) and reserved-pattern pass-through.

The packed path is checked harder, because it is selected per *dword*: all 256×256 ordered byte
pairs in six lane arrangements, plus 4-tuples over an 18-byte alphabet, each across the full
`k ∈ [0,255]`. It asserts the fast path fires **exactly** when a predicate written from the format
definition says it should, so a merely-conservative fast path fails too, and asserts non-vacuity
(fails if it never built a mixed dword, never took the fast path, or always took it).
**255M dwords, PASS, 3.8 s.** Seven deliberate mutations of the SWAR logic all produce failures.

**2. The plumbing, byte for byte, on the GPU.** The host check says nothing about the driver —
the `k` derivation, the tile-max reduction, LDS staging, `uint4` addressing. `--check-wgrad-colwise`
covers those. Result at tp=4: 3,145,728 data and 98,304 scale bytes match, 83,452 elements
rescaled.

**3. Three implementations agreeing.** `inflight`, `deferred` and `phase` produce **byte-identical**
output and all pass the gradient checks. Three independent paths agreeing bit-for-bit is stronger
evidence than any one passing.

**4. End to end.** 22 tests: both wgrad modes, tp 4 and 8, plus `K=16384` interleaved coverage.
`weight.grad` lands at rel. err 0.0071 against a non-overlapped reference (tol 0.125) — real
double-quantization error, as expected, since the derived copy re-blocks the row-wise
quantization rather than re-quantizing from high precision.

**The negative control.** With `NVTE_AG_NO_WGRAD_COPY=1` the region is untouched, every byte
differs, and `weight.grad` fails as a zero tensor. That is what proves the wgrad GEMM actually
consumes what the kernel writes.

### The trap: a constant grad output proves nothing

`run_fwd_bwd` used `loss = out.sum()`, whose gradient is **identically 1.0**. Every MXFP8 block
then lands on the same scale, every shift is zero, and the derivation degenerates to a copy. The
pre-existing row-parallel MXFP8 test passed in exactly that state — it exercised the plumbing and
never rescaled a single element.

`--grad-scale-binades=N` weights the loss per token by a random power of two **times a normal
factor**. The power of two alone spreads the scales but leaves every gradient exactly
representable, so the rescale would run without ever rounding; the normal factor is what makes it
round. The test asserts the reported non-zero-shift count exceeds zero, so a future change that
flattens the scales fails loudly instead of passing vacuously.

### The gold ISA gate

`MXFP8_AG_REFACTOR.md` §3.1 makes `cdna4/mxfp8_gemm.cpp`'s device ISA the binding criterion. Its
**945-file dependency closure contains none of the files touched here** (verified with `-M`), so
its ISA cannot have changed. Note the instruction counts recorded in that document are stale
relative to current HEAD for unrelated reasons — `v_mfma` still 84096, but `s_waitcnt` is 107686
against the recorded 119866 — so do not use them as a gate without re-baselining.

---

## Results

`overlap_bench` config 3 (AG+GEMM NN, row dgrad), MXFP8, TP=8, 24 Llama-3-dense shapes,
`WARM=10 ITERS=25 REPS=2`, both windows CLEAN (`kfd_max=8`). Baseline is **TE's non-overlapped
path in TE's own ordering, timed as one region** — `base.py:1843` via `linear.py:932`, AG then
GEMM, serial, one stream — not `collective + GEMM`. `speedup = baseline exposed / fused`.

| arm | geomean | min | max | wins |
|---|---|---|---|---|
| two all-gathers (`NVTE_AG_WGRAD_COPY=gather`) | **1.277x** | 1.121 | 1.440 | 24/24 |
| direct transpose (`=transpose`, `NVTE_AG_WGRAD_TRANSPOSE=inflight`) | **1.387x** | 1.069 | 1.705 | 24/24 |

Head to head, per shape: **inflight / gather geomean 1.085**, min 0.945, max 1.361, better on
16 of 24 shapes. Both arms beat the baseline on every shape.

**The check that makes these quotable:** the `gather` arm reproduces **1.277x** against the
**1.289x** recorded for this config in `overlap_bench/MXFP8_AG_RESULTS.md` from an earlier
build. That is within run-to-run variation, so the harness, the build and the shape list are
behaving as they did before, and the 1.387x sits on the same footing rather than on a
methodology change.

`NVTE_AG_DIAG=1` was on for both runs and confirms each arm ran the mode it was asked for
(`wgrad_copy=gather` / `wgrad_copy=transpose/inflight`) rather than silently falling back to a
default — which is a real hazard here, see the note on `run.sh` env forwarding below.

### Two harness traps, both of which produce believable wrong numbers

1. **`run.sh` forwards only the env vars it names.** Open MPI passes no arbitrary environment to
   the ranks, so a variable not listed in its `-x` loop is *silently absent inside the rank* and
   the kernel takes its default. The loop still named the pre-rename toggles, so both arms would
   have run `transpose/inflight` and reported the same number twice. The list now carries
   `NVTE_AG_WGRAD_COPY`, `NVTE_AG_WGRAD_TRANSPOSE` and `NVTE_AG_NO_WGRAD_COPY`.
2. **`overlap_bench` embeds the contract struct.** It builds `KittensAuxAgRegion` itself, so the
   rename to `KittensAuxColwiseRegion` plus the added `mode` field made the existing binary an
   ABI mismatch against the rebuilt library. It must be rebuilt whenever `comm_gemm.h` changes:
   `TE_ROOT=<this tree> OMPI_HOME=/usr make`.

---

## Decisions worth arguing about

1. **Tile-max rather than an exact per-column target.** The paper's shortcut shares one scale
   across a tile's 32 columns. An exact per-column max is tighter, nearly free in this layout
   (a thread owns whole columns, so the reduction is lane-local), and would eliminate almost all
   remaining underflow — but it makes shifts signed (needing a renormalize-up branch), removes
   overflow-impossible-by-construction, must match `float_to_e8m0`'s round-**up** rule rather than
   the paper's, and the host oracle would have to be extended first. Specified in SCOPE §5 and at
   the decision point in the code; **not implemented**.
2. **Keeping the second all-gather.** It costs two dead kernel parameters and a branch. It is
   also what made the comparison a single-build, single-process-image measurement.
3. **Three transpose implementations.** `phase` is dead weight once `inflight` is trusted, but it
   is the fallback for geometries in-flight cannot cover and the baseline the other two were
   proven against.
4. **Env-var coupling between C++ and Python.** See the mode-selection section. The cheapest
   thing in this change that could silently produce wrong gradients.

---

## Corrections made along the way

Recorded because each cost time and would cost it again:

- **The gather path was deleted before being restored.** The first implementation removed it
  outright; it was reinstated as a runtime option on request. Keeping it turned out to improve the
  measurement, not just the optionality.
- **A 5-iteration smoke test was badly wrong.** It put in-flight at 40% *slower* than the
  all-gather on `65536×768×4096`. The 25-iteration sweep puts that shape at **1.219 the other
  way**. Smoke numbers at that iteration count are not evidence and were reported as though they
  were.
- **A running bash script was edited.** `sweep24_colwise.sh` was modified mid-run; bash re-read it
  at the new byte offset, re-executed a line and died. The data survived (deduped) but the run
  had no terminator.
- **`pkill -f torchrun` matched its own shell.** Kill by PID.
- **A stale hipified chain nearly benchmarked the wrong code.** `overlap_common_hip.cuh` and
  `mxfp8_direct_transpose_hip.cuh` were left over from an earlier build and were *self-consistent*
  without the SWAR path. The kittens target compiles the unsuffixed chain so they were unused, but
  a stale-but-consistent pair is exactly what silently measures the wrong thing.
- **The derivation was `__forceinline__`, and it cost ~45% on the GEMM.** It shares a
  `__global__` function with the MFMA main loop, and register allocation is worst-case over a
  whole function, so its demand merged into that loop's and the INNER LOOP spilled — even in
  configurations where the derivation never ran. Spills went 40 -> 60 VGPR and 32 -> 122 SGPR
  against HEAD, and the fused kernel with no wgrad copy at all went from 17.21 ms to 25.59 ms on
  `65536x13312x16384`. Marking the seven cold entry points `__noinline__` recovers it exactly
  (17.22 ms). The comments at each one say it is load-bearing, because it reads like a hint.
  **This was written down as a risk in SCOPE §7.1 with the check to run, and the check was never
  run.** It stayed invisible because every correctness gate passed and every arm in the sweep was
  slowed equally, so nothing *within* the comparison looked wrong — it only surfaced on the first
  attempt to compare against a number from a different build. A self-consistent A/B is not
  evidence that its absolute numbers are sound.
- **The first benchmark campaign used the wrong harness entirely.** `bench_ag_mxfp8.py` compares
  against RCCL-all-gather-then-GEMM with a per-iteration host barrier; its baseline does not
  reproduce TE's ordering and, for this config, does not produce the column-scaled copy at all,
  so every arm that did produce one scored below 1.0 against it. `overlap_bench` config 3 is the
  instrument for this path: its baseline is TE's own non-overlapped ordering and its config 3
  already carries the aux region. Every number from the earlier campaign is superseded by
  "Results" above.
- **The sweep preflight false-positived on stale KFD handles.** `rocm-smi --showpids` lists dead
  PIDs with their VRAM column intact. `sweep24_colwise.sh` now checks `/proc/$pid` for liveness;
  `sweep24.sh` still has the original check.

---

## Gaps

- **The per-column target** (SCOPE §5) is specified and unimplemented.
- **The interleaved path cannot stream.** `gather_peer_tile_plus_scales` mirrors peer scales for a
  k-range of all 256 rows while its data copy owns all columns of 32 rows — orthogonal slices, so
  a streaming band cannot see the scales it needs. `K >= 16384` falls back to the pool, which is
  correct but pays the re-read. A future variant could read the peer's scales directly.
- **The host-side staging `gather` pays** — a columnwise quantize plus UB copy, per layer, per
  step — is deleted by `transpose` and is not in any measurement here.
  `bench_colwise_staging.py` exists for it and has not been run.
- **E5M2 is untested on hardware.** Its bit math is covered exhaustively on the host; no
  `Format.HYBRID` run has been made.
- **No E4M3 accuracy probe on real gradients** — the counter of coarsened/flushed elements that
  would decide gap 1.
- **`layernorm_mlp.py` stays unreachable** behind the gelu gate, kept in step with `linear.py`.
- **The `_wgrad` region need not be a Userbuffers region** in transpose mode; nothing is
  communicated through it. Making it a plain allocation would drop a UB region per row-parallel
  layer. Kept as-is because it is the smaller diff and changes no configuration.
