<!-- Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved. -->
<!-- License for AMD contributions = MIT. See LICENSE for more information -->

# The fused MXFP8 AG+GEMM (NN) derives its column-wise dY instead of gathering it

**Status: implemented, with the paper's tile-max rule, and selected by default. The second
all-gather is KEPT as a runtime-selectable alternative. The exact per-column target is the next
step and is specified in §5.**

The fused MXFP8 dgrad kernel can now produce wgrad's column-wise copy of dY two ways, chosen at
runtime by `NVTE_AG_WGRAD_COPY`:

| mode | how the copy is produced | staged by Python |
|---|---|---|
| `transpose` (default) | derived in-kernel from the row-wise bytes already gathered, by exponent arithmetic alone — arXiv:2511.02302 Algorithm 1 at MXFP8's block size of 32 | nothing |
| `gather` | a second all-gather into the region, as before | this rank's column-scaled shard, before the dgrad GEMM launches |

`transpose` has three implementations, selected by `NVTE_AG_WGRAD_TRANSPOSE`, which exist because
where the derivation runs is an empirical question, not a design principle:

| | where it runs | measured, §8 |
|---|---|---|
| `inflight` (default) | inside the gather copy, re-encoding bytes already in registers — no second pass over the data | **fastest** |
| `deferred` | a band pool drained between GEMM tiles, entirely off the gather's critical path | middle |
| `phase` | a serial pass after the gather round — the original two-pass design | slowest |

All of it lives in one build, so the arms are measurable back to back in one process image and no
build difference can be mistaken for a mode difference.

Companions: `MXFP8_DIRECT_TRANSPOSE_REVIEW.md` (the walkthrough: what changed, in reading order,
and what to push on), `MXFP8_WGRAD_AG_HANDOFF.md` (why a second copy of dY is needed at all),
`MXFP8_WGRAD_AG_REVIEW.md` (the gather this replaces, commit `6eb5bff1c`), and
`../mxfp8_direct_transpose.py` (the host derivation and loss characterisation this is built on).

---

## 1. Why it works, and why it is simpler than the paper

Three facts drive the design:

1. **There is no data transpose to do.** TE's MXFP8 column-wise tensor has the *same* `[S, O]`
   row-major data layout as the row-wise one — `MXFP8Quantizer.get_columnwise_shape()` returns
   `rowwise_data_shape` verbatim (`pytorch/tensor/mxfp8_tensor.py:267`). Only the scale array
   changes shape: `[S, O/32]` row-wise, `[S/32, O]` column-wise (`get_scale_shape`, HIP branch).
   The paper's Step 1 ("move the bytes") therefore disappears. What remains is Steps 2–3: choose
   a target scale per 32×32 tile, and shift each element's exponent onto it. Elements do not move.

2. **With the tile-max rule the target needs no element scan.** `T'` is the max of the tile's 32
   row scales — 32 bytes out of the row-wise scale array. That makes the conversion a streaming
   elementwise op with ~15 live VGPRs, which matters because this code shares a kernel with the
   MFMA main loop and register allocation is worst-case over all paths.

3. **The gathered tensor is the only input needed.** Every rank already holds the full row-wise
   `[S, O]` dY in the `_dgrad` Userbuffers region after the primary gather. The `_wgrad` region
   stops being a communication buffer and becomes a local destination.

### The element format: E4M3, not E5M2

TE does route backward tensors through the "bwd" dtype slot
(`pytorch/quantization.py:1645`, `get_fp8_te_dtype(recipe, mode == "forward")`), but
`MXFP8BlockScaling.fp8_format` **defaults to `Format.E4M3`** and `__post_init__` asserts pure
E5M2 out (`common/recipe/__init__.py:395,401`). E5M2 grads happen only under `Format.HYBRID`.
Every distributed test here builds a bare `MXFP8BlockScaling()`, so **dY is E4M3 in CI** — the
lossier of the two. Both are implemented; the kernel already carries the gathered operand's
format as its `CBSZ` template parameter, so the derivation gets it for free.

Elements preserved bit-exactly, from `mxfp8_direct_transpose.py`:

| source tensor | E4M3 | E5M2 | max shift k |
|---|---|---|---|
| all-normal (pure Eqs. 12–16) | 100.000% | 100.000% | 4 |
| ordinary gaussian activations | 99.988% | 100.000% | 2 |
| ±3 binades of per-row scale spread | 99.902% | 100.000% | 8 |
| 12 binades of per-row scale spread | 97.589% | 100.000% | 13 |
| full-range stress (not realistic) | 27.607% | 28.358% | 83 |

Underflow is the only failure mode; overflow is impossible because `T'` is a max, so every shift
is downward. Note the loss is measured against the *row-wise quantized* values, while the copy
this replaces was an independent quantization of the high-precision dY. Both sit within one
quantization step of the true dY: this is a change of error, not plainly an increase of it.

---

## 2. What the default no longer does

Everything in this table still happens in `gather` mode; the point is that `transpose` skips it.

| in `gather` mode | in `transpose` mode |
|---|---|
| Python: a second column-wise MXFP8 quantize of `grad_output_arg` | not done |
| Python: `copy_into_buffer` + `copy_scales_into_buffer` of the local shard | not done |
| host: aux `ag_ready_kernel` launch + cross-rank handshake | not done |
| host: aux `gather_scales` launch | not done |
| kernel: aux `gather_all<1,true,false>` round | the derivation instead |
| region carries `peer_ub`, `arrive_local/offset/stride/value`, `reg`, `signal` | all zero/unused |
| `_ag_signal_base` advanced on the aux communicator | not advanced |

Byte traffic per rank at `S=8192` gathered tokens, `O=6144`, `tp=8`:

| | wire (xGMI) | HBM read | HBM write |
|---|---|---|---|
| `gather` | ~45.4 MB | ~19 MB | ~57 MB |
| `transpose` | 0 | ~52 MB | ~52 MB |

Writes are about unchanged — the gather already wrote those bytes — and reads grow because every
rank now derives all `S` rows rather than receiving `(tp-1)/tp` of them. That redundancy is
unavoidable: wgrad needs every token on every rank. The trade is ~45 MB of link traffic for
~33 MB of extra HBM reads, two fewer kernel launches, and one fewer cross-rank dependency.
**None of this has been measured yet** — see §8.

---

## 3. The algorithm as implemented

`cdna4/overlap/mxfp8_direct_transpose.cuh` holds the element rule; `overlap_common.cuh` holds
`mx_colwise_band` / `mx_colwise_all`.

For NN: `M = S` (tokens, the gathered axis), `K = O` (hidden, the contraction axis),
`m_local = S/tp`, `scale_K = K/32`. Both regions shard identically, which `hk_fused_ag_gemm`
asserts. For global 32-row band `b`, chunk `c = b / (m_local/32)`, local band `bl`, row `r`,
column `k`, block `kb = k/32`:

```
src  data   ub  + c*chunk_bytes  + (bl*32 + r)*K       + k
src  scale  ub  + scale_base     + c*scale_chunk_bytes + (bl*32 + r)*scale_K + kb   // [m_local, scale_K]
dst  data   aux + c*chunk_bytes  + (bl*32 + r)*K       + k                          // same position
dst  scale  aux + aux_scale_base + c*scale_chunk_bytes + bl*K + k                   // [m_local/32, K]
```

Step 2 is `T'[kb] = max over the band's 32 rows`. Step 3 re-encodes each element with
`k = T'[kb] - T[r][kb] >= 0`, transcribed from `direct_transpose()` in the reference script:
a normal element that stays normal keeps its mantissa and loses `k` from its exponent field;
anything else lands in, or sinks through, the subnormal range where the shift comes out of the
mantissa and rounds to nearest even.

**Two cases the reference script does not handle, which the kernel does:**

- *Inf/NaN element patterns.* The script shifts them like any other value, so an E5M2 `0x7C`
  would silently become finite, and at `k=0` it would trip the script's own exponent-range
  assert. `mx_rescale()` passes the reserved patterns through untouched. A gathered tensor is not
  ours to trust.
- *A `0xFF` scale byte.* `ptx::float_to_e8m0` emits `0xFF` for NaN and `0xFE` for Inf
  (`util/ptx_hip.cuh:417-422`). One NaN scale makes the band's `T'` `0xFF` and flushes the band.
  This propagates rather than being clamped — a NaN scale means the layer is already NaN — but it
  is a failure mode that did not exist while the bytes were merely copied. Zero blocks are safe
  (`val == 0.0f -> 0x00`, the minimum).

---

## 4. Where it runs

Three placements, all reachable (§header); `inflight` is the default and the fastest (§8).

**`inflight`** re-encodes inside the gather copy itself, so there is no separate phase and the
data is never read twice. Each gather sub-block already owns exactly one 32-row band: in
`gather_peer_tile`, `sub_bytes = align16(ceil(256*K / gath_wg))`, which at `gath_wg == 8` is
exactly `32*K` for any K that is a multiple of 32 — and MXFP8 always passes the `GATH_WG`
constant, not the `gath_wg_nn()` heuristic. The target scale needs only the scale array, which is
32x smaller, so the band's `T'` is reduced in LDS up front and the data then streams once.

The ordering inside that loop is load-bearing and deliberately not tidy: the row-wise store is
issued **before** the re-encode touches the register, so the row-wise stream stays ahead of the
fence that precedes `AG_PUBLISH`. The column-wise stores could not be moved past the publish —
holding a band across it needs 64-128 VGPRs/thread from a runtime-trip-count loop — so in flight
does pay a second store stream inside the publish window. Whether that is free depends on the
gather having spare write bandwidth, which §8 shows it does.

Anything the gate `mx_can_stream_colwise` rejects — `gath_wg != 8`, an unexpected tile layout,
LDS that will not fit, or the interleaved-scales geometry — falls back to the band pool rather
than mis-striding.

**`deferred`** leaves the gather untouched and drains bands from a shared pool one per scheduling
step, between GEMM tiles, so the derivation overlaps compute without touching arrivals. This is
also where our own rank's chunk is always handled: nothing gathers those rows, so nothing can
stream them, and a dedicated phase for them would put 1/tp of the work back on the critical path.

**`phase`** is the original serial pass after the gather round, kept as a baseline and a safety
net.

**Work decomposition.** Bands are 32 rows × K columns, `S/32` of them, round-robin over the
gatherer blocks. They cover **all `tp` chunks**, not just the `tp-1` the gather visits: nobody
ships us our own rows, and nothing else writes the column-wise copy of them any more. `GATH_WG`
is the constant 8 for MXFP8, so `NGATH` is 56 at `tp=8` and 24 at `tp=4`.

**Ordering.** A band in a peer's chunk waits on that chunk's 256-row tile arrival flag — the same
flag the compute blocks use. The wait only ever points from a block that has finished its own
gather to one that has not, so it adds no cycle, and all `NGATH` blocks are co-resident
(`GRID_CAP = 256`, one block per CU).

**Inner loop**, over column windows of `NUM_THREADS * 16 = 8192`: stage the window's source
scales in LDS (≤ 8 KB, borrowed from `As[0][0]` exactly as the interleaved scale packer borrows
`scale_A_smem`), reduce them to `T'`, write the output scale row as one `uint4` of a repeated
byte per thread, then walk the 32 rows loading and storing one `uint4` each. Coalesced 16 B per
lane in and out, one byte of scale lookup per lane per row.

---

## 5. NEXT STEP — the exact per-column target

**What shipped is the paper's shortcut.** `T'` is the max of 32 *scales*, shared by all 32 columns
of a tile. A tighter target exists and is worth doing: the exact per-column maximum, i.e. the
smallest scale that still represents that column's largest element — which is what TE's own
column-wise quantizer would pick. It is never larger than the tile max and is usually several
binades smaller, so it would turn nearly every remaining underflow into an exact value.

**It is cheap in this layout.** A thread owns whole columns, so the reduction is over 32 elements
it already holds: no cross-lane traffic, no LDS, no second pass over memory. That is the reason
to expect it to be close to free, and the reason it is worth revisiting.

**What it costs, and why it was not first:**

- **Shifts become signed.** With a per-column target an element can move *up*, so a subnormal
  source must be renormalized (a leading-zero count on a 2–3 bit mantissa) instead of just having
  its exponent field shifted. `mx_rescale()` takes `k >= 0` today and needs that branch added.
- **Overflow stops being structurally impossible.** Today it cannot happen because the target is a
  max. With a per-column target it rests on an argument — that the column's own maximum, by
  construction, still fits — which is true but is now something to get right rather than something
  the shape of the rule guarantees.
- **The rounding rule has to match TE.** The target must be chosen the way
  `ptx::float_to_e8m0(amax * max_norm_rcp)` does, which rounds the exponent **up**
  (`util/ptx_hip.cuh:429-432`), not the `floor(log2(amax)) - emax` plus bump that the paper and
  the reference script write.
- **The host oracle does not cover it.** `mxfp8_direct_transpose.py` proves the max-of-scales rule
  only. `mxfp8_direct_transpose_check.cpp` would have to be extended to the signed-shift case —
  including a renormalizing reference — *before* any of this could be trusted. That extension is
  the first task, not the kernel change.

**When to do it.** Gate it on measured E4M3 loss on real gradients (§8). E5M2 is already lossless
on every realistic case in the table, and E4M3 keeps 97.6% of elements exact even at a 12-binade
spread — so the case for the extra correctness surface has to be made with numbers, not with the
observation that a tighter scale exists.

The same note, in shorter form, sits at the decision point in the code: `overlap_common.cuh`,
step 2 of `mx_colwise_band`.

---

## 6. What changed

**Kernel / device**

1. `cdna4/overlap/mxfp8_direct_transpose.cuh` *(new)* — `Fmt<CODE>` traits, `rne_shift_right`,
   `mx_rescale<CODE>` (the per-element rule), and `mx_rescale_dword<CODE>` /
   `mx_rescale_dword_fast<CODE>` (the packed path, §6.1). Compiles on the host too (`HK_MXT_FN`
   degrades to `inline`), which is what makes §7's exhaustive check possible.
2. `cdna4/overlap/overlap_common.cuh` — `MX_BLOCK`, `MX_COLWISE_WIN`, `MX_COLWISE_SMEM_BYTES`,
   `mx_colwise_band<CODE>`, `mx_colwise_all<CODE>`.
3. `cdna4/overlap/fused_ag_gemm_nn.cuh` — `aux_peers, aux_dst` → `aux_dst, aux_scale_base`; the
   aux `gather_all` becomes `mx_colwise_all<CBSZ>`; threaded through both launch wrappers, the
   bulk wrapper (`nullptr, 0`) and the function typedef.
4. `cdna4/overlap/fused_ag_gemm_tn.cuh` — the two dead `[[maybe_unused]]` params follow the
   shared `persistent_fn_t`.
5. `cdna4/overlap/comm_gemm.cpp` — aux peer table, aux `ag_ready_kernel` and aux `gather_scales`
   deleted; `NVTE_AG_NO_AUX_DATA` / `NVTE_AG_NO_AUX_SCALES` replaced by one
   `NVTE_AG_NO_AUX_TRANSPOSE`; `NVTE_AG_DIAG` now reports `colwise=on|off`.

### 6.1 The packed element path

The conversion is ALU-bound, not bandwidth-bound: before optimisation every shape in the sweep
landed at ~400-430 GB/s of HBM traffic regardless of size, from blocks that could pull several
times that. The per-byte routine was ~13 VALU ops and, worse, 3-4 divergent branch pairs per byte.

The dominant case — all four bytes of a dword normal, and still normal after the shift — is
exactly `v - (k << MAN_BITS)` per byte, i.e. one packed 32-bit subtract for four elements, since
the exponent field absorbs the whole shift and no lane borrows. `mx_rescale_dword_fast` tests for
that case with two SWAR predicates (all lanes satisfy `E >= k+1`; no lane holds a reserved
inf/NaN pattern) and falls back to the per-byte routine for any dword that fails. Measured on the
gfx950 ISA: ~2.8 VALU/byte against ~13, and divergent branches per 16-byte row from ~56 to 4.

Two details that make it correct rather than approximately correct. A 32-bit shift mixes a
neighbour byte's bits into the lane above, but an FP8 byte is `1 + EXP_BITS + MAN_BITS == 8`, so
the sign bit sits between the exponent field and the neighbour's bits and `EXP_MASK` stops clear
of them in both formats. And `k` is a difference of two E8M0 bytes, so it reaches 255; it is
clamped to `EXP_MASK` before the lane arithmetic, which changes no answer (`E <= EXP_MASK`, so a
larger `k` admits no byte anyway) but keeps every lane in range.

**Contract / host**

6. `gemm/kittens/comm_gemm.h` — `KittensAuxAgRegion` → `KittensAuxColwiseRegion`
   `{dst, chunk_bytes, scale_base_offset, scale_chunk_bytes}`; `args.aux_ag` → `args.aux_colwise`.
7. `comm_gemm_overlap/rocm_comm_gemm_overlap.cpp` — `AuxAgSource` → `AuxColwiseSource`, peer and
   arrival resolution dropped, `_ag_signal_base` advance dropped. The shape validation is **kept**
   — the band addressing rests on it — and gains a 16-byte alignment check, because the derivation
   writes whole vectors of both data and scales. Its documented blind spot ("cannot catch a region
   staged with row-wise bytes") is gone: nothing is staged into the region at all.
8. `include/transformer_engine/comm_gemm_overlap.h` — `aux_ag_comm` → `aux_colwise_comm`.

**Python**

9. `pytorch/module/base.py` — `userbuffers_view_for_derived_columnwise()`: an
   `MXFP8TensorStorage` view over the region the kernel fills. No quantize, no copy, no collective.
10. `pytorch/module/linear.py`, `pytorch/module/layernorm_mlp.py` — call it instead of
    `fill_userbuffers_buffer_for_all_gather`. The ordering constraint the previous review flagged
    as "most likely to be subtly wrong" (every rank must stage before the kernel launches) is
    gone, because nothing is staged and no peer reads the region.

Untouched: `csrc/extensions*` and `cpp_extensions/gemm.py` (the `ub2` / `comm_overlap2` plumbing
was already right), the bulk path, the RS kernels, and both shared `mxfp8_*_mainloop.inc`.

---

## 7. Verification

### 7.0 Mode selection, and the one way it can go wrong

`NVTE_AG_WGRAD_COPY` is read in two places: `kittens_aux_colwise_mode()` in
`gemm/kittens/comm_gemm.h` (C++, cached; stamped into the region struct so the two C++ layers
cannot disagree) and `ag_wgrad_copy_mode()` in `pytorch/module/base.py` (cached; rejects a value
that is neither mode, so a typo fails loudly). Python rejects an unknown value; C++ treats
anything that is not `gather` as `transpose`.

**Nothing cross-checks the two read sites.** If they ever disagreed, Python would stage for one
mode while the kernel ran the other: in one direction the staged shard is overwritten by the
derived copy, in the other the kernel gathers bytes nobody staged. Both are silently wrong, with
no error anywhere. In practice both latch within microseconds of each other on the first backward
pass of one process, so divergence needs `os.environ` mutated between those two reads. The robust
fix is to stop coupling through the environment — pass the mode down through `general_gemm` —
or, more cheaply, expose the C++ mode through pybind and assert equality once. Neither is done.

### 7.1 The element rule — exhaustive, on the host

`../mxfp8_direct_transpose_check.cpp` compiles the shipped header on the host and checks all 256
bit patterns × shifts `k ∈ [0, 40]` × both formats against a reference encoder that shares no
logic with it: a sorted search over the format's value grid with round-to-nearest-even, exactly
as `mxfp8_direct_transpose.py::encode` does. It also asserts the paper's invariant directly —
elements that never leave the normal range come back bit-identical — and that reserved patterns
survive every shift.

The packed path of §6.1 is checked the same way, and harder, because it is selected per DWORD:
a per-byte-uniform sweep would never build a dword whose four bytes disagree about which branch
they need. So it sweeps all 256x256 ordered byte pairs in six lane arrangements (exhaustive over
two-pattern disagreements at every lane position), plus 4-tuples over an 18-byte alphabet for
three- and four-way splits, each across the full `k in [0, 255]` two E8M0 bytes can produce. It
asserts the fast path fires EXACTLY when a predicate written from the format definition says it
should, so a merely-conservative fast path fails too, and it asserts non-vacuity: the run fails if
it never built a mixed dword, never took the fast path, or always took it.

```
g++ -O2 -std=c++17 -Wall -Wextra -o /tmp/mxt_check mxfp8_direct_transpose_check.cpp && /tmp/mxt_check
  E4M3   10414 pairs: 2204 exact, 732 coarsened, 7478 flushed to zero; 82 reserved passed through
  E5M2   10168 pairs: 3990 exact, 484 coarsened, 5694 flushed to zero; 328 reserved passed through
  E4M3   127602688 dwords: 1875194 took the packed path, 3097136 had bytes disagreeing
  E5M2   127602688 dwords: 3639240 took the packed path, 5427456 had bytes disagreeing
PASS                                                                      (3.8 s)
```

(The flush counts are high only because `k` runs past anything real.) The checks were
mutation-tested: leaking the sign bit into the mask, dropping the reserved-pattern exclusion,
comparing against `k` instead of `k+1`, subtracting `k` instead of `k << MAN_BITS`, a mask one bit
too narrow, a guard bit in the wrong place, and removing the `k` clamp all produce failures.

**What this does NOT cover:** the driver. It says nothing about `mx_colwise_band`'s `k`
derivation, the tile-max reduction, the LDS staging, or the `uint4` addressing — those are
unchanged by the optimisation and are what §7.2 covers instead.

### 7.2 The plumbing — byte for byte, on the GPU

`run_layer_with_overlap.py --check-wgrad-colwise` compares the `_wgrad` region the kernel wrote
against a torch transpose of the row-wise `_dgrad` region it read: all `tp` chunks, data and
scales, byte for byte. There is no tolerance to choose. This is what covers the addressing —
which chunk, which band, which scale byte — that §7.1 cannot see.

### 7.3 The trap: a constant grad output proves nothing

`run_fwd_bwd` used `loss = out.sum()`, whose gradient is **identically 1.0**. Every MXFP8 block
then lands on the same scale, every shift is zero, and the derivation degenerates to a copy. The
pre-existing row-parallel MXFP8 test passed in exactly that state — it exercised the plumbing and
never once rescaled an element.

`--grad-scale-binades=N` now weights the loss per token by a random power of two **times a normal
factor**. The power of two alone spreads the scales but leaves every gradient exactly
representable, so the rescale runs without ever rounding; the normal factor is what makes it
round. The test asserts the reported non-zero-shift count is greater than zero, so a future change
that flattens the scales again fails loudly instead of passing vacuously.

### 7.4 Results, 4 and 8 ranks

| check | result |
|---|---|
| `--check-wgrad-colwise`, `tp=4`, 6 binades | PASSED: 3,145,728 data and 98,304 scale bytes match, 83,452 elements rescaled |
| `weight.grad` vs the non-overlapped reference | rel. err 0.0071 (tol 0.125) — real double-quantization error, as expected |
| `output`, `input.grad`, `bias.grad` | rel. err 0.0 |
| **negative control** (`NVTE_AG_NO_AUX_TRANSPOSE=1`) | region all zeros: 3,145,728/3,145,728 data and 98,304/98,304 scale bytes differ, `weight.grad` fails as a zero tensor |
| `NVTE_AG_DIAG=1` | `colwise=on`, confirming the arm actually ran |
| `test_rocm_fused_overlap.py -k mxfp8` | see below |

The negative control is what makes the rest meaningful: it proves the wgrad GEMM genuinely
consumes the region the kernel writes.

### 7.5 The gold ISA gate

`MXFP8_AG_REFACTOR.md` §3.1 makes `cdna4/mxfp8_gemm.cpp`'s device ISA the binding acceptance
criterion. Its include closure is `kittens.cuh`, `../kittens_common.h`,
`../kittens_kernel_common.cuh`, `mxfp8_gemm_helper.cuh` and the two `mxfp8_*_mainloop.inc` files.
**None of them is touched by this change** — every edit is under `overlap/` plus the
`gemm/kittens/comm_gemm.h` contract header, which that TU does not include (`git status` over the
closure is empty). The ISA is unchanged by construction rather than by measurement; the
instruction counts were captured as corroboration.

---

## 8. Measured

**The headline measurement lives in `MXFP8_DIRECT_TRANSPOSE_REVIEW.md`, section "Results".**
Against TE's non-overlapped path in TE's own ordering (`overlap_bench` config 3, MXFP8, TP=8,
24 shapes): the second all-gather is **1.277x** and the direct transpose **1.387x**, both 24/24,
with the transpose ahead of the all-gather by **1.085x** per-shape geomean.

The numbers that used to be in this section came from `sweep24_colwise.sh` /
`bench_ag_mxfp8.py` and are **withdrawn**, for two independent reasons, either of which alone
would disqualify them:

- **Wrong instrument.** That harness compares against RCCL-all-gather-then-GEMM under a
  per-iteration host barrier. Its baseline does not reproduce TE's ordering, and for this config
  it does not produce the column-scaled copy at all — so every arm that did produce one was
  charged for work the baseline skipped, and scored below 1.0 against it. `overlap_bench`
  config 3 exists precisely to not make that mistake, and its `with_aux` already carries this
  region.
- **Wrong build.** They were taken while the derivation's entry points were `__forceinline__`,
  which cost ~45% on the GEMM itself (see §9 and the REVIEW's corrections list). Every arm was
  slowed equally, so the comparison looked self-consistent while every absolute number was wrong.

What survives from that campaign is the ranking it produced — `inflight` > `deferred` > `phase`,
and `inflight` > `gather` — which the `overlap_bench` result independently agrees with for the
one comparison it covers. The `deferred` and `phase` arms have not been re-measured on the fixed
build with the right harness.

---

## 9. Not done

- **`deferred` and `phase` are unmeasured on the fixed build.** §8's result covers `inflight`
  against `gather` and the baseline, which is the comparison that matters; the other two
  placements were only ever measured on the slow build.
- **Error bars.** §8's arms are one `overlap_bench` window each (REPS=2). The `gather` arm
  reproducing the previously recorded figure to within 1% is the evidence that they are stable;
  a repeat window would make that an argument from data rather than from one agreement.
- **The host-side staging `gather` pays is still unmeasured.** A columnwise quantize plus UB copy,
  per layer, per step, which `transpose` deletes outright. `bench_colwise_staging.py` exists for
  it and has not been run. It sits outside the kernel region `overlap_bench` times, so §8
  understates the difference in a real model.
- **`gath_wg` may want re-tuning.** MXFP8 passes the constant 8. The post-primary work changed
  from link-bound to HBM-bound, so the width that balanced AG bandwidth against GEMM FLOPs is not
  obviously still right.
- ~~**Register pressure not checked.**~~ **Checked, and it had bitten.** The derivation's entry
  points were `__forceinline__` and their register demand merged into the MFMA main loop's
  allocation, spilling the inner loop and costing ~45% on the GEMM even when the derivation never
  ran. `__noinline__` on the seven cold entry points recovers it exactly (17.22 ms against HEAD's
  17.21 on `65536x13312x16384`). Re-check `.vgpr_count` / `.vgpr_spill_count` after any change to
  these paths: HEAD is 256 / 40, this tree is 256 / 43, and the broken build was 256 / 60.
- **Interleaved scales (`K >= 16384`) cannot stream.** `gather_peer_tile_plus_scales` mirrors
  peer scales for a k-range of all 256 rows while its data copy owns all columns of 32 rows —
  orthogonal slices, so a streaming band cannot see the scales it needs. That geometry falls back
  to the band pool, which is correct but pays the re-read. A future in-flight-interleaved variant
  could read the peer's scales directly (K remote bytes per band, 1/32 of the data). Coverage for
  this path now exists: `test_fused_ag_overlap_row_parallel_mxfp8_interleaved` runs a 128x128 head
  config landing exactly on the threshold.
- **E5M2 untested on hardware.** The instantiation compiles and §7.1 covers its bit math
  exhaustively, but no `Format.HYBRID` run has been made.
- **The E4M3 accuracy probe.** Counters of coarsened / flushed elements and observed max `k` on
  real gradients — the number that decides §5.
- **`layernorm_mlp.py` stays unreachable** behind the gelu gate, as before; it is kept in step
  with `linear.py` but is not exercised.
- **The `_wgrad` region need not be a Userbuffers region any more.** Nothing is communicated
  through it. Making it a plain allocation would drop a UB region per row-parallel layer and
  remove `fused_wgrad_ag_eligible`'s dependency on `_wgrad` being configured `fused`. Keeping it
  was the smaller diff and changes no configuration.
