<!-- Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
     License for AMD contributions = MIT. See LICENSE for more information -->

# MoE (expert parallel): forward and backward

An EP=8 bf16 MoE layer (top-k routing, SwiGLU experts) for gfx950. The forward and the backward are each **one
kernel launch** on the caller's stream. The forward takes the routing in device memory and plans it itself: there is
no separate planning launch. The token all-to-all (dispatch and combine) runs inside the same kernel as the grouped
GEMMs, through symmetric peer buffers (`KosmosComm`). Every MoE layer has its own context; all of them share one
workspace that holds what a call needs only while it runs. The training API (Training API, below) instead keeps
everything a backward reads from its forward in buffers the caller owns, on one instance per configuration.

* Library: `src/moe` plus `src/utils`, built as `build/libkosmos_moe.a` (`make moe`, which also builds
  `build/moe_linkcheck`). No MPI and no other communication library.
* Kernels: `moe_fwd_single` (`moe_fwd_dev.cuh`, with its start phase in `moe_plan.cuh`) and `moe_bwd_single`
  (`moe_bwd_single.cuh` and the `moe_*.cuh` / `*.inc` files it includes). With the fused router (below) the same
  kernels are instantiated with the router in `moe_router.cu`, a separate translation unit, so the router-free
  kernels compile exactly as without it. Top-k 1, 2 and 4 use the small-k instantiations (`moe_*_ks.cu`), which
  differ only in their folds; the kernels for 6, 8 and 10 compile exactly as without them. Probs on input
  (`cfg.probs_in`, top-1) has its own instantiations (`moe_*_ks_pin.cu`). The MXFP8 kernels
  (`moe_mx_*.inc`, `moe_mx_quant.cuh`, `moe_mx_train.cuh`) are instantiated in `moe_mx.cu` (`moe_mx_q8.cu` with the
  fp8 combines) the same way, their small-k and probs-on-input builds in `moe_mx_ks.cu` and `moe_mx_ks_pin.cu`. The
  training API's kernels are `moe_train_api.cu`'s and `moe_train_api_ks.cu`'s (the same templates with its call,
  `TrainApiCall` / `MOE_BWD_TRAIN_API`). Every kernel translation unit is built once per activation (`act`).

## How one launch works

Both kernels launch `wgcap` persistent workgroups (8 waves each, at most one per CU). A few workgroups start in a
communication role and join the tile loop when their pushes run out; all others claim GEMM tiles from queues whose
order puts every producer ahead of its consumers. Cross-GPU data is published with a release atomic (or wide
write-through stores drained with `vmcnt(0)` before a relaxed add) and acquired by the consumer after it observes
the count.

### Forward: plan, dispatch, GEMM1 + SwiGLU, GEMM2, combine, fold

1. **Plan.** Workgroups outside the dispatch crew sort this rank's routing (`topk_idx`, in chunks of 2048 entries):
   each chunk gets its per-expert counts and every entry its stable rank within the chunk. Once every chunk is
   counted, each chunk's workgroup reads all the chunk counts, derives its chunk's prefix and the send offsets, and
   writes its entries into the send list (each entry's slot `t * k + j`, in (expert, destination) order); meanwhile
   one more workgroup publishes this rank's per-expert totals into every rank's count slot (two slots, alternating per
   call) with a release add on a per-source counter. Only counts cross ranks, never the routing itself.
2. **Layout.** Every workgroup waits until all 8 sources have published this call's counts (counter >= calls so far),
   acquires, and derives the whole layout in LDS: per-(expert, rank) padded row offsets, send and receive offsets.
   The dispatch crew waits there while the others sort, so a rank arriving last finds its peers' counts already
   published. The wait is also a barrier: no rank writes to a peer for call `p` before every rank has finished its
   call `p - 1` (which also makes the two count slots safe). A routing whose largest per-rank row count exceeds the
   capacity runs in rounds (bf16 and MXFP8, training and inference, see Over capacity). An expert id
   outside `[0, 8 * E_loc)` (e.g. -1 for a padding token) is dropped (weight 0, bit 1): it adds nothing to `out`,
   `dX` or `dprobs`; a token routed twice to one expert sets bit 2.
3. **Tables.** The same workgroups write the forward's tables (tile queues, gather rows, arrival and combine
   tables), zero the zero row, and release them once; every workgroup waits for all of them once, before its first
   tile. After that release they write the backward's tables and settings and zero the padding rows the wgrads read
   directly.
4. **Dispatch push.** Workgroups `[0, ncomm)` copy this rank's token rows into the receive arena of each expert's
   owner (one unit per (expert, destination), split into `ncomm / 8` slices), each row with its slot `t * k + j` into
   the owner's slot array for the combine. Every slice release-adds 1 to the destination's (source, expert) signal.
   `ncomm` is 32, 24 or 16 for `I` below 1024, below 3072, and from 3072 up; with `cfg.mxfp8` (half the dispatch
   bytes, faster GEMMs) it is `24576 / I` rounded to 8 in [16, 48], doubled when the forward also pushes the
   column-wise `Xp`.
5. **GEMM1 + SwiGLU.** Tiles of `A = silu(gate) * up` are claimed from per-XCD buckets. A tile first waits until
   every source with rows in its 256-row block has published (signal >= epoch count), then acquires.
   When the caller passes `G` (training), the epilogue first stores the tile's bf16 pre-activations (gate and up) to
   `G` for the backward. Each finished tile bumps its row block's counter.
6. **GEMM2.** Claimed after the last GEMM1 ticket. A GEMM2 tile waits until every GEMM1 tile of its row block has
   published, then writes `Y` with write-through stores and bumps the row block's `tile_done`.
7. **Combine push.** Workgroups `[0, fcomm)` (`fcomm = min(128, wgcap rounded down to 8)`) push the `Y` rows of a
   unit (expert, token owner) back to the owner's combine arena once all its row blocks are done. Workgroup `w`
   serves destination `w % 8`; the `fcomm / 8` workgroups of a destination share its units as one claim pool: each
   takes the first ready unit from its own starting expert, wins it with a compare-and-swap on the unit's claim word,
   then acquires the unit's `tile_done` (which also orders its slot array, through the GEMM1 tiles' acquire of its
   dispatch signal). They start after this rank's dispatch has drained and stop when every unit of their destination
   is taken.
8. **Fold.** Once every row owed to this rank has arrived and this rank's combine workgroups have finished,
   items of 8 tokens compute `out[t] = sum_j w[t][j] * Y[t][j]` in fp32 and store bf16.
9. **Reset.** The last workgroup zeroes every counter and claim word, then consumes the arrival credits (waits until
   each source's count reaches what it owes, and subtracts it). The dispatch signals and count counters are never
   cleared: they are compared against a per-call epoch. Before its first push, each dispatch workgroup also waits
   until the peer has finished the previous backward on this context, so a forward never overwrites activations (or
   slots) a peer still reads.

### Backward: dispatch + dtopkw, dgrad2 + SwiGLU-bwd, wgrad2, dgrad1, combine, wgrad1, fold

1. **Dispatch push + dtopkw.** The first `dncomm` workgroups push `dY = bf16(w * dOut)` rows to the expert owners
   (the router weight is applied here) and compute `dtopkw[t][j] = <dOut[t], Y[t][j]>` from the forward's combine
   arena, one wave per row, into workspace scratch; each dispatch workgroup then adds to a count the fold acquires.
   Each pushed row block release-adds to the destination's per-(source, row block) signal. The receive and combine arenas belong to the workspace, so consecutive backwards may be different layers
   with no forward in between: at entry every rank stores its backward call count into each peer, and before its
   first push each dispatch workgroup waits until its destination has started this backward (finished the previous
   one). A backward right after a forward skips that wait: the forward's count barrier already ordered it after
   every peer's previous backward. The other workgroups meanwhile zero the receive-arena rows this call reads but no
   peer writes (each expert's rows up to its count rounded to 64, and the zero row); a workgroup's first dgrad2 or
   wgrad2 tile waits for them.
2. **One tile queue**, in this order:
   * dgrad2 of expert `e` (NN, `dA = dY W2`, with the SwiGLU backward fused into the epilogue, producing `dG`),
     interleaved with wgrad2 of expert `e - 1` (NT, `dW2 = dY^T A`). Both wait for the dispatched rows they read.
     dgrad2 publishes `dG` per row block with a release add.
   * dgrad1 (NN, `dXp = dG W1`), gated per row block on dgrad2. It stores `dXp` write-through and publishes per row
     block.
   * wgrad1 (NT, `dW1 = dG^T Xp`), each expert gated on its whole `dG`. With `I > 1024`, wgrad1 of each expert in
     the first half follows dgrad1 of the next expert, and the other experts follow all dgrad1, longest K first. With
     a smaller `I`, all wgrad1 follows dgrad1, longest K first: the wgrad1 left after dgrad1 is then too short to
     cover the combine push, which would be delayed by the interleave.
3. **Combine push.** The first `cncomm` workgroups are dual-role: before each tile claim they push any unit of
   `dXp` rows whose row blocks are complete back to the token owners.
4. **Fold.** Once every row owed to this rank has arrived and every dispatch workgroup has counted in, items of 16
   tokens compute `dX[t] = bf16(sum_j dXp[t][j])` (fp32 sum, round to nearest even) and write the tokens' full
   `dprobs[t][:]` rows: `dtopkw[t][j]` at expert `topk_idx[t][j]`, zero elsewhere (the expert ids and the dropped
   entries, which the sum skips, from the forward plan's per-context packed routing). With the fused router the
   router's dX rows join the sum and there is no `dprobs`.
5. **Reset.** The last workgroup zeroes every counter and consumes the dispatch and arrival credits.

`dncomm`, `cncomm` and the wgrad2 / wgrad1 run lengths (tiles per claim) are set once per context from the shape,
not per call: a cost model of the push bytes and the GEMM work (run lengths: claim cost against tile time), fed with
a balanced routing's expected counts (`T k / (EP E_loc)` rows per source and expert, giving the off-rank push rows,
the padded rows and the mean rows per expert) and clamped to the cap, the same on every rank. For MXFP8 training
the model takes the GEMM work at 0.65 of bf16's, counts the column-wise `Xp` in the dispatch bytes when the
backward pushes it, and keeps its own margins (dispatch 1.0, combine 1.0 with at least 32 workgroups) where bf16
takes 1.5 and 0.75 with at least 24 (each matched to a sweep of the counts). The kernel is
deadlock-free at any cap in [9, 256] because pushes wait only for their destination to start the same call, every
queue bucket has a resident home workgroup, and every producer precedes its consumers in the queue.

## API

Include `src/moe/moe.h` (it includes `src/utils/kosmos_comm.h`) and link `build/libkosmos_moe.a -lamdhip64`.

```
kosmos_moe_validate(&cfg)                                    host-only configuration check
kosmos_moe_arena_bytes(&cfg, rows, &layer_bytes, &shared_bytes)   symmetric bytes per context and per workspace
kosmos_moe_workspace_create(&ws, comm, &cfg)                 local; comm may be NULL
kosmos_moe_create(&ctx, ws)                                  local; one context per MoE layer, configured as ws
kosmos_moe_prepare(ctx)                                      collective: the first forward's setup, before a capture
kosmos_moe_forward(ctx, X, W1, W2, topk_idx, topk_w, out, G, stream, ev)   1 launch, plans the device routing
kosmos_moe_forward_mxfp8(ctx, X, Xs, W1, W1s, W2, W2s, topk_idx, topk_w, out, stream, ev)   1 launch, MXFP8 GEMMs
kosmos_moe_forward_mxfp8_train(ctx, X, W1, W1s, W2, W2s, topk_idx, topk_w, out, G, stream, ev)   1 launch, bf16 X
kosmos_moe_backward_mxfp8(ctx, dout, G, X, W1c, W1cs, W2c, W2cs, dX, dprobs, dW1, dW2, stream, ev)   1 launch
kosmos_moe_quantize_mxfp8(W, nmat, rows, cols, col, q, s, stream)   1 launch, bf16 weights to MXFP8
kosmos_moe_plan(ctx, topk_idx, topk_w)                       host arrays: collective, host-blocking, resets all state
kosmos_moe_plan_status(ctx, &status)                         host-blocking: status of the forwards since the last query
kosmos_moe_backward(ctx, dout, G, W1, W2, dX, dprobs, dW1, dW2, stream, ev)   1 launch
kosmos_moe_forward_router(ctx, X, Wr, bias, logit_bias, W1, W2, out, G, aux_loss, tpe, seed, stream, ev)   1 launch,
                                                             routes too
kosmos_moe_backward_router(ctx, dout, G, X, W1, W2, Wr, aux_scale, dX, dW1, dW2, dWr, dlogit_bias, stream, ev)
                                                             1 launch
kosmos_moe_info(ctx), kosmos_moe_error(ctx)                  derived settings, last error message
kosmos_moe_pool_resize(ws, bytes)                            training: collective, host-blocking (Saved-state pool)
kosmos_moe_pool_info(ws, &bytes, &head_bytes, &reserve_bytes), kosmos_moe_workspace_error(ws)
kosmos_moe_set_capacity(ws, rows)                            collective, host-blocking (Capacity changes)
kosmos_moe_capacity_query(ws, &stat), kosmos_moe_capacity_bytes(ws, rows, &bytes)   collective / local (same)
kosmos_moe_destroy(ctx)                                      local
kosmos_moe_workspace_destroy(ws)                             local; after every context on it
```

### Training API: caller-owned saved state

```
kosmos_moe_instance_bytes(&cfg, &bytes)                      symmetric bytes of an instance (an external comm's)
kosmos_moe_instance_create(&inst, comm, &cfg)                collective, host-blocking; comm may be NULL
kosmos_moe_saved_bytes(&cfg, rows, &sb)                      bytes of each saved buffer (xp, g, a, y, rec)
kosmos_moe_train_forward(inst, X, W1, W2, topk_idx, topk_w, out, saved, stream)   1 launch; saved NULL: inference
kosmos_moe_train_backward(inst, dout, X, W1, W2, topk_idx, topk_w, saved, dX, dprobs, dW1, dW2, acc, stream)
                                                             1 launch
kosmos_moe_instance_status(inst, &status), kosmos_moe_instance_info(inst), kosmos_moe_instance_error(inst)
kosmos_moe_instance_destroy(inst)                            local
```

An instance is one workspace and one context for one EP group and configuration (bf16 training: `cfg.backward`,
`cfg.capacity_rows > 0`, no fused router, shared expert, MXFP8 or probs on input; `pool_bytes` and
`comm_offset` ignored). It owns the kernels' settings and the symmetric buffers of the call in progress, all on one
comm (`comm = NULL`: a managed one; else at least `kosmos_moe_instance_bytes`, e.g. `kosmos_comm_wrap` of a framework's
symmetric memory, kept until the instance is destroyed): the workspace's arena first, then the context's forward
arena with its receive rows. It has no pool. Creation is collective and does all a first call would, so every later
call enqueues one kernel and can be captured.

Everything a backward reads from its forward lives in the caller's `KosmosMoeSaved` (every pointer 256-byte aligned;
`tier` and `rows` equal on every rank): `xp` (Xp), `g` and `a` (GEMM1's pre-activations and A), `y` (Y by token slot,
`[T k][H]`), and `rec`, the route record: this call's kept tier, passes, rows and status, its count matrix, the
receive slots and the tables its backward reads. Their sizes come from `kosmos_moe_saved_bytes(&cfg, rows)`: `rows` is
the busiest rank's routed rows (every local expert padded to 256) the buffers keep; `rec` and `y` do not depend on it.
So any number of forwards may be in flight, backwards come in any order and after any other calls, a forward whose
backward never runs leaves nothing behind, and the buffers may move between a forward and its backward (`rec` holds no
address). `X`, `W1`, `W2`, `topk_idx` and `topk_w` must hold the forward's values at the backward.

* **Tier.** A forward keeps `saved->tier` (`KOSMOS_MOE_SAVE_ALL`: Xp, G and A; `_XP`: Xp, its backward recomputes
  GEMM1; `_NONE`: nothing, its backward pushes `X` again and recomputes), or nothing when the call's largest per-rank
  row count exceeds `saved->rows` or the instance's capacity (rounds); every rank decides alike. Every tier gives the
  same outputs bit for bit. Status (`kosmos_moe_instance_status`, the OR since the last query): 1, 2 and 8 as
  `kosmos_moe_plan_status`; 16 a call kept Xp only and 32 nothing where its tier asked for more.
* **Forward.** As `kosmos_moe_forward` with the caller's `G`, with Xp landing in the instance's receive arena: GEMM1's
  epilogue writes `G` and `A` to `g` and `a` (a tier below all: `A` to the workspace's recompute buffer, `G` nowhere);
  with Xp kept, the GEMM1 tiles of three row blocks in four (`XP_LDS_PERIOD`, `XP_LDS_BLOCKS`) also store the block's
  rows from their LDS stages to `xp`, each column tile its share of the k-steps; the fold starts per item once the
  item's rows have landed (the units' arrival words, as MXFP8 training's) and writes each `Y` row it reads to `y`; once
  the tile queue is drained, the workgroups waiting for the fold copy the receive slots and the tables into the record
  and the other Xp row blocks and the zero row into `xp`, each after acquiring every plan workgroup's tables (a count of
  finished plan workgroups) and, for the slots and Xp, every source's dispatch; the last workgroup writes the record's
  header and count matrix.
* **Backward.** Every workgroup first copies its share of the record's tables back into the instance's (a call in one
  pass: also the receive slots; a call in rounds: the count matrix and epoch the rounds' plan reads, through the
  context's copy of the forward arguments `SingleArgs::fa`), then all wait on one counter; the rest is the backward
  above, with Xp, `G`, `A`, `Y` and `X` the call's.
* **Weight gradients.** `acc = 0`: `dW1`, `dW2` bf16, overwritten; `acc = 1`: fp32 (`main_grad`), each tile's fp32
  sum added; `acc = 2`: bf16, each element `bf16(old + sum)` rounded once (an expert split over rounds adds per round).

### Configuration (`KosmosMoeConfig`, identical on every rank except `coll.rank`; set on the workspace)

| field | meaning |
|---|---|
| `H`, `I` | hidden size, expert intermediate size |
| `E_loc`, `k` | experts per rank (global experts `8 * E_loc`), experts per token |
| `T` | tokens per rank |
| `wgcap` | launch width of every MoE kernel, in [9, 256] |
| `backward` | nonzero to allocate the backward's state |
| `capacity_rows` | arena rows per rank to reserve (0: size from the routing; a device routing needs > 0) |
| `pool_bytes` | training: bytes of the saved-state pool shared by every context (grown to at least one call's block; see Saved-state pool) |
| `comm_offset` | byte offset of the workspace's arena in a caller-supplied comm, multiple of 4096 |
| `coll` | all-gather callback over the 8 EP ranks (`KosmosCollective`), with `rank` and `nranks = 8` |
| `router` | the fused router (`KosmosMoeRouterConfig`, `on = 0`: none) |
| `shared`, `mxfp8` | the shared expert; MXFP8 operands (with `backward`: MXFP8 training) |
| `fp8_combine` | with `mxfp8` and `backward`: MXFP8 forward and backward combines (default 0: bf16; see MXFP8 training) |
| `probs_in` | Llama 4's probs on input (Megatron's `moe_apply_probs_on_input`; k = 1, no fused router): the dispatch pushes each row as `bf16(w x)` and the fold is unweighted, `out = sum_j Y_j` (+ the shared expert); the backward pushes `dY = dOut` unweighted, and its fold computes `dX = bf16(w dXp)` (+ the shared expert's dX) and `dprobs = <dXp, X>`, reading the forward's `X` and weights (the caller keeps `X` unchanged until the backward). MXFP8 training quantizes `bf16(w x)` instead of `x`, both row-wise (GEMM1) and column-wise (wgrad1); `kosmos_moe_forward_mxfp8` takes `X` already as the MXFP8 of `bf16(w x)` |
| `act` | the GLU of the routed and shared experts: `KOSMOS_MOE_ACT_SWIGLU` (0), or `KOSMOS_MOE_ACT_SWIGLU_CLAMP10`, DeepSeek-V4's `silu(min(g, 10)) * clamp(u, -10, 10)` as TE's `clamped_swiglu_v2` (limit 10, alpha 1, offset 0), whose backward zeroes `dg` where `g > 10` and `du` where `abs(u) > 10`; a compile-time policy, one kernel build per activation |

### Tensors (row-major; bf16 unless noted)

| tensor | shape |
|---|---|
| `topk_idx` | `[T][k]` int32 global expert ids in `[0, 8 * E_loc)`, or -1 (dropped); expert `g` lives on rank `g / E_loc`; no repeats per token |
| `topk_w` | `[T][k]` fp32 router weights |
| `X`, `out`, `dout` | `[T][H]` |
| `W1`, `dW1` | `[E_loc][2I][H]`, rows in blocks of 256: 128 gate rows, then the matching 128 up rows |
| `W2`, `dW2` | `[E_loc][H][I]` |
| `G` | training: `KOSMOS_MOE_SAVE` (the forward saves into the pool) or null (not saved) |
| `dX` | `[T][H]` |
| `dprobs` | `[T][8 * E_loc]` fp32: the gradient of `topk_w` at each token's routed experts, zero elsewhere |

All outputs are overwritten, not accumulated. Router weights stay fp32 and are applied at the fold (forward) and
the unfold (backward).

### Contract

* **Order.** create the workspace and the contexts, then any sequence of forward / backward on any of the contexts,
  then destroy the contexts and the workspace. Every rank issues the same calls in the same order on one stream.
  Calls are stream-ordered: no host synchronization or barrier is needed between them, on one rank or across ranks.
* **Routing.** Each forward plans its own routing; the backward reuses the tables of the most recent forward.
  * `kosmos_moe_forward` with `topk_idx` and `topk_w` in device memory (int32 and fp32, `[T][k]`) enqueues one kernel
    and nothing else: no host synchronization, allocation, memset, copy or host collective. The pointers must stay
    valid until that kernel has run. It needs `capacity_rows > 0`: the first such call on a context is collective and
    host-blocking (it maps the context's arena and every table for `capacity_rows` rows per rank, rounded up to 256,
    and the workspace's buffers on the first context). A routing over capacity or with bad ids is handled as in the
    plan step above (rounds, status bit 8; bits 1, 2).
  * `kosmos_moe_plan` takes host arrays: it synchronizes the device, checks the routing, sizes (or grows) the
    context's arena and the workspace from it, zeroes the context's and the workspace's state, keeps a device copy of
    the routing, and ends with a collective agreement that also acts as a barrier. It fills `kosmos_moe_info` with
    the routing's settings. A forward with `topk_idx` and `topk_w` null uses the routing of the latest
    `kosmos_moe_plan`.
  * `kosmos_moe_plan_status` synchronizes the device and returns the OR of the status bits of every forward since the
    last query (errors: bits 1, 2; none: all executed; bit 4, over capacity, is no longer set), and refreshes the
    per-routing fields of `kosmos_moe_info` from the latest forward. Notes, not errors: 8 a call ran in rounds (any
    forward), and in training 16 a forward kept Xp only, 32 one kept nothing; a storing forward that cannot keep its
    Xp runs in rounds, so bit 64 (a single pass that kept nothing) is an error.
* **No per-call reset.** Forward and backward reset their own state in-kernel (counters zeroed by the last
  workgroup, dispatch signals epoch-compared, arrival credits consumed). There is no memset or copy per call.
* **Backward inputs.** The backward reads what the context's most recent forward saved in the pool (it refuses a
  forward that did not store). So each backward must follow the forward
  it differentiates with no other forward on the same context in between. Calls on other contexts of the workspace
  may come in between, in any order: all forwards and then the backwards in reverse, a recomputed forward right
  before its backward, or any interleaving. Inference passes `G` null and does not store it.
* **Recovery.** After a failed or aborted call, quiesce every rank and call `kosmos_moe_plan` again on every rank;
  it zeroes the context's and the workspace's state (a device-routing forward may follow it).
* **Events.** The optional `ev` array is recorded on the stream: the forward records `ev[0]` before and `ev[1]`
  after its launch, the backward records `ev[0]` after its launch.

### Saved-state pool (training)

A bf16 forward's saved state is its block `[Xp | G | A]` of `R` rows (an MXFP8 forward's below). The contexts of a workspace do not each
reserve one: every forward writes its block at the cursor of one symmetric pool of `pool_bytes` (the same cursor on
every rank), and its last workgroup commits as much of it as the pool can keep while still leaving room for the
`Xp` of every context whose forward is still to come:

* **all**: the backward reads `Xp`, `G` and `A`;
* **Xp only**: `G` and `A` are dropped; the backward first recomputes GEMM1 (`G`, and `A = SwiGLU(G)`) from `Xp` in
  per-tile work its dgrad2 and wgrad2 wait on;
* **nothing**: only in rounds (Over capacity). A storing forward whose pool has no room for even a full-capacity
  `Xp` runs as one round: the pool's last commit or release records whether the next call's `Xp` fits (any slot's,
  and the slot it updated with that slot's own block replaced), and the forward reads that before its first pass. The
  backward's dispatch workgroups push every rank's `X` rows again to their experts' ranks, beside `dY`, and the
  recompute reads them as they land. This reads the forward's `X`: the caller keeps it unchanged until the backward.
  A single pass never keeps nothing (status bit 64 would report one).

MXFP8 training's block is `[G | column-wise A and its scales | (I < 1024) column-wise Xp and its scales]` of `R` rows
(every rank's `Mpad` plus the zero row's block), the column-wise `Xp` written by its sources into their destinations'
pools at the same offset (every rank's pool decides alike). There is no `Xp`-only tier and no reserve for later layers:
a forward keeps its whole block, or, after rounds, nothing. A storing forward whose pool has no room for a full-capacity
block runs as one round (as bf16's), and its backward runs in rounds, re-sending `X`, quantizing it once on arrival
and recomputing GEMM1 into the pool's head (the next base, free of every stored block until the backward frees its
record). The shared expert's column-wise `As` stays in the context (the backward has no shared GEMM1 recompute); its
row-wise `As`, read only by the forward, is in the workspace.

Each backward frees its context's block. Decisions are per call, so only the layers that do not fit degrade, each
within its one launch. The pool needs no reset and is never refused: it is the ranks' least `pool_bytes` (every rank
decides alike), grown to at least one call's block, and sizing it for `n` layers' blocks keeps `n` of them in full.

`kosmos_moe_pool_resize(ws, bytes)` (collective, host-blocking) re-allocates the pool between steps, e.g. once the
first step has shown what memory the framework needs (`pool_bytes = 0` starts at one call's block, its storing calls
in rounds). It drains every rank's device and is refused on every rank, changing nothing, inside a stream capture
(`hipErrorStreamCaptureUnsupported`) or while a stored block awaits its backward (`hipErrorNotReady`). Otherwise
every rank unmaps the old pool before any rank frees it, the new one is the ranks' least `bytes` (halved while the
allocation fails, down to one call's block), and every context is re-pointed before any rank returns. Calls captured
into a graph before keep the old pool's addresses: resize before capturing. `kosmos_moe_pool_info` gives the pool's
bytes, one call's block and the `Xp` reserved per layer still to store.

### Capacity changes

`kosmos_moe_set_capacity(ws, rows)` (collective, host-blocking) makes the ranks' largest `rows`, rounded up to 256
and at most what one call can need, the workspace's capacity. It is refused as `kosmos_moe_pool_resize`; otherwise it
closes every open context's arenas and the workspace (the pool too if it no longer holds one call's block) and opens
them again at the new rows, every device argument built anew, so it grows and shrinks alike and never holds both
sizes. A backward of a forward before it is refused. If an allocation fails on any rank the old capacity is kept and
every rank gets `hipErrorOutOfMemory`. Calls captured into a graph before keep the old buffers.
`kosmos_moe_capacity_query(ws, &stat)` (collective, between calls) returns, since the previous query or change, the
most rows a rank needed in any call (each forward folds its full-routing peak, before any round window, into a
running max in its plan record, one atomic per call), the capacity and its largest value, how many contexts set status 8 (or 4, no longer set), and how many of those in rounds
needed more than their capacity (the others the pool forced into rounds); it clears their status and peaks
(`kosmos_moe_plan_status` still returns those bits). `kosmos_moe_capacity_bytes` gives the device bytes the capacity
sizes at `rows`.

### Over capacity

A device routing can need more rows on some rank than `capacity_rows`. The forward and the backward then run the call
in rounds inside their one launch. Each receiver's rows (its experts, padded, in order) are cut into windows of at most
the capacity, at expert boundaries or, inside an expert larger than the capacity, at 256-row multiples; round `r`
dispatches every source's rows of every receiver's window `r` and runs the call's GEMMs and combine on them, reusing
the same capacity-sized buffers. Every rank derives the same windows from the call's counts, so all take the same
number of rounds.

* **Forward.** A first pass with the empty layout (the router, the shared expert) counts the rounds; each round then
  derives its layout from the counts clipped to its windows, and Y lands in the combine arena by slot, so the fold
  runs once, after the last round. Between rounds a workgroup barrier resets the round's counters and returns its
  arrival signals, and each rank tells its peers (RFREE) that their next round may write its rows.
* **Backward.** The forward keeps nothing in the pool. Each round's tables are derived again from the forward's counts;
  dY and X (with their slots) go out per round, GEMM1 is recomputed, and an expert split over rounds adds its later
  rounds' weight gradients to the first's (fp32 sum, rounded to bf16 once more per round). The unfold runs once.
* **MXFP8 training forward.** As bf16's: each round pushes its window's fp8 rows from the call's one quantization of
  `X`, and a call in rounds keeps nothing in the pool. With `I < 1024` the windows cut each
  expert's 32-row padded (source, expert) segments, so every column-wise group of 32 rows is the single pass's.
* **MXFP8 inference forward** (`kosmos_moe_forward_mxfp8`, every kernel family: k in {6, 8, 10}, small k, probs on
  input). As the training forward's rounds, with less to do: `X` arrives fp8 with its scales, so each round pushes its
  window's rows and scales straight from the caller's `X` and `X_scale`; nothing is stored. Its layout is unpadded
  (no column-wise `Xp`, at any `I`), so the windows are bf16's. Every GEMM tile, `A` quantization and fold sum is the
  single pass's, so `out` is bit-identical to one pass with enough capacity. The round-free code (the call's first
  pass, which over capacity is the empty layout's) is the same kernel's: a call that fits runs it alone.
* **MXFP8 training backward.** As bf16's, and each round's `X` rows (with `cfg.probs_in` `bf16(w x)`) land beside `dY`
  in the workspace's forward receive arena, in bf16: the receiver quantizes each landed block (256 rows by 1024
  columns) once, row-wise into its own buffer and column-wise for wgrad1, and recomputes GEMM1 with the forward's
  tile (`G`, and `A` column-wise for wgrad2, at the pool's head), so every fp8 operand is the single pass's. A later
  round of a split expert adds its fp32 weight-gradient tile to the stored bf16 tile, rounded to nearest even once.

Calls that fit run as before, in one pass. A larger `capacity_rows` means fewer rounds for skewed calls; its rows are
the workspace's buffers and each context's tables, and the size of each pool block. A storing
forward with no room in the pool for its `Xp` also runs as one round (see Saved-state pool).

### Fused router

With `cfg.router.on`, `kosmos_moe_forward_router` also computes the routing: the logits `X Wr^T` (+ `logit_bias`; fp32),
the score, the top-k selection and weights, the aux-loss value and the per-expert dispatch counts, then runs the forward
on that routing in the same launch. `kosmos_moe_backward_router` differentiates it: besides the MoE gradients it
computes the dlogits (from the dispatch weights' gradient and the aux-loss gradient), adds the router's dX into `dX`,
and writes `dWr`. Megatron semantics (`KosmosMoeRouterConfig`):

* `score` (a compile-time policy: the router's kernels are built once per score function):
  * `KOSMOS_ROUTER_SOFTMAX` takes the top k of the logits and a softmax over them (equal to a softmax over all experts
    renormalized over the top k: Qwen3, Mixtral);
  * `KOSMOS_ROUTER_SIGMOID` (DeepSeek-V3) and `KOSMOS_ROUTER_SQRTSOFTPLUS` (`sqrt(softplus(logit))`, DeepSeek-V4)
    take the top k of score + bias, and weight by the scores normalized to sum 1 when k > 1 (as they are at k = 1);
  * `KOSMOS_ROUTER_SOFTMAX_PRE` takes a softmax over all experts, then its top k, not renormalized (Grok, OLMoE).
  Ties are broken toward the lower expert.
* `groups`, `group_topk` (0: none): the top k is taken within the `group_topk` groups (of
  `8 E_loc / groups` experts) with the largest sum of their top `k / group_topk` selection scores (score + bias).
* `aux_groups`: 0 no aux loss; 1 over the batch (`aux_loss`); n > 1 over n sequences, token t in sequence `t % n`
  (`seq_aux_loss`). `aux_coeff`: its coefficient. `scale`: the weight scale (0 means 1).
* `force`: 1 replaces the logits by N(0, 1) draws seeded by the device word `seed` and the rank (the GEMM still runs).

`bias` (fp32 `[8 E_loc]`, sigmoid and sqrtsoftplus, may be null) shifts the selection only. `logit_bias` (fp32
`[8 E_loc]`, may be null; GPT-OSS's router linear bias) is added to the logits, and `dlogit_bias` (fp32 `[8 E_loc]`,
null: not computed) receives its gradient, the dlogits summed over the T tokens in a fixed order (deterministic). The
caller's `topk_idx` / `topk_w` path (no fused router) takes any other routing, e.g. hash routing (DeepSeek-V4's first
layers). `Wr` and `dWr` are bf16 `[8 E_loc][H]`,
`aux_loss` fp32 `[1]`, `tpe` int32 `[8 E_loc]` (this rank's tokens per expert, for the expert-bias update).
`aux_scale` (device fp32 `[1]`, read when the kernel runs) scales the aux-loss gradient (Megatron's
`main_loss_backward_scale`). Limits: `T` a multiple of 256,
`8 E_loc <= 512`, `group_topk` divides `k` and `k / group_topk <= 4`, `aux_groups <= 4`. A context keeps the logits,
aux counts and draw seed its backward needs; the per-call scratch (aux partials, dtopkw, dlogits, router dX,
counters) is in the workspace.

Graphs: the forward and backward epochs live on the device, so a context's calls can be captured into a HIP graph and
replayed, interleaved with eager calls, once it has run a forward or `kosmos_moe_prepare` on every rank (a first
forward inside a capture fails with `hipErrorStreamCaptureUnsupported`). `kosmos_moe_plan` resets the state; a plan
that grows the arenas re-allocates them, so graphs that captured the context must be captured again.

### MXFP8 forward

`kosmos_moe_forward_mxfp8` is the forward (same launch, routing and contract) with OCP MXFP8 operands: `X`, `W1`,
`W2` are fp8 e4m3 in the bf16 layouts above, each with row-wise e8m0 scales, one per 32 consecutive elements of a row
(`Xs [T][H/32]`, `W1s [E_loc][2I][H/32]`, `W2s [E_loc][H][I/32]`, unswizzled). `out` is bf16.

* The dispatch pushes each fp8 row with its scales (half the bf16 bytes); GEMM1 and GEMM2 use the block-scaled
  16x16x128 MFMA, with the scales of each K step's 256 + 256 rows fetched into LDS and packed per lane.
* GEMM1's epilogue quantizes `A = silu(gate) * up` (fp32) to MXFP8: per 32 columns of a row the scale is `2^e`, `e`
  the least integer with `amax 2^-e <= 448`, and the elements are rounded to nearest even. GEMM2 reads that `A`.
* Inference: `G` is not stored, so a backward after this forward is refused. Needs `H` a multiple of 512 and no
  fused router (`hipErrorNotSupported` otherwise).

### MXFP8 training (`cfg.mxfp8 = 1` with `cfg.backward = 1`)

All four backward GEMMs are MXFP8 (the TE recipe: dgrads row-wise gradient by column-wise weight, wgrads column-wise
by column-wise, blocks of 32 along each GEMM's K), and no fp8 value is ever requantized: every fp8 operand is one
quantization of its bf16 or fp32 value.

* `kosmos_moe_forward_mxfp8_train` takes bf16 `X`: all workgroups first quantize this rank's tokens row-wise once
  (into the workspace), and the dispatch then pushes fp8 rows as the inference forward does. GEMM1's epilogue stores
  `G` (bf16) and, besides the row-wise `A` for GEMM2, a column-wise (32 rows) `A` for wgrad2. `A` is rounded to bf16
  before it is quantized (TE's gated activation + MXFP8 cast).
* `cfg.fp8_combine = 1` (default 0: bf16 combines) makes both combines MXFP8. GEMM2's epilogue quantizes each `Y`
  row and dgrad1's epilogue each `dXp` row from the fp32 accumulators (e4m3, one E8M0 per 32 columns) into its
  2H-byte row slot (values at bytes `[0, H)`, scales at `[H, H + H/32)`); the pushes send only those bytes
  (`1.03H` instead of `2H` per row); the folds and the backward's `dtopkw` dequantize to fp32. The forward and backward
  must use the same setting (the backward reads the forward's combine rows). Its kernels are separate instantiations
  (`moe_mx_q8.cu`), so with 0 the MXFP8 kernels are unchanged. It costs output accuracy (each expert row goes through
  fp8 once): about 31.5 dB SNR of `out` against the unrounded reference instead of 53 dB.
* For `I >= 1024` the receive arena packs each expert's rows by source as in bf16 (only the expert is padded, to 256
  rows), so the wgrads' token-wise 32-row blocks follow the experts' row order and may span sources; for `I < 1024`
  each (source, expert) segment is padded to 32 rows (zero rows, written by the receiver), so a source owns all its
  groups. The column-wise `Xp` (wgrad1's operand) goes into a peer-written region: each source quantizes the 32-row
  groups inside its own segment from bf16 `X`; for `I >= 1024` the backward's dispatch writes them after its `dY`
  push (link idle time before the combine) into the workspace's arena, ordered against other layers' backwards as
  the `dY` rows are, each dispatch workgroup taking a contiguous slice of the destination's groups over its experts,
  and counts them on a per-(source, expert) word that wgrad1 waits for per expert; the receiver also zeroes each
  expert's rows past its last group that wgrad1 reads, counted on its own word; for `I < 1024`, where the backward
  stays link-bound, the forward's dispatch writes them after the rows into the context's arena (a source's groups
  then precede its backward, so every `dY` row of it that `dG` waited for). With packed segments, the groups
  no single source owns (at most 8 per expert: those spanning a source boundary or the expert's end) are first
  quantizations too: each source pushes its head and tail rows of those groups (under 32 each) in bf16 into a shadow
  region next to the column-wise `Xp`, counted on a per-expert word; the receiver's dispatch workgroups wait for the
  rows and quantize the groups, counted on the receiver's own word.
* `kosmos_moe_backward_mxfp8` takes the column-wise weights `W1c` (`[E_loc][2I][H]` fp8, scales `[E_loc][2I/32][H]`)
  and `W2c` (`[E_loc][H][I]`, scales `[E_loc][H/32][I]`), and `X` again (bf16, as given to the forward); a backward in
  rounds also reads the forward's row-wise `W1` and `W1_scale`, which the caller keeps unchanged until then. The
  dispatch pushes bf16 `dY`; quantization items (256 rows by 1024 columns) give it row-wise (dgrad2) and column-wise
  (wgrad2). dgrad2 computes the transposed tile and its epilogue (SwiGLU backward in fp32, rounded to bf16) stages
  `dG` row-wise (dgrad1) and column-wise (wgrad1) in LDS for full-row stores. The wgrads run over each expert's rows
  rounded up to 128 (at least 256), with zero rows past the count.
* The shared expert (`cfg.shared`) runs in the same launches as in bf16:
  `kosmos_moe_forward_shared_mxfp8_train(ctx, X, W1, W1s, W2, W2s, Ws1, Ws1s, Ws2, Ws2s, topk_idx, topk_w, out, G, Gs,
  sgate, stream, ev)` and `kosmos_moe_backward_shared_mxfp8(ctx, dout, G, Gs, X, W1c, W1cs, W2c, W2cs, Ws1c, Ws1cs,
  Ws2c, Ws2cs, dX, dprobs, dW1, dW2, dWs1, dWs2, acc, sgate, dsgate, stream, ev)`, with `Ws1` (Megatron's `[gate; up]`)
  and `Ws2` row-wise and column-wise as one expert's `W1`, `W2` (`nmat = 1`), and `Gs` required. Its tiles read the
  row-wise `X` the routed experts read (with `cfg.probs_in` a second quantization of `X` itself); its activation
  `As = bf16(sgate bf16(act))` is stored row-wise (fc2) and column-wise (wgrad2); its backward quantizes `dOut` both
  ways and `X` column-wise, and dgrad2's epilogue stages `dGs` as the routed `dG` in the `[gate; up]` column order.
* The fused router (`cfg.router`) runs in the same launches too: `kosmos_moe_forward_router_mxfp8_train(ctx, X, Wr,
  bias, logit_bias, W1, W1s, W2, W2s, out, G, aux_loss, tpe, seed, stream, ev)` and `kosmos_moe_backward_router_mxfp8(ctx,
  dout, G, X, W1c, W1cs, W2c, W2cs, Wr, aux_scale, dX, dW1, dW2, dWr, dlogit_bias, stream, ev)`, with the shared expert
  `kosmos_moe_forward_router_shared_mxfp8_train` and `kosmos_moe_backward_router_shared_mxfp8` (the shared calls'
  operands after the router's). The router itself stays bf16 as in the bf16 calls (logits, scores, routing, aux loss;
  dlogits, `dWr`, its dX); its dX tiles run the wgrad body on the dlogits transposed, which the dlogits items also
  write, so the kernel holds no bf16 dgrad body.
* Needs `H` a multiple of 1024 and `T` a multiple of 256 (`X` is quantized in 256-row blocks). A bf16 backward after
  an MXFP8 forward (and vice versa) is refused, and so is a bf16 router forward on an MXFP8 context.
* `kosmos_moe_quantize_mxfp8` makes the weights of these calls from `nmat` bf16 matrices `[rows][cols]`: `q` the e4m3
  values, `s` the e8m0 scales, row-wise `[nmat][rows][cols/32]` (`col = 0`: `W1`, `W2`) or column-wise
  `[nmat][rows/32][cols]` (`col = 1`: `W1c`, `W2c`), with the kernels' scale rule; it matches TE's MXFP8 quantizer.

### `wgcap`

Every MoE launch has exactly `wgcap` workgroups, which reserves the remaining CUs for other work (for example
another collective) without a second stream. It bounds concurrency, not placement. The forward's `ncomm` and
`fcomm` and the backward's `dncomm` and `cncomm` are clamped to fit the cap; `kosmos_moe_info` reports them. The
cap changes scheduling only: every output element keeps one owner and a fixed reduction order. The library accepts
[9, 256]; the forward benchmark harness additionally requires a multiple of 8.

### Memory: contexts and the workspace

A context keeps what its backward needs from its forward besides its pool block: the combine arena (`Y` per routing
slot, for `dtopkw`), slot array, signals and count slots and the routing tables (MXFP8 with the shared expert: also its
column-wise `As`). MXFP8 training's backward reads neither the forward's receive arena nor the row-wise `A`, so those
live in the workspace.
Its peer-written part lives in the context's own symmetric buffer (a managed `KosmosComm`: hipMalloc + IPC handles
exchanged through `cfg.coll`) of `layer_bytes` from `kosmos_moe_arena_bytes`, mapped by the first `kosmos_moe_plan`
or device-routing forward on the context; a plan re-maps it larger when a new routing needs more rows.

The workspace holds what one call needs only while it runs, once for all its contexts: the backward's peer-written
receive and combine arenas with their signals (`shared_bytes`), `dG`, and one buffer that is the forward's `Y` and the
backward's `dXp`; with MXFP8 training also the forward's receive arena (in `shared_bytes`; a peer pushes rows into it
only after this rank published its counts for that call, so after its previous call ended), the row-wise `A`, the
row-wise fp8 `X` of a backward in rounds, and for `I >= 1024` the column-wise `Xp` and its shadow region (in
`shared_bytes`). It is mapped by the first context that maps its arena, at that context's rows; a context that needs
more rows grows it (collective: every rank drains its device first, and the other contexts are re-pointed).
Its symmetric arena lives in (see [`src/tpsp/README.md`](../tpsp/README.md) for create vs wrap):

* `comm = NULL`: a managed comm, owned by the workspace.
* An existing comm (managed, or wrapped around buffers the caller already maps on all 8 ranks). Its rank and size
  must match `cfg.coll`, and it must hold at least `comm_offset + shared_bytes` from
  `kosmos_moe_arena_bytes(&cfg, rows, ...)`, where `rows` is the larger of `capacity_rows` and the largest per-rank
  row count of the routing (routed rows per local expert, each rounded up to 256, summed). A plan that needs more is
  refused. Other ops can share the comm at disjoint offsets.

### Limits

* EP = 8 ranks, one GPU each, in one node.
* `k` in {1, 2, 4, 6, 8, 10}; `E_loc <= 64`.
* `H` and `I` multiples of 256 (the GEMM tile), and below the 32-bit buffer offsets: `4 I H` and `4 T I` under
  2^32 bytes, `2 T H` under 2^31.
* `kosmos_moe_plan` also refuses a routing whose per-rank row count overflows 32-bit operand offsets (`(rows + 1) 2H`
  or `(rows + 1) 4I` bytes), or that exceeds a kernel table limit; `kosmos_moe_error` says which.
* MXFP8 training backward: its translation units are built without MachineLICM, because the over-capacity rounds
  body exceeds the register file with it. The round-free path pays for this in its per-tile scheduling and wait code
  (re-loaded arguments, longer poll loops): the backward is slower than main on `qwen30` (I < 1024) and within noise or
  faster on the other benchmarked shapes. Open: shrink the rounds body until it is spill-free under MachineLICM.

Benchmarked shapes (`bench/common/moe_shapes.h`), T = 8192 tokens per rank:

| shape | H | I | experts | k | E_loc |
|---|--:|--:|--:|--:|--:|
| dsv3 | 7168 | 2048 | 256 | 8 | 32 |
| qwen35 | 4096 | 1024 | 512 | 10 | 64 |
| dsv4pro | 7168 | 3072 | 384 | 6 | 48 |
| kimi27 | 7168 | 2048 | 384 | 8 | 48 |
| glm52 | 6144 | 2048 | 256 | 8 | 32 |
| qwen235 | 4096 | 1536 | 128 | 8 | 16 |
| qwen30 | 2048 | 768 | 128 | 8 | 16 |
| gate8 (checks only, T = 1024) | 2048 | 256 | 16 | 8 | 2 |
| llama4scout (probs on input) | 5120 | 8192 | 16 | 1 | 2 |
| llama4maverick (probs on input) | 5120 | 8192 | 128 | 1 | 16 |
| l4scoutpx (probs on input) | 6144 | 3072 | 16 | 1 | 2 |
| l4mavpx (probs on input) | 6144 | 3072 | 128 | 1 | 16 |
| gate1p (probs on input, checks only, T = 1024) | 2048 | 256 | 16 | 1 | 2 |
| glm45 | 5120 | 1536 | 160 | 8 | 20 |
| ernie45 | 8192 | 3584 | 64 | 8 | 8 |
| dbrx | 6144 | 10752 | 16 | 4 | 2 |
| h2816 (checks only, T = 1024) | 2816 | 1280 | 16 | 2 | 2 |
| h2816p (probs on input, checks only, T = 1024) | 2816 | 1280 | 16 | 1 | 2 |

`llama4scout` and `llama4maverick` are the Llama 4 models' expert dims; `l4scoutpx` / `l4mavpx` run their expert
counts and routing at H 6144, I 3072. A top-1 routing draw weights each token by the sigmoid of its score (Llama 4's
router), not 1.

### Python

`make python` builds the module `kosmos` (`build/python`). It mirrors the C API one call per entry point; every
buffer and the stream (`hipStream_t`) is an integer address, 0 for null, and the caller allocates all of them
(training: `G = kosmos.MOE_SAVE`, the forward saving into the pool of `pool_bytes`). A failed call raises `RuntimeError` with `kosmos_moe_error`; a refused configuration raises
`ValueError`.

```
ws = kosmos.MoeWorkspace(H, I, E_loc, k, T, rank, allgather, wgcap=256, capacity_rows=0, backward=True,
                         router=None, shared=False, mxfp8=False, pool_bytes=0)
                                                # router: kosmos.MoeRouterConfig(score, groups, group_topk,
                                                #         aux_groups, aux_coeff, scale, force)
ws.pool_resize(bytes)                           # collective: 0 done; refused: 1 a stored block, 2 a capture
ws.pool_info()                                  # {"bytes", "head_bytes", "reserve_bytes"}
ws.set_capacity(rows)                           # collective: as pool_resize; 3 out of memory, old capacity kept
ws.capacity_query()                             # collective: {"peak", "rows", "max_rows", "nctx", "rounds",
                                                #              "overflow", "cap_rounds"}
ws.capacity_bytes(rows)                         # device bytes the capacity sizes at rows
ctx = kosmos.MoeContext(ws)                     # one per MoE layer
ctx.prepare()                                   # collective; before capturing the context's calls in a graph
ctx.forward(X, W1, W2, topk_idx, topk_w, out, G, stream)
ctx.backward(dout, G, W1, W2, dX, dprobs, dW1, dW2, stream)
ctx.forward_router(X, Wr, bias, W1, W2, out, G, aux_loss, tpe, seed, stream, logit_bias=0)
ctx.backward_router(dout, G, X, W1, W2, Wr, aux_scale, dX, dW1, dW2, dWr, stream, dlogit_bias=0)
ctx.forward_mxfp8_train(X, W1, W1_scale, W2, W2_scale, topk_idx, topk_w, out, G, stream)   # mxfp8=True
ctx.backward_mxfp8(dout, G, X, W1c, W1c_scale, W2c, W2c_scale, dX, dprobs, dW1, dW2, stream)
ctx.forward_shared_mxfp8_train(X, W1, W1_scale, W2, W2_scale, Ws1, Ws1_scale, Ws2, Ws2_scale, topk_idx, topk_w,
                               out, G, Gs, stream, sgate=0)   # mxfp8=True, shared=True
ctx.backward_shared_mxfp8(dout, G, Gs, X, W1c, W1c_scale, W2c, W2c_scale, Ws1c, Ws1c_scale, Ws2c, Ws2c_scale, dX,
                          dprobs, dW1, dW2, dWs1, dWs2, acc, stream, sgate=0, dsgate=0)
ctx.forward_router_mxfp8_train(X, Wr, bias, W1, W1_scale, W2, W2_scale, out, G, aux_loss, tpe, seed, stream,
                               logit_bias=0)   # mxfp8=True, router
ctx.backward_router_mxfp8(dout, G, X, W1c, W1c_scale, W2c, W2c_scale, Wr, aux_scale, dX, dW1, dW2, dWr, stream,
                          dlogit_bias=0)
ctx.forward_router_shared_mxfp8_train(X, Wr, bias, W1, W1_scale, W2, W2_scale, Ws1, Ws1_scale, Ws2, Ws2_scale, out,
                                      G, Gs, aux_loss, tpe, seed, stream, logit_bias=0, sgate=0)
ctx.backward_router_shared_mxfp8(dout, G, Gs, X, W1c, W1c_scale, W2c, W2c_scale, Ws1c, Ws1c_scale, Ws2c, Ws2c_scale,
                                 Wr, aux_scale, dX, dW1, dW2, dWr, dWs1, dWs2, acc, stream, dlogit_bias=0, sgate=0,
                                 dsgate=0)
kosmos.quantize_mxfp8(W, nmat, rows, cols, col, q, s, stream)
ctx.plan(topk_idx, topk_w)                      # host arrays
ctx.plan_status(), ctx.g_bytes, ctx.info(), ctx.error()
ctx.destroy(); ws.destroy()                     # explicit: dropping the objects does not free them
```

The training API:

```
kosmos.SAVE_NONE, kosmos.SAVE_XP, kosmos.SAVE_ALL
kosmos.instance_bytes(H, I, E_loc, k, T, capacity_rows, rank, allgather, wgcap=256, act=0)
                                                # bytes to allocate for ext_* (the instance's and the comm's flags)
inst = kosmos.MoeInstance(H, I, E_loc, k, T, rank, allgather, wgcap=256, capacity_rows=..., act=0,
                          ext_local=0, ext_peers=0, ext_bytes=0)   # ext_*: a symmetric buffer, kosmos_comm_wrap'ed
inst.saved_bytes(rows)                          # {"xp", "g", "a", "y", "rec"}
inst.forward(X, W1, W2, topk_idx, topk_w, out, tier, rows, xp, g, a, y, rec, stream)   # tier -1: inference
inst.backward(dout, X, W1, W2, topk_idx, topk_w, tier, rows, xp, g, a, y, rec, dX, dprobs, dW1, dW2, acc, stream)
inst.status(), inst.info(), inst.error()
inst.destroy()
```

`allgather(send: bytes)` is the host all-gather over the 8 EP ranks (`cfg.coll`): it returns the 8 ranks' `send`
concatenated in rank order, as any bytes-like object. It runs only in the collective calls: the first
`forward` / `forward_router` on a context that maps its arena, and `plan`.

## Correctness

**Inputs** are generated in device memory from (shape, T, dist); no data files (`bench/common/moe_inputs.cuh`).
Routing is a Gumbel top-k draw per source rank, balanced (`dist 0`) or skewed (`dist 1`: the logits of the first
E/8 experts raised by log 3); the weights are the softmax of the selected scores. Values are hash-based normal
draws rounded to bf16: forward X, W1, W2 ~ N(0, 1); backward X ~ N(0, 1), W1 ~ N(0, 1/H), W2 ~ N(0, 1/I),
dOut ~ N(0, 1/H).

**CPU reference** (`bench/common/moe_check.cpp`): C++ with OpenMP and fp32 accumulation, sharing no code with the
kernels. It reproduces the library's bf16 roundings: `G = X W1^T`, `A = bf16(silu(gate) * up)`, `Y = bf16(A W2^T)`,
`out = bf16(sum_j w_j Y_j)`; backward `dY = bf16(w * dOut)`, `dprobs[t][topk_idx[t][j]] = <dOut, Y_j>`, `dA = dY W2`,
`dG = bf16(SwiGLU'(bf16(G)) * dA)`, `dXp = bf16(dG W1)`, `dX = bf16(sum_j dXp_j)`, `dW2 = dY^T A`, `dW1 = dG^T Xp`. With
probs on input `Xp = bf16(w X)`, `out = bf16(sum_j Y_j)`, `dY = dOut`, `dprobs = <dXp_j, X>` and `dX = bf16(sum_j w_j
dXp_j)`. Each rank computes the rows routed to its experts and returns them over MPI to the token's source rank.

**MXFP8** (`--mxfp8`): the harness quantizes the generated bf16 `X`, `W1`, `W2` to MXFP8 on the device
(`kosmos_moe_quantize_mxfp8`) and hands the reference their dequantized values, which bf16 holds exactly. The reference
quantizes `A` like the kernel. An `A` element whose fp32 value lies within its accumulation-error bound (`G` taken to
`2^-18 sum |x w|`) of a rounding or scale boundary may round either way on the device; each such element widens the
tolerance of the outputs it reaches by `|its two roundings' difference| * |W2|` (the check prints the mean and max
widening, typically 0.02 to 0.14 atol, and how many elements needed it).

In the backward (`moe_bwd_dx --mxfp8 1`) the reference quantizes, from the bf16 inputs, `X` and `dY` both ways (the
column-wise blocks over each expert's rows in the arena's order, padded to 32 by zero rows per expert, or per source for
`I < 1024`) and `W1`, `W2` both ways, and `A` and `dG` from their fp32 values as the kernel does; the ambiguity widening
is carried through to `dX` (`dG`), `dprobs` (`A`, as `sum_i unc(A_i) |(W2^T dOut)_i|`; with probs on input `dG`, as
`sum_h unc(dXp_h) |x_h|`), `dW1` (`dG`) and `dW2` (`A`). The shared expert's MXFP8 reference (`moe_train --mxfp8 1 --shared 1`) quantizes
`X`, `Ws1`, `Ws2`, `dOut`, `As = bf16(sgate bf16(act))` and `dGs` the same ways, widening `out`, `dX`, `dWs1` and
`dWs2` likewise and `dsgate` by `act`'s spread over the ambiguous roundings of `Gs`.

With `--fp8comb 1` the reference rounds each expert's fp32 `Y` and `dXp` row to MXFP8 as the epilogues do; an element
within its error bound (the ambiguity above plus `2^-8 |v|`) of an fp8 rounding or scale boundary widens the tolerance
by the spread of its possible values; `out_q` (the verdict) uses these rows and `out` (information) the bf16 ones,
and `dtopkw` is `<dOut, Y_fp8>`. Every MXFP8 check also prints each output's rel-L2 and SNR against the bf16-path
reference (no MXFP8 quantization of any operand, fp32 accumulation; information only).

**Checker.** A tensor passes if every checked element satisfies `|c - r| <= atol + rtol * |r|` with
`atol = atol_rms * RMS(r)` over the checked elements, and the whole GPU output has no non-finite value.

| tensor | stored | rtol | atol_rms |
|---|---|--:|--:|
| out | bf16 | 2^-7 | 2^-4 |
| dW1, dW2 | bf16 | 2^-7 | 2^-4 |
| dX | bf16 | 2^-8 | 2^-4 |
| dprobs | fp32 | 2^-8 | 2^-4 |

**Coverage.** `--check N` checks N tokens per rank (full rows of out and dX, the routed entries of dprobs; every
other dprobs entry of every token must be exactly 0) and, in the backward, N
(expert, i) columns per rank (`dW2[:, i]` and both W1 rows of `i`, each with its full reduction over the expert's
rows), drawn deterministically; `--check -1` checks everything. gate8 is checked in full, the 7 shapes with
N = 1024.

**Determinism and state.** Every timed forward call must reproduce the first call's output bit for bit; every
backward iteration must reproduce iteration 0's dX, dprobs, dW1 and dW2. After every call the harnesses read back
every counter, signal plane and zero row and require them to be reset. Destinations and intermediates are poisoned
before every call (outputs with NaN, so an unwritten element fails the non-finite clause).

* `bench/moe_fwd/gate.sh`: 19 cells (gate8 at T = 1024 in full; 9 shapes x {balanced, skew} at T = 8192, sampled;
  `MXFP8=1`: the MXFP8 forward, `MXTRAIN=1`: the MXFP8 training forward). Each must print `[check] RESULT PASS`,
  `[determinism] ... PASS`, 0 `[tdgate]` failures, no refusal, and the expected derived settings. A fifth column (or
  `ROWCAP`) sets `--cap`: then also `[rounds] ... PASS` (status 8 in rounds, never 4; `out` bitwise equal to one pass).
* `bench/moe_bwd/check.sh <shape>` (`MXFP8=1`: the MXFP8 training forward and backward; `DIST=1`: the skewed
  routing): one backward run with the
  check (full for gate8, sampled otherwise, 2
  iterations); requires `[check] RESULT PASS`, `[determinism] ... PASS` and 0 `[resetchk]` failures.
* `bench/moe_train/check.sh <shape>`: the training pattern. `CALLS` (4) calls, each with a new device routing
  (balanced and skewed draws, experts and tokens rotated per call), each forward (storing `G`) + backward back to back
  on one stream with no host synchronization; the arenas, intermediates, routing tables and `G` are poisoned in-stream
  before every forward. Every call's outputs are checked against the CPU reference, and a second pass must reproduce
  them bit for bit. `--cap ROWS` below the largest routing's need runs the calls in rounds (with `--layers`, bf16).
  With `LAYERS=N` (`--layers N`; with `--mxfp8 1` the MXFP8 training calls), N layers with their own weights, inputs
  and routings share one workspace under four call orders: forwards L0..LN-1 then backwards LN-1..L0; the same with
  the first N/2 layers recomputed (a forward without `G`, then again with `G` right before the backward); a random
  valid interleaving; and a workspace
  sized for the smallest routing that a host plan grows while another layer is between its forward and its
  backward. The workspace's data is poisoned in-stream before every call (the receive arena of a backward that
  follows a forward before that forward: peers may push into it as soon as they finish that forward) and a layer's
  state before its forwards; every call is checked against the CPU reference, and a second pass must reproduce
  every output bit for bit.
* `bench/moe_train/check.sh <shape> --api 1`: the training API (Training API, above), bf16, in 15 scenarios on four
  input sets (own `X`, `dOut` and routing, balanced and skewed; shared `W1`, `W2`). The existing API first runs each
  set's forward and backward on a workspace at the main capacity (the largest set's rows) and at the rounds capacity
  (half the smallest set's rows), for the reference outputs. On an instance at the main capacity: each set alone at
  tiers ALL, XP and NONE and with saved rows below the call's; two forwards in flight with their backwards in FIFO and
  LIFO order; three forwards, backwards reversed; an inference forward between a forward and its backward; an abandoned
  forward followed by training calls; a forward discarded, the forward again, its backward; the saved buffers moved (the
  old ones poisoned) between a forward and its backward; and `acc` 1 and 2, three calls added into a non-zero buffer
  (another set's `dW1`, `dW2`). On an instance at the rounds capacity: each set alone, two in flight (LIFO) and an
  inference forward in between, every call in rounds. Every saved buffer is sized by `kosmos_moe_saved_bytes` (null
  where the tier needs none), followed by 4096 bytes that no call may write, and poisoned before each forward with the
  outputs; the instance's state is poisoned as the existing API's before each call. Every output must equal the existing
  API's for its set bit for bit, each forward's record header must hold the expected kept tier, passes and status, and
  the instance's status after each scenario the OR of its calls' notes (8 in rounds, 16 and 32 below the asked tier).
  `acc` 1 into zeros gives the kernel's own fp32 tiles, which must round to the `acc` 0 result; the three `acc` 1 calls
  must equal the same fp32 adds of those tiles, and the `acc` 2 calls `bf16(old + tile)` per call, bit for bit. The
  calls alone at tier ALL, on both instances, are checked against the CPU reference in the first pass; the whole
  sequence runs twice and every output must be bitwise equal between the passes. `--act 1` runs ClampedGLU; `--gstd 5`
  makes the clamp act on a few percent of the gate and up values (the default `G`, standard deviation 1, never reaches
  the limit 10).

## Benchmarks

```
bench/moe/run.sh                                       # fwd, bwd x balanced, skewed; 9 shapes, KOSMOS vs PyTorch
SHAPES=bench/moe/smoke.txt DIRS=fwd DISTS=0 bench/moe/run.sh   # 2 shapes, forward, balanced
make moe_fwd moe_bwd moe_train moe_bench               # build/moe_{fwd,bwd_dx,train,bench}
bench/moe_fwd/gate.sh                                  # forward gate, 19 cells
CELLS=bench/moe_fwd/smoke.txt bench/moe_fwd/gate.sh    # 1 cell
bench/moe_bwd/check.sh gate8                           # backward check, every element
bench/moe_bwd/check.sh dsv3                            # backward check, sampled
bench/moe_train/check.sh gate8                         # training pattern, every element checked
LAYERS=4 bench/moe_train/check.sh gate8                # 4 layers on one workspace, four call orders
bench/moe_train/check.sh gate8 --api 1                  # the training API: tiers, call orders, acc, rounds
bench/moe_train/check.sh gate8 --api 1 --act 1 --gstd 5 # the same with ClampedGLU
mpirun -np 8 build/moe_train --shape qwen235 --timing 5              # a new routing per call: fwd, bwd
```

`bench/moe/run.sh` builds `moe_bench`, the routing generator and the report tool, then runs per (direction, routing,
precision) a KOSMOS process (`mpirun -np 8 build/moe_bench`) and a PyTorch baseline process (`torchrun --nproc-per-node
8 bench/moe/baseline.py`), all in an order shuffled from `SEED` (`ORDER_<STAMP>.txt`). Each process walks every shape of
`SHAPES` (`bench/moe/shapes.txt`: `<shape> <T>`, the shape a row of `bench/common/moe_shapes.h`) `REPS` times.
Environment: `OUT_DIR` (`work/moe`), `SHAPES`, `DIRS` (`fwd bwd`), `DISTS` (`0 1`), `PRECISIONS` (`bf16 mxfp8`; mxfp8:
both processes with `--mxfp8`, CSVs and logs with `_mx`), `WARM` (10), `ITERS` (25), `REPS` (3), `SEED` (also the
in-process arm order), `STAMP`, `ABLATIONS` (`unfused`, below; empty leaves it out), `PYTHON` (`python3`, with a ROCm
build of PyTorch), `DRYRUN=1` (print the commands only). Direct use: `mpirun -np 8 build/moe_bench --dir fwd|bwd --dist
0|1 --shapes FILE [--warm N] [--iters N] [--reps N] [--unfused] [--mxfp8] [--seed N]` and `torchrun --nproc-per-node 8
bench/moe/baseline.py --dir fwd|bwd --dist 0|1 --shapes FILE [--warm N] [--iters N] [--reps N] [--mxfp8] [--gstd S]`.
Per run a CSV and a log: KOSMOS `<dir>_d<dist>[_mx]_<STAMP>.csv` (`Shape, dist, T, H, I, E, k, kosmos_ms,
kosmos_warm_ms, correctness, state, rep`, plus the ablation's columns), the baseline
`<dir>_d<dist>[_mx]_base_<STAMP>.csv` (`Shape, dist, T, H, I, E, k, torch_ms, torch_warm_ms, base_correctness, rep`);
`build/kosmos_report moe` joins them per shape and rep into `SUMMARY_<STAMP>.md` (a baseline FAIL fails the shape; a
missing baseline row shows `NOBASE`).

* Arms: KOSMOS (`kosmos_moe_forward` / `kosmos_moe_backward`, `cfg.backward`: the forward a training forward that stores
  `G`), plus KOSMOS unfused with `--unfused` in a random order every iteration. `--mxfp8`: MXFP8 training instead
  (`cfg.mxfp8` and `cfg.backward`, bf16 combines; `kosmos_moe_forward_mxfp8_train` / `kosmos_moe_backward_mxfp8`, inputs
  quantized as `moe_fwd --mxtrain` and `moe_bwd_dx --mxfp8 1` do, the MXFP8 CPU reference), same CSV; a shape whose
  configuration the library refuses is skipped with the reason. The PyTorch baseline in its own process, on the same
  inputs (the same generators, bit for bit) and routing (`gen_routing_ref` files in `$KOSMOS_WORK/routing`).
* Rep 0 checks every KOSMOS arm twice: the second call against the CPU reference (1024 sampled tokens, and for the
  backward 1024 weight-gradient columns, per rank), both calls bitwise equal; `state` reads back the counters, signal
  planes and zero rows after each call. The baseline checks one call against an fp32 PyTorch reference with the same
  per-tensor rtol/atol (1024 sampled tokens per rank; the weight gradients in full); the MXFP8 baseline is gated
  by SNR >= 20 dB and cosine >= 0.995 per output against the same fp32 reference (no MXFP8 quantization modelled),
  while KOSMOS MXFP8 is checked elementwise against the MXFP8 CPU reference.
* Cold: before every call the destinations are poisoned, the host synchronizes and the ranks barrier. Forward:
  events bracket the call. Backward: each call follows its own arm's forward on the same stream, and the backward's
  device span is timed. Warm, after the cold iterations: at least 2 s of back-to-back calls, then 16 calls back to
  back in chunks of 4 (host sync and barrier only between chunks, no poisoning), each timed by its own events. Every
  arm sleeps the GPU for 4M cycles (`torch.cuda._sleep` in the baseline, a device sleep kernel in `moe_bench`) before
  each timed region (and each warm chunk), so the host's dispatch before the first kernel is not timed; the baseline's
  count synchronization stays inside its timed region.
* Statistic (as TP+SP): per iteration the slowest rank, the median over `ITERS` after `WARM` warm-ups, then the
  report's median over the reps. Speedup = PyTorch / KOSMOS. TFLOP/s count routed rows only (forward `6 T k H I`,
  backward `12 T k H I` per rank).
* `build/moe_fwd`: `--shape`, `--T` (8192), `--dist` (0), `--iters` (6), `--warm` (0), `--wgcap` (256), `--check N`
  (1024; -1 all; 0 off), `--sweep` (label), `--wcalls N` (0; N warm calls after the cold ones, a multiple of 4),
  `--mxfp8` (the MXFP8 forward), `--mxtrain` (the MXFP8 training forward), `--fp8comb 1` (with `--mxtrain`:
  `cfg.fp8_combine`), `--act N` (`cfg.act`; default the shape's), `--gstd S` (W1 drawn so that G has standard deviation
  S; default the harness's, about `sqrt(H)`), `--cap ROWS` (`-n`: the routing's rows / n; `capacity_rows` with the
  routing on the device instead of `kosmos_moe_plan`: a routing over it runs in rounds, its status and `out` checked
  against one pass on a second workspace with enough capacity). `launch.sh` takes `WARM` (2), `CHECK` (0), `WGCAP`, `WCALLS`, `MXFP8`,
  `MXTRAIN` and `FP8COMB` from the environment and refuses a binary older than its sources; `gate.sh` takes `MXFP8`,
  `MXTRAIN` and `FP8COMB` too.
* `build/moe_bwd_dx`: `--shape`, `--T` (8192; gate8 1024), `--dist` (0), `--wgcap` (256), `--iters` (2), `--check N`
  (1024), `--timing N` (N timed iterations, no check), `--wcalls N` (with `--timing`: N warm forward + backward pairs
  after the cold ones), `--mxfp8 1` (MXFP8 training forward and backward), `--fp8comb 1` (with `--mxfp8 1`:
  `cfg.fp8_combine`), `--act N`, `--gstd S` (default 1). `check.sh` passes extra flags through and takes `CHECK`,
  `DIST`, `ITERS`, `WGCAP` and `MXFP8` from the environment.
* `build/moe_train`: `--shape`, `--T`, `--calls` (4), `--check N`, `--wgcap`, `--cap ROWS` (0: the largest routing's
  need), `--timing R` (R rounds of the calls, `--twarm W` (1) of them discarded; `--poison 1`: the outputs poisoned
  on the stream before every call, outside the timed span), `--layers N` (the multi-layer check; not with
  `--timing` or `--cap`), `--act N`, `--gstd S` (default 1; the shared expert's too), `--router sig|soft|ssp|pre`
  (the fused router's presets), `--force 0|1`, `--lbias 1` (a logit bias, and its gradient checked), `--ties 1`
  (`Wr` rows and the biases equal on expert pairs: exact ties, resolved toward the lower expert), `--sgate 0|1` (the
  shared expert gated, `out = routed + s_t shared` with `s_t = sigmoid(x_t w_sg)`, and `dsgate` checked; default on
  for a shape with a gate, `qwen35`, with `--shared 1` and one layer), `--pin 0|1` (`cfg.probs_in`; default the shape's;
k = 1, no `--router`), `--fp8comb 1` (with `--layers` and `--mxfp8 1`: `cfg.fp8_combine`). `--mxfp8 1` takes
`--shared 1` (with `--sgate`, `--acc`), checked against the MXFP8 shared-expert reference, and `--router`.
`--api 1` runs the training API's scenarios (`check.sh` above) and takes `--T`, `--check`, `--wgcap`, `--act`, `--gstd`
and `--skew` only.
* The harnesses refuse unknown flags, and compile the library's host API into their own translation unit.

**PyTorch baseline (`bench/moe/baseline.py`).** Eager PyTorch on the current stream, collectives through
`torch.distributed` (RCCL): a stable `torch.argsort` of the routed slots by expert and `index_select` of the token rows;
the per-expert counts exchanged with `dist.all_to_all_single` and copied to the host (one host synchronization, as in
Megatron's all-to-all dispatcher); the rows dispatched with `dist.all_to_all_single` (uneven splits) and sorted by local
expert; per-expert `torch.mm` (hipBLASLt, PyTorch's default algorithm choice); SwiGLU as `F.silu(gate) * up` (a shape
with `act` 1: `F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)`); the GEMM2 rows sent back with `dist.all_to_all_single`
and combined as an fp32 weighted sum. Backward, after its own forward: `dY = bf16(w * dOut)` and `dtopkw = <dOut, Y>`
(scattered into dense `dprobs`), dispatch, per expert dgrad2 and wgrad2 (`torch.mm`), `aten.silu_backward` (masked as
TE's for `act` 1), per expert dgrad1 and wgrad1, combine and the dX sum over the top-k, with no host synchronization.
The experts are one `torch.autograd.Function` (forward: the per-expert GEMMs and SwiGLU, saving X, G and A; backward:
the dgrad / wgrad GEMMs above, run by `torch.autograd.backward`), so the forward is a training forward in both
precisions. MXFP8 (`--mxfp8`, Megatron's MXFP8 grouped MLP): the same dispatch and combine around Transformer Engine's
`Fp8Padding` (256 rows per expert), `GroupedLinear` fc1 and fc2 under `te.autocast` with `MXFP8BlockScaling` on the
HipKittens MXFP8 grouped GEMM (`NVTE_USE_HIPKITTENS_GROUPED_GEMM=1`; fallback lines are counted per run), torch
SwiGLU and `Fp8Unpadding`; weights quantized once (`is_first_microbatch`), the backward by autograd.
`torch._grouped_mm` is not used: its ROCm path (CK) reads the group offsets on the host after a stream synchronization
and copies a non-column-major B, every call.

**MegaMoE comparison (optional, internal only, `bench/megamoe`).**
`PRIMUS_TURBO_ROOT=/path/to/Primus-Turbo bench/megamoe/run.sh fwd|bwd <shape> [dist]` times Primus-Turbo's fused
MegaMoE op under the same protocol (Python, PyTorch and a built Primus-Turbo checkout required; not shipped);
`bench/run_all.sh mega` runs it for every shape of `MOE_SHAPES`, direction and routing, `REPS` sweeps each, and the
report adds it to the results when its logs exist.

* Forward (`mega_fwd2.py`): the op's full forward (routing prologue, dispatch, GEMM1, SwiGLU, GEMM2, combine) as one
  timed region per iteration, after poisoning its combine output buffer, a host synchronization and a barrier.
* Backward (`mega_bwd.py`): `torch.autograd.grad` of the op (dx, dW1, dW2, dtopkw), timed right after an untimed
  forward on the same stream.
* Routing: `gen_routing.sh` writes the harnesses' own routing draw (`gen_routing_ref.cpp`), so both arms run on
  the same tokens-to-experts assignment.
* Its output is compared with a warm-up result as relative L2 error, max over ranks; gated only when `MEGA_TOL` is
  set. `run.sh` also takes `T`, `SWEEPS`, `WARM`, `ITERS` and `WCALLS` (warm calls, the same protocol as ours:
  `sweep N warm:`) and refuses `WGCAP`.

Results: [`results/MI350X.md`](../../results/MI350X.md).

### Ablations

```
ABLATIONS= bench/moe/run.sh                            # without the KOSMOS unfused arm (on by default)
```

**KOSMOS unfused (`moe_bench --unfused`; bf16, or MXFP8 training with `--mxfp8`).** On by default (`ABLATIONS=`
leaves it out). The same call as separate launches of the library kernel's own code, back to back on one stream, as a
second arm of the same process (cold only, checked like the others in rep 0). Forward: plan, dispatch, GEMM1 + SwiGLU,
GEMM2, combine + fold; backward (after the fused forward): dispatch + dtopkw, dgrad2 + wgrad2, dgrad1, wgrad1, combine +
fold. MXFP8: the plan launch also quantizes X and the backward's dispatch launch the received dY rows, so the GEMM
launches read only what earlier launches wrote; the column-wise Xp goes with the dispatch that pushes it (the forward
for I < 1024, else the backward). The launches are bench-only instantiations (template `PART` in `moe_fwd_dev.cuh`,
`MOE_BWD_PART` in `moe_bwd_single.inc`; the library ships only the fused kernels), one object per precision, kernel
family (k in {6, 8, 10}, small k, probs on input) and activation, with one epoch per call and the reset in the last
launch; each is padded to the LDS of its precision's fused k >= 6 kernel, so it keeps one workgroup per CU. CSV columns:
the five parts of the rank with the largest unfused time per iteration (median), `unfused_comm_ms` (plan, dispatch,
combine), `unfused_gemm_ms` (the GEMM launches), `unfused_ms`, `overlap_eff = (comm + GEMM - KOSMOS) / min(comm, GEMM)`
and `speedup_vs_unfused = unfused / KOSMOS`.
The report puts them in its `Ablations` section.
