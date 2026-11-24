<!-- Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
     License for AMD contributions = MIT. See LICENSE for more information -->

# MoE (expert parallel): the bf16 training forward and backward

An EP=8 bf16 MoE layer (top-k routing, GLU experts) for gfx950, behind one training API.
The forward and the backward are each **one kernel launch** on the caller's stream.
The forward takes the routing in device memory and plans it in the same launch.
The token all-to-all (dispatch and combine) runs inside the same kernel as the grouped GEMMs, through symmetric peer buffers (`KosmosComm`).
Everything a backward reads from its forward lives in buffers the caller owns, so forwards and backwards may come in any order.

* Library: `src/moe` plus `src/utils`, built as `build/libkosmos_moe.a` (`make moe`, which also builds `build/moe_linkcheck`). No MPI and no other communication library.
* Kernels: the forward `moe_fwd_single` (`moe_fwd_dev.cuh`, `moe_fwd_pass.inc`, its start phase in `moe_plan.cuh`) and the backward `moe_bwd_single_train_api` (`moe_bwd_single.cuh`, `moe_bwd_single.inc`, `moe_bwd_pass.inc`), instantiated in `moe_train_api.cu` (k in {6, 8, 10}) and `moe_train_api_ks.cu` (k in {1, 2, 4}, which differ only in their folds).
* Each kernel translation unit is built once per activation: SwiGLU, and with `-DMOE_ACT_CLAMP10` the clamped GLU (`*_c10.o`).

## How one launch works

Both kernels launch `wgcap` persistent workgroups (8 waves each, at most one per CU).
A few workgroups start in a communication role and join the tile loop when their pushes run out; all others claim GEMM tiles from queues whose order puts every producer ahead of its consumers.
Cross-GPU data is published with a release atomic (or wide write-through stores drained with `vmcnt(0)` before a relaxed add) and acquired by the consumer after it observes the count.

* **Forward.** The plan sorts this rank's routing and publishes only per-expert counts to every rank; every workgroup then derives the whole layout in LDS.
  The dispatch crew pushes token rows to each expert's owner; GEMM1 + GLU tiles start as soon as their row block has arrived, GEMM2 tiles as soon as their row block's GEMM1 tiles are done.
  The combine crew pushes `Y` rows back to the token owners, and the fold computes `out[t] = sum_j w[t][j] Y[t][j]` in fp32 and stores bf16.
  The forward also writes what its backward needs into the caller's saved buffers (below).
* **Backward.** Every workgroup first restores the forward's tables from the saved record.
  The dispatch crew pushes `dY = bf16(w dOut)` and computes `dtopkw[t][j] = <dOut[t], Y[t][j]>`; one tile queue runs dgrad2 with the GLU backward in its epilogue, wgrad2, dgrad1 and wgrad1; dual-role workgroups push `dXp` rows back; the fold computes `dX` (fp32 sum, round to nearest even) and `dprobs`.
* **Reset.** The last workgroup of each launch zeroes every counter; cross-call ordering uses monotone epochs, so there is no per-call memset or copy.

`dncomm`, `cncomm` and the wgrad run lengths are set once at instance creation from the shape (a cost model fed with a balanced routing's expected counts), the same on every rank.
The kernels are deadlock-free at any cap in [9, 256] because pushes wait only for their destination to start the same call, every queue bucket has a resident home workgroup, and every producer precedes its consumers in the queue.

## API

Include `src/moe/moe.h` (it includes `src/utils/kosmos_comm.h`) and link `build/libkosmos_moe.a -lamdhip64`.

```
kosmos_moe_instance_bytes(&cfg, &bytes)                      symmetric bytes of an instance (an external comm's)
kosmos_moe_instance_create(&inst, comm, &cfg)                collective, host-blocking; comm may be NULL
kosmos_moe_saved_bytes(&cfg, rows, &sb)                      bytes of each saved buffer (xp, g, a, y, route)
kosmos_moe_train_forward(inst, X, W1, W2, topk_idx, topk_w, out, saved, stream)   1 launch; saved NULL: inference
kosmos_moe_train_backward(inst, dout, X, W1, W2, topk_idx, topk_w, saved, dX, dprobs, dW1, dW2, acc, stream)
                                                             1 launch
kosmos_moe_instance_status(inst, &status)                    host-blocking: status bits since the last query
kosmos_moe_instance_info(inst), kosmos_moe_instance_error(inst)
kosmos_moe_instance_destroy(inst)                            local
```

An instance serves one EP group and one configuration.
It owns the kernels' settings and the symmetric buffers of the call in progress, all on one comm: `comm = NULL` makes a managed one; otherwise the comm must hold at least `kosmos_moe_instance_bytes` (e.g. `kosmos_comm_wrap` of a framework's symmetric memory, kept until the instance is destroyed).
Creation is collective and does everything a first call would, so every later call enqueues one kernel and can be captured into a graph.
The instance allocates once, at creation, at `capacity_rows` rows per rank (rounded up to 256); it never re-allocates.

Everything a backward reads from its forward lives in the caller's `KosmosMoeSaved` (every pointer 256-byte aligned; `tier` and `rows` equal on every rank): `xp` (the received rows Xp), `g` and `a` (GEMM1's pre-activations and its GLU output), `y` (Y by token slot, `[T k][H]`), and `route`, the route record: the call's kept tier, passes, rows and status, its count matrix, the receive slots and the tables its backward reads.
Their sizes come from `kosmos_moe_saved_bytes(&cfg, rows)`, where `rows` is the busiest rank's routed rows (every local expert padded to 256) the buffers keep; `route` and `y` do not depend on it.
So any number of forwards may be in flight, backwards come in any order, a forward whose backward never runs leaves nothing behind, and the buffers may move between a forward and its backward (`route` holds no address).
`X`, `W1`, `W2`, `topk_idx` and `topk_w` must hold the forward's values at the backward.

* **Tier.** A forward keeps `saved->tier`: `KOSMOS_MOE_SAVE_ALL` (Xp, G and A), `_XP` (Xp; its backward recomputes GEMM1) or `_NONE` (nothing; its backward pushes `X` again and recomputes).
  A call whose busiest rank routes more rows than `saved->rows`, or that runs in rounds, keeps nothing; every rank decides alike.
  Every tier gives the same outputs bit for bit.
* **Weight gradients.** `acc = 0`: `dW1`, `dW2` bf16, overwritten; `acc = 1`: fp32 (`main_grad`), each tile's fp32 sum added; `acc = 2`: bf16, each element `bf16(old + sum)` rounded once (an expert split over rounds adds once per round).
* **Status** (`kosmos_moe_instance_status`, the OR over the calls since the last query; it synchronizes the device). Errors: 1 an expert id outside `[0, 8 E_loc)` (dropped: it adds nothing), 2 a token routed twice to one expert. Notes: 8 a call ran in rounds, 32 a forward kept nothing where its tier asked for more.

### Over capacity

A routing can need more rows on some rank than the capacity.
The forward and the backward then run the call in rounds inside their one launch: each receiver's rows are cut into windows of at most the capacity (at expert boundaries, or inside a larger expert at 256-row multiples), and round `r` dispatches every source's rows of window `r` and runs the GEMMs and combine on them, reusing the same buffers.
Every rank derives the same windows from the call's counts, so all take the same number of rounds.
Such a call keeps `SAVE_NONE` and sets status bit 8 (and 32 if its tier asked for more); its backward re-derives each round's tables from the recorded counts, pushes `X` again and recomputes GEMM1.

### Configuration (`KosmosMoeConfig`, identical on every rank except `coll.rank`)

| field | meaning |
|---|---|
| `H`, `I` | hidden size, expert intermediate size |
| `E_loc`, `k` | experts per rank (global experts `8 E_loc`), experts per token |
| `T` | tokens per rank |
| `wgcap` | launch width of both kernels, in [9, 256] |
| `capacity_rows` | rows per rank the instance allocates for (> 0; rounded up to 256) |
| `coll` | all-gather callback over the 8 EP ranks (`KosmosCollective`), with `rank` and `nranks = 8` |
| `act` | `KOSMOS_MOE_ACT_SWIGLU` (0), or `KOSMOS_MOE_ACT_SWIGLU_CLAMP10`, DeepSeek-V4's `silu(min(g, 10)) clamp(u, -10, 10)` (TE's `clamped_swiglu_v2` with limit 10, alpha 1, offset 0), whose backward zeroes `dg` where `g > 10` and `du` where `abs(u) > 10` |

### Tensors (row-major; bf16 unless noted)

| tensor | shape |
|---|---|
| `topk_idx` | `[T][k]` int32 global expert ids in `[0, 8 E_loc)`, or -1 (dropped); expert `g` lives on rank `g / E_loc`; no repeats per token |
| `topk_w` | `[T][k]` fp32 router weights |
| `X`, `out`, `dout`, `dX` | `[T][H]` |
| `W1`, `dW1` | `[E_loc][2I][H]`, rows in blocks of 256: 128 gate rows, then the matching 128 up rows |
| `W2`, `dW2` | `[E_loc][H][I]` |
| `dprobs` | `[T][8 E_loc]` fp32: the gradient of `topk_w` at each token's routed experts, zero elsewhere |

`out`, `dX` and `dprobs` are overwritten.
Router weights stay fp32 and are applied at the fold (forward) and the unfold (backward).

### Contract

* Every rank issues the same calls in the same order on one stream.
  Calls are stream-ordered: no host synchronization or barrier is needed between them, on one rank or across ranks.
* A call enqueues one kernel and nothing else: no host synchronization, allocation, memset, copy or host collective. The pointers must stay valid until that kernel has run.
* `wgcap` bounds concurrency, not placement: every output element keeps one owner and a fixed reduction order.

### Limits

* EP = 8 ranks, one GPU each, in one node.
* `k` in {1, 2, 4, 6, 8, 10}; `E_loc <= 64`.
* `H` and `I` multiples of 256 (the GEMM tile), and below the 32-bit buffer offsets: `4 I H` and `4 T I` under 2^32 bytes, `2 T H` under 2^31; the capacity's `(rows + 1) 2H` and `(rows + 1) 4I` bytes under 2^32.

### Python

`make python` builds the module `kosmos` (`build/python`).
Every buffer and the stream (`hipStream_t`) is an integer address, 0 for null, and the caller allocates all of them.
A failed call raises `RuntimeError` with `kosmos_moe_instance_error`; a refused configuration raises `ValueError`.

```
kosmos.SAVE_NONE, kosmos.SAVE_XP, kosmos.SAVE_ALL, kosmos.ACT_SWIGLU, kosmos.ACT_SWIGLU_CLAMP10
kosmos.instance_bytes(H, I, E_loc, k, T, capacity_rows, rank, allgather, wgcap=256, act=0)
                                                # bytes to allocate for ext_* (the instance's and the comm's flags)
inst = kosmos.MoeInstance(H, I, E_loc, k, T, rank, allgather, wgcap=256, capacity_rows=..., act=0,
                          ext_local=0, ext_peers=0, ext_bytes=0)   # ext_*: a symmetric buffer, kosmos_comm_wrap'ed
inst.saved_bytes(rows)                          # {"xp", "g", "a", "y", "route"}
inst.forward(X, W1, W2, topk_idx, topk_w, out, tier, rows, xp, g, a, y, route, stream)   # tier -1: inference
inst.backward(dout, X, W1, W2, topk_idx, topk_w, tier, rows, xp, g, a, y, route, dX, dprobs, dW1, dW2, acc, stream)
inst.status(), inst.info(), inst.error()
inst.destroy()                                  # explicit: dropping the object does not free it
```

`allgather(send: bytes)` is the host all-gather over the 8 EP ranks: it returns the 8 ranks' `send` concatenated in rank order, as any bytes-like object.
It runs only while the instance is created.

## Training API internals

The forward decides the tier it keeps from its first pass's layout, alike on every rank (`TRAIN_INT_KEPT`, `moe_fwd_pass.inc`), and writes the saved buffers as follows; an inference forward (`saved` NULL) writes none of them.

| buffer | written by | tiers |
|---|---|---|
| `xp` | GEMM1, from the A tiles it holds in LDS (3 of every 4 row blocks, `XP_GEMM1_*`), and the combine's tail (the 4th) | XP, ALL |
| `g` | GEMM1's epilogue | ALL |
| `a` | GEMM1's epilogue (other tiers: the instance's scratch) | ALL |
| `y` | the fold, each token slot's row as it reads it | NONE, XP, ALL |
| `route` | the combine's tail (receive slots and tables, `route_fwd_copy`), then the last workgroup (header and count matrix, `route_fwd_finish`) | NONE, XP, ALL |

The backward first copies the record's tables back into the instance (`route_bwd_restore`).
It then reads Xp, G and A (ALL), reads Xp and recomputes G and A (XP), or pushes X again and recomputes both (NONE).

The route record (`kosmos_moe_saved_bytes(...).route` bytes; constants in `moe_fwd_dev.cuh`, layout in `moe_impl.cuh`, `route_layout`):

| byte offset | contents |
|---|---|
| 0 | header, 9 ints (`ROUTE_HDR_*`): kept tier, passes, the backward's rows + 1 (Mpad; in rounds the busiest rank's), Mpad, the busiest rank's Mpad, status bits, epoch, receive slots in use, `ROUTE_MAGIC` |
| 256 (`ROUTE_COUNTS_OFF`) | the call's count matrix, 8 x 8 `E_loc` ints |
| `off_slot`, 256-aligned | the receive slots, one int per row at the capacity |
| then, each 256-aligned | 14 instance tables in a fixed order (`route_tables`), copied in pieces of at most 64 KiB |

## Correctness

`bench/moe_train/check.sh <shape> [T] --api 1 [--act 1 --gstd 5]` runs `build/moe_train` on 8 ranks: four input sets (two balanced, two skewed routings) in 15 scenarios on an instance at the largest set's rows and on one at a rounds capacity.
The scenarios cover tiers ALL, XP and NONE, saved rows below the call's, two and three forwards in flight with their backwards in either order, an inference forward in between, abandoned and discarded forwards, saved buffers moved before the backward, and `acc` 1 and 2 over three calls on non-zero buffers.
Every output is compared bitwise with its set's first call on the same instance (each set alone, tier ALL), whose calls are checked against the C++/OpenMP CPU reference (`bench/common/moe_check.cpp`; small shapes in full, large ones on a deterministic sample); record headers, status bits and a guard after every saved buffer are checked; the whole sequence runs twice and must be bitwise equal.
`--dump P` writes every output's hashes, the routings' hashes and the size queries to `P.r<rank>`, and `--compare P` checks a run against such a dump (e.g. from another build).

```
make moe_train
bench/moe_train/check.sh gate8 --api 1
bench/moe_train/check.sh gate4 --api 1 --act 1 --gstd 5
bench/moe_train/check.sh dsv3 --api 1
```

`bench/run_all.sh check` runs the eight checks of `TRAIN_API_CHECKS` (k 8, 4, 2, 1; SwiGLU and the clamped GLU; dsv3, dsv4pro).
