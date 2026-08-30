<!--
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TE-ROCm Divergence Manifest v2.4 — Summary

**Base:** fork `8af6efcd9a40…` (dev, IFU-2.18 merge of 2026-08-27) vs upstream **`868d8d9216da…`** on
`main`. Measured 2026-08-28 with **GNU `diff`** (normative; git's Myers algorithm disagrees).

> ### v2.3 → v2.4: the measurement base was wrong
>
> v2.3 measured against `release_v2.18`'s **tip** (`27486e03`). The fork never merged that. The IFU
> merge commit `3a49d650` has upstream parent **`868d8d92`**, which is exactly
> `merge-base(upstream/main, release_v2.18)` — the point 2.18 branched off main.
>
> Measuring against the release tip charged **15 upstream commits (1,339 lines in scope)** to ROCm.
> Consequences: **five entries were wholly phantom**, ten were inflated, and the reported divergence
> **trend inverted**. All corrected here. `upstream_sha` is now derived from the merge commit's second
> parent, and CI asserts it equals `merge-base(upstream/main, fork/dev)` so this cannot recur.

**Status:** file-complete; feature decomposition and `added_class` regeneration due at Stage 0 exit.
**Schema v1.3** retains v2.1.1 governance (`runtime_eligible: never|candidate|proven` +
`runtime_evidence`; M1 added/removed split; structural numeric fields) and adds `v23_values`,
`stale_added_class`, `retired_entries`, `layer_totals`, `rocm_only_volume`, `open_decisions`, and
`seam_early_binding_sites`.

---

## Corrected divergence (GNU diff, vs `868d8d92`, modified files only)

| framework | files | added | removed | sum |
|---|---|---|---|---|
| PyTorch Python (PT-001..045, 41 live) | 41 | 2,142 | 253 | 2,395 |
| JAX Python (JX-001..019, 18 live) | 18 | 563 | 120 | 683 |
| **Vendored-tree Python (CM-001..003) — NEW** | **3** | **241** | **40** | **281** |

| C/C++ layer | files | added | removed | sum |
|---|---|---|---|---|
| `common/` **C++ only** | 85 | 6,023 | 297 | 6,320 |
| `common/` **Python** | 2 | 205 | 36 | 241 |
| NVTE public headers | 9 | 329 | 3 | 332 |
| `pytorch/csrc/` | 21 | 1,298 | 49 | 1,347 |
| `jax/csrc/` — **newly measured** | 15 | 239 | 25 | 264 |

| tests | files | added | removed | sum |
|---|---|---|---|---|
| `tests/pytorch` | 46 | 2,188 | 211 | 2,399 |
| `tests/cpp` | 35 | 1,956 | 132 | 2,088 |
| `tests/jax` | 13 | 951 | 73 | 1,024 |
| `tests/cpp_distributed` | 0 | 0 | 0 | 0 |

**Totals:** non-test **11,622**; all in-scope **17,133**.

> The non-test total is *not* directly comparable to v2.3's "~12,000": v2.4 additionally counts
> `jax/csrc` (264) and the package root (40), which v2.3's entry list omitted despite its declared
> scope. On v2.3's own narrower footing the corrected figure is **11,318**.

`added_class` (sidecar / guard / unmarked) is **withheld in v2.4**. Hunk classification is
base-dependent — upstream changed several of these files between `868d8d92` and the release tip, so
added-line composition can shift even where the added *count* is unchanged. Every v2.3 triple is
preserved as `stale_added_class`; canonical-v2 must be re-run before Stage-0 exit.

---

## Five entries retired — zero divergence at the true base

| ID | path | v2.3 delta | actually |
|---|---|---|---|
| **PT-005** | `pytorch/ep.py` | 85 / 74 | upstream PR #3187 (NCCL EP zero-copy) |
| **PT-015** | `tensor/storage/float8_blockwise_tensor_storage.py` | 0 / 56 | upstream #3242 / #3171 |
| **PT-016** | `pytorch/newton_schulz.py` | 8 / 34 | upstream cuSolverMp changes |
| **PT-042** | `pytorch/ops/op.py` | 5 / 0 | upstream #3242 |
| **JX-018** | `jax/router.py` | 2 / 1 | upstream #3237 |

IDs are burned, never reused.

**PT-005 was v2.3's headline "new 2.18 divergence front."** It is upstream's own commit. Of the two
claimed new fronts, only `jax/moe.py` (JX-005, 48/0) is real — confirmed unchanged at the corrected
base.

**The removal-dominant entries were the tell.** v2.3 characterised PT-014 as *"fork removes upstream's
torch symm-mem MemPool helpers."* Wrong in kind: upstream **added** those 51 lines in #3187 *after* the
branch point. The fork never had them and never removed them. PT-014 is **4 / 0**, not 9 / 51.

Ten entries inflated rather than phantom — largest: PT-001 `cpp_extensions/gemm.py` 312/46 → **302/1**;
PT-003 `module/grouped_linear.py` 162/78 → **149/36**; PT-002 `module/base.py` 289/58 → **289/23**.

---

## The trend claim reverses

v2.3 reported *"every layer grew … PT +110 added / +323 removed"* as the live regrowth signal. Both
snapshots were measured against release **tips** the fork never merged. The 2.17 IFU merged
`2e559f06`, which is on `release_v2.17` but **not on main** — confirmed by the team as done the wrong
way. Measuring each snapshot against the base it actually merged:

| PyTorch Python | 2.17 | 2.18 | Δ |
|---|---|---|---|
| v2.3 (vs release tips) | 41 f / 2,166 / 298 | 45 f / 2,276 / 621 | +4 f, **+110 / +323** |
| **v2.4 (vs true bases)** | 40 f / 2,156 / 265 | 41 f / 2,142 / 253 | +1 f, **−14 / −12** |

PyTorch Python divergence **shrank** across the IFU cycle. JAX growth is real but modest: 16 → 18
files, +64 added. The regression alarm remains a sound mechanism; it simply has **no observed
instance** justifying it yet.

The 2.17 mis-merge left **no content residue** — `merge-base(fork/dev, main)` is now exactly
`868d8d92` and the EP work from those 10 non-main commits is present at that base. Its cost was
conflict burden during that IFU.

---

## New: 281 lines of ungoverned vendored Python (CM-001..003)

v2.3's `measured_scope` said "transformer_engine (all layers)" but its entries covered only
`pytorch/` and `jax/`. Three diverged files had **no entry, no owner, no test_ids, no expiry** — and
they are the three the plugin architecture leans on hardest:

| ID | path | delta | why it matters |
|---|---|---|---|
| **CM-001** | `transformer_engine/__init__.py` | 36 / 4 | **the activation seam** §3.2 targets |
| **CM-002** | `common/__init__.py` | 119 / 31 | **BOOT-001's real target** |
| **CM-003** | `common/recipe/__init__.py` | 86 / 5 | **`CustomRecipe`'s home** |
| CM-004 | `common/ck_fused_attn/check_aiter_mha_args.py` | 112 (ROCm-only) | relocates with CK |

### The `common/` Python/C++ boundary

`common/` is **223 files: 208 C/C++/CUDA and 9 Python (3,887 upstream lines)**. BK-001 previously
moved it *wholesale* to a C++ backend repo. It cannot: the Python is the package root and the recipe
API, imported by both framework layers. Only **2 of 9** diverge (241 lines); `common/triton/*.py`
(6 files, 2,759 ln, imported directly by the pytorch layer) and `common/utils.py` are **verbatim**.

BK-001 is now C++ only (6,320 ln); **BK-001B** carries the Python.

### Three consequences for the architecture

1. **BOOT-001 rescoped: 3 → ~190 lines.** v2.3 marked it `measurement_quality: exact` at 3 changed
   lines across "2–3 files." The real surface is CM-001 + CM-002 across 21 hunks, spanning native
   loading, PyPI package identity, the `te_rocm_build` global, and `is_fp8_fnuz`.

2. **A packaging contradiction (CM-002-PKGID).** §3.2 decides the ROCm distribution keeps the
   canonical name `transformer-engine`. `common/__init__.py` currently implements
   `transformer-engine-rocm7` / `-rocm10` with `rocm_pytorch` / `rocm_jax` extras, and **changes an
   upstream function signature** (`get_te_core_package_info(rocm: bool)`). Stage 1 must *rewrite*
   this, not hook it.

3. **§3.7 needs amending (CM-003-RECIPEMXFP4).** MXFP4 is routed through the CustomRecipe adapter, but
   `Recipe.mxfp4()` is a classmethod **injected into upstream's base `Recipe` class**, and
   `MXFP4BlockScaling` is a `Recipe` subclass — not a `CustomRecipe`. `CustomRecipe(qfactory=…)` does
   not discharge either.

---

## New: ABI-001 — a contract path §3.3 doesn't model

`Format.E4M3.max_fwd` — a pure **upstream** Python API — resolves through `is_fp8_fnuz()` → ctypes →
**`nvte_uses_fp8_fnuz()`**, implemented in `common/amd_detail/system.cpp`. Neither that symbol nor
`nvte_is_rocm_build` is declared in the public NVTE headers, so both sit **outside the 332-line
HDR-A/B/C accounting entirely**.

After the split those symbols live in the backend `.so` — so vendored-upstream Python reaches
`TE_ROCM_CORE_ABI` **directly, bypassing `TE_ROCM_EXTENSION_API`**, which is supposed to be the only
route to the backend. Required before Gate B: complete the ctypes symbol inventory, then either route
these through the extension API or define a fourth versioned contract surface.

Related: **`is_fp8_fnuz` is implemented three times** — CM-002 (ctypes), PT-021 `pytorch/utils.py:644`,
and REL-003 `jax/util.py:25`, which **shells out to a subprocess**. Strongest concrete case for the
§3.6 capability provider, and its obvious first item.

---

## New in v2.4.1: ABI-002 — part of the PyTorch seam is defined under `common/`

Found by the static seam inventory (`tools/seam_inventory.py`). `pybind.cpp:216` expands
**`NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m)`**, a macro in `common/util/pybind_helper.h` that
registers 17 names on the `transformer_engine_torch` module — 10 enums, 3 stateful `CommOverlap*`
classes, 4 functions — **11 of them demanded by upstream Python**, including `DType`, all six `NVTE_*`
attention enums, `CommOverlapType`, and `get_stream_priority_range`. The first inventory pass scanned
only `pytorch/csrc` and reported all of them as missing.

That header sits in the `common/` C++ bucket — the **backend** side of the BK-001 split. So a
substantive part of the torch extension's Python surface is defined by a file that moves to the
backend repo. Same family as ABI-001: a contract crossing the manifest didn't model. Stage 0 must
decide whether the torch extension carries the header, or the extension API's inventory is generated
from backend headers.

**The fork's entire divergence in that file (25/11) is the `NVTE_Fused_Attn_Backend` enum split**
(ABI-002-FAENUM). The fork's macro binds *different members* per branch — CUDA
`{F16_max512_seqlen, F16_arbitrary_seqlen, FP8, No_Backend}`, ROCm `{AOTriton, CK, No_Backend}` —
under the same Python name. Upstream Python early-binds this enum and compares against members that
don't exist on ROCm, which is the real reason the PT-010/011/012 attention patches exist. Two
consequences:

- §3.5's "enums re-exported from the compiled extension" does not resolve it — it re-exports an
  enum with the wrong members. Until HDR-B2 lands, the facade ships an enum with **both** member
  sets, or PT-010/011/012 remain build-tier patches.
- A **name-only** inventory reports this enum as supplied. `TE_ROCM_EXTENSION_API` conformance
  must cover enum *values*. Added as backtest case 12 and a Stage-0 exit requirement.

---

## v2.4.2 / v2.4.3: two repository decisions and the C++ strategy

**One repository.** All C++ stays in `ROCm/TransformerEngine`; the compiled layers become build
targets, not a second repo. BK-001/002/003 re-dispositioned `backend-repo` → `build-target`.

**Upstream is a submodule.** `3rdparty/transformer_engine_nvidia` at `868d8d92`, with
`update = none` so the repo's `--init --recursive` skips it (upstream's `cutlass`/`nccl` never enter
the build); the assembler initializes it explicitly. Its gitlink is now the **source of truth** for
`upstream_sha`; the merge-base is a cross-check. CI asserts the three-way identity.

**C++ maintenance strategy — provisional.** The 85 modified upstream C++ files are all hipified and
78 carry ROCm guards (~480 sites); they are a merge product, not an AMD backend. Schema 1.4 adds
`cxx_strategy` (`freeze` | `patch-queue` | `native-hip`), set **provisionally to `patch-queue`** on
BK-001 — vendored upstream source from the submodule + build-time guard patches, then hipify as
today. Open for discussion as `open_decisions.cxx_maintenance_strategy` (due Stage 4); the new
**`backtest_plan.cxx_arm`** measures the guard-patch trip rate across 2.15→2.17 to decide it.
The 65 ROCm-only native files carry no `cxx_strategy` — they have no upstream ancestor.

---

## New: the seam's early-binding surface

Upstream has **44** late-bound `import transformer_engine_torch as tex` sites and **9** early-binding
`from transformer_engine_torch import …` sites — including a **star-import** at
`pytorch/cpp_extensions/__init__.py:6`. The facade must be installed in `sys.modules` before any of the
9 import. The star-import copies the entire facade namespace at import time, so **facade `__all__`
fidelity is load-bearing, not cosmetic** — any ROCm-extra symbol leaks into
`te.pytorch.cpp_extensions`. Add `__all__` equality to the EXTENSION_API conformance test.

---

## EP: cleaner than v2.3 claimed

`capability_graph_ep` is retained but is **no longer evidenced by PT-005/PT-014**, which were
artifacts. EP Python (`pytorch/ep.py`, `jax/ep.py`, `cpp_extensions/ep.py`, `flax/moe.py`) is
**verbatim upstream** in the fork — zero delta — with the FFI define off on ROCm, and EP tests at all
three levels are verbatim. That makes EP a *better* demonstration of collection-time,
capability-driven skips: there is no fork divergence to retire, only a capability to model.

---

## ROCm-only volume (migration_volume, not overlap debt)

| area | files | lines |
|---|---|---|
| `common/` (all ROCm-only) | 65 | **20,968** |
| ↳ `ck_fused_attn` / `aotriton` / `rocshmem_api` / `amd_detail` | 9 / 2 / 3 / 2 | 1,956 / 138 / 202 / 130 |
| pytorch Python | 22 | 8,370 |
| ↳ REL-001 `triton_kernels/` | 18 | 6,566 |
| ↳ REL-002 MXFP4 + fsdp2 | 4 | 1,804 |
| `pytorch/csrc` | 2 | 433 |
| `tests/pytorch` / `tests/cpp` | 21 / 6 | 5,314 / 3,116 |
| REL-003 `jax/util.py` | 1 | **37** (v2.3 estimated 120) |

v2.3 called ROCm-only volume "~17,000, estimate, not line-counted." It is now counted: `common/` alone
is 20,968.

---

## Four decisions now open

1. **`release_218_gap`** *(Stage 0)* — the fork sits at the 2.18 **branch point**, not the release.
   Pinning `868d8d92` ships a distribution labelled 2.18 that lacks 2.18's entire stabilisation
   (#3242, #3171, #3187, #3056, #3269, and the VERSION bump). Take the 15 commits, or pin a later main
   commit?
2. **`ifu_sourcing_policy`** *(Stage 0)* — 2.17 merged off-main; 2.18 merged a main branch point.
   Neither follows a stated policy, because none exists. Write one: always merge upstream `main` at a
   chosen SHA.
3. **`contract_surface_for_ctypes`** *(Gate B)* — ABI-001.
4. **`packaging_name_conflict`** *(Stage 1)* — §3.2 vs CM-002-PKGID.

---

## Metrics

M1 physical-overlap-debt (added/removed separately) / M2 carried-compatibility-debt. Creating a patch
closes M1 and opens M2; only true retirement reduces M2. `metric_class` (`m1` | `migration_volume` |
`inventory_only`) keeps ROCm-only relocation volume out of overlap debt.

## Backtest cases

Eleven risk-role cases (was eight). **CM-002**, **CM-003** and **ABI-001** were added — the bootstrap,
recipe and ctypes surfaces are the highest-risk items and were entirely absent from v2.3's list.
Stage 2 must derive the 2.15 base the same way: from that IFU merge commit's second parent, never from
`release_v2.15`'s tip.
