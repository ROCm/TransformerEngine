<!--
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TransformerEngine on ROCm: From Hard Fork to Pinned Upstream + Plugin

**Proposal v2.2 — architecture, migration, implementation, and validation plan**

| | |
|---|---|
| Status | v2.2 — execution charter for Stage 0-2 approval (three external review rounds plus a base-measurement audit; review changelog in a separate file) |
| Date | 2026-08-28 |
| Author | Wen (+ team, TBD) |
| Measurement base | `ROCm/TransformerEngine` `dev@8af6efc` (full SHA `8af6efcd9a40…`, 2026-08-27, **IFU-2.18 merge**, version base 2.18.0.dev0) vs **`NVIDIA/TransformerEngine` `main` at `868d8d9216da…`** — the upstream commit the fork's IFU merge actually took, and exactly `merge-base(upstream/main, release_v2.18)`. **This base changed in v2.2**; see §2.0. Upstream `main` at 2.20.0.dev0 at time of writing. |
| Companion artifacts | `te-rocm-divergence-manifest.yaml` **v2.4** — file-complete ledger (41 PyTorch + 18 JAX + 3 vendored-root + 18 structural, rebased to the corrected base; five v2.3 entries retired as measurement artifacts), `te-rocm-divergence-manifest-summary.md` v2.4, `te-rocm-proposal-changelog.md` |
| v2.1.1 → v2.2 | **Measurement base corrected** (§2.0) — five phantom manifest entries retired, ten restated, the divergence trend reversed; `common/` Python↔C++ boundary resolved and BK-001 split; BOOT-001 rescoped 3 → ~190 lines; a fourth contract path (ABI-001) identified; §3.7 amended for base-class injection; four new open decisions |

---

## 1. Executive summary

The ROCm/TransformerEngine fork carries **~11,600 modified lines** interleaved with NVIDIA's files
across the core C++ library, headers, both framework binding layers, and the Python layers —
**~17,100 including the test suites** — plus **~29,800 lines of ROCm-only additions** (~38,200 with
test suites; now counted rather than estimated — `common/` alone holds 20,968, against v2.1.1's
uncounted "roughly 17,000"). Every upstream release forces a whole-repo merge: the
2.15→2.17 sync consumed roughly five weeks and the 2.17→2.18 IFU took ~3.5 weeks, while upstream cuts
minor releases roughly monthly.

This proposal restructures the project so that **pinned upstream TE Python is vendored with one
bootstrap patch and a governed, versioned compatibility layer**, with all ROCm functionality delivered
through a plugin occupying the framework-extension seam and an AMD-maintained compiled backend. The
refactor removes routine textual merge conflicts; it deliberately **replaces them with explicit,
versioned compatibility surfaces** that this document designs.

The seam is grounded in three independent implementations that converged on it: **TE Lite**
(`furmanek/dev-lite`, measured at parity — 1712.0 vs 1712.5 ms/iter, LLaMA-3-8B, 8×MI300X, May 2026 on
a 2.12 base, to be re-established); **FlagOS `TransformerEngine-Plugin-FL`**; and **upstream PR #3401**,
an `NVTE_PLUGIN` initialization hook approved by an NVIDIA TE core maintainer on 2026-08-20. Read
precisely: #3401 is strong evidence for an initialization-time interposition seam; it does **not** yet
define a stable multi-backend operator, quantization, or communication contract.

**Projected structural effects derived from the measured divergence** (targets, to be validated by the
Stage-2 historical backtest):

- Compiled layers move to an **AMD-maintained backend repo**: `common/` **C++ only** (85 modified
  files / 6,320 lines, plus 65 ROCm-only files / 20,968 lines) and `csrc/` (PyTorch 1,347; JAX 264).
  This removes routine textual merge overlap while **retaining an explicit semantic compatibility
  obligation**, managed through the versioned contracts of §3.3.
- **`common/` does not move wholesale.** It is 223 files, of which **9 are Python (3,887 upstream
  lines)** — the package root and the recipe API, imported by both framework layers. Only 2 diverge
  (241 lines); the Triton package and `utils.py` are verbatim. See §4.1.
- Of ~2,142 entangled PyTorch-Python added lines: a substantial share discharges via upstream's
  `CustomRecipe` extension point (behind a per-version adapter, with the §3.7 amendment), a further
  share converts to capability queries, and the remainder is genuine feature divergence, partly
  upstreamable. **Exact bucket percentages are withheld pending `added_class` regeneration** — hunk
  classification is base-dependent and every v2.1.1 percentage was computed against the wrong base.
- JAX (~563 added lines) is ~4× smaller; ~100 lines are upstream-PR material, ~75 vendor-neutral compat
  shims, ~120 workarounds for bugs in **AMD's own ROCm JAX plugin** (retire internally). JAX ships as a
  **separate beta track**.
- **Patch-floor hypothesis**: if every NVIDIA PR is rejected and no hook is extended, the remaining
  divergence is on the order of **~2,400 (PyTorch, incl. the vendored-root files) + ~350-450 (JAX)**
  lines in the two-tier compatibility layer. This floor **will be validated through a historical
  2.15→2.17 sync replay and two live upstream pin bumps** before being asserted as fact.
- Sync workflow **target**: reduce certified pin-bump effort from weeks to days. Stated as a target
  until the backtest measures it.

Per-op kernel **mix-and-match** remains a design goal of the registry but is **explicitly not on the
critical path**: the first production milestone runs the compiled backend exclusively.

**Authorization requested**: *Approve Stages 0-2 as a bounded architecture-validation program. Stage 0
must produce a governance-complete, exact-SHA manifest with regenerated hunk classification, resolve
the four open decisions of §11, and secure approved performance/backtest thresholds. Stage 2 provides
Gate A evidence for funding Stage 3. A certified Stage 3 package provides Gate B evidence for backend
extraction. This approval does not authorize backend movement, consumer migration, or fork sunset.*

---

## 2. Motivation: the measured problem

### 2.0 A correction that changes several conclusions (new in v2.2)

v2.1.1 and manifest v2.3 measured divergence against **`release_v2.18`'s tip** (`27486e03`). The fork
never merged that commit. The IFU merge commit `3a49d650` has upstream parent **`868d8d92`**, which is
exactly `merge-base(upstream/main, release_v2.18)` — the point 2.18 branched off `main`.

`release_v2.18` then took **15 further commits the fork does not have**, totalling **1,339 lines across
38 in-scope files**. Measuring against the release tip charged all of that to ROCm. Effects:

- **Five manifest entries were wholly phantom** — PT-005 `ep.py` (85/74), PT-015 (0/56), PT-016 (8/34),
  PT-042 (5/0), JX-018 (2/1) — with **zero** divergence at the true base. PT-005 was v2.1.1's headline
  "new 2.18 divergence front"; it is upstream PR #3187.
- **The removal-dominant entries were misread in kind.** §4.2 previously stated *"the fork removes
  upstream's torch symm-mem MemPool helpers."* Upstream **added** those 51 lines in #3187 *after* the
  branch point; the fork never had them. PT-014 is 4/0, not 9/51.
- **The divergence trend inverted** — see §2.2.
- Ten further entries were inflated (largest: `cpp_extensions/gemm.py` 312/46 → 302/1).

Two procedural fixes follow, both now Stage-0 gated: `upstream_sha` is **derived from the IFU merge
commit's second parent** and CI asserts it equals `merge-base(upstream/main, fork/dev)`; and the diff
tool is **pinned to GNU `diff`** (git's default Myers algorithm disagrees on some files, and M1
burn-down plus the divergence-regression alarm are line-count based).

A separate finding: **the fork is not at 2.18.** It sits at the branch point, missing 2.18's entire
stabilisation — FP8 block scaling for fusible ops (#3242), the fused FP8 block-scaling path in
GroupedLinear (#3171), NCCL EP zero-copy (#3187), cuDNN D=256 BWD attention (#3056), a CP padding
bugfix (#3269), and `Changed VERSION to 2.18.0` (hence `2.18.0.dev0`). Since the architecture vendors a
**pinned** upstream, which commit we pin is now an explicit Stage-0 decision (§11).

### 2.1 Divergence inventory (dev@8af6efc vs upstream main @868d8d92)

| Layer | Modified | ROCm-only additions |
|---|---|---|
| `common/` **C++** (→ core lib) | 85 files / 6,320 lines | 65 files / 20,968 lines incl. trees `ck_fused_attn/` (1,956), `aotriton/` (138), `rocshmem_api/` (202), `amd_detail/` (130) |
| `common/` **Python** (stays vendored) | **2 files / 241 lines** | 1 file / 112 lines |
| NVTE public headers | 9 files / 332 lines | — |
| `pytorch/csrc/` (→ torch ext) | 21 files / 1,347 lines | 2 files / 433 lines |
| `jax/csrc/` (→ jax ext) | **15 files / 264 lines** *(newly measured)* | 0 |
| `pytorch/` Python | **41 files / 2,142 added + 253 removed** | 22 files / 8,370 lines |
| `jax/` Python | **18 files / 563 added + 120 removed** | 1 file / 37 lines |
| **package root** `transformer_engine/__init__.py` | **1 file / 36 added + 4 removed** | — |
| `tests/pytorch` | 46 files / 2,399 lines | 21 files / 5,314 lines |
| `tests/cpp` | 35 files / 2,088 lines | 6 files / 3,116 lines |
| `tests/jax` | 13 files / 1,024 lines | 0 |
| `tests/cpp_distributed` | 0 — carried verbatim | — |

Counts are added+removed under GNU `diff`. Three rows are new to v2.2: `common/` Python, `jax/csrc`,
and the package root — all inside the declared measured scope but absent from v2.1.1's tables.

### 2.2 Why the current model is expensive

Every modified line lives in a file NVIDIA also edits; sync cost scales with upstream's velocity. The
2.15→2.17 sync reconciled attention almost entirely (`context_parallel.py`: 1,619 diverged lines → 23)
— proof convergence is achievable.

**v2.1.1's counter-force claim does not survive the base correction.** It reported *"every layer grew —
PyTorch Python +110 added / +323 removed"* as a regrowth dynamic "observed live across a single IFU
cycle." Both snapshots were measured against release **tips** the fork never merged. Measured against
the bases actually merged:

| PyTorch Python | 2.17 | 2.18 | Δ |
|---|---|---|---|
| v2.1.1 (vs release tips) | 41 f / 2,166 / 298 | 45 f / 2,276 / 621 | +4 f, +110 / +323 |
| **v2.2 (vs true bases)** | 40 f / 2,156 / 265 | 41 f / 2,142 / 253 | +1 f, **−14 / −12** |

PyTorch Python divergence **shrank slightly** over the cycle. JAX growth is real but modest (16 → 18
files, +64 added), and only one of the two claimed new fronts — `jax/moe.py` — exists.

The regression alarm of §5 remains justified as a *mechanism*: nothing prevents regrowth, and the
manifest must detect it. But it is a precaution, **not a response to an observed event**, and this
document no longer claims otherwise.

**A second cost source, previously unattributed: our own IFU procedure.** The 2.17 IFU merged
`2e559f06`, which is on `release_v2.17` but **not on `main`**, pulling 10 non-main commits into fork
history and contributing conflict burden to the 2.18 IFU. The 2.18 IFU merged a main commit but took
the branch point. Neither follows a stated policy because none exists. Part of the sync cost the
refactor is meant to remove is therefore procedural and fixable **immediately**, independent of this
proposal — which strengthens the case for writing the policy down in Stage 0, and slightly weakens the
inference from "IFU took 3.5 weeks" to "the fork model costs 3.5 weeks per release."

### 2.3 Key characterization findings

1. `csrc/` is lightly modified, heavily extended; genuine shim-behavior modification is a minority of
   the delta. `jax/csrc` is now measured at 264 lines with zero ROCm-only files.
2. PyTorch Python entanglement has three shapes — sidecar wiring, guarded branches, and unmarked
   semantic divergence concentrated in four `module/` files (transpose-cache toggle, FSDP2,
   split-accumulator config, UB/fused-AG plumbing, an NVFP4 correctness fix). **Bucket line counts are
   withheld pending `added_class` regeneration against the corrected base.**
3. The NVTE C headers are themselves forked (332 lines): the C ABI is not a viable plugin boundary
   today; the framework-extension seam is.
4. **`common/` is not a pure C++ tree** (new in v2.2). It is 223 files, 9 of them Python totalling
   3,887 upstream lines — the package root and the recipe API. This is the single largest structural
   correction in this revision; see §4.1.
5. **The seam has a measurable early-binding surface** (new in v2.2). Upstream has 44 late-bound
   `import transformer_engine_torch as tex` sites and **9** early-binding
   `from transformer_engine_torch import …` sites, one of which is a **star-import**
   (`pytorch/cpp_extensions/__init__.py:6`). See §3.5.
6. Three independent teams hit the same two walls — the divergent `NVTE_Fused_Attn_Backend` enum and
   the comm-gemm-overlap class hierarchy — making them the two structural upstream asks.

---

## 3. Target architecture

### 3.1 Overview

```
                 ┌──────────────────────────────────────────────┐
                 │ Consumers: Megatron-LM(ROCm), Primus,        │
                 │ Lumen/LumenRL, MaxText, customer code        │
                 └──────────────────┬───────────────────────────┘
                                    │ import transformer_engine.{pytorch,jax}
                 ┌──────────────────▼───────────────────────────┐
                 │ VENDORED UPSTREAM TE PYTHON (pinned)         │
                 │  incl. common/ Python: __init__, recipe/,    │
                 │       triton/ (verbatim), utils.py           │
                 │  + BOOT-001 bootstrap patch (build-time)     │
                 │  + build-time source patch queue             │
                 │  + runtime override registry (at plugin load)│
                 └───────┬───────────────────────────┬──────────┘
          PyTorch: tex seam                   JAX (beta track): FFI-name seam
  sys.modules["transformer_engine_torch"]     te_*_ffi registration
                 ┌───────▼──────────┐          ┌──────▼───────────┐
                 │ te-rocm plugin   │          │ te-rocm plugin   │
                 │ facade + registry│          │ handler dict +   │
                 │ + caps + overrides│         │ trace-time sel.  │
                 └───────┬──────────┘          └──────┬───────────┘
                         │  TE_ROCM_EXTENSION_API     │
                 ┌───────▼──────────┐          ┌──────▼───────────┐
                 │ transformer_     │          │ ROCm FFI handler │
                 │ engine_torch_rocm│          │ extension        │
                 └───────┬──────────┘          └──────┬───────────┘
                         │      TE_ROCM_CORE_ABI      │
                 ┌───────▼────────────────────────────▼──────────┐
                 │ libtransformer_engine_rocm.so                 │
                 │ (common/ C++ only; AMD-maintained repo)       │
                 └───────────────────────────────────────────────┘
                         ▲
                         └── ABI-001: vendored Python reaches this
                             directly via ctypes, bypassing the
                             extension API. Must be closed (§3.3).
```

Principles:

- **Pinned upstream Python, vendored, with a named bootstrap exception.** A verbatim tree cannot boot
  on ROCm. BOOT-001 is a build-time bootstrap patch applied at wheel assembly; on ROCm, **plugin
  initialization failure is fatal**. **Rescoped in v2.2**: the target surface is ~190 lines across
  `transformer_engine/__init__.py` (36/4) and `common/__init__.py` (119/31, 21 hunks) — not the "2-3
  files, 3 lines" v2.1.1 asserted as an *exact* measurement. It must cover native loading, PyPI package
  identity, the `te_rocm_build` global, and `is_fp8_fnuz`. Retirement path: the upstream
  python-only/absent-native build-mode ask.
- **All other divergence lives in the two-tier compatibility layer (§3.4)** — never in ad-hoc edits to
  the vendored tree.
- **One seam per framework**; stateful C++ classes re-exported, not dispatched.
- **Compiled backend only at first.**
- **Every mechanism has a no-cooperation fallback** (shim import if #3401 stalls).

### 3.2 Repositories and packaging

Repo layout as v1 (backend/, torch_ext/, jax_ext/, plugin/{facade,jax,capabilities,recipes,kernels,
patches,testing}, tests/, manifest/, ci/) with upstream pinned source-only under
`3rdparty/transformer_engine` (no recursive submodule init). **CI asserts the submodule SHA matches the
manifest's declared base *and* that the base equals `merge-base(upstream/main, fork/dev)`** — the check
that would have caught the v2.1.1 base error.

**Package ownership — decided.** Two distributions installing into one `transformer_engine` package is
unsafe. For the first production version the ROCm distribution **retains the canonical project name
`transformer-engine`, served from the AMD index as the primary/only index**, with a PEP 440 local
version (e.g. `2.18.0+rocm7.0.amd1`; exact qualifier fixed in Stage 1).
`transformer-engine-rocm` as a *different* project name is rejected for certified deployments.

> **Open conflict (new in v2.2).** The fork's `common/__init__.py` currently implements the *rejected*
> scheme: core packages `transformer-engine-rocm7` / `transformer-engine-rocm10`, extras
> `rocm_pytorch` / `rocm_jax`, and a **changed upstream function signature**
> (`get_te_core_package_info(rocm: bool)`). Stage 1 must **rewrite** this logic, not hook it. Tracked
> as manifest `CM-002-PKGID` and open decision `packaging_name_conflict`.

**Wheels** (unchanged in shape): `te-rocm-core` (binary, unique SONAME, `$ORIGIN` loading); torch
extension (prebuilt for the certified matrix, sdist fallback); jax extension; pure-Python (vendored
upstream tree post-BOOT-001 + plugin + both patch tiers — rebuilt every sync, and only this wheel);
meta extras with exact `==` pins.

**Certified environment matrix** embedded in release metadata: ROCm version(s), GPU arch set, PyTorch
version + C++ ABI, Python ABI, JAX/jaxlib, AITER, Triton, RCCL/ROC-SHMEM.

**Activation — automatic and immutable after import**, implemented **inside the ROCm-owned
`transformer_engine` package itself**: `transformer_engine/__init__.py` detects the embedded ROCm
distribution marker, imports `te_rocm.bootstrap`, installs the facade, then imports the remaining TE
Python. That file is manifest entry **CM-001** and carries 40 diverged lines today. The same wheel
ships a top-level `transformer_engine_torch` bootstrap module so the **direct
`import transformer_engine_torch` order is supported**. No `.pth`, no `sitecustomize`. In certified
mode `NVTE_PLUGIN` **cannot** substitute an arbitrary plugin; it functions only in an explicitly
non-certified debug mode. Rank-consistency is validated **after process-group initialization**, never
at import.

### 3.3 Versioned compatibility contracts

Every release embeds and validates:

1. **`TE_UPSTREAM_PY_CONTRACT`** — which upstream Python contract the compatibility layer + extension
   emulate. Conformance: upstream test suite under overlay + symbol/behavior inventory. **Must include
   `_FormatHelper` → `_FormatHelperFP8` field renaming** (see CM-003-FNUZFMT) and facade `__all__`.
2. **`TE_ROCM_EXTENSION_API`** — the facade ↔ compiled-extension contract: symbol kinds, callable
   signatures, enum values, class methods/properties, tensor-state requirements. A typed inventory, not
   a name list. Upstream-required symbols ⊆ facade surface; ROCm extras allowlisted; feature-gated
   groups keyed to the capability that gates them.
3. **`TE_ROCM_CORE_ABI`** — the internal extension ↔ backend contract (private, versioned; unique
   SONAME enforces link identity).

Separately embedded per artifact: the **extension build-compatibility tuple** (Python ABI, PyTorch
version, `_GLIBCXX_USE_CXX11_ABI`, ROCm version, GPU targets). Binary compatibility is asserted by this
tuple, not by the API inventory.

> **A fourth path exists and is currently unmodelled (ABI-001, new in v2.2).**
> `Format.E4M3.max_fwd` — a pure *upstream* Python API — resolves through `is_fp8_fnuz()` → ctypes →
> **`nvte_uses_fp8_fnuz()`**, implemented in `common/amd_detail/system.cpp`. Neither that symbol nor
> `nvte_is_rocm_build` is declared in the public NVTE headers, so both fall outside the 332-line
> header accounting entirely. After the split they live in the backend `.so`, so **vendored-upstream
> Python reaches `TE_ROCM_CORE_ABI` directly, bypassing `TE_ROCM_EXTENSION_API`** — which is supposed
> to be the only route to the backend. Required before Gate B: complete the ctypes symbol inventory,
> then either route these through the extension API or define a fourth, explicitly versioned
> vendored-Python → core contract surface.

Backend provenance discipline: the backend repo is **AMD-maintained, partly derived from upstream TE
under Apache-2.0**; an **upstream-origin ledger** maps copied C++ files to their upstream ancestors.

### 3.4 The two-tier compatibility layer

Runtime replacement is safe only for late-bound leaf functions and module attributes. It is unsafe for
base classes, decorator-time registration, module-level constants, defaults captured at definition,
`from x import y` references already copied elsewhere, autograd functions registered at import, JAX
primitive registration, already-compiled callables, and pickle-visible class paths.

| Tier | Applied | Use for | Verification |
|---|---|---|---|
| **Build-time source patch queue** | at pure-Python wheel assembly | import-sensitive definitions: classes, decorators, module constants, registrations; anything not proven late-bound (the default) | target-level applicability fingerprints; per-patch tests; certified **bundle hash** at production load |
| **Runtime override registry** | at plugin load | late-bound functions and attributes only, each individually proven safe | target identity + signature checks; per-override behavior tests |

**Certification vs applicability.** The released bundle is bound to exactly one certified upstream
SHA; individual patches carry target-level symbol/structural/AST fingerprints valid across candidate
SHAs. During a candidate pin bump the new SHA is explicitly allowed: unchanged targets reapply, changed
targets fail loudly **by patch ID**. Per-patch exact-SHA preconditions are excluded.

Both tiers share the governance model: one tracked item per independently retiring feature, rationale
+ upstream link + expiry condition, paired test, counted in **M2** (§6.3). Per-function source-hash
checks run at build/certification time; production validates the certified bundle hash + version tuple.

### 3.5 PyTorch facade and dispatch contract

Facade: registered as `transformer_engine_torch` (via #3401 hook or shim), delegating to
`transformer_engine_torch_rocm`; **module attributes eagerly populated**, preserving `__all__`,
`dir()`, `__spec__`, class `__module__`, and introspection; enums re-exported from the compiled
extension. Stateful classes are **directly re-exported when the extension contract matches; otherwise a
version-specific wrapper/factory adapts** them.

> **Early-binding surface, measured (new in v2.2).** Upstream has **44** late-bound
> `import transformer_engine_torch as tex` sites — facade-tolerant — and **9** early-binding
> `from transformer_engine_torch import …` sites that copy references at import time:
> `cpp_extensions/__init__.py` (**`import *`**), `cpp_extensions/fused_attn.py`, `ops/_common.py`,
> `ops/basic/layer_norm.py`, `ops/basic/rmsnorm.py`, `ops/fused/userbuffers_{forward,backward}_linear.py`,
> `optimizers/__init__.py`, `tensor/utils.py`.
>
> The facade must be installed in `sys.modules` **before any of these nine import**. The star-import
> copies the entire facade namespace into `te.pytorch.cpp_extensions` at import time, which makes
> **facade `__all__` fidelity load-bearing rather than cosmetic**: any ROCm-extra symbol leaks into
> that namespace. Add `__all__` equality to EXTENSION_API conformance, and make these nine the explicit
> targets of the Stage-1 import-order tests.

Per-op registry — dispatch contract (binding for any non-default implementation):

- **Dispatch key** includes op and direction, dtypes and quantization recipe, rowwise/columnwise and
  scale layout, shape/alignment class, GPU arch, framework/library versions, training-vs-inference,
  determinism mode, graph-capture/compile mode, workspace and tensor-state requirements, distributed
  mode.
- **Selection strictly before launch**; `supports(op, context)` is **pure** — no async kernel, no
  mutation, no visible allocation. **Never** catch-a-kernel-exception-and-retry.
- **Strict failure is the training default**; best-effort fallback is an explicit debug mode.
- **Policy freezes at first compile/graph capture.**
- **Diagnostic snapshot** on request.
- **Initial milestone: `compiled` only.**

### 3.6 Capability contract

The provider distinguishes static hardware capability, version-dependent capability, dynamic operation
eligibility, and policy. Contract shape:
`backend.supports(op, context) -> decision{supported, reason, constraints, implementation_version}` —
rejection **reasons are mandatory**. Upstream vehicle: push #3113 from device *identity* toward
*capability*.

> **First concrete item: FNUZ detection.** `is_fp8_fnuz` is implemented **three times** —
> `common/__init__.py` (ctypes), `pytorch/utils.py:644`, and `jax/util.py:25`, which **shells out to a
> subprocess** (`python -c "import transformer_engine as te; exit(not te.common.is_fp8_fnuz())"`).
> Retiring all three under one provider (`te.fp8.fnuz`) is the smallest end-to-end proof the capability
> design works, and it spans all three layers.

### 3.7 CustomRecipe adapter and sidecar rule

`CustomRecipe(qfactory=...)` is verified duck-typed and explicitly experimental. The plugin defines a
**stable internal quantizer protocol** with a thin **per-upstream-version adapter**, localizing changes
in `QuantizerRole`/request types/duck-typing assumptions and providing the single place to absorb
upstream `isinstance` leaks. Project rule: *CustomRecipe (via the adapter) decides who quantizes; the
registry decides who executes.*

> **Amendment (new in v2.2).** MXFP4 cannot be discharged by CustomRecipe alone. In
> `common/recipe/__init__.py` the fork adds **`MXFP4BlockScaling`, a `Recipe` subclass** — not a
> `CustomRecipe` — and **injects a `Recipe.mxfp4()` classmethod into upstream's base `Recipe` class**.
> `CustomRecipe(qfactory=…)` discharges neither. Both require build-tier patches on upstream class
> definitions (manifest `CM-003-MXFP4CLASS`, `CM-003-RECIPEMXFP4`). The same file also rewrites the
> `Format` enum members for FNUZ FP8 (`CM-003-FNUZFMT`) — definition-time and import-sensitive, hence
> `runtime_eligible: never`. The adapter still owns the quantizer protocol; it does not own recipe
> class identity.

### 3.8 JAX plugin (separate beta track)

Registration-time seam: the plugin supplies a ROCm handler dict for the `te_*_ffi` names; trace-time
selection among FFI handler / Pallas-Triton / native-JAX fallback. `jax/util.py` (37 lines, not the
120 previously estimated) relocates as the capability provider; sharding/partitioning rules registered
by the plugin. Conformance inventories FFI names **plus** attribute schemas, lowering parameters,
layout conventions, and handler ABI. Selection binds at trace; policy freezes per executable. Works
with zero upstream cooperation. `jax/csrc` is now measured: 15 files / 264 lines, zero ROCm-only files.

---

## 4. Disposition of every ROCm modification, by layer

Taxonomy: `backend-repo` | `relocate` | `custom-recipe` | `capability` | `feature-pr` | `compat-pr` |
`upstream-pr` | `hold-nvidia` | `hold-internal` | `patch` (with `patch_timing: build | runtime`) |
`test-overlay` | `build-system` | `delete`.

### 4.1 C++ side — and the `common/` boundary

**`common/` C++ → `backend-repo`** (85 modified files / 6,320 lines + 65 ROCm-only / 20,968). The move
removes *routine textual merge overlap*, not the compatibility obligation, which is carried by the
§3.3 contracts, the origin ledger, and Job-B conformance. Ships as `libtransformer_engine_rocm.so`
(unique SONAME, `$ORIGIN` loading).

> **`common/` Python stays vendored (BK-001B, new in v2.2).** `common/` is **223 files: 208 C/C++/CUDA
> and 9 Python (3,887 upstream lines)**. v2.1.1 moved the directory "wholesale"; it cannot. The Python
> is the package root and the recipe API, imported by both framework layers
> (`from transformer_engine.common.recipe import Recipe`, `…common.triton.permutation import …`).
>
> Only **2 of 9 diverge**, totalling 241 lines: `common/__init__.py` (119/31, → CM-002) and
> `common/recipe/__init__.py` (86/5, → CM-003). **`common/triton/*.py` (6 files, 2,759 lines) and
> `common/utils.py` (56 lines) are verbatim — carry them unpatched.** ROCm-only
> `ck_fused_attn/check_aiter_mha_args.py` (112 ln) relocates with the CK backend (CM-004).

**`pytorch/csrc/` (1,347 ln) → `backend-repo`** as the standalone `transformer_engine_torch_rocm`
extension embedding `TE_ROCM_EXTENSION_API`. **`jax/csrc/` (264 ln) → `backend-repo`** on the beta
track.

**NVTE headers (332 ln)** — bucket A (~180, additive) → `upstream-pr`; primary ask: **opaque backend
identifiers / versioned capability result structures** rather than permanently reserved vendor enum
slots, with enum-range reservation as fallback; `*_cuda_custom` symbols get AMD-private names;
fallback `nvte_rocm.h`. B1 `comm_gemm_overlap.h` (87) → `hold-nvidia` via the #3401 contract
discussion; B2 fused-attn enum (34) → capability RFC; C (~20) → `delete`. *v2.1.1's "new small
`cast.h` delta" was phantom — `cast.h` drops out of the diverged set entirely at the corrected base
(10 diverged headers → 9).*

**ROCm-private symbols not in any header** → `ABI-001`. See §3.3.

### 4.2 PyTorch Python (41 files; 2,142 added / 253 removed)

ROCm-only additions (8,370 ln) → `relocate`: `triton_kernels/` (18 files, 6,566 ln) → `plugin/kernels/`,
registered **after** the compiled-only milestone; MXFP4 stack + fsdp2 allgather (4 files, 1,804 ln) →
plugin namespace via the CustomRecipe adapter, subject to the §3.7 amendment.

Entangled modifications keep the three-shape taxonomy — sidecar wiring, guarded branches, unmarked
semantic — with dispositions unchanged: sidecar → `custom-recipe` via adapter + plugin tex, residue
`patch` (build); guarded → `capability`, interim patches **build-tier by default**, the runtime tier
requiring `runtime_eligible: proven`; unmarked semantic → atomic feature sub-entries (NVFP4 fix →
bugfix PR; transpose-cache, split-acc, fsdp2 → feature PRs; UB/fused-AG → hold-nvidia).

**Per-bucket line counts are withheld in v2.2** pending `added_class` regeneration; the v2.1.1 triples
(503 / 1,117 / 656) were computed against the wrong base and are retained in the manifest only as
`stale_added_class`.

**Retired from this section**: `ep.py` and the `distributed.py` symm-mem characterization (§2.0). EP
Python across both frameworks is **verbatim upstream** in the fork — zero delta — with the FFI define
off on ROCm, and EP tests at all three levels are verbatim. EP therefore remains the canonical
capability-driven collection-time skip demonstration, but as a *pure capability-modelling* exercise
with **no fork divergence to retire**.

### 4.3 JAX Python (18 files; 563 added / 120 removed) — beta track

Target-neutral compile + Gluon (~100) → `upstream-pr`; HSACO-file + prewarm (~120) → `hold-internal`
(ROCm JAX plugin team owns the root causes); `sharding.py` (~79) → `compat-pr`; capability branching →
`capability`; `util.py` (37) → `relocate`; `moe.py` (48) → vendor-neutral `upstream-pr` — **the only
genuine new 2.18 front**; small residue → `patch` (build).

### 4.4 Tests

The gate is **upstream-source conformance under the ROCm policy overlay**. Skip discipline (binding):
skips only at **collection/setup time** from a declared capability with the provider's rejection reason
recorded; **a runtime backend refusal is a test failure**, never a skip; known bugs use
`xfail(strict=True)` with issue link, owner, expiry, last-seen version; **budgets** — skip/xfail counts
and tolerance widths cannot grow silently; a **named, versioned no-overlay subset** must pass with zero
policy modification.

Relocations and the two-job CI as in v1, plus install/upgrade/downgrade/uninstall/mixed-index packaging
tests and import-order tests targeting the nine early-binding sites of §3.5.

---

## 5. Sync workflow after the refactor

```
0. Derive the candidate base from upstream MAIN at a chosen SHA. CI asserts
   base == merge-base(upstream/main, fork/dev). Never a release-branch tip
   and never a tag.                                                   (policy)
1. Bump the 3rdparty pin (certification mode: candidate SHA allowed)  (one commit)
2. Contract check: TE_UPSTREAM_PY_CONTRACT typed inventory diff
   (symbols, signatures, enums, class members, __all__, feature-gated
   optional groups) + FFI schema (jax) + ctypes symbol inventory
   (ABI-001)                                                          (minutes)
3. Apply the build-tier patch queue using target-level applicability
   fingerprints: unchanged targets reapply; changed targets fail loudly
   by patch ID                                                        (minutes)
4. Runtime-override identity/signature checks                         (minutes)
5. Repair only patches whose targets changed; update M1/M2            (the work)
6. Upstream suite under overlay + no-overlay subset + plugin suite
   + per-op conformance                                               (hours)
7. TE_ROCM_EXTENSION_API + build-tuple conformance decides binary reuse:
   an optimization proven per-sync, not assumed
8. Perf spot-check; full certification; PROMOTE the candidate SHA;
   regenerate the manifest with GNU diff; tag the pure-Python wheel
```

Standing jobs: nightly Job B vs upstream `main` — **non-gating early warning** with categorized
outcomes; manifest regeneration per sync, where entries whose measured delta grew signal divergence
regression. New in v2.2: steps 0 and 8 encode the two procedural fixes from §2.0.

**Claim discipline**: "days rather than weeks" is the *target*; it becomes a claim only after the
Stage-2 backtest and two live pin bumps measure it.

---

## 6. Migration ledger

### 6.1 Manifest v2.4 (companion file)

File-complete at PT-001..PT-045 (41 live, 4 retired), JX-001..JX-019 (18 live, 1 retired),
**CM-001..CM-004 (new)**, and 18 structural entries (BOOT-001, BK-001, **BK-001B**, BK-002, BK-003,
**ABI-001**, REL-001..003, HDR-A/B1/B2/C, TST-001..003, CI-001). Schema records `added_lines`/
`removed_lines` separately, `patch_timing`, `fallback_mechanism`, `last_verified_base`, plus new
`v23_values`, `stale_added_class`, `retired_entries`, `layer_totals`, `rocm_only_volume`,
`seam_early_binding_sites`, and `open_decisions`.

### 6.2 Consumer migration and fork sunset

As v1; per-consumer activation simplified by automatic activation; pilots gated on §8; fork freeze date
is a Stage-9 entry criterion stated up front; in-flight branches land-or-retire.

### 6.3 Two burn-down metrics

- **M1 — physical overlap debt**: in-place lines/files in vendored upstream sources. Terminal:
  relocated / PR-merged / moved into the compatibility layer.
- **M2 — carried compatibility debt**: active build patches + runtime overrides + overlay skips/xfails
  + unsupported contract items. Creating a patch *closes M1 and opens M2*; only true retirement reduces
  M2.

Both must be recomputed against the corrected base before they are reported to a gate; the v2.1.1
baselines are not comparable.

---

## 7. Implementation plan (staged; go/no-go gated)

| Stage | Work | Exit gate |
|---|---|---|
| **0. Baseline + control** (~2 wk, overlaps all) | Complete atomic manifest; **regenerate `added_class` against the corrected base**; **resolve the four open decisions of §11**; **write the IFU sourcing policy**; freeze new in-place edits, enforced by branch protection/CI; name an **executive sponsor**; capture baselines. Upstream engagement starts: NVFP4 bugfix PR; #3401 comments; #3113 capability push; ROCm-JAX-plugin issues filed internally | Manifest meets its encoded `stage0_exit_requirements` (now incl. regenerated classification, CM-001..004 characterized, ABI-001 inventory complete, `release_218_gap` decided, CI asserting the merge-base identity); freeze enforced in CI |
| **1. Seam proof inside the current fork** (3-5 day spike, then ~2 wk hardening) | BOOT-001 + the production in-package bootstrap (CM-001/CM-002); facade delegating 100% to the fork's existing compiled tex; vendored upstream Python at the pinned base; typed symbol inventory incl. `__all__`; **import-order tests against the nine early-binding sites**; **resolve `packaging_name_conflict`**; one Megatron smoke | Imports, stateful classes, symbols, compile/capture smoke, and checkpoint load work **without moving any C++**, against a **Stage-1 performance budget approved in Stage 0** |
| **2. Historical sync backtest** (~2-3 wk) | A **counterfactual engineering experiment** (retrospective, hindsight-advantaged). Reconstruct actual historical inputs — **deriving the 2.15 base from that IFU merge commit's second parent, never from `release_v2.15`'s tip**. Build the layer over the risk-weighted entry set (now 11 cases incl. CM-002, CM-003, ABI-001); bump in certification mode | **Falsifiable numeric thresholds approved during Stage 0, before the backtest starts**: every selected incompatibility reapplies or fails loudly by item ID; zero silently dropped behavior; zero in-place edits outside the patch queue; repair effort below a stated fraction of reconstructed historical effort; contract inventory identifies every binary-rebuild requirement. **Gate A** input |
| **3. Compatibility-layer productionization** (~4 wk; funded at Gate A) | Tiered patch queue + override registry with full governance; three contracts embedded **plus the ABI-001 resolution**; production bootstrap + package per §3.2; diagnostics; install/uninstall tests | Certified PyTorch package **using the existing fork backend** → **Gate B** input |
| **4. Backend extraction** (~4-6 wk; authorized at Gate B) | Move `common/` **C++** + `csrc/`; keep `common/` Python vendored (BK-001B); SONAME; private versioned core ABI; prebuilt wheels; origin ledger | Same behavior and performance as Stage 3 |
| **5. Sidecar retirement** (~3-4 wk) | CustomRecipe adapter; relocate MXFP4 + triton_kernels; **CM-003 recipe-class patches**; capability provider (PyTorch), starting with `te.fp8.fnuz` retiring all three `is_fp8_fnuz` copies | Measurable M2 reduction; Stage-5 checkpoint gate (§8.5) |
| **6. Registry expansion** (ongoing) | AITER/Triton/reference implementations one op family at a time | Each family gated individually |
| **7. JAX beta** (parallel from S3) | Handler dict, JIT policy, sharding registration, version matrix, inventory conformance | JAX tests + named MaxText-class workload |
| **8. Consumer migration** (~1 quarter) | Dual-read checkpoints, rollback drill, Primus + Megatron pilots, ops docs; **two live upstream pin bumps certified** | Two consumers + two pin bumps green |
| **9. Fork sunset** | Freeze/archive after coexistence window | No active consumer depends on the fork |

**Requested now: Stages 0-2.**

---

## 8. Validation plan

### 8.1 Conformance (structural)
- Typed symbol inventory per `TE_ROCM_EXTENSION_API` — kind, signature, enum values, class members,
  representative behavior — upstream-required symbols ⊆ facade surface; ROCm extras allowlisted;
  **facade `__all__` equality** (§3.5). JAX: FFI names **plus** attribute schemas, lowering params,
  layout conventions, handler ABI. **Plus the ABI-001 ctypes symbol inventory.**
- Compatibility-layer checks: build-tier target-level applicability fingerprints at build; certified
  bundle hash + version tuple at load; runtime-override identity/signature at load.
- Import-order tests (direct-ext / TE-first / consumer-first / reload / fork), explicitly covering the
  nine early-binding sites; packaging lifecycle tests.
- **Base identity check**: `upstream_sha == merge-base(upstream/main, fork/dev)`.

### 8.2 Numerics
- Per-op paired tests for every non-default implementation against **two oracles**: the compiled
  backend *and* a higher-precision reference.
- Property-based/randomized shape, alignment, layout, and empty/degenerate cases.
- Tensor-state contract tests (transpose-cache validity, scale-inv layouts, mixed-dtype routing,
  MXFP4 round-trips).
- **FNUZ format correctness**: `Format.E4M3.max_fwd` / `max_bwd` resolve correctly on FNUZ and OCP
  hardware — the ABI-001 path end-to-end.
- Harnesses synchronize around asynchronous HIP failures.

### 8.3 Performance
- Per-op microbench parity on the hot set; e2e tokens/sec on ≥2 reference configs within stated
  margins; **memory, workspace size, compile time, import time, and leak checks**; explicit baselines;
  repeated runs with variance bounds.
- Facade dispatch overhead microcheck.

### 8.4 End-to-end training
- Loss-curve equivalence within pre-agreed bands; **statistical envelopes for non-deterministic
  kernels**; BF16 + FP8 recipes; TP/PP/FSDP2 matrix; comm-overlap on the compiled path;
  torch.compile/hipGraph rows. The **EP row is an unsupported-path test**: capability-gated refusal
  with a clear reason and collection-time skips firing from `te.ep.eligible`. JAX certified on its own
  milestone or marked beta.

### 8.5 Checkpoint compatibility
- **Stage 1/2 gate**: existing BF16/FP8 state dicts, amax histories, `extra_state`, facade
  compatibility — fork→plugin load + continue within band.
- **Stage 5 gate**: relocated MXFP4/MXFP8 classes — pickle paths, safe-globals, class-path compat
  shims, CustomRecipe-adapter state, full fork↔plugin rollback. **Includes `MXFP4BlockScaling` class
  identity**, which moves under CM-003-MXFP4CLASS.

### 8.6 Release certification checklist (per tag)
Base identity verified → contracts embedded + validated → conformance green → numerics green (dual
oracle) → checkpoint round-trips green → perf gates → e2e bands → manifest M1/M2 updated + regenerated
with GNU diff → certified matrix + provenance embedded → tag.

---

## 9. Upstream and internal engagement

NVIDIA-facing order: NVFP4 bugfix PR → #3401 comments (contract doc; absent-native mode — also retires
BOOT-001; jax arm) → #3113 capability framing → opaque-identifier/versioned-capability ask (enum-range
fallback) → feature PRs (transpose-cache, split-acc, fsdp2) → `jax/moe.py` lazy-import portability PR →
Triton target-neutral + Gluon PRs → sharding compat PRs → comm-overlap factory (long arc). One named
owner for the relationship.

AMD-internal: **write and enforce the IFU sourcing policy** (§2.2 — the cheapest available win, and
independent of this proposal); decide the `release_218_gap`; ROCm JAX plugin fixes (retires ~120 lines
unilaterally); consumer coordination; fork sunset decision; watch-item: FL/#3401 contract ossifying
around FlagGems assumptions without AMD input.

---

## 10. Risks

| Risk | Mitigation / fallback |
|---|---|
| **Measurement base drift misdirects the program** *(realized once, v2.1.1)* | `upstream_sha` derived from the IFU merge commit's second parent; CI asserts `== merge-base(main, fork/dev)`; diff tool pinned to GNU diff; manifest records `v23_values` so restatements stay auditable |
| **Pinning a pre-release base ships a mislabelled distribution** | `release_218_gap` decided in Stage 0 before any pin is certified |
| #3401 stalls / changes shape | Shim activation (proven); one-page delta to hook |
| NVIDIA PRs rejected | Two-tier patch floor — asserted only after backtest + two pin bumps validate it |
| Upstream moves the seam | No cheap insurance; early warning = contract presence + nightly Job B; the pin controls when it enters our build |
| Semantic contract drift post-split | Three versioned contracts; typed inventories; origin ledger; Job B |
| **Vendored Python reaches the core ABI outside the extension API** | ABI-001: complete the ctypes inventory; route through the extension API or define a fourth versioned surface; resolve before Gate B |
| **`common/` Python mistakenly relocated with the C++** | BK-001 is C++ only; BK-001B carries the Python; CI asserts no `.py` under the backend repo's `common/` import path |
| Import-sensitive patch applied at runtime | Two-tier rule: build-tier default; runtime requires proven late-boundness |
| **Facade `__all__` drift leaks ROCm symbols via the star-import** | `__all__` equality in EXTENSION_API conformance; import-order tests on the nine sites |
| Package co-ownership clobber | Sole-ownership decision + install-lifecycle tests; fingerprint as defense in depth |
| **§3.2 naming decision contradicted by the current bootstrap** | `packaging_name_conflict` resolved in Stage 1 before the packaging design freezes |
| Post-launch kernel fallback corrupting state | Dispatch contract: pre-launch predicates only; strict default |
| Policy epoch vs compiled/captured executables | Freeze policy at first capture/compile |
| Skip discipline masking provider regressions | Runtime refusal = failure; budgets; no-overlay subset |
| CustomRecipe experimental churn | Per-version adapter isolates it; §3.7 amendment covers recipe class identity separately |
| Checkpoint incompatibility late | §8.5 split gates, both before Stage 8 |
| Backtest scoping too narrow | Stated as partial; completed by two live pin bumps before "days" is claimed |

---

## 11. Decisions and open items

**Decided:** source-only upstream submodule + BOOT-001; two-tier compatibility layer with build-tier
default; three versioned contracts (plus an ABI-001 resolution pending); `libtransformer_engine_rocm`
SONAME + origin ledger; sole `transformer_engine` package ownership on AMD indexes; automatic
in-package activation with `NVTE_PLUGIN` functional only in non-certified debug mode; compiled-only
registry first; JAX as separate beta; staged plan with Gate A and Gate B; two burn-down metrics; skip
discipline per §4.4; **`common/` splits C++ (backend repo) from Python (stays vendored)**;
**`upstream_sha` derivation and GNU-diff pinning**.

**Open — four now carry due dates:**

| # | Decision | Due |
|---|---|---|
| 1 | **`release_218_gap`** — take the 15 `release_v2.18` commits before pinning, or pin a later `main` commit? The fork currently ships a tree labelled 2.18 that lacks 2.18's stabilisation. | Stage 0 |
| 2 | **`ifu_sourcing_policy`** — always merge upstream `main` at a chosen SHA; never a release tip. Fixes a live, self-inflicted cost source. | Stage 0 |
| 3 | **`packaging_name_conflict`** — §3.2's canonical `transformer-engine` vs the fork's `transformer-engine-rocm7/-rocm10` + `rocm_pytorch/rocm_jax` extras. | Stage 1 |
| 4 | **`contract_surface_for_ctypes`** — route ABI-001 symbols through `TE_ROCM_EXTENSION_API`, or define a fourth versioned contract? | Gate B |

Also open: workstream owners; fork freeze date; pilot order; exact numeric bands; FL adopt-vs-reimplement;
`qa/`, `docs/`, `examples/`, `benchmarks/` audit; backtest entry selection finalized.

---

## Appendix A — Measurement methodology

**Base selection (normative).** The upstream base is the **second parent of the fork's IFU merge
commit**, and CI asserts it equals `merge-base(upstream/main, fork/dev)`. It is never a release-branch
tip and never a tag. For `dev@8af6efc` that base is `868d8d9216da…`. v2.1.1 used `release_v2.18`'s tip
and thereby attributed 15 upstream commits (1,339 in-scope lines) to ROCm; see §2.0.

**Diff tool (normative).** GNU `diff`; `added` = count of `^>` lines, `removed` = count of `^<`. The
"lines" figure in summary tables is added+removed. git's default Myers algorithm disagrees on some
files — at this base, `cpp_extensions/fused_attn.py` by one line — and must not be substituted, because
M1 burn-down and the divergence-regression alarm are line-count based.

**Measured scope.** `transformer_engine/` — **all** layers, including the package root
(`transformer_engine/__init__.py`) and `common/` Python, both of which v2.1.1 omitted from its entry
list despite declaring them in scope — NVTE headers, `tests/pytorch` (recursive), `tests/cpp`,
`tests/jax`, and `tests/cpp_distributed` (zero divergence; carried verbatim). `build_tools/` and
`setup.py` are measured (366 + 151 lines) but tracked as build-system scope. Files identical between
the trees produce no manifest entries by design. There are **no upstream files absent from the fork**
within measured scope.

**Classification.** Hunk-granular classification of added lines is **base-dependent** and is being
regenerated; v2.1.1's canonical-v2 triples were computed against the wrong base and are retained only
as `stale_added_class`. Added and removed lines are recorded separately.

**Reproduction.** Regenerate per sync; per-entry delta growth is the divergence-regression signal.
Reference-design facts read from checkouts; PR facts from GitHub threads as of 2026-08-28.
