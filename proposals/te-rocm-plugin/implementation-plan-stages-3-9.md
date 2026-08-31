<!--
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Implementation plan — Stages 3–9

**For:** proposal v2.3 / manifest v2.4.3. **Companion to:** `implementation-plan.md` (Stages 0–2),
whose facts F1–F12 and work-package conventions carry over. **Scope:** everything after Gate A.

**How to read the contingency markers.** Stages 0–2 are authorized; everything here is not, yet.
Each work package is tagged with what unlocks it:

| Tag | Meaning |
|---|---|
| `[GATE-A]` | funded by Gate A (Stage 2 backtest + Stage 1 exit) |
| `[GATE-B]` | authorized by Gate B (a certified Stage 3 package on the existing backend) |
| `[EVIDENCE: x]` | shape depends on a number produced earlier — the package is planned, its *form* is not fixed |
| `[DECIDED]` | already settled on 2026-08-29/30; no longer contingent |

Estimates are engineer-days for one engineer unless noted, and are rougher than Stages 0–2 — they
are here to expose sequencing and dependencies, not to be held to.

---

## 1. Ground truth added for these stages

Continues the numbering from `implementation-plan.md`.

| # | Fact | Where | Consequence |
|---|---|---|---|
| F13 | The core library is CMake target `transformer_engine` → `libtransformer_engine.so`, with `INSTALL_RPATH "$ORIGIN/lib;$ORIGIN/transformer_engine/lib"`. **The torch extension is not linked against it** — `build_tools/pytorch.py` links only `nvshmem_host`/`mpi`. Core symbols resolve because `common/__init__.py:_load_core_library` preloads the `.so` with `RTLD_GLOBAL \| RTLD_LAZY` before the extension loads | `common/CMakeLists.txt:436,934`, `build_tools/pytorch.py:122-155`, `common/__init__.py:415` | Renaming to `libtransformer_engine_rocm.so` is `OUTPUT_NAME` + the loader's `so_prefix` — no link changes. But **a SONAME cannot enforce ABI identity here**, because nothing links by SONAME; the extension binds symbol-by-symbol at dlopen. Core-ABI versioning must be a **load-time check** (a version symbol read via ctypes and compared) — the same mechanism as ABI-001, which turns ABI-001 from a wart into the design |
| F14 | The fork already ships 10 Triton kernel modules under `pytorch/triton_kernels/` (cast, cast_transpose, gemm/, gmm/, grouped_gemm, layernorm, rmsnorm, norms_common, common, utils), dispatched from **12 guarded lazy-import sites** in `module/`, `tensor/`, `ops/`, `cpp_extensions/gemm.py` | `grep triton_kernels` | These are Stage 6's first registry families — cast/quantize, norms, gemm, grouped_gemm — and the 12 sites are exactly the sidecar wiring Stage 5 retires |
| F15 | `MXFP4Tensor.__reduce_ex__` and `MXFP8Tensor.__reduce_ex__` point at **module-level functions**, so a pickled tensor's stream carries `transformer_engine.pytorch.tensor.mxfp4_tensor._make_…` as a `GLOBAL`. MXFP8 already carries a compat classmethod for an *earlier* path change | `tensor/mxfp4_tensor.py:451-471`, `tensor/mxfp8_tensor.py:848-1000` | Relocating the MXFP4 stack (REL-002) breaks unpickling of existing checkpoints unless the old module path keeps a shim. There is in-repo precedent for exactly that shim. This is the §8.5 Stage-5 checkpoint gate, made concrete |
| F16 | The JAX seam is a dict: `transformer_engine_jax.registrations()` returns 35 `te_*_ffi` handlers, and `jax/cpp_extensions/base.py:269-270` loops over it calling `ffi.register_ffi_target(name, value, platform="ROCM" if is_hip_extension() else "CUDA")`. `NVTE_JAX_CUSTOM_CALLS` per-op kill switch exists | `jax/csrc/extensions/pybind.cpp:26-40`, `jax/cpp_extensions/base.py:48-57,269-274` | A JAX plugin supplies or overrides dict entries *before* that loop. No synthesized module needed on the JAX side either — the seam is registration-time, as the proposal said |
| F17 | Today's packaging: the Python package is distribution `transformer_engine`; the core wheel is `transformer_engine_rocm{major}` (`setup.py:482-485`), built by `build_tools/wheel_utils/build_wheels.sh`. Names are **kept** (decision 3) | `setup.py`, `build_wheels.sh:68-93` | Stage 3 packaging is a *split*, not a rename: the Python distribution becomes "overlay + plugin + patch queue", the core wheel stays as is; the torch-extension wheel is new |
| F18 | Six manifest items are `runtime_eligible: candidate`: PT-001-MXFP4, PT-010-SEL, PT-011-SEL, PT-012-SEL, PT-026, PT-039 — five are attention backend selection | manifest | Stage 3's runtime-override tier has exactly six candidates to *prove* late-bound; nothing else may enter it |
| F19 | The in-repo Megatron integration test (`qa/L1_pytorch_mcore_integration/test.sh`) clones NVIDIA's `Megatron-LM core_r0.12.0` and probes with `nvidia-smi` — it is the upstream test, not a ROCm consumer test. ROCm's Megatron-LM and Primus live in other repositories | `qa/L1_pytorch_mcore_integration/test.sh` | Stage 8 pilots are **external-repo** work; the only in-repo lever is the `examples/` and `qa/` trees, which need a ROCm-Megatron equivalent of that script |
| F20 | CI already has `rocm-ci.yml` and `rocm-wheels-build.yml`; both run `git submodule update --init --recursive`, which by design skips `3rdparty/transformer_engine_nvidia` | `.github/workflows/` | Stage 3 CI work extends these two files; the identity check and Job B are new jobs, not new workflows |

---

## 2. Critical path, Stages 3–9

```
 GATE A ──► S3 compat-layer productionization ──► certified pkg on EXISTING backend ──► GATE B
                │                                                                         │
                ├──► S7 JAX beta (parallel from S3; own milestone) ──────────────────────┐ │
                │                                                                        │ ▼
                └─(evidence: cxx_arm)──────────────────────────────────► S4 backend separation (in-repo)
                                                                                         │
                                                                                         ▼
                                                              S5 sidecar retirement ──► S6 registry (per family, ongoing)
                                                                                         │
                                                              S8 consumer migration + 2 live pin bumps ◄─┘
                                                                                         │
                                                              S9 retire the merge-based IFU (one repo: nothing to "sunset")
```

Two facts about this path that the proposal's table does not make obvious:

- **S4 is where the C++ question is actually answered**, and it is the largest stage. Stage 1-3
  never touch C++; the cxx_arm number arrives at Gate A; S4.1 spends it.
- **S9 has changed meaning.** With one repository there is no fork to archive. "Sunset" now means:
  the last merge-based IFU has happened; every subsequent upstream intake is a pin bump through the
  patch queues. That is a *procedural* end state, reached when S8's two live pin bumps succeed.

---

## 3. Stage 3 — compatibility-layer productionization `[GATE-A]` · ~4 wk

Goal: the prototype's overlay becomes a **certified package**, still on the fork's existing
compiled backend, so that Gate B judges the Python layer in isolation.

### S3.1 · Patch-queue governance — **DONE 2026-08-31**: `patch_fingerprints.py` (symbol-level AST fingerprints; self-check 45/45; dry run vs upstream main: 17 unchanged / 2 moved / 25 changed by symbol / 1 gone), `check_manifest.py` (M1/M2 + invariants), `# tests:` headers enforced

The P4 assembler used `git apply --check` as its applicability test. Productionize:

- **Target-level fingerprints** (§3.4): per patch, record the target symbol(s) (function/class name),
  a normalized-AST hash of the target's *pre-patch* source, and ±3 lines of context. A pin bump
  reports, per patch ID: `target-unchanged` (reapply), `target-moved` (reapply with offset),
  `target-changed` (trip, show AST diff). Exact-SHA preconditions remain excluded.
- **Per-patch tests**: every patch file's header `# tests:` must name ≥1 test; the assembler fails on
  an empty field. Retired-unchanged entries carry the test that proved the upstream file works.
- **M2 counter**: `tools/check_manifest.py` emits M1/M2 per §6.3 — M2 = active patches + runtime
  overrides + overlay skips/xfails + unsupported contract items — and CI fails if M2 rises without a
  manifest entry explaining it.
- **Bundle hash**: `overlay-manifest.json` gets signed into the wheel; `common/__init__.py` (CM-002
  patch) validates it at load and refuses a mismatched overlay.

| Exit | A pin bump to a *synthetic* target (the base + one hand-made upstream change to a patched function) reports the trip by ID with the AST diff; M2 is computed, not hand-counted. |

### S3.2 · Runtime-override tier — **DONE 2026-08-31**: census (`override_census.py`) disqualified all four needed candidates with file:line evidence; PT-026 census-passed but swap test blocked on CP tests; **runtime tier ships EMPTY, by measurement**

Six candidates. Each must be **proven** late-bound before it may leave the build tier:

1. **Reference census**: AST walk of the whole overlay for every name that imports or aliases the
   target (`from x import f`, `g = x.f`, decorator use, default-argument capture, autograd
   `Function` registration). Zero early-bound references → eligible.
2. **Swap test**: install the override *after* `import transformer_engine.pytorch`, run the paired
   test, confirm the override is what executed (a sentinel in the override's return path).
3. Record `runtime_evidence: {census: …, swap_test: …}` in the manifest; flip
   `runtime_eligible: proven`.

Expected outcome: the five attention-selection items pass (they are leaf functions looked up
per call); PT-001-MXFP4 likely fails the census (`cpp_extensions/gemm.py` is early-bound via the
star-import) and stays build-tier. **Either result is fine — the point is the procedure.** The
registry itself: `te_rocm/overrides.py`, applied at the end of `load_framework_extension`, identity-
and signature-checked at load.

### S3.3 · The contracts, embedded — **CORE_ABI DONE 2026-08-31**: `nvte_rocm.h` + `nvte_rocm_core_abi_version()` + load-time refusal (both paths verified); ABI-001 closed. PY_CONTRACT = bundle hash (done); EXTENSION_API inventory = P6 tests (names/values/signatures) — richer signature typing remains open

| Contract | Source of the inventory | Check |
|---|---|---|
| `TE_UPSTREAM_PY_CONTRACT` | submodule pin + `overlay-manifest.json` + P6's typed inventory of upstream symbols the patches touch | patch fingerprints (S3.1) |
| `TE_ROCM_EXTENSION_API` | `seam_inventory.py --values --signatures`: names, enum members, function arity from pybind docstrings, class members | `tests/te_rocm/test_seam_*` (P6), now blocking |
| `TE_ROCM_CORE_ABI` | **new**: `nvte_rocm_core_abi_version()` exported by the core lib and read via ctypes at `_load_core_library` — alongside `nvte_is_rocm_build` / `nvte_uses_fp8_fnuz`, all three declared in one small public header `nvte_rocm.h` | load-time compare against the value the extension was built with (F13: this is the *only* way to enforce core-ABI identity, since nothing links by SONAME) |

This closes **ABI-001** as a work item: the two ctypes symbols become the first members of a
deliberately tiny, versioned introspection API — plus `seam_inventory.py` gains a
`_TE_LIB_CTYPES.<name>` scan so ctypes demand is never invisible again.

### S3.4 · Packaging — **DONE 2026-08-31** `[DECIDED: names kept]`: `NVTE_BUILD_OVERLAY=1` emits the pure-Python `transformer_engine` wheel (py3-none-any, provenance manifest in-package); core wheel python-tree-stripped in `build_wheels.sh` (shadowing hazard closed); torch extension stays sdist-shipped (wheel compiled from sdist verified); lifecycle T1-T5b all PASS (`tests/te_rocm/lifecycle.sh`, baselines/2026-08-31-lifecycle.json). JAX wheel arm deferred to S7 as planned

Today: one Python distribution `transformer_engine` (source tree) + core wheel
`transformer_engine_rocm{major}`. After:

| Distribution | Contents | Rebuilt |
|---|---|---|
| `transformer_engine` (pure Python, `py3-none-any`) | overlay (vendored upstream + patches) + `te_rocm/` plugin + `overlay-manifest.json` | every pin bump — and *only* this |
| `transformer_engine_rocm{major}` (existing) | `libtransformer_engine_rocm.so` (renamed in S4; in S3 still `libtransformer_engine.so`) | backend changes |
| `transformer_engine_rocm_pytorch` (**new**) | `transformer_engine_rocm_torch` extension + build-compat tuple | torch/ROCm/ABI tuple changes |
| `transformer_engine_rocm_jax` (existing name) | JAX extension | S7 |

Work: `setup.py` learns to emit the pure-Python wheel from `build/overlay/` (BuildPy already
generates `_rocm_init.py` into the tree — same mechanism, larger payload); the extension gets its
own `setup()`; `build_wheels.sh` gains the split. **Lifecycle tests** (§4.4): install / upgrade /
downgrade / uninstall / mixed-index, run in a clean venv in CI; the mixed-index case asserts the
loud-failure property (P3 test b).

### S3.5 · Diagnostics snapshot — **DONE 2026-08-31**: `python -m transformer_engine.te_rocm_diagnostics` (tree provenance, seam identity, core ABI, bundle hash/patch ids)

`te_rocm.diagnostics.snapshot()` → dict: pin SHA, overlay bundle hash, applied patch IDs, active
runtime overrides, contract versions (all three), build-compat tuple, `IS_HIP_EXTENSION`, ROCm/HIP
versions, GPU arch, and — once S6 exists — selected implementation per op with rejection reasons.
Printed by `python -m transformer_engine.te_rocm`. This is what a bug report attaches.

### S3.6 · CI wiring — **DONE 2026-08-31**: `.github/workflows/te-plugin-governance.yml` — blocking: branch-aware three-way pin identity, divergence reproduction, seam-name inventory (LayerNorm allowlisted), check_manifest, fingerprint self-check, patch staleness (`gen --verify`: apply-at-pin, byte-compare), overlay assemble+py_compile; nightly Job B (non-gating) = fingerprint drift vs upstream main. Lifecycle tests (`tests/te_rocm/lifecycle.sh`) run in the wheel pipeline instead — building any wheel, even the pure one, needs the ROCm toolchain (`rocm_build()`)

Add to `rocm-ci.yml`: (a) the **branch-aware** three-way identity job — on `dev`, submodule HEAD ==
manifest `upstream_sha` == `merge-base(upstream/main, HEAD)`; on `release_v*_rocm`, against
`upstream/release_v*`; (b) `tools/measure_divergence.sh`, `seam_inventory.py`, `classify_hunks.py`,
`check_manifest.py` as blocking checks; (c) the G3 freeze check (in-place edits to the overlay's
targets rejected outside `patches/`); (d) **Job B**: nightly, non-gating, assembles the overlay
against upstream `main` HEAD and reports outcomes in four categories — patch drift / API drift /
dependency-environment / upstream test failure. Add to `rocm-wheels-build.yml`: the four-wheel
split and the lifecycle tests.

### S3.7 · Stage-3 certification → Gate B packet — **PACKET COMPLETE 2026-08-31** (`certification-stage3.md`; all §8.6 rows PASS; e2e graded on aggregate after a noise series — see methodology note). Gate B decision itself rests with the certifier

Full §8.6 checklist on the **existing** backend: contracts embedded and validated → upstream suite
under overlay + the named no-overlay subset → numerics (dual oracle on the compiled path only,
since no alternative implementations exist yet) → checkpoint round-trips → perf within the signed
budget → M1/M2 → matrix + provenance → tag.

| Gate B question | *Is the Python compatibility layer production-grade, independent of any backend change?* If yes, the C++ work in S4 proceeds against a stable Python contract. |

---

## 4. Stage 4 — backend separation, in this repository `[GATE-B]` · ~6-8 wk (was 4-6; C++ queue added)

Goal: the compiled layers become independently built, versioned targets, and — the part the
proposal never planned — the C++ gets a maintenance mechanism.

### S4.1 · Decide `cxx_maintenance_strategy` — 1 day `[EVIDENCE: cxx_arm]`

Read `thresholds.yaml`'s bands against the B3 trip rate. Assign `cxx_strategy` **per file** in the
manifest (BK-001 gets per-file children for the 85). Expected shape given the guard census (78 of 85
guarded, ~480 sites): most files `patch-queue`; a handful of guard-dense files where AMD already
owns the kernel path (attention dispatch, fused-attn glue) → `native-hip`. `freeze` is not an
option for any file.

### S4.2 · C++ patch queue over the submodule — 10 days `[EVIDENCE: S4.1]`

Extend the P4 assembler with a C++ mode: for each `cxx_strategy: patch-queue` file, source =
`3rdparty/transformer_engine_nvidia/transformer_engine/common/<path>`, patch = the fork's guard
edits keyed `CXX-<n>`, output into `build/overlay/common/`; **then** `hipify_torch` runs exactly as
today on the assembled tree. Fingerprints are function-level (a C++ AST is out of reach without
libclang; use `ctags`/`universal-ctags` symbol ranges + context — good enough, and honest about it).
The B2 C++ arm already produced the first ~10 patches; this generalizes to the set S4.1 chose.

For `native-hip` files: move to `common/amd_detail/` (or a sibling native dir), add to hipify's
exclusion list, delete the CUDA-shaped original. Each conversion is one reviewable commit with the
kernel's paired test.

| Exit | `assemble_overlay.py --cxx` builds a `common/` tree that compiles and passes `ci/core.sh` bit-for-bit equivalent to the pre-S4 tree (compare test outputs, not binaries). |

### S4.3 · Library identity — 2 days `[DECIDED: one repo]`

`OUTPUT_NAME libtransformer_engine_rocm` in `common/CMakeLists.txt`; `so_prefix` in
`common/__init__.py` follows; keep `INSTALL_RPATH "$ORIGIN/…"`. Versioned `SOVERSION` for hygiene —
but per F13 the *enforced* identity is S3.3's load-time `nvte_rocm_core_abi_version()` check, not
the SONAME. Bump the ABI version on any change to a symbol the extension or the ctypes callers use.

### S4.4 · Origin ledger — 2 days

Generated, not hand-written: for every file under `common/` that has an ancestor in the submodule,
`origin-ledger.json` records `{fork_path, upstream_path, upstream_blob_at_pin, cxx_strategy,
patch_ids}`. Files under the ROCm-only trees record `upstream: null`. Regenerated per pin bump; a
diff of the ledger *is* the C++ upstream-intake report.

### S4.5 · `pybind_helper.h` (ABI-002) — 1 day

Decision from the S0 exit requirement, now implemented: the header is compiled into the **torch
extension target**, and its enum member lists are generated from the backend's public headers so
`EXTENSION_API`'s enum-value inventory and the binding cannot drift apart. `NVTE_Fused_Attn_Backend`
keeps both member sets until HDR-B2 lands upstream.

### S4.6 · Prebuilt extension wheels for the certified matrix — 3 days

`transformer_engine_rocm_pytorch` built per (Python ABI × torch version × ROCm major × GPU-arch
set) from the certified matrix (§3.2); sdist fallback. `rocm-wheels-build.yml` matrix expansion.
Each wheel embeds its build-compat tuple; the loader compares it at import.

### S4.7 · Stage-4 exit

Same behaviour and performance as Stage 3 (re-run the S3.7 checklist unchanged) **plus**: the C++
patch queue reapplies on one live pin bump with the trip rate reported, and the ROCm-only trees are
byte-identical to before S4.

---

## 5. Stage 5 — sidecar retirement · ~4 wk

Goal: the 622 hunk-lines of sidecar wiring (manifest `added_class.sidecar`, PyTorch) and the
Recipe-class injections go away, and the guard bucket starts converting to capability queries.

### S5.1 · CustomRecipe adapter — 5 days `[F8]`

- `te_rocm/recipes/adapter_2_18.py`: the per-upstream-version adapter implementing upstream's
  `qfactory(role) -> Quantizer | QuantizerRequest` protocol on top of a stable internal
  `te_rocm.recipes.QuantizerProtocol`. One file per certified upstream version.
- `MXFP4BlockScaling` becomes a `CustomRecipe` subclass whose `qfactory` returns the MXFP4
  quantizer → dispatch flows through upstream's own `recipe.custom()` → `CustomRecipeState` path.
- **Retire CM-003-RECIPEMXFP4** (the `Recipe.mxfp4()` injection) and the `recipe.mxfp4()` branch in
  `quantization.py:1447`. Every upstream `isinstance(recipe, …)` leak that blocks this is filed as a
  one-line duck-typing PR upstream (§3.7) and carried as a patch until merged.
- Keep `CM-003-FNUZFMT` (the `Format` enum rewrite) as a build-tier patch — it is not a recipe
  concern; it retires with the capability provider (S5.3).

### S5.2 · Relocation with pickle shims — 4 days `[F15]`

Move `tensor/mxfp4_tensor.py`, `tensor/storage/mxfp4_tensor_storage.py`,
`custom_recipes/quantization_mxfp4.py`, `tensor/fsdp2_allgather_tensor.py` and the 10
`triton_kernels/` modules under `transformer_engine/te_rocm/`. For each moved module that
contributes a pickle `GLOBAL` (F15: `_make_mxfp4_tensor_in_reduce_ex`, the storage classes), leave
a **shim module at the old path** that re-exports the names — the exact precedent MXFP8 already
carries for its earlier rename. Register the new class paths with `torch.serialization.add_safe_globals`.

| Stage-5 checkpoint gate (§8.5) | a checkpoint written *before* S5 (BF16, FP8-delayed, MXFP4 weights) loads *after* S5 and trains 50 steps within band; a checkpoint written after S5 loads on a pre-S5 build (rollback) — this direction needs the shim to exist on both sides, which is why the shim ships one release *before* the move. |

### S5.3 · Capability provider (PyTorch) — 6 days

`te_rocm/capabilities.py` implementing §3.6's `supports(op, context) -> decision{supported,
reason, …}` and the static/version/dynamic split. Order of retirement, chosen by the manifest's
guard census:

1. **`te.fp8.fnuz`** — replaces the three `is_fp8_fnuz` implementations (`common/__init__.py`
   ctypes, `pytorch/utils.py:644`, `jax/util.py:25` subprocess) with one provider call backed by
   S3.3's introspection API. Also retires `CM-003-FNUZFMT`'s runtime dependency.
2. **Attention backend eligibility** — the PT-010/011/012/026/039 selection logic (now runtime
   overrides, S3.2) becomes provider queries; this is the ROCm-side half of HDR-B2 and stays until
   upstream's capability RFC lands.
3. **`IS_HIP_EXTENSION` sites file by file**, largest guard bucket first (`base.py` 76 hunk-lines,
   `mxfp8_tensor.py`, `cpp_extensions/gemm.py`), each converted guard replaced by a named
   capability with a reason string. Progress metric: `classify_hunks.py`'s **guard** bucket for the
   file goes down; **unmarked** must not go up.

### S5.4 · Stage-5 exit

M2 measurably lower (sidecar bucket → near zero; guard bucket down by the converted files); the
checkpoint gate above green in both directions; `te.fp8.fnuz` has exactly one implementation.

---

## 6. Stage 6 — registry expansion, one op family at a time · ongoing

Goal: the compiled-only milestone ends; AITER / Triton / reference implementations enter under the
§3.5 dispatch contract. **This is the stage where the synthesized facade becomes necessary** — the
loader alias re-exports the compiled module verbatim, and the registry needs a module whose
attributes can differ from it. Build it here, from the P2 description the Stages 0-2 plan retired.

Family order, by number of existing call sites and by what the fork already ships (F14):

| Family | Existing ROCm implementation | Call sites today | Tensor-state tests |
|---|---|---|---|
| 1. cast / quantize | `triton_kernels/cast.py` (`te_quantize_triton`), `cast_transpose.py` | 6 | transpose-cache validity, scale-inv layout, FNUZ vs OCP |
| 2. norms | `layernorm.py`, `rmsnorm.py`, `norms_common.py` | 4 | saved-tensor logic (`test_layernorm_saved_tensors_logic.py`) |
| 3. gemm | `triton_kernels/gemm/` | 1 (`cpp_extensions/gemm.py:506`) | rowwise/columnwise, mixed dtype |
| 4. grouped gemm | `grouped_gemm.py`, `gmm/` | 1 | per-expert pointer packing |
| 5+ | AITER attention (`ck_fused_attn` already behind the compiled path), others | — | — |

Per family, unchanged from the proposal and binding: pure `supports()` predicate (no launches, no
allocation); selection strictly before launch; strict failure as training default; policy frozen at
first compile/capture (including hipGraph); dual-oracle numerics (compiled + high-precision
reference); diagnostics snapshot shows the selection and every rejection reason. A family ships
only when its paired conformance test and tensor-state tests are green.

---

## 7. Stage 7 — JAX beta track · parallel from S3, own milestone

### S7.1 · Handler-dict seam — 2 days `[F16]`

`te_rocm/jax_handlers.py` returns a dict; `cpp_extensions/base.py`'s registration loop (CM-005, a
new build-tier patch of ~4 lines) merges it over `transformer_engine_jax.registrations()` before
`ffi.register_ffi_target`. Selection binds at trace; policy freezes per executable (a jitted function
never re-selects). No synthesized module.

### S7.2 · FFI inventory — 3 days

Extend `seam_inventory.py` with a JAX mode: the 35 `te_*_ffi` names **plus** attribute schemas
(the `pybind11::arg` kwargs per handler), lowering parameters, and layout conventions read from the
Python primitives. Name-only inventories are not enough here for the same reason enum-value
inventories were needed on the torch side.

### S7.3 · JAX overlay + capability provider — 5 days

The P4 assembler is framework-neutral; JX-001..019 become patches (18 files, 563/120). `jax/util.py`
(37 lines, REL-003) → `te_rocm.capabilities` (shared with S5.3; the subprocess `is_fp8_fnuz` dies
here). JX-001's HSACO/prewarm workarounds (`hold-internal`) stay as patches until the ROCm JAX
plugin fixes land; `sharding.py` (JX-002) goes upstream as a compat PR; `moe.py` (JX-005) likewise.

### S7.4 · JAX milestone

`tests/jax` under overlay + a named MaxText-class workload within a JAX-specific signed band
(added to `thresholds.yaml` when the workload is chosen). Until then the JAX wheel is marked beta.

---

## 8. Stage 8 — consumer migration + two live pin bumps · ~1 quarter, calendar-gated

### S8.1 · Pilots — external repos `[F19]`

ROCm Megatron-LM and Primus consume the packaged wheels. In-repo work: a ROCm equivalent of
`qa/L1_pytorch_mcore_integration/test.sh` pointing at ROCm/Megatron-LM, run in `rocm-ci.yml` as the
e2e row of §8.4. Everything else is coordination in the consumers' repos.

### S8.2 · Dual-read checkpoints and rollback drill — 3 days

The S5.2 gate exercised in production shape: a pilot trains N steps on the packaged build, rolls
back to the previous release, continues, rolls forward. Documented as a runbook.

### S8.3 · Two live upstream pin bumps — the real test

Two consecutive upstream intakes executed **purely** as §5's workflow: bump the submodule to a new
`main` SHA (per the two-track policy), assemble, repair tripped patches by ID (Python and C++
queues), certify, ship. Record wall-clock and engineer-days per bump. **This is where "days rather
than weeks" becomes a claim or is retracted** — the proposal's claim discipline (§5) makes this
the only evidence that counts. The backtest's numbers were retrospective; these are live.

### S8.4 · Ops documentation

Runbooks: pin bump, patch repair, rollback, diagnostics snapshot triage, C++ intake via the origin
ledger. One page each.

---

## 9. Stage 9 — retire the merge-based IFU

Reinterpreted for one repository. There is no fork to freeze; there is a *procedure* to retire.

| Entry criterion | S8.3's two pin bumps succeeded within the signed budget |
|---|---|
| Work | Delete the merge-based IFU procedure from `CONTRIBUTING.rst`; the two-track policy's "IFU" *is* the pin bump. Make the G3 freeze permanent on `dev`. Remove the fork-era in-place-divergence tooling that the patch queues replaced. |
| Exit | An upstream intake performed with zero whole-repo merge activity — the submodule moves, the queues reapply, CI certifies. |

---

## 10. Gates, in order

| Gate | Evidence | Unlocks |
|---|---|---|
| **A** | Stage 1 exit (EXIT-B + P8 within budget) + B3 report incl. C++ arm (informational) | Stage 3 — **PASSED 2026-08-31** (decision: Wen; packet: EXIT-A/B, P6, P8 PASS, backtest B3) |
| **B** | Certified Stage 3 package on the existing backend (S3.7) | Stage 4 |
| S4 exit | S3.7 checklist unchanged + C++ queue reapplies on one live bump | Stage 5 |
| S5 exit | M2 down, checkpoint gate green both directions | Stage 6 (families), Stage 8 pilots |
| JAX milestone | `tests/jax` + MaxText-class band | JAX wheel leaves beta |
| S8 exit | two consumers + two pin bumps within budget | Stage 9 |

---

## 11. Rough calendar

Sequential critical path only; JAX runs alongside from S3.

| Stage | Duration | Cumulative from Gate A |
|---|---|---|
| 3 | ~4 wk | 4 wk → Gate B |
| 4 | ~6-8 wk | 10-12 wk |
| 5 | ~4 wk | 14-16 wk |
| 6 | ongoing from here | — |
| 8 | ~1 quarter, calendar-gated on upstream releases | ~26-28 wk |
| 9 | after S8 | — |

The dominant uncertainty is S4's C++ queue, and it is exactly the thing the cxx_arm exists to
de-risk before the stage is funded.

---

## 12. Risks specific to Stages 3–9

| Risk | Where | Mitigation built in |
|---|---|---|
| C++ patch-queue trip rate is high and S4 balloons | S4.2 | cxx_arm decides *before* funding; `native-hip` per file is the pressure valve; S4 is scoped at 6-8 wk *with* the queue |
| Core-ABI drift undetected because nothing links by SONAME | S3.3 / S4.3 | load-time `nvte_rocm_core_abi_version()` check (F13) — the SONAME is hygiene, the version symbol is the enforcement |
| A runtime override that "proved" late-bound regresses when upstream adds an early-bound import | S3.2 | the census is re-run on every pin bump; a new early-bound reference flips the item back to build tier and fails CI |
| Relocated MXFP4 breaks old checkpoints | S5.2 | pickle-path shims shipped one release ahead; gate tested in both directions |
| `isinstance(recipe, MXFP4BlockScaling)` leaks in upstream block the CustomRecipe route | S5.1 | each is a one-line duck-typing PR; carried as a patch meanwhile, counted in M2 |
| Registry `supports()` predicate does a launch "just to check" | S6 | purity is a test: `supports()` runs under a HIP-launch interposer that fails on any kernel |
| JAX handler dict diverges from the primitives' expected attribute schema | S7.2 | schema-level inventory, not name-level |
| "Days not weeks" claimed on the backtest alone | S8.3 | claim discipline: only two live bumps may make it |
| One repo means the freeze must never lapse | S9 | G3 check permanent on `dev`; the retired IFU procedure is deleted, not deprecated |
