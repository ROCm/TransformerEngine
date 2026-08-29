<!--
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Implementation plan — Stages 0–2

**For:** proposal v2.2 / manifest v2.4.1. **Scope:** the work authorized by the Stage 0-2 approval:
a working prototype of the seam (Stage 1), the governance that makes its results trustworthy
(Stage 0), and the backtest that produces Gate A evidence (Stage 2). Nothing here moves C++, migrates
a consumer, or touches the fork's release line.

**Organizing principle:** the prototype is on the critical path; governance runs beside it; the
backtest consumes the prototype. Every work package has a file-level scope, an entry condition, an
exit test, and an environment requirement. Estimates are engineer-days for one engineer unless noted.

---

## 1. Ground truth the plan rests on

Facts read from the code at `dev@8af6efc` vs upstream `868d8d92`, not from the proposal's prose.
If any of these stop being true, the affected work packages change.

| # | Fact | Where | Consequence |
|---|---|---|---|
| F1 | The fork builds its extension under upstream's exact name `transformer_engine_torch` | `build_tools/pytorch.py:150` | Renaming it is the single change that makes the facade necessary and testable |
| F2 | A build-time bootstrap slot already exists: `setup.py:BuildPy` generates `transformer_engine/_rocm_init.py` from `build_tools/templates/_rocm_init.py`, imported at `transformer_engine/__init__.py:19` **before** `import transformer_engine.common` (line 26) | `setup.py:46-75` | BOOT-001 extends this mechanism; no new hook needed. Facade install goes in the same pre-`common` region |
| F3 | Upstream's package root touches only `transformer_engine.common` (line 14) before framework loading; `common/` Python never imports the seam (2 docstring mentions only) | `transformer_engine/__init__.py`, `common/recipe/__init__.py:654` | Installing the facade above line 14 is provably before every seam import; no ordering race exists |
| F4 | 44 late-bound `import transformer_engine_torch as tex` sites; 9 early-bound `from … import` sites incl. a star-import at `pytorch/cpp_extensions/__init__.py:6` | seam inventory | Facade must be a real module object with eager attributes and a faithful `__all__` |
| F5 | Seam is **closed on names**: 161 demanded, 176 ROCm-reachable; the one MISSING (`tex.LayerNorm`) is an upstream dead-code bug | `seam-inventory-868d8d92.txt` | No facade shim needs to synthesize a function. Conformance work is on signatures/values, not names |
| F6 | Seam is **open on enum values**: `NVTE_Fused_Attn_Backend` binds `{AOTriton, CK, No_Backend}` on ROCm vs `{F16_max512_seqlen, F16_arbitrary_seqlen, FP8, No_Backend}` upstream; upstream Python early-binds it and compares against CUDA members | ABI-002-FAENUM, `common/util/pybind_helper.h:20-33` | The attention patches PT-010/011/012 are **mandatory** in the prototype; `EXTENSION_API` conformance must diff enum values |
| F7 | 17 seam names are registered from `common/util/pybind_helper.h` (backend side of BK-001), 11 upstream-demanded | ABI-002 | Extension build must keep including that header; not a Stage 1 problem, a Stage 4 one |
| F8 | Recipe dispatch is classmethod-based (`recipe.delayed()/.mxfp8()/.nvfp4()/.custom()`), with `recipe.custom()` → `CustomRecipeState` | upstream `pytorch/quantization.py:1317-1327` at `868d8d92`; fork equivalent at `:1439-1451` with `recipe.mxfp4()` inserted at `:1447` | MXFP4 has a legitimate hook (`custom()`); the fork's injected `Recipe.mxfp4()` is a shortcut that must become a build-tier patch until CustomRecipe absorbs it (Stage 5) |
| F9 | Vendored-Python divergence is 41 PyTorch + 18 JAX + 3 root/common files, 2,142+253 / 563+120 / 241+40 lines | manifest v2.4.1 | That is the patch queue's upper bound. PyTorch-only prototype: **44 files** |
| F10 | This workstation has an MI350X and torch 2.12.0+rocm7.14.0 but **no TE build**; the repo rule is build only in the designated container | `CLAUDE.md` | Pure-Python work packages run here; anything that builds or runs kernels needs the container named first |

---

## 2. Critical path

```
  G0 decisions ─┐
                │      ┌─── P1 rename ext ──► P2 facade+bootstrap ──► P3 EXIT-A ──┐
  G1 classify ──┼──────┤      (fork's own Python through the facade; no vendoring)  │
                │      │                                                           ▼
  G2 CI base  ──┤      └─── P4 overlay tooling ──► P5 vendor 868d8d92 + patch queue ┤
                │                                                                  ▼
  G3 freeze ────┘                                       P6 conformance ──► P7 EXIT-B ──► P8 perf/ckpt gate
                                                                                              │
  G6 thresholds (BEFORE B2 starts) ──────────────────────────────────────────────────────────┼──► B1 reconstruct
                                                                                              │       │
                                                                                              └──────►B2 replay ──► B3 report ──► GATE A
```

Two exits inside Stage 1, deliberately:

- **EXIT-A** — the facade holds *with the fork's own Python, unchanged*. Isolates the mechanism
  (F1-F4) from vendoring. If this fails, the architecture is falsified for the cost of ~3 days.
- **EXIT-B** — vendored upstream Python + patch queue passes the representative suite through the
  facade. This is the proposal's Stage 1 gate.

---

## 3. Track P — the prototype (Stage 1)

### P0 · Environment and baseline — 1 day · **container**

| | |
|---|---|
| Do | Record container image/tag, ROCm, GPU arch, TE commit, submodule state (repo rule). Build the fork as-is. Run the representative suite (§3.9) and capture pass/fail + wall time as the baseline. Capture import time (`python -X importtime -c "import transformer_engine.pytorch"`). |
| Exit | Baseline numbers committed to `proposals/te-rocm-plugin/baselines/<date>-fork.json`. |
| Blocks | Everything in Track P that runs code. |

### P1 · Rename the extension — 0.5 day · **container**

| | |
|---|---|
| Do | `build_tools/pytorch.py:150`: `name="transformer_engine_torch"` → `"transformer_engine_torch_rocm"`. Rebuild. Confirm `import transformer_engine.pytorch` now **fails** with `ModuleNotFoundError: transformer_engine_torch`. |
| Why | Makes the old name genuinely absent, so a facade bug fails loudly (the clean-install property from F3) instead of being masked by the compiled module answering under the old name. |
| Exit | The failure is observed and its traceback names one of the 9 early-bound sites or the root. |

### P2 · Facade + bootstrap — 2 days · **this workstation** (pure Python), verified in container

Files created (all new, all AMD-owned, none inside the vendored tree):

```
transformer_engine/te_rocm/__init__.py
transformer_engine/te_rocm/bootstrap.py     install(): build facade, sys.modules insert, idempotent
transformer_engine/te_rocm/facade.py        make_facade(ext) -> ModuleType
transformer_engine_torch/__init__.py        top-level bootstrap module for direct-import order (Sec 3.2)
```

`facade.py` contract (F4):

- Real `types.ModuleType("transformer_engine_torch")`; `__spec__`, `__file__`, `__package__` set so
  `importlib`, `inspect`, and pickle behave.
- **Eager** attribute copy from `transformer_engine_torch_rocm` — every public name, plus every name
  in the extension's `__all__` if present. No `__getattr__` laziness (the star-import copies whatever
  exists at import time).
- `__all__` = extension's public names **minus** an explicit denylist of ROCm extras (the 25 in the
  inventory) so they do not leak into `te.pytorch.cpp_extensions`. Allowlist, not blocklist, is the
  Stage 3 form; a denylist is enough to prove the mechanism.
- Enums and classes re-exported **by identity** (same object), never wrapped — F6 is handled by
  patches, not by the facade synthesizing members.
- Immutable after install: a second `install()` is a no-op; installing after any
  `transformer_engine.pytorch` import raises (detects P1's failure mode rather than hiding it).

`bootstrap.py` is wired at **`transformer_engine/__init__.py`, immediately after the `_rocm_init`
block and before `import transformer_engine.common`** (F2, F3). Same try/except-ImportError shape
as `_rocm_init`, but on ROCm a failure is **fatal** (proposal §3.1) — the except branch re-raises with
a message naming the missing extension.

`transformer_engine_torch/__init__.py` (top-level): calls `bootstrap.install()` then
`from transformer_engine_torch import *` from the facade it just installed. Makes
`import transformer_engine_torch` as the *first* import a supported order, not an error.

| Exit | `python -c "import transformer_engine_torch as tex; import transformer_engine.pytorch as te; assert tex is sys.modules['transformer_engine_torch']"` in both import orders; `tex.DType is transformer_engine_torch_rocm.DType`. |

### P3 · EXIT-A — the facade holds with the fork's own Python — 1 day · **container**

The fork's current `transformer_engine/pytorch/*.py` is left **completely untouched**. Only P1 + P2
are applied.

| Run | The representative suite (§3.9) plus the four seam-order tests below. |
| Pass | Identical pass/fail set to the P0 baseline. Import-time delta within the Stage-1 budget (G6). |
| Seam-order tests | (a) `import transformer_engine.pytorch` first; (b) `import transformer_engine_torch` first; (c) a consumer that imports `torch`, then `transformer_engine.pytorch.module.linear` directly; (d) `importlib.reload(transformer_engine.pytorch.cpp_extensions)` then `dir()` equality against pre-reload. |
| If it fails | Stop. Diagnose by which of F1-F4 broke. This is the cheapest falsification point in the whole program and it must be reported as such, not worked around. |

### P4 · Overlay assembly tooling — 2 days · **this workstation**

The pure-Python wheel is *upstream tree + patch queue*. Build the assembler before vendoring, so
vendoring is a one-command operation from day one.

```
proposals/te-rocm-plugin/tools/assemble_overlay.py
  --upstream <sha>            git worktree of NVIDIA/TE at the pinned base (source only, no submodule init)
  --patches  patches/         ordered queue; one file per manifest entry: PT-002.patch, CM-003.patch ...
  --out      build/overlay/   assembled transformer_engine/ tree
  --check                     dry run: report per-patch applicability, exit 1 on any failure
```

- Patch format: unified diff against the upstream file, **named by manifest ID**, with a 6-line
  header (`# manifest: PT-002`, `# base: 868d8d92`, `# mechanism: capability`, `# expiry: …`,
  `# tests: …`, `# owner: …`). The assembler refuses a patch whose header ID is not in the manifest.
- Applicability fingerprint for the prototype = `git apply --check` with context. The AST-level
  fingerprint from §3.4 is Stage 3.
- Ordering and dependencies come from the manifest's `dependencies:` fields; the assembler
  topologically sorts and fails on cycles.
- Emits `overlay-manifest.json`: which patches applied, their hashes, the upstream SHA → this is the
  **bundle hash** the certified load check reads later.

| Exit | `--check` against an *unpatched* upstream tree applies zero patches cleanly and reports the full queue as "target present"; applying all patches produces a tree that `python -m py_compile`s. |

### P5 · Vendor 868d8d92 + build the patch queue — 5-7 days · **this workstation** for patches, **container** for the loop

This is the bulk of Stage 1. Order is chosen so the tree imports as early as possible, then runs.

| Step | Files | Manifest | Days |
|---|---|---|---|
| 5.1 Import path | `transformer_engine/__init__.py`, `common/__init__.py` (loader, `te_rocm_build`, `is_fp8_fnuz`) | CM-001, CM-002 | 1 |
| 5.2 Recipe | `common/recipe/__init__.py` — FNUZ formats, `MXFP4BlockScaling`, `Recipe.mxfp4()` | CM-003 (all three) | 0.5 |
| 5.3 Attention | `attention/dot_product_attention/{backends,utils,context_parallel}.py`, `cpp_extensions/fused_attn.py` — the enum-value gap (F6) | PT-010/011/012/026 | 1 |
| 5.4 Module core | `module/base.py`, `module/{linear,layernorm_linear,layernorm_mlp,grouped_linear}.py`, `module/_common.py` | PT-002/003/006/008/009/023 | 2 |
| 5.5 Quantization + tensors | `quantization.py`, `tensor/*`, `cpp_extensions/gemm.py`, `constants.py`, `utils.py` | PT-001/004/007/017/018/021/025/027/031/035/036/041 | 1.5 |
| 5.6 Remainder | optimizers, ops, graph, jit, transformer, `__init__`, `_extra_state`, fp8_padding, cross_entropy, setup | the rest of PT-* | 1 |

Rules for every patch, enforced by the assembler header check:

- **Build tier only.** No runtime overrides in the prototype (compiled-only milestone; nothing is
  `runtime_eligible: proven` yet). The registry exists as an empty module so Stage 3 has a slot.
- One patch per manifest **leaf**. A file with three features gets three patches with explicit
  `dependencies`. This is what makes M2 countable.
- **`try upstream unchanged first`** is applied literally: before writing a patch, run the suite with
  the upstream file verbatim. If it passes, no patch — the manifest entry's `status` becomes
  `retired-unchanged`. Expect this for several small entries in 5.6.
- ROCm-only files (`triton_kernels/`, MXFP4 stack, `fsdp2_allgather_tensor.py`) are **copied**, not
  patched — they have no upstream target. They land under `transformer_engine/pytorch/` unchanged in
  the prototype; relocation to `plugin/` is Stage 5.

| Exit | `assemble_overlay.py` produces a tree; every manifest PT/CM leaf is either a patch file or `retired-unchanged`; import succeeds; the loop in P7 begins. |

### P6 · Conformance tests — 2 days · **this workstation**, run in container

```
tests/te_rocm/test_seam_names.py       seam_inventory.py as a pytest: MISSING == {LayerNorm}; extras ⊆ denylist
tests/te_rocm/test_seam_values.py      NEW: enum VALUE inventory (F6) — every enum upstream references,
                                        member-by-member, vs the extension; expected-diff file for
                                        NVTE_Fused_Attn_Backend so the test is green with the known gap documented
tests/te_rocm/test_seam_signatures.py  for each demanded function: inspect.signature via pybind docstring
                                        parse vs upstream call-site arity (coarse; exact is Stage 3)
tests/te_rocm/test_import_order.py     the four P3 orders + fork/reload
tests/te_rocm/test_facade_identity.py  __all__ equality, is-identity of enums/classes, __module__ names,
                                        pickle round-trip of one enum and one class
tests/te_rocm/test_overlay_bundle.py   bundle hash matches overlay-manifest.json; every applied patch's
                                        manifest ID is status: proposed|active (not retired)
```

`seam_inventory.py` gets a `--values` mode for `test_seam_values.py` (walk `.value("NAME"` under
each `enum_<>` with the same `#if` tracking).

| Exit | All six green in the container against the P5 overlay, with `NVTE_Fused_Attn_Backend` in the expected-diff file. |

### P7 · EXIT-B — the Stage 1 gate — iterate, ~3 days · **container**

Loop: run representative suite on the overlay → triage failures by manifest ID → fix patch → rerun.

| Pass | Same pass/fail set as P0 baseline, **or** every delta explained by a manifest ID with a written reason. Zero in-place edits to files under `build/overlay/` that are not produced by a patch (assembler verifies by hash). P6 green. |
| Deliverable | `baselines/<date>-overlay.json` alongside the P0 baseline; the diff between them is the Stage-1 report. |

### P8 · Stage-1 performance and checkpoint gate — 2 days · **container**, one needs **8 GPUs**

Against the budget approved in G6 — the budget must exist *before* this runs.

| Check | How |
|---|---|
| Facade call overhead | microbench: 1e6 calls of `tex.<cheap fn>` direct vs via facade; report ns/call delta |
| Import time | `-X importtime` total for `transformer_engine.pytorch`, fork vs overlay |
| Graph breaks | `torch.compile` on `te.Linear` + `te.LayerNormMLP` with `TORCH_LOGS=graph_breaks`; count must not increase |
| hipGraph capture | `tests/pytorch/test_cuda_graphs.py` subset passes |
| Checkpoint | save BF16 + FP8-delayed state dicts on the fork build; load and continue 50 steps on the overlay; loss within band (§8.5 Stage 1/2 gate). `test_checkpoint.py` + one hand-written round-trip |
| e2e smoke | one Megatron-LM(ROCm) LLaMA-class config, 8×MI300X/MI350X, N steps; tokens/sec within budget margin of fork |

---

## 4. Track G — Stage 0 governance (parallel; some items gate Track P)

### G0 · The four decisions — owner: sponsor · **gates P5 (G0.1, G0.4), P8 (G0.2)**

| | Decision | Needed by | Default if undecided |
|---|---|---|---|
| G0.1 | `release_218_gap` — pin `868d8d92`, or take the 15 release commits / a later `main` SHA | **P5** (it's the vendoring SHA) | Prototype pins `868d8d92` as measured; decision recorded as provisional. Re-vendoring is one assembler run |
| G0.2 | `ifu_sourcing_policy` — always merge upstream `main` at a chosen SHA | Before the next IFU | Write it now; one paragraph in `CONTRIBUTING.rst` |
| G0.3 | `packaging_name_conflict` — canonical `transformer-engine` vs the fork's `-rocm7/-rocm10` | **P5.1** (CM-002-PKGID is in that patch) | Prototype keeps the fork's current names; patch is marked `provisional`. Stage 3 rewrites |
| G0.4 | `contract_surface_for_ctypes` (ABI-001) | Gate B | Not needed for the prototype; `is_fp8_fnuz` stays as-is in CM-002 |

### G1 · Regenerate `added_class` against 868d8d92 — 1 day · needs the **canonical-v2 classifier** (not in repo)

Run the classifier against the corrected base; replace every `added_class: REGENERATE` and
`m1_added_lines: PENDING_RECLASSIFICATION`; set `added_class_status: CURRENT`. Commit the classifier
into `tools/` so this stops being a manual step.

### G2 · CI base assertion — 0.5 day

Add a job to the existing GitHub workflow that runs `tools/measure_divergence.sh` (exits 2 on
merge-base mismatch) and `tools/seam_inventory.py` (exits 1 while OPEN — allowlist `LayerNorm` so
it can be made blocking). Non-gating for one cycle, then gating.

### G3 · Freeze on in-place edits — 1 day

CI check: any PR touching `transformer_engine/{pytorch,jax,common}/**/*.py` or
`transformer_engine/__init__.py` fails unless the change is a file under `patches/` or a file the
manifest lists as ROCm-only. Exempt `wen/dev-plugin` and the IFU branches. This is the mechanism
that stops the regression alarm from ever having a real instance.

### G4 · Manifest completion — 2 days, spread

Owners at workstream and leaf level; sponsor named; the two compound features (PT-006-FEAT,
PT-009-FEAT) split — P5.4 forces this anyway, since each needs its own patch; residues characterized
as they get patched in P5. `stage0_exit_requirements` becomes a script: `tools/check_manifest.py`
that fails on any unmet requirement.

### G5 · Upstream engagement — ongoing, one owner

In the proposal's order: NVFP4 bugfix PR (PT-008-NVFP4FIX is ready to extract the moment P5.4
writes its patch); `tex.LayerNorm` dead-code report (free); #3401 comments; `jax/moe.py` lazy-import
PR. Every merged PR retires a patch file — that's M2 going down.

### G6 · Numeric thresholds — **before P8 and before B2** · sponsor + one engineer, half a day

Two documents, both signed before the measurement they govern:

- **Stage-1 performance budget**: max facade ns/call, max import-time increase (ms and %), e2e
  throughput margin (%), graph-break delta (0), checkpoint-continuation loss band.
- **Backtest thresholds** (§7 Stage 2 list): repair-effort fraction, certification engineer-day
  budget, and the pass/fail rule per case.

Written into `proposals/te-rocm-plugin/thresholds.yaml` with a `approved_by` / `approved_on` field.
P8 and B2 read the file and refuse to run if it is unsigned.

---

## 5. Track B — the historical backtest (Stage 2)

### B1 · Reconstruct the inputs — 2 days

| Input | Derivation |
|---|---|
| Fork state pre-2.15-sync | `origin/release-sync-v2.15-260630` first parent, or the commit before that branch's IFU merge |
| Upstream 2.15 base | **second parent of that IFU merge commit** — the rule from Appendix A; `measure_divergence.sh --base` will assert it. Do **not** use `release_v2.15`'s tip |
| Upstream 2.17 target | the second parent of the 2.17 IFU merge = `2e559f06` (on `release_v2.17`, not `main` — record this as a known deviation, since the experiment must replay history, not correct it) |
| Historical effort | the branch's commit dates + PR review timeline: mid-July → Aug 22 |

### B2 · Replay — 5-7 days

1. From the pre-sync fork, extract the risk-weighted case set (12 cases in the manifest's
   `backtest_plan`) as patches against the 2.15 base, using the P4 assembler.
2. Bump the assembler's `--upstream` to the 2.17 target in certification mode.
3. Record per patch: reapplied cleanly / tripped by ID / silently wrong (the last is found by the
   P6 tests, which is why B2 depends on P6).
4. Repair only tripped patches; log engineer-hours per repair.

### B3 · Report against G6 thresholds — 1 day

Every threshold pass/fail, every case's outcome, the effort fraction vs historical, and — stated
plainly — what the backtest could not test (long-tail patches, package behaviour, the full extension
contract). That report plus the P7/P8 results is the **Gate A** packet.

---

## 6. Environment matrix

| Work package | Workstation (MI350X, no TE build) | Container (TE build) | Multi-GPU |
|---|---|---|---|
| P2 facade, P4 assembler, P6 tests (authoring), all patch authoring, all of Track G, B1 | ✔ | | |
| P0, P1, P3, P5 loop, P6 (running), P7, P8 microbench/import/compile/ckpt, B2 | | ✔ | |
| P8 e2e smoke | | ✔ | ✔ 8 GPUs |

Per the repo rule, the container image/tag and launch command are needed before P0. Nothing in the
left column is blocked on it.

---

## 7. Gate criteria

**Stage 0 exit** — `tools/check_manifest.py` passes (all 14 `stage0_exit_requirements`, incl. G1
regeneration, ABI-002 disposition, enum-value conformance); G2 and G3 live in CI; G6 signed.

**Stage 1 exit (EXIT-B + P8)** — overlay passes the representative suite through the facade with
every delta manifest-attributed; P6 green; P8 within the signed budget; checkpoint continuation
within band. **Without moving any C++.**

**Gate A** — Stage 1 exit + B3 report. Funds Stage 3. The question Gate A answers is narrow: *does
the compatibility layer reapply across a real upstream delta with effort a stated fraction of the
historical sync?* It does not answer whether the package, the long-tail patches, or the full
extension contract are production-ready — that is Gate B's question, answered by Stage 3.

---

## 8. Deliberately out of scope for the prototype

- **Runtime override registry** — exists as an empty module; nothing is `proven` yet.
- **AITER/Triton registry entries** — compiled-only (proposal §3.5).
- **CustomRecipe adapter** — MXFP4 goes in via CM-003 patches (F8); the adapter is Stage 5.
- **Backend extraction, SONAME, origin ledger** — Stage 4.
- **JAX** — separate beta track; the prototype is PyTorch-only. The assembler is framework-neutral
  so Stage 7 reuses it.
- **Packaging** (sole-ownership, local version, prebuilt wheels) — Stage 3. The prototype installs
  editable from the overlay directory.
- **Any change to the fork's `dev` branch.** Everything lands on `wen/dev-plugin` until Gate A.

---

## 9. Representative suite (§3.9 referenced above)

Chosen to cover each seam-sensitive area once, cheaply; the full suite is Stage 3.

| Area | Test | Why |
|---|---|---|
| Numerics through the seam | `tests/pytorch/test_numerics.py` (subset: Linear, LayerNormMLP, BF16+FP8-delayed) | most `tex.*` calls per test |
| Recipes | `tests/pytorch/test_custom_recipe.py`, `test_recipe.py` | CM-003, F8 |
| Attention + enum values | `tests/pytorch/attention/test_attention.py` (subset) | F6 — will fail until P5.3 |
| Graph capture | `tests/pytorch/test_cuda_graphs.py` | P8 |
| Checkpoint | `tests/pytorch/test_checkpoint.py` | §8.5 |
| Fused optimizers (early-bound site) | `tests/pytorch/test_fused_optimizer.py` | `optimizers/__init__.py` is one of the 9 |
| Ops (early-bound sites) | `tests/pytorch/test_fusible_ops.py` | `ops/basic/{layer_norm,rmsnorm}.py` |
| MXFP4 (ROCm-only, copied) | `tests/pytorch/mxfp4/` | proves copied files still bind |
| Compile | `tests/pytorch/test_torch_compile.py` | P8 graph-break count |

---

## 10. Risks specific to this plan

| Risk | Where it bites | Mitigation built in |
|---|---|---|
| EXIT-A fails on a facade property nobody anticipated (`__spec__`, pickling, `inspect`) | P3 | P2's identity test list; P3 is *designed* to be the cheap failure point — report, don't work around |
| Enum-value gap is deeper than attention (another enum diverges silently) | P5, P7 | P6 `test_seam_values.py` runs on every enum, not just the known one |
| Patch queue grows past the manifest floor because "try upstream unchanged" is skipped under time pressure | P5 | assembler refuses a patch without a manifest ID; `retired-unchanged` is the cheaper path, so incentives align |
| G6 thresholds get written *after* seeing P8/B2 numbers | P8, B2 | `thresholds.yaml` signature check; unsigned → the scripts refuse to run |
| `release_218_gap` decided late, forcing a re-vendor mid-P5 | P5 | assembler makes re-vendoring one command; patches are keyed to targets, not SHAs |
| Backtest reproduces the 2.17 off-main merge and someone "fixes" it, invalidating the replay | B1 | recorded as a known deviation up front; the experiment replays history |
| The container isn't available when P0 is ready | P0 | Track G and the left column of §6 carry ~2 weeks of unblocked work |
