<!--
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Implementation plan — Stages 0–2

**For:** proposal v2.3 / manifest v2.4.2. **Scope:** the work authorized by the Stage 0-2 approval:
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
| F1 | The fork builds its extension under upstream's exact name `transformer_engine_torch` | `build_tools/pytorch.py:150` | Renaming it is the single change that makes the seam necessary and testable |
| F2 | A build-time bootstrap slot already exists: `setup.py:BuildPy` generates `transformer_engine/_rocm_init.py` from `build_tools/templates/_rocm_init.py`, imported at `transformer_engine/__init__.py:19` **before** `import transformer_engine.common` (line 26) | `setup.py:46-75` | BOOT-001 extends this mechanism for what genuinely needs a pre-`common` hook (ROCm SDK preload, `te_rocm_build` detection). **The seam does not live here** — see F3 |
| F3 | **Upstream's own loader is the seam.** `common/__init__.py:load_framework_extension` finds the `.so` by file path, builds the module object, and inserts it into `sys.modules` under a name *it* chooses. It runs at `pytorch/__init__.py:18`, before the first submodule import at line 19 (`jax/__init__.py:32` likewise). `common/` Python never imports the seam (2 docstring mentions only) | upstream `common/__init__.py:147-190`, `pytorch/__init__.py:13-19` | "Install before the first seam import" is **upstream's guarantee, not ours**. No synthesized facade, no bootstrap module, no ordering race. The seam is a two-line alias inside a file the patch queue already carries (CM-002). A synthesized module is needed only when the registry routes ops away from the compiled extension (Stage 6) |
| F4 | 44 late-bound `import transformer_engine_torch as tex` sites; 9 early-bound `from … import` sites incl. a star-import at `pytorch/cpp_extensions/__init__.py:6` | seam inventory | The aliased extension module satisfies all of this by construction (it *is* a real module with eager attributes). `__all__` fidelity — hiding the 25 ROCm extras from the star-import — is a Stage 3 allowlist item, not a prototype requirement |
| F5 | Seam is **closed on names**: 161 demanded, 176 ROCm-reachable; the one MISSING (`tex.LayerNorm`) is an upstream dead-code bug | `seam-inventory-868d8d92.txt` | No facade shim needs to synthesize a function. Conformance work is on signatures/values, not names |
| F6 | Seam is **open on enum values**: `NVTE_Fused_Attn_Backend` binds `{AOTriton, CK, No_Backend}` on ROCm vs `{F16_max512_seqlen, F16_arbitrary_seqlen, FP8, No_Backend}` upstream; upstream Python early-binds it and compares against CUDA members | ABI-002-FAENUM, `common/util/pybind_helper.h:20-33` | The attention patches PT-010/011/012 are **mandatory** in the prototype; `EXTENSION_API` conformance must diff enum values |
| F7 | 17 seam names are registered from `common/util/pybind_helper.h` (backend side of BK-001), 11 upstream-demanded | ABI-002 | Extension build must keep including that header; not a Stage 1 problem, a Stage 4 one |
| F8 | Recipe dispatch is classmethod-based (`recipe.delayed()/.mxfp8()/.nvfp4()/.custom()`), with `recipe.custom()` → `CustomRecipeState` | upstream `pytorch/quantization.py:1317-1327` at `868d8d92`; fork equivalent at `:1439-1451` with `recipe.mxfp4()` inserted at `:1447` | MXFP4 has a legitimate hook (`custom()`); the fork's injected `Recipe.mxfp4()` is a shortcut that must become a build-tier patch until CustomRecipe absorbs it (Stage 5) |
| F9 | Vendored-Python divergence is 41 PyTorch + 18 JAX + 3 root/common files, 2,142+253 / 563+120 / 241+40 lines | manifest v2.4.1 | That is the patch queue's upper bound. PyTorch-only prototype: **44 files** |
| F10 | This session already runs **inside the build container** (Ubuntu 24.04.4, HIP 7.14, 8 × gfx950, torch 2.12.0+rocm7.14.0) but TE is **not yet built** | P0 record | Nothing in Track P is blocked on environment. P0 (build + baseline) can start immediately; the build itself is the first expensive step |
| F11 | Upstream is pinned as the submodule **`3rdparty/transformer_engine_nvidia`** at `868d8d92`, with `update = none` in `.gitmodules`. The repo's documented `git submodule update --init --recursive` (`CLAUDE.md:69`, three CI sites) therefore **skips it**; only the assembler initializes it, explicitly and non-recursively, so upstream's own `cutlass`/`nccl`/`googletest` never enter the build (proposal §3.2). The base also happens to be an ancestor of `origin/dev`, but that is a **cross-check, not the source of truth** | `.gitmodules`, `git -C 3rdparty/transformer_engine_nvidia rev-parse HEAD` | The pin is explicit and governed: CI asserts `submodule HEAD == manifest upstream_sha == merge-base(upstream/main, dev)`. The vendored tree is read from the submodule, never from the fork's history. **All C++ stays in this repo** — there is no separate backend repo; Stage 4 becomes build-target separation inside this tree |
| F12 | The 85 modified upstream C++ files are **all hipified** (none in the native-excluded dirs); **78 carry ROCm guards, ~480 sites** (287 `#ifdef __HIP_PLATFORM_AMD__`, 179 `#ifndef`); no generated `_hip.*` files are in the tree. The 65 ROCm-only files (20,968 ln) have no upstream ancestor | `measure_divergence.sh`, `grep` over the modified set | The post-split C++ maintenance strategy (proposal §4.1a) is **provisional B — patch queue over the submodule, then hipify** — and is decided at Stage 4 on the strength of the backtest's **C++ arm** (B2 step 5). The prototype does not touch the C++ tree |

---

## 2. Critical path

```
  G0 decisions ─┐
                │      ┌─── P1 rename ext ──► P2 loader alias ──────► P3 EXIT-A ──┐
  G1 classify ──┼──────┤      (fork's own Python through the seam; no vendoring)    │
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

- **EXIT-A** — the seam holds *with the fork's own Python, unchanged*. Isolates the mechanism
  (F1-F4) from vendoring. If this fails, the architecture is falsified for the cost of ~3 days.
- **EXIT-B** — vendored upstream Python + patch queue passes the representative suite through the
  seam. This is the proposal's Stage 1 gate.

---

## 3. Track P — the prototype (Stage 1)

### P0 · Environment and baseline — 1 day · **container** — **DONE 2026-08-30** (`baselines/2026-08-30-fork.json`)

| | |
|---|---|
| Do | Record container image/tag, ROCm, GPU arch, TE commit, submodule state (repo rule). **Recorded 2026-08-30**: host `cv350-rck-g03-e05-18`, Ubuntu 24.04.4, HIP 7.14.60850, **8 × gfx950 (MI350X)**, Python 3.12.3, torch 2.12.0+rocm7.14.0, TE `wen/dev-plugin`, submodule `868d8d92`; image tag not exposed in the environment — record it from the launcher. Build the fork as-is. Run the representative suite (§3.9) and capture pass/fail + wall time as the baseline. Capture import time (`python -X importtime -c "import transformer_engine.pytorch"`). |
| Exit | Baseline numbers committed to `proposals/te-rocm-plugin/baselines/<date>-fork.json`. |
| Blocks | Everything in Track P that runs code. |

### P1 · Rename the extension — 0.5 day · **container** — **DONE 2026-08-30** (name: `transformer_engine_rocm_torch`, 2.3 min)

| | |
|---|---|
| Do | `build_tools/pytorch.py:150`: `name="transformer_engine_torch"` → `"transformer_engine_rocm_torch"`. **Delete any existing `transformer_engine_torch*.so` from the package dir before rebuilding** — the loader's finder matches by prefix and returns the first hit (`common/__init__.py:_find_shared_object_in_te_dir`), so a stale old `.so` would be loaded and mask the failure this step exists to observe. Rebuild. Confirm `import transformer_engine.pytorch` now **fails** at `load_framework_extension` because `PyInit_transformer_engine_torch` is absent from the renamed `.so`. |
| Why | Makes the old name genuinely absent, so a seam bug fails loudly (the clean-install property from F3) instead of being masked by the compiled module answering under the old name. |
| Exit | The failure is observed and its traceback names one of the 9 early-bound sites or the root. |

### P2 · The seam: an alias in upstream's loader — 0.5 day — **DONE 2026-08-30** (two edits in `common/__init__.py`)

**No new modules.** An earlier draft of this plan specified `te_rocm/facade.py`, `te_rocm/bootstrap.py`
and a top-level `transformer_engine_torch` bootstrap package. They were designed from the proposal's
§3.5 description, which comes from TE Lite and FlagOS — projects that *replaced* the extension with
pure Python and therefore had to synthesize a module. For the compiled-only milestone, upstream's
loader (F3) already does everything a facade would, and doing it again by copying is strictly worse
(a copy can drift; an alias cannot).

The seam is two lines inside `load_framework_extension` — a function the patch queue already carries
as **CM-002**:

```python
# common/__init__.py :: load_framework_extension("torch")          [part of the CM-002 patch]
compiled = f"{module_name}_rocm"                       # PyInit_<name> must match spec.name
spec = importlib.util.spec_from_file_location(compiled, _get_shared_object_file(framework))
solib = importlib.util.module_from_spec(spec)
sys.modules[compiled] = solib
spec.loader.exec_module(solib)
sys.modules[module_name] = solib                       # <-- the seam: upstream's name, ROCm's module
```

Free by construction, because the module *is* the extension: eager attributes, `__spec__`/`__file__`,
`dir()`, and enum/class **identity** (`tex.DType is transformer_engine_rocm_torch.DType`). Nothing to
reconstruct, nothing that can be reconstructed wrongly.

Deliberately **not** done in the prototype — each is a later-stage item, and adding it now would
put our code in the path EXIT-A is meant to test:

- **`__all__` filtering** of the 25 ROCm extras. They leak into `te.pytorch.cpp_extensions` via the
  star-import; harmless, recorded by the inventory. The allowlist belongs to `EXTENSION_API` (Stage 3).
- **Direct-import-first order** (`import transformer_engine_torch` before `transformer_engine`).
  Upstream's own tests never use it — all import `transformer_engine.pytorch` first. On a clean install
  it fails loudly with `ModuleNotFoundError`, which is the *desired* property (no NVIDIA module can
  answer under the name). Revisit in Stage 3 only if a consumer needs it.
- **A synthesized module object.** Needed only when the registry routes individual ops away from the
  compiled extension (Stage 6).

One visible effect of the rename, to be **asserted, not hidden**: every pybind class now reports
`__module__ == "transformer_engine_rocm_torch"`. Checked for checkpoint impact — `get_extra_state`
(`module/base.py:1424`) pickles a recipe dataclass, CPU tensors and a dict, **no pybind objects** — so
existing checkpoints are unaffected. (Recipe *class* paths are a separate matter: CM-003 / §8.5
Stage 5.)

| Exit | `python -c "import sys, transformer_engine.pytorch; a = sys.modules['transformer_engine_torch']; b = sys.modules['transformer_engine_rocm_torch']; assert a is b; import transformer_engine_torch as tex; assert tex is a and tex.DType is b.DType"`. |

### P3 · EXIT-A — the seam holds with the fork's own Python — 1 day — **REACHED 2026-08-30**: 0 outcome flips vs P0 across 10 files; import 2.92 s → 2.63 s; all four import orders; fork-saved checkpoints load (`baselines/2026-08-30-seam-exit-a.json`)

The fork's current `transformer_engine/pytorch/*.py` is left **completely untouched**. Only P1 + P2
are applied — which means EXIT-A now tests **upstream's own mechanism under a rename, with nothing
of ours in the path** except the alias line.

| Run | The representative suite (§3.9) plus the four seam-order tests below. |
| Pass | Identical pass/fail set to the P0 baseline. Import-time delta within the Stage-1 budget (G6). |
| Seam-order tests | (a) `import transformer_engine.pytorch` first — the only order upstream uses; (b) `import transformer_engine_torch` **first**, on a clean install → must raise `ModuleNotFoundError`. This asserts the loud-failure property (F3): a wrong order fails, it does not split-brain; (c) a consumer that imports `torch`, then `transformer_engine.pytorch.module.linear` directly; (d) `importlib.reload(transformer_engine.pytorch.cpp_extensions)` then `dir()` equality against pre-reload. |
| If it fails | Stop. Diagnose by which of F1-F4 broke. This is the cheapest falsification point in the whole program and it must be reported as such, not worked around. |

### P4 · Overlay assembly tooling — 2 days — **DONE 2026-08-30** (`tools/assemble_overlay.py`; 62 seed patches; self-test 224/224 files identical; overlay imports and runs on GPU)

The pure-Python wheel is *upstream tree + patch queue*. Build the assembler before vendoring, so
vendoring is a one-command operation from day one.

```
proposals/te-rocm-plugin/tools/assemble_overlay.py
  --upstream <sha>            read from the 3rdparty/transformer_engine_nvidia submodule (F11). The assembler
                              runs `git submodule update --init --checkout 3rdparty/transformer_engine_nvidia`
                              itself - explicitly, NON-recursive - then asserts the submodule's HEAD equals
                              <sha> and equals the manifest's upstream_sha; refuses to run otherwise.
                              Moving the pin (release_218_gap) = checkout a new SHA in the submodule +
                              update the manifest; the assembler catches any disagreement between them.
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

### P5 · Vendor 868d8d92 + build the patch queue — **SHRINK DONE 2026-08-31 (corrected)**: 62 seeds → 45 active (17 retired on executed+Phase-C evidence, 3 gfx950-only; 7 untested-kept; 19 needed incl. a 20h hipGraph-hang proof for PT-028). First shrink run withdrawn (runner imported the fork tree). Feature-level splitting still to do

This is the bulk of Stage 1. Order is chosen so the tree imports as early as possible, then runs.

| Step | Files | Manifest | Days |
|---|---|---|---|
| 5.1 Import path | `transformer_engine/__init__.py`, `common/__init__.py` (loader **+ the P2 seam alias**, `te_rocm_build`, `is_fp8_fnuz`) | CM-001, CM-002 | 1 |
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
tests/te_rocm/test_seam_names.py       seam_inventory.py as a pytest: MISSING == {LayerNorm}; extras recorded
                                        (leak accepted in the prototype; the allowlist is Stage 3)
tests/te_rocm/test_seam_values.py      NEW: enum VALUE inventory (F6) — every enum upstream references,
                                        member-by-member, vs the extension; expected-diff file for
                                        NVTE_Fused_Attn_Backend so the test is green with the known gap documented
tests/te_rocm/test_seam_signatures.py  for each demanded function: inspect.signature via pybind docstring
                                        parse vs upstream call-site arity (coarse; exact is Stage 3)
tests/te_rocm/test_import_order.py     the four P3 orders incl. the loud-failure assertion, + fork/reload
tests/te_rocm/test_seam_identity.py    sys.modules alias is-identity; every pybind class reports
                                        __module__ == "transformer_engine_rocm_torch" (documents the rename);
                                        pickle round-trip of one enum and one class
tests/te_rocm/test_overlay_bundle.py   bundle hash matches overlay-manifest.json; every applied patch's
                                        manifest ID is status: proposed|active (not retired)
```

`seam_inventory.py` gets a `--values` mode for `test_seam_values.py` (walk `.value("NAME"` under
each `enum_<>` with the same `#if` tracking).

| Exit | All six green in the container against the P5 overlay, with `NVTE_Fused_Attn_Backend` in the expected-diff file. |

### P7 · EXIT-B — the Stage 1 gate — **REACHED 2026-08-31 (corrected run)**: upstream 868d8d92 Python + 45 patches, overlay import verified per process, 0 outcome flips vs P0; import 2.61 s (`baselines/2026-08-30-p5-phase-c.json`)

Loop: run representative suite on the overlay → triage failures by manifest ID → fix patch → rerun.

| Pass | Same pass/fail set as P0 baseline, **or** every delta explained by a manifest ID with a written reason. Zero in-place edits to files under `build/overlay/` that are not produced by a patch (assembler verifies by hash). P6 green. |
| Deliverable | `baselines/<date>-overlay.json` alongside the P0 baseline; the diff between them is the Stage-1 report. |

### P8 · Stage-1 performance and checkpoint gate — **PASS 2026-08-31** (`baselines/2026-08-30-p8-gate.json`): import parity (2.64 s both), 0 graph breaks both trees, checkpoint continuation 50 steps max rel diff **0.0**, 8-GPU DDP proxy 1.4696 vs 1.4662 M tokens/s (+0.2 %); Megatron-LM(ROCm) e2e row pending external setup

Against the budget approved in G6 — the budget must exist *before* this runs.

| Check | How |
|---|---|
| Seam call overhead | **Expected zero by construction** — the alias means `tex.<fn> is transformer_engine_rocm_torch.<fn>`; assert that identity for every demanded function. Keep a 1e6-call microbench (direct vs via `tex`) only as evidence for the record; a non-zero delta means the alias was replaced by something else and is a bug, not a cost |
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
| G0.1 | `release_218_gap` | **P5** | **DECIDED 2026-08-30**: prototype pins `868d8d92`; shipping pin follows the two-track branch policy (stabilisation commits land on the release branch, not `dev`) |
| G0.2 | `ifu_sourcing_policy` | Before the next IFU | **DECIDED 2026-08-30**: two-track — `dev` ← upstream `main`; `release_vX_rocm` ← upstream `release_vX`; never release→dev; release-only fixes cherry-picked as tracked patches. Remaining action: make G2's identity check branch-aware; write the paragraph into `CONTRIBUTING.rst` (location to confirm) |
| G0.3 | `packaging_name_conflict` | **P5.1** | **DECIDED 2026-08-30**: keep `transformer_engine_rocm7/_rocm10/_rocm_jax`; v2.2 scheme withdrawn. CM-002-PKGID is no longer a conflict. Follow-up: turn the `get_te_core_package_info(rocm: bool)` signature change into an additive helper so the patch stops tripping |
| G0.4 | `contract_surface_for_ctypes` (ABI-001) | Gate B | Open. Leaning: tiny versioned core-introspection C API (~3 symbols) in a public header. Not needed for the prototype |
| G0.5 | `cxx_maintenance_strategy` | Stage 4 | Open, provisional B; evidence from B2's C++ arm |

### G1 · Regenerate `added_class` against 868d8d92 — 1 day · needs the **canonical-v2 classifier** (not in repo)

Run the classifier against the corrected base; replace every `added_class: REGENERATE` and
`m1_added_lines: PENDING_RECLASSIFICATION`; set `added_class_status: CURRENT`. Commit the classifier
into `tools/` so this stops being a manual step.

### G2 · CI base assertion — 0.5 day

Add a job to the existing GitHub workflow that asserts the three-way pin identity —
`git -C 3rdparty/transformer_engine_nvidia rev-parse HEAD` == manifest `upstream_sha` ==
`merge-base(upstream/main, dev)` — then runs `tools/measure_divergence.sh --base <submodule HEAD>`
and `tools/seam_inventory.py` (exits 1 while OPEN — allowlist `LayerNorm` so it can be made
blocking). Non-gating for one cycle, then gating. The job must initialize the submodule itself,
non-recursively; the workflow's existing `--init --recursive` skips it by design (F11).

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

- **Stage-1 performance budget**: seam ns/call (expected 0; asserted by identity, see P8), max import-time increase (ms and %), e2e
  throughput margin (%), graph-break delta (0), checkpoint-continuation loss band.
- **Backtest thresholds** (§7 Stage 2 list): repair-effort fraction, certification engineer-day
  budget, and the pass/fail rule per case.

Written into `proposals/te-rocm-plugin/thresholds.yaml` with a `approved_by` / `approved_on` field.
P8 and B2 read the file and refuse to run if it is unsigned.

**DONE 2026-08-30.** `thresholds.yaml` is signed: seam overhead 0 (by identity); import time ≤ +10 %;
e2e throughput ≤ 1 % loss; graph breaks +0; checkpoint continuation 50 steps within 1e-3 relative;
backtest zero silent failures, repair ≤ 25 % of historical, certification ≤ 5 engineer-days; C++ arm
bands < 0.30 mechanized / 0.30–0.60 discuss / > 0.60 relabelled IFU. Owners are **deferred** — no
sponsor or workstream owners at this stage; `owner: TBD` fields are inert.

---

## 5. Track B — the historical backtest (Stage 2)

### B1 · Reconstruct the inputs — **DONE 2026-08-31**: PRE `bc3766e6d`, BASE `42b84005` (= release_v2.15 tip, what history merged), TGT `2e559f06`; all 12 case files exist at their current paths in the 2.15 era

| Input | Derivation |
|---|---|
| Fork state pre-2.15-sync | `origin/release-sync-v2.15-260630` first parent, or the commit before that branch's IFU merge |
| Upstream 2.15 base | **second parent of that IFU merge commit** — the rule from Appendix A; `measure_divergence.sh --base` will assert it. Do **not** use `release_v2.15`'s tip |
| Upstream 2.17 target | the second parent of the 2.17 IFU merge = `2e559f06` (on `release_v2.17`, not `main` — record this as a known deviation, since the experiment must replay history, not correct it) |
| Historical effort | the branch's commit dates + PR review timeline: mid-July → Aug 22 |
| **C++ arm inputs** | the guard edits in the pre-sync fork's `common/` C++ vs the 2.15 base, expressed as one patch per file by the P4 assembler (same format as the Python queue, `cxx_strategy: patch-queue`). Sample = the manifest's `backtest_plan.cxx_arm` selection rule; confirm the candidate files against actual 2.15→2.17 upstream churn in `common/` |

### B2 · Replay — **DONE 2026-08-31** (`tools/backtest.py`): 12 Python cases + 8 C++-arm files as whole-file patches at BASE, applied against TGT. Python: 3 reapplied, 9 TRIPPED loudly by ID (3-way repair proxy: 1 clean, rest 1–9 conflict hunks). C++: 4/8 tripped, every trip 1 hunk. Zero silent outcomes. Caveat: whole-file patches are an upper bound on trips

1. From the pre-sync fork, extract the risk-weighted case set (12 cases in the manifest's
   `backtest_plan`) as patches against the 2.15 base, using the P4 assembler.
2. Bump the assembler's `--upstream` to the 2.17 target in certification mode.
3. Record per patch: reapplied cleanly / tripped by ID / silently wrong (the last is found by the
   P6 tests, which is why B2 depends on P6).
4. Repair only tripped patches; log engineer-hours per repair.
5. **C++ arm.** Bump the C++ patch set (B1 inputs) the same way; for each file record re-applied /
   tripped / applied-but-hipify-or-compile-fails (the third is the C++ analogue of "silently wrong"
   and needs the container). Report the **trip rate** and the repair hours per tripped file. This
   number is the evidence for the Stage-4 `cxx_maintenance_strategy` decision (proposal §4.1a): a
   high trip rate means strategy B is a relabelled IFU; a low one means it is a real mechanization.
   The arm is **informational at Gate A**, not pass/fail — the C++ strategy is not a Stage 0-2
   deliverable.

### B3 · Report — **DONE 2026-08-31** (`baselines/2026-08-31-backtest-b3.json`): case rule satisfied (100 % reapply-or-trip-by-ID, 0 silent); cxx_arm trip rate 0.50 → band **discuss**; limitations stated (no 2.15-era build; conflict-hunk proxy for repair effort)

Every threshold pass/fail, every case's outcome, the effort fraction vs historical, **the C++ arm's
trip rate and per-file outcomes (informational)**, and — stated plainly — what the backtest could
not test (long-tail patches, package behaviour, the full extension contract). That report plus the
P7/P8 results is the **Gate A** packet.

---

## 6. Environment matrix

| Work package | Workstation (MI350X, no TE build) | Container (TE build) | Multi-GPU |
|---|---|---|---|
| P2 loader alias, P4 assembler, P6 tests (authoring), all patch authoring, all of Track G, B1 | ✔ | | |
| P0, P1, P3, P5 loop, P6 (running), P7, P8 microbench/import/compile/ckpt, B2 | | ✔ | |
| P8 e2e smoke | | ✔ | ✔ 8 GPUs |

This session already runs in the container (F10); the "workstation" column is retained only to mark
which packages do not need a TE build or GPUs at all. Nothing is blocked on environment.

---

## 7. Gate criteria

**Stage 0 exit** — `tools/check_manifest.py` passes (all 14 `stage0_exit_requirements`, incl. G1
regeneration, ABI-002 disposition, enum-value conformance); G2 and G3 live in CI; G6 signed.

**Stage 1 exit (EXIT-B + P8)** — overlay passes the representative suite through the seam with
every delta manifest-attributed; P6 green; P8 within the signed budget; checkpoint continuation
within band. **Without moving any C++.**

**Gate A** — Stage 1 exit + B3 report. Funds Stage 3. The question Gate A answers is narrow: *does
the compatibility layer reapply across a real upstream delta with effort a stated fraction of the
historical sync?* It does not answer whether the package, the long-tail patches, or the full
extension contract are production-ready — that is Gate B's question, answered by Stage 3.

---

## 8. Deliberately out of scope for the prototype

- **A synthesized facade module** (`te_rocm/facade.py`, `te_rocm/bootstrap.py`, a top-level
  `transformer_engine_torch` package) — the compiled-only seam is an alias in upstream's own loader
  (P2, F3). A synthesized module is needed only when the registry routes ops away from the compiled
  extension (Stage 6). Proposal §3.5 over-specifies this for the compiled-only milestone.
- **Runtime override registry** — exists as an empty module; nothing is `proven` yet.
- **AITER/Triton registry entries** — compiled-only (proposal §3.5).
- **CustomRecipe adapter** — MXFP4 goes in via CM-003 patches (F8); the adapter is Stage 5.
- **Backend separation, SONAME, origin ledger** — Stage 4. Decided: **the C++ stays in this repo**;
  there is no separate backend repo. Stage 4 separates by build target and directory, not by
  repository (manifest BK-001/002/003 and proposal §3.2/§4.1 updated accordingly).
- **The C++ maintenance strategy** (proposal §4.1a) — provisional **B** (patch queue over the
  submodule, then hipify), C selectively, A never as policy. Decided at Stage 4 with the backtest's
  C++ arm in hand (B2 step 5). The prototype builds the fork's C++ exactly as today.
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
| EXIT-A fails on a loader/alias property nobody anticipated (`PyInit_` name mismatch, `__module__`-sensitive code, reload) | P3 | P2's identity test list; P3 is *designed* to be the cheap failure point — report, don't work around |
| Enum-value gap is deeper than attention (another enum diverges silently) | P5, P7 | P6 `test_seam_values.py` runs on every enum, not just the known one |
| Patch queue grows past the manifest floor because "try upstream unchanged" is skipped under time pressure | P5 | assembler refuses a patch without a manifest ID; `retired-unchanged` is the cheaper path, so incentives align |
| G6 thresholds get written *after* seeing P8/B2 numbers | P8, B2 | `thresholds.yaml` signature check; unsigned → the scripts refuse to run |
| `release_218_gap` decided late, forcing a re-vendor mid-P5 | P5 | assembler makes re-vendoring one command; patches are keyed to targets, not SHAs |
| Backtest reproduces the 2.17 off-main merge and someone "fixes" it, invalidating the replay | B1 | recorded as a known deviation up front; the experiment replays history |
| The container isn't available when P0 is ready | P0 | Track G and the left column of §6 carry ~2 weeks of unblocked work |
