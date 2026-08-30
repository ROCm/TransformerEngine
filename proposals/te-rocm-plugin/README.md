<!--
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TE-ROCm plugin refactor — proposal and divergence ledger

Working documents for moving `ROCm/TransformerEngine` from a hard fork to **pinned upstream TE Python
plus a ROCm plugin and an AMD-maintained compiled backend**.

| File | What it is |
|---|---|
| [`proposal.md`](proposal.md) | **v2.3** — architecture, migration, implementation and validation plan. The document requesting Stage 0-2 approval. |
| [`divergence-manifest.yaml`](divergence-manifest.yaml) | **v2.4.3** / schema 1.4 — file-complete ledger, 81 entries, machine-readable governance. |
| [`divergence-manifest-summary.md`](divergence-manifest-summary.md) | **v2.4.3** — reviewer-facing summary of the manifest. |
| [`manifest-corrections-v2.3-to-v2.4.yaml`](manifest-corrections-v2.3-to-v2.4.yaml) | Field-level audit trail of the v2.3 → v2.4 restatement. |
| [`implementation-plan.md`](implementation-plan.md) | Work breakdown for Stages 0–2: prototype track (P0–P8, two exits), governance track (G0–G6), backtest track (B1–B3), environment matrix, gate criteria. |
| [`implementation-plan-stages-3-9.md`](implementation-plan-stages-3-9.md) | Work breakdown for Stages 3–9 with contingency tags (`[GATE-A]`, `[GATE-B]`, `[EVIDENCE: …]`); adds facts F13–F20, incl. that the extension is not linked against the core lib (F13) and the JAX handler-dict seam (F16). |
| [`thresholds.yaml`](thresholds.yaml) | **Signed 2026-08-30** — Stage-1 performance budget, Stage-2 backtest thresholds, C++ arm reporting bands. P8 and B2 refuse to run if unsigned. |
| [`tools/assemble_overlay.py`](tools/assemble_overlay.py) | Builds the pure-Python overlay: upstream tree from the submodule + `patches/` in dependency order + fork-only files + symlinked binaries. `check` / `build` / `gen` / `diff-fork`. |
| [`patches/`](patches/) | The build-tier patch queue, one file per manifest entry, seeded from the fork's divergence. P5 shrinks and splits these. |
| [`tools/measure_divergence.sh`](tools/measure_divergence.sh) | Regenerates every figure in the above. Run it per sync. |

## Measurement rules (normative — read before quoting any number)

Both rules exist because breaking them already produced a materially wrong manifest (v2.3).

**1. The upstream base is the second parent of the fork's IFU merge commit**, and it must equal
`merge-base(upstream/main, fork/dev)`. Never a release-branch tip, never a tag.

Manifest v2.3 measured against `release_v2.18`'s tip (`27486e03`), which the fork never merged. The
fork's IFU merge took `868d8d92` — the point `release_v2.18` branched off `main`. That charged 15
upstream commits (1,339 in-scope lines) to ROCm, produced **five wholly phantom entries**, inflated ten
more, and **inverted the reported divergence trend**.

**2. Use GNU `diff`**, not `git diff`. `added` = `^>` lines, `removed` = `^<`; the "lines" figure in
summary tables is `added + removed`. git's default Myers algorithm disagrees on some files, and M1
burn-down plus the divergence-regression alarm are line-count based.

`tools/measure_divergence.sh` implements both, auto-detects the base, and **exits 2** if the merge-base
assertion fails. Wire that assertion into CI.

```bash
proposals/te-rocm-plugin/tools/measure_divergence.sh            # layer totals + ROCm-only volume
proposals/te-rocm-plugin/tools/measure_divergence.sh --per-file # per-file table
```

## Current state (fork `8af6efc` vs upstream `main` `868d8d92`)

Modified: **11,622** lines non-test, **17,133** in-scope including tests.
ROCm-only additions: **29,808** non-test (**38,238** with tests).

`added_class` (sidecar / guard / unmarked) is **withheld** in v2.4 pending re-classification against
the corrected base — hunk composition is base-dependent, so every v2.3 triple is retained only as
`stale_added_class`. Regenerating it is a Stage-0 exit requirement.

## Open decisions carried by these documents

| # | Decision | Due |
|---|---|---|
| 1 | `release_218_gap` — **decided 2026-08-30**: prototype pins `868d8d92`; stabilisation commits belong on the release branch under the two-track policy. | done |
| 2 | `ifu_sourcing_policy` — **decided 2026-08-30, two-track**: `dev` ← upstream `main`; `release_vX_rocm` ← upstream `release_vX`; never release→dev; release-only fixes cherry-picked as tracked patches. | done |
| 3 | `packaging_name_conflict` — **decided 2026-08-30**: keep the current `transformer_engine_rocm7/_rocm10/_rocm_jax` names; the canonical-name scheme is withdrawn. | done |
| 4 | `contract_surface_for_ctypes` — ABI-001: vendored upstream Python reaches the core ABI by ctypes, bypassing the extension API. | Gate B |
| 5 | `cxx_maintenance_strategy` — how the 85 hipified, guard-laden upstream C++ files are maintained after the split. **Provisional B**: patch queue over the submodule, then hipify. Decided on the backtest's C++ arm (trip rate of guard patches across 2.15→2.17). | Stage 4 |

## Note on copyright headers

These files carry an **AMD-only** header, deviating from the repo rule that new files get both AMD and
NVIDIA lines. That rule targets source files in a tree derived from upstream; these are AMD-authored
planning documents with no upstream provenance, so asserting NVIDIA copyright over them would be
incorrect. Change it if the project prefers uniformity.

## Seam inventory (static, no build required)

`tools/seam_inventory.py` answers whether the ROCm extension exposes every name upstream's Python
asks for — the coarse filter for `TE_ROCM_EXTENSION_API`. It AST-walks upstream at the pinned base
for the demand side and parses the fork's pybind sources (`pytorch/csrc` **and** `common/util`,
where the shared `NVTE_DECLARE_COMMON_PYBIND11_HANDLES` macro lives) for the supply side,
tracking `#if` nesting so CUDA-only registrations are reported separately.

```bash
proposals/te-rocm-plugin/tools/seam_inventory.py --base 868d8d9216da361c666519652115e23688db5211
```

Result at `868d8d92` is in `seam-inventory-868d8d92.txt`: 161 demanded / 176 ROCm-reachable.
**No genuine ROCm gap.** The single MISSING name (`tex.LayerNorm`) is unregistered in upstream's own
csrc too — an upstream dead-code bug in `onnx_extensions.py`. Eight names are CUDA-only by design
(cuSolverMp Newton-Schulz, cuBLASLt version, grouped-tensor GEMM) and every caller is
capability-guarded upstream. Exits 1 while the surface is OPEN so it can gate CI once `LayerNorm`
is allowlisted.
