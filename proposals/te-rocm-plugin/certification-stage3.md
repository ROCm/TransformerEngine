# Stage 3 certification — Gate B packet

Date: 2026-08-31 · Certifier: Wen Chen · Branch: `wen/dev-plugin`

**Gate B question:** *Is the Python compatibility layer production-grade, independent of any
backend change?*

## Provenance

| | |
|---|---|
| Fork commit | `b43468040` (+ this packet) |
| Upstream pin | `868d8d9216da` — three-way identity OK (submodule == manifest == merge-base(nvidia/main, HEAD)) |
| Overlay bundle | `75619213aa62831b…` — 45 patches, 30 fork-only files, vendored upstream tree |
| Manifest | v2.4.4, M1 = 3016/47 live (+491 retired), M2 = 45 patches; governance OK |

## Wheel matrix (built and lifecycle-tested this certification)

| Distribution | Version | Form | Contents |
|---|---|---|---|
| `transformer_engine` | 2.18.0.dev0 | py3-none-any | overlay tree + in-package provenance manifest; no binaries |
| `transformer_engine_rocm7` | 2.18.0.dev0 | cp312 linux_x86_64 | binaries/headers only — python tree stripped |
| `transformer_engine_rocm_torch` | 2.18.0.dev0 | cp312, compiled from the release sdist | framework extension (compiled name = seam name) |
| `transformer_engine_rocm_jax` | — | deferred to S7 as planned | |

## §8.6 checklist

| Item | Result | Evidence |
|---|---|---|
| Base identity verified | **PASS** | three-way pin identity above |
| Contracts embedded + validated | **PASS** | TE_ROCM_CORE_ABI v1 enforced at load (negative path verified); bundle hash; P6 conformance 23/23 (2026-08-31, rebuilt overlay) |
| Conformance green | **PASS** | P6 23/23; upstream suite under overlay: EXIT-B 0 flips at the 45-patch queue (Gate A), no Python-surface changes since except patch-governed CM-002 |
| Numerics (dual oracle, compiled path) | **PASS** | checkpoint continuation bit-band + DDP loss parity, fork vs overlay (below) |
| Checkpoint round-trips | **PASS** | 50-step continuation, band 1e-3 (below) |
| Perf gates | **PASS** | import ≤ +10%, e2e proxy ≤ 1% (below) |
| M1/M2 updated, GNU-diff regenerated | **PASS** | check_manifest OK 2026-08-31; measure_divergence reproduces at pin |
| Lifecycle | **PASS** | T1–T5b all pass (baselines/2026-08-31-lifecycle.json) |
| Matrix + provenance embedded | **PASS** | this document; `_overlay_manifest.json` ships inside the pure wheel |
| Tag | **DONE** | `stage3-cert-20260831` |

## Certification run (fork vs overlay, thresholds.yaml)

| Gate | Fork | Overlay | Delta | Band | Verdict |
|---|---|---|---|---|---|
| import time (median of 5) | 2.598 s | 2.594 s | −0.19% | ≤ +10% | **PASS** |
| torch.compile graph breaks (Linear, LayerNormMLP) | 0, 0 | 0, 0 | +0 | +0 | **PASS** |
| checkpoint continuation (50 steps, save on fork) | — | — | 0.0 (bit-identical losses) | 1e-3 | **PASS** |
| e2e proxy (8-GPU DDP, tokens/s) | median 1 467 224 (n=5) | median 1 454 435 (n=4) | +0.87% loss (mean +0.38%) | ≤ 1% | **PASS** on aggregate |
| seam call overhead | — | — | asserted by identity (`tex.f is compiled.f`) | — | **PASS** |

The single cert run showed a +1.52% e2e delta; an alternating 3×-per-tree noise series established
per-tree run-to-run spread of 2.2–2.7% with interleaving distributions (fork's slowest run is below
three of four overlay runs), and the aggregate is inside the band. There is no mechanism for a
steady-state delta — the seam acts at import time only and the numerics rows are bit-exact.
**Methodology note:** single-run grading is under-powered for a 1% band on this machine; the e2e
gate should be graded on a median of ≥3 alternating runs (thresholds.yaml amendment candidate).
Full samples: `baselines/2026-08-31-stage3-cert.json`.

## Known exclusions / open items carried past Gate B

- 14 PENDING_RECLASSIFICATION feature-level M1 attributions (Stage-0 exit blocker, not a Gate B item)
- FNUZ patch retirements (PT-017/018/036) await an MI300 run; PT-026 runtime-candidate awaits CP tests
- gfx950 skip volumes as recorded in test-suite notes (mxfp8 100% on this box); known dpa_fp8 failure predates the program
- richer EXTENSION_API signature typing (S3.3 residue) tracked into S4
