# Runbook: upstream pin bump (§5 workflow — replaces the merge-based IFU)

One page. Prereqs: build container, GPU box, `universal-ctags`, clean tree on the working branch.

1. **Preview first** (no tree changes): nightly Job B output, or locally
   `git -C 3rdparty/transformer_engine_nvidia fetch origin <ref>` then
   `tools/patch_fingerprints.py verify --upstream <sha>` (Python queue) and
   `tools/cxx_fingerprints.py verify --upstream <sha>` (C++ queue). The output *is* the repair
   plan: per-patch verdicts with touched symbols named. Trip-rate expectations: see
   `thresholds.yaml` cxx bands; record actuals either way (Stage-8 evidence).
2. **Branch-aware target**: dev-lineage bumps to upstream `main`; `release_vX_rocm` bumps to
   upstream `release_vX`. Never mix (two-track policy, manifest §2.2).
3. **Bump**: `git -C 3rdparty/transformer_engine_nvidia checkout <sha>`; update
   `metadata.upstream_sha` in `divergence-manifest.yaml` (three-way identity check in CI will
   hold you honest: submodule == manifest == merge-base).
4. **Reapply + repair**: `tools/assemble_overlay.py build` (Python queue) and
   `tools/cxx_queue.py verify` + `assemble` (C++ queue). A tripped patch names its file and
   symbols — repair the patch (not the tree), regenerate with `gen <ID>`, keep the `# tests:`
   header current. Hunks that vanished because upstream absorbed them: retire the patch
   (move to `patches/retired/`, flip `p5_status`, keep the residue invariant green).
5. **Regenerate derived state**: both fingerprint sets, `origin-ledger.json`,
   `measure_divergence.sh` figures into the manifest; `classify_hunks.py` per-entry deltas
   (divergence GROWTH is the regression signal).
6. **Certify**: rebuild core + extension; run the §8.6 chain (P6 conformance, checkpoint gate,
   perf gates vs `thresholds.yaml` — e2e on median-of-3 alternating runs), lifecycle wheels.
7. **Record**: wall-clock + engineer-days into the Stage-8 log. Two clean live bumps = the
   "days rather than weeks" claim; a bad one = revisit `cxx-strategy.yaml` with the new data.
