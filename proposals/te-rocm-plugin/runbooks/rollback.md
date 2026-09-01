# Runbook: release rollback with checkpoints (S5.2 gate, production shape)

Invariant that makes this work: pickle GLOBALs are pinned to stable module paths and shim
modules exist at the old paths on BOTH sides of any relocation (S5.2; verified bit-identical
both directions — baselines/2026-09-01-s52-ckpt-gate.json).

1. Install the previous release's wheels (all three dists at ONE version — the loader enforces
   version identity across `transformer_engine`, `transformer_engine_rocm{N}`,
   `transformer_engine_rocm_torch`).
2. Load the checkpoint written by the newer release. Expected: loads cleanly; if a class path
   fails to resolve, the newer release moved a pickled class without a shim — that is a
   release-blocking bug, not an ops problem (file it).
3. Continue N steps; loss curve must sit inside the 1e-3 relative band vs the pre-rollback run.
4. Roll forward: newer wheels again, load the rollback-era checkpoint, same band.
5. `python -m transformer_engine.te_rocm_diagnostics` before and after — attach both snapshots
   to the drill record.
