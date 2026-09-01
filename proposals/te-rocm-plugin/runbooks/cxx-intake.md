# Runbook: C++ upstream intake via the origin ledger

1. `tools/origin_ledger.py` regenerates `origin-ledger.json` at the current pin; `git diff`
   of the ledger between two pins IS the C++ intake report: per file — upstream blob moved?
   state flipped (identical <-> diverged)? strategy/patch ids changed?
2. For `patch-queue` files, `tools/cxx_fingerprints.py verify --upstream <sha>` names the
   symbols behind every `target-changed` — that list is the repair work, nothing else is.
3. `native-hip` class = ROCm-only trees only (fused_attn_rocm/, amd_detail/, ck_fused_attn/,
   rocshmem_api/, ck_grouped_gemm/, rocm_gemm). Rule from S4.1 (twice-amended, measured):
   whole-file conversion of upstream-ancestor files is counter-indicated; the future criterion
   is an EXTRACTABLE AMD-only kernel >50% of the file, moved alone. A file tripping on 2
   consecutive pin bumps is promoted to native-hip REVIEW (review, not automatic conversion).
4. Tree identity is the exit: `cxx_queue.py assemble` must report byte-identical vs the fork's
   common/ tree. If it cannot, the queue and the tree have diverged — fix the queue.
