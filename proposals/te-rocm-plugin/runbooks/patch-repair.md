# Runbook: repairing a tripped patch

1. Identify: the fingerprint verify names the patch, its file, and the symbols upstream touched
   (`target-changed`), moved (`target-moved`), or removed (`file-gone`).
2. `target-moved`: usually clean — `git apply` often still succeeds; if not, re-anchor the hunk
   context and `gen <ID>` from the repaired fork file.
3. `target-changed`: open upstream's new version of the symbol; re-express the fork's edit on
   top of it in the FORK TREE, then `tools/assemble_overlay.py gen <ID>` (or `cxx_queue.py gen`)
   to regenerate the patch from the tree. Never hand-edit patch hunks.
4. `file-gone`: find where upstream moved the logic (fingerprints report the symbol names —
   grep upstream for them); the patch either follows the code or retires.
5. Verify: `gen --verify` / `cxx_queue.py verify` (apply-at-pin, byte-compare) must return
   0 stale; `check_retired_residue.py` must stay at 0 violations; run the patch's `# tests:`.
6. A patch whose divergence upstream absorbed entirely: retire it — that is the program
   working, log it in the manifest.
