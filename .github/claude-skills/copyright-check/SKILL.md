---
name: copyright-check
description: Verify AMD copyright header compliance on files modified or introduced by ROCm. Checks presence, format, and year correctness. Use whenever reviewing a PR on the ROCm TransformerEngine fork, or when asked to audit copyright headers.
argument-hint: [base-branch] [--paths <glob>...]
allowed-tools: [Read, Glob, Grep, Bash]
---

# Copyright Header Check

Audits AMD copyright headers on files that ROCm has modified or introduced. The
existing `qa/L0_license/copyright_checker.py` only validates NVIDIA headers —
this skill is the AMD-side counterpart.

## Arguments

- `<base-branch>` — diff against this ref (default: `dev`).
- `--paths <glob>...` — restrict the check to specific paths. Without this,
  every file changed in the diff is checked.

## Scope detection

Determine the file set to audit:
```
base="${1:-dev}"
git fetch origin "$base" --quiet
git diff --name-status "origin/$base"...HEAD
```
- `A` = added — counts as **ROCm-introduced**.
- `M` / `R` = modified or renamed — counts as **ROCm-modified**.
- `D` = deleted — skip.

Skip files that the existing checker also skips (see `qa/L0_license/config.json`):
binaries, `3rdparty/`, `LICENSE`, `VERSION`, `.png`, `.ipynb`, `.json`, `.md`,
`.txt`, Dockerfiles, generated files. Also skip files outside these extensions:
`c, cpp, cu, h, cuh, hpp, hip, py, sh, cmake, yml, yaml, toml, rst, cfg`,
plus `CMakeLists.txt`. If a file has none of those, note it as "unchecked
filetype" and move on — do not flag.

## Header rules

Read the **first 15 lines** of each in-scope file (headers may sit below a
shebang and a coding declaration). The current year is `$(date +%Y)` — call it
`Y`. Use that value, do not hardcode it.

### Comment-style families
| Family | Comment marker | File types |
|---|---|---|
| hash | `#` | py, sh, cmake, yml, yaml, toml, cfg, `CMakeLists.txt` |
| c-block | `/*` … `*/` or `//` | c, cpp, cu, h, cuh, hpp, hip |
| rst | `..` (indented) | rst |

### AMD copyright line — required form
```
<comment> Copyright (c) <YEARS>, Advanced Micro Devices, Inc. All rights reserved.
```
where `<YEARS>` is either `Y` (single year) or `Yfirst-Y` (range ending in
the current year). Use a regex like:

```
Copyright \(c\) (\d{4})(-(\d{4}))?, Advanced Micro Devices, Inc\. All rights reserved\.
```

### NVIDIA copyright line — preserved form
```
<comment> Copyright (c) <YEARS>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
```

### File classification (in scope)

For each file, classify by its **post-change** content combined with its diff
status:

1. **ROCm-only file** — added by ROCm, no NVIDIA copyright present.
   Required: AMD copyright. NVIDIA copyright must NOT be added.
2. **Modified upstream file** — already existed with a NVIDIA copyright before
   this PR, ROCm is now changing it.
   Required: BOTH AMD and NVIDIA copyright lines, AMD line first.
3. **Already-mixed file** — already had both AMD and NVIDIA headers, ROCm is
   modifying it again.
   Required: both lines remain; AMD year extended to include `Y`.
4. **Cherry-picked from upstream** — added in this PR but contains a NVIDIA
   copyright (file pulled from upstream NVIDIA TE).
   Required: NVIDIA copyright preserved verbatim; AMD copyright added only if
   the cherry-pick included AMD-authored modifications.

To distinguish (1) from (4) on added files: check `git log` of the upstream
remote (`origin/upstream-main` or whatever upstream tracking branch exists —
fall back to checking whether the path exists on `origin/main`). If unsure,
flag as "ambiguous" and let the human decide.

To know whether a *modified* file had a NVIDIA copyright before the PR:
```
git show "origin/$base:$path" | head -15 | grep -F "NVIDIA CORPORATION"
```

## Year correctness

For every file in scope, the AMD copyright year must include the current year
`Y`:
- Single year `YYYY`: must equal `Y`.
- Range `Yfirst-Ylast`: `Ylast` must equal `Y`. `Yfirst` must be ≤ `Ylast` and
  ≥ the year the AMD copyright was first added (best-effort: `git log
  --diff-filter=A --follow --format=%ad --date=format:%Y -- $path | tail -1`).

For NVIDIA copyrights on modified files: the year MUST NOT have been changed
by this PR. Compare against `git show "origin/$base:$path"`. ROCm changes do
not extend NVIDIA's copyright window.

## Recommended-but-optional marker

Files in family (2) often carry the marker line:
```
<comment> This file was modified for portability to AMDGPU
```
Treat its absence as a **note**, not a failure. Its presence makes review
easier but is not legally required.

## What NOT to flag

- Headers in files outside the diff scope (don't audit the whole tree).
- Stylistic differences (spacing, ordering of unrelated lines).
- The `License for AMD contributions = MIT.` shorthand line — it's a valid
  alternative to `See LICENSE for license information.` for AMD-only files.
- Files in `exclude_copyright` from `qa/L0_license/config.json`.

## Output format

```
## Copyright Header Audit: <base>...HEAD

### Summary
| Status | Count |
|---|---:|
| OK | <n> |
| Missing AMD copyright | <n> |
| Stale AMD year | <n> |
| Altered NVIDIA copyright | <n> |
| Ambiguous (needs human review) | <n> |

### Findings

#### <path>:<line>
- **Classification:** <ROCm-only | Modified upstream | Already-mixed | Cherry-pick | Ambiguous>
- **Issue:** <one-line description>
- **Found:**   `<relevant line(s) from the file>`
- **Expected:** `<the line as it should be>`
- **Fix:** <minimal patch — usually a 1-3 line header replacement>

(repeat per finding; group by file)
```

If everything passes, output `All <n> files in scope have correct AMD
copyright headers.` and stop — no per-file noise.

## Notes for re-runs

When this skill is invoked from `/claude review` on the same PR a second time,
prior findings posted by `claude[bot]` will already exist as inline comments.
The orchestrator (the calling workflow) is responsible for deduplication; this
skill should always emit the full current set of findings and let the caller
decide what to post.
