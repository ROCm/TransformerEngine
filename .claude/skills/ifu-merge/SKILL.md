---
name: ifu-merge
description: >
  Guide for performing IFU (Integrate From Upstream) merges on the TransformerEngine ROCm fork.
  This skill walks through resolving merge conflicts between upstream NVIDIA TransformerEngine
  and the AMD ROCm fork, then fixing build and runtime errors systematically. Use this skill
  whenever the user mentions IFU, merge upstream, merge NVIDIA, upstream merge,
  IFU merge, internal feature update, or is working on integrating new NVIDIA TE changes into
  the ROCm fork — even if they don't use the term "IFU" explicitly.
---

# TransformerEngine ROCm — IFU Merge Guide

An IFU (Integrate From Upstream) merges upstream NVIDIA TransformerEngine commits into the AMD ROCm fork. The merge is often committed with conflict markers still in place, then conflicts are resolved in follow-up commits.

This guide captures generalized strategies for resolving these merges.

---

## Strategy 1: Conflict Resolution Mental Model

Every conflict falls into one of these categories. Recognizing the category tells you how to resolve it:

| Category | Recognition | Resolution |
|---|---|---|
| **API Refactor** | Upstream restructured function signatures, parameter passing, or return values | Take upstream's new API structure; integrate ROCm-specific fields into the new pattern |
| **Feature Addition** | Both sides added new parameters, attributes, docstrings, or code paths | Keep both — upstream features and ROCm features are independent and coexist |
| **Guard Removal** | Upstream removed `#ifdef`/`#ifndef` platform guards (they don't need them) | Re-add the guards — they protect ROCm-specific code paths |
| **Feature Support Divergence** | Upstream adds code for features not yet supported on ROCm | Guard with `#ifndef USE_ROCM`; verify current ROCm support status first — features may become supported over time |
| **Convention Change** | Upstream changed a data format, tensor shape, or API contract (e.g., different tensor dimensions, new enum values, reordered arguments) | Adopt the new convention everywhere — then find and update every downstream consumer, including test code. The compiler won't catch these; they cause silent wrong results or cryptic runtime errors |
| **Copyright/Whitespace** | Trivial formatting or copyright header differences | Keep ROCm (AMD) copyright headers; take upstream for whitespace |

**Key principle**: When upstream refactors a pattern (e.g., bundling arguments into a tuple), always adopt the new pattern and adapt ROCm-specific fields into it. Fighting the upstream structure creates maintenance debt.

**Watch for paired sites**: If upstream refactors a function signature, there are always at least TWO places to update — where values are packed into the new structure and where they are unpacked. Missing one causes silent bugs.

---

## Strategy 2: Understanding the Hipify Pipeline

PyTorch's hipify tool automatically converts CUDA source files to HIP. Understanding what it does and doesn't convert is essential.

**What hipify converts:**
- `#include` directives (e.g., `cuda.h` -> `hip/hip_runtime.h`)
- File extensions in generated output (`.cu` -> `.hip`, `.cpp` -> `_hip.cpp`)

**What hipify does NOT convert:**
- C++ namespace references (e.g., `c10::cuda::CUDAGuard` stays as-is)
- Semantic differences between CUDA and HIP APIs (e.g., warp size, mask types)
- ROCm-specific device behavior (e.g., tensor device masquerading)

**What hipify preserves faithfully:**
- Preprocessor guards (`#ifndef USE_ROCM`, `#ifdef __HIP_PLATFORM_AMD__`). This means adding guards to source `.cpp` files will propagate into the generated `_hip.cpp` output. Use this to exclude CUDA-only code paths from ROCm builds.

**Rules that follow:**
- Never edit `*_hip.cpp` or `.hip` files — they are regenerated from source files
- Always edit the source `.cpp`/`.cu` files — hipify will process them
- When you see a build error in a `_hip.cpp` file, fix the corresponding source file
- Use conditional compilation to handle namespace/API differences that hipify can't resolve

**Platform guard conventions by code layer:**

| Layer | Guard Macro | Example |
|---|---|---|
| PyTorch CSRC (`.cpp` source files) | `#ifdef USE_ROCM` / `#ifndef USE_ROCM` | DeviceGuard, scale swizzling |
| Common layer (`.cu` files that get hipified) | `#ifdef __HIP_PLATFORM_AMD__` | Warp masks, kernel dispatch |
| Python code | `IS_HIP_EXTENSION` (from `torch.utils.cpp_extension`) | Workspace sizing, feature flags |

Using the wrong guard macro for a given layer is a common source of subtle bugs — the code compiles but the guard doesn't activate correctly.

---

## Strategy 3: Auditing Non-Conflicting Changes

**This is the single most important strategy and the one most often skipped.** Conflict markers are visible and localized. The real danger lies in changes that merge cleanly — upstream modifications that `git merge` silently accepts because the ROCm fork didn't touch those exact lines. A removed platform guard, a changed convention, a new CUDA-only code path — these slip through without any marker.

**Five patterns to watch for:**

1. **Guard Removal**: Upstream deletes platform guards because they don't need them. If ROCm didn't modify those exact lines, the merge silently removes the guards.

2. **New CUDA-only Code**: Upstream adds new functions or code paths using CUDA-only APIs. These compile fine on CUDA but fail on ROCm.

3. **New Upstream Files**: Upstream introduces entirely new files (e.g., `swizzle.cpp`) that have no ROCm guards at all. These files get hipified and compiled on ROCm, potentially referencing CUDA-only symbols. Check every new file for functions that should be guarded.

4. **Refactored Utilities**: Upstream moves or rewrites utility functions that had ROCm-specific behavior. The new version may lack the ROCm-specific handling, and the fork's old copy becomes stale.

5. **Convention Changes**: Upstream changes a data format, tensor shape, or API contract without any code conflict. Every downstream consumer of that convention must be updated manually — the compiler won't catch these.

**How to systematically audit:**

```bash
# Get the merge parents
git cat-file -p <merge-commit>

# See ALL upstream changes (not just conflicting ones)
git diff <rocm-parent>..<upstream-parent> --stat

# Check for removed guards
git diff <rocm-parent>..<upstream-parent> -- <file> | grep -E "^-.*(__HIP_PLATFORM_AMD__|USE_ROCM|IS_HIP_EXTENSION)"

# Find entirely new files from upstream
git diff <rocm-parent>..<upstream-parent> --diff-filter=A --name-only

# Check CMakeLists.txt for source list changes
git diff <rocm-parent>..<upstream-parent> -- "*/CMakeLists.txt"
```

**What to look for in each changed file:**
- Removed platform guards (`USE_ROCM`, `__HIP_PLATFORM_AMD__`, `IS_HIP_EXTENSION`)
- New CUDA API usage (e.g., `CUDAGuard`, cuBLAS/cuDNN calls) without ROCm equivalents
- Refactored utility functions that previously had ROCm-specific behavior
- Changes to CMakeLists.txt source file lists — files moved between `if(USE_CUDA)` and shared lists, or vice versa
- Changes to workspace allocation, scale computation, or hardware-specific sizing
- Convention changes in tensor shapes, data formats, or function contracts

**What to look for in CMakeLists.txt specifically:**
- Files that should be in `if(USE_CUDA)` but ended up in the shared source list (causes ROCm compilation of CUDA-only code)
- Files that were in `if(USE_CUDA)` on `dev` but were moved during conflict resolution
- New source files from upstream that need to be placed in the correct list

---

## Strategy 4: The Guard Dependency Contract

Platform guards form a dependency graph. If you guard an implementation, every unguarded call site becomes a dangling reference. If you remove a guard, the implementation must work on both platforms.

**The rule**: guarded implementations require guarded call sites, and unguarded call sites require unguarded implementations.

When you fix an issue by adding a guard (e.g., moving a `.cu` file into `if(USE_CUDA)`), trace all references to the symbols it defines:
1. Search for every call site of the guarded function
2. Verify each call site is also guarded, OR has an alternative code path for ROCm
3. Check header files — the declaration may still be visible even if the implementation is guarded, causing linker errors instead of compile errors
4. Check pybind registrations — a Python-exposed function that calls a guarded symbol will fail at import time

A common pattern: the implementation is guarded in one file, a second call site is properly guarded, but a third call site in a different file (especially a new upstream file) is not. The fix for the implementation creates the linker error at the third site.

---

## Strategy 5: Iterative Build-Test-Fix

After conflict resolution and auditing, use an iterative loop:

1. **Build** — compile the project
2. **Categorize the error** — missing include? undeclared identifier? duplicate symbol? preprocessor guard issue? API incompatibility? undefined symbol at link time?
3. **Find the root cause, not the symptom** — when a CUDA-only file fails to compile on ROCm due to a type mismatch, the surface fix is to change the type. But the root cause may be that the file shouldn't compile on ROCm at all. Always ask: "Is the right fix to make this code work here, or to prevent it from running here?"
4. **Apply the appropriate fix pattern** — update the source code with the appropriate fix, guarding as needed
5. **Trace the dependency chain** — after applying a fix, ask "what depends on what I just changed?" and verify both directions
6. **Rebuild** and repeat until clean

**After a clean build, test in order of increasing scope:**
1. Basic module import — catches missing symbols, broken dynamic linking
2. Core operations (GEMM, normalization) — catches API mismatches, incorrect workspace sizing
3. Higher-level tests (attention, transformer layers) — catches integration issues
4. Full test suites — catches edge cases

Each phase catches a different class of error, and errors from earlier phases are much cheaper to fix.

**Runtime errors require different investigation than build errors.** A common pattern: upstream refactors a Python function and the new version works on CUDA but uses different hardware-specific parameters (workspace sizes, alignment requirements, feature flags). These only manifest at runtime with specific configurations (dtype, dimensions, GPU architecture).

**When investigating "works on dev, fails here"**: the critical question isn't "what's different?" — it's "why did it work on dev at all?" A test might pass not because the feature is supported, but because a chain of incidental conditions causes it to be skipped. Understanding the mechanism of "working" is essential to understanding when it will break.

---

## Strategy 6: Stale Fork-Specific Code

Over successive IFUs, the ROCm fork accumulates its own implementations of functions that upstream later refactors or relocates. When upstream moves a function from file A to a new file B, the merge brings in file B but doesn't remove the fork's copy in file A. Now you have two definitions in the same namespace — a potential duplicate symbol error on CUDA, and confusion about which version is canonical.

**After each IFU:**
1. Check if the ROCm fork has files that don't exist on upstream (e.g., `util.cpp` with `#ifndef USE_ROCM`)
2. For each fork-specific file, check if upstream added equivalent functionality in a different file
3. If upstream's new file supersedes the fork's copy, verify the fork's copy is either removed or properly isolated

---

## Strategy 7: ROCm vs CUDA Feature Parity

Not all CUDA features are available on ROCm. The set of supported features evolves over time as AMD adds capabilities. When resolving an IFU:

1. **Check current support** before deciding how to guard new upstream code
2. **Look at existing codebase patterns** — if similar code is already guarded elsewhere, follow the same pattern
3. **When in doubt, guard conservatively** and add a TODO comment indicating the feature may become available
4. **Ask the developer** if you're unsure about the current support status of a specific feature

---

## Workflow

### Step 1: Setup

```bash
git cat-file -p <merge-commit>  # Identify parent1 (ROCm) and parent2 (upstream)
grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" . \
  --include="*.py" --include="*.cpp" --include="*.cu" \
  --include="*.h" --include="*.cuh"  # Count remaining conflicts
```

### Step 2: Resolve Conflict Markers

Apply Strategy 1 (conflict resolution mental model) to each file. Verify zero markers remain.

### Step 3: Audit Non-Conflicting Changes

Apply Strategy 3 — audit ALL upstream changes, not just conflicting ones. Check new files, removed guards, convention changes, and CMakeLists.txt source list placement. This step is most often skipped and most often the source of post-merge failures.

### Step 4: Build and Fix Iteratively

Apply Strategy 5. Python syntax check first (`python -m py_compile`), then C++ build, then runtime tests. For every fix, apply Strategy 4 — trace the guard dependency chain to completion before rebuilding.

### Step 5: Run Tests

Test in order of increasing scope per Strategy 5. Investigate runtime failures by checking whether upstream changed hardware-specific parameters, utility functions, or data format conventions.

### Step 6: Clean Up Stale Code

Apply Strategy 6. Check fork-specific files for functions that upstream has superseded in new files.

---

## Hard Rules

- **Never edit `*_hip.cpp` files** — they are auto-generated by hipify
- **Always edit source `.cpp`/`.cu` files** — hipify will process them
- **Use the correct platform guard for each layer** (see Strategy 2)
- **Guard unsupported features** — check current ROCm support status; guard conservatively if unsure
- **Audit all upstream changes** — not just the ones that caused conflicts
- **Trace the guard dependency chain** — guarding an implementation requires guarding every call site (see Strategy 4)
- **Test failures are informational, not directional** — trace backward from the failure to find the violated invariant; resist the urge to fix the test
