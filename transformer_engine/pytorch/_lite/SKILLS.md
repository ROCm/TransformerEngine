# Transformer Engine Lite — Working Notes

Operational knowledge for engineers (and agents) modifying `tealite`. The
[README](README.md) is the feature/coverage reference; this file is the
"what we have learned" complement — invariants, gotchas, dead ends, and
measurement protocol that aren't visible from the code alone.

## Mental model

- **Lite is a `sys.modules` swap, not a fork.** `transformer_engine.pytorch`
  registers `_lite` as `transformer_engine_torch` at import time when
  `NVTE_LITE=1` (or the `LITE_BUILD` marker is present). Every `tex.<fn>` call
  in the rest of the codebase resolves into `_lite/`. **Implication:** you
  almost never need to change call sites in `module/`, `cpp_extensions/`, etc.
  — implementing the function in `_lite/` is enough.
- **Tiered fallback per subsystem:** AITER → bundled Triton → PyTorch-native.
  If you add a new path, preserve this order; PyTorch fallback must stay
  reachable when AITER is missing or rejects the inputs.
- **Quantizer drives the kernel, not the recipe enum.** The dispatch decisions
  in `gemm.py`, `quantize.py`, and `norms.py` branch on the *Quantizer class*
  (`Float8Quantizer`, `Float8CurrentScalingQuantizer`, `Float8BlockQuantizer`,
  `MXFP8Quantizer`) and on `_scale_inv.numel()` / `_scale_inv.shape` — never
  on a recipe string. When adding a new path, key off the same.

## Where to add code

| Need to | Do |
|---|---|
| Replace a `tex` C++ function in lite | Implement it in `_lite/<topic>.py`, export from `_lite/__init__.py` |
| Wrap a new AITER kernel | Add a thin dispatcher in the relevant `_lite/<topic>.py`; gate it on `_aiter_available()` from `aiter_utils.py` (lru_cached) |
| Cover a new compound module that already calls `tex.*` | **Do nothing in `_lite/`.** The tex hot-swap is sufficient. Verify with a smoke test. (Validated for `te.GroupedLinear` BF16 — 2026-04-28.) |
| Change a Quantizer's behavior in lite | Edit the dispatch in `_lite/quantize.py` and `_lite/norms.py`; do **not** modify the shared `Float8Tensor` class — the same class is used by full TE |
| Add a perf-sensitive elementwise path | Write or reuse a Triton kernel before adding PyTorch ops — fragmented PyTorch elementwise launches are the dominant remaining penalty (see "Performance baselines") |

The fused `LayerNormLinear` / `LayerNormMLP` are pure-Python `autograd.Function`
subclasses (`fused_layernorm_linear.py`, `fused_layernorm_mlp.py`) loaded
**lazily** from `__init__.py` to avoid circular import with the tex
registration. If you touch the `__init__.py` import order, run `TestImport`
and `TestLiteLayerNormLinear` to verify lazy-load still works.

## Numerical & dispatch hazards

1. **Scale shape selects the kernel family — not just a numeric value.**
   `torch._scaled_mm` routes per-tensor scalars to the F8NBS/F8B8NBS kernel
   family (covers same-dtype *and* mixed-dtype FP8); broadcasting the same
   scalar to `(M,1)` / `(1,N)` forces the rowwise family, which has no
   mixed-dtype coverage on current ROCm. Per-tensor → 0-dim scalar; per-row →
   `(M,1)` / `(1,N)`. Never broadcast scalar to rowwise shape. (Fixed in
   `5a660e9c`.)
2. **AITER `gemm_a8w8_CK` rejects mixed FP8 dtypes.** Both operands must
   share the same FP8 dtype. Standard FP8 training uses E4M3 × E5M2 for
   dgrad/wgrad — ~48% of backward GEMMs hit this. Route mixed-dtype to
   `torch._scaled_mm` (the default `pytorch` backend already does this).
3. **AITER fused RMSNorm+FP8 kernels write only `_data`, not the columnwise
   transpose buffer.** If `make_empty(columnwise_usage=True)` allocated it,
   the buffer is uninitialized — set `_transpose_invalid = True` after
   filling `_data`, or downstream `update_usage(columnwise_usage=True)`
   trusts stale bytes.
4. **Per-row scales on the reduction axis can't use per-token GEMM.** wgrad
   under `Float8CurrentScaling` falls back to per-tensor `gemm_a8w8_CK`. This
   is correct (reduction across tokens averages out outliers); don't try to
   "fix" it without changing the operand layout.
5. **`Float8CurrentScalingQuantizer` per-row is strictly better than per-tensor**
   in lite — the AITER `dynamic_per_token_quant_fp8_i8` path intercepts
   *before* per-tensor quantize. Don't restore the per-tensor branch as a
   "default" — it loses precision and adds 2 HBM round-trips (see README §
   FP8 Training).
6. **AITER `aiter/ops/triton/fused_fp8_quant.py` line ~83 has a bug:**
   `out1_col_stride = out2.stride(1)` should be `out1.stride(1)` — crashes
   when `output_unquantized_inp1=True` and `inp2=None`. Reported upstream;
   may already be fixed when you read this.
7. **`gemm_a8w8_CK` falls back to default config for untuned shapes.** The
   warning *"shape ... not found tuned config in a8w8_tuned_gemm.csv, will
   use default config!"* means CK is running with `splitK=0` — no
   exception, just slow. Non-round M (e.g. 8184 = 2046×4) is the usual
   miss. Either run AITER's a8w8 tuner against your shape set or prefer the
   `pytorch` GEMM backend (default).

## Multi-node planning

- **TE-lite has no TP/SP** — `fused_layernorm_linear.py:456` and
  `fused_layernorm_mlp.py:505` hardcode `self.tp_size = 1`. Kwargs are
  accepted for API compatibility; setting `tp_size > 1` blows up downstream
  in Megatron's QKV reshape. Multi-node plans for tealite must be
  FSDP/HSDP-shaped.
- **Comm-overlap (AG/RS + GEMM) is unimplemented;** `_lite/comm.py` raises.
- **Expert parallelism works via `_lite/mori_ep.py`** but is a standalone
  primitive — call its dispatch/combine APIs explicitly; there is no
  integration into a TE `MoE` module.
- **MoE BF16 GroupedLinear works for free** through the tex hot-swap →
  `_lite/grouped_gemm.py` (AITER Triton GMM). FP8 grouped GEMM is NYI:
  AITER's generic GMM is BF16/FP16 only, and FP8 expert compute lives
  separately in `aiter.fused_moe`. Run BF16-only for MoE until that path
  lands. (`TestGroupedLinear::test_fp8_forward` is xfail-strict.)

## Performance baselines (LLaMA-3-8B, 8×MI300X, seq=2048, RECOMPUTE=0)

As of 2026-05-01, lite ≈ full at the same TE commit:

| Mode | ms/iter | tok/GPU/s |
|---|---:|---:|
| full @ same commit | 1712.5 | 9567 |
| lite | 1712.0 | 9570 |

**The earlier "lite is 5–10% slower" gap was a stale-build artifact** —
the supposedly-faster full was an older commit (`f141f34b`, 1599 ms). There
is a confirmed ~7% regression between `f141f34b` and HEAD that is *not* a
lite issue; bisect harness in `/root/bisect/`.

The dominant remaining lite-vs-full kernel-time penalty (when one exists) is
**Triton-fragmented elementwise/copy ops** — top offenders are
`multi_tensor_apply_kernel`, fused SwiGLU+bias kernels, FSDP shape-shuffle
copies. GEMM, FMHA, RCCL, and fused norm+quant are all at parity or better.

### How to re-profile

```bash
NVTE_LITE_GEMM_BACKEND=pytorch NVTE_LITE_DIAG=1 <your-launcher>
```

Sanity-check the diag counters in stdout:

- `pytorch_scaled_mm_ok` ≈ 2/3 of FP8 GEMMs (fwd + dgrad)
- `pytorch_aiter_fallback_ok` ≈ 1/3 (wgrad — K=8184 hits `k_not_div16`)
- `pytorch_dequant_matmul` should be **zero**. Any hits = both `_scaled_mm`
  and AITER rejected; that's a 100–1000× slowdown.

For Megatron runs, pass `--attention-backend fused` to force the AITER AOT
`fmha_v3_fwd/bwd` path. Without it, Megatron defaults to `auto`, sets
`NVTE_FLASH_ATTN=1`, and routes through ROCm `flash_attn` 2.8.3 which
bypasses `_lite/attention.py` entirely. Don't try to set `NVTE_FLASH_ATTN=0`
directly — Megatron asserts on it; use the CLI flag.

### Apples-apples discipline

- Always check **loss-AR async fix** symmetry between full and lite
  containers before quoting a %; one side missing the fix is worth ~70 ms
  (3.5%) and can flip the apparent winner.
- Always compare at the **same TE commit** — there is a real ~7% regression
  in TE itself between Jan and May 2026.
- `RECOMPUTE=0`, `seq_len=2048` is the current standard config for new
  measurements (CK earlier needed `seq_len=4098` to hit a tuned config; that
  workaround is irrelevant on the `pytorch` backend).

## Debug tooling

| Flag | What it does |
|---|---|
| `NVTE_LITE_DIAG=1` | One-shot prints from `_lite/{gemm,norms,attention,quantize}.py` and `module/base.py`; per-bucket counters (`[LITE-GEMM]`, `[LITE-NORM]`, `[LITE-ATTN]`, `[LITE-QUANT]`, `[LITE-NONCONTIG]`, `[LITE-SCALED-MM-FAIL]`, `[LITE-GEMM-CK-FAIL]`). Zero overhead when off. |
| `NVTE_LITE_AMAX_FUSED=0` | Falls back from the Triton multi-tensor-apply amax/scale kernel to the per-group Python loop (`_lite/quantize.py`, ~14 kernel launches × N groups). For A/B against the fused path. |
| `NVTE_LITE_SKIP_FP8_DGRAD_FOR_NORM=1` | Opt-in: skip the BF16→FP8 cast on dgrad output when only the norm backward consumes it. **Shelved**: mechanically perfect (1484 casts eliminated) but wall-time within noise — kept env-gated. See open-question note in dgrad-skip memory if reviving. |
| `NVTE_CONTIG_DIAG=1` `NVTE_CONTIG_DIAG_DUMP_STEP=N` | Counts and times every `prepare_forward` `.contiguous()` materialize per `(module, shape, stride, caller)`. Diff full-vs-lite stdout to see where lite has extra materializes. Phase 1 result (2026-05-01): same single site, same 64 calls/step, **lite 3× faster** at the materialize itself — drops the AITER 3D-strided-input kernel patch from the queue. |

## Test conventions

`tests/pytorch/test_lite.py` sets `NVTE_LITE=1` **before** importing TE — so
the C++ extension never loads, even on a full build. Don't reorder those
lines.

- New kernels / dispatch paths should land with a regression test in
  `test_lite.py`, in the class closest to the feature (new GEMM kernel →
  `TestGemm`; new recipe-level feature → `TestRecipeIntegration`).
- FP8-recipe tests should parametrize via `_RECIPES_FWD_BWD` /
  `_RECIPES_FWD` so they skip cleanly on unsupported hardware.
- The standard numerical check is **FP8-vs-bf16 cosine similarity** ≥ 0.9
  for single modules, ≥ 0.75 for `TransformerLayer`. This catches silent
  wrong-dispatch and scale-broadcast bugs that exact-tolerance checks miss.

### Monkeypatch gotcha

`_lite.quantize` is shadowed in the package namespace by the `quantize`
function re-exported from `_lite/__init__.py`. To monkeypatch a module-level
kernel attr (e.g. `_aiter_dynamic_per_token_quant`) you must reach the
*module*, not the function — use:

```python
import sys
mod = sys.modules["transformer_engine.pytorch._lite.quantize"]
monkeypatch.setattr(mod, "_aiter_dynamic_per_token_quant", spy)
```

`import transformer_engine.pytorch._lite.quantize as q` resolves `q` to the
function via attribute lookup, **not** the module — patching `q.foo` won't
affect dispatch. `_lite.norms` is not shadowed; `import as` works there.

## Discipline

- **Wait for profile data before optimizing.** Code-inspection guesses at
  hotspots miss the real bottleneck often enough that this is the safer
  default. When something is reported slow, ask for / wait on the top-N
  kernels with self-CUDA-time and call counts before writing code.
- **Verify "genuine cost" claims with measurement, not deduction.** The
  `prepare_forward` materialize was assumed to be a lite penalty for two
  weeks; the contig-diag harness showed lite is 3× *faster* at it. Build the
  diff harness before writing the patch.
- **A/B every speculative perf change.** Multiple bypass attempts at
  `prepare_forward` (`NVTE_LITE_SKIP_NONCONTIG`, `_to_bshd` strided view,
  `ROPE_FUSION=0`) all looked like wins on paper and all regressed wall
  time. Net-positive must be observed, not predicted.

## Dead ends — don't retry

- **Pad-M to next power of 2** (reverted `eac04dd8` → `ccb1f30b`) — inflated
  weight N dims by 12.5%. Current div-by-16 pad (`3ed9d8ae`, only 8184→8192)
  is the correct version.
- **Tuning AITER CK CSV for forward shapes** — irrelevant since fwd is on
  `_scaled_mm` (hipBLASLt) under the default `pytorch` backend.
- **Dequant + matmul as fallback for FP8** — catastrophically slow (~206
  s/iter). Always fall through to AITER before dequant+matmul (`e8272800`).
- **Broadcasting per-tensor scalar scales to rowwise shapes** — see hazard
  #1 above.
- **`NVTE_LITE_SKIP_NONCONTIG`-style env-var bypass of `prepare_forward`'s
  `.contiguous()`** — downstream FP8 quantize + GEMM 3D→2D reshape paths
  re-materialize anyway, more expensively. Reverted in `1bc68c3f`.
- **`ROPE_FUSION=0`** to avoid the BSHD round-trip — `_to_bshd` then makes 3
  input copies (q+k+v at SBHD→BSHD) instead of 1 output copy. +87 ms.
- **`_to_bshd` strided-view (`e4a05c50`)** — unreachable with current
  Megatron BSHD path. Reverted in `c62e9771`.
- **`NVTE_LITE_SKIP_FP8_DGRAD_FOR_NORM=1` as default** — works mechanically,
  but the standalone amax-only reduction added to preserve DelayedScaling
  amax history is memory-bound over the same BF16 tensor and roughly cancels
  the savings. Keep env-gated; don't flip default. Open question: an
  unexplained +8 `pytorch_scaled_mm_ok` calls under skip=1 — worth chasing
  if reviving.

## Untracked patches & TODOs (as of 2026-05-06)

- AITER-side defensive K-innermost asserts in
  `/root/WORK/aiter/aiter/ops/triton/gemm/basic/gemm_a8w8{,_per_token_scale,_blockscale}.py`
  — drafted, **uncommitted**. Send upstream when convenient.
- `[LITE-*]` diag counter print sites — Jason wants stripped before merging
  to `dev`. All gated behind `NVTE_LITE_DIAG=1` so they're harmless in
  production, but noise in the source.
- AITER `fused_fp8_quant.py:83` upstream bug (`out2.stride(1)` →
  `out1.stride(1)`) — report or send a one-line PR.
