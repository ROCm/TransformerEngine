# base.py IS_HIP_EXTENSION census (plugin plan S5.3, 2026-09-01)

Per-site verdict for the file with the largest guard bucket. Rule: a guard becomes a **named
capability** only when it encodes what the platform *supports* (with a reason a user can read);
guards that are imports, dispatch plumbing, or ROCm-side policy stay guards — pretending they are
capabilities would relabel, not shrink, the divergence.

| Line (pre-conversion) | Site | Verdict |
|---|---|---|
| 59 | fork-only imports (FSDPAGTensor, te_quantize_triton) | plumbing — import guard stays |
| 335 | `_rocm_layers` default overlap-method lists | policy/config — stays (defaults, not support) |
| 375 | `external_gemm_to_overlap` | capability-shaped — see verdict below |
| 398 | `set_sm_margin` default | policy (performance default) — stays |
| 563 | duplicate-buffer cleanup in UB cfg | ROCm-side bugfix behavior — stays |
| 570 | bulk overlap `NotImplementedError` | capability-shaped — see verdict below |
| 678 | `tex.reset_fused_ag_gemm_cache()` | plumbing — ROCm-only API call |
| 789 | MXFP8 dims %128 check relaxed on ROCm | capability-shaped — see verdict below |
| 1549 | `SKIP_FP8_REDUCTION_FOR_FSDP2` | ROCm fsdp2 policy — stays (S5.3 fsdp2 family later) |
| 1862 | transpose-cache columnwise off | ROCm fsdp2/cache policy — stays |
| 1878 | `FSDPAGTensor` wrap | ROCm fsdp2 mechanism — stays |
| 2069 | triton vs tex quantize dispatch | kernel dispatch plumbing — stays |

**Measured verdict (2026-09-01): local conversion is counter-indicated.** The three
capability-shaped sites were converted and measured with `classify_hunks.py`: guard lines fell
only 719 -> 713 while unmarked rose 1236 -> 1255 and PT-002 GREW +287 -> +296 added lines - the
provider-query lines are still fork divergence inside a patched shared file, just less legible
to the classifier. Exactly the relabeling trap the "unmarked must not go up" rule exists to
catch; the conversions were reverted. Guards in shared files shrink only when UPSTREAM accepts
capability queries (#3113 / HDR-B2) so the guard itself disappears from the patch. The fsdp2
family (1549/1862/1878) remains one coherent future capability once that lands.
