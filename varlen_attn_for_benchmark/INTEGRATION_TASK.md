# Varlen Attention (CK Team Kernels) – Integration Task

Consolidated planning and implementation guide for integrating the CK team’s unfused variable-length attention HIP kernels into Transformer Engine (TE) for the SciForum cross-attention use case.

---

## 1. Background & Problem

- **Use case:** SciForum uses a special **cross-attention** where:
  - **Q sequence length is always 1** (one query token per batch item).
  - **KV sequence length is small and variable** (2–16).
  - **Batch size is very large** (~30K).
  - **Layout:** THD (batch, seq, heads, head_dim), **BF16**, variable-length (JAX-style packing).

- **Current issue:** Flash/CK fused attention is tuned for large `seq_q × seq_kv`. With `seq_q=1` and tiny `seq_kv`, it does unnecessary work and is **~20× slower** than JAX unfused attention (see `varlen_attn/sciforium.png` or `SciforiumCrossAttn.csv`).

- **Customer requirement:** Keep the **exact same input/output format** (no conversion to square masks). Implement an unfused path that matches JAX unfused performance while reusing the existing TE API.

---

## 2. Solution: CK Team’s HIP Kernels

The CK team provided **unfused** HIP kernels in `varlen_attn/`:

| File | Role |
|------|------|
| `varlen_attn/attn_fwd.cpp` | Forward: 3 kernels (scores → mask+softmax → output) + host launcher + CPU ref + test |
| `varlen_attn/attn_bwd.cpp` | Backward: 4 kernels (grad_V → grad_attn → softmax bwd → grad_Q/grad_K) + host launcher + CPU ref + test |
| `varlen_attn/attn_py_ref.py` | Python/NumPy reference and optional PyTorch comparison |

**Constraints:** `seq_q == 1`, `max_seq_kv <= 16` (compile-time capacity; runtime per-batch length is variable 2–16).

---

## 3. Integration Design Choices

- **Where to plug in:** **CK fused attention** under **fused_attn_rocm** (`transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp`).
- **Recommendation (from task):** Plug into the **existing CK flow**; treat this as a **specialized path** when `max_seq_q == 1` and `max_seq_kv <= 16` (variable-length mode only).

---

## 4. Runtime Dispatch: Why `get_runtime_max_seqlen` Is Needed

- In variable-length (THD/packed) mode, **max sequence lengths are not passed as host scalars**; they are encoded in **device** cumulative arrays (`cu_seqlens_kv`, `cu_seqlens_kv_padded`, and similarly for Q).
- To decide whether to call the **specialized varlen kernel**, TE must know at runtime:
  - `max_seqlen_q` (over all batches)
  - `max_seqlen_kv` (over all batches)
- **Approach:** Use the existing utility that does a small device kernel + **host sync** to compute the max segment length from the device arrays, then branch on the host.

**Code pointer – get runtime max sequence length:**

- **Implementation:** `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_utils.cpp` (lines 66–99)
  - `get_runtime_max_seqlen_kernel`: device kernel, `atomicMax` of `(cu_seqlen_padded[tid+1] - cu_seqlen_padded[tid])` (or non-padded variant).
  - `get_runtime_max_seqlen`: launches kernel, `hipMemcpyAsync` (device→host), `hipStreamSynchronize`, returns `uint64_t`.
- **Declaration:** `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_utils.hpp` – `get_runtime_max_seqlen(b, cu_seqlen_ptr, cu_seqlen_padded_ptr, workspace, stream)`.

**Usage in integration:**

1. Before choosing the attention path, call `get_runtime_max_seqlen` for Q and for KV (with a small workspace, e.g. one `uint64_t`).
2. If `max_seqlen_q == 1` and `max_seqlen_kv <= 16` (and layout/mode match), **dispatch to the varlen (CK team) launcher**; otherwise use the normal CK fused attention path.

---

## 5. Softmax LSE vs Attention Weights (Buffer “Hack”)

- **Normal CK fused attention:** Forward writes **softmax LSE** (log-sum-exp) to an auxiliary buffer; backward reads it. Shape is effectively `[b, h, max_seqlen_q]` (or packed equivalent).
- **Varlen kernels:** There is **no LSE**; they produce **attention weights** (softmax output). Shape: `[batch, head_num, seq_q, max_seq_kv]` = `[b, h, 1, 16]` in our case.
- **Task suggestion:** Reuse the **same auxiliary buffer** that TE uses for “softmax LSE” to store **attention weights** for the varlen path. So:
  - **Forward (varlen):** Write attention weights (output of kernel 2, input of kernel 3) into that buffer. No LSE.
  - **Backward (varlen):** Read “softmax stats” as **attention weights** (and optionally same buffer for workspace for grad_attn/grad_scores).
- **Code pointers:**
  - Forward: attention weights are the **output of** `apply_mask_and_softmax_kernel` and **input to** `compute_output_kernel` – see `varlen_attn/attn_fwd.cpp` (e.g. around 431, 440).
  - TE CK forward: auxiliary tensor `output_S` (softmax LSE) – `fused_attn_ck.cpp` around 1590–1618 (Aux_CTX_Tensors, `output_S` shape for ragged: `{max_tokens_q, h_q, 1}` or `{b, h_q, max_seqlen_q, 1}`).
  - TE CK backward: `output_S` is passed in as input – `fused_attn_ck_bwd_kvpacked` (e.g. around 1689–1690); in varlen path this will hold attention weights.
- **Sizing:** For varlen, attention weights size = `b * h * 1 * max_seq_kv * sizeof(T)` (e.g. `b * h * 16 * 2` for BF16). Task says storage is acceptable for this config.

---

## 6. Workspace and Buffer Allocation

- **Varlen workspace size (from CK kernels):**
  - Forward: `AttnForwardKernelLauncher::calc_workspace_size()` → `b * head_num * seq_q * max_seq_kv * sizeof(T)` (see `varlen_attn/attn_fwd.cpp` ~385–393).
  - Backward: same formula – `AttnBackwardKernelLauncher::calc_workspace_size()` in `varlen_attn/attn_bwd.cpp` ~377–384.
- **Integration:** When dispatching to the varlen path, ensure:
  - **Workspace** is at least this size (use the same formula).
  - **Aux buffer** (softmax LSE slot) is used for attention weights as above; its size must be at least `b * h * 1 * max_seq_kv * sizeof(T)` for the varlen path.
- **TE today:** Workspace size is often computed inside `fused_attn_ck_fwd_impl` / `fused_attn_ck_bwd_impl`. You will need to:
  - Allocate based on which path will run (after `get_runtime_max_seqlen` tells you).

---

## 7. API Contract: Varlen Forward

**Host API (conceptual):** `AttnForwardKernelLauncher<T, Config>::run_attn_fwd_kernel(...)`

- **Inputs (device pointers):**
  - `Q`: `[batch, seq_q, head_num, head_dim]` (seq_q = 1)
  - `K`, `V`: `[total_padded_seq_kv, head_num, head_dim]` (use `cu_seqlens_kv_padded` for per-batch offset)
  - `dropout_mask`: optional; shape `[batch, head_num, seq_q, max_seq_kv]`
  - `dropout_p`, `sqr_dk_scale` (float)
  - `cu_seqlens_kv`, `cu_seqlens_kv_padded`: `[batch+1]`, int
- **Outputs:**
  - `O`: `[batch, seq_q, head_num, head_dim]`
  - **Attention weights** must be written to the buffer that TE will pass to backward (reuse softmax LSE buffer); that buffer is the **workspace** used inside the launcher for scores then attn weights – so you must **copy** the workspace (or the attn-weights part) into the auxiliary tensor before returning, or wire the auxiliary tensor as the workspace for the varlen path.

**Code pointers:**

- Launcher: `varlen_attn/attn_fwd.cpp` ~383–439 (`AttnForwardKernelLauncher::run_attn_fwd_kernel`).
- Step 1: `compute_scores_kernel` (~64–191).
- Step 2: `apply_mask_and_softmax_kernel` (~193–298); **output** is attention weights in `workspace`.
- Step 3: `compute_output_kernel` (~300–381).

---

## 8. API Contract: Varlen Backward

**Host API:** `AttnBackwardKernelLauncher<T, Config>::run_attn_bwd_kernel(...)`

- **Inputs:** `Q`, `K`, `V`, `grad_O`, **`attn_weights`** (from forward, same buffer as “softmax LSE”), `dropout_mask`, `dropout_p`, `sqr_dk_scale`, `cu_seqlens_kv`, `cu_seqlens_kv_padded`.
- **Outputs:** `grad_Q`, `grad_K`, `grad_V` (same shapes as Q, K, V).
- **Workspace:** Same size as forward; used for grad_attn then grad_scores.

**Code pointers:**

- Launcher: `varlen_attn/attn_bwd.cpp` ~372–419 (`AttnBackwardKernelLauncher::run_attn_bwd_kernel`).
- Step 1: `compute_grad_v_kernel` (~62–154).
- Step 2: `compute_grad_attn_kernel` (~156–290).
- Step 3: `softmax_backward_kernel` (~292–378).
- Step 4: `compute_grad_qk_kernel` (~380–369).

---

## 9. Compile-Time vs Runtime: max_seq_kv

- The varlen kernel is **templated** with `MAX_SEQ_KV = 16`: one kernel instance that can handle **any** run where the **runtime** max KV length over all batches is **≤ 16**.
- **Runtime:** Use `get_runtime_max_seqlen` to get the actual max KV length (e.g. 10). If it’s ≤ 16 (and max Q == 1), call the varlen launcher.
- **Per-batch:** Inside the kernel, actual length comes from `cu_seqlens_kv[b+1]-cu_seqlens_kv[b]`; only indices `0..seq_kv-1` are used; the rest are padded/masked. So one kernel with capacity 16 serves all cases 2–16.

---

## 10. Where to Hook in TE

- **Entry points:** Fused attention is invoked from `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`:
  - Forward: `nvte_fused_attn_fwd_kvpacked` → `fused_attn_ck_fwd_kvpacked` when backend is CK (around 563–577).
  - Backward: `nvte_fused_attn_bwd_kvpacked` → `fused_attn_ck_bwd_kvpacked` (around 656–675).
- **Place to add varlen branch:** Inside `fused_attn_ck_fwd_kvpacked` and `fused_attn_ck_bwd_kvpacked` in `transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp`:
  1. After extracting pointers and shapes (Q, KV, cu_seqlens, etc.), call `get_runtime_max_seqlen` for Q and for KV (need a small workspace for the atomic result).
  2. If `max_seqlen_q == 1 && max_seqlen_kv <= 16` (and layout is THD / variable-length, and any other constraints you enforce), call the **varlen** forward/backward implementation instead of `fused_attn_ck_fwd_impl` / `fused_attn_ck_bwd_impl`.
  3. Ensure workspace and auxiliary buffer sizes are sufficient for the varlen path (see sections 5 and 6).

**Relevant TE files:**

- `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`: dispatch to CK (563–577 fwd, 656–675 bwd).
- `transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp`: `fused_attn_ck_fwd_kvpacked` (1541), `fused_attn_ck_bwd_kvpacked` (1683); Aux_CTX_Tensors / output_S (softmax LSE) around 1590–1618 (fwd), 1689–1690 (bwd).
- `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_utils.cpp` (66–99): `get_runtime_max_seqlen`.
- `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_utils.hpp`: declaration of `get_runtime_max_seqlen`.

---

## 11. Reference: Upstream NVIDIA Unfused Attention

- For comparison: `transformer_engine/common/fused_attn/fused_attn_f16_max512_seqlen.cu` (upstream NVIDIA) also has an unfused path and stores **attention_weight** with shape `[batch, num_heads, q_max_seq_len, kv_max_seq_len]` (e.g. line 1247). Same idea as reusing “LSE” buffer for attention weights.

---

## 12. Testing

- **JAX test:** Add a config in `tests/jax/test_fused_attn.py` that hits the varlen path: **s_q = 1**, **s_kv &lt; 16** (e.g. 8 or 16), with a layout that uses variable-length (e.g. `QKVLayout.THD_T2HD` or `THD_THD_THD`). The parametrized configs are around lines 984–1031; add a new `pytest.param` for e.g. `(batch, 1, s_kv, h_q, h_kv, d_qk, d_v, dtype)` with `s_kv <= 16`.
- **Note:** By the time the call reaches TE common, input is already in BSHD/unpacked form; you don’t need to implement JAX sequence packing yourself, only ensure the test config triggers the varlen path (s_q=1, s_kv≤16).

---

## 13. Checklist (Summary)

- [ ] Copy or adapt varlen HIP kernels (attn_fwd / attn_bwd) into TE (e.g. under `ck_fused_attn` or in `fused_attn_rocm`).
- [ ] Use `get_runtime_max_seqlen` (ck_fused_attn_utils) to compute runtime `max_seqlen_q` and `max_seqlen_kv` before dispatch.
- [ ] In CK fwd/bwd kvpacked wrappers, branch: if `max_seqlen_q==1 && max_seqlen_kv<=16` (and layout/mode OK), call varlen launcher.
- [ ] Reuse softmax LSE auxiliary buffer for **attention weights** in varlen path (forward: write; backward: read).
- [ ] Ensure workspace size for varlen path is at least `b * h * 1 * max_seq_kv * sizeof(T)` (and that overall workspace allocation accounts for varlen when that path is chosen).
- [ ] Map TE types (e.g. BF16) to varlen template type (e.g. `hip_bfloat16`); support only the dtypes needed (e.g. BF16).
- [ ] Add JAX test config with s_q=1, s_kv≤16 to `tests/jax/test_fused_attn.py`.

---

## 14. File Reference Quick Index

| Purpose | Path |
|--------|------|
| Task / context | `task.txt` |
| Varlen forward kernels + launcher | `varlen_attn/attn_fwd.cpp` (383–439 launcher; 64–381 kernels) |
| Varlen backward kernels + launcher | `varlen_attn/attn_bwd.cpp` (372–419 launcher; 62–369 kernels) |
| Python reference | `varlen_attn/attn_py_ref.py` |
| Runtime max seqlen | `transformer_engine/common/ck_fused_attn/src/ck_fused_attn_utils.cpp` (66–99), `.hpp` |
| CK fwd/bwd kvpacked entry | `transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp` (1541 fwd, 1683 bwd) |
| Fused attn dispatch | `transformer_engine/common/fused_attn_rocm/fused_attn.cpp` (563–577, 656–675) |
| Backend enum | `transformer_engine/common/include/transformer_engine/fused_attn.h` (~148) |
| JAX fused attn tests | `tests/jax/test_fused_attn.py` (~990, parametrized configs) |
| NVIDIA unfused reference | `transformer_engine/common/fused_attn/fused_attn_f16_max512_seqlen.cu` (e.g. 1247 attention_weight) |
