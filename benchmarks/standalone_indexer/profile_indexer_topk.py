"""Profile the standalone fused score + streaming top-k Triton kernel (bf16).

Extracted from ``transformer_engine/jax/triton_extensions/indexer.py`` --
``_score_topk_kernel`` only, no transformer_engine / jax dependency.

Computes the same scores as the forward kernel but never materializes the
(B, oH, T_t, T_s) score matrix -- returns the top-k indices into T_s directly.
top-k is comparison-only and counted as 0 FLOP, so reported TFLOPS reflect the
score compute.

Run:
  python benchmarks/standalone_indexer/profile_indexer_topk.py
"""

import torch

from _common import make_kernel_inputs, time_fn
# Drive the ACTUAL source kernels in
# transformer_engine/jax/triton_extensions/indexer.py (via the indexer_bridge
# adapter).
from indexer_bridge import score_topk


def kernel_flops(B, oH, T, S, H, d_i):
    # Score matmul + weighted H-reduction; top-k is 0 FLOP.
    n = B * oH
    return 2 * (n * T * H * S * d_i + n * T * S * H)


def torch_scores(Hq, Hk, W_o):
    scores = torch.einsum("bothi,bosi->boths", Hq.float(), Hk.float())
    scores = torch.relu(scores)
    return torch.einsum("boths,both->bots", scores, W_o.float())


CONFIGS = [
    #(B, oH, T,    S,    H,  d_i)
    ( 2, 64, 1024, 1024, 64, 128),
]

K_TOPK = 512


def check_correctness():
    B, oH, T, S, H, d_i = 1, 2, 64, 256, 8, 128
    k = 64
    Hq, Hk, W_o = make_kernel_inputs(B, oH, T, S, H, d_i, torch.bfloat16)
    idx = score_topk(Hq, Hk, W_o, k=k).long()
    scores = torch_scores(Hq, Hk, W_o)
    # Compare the score *values* at the selected indices (robust to tie order).
    sel = torch.gather(scores, -1, idx)
    sel_sorted = torch.sort(sel, dim=-1, descending=True).values
    ref_vals = torch.topk(scores, k, dim=-1).values
    rel = (sel_sorted - ref_vals).norm() / ref_vals.norm().clamp_min(1e-9)
    print(f"    correctness (k={k}): top-k value rel err vs torch = {rel.item():.4e}")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}\nk = {K_TOPK}\n")
    check_correctness()
    print()
    for B, oH, T, S, H, d_i in CONFIGS:
        Hq, Hk, W_o = make_kernel_inputs(B, oH, T, S, H, d_i, torch.bfloat16)
        flops = kernel_flops(B, oH, T, S, H, d_i)

        print(f"--- B={B} oH={oH} T={T} S={S} H={H} d_i={d_i} bfloat16 ---")
        print(f"    kernel work = {flops/1e9:.2f} GFLOPs/call (top-k = 0 FLOP)")

        try:
            sec = time_fn(lambda: score_topk(Hq, Hk, W_o, k=K_TOPK))
            ms = sec * 1e3
            tflops = flops / sec / 1e12
            print(f"    {'score_topk':<14} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s")
        except Exception as e:  # noqa: BLE001
            print(f"    {'score_topk':<14} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")
        print()


if __name__ == "__main__":
    main()
