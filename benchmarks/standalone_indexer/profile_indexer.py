"""Profile the standalone forward score-reduce Triton kernel (bf16).

Extracted from ``transformer_engine/jax/triton_extensions/indexer.py`` --
``_score_reduce_kernel`` only, no transformer_engine / jax dependency.

Measures wall time and effective TFLOPS for the fused kernel:

    scores = relu(einsum("...thi,...si->...ths", Hq, Hk))   # never written
    O      = einsum("...ths,...th->...ts", scores, W_o)

Run:
  python benchmarks/standalone_indexer/profile_indexer.py
"""

import torch

from _common import make_kernel_inputs, time_fn
# Drive the ACTUAL source kernel in
# transformer_engine/jax/triton_extensions/indexer.py (via the indexer_bridge
# adapter). Swap to `indexer_kernels` to profile the self-contained copy.
from indexer_bridge import score_reduce


# --- FLOP accounting ------------------------------------------------------------

def kernel_flops(B, oH, T, S, H, d_i):
    # The kernel does the score matmul + the weighted H-reduction; the four
    # projection GEMMs (C_q, H_q, H_k, W_o) are excluded -- they're not in the
    # kernel. 2 flops per multiply-add.
    #   scores = relu(Hq @ Hk^T)   : 2 * B*oH * T * H * S * d_i
    #   O      = sum_h scores * W_o : 2 * B*oH * T * S * H
    n = B * oH
    return 2 * (n * T * H * S * d_i + n * T * S * H)


# --- Reference (torch) for a correctness sanity check ---------------------------

def torch_reference(Hq, Hk, W_o):
    # Hq (B,oH,T,H,d_i), Hk (B,oH,S,d_i), W_o (B,oH,T,H) -> O (B,oH,T,S)
    scores = torch.einsum("bothi,bosi->boths", Hq.float(), Hk.float())
    scores = torch.relu(scores)
    O = torch.einsum("boths,both->bots", scores, W_o.float())
    return O


# --- Driver ---------------------------------------------------------------------

CONFIGS = [
    #(B, oH, T,    S,    H,  d_i)
    ( 2, 64, 4096, 4096, 64, 128),
]


def check_correctness():
    B, oH, T, S, H, d_i = 1, 2, 128, 256, 8, 128
    Hq, Hk, W_o = make_kernel_inputs(B, oH, T, S, H, d_i, torch.bfloat16)
    O = score_reduce(Hq, Hk, W_o).float()
    ref = torch_reference(Hq, Hk, W_o)
    rel = (O - ref).norm() / ref.norm().clamp_min(1e-9)
    print(f"    correctness: rel L2 err vs torch fp32 ref = {rel.item():.4e}")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    check_correctness()
    print()
    for B, oH, T, S, H, d_i in CONFIGS:
        Hq, Hk, W_o = make_kernel_inputs(B, oH, T, S, H, d_i, torch.bfloat16)
        flops = kernel_flops(B, oH, T, S, H, d_i)

        print(f"--- B={B} oH={oH} T={T} S={S} H={H} d_i={d_i} bfloat16 ---")
        print(f"    kernel work = {flops/1e9:.2f} GFLOPs/call")

        try:
            sec = time_fn(lambda: score_reduce(Hq, Hk, W_o))
            ms = sec * 1e3
            tflops = flops / sec / 1e12
            print(f"    {'score_reduce':<14} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s")
        except Exception as e:  # noqa: BLE001
            print(f"    {'score_reduce':<14} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")

        # Report the autotuner-selected config.
        from indexer_bridge import _score_reduce_kernel
        print("    Best config: ", _score_reduce_kernel.best_config)


if __name__ == "__main__":
    main()
