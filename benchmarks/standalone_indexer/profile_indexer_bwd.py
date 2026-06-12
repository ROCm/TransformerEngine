"""Profile the standalone backward score-chunk Triton kernel (bf16).

Extracted from ``transformer_engine/jax/triton_extensions/indexer.py`` --
``_score_dscores_chunk_kernel`` only, no transformer_engine / jax dependency.

The original backward loops over H/H_CHUNK chunks, each chunk fusing
(score recompute + relu + mask + dO*W_o broadcast) into dscores_chunk and
reducing dWo_chunk in registers. The dHq/dHk reductions on dscores_chunk are
plain GEMMs (hipBLASLt einsums) and are NOT part of this kernel, so they are
excluded here. We time the per-chunk kernel and the full chunk sweep.

Run:
  python benchmarks/standalone_indexer/profile_indexer_bwd.py
"""

import torch

from _common import make_kernel_inputs, time_fn
# Drive the ACTUAL source kernel in
# transformer_engine/jax/triton_extensions/indexer.py (via the indexer_bridge
# adapter). Swap to `indexer_kernels` to profile the self-contained copy.
from indexer_bridge import bwd_h_chunk, score_dscores_chunk


def kernel_flops_per_chunk(B, oH, T, S, H_CHUNK, d_i):
    # Per chunk: score recompute matmul + the dWo sum-reduction over s.
    n = B * oH
    return 2 * (n * T * H_CHUNK * S * d_i + n * T * H_CHUNK * S)


def torch_reference_chunk(Hq_chunk, Hk, W_o_chunk, dO):
    # scores = relu(Hq_chunk @ Hk^T), then
    #   dWo[...,h]   = sum_s relu(scores)[...,h,s] * dO[...,s]
    #   dscores[...] = (scores>0) * dO[...,s] * W_o[...,h]
    scores = torch.einsum("bothi,bosi->boths", Hq_chunk.float(), Hk.float())
    relu_mask = scores > 0
    h_relu = torch.relu(scores)
    dWo = torch.einsum("boths,bots->both", h_relu, dO.float())
    dscores = relu_mask.float() * (dO.float()[:, :, :, None, :]
                                   * W_o_chunk.float()[..., None])
    return dscores, dWo


CONFIGS = [
    #(B, oH, T,    S,    H,  d_i)
    ( 2, 64, 1024, 1024, 64, 128),
]


def check_correctness():
    B, oH, T, S, H, d_i = 1, 2, 128, 256, 8, 128
    H_CHUNK = bwd_h_chunk(H)
    Hq, Hk, W_o = make_kernel_inputs(B, oH, T, S, H, d_i, torch.bfloat16)
    dO = torch.randn((B, oH, T, S), dtype=torch.float32, device="cuda")
    Hq_c = Hq[:, :, :, :H_CHUNK, :].contiguous()
    W_o_c = W_o[:, :, :, :H_CHUNK].contiguous()
    dscores, dWo = score_dscores_chunk(Hq_c, Hk, W_o_c, dO)
    ref_ds, ref_dwo = torch_reference_chunk(Hq_c, Hk, W_o_c, dO)
    ds_err = (dscores.float() - ref_ds).norm() / ref_ds.norm().clamp_min(1e-9)
    dwo_err = (dWo.float() - ref_dwo).norm() / ref_dwo.norm().clamp_min(1e-9)
    print(f"    correctness (H_CHUNK={H_CHUNK}): dscores rel err = {ds_err.item():.4e}, "
          f"dWo rel err = {dwo_err.item():.4e}")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    check_correctness()
    print()
    for B, oH, T, S, H, d_i in CONFIGS:
        H_CHUNK = bwd_h_chunk(H)
        n_chunks = H // H_CHUNK
        Hq, Hk, W_o = make_kernel_inputs(B, oH, T, S, H, d_i, torch.bfloat16)
        dO = torch.randn((B, oH, T, S), dtype=torch.float32, device="cuda")

        # Pre-slice the per-chunk views the original scan feeds the kernel.
        chunks = [
            (Hq[:, :, :, c * H_CHUNK:(c + 1) * H_CHUNK, :].contiguous(),
             W_o[:, :, :, c * H_CHUNK:(c + 1) * H_CHUNK].contiguous())
            for c in range(n_chunks)
        ]

        per_chunk_flops = kernel_flops_per_chunk(B, oH, T, S, H_CHUNK, d_i)
        total_flops = per_chunk_flops * n_chunks

        print(f"--- B={B} oH={oH} T={T} S={S} H={H} d_i={d_i} bfloat16 ---")
        print(f"    H_CHUNK={H_CHUNK}  n_chunks={n_chunks}")
        print(f"    per-chunk work = {per_chunk_flops/1e9:.2f} GFLOPs   "
              f"full sweep = {total_flops/1e9:.2f} GFLOPs")

        # Single-chunk timing.
        Hq_c, W_o_c = chunks[0]
        try:
            sec = time_fn(lambda: score_dscores_chunk(Hq_c, Hk, W_o_c, dO))
            ms = sec * 1e3
            tflops = per_chunk_flops / sec / 1e12
            print(f"    {'one chunk':<14} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s")
        except Exception as e:  # noqa: BLE001
            print(f"    {'one chunk':<14} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")

        # Full chunk-sweep timing (the work one backward pass issues).
        def run_all():
            for hq_c, wo_c in chunks:
                score_dscores_chunk(hq_c, Hk, wo_c, dO)

        try:
            sec = time_fn(run_all)
            ms = sec * 1e3
            tflops = total_flops / sec / 1e12
            print(f"    {'full sweep':<14} {ms:8.3f} ms   {tflops:6.2f} TFLOP/s")
        except Exception as e:  # noqa: BLE001
            print(f"    {'full sweep':<14} FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")
        print()
        # Report the autotuner-selected config.
        from indexer_bridge import _score_dscores_chunk_kernel
        print("    Best config: ", _score_dscores_chunk_kernel.best_config)


if __name__ == "__main__":
    main()
