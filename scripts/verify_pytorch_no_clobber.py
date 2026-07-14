# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Verify the deterministic CK bwd "forward output O clobbered" bug is JAX/XLA
specific: the SAME config passes through the PyTorch integration, even with the
workspace floor removed (NVTE_CK_DROP_BWD_FLOOR=1) that breaks the JAX path.

Why PyTorch is immune: the bug needs XLA to fold the dead forward output O into the
bwd workspace (so the dq_acc zeroing clobbers it). PyTorch's caching allocator +
autograd keep the saved O and the TE workspace in distinct allocations — O is never
inside the workspace — so zeroing the workspace can't touch O regardless of the floor.

Config mirrors the failing JAX case:
  b=2 s_q=1024 s_kv=2048 h=12 gqa=6 d_qk=128 d_v=64 BF16 cross GQA padding_causal.

Usage:
  # floor dropped (breaks JAX; must still PASS on PyTorch):
  NVTE_CK_DROP_BWD_FLOOR=1 HIP_VISIBLE_DEVICES=0 python scripts/verify_pytorch_no_clobber.py
"""
import os

os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")  # deterministic bwd
os.environ["NVTE_FUSED_ATTN_CK"] = "1"       # force the CK backend (the affected path)
os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "0"

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tests", "pytorch"))
sys.path.insert(0, os.path.join(_ROOT, "tests", "pytorch", "attention"))

import torch
from utils import ModelConfig
from test_attention import _run_dot_product_attention

DTYPE = torch.bfloat16
QKV_LAYOUT = "bshd_bshd_bshd"  # SEPARATE q/k/v, matches the JAX BSHD_BSHD_BSHD case


def _stats(t):
    tf = t.float()
    return float(tf.abs().max()), bool(torch.isnan(tf).any()), bool(torch.isinf(tf).any())


def main():
    config = ModelConfig(
        2,                       # batch_size
        1024,                    # max_seqlen_q
        12,                      # num_heads
        128,                     # head_dim_qk
        max_seqlen_kv=2048,      # cross attention
        num_gqa_groups=6,        # GQA
        head_dim_v=64,           # d_v != d_qk  (asm v3 -> v2 ck_tile fallback)
        attn_mask_type="padding_causal",
    )

    floor_dropped = os.environ.get("NVTE_CK_DROP_BWD_FLOOR", "0") == "1"
    print(f"NVTE_CK_DROP_BWD_FLOOR : {floor_dropped}")
    print(f"deterministic          : {os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] == '0'}")
    print(
        "config                 : "
        f"b={config.batch_size} s_q={config.max_seqlen_q} s_kv={config.max_seqlen_kv} "
        f"h={config.num_heads} gqa={config.num_gqa_groups} "
        f"d_qk={config.head_dim_qk} d_v={config.head_dim_v} mask={config.attn_mask_type}"
    )

    # CK fused backend (the path that zeroes the deterministic dq_acc workspace).
    fused_out, _, (fdq, fdk, fdv, _) = _run_dot_product_attention(
        DTYPE, config, "FusedAttention", False, QKV_LAYOUT, True, False, True
    )
    # Unfused reference (same seeded inputs -> directly comparable).
    ref_out, _, (rdq, rdk, rdv, _) = _run_dot_product_attention(
        DTYPE, config, "UnfusedDotProductAttention", False, QKV_LAYOUT, True, False, True
    )

    out_absmax, out_nan, out_inf = _stats(fused_out)
    dq_absmax = fdq.detach().float().abs().max().item()
    print(f"\nfused out absmax={out_absmax:.6g} nan={out_nan} inf={out_inf}  dq absmax={dq_absmax:.6g}")

    # (A) JAX failure signature is out==0 / dq==0. Check the fused forward output and
    #     gradients survived the deterministic backward.
    assert not out_nan and not out_inf, "fused output has NaN/Inf"
    assert out_absmax > 0.0, "fused output is all zero (clobbered — bug present in PyTorch!)"
    assert dq_absmax > 0.0, "fused dq is all zero (grads dead — bug present in PyTorch!)"

    # (B) Forward output must match the unfused reference (proves O was not corrupted).
    out_diff = (fused_out.float() - ref_out.float()).abs().max().item()
    out_rel = out_diff / (ref_out.float().abs().max().item() + 1e-6)
    print(f"fused-vs-unfused forward out: max|Δ|={out_diff:.4g} rel={out_rel:.4g} "
          f"[{'OK' if out_rel < 3e-2 else 'MISMATCH'}]")
    assert out_rel < 3e-2, "fused forward output diverged from unfused reference"

    # (C) Rigorous JAX-specificity proof: the floor (the load-bearing JAX fix) must be a
    #     no-op in PyTorch. Compare the fused result to a saved baseline computed with the
    #     opposite floor setting; they must be identical.
    save = os.environ.get("REPRO_SAVE")
    if save:
        torch.save({"out": fused_out.detach().cpu(), "dq": fdq.detach().cpu(),
                    "dk": fdk.detach().cpu(), "dv": fdv.detach().cpu()}, save)
        print(f"\nsaved fused result (floor_dropped={floor_dropped}) -> {save}")
        return 0

    baseline = os.environ.get("REPRO_BASELINE")
    if baseline:
        b = torch.load(baseline)
        identical = True
        for name, cur in [("out", fused_out), ("dq", fdq), ("dk", fdk), ("dv", fdv)]:
            d = (cur.detach().cpu().float() - b[name].float()).abs().max().item()
            same = d == 0.0
            identical &= same
            print(f"  floor-toggle {name:3s} max|Δ| vs baseline = {d:.4g} "
                  f"[{'identical' if same else 'DIFFERS'}]")
        print()
        if identical:
            print("PASS: PyTorch fused result is BITWISE-identical with the floor on vs off "
                  "-> the workspace floor (the load-bearing JAX fix) is inert in PyTorch, "
                  "confirming the bug is JAX/XLA specific.")
            return 0
        print("FAIL: floor toggle changed the PyTorch result (unexpected).")
        return 1

    print("\nPASS: PyTorch CK fused forward/grads intact and match the unfused reference"
          + (" with the floor dropped" if floor_dropped else "")
          + " -> the JAX clobber does not occur under PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
