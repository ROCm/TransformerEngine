# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Python-level reproducer for the deterministic CK fused-attn backward
"forward output O clobbered" bug (ROCm/TransformerEngine PR #599).

Root cause: under jit(value_and_grad(...)) the forward output O is a dead residual
after the bwd custom call, so XLA folds O's storage INTO the bwd workspace. The
deterministic dq_acc scratch (carved from that same XLA workspace since commit
8fd79f92) is zeroed BEFORE the bwd reads O, so O is clobbered:
  - the primal reader mean(O)  -> reads zeros  -> value == 0
  - dot_do_o (grad)            -> reads zeros  -> dq == 0

The committed build MASKS this with a workspace "floor" that pushes O past the
zeroed extent, so the bug only manifests with the floor removed. Run with
NVTE_CK_DROP_BWD_FLOOR=1 to expose it (requires the debug toggle in
ck_attn_bwd_workspace_size). Config below is the failing case:
  b=2 s_q=1024 s_kv=2048 h=12 gqa=6 d_qk=128 d_v=64 BF16 CROSS GQA SEPARATE
  PADDING_CAUSAL, deterministic.

Usage:
  # exposes the bug (needs the floor-drop toggle build):
  NVTE_CK_DROP_BWD_FLOOR=1 HIP_VISIBLE_DEVICES=0 python scripts/repro_ck_bwd_o_clobber.py
  # committed build (floor on) -> masked, prints PASS:
  HIP_VISIBLE_DEVICES=0 python scripts/repro_ck_bwd_o_clobber.py
"""
import os

# Force the deterministic backward (this is what selects the v2 ck_tile path that
# zeroes the large dq_acc workspace).
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tests", "jax"))

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
)
from test_fused_attn import FusedAttnRunner, SeqDescFormat, customcall_fused_dpa


def _build_runner():
    r = FusedAttnRunner(
        2,        # batch
        1024,     # q seqlen
        2048,     # kv seqlen
        12,       # attn heads
        6,        # gqa groups
        128,      # d_qk
        64,       # d_v   (!= d_qk -> asm v3 falls back to v2 ck_tile)
        AttnBiasType.NO_BIAS,
        AttnMaskType.PADDING_CAUSAL_MASK,
        AttnSoftmaxType.VANILLA_SOFTMAX,
        0.0,      # dropout
        jnp.bfloat16,
        True,     # is_training
        QKVLayout.BSHD_BSHD_BSHD,   # SEPARATE -> group mode via bshd_to_thd
        None,
        None,
        seq_desc_format=SeqDescFormat.Seqlens,
    )
    r._setup_inputs()
    return r


def _kwargs(r):
    return dict(
        attn_bias_type=r.attn_bias_type,
        attn_mask_type=r.attn_mask_type,
        softmax_type=r.softmax_type,
        scaling_factor=r.scaling_factor,
        dropout_probability=r.dropout_prob,
        is_training=r.is_training,
        qkv_layout=r.qkv_layout,
        max_segments_per_seq=r._get_max_segments_per_sequence(),
        window_size=r.window_size,
    )


def main():
    r = _build_runner()
    kwargs = _kwargs(r)

    def fwd_out(q, k, v):
        return customcall_fused_dpa(q, k, v, None, None, r.sequence_desciptor, None, **kwargs)

    # Sum only the valid (non-padded) forward outputs; this is the quantity the
    # test's value_and_grad returns as `primitive_out`.
    def loss(q, k, v):
        o = fwd_out(q, k, v)
        valid = jnp.where(r.pad_q[..., None, None], 0.0, o).astype(jnp.float32)
        return jnp.sum(valid)

    # (1) Forward-only reference: O is never fed to a bwd, so it is never folded
    #     into a workspace -> always correct.
    ref_loss = float(jit(loss)(r.q, r.k, r.v))

    # (2) value_and_grad: O becomes a dead residual for the deterministic bwd, which
    #     zeroes the dq_acc workspace that XLA overlapped with O.
    (val, (dq, dk, dv)) = jit(value_and_grad(loss, argnums=(0, 1, 2)))(r.q, r.k, r.v)
    val = float(val)
    dq_absmax = float(jnp.max(jnp.abs(dq)))

    floor_dropped = os.environ.get("NVTE_CK_DROP_BWD_FLOOR", "0") == "1"
    print(f"backend                 : {r.backend}")
    print(f"NVTE_CK_DROP_BWD_FLOOR  : {floor_dropped}")
    print(f"forward-only loss (ref) : {ref_loss:.6g}")
    print(f"value_and_grad loss     : {val:.6g}   (== primitive_out in the test)")
    print(f"value_and_grad dq absmax: {dq_absmax:.6g}")

    # The forward-only reference must be a real (nonzero, finite) value.
    assert ref_loss != 0.0 and ref_loss == ref_loss, "reference loss is degenerate"

    reproduced = (val == 0.0) or (val != val)  # zeroed or NaN
    grads_dead = dq_absmax == 0.0

    print()
    if reproduced:
        print("REPRODUCED: value_and_grad forward output was clobbered "
              f"(loss {val:.6g} vs reference {ref_loss:.6g}); "
              f"grads {'also zeroed' if grads_dead else 'nonzero'}.")
        print("  -> O was folded into the bwd workspace and zeroed by the dq_acc memset.")
        return 1
    print("PASS: forward output intact "
          f"(loss {val:.6g} matches reference {ref_loss:.6g}).")
    if not floor_dropped:
        print("  (floor is masking the bug; rerun with NVTE_CK_DROP_BWD_FLOOR=1 to expose it.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
