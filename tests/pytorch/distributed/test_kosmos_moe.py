# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Fused expert-parallel MoE (te.MoE, KOSMOS: fused router, routed experts and shared expert in one kernel launch
for the forward and one for the backward): 8-GPU checks against an fp32 torch reference."""

import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

import transformer_engine.pytorch  # pylint: disable=unused-import
import transformer_engine_torch as tex

TEST_ROOT = Path(__file__).parent.resolve()
EP = 8

if torch.cuda.device_count() < EP:
    pytest.skip(f"The fused MoE needs {EP} GPUs.", allow_module_level=True)
if not tex.moe_is_supported():
    pytest.skip("The fused MoE needs a TE build with it and a gfx950 device.", allow_module_level=True)

CONFIGS = {
    "softmax": ["--router", "softmax"],
    "softmax_no_aux": ["--router", "softmax", "--aux-groups", "0"],
    "sigmoid_biases_seq_aux": ["--router", "sigmoid", "--expert-bias", "--router-bias", "--aux-groups", "2"],
    "sqrtsoftplus_grouped": ["--router", "sqrtsoftplus", "--groups", "4", "--group-topk", "2"],
    "softmax_pre": ["--router", "softmax_pre"],
    "shared": ["--router", "sigmoid", "--shared"],
    "shared_gated": ["--router", "sigmoid", "--expert-bias", "--shared", "--shared-gate"],
    "clamp10_shared": ["--router", "softmax", "--clamp10", "--shared"],
    "rounds_shared": ["--router", "softmax", "--shared", "--cap-frac", "0.5"],
    "qwen30_shared": ["--router", "softmax", "--shared", "--ffn", "768", "--experts", "128", "--tokens", "2048"],
    "mxfp8": ["--router", "softmax", "--mxfp8"],
    "mxfp8_shared_gated": ["--router", "sigmoid", "--mxfp8", "--shared", "--shared-gate"],
    "mxfp8_fp8_combine_shared": ["--router", "softmax", "--mxfp8", "--fp8-combine", "--shared"],
    "mxfp8_rounds_shared": ["--router", "softmax", "--mxfp8", "--shared", "--cap-frac", "0.5"],
    "k1": ["--router", "softmax", "--topk", "1"],
    "k2_shared": ["--router", "sigmoid", "--topk", "2", "--shared"],
    "k4_shared": ["--router", "softmax", "--topk", "4", "--shared"],
    "k6_shared_gated": ["--router", "sigmoid", "--topk", "6", "--shared", "--shared-gate"],
    "k10_shared_gated": ["--router", "softmax", "--topk", "10", "--shared", "--shared-gate"],
    "grouped_shared_gated": ["--router", "sqrtsoftplus", "--groups", "4", "--group-topk", "2", "--shared", "--shared-gate"],
    "mxfp8_biases_shared": ["--router", "sigmoid", "--mxfp8", "--expert-bias", "--router-bias", "--aux-groups", "2", "--shared"],
    "mxfp8_grouped_shared": ["--router", "sqrtsoftplus", "--mxfp8", "--groups", "4", "--group-topk", "2", "--shared"],
    "layers4_shared_gated": ["--router", "sigmoid", "--shared", "--shared-gate", "--layers", "4"],
    "layers3_mxfp8_shared": ["--router", "softmax", "--mxfp8", "--shared", "--layers", "3"],
    "train3_shared": ["--router", "sigmoid", "--expert-bias", "--shared", "--train-steps", "3"],
    "train3_mxfp8_shared": ["--router", "softmax", "--mxfp8", "--shared", "--train-steps", "3"],
    "qwen30_T8192_shared": ["--router", "softmax", "--shared", "--ffn", "768", "--experts", "128", "--tokens", "8192"],
    "checkpoint_shared": ["--router", "softmax", "--shared", "--ckpt-steps", "12"],
    "checkpoint_mxfp8_shared": ["--router", "sigmoid", "--mxfp8", "--shared", "--ckpt-steps", "8"],
    "abandon_shared_gated": ["--router", "sigmoid", "--shared", "--shared-gate", "--abandon-steps", "10"],
    "abandon_mxfp8_shared": ["--router", "softmax", "--mxfp8", "--shared", "--abandon-steps", "8"],
    "main_grad_fp32": ["--router", "softmax", "--main-grad", "fp32"],
    "main_grad_fp32_biases_shared_gated": ["--router", "sigmoid", "--router-bias", "--expert-bias", "--shared",
                                           "--shared-gate", "--main-grad", "fp32"],
    "main_grad_bf16_biases_shared": ["--router", "sigmoid", "--router-bias", "--shared", "--main-grad", "bf16"],
    "main_grad_fp32_k2_shared": ["--router", "softmax", "--topk", "2", "--shared", "--main-grad", "fp32"],
    "main_grad_fp32_rounds_shared": ["--router", "softmax", "--shared", "--cap-frac", "0.5", "--main-grad", "fp32"],
    "main_grad_fp32_mxfp8": ["--router", "softmax", "--mxfp8", "--main-grad", "fp32"],
    "main_grad_bf16_mxfp8_shared": ["--router", "sigmoid", "--router-bias", "--mxfp8", "--shared", "--main-grad",
                                    "bf16"],
    "main_grad_fp32_mxfp8_rounds_shared": ["--router", "softmax", "--mxfp8", "--shared", "--cap-frac", "0.5",
                                           "--main-grad", "fp32"],
}


# The caching allocator with expandable segments (Megatron's MoE configs set it): the buffers the ranks share come
# from the fused MoE's own pool of plain segments, so it runs unchanged.
ENV_CONFIGS = {
    "expandable_segments_shared": ["--router", "softmax", "--shared"],
    "expandable_segments_mxfp8_shared_gated": ["--router", "sigmoid", "--mxfp8", "--shared", "--shared-gate"],
}
EXPANDABLE = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}


def _run(args, env):
    cmd = ["torchrun", f"--nproc_per_node={EP}", str(TEST_ROOT / "run_kosmos_moe.py"), *args]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, check=False)
    if result.returncode != 0 or "KOSMOS MoE CHECK PASS" not in result.stdout:
        raise AssertionError(f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr[-4000:]}")


@pytest.mark.parametrize("config", list(ENV_CONFIGS))
def test_kosmos_moe_expandable_segments(config):
    """The full check of run_kosmos_moe.py with the caching allocator's expandable segments on."""
    _run(ENV_CONFIGS[config], {**os.environ, **EXPANDABLE})


def test_kosmos_moe_rank_failure_exits():
    """One rank raises (unrelated to the MoE) with a workspace live and never finalized: the job ends promptly,
    with that error, instead of a rank waiting in a collective at interpreter exit."""
    cmd = ["torchrun", f"--nproc_per_node={EP}", str(TEST_ROOT / "run_kosmos_moe_fail.py")]
    t0 = time.monotonic()
    result = subprocess.run(cmd, env=os.environ, capture_output=True, text=True, timeout=600, check=False)
    elapsed = time.monotonic() - t0
    assert result.returncode != 0, "the failing rank's error must fail the job"
    assert "an error unrelated to the fused MoE" in result.stderr + result.stdout
    assert elapsed < 60, f"the job took {elapsed:.0f} s to end after one rank failed"


@pytest.mark.parametrize("config", list(CONFIGS))
def test_kosmos_moe(config):
    """Forward and backward vs an fp32 torch reference (MXFP8 and clamped SwiGLU by SNR); exactly one kernel launch
    for the forward and one for the backward; two forwards in flight; no-grad forward; determinism; refusals;
    optionally several layers on one workspace, SGD steps, non-reentrant activation checkpointing, discarded
    forwards (their saved state released once every rank dropped it) and weight gradients added into main_grad
    inside the backward launch (fuse_wgrad_accumulation)."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={EP}",
        str(TEST_ROOT / "run_kosmos_moe.py"),
        *CONFIGS[config],
    ]
    result = subprocess.run(
        cmd, env=os.environ, capture_output=True, text=True, timeout=900, check=False
    )
    if result.returncode != 0 or "KOSMOS MoE CHECK PASS" not in result.stdout:
        raise AssertionError(f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr[-4000:]}")
