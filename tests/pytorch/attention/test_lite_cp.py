# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest launcher for context parallelism tests in lite mode (BSHD & SBHD).

These tests verify that CP works in lite mode without the C++ thd_* helpers.
Only BSHD and SBHD formats are tested -- THD requires the thd_* implementations
that are stubbed out in _lite/context_parallel.py.

Run with:
    NVTE_LITE=1 pytest tests/pytorch/attention/test_lite_cp.py -v

Requires at least 2 GPUs (4 for a2a+p2p).
"""

import os
import subprocess
import sys
import pathlib
import logging

import pytest
import torch

logging.basicConfig(level=logging.INFO)

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = str(_SCRIPT_DIR.parent.parent.parent)
_WORKER_SCRIPT = str(_SCRIPT_DIR / "run_lite_cp_test.py")

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

# Config names matching CPTestConfig in run_lite_cp_test.py
_CONFIGS_CAUSAL = ["mha_causal", "gqa_causal"]
_CONFIGS_NO_MASK = ["mha_no_mask", "gqa_no_mask"]

_QKV_FORMATS = ["bshd", "sbhd"]

# CP comm types that work with BSHD/SBHD (no THD needed)
_CP_COMM_TYPES = ["p2p", "all_gather", "a2a"]


def _get_num_gpus(cp_comm_type):
    """Return number of GPUs required for a given CP comm type."""
    if cp_comm_type == "a2a+p2p":
        return 4
    return 2


def _run_worker(num_gpus, **kwargs):
    """Launch the multi-process test worker and check its exit code."""
    args = [
        sys.executable,
        "-m", "torch.distributed.launch",
        f"--nproc-per-node={num_gpus}",
        _WORKER_SCRIPT,
    ]
    for k, v in kwargs.items():
        args.append(f"{k}={v}")

    env = os.environ.copy()
    env["NVTE_LITE"] = "1"
    # Ensure repo root is on PYTHONPATH for dev-tree runs
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    if result.returncode != 0:
        # Print full output for debugging
        logging.error("STDOUT:\n%s", result.stdout)
        logging.error("STDERR:\n%s", result.stderr)
        pytest.fail(
            f"CP test worker failed (exit {result.returncode}). "
            f"See log output above for details."
        )


def _skip_if_insufficient_gpus(num_gpus):
    if torch.cuda.device_count() < num_gpus:
        pytest.skip(f"Test requires {num_gpus} GPUs, found {torch.cuda.device_count()}")


# ---------------------------------------------------------------------------
# Tests: P2P (ring exchange)
# ---------------------------------------------------------------------------

class TestCPWithP2P:
    """Context parallelism with P2P (ring) KV exchange -- BSHD & SBHD."""

    @pytest.mark.parametrize("qkv_format", _QKV_FORMATS)
    @pytest.mark.parametrize("config_name", _CONFIGS_CAUSAL + _CONFIGS_NO_MASK)
    def test_p2p(self, config_name, qkv_format):
        num_gpus = _get_num_gpus("p2p")
        _skip_if_insufficient_gpus(num_gpus)
        attn_mask_type = "causal" if "causal" in config_name else "no_mask"
        _run_worker(
            num_gpus,
            config_name=config_name,
            qkv_format=qkv_format,
            cp_comm_type="p2p",
            attn_mask_type=attn_mask_type,
        )


# ---------------------------------------------------------------------------
# Tests: All-Gather
# ---------------------------------------------------------------------------

class TestCPWithAllGather:
    """Context parallelism with KV All-Gather -- BSHD & SBHD."""

    @pytest.mark.parametrize("qkv_format", _QKV_FORMATS)
    @pytest.mark.parametrize("config_name", _CONFIGS_CAUSAL + _CONFIGS_NO_MASK)
    def test_all_gather(self, config_name, qkv_format):
        num_gpus = _get_num_gpus("all_gather")
        _skip_if_insufficient_gpus(num_gpus)
        attn_mask_type = "causal" if "causal" in config_name else "no_mask"
        _run_worker(
            num_gpus,
            config_name=config_name,
            qkv_format=qkv_format,
            cp_comm_type="all_gather",
            attn_mask_type=attn_mask_type,
        )


# ---------------------------------------------------------------------------
# Tests: A2A (Ulysses)
# ---------------------------------------------------------------------------

class TestCPWithA2A:
    """Context parallelism with All-to-All (Ulysses) -- BSHD & SBHD.

    A2A requires num_heads and num_gqa_groups divisible by cp_size.
    The test configs satisfy this for cp_size=2.
    """

    @pytest.mark.parametrize("qkv_format", _QKV_FORMATS)
    @pytest.mark.parametrize("config_name", _CONFIGS_CAUSAL + _CONFIGS_NO_MASK)
    def test_a2a(self, config_name, qkv_format):
        num_gpus = _get_num_gpus("a2a")
        _skip_if_insufficient_gpus(num_gpus)
        attn_mask_type = "causal" if "causal" in config_name else "no_mask"
        _run_worker(
            num_gpus,
            config_name=config_name,
            qkv_format=qkv_format,
            cp_comm_type="a2a",
            attn_mask_type=attn_mask_type,
        )


# ---------------------------------------------------------------------------
# Tests: dtype coverage
# ---------------------------------------------------------------------------

class TestCPDtypes:
    """Verify CP works with both bf16 and fp16 in lite mode."""

    @pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
    def test_p2p_bshd_dtypes(self, dtype_str):
        _skip_if_insufficient_gpus(2)
        _run_worker(
            2,
            config_name="mha_causal",
            qkv_format="bshd",
            cp_comm_type="p2p",
            attn_mask_type="causal",
            dtype_str=dtype_str,
        )

    @pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
    def test_a2a_bshd_dtypes(self, dtype_str):
        _skip_if_insufficient_gpus(2)
        _run_worker(
            2,
            config_name="mha_causal",
            qkv_format="bshd",
            cp_comm_type="a2a",
            attn_mask_type="causal",
            dtype_str=dtype_str,
        )
