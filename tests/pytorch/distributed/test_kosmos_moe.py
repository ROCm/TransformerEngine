# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""KOSMOS fused expert-parallel MoE (transformer_engine.pytorch.kosmos), eight ranks.

Each check runs in its own torchrun of run_kosmos_moe.py, under a timeout: a KOSMOS kernel whose
peers never arrive waits forever, and the timeout turns that into a failure instead of a hung job.
Skipped unless KOSMOS is built and eight gfx950 GPUs are visible.
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from transformer_engine.pytorch import kosmos

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS = 8
TIMEOUT_S = 900

if torch.cuda.device_count() < NUM_PROCS:
    pytest.skip(f"KOSMOS MoE needs {NUM_PROCS} GPUs", allow_module_level=True)
if not kosmos.is_available():
    pytest.skip("KOSMOS MoE not built, or the GPU is not gfx950", allow_module_level=True)

CHECKS = [
    "layout",
    "numerics",
    "mutation",
    "determinism",
    "two_routings",
    "three_layers",
    "inflight_fifo",
    "inflight_lifo",
    "nograd_between",
    "forward_without_backward",
    "save_tiers",
    "saved_rows_below",
    "mutation_saved_xp",
    "mutation_saved_y",
    "inference_allocates_nothing",
    "skew_rounds",
    "dropped_ids",
    "degenerate",
    "create_destroy_twice",
    "main_grad_acc",
    "saved_tensor_hooks",
    "version_counter",
    "two_instances",
    "checkpoint_reentrant",
    "checkpoint_nonreentrant",
    "symm_vs_own",
    "te_ep_reference",
    "graph_one_layer",
    "graph_two_microbatches",
    "graph_routing",
    "graph_save_tiers",
    "graph_mutation",
    "graph_first_call_in_capture",
]


def _run(check: str, shape: str) -> None:
    # torchrun first, so conftest.py bounds it with `timeout` (SIGTERM lets torchrun reap its
    # workers) before subprocess.run's own timeout would orphan them
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_PROCS}",
        str(TEST_ROOT / "run_kosmos_moe.py"),
        "--test",
        check,
        "--shape",
        shape,
    ]
    result = subprocess.run(cmd, env=os.environ, check=False, timeout=TIMEOUT_S)
    assert result.returncode == 0


@pytest.mark.parametrize("check", CHECKS)
def test_kosmos_moe(check):
    _run(check, "small")


@pytest.mark.parametrize(
    "check",
    ["numerics", "mutation", "three_layers", "skew_rounds", "save_tiers", "saved_rows_below"],
)
def test_kosmos_moe_small_inter(check):
    """I < 1024 takes other padding and crew sizes inside KOSMOS."""
    _run(check, "small_i512")
