#!/usr/bin/python3
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

import os
from typing import List
import pytest
import subprocess
from pathlib import Path
from transformer_engine.pytorch import torch_version
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import torch
from run_fsdp2_fp8_model import SimpleNet

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

NUM_PROCS: int = torch.cuda.device_count()

def assert_allclose(
    l1: List[torch.Tensor], l2: List[torch.Tensor], atol: float, rtol: float = None
) -> bool:
    """Ensures two lists are equal."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    for i, (t1, t2) in enumerate(zip(l1, l2)):
        tols = dict(atol=atol)
        if rtol is not None:
            tols["rtol"] = rtol
        result = torch.allclose(t1, t2, **tols)
        if not result:
            diff = torch.abs(t1 - t2)
            tol = atol + (rtol * torch.abs(t2))
            exceed_mask = diff > tol
            if exceed_mask.any():
                indices = torch.nonzero(exceed_mask, as_tuple=True)
                max_diff = diff[exceed_mask].max()
                max_idx = (diff[exceed_mask] == max_diff).nonzero(as_tuple=True)[0][0]
                max_location = [idx[max_idx].item() for idx in indices]
                msg = (
                    f"Outputs not close enough in tensor at idx={i}. "
                    f"Maximum difference at location {max_location} "
                    f"with {t1[exceed_mask][max_idx].item()} vs {t2[exceed_mask][max_idx].item()} "
                    f"(diff {max_diff.item()})."
                )
            raise AssertionError(msg)

def _run_test(fp_init, recipe):
    test_dir = Path(__file__).parent.resolve()
    fsdp_script = test_dir / "run_fsdp2_fp8_model.py"
    
    test_cmd = ["torchrun", f"--nproc_per_node={NUM_PROCS}", "--master-port=29501", str(fsdp_script)]

    if fp_init:
        test_cmd += ["--fp8-init"]
    test_cmd += ["--recipe", recipe]
    
    subprocess.run(test_cmd + ['--use-fsdp2','--gradients-save-file', 'all_iters_fsdp2.pt'], env=os.environ, check=True)
    subprocess.run(test_cmd + ['--gradients-save-file', 'all_iters_dp.pt'], env=os.environ, check=True)
        
    # Load outputs
    output_fsdp = torch.load("all_iters_fsdp2.pt", map_location="cpu")
    output_dp = torch.load("all_iters_dp.pt", map_location="cpu")
    
    for idx, (te_output_no_cache, te_output_cache) in enumerate(zip(output_fsdp, output_dp)):
    
        print(f"Comparing FSDP {te_output_no_cache[0]}, DDP {te_output_cache[0]} at index {idx}...")
        assert_allclose(te_output_no_cache[1], te_output_cache[1], atol=0, rtol=0)
        print(f"Tensor at index {idx} passed comparison.")


@pytest.fixture
def cleanup_artifacts():
    yield  # run the test first
    for fname in ["all_iters_fsdp2.pt", "all_iters_dp.pt", "fsdp_model.pth", "shared_input.pt"]:
        if os.path.exists(fname):
            os.remove(fname)

@pytest.mark.skipif(NUM_PROCS < 4, reason="Requires 4+ GPUs")
@pytest.mark.skipif(NUM_PROCS % 2 != 0, reason="Requires even number of GPUs")
@pytest.mark.skipif(not torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
@pytest.mark.parametrize("fp8_init", ([False]))
@pytest.mark.parametrize("recipe", (["delayed", "current", "mxfp8"]))
@pytest.mark.usefixtures("cleanup_artifacts")
def test_distributed(fp8_init, recipe):

    batch_size = 2048
    input_size = 2048
    from pathlib import Path

    input_path = Path("shared_input.pt")
    if input_path.exists():
        input_data = torch.load(input_path).to('cuda')
    else:
        input_data = torch.randn(batch_size, input_size, requires_grad=True).to('cuda')
        torch.save(input_data.cpu(), input_path)
        print("Generated and saved shared input tensor.")

    model = SimpleNet(input_size, 2048, 2048)
    torch.save(model.state_dict(), 'fsdp_model.pth')

    if torch.cuda.device_count() < 4:
        pytest.skip("FSDP2 test requires at least 4 GPUs")

    if fp8_init and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if recipe == "mxfp8" and not mxfp8_available:  
        pytest.skip(reason_for_no_mxfp8)

    _run_test(fp8_init, recipe)


def test_dummy() -> None:
    """Dummy test

    pytest returns exit code 5 if all tests are skipped.

    """
    pass
