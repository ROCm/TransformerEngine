# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import torch
from typing import Dict

from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.module.base import get_workspace
import logging


def _run_gemm_timing(name: str, gemm_parameters: Dict) -> float:
    """ Run GEMM operations and return average execution time."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    timing = []
    for _ in range(5):
        start.record()
        general_gemm(**gemm_parameters)
        end.record()
        torch.cuda.current_stream().synchronize()
        timing.append(start.elapsed_time(end))
    
    timing = timing[2:]  # Discard the initial timings to avoid warm-up effects
    mean = sum(timing) / len(timing)
    abs_dev = [abs(t - mean) for t in timing]
    logging.info(f"{name} timing: {mean} ms with deviation {sum(abs_dev)/len(timing)}")
    if any(d > 0.2 * mean for d in abs_dev):
        raise RuntimeError(f"High timing deviation detected: {timing}")
    return mean


def test_gemm_sm_count():
    """ Test Math SM count for GEMM operation by comparing performance with different SM counts."""
    M = 2304
    N = 4096
    K = 768
    datatype = torch.float32
    A = torch.randn((K, M), device="cuda", dtype=datatype)
    B = torch.randn((N, K), device="cuda", dtype=datatype)
    gemm_parameters = {'A': A, 'B': B, 'layout': "NN", 'workspace': get_workspace()}
  
    with torch.cuda.stream(torch.cuda.Stream()):
        full_timing = _run_gemm_timing("Full", gemm_parameters)
    
    cu_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    os.environ['NVTE_EXT_MARGIN_SM'] = str(95 * cu_count // 100) # Use 5% of SMs
    with torch.cuda.stream(torch.cuda.Stream()):
        constrained_timing = _run_gemm_timing("Constrained", gemm_parameters)
    
    assert full_timing * 5  < constrained_timing, "GEMM performance is not changed with constrained SM count"
