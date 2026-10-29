# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

import os
import torch
from typing import Dict

from transformer_engine.pytorch.cpp_extensions import general_gemm
import logging


def _run_gemm_timing(gemm_parameters: Dict, reps: int = 10) -> float:
    """ Run GEMM operations back to back and return the average time of one, in ms.

    A GPU busy-wait goes first so the launches queue up behind it: the timed window then holds
    the GEMMs' GPU time only, not the host's launch overhead, which varies with CPU load.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda._sleep(20_000_000)
    start.record()
    for _ in range(reps):
        general_gemm(**gemm_parameters)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def test_gemm_sm_count():
    """ Test Math SM count for GEMM operation by comparing performance with different SM counts."""
    M = 2304
    N = 4096
    K = 768
    datatype = torch.float32
    A = torch.randn((K, M), device="cuda", dtype=datatype)
    B = torch.randn((N, K), device="cuda", dtype=datatype)
    gemm_parameters = {'A': A, 'B': B, 'layout': "NN"}

    cu_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    margin = str(95 * cu_count // 100) # Use 5% of SMs
    general_gemm(**gemm_parameters) # Warm-up

    # Full and constrained samples alternate, so a clock change during the test moves both. Anything
    # else running only ever adds time, so the fastest sample of each is the one to compare
    full, constrained = [], []
    try:
        for _ in range(7):
            os.environ.pop('NVTE_EXT_MARGIN_SM', None)
            full.append(_run_gemm_timing(gemm_parameters))
            os.environ['NVTE_EXT_MARGIN_SM'] = margin
            constrained.append(_run_gemm_timing(gemm_parameters))
    finally:
        os.environ.pop('NVTE_EXT_MARGIN_SM', None)
    full_timing = min(full)
    constrained_timing = min(constrained)
    logging.info(f"Full timing: {full_timing} ms, constrained timing: {constrained_timing} ms")

    assert full_timing * 5  < constrained_timing, "GEMM performance is not changed with constrained SM count"
