# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


import os

import torch
import triton


def get_ln_sm_margin(sm_margin_type):
    assert sm_margin_type in {"FWD", "BWD", "INF"}
    try:
        sm_margin = max(
            int(os.getenv(f"NVTE_{sm_margin_type}_LAYERNORM_SM_MARGIN", "0")), 0
        )
    except ValueError:
        sm_margin = 0
    assert sm_margin >= 0
    return sm_margin


def get_fwd_ln_sm_margin():
    return get_ln_sm_margin("FWD")


def get_bwd_ln_sm_margin():
    return get_ln_sm_margin("BWD")


def get_inf_ln_sm_margin():
    return get_ln_sm_margin("INF")


def get_num_sms(sm_margin=None):
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if sm_margin is not None and sm_margin > 0:
        num_sms = max(num_sms - int(sm_margin), 1)
    return num_sms


def block_size(x):
    max_fused_size = 65536 // x.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(x.shape[1]))
    return block_size


def use_blocked(x):
    return x.shape[1] > block_size(x)
