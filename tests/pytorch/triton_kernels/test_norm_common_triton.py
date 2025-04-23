# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information


from transformer_engine.pytorch.triton_kernels.norm_common_triton import get_num_sms


def test_sm_margin():
    num_sms = get_num_sms()
    assert num_sms > 0
    assert get_num_sms(0) == num_sms
    assert get_num_sms(-5) == num_sms
    assert get_num_sms(1) == num_sms - 1
    assert get_num_sms(100 * num_sms) == 1
