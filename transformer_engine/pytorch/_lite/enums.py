# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-Python re-declarations of C++ enum types from transformer_engine_torch."""

import enum


class DType(enum.IntEnum):
    """Data type enum matching transformer_engine::DType."""
    kByte = 0
    kInt32 = 1
    kFloat32 = 2
    kFloat16 = 3
    kBFloat16 = 4
    kFloat8E4M3 = 5
    kFloat8E5M2 = 6
    kFloat4E2M1 = 7


class NVTE_Bias_Type(enum.IntEnum):
    """Bias type for fused attention."""
    NVTE_NO_BIAS = 0
    NVTE_PRE_SCALE_BIAS = 1
    NVTE_POST_SCALE_BIAS = 2
    NVTE_ALIBI = 3


class NVTE_Mask_Type(enum.IntEnum):
    """Mask type for fused attention."""
    NVTE_NO_MASK = 0
    NVTE_PADDING_MASK = 1
    NVTE_CAUSAL_MASK = 2
    NVTE_PADDING_CAUSAL_MASK = 3
    NVTE_CAUSAL_BOTTOM_RIGHT_MASK = 4
    NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK = 5


class NVTE_Softmax_Type(enum.IntEnum):
    """Softmax type for fused attention."""
    NVTE_VANILLA_SOFTMAX = 0
    NVTE_OFF_BY_ONE_SOFTMAX = 1
    NVTE_LEARNABLE_SOFTMAX = 2


class NVTE_QKV_Format(enum.IntEnum):
    """QKV tensor format for fused attention."""
    NVTE_BSHD = 0
    NVTE_SBHD = 1
    NVTE_THD = 2
    NVTE_SBHD_2BSHD = 3
    NVTE_BSHD_2SBHD = 4
    NVTE_THD_2BSHD = 5
    NVTE_THD_2SBHD = 6


class NVTE_QKV_Layout(enum.IntEnum):
    """QKV layout for fused attention."""
    NVTE_SB3HD = 0
    NVTE_SBH3D = 1
    NVTE_SBHD_SB2HD = 2
    NVTE_SBHD_SBH2D = 3
    NVTE_SBHD_SBHD_SBHD = 4
    NVTE_BS3HD = 5
    NVTE_BSH3D = 6
    NVTE_BSHD_BS2HD = 7
    NVTE_BSHD_BSH2D = 8
    NVTE_BSHD_BSHD_BSHD = 9
    NVTE_T3HD = 10
    NVTE_TH3D = 11
    NVTE_THD_T2HD = 12
    NVTE_THD_TH2D = 13
    NVTE_THD_THD_THD = 14
    NVTE_SBHD_BSHD_BSHD = 15
    NVTE_BSHD_SBHD_SBHD = 16
    NVTE_THD_BSHD_BSHD = 17
    NVTE_THD_SBHD_SBHD = 18
    NVTE_Paged_KV_BSHD_BSHD_BSHD = 19
    NVTE_Paged_KV_BSHD_SBHD_SBHD = 20
    NVTE_Paged_KV_SBHD_BSHD_BSHD = 21
    NVTE_Paged_KV_SBHD_SBHD_SBHD = 22
    NVTE_Paged_KV_THD_BSHD_BSHD = 23
    NVTE_Paged_KV_THD_SBHD_SBHD = 24


class NVTE_Fused_Attn_Backend(enum.IntEnum):
    """Fused attention backend selection (ROCm values)."""
    NVTE_AOTriton = 0
    NVTE_CK = 1
    NVTE_No_Backend = 2
    # Lite-mode additions
    NVTE_SDPA = 100
    NVTE_Flash = 101
    # Included for API parity with the full build. Lite does not actually
    # implement an FP8 attention kernel — get_fused_attn_backend raises
    # NotImplementedError when FP8 inputs are requested (fp8_dpa=True).
    NVTE_FP8 = 200


class Float8BlockScaleTensorFormat(enum.IntEnum):
    """Block scale tensor format."""
    GEMM_READY = 0
    COMPACT = 1


class FP8FwdTensors(enum.IntEnum):
    """FP8 forward tensor indices."""
    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM1_OUTPUT = 2
    GEMM2_INPUT = 3
    GEMM2_WEIGHT = 4
    GEMM2_OUTPUT = 5
    GEMM3_INPUT = 6
    GEMM3_WEIGHT = 7
    GEMM3_OUTPUT = 8


class FP8BwdTensors(enum.IntEnum):
    """FP8 backward tensor indices."""
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
    GRAD_OUTPUT2 = 2
    GRAD_INPUT2 = 3
    GRAD_OUTPUT3 = 4
    GRAD_INPUT3 = 5


class CommOverlapType(enum.IntEnum):
    """Communication overlap type."""
    RS = 0
    AG = 1


class CommOverlapAlgo(enum.IntEnum):
    """Communication overlap algorithm."""
    BULK_OVERLAP_AG = 0
    BULK_OVERLAP_RS = 1
    SPLIT_PIPELINED_AG_P2P = 2
    SPLIT_PIPELINED_RS = 3
    SPLIT_PIPELINED_RS_P2P = 4
    ATOMIC_GEMM_RS = 5
    ATOMIC_GEMM_AG_P2P = 6
    ATOMIC_GEMM_RS_P2P = 7
    EXTERNAL_BULK_OVERLAP_AG = 8


class FP8TensorMeta:
    """FP8 tensor metadata (pure Python replacement)."""
    def __init__(self):
        self.scale = None
        self.scale_inv = None
        self.amax_history = None


class CommOverlapCore:
    """Stub for CommOverlapCore."""
    def is_atomic_gemm(self):
        return False

    def is_p2p_overlap(self):
        return False

    def is_fp8_ubuf(self):
        return False
