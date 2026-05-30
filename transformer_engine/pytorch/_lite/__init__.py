# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Transformer Engine Lite -- Pure-Python drop-in replacement for transformer_engine_torch.

This module provides the same API surface as the compiled C++ extension but uses
Triton kernels, AITER, and PyTorch-native operations instead.

Activate by setting NVTE_LITE=1 before importing transformer_engine.
"""

# Re-export all enums and types
from .enums import (
    DType,
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_Softmax_Type,
    NVTE_QKV_Format,
    NVTE_QKV_Layout,
    NVTE_Fused_Attn_Backend,
    Float8BlockScaleTensorFormat,
    FP8FwdTensors,
    FP8BwdTensors,
    CommOverlapType,
    CommOverlapAlgo,
    FP8TensorMeta,
    CommOverlapCore,
)

# Re-export operation implementations
from .activations import (
    gelu, geglu, qgelu, qgeglu,
    relu, reglu, srelu, sreglu,
    silu, swiglu, clamped_swiglu,
    dgelu, dgeglu, dqgelu, dqgeglu,
    drelu, dreglu, dsrelu, dsreglu,
    dsilu, dswiglu, clamped_dswiglu,
    dbias_dgelu, dbias_dsilu, dbias_drelu, dbias_dqgelu, dbias_dsrelu,
)
from .norms import (
    layernorm_fwd, layernorm_bwd,
    rmsnorm_fwd, rmsnorm_bwd, rmsnorm_bwd_add,
)
from .quantize import (
    quantize, dequantize, bgrad_quantize,
    multi_tensor_quantize, split_quantize,
    compute_amax, fused_amax_and_scale_update_after_reduction,
    fp8_block_scaling_compute_partial_amax, fp8_block_scaling_partial_cast,
)
from .gemm import generic_gemm
from .grouped_gemm import te_general_grouped_gemm
from .softmax import (
    scaled_softmax_forward, scaled_softmax_backward,
    scaled_masked_softmax_forward, scaled_masked_softmax_backward,
    scaled_upper_triang_masked_softmax_forward, scaled_upper_triang_masked_softmax_backward,
    scaled_aligned_causal_masked_softmax_forward, scaled_aligned_causal_masked_softmax_backward,
)
from .attention import (
    get_fused_attn_backend,
    fused_attn_fwd, fused_attn_bwd,
    fa_prepare_fwd, fa_prepare_bwd,
    copy_to_kv_cache,
    convert_thd_to_bshd, convert_bshd_to_thd,
)
from .rope import (
    fused_rope_forward, fused_rope_backward,
    fused_qkv_rope_forward, fused_qkv_rope_backward,
)
from .dropout import dropout_fwd, dropout_bwd
from .transpose import fp8_transpose, swap_first_dims
from .permutation import (
    moe_permute_fwd, moe_permute_bwd,
    moe_unpermute_fwd, moe_unpermute_bwd,
)
from .multi_tensor import (
    multi_tensor_scale, multi_tensor_l2norm, multi_tensor_unscale_l2norm,
    multi_tensor_adam, multi_tensor_adam_param_remainder,
    multi_tensor_adam_fp8,
    multi_tensor_adam_capturable, multi_tensor_adam_capturable_master,
    multi_tensor_sgd,
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)
from .router import (
    fused_topk_with_score_function_fwd, fused_topk_with_score_function_bwd,
    fused_score_for_moe_aux_loss_fwd, fused_score_for_moe_aux_loss_bwd,
    fused_moe_aux_loss_fwd, fused_moe_aux_loss_bwd,
)
from .comm import (
    CommOverlapHelper, CommOverlap, CommOverlapP2P,
    CommOverlapBase, CommOverlapP2PBase,
    bulk_overlap_ag_with_external_gemm,
    init_nvshmem_backend, create_nvshmem_tensor,
    nvshmem_send_on_current_stream, nvshmem_wait_on_current_stream, nvshmem_finalize,
    device_supports_multicast, get_stream_priority_range, ubuf_built_with_mpi,
)
from .misc import get_num_cublas_streams
from .context_parallel import (
    thd_read_half_tensor, thd_second_half_lse_correction,
    thd_read_second_half_lse, thd_out_correction,
    thd_grad_correction, thd_get_partitioned_indices,
)
from .mori_ep import (
    mori_ep_available,
    init_mori_ep,
    finalize_mori_ep,
    is_mori_ep_initialized,
    mask_to_index,
    index_to_mask,
    MoriExpertParallel,
    MoriEPDispatch,
    MoriEPCombine,
    MoriEPDispatchStdMoE,
    MoriEPCombineStdMoE,
)
from .padding import fused_multi_row_padding, fused_multi_row_unpadding

# Note: fused_layernorm_linear and fused_layernorm_mlp are NOT imported here
# because they import `transformer_engine_torch as tex` which resolves to this
# module, creating a circular import. They are accessed via
# transformer_engine.pytorch.module.__init__ when NVTE_LITE=1.
