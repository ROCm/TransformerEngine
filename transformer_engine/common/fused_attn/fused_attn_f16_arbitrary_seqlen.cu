/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "fused_attn_f16_arbitrary_seqlen.h"

#ifndef __HIP_PLATFORM_AMD__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>
#endif
#include <map>
#include <vector>

#include "../common.h"
#include "utils.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"

#ifdef __HIP_PLATFORM_AMD__
#define DISPATCH_BF16_AND_F16_TYPES(DATATYPE, NAME, ...)                       \
    switch (DATATYPE) {                                                        \
    case transformer_engine::DType::kBFloat16: {                               \
        using scalar_t = ck_tile::bf16_t;                                      \
        __VA_ARGS__;                                                           \
    } break;                                                                   \
    case transformer_engine::DType::kFloat16: {                                \
        using scalar_t = ck_tile::half_t;                                      \
        __VA_ARGS__;                                                           \
    } break;                                                                   \
    default:                                                                   \
        break;                                                                 \
    }

const std::unordered_map<transformer_engine::DType, std::string> dtype_mapping =
    {{transformer_engine::DType::kBFloat16, "bf16"},
     {transformer_engine::DType::kFloat16, "fp16"}};


namespace transformer_engine {
namespace fused_attn {

template <typename DataType>
void fused_attn_arbitrary_seqlen_fwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
    int64_t bias_b, int64_t bias_h, bool is_training, float scaling_factor,
    float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
    NVTE_QKV_Layout layout, NVTE_Bias_Type bias_value,
    NVTE_Mask_Type mask_value, void* devPtrQ, void* devPtrK, void* devPtrV,
    void* devPtrBias, void* devPtrSoftmaxStats, void* devPtrO,
    void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
    const std::string& data_type, hipStream_t stream) {
    using TypeConfig = FmhaFwdTypeConfig<DataType>;
    using QDataType = typename TypeConfig::QDataType;
    using KDataType = typename TypeConfig::KDataType;
    using VDataType = typename TypeConfig::VDataType;
    using BiasDataType = typename TypeConfig::BiasDataType;
    using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
    using LSEDataType = typename TypeConfig::LSEDataType;
    using SaccDataType = typename TypeConfig::SaccDataType;
    using SMPLComputeDataType = typename TypeConfig::SMPLComputeDataType;
    using PDataType = typename TypeConfig::PDataType;
    using OaccDataType = typename TypeConfig::OaccDataType;
    using ODataType = typename TypeConfig::ODataType;

    bool has_dropout = (is_training && dropout_probability > 0.f);
    bool has_lse = (devPtrSoftmaxStats != nullptr);

    /* CK input parameters */
    ck_tile::index_t batch = b;
    ck_tile::index_t seqlen_q = s_q;
    ck_tile::index_t nhead = h;
    ck_tile::index_t hdim_q = d;
    ck_tile::index_t seqlen_k = s_kv;
    ck_tile::index_t nhead_k = hg;
    ck_tile::index_t hdim_v = d;
    ck_tile::index_t max_seqlen_q = s_q;
    ck_tile::index_t max_seqlen_k = s_kv;
    float scale_s = scaling_factor;
    float scale_p = 1.f;
    float scale_o = 1.f;
    float p_drop = dropout_probability;
    bool is_group_mode = false;
    bool is_v_rowmajor = true;
    bool do_fp8_static_quant = false;

    bias_enum bias_type;
    mask_enum mask_type;
    ck_tile::index_t left, right;
    ck_tile::stream_config stream_config{stream};
    if (hdim_q % 8 != 0) {
        NVTE_ERROR("Invalid head dimension: hdim_q must be a multiple of 8");
    }
    if (hdim_v % 8 != 0) {
        NVTE_ERROR("Invalid head dimension: hdim_v must be a multiple of 8");
    }
    if (bias_value == NVTE_Bias_Type::NVTE_NO_BIAS) {
        bias_type = bias_enum::no_bias;
    } else if (bias_value == NVTE_Bias_Type::NVTE_ALIBI) {
        bias_type = bias_enum::alibi;
    } else {
        NVTE_ERROR("Unsupported bias type");
    }

    if (mask_value == NVTE_Mask_Type::NVTE_NO_MASK) {
        mask_type = mask_enum::no_mask;
    } else if (mask_value == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
               mask_value == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) {
        mask_type = mask_enum::mask_top_left;
        left = -1;
        right = 0;
    } else {
        NVTE_ERROR("Unsupported mask type");
    }

    try {
        ck_tile::index_t shape_batch = batch;
        ck_tile::index_t shape_seqlen_q = seqlen_q;
        ck_tile::index_t shape_seqlen_k = seqlen_k;
        bool s_randval = false;

        auto fmha_traits = fmha_fwd_traits{
            hdim_q,    hdim_v,    data_type, is_group_mode, is_v_rowmajor,
            mask_type, bias_type, has_lse,   has_dropout,   do_fp8_static_quant};

        auto fmha_args = [&]() {
            // setup stride_* arguments
            const ck_tile::index_t stride_q = nhead * hdim_q;
            const ck_tile::index_t stride_k = nhead_k * hdim_q;
            const ck_tile::index_t stride_v = nhead_k * hdim_v;
            const ck_tile::index_t stride_bias = shape_seqlen_k;
            const ck_tile::index_t stride_randval = max_seqlen_k;
            const ck_tile::index_t stride_o = nhead * hdim_v;
            // setup nhead_stride_* arguments
            const ck_tile::index_t nhead_stride_q = hdim_q;
            const ck_tile::index_t nhead_stride_k = hdim_q;
            const ck_tile::index_t nhead_stride_v = hdim_v;
            const ck_tile::index_t nhead_stride_bias = 0;
            const ck_tile::index_t nhead_stride_randval =
                shape_seqlen_q * max_seqlen_k;
            const ck_tile::index_t nhead_stride_lse = shape_seqlen_q;
            const ck_tile::index_t nhead_stride_o = hdim_v;
            // setup batch_stride_* arguments
            const ck_tile::index_t batch_stride_q = nhead * shape_seqlen_q * hdim_q;
            const ck_tile::index_t batch_stride_k =
                nhead_k * shape_seqlen_k * hdim_q;
            const ck_tile::index_t batch_stride_v =
                nhead_k * shape_seqlen_k * hdim_v;
            const ck_tile::index_t batch_stride_bias = 0;
            const ck_tile::index_t batch_stride_randval =
                nhead * shape_seqlen_q * max_seqlen_k;
            const ck_tile::index_t batch_stride_lse = nhead * shape_seqlen_q;
            const ck_tile::index_t batch_stride_o = nhead * shape_seqlen_q * hdim_v;

            return fmha_fwd_args{devPtrQ,
                                 devPtrK,
                                 devPtrV,
                                 devPtrBias,
                                 nullptr,
                                 devPtrSoftmaxStats,
                                 devPtrO,
                                 devPtrCuSeqlensQ,
                                 devPtrCuSeqlensKV,
                                 nullptr, /* seqlen_k_ptr */
                                 shape_seqlen_q,
                                 shape_seqlen_k,
                                 batch,
                                 max_seqlen_q,
                                 hdim_q,
                                 hdim_v,
                                 nhead,
                                 nhead_k,
                                 scale_s,
                                 scale_p,
                                 scale_o,
                                 stride_q,
                                 stride_k,
                                 stride_v,
                                 stride_bias,
                                 stride_randval,
                                 stride_o,
                                 nhead_stride_q,
                                 nhead_stride_k,
                                 nhead_stride_v,
                                 nhead_stride_bias,
                                 nhead_stride_randval,
                                 nhead_stride_lse,
                                 nhead_stride_o,
                                 batch_stride_q,
                                 batch_stride_k,
                                 batch_stride_v,
                                 batch_stride_bias,
                                 batch_stride_randval,
                                 batch_stride_lse,
                                 batch_stride_o,
                                 left,
                                 right,
                                 static_cast<ck_tile::index_t>(mask_type),
                                 p_drop,
                                 s_randval,
                                 {drop_seed, drop_offset}};
        }();

        fmha_fwd(fmha_traits, fmha_args, stream_config);
    } catch (std::runtime_error &e) {
        NVTE_ERROR(e.what());
    }
}

template<typename DataType>
__global__ void reshape_and_sum(
    DataType* dk, const DataType* dk_expanded,
    DataType* dv, const DataType* dv_expanded,
    int batch_size,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_size) {
    static_assert(std::is_arithmetic<DataType>::value,
                  "reshape_and_sum only supports arithmetic types");
}

template<>
__global__ void reshape_and_sum<ck_tile::half_t>(
    ck_tile::half_t* dk, const ck_tile::half_t* dk_expanded,
    ck_tile::half_t* dv, const ck_tile::half_t* dv_expanded,
    int batch_size,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_size) {
    int batch_idx = blockIdx.x;
    int seqlen_idx = blockIdx.y;
    int head_k_idx = blockIdx.z;
    int thread_idx = threadIdx.x;
    int head_idx_offset = num_heads / num_heads_k;

    if (thread_idx < head_size) {
        float sum_dk = 0.0f;
        float sum_dv = 0.0f;
        int read_idx = ((batch_idx * seqlen_k + seqlen_idx) *
                         num_heads_k + head_k_idx) *
                         head_idx_offset * head_size + thread_idx;
        int write_idx = ((batch_idx * seqlen_k + seqlen_idx) *
                          num_heads_k + head_k_idx) *
                          head_size + thread_idx;
        for (int j = 0; j < head_idx_offset; j++) {
            sum_dk += dk_expanded[read_idx];
            sum_dv += dv_expanded[read_idx];
            read_idx += head_size;
        }
        dk[write_idx] = sum_dk;
        dv[write_idx] = sum_dv;
    }
}

template<>
__global__ void reshape_and_sum<ck_tile::bf16_t>(
    ck_tile::bf16_t* dk, const ck_tile::bf16_t* dk_expanded,
    ck_tile::bf16_t* dv, const ck_tile::bf16_t* dv_expanded,
    int batch_size,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_size) {
    int batch_idx = blockIdx.x;
    int seqlen_idx = blockIdx.y;
    int head_k_idx = blockIdx.z;
    int thread_idx = threadIdx.x;
    int head_idx_offset = num_heads / num_heads_k;

    if (thread_idx < head_size) {
        float sum_dk = 0.0f;
        float sum_dv = 0.0f;
        int read_idx = ((batch_idx * seqlen_k + seqlen_idx) *
                         num_heads_k + head_k_idx) *
                         head_idx_offset * head_size + thread_idx;
        int write_idx = ((batch_idx * seqlen_k + seqlen_idx) *
                          num_heads_k + head_k_idx) *
                          head_size + thread_idx;
        for (int j = 0; j < head_idx_offset; j++) {
            sum_dk += ck_tile::bf16_to_float(dk_expanded[read_idx]);
            sum_dv += ck_tile::bf16_to_float(dv_expanded[read_idx]);
            read_idx += head_size;
        }
        dk[write_idx] = ck_tile::float_to_bf16(sum_dk);
        dv[write_idx] = ck_tile::float_to_bf16(sum_dv);
    }
}

template <typename DataType>
void fused_attn_arbitrary_seqlen_bwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
    int64_t bias_b, int64_t bias_h, float scaling_factor,
    float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
    NVTE_QKV_Layout layout, NVTE_Bias_Type bias_value,
    NVTE_Mask_Type mask_value, void* devPtrQ, void* devPtrKTranspose,
    void* devPtrVTranspose, void* devPtrO, void* devPtrSoftmaxStats,
    void* devPtrBias, void* devPtrdQ, void* devPtrdK, void* devPtrdV,
    void* devPtrdO, void* devPtrdBias, void* devPtrCuSeqlensQ,
    void* devPtrCuSeqlensKV, const std::string& data_type, hipStream_t stream) {
    using TypeConfig = FmhaBwdTypeConfig<DataType>;
    using QDataType = typename TypeConfig::QDataType;
    using KDataType = typename TypeConfig::KDataType;
    using VDataType = typename TypeConfig::VDataType;
    using GemmDataType = typename TypeConfig::GemmDataType;
    using BiasDataType = typename TypeConfig::BiasDataType;
    using LSEDataType = typename TypeConfig::LSEDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using DDataType = typename TypeConfig::DDataType;
    using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
    using ODataType = typename TypeConfig::ODataType;
    using OGradDataType = typename TypeConfig::OGradDataType;
    using QGradDataType = typename TypeConfig::QGradDataType;
    using KGradDataType = typename TypeConfig::KGradDataType;
    using VGradDataType = typename TypeConfig::VGradDataType;
    using BiasGradDataType = typename TypeConfig::BiasGradDataType;

    bool is_mqa_gqa = (h > hg);
    bool has_dropout = (dropout_probability > 0.f);
    bool has_dbias = (devPtrdBias != nullptr);

    /* CK input parameters */
    ck_tile::index_t batch = b;
    ck_tile::index_t seqlen_q = s_q;
    ck_tile::index_t nhead = h;
    ck_tile::index_t hdim_q = d;
    ck_tile::index_t seqlen_k = s_kv;
    ck_tile::index_t nhead_k = hg;
    ck_tile::index_t hdim_v = d;
    ck_tile::index_t max_seqlen_q = s_q;
    ck_tile::index_t max_seqlen_k = s_kv;
    float scale_s = scaling_factor;
    float p_drop = dropout_probability;
    float p_undrop = 1.0 - p_drop;
    float rp_undrop = 1.0 / p_undrop;
    bool is_group_mode = false;
    bool do_fp8_static_quant = false;

    bias_enum bias_type;
    mask_enum mask_type;
    int32_t left, right;
    ck_tile::stream_config stream_config{stream};
    static thread_local DeviceMemoryManager d_mgr{stream};
    static thread_local DeviceMemoryManager dk_expanded_mgr{stream};
    static thread_local DeviceMemoryManager dv_expanded_mgr{stream};

    if (bias_value == NVTE_Bias_Type::NVTE_NO_BIAS) {
        bias_type = bias_enum::no_bias;
    } else if (bias_value == NVTE_Bias_Type::NVTE_ALIBI) {
        bias_type = bias_enum::alibi;
    } else {
        NVTE_ERROR("Unsupported bias type");
    }

    if (mask_value == NVTE_Mask_Type::NVTE_NO_MASK) {
        mask_type = mask_enum::no_mask;
    } else if (mask_value == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
               mask_value == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) {
        mask_type = mask_enum::mask_top_left;
        left = -1;
        right = 0;
    } else {
        NVTE_ERROR("Unsupported mask type");
    }

    try {
        ck_tile::index_t shape_batch = batch;
        ck_tile::index_t shape_seqlen_q = seqlen_q;
        ck_tile::index_t shape_seqlen_k = seqlen_k;
        bool s_randval = false;

        d_mgr.resize(sizeof(DDataType) * batch * nhead * max_seqlen_q);
        if (is_mqa_gqa) {
            dk_expanded_mgr.resize(sizeof(KGradDataType) * batch * nhead * shape_seqlen_k * hdim_q);
            dv_expanded_mgr.resize(sizeof(VGradDataType) * batch * nhead * shape_seqlen_k * hdim_v);
        }
        HIP_CHECK_ERROR(
            hipMemsetAsync(devPtrdQ,
                           0,
                           sizeof(QGradDataType) * batch * max_seqlen_q * nhead * hdim_q,
                           stream));

        auto fmha_traits =
            fmha_bwd_traits{hdim_q,    hdim_v,    data_type, is_group_mode,
                            mask_type, bias_type, has_dbias, has_dropout};

        auto fmha_args = [&]() {
            // setup stride_* arguments
            const ck_tile::index_t stride_q = nhead * hdim_q;
            const ck_tile::index_t stride_k = nhead_k * hdim_q;
            const ck_tile::index_t stride_v = nhead_k * hdim_v;
            const ck_tile::index_t stride_bias = max_seqlen_k;
            const ck_tile::index_t stride_o = nhead * hdim_v;
            const ck_tile::index_t stride_randval = max_seqlen_k;
            const ck_tile::index_t stride_do = nhead * hdim_v;
            const ck_tile::index_t stride_dk = nhead * hdim_q;
            const ck_tile::index_t stride_dv = nhead * hdim_v;
            const ck_tile::index_t stride_dbias = nhead * max_seqlen_k;
            // setup nhead_stride_* arguments
            const ck_tile::index_t nhead_stride_q = hdim_q;
            const ck_tile::index_t nhead_stride_k = hdim_q;
            const ck_tile::index_t nhead_stride_v = hdim_v;
            const ck_tile::index_t nhead_stride_bias = 0;
            const ck_tile::index_t nhead_stride_o = hdim_v;
            const ck_tile::index_t nhead_stride_randval = shape_seqlen_q * max_seqlen_k;
            const ck_tile::index_t nhead_stride_do = hdim_v;
            const ck_tile::index_t nhead_stride_lsed = max_seqlen_q;
            const ck_tile::index_t nhead_stride_dbias = max_seqlen_k;
            // setup batch_stride_* arguments
            const ck_tile::index_t batch_stride_q = nhead * shape_seqlen_q * hdim_q;
            const ck_tile::index_t batch_stride_k = nhead_k * shape_seqlen_k * hdim_q;
            const ck_tile::index_t batch_stride_v = nhead_k * shape_seqlen_k * hdim_v;
            const ck_tile::index_t batch_stride_bias = 0;
            const ck_tile::index_t batch_stride_o = nhead * shape_seqlen_q * hdim_v;
            const ck_tile::index_t batch_stride_randval = nhead * shape_seqlen_q * max_seqlen_k;
            const ck_tile::index_t batch_stride_do = nhead * shape_seqlen_q * hdim_v;
            const ck_tile::index_t batch_stride_lsed = nhead * max_seqlen_q;
            const ck_tile::index_t batch_stride_dk = nhead * shape_seqlen_k * hdim_q;
            const ck_tile::index_t batch_stride_dv = nhead * shape_seqlen_k * hdim_v;
            const ck_tile::index_t batch_stride_dbias = nhead * shape_seqlen_q * max_seqlen_k;

            return fmha_bwd_args{devPtrQ,
                                 devPtrKTranspose,
                                 devPtrVTranspose,
                                 devPtrBias,
                                 devPtrO,
                                 devPtrSoftmaxStats,
                                 devPtrdO,
                                 d_mgr.get_allocated_block(),
                                 nullptr,
                                 devPtrdQ,
                                 is_mqa_gqa ? dk_expanded_mgr.get_allocated_block() : devPtrdK,
                                 is_mqa_gqa ? dv_expanded_mgr.get_allocated_block() : devPtrdV,
                                 devPtrdBias,
                                 devPtrCuSeqlensQ,
                                 devPtrCuSeqlensKV,
                                 nullptr, /* seqlen_k_ptr */
                                 shape_seqlen_q,
                                 shape_seqlen_k,
                                 batch,
                                 max_seqlen_q,
                                 max_seqlen_k,
                                 hdim_q,
                                 hdim_v,
                                 nhead,
                                 nhead_k,
                                 scale_s,
                                 stride_q,
                                 stride_k,
                                 stride_v,
                                 stride_bias,
                                 stride_o,
                                 stride_randval,
                                 stride_do,
                                 stride_dk,
                                 stride_dv,
                                 stride_dbias,
                                 nhead_stride_q,
                                 nhead_stride_k,
                                 nhead_stride_v,
                                 nhead_stride_bias,
                                 nhead_stride_o,
                                 nhead_stride_randval,
                                 nhead_stride_do,
                                 nhead_stride_lsed,
                                 nhead_stride_dbias,
                                 batch_stride_q,
                                 batch_stride_k,
                                 batch_stride_v,
                                 batch_stride_bias,
                                 batch_stride_o,
                                 batch_stride_randval,
                                 batch_stride_do,
                                 batch_stride_lsed,
                                 batch_stride_dk,
                                 batch_stride_dv,
                                 batch_stride_dbias,
                                 left,
                                 right,
                                 static_cast<ck_tile::index_t>(mask_type),
                                 p_drop,
                                 p_undrop,
                                 s_randval,
                                 {drop_seed, drop_offset}};
        }();

        fmha_bwd(fmha_traits, fmha_args, stream_config);

        if (is_mqa_gqa) {
            dim3 grid(batch, seqlen_k, nhead_k);
            dim3 block(hdim_q);

            // Launch the kernel for devPtrdK & devPtrdV
            hipLaunchKernelGGL(reshape_and_sum<KGradDataType>, grid, block, 0, stream,
                static_cast<KGradDataType*>(devPtrdK),
                static_cast<KGradDataType*>(dk_expanded_mgr.get_allocated_block()),
                static_cast<VGradDataType*>(devPtrdV),
                static_cast<VGradDataType*>(dv_expanded_mgr.get_allocated_block()),
                batch,
                seqlen_k,
                nhead,
                nhead_k,
                hdim_q);
        }
    } catch (std::runtime_error &e) {
        NVTE_ERROR(e.what());
    }
}
} // namespace fused_attn

using namespace transformer_engine::fused_attn;
void fused_attn_arbitrary_seqlen_fwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
    bool is_training, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor* input_Q, const Tensor* input_K,
    const Tensor* input_V, const Tensor* input_Bias, Tensor* output_O,
    NVTETensorPack* Aux_CTX_Tensors, const Tensor* cu_seqlens_q,
    const Tensor* cu_seqlens_kv, const Tensor* rng_state,
    hipStream_t stream) {
    using namespace transformer_engine;

    const auto QKV_type = input_Q->data.dtype;
    void* devPtrQ = input_Q->data.dptr;
    void* devPtrK = input_K->data.dptr;
    void* devPtrV = input_V->data.dptr;
    void* devPtrO = output_O->data.dptr;
    void* devPtrS = nullptr;
    void* devPtrBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    void* devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void* devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    if (Aux_CTX_Tensors->size == 0) {
        Aux_CTX_Tensors->size = 2;
        Tensor* output_S =
            reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
        output_S->data.dptr = nullptr;
        output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
        output_S->data.dtype = DType::kFloat32;
        Tensor* output_rng_state =
            reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = nullptr;
        output_rng_state->data.shape = {2};
        output_rng_state->data.dtype = DType::kInt64;
    } else if (Aux_CTX_Tensors->size == 2) {
        Tensor* output_S =
            reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor* output_rng_state =
            reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
    } else if (Aux_CTX_Tensors->size == 3) {
        Tensor* output_S =
            reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor* output_rng_state =
            reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
        Tensor* output_bias =
            reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
        output_bias->data.dptr = devPtrBias;
    } else {
        NVTE_ERROR("Unexpected Aux_CTX_Tensors->size");
    }

    void* devPtrRngState = rng_state->data.dptr;
    uint64_t* host_rng_state = new uint64_t[2];
    HIP_CHECK_ERROR(hipMemcpyAsync(host_rng_state, devPtrRngState, 2 * sizeof(uint64_t),
                                   hipMemcpyDeviceToHost, stream));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));
    uint64_t drop_seed = host_rng_state[0];
    uint64_t drop_offset = host_rng_state[1];
    delete[] host_rng_state;

    /*
    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);
    */

    auto iter = dtype_mapping.find(QKV_type);
    if (iter == dtype_mapping.cend()) {
        NVTE_ERROR("Unexpected QKV_type");
    }

    DISPATCH_BF16_AND_F16_TYPES(
        QKV_type, "fused_attn_arbitrary_seqlen_fwd_impl",
        fused_attn_arbitrary_seqlen_fwd_impl<scalar_t>(
            batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv,
            head_dim, bias_b, bias_h, is_training, attn_scale, p_dropout,
            drop_seed, drop_offset, qkv_layout, bias_type, mask_type, devPtrQ,
            devPtrK, devPtrV, devPtrBias, devPtrS, devPtrO, devPtrCuSeqlensQ,
            devPtrCuSeqlensKV, iter->second, stream);)
}

void fused_attn_arbitrary_seqlen_bwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, const Tensor* input_Q,
    const Tensor* input_K, const Tensor* input_V, const Tensor* input_O,
    const Tensor* input_dO, const Tensor* input_Bias, Tensor* output_S,
    Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
    Tensor* output_dBias, const Tensor* cu_seqlens_q,
    const Tensor* cu_seqlens_kv, const Tensor* rng_state,
    hipStream_t stream) {
    using namespace transformer_engine;

    const auto QKV_type = input_Q->data.dtype;
    void* devPtrQ = input_Q->data.dptr;
    void* devPtrK = input_K->data.dptr;
    void* devPtrV = input_V->data.dptr;
    void* devPtrO = input_O->data.dptr;
    void* devPtrdO = input_dO->data.dptr;
    void* devPtrBias = nullptr;
    void* devPtrdBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) &&
        (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
        devPtrBias = input_Bias->data.dptr;
        devPtrdBias = output_dBias->data.dptr;
        bias_b = output_dBias->data.shape[0];
        bias_h = output_dBias->data.shape[1];
    }

    void* devPtrdQ = output_dQ->data.dptr;
    void* devPtrdK = output_dK->data.dptr;
    void* devPtrdV = output_dV->data.dptr;
    void* devPtrSoftmaxStats = output_S->data.dptr;
    void* devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void* devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;
    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void*>(
        reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    void* devPtrRngState = rng_state->data.dptr;
    uint64_t* host_rng_state = new uint64_t[2];
    HIP_CHECK_ERROR(hipMemcpyAsync(host_rng_state, devPtrRngState, 2 * sizeof(uint64_t),
                                   hipMemcpyDeviceToHost, stream));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));
    uint64_t drop_seed = host_rng_state[0];
    uint64_t drop_offset = host_rng_state[1];
    delete[] host_rng_state;

    auto iter = dtype_mapping.find(QKV_type);
    if (iter == dtype_mapping.cend()) {
        NVTE_ERROR("Unexpected QKV_type");
    }

    DISPATCH_BF16_AND_F16_TYPES(
        QKV_type, "fused_attn_arbitrary_seqlen_bwd_impl",
        fused_attn_arbitrary_seqlen_bwd_impl<scalar_t>(
            batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv,
            head_dim, bias_b, bias_h, attn_scale, p_dropout, drop_seed,
            drop_offset, qkv_layout, bias_type, mask_type, devPtrQ, devPtrK,
            devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias, devPtrdQ,
            devPtrdK, devPtrdV, devPtrdO, devPtrdBias, devPtrCuSeqlensQ,
            devPtrCuSeqlensKV, iter->second, stream);)
}
} // namespace transformer_engine
#else
#if (CUDNN_VERSION >= 8900)
#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

namespace transformer_engine {
namespace fused_attn {
void fused_attn_arbitrary_seqlen_fwd_impl(
                int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                int64_t bias_b, int64_t bias_h,
                bool is_training, float scaling_factor, float dropout_probability,
                NVTE_QKV_Layout layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                void *devPtrQ, void *devPtrK, void *devPtrV, void *devPtrBias,
                void *devPtrSoftmaxStats, void *devPtrO,
                void* devPtrDropoutSeed, void* devPtrDropoutOffset,
                void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
                cudnn_frontend::DataType_t tensorType,
                void *workspace, size_t *workspace_size,
                cudaStream_t stream, cudnnHandle_t handle) {
    bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
    bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
    bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK)
        || (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
    bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK)
        || (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
    bool is_dropout = (is_training && dropout_probability != 0.0f);

    try {
        FADescriptor_v1 descriptor{b,                   h,
                                   hg,                  s_q,
                                   s_kv,                d,
                                   scaling_factor,      is_training,
                                   dropout_probability, layout,
                                   bias_type,           mask_type,
                                   tensorType};

        namespace fe = cudnn_frontend;
        using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
              std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // K
              std::shared_ptr<fe::graph::Tensor_attributes>,  // V
              std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
              std::shared_ptr<fe::graph::Tensor_attributes>,  // O
              std::shared_ptr<fe::graph::Tensor_attributes>,  // Stats
              std::shared_ptr<fe::graph::Tensor_attributes>,  // bias
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_kv
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dropout_seed
              std::shared_ptr<fe::graph::Tensor_attributes> >;  // dropout_offset

        using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
        static thread_local CacheType sdpa_f16_fprop_cache;

        // Get plan from cache if cache is available, otherwise create one
        auto get_graph = [&](CacheType &cache, const FADescriptor_v1 &descriptor)
            -> graph_and_tensors {
            // if hit, return
            auto it = cache.find(descriptor);
            if (it != cache.end()) {
                auto graph = it->second;
                return graph;
            }

            // otherwise, build the op_graph and the plan. Then update cache
            auto mha_graph = std::make_shared<fe::graph::Graph>();
            mha_graph->set_io_data_type(tensorType)
                    .set_intermediate_data_type(fe::DataType_t::FLOAT)
                    .set_compute_data_type(fe::DataType_t::FLOAT);

            std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, attn_scale;
            std::shared_ptr<fe::graph::Tensor_attributes> bias, seq_q, seq_kv;
            std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

            std::vector<int64_t> q_stride(4);
            std::vector<int64_t> k_stride(4);
            std::vector<int64_t> v_stride(4);
            generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
            Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("Q")
                            .set_dim({b, h, s_q, d})
                            .set_stride(q_stride));
            K = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("K")
                            .set_dim({b, hg, s_kv, d})
                            .set_stride(k_stride));
            V = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("V")
                            .set_dim({b, hg, s_kv, d})
                            .set_stride(v_stride));

            attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

            fe::graph::SDPA_attributes sdpa_options;
            sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_is_inference(!is_training)
                            .set_causal_mask(is_causal)
                            .set_attn_scale(attn_scale);

            sdpa_options.set_alibi_mask(is_alibi);

            if (is_bias) {
                bias = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim({bias_b, bias_h, s_q, s_kv})
                                .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
                sdpa_options.set_bias(bias);
            }

            if (is_padding) {
                seq_q  = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("seq_q")
                                .set_dim({b, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT32));
                seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("seq_kv")
                                .set_dim({b, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT32));
                sdpa_options.set_padding_mask(is_padding)
                                .set_seq_len_q(seq_q)
                                .set_seq_len_kv(seq_kv);
            }

            if (is_dropout) {
                dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Seed")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT64));
                dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Offset")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT64));
                sdpa_options.set_dropout(
                                dropout_probability, dropout_seed, dropout_offset);
            }

            auto [O, Stats] = mha_graph->sdpa(Q, K, V, sdpa_options);

            std::vector<int64_t> o_stride(4);
            generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
            O->set_output(true).set_dim({b, h, s_q, d}).set_stride(o_stride);

            if (is_training) {
                Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                        .set_dim({b, h, s_q, 1})
                        .set_stride({h * s_q, s_q, 1, 1});
            }

            std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                    std::shared_ptr<fe::graph::Tensor_attributes> >  // O
            key_tensors_tuple = std::make_tuple(Q, K, V, attn_scale, O);
            auto Stats_tuple = is_training ? std::make_tuple(Stats) : std::make_tuple(nullptr);
            auto bias_tuple = is_bias ? std::make_tuple(bias) : std::make_tuple(nullptr);
            auto padding_tuple = is_padding ?
                std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
            auto dropout_tuple = is_dropout ?
                std::make_tuple(dropout_seed, dropout_offset) : std::make_tuple(nullptr, nullptr);
            auto return_empty_tuple = std::tuple_cat(
                std::make_tuple(nullptr), key_tensors_tuple,
                Stats_tuple, bias_tuple, padding_tuple, dropout_tuple);

            NVTE_CHECK_CUDNN_FE(mha_graph->validate());
            NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
            NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

            auto return_tuple = std::tuple_cat(
                std::make_tuple(mha_graph), key_tensors_tuple,
                Stats_tuple, bias_tuple, padding_tuple, dropout_tuple);
            cache.insert({descriptor, return_tuple});

            return return_tuple;
        };

        auto [mha_graph, Q, K, V, attn_scale, O, Stats,
            bias, seq_q, seq_kv, dropout_seed, dropout_offset] = get_graph(
                sdpa_f16_fprop_cache, descriptor);

        auto plan_workspace_size = mha_graph->get_workspace_size();

        // Exit to request upper level API to allocate memory if needed
        size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
        if (workspace == nullptr) {
            *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
            return;
        }

        // cuDNN stream check needs to be moved here to support dummy kernel calls with
        // null streams for sizing the cuDNN workspace.
        NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

        // Build variant pack
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
            {Q, devPtrQ},
            {K, devPtrK},
            {V, devPtrV},
            {attn_scale, &scaling_factor},
            {O, devPtrO}};

        if (is_training) {
            variant_pack[Stats] = devPtrSoftmaxStats;
        }

        if (is_bias) {
            variant_pack[bias] = devPtrBias;
        }

        if (is_padding) {
            constexpr size_t nthreads_per_block = 128;
            const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
            void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
            void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
            cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
                b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
                static_cast<const int32_t *>(devPtrCuSeqlensKV),
                static_cast<int32_t *>(devActualSeqlenQ),
                static_cast<int32_t *>(devActualSeqlenKV));
            variant_pack[seq_q]  = devActualSeqlenQ;
            variant_pack[seq_kv] = devActualSeqlenKV;
        }

        if (is_dropout) {
            variant_pack[dropout_seed] = devPtrDropoutSeed;
            variant_pack[dropout_offset] = devPtrDropoutOffset;
        }

        NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
    } catch (cudnn_frontend::cudnnException &e) {
        NVTE_ERROR(e.what());
    }
}

void fused_attn_arbitrary_seqlen_bwd_impl(
                int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                int64_t bias_b, int64_t bias_h,
                float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
                NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                void* devPtrQ, void* devPtrKTranspose, void* devPtrVTranspose,
                void* devPtrO, void* devPtrSoftmaxStats, void* devPtrBias,
                void* devPtrdQ, void* devPtrdK, void* devPtrdV, void* devPtrdO, void* devPtrdBias,
                void* devPtrDropoutSeed, void* devPtrDropoutOffset,
                void* devPtrCuSeqlensQ, void* devPtrCuSeqlensKV,
                cudnn_frontend::DataType_t tensorType, void *workspace, size_t *workspace_size,
                cudaStream_t stream, cudnnHandle_t handle) {
    bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
    bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
    bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK)
        || (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
    bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK)
        || (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
    bool is_dropout = (dropout_probability != 0.0f);

    try {
        FADescriptor_v1 descriptor{b,                   h,
                                   hg,                  s_q,
                                   s_kv,                d,
                                   scaling_factor,      true,
                                   dropout_probability, layout,
                                   bias_type,           mask_type,
                                   tensorType};

        namespace fe = cudnn_frontend;
        using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
              std::shared_ptr<fe::graph::Tensor_attributes>,  // q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // k
              std::shared_ptr<fe::graph::Tensor_attributes>,  // v
              std::shared_ptr<fe::graph::Tensor_attributes>,  // o
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
              std::shared_ptr<fe::graph::Tensor_attributes>,  // stats
              std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dV
              std::shared_ptr<fe::graph::Tensor_attributes>,  // bias
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dBias
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_kv
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dropout_seed
              std::shared_ptr<fe::graph::Tensor_attributes> >;  // dropout_offset

        using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
        static thread_local CacheType sdpa_f16_bprop_cache;

        // Get plan from cache if cache is available, otherwise create one
        auto get_graph = [&](CacheType &cache, const FADescriptor_v1 &descriptor)
            -> graph_and_tensors {
            // if hit, return
            auto it = cache.find(descriptor);
            if (it != cache.end()) {
                auto graph = it->second;
                return graph;
            }

            // otherwise, build the op_graph and the plan. Then update cache
            auto mha_graph = std::make_shared<fe::graph::Graph>();
            mha_graph->set_io_data_type(tensorType)
                    .set_intermediate_data_type(fe::DataType_t::FLOAT)
                    .set_compute_data_type(fe::DataType_t::FLOAT);

            std::shared_ptr<fe::graph::Tensor_attributes> q, k, v, o, dO, stats, attn_scale;
            std::shared_ptr<fe::graph::Tensor_attributes> bias, dBias, seq_q, seq_kv;
            std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

            std::vector<int64_t> q_stride(4);
            std::vector<int64_t> k_stride(4);
            std::vector<int64_t> v_stride(4);
            std::vector<int64_t> o_stride(4);
            generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
            generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
            q = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("Q")
                            .set_dim({b, h, s_q, d})
                            .set_stride(q_stride));
            k = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("K")
                            .set_dim({b, hg, s_kv, d})
                            .set_stride(k_stride));
            v = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("V")
                            .set_dim({b, hg, s_kv, d})
                            .set_stride(v_stride));
            o = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("O")
                            .set_dim({b, h, s_q, d})
                            .set_stride(o_stride));
            dO = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("dO")
                            .set_dim({b, h, s_q, d})
                            .set_stride(o_stride));
            stats = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("stats")
                            .set_dim({b, h, s_q, 1})
                            .set_stride({h * s_q, s_q, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));

            attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

            fe::graph::SDPA_backward_attributes sdpa_backward_options;
            sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                            .set_name("flash_attention_backward")
                            .set_causal_mask(is_causal)
                            .set_attn_scale(attn_scale);

            sdpa_backward_options.set_alibi_mask(is_alibi);

            if (is_bias) {
                bias = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim({bias_b, bias_h, s_q, s_kv})
                                .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
                dBias = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("dBias")
                                .set_dim({bias_b, bias_h, s_q, s_kv})
                                .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
                sdpa_backward_options.set_bias(bias);
                sdpa_backward_options.set_dbias(dBias);
            }

            if (is_padding) {
                seq_q  = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("seq_q")
                                .set_dim({b, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT32));
                seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("seq_kv")
                                .set_dim({b, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT32));
                sdpa_backward_options.set_padding_mask(is_padding)
                                .set_seq_len_q(seq_q)
                                .set_seq_len_kv(seq_kv);
            }

            if (is_dropout) {
                dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Seed")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT64));
                dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Offset")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(fe::DataType_t::INT64));
                sdpa_backward_options.set_dropout(
                                dropout_probability, dropout_seed, dropout_offset);
            }

            auto [dQ, dK, dV] = mha_graph->sdpa_backward(
                q, k, v, o, dO, stats, sdpa_backward_options);

            dQ->set_output(true)
                    .set_dim({b, h, s_q, d})
                    .set_stride(q_stride);
            dK->set_output(true)
                    .set_dim({b, hg, s_kv, d})
                    .set_stride(k_stride);
            dV->set_output(true)
                    .set_dim({b, hg, s_kv, d})
                    .set_stride(v_stride);

            std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // q
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // k
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // v
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // o
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // stats
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
                    std::shared_ptr<fe::graph::Tensor_attributes> >  // dV
            key_tensors_tuple = std::make_tuple(q, k, v, o, dO, stats, attn_scale, dQ, dK, dV);
            auto bias_tuple = is_bias ?
                std::make_tuple(bias, dBias) : std::make_tuple(nullptr, nullptr);
            auto padding_tuple = is_padding ?
                std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
            auto dropout_tuple = is_dropout ?
                std::make_tuple(dropout_seed, dropout_offset) : std::make_tuple(nullptr, nullptr);
            auto return_empty_tuple = std::tuple_cat(
                std::make_tuple(nullptr), key_tensors_tuple,
                bias_tuple, padding_tuple, dropout_tuple);

            NVTE_CHECK_CUDNN_FE(mha_graph->validate());
            NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
            NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

            auto return_tuple = std::tuple_cat(
                std::make_tuple(mha_graph), key_tensors_tuple,
                bias_tuple, padding_tuple, dropout_tuple);
            cache.insert({descriptor, return_tuple});

            return return_tuple;
        };

        auto [mha_graph, q, k, v, o, dO, stats, attn_scale, dQ, dK, dV,
            bias, dBias, seq_q, seq_kv, dropout_seed, dropout_offset] = get_graph(
                sdpa_f16_bprop_cache, descriptor);

        auto plan_workspace_size = mha_graph->get_workspace_size();

        // Exit to request upper level API to allocate memory if needed
        size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
        if (workspace == nullptr) {
            *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
            return;
        }

        // cuDNN stream check needs to be moved here to support dummy kernel calls with
        // null streams for sizing the cuDNN workspace.
        NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

        // build variant pack
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
            {q, devPtrQ},
            {k, devPtrKTranspose},
            {v, devPtrVTranspose},
            {o, devPtrO},
            {dO, devPtrdO},
            {stats, devPtrSoftmaxStats},
            {attn_scale, &scaling_factor},
            {dQ, devPtrdQ},
            {dK, devPtrdK},
            {dV, devPtrdV},
        };

        if (is_bias) {
            variant_pack[bias] = devPtrBias;
            variant_pack[dBias] = devPtrdBias;
        }

        if (is_padding) {
            constexpr size_t nthreads_per_block = 128;
            const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
            void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
            void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
            cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
                b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
                static_cast<const int32_t *>(devPtrCuSeqlensKV),
                static_cast<int32_t *>(devActualSeqlenQ),
                static_cast<int32_t *>(devActualSeqlenKV));
            variant_pack[seq_q]  = devActualSeqlenQ;
            variant_pack[seq_kv] = devActualSeqlenKV;
        }

        if (is_dropout) {
            variant_pack[dropout_seed] = devPtrDropoutSeed;
            variant_pack[dropout_offset] = devPtrDropoutOffset;
        }

        NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
    } catch (cudnn_frontend::cudnnException &e) {
        NVTE_ERROR(e.what());
    }
}
}  // namespace fused_attn

using namespace transformer_engine::fused_attn;
void fused_attn_arbitrary_seqlen_fwd_qkvpacked(
    size_t batch, size_t num_attn_heads, size_t max_seqlen, size_t head_dim, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor *input_QKV, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const auto QKV_type = input_QKV->data.dtype;
    void *devPtrQKV = input_QKV->data.dptr;
    NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
    size_t stride = 0;
    if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
        stride = typeToSize(QKV_type) * num_attn_heads * head_dim;
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
        stride = typeToSize(QKV_type) * head_dim;
    }
    void *devPtrQ = static_cast<void *>(devPtrQKV);
    void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
    void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

    void *devPtrBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
        devPtrBias = input_Bias->data.dptr;
        bias_b = input_Bias->data.shape[0];
        bias_h = input_Bias->data.shape[1];
    }
    void *devPtrO = output_O->data.dptr;
    void *devPtrS = nullptr;
    void *devPtrCuSeqlens = cu_seqlens->data.dptr;

    if (Aux_CTX_Tensors->size == 0) {
        if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
            Aux_CTX_Tensors->size = 3;
            Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
            output_S->data.dptr = nullptr;
            output_S->data.shape = {batch, num_attn_heads, max_seqlen, 1};
            output_S->data.dtype = DType::kFloat32;
            Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
            output_rng_state->data.dptr = nullptr;
            output_rng_state->data.shape = {2};
            output_rng_state->data.dtype = DType::kInt64;
            Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
            output_bias->data.dptr = nullptr;
            output_bias->data.shape = {bias_b, bias_h, max_seqlen, max_seqlen};
            output_bias->data.dtype = QKV_type;
        } else {
            Aux_CTX_Tensors->size = 2;
            Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
            output_S->data.dptr = nullptr;
            output_S->data.shape = {batch, num_attn_heads, max_seqlen, 1};
            output_S->data.dtype = DType::kFloat32;
            Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
            output_rng_state->data.dptr = nullptr;
            output_rng_state->data.shape = {2};
            output_rng_state->data.dtype = DType::kInt64;
        }
    } else if (Aux_CTX_Tensors->size == 2) {
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
    } else if (Aux_CTX_Tensors->size == 3) {
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
        Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
        output_bias->data.dptr = devPtrBias;
    } else {
        NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
    }

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    fused_attn_arbitrary_seqlen_fwd_impl(batch, num_attn_heads, num_attn_heads,
                                max_seqlen, max_seqlen, head_dim, bias_b, bias_h,
                                is_training, attn_scale, p_dropout, qkv_layout,
                                bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrBias, devPtrS, devPtrO,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlens, devPtrCuSeqlens,
                                get_cudnn_fe_dtype(QKV_type),
                                workspace->data.dptr, &workspace_size,
                                stream, handle);

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}

void fused_attn_arbitrary_seqlen_bwd_qkvpacked(size_t batch, size_t num_attn_heads,
                                  size_t max_seqlen, size_t head_dim, float attn_scale,
                                  float p_dropout, NVTE_QKV_Layout qkv_layout,
                                  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                  const Tensor *input_QKV, const Tensor *input_O,
                                  const Tensor *input_dO, const Tensor *input_Bias,
                                  Tensor *output_S,
                                  Tensor *output_dQKV, Tensor *output_dBias,
                                  const Tensor *cu_seqlens, const Tensor *rng_state,
                                  Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const auto QKV_type = input_QKV->data.dtype;
    void *devPtrQKV = input_QKV->data.dptr;

    NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
    size_t stride = 0;
    if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
        stride = typeToSize(QKV_type) * num_attn_heads * head_dim;
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
        stride = typeToSize(QKV_type) * head_dim;
    }
    void *devPtrQ = devPtrQKV;
    void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
    void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

    void* devPtrO = input_O->data.dptr;
    void *devPtrdO = input_dO->data.dptr;
    void *devPtrBias = nullptr;
    void *devPtrdBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
        devPtrBias = input_Bias->data.dptr;
        devPtrdBias = output_dBias->data.dptr;
        bias_b = output_dBias->data.shape[0];
        bias_h = output_dBias->data.shape[1];
    }

    void *devPtrdQKV = output_dQKV->data.dptr;
    void *devPtrdQ = devPtrdQKV;
    void *devPtrdK = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + stride);
    void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + 2 * stride);

    void *devPtrSoftmaxStats = nullptr;
    devPtrSoftmaxStats = output_S->data.dptr;

    void *devPtrCuSeqlens = cu_seqlens->data.dptr;

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    fused_attn_arbitrary_seqlen_bwd_impl(batch, num_attn_heads, num_attn_heads,
                                max_seqlen, max_seqlen, head_dim, bias_b, bias_h,
                                attn_scale, p_dropout, qkv_layout,
                                bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias,
                                devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdBias,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlens, devPtrCuSeqlens,
                                get_cudnn_fe_dtype(QKV_type), workspace->data.dptr,
                                &workspace_size, stream, handle);

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}
void fused_attn_arbitrary_seqlen_fwd_kvpacked(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_KV,
    const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *rng_state, Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const auto QKV_type = input_Q->data.dtype;
    void *devPtrQ = input_Q->data.dptr;
    void *devPtrKV = input_KV->data.dptr;
    NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
    size_t stride = 0;
    if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
        stride = typeToSize(QKV_type) * num_gqa_groups * head_dim;
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
        stride = typeToSize(QKV_type) * head_dim;
    }
    void *devPtrK = devPtrKV;
    void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

    void *devPtrBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
        devPtrBias = input_Bias->data.dptr;
        bias_b = input_Bias->data.shape[0];
        bias_h = input_Bias->data.shape[1];
    }
    void *devPtrO = output_O->data.dptr;
    void *devPtrS = nullptr;

    void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    if (Aux_CTX_Tensors->size == 0) {
        if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
            Aux_CTX_Tensors->size = 3;
            Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
            output_S->data.dptr = nullptr;
            output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
            output_S->data.dtype = DType::kFloat32;
            Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
            output_rng_state->data.dptr = nullptr;
            output_rng_state->data.shape = {2};
            output_rng_state->data.dtype = DType::kInt64;
            Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
            output_bias->data.dptr = nullptr;
            output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
            output_bias->data.dtype = QKV_type;
        } else {
            Aux_CTX_Tensors->size = 2;
            Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
            output_S->data.dptr = nullptr;
            output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
            output_S->data.dtype = DType::kFloat32;
            Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
            output_rng_state->data.dptr = nullptr;
            output_rng_state->data.shape = {2};
            output_rng_state->data.dtype = DType::kInt64;
        }
    } else if (Aux_CTX_Tensors->size == 2) {
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
    } else if (Aux_CTX_Tensors->size == 3) {
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
        Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
        output_bias->data.dptr = devPtrBias;
    } else {
        NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
    }

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    fused_attn_arbitrary_seqlen_fwd_impl(batch, num_attn_heads, num_gqa_groups,
                                max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
                                is_training, attn_scale, p_dropout, qkv_layout,
                                bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrBias, devPtrS, devPtrO,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_fe_dtype(QKV_type),
                                workspace->data.dptr, &workspace_size,
                                stream, handle);

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}

void fused_attn_arbitrary_seqlen_bwd_kvpacked(
                                  size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                                  size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                                  float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                                  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                  const Tensor *input_Q, const Tensor *input_KV,
                                  const Tensor *input_O, const Tensor *input_dO,
                                  const Tensor *input_Bias, Tensor *output_S,
                                  Tensor *output_dQ, Tensor *output_dKV,
                                  Tensor *output_dBias, const Tensor *cu_seqlens_q,
                                  const Tensor *cu_seqlens_kv,
                                  const Tensor *rng_state, Tensor *workspace,
                                  cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const auto QKV_type = input_Q->data.dtype;
    void *devPtrQ = input_Q->data.dptr;
    void *devPtrKV = input_KV->data.dptr;
    NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
    size_t stride = 0;
    if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
        stride = typeToSize(QKV_type) * num_gqa_groups * head_dim;
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
        stride = typeToSize(QKV_type) * head_dim;
    }
    void *devPtrK = devPtrKV;
    void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

    void* devPtrO = input_O->data.dptr;
    void *devPtrdO = input_dO->data.dptr;
    void *devPtrBias = nullptr;
    void *devPtrdBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
        devPtrBias = input_Bias->data.dptr;
        devPtrdBias = output_dBias->data.dptr;
        bias_b = output_dBias->data.shape[0];
        bias_h = output_dBias->data.shape[1];
    }

    void *devPtrdQ = output_dQ->data.dptr;
    void *devPtrdKV = output_dKV->data.dptr;
    void *devPtrdK = devPtrdKV;
    void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdKV) + stride);

    void *devPtrSoftmaxStats = nullptr;
    devPtrSoftmaxStats = output_S->data.dptr;

    void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    fused_attn_arbitrary_seqlen_bwd_impl(batch, num_attn_heads, num_gqa_groups,
                                max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
                                attn_scale, p_dropout, qkv_layout,
                                bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias,
                                devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdBias,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_fe_dtype(QKV_type), workspace->data.dptr,
                                &workspace_size, stream, handle);

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}

void fused_attn_arbitrary_seqlen_fwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_K,
    const Tensor *input_V, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const auto QKV_type = input_Q->data.dtype;
    void *devPtrQ = input_Q->data.dptr;
    void *devPtrK = input_K->data.dptr;
    void *devPtrV = input_V->data.dptr;
    void *devPtrO = output_O->data.dptr;
    void *devPtrS = nullptr;
    void *devPtrBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
        devPtrBias = input_Bias->data.dptr;
        bias_b = input_Bias->data.shape[0];
        bias_h = input_Bias->data.shape[1];
    }

    void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    if (Aux_CTX_Tensors->size == 0) {
        if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
            Aux_CTX_Tensors->size = 3;
            Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
            output_S->data.dptr = nullptr;
            output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
            output_S->data.dtype = DType::kFloat32;
            Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
            output_rng_state->data.dptr = nullptr;
            output_rng_state->data.shape = {2};
            output_rng_state->data.dtype = DType::kInt64;
            Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
            output_bias->data.dptr = nullptr;
            output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
            output_bias->data.dtype = QKV_type;
        } else {
            Aux_CTX_Tensors->size = 2;
            Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
            output_S->data.dptr = nullptr;
            output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
            output_S->data.dtype = DType::kFloat32;
            Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
            output_rng_state->data.dptr = nullptr;
            output_rng_state->data.shape = {2};
            output_rng_state->data.dtype = DType::kInt64;
        }
    } else if (Aux_CTX_Tensors->size == 2) {
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
    } else if (Aux_CTX_Tensors->size == 3) {
        Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS = output_S->data.dptr;
        Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
        Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
        output_bias->data.dptr = devPtrBias;
    } else {
        NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
    }

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    fused_attn_arbitrary_seqlen_fwd_impl(batch, num_attn_heads, num_gqa_groups,
                                max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
                                is_training, attn_scale, p_dropout, qkv_layout,
                                bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrBias, devPtrS, devPtrO,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_fe_dtype(QKV_type),
                                workspace->data.dptr, &workspace_size,
                                stream, handle);

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}

void fused_attn_arbitrary_seqlen_bwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                                  size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                                  float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                                  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                  const Tensor *input_Q, const Tensor *input_K,
                                  const Tensor *input_V, const Tensor *input_O,
                                  const Tensor *input_dO, const Tensor *input_Bias,
                                  Tensor *output_S,
                                  Tensor *output_dQ, Tensor *output_dK, Tensor *output_dV,
                                  Tensor *output_dBias, const Tensor *cu_seqlens_q,
                                  const Tensor *cu_seqlens_kv,
                                  const Tensor *rng_state, Tensor *workspace,
                                  cudaStream_t stream, cudnnHandle_t handle) {
    using namespace transformer_engine;

    const auto QKV_type = input_Q->data.dtype;
    void *devPtrQ = input_Q->data.dptr;
    void *devPtrK = input_K->data.dptr;
    void *devPtrV = input_V->data.dptr;
    void* devPtrO = input_O->data.dptr;
    void *devPtrdO = input_dO->data.dptr;
    void *devPtrBias = nullptr;
    void *devPtrdBias = nullptr;
    size_t bias_b = 0;
    size_t bias_h = 0;
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
        devPtrBias = input_Bias->data.dptr;
        devPtrdBias = output_dBias->data.dptr;
        bias_b = output_dBias->data.shape[0];
        bias_h = output_dBias->data.shape[1];
    }

    void *devPtrdQ = output_dQ->data.dptr;
    void *devPtrdK = output_dK->data.dptr;
    void *devPtrdV = output_dV->data.dptr;
    void *devPtrSoftmaxStats = nullptr;
    devPtrSoftmaxStats = output_S->data.dptr;

    void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    void* devPtrDropoutSeed = rng_state->data.dptr;
    void* devPtrDropoutOffset = reinterpret_cast<void *>(
                    reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    fused_attn_arbitrary_seqlen_bwd_impl(batch, num_attn_heads, num_gqa_groups,
                                max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
                                attn_scale, p_dropout, qkv_layout, bias_type, mask_type,
                                devPtrQ, devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias,
                                devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdBias,
                                devPtrDropoutSeed, devPtrDropoutOffset,
                                devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                                get_cudnn_fe_dtype(QKV_type), workspace->data.dptr,
                                &workspace_size, stream, handle);

    if (workspace_size > 0) {
        if (workspace->data.dptr == nullptr) {
            workspace->data.shape = {workspace_size};
            workspace->data.dtype = DType::kByte;
            return;
        }
    } else if (workspace_size == 0) {
        workspace->data.shape = {1};
        workspace->data.dtype = DType::kByte;
        return;
    } else {
        NVTE_ERROR("Unexpected workspace_size.");
    }
}
}  // namespace transformer_engine
#endif  // CUDNN_VERSION >= 8900
#endif  // __HIP_PLATFORM_AMD__ 
