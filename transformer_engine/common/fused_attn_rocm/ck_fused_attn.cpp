#include "ck_fused_attn.h"

#include <ck_fmha.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "../util/system.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn_rocm {

void ck_fused_attn_fwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                       size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim, bool is_training,
                       float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                       NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, const Tensor *input_Q,
                       const Tensor *input_K, const Tensor *input_V, const Tensor *input_Bias,
                       Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                       const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                       const Tensor *rng_state, hipStream_t stream) {
    const auto QKV_type     = input_Q->data.dtype;
    void *devPtrQ           = input_Q->data.dptr;
    void *devPtrK           = input_K->data.dptr;
    void *devPtrV           = input_V->data.dptr;
    void *devPtrO           = output_O->data.dptr;
    void *devPtrS           = nullptr;
    void *devPtrBias        = nullptr;
    size_t bias_b           = 0;
    size_t bias_h           = 0;
    void *devPtrCuSeqlensQ  = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;

    uint32_t bias_value;
    if (bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) {
        bias_value = 0;
    } else if (bias_type == NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS) {
        NVTE_ERROR("Only NVTE_NO_BIAS is supported for now");
    } else if (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
        NVTE_ERROR("Only NVTE_NO_BIAS is supported for now");
    } else if (bias_type == NVTE_Bias_Type::NVTE_ALIBI) {
        NVTE_ERROR("Only NVTE_NO_BIAS is supported for now");
    } else {
        NVTE_ERROR("Unexpected bias_type");
    }

    uint32_t mask_value;
    if (mask_type == NVTE_Mask_Type::NVTE_NO_MASK) {
        mask_value = 0;
    } else if (mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) {
        mask_value = 1;
    } else if (mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) {
        NVTE_ERROR("NVTE_PADDING_MASK is not supported for now");
    } else if (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) {
        mask_value = 1;
    } else {
        NVTE_ERROR("Unexpected mask_type");
    }

    if (Aux_CTX_Tensors->size == 0) {
        Aux_CTX_Tensors->size        = 2;
        Tensor *output_S             = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        output_S->data.dptr          = nullptr;
        output_S->data.shape         = {batch, num_attn_heads, max_seqlen_q, 1};
        output_S->data.dtype         = DType::kFloat32;
        Tensor *output_rng_state     = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr  = nullptr;
        output_rng_state->data.shape = {2};
        output_rng_state->data.dtype = DType::kInt64;
    } else if (Aux_CTX_Tensors->size == 2) {
        Tensor *output_S            = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS                     = output_S->data.dptr;
        Tensor *output_rng_state    = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
    } else if (Aux_CTX_Tensors->size == 3) {
        Tensor *output_S            = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
        devPtrS                     = output_S->data.dptr;
        Tensor *output_rng_state    = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
        output_rng_state->data.dptr = rng_state->data.dptr;
        Tensor *output_bias         = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
        output_bias->data.dptr      = devPtrBias;
    } else {
        NVTE_ERROR("Unexpected Aux_CTX_Tensors->size");
    }

    uint64_t drop_seed   = 0;
    uint64_t drop_offset = 0;
    if (p_dropout > 0) {
        void *devPtrRngState                           = rng_state->data.dptr;
        static thread_local uint64_t host_rng_state[2] = {0};
        NVTE_CHECK_CUDA(hipMemcpyAsync(host_rng_state, devPtrRngState, 2 * sizeof(uint64_t),
                                       hipMemcpyDeviceToHost, stream));
        NVTE_CHECK_CUDA(hipStreamSynchronize(stream));
        drop_seed   = host_rng_state[0];
        drop_offset = host_rng_state[1];
    }

    ck_fused_attn_fwd_impl(batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv,
                           head_dim, bias_b, bias_h, is_training, attn_scale, p_dropout, drop_seed,
                           drop_offset, bias_value, mask_value, devPtrQ, devPtrK, devPtrV,
                           devPtrBias, devPtrS, devPtrO, devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                           get_datatype_str(QKV_type), stream);
}

void ck_fused_attn_bwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                       size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim, float attn_scale,
                       float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                       NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_K,
                       const Tensor *input_V, const Tensor *input_O, const Tensor *input_dO,
                       const Tensor *input_Bias, Tensor *output_S, Tensor *output_dQ,
                       Tensor *output_dK, Tensor *output_dV, Tensor *output_dBias,
                       const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                       const Tensor *rng_state, Tensor *workspace, bool deterministic,
                       hipStream_t stream) {
    const auto QKV_type = input_Q->data.dtype;
    void *devPtrQ       = input_Q->data.dptr;
    void *devPtrK       = input_K->data.dptr;
    void *devPtrV       = input_V->data.dptr;
    void *devPtrO       = input_O->data.dptr;
    void *devPtrdO      = input_dO->data.dptr;
    void *devPtrBias    = nullptr;
    void *devPtrdBias   = nullptr;
    size_t bias_b       = 0;
    size_t bias_h       = 0;

    uint32_t bias_value;
    if (bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) {
        bias_value = 0;
    } else if (bias_type == NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS) {
        NVTE_ERROR("Only NVTE_NO_BIAS is supported for now");
    } else if (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
        NVTE_ERROR("Only NVTE_NO_BIAS is supported for now");
    } else if (bias_type == NVTE_Bias_Type::NVTE_ALIBI) {
        NVTE_ERROR("Only NVTE_NO_BIAS is supported for now");
    } else {
        NVTE_ERROR("Unexpected bias_type");
    }

    uint32_t mask_value;
    if (mask_type == NVTE_Mask_Type::NVTE_NO_MASK) {
        mask_value = 0;
    } else if (mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) {
        mask_value = 1;
    } else if (mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) {
        NVTE_ERROR("NVTE_PADDING_MASK is not supported for now");
    } else if (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) {
        mask_value = 1;
    } else {
        NVTE_ERROR("Unexpected mask_type");
    }

    /*
    if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) &&
        (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
      devPtrBias = input_Bias->data.dptr;
      devPtrdBias = output_dBias->data.dptr;
      bias_b = output_dBias->data.shape[0];
      bias_h = output_dBias->data.shape[1];
    }
    */

    void *devPtrdQ           = output_dQ->data.dptr;
    void *devPtrdK           = output_dK->data.dptr;
    void *devPtrdV           = output_dV->data.dptr;
    void *devPtrSoftmaxStats = output_S->data.dptr;
    void *devPtrCuSeqlensQ   = cu_seqlens_q->data.dptr;
    void *devPtrCuSeqlensKV  = cu_seqlens_kv->data.dptr;
    void *devPtrDropoutSeed  = rng_state->data.dptr;
    void *devPtrDropoutOffset =
        reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

    size_t workspace_size = 0;

    uint64_t drop_seed   = 0;
    uint64_t drop_offset = 0;
    if (p_dropout > 0) {
        void *devPtrRngState = rng_state->data.dptr;
        static thread_local uint64_t host_rng_state[2] = {0};
        NVTE_CHECK_CUDA(hipMemcpyAsync(host_rng_state, devPtrRngState, 2 * sizeof(uint64_t),
                                       hipMemcpyDeviceToHost, stream));
        NVTE_CHECK_CUDA(hipStreamSynchronize(stream));
        drop_seed   = host_rng_state[0];
        drop_offset = host_rng_state[1];
    }

    ck_fused_attn_bwd_impl(batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv,
                           head_dim, bias_b, bias_h, attn_scale, p_dropout, drop_seed, drop_offset,
                           bias_value, mask_value, devPtrQ, devPtrK, devPtrV, devPtrO,
                           devPtrSoftmaxStats, devPtrBias, devPtrdQ, devPtrdK, devPtrdV, devPtrdO,
                           devPtrdBias, devPtrCuSeqlensQ, devPtrCuSeqlensKV,
                           get_datatype_str(QKV_type), workspace->data.dptr, &workspace_size,
                           deterministic, stream);

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
        NVTE_ERROR("Unexpected workspace_size");
    }

}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
