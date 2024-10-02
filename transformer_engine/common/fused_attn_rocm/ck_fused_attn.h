#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FUSED_ATTN_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_CK_FUSED_ATTN_H_

#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace fused_attn_rocm {

void ck_fused_attn_fwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                       size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim, bool is_training,
                       float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                       NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, const Tensor *input_Q,
                       const Tensor *input_K, const Tensor *input_V, const Tensor *input_Bias,
                       Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                       const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                       const Tensor *rng_state, Tensor *workspace, cudaStream_t stream);

void ck_fused_attn_bwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                       size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim, float attn_scale,
                       float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                       NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_K,
                       const Tensor *input_V, const Tensor *input_O, const Tensor *input_dO,
                       const Tensor *input_Bias, Tensor *output_S, Tensor *output_dQ,
                       Tensor *output_dK, Tensor *output_dV, Tensor *output_dBias,
                       const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                       const Tensor *rng_state, Tensor *workspace, bool deterministic,
                       cudaStream_t stream);

}  // namespace fused_attn_rocm
}  // namespace transformer_engine

#endif
