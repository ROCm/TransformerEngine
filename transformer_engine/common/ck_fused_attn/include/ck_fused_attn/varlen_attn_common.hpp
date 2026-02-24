/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_VARLEN_ATTN_COMMON_H
#define CK_FUSED_ATTN_VARLEN_ATTN_COMMON_H

namespace ck_fused_attn {
namespace varlen {

enum class CausalMaskType {
    DISABLE      = 0,
    TOP_LEFT     = 1,
    BOTTOM_RIGHT = 2
};

template <int BS,
          int HEAD_NUM,
          int SEQ_Q,
          int MAX_SEQ_KV,
          int HEAD_DIM,
          int STEP2_BLOCK_SIZE     = 256,
          bool ENABLE_DROPOUT_MASK = true,
          CausalMaskType MASK_TYPE = CausalMaskType::DISABLE>
struct FmhaKernelConfig {
    static constexpr int bs                        = BS;
    static constexpr int head_num                  = HEAD_NUM;
    static constexpr int seq_q                     = SEQ_Q;
    static constexpr int max_seq_kv                = MAX_SEQ_KV;
    static constexpr int head_dim                  = HEAD_DIM;
    static constexpr int step2_block_size          = STEP2_BLOCK_SIZE;
    static constexpr bool enable_dropout_mask      = ENABLE_DROPOUT_MASK;
    static constexpr CausalMaskType mask_type      = MASK_TYPE;
};

// Configs used for TE integration: max batch 65536, max heads 256, seq_q=1, max_seq_kv=16
using ConfigHeadDim64  = FmhaKernelConfig<65536, 256, 1, 16, 64, 256, false, CausalMaskType::DISABLE>;
using ConfigHeadDim128 = FmhaKernelConfig<65536, 256, 1, 16, 128, 256, false, CausalMaskType::DISABLE>;

}  // namespace varlen
}  // namespace ck_fused_attn

#endif  // CK_FUSED_ATTN_VARLEN_ATTN_COMMON_H
