/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "config_and_params.h"
#include "fused_attn_f16_arbitrary_seqlen.h"
#include "fused_attn_fp8.h"
#include "utils.h"

namespace transformer_engine {

std::string to_string(NVTE_QKV_Layout layout) {
  switch (layout) {
    case NVTE_SB3HD:
      return "NVTE_SB3HD";
    case NVTE_SBH3D:
      return "NVTE_SBH3D";
    case NVTE_SBHD_SB2HD:
      return "NVTE_SBHD_SB2HD";
    case NVTE_SBHD_SBH2D:
      return "NVTE_SBHD_SBH2D";
    case NVTE_SBHD_SBHD_SBHD:
      return "NVTE_SBHD_SBHD_SBHD";
    case NVTE_BS3HD:
      return "NVTE_BS3HD";
    case NVTE_BSH3D:
      return "NVTE_BSH3D";
    case NVTE_BSHD_BS2HD:
      return "NVTE_BSHD_BS2HD";
    case NVTE_BSHD_BSH2D:
      return "NVTE_BSHD_BSH2D";
    case NVTE_BSHD_BSHD_BSHD:
      return "NVTE_BSHD_BSHD_BSHD";
    case NVTE_T3HD:
      return "NVTE_T3HD";
    case NVTE_TH3D:
      return "NVTE_TH3D";
    case NVTE_THD_T2HD:
      return "NVTE_THD_T2HD";
    case NVTE_THD_TH2D:
      return "NVTE_THD_TH2D";
    case NVTE_THD_THD_THD:
      return "NVTE_THD_THD_THD";
    case NVTE_SBHD_BSHD_BSHD:
      return "NVTE_SBHD_BSHD_BSHD";
    case NVTE_BSHD_SBHD_SBHD:
      return "NVTE_BSHD_SBHD_SBHD";
    case NVTE_THD_BSHD_BSHD:
      return "NVTE_THD_BSHD_BSHD";
    case NVTE_THD_SBHD_SBHD:
      return "NVTE_THD_SBHD_SBHD";
    case NVTE_Paged_KV_BSHD_BSHD_BSHD:
      return "NVTE_Paged_KV_BSHD_BSHD_BSHD";
    case NVTE_Paged_KV_BSHD_SBHD_SBHD:
      return "NVTE_Paged_KV_BSHD_SBHD_SBHD";
    case NVTE_Paged_KV_SBHD_BSHD_BSHD:
      return "NVTE_Paged_KV_SBHD_BSHD_BSHD";
    case NVTE_Paged_KV_SBHD_SBHD_SBHD:
      return "NVTE_Paged_KV_SBHD_SBHD_SBHD";
    case NVTE_Paged_KV_THD_BSHD_BSHD:
      return "NVTE_Paged_KV_THD_BSHD_BSHD";
    case NVTE_Paged_KV_THD_SBHD_SBHD:
      return "NVTE_Paged_KV_THD_SBHD_SBHD";
    default:
      return "UNKNOWN_QKV_LAYOUT(" + std::to_string(static_cast<int>(layout)) + ")";
  }
}

std::string to_string(NVTE_QKV_Format format) {
  switch (format) {
    case NVTE_SBHD:
      return "NVTE_SBHD";
    case NVTE_BSHD:
      return "NVTE_BSHD";
    case NVTE_THD:
      return "NVTE_THD";
    case NVTE_BSHD_2SBHD:
      return "NVTE_BSHD_2SBHD";
    case NVTE_SBHD_2BSHD:
      return "NVTE_SBHD_2BSHD";
    case NVTE_THD_2BSHD:
      return "NVTE_THD_2BSHD";
    case NVTE_THD_2SBHD:
      return "NVTE_THD_2SBHD";
    default:
      return "UNKNOWN_QKV_FORMAT(" + std::to_string(static_cast<int>(format)) + ")";
  }
}

}  // namespace transformer_engine

// map NVTE_QKV_Layout to NVTE_QKV_Layout_Group
NVTE_QKV_Layout_Group nvte_get_qkv_layout_group(NVTE_QKV_Layout qkv_layout) {
  switch (qkv_layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_T3HD:
      return NVTE_QKV_Layout_Group::NVTE_3HD;
    case NVTE_QKV_Layout::NVTE_SBH3D:
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_TH3D:
      return NVTE_QKV_Layout_Group::NVTE_H3D;
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
      return NVTE_QKV_Layout_Group::NVTE_HD_2HD;
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
      return NVTE_QKV_Layout_Group::NVTE_HD_H2D;
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
    case NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_THD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_THD_BSHD_BSHD:
      return NVTE_QKV_Layout_Group::NVTE_HD_HD_HD;
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_SBHD_SBHD:
      return NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD;
    case NVTE_QKV_Layout::NVTE_BHSD_BHSD_BHSD:
      return NVTE_QKV_Layout_Group::NVTE_SD_SD_SD;
    default:
      NVTE_ERROR("Unsupported qkv_layout ", transformer_engine::to_string(qkv_layout),
                 " in nvte_get_qkv_layout_group.");
  }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format
NVTE_QKV_Format nvte_get_qkv_format(NVTE_QKV_Layout qkv_layout) {
  switch (qkv_layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
    case NVTE_QKV_Layout::NVTE_SBH3D:
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_SBHD_SBHD:
      return NVTE_QKV_Format::NVTE_SBHD;
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_BSHD_BSHD:
      return NVTE_QKV_Format::NVTE_BSHD;
    case NVTE_QKV_Layout::NVTE_T3HD:
    case NVTE_QKV_Layout::NVTE_TH3D:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
      return NVTE_QKV_Format::NVTE_THD;
    case NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_BSHD_BSHD:
      return NVTE_QKV_Format::NVTE_SBHD_2BSHD;
    case NVTE_QKV_Layout::NVTE_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_SBHD_SBHD:
      return NVTE_QKV_Format::NVTE_BSHD_2SBHD;
    case NVTE_QKV_Layout::NVTE_THD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_BSHD_BSHD:
      return NVTE_QKV_Format::NVTE_THD_2BSHD;
    case NVTE_QKV_Layout::NVTE_THD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_SBHD_SBHD:
      return NVTE_QKV_Format::NVTE_THD_2SBHD;
    case NVTE_QKV_Layout::NVTE_BHSD_BHSD_BHSD:
      return NVTE_QKV_Format::NVTE_BHSD;
    default:
      NVTE_ERROR("Unsupported qkv_layout ", transformer_engine::to_string(qkv_layout),
                 " in nvte_get_qkv_format.");
  }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format for Q
NVTE_QKV_Format nvte_get_q_format(NVTE_QKV_Layout qkv_layout) {
  const NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  switch (qkv_format) {
    case NVTE_QKV_Format::NVTE_SBHD:
    case NVTE_QKV_Format::NVTE_SBHD_2BSHD:
      return NVTE_QKV_Format::NVTE_SBHD;
    case NVTE_QKV_Format::NVTE_BSHD:
    case NVTE_QKV_Format::NVTE_BSHD_2SBHD:
      return NVTE_QKV_Format::NVTE_BSHD;
    case NVTE_QKV_Format::NVTE_THD:
    case NVTE_QKV_Format::NVTE_THD_2BSHD:
    case NVTE_QKV_Format::NVTE_THD_2SBHD:
      return NVTE_QKV_Format::NVTE_THD;
    case NVTE_QKV_Format::NVTE_BHSD:
      return NVTE_QKV_Format::NVTE_BHSD;
    default:
      NVTE_ERROR("Unsupported qkv_format ", transformer_engine::to_string(qkv_format),
                 " in nvte_get_q_format.");
  }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format for KV
NVTE_QKV_Format nvte_get_kv_format(NVTE_QKV_Layout qkv_layout) {
  const NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  switch (qkv_format) {
    case NVTE_QKV_Format::NVTE_SBHD:
    case NVTE_QKV_Format::NVTE_BSHD_2SBHD:
    case NVTE_QKV_Format::NVTE_THD_2SBHD:
      return NVTE_QKV_Format::NVTE_SBHD;
    case NVTE_QKV_Format::NVTE_BSHD:
    case NVTE_QKV_Format::NVTE_SBHD_2BSHD:
    case NVTE_QKV_Format::NVTE_THD_2BSHD:
      return NVTE_QKV_Format::NVTE_BSHD;
    case NVTE_QKV_Format::NVTE_THD:
      return NVTE_QKV_Format::NVTE_THD;
    case NVTE_QKV_Format::NVTE_BHSD:
      return NVTE_QKV_Format::NVTE_BHSD;
    default:
      NVTE_ERROR("Unsupported qkv_format ", transformer_engine::to_string(qkv_format),
                 " in nvte_get_kv_format.");
  }
}

namespace {

// The per-thread storage for the diagnostic string
thread_local std::string fused_attn_backend_message_buffer;

// Records `reason` in the per-thread buffer and `message`; returns with "no backend"
[[nodiscard]] NVTE_Fused_Attn_Backend reject(const char **message, std::string reason) {
  if (message != nullptr) {
    fused_attn_backend_message_buffer = std::move(reason);
    *message = fused_attn_backend_message_buffer.c_str();
  }
  return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
}

}  // namespace

// Fused attention backend query: runs TE's specific rules first, then cuDNN's. First rejection wins.
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend_v2(NVTEFusedAttnConfig config,
                                                       const char **message) {
  NVTE_API_CALL(nvte_get_fused_attn_backend_v2);
  using namespace transformer_engine;
  using namespace transformer_engine::fused_attn;
  FusedAttnConfig cfg = *get_fused_attn_config(config);
  cfg.derive();
  if (message != nullptr) *message = "";

  cudnnHandle_t handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const auto cudnn_runtime_version = cudnnGetVersion();
  const int sm_arch = cuda::sm_arch(cuda::current_device());

  // THD + 64-bit ragged offsets require cuDNN >= 9.5
  if (cfg.needs_64bit_ragged_offset && cudnn_runtime_version < 90500) {
    return reject(
        message,
        "This config requires 64-bit ragged offsets, which is only supported by cuDNN >= 9.5.");
  }

  // THD input requires a padding mask
  if ((cfg.is_ragged_q || cfg.is_ragged_kv) && !cfg.is_padding) {
    return reject(
        message,
        "THD format requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
  }

  if ((cfg.is_ragged_q && cfg.num_tokens_q == 0) || (cfg.is_ragged_kv && cfg.num_tokens_kv == 0)) {
    return reject(message,
                  "THD format requires num_tokens_q / num_tokens_kv to be set for the ragged "
                  "inputs.");
  }

  // Paged KV requires a padding mask
  if (cfg.is_paged_kv && !cfg.is_padding) {
    return reject(message,
                  "Paged KV requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
  }

  // Paged KV requires cache dimensions to be set
  if (cfg.is_paged_kv &&
      (cfg.num_pages_k == 0 || cfg.num_pages_v == 0 || cfg.page_size_k == 0 ||
       cfg.page_size_v == 0 || cfg.max_pages_per_seq_k == 0 || cfg.max_pages_per_seq_v == 0)) {
    return reject(message,
                  "Paged KV requires num_pages, page_size and max_pages_per_seq to be set for both "
                  "K and V.");
  }

  // Fused-attention does not support pre-scale bias
  if (cfg.bias_type == NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS) {
    return reject(message, "Fused attention does not support pre-scale bias.");
  }

  const bool is_fp8 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat8E4M3 || cfg.qkv_dtype == NVTEDType::kNVTEFloat8E5M2);
  const bool is_f16_or_bf16 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat16 || cfg.qkv_dtype == NVTEDType::kNVTEBFloat16);

  auto each_pass = [&](auto &&verdict) -> std::string {
    if (cfg.check_for_forward_support) {
      std::string reason = verdict(Pass::Fwd);
      if (!reason.empty()) return reason;
    }
    if (cfg.is_training && cfg.check_for_backward_support) {
      std::string reason = verdict(Pass::Bwd);
      if (!reason.empty()) return reason;
    }
    return "";
  };

  // F16/BF16 support checks
  if (is_f16_or_bf16) {
    if (cfg.is_ragged_q && cfg.is_ragged_kv && sm_arch < 90) {
      return reject(message, "F16/BF16 fused attention with THD format requires sm90 or later.");
    }
    if ((cfg.is_ragged_q || cfg.is_ragged_kv) && sm_arch < 90 && cudnn_runtime_version < 90700) {
      return reject(message,
                    "F16/BF16 fused attention with a ragged Q and non-ragged KV requires cuDNN "
                    ">= 9.7 before sm90.");
    }
    const bool has_sliding_window = !(cfg.window_size_left == -1 &&
                                      (cfg.window_size_right == -1 || cfg.window_size_right == 0));
    if (cfg.is_causal_bottom_right && has_sliding_window && cfg.max_seqlen_q != cfg.max_seqlen_kv &&
        cudnn_runtime_version <= 90700 && sm_arch >= 100) {
      return reject(message,
                    "Known cuDNN <= 9.7.0 issue with bottom-right causal masking and a sliding "
                    "window for cross-attention on sm100. Please upgrade cuDNN.");
    }
    if (cudnn_runtime_version <= 91500 && cfg.is_training &&
        (cfg.qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
         cfg.qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        (cfg.max_seqlen_kv % 128 != 0) && cfg.cuda_graph && !cfg.is_padding) {
      return reject(message, "Known cuDNN <= 9.15 issue with CUDA graph. Please upgrade cuDNN.");
    }
    if (cfg.is_training && cfg.check_for_backward_support && cfg.uses_ragged_stats &&
        cfg.softmax_type == NVTE_Softmax_Type::NVTE_LEARNABLE_SOFTMAX &&
        cudnn_runtime_version < 92600) {
      return reject(
          message,
          "Known cuDNN < 9.26.0 issue with THD learnable softmax backward. Please upgrade cuDNN.");
    }

    // Run cuDNN support checks
    std::string cudnn_reason =
        each_pass([&](Pass pass) { return support_verdict_f16(cfg, pass, handle); });
    if (!cudnn_reason.empty()) return reject(message, std::move(cudnn_reason));
    return NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
  }

  // FP8 support checks
  if (is_fp8) {
    if (cfg.return_max_logit) {
      return reject(message, "FP8 fused attention does not support return_max_logit=True.");
    }
    if (cfg.qkv_format != NVTE_QKV_Format::NVTE_BSHD &&
        cfg.qkv_format != NVTE_QKV_Format::NVTE_SBHD &&
        cfg.qkv_format != NVTE_QKV_Format::NVTE_BHSD &&
        cfg.qkv_format != NVTE_QKV_Format::NVTE_THD) {
      return reject(message, "FP8 fused attention supports BSHD/SBHD/BHSD/THD formats, found " +
                                 std::to_string(static_cast<int>(cfg.qkv_format)) + ".");
    }
    if (cfg.is_training && cfg.check_for_backward_support &&
        cfg.dqkv_layout != NVTE_QKV_Layout_NOT_SET) {
      const NVTE_QKV_Format dqkv_format = nvte_get_qkv_format(cfg.dqkv_layout);
      if (dqkv_format != NVTE_QKV_Format::NVTE_BSHD && dqkv_format != NVTE_QKV_Format::NVTE_SBHD &&
          dqkv_format != NVTE_QKV_Format::NVTE_BHSD && dqkv_format != NVTE_QKV_Format::NVTE_THD) {
        return reject(message,
                      "FP8 fused attention supports BSHD/SBHD/BHSD/THD gradient formats, found " +
                          std::to_string(static_cast<int>(dqkv_format)) + ".");
      }
    }
    if (cfg.qkv_format == NVTE_QKV_Format::NVTE_THD) {
      if (cudnn_runtime_version < 92300) {
        return reject(message,
                      "FP8 fused attention with THD format requires cuDNN 9.23.0 or later!");
      }
      if (cfg.is_training && cfg.check_for_backward_support && sm_arch < 100) {
        return reject(message,
                      "FP8 fused attention with THD format supports backward on sm100+ only!");
      }
      if (cfg.is_training && cfg.check_for_backward_support &&
          cfg.softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX &&
          cudnn_runtime_version < 92600) {
        return reject(message,
                      "FP8 fused attention with THD format and a sink token requires cuDNN 9.26.0 "
                      "or later for backward!");
      }
      if (sm_arch >= 100 && (cfg.head_dim_qk > 128 || cfg.head_dim_v > 128)) {
        return reject(message,
                      "FP8 fused attention with THD format supports head dimensions up to 128 on "
                      "sm100+ only!");
      }
    }
    if (cfg.is_bias) {
      return reject(message, "FP8 fused attention does not support pre/post_scale_bias yet!");
    }
    if (cfg.is_alibi) {
      return reject(message, "FP8 fused attention does not support ALiBi yet!");
    }
    const char *const recipe_reason =
        "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 "
        "recipes!";
    if (cfg.check_for_forward_support &&
        !(cfg.is_delayed_scaling_fwd || cfg.is_current_scaling_fwd || cfg.is_mxfp8_fwd)) {
      return reject(message, recipe_reason);
    }
    if (cfg.is_training && cfg.check_for_backward_support &&
        !(cfg.is_delayed_scaling_bwd || cfg.is_current_scaling_bwd || cfg.is_mxfp8_bwd)) {
      return reject(message, recipe_reason);
    }
    if (cfg.is_mxfp8 && cudnn_runtime_version < 92100) {
      return reject(message, "MXFP8 fused attention requires cuDNN 9.21.0 or later!");
    }

    // Run cuDNN support checks
    std::string cudnn_reason =
        each_pass([&](Pass pass) { return support_verdict_fp8(cfg, pass, handle); });
    if (!cudnn_reason.empty()) return reject(message, std::move(cudnn_reason));
    return NVTE_Fused_Attn_Backend::NVTE_FP8;
  }

  // Unsupported dtype
  return reject(message, "Unsupported QKV dtype qkv_dtype=" + std::to_string(cfg.qkv_dtype) + " .");
}

// Select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic) {
  NVTE_API_CALL(nvte_get_fused_attn_backend);
  transformer_engine::fused_attn::FusedAttnConfig cfg{};
  cfg.qkv_layout = qkv_layout;
  cfg.bias_type = bias_type;
  cfg.attn_mask_type = attn_mask_type;
  cfg.softmax_type = softmax_type;
  cfg.dropout = dropout;
  cfg.max_seqlen_q = max_seqlen_q;
  cfg.max_seqlen_kv = max_seqlen_kv;
  cfg.window_size_left = window_size_left;
  cfg.window_size_right = window_size_right;
  cfg.cuda_graph = cuda_graph;
  NVTE_CHECK(q_dtype == kv_dtype, "Q and KV must have the same data type.");
<<<<<<< HEAD
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  const bool is_thd_layout =
      q_format == NVTE_QKV_Format::NVTE_THD || kv_format == NVTE_QKV_Format::NVTE_THD;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  auto cudnn_runtime_version = cudnnGetVersion();

  // For ragged offsets we only support 32-bit prior to cuDNN 9.5
  // Only used when THD format is requested.
  const bool requires_64bit_ragged_offset =
      (qkv_format == NVTE_THD && fused_attn::get_ragged_offset_dtype(
                                     layout_group, num_attn_heads, num_gqa_groups, max_seqlen_q,
                                     max_seqlen_kv, head_dim_qk, head_dim_v) == DType::kInt64);
  const bool supported_ragged_offset_size =
      (!requires_64bit_ragged_offset || cudnn_runtime_version >= 90500);

  if ((q_dtype == NVTEDType::kNVTEFloat8E4M3 || q_dtype == NVTEDType::kNVTEFloat8E5M2) &&
      sm_arch_ >= 90 && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
      (
          // 9.2.1: {bshd, sbhd}, any seqlen, d=128, {no_mask, causal}
          (cudnn_runtime_version >= 90201 && sm_arch_ < 100 && max_seqlen_q % 128 == 0 &&
           max_seqlen_kv % 128 == 0 && head_dim_qk == 128 && head_dim_v == 128 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) ||
          // 9.7: {bshd, sbhd}, any seqlen, d<=256 for sm90 and d<=128 for sm100, {padding, padding_causal}
          (cudnn_runtime_version >= 90700 &&
           // TODO (cyang): add is_training to nvte_get_fused_attn_backend
           // sm90: fwd d<=256, bwd d=128 only
           // sm100: fwd d<=128, bwd d<=128
           ((sm_arch_ < 100 && (!is_training) && head_dim_qk <= 256 && head_dim_v <= 256) ||
            (sm_arch_ < 100 && is_training && head_dim_qk == 128 && head_dim_v == 128) ||
            (sm_arch_ >= 100 && head_dim_qk <= 128 && head_dim_v <= 128)) &&
           head_dim_qk % 16 == 0 && head_dim_v % 16 == 0 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
            (sm_arch_ >= 100 &&
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK))) ||
          // 9.21: d_qk=192, d_v=128
          (cudnn_runtime_version >= 92100 && sm_arch_ >= 100 && head_dim_qk <= 192 &&
           head_dim_v <= 128 && head_dim_qk % 16 == 0 && head_dim_v % 16 == 0 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK))) &&
      // pre-9.21: {bshd, sbhd}, {vanilla}
      // 9.21+: {bshd, sbhd, bhsd}, {vanilla, off-by-one, learnable}
      // 9.23+: {thd}; sm90 fwd only, sm100+ fwd/bwd
      ((cudnn_runtime_version < 92100 &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        softmax_type == NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX) ||
       (cudnn_runtime_version >= 92100 &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD ||
         qkv_format == NVTE_QKV_Format::NVTE_BHSD)) ||
       ((cudnn_runtime_version >= 92300 && (sm_arch_ >= 100 || (sm_arch_ >= 90 && !is_training))) &&
        qkv_format == NVTE_QKV_Format::NVTE_THD && supported_ragged_offset_size &&
        (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
         attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
         attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK))) &&
      // 9.10.0: known bugs with SDPA FP8
      (cudnn_runtime_version != 91000) && !return_max_logit) {
    backend = NVTE_Fused_Attn_Backend::NVTE_FP8;
  } else if ((q_dtype == NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16)) {
    bool flag_arb = false;
    if (
        // TODO(cyang): replace with cudnn-frontend check_support for cleaner logic and better error messaging
        // architecture
        ((cudnn_runtime_version < 8903 && (sm_arch_ == 80 || sm_arch_ == 90)) ||
         (cudnn_runtime_version >= 8903 && sm_arch_ >= 80 && sm_arch_ < 100) ||
         (cudnn_runtime_version >= 90700 && sm_arch_ >= 100)) &&
        // sequence length
        ((cudnn_runtime_version < 90000 && max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0) ||
         (cudnn_runtime_version >= 90000)) &&
        // number of heads
        ((cudnn_runtime_version < 8907 && num_attn_heads == num_gqa_groups) ||
         (cudnn_runtime_version >= 8907)) &&
        // head dimension
        // multiples of 8
        (head_dim_qk % 8 == 0 && head_dim_v % 8 == 0 &&
         // <= 128
         ((head_dim_qk <= 128 && head_dim_v <= 128) ||
          // 9.1: <= 256 + Hopper + fprop
          // 9.5: <= 256 + Hopper + bprop
          (head_dim_qk <= 256 && head_dim_v <= 256 &&
           ((!is_training && sm_arch_ == 90 && cudnn_runtime_version >= 90100) ||
            (is_training && sm_arch_ == 90 && cudnn_runtime_version >= 90500))) ||
          // 9.9: any head_dim + Blackwell + fprop + non_paged + sq > 1
          (!is_training && sm_arch_ >= 100 && cudnn_runtime_version >= 90900 && max_seqlen_q > 1 &&
           layout_group != NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD) ||
          // 9.10.2: any head_dim + any arch + fprop + paged
          // 9.10.2: any head_dim + any arch + fprop + non_paged + sq > 1
          // 9.10.2: any head_dim + any arch + fprop + non_paged + sq = 1 + {no_mask, padding, BRCM, padding_BRCM}
          (!is_training && cudnn_runtime_version >= 91002 &&
           (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD || max_seqlen_q > 1 ||
            (max_seqlen_q == 1 && attn_mask_type != NVTE_Mask_Type::NVTE_CAUSAL_MASK &&
             attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK))) ||
          // 9.11: d_qk = 192, d_v = 128 + Blackwell + bprop + non-paged
          (head_dim_qk == 192 && head_dim_v == 128 && is_training && sm_arch_ >= 100 &&
           cudnn_runtime_version >= 91100) ||
          // 9.23: d_qk = d_v = 256 + SM10x (cuDNN FE 1.24 / BE 9.23+) + bprop + non-paged.
          // THD layouts require cuDNN FE 1.26 / BE 9.25+ for execution-plan support.
          (head_dim_qk == 256 && head_dim_v == 256 && is_training && sm_arch_ >= 100 &&
           sm_arch_ < 110 && cudnn_runtime_version >= (is_thd_layout ? 92500 : 92300) &&
           layout_group != NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD &&
           // The FE forces this path onto the deterministic bprop algorithm, which on
           // Blackwell rejects dBias, dropout, and ALiBi (and supports vanilla softmax only).
           bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0 &&
           softmax_type == NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX &&
           // Non-causal D=256 supports only full-window attention; SWA is allowed only for causal masks.
           ((window_size_left == -1 && window_size_right == -1) ||
            ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
              attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
              attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK ||
              attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) &&
             (window_size_right == -1 || window_size_right == 0))))) &&
         // 9.11+ bug: 128 < d_qk <= 256, 128 < d_v <= 256 + Hopper + bprop + MLA
         // Conditional to temporarily use blanket cudnn_runtime_version >= 9.11 until fixed
         (!((cudnn_runtime_version >= 91100) && is_training && sm_arch_ == 90 &&
            head_dim_qk >= 128 && head_dim_v >= 128 && !(head_dim_qk == 192 && head_dim_v == 128) &&
            head_dim_qk != head_dim_v))) &&
        // bias type
        ((cudnn_runtime_version < 8906 && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) ||
         (cudnn_runtime_version >= 8906 &&
          (bias_type == NVTE_Bias_Type::NVTE_NO_BIAS ||
           (bias_type == NVTE_Bias_Type::NVTE_ALIBI &&
            attn_mask_type != NVTE_Mask_Type::NVTE_NO_MASK &&
            attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
            attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
            attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
            sm_arch_ >= 90) ||
           (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && sm_arch_ >= 90))) ||
         (cudnn_runtime_version >= 90000 &&
          (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && sm_arch_ >= 80))) &&
        // mask type
        // pre-8.9.6: causal
        ((cudnn_runtime_version < 8906 && attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
         // 8.9.6: {bshd, sbhd} + {no_mask, causal, padding, padding_causal}
         (cudnn_runtime_version >= 8906 &&
          (qkv_format == NVTE_QKV_Format::NVTE_SBHD || qkv_format == NVTE_QKV_Format::NVTE_BSHD) &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) ||
         // 9.1: adds thd + {padding, padding_causal}
         (cudnn_runtime_version >= 90100 && qkv_format == NVTE_QKV_Format::NVTE_THD &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)) ||
         // 9.3: adds {bshd, sbhd} + causal_bottom_right + self/cross-attn (sq <= skv)
         (cudnn_runtime_version >= 90300 &&
          (qkv_format == NVTE_QKV_Format::NVTE_SBHD || qkv_format == NVTE_QKV_Format::NVTE_BSHD) &&
          attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
          max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0 && max_seqlen_q <= max_seqlen_kv &&
          bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
         // 9.5: adds {paged_kv_bshd, paged_kv_sbhd} + {padding, padding_causal, padding_causal_bottom_right}
         (cudnn_runtime_version >= 90500 &&
          layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
           (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
            max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0 && max_seqlen_q <= max_seqlen_kv)) &&
          bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
         // 9.6: adds {bshd, sbhd, thd} + padding_causal_bottom_right + self/cross-attn (sq <= skv)
         (cudnn_runtime_version >= 90600 &&
          attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
          max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0 && max_seqlen_q <= max_seqlen_kv &&
          bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
         // 9.7: removes s_q/s_kv % 64 = 0 for {causal_bottom_right, padding_causal_bottom_right}
         // for any q_format/kv_format, and paged/non-paged
         (cudnn_runtime_version >= 90700 &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
           ((attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) &&
            bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
           ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) &&
            max_seqlen_q <= max_seqlen_kv)))) &&
        // bias + mask combination
        (!(cudnn_runtime_version >= 8906 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) &&
           bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)) &&
        // qkv format
        (qkv_format == NVTE_QKV_Format::NVTE_SBHD || qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
         qkv_format == NVTE_QKV_Format::NVTE_BHSD ||
         (qkv_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90 &&
          ((cudnn_runtime_version >= 90100 && num_attn_heads == num_gqa_groups) ||
           cudnn_runtime_version >= 90600)) ||
         ((q_format == NVTE_QKV_Format::NVTE_SBHD || q_format == NVTE_QKV_Format::NVTE_BSHD ||
           q_format == NVTE_QKV_Format::NVTE_BHSD ||
           (q_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90) ||
           kv_format == NVTE_QKV_Format::NVTE_SBHD || kv_format == NVTE_QKV_Format::NVTE_BSHD ||
           kv_format == NVTE_QKV_Format::NVTE_BHSD ||
           (kv_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90)) &&
          cudnn_runtime_version >= 90700)) &&
        // sliding window
        // pre-9.2: full attn, causal
        ((cudnn_runtime_version < 90200 && window_size_left == -1 &&
          (window_size_right == -1 || window_size_right == 0)) ||
         // 9.2: SWA (left, 0) + top-left diagonal + {bshd, sbhd}
         (cudnn_runtime_version >= 90200 &&
          ((window_size_left == -1 && window_size_right == -1 &&
            attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK) ||
           ((window_size_left == -1 || window_size_left >= 0) && window_size_right == 0 &&
            (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
             (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
              max_seqlen_q == max_seqlen_kv)) &&
            max_seqlen_q <= max_seqlen_kv && dropout == 0.0 &&
            bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
            (qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
             qkv_format == NVTE_QKV_Format::NVTE_SBHD)))) ||
         // 9.6: SWA (left, 0) + top-left/bottom-right diagonal + {bshd, sbhd, thd}
         (cudnn_runtime_version >= 90600 &&
          ((window_size_left == -1 && (window_size_right == -1 || window_size_right == 0)) ||
           ((window_size_left >= 0 || window_size_left == -1) &&
            (window_size_right >= 0 || window_size_right == -1) &&
            ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
              // TODO(cyang): fix bug for BRCM + cross-attention on sm100
              (sm_arch_ < 100 || (sm_arch_ >= 100 && ((max_seqlen_q == max_seqlen_kv &&
                                                       cudnn_runtime_version <= 90700) ||
                                                      cudnn_runtime_version > 90700)))) ||
             attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
             (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
              (sm_arch_ < 100 || (sm_arch_ >= 100 && ((max_seqlen_q == max_seqlen_kv &&
                                                       cudnn_runtime_version <= 90700) ||
                                                      cudnn_runtime_version > 90700))))) &&
            max_seqlen_q <= max_seqlen_kv && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
            dropout == 0.0)))) &&
        // check 64-bit ragged offset support
        (supported_ragged_offset_size) &&
        // 9.10.0/9.10.1: known bugs with SDPA F16
        (cudnn_runtime_version != 91000) && (cudnn_runtime_version != 91001) &&
        // softmax type
        // pre-9.13.1: vanilla
        // 9.13.1+: vanilla, off-by-one, learnable
        (cudnn_runtime_version >= 91301 ||
         (cudnn_runtime_version < 91301 &&
          softmax_type == NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX)) &&
        // max_logit
        // pre-9.21: no (the composite softmax node rejects the Stats + Max output combination)
        // 9.21+: yes (Stats + Max via the unified softmax node)
        (!return_max_logit || cudnn_runtime_version >= 92100) &&
        // determinism on Blackwell
        // pre-9.18.1: fwd: deterministic; bwd: non-deterministic
        // 9.18.1+: fwd: deterministic; bwd: non-deterministic/deterministic
        (sm_arch_ < 100 ||
         (sm_arch_ >= 100 && (!is_training ||
                              (is_training && !deterministic &&
                               (dropout == 0.0 || bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)) ||
                              (is_training && deterministic && cudnn_runtime_version >= 91801 &&
                               dropout == 0.0 && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS))))) {
      flag_arb = true;
    }
    if (flag_arb) {
      backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
    }
    if (cudnn_runtime_version < 8900 &&
        backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: FP16/BF16 fused attention is supported by cuDNN 8.9.0+."
                   " Please upgrade your cuDNN version if possible."
                << std::endl;
    }
    if ((cudnn_runtime_version == 91400) && (max_seqlen_kv > 1024) && (window_size_left != -1) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_CAUSAL_MASK) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK)) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: Given combination of attention mask (non-causal) and "
                   "max_seqlen_kv (> 1024) does not support fused attention for cuDNN 9.14.0. "
                   " Please upgrade your cuDNN version if possible."
                << std::endl;
    }
    if ((cudnn_runtime_version <= 91500) && is_training &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        (max_seqlen_kv % 128 != 0) && cuda_graph &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK)) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: Given combination of attention mask (non-padding),"
                   " max_seqlen_kv (not divisible by 128), and qkv_format (BSHD/SBHD) for"
                   " backward fused attention with graph capture requires cuDNN 9.15.1+. "
                   "Please upgrade your cuDNN version if possible."
                << std::endl;
    }
    if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen && sm_arch_ == 120) {
      if (cudnn_runtime_version < 91801) {
        backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
        std::cout << "Warning: Given combination of sm_arch_ == 120 and cudnn_runtime_version < "
                     "91801 is not supported. "
                  << " Please upgrade your cuDNN version if possible." << std::endl;
      } else if (deterministic && is_training) {
        backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
        std::cout << "Warning: Deterministic fused attention on SM120 is not supported."
                  << std::endl;
      } else {
        // Known missing support for T3HD/TH3D layouts on SM120
        const bool is_t3hd_or_th3d =
            (qkv_layout == NVTE_QKV_Layout::NVTE_T3HD || qkv_layout == NVTE_QKV_Layout::NVTE_TH3D);
        if (is_t3hd_or_th3d) {
          backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
          std::cout << "Warning: Given combination of T3HD/TH3D layouts on SM120 is not supported. "
                    << " Please consider using other THD layouts if possible." << std::endl;
        }
      }
    }
  } else {
    backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
=======
  cfg.qkv_dtype = q_dtype;
  cfg.o_dtype = q_dtype;
  cfg.do_dtype = q_dtype;
  cfg.dqkv_dtype = q_dtype;
  cfg.num_attn_heads = num_attn_heads;
  cfg.num_gqa_groups = num_gqa_groups;
  cfg.head_dim_qk = head_dim_qk;
  cfg.head_dim_v = head_dim_v;
  cfg.is_training = is_training;
  cfg.return_max_logit = return_max_logit;
  cfg.deterministic = deterministic;
  // fill in the missing fields with the most common use case;
  // otherwise it would return NVTE_No_Backend always
  cfg.batch_size = 1;
  cfg.num_tokens_q = cfg.batch_size * max_seqlen_q;
  cfg.num_tokens_kv = cfg.batch_size * max_seqlen_kv;
  cfg.o_format = nvte_get_q_format(qkv_layout);
  cfg.do_format = cfg.o_format;
  cfg.dqkv_layout = qkv_layout;
  if (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
    cfg.bias_batch_size = cfg.batch_size;
    cfg.bias_num_heads = num_attn_heads;
    cfg.bias_seqlen_q = max_seqlen_q;
    cfg.bias_seqlen_kv = max_seqlen_kv;
  }

  return nvte_get_fused_attn_backend_v2(reinterpret_cast<NVTEFusedAttnConfig>(&cfg),
                                        /*message=*/nullptr);
}

// Fused attention forward: create a config based on the params, check which backend supports it,
// and run that backend's implementation.
void nvte_fused_attn_fwd_v2(NVTEFusedAttnFwdParams params) {
  NVTE_API_CALL(nvte_fused_attn_fwd_v2);
  using namespace transformer_engine;
  using namespace transformer_engine::fused_attn;
  const FusedAttnFwdParams &p = *get_fused_attn_fwd_params(params);
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(p.cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(p.cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(p.cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(p.cu_seqlens_kv_padded);
  const Tensor *input_page_table_k = convertNVTETensorCheck(p.page_table_k);
  const Tensor *input_page_table_v = convertNVTETensorCheck(p.page_table_v);
  const Tensor *input_rng_state = convertNVTETensorCheck(p.rng_state);
  const Tensor *input_Q = convertNVTETensorCheck(p.Q);
  const Tensor *input_K = convertNVTETensorCheck(p.K);
  const Tensor *input_V = convertNVTETensorCheck(p.V);
  const Tensor *input_Bias = convertNVTETensorCheck(p.Bias);
  const Tensor *input_SoftmaxOffset = convertNVTETensorCheck(p.SoftmaxOffset);
  Tensor *input_output_S = convertNVTETensorCheck(p.S);
  Tensor *output_O = convertNVTETensorCheck(p.O);
  Tensor *wkspace = convertNVTETensor(p.workspace);

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  FusedAttnConfig cfg = p.make_config();
  cfg.derive();
  const char *fused_attn_reject_reason = nullptr;
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend_v2(
      reinterpret_cast<NVTEFusedAttnConfig>(&cfg), &fused_attn_reject_reason);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    fused_attn_arbitrary_seqlen_fwd(cfg, input_Q, input_K, input_V, input_Bias, input_SoftmaxOffset,
                                    output_O, p.Aux_CTX_Tensors, input_cu_seqlens_q,
                                    input_cu_seqlens_kv, input_cu_seqlens_q_padded,
                                    input_cu_seqlens_kv_padded, input_page_table_k,
                                    input_page_table_v, input_rng_state, wkspace, p.stream, handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    fused_attn_fp8_fwd(cfg, input_Q, input_K, input_V, input_SoftmaxOffset, input_output_S,
                       output_O, p.Aux_CTX_Tensors, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, input_rng_state,
                       wkspace, p.stream, handle);
  } else {
    NVTE_ERROR("Fused attention is not supported for the user configuration: ",
               fused_attn_reject_reason);
>>>>>>> 796346c0e0497b1f56a8d36ec76e02db5a6fed47
  }
}

// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor Bias, const NVTETensor SoftmaxOffset, NVTETensor S,
                         NVTETensor O, NVTETensorPack *Aux_CTX_Tensors,
                         const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                         const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, const NVTETensor page_table_k,
                         const NVTETensor page_table_v, const NVTETensor rng_state,
                         size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                         bool return_max_logit, bool cuda_graph, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
                         NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type,
                         NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
                         int64_t window_size_left, int64_t window_size_right,
                         bool bottom_right_diagonal, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_attn_fwd);
  transformer_engine::fused_attn::FusedAttnFwdParams p{};
  p.Q = Q;
  p.K = K;
  p.V = V;
  p.Bias = Bias;
  p.SoftmaxOffset = SoftmaxOffset;
  p.S = S;
  p.O = O;
  p.Aux_CTX_Tensors = Aux_CTX_Tensors;
  p.cu_seqlens_q = cu_seqlens_q;
  p.cu_seqlens_kv = cu_seqlens_kv;
  p.cu_seqlens_q_padded = cu_seqlens_q_padded;
  p.cu_seqlens_kv_padded = cu_seqlens_kv_padded;
  p.page_table_k = page_table_k;
  p.page_table_v = page_table_v;
  p.rng_state = rng_state;
  p.is_training = is_training;
  p.cuda_graph = cuda_graph;
  p.return_max_logit = return_max_logit;
  p.attn_mask_type = attn_mask_type;
  p.bias_type = bias_type;
  p.window_size_left = window_size_left;
  p.window_size_right = window_size_right;
  p.bottom_right_diagonal = bottom_right_diagonal;
  p.softmax_type = softmax_type;
  p.dropout = dropout;
  p.attn_scale = attn_scale;
  p.qkv_layout = qkv_layout;
  p.o_format = o_format;
  p.qkv_scale_inv_format = qkv_scale_inv_format;
  p.max_seqlen_q = max_seqlen_q;
  p.max_seqlen_kv = max_seqlen_kv;
  p.workspace = workspace;
  p.stream = stream;
  nvte_fused_attn_fwd_v2(reinterpret_cast<NVTEFusedAttnFwdParams>(&p));
}

// Fused attention backward. Same shape as nvte_fused_attn_fwd_v2, whose comment sketches the path;
// this one asks the selector for backward support and probes the backward builders.
void nvte_fused_attn_bwd_v2(NVTEFusedAttnBwdParams params) {
  NVTE_API_CALL(nvte_fused_attn_bwd_v2);
  using namespace transformer_engine;
  using namespace transformer_engine::fused_attn;
  const FusedAttnBwdParams &p = *get_fused_attn_bwd_params(params);
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(p.cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(p.cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(p.cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(p.cu_seqlens_kv_padded);
  const Tensor *input_Q = convertNVTETensorCheck(p.Q);
  const Tensor *input_K = convertNVTETensorCheck(p.K);
  const Tensor *input_V = convertNVTETensorCheck(p.V);
  const Tensor *input_O = convertNVTETensorCheck(p.O);
  const Tensor *input_dO = convertNVTETensorCheck(p.dO);
  const Tensor *input_S = convertNVTETensorCheck(p.S);
  Tensor *input_output_dP = convertNVTETensorCheck(p.dP);
  Tensor *output_dQ = convertNVTETensorCheck(p.dQ);
  Tensor *output_dK = convertNVTETensorCheck(p.dK);
  Tensor *output_dV = convertNVTETensorCheck(p.dV);
  Tensor *output_dBias = convertNVTETensorCheck(p.dBias);
  Tensor *output_dSoftmaxOffset = convertNVTETensorCheck(p.dSoftmaxOffset);
  Tensor *wkspace = convertNVTETensor(p.workspace);

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  FusedAttnConfig cfg = p.make_config();
  // Derived here, not by the query below: the query works on its own copy, and it is this config
  // that goes on to the backend and must arrive with its derived fields filled in.
  cfg.derive();
  const char *fused_attn_reject_reason = nullptr;
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend_v2(
      reinterpret_cast<NVTEFusedAttnConfig>(&cfg), &fused_attn_reject_reason);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    size_t i = 0;
    Tensor *output_S = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_rng_state = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_Bias = nullptr, *input_SoftmaxOffset = nullptr;
    if ((p.bias_type != NVTE_NO_BIAS) && (p.bias_type != NVTE_ALIBI)) {
      input_Bias = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    if (p.softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_arbitrary_seqlen_bwd(
        cfg, input_Q, input_K, input_V, input_O, input_dO, input_Bias, input_SoftmaxOffset,
        output_S, output_dQ, output_dK, output_dV, output_dBias, output_dSoftmaxOffset,
        input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded,
        input_cu_seqlens_kv_padded, input_rng_state, wkspace, p.stream, handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
<<<<<<< HEAD
    fused_attn_fp8_fwd(b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, t_q, t_kv, is_training,
                       attn_scale, dropout, qkv_layout, o_format, qkv_scale_inv_format, bias_type,
                       attn_mask_type, softmax_type, window_size_left, window_size_right,
                       bottom_right_diagonal, input_Q, input_K, input_V, input_SoftmaxOffset,
                       input_output_S, output_O, Aux_CTX_Tensors, input_cu_seqlens_q,
                       input_cu_seqlens_kv, input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded,
                       input_rng_state, wkspace, stream, handle);
=======
    size_t i = 0;
    const Tensor *input_M = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_rng_state = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_SoftmaxOffset = nullptr;
    if (p.softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    const Tensor *input_dO_f16 = nullptr;
    if (input_dO->scaling_mode == NVTE_MXFP8_1D_SCALING) {
      input_dO_f16 = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_fp8_bwd(cfg, input_Q, input_K, input_V, input_O, input_dO, input_dO_f16, input_M,
                       input_S, input_SoftmaxOffset, input_output_dP, output_dQ, output_dK,
                       output_dV, output_dSoftmaxOffset, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, input_rng_state,
                       wkspace, p.stream, handle);
>>>>>>> 796346c0e0497b1f56a8d36ec76e02db5a6fed47
  } else {
    NVTE_ERROR("Fused attention is not supported for this configuration: ",
               fused_attn_reject_reason);
  }
}

// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor O, const NVTETensor dO, const NVTETensor S, NVTETensor dP,
                         const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ, NVTETensor dK,
                         NVTETensor dV, NVTETensor dBias, NVTETensor dSoftmaxOffset,
                         const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                         const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, size_t max_seqlen_q,
                         size_t max_seqlen_kv, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
                         NVTE_QKV_Format do_format, NVTE_QKV_Layout dqkv_layout,
                         NVTE_QKV_Format qkv_scale_inv_format, NVTE_QKV_Format do_scale_inv_format,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         NVTE_Softmax_Type softmax_type, int64_t window_size_left,
                         int64_t window_size_right, bool bottom_right_diagonal, bool deterministic,
                         bool cuda_graph, NVTETensor workspace, cudaStream_t stream) {
<<<<<<< HEAD
  NVTE_API_CALL(nvte_flash_attn_bwd);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(cu_seqlens_kv_padded);
  const Tensor *input_Q = convertNVTETensorCheck(Q);
  const Tensor *input_K = convertNVTETensorCheck(K);
  const Tensor *input_V = convertNVTETensorCheck(V);
  const Tensor *input_O = convertNVTETensorCheck(O);
  const Tensor *input_dO = convertNVTETensorCheck(dO);
  const Tensor *input_S = convertNVTETensorCheck(S);
  Tensor *input_output_dP = convertNVTETensorCheck(dP);
  Tensor *output_dQ = convertNVTETensorCheck(dQ);
  Tensor *output_dK = convertNVTETensorCheck(dK);
  Tensor *output_dV = convertNVTETensorCheck(dV);
  Tensor *output_dBias = convertNVTETensorCheck(dBias);
  Tensor *output_dSoftmaxOffset = convertNVTETensorCheck(dSoftmaxOffset);
  Tensor *wkspace = convertNVTETensor(workspace);

  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  auto *q_dims = input_Q->data.shape.data();
  auto *k_dims = input_K->data.shape.data();
  auto *v_dims = input_V->data.shape.data();
  AttentionShape q_shape(q_format, q_dims);
  AttentionShape k_shape(kv_format, k_dims);
  AttentionShape v_shape(kv_format, v_dims);
  size_t b = q_shape.b(), h_q = q_shape.h(), d_qk = q_shape.d(), t_q = q_shape.t();
  size_t h_kv = k_shape.h(), t_kv = k_shape.t(), d_v = v_shape.d();
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    b = input_cu_seqlens_q->data.shape[0] - 1;
  } else if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    b = input_cu_seqlens_kv->data.shape[0] - 1;
  }

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      true, Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, softmax_type, dropout, h_q,
      h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, window_size_left, window_size_right, false,
      cuda_graph, deterministic);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    size_t i = 0;
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_Bias, *input_SoftmaxOffset;
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      input_Bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_arbitrary_seqlen_bwd(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, t_q, t_kv, attn_scale, dropout,
        qkv_layout, o_format, do_format, dqkv_layout, bias_type, attn_mask_type, softmax_type,
        window_size_left, window_size_right, bottom_right_diagonal, deterministic, input_Q, input_K,
        input_V, input_O, input_dO, input_Bias, input_SoftmaxOffset, output_S, output_dQ, output_dK,
        output_dV, output_dBias, output_dSoftmaxOffset, input_cu_seqlens_q, input_cu_seqlens_kv,
        input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, input_rng_state, wkspace, stream,
        handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    size_t i = 0;
    const Tensor *input_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_SoftmaxOffset = nullptr;
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    const Tensor *input_dO_f16 = nullptr;
    if (input_dO->scaling_mode == NVTE_MXFP8_1D_SCALING) {
      input_dO_f16 = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_fp8_bwd(b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, t_q, t_kv, attn_scale,
                       dropout, qkv_layout, o_format, do_format, dqkv_layout, qkv_scale_inv_format,
                       do_scale_inv_format, bias_type, attn_mask_type, softmax_type,
                       window_size_left, window_size_right, bottom_right_diagonal, deterministic,
                       input_Q, input_K, input_V, input_O, input_dO, input_dO_f16, input_M, input_S,
                       input_SoftmaxOffset, input_output_dP, output_dQ, output_dK, output_dV,
                       output_dSoftmaxOffset, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, input_rng_state,
                       wkspace, stream, handle);
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
=======
  NVTE_API_CALL(nvte_fused_attn_bwd);
  transformer_engine::fused_attn::FusedAttnBwdParams p{};
  p.Q = Q;
  p.K = K;
  p.V = V;
  p.O = O;
  p.dO = dO;
  p.S = S;
  p.dP = dP;
  p.Aux_CTX_Tensors = Aux_CTX_Tensors;
  p.dQ = dQ;
  p.dK = dK;
  p.dV = dV;
  p.dBias = dBias;
  p.dSoftmaxOffset = dSoftmaxOffset;
  p.cu_seqlens_q = cu_seqlens_q;
  p.cu_seqlens_kv = cu_seqlens_kv;
  p.cu_seqlens_q_padded = cu_seqlens_q_padded;
  p.cu_seqlens_kv_padded = cu_seqlens_kv_padded;
  p.deterministic = deterministic;
  p.cuda_graph = cuda_graph;
  p.attn_mask_type = attn_mask_type;
  p.bias_type = bias_type;
  p.window_size_left = window_size_left;
  p.window_size_right = window_size_right;
  p.bottom_right_diagonal = bottom_right_diagonal;
  p.softmax_type = softmax_type;
  p.dropout = dropout;
  p.attn_scale = attn_scale;
  p.qkv_layout = qkv_layout;
  p.o_format = o_format;
  p.do_format = do_format;
  p.dqkv_layout = dqkv_layout;
  p.qkv_scale_inv_format = qkv_scale_inv_format;
  p.do_scale_inv_format = do_scale_inv_format;
  p.max_seqlen_q = max_seqlen_q;
  p.max_seqlen_kv = max_seqlen_kv;
  p.workspace = workspace;
  p.stream = stream;
  nvte_fused_attn_bwd_v2(reinterpret_cast<NVTEFusedAttnBwdParams>(&p));
>>>>>>> 796346c0e0497b1f56a8d36ec76e02db5a6fed47
}

uint32_t nvte_get_runtime_num_segments(NVTETensor cu_seqlen, NVTETensor workspace, size_t len,
                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_get_runtime_num_segments);
  using namespace transformer_engine::fused_attn;
  return GetRuntimeNumSegments(cu_seqlen, workspace, len, stream);
}

void nvte_populate_rng_state_async(NVTETensor rng_state_dst, const NVTETensor seed,
                                   size_t q_max_seqlen, size_t kv_max_seqlen,
                                   NVTE_Fused_Attn_Backend backend, cudaStream_t stream) {
  NVTE_API_CALL(nvte_populate_rng_state_async);
  using namespace transformer_engine::fused_attn;
  PopulateRngStateAsync(rng_state_dst, seed, q_max_seqlen, kv_max_seqlen, backend, stream);
}
