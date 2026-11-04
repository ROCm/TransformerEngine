/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include <iostream>
#include <string>
#ifdef USE_FUSED_ATTN_MI450
// ATTN_NS (the namespace of the library's exported symbols) comes from the mi450_attn target
#include <attn_abi.h>
#endif  // USE_FUSED_ATTN_MI450
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_mi450.h"
#include "utils.h"

#ifdef USE_FUSED_ATTN_MI450
namespace {

// Nothing is needed; a non-empty request makes the sizing pass (null workspace) unambiguous for every
// framework, including those that hand back a null buffer for a 0-byte workspace.
constexpr size_t kMI450WorkspaceBytes = sizeof(int32_t);

int to_attn_layout(NVTE_QKV_Layout qkv_layout) {
  switch (nvte_get_qkv_format(qkv_layout)) {
    case NVTE_QKV_Format::NVTE_BSHD:
      return ATTN_BSHD;
    case NVTE_QKV_Format::NVTE_SBHD:
      return ATTN_SBHD;
    case NVTE_QKV_Format::NVTE_THD:
      return ATTN_THD;
    default:
      NVTE_ERROR("MI450 fused attn: unsupported qkv_layout.");
  }
}

}  // namespace
#endif  // USE_FUSED_ATTN_MI450

namespace transformer_engine {
namespace fused_attn_rocm {

// check the fused attn config to see whether the mi450 backend supports it
//
// Two layers decide. First the conventions of the C ABI of te-mi450-attn-kernels
// (kernels/attn_abi.h), which TE has to map its config onto: bf16, no bias/dropout/sink,
// d_qk == d_v, separate contiguous Q/K/V in one of bshd/sbhd/thd (packed thd, i.e. no padding between
// sequences), no per-sequence padding for the batch layouts, and a bottom-right aligned (left, right) window
// with causal folded in as right = 0. Then the support queries of the library's dispatcher (attn_fwd_supported,
// plus attn_bwd_supported when training), which accept a config when one of its backends implements it.
bool is_mi450_backend_supported(
  bool is_training,
  bool deterministic,
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  NVTE_Softmax_Type softmax_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim_qk,
  size_t head_dim_v,
  int64_t window_size_left,
  int64_t window_size_right,
  std::string* reason) {

#ifdef USE_FUSED_ATTN_MI450
  const bool nvte_log_mi450_config = getenv<bool>("NVTE_LOG_MI450_CONFIG");
  auto reject = [&](const std::string& why) {
    if (nvte_log_mi450_config) {
      std::cout << "MI450 fused attn: " << why << std::endl;
    }
    if (reason != nullptr) {
      *reason = why;
    }
    return false;
  };

  // The kernels are built for gfx1250 only.
  if (cuda::sm_arch(cuda::current_device()) != 125) {
    return reject("requires gfx1250");
  }
  if (q_dtype != kv_dtype || q_dtype != NVTEDType::kNVTEBFloat16) {
    return reject("q, k, v data type has to be bf16");
  }
  if (bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) {
    return reject("bias is not supported");
  }
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    return reject("only vanilla softmax is supported");
  }
  if (dropout != 0.f) {
    return reject("dropout is not supported");
  }
  if (head_dim_qk != head_dim_v) {
    return reject("head_dim_qk must equal head_dim_v");
  }
  if (num_gqa_groups == 0 || num_attn_heads % num_gqa_groups != 0) {
    return reject("num of attention heads must be divisible by num of gqa groups");
  }

  // Q, K and V are separate tensors in the same format.
  if (!(qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD || qkv_layout == NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD ||
        qkv_layout == NVTE_QKV_Layout::NVTE_THD_THD_THD)) {
    return reject("qkv_layout must be one of bshd_bshd_bshd, sbhd_sbhd_sbhd, thd_thd_thd");
  }

  // THD carries the sequence lengths; the batch layouts have no per-sequence lengths in the ABI.
  const bool is_ragged  = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;
  const bool is_padding = is_padding_mask(attn_mask_type);
  if (is_ragged != is_padding) {
    return reject("thd requires a padding mask; bshd and sbhd require a non-padding mask");
  }

  // window size must follow check_set_window_size
  const bool is_causal = is_causal_mask(attn_mask_type);
  if (is_causal) {
    if (window_size_left < -1 || window_size_right != 0) {
      return reject("when mask contains causal, window size should be (-1, 0) or (>=0, 0)");
    }
  } else if (!((window_size_left == -1 && window_size_right == -1) ||
               (window_size_left >= 0 && window_size_right >= 0))) {
    return reject("when no causal mask, window size should be (-1, -1) or (>=0, >=0)");
  }

  // The ABI aligns masks bottom-right. Top-left causal is the same mask only when every sequence has
  // s_q == s_kv, which is known here for the batch layouts but not for thd.
  const bool is_top_left_causal = attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
                                  attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK;
  if (is_top_left_causal && (is_ragged || max_seqlen_q != max_seqlen_kv)) {
    return reject("top-left causal needs s_q == s_kv (use causal_bottom_right, or bshd/sbhd)");
  }

  // What the kernels implement, as declared by the library.
  const int layout = to_attn_layout(qkv_layout);
  const int h_q    = static_cast<int>(num_attn_heads);
  const int h_kv   = static_cast<int>(num_gqa_groups);
  const int d      = static_cast<int>(head_dim_qk);
  const int s_q    = static_cast<int>(max_seqlen_q);
  const int s_kv   = static_cast<int>(max_seqlen_kv);
  const int left   = static_cast<int>(window_size_left);
  const int right  = static_cast<int>(window_size_right);
  const int det    = deterministic ? 1 : 0;

  // The dispatcher's reason lists every backend's.
  const char* why = nullptr;
  if (ATTN_SYM(attn_fwd_supported)(layout, h_q, h_kv, d, s_q, s_kv, left, right, &why) != ATTN_OK) {
    return reject(std::string("attn_fwd_supported: ") + (why != nullptr ? why : "not supported"));
  }
  if (is_training &&
      ATTN_SYM(attn_bwd_supported)(layout, h_q, h_kv, d, s_q, s_kv, left, right, det, &why) != ATTN_OK) {
    return reject(std::string("attn_bwd_supported: ") + (why != nullptr ? why : "not supported"));
  }
  if (nvte_log_mi450_config) {
    std::cout << "MI450 fused attn: fwd on "
              << ATTN_SYM(attn_backend_name)(
                     ATTN_SYM(attn_fwd_backend)(layout, h_q, h_kv, d, s_q, s_kv, left, right, nullptr));
    if (is_training) {
      std::cout << ", bwd on "
                << ATTN_SYM(attn_backend_name)(
                       ATTN_SYM(attn_bwd_backend)(layout, h_q, h_kv, d, s_q, s_kv, left, right, det, nullptr));
    }
    std::cout << std::endl;
  }
  return true;
#else
  if (reason != nullptr) {
    *reason = "MI450 fused attn backend not compiled";
  }
  return false;
#endif  // USE_FUSED_ATTN_MI450
}

}  // namespace fused_attn_rocm

#ifdef USE_FUSED_ATTN_MI450
namespace {

// cu_seqlens for the ABI: null for the batch layouts, the packed prefix sums for thd.
std::pair<const int*, const int*> abi_cu_seqlens(
  bool is_ragged,
  const transformer_engine::Tensor* input_cu_seqlens_q,
  const transformer_engine::Tensor* input_cu_seqlens_kv,
  const transformer_engine::Tensor* input_cu_seqlens_q_padded,
  const transformer_engine::Tensor* input_cu_seqlens_kv_padded) {
  if (!is_ragged) {
    return {nullptr, nullptr};
  }
  // The ABI's thd tensors are packed: token offsets are cu_seqlens. A distinct padded offsets tensor may
  // hold other values, which the kernels cannot honor.
  auto is_packed = [](const transformer_engine::Tensor* cu_seqlens, const transformer_engine::Tensor* padded) {
    return padded == nullptr || padded->data.dptr == nullptr || padded->data.dptr == cu_seqlens->data.dptr;
  };
  NVTE_CHECK(is_packed(input_cu_seqlens_q, input_cu_seqlens_q_padded) &&
                 is_packed(input_cu_seqlens_kv, input_cu_seqlens_kv_padded),
             "MI450 fused attn: thd requires packed sequences (cu_seqlens_*_padded must be cu_seqlens_*).");
  return {static_cast<const int*>(input_cu_seqlens_q->data.dptr),
          static_cast<const int*>(input_cu_seqlens_kv->data.dptr)};
}

// The ABI aligns sliding windows bottom-right; a top-left window differs whenever s_q != s_kv.
void check_window_alignment(
  bool is_ragged, size_t max_seqlen_q, size_t max_seqlen_kv,
  int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal) {
  const bool has_window   = window_size_left >= 0 || window_size_right > 0;
  const bool may_be_cross = is_ragged || max_seqlen_q != max_seqlen_kv;
  NVTE_CHECK(!(has_window && may_be_cross && !bottom_right_diagonal),
             "MI450 fused attn: sliding window must be bottom-right aligned when s_q != s_kv.");
}

void check_status(int status, const char* entry) {
  if (status == ATTN_NOT_IMPLEMENTED) {
    NVTE_ERROR(std::string("MI450 fused attn: ") + entry + " returned ATTN_NOT_IMPLEMENTED for a configuration " +
               entry + "_supported accepted.");
  }
  NVTE_CHECK(status != ATTN_LAUNCH_ERROR, std::string("MI450 fused attn: ") + entry + " failed to launch.");
  NVTE_CHECK(status == ATTN_OK, std::string("MI450 fused attn: ") + entry + " returned " + std::to_string(status));
}

}  // namespace
#endif  // USE_FUSED_ATTN_MI450

using namespace transformer_engine::fused_attn_rocm;

void fused_attn_mi450_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
  NVTE_QKV_Layout qkv_layout,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
  Tensor* output_O, NVTETensorPack* Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream) {

#ifdef USE_FUSED_ATTN_MI450
  NVTE_CHECK(input_Q->data.dtype == DType::kBFloat16 && input_K->data.dtype == DType::kBFloat16 &&
                 input_V->data.dtype == DType::kBFloat16 && output_O->data.dtype == DType::kBFloat16,
             "MI450 fused attn requires bf16 Q, K, V and O.");

  const bool is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;
  // softmax LSE in the ABI layout: [b, h_q, s_q] for bshd/sbhd, [t_q, h_q] for thd
  const size_t max_tokens_q = input_Q->data.numel() / h_q / d;

  if (Aux_CTX_Tensors->size == 0) {
    Aux_CTX_Tensors->size = 2;
    Tensor* output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    output_S->data.dptr = nullptr;
    if (is_ragged) {
      output_S->data.shape = {max_tokens_q, h_q, 1};
    } else {
      output_S->data.shape = {b, h_q, max_seqlen_q, 1};
    }
    output_S->data.dtype = DType::kFloat32;
    Tensor* output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr  = nullptr;
    output_rng_state->data.shape = {2};
    output_rng_state->data.dtype = DType::kInt64;
    // sizing pass: the caller allocates the aux tensors and calls again
    set_workspace_size(workspace, kMI450WorkspaceBytes);
    return;
  }
  NVTE_CHECK(Aux_CTX_Tensors->size == 2, "Unexpected Aux_CTX_Tensors->size.");
  Tensor* output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
  Tensor* output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
  output_rng_state->data.dptr = rng_state->data.dptr;
  if (workspace->data.dptr == nullptr) {
    set_workspace_size(workspace, kMI450WorkspaceBytes);
    return;
  }

  check_window_alignment(is_ragged, max_seqlen_q, max_seqlen_kv, window_size_left, window_size_right,
                         bottom_right_diagonal);
  const auto [cu_seqlens_q, cu_seqlens_kv] = abi_cu_seqlens(
      is_ragged, input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded);

  const int status = ATTN_SYM(attn_fwd)(
      to_attn_layout(qkv_layout), input_Q->data.dptr, input_K->data.dptr, input_V->data.dptr, output_O->data.dptr,
      static_cast<float*>(output_S->data.dptr), cu_seqlens_q, cu_seqlens_kv, static_cast<int>(b),
      static_cast<int>(h_q), static_cast<int>(h_kv), static_cast<int>(d), static_cast<int>(max_seqlen_q),
      static_cast<int>(max_seqlen_kv), attn_scale, static_cast<int>(window_size_left),
      static_cast<int>(window_size_right), stream);
  check_status(status, "attn_fwd");
#else
  NVTE_ERROR("MI450 fused attn backend not compiled.");
#endif  // USE_FUSED_ATTN_MI450
}

void fused_attn_mi450_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
  float attn_scale, int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
  bool deterministic,
  NVTE_QKV_Layout qkv_layout,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
  const Tensor* input_O, const Tensor* input_dO, const Tensor* input_S,
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  Tensor* workspace,
  cudaStream_t stream) {

#ifdef USE_FUSED_ATTN_MI450
  if (workspace->data.dptr == nullptr) {
    // sizing pass
    set_workspace_size(workspace, kMI450WorkspaceBytes);
    return;
  }
  NVTE_CHECK(input_Q->data.dtype == DType::kBFloat16 && input_K->data.dtype == DType::kBFloat16 &&
                 input_V->data.dtype == DType::kBFloat16 && input_O->data.dtype == DType::kBFloat16 &&
                 input_dO->data.dtype == DType::kBFloat16,
             "MI450 fused attn requires bf16 Q, K, V, O and dO.");
  NVTE_CHECK(input_S->data.dtype == DType::kFloat32, "MI450 fused attn requires an fp32 softmax LSE.");

  const bool is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;
  check_window_alignment(is_ragged, max_seqlen_q, max_seqlen_kv, window_size_left, window_size_right,
                         bottom_right_diagonal);
  const auto [cu_seqlens_q, cu_seqlens_kv] = abi_cu_seqlens(
      is_ragged, input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded);

  const int status = ATTN_SYM(attn_bwd)(
      to_attn_layout(qkv_layout), input_Q->data.dptr, input_K->data.dptr, input_V->data.dptr, input_O->data.dptr,
      static_cast<const float*>(input_S->data.dptr), input_dO->data.dptr, output_dQ->data.dptr,
      output_dK->data.dptr, output_dV->data.dptr, cu_seqlens_q, cu_seqlens_kv, static_cast<int>(b),
      static_cast<int>(h_q), static_cast<int>(h_kv), static_cast<int>(d), static_cast<int>(max_seqlen_q),
      static_cast<int>(max_seqlen_kv), attn_scale, static_cast<int>(window_size_left),
      static_cast<int>(window_size_right), deterministic ? 1 : 0, stream);
  check_status(status, "attn_bwd");
#else
  NVTE_ERROR("MI450 fused attn backend not compiled.");
#endif  // USE_FUSED_ATTN_MI450
}

}  // namespace transformer_engine
