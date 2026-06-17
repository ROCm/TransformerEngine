/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "fused_attn_smallseq.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "utils.h"

#ifdef USE_FUSED_ATTN_CK
#include <ck_fused_attn/ck_fused_attn.hpp>
#include "attn_bwd_mfma_16x16.h"
#include "attn_fwd_mfma_dispatch.h"
#endif

namespace transformer_engine {
namespace fused_attn_rocm {

bool small_seq_static_config_ok(NVTEDType q_dtype,
                                NVTEDType kv_dtype,
                                NVTE_Bias_Type bias_type,
                                float dropout,
                                size_t head_dim_qk,
                                size_t head_dim_v,
                                size_t num_attn_heads,
                                size_t num_gqa_groups,
                                NVTE_Mask_Type mask_type) {
  if(dropout != 0.0f) return false;
  if(bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) return false;
  if(q_dtype != kv_dtype) return false;
  if(!(q_dtype == NVTEDType::kNVTEFloat16 || q_dtype == NVTEDType::kNVTEBFloat16)) return false;
  if(head_dim_qk != head_dim_v) return false;
  if(head_dim_qk != 128 && head_dim_qk != 256) return false;
  if(num_gqa_groups == 0 || num_attn_heads % num_gqa_groups != 0) return false;
  if(num_attn_heads != num_gqa_groups) return false;
  if(num_attn_heads != 16 && num_attn_heads != 32) return false;
  if(!(is_padding_mask(mask_type) || mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) return false;
  return true;
}

bool is_runtime_small_seq_eligible(size_t runtime_max_seqlen_q, size_t runtime_max_seqlen_kv) {
  return runtime_max_seqlen_q > 0 && runtime_max_seqlen_q <= kSmallSeqMaxSeqlen &&
         runtime_max_seqlen_kv > 0 && runtime_max_seqlen_kv <= kSmallSeqMaxSeqlen;
}

bool supports_hip_small_seq(size_t num_attn_heads,
                            size_t num_gqa_groups,
                            size_t head_dim_qk,
                            size_t head_dim_v) {
  if(num_attn_heads != num_gqa_groups) return false;
  if(num_attn_heads != 16 && num_attn_heads != 32) return false;
  if(head_dim_qk != head_dim_v) return false;
  return head_dim_qk == 128 || head_dim_qk == 256;
}

size_t small_seq_extra_workspace_bytes(size_t max_tokens_q) {
  return 2 * sizeof(uint64_t) + max_tokens_q * sizeof(int32_t);
}

bool is_nvte_ck_small_seq_enabled() {
  if (transformer_engine::cuda::sm_arch() != 94) {
    return false;
  }
  const char* env_p = std::getenv("NVTE_FUSED_ATTN_CK_SMALLSEQ");
  return env_p != nullptr && std::strcmp(env_p, "1") == 0;
}

#ifndef USE_FUSED_ATTN_CK

bool fused_attn_smallseq_fwd(size_t, size_t, size_t, size_t, size_t, float,
                             const void*, const void*, const void*, void*, void*,
                             const void*, const void*, const void*, const void*,
                             const void*, NVTEDType, cudaStream_t) {
  return false;
}

bool fused_attn_smallseq_bwd(size_t, size_t, size_t, size_t, size_t, float,
                             const void*, const void*, const void*, const void*, const void*,
                             void*, void*, void*,
                             const void*, const void*, const void*, const void*,
                             NVTEDType, cudaStream_t) {
  return false;
}

#else  // USE_FUSED_ATTN_CK

// HIP small-seq kernels: head dim 512 is not supported here — upstream instantiations exceed the
// 64 KiB LDS limit on CDNA (gfx942 / gfx950) for the 17×17 small-seq tile configuration.
namespace {

constexpr int kMaxBsInst = 16384;

hipStream_t to_hip_stream(cudaStream_t s) {
  return reinterpret_cast<hipStream_t>(s);
}

template <typename T, int HEAD_NUM, int HEAD_DIM>
bool launch_fwd_inst(size_t actual_batch,
                     float attn_scale,
                     const T* Q,
                     const T* K,
                     const T* V,
                     T* O,
                     float* softmax_lse,
                     const int* cu_q,
                     const int* cu_qp,
                     const int* cu_kv,
                     const int* cu_kvp,
                     const int* padded_q_to_batch,
                     int total_padded_q,
                     hipStream_t stream) {
  if(actual_batch > static_cast<size_t>(kMaxBsInst)) {
    return false;
  }
  using Config =
      FmhaKernelConfig<kMaxBsInst, HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnForwardMfmaDispatchLauncher<T, Config>;
  const float sqr_dk_scale = attn_scale / std::sqrt(static_cast<float>(HEAD_DIM));
  Launcher::run_attn_fwd_kernel(Q, K, V, nullptr, 0.0f, sqr_dk_scale, O, softmax_lse, cu_q, cu_qp,
                                cu_kv, cu_kvp, padded_q_to_batch, total_padded_q);
  NVTE_CHECK_CUDA(hipStreamSynchronize(stream));
  return true;
}

template <typename T>
bool launch_fwd_dispatch(size_t batch,
                        size_t num_heads,
                        int head_dim,
                        float attn_scale,
                        const T* Q,
                        const T* K,
                        const T* V,
                        T* O,
                        float* softmax_lse,
                        const int* cu_q,
                        const int* cu_qp,
                        const int* cu_kv,
                        const int* cu_kvp,
                        const int* padded_q_to_batch,
                        int total_padded_q,
                        hipStream_t stream) {
  if(num_heads == 16) {
    if(head_dim == 128) {
      return launch_fwd_inst<T, 16, 128>(batch, attn_scale, Q, K, V, O, softmax_lse, cu_q, cu_qp,
                                         cu_kv, cu_kvp, padded_q_to_batch, total_padded_q, stream);
    }
    if(head_dim == 256) {
      return launch_fwd_inst<T, 16, 256>(batch, attn_scale, Q, K, V, O, softmax_lse, cu_q, cu_qp,
                                         cu_kv, cu_kvp, padded_q_to_batch, total_padded_q, stream);
    }
  }
  if(num_heads == 32) {
    if(head_dim == 128) {
      return launch_fwd_inst<T, 32, 128>(batch, attn_scale, Q, K, V, O, softmax_lse, cu_q, cu_qp,
                                         cu_kv, cu_kvp, padded_q_to_batch, total_padded_q, stream);
    }
    if(head_dim == 256) {
      return launch_fwd_inst<T, 32, 256>(batch, attn_scale, Q, K, V, O, softmax_lse, cu_q, cu_qp,
                                         cu_kv, cu_kvp, padded_q_to_batch, total_padded_q, stream);
    }
  }
  return false;
}

template <typename T, int HEAD_NUM, int HEAD_DIM>
bool launch_bwd_inst(size_t actual_batch,
                     float attn_scale,
                     const T* Q,
                     const T* K,
                     const T* V,
                     const T* dO,
                     const float* softmax_lse,
                     T* dQ,
                     T* dK,
                     T* dV,
                     const int* cu_q,
                     const int* cu_qp,
                     const int* cu_kv,
                     const int* cu_kvp,
                     hipStream_t stream) {
  if(actual_batch > static_cast<size_t>(kMaxBsInst)) {
    return false;
  }
  using Config =
      FmhaKernelConfig<kMaxBsInst, HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnBackwardMfma16x16KernelLauncher<T, Config>;
  const float sqr_dk_scale = attn_scale / std::sqrt(static_cast<float>(HEAD_DIM));
  Launcher::run_attn_bwd_kernel(Q, K, V, dO, softmax_lse, dQ, dK, dV, sqr_dk_scale, cu_q, cu_qp,
                                cu_kv, cu_kvp);
  NVTE_CHECK_CUDA(hipStreamSynchronize(stream));
  return true;
}

template <typename T>
bool launch_bwd_dispatch(size_t batch,
                        size_t num_heads,
                        int head_dim,
                        float attn_scale,
                        const T* Q,
                        const T* K,
                        const T* V,
                        const T* dO,
                        const float* softmax_lse,
                        T* dQ,
                        T* dK,
                        T* dV,
                        const int* cu_q,
                        const int* cu_qp,
                        const int* cu_kv,
                        const int* cu_kvp,
                        hipStream_t stream) {
  if(num_heads == 16) {
    if(head_dim == 128) {
      return launch_bwd_inst<T, 16, 128>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                         cu_q, cu_qp, cu_kv, cu_kvp, stream);
    }
    if(head_dim == 256) {
      return launch_bwd_inst<T, 16, 256>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                         cu_q, cu_qp, cu_kv, cu_kvp, stream);
    }
  }
  if(num_heads == 32) {
    if(head_dim == 128) {
      return launch_bwd_inst<T, 32, 128>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                         cu_q, cu_qp, cu_kv, cu_kvp, stream);
    }
    if(head_dim == 256) {
      return launch_bwd_inst<T, 32, 256>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                         cu_q, cu_qp, cu_kv, cu_kvp, stream);
    }
  }
  return false;
}

}  // namespace

bool fused_attn_smallseq_fwd(size_t batch_size,
                             size_t num_heads,
                             size_t head_dim_qk,
                             size_t max_tokens_q,
                             size_t max_tokens_kv,
                             float attn_scale,
                             const void* dev_ptr_q,
                             const void* dev_ptr_k,
                             const void* dev_ptr_v,
                             void* dev_ptr_o,
                             void* dev_ptr_softmax_lse,
                             const void* dev_ptr_cu_seqlens_q,
                             const void* dev_ptr_cu_seqlens_q_padded,
                             const void* dev_ptr_cu_seqlens_kv,
                             const void* dev_ptr_cu_seqlens_kv_padded,
                             const void* dev_ptr_padded_q_to_batch,
                             NVTEDType dtype,
                             cudaStream_t stream) {
  (void)max_tokens_kv;
  const int* cu_q = static_cast<const int*>(dev_ptr_cu_seqlens_q);
  const int* cu_qp = static_cast<const int*>(dev_ptr_cu_seqlens_q_padded);
  const int* cu_kv = static_cast<const int*>(dev_ptr_cu_seqlens_kv);
  const int* cu_kvp = static_cast<const int*>(dev_ptr_cu_seqlens_kv_padded);
  const int* padded_q_to_batch = static_cast<const int*>(dev_ptr_padded_q_to_batch);
  float* softmax_lse = static_cast<float*>(dev_ptr_softmax_lse);
  const int total_padded_q = static_cast<int>(max_tokens_q);
  const int hd = static_cast<int>(head_dim_qk);
  const hipStream_t hip_stream = to_hip_stream(stream);

  if(!supports_hip_small_seq(num_heads, num_heads, head_dim_qk, head_dim_qk)) {
    return false;
  }

  if(dtype == NVTEDType::kNVTEBFloat16) {
    using T = hip_bfloat16;
    const T* Q = static_cast<const T*>(dev_ptr_q);
    const T* K = static_cast<const T*>(dev_ptr_k);
    const T* V = static_cast<const T*>(dev_ptr_v);
    T* O = static_cast<T*>(dev_ptr_o);
    return launch_fwd_dispatch<T>(batch_size, num_heads, hd, attn_scale, Q, K, V, O, softmax_lse,
                                    cu_q, cu_qp, cu_kv, cu_kvp, padded_q_to_batch, total_padded_q,
                                    hip_stream);
  }
  if(dtype == NVTEDType::kNVTEFloat16) {
    using T = __half;
    const T* Q = static_cast<const T*>(dev_ptr_q);
    const T* K = static_cast<const T*>(dev_ptr_k);
    const T* V = static_cast<const T*>(dev_ptr_v);
    T* O = static_cast<T*>(dev_ptr_o);
    return launch_fwd_dispatch<T>(batch_size, num_heads, hd, attn_scale, Q, K, V, O, softmax_lse,
                                    cu_q, cu_qp, cu_kv, cu_kvp, padded_q_to_batch, total_padded_q,
                                    hip_stream);
  }
  return false;
}

bool fused_attn_smallseq_bwd(size_t batch_size,
                             size_t num_heads,
                             size_t head_dim_qk,
                             size_t max_tokens_q,
                             size_t max_tokens_kv,
                             float attn_scale,
                             const void* dev_ptr_q,
                             const void* dev_ptr_k,
                             const void* dev_ptr_v,
                             const void* dev_ptr_do,
                             const void* dev_ptr_softmax_lse,
                             void* dev_ptr_dq,
                             void* dev_ptr_dk,
                             void* dev_ptr_dv,
                             const void* dev_ptr_cu_seqlens_q,
                             const void* dev_ptr_cu_seqlens_q_padded,
                             const void* dev_ptr_cu_seqlens_kv,
                             const void* dev_ptr_cu_seqlens_kv_padded,
                             NVTEDType dtype,
                             cudaStream_t stream) {
  (void)max_tokens_q;
  (void)max_tokens_kv;
  const int* cu_q = static_cast<const int*>(dev_ptr_cu_seqlens_q);
  const int* cu_qp = static_cast<const int*>(dev_ptr_cu_seqlens_q_padded);
  const int* cu_kv = static_cast<const int*>(dev_ptr_cu_seqlens_kv);
  const int* cu_kvp = static_cast<const int*>(dev_ptr_cu_seqlens_kv_padded);
  const float* softmax_lse = static_cast<const float*>(dev_ptr_softmax_lse);
  const int hd = static_cast<int>(head_dim_qk);
  const hipStream_t hip_stream = to_hip_stream(stream);

  if(!supports_hip_small_seq(num_heads, num_heads, head_dim_qk, head_dim_qk)) {
    return false;
  }

  if(dtype == NVTEDType::kNVTEBFloat16) {
    using T = hip_bfloat16;
    const T* Q = static_cast<const T*>(dev_ptr_q);
    const T* K = static_cast<const T*>(dev_ptr_k);
    const T* V = static_cast<const T*>(dev_ptr_v);
    const T* dO = static_cast<const T*>(dev_ptr_do);
    T* dQ = static_cast<T*>(dev_ptr_dq);
    T* dK = static_cast<T*>(dev_ptr_dk);
    T* dV = static_cast<T*>(dev_ptr_dv);
    return launch_bwd_dispatch<T>(batch_size, num_heads, hd, attn_scale, Q, K, V, dO, softmax_lse,
                                  dQ, dK, dV, cu_q, cu_qp, cu_kv, cu_kvp, hip_stream);
  }
  if(dtype == NVTEDType::kNVTEFloat16) {
    using T = __half;
    const T* Q = static_cast<const T*>(dev_ptr_q);
    const T* K = static_cast<const T*>(dev_ptr_k);
    const T* V = static_cast<const T*>(dev_ptr_v);
    const T* dO = static_cast<const T*>(dev_ptr_do);
    T* dQ = static_cast<T*>(dev_ptr_dq);
    T* dK = static_cast<T*>(dev_ptr_dk);
    T* dV = static_cast<T*>(dev_ptr_dv);
    return launch_bwd_dispatch<T>(batch_size, num_heads, hd, attn_scale, Q, K, V, dO, softmax_lse,
                                  dQ, dK, dV, cu_q, cu_qp, cu_kv, cu_kvp, hip_stream);
  }
  return false;
}

#endif  // USE_FUSED_ATTN_CK

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
