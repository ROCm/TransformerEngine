/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_fused_attn/ck_fused_attn.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_bwd_mfma_16x16.h"
#include "attn_fwd_mfma_dispatch.h"

#define CK_SMALL_SEQ_TYPE_SWITCH_16BIT(DTYPE, TYPE_NAME, ...) \
  do {                                                        \
    if((DTYPE) == DType::kBFloat16) {                         \
      using TYPE_NAME = hip_bfloat16;                         \
      __VA_ARGS__                                             \
    } else if((DTYPE) == DType::kFloat16) {                   \
      using TYPE_NAME = __half;                               \
      __VA_ARGS__                                             \
    }                                                         \
  } while(0)

namespace ck_fused_attn {

namespace {

using namespace small_seq_kernels;

template <typename T, int HEAD_NUM, int HEAD_DIM>
void launch_fwd_thd_inst(size_t batch,
                         int total_padded_q,
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
                         hipStream_t stream) {
  using Config =
      FmhaKernelConfig<HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnForwardMfmaDispatchLauncher<T, Config>;
  Launcher::run_attn_fwd_kernel(Q, K, V, nullptr, 0.0f, attn_scale, O, softmax_lse,
                                0, cu_q, cu_qp, cu_kv, cu_kvp, total_padded_q,
                                static_cast<int>(batch), stream);
  HIP_CHECK(hipGetLastError());
}

template <typename T, int HEAD_NUM, int HEAD_DIM>
void launch_fwd_bshd_inst(size_t batch,
                          int seqlen_q,
                          int seqlen_kv,
                          float attn_scale,
                          const T* Q,
                          const T* K,
                          const T* V,
                          T* O,
                          float* softmax_lse,
                          hipStream_t stream) {
  using Config =
      FmhaKernelConfig<HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnForwardMfmaDispatchLauncher<T, Config>;
  const int total_tokens_q = static_cast<int>(batch) * seqlen_q;
  Launcher::run_attn_fwd_kernel(Q, K, V, nullptr, 0.0f, attn_scale, O, softmax_lse, seqlen_q,
                                nullptr, nullptr, nullptr, nullptr, total_tokens_q,
                                static_cast<int>(batch), stream);
  (void)seqlen_kv;
  HIP_CHECK(hipGetLastError());
}

template <typename T>
void launch_fwd_thd_dispatch(size_t batch,
                             size_t num_heads,
                             int head_dim,
                             int total_padded_q,
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
                             hipStream_t stream) {
  if(num_heads == 16) {
    if(head_dim == 128) {
      launch_fwd_thd_inst<T, 16, 128>(batch, total_padded_q, attn_scale, Q, K, V, O, softmax_lse,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
    if(head_dim == 256) {
      launch_fwd_thd_inst<T, 16, 256>(batch, total_padded_q, attn_scale, Q, K, V, O, softmax_lse,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
  }
  if(num_heads == 32) {
    if(head_dim == 128) {
      launch_fwd_thd_inst<T, 32, 128>(batch, total_padded_q, attn_scale, Q, K, V, O, softmax_lse,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
    if(head_dim == 256) {
      launch_fwd_thd_inst<T, 32, 256>(batch, total_padded_q, attn_scale, Q, K, V, O, softmax_lse,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
  }
}

template <typename T>
void launch_fwd_bshd_dispatch(size_t batch,
                              size_t num_heads,
                              int head_dim,
                              int seqlen_q,
                              int seqlen_kv,
                              float attn_scale,
                              const T* Q,
                              const T* K,
                              const T* V,
                              T* O,
                              float* softmax_lse,
                              hipStream_t stream) {
  if(num_heads == 16) {
    if(head_dim == 128) {
      launch_fwd_bshd_inst<T, 16, 128>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, O,
                                       softmax_lse, stream);
      return;
    }
    if(head_dim == 256) {
      launch_fwd_bshd_inst<T, 16, 256>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, O,
                                       softmax_lse, stream);
      return;
    }
  }
  if(num_heads == 32) {
    if(head_dim == 128) {
      launch_fwd_bshd_inst<T, 32, 128>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, O,
                                       softmax_lse, stream);
      return;
    }
    if(head_dim == 256) {
      launch_fwd_bshd_inst<T, 32, 256>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, O,
                                       softmax_lse, stream);
      return;
    }
  }
}

template <typename T, int HEAD_NUM, int HEAD_DIM>
void launch_bwd_thd_inst(size_t batch,
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
  using Config =
      FmhaKernelConfig<HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnBackwardMfma16x16KernelLauncher<T, Config>;
  Launcher::run_attn_bwd_kernel(Q, K, V, dO, softmax_lse, dQ, dK, dV, attn_scale, 0, cu_q, cu_qp,
                                cu_kv, cu_kvp, static_cast<int>(batch), stream);
  HIP_CHECK(hipGetLastError());
}

template <typename T, int HEAD_NUM, int HEAD_DIM>
void launch_bwd_bshd_inst(size_t batch,
                          int seqlen_q,
                          int seqlen_kv,
                          float attn_scale,
                          const T* Q,
                          const T* K,
                          const T* V,
                          const T* dO,
                          const float* softmax_lse,
                          T* dQ,
                          T* dK,
                          T* dV,
                          hipStream_t stream) {
  using Config =
      FmhaKernelConfig<HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnBackwardMfma16x16KernelLauncher<T, Config>;
  Launcher::run_attn_bwd_kernel(Q, K, V, dO, softmax_lse, dQ, dK, dV, attn_scale, seqlen_q,
                                nullptr, nullptr, nullptr, nullptr, static_cast<int>(batch),
                                stream);
  (void)seqlen_kv;
  HIP_CHECK(hipGetLastError());
}

template <typename T>
void launch_bwd_thd_dispatch(size_t batch,
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
      launch_bwd_thd_inst<T, 16, 128>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
    if(head_dim == 256) {
      launch_bwd_thd_inst<T, 16, 256>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
  }
  if(num_heads == 32) {
    if(head_dim == 128) {
      launch_bwd_thd_inst<T, 32, 128>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
    if(head_dim == 256) {
      launch_bwd_thd_inst<T, 32, 256>(batch, attn_scale, Q, K, V, dO, softmax_lse, dQ, dK, dV,
                                      cu_q, cu_qp, cu_kv, cu_kvp, stream);
      return;
    }
  }
}

template <typename T>
void launch_bwd_bshd_dispatch(size_t batch,
                              size_t num_heads,
                              int head_dim,
                              int seqlen_q,
                              int seqlen_kv,
                              float attn_scale,
                              const T* Q,
                              const T* K,
                              const T* V,
                              const T* dO,
                              const float* softmax_lse,
                              T* dQ,
                              T* dK,
                              T* dV,
                              hipStream_t stream) {
  if(num_heads == 16) {
    if(head_dim == 128) {
      launch_bwd_bshd_inst<T, 16, 128>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, dO,
                                       softmax_lse, dQ, dK, dV, stream);
      return;
    }
    if(head_dim == 256) {
      launch_bwd_bshd_inst<T, 16, 256>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, dO,
                                       softmax_lse, dQ, dK, dV, stream);
      return;
    }
  }
  if(num_heads == 32) {
    if(head_dim == 128) {
      launch_bwd_bshd_inst<T, 32, 128>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, dO,
                                       softmax_lse, dQ, dK, dV, stream);
      return;
    }
    if(head_dim == 256) {
      launch_bwd_bshd_inst<T, 32, 256>(batch, seqlen_q, seqlen_kv, attn_scale, Q, K, V, dO,
                                       softmax_lse, dQ, dK, dV, stream);
      return;
    }
  }
}

}  // namespace

size_t small_seq_thd_extra_workspace_bytes() {
  // [max_seqlen_q probe][max_seqlen_kv probe] for THD runtime eligibility checks.
  return 2 * sizeof(uint64_t);
}

void ck_attn_smallseq_fwd_thd(size_t batch_size,
                              size_t num_heads,
                              size_t head_dim_qk,
                              size_t max_tokens_q,
                              size_t max_tokens_kv,
                              float attn_scale,
                              const void* q_ptr,
                              const void* k_ptr,
                              const void* v_ptr,
                              void* o_ptr,
                              void* softmax_lse_ptr,
                              const void* cu_seqlens_q_ptr,
                              const void* cu_seqlens_q_padded_ptr,
                              const void* cu_seqlens_kv_ptr,
                              const void* cu_seqlens_kv_padded_ptr,
                              DType dtype,
                              hipStream_t stream) {
  const int* cu_q = static_cast<const int*>(cu_seqlens_q_ptr);
  const int* cu_qp = static_cast<const int*>(cu_seqlens_q_padded_ptr);
  const int* cu_kv = static_cast<const int*>(cu_seqlens_kv_ptr);
  const int* cu_kvp = static_cast<const int*>(cu_seqlens_kv_padded_ptr);
  float* softmax_lse = static_cast<float*>(softmax_lse_ptr);
  const int total_padded_q = static_cast<int>(max_tokens_q);
  const int hd = static_cast<int>(head_dim_qk);

  CK_SMALL_SEQ_TYPE_SWITCH_16BIT(dtype, T, {
    const T* Q = static_cast<const T*>(q_ptr);
    const T* K = static_cast<const T*>(k_ptr);
    const T* V = static_cast<const T*>(v_ptr);
    T* O = static_cast<T*>(o_ptr);
    launch_fwd_thd_dispatch<T>(batch_size, num_heads, hd, total_padded_q, attn_scale, Q, K, V, O,
                               softmax_lse, cu_q, cu_qp, cu_kv, cu_kvp, stream);
  });
}

void ck_attn_smallseq_fwd_bshd(size_t batch_size,
                               size_t num_heads,
                               size_t seqlen_q,
                               size_t seqlen_kv,
                               size_t head_dim_qk,
                               float attn_scale,
                               const void* q_ptr,
                               const void* k_ptr,
                               const void* v_ptr,
                               void* o_ptr,
                               void* softmax_lse_ptr,
                               DType dtype,
                               hipStream_t stream) {
  float* softmax_lse = static_cast<float*>(softmax_lse_ptr);
  const int hd = static_cast<int>(head_dim_qk);
  const int sq = static_cast<int>(seqlen_q);
  const int skv = static_cast<int>(seqlen_kv);

  CK_SMALL_SEQ_TYPE_SWITCH_16BIT(dtype, T, {
    const T* Q = static_cast<const T*>(q_ptr);
    const T* K = static_cast<const T*>(k_ptr);
    const T* V = static_cast<const T*>(v_ptr);
    T* O = static_cast<T*>(o_ptr);
    launch_fwd_bshd_dispatch<T>(batch_size, num_heads, hd, sq, skv, attn_scale, Q, K, V, O,
                                softmax_lse, stream);
  });
}

void ck_attn_smallseq_bwd_thd(size_t batch_size,
                              size_t num_heads,
                              size_t head_dim_qk,
                              size_t max_tokens_q,
                              size_t max_tokens_kv,
                              float attn_scale,
                              const void* q_ptr,
                              const void* k_ptr,
                              const void* v_ptr,
                              const void* do_ptr,
                              const void* softmax_lse_ptr,
                              void* dq_ptr,
                              void* dk_ptr,
                              void* dv_ptr,
                              const void* cu_seqlens_q_ptr,
                              const void* cu_seqlens_q_padded_ptr,
                              const void* cu_seqlens_kv_ptr,
                              const void* cu_seqlens_kv_padded_ptr,
                              DType dtype,
                              hipStream_t stream) {
  const int* cu_q = static_cast<const int*>(cu_seqlens_q_ptr);
  const int* cu_qp = static_cast<const int*>(cu_seqlens_q_padded_ptr);
  const int* cu_kv = static_cast<const int*>(cu_seqlens_kv_ptr);
  const int* cu_kvp = static_cast<const int*>(cu_seqlens_kv_padded_ptr);
  const float* softmax_lse = static_cast<const float*>(softmax_lse_ptr);
  const int hd = static_cast<int>(head_dim_qk);

  CK_SMALL_SEQ_TYPE_SWITCH_16BIT(dtype, T, {
    const T* Q = static_cast<const T*>(q_ptr);
    const T* K = static_cast<const T*>(k_ptr);
    const T* V = static_cast<const T*>(v_ptr);
    const T* dO = static_cast<const T*>(do_ptr);
    T* dQ = static_cast<T*>(dq_ptr);
    T* dK = static_cast<T*>(dk_ptr);
    T* dV = static_cast<T*>(dv_ptr);
    launch_bwd_thd_dispatch<T>(batch_size, num_heads, hd, attn_scale, Q, K, V, dO, softmax_lse, dQ,
                               dK, dV, cu_q, cu_qp, cu_kv, cu_kvp, stream);
  });
}

void ck_attn_smallseq_bwd_bshd(size_t batch_size,
                               size_t num_heads,
                               size_t seqlen_q,
                               size_t seqlen_kv,
                               size_t head_dim_qk,
                               float attn_scale,
                               const void* q_ptr,
                               const void* k_ptr,
                               const void* v_ptr,
                               const void* do_ptr,
                               const void* softmax_lse_ptr,
                               void* dq_ptr,
                               void* dk_ptr,
                               void* dv_ptr,
                               DType dtype,
                               hipStream_t stream) {
  const float* softmax_lse = static_cast<const float*>(softmax_lse_ptr);
  const int hd = static_cast<int>(head_dim_qk);
  const int sq = static_cast<int>(seqlen_q);
  const int skv = static_cast<int>(seqlen_kv);

  CK_SMALL_SEQ_TYPE_SWITCH_16BIT(dtype, T, {
    const T* Q = static_cast<const T*>(q_ptr);
    const T* K = static_cast<const T*>(k_ptr);
    const T* V = static_cast<const T*>(v_ptr);
    const T* dO = static_cast<const T*>(do_ptr);
    T* dQ = static_cast<T*>(dq_ptr);
    T* dK = static_cast<T*>(dk_ptr);
    T* dV = static_cast<T*>(dv_ptr);
    launch_bwd_bshd_dispatch<T>(batch_size, num_heads, hd, sq, skv, attn_scale, Q, K, V, dO,
                                softmax_lse, dQ, dK, dV, stream);
  });
}

}  // namespace ck_fused_attn
