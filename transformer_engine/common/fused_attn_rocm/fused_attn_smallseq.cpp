/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "fused_attn_smallseq.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "utils.h"

#ifdef USE_FUSED_ATTN_CK
#include <ck_fused_attn/ck_fused_attn.hpp>
#include "attn_bwd_mfma_16x16.h"
#include "attn_fwd_mfma_dispatch.h"
#endif

// Dispatch a 16-bit floating-point NVTEDType to its concrete HIP element type `TYPE_NAME` and run
// the given statement block, so the bf16/fp16 paths share one body. Mirrors the
// CK_FUSED_ATTN_TYPE_SWITCH_16BIT pattern used in ck_fused_attn. If dtype is neither 16-bit float
// the block is not run.
#define SMALL_SEQ_TYPE_SWITCH_16BIT(DTYPE, TYPE_NAME, ...)         \
  do {                                                             \
    if ((DTYPE) == NVTEDType::kNVTEBFloat16) {                     \
      using TYPE_NAME = hip_bfloat16;                              \
      __VA_ARGS__                                                  \
    } else if ((DTYPE) == NVTEDType::kNVTEFloat16) {               \
      using TYPE_NAME = __half;                                    \
      __VA_ARGS__                                                  \
    }                                                              \
  } while (0)

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
  // fp16 is intentionally unsupported on the small-seq path for now: the MFMA kernels are
  // bf16-only (they load Q/K/V as bf16 and use bf16 MFMA intrinsics), so fp16 would be silently
  // downcast and produce wrong results. Reject fp16 here so it falls back to the regular CK path.
  if(q_dtype != NVTEDType::kNVTEBFloat16) return false;
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

// Extra device scratch the small-seq forward path reserves as a prefix of the CK fused-attn workspace.
// It holds, in order:
//   - 2 x uint64_t : output slots for get_runtime_max_seqlen() (runtime max seqlen of Q and of
//                    KV)
//   - max_tokens_q x int32_t : the padded_q_to_batch map (padded Q token index -> batch index)
//                    built by build_padded_q_to_batch_kernel() and consumed by the forward
//                    MFMA kernel.
size_t small_seq_fwd_extra_workspace_bytes(size_t max_tokens_q) {
  return 2 * sizeof(uint64_t) + max_tokens_q * sizeof(int32_t);
}

// Extra device scratch the small-seq backward path reserves as a prefix of the CK fused-attn
// workspace. Backward recomputes P from the LSE and takes no padded_q_to_batch map, so it only
// needs the 2 x uint64_t runtime-max-seqlen probe slots (Q and KV).
size_t small_seq_bwd_extra_workspace_bytes() {
  return 2 * sizeof(uint64_t);
}

bool is_nvte_ck_small_seq_enabled() {
  const int arch = transformer_engine::cuda::sm_arch();
  if (arch != 94 && arch != 95) {
    return false;
  }
  const char* env_p = std::getenv("NVTE_FUSED_ATTN_CK_SMALLSEQ");
  return env_p != nullptr && std::strcmp(env_p, "1") == 0;
}

#ifdef USE_FUSED_ATTN_CK

// HIP small-seq kernels: head dim 512 is not supported here — upstream instantiations exceed the
// 64 KiB LDS limit on gfx942 for the 17×17 small-seq tile configuration.
namespace {
using namespace small_seq_kernels;

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
  using Config =
      FmhaKernelConfig<HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnForwardMfmaDispatchLauncher<T, Config>;
  // attn_scale is already the full softmax scale (e.g. 1/sqrt(head_dim)) as passed by the
  // framework and used verbatim by the regular CK path; the small-seq kernel applies it
  // directly (matches the standalone tests and the reference impl). Do NOT divide by
  // sqrt(HEAD_DIM) again — that double-scales and flattens the softmax.
  const float sqr_dk_scale = attn_scale;
  // Launch on the caller's stream; correctness relies on stream ordering (the rest of the CK
  // pipeline runs on the same stream), so no host-side synchronize is needed here — only a
  // launch-error check, matching the pattern used elsewhere in fused_attn_ck.cpp.
  Launcher::run_attn_fwd_kernel(Q, K, V, nullptr, 0.0f, sqr_dk_scale, O, softmax_lse, cu_q, cu_qp,
                                cu_kv, cu_kvp, padded_q_to_batch, total_padded_q,
                                static_cast<int>(actual_batch), stream);
  NVTE_CHECK_CUDA(hipGetLastError());
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
  using Config =
      FmhaKernelConfig<HEAD_NUM, 17, HEAD_DIM, 256, false, CausalMaskType::DISABLE, 17>;
  using Launcher = AttnBackwardMfma16x16KernelLauncher<T, Config>;
  // See launch_fwd_inst: attn_scale is the full softmax scale already; pass it through unchanged.
  const float sqr_dk_scale = attn_scale;
  // Launch on the caller's stream; stream ordering guarantees the two backward kernels finish
  // before subsequent same-stream pipeline work reads their outputs, so a host synchronize is
  // unnecessary — a launch-error check suffices.
  Launcher::run_attn_bwd_kernel(Q, K, V, dO, softmax_lse, dQ, dK, dV, sqr_dk_scale, cu_q, cu_qp,
                                cu_kv, cu_kvp, static_cast<int>(actual_batch), stream);
  NVTE_CHECK_CUDA(hipGetLastError());
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

#endif  // USE_FUSED_ATTN_CK

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
#ifndef USE_FUSED_ATTN_CK
  return false;
#else
  const bool nvte_log_ck_config = getenv<bool>("NVTE_LOG_CK_CONFIG");
  if(nvte_log_ck_config) {
    std::cout << std::endl << "attn_fwd(ck small-seq): ";
    std::cout << "batch: " << batch_size << ", ";
    std::cout << "num_heads: " << num_heads << ", ";
    std::cout << "head_dim: " << head_dim_qk << ", ";
    std::cout << "max_tokens_q: " << max_tokens_q << ", ";
    std::cout << "max_tokens_kv: " << max_tokens_kv << ", ";
    std::cout << "attn_scale: " << attn_scale << ", ";
    std::cout << "dtype: " << static_cast<int>(dtype) << std::endl;
  }

  const int* cu_q = static_cast<const int*>(dev_ptr_cu_seqlens_q);
  const int* cu_qp = static_cast<const int*>(dev_ptr_cu_seqlens_q_padded);
  const int* cu_kv = static_cast<const int*>(dev_ptr_cu_seqlens_kv);
  const int* cu_kvp = static_cast<const int*>(dev_ptr_cu_seqlens_kv_padded);
  const int* padded_q_to_batch = static_cast<const int*>(dev_ptr_padded_q_to_batch);
  float* softmax_lse = static_cast<float*>(dev_ptr_softmax_lse);
  const int total_padded_q = static_cast<int>(max_tokens_q);
  const int hd = static_cast<int>(head_dim_qk);

  bool ran = false;
  SMALL_SEQ_TYPE_SWITCH_16BIT(dtype, T, {
    const T* Q = static_cast<const T*>(dev_ptr_q);
    const T* K = static_cast<const T*>(dev_ptr_k);
    const T* V = static_cast<const T*>(dev_ptr_v);
    T* O = static_cast<T*>(dev_ptr_o);
    ran = launch_fwd_dispatch<T>(batch_size, num_heads, hd, attn_scale, Q, K, V, O, softmax_lse,
                                 cu_q, cu_qp, cu_kv, cu_kvp, padded_q_to_batch, total_padded_q,
                                 stream);
  });

  if(nvte_log_ck_config && !ran) {
    std::cout << "attn_fwd(ck small-seq): kernel not launched for this config; "
              << "falling back to regular ck/aiter" << std::endl;
  }
  return ran;
#endif  // USE_FUSED_ATTN_CK
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
#ifndef USE_FUSED_ATTN_CK
  return false;
#else
  const bool nvte_log_ck_config = getenv<bool>("NVTE_LOG_CK_CONFIG");
  if(nvte_log_ck_config) {
    std::cout << std::endl << "attn_bwd(ck small-seq): ";
    std::cout << "batch: " << batch_size << ", ";
    std::cout << "num_heads: " << num_heads << ", ";
    std::cout << "head_dim: " << head_dim_qk << ", ";
    std::cout << "max_tokens_q: " << max_tokens_q << ", ";
    std::cout << "max_tokens_kv: " << max_tokens_kv << ", ";
    std::cout << "attn_scale: " << attn_scale << ", ";
    std::cout << "dtype: " << static_cast<int>(dtype) << std::endl;
  }

  const int* cu_q = static_cast<const int*>(dev_ptr_cu_seqlens_q);
  const int* cu_qp = static_cast<const int*>(dev_ptr_cu_seqlens_q_padded);
  const int* cu_kv = static_cast<const int*>(dev_ptr_cu_seqlens_kv);
  const int* cu_kvp = static_cast<const int*>(dev_ptr_cu_seqlens_kv_padded);
  const float* softmax_lse = static_cast<const float*>(dev_ptr_softmax_lse);
  const int hd = static_cast<int>(head_dim_qk);

  bool ran = false;
  SMALL_SEQ_TYPE_SWITCH_16BIT(dtype, T, {
    const T* Q = static_cast<const T*>(dev_ptr_q);
    const T* K = static_cast<const T*>(dev_ptr_k);
    const T* V = static_cast<const T*>(dev_ptr_v);
    const T* dO = static_cast<const T*>(dev_ptr_do);
    T* dQ = static_cast<T*>(dev_ptr_dq);
    T* dK = static_cast<T*>(dev_ptr_dk);
    T* dV = static_cast<T*>(dev_ptr_dv);
    ran = launch_bwd_dispatch<T>(batch_size, num_heads, hd, attn_scale, Q, K, V, dO, softmax_lse,
                                 dQ, dK, dV, cu_q, cu_qp, cu_kv, cu_kvp, stream);
  });

  if(nvte_log_ck_config && !ran) {
    std::cout << "attn_bwd(ck small-seq): kernel not launched for this config; "
              << "falling back to regular ck/aiter" << std::endl;
  }
  return ran;
#endif  // USE_FUSED_ATTN_CK
}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
