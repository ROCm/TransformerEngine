/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "transformer_engine/fused_attn.h"
#include "../common.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn_rocm {

using namespace transformer_engine;

size_t nvte_dtype_size(DType t_dtype){
  switch(t_dtype){
    case DType::kByte: 
      return 1;
    case DType::kInt32: 
      return 4;
    case DType::kInt64: 
      return 8;
    case DType::kFloat32: 
      return 4;
    case DType::kFloat16: 
      return 2;
    case DType::kBFloat16: 
      return 2;
    case DType::kFloat8E4M3: 
    case DType::kFloat8E5M2: 
      return 1;
    default:
      return 1;
  }
  return 1;
}

// get matrix strides based on matrix type
void generateMatrixStrides(
            uint64_t b, uint64_t h,
            uint64_t s_q, uint64_t s_kv,
            uint64_t d, uint64_t* stride,
            NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix) {
    // CK/AOTriton internally takes BHSD for implementation
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    switch (layout) {
        case NVTE_QKV_Layout::NVTE_SB3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 3 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 3 * h * d;
                    stride[head_dim_idx] = 3 * d;
                    stride[seqlen_dim_idx] = b * 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 2 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * 2 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = 2 * h * d;
                    stride[head_dim_idx] = 2 * d;
                    stride[seqlen_dim_idx] = b * 2 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BS3HD:
        case NVTE_QKV_Layout::NVTE_T3HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = s_q * 3 * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = 3 * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                    stride[batch_dim_idx] = s_q * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_BSH3D:
        case NVTE_QKV_Layout::NVTE_TH3D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_q * 3 * h * d;
                     stride[head_dim_idx] = 3 * d;
                     stride[seqlen_dim_idx] = 3 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
        case NVTE_QKV_Layout::NVTE_THD_T2HD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_kv * 2 * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = 2 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
        case NVTE_QKV_Layout::NVTE_THD_TH2D:
            if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                     stride[batch_dim_idx] = s_kv * 2 * h * d;
                     stride[head_dim_idx] = 2 * d;
                     stride[seqlen_dim_idx] = 2 * h * d;
                     stride[hidden_dim_idx] = 1;
             } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                 || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                     stride[batch_dim_idx] = s_q * h * d;
                     stride[head_dim_idx] = d;
                     stride[seqlen_dim_idx] = h * d;
                     stride[hidden_dim_idx] = 1;
             }
             break;
        case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
        case NVTE_QKV_Layout::NVTE_THD_THD_THD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = s_q * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = s_kv * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        case NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD:
            if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
                    stride[batch_dim_idx] = h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = b * h * d;
                    stride[hidden_dim_idx] = 1;
            } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix)
                || (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
                    stride[batch_dim_idx] = s_kv * h * d;
                    stride[head_dim_idx] = d;
                    stride[seqlen_dim_idx] = h * d;
                    stride[hidden_dim_idx] = 1;
            }
            break;
        default:
            NVTE_ERROR("Unsupported layout: ", static_cast<int>(layout));
            break;
    }
}

__global__ void populate_rng_state_kernel(int64_t *rng_state_dst, const int64_t *const seed,
                                          int64_t offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > 0) return;
  rng_state_dst[0] = seed[0];
  rng_state_dst[1] = offset;
}

__global__ void get_runtime_num_segments_kernel(int32_t *cu_seqlen, size_t max_batch_size, uint32_t *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= max_batch_size) return;

  if (cu_seqlen[tid+1] - cu_seqlen[tid] > 0) {
    // atomicAdd only support 32 bits dtype
    atomicAdd(out, 1);
  }
}

void PopulateRngStateAsync(void *rng_state_dst, const void *seed, size_t q_max_seqlen,
                           size_t kv_max_seqlen, NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream) {
  //both aiter and aotriton now follows flash-attn rng design
  size_t increment = 16;
  auto offset = FusedAttnOffsetManager::Instance().GetAndUpdateOffset(increment);
  populate_rng_state_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<int64_t *>(rng_state_dst),
                                                 reinterpret_cast<const int64_t *>(seed), offset);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

uint32_t GetRuntimeNumSegments(void *cu_seqlen, void *workspace, size_t max_batch_size, cudaStream_t stream) {
  // workspace size requires 4 bytes
  uint32_t *dout = static_cast<uint32_t *>(workspace);
  uint32_t hout{};
  cudaMemsetAsync(dout, 0, sizeof(uint32_t), stream);
  constexpr int threads = 128;
  const int blocks = (max_batch_size - 1) / threads + 1; // ceil
  get_runtime_num_segments_kernel<<<blocks, threads, 0, stream>>>(static_cast<int32_t *>(cu_seqlen),
                                                                  max_batch_size, dout);
  cudaMemcpyAsync(&hout, dout, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return hout;
}

__global__ void extract_seed_and_offset(int64_t *rng_state_ptr, bool captured, int64_t *seed_ptr,
                                        uint64_t seed_val, int64_t *offset_ptr, uint64_t offset_val,
                                        uint32_t offset_intragraph) {
  if (captured) {
    rng_state_ptr[0] = *seed_ptr;
    rng_state_ptr[1] = static_cast<int64_t>(*offset_ptr + static_cast<int64_t>(offset_intragraph));
  } else {
    rng_state_ptr[0] = static_cast<int64_t>(seed_val);
    rng_state_ptr[1] = static_cast<int64_t>(offset_val);
  }
}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine

void nvte_extract_seed_and_offset(int64_t *rng_state_ptr, int captured, int64_t *seed_ptr,
                                  uint64_t seed_val, int64_t *offset_ptr, uint64_t offset_val,
                                  uint32_t offset_intragraph, cudaStream_t stream) {
  NVTE_API_CALL(nvte_extract_seed_and_offset);
  transformer_engine::fused_attn_rocm::extract_seed_and_offset<<<1, 1, 0, stream>>>(
        rng_state_ptr, captured, seed_ptr, seed_val, offset_ptr, offset_val, offset_intragraph);
}
