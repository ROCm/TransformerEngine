/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "fused_attn_unfused_smallseq.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "utils.h"
#include "../common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <utility>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace transformer_engine {
namespace fused_attn_rocm {

// Kernel to compute max sequence length for Q
__global__ void get_runtime_max_seqlen_q_kernel(
  uint64_t b,
  const int32_t* cu_seqlen_ptr, 
  const int32_t* cu_seqlen_padded_ptr, 
  uint64_t *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid >= b){
    return;
  }
  if(cu_seqlen_padded_ptr){
    atomicMax(out, cu_seqlen_padded_ptr[tid+1] - cu_seqlen_padded_ptr[tid]);
  }else{
    atomicMax(out, cu_seqlen_ptr[tid+1] - cu_seqlen_ptr[tid]);
  }
}

// Kernel to compute max sequence length for KV
__global__ void get_runtime_max_seqlen_kv_kernel(
  uint64_t b,
  const int32_t* cu_seqlen_ptr, 
  const int32_t* cu_seqlen_padded_ptr, 
  uint64_t *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid >= b){
    return;
  }
  if(cu_seqlen_padded_ptr){
    atomicMax(out, cu_seqlen_padded_ptr[tid+1] - cu_seqlen_padded_ptr[tid]);
  }else{
    atomicMax(out, cu_seqlen_ptr[tid+1] - cu_seqlen_ptr[tid]);
  }
}

// Get runtime maximum sequence lengths for Q and KV
std::pair<uint64_t, uint64_t> get_runtime_max_seqlen_q_kv(
  uint64_t batch_size,
  const void* cu_seqlens_q_ptr,
  const void* cu_seqlens_kv_ptr,
  const void* cu_seqlens_q_padded_ptr,
  const void* cu_seqlens_kv_padded_ptr,
  void* workspace,
  cudaStream_t stream) {
  
  uint64_t runtime_max_seqlen_q = 0;
  uint64_t runtime_max_seqlen_kv = 0;
  
  // Handle edge case: batch_size == 0
  if(batch_size == 0){
    return std::make_pair(runtime_max_seqlen_q, runtime_max_seqlen_kv);
  }
  
  // workspace should have space for 2 uint64_t values
  uint64_t* runtime_max_seqlen_q_ptr = static_cast<uint64_t*>(workspace);
  uint64_t* runtime_max_seqlen_kv_ptr = runtime_max_seqlen_q_ptr + 1;
  
  // Reset result buffers to 0
  hipMemsetAsync(runtime_max_seqlen_q_ptr, 0, sizeof(uint64_t), stream);
  hipMemsetAsync(runtime_max_seqlen_kv_ptr, 0, sizeof(uint64_t), stream);
  
  constexpr int threads = 128;
  const int blocks = (static_cast<int64_t>(batch_size) - 1) / threads + 1; // ceil
  
  // Launch kernels to compute max sequence lengths
  if(cu_seqlens_q_ptr) {
    get_runtime_max_seqlen_q_kernel<<<blocks, threads, 0, stream>>>(
      batch_size, 
      static_cast<const int32_t*>(cu_seqlens_q_ptr),
      static_cast<const int32_t*>(cu_seqlens_q_padded_ptr),
      runtime_max_seqlen_q_ptr);
  }
  
  if(cu_seqlens_kv_ptr) {
    get_runtime_max_seqlen_kv_kernel<<<blocks, threads, 0, stream>>>(
      batch_size, 
      static_cast<const int32_t*>(cu_seqlens_kv_ptr),
      static_cast<const int32_t*>(cu_seqlens_kv_padded_ptr),
      runtime_max_seqlen_kv_ptr);
  }
  
  // Synchronize and copy results back
  hipMemcpyAsync(&runtime_max_seqlen_q, runtime_max_seqlen_q_ptr, sizeof(uint64_t), hipMemcpyDeviceToHost, stream);
  hipMemcpyAsync(&runtime_max_seqlen_kv, runtime_max_seqlen_kv_ptr, sizeof(uint64_t), hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
  
  return std::make_pair(runtime_max_seqlen_q, runtime_max_seqlen_kv);
}

// Calculate workspace size for forward pass
// Workspace stores attention weights: [batch, head_num, seq_q, max_seq_kv]
void fused_attn_unfused_smallseq_fwd_workspace_size(
  size_t b, size_t h_q, size_t max_seqlen_q, size_t max_seqlen_kv,
  DType dtype,
  void* workspace,
  size_t* workspace_size) {
  
  // Workspace size = batch * head_num * seq_q * max_seq_kv * sizeof(dtype)
  size_t attn_weights_size = b * h_q * max_seqlen_q * max_seqlen_kv * nvte_dtype_size(dtype);
  
  if(workspace == nullptr) {
    // Request workspace size from upper level API
    *workspace_size = attn_weights_size;
    return;
  }
  
  // Workspace is provided, verify it's large enough
  // (This check is done by the caller, but we can add validation here if needed)
}

// Calculate workspace size for backward pass
// Backward uses same workspace size as forward (for temporary storage)
void fused_attn_unfused_smallseq_bwd_workspace_size(
  size_t b, size_t h_q, size_t max_seqlen_q, size_t max_seqlen_kv,
  DType dtype,
  void* workspace,
  size_t* workspace_size) {
  
  // Backward pass needs same workspace size as forward
  // Workspace is used for temporary storage during backward computation
  size_t workspace_bytes = b * h_q * max_seqlen_q * max_seqlen_kv * nvte_dtype_size(dtype);
  
  if(workspace == nullptr) {
    // Request workspace size from upper level API
    *workspace_size = workspace_bytes;
    return;
  }
  
  // Workspace is provided, verify it's large enough
  // (This check is done by the caller, but we can add validation here if needed)
}

// Check the fused attn config to see whether it's unfused_smallseq backend supported
bool is_unfused_smallseq_backend_supported(
  NVTEDType q_dtype,
  NVTEDType kv_dtype,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type attn_mask_type,
  float dropout,
  size_t num_attn_heads, size_t num_gqa_groups,
  size_t max_seqlen_q, size_t max_seqlen_kv,
  size_t head_dim_qk, 
  size_t head_dim_v, 
  int64_t window_size_left, 
  int64_t window_size_right) {

  // Debug info setting (optional, for troubleshooting)
  bool nvte_log_unfused_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_UNFUSED_CONFIG")) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_unfused_config = true;
  }

  // Filter 1: Check data type - must be FP16 or BF16, and Q and KV must match
  if((q_dtype != kv_dtype) || 
     !((q_dtype == NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16))){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: Q, K, V data type must be FP16 or BF16 and match" << std::endl;
    }
    return false;
  }

  // Filter 2: Check layout - must be THD (ragged format)
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if(qkv_format != NVTE_QKV_Format::NVTE_THD){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: Only THD (ragged) layout is supported" << std::endl;
    }
    return false;
  }

  // Filter 3: Check sequence length constraints
  // seq_q must be exactly 1, seq_kv must be <= 16
  if(max_seqlen_q != 1){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: max_seqlen_q must be 1, got " << max_seqlen_q << std::endl;
    }
    return false;
  }
  
  if(max_seqlen_kv > 16){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: max_seqlen_kv must be <= 16, got " << max_seqlen_kv << std::endl;
    }
    return false;
  }

  // Filter 4: Check bias type - no bias supported (based on HIP kernel implementation)
  if(bias_type != NVTE_Bias_Type::NVTE_NO_BIAS){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: Only NO_BIAS is supported" << std::endl;
    }
    return false;
  }

  // Filter 5: Check mask types - support NO_MASK, PADDING_MASK, and causal masks
  // Based on the HIP kernel code, it supports DISABLE, TOP_LEFT, BOTTOM_RIGHT causal masks
  bool is_supported_mask = 
    (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK) ||
    (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
    (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
    (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) ||
    (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK) ||
    (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  
  if(!is_supported_mask){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: Unsupported mask type" << std::endl;
    }
    return false;
  }

  // Filter 6: Check head dimensions - reasonable limits
  // The HIP kernels don't have explicit limits, but we should check for reasonable values
  if(head_dim_qk == 0 || head_dim_v == 0 || head_dim_qk > 512 || head_dim_v > 512){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: Invalid head dimensions" << std::endl;
    }
    return false;
  }

  // Filter 7: Check num_heads constraints
  if(num_attn_heads == 0 || num_gqa_groups == 0 || num_attn_heads % num_gqa_groups != 0){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: Invalid num_heads or num_gqa_groups" << std::endl;
    }
    return false;
  }

  // Filter 8: Check sliding window - not supported by this backend
  // The HIP kernels don't implement sliding window attention
  if(window_size_left != -1 || window_size_right != -1){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: Sliding window attention not supported" << std::endl;
    }
    return false;
  }

  // Filter 9: Check layout group - must be separate Q, K, V (not packed)
  // The HIP kernels expect separate Q, K, V tensors
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  bool is_qkvpacked = (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD || 
                       layout_group == NVTE_QKV_Layout_Group::NVTE_H3D);
  bool is_kvpacked = (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD || 
                      layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D);
  
  // We support separate Q, K, V (THD_THD_THD) and KV packed (THD_T2HD, THD_TH2D)
  // But not QKV packed
  if(is_qkvpacked){
    if(nvte_log_unfused_config){
      std::cout << "Unfused smallseq backend: QKV packed layout not supported" << std::endl;
    }
    return false;
  }

  // All checks passed
  return true;
}

// ============================================================================
// HIP Kernel Implementations (Runtime Versions)
// ============================================================================

// Forward Kernel 1: Compute scores = Q @ K^T * scale
// Runtime version adapted from template version
template <typename T>
__global__ void compute_scores_kernel_runtime(
    const T* Q,
    const T* K,
    T* scores,
    float scale,
    const int* cu_seqlens_kv,
    const int* cu_seqlens_kv_padded,
    int batch_size,
    int head_num,
    int seq_q,
    int max_seq_kv,
    int head_dim) {
    constexpr int block_k = 64;
    constexpr int thread_block_size = 64;
    constexpr int tasks_per_block = 16;
    
    int base_block_offset = blockIdx.x * thread_block_size * tasks_per_block;
    int thread_id = threadIdx.x;
    
    for(int task = 0; task < tasks_per_block; task++) {
        int cur_batch_idx = base_block_offset + task * thread_block_size + thread_id;
        int batch_idx = cur_batch_idx / (seq_q * head_num);
        int seq_head_idx = cur_batch_idx % (seq_q * head_num);
        int seq_idx = seq_head_idx / head_num;
        int head_idx = seq_head_idx % head_num;
        
        if(batch_idx >= batch_size)
            continue;
        
        int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int kv_offset = cu_seqlens_kv_padded[batch_idx];
        
        float results[16]; // max_seq_kv <= 16
        T fetch_Q[block_k];
        T fetch_K[block_k];
        
        // For THD layout: Q is [max_tokens_q, head_num, head_dim] where max_tokens_q = b * seq_q
        // For seq_q=1: max_tokens_q = b, so batch_idx directly maps to token index
        // Q access: Q[(batch_idx * head_num + head_idx) * head_dim] for THD with seq_q=1
        // But original varlen code uses: Q[(batch_idx * seq_q * head_num + seq_idx * head_num + head_idx) * head_dim]
        // For seq_q=1, seq_idx=0: this becomes Q[(batch_idx * head_num + head_idx) * head_dim] - same!
        T* Q_ptr = (T*)&Q[(batch_idx * seq_q * head_num + seq_idx * head_num + head_idx) * head_dim];
        // K is [max_tokens_kv, head_num, head_dim] - kv_offset is the token offset for this batch
        T* K_ptr = (T*)&K[(kv_offset * head_num + head_idx) * head_dim];
        T* score_ptr = (T*)&scores[cur_batch_idx * max_seq_kv];
        
        uint4 ls_dwordx4_tmp_var;
        for(int i = 0; i < seq_kv; i++)
            results[i] = 0.0f;
        
        for(int dim_offset = 0; dim_offset < head_dim; dim_offset += block_k) {
            // Load Q - handle BF16 (8 per uint4) vs FP16 (4 per uint4)
            if constexpr(std::is_same<T, hip_bfloat16>::value) {
                for(int k = 0; k < block_k / 8; k++) {
                    ls_dwordx4_tmp_var = *((uint4*)&Q_ptr[dim_offset + k * 8]);
                    fetch_Q[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                    fetch_Q[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                    fetch_Q[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                    fetch_Q[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                    fetch_Q[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                    fetch_Q[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                    fetch_Q[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                    fetch_Q[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                }
            } else {
                for(int k = 0; k < block_k / 4; k++) {
                    ls_dwordx4_tmp_var = *((uint4*)&Q_ptr[dim_offset + k * 4]);
                    fetch_Q[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                    fetch_Q[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                    fetch_Q[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                    fetch_Q[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                }
            }
            
            // Compute Q @ K^T for each kv position
            for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++) {
                if constexpr(std::is_same<T, hip_bfloat16>::value) {
                    for(int k = 0; k < block_k / 8; k++) {
                        ls_dwordx4_tmp_var = *((uint4*)&K_ptr[kv_idx * head_num * head_dim + dim_offset + k * 8]);
                        fetch_K[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                        fetch_K[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                        fetch_K[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                        fetch_K[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                        fetch_K[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                        fetch_K[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                        fetch_K[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                        fetch_K[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                    }
                } else {
                    for(int k = 0; k < block_k / 4; k++) {
                        ls_dwordx4_tmp_var = *((uint4*)&K_ptr[kv_idx * head_num * head_dim + dim_offset + k * 4]);
                        fetch_K[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                        fetch_K[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                        fetch_K[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                        fetch_K[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                    }
                }
                for(int k = 0; k < block_k; k++) {
                    results[kv_idx] += static_cast<float>(fetch_Q[k]) * static_cast<float>(fetch_K[k]);
                }
            }
        }
        
        for(int i = 0; i < seq_kv; i++) {
            score_ptr[i] = T(results[i] * scale);
        }
        for(int i = seq_kv; i < max_seq_kv; i++) {
            score_ptr[i] = T(-1e9f);
        }
    }
}

// Forward Kernel 2: Apply mask and softmax
template <typename T>
__global__ void apply_mask_and_softmax_kernel_runtime(
    T* scores,
    const T* dropout_mask,
    float dropout_scale,
    const int* cu_seqlens_kv,
    int batch_size,
    int head_num,
    int seq_q,
    int max_seq_kv,
    int mask_type, // 0=NO_MASK, 1=TOP_LEFT, 2=BOTTOM_RIGHT
    bool enable_dropout) {
    constexpr int block_size = 256;
    constexpr int per_score_size = 1 * 16; // seq_q=1, max_seq_kv=16
    constexpr int valid_thread_range = block_size / per_score_size * per_score_size;
    
    const uint32_t block_id = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t cur_block_offset = block_id * valid_thread_range + thread_id;
    const uint32_t total_elt = batch_size * head_num * seq_q * max_seq_kv;
    
    if(cur_block_offset >= total_elt || thread_id >= valid_thread_range)
        return;
    
    __shared__ T tmp_scores[valid_thread_range];
    // row_num = valid_thread_range / max_seq_kv
    // valid_thread_range = 256 / 16 * 16 = 256
    // max_seq_kv <= 16, so row_num ranges from 16 (max_seq_kv=16) to 128 (max_seq_kv=2)
    // Use maximum possible size for shared memory (worst case: max_seq_kv=2, row_num=128)
    constexpr int max_row_num = valid_thread_range / 2; // = 128 (worst case)
    __shared__ T row_max[max_row_num];
    __shared__ T row_sum[max_row_num];
    const int row_num = valid_thread_range / max_seq_kv;
    
    int global_row_idx = cur_block_offset / max_seq_kv;
    int batch_idx = global_row_idx / (seq_q * head_num);
    int k_idx = cur_block_offset % max_seq_kv;
    int local_row_idx = thread_id / max_seq_kv;
    
    int seq_kv = (batch_idx < batch_size) ? (cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx]) : max_seq_kv;
    
    T score_value = scores[cur_block_offset];
    tmp_scores[thread_id] = score_value;
    
    // Apply mask
    if(mask_type == 1) { // TOP_LEFT
        int q_idx = 0; // seq_q = 1
        if(k_idx > q_idx || k_idx >= seq_kv) {
            tmp_scores[thread_id] = T(-1e9f);
        }
    } else if(mask_type == 2) { // BOTTOM_RIGHT
        int q_idx = 0;
        if(k_idx < q_idx || k_idx >= seq_kv) {
            tmp_scores[thread_id] = T(-1e9f);
        }
    } else { // NO_MASK or PADDING_MASK
        if(k_idx >= seq_kv) {
            tmp_scores[thread_id] = T(-1e9f);
        }
    }
    __syncthreads();
    
    // Find max - each row finds its max value
    if(local_row_idx < row_num && local_row_idx < max_row_num) {
        T max_val = T(-1e9f);
        int row_start = local_row_idx * max_seq_kv;
        int row_end = row_start + max_seq_kv;
        for(int i = row_start; i < row_end && i < valid_thread_range; i++) {
            max_val = max(max_val, tmp_scores[i]);
        }
        row_max[local_row_idx] = max_val;
    }
    __syncthreads();
    
    // Compute exp and sum
    T row_max_val = (local_row_idx < max_row_num) ? row_max[local_row_idx] : T(-1e9f);
    T exp_val = T(exp(float(tmp_scores[thread_id] - row_max_val)));
    tmp_scores[thread_id] = exp_val;
    __syncthreads();
    
    if(local_row_idx < row_num && local_row_idx < max_row_num) {
        T sum = T(0.0f);
        int row_start = local_row_idx * max_seq_kv;
        int row_end = row_start + max_seq_kv;
        for(int i = row_start; i < row_end && i < valid_thread_range; i++) {
            sum += tmp_scores[i];
        }
        row_sum[local_row_idx] = sum;
    }
    __syncthreads();
    
    // Normalize and apply dropout
    T row_sum_val = (local_row_idx < max_row_num) ? row_sum[local_row_idx] : T(1.0f);
    T attn_weight = tmp_scores[thread_id] / row_sum_val;
    if(enable_dropout && dropout_mask) {
        attn_weight = attn_weight * dropout_mask[cur_block_offset] * T(dropout_scale);
    }
    
    scores[cur_block_offset] = attn_weight;
}

// Forward Kernel 3: Compute output = attn_weights @ V
template <typename T>
__global__ void compute_output_kernel_runtime(
    const T* attn_weights,
    const T* V,
    T* O,
    const int* cu_seqlens_kv,
    const int* cu_seqlens_kv_padded,
    int batch_size,
    int head_num,
    int seq_q,
    int max_seq_kv,
    int head_dim) {
    constexpr int block_k = 8;
    constexpr int dwordx4_load_elt = 16 / sizeof(T);
    constexpr int warp_size = 64;
    constexpr int tasks_per_block = 2; // Match original template
    // For BF16/FP16: sizeof(T)=2, so dwordx4_load_elt=8, block_k/dwordx4_load_elt=1
    // For Float32: sizeof(T)=4, so dwordx4_load_elt=4, block_k/dwordx4_load_elt=2
    constexpr int array_size = block_k / dwordx4_load_elt;
    static_assert(array_size > 0, "array_size must be > 0");
    
    int process_head_per_warp = warp_size / (head_dim / block_k);
    int base_block_offset = blockIdx.x * process_head_per_warp * tasks_per_block;
    int thread_id = threadIdx.x;
    int thread_batch_offset = thread_id / (head_dim / block_k);
    int thread_head_offset = thread_id % (head_dim / block_k) * block_k;
    
    uint4 load_dwordx4_tmp_var[array_size];
    uint4 store_dwordx4_tmp_var[array_size];
    T attn[max_seq_kv];
    
    // Process multiple tasks per block (matching original template)
    for(int task = 0; task < tasks_per_block; task++) {
        int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
        int cur_idx = block_batch_head_idx + thread_batch_offset;
        
        int batch_idx = cur_idx / (seq_q * head_num);
        int seq_head_idx = cur_idx % (seq_q * head_num);
        int seq_q_idx = seq_head_idx / head_num;
        int head_idx = seq_head_idx % head_num;
        
        if(batch_idx >= batch_size)
            continue;
        
        int seq_kv = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int kv_offset = cu_seqlens_kv_padded[batch_idx];
        
        // Initialize store buffer to zero (matching original)
        for(int i = 0; i < array_size; i++) {
            store_dwordx4_tmp_var[i].x = 0;
            store_dwordx4_tmp_var[i].y = 0;
            store_dwordx4_tmp_var[i].z = 0;
            store_dwordx4_tmp_var[i].w = 0;
        }
        
        // Load attention weights
        for(int i = 0; i < max_seq_kv; i++)
            attn[i] = attn_weights[cur_idx * max_seq_kv + i];
        
        // Compute output = attn_weights @ V
        for(int j = 0; j < seq_kv; j++) {
            // Load V values using vectorized loads
            for(int i = 0; i < array_size; i++) {
                load_dwordx4_tmp_var[i] = *((uint4*)&V[((kv_offset + j) * head_num + head_idx) * head_dim +
                                                     thread_head_offset + i * dwordx4_load_elt]);
            }
            // Accumulate: store += attn[j] * load
            for(int b = 0; b < block_k; b++) {
                int array_idx = b / dwordx4_load_elt;
                ((T*)&store_dwordx4_tmp_var[array_idx])[b % dwordx4_load_elt] +=
                    attn[j] * ((T*)&load_dwordx4_tmp_var[array_idx])[b % dwordx4_load_elt];
            }
        }
        
        // Store output using vectorized stores
        for(int i = 0; i < array_size; i++) {
            *((uint4*)&O[(batch_idx * seq_q * head_num + seq_q_idx * head_num + head_idx) * head_dim +
                         thread_head_offset + i * dwordx4_load_elt]) = store_dwordx4_tmp_var[i];
        }
    }
}

// Helper function to convert mask type to kernel parameter
int mask_type_to_kernel_param(NVTE_Mask_Type mask_type) {
    if(mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK || 
       mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) {
        return 1; // TOP_LEFT
    } else if(mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK ||
              mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) {
        return 2; // BOTTOM_RIGHT
    }
    return 0; // NO_MASK or PADDING_MASK
}

// Forward launcher function
void fused_attn_unfused_smallseq_fwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, 
  size_t d_qk, size_t d_v,
  bool is_training, float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, const Tensor* input_Bias, 
  Tensor* output_O, NVTETensorPack *Aux_CTX_Tensors,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor *workspace,
  cudaStream_t stream) {
  
  const DType QKV_type = input_Q->data.dtype;
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;
  void *devPtrO = output_O->data.dptr;
  
  void *devPtrCuSeqlensQ = input_cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = input_cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = input_cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = input_cu_seqlens_kv_padded->data.dptr;
  
  // Debug: confirm specialized unfused_smallseq kernel is called (set NVTE_DEBUG_UNFUSED_SMALLSEQ=1)
  {
    bool debug = false;
    if (const char* p = std::getenv("NVTE_DEBUG_UNFUSED_SMALLSEQ")) debug = (std::string(p) == "1");
    if (debug) {
      std::cerr << "[Unfused_SmallSeq FWD] ENTERED specialized kernel: b=" << b
                << " h_q=" << h_q << " h_kv=" << h_kv
                << " max_seqlen_q=" << max_seqlen_q << " max_seqlen_kv=" << max_seqlen_kv
                << " d_qk=" << d_qk << " d_v=" << d_v << std::endl;
    }
  }
  // Optional verbose config (set NVTE_LOG_UNFUSED_CONFIG=1)
  bool nvte_log_unfused_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_UNFUSED_CONFIG")) {
    if(std::string(env_p) == "1") {
      nvte_log_unfused_config = true;
    }
  }
  if(nvte_log_unfused_config) {
    std::cerr << "[Unfused_SmallSeq FWD] Pointers: Q=" << devPtrQ << ", K=" << devPtrK
              << ", V=" << devPtrV << ", O=" << devPtrO
              << ", Workspace=" << workspace->data.dptr << std::endl;
  }
  
  // Calculate workspace size needed for attention weights
  // Attention weights shape: [batch, head_num, seq_q, max_seq_kv] = [b, h_q, 1, max_seqlen_kv]
  size_t attn_weights_size = b * h_q * max_seqlen_q * max_seqlen_kv * nvte_dtype_size(QKV_type);
  
  // Handle workspace size calculation if workspace is not allocated
  if(workspace->data.dptr == nullptr) {
    workspace->data.shape = {attn_weights_size};
    workspace->data.dtype = DType::kByte;
    return;
  }
  
  void *devPtrWorkspace = workspace->data.dptr;
  
  // Set up Aux_CTX_Tensors to store attention weights in softmax_LSE buffer
  // This is a "storage hack": we reuse the softmax_LSE storage for attention weights
  // The softmax_LSE buffer is typically [max_tokens_q, h_q, 1] for THD layout
  // Attention weights are [b, h_q, 1, max_seqlen_kv] = [max_tokens_q, h_q, max_seqlen_kv]
  // Since seq_q=1, max_tokens_q = b, so the shapes are compatible
  void *devPtrAttnWeights = nullptr;
  bool is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;
  size_t max_tokens_q = std::accumulate((input_Q->data).shape.begin(), (input_Q->data).shape.end(), 
                                        static_cast<size_t>(1), std::multiplies<size_t>()) / h_q / d_qk;
  
  // Calculate softmax_LSE buffer size (what CK would allocate)
  // For THD layout: [max_tokens_q, h_q, 1] with float32
  size_t softmax_lse_size = max_tokens_q * h_q * 1 * sizeof(float);
  
  // Calculate attention weights buffer size
  size_t attn_weights_buffer_size = b * h_q * max_seqlen_q * max_seqlen_kv * nvte_dtype_size(QKV_type);
  
  // Validate that attention weights fit in softmax_LSE buffer
  // For BF16/FP16: attn_weights_buffer_size = b * h_q * 1 * max_seqlen_kv * 2 bytes
  // For float32 softmax_LSE: softmax_lse_size = max_tokens_q * h_q * 1 * 4 bytes
  // Since max_tokens_q = b (for seq_q=1), we have:
  //   attn_weights_buffer_size = b * h_q * max_seqlen_kv * 2
  //   softmax_lse_size = b * h_q * 4
  // We need: b * h_q * max_seqlen_kv * 2 <= b * h_q * 4
  // Which means: max_seqlen_kv * 2 <= 4, so max_seqlen_kv <= 2
  // But we support up to 16, so we need to use a different approach:
  // We'll store attention weights in the workspace, and use Aux_CTX_Tensors
  // to point to the workspace (or allocate a separate buffer if needed)
  
  if (Aux_CTX_Tensors->size == 0) {
    // Allocate Aux_CTX_Tensors for attention weights storage
    // We'll use the workspace for attention weights, but set up Aux_CTX_Tensors
    // to point to it so backward can access it
    Aux_CTX_Tensors->size = 2;
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    output_S->data.dptr = nullptr; // Will be set to workspace pointer
    if(is_ragged) {
      // For THD layout, attention weights shape: [max_tokens_q, h_q, max_seqlen_kv]
      // This matches the logical layout even though we're using workspace
      output_S->data.shape = {max_tokens_q, h_q, max_seqlen_kv};
    } else {
      output_S->data.shape = {b, h_q, max_seqlen_q, max_seqlen_kv};
    }
    output_S->data.dtype = QKV_type; // Store as same dtype as QKV (not float32 like softmax_LSE)
    
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = nullptr;
    output_rng_state->data.shape = {2};
    output_rng_state->data.dtype = DType::kInt64;
  } else if (Aux_CTX_Tensors->size >= 1) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrAttnWeights = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    if(output_rng_state && rng_state) {
      output_rng_state->data.dptr = rng_state->data.dptr;
    }
  }
  
  // Use workspace for attention weights (storage hack: we'll copy to Aux_CTX_Tensors if needed)
  // For now, we use workspace directly and copy to Aux_CTX_Tensors buffer after computation
  if(!devPtrAttnWeights) {
    // If Aux_CTX_Tensors buffer is not allocated, use workspace directly
    devPtrAttnWeights = devPtrWorkspace;
  }
  
  int mask_type_param = mask_type_to_kernel_param(attn_mask_type);
  bool enable_dropout = (dropout > 0.0f) && is_training;
  float dropout_scale = enable_dropout ? (1.0f / (1.0f - dropout)) : 1.0f;
  
  // Launch kernels based on data type
  // Note: This backend only supports FP16/BF16, so use NON_FP8ONLY macro
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(QKV_type, Type, {
    // Kernel 1: Compute scores
    int merge_bs = b * h_q;
    dim3 block1(64);
    dim3 grid1((merge_bs + 63) / 64);
    compute_scores_kernel_runtime<Type><<<grid1, block1, 0, stream>>>(
      (const Type*)devPtrQ, (const Type*)devPtrK, (Type*)devPtrWorkspace,
      attn_scale, (const int*)devPtrCuSeqlensKV, (const int*)devPtrSeqOffsetsKV,
      b, h_q, 1, max_seqlen_kv, d_qk);
    // Do not call hipGetLastError() here: JAX uses HIP stream capture; it invalidates capture
    // (hipErrorStreamCaptureInvalidated 901) or causes unjoined fork (904). Errors are reported
    // when the stream is synchronized (e.g. at ffi_with_cuda_error_check or graph execution).

    // Kernel 2: Apply mask and softmax
    constexpr int block_size = 256;
    constexpr int per_score_size = 1 * 16;
    constexpr int valid_thread_range = block_size / per_score_size * per_score_size;
    dim3 grid2((merge_bs * 1 * max_seqlen_kv + valid_thread_range - 1) / valid_thread_range);
    dim3 block2(block_size);
    apply_mask_and_softmax_kernel_runtime<Type><<<grid2, block2, 0, stream>>>(
      (Type*)devPtrWorkspace, nullptr, dropout_scale,
      (const int*)devPtrCuSeqlensKV,
      b, h_q, 1, max_seqlen_kv, mask_type_param, enable_dropout);

    // Store attention weights for backward pass
    // If Aux_CTX_Tensors buffer is allocated, copy from workspace to it
    // Otherwise, the workspace itself contains the attention weights
    if(devPtrAttnWeights && devPtrAttnWeights != devPtrWorkspace) {
      // Copy attention weights from workspace to Aux_CTX_Tensors buffer
      size_t attn_weights_size = b * h_q * max_seqlen_q * max_seqlen_kv * sizeof(Type);
      NVTE_CHECK(hipMemcpyAsync(devPtrAttnWeights, devPtrWorkspace, attn_weights_size, 
                                hipMemcpyDeviceToDevice, stream) == hipSuccess,
                 "Failed to copy attention weights to Aux_CTX_Tensors buffer");
    } else if(Aux_CTX_Tensors->size >= 1) {
      // Set Aux_CTX_Tensors to point to workspace (if buffer was pre-allocated)
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      if(output_S->data.dptr == nullptr) {
        // Buffer not allocated yet, point to workspace
        output_S->data.dptr = devPtrWorkspace;
      }
    }
    
    // Kernel 3: Compute output
    // Ensure d_v is divisible by 8 for vectorized loads
    NVTE_CHECK(d_v % 8 == 0, "head_dim_v must be divisible by 8 for vectorized loads");
    constexpr int tasks_per_block = 2; // Match original template
    int process_head_per_warp = 64 / (d_v / 8);
    if(process_head_per_warp == 0) process_head_per_warp = 1; // Safety check
    // Original uses: (merge_bs / process_head_per_warp + 2 - 1) / 2
    // Which is equivalent to: ceil(merge_bs / process_head_per_warp / tasks_per_block)
    dim3 grid3((merge_bs / process_head_per_warp + tasks_per_block - 1) / tasks_per_block);
    dim3 block3(64);
    compute_output_kernel_runtime<Type><<<grid3, block3, 0, stream>>>(
      (const Type*)devPtrWorkspace, (const Type*)devPtrV, (Type*)devPtrO,
      (const int*)devPtrCuSeqlensKV, (const int*)devPtrSeqOffsetsKV,
      b, h_q, 1, max_seqlen_kv, d_v);
  });
}

// Backward launcher function (stub for now - full implementation in next phase)
void fused_attn_unfused_smallseq_bwd(
  size_t b, size_t h_q, size_t h_kv, size_t max_seqlen_q, size_t max_seqlen_kv, 
  size_t d_qk, size_t d_v,
  float attn_scale, float dropout, 
  NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  bool deterministic,
  const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V, 
  const Tensor* input_O, const Tensor* input_dO, const Tensor* input_Bias, 
  const Tensor* output_S,
  Tensor* output_dQ, Tensor* output_dK, Tensor* output_dV,
  Tensor* output_dBias,
  const Tensor* input_cu_seqlens_q,
  const Tensor* input_cu_seqlens_kv,
  const Tensor* input_cu_seqlens_q_padded,
  const Tensor* input_cu_seqlens_kv_padded,
  const Tensor* rng_state,
  Tensor* workspace,
  cudaStream_t stream) {
  
  // Debug: confirm specialized unfused_smallseq backward is called (set NVTE_DEBUG_UNFUSED_SMALLSEQ=1)
  {
    bool debug_bwd = false;
    if (const char* p = std::getenv("NVTE_DEBUG_UNFUSED_SMALLSEQ")) debug_bwd = (std::string(p) == "1");
    if (debug_bwd) {
      std::cerr << "[Unfused_SmallSeq BWD] ENTERED specialized kernel: b=" << b
                << " h_q=" << h_q << " h_kv=" << h_kv
                << " max_seqlen_q=" << max_seqlen_q << " max_seqlen_kv=" << max_seqlen_kv
                << " d_qk=" << d_qk << " d_v=" << d_v << std::endl;
    }
  }

  // Handle workspace size calculation if workspace is not allocated
  const DType QKV_type = input_Q->data.dtype;
  size_t workspace_size = b * h_q * max_seqlen_q * max_seqlen_kv * nvte_dtype_size(QKV_type);
  if(workspace->data.dptr == nullptr) {
    workspace->data.shape = {workspace_size};
    workspace->data.dtype = DType::kByte;
    return;
  }
  
  // Extract attention weights from output_S (which was stored in forward pass)
  // output_S points to the Aux_CTX_Tensors buffer that contains attention weights
  void *devPtrAttnWeights = nullptr;
  
  if(output_S && output_S->data.dptr) {
    devPtrAttnWeights = output_S->data.dptr;
  } else {
    NVTE_ERROR("Attention weights not found in Aux_CTX_Tensors. Backward pass requires forward pass attention weights.");
  }
  
  // Validate attention weights buffer size
  size_t expected_attn_weights_size = b * h_q * max_seqlen_q * max_seqlen_kv * nvte_dtype_size(QKV_type);
  size_t actual_buffer_size = std::accumulate(output_S->data.shape.begin(), output_S->data.shape.end(),
                                              static_cast<size_t>(1), std::multiplies<size_t>()) *
                              nvte_dtype_size(output_S->data.dtype);
  
  if(actual_buffer_size < expected_attn_weights_size) {
    NVTE_ERROR("Attention weights buffer size mismatch. Expected at least " + 
               std::to_string(expected_attn_weights_size) + " bytes, got " + 
               std::to_string(actual_buffer_size) + " bytes.");
  }
  
  // TODO: Implement backward kernels (grad_V, grad_attn, softmax_backward, grad_Q/grad_K)
  NVTE_ERROR("Backward pass not yet implemented for unfused_smallseq backend");
}

}  // namespace fused_attn_rocm
}  // namespace transformer_engine
