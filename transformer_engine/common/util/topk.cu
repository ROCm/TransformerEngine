/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/topk.h>

#include "../common.h"
#include "standalone_topk.cuh"

void nvte_topk(cudaStream_t stream, const NVTETensor keys_in, const NVTETensor lengths_in,
               NVTETensor keys_out, NVTETensor indices_out, NVTETensor workspace, int batch_size,
               int seq_len, int k) {
  NVTE_API_CALL(nvte_topk);
  using namespace transformer_engine;

  Tensor *workspace_tensor = convertNVTETensor(workspace);

  if (workspace_tensor->data.numel() == 0) {
    size_t workspace_bytes = 0;
    nv::standalone_topk<float, int>(nullptr, workspace_bytes, nullptr, batch_size, seq_len, k,
                                    nullptr, nullptr, /*greater=*/true, /*stream=*/nullptr,
                                    /*lengths=*/nullptr, /*is_prefill=*/false);
    workspace_tensor->data.shape = {workspace_bytes};
    workspace_tensor->data.dtype = DType::kByte;
    return;
  }

  const Tensor *keys_in_tensor = convertNVTETensorCheck(keys_in);
  const Tensor *lengths_tensor = convertNVTETensorCheck(lengths_in);
  Tensor *keys_out_tensor = convertNVTETensor(keys_out);
  Tensor *indices_tensor = convertNVTETensor(indices_out);

  void *d_workspace = workspace_tensor->data.dptr;
  size_t workspace_bytes = workspace_tensor->data.numel();
  const int *d_lengths = reinterpret_cast<const int *>(lengths_tensor->data.dptr);
  int *d_indices = reinterpret_cast<int *>(indices_tensor->data.dptr);

  auto dtype = keys_in_tensor->data.dtype;

#define DISPATCH_TOPK(T)                                                                           \
  do {                                                                                             \
    const T *d_in = reinterpret_cast<const T *>(keys_in_tensor->data.dptr);                        \
    T *d_out = reinterpret_cast<T *>(keys_out_tensor->data.dptr);                                  \
    nv::standalone_topk<T, int>(d_workspace, workspace_bytes, d_in, batch_size, seq_len, k, d_out, \
                                d_indices, /*greater=*/true, stream, const_cast<int *>(d_lengths), \
                                /*is_prefill=*/false);                                             \
  } while (0)

  if (dtype == DType::kBFloat16) {
#ifdef __HIP_PLATFORM_AMD__
    // hipCUB specializes NumericTraits for hip_bfloat16, not __hip_bfloat16,
    // and the radix select needs those traits to twiddle the key bits.
    DISPATCH_TOPK(hip_bfloat16);
#else
    DISPATCH_TOPK(__nv_bfloat16);
#endif
  } else if (dtype == DType::kFloat32) {
    DISPATCH_TOPK(float);
  } else {
    NVTE_ERROR("nvte_topk: unsupported key dtype (supported: float32, bfloat16)");
  }

#undef DISPATCH_TOPK
}
