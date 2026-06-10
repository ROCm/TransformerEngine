/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"
#include "multi_tensor_cache.h"

namespace transformer_engine::pytorch {

static CustomMultiTensorCache<5> g_adam_param_remainder_cache;  // g, p, m, v, p_remainder
static CustomMultiTensorCache<4> g_adam_cache;                  // g, p, m, v
static CustomMultiTensorCache<5> g_adam_master_cache;           // g, p, m, v, p_master

void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay) {
  const size_t num_lists = tensor_lists.size();
  const int ntensors = tensor_lists[0].size();
  int total_chunks = 0;
  for (int t = 0; t < ntensors; t++) {
    total_chunks += static_cast<int>(
        (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size);
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);

  if (num_lists == 4) {
    g_adam_cache.ensure(ntensors, chunk_size, total_chunks, tensor_lists, stream);
    nvte_multi_tensor_adam_cuda_custom(
        chunk_size, noop_flag_cu.data(),
        static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[0][0].scalar_type())),
        static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[1][0].scalar_type())),
        static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[2][0].scalar_type())),
        g_adam_cache.addresses_dev.data_ptr<int64_t>(),
        g_adam_cache.sizes_dev.data_ptr<int64_t>(),
        g_adam_cache.block_to_tensor_dev.data_ptr<int>(),
        g_adam_cache.chunk_offsets_dev.data_ptr<int>(),
        total_chunks, 0,
        lr, beta1, beta2, epsilon, step, mode, bias_correction,
        weight_decay, stream);
  } else {
    g_adam_master_cache.ensure(ntensors, chunk_size, total_chunks, tensor_lists, stream);
    nvte_multi_tensor_adam_cuda_custom(
        chunk_size, noop_flag_cu.data(),
        static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[0][0].scalar_type())),
        static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[1][0].scalar_type())),
        static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[2][0].scalar_type())),
        g_adam_master_cache.addresses_dev.data_ptr<int64_t>(),
        g_adam_master_cache.sizes_dev.data_ptr<int64_t>(),
        g_adam_master_cache.block_to_tensor_dev.data_ptr<int>(),
        g_adam_master_cache.chunk_offsets_dev.data_ptr<int>(),
        total_chunks, 1,
        lr, beta1, beta2, epsilon, step, mode, bias_correction,
        weight_decay, stream);
  }
}

void multi_tensor_adam_param_remainder_cuda(int chunk_size, at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay) {
  const int ntensors = tensor_lists[0].size();
  int total_chunks = 0;
  for (int t = 0; t < ntensors; t++) {
    total_chunks += static_cast<int>(
        (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  g_adam_param_remainder_cache.ensure(ntensors, chunk_size, total_chunks,
                                     tensor_lists, stream);

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  nvte_multi_tensor_adam_param_remainder_cuda_custom(
      chunk_size, noop_flag_cu.data(),
      static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[0][0].scalar_type())),
      static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[2][0].scalar_type())),
      g_adam_param_remainder_cache.addresses_dev.data_ptr<int64_t>(),
      g_adam_param_remainder_cache.sizes_dev.data_ptr<int64_t>(),
      g_adam_param_remainder_cache.block_to_tensor_dev.data_ptr<int>(),
      g_adam_param_remainder_cache.chunk_offsets_dev.data_ptr<int>(),
      total_chunks, lr, beta1, beta2, epsilon, step, mode, bias_correction,
      weight_decay, stream);
}

void multi_tensor_adam_fp8_cuda(int chunk_size, at::Tensor noop_flag,
                                std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, DType fp8_dtype) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_adam_fp8_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(),
                                  num_lists, num_tensors, lr, beta1, beta2, epsilon, step, mode,
                                  bias_correction, weight_decay, static_cast<NVTEDType>(fp8_dtype),
                                  at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_capturable_cuda(int chunk_size, at::Tensor noop_flag,
                                       std::vector<std::vector<at::Tensor>> tensor_lists,
                                       at::Tensor lr, const float beta1, const float beta2,
                                       const float epsilon, at::Tensor step, const int mode,
                                       const int bias_correction, const float weight_decay,
                                       at::Tensor inv_scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      lr_cu.data(), beta1, beta2, epsilon, step_cu.data(), mode, bias_correction, weight_decay,
      inv_scale_cu.data(), at::cuda::getCurrentCUDAStream());
}

void multi_tensor_adam_capturable_master_cuda(int chunk_size, at::Tensor noop_flag,
                                              std::vector<std::vector<at::Tensor>> tensor_lists,
                                              at::Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, at::Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              at::Tensor inv_scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_master_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      lr_cu.data(), beta1, beta2, epsilon, step_cu.data(), mode, bias_correction, weight_decay,
      inv_scale_cu.data(), at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
