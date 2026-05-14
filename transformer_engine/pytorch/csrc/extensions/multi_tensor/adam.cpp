/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

// Cache for device-side mapping arrays used by the custom adam param-remainder
// kernel.  During training the tensor list structure (shapes and data pointers)
// is typically identical across iterations, so we can avoid per-call device
// allocations and H2D memcpy by caching the arrays and only re-uploading
// when something changes.
struct CustomAdamParamRemainderCache {
  static constexpr int kDepth = 5;  // g, p, m, v, p_remainder

  std::vector<int64_t> addresses_host;  // [ntensors * kDepth]
  std::vector<int64_t> sizes_host;      // [ntensors]
  at::Tensor addresses_dev;
  at::Tensor sizes_dev;
  at::Tensor block_to_tensor_dev;
  at::Tensor chunk_offsets_dev;
  int total_chunks = 0;
  int chunk_size = 0;

  bool shapes_valid(int ntensors, int cs,
                    const std::vector<std::vector<at::Tensor>> &tensor_lists) const {
    if (chunk_size != cs || static_cast<int>(sizes_host.size()) != ntensors)
      return false;
    for (int t = 0; t < ntensors; t++) {
      if (sizes_host[t] != static_cast<int64_t>(tensor_lists[0][t].numel()))
        return false;
    }
    return true;
  }

  bool addresses_valid(int ntensors,
                       const std::vector<std::vector<at::Tensor>> &tensor_lists) const {
    if (static_cast<int>(addresses_host.size()) != ntensors * kDepth)
      return false;
    for (int t = 0; t < ntensors; t++) {
      for (int d = 0; d < kDepth; d++) {
        if (addresses_host[t * kDepth + d] !=
            reinterpret_cast<int64_t>(tensor_lists[d][t].data_ptr()))
          return false;
      }
    }
    return true;
  }

  void rebuild(int ntensors, int cs, int tc,
               const std::vector<std::vector<at::Tensor>> &tensor_lists,
               cudaStream_t stream) {
    chunk_size = cs;
    total_chunks = tc;

    addresses_host.clear();
    sizes_host.clear();
    std::vector<int> block_to_tensor_host;
    std::vector<int> chunk_offsets_host;
    addresses_host.reserve(ntensors * kDepth);
    sizes_host.reserve(ntensors);
    block_to_tensor_host.reserve(tc);
    chunk_offsets_host.reserve(ntensors);

    int running_offset = 0;
    for (int t = 0; t < ntensors; t++) {
      const auto &tensor = tensor_lists[0][t];
      const int64_t tensor_numel = tensor.numel();
      const int chunks_this_tensor = static_cast<int>(
          (tensor_numel + cs - 1) / cs);
      for (int d = 0; d < kDepth; d++) {
        addresses_host.push_back(
            reinterpret_cast<int64_t>(tensor_lists[d][t].data_ptr()));
      }
      sizes_host.push_back(tensor_numel);
      chunk_offsets_host.push_back(running_offset);
      for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
        block_to_tensor_host.push_back(t);
      }
      running_offset += chunks_this_tensor;
    }

    auto int_options = tensor_lists[0][0].options().dtype(at::kInt);
    auto long_options = tensor_lists[0][0].options().dtype(at::kLong);
    addresses_dev = at::empty({ntensors * kDepth}, long_options);
    sizes_dev = at::empty({ntensors}, long_options);
    block_to_tensor_dev = at::empty({tc}, int_options);
    chunk_offsets_dev = at::empty({ntensors}, int_options);

    NVTE_CHECK_CUDA(cudaMemcpyAsync(addresses_dev.data_ptr(), addresses_host.data(),
                                    ntensors * kDepth * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(sizes_dev.data_ptr(), sizes_host.data(),
                                    ntensors * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(block_to_tensor_dev.data_ptr(),
                                    block_to_tensor_host.data(),
                                    tc * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(chunk_offsets_dev.data_ptr(),
                                    chunk_offsets_host.data(),
                                    ntensors * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
  }

  void update_addresses(int ntensors,
                        const std::vector<std::vector<at::Tensor>> &tensor_lists,
                        cudaStream_t stream) {
    addresses_host.clear();
    addresses_host.reserve(ntensors * kDepth);
    for (int t = 0; t < ntensors; t++) {
      for (int d = 0; d < kDepth; d++) {
        addresses_host.push_back(
            reinterpret_cast<int64_t>(tensor_lists[d][t].data_ptr()));
      }
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(addresses_dev.data_ptr(),
                                    addresses_host.data(),
                                    ntensors * kDepth * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
  }

  int ensure(int ntensors, int cs, int tc,
             const std::vector<std::vector<at::Tensor>> &tensor_lists,
             cudaStream_t stream) {
    if (!shapes_valid(ntensors, cs, tensor_lists)) {
      rebuild(ntensors, cs, tc, tensor_lists, stream);
    } else if (!addresses_valid(ntensors, tensor_lists)) {
      update_addresses(ntensors, tensor_lists, stream);
    }
    return total_chunks;
  }
};

static CustomAdamParamRemainderCache g_adam_param_remainder_cache;

template <int kDepth>
struct CustomAdamCache {

  std::vector<int64_t> addresses_host;
  std::vector<int64_t> sizes_host;
  at::Tensor addresses_dev;
  at::Tensor sizes_dev;
  at::Tensor block_to_tensor_dev;
  at::Tensor chunk_offsets_dev;
  int total_chunks = 0;
  int chunk_size = 0;

  bool shapes_valid(int ntensors, int cs,
                    const std::vector<std::vector<at::Tensor>> &tensor_lists) const {
    if (chunk_size != cs || static_cast<int>(sizes_host.size()) != ntensors)
      return false;
    for (int t = 0; t < ntensors; t++) {
      if (sizes_host[t] != static_cast<int64_t>(tensor_lists[0][t].numel()))
        return false;
    }
    return true;
  }

  bool addresses_valid(int ntensors,
                       const std::vector<std::vector<at::Tensor>> &tensor_lists) const {
    if (static_cast<int>(addresses_host.size()) != ntensors * kDepth)
      return false;
    for (int t = 0; t < ntensors; t++) {
      for (int d = 0; d < kDepth; d++) {
        if (addresses_host[t * kDepth + d] !=
            reinterpret_cast<int64_t>(tensor_lists[d][t].data_ptr()))
          return false;
      }
    }
    return true;
  }

  void rebuild(int ntensors, int cs, int tc,
               const std::vector<std::vector<at::Tensor>> &tensor_lists,
               cudaStream_t stream) {
    chunk_size = cs;
    total_chunks = tc;

    addresses_host.clear();
    sizes_host.clear();
    std::vector<int> block_to_tensor_host;
    std::vector<int> chunk_offsets_host;
    addresses_host.reserve(ntensors * kDepth);
    sizes_host.reserve(ntensors);
    block_to_tensor_host.reserve(tc);
    chunk_offsets_host.reserve(ntensors);

    int running_offset = 0;
    for (int t = 0; t < ntensors; t++) {
      const auto &tensor = tensor_lists[0][t];
      const int64_t tensor_numel = tensor.numel();
      const int chunks_this_tensor = static_cast<int>(
          (tensor_numel + cs - 1) / cs);
      for (int d = 0; d < kDepth; d++) {
        addresses_host.push_back(
            reinterpret_cast<int64_t>(tensor_lists[d][t].data_ptr()));
      }
      sizes_host.push_back(tensor_numel);
      chunk_offsets_host.push_back(running_offset);
      for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
        block_to_tensor_host.push_back(t);
      }
      running_offset += chunks_this_tensor;
    }

    auto int_options = tensor_lists[0][0].options().dtype(at::kInt);
    auto long_options = tensor_lists[0][0].options().dtype(at::kLong);
    addresses_dev = at::empty({ntensors * kDepth}, long_options);
    sizes_dev = at::empty({ntensors}, long_options);
    block_to_tensor_dev = at::empty({tc}, int_options);
    chunk_offsets_dev = at::empty({ntensors}, int_options);

    NVTE_CHECK_CUDA(cudaMemcpyAsync(addresses_dev.data_ptr(), addresses_host.data(),
                                    ntensors * kDepth * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(sizes_dev.data_ptr(), sizes_host.data(),
                                    ntensors * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(block_to_tensor_dev.data_ptr(),
                                    block_to_tensor_host.data(),
                                    tc * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(chunk_offsets_dev.data_ptr(),
                                    chunk_offsets_host.data(),
                                    ntensors * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
  }

  void update_addresses(int ntensors,
                        const std::vector<std::vector<at::Tensor>> &tensor_lists,
                        cudaStream_t stream) {
    addresses_host.clear();
    addresses_host.reserve(ntensors * kDepth);
    for (int t = 0; t < ntensors; t++) {
      for (int d = 0; d < kDepth; d++) {
        addresses_host.push_back(
            reinterpret_cast<int64_t>(tensor_lists[d][t].data_ptr()));
      }
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(addresses_dev.data_ptr(),
                                    addresses_host.data(),
                                    ntensors * kDepth * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
  }

  int ensure(int ntensors, int cs, int tc,
             const std::vector<std::vector<at::Tensor>> &tensor_lists,
             cudaStream_t stream) {
    if (!shapes_valid(ntensors, cs, tensor_lists)) {
      rebuild(ntensors, cs, tc, tensor_lists, stream);
    } else if (!addresses_valid(ntensors, tensor_lists)) {
      update_addresses(ntensors, tensor_lists, stream);
    }
    return total_chunks;
  }
};

static CustomAdamCache<4> g_adam_cache;          // g, p, m, v
static CustomAdamCache<5> g_adam_master_cache;   // g, p, m, v, p_master

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
