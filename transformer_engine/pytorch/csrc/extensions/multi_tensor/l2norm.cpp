/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

// Cache for device-side mapping arrays used by the custom l2norm kernel.
// During training the tensor list structure (shapes and data pointers) is
// typically identical across iterations, so we can avoid per-call device
// allocations and H2D memcpy by caching the arrays and only re-uploading
// when something changes.
struct CustomL2NormCache {
  std::vector<int64_t> addresses_host;
  std::vector<int> sizes_host;
  at::Tensor addresses_dev;
  at::Tensor sizes_dev;
  at::Tensor block_to_tensor_dev;
  at::Tensor chunk_offsets_dev;
  at::Tensor output_dev;
  at::Tensor ret_dev;
  int total_chunks = 0;
  int chunk_size = 0;

  // Check whether the cached shape metadata still matches.
  bool shapes_valid(int ntensors, int cs,
                    const std::vector<std::vector<at::Tensor>> &tensor_lists) const {
    if (chunk_size != cs || static_cast<int>(sizes_host.size()) != ntensors)
      return false;
    for (int t = 0; t < ntensors; t++) {
      if (sizes_host[t] != static_cast<int>(tensor_lists[0][t].numel()))
        return false;
    }
    return true;
  }

  // Check whether cached data pointers still match.
  bool addresses_valid(int ntensors,
                       const std::vector<std::vector<at::Tensor>> &tensor_lists) const {
    if (static_cast<int>(addresses_host.size()) != ntensors)
      return false;
    for (int t = 0; t < ntensors; t++) {
      if (addresses_host[t] !=
          reinterpret_cast<int64_t>(tensor_lists[0][t].data_ptr()))
        return false;
    }
    return true;
  }

  // Full rebuild: shapes changed, so all arrays need to be re-created.
  void rebuild(int ntensors, int cs, int tc,
               const std::vector<std::vector<at::Tensor>> &tensor_lists,
               cudaStream_t stream) {
    chunk_size = cs;
    total_chunks = tc;

    addresses_host.clear();
    sizes_host.clear();
    std::vector<int> block_to_tensor_host;
    std::vector<int> chunk_offsets_host;
    addresses_host.reserve(ntensors);
    sizes_host.reserve(ntensors);
    block_to_tensor_host.reserve(tc);
    chunk_offsets_host.reserve(ntensors);

    int running_offset = 0;
    for (int t = 0; t < ntensors; t++) {
      const auto &tensor = tensor_lists[0][t];
      const int tensor_numel = static_cast<int>(tensor.numel());
      const int chunks_this_tensor = (tensor_numel + cs - 1) / cs;
      addresses_host.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
      sizes_host.push_back(tensor_numel);
      chunk_offsets_host.push_back(running_offset);
      for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
        block_to_tensor_host.push_back(t);
      }
      running_offset += chunks_this_tensor;
    }

    auto int_options = tensor_lists[0][0].options().dtype(at::kInt);
    auto long_options = tensor_lists[0][0].options().dtype(at::kLong);
    auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
    addresses_dev = at::empty({ntensors}, long_options);
    sizes_dev = at::empty({ntensors}, int_options);
    block_to_tensor_dev = at::empty({tc}, int_options);
    chunk_offsets_dev = at::empty({ntensors}, int_options);
    output_dev = at::empty({tc}, float_options);
    ret_dev = at::empty({1}, float_options);

    NVTE_CHECK_CUDA(cudaMemcpyAsync(addresses_dev.data_ptr(), addresses_host.data(),
                                    ntensors * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(sizes_dev.data_ptr(), sizes_host.data(),
                                    ntensors * sizeof(int),
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

  // Addresses-only update: shapes unchanged but data pointers moved.
  void update_addresses(int ntensors,
                        const std::vector<std::vector<at::Tensor>> &tensor_lists,
                        cudaStream_t stream) {
    addresses_host.clear();
    addresses_host.reserve(ntensors);
    for (int t = 0; t < ntensors; t++) {
      addresses_host.push_back(
          reinterpret_cast<int64_t>(tensor_lists[0][t].data_ptr()));
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(addresses_dev.data_ptr(),
                                    addresses_host.data(),
                                    ntensors * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
  }

  // Ensure the cache is up to date. Returns total_chunks.
  int ensure(int ntensors, int cs, int tc,
             const std::vector<std::vector<at::Tensor>> &tensor_lists,
             cudaStream_t stream) {
    if (!shapes_valid(ntensors, cs, tensor_lists)) {
      rebuild(ntensors, cs, tc, tensor_lists, stream);
    } else if (!addresses_valid(ntensors, tensor_lists)) {
      update_addresses(ntensors, tensor_lists, stream);
    }
    // else: full cache hit, no memcpy needed
    return total_chunks;
  }
};

static CustomL2NormCache g_l2norm_cache;
static CustomL2NormCache g_unscale_l2norm_cache;

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  const int ntensors = tensor_lists[0].size();
  int total_chunks = 0;
  for (int t = 0; t < ntensors; t++) {
    total_chunks += (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
  }

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);

  if (!per_tensor) {
    auto stream = at::cuda::getCurrentCUDAStream();
    g_l2norm_cache.ensure(ntensors, chunk_size, total_chunks, tensor_lists, stream);

    auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
    auto output_cu = makeTransformerEngineTensor(g_l2norm_cache.output_dev);
    auto ret_cu = makeTransformerEngineTensor(g_l2norm_cache.ret_dev);
    nvte_multi_tensor_l2norm_cuda_custom(
      chunk_size, noop_flag_cu.data(),
      static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[0][0].scalar_type())),
      g_l2norm_cache.addresses_dev.data_ptr<int64_t>(),
      g_l2norm_cache.sizes_dev.data_ptr<int>(),
      g_l2norm_cache.block_to_tensor_dev.data_ptr<int>(),
      g_l2norm_cache.chunk_offsets_dev.data_ptr<int>(),
      total_chunks, output_cu.data(), ret_cu.data(), stream);
    auto ret_per_tensor = at::empty({0}, float_options);
    return std::tuple<at::Tensor, at::Tensor>(g_l2norm_cache.ret_dev, ret_per_tensor);
  }

  // per_tensor path: use multi_tensor_apply
  const int output_size = total_chunks > 320 ? total_chunks : 320;
  auto output = at::zeros({output_size}, float_options);
  auto ret = at::empty({1}, float_options);

  int max_chunks_per_tensor = -1;
  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
    if (max_chunks_this_tensor > max_chunks_per_tensor)
      max_chunks_per_tensor = max_chunks_this_tensor;
  }
  auto output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
  auto ret_per_tensor = at::zeros({ntensors}, float_options);

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_cu = makeTransformerEngineTensor(output);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto ret_per_tensor_cu = makeTransformerEngineTensor(ret_per_tensor);

  nvte_multi_tensor_l2norm_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                                num_tensors, output_cu.data(), output_per_tensor_cu.data(),
                                ret_cu.data(), ret_per_tensor_cu.data(), per_tensor,
                                max_chunks_per_tensor, at::cuda::getCurrentCUDAStream());

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}

std::tuple<at::Tensor, at::Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::Tensor inv_scale, at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  const int ntensors = tensor_lists[0].size();
  int total_chunks = 0;
  for (int t = 0; t < ntensors; t++) {
    total_chunks += (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
  }

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);

  if (!per_tensor) {
    auto stream = at::cuda::getCurrentCUDAStream();
    g_unscale_l2norm_cache.ensure(ntensors, chunk_size, total_chunks, tensor_lists, stream);

    auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
    auto output_cu = makeTransformerEngineTensor(g_unscale_l2norm_cache.output_dev);
    auto ret_cu = makeTransformerEngineTensor(g_unscale_l2norm_cache.ret_dev);
    auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);
    nvte_multi_tensor_unscale_l2norm_cuda_custom(
      chunk_size, noop_flag_cu.data(),
      static_cast<NVTEDType>(GetTransformerEngineDType(tensor_lists[0][0].scalar_type())),
      g_unscale_l2norm_cache.addresses_dev.data_ptr<int64_t>(),
      g_unscale_l2norm_cache.sizes_dev.data_ptr<int>(),
      g_unscale_l2norm_cache.block_to_tensor_dev.data_ptr<int>(),
      g_unscale_l2norm_cache.chunk_offsets_dev.data_ptr<int>(),
      total_chunks, output_cu.data(), ret_cu.data(),
      inv_scale_cu.data(), stream);
    auto ret_per_tensor = at::empty({0}, float_options);
    return std::tuple<at::Tensor, at::Tensor>(g_unscale_l2norm_cache.ret_dev, ret_per_tensor);
  }

  // per_tensor path: use multi_tensor_apply
  const int output_size = total_chunks > 320 ? total_chunks : 320;
  auto output = at::zeros({output_size}, float_options);
  auto ret = at::empty({1}, float_options);

  int max_chunks_per_tensor = -1;
  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
    if (max_chunks_this_tensor > max_chunks_per_tensor)
      max_chunks_per_tensor = max_chunks_this_tensor;
  }
  auto output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
  auto ret_per_tensor = at::zeros({ntensors}, float_options);

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_cu = makeTransformerEngineTensor(output);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto ret_per_tensor_cu = makeTransformerEngineTensor(ret_per_tensor);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_unscale_l2norm_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      output_cu.data(), output_per_tensor_cu.data(), ret_cu.data(), ret_per_tensor_cu.data(),
      inv_scale_cu.data(), per_tensor, max_chunks_per_tensor, at::cuda::getCurrentCUDAStream());

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}

}  // namespace transformer_engine::pytorch
