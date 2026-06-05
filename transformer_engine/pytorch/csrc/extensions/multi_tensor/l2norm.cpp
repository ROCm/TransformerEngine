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
