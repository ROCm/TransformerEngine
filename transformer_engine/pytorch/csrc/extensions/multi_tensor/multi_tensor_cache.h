/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <cstdint>
#include <mutex>
#include <vector>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <transformer_engine/transformer_engine.h>

namespace transformer_engine::pytorch {

// Cache for device-side mapping arrays used by custom multi_tensor kernels.
// During training the tensor list structure (shapes and data pointers) is
// typically identical across iterations, so we can avoid per-call device
// allocations and H2D memcpy by caching the arrays and only re-uploading
// when something changes.
//
// kDepth = number of tensor lists (e.g. 4 for [g,p,m,v], 5 for [g,p,m,v,master]).
template <int kDepth>
struct CustomMultiTensorCache {
  std::vector<int64_t> addresses_host;  // [ntensors * kDepth]
  std::vector<int64_t> sizes_host;      // [ntensors]
  at::Tensor addresses_dev;
  at::Tensor sizes_dev;
  at::Tensor block_to_tensor_dev;
  at::Tensor chunk_offsets_dev;
  int total_chunks = 0;
  int chunk_size = 0;
  std::mutex mtx;

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
    std::lock_guard<std::mutex> lock(mtx);
    if (!shapes_valid(ntensors, cs, tensor_lists)) {
      rebuild(ntensors, cs, tc, tensor_lists, stream);
    } else if (!addresses_valid(ntensors, tensor_lists)) {
      update_addresses(ntensors, tensor_lists, stream);
    }
    return total_chunks;
  }
};

// Cache for L2 norm computation (single tensor list, int sizes, includes output/ret buffers).
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
  std::mutex mtx;

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

  int ensure(int ntensors, int cs, int tc,
             const std::vector<std::vector<at::Tensor>> &tensor_lists,
             cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!shapes_valid(ntensors, cs, tensor_lists)) {
      rebuild(ntensors, cs, tc, tensor_lists, stream);
    } else if (!addresses_valid(ntensors, tensor_lists)) {
      update_addresses(ntensors, tensor_lists, stream);
    }
    return total_chunks;
  }
};

}  // namespace transformer_engine::pytorch
