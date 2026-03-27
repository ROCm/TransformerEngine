/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#pragma once

#include <random>
#include <vector>
#include <cmath>
#include <memory>
#include <array>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <benchmark/benchmark.h>

#include <transformer_engine/transformer_engine_hip.h>
#include "test_common_hip.h"

namespace te_bench {

#define HIP_CHECK(call)                                                \
  do {                                                                 \
    hipError_t err = call;                                             \
    if (err != hipSuccess) {                                           \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,  \
              hipGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer(size_t count) : count_(count) {
    HIP_CHECK(hipMalloc(&ptr_, count * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr_) {
      hipError_t err = hipFree(ptr_);
      (void)err;
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }
  size_t count() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

  void upload(const std::vector<T> &host_data) {
    if (host_data.size() != count_) {
      throw std::runtime_error("Size mismatch in upload");
    }
    HIP_CHECK(hipMemcpy(ptr_, host_data.data(), bytes(), hipMemcpyHostToDevice));
  }

  void download(std::vector<T> &host_data) const {
    host_data.resize(count_);
    HIP_CHECK(hipMemcpy(host_data.data(), ptr_, bytes(), hipMemcpyDeviceToHost));
  }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

template <typename T>
std::vector<T> generate_random_data(size_t count, T min_val = -1.0, T max_val = 1.0) {
  std::vector<T> data(count);
  std::mt19937 gen(42);

  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (auto &val : data) {
      val = dist(gen);
    }
  } else {
    std::uniform_int_distribution<int> dist(static_cast<int>(min_val), static_cast<int>(max_val));
    for (auto &val : data) {
      val = static_cast<T>(dist(gen));
    }
  }

  return data;
}

__global__ void scale_shift_kernel(float *data, size_t count, float scale, float offset) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = data[idx] * scale + offset;
  }
}

inline void fill_random_uniform_gpu(float *dptr, size_t count, float min_val = -2.0f, float max_val = 1.0f, hipStream_t stream = 0) {
  hiprandGenerator_t gen;
  hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
  hiprandSetPseudoRandomGeneratorSeed(gen, 42);
  if (stream != 0) {
    hiprandSetStream(gen, stream);
  }
  hiprandGenerateUniform(gen, dptr, count);
  float scale = max_val - min_val;
  float offset = min_val;

  size_t threads = 256;
  size_t blocks = (count + threads - 1) / threads;
  scale_shift_kernel<<<blocks, threads, 0, stream>>>(dptr, count, scale, offset);

  hiprandDestroyGenerator(gen);
}

template<typename T>
__global__ void cast_fp32_kernel(const float *in, T *out, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = static_cast<T>(in[idx]);
  }
}

template<typename T>
inline void fill_random_uniform_gpu_typed(T *dptr, size_t count, float min_val = -2.0f, float max_val = 1.0f, hipStream_t stream = 0) {
  if constexpr (std::is_same_v<T, float>) {
    fill_random_uniform_gpu(dptr, count, min_val, max_val, stream);
  } else {
    DeviceBuffer<float> temp_fp32(count);
    fill_random_uniform_gpu(temp_fp32.get(), count, min_val, max_val, stream);

    size_t threads = 256;
    size_t blocks  = (count + threads - 1) / threads;
    cast_fp32_kernel<<<blocks, threads, 0, stream>>>(temp_fp32.get(), dptr, count);
  }
}

inline void warmup_gpu(int iterations = 10) {
  DeviceBuffer<float> dummy(1024);
  for (int i = 0; i < iterations; ++i) {
    HIP_CHECK(hipMemset(dummy.get(), 0, dummy.bytes()));
  }
  HIP_CHECK(hipDeviceSynchronize());
}

inline double calculate_bandwidth_gbps(size_t bytes, double time_ns) {
  return (bytes / 1e9) / (time_ns / 1e9);
}

inline void set_items_processed(benchmark::State &state, size_t items_per_iter) {
  state.SetItemsProcessed(state.iterations() * items_per_iter);
}

inline void set_bytes_processed(benchmark::State &state, size_t bytes_per_iter) {
  state.SetBytesProcessed(state.iterations() * bytes_per_iter);
}

class TensorCache {
 public:
  struct CacheKey {
    std::string name;
    size_t rows;
    size_t cols;
    transformer_engine::DType dtype;
    bool rowwise;
    bool colwise;
    NVTEScalingMode scaling_mode;

    bool operator<(const CacheKey &other) const {
      return std::tie(name, rows, cols, dtype, rowwise, colwise, scaling_mode) <
             std::tie(other.name, other.rows, other.cols, other.dtype, other.rowwise, other.colwise, other.scaling_mode);
    }
  };

  static test::Tensor &get_or_create(const std::string &name,
                                      const std::vector<size_t> &shape,
                                      transformer_engine::DType dtype,
                                      bool rowwise = true,
                                      bool colwise = false,
                                      NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING,
                                      bool initialize_random = false) {
    CacheKey key{name, shape[0], shape[1], dtype, rowwise, colwise, scaling_mode};

    static auto* cache = new std::map<CacheKey, std::unique_ptr<test::Tensor>>();

    auto it = cache->find(key);
    if (it == cache->end()) {
      auto tensor_ptr = std::make_unique<test::Tensor>(name, shape, dtype, rowwise, colwise, scaling_mode);

      if (initialize_random && dtype != transformer_engine::DType::kFloat8E4M3 &&
          dtype != transformer_engine::DType::kFloat8E5M2) {
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));

        size_t count = shape[0] * shape[1];
        void *data_ptr = tensor_ptr->rowwise_dptr();

        if (dtype == transformer_engine::DType::kFloat32) {
          fill_random_uniform_gpu(static_cast<float*>(data_ptr), count, -2.0f, 1.0f, stream);
        } else if (dtype == transformer_engine::DType::kFloat16) {
          fill_random_uniform_gpu_typed<__half>(static_cast<__half*>(data_ptr), count, -2.0f, 1.0f, stream);
        } else if (dtype == transformer_engine::DType::kBFloat16) {
          fill_random_uniform_gpu_typed<hip_bfloat16>(static_cast<hip_bfloat16*>(data_ptr), count, -2.0f, 1.0f, stream);
        }

        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipStreamDestroy(stream));
      }

      (*cache)[key] = std::move(tensor_ptr);
      it = cache->find(key);
    }

    return *(it->second);
  }
};
} // namespace te_bench
