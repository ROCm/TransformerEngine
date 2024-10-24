/*************************************************************************
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include <type_traits>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>
#ifdef USE_HIPBLASLT
#include <vector>
#include <forward_list>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <chrono>
#include <optional>
#include <hipblaslt/hipblaslt.h>
#endif
#ifdef USE_ROCBLAS
#define ROCBLAS_BETA_FEATURES_API 
#include <rocblas/rocblas.h>
#include <hipcub/hipcub.hpp>
#endif
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdint>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../util/logging.h"

namespace {

#ifdef USE_HIPBLASLT

#if HIP_VERSION >= 60000000
typedef hipDataType hipblasltDatatype_t;
typedef hipblasComputeType_t hipblasLtComputeType_t;
#define HIPBLASLT_R_16F HIP_R_16F
#define HIPBLASLT_R_32F HIP_R_32F
#define HIPBLASLT_R_16B HIP_R_16BF
#define HIPBLASLT_R_8F_E4M3 HIP_R_8F_E4M3_FNUZ
#define HIPBLASLT_R_8F_E5M2 HIP_R_8F_E5M2_FNUZ
#define HIPBLASLT_COMPUTE_F32 HIPBLAS_COMPUTE_32F
#endif // #if HIP_VERSION >= 60000000

hipblasltDatatype_t get_hipblaslt_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return HIPBLASLT_R_16F;
    case DType::kFloat32:
      return HIPBLASLT_R_32F;
    case DType::kBFloat16:
      return HIPBLASLT_R_16B;
    case DType::kFloat8E4M3:
      return HIPBLASLT_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return HIPBLASLT_R_8F_E5M2;
    default:
      NVTE_ERROR("Invalid type");
  }
}
#endif

#ifdef USE_ROCBLAS
rocblas_datatype get_rocblas_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return rocblas_datatype_f16_r;
    case DType::kFloat32:
      return rocblas_datatype_f32_r;
    case DType::kBFloat16:
      return rocblas_datatype_bf16_r;
    case DType::kFloat8E4M3:
      return rocblas_datatype_f8_r;
    case DType::kFloat8E5M2:
      return rocblas_datatype_bf8_r;
    default:
      NVTE_ERROR("Invalid type");
  }
}
#endif

} //namespace

namespace transformer_engine {

#ifdef USE_ROCBLAS

namespace detail {

struct Empty {};

__device__ inline fp32 identity(fp32 value, const Empty&) {
  return value;
}

__inline__ __device__
float gelu(float x, const Empty&)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}


__inline__ __device__
float gelu_forward(float x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}


template <typename T, int THREADS_PER_BLOCK>
__global__
void gelu_forward_kernel(const float* in, T* out, float* amax, const float* scale, int m, int n) {
  // fp8 output flow
  if constexpr(std::is_same<T, fp8e4m3>::value ||std::is_same<T, fp8e5m2>::value){
    typedef hipcub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage block_temp_storage;
    float thread_amax = 0;
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x){
      float x = in[id];
      float y = gelu_forward(x); 
      out[id] = (T)((*scale)*y);
      thread_amax=std::fmax(std::fabs(y), thread_amax);
    }
    float block_amax = BlockReduce(block_temp_storage).Reduce(thread_amax, hipcub::Max());
    if(threadIdx.x==0){
      atomicMaxFloat(amax, block_amax);
    }
  }else{
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x){
      float x = in[id];
      float y = gelu_forward(x); 
      out[id] = (T)(y);
    }
  }
}


template <typename T>
void gelu_forward_kernelLauncher(const float* in, T* out, float* amax, const float* scale, int m, int n, hipStream_t stream) {
  dim3 block, grid;
  constexpr int THREADS_PER_BLOCK = 1024;
  block.x = THREADS_PER_BLOCK;
  grid.x = ceil(1.0*m * n / THREADS_PER_BLOCK);
  hipLaunchKernelGGL(( gelu_forward_kernel<T, THREADS_PER_BLOCK>), dim3(grid), dim3(block), 0, stream, in, out, amax, scale, m, n);
}


__inline__ __device__
float gelu_backward(float x, float dy){
  constexpr float kBeta = 0.7978845608028654f; 
  constexpr float kKappa = 0.044715f;
  float x_sq = x * x;
  float x_cube = x_sq * x;
  float tanh_inner = tanhf((kBeta * (x + kKappa * x_cube)));

  float left = 0.5 * x;
  float right = 1.0f + tanh_inner;

  float left_derivative = 0.5 * right;

  float tanh_derivative = 1 - tanh_inner * tanh_inner;
  float inner_derivative = kBeta * (1.0f + 3.0 * kKappa * x_sq);
  float right_derivative = left * tanh_derivative * inner_derivative;

  return dy * (left_derivative + right_derivative);
}

template <typename T, typename Taux>
__global__ 
void gelu_backward_kernel(const float* dy, T* out, const Taux* __restrict pre_gelu_out, int m, int n) {
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    float x = (float)pre_gelu_out[id];
    float dx = (float)gelu_backward(x, dy[id]); 
    out[id] = (T)(dx);
  }
}

template <typename T, typename Taux>
void gelu_backward_kernelLauncher(const float* in, T* out, const Taux* pre_gelu_out, int m, int n, hipStream_t stream) {
  int blocks_per_row = ceil(float(n)/1024);
  dim3 grid(min(m * blocks_per_row, 65536));
  dim3 block(min(n, 1024));
  hipLaunchKernelGGL(( gelu_backward_kernel<T, Taux>), dim3(grid), dim3(block), 0, stream, in, out, pre_gelu_out, m, n);
}

template <typename T, typename Tb, int THREADS_PER_BLOCK>
__global__ 
void add_bias_kernel(const float* in, T* out, const Tb* __restrict bias, float* amax, const float* scale, int m, int n){
  // fp8 output flow
  if constexpr(std::is_same<T, fp8e4m3>::value ||std::is_same<T, fp8e5m2>::value){
    typedef hipcub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage block_temp_storage;
    float thread_amax = 0;
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x){
      float reg_bias = (float)bias[id % n];
      float val = in[id] + reg_bias;
      out[id] = (T)((*scale)*val);
      // deal with amax of D
      thread_amax=std::fmax(std::fabs(val), thread_amax);
    }
    // num_valid can be ignored since each thread amax is set to 0
    float block_amax = BlockReduce(block_temp_storage).Reduce(thread_amax, hipcub::Max());
    if(threadIdx.x==0){
      atomicMaxFloat(amax, block_amax);
    }
  }else{
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x){
      float reg_bias = (float)bias[id % n];
      float val = in[id] + reg_bias;
      out[id] = (T)(val);
    }
  }
}


template <typename T, typename Tb>
void add_bias_kernelLauncher(const float* in, T* out, const Tb* __restrict bias, float* amax, const float* scale, int m, int n, hipStream_t stream) {
  dim3 block, grid;
  constexpr int THREADS_PER_BLOCK = 1024;
  block.x = THREADS_PER_BLOCK;
  grid.x = ceil(1.0*m * n / THREADS_PER_BLOCK);
  hipLaunchKernelGGL(( add_bias_kernel<T, Tb, THREADS_PER_BLOCK>), dim3(grid), dim3(block), 0, stream, in, out, bias, amax, scale, m, n);

}

template <typename T, typename Taux, typename Tb, int THREADS_PER_BLOCK>
__global__ 
void add_bias_gelu_kernel(const float* in, T* out, Taux* pre_gelu_out, const Tb* __restrict bias, float* amax, const float* scale, int m, int n){
  // fp8 output flow
  if constexpr(std::is_same<T, fp8e4m3>::value ||std::is_same<T, fp8e5m2>::value){
    // only need to deal with amax and scale of D, no need to deal with amax and scale of pre_gelu_out
    typedef hipcub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage block_temp_storage;
    float thread_amax = 0;
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x){
      float reg_bias = (float)bias[id % n];
      float val = in[id] + reg_bias;
      // pre_gelu_out guaranteed not to be fp8 type
      pre_gelu_out[id] = (Taux)(val);
      val = gelu_forward(val);
      out[id] = (T)((*scale)*val);
      // deal with amax of D
      thread_amax=std::fmax(std::fabs(val), thread_amax);
    }
    // num_valid can be ignored since each thread amax is set to 0
    float block_amax = BlockReduce(block_temp_storage).Reduce(thread_amax, hipcub::Max());
    if(threadIdx.x==0){
      atomicMaxFloat(amax, block_amax);
    }
  }else{
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x){
      float reg_bias = (float)bias[id % n];
      float val = in[id] + reg_bias;
      pre_gelu_out[id] = (Taux)(val);
      out[id] = (T)(gelu_forward(val));
    }
  }
}

template <typename T, typename Taux, typename Tb>
void add_bias_gelu_kernelLauncher(const float* in, T* out, Taux* pre_gelu_out, const Tb* __restrict bias, float* amax, const float* scale, int m, int n, hipStream_t stream) {
  dim3 block, grid;
  constexpr int THREADS_PER_BLOCK = 1024;
  block.x = THREADS_PER_BLOCK;
  grid.x = ceil(1.0*m * n / THREADS_PER_BLOCK);
  hipLaunchKernelGGL(( add_bias_gelu_kernel<T, Taux, Tb, THREADS_PER_BLOCK>), dim3(grid), dim3(block), 0, stream, in, out, pre_gelu_out, bias, amax, scale, m, n );

}

template <typename Tin, typename T>
__global__ 
void identity_kernel(const Tin* in, T* out, int n) {
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += blockDim.x * gridDim.x)
  {
    Tin val = in[id];
    out[id] = (T)(val);
  }
}


template <typename Tin, typename T>
void identity_kernelLauncher(const Tin* in, T* out, int n, hipStream_t stream) {
  dim3 block, grid;
  block.x = 1024;
  grid.x = ceil( n / 1024.);
  hipLaunchKernelGGL(( identity_kernel<Tin, T>), dim3(grid), dim3(block), 0, stream, in, out, n );
}

template <typename T, int THREADS_PER_BLOCK>
__global__ 
void identity_output_kernel(const float* in, T* out, float* amax, const float* scale, int n) {
  if constexpr(std::is_same<T, fp8e4m3>::value ||std::is_same<T, fp8e5m2>::value){
    typedef hipcub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage block_temp_storage;
    float thread_amax = 0;
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += blockDim.x * gridDim.x){
      float val = in[id];
      out[id] = (T)((*scale)*val);
      // deal with amax of D
      thread_amax=std::fmax(std::fabs(val), thread_amax);
    }
    // num_valid can be ignored since each thread amax is set to 0
    float block_amax = BlockReduce(block_temp_storage).Reduce(thread_amax, hipcub::Max());
    if(threadIdx.x==0){
      atomicMaxFloat(amax, block_amax);
    }
  }else{
    for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += blockDim.x * gridDim.x){
      float val = in[id];
      out[id] = (T)(val);
    }
  }
}


template <typename T>
void identity_output_kernelLauncher(const float* in, T* out, float* amax, const float* scale, int n, hipStream_t stream) {
  dim3 block, grid;
  constexpr int THREADS_PER_BLOCK = 1024;
  block.x = THREADS_PER_BLOCK;
  grid.x = ceil( 1.0*n / THREADS_PER_BLOCK);
  hipLaunchKernelGGL(( identity_output_kernel<T, THREADS_PER_BLOCK>), dim3(grid), dim3(block), 0, stream, in, out, amax, scale, n );
}

template <typename Tin, int THREADS_PER_BLOCK>
__global__
void bias_gradient_kernel(const Tin* in, float* out, int m, int n) {
  typedef hipcub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;

  int BLOCKS_PER_COL = ceil(float(m)/THREADS_PER_BLOCK);
  int THREADS_PER_COL = BLOCKS_PER_COL * THREADS_PER_BLOCK;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int col_idx = idx / THREADS_PER_COL;
  int row_idx = idx % THREADS_PER_COL;
  float thread_data;
  if (row_idx < m)
    thread_data = (float)in[row_idx * n + col_idx];
  float local_sum;
  if (row_idx < (BLOCKS_PER_COL-1) * THREADS_PER_BLOCK) {
    local_sum = BlockReduce(block_temp_storage).Sum(thread_data);
  }
  else {
    local_sum = BlockReduce(block_temp_storage).Sum(thread_data, m-(BLOCKS_PER_COL-1)*THREADS_PER_BLOCK);
  }
  if (threadIdx.x == 0)
    atomicAdd(&out[col_idx], local_sum);
}

template <typename Tin>
void bias_gradient_kernelLauncher(const Tin* in, float* out, int m, int n, bool stream_order_alloc, hipStream_t stream) { 
  dim3 block, grid;
  constexpr int THREADS_PER_BLOCK = 1024;
  int BLOCKS_PER_COL = ceil(float(m)/THREADS_PER_BLOCK);
  block.x = THREADS_PER_BLOCK;
  grid.x = BLOCKS_PER_COL*n;
  if(! stream_order_alloc){
    NVTE_CHECK_CUDA( hipMemset(out, 0, n*sizeof(float)) );
  }else{
#if HIP_VERSION >= 50300000
    NVTE_CHECK_CUDA( hipMemsetAsync(out, 0, n*sizeof(float), stream) );
#else
    NVTE_ERROR("Stream order allocation is supported on ROCm 5.3 and above.");
#endif
  }
  hipLaunchKernelGGL(( bias_gradient_kernel<Tin, THREADS_PER_BLOCK>), dim3(grid), dim3(block), 0, stream, in, out, m, n);
}

} // namespace detail

transformer_engine::DType get_transformer_engine_dtype(const rocblas_datatype t) {
  using namespace transformer_engine;
  switch (t) {
    case rocblas_datatype_f16_r:
      return DType::kFloat16;
    case rocblas_datatype_f32_r:
      return DType::kFloat32;
    case rocblas_datatype_bf16_r:
      return DType::kBFloat16;
    case rocblas_datatype_f8_r:
      return DType::kFloat8E4M3;
    case rocblas_datatype_bf8_r:
      return DType::kFloat8E5M2;
    default:
      NVTE_ERROR("Invalid type");
  }
}
#endif //USE_ROCBLAS

#ifdef USE_HIPBLASLT

namespace {

static class HandlePool {
public:
  hipblasLtHandle_t get(int device_id) 
  {
    std::lock_guard<std::mutex> lock(mt);

    if (pool.empty())
    {
      int device_count = 0; 
      NVTE_CHECK_CUDA(hipGetDeviceCount(&device_count));
      pool.resize(device_count);
      return nullptr;
    }

    if (!pool[device_id].empty())
    {
      hipblasLtHandle_t h = pool[device_id].front();
      pool[device_id].pop_front();
      return h;
    }

    return nullptr;
  }

  hipblasLtHandle_t obtain(int device_id) 
  {
    hipblasLtHandle_t h = get(device_id);
    if (h == nullptr)
    {
      NVTE_CHECK_HIPBLASLT(hipblasLtCreate(&h));
    }
    return h;
  }

  void store(const std::vector<hipblasLtHandle_t>& handles)
  {
    std::lock_guard<std::mutex> lock(mt);
    if (pool.empty())
    {
      std::cout << "[ERROR] Attempt to store handles to invalid pool" << std::endl;
    }
    for (unsigned int i=0; i<pool.size(); i++)
    {
      if (handles[i] != nullptr)
      {
        pool[i].push_front(handles[i]);
      }
    }
  }

  ~HandlePool() {
#if DESTROY_HIPBLASLT_HANDLES_POOL
    std::lock_guard<std::mutex> lock(mt);
    for (auto & hlist : pool)
    {
      for (auto & h : hlist)
      {
        hipblasLtDestroy(h);
      }
    }
    pool.clear();
#endif
  }

  inline size_t get_size() const
  {
    return pool.size();
  }

private:
  std::mutex mt;
  using Pool = std::vector<std::forward_list<hipblasLtHandle_t>>;
  // Order of destructors between thread_local and global is not actually guaranteed
  // As a simple w/a make pool storage "leaky"
  // Just do not destruct it and do not destroy hipbladLt handles
  // Let OS deal with it on application exit
#if DESTROY_HIPBLASLT_HANDLES_POOL
  Pool pool;
#else
  Pool &pool = *new Pool();
#endif
} handle_pool;


thread_local static class HandleCache {
public:
  hipblasLtHandle_t get(int device_id) const
  {
    return d.empty() ? nullptr : d[device_id];
  }

  hipblasLtHandle_t obtain(int device_id)
  {
    hipblasLtHandle_t h = get(device_id);
    if (h)
    {
      return h;
    }
    h = handle_pool.obtain(device_id);
    set(device_id, h);
    return h;
  }

  void set(int device_id, hipblasLtHandle_t h) 
  { 
    if (d.empty())
    {
      d.resize(handle_pool.get_size());
    }
    d[device_id] = h;
  }

  ~HandleCache()
  {
    if (!d.empty())
    {
      handle_pool.store(d);
    }
  }

private:
  std::vector<hipblasLtHandle_t> d;
} cached_handles;


class csv_helper
{
public:
  struct start {};
  struct end {};

  csv_helper(std::ostream& os, char sep_val) : m_os{ os }, m_sep_val(sep_val), m_start(true), m_sep("") {}

  csv_helper& operator << (const start&)
  {
    m_start = true;
    return *this;
  }

  csv_helper& operator << (const end&)
  {
    m_sep="";
    m_start = false;
    return *this;
  }

  template< typename T>
  csv_helper& operator<<(const T& v)
  {
    m_os << m_sep << v;
    if (m_start)
    {
      m_start = false;
      m_sep = m_sep_val;
    }
    return *this;
  }

private:
  std::ostream& m_os;
  char m_sep_val;
  bool m_start;
  std::string m_sep;
};


template<typename T>
class NameMapper
{
public:
  NameMapper(const std::unordered_map<T, std::string_view>& name_map): map(name_map) {}
  const std::string_view &getName(const T &val) {
    return map.at(val);
  }
  T getValue(const std::string& name, const char *label="")
  {
    for (auto iter = map.begin(); iter != map.end(); ++iter)
    {
        if (name == iter->second) return iter->first;
    }
    NVTE_ERROR("Invalid ", label, " name: ", name);
  }
protected: 
  const std::unordered_map<T, std::string_view> &map;
};

static std::unordered_map<hipblasltDatatype_t, std::string_view> type_name_map = {
  {HIPBLASLT_R_32F, "float32"},
  {HIPBLASLT_R_16F, "float16"},
  {HIPBLASLT_R_16B, "bfloat16"},
  {HIPBLASLT_R_8F_E4M3, "float8e4m3"},
  {HIPBLASLT_R_8F_E5M2, "float8e5m2"},
};
static NameMapper<hipblasltDatatype_t> typeNameMapper(type_name_map);

static std::unordered_map<hipblasOperation_t, std::string_view> trans_name_map = {
  {HIPBLAS_OP_N, "N"},
  {HIPBLAS_OP_T, "T"}
};
static NameMapper<hipblasOperation_t> transposeNameMapper(trans_name_map);

static std::unordered_map<hipblasLtEpilogue_t, std::string_view> epi_name_map = {
  {HIPBLASLT_EPILOGUE_DEFAULT, "-"},
  {HIPBLASLT_EPILOGUE_BIAS, "bias"},
  {HIPBLASLT_EPILOGUE_GELU_AUX, "geluaux"},
  {HIPBLASLT_EPILOGUE_GELU_AUX_BIAS, "geluauxbias"},
  {HIPBLASLT_EPILOGUE_DGELU, "dgelu"},
  {HIPBLASLT_EPILOGUE_DGELU_BGRAD, "dgelubgrad"},
  {HIPBLASLT_EPILOGUE_BGRADB, "bgradb"}
};
static NameMapper<hipblasLtEpilogue_t> epilogueNameMapper(epi_name_map);

static std::unordered_map<hipblasLtComputeType_t, std::string_view> comp_name_map = {
  {HIPBLASLT_COMPUTE_F32, "f32"}
};
static NameMapper<hipblasLtComputeType_t> computeNameMapper(comp_name_map);

static class GemmAlgoCache {
public:
  struct Key {
    int deviceCap;
    hipblasltDatatype_t a_type, b_type, d_type, bias_type;
    int m, n, k;
    int lda, ldb, ldd;
    hipblasOperation_t transa, transb;
    hipblasLtEpilogue_t epilogue;

    Key(int deviceCap_,
        hipblasltDatatype_t a_type_, hipblasltDatatype_t b_type_,
        hipblasltDatatype_t d_type_, hipblasltDatatype_t bias_type_,
        int m_, int n_, int k_, int lda_, int ldb_, int ldd_,
        hipblasOperation_t transa_, hipblasOperation_t transb_,
        hipblasLtEpilogue_t epilogue_):
        deviceCap(deviceCap_),
        a_type(a_type_), b_type(b_type_),
        d_type(d_type_), bias_type(bias_type_),
        m(m_), n(n_), k(k_), lda(lda_), ldb(ldb_), ldd(ldd_),
        transa(transa_), transb(transb_),
        epilogue(epilogue_) {}

    Key() {}

    bool operator==(const Key &val) const
    {
      return ((deviceCap == val.deviceCap)
      && (a_type == val.a_type) && (b_type == val.b_type)
      && (d_type == val.d_type) && (bias_type == val.bias_type)
      && (m == val.m) && (n == val.n) && (k == val.k)
      && (lda == val.lda) && (ldb == val.ldb) && (ldd == val.ldd)
      && (transa == val.transa) && (transb == val.transb)
      && (epilogue == val.epilogue) );
    }

    struct Comp
    {
      bool operator()(const Key& lhs, const Key& rhs) const
      {
        return ::std::string_view((const char*)&lhs, sizeof(lhs)) < ::std::string_view((const char*)&rhs, sizeof(rhs));
      }
    };
  };

  void init()
  {
    std::lock_guard<std::mutex> lock(mt);
    int device_count = 0; 
    NVTE_CHECK_CUDA(hipGetDeviceCount(&device_count));
    dev_cap.resize(device_count);
    for (int i=0; i<device_count; i++)
    {
      hipDeviceProp_t prop;
      NVTE_CHECK_CUDA(hipGetDeviceProperties(&prop, i));
      dev_cap[i] = prop.major*100 + prop.minor;
    }
    load_();
    save_();
  }

  inline int device_cap(int device_id)
  {
    if (dev_cap.empty())
      init();
    return dev_cap[device_id];
  }

  struct Algo {
    std::optional<hipblasLtMatmulAlgo_t> algo;
    int64_t algoId;
    int index;
    size_t ws_size_min;
    size_t ws_size_max;
    Algo(): algo(), index(-1), algoId(), ws_size_min(0), ws_size_max(0) {}
    Algo(int idx, int64_t id, size_t ws_min, size_t ws_max): algo(), index(idx), algoId(id), ws_size_min(ws_min), ws_size_max(ws_max) {}
    inline bool hasId() { return index>=0; } const
    static inline int64_t getAlgoId(const hipblasLtMatmulAlgo_t &algo)
    {
      return *(const int64_t*)&algo;
    }
  };

  bool find(const Key &cfg, size_t ws_size, Algo &algo)
  {
    std::lock_guard<std::mutex> lock(mt);
    if (auto *pentry = find_(cfg, ws_size, ws_size); pentry != nullptr)
    {
      algo = *pentry;
      return true;
    }
    return false;
  }

  void store(const Key &cfg, const Algo &algo)
  {
    size_t ws_size_min = algo.ws_size_min;
    size_t ws_size_max = algo.ws_size_max;
    NVTE_CHECK(ws_size_max >= ws_size_min, "Invalid WS size");
    std::lock_guard<std::mutex> lock(mt);

    //Remove overlapping with existing entries;
    while (auto* pentry = find_(cfg, ws_size_min, ws_size_max)) {
      if (pentry->ws_size_min <= ws_size_min && pentry->ws_size_max >= ws_size_max)
      {
        *pentry = algo;
        save_();
        return;
      }

      if (ws_size_max > pentry->ws_size_max)
      {
        ws_size_min = pentry->ws_size_max + 1;
      }
      else if (ws_size_min < pentry->ws_size_min)
      {
        ws_size_max = pentry->ws_size_min - 1;
      }
      else
      {
        //Should never be here
        NVTE_ERROR("Cannot merge WS size range");
      }
    }

    //Merge to adjusted entry if possible
    auto* pentry = find_(cfg, ws_size_min - 1, ws_size_min);
    if (pentry && pentry->algoId == algo.algoId)
    {
      pentry->algo = algo.algo;
      pentry->ws_size_max = ws_size_max;
      save_();
    }
    else
    {
      auto it = d.emplace(cfg, algo);
      it->second.ws_size_min = ws_size_min;
      it->second.ws_size_max = ws_size_max;
      save_(it->first, it->second);
    }
  }

protected:

  Algo* find_(const Key &cfg, size_t ws_min, size_t ws_max)
  {
    const auto key_range = d.equal_range(cfg);
    for (auto i = key_range.first; i != key_range.second; i++)
    {
      if (ws_min <= i->second.ws_size_max && ws_max >= i->second.ws_size_min)
      {
        return &i->second;
      }
    }
    return nullptr;
  }

  void header_(std::ostream& ofs)
  {
    csv_helper fs(ofs, csv_sep);
    fs << "dev_cap" << "m" << "n"  << "k" << "trans_a" << "trans_b" 
    << "type_a" << "type_b" << "type_d" << "bias_type" 
    << "lda" << "ldb" << "ldd" << "epi" << "comp" << "scale"
    << "ws_min" << "ws_max" << "algo_id" << "aidx";
  }
  
  void load_()
  {
    const char* env = std::getenv("TE_HIPBLASLT_ALGO_LOAD");
    if (env == nullptr || env[0] == '\0')
    {
      return;
    }
    std::ifstream ifs{env};
    if (!ifs.is_open())
    {
      std::cerr << "Could not load autotune results storage " << env << "\n";
      return;
    }
    std::cout << "Loading autotune results from " << env << "\n";

    Key cfg;
    std::string line;
    std::getline(ifs, line); // the first line with legend
    {
      std::ostringstream hline;
      header_(hline);
      if (hline.str() != line) {
        std::cerr << "Incorrect algo storage legend. Expected " << hline.str() << "\n";
        return;
      }
    }

    while(std::getline(ifs, line)) 
    {
      line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
      if (auto pos = line.find_last_not_of(" \t\n\r\f\v"); pos != std::string::npos)
      {
        line.resize(pos+1);
      }
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      char c;
      std::string type_a, type_b, type_d, bias_type, trans_a, trans_b, epi, comp, scale;
      int64_t algo_id;
      int algo_idx;
      size_t ws_min, ws_max;

      is >> std::skipws;
      is >> cfg.deviceCap >> c >> cfg.m >> c >> cfg.n >> c >> cfg.k >> c;

      //Filter out entries for devices not presented on the curent system
      bool b_found = false;
      for (int i=0; i<dev_cap.size(); i++)
      {
        if (dev_cap[i] == cfg.deviceCap)
        {
          b_found = true;
          break;
        }
      }
      if (!b_found) continue;

      std::getline(is, trans_a, csv_sep);
      std::getline(is, trans_b, csv_sep);
      std::getline(is, type_a, csv_sep);
      std::getline(is, type_b, csv_sep);
      std::getline(is, type_d, csv_sep);
      std::getline(is, bias_type, csv_sep);
      is >> cfg.lda >> c >> cfg.ldb >> c >> cfg.ldd >> c;
      std::getline(is, epi, csv_sep);
      std::getline(is, comp, csv_sep);
      std::getline(is, scale, csv_sep);
      is >> ws_min >> c >> ws_max >> c >> algo_id >> c >> algo_idx;
  
      if (is.bad())
      {
        std::cerr << "Parsing CSV line failed: " << line << "\n";
        return;
      }

      if (ws_min > ws_max)
      {
        std::cout << "[WARNING] Invalid WS size at " << line << "\n";
        continue;
      }
  
      cfg.a_type = typeNameMapper.getValue(type_a, "type_a");
      cfg.b_type = typeNameMapper.getValue(type_b, "type_b");
      cfg.d_type = typeNameMapper.getValue(type_d, "type_d");
      cfg.bias_type = (bias_type == "-") ? (hipblasltDatatype_t)-1 : typeNameMapper.getValue(bias_type, "bias_type");

      cfg.transa = transposeNameMapper.getValue(trans_a, "trans_a");
      cfg.transb = transposeNameMapper.getValue(trans_b, "trans_b");

      cfg.epilogue = epilogueNameMapper.getValue(epi, "epi");
      //Check and filter out compute and scale types
      if (computeNameMapper.getValue(comp, "comp") != HIPBLASLT_COMPUTE_F32 || typeNameMapper.getValue(scale, "scale") != HIPBLASLT_R_32F)
      {
        continue;
      }

      if (find_(cfg, ws_min, ws_max))
      {
          std::cout << "[WARNING] Duplicated/overlapped entry in algo cache\n";
          continue;
      }

      d.emplace(cfg, Algo(algo_idx, algo_id, ws_min, ws_max));
    }
  }

  bool can_save_(bool reopen = false)
  {
    if (!save_fs)
    {
        save_fs_name = std::getenv("TE_HIPBLASLT_ALGO_SAVE");
        if (save_fs_name == nullptr || save_fs_name[0] == '\0')
        {
          return false;
        }
        save_fs = std::make_unique<std::ofstream>();
        std::cout << "Saving autotune results to " << save_fs_name << "\n";
    }

    if (reopen)
    {
      if (save_fs->is_open())
      {
        save_fs->close();
      }
      save_fs->open(save_fs_name, std::ios_base::trunc);
    }

    if (save_fs->is_open() && !save_fs->bad())
    {
      return true;
    }
    else
    {
      if (reopen) std::cerr << "Could not open autotune results storage " << save_fs_name << "\n";
      return false;
    }
  }

  void save_()
  {
    if (!can_save_(true))
    {
      return;
    }
    header_(*save_fs);
    *save_fs << "\n";

    for (const auto &elem: d)
    {
      save_(elem.first, elem.second);
    }
  }

  void save_(const Key &cfg, const Algo &algo)
  {
    if (!can_save_())
    {
      return;
    }
    csv_helper csv(*save_fs, csv_sep);
    csv << cfg.deviceCap << cfg.m << cfg.n << cfg.k 
      << transposeNameMapper.getName(cfg.transa) << transposeNameMapper.getName(cfg.transb)
      << typeNameMapper.getName(cfg.a_type) << typeNameMapper.getName(cfg.b_type) << typeNameMapper.getName(cfg.d_type)
      << ((cfg.bias_type == (hipblasltDatatype_t)-1) ? "-" : typeNameMapper.getName(cfg.bias_type))
      << cfg.lda << cfg.ldb << cfg.ldd << epilogueNameMapper.getName(cfg.epilogue)
      << computeNameMapper.getName(HIPBLASLT_COMPUTE_F32) << typeNameMapper.getName(HIPBLASLT_R_32F)
      << algo.ws_size_min << algo.ws_size_max << algo.algoId << algo.index << csv_helper::end() << "\n";
  }

private:
  std::vector<int> dev_cap;
  constexpr static char csv_sep = ','; 
  std::unique_ptr<std::ofstream> save_fs;
  const char *save_fs_name;
  std::mutex mt;
  /* Map of problem config to tuple of ws_size and Algo
   * When searching, elements matching Key are filtered 
   * for requested WS size be between Algo.ws_size and pair.first
   */
  std::multimap<Key, Algo, Key::Comp> d;
} algoCache;

static inline int getIntEnv(const char *name, int defval, int minval)
{
  int val = defval;
  const char* env = std::getenv(name);
  if (env != nullptr && env[0] != '\0')
  {
     val = atoi(env);
     if (val < minval)
     {
        val = minval;
     }
  }
  return val;
}

} //namespace

void hipblaslt_gemm(const Tensor *inputA,
                 const Tensor *inputB,
                 Tensor *outputD,
                 const Tensor *inputBias,
                 Tensor *outputPreGelu,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 hipblasOperation_t transa,
                 hipblasOperation_t transb,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool accumulate,
                 bool use_split_accumulator,
                 int math_sm_count,
                 int m_split,
                 int n_split,
                 bool gemm_producer,
                 const Tensor *inputCounter,
                 hipStream_t stream
) {
  void *A = inputA->data.dptr;
  void *A_scale_inverse = inputA->scale_inv.dptr;
  void *B = inputB->data.dptr;
  void *B_scale_inverse = inputB->scale_inv.dptr;
  void *D = outputD->data.dptr;
  void *bias_ptr = inputBias->data.dptr;
  const bool bias = bias_ptr != nullptr;
  void *pre_gelu_out = outputPreGelu->data.dptr;
  const bool gelu = pre_gelu_out != nullptr;
  const bool use_fp8 = is_fp8_dtype(inputA->data.dtype) ||
                       is_fp8_dtype(inputB->data.dtype);
  const hipblasltDatatype_t A_type = get_hipblaslt_dtype(inputA->data.dtype);
  const hipblasltDatatype_t B_type = get_hipblaslt_dtype(inputB->data.dtype);
  const hipblasltDatatype_t D_type = get_hipblaslt_dtype(outputD->data.dtype);
  const hipblasltDatatype_t bias_type = get_hipblaslt_dtype(inputBias->data.dtype);

  NVTE_CHECK(!is_fp8_dtype(inputA->data.dtype) || A_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");
  NVTE_CHECK(!is_fp8_dtype(inputB->data.dtype) || B_scale_inverse != nullptr,
             "FP8 input to GEMM requires inverse of scale!");

  // check consistency of arguments:
  // if fp8 is desired, context cannot be null
  // fp8 + gelu fusion + fp8 aux is unavailable right now.
  if (use_fp8) {
    NVTE_CHECK(!gelu, "fp8 gemm + gelu fusion is unavailable right now!");
  }
  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  int device_id;
  NVTE_CHECK_CUDA(hipGetDevice(&device_id));

  hipblasLtHandle_t handle = cached_handles.get(device_id);
  if (handle == nullptr)
  {
    handle = cached_handles.obtain(device_id);
  }

  hipblasLtMatmulDesc_t       operationDesc = nullptr;
  hipblasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t) ldd;

  // default to tf32 except for e5m2 inputs where the config is not supported
  hipblasLtComputeType_t gemm_compute_type = HIPBLASLT_COMPUTE_F32;

  // Create matrix descriptors. Not setting any extra attributes.
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&Adesc, A_type,
                                               transa == HIPBLAS_OP_N ? m : k,
                                               transa == HIPBLAS_OP_N ? k : m,
                                               lda));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&Bdesc, B_type,
                                               transb == HIPBLAS_OP_N ? k : n,
                                               transb == HIPBLAS_OP_N ? n : k,
                                               ldb));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&operationDesc, gemm_compute_type, HIPBLASLT_R_32F));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                   &transa, sizeof(transa)));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                   &transb, sizeof(transb)));

  // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
  // Note: gelu fusion isn't available right now, and we don't need
  // amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    /*
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_FAST_ACCUM, //TODO: We don't have fast accum mode yet
                                                     &fastAccuMode,
                                                     sizeof(fastAccuMode)));
    */
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                     &A_scale_inverse,
                                                     sizeof(A_scale_inverse)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                     &B_scale_inverse,
                                                     sizeof(B_scale_inverse)));
    if (bias) {
      NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                       HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                       &bias_type, sizeof(bias_type)));
    }
  }

  if (bias && gelu) {
    if (grad) {
      epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias_ptr, sizeof(bias_ptr)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
                            operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                      &ld_gelumat, sizeof(ld_gelumat)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = HIPBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_BIAS;
    }
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias_ptr, sizeof(bias_ptr)));
  } else if (gelu) {
    if (grad) {
      epilogue = HIPBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;
    }
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
                            operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                     &ld_gelumat, sizeof(ld_gelumat)));
  }

  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                   HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue)));

  GemmAlgoCache::Key gemm_cfg(algoCache.device_cap(device_id), A_type, B_type, D_type, 
    use_fp8 ? bias_type : (hipblasltDatatype_t)-1,
    m, n, k, lda, ldb, ldd, transa, transb, epilogue );
  GemmAlgoCache::Algo cached_algo;
  if (algoCache.find(gemm_cfg, workspaceSize, cached_algo) == 0 || !cached_algo.algo.has_value())
  {
    int firstAlgo = getIntEnv("TE_HIPBLASLT_ALGO_SELECTION", 0, 0);
    int tuneLoopCount = getIntEnv("TE_HIPBLASLT_TUNING_RUN_COUNT", 0, 0);
    int algoTuneCount = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> algoArr;

    if (tuneLoopCount)
    {
      /* HIPBLASLT may return hundreds of algos for some configs
       * Limit amount by default. User may override with env
       */
      static const int defaultAlgoCount = 16;
      algoTuneCount = getIntEnv("TE_HIPBLASLT_TUNING_ALGO_COUNT", defaultAlgoCount, 1);
    }
    algoTuneCount += firstAlgo;
    int algoTotalCount = cached_algo.hasId() ? std::max(algoTuneCount, (cached_algo.index + 1)) : algoTuneCount;
    algoArr.resize(algoTotalCount);

    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
                            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            &workspaceSize, sizeof(workspaceSize)));

    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Ddesc,
                                                    Ddesc, preference, algoTotalCount, algoArr.data(),
                                                    &algoTotalCount));
    algoArr.resize(algoTotalCount);

    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceDestroy(preference));

    //If cached algo exists in persistent storage we just need to find matching hipblasLtMatmulAlgo_t
    if (cached_algo.hasId())
    {
      int idx = (cached_algo.index < algoTotalCount) ? cached_algo.index : 0;
      for (int i=0; i<algoTotalCount; i++)
      {
        const auto &algo = algoArr[idx];
        if (algo.state == HIPBLAS_STATUS_SUCCESS)
        {
          if (cached_algo.algoId == cached_algo.getAlgoId(algo.algo))
          {
            cached_algo.algo = algo.algo;
            if (algo.workspaceSize != cached_algo.ws_size_min || idx != cached_algo.index)
            {
              cached_algo.ws_size_min = algo.workspaceSize;
              cached_algo.index = idx;
              algoCache.store(gemm_cfg, cached_algo);
            }
            break;
          }
        }
        idx = (idx + 1) % algoTotalCount;
      }
      if (!cached_algo.algo.has_value())
      {
        std::cout << "[WARNING] Cannot find cached algoId " << cached_algo.algoId << " in hipBLASLt results" << std::endl;
      }
    }

    //No suitable entry in autotune cache or could not find matched algo in hipBLASLt results
    if (!cached_algo.algo.has_value())
    {

      int bestAlgo = -1;
      if (tuneLoopCount > 0)
      {
        std::cout << "[INFO] Perform hipBLASLt algo selection on GPU" << device_id
                  << " in range [" << firstAlgo << "-" << (algoTuneCount - 1) << "] with "
                  << tuneLoopCount << " loops " << std::endl;

        hipStream_t profilingStream;
        NVTE_CHECK_CUDA(hipStreamCreateWithFlags(&profilingStream, hipStreamNonBlocking));
        using tuning_clock = std::chrono::steady_clock;
        tuning_clock::now(); //the first call takes little longer so do it outside the loop
        tuning_clock::duration bestTime = tuning_clock::duration::max();

        for (int algo=firstAlgo; algo<algoTuneCount; algo++)
        {
            if (algoArr[algo].state != HIPBLAS_STATUS_SUCCESS)
            {
              continue;
            }
            // Warm-up call
            NVTE_CHECK_HIPBLASLT(hipblasLtMatmul(handle,
                                            operationDesc,
                                            static_cast<const void*>(&one),         /* alpha */
                                            A,                                      /* A */
                                            Adesc,
                                            B,                                      /* B */
                                            Bdesc,
                                            static_cast<const void*>(&beta),        /* beta */
                                            D,                                      /* C */
                                            Ddesc,
                                            D,                                      /* D */
                                            Ddesc,
                                            &algoArr[algo].algo,                    /* algo */
                                            workspace,                              /* workspace */
                                            workspaceSize,
                                            profilingStream));                       /* stream */
          NVTE_CHECK_CUDA(hipStreamSynchronize(profilingStream));

          //Profiling loop
          tuning_clock::time_point startTime = tuning_clock::now();
          for (int loop=0; loop<tuneLoopCount; loop++)
          {
            NVTE_CHECK_HIPBLASLT(hipblasLtMatmul(handle,
                                            operationDesc,
                                            static_cast<const void*>(&one),         /* alpha */
                                            A,                                      /* A */
                                            Adesc,
                                            B,                                      /* B */
                                            Bdesc,
                                            static_cast<const void*>(&beta),        /* beta */
                                            D,                                      /* C */
                                            Ddesc,
                                            D,                                      /* D */
                                            Ddesc,
                                            &algoArr[algo].algo,                    /* algo */
                                            workspace,                              /* workspace */
                                            workspaceSize,
                                            profilingStream));                       /* stream */
          }
          NVTE_CHECK_CUDA(hipStreamSynchronize(profilingStream));
          tuning_clock::duration algoTime = tuning_clock::now() - startTime; 
          if (algoTime < bestTime)
          {
            bestAlgo = algo;
            bestTime = algoTime;
          }
        }

        NVTE_CHECK_CUDA(hipStreamDestroy(profilingStream));
        if (bestAlgo >= 0)
        {
          std::cout << "[INFO] Select hipBLASLt algo " << bestAlgo << " with time "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(bestTime).count() / tuneLoopCount
                    << " ns" << std::endl;
        }
      }
      else if (firstAlgo < algoTuneCount)
      {
        if (firstAlgo != 0)
        {
          std::cout << "[INFO] Select hipBLASLt algo " << firstAlgo << std::endl;
        }
        bestAlgo = firstAlgo;
      }

      if (bestAlgo < 0) {
        NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Ddesc));
        NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Bdesc));
        NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Adesc));
        NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescDestroy(operationDesc));
        throw std::runtime_error("Unable to find any suitable algorithms");
      }
      cached_algo.algo = algoArr[bestAlgo].algo;
      cached_algo.index = bestAlgo;
      cached_algo.algoId = cached_algo.getAlgoId(algoArr[bestAlgo].algo);
      cached_algo.ws_size_min = algoArr[bestAlgo].workspaceSize;
      cached_algo.ws_size_max = workspaceSize;

      algoCache.store(gemm_cfg, cached_algo);
    }
  }

  // D = alpha * (A * B) + beta * C
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmul(handle,
                                   operationDesc,
                                   static_cast<const void*>(&one),         /* alpha */
                                   A,                                      /* A */
                                   Adesc,
                                   B,                                      /* B */
                                   Bdesc,
                                   static_cast<const void*>(&beta),        /* beta */
                                   D,                                      /* C */
                                   Ddesc,
                                   D,                                      /* D */
                                   Ddesc,
                                   &cached_algo.algo.value(),              /* algo */
                                   workspace,                              /* workspace */
                                   workspaceSize,
                                   stream));                               /* stream */


  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescDestroy(operationDesc));
}
#endif //USE_HIPBLASLT

#ifdef USE_ROCBLAS // Use rocblas + kernel, no fusion
void rocblas_gemm(const Tensor *inputA,
                 const Tensor *inputB,
                 Tensor *outputD,
                 const Tensor *inputBias,
                 Tensor *outputPreGelu,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 rocblas_operation transa,
                 rocblas_operation transb,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool accumulate,
                 bool use_split_accumulator,
                 int math_sm_count,
                 int m_split,
                 int n_split,
                 bool gemm_producer,
                 const Tensor *inputCounter,
                 hipStream_t stream
) { 
  void *A = inputA->data.dptr;
  void *A_scale_inverse = inputA->scale_inv.dptr;
  void *B = inputB->data.dptr;
  void *B_scale_inverse = inputB->scale_inv.dptr;
  void *C = outputD->data.dptr;
  void *D = outputD->data.dptr;
  void *D_scale = outputD->scale.dptr;
  void *D_amax = outputD->amax.dptr;
  void *bias_ptr = inputBias->data.dptr;
  const bool bias = bias_ptr != nullptr;
  void *pre_gelu_out = outputPreGelu->data.dptr;
  const bool gelu = pre_gelu_out != nullptr;
  const bool use_fp8 = is_fp8_dtype(inputA->data.dtype) ||
                       is_fp8_dtype(inputB->data.dtype);
  const rocblas_datatype A_type = get_rocblas_dtype(inputA->data.dtype);
  const rocblas_datatype B_type = get_rocblas_dtype(inputB->data.dtype);
  const rocblas_datatype D_type = get_rocblas_dtype(outputD->data.dtype);
  const rocblas_datatype bias_type = get_rocblas_dtype(inputBias->data.dtype);
  const rocblas_datatype gelu_type = get_rocblas_dtype(outputPreGelu->data.dtype);
  
  // check consistency of arguments:
  // if fp8 is desired, context cannot be null
  // fp8 + gelu fusion + fp8 aux is unavailable right now.
  if (use_fp8 && gelu) {
    NVTE_CHECK(!is_fp8_dtype(outputPreGelu->data.dtype),
             "fp8 Aux output for gemm + gelu fusion not supported!");
  }
  if (is_fp8_dtype(outputD->data.dtype)) {
    NVTE_CHECK(!accumulate,
             "Accumulation mode not supported with FP8 GEMM output!");
  }
  // fp8 + grad unavailable in upstream
  NVTE_CHECK(!(use_fp8 && grad), "fp8 + grad not supported!");

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  float alpha = 1.0;
  if (use_fp8) {
     float A_scale_inv, B_scale_inv;
     hipMemcpy(&A_scale_inv, A_scale_inverse, sizeof(float), hipMemcpyDeviceToHost);
     hipMemcpy(&B_scale_inv, B_scale_inverse, sizeof(float), hipMemcpyDeviceToHost);
     alpha = A_scale_inv * B_scale_inv;
  }

  rocblas_handle handle;
  NVTE_CHECK_ROCBLAS(rocblas_create_handle(&handle));
  NVTE_CHECK_ROCBLAS(rocblas_set_stream(handle, stream));

  // extract the stream order alloc env
  bool stream_order_alloc = false;
  if (const char* env_p = std::getenv("ROCBLAS_STREAM_ORDER_ALLOC") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      stream_order_alloc = true;
  }

  int64_t ld_gelumat = (int64_t) ldd;


  NVTE_CHECK((A_type==rocblas_datatype_f16_r && B_type==rocblas_datatype_f16_r && D_type==rocblas_datatype_f16_r) || 
       (A_type==rocblas_datatype_bf16_r && B_type==rocblas_datatype_bf16_r && D_type==rocblas_datatype_bf16_r) || 
       (A_type==rocblas_datatype_f32_r && B_type==rocblas_datatype_f32_r && D_type==rocblas_datatype_f32_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_f32_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_f16_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_bf16_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_f8_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_bf8_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_bf8_r && D_type==rocblas_datatype_f32_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_bf8_r && D_type==rocblas_datatype_f16_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_bf8_r && D_type==rocblas_datatype_bf16_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_bf8_r && D_type==rocblas_datatype_f8_r) ||
       (A_type==rocblas_datatype_f8_r && B_type==rocblas_datatype_bf8_r && D_type==rocblas_datatype_bf8_r) ||
       (A_type==rocblas_datatype_bf8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_f32_r) ||
       (A_type==rocblas_datatype_bf8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_f16_r) ||
       (A_type==rocblas_datatype_bf8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_bf16_r)||
       (A_type==rocblas_datatype_bf8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_f8_r)||
       (A_type==rocblas_datatype_bf8_r && B_type==rocblas_datatype_f8_r && D_type==rocblas_datatype_bf8_r),
      "Only the following combinations of data types are enabled now!\n\
1. input: fp32, output: fp32.\n\
2. input: fp16, output: fp16.\n\
3. input: bf16, output: bf16.\n\
4. input: fp8/bf8, output: fp8/bf8, fp16/bf16, fp32");


  //If D is not fp32, then we need a temp buffer for GEMM result before applying epilogues. Otherwise, we can apply epilogues in-place.
  // with bias or gelu, allocate fp32 D_temp if the output is not fp32
  // with input fp8/bf8 (use_fp8) and bf16 output, need a fp32 D_temp, as rocblas does not support this case (fp8/bf8 input fp16/fp32 output is supported)
  // with use_fp8 true and fp8/bf8 output, need fp32 D_temp to support amax and scale operation
  void* D_temp;
  if (((bias || gelu) && (D_type==rocblas_datatype_f16_r ||D_type==rocblas_datatype_bf16_r))|| 
      (use_fp8 && (D_type==rocblas_datatype_bf16_r||D_type==rocblas_datatype_f8_r||D_type==rocblas_datatype_bf8_r))) {
    if(! stream_order_alloc){
      NVTE_CHECK_CUDA( hipMalloc(&D_temp, sizeof(float)*m*n) );
    }else{
#if HIP_VERSION >= 50300000
      NVTE_CHECK_CUDA( hipMallocAsync(&D_temp, sizeof(float)*m*n, stream) );
#else
      NVTE_ERROR("Stream order allocation is supported on ROCm 5.3 and above.");
#endif  
    }
  }else {
    D_temp = D;
  }

  // When Ti=To=fp16 and there is no bias or gelu, D_temp points to D and we would like it to be fp16
  rocblas_datatype D_temp_type = rocblas_datatype_f32_r;
  if (!(bias || gelu) && (A_type==rocblas_datatype_f16_r && B_type==rocblas_datatype_f16_r && D_type==rocblas_datatype_f16_r)) {
    D_temp_type = rocblas_datatype_f16_r;
  }
  // When Ti=To=bf16 and there is no bias or gelu, D_temp points to D and we would like it to be bf16
  if (!(bias || gelu) && (A_type==rocblas_datatype_bf16_r && B_type==rocblas_datatype_bf16_r && D_type==rocblas_datatype_bf16_r)) {
    D_temp_type = rocblas_datatype_bf16_r;
  }
  // When Ti in fp8 or bf8, To=fp16, there is no bias or gelu, D_temp points to D and we would like it to be fp16, as rocblas support this case.
  if ((!(bias||gelu))&& (use_fp8 && D_type==rocblas_datatype_f16_r)) {
    D_temp_type = rocblas_datatype_f16_r;
  }
  
  if(accumulate && (D_temp!=D || D_temp_type!=D_type)){
    DType output_dtype = get_transformer_engine_dtype(D_type);
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
      //D_temp allocated only with fp32
      detail::identity_kernelLauncher<OType, float>(reinterpret_cast<const OType*>(D),
                                                    reinterpret_cast<float*>(D_temp), 
                                                    m*n,
                                                    stream);
    );  
  }

  // D = alpha * (A * B) + beta * C
  if (use_fp8) {
    rocblas_computetype computeType = rocblas_compute_type_f32;
    NVTE_CHECK_ROCBLAS(rocblas_gemm_ex3(handle, transa, transb, m, n, k, &alpha,
                                       A, A_type, lda,
                                       B, B_type, ldb,
                                       &beta, D_temp, D_temp_type, ldd, D_temp, D_temp_type, ldd,
                                       computeType, rocblas_gemm_algo::rocblas_gemm_algo_standard,0,0));
  }else {
    rocblas_datatype computeType = rocblas_datatype_f32_r;
    uint32_t flags = rocblas_gemm_flags_none;
    if((A_type==rocblas_datatype_f16_r && B_type==rocblas_datatype_f16_r) && grad){
      flags = rocblas_gemm_flags_fp16_alt_impl; 
    }
    NVTE_CHECK_ROCBLAS(rocblas_gemm_ex(handle, transa, transb, m, n, k, &alpha,
                                      A, A_type, lda,
                                      B, B_type, ldb,
                                      &beta, D_temp, D_temp_type, ldd, D_temp, D_temp_type, ldd,
                                      computeType, rocblas_gemm_algo::rocblas_gemm_algo_standard,0,flags));
  }

  NVTE_CHECK_ROCBLAS(rocblas_destroy_handle(handle));

  int batch_size, input_dim, output_dim;
  if (bias && gelu) {
    if (grad) {
      // epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
      // Apply GELU gradient to D_temp and store in D 
      // Apply bias gradient to D (D is already the result of GELU gradient) and store in bias_ptr; 
      // This case is NN
      // D_temp is of shape is (m, n) in column major and thus is of shape (n, m) in row major
      // The bias vector length is m. So it will be reduced along axis 0 in row major
      // (TODO): The cublasLt doc is not very clear wrt the bias gradient here.
      // It does not explicitly say that it goes through GELU gradient first. We will need to
      // confirm in the future. As of now, my implementation for the bias gradient takes
      // the GELU gradient result in lower precision (D). It might be better to take the GELU
      // gradient result in fp32 but as it requires some kernel changes I would only do that
      // once we confirm that this is the right form of the epilogue.
      // This is for linear1 -> gelu -> linear2 
      // compute dX = dY * W for linear2
      // gemm_ex(A=W, B=dY)
      batch_size = n;
      input_dim = m; // input dimension of the second linear layer is the output dimension of the first linear layer
      output_dim = k;
      DType output_dtype = get_transformer_engine_dtype(D_type);
      DType gelu_dtype = get_transformer_engine_dtype(gelu_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType, 
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(gelu_dtype, GType, 
          detail::gelu_backward_kernelLauncher<OType, GType>(reinterpret_cast<const float*>(D_temp), 
                                                             reinterpret_cast<OType*>(D), 
                                                             reinterpret_cast<const GType*>(pre_gelu_out), 
                                                             batch_size, 
                                                             input_dim,
                                                             stream);
        );  
      );

      void* bias_tmp;
      if (bias_type != rocblas_datatype_f32_r) {
        if(! stream_order_alloc){
          NVTE_CHECK_CUDA( hipMalloc(&bias_tmp, sizeof(float)*input_dim) ); // The bias gradient is for the first linear layer
        }else{
#if HIP_VERSION >= 50300000
          NVTE_CHECK_CUDA( hipMallocAsync(&bias_tmp, sizeof(float)*input_dim, stream) );
#else
          NVTE_ERROR("Stream order allocation is supported on ROCm 5.3 and above.");
#endif  
        }
      }else {
        bias_tmp = bias_ptr;
      }

      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
        detail::bias_gradient_kernelLauncher<OType>(reinterpret_cast<const OType*>(D), 
                                                    reinterpret_cast<float*>(bias_tmp), 
                                                    batch_size, 
                                                    input_dim,
                                                    stream_order_alloc,
                                                    stream);
      );

      if (bias_type != rocblas_datatype_f32_r) {
        DType bias_dtype = get_transformer_engine_dtype(bias_type);
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(bias_dtype, BType,
          detail::identity_kernelLauncher<float, BType>(reinterpret_cast<const float*>(bias_tmp), 
                                                        reinterpret_cast<BType*>(bias_ptr),
                                                        input_dim,
                                                        stream);
        );  
        if(! stream_order_alloc){
          NVTE_CHECK_CUDA( hipFree(bias_tmp) ); 
        }else{
#if HIP_VERSION >= 50300000
          NVTE_CHECK_CUDA( hipFreeAsync(bias_tmp, stream) );
#else
          NVTE_ERROR("Stream order allocation is supported on ROCm 5.3 and above.");
#endif
        }
      }

    } else {
      // epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
      // Add bias_ptr to D_temp and store in pre_gelu_out, and apply GELU to the pre_gelu_output and then store in D
      // D_temp is of shape is (m, n) in column major and thus is of shape (n, m) in row major
      // gemm_ex(A=W, B=X, transA=T)
      batch_size = n;
      input_dim = k;
      output_dim = m;
      DType output_dtype = get_transformer_engine_dtype(D_type);
      DType bias_dtype = get_transformer_engine_dtype(bias_type);
      DType gelu_dtype = get_transformer_engine_dtype(gelu_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(gelu_dtype, GType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(bias_dtype, BType,
            detail::add_bias_gelu_kernelLauncher<OType, GType, BType>(reinterpret_cast<const float*>(D_temp), 
                                                                      reinterpret_cast<OType*>(D), 
                                                                      reinterpret_cast<GType*>(pre_gelu_out), 
                                                                      reinterpret_cast<const BType*>(bias_ptr), 
                                                                      reinterpret_cast<float*>(D_amax),
                                                                      reinterpret_cast<const float*>(D_scale),
                                                                      batch_size, 
                                                                      output_dim,
                                                                      stream);
          );
        );
      );
    }
  }else if (bias) {
    if (grad) {
      // grad output is always input B
      // epilogue = CUBLASLT_EPILOGUE_BGRADB;
      // Apply bias gradient to matrix B and store in bias_ptr, reduce along the k dimension, output bias length is n
      // As B is transposed, is of shape (n, k) in column major, and is of shape (k, n) in row major.
      // bias gradient vector length is n. So it will be reduced along axis 0 in row major.
      // The backward pass calculate the bias gradient along with dW = dY^T * X
      // gemm_ex(A=X, B = dY, transB=T)
      batch_size = k;
      input_dim = m;
      output_dim = n;
      void * bias_tmp;
      if (bias_type != rocblas_datatype_f32_r) {
        if(! stream_order_alloc){
          NVTE_CHECK_CUDA( hipMalloc(&bias_tmp, sizeof(float)*output_dim) );
        }else{
#if HIP_VERSION >= 50300000
          NVTE_CHECK_CUDA( hipMallocAsync(&bias_tmp, sizeof(float)*output_dim, stream) );
#else
          NVTE_ERROR("Stream order allocation is supported on ROCm 5.3 and above.");
#endif  
        }
      }else {
        bias_tmp = bias_ptr;
      }

      DType input_dtype = get_transformer_engine_dtype(B_type);
      DType output_dtype = get_transformer_engine_dtype(D_type);
      DType bias_dtype = get_transformer_engine_dtype(bias_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(input_dtype, IType,
        detail::bias_gradient_kernelLauncher<IType>(reinterpret_cast<const IType*>(B), 
                                                    reinterpret_cast<float*>(bias_tmp), 
                                                    batch_size, 
                                                    output_dim,
                                                    stream_order_alloc,
                                                    stream);
      );
      if (bias_type != rocblas_datatype_f32_r) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(bias_dtype, BType,
          detail::identity_kernelLauncher<float, BType>(reinterpret_cast<const float*>(bias_tmp), 
                                                        reinterpret_cast<BType*>(bias_ptr),
                                                        output_dim,
                                                        stream);
        );  
        if(! stream_order_alloc){
          NVTE_CHECK_CUDA( hipFree(bias_tmp) ); 
        }else{
#if HIP_VERSION >= 50300000
          NVTE_CHECK_CUDA( hipFreeAsync(bias_tmp, stream) );
#else
          NVTE_ERROR("Stream order allocation is supported on ROCm 5.3 and above.");
#endif
        }
      }
      if (D_type == rocblas_datatype_f16_r || D_type == rocblas_datatype_bf16_r) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
          detail::identity_kernelLauncher<float, OType>(reinterpret_cast<const float*>(D_temp), 
                                                        reinterpret_cast<OType*>(D),
                                                        input_dim*output_dim,
                                                        stream);
        );  
      }
    } else {
      // epilogue = CUBLASLT_EPILOGUE_BIAS;
      // Broadcast bias and add it to D_temp and store in D. The bias vector length is m 
      // D_temp is of shape is (m, n) in column major and thus is of shape (n, m) in row major
      // gemm_ex(A=W, B=X, transA=T)
      batch_size = n;
      input_dim = k;
      output_dim = m;
      DType output_dtype = get_transformer_engine_dtype(D_type);
      DType bias_dtype = get_transformer_engine_dtype(bias_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(bias_dtype, BType,
          detail::add_bias_kernelLauncher<OType, BType>(reinterpret_cast<const float*>(D_temp), 
                                                        reinterpret_cast<OType*>(D), 
                                                        reinterpret_cast<const BType*>(bias_ptr), 
                                                        reinterpret_cast<float*>(D_amax), 
                                                        reinterpret_cast<const float*>(D_scale), 
                                                        batch_size, 
                                                        output_dim,
                                                        stream);
        );
      );
    }
  }else if (gelu) {
    if (grad) {
      // epilogue = CUBLASLT_EPILOGUE_DGELU;
      // Take input from pre_gelu_out and apply GELU gradients to D_temp and store result in D
      // D_temp is of shape is (m, n) in column major and thus is of shape (n, m) in row major
      // gemm_ex(A=W, B=dY) 
      batch_size = n;
      input_dim = m;
      output_dim = k;
      DType output_dtype = get_transformer_engine_dtype(D_type);
      DType gelu_dtype = get_transformer_engine_dtype(gelu_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(gelu_dtype, GType, 
          detail::gelu_backward_kernelLauncher<OType, GType>(reinterpret_cast<const float*>(D_temp), 
                                                             reinterpret_cast<OType*>(D), 
                                                             reinterpret_cast<const GType*>(pre_gelu_out), 
                                                             batch_size, 
                                                             input_dim,
                                                             stream);
        );
      );  
    } else {
      // epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
      // Store (quantized) D_temp in pre_gelu_out, and apply GELU to D_temp then store in D
      // D_temp is of shape is (m, n) in column major and thus is of shape (n, m) in row major
      // gemm_ex(A=W, B=X, transA=T)
      batch_size = n;
      input_dim = k;
      output_dim = m;

      DType gelu_dtype = get_transformer_engine_dtype(gelu_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(gelu_dtype, GType, 
        detail::identity_kernelLauncher<float, GType>(reinterpret_cast<const float*>(D_temp), 
                                                      reinterpret_cast<GType*>(pre_gelu_out), 
                                                      batch_size*output_dim, 
                                                      stream);
      );  
      DType output_dtype = get_transformer_engine_dtype(D_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
        detail::gelu_forward_kernelLauncher<OType>(reinterpret_cast<const float*>(D_temp), 
                                                   reinterpret_cast<OType*>(D), 
                                                   reinterpret_cast<float*>(D_amax), 
                                                   reinterpret_cast<const float*>(D_scale), 
                                                   batch_size,
                                                   output_dim, 
                                                   stream);
      );  
    }
  } else { // No epilogue - !(bias || gelu)
    if (use_fp8 && (D_type==rocblas_datatype_bf16_r || D_type == rocblas_datatype_f8_r || D_type == rocblas_datatype_bf8_r)) {
      DType output_dtype = get_transformer_engine_dtype(D_type);
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output_dtype, OType,
        detail::identity_output_kernelLauncher<OType>(reinterpret_cast<const float*>(D_temp), 
                                                      reinterpret_cast<OType*>(D),
                                                      reinterpret_cast<float*>(D_amax), 
                                                      reinterpret_cast<const float*>(D_scale), 
                                                      m*n,
                                                      stream);
      );  
    }
  }
  
  if (((bias || gelu) && (D_type==rocblas_datatype_f16_r ||D_type==rocblas_datatype_bf16_r))||
      (use_fp8 && (D_type==rocblas_datatype_bf16_r || D_type==rocblas_datatype_f8_r || D_type==rocblas_datatype_bf8_r))) {
    if(! stream_order_alloc){
      NVTE_CHECK_CUDA( hipFree(D_temp) );
    }else{
#if HIP_VERSION >= 50300000
      NVTE_CHECK_CUDA( hipFreeAsync(D_temp, stream) );
#else
      NVTE_ERROR("Stream order allocation is supported on ROCm 5.3 and above.");
#endif
    }
  }
}

#endif //USE_ROCBLAS

void cublas_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                 const Tensor *inputBias, Tensor *outputPreGelu, int m, int n, int k, int lda,
                 int ldb, int ldd, bool transa, bool transb, bool grad,
                 void *workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                 int math_sm_count, int m_split, int n_split, bool gemm_producer,
                 const Tensor *inputCounter, hipStream_t stream)
{
/*If no backend is specified with env variable use HIPBLASLT unless it is disabled
  If HIPBLASLT backend is enabled and requested, use it despite ROCBLAS status
  Otherwise use ROCBLAS 
*/

  bool use_hipblaslt = std::getenv("NVTE_USE_HIPBLASLT") != nullptr;
  bool use_rocblas = std::getenv("NVTE_USE_ROCBLAS") != nullptr;

#if !defined(USE_HIPBLASLT) && !defined(USE_ROCBLAS)
#error GEMM backend is not specified
#elif !defined(USE_HIPBLASLT)
  if (use_hipblaslt)
  {
    use_hipblaslt = false;
    std::cout << "[NOTICE] hipBLASLt is not enabled, NVTE_USE_HIPBLASLT env is ignored\n";
  }
#elif !defined(USE_ROCBLAS)
  if (use_rocblas)
  {
    use_rocblas = false;
    std::cout << "[NOTICE] rocBLAS is not enabled, NVTE_USE_ROCBLAS env is ignored\n";
  }
#else
  if (use_hipblaslt && use_rocblas)
  {
    use_rocblas = false;
    std::cout << "[NOTICE] Two GEMM backend are enabled, hipBLASLt will be used\n";
  }
#endif

#ifdef USE_HIPBLASLT
  if (use_hipblaslt || !use_rocblas)
  {
    hipblaslt_gemm(inputA, inputB, outputD, inputBias, outputPreGelu, 
                 m, n, k, lda, ldb, ldd, 
                (transa) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                (transb) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                 grad,
                 workspace, workspaceSize, accumulate, use_split_accumulator,
                 math_sm_count, m_split, n_split, gemm_producer,
                 inputCounter, stream);
    return;
  }
#endif

#ifdef USE_ROCBLAS
  {
    rocblas_gemm(inputA, inputB, outputD, inputBias, outputPreGelu, 
                 m, n, k, lda, ldb, ldd, 
                (transa) ? rocblas_operation_transpose : rocblas_operation_none,
                (transb) ? rocblas_operation_transpose : rocblas_operation_none,
                 grad,
                 workspace, workspaceSize, accumulate, use_split_accumulator,
                 math_sm_count, m_split, n_split, gemm_producer,
                 inputCounter, stream);
  }
#endif
}

} //namespace transformer_engine
