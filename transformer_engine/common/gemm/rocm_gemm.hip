/*************************************************************************
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef USE_HIPBLASLT

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
#endif //#ifndef USE_HIPBLASLT

#ifdef USE_HIPBLASLT

static class HandlePool {
public:
  hipblasLtHandle_t get(int device_id) 
  {
    std::lock_guard<std::mutex> lock(mt);

    if (d.empty())
    {
      int device_count = 0; 
      NVTE_CHECK_CUDA(hipGetDeviceCount(&device_count));
      d.resize(device_count);
      return nullptr;
    }

    if (!d[device_id].empty())
    {
      hipblasLtHandle_t h = d[device_id].front();
      d[device_id].pop_front();
      return h;
    }

    return nullptr;
  }

  hipblasLtHandle_t obtain(int device_id) 
  {
    hipblasLtHandle_t h = get(device_id);
    if (h == nullptr)
    {
      NVTE_CHECK_CUBLAS(hipblasLtCreate(&h));
    }
    return h;
  }

  void store(int device_id, hipblasLtHandle_t h)
  {
    std::lock_guard<std::mutex> lock(mt);
    d[device_id].push_front(h);
  }

  void store(const std::vector<hipblasLtHandle_t>& h)
  {
    std::lock_guard<std::mutex> lock(mt);
    if (d.empty())
    {
      // In case of non-graceful application exit
      // the pool may be destructed before all per-thread HandleCache
      // Just drop handles then
      std::cout << "[ERROR] Attempt to store handles to invalid pool" << std::endl;
      return;
    }
    for (unsigned int i=0; i<d.size(); i++)
    {
      if (h[i] != nullptr)
      {
        d[i].push_front(h[i]);
      }
    }
  }

  ~HandlePool() {
    std::lock_guard<std::mutex> lock(mt);
    for (auto & hlist : d)
    {
      for (auto & h : hlist)
      {
        hipblasLtDestroy(h);
      }
    }
    d.clear();
  }

  inline size_t get_size() const
  {
    return d.size();
  }

private:
  std::mutex mt;
  std::vector<std::forward_list<hipblasLtHandle_t>> d;
} handle_pool;

thread_local static class HandleCache {
public:
  hipblasLtHandle_t get(int device_id) const
  {
    return d.empty() ? nullptr : d[device_id];
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


void cublas_gemm(const Tensor *inputA,
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
  const hipblasltDatatype_t A_type = get_cuda_dtype(inputA->data.dtype);
  const hipblasltDatatype_t B_type = get_cuda_dtype(inputB->data.dtype);
  const hipblasltDatatype_t D_type = get_cuda_dtype(outputD->data.dtype);
  const hipblasltDatatype_t bias_type = get_cuda_dtype(inputBias->data.dtype);

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
    handle = handle_pool.obtain(device_id);
    cached_handles.set(device_id, handle);
  }

  hipblasLtMatmulDesc_t       operationDesc = nullptr;
  hipblasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  int                             returnedResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult = {}; //TODO: Is this Okay?
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t) ldd;

  // default to tf32 except for e5m2 inputs where the config is not supported
  hipblasLtComputeType_t gemm_compute_type = HIPBLASLT_COMPUTE_F32;

  // Create matrix descriptors. Not setting any extra attributes.
  NVTE_CHECK_CUBLAS(hipblasLtMatrixLayoutCreate(&Adesc, A_type,
                                               transa == HIPBLAS_OP_N ? m : k,
                                               transa == HIPBLAS_OP_N ? k : m,
                                               lda));
  NVTE_CHECK_CUBLAS(hipblasLtMatrixLayoutCreate(&Bdesc, B_type,
                                               transb == HIPBLAS_OP_N ? k : n,
                                               transb == HIPBLAS_OP_N ? n : k,
                                               ldb));
  NVTE_CHECK_CUBLAS(hipblasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  NVTE_CHECK_CUBLAS(hipblasLtMatmulDescCreate(&operationDesc, gemm_compute_type, HIPBLASLT_R_32F));
  NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                   &transa, sizeof(transa)));
  NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                   &transb, sizeof(transb)));

  // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
  // Note: gelu fusion isn't available right now, and we don't need
  // amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    /*
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_FAST_ACCUM, //TODO: We don't have fast accum mode yet
                                                     &fastAccuMode,
                                                     sizeof(fastAccuMode)));
    */
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                     &A_scale_inverse,
                                                     sizeof(A_scale_inverse)));
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                     &B_scale_inverse,
                                                     sizeof(B_scale_inverse)));
    if (bias) {
      NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
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
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias_ptr, sizeof(bias_ptr)));
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(
                            operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                      &ld_gelumat, sizeof(ld_gelumat)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = HIPBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_BIAS;
    }
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias_ptr, sizeof(bias_ptr)));
  } else if (gelu) {
    if (grad) {
      epilogue = HIPBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;
    }
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(
                            operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                     &ld_gelumat, sizeof(ld_gelumat)));
  }

  NVTE_CHECK_CUBLAS(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                   HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue)));

  NVTE_CHECK_CUBLAS(hipblasLtMatmulPreferenceCreate(&preference));
  NVTE_CHECK_CUBLAS(hipblasLtMatmulPreferenceSetAttribute(
                          preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          &workspaceSize, sizeof(workspaceSize)));

  NVTE_CHECK_CUBLAS(hipblasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Ddesc,
                                                   Ddesc, preference, 1, &heuristicResult,
                                                   &returnedResults));

  if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C
  NVTE_CHECK_CUBLAS(hipblasLtMatmul(handle,
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
                                   &heuristicResult.algo,                  /* algo */
                                   workspace,                              /* workspace */
                                   workspaceSize,
                                   stream));                               /* stream */


  NVTE_CHECK_CUBLAS(hipblasLtMatmulPreferenceDestroy(preference));
  NVTE_CHECK_CUBLAS(hipblasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_CUBLAS(hipblasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_CUBLAS(hipblasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_CUBLAS(hipblasLtMatmulDescDestroy(operationDesc));
}
#else // Use rocblas + kernel, no fusion
void cublas_gemm(const Tensor *inputA,
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
  const rocblas_datatype A_type = get_cuda_dtype(inputA->data.dtype);
  const rocblas_datatype B_type = get_cuda_dtype(inputB->data.dtype);
  const rocblas_datatype D_type = get_cuda_dtype(outputD->data.dtype);
  const rocblas_datatype bias_type = get_cuda_dtype(inputBias->data.dtype);
  const rocblas_datatype gelu_type = get_cuda_dtype(outputPreGelu->data.dtype);
  
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
  NVTE_CHECK_CUBLAS(rocblas_create_handle(&handle));
  NVTE_CHECK_CUBLAS(rocblas_set_stream(handle, stream));

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

  // D = alpha * (A * B) + beta * C
  if (use_fp8) {
    rocblas_computetype computeType = rocblas_compute_type_f32;
    NVTE_CHECK_CUBLAS(rocblas_gemm_ex3(handle, transa, transb, m, n, k, &alpha,
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
    NVTE_CHECK_CUBLAS(rocblas_gemm_ex(handle, transa, transb, m, n, k, &alpha,
                                      A, A_type, lda,
                                      B, B_type, ldb,
                                      &beta, D_temp, D_temp_type, ldd, D_temp, D_temp_type, ldd,
                                      computeType, rocblas_gemm_algo::rocblas_gemm_algo_standard,0,flags));
  }

  NVTE_CHECK_CUBLAS(rocblas_destroy_handle(handle));

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

#endif // #ifdef USE_HIPBLASLT

