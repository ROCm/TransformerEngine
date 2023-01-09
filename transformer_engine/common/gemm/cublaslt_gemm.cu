/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/logging.h>
#include <transformer_engine/gemm.h>
#ifndef __HIP_PLATFORM_HCC__
#include <cublasLt.h>
#endif
#include <cublas_v2.h>
#include "../common.h"
#include "../util/vectorized_pointwise.h"
#ifdef __HIP_PLATFORM_HCC__
#include <hipcub/hipcub.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#endif


namespace transformer_engine {


#ifdef __HIP_PLATFORM_HCC__


#define TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kFloat32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat16: \
            { \
                using type = fp16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kBFloat16: \
        case DType::kFloat8E5M2: \
        case DType::kFloat8E4M3: \
            { \
                NVTE_ERROR("Bfloat16 and FP8 type not instantiated"); \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }


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

template <typename Tin, typename T>
__global__
void gelu_forward_kernel(const Tin* in, T* out, int m, int n) {
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    Tin x = (Tin)(__ldg(&in[id]));
    float y = gelu_forward((float)x); 
    out[id] = (T)(y);
  }

}

template <typename Tin, typename T>
void gelu_forward_kernelLauncher(const Tin* in, T* out, int m, int n, hipStream_t stream) {
  int blocks_per_row = ceil(float(n)/1024);
  dim3 grid(min(m * blocks_per_row, 65536));
  dim3 block(min(n, 1024));
  hipLaunchKernelGGL(( gelu_forward_kernel<Tin, T>), dim3(grid), dim3(block), 0, stream, in, out, m, n);
}


__inline__ __device__
float gelu_backward(float x, float dy)
{
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

template <typename Tin, typename T>
__global__ 
void gelu_backward_kernel(const Tin* dy, T* out, const T* __restrict pre_gelu_out, int m, int n) {
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    Tin x = (Tin)(__ldg(&pre_gelu_out[id]));
    Tin dx = gelu_backward((float)x, (float)dy[id]); 
    out[id] = (T)(dx);
  }
}

template <typename Tin, typename T>
void gelu_backward_kernelLauncher(const Tin* in, T* out, const T* pre_gelu_out, int m, int n, hipStream_t stream) {
  int blocks_per_row = ceil(float(n)/1024);
  dim3 grid(min(m * blocks_per_row, 65536));
  dim3 block(min(n, 1024));
  hipLaunchKernelGGL(( gelu_backward_kernel<Tin, T>), dim3(grid), dim3(block), 0, stream, in, out, pre_gelu_out, m, n);
}

template <typename Tin, typename T>
__global__ 
void add_bias_kernel(const Tin* in, T* out, const T* __restrict bias, int m, int n)
{
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    Tin reg_bias = (Tin)(__ldg(&bias[id % n]));
    Tin val = in[id] + reg_bias;
    out[id] = (T)(val);
  }
}


template <typename Tin, typename T>
void add_bias_kernelLauncher(const Tin* in, T* out, const T* __restrict bias, int m, int n, hipStream_t stream) {
  dim3 block, grid;
  block.x = 1024;
  grid.x = ceil(m * n / 1024.);
  hipLaunchKernelGGL(( add_bias_kernel<Tin, T>), dim3(grid), dim3(block), 0, stream, in, out, bias, m, n);

}

template <typename Tin, typename T>
__global__ 
void add_bias_gelu_kernel(const Tin* in, T* out, T* pre_gelu_out, const T* __restrict bias, int m, int n)
{
  for(int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x)
  {
    Tin reg_bias = (Tin)(__ldg(&bias[id % n]));
    Tin val = in[id] + reg_bias;
    pre_gelu_out[id] = (T)(val);
    out[id] = (T)(gelu_forward(val));
  }
}

template <typename Tin, typename T>
void add_bias_gelu_kernelLauncher(const Tin* in, T* out, T* pre_gelu_out, const T* __restrict bias, int m, int n, hipStream_t stream) {
  dim3 block, grid;
  block.x = 1024;
  grid.x = ceil(m * n / 1024.);
  hipLaunchKernelGGL(( add_bias_gelu_kernel<Tin, T>), dim3(grid), dim3(block), 0, stream, in, out, pre_gelu_out, bias, m, n );

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
/*
template <typename Tin, int THREADS_PER_BLOCK>
__global__
void bias_gradient_kernel(const Tin* in, float* out, int m, int n) {
  typedef hipcub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;

  int BLOCKS_PER_ROW = ceil(float(n)/THREADS_PER_BLOCK);
  int THREADS_PER_ROW = BLOCKS_PER_ROW * THREADS_PER_BLOCK;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int row_idx = idx / THREADS_PER_ROW;
  int col_idx = idx % THREADS_PER_ROW;
  float thread_data;
  if (col_idx < n)
    thread_data = (float)in[row_idx * n + col_idx];
  float local_sum;
  if (col_idx < (BLOCKS_PER_ROW-1) * THREADS_PER_BLOCK) {
    local_sum = BlockReduce(block_temp_storage).Sum(thread_data);
  }
  else {
    local_sum = BlockReduce(block_temp_storage).Sum(thread_data, n-(BLOCKS_PER_ROW-1)*THREADS_PER_BLOCK);
  }
  if (threadIdx.x == 0)
    atomicAdd(&out[row_idx], local_sum);


}
*/

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
void bias_gradient_kernelLauncher(const Tin* in, float* out, int m, int n, hipStream_t stream) { 
  dim3 block, grid;
  constexpr int THREADS_PER_BLOCK = 1024;
  int BLOCKS_PER_COL = ceil(float(m)/THREADS_PER_BLOCK);
  block.x = THREADS_PER_BLOCK;
  grid.x = BLOCKS_PER_COL*n;
  NVTE_CHECK_CUDA( hipMemset(out, 0, n*sizeof(float)) );
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
    /* HIP-TODO: Add back after installing rocblas with FP8 types
    case DType::kFloat8E4M3:
      return CUDA_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return CUDA_R_8F_E5M2;
    */
    default:
      NVTE_ERROR("Invalid type");
  }
}

void cublas_gemm(void* A,
                 void* A_scale_inverse,
                 void* B,
                 void *B_scale_inverse,
                 void* D,
                 void* bias_ptr,
                 void* pre_gelu_out,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 rocblas_datatype A_type,
                 rocblas_datatype B_type,
                 rocblas_datatype D_type,
                 rocblas_datatype bias_type,
                 rocblas_operation transa,
                 rocblas_operation transb,
                 bool bias,
                 bool gelu,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool use_fp8,
                 bool accumulate,
                 bool use_split_accumulator,
                 cudaStream_t stream
) { 
    // check consistency of arguments:
    // if fp8 is desired, context cannot be null
    // fp8 + gelu fusion is unavailable right now.

    if (use_fp8) {
      NVTE_CHECK(!gelu, "fp8 gemm + gelu fusion is unavailable right now!");
    }

    float one = 1.0;
    float zero = 0.0;
    float beta = (accumulate) ? one : zero;


    rocblas_handle handle;
    NVTE_CHECK_CUBLAS(rocblas_create_handle(&handle));
    rocblas_datatype computeType =  rocblas_datatype_f32_r;

    int64_t ld_gelumat = (int64_t) ldd;


    // We don't deal with fp8 for now
    NVTE_CHECK(!use_fp8, "fp8 gemm  is unavailable right now!");

    NVTE_CHECK((A_type==rocblas_datatype_f16_r && B_type==rocblas_datatype_f16_r && D_type==rocblas_datatype_f16_r) || 
	       (A_type==rocblas_datatype_f32_r && B_type==rocblas_datatype_f32_r && D_type==rocblas_datatype_f32_r),
		    "Only fp32 and fp16 GEMMs are available now!");


    void* D_temp;
    if ((bias || gelu) && (A_type==rocblas_datatype_f16_r && B_type==rocblas_datatype_f16_r && D_type==rocblas_datatype_f16_r)) {
	NVTE_CHECK_CUDA( hipMalloc(&D_temp, sizeof(float)*m*n) );
    }
    else {
	D_temp = D;
    }

    // When Ti=To=fp16 and there is no bias or gelu, D_temp points to D and we would like it to be fp16
    rocblas_datatype D_temp_type = rocblas_datatype_f32_r;
    if (!(bias || gelu) && (A_type==rocblas_datatype_f16_r && B_type==rocblas_datatype_f16_r && D_type==rocblas_datatype_f16_r)) {
        D_temp_type = rocblas_datatype_f16_r;
    }


    // D = alpha * (A * B) + beta * C
    // TODO: Can we search for rocblas_gemm_algo??
    NVTE_CHECK_CUBLAS(rocblas_gemm_ex(handle, transa, transb, m, n, k, &one,
                                  A, A_type, lda,
                                  B, B_type, ldb,
                                  &beta, D_temp, D_temp_type, ldd, D_temp, D_temp_type, ldd,
				  computeType, rocblas_gemm_algo::rocblas_gemm_algo_standard,0,0));
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
	    DType input_dtype = get_transformer_engine_dtype(rocblas_datatype_f32_r);
	    DType output_dtype = get_transformer_engine_dtype(D_type);
            TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(input_dtype, IType,
	      TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::gelu_backward_kernelLauncher<IType, OType>(reinterpret_cast<const IType*>(D_temp), 
					       reinterpret_cast<OType*>(D), 
					       reinterpret_cast<const OType*>(pre_gelu_out), 
					       batch_size, 
					       input_dim,
					       0);
	      );  
	    ); 

	    void* bias_tmp;
	    if (D_type == rocblas_datatype_f16_r) {
	      NVTE_CHECK_CUDA( hipMalloc(&bias_tmp, sizeof(float)*input_dim) ); // The bias gradient is for the first linear layer
	    }
	    else {
	      bias_tmp = bias_ptr;
	    }

	    TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::bias_gradient_kernelLauncher<OType>(reinterpret_cast<const OType*>(D), 
					       reinterpret_cast<float*>(bias_tmp), 
					       batch_size, 
					       input_dim,
					       0);
            );

	    if (D_type == rocblas_datatype_f16_r) {
	      TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::identity_kernelLauncher<float, OType>(reinterpret_cast<const float*>(bias_tmp), 
		       reinterpret_cast<OType*>(bias_ptr),
		       input_dim,
		       0);
	      );  
	      NVTE_CHECK_CUDA( hipDeviceSynchronize() );
	      NVTE_CHECK_CUDA( hipFree(bias_tmp) ); 
	    }

        } else {
            // epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
	    // Add bias_ptr to D_temp and store in pre_gelu_out, and apply GELU to the pre_gelu_output and then store in D
	    // D_temp is of shape is (m, n) in column major and thus is of shape (n, m) in row major
	    // gemm_ex(A=W, B=X, transA=T)
	    batch_size = n;
	    input_dim = k;
	    output_dim = m;
	    DType input_dtype = get_transformer_engine_dtype(rocblas_datatype_f32_r);
	    DType output_dtype = get_transformer_engine_dtype(D_type);
            TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(input_dtype, IType,
	      TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::add_bias_gelu_kernelLauncher<IType, OType>(reinterpret_cast<const IType*>(D_temp), 
					       reinterpret_cast<OType*>(D), 
					       reinterpret_cast<OType*>(pre_gelu_out), 
					       reinterpret_cast<const OType*>(bias_ptr), 
					       batch_size, 
					       output_dim,
					       0);
              );
            );
        }
    } else if (bias) {
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
	    if (B_type == rocblas_datatype_f16_r) {
	      NVTE_CHECK_CUDA( hipMalloc(&bias_tmp, sizeof(float)*output_dim) );
	    }
	    else {
	      bias_tmp = bias_ptr;
	    }

	    DType input_dtype = get_transformer_engine_dtype(B_type);
	    DType output_dtype = get_transformer_engine_dtype(D_type);
            TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(input_dtype, IType,
		detail::bias_gradient_kernelLauncher<IType>(reinterpret_cast<const IType*>(B), 
					       reinterpret_cast<float*>(bias_tmp), 
					       batch_size, 
					       output_dim,
					       0);
            );
	    if (B_type == rocblas_datatype_f16_r) {
		TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		  detail::identity_kernelLauncher<float, OType>(reinterpret_cast<const float*>(bias_tmp), 
			 reinterpret_cast<OType*>(bias_ptr),
			 output_dim,
			 0);
		);  
	      NVTE_CHECK_CUDA( hipDeviceSynchronize() );
	      NVTE_CHECK_CUDA( hipFree(bias_tmp) ); 
	    }
	    if (D_type == rocblas_datatype_f16_r) {
		TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		  detail::identity_kernelLauncher<float, OType>(reinterpret_cast<const float*>(D_temp), 
			 reinterpret_cast<OType*>(D),
			 input_dim*output_dim,
			 0);
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
	    DType input_dtype = get_transformer_engine_dtype(rocblas_datatype_f32_r);
	    DType output_dtype = get_transformer_engine_dtype(D_type);
            TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(input_dtype, IType,
	      TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::add_bias_kernelLauncher<IType, OType>(reinterpret_cast<const IType*>(D_temp), 
					       reinterpret_cast<OType*>(D), 
					       reinterpret_cast<const OType*>(bias_ptr), 
					       batch_size, 
					       output_dim,
					       0);
              );
            );
        }
    } else if (gelu) {
        if (grad) {
            // epilogue = CUBLASLT_EPILOGUE_DGELU;
	    // Take input from pre_gelu_out and apply GELU gradients to D_temp and store result in D
	    // D_temp is of shape is (m, n) in column major and thus is of shape (n, m) in row major
	    // gemm_ex(A=W, B=dY) 
	    batch_size = n;
	    input_dim = m;
	    output_dim = k;
	    DType input_dtype = get_transformer_engine_dtype(rocblas_datatype_f32_r);
	    DType output_dtype = get_transformer_engine_dtype(D_type);
            TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(input_dtype, IType,
	      TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::gelu_backward_kernelLauncher<IType, OType>(reinterpret_cast<const IType*>(D_temp), 
					       reinterpret_cast<OType*>(D), 
					       reinterpret_cast<const OType*>(pre_gelu_out), 
					       batch_size, 
					       input_dim,
					       0);
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
	    DType input_dtype = get_transformer_engine_dtype(rocblas_datatype_f32_r);
	    DType output_dtype = get_transformer_engine_dtype(D_type);
            TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(input_dtype, IType,
	      TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::gelu_forward_kernelLauncher<IType, OType>(reinterpret_cast<const IType*>(D_temp), 
					       reinterpret_cast<OType*>(D), 
					       batch_size,
					       output_dim, 
					       0);
	      );  
	    ); 
            TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(input_dtype, IType,
	      TRANSFORMER_ENGINE_TYPE_SWITCH_ROCM_SIM(output_dtype, OType,
		detail::identity_kernelLauncher<IType, OType>(reinterpret_cast<const IType*>(D_temp), 
					       reinterpret_cast<OType*>(pre_gelu_out), 
					       batch_size*output_dim, 
					       0);
	      );  
	    ); 
        }
    }
    if ((bias || gelu) && (A_type==rocblas_datatype_f16_r && B_type==rocblas_datatype_f16_r && D_type==rocblas_datatype_f16_r)) {
      	NVTE_CHECK_CUDA( hipFree(D_temp) );
    }
/*
    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    */
}
#else
void cublas_gemm(void* A,
                 void* A_scale_inverse,
                 void* B,
                 void *B_scale_inverse,
                 void* D,
                 void* bias_ptr,
                 void* pre_gelu_out,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 cudaDataType_t A_type,
                 cudaDataType_t B_type,
                 cudaDataType_t D_type,
                 cudaDataType_t bias_type,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 bool bias,
                 bool gelu,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool use_fp8,
                 bool accumulate,
                 bool use_split_accumulator,
                 cudaStream_t stream
) {
    // check consistency of arguments:
    // if fp8 is desired, context cannot be null
    // fp8 + gelu fusion is unavailable right now.
    if (use_fp8) {
      NVTE_CHECK(!gelu, "fp8 gemm + gelu fusion is unavailable right now!");
    }

    float one = 1.0;
    float zero = 0.0;
    float beta = (accumulate) ? one : zero;

    cublasLtHandle_t handle;
    NVTE_CHECK_CUBLAS(cublasLtCreate(&handle));

    cublasLtMatmulDesc_t       operationDesc = nullptr;
    cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    int                             returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

    int64_t ld_gelumat = (int64_t) ldd;

    // default to tf32 except for e5m2 inputs where the config is not supported
    cublasComputeType_t gemm_compute_type = (A_type == CUDA_R_8F_E5M2 || B_type == CUDA_R_8F_E5M2)
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_32F_FAST_TF32;

    // Create matrix descriptors. Not setting any extra attributes.
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type,
                                                 transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m,
                                                 lda));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type,
                                                 transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k,
                                                 ldb));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

    NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                     &transa, sizeof(transa)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                     &transb, sizeof(transb)));

    // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
    // Note: gelu fusion isn't available right now, and we don't need
    // amax(D) either (next op is high precision).
    if (use_fp8) {
        // Split accumulator.
        const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                         &fastAccuMode,
                                                         sizeof(fastAccuMode)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                         &A_scale_inverse,
                                                         sizeof(A_scale_inverse)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                         &B_scale_inverse,
                                                         sizeof(B_scale_inverse)));
        if (bias) {
            NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                             CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                             &bias_type, sizeof(bias_type)));
        }
    }

    if (bias && gelu) {
        if (grad) {
            epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
        } else {
            epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
        }
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                         &bias_ptr, sizeof(bias_ptr)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                                operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                &pre_gelu_out, sizeof(pre_gelu_out)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                         &ld_gelumat, sizeof(ld_gelumat)));
    } else if (bias) {
        if (grad) {
            // grad output is always input B
            epilogue = CUBLASLT_EPILOGUE_BGRADB;
        } else {
            epilogue = CUBLASLT_EPILOGUE_BIAS;
        }
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                         &bias_ptr, sizeof(bias_ptr)));
    } else if (gelu) {
        if (grad) {
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
        }
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                                operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                &pre_gelu_out, sizeof(pre_gelu_out)));
        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                         CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                         &ld_gelumat, sizeof(ld_gelumat)));
    }

    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue, sizeof(epilogue)));

    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
                            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            &workspaceSize, sizeof(workspaceSize)));

    NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Ddesc,
                                                     Ddesc, preference, 1, &heuristicResult,
                                                     &returnedResults));

    if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

    // D = alpha * (A * B) + beta * C
    NVTE_CHECK_CUBLAS(cublasLtMatmul(handle,
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


    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}
#endif
}  // namespace transformer_engine

namespace {
#ifdef __HIP_PLATFORM_HCC__
rocblas_datatype get_cuda_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return rocblas_datatype_f16_r;
    case DType::kFloat32:
      return rocblas_datatype_f32_r;
    case DType::kBFloat16:
      return rocblas_datatype_bf16_r;
    /* HIP-TODO: Add back after installing rocblas with FP8 types
    case DType::kFloat8E4M3:
      return CUDA_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return CUDA_R_8F_E5M2;
    */
    default:
      NVTE_ERROR("Invalid type");
  }
}
#else
cudaDataType_t get_cuda_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return CUDA_R_16F;
    case DType::kFloat32:
      return CUDA_R_32F;
    case DType::kBFloat16:
      return CUDA_R_16BF;
    case DType::kFloat8E4M3:
      return CUDA_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return CUDA_R_8F_E5M2;
    default:
      NVTE_ERROR("Invalid type");
  }
}
#endif
bool is_fp8_dtype(const transformer_engine::DType t) {
  return t == transformer_engine::DType::kFloat8E4M3 ||
         t == transformer_engine::DType::kFloat8E5M2;
}

}  // namespace

void nvte_cublas_gemm(const NVTETensor A,
                      const NVTETensor A_scale_inverse,
                      const NVTETensor B,
                      const NVTETensor B_scale_inverse,
                      NVTETensor D,
                      const NVTETensor bias,
                      NVTETensor pre_gelu_out,
                      bool transa,
                      bool transb,
                      bool grad,
                      NVTETensor workspace,
                      bool accumulate,
                      bool use_split_accumulator,
                      cudaStream_t stream) {
  using namespace transformer_engine;
  const Tensor *inputA = reinterpret_cast<const Tensor*>(A);
  const Tensor *inputB = reinterpret_cast<const Tensor*>(B);
  const Tensor *Ainvscale = reinterpret_cast<const Tensor*>(A_scale_inverse);
  const Tensor *Binvscale = reinterpret_cast<const Tensor*>(B_scale_inverse);
  Tensor *outputD = reinterpret_cast<Tensor*>(D);
  const Tensor *biasTensor = reinterpret_cast<const Tensor*>(bias);
  Tensor *outputGelu = reinterpret_cast<Tensor*>(pre_gelu_out);
  Tensor *wspace = reinterpret_cast<Tensor*>(workspace);

  const int m = transa ? inputA->shape[0] : inputA->shape[1];
  const int k = transa ? inputA->shape[1] : inputA->shape[0];
  const int n = transb ? inputB->shape[1] : inputB->shape[0];
  int lda, ldb, ldd;
  if (transa && !transb) {  // TN
    lda = k;
    ldb = k;
    ldd = m;
  } else if (!transa && !transb) {  // NN
    lda = m;
    ldb = k;
    ldd = m;
  } else if (!transa && transb) {  // NT
    lda = m;
    ldb = n;
    ldd = m;
  } else {  // TT
    NVTE_ERROR("TT layout not allowed.");
  }

  bool nvte_log_gemm_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_GEMM_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      nvte_log_gemm_config = true;
  }

  if (nvte_log_gemm_config) 
    std::cout << "m=" << m << " k=" << k << " n=" << n 
	      << " transa=" << (transa?"T":"N")
	      << " transb=" << (transb?"T":"N")
	      << " A_type=" << (int)inputA->dtype
	      << " B_type=" << (int)inputB->dtype
	      << " D_type=" << (int)outputD->dtype
	      << " grad=" << grad
	      << " bias=" << (biasTensor->dptr != nullptr)
	      << " gelu=" << (outputGelu->dptr != nullptr)
	      << " accumulate=" << accumulate
	      << std::endl;

  cublas_gemm(inputA->dptr, Ainvscale->dptr,
              inputB->dptr, Binvscale->dptr,
              outputD->dptr, biasTensor->dptr,
              outputGelu->dptr,
              m, n, k,
              lda, ldb, ldd,
              get_cuda_dtype(inputA->dtype),
              get_cuda_dtype(inputB->dtype),
              get_cuda_dtype(outputD->dtype),
              get_cuda_dtype(biasTensor->dtype),
              (transa) ? CUBLAS_OP_T : CUBLAS_OP_N,
              (transb) ? CUBLAS_OP_T : CUBLAS_OP_N,
              biasTensor->dptr != nullptr,
              outputGelu->dptr != nullptr,
              grad, wspace->dptr,
              wspace->shape[0],
              is_fp8_dtype(inputA->dtype) || is_fp8_dtype(inputB->dtype),
              accumulate, use_split_accumulator,
              stream);
}
