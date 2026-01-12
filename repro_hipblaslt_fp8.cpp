#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  std::cerr<<"HIP error "<<hipGetErrorString(e)<<" at "<<__LINE__<<"\n"; std::exit(1);} } while(0)

#define CHECK_LT(x) do { hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){ \
  std::cerr<<"hipBLASLt error "<<(int)s<<" at "<<__LINE__<<"\n"; std::exit(1);} } while(0)

__global__ void diff_kernel(const __hip_bfloat16* a, const __hip_bfloat16* b, size_t n,
                            int* nan_count, float* max_abs) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float fa = (float)a[i];
  float fb = (float)b[i];
  if (!isfinite(fa) || !isfinite(fb)) atomicAdd(nan_count, 1);
  float d = fabsf(fa - fb);
  // naive atomic max for float via CAS
  int* as_i = (int*)max_abs;
  int old = *as_i, assumed;
  do {
    assumed = old;
    float cur = __int_as_float(assumed);
    float nxt = d > cur ? d : cur;
    old = atomicCAS(as_i, assumed, __float_as_int(nxt));
  } while (assumed != old);
}

static hipblasLtMatmulAlgo_t algo_from_id(hipblasLtHandle_t h, int algo_id) {
  std::vector<hipblasLtMatmulHeuristicResult_t> arr;
  std::vector<int> idx{algo_id};
  hipblasStatus_t st = hipblaslt_ext::getAlgosFromIndex(h, idx, arr);
  if (st != HIPBLAS_STATUS_SUCCESS || arr.empty() || arr[0].state != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "getAlgosFromIndex failed for algo_id=" << algo_id << " status=" << (int)st << "\n";
    std::exit(2);
  }
  return arr[0].algo;
}

int main(int argc, char** argv) {
  const int algo_bad  = (argc > 1) ? std::atoi(argv[1]) : 620054;
  const int algo_good = (argc > 2) ? std::atoi(argv[2]) : 620086;

  const int m = 2048, n = 16384, k = 576;
  const hipblasOperation_t transA = HIPBLAS_OP_T;
  const hipblasOperation_t transB = HIPBLAS_OP_N;

  // Layouts match TE for this config:
  // A desc rows=(transA==N?m:k)=k, cols=(transA==N?k:m)=m, ld=lda=k
  const int64_t lda = k;
  const int64_t ldb = k;
  const int64_t ldd = m;

  size_t elemsA = (size_t)k * (size_t)m;      // 576 * 2048
  size_t elemsB = (size_t)k * (size_t)n;      // 576 * 16384
  size_t elemsD = (size_t)m * (size_t)n;      // 2048 * 16384

  // Host init (small values to avoid overflow)
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__hip_fp8_e4m3> hA(elemsA);
  std::vector<__hip_fp8_e5m2> hB(elemsB);
  for (size_t i = 0; i < elemsA; i++) hA[i] = __hip_fp8_e4m3(dist(rng));
  for (size_t i = 0; i < elemsB; i++) hB[i] = __hip_fp8_e5m2(dist(rng));

  __hip_fp8_e4m3* dA = nullptr;
  __hip_fp8_e5m2* dB = nullptr;
  __hip_bfloat16 *dD_bad = nullptr, *dD_good = nullptr;
  CHECK_HIP(hipMalloc(&dA, elemsA * sizeof(*dA)));
  CHECK_HIP(hipMalloc(&dB, elemsB * sizeof(*dB)));
  CHECK_HIP(hipMalloc(&dD_bad,  elemsD * sizeof(*dD_bad)));
  CHECK_HIP(hipMalloc(&dD_good, elemsD * sizeof(*dD_good)));
  CHECK_HIP(hipMemcpy(dA, hA.data(), elemsA * sizeof(*dA), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dB, hB.data(), elemsB * sizeof(*dB), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemset(dD_bad,  0, elemsD * sizeof(*dD_bad)));
  CHECK_HIP(hipMemset(dD_good, 0, elemsD * sizeof(*dD_good)));

  float *dA_scale = nullptr, *dB_scale = nullptr;
  CHECK_HIP(hipMalloc(&dA_scale, sizeof(float)));
  CHECK_HIP(hipMalloc(&dB_scale, sizeof(float)));
  float scale_a = 9.31876e-05;
  float scale_b = 7.79567e-12;
  CHECK_HIP(hipMemcpy(dA_scale, &scale_a, sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dB_scale, &scale_b, sizeof(float), hipMemcpyHostToDevice));

  void* workspace = nullptr;
  size_t workspaceSize = 64ull * 1024ull * 1024ull;
  CHECK_HIP(hipMalloc(&workspace, workspaceSize));

  hipblasLtHandle_t h;
  CHECK_LT(hipblasLtCreate(&h));

  hipblasLtMatmulDesc_t op;
  hipblasLtMatrixLayout_t Ad, Bd, Cd, Dd;
  CHECK_LT(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));

  CHECK_LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CHECK_LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  auto scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  CHECK_LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dA_scale, sizeof(dA_scale)));
  CHECK_LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dB_scale, sizeof(dB_scale)));
  CHECK_LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  CHECK_LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

  CHECK_LT(hipblasLtMatrixLayoutCreate(&Ad, HIP_R_8F_E4M3, (int64_t)k, (int64_t)m, lda));
  CHECK_LT(hipblasLtMatrixLayoutCreate(&Bd, HIP_R_8F_E5M2, (int64_t)k, (int64_t)n, ldb));
  CHECK_LT(hipblasLtMatrixLayoutCreate(&Dd, HIP_R_16BF,    (int64_t)m, (int64_t)n, ldd));
  Cd = Dd;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  auto run = [&](int algo_id, __hip_bfloat16* dD) {
    hipblasLtMatmulAlgo_t algo = algo_from_id(h, algo_id);
    CHECK_LT(hipblasLtMatmul(h, op,
                             &alpha,
                             dA, Ad,
                             dB, Bd,
                             &beta,
                             dD, Cd,
                             dD, Dd,
                             &algo,
                             workspace, workspaceSize,
                             stream));
    CHECK_HIP(hipStreamSynchronize(stream));
  };

  std::cout << "Running bad algo_id=" << algo_bad << "\n";
  run(algo_bad, dD_bad);
  std::cout << "Running good algo_id=" << algo_good << "\n";
  run(algo_good, dD_good);

  int* d_nan = nullptr;
  float* d_max = nullptr;
  CHECK_HIP(hipMalloc(&d_nan, sizeof(int)));
  CHECK_HIP(hipMalloc(&d_max, sizeof(float)));
  CHECK_HIP(hipMemset(d_nan, 0, sizeof(int)));
  float z = 0.0f;
  CHECK_HIP(hipMemcpy(d_max, &z, sizeof(float), hipMemcpyHostToDevice));

  int threads = 256;
  int blocks = (int)((elemsD + threads - 1) / threads);
  hipLaunchKernelGGL(diff_kernel, dim3(blocks), dim3(threads), 0, stream, dD_bad, dD_good, elemsD, d_nan, d_max);
  CHECK_HIP(hipStreamSynchronize(stream));

  int h_nan = 0;
  float h_max = 0.0f;
  CHECK_HIP(hipMemcpy(&h_nan, d_nan, sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(&h_max, d_max, sizeof(float), hipMemcpyDeviceToHost));

  std::cout << "nan_or_inf_count=" << h_nan << " max_abs_diff=" << h_max << "\n";
  for (int iter = 0; iter < 1000; iter++) {
    if (iter % 100 == 0) std::cout << "Iteration " << iter << "\n";
    
    CHECK_HIP(hipMemset(dD_bad,  0, elemsD * sizeof(*dD_bad)));
    run(algo_bad, dD_bad);
    
    CHECK_HIP(hipMemset(d_nan, 0, sizeof(int)));
    CHECK_HIP(hipMemcpy(d_max, &z, sizeof(float), hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(diff_kernel, dim3(blocks), dim3(threads), 0, stream, 
                      dD_bad, dD_good, elemsD, d_nan, d_max);
    CHECK_HIP(hipStreamSynchronize(stream));
    
    CHECK_HIP(hipMemcpy(&h_nan, d_nan, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(&h_max, d_max, sizeof(float), hipMemcpyDeviceToHost));
    
    if (h_nan != 0 || h_max > 1e-2f) {
      std::cout << "FAIL at iteration " << iter << ": nan_count=" << h_nan 
                << " max_diff=" << h_max << "\n";
      break;
    }
  }
  return (h_nan != 0 || h_max > 1e-2f) ? 3 : 0;
}