/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef AITER_GEMM_H
#define AITER_GEMM_H

#include <cstdint>
#include <hip/hip_runtime.h>

// TE-side wrapper around AITER's torch-free a4w4 (FP4 x FP4) GEMM kernels.
//
// This header is intentionally free of any AITER / QoLA headers so that
// libtransformer_engine.so can consume it without pulling in the AITER kernel
// headers.  The translation to AITER's aiter_tensor_t POD lives in the .cpp.
//
// Kernel selection (tuned-CSV lookup) and weight/scale pre-shuffling are the
// caller's responsibility -- these entry points are thin executors that take a
// resolved kernel name and already-shuffled inputs.
namespace aiter_gemm {

// Mirrors AiterDtype in AITER's aiter_enum.h (only the subset a4w4 needs).
enum class DType {
  fp4x2 = 0,  /*!< two packed FP4 (E2M1) values per byte */
  e8m0  = 1,  /*!< 8-bit exponent-only microscaling factor (1 byte) */
  bf16  = 2,
  fp16  = 3,
  fp32  = 4,
  u8    = 5,
  i8    = 6,
};

// Lightweight tensor descriptor (raw device pointer + layout).  The caller owns
// the storage; the descriptor must outlive the call but not the storage.
struct TensorDesc {
  void* ptr = nullptr;
  int ndim = 0;
  int64_t shape[8] = {0};
  int64_t strides[8] = {0};
  DType dtype = DType::fp4x2;
  int device_id = 0;
};

// CK blockscale a4w4 GEMM: Y = XQ @ WQ^T with per-1x32 microscaling.
//   XQ      [M, K/2]  fp4x2
//   WQ      [N, K/2]  fp4x2
//   x_scale [M, K/32] e8m0
//   w_scale [N, K/32] e8m0
//   Y       [M, N]    bf16 / fp16   (output, pre-allocated)
// `kernel_name` empty -> default heuristic; non-empty must exist in the
// compiled registry.  Returns hipSuccess on success, hipErrorUnknown on a
// kernel-side failure (message logged).
hipError_t gemm_a4w4_blockscale(const TensorDesc& XQ,
                                const TensorDesc& WQ,
                                const TensorDesc& x_scale,
                                const TensorDesc& w_scale,
                                const TensorDesc& Y,
                                int split_k,
                                const char* kernel_name,
                                hipStream_t stream);

// ASM (f4gemm) a4w4 GEMM: D = alpha*A*B + beta*C.
//   A/B/scales/out layout as above; `bias` may be null.
//   `kernel_name` empty -> ASM heuristic.
hipError_t gemm_a4w4_asm(const TensorDesc& A,
                         const TensorDesc& B,
                         const TensorDesc& a_scale,
                         const TensorDesc& b_scale,
                         const TensorDesc& out,
                         const TensorDesc* bias,
                         const char* kernel_name,
                         float alpha,
                         float beta,
                         int bpreshuffle,
                         int log2_k_split,
                         hipStream_t stream);

}  // namespace aiter_gemm

#endif  // AITER_GEMM_H
