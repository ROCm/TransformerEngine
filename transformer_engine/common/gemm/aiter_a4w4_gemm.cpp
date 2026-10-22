/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// AITER a4w4 (MXFP4) GEMM.  The kernels live in QoLA-built shared objects that
// export a C ABI over qola_tensor_t; this file translates TE tensors into that
// descriptor and picks the CK or ASM backend.
//
// When TE is built without the AITER a4w4 backend (USE_AITER_GEMM undefined
// -- e.g. non-gfx950), the entry point still exists but throws.

#include <transformer_engine/aiter_gemm.h>

#include "../common.h"

#ifdef USE_AITER_GEMM

#include <string>

#include "../util/cuda_runtime.h"
#include "qola_gemm_a4w4.h"

namespace transformer_engine {
namespace {

// Size of the buffer handed to QoLA for failure messages.
constexpr size_t kQolaErrLen = 512;

int to_qola_dtype(const DType dtype) {
  switch (dtype) {
    case DType::kFloat4E2M1:
      return QOLA_DTYPE_FP4X2;
    case DType::kFloat8E8M0:
      return QOLA_DTYPE_E8M0;
    case DType::kBFloat16:
      return QOLA_DTYPE_BF16;
    case DType::kFloat16:
      return QOLA_DTYPE_FP16;
    default:
      NVTE_ERROR("AITER a4w4 GEMM: unsupported dtype ", to_string(dtype));
  }
}

// Describe a contiguous buffer as a 2D qola_tensor_t.  FP4 is stored two
// values per byte, so its last dimension is halved to count packed elements.
qola_tensor_t make_qola_tensor(const SimpleTensor &t, const size_t rows, size_t cols,
                               const int device_id) {
  if (t.dtype == DType::kFloat4E2M1) {
    NVTE_CHECK(cols % 2 == 0, "AITER a4w4 GEMM: FP4 last dimension must be even (got ", cols, ")");
    cols /= 2;
  }
  qola_tensor_t q{};
  q.ptr        = t.dptr;
  q.ndim       = 2;
  q.dtype      = to_qola_dtype(t.dtype);
  q.device_id  = device_id;
  q.shape[0]   = static_cast<int64_t>(rows);
  q.shape[1]   = static_cast<int64_t>(cols);
  q.strides[0] = static_cast<int64_t>(cols);
  q.strides[1] = 1;
  return q;
}

qola_tensor_t make_qola_data(const Tensor &t, const int device_id) {
  return make_qola_tensor(t.data, t.flat_first_dim(), t.flat_last_dim(), device_id);
}

qola_tensor_t make_qola_scale(const Tensor &t, const int device_id) {
  NVTE_CHECK(t.scale_inv.shape.size() == 2, "AITER a4w4 GEMM: scale_inv must be 2D");
  return make_qola_tensor(t.scale_inv, t.scale_inv.shape[0], t.scale_inv.shape[1], device_id);
}

void check_mxfp4_operand(const Tensor &t, const char *name) {
  NVTE_CHECK(t.scaling_mode == NVTE_MXFP4_1D_SCALING, "AITER a4w4 GEMM: ", name,
             " must use MXFP4 scaling (got ", to_string(t.scaling_mode), ")");
  NVTE_CHECK(t.has_data() && t.data.dtype == DType::kFloat4E2M1, "AITER a4w4 GEMM: ", name,
             " must have FP4 row-wise data");
  NVTE_CHECK(t.scale_inv.dptr != nullptr && t.scale_inv.dtype == DType::kFloat8E8M0,
             "AITER a4w4 GEMM: ", name, " must have E8M0 row-wise scale_inv");
}

}  // namespace
}  // namespace transformer_engine

#endif  // USE_AITER_GEMM

void nvte_aiter_gemm_a4w4(const NVTETensor A, const NVTETensor B, NVTETensor D,
                          const char *kernel_name, int split_k, cudaStream_t stream) {
  NVTE_API_CALL(nvte_aiter_gemm_a4w4);
#ifdef USE_AITER_GEMM
  using namespace transformer_engine;
  const Tensor &a = *convertNVTETensorCheck(A);
  const Tensor &b = *convertNVTETensorCheck(B);
  const Tensor &d = *convertNVTETensorCheck(D);
  check_mxfp4_operand(a, "A");
  check_mxfp4_operand(b, "B");
  NVTE_CHECK(d.data.dtype == DType::kBFloat16 || d.data.dtype == DType::kFloat16,
             "AITER a4w4 GEMM: output must be BF16 or FP16");

  const int device_id = cuda::current_device();
  const qola_tensor_t a_data  = make_qola_data(a, device_id);
  const qola_tensor_t b_data  = make_qola_data(b, device_id);
  const qola_tensor_t a_scale = make_qola_scale(a, device_id);
  const qola_tensor_t b_scale = make_qola_scale(b, device_id);
  const qola_tensor_t out     = make_qola_tensor(d.data, d.flat_first_dim(), d.flat_last_dim(), device_id);

  // Same routing as AITER's gemm_a4w4: ASM kernel names are mangled C++
  // symbols, CK kernel names are not.  The tuned split-K value is already
  // expressed the way the selected backend expects it.
  const std::string name = kernel_name != nullptr ? kernel_name : "";
  const bool use_ck = !name.empty() && name.find("_ZN") == std::string::npos;

  char err[kQolaErrLen] = {};
  int rc;
  if (use_ck) {
    rc = QOLA_C(gemm_a4w4_blockscale)(&a_data, &b_data, &a_scale, &b_scale, &out, split_k,
                                      name.c_str(), stream, err, sizeof(err));
  } else {
    rc = QOLA_C(gemm_a4w4_asm)(&a_data, &b_data, &a_scale, &b_scale, &out, /*bias=*/nullptr,
                               name.c_str(), /*alpha=*/1.0f, /*beta=*/0.0f, /*bpreshuffle=*/1,
                               /*log2_k_split=*/split_k, stream, err, sizeof(err));
  }
  NVTE_CHECK(rc == 0, "AITER a4w4 ", use_ck ? "CK blockscale" : "ASM", " GEMM failed (rc=", rc,
             "): ", err);
#else
  NVTE_ERROR("TransformerEngine was built without the AITER a4w4 GEMM backend");
#endif  // USE_AITER_GEMM
}
