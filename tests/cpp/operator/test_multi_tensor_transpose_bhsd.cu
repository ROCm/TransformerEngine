/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/fused_attn.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

struct PermuteCase {
  size_t B, S, H, D_in, D_out;
};

// out[b][h][s][d] = in[b][s][h][d] (BSHD) or in[s][b][h][d] (SBHD), zero-padded
// past D_in when D_out is larger.
template <typename T>
void compute_ref(const T *in, T *out, const PermuteCase &c, bool is_bshd) {
  for (size_t b = 0; b < c.B; ++b) {
    for (size_t h = 0; h < c.H; ++h) {
      for (size_t s = 0; s < c.S; ++s) {
        const size_t in_off =
            (is_bshd ? ((b * c.S + s) * c.H + h) : ((s * c.B + b) * c.H + h)) * c.D_in;
        T *dst = out + ((b * c.H + h) * c.S + s) * c.D_out;
        for (size_t d = 0; d < c.D_out; ++d) {
          dst[d] = (d < c.D_in) ? in[in_off + d] : static_cast<T>(0);
        }
      }
    }
  }
}

// Shared-memory transpose needs 32*(32*D_pad+4) bytes; query the device budget.
template <typename T>
bool exceeds_smem_budget(const std::vector<PermuteCase> &cases) {
  size_t d_in_max = 0;
  for (const auto &c : cases) d_in_max = std::max(d_in_max, c.D_in);
  const size_t d_bytes = d_in_max * sizeof(T);
  if (d_bytes % 4 == 0) return false;  // vectorized path, no shared memory
  const size_t d_pad = (d_bytes + 3u) & ~size_t(3);
  const size_t needed = 32 * (32 * d_pad + 4);
  int max_smem = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  return needed > static_cast<size_t>(max_smem);
}

// D_in * sizeof(T) % 4 selects vectorized vs shared-memory path; TMA is CUDA-only.
template <typename T>
void performTest(const std::vector<PermuteCase> &cases, bool is_bshd, DType dtype) {
  if (exceeds_smem_budget<T>(cases)) {
    GTEST_SKIP() << "shared-memory transpose exceeds this device's shared memory";
  }
  const size_t n = cases.size();
  std::vector<std::unique_ptr<Tensor>> ins, outs;
  std::vector<NVTETensor> in_h, out_h;
  std::vector<std::vector<T>> refs(n);

  for (size_t i = 0; i < n; ++i) {
    const auto &c = cases[i];
    const std::vector<size_t> in_shape = is_bshd ? std::vector<size_t>{c.B, c.S, c.H, c.D_in}
                                                 : std::vector<size_t>{c.S, c.B, c.H, c.D_in};
    const std::vector<size_t> out_shape{c.B, c.H, c.S, c.D_out};

    auto in = std::make_unique<Tensor>("in_" + std::to_string(i), in_shape, dtype);
    auto out = std::make_unique<Tensor>("out_" + std::to_string(i), out_shape, dtype);
    fillUniform(in.get());
    // Poison output so unwritten positions cannot match the expected zero pad.
    NVTE_CHECK_CUDA(cudaMemset(out->rowwise_dptr(), 0xCD, c.B * c.H * c.S * c.D_out * sizeof(T)));

    in->to_cpu();
    refs[i].resize(c.B * c.H * c.S * c.D_out);
    compute_ref<T>(in->rowwise_cpu_dptr<T>(), refs[i].data(), c, is_bshd);

    in_h.push_back(in->data());
    out_h.push_back(out->data());
    ins.emplace_back(std::move(in));
    outs.emplace_back(std::move(out));
  }

  nvte_multi_tensor_transpose_to_bhsd(
      in_h.data(), out_h.data(), n,
      is_bshd ? NVTE_QKV_Format::NVTE_BSHD : NVTE_QKV_Format::NVTE_SBHD, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  NVTE_CHECK_CUDA(cudaGetLastError());

  // A pure layout permute, so byte equality is the right assertion.
  for (size_t i = 0; i < n; ++i) {
    outs[i]->to_cpu();
    compareResults("bhsd_" + std::to_string(i),
                   reinterpret_cast<const uint8_t *>(outs[i]->rowwise_cpu_dptr<T>()),
                   reinterpret_cast<const uint8_t *>(refs[i].data()), refs[i].size() * sizeof(T));
  }
}

class MultiTensorTransposeBhsdTestSuite
    : public ::testing::TestWithParam<std::tuple<std::vector<PermuteCase>, bool>> {};

TEST_P(MultiTensorTransposeBhsdTestSuite, TestFp16) {
  const auto cases = std::get<0>(GetParam());
  const bool is_bshd = std::get<1>(GetParam());
  performTest<fp16>(cases, is_bshd, DType::kFloat16);
}

TEST_P(MultiTensorTransposeBhsdTestSuite, TestByte) {
  const auto cases = std::get<0>(GetParam());
  const bool is_bshd = std::get<1>(GetParam());
  performTest<byte>(cases, is_bshd, DType::kByte);
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, MultiTensorTransposeBhsdTestSuite,
    ::testing::Combine(::testing::Values(
                           // Single tensor, D_in*elem a multiple of 4 -> vectorized fallback.
                           std::vector<PermuteCase>{{2, 64, 4, 64, 64}},
                           // Odd D_in with 1-byte elements: shared-memory transpose.
                           std::vector<PermuteCase>{{2, 48, 3, 33, 33}},
                           // D_out > D_in exercises the zero pad on the vectorized path.
                           std::vector<PermuteCase>{{1, 32, 2, 40, 64}},
                           // Same pad via the shared-memory transpose, within gfx942 budget.
                           std::vector<PermuteCase>{{1, 40, 2, 21, 32}},
                           // Multiple tensors per launch; B must match across the group.
                           std::vector<PermuteCase>{
                               {2, 64, 4, 64, 64}, {2, 33, 2, 16, 24}, {2, 16, 1, 8, 8}},
                           // S below one tile and H = 1, the degenerate grid.
                           std::vector<PermuteCase>{{1, 1, 1, 16, 16}}),
                       ::testing::Bool()),
    [](const testing::TestParamInfo<MultiTensorTransposeBhsdTestSuite::ParamType> &info) {
      return "case" + std::to_string(info.index) + "_n" +
             std::to_string(std::get<0>(info.param).size()) +
             (std::get<1>(info.param) ? "_bshd" : "_sbhd");
    });

}  // namespace
