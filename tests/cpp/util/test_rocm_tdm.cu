/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "util/rocm_tdm.cuh"
#include "../test_common.h"

namespace {

constexpr uint32_t kWaves = 4;
constexpr uint32_t kTileRows = 8;
constexpr uint32_t kTileCols = 64;

// Each wave loads its own tile into LDS, dumps it, adds 1 and stores it back.
template <typename T>
__global__ void __launch_bounds__(kWaves * 32)
    tdm_roundtrip_kernel(const T *in, T *out, T *lds_dump, uint32_t rows, uint32_t cols,
                         uint32_t ld_in, uint32_t ld_out) {
#ifdef NVTE_ROCM_HAS_TDM
  __shared__ alignas(128) T tile[kWaves][kTileRows][kTileCols];
  const uint32_t wave = threadIdx.x / 32;
  const uint32_t row = (blockIdx.y * kWaves + wave) * kTileRows;
  const uint32_t col = blockIdx.x * kTileCols;
  transformer_engine::tdm::load_2d(tile[wave], in, rows, cols, ld_in, row, col);
  transformer_engine::tdm::wait<0>();
  T *dump = lds_dump + ((blockIdx.y * gridDim.x + blockIdx.x) * kWaves + wave) * kTileRows * kTileCols;
  for (uint32_t i = threadIdx.x % 32; i < kTileRows * kTileCols; i += 32) {
    T &v = tile[wave][i / kTileCols][i % kTileCols];
    dump[i] = v;
    v = v + 1;
  }
  transformer_engine::tdm::store_2d(out, tile[wave], rows, cols, ld_out, row, col);
  transformer_engine::tdm::wait<0>();
#endif
}

template <typename T>
void run_roundtrip(uint32_t rows, uint32_t cols, uint32_t ld_in, uint32_t ld_out) {
  SCOPED_TRACE(testing::Message() << "rows=" << rows << " cols=" << cols << " ld_in=" << ld_in
                                  << " ld_out=" << ld_out);
  const dim3 grid((cols + kTileCols - 1) / kTileCols,
                  (rows + kWaves * kTileRows - 1) / (kWaves * kTileRows));
  // Every row a tile can touch, so stray stores are detected.
  const size_t out_rows = grid.y * kWaves * kTileRows;
  const T sentinel = static_cast<T>(0xA5A5A5A5A5A5A5A5ull);
  const T pad = static_cast<T>(0xEEEEEEEEEEEEEEEEull);

  std::vector<T> h_in(static_cast<size_t>(rows) * ld_in);
  std::vector<T> h_out(out_rows * ld_out, sentinel);
  std::vector<T> h_dump(static_cast<size_t>(grid.x) * grid.y * kWaves * kTileRows * kTileCols);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < ld_in; ++c) {
      h_in[r * ld_in + c] = c < cols ? static_cast<T>((r * 37 + c * 11) % 150 + 1) : pad;
    }
  }

  T *d_in, *d_out, *d_dump;
  NVTE_CHECK_CUDA(cudaMalloc(&d_in, h_in.size() * sizeof(T)));
  NVTE_CHECK_CUDA(cudaMalloc(&d_out, h_out.size() * sizeof(T)));
  NVTE_CHECK_CUDA(cudaMalloc(&d_dump, h_dump.size() * sizeof(T)));
  NVTE_CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(T), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(d_out, h_out.data(), h_out.size() * sizeof(T), cudaMemcpyHostToDevice));
  tdm_roundtrip_kernel<T><<<grid, kWaves * 32>>>(d_in, d_out, d_dump, rows, cols, ld_in, ld_out);
  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  NVTE_CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(T), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(h_dump.data(), d_dump, h_dump.size() * sizeof(T), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaFree(d_in));
  NVTE_CHECK_CUDA(cudaFree(d_out));
  NVTE_CHECK_CUDA(cudaFree(d_dump));

  size_t load_mismatches = 0;
  for (size_t tile = 0; tile < h_dump.size() / (kTileRows * kTileCols); ++tile) {
    const size_t block = tile / kWaves, wave = tile % kWaves;
    const size_t row0 = ((block / grid.x) * kWaves + wave) * kTileRows;
    const size_t col0 = (block % grid.x) * kTileCols;
    for (size_t i = 0; i < kTileRows * kTileCols; ++i) {
      const size_t r = row0 + i / kTileCols, c = col0 + i % kTileCols;
      const T expected = (r < rows && c < cols) ? h_in[r * ld_in + c] : T(0);
      load_mismatches += h_dump[tile * kTileRows * kTileCols + i] != expected;
    }
  }
  size_t store_mismatches = 0;
  for (size_t r = 0; r < out_rows; ++r) {
    for (size_t c = 0; c < ld_out; ++c) {
      const T expected =
          (r < rows && c < cols) ? static_cast<T>(h_in[r * ld_in + c] + 1) : sentinel;
      store_mismatches += h_out[r * ld_out + c] != expected;
    }
  }
  EXPECT_EQ(load_mismatches, 0u) << "LDS contents after load_2d (out of bounds must read 0)";
  EXPECT_EQ(store_mismatches, 0u) << "global contents after store_2d (out of bounds untouched)";
}

}  // namespace

template <typename T>
class RocmTdmTest : public ::testing::Test {};

using TdmElementTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(RocmTdmTest, TdmElementTypes);

TYPED_TEST(RocmTdmTest, RoundTrip2D) {
  if (test::getDeviceComputeCapability() != 125) {
    GTEST_SKIP() << "TDM requires gfx1250";
  }
  run_roundtrip<TypeParam>(64, 128, 128, 128);  // whole tiles
  run_roundtrip<TypeParam>(100, 200, 256, 216);  // partial tiles, row pitch != cols
  run_roundtrip<TypeParam>(1, 1, 16, 16);
}
