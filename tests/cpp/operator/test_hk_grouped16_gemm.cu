/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// HipKittens BF16/FP16 grouped GEMM through nvte_multi_tensor_gemm (NVTE_USE_HIPKITTENS_GROUPED_GEMM=1):
// OpenMP CPU reference with atol + rtol, NaN-poisoned outputs, NaN guard rows past every activation and
// output, zero-row and unaligned experts, ragged weight dims, bias / dbias / accumulate, and bitwise
// run-to-run and grid-to-grid determinism. Fails if the call does not stay on HipKittens.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;

#ifdef __HIP_PLATFORM_AMD__

namespace {

enum Layout { TN, NN, NT };

constexpr size_t kGuard = 256;

uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

float u2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t to_bf16(float f) {
  const uint32_t u = f2u(f);
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// IEEE binary16, round to nearest even; |f| stays in the normal range here.
uint16_t to_fp16(float f) {
  const uint32_t u = f2u(f), s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
  if (a < 0x38800000u) {
    return static_cast<uint16_t>(s | static_cast<uint32_t>(std::nearbyint(u2f(a) * 16777216.0f)));
  }
  uint32_t h = (((a >> 23) - 112u) << 10) | ((a & 0x7fffffu) >> 13);
  const uint32_t rest = a & 0x1fffu;
  h += rest > 0x1000u || (rest == 0x1000u && (h & 1u));
  return static_cast<uint16_t>(s | h);
}

float from16(uint16_t h, DType t) {
  if (t == DType::kBFloat16) {
    return u2f(static_cast<uint32_t>(h) << 16);
  }
  const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  if (e == 0) {
    return (s ? -1.0f : 1.0f) * static_cast<float>(m) / 16777216.0f;
  }
  return u2f(s | ((e + 112u) << 23) | (m << 13));
}

// A device tensor of rows x cols elements of type t, followed by kGuard NaN rows when guarded; host copy in float.
struct Buf {
  DType t;
  size_t rows = 0, cols = 0;
  bool guard = false;
  void *dev = nullptr;
  std::vector<float> host;
  std::vector<uint8_t> init;

  Buf() = default;
  Buf(const Buf &) = delete;
  Buf &operator=(const Buf &) = delete;

  size_t esize() const { return t == DType::kFloat32 ? 4 : 2; }
  size_t bytes() const { return (rows + (guard ? kGuard : 0)) * cols * esize(); }

  void alloc(DType type, size_t r, size_t c, bool g) {
    t = type;
    rows = r;
    cols = c;
    guard = g;
    host.assign(rows * cols, 0.f);
    if (rows * cols > 0) {
      NVTE_CHECK_CUDA(cudaMalloc(&dev, bytes()));
    }
  }
  void poison() {
    if (dev) {
      NVTE_CHECK_CUDA(cudaMemset(dev, 0xff, bytes()));
    }
  }
  // Uniform(-scale, scale) values, rounded to t, into host and device (guard rows stay NaN).
  void fill(std::mt19937 &rng, float scale) {
    std::uniform_real_distribution<float> u(-scale, scale);
    init.resize(rows * cols * esize());
    for (size_t i = 0; i < rows * cols; i++) {
      const float v = u(rng);
      if (t == DType::kFloat32) {
        host[i] = v;
        std::memcpy(&init[i * 4], &v, 4);
      } else {
        const uint16_t h = t == DType::kBFloat16 ? to_bf16(v) : to_fp16(v);
        host[i] = from16(h, t);
        std::memcpy(&init[i * 2], &h, 2);
      }
    }
    restore();
  }
  // The filled values again, guard rows NaN.
  void restore() {
    poison();
    if (dev) {
      NVTE_CHECK_CUDA(cudaMemcpy(dev, init.data(), init.size(), cudaMemcpyHostToDevice));
    }
  }
  std::vector<uint8_t> download() const {
    std::vector<uint8_t> raw(dev ? bytes() : 0);
    if (dev) {
      NVTE_CHECK_CUDA(cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost));
    }
    return raw;
  }
  float at(const std::vector<uint8_t> &raw, size_t i) const {
    if (t == DType::kFloat32) {
      float v;
      std::memcpy(&v, &raw[i * 4], 4);
      return v;
    }
    uint16_t h;
    std::memcpy(&h, &raw[i * 2], 2);
    return from16(h, t);
  }
  ~Buf() {
    if (dev) {
      (void)cudaFree(dev);
    }
  }
};

struct Case {
  const char *label;
  Layout layout;
  size_t N, K;
  std::vector<size_t> m;
  DType in, out;
  bool bias, acc;
};

double tol_rel(DType out) {
  return out == DType::kBFloat16 ? 1.6e-2 : out == DType::kFloat16 ? 2e-3 : 1e-4;
}

// Restores the grouped-GEMM env vars on scope exit.
struct EnvGuard {
  static constexpr const char *kVars[] = {"NVTE_USE_HIPKITTENS_GROUPED_GEMM", "NVTE_USE_CK_GROUPED_GEMM",
                                          "NVTE_USE_CUTLASS_GROUPED_GEMM",
                                          "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"};
  std::vector<std::pair<bool, std::string>> saved;
  EnvGuard() {
    for (const char *v : kVars) {
      const char *x = std::getenv(v);
      saved.emplace_back(x != nullptr, x ? x : "");
    }
  }
  ~EnvGuard() {
    for (size_t i = 0; i < saved.size(); i++) {
      if (saved[i].first) {
        setenv(kVars[i], saved[i].second.c_str(), 1);
      } else {
        unsetenv(kVars[i]);
      }
    }
  }
};

void run_case(const Case &c) {
  cudaDeviceProp prop;
  NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  if (std::string(prop.gcnArchName).find("gfx950") == std::string::npos) {
    GTEST_SKIP() << "HipKittens BF16/FP16 grouped GEMM is built for gfx950";
  }
  EnvGuard env;
  setenv("NVTE_USE_HIPKITTENS_GROUPED_GEMM", "1", 1);
  unsetenv("NVTE_USE_CK_GROUPED_GEMM");
  setenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1", 1);

  const int E = static_cast<int>(c.m.size());
  std::mt19937 rng(12345);
  std::vector<Buf> A(E), B(E), D(E), bias(E);
  std::vector<std::vector<float>> d_init(E);
  for (int e = 0; e < E; e++) {
    const size_t m = c.m[e];
    if (c.layout == TN) {
      A[e].alloc(c.in, c.N, c.K, false);
      B[e].alloc(c.in, m, c.K, true);
      D[e].alloc(c.out, m, c.N, true);
    } else if (c.layout == NN) {
      A[e].alloc(c.in, c.N, c.K, false);
      B[e].alloc(c.in, m, c.N, true);
      D[e].alloc(c.out, m, c.K, true);
    } else {
      A[e].alloc(c.in, m, c.K, true);
      B[e].alloc(c.in, m, c.N, true);
      D[e].alloc(c.out, c.N, c.K, true);
    }
    A[e].fill(rng, 1.f);
    B[e].fill(rng, 1.f);
    if (c.bias) {
      bias[e].alloc(c.layout == NT ? c.out : c.in, c.N, 1, true);
      bias[e].fill(rng, 1.f);
    }
  }
  const size_t kred_tn_nn = c.layout == TN ? c.K : c.N;
  for (int e = 0; e < E; e++) {
    if (c.acc) {
      const size_t kred = c.layout == NT ? c.m[e] : kred_tn_nn;
      D[e].fill(rng, static_cast<float>(std::sqrt(std::max<size_t>(kred, 1)) / 3.0));
      d_init[e] = D[e].host;
    }
  }
  auto init_outputs = [&]() {
    for (int e = 0; e < E; e++) {
      if (c.acc) {
        D[e].restore();
      } else {
        D[e].poison();
      }
      if (c.bias && c.layout == NT) {
        bias[e].poison();
      }
    }
  };

  const int num_streams = nvte_get_num_compute_streams();
  const size_t ws_bytes = 33'554'432;
  std::vector<void *> ws_dev(num_streams);
  std::vector<TensorWrapper> ws_w;
  for (int s = 0; s < num_streams; s++) {
    NVTE_CHECK_CUDA(cudaMalloc(&ws_dev[s], ws_bytes));
    ws_w.emplace_back(ws_dev[s], std::vector<size_t>{ws_bytes}, DType::kByte);
  }
  std::vector<TensorWrapper> wA, wB, wD, wBias, wGelu;
  for (int e = 0; e < E; e++) {
    wA.emplace_back(A[e].dev, std::vector<size_t>{A[e].rows, A[e].cols}, c.in);
    wB.emplace_back(B[e].dev, std::vector<size_t>{B[e].rows, B[e].cols}, c.in);
    wD.emplace_back(D[e].dev, std::vector<size_t>{D[e].rows, D[e].cols}, c.out);
    if (c.bias) {
      wBias.emplace_back(bias[e].dev, std::vector<size_t>{c.N}, bias[e].t);
    } else {
      wBias.emplace_back(nullptr, std::vector<size_t>{0}, c.in);
    }
    wGelu.emplace_back(nullptr, std::vector<size_t>{0}, c.in);
  }
  std::vector<NVTETensor> pA, pB, pD, pBias, pGelu, pWs;
  for (int e = 0; e < E; e++) {
    pA.push_back(wA[e].data());
    pB.push_back(wB[e].data());
    pD.push_back(wD[e].data());
    pBias.push_back(wBias[e].data());
    pGelu.push_back(wGelu[e].data());
  }
  for (auto &w : ws_w) {
    pWs.push_back(w.data());
  }

  // One call with stderr captured; returns the HipKittens decline / fallback lines.
  auto call = [&](int cu_count) {
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    std::FILE *cap = std::tmpfile();
    dup2(fileno(cap), fileno(stderr));
    nvte_multi_tensor_gemm(pA.data(), pB.data(), pD.data(), pBias.data(), pGelu.data(), E,
                           c.layout == TN, c.layout == NT, /*grad=*/c.layout != TN, pWs.data(), c.acc,
                           /*use_split_accumulator=*/false, cu_count, 0);
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::string text;
    std::rewind(cap);
    char buf[1024];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) {
      text.append(buf, n);
    }
    std::fclose(cap);
    return text;
  };
  auto snapshot = [&]() {
    std::vector<std::vector<uint8_t>> s;
    for (int e = 0; e < E; e++) {
      s.push_back(D[e].download());
      if (c.bias && c.layout == NT) {
        s.push_back(bias[e].download());
      }
    }
    return s;
  };

  init_outputs();
  const std::string log = call(0);
  ASSERT_EQ(log.find("[HK-grouped16]"), std::string::npos) << "declined by HipKittens: " << log;
  ASSERT_EQ(log.find("Fallback"), std::string::npos) << log;
  const auto first = snapshot();

  const double rtol = tol_rel(c.out);
  for (int e = 0; e < E; e++) {
    const size_t m = c.m[e];
    const Buf &d = D[e];
    const std::vector<uint8_t> &raw = first[c.bias && c.layout == NT ? 2 * e : e];
    for (size_t i = d.rows * d.cols * d.esize(); i < raw.size(); i++) {
      ASSERT_EQ(raw[i], 0xff) << "guard row written, expert " << e;
    }
    std::vector<double> ref(d.rows * d.cols, 0.0);
    const std::vector<float> &a = A[e].host, &b = B[e].host;
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < d.rows; r++) {
      for (size_t col = 0; col < d.cols; col++) {
        double s = 0.0;
        if (c.layout == TN) {  // D[r][n] = sum_k X[r][k] W[n][k]
          for (size_t k = 0; k < c.K; k++) {
            s += double(b[r * c.K + k]) * a[col * c.K + k];
          }
        } else if (c.layout == NN) {  // D[r][k] = sum_n dY[r][n] W[n][k]
          for (size_t n = 0; n < c.N; n++) {
            s += double(b[r * c.N + n]) * a[n * c.K + col];
          }
        } else {  // D[n][k] = sum_t dY[t][n] X[t][k]
          for (size_t t = 0; t < m; t++) {
            s += double(b[t * c.N + r]) * a[t * c.K + col];
          }
        }
        if (c.acc) {
          s += d_init[e][r * d.cols + col];
        }
        if (c.bias && c.layout == TN) {
          s += bias[e].host[col];
        }
        ref[r * d.cols + col] = s;
      }
    }
    const double atol = rtol * std::sqrt(double(std::max<size_t>(c.layout == NT ? m : kred_tn_nn, 1))) / 3.0;
    size_t bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
      const double got = d.at(raw, i), err = std::fabs(got - ref[i]);
      const double lim = atol + rtol * std::fabs(ref[i]);
      worst = std::max(worst, std::isfinite(got) ? err / lim : INFINITY);
      bad += !(err <= lim);
    }
    EXPECT_EQ(bad, 0u) << "expert " << e << " (m = " << m << "): " << bad << " of " << ref.size()
                       << " elements out of tolerance, worst err/tol " << worst;
    if (c.bias && c.layout == NT) {
      const std::vector<uint8_t> &rb = first[2 * e + 1];
      for (size_t i = c.N * bias[e].esize(); i < rb.size(); i++) {
        ASSERT_EQ(rb[i], 0xff) << "dbias guard written, expert " << e;
      }
      size_t bad_b = 0;
      const double atol_b = rtol * std::sqrt(double(std::max<size_t>(m, 1)));
      for (size_t n = 0; n < c.N; n++) {
        double s = 0.0;
        for (size_t t = 0; t < m; t++) {
          s += b[t * c.N + n];
        }
        bad_b += !(std::fabs(bias[e].at(rb, n) - s) <= atol_b + rtol * std::fabs(s));
      }
      EXPECT_EQ(bad_b, 0u) << "dbias of expert " << e << ": " << bad_b << " of " << c.N << " out of tolerance";
    }
  }

  // Determinism: same call on a narrow grid (stealing), then on the full grid again.
  init_outputs();
  ASSERT_EQ(call(7).find("[HK-grouped16]"), std::string::npos);
  EXPECT_TRUE(snapshot() == first) << "narrow-grid result differs bitwise";
  init_outputs();
  ASSERT_EQ(call(0).find("[HK-grouped16]"), std::string::npos);
  EXPECT_TRUE(snapshot() == first) << "second run differs bitwise";
  for (void *p : ws_dev) {
    NVTE_CHECK_CUDA(cudaFree(p));
  }
}

const std::vector<size_t> kSplits = {0, 1, 255, 256, 257, 300, 1000, 0, 63, 64, 65, 513};

std::vector<Case> make_cases() {
  std::vector<Case> v;
  const char *names[3] = {"TN", "NN", "NT"};
  for (Layout l : {TN, NN, NT}) {
    for (DType in : {DType::kBFloat16, DType::kFloat16}) {
      for (DType out : {in, DType::kFloat32}) {
        for (int bias = 0; bias < 2; bias++) {
          if (l == NN && bias) {
            continue;
          }
          for (int acc = 0; acc < 2; acc++) {
            v.push_back({names[l], l, 512, 768, kSplits, in, out, bool(bias), bool(acc)});
          }
        }
      }
    }
  }
  // Ragged weight dims: the reduction one a multiple of 128 only, the others of 8.
  v.push_back({"TN_ragged", TN, 328, 640, kSplits, DType::kBFloat16, DType::kBFloat16, true, false});
  v.push_back({"TN_ragged", TN, 328, 640, kSplits, DType::kBFloat16, DType::kFloat32, false, true});
  v.push_back({"NN_ragged", NN, 640, 328, kSplits, DType::kBFloat16, DType::kBFloat16, false, false});
  v.push_back({"NN_ragged", NN, 640, 328, kSplits, DType::kFloat16, DType::kFloat32, false, true});
  v.push_back({"NT_ragged", NT, 328, 200, kSplits, DType::kBFloat16, DType::kBFloat16, true, false});
  v.push_back({"NT_ragged", NT, 328, 200, kSplits, DType::kBFloat16, DType::kFloat32, true, true});
  // An odd count of 64-wide k-steps in the reduction dim (TN: K, NN: N): the drain's extra step is skipped.
  for (size_t red : {2880, 192, 64}) {
    for (DType in : {DType::kBFloat16, DType::kFloat16}) {
      for (int bias = 0; bias < 2; bias++) {
        v.push_back({"TN_k64odd", TN, 512, red, kSplits, in, in, bool(bias), false});
      }
      v.push_back({"NN_k64odd", NN, red, 512, kSplits, in, in, false, false});
    }
  }
  // More experts than one launch holds.
  std::vector<size_t> many(70);
  for (size_t i = 0; i < many.size(); i++) {
    many[i] = (i * 2654435761u) % 400;
  }
  v.push_back({"TN_70", TN, 256, 512, many, DType::kBFloat16, DType::kBFloat16, false, false});
  v.push_back({"NN_70", NN, 256, 512, many, DType::kBFloat16, DType::kBFloat16, false, false});
  v.push_back({"NT_70", NT, 256, 512, many, DType::kBFloat16, DType::kFloat32, false, true});
  return v;
}

std::string case_name(const testing::TestParamInfo<Case> &info) {
  const Case &c = info.param;
  auto dt = [](DType t) { return t == DType::kBFloat16 ? "bf16" : t == DType::kFloat16 ? "fp16" : "fp32"; };
  return std::string(c.label) + "_" + dt(c.in) + "_" + dt(c.out) + (c.bias ? "_bias" : "") +
         (c.acc ? "_acc" : "") + "_" + std::to_string(c.N) + "x" + std::to_string(c.K) + "_" +
         std::to_string(info.index);
}

}  // namespace

class HKGrouped16TestSuite : public ::testing::TestWithParam<Case> {};

TEST_P(HKGrouped16TestSuite, MatchesCpuReference) { run_case(GetParam()); }

INSTANTIATE_TEST_SUITE_P(OperatorTest, HKGrouped16TestSuite, ::testing::ValuesIn(make_cases()), case_name);

#endif  // __HIP_PLATFORM_AMD__
