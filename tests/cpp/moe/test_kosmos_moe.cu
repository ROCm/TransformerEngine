/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

// Tests of the fused expert-parallel MoE C API (nvte_rocm_moe_*, KOSMOS kernels) on 8 GPUs without MPI: every TEST
// forks one process per GPU (before any HIP call in the gtest process; hipIpc handles do not work within one
// process), and the ranks' allgather runs over a shared-memory segment the parent maps before forking.

#include <omp.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/moe.h>
#include <transformer_engine/transformer_engine.h>

#include "kosmos_moe/moe/moe.h"

// The kernels' bounded waits, with a short bound for WaitBoundTraps (this test's own kernel only).
#define MOE_WAIT_LIMIT_S 3
#include "kosmos_moe/moe/moe_wait.cuh"

namespace {

using transformer_engine::DType;
using transformer_engine::TensorWrapper;

// ---------------------------------------------------------------------------------------------------------------------
// Problem size and configuration
// ---------------------------------------------------------------------------------------------------------------------

constexpr int      kRanks    = NVTE_ROCM_MOE_EP_SIZE;
constexpr int      kH        = 2048;
constexpr int      kI        = 256;
constexpr int      kEloc     = 2;
constexpr int      kE        = kRanks * kEloc;
constexpr int      kK        = 8;
constexpr int      kT        = 1024;
constexpr int      kWgcap    = 240;
constexpr int      kCapacity = kT * kK + 256 * kEloc;
// MXFP8 with I < 1024 pads every (source, expert) segment to 32 rows: room for the call in one pass.
constexpr int      kCapacityMx = 2 * kT * kK;
constexpr float    kWStd     = 0.05f;
constexpr float    kAuxCoeff = 1e-2f;
constexpr uint64_t kSeed     = 20261124;

constexpr double kRelTol = 2e-2;  // max|c - r| / max|r| (bf16)
constexpr double kAuxTol = 1e-3;  // aux loss, relative
constexpr double kMinSnr = 20.0;  // dB (MXFP8 vs the bf16 reference)

constexpr int kTestTimeoutS  = 1800;  // whole test, all ranks
constexpr int kAfterFailureS = 300;   // the other ranks, once one has failed

constexpr int kExitPass      = 0;
constexpr int kExitFail      = 1;
constexpr int kExitException = 2;
constexpr int kExitSkip      = 77;

NVTERocmMoeConfig make_config(bool mxfp8) {
  NVTERocmMoeConfig c = NVTE_ROCM_MOE_CONFIG_INIT;
  c.hidden_size              = kH;
  c.ffn_hidden_size          = kI;
  c.num_local_experts        = kEloc;
  c.topk                     = kK;
  c.num_tokens               = kT;
  c.wgcap                    = kWgcap;
  c.capacity_rows            = mxfp8 ? kCapacityMx : kCapacity;
  c.training                 = 1;
  c.activation               = kNVTERocmMoeSwiGLU;
  c.mxfp8                    = mxfp8 ? 1 : 0;
  c.shared_expert            = 1;
  c.router                   = 1;
  c.router_config.score      = kNVTERocmMoeRouterSoftmax;
  c.router_config.aux_groups = 1;
  c.router_config.aux_coeff  = kAuxCoeff;
  c.router_config.scale      = 1.0f;
  c.pool_bytes               = (size_t)4 * c.capacity_rows * 2 * kH * 4;
  return c;
}

// ---------------------------------------------------------------------------------------------------------------------
// Shared memory between the ranks: the allgather slots, a barrier, an abort flag and each rank's report
// ---------------------------------------------------------------------------------------------------------------------

constexpr size_t kSlotBytes = 1 << 20;
constexpr size_t kMsgBytes  = 16384;

static_assert(std::atomic<int>::is_always_lock_free, "the cross-process barrier needs lock-free atomics");

struct Shm {
  std::atomic<int> abort;
  std::atomic<int> bar_count;
  std::atomic<int> bar_sense;
  int nranks;
  char msg[kRanks][kMsgBytes];
  alignas(64) char slots[kRanks][kSlotBytes];
};

Shm *shm_create(int nranks) {
  void *p = mmap(nullptr, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return nullptr;
  }
  Shm *s = new (p) Shm;
  s->abort.store(0);
  s->bar_count.store(0);
  s->bar_sense.store(0);
  s->nranks = nranks;
  std::memset(s->msg, 0, sizeof(s->msg));
  return s;
}

void shm_destroy(Shm *s) {
  s->~Shm();
  munmap(s, sizeof(Shm));
}

// One rank's process: its device, stream, EP communicator and findings.
struct Rank {
  int rank = 0;
  Shm *shm = nullptr;
  int sense = 0;
  NVTERocmMoeComm comm = NVTE_ROCM_MOE_COMM_INIT;
  cudaStream_t stream = nullptr;
  bool skip = false;
  std::vector<std::string> fails;
  std::string info;

  void fail(const std::string &s) { fails.push_back(s); }
  void note(const std::string &s) { info += (info.empty() ? "" : "; ") + s; }
};

// Sense-reversing barrier over the shared segment; false when another rank aborted.
bool barrier(Rank *r) {
  Shm *s = r->shm;
  r->sense = !r->sense;
  if (s->bar_count.fetch_add(1) == s->nranks - 1) {
    s->bar_count.store(0);
    s->bar_sense.store(r->sense);
    return true;
  }
  const timespec nap = {0, 20000};
  while (s->bar_sense.load() != r->sense) {
    if (s->abort.load()) {
      return false;
    }
    nanosleep(&nap, nullptr);
  }
  return true;
}

// NVTERocmMoeComm::allgather: each rank writes its bytes into its slot, then copies every slot out (in slot-sized
// chunks).
int shm_allgather(void *ctx, const void *send, void *recv, size_t bytes) {
  Rank *r = static_cast<Rank *>(ctx);
  Shm *s = r->shm;
  for (size_t off = 0; off < bytes; off += kSlotBytes) {
    const size_t n = std::min(kSlotBytes, bytes - off);
    std::memcpy(s->slots[r->rank], static_cast<const char *>(send) + off, n);
    if (!barrier(r)) {
      return 1;
    }
    for (int q = 0; q < s->nranks; q++) {
      std::memcpy(static_cast<char *>(recv) + q * bytes + off, s->slots[q], n);
    }
    if (!barrier(r)) {
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Forking the ranks
// ---------------------------------------------------------------------------------------------------------------------

struct RankOutcome {
  int code = -1;
  std::string what;
};

using RankBody = std::function<void(Rank &)>;

int child_main(int rank, Shm *shm, const RankBody &body) {
  Rank r;
  r.rank        = rank;
  r.shm         = shm;
  r.comm.ctx    = &r;
  r.comm.rank   = rank;
  r.comm.nranks = shm->nranks;
  r.comm.allgather = shm_allgather;
  omp_set_num_threads(std::max(1, omp_get_num_procs() / kRanks));
  int code = kExitPass;
  try {
    body(r);
    code = r.skip ? kExitSkip : r.fails.empty() ? kExitPass : kExitFail;
  } catch (const std::exception &e) {
    shm->abort.store(1);
    r.fail(std::string("exception: ") + e.what());
    code = kExitException;
  } catch (...) {
    shm->abort.store(1);
    r.fail("unknown exception");
    code = kExitException;
  }
  if (r.stream) {
    cudaStreamDestroy(r.stream);
  }
  std::string m;
  for (const std::string &f : r.fails) {
    m += "FAIL " + f + "\n";
  }
  m += r.info;
  std::snprintf(shm->msg[rank], kMsgBytes, "%s", m.c_str());
  return code;
}

// Runs body on nranks forked processes (rank r is child r) and collects their exit codes and reports. The parent does
// no HIP call. Once a rank fails the others get kAfterFailureS to finish before they are killed.
std::vector<RankOutcome> run_ranks(int nranks, const RankBody &body) {
  std::vector<RankOutcome> res(nranks);
  Shm *shm = shm_create(nranks);
  if (!shm) {
    for (RankOutcome &o : res) {
      o.what = "mmap of the shared segment failed";
    }
    return res;
  }
  std::fflush(nullptr);
  std::vector<pid_t> pid(nranks, -1);
  using clock = std::chrono::steady_clock;
  clock::time_point kill_at = clock::now() + std::chrono::seconds(kTestTimeoutS);
  for (int r = 0; r < nranks; r++) {
    pid[r] = fork();
    if (pid[r] == 0) {
      _exit(child_main(r, shm, body));
    }
    if (pid[r] < 0) {
      res[r].what = "fork failed";
      shm->abort.store(1);
      kill_at = clock::now();
      break;
    }
  }
  std::vector<bool> done(nranks, false), killed(nranks, false);
  int alive = 0;
  for (int r = 0; r < nranks; r++) {
    alive += pid[r] > 0;
    done[r] = pid[r] <= 0;
  }
  while (alive > 0) {
    for (int r = 0; r < nranks; r++) {
      int st = 0;
      if (done[r] || waitpid(pid[r], &st, WNOHANG) != pid[r]) {
        continue;
      }
      done[r] = true;
      alive--;
      if (WIFEXITED(st)) {
        res[r].code = WEXITSTATUS(st);
      } else if (WIFSIGNALED(st)) {
        res[r].what = std::string(killed[r] ? "killed after the timeout" : "died on signal ") +
                      (killed[r] ? "" : std::to_string(WTERMSIG(st)));
      }
      if (res[r].code != kExitPass && res[r].code != kExitSkip) {
        shm->abort.store(1);
        kill_at = std::min(kill_at, clock::now() + std::chrono::seconds(kAfterFailureS));
      }
    }
    if (alive > 0 && clock::now() >= kill_at) {
      for (int r = 0; r < nranks; r++) {
        if (!done[r] && !killed[r]) {
          kill(pid[r], SIGKILL);
          killed[r] = true;
        }
      }
    }
    usleep(20000);
  }
  for (int r = 0; r < nranks; r++) {
    res[r].what += (res[r].what.empty() ? "" : "\n") + std::string(shm->msg[r]);
  }
  shm_destroy(shm);
  return res;
}

void expect_all_ranks_pass(const std::vector<RankOutcome> &res) {
  for (size_t r = 0; r < res.size(); r++) {
    if (res[r].code == kExitPass) {
      std::printf("[  rank %zu  ] %s\n", r, res[r].what.c_str());
    } else {
      ADD_FAILURE() << "rank " << r << " (exit code " << res[r].code << "):\n" << res[r].what;
    }
  }
}

// Empty when the fused MoE can run here: 8 gfx950 devices and nvte_rocm_moe_is_supported(); probed in a forked child.
const std::string &skip_reason() {
  static const std::string why = [] {
    const std::vector<RankOutcome> res = run_ranks(1, [](Rank &r) {
      int n = 0;
      if (cudaGetDeviceCount(&n) != cudaSuccess || n < kRanks) {
        r.skip = true;
        r.note("needs " + std::to_string(kRanks) + " devices, found " + std::to_string(n));
        return;
      }
      for (int d = 0; d < kRanks; d++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess || std::strncmp(prop.gcnArchName, "gfx950", 6) != 0) {
          r.skip = true;
          r.note("device " + std::to_string(d) + " is not gfx950");
          return;
        }
      }
      if (cudaSetDevice(0) != cudaSuccess || !nvte_rocm_moe_is_supported()) {
        r.skip = true;
        r.note("nvte_rocm_moe_is_supported() is 0");
      }
    });
    if (res[0].code == kExitPass) {
      return std::string();
    }
    return "fused MoE unavailable: " + res[0].what;
  }();
  return why;
}

#define SKIP_IF_UNSUPPORTED()                \
  do {                                       \
    const std::string &why_ = skip_reason(); \
    if (!why_.empty()) {                     \
      GTEST_SKIP() << why_;                  \
    }                                        \
  } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// Host data: deterministic N(0, 1) draws by (seed, tensor, rank, index), rounded to bf16
// ---------------------------------------------------------------------------------------------------------------------

uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

enum Tag { kTagX = 1, kTagDout, kTagW1, kTagW2, kTagWr, kTagWs1, kTagWs2 };
constexpr int kAllRanks = -1;  // tensors replicated on every rank

uint64_t stream_key(uint64_t seed, int tag, int rank) {
  return mix64(mix64(seed) ^ mix64(((uint64_t)tag << 32) | (uint32_t)rank));
}

float normal_at(uint64_t key, uint64_t i) {
  const uint64_t h  = mix64(key ^ mix64(i));
  const double   u1 = (double)((h >> 40) + 1) / 16777216.0;
  const double   u2 = (double)((h >> 16) & 0xFFFFFF) / 16777216.0;
  return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
}

uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
    return 0x7FC0;
  }
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

float bf2f(uint16_t b) {
  const uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A bf16 tensor: its bits (uploaded) and their values (the reference's input).
struct HostMat {
  std::vector<uint16_t> b;
  std::vector<float> f;
};

HostMat gen(size_t n, uint64_t key, float sd) {
  HostMat m;
  m.b.resize(n);
  m.f.resize(n);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < (int64_t)n; i++) {
    m.b[i] = f2bf(sd * normal_at(key, (uint64_t)i));
    m.f[i] = bf2f(m.b[i]);
  }
  return m;
}

struct RankData {
  HostMat x;     // [T][H]
  HostMat dout;  // [T][H]
  HostMat w1;    // [E_loc][2I][H], KOSMOS layout: per 256-row block 128 gate rows, then the matching 128 up rows
  HostMat w2;    // [E_loc][H][I]
};

struct Problem {
  HostMat wr;   // [E][H], identical on every rank
  HostMat ws1;  // [2I][H], [gate; up], identical on every rank
  HostMat ws2;  // [H][I], identical on every rank
  std::vector<RankData> rk;
};

// The inputs of every rank (all_ranks; the reference needs other ranks' tokens and experts) or only rank's.
Problem make_problem(uint64_t seed, int rank, bool all_ranks) {
  Problem p;
  p.wr  = gen((size_t)kE * kH, stream_key(seed, kTagWr, kAllRanks), kWStd);
  p.ws1 = gen((size_t)2 * kI * kH, stream_key(seed, kTagWs1, kAllRanks), kWStd);
  p.ws2 = gen((size_t)kH * kI, stream_key(seed, kTagWs2, kAllRanks), kWStd);
  p.rk.resize(kRanks);
  for (int q = 0; q < kRanks; q++) {
    if (!all_ranks && q != rank) {
      continue;
    }
    p.rk[q].x    = gen((size_t)kT * kH, stream_key(seed, kTagX, q), 1.0f);
    p.rk[q].dout = gen((size_t)kT * kH, stream_key(seed, kTagDout, q), 1.0f);
    p.rk[q].w1   = gen((size_t)kEloc * 2 * kI * kH, stream_key(seed, kTagW1, q), kWStd);
    p.rk[q].w2   = gen((size_t)kEloc * kH * kI, stream_key(seed, kTagW2, q), kWStd);
  }
  return p;
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU reference (fp32 math on the bf16 inputs, double where it accumulates few terms)
// ---------------------------------------------------------------------------------------------------------------------

inline float dotf(const float *a, const float *b, int n) {
  float s = 0.0f;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < n; i++) {
    s += a[i] * b[i];
  }
  return s;
}

inline double dotd(const float *a, const float *b, int n) {
  double s = 0.0;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < n; i++) {
    s += (double)a[i] * b[i];
  }
  return s;
}

inline void axpy(float a, const float *x, float *y, int n) {
#pragma omp simd
  for (int i = 0; i < n; i++) {
    y[i] += a * x[i];
  }
}

// fc1 rows of GLU column i: routed experts interleave 128-row blocks of gate and up rows; the shared expert is
// [gate; up].
inline size_t gate_row(int i, bool interleaved) {
  return interleaved ? (size_t)(i / 128) * 256 + i % 128 : (size_t)i;
}

inline size_t up_row(int i, bool interleaved) {
  return interleaved ? gate_row(i, true) + 128 : (size_t)kI + i;
}

const float *expert_w1(const Problem &p, int e) {
  return p.rk[e / kEloc].w1.f.data() + (size_t)(e % kEloc) * 2 * kI * kH;
}

const float *expert_w2(const Problem &p, int e) {
  return p.rk[e / kEloc].w2.f.data() + (size_t)(e % kEloc) * kH * kI;
}

struct Routing {
  std::vector<int> idx;  // [T][k], in descending logit order
  std::vector<float> w;  // [T][k]
  std::vector<int> tpe;  // [E]
  double aux = 0.0;
  int near_ties = 0;  // tokens whose k-th and (k+1)-th logits are within 1e-5 (relative)
};

// The top n of v, by value then lower index.
std::array<int, kE> ranked(const double *v) {
  std::array<int, kE> o;
  std::iota(o.begin(), o.end(), 0);
  std::stable_sort(o.begin(), o.end(), [v](int a, int b) { return v[a] > v[b]; });
  return o;
}

// The fused router, softmax score: top k of the logits, weights the softmax over the selected logits; aux loss over
// the batch from the softmax over all experts and the top-k counts.
Routing route(const float *x, const float *wr) {
  Routing rt;
  rt.idx.resize((size_t)kT * kK);
  rt.w.resize((size_t)kT * kK);
  rt.tpe.assign(kE, 0);
  std::vector<double> logits((size_t)kT * kE);
#pragma omp parallel for schedule(static)
  for (int t = 0; t < kT; t++) {
    for (int e = 0; e < kE; e++) {
      logits[(size_t)t * kE + e] = dotd(x + (size_t)t * kH, wr + (size_t)e * kH, kH);
    }
  }
  std::vector<double> psum(kE, 0.0), cnt(kE, 0.0);
  for (int t = 0; t < kT; t++) {
    const double *l = &logits[(size_t)t * kE];
    const std::array<int, kE> o = ranked(l);
    const double a = l[o[kK - 1]], b = l[o[kK]];
    rt.near_ties += std::fabs(a - b) <= 1e-5 * std::max(std::fabs(a), std::fabs(b));
    const double mx = l[o[0]];
    double z = 0.0;
    for (int j = 0; j < kK; j++) {
      z += std::exp(l[o[j]] - mx);
    }
    for (int j = 0; j < kK; j++) {
      rt.idx[(size_t)t * kK + j] = o[j];
      rt.w[(size_t)t * kK + j]   = (float)(std::exp(l[o[j]] - mx) / z);
      rt.tpe[o[j]]++;
    }
    std::array<double, kE> p;
    double zall = 0.0;
    for (int e = 0; e < kE; e++) {
      p[e] = std::exp(l[e] - mx);
      zall += p[e];
    }
    for (int e = 0; e < kE; e++) {
      p[e] /= zall;
      psum[e] += p[e];
    }
    const std::array<int, kE> am = ranked(p.data());
    for (int j = 0; j < kK; j++) {
      cnt[am[j]] += 1.0;
    }
  }
  double s = 0.0;
  for (int e = 0; e < kE; e++) {
    s += psum[e] * cnt[e];
  }
  rt.aux = kE * (double)kAuxCoeff / ((double)kK * kT * kT) * s;
  return rt;
}

struct Scratch {
  std::vector<float> g, u, a, y, da, dg, du;
  Scratch() : g(kI), u(kI), a(kI), y(kH), da(kI), dg(kI), du(kI) {}
};

// g, u = W1 x; a = silu(g) u; y = W2 a (want_y).
void mlp_fwd(const float *x, const float *w1, const float *w2, bool interleaved, bool want_y, Scratch &s) {
  for (int i = 0; i < kI; i++) {
    s.g[i] = dotf(w1 + gate_row(i, interleaved) * kH, x, kH);
    s.u[i] = dotf(w1 + up_row(i, interleaved) * kH, x, kH);
    s.a[i] = s.g[i] / (1.0f + std::exp(-s.g[i])) * s.u[i];
  }
  for (int h = 0; want_y && h < kH; h++) {
    s.y[h] = dotf(w2 + (size_t)h * kI, s.a.data(), kI);
  }
}

// From dY = scale * dy: dA = W2^T dY, dg = dA u dsilu(g), du = dA silu(g).
void mlp_bwd(const float *dy, float scale, const float *w2, Scratch &s) {
  std::fill(s.da.begin(), s.da.end(), 0.0f);
  for (int h = 0; h < kH; h++) {
    axpy(scale * dy[h], w2 + (size_t)h * kI, s.da.data(), kI);
  }
  for (int i = 0; i < kI; i++) {
    const float g = s.g[i], sg = 1.0f / (1.0f + std::exp(-g));
    s.dg[i] = s.da[i] * s.u[i] * sg * (1.0f + g * (1.0f - sg));
    s.du[i] = s.da[i] * g * sg;
  }
}

// dx += W1^T [dg; du].
void mlp_dx(const float *w1, bool interleaved, const Scratch &s, float *dx) {
  for (int i = 0; i < kI; i++) {
    axpy(s.dg[i], w1 + gate_row(i, interleaved) * kH, dx, kH);
    axpy(s.du[i], w1 + up_row(i, interleaved) * kH, dx, kH);
  }
}

// out[rowmap[r]][:] = sum_p coef[p][r] vec[p][:] for r < nrows (rowmap empty: identity); blocks of rows share each
// vec[p] in cache.
void outer_sum(const float *coef, size_t ldc, int nrows, const std::vector<const float *> &vec, int len,
               const std::vector<size_t> &rowmap, float *out) {
  constexpr int kB = 16;
#pragma omp parallel for schedule(dynamic)
  for (int r0 = 0; r0 < nrows; r0 += kB) {
    const int nb = std::min(kB, nrows - r0);
    std::vector<float> acc((size_t)kB * len, 0.0f);
    for (size_t p = 0; p < vec.size(); p++) {
      for (int b = 0; b < nb; b++) {
        const float c = coef[p * ldc + r0 + b];
        if (c != 0.0f) {
          axpy(c, vec[p], acc.data() + (size_t)b * len, len);
        }
      }
    }
    for (int b = 0; b < nb; b++) {
      const size_t row = rowmap.empty() ? (size_t)(r0 + b) : rowmap[r0 + b];
      std::copy(acc.begin() + (size_t)b * len, acc.begin() + (size_t)(b + 1) * len, out + row * len);
    }
  }
}

struct Ref {
  std::vector<float> out, dx, dwr, dw1, dw2, dws1, dws2;
  std::vector<int> tpe;
  double aux = 0.0;
  int near_ties = 0;  // over every rank's tokens
};

// Rank r's outputs and gradients. Its tokens: out, dX (routed + router + shared), dWr and dWs (its tokens only). Its
// experts: dW1, dW2 over every rank's tokens routed to them.
Ref reference(const Problem &p, int r) {
  std::vector<Routing> rt(kRanks);
  Ref R;
  for (int q = 0; q < kRanks; q++) {
    rt[q] = route(p.rk[q].x.f.data(), p.wr.f.data());
    R.near_ties += rt[q].near_ties;
  }
  R.tpe = rt[r].tpe;
  R.aux = rt[r].aux;
  const float *X = p.rk[r].x.f.data(), *D = p.rk[r].dout.f.data();
  const float *Wr = p.wr.f.data(), *Ws1 = p.ws1.f.data(), *Ws2 = p.ws2.f.data();
  R.out.assign((size_t)kT * kH, 0.0f);
  R.dx.assign((size_t)kT * kH, 0.0f);
  std::vector<float> dlogits((size_t)kT * kE, 0.0f), sact((size_t)kT * kI), sdgu((size_t)kT * 2 * kI);

  // This rank's tokens.
#pragma omp parallel
  {
    Scratch s;
#pragma omp for schedule(dynamic, 4)
    for (int t = 0; t < kT; t++) {
      const float *x = X + (size_t)t * kH, *d = D + (size_t)t * kH;
      float *out = R.out.data() + (size_t)t * kH, *dx = R.dx.data() + (size_t)t * kH;
      const int *idx = rt[r].idx.data() + (size_t)t * kK;
      const float *w = rt[r].w.data() + (size_t)t * kK;
      std::array<double, kK> dw;
      double wdw = 0.0;
      for (int j = 0; j < kK; j++) {
        const float *w1 = expert_w1(p, idx[j]), *w2 = expert_w2(p, idx[j]);
        mlp_fwd(x, w1, w2, true, true, s);
        axpy(w[j], s.y.data(), out, kH);
        dw[j] = dotd(d, s.y.data(), kH);
        wdw += w[j] * dw[j];
        mlp_bwd(d, w[j], w2, s);
        mlp_dx(w1, true, s, dx);
      }
      for (int j = 0; j < kK; j++) {
        const float dl = (float)(w[j] * (dw[j] - wdw));
        dlogits[(size_t)t * kE + idx[j]] = dl;
        axpy(dl, Wr + (size_t)idx[j] * kH, dx, kH);
      }
      mlp_fwd(x, Ws1, Ws2, false, true, s);
      axpy(1.0f, s.y.data(), out, kH);
      std::copy(s.a.begin(), s.a.end(), sact.begin() + (size_t)t * kI);
      mlp_bwd(d, 1.0f, Ws2, s);
      mlp_dx(Ws1, false, s, dx);
      std::copy(s.dg.begin(), s.dg.end(), sdgu.begin() + (size_t)t * 2 * kI);
      std::copy(s.du.begin(), s.du.end(), sdgu.begin() + (size_t)t * 2 * kI + kI);
    }
  }
  std::vector<const float *> xrows(kT), arows(kT);
  for (int t = 0; t < kT; t++) {
    xrows[t] = X + (size_t)t * kH;
    arows[t] = sact.data() + (size_t)t * kI;
  }
  R.dwr.assign((size_t)kE * kH, 0.0f);
  R.dws1.assign((size_t)2 * kI * kH, 0.0f);
  R.dws2.assign((size_t)kH * kI, 0.0f);
  outer_sum(dlogits.data(), kE, kE, xrows, kH, {}, R.dwr.data());
  outer_sum(sdgu.data(), 2 * kI, 2 * kI, xrows, kH, {}, R.dws1.data());
  outer_sum(D, kH, kH, arows, kI, {}, R.dws2.data());

  // This rank's experts, over every rank's tokens.
  std::vector<size_t> w1rows(2 * kI);
  for (int i = 0; i < kI; i++) {
    w1rows[i]      = gate_row(i, true);
    w1rows[kI + i] = up_row(i, true);
  }
  R.dw1.assign((size_t)kEloc * 2 * kI * kH, 0.0f);
  R.dw2.assign((size_t)kEloc * kH * kI, 0.0f);
  for (int el = 0; el < kEloc; el++) {
    const int e = r * kEloc + el;
    std::vector<std::array<int, 2>> pairs;  // (rank, token)
    std::vector<float> pw;
    for (int q = 0; q < kRanks; q++) {
      for (int t = 0; t < kT; t++) {
        for (int j = 0; j < kK; j++) {
          if (rt[q].idx[(size_t)t * kK + j] == e) {
            pairs.push_back({q, t});
            pw.push_back(rt[q].w[(size_t)t * kK + j]);
          }
        }
      }
    }
    const int np = (int)pairs.size();
    const float *w1 = expert_w1(p, e), *w2 = expert_w2(p, e);
    std::vector<float> a((size_t)np * kI), dgu((size_t)np * 2 * kI), dy((size_t)np * kH);
    std::vector<const float *> px(np), pa(np);
#pragma omp parallel
    {
      Scratch s;
#pragma omp for schedule(dynamic, 8)
      for (int i = 0; i < np; i++) {
        const size_t off = (size_t)pairs[i][1] * kH;
        px[i] = p.rk[pairs[i][0]].x.f.data() + off;
        pa[i] = a.data() + (size_t)i * kI;
        const float *d = p.rk[pairs[i][0]].dout.f.data() + off;
        for (int h = 0; h < kH; h++) {
          dy[(size_t)i * kH + h] = pw[i] * d[h];
        }
        mlp_fwd(px[i], w1, w2, true, false, s);
        mlp_bwd(dy.data() + (size_t)i * kH, 1.0f, w2, s);
        std::copy(s.a.begin(), s.a.end(), a.begin() + (size_t)i * kI);
        std::copy(s.dg.begin(), s.dg.end(), dgu.begin() + (size_t)i * 2 * kI);
        std::copy(s.du.begin(), s.du.end(), dgu.begin() + (size_t)i * 2 * kI + kI);
      }
    }
    outer_sum(dgu.data(), 2 * kI, 2 * kI, px, kH, w1rows, R.dw1.data() + (size_t)el * 2 * kI * kH);
    outer_sum(dy.data(), kH, kH, pa, kI, {}, R.dw2.data() + (size_t)el * kH * kI);
  }
  return R;
}

// ---------------------------------------------------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------------------------------------------------

void check_cuda(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

#define CK(call) check_cuda((call), #call)

class DevBuf {
 public:
  DevBuf() = default;
  explicit DevBuf(size_t bytes) : bytes_(bytes) { CK(cudaMalloc(&p_, bytes)); }
  ~DevBuf() {
    if (p_) {
      cudaFree(p_);
    }
  }
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
  DevBuf(DevBuf &&o) noexcept : p_(o.p_), bytes_(o.bytes_) { o.p_ = nullptr; }
  DevBuf &operator=(DevBuf &&o) noexcept {
    std::swap(p_, o.p_);
    std::swap(bytes_, o.bytes_);
    return *this;
  }
  void *get() const { return p_; }
  size_t bytes() const { return bytes_; }

 private:
  void *p_ = nullptr;
  size_t bytes_ = 0;
};

template <typename V>
DevBuf to_device(const std::vector<V> &v) {
  DevBuf b(v.size() * sizeof(V));
  CK(cudaMemcpy(b.get(), v.data(), b.bytes(), cudaMemcpyHostToDevice));
  return b;
}

template <typename V>
std::vector<V> to_host(const DevBuf &b) {
  std::vector<V> v(b.bytes() / sizeof(V));
  CK(cudaMemcpy(v.data(), b.get(), b.bytes(), cudaMemcpyDeviceToHost));
  return v;
}

TensorWrapper wrap(const DevBuf &b, const std::vector<size_t> &shape, DType dt) {
  return TensorWrapper(b.get(), shape, dt);
}

// An MXFP8 weight [nmat][rows][cols]: e4m3 row-wise and column-wise data, unswizzled e8m0 scales [nmat][rows][cols/32]
// and [nmat][rows/32][cols], quantized on the device by KOSMOS.
struct MxWeight {
  DevBuf q, s, qc, sc;
  TensorWrapper t{NVTE_MXFP8_1D_SCALING};
};

void quantize(const DevBuf &w, size_t nmat, size_t rows, size_t cols, cudaStream_t st, MxWeight &m) {
  const size_t n = nmat * rows * cols;
  m.q  = DevBuf(n);
  m.s  = DevBuf(n / 32);
  m.qc = DevBuf(n);
  m.sc = DevBuf(n / 32);
  CK(kosmos_moe_quantize_mxfp8(w.get(), (int)nmat, (int)rows, (int)cols, 0, m.q.get(), m.s.get(), st));
  CK(kosmos_moe_quantize_mxfp8(w.get(), (int)nmat, (int)rows, (int)cols, 1, m.qc.get(), m.sc.get(), st));
  CK(cudaStreamSynchronize(st));
  m.t.set_rowwise_data(m.q.get(), DType::kFloat8E4M3, std::vector<size_t>{nmat, rows, cols});
  m.t.set_rowwise_scale_inv(m.s.get(), DType::kFloat8E8M0, std::vector<size_t>{nmat, rows, cols / 32});
  m.t.set_columnwise_data(m.qc.get(), DType::kFloat8E4M3, std::vector<size_t>{nmat, rows, cols});
  m.t.set_columnwise_scale_inv(m.sc.get(), DType::kFloat8E8M0, std::vector<size_t>{nmat, rows / 32, cols});
}

struct DevInputs {
  DevBuf x, dout, wr, w1, w2, ws1, ws2, aux_scale;
  TensorWrapper tx, tdout, twr, tw1, tw2, tws1, tws2, taux_scale;
  MxWeight m1, m2, ms1, ms2;
  bool mxfp8 = false;

  NVTETensor fc1() const { return mxfp8 ? m1.t.data() : tw1.data(); }
  NVTETensor fc2() const { return mxfp8 ? m2.t.data() : tw2.data(); }
  NVTETensor sfc1() const { return mxfp8 ? ms1.t.data() : tws1.data(); }
  NVTETensor sfc2() const { return mxfp8 ? ms2.t.data() : tws2.data(); }
};

std::unique_ptr<DevInputs> upload(const Problem &p, int r, bool mxfp8, cudaStream_t st) {
  auto in   = std::make_unique<DevInputs>();
  in->mxfp8 = mxfp8;
  in->x     = to_device(p.rk[r].x.b);
  in->dout  = to_device(p.rk[r].dout.b);
  in->wr    = to_device(p.wr.b);
  in->w1    = to_device(p.rk[r].w1.b);
  in->w2    = to_device(p.rk[r].w2.b);
  in->ws1   = to_device(p.ws1.b);
  in->ws2   = to_device(p.ws2.b);
  in->aux_scale = to_device(std::vector<float>{0.0f});
  in->tx    = wrap(in->x, {kT, kH}, DType::kBFloat16);
  in->tdout = wrap(in->dout, {kT, kH}, DType::kBFloat16);
  in->twr   = wrap(in->wr, {kE, kH}, DType::kBFloat16);
  in->tw1   = wrap(in->w1, {kEloc, 2 * kI, kH}, DType::kBFloat16);
  in->tw2   = wrap(in->w2, {kEloc, kH, kI}, DType::kBFloat16);
  in->tws1  = wrap(in->ws1, {2 * kI, kH}, DType::kBFloat16);
  in->tws2  = wrap(in->ws2, {kH, kI}, DType::kBFloat16);
  in->taux_scale = wrap(in->aux_scale, {1}, DType::kFloat32);
  if (mxfp8) {
    quantize(in->w1, kEloc, 2 * kI, kH, st, in->m1);
    quantize(in->w2, kEloc, kH, kI, st, in->m2);
    quantize(in->ws1, 1, 2 * kI, kH, st, in->ms1);
    quantize(in->ws2, 1, kH, kI, st, in->ms2);
  }
  return in;
}

struct DevOutputs {
  DevBuf out, dx, dwr, dw1, dw2, dws1, dws2, aux, tpe, pre_act;
  TensorWrapper tout, tdx, tdwr, tdw1, tdw2, tdws1, tdws2, taux, ttpe, tpre_act;
};

std::unique_ptr<DevOutputs> make_outputs() {
  auto o      = std::make_unique<DevOutputs>();
  const size_t b = 2;
  o->out      = DevBuf((size_t)kT * kH * b);
  o->dx       = DevBuf((size_t)kT * kH * b);
  o->dwr      = DevBuf((size_t)kE * kH * b);
  o->dw1      = DevBuf((size_t)kEloc * 2 * kI * kH * b);
  o->dw2      = DevBuf((size_t)kEloc * kH * kI * b);
  o->dws1     = DevBuf((size_t)2 * kI * kH * b);
  o->dws2     = DevBuf((size_t)kH * kI * b);
  o->aux      = DevBuf(sizeof(float));
  o->tpe      = DevBuf(kE * sizeof(int32_t));
  o->pre_act  = DevBuf((size_t)kT * 2 * kI * b);
  o->tout     = wrap(o->out, {kT, kH}, DType::kBFloat16);
  o->tdx      = wrap(o->dx, {kT, kH}, DType::kBFloat16);
  o->tdwr     = wrap(o->dwr, {kE, kH}, DType::kBFloat16);
  o->tdw1     = wrap(o->dw1, {kEloc, 2 * kI, kH}, DType::kBFloat16);
  o->tdw2     = wrap(o->dw2, {kEloc, kH, kI}, DType::kBFloat16);
  o->tdws1    = wrap(o->dws1, {2 * kI, kH}, DType::kBFloat16);
  o->tdws2    = wrap(o->dws2, {kH, kI}, DType::kBFloat16);
  o->taux     = wrap(o->aux, {1}, DType::kFloat32);
  o->ttpe     = wrap(o->tpe, {kE}, DType::kInt32);
  o->tpre_act = wrap(o->pre_act, {kT, 2 * kI}, DType::kBFloat16);
  return o;
}

// NaN-fills every output, so that one a call does not write shows.
void poison(const DevOutputs &o, cudaStream_t st) {
  for (const DevBuf *b : {&o.out, &o.dx, &o.dwr, &o.dw1, &o.dw2, &o.dws1, &o.dws2, &o.aux, &o.tpe, &o.pre_act}) {
    CK(cudaMemsetAsync(b->get(), 0xFF, b->bytes(), st));
  }
}

void forward(NVTERocmMoeLayer layer, const DevInputs &in, const DevOutputs &o, cudaStream_t st) {
  NVTERocmMoeForwardArgs a;
  std::memset(&a, 0, sizeof(a));
  a.struct_size       = sizeof(a);
  a.x                 = in.tx.data();
  a.router_weight     = in.twr.data();
  a.fc1_weight        = in.fc1();
  a.fc2_weight        = in.fc2();
  a.shared_fc1_weight = in.sfc1();
  a.shared_fc2_weight = in.sfc2();
  a.shared_pre_act    = o.tpre_act.data();
  a.out               = o.tout.data();
  a.aux_loss          = o.taux.data();
  a.tokens_per_expert = o.ttpe.data();
  a.save_for_backward = 1;
  nvte_rocm_moe_forward(layer, &a, st);
}

// fp32 main_grad buffers of the router, fc1, fc2, shared fc1 and shared fc2 weights (wgrad_accumulate 1).
struct MainGrads {
  static constexpr int kN = 5;
  DevBuf b[kN];
  TensorWrapper t[kN];
};

std::unique_ptr<MainGrads> make_main_grads() {
  auto m = std::make_unique<MainGrads>();
  const std::vector<size_t> shapes[MainGrads::kN] = {
      {kE, kH}, {kEloc, 2 * kI, kH}, {kEloc, kH, kI}, {2 * kI, kH}, {kH, kI}};
  for (int i = 0; i < MainGrads::kN; i++) {
    size_t n = 1;
    for (size_t d : shapes[i]) {
      n *= d;
    }
    m->b[i] = DevBuf(n * sizeof(float));
    m->t[i] = wrap(m->b[i], shapes[i], DType::kFloat32);
  }
  return m;
}

void backward(NVTERocmMoeLayer layer, const DevInputs &in, const DevOutputs &o, cudaStream_t st,
              const MainGrads *mg = nullptr) {
  NVTERocmMoeBackwardArgs a;
  std::memset(&a, 0, sizeof(a));
  a.struct_size       = sizeof(a);
  a.dout              = in.tdout.data();
  a.x                 = in.tx.data();
  a.router_weight     = in.twr.data();
  a.aux_loss_scale    = in.taux_scale.data();
  a.fc1_weight        = in.fc1();
  a.fc2_weight        = in.fc2();
  a.shared_fc1_weight = in.sfc1();
  a.shared_fc2_weight = in.sfc2();
  a.shared_pre_act    = o.tpre_act.data();
  a.dx                = o.tdx.data();
  a.fc1_wgrad         = o.tdw1.data();
  a.fc2_wgrad         = o.tdw2.data();
  a.router_wgrad      = o.tdwr.data();
  a.shared_fc1_wgrad  = o.tdws1.data();
  a.shared_fc2_wgrad  = o.tdws2.data();
  a.wgrad_accumulate = 0;
  if (mg) {
    a.router_wgrad     = mg->t[0].data();
    a.fc1_wgrad        = mg->t[1].data();
    a.fc2_wgrad        = mg->t[2].data();
    a.shared_fc1_wgrad = mg->t[3].data();
    a.shared_fc2_wgrad = mg->t[4].data();
    a.wgrad_accumulate = 1;
  }
  nvte_rocm_moe_backward(layer, &a, st);
}

// bf16 bits of x, rounded to nearest even (finite x).
uint16_t bf16_rne(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

struct HostResult {
  std::vector<uint16_t> out, dx, dwr, dw1, dw2, dws1, dws2;
  std::vector<float> aux;
  std::vector<int32_t> tpe;
};

HostResult fetch(const DevOutputs &o, cudaStream_t st) {
  CK(cudaStreamSynchronize(st));
  HostResult h;
  h.out  = to_host<uint16_t>(o.out);
  h.dx   = to_host<uint16_t>(o.dx);
  h.dwr  = to_host<uint16_t>(o.dwr);
  h.dw1  = to_host<uint16_t>(o.dw1);
  h.dw2  = to_host<uint16_t>(o.dw2);
  h.dws1 = to_host<uint16_t>(o.dws1);
  h.dws2 = to_host<uint16_t>(o.dws2);
  h.aux  = to_host<float>(o.aux);
  h.tpe  = to_host<int32_t>(o.tpe);
  return h;
}

struct Field {
  const char *name;
  const std::vector<uint16_t> *dev;
  const std::vector<float> *ref;
};

std::vector<Field> fields(const HostResult &h, const Ref *R) {
  return {{"out", &h.out, R ? &R->out : nullptr},          {"dx", &h.dx, R ? &R->dx : nullptr},
          {"router_wgrad", &h.dwr, R ? &R->dwr : nullptr},  {"fc1_wgrad", &h.dw1, R ? &R->dw1 : nullptr},
          {"fc2_wgrad", &h.dw2, R ? &R->dw2 : nullptr},     {"shared_fc1_wgrad", &h.dws1, R ? &R->dws1 : nullptr},
          {"shared_fc2_wgrad", &h.dws2, R ? &R->dws2 : nullptr}};
}

// Bitwise comparison of two runs' outputs and gradients.
void expect_bitwise_equal(Rank &rk, const char *what, const HostResult &a, const HostResult &b) {
  const std::vector<Field> fa = fields(a, nullptr), fb = fields(b, nullptr);
  for (size_t f = 0; f < fa.size(); f++) {
    size_t ndiff = 0;
    for (size_t i = 0; i < fa[f].dev->size(); i++) {
      ndiff += (*fa[f].dev)[i] != (*fb[f].dev)[i];
    }
    if (ndiff) {
      rk.fail(std::string(what) + ": " + fa[f].name + " differs in " + std::to_string(ndiff) + " elements");
    }
  }
  if (std::memcmp(a.aux.data(), b.aux.data(), sizeof(float)) != 0) {
    rk.fail(std::string(what) + ": aux_loss differs");
  }
  if (a.tpe != b.tpe) {
    rk.fail(std::string(what) + ": tokens_per_expert differs");
  }
}

struct ErrStat {
  double rel = 0.0;  // max|c - r| / max|r|
  double snr = 0.0;  // dB
  bool finite = true;
};

ErrStat compare(const std::vector<uint16_t> &c, const std::vector<float> &r) {
  double maxd = 0.0, maxr = 0.0, se = 0.0, sr = 0.0;
  ErrStat s;
  for (size_t i = 0; i < r.size(); i++) {
    const double v = bf2f(c[i]), d = v - r[i];
    s.finite = s.finite && std::isfinite(v);
    maxd = std::max(maxd, std::fabs(d));
    maxr = std::max(maxr, std::fabs((double)r[i]));
    se += d * d;
    sr += (double)r[i] * r[i];
  }
  s.rel = maxr > 0.0 ? maxd / maxr : maxd;
  s.snr = se > 0.0 ? 10.0 * std::log10(sr / se) : 999.0;
  return s;
}

// bf16: relative error per tensor; MXFP8: SNR. tokens_per_expert exact and the aux loss in both (bf16 router).
void expect_matches_reference(Rank &rk, const HostResult &h, const Ref &R, bool mxfp8) {
  char buf[256];
  int ndiff = 0;
  for (int e = 0; e < kE; e++) {
    ndiff += h.tpe[e] != R.tpe[e];
  }
  if (ndiff) {
    std::snprintf(buf, sizeof(buf), "tokens_per_expert: %d of %d experts differ (reference near ties: %d)", ndiff, kE,
                  R.near_ties);
    rk.fail(buf);
  }
  const double arel = std::fabs(h.aux[0] - R.aux) / std::fabs(R.aux);
  std::snprintf(buf, sizeof(buf), "aux %.6g ref %.6g rel %.2e", h.aux[0], R.aux, arel);
  rk.note(buf);
  if (!(arel <= kAuxTol)) {
    rk.fail(std::string("aux_loss: ") + buf);
  }
  for (const Field &f : fields(h, &R)) {
    const ErrStat s = compare(*f.dev, *f.ref);
    std::snprintf(buf, sizeof(buf), "%s rel %.2e snr %.1f dB", f.name, s.rel, s.snr);
    rk.note(buf);
    const bool ok = s.finite && (mxfp8 ? s.snr >= kMinSnr : s.rel <= kRelTol);
    if (!ok) {
      rk.fail(std::string(buf) + (s.finite ? "" : " (non-finite values)"));
    }
  }
}

// Workspace and layer handles. Their destruction is collective: skipped (and the other ranks told to abort) when this
// rank is unwinding from an exception or another rank aborted, as the peers may never join.
class Workspace {
 public:
  Workspace(Rank &rk, const NVTERocmMoeConfig &cfg) : shm_(rk.shm) {
    char why[512] = {0};
    if (!nvte_rocm_moe_config_valid(&cfg, why, sizeof(why))) {
      throw std::runtime_error(std::string("nvte_rocm_moe_config_valid refused the configuration: ") + why);
    }
    ws_ = nvte_rocm_moe_workspace_create(&cfg, &rk.comm);
  }
  ~Workspace() {
    if (std::uncaught_exceptions() > 0 || shm_->abort.load()) {
      shm_->abort.store(1);
      return;
    }
    try {
      nvte_rocm_moe_workspace_destroy(ws_);
    } catch (...) {
      shm_->abort.store(1);
    }
  }
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;
  NVTERocmMoeWorkspace get() const { return ws_; }

 private:
  Shm *shm_;
  NVTERocmMoeWorkspace ws_ = nullptr;
};

class Layer {
 public:
  Layer(Rank &rk, const Workspace &ws) : shm_(rk.shm), l_(nvte_rocm_moe_layer_create(ws.get())) {}
  ~Layer() {
    if (std::uncaught_exceptions() > 0 || shm_->abort.load()) {
      shm_->abort.store(1);
      return;
    }
    try {
      nvte_rocm_moe_layer_destroy(l_);
    } catch (...) {
      shm_->abort.store(1);
    }
  }
  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;
  NVTERocmMoeLayer get() const { return l_; }

 private:
  Shm *shm_;
  NVTERocmMoeLayer l_;
};

void init_device(Rank &rk) {
  CK(cudaSetDevice(rk.rank));
  CK(cudaStreamCreateWithFlags(&rk.stream, cudaStreamNonBlocking));
}

// Status bits of the layer's calls: errors (bad expert, duplicate) fail; these cases are sized to run in one pass
// keeping everything, so rounds or a reduced keep (recompute) fail too: they would test another path.
void check_status(Rank &rk, const Layer &l, const char *what) {
  const int st = nvte_rocm_moe_layer_status(l.get());
  rk.note(std::string(what) + " status " + std::to_string(st));
  const int bad = kNVTERocmMoeStatusBadExpert | kNVTERocmMoeStatusDuplicate | kNVTERocmMoeStatusRounds |
                  kNVTERocmMoeStatusKeptXpOnly | kNVTERocmMoeStatusKeptNothing;
  if (st & bad) {
    rk.fail(std::string(what) + ": layer status " + std::to_string(st));
  }
}

HostResult run_step(const Layer &l, const DevInputs &in, const DevOutputs &o, cudaStream_t st) {
  poison(o, st);
  forward(l.get(), in, o, st);
  backward(l.get(), in, o, st);
  return fetch(o, st);
}

// Forward and backward against the CPU reference.
void reference_case(Rank &rk, bool mxfp8) {
  const Problem p = make_problem(kSeed, rk.rank, true);
  const Ref R = reference(p, rk.rank);
  init_device(rk);
  Workspace ws(rk, make_config(mxfp8));
  Layer l(rk, ws);
  const std::unique_ptr<DevInputs> in = upload(p, rk.rank, mxfp8, rk.stream);
  const std::unique_ptr<DevOutputs> o = make_outputs();
  const HostResult h = run_step(l, *in, *o, rk.stream);
  check_status(rk, l, "layer");
  expect_matches_reference(rk, h, R, mxfp8);
}

}  // namespace

// Fused router (softmax, aux loss over the batch) and ungated shared expert, bf16, against the CPU reference.
TEST(KosmosMoe, RouterSharedBf16) {
  SKIP_IF_UNSUPPORTED();
  expect_all_ranks_pass(run_ranks(kRanks, [](Rank &rk) { reference_case(rk, false); }));
}

// The same forward and backward twice on one layer: bitwise-equal outputs and gradients.
TEST(KosmosMoe, Determinism) {
  SKIP_IF_UNSUPPORTED();
  expect_all_ranks_pass(run_ranks(kRanks, [](Rank &rk) {
    const Problem p = make_problem(kSeed, rk.rank, false);
    init_device(rk);
    Workspace ws(rk, make_config(false));
    Layer l(rk, ws);
    const std::unique_ptr<DevInputs> in = upload(p, rk.rank, false, rk.stream);
    const std::unique_ptr<DevOutputs> o = make_outputs();
    const HostResult a = run_step(l, *in, *o, rk.stream);
    const HostResult b = run_step(l, *in, *o, rk.stream);
    check_status(rk, l, "layer");
    expect_bitwise_equal(rk, "second run", a, b);
  }));
}

// Two layer handles on one workspace with both forwards in flight (forward A, forward B, backward B, backward A):
// bitwise equal to A and B each run alone on a single handle.
TEST(KosmosMoe, InFlight) {
  SKIP_IF_UNSUPPORTED();
  expect_all_ranks_pass(run_ranks(kRanks, [](Rank &rk) {
    const Problem pa = make_problem(kSeed + 1, rk.rank, false);
    const Problem pb = make_problem(kSeed + 2, rk.rank, false);
    init_device(rk);
    Workspace ws(rk, make_config(false));
    Layer alone(rk, ws), la(rk, ws), lb(rk, ws);
    const std::unique_ptr<DevInputs> ina = upload(pa, rk.rank, false, rk.stream);
    const std::unique_ptr<DevInputs> inb = upload(pb, rk.rank, false, rk.stream);
    const std::unique_ptr<DevOutputs> oa = make_outputs(), ob = make_outputs();
    const HostResult ra = run_step(alone, *ina, *oa, rk.stream);
    const HostResult rb = run_step(alone, *inb, *ob, rk.stream);
    poison(*oa, rk.stream);
    poison(*ob, rk.stream);
    forward(la.get(), *ina, *oa, rk.stream);
    forward(lb.get(), *inb, *ob, rk.stream);
    backward(lb.get(), *inb, *ob, rk.stream);
    backward(la.get(), *ina, *oa, rk.stream);
    const HostResult fa = fetch(*oa, rk.stream), fb = fetch(*ob, rk.stream);
    check_status(rk, alone, "alone");
    check_status(rk, la, "A");
    check_status(rk, lb, "B");
    expect_bitwise_equal(rk, "A in flight", ra, fa);
    expect_bitwise_equal(rk, "B in flight", rb, fb);
  }));
}

// MXFP8 training (bf16 router, MXFP8 expert and shared-expert GEMMs) against the bf16 CPU reference, by SNR.
TEST(KosmosMoe, Mxfp8RouterShared) {
  SKIP_IF_UNSUPPORTED();
  expect_all_ranks_pass(run_ranks(kRanks, [](Rank &rk) { reference_case(rk, true); }));
}

// main_grad accumulation (wgrad_accumulate 1), inside the one backward launch: into fp32 zeros the weight gradients
// are the kernel's fp32 tile sums, which round to the plain backward's bf16 gradients bit for bit; two more backwards
// into random main_grad give exactly the same two fp32 adds on the host.
TEST(KosmosMoe, MainGradAccumulate) {
  SKIP_IF_UNSUPPORTED();
  expect_all_ranks_pass(run_ranks(kRanks, [](Rank &rk) {
    const Problem p = make_problem(kSeed + 3, rk.rank, false);
    init_device(rk);
    Workspace ws(rk, make_config(false));
    Layer l(rk, ws);
    const std::unique_ptr<DevInputs> in = upload(p, rk.rank, false, rk.stream);
    const std::unique_ptr<DevOutputs> o = make_outputs();
    const std::unique_ptr<MainGrads> mg = make_main_grads();
    const HostResult plain = run_step(l, *in, *o, rk.stream);
    const std::vector<uint16_t> *bf[MainGrads::kN] = {&plain.dwr, &plain.dw1, &plain.dw2, &plain.dws1, &plain.dws2};
    const char *names[MainGrads::kN] = {"router", "fc1", "fc2", "shared_fc1", "shared_fc2"};
    for (int i = 0; i < MainGrads::kN; i++) {
      CK(cudaMemsetAsync(mg->b[i].get(), 0, mg->b[i].bytes(), rk.stream));
    }
    forward(l.get(), *in, *o, rk.stream);
    backward(l.get(), *in, *o, rk.stream, mg.get());
    CK(cudaStreamSynchronize(rk.stream));
    std::vector<float> tiles[MainGrads::kN], start[MainGrads::kN];
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)rk.rank;
    for (int i = 0; i < MainGrads::kN; i++) {
      tiles[i] = to_host<float>(mg->b[i]);
      size_t nd = 0;
      for (size_t j = 0; j < tiles[i].size(); j++) {
        nd += bf16_rne(tiles[i][j]) != (*bf[i])[j];
      }
      if (nd) {
        rk.fail(std::string("main_grad ") + names[i] + " from 0: " + std::to_string(nd) +
                " elements differ from the plain gradient");
      }
      start[i].resize(tiles[i].size());
      for (float &v : start[i]) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v = (float)((int)(s >> 40) - (1 << 23)) / (float)(1 << 23);
      }
      CK(cudaMemcpy(mg->b[i].get(), start[i].data(), mg->b[i].bytes(), cudaMemcpyHostToDevice));
    }
    for (int rep = 0; rep < 2; rep++) {
      forward(l.get(), *in, *o, rk.stream);
      backward(l.get(), *in, *o, rk.stream, mg.get());
    }
    CK(cudaStreamSynchronize(rk.stream));
    check_status(rk, l, "layer");
    for (int i = 0; i < MainGrads::kN; i++) {
      const std::vector<float> got = to_host<float>(mg->b[i]);
      size_t nd = 0;
      for (size_t j = 0; j < got.size(); j++) {
        const float want = (start[i][j] + tiles[i][j]) + tiles[i][j];
        nd += std::memcmp(&got[j], &want, 4) != 0;
      }
      if (nd) {
        rk.fail(std::string("main_grad ") + names[i] + " += 2 calls: " + std::to_string(nd) + " elements differ");
      }
    }
  }));
}

// A kernel's wait on another rank that never comes (the flag is never set) is bounded: KOSMOS's spin_backoff traps
// after MOE_WAIT_LIMIT_S (3 s in this test; 600 s in the library), so the process fails instead of hanging.
__global__ void wait_for_absent_rank(int *flag) {
  int n = 0;
  while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
    hk_moe::spin_backoff(n);
  }
}

TEST(KosmosMoe, WaitBoundTraps) {
  SKIP_IF_UNSUPPORTED();
  const auto t0 = std::chrono::steady_clock::now();
  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    struct rlimit no_core = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &no_core);  // no core files from the expected GPU exception
    (void)cudaSetDevice(0);
    int *flag = nullptr;
    if (cudaMalloc(&flag, sizeof(int)) != cudaSuccess || cudaMemset(flag, 0, sizeof(int)) != cudaSuccess) {
      _exit(3);
    }
    wait_for_absent_rank<<<8, 256>>>(flag);
    // The trap fails the kernel: the runtime aborts the process, or the synchronize returns the error.
    _exit(cudaDeviceSynchronize() != cudaSuccess ? 4 : 0);
  }
  int status = 0;
  pid_t r = 0;
  while ((r = waitpid(pid, &status, WNOHANG)) == 0 &&
         std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (r == 0) {
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    FAIL() << "the wait did not end within 60 s";
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_TRUE(WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) == 4))
      << "the wait should end in a GPU exception (status " << status << ")";
  EXPECT_GE(s, 3.0) << "trapped before the bound";
  EXPECT_LT(s, 30.0) << "trapped long after the bound";
}

// Agreed release: each rank releases A's stored forward at a different point of the same call sequence (rank r after
// r % 3 calls on B); once the agreement is read (KOSMOS_MOE_REL_LAG forwards later, then release_sync) A is reclaimed
// alike on every rank, and its reuse is bitwise equal to a fresh handle. A released forward refuses its backward.
TEST(KosmosMoe, AgreedRelease) {
  SKIP_IF_UNSUPPORTED();
  expect_all_ranks_pass(run_ranks(kRanks, [](Rank &rk) {
    const Problem pa = make_problem(kSeed + 3, rk.rank, false);
    const Problem pb = make_problem(kSeed + 4, rk.rank, false);
    const Problem pn = make_problem(kSeed + 5, rk.rank, false);
    init_device(rk);
    Workspace ws(rk, make_config(false));
    Layer la(rk, ws), lb(rk, ws);
    const std::unique_ptr<DevInputs> ina = upload(pa, rk.rank, false, rk.stream);
    const std::unique_ptr<DevInputs> inb = upload(pb, rk.rank, false, rk.stream);
    const std::unique_ptr<DevInputs> inn = upload(pn, rk.rank, false, rk.stream);
    const std::unique_ptr<DevOutputs> oa = make_outputs(), ob = make_outputs(), oc = make_outputs();
    auto expect_state = [&rk](const Layer &l, const char *what, int want) {
      const int st = nvte_rocm_moe_layer_release_state(l.get());
      if (st != want) {
        rk.fail(std::string(what) + ": release state " + std::to_string(st) + ", expected " + std::to_string(want));
      }
    };
    auto run_b = [&](int n) {
      for (int i = 0; i < n; i++) {
        run_step(lb, *inb, *ob, rk.stream);
      }
    };

    // 1. A stores a forward.
    poison(*oa, rk.stream);
    forward(la.get(), *ina, *oa, rk.stream);
    CK(cudaStreamSynchronize(rk.stream));
    expect_state(la, "A after its forward", kNVTERocmMoeReleaseStored);

    // 2. Each rank releases A at its own point of the same sequence of calls on B.
    for (int i = 0; i < 6; i++) {
      if (i == rk.rank % 3) {
        nvte_rocm_moe_layer_release(la.get());
        expect_state(la, "A right after its release", kNVTERocmMoeReleasePending);
      }
      run_b(1);
    }

    // 3. More forwards than the lag, then read every agreement: A reclaimed on every rank.
    run_b(6);
    nvte_rocm_moe_release_sync(ws.get());
    const int st = nvte_rocm_moe_layer_release_state(la.get());
    expect_state(la, "A after release_sync", kNVTERocmMoeReleaseReclaimed);
    std::array<int, kRanks> all;
    if (rk.comm.allgather(&rk, &st, all.data(), sizeof(int)) != 0) {
      throw std::runtime_error("allgather of the release states failed");
    }
    for (int q = 0; q < kRanks; q++) {
      if (all[q] != st) {
        rk.fail("A's release state is " + std::to_string(all[q]) + " on rank " + std::to_string(q) + ", " +
                std::to_string(st) + " here");
      }
    }

    // 4. The reclaimed A runs a new forward and backward, bitwise equal to a fresh handle.
    const HostResult ra = run_step(la, *inn, *oa, rk.stream);
    expect_state(la, "A after its reuse", kNVTERocmMoeReleaseIdle);
    Layer lc(rk, ws);
    const HostResult rc = run_step(lc, *inn, *oc, rk.stream);
    expect_bitwise_equal(rk, "reclaimed A vs fresh C", ra, rc);

    // 5. A released forward refuses its backward; C is then reclaimed before its destruction.
    forward(lc.get(), *inn, *oc, rk.stream);
    nvte_rocm_moe_layer_release(lc.get());
    expect_state(lc, "C right after its release", kNVTERocmMoeReleasePending);
    bool refused = false;
    try {
      backward(lc.get(), *inn, *oc, rk.stream);
    } catch (const std::runtime_error &) {
      refused = true;
    }
    if (!refused) {
      rk.fail("backward of a released forward was not refused");
    }
    run_b(5);
    nvte_rocm_moe_release_sync(ws.get());
    expect_state(lc, "C after release_sync", kNVTERocmMoeReleaseReclaimed);
    check_status(rk, la, "A");
    check_status(rk, lb, "B");
    check_status(rk, lc, "C");
  }));
}
