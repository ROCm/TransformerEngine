// Reproducer: the AITER/CK fused-attention backward launcher corrupts memory
// when its dispatch is captured into a HIP graph and replayed.
//
// Root cause -- composable_kernel/example/ck_tile/01_fmha/fmha_bwd.hpp,
// fmha_bwd_launcher::prepare_workspace_async(): workspace metadata is packed by a
// hipLaunchHostFunc host node whose closure deletes ITSELF after running (and the
// pinned staging is released the same way). A host node re-runs on every graph
// launch, so the 2nd replay invokes a freed closure -> bad_function_call /
// heap-use-after-free / double-free.
//
// This driver PRE-ALLOCATES the device workspace and pinned staging and serves
// them via the callbacks, so nothing is allocated during capture -- leaving the
// host-node lifetime defect as the only replay-unsafe thing. (It therefore also
// shows that pre-sizing the workspace does not, by itself, make V2 capturable.)
//
// Build A -- plain AITER (default, no QoLA), from an AITER checkout $AITER:
//   cd $AITER/op_tests/cpp/mha && ./build_mha.sh bwd          # builds libmha_bwd.so
//   hipcc -std=c++20 -O2 -g -DUSE_ROCM=1 -DENABLE_CK=1 \
//     -I$AITER/3rdparty/composable_kernel/include \
//     -I$AITER/3rdparty/composable_kernel/example/ck_tile/01_fmha/ \
//     -I$AITER/csrc/include \
//     ck_graph_capture_repro.cpp -L$AITER/op_tests/cpp/mha -lmha_bwd \
//     -Wl,-rpath,$AITER/op_tests/cpp/mha -o repro
//
// Build B -- validate against a prebuilt QoLA lib (same -I flags, add -DUSE_QOLA):
//     ... -DUSE_QOLA ... ck_graph_capture_repro.cpp \
//     -L$LIBDIR -l:te_libmha_bwd.so -Wl,-rpath,$LIBDIR -o repro
//
// Run:  ./repro      # 3 replays -> crashes on replay #1
//       ./repro 1    # 1 replay  -> completes cleanly (control)

#include "ck_tile/host/stream_config.hpp"
#include "mha_bwd.h"   // aiter::mha_bwd_args, aiter::mha_bwd

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

// Default: call aiter::mha_bwd directly (plain AITER). -DUSE_QOLA routes through
// qola::te::mha_bwd from a prebuilt QoLA lib, which just forwards into the same
// aiter::mha_bwd -- the defect is identical either way.
#if defined(USE_QOLA)
namespace qola { namespace te {
float mha_bwd(const aiter::mha_bwd_args&, const ck_tile::stream_config&);
} }
#define MHA_BWD(args, sc) qola::te::mha_bwd((args), (sc))
#else
#define MHA_BWD(args, sc) aiter::mha_bwd((args), (sc))
#endif

#define HIP_CHECK(expr)                                                                   \
  do {                                                                                    \
    hipError_t _e = (expr);                                                               \
    if (_e != hipSuccess) {                                                               \
      std::fprintf(stderr, "HIP error: %s at %s:%d -> %s\n", #expr, __FILE__, __LINE__,   \
                   hipGetErrorString(_e));                                                \
      std::exit(2);                                                                       \
    }                                                                                     \
  } while (0)

// Zeroed device buffer; contents are irrelevant (we exercise control flow, not math).
static void* dmalloc(size_t bytes) {
  void* p = nullptr;
  HIP_CHECK(hipMalloc(&p, bytes));
  HIP_CHECK(hipMemset(p, 0, bytes));
  return p;
}

int main(int argc, char** argv) {
  const int replays = (argc > 1) ? std::atoi(argv[1]) : 3;

  // Contiguous BSHD, bf16, batch mode, no MQA / bias / dropout.
  const int b = 2, h = 4, hk = 4, s_q = 512, s_k = 512, d = 128, dv = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const int elem = 2;  // sizeof(bf16)

  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::fprintf(stderr, "device: %s (%s), replays=%d\n", prop.name, prop.gcnArchName, replays);

  // Forward I/O + gradient tensors.
  void* q    = dmalloc((size_t)b * s_q * h  * d  * elem);
  void* k    = dmalloc((size_t)b * s_k * hk * d  * elem);
  void* v    = dmalloc((size_t)b * s_k * hk * dv * elem);
  void* o    = dmalloc((size_t)b * s_q * h  * dv * elem);
  void* lse  = dmalloc((size_t)b * h * s_q * sizeof(float));
  void* sm_d = dmalloc((size_t)b * h * s_q * sizeof(float));  // softmax_d (d_ptr)
  void* dout = dmalloc((size_t)b * s_q * h  * dv * elem);
  void* dq   = dmalloc((size_t)b * s_q * h  * d  * elem);
  void* dk   = dmalloc((size_t)b * s_k * h  * d  * elem);
  void* dvg  = dmalloc((size_t)b * s_k * h  * dv * elem);

  // AOT device workspace: reserved once, handed out (never grown) by the callback,
  // so no allocation happens during graph capture. Deterministic-CK upper bound is
  // 4*nsplits*h*(b*s_q)*d plus the host-metadata page; +1 MiB of slack.
  const int kN0 = (d <= 128) ? 128 : 64;
  const int nsplits = (s_k + kN0 - 1) / kN0;
  const size_t ws_bytes = 4096 + (size_t)4 * nsplits * h * (b * s_q) * d + (1u << 20);
  void* ws = dmalloc(ws_bytes);
  auto workspace_alloc = [ws, ws_bytes](size_t bytes, bool zero_init) -> void* {
    if (bytes > ws_bytes) {
      std::fprintf(stderr, "FATAL: launcher wants %zu > reserved %zu\n", bytes, ws_bytes);
      std::exit(3);
    }
    if (zero_init && bytes) HIP_CHECK(hipMemsetAsync(ws, 0, bytes, nullptr));
    return ws;
  };

  // Pinned staging: allocated once, never freed (no-op deleter). This removes the
  // pinned-buffer release as a variable, so any replay failure is purely CK's
  // self-deleting pack closure.
  void* pin = nullptr;
  HIP_CHECK(hipHostMalloc(&pin, (1u << 20), hipHostMallocDefault));
  auto pinned_host_alloc = [pin](size_t) -> std::shared_ptr<void> {
    return std::shared_ptr<void>(pin, [](void*) {});
  };

  // Contiguous BSHD strides (per-seqlen, per-head, per-batch).
  const int stride_q = h * d,   nhead_stride_q = d,   batch_stride_q = h  * s_q * d;
  const int stride_k = hk * d,  nhead_stride_k = d,   batch_stride_k = hk * s_k * d;
  const int stride_v = hk * dv, nhead_stride_v = dv,  batch_stride_v = hk * s_k * dv;
  const int stride_o = h * dv,  nhead_stride_o = dv,  batch_stride_o = h  * s_q * dv;
  const int stride_do = h * dv, nhead_stride_do = dv, batch_stride_do = h * s_q * dv;
  const int stride_dk = h * d,  batch_stride_dk = h * s_k * d;
  const int stride_dv = h * dv, batch_stride_dv = h * s_k * dv;
  const int nhead_stride_lsed = s_q, batch_stride_lsed = h * s_q;

  // Route to the exact buggy path:
  //   use_asm_v3=false      -> fmha_v3_bwd() bails, so dispatch falls to the CK V2
  //                            launcher (the explicit "no asm" request).
  //   is_deterministic=true -> forces a non-QrQtrDor kernel (host_ws != 0), so
  //                            prepare_workspace_async runs the host-pack path with
  //                            the self-deleting closures. A QrQtrDor kernel has
  //                            host_ws == 0, skips that path, and would hide the bug.
  aiter::mha_bwd_args args{
      /*use_asm_v3*/ false, /*v3_atomic_fp32*/ true, /*v3_bf16_cvt*/ false, /*v3_api_check*/ false,
      /*hdim_q*/ d, /*hdim_v*/ dv, /*data_type*/ std::string("bf16"),
      /*is_group_mode*/ false, /*mask_type*/ 1 /*top-left causal*/, /*bias_type*/ 0,
      /*has_dbias*/ false, /*has_dropout*/ false, /*is_store_randval*/ false,
      /*is_deterministic*/ true,
      q, k, v, /*bias*/ nullptr, o, lse, dout, sm_d, /*randval*/ nullptr,
      dq, dk, dvg, /*dbias*/ nullptr, /*sink*/ nullptr, /*d_sink*/ nullptr,
      /*seqstart_q*/ nullptr, /*seqstart_k*/ nullptr,
      /*seqlen_q_ptr*/ nullptr, /*seqlen_k_ptr*/ nullptr,
      /*cu_seqlen_q*/ nullptr, /*cu_seqlen_k*/ nullptr,
      /*seqlen_q*/ s_q, /*seqlen_k*/ s_k, /*batch*/ b,
      /*max_seqlen_q*/ s_q, /*max_seqlen_k*/ s_k, /*nhead_q*/ h, /*nhead_k*/ hk, /*scale*/ scale,
      stride_q, stride_k, stride_v, /*stride_bias*/ 0, stride_o, /*stride_randval*/ s_k, stride_do,
      /*stride_dq*/ stride_q, stride_dk, stride_dv, /*stride_dbias*/ 0,
      nhead_stride_q, nhead_stride_k, nhead_stride_v, /*nhead_stride_bias*/ 0, nhead_stride_o,
      /*nhead_stride_randval*/ s_q * s_k, nhead_stride_do, nhead_stride_lsed,
      /*nhead_stride_dq*/ nhead_stride_q, /*nhead_stride_dk*/ nhead_stride_k,
      /*nhead_stride_dv*/ nhead_stride_v, /*nhead_stride_dbias*/ 0,
      batch_stride_q, batch_stride_k, batch_stride_v, /*batch_stride_bias*/ 0, batch_stride_o,
      /*batch_stride_randval*/ h * s_q * s_k, batch_stride_do, batch_stride_lsed,
      /*batch_stride_dq*/ batch_stride_q, batch_stride_dk, batch_stride_dv, /*batch_stride_dbias*/ 0,
      /*window_left*/ -1, /*window_right*/ 0, /*p_drop*/ 0.0f, /*p_undrop*/ 1.0f,
      /*drop_seed_offset*/ std::pair<uint64_t, uint64_t>{0, 0},
      workspace_alloc, pinned_host_alloc};

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  ck_tile::stream_config sc{stream, /*time_kernel=*/false};  // no internal timing/sync

  // Warm up eagerly: triggers kernel module load and confirms the config is built.
  float t = MHA_BWD(args, sc);
  HIP_CHECK(hipStreamSynchronize(stream));
  if (t < 0) {
    std::fprintf(stderr, "config not supported by this build (returned %f)\n", t);
    return 4;
  }
  std::fprintf(stderr, "warmup ok (%.3f ms)\n", t);

  // Capture the real mha_bwd dispatch (kernels + the prepare_workspace_async host
  // nodes) into a graph.
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  MHA_BWD(args, sc);
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  std::fprintf(stderr, "graph captured + instantiated.\n");

  // Replay. Launch 0 runs the pack closure then deletes it; launch 1+ re-fire the
  // host node on the freed closure -> crash.
  for (int i = 0; i < replays; ++i) {
    std::fprintf(stderr, "graph launch %d ...\n", i);
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    std::fprintf(stderr, "  launch %d returned\n", i);
  }

  std::fprintf(stderr, "completed %d replay(s) without a crash (unexpected if >1)\n", replays);
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  return 0;
}
