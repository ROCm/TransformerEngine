/*************************************************************************
 * Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "qola_mha_fwd.h"
#include "ck_fused_attn_utils.hpp"

// Staged gfx1250 forward dispatch. When this build includes the CK-free V3
// forward library (te_v3_libmha_fwd.so, built for gfx1250), declare its
// namespaced entry point so ck_attn_fwd can route to it on gfx1250 devices at
// runtime. The CK-full path (QOLA_NS(mha_fwd) == qola::te::mha_fwd) is used on
// all other archs.
#if defined(NVTE_AITER_V3_FWD_GFX1250)
namespace qola { namespace te_v3 {
float mha_fwd(const aiter::mha_fwd_args& args, const ck_tile::stream_config& stream_config);
}}  // namespace qola::te_v3
#endif

namespace ck_fused_attn{

#if defined(NVTE_AITER_V3_FWD_GFX1250)
namespace {
// True when the active device is gfx1250 (gcnArchName may carry feature
// suffixes, e.g. "gfx1250:sramecc+", so match on prefix).
bool is_gfx1250_device(){
  int dev = 0;
  if(hipGetDevice(&dev) != hipSuccess){ return false; }
  hipDeviceProp_t prop{};
  if(hipGetDeviceProperties(&prop, dev) != hipSuccess){ return false; }
  return prop.major == 12 && prop.minor == 5;
}

// D64 gfx1250 fmha_fwd_with_sink_asm (ENABLE_SINK=1): requires non-null sink_ptr
// of shape [nhead] fp32 in "AITER post-scale domain".  The kernel adds
// exp(sink_val[h]) to every row's softmax denominator.  We initialize to
// -1e30f so expf(-1e30f)=0.0f in fp32 — zero contribution, matching the
// UnfusedDotProductAttention reference which has no sink term.
// D128 (ENABLE_SINK=0): dispatch guard rejects sink_ptr!=nullptr; leave null.
//
// Single static buffer, allocated once, kept for the process lifetime.
constexpr int kSinkBufMaxHeads = 256;
static float*          s_sink_buf  = nullptr;
static std::once_flag  s_sink_once;

const void* get_gfx1250_sink_buf(){
  std::call_once(s_sink_once, [](){
    if(hipMalloc(&s_sink_buf, kSinkBufMaxHeads * sizeof(float)) != hipSuccess){
      s_sink_buf = nullptr;
      return;
    }
    std::vector<float> fill(kSinkBufMaxHeads, -1e30f);
    hipMemcpy(s_sink_buf, fill.data(),
              kSinkBufMaxHeads * sizeof(float), hipMemcpyHostToDevice);
  });
  return s_sink_buf;
}
}  // namespace
#endif

// print the fmha traits and fmha_args when calling ck apis
void log_fwd_config(const char* func_name, bool has_dropout, const aiter::mha_fwd_args& fmha_args, std::ostream* log_file){

  (*log_file) << "\n" << func_name << "\n";

  // debug fmha_traits
  (*log_file) << "\nfmha_traits: \n";
  log_value(log_file, "hdim_q", fmha_args.hdim_q);
  log_value(log_file, "hdim_v", fmha_args.hdim_v);
  log_value(log_file, "data_type", fmha_args.data_type);
  log_value(log_file, "is_group_mode", fmha_args.is_group_mode);
  log_value(log_file, "has_lse", fmha_args.has_lse);
  log_value(log_file, "has_dropout", has_dropout);
  log_value(log_file, "skip_min_seqlen_q", (fmha_args.min_seqlen_q != 0));
  log_value(log_file, "use_asm_v3", fmha_args.use_asm_v3);
  log_value(log_file, "how_v3_bf16_cvt", fmha_args.how_v3_bf16_cvt);
  // debug fmha_args
  (*log_file) << "\nfmha_args: \n";

  log_value(log_file, "q_ptr", fmha_args.q_ptr);
  log_value(log_file, "k_ptr", fmha_args.k_ptr);
  log_value(log_file, "v_ptr", fmha_args.v_ptr);
  log_value(log_file, "bias_ptr", fmha_args.bias_ptr);
  log_value(log_file, "rand_val_ptr", fmha_args.rand_val_ptr);
  log_value(log_file, "lse_ptr", fmha_args.lse_ptr);
  log_value(log_file, "o_ptr", fmha_args.o_ptr);

  log_value(log_file, "seqstart_q_ptr", fmha_args.seqstart_q_ptr);
  log_value(log_file, "seqstart_k_ptr", fmha_args.seqstart_k_ptr);
  log_value(log_file, "seqlen_q_ptr", fmha_args.seqlen_q_ptr);
  log_value(log_file, "seqlen_k_ptr", fmha_args.seqlen_k_ptr);
  log_value(log_file, "cu_seqlen_q_ptr", fmha_args.cu_seqlen_q_ptr);
  log_value(log_file, "cu_seqlen_k_ptr", fmha_args.cu_seqlen_k_ptr);

  log_value(log_file, "seqlen_q", fmha_args.seqlen_q);
  log_value(log_file, "seqlen_k", fmha_args.seqlen_k);
  log_value(log_file, "batch", fmha_args.batch);
  log_value(log_file, "max_seqlen_q", fmha_args.max_seqlen_q);
  log_value(log_file, "hdim_q", fmha_args.hdim_q);
  log_value(log_file, "hdim_v", fmha_args.hdim_v);
  log_value(log_file, "nhead_q", fmha_args.nhead_q);
  log_value(log_file, "nhead_k", fmha_args.nhead_k);
  log_value(log_file, "scale_s", fmha_args.scale_s);
  log_value(log_file, "logits_soft_cap", fmha_args.logits_soft_cap);

  log_value(log_file, "stride_q", fmha_args.stride_q);
  log_value(log_file, "stride_k", fmha_args.stride_k);
  log_value(log_file, "stride_v", fmha_args.stride_v);
  log_value(log_file, "stride_bias", fmha_args.stride_bias);
  log_value(log_file, "stride_randval", fmha_args.stride_randval);
  log_value(log_file, "stride_o", fmha_args.stride_o);
  log_value(log_file, "nhead_stride_q", fmha_args.nhead_stride_q);
  log_value(log_file, "nhead_stride_k", fmha_args.nhead_stride_k);
  log_value(log_file, "nhead_stride_v", fmha_args.nhead_stride_v);
  log_value(log_file, "nhead_stride_bias", fmha_args.nhead_stride_bias);
  log_value(log_file, "nhead_stride_randval", fmha_args.nhead_stride_randval);
  log_value(log_file, "nhead_stride_lse", fmha_args.nhead_stride_lse);
  log_value(log_file, "nhead_stride_o", fmha_args.nhead_stride_o);
  log_value(log_file, "batch_stride_q", fmha_args.batch_stride_q);
  log_value(log_file, "batch_stride_k", fmha_args.batch_stride_k);
  log_value(log_file, "batch_stride_v", fmha_args.batch_stride_v);
  log_value(log_file, "batch_stride_bias", fmha_args.batch_stride_bias);
  log_value(log_file, "batch_stride_randval", fmha_args.batch_stride_randval);
  log_value(log_file, "batch_stride_lse", fmha_args.batch_stride_lse);
  log_value(log_file, "batch_stride_o", fmha_args.batch_stride_o);

  log_value(log_file, "window_size_left", fmha_args.window_size_left);
  log_value(log_file, "window_size_right", fmha_args.window_size_right);
  log_value(log_file, "mask_type", fmha_args.mask_type);
  log_value(log_file, "bias_type", fmha_args.bias_type);
  log_value(log_file, "min_seqlen_q", fmha_args.min_seqlen_q);
  log_value(log_file, "p_drop", fmha_args.p_drop);
  log_value(log_file, "s_randval", fmha_args.s_randval);

  log_value(log_file, "dropout_seed_ptr", std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)));
  log_value(log_file, "dropout_offset_ptr", std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset)));
}

void dump_fwd_timings(const char* dump_path, float average_runtime){
  std::ofstream file;
  file.open(std::string(dump_path) + "aiter-fwd-timings.txt", std::ios_base::app);
  file << average_runtime << "\n";
}

// Populate the AITER mha_fwd_args from TE's CKAttnFwdArgs. Shared by ck_attn_fwd
// (real launch) and ck_attn_fwd_uses_v3 (v3 availability probe) so the probe can
// never disagree with the launch. v3_api_check is left false here; callers flip it.
// The stream-dependent max_seqlen override (NVTE_CK_RUNTIME_MAX_SEQLEN) is applied
// by ck_attn_fwd after this returns; it does not affect v3 kernel selection.
aiter::mha_fwd_args build_fwd_fmha_args(const CKAttnFwdArgs& args){

  bias_enum bias_type = bias_enum::no_bias;
  BiasShape bias_shape = BiasShape::k11SS;
  if (!args.is_group_mode()) {
    std::tie(bias_type, bias_shape) = get_ck_bias_type_shape(&args);
  }

  aiter::mha_fwd_args fmha_args{};
  fmha_args.q_ptr = args.q_ptr;
  fmha_args.k_ptr = args.k_ptr;
  fmha_args.v_ptr = args.v_ptr;

  fmha_args.batch    = args.b;
  fmha_args.seqlen_q = args.s_q; // unused in group mode
  fmha_args.hdim_q   = args.d_qk;
  fmha_args.hdim_v   = args.d_v;
  fmha_args.nhead_q  = args.h;
  fmha_args.nhead_k  = args.hg;

  fmha_args.stride_q       = args.stride_s_q;
  fmha_args.stride_k       = args.stride_s_k;
  fmha_args.stride_v       = args.stride_s_v;
  fmha_args.nhead_stride_q = args.stride_h_q;
  fmha_args.nhead_stride_k = args.stride_h_k;
  fmha_args.nhead_stride_v = args.stride_h_v;
  fmha_args.batch_stride_q = args.stride_b_q;
  fmha_args.batch_stride_k = args.stride_b_k;
  fmha_args.batch_stride_v = args.stride_b_v;

  if (args.is_group_mode()) {
    fmha_args.seqstart_q_ptr = args.cu_seqlen_q_padded_ptr==nullptr? args.cu_seqlen_q_ptr : args.cu_seqlen_q_padded_ptr;
    fmha_args.seqstart_k_ptr = args.cu_seqlen_kv_padded_ptr==nullptr? args.cu_seqlen_kv_ptr : args.cu_seqlen_kv_padded_ptr;
    fmha_args.cu_seqlen_q_ptr = args.cu_seqlen_q_ptr;
    fmha_args.cu_seqlen_k_ptr = args.cu_seqlen_kv_ptr;
  } else {
    fmha_args.seqstart_q_ptr = nullptr;
    fmha_args.seqstart_k_ptr = nullptr;
    fmha_args.cu_seqlen_q_ptr = nullptr;
    fmha_args.cu_seqlen_k_ptr = nullptr;
  }
  fmha_args.seqlen_q_ptr = nullptr;
  fmha_args.seqlen_k_ptr = nullptr;

  fmha_args.bias_ptr = bias_type==bias_enum::alibi? args.alibi_slope_ptr : args.bias_ptr;
  fmha_args.lse_ptr  = args.lse_ptr;
  fmha_args.o_ptr    = args.o_ptr;

  fmha_args.block_scale_seqstart_q_ptr = nullptr;
  fmha_args.block_scale_seqstart_k_ptr = nullptr;
  fmha_args.sink_ptr = nullptr;
#if defined(NVTE_AITER_V3_FWD_GFX1250)
  // D64 (ENABLE_SINK=1): fmha_fwd_gfx1250_batched requires non-null sink_ptr and
  // reads each element as a per-head logit added to the softmax denominator.
  // We pass -1e30f so exp(-1e30f)=0.0f in fp32 — zero contribution, matching
  // the UnfusedDotProductAttention reference.
  // D128 (ENABLE_SINK=0): dispatch guard rejects sink_ptr!=nullptr, so leave null.
  if(is_gfx1250_device() && args.d_qk == 64 && args.h <= kSinkBufMaxHeads) {
    fmha_args.sink_ptr = get_gfx1250_sink_buf();
  }
#endif
  fmha_args.seqlen_k     = args.s_kv; // unused in group mode (or kvcache enabled)
  fmha_args.max_seqlen_q = args.s_q;

  fmha_args.scale_s = args.scaling_factor;

  fmha_args.logits_soft_cap = 0.f;

  // bias is of shape [b, h , s_q, s_kv]
  fmha_args.stride_bias = args.is_group_mode()? 0 : (bias_type==bias_enum::alibi? 0: args.s_kv);
  fmha_args.stride_o          = args.stride_s_o;
  fmha_args.nhead_stride_bias = get_nhead_stride_bias(bias_shape, args.s_q, args.s_kv, args.is_group_mode());
  fmha_args.batch_stride_bias = get_batch_stride_bias(args.bias_h, bias_shape, args.s_q, args.s_kv, args.is_group_mode(), true);
  // softmax_lse is of shape [b, h, s_q]
  fmha_args.nhead_stride_lse  = args.is_group_mode()? args.max_tokens_q : args.s_q;
  fmha_args.batch_stride_lse  = args.is_group_mode()? 0 : args.h * args.s_q;
  fmha_args.nhead_stride_o    = args.stride_h_o;
  fmha_args.batch_stride_o    = args.stride_b_o;

  fmha_args.window_size_left  = args.window_size_left;
  fmha_args.window_size_right = args.window_size_right;
  fmha_args.mask_type         = static_cast<ck_tile::index_t>(static_cast<mask_enum>(args.attn_mask_type));

  fmha_args.rand_val_ptr = nullptr;

  fmha_args.stride_randval       = args.s_kv;
  // Unused
  fmha_args.nhead_stride_randval = 0;
  fmha_args.batch_stride_randval = 0;
  fmha_args.nhead_stride_q_descale = 0;
  fmha_args.nhead_stride_k_descale = 0;
  fmha_args.nhead_stride_v_descale = 0;
  fmha_args.batch_stride_q_descale = 0;
  fmha_args.batch_stride_k_descale = 0;
  fmha_args.batch_stride_v_descale = 0;

  fmha_args.p_drop    = args.dropout_probability;
  fmha_args.s_randval = 0;
  fmha_args.drop_seed_offset = std::pair<const void*, const void*>{args.philox_seed_ptr, args.philox_offset_ptr};
  fmha_args.use_asm_v3      = args.uses_fwd_v3;
  fmha_args.how_v3_bf16_cvt = args.how_v3_bf16_cvt;
  fmha_args.v3_api_check    = false;
  fmha_args.data_type       = get_data_type_str(args.dtype);
  fmha_args.is_group_mode   = args.is_group_mode();
  fmha_args.bias_type       = static_cast<int>(bias_type);
  fmha_args.has_lse         = args.lse_ptr!=nullptr;
#if ENABLE_CK
  fmha_args.qscale_type     = static_cast<int>(quant_scale_enum::no_scale);
#else
  // quant_scale_enum lives in the CK example headers (quant.hpp), absent in the
  // CK-free build. no_scale == 0; this fwd path is unused on gfx1250 anyway.
  fmha_args.qscale_type     = 0;
#endif
  fmha_args.has_sink        = false;
  fmha_args.q_descale_ptr    = nullptr;
  fmha_args.k_descale_ptr    = nullptr;
  fmha_args.v_descale_ptr    = nullptr;
  fmha_args.sink_size        = 0;
  fmha_args.min_seqlen_q     = 0;
  fmha_args.block_scale_size_q  = 0;
  fmha_args.block_scale_size_kv = 0;

  fmha_args.num_splits = args.num_splits;
  fmha_args.splitkv_workspace_ptr = args.splitkv_workspace_ptr;

  return fmha_args;
}

// Probe whether AITER's v3 (asm) forward path will run for this config, without
// launching any kernel. Builds the same args as ck_attn_fwd and relies on AITER's
// v3_api_check dry-run (returns 1 when v3 is available, -1 otherwise).
bool ck_attn_fwd_uses_v3(const CKAttnFwdArgs& args){
#if defined(NVTE_AITER_V3_FWD_GFX1250)
  // The gfx1250 tier is asm-v3 by construction (CK-free, no v2 launcher to fall
  // back to), so ck_attn_fwd always routes there on that device.
  if(is_gfx1250_device()){
    return true;
  }
#endif
#if defined(NVTE_AITER_CK_FULL)
  aiter::mha_fwd_args fmha_args = build_fwd_fmha_args(args);
  fmha_args.v3_api_check = true;
  // No kernel is launched in check mode, so the stream/log flags are irrelevant.
  ck_tile::stream_config stream_config{nullptr, false, false};
  return QOLA_NS(mha_fwd)(fmha_args, stream_config) == 1;
#else
  // gfx1250-only build: no CK-full forward library to dry-run against.
  (void)args;
  return false;
#endif
}

hipError_t ck_attn_fwd(const CKAttnFwdArgs& args, hipStream_t stream){

  bool has_dropout = (args.is_training && args.dropout_probability > 0.f);

  bool ck_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_log_config = true;
  }
  const char* dump_path = std::getenv("NVTE_DUMP_AITER_RT");
  auto* log_file = get_ck_log_stream();
  // print kernel name on verbose mode
  ck_tile::stream_config stream_config{stream, dump_path!=nullptr, get_ck_log_stream() != nullptr};

  aiter::mha_fwd_args fmha_args = build_fwd_fmha_args(args);

  if(const char* env_p = std::getenv("NVTE_CK_RUNTIME_MAX_SEQLEN")){
    if(args.is_group_mode() && std::string(env_p) == "1"){
      if(log_file){
        *log_file << "attn_fwd(ck): Enabling runtime max_seqlen calculation for small seqlen optimization.";
      }
      fmha_args.max_seqlen_q = get_runtime_max_seqlen(args.b, args.cu_seqlen_q_ptr, args.cu_seqlen_q_padded_ptr, args.lse_ptr, stream);
    }
  }

  // print ck traits and fmha_args when needed
  if(log_file){
     log_fwd_config(__FUNCTION__, has_dropout, fmha_args, log_file);
  }

  float average_runtime;
#if defined(NVTE_AITER_V3_FWD_GFX1250)
  // Pre-fill O and LSE before calling the gfx1250 ASM forward kernel.
  // The kernel ABI requires a valid (allocated) LSE buffer regardless of
  // return_lse; the kernel may touch lse_ptr even when return_lse=0.
  // O/LSE pre-initialization is handled inside fmha_fwd_gfx1250_batched
  // (in aiter/csrc/cpp_itfs/mha_fwd.cu) as part of the kernel calling convention.
  if(is_gfx1250_device()){
    if(fmha_args.lse_ptr == nullptr)
      throw std::runtime_error(
        "ck_fused_attn fwd: lse_ptr is null on gfx1250 — caller must allocate softmax LSE.");
    if(fmha_args.o_ptr == nullptr)
      throw std::runtime_error(
        "ck_fused_attn fwd: o_ptr is null on gfx1250 — caller must allocate output.");
    average_runtime = qola::te_v3::mha_fwd(fmha_args, stream_config);
  } else
#endif
  {
#if defined(NVTE_AITER_CK_FULL)
    average_runtime = QOLA_NS(mha_fwd)(fmha_args, stream_config);
#else
    // gfx1250-only build: no CK-full forward library exists (gfx1250 has no
    // forward kernels). The unified backend selector never picks CK on gfx1250,
    // so this path is unreachable at runtime; the guard only keeps the link
    // closed when te_libmha_fwd.so is absent.
    throw std::runtime_error(
      "ck_fused_attn fwd: no CK-full AITER forward library in this build "
      "(gfx1250 has no forward kernels).");
#endif
  }
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }
  if(dump_path){
    dump_fwd_timings(dump_path, average_runtime);
  }
  return hipSuccess;
}

int ck_attn_fwd_num_splits(const CKAttnFwdArgs& args){
#if FAV_NATIVE_ON
  aiter::mha_fwd_args fmha_args = build_fwd_fmha_args(args);
  return QOLA_NS(mha_fwd_calculate_num_splits)(fmha_args);
#else
  return -1;
#endif
}

size_t ck_attn_fwd_workspace_size(const CKAttnFwdArgs& args){
#if FAV_NATIVE_ON
  aiter::mha_fwd_args fmha_args = build_fwd_fmha_args(args);
  return QOLA_NS(mha_fwd_workspace_size)(fmha_args);
#else
  return 0;
#endif
}

}//namespace ck_fused_attn

