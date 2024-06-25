#include <stdexcept>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "ck_tile/host.hpp"
#include "bias.hpp"
#include "mask.hpp"
#include "fmha_fwd.hpp"
#include "fmha_bwd.hpp"

namespace ck_fused_attn{

#define CK_FUSED_ATTN_TYPE_SWITCH_ALL(dtype, type, ...)                         \
  switch (dtype) {                                                              \
    case DType::kBFloat16: {                                                    \
        using type = ck_tile::bf16_t;                                           \
        {__VA_ARGS__}                                                           \
    } break;                                                                    \
    case DType::kFloat16: {                                                     \
        using type = ck_tile::half_t;                                           \
        {__VA_ARGS__}                                                           \
    } break;                                                                    \
    default:                                                                    \
        throw std::runtime_error("Invalid type.");                              \
    }

template <typename T>
struct TypeName{
  static const char* get(){
    return "";
  }
};

template <>
struct TypeName<ck_tile::bf16_t>{
  static const char* get(){
      return "bf16";
  }
};

template <>
struct TypeName<ck_tile::half_t>{
  static const char* get(){
      return "fp16";
  }
};

template <typename DataType>
hipError_t ck_attn_fwd_impl(
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  const void* q_ptr, 
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  bool is_training,
  float scaling_factor,
  float dropout_probability,
  uint64_t philox_seed, uint64_t philox_offset,
  bool is_causal,
  void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  void* lse_ptr, // lse shape (b_q, h_q, s_q) and stride (h_qxs_q, s_q, 1) fixed by ck_fused_attn
  hipStream_t stream){

  using TypeConfig = FmhaFwdTypeConfig<DataType>;
  using QDataType = typename TypeConfig::QDataType;
  using KDataType = typename TypeConfig::KDataType;
  using VDataType = typename TypeConfig::VDataType;
  using BiasDataType = typename TypeConfig::BiasDataType;
  using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
  using LSEDataType = typename TypeConfig::LSEDataType;
  using SaccDataType = typename TypeConfig::SaccDataType;
  using SMPLComputeDataType = typename TypeConfig::SMPLComputeDataType;
  using PDataType = typename TypeConfig::PDataType;
  using OaccDataType = typename TypeConfig::OaccDataType;
  using ODataType = typename TypeConfig::ODataType;

  bool has_dropout = (is_training && dropout_probability > 0.f);
  bool has_lse = (lse_ptr != nullptr);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t seqlen_q = s_q;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d;
  ck_tile::index_t seqlen_k = s_kv;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_k = s_kv;
  float scale_s = scaling_factor;
  float scale_p = 1.f;
  float scale_o = 1.f;
  float p_drop = dropout_probability;
  bool is_group_mode = false;
  bool is_v_rowmajor = true;
  bool do_fp8_static_quant = false;

  //TODO: support other bias and mask
  bias_enum bias_type = bias_enum::no_bias;
  mask_enum mask_type;
  ck_tile::index_t left, right;
  ck_tile::stream_config stream_config{stream};

  if (is_causal){
    mask_type = mask_enum::mask_top_left;
    left = -1;
    right = 0;
  } else {
    mask_type = mask_enum::no_mask;
  }

  ck_tile::index_t shape_batch = batch;
  ck_tile::index_t shape_seqlen_q = seqlen_q;
  ck_tile::index_t shape_seqlen_k = seqlen_k;

  std::string data_type(TypeName<DataType>::get());

  auto fmha_traits = fmha_fwd_traits{
    hdim_q,    hdim_v,    data_type, is_group_mode, is_v_rowmajor,
    mask_type, bias_type, has_lse,   has_dropout,   do_fp8_static_quant};

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // TODO: support bias later
    const ck_tile::index_t stride_bias = 0;
    const ck_tile::index_t stride_randval = max_seqlen_k;
    const ck_tile::index_t stride_o = stride_s_o;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    const ck_tile::index_t nhead_stride_bias = 0;
    //TODO: randval never used, can we remove it
    const ck_tile::index_t nhead_stride_randval =
        shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t nhead_stride_lse = shape_seqlen_q;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = stride_b_q;
    const ck_tile::index_t batch_stride_k = stride_b_k;
    const ck_tile::index_t batch_stride_v = stride_b_v;
    const ck_tile::index_t batch_stride_bias = 0;
    //TODO: randval never used, can we remove it
    const ck_tile::index_t batch_stride_randval =
        nhead * shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t batch_stride_lse = nhead * shape_seqlen_q;
    const ck_tile::index_t batch_stride_o = stride_b_o;

    return fmha_fwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         nullptr,//bias_ptr
                         nullptr,//rand_val_ptr
                         lse_ptr,
                         o_ptr,
                         nullptr,//cu_seqlen_q
                         nullptr,//cu_seqlen_kv
                         nullptr, /* seqlen_k_ptr */
                         shape_seqlen_q,
                         shape_seqlen_k,
                         batch,
                         max_seqlen_q,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
                         scale_p,
                         scale_o,
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_bias,
                         stride_randval,
                         stride_o,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         nhead_stride_bias,
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         batch_stride_bias,
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         p_drop,
                         false,
                         {philox_seed, philox_offset}};
  }();

  fmha_fwd(fmha_traits, fmha_args, stream_config);

  return hipSuccess;
} 

hipError_t ck_attn_fwd(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  const void* q_ptr, 
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  bool is_training,
  float scaling_factor,
  float dropout_probability,
  uint64_t philox_seed, uint64_t philox_offset,
  bool is_causal,
  void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  void* lse_ptr, 
  hipStream_t stream){
  
  hipError_t ret;
  CK_FUSED_ATTN_TYPE_SWITCH_ALL(dtype, type,
    ret = ck_attn_fwd_impl<type>(b, h, hg, s_q, s_kv, d,
                                 q_ptr, 
                                 stride_b_q, stride_h_q, stride_s_q,
                                 k_ptr, 
                                 stride_b_k, stride_h_k, stride_s_k,
                                 v_ptr, 
                                 stride_b_v, stride_h_v, stride_s_v,
                                 is_training,
                                 scaling_factor,
                                 dropout_probability,
                                 philox_seed, philox_offset,
                                 is_causal,
                                 o_ptr,
                                 stride_b_o, stride_h_o, stride_s_o,
                                 lse_ptr,
                                 stream);
  );
  return ret;
}

template <typename DataType>
hipError_t ck_attn_bwd_impl(  
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  const void* q_ptr, 
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  const void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  const void* lse_ptr, 
  const void* do_ptr, 
  uint64_t stride_b_do, uint64_t stride_h_do, uint64_t stride_s_do,
  bool is_training,
  float scaling_factor,
  float dropout_probability,
  uint64_t philox_seed, uint64_t philox_offset,
  bool is_causal,
  void* dq_ptr, 
  uint64_t stride_b_dq, uint64_t stride_h_dq, uint64_t stride_s_dq,
  void* dk_ptr, 
  uint64_t stride_b_dk, uint64_t stride_h_dk, uint64_t stride_s_dk,
  void* dv_ptr, 
  uint64_t stride_b_dv, uint64_t stride_h_dv, uint64_t stride_s_dv,
  void* workspace_ptr,
  hipStream_t stream){
  using TypeConfig = FmhaBwdTypeConfig<DataType>;
  using QDataType = typename TypeConfig::QDataType;
  using KDataType = typename TypeConfig::KDataType;
  using VDataType = typename TypeConfig::VDataType;
  using GemmDataType = typename TypeConfig::GemmDataType;
  using BiasDataType = typename TypeConfig::BiasDataType;
  using LSEDataType = typename TypeConfig::LSEDataType;
  using AccDataType = typename TypeConfig::AccDataType;
  using DDataType = typename TypeConfig::DDataType;
  using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
  using ODataType = typename TypeConfig::ODataType;
  using OGradDataType = typename TypeConfig::OGradDataType;
  using QGradDataType = typename TypeConfig::QGradDataType;
  using KGradDataType = typename TypeConfig::KGradDataType;
  using VGradDataType = typename TypeConfig::VGradDataType;
  using BiasGradDataType = typename TypeConfig::BiasGradDataType;

  bool has_dropout = (dropout_probability > 0.f);
  bool has_dbias = false;

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t seqlen_q = s_q;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d;
  ck_tile::index_t seqlen_k = s_kv;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_k = s_kv;
  float scale_s = scaling_factor;
  float scale_p = 1.f;
  float scale_o = 1.f;
  float p_drop = dropout_probability;
  float p_undrop = 1.0 - p_drop;
  float rp_undrop = 1.0 / p_undrop;
  bool is_group_mode = false;
  bool is_v_rowmajor = true;
  bool do_fp8_static_quant = false;

  bias_enum bias_type;
  mask_enum mask_type;
  int32_t left, right;
  ck_tile::stream_config stream_config{stream};

  if (is_causal){
    mask_type = mask_enum::mask_top_left;
    left = -1;
    right = 0;
  } else {
    mask_type = mask_enum::no_mask;
  }

  ck_tile::index_t shape_batch = batch;
  ck_tile::index_t shape_seqlen_q = seqlen_q;
  ck_tile::index_t shape_seqlen_k = seqlen_k;

  std::string data_type(TypeName<DataType>::get());

  auto fmha_traits =
    fmha_bwd_traits{hdim_q,    hdim_v,    data_type, is_group_mode,
                    mask_type, bias_type, has_dbias, has_dropout};

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // TODO: support bias later
    const ck_tile::index_t stride_bias = 0;
    const ck_tile::index_t stride_o = stride_s_o;
    const ck_tile::index_t stride_randval = max_seqlen_k;
    const ck_tile::index_t stride_do = stride_s_do;
    const ck_tile::index_t stride_dk = stride_s_dk;
    const ck_tile::index_t stride_dv = stride_s_dv;
    // TODO: support bias later
    const ck_tile::index_t stride_dbias = 0;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    // TODO: support bias later
    const ck_tile::index_t nhead_stride_bias = 0;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    const ck_tile::index_t nhead_stride_randval =
        shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t nhead_stride_do = stride_h_do;
    // TODO: buffer?
    const ck_tile::index_t nhead_stride_lsed = max_seqlen_q;
    const ck_tile::index_t nhead_stride_dbias = max_seqlen_k;
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = stride_b_q;
    const ck_tile::index_t batch_stride_k = stride_b_k;
    const ck_tile::index_t batch_stride_v = stride_b_v;
    const ck_tile::index_t batch_stride_bias = 0;
    const ck_tile::index_t batch_stride_o = stride_b_o;
    const ck_tile::index_t batch_stride_randval =
        nhead * shape_seqlen_q * max_seqlen_k;
    const ck_tile::index_t batch_stride_do = stride_b_do;
    const ck_tile::index_t batch_stride_lsed = nhead * max_seqlen_q;
    const ck_tile::index_t batch_stride_dk = stride_b_dk;
    const ck_tile::index_t batch_stride_dv = stride_b_dv;
    const ck_tile::index_t batch_stride_dbias =
        nhead * shape_seqlen_q * max_seqlen_k;

    return fmha_bwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         nullptr,
                         o_ptr,
                         lse_ptr,
                         do_ptr,
                         workspace_ptr,
                         nullptr,
                         dq_ptr,
                         dk_ptr,
                         dv_ptr,
                         nullptr,
                         nullptr,//cu_seqlen_q
                         nullptr,//cu_seqlen_kv
                         nullptr, /* seqlen_k_ptr */
                         shape_seqlen_q,
                         shape_seqlen_k,
                         batch,
                         max_seqlen_q,
                         max_seqlen_k,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_bias,
                         stride_o,
                         stride_randval,
                         stride_do,
                         stride_dk,
                         stride_dv,
                         stride_dbias,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         nhead_stride_bias,
                         nhead_stride_o,
                         nhead_stride_randval,
                         nhead_stride_do,
                         nhead_stride_lsed,
                         nhead_stride_dbias,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         batch_stride_bias,
                         batch_stride_o,
                         batch_stride_randval,
                         batch_stride_do,
                         batch_stride_lsed,
                         batch_stride_dk,
                         batch_stride_dv,
                         batch_stride_dbias,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         p_drop,
                         p_undrop,
                         false,
                         {philox_seed, philox_offset}};
  }();
  fmha_bwd(fmha_traits, fmha_args, stream_config);
  return hipSuccess;
}


hipError_t ck_attn_bwd(  
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  const void* q_ptr, 
  uint64_t stride_b_q, uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_b_k, uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_b_v, uint64_t stride_h_v, uint64_t stride_s_v,
  const void* o_ptr, 
  uint64_t stride_b_o, uint64_t stride_h_o, uint64_t stride_s_o,
  const void* lse_ptr, 
  const void* do_ptr, 
  uint64_t stride_b_do, uint64_t stride_h_do, uint64_t stride_s_do,
  bool is_training,
  float scaling_factor,
  float dropout_probability,
  uint64_t philox_seed, uint64_t philox_offset,
  bool is_causal,
  void* dq_ptr, 
  uint64_t stride_b_dq, uint64_t stride_h_dq, uint64_t stride_s_dq,
  void* dk_ptr, 
  uint64_t stride_b_dk, uint64_t stride_h_dk, uint64_t stride_s_dk,
  void* dv_ptr, 
  uint64_t stride_b_dv, uint64_t stride_h_dv, uint64_t stride_s_dv,
  void* workspace_ptr,
  hipStream_t stream){

  hipError_t ret;
  CK_FUSED_ATTN_TYPE_SWITCH_ALL(dtype, type,
    ret = ck_attn_bwd_impl<type>(b, h, hg, s_q, s_kv, d,
                                 q_ptr, 
                                 stride_b_q, stride_h_q, stride_s_q,
                                 k_ptr, 
                                 stride_b_k, stride_h_k, stride_s_k,
                                 v_ptr, 
                                 stride_b_v, stride_h_v, stride_s_v,
                                 o_ptr, 
                                 stride_b_o, stride_h_o, stride_s_o,
                                 lse_ptr,
                                 do_ptr, 
                                 stride_b_do, stride_h_do, stride_s_do,
                                 is_training,
                                 scaling_factor,
                                 dropout_probability,
                                 philox_seed, philox_offset,
                                 is_causal,
                                 dq_ptr, 
                                 stride_b_dq, stride_h_dq, stride_s_dq,
                                 dk_ptr, 
                                 stride_b_dk, stride_h_dk, stride_s_dk,
                                 dv_ptr, 
                                 stride_b_dv, stride_h_dv, stride_s_dv,
                                 workspace_ptr,
                                 stream);
  );
  return ret;
}


}//namespace ck_fused_attn
