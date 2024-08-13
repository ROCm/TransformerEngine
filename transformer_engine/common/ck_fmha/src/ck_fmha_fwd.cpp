#include <stdexcept>

#include "bias.hpp"
#include "ck_fmha.h"
#include "ck_tile/host.hpp"
#include "fmha_fwd.hpp"
#include "mask.hpp"

void ck_fused_attn_fwd_impl(int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                            int64_t bias_b, int64_t bias_h, bool is_training, float scaling_factor,
                            float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
                            uint32_t bias_value, uint32_t mask_value, void *devPtrQ, void *devPtrK,
                            void *devPtrV, void *devPtrBias, void *devPtrSoftmaxStats,
                            void *devPtrO, void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, hipStream_t stream) {
    /* CK input parameters */
    ck_tile::index_t batch        = b;
    ck_tile::index_t seqlen_q     = s_q;
    ck_tile::index_t nhead        = h;
    ck_tile::index_t hdim_q       = d;
    ck_tile::index_t seqlen_k     = s_kv;
    ck_tile::index_t nhead_k      = hg;
    ck_tile::index_t hdim_v       = d;
    ck_tile::index_t max_seqlen_q = s_q;
    float scale_s                 = scaling_factor;
    float scale_p                 = 1.f;
    float scale_o                 = 1.f;
    float p_drop                  = dropout_probability;
    bool is_group_mode            = false;
    bool is_v_rowmajor            = true;
    bool do_fp8_static_quant      = false;
    bool has_dropout              = (is_training && p_drop > 0.f);
    bool has_lse                  = (devPtrSoftmaxStats != nullptr);

    bias_enum bias_type;
    mask_enum mask_type;
    ck_tile::index_t left, right;
    ck_tile::stream_config stream_config{stream};
    if (nhead % nhead_k != 0) {
        throw std::invalid_argument("nhead must be a multiple of nhead_k");
    }
    if (bias_value == 0) {
        bias_type = bias_enum::no_bias;
    } else {
        throw std::runtime_error("Unsupported bias type");
    }

    if (mask_value == 0) {
        mask_type = mask_enum::no_mask;
    } else if (mask_value == 1) {
        mask_type = mask_enum::mask_top_left;
        left      = -1;
        right     = 0;
    } else {
        throw std::runtime_error("Unsupported mask type");
    }

    ck_tile::index_t shape_seqlen_q = seqlen_q;
    ck_tile::index_t shape_seqlen_k = seqlen_k;
    bool s_randval                  = false;

    auto fmha_traits =
        fmha_fwd_traits{hdim_q,    hdim_v,    data_type, is_group_mode, is_v_rowmajor,
                        mask_type, bias_type, has_lse,   has_dropout,   do_fp8_static_quant};

    auto fmha_args = [&]() {
        // setup stride_* arguments
        const ck_tile::index_t stride_q       = nhead * hdim_q;
        const ck_tile::index_t stride_k       = nhead_k * hdim_q;
        const ck_tile::index_t stride_v       = nhead_k * hdim_v;
        const ck_tile::index_t stride_bias    = 0;
        const ck_tile::index_t stride_randval = 0;
        const ck_tile::index_t stride_o_acc   = 0;
        const ck_tile::index_t stride_o       = nhead * hdim_v;
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q       = hdim_q;
        const ck_tile::index_t nhead_stride_k       = hdim_q;
        const ck_tile::index_t nhead_stride_v       = hdim_v;
        const ck_tile::index_t nhead_stride_bias    = 0;
        const ck_tile::index_t nhead_stride_randval = 0;
        const ck_tile::index_t nhead_stride_lse     = max_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = 0;
        const ck_tile::index_t nhead_stride_o_acc   = 0;
        const ck_tile::index_t nhead_stride_o       = hdim_v;
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = nhead * shape_seqlen_q * hdim_q;
        const ck_tile::index_t batch_stride_k       = nhead_k * shape_seqlen_k * hdim_q;
        const ck_tile::index_t batch_stride_v       = nhead_k * shape_seqlen_k * hdim_v;
        const ck_tile::index_t batch_stride_bias    = 0;
        const ck_tile::index_t batch_stride_randval = 0;
        const ck_tile::index_t batch_stride_lse     = nhead * max_seqlen_q;
        const ck_tile::index_t batch_stride_lse_acc = 0;
        const ck_tile::index_t batch_stride_o_acc   = 0;
        const ck_tile::index_t batch_stride_o       = nhead * shape_seqlen_q * hdim_v;
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = batch * nhead * max_seqlen_q;
        const ck_tile::index_t split_stride_o_acc   = batch * nhead * max_seqlen_q * hdim_v;

        return fmha_fwd_args{devPtrQ,
                             devPtrK,
                             devPtrV,
                             devPtrBias,
                             nullptr,
                             nullptr,
                             nullptr,
                             devPtrSoftmaxStats,
                             devPtrO,
                             devPtrCuSeqlensQ,
                             devPtrCuSeqlensKV,
                             nullptr,
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             1, /* num_splits */
                             scale_s,
                             scale_p,
                             scale_o,
                             stride_q,
                             stride_k,
                             stride_v,
                             stride_bias,
                             stride_randval,
                             stride_o_acc,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_randval,
                             nhead_stride_lse,
                             nhead_stride_lse_acc,
                             nhead_stride_o_acc,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_randval,
                             batch_stride_lse,
                             batch_stride_lse_acc,
                             batch_stride_o_acc,
                             batch_stride_o,
                             split_stride_lse_acc,
                             split_stride_o_acc,
                             left,
                             right,
                             static_cast<ck_tile::index_t>(mask_type),
                             p_drop,
                             s_randval,
                             {drop_seed, drop_offset}};
    }();

    float avg_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
    if (avg_time < 0.f) {
        throw std::runtime_error("Forward pass: Not supported yet");
    }
}
