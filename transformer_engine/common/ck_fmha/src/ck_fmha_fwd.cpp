#include <stdexcept>

#include "bias.hpp"
#include "ck_fmha.h"
#include "ck_tile/host.hpp"
#include "fmha_fwd.hpp"
#include "mask.hpp"

void ck_fused_attn_fwd_impl(int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                            int64_t bias_b, int64_t bias_h, bool is_training, float scaling_factor,
                            float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
                            uint32_t bias_type, uint32_t mask_type, void *devPtrQ, void *devPtrK,
                            void *devPtrV, void *devPtrBias, void *devPtrSoftmaxStats,
                            void *devPtrO, void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, void *workspace, size_t *workspace_size,
                            hipStream_t stream) {
    /* CK input parameters */
    ck_tile::index_t batch          = b;
    ck_tile::index_t seqlen_q       = s_q;
    ck_tile::index_t nhead          = h;
    ck_tile::index_t hdim_q         = d;
    ck_tile::index_t seqlen_k       = s_kv;
    ck_tile::index_t nhead_k        = hg;
    ck_tile::index_t hdim_v         = d;
    ck_tile::index_t max_seqlen_q   = s_q;
    ck_tile::index_t shape_batch    = b;
    ck_tile::index_t shape_seqlen_q = s_q;
    ck_tile::index_t shape_seqlen_k = s_kv;

    float scale_s            = scaling_factor;
    float scale_p            = 1.f;
    float scale_o            = 1.f;
    float p_drop             = dropout_probability;
    bool s_randval           = false;
    bool is_group_mode       = false;
    bool is_v_rowmajor       = true;
    bool do_fp8_static_quant = false;
    bool has_dropout         = (is_training && p_drop > 0.f);
    bool has_lse             = (devPtrSoftmaxStats != nullptr);

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;

    if (nhead % nhead_k != 0) {
        throw std::invalid_argument("nhead must be a multiple of nhead_k");
    }

    if (bias_type != 0) {
        throw std::runtime_error("Unsupported bias type");
    }

    if (mask_type == 0 || mask_type == 1) {
        window_size_left  = -1;
        window_size_right = 0;
    } else {
        throw std::runtime_error("Unsupported mask type");
    }

    if (workspace == nullptr) {
        // CK FMHA FWD does not require any additional workspace memory
        *workspace_size = 0;
        return;
    }

    const auto init_traits = [&](auto &traits) {
        // fmha_fwd_traits or fmha_splitkv_traits
        traits.hdim_q              = hdim_q;
        traits.hdim_v              = hdim_v;
        traits.data_type           = data_type;
        traits.is_group_mode       = is_group_mode;
        traits.is_v_rowmajor       = is_v_rowmajor;
        traits.mask_type           = static_cast<mask_enum>(mask_type);
        traits.bias_type           = static_cast<bias_enum>(bias_type);
        traits.has_lse             = has_lse;
        traits.has_dropout         = has_dropout;
        traits.do_fp8_static_quant = do_fp8_static_quant;
    };

    const auto init_args = [&](auto &args) {
        const ck_tile::index_t stride_q       = nhead * hdim_q;
        const ck_tile::index_t stride_k       = nhead_k * hdim_q;
        const ck_tile::index_t stride_v       = nhead_k * hdim_v;
        const ck_tile::index_t stride_bias    = shape_seqlen_k;
        const ck_tile::index_t stride_randval = 0;
        const ck_tile::index_t stride_o       = nhead * hdim_v;

        const ck_tile::index_t nhead_stride_q       = hdim_q;
        const ck_tile::index_t nhead_stride_k       = hdim_q;
        const ck_tile::index_t nhead_stride_v       = hdim_v;
        const ck_tile::index_t nhead_stride_bias    = 0;
        const ck_tile::index_t nhead_stride_randval = 0;
        const ck_tile::index_t nhead_stride_lse     = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_o       = hdim_v;

        const ck_tile::index_t batch_stride_q       = nhead * shape_seqlen_q * hdim_q;
        const ck_tile::index_t batch_stride_k       = nhead_k * shape_seqlen_k * hdim_q;
        const ck_tile::index_t batch_stride_v       = nhead_k * hdim_v * shape_seqlen_k;
        const ck_tile::index_t batch_stride_bias    = 0;
        const ck_tile::index_t batch_stride_randval = 0;
        const ck_tile::index_t batch_stride_lse     = nhead * shape_seqlen_q;
        const ck_tile::index_t batch_stride_o       = nhead * shape_seqlen_q * hdim_v;

        args.q_ptr        = devPtrQ;
        args.k_ptr        = devPtrK;
        args.v_ptr        = devPtrV;
        args.bias_ptr     = devPtrBias;
        args.rand_val_ptr = nullptr;
        args.lse_ptr      = devPtrSoftmaxStats;
        args.o_ptr        = devPtrO;

        args.seqstart_q_ptr = devPtrCuSeqlensQ;
        args.seqstart_k_ptr = devPtrCuSeqlensKV;
        args.seqlen_k_ptr   = nullptr;

        args.scale_s = scale_s;
        args.scale_p = scale_p;
        args.scale_o = scale_o;

        args.seqlen_q     = shape_seqlen_q;
        args.seqlen_k     = shape_seqlen_k;
        args.batch        = batch;
        args.max_seqlen_q = max_seqlen_q;
        args.hdim_q       = hdim_q;
        args.hdim_v       = hdim_v;
        args.nhead_q      = nhead;
        args.nhead_k      = nhead_k;

        args.stride_q       = stride_q;
        args.stride_k       = stride_k;
        args.stride_v       = stride_v;
        args.stride_bias    = stride_bias;
        args.stride_randval = stride_randval;
        args.stride_o       = stride_o;

        args.nhead_stride_q       = nhead_stride_q;
        args.nhead_stride_k       = nhead_stride_k;
        args.nhead_stride_v       = nhead_stride_v;
        args.nhead_stride_bias    = nhead_stride_bias;
        args.nhead_stride_randval = nhead_stride_randval;
        args.nhead_stride_lse     = nhead_stride_lse;
        args.nhead_stride_o       = nhead_stride_o;

        args.batch_stride_q       = batch_stride_q;
        args.batch_stride_k       = batch_stride_k;
        args.batch_stride_v       = batch_stride_v;
        args.batch_stride_bias    = batch_stride_bias;
        args.batch_stride_randval = batch_stride_randval;
        args.batch_stride_lse     = batch_stride_lse;
        args.batch_stride_o       = batch_stride_o;

        args.window_size_left  = window_size_left;
        args.window_size_right = window_size_right;
        args.mask_type         = mask_type;

        args.p_drop           = p_drop;
        args.s_randval        = s_randval;
        args.drop_seed_offset = std::tie(drop_seed, drop_offset);
    };

    fmha_fwd_traits fmha_traits;
    init_traits(fmha_traits);

    fmha_fwd_args fmha_args;
    init_args(fmha_args);

    ck_tile::stream_config stream_config{stream};

    float avg_time = fmha_fwd(fmha_traits, fmha_args, stream_config);
    if (avg_time < 0.f) {
        throw std::runtime_error("Forward pass: Not supported yet");
    }
}
