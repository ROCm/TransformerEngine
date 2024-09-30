#include <stdexcept>

#include "bias.hpp"
#include "ck_fmha.h"
#include "ck_tile/host.hpp"
#include "fmha_bwd.hpp"
#include "mask.hpp"

// #include <mutex>
// std::mutex g_mtx;

template <typename DataType>
__global__ void reshape_and_sum(DataType *dk, const DataType *dk_expanded, DataType *dv,
                                const DataType *dv_expanded, int batch_size, int seqlen_k,
                                int num_heads, int num_heads_k, int head_size) {
    static_assert(std::is_arithmetic<DataType>::value,
                  "reshape_and_sum only supports arithmetic types");
}

template <>
__global__ void reshape_and_sum<ck_tile::half_t>(ck_tile::half_t *dk,
                                                 const ck_tile::half_t *dk_expanded,
                                                 ck_tile::half_t *dv,
                                                 const ck_tile::half_t *dv_expanded, int batch_size,
                                                 int seqlen_k, int num_heads, int num_heads_k,
                                                 int head_size) {
    int batch_idx       = blockIdx.x;
    int seqlen_idx      = blockIdx.y;
    int head_k_idx      = blockIdx.z;
    int thread_idx      = threadIdx.x;
    int head_idx_offset = num_heads / num_heads_k;

    if (thread_idx < head_size) {
        float sum_dk = 0.0f;
        float sum_dv = 0.0f;
        int read_idx = ((batch_idx * seqlen_k + seqlen_idx) * num_heads_k + head_k_idx) *
                           head_idx_offset * head_size +
                       thread_idx;
        int write_idx =
            ((batch_idx * seqlen_k + seqlen_idx) * num_heads_k + head_k_idx) * head_size +
            thread_idx;
        for (int j = 0; j < head_idx_offset; j++) {
            sum_dk += dk_expanded[read_idx];
            sum_dv += dv_expanded[read_idx];
            read_idx += head_size;
        }
        dk[write_idx] = sum_dk;
        dv[write_idx] = sum_dv;
    }
}

template <>
__global__ void reshape_and_sum<ck_tile::bf16_t>(ck_tile::bf16_t *dk,
                                                 const ck_tile::bf16_t *dk_expanded,
                                                 ck_tile::bf16_t *dv,
                                                 const ck_tile::bf16_t *dv_expanded, int batch_size,
                                                 int seqlen_k, int num_heads, int num_heads_k,
                                                 int head_size) {
    int batch_idx       = blockIdx.x;
    int seqlen_idx      = blockIdx.y;
    int head_k_idx      = blockIdx.z;
    int thread_idx      = threadIdx.x;
    int head_idx_offset = num_heads / num_heads_k;

    if (thread_idx < head_size) {
        float sum_dk = 0.0f;
        float sum_dv = 0.0f;
        int read_idx = ((batch_idx * seqlen_k + seqlen_idx) * num_heads_k + head_k_idx) *
                           head_idx_offset * head_size +
                       thread_idx;
        int write_idx =
            ((batch_idx * seqlen_k + seqlen_idx) * num_heads_k + head_k_idx) * head_size +
            thread_idx;
        for (int j = 0; j < head_idx_offset; j++) {
            sum_dk += ck_tile::bf16_to_float(dk_expanded[read_idx]);
            sum_dv += ck_tile::bf16_to_float(dv_expanded[read_idx]);
            read_idx += head_size;
        }
        dk[write_idx] = ck_tile::float_to_bf16(sum_dk);
        dv[write_idx] = ck_tile::float_to_bf16(sum_dv);
    }
}

void ck_fused_attn_bwd_impl(int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d,
                            int64_t bias_b, int64_t bias_h, float scaling_factor,
                            float dropout_probability, uint64_t drop_seed, uint64_t drop_offset,
                            uint32_t bias_type, uint32_t mask_type, void *devPtrQ,
                            void *devPtrKTranspose, void *devPtrVTranspose, void *devPtrO,
                            void *devPtrSoftmaxStats, void *devPtrBias, void *devPtrdQ,
                            void *devPtrdK, void *devPtrdV, void *devPtrdO, void *devPtrdBias,
                            void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, void *workspace, size_t *workspace_size,
                            bool deterministic, hipStream_t stream) {
    /* CK input parameters */
    ck_tile::index_t batch          = b;
    ck_tile::index_t seqlen_q       = s_q;
    ck_tile::index_t nhead          = h;
    ck_tile::index_t hdim_q         = d;
    ck_tile::index_t seqlen_k       = s_kv;
    ck_tile::index_t nhead_k        = hg;
    ck_tile::index_t hdim_v         = d;
    ck_tile::index_t max_seqlen_q   = s_q;
    ck_tile::index_t max_seqlen_k   = s_kv;
    ck_tile::index_t shape_batch    = b;
    ck_tile::index_t shape_seqlen_q = s_q;
    ck_tile::index_t shape_seqlen_k = s_kv;
    ck_tile::index_t kN0            = (hdim_q <= 128) ? 128 : 64;
    ck_tile::index_t nsplits = deterministic ? ck_tile::integer_divide_ceil(max_seqlen_k, kN0) : 1;

    float scale_s          = scaling_factor;
    float p_drop           = dropout_probability;
    float p_undrop         = 1.0 - p_drop;
    bool  is_group_mode    = false;
    bool  is_mqa_gqa       = (h > hg);
    bool  has_dropout      = (p_drop > 0.f);
    bool  has_dbias        = (devPtrdBias != nullptr);
    bool  s_randval        = false;
    bool  ext_asm          = true;
    bool  asm_atomic_fp32  = false;
    bool  asm_no_coex      = false;
    bool  asm_rtz_cvt      = true;

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

    constexpr size_t float_size = sizeof(float);
    constexpr size_t bf16_size  = sizeof(ck_tile::bf16_t);

    size_t d_size           = float_size * batch * nhead * max_seqlen_q;
    size_t dq_size          = bf16_size * batch * shape_seqlen_q * nhead * hdim_q;
    //size_t dq_acc_size      = float_size * nsplits * shape_batch * shape_seqlen_q * nhead * hdim_q;
    //size_t dq_acc_size      = 0;
    size_t dk_expanded_size = 0;
    size_t dv_expanded_size = 0;

    if (is_mqa_gqa) {
        dk_expanded_size = bf16_size * batch * nhead * shape_seqlen_k * hdim_q;
        dv_expanded_size = bf16_size * batch * nhead * shape_seqlen_k * hdim_v;
    }

    if (workspace == nullptr) {
        //*workspace_size = d_size + dq_acc_size + dk_expanded_size + dv_expanded_size;
        *workspace_size = d_size + dk_expanded_size + dv_expanded_size;
        return;
    }

    void *devPtrD          = workspace;
    //void *devPtrdQAcc      = reinterpret_cast<void *>(reinterpret_cast<char *>(devPtrD) + d_size);
    void *devPtrdKExpanded = nullptr;
    void *devPtrdVExpanded = nullptr;
    CHECK_HIP_ERROR(hipMemsetAsync(devPtrdQ, 0, dq_size, stream));
    //CHECK_HIP_ERROR(hipMemsetAsync(devPtrdQAcc, 0, dq_acc_size, stream));
    if (is_mqa_gqa) {
        devPtrdKExpanded =
            reinterpret_cast<void *>(reinterpret_cast<char *>(devPtrD) + d_size);
            // reinterpret_cast<void *>(reinterpret_cast<char *>(devPtrdQAcc) + dq_acc_size);
        devPtrdVExpanded =
            reinterpret_cast<void *>(reinterpret_cast<char *>(devPtrdKExpanded) + dk_expanded_size);
    }

    const auto init_traits = [&](auto &traits) {
        traits.hdim_q             = hdim_q;
        traits.hdim_v             = hdim_v;
        traits.data_type          = data_type;
        traits.is_group_mode      = is_group_mode;
        traits.mask_type          = static_cast<mask_enum>(mask_type);
        traits.bias_type          = static_cast<bias_enum>(bias_type);
        traits.has_dbias          = has_dbias;
        traits.has_dropout        = has_dropout;
        traits.is_store_randval   = s_randval;
        traits.is_deterministic   = deterministic;
        traits.uses_ext_asm       = ext_asm;
        traits.is_asm_atomic_fp32 = asm_atomic_fp32;
        traits.is_asm_no_coex     = asm_no_coex;
        traits.is_asm_rtz_cvt     = asm_rtz_cvt;
    };

    const auto init_args = [&](auto &args) {
        const ck_tile::index_t stride_q       = nhead * hdim_q;
        const ck_tile::index_t stride_k       = nhead_k * hdim_q;
        const ck_tile::index_t stride_v       = nhead_k * hdim_v;
        const ck_tile::index_t stride_bias    = max_seqlen_k;
        const ck_tile::index_t stride_o       = nhead * hdim_v;
        const ck_tile::index_t stride_randval = max_seqlen_k;
        const ck_tile::index_t stride_do      = nhead * hdim_v;
        const ck_tile::index_t stride_dk      = nhead * hdim_q;
        const ck_tile::index_t stride_dv      = nhead * hdim_v;
        const ck_tile::index_t stride_dbias   = nhead * max_seqlen_k;

        const ck_tile::index_t nhead_stride_q       = hdim_q;
        const ck_tile::index_t nhead_stride_k       = hdim_q;
        const ck_tile::index_t nhead_stride_v       = hdim_v;
        const ck_tile::index_t nhead_stride_bias    = 0;
        const ck_tile::index_t nhead_stride_o       = hdim_v;
        const ck_tile::index_t nhead_stride_randval = shape_seqlen_q * max_seqlen_k;
        const ck_tile::index_t nhead_stride_do      = hdim_v;
        const ck_tile::index_t nhead_stride_lsed    = max_seqlen_q;
        const ck_tile::index_t nhead_stride_dbias   = max_seqlen_k;

        const ck_tile::index_t batch_stride_q       = nhead * shape_seqlen_q * hdim_q;
        const ck_tile::index_t batch_stride_k       = nhead_k * shape_seqlen_k * hdim_q;
        const ck_tile::index_t batch_stride_v       = nhead_k * shape_seqlen_k * hdim_v;
        const ck_tile::index_t batch_stride_bias    = 0;
        const ck_tile::index_t batch_stride_o       = nhead * shape_seqlen_q * hdim_v;
        const ck_tile::index_t batch_stride_randval = nhead * shape_seqlen_q * max_seqlen_k;
        const ck_tile::index_t batch_stride_do      = nhead * shape_seqlen_q * hdim_v;
        const ck_tile::index_t batch_stride_lsed    = nhead * max_seqlen_q;
        const ck_tile::index_t batch_stride_dk      = nhead * shape_seqlen_k * hdim_q;
        const ck_tile::index_t batch_stride_dv      = nhead * shape_seqlen_k * hdim_v;
        const ck_tile::index_t batch_stride_dbias   = nhead * shape_seqlen_q * max_seqlen_k;

        const ck_tile::index_t split_stride_dq_acc = shape_batch * nhead * shape_seqlen_q * hdim_q;

        args.q_ptr    = devPtrQ;
        args.k_ptr    = devPtrKTranspose;
        args.v_ptr    = devPtrVTranspose;
        args.bias_ptr = devPtrBias;
        args.o_ptr    = devPtrO;
        args.lse_ptr  = devPtrSoftmaxStats;
        args.do_ptr   = devPtrdO;

        args.d_ptr        = devPtrD;
        args.rand_val_ptr = nullptr;
        args.dq_ptr       = devPtrdQ;
        args.dk_ptr       = (is_mqa_gqa ? devPtrdKExpanded : devPtrdK);
        args.dv_ptr       = (is_mqa_gqa ? devPtrdVExpanded : devPtrdV);
        args.dbias_ptr    = devPtrdBias;
        args.dq_acc_ptr   = nullptr;
        // args.dq_acc_ptr   = devPtrdQAcc;

        args.seqstart_q_ptr = devPtrCuSeqlensQ;
        args.seqstart_k_ptr = devPtrCuSeqlensKV;
        args.seqlen_k_ptr   = nullptr;

        args.seqlen_q     = shape_seqlen_q;
        args.seqlen_k     = shape_seqlen_k;
        args.batch        = batch;
        args.max_seqlen_q = max_seqlen_q;
        args.max_seqlen_k = max_seqlen_k;
        args.hdim_q       = hdim_q;
        args.hdim_v       = hdim_v;
        args.nhead_q      = nhead;
        args.nhead_k      = nhead_k;

        args.scale = scale_s;

        args.stride_q       = stride_q;
        args.stride_k       = stride_k;
        args.stride_v       = stride_v;
        args.stride_bias    = stride_bias;
        args.stride_o       = stride_o;
        args.stride_randval = stride_randval;
        args.stride_do      = stride_do;
        args.stride_dq_acc  = stride_q;
        args.stride_dq      = stride_q;
        args.stride_dk      = stride_dk;
        args.stride_dv      = stride_dv;
        args.stride_dbias   = stride_dbias;

        args.nhead_stride_q       = nhead_stride_q;
        args.nhead_stride_k       = nhead_stride_k;
        args.nhead_stride_v       = nhead_stride_v;
        args.nhead_stride_bias    = nhead_stride_bias;
        args.nhead_stride_o       = nhead_stride_o;
        args.nhead_stride_randval = nhead_stride_randval;
        args.nhead_stride_do      = nhead_stride_do;
        args.nhead_stride_lsed    = nhead_stride_lsed;
        args.nhead_stride_dq_acc  = nhead_stride_q;
        args.nhead_stride_dq      = nhead_stride_q;
        args.nhead_stride_dk      = nhead_stride_k;
        args.nhead_stride_dv      = nhead_stride_v;
        args.nhead_stride_dbias   = nhead_stride_dbias;

        args.batch_stride_q       = batch_stride_q;
        args.batch_stride_k       = batch_stride_k;
        args.batch_stride_v       = batch_stride_v;
        args.batch_stride_bias    = batch_stride_bias;
        args.batch_stride_o       = batch_stride_o;
        args.batch_stride_randval = batch_stride_randval;
        args.batch_stride_do      = batch_stride_do;
        args.batch_stride_lsed    = batch_stride_lsed;
        args.batch_stride_dq_acc  = batch_stride_q;
        args.batch_stride_dq      = batch_stride_q;
        args.batch_stride_dk      = batch_stride_dk;
        args.batch_stride_dv      = batch_stride_dv;
        args.batch_stride_dbias   = batch_stride_dbias;

        args.split_stride_dq_acc = split_stride_dq_acc;

        args.window_size_left  = window_size_left;
        args.window_size_right = window_size_right;
        args.mask_type         = mask_type;

        args.p_drop           = p_drop;
        args.p_undrop         = p_undrop;
        args.drop_seed_offset = std::make_tuple(drop_seed, drop_offset);
    };

    fmha_bwd_traits fmha_traits;
    init_traits(fmha_traits);
    /*
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        std::cout << "fmha_traits:" << std::endl;
        std::cout << "  hdim_q: " << fmha_traits.hdim_q << std::endl;
        std::cout << "  hdim_v: " << fmha_traits.hdim_v << std::endl;
        std::cout << "  data_type: " << fmha_traits.data_type << std::endl;
        std::cout << "  is_group_mode: " << fmha_traits.is_group_mode << std::endl;
        std::cout << "  mask_type: " << static_cast<uint32_t>(fmha_traits.mask_type) << std::endl;
        std::cout << "  bias_type: " << static_cast<uint32_t>(fmha_traits.bias_type) << std::endl;
        std::cout << "  has_dbias: " << fmha_traits.has_dbias << std::endl;
        std::cout << "  has_dropout: " << fmha_traits.has_dropout << std::endl;
        std::cout << "  is_store_randval: " << fmha_traits.is_store_randval << std::endl;
        std::cout << "  is_deterministic: " << fmha_traits.is_deterministic << std::endl;
        std::cout << "  uses_ext_asm: " << fmha_traits.uses_ext_asm << std::endl;
        std::cout << "  is_asm_atomic_fp32: " << fmha_traits.is_asm_atomic_fp32 << std::endl;
        std::cout << "  is_asm_no_coex: " << fmha_traits.is_asm_no_coex << std::endl;
    }
    */
    fmha_bwd_args fmha_args;
    init_args(fmha_args);

    ck_tile::stream_config stream_config{stream};

    float avg_time = fmha_bwd(fmha_traits, fmha_args, stream_config);
    if (avg_time < 0.f) {
        throw std::runtime_error("Backward pass: Not supported yet");
    }

    if (is_mqa_gqa) {
        dim3 grid(batch, seqlen_k, nhead_k);
        dim3 block(hdim_q);

        // Launch the kernel for devPtrdK & devPtrdV
        hipLaunchKernelGGL(reshape_and_sum<ck_tile::bf16_t>, grid, block, 0, stream,
                           static_cast<ck_tile::bf16_t *>(devPtrdK),
                           static_cast<ck_tile::bf16_t *>(devPtrdKExpanded),
                           static_cast<ck_tile::bf16_t *>(devPtrdV),
                           static_cast<ck_tile::bf16_t *>(devPtrdVExpanded), batch, seqlen_k, nhead,
                           nhead_k, hdim_q);
        //CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }
}
