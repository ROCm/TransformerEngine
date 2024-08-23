#include <stdexcept>

#include "bias.hpp"
#include "ck_fmha.h"
#include "ck_tile/host.hpp"
#include "fmha_bwd.hpp"
#include "mask.hpp"

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
                            uint32_t bias_value, uint32_t mask_value, void *devPtrQ,
                            void *devPtrKTranspose, void *devPtrVTranspose, void *devPtrO,
                            void *devPtrSoftmaxStats, void *devPtrBias, void *devPtrdQ,
                            void *devPtrdK, void *devPtrdV, void *devPtrdO, void *devPtrdBias,
                            void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV,
                            const std::string &data_type, void *workspace, size_t *workspace_size,
                            bool deterministic, hipStream_t stream) {
    /* CK input parameters */
    ck_tile::index_t batch        = b;
    ck_tile::index_t seqlen_q     = s_q;
    ck_tile::index_t nhead        = h;
    ck_tile::index_t hdim_q       = d;
    ck_tile::index_t seqlen_k     = s_kv;
    ck_tile::index_t nhead_k      = hg;
    ck_tile::index_t hdim_v       = d;
    ck_tile::index_t max_seqlen_q = s_q;
    ck_tile::index_t max_seqlen_k = s_kv;
    float scale_s                 = scaling_factor;
    float p_drop                  = dropout_probability;
    float p_undrop                = 1.0 - p_drop;
    bool is_group_mode            = false;
    bool is_mqa_gqa               = (h > hg);
    bool has_dropout              = (p_drop > 0.f);
    bool has_dbias                = (devPtrdBias != nullptr);
    bool s_randval                = false;

    bias_enum bias_type;
    mask_enum mask_type;
    int32_t left, right;
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

    /*
    static thread_local DeviceMemoryManager d_mgr{stream};
    static thread_local DeviceMemoryManager dq_acc_mgr{stream};
    static thread_local DeviceMemoryManager dk_expanded_mgr{stream};
    static thread_local DeviceMemoryManager dv_expanded_mgr{stream};
    */

    const ck_tile::index_t shape_batch    = batch;
    const ck_tile::index_t shape_seqlen_q = seqlen_q;
    const ck_tile::index_t shape_seqlen_k = seqlen_k;
    const ck_tile::index_t kN0            = (hdim_q <= 128) ? 128 : 64;
    const ck_tile::index_t nsplits =
        deterministic ? ck_tile::integer_divide_ceil(max_seqlen_k, kN0) : 1;

    /*
    d_mgr.resize(sizeof(float) * batch * nhead * max_seqlen_q);
    dq_acc_mgr.resize(sizeof(float) * nsplits * shape_batch * shape_seqlen_q *
                      nhead * hdim_q);
    if (is_mqa_gqa) {
      dk_expanded_mgr.resize(sizeof(ck_tile::bf16_t) * batch * nhead *
                             shape_seqlen_k * hdim_q);
      dv_expanded_mgr.resize(sizeof(ck_tile::bf16_t) * batch * nhead *
                             shape_seqlen_k * hdim_v);
    }
    dq_acc_mgr.set_zero();
    */
    constexpr size_t float_size = sizeof(float);
    constexpr size_t bf16_size  = sizeof(ck_tile::bf16_t);

    size_t d_size           = float_size * batch * nhead * max_seqlen_q;
    size_t dq_acc_size      = float_size * nsplits * shape_batch * shape_seqlen_q * nhead * hdim_q;
    size_t dk_expanded_size = 0;
    size_t dv_expanded_size = 0;

    if (is_mqa_gqa) {
        dk_expanded_size = bf16_size * batch * nhead * shape_seqlen_k * hdim_q;
        dv_expanded_size = bf16_size * batch * nhead * shape_seqlen_k * hdim_v;
    }
    if (workspace == nullptr) {
        *workspace_size = d_size + dq_acc_size + dk_expanded_size + dv_expanded_size;
        return;
    }

    void *devPtrD          = workspace;
    void *devPtrdQAcc      = reinterpret_cast<void *>(reinterpret_cast<char *>(devPtrD) + d_size);
    void *devPtrdKExpanded = nullptr;
    void *devPtrdVExpanded = nullptr;
    CHECK_HIP_ERROR(hipMemsetAsync(devPtrdQAcc, 0, dq_acc_size, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    if (is_mqa_gqa) {
        devPtrdKExpanded =
            reinterpret_cast<void *>(reinterpret_cast<char *>(devPtrdQAcc) + dq_acc_size);
        devPtrdVExpanded =
            reinterpret_cast<void *>(reinterpret_cast<char *>(devPtrdKExpanded) + dk_expanded_size);
    }

    auto fmha_traits =
        fmha_bwd_traits{hdim_q,    hdim_v,    data_type,   is_group_mode, mask_type,
                        bias_type, has_dbias, has_dropout, s_randval,     deterministic};

    auto fmha_args = [&]() {
        // setup stride_* arguments
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
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q       = hdim_q;
        const ck_tile::index_t nhead_stride_k       = hdim_q;
        const ck_tile::index_t nhead_stride_v       = hdim_v;
        const ck_tile::index_t nhead_stride_bias    = 0;
        const ck_tile::index_t nhead_stride_o       = hdim_v;
        const ck_tile::index_t nhead_stride_randval = shape_seqlen_q * max_seqlen_k;
        const ck_tile::index_t nhead_stride_do      = hdim_v;
        const ck_tile::index_t nhead_stride_lsed    = max_seqlen_q;
        const ck_tile::index_t nhead_stride_dbias   = max_seqlen_k;
        // setup batch_stride_* arguments
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
        const ck_tile::index_t split_stride_dq_acc  = shape_batch * nhead * shape_seqlen_q * hdim_q;

        return fmha_bwd_args{devPtrQ,
                             devPtrKTranspose,
                             devPtrVTranspose,
                             devPtrBias,
                             devPtrO,
                             devPtrSoftmaxStats,
                             devPtrdO,
                             devPtrD,
                             nullptr,
                             devPtrdQ,
                             is_mqa_gqa ? devPtrdKExpanded : devPtrdK,
                             is_mqa_gqa ? devPtrdVExpanded : devPtrdV,
                             devPtrdBias,
                             devPtrdQAcc, /* dq_acc_buf */
                             devPtrCuSeqlensQ,
                             devPtrCuSeqlensKV,
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
                             stride_q,  // stride_dq_acc
                             stride_q,  // stride_dq
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
                             nhead_stride_q,  // nhead_stride_dq_acc
                             nhead_stride_q,  // nhead_stride_dq
                             nhead_stride_k,  // nhead_stride_dk
                             nhead_stride_v,  // nhead_stride_dv
                             nhead_stride_dbias,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_o,
                             batch_stride_randval,
                             batch_stride_do,
                             batch_stride_lsed,
                             batch_stride_q,  // batch_stride_dq_acc
                             batch_stride_q,  // batch_stride_dq
                             batch_stride_dk,
                             batch_stride_dv,
                             batch_stride_dbias,
                             split_stride_dq_acc,
                             left,
                             right,
                             static_cast<ck_tile::index_t>(mask_type),
                             p_drop,
                             p_undrop,
                             {drop_seed, drop_offset}};
    }();

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
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }
}
