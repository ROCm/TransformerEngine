#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include "common_hip.h"
#include "extensions_hip.h"

namespace te_mxfp4 {
extern "C" void launch_cast_transpose_mxfp4_shuffled(
    const void* input,
    void* rowwise_fp4,
    void* rowwise_scale,
    void* colwise_fp4,
    void* colwise_scale,
    int M, int N,
    bool use_rowwise,
    bool use_colwise,
    bool shuffle_scales,
    bool use_hadamard,
    bool shuffle_rowwise_fp4,
    bool shuffle_colwise_fp4,
    int rowwise_scale_stride,
    int colwise_scale_stride,
    int rowwise_scale_N,
    int rowwise_scale_M_pad,
    int rowwise_scale_N_pad,
    int colwise_scale_M,
    int colwise_scale_N,
    int colwise_scale_M_pad,
    int colwise_scale_N_pad,
    hipStream_t stream
);
}

namespace transformer_engine::pytorch {

inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cast_transpose_mxfp4_fused_shuffle(
    at::Tensor input,
    std::optional<at::Tensor> rowwise_fp4_out,
    std::optional<at::Tensor> rowwise_scale_out,
    std::optional<at::Tensor> colwise_fp4_out,
    std::optional<at::Tensor> colwise_scale_out,
    bool shuffle_rowwise_scale,
    bool shuffle_colwise_scale,
    bool shuffle_rowwise_fp4,
    bool shuffle_colwise_fp4,
    bool use_hadamard
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "Input must be BFloat16");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    const int M = input.size(0);
    const int N = input.size(1);
    constexpr int MXFP4_BLOCK_SIZE = 32;

    constexpr int SHUFFLE_BN = 16;
    constexpr int SHUFFLE_BK = 32;

    TORCH_CHECK(N % MXFP4_BLOCK_SIZE == 0, "N must be divisible by 32");

    if (shuffle_rowwise_fp4) {
        TORCH_CHECK(M % SHUFFLE_BN == 0, "M must be divisible by 16 for shuffled rowwise FP4");
        TORCH_CHECK((N / 2) % SHUFFLE_BK == 0, "N/2 must be divisible by 32 for shuffled rowwise FP4");
    }
    if (shuffle_colwise_fp4) {
        TORCH_CHECK(N % SHUFFLE_BN == 0, "N must be divisible by 16 for shuffled colwise FP4");
        TORCH_CHECK((M / 2) % SHUFFLE_BK == 0, "M/2 must be divisible by 32 for shuffled colwise FP4");
    }

    auto device = input.device();
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    bool use_rowwise = rowwise_fp4_out.has_value() || shuffle_rowwise_scale || shuffle_rowwise_fp4;
    bool use_colwise = colwise_fp4_out.has_value() || shuffle_colwise_scale || shuffle_colwise_fp4;
    if (!use_rowwise && !use_colwise) use_rowwise = true;

    at::Tensor rowwise_fp4, rowwise_scale;
    int rowwise_scale_stride = 1;
    int rowwise_scale_N = cdiv(N, MXFP4_BLOCK_SIZE);
    int rowwise_scale_M_pad = cdiv(M, 256) * 256;
    int rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

    if (use_rowwise) {
        if (rowwise_fp4_out.has_value()) {
            rowwise_fp4 = rowwise_fp4_out.value();
        } else {
            rowwise_fp4 = at::empty({M, N / 2}, at::TensorOptions().dtype(at::kByte).device(device));
        }

        if (rowwise_scale_out.has_value()) {
            rowwise_scale = rowwise_scale_out.value();
            rowwise_scale_stride = rowwise_scale.stride(0);
            rowwise_scale_M_pad = rowwise_scale.size(0);
            rowwise_scale_N_pad = rowwise_scale.size(1);
        } else {
            if (shuffle_rowwise_scale) {
                int padded_M = cdiv(M, 256) * 256;
                rowwise_scale = at::empty({padded_M, rowwise_scale_N_pad},
                    at::TensorOptions().dtype(at::kByte).device(device));
            } else {
                rowwise_scale = at::empty({M, rowwise_scale_N},
                    at::TensorOptions().dtype(at::kByte).device(device));
            }
            rowwise_scale_stride = rowwise_scale.stride(0);
        }
    } else {
        rowwise_fp4 = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
        rowwise_scale = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    }

    at::Tensor colwise_fp4, colwise_scale;
    int colwise_scale_stride = 1;
    int colwise_scale_M = N;
    int colwise_scale_N = cdiv(M, MXFP4_BLOCK_SIZE);
    int colwise_scale_M_pad = cdiv(N, 256) * 256;
    int colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

    if (use_colwise) {
        if (colwise_fp4_out.has_value()) {
            colwise_fp4 = colwise_fp4_out.value();
        } else {
            colwise_fp4 = at::empty({N, M / 2}, at::TensorOptions().dtype(at::kByte).device(device));
        }

        if (colwise_scale_out.has_value()) {
            colwise_scale = colwise_scale_out.value();
            colwise_scale_stride = colwise_scale.stride(0);
            colwise_scale_M_pad = colwise_scale.size(0);
            colwise_scale_N_pad = colwise_scale.size(1);
        } else {
            if (shuffle_colwise_scale) {
                int padded_N = cdiv(N, 256) * 256;
                colwise_scale = at::empty({padded_N, colwise_scale_N_pad},
                    at::TensorOptions().dtype(at::kByte).device(device));
            } else {
                colwise_scale = at::empty({N, colwise_scale_N},
                    at::TensorOptions().dtype(at::kByte).device(device));
            }
            colwise_scale_stride = colwise_scale.stride(0);
        }
    } else {
        colwise_fp4 = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
        colwise_scale = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    }

    te_mxfp4::launch_cast_transpose_mxfp4_shuffled(
        input.data_ptr(),
        use_rowwise ? rowwise_fp4.data_ptr() : nullptr,
        use_rowwise ? rowwise_scale.data_ptr() : nullptr,
        use_colwise ? colwise_fp4.data_ptr() : nullptr,
        use_colwise ? colwise_scale.data_ptr() : nullptr,
        M, N,
        use_rowwise,
        use_colwise,
        shuffle_rowwise_scale || shuffle_colwise_scale,
        use_hadamard,
        shuffle_rowwise_fp4,
        shuffle_colwise_fp4,
        rowwise_scale_stride,
        colwise_scale_stride,
        rowwise_scale_N,
        rowwise_scale_M_pad,
        rowwise_scale_N_pad,
        colwise_scale_M,
        colwise_scale_N,
        colwise_scale_M_pad,
        colwise_scale_N_pad,
        stream
    );

    return std::make_tuple(rowwise_fp4, rowwise_scale, colwise_fp4, colwise_scale);
}

}
