/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file swizzle.h
 *  \brief Functions to convert scaling factors into format expected by GEMM.
 */

#ifndef TRANSFORMER_ENGINE_SWIZZLE_H_
#define TRANSFORMER_ENGINE_SWIZZLE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Swizzling scaling factors into the required interleaved layout for GEMM
 *
 *  \param[in]     input        Input tensor with non-swizzled scale_inv.
 *  \param[in,out] output       Output tensor which hosts swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scale_inv is stored in row-major.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantitized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_swizzle_scaling_factors(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Swizzling scaling factors into the required interleaved layout for GEMM
 *
 *  \param[in]     inputs       Input tensors with non-swizzled scale_inv.
 *  \param[in,out] outputs      Output tensors which hosts swizzled scale_inv.
 *  \param[in]     num_tensors  Number of input and output tensors.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scale_inv is stored in row-major.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantitized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_multi_tensor_swizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                               const size_t num_tensors, cudaStream_t stream);

/*! \brief Swizzling FP8 block scaling scaling factors into mxfp8 interleaved layout for GEMM
 *
 *  \param[in]     input        Input FP8 block-scaled tensor.
 *  \param[in,out] output       Output mxfp8 tensor which hosts swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  This function is used for emulating the FP8 block scaling recipe on Blackwell and newer as it
 *  not natively supported by cublasLt on architectures other than Hopper.

 *  Requirements:
 *  - input is an FP8 block scaling tensor
 *  - input has rowwise usage
 *  - output is an MXFP8 tensor
 *  - output has rowwise usage
 *  - output.scale_inv has appropriate shape
 *  */
void nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(const NVTETensor input, NVTETensor output,
                                                         cudaStream_t stream);

/*! \brief Swizzling scaling factors into the AITER e8m0_shuffle layout for GEMM
 *
 *  This produces the scale layout expected by hipBLASLt's
 *  HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT mode (gfx1250/MI450).
 *
 *  The layout matches AITER's e8m0_shuffle:
 *    scale = scale.view(M//32, 2, 16, N//8, 2, 4)
 *    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
 *    scale = scale.view(M, N)
 *
 *  \param[in]     input        Input tensor with non-swizzled scale_inv (MXFP8).
 *  \param[in,out] output       Output tensor which hosts swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - Input scaling mode is NVTE_MXFP8_1D_SCALING.
 *  - scale_inv M dimension is padded to a multiple of 32.
 *  - scale_inv K dimension is padded to a multiple of 8.
 */
void nvte_swizzle_scaling_factors_aiter(const NVTETensor input, NVTETensor output,
                                        cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_SWIZZLE_H_
