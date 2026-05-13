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

#ifdef __HIP_PLATFORM_AMD__
/*! \brief Swizzle MX (E8M0) scaling factors into gfx1250 Tensile 3D layout for GEMM
 *
 *  Tensile 3D layout: groups M into blocks of 4, then permutes {1, 0, 2}.
 *  For source (m, k): dst = (m/4) * (K_scale * 4) + k * 4 + (m % 4)
 *
 *  \param[in]     input        Input tensor with non-swizzled scale_inv (MXFP8).
 *  \param[in,out] output       Output tensor which hosts swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - Input scaling mode is NVTE_MXFP8_1D_SCALING.
 *  - scale_inv M dimension is padded to a multiple of 4.
 */
void nvte_swizzle_scaling_factors_mx(const NVTETensor input, NVTETensor output,
                                        cudaStream_t stream);
#endif  // __HIP_PLATFORM_AMD__

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_SWIZZLE_H_
