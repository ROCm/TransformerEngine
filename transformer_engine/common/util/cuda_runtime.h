/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_

#ifndef USE_ROCM
#include <cuda_runtime_api.h>
#else
#include <hip/hip_runtime.h>
#endif
#include <string>

namespace transformer_engine {

namespace cuda {

/* \brief Number of accessible devices */
int num_devices();

/* \brief Which device is currently being used */
int current_device();

/* \brief Compute capability of device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Compute capability as int. Last digit is minor revision,
 *         remaining digits are major revision.
 */
int sm_arch(int device_id = -1);

#ifdef USE_ROCM
/* \brief Compute capability of device
 *
 * \param[in] device_id HIP device (default is current device)
 *
 * \return GPU arch name and compute capabilities string.
 */
const std::string &sm_arch_name(int device_id = -1);
#endif

/* \brief Number of multiprocessors on a device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Number of multiprocessors
 */
int sm_count(int device_id = -1);

#ifndef USE_ROCM
/* \brief Path to CUDA Toolkit headers
 *
 * The path can be configured by setting NVTE_CUDA_INCLUDE_DIR in the
 * environment. Otherwise searches in common install paths.
 *
 * \param[in] required Whether to throw exception if not found
 *
 * \return Path to include directory, or an empty string if not found
 */
const std::string &include_directory(bool required = false);
#endif

}  // namespace cuda

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
