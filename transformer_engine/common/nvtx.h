/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NVTX_H_
#define TRANSFORMER_ENGINE_COMMON_NVTX_H_

#ifndef __HIP_PLATFORM_AMD__
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
#include <string>

namespace transformer_engine::nvtx {

struct NVTXWrapper {
  explicit NVTXWrapper(const std::string &name) { nvtxRangePush(name.c_str()); }

  ~NVTXWrapper() { nvtxRangePop(); }
};

}  // namespace transformer_engine::nvtx

#endif  // TRANSFORMER_ENGINE_COMMON_NVTX_H_
