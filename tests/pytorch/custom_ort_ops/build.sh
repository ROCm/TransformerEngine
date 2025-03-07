# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -ex

: ${CUSTOM_ORT_OPS_PATH=$(dirname $(realpath $0))}
cd ${CUSTOM_ORT_OPS_PATH}

# Download ONNX Runtime source
git clone --depth=1 -b rel-1.19.2 --single-branch https://github.com/microsoft/onnxruntime.git || true

# Configure and build with CMake
mkdir -p build
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=. -DUSE_ROCM=$NVTE_USE_ROCM
cmake --build build --verbose
cmake --install build --verbose
