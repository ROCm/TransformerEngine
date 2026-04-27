#!/bin/bash

export NVTE_FRAMEWORK=pytorch
export NVTE_ROCM_ARCH=gfx950
export NVTE_USE_ROCM=1
export CU_NUM=256

pip install -U ninja psutil pybind11

export NVTE_AITER_PREBUILT_BASE_URL=https://compute-artifactory.amd.com:5000/artifactory/rocm-generic-local/te-ci/aiter-prebuilts

pip install -ve . --no-build-isolation
 
