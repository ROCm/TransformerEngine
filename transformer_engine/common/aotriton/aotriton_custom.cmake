# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#Set AOTriton commit SHA1 to environment variable
#so AOTriton build system doesn't need to parse git repository
set(ENV{AOTRITON_CI_SUPPLIED_SHA1} ${TE_AOTRITON_COMMIT_SHA1})
