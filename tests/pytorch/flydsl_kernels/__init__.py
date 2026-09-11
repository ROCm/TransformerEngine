# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# License for AMD contributions = MIT. See LICENSE for more information

"""FlyDSL GEMM test package.

Gives ``test_gemm.py`` a package-qualified module name so it does not collide
under pytest's default prepend import mode with the identically named
``tests/pytorch/triton_kernels/test_gemm.py``.
"""
