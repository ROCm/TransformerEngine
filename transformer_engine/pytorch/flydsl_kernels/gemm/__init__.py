# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""FlyDSL GEMM kernels (dense, non-grouped) for BF16/FP16/FP32/FP8/MXFP8."""

# Supported flydsl release series. Single source of truth for the version
# requirement, since flydsl is user-installed (not a declared dependency). A
# missing, too-old, or too-new package raises ImportError here, which
# general_gemm catches to warn once and fall back to the default backend.
# flydsl is pre-1.0, so the minor version is the breaking-change axis: only
# the exact 0.3.x series is accepted, since a bump to 0.4.x is expected to
# change the API.
_MIN_FLYDSL = (0, 3)
_MAX_FLYDSL = (0, 4)  # exclusive upper bound


def _check_flydsl_version() -> None:
    from importlib.metadata import version

    # PackageNotFoundError subclasses ImportError, so a missing package is
    # handled by the same fallback path as an unsupported one.
    installed = version("flydsl")
    try:
        major_minor = tuple(int(part) for part in installed.split(".")[:2])
    except ValueError:
        # Unparseable version string: the package imported, so let it proceed
        # rather than falsely block a valid install.
        return
    if not _MIN_FLYDSL <= major_minor < _MAX_FLYDSL:
        raise ImportError(
            f"flydsl {installed} is installed but the FlyDSL GEMM backend requires "
            f">= {_MIN_FLYDSL[0]}.{_MIN_FLYDSL[1]}, < {_MAX_FLYDSL[0]}.{_MAX_FLYDSL[1]}"
        )


_check_flydsl_version()

from .exceptions import FlyDSLUnsupportedError
from .gemm_wrappers import te_generic_gemm_flydsl

__all__ = [
    "FlyDSLUnsupportedError",
    "te_generic_gemm_flydsl",
]
