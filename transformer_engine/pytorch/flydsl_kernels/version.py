# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Supported FlyDSL version range, shared by every FlyDSL kernel backend."""

# Supported flydsl release series. Single source of truth for the version
# requirement, since flydsl is user-installed (not a declared dependency).
# flydsl is pre-1.0, so the minor version is the breaking-change axis: only
# the exact 0.3.x series is accepted, since a bump to 0.4.x is expected to
# change the API.
_MIN_FLYDSL = (0, 3)
_MAX_FLYDSL = (0, 4)  # exclusive upper bound


def _check_flydsl_version(backend: str) -> None:
    """Raise ImportError unless a supported flydsl is installed.

    A missing, too-old, or too-new package all raise ImportError, so callers can
    treat every failure mode alike. ``backend`` names the caller in the message.
    """
    from importlib.metadata import version
    from packaging.version import Version, InvalidVersion

    _min = Version(f"{_MIN_FLYDSL[0]}.{_MIN_FLYDSL[1]}")
    _max = Version(f"{_MAX_FLYDSL[0]}.{_MAX_FLYDSL[1]}")

    installed = version("flydsl")
    try:
        parsed = Version(installed)
    except InvalidVersion as e:
        # A valid install exposes a PEP 440 version string; an unparseable one
        # is a broken install, so block it rather than let it proceed.
        raise ImportError(f"flydsl version {installed!r} is not a valid PEP 440 version") from e
    if not _min <= parsed < _max:
        raise ImportError(
            f"flydsl {installed} is installed but the FlyDSL {backend} backend requires "
            f">= {_MIN_FLYDSL[0]}.{_MIN_FLYDSL[1]}, < {_MAX_FLYDSL[0]}.{_MAX_FLYDSL[1]}"
        )
