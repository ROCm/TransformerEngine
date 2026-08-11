# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Dtype-independent primitives shared by the FlyDSL GEMM kernels.

These helpers carry no dtype specialization, so the per-dtype utils modules
(``fp16_gemm_utils`` / ``fp8_gemm_utils``) re-export them instead of keeping
private copies. Anything that varies by element type (global-swizzle layouts,
G2S/S2R loaders, buffer-tensor makers) stays in the per-dtype modules.
"""

import flydsl.expr as fx
from flydsl.expr import arith, rocdl

from .exceptions import FlyDSLUnsupportedError


def require_block_tiling(m, n, k, *, block_m, block_n, block_k, label, min_k_tiles=4):
    """Validate the host-side M/N/K tiling contract shared by every GEMM core.

    ``label`` is the human-readable kernel identifier used in the error text
    (e.g. ``"FP16 GEMM"`` or ``"MXFP8 TN GEMM"``). Raises
    ``FlyDSLUnsupportedError`` so callers fall back to the default backend on
    unsupported shapes. Returns the number of K tiles.
    """
    if m % block_m != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL {label} requires M to be a multiple of {block_m}, got M={m}"
        )
    if n % block_n != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL {label} requires N to be a multiple of {block_n}, got N={n}"
        )
    if k % block_k != 0:
        raise FlyDSLUnsupportedError(
            f"FlyDSL {label} requires K to be a multiple of {block_k}, got K={k}"
        )
    num_k_tiles = k // block_k
    if num_k_tiles < min_k_tiles:
        raise FlyDSLUnsupportedError(
            f"FlyDSL {label} requires at least {min_k_tiles} K{block_k} tiles, "
            f"got K={k} ({num_k_tiles} tiles)"
        )
    return num_k_tiles


# FlyDSL packs flat operand byte views into an int32 launch signature, so any
# single operand of >= 2 GiB overflows the 'i' pack format at launch. Guard for
# it up front so oversized shapes fall back to the default backend instead of
# aborting with an unrecoverable struct.error.
_MAX_LAUNCH_BYTES = 2**31 - 1


def require_launch_size(label, *tensors):
    """Reject operands whose byte size exceeds the int32 launch-argument limit.

    ``tensors`` is an iterable of ``(name, tensor)`` pairs. Raises
    ``FlyDSLUnsupportedError`` (so ``general_gemm`` falls back to the C++
    backend) for the first operand at or above ``_MAX_LAUNCH_BYTES``; skips
    ``None`` tensors (e.g. an absent bias/aux).
    """
    for name, t in tensors:
        if t is None:
            continue
        nbytes = t.numel() * t.element_size()
        if nbytes > _MAX_LAUNCH_BYTES:
            raise FlyDSLUnsupportedError(
                f"FlyDSL {label} operand {name} is {nbytes} bytes, which exceeds "
                f"the int32 launch-argument limit of {_MAX_LAUNCH_BYTES}"
            )


def cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


ceildiv = cdiv


def divmod(a, b):  # pylint: disable=redefined-builtin
    """Integer divmod that works on DSL values (e.g. ``Int32``).

    The builtin ``divmod`` rejects DSL scalar types, so this uses the overloaded
    ``//`` / ``%`` operators to emit the corresponding ops.
    """
    return (a // b, a % b)


def min(a, b):  # pylint: disable=redefined-builtin
    return arith.select(a < b, a, b)


def encode_waitcnt(vmcnt=63, lgkmcnt=15):
    """Encode the CDNA4/gfx950 ``S_WAITCNT`` SIMM16 operand.

    ``rocdl.s_waitcnt`` accepts the raw 16-bit immediate operand of the
    32-bit ``S_WAITCNT`` ISA instruction. On CDNA4, that SIMM16 field is:

        SIMM16[3:0]   = vmcnt[3:0]
        SIMM16[6:4]   = expcnt[2:0]
        SIMM16[11:8]  = lgkmcnt[3:0]
        SIMM16[15:14] = vmcnt[5:4]

    ``vmcnt`` is therefore one six-bit counter split across two noncontiguous
    fields; bits [5:4] are placed in SIMM16[15:14], while bits [3:0] remain
    in SIMM16[3:0].

    A wait-counter field set to its maximum representable value is effectively
    unconstrained: the instruction does not wait on that counter. This helper
    always encodes ``expcnt=7`` and defaults to ``vmcnt=63`` and ``lgkmcnt=15``,
    so callers specify only the counters on which they intend to wait.

    For example, ``encode_waitcnt(lgkmcnt=0)`` returns ``0xC07F``, which the
    assembler renders as ``s_waitcnt lgkmcnt(0)``.
    See: https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
    """
    if not 0 <= vmcnt <= 63:
        raise ValueError(f"vmcnt must be in [0, 63], got {vmcnt}")
    if not 0 <= lgkmcnt <= 15:
        raise ValueError(f"lgkmcnt must be in [0, 15], got {lgkmcnt}")

    return (
        (7 << 4)  # expcnt=7 -> SIMM16[6:4] (unconstrained)
        | (vmcnt & 0x0F)  # vmcnt[3:0] -> SIMM16[3:0]
        | ((lgkmcnt & 0x0F) << 8)  # lgkmcnt[3:0] -> SIMM16[11:8]
        | ((vmcnt & 0x30) << 10)  # vmcnt[5:4] -> SIMM16[15:14]
    )


# Keep the documented gfx950 encoding invariant executable and import-time cheap.
assert encode_waitcnt(lgkmcnt=0) == 0xC07F


def barrier(vmcnt=63, lgkmcnt=15):
    if vmcnt != 63 or lgkmcnt != 15:
        rocdl.s_waitcnt(encode_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt))
    rocdl.s_barrier()


def xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x
    num_wg = num_pid_m * num_pid_n

    # Simple row-major path.
    simple_m, simple_n = divmod(wgid, num_pid_n)

    # XCD-remapped grouped-M path.
    intra_xcd, xcd = divmod(wgid, NUM_XCDS)
    wgid_remap = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = divmod(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg < SWIZZLE_THRESHOLD) | (num_wg % NUM_XCDS != 0)
    return (
        arith.select(use_simple, simple_m, pid_m),
        arith.select(use_simple, simple_n, pid_n),
    )


# XOR swizzle over a 128-wide tile: col ^= ((row//2) % 8) * 16, row unchanged.
# Equivalent to CuTe Swizzle<B=3, M=4, S=4> (vec=16, perPhase=2, maxPhase=8).
# Callers must keep col < 128 so the XOR stays within the row. Self-inverse, so
# the same call is used on both the LDS store (via compute_global_swizzle) and
# the MFMA-feeding LDS read (S2RLoader). ``col_in_bytes`` is a byte coordinate.
def swizzle_128(row, col_in_bytes):
    """HK 128-byte row XOR swizzle; ``col_in_bytes`` is a byte coordinate."""
    offset = row * 128 + col_in_bytes
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))
