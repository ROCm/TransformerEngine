# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted by AMD from the FlyDSL project's GEMM utility helpers.
"""Dtype-independent primitives shared by the FlyDSL GEMM kernels.

These helpers carry no dtype specialization, so the per-dtype utils modules
(``fp16_gemm_utils`` / ``fp8_gemm_utils``) re-export them instead of keeping
private copies. Anything that varies by element type (global-swizzle layouts,
G2S/S2R loaders, buffer-tensor makers) stays in the per-dtype modules.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm, rocdl as _rocdl_ops
from flydsl._mlir.extras import types as _T
from flydsl.expr import arith, rocdl
from flydsl.expr.utils.arith import ArithValue
from flydsl.runtime.device import is_rdna_arch

from .exceptions import FlyDSLUnsupportedError


# ---------------------------------------------------------------------------
# Low-level AMD buffer primitives.
#
# The permute-free wgrad kernel needs a raw V# (``!llvm.ptr<8>``) buffer
# resource because its global->LDS DMA (``rocdl.raw_ptr_buffer_load_lds``) takes
# one as an operand -- something the tile/layout copy atoms cannot express. The
# forward/dgrad scalar route loads and the ``StoreCBf16`` epilogue moved to the
# copy API instead; only these irreducible helpers remain. Implemented directly
# on the ``rocdl``/``llvm`` dialect ops (formerly FlyDSL's ``expr.buffer_ops``,
# which upstream relocated out of the importable package).
# ---------------------------------------------------------------------------


def _get_buffer_flags(arch=None):
    """AMD buffer resource descriptor (V#) flags word (bits 127:96).

    CDNA (gfx9xx): ``(7 << 12) | (4 << 15)``; RDNA adds the reserved bit 24 and
    OOB_SELECT=2. Mirrors LLVM's ``AMDGPUToROCDL makeBufferRsrc()``.
    """
    import os

    if arch is None:
        arch = os.environ.get("FLYDSL_GPU_ARCH")
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


def _unwrap_value(value):
    """Recursively unwrap ArithValue / DSL Numeric wrappers to an ``ir.Value``."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


def _create_i32_constant(value: int) -> ir.Value:
    i32_type = _T.i32()
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    attr = ir.IntegerAttr.get(i32_type, value)
    return _unwrap_value(std_arith.ConstantOp(i32_type, attr).result)


def _create_i16_constant(value: int) -> ir.Value:
    i16_type = _T.i16()
    attr = ir.IntegerAttr.get(i16_type, value)
    return _unwrap_value(std_arith.ConstantOp(i16_type, attr).result)


def _create_i64_constant(value: int) -> ir.Value:
    i64_type = _T.i64()
    attr = ir.IntegerAttr.get(i64_type, value)
    return _unwrap_value(std_arith.ConstantOp(i64_type, attr).result)


def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
    """Extract the base address of a fly.memref / pointer as an index value.

    Used by the permute-free kernels to rebase a global operand in i64 (avoiding
    i32 element-offset overflow on large token pools) before ``make_view``.
    """
    from flydsl._mlir.dialects import fly as _fly
    from flydsl._mlir.dialects import memref as _memref

    raw = _unwrap_value(tensor)
    try:
        ir.MemRefType(raw.type)
        return _memref.extract_aligned_pointer_as_index(raw)
    except ValueError:
        pass

    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw)
    i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
    return _unwrap_value(std_arith.IndexCastOp(ir.IndexType.get(), i64_val).result)


def get_element_ptr(
    base_ptr,
    byte_offset=None,
    static_byte_offset: int = 0,
    elem_type=None,
    no_wrap_flags=None,
) -> ir.Value:
    """Build an LLVM GEP from a base pointer plus byte offsets (i8 element type)."""
    _gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _unwrap_value(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(f"static_byte_offset must be int, got {type(static_byte_offset).__name__}")
    if elem_type is None:
        elem_type = _T.i8()
    elif callable(elem_type):
        elem_type = elem_type()

    if byte_offset is None:
        dynamic_indices = []
        raw_constant_indices = [int(static_byte_offset)]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        raw_constant_indices = [int(byte_offset) + int(static_byte_offset)]
    else:
        offset_val = _unwrap_value(byte_offset)
        if isinstance(offset_val.type, ir.IndexType):
            offset_val = _unwrap_value(std_arith.IndexCastOp(_T.i64(), offset_val).result)
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError(
                "byte_offset must be int, index, or integer-typed MLIR value; "
                f"got {offset_val.type}"
            )

        if static_byte_offset != 0:
            static_type = offset_val.type
            static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
            static_const = _unwrap_value(std_arith.ConstantOp(static_type, static_attr).result)
            offset_val = _unwrap_value(std_arith.AddIOp(offset_val, static_const).result)

        dynamic_indices = [offset_val]
        raw_constant_indices = [_gep_dynamic_index_sentinel]

    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        dynamic_indices,
        raw_constant_indices,
        elem_type,
        no_wrap_flags,
    ).result


def make_buffer_rsrc_from_addr(addr_i64, *, num_records_bytes=None) -> ir.Value:
    """Create an AMD V# buffer resource (``!llvm.ptr<8>``) from a raw i64 address.

    Used for kernel-arg pointers with no fly.memref (e.g. the wgrad DMA operands
    and route-metadata tensors). ``num_records_bytes`` bounds the hardware OOB
    check; ``None`` uses the max size (no effective bounds).
    """
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = ir.Type.parse("!llvm.ptr")
    base_ptr = llvm.IntToPtrOp(ptr_type, addr_i64).result
    flags = _create_i32_constant(_get_buffer_flags())
    stride = _create_i16_constant(0)
    if num_records_bytes is None:
        num_records = _create_i64_constant(0xFFFFFFFF)
    elif isinstance(num_records_bytes, int):
        nbytes = int(num_records_bytes)
        if nbytes < 0:
            nbytes = 0
        if nbytes > 0xFFFFFFFF:
            nbytes = 0xFFFFFFFF
        num_records = _create_i64_constant(nbytes)
    else:
        num_records = _unwrap_value(num_records_bytes)
        i64_type = _T.i64()
        if not isinstance(num_records.type, ir.IntegerType) or num_records.type.width != 64:
            if isinstance(num_records.type, ir.IndexType):
                num_records = _unwrap_value(std_arith.IndexCastOp(i64_type, num_records).result)
            else:
                num_records = _unwrap_value(std_arith.ExtSIOp(i64_type, num_records).result)
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return _rocdl_ops.MakeBufferRsrcOp(rsrc_type, base_ptr, stride, num_records, flags).result


def raw_buffer_load(rsrc, offset, dtype):
    """Scalar buffer load of one ``dtype`` element at element ``offset`` (V# path)."""
    if hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    offset = _unwrap_value(offset)
    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        offset = _unwrap_value(std_arith.IndexCastOp(_T.i32(), offset).result)
    element_bytes = dtype.width // 8
    offset = _unwrap_value(std_arith.MulIOp(offset, _create_i32_constant(element_bytes)).result)
    soffset = _create_i32_constant(0)
    aux = _create_i32_constant(0)
    return _rocdl_ops.RawPtrBufferLoadOp(dtype, rsrc, offset, soffset, aux).result


def raw_buffer_load_i32(rsrc, offset):
    """Load one i32 scalar from a V# buffer resource at element ``offset``."""
    return raw_buffer_load(rsrc, offset, _T.i32())


def raw_buffer_store(data, rsrc, offset):
    """Scalar buffer store of ``data`` at element ``offset`` (V# path)."""
    if hasattr(data, "ir_value"):
        data = data.ir_value()
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = _unwrap_value(offset)
    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        offset = _unwrap_value(std_arith.IndexCastOp(_T.i32(), offset).result)
    element_bytes = data.type.width // 8
    offset = _unwrap_value(std_arith.MulIOp(offset, _create_i32_constant(element_bytes)).result)
    soffset = _create_i32_constant(0)
    aux = _create_i32_constant(0)
    _rocdl_ops.RawPtrBufferStoreOp(data, rsrc, offset, soffset, aux)


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


def _i64(v):
    # widen an i32 runtime value to i64 (avoids overflow in worst-case base offsets)
    return ArithValue(arith.extsi(fx.T.i64, _unwrap_value(v)), signed=True)


def make_value_attrs(waves_per_eu, agpr_alloc, fwg):
    """Kernel value_attrs. agpr_alloc: 0 = compiler default; N>0 = force exactly
    N AGPRs ("N,N"); -N = allow up to N ("0,N")."""
    d = {"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": fwg}
    if agpr_alloc != 0:
        if agpr_alloc < 0:
            alloc = f"0,{-agpr_alloc}"
        else:
            alloc = f"{agpr_alloc},{agpr_alloc}"
        d["passthrough"] = [
            ["amdgpu-agpr-alloc", alloc],
            ["amdgpu-mfma-vgpr-form", "false"],
        ]
    return d
