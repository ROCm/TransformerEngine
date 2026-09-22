/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

// Nothing here is CUDA, so hipify leaves the file alone and the include above survives verbatim;
// this is the marker that convention uses to say so. It stays commented out because the bit math
// is also compiled on the host, by mxfp8_direct_transpose_check.cpp.
// #include "hip/hip_runtime.h"

#include <cstdint>

// Re-encoding an MXFP8 element against a different E8M0 scale, by exponent arithmetic alone.
// This is the "direct transpose" of arXiv:2511.02302 (Algorithm 1, Eqs. 12-16) at MXFP8's block
// size of 32. The host-side derivation, the invariants and the loss characterisation live in
// mxfp8_direct_transpose.py, which every line below is transcribed from; keep the two in step.
//
// WHAT IT IS FOR
//     A row-wise MXFP8 tensor carries one scale per 32 elements along the last axis; the
//     column-wise view of the same tensor carries one scale per 32 elements along the first.
//     TE stores both with identical [rows, cols] data layout (MXFP8Quantizer::get_columnwise_shape
//     returns the row-wise shape verbatim), so converting between them moves no bytes: it only
//     re-groups the scales and re-encodes each element against its new group's scale.
//
// WHY THE RE-ENCODE IS FREE IN THE NORMAL RANGE
//     An element's real value is (the FP8 number) * (its group's scale). Re-grouping must not
//     change it, so when the scale grows by 2^k the stored number must shrink by 2^k -- and for a
//     normal FP8 number the exponent field absorbs that whole step. Sign and mantissa never move.
//     MXFP8 meets the paper's power-of-two-scale precondition for free: an E8M0 scale *is* 2^T.
//
// WHY OVERFLOW CANNOT HAPPEN
//     The target scale is a MAX over the source scales in the tile, so k >= 0 always and values
//     only ever move down. Underflow is the sole failure mode, and it is the price of that.
//
// THE TWO BRANCHES THE PAPER'S PSEUDOCODE OMITS
//     Eqs. 10-16 assume normal numbers. A subnormal has no exponent field to absorb the shift, so
//     the division comes out of the mantissa and rounds: (a) elements that arrive subnormal,
//     (b) elements whose shifted exponent falls below the normal floor. Both land in the else
//     branch of mx_rescale() below.

#if defined(__HIPCC__) || defined(__CUDACC__)
#define HK_MXT_FN __device__ __forceinline__
#else
#define HK_MXT_FN inline          // so the bit math is exercisable by a host test
#endif

namespace hk_mxfp8_transpose {

// An OCP FP8 element format, per arXiv:2209.05433 Table 1. CODE matches the kernels' CBSZ/BLGP
// operand-format codes and mxfp8_gemm.cpp's fp8_code(): 0 = E4M3, 1 = E5M2.
template <int CODE> struct Fmt;

// E4M3 spends a single bit pattern on NaN (S.1111.111) and has no infinities.
template <> struct Fmt<0> {
    static constexpr int  MAN_BITS = 3;
    static constexpr int  MAN_MASK = 0x7;
    static constexpr int  EXP_MASK = 0xF;
    static constexpr int  IMPLICIT = 8;      // significand of 1.0 == 2^MAN_BITS
    static constexpr bool HAS_INF  = false;
};

// E5M2 reserves the whole all-ones exponent for inf/NaN.
template <> struct Fmt<1> {
    static constexpr int  MAN_BITS = 2;
    static constexpr int  MAN_MASK = 0x3;
    static constexpr int  EXP_MASK = 0x1F;
    static constexpr int  IMPLICIT = 4;
    static constexpr bool HAS_INF  = true;
};

// sig / 2^count, rounded to nearest even. count <= 0 is the identity and count >= 8 is zero,
// because sig <= IMPLICIT + MAN_MASK == 15 in either format -- which is also what keeps the
// shifts below the width of the type when k is large.
HK_MXT_FN int rne_shift_right(int sig, int count) {
    if (count <= 0) {
        return sig;
    }
    if (count >= 8) {
        return 0;
    }
    const int trunc = sig >> count;
    const int rem   = sig - (trunc << count);
    const int half  = 1 << (count - 1);
    return trunc + ((rem > half) || (rem == half && (trunc & 1)));
}

// Re-encode one element against a scale that is 2^k times larger. k >= 0 is the caller's
// contract (the target is a max), and k == 0 is the identity. An exact per-column target would
// make k signed and need a renormalize-up branch here -- see the NEXT STEP note at step 2 in
// overlap_common.cuh::mx_colwise_band.
template <int CODE>
HK_MXT_FN uint8_t mx_rescale(uint8_t v, int k) {
    using F = Fmt<CODE>;
    const int E  = (v >> F::MAN_BITS) & F::EXP_MASK;
    const int Mn = v & F::MAN_MASK;

    // Reserved patterns encode no scaled value, so shifting one turns an Inf into a finite number
    // and walks an E5M2 NaN's exponent out of range. The reference script does not model this
    // because it manufactures its own inputs; a gathered tensor is not ours to trust.
    if constexpr (F::HAS_INF) {
        if (E == F::EXP_MASK) {
            return v;
        }
    } else {
        if (E == F::EXP_MASK && Mn == F::MAN_MASK) {
            return v;
        }
    }
    if (k == 0) {
        return v;
    }

    // Unified sig * 2^exp form: normal -> sig = IMPLICIT + M, exp = E - (bias + man_bits);
    // subnormal -> sig = M, exp = 1 - (bias + man_bits). Both rules then differ only in the
    // exponent-field value that stands in for E, which is what turns the rescale into one
    // subtraction instead of two cases.
    const bool normal = (E >= 1);
    const int  sig    = normal ? (F::IMPLICIT + Mn) : Mn;
    const int  new_E  = (normal ? E : 1) - k;

    int out_E, out_M;
    if (normal && new_E >= 1) {
        out_E = new_E;                          // Eqs. 12-16: exact, mantissa untouched
        out_M = Mn;
    } else {
        out_M = rne_shift_right(sig, 1 - new_E);   // into, or through, the subnormal range
        out_E = 0;
        if (out_M == F::IMPLICIT) {                // rounded up out of the subnormals
            out_E = 1;
            out_M = 0;
        }
    }
    return static_cast<uint8_t>((v & 0x80) | (out_E << F::MAN_BITS) | out_M);
}

// ---------------------------------------------------------------------------------------------
// The packed-byte fast path: four elements per 32-bit subtract.
//
// mx_rescale() above is ~10 scalar integer ops per byte, and the branch it almost always takes is
// the cheapest one: a normal element whose exponent stays normal (E >= 1 and E - k >= 1) keeps its
// sign and its mantissa and only has k taken off its exponent field. On the packed byte that whole
// operation is
//     v - (k << MAN_BITS)
// and the subtraction cannot borrow out of the exponent field -- E >= k + 1 says the field alone
// is big enough to absorb it -- so it never disturbs the sign bit, and four bytes can be done with
// one 32-bit subtract. What costs something is PROVING, cheaply, that all four bytes of a dword
// take that branch. The two SWAR tests below do that; any dword that fails one falls back to the
// scalar routine, so the two paths are exchangeable by construction and bit-identical by test
// (mxfp8_direct_transpose_check.cpp checks them against each other, and against the grid-search
// reference encoder, over every byte pattern and every mixed-branch dword it can build).
//
// EXTRACTING THE FOUR EXPONENT FIELDS
//     A 32-bit shift is not a per-byte shift: `x >> MAN_BITS` drags the low MAN_BITS bits of byte
//     i+1 down into the top of lane i. Lane i afterwards holds
//         bits [0, EXP_BITS)                   byte i's exponent field
//         bit  EXP_BITS == 7 - MAN_BITS        byte i's sign
//         bits [8 - MAN_BITS, 8)               byte i+1's mantissa
//     so the mask must stop below bit 8 - MAN_BITS. EXP_MASK does, with a bit to spare: an FP8
//     byte is 1 + EXP_BITS + MAN_BITS == 8 bits, so the sign sits between the exponent field and
//     the neighbour's bits in both formats.
//         E4M3: shift 3, lane = [ E:4 | S:1 | neighbour M:3 ], mask 0x0F -- clean.
//         E5M2: shift 2, lane = [ E:5 | S:1 | neighbour M:2 ], mask 0x1F -- clean.
//
// TESTING E >= k + 1 IN ALL FOUR LANES AT ONCE
//     Set a guard bit above each extracted field and subtract (k+1) from every lane: lane i
//     becomes 0x80 + E - (k+1), and its guard bit survives exactly when E >= k + 1. The lane value
//     stays in [0x80 - 31, 0x80 + 31] -- so no lane borrows from its neighbour and none carries
//     into it -- because 0 <= E <= EXP_MASK <= 31 and k is first clamped to EXP_MASK. That clamp
//     is free: E <= EXP_MASK, so k >= EXP_MASK admits no byte in the first place.
//
// EXCLUDING THE RESERVED PATTERNS
//     Reserved bytes encode no scaled value and mx_rescale() returns them untouched, but they pass
//     the E >= k + 1 test (their exponent field is the largest one), so they must be excluded
//     explicitly.
//         E5M2 reserves the whole all-ones exponent. e is already masked to [0, 31], so e + 1
//         reaches bit 5 in a lane iff that lane's E was 31, and cannot carry out of the lane.
//         E4M3 reserves the single slot S.1111.111, i.e. the only byte whose seven low bits are
//         all set. Masking the sign off leaves [0, 127] per lane, and +1 reaches bit 7 iff that
//         lane was 0x7F -- again with no carry out of the lane.
// ---------------------------------------------------------------------------------------------

// True (and `out` written) iff all four bytes of `x` re-encode by exponent subtraction alone, in
// which case `out` is byte-for-byte what mx_rescale<CODE>() would produce for each of them. False
// leaves `out` alone and means "use the scalar path"; it is a conservative answer only in the
// sense that it is never wrong, never in the sense that it is imprecise -- it is true exactly when
// every byte is a non-reserved normal whose exponent stays normal. k >= 0 is the caller's
// contract, as for mx_rescale(); k == 0 is admitted and is the identity.
template <int CODE>
HK_MXT_FN bool mx_rescale_dword_fast(uint32_t x, int k, uint32_t &out) {
    using F = Fmt<CODE>;
    constexpr uint32_t ONES  = 0x01010101u;                               // one per byte lane
    constexpr uint32_t GUARD = 0x80808080u;                               // the lanes' guard bits
    constexpr uint32_t EXPS  = static_cast<uint32_t>(F::EXP_MASK) * ONES;
    constexpr uint32_t OVER  = static_cast<uint32_t>(F::EXP_MASK + 1) * ONES;  // just above a field

    // k is a difference of two E8M0 bytes, so it can be anything up to 255, and the lane
    // arithmetic below only stays inside its byte while k <= EXP_MASK. Clamping is both the cheap
    // way to get that and a free one: E <= EXP_MASK, so no byte can satisfy E >= k + 1 once
    // k >= EXP_MASK, and pinning k there changes no answer. Unsigned, so a negative k -- which is
    // outside mx_rescale()'s contract too -- clamps the same way instead of running off. A clamp
    // rather than an early return so that the two constants derived from it stay in straight-line
    // code, where the compiler hoists them out of the four calls a 16-byte row makes.
    const uint32_t kc   = (static_cast<uint32_t>(k) < static_cast<uint32_t>(F::EXP_MASK))
                              ? static_cast<uint32_t>(k)
                              : static_cast<uint32_t>(F::EXP_MASK);
    const uint32_t krep = kc * ONES;                    // k in every lane; lanes hold k <= 31

    const uint32_t e = (x >> F::MAN_BITS) & EXPS;
    if ((((e | GUARD) - (krep + ONES)) & GUARD) != GUARD) {
        return false;                                                     // some lane leaves normal
    }
    if constexpr (F::HAS_INF) {
        if (((e + ONES) & OVER) != 0u) {
            return false;                                                 // some lane is inf/NaN
        }
    } else {
        if ((((x & ~GUARD) + ONES) & GUARD) != 0u) {
            return false;                                                 // some lane is the NaN slot
        }
    }

    // krep's lanes hold k <= EXP_MASK, so shifting the whole word left by MAN_BITS shifts
    // each lane's k into place without any of them spilling into the lane above.
    out = x - (krep << F::MAN_BITS);
    return true;
}

// Re-encode four packed elements against a scale 2^k times larger: the fast path when it applies,
// the scalar routine byte by byte when it does not. Bit-identical to four mx_rescale<CODE>() calls
// for every input.
template <int CODE>
HK_MXT_FN uint32_t mx_rescale_dword(uint32_t x, int k) {
    uint32_t out;
    if (mx_rescale_dword_fast<CODE>(x, k, out)) { // vectorized path for normals
        return out;
    }
    out = 0;                    // trip count 4: unrolled by every compiler that sees this
    for (int j = 0; j < 4; j++) {
        out |= static_cast<uint32_t>(mx_rescale<CODE>(static_cast<uint8_t>(x >> (8 * j)), k))
               << (8 * j);
    }
    return out;
}

}  // namespace hk_mxfp8_transpose
