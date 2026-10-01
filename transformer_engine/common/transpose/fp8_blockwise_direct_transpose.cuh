/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
#pragma once

// Nothing here is CUDA, so hipify leaves the file alone and the include above survives verbatim;
// this is the marker that convention uses to say so. It stays commented out because the bit math
// is also compiled on the host, by fp8_blockwise_direct_transpose_check.cpp.
// #include "hip/hip_runtime.h"

#include <cstdint>

// Re-encoding a 1x128 blockwise FP8 element against a different power-of-two scale, by exponent
// arithmetic alone. This is the "direct transpose" of arXiv:2511.02302 (Algorithm 1, Eqs. 12-16)
// at the 1D blockwise recipe's block size of 128, and the blockwise twin of
// gemm/kittens/cdna4/overlap/mxfp8_direct_transpose.cuh; keep the two in step.
//
// WHAT IT IS FOR
//     A row-wise 1x128 tensor [M, K] carries one FP32 scale_inv per 128 elements along K; the
//     column-wise tensor carries one per 128 elements along M, and TE stores its payload
//     transposed ([K, M]). Producing the column-wise operand from the row-wise one normally means
//     dequantize -> column amax -> requantize, which rounds every element a second time. With
//     power-of-two scales (force_pow_2_scales, the DeepSeek-style recipe) it does not have to:
//     give every column of a 128x128 tile the MAX of the tile's 128 row scales, and each element
//     of row r only needs re-encoding against a scale 2^k_r larger, k_r = log2(s_tile / s_row[r]).
//
// WHY THE RE-ENCODE IS FREE IN THE NORMAL RANGE
//     An element's real value is (the FP8 number) * (its scale). When the scale grows by 2^k the
//     stored number must shrink by 2^k -- and for a normal FP8 number the exponent field absorbs
//     that whole step. Sign and mantissa never move.
//
// WHY OVERFLOW CANNOT HAPPEN
//     The target scale is a MAX over the source scales in the tile, so k >= 0 always and values
//     only ever move down. Underflow is the sole failure mode, and it is the price of that.
//
// THE TWO BRANCHES THE PAPER'S PSEUDOCODE OMITS
//     Eqs. 10-16 assume normal numbers. A subnormal has no exponent field to absorb the shift, so
//     the division comes out of the mantissa and rounds: (a) elements that arrive subnormal,
//     (b) elements whose shifted exponent falls below the normal floor. Both land in the else
//     branch of fp8_rescale() below.
//
// WHAT DIFFERS FROM MXFP8
//     The scales are FP32 powers of two rather than E8M0 bytes, so k is the difference of the two
//     scales' FP32 exponent fields (pow2_scale_shift). And gfx942 stores FP8 as FNUZ: no
//     infinities, no negative zero, and 0x80 -- the slot negative zero would take -- is the only
//     NaN. Exponent arithmetic does not care about the bias, so FNUZ changes only the reserved
//     pattern and one rounding outcome: a negative element that underflows to zero must come out
//     as 0x00, because 0x80 would turn it into a NaN.

#if defined(__HIPCC__) || defined(__CUDACC__)
#define TE_FP8_DT_FN __host__ __device__ __forceinline__
#else
#define TE_FP8_DT_FN inline       // so the bit math is exercisable by a host test
#endif

namespace transformer_engine {
namespace fp8_blockwise_direct_transpose {

// An FP8 element format. CODE: 0 = E4M3, 1 = E5M2. FNUZ selects the gfx942 encodings (bias one
// larger, no infinities, no negative zero, NaN = 0x80) over the OCP ones of arXiv:2209.05433.
template <int CODE, bool FNUZ>
struct Fmt;

// OCP E4M3 spends a single bit pattern on NaN (S.1111.111) and has no infinities.
template <>
struct Fmt<0, false> {
  static constexpr int MAN_BITS = 3;
  static constexpr int MAN_MASK = 0x7;
  static constexpr int EXP_MASK = 0xF;
  static constexpr int IMPLICIT = 8;  // significand of 1.0 == 2^MAN_BITS
  static constexpr bool FNUZ = false;
  static constexpr bool HAS_INF = false;
};

// OCP E5M2 reserves the whole all-ones exponent for inf/NaN.
template <>
struct Fmt<1, false> {
  static constexpr int MAN_BITS = 2;
  static constexpr int MAN_MASK = 0x3;
  static constexpr int EXP_MASK = 0x1F;
  static constexpr int IMPLICIT = 4;
  static constexpr bool FNUZ = false;
  static constexpr bool HAS_INF = true;
};

// FNUZ formats: every exponent value is a number; the only reserved byte is 0x80.
template <>
struct Fmt<0, true> {
  static constexpr int MAN_BITS = 3;
  static constexpr int MAN_MASK = 0x7;
  static constexpr int EXP_MASK = 0xF;
  static constexpr int IMPLICIT = 8;
  static constexpr bool FNUZ = true;
  static constexpr bool HAS_INF = false;
};

template <>
struct Fmt<1, true> {
  static constexpr int MAN_BITS = 2;
  static constexpr int MAN_MASK = 0x3;
  static constexpr int EXP_MASK = 0x1F;
  static constexpr int IMPLICIT = 4;
  static constexpr bool FNUZ = true;
  static constexpr bool HAS_INF = false;
};

// k for re-encoding against `target` when the element was encoded against `source`, both
// positive power-of-two FP32 scales given as bit patterns: the difference of their exponent
// fields. target >= source (it is a max) makes this >= 0.
TE_FP8_DT_FN int pow2_scale_shift(uint32_t target_bits, uint32_t source_bits) {
  return static_cast<int>((target_bits >> 23) & 0xFF) - static_cast<int>((source_bits >> 23) & 0xFF);
}

// sig / 2^count, rounded to nearest even. count <= 0 is the identity and count >= 8 is zero,
// because sig <= IMPLICIT + MAN_MASK == 15 in either format -- which is also what keeps the
// shifts below the width of the type when k is large.
TE_FP8_DT_FN int rne_shift_right(int sig, int count) {
  if (count <= 0) {
    return sig;
  }
  if (count >= 8) {
    return 0;
  }
  const int trunc = sig >> count;
  const int rem = sig - (trunc << count);
  const int half = 1 << (count - 1);
  return trunc + ((rem > half) || (rem == half && (trunc & 1)));
}

// Re-encode one element against a scale that is 2^k times larger. k >= 0 is the caller's
// contract (the target is a max), and k == 0 is the identity.
template <typename F>
TE_FP8_DT_FN uint8_t fp8_rescale(uint8_t v, int k) {
  const int E = (v >> F::MAN_BITS) & F::EXP_MASK;
  const int Mn = v & F::MAN_MASK;

  // Reserved patterns encode no scaled value, so shifting one would turn an Inf into a finite
  // number or walk a NaN out of its slot. A received tensor is not ours to trust.
  if constexpr (F::FNUZ) {
    if (v == 0x80) {
      return v;
    }
  } else if constexpr (F::HAS_INF) {
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
  const int sig = normal ? (F::IMPLICIT + Mn) : Mn;
  const int new_E = (normal ? E : 1) - k;

  int out_E, out_M;
  if (normal && new_E >= 1) {
    out_E = new_E;  // Eqs. 12-16: exact, mantissa untouched
    out_M = Mn;
  } else {
    out_M = rne_shift_right(sig, 1 - new_E);  // into, or through, the subnormal range
    out_E = 0;
    if (out_M == F::IMPLICIT) {  // rounded up out of the subnormals
      out_E = 1;
      out_M = 0;
    }
  }
  if constexpr (F::FNUZ) {
    if (out_E == 0 && out_M == 0) {
      return 0;  // FNUZ has one zero; the sign bit on it would spell NaN
    }
  }
  return static_cast<uint8_t>((v & 0x80) | (out_E << F::MAN_BITS) | out_M);
}

// ---------------------------------------------------------------------------------------------
// The packed-byte fast path: four elements per 32-bit subtract.
//
// The 1D blockwise k is per ROW (one row scale per element of a tile row), so the four bytes of
// a dword loaded along K always share it -- the same situation as the MXFP8 header, and the same
// trick applies. fp8_rescale() above is ~10 scalar integer ops per byte, and the branch it almost
// always takes is the cheapest one: a normal element whose exponent stays normal (E >= 1 and
// E - k >= 1) keeps its sign and its mantissa and only has k taken off its exponent field. On the
// packed byte that whole operation is
//     v - (k << MAN_BITS)
// and the subtraction cannot borrow out of the exponent field -- E >= k + 1 says the field alone
// is big enough to absorb it -- so it never disturbs the sign bit, and four bytes can be done with
// one 32-bit subtract. What costs something is PROVING, cheaply, that all four bytes of a dword
// take that branch. The two SWAR tests below do that; any dword that fails one falls back to the
// scalar routine, so the two paths are exchangeable by construction and bit-identical by test
// (fp8_blockwise_direct_transpose_check.cpp checks them against each other, and against a
// reference encoder that searches every code, over every byte pattern and shift).
//
// EXTRACTING THE FOUR EXPONENT FIELDS
//     A 32-bit shift is not a per-byte shift: `x >> MAN_BITS` drags the low MAN_BITS bits of byte
//     i+1 down into the top of lane i. Lane i afterwards holds
//         bits [0, EXP_BITS)                   byte i's exponent field
//         bit  EXP_BITS == 7 - MAN_BITS        byte i's sign
//         bits [8 - MAN_BITS, 8)               byte i+1's mantissa
//     so the mask must stop below bit 8 - MAN_BITS. EXP_MASK does, with a bit to spare.
//
// TESTING E >= k + 1 IN ALL FOUR LANES AT ONCE
//     Set a guard bit above each extracted field and subtract (k+1) from every lane: lane i
//     becomes 0x80 + E - (k+1), and its guard bit survives exactly when E >= k + 1. The lane value
//     stays in [0x80 - 32, 0x80 + 31] -- so no lane borrows from its neighbour and none carries
//     into it -- because 0 <= E <= EXP_MASK <= 31 and k is first clamped to EXP_MASK. That clamp
//     is free: E <= EXP_MASK, so k >= EXP_MASK admits no byte in the first place.
//
// EXCLUDING THE RESERVED PATTERNS
//     OCP reserved bytes pass the E >= k + 1 test (their exponent field is the largest one), so
//     they must be excluded explicitly, exactly as in the MXFP8 header. FNUZ needs no test: its
//     one reserved byte, 0x80, has a zero exponent field and never passes E >= k + 1.
// ---------------------------------------------------------------------------------------------

// True (and `out` written) iff all four bytes of `x` re-encode by exponent subtraction alone, in
// which case `out` is byte-for-byte what fp8_rescale<F>() would produce for each of them. False
// leaves `out` alone and means "use the scalar path". k >= 0 is the caller's contract.
template <typename F>
TE_FP8_DT_FN bool fp8_rescale_dword_fast(uint32_t x, int k, uint32_t &out) {
  constexpr uint32_t ONES = 0x01010101u;   // one per byte lane
  constexpr uint32_t GUARD = 0x80808080u;  // the lanes' guard bits
  constexpr uint32_t EXPS = static_cast<uint32_t>(F::EXP_MASK) * ONES;
  constexpr uint32_t OVER = static_cast<uint32_t>(F::EXP_MASK + 1) * ONES;  // just above a field

  // k is a difference of two FP32 exponent fields, so it can be anything up to 254, and the lane
  // arithmetic below only stays inside its byte while k <= EXP_MASK. Clamping pins it there
  // without changing any answer (no byte can satisfy E >= k + 1 once k >= EXP_MASK), and keeps
  // the derived constants in straight-line code. Unsigned, so a negative k clamps the same way.
  const uint32_t kc = (static_cast<uint32_t>(k) < static_cast<uint32_t>(F::EXP_MASK))
                          ? static_cast<uint32_t>(k)
                          : static_cast<uint32_t>(F::EXP_MASK);
  const uint32_t krep = kc * ONES;  // k in every lane; lanes hold k <= 31

  const uint32_t e = (x >> F::MAN_BITS) & EXPS;
  if ((((e | GUARD) - (krep + ONES)) & GUARD) != GUARD) {
    return false;  // some lane leaves normal
  }
  if constexpr (!F::FNUZ) {
    if constexpr (F::HAS_INF) {
      if (((e + ONES) & OVER) != 0u) {
        return false;  // some lane is inf/NaN
      }
    } else {
      if ((((x & ~GUARD) + ONES) & GUARD) != 0u) {
        return false;  // some lane is the NaN slot
      }
    }
  }

  // krep's lanes hold k <= EXP_MASK, so shifting the whole word left by MAN_BITS shifts
  // each lane's k into place without any of them spilling into the lane above.
  out = x - (krep << F::MAN_BITS);
  return true;
}

// Re-encode four packed elements against a scale 2^k times larger: the fast path when it applies,
// the scalar routine byte by byte when it does not. Bit-identical to four fp8_rescale<F>() calls
// for every input.
template <typename F>
TE_FP8_DT_FN uint32_t fp8_rescale_dword(uint32_t x, int k) {
  uint32_t out;
  if (fp8_rescale_dword_fast<F>(x, k, out)) {  // vectorized path for normals
    return out;
  }
  out = 0;  // trip count 4: unrolled by every compiler that sees this
  for (int j = 0; j < 4; j++) {
    out |= static_cast<uint32_t>(fp8_rescale<F>(static_cast<uint8_t>(x >> (8 * j)), k))
           << (8 * j);
  }
  return out;
}

}  // namespace fp8_blockwise_direct_transpose
}  // namespace transformer_engine

#undef TE_FP8_DT_FN
