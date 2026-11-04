/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 * Portions adapted from HipKittens, Copyright (c) 2024 HazyResearch, MIT License
 * (see 3rdparty/hipkittens/LICENSE).
*************************************************************************/

// Grouped NT (wgrad): D_e [N][K] = alpha dY_e^T X_e (+ beta D_e) over the m_e tokens, X_e [m_e][K], dY_e [m_e][N];
// BGRAD: also dbias_e [N] = column sums of dY_e.

#include "hip/hip_runtime.h"
#include "grouped16_common.cuh"

namespace te_kittens {
namespace cdna4 {
namespace grouped16 {

constexpr int NT_REG_M = BLOCK_ROW / WARPS_ROW;
constexpr int NT_REG_N = BLOCK_COL / WARPS_COL;

// LDS-DMA of k-step kt (clamped to the last real one) of the 128 columns at col0 + 128 j of a [rows][lead] operand.
struct NtLoader {
    const uint32_t (&sw)[2];
    const i32x4 &srd;
    uint32_t lead_bytes;
    int k_last, col0;
    const uint32_t (&lds)[2][2];

    __device__ __forceinline__ uint32_t soff(int j, int kt) const {
        return (uint32_t)min(kt, k_last) * (uint32_t)K_STEP * lead_bytes + (uint32_t)((col0 + j * HALF_COL) * 2);
    }
    __device__ __forceinline__ void load(int p, int j, int kt) const {
        dma2(sw, srd, soff(j, kt), lds[p][j]);
    }
    __device__ __forceinline__ void load_h(int p, int j, int kt, int h) const {
        if (h) {
            dma_h<1>(sw, srd, soff(j, kt), lds[p][j]);
        } else {
            dma_h<0>(sw, srd, soff(j, kt), lds[p][j]);
        }
    }
};

/* dbias of the 256 columns from 256 cb (those below ld): thread (g, c) sums 8 columns c of rows g, g + 16, ...; the
 * 16 partial rows are added in a fixed order through LDS. */
template <typename T, typename O>
__device__ __forceinline__ void bgrad_unit(const T *dy, int rows, int ld, int cb, O *out, unsigned char *lds) {
    typedef __attribute__((address_space(1))) const u32x4_t gu4;
    const int tid = fs_tid();
    const int c   = tid & 31, g = tid >> 5;
    const __attribute__((address_space(1))) T *p = (const __attribute__((address_space(1))) T *)fs_uniform_ptr(dy) +
                                                  cb * BLOCK_COL + c * 8;
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int rows_c = cb * BLOCK_COL + c * 8 < ld ? rows : 0;
#pragma unroll 4
    for (int r = g; r < rows_c; r += 16) {
        const u32x4_t w = *(gu4 *)(p + (size_t)r * ld);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            s[2 * i] += lo16<typename Elem<T>::raw>(w[i]);
            s[2 * i + 1] += hi16<typename Elem<T>::raw>(w[i]);
        }
    }
    auto *sl = (__attribute__((address_space(3))) float *)(lds_u8 *)lds;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; i++) {
        sl[g * BLOCK_COL + c * 8 + i] = s[i];
    }
    __syncthreads();
    if (tid < BLOCK_COL && cb * BLOCK_COL + tid < ld) {
        float v = 0.f;
#pragma unroll
        for (int q = 0; q < 16; q++) {
            v += sl[q * BLOCK_COL + tid];
        }
        __attribute__((address_space(1))) O *o = (__attribute__((address_space(1))) O *)fs_uniform_ptr(out);
        if constexpr (std::is_same_v<O, float>) {
            o[cb * BLOCK_COL + tid] = v;
        } else {
            o[cb * BLOCK_COL + tid] = (O)v;
        }
    }
}

template <typename T, typename O, bool BGRAD, bool ACC>
__global__ __launch_bounds__(NUM_THREADS, 2)
void nt_kernel(const Table t, int ne, int Nw, int Kw, float alpha, float beta, int *ctr) {
    using ST_AB = st<T, K_STEP, HALF_COL, st_16x32_s>;
    using RT_A  = rt<T, K_STEP, NT_REG_M / 2, col_l, rt_32x16_s>;
    using RT_B  = rt<T, K_STEP, NT_REG_N / 2, col_l, rt_32x16_s>;
    using RT_C  = rt_fl<NT_REG_M / 2, NT_REG_N / 2, col_l, rt_16x16_s>;
    using R     = typename Elem<T>::raw;
    __shared__ alignas(16) unsigned char s_tiles[LDS_TILES];
    __shared__ int s_unit;
    static_assert(8 * sizeof(ST_AB) == LDS_TILES, "NT tiles must fill the region");
    ST_AB(&As)[2][2] = *reinterpret_cast<ST_AB(*)[2][2]>(s_tiles);
    ST_AB(&Bs)[2][2] = *reinterpret_cast<ST_AB(*)[2][2]>(s_tiles + 4 * sizeof(ST_AB));

    const int tiles   = t.unit_off[ne];
    const int nb      = (Nw + BLOCK_ROW - 1) / BLOCK_ROW;
    const int units   = tiles + (BGRAD ? ne * nb : 0);
    const int w_tiles = (Kw + BLOCK_COL - 1) / BLOCK_COL;
#pragma unroll 1
    for (;;) {
        const int u = claim(ctr, units, &s_unit);
        if (u >= units) {
            break;
        }
        if (BGRAD && u >= tiles) {
            const int e = (u - tiles) / nb;
            bgrad_unit<T>((const T *)t.b[e], t.rows[e], Nw, u - tiles - e * nb, (O *)t.bias[e], s_tiles);
            continue;
        }
        const int e    = grouped_sched::unit_expert<MAX_EXPERTS>(t.unit_off, ne, u);
        const int rows = t.rows[e];
        int mblk, nblk;
        grouped_sched::unit_tile(u - t.unit_off[e], w_tiles, nb, mblk, nblk);
        const int k_real  = (rows + K_STEP - 1) / K_STEP;
        // The pipeline retires k-steps in pairs; for an odd k_real the extra step reloads k-step k_real - 1 (never
        // read) and tail 2 skips its reads and MMAs but keeps its barriers.
        const int k_tiles = k_real + (k_real & 1);
        const uint32_t a_lead = (uint32_t)Nw * sizeof(T), b_lead = (uint32_t)Kw * sizeof(T);
        const R *a_base = fs_uniform_ptr((const R *)t.b[e]);
        const R *b_base = fs_uniform_ptr((const R *)t.a[e]);
        const i32x4 a_srd = fs_srd(a_base, (uint32_t)rows * a_lead);
        const i32x4 b_srd = fs_srd(b_base, (uint32_t)rows * b_lead);

        int tid_o = (int)threadIdx.x;
        asm volatile("" : "+v"(tid_o));
        const int wv_o   = __builtin_amdgcn_readfirstlane(tid_o >> 6);
        const int warp_m = wv_o / WARPS_COL;
        const int warp_n = wv_o % WARPS_COL;
        const int wid    = wv_o % NUM_WARPS;
        const uintptr_t lds_off = wid * 16 * kittens::WARP_THREADS;
        uint32_t a_lds[2][2], b_lds[2][2];
#pragma unroll
        for (int p_ = 0; p_ < 2; p_++) {
#pragma unroll
            for (int j_ = 0; j_ < 2; j_++) {
                a_lds[p_][j_] = __builtin_amdgcn_readfirstlane(
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[p_][j_].data[0]) + lds_off));
                b_lds[p_][j_] = __builtin_amdgcn_readfirstlane(
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[p_][j_].data[0]) + lds_off));
            }
        }
        uint32_t sw_A[2], sw_B[2];
        int lane_t = (int)threadIdx.x & 63;
        asm volatile("" : "+v"(lane_t));
        tile_lane_offsets<ST_AB>(lane_t, wv_o, (int)a_lead, sw_A);
        tile_lane_offsets<ST_AB>(lane_t, wv_o, (int)b_lead, sw_B);
        const NtLoader ldA{sw_A, a_srd, a_lead, k_real - 1, nblk * BLOCK_ROW, a_lds};
        const NtLoader ldB{sw_B, b_srd, b_lead, k_real - 1, mblk * BLOCK_COL, b_lds};
#define NT_READ_A(TL, P, J) \
    nt_load_frag<((P) * 2 + (J)) * (int)sizeof(ST_AB)>(TL, As[0][0], warp_m * (NT_REG_M / 2), lane_t)
#define NT_READ_B(TL, P, J) \
    nt_load_frag<((P) * 2 + (J)) * (int)sizeof(ST_AB)>(TL, Bs[0][0], warp_n * (NT_REG_N / 2), lane_t)

        RT_C c00, c01, c10, c11;
        RT_A A_tile;
        RT_B B_tile_0, B_tile_1;
        const int lane0 = lane_t;

        ldB.load(0, 0, 0);
        ldA.load(0, 0, 0);
        ldB.load(0, 1, 0);
        ldA.load(0, 1, 0);
        if (warp_m == 1) {
            __builtin_amdgcn_s_barrier();
        }
#include "grouped16_nt_kloop.inc"
#undef NT_READ_A
#undef NT_READ_B
        const int rf0 = __builtin_amdgcn_readfirstlane(warp_m);
        const int cf0 = __builtin_amdgcn_readfirstlane(warp_n);
        int lane_epi = lane0;
        asm volatile("" : "+v"(lane_epi));
        const O *d_base = fs_uniform_ptr((const O *)t.d[e] + (size_t)nblk * BLOCK_ROW * Kw + mblk * BLOCK_COL);
        const int d_rows = min(Nw - nblk * BLOCK_ROW, BLOCK_ROW);
        const i32x4 d_srd = fs_srd(d_base, (uint32_t)d_rows * (uint32_t)Kw * sizeof(O));
        epi_staged<O, ACC>(s_tiles, c00, c01, c10, c11, rf0, cf0, lane_epi, d_srd, Kw, Kw - mblk * BLOCK_COL, alpha,
                           beta);
    }
}

template <typename T, typename O, bool BGRAD, bool ACC>
static void run(const Launch &l) {
    const auto kernel = nt_kernel<T, O, BGRAD, ACC>;
    for (int i = 0; i < l.num_tables; i++) {
        const int ne    = l.experts_per_table[i];
        const int units = l.tables[i].unit_off[ne] + (BGRAD ? ne * ((l.N + BLOCK_ROW - 1) / BLOCK_ROW) : 0);
        hk_check(hipMemsetAsync(l.ctr, 0, XCDS * sizeof(int), l.stream), "grouped16 nt counters");
        hipLaunchKernelGGL(kernel, dim3(min(units, l.grid_cap)), dim3(NUM_THREADS), 0, l.stream, l.tables[i], ne, l.N,
                           l.K, l.alpha, l.beta, l.ctr);
        hk_check(hipGetLastError(), "grouped16 nt_kernel");
    }
}

template <typename T>
static bool run_t(const Launch &l, bool f32, bool bgrad, bool acc) {
    using R = typename Elem<T>::raw;
    if (f32) {
        bgrad ? (acc ? run<T, float, true, true>(l) : run<T, float, true, false>(l))
              : (acc ? run<T, float, false, true>(l) : run<T, float, false, false>(l));
    } else {
        bgrad ? (acc ? run<T, R, true, true>(l) : run<T, R, true, false>(l))
              : (acc ? run<T, R, false, true>(l) : run<T, R, false, false>(l));
    }
    return true;
}

bool launch_nt(const Launch &l, int in_dtype, int out_dtype, bool bgrad, bool acc) {
    const bool f32 = out_dtype == KITTENS_FLOAT32;
    return in_dtype == KITTENS_BFLOAT16 ? run_t<bf16>(l, f32, bgrad, acc) : run_t<half>(l, f32, bgrad, acc);
}

}  // namespace grouped16
}  // namespace cdna4
}  // namespace te_kittens
