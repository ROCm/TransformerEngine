/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include "kittens.cuh"

namespace te_kittens::cdna4::mxfp8 {

// Packs raw scales into the lane-native layout the GEMM kernels read.
// A: STEP=64,NG=4; B: STEP=32,NG=8 (hi/lo tile pair).
// FUSED=2 grids for the largest expert, so smaller ones exit early.
template<bool COLWISE, int STEP, int NG, int FUSED = 0, bool SHARDED = false>
__global__ void pack_scales_kernel([[maybe_unused]] const uint8_t *__restrict__ scales,
    uint32_t *__restrict__ ln, int dim, int scale_K, int k_iters, int tiles_per_col,
    [[maybe_unused]] const uint8_t *const *__restrict__ scale_ptrs = nullptr,
    [[maybe_unused]] int expert_stride = 0,
    [[maybe_unused]] const int *__restrict__ k_iters_arr = nullptr,
    [[maybe_unused]] const int *__restrict__ output_offsets = nullptr,
    [[maybe_unused]] int cblk_off = 0, [[maybe_unused]] int tiles_local = 0) {

    constexpr int TILE_WORDS = 256;
    constexpr int PAD_WORDS  = (NG - 1) * STEP + 64; // covers OOB pack_scales read
    __shared__ uint32_t tile[PAD_WORDS];

    const uint8_t *my_scales;
    uint32_t *my_ln;
    int my_k_iters, my_scale_K;

    if constexpr (FUSED == 1) {
        int expert_id = blockIdx.y;
        my_scales  = scale_ptrs[expert_id];
        my_ln      = ln + (size_t)expert_id * expert_stride;
        my_k_iters = k_iters;
        my_scale_K = scale_K;
    } else if constexpr (FUSED == 2) {
        int expert_id = blockIdx.y;
        my_scales  = scale_ptrs[expert_id];
        my_ln      = ln + output_offsets[expert_id];
        my_k_iters = k_iters_arr[expert_id];
        my_scale_K = my_k_iters * 4;
    } else {
        my_scales  = scales;
        my_ln      = ln;
        my_k_iters = k_iters;
        my_scale_K = scale_K;
    }

    int tile_id = blockIdx.x;
    int k_iter, cblk;
    size_t out_tile;
    if constexpr (SHARDED) {
        if (tile_id >= my_k_iters * tiles_local) return;
        k_iter   = tile_id / tiles_local;
        cblk     = tile_id % tiles_local;
        out_tile = (size_t)k_iter * tiles_per_col + cblk_off + cblk;
    } else {
        if (tile_id >= my_k_iters * tiles_per_col) return;
        k_iter   = tile_id / tiles_per_col;
        cblk     = tile_id % tiles_per_col;
        out_tile = (size_t)tile_id;
    }
    int kb_base = k_iter * 4;
    int row0    = cblk * TILE_WORDS;

    for (int i = threadIdx.x; i < PAD_WORDS; i += blockDim.x) {
        uint32_t p = 0;
        if (i < TILE_WORDS) {
            int row = row0 + i;
            if constexpr (COLWISE) {
                int base = kb_base * dim + row;
                p  =  (uint32_t)my_scales[base]              | ((uint32_t)my_scales[base +     dim] << 8)
                   | ((uint32_t)my_scales[base + 2 * dim] << 16) | ((uint32_t)my_scales[base + 3 * dim] << 24);
            } else {
                __builtin_memcpy(&p, &my_scales[(size_t)row * my_scale_K + kb_base], 4);
            }
        }
        tile[i] = p; // OOB tail (i>=256) zero-filled
    }
    __syncthreads();

    int tid = threadIdx.x, lane = tid % 64, grp = tid / 64;
    kittens::fp8e8m0_4 out = kittens::pack_scales((const kittens::fp8e8m0 *)tile, grp * STEP);
    my_ln[(out_tile * NG + grp) * 64 + lane] = out;
}

template<bool COLWISE, int STEP, int NG>
void launch_pack_scales(const uint8_t *scales, uint32_t *ln, int dim, int scale_K, int k_iters, hipStream_t stream) {
    int tiles_per_col = dim / 256;
    pack_scales_kernel<COLWISE, STEP, NG><<<k_iters * tiles_per_col, NG * 64, 0, stream>>>(
        scales, ln, dim, scale_K, k_iters, tiles_per_col);
}

// Packs only the rank's own chunk into a full-width output, for the overlap path's interleaved
// case: peer rows are packed by the gatherer block that fetches them, ours has no gatherer.
template<bool COLWISE, int STEP, int NG>
void launch_pack_scales_sharded(const uint8_t *scales, uint32_t *ln, int cblk_off, int tiles_local,
                                int tiles_per_col, int scale_K, int k_iters, hipStream_t stream) {
    pack_scales_kernel<COLWISE, STEP, NG, 0, true><<<k_iters * tiles_local, NG * 64, 0, stream>>>(
        scales, ln, 0, scale_K, k_iters, tiles_per_col, nullptr, 0, nullptr, nullptr,
        cblk_off, tiles_local);
}


struct TileDma {
    kittens::i32x4 srd[2][2];
    uint32_t m0[2][2];
    uint32_t voff[2];
    uint32_t k_step;
};

template <int NUM_WARPS, typename ST>
__device__ __forceinline__ TileDma make_tile_dma(const void *base, uint32_t half_step, uint32_t k_step,
                                                 uint32_t range, uint32_t row_stride, ST (&tiles)[2][2]) {
    constexpr int bpw = ST::underlying_subtile_bytes_per_thread * kittens::WARP_THREADS;
    static_assert(sizeof(ST) == 2 * NUM_WARPS * bpw);
    static_assert(ST::underlying_subtiles_per_row == 1);
    static_assert(bpw == 1024);
    const int wid = kittens::warpid() % NUM_WARPS;
    const uint8_t *p = reinterpret_cast<const uint8_t *>(base);
    TileDma d;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        d.srd[h][0] = kittens::make_srsrc(p + h * half_step, range);
        d.srd[h][1] = kittens::make_srsrc(p + h * half_step - bpw, range + bpw);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            d.srd[h][0][i] = __builtin_amdgcn_readfirstlane(d.srd[h][0][i]);
            d.srd[h][1][i] = __builtin_amdgcn_readfirstlane(d.srd[h][1][i]);
        }
    }
#pragma unroll
    for (int t = 0; t < 2; t++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            d.m0[t][h] = kittens::to_sgpr_u32(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&tiles[t][h].data[0]))
                                              + wid * 2 * bpw);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int lbo = kittens::laneid() * ST::underlying_subtile_bytes_per_thread
                      + ((wid % (NUM_WARPS / 2)) * 2 + i) * bpw + (wid / (NUM_WARPS / 2)) * NUM_WARPS * bpw;
        const int sub = lbo / ST::underlying_subtile_bytes;
        const int sbo = lbo % ST::underlying_subtile_bytes;
        const uint32_t swz = ST::swizzle({sbo / ST::underlying_subtile_row_bytes,
                                          sbo % ST::underlying_subtile_row_bytes});
        d.voff[i] = (swz / ST::underlying_subtile_row_bytes + sub * ST::underlying_subtile_rows) * row_stride
                  + swz % ST::underlying_subtile_row_bytes;
    }
    d.k_step = k_step;
    return d;
}

__device__ __forceinline__ void tile_dma_issue(const TileDma &d, int h, int kb) {
    asm volatile("buffer_load_dwordx4 %0, %2, %4 offen lds\n\t"
                 "buffer_load_dwordx4 %1, %3, %4 offen offset:1024 lds"
                 :: "v"(d.voff[0]), "v"(d.voff[1]), "s"(d.srd[h][0]), "s"(d.srd[h][1]), "s"(kb * d.k_step)
                 : "memory");
}

__device__ __forceinline__ void tile_dma_load(const TileDma &d, int t, int h, int kb) {
    asm volatile("s_mov_b32 m0, %5\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %0, %2, %4 offen lds\n\t"
                 "buffer_load_dwordx4 %1, %3, %4 offen offset:1024 lds"
                 :: "v"(d.voff[0]), "v"(d.voff[1]), "s"(d.srd[h][0]), "s"(d.srd[h][1]), "s"(kb * d.k_step),
                    "s"(t ? d.m0[1][h] : d.m0[0][h])
                 : "memory", "m0");
}

struct ScaleDma {
    kittens::i32x4 srd_a, srd_b;
    uint32_t m0[2];
    uint32_t a_step, b_step;
};

template <typename GL, typename ST>
__device__ __forceinline__ ScaleDma make_scale_dma(const GL &sa, const GL &sb, int sa_tile, int sb_tile,
                                                   int sa_tiles_per_k, int sb_tiles_per_k, ST (&smem)[2][3]) {
    constexpr uint32_t TB = sizeof(ST);
    static_assert(TB == 1024);
    const uint8_t *pa = reinterpret_cast<const uint8_t *>(sa.raw_ptr) + (size_t)sa_tile * TB;
    const uint8_t *pb = reinterpret_cast<const uint8_t *>(sb.raw_ptr) + (size_t)sb_tile * TB - TB;
    ScaleDma d;
    d.srd_a = kittens::make_srsrc(pa, (uint32_t)((size_t)(sa.batch() - sa_tile) * TB));
    d.srd_b = kittens::make_srsrc(pb, (uint32_t)((size_t)(sb.batch() - sb_tile + 1) * TB));
#pragma unroll
    for (int i = 0; i < 4; i++) {
        d.srd_a[i] = __builtin_amdgcn_readfirstlane(d.srd_a[i]);
        d.srd_b[i] = __builtin_amdgcn_readfirstlane(d.srd_b[i]);
    }
#pragma unroll
    for (int s = 0; s < 2; s++) {
        d.m0[s] = kittens::to_sgpr_u32(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smem[s][0].data[0])));
    }
    d.a_step = sa_tiles_per_k * TB;
    d.b_step = sb_tiles_per_k * TB;
    return d;
}

__device__ __forceinline__ void scale_dma_issue(const ScaleDma &d, int k) {
    if (kittens::warpid() == 0) {
        asm volatile("buffer_load_dwordx4 %0, %1, %2 offen lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:1024 lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:2048 lds"
                     :: "v"(kittens::laneid() * 16), "s"(d.srd_a), "s"(k * d.a_step), "s"(d.srd_b), "s"(k * d.b_step)
                     : "memory");
    }
}

__device__ __forceinline__ void scale_dma_load(const ScaleDma &d, int s, int k) {
    if (kittens::warpid() == 0) {
        asm volatile("s_mov_b32 m0, %5\n\ts_nop 0\n\t"
                     "buffer_load_dwordx4 %0, %1, %2 offen lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:1024 lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:2048 lds"
                     :: "v"(kittens::laneid() * 16), "s"(d.srd_a), "s"(k * d.a_step), "s"(d.srd_b), "s"(k * d.b_step),
                        "s"(s ? d.m0[1] : d.m0[0])
                     : "memory", "m0");
    }
}

template <typename RT, typename ST>
__device__ __forceinline__ uint32_t row_frag_addr(const ST &st, int row_tile, int k) {
    const int lane = kittens::laneid();
    const int row  = lane % RT::base_tile_rows;
    const int col  = RT::base_tile_stride * (lane / RT::base_tile_rows) + k * RT::base_tile_elements_per_stride_group;
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&st.data[0]))
         + row_tile * (RT::rows / ST::underlying_subtile_rows) * ST::underlying_subtile_bytes + ST::swizzle({row, col});
}

template <typename ST, int BUF, typename RT>
__device__ __forceinline__ void load_rows(RT &dst, uint32_t addr0, uint32_t addr1) {
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
        asm volatile("ds_read_b128 %0, %1 offset:%2"
                     : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[i][0].data[0]))
                     : "v"(addr0), "i"(BUF * sizeof(ST) + i * ST::underlying_subtile_bytes)
                     : "memory");
        asm volatile("ds_read_b128 %0, %1 offset:%2"
                     : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[i][0].data[4]))
                     : "v"(addr1), "i"(BUF * sizeof(ST) + i * ST::underlying_subtile_bytes)
                     : "memory");
    }
}

template <typename RT, typename ST>
__device__ __forceinline__ void col_frag_addrs(uint32_t (&addr)[RT::height], const ST &st, int col_offset) {
    const int laneid   = kittens::laneid();
    const int block_id = laneid / 16;
    const int l_within = laneid % 16;
    const int tr_k_grp = l_within / 2;
    const int m_half   = l_within & 1;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
        const int m_col = col_offset + i * RT::base_tile_rows + m_half * 8;
        addr[i] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&st.data[0]))
                + block_id * ST::underlying_subtile_bytes + ST::swizzle({tr_k_grp, m_col});
    }
}

template <typename ST, int BUF, typename RT>
__device__ __forceinline__ void load_cols(RT &dst, const uint32_t (&addr)[RT::height]) {
    constexpr int OFF = BUF * sizeof(ST);
    constexpr int SUB = ST::underlying_subtile_bytes;
    constexpr uint32_t K_HI_XOR = 1088u;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
        asm volatile("ds_read_b64_tr_b8 %0, %2 offset:%3\n"
                     "ds_read_b64_tr_b8 %1, %2 offset:%4\n"
                     : "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[0])),
                       "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[4]))
                     : "v"(addr[i]), "i"(OFF), "i"(OFF + 4 * SUB)
                     : "memory");
        asm volatile("ds_read_b64_tr_b8 %0, %2 offset:%3\n"
                     "ds_read_b64_tr_b8 %1, %2 offset:%4\n"
                     : "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[2])),
                       "=&v"(*reinterpret_cast<float2 *>(&dst.tiles[i][0].data[6]))
                     : "v"(addr[i] ^ K_HI_XOR), "i"(OFF), "i"(OFF + 4 * SUB)
                     : "memory");
    }
}

template <typename ST, typename GL, int COPIES>
__device__ __forceinline__ void g2l_srd(ST &dst, const GL &src, const kittens::coord<ST> &idx,
                                        const uint32_t (&sw)[COPIES], kittens::i32x4 srd, const void *base,
                                        uint32_t lds_base) {
    const kittens::coord<> uc = idx.template unit_coord<2, 3>();
    const uint8_t *gptr = reinterpret_cast<const uint8_t *>(&src[uc]);
    const uint32_t soff = kittens::to_sgpr_u32(static_cast<uint32_t>(gptr - reinterpret_cast<const uint8_t *>(base)));
    uint32_t lds = kittens::to_sgpr_u32(lds_base);
    constexpr int bytes_per_memcpy = ST::rows * ST::cols * sizeof(typename ST::dtype) / COPIES;
#pragma unroll
    for (int i = 0; i < COPIES; i++) {
        asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                     :: "s"(lds), "v"(sw[i]), "s"(srd), "s"(soff) : "memory", "m0");
        lds += bytes_per_memcpy;
    }
}

}  // namespace te_kittens::cdna4::mxfp8
