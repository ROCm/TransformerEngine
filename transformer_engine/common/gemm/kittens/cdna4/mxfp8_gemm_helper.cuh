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

constexpr int LANE_LOAD_BYTES = 16;
constexpr int WAVE_LOAD_BYTES = LANE_LOAD_BYTES * kittens::WARP_THREADS;
constexpr int TR_GROUP_LANES  = 16;
constexpr int TR_LANE_BYTES   = 8;

// Buffer-to-LDS tile load state. srd[h][1] starts WAVE_LOAD_BYTES early because the instruction offset
// advances the global and the LDS address alike.
struct TileLoad {
    kittens::i32x4 srd[2][2];
    uint32_t lds[2][2];
    uint32_t sw[2];
    uint32_t k_step;
};

template <int NUM_WARPS, typename ST>
__device__ __forceinline__ TileLoad prefill_tile_load(const void *base, uint32_t half_bytes, uint32_t k_step,
                                                      uint32_t range, uint32_t row_stride, ST (&tiles)[2][2]) {
    static_assert(sizeof(ST) == 2 * NUM_WARPS * WAVE_LOAD_BYTES);
    static_assert(ST::underlying_subtiles_per_row == 1);
    static_assert(ST::underlying_subtile_bytes_per_thread == LANE_LOAD_BYTES);
    const int warp = kittens::warpid() % NUM_WARPS;
    const uint8_t *src = reinterpret_cast<const uint8_t *>(base);
    TileLoad d;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        d.srd[h][0] = kittens::make_srsrc(src + h * half_bytes, range);
        d.srd[h][1] = kittens::make_srsrc(src + h * half_bytes - WAVE_LOAD_BYTES, range + WAVE_LOAD_BYTES);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            d.srd[h][0][i] = __builtin_amdgcn_readfirstlane(d.srd[h][0][i]);
            d.srd[h][1][i] = __builtin_amdgcn_readfirstlane(d.srd[h][1][i]);
        }
    }
#pragma unroll
    for (int stage = 0; stage < 2; stage++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t tile_lds = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&tiles[stage][h].data[0]));
            d.lds[stage][h] = kittens::to_sgpr_u32(tile_lds + warp * 2 * WAVE_LOAD_BYTES);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int lane_off    = kittens::laneid() * LANE_LOAD_BYTES;
        const int byte        = lane_off + ((warp % (NUM_WARPS / 2)) * 2 + i) * WAVE_LOAD_BYTES +
                                (warp / (NUM_WARPS / 2)) * NUM_WARPS * WAVE_LOAD_BYTES;
        const int subtile     = byte / ST::underlying_subtile_bytes;
        const int subtile_off = byte % ST::underlying_subtile_bytes;
        const uint32_t swizzled = ST::swizzle({subtile_off / ST::underlying_subtile_row_bytes,
                                               subtile_off % ST::underlying_subtile_row_bytes});
        d.sw[i] = (swizzled / ST::underlying_subtile_row_bytes + subtile * ST::underlying_subtile_rows) * row_stride +
                  swizzled % ST::underlying_subtile_row_bytes;
    }
    d.k_step = k_step;
    return d;
}

// m0 must already hold this warp's LDS destination.
__device__ __forceinline__ void tile_load(const TileLoad &d, int h, int k) {
    asm volatile("buffer_load_dwordx4 %0, %2, %4 offen lds\n\t"
                 "buffer_load_dwordx4 %1, %3, %4 offen offset:%5 lds"
                 :: "v"(d.sw[0]), "v"(d.sw[1]), "s"(d.srd[h][0]), "s"(d.srd[h][1]), "s"(k * d.k_step),
                    "i"(WAVE_LOAD_BYTES)
                 : "memory");
}

// The LDS DMA reads m0, which needs one wait state after the s_mov.
__device__ __forceinline__ void tile_load_m0(const TileLoad &d, int stage, int h, int k) {
    asm volatile("s_mov_b32 m0, %6\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %0, %2, %4 offen lds\n\t"
                 "buffer_load_dwordx4 %1, %3, %4 offen offset:%5 lds"
                 :: "v"(d.sw[0]), "v"(d.sw[1]), "s"(d.srd[h][0]), "s"(d.srd[h][1]), "s"(k * d.k_step),
                    "i"(WAVE_LOAD_BYTES), "s"(stage ? d.lds[1][h] : d.lds[0][h])
                 : "memory", "m0");
}

// Scale load state: smem[stage] holds the A tile then the B lo/hi pair. sb_srd starts one tile early
// because the instruction offset advances the global and the LDS address alike.
struct ScaleLoad {
    kittens::i32x4 sa_srd, sb_srd;
    uint32_t lds[2];
    uint32_t sa_step, sb_step;
};

template <typename GL, typename ST>
__device__ __forceinline__ ScaleLoad prefill_scale_load(const GL &sa, const GL &sb, int sa_tile, int sb_tile,
                                                        int sa_tiles_per_k, int sb_tiles_per_k, ST (&smem)[2][3]) {
    constexpr uint32_t TILE_BYTES = sizeof(ST);
    static_assert(TILE_BYTES == WAVE_LOAD_BYTES);
    const uint8_t *sa_ptr = reinterpret_cast<const uint8_t *>(sa.raw_ptr) + (size_t)sa_tile * TILE_BYTES;
    const uint8_t *sb_ptr = reinterpret_cast<const uint8_t *>(sb.raw_ptr) + (size_t)sb_tile * TILE_BYTES - TILE_BYTES;
    ScaleLoad d;
    d.sa_srd = kittens::make_srsrc(sa_ptr, (uint32_t)((size_t)(sa.batch() - sa_tile) * TILE_BYTES));
    d.sb_srd = kittens::make_srsrc(sb_ptr, (uint32_t)((size_t)(sb.batch() - sb_tile + 1) * TILE_BYTES));
#pragma unroll
    for (int i = 0; i < 4; i++) {
        d.sa_srd[i] = __builtin_amdgcn_readfirstlane(d.sa_srd[i]);
        d.sb_srd[i] = __builtin_amdgcn_readfirstlane(d.sb_srd[i]);
    }
#pragma unroll
    for (int stage = 0; stage < 2; stage++) {
        const uint32_t stage_lds = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smem[stage][0].data[0]));
        d.lds[stage] = kittens::to_sgpr_u32(stage_lds);
    }
    d.sa_step = sa_tiles_per_k * TILE_BYTES;
    d.sb_step = sb_tiles_per_k * TILE_BYTES;
    return d;
}

// m0 must already hold the LDS destination; warp 0 loads all three scale tiles.
__device__ __forceinline__ void scale_load(const ScaleLoad &d, int k) {
    if (kittens::warpid() == 0) {
        asm volatile("buffer_load_dwordx4 %0, %1, %2 offen lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:%5 lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:%6 lds"
                     :: "v"(kittens::laneid() * LANE_LOAD_BYTES), "s"(d.sa_srd), "s"(k * d.sa_step), "s"(d.sb_srd),
                        "s"(k * d.sb_step), "i"(WAVE_LOAD_BYTES), "i"(2 * WAVE_LOAD_BYTES)
                     : "memory");
    }
}

__device__ __forceinline__ void scale_load_m0(const ScaleLoad &d, int stage, int k) {
    if (kittens::warpid() == 0) {
        asm volatile("s_mov_b32 m0, %7\n\ts_nop 0\n\t"
                     "buffer_load_dwordx4 %0, %1, %2 offen lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:%5 lds\n\t"
                     "buffer_load_dwordx4 %0, %3, %4 offen offset:%6 lds"
                     :: "v"(kittens::laneid() * LANE_LOAD_BYTES), "s"(d.sa_srd), "s"(k * d.sa_step), "s"(d.sb_srd),
                        "s"(k * d.sb_step), "i"(WAVE_LOAD_BYTES), "i"(2 * WAVE_LOAD_BYTES),
                        "s"(stage ? d.lds[1] : d.lds[0])
                     : "memory", "m0");
    }
}

template <typename RT, typename ST>
__device__ __forceinline__ uint32_t row_frag_addr(const ST &st, int row_tile, int k_half) {
    const int lane = kittens::laneid();
    const int row  = lane % RT::base_tile_rows;
    const int col  = RT::base_tile_stride * (lane / RT::base_tile_rows);
    const int k    = k_half * RT::base_tile_elements_per_stride_group;
    const uint32_t tile_off = row_tile * (RT::rows / ST::underlying_subtile_rows) * ST::underlying_subtile_bytes;
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&st.data[0])) + tile_off + ST::swizzle({row, col + k});
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
    const int lane     = kittens::laneid();
    const int block_id = lane / TR_GROUP_LANES;
    const int k_row    = (lane % TR_GROUP_LANES) / 2;
    const int m_half   = lane & 1;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
        const int m_col = col_offset + i * RT::base_tile_rows + m_half * TR_LANE_BYTES;
        const uint32_t sub_off = block_id * ST::underlying_subtile_bytes + ST::swizzle({k_row, m_col});
        addr[i] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&st.data[0])) + sub_off;
    }
}

template <typename ST, int BUF, typename RT>
__device__ __forceinline__ void load_cols(RT &dst, const uint32_t (&addr)[RT::height]) {
    static_assert(std::is_same_v<typename ST::shape, kittens::ducks::st_shape::st_16x128>);
    static_assert(sizeof(typename ST::dtype) == 1);
    static_assert(ST::underlying_subtile_bytes == 2048);
    static_assert(sizeof(ST) % ST::underlying_subtile_bytes == 0);
    constexpr int OFF = BUF * sizeof(ST);
    constexpr int SUB = ST::underlying_subtile_bytes;
    // Upper 8 K rows: +1024 B sets offset bit 10, which st_16x128::swizzle maps to bit 6 ((1024 >> 8) << 4).
    // Applied to the absolute address, so the tile arrays are declared alignas(SUB).
    constexpr uint32_t K_HALF_BYTES = ST::underlying_subtile_rows / 2 * ST::underlying_subtile_row_bytes;
    constexpr uint32_t K_HI_XOR     = K_HALF_BYTES | ((K_HALF_BYTES >> 8) << 4);
    static_assert(K_HI_XOR == 1088);
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

// The LDS DMA reads m0, which needs one wait state after the s_mov.
template <typename ST, typename GL, int COPIES>
__device__ __forceinline__ void load_srd(ST &dst, const GL &src, const kittens::coord<ST> &idx,
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
