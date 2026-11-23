/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
    const bool static_sched = num_tiles <= SCHED_ROUNDS * (int)gridDim.x;
    const int limit = (num_tiles / (NUM_XCDS_AFF * SCHED_CHUNK)) * (NUM_XCDS_AFF * SCHED_CHUNK);
    if (threadIdx.x == 0) {
        const int tail = num_tiles - limit;
        int off = 0;
        for (int b = 0; b < NUM_XCDS_AFF; b++) {
            buckets.off[b] = off;
            buckets.cnt[b] = limit / NUM_XCDS_AFF + (tail > b ? (tail - b + NUM_XCDS_AFF - 1) / NUM_XCDS_AFF : 0);
            off += buckets.cnt[b];
        }
    }
    int iter = 0;
    int wgid = 0;
    int zero;
    asm volatile("s_mov_b32 %0, 0" : "=s"(zero));
    if (zero != 0) {
        asm volatile("" : "+s"(wgid));
        goto tile;
    }
    asm volatile("" : "+s"(iter));
#pragma unroll 1
    for (;; iter++) {
        if (threadIdx.x == 0) {
            if (static_sched) {
                const int i = (int)blockIdx.x + iter * (int)gridDim.x;
                s_wgid = i < num_tiles ? kittens::chiplet_transform_chunked(i, num_tiles, NUM_XCDS_AFF, SCHED_CHUNK)
                                       : num_tiles;
            } else {
                int *bucket_ctr = ctr;
                const int my_pe = (int)blockIdx.x % NUM_XCDS_AFF;
                int s_tile_idx;
#include "xcd_steal.inc"
                s_wgid = num_tiles;
                if (s_tile_idx < num_tiles) {
                    int bb = NUM_XCDS_AFF - 1;
                    while (bb > 0 && (buckets.cnt[bb] == 0 || buckets.off[bb] > s_tile_idx)) {
                        bb--;
                    }
                    const int idx   = s_tile_idx - buckets.off[bb];
                    const int nfull = limit / NUM_XCDS_AFF;
                    if (idx < nfull) {
                        s_wgid = (idx / SCHED_CHUNK) * (NUM_XCDS_AFF * SCHED_CHUNK) + bb * SCHED_CHUNK
                               + idx % SCHED_CHUNK;
                    } else {
                        s_wgid = limit + bb + NUM_XCDS_AFF * (idx - nfull);
                    }
                }
            }
        }
        __syncthreads();
        wgid = __builtin_amdgcn_readfirstlane(s_wgid);
        if (wgid >= num_tiles) {
            break;
        }
    tile:
