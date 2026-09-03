// Steal order: own bucket, then the local chunk, then the other XCDs.
int found = -1;
const int b0 = (int)blockIdx.x % NUM_XCDS_AFF;
for (int s = 0; s <= NUM_XCDS_AFF; s++) {
    int bb;
    if (s == 0) {
        bb = b0;
    } else if (s == 1) {
        bb = my_pe;
    } else {
        bb = (b0 + s - 1) & (NUM_XCDS_AFF - 1);
    }
    if (buckets.cnt[bb] == 0) continue;
    if (__hip_atomic_load(&bucket_ctr[bb], __ATOMIC_RELAXED,
                          __HIP_MEMORY_SCOPE_AGENT) >= buckets.cnt[bb]) continue;
    const int idx = atomicAdd(&bucket_ctr[bb], 1);
    if (idx < buckets.cnt[bb]) {
        found = buckets.off[bb] + idx;
        break;
    }
}
s_tile_idx = (found < 0) ? num_tiles : found;
