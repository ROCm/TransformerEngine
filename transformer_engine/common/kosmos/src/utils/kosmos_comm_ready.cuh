/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The communicator's in-kernel cross-rank arrival handshake: per call, every rank publishes its epoch to every peer.
#pragma once

#include <hip/hip_runtime.h>

#include "kosmos_comm_impl.h"

typedef __attribute__((address_space(1))) uint64_t kosmos_gu64;

struct KosmosReady {
  void *const *peers;
  const char *local;
  const uint64_t *ctr;
};

inline KosmosReady kosmos_comm_ready(KosmosComm *c) {
  return KosmosReady{c->d_flag_peers, c->flag_local, reinterpret_cast<const uint64_t *>(c->ctrs)};
}

// The epoch is one more than the comm's ready counter, a device word that the call's last kernel advances, so graph
// replays and eager calls stay in step.
__device__ __forceinline__ uint64_t kosmos_ready_epoch(const KosmosReady &r) {
  return *(const kosmos_gu64 *)r.ctr + 1;
}

// Called by one workgroup at kernel start, after all of this rank's earlier stream work (which includes its previous
// call on the comm): the release store makes that work visible to the peer that acquires the epoch.
__device__ __forceinline__ void kosmos_ready_arrive(const KosmosReady &r, int rank, int nranks) {
  const int c = static_cast<int>(threadIdx.x);
  if (c < nranks && c != rank) {
    kosmos_gu64 *pub = (kosmos_gu64 *)(static_cast<char *>(r.peers[c]) +
                                       static_cast<size_t>(rank) * KOSMOS_COMM_FLAG_STRIDE);
    __hip_atomic_store(pub, kosmos_ready_epoch(r), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// The lane is opaque so a wait inside a persistent loop rebuilds its flag address instead of holding it in VGPRs.
__device__ __forceinline__ void kosmos_ready_wait(const KosmosReady &r, int rank, int nranks) {
  int c = static_cast<int>(threadIdx.x);
  asm volatile("" : "+v"(c));
  if (c < nranks && c != rank) {
    const kosmos_gu64 *f = (const kosmos_gu64 *)(r.local + static_cast<size_t>(c) * KOSMOS_COMM_FLAG_STRIDE);
    const uint64_t e = kosmos_ready_epoch(r);
    while (__hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < e) {
      __builtin_amdgcn_s_sleep(1);
    }
    __hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
}

// By thread 0 of the call's last kernel's last workgroup out (every workgroup has read the epoch): the ops' tile
// counters sit KOSMOS_COMM_TILE_OFF bytes past the ready counter.
__device__ __forceinline__ void kosmos_ready_done(int *tile_counter) {
  kosmos_gu64 *ctr = (kosmos_gu64 *)(reinterpret_cast<char *>(tile_counter) - KOSMOS_COMM_TILE_OFF);
  __hip_atomic_fetch_add(ctr, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
