/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The KOSMOS communicator: one symmetric peer buffer per process group, shared by every op.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

typedef int (*kosmos_allgather_fn)(void *ctx, const void *send, void *recv, size_t bytes);

struct KosmosCollective {
  kosmos_allgather_fn allgather;
  void *ctx;
  int rank;
  int nranks;
};

constexpr int KOSMOS_COMM_MAX_RANKS = 64;
constexpr size_t KOSMOS_COMM_FLAG_STRIDE = 256;

typedef struct KosmosComm_ KosmosComm;

typedef struct {
  void *local;
  void *const *peers;
  void *const *flag_peers;
  size_t flag_bytes;
  size_t bytes;
  int rank, nranks;
  const KosmosCollective *coll;
} KosmosExternalBuffers;

typedef struct {
  void *local;
  void *peers[KOSMOS_COMM_MAX_RANKS];
  size_t bytes;
  int rank, nranks;
  int managed;
} KosmosCommInfo;

// Device memory for everything KOSMOS allocates (unset: hipMalloc / hipFree). alloc returns 0 and a pointer to at least
// bytes, 256-byte aligned. ipc: the buffer is shared with the other ranks by an IPC handle of the allocation that
// holds it and its offset in it, so it must work with hipIpcGetMemHandle (plain hipMalloc memory, not expandable or
// VMM segments). free gets back a pointer alloc returned.
typedef struct {
  int (*alloc)(void *ctx, size_t bytes, int ipc, void **ptr);
  void (*free)(void *ctx, void *ptr);
  void *ctx;
} KosmosAllocator;

// The calling thread's allocator for the KOSMOS calls that follow (null: hipMalloc); returns the previous one. The
// struct must stay valid while it is set; memory is returned to the allocator it came from, whichever is set then.
const KosmosAllocator *kosmos_set_thread_allocator(const KosmosAllocator *a);
// Bytes of device memory KOSMOS currently holds from hipMalloc (allocated with no allocator set).
size_t kosmos_untracked_bytes(void);

// Why the calling thread's last kosmos_comm_create failed ("" if it did not).
const char *kosmos_comm_create_error(void);

size_t kosmos_comm_flag_bytes(int nranks);
hipError_t kosmos_comm_create(KosmosComm **comm, const KosmosCollective *coll, size_t bytes);
hipError_t kosmos_comm_wrap(KosmosComm **comm, const KosmosExternalBuffers *ext);
hipError_t kosmos_comm_reset(KosmosComm *comm);
hipError_t kosmos_comm_destroy(KosmosComm *comm);
hipError_t kosmos_comm_info(const KosmosComm *comm, KosmosCommInfo *info);
const char *kosmos_comm_error(const KosmosComm *comm);
