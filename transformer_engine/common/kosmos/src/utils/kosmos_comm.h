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

size_t kosmos_comm_flag_bytes(int nranks);
hipError_t kosmos_comm_create(KosmosComm **comm, const KosmosCollective *coll, size_t bytes);
hipError_t kosmos_comm_wrap(KosmosComm **comm, const KosmosExternalBuffers *ext);
hipError_t kosmos_comm_reset(KosmosComm *comm);
hipError_t kosmos_comm_destroy(KosmosComm *comm);
hipError_t kosmos_comm_info(const KosmosComm *comm, KosmosCommInfo *info);
const char *kosmos_comm_error(const KosmosComm *comm);
