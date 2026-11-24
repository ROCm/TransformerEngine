/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The communicator's layout and the host plumbing shared by kosmos_comm.cpp and the op families.
#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "kosmos_comm.h"

// Per-comm device words that never move: the ready counter, then the ops' tile and XCD bucket counters.
constexpr size_t KOSMOS_COMM_CTR_BYTES  = 256;
constexpr size_t KOSMOS_COMM_TILE_OFF   = 64;
constexpr size_t KOSMOS_COMM_BUCKET_OFF = 128;

struct KosmosCommClient {
  void *state;
  void (*reset)(void *state);
  void (*destroy)(void *state);
};

struct KosmosCommBlock {
  void *ptr;
  size_t bytes;
  std::vector<std::pair<void *, size_t>> retired;
};

struct KosmosComm_ {
  int rank, nranks;
  int managed;
  int has_coll;
  KosmosCollective coll;
  char *local;
  char *peer[KOSMOS_COMM_MAX_RANKS];
  size_t bytes;
  char *flag_local;
  void **d_flag_peers;
  char *ctrs;
  void *ipc_base;
  void *peer_map[KOSMOS_COMM_MAX_RANKS];  // the peers' opened IPC mappings (peer[i] is at an offset in them)
  size_t flag_bytes;
  KosmosCommClient client;
  KosmosCommBlock counters;
  int capture_refused;
  std::mutex mu;
  std::string err;
};

int kosmos_comm_barrier(KosmosComm *comm);
hipError_t kosmos_comm_region_ok(KosmosComm *comm, const char *op, size_t offset, size_t bytes);
hipError_t kosmos_comm_launch_failed(KosmosComm *comm, const char *op);
bool kosmos_comm_setup_ok(KosmosComm *comm, hipStream_t stream);
void *kosmos_comm_counters(KosmosComm *comm, size_t bytes, hipStream_t stream);

namespace kosmos_host {

hipError_t set_error(std::string &err, hipError_t e, const char *fmt, ...);
int coll_agree(const KosmosCollective &coll, bool ok);

inline size_t align_up(size_t x, size_t a) {
  return (x + a - 1) / a * a;
}

struct Carve {
  char *base;
  size_t used;
  size_t cap;
  void *take(size_t bytes) {
    void *p = base + used;
    used += align_up(bytes, 256);
    return p;
  }
  bool fits() const {
    return used <= cap;
  }
};

// Device memory through the thread's KosmosAllocator (or hipMalloc), returned by dev_free to where it came from.
hipError_t dev_alloc(void **p, size_t bytes, bool ipc = false);
hipError_t dev_free(void *p);

template <typename T>
hipError_t dalloc(T **p, size_t n) {
  *p = nullptr;
  return dev_alloc((void **)p, std::max<size_t>(n, 1) * sizeof(T));
}

template <typename T>
hipError_t dzero(T **p, size_t n) {
  hipError_t e = dalloc(p, n);
  return e != hipSuccess ? e : hipMemset(*p, 0, std::max<size_t>(n, 1) * sizeof(T));
}

template <typename T>
hipError_t dupload(T **p, const std::vector<T> &h) {
  hipError_t e = dalloc(p, h.size());
  if (e != hipSuccess || h.empty()) {
    return e;
  }
  return hipMemcpy(*p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice);
}

template <typename T>
void dfree(T *&p) {
  if (p) {
    (void)dev_free((void *)p);
  }
  p = nullptr;
}

template <typename S>
void client_reset(void *s) {
  static_cast<S *>(s)->reset();
}

template <typename S>
void client_destroy(void *s) {
  delete static_cast<S *>(s);
}

template <typename S>
S &client_state(KosmosComm *c) {
  if (!c->client.state) {
    c->client.state = new S();
    c->client.reset = client_reset<S>;
    c->client.destroy = client_destroy<S>;
  }
  return *static_cast<S *>(c->client.state);
}

}
