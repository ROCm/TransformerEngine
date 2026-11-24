/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/
// The KOSMOS communicator: create, wrap, reset, destroy and the op families' shared plumbing.
#include "kosmos_comm_impl.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using kosmos_host::align_up;

// The thread's allocator, and where each live allocation came from (it is freed there, whichever is set then).
thread_local const KosmosAllocator *t_alloc = nullptr;
thread_local std::string t_create_err;
struct Owned {
  KosmosAllocator a;
  size_t bytes;
  bool ext;
};
std::mutex g_mu;
std::unordered_map<void *, Owned> g_owned;
size_t g_untracked = 0;

// Opened peer IPC mappings, shared by every comm of the process: one allocation (e.g. a caching-allocator segment)
// may hold several KOSMOS buffers, and a handle is opened once per process.
struct Mapping {
  void *base;
  int refs;
};
std::map<std::string, Mapping> g_maps;

hipError_t map_open(const hipIpcMemHandle_t &h, void **base) {
  std::lock_guard<std::mutex> lock(g_mu);
  const std::string key((const char *)&h, sizeof(h));
  auto it = g_maps.find(key);
  if (it != g_maps.end()) {
    it->second.refs++;
    *base = it->second.base;
    return hipSuccess;
  }
  const hipError_t e = hipIpcOpenMemHandle(base, h, hipIpcMemLazyEnablePeerAccess);
  if (e == hipSuccess) {
    g_maps[key] = Mapping{*base, 1};
  }
  return e;
}

hipError_t map_close(void *base) {
  std::lock_guard<std::mutex> lock(g_mu);
  for (auto it = g_maps.begin(); it != g_maps.end(); ++it) {
    if (it->second.base == base) {
      if (--it->second.refs == 0) {
        g_maps.erase(it);
        return hipIpcCloseMemHandle(base);
      }
      return hipSuccess;
    }
  }
  return hipErrorInvalidValue;
}

bool coll_ok(const KosmosCollective *c) {
  return c && c->allgather && c->nranks >= 1 && c->nranks <= KOSMOS_COMM_MAX_RANKS && c->rank >= 0 &&
         c->rank < c->nranks;
}

bool agree(KosmosComm *c, bool ok) {
  return kosmos_host::coll_agree(c->coll, ok) > 0;
}

hipError_t device_tables(KosmosComm *c, const std::vector<void *> &flags) {
  const size_t b = (size_t)c->nranks * sizeof(void *);
  hipError_t e = kosmos_host::dzero(&c->ctrs, KOSMOS_COMM_CTR_BYTES);
  if (e == hipSuccess) {
    e = kosmos_host::dalloc(&c->d_flag_peers, c->nranks);
  }
  return e != hipSuccess ? e : hipMemcpy(c->d_flag_peers, flags.data(), b, hipMemcpyHostToDevice);
}

void free_retired(KosmosComm *c) {
  for (const auto &q : c->counters.retired) {
    (void)kosmos_host::dev_free(q.first);
  }
  c->counters.retired.clear();
}

hipError_t release(KosmosComm *c) {
  hipError_t first = hipDeviceSynchronize();
  if (c->client.destroy) {
    c->client.destroy(c->client.state);
    c->client = KosmosCommClient{};
  }
  free_retired(c);
  kosmos_host::dfree(c->counters.ptr);
  kosmos_host::dfree(c->ctrs);
  c->counters.bytes = 0;
  if (c->d_flag_peers) {
    const hipError_t e = kosmos_host::dev_free(c->d_flag_peers);
    first = first != hipSuccess ? first : e;
    c->d_flag_peers = nullptr;
  }
  if (c->managed) {
    for (int i = 0; i < c->nranks; i++) {
      if (i != c->rank && c->peer_map[i]) {
        const hipError_t e = map_close(c->peer_map[i]);
        first = first != hipSuccess ? first : e;
      }
      c->peer[i] = nullptr;
      c->peer_map[i] = nullptr;
    }
    if (kosmos_comm_barrier(c) && first == hipSuccess) {
      first = hipErrorUnknown;
    }
    if (c->ipc_base) {
      const hipError_t e = kosmos_host::dev_free(c->ipc_base);
      first = first != hipSuccess ? first : e;
      c->ipc_base = nullptr;
    }
  }
  return first;
}

}

const KosmosAllocator *kosmos_set_thread_allocator(const KosmosAllocator *a) {
  const KosmosAllocator *prev = t_alloc;
  t_alloc = a;
  return prev;
}

size_t kosmos_untracked_bytes(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_untracked;
}

const char *kosmos_comm_create_error(void) {
  return t_create_err.c_str();
}

hipError_t kosmos_host::dev_alloc(void **p, size_t bytes, bool ipc) {
  *p = nullptr;
  bytes = std::max<size_t>(bytes, 1);
  const KosmosAllocator *a = t_alloc;
  const bool ext = a && a->alloc && a->free;
  if (ext) {
    if (a->alloc(a->ctx, bytes, ipc ? 1 : 0, p) != 0 || !*p) {
      *p = nullptr;
      return hipErrorOutOfMemory;
    }
  } else {
    const hipError_t e = hipMalloc(p, bytes);
    if (e != hipSuccess) {
      *p = nullptr;
      return e;
    }
  }
  std::lock_guard<std::mutex> lock(g_mu);
  g_owned[*p] = Owned{ext ? *a : KosmosAllocator{}, bytes, ext};
  if (!ext) {
    g_untracked += bytes;
  }
  return hipSuccess;
}

hipError_t kosmos_host::dev_free(void *p) {
  if (!p) {
    return hipSuccess;
  }
  Owned o{};
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_owned.find(p);
    if (it != g_owned.end()) {
      o = it->second;
      found = true;
      g_owned.erase(it);
      if (!o.ext) {
        g_untracked -= o.bytes;
      }
    }
  }
  if (found && o.ext) {
    o.a.free(o.a.ctx, p);
    return hipSuccess;
  }
  return hipFree(p);
}

hipError_t kosmos_host::set_error(std::string &err, hipError_t e, const char *fmt, ...) {
  char b[512];
  va_list a;
  va_start(a, fmt);
  vsnprintf(b, sizeof b, fmt, a);
  va_end(a);
  err = b;
  return e;
}

int kosmos_host::coll_agree(const KosmosCollective &coll, bool ok) {
  int mine = ok ? 1 : 0;
  std::vector<int> all((size_t)coll.nranks);
  if (coll.allgather(coll.ctx, &mine, all.data(), sizeof(int))) {
    return -1;
  }
  for (int v : all) {
    ok = ok && v;
  }
  return ok ? 1 : 0;
}

int kosmos_comm_barrier(KosmosComm *comm) {
  return kosmos_host::coll_agree(comm->coll, true) < 0;
}

hipError_t kosmos_comm_region_ok(KosmosComm *comm, const char *op, size_t offset, size_t bytes) {
  if (offset % 256 != 0 || offset > comm->bytes || bytes > comm->bytes - offset) {
    return kosmos_host::set_error(comm->err, hipErrorInvalidValue,
                                  "%s: region [%zu, %zu + %zu) is not 256-aligned or exceeds the comm's %zu bytes",
                                  op, offset, offset, bytes, comm->bytes);
  }
  return hipSuccess;
}

hipError_t kosmos_comm_launch_failed(KosmosComm *comm, const char *op) {
  if (comm->capture_refused) {
    comm->capture_refused = 0;
    return kosmos_host::set_error(comm->err, hipErrorStreamCaptureUnsupported,
                                  "%s: first call for this shape or offset inside a stream capture: call the op's "
                                  "prepare (or the op once) before capturing", op);
  }
  const hipError_t e = hipGetLastError();
  return kosmos_host::set_error(comm->err, e != hipSuccess ? e : hipErrorLaunchFailure,
                                "%s: plan upload, counter allocation or launch failed (%s)", op,
                                hipGetErrorString(e));
}

// First-use setup (allocation, upload, arming) is refused inside a stream capture, which it would invalidate or join.
bool kosmos_comm_setup_ok(KosmosComm *comm, hipStream_t stream) {
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  const bool ok = hipStreamIsCapturing(stream, &st) == hipSuccess && st == hipStreamCaptureStatusNone;
  comm->capture_refused = !ok;
  return ok;
}

// A grown block retires the old one, kept until destroy: a graph captured before keeps using it.
void *kosmos_comm_counters(KosmosComm *comm, size_t bytes, hipStream_t stream) {
  KosmosCommBlock &cb = comm->counters;
  if (cb.bytes < bytes) {
    if (!kosmos_comm_setup_ok(comm, stream)) {
      return nullptr;
    }
    void *p = nullptr;
    if (kosmos_host::dev_alloc(&p, bytes) != hipSuccess) {
      return nullptr;
    }
    if (hipMemsetAsync(p, 0, bytes, stream) != hipSuccess) {
      (void)kosmos_host::dev_free(p);
      return nullptr;
    }
    if (cb.ptr) {
      cb.retired.emplace_back(cb.ptr, cb.bytes);
    }
    cb.ptr = p;
    cb.bytes = bytes;
  }
  return cb.ptr;
}

size_t kosmos_comm_flag_bytes(int nranks) {
  return nranks > 0 ? (size_t)nranks * KOSMOS_COMM_FLAG_STRIDE : 0;
}

hipError_t kosmos_comm_create(KosmosComm **comm, const KosmosCollective *coll, size_t bytes) {
  if (!comm) {
    return hipErrorInvalidValue;
  }
  *comm = nullptr;
  t_create_err.clear();
  if (!coll_ok(coll) || !bytes) {
    return hipErrorInvalidValue;
  }
  KosmosComm *c = new KosmosComm_();
  c->rank = coll->rank;
  c->nranks = coll->nranks;
  c->managed = 1;
  c->has_coll = 1;
  c->coll = *coll;
  c->bytes = align_up(bytes, 4096);
  c->flag_bytes = kosmos_comm_flag_bytes(c->nranks);
  const size_t total = c->bytes + align_up(c->flag_bytes, 4096);

  // The buffer may sit inside a larger allocation (an external allocator's): its IPC handle is the allocation's, and
  // the peers add the offset.
  struct Share {
    hipIpcMemHandle_t h;
    size_t off;
    int ok;
  };
  Share mine;
  memset(&mine, 0, sizeof(mine));
  hipError_t e = kosmos_host::dev_alloc(&c->ipc_base, total, true);
  if (e != hipSuccess) {
    c->ipc_base = nullptr;
  } else if ((e = hipMemset(c->ipc_base, 0, total)) == hipSuccess && (e = hipDeviceSynchronize()) == hipSuccess) {
    hipDeviceptr_t base = nullptr;
    size_t range = 0;
    e = hipMemGetAddressRange(&base, &range, (hipDeviceptr_t)c->ipc_base);
    if (e == hipSuccess) {
      mine.off = (size_t)((char *)c->ipc_base - (char *)base);
      e = hipIpcGetMemHandle(&mine.h, (void *)base);
    }
    if (e != hipSuccess) {
      (void)hipGetLastError();  // the failed call's sticky error would fail the caller's next unrelated call
      (void)kosmos_host::set_error(t_create_err, e, "IPC handle of the %zu-byte comm buffer failed (%s): its memory"
                                   " must be plain hipMalloc memory, not an expandable or VMM segment",
                                   total, hipGetErrorString(e));
    }
  } else {
    (void)hipGetLastError();
    (void)kosmos_host::set_error(t_create_err, e, "allocation of the %zu-byte comm buffer failed (%s)", total,
                                 hipGetErrorString(e));
  }
  mine.ok = e == hipSuccess;
  std::vector<Share> all((size_t)c->nranks);
  if (coll->allgather(coll->ctx, &mine, all.data(), sizeof(Share))) {
    if (c->ipc_base) {
      (void)kosmos_host::dev_free(c->ipc_base);
    }
    delete c;
    return hipErrorUnknown;
  }
  bool ok = true;
  for (const Share &s : all) {
    ok = ok && s.ok;
  }
  std::vector<void *> flags((size_t)c->nranks);
  if (ok) {
    for (int i = 0; i < c->nranks && e == hipSuccess; i++) {
      void *p = c->ipc_base;
      if (i != c->rank) {
        void *m = nullptr;
        if ((e = map_open(all[(size_t)i].h, &m)) != hipSuccess) {
          (void)hipGetLastError();
          (void)kosmos_host::set_error(t_create_err, e, "IPC open of rank %d's buffer: %s", i, hipGetErrorString(e));
          p = nullptr;
        } else {
          c->peer_map[i] = m;
          p = (char *)m + all[(size_t)i].off;
        }
      }
      c->peer[i] = (char *)p;
      flags[(size_t)i] = p ? (char *)p + c->bytes : nullptr;
    }
    if (e == hipSuccess) {
      e = device_tables(c, flags);
    }
  }
  if (!agree(c, ok && e == hipSuccess)) {
    if (t_create_err.empty()) {
      t_create_err = "another rank failed to set up its comm buffer (see that rank's error)";
    }
    (void)release(c);
    delete c;
    return e != hipSuccess ? e : hipErrorUnknown;
  }
  c->local = c->peer[c->rank];
  c->flag_local = c->local + c->bytes;
  c->bytes = bytes;
  *comm = c;
  return hipSuccess;
}

hipError_t kosmos_comm_wrap(KosmosComm **comm, const KosmosExternalBuffers *ext) {
  if (!comm) {
    return hipErrorInvalidValue;
  }
  *comm = nullptr;
  if (!ext || !ext->local || !ext->peers || !ext->bytes || ext->nranks < 1 || ext->nranks > KOSMOS_COMM_MAX_RANKS ||
      ext->rank < 0 || ext->rank >= ext->nranks) {
    return hipErrorInvalidValue;
  }
  if (ext->coll && (!coll_ok(ext->coll) || ext->coll->rank != ext->rank || ext->coll->nranks != ext->nranks)) {
    return hipErrorInvalidValue;
  }
  const int n = ext->nranks;
  const size_t fb = kosmos_comm_flag_bytes(n);
  std::vector<void *> peers((size_t)n), flags((size_t)n);
  hipError_t e = hipMemcpy(peers.data(), ext->peers, (size_t)n * sizeof(void *), hipMemcpyDeviceToHost);
  if (e != hipSuccess) {
    return e;
  }
  if (peers[(size_t)ext->rank] != ext->local) {
    return hipErrorInvalidValue;
  }
  size_t data = ext->bytes;
  if (ext->flag_peers) {
    if (ext->flag_bytes < fb) {
      return hipErrorInvalidValue;
    }
    e = hipMemcpy(flags.data(), ext->flag_peers, (size_t)n * sizeof(void *), hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
      return e;
    }
  } else {
    if (ext->bytes < fb + 256) {
      return hipErrorInvalidValue;
    }
    data = (ext->bytes - fb) & ~(size_t)255;
    for (int i = 0; i < n; i++) {
      flags[(size_t)i] = (char *)peers[(size_t)i] + data;
    }
  }
  KosmosComm *c = new KosmosComm_();
  c->rank = ext->rank;
  c->nranks = n;
  c->managed = 0;
  c->has_coll = ext->coll != nullptr;
  if (ext->coll) {
    c->coll = *ext->coll;
  }
  for (int i = 0; i < n; i++) {
    c->peer[i] = (char *)peers[(size_t)i];
  }
  c->local = (char *)ext->local;
  c->bytes = data;
  c->flag_local = (char *)flags[(size_t)ext->rank];
  c->flag_bytes = fb;
  if ((e = device_tables(c, flags)) != hipSuccess) {
    (void)release(c);
    delete c;
    return e;
  }
  if (c->has_coll) {
    e = hipMemset(c->flag_local, 0, fb);
    if (e == hipSuccess) {
      e = hipDeviceSynchronize();
    }
    if (!agree(c, e == hipSuccess)) {
      (void)release(c);
      delete c;
      return e != hipSuccess ? e : hipErrorUnknown;
    }
  }
  *comm = c;
  return hipSuccess;
}

hipError_t kosmos_comm_reset(KosmosComm *comm) {
  if (!comm) {
    return hipErrorInvalidValue;
  }
  hipError_t e = hipSuccess;
  {
    std::lock_guard<std::mutex> lock(comm->mu);
    e = hipDeviceSynchronize();
    if (comm->counters.ptr) {
      const hipError_t e2 = hipMemset(comm->counters.ptr, 0, comm->counters.bytes);
      e = e != hipSuccess ? e : e2;
    }
    for (const auto &q : comm->counters.retired) {
      const hipError_t e2 = hipMemset(q.first, 0, q.second);
      e = e != hipSuccess ? e : e2;
    }
    const hipError_t e5 = hipMemset(comm->ctrs, 0, KOSMOS_COMM_CTR_BYTES);
    e = e != hipSuccess ? e : e5;
    if (comm->client.reset) {
      comm->client.reset(comm->client.state);
    }
    const hipError_t e3 = hipMemset(comm->flag_local, 0, comm->flag_bytes);
    const hipError_t e4 = hipDeviceSynchronize();
    e = e != hipSuccess ? e : (e3 != hipSuccess ? e3 : e4);
    comm->err.clear();
  }
  if (comm->has_coll && kosmos_comm_barrier(comm) && e == hipSuccess) {
    e = hipErrorUnknown;
  }
  return e;
}

hipError_t kosmos_comm_destroy(KosmosComm *comm) {
  if (!comm) {
    return hipSuccess;
  }
  const hipError_t e = release(comm);
  delete comm;
  return e;
}

hipError_t kosmos_comm_info(const KosmosComm *comm, KosmosCommInfo *info) {
  if (!comm || !info) {
    return hipErrorInvalidValue;
  }
  memset(info, 0, sizeof(*info));
  info->local = comm->local;
  for (int i = 0; i < comm->nranks; i++) {
    info->peers[i] = comm->peer[i];
  }
  info->bytes = comm->bytes;
  info->rank = comm->rank;
  info->nranks = comm->nranks;
  info->managed = comm->managed;
  return hipSuccess;
}

const char *kosmos_comm_error(const KosmosComm *comm) {
  return comm ? comm->err.c_str() : "null comm";
}
