/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include <hip/hip_runtime.h>
#include <rocshmem.hpp>

#include "../util/logging_hip.h"
#include "rocshmem_waitkernel.hpp"

using namespace rocshmem;

__global__ void wait_until_on_stream_and_reset(uint64_t *wait_flag,
                                               uint64_t wait_value,
                                               uint64_t signal_reset) {
  rocshmem_ulonglong_wait_until((unsigned long long*)wait_flag, 
                                ROCSHMEM_CMP_EQ,
                                (unsigned long long)wait_value);
}

__global__ void rocshmem_putmem_signal_kernel(void* dst_ptr, const void* src_ptr, 
                                              size_t nelement, uint64_t* sig_addr,
                                              uint64_t sigval, int peer) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem_putmem(dst_ptr, src_ptr, nelement, peer);
    rocshmem_fence();
    rocshmem_ulonglong_p((unsigned long long*)sig_addr,
                         (unsigned long long)sigval,
                         peer);
  }
}

void te_rocshmem_putmem_signal(void* dst_ptr, const void* src_ptr, size_t nelement, 
                            uint64_t* sig_addr, uint64_t sigval, int peer, 
                            hipStream_t cur_stream) {
  hipLaunchKernelGGL(rocshmem_putmem_signal_kernel,
                     dim3(1), dim3(1), 0, cur_stream,
                     dst_ptr, src_ptr, nelement, sig_addr, 
                     sigval, peer);
}

void te_rocshmem_wait_on_stream(uint64_t* sig_addr,
                             WaitKind wait_kind,
                             hipStream_t cur_stream) {
  uint64_t wait_value = 1;
  uint64_t signal_reset = 0;

  NVTE_CHECK(wait_kind >= WaitKind::KERNEL_WAIT &&
              wait_kind <= WaitKind::STREAM_WAIT,
              "Invalid wait kind");

  switch (wait_kind) {
// ### wait_until_on_stream not yet implemented for rocshmem  ###
// ### KernelWait is robust but slightly slower due to launch ###
    case WaitKind::ROCSHMEM_WAIT:
    printf("WARNING: rocshmem wait is not implemented yet, defaulting to kernel wait.\n");
    //   rocshmem__ulonglong_wait_until_on_stream(sig_addr,
    //                                 ROCSHMEM_CMP_EQ,
    //                                 wait_value,
    //                                 cur_stream);
    //   hipStreamWriteValue64(cur_stream,
    //                         reinterpret_cast<hipDeviceptr_t>(sig_addr),
    //                         signal_reset, 0);
    //   break;
    case WaitKind::KERNEL_WAIT:
      hipLaunchKernelGGL(wait_until_on_stream_and_reset,
                         dim3(1), dim3(1), 0, cur_stream,
                         sig_addr, wait_value, signal_reset);
      hipStreamWriteValue64(cur_stream,
                            reinterpret_cast<hipDeviceptr_t>(sig_addr),
                            signal_reset, 0);
      break;
    case WaitKind::STREAM_WAIT:
      hipStreamWaitValue64(cur_stream,
                           reinterpret_cast<hipDeviceptr_t>(sig_addr),
                           wait_value, hipStreamWaitValueGte);
      hipStreamWriteValue64(cur_stream,
                            reinterpret_cast<hipDeviceptr_t>(sig_addr),
                            signal_reset, 0);
      break;
  }
}

int te_rocshmem_init_thread(int required, int* provided) {
  if (required == 0 && provided == nullptr) {
    rocshmem_init();
    return 0;
  } else {
    return rocshmem_init_thread(required, provided);
  }
}

void te_rocshmem_finalize() {
  rocshmem_finalize();
}

int te_rocshmem_my_pe() {
  return rocshmem_my_pe();
}

int te_rocshmem_n_pes() {
  return rocshmem_n_pes();
}

void* te_rocshmem_malloc(size_t size) {
  return rocshmem_malloc(size);
}

void te_rocshmem_free(void* ptr) {
  rocshmem_free(ptr);
}

void te_rocshmem_wait_until(uint64_t* signal_addr, uint64_t expected_value,
                             hipStream_t stream);