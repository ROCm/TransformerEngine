/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#pragma once

#include <cstdint>

enum class WaitKind : uint8_t {
    KERNEL_WAIT = 0,
    ROCSHMEM_WAIT = 1,
    STREAM_WAIT = 2
};

void te_rocshmem_wait_on_stream(uint64_t *sig_addr, WaitKind wait_kind, hipStream_t cur_stream);

void te_rocshmem_putmem_signal(void* dst_ptr, const void* src_ptr, size_t nelement, 
                               uint64_t* sig_addr, uint64_t sigval, int peer, hipStream_t cur_stream);

/* 
These are minimal wrappers around rocshmem functions. As pytorch is a cpp extension,
rocshmem is a static library, and rocshmem does not have separate host / device libraries
we need to move these to common, which handles device code properly.
*/                            
int te_rocshmem_init_thread(int required, int* provided);
void te_rocshmem_finalize();
int te_rocshmem_my_pe();
int te_rocshmem_n_pes();
void* te_rocshmem_malloc(size_t size);
void te_rocshmem_free(void* ptr);
void te_rocshmem_wait_until(uint64_t* signal_addr, uint64_t expected_value,
                             hipStream_t stream);