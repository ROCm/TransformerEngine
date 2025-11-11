/*************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "../extensions.h"

#ifdef NVTE_ENABLE_ROCSHMEM
#include <mpi.h>
#include <rocshmem_api/rocshmem_waitkernel.hpp>
#endif

#include <cuda.h>
#include <cuda_fp8.h>
#include <torch/cuda.h>
#include <torch/extension.h>

namespace transformer_engine::pytorch {

void init_rocshmem_backend(c10d::ProcessGroup *process_group) {
#ifdef NVTE_ENABLE_ROCSHMEM
    auto backend_type = process_group->getBackendType();
    NVTE_CHECK(
        backend_type == c10d::ProcessGroup::BackendType::NCCL ||
        backend_type == c10d::ProcessGroup::BackendType::MPI,
        "Currently only support NCCL or MPI bootstrap for ROCSHMEM. Found: ", 
        c10d::ProcessGroup::backendTypeToString(backend_type)
    );

    int my_rank = process_group->getRank();
    int num_ranks = process_group->getSize();

    static std::once_flag rocshmem_init_flag;
    static bool rocshmem_init_success = false;
    
    std::call_once(rocshmem_init_flag, [backend_type, my_rank, num_ranks]() {
        if (backend_type == c10d::ProcessGroup::BackendType::MPI) {
            int mpi_is_initialized = 0;
            MPI_Initialized(&mpi_is_initialized);
            if (!mpi_is_initialized) {
                int provided;
                MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
                NVTE_CHECK(
                    provided >= MPI_THREAD_SINGLE, 
                    "MPI initialization failed to provide required thread level."
                );
            }
            
            int provided;
            int ret = te_rocshmem_init_thread(MPI_THREAD_MULTIPLE, &provided);
            NVTE_CHECK(
                ret == 0,
                "rocshmem_init_thread() failed with return code: ", ret
            );
            NVTE_CHECK(
                provided >= MPI_THREAD_MULTIPLE,
                "ROCm SHMEM initialization failed to provide MPI_THREAD_MULTIPLE support. Got: ", provided
            );
        } else {
            setenv("ROCSHMEM_RANK", std::to_string(my_rank).c_str(), 0);
            setenv("ROCSHMEM_SIZE", std::to_string(num_ranks).c_str(), 0);
            
            int ret = te_rocshmem_init_thread(0, nullptr);
            NVTE_CHECK(
                ret == 0,
                "rocshmem_init_thread() failed with return code: ", ret,
                ". Make sure PMI or compatible launcher environment is available."
            );
        }
        
        rocshmem_init_success = true;
    });
    
    NVTE_CHECK(rocshmem_init_success, "ROCm SHMEM initialization failed");
    
    int pe = te_rocshmem_my_pe();
    int npes = te_rocshmem_n_pes();

    NVTE_CHECK(
        pe == my_rank,
        "ROCShmem PE rank mismatch: ProcessGroup rank (", my_rank,
        ") != rocshmem_my_pe() (", pe, ")");
    
    NVTE_CHECK(
        npes == num_ranks,
        "ROCShmem total PE count mismatch: ProcessGroup size (", num_ranks,
        ") != rocshmem_n_pes() (", npes, ")");
#else
  NVTE_ERROR("Internal TE error: init_rocshmem_backend cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled without NVTE_ENABLE_ROCSHMEM!");
#endif
}

void rocshmem_wait_on_current_stream(torch::Tensor signal, const std::string &wait_kind) {
#ifdef NVTE_ENABLE_ROCSHMEM
  uint64_t* sig_addr = reinterpret_cast<uint64_t*>(signal.data_ptr());
  hipStream_t cur_stream = (hipStream_t)at::hip::getCurrentHIPStream();

  WaitKind kind_enum;
  if (wait_kind == "kernel") 
    kind_enum = WaitKind::KERNEL_WAIT;
  else if (wait_kind == "rocshmem" || wait_kind == "nvshmem") 
    kind_enum = WaitKind::ROCSHMEM_WAIT;
  else if (wait_kind == "stream") 
    kind_enum = WaitKind::STREAM_WAIT;
  else 
    NVTE_CHECK(false, "Invalid wait kind: ", wait_kind);

  te_rocshmem_wait_on_stream(sig_addr, kind_enum, cur_stream);
#else
  NVTE_ERROR(
      "Internal TE error: rocshmem_wait_on_current_stream cannot be initialized with valid PyTorch ",
      "distributed process groups when TE is compiled without NVTE_ENABLE_ROCSHMEM!");
#endif
}

torch::Tensor create_rocshmem_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
#ifdef NVTE_ENABLE_ROCSHMEM
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kHIP)
                         .device_index(c10::hip::current_device());

  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

  void* ptr = te_rocshmem_malloc(size);
  NVTE_CHECK(ptr != nullptr, "rocshmem_malloc failed for ", size, " bytes");

  return at::from_blob(
      ptr, shape,
      [](void* p) { te_rocshmem_free(p); },
      option_gpu);
#else
  NVTE_ERROR("Internal TE error: create_rocshmem_tensor cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled without NVTE_ENABLE_ROCSHMEM!");
#endif
}

void rocshmem_send_on_current_stream(torch::Tensor src, torch::Tensor dst, int peer,
                                    torch::Tensor signal) {
#ifdef NVTE_ENABLE_ROCSHMEM
  void* src_ptr = reinterpret_cast<void*>(src.data_ptr());
  void* dst_ptr = reinterpret_cast<void*>(dst.data_ptr());
  uint64_t* sig_addr = reinterpret_cast<uint64_t*>(signal.data_ptr());
  size_t nelement = src.numel() * src.element_size();
  uint64_t sigval = 1;

  at::hip::HIPStream cur_stream = at::hip::getCurrentHIPStream();

  te_rocshmem_putmem_signal(dst_ptr, src_ptr, nelement, sig_addr, sigval, peer, (hipStream_t)cur_stream);
#else
  NVTE_ERROR(
      "Internal TE error: rocshmem_send_on_current_stream cannot be initialized with valid PyTorch ",
      "distributed process groups when TE is compiled without NVTE_ENABLE_ROCSHMEM!");
#endif
}

void rocshmem_finalize() {
#ifdef NVTE_ENABLE_ROCSHMEM
  te_rocshmem_finalize();
#else
  NVTE_ERROR("Internal TE error: nvshmem_finalize cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled without NVTE_ENABLE_ROCSHMEM!");
#endif
}

}  // namespace transformer_engine::pytorch
