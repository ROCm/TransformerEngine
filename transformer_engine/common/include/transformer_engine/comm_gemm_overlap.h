/*************************************************************************
 * This file was modified for portability to AMDGPU
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_OVERLAP_H_

#include <cuda.h>
#include <cuda_fp8.h>
#include <nccl.h>
#include <transformer_engine/comm_gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <functional>

#include "common/comm_gemm_overlap/userbuffers/userbuffers.h"

#ifdef __HIP_PLATFORM_AMD__
struct KosmosComm_;
#endif

#ifdef __HIP_PLATFORM_AMD__
#define NVTE_COMM_OVERLAP_MAX_STREAMS NVTE_ROCM_MAX_RINGS
#else
#define NVTE_COMM_OVERLAP_MAX_STREAMS 3
#endif


/* \brief Check if TE is built with cuBLASMp.
 *
 * \return True if TE is built with cuBLASMp.
 */
bool nvte_built_with_cublasmp();

namespace transformer_engine {

/* \brief Check if Userbufers bootstraps with direct calls to MPI collectives.
 *        This can turned on by building Transformer Engine with the `NVTE_UB_WITH_MPI=1` option.
 *
 * \return True if Userbuffers is built with MPI
 */
bool ubuf_built_with_mpi();

enum class CommOverlapType { RS = 0, AG = 1 };

enum class CommOverlapAlgo {
  BULK_OVERLAP_AG = 0,
  BULK_OVERLAP_RS = 1,
  SPLIT_PIPELINED_AG_P2P = 2,
  SPLIT_PIPELINED_RS = 3,
  SPLIT_PIPELINED_RS_P2P = 4,
  ATOMIC_GEMM_RS = 5,
  ATOMIC_GEMM_AG_P2P = 6,
  ATOMIC_GEMM_RS_P2P = 7,
  EXTERNAL_BULK_OVERLAP_AG = 8,
};

class CommOverlapCore {
 protected:
  static inline communicator *_ub_comm{nullptr};
  static inline bool _comm_created{false};

  int _rank;
  int _tp_id;
  int _tp_size;
  int _num_splits;
  int _math_sms;
  int _num_comm_sm;
  int _cga_size;
  int _use_ce;
  int _ub_reg;
  int _gemm_priority;
  int _comm_priority;
  bool _atomic_gemm{false};
  bool _is_p2p{false};

  bool _with_cublasmp{false};
  NVTECommGemmCtx *_cublasmp_ctx{nullptr};
  NVTECommGemmAlgoType _algo_type = kNVTECommGemmAlgoDefault;

  TensorWrapper _ubuf;
  TensorWrapper _counter;
  float *_ubuf_scale_inv;
  bool _ubuf_scale_inv_initialized{false};

  std::vector<cudaStream_t> _stream_compute;
  cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm, _comm_launch_event;

 private:
  void initialize(int tp_size, int num_splits, int num_max_streams, int comm_cga_size,
                  int gemm_priority, int comm_priority, int num_comm_sm, bool set_sm_margin,
                  bool use_ce, bool atomic_gemm);

 public:
  CommOverlapCore() {}  // dummy constructor for exposing type to Python

  // Constructor for Userbuffers backend
  CommOverlapCore(int myrank, int numranks, int mylocal, int numlocal, int mynode, int numnodes,
                  int tp_size, ExtAllgatherOp allgather_handle, ExtBarrierOp barrier_handle,
                  int num_splits, int num_max_streams, int comm_cga_size, int gemm_priority,
                  int comm_priority, int num_comm_sm, bool set_sm_margin, bool use_ce,
                  bool atomic_gemm);

  // Constructor for cuBLASMp backend
  CommOverlapCore(ncclComm_t nccl_comm_ptr, int tp_rank, int tp_size, int num_comm_sm, bool is_p2p,
                  bool atomic_gemm);

  virtual ~CommOverlapCore();

  void *get_ubuf_dptr() { return _ubuf.dptr(); }

  void set_ubuf_scale_inv(float *scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  virtual void copy_into_buffer(cudaStream_t stream, const TensorWrapper &source, bool local_chunk,
                                bool rowwise = true) {
    NVTE_ERROR("Operation is not implemented.");
  }

  TensorWrapper get_tensor_chunk(const TensorWrapper &source, size_t offset,
                                 const std::vector<size_t> &shape);

  TensorWrapper get_buffer_chunk_like(const TensorWrapper &source, size_t offset,
                                      const std::vector<size_t> &shape);

  int get_tp_size() { return _tp_size; }

  bool is_atomic_gemm() { return _atomic_gemm; }

  bool is_p2p_overlap() { return _is_p2p; }


  bool is_fp8_ubuf() { return _ubuf.element_size() == 1; }

  virtual bool is_fused() { return false; }

  bool with_cublasmp() { return _with_cublasmp; }

  virtual void bulk_overlap(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                            bool transb, TensorWrapper &D, TensorWrapper &bias,
                            TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                            bool accumulate, bool use_split_accumulator, CommOverlapType comm_type,
                            TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void atomic_gemm_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                      bool transb, TensorWrapper &D, TensorWrapper &bias,
                                      TensorWrapper &pre_gelu_out, TensorWrapper &workspace,
                                      bool grad, bool accumulate, bool use_split_accumulator,
                                      TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator,
                                TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void rocm_split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator,
                                TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void atomic_gemm_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                      bool transb, TensorWrapper &D, TensorWrapper &bias,
                                      TensorWrapper &pre_gelu_out, TensorWrapper &workspace,
                                      bool grad, bool accumulate, bool use_split_accumulator,
                                      TensorWrapper &B_copy, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void bulk_overlap_external_ag(cudaStream_t send_stream, cudaStream_t recv_stream,
                                        cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void rocm_split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void fused_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void fused_overlap_bulk_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                     bool transb, TensorWrapper &D, TensorWrapper &bias,
                                     TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                     bool accumulate, bool use_split_accumulator,
                                     cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void fused_overlap_bulk_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                     bool transb, TensorWrapper &D, TensorWrapper &bias,
                                     TensorWrapper &pre_gelu_out, TensorWrapper &workspace,
                                     bool grad, bool accumulate, bool use_split_accumulator,
                                     TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void fused_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator,
                                TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }
};  // CommOverlapCore

class CommOverlapBase : public CommOverlapCore {
 protected:
  int _rs_kernel_type;
  bool _rs_overlap_first_gemm;
  cudaStream_t _stream_comm;
  cudaEvent_t _start_d2dcopy;

 private:
  void initialize(const std::vector<size_t> &buffer_shape, DType buffer_dtype,
                  bool rs_overlap_first_gemm);

 public:
  CommOverlapBase() {}  // dummy constructor for exposing type to Python

  // Constructor for Userbuffers backend
  CommOverlapBase(const std::vector<size_t> &buffer_shape, DType buffer_dtype, int myrank,
                  int numranks, int mylocal, int numlocal, int mynode, int numnodes, int tp_size,
                  ExtAllgatherOp allgather_handle, ExtBarrierOp barrier_handle, int num_splits = 3,
                  int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 2,
                  int gemm_priority = 0, int comm_priority = 0, int num_comm_sm = 16,
                  bool set_sm_margin = true, bool atomic_gemm = false,
                  bool rs_overlap_first_gemm = false);

  // Constructor for cuBLASMp backend
  CommOverlapBase(ncclComm_t nccl_comm_ptr, int tp_rank, int tp_size, int num_comm_sm = 16,
                  bool atomic_gemm = false)
      : CommOverlapCore(nccl_comm_ptr, tp_rank, tp_size, num_comm_sm, false, atomic_gemm) {}

  virtual ~CommOverlapBase();

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf
  */
  void bulk_overlap(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                    TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                    TensorWrapper &workspace, bool grad, bool accumulate,
                    bool use_split_accumulator, CommOverlapType comm_type, TensorWrapper &rs_output,
                    cudaStream_t stream_main) override;


  void atomic_gemm_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                              cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }

  void split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &B_copy,
                        cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void atomic_gemm_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &rs_output,
                              cudaStream_t stream_main) override;

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &rs_output,
                        cudaStream_t stream_main) override;

  void bulk_overlap_external_ag(cudaStream_t send_stream, cudaStream_t recv_stream,
                                cudaStream_t stream_main) override;

  void rocm_split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                           bool transb, TensorWrapper &D, TensorWrapper &bias,
                           TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                           bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                           cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }

  void rocm_split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &rs_output,
                        cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
                        }
};  // CommOverlapBase

class CommOverlapP2PBase : public CommOverlapCore {
 protected:
  bool _is_reduce_scatter{false};
  bool _use_multiatomic_ag{false};
  bool _aggregate;
  int _next_rank;
  int _prev_rank;
  int _rank_round_tp;
  int _num_ubuf_chunks;
  int _self_chunk_id;
  std::vector<TensorWrapper> _ubufs;
  std::vector<cudaStream_t> _stream_send;
#ifdef __HIP_PLATFORM_AMD__  
  std::vector<cudaStream_t> l_stream_send, l_stream_recv;
#endif
  cudaStream_t _stream_recv;
  cudaEvent_t _stop_send, _stop_recv;

  uint64_t _ag_signal_base = 0;
  uint64_t _rs_signal_base = 0;
  bool _fused{false};
#ifdef __HIP_PLATFORM_AMD__
  // KOSMOS backend of the fused methods: a wrapped view of _ubuf, created on first use
  size_t _ubuf_bytes{0};
  ::KosmosComm_ *_kosmos_comm{nullptr};
  bool _kosmos_tried{false};
  int _rs_backend{0};
  unsigned _kosmos_logged{0};
  // MXFP8 region of a fused all-gather buffer: fp8 data then its e8m0 scales; a dual buffer holds two such parts
  size_t _kosmos_mx_bytes{0};
  bool _kosmos_dual{false};
#endif

 private:
  void initialize(const std::vector<size_t> &buffer_shape, DType buffer_dtype,
                  CommOverlapType comm_type, bool aggregate);
#ifdef __HIP_PLATFORM_AMD__
  ::KosmosComm_ *kosmos_comm();
  void kosmos_log(int op, bool kosmos, const char *what);
#endif

 public:
  CommOverlapP2PBase() {}  // dummy constructor for exposing type to Python

  // Constructor for Userbuffers backend
  CommOverlapP2PBase(const std::vector<size_t> &buffer_shape, DType buffer_dtype, int myrank,
                     int numranks, int mylocal, int numlocal, int mynode, int numnodes, int tp_size,
                     ExtAllgatherOp allgather_handle, ExtBarrierOp barrier_handle,
                     CommOverlapType comm_type, int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS,
                     int comm_cga_size = 1, int gemm_priority = 0, int comm_priority = 0,
                     int num_comm_sm = 1, bool set_sm_margin = false, bool use_ce = true,
                     bool atomic_gemm = false, bool aggregate = false, bool fused = false,
                     bool kosmos_dual = false);

  // Constructor for cuBLASMp backend
  CommOverlapP2PBase(ncclComm_t nccl_comm_ptr, int tp_rank, int tp_size, int num_comm_sm = 1,
                     bool atomic_gemm = false)
      : CommOverlapCore(nccl_comm_ptr, tp_rank, tp_size, num_comm_sm, true, atomic_gemm) {}

  virtual ~CommOverlapP2PBase();

  void copy_into_buffer(cudaStream_t stream, const TensorWrapper &source, bool local_chunk,
                        bool rowwise = true) override;

  TensorWrapper get_buffer_chunk_by_id(const TensorWrapper &source, size_t buffer_id);

  void bulk_overlap(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                    TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                    TensorWrapper &workspace, bool grad, bool accumulate,
                    bool use_split_accumulator, CommOverlapType comm_type, TensorWrapper &rs_output,
                    cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }

  /*
  ** Split AllGather + AtomicGEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  void atomic_gemm_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                              cudaStream_t stream_main) override;

  /*
  ** Split AllGather + GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  void split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &B_copy,
                        cudaStream_t stream_main) override;

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void atomic_gemm_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &rs_output,
                              cudaStream_t stream_main) override;

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &rs_output,
                        cudaStream_t stream_main) override;
  /*
  ** ROCm Multiring ReduceScatter + GEMM
  */
  void rocm_split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &rs_output,
                        cudaStream_t stream_main) override;

  /*
  ** ROCm Multiring AllGather + GEMM
  */
  void rocm_split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &B_copy,
                        cudaStream_t stream_main) override;

  /*
  ** ROCm fused AllGather + GEMM implemented with hipKittens
  */
  void fused_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &B_copy,
                        cudaStream_t stream_main) override;

  /*
  ** ROCm fused bulk AllGather implemented with hipKittens
  */
  void fused_overlap_bulk_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                             bool transb, TensorWrapper &D, TensorWrapper &bias,
                             TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                             bool accumulate, bool use_split_accumulator,
                             cudaStream_t stream_main) override;

  /*
  ** ROCm fused bulk ReduceScatter implemented with hipKittens
  */
  void fused_overlap_bulk_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                             bool transb, TensorWrapper &D, TensorWrapper &bias,
                             TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                             bool accumulate, bool use_split_accumulator, TensorWrapper &rs_output,
                             cudaStream_t stream_main) override;

  /*
  ** ROCm fused ReduceScatter + GEMM implemented with hipKittens
  */
  void fused_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &rs_output,
                        cudaStream_t stream_main) override;

  bool is_fused() override { return _fused; }

#ifdef __HIP_PLATFORM_AMD__
  bool fused_bulk_rs_fp32();
  bool kosmos_mxfp8(bool bulk);
  bool kosmos_dgrad_wgrad();
  void kosmos_log_fallback(int op, const char *what);

  /*
  ** ROCm fused AllGather + dgrad and wgrad GEMMs (KOSMOS, MXFP8)
  */
  void fused_overlap_ag_dgrad_wgrad(const TensorWrapper &W, const TensorWrapper &X, TensorWrapper &dX,
                                    TensorWrapper &dW, TensorWrapper &workspace, bool accumulate,
                                    cudaStream_t stream_main);
#else
  bool fused_bulk_rs_fp32() { return false; }
  bool kosmos_mxfp8(bool bulk) { return false; }
  bool kosmos_dgrad_wgrad() { return false; }
  void kosmos_log_fallback(int op, const char *what) {}
  void fused_overlap_ag_dgrad_wgrad(const TensorWrapper &W, const TensorWrapper &X, TensorWrapper &dX,
                                    TensorWrapper &dW, TensorWrapper &workspace, bool accumulate,
                                    cudaStream_t stream_main) {
    NVTE_ERROR("Operation not supported.");
  }
#endif

  /*
  ** This function overlaps the AG for the current communicator object with the GEMM for the overlap_gemm object.
  ** The gemm for overlap_gemm is assumed to have been previously started.
  */
  void bulk_overlap_external_ag(cudaStream_t send_stream, cudaStream_t recv_stream,
                                cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }
};  // CommOverlapP2PBase

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMM_GEMM_OVERLAP_H_
