/*************************************************************************
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <transformer_engine/comm_gemm_overlap.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include "common/common.h"
#include "common/util/cuda_driver.h"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"
#include "common/util/system.h"
#include "userbuffers/userbuffers.h"
#ifdef USE_HIPKITTENS_GEMM
#include "../gemm/kittens/fused_ag_gemm.h"
#endif

namespace transformer_engine {
#if 0
// Recursive doubling AG code for future reference
void CommOverlapP2PBase::rocm_split_overlap_ag_rd(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;
  // Get GEMM dimensions between TN and NN input layouts
  const size_t m = (transa) ? A.size(0) : A.size(1);
  const size_t k = (transa) ? A.size(1) : A.size(0);
  const size_t n_chunk = _ubufs[0].size(0);
  const int comm_bytes = _ubufs[0].bytes();
  const bool do_gelu = pre_gelu_out.numel() > 0;
  const size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Check B copy sizing
  if (B_copy.numel() > 0) {
    NVTE_CHECK(B_copy.numel() == _ubuf.numel(), "Expected all-gathered B copy buffer with ",
               _ubuf.numel(), " elements but got ", B_copy.numel());
    NVTE_CHECK(B_copy.element_size() == _ubuf.element_size(),
               "Expected all-gathered B copy buffer with ", _ubuf.element_size() * 8,
               "-bit data type but got ", B_copy.element_size() * 8, "-bit");
  }

  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[0], _start_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }

  int steps = 31 - __builtin_clz(_tp_size);

  // Chunk dims
  std::vector<size_t> input_b_chunk_shape =
      (transb ? std::vector<size_t>{k, n_chunk} : std::vector<size_t>{n_chunk, k});
  std::vector<size_t> output_chunk_shape = {n_chunk, m};
  size_t input_b_chunk_size = n_chunk * k;
  size_t output_chunk_size = n_chunk * m;

  // GEMM
  auto input_b_chunk =
      get_buffer_chunk_like(B, input_b_chunk_size * _tp_id, input_b_chunk_shape);
  auto output_chunk =
      get_tensor_chunk(D, output_chunk_size * _tp_id, output_chunk_shape);
  auto aux_chunk =
      (do_gelu)
          ? get_tensor_chunk(pre_gelu_out, output_chunk_size * _tp_id, {n_chunk, k})
          : TensorWrapper(nullptr, std::vector<size_t>{0}, pre_gelu_out.dtype());
  auto workspace_chunk = get_tensor_chunk(
      workspace, (_tp_id % _stream_compute.size()) * workspace_size_chunk, {workspace_size_chunk});

  nvte_cublas_gemm(A.data(), input_b_chunk.data(), output_chunk.data(), bias.data(),
                    aux_chunk.data(), transa, transb, grad, workspace_chunk.data(), accumulate,
                    use_split_accumulator, _math_sms,
                    _stream_compute[_tp_id % _stream_compute.size()]);

  std::vector<size_t> owned_chunks;
  owned_chunks.reserve(_tp_size);
  owned_chunks.push_back(_tp_id);
  size_t offset = 1;

  for (int step = 0; step < steps; step++) {
    int send_rank = (_tp_id + offset) % _tp_size;
    int recv_rank = (_tp_id - offset + _tp_size) % _tp_size;
    
    for (int i = 0; i < owned_chunks.size(); i++) {
      size_t send_offset = owned_chunks[i] * comm_bytes;
      userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset,
                       comm_bytes, _ub_comm, send_rank, _stream_send[i % _stream_send.size()]);
    }

    std::vector<size_t> new_chunks;
    for (size_t i = 0; i < owned_chunks.size(); i++) {
      size_t new_chunk_id = (recv_rank + i * offset) % _tp_size;
      if (new_chunk_id >= _tp_size || 
          std::find(owned_chunks.begin(), owned_chunks.end(), new_chunk_id) != owned_chunks.end()) continue;
      size_t recv_offset  = new_chunk_id * comm_bytes;
      size_t stream_id    = new_chunks.size() % _stream_compute.size();

      userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset,
                       comm_bytes, _ub_comm, recv_rank, _stream_recv);

      NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[stream_id], _stop_recv, 0));

      auto input_b_chunk = get_buffer_chunk_like(B, input_b_chunk_size * new_chunk_id, input_b_chunk_shape);
      output_chunk = get_tensor_chunk(D, output_chunk_size * new_chunk_id, output_chunk_shape);
      aux_chunk = (do_gelu) ? get_tensor_chunk(pre_gelu_out, output_chunk_size * new_chunk_id, {n_chunk, k})
                            : TensorWrapper(nullptr, std::vector<size_t>{0}, pre_gelu_out.dtype());
      workspace_chunk = get_tensor_chunk(workspace, stream_id * workspace_size_chunk, {workspace_size_chunk});

      nvte_cublas_gemm(A.data(), input_b_chunk.data(), output_chunk.data(), bias.data(),
                       aux_chunk.data(), transa, transb, grad, workspace_chunk.data(), accumulate,
                       use_split_accumulator, _math_sms,
                       _stream_compute[stream_id]);
      
      new_chunks.push_back(new_chunk_id);
    }
    owned_chunks.insert(owned_chunks.end(), new_chunks.begin(), new_chunks.end());
    offset <<= 1;
  }

  if (B_copy.numel() > 0) {
    NVTE_CHECK_CUDA(cudaMemcpyAsync(B_copy.dptr(), _ubuf.dptr(), _ubuf.bytes(),
                                    cudaMemcpyDeviceToDevice, _stream_send[0]));
  }

  _ub_comm->sms = ori_sms;
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send[0]));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));
} // rocm_split_overlap_ag_rd
#endif // #if 0

// TODO: Generalize for TP other than 2,4,8 using Walecki construction 
constexpr int tp_next_8[7][8] = {
  {1, 5, 4, 6, 3, 2, 7, 0},
  {2, 6, 1, 0, 5, 7, 4, 3},
  {3, 7, 0, 5, 6, 4, 1, 2},
  {4, 3, 6, 2, 7, 0, 5, 1},
  {5, 2, 7, 4, 1, 3, 0, 6},
  {6, 0, 5, 7, 2, 1, 3, 4},
  {7, 4, 3, 1, 0, 6, 2, 5},
};

constexpr int tp_prev_8[7][8] = {
  {7, 0, 5, 4, 2, 1, 3, 6},
  {3, 2, 0, 7, 6, 4, 1, 5},
  {2, 6, 7, 0, 5, 3, 4, 1},
  {5, 7, 3, 1, 0, 6, 2, 4},
  {6, 4, 1, 5, 3, 0, 7, 2},
  {1, 5, 4, 6, 7, 2, 0, 3},
  {4, 3, 6, 2, 1, 7, 5, 0},
};

// No full Hamiltonian decomposition for TP=4 TP=6 (Tillson’s Theorem)
// Further optimization for these cases may be multiring w/ RD for example
constexpr int tp_next_4[2][4] = {
  {1, 2, 3, 0},
  {3, 0, 1, 2},
};

constexpr int tp_prev_4[2][4] = {
  {3, 0, 1, 2},
  {1, 2, 3, 0}
};

template<int NUM_RINGS, int TP_SIZE>
constexpr bool multiring_hamiltonian_check(const int (&next)[NUM_RINGS][TP_SIZE]) {
    for (int r = 0; r < NUM_RINGS; ++r) {
        bool visited[TP_SIZE] = {};

        int curr = 0;
        for (int step = 0; step < TP_SIZE; ++step) {
            if (visited[curr]) return false;
            visited[curr] = true;
            curr = next[r][curr];
        }

        if (curr != 0) return false;

        for (int i = 0; i < TP_SIZE; ++i) {
            if (!visited[i]) return false;
        }
    }
    return true;
}

template<int NUM_RINGS, int TP_SIZE>
constexpr bool rings_are_unique(
    const int next[NUM_RINGS][TP_SIZE])
{
    for (int src = 0; src < TP_SIZE; ++src) {
        bool seen[TP_SIZE] = {};

        for (int r = 0; r < NUM_RINGS; ++r) {
            int dst = next[r][src];

            // No self-send
            if (dst == src)
                return false;

            if (seen[dst])
                return false;

            seen[dst] = true;
        }
    }
    return true;
}

template<int NUM_RINGS, int TP_SIZE>
constexpr bool prev_is_inverse_of_next(
    const int next[NUM_RINGS][TP_SIZE],
    const int prev[NUM_RINGS][TP_SIZE])
{
    for (int r = 0; r < NUM_RINGS; ++r) {
        for (int i = 0; i < TP_SIZE; ++i) {
            int n = next[r][i];
            int p = prev[r][i];

            if (n < 0 || n >= TP_SIZE) return false;
            if (p < 0 || p >= TP_SIZE) return false;

            if (prev[r][n] != i) return false;
            if (next[r][p] != i) return false;
        }
    }
    return true;
}

static_assert(multiring_hamiltonian_check<2,4>(tp_next_4), "Non-Hamiltonian ring present!");
static_assert(multiring_hamiltonian_check<7,8>(tp_next_8), "Non-Hamiltonian ring present!");

static_assert(rings_are_unique<2,4>(tp_next_4), "Rings overlap");
static_assert(rings_are_unique<7,8>(tp_next_8), "Rings overlap");

static_assert(prev_is_inverse_of_next<2,4>(tp_next_4, tp_prev_4), "tp_prev_4 is not inverse of tp_next_4");
static_assert(prev_is_inverse_of_next<7,8>(tp_next_8, tp_prev_8), "tp_prev_8 is not inverse of tp_next_8");

#ifdef USE_HIPKITTENS_GEMM
// Fused all-gather + GEMM, launched by fused_overlap_ag below.
static bool hk_fused_ag_gemm(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb, TensorWrapper &D,
                             const TensorWrapper &bias, const TensorWrapper &pre_gelu_out,
                             const TensorWrapper &B_copy, TensorWrapper &workspace, bool accumulate,
                             const TensorWrapper &ubuf, const TensorWrapper &chunk, communicator *comm,
                             int reg, int tp_id, int tp_size, uint64_t signal, size_t scale_base_offset,
                             size_t scale_chunk_bytes, cudaStream_t stream) {
  // TODO: Add bias support
  NVTE_CHECK(B.dptr() == ubuf.dptr(),
             "fused AG+GEMM reached with invalid B tensor!");

  NVTE_CHECK(!transb && !accumulate && bias.numel() == 0 && pre_gelu_out.numel() == 0 && B_copy.numel() == 0,
             "fused AG+GEMM reached with an unsupported epilogue");
  bool is_bf16 = A.dtype() == DType::kBFloat16 && ubuf.dtype() == DType::kBFloat16 && D.dtype() == DType::kBFloat16;
  //TODO: Extend to E5M2 and mixed types
  bool is_fp8 = A.dtype() == DType::kFloat8E4M3 && B.dtype() == DType::kFloat8E4M3 && D.dtype() == DType::kBFloat16;
  NVTE_CHECK(is_bf16 || is_fp8,
             "fused AG+GEMM reached with unsupported operand types");

  auto A_tensor = convertNVTETensorCheck(A.data());
  auto B_tensor = convertNVTETensorCheck(B.data());

  NVTE_CHECK(A_tensor->scaling_mode == B_tensor->scaling_mode,
             "fused AG+GEMM expects A and B tensors to have the same scaling mode");
  if (is_fp8 && A_tensor->scaling_mode != NVTE_MXFP8_1D_SCALING) {
    NVTE_ERROR("fused AG+GEMM with fp8 UB only supports MXFP8_1D_SCALING recipe");
  }

  // MXFP8 supports TN and NN. B is consumed row-wise in both (transb is false either way);
  // A is row-wise for TN and column-wise for NN, matching the scale selection just below.
  if (is_fp8) {
    if (transa) {
      NVTE_CHECK(A_tensor->has_data(),
                 "fused AG+GEMM with MXFP8 reached with A missing row-wise usage");
    } else {
      NVTE_CHECK(A_tensor->has_columnwise_data(),
                 "fused AG+GEMM with MXFP8 reached with A missing column-wise usage");
    }
    NVTE_CHECK(B_tensor->has_data(), "fused AG+GEMM with MXFP8 reached with B missing row-wise usage");
  }

  const void* scale_A = nullptr;
  const void* scale_B = nullptr;
  if (is_fp8) {
    scale_A = transa ? A_tensor->scale_inv.dptr : A_tensor->columnwise_scale_inv.dptr;
    scale_B = transb ? B_tensor->columnwise_scale_inv.dptr : B_tensor->scale_inv.dptr;
  }

  const size_t m       = (transa) ? A.size(0) : A.size(1);
  const size_t k       = (transa) ? A.size(1) : A.size(0);
  const size_t n_chunk = chunk.size(0);
  NVTE_CHECK((tp_size == 4 || tp_size == 8) && m % 256 == 0 && k % 128 == 0 && k >= 256 && n_chunk % 256 == 0,
             "fused AG+GEMM reached with an ineligible shape (m=", m, " k=", k, " n_chunk=", n_chunk,
             " tp_size=", tp_size, ")");

  const int rank_round_tp = comm->myrank - tp_id;
  // Row-wise and column-wise MXFP8 data are quantized independently, so the operand buffer has to
  // come from the same usage as the scales picked above: row-wise for TN, column-wise for NN.
  // TensorWrapper::dptr() is hard-wired to the row-wise buffer -- see cublaslt_gemm.cu:185 for the
  // same pairing on the non-overlapped path.
  KittensFusedAgGemmArgs args{
      (is_fp8 && !transa) ? A.columnwise_dptr() : A.dptr(), ubuf.dptr(), D.dptr(), scale_A, scale_B,
      reinterpret_cast<char *>(comm->gpu_ptrs) + reg * comm->nvsize * sizeof(void *),
      rank_round_tp % comm->nvsize, comm->nvsize,
      GET_RECV_PTR_BY_INDEX(rank_round_tp, comm, reg, 0), comm->gpu_ptrs,
      static_cast<size_t>(GET_SEND_PTR_BY_INDEX(0, comm, reg, 0) - reinterpret_cast<char *>(comm->peer_ptr[0][0])),
      static_cast<size_t>(GET_RECV_PTR_BY_INDEX(1, comm, reg, 0) - GET_RECV_PTR_BY_INDEX(0, comm, reg, 0)),
      signal, static_cast<int>(m), static_cast<int>(n_chunk * tp_size), static_cast<int>(k), transa,
      tp_id, tp_size, chunk.bytes(), scale_base_offset, scale_chunk_bytes, workspace.dptr(), workspace.bytes(), stream};
  if (A_tensor->scaling_mode == NVTE_MXFP8_1D_SCALING) {
    return kittens_fused_ag_gemm_mxfp8(args);
  }
  return kittens_fused_ag_gemm_bf16(args);
}
#endif

void CommOverlapP2PBase::fused_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main) {
#ifdef USE_HIPKITTENS_GEMM
  if (kittens_fused_ag_gemm_supported(cuda::sm_arch())) {
    const bool launched = hk_fused_ag_gemm(A, transa, B, transb, D, bias, pre_gelu_out, B_copy,
                                           workspace, accumulate, _ubuf, _ubufs[0], _ub_comm,
                                           _ub_reg, _tp_id, _tp_size, _ag_signal_base + _tp_size,
                                           _scale_base_offset, _scale_chunk_bytes,
                                           stream_main);
    NVTE_CHECK(launched, "fused AG+GEMM failed to launch");
    _ag_signal_base += _tp_size;
    return;
  }
#endif
  NVTE_ERROR("fused AG+GEMM was selected but is not built into this library");
}

// TODO: Introduce HIPGraphs for dependency management.
void CommOverlapP2PBase::rocm_split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;
  // Get GEMM dimensions between TN and NN input layouts
  const size_t m = (transa) ? A.size(0) : A.size(1);
  const size_t k = (transa) ? A.size(1) : A.size(0);
  const size_t n_chunk = _ubufs[0].size(0);
  // Get communication and GEMM output chunk sizes
  const int comm_bytes = _ubufs[0].bytes();
  const bool do_gelu = pre_gelu_out.numel() > 0;
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  const int max_rings = (_tp_size == 4) ? 2 :
                        (_tp_size == 6) ? 4 :
                         _tp_size - 1;
  const int num_rings = std::min({
                                  transformer_engine::getenv<int>("GPU_MAX_HW_QUEUES", 4), 
                                  _tp_size - 1,
                                  max_rings
                                });

  const int *next, *prev;
  switch (_tp_size) {
    case 8:
      next = reinterpret_cast<const int*>(tp_next_8);
      prev = reinterpret_cast<const int*>(tp_prev_8);
      break;
    case 4:
      next = reinterpret_cast<const int*>(tp_next_4);
      prev = reinterpret_cast<const int*>(tp_prev_4);
      break;
    case 2:
      return this->split_overlap_ag(A, transa, B, transb, D, bias, pre_gelu_out, workspace, grad,
                                    accumulate, use_split_accumulator, B_copy, stream_main);
    default:
      NVTE_ERROR("ROCm supports TP sizes of 2, 4, 8 only.");
  }

  const int alignment        = 256;
  const int base_slice_bytes = (comm_bytes / num_rings) & ~(alignment - 1);
  const int total_base_bytes = base_slice_bytes * num_rings;
  const int remainder_bytes  = comm_bytes - total_base_bytes;

  const size_t base_n_slice = n_chunk / num_rings;
  const size_t remainder_n  = n_chunk - (base_n_slice * num_rings);

  // Check B copy sizing
  if (B_copy.numel() > 0) {
    NVTE_CHECK(B_copy.numel() == _ubuf.numel());
    NVTE_CHECK(B_copy.element_size() == _ubuf.element_size());
  }

  const uint64_t ag_signal_base = _ag_signal_base + _tp_size;
  uint64_t signal_val;

  auto get_slice_info = [&](int ring) -> std::pair<size_t, int> {
    size_t offset = ring * base_slice_bytes;
    int size = base_slice_bytes;
    if (ring == num_rings - 1)
      size += remainder_bytes;
    return {offset, size};
  };
  
  auto get_slice_n = [&](int ring) -> size_t {
    return base_n_slice + (ring == num_rings - 1 ? remainder_n : 0);
  };
  
  auto get_chunk_id = [&](int ring, int step) {
    int owner = _tp_id;
    for (int s = 0; s < step; ++s)
      owner = prev[ring * _tp_size + owner];
    return owner;
  };

  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));

  for (int r = 0; r < num_rings; ++r) {
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(l_stream_send[r], _start_compute, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(l_stream_recv[r], _start_compute, 0));
  }

  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }

  const int total_slices = _tp_size * num_rings;
  std::vector<cudaEvent_t> slice_events(total_slices);

  for (int i = 0; i < total_slices; i++) {
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&slice_events[i], cudaEventDisableTiming));
  }
  
  auto get_event = [&](int chunk, int ring) {
    return slice_events[chunk * num_rings + ring];
  };

  for (int r = 0; r < num_rings; r++) {
    NVTE_CHECK_CUDA(cudaEventRecord(get_event(_tp_id, r), stream_main));
  }

  auto get_slice_offset = [&](int chunk, int ring) {
    auto [ring_offset, _] = get_slice_info(ring);
    return chunk * comm_bytes + ring_offset;
  };

  auto launch_slice_gemm = [&](int ring_id, int step) {
    int chunk_id = get_chunk_id(ring_id, step);
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[ring_id], 
                                        get_event(chunk_id, ring_id), 0));
    size_t n_slice = get_slice_n(ring_id);

    size_t input_b_slice_elems = n_slice * k;
    size_t output_slice_elems  = n_slice * m;
    
    size_t b_elem_offset = chunk_id * n_chunk * k;
    size_t d_elem_offset = chunk_id * n_chunk * m;
    
    for (int r = 0; r < ring_id; r++) {
      size_t prev_n  = get_slice_n(r);
      b_elem_offset += prev_n * k;
      d_elem_offset += prev_n * m;
    }

    std::vector<size_t> input_b_slice_shape =
        (transb ? std::vector<size_t>{k, n_slice} : std::vector<size_t>{n_slice, k});
    std::vector<size_t> output_slice_shape = {n_slice, m};

    auto input_b_slice = get_buffer_chunk_like(B, b_elem_offset, input_b_slice_shape);
    auto output_slice  = get_tensor_chunk(D, d_elem_offset, output_slice_shape);
    
    auto aux_slice = (do_gelu)
        ? get_tensor_chunk(pre_gelu_out, d_elem_offset, {n_slice, k})
        : TensorWrapper(nullptr, std::vector<size_t>{0}, pre_gelu_out.dtype());
    
    auto workspace_chunk = get_tensor_chunk(workspace, ring_id * workspace_size_chunk, 
                                            {workspace_size_chunk});

    nvte_cublas_gemm(A.data(), input_b_slice.data(), output_slice.data(), bias.data(),
                     aux_slice.data(), transa, transb, grad, workspace_chunk.data(), 
                     accumulate, use_split_accumulator, _math_sms, 
                     _stream_compute[ring_id]);
  };

  for (int step = 0; step < _tp_size; step++) {
    for (int r = 0; r < num_rings; r++) {
      if (step < _tp_size - 1) {
        int curr_chunk_id      = get_chunk_id(r, step);
        int next_recv_chunk_id = get_chunk_id(r, step + 1);

        int next_rank = next[r * _tp_size + _tp_id];
        int prev_rank = prev[r * _tp_size + _tp_id];

        size_t send_off = get_slice_offset(curr_chunk_id, r);

        auto [_, slice_bytes] = get_slice_info(r);

        if (step > 0) {
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(l_stream_send[r], get_event(curr_chunk_id, r), 0));
        }

        {
          int peerlocal = next_rank % _ub_comm->nvsize;
          void *flagptr = GET_SEND_PTR_BY_INDEX(peerlocal, _ub_comm, _ub_reg, r);
          void *srcptr  = reinterpret_cast<char *>(_ub_comm->mem_ptr[_ub_reg]) + send_off;
          void *dstptr  = reinterpret_cast<char *>(_ub_comm->peer_ptr[_ub_reg][peerlocal]) + send_off;

          NVTE_CHECK_CUDA(cudaMemcpyAsync(dstptr, srcptr, slice_bytes, cudaMemcpyDeviceToDevice, l_stream_send[r]));
          signal_val = ag_signal_base + step + 1;
          hipStreamWriteValue64(l_stream_send[r], flagptr, signal_val, 0);
        }

        {
          void *flagptr = GET_RECV_PTR_BY_INDEX(prev_rank, _ub_comm, _ub_reg, r);
          signal_val = ag_signal_base + step + 1;
          hipStreamWaitValue64(l_stream_recv[r], flagptr, signal_val, hipStreamWaitValueGte, 0xFFFFFFFFFFFFFFFF);
        }
        
        NVTE_CHECK_CUDA(cudaEventRecord(get_event(next_recv_chunk_id, r), l_stream_recv[r]));
      }
    }

    for (int r = 0; r < num_rings; r++) {
      launch_slice_gemm(r, step);
    }
  }

  _ag_signal_base = signal_val;

  if (B_copy.numel() > 0) {
    for (int r = 0; r < num_rings; r++) {
        int last_chunk = get_chunk_id(r, _tp_size - 1);
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(l_stream_send[0], get_event(last_chunk, r), 0));
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(B_copy.dptr(), _ubuf.dptr(), _ubuf.bytes(),
                                    cudaMemcpyDeviceToDevice, l_stream_send[0]));
  }

  _ub_comm->sms = ori_sms;

  for (auto& s : _stream_compute) {
      NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, s));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }
  
  for (int r = 0; r < num_rings; r++) {
      NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, l_stream_send[r]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));

      NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, l_stream_recv[r]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));
  }

  for (auto& ev : slice_events) {
    NVTE_CHECK_CUDA(cudaEventDestroy(ev));
  }
}  // CommOverlapP2PBase::rocm_split_overlap_ag

void CommOverlapP2PBase::rocm_split_overlap_rs(const TensorWrapper &A, bool transa,
                                               const TensorWrapper &B, bool transb, TensorWrapper &D,
                                               TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                                               TensorWrapper &workspace, bool grad, bool accumulate,
                                               bool use_split_accumulator, TensorWrapper &rs_output,
                                               cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;

  const size_t m = transa ? A.size(0) : A.size(1);
  const size_t k = transa ? A.size(1) : A.size(0);
  const size_t n_chunk = _ubufs[0].size(0);
  const int comm_bytes = _ubufs[0].bytes();

  const size_t input_chunk_size = n_chunk * k;
  const size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  const uint64_t rs_signal_base = _rs_signal_base + _tp_size;
  int64_t signal_val;

  // Catch up all streams to main
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  for (size_t i = 0; i < l_stream_send.size(); i++)
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(l_stream_send[i], _start_compute, 0));
  for (size_t i = 0; i < l_stream_recv.size(); i++)
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(l_stream_recv[i], _start_compute, 0));
  for (size_t i = 0; i < _stream_compute.size(); i++)
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));

  // Anchor event for the staggered GEMM chaining below.
  userbuffers_tiny_delay(l_stream_send[0]);
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, l_stream_send[0]));

  for (int i = 0; i < _tp_size; i++) {
    int stream_id = i % _stream_compute.size();
    int input_b_chunk_id = (_tp_id + i + 1) % _tp_size;

    auto input_b_chunk = get_tensor_chunk(B, input_b_chunk_id * input_chunk_size, {n_chunk, k});
    auto output_chunk = get_buffer_chunk_by_id(D, i);
    auto workspace_chunk = get_tensor_chunk(workspace, stream_id * workspace_size_chunk, {workspace_size_chunk});

    // Serialize GEMM i behind GEMM i-2 to enforce launch order across compute streams.
    if (i == 1) {
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[stream_id], _start_compute, 0));
    } else if (i > 1) {
      NVTE_CHECK_CUDA(
          cudaEventRecord(_start_compute, _stream_compute[(i - 2) % _stream_compute.size()]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[stream_id], _start_compute, 0));
    }

    nvte_cublas_gemm(A.data(), input_b_chunk.data(), output_chunk.data(), bias.data(),
                     pre_gelu_out.data(), transa, transb, grad, workspace_chunk.data(),
                     accumulate, use_split_accumulator, _math_sms, _stream_compute[stream_id]);

    if (i > 0) {
      // Each step uses its own send/recv stream — fully parallel since each
      // send goes to a unique destination rank (the chunk owner)
      int comm_stream_id = i - 1;
      int prev_stream_id = (i - 1) % _stream_compute.size();

      const int send_offset = comm_bytes * (i - 1);
      const int recv_offset = comm_bytes * (i - 1 + _tp_size);
      const int send_rank = (_tp_id + i) % _tp_size + _rank_round_tp;
      const int recv_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
      signal_val = rs_signal_base + i;

      // Wait for GEMM of previous chunk before sending
      NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, _stream_compute[prev_stream_id]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(l_stream_send[comm_stream_id], _start_comm, 0));

      // Send partial to chunk owner
      {
        int peerlocal = send_rank % _ub_comm->nvsize;
        void *srcptr  = reinterpret_cast<char *>(_ub_comm->mem_ptr[_ub_reg]) + send_offset;
        void *dstptr  = reinterpret_cast<char *>(_ub_comm->peer_ptr[_ub_reg][peerlocal]) + recv_offset;
        void *flagptr = GET_SEND_PTR_BY_INDEX(peerlocal, _ub_comm, _ub_reg, comm_stream_id);

        NVTE_CHECK_CUDA(cudaMemcpyAsync(dstptr, srcptr, comm_bytes,
                                        cudaMemcpyDeviceToDevice, l_stream_send[comm_stream_id]));
        hipStreamWriteValue64(l_stream_send[comm_stream_id], flagptr, signal_val, 0);
      }

      // Wait for incoming partial from chunk contributor
      {
        void *flagptr = GET_RECV_PTR_BY_INDEX(recv_rank, _ub_comm, _ub_reg, comm_stream_id);
        hipStreamWaitValue64(l_stream_recv[comm_stream_id], flagptr, signal_val,
                             hipStreamWaitValueGte, 0xFFFFFFFFFFFFFFFF);
      }
    }
  }

  _rs_signal_base = signal_val;

  // Sync all streams back to main
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }
  for (int i = 0; i < _tp_size - 1; i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, l_stream_send[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, l_stream_recv[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));
  }

  // Reduce: received partials live at _ubufs[_tp_size-1] through _ubufs[2*_tp_size-2]
  // plus local partial at _ubufs[_tp_size-1], matching single ring layout exactly
  char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].dptr());
  char *rs_output_ptr  = reinterpret_cast<char *>(rs_output.dptr());

  if (_ubuf.element_size() == 1 && rs_output.element_size() == 2) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        D.dtype(), fp8_type,
        reduce_fp8_in_bf16_out<fp8_type>(reduce_buf_ptr, rs_output_ptr, D.scale_inv(), _tp_size,
                                         _ubufs[0].numel(), stream_main););
  } else {
    reduce_bf16(reduce_buf_ptr, rs_output_ptr, _tp_size, _ubufs[0].numel(), stream_main);
  }

  _ub_comm->sms = ori_sms;
}

} // namespace transformer_engine
