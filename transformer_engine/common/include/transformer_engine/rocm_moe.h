/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

/*! \file rocm_moe.h
 *  \brief Fused expert-parallel MoE training on ROCm (KOSMOS, gfx950): the routed experts of an
 *         MoE layer (dispatch, GEMM1 + GLU, GEMM2, combine and the weighted fold; in the backward
 *         also both weight gradients) as one kernel launch per pass, EP = 8 ranks in one node,
 *         bf16.
 *
 *  An instance holds, per configuration, KOSMOS's settings, a symmetric buffer every rank writes
 *  into and its scratch, allocated once at creation. What a backward reads from its forward (the
 *  saved state) lives in caller-owned buffers (NVTERocmMoeSaved), so any number of forwards may be
 *  in flight, backwards may come in any order, and a forward without a backward leaves nothing
 *  behind in the instance.
 *
 *  Calls marked collective must be made by every rank of the EP group at the same point. Every
 *  rank makes the same forward and backward calls in the same order on one stream: the kernels
 *  wait on their peers.
 *
 *  Versioned structs begin with struct_size: the bytes the caller's layout has, or 0 for the
 *  base layout. Fields past them read as zero; outputs are written only up to them.
 */

#ifndef TRANSFORMER_ENGINE_ROCM_MOE_H_
#define TRANSFORMER_ENGINE_ROCM_MOE_H_

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Ranks in the EP group. */
#define NVTE_ROCM_MOE_EP_SIZE 8

/*! \brief GLU of the experts' [gate; up] GEMM1 output g, u. */
enum NVTERocmMoeActivation {
  kNVTERocmMoeSwiGLU = 0,        /*!< silu(g) u */
  kNVTERocmMoeSwiGLUClamp10 = 1, /*!< silu(min(g, 10)) clamp(u, -10, 10) */
};

/*! \brief The most a training forward keeps for its backward. */
enum NVTERocmMoeSaveTier {
  kNVTERocmMoeSaveNone = 0, /*!< the backward sends x to the experts again and recomputes */
  kNVTERocmMoeSaveXp = 1,   /*!< the dispatched rows Xp; the backward recomputes GEMM1 */
  kNVTERocmMoeSaveAll = 2,  /*!< Xp, GEMM1's output G and the activation A */
};

/*! \brief Status bits (nvte_rocm_moe_instance_status). */
enum NVTERocmMoeStatus {
  kNVTERocmMoeStatusBadExpert = 1,   /*!< an expert id outside [0, 8 E_loc); -1 included */
  kNVTERocmMoeStatusDuplicate = 2,   /*!< a token routed twice to one expert */
  kNVTERocmMoeStatusRounds = 8,      /*!< a call needed more rows than the capacity: in rounds */
  kNVTERocmMoeStatusKeptNothing = 32 /*!< a forward kept nothing, less than its tier allowed */
};

/*! \brief Configuration, identical on every rank. */
typedef struct {
  size_t struct_size;
  int hidden_size;       /*!< H, a multiple of 256 */
  int ffn_hidden_size;   /*!< I per expert, a multiple of 256 */
  int num_local_experts; /*!< E_loc, experts per rank; global expert e is on rank e / E_loc */
  int topk;              /*!< k in {1, 2, 4, 6, 8, 10} */
  int num_tokens;        /*!< T, tokens per rank and call */
  int capacity_rows;     /*!< receive rows per rank the instance is sized for (> 0) */
  int wgcap;             /*!< workgroups per launch, [9, 256] */
  int activation;        /*!< NVTERocmMoeActivation */
} NVTERocmMoeConfig;

/*! \brief The EP group's host all-gather, used only by nvte_rocm_moe_instance_create.
 *
 *  allgather gathers bytes from every rank into recv (nranks * bytes, in rank order) and returns 0
 *  on success; it is called on the calling thread.
 */
typedef struct {
  size_t struct_size;
  void *ctx;
  int rank;
  int nranks; /*!< NVTE_ROCM_MOE_EP_SIZE */
  int (*allgather)(void *ctx, const void *send, void *recv, size_t bytes);
} NVTERocmMoeComm;

/*! \brief A symmetric buffer the caller allocated and mapped on every rank (e.g. PyTorch
 *         symmetric memory), for the instance to use instead of allocating its own.
 */
typedef struct {
  size_t struct_size;
  void *local;        /*!< this rank's buffer */
  void *const *peers; /*!< device array, each rank's buffer as this rank maps it; [rank] is local */
  size_t bytes;       /*!< at least nvte_rocm_moe_symmetric_buffer_bytes */
} NVTERocmMoeSymmetricBuffer;

/*! \brief One forward's saved state, caller-owned: kNVTEByte buffers of the sizes
 *         nvte_rocm_moe_saved_bytes gives for rows, 256-byte aligned; tier and rows equal on every
 *         rank. xp is absent (null) below kNVTERocmMoeSaveXp, g and a below kNVTERocmMoeSaveAll.
 */
typedef struct {
  size_t struct_size;
  int tier;         /*!< NVTERocmMoeSaveTier */
  int rows;         /*!< the rows xp, g and a are sized for, in [1, the capacity rows] */
  NVTETensor xp;    /*!< dispatched rows */
  NVTETensor g;     /*!< GEMM1's output */
  NVTETensor a;     /*!< activation */
  NVTETensor y;     /*!< GEMM2's output per routed token; always */
  NVTETensor route; /*!< the route record: layout, receive slots, the tier kept, status; always */
} NVTERocmMoeSaved;

/*! \brief Bytes of each saved buffer. */
typedef struct {
  size_t struct_size;
  size_t xp;
  size_t g;
  size_t a;
  size_t y;
  size_t route;
} NVTERocmMoeSavedBytes;

/*! \brief The instance's derived launch settings (KOSMOS's KosmosMoeInfo), for diagnostics. */
typedef struct {
  size_t struct_size;
  int Mpad;
  int Mpad_max;
  int fwd_ncomm;
  int fwd_fcomm;
  int fwd_burst;
  int nt1;
  int nt2;
  int bwd_cncomm_clamped;
  int bwd_grid;
  int bwd_dncomm_all[NVTE_ROCM_MOE_EP_SIZE];
  int bwd_cncomm_all[NVTE_ROCM_MOE_EP_SIZE];
  double bwd_dpush_gb;
  double bwd_lega_tflop;
  double bwd_cpush_gb;
  double bwd_wg2_tiles;
  double bwd_wg2_meank;
  int bwd_wg2run;
  int bwd_wg1run;
  int arena_rows;
} NVTERocmMoeInfo;

typedef struct NVTERocmMoeInstance_ *NVTERocmMoeInstance;

/*! \brief Whether Transformer Engine was built with the fused MoE (ROCm, gfx950, USE_KOSMOS_MOE).
 */
int nvte_rocm_moe_is_built(void);

/*! \brief Whether the configuration is supported (host only). */
int nvte_rocm_moe_config_valid(const NVTERocmMoeConfig *cfg);

/*! \brief Bytes of the instance's symmetric buffer. */
size_t nvte_rocm_moe_instance_bytes(const NVTERocmMoeConfig *cfg);

/*! \brief Bytes to allocate for an NVTERocmMoeSymmetricBuffer: the instance's, the peer flags
 *         carved from the buffer's end and 256 bytes of slack, rounded up to 4096.
 */
size_t nvte_rocm_moe_symmetric_buffer_bytes(const NVTERocmMoeConfig *cfg);

/*! \brief Create an instance (collective, host-blocking).
 *
 *  Does all setup a first call would, so that later calls enqueue one kernel each and can be
 *  captured in a CUDA graph. cfg and comm are copied; comm->ctx and the buffer must outlive the
 *  instance.
 *
 *  \param[in] cfg     Configuration.
 *  \param[in] comm    The EP group's all-gather.
 *  \param[in] buffer  Symmetric buffer to use, or null: the instance allocates its own.
 */
NVTERocmMoeInstance nvte_rocm_moe_instance_create(const NVTERocmMoeConfig *cfg,
                                                  const NVTERocmMoeComm *comm,
                                                  const NVTERocmMoeSymmetricBuffer *buffer);

/*! \brief Destroy an instance (host-blocking). Its symmetric buffer must no longer be in use by any
 *         rank.
 */
void nvte_rocm_moe_instance_destroy(NVTERocmMoeInstance inst);

/*! \brief The instance's last error message (empty if none). */
const char *nvte_rocm_moe_instance_error(NVTERocmMoeInstance inst);

/*! \brief The instance's derived settings. */
void nvte_rocm_moe_instance_info(NVTERocmMoeInstance inst, NVTERocmMoeInfo *info);

/*! \brief OR of the status bits of every call since the last query, which it clears
 *         (host-blocking).
 */
int nvte_rocm_moe_instance_status(NVTERocmMoeInstance inst);

/*! \brief Bytes of each saved buffer for rows saved rows (1 <= rows <= the capacity rows rounded up
 *         to 256).
 */
void nvte_rocm_moe_saved_bytes(const NVTERocmMoeConfig *cfg, int rows,
                               NVTERocmMoeSavedBytes *bytes);

/*! \brief Forward: one launch on stream.
 *
 *  The forward keeps saved->tier if its rows fit saved->rows and it runs in one pass, else nothing
 *  (status bit 32 if it asked for more), and records which in saved->route. x, w1, w2, topk_idx
 *  and topk_weights must be unchanged at the backward.
 *
 *  \param[in]     inst          Instance.
 *  \param[in]     x             [T, H] bf16 tokens.
 *  \param[in]     w1            [E_loc, 2I, H] bf16, rows in blocks of 256: 128 gate rows, then the
 *                               matching 128 up rows.
 *  \param[in]     w2            [E_loc, H, I] bf16.
 *  \param[in]     topk_idx      [T, k] int32 global expert ids; -1 drops a route.
 *  \param[in]     topk_weights  [T, k] fp32 routing weights.
 *  \param[out]    out           [T, H] bf16.
 *  \param[in,out] saved         Saved state to fill, or null: nothing is kept (inference).
 *  \param[in]     stream        Stream.
 */
void nvte_rocm_moe_train_forward(NVTERocmMoeInstance inst, const NVTETensor x, const NVTETensor w1,
                                 const NVTETensor w2, const NVTETensor topk_idx,
                                 const NVTETensor topk_weights, NVTETensor out,
                                 const NVTERocmMoeSaved *saved, cudaStream_t stream);

/*! \brief Backward of the forward that filled saved (the buffers may have moved): one launch on
 *         stream.
 *
 *  \param[in]     inst          Instance.
 *  \param[in]     dout          [T, H] bf16 gradient of the forward's output.
 *  \param[in]     x             The forward's x.
 *  \param[in]     w1            The forward's w1.
 *  \param[in]     w2            The forward's w2.
 *  \param[in]     topk_idx      The forward's topk_idx.
 *  \param[in]     topk_weights  The forward's topk_weights.
 *  \param[in]     saved         The forward's saved state.
 *  \param[out]    dx            [T, H] bf16.
 *  \param[out]    dprobs        [T, 8 E_loc] fp32 gradient of the routing weights, zero at the
 *                               experts a token did not use.
 *  \param[in,out] dw1           Shape of w1: bf16, overwritten (accumulate 0); fp32, added into
 *                               (1); bf16, added into (2).
 *  \param[in,out] dw2           Shape of w2, as dw1.
 *  \param[in]     accumulate    0, 1 or 2, as above.
 *  \param[in]     stream        Stream.
 */
void nvte_rocm_moe_train_backward(NVTERocmMoeInstance inst, const NVTETensor dout,
                                  const NVTETensor x, const NVTETensor w1, const NVTETensor w2,
                                  const NVTETensor topk_idx, const NVTETensor topk_weights,
                                  const NVTERocmMoeSaved *saved, NVTETensor dx, NVTETensor dprobs,
                                  NVTETensor dw1, NVTETensor dw2, int accumulate,
                                  cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_ROCM_MOE_H_
