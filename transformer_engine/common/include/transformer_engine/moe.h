/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

/*! \file moe.h
 *  \brief Fused expert-parallel MoE layer (KOSMOS): dispatch, grouped GEMMs, GLU and combine in one launch for the
 *         forward and one for the backward, over symmetric peer memory. ROCm gfx950, EP = 8 ranks in one node.
 *
 *  A workspace holds what one call needs while it runs plus the training saved-state pool, shared by all its layers.
 *  A layer handle is one in-flight forward: a backward reads what the handle's most recent saving forward left, so a
 *  caller with several forwards of the same module in flight (pipeline parallelism) uses one handle per forward.
 *
 *  Calls marked collective must be issued by every EP rank in the same order; per-step calls are stream-ordered and
 *  must also be issued in the same order on every rank. A failed per-step call leaves the workspace in an undefined
 *  state: quiesce every rank and call nvte_moe_layer_reset on every rank.
 */

#ifndef TRANSFORMER_ENGINE_MOE_H_
#define TRANSFORMER_ENGINE_MOE_H_

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Number of EP ranks the fused MoE supports. */
#define NVTE_MOE_EP_SIZE 8

/*! \brief GLU of the experts' [gate; up] fc1 output. */
enum NVTEMoeActivation {
  kNVTEMoeSwiGLU = 0,        /*!< silu(g) * u */
  kNVTEMoeSwiGLUClamp10 = 1, /*!< silu(min(g, 10)) * clamp(u, -10, 10) (clamped_swiglu_v2, limit 10) */
};

/*! \brief Score function of the fused router (Megatron semantics). */
enum NVTEMoeRouterScore {
  kNVTEMoeRouterSoftmax = 0,      /*!< softmax over the top k logits */
  kNVTEMoeRouterSigmoid = 1,      /*!< sigmoid, top k of score + expert bias, normalized */
  kNVTEMoeRouterSqrtSoftplus = 2, /*!< sqrt(softplus), as sigmoid */
  kNVTEMoeRouterSoftmaxPre = 3,   /*!< softmax over all experts, then top k, not renormalized */
};

/*! \brief Host collectives over the EP group, and the device-memory allocator.
 *
 *  allgather gathers bytes from every rank into recv (nranks * bytes, rank order) and returns 0 on success. It is
 *  only called from collective calls, on the calling thread. ctx must stay valid until the workspace is destroyed.
 *  alloc / free (optional, both or neither): every byte of device memory the workspace and its layers allocate.
 *  With ipc set, the buffer is one the EP ranks write into each other, shared by TE with an IPC handle of the
 *  allocation holding it and its offset in it (exchanged through allgather): it must work with
 *  hipIpcGetMemHandle, i.e. plain hipMalloc memory (not expandable or VMM segments). alloc returns 0 and a
 *  256-byte aligned pointer; free gets it back. Null: hipMalloc / hipFree.
 */
typedef struct {
  size_t struct_size;
  void *ctx;
  int rank;
  int nranks;
  int (*allgather)(void *ctx, const void *send, void *recv, size_t bytes);
  int (*alloc)(void *ctx, size_t bytes, int ipc, void **ptr);
  void (*free)(void *ctx, void *ptr);
} NVTEMoeComm;

/*! \brief Fused-router configuration. */
typedef struct {
  int score;         /*!< NVTEMoeRouterScore */
  int groups;        /*!< expert groups for group-limited routing (0: none) */
  int group_topk;    /*!< groups selected per token */
  int aux_groups;    /*!< 0: no aux loss; 1: over the batch; n > 1: per sequence, token t in sequence t % n */
  float aux_coeff;   /*!< aux-loss coefficient */
  float scale;       /*!< routing-weight scale (0 means 1) */
} NVTEMoeRouterConfig;

/*! \brief Layer configuration, identical on every EP rank. Begins with struct_size (0: the base layout). */
typedef struct {
  size_t struct_size;
  int hidden_size;       /*!< H, multiple of 256 (MXFP8 training: 1024) */
  int ffn_hidden_size;   /*!< I per expert, multiple of 256 */
  int num_local_experts; /*!< E_loc; global experts NVTE_MOE_EP_SIZE * E_loc */
  int topk;              /*!< k in {1, 2, 4, 6, 8, 10} */
  int num_tokens;        /*!< T tokens per rank (fixed per workspace) */
  int wgcap;             /*!< launch width in workgroups, [9, 256] */
  int capacity_rows;     /*!< receive rows per rank to reserve (> 0) */
  size_t pool_bytes;     /*!< training saved-state pool bytes per rank (0: one call's block) */
  int training;          /*!< nonzero: backward supported */
  int activation;        /*!< NVTEMoeActivation */
  int mxfp8;             /*!< MXFP8 GEMM operands */
  int fp8_combine;       /*!< MXFP8 training: MXFP8 combines */
  int probs_on_input;    /*!< k = 1: routing weight applied to the input, unweighted fold */
  int shared_expert;     /*!< a shared expert runs in the same launches */
  int router;            /*!< nonzero: the routing is computed by the fused router (router_config) */
  NVTEMoeRouterConfig router_config;
} NVTEMoeConfig;

#define NVTE_MOE_CONFIG_INIT {sizeof(NVTEMoeConfig)}
#define NVTE_MOE_COMM_INIT {sizeof(NVTEMoeComm)}

typedef struct NVTEMoeWorkspace_ *NVTEMoeWorkspace;
typedef struct NVTEMoeLayer_ *NVTEMoeLayer;

/*! \brief Forward operands. Absent tensors are null NVTETensors or tensors without data.
 *
 *  bf16 unless noted; weights are bf16 or MXFP8 (row-wise data and unswizzled scale_inv), as the config says.
 *  - x [T, H]: bf16 (MXFP8 inference: MXFP8 row-wise). Read again by some backwards: keep it unchanged until then.
 *  - topk_idx [T, k] int32 global expert ids (-1: dropped), topk_weights [T, k] fp32: absent with the fused router.
 *  - router_weight [8 E_loc, H]; expert_bias, router_bias [8 E_loc] fp32 (optional); seed [1] int32 tensor
 *    holding a uint32 (forced routing only).
 *  - fc1_weight [E_loc, 2I, H] (rows in 256-row blocks: 128 gate rows, then the matching 128 up rows);
 *    fc2_weight [E_loc, H, I].
 *  - shared_fc1_weight [2I, H] ([gate; up] rows), shared_fc2_weight [H, I], shared_gate [T] fp32 (optional),
 *    shared_pre_act [T, 2I] (training output, kept until the backward; required by a saving forward with the shared
 *    expert, and always with MXFP8).
 *  - out [T, H]; aux_loss [1] fp32 (fused router with an aux loss) and tokens_per_expert [8 E_loc] int32 (fused
 *    router).
 */
typedef struct {
  size_t struct_size;
  NVTETensor x;
  NVTETensor topk_idx;
  NVTETensor topk_weights;
  NVTETensor router_weight;
  NVTETensor expert_bias;
  NVTETensor router_bias;
  NVTETensor seed;
  NVTETensor fc1_weight;
  NVTETensor fc2_weight;
  NVTETensor shared_fc1_weight;
  NVTETensor shared_fc2_weight;
  NVTETensor shared_gate;
  NVTETensor shared_pre_act;
  NVTETensor out;
  NVTETensor aux_loss;
  NVTETensor tokens_per_expert;
  int save_for_backward; /*!< training: keep the saved state for nvte_moe_backward */
} NVTEMoeForwardArgs;

/*! \brief Backward operands (the forward's, plus gradients). Gradients are overwritten unless noted.
 *
 *  Weights: bf16, or MXFP8 with column-wise data and scales (and the row-wise ones the forward read).
 *  - dprobs [T, 8 E_loc] fp32: gradient of the routing weights at each token's experts, zero elsewhere (caller
 *    routing only).
 *  - aux_loss_scale [1] fp32 (fused router): the aux-loss gradient scale, read when the kernel runs.
 *  - fc1_wgrad [E_loc, 2I, H], fc2_wgrad [E_loc, H, I], router_wgrad [8 E_loc, H], shared_fc1_wgrad [2I, H],
 *    shared_fc2_wgrad [H, I] and router_bias_grad [8 E_loc] (optional), per wgrad_accumulate (main_grad
 *    accumulation, inside the one backward launch): 0 overwritten, bf16 (router_bias_grad fp32); 1 added into,
 *    fp32; 2 added into, bf16 (each element rounded once). Weight gradients 16-byte aligned.
 *  - shared_gate_grad [T] fp32 (optional), overwritten.
 */
typedef struct {
  size_t struct_size;
  NVTETensor dout;
  NVTETensor x;
  NVTETensor router_weight;
  NVTETensor aux_loss_scale;
  NVTETensor fc1_weight;
  NVTETensor fc2_weight;
  NVTETensor shared_fc1_weight;
  NVTETensor shared_fc2_weight;
  NVTETensor shared_gate;
  NVTETensor shared_pre_act;
  NVTETensor dx;
  NVTETensor dprobs;
  NVTETensor fc1_wgrad;
  NVTETensor fc2_wgrad;
  NVTETensor router_wgrad;
  NVTETensor router_bias_grad;
  NVTETensor shared_fc1_wgrad;
  NVTETensor shared_fc2_wgrad;
  NVTETensor shared_gate_grad;
  int wgrad_accumulate;
} NVTEMoeBackwardArgs;

/*! \brief Statistics since the previous query or capacity change (see nvte_moe_capacity_query). */
typedef struct {
  int peak;       /*!< most receive rows a rank needed in any call */
  int rows;       /*!< current capacity */
  int max_rows;   /*!< largest capacity one call can need */
  int nlayers;    /*!< open layer handles */
  int rounds;     /*!< handles that ran a call in rounds */
  int cap_rounds; /*!< of those, how many needed more rows than the capacity */
} NVTEMoeCapacityStat;

/*! \brief Layer status bits (nvte_moe_layer_status). */
enum NVTEMoeStatus {
  kNVTEMoeStatusBadExpert = 1,     /*!< an expert id outside [0, 8 E_loc) (also every -1 padding row) */
  kNVTEMoeStatusDuplicate = 2,     /*!< an expert repeated in a token's top k */
  kNVTEMoeStatusRounds = 8,        /*!< a call ran in rounds */
  kNVTEMoeStatusKeptXpOnly = 16,   /*!< bf16 training: a forward kept only Xp (backward recomputes GEMM1) */
  kNVTEMoeStatusKeptNothing = 32,  /*!< training: a forward kept nothing (backward re-sends X) */
};

/*! \brief Whether the fused MoE is built and the current device supports it. */
int nvte_moe_is_supported(void);

/*! \brief Host-only configuration check; returns 0 and writes the reason to why (if not null) when refused. */
int nvte_moe_config_valid(const NVTEMoeConfig *cfg, char *why, size_t why_len);

/*! \brief Create a workspace (local). cfg and comm are copied; comm->ctx must outlive the workspace. */
NVTEMoeWorkspace nvte_moe_workspace_create(const NVTEMoeConfig *cfg, const NVTEMoeComm *comm);

/*! \brief Destroy a workspace (collective), after every layer on it. */
void nvte_moe_workspace_destroy(NVTEMoeWorkspace ws);

/*! \brief Create a layer handle on a workspace (local; its first forward or prepare is collective). */
NVTEMoeLayer nvte_moe_layer_create(NVTEMoeWorkspace ws);

/*! \brief Destroy a layer handle (collective). Frees its saved state, if any. */
void nvte_moe_layer_destroy(NVTEMoeLayer layer);

/*! \brief Map the layer's buffers now (collective, host-blocking), so that its calls can be graph-captured. */
void nvte_moe_layer_prepare(NVTEMoeLayer layer);

/*! \brief Re-zero the layer's and the workspace's state after a failed call (collective, host-blocking). */
void nvte_moe_layer_reset(NVTEMoeLayer layer);

/*! \brief OR of the status bits of the layer's calls since the last query (local, host-blocking). */
int nvte_moe_layer_status(NVTEMoeLayer layer);

/*! \brief Saved-state ownership of a layer handle (nvte_moe_layer_release_state). */
enum NVTEMoeReleaseState {
  kNVTEMoeReleaseIdle = 0,      /*!< nothing stored */
  kNVTEMoeReleaseStored = 1,    /*!< a saving forward awaits its backward */
  kNVTEMoeReleasePending = 2,   /*!< released on this rank, not yet on every rank */
  kNVTEMoeReleaseReclaimed = 3, /*!< released on every rank: the handle may run a new forward */
};

/*! \brief The handle's saving forward will get no backward (local, host-only, thread-safe; e.g. from a finalizer).
 *
 *  Its pool block is freed once every rank released it: the ranks agree inside their next forwards (any layer of the
 *  workspace), and the handle becomes kNVTEMoeReleaseReclaimed a few forwards later, at the same point of the call
 *  sequence on every rank. Off for calls captured into a graph.
 */
void nvte_moe_layer_release(NVTEMoeLayer layer);

/*! \brief The handle's NVTEMoeReleaseState (local, host-only). */
int nvte_moe_layer_release_state(NVTEMoeLayer layer);

/*! \brief Read every launched forward's agreement (host-blocking; call it alike on every rank). */
void nvte_moe_release_sync(NVTEMoeWorkspace ws);

/*! \brief Forward: one launch on stream. */
void nvte_moe_forward(NVTEMoeLayer layer, const NVTEMoeForwardArgs *args, cudaStream_t stream);

/*! \brief Backward of the layer's most recent saving forward: one launch on stream. */
void nvte_moe_backward(NVTEMoeLayer layer, const NVTEMoeBackwardArgs *args, cudaStream_t stream);

/*! \brief Resize the saved-state pool (collective, host-blocking, between steps).
 *  \return 0 resized; 1 refused (a saved forward awaits its backward); 2 refused (inside a stream capture).
 */
int nvte_moe_pool_resize(NVTEMoeWorkspace ws, size_t bytes);

/*! \brief Pool bytes per rank, one call's block, and the Xp bytes reserved per layer still to save. */
void nvte_moe_pool_info(NVTEMoeWorkspace ws, size_t *bytes, size_t *head_bytes, size_t *reserve_bytes);

/*! \brief Set the receive capacity in rows (collective, host-blocking, between steps).
 *  \return as nvte_moe_pool_resize; 3: an allocation failed on some rank and the old capacity is kept.
 */
int nvte_moe_set_capacity(NVTEMoeWorkspace ws, int rows);

/*! \brief Read and clear the capacity statistics of every open layer (collective, host-blocking). */
void nvte_moe_capacity_query(NVTEMoeWorkspace ws, NVTEMoeCapacityStat *stat);

/*! \brief Device bytes nvte_moe_set_capacity would allocate at rows. */
size_t nvte_moe_capacity_bytes(NVTEMoeWorkspace ws, int rows);

/*! \brief Bytes of device memory the fused MoE currently holds from hipMalloc, i.e. allocated outside every
 *  workspace's NVTEMoeComm allocator (0 when every workspace has one). */
size_t nvte_moe_untracked_bytes(void);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_MOE_H_
