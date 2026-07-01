// Check what mha_bwd_workspace_size returns vs the old formula for the failing config.
// Config: b=2, s_q=1024, s_kv=2048, h_q=12, h_kv=6, d_qk=128, d_v=64, bf16, group mode, deterministic.
//
// ---------------------------------------------------------------------------
// BUILD & RUN  (inside the TE container, e.g. `docker exec <container> bash -lc '...'`,
// with the repo mounted at /workspace and TE already built via
// `pip install -e . --no-build-isolation` so transformer_engine/lib/ and
// build/aiter-prebuilts/ exist).
//
//   cd /workspace
//   PRE=$(ls -d build/aiter-prebuilts/rocm-7_aiter-*/ | head -1)   # QoLA public headers
//   CK=transformer_engine/lib/ck_jit                              # mha_bwd.h, fmha_bwd.hpp, ck_tile
//   hipcc -std=c++17 -w -DENABLE_CK=1 repro/check_ws_size.cpp -o /tmp/check_ws_size \
//     -I "${PRE}include" \
//     -I $CK/csrc/include \
//     -I $CK/3rdparty/composable_kernel/include \
//     -I $CK/3rdparty/composable_kernel/example/ck_tile/01_fmha \
//     -L /workspace/transformer_engine/lib -Wl,-rpath,/workspace/transformer_engine/lib \
//     -l:te_libmha_bwd.so
//   /tmp/check_ws_size
//
// Notes:
//   * -DENABLE_CK=1 is REQUIRED: mha_bwd_workspace_size() and the fmha_bwd.hpp
//     include are gated behind `#if ENABLE_CK` in qola_mha_bwd.h.
//   * Link the .so with `-l:te_libmha_bwd.so` (NOT as a positional argument):
//     hipcc injects `-x hip`, which would otherwise try to *compile* the .so as
//     HIP source ("expected unqualified-id" at te_libmha_bwd.so:1:1).
//   * The -Wl,-rpath keeps the .so resolvable at run time without LD_LIBRARY_PATH.
//   * First run JIT-compiles the CK kernel (slow once; cached under ~/.cache/te_ck_jit).
//
// Expected output on the affected branch (aiter e95f7d4e):
//   deterministic      : 138416128 bytes = 132.0 MiB   (nsplits=11, kN0=192 tile)
//   non-deterministic  :  12587008 bytes =  12.0 MiB   (nsplits=1)
//   old formula        : 201326592 bytes = 192.0 MiB   (nsplits=16, kN0=128)
//   => WARNING: underallocation (132 < 192).
// ---------------------------------------------------------------------------

#include "mha_bwd.h"        // fmha_bwd_traits, mha_bwd_workspace_size
#include "qola_common.h"
#include "qola_mha_bwd.h"

#include <cstdio>
#include <cmath>

int main() {
    // Matches make_bwd_traits for this config in group mode (bshd_to_thd):
    // seqlen_q = max_tokens_q = 2048, seqlen_k = max_tokens_kv = 4096
    const int b            = 2;
    const int s_q          = 1024;   // max_seqlen_q (per-sequence)
    const int s_kv         = 2048;   // max_seqlen_kv (per-sequence)
    const int max_tokens_q = 2048;   // total tokens for q (b * s_q)
    const int max_tokens_kv= 4096;   // total tokens for kv (b * s_kv)
    const int h_q          = 12;
    const int h_kv         = 6;
    const int d_qk         = 128;
    const int d_v          = 64;

    // Build fmha_bwd_traits matching make_bwd_traits() for group mode + deterministic
    ::fmha_bwd_traits traits{
        /* seqlen_q      */ max_tokens_q,   // group mode: max_tokens_q
        /* seqlen_k      */ max_tokens_kv,  // group mode: max_tokens_kv
        /* batch         */ b,
        /* max_seqlen_q  */ s_q,
        /* max_seqlen_k  */ s_kv,
        /* hdim_q        */ d_qk,
        /* hdim_v        */ d_v,
        /* nhead_q       */ h_q,
        /* nhead_k       */ h_kv,
        /* data_type     */ "bf16",
        /* is_group_mode */ true,
        /* mask_type     */ mask_enum::no_mask,
        /* bias_type     */ bias_enum::no_bias,
        /* has_dbias     */ false,
        /* has_dropout   */ false,
        /* is_store_randval */ false,
        /* is_deterministic */ true,
    };

    size_t ws = qola::te::mha_bwd_workspace_size(traits);
    printf("mha_bwd_workspace_size (group, deterministic): %zu bytes = %.1f MiB\n",
           ws, ws / 1024.0 / 1024.0);

    // Also test non-deterministic
    traits.is_deterministic = false;
    size_t ws_nd = qola::te::mha_bwd_workspace_size(traits);
    printf("mha_bwd_workspace_size (group, non-deterministic): %zu bytes = %.1f MiB\n",
           ws_nd, ws_nd / 1024.0 / 1024.0);

    // OLD formula (for comparison):
    int kN0 = (d_qk <= 128) ? 128 : 64;
    int nsplits = (int)ceil((double)s_kv / kN0);  // used s_kv, not max_tokens_kv
    size_t old_bytes = (size_t)nsplits * h_q * max_tokens_q * d_qk * sizeof(float);
    printf("\nOld formula: nsplits=%d (s_kv=%d/kN0=%d), dq_acc=%zu bytes = %.1f MiB\n",
           nsplits, s_kv, kN0, old_bytes, old_bytes / 1024.0 / 1024.0);

    if (ws < old_bytes) {
        printf("\nWARNING: new workspace (%zu) < old dq_acc (%zu) — potential underallocation!\n", ws, old_bytes);
    } else {
        printf("\nnew workspace >= old dq_acc — size OK\n");
    }
    return 0;
}
