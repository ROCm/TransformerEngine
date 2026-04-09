ENABLE_FA3 = False
#ENABLE_TRITON = True
ENABLE_TRITON = False

import argparse
import csv
import gc
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import transformer_engine.jax.attention as te
from tqdm import tqdm

from utils import gen_data, jax_attention as _jax_attention, segment_ids_to_cu_seqlens

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# TE CI docker image does not support it
#jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

# fmt: off
DTYPE_MAP = {"float16": jnp.float16, "float32": jnp.float32, "bfloat16": jnp.bfloat16, "int32": jnp.int32, "int64": jnp.int64}
DTYPE_MAP_INV = {v: k for k, v in DTYPE_MAP.items()}
# fmt: on

# ===== Utils =====
# fmt: off
def bshd_to_thd(tensor):
    return tensor.reshape((-1, tensor.shape[2], tensor.shape[3]))


def thd_to_bshd(tensor, bsize):
    return tensor.reshape((bsize, -1, tensor.shape[1], tensor.shape[2]))


def make_noop_forward_builder(prof_fn):
    def builder_fn(do, q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout, nr_segments):
        res_fn = jax.jit(prof_fn, static_argnames=("sm_scale", "causal", "window_size", "layout", "nr_segments"))
        return res_fn, (q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout, nr_segments)

    return builder_fn


def make_noop_backward_builder(fwd_fn, bwd_fn):
    def builder_fn(do, q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout, nr_segments):
        out, softmax_lse = fwd_fn(q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout)[:2]
        res_fn = jax.jit(bwd_fn, static_argnames=("sm_scale", "causal", "window_size", "layout", "nr_segments"))
        return res_fn, (do, q, k, v, segment_ids_q, segment_ids_kv, out, softmax_lse, sm_scale, causal, window_size, layout, nr_segments)

    return builder_fn


def make_vjp_backward_builder(fwd_fn):
    def builder_fn(do, q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout, nr_segments):
        _, pullback = jax.vjp(lambda q, k, v: fwd_fn(q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout, nr_segments), q, k, v)
        return jax.jit(pullback), (do,)

    return builder_fn
# fmt: on


# ===== Implementations adapted for profiling =====
def jax_attention(q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout, nr_segments):
    window = (-1, -1) if (window_size is None or window_size == -1) else (window_size, 0)
    return _jax_attention(q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window)


jax_attn_forward_builder = make_noop_forward_builder(jax_attention)
jax_attn_backward_builder = make_vjp_backward_builder(jax_attention)


def te_fa_forward(q, k, v, segment_ids_q, segment_ids_kv, sm_scale, causal, window_size, layout, nr_segments):
    if layout == "thd":
        qkv_layout = te.QKVLayout.THD_THD_THD
    elif layout == "bshd":
        qkv_layout = te.QKVLayout.BSHD_BSHD_BSHD
    else:
        raise ValueError("Unsupported layout")
    window_size = None if (window_size is None or window_size == -1) else (window_size, -1)
    is_thd = layout == "thd"
    attn_out = te.fused_attn(
        qkv=(q, k, v),
        bias=None,
        sequence_descriptor=te.SequenceDescriptor.from_segment_ids_and_pos(
            (segment_ids_q + 1, segment_ids_kv + 1),
            None,
            is_thd=is_thd,
            is_segment_ids_reordered=False,
        ),
        seed=None,
        attn_bias_type=te.AttnBiasType.NO_BIAS,
        attn_mask_type=(
            te.AttnMaskType.PADDING_CAUSAL_MASK if causal else te.AttnMaskType.PADDING_MASK
        ),
        qkv_layout=qkv_layout,
        softmax_type=te.AttnSoftmaxType.VANILLA_SOFTMAX,
        scaling_factor=sm_scale,
        dropout_probability=0.0,
        is_training=True,
        max_segments_per_seq=nr_segments,
        window_size=window_size,
    )
    return attn_out


te_fa_forward_builder = make_noop_forward_builder(te_fa_forward)
te_fa_backward_builder = make_vjp_backward_builder(te_fa_forward)


# fmt: off
if ENABLE_TRITON:
    from fax_kernels.triton_flash_attention.bwd_prefill import (
        attention_prefill_backward_triton_impl,
    )
    from fax_kernels.triton_flash_attention.fwd_prefill import (
        attention_prefill_forward_triton_impl,
    )

    def attention_forward_triton_impl(q, k, v, segment_ids, sm_scale, causal, window_size, layout):
        if layout == "thd":
            q, k, v = bshd_to_thd(q), bshd_to_thd(k), bshd_to_thd(v)
        cu_seqlens = segment_ids_to_cu_seqlens(segment_ids, 256)
        return attention_prefill_forward_triton_impl(
            q, k, v, sm_scale, None, causal, None, layout, cu_seqlens, cu_seqlens, segment_ids.shape[1],
            segment_ids.shape[1], None, None, 0.0, None, None, False, False, None, None, None, None
        )

    def attention_backward_triton_impl(do, q, k, v, segment_ids, out, softmax_lse, sm_scale, causal, window_size, layout):
        if layout == "thd":
            do, q, k, v = bshd_to_thd(do), bshd_to_thd(q), bshd_to_thd(k), bshd_to_thd(v)
        cu_seqlens = segment_ids_to_cu_seqlens(segment_ids, 256)
        return attention_prefill_backward_triton_impl(
            do, q, k, v, out, softmax_lse, sm_scale, None, causal, layout, cu_seqlens, cu_seqlens, segment_ids.shape[1],
            segment_ids.shape[1], 0.0, None, None, False, True, None, None, None, None
        )

    triton_fa_forward_builder = make_noop_forward_builder(attention_forward_triton_impl)
    triton_fa_backward_builder = make_noop_backward_builder(attention_forward_triton_impl, attention_backward_triton_impl)


if ENABLE_FA3:
    from fax_kernels import flash_mha_3 as _flash_mha_3

    def flash_mha_3(q, k, v, segment_ids, sm_scale, causal, window_size, layout):
        window_size = (-1, -1) if (window_size is None or window_size == -1) else (window_size, 0)
        return _flash_mha_3(q, k, v, segment_ids, sm_scale, causal, window_size, 5000)

    flash_mha_3_forward_builder = make_noop_forward_builder(flash_mha_3)
    flash_mha_3_backward_builder = make_vjp_backward_builder(flash_mha_3)
# fmt: on


# ===== Profiling core =====
FA_FUNCTIONS_LUT = {
    "jax": {"fwd": jax_attn_forward_builder, "bwd": jax_attn_backward_builder},
    "te": {"fwd": te_fa_forward_builder, "bwd": te_fa_backward_builder},
}
if ENABLE_TRITON:
    FA_FUNCTIONS_LUT["triton"] = {
        "fwd": triton_fa_forward_builder,
        "bwd": triton_fa_backward_builder,
    }
if ENABLE_FA3:
    FA_FUNCTIONS_LUT["fa3"] = {
        "fwd": flash_mha_3_forward_builder,
        "bwd": flash_mha_3_backward_builder,
    }


def block_until_ready_tree(x):
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), x)


# fmt: off
@dataclass
class Case:
    kernel_name: Literal["jax", "triton", "te", "fa3"]
    name_suffix: str
    mode: str
    layout: str
    dtype: jnp.dtype
    batch_size: int
    seqlen_q: int
    seqlen_kv: int
    nheads: int
    dim: int
    gqa_ratio: int
    causal: bool
    nr_segments: int
    sliding_window_size: int | None

    def profile_case(self, repeats: int, warmups: int, tensorboard_logdir: str | None):
        # Setup
        fn_builder = FA_FUNCTIONS_LUT[self.kernel_name][self.mode]
        sm_scale = 1 / self.dim
        q, k, v, do, segids_q, segids_kv = gen_data(self.dtype, self.batch_size, self.seqlen_q, self.seqlen_kv, self.nheads, self.dim, self.gqa_ratio, self.nr_segments)
        jit_fn, args = fn_builder(do, q, k, v, segids_q, segids_kv, sm_scale, self.causal, self.sliding_window_size, self.layout, self.nr_segments)

        # Memory profiling
        ma = jit_fn.lower(*args).compile().memory_analysis()
        mem_mib = (ma.argument_size_in_bytes + ma.generated_code_size_in_bytes + ma.output_size_in_bytes + ma.temp_size_in_bytes) / (1024**2)

        # Steptime profiling
        for _ in range(warmups):
            block_until_ready_tree(jit_fn(*args))
        steptimes_ms = []

        for _ in range(repeats):
            start = time.perf_counter_ns()
            block_until_ready_tree(jit_fn(*args))
            steptimes_ms.append((time.perf_counter_ns() - start) / 1e6)

        # TB profiling
        if tensorboard_logdir is not None:
            jax.profiler.start_trace(tensorboard_logdir)
            block_until_ready_tree(jit_fn(*args))
            jax.profiler.stop_trace()

        return {
            "fn_name": self.kernel_name + self.name_suffix,
            "mode": self.mode,
            "layout": self.layout,
            "dtype": DTYPE_MAP_INV[self.dtype],
            "batch_size": self.batch_size,
            "seqlen_q": self.seqlen_q,
            "seqlen_kv": self.seqlen_kv,
            "nheads": self.nheads,
            "dim": self.dim,
            "gqa_ratio": self.gqa_ratio,
            "causal": self.causal,
            "num_segments": self.nr_segments,
            "sliding_window_size": self.sliding_window_size,
            "min_steptime_ms": f"{min(steptimes_ms):.3f}",
            "median_steptime_ms": f"{float(np.median(steptimes_ms)):.3f}",
            "mean_steptime_ms": f"{float(np.mean(steptimes_ms)):.3f}",
            "q1_steptime_ms": f"{float(np.percentile(steptimes_ms, 25)):.3f}",
            "q3_steptime_ms": f"{float(np.percentile(steptimes_ms, 75)):.3f}",
            "memory_total_mib": f"{mem_mib:.3f}",
            "memory_inputs_mib": f"{ma.argument_size_in_bytes / (1024**2):.3f}",
            "memory_outputs_mib": f"{ma.output_size_in_bytes / (1024**2):.3f}",
            "memory_temp_mib": f"{ma.temp_size_in_bytes / (1024**2):.3f}",
            "memory_code_mib": f"{ma.generated_code_size_in_bytes / (1024**2):.3f}",
        }
# fmt: on


# ===== Runtime =====
def str_to_dtype(x):
    if x.lower() not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {x}")
    return DTYPE_MAP[x.lower()]


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-names", nargs="+", type=str, required=True, choices=["jax", "triton", "te", "fa3"])
    parser.add_argument("--name-suffix", type=str, default="")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--dtypes", nargs="+", type=str_to_dtype, default=[jnp.bfloat16])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--seqlens-q", nargs="+", type=int, default=[2048])
    parser.add_argument("--seqlens-kv", nargs="+", type=int, default=[2048])
    parser.add_argument("--nheads", nargs="+", type=int, default=[32])
    parser.add_argument("--dims", nargs="+", type=int, default=[128])
    parser.add_argument("--gqa-ratios", nargs="+", type=int, default=[1])
    parser.add_argument("--nr-segments", nargs="+", type=int, default=[1])
    parser.add_argument("--modes", nargs="+", type=str, choices=["fwd", "bwd"], default=["fwd"])
    parser.add_argument("--layouts", nargs="+", type=str, choices=["bshd", "thd"], default=["bshd"])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[-1])
    parser.add_argument("--non-causal", action="store_true", default=False)
    parser.add_argument("--tensorboard-logdir", type=str, default=None)
    args = parser.parse_args()

    configs = []
    config_product = product(args.dtypes, args.batch_sizes, args.seqlens_q, args.seqlens_kv, args.nheads,
                             args.dims, args.gqa_ratios, args.modes, args.layouts, args.nr_segments, args.window_sizes)
    for dtype, bsz, sq, skv, nh, dm, gqa, mode, layout, nr_segment, window_size in config_product:
        if not ((nr_segment > 1 and layout == "bshd") or (window_size > sq)):
            configs.append(dict(name_suffix=args.name_suffix, mode=mode, layout=layout, dtype=dtype, batch_size=bsz,
                                seqlen_q=sq, seqlen_kv=skv, nheads=nh, dim=dm, gqa_ratio=gqa, causal=not args.non_causal,
                                nr_segments=nr_segment, sliding_window_size=window_size))
    # fmt: on

    for cfg in tqdm(configs):
        kernel_results = {}
        for kernel_name in args.kernel_names:
            case = Case(kernel_name=kernel_name, **cfg)
            try:
                kernel_results[kernel_name] = case.profile_case(
                    args.repeats, args.warmups, args.tensorboard_logdir
                )
            except Exception as e:
                print(f"Failed case: {case}")
                print(e)
            jax.clear_caches()
            gc.collect()

        if not kernel_results or args.csv is None:
            continue

        first = next(iter(kernel_results.values()))
        row = {
            "mode": first["mode"],
            "layout": first["layout"],
            "dtype": first["dtype"],
            "batch_size": first["batch_size"],
            "seqlen_q": first["seqlen_q"],
            "seqlen_kv": first["seqlen_kv"],
            "nheads": first["nheads"],
            "dim": first["dim"],
            "gqa_ratio": first["gqa_ratio"],
            "causal": first["causal"],
            "num_segments": first["num_segments"],
            "sliding_window_size": first["sliding_window_size"],
        }

        timing_fields = [
            "min_steptime_ms", "median_steptime_ms", "mean_steptime_ms",
            "q1_steptime_ms", "q3_steptime_ms", "memory_total_mib",
        ]
        for kname in args.kernel_names:
            res = kernel_results.get(kname)
            for field in timing_fields:
                col = f"{kname}_{field}"
                row[col] = res[field] if res else ""

        means = []
        for kname in args.kernel_names:
            res = kernel_results.get(kname)
            means.append(float(res["mean_steptime_ms"]) if res else None)
        if len(means) == 2 and all(m is not None and m > 0 for m in means):
            row["speedup_mean"] = f"{means[1] / means[0]:.2f}x"
        else:
            row["speedup_mean"] = ""

        file_exists = os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
