# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# See LICENSE for license information.

import os, sys
from pathlib import Path
import pandas as pd
import argparse
from functools import partial
from itertools import product
import jax
from jax import numpy as jnp
import csv
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
)
from transformer_engine.jax import fp8_autocast

# Add test_fused_attn to the sys path 
tests_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../tests/jax/")
)
sys.path.append(tests_path)

from test_fused_attn import (
    FusedAttnRunner,
    FusedAttnHelper,
    SeqDescFormat,
    BiasShape,
    customcall_fused_dpa,
)


# "b, s_q, s_kv, h_q, h_kv, d_qk, d_v,"
SHAPES = ((2, 2048, 2048, 12, 12, 64, 64),)

# data type
DTYPES = [jnp.float16, jnp.bfloat16]

ATTN_MASK_TYPES = (
    AttnMaskType.NO_MASK,
    AttnMaskType.PADDING_MASK,
    AttnMaskType.CAUSAL_MASK,
    AttnMaskType.PADDING_CAUSAL_MASK,
)
QKV_LAYOUTS = (
    QKVLayout.BS3HD,
    QKVLayout.BSHD_BS2HD,
    QKVLayout.BSHD_BSHD_BSHD,
    QKVLayout.T3HD,
    QKVLayout.THD_T2HD,
    QKVLayout.THD_THD_THD
)
SEQ_DESC_FORMATS = (SeqDescFormat.Mask, SeqDescFormat.Seqlens, SeqDescFormat.SegmentIDs)
SWA = (True, False)
IS_TRAINING = (True, False)
DROPOUT = (0.0, 0.1)
BIAS_CONFIGS = ((AttnBiasType.NO_BIAS, None), (AttnBiasType.POST_SCALE_BIAS, BiasShape._1HSS))
CONFIGS = tuple(
    product(
        SHAPES,
        DTYPES,
        ATTN_MASK_TYPES,
        QKV_LAYOUTS,
        SEQ_DESC_FORMATS,
        SWA,
        IS_TRAINING,
        DROPOUT,
        BIAS_CONFIGS,
    )
)

COLUMNS = [
    "batch_size",
    "q_seq_len",
    "kv_seq_len",
    "q_heads",
    "kv_heads",
    "qk_dim",
    "v_dim",
    "attn_bias_type",
    "attn_mask_type",
    "dropout",
    "dtype",
    "is_training",
    "qkv_layout",
    "bias_shape",
    "swa",
    "seq_desc_format",
    "mode",
    "time",
]

CWD = os.getcwd()

class FusedAttnBenchRunner(FusedAttnRunner):
    def bench_forward(self, warmup, iters):
        """
        Test forward without JIT
        """
        self._setup_inputs()
        customcall_args = [
            jax.device_put(self.cp_reorder_fn(self.q), self.qkvo_sharding),
            jax.device_put(self.cp_reorder_fn(self.k), self.qkvo_sharding),
            jax.device_put(self.cp_reorder_fn(self.v), self.qkvo_sharding),
            jax.device_put(self.bias, self.bias_sharding),
            jax.device_put(self.sequence_desciptor, self.seq_desc_sharding),
            jax.device_put(self.dropout_rng, self.dropout_rng_sharding),
        ]
        kwargs = {
            "attn_bias_type": self.attn_bias_type,
            "attn_mask_type": self.attn_mask_type,
            "scaling_factor": self.scaling_factor,
            "dropout_probability": self.dropout_prob,
            "is_training": self.is_training,
            "qkv_layout": self.qkv_layout,
            "max_segments_per_seq": self._get_max_segments_per_sequence(),
            "window_size": self.window_size,
            "context_parallel_strategy": self.cp_strategy,
            "context_parallel_causal_load_balanced": self.cp_load_balanced,
        }

        customcall_fused_dpa_jit = jax.jit(
            partial(customcall_fused_dpa, **kwargs),
            static_argnames=kwargs.keys(),
            in_shardings=[
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.bias_sharding,
                self.seq_desc_sharding,
                self.dropout_rng_sharding,
            ],
        )
        with self.mesh, fp8_autocast(mesh_resource=self.mesh_resource):
            for _ in range(warmup):
                customcall_fused_dpa_jit(*customcall_args)

            os.environ["NVTE_DUMP_AITER_RT"] = args.timings_dir

            for _ in range(iters):
                customcall_fused_dpa_jit(*customcall_args)

            del os.environ["NVTE_DUMP_AITER_RT"]

def _filter_configs(configs):
    for config in configs:
        (
            shape,
            dtype,
            attn_mask_type,
            qkv_layout,
            seq_desc_format,
            swa,
            is_training,
            dropout_prob,
            bias_config
        ) = config
        b, s_q, s_kv, h_q, h_kv, d_qk, d_v = shape
        attn_bias_type, bias_shape = bias_config
        window_size = None
        if swa:
            window_size = (s_kv // 10, 0)
        if qkv_layout.is_thd():
            if not attn_mask_type.is_padding():
                continue
            if seq_desc_format == SeqDescFormat.Mask:
                continue
        if qkv_layout.is_qkvpacked():
            if (s_q != s_kv) or h_q != h_kv:
                continue
        if s_q > s_kv and window_size is not None:
            continue
        if d_qk != d_v and not qkv_layout.is_separate():
            continue

        backend = FusedAttnHelper(
            dtype,
            dtype,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            h_q, h_kv,
            s_q, s_kv,
            d_qk, d_v,
            (-1, -1) if window_size is None else window_size,
        ).get_fused_attn_backend()
        if backend == -1:
            continue
        if (
            attn_bias_type == AttnBiasType.POST_SCALE_BIAS
            and bias_shape != BiasShape._1HSS
        ):
            if attn_mask_type.is_padding():
                continue
        yield config

# Runs profiler and records timing information
def benchmark_dot_product_attention_profiler(args):
    rows = []
    timings_path = {}
    for n, config in enumerate(_filter_configs(CONFIGS)):
        (
            shape,
            dtype,
            attn_mask_type,
            qkv_layout,
            seq_desc_format,
            swa,
            is_training,
            dropout_prob,
            bias_config
        ) = config
        b, s_q, s_kv, h_q, h_kv, d_qk, d_v = shape
        attn_bias_type, bias_shape = bias_config
        window_size = None
        if swa:
            window_size = (s_kv // 10, 0)
        output = {
            "batch_size":b,
            "q_seq_len":s_q,
            "kv_seq_len":s_kv,
            "q_heads":h_q,
            "kv_heads":h_kv,
            "qk_dim":d_qk,
            "v_dim":d_v,
            "attn_bias_type":attn_bias_type,
            "attn_mask_type":attn_mask_type,
            "dropout":dropout_prob,
            "dtype":dtype,
            "is_training":is_training,
            "qkv_layout":qkv_layout,
            "bias_shape":bias_shape,
            "swa":swa,
            "seq_desc_format":seq_desc_format,
        }
        if args.v:
            print(f"Progress: {n+1}")
        if args.v > 1:
            print(output)
        runner = FusedAttnBenchRunner(
            b, s_q, s_kv,
            h_q, h_kv,
            d_qk, d_v,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            True,
            dtype,
            is_training,
            qkv_layout,
            bias_shape,
            window_size,
            seq_desc_format,
        )
        runner.bench_forward(args.warmup, args.iters)

        timings_path["fwd"] = Path(args.timings_dir) / 'aiter-fwd-timings.txt'
        fwd_times = pd.read_csv(timings_path["fwd"], header=None, dtype=float)
        os.remove(timings_path["fwd"])
        rows.extend([output | {"mode": "fwd", "time": t} for t in fwd_times[0].to_list()])
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

class env_manager:
    def __init__(self, fwd, bwd):
        self.vals = {}
        self.config = ((fwd, "FWD"), (bwd, "BWD"))

    def __enter__(self):
        for flag, mode in self.config:
            if flag:
                self.vals[mode] = os.environ.get(f"NVTE_CK_USES_{mode}_V3")
                os.environ[f"NVTE_CK_USES_{mode}_V3"] = "1"

    def __exit__(self, exc_type, exc_value, traceback):
        for flag, mode in self.config:
            if flag:
                del os.environ[f"NVTE_CK_USES_{mode}_V3"]
                if self.vals[mode]:
                    os.environ[f"NVTE_CK_USES_{mode}_V3"] = self.vals[mode]

def main(args):
    with env_manager(args.fwd_v3, args.bwd_v3):
        benchmark_dot_product_attention_profiler(args)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fwd-v3", action="store_true", help="Use NVTE_CK_USES_FWD_V3=1 for AITER fwd kernels")
    parser.add_argument("--bwd-v3", action="store_true", help="Use NVTE_CK_USES_BWD_V3=1 for AITER bwd kernels")
    parser.add_argument("-v", action='count', default=0, help="Whether to include verbose debug outputs.")
    parser.add_argument("--output", type=str, help="The .csv file to output run times to.")
    parser.add_argument("--warmup", type=int, default=5, help="The number of iterations to run the kernel before logging run time.")
    parser.add_argument("--iters", type=int, default=50, help="The number of iterations to run the kernel while logging run time.")
    parser.add_argument("--timings-dir", type=str, help="The directory containing the 'aiter-\{fwd, bwd\}-timings.txt' files.")
    args = parser.parse_args()
    main(args)
