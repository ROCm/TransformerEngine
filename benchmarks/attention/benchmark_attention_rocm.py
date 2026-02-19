# This file was modified for portability to AMDGPU
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# See LICENSE for license information.

import os, sys, time, shutil
import argparse
import subprocess
import pandas as pd
import numpy as np
import torch
import transformer_engine
from transformer_engine_torch import NVTE_Fused_Attn_Backend

# Add paths tests/pytorch/ and tests/pytorch/attention to the sys path 
tests_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../tests")
)
sys.path.append(tests_path + "/pytorch")
sys.path.append(tests_path + "/pytorch/attention")

# Add tests/pytorch/utils.py path into sys path
from utils import (
    ModelConfig,
    get_available_attention_backends,
)
from test_attention import (
    _run_dot_product_attention,
)

pd.set_option("display.precision", 4)

# -------------------- Benchmark Settings --------------------
# Data type
dtype = torch.bfloat16
# Number of warmup iterations before profiling
warmup_iters = 20
# Number of iterations after warmup iterations
num_iters = 10
# Checkpointing attention
ckpt_attn = False
# Workspace optimization for attention
workspace_opt = True
# QKV memory layout
qkv_layout = "bshd_bshd_bshd"
# Padding between sequences for qkv_format=thd
pad_between_seqs = False
# Training mode
is_training = True

model_configs = {
    #   test:             b,  sq, h, d
    "test_0": ModelConfig(2, 512, 16, 64),  # short seq
    "test_1": ModelConfig(2, 2048, 16, 128, attn_mask_type="causal"),  # longer seq, mask
    "test_2": ModelConfig(2, 2048, 16, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias"),  # bias
    "test_3": ModelConfig(2, 8192, 32, 128, num_gqa_groups=4, attn_mask_type="causal"),  # GQA
    "test_4": ModelConfig(2, 8192, 128, 128, num_gqa_groups=8, attn_mask_type="causal_bottom_right")
}

# DataFrame indices and columns for results
indices = [model for model in model_configs.keys()]
columns = [
    "FusedAttention CK Module",
    "FusedAttention CK Kernels (fwd)",
    "FusedAttention CK Kernels (bwd)",
    "FusedAttention CK Kernels (fwd+bwd)",
    "FusedAttention CK TFLOPs (fwd)",
    "FusedAttention CK TFLOPs (bwd)",

    "FlashAttention Module",
    "FlashAttention Kernels (fwd)",
    "FlashAttention Kernels (bwd)",
    "FlashAttention Kernels (fwd+bwd)",
    "FlashAttention TFLOPs (fwd)",
    "FlashAttention TFLOPs (bwd)",
    "Fused vs Flash Kernels Speedup (fwd+bwd)",

    "FusedAttention AOTriton Module",
    "FusedAttention AOTriton Kernels (fwd)",
    "FusedAttention AOTriton Kernels (bwd)",
    "FusedAttention AOTriton Kernels (fwd+bwd)",
    "FusedAttention AOTriton TFLOPs (fwd)",
    "FusedAttention AOTriton TFLOPs (bwd)",
]

# Output CSV filename
output_csv = "times.csv"
# Output directory name
output_dir_name = "profiler_outputs"
# Current working directory
cwd = os.getcwd()

# All attention backend environment variables
ATTENTION_ENV_VARS = [
    "NVTE_FUSED_ATTN",
    "NVTE_FLASH_ATTN", 
    "NVTE_FUSED_ATTN_AOTRITON",
    "NVTE_FUSED_ATTN_CK",
    "NVTE_UNFUSED_ATTN",
    "NVTE_CK_USES_BWD_V3",
    "NVTE_CK_USES_FWD_V3",
    "NVTE_CK_IS_V3_ATOMIC_FP32",
]

def cleanup_env():
    """Set all attention-related environment variables to 0."""
    for var in ATTENTION_ENV_VARS:
        os.environ[var] = "0"

def setup_backend_env(backend_name, use_ck_bwd_v3=True, use_ck_fwd_v3=True, use_ck_v3_a16=False):
    cleanup_env()
    
    if backend_name == "flash":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    elif backend_name == "fused_ck":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN_CK"] = "1"
        os.environ["NVTE_CK_USES_BWD_V3"] = "1" if use_ck_bwd_v3 else "0"
        if use_ck_bwd_v3:
            os.environ["NVTE_CK_IS_V3_ATOMIC_FP32"] = "0" if use_ck_v3_a16 else "1"
        os.environ["NVTE_CK_USES_FWD_V3"] = "1" if use_ck_fwd_v3 else "0"
    elif backend_name == "fused_aotriton":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "1"

# Kernel name patterns for identifying kernels in profiler output
KERNEL_PATTERNS = {
    # Flash Attention patterns
    "flash_fwd": "FmhaFwd",
    "flash_bwd": "FmhaBwd",
    
    # CK patterns (v2 and v3)
    "ck_fwd_v2": "ck_tile::FmhaFwdKernel",
    "ck_bwd_v2": "ck_tile::FmhaBwd",
    "ck_fwd_v3": "aiter::fmha_fwd",
    "ck_bwd_v3": "aiter::fmha_bwd",

    # AOTriton patterns
    "aotriton_fwd": "attn_fwd",
    "aotriton_bwd": "bwd",
}

# Runs benchmark with warmup iterations and profiles using rocprof
def benchmark_dot_product_attention(model, attention, column_name, dirname):
    config = model_configs[model]

    for i in range(warmup_iters):
        attn_fwd, attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                attention,
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
    os.makedirs(dirname, exist_ok=True)
    before_files = set(os.listdir(cwd))
    # Profiling command using rocprof
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    prof_cmd = [
            "rocprof",
            "--hip-trace",
            "--basenames off",
            "python",
            "-c",
            f""" "import sys; sys.path.insert(0, '{benchmark_dir}'); import benchmark_attention_rocm;""",
            f"""benchmark_attention_rocm.benchmark_dot_product_attention_profiler("""
            f"""'{model}', '{attention}', '{column_name}')" """,
        ]
    prof_cmd = " ".join(prof_cmd)
    subprocess.call(prof_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    after_files = set(os.listdir(cwd))
    new_files = after_files - before_files

    for f in new_files:
        src_path = os.path.join(cwd, f)
        dst_path = os.path.join(dirname, f)
        if os.path.isfile(src_path):  # Only move files, not directories
            shutil.move(src_path, dst_path)
    torch.cuda.empty_cache()
    
# Runs profiler and records timing information
def benchmark_dot_product_attention_profiler(model, attention, column_name):
    config = model_configs[model]
    torch.cuda.synchronize()
    attn_start = time.time()
    
    for i in range(num_iters):
        attn_fwd, attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                attention,
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
    torch.cuda.synchronize()
    attn_time = time.time() - attn_start

    output_csv_path = os.path.join(cwd, output_csv)
    df_times = pd.read_csv(output_csv_path, index_col=0)
    df_times.loc[model, column_name] = attn_time * 1e3 / num_iters
    df_times.to_csv(output_csv_path)
    torch.cuda.empty_cache()

# Calculate TFLOPs for attention operations
def calculate_attention_tflops(batch_size, seq_len, num_heads_q, head_dim_qk, fwd_time_ms, bwd_time_ms, is_causal):
    # Calculate total fwdFLOPs
    fwd_flops = (0.5 if is_causal else 1.0) * 4 * batch_size * seq_len * seq_len * num_heads_q * head_dim_qk / 1e12
    # Calculate forward TFLOPs
    fwd_tflops = fwd_flops / (fwd_time_ms / 1000.0)
    # Calculate backward TFLOPs
    bwd_tflops = (fwd_flops / (bwd_time_ms / 1000.0)) * 2.5
    return fwd_tflops, bwd_tflops

# Helper function to extract timing results from profiler logs
def parse_helper(model, dirname, fwd_search_pattern, bwd_search_pattern, column_name, df_times):
    df = pd.read_csv(os.path.join(dirname, "results.stats.csv"))

    # Extract kernel timing values
    fwd_values = df[df["Name"].str.contains(fwd_search_pattern, regex=False)]["AverageNs"].to_numpy()
    bwd_values = df[df["Name"].str.contains(bwd_search_pattern, regex=False)]["AverageNs"].to_numpy()

    if len(fwd_values) == 0 or len(bwd_values) == 0:
        return False  # Kernels not found
    
    t_attn_avg = np.empty(len(fwd_values) + len(bwd_values))
    t_attn_avg[:len(fwd_values)] = fwd_values
    t_attn_avg[len(fwd_values):] = bwd_values
    
    # Store results in DataFrame (convert from ns to ms)
    fwd_time_ms = t_attn_avg[:len(fwd_values)].sum() / 1e6
    bwd_time_ms = t_attn_avg[len(fwd_values):].sum() / 1e6
    
    df_times.loc[model, f"{column_name} Kernels (fwd)"] = fwd_time_ms
    df_times.loc[model, f"{column_name} Kernels (bwd)"] = bwd_time_ms
    df_times.loc[model, f"{column_name} Kernels (fwd+bwd)"] = fwd_time_ms + bwd_time_ms
    
    # Calculate TFLOPs for both forward and backward
    config = model_configs[model]
    is_causal = "causal" in config.attn_mask_type.lower()
    fwd_tflops, bwd_tflops = calculate_attention_tflops(
        config.batch_size, config.max_seqlen_q, config.num_heads, 
        config.head_dim_qk, fwd_time_ms, bwd_time_ms, is_causal
    )
    
    df_times.loc[model, f"{column_name} TFLOPs (fwd)"] = fwd_tflops
    df_times.loc[model, f"{column_name} TFLOPs (bwd)"] = bwd_tflops
    
    return True

# Parses profiler logs for different attention backends
def parse_results(model, df_times, perf_dir_flash_attn, perf_dir_fused_ck, perf_dir_fused_aotriton, use_ck_bwd_v3, use_ck_fwd_v3):
    # Parse Flash Attention
    if perf_dir_flash_attn:
        parse_helper(model, perf_dir_flash_attn, KERNEL_PATTERNS["flash_fwd"], KERNEL_PATTERNS["flash_bwd"], "FlashAttention", df_times)
    
    # Parse FusedAttention CK (use v3 or v2 patterns based on flags)
    if perf_dir_fused_ck:
        fwd_pattern = KERNEL_PATTERNS["ck_fwd_v3"] if use_ck_fwd_v3 else KERNEL_PATTERNS["ck_fwd_v2"]
        bwd_pattern = KERNEL_PATTERNS["ck_bwd_v3"] if use_ck_bwd_v3 else KERNEL_PATTERNS["ck_bwd_v2"]
        parse_helper(model, perf_dir_fused_ck, fwd_pattern, bwd_pattern, "FusedAttention CK", df_times)
    
    # Parse AOTriton
    if perf_dir_fused_aotriton:
        parse_helper(model, perf_dir_fused_aotriton, KERNEL_PATTERNS["aotriton_fwd"], KERNEL_PATTERNS["aotriton_bwd"], "FusedAttention AOTriton", df_times)
    
    # Calculate speedup if both Flash and Fused CK results exist
    if perf_dir_flash_attn and perf_dir_fused_ck:
        flash_time = df_times.loc[model, "FlashAttention Kernels (fwd+bwd)"]
        fused_time = df_times.loc[model, "FusedAttention CK Kernels (fwd+bwd)"]
        if flash_time > 0 and fused_time > 0:
            df_times.loc[model, "Fused vs Flash Kernels Speedup (fwd+bwd)"] = flash_time / fused_time

# Post-benchmark sanity checks
def sanity_checks(
    profiler_root: str = None,
    csv_path: str = None,
    tolerance_pct: float = 5.0,
):
    """
    • Verifies that every model/backend that *should* have run produced
        profiler_root/<dir>/results.stats.csv
    • Non-zero exit code on any failure (CI friendly)
    """
    if profiler_root is None:
        profiler_root = output_dir_name
    if csv_path is None:
        csv_path = output_csv
    print("\n============= Sanity-check results =============")
    ok_overall = True
    times_csv_path = os.path.join(cwd, csv_path)
    df = pd.read_csv(times_csv_path, index_col=0)
    
    tol = tolerance_pct / 100.0
    profiler_root = os.path.join(cwd, profiler_root)

    dir_pattern = {
        "FlashAttention":           "prof_flash_{model}",
        "FusedAttention CK":        "prof_fused_ck_{model}",
        "FusedAttention AOTriton":  "prof_fused_aotriton_{model}",
    }

    for model, cfg in model_configs.items():
        avail, _, fused_bes = get_available_attention_backends(
            cfg,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            window_size=cfg.window_size,
            pad_between_seqs=pad_between_seqs,
        )
        flash_ok, fused_ok, _ = avail

        expected = {}
        if flash_ok:
            expected["FlashAttention"] = dir_pattern["FlashAttention"]
        if fused_ok:
            if NVTE_Fused_Attn_Backend.NVTE_CK in fused_bes:
                expected["FusedAttention CK"] = dir_pattern["FusedAttention CK"]
            if NVTE_Fused_Attn_Backend.NVTE_AOTriton in fused_bes:
                expected["FusedAttention AOTriton"] = dir_pattern["FusedAttention AOTriton"]

        print(f"{model}:")
        # Rocprof run status
        for be, pat in expected.items():
            stats = os.path.join(profiler_root, pat.format(model=model), "results.stats.csv")
            if os.path.isfile(stats):
                print(f"  [{be:<22}] Profiling successful")
            else:
                ok_overall = False
                raise FileNotFoundError(f"Error while profiling {model} [{be}], results.stats.csv not found")

        print("-" * 60)
    return ok_overall


def main(args):
    output_dir = os.path.join(cwd, output_dir_name + "/")
    output_csv_path = os.path.join(cwd, output_csv)

    # Clean up old outputs in cwd
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    os.makedirs(output_dir)

    df_times = pd.DataFrame(index=indices, columns=columns)
    df_times = df_times.infer_objects(copy=False)
    df_times.fillna(0.0, inplace=True)
    df_times.index.name = "Model"
    df_times.to_csv(output_csv_path)
    
    device_id = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_id)
    print(
        f"Device {device_id}: "
        f"{device_properties.name} GPU, "
        f"{device_properties.gcnArchName.split(':')[0]} architecture, "
        f"{device_properties.total_memory/1024**3:.1f}GB memory"
    )
    # Benchmarking starts..
    for model in model_configs.keys():
        config = model_configs[model]
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            window_size=config.window_size,
            pad_between_seqs=pad_between_seqs,
        )
        flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends

        if not(fused_attn_supported or flash_attn_supported):
            print(f"No attention backend detected for {model}")
            continue

        print(
            f'Running {model} with {"Fused Attention" if fused_attn_supported else ""}'
            f'{" and flash-attention" if flash_attn_supported else ""}...'
        )

        perf_dir_flash_attn, perf_dir_fused_ck, perf_dir_fused_aotriton = None, None, None
        
        # Benchmark Flash Attention
        if flash_attn_supported:
            setup_backend_env("flash")
            perf_dir_flash_attn = os.path.join(output_dir, f"prof_flash_{model}")
            benchmark_dot_product_attention(model, "FlashAttention", "FlashAttention Module", perf_dir_flash_attn)
           
        # Benchmark Fused Attention CK (with v2/v3 based on flags)
        if fused_attn_supported and NVTE_Fused_Attn_Backend.NVTE_CK in fused_attn_backends:
            setup_backend_env("fused_ck", use_ck_bwd_v3=args.use_ck_bwd_v3, use_ck_fwd_v3=args.use_ck_fwd_v3, use_ck_v3_a16=args.use_ck_v3_a16)
            perf_dir_fused_ck = os.path.join(output_dir, f"prof_fused_ck_{model}")
            benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention CK Module", perf_dir_fused_ck)
        
        # AOTriton Backend
        if fused_attn_supported and NVTE_Fused_Attn_Backend.NVTE_AOTriton in fused_attn_backends:
            setup_backend_env("fused_aotriton")
            perf_dir_fused_aotriton = os.path.join(output_dir, f"prof_fused_aotriton_{model}")
            benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention AOTriton Module", perf_dir_fused_aotriton)

        df_times = pd.read_csv(output_csv_path, index_col=0)
        parse_results(model, df_times, perf_dir_flash_attn, perf_dir_fused_ck, perf_dir_fused_aotriton, args.use_ck_bwd_v3, args.use_ck_fwd_v3)
        df_times.to_csv(output_csv_path)

    df_times = pd.read_csv(output_csv_path)
    df_times.index = list(model_configs.keys())
    timing_df = df_times[
        [
            "FusedAttention CK Kernels (fwd)",
            "FusedAttention CK Kernels (bwd)",
            "FusedAttention CK Kernels (fwd+bwd)",
            "FlashAttention Kernels (fwd+bwd)",
            "Fused vs Flash Kernels Speedup (fwd+bwd)",
        ]
    ].copy()
    timing_df.columns = [
        "CK fwd (ms)",
        "CK bwd (ms)",
        "CK fwd+bwd (ms)",
        "Flash fwd+bwd (ms)",
        "CK/Flash Speedup",
    ]
    print(timing_df)
    print()
    tflops_df = df_times[
        [
            "FusedAttention CK TFLOPs (fwd)",
            "FusedAttention CK TFLOPs (bwd)",
        ]
    ].copy()
    tflops_df.columns = ["CK FWD TFLOPs", "CK BWD TFLOPs"]
    print(tflops_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ck_bwd_v3", action="store_false", dest="use_ck_bwd_v3", help="Set NVTE_CK_USES_BWD_V3=0 for CK bwd kernels")
    parser.add_argument("--no_ck_fwd_v3", action="store_false", dest="use_ck_fwd_v3", help="Set NVTE_CK_USES_FWD_V3=0 for CK fwd kernels")
    parser.add_argument("--use_ck_v3_a16", action="store_true", help="Use NVTE_CK_IS_V3_ATOMIC_FP32=0 for atomic16. Default is 1")
    parser.add_argument("--run_sanity_checks", action="store_true", help="After benchmarking, verify profiler outputs.")
    args = parser.parse_args()
    main(args)
    if args.run_sanity_checks:
        sanity_checks()
