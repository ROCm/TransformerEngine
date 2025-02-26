# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# See LICENSE for license information.

import os, sys, time
import subprocess
import pandas as pd
import numpy as np
import torch
import nvtx
import transformer_engine
from transformer_engine_torch import NVTE_Fused_Attn_Backend

cwd = os.getcwd()
if "benchmark" in cwd:
    trimmed_path = cwd[:cwd.index("benchmark")]
    sys.path.append(trimmed_path)
else:
    sys.path.append(cwd)

from tests.pytorch.fused_attn.test_fused_attn import (
    ModelConfig,
    _get_attention_backends,
    _run_dot_product_attention,
)

pd.set_option("display.precision", 4)

# data type
dtype = torch.bfloat16
# number of iterations after 3 warmup iterations
num_iters = 3
# checkpointing
ckpt_attn = False
# workspace optimization path for cuDNN attention
workspace_opt = True
# QKV memory layout
qkv_layout = "bshd_bshd_bshd"
# padding between sequences for qkv_format=thd
pad_between_seqs = False
# training mode
is_training = True

model_configs = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,              bias
    "test_0": ModelConfig(2, 16, 16, 64, 512, 512, 0.0, "no_mask", "no_bias"),  # short seq
    "test_1": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "no_bias"),  # longer seq, mask
    "test_2": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),  # bias
    "test_3": ModelConfig(2, 32, 4, 128, 8192, 8192, 0.0, "causal", "no_bias"),  # GQA
}

# Runs for warmup iterations and started profiling using rocprof
def benchmark_dot_product_attention(model, attention, column_name, filename):
    config = model_configs[model]

    warmup_iters = 3
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
        
    prof_cmd = [
            "rocprof",
            "--hip-trace",
            "--basenames off",
            "python",
            "-c",
            f""" "import benchmark_attention_amd;""",
            f"""benchmark_attention_amd.benchmark_dot_product_attention_profiler("""
            f"""'{model}', '{attention}', '{column_name}')" """,
        ]
    prof_cmd = " ".join(prof_cmd)
    subprocess.call(prof_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    if os.path.exists("results.stats.csv"):
        os.rename("results.stats.csv", filename)
    else:
        print("Error: results.stats.csv not found!")
    torch.cuda.empty_cache()
    
# Profiler helper function for rocprof
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

    df = pd.read_csv("times.csv")
    last_row_index = len(df) - 1
    df.loc[last_row_index, column_name] = attn_time * 1e3 / num_iters
    df.to_csv("times.csv", index=False)
    torch.cuda.empty_cache()

def parse_helper(filename, fwd_search_pattern, bwd_search_pattern, column_name, df_times):
    row = len(df_times.index) - 1
    df = pd.read_csv(os.path.join("./", filename))
    
    t_attn_avg = np.empty(4)
    t_attn_avg[0] = df[df["Name"].str.contains(fwd_search_pattern)]["AverageNs"].to_numpy()
    t_attn_avg[1:4] = df[df["Name"].str.contains(bwd_search_pattern)]["AverageNs"].to_numpy()
    
    df_times.loc[row, f"{column_name} Kernels (fwd)"] = t_attn_avg[0] / 1e6
    df_times.loc[row, f"{column_name} Kernels (bwd)"] = t_attn_avg[1:4].sum() / 1e6
    df_times.loc[row, f"{column_name} Kernels (fwd+bwd)"] = t_attn_avg.sum() / 1e6

    return df_times

# Parser function to parse the results.stats file form rocprof
# This function gathers Avg timing information for both Fwd and Bwd Kernels.
def parse_results(per_cudnn, per_flash, model):
    df_times = pd.read_csv("times.csv")
    row = len(df_times.index) - 1

    if per_cudnn > 0:
        # FUSED
        filename = f"prof_fused_{model}.csv"
        df_times = parse_helper(filename, "FmhaFwd", "FmhaBwd", "FusedAttention", df_times)

        #CK
        filename = f"prof_fused_ck_{model}.csv"
        if os.path.exists(filename):
            df_times = parse_helper(filename, "FmhaFwd", "FmhaBwd", "FusedAttention CK", df_times)
        
        #AOTriton
        filename = f"prof_fused_aotriton_{model}.csv"
        if os.path.exists(filename):
            df_times = parse_helper(filename, "attn_fwd", "bwd", "FusedAttention AOTriton", df_times)
        
    #FLASH
    if per_flash > 0:
        filename = f"prof_flash_{model}.csv"
        df_times = parse_helper(filename, "FmhaFwd", "FmhaBwd", "FlashAttention", df_times)
        
    if per_cudnn > 0 and per_flash > 0:
        df_times.loc[row, "Fused vs Flash Kernels Speedup (fwd+bwd)"] = (
            df_times.loc[row, "FlashAttention Kernels (fwd+bwd)"]
            / df_times.loc[row, "FusedAttention Kernels (fwd+bwd)"]
        )
    df_times.to_csv("times.csv", index=False)


def main():
    # Creating the required columns to benchmark
    times = pd.DataFrame(
        columns=[
            "FusedAttention Module",
            "FusedAttention Kernels (fwd)",
            "FusedAttention Kernels (bwd)",
            "FusedAttention Kernels (fwd+bwd)",
            "FlashAttention Module",
            "FlashAttention Kernels (fwd)",
            "FlashAttention Kernels (bwd)",
            "FlashAttention Kernels (fwd+bwd)",
            "Fused vs Flash Kernels Speedup (fwd+bwd)",
            "FusedAttention CK Module ",
            "FusedAttention CK Kernels (fwd)",
            "FusedAttention CK Kernels (bwd)",
            "FusedAttention CK Kernels (fwd+bwd)",
            "FusedAttention AOTriton Module",
            "FusedAttention AOTriton Kernels (fwd)",
            "FusedAttention AOTriton Kernels (bwd)",
            "FusedAttention AOTriton Kernels (fwd+bwd)",
        ]
    )
    times.to_csv("times.csv", index=False)

    device_id = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_id)
    print(
        f"Device {device_id}: "
        f"{device_properties.name} GPU, "
        f"sm{device_properties.major}{device_properties.minor} compute capability, "
        f"{device_properties.total_memory/1024**3:.1f}GB memory"
    )
    # Benchmarking starts..
    for model in model_configs.keys():
        config = model_configs[model]
        available_backends, fused_attn_backends = _get_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            window_size=config.window_size,
            pad_between_seqs=pad_between_seqs,
        )
        flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    
        if not(fused_attn_supported or flash_attn_supported):
            print("No attention backend's detected for ",model)
            continue

        print(
            f'Running {model} with {"cuDNN attention" if fused_attn_supported else ""}'
            f'{" and flash-attention" if flash_attn_supported else ""}...'
        )
        # Initialize the row for current model
        df = pd.read_csv("times.csv")
        new_row = [0.0] * len(df.columns)
        df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)
        df.to_csv("times.csv", index=False)

        # Benchmark for each attention backend
        if flash_attn_supported:
            benchmark_dot_product_attention(model, "FlashAttention", "FlashAttention Module", f"prof_flash_{model}.csv")
        
        if fused_attn_supported:
            
            benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention Module", f"prof_fused_{model}.csv")
            
            if NVTE_Fused_Attn_Backend.NVTE_CK in fused_attn_backends:
                #CK Backend
                os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "0"
                # os.environ["NVTE_CK_USES_BWD_V3"] = "1"
                os.environ["NVTE_FUSED_ATTN_CK"] = "1"
                os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
                benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention CK Module", f"prof_fused_ck_{model}.csv")

            if NVTE_Fused_Attn_Backend.NVTE_AOTriton in fused_attn_backends:
                #AOTRITON Backend
                os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
                os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "1"
                # os.environ["NVTE_CK_USES_BWD_V3"] = "0"
                os.environ["NVTE_FUSED_ATTN_CK"] = "0"
                benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention AOTriton Module", f"prof_fused_aotriton_{model}.csv")

            del os.environ["NVTE_FUSED_ATTN_CK"]
            # del os.environ["NVTE_CK_USES_BWD_V3"]
            del os.environ["NVTE_FUSED_ATTN_AOTRITON"]

        if fused_attn_supported:
            num_kernels_cudnn = 4
            if config.attn_bias_type == "post_scale_bias":
                num_kernels_cudnn = num_kernels_cudnn + 1
            if config.num_heads != config.num_gqa_groups:
                num_kernels_cudnn = num_kernels_cudnn + 2
        else:
            num_kernels_cudnn = 0
        num_kernels_flash = 4 if flash_attn_supported else 0
        
        # Parser to populate csv file
        parse_results(num_kernels_cudnn, num_kernels_flash, model)
        
    df_times = pd.read_csv("times.csv")
    df_times.index = list(model_configs.keys())
    a = df_times[
        [
            "FusedAttention Kernels (fwd+bwd)",
            "FlashAttention Kernels (fwd+bwd)",
            "Fused vs Flash Kernels Speedup (fwd+bwd)",
        ]
    ]
    a.columns = ["cuDNN fwd+bwd (ms)", "flash-attn fwd+bwd (ms)", "cuDNN vs flash speedup"]
    print()
    print(a)


if __name__ == "__main__":
    main()