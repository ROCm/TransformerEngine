# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# See LICENSE for license information.

import os, sys, time, shutil
import subprocess
import pandas as pd
import numpy as np
import torch
import nvtx
import transformer_engine
from transformer_engine_torch import NVTE_Fused_Attn_Backend

cwd = os.getcwd()
if "TransformerEngine" in cwd:
    index = cwd.index("TransformerEngine") + len("TransformerEngine")
    trimmed_path = cwd[:index]
    sys.path.append(trimmed_path)

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

indices = [model for model in model_configs.keys()]
columns = [
        "FusedAttention Module",
        "FusedAttention Kernels (fwd)",
        "FusedAttention Kernels (bwd)",
        "FusedAttention Kernels (fwd+bwd)",
        "FlashAttention Module",
        "FlashAttention Kernels (fwd)",
        "FlashAttention Kernels (bwd)",
        "FlashAttention Kernels (fwd+bwd)",
        "Fused vs Flash Kernels Speedup (fwd+bwd)",
        "FusedAttention CK Module",
        "FusedAttention CK Kernels (fwd)",
        "FusedAttention CK Kernels (bwd)",
        "FusedAttention CK Kernels (fwd+bwd)",
        "FusedAttention AOTriton Module",
        "FusedAttention AOTriton Kernels (fwd)",
        "FusedAttention AOTriton Kernels (bwd)",
        "FusedAttention AOTriton Kernels (fwd+bwd)",
    ]

output_csv="times.csv"
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
            f""" "import benchmark_attention_rocm;""",
            f"""benchmark_attention_rocm.benchmark_dot_product_attention_profiler("""
            f"""'{model}', '{attention}', '{column_name}')" """,
        ]
    prof_cmd = " ".join(prof_cmd)
    subprocess.call(prof_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    if os.path.exists("results.stats.csv"):
        shutil.move("results.stats.csv", filename)
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

    df_times = pd.read_csv(output_csv, index_col=0)
    df_times.loc[model, column_name] = attn_time * 1e3 / num_iters
    df_times.to_csv(output_csv)
    torch.cuda.empty_cache()

def parse_helper(model, filename, fwd_search_pattern, bwd_search_pattern, column_name, df_times):
    df = pd.read_csv(filename)
    
    t_attn_avg = np.empty(4)
    t_attn_avg[0] = df[df["Name"].str.contains(fwd_search_pattern)]["AverageNs"].to_numpy()
    t_attn_avg[1:4] = df[df["Name"].str.contains(bwd_search_pattern)]["AverageNs"].to_numpy()
    
    df_times.loc[model, f"{column_name} Kernels (fwd)"] = t_attn_avg[0] / 1e6
    df_times.loc[model, f"{column_name} Kernels (bwd)"] = t_attn_avg[1:4].sum() / 1e6
    df_times.loc[model, f"{column_name} Kernels (fwd+bwd)"] = t_attn_avg.sum() / 1e6

# Parser function to parse the results.stats file form rocprof
# This function gathers Avg timing information for both Fwd and Bwd Kernels.
def parse_results(model, df_times, filename_flash_attn, filename_fused_attn, filename_fused_ck, filename_fused_aotriton):
    if os.path.exists(filename_flash_attn):
        parse_helper(model, filename_flash_attn, "FmhaFwd", "FmhaBwd", "FlashAttention", df_times)

    if os.path.exists(filename_fused_attn):
        parse_helper(model, filename_fused_attn, "FmhaFwd", "FmhaBwd", "FusedAttention", df_times)

    if os.path.exists(filename_fused_ck):
        parse_helper(model, filename_fused_ck, "FmhaFwd", "FmhaBwd", "FusedAttention CK", df_times)

    if os.path.exists(filename_fused_aotriton):
        parse_helper(model, filename_fused_aotriton, "attn_fwd", "bwd", "FusedAttention AOTriton", df_times)

    if os.path.exists(filename_flash_attn) and os.path.exists(filename_fused_attn):
        df_times.loc[model, "Fused vs Flash Kernels Speedup (fwd+bwd)"] = (
            df_times.loc[model, "FlashAttention Kernels (fwd+bwd)"]
            / df_times.loc[model, "FusedAttention Kernels (fwd+bwd)"]
        )


def main():
    
    output_dir = "profiler_outputs/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    if (os.path.exists(output_csv)):
        os.remove(output_csv)

    df_times = pd.DataFrame(index=indices, columns=columns)
    df_times.fillna(0.0, inplace=True)
    df_times.index.name = "Model"
    df_times.to_csv(output_csv)
    
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
            print("No attention backend's detected for ", model)
            continue

        print(
            f'Running {model} with {"cuDNN attention" if fused_attn_supported else ""}'
            f'{" and flash-attention" if flash_attn_supported else ""}...'
        )
        
        # Benchmark for each attention backend
        if flash_attn_supported:
            filename_flash_attn = os.path.join("profiler_outputs/", f"prof_flash_{model}.csv")
            benchmark_dot_product_attention(model, "FlashAttention", "FlashAttention Module", filename_flash_attn)
           
        if fused_attn_supported:
            filename_fused_attn = os.path.join("profiler_outputs/", f"prof_fused_{model}.csv")
            benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention Module", filename_fused_attn)
            
            if NVTE_Fused_Attn_Backend.NVTE_CK in fused_attn_backends:
                #CK Backend
                os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "0"
                # os.environ["NVTE_CK_USES_BWD_V3"] = "1"
                os.environ["NVTE_FUSED_ATTN_CK"] = "1"
                os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
                os.environ["NVTE_FUSED_ATTN"] = "0"
                filename_fused_ck = os.path.join("profiler_outputs/", f"prof_fused_ck_{model}.csv")
                benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention CK Module", filename_fused_ck)
            # del os.environ["NVTE_CK_USES_BWD_V3"]

            if NVTE_Fused_Attn_Backend.NVTE_AOTriton in fused_attn_backends:
                #AOTRITON Backend
                os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
                os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "1"
                os.environ["NVTE_FUSED_ATTN_CK"] = "0"
                filename_fused_aotriton = os.path.join("profiler_outputs/", f"prof_fused_aotriton_{model}.csv")
                benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention AOTriton Module", filename_fused_aotriton)

            del os.environ["NVTE_FUSED_ATTN_CK"]
            del os.environ["NVTE_FUSED_ATTN_AOTRITON"]
            del os.environ["NVTE_FUSED_ATTN"]

        df_times = pd.read_csv("times.csv", index_col=0)
        parse_results(model, df_times, filename_flash_attn, filename_fused_attn, filename_fused_ck, filename_fused_aotriton)
        df_times.to_csv("times.csv")

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
