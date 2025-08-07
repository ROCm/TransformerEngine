# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
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

# Add test_fused_attn to the sys path 
tests_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../tests/pytorch/fused_attn")
)
sys.path.append(tests_path)

from test_fused_attn import (
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

# Define DataFrame indices and columns
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
cwd = os.getcwd()
# Runs benchmark with warmup iterations and profiles using rocprof
def benchmark_dot_product_attention(model, attention, column_name, dirname):
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
    os.makedirs(dirname)
    before_files = set(os.listdir("."))
    # Profiling command using rocprof
    prof_cmd = [
            "env | grep NVTE; "
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
    after_files = set(os.listdir("."))
    new_files = after_files - before_files

    for f in new_files:
        shutil.move(f, os.path.join(dirname, f))
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

    df_times = pd.read_csv(output_csv, index_col=0)
    df_times.loc[model, column_name] = attn_time * 1e3 / num_iters
    df_times.to_csv(output_csv)
    torch.cuda.empty_cache()

# Helper function to extract timing results from profiler logs
def parse_helper(model, dirname, fwd_search_pattern, bwd_search_pattern, column_name, df_times):
    df = pd.read_csv(os.path.join(dirname,"results.stats.csv"))

    # Extract kernel timing values    
    fwd_values = df[df["Name"].str.contains(fwd_search_pattern)]["AverageNs"].to_numpy()
    bwd_values = df[df["Name"].str.contains(bwd_search_pattern)]["AverageNs"].to_numpy()

    if len(bwd_values) == 0:
        return False  # CK V3 not supported or kernel_func not found
    
    t_attn_avg = np.empty(len(fwd_values) + len(bwd_values))
    t_attn_avg[:len(fwd_values)] = fwd_values
    t_attn_avg[len(fwd_values):] = bwd_values

    # Store results in DataFrame
    df_times.loc[model, f"{column_name} Kernels (fwd)"] = t_attn_avg[:len(fwd_values)].sum() / 1e6
    df_times.loc[model, f"{column_name} Kernels (bwd)"] = t_attn_avg[len(fwd_values):].sum() / 1e6
    df_times.loc[model, f"{column_name} Kernels (fwd+bwd)"] = t_attn_avg.sum() / 1e6

    return True

# Parses profiler logs for different attention backends
def parse_results(model, df_times, perf_dir_flash_attn, perf_dir_fused_attn, perf_dir_fused_ck, perf_dir_fused_aotriton, use_ck_bwd_v3):
    if perf_dir_flash_attn:
        parse_helper(model, perf_dir_flash_attn, "FmhaFwd", "FmhaBwd", "FlashAttention", df_times)

    if perf_dir_fused_attn:
        ck_v3_success = False
        if use_ck_bwd_v3:
            ck_v3_success = parse_helper(model, perf_dir_fused_ck, "FmhaFwd", "kernel_func", "FusedAttention", df_times)
        if not ck_v3_success:
            parse_helper(model, perf_dir_fused_ck, "FmhaFwd", "FmhaBwd", "FusedAttention", df_times)

    if perf_dir_fused_attn:
        ck_v3_success = False
        if use_ck_bwd_v3:
            ck_v3_success = parse_helper(model, perf_dir_fused_ck, "FmhaFwd", "kernel_func", "FusedAttention CK", df_times)
        if not ck_v3_success:
            parse_helper(model, perf_dir_fused_ck, "FmhaFwd", "FmhaBwd", "FusedAttention CK", df_times)
    
    if perf_dir_fused_aotriton:
        parse_helper(model, perf_dir_fused_aotriton, "attn_fwd", "bwd", "FusedAttention AOTriton", df_times)

    if perf_dir_flash_attn and perf_dir_fused_attn:
        df_times.loc[model, "Fused vs Flash Kernels Speedup (fwd+bwd)"] = (
            df_times.loc[model, "FlashAttention Kernels (fwd+bwd)"]
            / df_times.loc[model, "FusedAttention Kernels (fwd+bwd)"]
        )

###############################################################################
# Post-benchmark sanity checks
###############################################################################
def sanity_checks(
    profiler_root: str = "profiler_outputs",
    csv_path: str = "times.csv",
    tolerance_pct: float = 5.0,
):
    """
    • Verifies that every model/backend that *should* have run produced
        profiler_root/<dir>/results.stats.csv
    • Checks FusedAttention vs FusedAttention-CK timing within ±tolerance_pct
    • Non-zero exit code on any failure (CI friendly)
    """
    print("\n============= Sanity-check results =============")
    ok_overall = True
    times_csv_path = os.path.join(cwd, csv_path)
    df = pd.read_csv(times_csv_path, index_col=0)
    
    tol = tolerance_pct / 100.0
    profiler_root = os.path.join(cwd, profiler_root)

    dir_pattern = {
        "FlashAttention":           "prof_flash_{model}",
        "FusedAttention":           "prof_fused_{model}",
        "FusedAttention CK":        "prof_fused_ck_{model}",
        "FusedAttention AOTriton":  "prof_fused_aotriton_{model}",
    }

    for model, cfg in model_configs.items():
        avail, _, fused_bes = _get_attention_backends(
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
            expected["FusedAttention"] = dir_pattern["FusedAttention"]
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

        # Fused Vs Fused CK trace
        if "FusedAttention" in expected and "FusedAttention CK" in expected:
            f_fwd, f_bwd = df.loc[model, ["FusedAttention Kernels (fwd)",
                                          "FusedAttention Kernels (bwd)"]]
            c_fwd, c_bwd = df.loc[model, ["FusedAttention CK Kernels (fwd)",
                                          "FusedAttention CK Kernels (bwd)"]]
            if min(f_fwd, f_bwd, c_fwd, c_bwd) > 0:
                rel_fwd = abs(f_fwd - c_fwd) / max(f_fwd, c_fwd)
                rel_bwd = abs(f_bwd - c_bwd) / max(f_bwd, c_bwd)
                if rel_fwd < tol and rel_bwd < tol:
                    print(f"  [OK ] Fused vs CK diff <= {tolerance_pct}% "
                          f"(fwd {rel_fwd*100:.2f} %, bwd {rel_bwd*100:.2f} %)")
                else:
                    ok_overall = False
                    raise AssertionError(f" Fused vs CK kernel time diff > {tolerance_pct}% "
                          f"(fwd {rel_fwd*100:.2f} %, bwd {rel_bwd*100:.2f} %)")
        print("-" * 60)
    return ok_overall


def main(args):
    output_dir = "profiler_outputs/"
    run_dir = os.path.dirname(__file__)

    # Remove from current working directory
    if os.path.exists(os.path.join(cwd, output_dir)):
        shutil.rmtree(os.path.join(cwd, output_dir))
    if os.path.exists(os.path.join(cwd, output_csv)):
        os.remove(os.path.join(cwd, output_csv))

    # Remove from run directory
    if os.path.exists(os.path.join(run_dir, output_dir)):
        shutil.rmtree(os.path.join(run_dir, output_dir))
    if os.path.exists(os.path.join(run_dir, output_csv)):
        os.remove(os.path.join(run_dir, output_csv))

    os.chdir(run_dir)
    os.makedirs(output_dir)

    df_times = pd.DataFrame(index=indices, columns=columns)
    df_times = df_times.infer_objects(copy=False)
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
        available_backends,_, fused_attn_backends = _get_attention_backends(
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

        perf_dir_flash_attn, perf_dir_fused_attn, perf_dir_fused_ck, perf_dir_fused_aotriton = None, None, None, None
        
        # Benchmark for each attention backend
        if flash_attn_supported:
            os.environ.update({
                "NVTE_FUSED_ATTN": "0", "NVTE_FLASH_ATTN": "1",
                "NVTE_FUSED_ATTN_AOTRITON": "0", "NVTE_FUSED_ATTN_CK": "0" , "NVTE_UNFUSED_ATTN": "0"
            })
            perf_dir_flash_attn = os.path.join("profiler_outputs/", f"prof_flash_{model}")
            benchmark_dot_product_attention(model, "FlashAttention", "FlashAttention Module", perf_dir_flash_attn)
           
        if fused_attn_supported:

            os.environ.update({
                "NVTE_FUSED_ATTN": "1", "NVTE_FLASH_ATTN": "0",
                "NVTE_FUSED_ATTN_AOTRITON": "0", "NVTE_FUSED_ATTN_CK": "1", "NVTE_UNFUSED_ATTN": "0"
            })
            if args.use_ck_bwd_v3:
                os.environ["NVTE_CK_USES_BWD_V3"] = "1"
            
            # FusedAttention run
            perf_dir_fused_attn = os.path.join("profiler_outputs/", f"prof_fused_{model}")
            benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention Module", perf_dir_fused_attn)
            
            #FusedAttention CK run
            if NVTE_Fused_Attn_Backend.NVTE_CK in fused_attn_backends:
                perf_dir_fused_ck = os.path.join("profiler_outputs/", f"prof_fused_ck_{model}")
                benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention CK Module", perf_dir_fused_ck)

            if NVTE_Fused_Attn_Backend.NVTE_AOTriton in fused_attn_backends:
                #AOTRITON Backend
                os.environ.update({
                    "NVTE_FUSED_ATTN_AOTRITON": "1", "NVTE_FUSED_ATTN_CK": "0",
                    "NVTE_CK_USES_BWD_V3": "0", "NVTE_UNFUSED_ATTN": "0"
                })
                perf_dir_fused_aotriton = os.path.join("profiler_outputs/", f"prof_fused_aotriton_{model}")
                benchmark_dot_product_attention(model, "FusedAttention", "FusedAttention AOTriton Module", perf_dir_fused_aotriton)

            for var in ["NVTE_CK_USES_BWD_V3", "NVTE_FUSED_ATTN_AOTRITON", "NVTE_FUSED_ATTN_CK", "NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"]:
                os.environ.pop(var, None)

        df_times = pd.read_csv("times.csv", index_col=0)
        parse_results(model, df_times, perf_dir_flash_attn, perf_dir_fused_attn, perf_dir_fused_ck, perf_dir_fused_aotriton, args.use_ck_bwd_v3)
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
    if cwd != run_dir:
        final_profiler_dir = os.path.join(cwd, "profiler_outputs")
        if os.path.exists(final_profiler_dir):
            shutil.rmtree(final_profiler_dir)
        shutil.move("profiler_outputs", final_profiler_dir)

        final_csv_path = os.path.join(cwd, output_csv)
        if os.path.exists(final_csv_path):
            os.remove(final_csv_path)
        shutil.move(output_csv, final_csv_path)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ck_bwd_v3", action="store_true", help="Use NVTE_CK_USES_BWD_V3=1 for CK bwd kernels")
    parser.add_argument("--run_sanity_checks", action="store_true", help="After benchmarking, verify profiler outputs and Fused vs CK timing parity")
    args = parser.parse_args()
    main(args)
    if args.run_sanity_checks:
        sanity_checks()
