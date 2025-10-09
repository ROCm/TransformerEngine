#!/usr/bin/python3
# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import os
import sys
import argparse

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling, Format, DelayedScaling

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from transformer_engine.pytorch import torch_version
from transformer_engine.pytorch.fp8 import fp8_model_init
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_fsdp2=False):
        super(SimpleNet, self).__init__()
        self.fc1 = te.Linear(input_size, hidden_size, use_fsdp2=use_fsdp2)
        self.fc2 = te.Linear(hidden_size, output_size, use_fsdp2=use_fsdp2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items()}
    return custom_attrs


def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Toy example for debugging fully_shard()")
    parser.add_argument("--input-size", type=int, default=2048, help="Input size for the model")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden layer size")
    parser.add_argument("--output-size", type=int, default=2048, help="Output size for the model")
    parser.add_argument("--batch-size", type=int, default=2048, help="Output size for the model")
    parser.add_argument(
        "--fp8-init", action="store_true", default=False, help="Initialize primary weights in FP8."
    )
    parser.add_argument(
        "--iter", type=int, default=10, help="Number of iterations for forward pass"
    )
    parser.add_argument('--profile', action='store_true',
                       help='Enable pytorch profiling.')
    parser.add_argument('--profile-step-start', type=int, default=6,
                       help='Global step to start profiling.')
    parser.add_argument('--profile-step-end', type=int, default=7,
                       help='Global step to stop profiling.')
    parser.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')
    parser.add_argument('--tensorboard-dir', type=str, default='./fsdp2_tensorboard',
                       help='Write TensorBoard logs to this directory.')
    parser.add_argument('--gradients-save-file', type=str, default='all_iters.pt',
                       help='Write all the gradients across all the iterations to this file.')
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--use-fsdp2", action='store_true',
                       help='Enable New FSDP2 training.')
    # Adding hsdp_dim as a list argument, comma-separated
    parser.add_argument(
        "--sharding-dims",
        type=int,
        nargs="+",
        help='FSDP/HSDP sharding dimensions ("replicate", "shard")',
    )
    args = parser.parse_args(argv, namespace)
    if args.sharding_dims:
        assert len(args.sharding_dims) <= 2
    return args


sub_modules_to_wrap = [te.Linear]


def _train(args):
    assert "TORCHELASTIC_RUN_ID" in os.environ
    WORLD_RANK = int(os.getenv("RANK", "0"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    assert LOCAL_SIZE == WORLD_SIZE

    # Set device and initialize RNG states
    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize torch.distributed global process group and get DP/TP groups
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
    }
    assert dist.is_nccl_available()
    dist.init_process_group(**dist_init_kwargs)
    nccl_world = dist.new_group(backend="nccl")
    device = torch.device(f"cuda:{LOCAL_RANK}")

    # FP8 Configuration
    fp8_format = Format.HYBRID
    fp8_recipe = Float8CurrentScaling(fp8_format=fp8_format)

    if args.fp8_init:
        # Build the model with the specified context
        with fp8_model_init(enabled = True):
            model = SimpleNet(args.input_size, args.hidden_size, args.output_size, use_fsdp2=args.use_fsdp2)
    else:
        model = SimpleNet(args.input_size, args.hidden_size, args.output_size, use_fsdp2=args.use_fsdp2)
    # Move the model to the correct device
    model.load_state_dict(torch.load('fsdp_model.pth'))
    model.to(device)

    # Creating a DeviceMesh for fully_shard
    world_size = int(WORLD_SIZE)
    device_ids = list(range(world_size))

    # Apply FSDP/HSDP
    if args.use_fsdp2:
        custom_attrs = save_custom_attrs(model)
        if LOCAL_RANK == 0:
            print(f"Rank {LOCAL_RANK}: Applying FSDP fully_shard() to the model...")
            print(f"sharding-dims:{args.sharding_dims}")
        # Setup the sharding mesh for FSDP/HSDP
        if args.sharding_dims == None:  # FSDP
            mesh = DeviceMesh("cuda", device_ids)
        elif len(args.sharding_dims) == 1:
            assert args.sharding_dims[0] == device_ids[-1] + 1
            mesh = DeviceMesh("cuda", device_ids)
        elif len(args.sharding_dims) == 2:  # HSDP
            assert args.sharding_dims[0] * args.sharding_dims[1] == device_ids[-1] + 1
            mesh = init_device_mesh(
                "cuda",
                (args.sharding_dims[0], args.sharding_dims[1]),
                mesh_dim_names=("replicate", "shard"),
            )
        else:
            assert False
        for sub_module in model.modules():
            if any(
                isinstance(sub_module, sub_module_to_wrap) for sub_module_to_wrap in sub_modules_to_wrap
            ):
                fully_shard(sub_module, mesh=mesh)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        restore_custom_attrs(model, custom_attrs)
    else:
        model = DDP(model, device_ids=[LOCAL_RANK])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    input_path = Path("shared_input.pt")
    if input_path.exists():
        input_data = torch.load(input_path).to(device)
    else:
        input_data = torch.randn(args.batch_size, args.input_size, requires_grad=True).to(device)
        torch.save(input_data.cpu(), input_path)
        print("Generated and saved shared input tensor.")
    
    out_tensors = []
    prof = None
    if (
        args.profile
        and torch.distributed.get_rank() in args.profile_ranks
    ):
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=max(args.profile_step_start - 1, 0),
                warmup=1 if args.profile_step_start > 0 else 0,
                active=args.profile_step_end - args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
    for iteration in range(args.iter):
        if LOCAL_RANK == 0:
            print(f"Starting iteration...{iteration}")
        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
            prof.step()

        # Zero the parameter gradients
        optimizer.zero_grad()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            output = model(input_data)
        target = torch.randn(args.batch_size, args.output_size).to(device)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if LOCAL_RANK == 0:
            print(f"Rank {LOCAL_RANK}: Iteration {iteration} completed.")
        with torch.no_grad():
            for name, p in model.named_parameters():
                full_grad = None
                if p.grad is not None and hasattr(p.grad, 'full_tensor'):
                    # This call is required to be executed on ALL ranks
                    # to complete the collective communication.
                    full_grad = p.grad.full_tensor().detach().clone()
                elif p.grad is not None:
                    full_grad = p.grad.detach().clone()
                # 2. Only Rank 0 stores the result
                if LOCAL_RANK == 0 and p.requires_grad:
                    out_tensors.append((name, full_grad))
    if (
        args.profile
        and iteration == args.profile_step_end
        and torch.distributed.get_rank() in args.profile_ranks
    ):
        prof.stop()

    if LOCAL_RANK == 0:
        torch.save(out_tensors, args.gradients_save_file)

    # NOTE: In PyTorch < 2.6 there’s a teardown race where one rank may call
    # destroy_process_group() while other ranks still have in-flight NCCL ops,
    # which can trigger a NCCL/RCCL comm error. Newer releases (>= 2.6) fixed
    # this, but we kept a version-guarded barrier on older Torch for stability.
    if torch_version() < (2, 6, 0):
        dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
