#!/usr/bin/python3
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.#
# See LICENSE for license information.

import os
import sys
import argparse

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

import torch
import torch.nn.functional as F
from torch import nn, optim
from contextlib import nullcontext
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = te.Linear(input_size, hidden_size)
        self.fc2 = te.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="FP8 model memory visualization")
    parser.add_argument("--input-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--output-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--fp8-init", action="store_true", default=False)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv, namespace)


def _train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")

    # FP8 Configuration
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

    # Enable memory tracking
    torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='all')

    # Model initialization
    if args.fp8_init:
        from transformer_engine.pytorch import fp8_model_init
        with fp8_model_init(enabled=True):
            model = SimpleNet(args.input_size, args.hidden_size, args.output_size)
    else:
        model = SimpleNet(args.input_size, args.hidden_size, args.output_size)

    model.load_state_dict(torch.load('fsdp_model.pth'))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    from pathlib import Path

    input_path = Path("shared_input.pt")
    if input_path.exists():
        input_data = torch.load(input_path).to(device)
    else:
        input_data = torch.randn(args.batch_size, args.input_size, requires_grad=True).to(device)
        torch.save(input_data.cpu(), input_path)
        print("Generated and saved shared input tensor.")

    out_tensors = []
    for iteration in range(args.iter):
        print(f"Iteration {iteration}")
        optimizer.zero_grad()
        with te.fp8_autocast(enabled=True):
            output = model(input_data)
        target = torch.randn(args.batch_size, args.output_size).to(device)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        print(f"Iteration {iteration} completed.")
        for p in model.parameters():
                if p.requires_grad:
                    out_tensors.append(p.grad)
    torch.save(out_tensors, "all_iters_regular.pt")

    # Save memory snapshot
    snapshot = torch.cuda.memory._snapshot()
    import pickle
    with open('memory_snapshot.pickle', 'wb') as f:
        pickle.dump(snapshot, f)

    torch.cuda.memory._record_memory_history(enabled=None)
    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
