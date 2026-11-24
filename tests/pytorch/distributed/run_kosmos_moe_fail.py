# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""One rank fails with an exception unrelated to the fused MoE while a workspace is live (no finalize): the job must
end promptly (no collective at interpreter exit waiting for the other ranks)."""

import os
import sys

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te


def main() -> int:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    rank = dist.get_rank()
    T, H = 1024, 2048
    ws = te.MoEWorkspace(hidden_size=H, ffn_hidden_size=256, num_experts=16, topk=8, num_tokens=T,
                         ep_group=dist.group.WORLD, capacity_rows=T * 8 + 512, router="softmax",
                         shared_expert=True)
    m = te.MoE(ws)
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    out = m(x)[0]
    out.backward(torch.ones_like(out))
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        raise ValueError("an error unrelated to the fused MoE")
    dist.barrier()  # the other ranks wait for rank 0, as a training step's next collective would
    return 0


if __name__ == "__main__":
    sys.exit(main())
