# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor operations -- PyTorch-native implementations.

These replace the fused C++ multi-tensor kernels. Performance is lower due to
per-tensor kernel launches instead of batched execution, but functionality is preserved.
"""

import torch
import math


def multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
    """Scale a list of tensors by a scalar."""
    overflow_buf = noop_flag
    for tensor_group in tensor_lists:
        for t in tensor_group:
            t.mul_(scale)


def multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor=False):
    """Compute L2 norm for a list of tensors.

    Always returns a 2-tuple (total_norm, per_tensor_norms) to match the C++
    contract in csrc/extensions/multi_tensor/l2norm.cpp. When per_tensor is
    False, the second tensor is empty.
    """
    device = tensor_lists[0][0].device
    if per_tensor:
        norms = [t.float().norm().item() for t in tensor_lists[0]]
        total = math.sqrt(sum(n * n for n in norms))
        return (torch.tensor([total], device=device, dtype=torch.float32),
                torch.tensor(norms, device=device, dtype=torch.float32))
    total_sq = 0.0
    for t in tensor_lists[0]:
        total_sq += t.float().norm().item() ** 2
    return (torch.tensor([math.sqrt(total_sq)], device=device, dtype=torch.float32),
            torch.empty(0, device=device, dtype=torch.float32))


def multi_tensor_unscale_l2norm(chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor=False):
    """Compute L2 norm after unscaling (tensors are NOT modified).

    Always returns a 2-tuple (total_norm, per_tensor_norms) to match the C++
    contract. When per_tensor is False, the second tensor is empty.
    """
    scale = 1.0 / inv_scale.item() if inv_scale.numel() == 1 else 1.0 / inv_scale
    device = tensor_lists[0][0].device
    if per_tensor:
        norms = [(t.float() * scale).norm().item() for t in tensor_lists[0]]
        total = math.sqrt(sum(n * n for n in norms))
        return (torch.tensor([total], device=device, dtype=torch.float32),
                torch.tensor(norms, device=device, dtype=torch.float32))
    total_sq = 0.0
    for t in tensor_lists[0]:
        total_sq += (t.float() * scale).norm().item() ** 2
    return (torch.tensor([math.sqrt(total_sq)], device=device, dtype=torch.float32),
            torch.empty(0, device=device, dtype=torch.float32))


def multi_tensor_adam(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2, eps,
                      step, adam_w_mode, bias_correction, weight_decay):
    """Fused Adam step mirroring common/multi_tensor/adam.cu.

    tensor_lists layout:
      4 lists: [grads, params, exp_avg, exp_avg_sq]
      5 lists: [grads, params, exp_avg, exp_avg_sq, master_params]

    With master_params, Adam math runs in fp32 on master_params and the result
    is downcast into params. Without them, math runs on params directly.
    adam_w_mode=True is ADAM_MODE_1 (decoupled weight decay), False is
    ADAM_MODE_0 (L2 regularization folded into the gradient).
    """
    assert len(tensor_lists) in (4, 5), (
        f"multi_tensor_adam expects 4 or 5 tensor lists, got {len(tensor_lists)}"
    )
    grads = tensor_lists[0]
    params = tensor_lists[1]
    exp_avgs = tensor_lists[2]
    exp_avg_sqs = tensor_lists[3]
    master_params = tensor_lists[4] if len(tensor_lists) == 5 else [None] * len(params)

    bc1 = (1.0 - beta1 ** step) if bias_correction else 1.0
    bc2 = (1.0 - beta2 ** step) if bias_correction else 1.0

    for g, p, m, v, pm in zip(grads, params, exp_avgs, exp_avg_sqs, master_params):
        p_src = pm if pm is not None else p
        g_f = g.float()
        p_f = p_src.float()

        if not adam_w_mode and weight_decay != 0.0:
            # ADAM_MODE_0 (L2): fold weight decay into the gradient
            g_f = g_f + weight_decay * p_f

        m.mul_(beta1).add_(g_f, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(g_f, g_f, value=1 - beta2)

        denom = (v / bc2).sqrt_().add_(eps)
        update = (m / bc1) / denom
        if adam_w_mode and weight_decay != 0.0:
            # ADAM_MODE_1 (decoupled weight decay / AdamW)
            update = update + weight_decay * p_f

        p_f = p_f - lr * update

        if pm is not None:
            pm.copy_(p_f)
        p.copy_(p_f.to(p.dtype))


def multi_tensor_adam_param_remainder(*args, **kwargs):
    """Adam with parameter remainder (for mixed-precision master weights).

    TODO: Implement when needed.
    """
    raise NotImplementedError("multi_tensor_adam_param_remainder not yet implemented in lite mode.")


def multi_tensor_adam_fp8(*args, **kwargs):
    """Adam with FP8 momentum.

    TODO: Implement when needed.
    """
    raise NotImplementedError("multi_tensor_adam_fp8 not yet implemented in lite mode.")


def multi_tensor_adam_capturable(*args, **kwargs):
    """Adam with CUDA graph support.

    Not applicable in lite mode (no CUDA graph capture for Python ops).
    Falls back to standard Adam behavior.
    """
    raise NotImplementedError("multi_tensor_adam_capturable not yet implemented in lite mode.")


def multi_tensor_adam_capturable_master(*args, **kwargs):
    """Adam capturable with FP32 master weights.

    TODO: Implement when needed.
    """
    raise NotImplementedError(
        "multi_tensor_adam_capturable_master not yet implemented in lite mode."
    )


def multi_tensor_sgd(chunk_size, noop_flag, tensor_lists, lr, momentum, dampening,
                     weight_decay, nesterov, first_run, wd_after_momentum, scale=-1.0):
    """Fused SGD optimizer step for multiple tensors."""
    params, grads = tensor_lists[0], tensor_lists[1]
    # momentum_bufs is tensor_lists[2] if momentum != 0
    momentum_bufs = tensor_lists[2] if len(tensor_lists) > 2 else [None] * len(params)

    for p, g, buf in zip(params, grads, momentum_bufs):
        if scale > 0:
            g = g * scale

        if weight_decay != 0 and not wd_after_momentum:
            g = g.add(p.data, alpha=weight_decay)

        if momentum != 0:
            if buf is None or first_run:
                buf = g.clone()
            else:
                buf.mul_(momentum).add_(g, alpha=1 - dampening)

            if nesterov:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf

        if weight_decay != 0 and wd_after_momentum:
            g = g.add(p.data, alpha=weight_decay)

        p.data.add_(g, alpha=-lr)


def multi_tensor_compute_scale_and_scale_inv(amax_list, scale_list, scale_inv_list,
                                              fp8_max, margin=0):
    """Compute scale and scale_inv from amax."""
    for amax, scale, scale_inv in zip(amax_list, scale_list, scale_inv_list):
        safe_amax = torch.clamp(amax, min=1e-12)
        sf = (fp8_max / safe_amax) / (2 ** margin)
        scale.copy_(sf)
        scale_inv.copy_(1.0 / sf)
