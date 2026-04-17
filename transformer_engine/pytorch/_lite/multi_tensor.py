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
    """Scale tensors by a scalar, matching common/multi_tensor/scale.cu.

    tensor_lists is [in_list, out_list]. For each pair, writes
    out = cast(in * scale, out.dtype). If any element of `in` is non-finite,
    sets noop_flag[0] to 1 (the overflow flag).
    """
    in_list, out_list = tensor_lists[0], tensor_lists[1]
    any_non_finite = False
    for src, dst in zip(in_list, out_list):
        scaled = src.float() * scale
        if not any_non_finite and not torch.isfinite(src).all().item():
            any_non_finite = True
        dst.copy_(scaled.to(dst.dtype))
    if any_non_finite and noop_flag is not None and noop_flag.numel() > 0:
        noop_flag[0] = 1


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
        # Match C++ multi_tensor_apply semantics: the number of elements to
        # process is taken from tensor_lists[0] (grads), and the other tensors
        # are accessed by raw pointer. So m/v/p/pm may be larger than g — e.g.
        # Megatron's distributed optimizer stores m/v for the full vocab while
        # passing only this rank's TP shard as the gradient. Operate on flat
        # views truncated to g.numel().
        n = g.numel()
        g_f = g.reshape(-1).float()
        m_view = m.view(-1)[:n]
        v_view = v.view(-1)[:n]
        p_src_view = (pm if pm is not None else p).view(-1)[:n]
        p_f = p_src_view.float()

        if not adam_w_mode and weight_decay != 0.0:
            # ADAM_MODE_0 (L2): fold weight decay into the gradient
            g_f = g_f + weight_decay * p_f

        m_view.mul_(beta1).add_(g_f.to(m_view.dtype), alpha=1 - beta1)
        v_view.mul_(beta2).addcmul_(g_f.to(v_view.dtype),
                                     g_f.to(v_view.dtype), value=1 - beta2)

        denom = (v_view.float() / bc2).sqrt_().add_(eps)
        update = (m_view.float() / bc1) / denom
        if adam_w_mode and weight_decay != 0.0:
            # ADAM_MODE_1 (decoupled weight decay / AdamW)
            update = update + weight_decay * p_f

        p_new = p_f - lr * update

        if pm is not None:
            pm.view(-1)[:n].copy_(p_new)
            p.view(-1)[:n].copy_(p_new.to(p.dtype))
        else:
            p_src_view.copy_(p_new.to(p.dtype))


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
                     weight_decay, nesterov, first_run, wd_after_momentum, scale=1.0):
    """Fused SGD step mirroring common/multi_tensor/sgd.cu.

    tensor_lists layout:
      3 lists: [grads, weights, momentum_bufs]
      4 lists: [grads, weights, momentum_bufs, weights_fp16_copy]
    Math runs in fp32 on upcast copies; updates are written back in the source dtype.
    """
    if noop_flag is not None and noop_flag.numel() > 0 and noop_flag.item() != 0:
        return
    assert len(tensor_lists) in (3, 4), (
        f"multi_tensor_sgd expects 3 or 4 tensor lists, got {len(tensor_lists)}"
    )
    grads = tensor_lists[0]
    weights = tensor_lists[1]
    mom_bufs = tensor_lists[2]
    fp16_copies = tensor_lists[3] if len(tensor_lists) == 4 else [None] * len(weights)

    for g, w, mom, w_fp16 in zip(grads, weights, mom_bufs, fp16_copies):
        g_f = g.float() * scale
        w_f = w.float()

        if weight_decay != 0.0 and not wd_after_momentum:
            g_f = g_f + weight_decay * w_f

        if momentum != 0.0:
            if first_run:
                mom.copy_(g_f.to(mom.dtype))
            else:
                mom.mul_(momentum).add_(g_f.to(mom.dtype), alpha=1 - dampening)
            mom_f = mom.float()
            g_f = g_f + momentum * mom_f if nesterov else mom_f

        if weight_decay != 0.0 and wd_after_momentum:
            g_f = g_f + weight_decay * w_f

        w_f = w_f - lr * g_f
        w.copy_(w_f.to(w.dtype))
        if w_fp16 is not None:
            w_fp16.copy_(w_f.to(w_fp16.dtype))


def multi_tensor_compute_scale_and_scale_inv(amax_list, scale_list, scale_inv_list,
                                              fp8_max, margin=0):
    """Compute scale and scale_inv from amax."""
    for amax, scale, scale_inv in zip(amax_list, scale_list, scale_inv_list):
        safe_amax = torch.clamp(amax, min=1e-12)
        sf = (fp8_max / safe_amax) / (2 ** margin)
        scale.copy_(sf)
        scale_inv.copy_(1.0 / sf)
