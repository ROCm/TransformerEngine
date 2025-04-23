# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused Adam optimizer."""
import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import is_fp8_fnuz
from .multi_tensor_apply import multi_tensor_applier
from ..float8_tensor import Float8Tensor


def get_fp8_meta(fp8_tensor):
    """FP8 metadata getter."""
    if fp8_tensor._fp8_meta is None:
        raise RuntimeError("FP8 meta data is not initialized.")

    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
        forward=fp8_tensor._fp8_meta_forward,
    )

    fp8_meta_index = fp8_tensor._fp8_meta_index
    scale = fp8_tensor._fp8_meta[fp8_meta_key].scale[fp8_meta_index]
    amax = fp8_tensor._fp8_meta[fp8_meta_key].amax_history[0][fp8_meta_index]
    scale_inv = fp8_tensor._scale_inv
    return scale, amax, scale_inv


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to
        all the model's parameters into one or a few kernel launches.

    :class:`te.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = te.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`te.optimizers.FusedAdam` may be used with or without Amp.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        bias_correction (bool, optional): apply correction factor to
            moment estimates. (default: True)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training, currently can
           only be used with capturable set to True. (default: False)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        set_grad_none=True,
        capturable=False,
        master_weights=False,
    ):

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        if master_weights and not capturable:
            raise RuntimeError(
                "Master weights is currently only supported with the capturable version."
            )
        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        self.capturable = capturable
        self.master_weights = master_weights

        # Create full precision master weights
        self.param_groups_master = []
        for _, pg in enumerate(self.param_groups):
            param_list = pg["params"]
            self.param_groups_master.append(
                {
                    "params": [
                        p.clone().detach().float() if self.master_weights else None
                        for p in param_list
                    ],
                }
            )

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                for item in ["lr"]:
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        # Skip buffer
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self.multi_tensor_adam = tex.multi_tensor_adam
        self.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

        self.master_weight_dtype = master_weight_dtype
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_sq_dtype
        self.name_to_dtype_map = {
            "exp_avg": self.exp_avg_dtype,
            "exp_avg_sq": self.exp_avg_sq_dtype,
            "master_param": self.master_weight_dtype,
        }
        self.dtype_to_range_map = {
            torch.float16: torch.full(
                [1], torch.finfo(torch.float16).max / 2.0, dtype=torch.float32
            ),
            torch.uint8: torch.full([1], 240.0 if is_fp8_fnuz() else 448.0, dtype=torch.float32),
        }
        self._scales = {}
        self.use_decoupled_grad = use_decoupled_grad
        # Works only when master params is in FP32
        self.store_param_remainders = (
            store_param_remainders and master_weights and master_weight_dtype == torch.float32
        )

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super().zero_grad()

    def step(self, closure=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grad_scaler (torch.cuda.amp.GradScaler, optional):
                gradient scaler (default: None)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += (
                    1 if not self.capturable else (self._dummy_overflow_buf != 1).to(torch.int)
                )
            else:
                group["step"] = (
                    1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)
                )

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []

            for p, p_master in zip(group["params"], group_master["params"]):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).float()

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(p_master.data)
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state["exp_avg"])
                    v_16.append(state["exp_avg_sq"])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state["exp_avg"])
                    v_bf.append(state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(p_master.data)
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state["exp_avg"])
                    v_32.append(state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedAdam only support fp16 and fp32.")

            # If the optimizer is capturable, then if there's a grad scaler it works
            # on the GPU + a different multi_tensor_applier should be called
            if self.capturable:
                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None
                    else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device)
                    inv_scale = torch.ones((1,), device=device)

                if len(g_16) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_16, p_16, m_16, v_16, p_16_master]
                            if self.master_weights
                            else [g_16, p_16, m_16, v_16]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam_capturable,
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_32, p_32, m_32, v_32, p_32_master]
                            if self.master_weights
                            else [g_32, p_32, m_32, v_32]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )
            else:
                if len(g_16) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss
