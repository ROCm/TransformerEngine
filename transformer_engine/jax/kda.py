# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""Kimi Delta Attention (KDA) for JAX.

The Triton-backed implementation lives in
``transformer_engine.jax.triton_extensions.kda`` and shares its kernels with the
PyTorch version (``transformer_engine.pytorch.triton.kda``). It is imported
lazily because it requires ``triton``.
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

__all__ = ["kimi_delta_attn"]


class _KDAConfig(NamedTuple):
    scale: float
    output_final_state: bool
    max_seqlen: Optional[int]
    chunk_size: Optional[int]
    safe_gate: bool
    lower_bound: Optional[float]
    use_gate_in_kernel: bool
    use_qk_l2norm_in_kernel: bool
    use_beta_sigmoid_in_kernel: bool
    state_v_first: bool


def _kda_fwd_impl(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, cfg):
    from .triton_extensions.kda import kda_fwd  # pylint: disable=import-outside-toplevel

    return kda_fwd(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, **cfg._asdict())


@partial(jax.custom_vjp, nondiff_argnums=(9,))
def _kda(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, cfg):
    return _kda_fwd_impl(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, cfg)


def _kda_vjp_fwd(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, cfg):
    out = _kda_fwd_impl(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, cfg)
    # Only the inputs are saved; the backward recomputes the intermediates.
    return out, (q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens)


def _kda_vjp_bwd(cfg, res, grads):
    from .triton_extensions.kda import kda_bwd  # pylint: disable=import-outside-toplevel

    q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens = res
    do, dht = grads
    dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0 = kda_bwd(
        do.astype(v.dtype),
        dht,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        **cfg._asdict(),
    )
    return dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0, None


_kda.defvjp(_kda_vjp_fwd, _kda_vjp_bwd)


def kimi_delta_attn(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    A_log: Optional[jnp.ndarray] = None,
    dt_bias: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
    chunk_size: Optional[int] = None,
    cu_seqlens: Optional[jnp.ndarray] = None,
    max_seqlen: Optional[int] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    r"""Chunked Kimi Delta Attention, differentiable w.r.t. every float input.

    Same arguments and semantics as
    ``transformer_engine.pytorch.triton.kda.kimi_delta_attn``; all non-array
    arguments are static. Traceable under ``jax.jit``, including with
    ``cu_seqlens`` (whose values stay on the device; ``max_seqlen``, when given,
    must bound the longest sequence). Tokens past ``cu_seqlens[-1]`` are zero
    in the output.

    Returns:
        ``(o, final_state)``: ``o`` is ``[B, T, HV, V]`` in ``q``'s dtype.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} vs {k.shape}.")
    if K > 256:
        raise ValueError(f"Only key head dim <= 256 is supported, got {K}.")
    if HV % H != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={HV}) must be divisible by num_qk_heads (H={H})."
        )
    if tuple(g.shape) != (B, T, HV, K):
        raise ValueError(f"g must have shape {(B, T, HV, K)}, got {tuple(g.shape)}.")
    if tuple(beta.shape) != (B, T, HV):
        raise ValueError(f"beta must have shape {(B, T, HV)}, got {tuple(beta.shape)}.")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`."
            )
        if initial_state is not None and initial_state.shape[0] != cu_seqlens.shape[0] - 1:
            raise ValueError(
                "The number of initial states is expected to be equal to the number of "
                f"input sequences, {cu_seqlens.shape[0] - 1}, not {initial_state.shape[0]}."
            )
    if initial_state is not None and not jnp.issubdtype(initial_state.dtype, jnp.floating):
        raise ValueError(f"`initial_state` must be a float array, got {initial_state.dtype}.")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError(
                "`lower_bound` must be specified when `safe_gate=True` and "
                "`use_gate_in_kernel=True`."
            )
        if not -5 <= lower_bound < 0:
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")
    if scale is None:
        scale = K**-0.5
    elif scale <= 0:
        raise ValueError(f"`scale` must be positive, got {scale}.")

    cfg = _KDAConfig(
        scale=float(scale),
        output_final_state=output_final_state,
        max_seqlen=max_seqlen,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first,
    )
    o, final_state = _kda(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, cfg)
    return o.astype(q.dtype), final_state
