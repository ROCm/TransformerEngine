# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information
"""KOSMOS's W1 layout: GLU interleave 128 (TE ``glu_interleave_size=128``).

``W1`` is ``[E_loc, 2I, H]``; KOSMOS stores its rows in blocks of 256, 128 gate rows followed by the
matching 128 up rows. MegaMoE and an unfused reference store all ``I`` gate rows, then all ``I`` up
rows (``[gate; up]``). Convert once when the weights are loaded, never per call.
"""

import torch

GLU_INTERLEAVE = 128

__all__ = ["GLU_INTERLEAVE", "to_kosmos_w1", "from_kosmos_w1"]


def _split(w: torch.Tensor):
    if w.dim() < 2:
        raise ValueError(f"W1 must be [..., 2I, H], got shape {tuple(w.shape)}")
    two_i = w.shape[-2]
    if two_i % (2 * GLU_INTERLEAVE) != 0:
        raise ValueError(
            f"W1 has 2I = {two_i} rows; KOSMOS needs a multiple of {2 * GLU_INTERLEAVE}"
        )
    return tuple(w.shape[:-2]), two_i, w.shape[-1], two_i // (2 * GLU_INTERLEAVE)


def to_kosmos_w1(w1: torch.Tensor) -> torch.Tensor:
    """``[..., 2I, H]`` in ``[gate; up]`` order to KOSMOS's GLU-interleave-128 order (a new
    tensor)."""
    lead, two_i, h, nb = _split(w1)
    w = w1.reshape(*lead, 2, nb, GLU_INTERLEAVE, h).transpose(-4, -3)
    return w.reshape(*lead, two_i, h).contiguous()


def from_kosmos_w1(w1: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_kosmos_w1`: KOSMOS order back to ``[gate; up]`` (a new tensor)."""
    lead, two_i, h, nb = _split(w1)
    w = w1.reshape(*lead, nb, 2, GLU_INTERLEAVE, h).transpose(-4, -3)
    return w.reshape(*lead, two_i, h).contiguous()
