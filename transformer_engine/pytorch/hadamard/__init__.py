"""Hadamard transform utilities."""

from .hadamard import deterministic_hadamard_matrix, is_pow2
# random_hadamard_matrix disabled - requires hadamards.safetensors
from .transform import HadamardFactory, HadamardTransform
from .matrix import multihead_matmul

__all__ = [
    "deterministic_hadamard_matrix",
    # "random_hadamard_matrix",  # disabled
    "is_pow2",
    "HadamardFactory",
    "HadamardTransform",
    "multihead_matmul",
]
