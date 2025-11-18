# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from pathlib import Path
from typing import Optional

import torch
# from safetensors import safe_open  # DISABLED: Not needed for deterministic Hadamard


REPO_PATH = Path(__file__).parent / "hadamards.safetensors"


__all__ = ["deterministic_hadamard_matrix", "is_pow2"]  # random_hadamard_matrix disabled


# note that hadamard matrix multiplication can be accelerated using a library such as
# https://github.com/Dao-AILab/fast-hadamard-transform/tree/master


def deterministic_hadamard_matrix(
    size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Construct an n-by-n Hadamard matrix, using Sylvester's construction.
    `n` must be a power of 2.

    Adapated from https://github.com/scipy/scipy/blob/v1.15.2/scipy/linalg/_special_matrices.py  # noqa: E501

    :param size: order of the matrix, must be a power of 2
    :param dtype: data type of matrix
    :param device: device to construct matrix on
    :return: hadamard matrix of size `size`
    """
    if size <= 0:
        raise ValueError("Cannot construct deterministic hadamard of size <= 0")

    log2 = int(math.log2(size))
    if size != 2**log2:
        raise ValueError("Cannot construct deterministic hadamard of size != 2^n")

    H = torch.tensor([[1]], dtype=dtype, device=device)

    # Sylvester's construction
    for _ in range(log2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))

    return H



def is_pow2(n: int) -> bool:
    """
    Check if a number is a power of 2

    :param n: number to check
    :return: True iff `n` is a power of 2
    """
    return n > 0 and (n & (n - 1) == 0)


#     # normalize
#     return input.view(X.shape)
