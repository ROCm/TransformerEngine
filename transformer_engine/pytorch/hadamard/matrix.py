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

from enum import Enum
import torch


__all__ = ["TransformLocation", "apply_transform_weight", "multihead_matmul"]


class TransformLocation(str, Enum):
    """
    Enum representing which parameters/activations a transform weight should be applied
    to on a given module.

    | -------------------------------------------------------------------------------------------------------- |  # noqa: E501
    | Name            | Runtime     | Values        | Locations Where Inverse Could Be Applied                 |  # noqa: E501
    | --------------- | ----------- | ------------- | -------------------------------------------------------- |  # noqa: E501
    | `INPUT`         | online      | activations   | `prev.WEIGHT_OUTPUT`, `prev.OUTPUT`, `this.WEIGHT_INPUT` |  # noqa: E501
    | `WEIGHT_INPUT`  | offline     | weight        | `prev.WEIGHT_OUTPUT`, `prev.OUTPUT`, `this.INPUT`        |  # noqa: E501
    | `WEIGHT_OUTPUT` | offline     | weight        | `this.OUTPUT`, `next.INPUT`, `next.WEIGHT_INPUT`         |  # noqa: E501
    | `OUTPUT`        | online      | activations   | `this.WEIGHT_OUTPUT`, `next.INPUT`, `next.WEIGHT_INPUT`  |  # noqa: E501
    | `K_CACHE`       | online      | key_values    | `q_proj.Q_ATTN`                                          |  # noqa: E501
    | `Q_ATTN`        | online      | query_values  | `k_proj.K_CACHE`                                         |  # noqa: E501
    | -------------------------------------------------------------------------------------------------------- |  # noqa: E501
    """

    INPUT = "input"
    WEIGHT_INPUT = "weight_input"
    WEIGHT_OUTPUT = "weight_output"
    OUTPUT = "output"
    K_CACHE = "k_cache"
    Q_ATTN = "q_attn"


def apply_transform_weight(
    transform_weight: torch.Tensor,
    value: torch.Tensor,
    location: TransformLocation,
    module_type: type[torch.nn.Module],
) -> torch.Tensor:
    """
    Using the transform location, apply the transform_weight to the
    given value wrt linear weights. For more info on input and output transforms,
    see `TransformLocation`

    The following explains how weights should be applied to values according to location

    let  x          be input activation
         W          be weight,
         yh, xh, Wh be transformed output, input, weight

    note that
         y  = (x W.T)        // torch.nn.Linear

    Choose values for yh, xh, and Wh which incorporate matrix transforms

    let  V, Vi      be transform matrices on input side
         U, Ui      be transform matrices on output side

    pick xh = (x V)
         Wh = (U.T W Vi.T)
         yh = (y U)

    The following shows that `yh = (xh) (Wh).T` for the chosen values of yh, xh, and Wh

    (xh) (Wh).T = (x V) (U.T W Vi.T).T
                = (x V) (Vi W.T U)        // transpose matrix product identity
                = (x W.T) U
                = y U
                = yh

    :param transform_weight: transform weight to apply
    :param value: value to apply transform_weight to
    :param location: determines how weight should be applied
    :param model_type: result of type(module), passed in to determine application of
        weight transform
    :return: value after transform_weight has been applied
    """

    assert transform_weight.shape[0] == transform_weight.shape[1]

    if module_type == torch.nn.Linear:
        if location == TransformLocation.INPUT:
            return multihead_matmul(value, transform_weight)

        elif location == TransformLocation.WEIGHT_INPUT:
            # equivalent to (transform_weight @ value.T).T
            return multihead_matmul(value, transform_weight.T)

        elif location == TransformLocation.WEIGHT_OUTPUT:
            # equivalent to (value.T @ transform_weight).T
            return multihead_matmul(transform_weight.T, value)

        elif location == TransformLocation.OUTPUT:
            return multihead_matmul(value, transform_weight)

    # similar derivation to torch.nn.Linear, but `y = (x W)`
    elif module_type == torch.nn.Embedding:
        if location == TransformLocation.INPUT:
            return multihead_matmul(value, transform_weight)

        elif location == TransformLocation.WEIGHT_INPUT:
            return multihead_matmul(
                transform_weight,
                value,
            )

        elif location == TransformLocation.WEIGHT_OUTPUT:
            return multihead_matmul(value, transform_weight)

        elif location == TransformLocation.OUTPUT:
            return multihead_matmul(value, transform_weight)

    raise NotImplementedError(
        f"Applying transforms to {module_type} {location} is not supported"
    )


def multihead_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs A @ B for last two dims of two matrices A and B that possibly
    have different shapes, as is the case in multi-headed dimension. If
    shapes are different, this is equivalent to converting the last two dims
    of the smaller matrix into a block-diagonal matrix with the same shape as
    the last two dims of the larger matrix.

    E.g. if A is half the size of B, this function will perform
    [[A  ]  @ B
     [  A]]

    If B is a third of the size of A, this function will perform
    A @ [[B    ]
         [  B  ]
         [    B]]

    This function will error out if the shapes are not evenly divisble

    :param A: left-hand tensor
    :param B: right-hand tensor
    :return: result
    """
    if A.shape[-1] > B.shape[-2]:
        head_dim = B.shape[-2]
        num_heads = A.shape[-1] // head_dim
        A = A.unflatten(-1, (num_heads, head_dim))
        return (A @ B).flatten(-2, -1)
    elif A.shape[-1] < B.shape[-2]:
        head_dim = A.shape[-1]
        num_heads = B.shape[-2] // head_dim
        B = B.unflatten(-2, (num_heads, head_dim))
        return (A @ B).flatten(-3, -2)
    else:
        return A @ B