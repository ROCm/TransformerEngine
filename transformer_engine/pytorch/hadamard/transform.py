from typing import Optional

import torch
from torch import Tensor, device
from torch.distributed.tensor import DTensor, Replicate

from .matrix import multihead_matmul
from .hadamard import deterministic_hadamard_matrix
# random_hadamard_matrix disabled - requires hadamards.safetensors


class HadamardFactory:
    """
    Factory used to apply hadamard transforms to a model.

    Class-level configuration parameters can be set using the configure() method.
    Individual create_transform() calls can override these defaults.

    Class attributes:
        block_size: Default size of the Hadamard block
        randomized: Default whether to use randomized Hadamard transform
        dtype: Default data type for the transform
        seed: Default random seed used for randomization
    """

    # Class-level default configuration
    block_size: int = 32
    randomized: bool = True
    dtype: torch.dtype = torch.float32
    seed: Optional[int] = None
    generator: torch.Generator = torch.Generator()


    @classmethod
    def configure(
        cls,
        block_size: Optional[int] = None,
        randomized: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Configure class-level default parameters for HadamardFactory.

        :param block_size: Default size of the Hadamard block
        :param randomized: Default whether to use randomized Hadamard transform
        :param dtype: Default data type for the transform
        :param seed: Default random seed used for randomization
        """
        if block_size is not None:
            cls.block_size = block_size
        if randomized is not None:
            cls.randomized = randomized
        if dtype is not None:
            cls.dtype = dtype
        if seed is not None:
            cls.seed = seed
            cls.generator.manual_seed(seed)

    @classmethod
    def create_transform(
        cls,
        device: torch.device,
    ) -> 'HadamardTransform':
        """
        Create a HadamardTransform for applying to a module.

        Parameters default to class-level configuration set via configure().
        Any parameter explicitly provided will override the class default.

        :param device: Device to create the transform on
        :return: HadamardTransform instance
        """

        weight = cls._create_weight(device)
        perm = cls._create_permutation(weight) if cls.randomized else None
        return HadamardTransform(weight, perm)

    @classmethod
    def _create_weight(
        cls,
        device: device,
    ) -> Tensor:
        # Fixed: Use deterministic when randomized=False
        if cls.randomized:
            # DISABLED: Requires hadamards.safetensors file
            # data = random_hadamard_matrix(cls.block_size, cls.dtype, device, cls.generator)
            raise NotImplementedError("Randomized Hadamard requires hadamards.safetensors file")
        else:
            data = deterministic_hadamard_matrix(cls.block_size, cls.dtype,
                                                 device)
        return data

    @classmethod
    def _create_permutation(cls, weight: Tensor) -> Tensor:
        data = torch.randperm(weight.size(0), generator=cls.generator)
        return data

class HadamardTransform:
    """
    Hadamard transform that can be applied to tensors.

    :param weight: Hadamard matrix
    :param perm: Optional permutation tensor for randomized transforms
    """

    def __init__(
        self,
        weight: Tensor,
        perm: Optional[Tensor],
    ):
        self.weight = weight
        self.perm = perm
        self._scale = torch.tensor(weight.size(0),
                                   dtype=torch.float64,
                                   device=weight.device).sqrt()

    def __call__(self, value: Tensor, inverse: bool = False,
                 left_mul: bool = False) -> Tensor:
        """
        Apply the Hadamard transform to a tensor.

        :param value: Input tensor to transform
        :param inverse: If True, apply the inverse transform
        :param left_mul: If True, multiply from the left (weight @ value),
                         else from the right (value @ weight)
        :return: Transformed tensor
        """
        weight = self.weight

        if self.perm is not None:
            weight = weight[self.perm][:, self.perm]

        if inverse:
            weight = weight.T

        if isinstance(weight, DTensor):
            assert weight.placements[0] == Replicate()
            weight = weight.to_local()

        if left_mul:
            return (multihead_matmul(weight.to(device=value.device),
                                     value.to(dtype=weight.dtype)) /
                    self._scale).to(value.dtype)
        else:
            return (multihead_matmul(value.to(dtype=weight.dtype),
                                     weight.to(device=value.device)) /
                    self._scale).to(value.dtype)