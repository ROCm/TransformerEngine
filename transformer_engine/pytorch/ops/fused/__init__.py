# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compound tensor operation supported by the operation fuser."""

<<<<<<< origin/dev
from .backward_activation_bias import (
    BackwardActivationBias,
    fuse_backward_activation_bias,
)
from .backward_add_rmsnorm import (
    BackwardAddRMSNorm,
    fuse_backward_add_rmsnorm,
)
from .backward_linear_add import (
    BackwardLinearAdd,
    fuse_backward_linear_add,
)
from .backward_linear_scale import (
    BackwardLinearScale,
    fuse_backward_linear_scale,
)
from .forward_linear_bias_activation import (
    ForwardLinearBiasActivation,
    fuse_forward_linear_bias_activation,
)
from .forward_linear_bias_add import (
    ForwardLinearBiasAdd,
    fuse_forward_linear_bias_add,
)
from .forward_linear_scale_add import (
    ForwardLinearScaleAdd,
    fuse_forward_linear_scale_add,
)

from .userbuffers_backward_linear import (
    UserbuffersBackwardLinear,
    fuse_userbuffers_backward_linear,
)
from .userbuffers_forward_linear import (
    UserbuffersForwardLinear,
    fuse_userbuffers_forward_linear,
)
=======
from ..fuser import register_backward_fusion, register_forward_fusion
from .backward_activation_bias import BackwardActivationBias
from .backward_add_rmsnorm import BackwardAddRMSNorm
from .backward_linear_add import BackwardLinearAdd
from .backward_linear_scale import BackwardLinearScale
from .forward_linear_bias_activation import ForwardLinearBiasActivation
from .forward_linear_bias_add import ForwardLinearBiasAdd
from .forward_linear_scale_add import ForwardLinearScaleAdd
from .userbuffers_backward_linear import UserbuffersBackwardLinear
from .userbuffers_forward_linear import UserbuffersForwardLinear


# Register forward fusions
register_forward_fusion(UserbuffersForwardLinear.fuse_forward_ops)
register_forward_fusion(ForwardLinearBiasAdd.fuse_forward_ops)
register_forward_fusion(ForwardLinearBiasActivation.fuse_forward_ops)
register_forward_fusion(ForwardLinearScaleAdd.fuse_forward_ops)

# Register backward fusions
register_backward_fusion(UserbuffersBackwardLinear.fuse_backward_ops)
register_backward_fusion(BackwardLinearAdd.fuse_backward_ops)
register_backward_fusion(BackwardLinearScale.fuse_backward_ops)
register_backward_fusion(BackwardActivationBias.fuse_backward_ops)
register_backward_fusion(BackwardAddRMSNorm.fuse_backward_ops)
>>>>>>> 708d7c160ad6b2bf44c9c597083d4cbb4860f068
