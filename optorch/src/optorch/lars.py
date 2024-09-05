#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Functional Layerwise Adaptive Rate Scaling (LARS) optimizer."""

import typing as t
from dataclasses import dataclass

import torch
import tree

from optorch.common import OPTIMIZER_STATE_TYPE
from optorch.common import PARAMS_GRADS_TYPE


@dataclass
class LARSConfig:
    """LARS optimizer configuration."""

    max_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    trust_coefficient: float = 0.001
    eps: float = 1e-8
    steps: int = 0
    # The LR schedule value in in the range [0, 1]
    lr_schedule_value: t.Optional[t.Callable[[int], float]] = None


@torch.no_grad()
def lars(
    params: PARAMS_GRADS_TYPE,
    grads: PARAMS_GRADS_TYPE,
    momentum: PARAMS_GRADS_TYPE,
    config: LARSConfig,
) -> t.Tuple[PARAMS_GRADS_TYPE, PARAMS_GRADS_TYPE]:
    """Functional implementation of the LARS optimizer."""

    def _lars(param, grad, momentum):
        """Inplace LARS update rule."""
        if config.weight_decay > 0:
            grad = grad.add(param, alpha=config.weight_decay)

        param_norm = torch.norm(param)
        grad_norm = torch.norm(grad)

        # Update for the current scheduled LR.
        current_lr = config.max_lr
        if config.lr_schedule_value is not None:
            current_lr *= config.lr_schedule_value

        local_lr = current_lr * config.trust_coefficient * (param_norm / (grad_norm + config.eps))

        if config.momentum > 0:
            momentum = momentum * config.momentum + grad * local_lr
            param = param - momentum
        else:
            param = param - grad * local_lr

    tree.map_structure(_lars, params, grads, momentum)
    return params, momentum


@torch.no_grad()
def lars_step(
    params: PARAMS_GRADS_TYPE,
    grads: PARAMS_GRADS_TYPE,
    opt_state: OPTIMIZER_STATE_TYPE,
    config: LARSConfig,
) -> t.Tuple[PARAMS_GRADS_TYPE, OPTIMIZER_STATE_TYPE]:
    """Perform a single LARS optimization step."""
    momentum = opt_state["momentum"]
    params, momentum = lars(params, grads, momentum, config)
    opt_state = {"momentum": momentum}
    config.steps += 1
    return params, opt_state
