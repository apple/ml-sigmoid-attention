#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Functional Stochastic Gradient Descent (SGD) optimizer."""

import typing as t
from dataclasses import dataclass

import torch
import tree
from torch import nn

from optorch.common import OPTIMIZER_STATE_TYPE
from optorch.common import PARAMS_GRADS_TYPE
from optorch.common import zero_init_like


@dataclass
class SGDConfig:
    """Stochastic Gradient Descent with Momentum."""

    momentum: float = 0.9
    weight_decay: float = 0.0
    max_lr: float = 0.1
    steps: int = 0
    # The LR schedule value in in the range [0, 1]
    lr_schedule_value: t.Optional[t.Callable[[int], float]] = None


def default_sgd_state(model: nn.Module) -> t.Dict[str, torch.Tensor]:
    """Default SGD optimizer state."""
    return {
        "momentum": zero_init_like(model),
    }


@torch.no_grad()
def sgd(
    params: PARAMS_GRADS_TYPE,
    grads: PARAMS_GRADS_TYPE,
    momentum: PARAMS_GRADS_TYPE,
    config: SGDConfig,
) -> t.Tuple[PARAMS_GRADS_TYPE, PARAMS_GRADS_TYPE]:
    """Functional implementation of the SGD optimizer.

    :param params: List of model parameters.
    :param grads: List of gradients of the model parameters.
    :param momentum: List of momentum values for each parameter.
    :param config: Configuration for the SGD optimizer.
    :return: Updated model parameters, momentum values, and state steps.

    """

    def _sgd(param, grad, momentum):
        """Inplace SGD update rule."""
        if config.weight_decay > 0:
            update = grad.add(param, alpha=config.weight_decay)

        if config.momentum > 0:
            momentum.mul_(config.momentum).add_(grad)
            update = momentum

        # Update for the current scheduled LR.
        current_lr = config.max_lr
        if config.lr_schedule_value is not None:
            current_lr *= config.lr_schedule_value

        param.sub_(current_lr * update)

    tree.map_structure(_sgd, params, grads, momentum)
    return params, momentum


@torch.no_grad()
def sgd_step(
    params: PARAMS_GRADS_TYPE,
    grads: PARAMS_GRADS_TYPE,
    opt_state: OPTIMIZER_STATE_TYPE,
    config: SGDConfig,
) -> t.Tuple[PARAMS_GRADS_TYPE, OPTIMIZER_STATE_TYPE]:
    """Perform a single SGD optimization step.

    :param params: List of model parameters.
    :param grads: List of gradients of the model parameters.
    :param opt_state: Optimizer state dictionary.
    :param config: Configuration for the SGD optimizer.
    :return: Updated model parameters and optimizer state dictionary.

    """
    momentum = opt_state["momentum"]
    params, momentum = sgd(params, grads, momentum, config)
    opt_state = {"momentum": momentum}
    config.steps += 1
    return params, opt_state
