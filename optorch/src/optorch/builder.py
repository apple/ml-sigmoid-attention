#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Optorch: A (Jax inspired) PyTorch library for optimizers."""

import functools
import typing as t

import torch
from torch import nn

# Local imports
from optorch.sgd import default_sgd_state
from optorch.sgd import sgd_step
from optorch.sgd import SGDConfig

from optorch.lars import lars_step
from optorch.lars import LARSConfig

from optorch.adam import default_adam_state
from optorch.adam import adam_step
from optorch.adam import AdamConfig
from optorch.adam import AdamWDMode

# Adding a new optimizer entails adding an entry to each dict below.

DEFAULT_OPTIMIZER_CONFIGS = {
    "adam": AdamConfig,
    "adamw": functools.partial(AdamConfig, mode=AdamWDMode.ADAMW),
    "decoupled_adam": functools.partial(AdamConfig, mode=AdamWDMode.DECOUPLED),
    "lars": LARSConfig,
    "sgd": SGDConfig,
}


DEFAULT_OPTIMIZER_STATES = {
    "adam": default_adam_state,
    "adamw": default_adam_state,
    "decoupled_adam": default_adam_state,
    "lars": default_sgd_state,
    "sgd": default_sgd_state,
}


OPTIMIZER_STEP = {
    "adamw": adam_step,
    "adam": adam_step,
    "decoupled_adam": adam_step,
    "lars": lars_step,
    "sgd": sgd_step,
}


def init_optimizer_state(
    optimizer_name: str, model: nn.Module
) -> t.Dict[str, torch.Tensor]:
    """Initialize optimizer state.

    TODO(jramapuram): move this to each optimizer module.

    :param optimizer_name: Name of the optimizer.
    :param model: Model to optimize.
    :return: Initial optimizer state dictionary.

    """
    if optimizer_name not in DEFAULT_OPTIMIZER_CONFIGS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return DEFAULT_OPTIMIZER_STATES[optimizer_name](model)


@functools.cache
def get_optimizer_step(optimizer_name: str) -> t.Callable:
    """Get the vectorized optimizer function based on the optimizer name.

    :param optimizer_name: Name of the optimizer.
    :return: Vectorized optimizer function.

    """
    if optimizer_name not in OPTIMIZER_STEP:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return OPTIMIZER_STEP[optimizer_name]
