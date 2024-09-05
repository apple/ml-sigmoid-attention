#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Common utility functions for Optorch."""

import typing as t

import torch
from torch import nn

# Houses params or gradients as a list of tensors or dict of tensors.
PARAMS_GRADS_TYPE = t.Union[t.List[torch.Tensor], t.Dict[str, torch.Tensor]]

# Optimizer state can be any dict items.
OPTIMIZER_STATE_TYPE = t.Dict[str, t.Any]


def zero_init_like(model: nn.Module) -> t.Dict[str, torch.Tensor]:
    """Initialize a list of tensors with zeros.

    :param model: Model to initialize like
    :return: Dictionary of tensors.

    """
    return {name: torch.zeros_like(param) for name, param in model.named_parameters()}
