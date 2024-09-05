#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Test the functional LARS optimizer."""

import typing as t

import torch
from torch import nn
from torch.nn import functional as F

from optorch.builder import init_optimizer_state
from optorch.lars import LARSConfig
from optorch.lars import lars_step


def test_lars(step_fn: t.Callable = lars_step, lars_config: t.Optional[LARSConfig] = None):
    """Test forward backward of LARS.

    TODO(jramapuram): support equivalence tests with another LARS impl.

    """
    model_b = nn.Linear(10, 5)

    # Create random input and target data
    x = torch.randn(100, 10)
    y = torch.randn(100, 5)

    if lars_config is None:
        lars_config = LARSConfig(max_lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Initialize the optimizer state using init_optimizer_state
    opt_state = init_optimizer_state("lars", model_b)

    def mse_loss_func(params, inputs, targets):
        """Compute the mean squared error loss."""
        preds = torch.func.functional_call(model_b, params, (inputs,))
        loss = F.mse_loss(preds, targets)
        return loss

    # Compute grads with vjp
    params = dict(model_b.named_parameters())
    loss, vjp_fn = torch.func.vjp(mse_loss_func, params, x, y)
    grads = vjp_fn(torch.ones_like(loss))[0]  # pick the param grads
    print(
        f"params => {[(n, p.shape) for n, p in params.items()]}\n"
        f"grads  => {[(n, p.shape) for n, p in grads.items()]}"
    )

    # Test sgd_step.
    updated_params, opt_state = step_fn(params, grads, opt_state, lars_config)

    # Silly sanity checks.
    for (_, param), (_, grad), (_, updated_param) in zip(params.items(), grads.items(), updated_params.items()):
        assert param.shape == grad.shape == updated_param.shape

