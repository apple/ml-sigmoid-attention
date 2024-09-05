#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Test the functional SGD optimizer."""

import typing as t
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from optorch.builder import init_optimizer_state
from optorch.sgd import SGDConfig
from optorch.sgd import sgd_step


def test_sgd_step(step_fn: t.Callable = sgd_step, sgd_config: t.Optional[SGDConfig] = None):
    """Test equivalance of pytorch and sgd_step."""
    model_a = nn.Linear(10, 5)
    model_b = deepcopy(model_a)

    # Create random input and target data
    x = torch.randn(100, 10)
    y = torch.randn(100, 5)

    # Define the loss function and SGD configuration
    loss_fn = nn.MSELoss()

    if sgd_config is None:
        sgd_config = SGDConfig(max_lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Create PyTorch SGD optimizer
    pytorch_optimizer = torch.optim.SGD(
        model_a.parameters(),
        lr=sgd_config.max_lr,
        momentum=sgd_config.momentum,
        weight_decay=sgd_config.weight_decay,
    )

    # Forward pass and compute gradients
    loss = loss_fn(model_a(x), y)
    loss.backward()

    # Update parameters using PyTorch SGD optimizer
    pytorch_optimizer.step()

    # Initialize the optimizer state using init_optimizer_state
    opt_state = init_optimizer_state("sgd", model_b)

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
    updated_params, opt_state = step_fn(params, grads, opt_state, sgd_config)

    # Compare the updated parameters with PyTorch SGD optimizer
    for (_, pytorch_param), (_, custom_param) in zip(model_a.named_parameters(), updated_params.items()):
        assert (
            pytorch_param.shape == custom_param.shape
        ), f"Shape mismatch, pytorch {pytorch_param.shape} vs. ours {custom_param.shape}"
        assert torch.allclose(
            pytorch_param, custom_param, atol=1e-4
        ), f"Tolerance was high: {torch.norm(pytorch_param - custom_param)}"

