#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Test the functional Adam optimizer."""

import typing as t
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from optorch.adam import AdamConfig
from optorch.adam import AdamWDMode
from optorch.adam import adam_step
from optorch.builder import init_optimizer_state


def test_adam(step_fn: t.Callable = adam_step, adam_config: t.Optional[AdamConfig] = None):
    """Test equivalence of PyTorch and adam_step."""
    model_a = nn.Linear(10, 5)
    model_b = deepcopy(model_a)

    x_dataset = [
        torch.rand(100, 10),
        torch.rand(100, 10),
        torch.rand(100, 10),
        torch.rand(100, 10),
    ]
    y_dataset = [
        torch.rand(100, 5),
        torch.rand(100, 5),
        torch.rand(100, 5),
        torch.rand(100, 5),
    ]

    loss_fn = nn.MSELoss()

    if adam_config is None:
        # Set a stupid high LR to move far.
        adam_config = AdamConfig(max_lr=0.1, weight_decay=0.1)

    pytorch_opt_map = {
        AdamWDMode.ORIGINAL: torch.optim.Adam,
        AdamWDMode.ADAMW: torch.optim.AdamW,
        AdamWDMode.DECOUPLED: torch.optim.Adam,  # TODO(jramapuram): test against mosiac.
    }

    pytorch_optimizer = pytorch_opt_map[adam_config.mode](
        model_a.parameters(),
        lr=adam_config.max_lr,
        betas=adam_config.betas,
        eps=adam_config.eps,
        weight_decay=adam_config.weight_decay,
    )
    print(pytorch_optimizer)

    for x, y in zip(x_dataset, y_dataset):
        pytorch_optimizer.zero_grad()
        loss = loss_fn(model_a(x), y)
        loss.backward()
        pytorch_optimizer.step()

    opt_state = init_optimizer_state("adam", model_b)

    for x, y in zip(x_dataset, y_dataset):

        def mse_loss_func(params, inputs, targets):
            """Compute the mean squared error loss."""
            preds = torch.func.functional_call(model_b, params, (inputs,))
            loss = F.mse_loss(preds, targets)
            return loss

        params = dict(model_b.named_parameters())
        loss, vjp_fn = torch.func.vjp(mse_loss_func, params, x, y)
        grads = vjp_fn(torch.ones_like(loss))[0]

        # Params are updated in place, so no need to set params = updated_params
        updated_params, opt_state = step_fn(params, grads, opt_state, adam_config)

    for (_, pytorch_param), (_, custom_param) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert (
            pytorch_param.shape == custom_param.shape
        ), f"Shape mismatch, PyTorch {pytorch_param.shape} vs. ours {custom_param.shape}"
        assert torch.allclose(
            pytorch_param, custom_param, atol=1e-4
        ), f"Tolerance was high: {torch.norm(pytorch_param - custom_param)}"


def test_adamw_step():
    """Test equivalence of PyTorch and adamw_step."""
    adamw_config = AdamConfig(max_lr=0.1, weight_decay=0.1)
    adamw_config.mode = AdamWDMode.ADAMW
    test_adam(step_fn=adam_step, adam_config=adamw_config)


def test_decoupled_adam_step():
    """Test equivalence of PyTorch and decoupled adam step."""
    decoupled_adam_config = AdamConfig(max_lr=0.1, weight_decay=0.1)
    decoupled_adam_config.mode = AdamWDMode.DECOUPLED
    decoupled_adam_config.weight_decay = 0.0  # TODO(jramapuram): test against mosiac.
    test_adam(step_fn=adam_step, adam_config=decoupled_adam_config)
