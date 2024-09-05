#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Adam optimizer.

Adam: https://arxiv.org/abs/1412.6980.
AdamW & decoupled: https://arxiv.org/abs/1711.05101

"""

import typing as t
from dataclasses import dataclass
from enum import Enum

import torch
import tree
from torch import nn

# Local imports
from optorch.common import OPTIMIZER_STATE_TYPE
from optorch.common import PARAMS_GRADS_TYPE
from optorch.common import zero_init_like


class AdamWDMode(Enum):
    """Determines how Adam handles weight decay."""

    ORIGINAL = 0
    ADAMW = 1
    DECOUPLED = 2


@dataclass
class AdamConfig:
    """Adam optimizer configuration.

    NOTE: we skip implementing amsgrad.
    See: https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html#amsgrad

    """

    max_lr: float = 1e-3
    betas: t.Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    mode: AdamWDMode = AdamWDMode.ORIGINAL
    steps: int = 0
    # The LR schedule value in in the range [0, 1]
    lr_schedule_value: t.Optional[t.Callable[[int], float]] = None


def default_adam_state(model: nn.Module) -> t.Dict[str, torch.Tensor]:
    """Default Adam optimizer state."""
    return {
        "exp_avg": zero_init_like(model),
        "exp_avg_sq": zero_init_like(model),
    }


@torch.no_grad()
def adam(
    params: PARAMS_GRADS_TYPE,
    grads: PARAMS_GRADS_TYPE,
    exp_avg: PARAMS_GRADS_TYPE,
    exp_avg_sq: PARAMS_GRADS_TYPE,
    config: AdamConfig,
) -> t.Tuple[PARAMS_GRADS_TYPE, PARAMS_GRADS_TYPE, PARAMS_GRADS_TYPE]:
    """Functional implementation of the Adam optimizer.

    NOTE: This function can handle Adam, AdamW and decoupled WD Adam.

    :param params: Model parameters.
    :param grads: Gradients of the model parameters.
    :param exp_avg: Exponential moving average of gradients.
    :param exp_avg_sq: Exponential moving average of squared gradients.
    :param config: Configuration for the Adam optimizer.
    :return: Updated model parameters, exponential moving averages.

    """

    def _adam(param, grad, exp_avg, exp_avg_sq):
        """Inplace Adam / AdamW / Decoupled Adam update rule."""
        bias_correction1 = 1 - config.betas[0] ** (config.steps + 1)
        bias_correction2 = torch.ones_like(exp_avg_sq) * (1 - config.betas[1] ** (config.steps + 1))

        # Update for the current scheduled LR.
        current_lr = config.max_lr
        if config.lr_schedule_value is not None:
            current_lr *= config.lr_schedule_value

        # Adam logic: WD is added before the bias correction.
        if config.weight_decay > 0 and config.mode == AdamWDMode.ORIGINAL:
            grad.add_(param, alpha=config.weight_decay)

        # Compute first and second EMA moments.
        exp_avg.mul_(config.betas[0]).add_(grad, alpha=1 - config.betas[0])
        exp_avg_sq.mul_(config.betas[1]).add_(grad**2, alpha=1 - config.betas[1])

        # Compute bias-corrected moments and step size.
        denom = (exp_avg_sq.sqrt() / torch.sqrt(bias_correction2)).add_(config.eps)
        step_size = current_lr / bias_correction1

        # AdamW logic: WD is added after the bias correction.
        if config.weight_decay > 0 and config.mode == AdamWDMode.ADAMW:
            param.add_(param, alpha=-config.weight_decay * current_lr)

        param.add_(exp_avg / denom, alpha=-step_size)

        # Decoupled WD Adam logic: WD is added independently, but scaled by sched.
        if config.weight_decay > 0 and config.mode == AdamWDMode.DECOUPLED:
            lr_schedule_value = 1.0
            if config.lr_schedule_value is not None:
                lr_schedule = config.lr_schedule_value

            param.mul_(1 - config.weight_decay * lr_schedule_value)

    tree.map_structure(_adam, params, grads, exp_avg, exp_avg_sq)
    return params, exp_avg, exp_avg_sq


@torch.no_grad()
def adam_step(
    params: PARAMS_GRADS_TYPE,
    grads: PARAMS_GRADS_TYPE,
    opt_state: OPTIMIZER_STATE_TYPE,
    config: AdamConfig,
) -> t.Tuple[PARAMS_GRADS_TYPE, OPTIMIZER_STATE_TYPE]:
    """Perform a single Adam optimization step."""
    exp_avg, exp_avg_sq = opt_state["exp_avg"], opt_state["exp_avg_sq"]
    params, exp_avg, exp_avg_sq = adam(params, grads, exp_avg, exp_avg_sq, config)
    opt_state = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    config.steps += 1
    return params, opt_state
