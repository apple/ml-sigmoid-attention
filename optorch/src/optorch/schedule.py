#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Learning rate schedulers: all return [0, 1] schedules."""

import functools
import math
import typing as t
from dataclasses import dataclass


@dataclass
class LRSchedulerConfig:
    """LR scheduler configuration."""

    warmup_steps: int = 0
    warmup_start_value: float = 0.0
    total_steps: int = 0
    scheduler_type: str = "constant"
    lr_terminal_value: float = 0.0  # only cosine & linear_warmdown


def constant_lr(steps: int, warmup_steps: int, warmup_start_value: float) -> float:
    """Constant learning rate scheduler with optional warmup.

    :param steps: Current step.
    :param warmup_steps: Number of warmup steps.
    :param warmup_start_value: Initial value for warmup.
    :return: Learning rate.

    """
    if steps < warmup_steps:
        return warmup_start_value + (1.0 - warmup_start_value) * steps / warmup_steps

    return 1.0


def rsqrt_lr(steps: int, warmup_steps: int, warmup_start_value: float) -> float:
    """Inverse sqrt learning rate scheduler with optional warmup.

    :param steps: Current step.
    :param warmup_steps: Number of warmup steps.
    :param warmup_start_value: Initial value for warmup.
    :return: Learning rate.

    """
    assert warmup_steps > 0, "Inverse sqrt LR requires warmup."
    if steps < warmup_steps:
        return warmup_start_value + (1.0 - warmup_start_value) * steps / warmup_steps

    return math.sqrt(warmup_steps) / math.sqrt(steps)


def cosine_lr(
    steps: int,
    total_steps: int,
    warmup_steps: int,
    warmup_start_value: float,
    lr_terminal_value: float,
) -> float:
    """Cosine LR schedule.

    :param steps: Current step.
    :param warmup_steps: Number of warmup steps.
    :param warmup_start_value: Initial value for warmup.
    :param lr_terminal_value: Endpoint for LR that might not be 0.0
    :return: Learning rate.

    """
    if steps < warmup_steps:
        return warmup_start_value + (1.0 - warmup_start_value) * steps / warmup_steps

    progress = (steps - warmup_steps) / (total_steps - warmup_steps)
    return lr_terminal_value + (1.0 - lr_terminal_value) * 0.5 * (1 + math.cos(math.pi * progress))


def linear_warmdown_lr(
    steps: int,
    total_steps: int,
    warmup_steps: int,
    warmup_start_value: float,
    lr_terminal_value: float,
) -> float:
    """Simple linear ramp down.

    :param steps: Current step.
    :param warmup_steps: Number of warmup steps.
    :param warmup_start_value: Initial value for warmup.
    :param lr_terminal_value: Endpoint for LR that might not be 0.0
    :return: Learning rate.

    """
    if steps < warmup_steps:
        return warmup_start_value + (1.0 - warmup_start_value) * (steps / warmup_steps)

    progress = (steps - warmup_steps) / (total_steps - warmup_steps)
    return 1.0 + (lr_terminal_value - 1.0) * progress


def build_lr_scheduler(config: LRSchedulerConfig) -> t.Callable[[float, int], float]:
    """Builder object for LR schedules."""
    if config.scheduler_type == "constant":
        return functools.partial(
            constant_lr,
            warmup_steps=config.warmup_steps,
            warmup_start_value=config.warmup_start_value,
        )

    if config.scheduler_type == "rsqrt":
        return functools.partial(
            rsqrt_lr,
            warmup_steps=config.warmup_steps,
            warmup_start_value=config.warmup_start_value,
        )

    if config.scheduler_type == "cosine":
        return functools.partial(
            cosine_lr,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            warmup_start_value=config.warmup_start_value,
            lr_terminal_value=config.lr_terminal_value,
        )

    if config.scheduler_type == "linear_warmdown":
        return functools.partial(
            linear_warmdown_lr,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            warmup_start_value=config.warmup_start_value,
            lr_terminal_value=config.lr_terminal_value,
        )

    raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")
