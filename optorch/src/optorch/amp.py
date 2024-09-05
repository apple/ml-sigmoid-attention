#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Functional automatic mixed precision [BETA]."""

import typing as t

import torch


def init_grad_scaler(
    init_scale: float = 2.0**16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    precision: str = "torch.bfloat16",
    device: torch.device = torch.device("cpu"),
) -> t.Dict[str, t.Any]:
    """Initialize the gradient scaler state.

    :param init_scale: Initial scale factor (default: 2.0**16).
    :param growth_factor: Factor by which the scale is multiplied during update if no infs/NaNs occur (default: 2.0).
    :param backoff_factor: Factor by which the scale is multiplied during update if infs/NaNs occur (default: 0.5).
    :param growth_interval: Consecutive iterations without infs/NaNs to trigger scale by growth_factor (default: 2000).
    :param precision: The precision to use for mixed precision training (default: torch.bfloat16).
    :param device: The compute device to drop the state onto.
    :return: A dictionary representing the gradient scaler state.

    """
    return {
        "scale": torch.full((), init_scale, dtype=torch.float32, device=device),
        "growth_factor": growth_factor,
        "backoff_factor": backoff_factor,
        "growth_interval": growth_interval,
        "growth_tracker": torch.full((), 0, dtype=torch.int32, device=device),
        "precision": eval(precision),
    }


def scale_loss(loss: torch.Tensor, scaler_state: t.Dict[str, t.Any]) -> torch.Tensor:
    """Scale the loss tensor.

    :param loss: The loss tensor to be scaled.
    :param scaler_state: The gradient scaler state dictionary.
    :return: The scaled loss tensor.

    """
    return loss * scaler_state["scale"]


def unscale_grads(
    grads: t.Dict[str, torch.Tensor],
    scaler_state: t.Dict[str, t.Any],
    device: torch.device,
    allow_fp16: bool = False,
) -> t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]:
    """Unscale the gradients and check for infs/NaNs.

    :param grads: A dictionary of gradient tensors.
    :param scaler_state: The gradient scaler state dictionary.
    :param device: The device to perform the computations on.
    :param allow_fp16: Flag to allow FP16 gradients (default: False).
    :return: A tuple containing the unscaled gradients and the found_inf tensor.

    """
    unscaled_grads = {}
    inv_scale = scaler_state["scale"].double().reciprocal().float()

    for key, grad in grads.items():
        if grad is None:
            unscaled_grads[key] = None
            continue

        if not allow_fp16 and grad.dtype == torch.float16:
            raise ValueError("Attempting to unscale FP16 gradients.")

        unscaled_grads[key] = grad * inv_scale.to(device)

    found_inf = torch.full((), 0.0, dtype=torch.float32, device=device)

    for grad in unscaled_grads.values():
        if grad is not None:
            grad_inf = torch.isinf(grad).any().float()
            found_inf = torch.max(found_inf, grad_inf)

    return unscaled_grads, found_inf


def update_scale(scaler_state: t.Dict[str, t.Any], found_inf: torch.Tensor):
    """Update the scale factor based on the presence of infs/NaNs.

    :param scaler_state: The gradient scaler state dictionary.
    :param found_inf: The found_inf tensor indicating the presence of infs/NaNs.

    """
    torch._amp_update_scale_(
        scaler_state["scale"],
        scaler_state["growth_tracker"],
        found_inf,
        scaler_state["growth_factor"],
        scaler_state["backoff_factor"],
        scaler_state["growth_interval"],
    )
