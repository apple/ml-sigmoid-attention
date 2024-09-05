#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Parameter counters and related."""

import typing as t

from torch import nn


def number_of_embedding_parameters(
    model: nn.Module,
    only_trainable: bool = False,
    count: int = 0,
) -> int:
    """Get the number of embedding parameters in a model.

    :param model: The model.
    :param only_trainable: If `True` only count trainable parameters.
    :param count: Do not set, used for recursion.
    :return: The number of (trainable) embedding parameters in `model`.

    """
    if isinstance(model, nn.Embedding):
        for p in model.parameters():
            if p.requires_grad or not only_trainable:
                count += p.numel()
    for c in model.children():
        count = number_of_embedding_parameters(
            model=c,
            only_trainable=only_trainable,
            count=count,
        )
    return count


def number_of_parameters(model: nn.Module, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
    """Get the number of parameters in a model.

    :param model: The model.
    :param only_trainable: If `True`, count only models with grads.
    :param exclude_embeddings: If `True` ignore embeddings in the count.
    :return: Number of parameters,

    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad or not only_trainable)
    if exclude_embeddings:
        params -= number_of_embedding_parameters(model=model, only_trainable=only_trainable)
    return params


def number_of_buffers(model: t.Any) -> float:
    """Count for the number of buffers in the model."""
    return sum(p.numel() for p in model.buffers())


def get_param_buffer_info(model: nn.Module) -> t.Dict[str, int]:
    """Get the relevant parameter and buffer count info.

    :param model: The model.
    :return: The parameter buffer info.

    """
    result = {
        "total_parameters": number_of_parameters(model),
        "total_non_embedding_parameters": number_of_parameters(model, exclude_embeddings=True),
        "trainable_parameters": number_of_parameters(model, only_trainable=True),
        "trainable_non_embedding_parameters": number_of_parameters(model, exclude_embeddings=True, only_trainable=True),
        "total_buffers": number_of_buffers(model),
    }
    result["total_parameters_plus_buffers"] = result["total_parameters"] + result["total_buffers"]

    return result
