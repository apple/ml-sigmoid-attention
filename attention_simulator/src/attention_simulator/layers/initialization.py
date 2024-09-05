#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Initialization functions for the model."""

from torch import nn


def init_weights_timm(module: nn.Module):
    """Simple weight init similar to TiMM init."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)

        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    if isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
