#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Various linear projection strategies."""

import typing as t

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PostNormLinear(nn.Module):
    """Post-norm + linear projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        norm_layer: t.Callable = nn.LayerNorm,
        which_linear: t.Callable = nn.Linear,
    ):
        super().__init__()
        self.linear = which_linear(in_channels, out_channels, bias=bias)
        self.norm = norm_layer(out_channels)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(samples))


class PreNormLinear(nn.Module):
    """Pre-norm + linear projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        norm_layer: t.Callable = nn.LayerNorm,
        which_linear: t.Callable = nn.Linear,
    ):
        super().__init__()
        self.norm = norm_layer(in_channels)
        self.linear = which_linear(in_channels, out_channels, bias=bias)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(samples))


