#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Multi-layer perceptron modules."""

import typing as t

import torch
from torch import nn


class Mlp(nn.Module):
    """Good ol' multi-layer perceptron."""

    def __init__(
        self,
        in_features: int,
        hidden_features: t.Optional[int] = None,
        out_features: t.Optional[int] = None,
        act_layer: t.Callable = nn.GELU,
        bias: bool = False,
        drop: float = 0.0,
        norm_layer: t.Optional[t.Callable] = None,
        which_linear: t.Callable = nn.Linear,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = which_linear(in_features, hidden_features, bias=False)
        self.norm1 = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = which_linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
