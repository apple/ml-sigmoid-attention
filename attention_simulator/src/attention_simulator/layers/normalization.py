#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Normalization layers for PyTorch."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization: https://arxiv.org/abs/1910.07467

    :param normalized_shape: Input shape from an expected input of size
    :param eps: A value added to the denominator for numerical stability. Default: 1e-8

    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Applies RMS normalization to the input tensor.

        :param x: Input tensor of shape :math:`(*, H, W)`
        :return: RMS normalized tensor

        """
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * x.numel() ** (-1.0 / x.ndim)
        return self.weight * x / (rms_x + self.eps)


class LayerScale(nn.Module):
    """Layer scale layer from https://arxiv.org/abs/2103.17239"""

    def __init__(self, dim: int, init_values: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
