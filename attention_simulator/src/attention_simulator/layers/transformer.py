#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""All ye transformer layers."""

import typing as t

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Transformer block that supports any attention and MLP layer.

    :param dim: Latent dimension of transformer.
    :param attn_fn: Curried module for Attention.
    :param mlp_fn: Curried module for MLP.
    :param scaling_layer_fn: Curried module for scaling layer.
    :param norm_layer: Normalization layer to use in the Transformer block.

    """

    def __init__(
        self,
        dim: int,
        attn_fn: t.Callable,
        mlp_fn: t.Callable,
        scaling_layer_fn: t.Callable = None,
        norm_layer: t.Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_fn()  # Lazy init.
        self.ls1 = scaling_layer_fn(dim) if scaling_layer_fn is not None else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_fn()
        self.ls2 = scaling_layer_fn(dim) if scaling_layer_fn is not None else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: t.Optional[torch.Tensor] = None,
    ) -> t.Dict[str, t.Any]:
        """Infer the transformer block and return the following:

        1. Attention matrix.
        2. Attention matrix times V.
        3. linear_proj(Attention * V)
        4. MLP output.
        5. Block output: x + LS(MLP output)

        """
        attn_dict = self.attn(self.norm1(x), mask=mask)
        x = x + self.ls1(attn_dict["attn_proj"])
        mlp_output = self.mlp(self.norm2(x))
        final_residual_output = x + self.ls2(mlp_output)

        return {
            "mlp_output": mlp_output,
            "block_output": final_residual_output,
            **attn_dict,
        }
