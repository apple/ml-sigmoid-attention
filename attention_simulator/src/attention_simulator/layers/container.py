#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""All module containers lie here."""

import typing as t

import torch
from torch import nn

# Local imports
from attention_simulator.helpers.utils import extract_tensor_from_maybe_dict
from attention_simulator.helpers.utils import flatten_dict
from attention_simulator.helpers.utils import sanity_check_block_configs


class BlockContainer(nn.Module):
    """An abstract model type that can house a sequence of blocks.

    - With Attention blocks this can be a transformer.
    - Etc

    :param dim: Dimensionality of the input and output embeddings.
    :param input_embedding_fn: Callable module to embed inputs.
    :param output_embedding_fn: Callable module to embed latents to outputs.
    :param pe_layer_fn: Callable position embedding layer.
    :param init_fn: Weight initialzation function.
    :param which_linear: Linear layer type for final linear projection to vocab.
    :param block_fn: Callable transformer block (used for same block_fn)
    :param num_blocks: Number of transformer blocks (used for same block_fn).
    :param block_configs: Specifies each block independently.

    """

    def __init__(
        self,
        dim: int,
        input_embedding_fn: t.Callable,
        output_embedding_fn: t.Callable,
        pe_layer_fn: t.Callable,
        init_fn: t.Callable,
        block_fn: t.Optional[t.Callable] = None,
        num_blocks: t.Optional[int] = None,
        block_configs: t.Optional[t.List[t.Callable]] = None,
    ):
        super().__init__()
        sanity_check_block_configs(
            block_fn=block_fn, num_blocks=num_blocks, block_configs=block_configs
        )  # Sanity check the block configurations.

        self.input_embedding = input_embedding_fn()
        self.output_embedding = output_embedding_fn()
        self.pe_layer = pe_layer_fn()

        # Build the core blocks of the transformer.
        if block_configs is not None:  # Create blocks based on block_configs
            self.blocks = nn.ModuleList([block_fn() for block_fn in block_configs])
        else:  # Create blocks based on num_blocks
            self.blocks = nn.ModuleList([block_fn() for _ in range(num_blocks)])

        self.apply(init_fn)

    def forward(self, x: torch.Tensor, **unused_kwargs) -> t.Dict[str, torch.Tensor]:
        """Apply the embedding, add PE, infer the blocks, norm and project to vocab_size.."""
        embedded = self.input_embedding(x)
        x = extract_tensor_from_maybe_dict(embedded, "block_output")
        x = self.pe_layer(x)

        results = {"embedding": embedded, "pos_embed": x}
        for idx, block in enumerate(self.blocks):
            results[f"block{idx}"] = block(x)
            x = results[f"block{idx}"]["block_output"]

        results["output_embedding"] = self.output_embedding(x)
        final_output = extract_tensor_from_maybe_dict(results["output_embedding"], "block_output")
        return flatten_dict({"output_representation": final_output, **results})


class CausalBlockContainer(BlockContainer):
    """Extends BlockContainer to support causal models."""

    def forward(
        self,
        x: torch.Tensor,
        mask: t.Optional[torch.Tensor] = None,
    ) -> t.Dict[str, torch.Tensor]:
        """Apply the embedding, add PE, infer the blocks with masking, norm and project to vocab_size.."""
        embedded = self.input_embedding(x)
        x = extract_tensor_from_maybe_dict(embedded, "block_output")
        x = self.pe_layer(x)

        results = {"embedding": embedded, "pos_embed": x}
        for idx, block in enumerate(self.blocks):
            results[f"block{idx}"] = block(x, mask=mask)
            x = results[f"block{idx}"]["block_output"]

        results["output_embedding"] = self.output_embedding(x)
        final_output = extract_tensor_from_maybe_dict(results["output_embedding"], "block_output")
        return flatten_dict({"output_representation": final_output, **results})
