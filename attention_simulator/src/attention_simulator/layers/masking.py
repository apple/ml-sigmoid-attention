#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Masking functions for ze attention."""

import typing as t

import torch

from attention_simulator.helpers.utils import unsqueeze_like


def create_decoder_causal_mask(
    seq_len: int,
    device: torch.device,
    attn_mask: t.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create a causal mask for decoder autoregressive modeling.

    :param seq_len: Sequence length.
    :param device: Device to create the mask on.
    :param attn_mask: Attention mask (0 means padding).
    :return: Causal mask tensor ([B, 1, S, S] or [S, S])

    """
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

    if attn_mask is not None:
        # Merge with hugginface mask matrix (eg: padding a small seq).
        # Original mask is [B, L]
        B, S = attn_mask.shape
        attn_mask = attn_mask.view(B, 1, 1, S).bool()
        causal_mask = attn_mask & causal_mask  # [B, 1, S, S]

    return causal_mask


def create_cross_attention_causal_mask(
    target_seq_len: int,
    source_seq_len: int,
    device: torch.device,
    attn_mask: t.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create a causal mask for cross attention in autoregressive modeling.

    :param target_seq_len: Target sequence length.
    :param source_seq_len: Source sequence length.
    :param device: Device to create the mask on.
    :param attn_mask: Attention mask (0 means padding).
    :return: Cross attention causal mask tensor ([B, 1, T, S] or [T, S])

    """
    causal_mask = torch.tril(torch.ones(target_seq_len, source_seq_len, device=device)).bool()

    if attn_mask is not None:
        # Merge with huggingface mask matrix (eg: padding a small seq).
        # Original mask is [B, S]
        B, S = attn_mask.shape
        attn_mask = attn_mask.view(B, 1, 1, S).bool()
        causal_mask = attn_mask & causal_mask  # [B, 1, T, S]

    return causal_mask


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    attn_mask: t.Optional[torch.Tensor] = None,
    target_seq_len: t.Optional[int] = None,
) -> torch.Tensor:
    """Create a causal mask for autoregressive modeling.

    :param seq_len: Sequence length.
    :param device: Device to create the mask on.
    :param attn_mask: Attention mask (0 means padding).
    :param target_seq_len: Target sequence length (for cross attention).
    :return: Causal mask tensor ([B, 1, S, S] or [S, S] or [B, 1, T, S] or [T, S])

    """
    if target_seq_len is not None:
        # Cross attention needs a [T x S] mask.
        return create_cross_attention_causal_mask(
            source_seq_len=seq_len,
            target_seq_len=target_seq_len,
            device=device,
            attn_mask=attn_mask,
        )

    return create_decoder_causal_mask(seq_len=seq_len, device=device, attn_mask=attn_mask)


def softmax_masking_fn() -> t.Callable:
    """Generate a masking function for softmax.

    softmax_masking_fn()(
        attn=torch.randn(4, 4), mask=create_causal_mask(4, "cpu")
    ).softmax(dim=-1)

    tensor([[1.0000, 0.0000, 0.0000, 0.0000],
            [0.4814, 0.5186, 0.0000, 0.0000],
            [0.3055, 0.4237, 0.2708, 0.0000],
            [0.5775, 0.1821, 0.1019, 0.1385]])

    :return: A function that applies the appropriate masking logic.

    """

    def masking_fn(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply causal and padding masks to attention logits.

        :param attn: Attention logits tensor of shape [B, H, T, S]
        :param mask: Mask tensor of shape [B, T, S] or [T, S]
        :return: Masked attention logits

        """
        mask = unsqueeze_like(mask, attn).expand_as(attn)  # -> [B, H, T, S]
        return attn.masked_fill(~mask, -torch.finfo(attn.dtype).max)

    return masking_fn
