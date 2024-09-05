#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""All ye pos embeds lie here."""

import math
import typing as t
from functools import lru_cache

import tree
import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(maxsize=100)
@torch.jit.script
def precompute_freqs_cis(dim: int, seqlen: int, device: torch.device) -> torch.Tensor:
    """Precompute the frequency tensor for RoPE.

    :param dim: Embedding dimension.
    :param seqlen: Sequence length.
    :param device: Device to generate the frequencies on.
    :return: Precomputed frequency tensor.

    """
    with torch.no_grad():
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
        t = torch.arange(seqlen, device=device)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_pos_offset: int = 0,
    k_pos_offset: int = 0
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys.

    This function supports different sized Q and K (e.g., cross-attention) and different position offsets.

    :param xq: Query tensor.
    :param xk: Key tensor.
    :param q_pos_offset: Position offset for queries.
    :param k_pos_offset: Position offset for keys.
    :return: Updated query and key tensors.

    """
    batch_size_q, num_heads_q, seq_length_q, head_dim_q = xq.shape
    batch_size_k, num_heads_k, seq_length_k, head_dim_k = xk.shape

    assert head_dim_q == head_dim_k, "Head dimensions for queries and keys must be equal for RoPE."
    head_dim = head_dim_q

    max_seq_length = max(q_pos_offset + seq_length_q, k_pos_offset + seq_length_k)
    freqs_cis = precompute_freqs_cis(head_dim, seqlen=max_seq_length, device=xq.device)

    xq_ = torch.view_as_complex(xq.float().reshape(batch_size_q, num_heads_q, seq_length_q, head_dim // 2, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(batch_size_k, num_heads_k, seq_length_k, head_dim // 2, 2))

    freqs_cis_q = freqs_cis[q_pos_offset:q_pos_offset + seq_length_q, :head_dim // 2].unsqueeze(0).unsqueeze(0)
    freqs_cis_k = freqs_cis[k_pos_offset:k_pos_offset + seq_length_k, :head_dim // 2].unsqueeze(0).unsqueeze(0)

    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@lru_cache(maxsize=100)
def _get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)

    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return (
        get_slopes_power_of_2(closest_power_of_2) + _get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
    )


@lru_cache(maxsize=100)
def get_slopes(num_heads: int, batch_size: int, device: torch.device):
    return (
        torch.Tensor(_get_slopes(num_heads))
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
        .to(device)
    )


def alibi(attn: torch.Tensor, modulation_fn: t.Callable = None) -> torch.Tensor:
    """Adds ALiBi position embeddings."""
    B, num_heads, _, seq_len = attn.shape
    device = attn.device
    dtype = attn.dtype

    with torch.no_grad():
        position = torch.arange(seq_len, device=device)
        relative_position = position[None, :] - position[:, None]
        relative_position = (relative_position.unsqueeze(0).unsqueeze(0).repeat(B, num_heads, 1, 1)).to(
            dtype
        )  # [B, num_heads, N, N]
        slopes = get_slopes(num_heads, B, device)  # [B, num_heads, 1, 1]
        alibi = slopes * relative_position

    if modulation_fn is not None:
        alibi = modulation_fn(alibi)

    return attn + alibi


@lru_cache(maxsize=100)
@torch.jit.script
def sincos_1d_position_embedding(
    embed_dim: int, num_tokens: int, device: torch.device, temperature: float = 10000.0
) -> torch.Tensor:
    """One-dimensional positional embedding. Range: [-1, 1]

    :param embed_dim: Feature dimension
    :param num_tokens: Number of tokens
    :param device: The device to place the resultant position embedding.
    :param temperature: Temperature for sincos modulation.
    :return: Positional embedding tensor.

    """
    with torch.no_grad():
        assert embed_dim % 2 == 0
        pos_embed = torch.zeros(num_tokens, embed_dim, device=device)
        position = torch.arange(0, num_tokens, device=device).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, embed_dim, 2, dtype=torch.float, device=device) * -(math.log(temperature) / embed_dim))
        )
        pos_embed[:, 0::2] = torch.sin(position.float() * div_term)
        pos_embed[:, 1::2] = torch.cos(position.float() * div_term)
        return pos_embed.unsqueeze(0)  # [1, num_tokens, embed_dim]


class SinCosPE(nn.Module):
    """Simple SinCos Position Embedding Layer."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Add sincos positional embeddings.

        :param x: Input tensor of shape [B, S, F].
        :returns: Tokens with sincos PE added.

        """
        return x + sincos_1d_position_embedding(
            embed_dim=x.shape[-1], num_tokens=x.shape[1], device=x.device
        )


class Identity(nn.Module):
    """Identity with *args and **kwargs."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, samples, *unused_args, **unused_kwargs):
        return samples
