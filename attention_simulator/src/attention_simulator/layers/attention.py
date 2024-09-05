#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Attention modules for causal and non-causal attention."""

import functools
import typing as t

import torch
from torch import nn

# Local imports
from attention_simulator.layers.position_embedding import alibi
from attention_simulator.layers.position_embedding import apply_rotary_emb


class Attention(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        dim: int,
        attn_activ_fn: t.Callable,
        num_heads: int = 8,
        bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_bias: t.Optional[float] = None,
        attn_temp: t.Optional[float] = None,
        norm_layer: t.Callable = nn.LayerNorm,
        which_linear: t.Callable = nn.Linear,
        masking_fn: t.Optional[t.Callable] = None,
        qk_transform: t.Optional[t.Callable] = None,
        attn_logits_transform: t.Optional[t.Callable] = None,
    ):
        """Attention layer.

        :param dim: Dimensionality of the input and output embeddings.
        :param attn_activ_fn: f(attention_matrix).
        :param num_heads: Number of attention heads.
        :param bias: If True, add a learnable bias to query, key, value and proj.
        :param qk_norm: QK norm from https://arxiv.org/abs/2302.05442
        :param attn_drop: Dropout probability for the attention matrix.
        :param proj_drop: Dropout probability for the output projection.
        :param norm_layer: Normalization layer callable.
        :param which_linear: Linear layer type to use.
        :param masking_fn: g(attention_matrix, mask).
        :param qk_transform: Transform for xf(q), xf(k) -- eg: RoPE.
        :param attn_logits_transform: Transform to xf(QK^T/sqrt(d)) -- eg ALiBi.

        """
        super().__init__()
        self.num_heads = num_heads
        assert (
            dim % num_heads == 0
        ), f"num_heads needs to be divisible by embed dim, got dim={dim} & num_heads={num_heads}"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # These functions enable toggling activation and masking.
        self.attn_activ_fn = attn_activ_fn
        self.masking_fn = masking_fn

        # These functions can be used to modulate {q, k} and {attn}.
        self.qk_transform = qk_transform
        self.attn_logits_transform = attn_logits_transform

        # Layers to handle projections and normalization.
        self.qkv = which_linear(dim, dim * 3, bias=bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = which_linear(dim, dim, bias=bias)

        # Optional dropout layers.
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional learnable scalar temp and bias.
        if attn_temp is not None:
            self.register_buffer("attn_temp", torch.Tensor([attn_temp]))
        else:
            self.register_buffer("attn_temp", None)

        # Register the attention temperature and bias offsets (or init to unit values).
        # Temp is 0.0 and not 1.0 because we do exp().
        self.register_buffer("attn_bias", torch.Tensor([attn_bias if attn_bias is not None else 0.0]))
        self.register_buffer("attn_temp", torch.Tensor([attn_temp if attn_temp is not None else 0.0]))

    def extra_repr(self) -> str:
        """Extra repr for the class."""
        return (
            f"attn_temp={self.attn_temp}, \n"
            f"attn_bias={self.attn_bias}, \n"
            f"num_heads={self.num_heads}, \n"
            f"head_dim={self.head_dim}, \n"
            f"masking_fn={self.masking_fn}, \n"
            f"attn_activ_fn={self.attn_activ_fn}, \n"
            f"qk_transform={self.qk_transform}, \n"
            f"attn_logits_transform={self.attn_logits_transform},"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: t.Optional[torch.Tensor] = None,
        **unused_kwargs,
    ):
        output = {}  # house all results

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each are: [B, num_heads, seq_len, C // num_heads]

        q = self.q_norm(q)
        k = self.k_norm(k)
        # Apply an optional function on Q and K (eg: RoPE).
        if self.qk_transform is not None:
            q, k = self.qk_transform(q, k, q_pos_offset=0, k_pos_offset=0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply an optional function on attn logits (eg: ALiBi).
        if self.attn_logits_transform is not None:
            attn = self.attn_logits_transform(attn)

        # Apply causual / other masking.
        if mask is not None:
            attn = self.masking_fn(attn=attn, mask=mask)

        # Scale by temperature [default is 0.0]
        attn = attn * torch.exp(self.attn_temp)

        # Add attention bias [default is 0.0]
        attn = attn + self.attn_bias

        # Activation function and dropout.
        attn_matrix = self.attn_activ_fn(attn)
        attn = self.attn_drop(attn_matrix)

        # Apply attention to V and project back using a linear layer.
        attn_times_v = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(attn_times_v)
        final_proj_output = self.proj_drop(x)

        # Return everything for analysis.
        output.update(
            {
                "attn_matrix": attn_matrix,
                "attn_times_v": attn_times_v,
                "attn_proj": final_proj_output,
            }
        )
        return output


RoPEAttention = functools.partial(Attention, qk_transform=apply_rotary_emb)
ALiBiAttention = functools.partial(Attention, attn_logits_transform=alibi)


class CrossAttention(nn.Module):
    """Cross-attention layer.

    :param q_dim: Dimensionality of the Q tensor.
    :param kv_dim: Dimensionality of the {K, V} tensors.
    :param attn_activ_fn: f(attention_matrix).
    :param num_heads: Number of attention heads.
    :param bias: If True, add a learnable bias to query, key, value and proj.
    :param qk_norm: QK norm from https://arxiv.org/abs/2302.05442
    :param attn_drop: Dropout probability for the attention matrix.
    :param proj_drop: Dropout probability for the output projection.
    :param attn_bias: Optional learnable bias to add to the attention matrix.
    :param attn_temp: Optional learnable temperature scalar to modulate the attention matrix.
    :param norm_layer: Normalization layer callable.
    :param which_linear: Linear layer type to use.
    :param masking_fn: g(attention_matrix, mask).
    :param qk_transform: Transform for xf(q), xf(k) -- eg: RoPE.
    :param attn_logits_transform: Transform to xf(QK^T/sqrt(d)) -- eg ALiBi.

    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        attn_activ_fn: t.Callable,
        num_heads: int = 8,
        bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_bias: t.Optional[float] = None,
        attn_temp: t.Optional[float] = None,
        norm_layer: t.Callable = nn.LayerNorm,
        which_linear: t.Callable = nn.Linear,
        masking_fn: t.Optional[t.Callable] = None,
        qk_transform: t.Optional[t.Callable] = None,
        attn_logits_transform: t.Optional[t.Callable] = None,
    ):
        super().__init__()

        assert q_dim % num_heads == 0, "q_dim should be divisible by num_heads"
        assert kv_dim % num_heads == 0, "kv_dim should be divisible by num_heads"
        self.head_dim_q = q_dim // num_heads
        self.head_dim_kv = kv_dim // num_heads
        self.scale = self.head_dim_q**-0.5

        self.num_heads = num_heads
        self.attn_activ_fn = attn_activ_fn
        self.masking_fn = masking_fn

        self.qk_transform = qk_transform
        self.attn_logits_transform = attn_logits_transform

        self.q_proj = which_linear(q_dim, q_dim, bias=bias)
        self.kv_proj = which_linear(kv_dim, kv_dim * 2, bias=bias)
        self.q_norm = norm_layer(self.head_dim_q) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim_kv) if qk_norm else nn.Identity()
        self.v_norm = norm_layer(self.head_dim_kv)  # Always norm on V.
        self.proj = which_linear(kv_dim, q_dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Register the attention temperature and bias offsets (or init to unit values).
        # Temp is 0.0 and not 1.0 because we do exp().
        self.register_buffer("attn_bias", torch.Tensor([attn_bias if attn_bias is not None else 0.0]))
        self.register_buffer("attn_temp", torch.Tensor([attn_temp if attn_temp is not None else 0.0]))

    def extra_repr(self) -> str:
        """Extra repr for the class."""
        return (
            f"attn_temp={self.attn_temp}, \n"
            f"attn_bias={self.attn_bias}, \n"
            f"num_heads={self.num_heads}, \n"
            f"head_dim_q={self.head_dim_q}, \n"
            f"head_dim_kv={self.head_dim_kv}, \n"
            f"masking_fn={self.masking_fn}, \n"
            f"attn_activ_fn={self.attn_activ_fn}, \n"
            f"qk_transform={self.qk_transform}, \n"
            f"attn_logits_transform={self.attn_logits_transform},"
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: t.Optional[torch.Tensor] = None,
        **unused_kwargs,
    ):
        output = {}  # house all results

        q_shape = q.shape
        kv_shape = kv.shape

        q = self.q_proj(q).reshape(q_shape[0], q_shape[1], self.num_heads, self.head_dim_q).transpose(1, 2)
        kv = (
            self.kv_proj(kv)
            .reshape(kv_shape[0], kv_shape[1], 2, self.num_heads, self.head_dim_kv)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Apply an optional function on Q and K (eg: RoPE).
        if self.qk_transform is not None:
            q, k = self.qk_transform(q, k, q_pos_offset=0, k_pos_offset=0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.attn_logits_transform is not None:
            attn = self.attn_logits_transform(attn)

        # Apply causual / other masking.
        if mask is not None:
            attn = self.masking_fn(attn=attn, mask=mask)

        # Scale by temperature [default is 0.0]
        attn = attn * torch.exp(self.attn_temp)

        # Add attention bias [default is 0.0]
        attn = attn + self.attn_bias

        attn_matrix = self.attn_activ_fn(attn)
        attn = self.attn_drop(attn_matrix)

        attn_times_v = (attn @ v).transpose(1, 2).reshape(q_shape[0], q_shape[1], kv_shape[2])
        x = self.proj(attn_times_v)
        final_proj_output = self.proj_drop(x)

        output.update(
            {
                "attn_matrix": attn_matrix,
                "attn_times_v": attn_times_v,
                "attn_proj": final_proj_output,
            }
        )
        return output


RoPECrossAttention = functools.partial(CrossAttention, qk_transform=apply_rotary_emb)
ALiBiCrossAttention = functools.partial(CrossAttention, attn_logits_transform=alibi)
