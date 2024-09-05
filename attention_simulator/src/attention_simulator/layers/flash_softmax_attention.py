#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""FlashSoftmax attention layers."""

import functools
import typing as t

import torch
import torch.nn.functional as F
from torch import nn

# Local imports
from attention_simulator.layers.position_embedding import apply_rotary_emb

# Attempt to load softmax flash attention if it exists.
try:
    from flash_attn import flash_attn_func
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    pass


def torch_softmax_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_drop: float = 0.0,
    causal: bool = False,
    prescale: bool = True,
):
    """Reusable method for torch SDPA.

    :param q: (batch_size, seqlen, nheads_q, headdim)
    :param k: (batch_size, seqlen, nheads_kv, headdim)
    :param v: (batch_size, seqlen, nheads_kv, headdim)
    :param attn_drop: prob of attn dropout
    :param window_size: Unsupported for F.SDPA, kept for API.
    :param causal: apply causal masking if true.
    :param prescale: If `True`, apply scaling to logits before dot-product attention.

    """
    # -1 element of SDPA is head dimension, -2 element of SDPA is sequence length
    # einops: "batch seq heads headdim -> batch heads seq headdim"
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    scale_factor = q.size(-1) ** (-0.25)

    out = F.scaled_dot_product_attention(
        query=q * scale_factor if prescale else q,
        key=k * scale_factor if prescale else k,
        value=v,
        dropout_p=attn_drop,
        is_causal=causal,
        scale=1.0 if prescale else None,
    )

    # We always return batch seq embed_dim
    # einops: "batch heads seq headdim -> batch seq (heads headdim)"
    return out.permute(0, 2, 1, 3).flatten(2)


def flash2_softmax_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_drop: float = 0.0,
    window_size: t.Tuple = (-1, -1),
    alibi_slopes: t.Optional[torch.Tensor] = None,
    causal: bool = False,
    prescale: bool = True,
):
    """Unfused attn helper.

    NOTE: Routes to torch if not BF16/FP16.

    :param q: (batch_size, seqlen, nheads_q, headdim)
    :param k: (batch_size, seqlen, nheads_kv, headdim)
    :param v: (batch_size, seqlen, nheads_kv, headdim)
    :param attn_drop: prob of attn dropout
    :param window_size: If > -1 then apply local window attention.
    :param alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    :param causal: apply causal masking if true.
    :param prescale: If `True`, apply scaling to logits before dot-product attention.

    """
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype not in (torch.float16, torch.bfloat16)
        or v.dtype not in (torch.float16, torch.bfloat16)
    ):
        # Handle FP32 / TF32 tensors which aren't supported.
        return torch_softmax_attn(
            q=q, k=k, v=v, attn_drop=attn_drop, causal=causal, prescale=prescale
        )  # (batch_size, seqlen, nheads * headdim)

    scale_factor = q.size(-1) ** (-0.25)

    out = flash_attn_func(
        q=q * scale_factor if prescale else q,
        k=k * scale_factor if prescale else k,
        v=v,
        softmax_scale=1.0 if prescale else None,
        dropout_p=attn_drop,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        causal=causal,
    ).flatten(
        2
    )  # (batch_size, seqlen, nheads * headdim)
    return out


def fused_softmax_torch_attn(
    qkv: t.Optional[torch.Tensor] = None,
    attn_drop: float = 0.0,
    causal: bool = True,
    prescale: bool = True,
    **unused_kwargs,
):
    """Helper to unfold a fused QKV --> F.SDPA

    :param qkv: (batch_size, seqlen, 3, nheads, headdim).
    :param attn_drop: Prob of attn dropout.
    :param causal: apply causal masking if true.
    :param prescale: If `True`, apply scaling to logits before dot-product attention.

    """
    q, k, v = qkv.unbind(2)
    return torch_softmax_attn(
        q=q, k=k, v=k, attn_drop=attn_drop, causal=causal, prescale=prescale
    )  # (batch_size, seqlen, nheads * headdim)


def fused_softmax_flash2_attn(
    qkv: torch.Tensor,
    attn_drop: float = 0.0,
    window_size: t.Tuple = (-1, -1),
    alibi_slopes: t.Optional[torch.Tensor] = None,
    causal: bool = True,
    prescale: bool = True,
):
    """Fused attn helper.

    :param qkv: (batch_size, seqlen, 3, nheads, headdim)
    :param attn_drop: Prob of attn dropout
    :param window_size: If > -1 then apply local window attention.
    :param alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    :param causal: apply causal masking if true.
    :param prescale: If `True`, apply scaling to logits before dot-product attention.

    """
    scale_factor = torch.ones([1, 1, 3, 1, 1], dtype=qkv.dtype, device=qkv.device)
    scale_factor[:, :, :-1] *= qkv.size(-1) ** (-0.25)

    out = flash_attn_qkvpacked_func(
        qkv=qkv * scale_factor if prescale else qkv,
        softmax_scale=1.0 if prescale else None,
        dropout_p=attn_drop,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    ).flatten(2)
    return out  # (batch_size, seqlen, nheads * headdim)


class FlashSoftmaxAttention(nn.Module):
    """Attention layer with fused QKV projections."""

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        num_heads: int = 8,
        bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: t.Tuple = (-1, -1),
        alibi_slopes: t.Optional[torch.Tensor] = None,
        norm_layer: t.Callable = nn.LayerNorm,
        which_linear: t.Callable = nn.Linear,
        qk_transform: t.Optional[t.Callable] = None,
        which_flash_attn: t.Callable = flash2_softmax_attn,
    ):
        """Flash attention layer.

        :param dim: Dimensionality of the input and output embeddings.
        :param causal: Apply causal attention.
        :param num_heads: Number of attention heads.
        :param bias: If True, add a learnable bias to query, key, value and proj.
        :param qk_norm: QK norm from https://arxiv.org/abs/2302.05442
        :param attn_drop: Dropout probability for the attention matrix.
        :param proj_drop: Dropout probability for the output projection.
        :param window_size: Tuple describing window attention (if > -1).
        :param alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        :param norm_layer: Normalization layer callable.
        :param which_linear: Linear layer type to use.
        :param qk_transform: Transform for xf(q), xf(k) -- eg: RoPE.
        :param which_flash_attn: Which flash attention function to call.

        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.causal = causal
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.head_dim = dim // num_heads

        # Swap between flash and torch F.SDPA
        self.which_flash_attn = which_flash_attn
        if which_flash_attn == torch_softmax_attn:
            assert window_size[0] == -1 and window_size[1] == -1, "window_size not supported with torch F.SDPA"
            assert alibi_slopes is None, "ALiBi not supported with torch F.SDPA"

        # These functions can be used to modulate {q, k}.
        self.qk_transform = qk_transform

        # Layers to handle projections and normalization.
        self.qkv = which_linear(dim, dim * 3, bias=bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = which_linear(dim, dim, bias=bias)

        # Optional dropout layers.
        self.attn_drop = attn_drop  # Handled inside Flash.
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self) -> str:
        """Extra repr for the class."""
        return (
            f"causal={self.causal}, \n"
            f"num_heads={self.num_heads}, \n"
            f"head_dim={self.head_dim}, \n"
            f"window_size={self.window_size} \n"
            f"alibi_slopes={self.alibi_slopes}, \n"
            f"which_flash_attn={self.which_flash_attn} \n"
            f"qk_transform={self.qk_transform},"
        )

    def forward(
        self,
        x: torch.Tensor,
        **unused_kwargs,
    ) -> torch.Tensor:
        output = {}  # house all results

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        # QK norm.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply an optional function on Q and K (eg: RoPE).
        if self.qk_transform is not None:
            q, k = self.qk_transform(q, k, q_pos_offset=0, k_pos_offset=0)

        # Apply attention to V and project back using a linear layer.
        attn_times_v = self.which_flash_attn(
            q=q.to(v.dtype),
            k=k.to(v.dtype),
            v=v,
            attn_drop=self.attn_drop,
            causal=self.causal,
            window_size=self.window_size,
            alibi_slopes=self.alibi_slopes,
        )

        final_proj_output = self.proj_drop(self.proj(attn_times_v))

        # Return what is possible for analysis.
        output.update(
            {
                "attn_times_v": attn_times_v,
                "attn_proj": final_proj_output,
            }
        )
        return output


class FlashSoftmaxCrossAttention(nn.Module):
    """A cross-attention Flash implementation."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        causal: bool = True,
        num_heads: int = 8,
        bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: t.Tuple = (-1, -1),
        alibi_slopes: t.Optional[torch.Tensor] = None,
        norm_layer: t.Callable = nn.LayerNorm,
        which_linear: t.Callable = nn.Linear,
        qk_transform: t.Optional[t.Callable] = None,
        which_flash_attn: t.Callable = flash2_softmax_attn,
    ):
        """Cross attention layer.

        :param q_dim: Dimensionality of the Q tensor.
        :param kv_dim: Dimensionality of the {K, V} tensors.
        :param causal: Apply causal attention.
        :param num_heads: Number of attention heads.
        :param bias: If True, add a learnable bias to query, key, value and proj.
        :param qk_norm: QK norm from https://arxiv.org/abs/2302.05442
        :param attn_drop: Dropout probability for the attention matrix.
        :param proj_drop: Dropout probability for the output projection.
        :param window_size: Tuple describing window attention (if > -1).
        :param alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        :param norm_layer: Normalization layer callable.
        :param which_linear: Linear layer type to use.
        :param qk_transform: Transform for xf(q), xf(k) -- eg: RoPE.
        :param which_flash_attn: Which flash attention function to call.

        """
        super().__init__()
        assert q_dim % num_heads == 0 and kv_dim % num_heads == 0, "Dimensions should be divisible by num_heads"
        self.causal = causal
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.head_dim_q = q_dim // num_heads
        self.head_dim_kv = kv_dim // num_heads

        # Swap between flash and torch F.SDPA
        self.which_flash_attn = which_flash_attn
        if which_flash_attn == torch_softmax_attn:
            assert window_size[0] == -1 and window_size[1] == -1, "window_size not supported with torch F.SDPA"
            assert alibi_slopes is None, "ALiBi not supported with torch F.SDPA"

        # These functions can be used to modulate {q, k}.
        self.qk_transform = qk_transform

        # Layers to handle projections and normalization.
        self.q_proj = which_linear(q_dim, q_dim, bias=bias)
        self.kv_proj = which_linear(kv_dim, kv_dim * 2, bias=bias)
        self.q_norm = norm_layer(self.head_dim_q) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim_kv) if qk_norm else nn.Identity()
        self.v_norm = norm_layer(self.head_dim_kv)  # Always norm on V.
        self.proj = which_linear(kv_dim, q_dim, bias=bias)

        # Optional dropout layers.
        self.attn_drop = attn_drop  # Handled inside Flash.
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self) -> str:
        """Extra repr for the class."""
        return (
            f"causal={self.causal}, \n"
            f"num_heads={self.num_heads}, \n"
            f"head_dim_q={self.head_dim_q}, \n"
            f"head_dim_kv={self.head_dim_kv}, \n"
            f"window_size={self.window_size} \n"
            f"alibi_slopes={self.alibi_slopes}, \n"
            f"which_flash_attn={self.which_flash_attn} \n"
            f"qk_transform={self.qk_transform},"
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        **unused_kwargs,
    ) -> torch.Tensor:
        output = {}  # house all results

        q_shape = q.shape
        kv_shape = kv.shape

        q = self.q_proj(q).reshape(q_shape[0], q_shape[1], self.num_heads, self.head_dim_q)
        kv = self.kv_proj(kv).reshape(kv_shape[0], kv_shape[1], 2, self.num_heads, self.head_dim_kv)
        k, v = kv.unbind(2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Apply an optional function on Q and K (eg: RoPE).
        if self.qk_transform is not None:
            q, k = self.qk_transform(q, k, q_pos_offset=0, k_pos_offset=0)

        # Force the dtype back.
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # Apply attention to V and project back using a linear layer.
        attn_times_v = self.which_flash_attn(
            q=q,
            k=k,
            v=v,
            attn_drop=self.attn_drop,
            causal=self.causal,
            window_size=self.window_size,
            alibi_slopes=self.alibi_slopes,
        )
        final_proj_output = self.proj_drop(self.proj(attn_times_v))

        output.update(
            {
                "attn_times_v": attn_times_v,
                "attn_proj": final_proj_output,
            }
        )
        return output


RoPEFlashSoftmaxAttention = functools.partial(FlashSoftmaxAttention, qk_transform=apply_rotary_emb)
RoPEFlashCrossSoftmaxAttention = functools.partial(FlashSoftmaxCrossAttention, qk_transform=apply_rotary_emb)
