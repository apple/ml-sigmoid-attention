#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Activation functions for attention mechanisms."""

import torch
import torch.nn.functional as F


def softmax_attention(qk_matrix: torch.Tensor) -> torch.Tensor:
    """Applies softmax to QK^T

    :param qk_matrix: QK^T matrix.
    :return: Softmax attention.

    """
    return F.softmax(qk_matrix, dim=-1)


def unnormalized_sigmoid_attention(qk_matrix: torch.Tensor) -> torch.Tensor:
    """Applies sigmoid to QK^T

    :param qk_matrix: QK^T matrix.
    :return: Sigmoid attention.

    """
    return torch.sigmoid(qk_matrix)


def sqrt_seq_len_normalized_sigmoid_attention(
    qk_matrix: torch.Tensor,
) -> torch.Tensor:
    """Applies sigmoid to QK^T with 1/sqrt(sequence_length) normalization.

    :param qk_matrix: QK^T matrix.
    :return: Sigmoid attention.

    """
    seq_len = qk_matrix.size(-1)  # [L x L], so either dim will do.
    scaling = torch.diag(1.0 / torch.arange(1, seq_len + 1, device=qk_matrix.device).sqrt()).to(
        qk_matrix.dtype
    )  # [L x L]
    scaling = scaling.unsqueeze(0).unsqueeze(0).expand_as(qk_matrix)  # [L x L] --> [B, num_heads, L, L]
    return torch.einsum("bhij,bhik->bhij", torch.sigmoid(qk_matrix), scaling)


def seq_len_normalized_sigmoid_attention(qk_matrix: torch.Tensor) -> torch.Tensor:
    """Applies sigmoid to QK^T with 1/sequence_length normalization.

    :param qk_matrix: QK^T matrix.
    :return: Sigmoid attention.

    """
    seq_len = qk_matrix.size(-1)  # [L x L], so either dim will do.
    scaling = torch.diag(1.0 / torch.arange(1, seq_len + 1, device=qk_matrix.device)).to(qk_matrix.dtype)  # [L x L]
    scaling = scaling.unsqueeze(0).unsqueeze(0).expand_as(qk_matrix)  # [L x L] --> [B, num_heads, L, L]
    return torch.einsum("bhij,bhik->bhij", torch.sigmoid(qk_matrix), scaling)


ACTIVATION_MAPPING = {
    "softmax": softmax_attention,
    "sigmoid": unnormalized_sigmoid_attention,
    "sqrt_seq_len_normalized_sigmoid": sqrt_seq_len_normalized_sigmoid_attention,
    "seq_len_normalized_sigmoid": seq_len_normalized_sigmoid_attention,
}
