#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Helper functions."""

import typing as t

import torch
import tree


def flatten_dict(
    x: t.Dict[str, t.Dict[str, t.Any]],
    key_separator: str = "_",
) -> t.Dict[str, t.Any]:
    """
    Take a dictionary of dictionaries and flatten the outer level, merging their keys.
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    Example:
    .. code-block:: python
        d = {
            'a': {'b': 1, 'c': 2},
            'm': {'n': 3},
            'z': 4
        }
        fd = flatten_dict(d)
        fd
        {
            'z': 4,  # <-- unchanged
            'a_b': 1,
            'a_c': 2,
            'm_n': 3,
        }
    """

    def _recursive_flatten(d: t.Iterable, parent_key: str = "", sep: str = "_"):
        """Flattens a dict recursively."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, t.MutableMapping):
                items.extend(_recursive_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    # Flatten the remainder of the keys
    flattened = _recursive_flatten(x, sep=key_separator)
    num_flattened = len(tree.flatten(flattened))
    num_flattened_orig = len(tree.flatten(x))
    assert num_flattened == num_flattened_orig, (
        f"Number of flattenned keys ({num_flattened}) != " f"number of original keys ({num_flattened_orig})"
    )
    return flattened


def sanity_check_block_configs(
    block_fn: t.Optional[t.Callable] = None,
    num_blocks: t.Optional[int] = None,
    block_configs: t.Optional[t.List[t.Callable]] = None,
):
    """Sanity check the block configs.

    :param block_fn: Callable transformer block (used for same block_fn)
    :param num_blocks: Number of transformer blocks (used for same block_fn).
    :param block_configs: Specifies each block independently.

    """
    if block_configs is not None:
        if num_blocks is not None:
            raise ValueError("Cannot specify both 'num_blocks' and 'block_configs'.")
    elif num_blocks is not None:
        if block_fn is None:
            raise ValueError("'block_fn' must be specified when using 'num_blocks'.")
    else:
        raise ValueError("Either 'num_blocks' or 'block_configs' must be specified.")


def extract_tensor_from_maybe_dict(tensor_or_dict: t.Union[t.Dict[str, torch.Tensor], torch.Tensor], field_name: str):
    """If tensor_or_dict extract the field field_name, otherwise return.

    :param tensor_or_dict: A tensor or a dict of tensors.
    :param field_name: The field name to extract
    :returns: Extracted Tensor.

    """
    if isinstance(tensor_or_dict, dict):
        return tensor_or_dict[field_name]

    return tensor_or_dict


def unsqueeze_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unsqueeze x to have the same dims as target.

    :param x: Input tensor.
    :param target: Target tensor whose dims we want to match.
    :returns: Unsqueezed tensor.

    """
    diff = target.dim() - x.dim()
    if diff < 0:
        raise ValueError(f"Target tensor has fewer dimensions ({target.dim()}) than input tensor ({x.dim()})")

    # Create a view with added dimensions
    new_shape = (1,) * diff + x.shape
    return x.view(new_shape)

