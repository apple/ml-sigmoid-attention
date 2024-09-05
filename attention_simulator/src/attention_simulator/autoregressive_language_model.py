#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Training loop for a simple autoregressive transformer."""

import random
import typing as t

import pandas as pd
import torch
import torch.nn.functional as F
import tree
from torch import nn
from tqdm import tqdm

# Local imports
from attention_simulator.helpers.utils import flatten_dict
from attention_simulator.layers.masking import create_causal_mask
from optorch.amp import scale_loss
from optorch.amp import unscale_grads
from optorch.amp import update_scale


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the cross-entropy loss with masking for padding tokens.

    :param logits: Logits tensor of shape (batch_size, sequence_length, vocab_size).
    :param targets: Target tensor of shape (batch_size, sequence_length).
    :param mask: Mask tensor of shape (batch_size, sequence_length),
                 where 1 indicates a valid token and 0 indicates a padded token.
    :return: Cross-entropy loss.

    """
    batch_size, sequence_length, vocab_size = logits.shape
    logits = logits.view(batch_size * sequence_length, vocab_size)
    targets = targets.view(batch_size * sequence_length)
    mask = mask.view(batch_size * sequence_length)

    # Compute the cross-entropy loss only for the valid tokens
    loss = F.cross_entropy(logits, targets.long(), reduction="none")
    return (loss * mask).sum() / mask.sum()


def compute_stats(
    tensor_iterable: t.Iterable[torch.Tensor],
    reduction_map: t.Dict[str, t.Callable[[torch.Tensor], torch.Tensor]],
) -> t.Dict[str, float]:
    """Compute statistics for an iterable of tensors using a set of reducers.

    :param tensor_iterable: Iterable of tensors.
    :param reduction_map: Dictionary mapping statistic names to reduction functions.
    :return: Dictionary of computed statistics.

    """
    cache_key = random.random()  # Generate a random cache key for each invocation
    results = {}
    for reducer_name, reducer_fn in reduction_map.items():

        def cached_fn(tensor):
            return reducer_fn(tensor, cache_key)

        results[reducer_name] = tree.map_structure(cached_fn, tensor_iterable)

    return results


def gradient_step(
    model: nn.Module,
    loss: torch.Tensor,
    model_params: t.Dict[str, torch.Tensor],
    outputs: t.Dict[str, torch.Tensor],
    scaler_state: t.Optional[t.Dict[str, t.Any]],
    device: torch.device,
) -> t.Tuple[t.Dict[str, torch.Tensor], t.Dict[str, torch.Tensor], torch.Tensor]:
    """Perform the gradient computation step.

    :param model: The model.
    :param loss: The computed loss.
    :param model_params: Dictionary of model parameters.
    :param outputs: Dictionary of model outputs.
    :param scaler_state: Optional mixed precision state dictionary.
    :param device: The device to perform the computations on.
    :return: (model gradients, intermediary gradients, found_inf tensor).

    """
    intermediary_grads, model_grads = {}, {}
    found_inf = torch.full((), 0.0, dtype=torch.float32, device=device)

    if torch.is_grad_enabled():
        # Compute gradients for model parameters and intermediary variables.
        params_and_outputs = {**model_params, **outputs}
        grads = torch.autograd.grad(loss, params_and_outputs.values(), create_graph=False)
        grads = dict(zip(params_and_outputs.keys(), grads))

        # Extract model gradients and intermediary gradients.
        model_grads = {k: grads[k] for k in model_params}
        intermediary_grads = {k: grads[k] for k in outputs}

        # Mixed precision inf checks.
        if scaler_state is not None:
            model_grads, found_inf = unscale_grads(model_grads, scaler_state, device)

    return model_grads, intermediary_grads, found_inf


def optimizer_step(
    model_params: t.Dict[str, torch.Tensor],
    model_grads: t.Dict[str, torch.Tensor],
    opt_state: t.Dict[str, t.Any],
    optimizer_config: t.Any,
    optimizer_step_fn: t.Callable,
    scaler_state: t.Optional[t.Dict[str, t.Any]],
    found_inf: torch.Tensor,
) -> t.Tuple[t.Dict[str, torch.Tensor], t.Dict[str, t.Any], t.Optional[t.Dict[str, t.Any]]]:
    """Perform a single optimizer update step.

    :param model_params: Dictionary of model parameters.
    :param model_grads: Dictionary of model gradients.
    :param opt_state: Optimizer state dictionary.
    :param optimizer_config: Configuration for the optimizer.
    :param optimizer_step_fn: Vectorized optimizer function.
    :param scaler_state: Optional mixed precision state dictionary.
    :param found_inf: Tensor indicating if infs/NaNs were found.
    :return: Updated (model parameters, optimizer state, scaler state).

    """
    # Model params are updated in-place if no infs/NaNs are found.
    if opt_state is not None and optimizer_config is not None and found_inf == 0:
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            model_params, opt_state = optimizer_step_fn(model_params, model_grads, opt_state, optimizer_config)

            # Update mixed precision state.
            if scaler_state is not None:
                update_scale(scaler_state, found_inf)

    return model_params, opt_state, scaler_state


def infer_causal_mask(
    model: nn.Module,
    input_sequence: torch.Tensor,
    attn_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Infers the correct causal mask.

    :param model: The nn.Module.
    :param input_sequence: The input sequence to the model.
    :param attn_mask: Optional data mask.
    :param device: Compute device.
    :return: Appropriate mask for encoder + decoder or decoder only.

    """
    seq_len = input_sequence.shape[1]
    causal_mask = create_causal_mask(
        seq_len=seq_len, device=device, attn_mask=attn_mask, target_seq_len=None
    )  # The causal mask is a union of the hf data mask and a triangular one.
    return causal_mask


def step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    attn_mask: torch.Tensor,
    opt_state: t.Dict[str, t.Any],
    optimizer_step_fn: t.Callable,
    optimizer_config: t.Any,
    loss_func: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    reduction_map: t.Dict[str, t.Callable[[torch.Tensor], torch.Tensor]],
    scaler_state: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Dict[str, t.Any]:
    """Perform a single training/evaluation step.

    :param model: The model.
    :param x: Input tensor.
    :param y: Target tensor.
    :param attn_mask: Attention mask tensor.
    :param opt_state: Optimizer state dictionary.
    :param optimizer_step_fn: Optimizer stepping function.
    :param optimizer_config: Configuration for the optimizer.
    :param loss_func: Loss function.
    :param device: The device to perform the computations on.
    :param reduction_map: Dictionary mapping statistic names to reduction functions.
    :param scaler_state: Optional mixed precision state dictionary.
    :return: Dictionary containing the computed statistics.

    """
    # Move data and the mask to the correct device.
    input_sequence, target_sequence, attn_mask = (
        x.to(device),
        y.to(device),
        attn_mask.to(device),
    )
    causal_mask = infer_causal_mask(model=model, input_sequence=input_sequence, attn_mask=attn_mask, device=device)

    def compute_loss_and_outputs(
        model_params: t.Dict[str, torch.Tensor]
    ) -> t.Tuple[torch.Tensor, t.Dict[str, torch.Tensor]]:
        """Compute the loss and model outputs using functional forward."""
        outputs = torch.func.functional_call(model, model_params, (input_sequence, causal_mask))
        loss = loss_func(outputs["output_representation"], target_sequence, attn_mask)
        return loss, outputs

    # Unpack the parameters.
    model_params = dict(model.named_parameters())

    with torch.autocast(
        device_type=device.type if device.type != "mps" else "cpu",
        dtype=scaler_state["precision"] if scaler_state else torch.float32,
        enabled=scaler_state is not None,
    ):
        # Compute loss and outputs.
        unscaled_loss, outputs = compute_loss_and_outputs(model_params)
        loss = unscaled_loss

    # Unscope out of mixed precision.
    if scaler_state is not None:
        loss = scale_loss(unscaled_loss, scaler_state)

    # Compute gradients.
    model_grads, intermediary_grads, found_inf = gradient_step(
        model=model,
        loss=loss,
        model_params=model_params,
        outputs=outputs,
        scaler_state=scaler_state,
        device=device,
    )

    # Update model parameters and optimizer state.
    model_params, opt_state, scaler_state = optimizer_step(
        model_params=model_params,
        model_grads=model_grads,
        opt_state=opt_state,
        optimizer_config=optimizer_config,
        optimizer_step_fn=optimizer_step_fn,
        scaler_state=scaler_state,
        found_inf=found_inf,
    )

    # Compute all the statistics.
    with torch.no_grad():
        opt_state_stats = {}
        if opt_state is not None:
            opt_state_stats = compute_stats(opt_state, reduction_map)

        stats_dict = flatten_dict(
            {
                "output": compute_stats(outputs, reduction_map),
                "intermediary_grad": compute_stats(intermediary_grads, reduction_map),
                "model_grad": compute_stats(model_grads, reduction_map),
                "model_param": compute_stats(model_params, reduction_map),
                "optimizer_state": opt_state_stats,
            }
        )

    return {
        **stats_dict,
        "loss": unscaled_loss.item(),
        "opt_state": opt_state,
        "scaler_state": scaler_state,
    }


def make_dataloader_infinite(dataloader: t.Iterable):
    """Makes a dataloader infinite."""
    while True:
        yield from dataloader


def train(
    model: nn.Module,
    dataset: t.Iterable[t.Dict[str, torch.Tensor]],
    max_steps: int,
    device: torch.device,
    grapher: t.Any,
    opt_state: t.Dict[str, t.Any],
    optimizer_step: t.Callable,
    optimizer_config: t.Any,
    lr_scheduler: t.Callable,
    reduction_map: t.Dict[str, t.Callable[[torch.Tensor], torch.Tensor]],
    scaler_state: t.Optional[t.Dict[str, t.Any]] = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Train the autoregressive transformer model.

    :param model: Whatever model you want.
    :param dataset: Training dataset.
    :param max_steps: Number of training steps.
    :param device: Device to perform the computations on.
    :param grapher: An optional grapher object.
    :param opt_state: Optimizer state dictionary.
    :param optimizer_step: Vectorized optimizer function.
    :param optimizer_config: Configuration for the optimizer.
    :param lr_scheduler: Learning rate scheduler.
    :param reduction_map: Dictionary mapping statistic names to reduction functions.
    :param scaler_state: Optional mixed precision state dictionary.
    :param prefix: Optional prefix for logging.
    :return: Pandas DataFrame containing the training history.

    """
    history_data = []
    dataset_iterator = make_dataloader_infinite(dataset)

    for steps_seen in tqdm(range(max_steps)):
        batch = next(dataset_iterator)
        x, y, attn_mask = (
            batch["input_ids"],
            batch["targets"],
            batch["attention_mask"],
        )

        step_stats = step(
            model=model,
            x=x,
            y=y,
            attn_mask=attn_mask,
            opt_state=opt_state,
            optimizer_step_fn=optimizer_step,
            optimizer_config=optimizer_config,
            loss_func=cross_entropy_loss,
            device=device,
            reduction_map=reduction_map,
            scaler_state=scaler_state,
        )

        opt_state = step_stats.pop("opt_state")
        scaler_state = step_stats.pop("scaler_state")
        step_stats["lr"] = optimizer_config.max_lr

        if lr_scheduler is not None:  # Update LR schedule member & stats.
            optimizer_config.lr_schedule_value = lr_scheduler(steps_seen)
            step_stats["lr"] *= optimizer_config.lr_schedule_value

        step_stats["step"] = steps_seen
        history_data.append(step_stats)

        if grapher is not None:
            if steps_seen % grapher.config.log_every == 0:
                grapher.log(
                    {
                        **{f"train_{prefix}_{k}": v for k, v in step_stats.items()},
                        "GlobalSteps": steps_seen,
                    },
                )

    return pd.DataFrame(history_data)


def evaluate(
    model: nn.Module,
    dataset: t.Iterable[t.Dict[str, torch.Tensor]],
    device: torch.device,
    grapher: t.Any,
    reduction_map: t.Dict[str, t.Callable[[torch.Tensor], torch.Tensor]],
    scaler_state: t.Optional[t.Dict[str, t.Any]] = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Evaluate the autoregressive transformer model.

    :param model: Whatever model you want.
    :param dataset: Training dataset.
    :param device: Device to perform the computations on.
    :param grapher: An optional grapher object.
    :param precision: Evaluation precision.
    :param reduction_map: Dictionary mapping statistic names to reduction functions.
    :param scaler_state: Optional mixed precision state dictionary.
    :param prefix: Optional prefix for logging.
    :return: Pandas DataFrame containing the evaluation history.

    """
    history_data = []

    with torch.no_grad():
        for steps_seen, batch in tqdm(enumerate(dataset)):
            x, y, attn_mask = (
                batch["input_ids"],
                batch["targets"],
                batch["attention_mask"],
            )

            step_stats = step(
                model=model,
                x=x,
                y=y,
                attn_mask=attn_mask,
                opt_state=None,
                optimizer_step_fn=None,
                optimizer_config=None,
                scaler_state=scaler_state,
                loss_func=cross_entropy_loss,
                device=device,
                reduction_map=reduction_map,
            )

            scaler_state = step_stats.pop("scaler_state")
            step_stats["step"] = steps_seen
            history_data.append(step_stats)

            if grapher is not None:
                grapher.log(
                    {
                        **{f"eval_{prefix}_{k}": v for k, v in step_stats.items()},
                        "GlobalSteps": steps_seen,
                    }
                )

    return pd.DataFrame(history_data)


def sample_top_p(probs: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Perform top-p (nucleus) sampling on a probability distribution.

    https://github.com/pytorch/torchtune/blob/main/torchtune/utils/logits_transforms.py

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    :param probs: Probability scores.
    :param p: Probability threshold for top-p sampling.
    :return: Sampled token indices.

    """
    probs_sort, probs_index = torch.sort(probs, dim=-1, descending=True)
    probs_cumulative = probs_sort.cumsum(dim=-1)

    # Ignore tokens introducing more probability mass than needed
    discard_mask = probs_cumulative - probs_sort > p
    probs_sort[discard_mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # renormalize
    probs.scatter_(-1, probs_index, probs_sort)
    return probs


def sample_top_k(probs: torch.Tensor, k: torch.Tensor):
    """Filters the distribution to include the top-k highest prob tokens.

    https://github.com/pytorch/torchtune/blob/main/torchtune/utils/logits_transforms.py

    :param probs: Probability scores.
    :param k: The number of highest probability tokens to keep.
    :return: Sampled token indices.

    """
    top_k = min(k, probs.size(-1))
    probs_topk, _ = probs.topk(top_k)

    discard_mask = probs < probs_topk[..., -1]
    probs.masked_fill_(discard_mask, 0.0)

    probs.div_(probs.sum(dim=-1, keepdim=True))  # renormalize
    return probs


def generate(
    model: nn.Module,
    tokenizer: t.Any,
    input_text: str,
    max_length: int,
    temperature: float = 1.2,
    top_p: float = 0.95,
    top_k: int = 0,
    scaler_state: t.Optional[t.Dict[str, t.Any]] = None,
    device: torch.device = torch.device("mps"),
) -> str:
    """Generate text using the autoregressive transformer model.

    :param model: The autoregressive transformer model.
    :param tokenizer: The tokenizer used to encode and decode text.
    :param input_text: The input text prompt to start the generation.
    :param max_length: The maximum length of the generated text.
    :param temperature: The temperature for sampling. Higher values result => diverse output.
    :param top_p: The cumulative probability threshold for top-p sampling.
    :param top_k: The number of highest probability tokens to keep (0 disables).
    :param scaler_state: Optional mixed precision state dictionary.
    :param device: The device to run the generation on (default is MPS).
    :return: The generated text.

    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        for cur_pos in tqdm(range(max_length)):
            with torch.autocast(
                device_type=device.type if device.type != "mps" else "cpu",
                dtype=scaler_state["precision"] if scaler_state else torch.float32,
                enabled=scaler_state is not None,
            ):
                # Gather the model params and inputs.
                model_params = dict(model.named_parameters())
                model_inputs = input_ids

                # Construct the causal mask.
                mask_kwarg = {
                    "mask": infer_causal_mask(
                        model=model,
                        input_sequence=input_ids,
                        attn_mask=None,
                        device=device,
                    )
                }

                # Infer the model
                outputs = torch.func.functional_call(
                    model,
                    model_params,
                    (model_inputs,),
                    kwargs={**mask_kwarg},
                )

                # Scale by temperature and activate.
                logits = outputs["output_representation"][:, -1, :] / temperature
                probs = logits.softmax(dim=-1)

                # Optional sampling with top-p and top-k filtering.
                if top_p > 0.0 and top_p < 1.0:
                    probs = sample_top_p(probs=probs, p=top_p)

                if top_k > 0:
                    probs = sample_top_k(probs=probs, k=top_k)

                next_token = torch.multinomial(probs, num_samples=1)

                # Concat with inputs for next stage
                # TODO(jramapuram): add EOS break logic.
                input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids.squeeze().tolist(), clean_up_tokenization_spaces=True)
