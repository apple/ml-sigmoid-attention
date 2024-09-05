#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Example script to train an auto-regressive model."""

import functools
import logging
import typing as t

import hydra
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from attention_simulator.autoregressive_language_model import evaluate
from attention_simulator.autoregressive_language_model import generate
from attention_simulator.autoregressive_language_model import train
from attention_simulator.helpers.params import get_param_buffer_info
from optorch.builder import OPTIMIZER_STEP
from optorch.builder import init_optimizer_state

LOGGER = logging.getLogger(__name__)


def build_tokenizer(cfg: DictConfig) -> t.Any:
    """Build a tokenizer from the config."""
    tokenizer = hydra.utils.instantiate(cfg)

    # force this s.t. rotations by one make BOS of inputs becomes EOS of target
    tokenizer.bos_token = tokenizer.eos_token
    if not tokenizer.pad_token:  # add a pad token (otherwise we get an error).
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_mixed_precision_scaler(cfg: DictConfig, device: torch.device) -> t.Union[t.Dict[str, t.Any], None]:
    """Build an optional mixed precision scaler."""
    scaler_state = None
    if "scaler" in cfg:
        scaler_state = hydra.utils.instantiate(cfg.scaler, device=device)

    precision = scaler_state["precision"] if scaler_state else torch.float32
    LOGGER.info(
        f"[Precision]: device = {device.type} | " f"dtype = {precision} | " f"enabled = {scaler_state is not None}"
    )
    return scaler_state


def build_dataset(cfg: DictConfig, tokenizer: t.Any) -> DataLoader:
    """Load a dataset (like Tiny-Shakespeare)."""
    dataset_mapper = {
        "realnewslike": functools.partial(load_dataset, "allenai/c4", "realnewslike"),
        "tiny_shakespeare": functools.partial(load_dataset, "Trelis/tiny-shakespeare"),
        "wikitext": functools.partial(load_dataset, "wikitext", "wikitext-2-raw-v1"),
    }
    dataset = dataset_mapper[cfg.name]()[cfg.split]
    LOGGER.info(f"{cfg.name} dataset [split={cfg.split}] has {len(dataset)} samples.")

    class HuggingFaceDataset(Dataset):
        """Generic wrapper around HF dataset."""

        def __init__(self, dataset, tokenizer, max_length):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            key = cfg.key
            text = self.dataset[idx][key]
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze()
            attention_mask = encoded["attention_mask"].squeeze()

            # Use -100 for ignore_index in F.cross_entropy
            targets = torch.zeros_like(input_ids) - 100
            targets[:-1] = input_ids[1:].clone()

            return {
                "input_ids": input_ids,
                "targets": targets,
                "attention_mask": attention_mask,
            }

    return DataLoader(
        HuggingFaceDataset(dataset=dataset, tokenizer=tokenizer, max_length=cfg.max_seq_len),
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        pin_memory=cfg.pin_memory,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
    )


def build_transformer(cfg: DictConfig) -> nn.Module:
    # One line init. See yaml for all modules.
    # Everything is defined as a partial till initized.
    transformer = hydra.utils.instantiate(cfg.model)
    LOGGER.info(transformer)

    # Move to faster compute if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    transformer = transformer.to(device)
    if device == torch.device("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return transformer


def build_functional_optimizer(model: nn.Module, optimizer_cfg: DictConfig) -> t.Dict[str, t.Any]:
    LOGGER.info("Building functional optimizer...")

    # Instantiate the optimizer configuration
    opt_config = hydra.utils.instantiate(optimizer_cfg)
    opt_name = str(optimizer_cfg._target_.split(".")[-1]).replace("Config", "").lower()
    if opt_name == "adam":  # check for {adamw, adam_decoupled}
        opt_name = optimizer_cfg.mode.lower()
    LOGGER.info(f"Optimizer {opt_name} Config: {opt_config}")

    # Get the optimizer step function based on the optimizer name
    optimizer_step = OPTIMIZER_STEP[opt_name]

    # Initialize the optimizer state
    opt_state = init_optimizer_state(opt_name, model)

    return {
        "opt_state": opt_state,
        "optimizer_step": optimizer_step,
        "optimizer_config": opt_config,
    }


def gimme_my_metrics_mapper() -> t.Dict[str, t.Callable]:
    """Metrics reduction dictionary, add more here."""

    @functools.lru_cache(maxsize=100)
    def _norm_tensor_or_weights(x: torch.Tensor, cache_key: t.Any) -> torch.Tensor:
        """Norm a tensor on dim=-1 if we are a data tensor, else on all dims."""
        if x.dim() >= 3:
            out = x.flatten(1).norm(p=2, dim=-1)  # [B, F0, F1, F2] --> norm([B, F0*F1*F2], dim=-1)
            return out  # [B] dimensional

        # Assume weights or biases.
        return x.norm(p=2)

    @functools.lru_cache(maxsize=100)
    def _monte_carlo_entropy(x: torch.Tensor, cache_key: t.Any) -> torch.Tensor:
        if x.dim() >= 3:
            x = x.flatten(1)  # [B, F0, F1, F2] --> [B, F0*F1*F2]
            x = -x * torch.log(x + 1e-6)
            return x.sum(dim=-1)  # [B] dimensional

        # Assume weights or biases.
        return (-x * torch.log(x + 1e-6)).sum()

    # What stats do we want to gather?
    reduction_map = {
        "mean_pnorm": lambda x, cache_key: _norm_tensor_or_weights(x, cache_key).mean().item(),
        "max_pnorm": lambda x, cache_key: _norm_tensor_or_weights(x, cache_key).max().item(),
        "min_pnorm": lambda x, cache_key: _norm_tensor_or_weights(x, cache_key).min().item(),
        "mean_softmax_entropy": lambda x, cache_key: _monte_carlo_entropy(x, cache_key).mean().item(),
        "max_softmax_entropy": lambda x, cache_key: _monte_carlo_entropy(x, cache_key).max().item(),
        "min_softmax_entropy": lambda x, cache_key: _monte_carlo_entropy(x, cache_key).min().item(),
    }
    return reduction_map


def plot(
    history_df: pd.DataFrame,
    plot_params: t.Dict[str, t.Any],
    output_name: str,
    append: bool = False,
):
    """Plot the results."""
    LOGGER.info(history_df)

    valid_params = [params for params in plot_params.values() if isinstance(params, dict)]
    nrows = max(params["row_idx"] for params in valid_params) + 1
    ncols = max(params["col_idx"] for params in valid_params) + 1
    width = plot_params.get("width", 15)
    height = plot_params.get("height", 12.5)
    fontsize = plot_params.get("fontsize", 15)
    wspace = plot_params.get("wspace", 0.05)
    hspace = plot_params.get("hspace", 0.125)
    legend = plot_params.get("legend", False)
    legend_fontsize = plot_params.get("legend_fontsize", 12)
    legend_loc = plot_params.get("legend_loc", "best")

    if not append or not plt.get_fignums():
        if nrows == 1 or ncols == 1:
            fig, axs = plt.subplots(max(nrows, ncols), figsize=(width, height), sharex="all")
            axs = np.array([axs]) if not isinstance(axs, np.ndarray) else axs
        else:
            fig, axs = plt.subplots(nrows, ncols, figsize=(width, height), sharex="all", sharey="row")
    else:
        fig = plt.gcf()
        axs = fig.axes
        axs = np.array(axs).reshape(nrows, ncols)

    axs: plt.Axes

    # Create formatters for regular and log scales
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific("%1.2e")
    formatter.set_useOffset(False)

    log_formatter = mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False, minor_thresholds=(2, 0.4))

    for field, params in plot_params.items():
        if not isinstance(params, dict) or field.startswith("col"):
            continue

        row_idx = params["row_idx"]
        col_idx = params["col_idx"]
        linewidth = params.get("linewidth", 2)
        color = params.get("color", None)
        label = params.get("label", None)
        yscale = params.get("yscale", "linear")

        if nrows == 1 or ncols == 1:
            ax = axs[max(row_idx, col_idx)]
        else:
            ax = axs[row_idx, col_idx]

        ax.plot(history_df[field], linewidth=linewidth, color=color, label=label)

        if "ylabel" in params:
            ax.set_ylabel(params["ylabel"], fontsize=fontsize)

        if yscale == "log":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(log_formatter)
        else:
            ax.yaxis.set_major_formatter(formatter)

    for col_idx in range(ncols):
        if "title" in plot_params.get(f"col{col_idx}", {}):
            if nrows == 1:
                axs[col_idx].set_title(plot_params[f"col{col_idx}"]["title"], fontsize=fontsize)
            else:
                axs[0, col_idx].set_title(plot_params[f"col{col_idx}"]["title"], fontsize=fontsize)

    for idx in range(max(nrows, ncols)):
        if idx == max(nrows, ncols) - 1:
            if nrows == 1:
                axs[idx].set_xlabel("Steps", fontsize=fontsize)
            elif ncols == 1:
                axs[idx].set_xlabel("Steps", fontsize=fontsize)
            else:
                axs[nrows - 1, ncols // 2].set_xlabel("Steps", fontsize=fontsize)

    for ax in axs.flat:
        ax.grid(visible=True, which="major")
        if legend:
            ax.legend(fontsize=legend_fontsize, loc=legend_loc)

    # Remove spacing between plots with shared axis.
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.tight_layout()

    if not append:
        # Save the figure to PDF and PNG
        fig.savefig(f"{output_name}.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(f"{output_name}.png", format="png", bbox_inches="tight")


def loop(graph_fn: t.Callable, uid: str):
    """Run graph_fn, make some plots.

    :param graph_fn: Function to run.
    :param uid: Unique identifier for the run.

    """
    LOGGER.info(f"Starting {uid}...")
    history_df = graph_fn()  # Enjoy a glorious dataframe of stats.
    history_df.to_csv(f"{uid}_results.csv", index=False)

    # Add LR during training loop.
    lr_dict = {}
    if "lr" in history_df:
        lr_dict = {
            "lr": {
                "row_idx": 5,
                "col_idx": 0,
                "linewidth": 2,
                "color": "blue",
                "yscale": "linear",
                # "label": ...
                "ylabel": "Learning Rate",
            }
        }

    # Add grads of attn during training.
    grad_attn_times_v_dict = {}
    if "intermediary_grad_mean_pnorm_block0_attn_times_v" in history_df:
        grad_attn_times_v_dict = {
            "intermediary_grad_mean_pnorm_block0_attn_times_v": {
                "row_idx": 4,
                "col_idx": 0,
                "linewidth": 2,
                "color": "blue",
                "yscale": "log",
                "ylabel": r"$\left\Vert \frac{dL}{d\ act(\frac{QK^T}{\sqrt{d}}) V} \right\Vert$",
            }
        }

    # Add model W_qkv grads during training.
    model_grad_qkv_dict = {}
    if "model_grad_mean_pnorm_blocks.0.attn.qkv.weight" in history_df:
        model_grad_qkv_dict = {
            "model_grad_mean_pnorm_blocks.0.attn.qkv.weight": {
                "row_idx": 3,
                "col_idx": 0,
                "linewidth": 2,
                "color": "blue",
                "yscale": "log",
                "ylabel": r"$\left\Vert \frac{dL}{d W_{QKV}} \right\Vert$",
            }
        }

    # Add attn * v if it exists.
    output_max_attn_times_v_dict = {}
    if "output_max_pnorm_block0_attn_times_v" in history_df:
        output_max_attn_times_v_dict = {
            "output_max_pnorm_block0_attn_times_v": {
                "row_idx": 1,
                "col_idx": 0,
                "linewidth": 2,
                "color": "blue",
                "yscale": "log",
                "ylabel": r"$\left\Vert act(\frac{QK^T}{\sqrt{d}}) V \right\Vert$",
            }
        }

    # Add norm(model qkv weight) if it exists.
    model_param_qkv_dict = {}
    if "model_param_mean_pnorm_blocks.0.attn.qkv.weight" in history_df:
        model_param_qkv_dict = {
            "model_param_mean_pnorm_blocks.0.attn.qkv.weight": {
                "row_idx": 2,
                "col_idx": 0,
                "linewidth": 2,
                "color": "blue",
                "yscale": "linear",
                "ylabel": r"$\left\Vert W_{QKV} \right\Vert$",
            }
        }

    # Plot a few keys from history_df.
    plot(
        history_df=history_df,
        plot_params={
            "loss": {
                "row_idx": 0,
                "col_idx": 0,
                "linewidth": 2,
                "color": "blue",
                "yscale": "linear",
                # "label": ...
                "ylabel": "Train NLL",
            },
            **output_max_attn_times_v_dict,
            **model_param_qkv_dict,
            **model_grad_qkv_dict,
            **grad_attn_times_v_dict,
            **lr_dict,
            # Global plot details.
            "width": 10,
            "height": 15,
            "fontsize": 15,
            "legend": False,
            "legend_fontsize": 10,
            "legend_loc": "upper right",
        },
        output_name=f"{uid}_attn_sim",
        append=False,  # If True call plot() again to tack on to fig.
    )


def dataset_cfg_to_uid(cfg: DictConfig, seed: int) -> str:
    """Create a unique string based on a dataset config."""
    return f"{cfg.name}_" f"{cfg.split}_" f"bs={cfg.batch_size}_" f"seq={cfg.max_seq_len}_" f"seed={seed}"


@hydra.main(config_path="configs", config_name="experiment_tiny_shakespeare_transformer_sigmoid_rope")
def main(cfg: DictConfig) -> None:
    LOGGER.info(OmegaConf.to_yaml(cfg))

    # Seed all the things.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Build tokenizer.
    tokenizer = build_tokenizer(cfg.tokenizer)

    # Build the model (includes attn(s), mlp(s), embedding, etc).
    transformer = build_transformer(cfg)
    param_buffer_info = get_param_buffer_info(transformer)
    LOGGER.info(param_buffer_info)

    # Setup optimizer config and mapping function.
    optimizer_dict = build_functional_optimizer(transformer, cfg.optimizer)

    # Build an LR scheduler
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler)

    # What metrics do we want?
    reduction_map = gimme_my_metrics_mapper()

    # Build the optional grapher.
    grapher = None
    if "grapher" in cfg:
        grapher = hydra.utils.instantiate(cfg.grapher, metadata=OmegaConf.to_container(cfg))
        if hasattr(grapher, "config"):
            grapher_cfg = OmegaConf.to_container(cfg)
            grapher_cfg.update(param_buffer_info)
            grapher.config.update(grapher_cfg)

    # Iterate over all the train datasets one at a time.
    if "train_datasets" in cfg and cfg.train_datasets is not None:
        for dataset in cfg.train_datasets:
            train_dataset = build_dataset(dataset, tokenizer)
            uid = dataset_cfg_to_uid(dataset, cfg.seed)

            # Dynamically build the mixed precision scaler.
            scaler_state = build_mixed_precision_scaler(cfg.training, device)

            trainer_fn = functools.partial(
                train,
                model=transformer,
                dataset=train_dataset,
                max_steps=cfg.training.max_steps,
                device=device,
                grapher=grapher,
                reduction_map=reduction_map,
                lr_scheduler=lr_scheduler,
                scaler_state=scaler_state,
                prefix=uid,
                **optimizer_dict,  # Contains state, step and conf.
            )
            loop(graph_fn=trainer_fn, uid=uid)

    # Iterate over all the generation queries one at a time.
    if "generations" in cfg and cfg.generations is not None:
        for query in cfg.generations.queries:
            # Dynamically build the mixed precision scaler.
            scaler_state = build_mixed_precision_scaler(cfg.training, device)

            generate_fn = functools.partial(
                generate,
                model=transformer,
                tokenizer=tokenizer,
                input_text=query.input_text,
                max_length=query.max_length,
                top_p=query.top_p,
                temperature=query.temperature,
                scaler_state=scaler_state,
                device=device,
            )

            # Generate the requested text.
            LOGGER.info(f"[Generation]: {query}")
            LOGGER.info(generate_fn())

    # Iterate over all the eval datasets one at a time.
    if "test_datasets" in cfg and cfg.test_datasets is not None:
        for dataset in cfg.test_datasets:
            test_dataset = build_dataset(dataset, tokenizer)
            uid = dataset_cfg_to_uid(dataset, cfg.seed)

            # Dynamically build the mixed precision scaler.
            scaler_state = build_mixed_precision_scaler(
                cfg.evaluation if "evaluation" in cfg else {},
                device,
            )

            eval_fn = functools.partial(
                evaluate,
                model=transformer,
                dataset=test_dataset,
                device=device,
                grapher=grapher,
                reduction_map=reduction_map,
                scaler_state=scaler_state,
                prefix=uid,
            )
            loop(graph_fn=eval_fn, uid=uid)


if __name__ == "__main__":
    main()
