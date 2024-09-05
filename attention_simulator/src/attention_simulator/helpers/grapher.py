#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Grapher logic here."""

import os
import typing as t
from pathlib import Path


def init_wandb(
    api_key: str,
    base_url: str,
    project_name: str,
    log_dir: Path,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    resume: bool = False,
    offline: bool = False,
    log_every: int = 1000,
) -> t.Any:
    """Initialize WandB with for a specific server with login parameters.

    :param api_key: The API key.
    :param base_url: The base url.
    :param project_name: Name for project, to organize multiple runs for easy comparison.
    :param log_dir: Local dir WandB will use to store results and sync to server from.
    :param metadata: Dictionary to log to WandB (e.g. hyperparameters for the run).
    :param resume: If `True`, if WandB is run in the same log dir more than once with
        the same config, it will associate the same run to the current run and continue
        logging to that run (default: `False`).
    :param log_every: interval to log data.
    :return: The WandB run object.

    """
    import wandb  # lazy import as wandb is an optional dependency

    assert api_key is not None
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_BASE_URL"] = base_url

    # Create log dir.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if offline:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        config=metadata,
        project=project_name,
        dir=log_dir,
        resume=resume,
    )
    run.config["log_every"] = log_every  # add an extra member.
    wandb.define_metric("*", "GlobalSteps")

    return run
