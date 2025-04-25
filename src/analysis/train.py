from collections.abc import Iterable
from pathlib import Path

import attrs
import tyro

from analysis.base import ModelConfig, TrainConfig
from analysis.modal_infrastructure import (
    train_model_with_modal,
    train_models_with_modal,
)
from analysis.train_ops import train_model


@attrs.define
class CLIArgs:
    model: ModelConfig
    train: TrainConfig


def run_training(model_config: ModelConfig, train_config: TrainConfig) -> Path:
    if train_config.use_modal:
        local_path = train_model_with_modal(model_config, train_config, blocking=True)
    else:
        local_path = train_model(model_config, train_config)

    assert isinstance(local_path, Path)
    return local_path


def run_trainings(jobs: Iterable[tuple[ModelConfig, TrainConfig]]) -> list[Path]:
    # Determine whether to use modal and ensure usage is constant across runs.
    use_modal = next(iter(jobs))[1].use_modal
    for _, train_config in jobs:
        assert train_config.use_modal == use_modal

    if use_modal:
        return train_models_with_modal(jobs)

    return [train_model(model_config, train_config) for model_config, train_config in jobs]


if __name__ == "__main__":
    # CLI is only for single training
    config = tyro.cli(CLIArgs)
    run_training(config.model, config.train)
