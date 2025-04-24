import attrs
import tyro

from analysis.base import ModelConfig, TrainConfig
from analysis.modal_infrastructure import (
    train_model_with_modal,
)
from analysis.train_ops import train_model


@attrs.define
class CLIArgs:
    model: ModelConfig
    train: TrainConfig


def run_training(model_config: ModelConfig, train_config: TrainConfig):
    if train_config.use_modal:
        train_model_with_modal(model_config, train_config)
    else:
        train_model(model_config, train_config)


if __name__ == "__main__":
    config = tyro.cli(CLIArgs)
    run_training(config.model, config.train)
