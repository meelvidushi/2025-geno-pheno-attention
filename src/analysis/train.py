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


def check_cached_model(train_config: TrainConfig) -> Path | None:
    out_dir = train_config.expected_model_dir
    if out_dir and out_dir.exists() and out_dir.is_dir():
        return train_config.get_latest_version_dir()


def run_training(model_config: ModelConfig, train_config: TrainConfig) -> Path:
    existing_path = check_cached_model(train_config)
    if isinstance(existing_path, Path) and train_config.use_cache:
        print(f"Model with {train_config.name_prefix=} already found. Returning path to directory.")
        return existing_path

    if train_config.use_modal:
        local_path = train_model_with_modal(model_config, train_config, blocking=True)
    else:
        local_path = train_model(model_config, train_config)

    assert isinstance(local_path, Path)
    return local_path


def run_trainings(
    jobs: Iterable[tuple[ModelConfig, TrainConfig]],
) -> list[Path]:
    # Determine whether to use modal and ensure usage is constant across runs.
    jobs_list = list(jobs)
    use_modal = jobs_list[0][1].use_modal
    for _, train_config in jobs_list:
        assert train_config.use_modal == use_modal

    if use_modal:
        paths = []
        filtered_jobs = []

        for model_config, train_config in jobs_list:
            existing_path = check_cached_model(train_config)
            if isinstance(existing_path, Path) and train_config.use_cache:
                print(
                    f"Model with {train_config.name_prefix=} already found. "
                    f"Returning path to directory."
                )
                paths.append(existing_path)
            else:
                filtered_jobs.append((model_config, train_config))
                paths.append(train_config.expected_model_dir)

        if filtered_jobs:
            train_models_with_modal(filtered_jobs)

        return paths
    else:
        return [
            run_training(model_config, train_config) for model_config, train_config in jobs_list
        ]


if __name__ == "__main__":
    # CLI is only for single training
    config = tyro.cli(CLIArgs)
    run_training(config.model, config.train)
