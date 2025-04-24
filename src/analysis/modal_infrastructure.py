import os
import subprocess
from pathlib import Path

import attrs
import modal
import torch

from analysis.base import ModelConfig, TrainConfig
from analysis.dataset import (
    GENO_TEST_PATHNAME,
    GENO_TRAIN_PATHNAME,
    GENO_VAL_PATHNAME,
    PHENO_TEST_PATHNAME,
    PHENO_TRAIN_PATHNAME,
    PHENO_VAL_PATHNAME,
)

APP_NAME = "geno-pheno-attention-training"

# These are deployed functions of the app.
PREP_REMOTE_DIR_FUNCTION = modal.Function.from_name(
    APP_NAME,
    "prepare_remote_dirs_and_check_files",
)
TRAIN_FUNCTION = modal.Function.from_name(
    APP_NAME,
    "run_training",
)

MOUNT = Path("/data")
GPU_NAME = os.environ.get("MODAL_GPU_NAME", "A10G").upper()
MEMORY = int(os.environ.get("MODAL_RAM", 4092))
GPU_SPEC = f"{GPU_NAME}:1"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "plotly==6.0.1",
        "arcadia-pycolor==0.6.0",
        "matplotlib==3.9.2",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "torch==2.2.2",
        "h5py==3.13.0",
        "scikit-learn==1.6.1",
        "typer==0.15.2",
        "lightning==2.5.1",
        "tensorboard==2.19.0",
        "tyro==0.9.19",
        "pyyaml==6.0.1",
        "attrs==25.3.0",
    )
    .add_local_python_source(
        "analysis",
    )
)

with image.imports():
    from analysis.train_ops import train_model


volume = modal.Volume.from_name("geno-pheno-attention-data", create_if_missing=True)


@app.function(image=image, volumes={str(MOUNT): volume})
def prepare_remote_dirs_and_check_files(dirs_to_create: list[Path]):
    """Create directories and check existing files in the volume.

    Args:
        dirs_to_create: List of directory paths to create

    Returns:
        set[Path]: Set of existing file paths in the volume
    """
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    existing_files = set()
    for root, _, files in os.walk(MOUNT):
        root_path = Path(root)
        for file in files:
            existing_files.add(root_path / file)

    volume.commit()

    print(f"Found {len(existing_files)} existing files in volume")

    return existing_files


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=3600 * 24,  # Max timeout is 1 day
    volumes={str(MOUNT): volume},
    memory=MEMORY,
    cpu=1,
)
def run_training(model_config: ModelConfig, train_config: TrainConfig):
    # Add other Tensor core compatible GPUs here.
    if GPU_NAME in {"A10G"}:
        # Set matmul precision to medium for better performance with Tensor Cores
        torch.set_float32_matmul_precision("medium")

    run_log_dir = train_model(model_config, train_config)

    volume.commit()

    return run_log_dir


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    memory=1024,
)
@modal.web_server(6006)
def tensorboard_server(logdir: str = "./"):
    full_logdir = str(MOUNT / logdir)
    subprocess.Popen(f"tensorboard --logdir={full_logdir} --bind_all --port 6006", shell=True)


def setup_remote_directory(config: TrainConfig):
    local_paths: list[Path] = [
        config.data_dir / name
        for name in [
            GENO_TEST_PATHNAME,
            GENO_VAL_PATHNAME,
            GENO_TRAIN_PATHNAME,
            PHENO_TEST_PATHNAME,
            PHENO_VAL_PATHNAME,
            PHENO_TRAIN_PATHNAME,
        ]
    ]

    for local_path in local_paths:
        if not local_path.exists():
            raise FileNotFoundError(f"Local path '{local_path}' file not found.")

    dirs_to_create = set()
    dirs_to_create.add(MOUNT / config.save_dir)

    existing_files = PREP_REMOTE_DIR_FUNCTION.remote(list(dirs_to_create))

    files_to_upload: dict[Path, Path] = {}
    for local_path in local_paths:
        remote_path = MOUNT / local_path
        if remote_path not in existing_files:
            files_to_upload[local_path] = remote_path

    if not files_to_upload:
        print("All required files exist remotely.")
        return

    with volume.batch_upload() as batch:
        for local_path, remote_path in files_to_upload.items():
            # Within the batch_upload context manager, MOUNT is considered the working
            # directory, so we upload relative to the MOUNT
            batch.put_file(local_path, remote_path.relative_to(MOUNT))

    print("Files uploaded successfully")


def download_model(run_log_dir: Path):
    print(f"Training completed. Run artifacts saved in Modal volume at: {run_log_dir}")

    # Maintain the same directory structure locally as on the remote
    # Get the path relative to MOUNT (/data)
    relative_path = run_log_dir.relative_to(MOUNT)

    # Create local base directory that matches the remote structure
    local_base_dir = Path(".")
    local_full_path = local_base_dir / relative_path
    local_full_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading run directory to {local_full_path}...")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "geno-pheno-attention-data",
            str(relative_path),
            str(local_full_path.parent),
        ],
        check=True,
    )

    print(f"Model artifacts downloaded successfully to {local_full_path}")


def train_model_with_modal(model_config: ModelConfig, train_config: TrainConfig):
    print("Starting training job...")
    print(f"{model_config=}")
    print(f"{train_config=}")

    setup_remote_directory(train_config)

    remote_train_config = attrs.evolve(
        train_config,
        data_dir=MOUNT / train_config.data_dir,
        save_dir=MOUNT / train_config.save_dir,
    )
    run_log_dir = TRAIN_FUNCTION.remote(model_config, remote_train_config)

    print("Finished training.")

    download_model(run_log_dir)


if __name__ == "__main__":
    # Use train.py with --use-modal
    pass
