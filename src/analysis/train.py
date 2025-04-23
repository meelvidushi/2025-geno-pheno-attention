from datetime import datetime
from pathlib import Path

import attrs
import lightning as L
import typer
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from analysis.dataset import create_dataloaders, phenotype_names
from analysis.rijal_et_al import RijalEtAl

app = typer.Typer(pretty_exceptions_enable=False)


@attrs.define
class RunParams:
    data_dir: Path = Path("./data")
    save_dir: Path = Path("./checkpoints")
    phenotype: str = "23C"
    embedding_dim: int = 13
    num_layers: int = 3
    skip_connections: bool = False
    name_prefix: str = ""

    patience: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 200
    num_workers: int = 4
    gradient_clip_val: float = 0.0  # Default 0.0 means no clipping

    use_modal: bool = False


def _train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: RunParams,
):
    seq_length = next(iter(train_dataloader))[0].size(1)

    model = RijalEtAl(
        embedding_dim=config.embedding_dim,
        seq_length=seq_length,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_layers=config.num_layers,
        skip_connections=config.skip_connections,
    )

    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        verbose=True,
        mode="min",
    )

    # Create a versioned subdirectory with name prefix and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"{config.name_prefix}_{timestamp}" if config.name_prefix else timestamp
    experiment_dir = config.save_dir / version_name

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=experiment_dir,
        enable_checkpointing=True,
        log_every_n_steps=10,
        callbacks=[early_stopping],
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

    return model, trainer.log_dir


def train_model(config: RunParams):
    if config.phenotype not in phenotype_names:
        valid_names = ", ".join(phenotype_names)
        raise typer.BadParameter(f"Invalid phenotype name. Valid options are: {valid_names}")

    L.seed_everything(42)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = create_dataloaders(
        data_dir=config.data_dir,
        phenotype_name=config.phenotype,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model, log_dir = _train_model(
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        test_dataloader=dataloaders["test"],
        config=config,
    )

    print(f"Model training completed. Checkpoints saved to {config.save_dir}")

    return log_dir


if __name__ == "__main__":
    pass
