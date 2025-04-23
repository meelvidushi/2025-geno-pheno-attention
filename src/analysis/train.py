from datetime import datetime
from pathlib import Path

import attrs
import lightning as L
import pandas as pd
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from analysis.dataset import create_dataloaders, phenotype_names
from analysis.rijal_et_al import RijalEtAl
from analysis.transformer import GenoPhenoTransformer

app = typer.Typer(pretty_exceptions_enable=False)


@attrs.define
class RunParams:
    data_dir: Path = Path("./data")
    save_dir: Path = Path("./checkpoints")
    phenotype: str = "23C"

    model_type: str = "rijal_et_al"

    # Shared
    embedding_dim: int = 13
    num_layers: int = 3

    # Rijal et al only
    skip_connections: bool = False

    # Transformer only
    nhead: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    name_prefix: str = ""
    patience: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 200
    num_workers: int = 4
    gradient_clip_val: float = 0.0  # Default 0.0 means no clipping
    use_modal: bool = False
    modal_detach: bool = True


def save_metrics(trainer, model, val_dataloader, test_dataloader, checkpoint_callback):
    """Save model metrics to a CSV file."""
    # First, get metrics for the final model state
    final_val_results = trainer.validate(model, val_dataloader)[0]
    final_val_loss = final_val_results["val_loss"]
    final_val_r2 = final_val_results["val_r2"]

    # Extract best validation loss from the checkpoint callback
    best_val_loss = checkpoint_callback.best_model_score.item()

    # Load the best model from checkpoint using the same model class
    best_model_path = checkpoint_callback.best_model_path
    model_class = model.__class__
    best_model = model_class.load_from_checkpoint(best_model_path)
    best_model.eval()

    # Evaluate the best model on validation and test sets
    best_val_results = trainer.validate(best_model, val_dataloader)[0]
    best_val_r2 = best_val_results["val_r2"]

    test_results = trainer.test(best_model, dataloaders=test_dataloader)[0]
    test_loss = test_results["test_loss"]
    test_r2 = test_results["test_r2"]

    # Create DataFrame with metrics
    metrics_df = pd.DataFrame(
        {
            "metric": [
                "best_val_loss",
                "final_val_loss",
                "best_val_r2",
                "final_val_r2",
                "test_loss",
                "test_r2",
                "checkpoint_path",
            ],
            "value": [
                best_val_loss,
                final_val_loss,
                best_val_r2,
                final_val_r2,
                test_loss,
                test_r2,
                best_model_path,
            ],
        }
    )

    # Save to CSV
    metrics_path = Path(trainer.log_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Metrics saved to: {metrics_path}")

    return metrics_path


def _train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: RunParams,
):
    seq_length = next(iter(train_dataloader))[0].size(1)

    if config.model_type == "rijal_et_al":
        model = RijalEtAl(
            embedding_dim=config.embedding_dim,
            seq_length=seq_length,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            num_layers=config.num_layers,
            skip_connections=config.skip_connections,
        )
    elif config.model_type == "transformer":
        model = GenoPhenoTransformer(
            embedding_dim=config.embedding_dim,
            seq_length=seq_length,
            num_layers=config.num_layers,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
    else:
        raise NotImplementedError()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        verbose=True,
    )

    # Create a versioned subdirectory with name prefix and/or timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"{config.name_prefix}" if config.name_prefix else timestamp
    experiment_dir = config.save_dir / version_name

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=experiment_dir,
        enable_checkpointing=True,
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback],
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Save metrics to CSV using the best checkpoint
    save_metrics(trainer, model, val_dataloader, test_dataloader, checkpoint_callback)

    return model, trainer.log_dir


def train_model(config: RunParams):
    if config.phenotype not in phenotype_names:
        raise ValueError(f"Invalid phenotype name. Valid options are: {phenotype_names}")

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
