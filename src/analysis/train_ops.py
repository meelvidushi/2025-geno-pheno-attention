import random
from datetime import datetime
from pathlib import Path

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from analysis.base import ModelConfig, TrainConfig
from analysis.dataset import create_dataloaders
from analysis.modified_rijal_et_al import ModifiedRijalEtAl
from analysis.rijal_et_al import RijalEtAl
from analysis.transformer import Transformer

model_str_to_cls: dict[str, type[L.LightningModule]] = {
    "rijal_et_al": RijalEtAl,
    "modified": ModifiedRijalEtAl,
    "transformer": Transformer,
}


def train_model(model_config: ModelConfig, train_config: TrainConfig) -> Path:
    # Use the provided seed or generate a random one if None
    if train_config.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = train_config.seed

    L.seed_everything(seed)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        data_dir=train_config.data_dir,
        phenotypes=train_config.phenotypes,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        synthetic=train_config.synthetic_data,
    )

    model_type = model_config.model_type
    if model_type not in model_str_to_cls:
        raise NotImplementedError(
            f"{model_type} not an understood model type. Available: {list(model_str_to_cls.keys())}"
        )

    model_cls = model_str_to_cls[model_type]
    model = model_cls(model_config, train_config)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=train_config.patience,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="max",
        filename="best-{epoch:03d}-{val_loss:.4f}",
        verbose=True,
    )

    # Create a versioned subdirectory with name prefix and/or timestamp
    train_config.save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"{train_config.name_prefix}" if train_config.name_prefix else timestamp
    experiment_dir = train_config.save_dir / version_name

    trainer = L.Trainer(
        max_epochs=train_config.max_epochs,
        default_root_dir=experiment_dir,
        enable_checkpointing=True,
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback],
        gradient_clip_val=train_config.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Save metrics to CSV using the best checkpoint
    _save_metrics(trainer, model, val_dataloader, test_dataloader, checkpoint_callback)

    assert trainer.log_dir is not None
    return Path(trainer.log_dir)


def _save_metrics(
    trainer: L.Trainer,
    model: L.LightningModule,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    checkpoint_callback: ModelCheckpoint,
):
    """Save model metrics to a CSV file."""
    # First, get metrics for the final model state
    final_val_results = trainer.validate(model, val_dataloader)[0]
    final_val_loss = final_val_results["val_loss"]
    final_val_r2 = final_val_results["val_r2"]

    # Load the best model from checkpoint using the same model class
    best_model_path = checkpoint_callback.best_model_path
    model_class = model.__class__
    best_model = model_class.load_from_checkpoint(best_model_path)
    best_model.eval()

    best_val_results = trainer.validate(best_model, val_dataloader)[0]
    best_val_loss = best_val_results["val_loss"]
    best_val_r2 = best_val_results["val_r2"]

    test_results = trainer.test(best_model, dataloaders=test_dataloader)[0]
    test_loss = test_results["test_loss"]
    test_r2 = test_results["test_r2"]

    # Initialize the metrics dictionary with aggregate metrics
    metrics_dict = {
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "best_val_r2": best_val_r2,
        "final_val_r2": final_val_r2,
        "test_loss": test_loss,
        "test_r2": test_r2,
        "checkpoint_path": best_model_path,
    }

    # Extract per-phenotype R2 values
    for key, value in best_val_results.items():
        if key.startswith("val_r2_") and key != "val_r2":
            phenotype = key.replace("val_r2_", "")
            metrics_dict[f"best_val_r2_{phenotype}"] = value

    for key, value in test_results.items():
        if key.startswith("test_r2_") and key != "test_r2":
            phenotype = key.replace("test_r2_", "")
            metrics_dict[f"test_r2_{phenotype}"] = value

    metrics_df = pd.DataFrame(metrics_dict.items(), columns=pd.Index(["metric", "value"]))

    assert trainer.log_dir is not None
    metrics_path = Path(trainer.log_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_path


if __name__ == "__main__":
    pass
