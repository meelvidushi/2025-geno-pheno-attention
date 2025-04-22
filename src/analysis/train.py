from pathlib import Path

import lightning as L
import typer
from torch.utils.data import DataLoader

from analysis.dataset import create_dataloaders, phenotype_names
from analysis.rijal_et_al import Rijal2025

app = typer.Typer(pretty_exceptions_enable=False)


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    input_dim: int,
    query_dim: int,
    key_dim: int,
    learning_rate: float = 0.001,
    max_epochs: int = 200,
    save_dir: Path = Path("./checkpoints"),
):
    seq_length = next(iter(train_dataloader))[0].size(1)

    model = Rijal2025(
        input_dim=input_dim,
        query_dim=query_dim,
        key_dim=key_dim,
        seq_length=seq_length,
        learning_rate=learning_rate,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        default_root_dir=save_dir,
        enable_checkpointing=True,
        log_every_n_steps=10,
        val_check_interval=0.25,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

    return model


@app.command()
def main(
    data_dir: Path = typer.Option(Path("./data"), help="Directory containing the data files"),
    phenotype: str = typer.Option("23C", help="Name of the phenotype to predict"),
    input_dim: int = typer.Option(13, help="Dimension of input features"),
    query_dim: int = typer.Option(13, help="Dimension of the query matrix"),
    key_dim: int = typer.Option(13, help="Dimension of the key matrix"),
    batch_size: int = typer.Option(64, help="Batch size for training"),
    learning_rate: float = typer.Option(0.001, help="Learning rate for optimization"),
    max_epochs: int = typer.Option(200, help="Maximum number of training epochs"),
    save_dir: Path = typer.Option(Path("./checkpoints"), help="Directory to save checkpoints"),
    num_workers: int = typer.Option(4, help="Number of workers for data loading"),
):
    if phenotype not in phenotype_names:
        valid_names = ", ".join(phenotype_names)
        raise typer.BadParameter(f"Invalid phenotype name. Valid options are: {valid_names}")

    L.seed_everything(42)

    save_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = create_dataloaders(
        data_dir=data_dir,
        phenotype_name=phenotype,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = train_model(
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        test_dataloader=dataloaders["test"],
        input_dim=input_dim,
        query_dim=query_dim,
        key_dim=key_dim,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        save_dir=save_dir,
    )

    print(f"Model training completed. Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    app()
