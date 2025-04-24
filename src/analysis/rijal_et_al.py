from pathlib import Path

import attrs
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from analysis.dataset import create_dataloaders


@attrs.define
class RijalEtAlConfig:
    data_dir: Path = Path("./data")
    save_dir: Path = Path("./models")
    name_prefix: str = ""

    # Model parameters
    phenotype: str = "23C"
    embedding_dim: int = 13
    num_layers: int = 3
    skip_connections: bool = False

    # Training parameters
    patience: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 200
    num_workers: int = 4

    # Modal parameters
    use_modal: bool = False
    modal_detach: bool = True


class StackedAttention(nn.Module):
    def __init__(self, embedding_dim, seq_length, num_layers=3, skip_connections=False):
        """
        Implements a multi-layer attention mechanism.

        Args:
            embedding_dim (int): Dimension of input features.
            seq_length (int): Length of the sequence.
            num_layers (int): Number of attention layers.
            skip_connections (bool): Whether to use skip connections between attention layers.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query_dim = self.embedding_dim
        self.key_dim = self.embedding_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.skip_connections = skip_connections

        # Create learnable matrices for each layer
        self.query_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_dim, self.query_dim)) for _ in range(num_layers)]
        )
        self.key_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_dim, self.key_dim)) for _ in range(num_layers)]
        )
        self.value_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_dim, embedding_dim)) for _ in range(num_layers)]
        )

        # Learnable random projection matrix (reduces input dimensionality)
        self.random_matrix = nn.Parameter(torch.empty(self.seq_length, self.embedding_dim - 1))

        # Learnable coefficients for attended values
        self.coeffs_attended = nn.Parameter(torch.empty(seq_length, embedding_dim))

        # Learnable scalar offset for output adjustment
        self.offset = nn.Parameter(torch.randn(1))

        # Initialize parameters
        self.init_parameters()

    def init_parameters(self):
        init_scale = 0.03  # Small scale for initialization to prevent exploding gradients

        params = [
            *self.query_matrices,
            *self.key_matrices,
            *self.value_matrices,
            self.random_matrix,
            self.coeffs_attended,
            self.offset,
        ]

        for param in params:
            init.normal_(param, std=init_scale)

    def forward(self, x):
        # Apply a random projection and concatenate it with the last feature, which
        # consists entirely of ones
        attended_values = torch.cat(
            (torch.matmul(x[:, :, : self.seq_length], self.random_matrix), x[:, :, -1:]), dim=2
        )

        # Process through each attention layer
        for i in range(self.num_layers):
            query = torch.matmul(attended_values, self.query_matrices[i])
            key = torch.matmul(attended_values, self.key_matrices[i])
            value = torch.matmul(attended_values, self.value_matrices[i])
            scores = torch.matmul(query, key.transpose(1, 2))
            scores = torch.softmax(scores, dim=-1)  # Softmax for attention weighting

            # Apply attention and add skip connection if enabled
            attention_output = torch.matmul(scores, value)
            if self.skip_connections and i > 0:
                attended_values = attention_output + attended_values
            else:
                attended_values = attention_output

        # Compute final weighted sum using learned coefficients
        final_output = torch.einsum("bij,ij->b", attended_values, self.coeffs_attended)

        # Add offset term to adjust output scale
        output = final_output + self.offset

        return output


class RijalEtAl(L.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        seq_length: int,
        num_layers: int = 3,
        skip_connections: bool = False,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.skip_connections = skip_connections

        self.learning_rate = learning_rate

        self.model = StackedAttention(
            embedding_dim=embedding_dim,
            seq_length=seq_length,
            num_layers=num_layers,
            skip_connections=skip_connections,
        )

        self.loss_fn = nn.MSELoss()

        self.val_r2  = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

    def _prepare_batch(self, batch) -> torch.Tensor:
        genotypes, _ = batch
        batch_size = genotypes.size(0)

        # Create one-hot vector embedding for genotype data
        one_hot_input = torch.zeros(
            (batch_size, self.seq_length, self.seq_length), device=self.device
        )

        # Set the diagonal elements to the genotype values
        indices = torch.arange(self.seq_length, device=self.device)
        one_hot_input[:, indices, indices] = genotypes

        # Add a feature of ones (bias term)
        ones = torch.ones((batch_size, self.seq_length, 1), device=self.device)
        one_hot_input = torch.cat((one_hot_input, ones), dim=2)

        return one_hot_input

    def _process_batch(self, batch, phase):
        genotypes, phenotypes = batch

        # Remove NaN values if any
        nan_mask = torch.isnan(phenotypes)
        if nan_mask.any():
            valid_indices = ~nan_mask
            genotypes = genotypes[valid_indices]
            phenotypes = phenotypes[valid_indices]

            # Skip this batch if all values are NaN
            if genotypes.size(0) == 0:
                return None

        inputs = self._prepare_batch((genotypes, phenotypes))

        # Use torch.no_grad for validation and test phases
        if phase != "train":
            with torch.no_grad():
                outputs = self.forward(inputs)
        else:
            outputs = self.forward(inputs)

        # Calculate and log metrics
        loss = self.loss_fn(outputs, phenotypes)
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=(phase=="train"))

        if phase == "val":
            self.val_r2.update(outputs, phenotypes)
        elif phase == "test":
            self.test_r2.update(outputs, phenotypes)

        return loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        return self._process_batch(batch, "train")

    def validation_step(self, batch, _):
        self._process_batch(batch, "val")

    def test_step(self, batch, _):
        self._process_batch(batch, "test")

    def on_validation_epoch_end(self):
        r2 = self.val_r2.compute()
        self.log("val_r2", r2, prog_bar=True)
        self.val_r2.reset()

    def on_test_epoch_end(self):
        r2 = self.test_r2.compute()
        self.log("test_r2", r2)
        self.test_r2.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def save_metrics(
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

    metrics_dict = {
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "best_val_r2": best_val_r2,
        "final_val_r2": final_val_r2,
        "test_loss": test_loss,
        "test_r2": test_r2,
        "checkpoint_path": best_model_path,
    }
    metrics_df = pd.DataFrame(metrics_dict.items(), columns=pd.Index(["metric", "value"]))

    assert trainer.log_dir is not None
    metrics_path = Path(trainer.log_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_path


def train_single_phenotype(config: RijalEtAlConfig, phenotype: str):
    # Add the phenotype to the config.
    config = attrs.evolve(config, phenotype=phenotype)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(42)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        data_dir=config.data_dir,
        phenotype_name=config.phenotype,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # This is the number of loci--we need it to initialize the model.
    seq_length = next(iter(train_dataloader))[0].size(1)

    model = RijalEtAl(
        embedding_dim=config.embedding_dim,
        seq_length=seq_length,
        learning_rate=config.learning_rate,
        num_layers=config.num_layers,
        skip_connections=config.skip_connections,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        verbose=True,
        mode="min",
    )

    # Track and store the best model.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_r2",
        mode="max",
        filename="best-{epoch:02d}-{val_r2:.4f}",
        verbose=True,
    )

    # Create a versioned subdirectory with name prefix + the phenotype.
    experiment_dir = config.save_dir / f"{config.name_prefix}_{config.phenotype}"

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=experiment_dir,
        enable_checkpointing=True,
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Save metrics to CSV using the best checkpoint
    save_metrics(trainer, model, val_dataloader, test_dataloader, checkpoint_callback)

    return model, trainer.log_dir
