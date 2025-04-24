from pathlib import Path

import attrs
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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




############################################################
# Refactored StackedAttention
############################################################
class StackedAttention(nn.Module):
    """A more canonical multi‑layer self‑attention block.

    *   Each locus has a learnable *embedding* vector.
    *   Q/K/V projections are `nn.Linear` layers (with bias).
    *   Skip‑connections are optional.
    *   No extra scaling / LayerNorm / dropout is introduced on purpose.
    """

    def __init__(
        self,
        embedding_dim: int,
        seq_length: int,
        num_layers: int = 3,
        skip_connections: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.skip_connections = skip_connections

        # --- 1.  Locus‑level embeddings ------------------------------------
        #     "random_matrix" in the original code was effectively a per‑locus
        #     trainable vector multiplied by the genotype value.  Replace it
        #     with a standard embedding table.
        self.locus_embeddings = nn.Parameter(torch.empty(seq_length, embedding_dim))  # (L, D)

        # --- 2.  Q/K/V projections per layer --------------------------------
        self.q_linears = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim, bias=True) for _ in range(num_layers)]
        )
        self.k_linears = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim, bias=True) for _ in range(num_layers)]
        )
        self.v_linears = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim, bias=True) for _ in range(num_layers)]
        )

        # --- 3.  Output aggregation ----------------------------------------
        # Learnable per‑locus, per‑feature coefficients → scalar phenotype.
        self.coeffs_attended = nn.Parameter(torch.empty(seq_length, embedding_dim))  # (L, D)
        self.offset = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    # ---------------------------------------------------------------------
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.locus_embeddings)
        nn.init.xavier_uniform_(self.coeffs_attended)
        for q, k, v in zip(self.q_linears, self.k_linears, self.v_linears, strict=False):
            nn.init.xavier_uniform_(q.weight)
            nn.init.xavier_uniform_(k.weight)
            nn.init.xavier_uniform_(v.weight)
            if q.bias is not None:
                nn.init.zeros_(q.bias)
                nn.init.zeros_(k.bias)
                nn.init.zeros_(v.bias)
        nn.init.zeros_(self.offset)

    # ---------------------------------------------------------------------
    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            genotypes (Tensor): shape ``(B, L)`` continuous genotype values.
        Returns:
            Tensor: shape ``(B,)`` predicted phenotype.
        """
        # (B, L, D) = (B, L, 1) * (L, D)
        x = genotypes.unsqueeze(-1) * self.locus_embeddings  # broadcasting mult

        # ----------------------------------------------- attention stack ----
        for i in range(self.num_layers):
            q = self.q_linears[i](x)  # (B, L, D)
            k = self.k_linears[i](x)  # (B, L, D)
            v = self.v_linears[i](x)  # (B, L, D)

            # attention weights: (B, L, L)
            scores = torch.matmul(q, k.transpose(1, 2))  # no 1/√d scaling per spec
            attn = F.softmax(scores, dim=-1)
            x_next = torch.matmul(attn, v)  # (B, L, D)

            if self.skip_connections and i > 0:
                x = x + x_next
            else:
                x = x_next

        # -------------------------------------- aggregate loci to scalar ----
        # (B,) ← einsum_{bij,ij -> b}
        phenotype = torch.einsum("bij,ij->b", x, self.coeffs_attended) + self.offset
        return phenotype


############################################################
# Refactored LightningModule
############################################################
class RijalEtAl(L.LightningModule):
    """Lightning wrapper around *StackedAttention* with canonical utilities."""

    def __init__(
        self,
        embedding_dim: int,
        seq_length: int,
        num_layers: int = 3,
        skip_connections: bool = False,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.model = StackedAttention(
            embedding_dim=embedding_dim,
            seq_length=seq_length,
            num_layers=num_layers,
            skip_connections=skip_connections,
        )

        self.loss_fn = nn.MSELoss()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _strip_nan(genotypes: torch.Tensor, phenotypes: torch.Tensor):
        mask = ~torch.isnan(phenotypes)
        return genotypes[mask], phenotypes[mask]

    # ---------------------------------------------------------------- forward
    def forward(self, genotypes: torch.Tensor):  # (B, L)
        return self.model(genotypes)

    # ---------------------------------------------------- common step logic --
    def _step(self, batch, phase: str):
        genotypes, phenotypes = batch  # shapes: (B, L), (B,)
        genotypes, phenotypes = self._strip_nan(genotypes, phenotypes)
        if genotypes.numel() == 0:
            return None  # all‑NaN batch—skip

        preds = self(genotypes)
        loss = self.loss_fn(preds, phenotypes)
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=(phase == "train"))

        if phase == "val":
            self.val_r2.update(preds, phenotypes)
        elif phase == "test":
            self.test_r2.update(preds, phenotypes)
        return loss

    # ------------------------------------------------ lightning hooks --------
    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def on_validation_epoch_end(self):
        self.log("val_r2", self.val_r2.compute(), prog_bar=True)
        self.val_r2.reset()

    def on_test_epoch_end(self):
        self.log("test_r2", self.test_r2.compute())
        self.test_r2.reset()

    # ------------------------------------------------ optimizer -------------
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
