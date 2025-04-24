from abc import ABC, abstractmethod

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch.nn.utils import clip_grad_norm_

from analysis.train_ops import TrainConfig


class BaseModel(L.LightningModule, ABC):
    """A base model class.

    Subclasses must:
        * Define a forward pass that takes a genotypes Tensor (B, L) as input.
        * Call super().__init__(train_config)
    """
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.save_hyperparameters()

        self.train_config = train_config

        self.loss_fn = nn.MSELoss()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

    @abstractmethod
    def forward(self, genotypes: torch.Tensor):
        pass

    @staticmethod
    def _strip_nan(genotypes: torch.Tensor, phenotypes: torch.Tensor):
        mask = ~torch.isnan(phenotypes)
        return genotypes[mask], phenotypes[mask]

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

    def on_before_optimizer_step(self, optimizer):
        params = [p for p in self.parameters() if p.grad is not None]

        # Calculate the norm using the utility function. We set max_norm=inf to only
        # calculate, not clip here. Lightning's Trainer will handle the actual clipping
        # later if configured.
        norm = clip_grad_norm_(params, max_norm=float("inf"), norm_type=2.0)

        self.log(
            "gradient_norm",
            norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        if self.train_config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
        elif self.train_config.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.train_config.learning_rate,
            )
        else:
            raise NotImplementedError()

        if not self.train_config.lr_schedule:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "strict": True,
            },
        }
