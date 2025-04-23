from abc import ABC, abstractmethod

import lightning as L
import torch
from torch.nn.utils import clip_grad_norm_


class GenoPhenoBase(L.LightningModule, ABC):
    @abstractmethod
    def _prepare_batch(self, batch) -> torch.Tensor:
        pass

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

        loss = self.loss_fn(outputs, phenotypes)

        # Calculate and log metrics
        self.log(f"{phase}_loss", loss, prog_bar=True)

        result = {"loss": loss}

        # Calculate R2 for validation and test
        if phase != "train":
            phenotypes_mean = torch.mean(phenotypes)
            ss_tot = torch.sum((phenotypes - phenotypes_mean) ** 2)
            ss_res = torch.sum((phenotypes - outputs) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            self.log(f"{phase}_r2", r2, prog_bar=(phase != "test"))
            result[f"{phase}_r2"] = r2

        return result

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        result = self._process_batch(batch, "train")
        if result is None:
            return None
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        return self._process_batch(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._process_batch(batch, "test")

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        )

        # This dictionary structure tells Lightning how to manage the optimizer and scheduler.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "strict": True,
            },
        }

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
