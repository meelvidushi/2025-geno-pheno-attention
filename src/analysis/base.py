from abc import ABC, abstractmethod

import lightning as L
import torch


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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
