from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

GENO_TRAIN_PATHNAME = "geno_train.npy"
PHENO_TRAIN_PATHNAME = "pheno_train.npy"
GENO_VAL_PATHNAME = "geno_val.npy"
PHENO_VAL_PATHNAME = "pheno_val.npy"
GENO_TEST_PATHNAME = "geno_test.npy"
PHENO_TEST_PATHNAME = "pheno_test.npy"

phenotype_names: list[str] = [
    "23C",
    "25C",
    "27C",
    "30C",
    "33C",
    "35C",
    "37C",
    "cu",
    "suloc",
    "ynb",
    "eth",
    "gu",
    "li",
    "mann",
    "mol",
    "raff",
    "sds",
    "4NQO",
]


class GenoPhenoDataset(Dataset):
    def __init__(self, genotype_path: Path, phenotype_path: Path, phenotypes: list[str]):
        """Dataset for genotype-phenotype data

        Args:
            genotype_path: Path to genotype numpy file (.npy).
            phenotype_path: Path to phenotype numpy file (.npy).
            phenotypes: List of phenotype names to predict.
        """
        self.phenotypes_list = phenotypes
        self._phenotype_indices = [phenotype_names.index(pheno) for pheno in self.phenotypes_list]

        self.genotypes = torch.from_numpy(np.load(genotype_path)).float()
        self.phenotypes_data = torch.from_numpy(np.load(phenotype_path)).float()

        assert self.genotypes.size(0) == self.phenotypes_data.size(0)

    def __len__(self) -> int:
        return len(self.genotypes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        genotype = self.genotypes[idx]
        phenotypes = self.phenotypes_data[idx, self._phenotype_indices]

        return genotype, phenotypes


def create_dataloaders(
    data_dir: Path,
    phenotypes: list[str],
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders

    Args:
        data_dir: Directory containing the data files
        phenotypes: List of phenotype names to predict
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for dataloaders
        pin_memory: Whether to pin memory in dataloaders

    Returns:
        Tuple containing train, val, and test dataloaders
    """
    train_dataset = GenoPhenoDataset(
        data_dir / GENO_TRAIN_PATHNAME,
        data_dir / PHENO_TRAIN_PATHNAME,
        phenotypes,
    )

    val_dataset = GenoPhenoDataset(
        data_dir / GENO_VAL_PATHNAME,
        data_dir / PHENO_VAL_PATHNAME,
        phenotypes,
    )

    test_dataset = GenoPhenoDataset(
        data_dir / GENO_TEST_PATHNAME,
        data_dir / PHENO_TEST_PATHNAME,
        phenotypes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_dir = Path("datasets")
    dataset = GenoPhenoDataset(
        data_dir / GENO_TRAIN_PATHNAME,
        data_dir / PHENO_TRAIN_PATHNAME,
        phenotype_names,
    )

    dataloader = DataLoader(dataset, batch_size=8)
