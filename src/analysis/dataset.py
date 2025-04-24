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
    def __init__(self, genotype_path: Path, phenotype_path: Path, phenotype_name: str):
        """Dataset for genotype-phenotype data

        Args:
            genotype_path: Path to genotype numpy file (.npy).
            phenotype_path: Path to phenotype numpy file (.npy).
            phenotype_name: Name of the phenotype.
        """
        self.phenotype_name = phenotype_name
        self._phenotype_idx = phenotype_names.index(self.phenotype_name)

        self.genotypes = torch.from_numpy(np.load(genotype_path)).float()
        self.phenotypes = torch.from_numpy(np.load(phenotype_path)).float()

        assert self.genotypes.size(0) == self.phenotypes.size(0)

    def __len__(self) -> int:
        return len(self.genotypes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        genotype = self.genotypes[idx]
        phenotype = self.phenotypes[idx, self._phenotype_idx]

        return genotype, phenotype


def create_dataloaders(
    data_dir: Path,
    phenotype_name: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders

    Args:
        data_dir: Directory containing the data files
        phenotype_name: Name of the phenotype to predict
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for dataloaders
        pin_memory: Whether to pin memory in dataloaders

    Returns:
        Tuple containing train, val, and test dataloaders
    """
    train_dataset = GenoPhenoDataset(
        data_dir / GENO_TRAIN_PATHNAME,
        data_dir / PHENO_TRAIN_PATHNAME,
        phenotype_name,
    )

    val_dataset = GenoPhenoDataset(
        data_dir / GENO_VAL_PATHNAME,
        data_dir / PHENO_VAL_PATHNAME,
        phenotype_name,
    )

    test_dataset = GenoPhenoDataset(
        data_dir / GENO_TEST_PATHNAME,
        data_dir / PHENO_TEST_PATHNAME,
        phenotype_name,
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
