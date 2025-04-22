from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

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
) -> dict[str, DataLoader]:
    """Create train, validation, and test dataloaders

    Args:
        data_dir: Directory containing the data files
        phenotype_name: Name of the phenotype to predict
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for dataloaders
        pin_memory: Whether to pin memory in dataloaders

    Returns:
        Dictionary containing train, val, and test dataloaders
    """
    train_dataset = GenoPhenoDataset(
        data_dir / "geno_train.npy",
        data_dir / "pheno_train.npy",
        phenotype_name,
    )

    val_dataset = GenoPhenoDataset(
        data_dir / "geno_val.npy",
        data_dir / "pheno_val.npy",
        phenotype_name,
    )

    test_dataset = GenoPhenoDataset(
        data_dir / "geno_test.npy",
        data_dir / "pheno_test.npy",
        phenotype_name,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


if __name__ == "__main__":
    dataset = GenoPhenoDataset(
        Path("./data/geno_train.npy"),
        Path("./data/pheno_train.npy"),
        "23C",
    )

    dataloader = DataLoader(dataset, 16, shuffle=True)

    for batch in dataloader:
        geno, pheno = batch
        print(geno, pheno)
        break
