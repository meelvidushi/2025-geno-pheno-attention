import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


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
