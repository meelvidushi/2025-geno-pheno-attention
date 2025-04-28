import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.base import BaseModel, ModelConfig, TrainConfig


class _ModifiedRijalEtAl(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        seq_length: int,
        num_phenotypes: int = 1,
        num_layers: int = 3,
        init_scale: float = 0.03,
        skip_connections: bool = False,
        scaled_attention: bool = False,
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.num_phenotypes = num_phenotypes
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.skip_connections = skip_connections
        self.scaled_attention = scaled_attention
        self.layer_norm = layer_norm

        self.locus_embeddings = nn.Embedding(seq_length, embedding_dim)  # (L, D)
        self.register_buffer("locus_indices", torch.arange(self.seq_length))

        self.q_linears = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim, bias=True)
                for _ in range(num_layers)
            ]
        )
        self.k_linears = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim, bias=True)
                for _ in range(num_layers)
            ]
        )
        self.v_linears = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim, bias=True)
                for _ in range(num_layers)
            ]
        )

        if self.layer_norm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(embedding_dim) for _ in range(num_layers)]
            )

        self.dropout = nn.Dropout(dropout_rate)

        self.readout = nn.Linear(
            seq_length * embedding_dim,
            num_phenotypes,
            bias=True,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.locus_embeddings.weight, std=self.init_scale)
        for q, k, v in zip(self.q_linears, self.k_linears, self.v_linears, strict=True):
            nn.init.normal_(q.weight, std=self.init_scale)
            nn.init.normal_(k.weight, std=self.init_scale)
            nn.init.normal_(v.weight, std=self.init_scale)
            if q.bias is not None:
                nn.init.zeros_(q.bias)
                nn.init.zeros_(k.bias)
                nn.init.zeros_(v.bias)

    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            genotypes (Tensor): shape ``(B, L)`` continuous genotype values.
        Returns:
            Tensor: shape ``(B,)`` predicted phenotype.
        """
        embeddings = self.locus_embeddings(self.locus_indices)
        # (B, L, D) = (B, L, 1) * (L, D)
        x = genotypes.unsqueeze(-1) * embeddings

        for i in range(self.num_layers):
            residual = x

            # Pre-layer normalization.
            if self.layer_norm:
                x_norm = self.norms[i](x)
            else:
                x_norm = x

            q = self.q_linears[i](x_norm)  # (B, L, D)
            k = self.k_linears[i](x_norm)  # (B, L, D)
            v = self.v_linears[i](x_norm)  # (B, L, D)

            scores = torch.matmul(q, k.transpose(1, 2))
            if self.scaled_attention:
                scores = scores / math.sqrt(self.embedding_dim)
            attn = F.softmax(scores, dim=-1)

            attn_out = torch.matmul(attn, v)  # (B, L, D)

            if self.skip_connections and i > 0:
                x = residual + attn_out
            else:
                x = attn_out

        x = x.reshape(x.size(0), -1)  # (B, L*D)
        phenotypes = self.readout(self.dropout(x))

        return phenotypes


class ModifiedRijalEtAl(BaseModel):
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__(train_config)
        self.save_hyperparameters()

        self.model_config = model_config

        self.model = _ModifiedRijalEtAl(
            embedding_dim=model_config.embedding_dim,
            num_phenotypes=train_config.num_phenotypes,
            seq_length=model_config.seq_length,
            num_layers=model_config.num_layers,
            init_scale=model_config.init_scale,
            layer_norm=model_config.layer_norm,
            skip_connections=model_config.skip_connections,
            scaled_attention=model_config.scaled_attention,
            dropout_rate=model_config.dropout_rate,
        )

    def forward(self, genotypes: torch.Tensor):
        return self.model(genotypes)
