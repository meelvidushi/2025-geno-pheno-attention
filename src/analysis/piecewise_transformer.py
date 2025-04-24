import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.base import BaseModel, ModelConfig, TrainConfig


class _PiecewiseTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        seq_length: int,
        num_layers: int = 3,
        skip_connections: bool = False,
        scaled_attention: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        self.scaled_attention = scaled_attention

        self.locus_embeddings = nn.Parameter(torch.empty(seq_length, embedding_dim))  # (L, D)

        self.q_linears = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim, bias=True) for _ in range(num_layers)]
        )
        self.k_linears = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim, bias=True) for _ in range(num_layers)]
        )
        self.v_linears = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim, bias=True) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.coeffs_attended = nn.Parameter(torch.empty(seq_length, embedding_dim))  # (L, D)
        self.offset = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

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

    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            genotypes (Tensor): shape ``(B, L)`` continuous genotype values.
        Returns:
            Tensor: shape ``(B,)`` predicted phenotype.
        """
        # (B, L, D) = (B, L, 1) * (L, D)
        x = genotypes.unsqueeze(-1) * self.locus_embeddings  # broadcasting mult

        for i in range(self.num_layers):
            q = self.q_linears[i](x)  # (B, L, D)
            k = self.k_linears[i](x)  # (B, L, D)
            v = self.v_linears[i](x)  # (B, L, D)

            scores = torch.matmul(q, k.transpose(1, 2))

            if self.scaled_attention:
                scores = scores / math.sqrt(self.embedding_dim)

            attn = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)  # (B, L, D)

            attn_out = self.dropout(attn_out)  # dropout on attention output

            if self.skip_connections and i > 0:
                x = x + attn_out
            else:
                x = attn_out

        phenotype = torch.einsum("bij,ij->b", x, self.coeffs_attended) + self.offset

        return phenotype


class PiecewiseTransformer(BaseModel):
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__(train_config)
        self.save_hyperparameters()

        self.model_config = model_config

        self.model = _PiecewiseTransformer(
            embedding_dim=model_config.embedding_dim,
            seq_length=model_config.seq_length,
            num_layers=model_config.num_layers,
            skip_connections=model_config.skip_connections,
            dropout_rate=model_config.dropout_rate,
        )

    def forward(self, genotypes: torch.Tensor):
        return self.model(genotypes)
