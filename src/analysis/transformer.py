import torch
import torch.nn as nn

from analysis.base import BaseModel, ModelConfig, TrainConfig


class _Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout_rate: float = 0.1,
        num_phenotypes: int = 1,
        seq_length: int = 1164,
    ):
        super().__init__()

        if embedding_dim % nhead != 0:
            raise ValueError(f"d_model ({embedding_dim}) must be divisible by nhead ({nhead})")

        self.num_phenotypes = num_phenotypes
        self.d_model = embedding_dim
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate

        # === Input: Locus embeddings ===
        self.pos_embed = nn.Parameter(torch.empty(seq_length, embedding_dim)
         .normal_(mean=0.0, std=0.02))

        # === Transformer encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        # === Learned phenotype embeddings ===
        self.phenotype_embeddings = nn.Parameter(torch.randn(num_phenotypes, embedding_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=nhead,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, 1)

    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        B = genotypes.size(0)
        D = self.d_model
        L = self.seq_length
        P = self.num_phenotypes

        # === Encode loci ===
        x = genotypes.unsqueeze(-1) * self.pos_embed  # (B, L, D)
        # === Encode genotype sequence ===
        encoded = self.transformer_encoder(x)  # (B, L, D)

        # === Decode via phenotype tokens ===
        phenotype_queries = self.phenotype_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)
        attended, _ = self.cross_attention(
        query=phenotype_queries, key=encoded, value=encoded)
        # NEW ─ residual skip on decoder tokens
        attended = self.decoder_norm(attended + phenotype_queries)  # (B, P, D)

        out = self.output_projection(attended).squeeze(-1)  # (B, P)

        return out

class Transformer(BaseModel):
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__(train_config)
        self.save_hyperparameters()

        self.model_config = model_config

        self.model = _Transformer(
            embedding_dim=model_config.embedding_dim,
            nhead=model_config.nhead,
            num_layers=model_config.num_layers,
            dim_feedforward=model_config.dim_feedforward,
            dropout_rate=model_config.dropout_rate,
            num_phenotypes=train_config.num_phenotypes,
            seq_length=model_config.seq_length,
        )

    def forward(self, genotypes: torch.Tensor):
        return self.model(genotypes)
