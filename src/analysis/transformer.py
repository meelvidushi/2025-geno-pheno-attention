import torch
import torch.nn as nn

from analysis.base import BaseModel, ModelConfig, TrainConfig


class _Transformer(nn.Module):
    """A vanilla transformer.

    Uses pre-layer normalization.

    Args:
        d_model (int): Dimension of the embeddings and the transformer model.
        nhead (int): Number of attention heads in the multi-head attention.
        num_encoder_layers (int): Number of sub-encoder-layers in the encoder.
        dim_feedforward (int): Dimension of the feedforward network model.
        dropout (float): Dropout value.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1048,
        dropout_rate: float = 0.1,
        num_phenotypes: int = 1,
        seq_length: int = 1164,
    ):
        super().__init__()

        if embedding_dim % nhead != 0:
            raise ValueError(f"d_model ({embedding_dim}) must be divisible by nhead ({nhead})")

        self.num_phenotypes = num_phenotypes
        self.d_model = embedding_dim
        self.dropout_rate = dropout_rate
        self.seq_length = seq_length

        self.locus_embeddings = nn.Embedding(seq_length, embedding_dim)  # (L, D)
        self.register_buffer("locus_indices", torch.arange(self.seq_length))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        self.fc_dropout = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(
            embedding_dim,
            num_phenotypes,
            bias=True,
        )

    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        embeddings = self.locus_embeddings(self.locus_indices)

        # (B, L, D) = (B, L, 1) * (L, D)
        x = genotypes.unsqueeze(-1) * embeddings

        encoder_output = self.transformer_encoder(src=x)

        pooled_output = encoder_output.mean(dim=1)
        prediction = self.fc_out(self.fc_dropout(pooled_output))

        return prediction


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
