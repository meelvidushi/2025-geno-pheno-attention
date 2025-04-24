import torch
import torch.nn as nn
import torch.nn.init as init

from analysis.base import BaseModel, ModelConfig, TrainConfig


class _RijalEtAl(nn.Module):
    def __init__(self, embedding_dim, seq_length, num_layers=3):
        """
        Implements a multi-layer attention mechanism.

        Args:
            embedding_dim (int): Dimension of input features.
            seq_length (int): Length of the sequence.
            num_layers (int): Number of attention layers.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query_dim = self.embedding_dim
        self.key_dim = self.embedding_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        # Create learnable matrices for each layer
        self.query_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_dim, self.query_dim)) for _ in range(num_layers)]
        )
        self.key_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_dim, self.key_dim)) for _ in range(num_layers)]
        )
        self.value_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_dim, embedding_dim)) for _ in range(num_layers)]
        )

        # Learnable random projection matrix (reduces input dimensionality)
        self.random_matrix = nn.Parameter(torch.empty(self.seq_length, self.embedding_dim - 1))

        # Learnable coefficients for attended values
        self.coeffs_attended = nn.Parameter(torch.empty(seq_length, embedding_dim))

        # Learnable scalar offset for output adjustment
        self.offset = nn.Parameter(torch.randn(1))

        # Initialize parameters
        self.init_parameters()

    def init_parameters(self):
        init_scale = 0.03  # Small scale for initialization to prevent exploding gradients

        params = [
            *self.query_matrices,
            *self.key_matrices,
            *self.value_matrices,
            self.random_matrix,
            self.coeffs_attended,
            self.offset,
        ]

        for param in params:
            init.normal_(param, std=init_scale)

    def forward(self, one_hot_input: torch.Tensor):
        # Apply a random projection and concatenate it with the last feature, which
        # consists entirely of ones
        attended_values = torch.cat(
            (
                torch.matmul(
                    one_hot_input[:, :, : self.seq_length],
                    self.random_matrix,
                ),
                one_hot_input[:, :, -1:],
            ),
            dim=2,
        )

        # Process through each attention layer
        for i in range(self.num_layers):
            query = torch.matmul(attended_values, self.query_matrices[i])
            key = torch.matmul(attended_values, self.key_matrices[i])
            value = torch.matmul(attended_values, self.value_matrices[i])
            scores = torch.matmul(query, key.transpose(1, 2))
            scores = torch.softmax(scores, dim=-1)  # Softmax for attention weighting

            attended_values = torch.matmul(scores, value)

        # Compute final weighted sum using learned coefficients
        final_output = torch.einsum("bij,ij->b", attended_values, self.coeffs_attended)

        # Add offset term to adjust output scale
        output = final_output + self.offset

        return output


class RijalEtAl(BaseModel):
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__(train_config)
        self.save_hyperparameters()

        self.model_config = model_config

        self.model = _RijalEtAl(
            embedding_dim=model_config.embedding_dim,
            seq_length=model_config.seq_length,
            num_layers=model_config.num_layers,
        )

    def forward(self, genotypes: torch.Tensor):
        batch_size = genotypes.size(0)
        seq_length = self.model_config.seq_length

        # Create one-hot vector embedding for genotype data
        one_hot_input = torch.zeros((batch_size, seq_length, seq_length), device=self.device)

        # Set the diagonal elements to the genotype values
        indices = torch.arange(seq_length, device=self.device)
        one_hot_input[:, indices, indices] = genotypes

        # Add a feature of ones (bias term)
        ones = torch.ones((batch_size, seq_length, 1), device=self.device)
        one_hot_input = torch.cat((one_hot_input, ones), dim=2)

        return self.model(one_hot_input)
