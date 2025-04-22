"""This module recapitulates the results in Rijal et. al. 2025.

https://github.com/Emergent-Behaviors-in-Biology/GenoPhenoMapAttention/blob/main/experiment/single_env_attention_QTL_yeast_data.ipynb

ThreeLayerAttention:
    This is a copy-paste from the notebook with very minor edits required due to the
    notebook's use of global variables used within the class.

Rijal2025:
    This is a lightning module subclass.
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from analysis.base import GenoPhenoBase


class ThreeLayerAttention(nn.Module):
    def __init__(self, input_dim, query_dim, key_dim, seq_length):
        """
        Implements a three-layer attention mechanism.

        Args:
            input_dim (int): Dimension of input features.
            query_dim (int): Dimension of the query matrix.
            key_dim (int): Dimension of the key matrix.
        """
        super().__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.seq_length = seq_length

        # Learnable weight matrices for the first attention layer
        self.query_matrix_1 = nn.Parameter(torch.empty(input_dim, query_dim))
        self.key_matrix_1 = nn.Parameter(torch.empty(input_dim, key_dim))
        self.value_matrix_1 = nn.Parameter(torch.empty(input_dim, input_dim))

        # Learnable weight matrices for the second attention layer
        self.query_matrix_2 = nn.Parameter(torch.empty(input_dim, query_dim))
        self.key_matrix_2 = nn.Parameter(torch.empty(input_dim, key_dim))
        self.value_matrix_2 = nn.Parameter(torch.empty(input_dim, input_dim))

        # Learnable weight matrices for the third attention layer
        self.query_matrix_3 = nn.Parameter(torch.empty(input_dim, query_dim))
        self.key_matrix_3 = nn.Parameter(torch.empty(input_dim, key_dim))
        self.value_matrix_3 = nn.Parameter(torch.empty(input_dim, input_dim))

        # Learnable random projection matrix (reduces input dimensionality)
        self.random_matrix = nn.Parameter(torch.empty(self.seq_length, self.input_dim - 1))

        # Learnable coefficients for attended values
        self.coeffs_attended = nn.Parameter(torch.empty(seq_length, input_dim))

        # Learnable scalar offset for output adjustment
        self.offset = nn.Parameter(torch.randn(1))

        # Initialize parameters
        self.init_parameters()

    def init_parameters(self):
        init_scale = 0.03  # Small scale for initialization to prevent exploding gradients

        for param in [
            self.query_matrix_1,
            self.key_matrix_1,
            self.value_matrix_1,
            self.query_matrix_2,
            self.key_matrix_2,
            self.value_matrix_2,
            self.query_matrix_3,
            self.key_matrix_3,
            self.value_matrix_3,
            self.random_matrix,
            self.coeffs_attended,
            self.offset,
        ]:
            init.normal_(param, std=init_scale)

    def forward(self, x):
        """
        Forward pass through three layers of self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_loci, input_dim)

        Returns:
            Tensor: Final attended output of shape (batch_size,)
        """
        # Apply a random projection and concatenate it with the last feature, which
        # consists entirely of ones
        y = torch.cat(
            (torch.matmul(x[:, :, : self.seq_length], self.random_matrix), x[:, :, -1:]), dim=2
        )

        # First self-attention layer
        query_1 = torch.matmul(y, self.query_matrix_1)
        key_1 = torch.matmul(y, self.key_matrix_1)
        value_1 = torch.matmul(y, self.value_matrix_1)
        scores_1 = torch.matmul(query_1, key_1.transpose(1, 2))
        scores_1 = torch.softmax(scores_1, dim=-1)  # Softmax for attention weighting
        attended_values_1 = torch.matmul(scores_1, value_1)

        # Second self-attention layer
        query_2 = torch.matmul(attended_values_1, self.query_matrix_2)
        key_2 = torch.matmul(attended_values_1, self.key_matrix_2)
        value_2 = torch.matmul(attended_values_1, self.value_matrix_2)
        scores_2 = torch.matmul(query_2, key_2.transpose(1, 2))
        scores_2 = torch.softmax(scores_2, dim=-1)
        attended_values_2 = torch.matmul(scores_2, value_2)

        # Third self-attention layer
        query_3 = torch.matmul(attended_values_2, self.query_matrix_3)
        key_3 = torch.matmul(attended_values_2, self.key_matrix_3)
        value_3 = torch.matmul(attended_values_2, self.value_matrix_3)
        scores_3 = torch.matmul(query_3, key_3.transpose(1, 2))
        scores_3 = torch.softmax(scores_3, dim=-1)
        attended_values_3 = torch.matmul(scores_3, value_3)

        # Compute final weighted sum using learned coefficients
        attended_values_3 = torch.einsum("bij,ij->b", attended_values_3, self.coeffs_attended)

        # Add offset term to adjust output scale
        output = attended_values_3 + self.offset

        return output


class Rijal2025(GenoPhenoBase):
    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        key_dim: int,
        seq_length: int,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        self.model = ThreeLayerAttention(
            input_dim=input_dim,
            query_dim=query_dim,
            key_dim=key_dim,
            seq_length=seq_length,
        )

        self.loss_fn = nn.MSELoss()

    def _prepare_batch(self, batch) -> torch.Tensor:
        genotypes, _ = batch
        batch_size = genotypes.size(0)

        # Create one-hot vector embedding for genotype data
        one_hot_input = torch.zeros(
            (batch_size, self.seq_length, self.seq_length), device=self.device
        )

        # Set the diagonal elements to the genotype values
        indices = torch.arange(self.seq_length, device=self.device)
        one_hot_input[:, indices, indices] = genotypes

        # Add a feature of ones (bias term)
        ones = torch.ones((batch_size, self.seq_length, 1), device=self.device)
        one_hot_input = torch.cat((one_hot_input, ones), dim=2)

        return one_hot_input
