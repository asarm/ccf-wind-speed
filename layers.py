import torch
from torch import nn
from torch.nn import functional as F


class DualEmbeddingGCN(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float):
        super().__init__()
        self.self_projection = nn.Linear(input_size, output_size)
        self.neighbour_projection = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, self_embedding, influence_embedding, adjacency):
        neighbour = torch.matmul(
            adjacency, self.neighbour_projection(influence_embedding))
        output = F.relu(self.self_projection(self_embedding) + neighbour)
        return self.dropout(output)


class DynamicLagGCN(nn.Module):
    """Dynamic-view message passing."""

    def __init__(self, input_size: int, output_size: int, max_lag: int,
                 conv_channels: int, dropout: float):
        super().__init__()
        self.max_lag = max_lag
        self.lag_encoder = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_channels * max_lag, output_size),
        )
        self.self_projection = nn.Linear(input_size, output_size)
        self.neighbour_projection = nn.Linear(input_size, output_size)
        self.lag_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, self_embedding, influence_embedding, ccf_profile,
                adjacency):
        batch, targets, sources, lags = ccf_profile.shape
        if lags != self.max_lag:
            raise ValueError(f"expected {self.max_lag} lags, got {lags}")
        lag_encoding = torch.tanh(self.lag_encoder(
            ccf_profile.reshape(batch * targets * sources, 1, lags)))
        lag_encoding = lag_encoding.reshape(batch, targets, sources, -1)
        source_message = self.neighbour_projection(influence_embedding)
        edge_message = (
            source_message[:, None, :, :]
            + self.lag_scale * lag_encoding
        )
        neighbour = (adjacency.unsqueeze(-1) * edge_message).sum(dim=2)
        # The activation covers both self and neighbour terms.
        output = F.relu(self.self_projection(self_embedding) + neighbour)
        return self.dropout(output)
