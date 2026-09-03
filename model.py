import torch
from torch import nn
from torch.nn import functional as F

from .config import LACGNNConfig
from .layers import DualEmbeddingGCN, DynamicLagGCN


class RowPercentileTopK(nn.Module):
    """Row-wise percentile filtering followed by normalization."""

    def __init__(self, fraction: float, remove_diagonal: bool = True):
        super().__init__()
        self.fraction = fraction
        self.remove_diagonal = remove_diagonal

    def forward(self, scores):
        scores = scores.clamp_min(0.0)
        if self.remove_diagonal:
            diagonal = torch.eye(
                scores.shape[-1], dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(diagonal, 0.0)
        threshold = torch.quantile(
            scores, 1.0 - self.fraction, dim=-1, keepdim=True)
        filtered = torch.where(scores >= threshold, scores, 0.0)
        return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class FeatureLSTMEncoder(nn.Module):
    """Five feature-specific encoders; direction jointly consumes sin and cos."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        input_sizes = (1, 1, 1, 1, 2)
        self.encoders = nn.ModuleList([
            nn.LSTM(size, hidden_size, batch_first=True)
            for size in input_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        batch, _, steps, nodes = inputs.shape
        sequences = [inputs[:, index:index + 1] for index in range(4)]
        sequences.append(inputs[:, 4:6])
        embeddings = []
        for sequence, encoder in zip(sequences, self.encoders):
            sequence = sequence.permute(0, 3, 2, 1).reshape(
                batch * nodes, steps, sequence.shape[1])
            _, (hidden, _) = encoder(sequence)
            embeddings.append(hidden[-1].reshape(batch, nodes, -1))
        return self.dropout(torch.stack(embeddings, dim=2))


class DirectedGraphLearner(nn.Module):
    """Rows are targets and columns are information sources."""

    def __init__(self, nodes: int, embedding_size: int):
        super().__init__()
        self.source_embeddings = nn.Parameter(
            torch.randn(nodes, embedding_size))
        self.target_embeddings = nn.Parameter(
            torch.randn(nodes, embedding_size))

    def forward(self):
        source = F.normalize(self.source_embeddings, dim=-1)
        target = F.normalize(self.target_embeddings, dim=-1)
        # A[target, source] represents source -> target.
        return torch.nan_to_num(target @ source.transpose(0, 1), nan=0.0)


def _full_ccf(wind, max_lag: int):
    batch, steps, _ = wind.shape
    if not 1 <= max_lag < steps:
        raise ValueError("max_lag must be smaller than the input window")
    normalized = (wind - wind.mean(dim=1, keepdim=True)) / (
        wind.std(dim=1, keepdim=True) + 1e-5)
    normalized = normalized.permute(0, 2, 1)
    fft_size = 2 * steps
    spectrum = torch.fft.rfft(normalized, n=fft_size, dim=-1)
    cross_spectrum = (
        torch.conj(spectrum.unsqueeze(2)) * spectrum.unsqueeze(1))
    ccf = torch.fft.irfft(cross_spectrum, n=fft_size, dim=-1)
    indices = torch.cat((
        torch.arange(fft_size - max_lag, fft_size, device=wind.device),
        torch.arange(0, max_lag + 1, device=wind.device),
    ))
    return ccf[..., indices] / steps


def dynamic_ccf_graph(wind, max_lag: int, sparsifier: RowPercentileTopK):
    """Signed peak-lag direction and one-sided profiles."""
    ccf = _full_ccf(wind, max_lag)
    peak_strength, peak_index = ccf.abs().max(dim=-1)
    negative = peak_index < max_lag
    positive = peak_index > max_lag
    zero = peak_index == max_lag

    # A[target, source]: positive lag means i leads j, hence A[j, i].
    adjacency = torch.where(negative, peak_strength, 0.0)
    adjacency = torch.maximum(
        adjacency,
        torch.where(positive, peak_strength, 0.0).transpose(1, 2),
    )
    zero_edges = torch.where(zero, peak_strength, 0.0)
    adjacency = torch.maximum(adjacency, zero_edges)
    adjacency = torch.maximum(adjacency, zero_edges.transpose(1, 2))
    adjacency = sparsifier(adjacency)

    # For A[target, source], both expressions describe lags 1..max_lag in
    # the assigned propagation direction. Averaging removes FFT round-off.
    positive_profile = ccf[..., max_lag + 1:].transpose(1, 2)
    negative_profile = ccf[..., :max_lag].flip(-1)
    directional_profile = 0.5 * (positive_profile + negative_profile)
    return directional_profile.contiguous(), adjacency


def static_ccf_scores(wind, max_lag: int):
    """Similarity between full training-set CCF descriptors."""
    ccf = _full_ccf(wind, max_lag)
    nodes = wind.shape[-1]
    descriptors = ccf.reshape(1, nodes, -1)
    distance = torch.abs(
        descriptors.unsqueeze(2) - descriptors.unsqueeze(1)).mean(dim=-1)
    low = distance.amin(dim=(1, 2), keepdim=True)
    high = distance.amax(dim=(1, 2), keepdim=True)
    return (1.0 - (distance - low) / (high - low + 1e-6)).squeeze(0)


class LACGNN(nn.Module):
    VIEWS = ("learned", "dynamic", "static")

    def __init__(self, nodes: int, static_adjacency, config: LACGNNConfig):
        super().__init__()
        self.config = config
        self.feature_encoder = FeatureLSTMEncoder(
            config.lstm_hidden_size, config.dropout)
        self.graph_learner = DirectedGraphLearner(
            nodes, config.graph_embedding_size)
        self.sparsifier = RowPercentileTopK(config.top_k_fraction)
        self.register_buffer("static_scores", static_adjacency)

        self.self_weights = nn.ParameterDict({
            view: nn.Parameter(torch.randn(nodes, 5)) for view in self.VIEWS
        })
        self.influence_weights = nn.ParameterDict({
            view: nn.Parameter(torch.randn(nodes, 5)) for view in self.VIEWS
        })
        self.learned_gnn = DualEmbeddingGCN(
            config.lstm_hidden_size, config.gnn_hidden_size, config.dropout)
        self.dynamic_gnn = DynamicLagGCN(
            config.lstm_hidden_size, config.gnn_hidden_size, config.max_lag,
            config.lag_conv_channels, config.dropout)
        self.static_gnn = DualEmbeddingGCN(
            config.lstm_hidden_size, config.gnn_hidden_size, config.dropout)

        self.view_normalization = nn.LayerNorm(3 * config.gnn_hidden_size)
        self.view_scorer = nn.Linear(3 * config.gnn_hidden_size, 3)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.gnn_hidden_size, config.prediction_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prediction_hidden_size, config.output_length),
        )

    @staticmethod
    def _weighted_embedding(embeddings, weights):
        weights = F.softmax(weights, dim=-1)
        return (embeddings * weights[None, :, :, None]).sum(dim=2)

    def encode_views(self, inputs):
        embeddings = self.feature_encoder(
            inputs[:, :, -self.config.input_length:, :])
        mixed = {}
        for view in self.VIEWS:
            mixed[(view, "self")] = self._weighted_embedding(
                embeddings, self.self_weights[view])
            mixed[(view, "influence")] = self._weighted_embedding(
                embeddings, self.influence_weights[view])

        learned_adjacency = self.sparsifier(self.graph_learner())
        learned = self.learned_gnn(
            mixed[("learned", "self")],
            mixed[("learned", "influence")], learned_adjacency)

        profile, dynamic_adjacency = dynamic_ccf_graph(
            inputs[:, 0], self.config.max_lag, self.sparsifier)
        dynamic = self.dynamic_gnn(
            mixed[("dynamic", "self")],
            mixed[("dynamic", "influence")], profile, dynamic_adjacency)

        static_adjacency = self.sparsifier(self.static_scores)
        static = self.static_gnn(
            mixed[("static", "self")],
            mixed[("static", "influence")], static_adjacency)
        return torch.stack((learned, dynamic, static), dim=2)

    def fuse_views(self, views):
        concatenated = views.flatten(start_dim=2)
        weights = F.softmax(
            self.view_scorer(self.view_normalization(concatenated)), dim=-1)
        fused = (views * weights.unsqueeze(-1)).sum(dim=2)
        return fused, weights

    def forward(self, inputs, return_attention: bool = False):
        fused, attention = self.fuse_views(self.encode_views(inputs))
        prediction = self.prediction_head(fused).permute(0, 2, 1)
        return (prediction, attention) if return_attention else prediction
