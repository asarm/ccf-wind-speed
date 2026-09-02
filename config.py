from dataclasses import dataclass


@dataclass(frozen=True)
class LACGNNConfig:
    input_length: int = 48
    ccf_window: int = 48
    output_length: int = 24
    window_stride: int = 3
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 10
    scheduler_patience: int = 4
    scheduler_factor: float = 0.75
    lstm_hidden_size: int = 128
    gnn_hidden_size: int = 128
    graph_embedding_size: int = 32
    prediction_hidden_size: int = 64
    dropout: float = 0.20
    max_lag: int = 24
    top_k_fraction: float = 0.25
    lag_conv_channels: int = 8

    def __post_init__(self):
        if self.ccf_window < self.input_length:
            raise ValueError("ccf_window cannot be shorter than input_length")
        if self.ccf_window <= self.max_lag:
            raise ValueError("ccf_window must be greater than max_lag")
        if not 0.0 < self.top_k_fraction < 1.0:
            raise ValueError("top_k_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave a test split")
