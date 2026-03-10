import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime


from gnn_layers import get_gnn_layer

# --- Configuration ---
DATA_DIR = './datasets/ceda_data'
WINDOW_SIZE = 48
MODEL_WINDOW = 48
HORIZON_SIZE = 24
STEP_SIZE = 3
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20
TEST_SPLIT = 0.20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
D_STATION_EMB = 32
D_MODEL = 128 # 128, 200
D_HIDDEN_GCN = 128 # 256, 300
NUM_HORIZON_GROUPS = 1  # Number of horizon groups with separate attention/heads

# New GNN hyperparameters
NUM_GNN_LAYERS = 1  # 1 or 2
GNN_TYPE = 'GCN'

# --- Data Processing ---
HOURLY_DATA_STATIONS = [
    'Portland', 'Seattle', 'Los Angeles', 'San Diego',
    'Las Vegas', 'Phoenix', 'Albuquerque', 'Denver', 'San Antonio',
    'Dallas', 'Houston', 'Kansas City', 'Minneapolis', 'Saint Louis',
    'Chicago', 'Nashville', 'Indianapolis', 'Atlanta', 'Detroit',
    'Jacksonville', 'Charlotte', 'Pittsburgh', 'Toronto', 'Philadelphia',
    'Montreal', 'Boston'
]

def load_and_clean_data(data_dir):
    """Loads, cleans, and aligns data from CSV files."""
    print("Loading data...")
    wind = pd.read_csv(os.path.join(data_dir, 'wind_speed.csv'))[1:]
    pressure = pd.read_csv(os.path.join(data_dir, 'pressure.csv'))[1:]
    humidity = pd.read_csv(os.path.join(data_dir, 'humidity.csv'))[1:]
    temperature = pd.read_csv(os.path.join(data_dir, 'temperature.csv'))[1:]
    wind_direction = pd.read_csv(os.path.join(data_dir, 'wind_direction.csv'))[1:]

    # Datetime handling
    wind['datetime'] = pd.to_datetime(wind['datetime'])
    wind.set_index('datetime', inplace=True)

    # Align other dataframes to wind index
    for df in [pressure, humidity, temperature, wind_direction]:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    # Dataset-specific station filtering
    if 'hourly_data' in data_dir and 'ireland' not in data_dir:
        # Use predefined station list for hourly_data
        stations_to_keep = [s for s in HOURLY_DATA_STATIONS if s in wind.columns]
        cities2exclude = [c for c in wind.columns if c not in stations_to_keep]
        print(f"Using predefined station list: {len(stations_to_keep)} stations")
    else:
        # Check missing data ratio across ALL features
        cities2exclude = set()
        total_rows = len(wind)

        # All dataframes: wind, pressure, humidity, temperature, wind_direction
        dfs_check = [wind, pressure, humidity, temperature, wind_direction]
        df_names = ['wind', 'pressure', 'humidity', 'temperature', 'wind_direction']

        for name, df in zip(df_names, dfs_check):
            # Ensure index alignment for valid count
            # We assume roughly same size, but use len(df) to be safe
            current_rows = len(df)
            missing_ratio = df.isna().sum() / current_rows
            bad_cities = list(missing_ratio[missing_ratio >= 0.30].index)
            if bad_cities:
                print(f"  {name}: {len(bad_cities)} stations with >=30% missing data")
                cities2exclude.update(bad_cities)

        cities2exclude = list(cities2exclude)
        print(f"Excluding {len(cities2exclude)} stations: {cities2exclude}")

    # Find common columns across all dataframes
    common_columns = set(wind.columns) & set(pressure.columns) & set(humidity.columns) & set(temperature.columns) & set(wind_direction.columns)
    # Remove excluded cities
    common_columns = common_columns - set(cities2exclude)
    common_columns = sorted(list(common_columns))  # Sort for consistent ordering
    print(f"Common stations across all features: {len(common_columns)}")

    # Drop columns and interpolate - use only common columns
    dfs = {'wind': wind, 'pressure': pressure, 'humidity': humidity, 'temperature': temperature, 'wind_direction': wind_direction}
    clean_dfs = {}

    for name, df in dfs.items():
        df_clean = df[common_columns]  # Keep only common columns
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        df_clean = df_clean.dropna() # Drop rows that still have NaNs
        clean_dfs[name] = df_clean
        # print(f"DEBUG: {name} - Range: {df_clean.index.min()} to {df_clean.index.max()} | Len: {len(df_clean)}")

    # Find common timestamps across ALL features
    common_index = clean_dfs['wind'].index
    for name in clean_dfs:
        common_index = common_index.intersection(clean_dfs[name].index)

    print(f"Common timestamps across all features: {len(common_index)}")

    if len(common_index) == 0:
        print("CRITICAL ERROR: No overlapping timestamps found! Check data files.")
        return clean_dfs, []

    # Reindex all to the common intersection
    for name in clean_dfs:
        clean_dfs[name] = clean_dfs[name].reindex(common_index)

        # Verify no NaNs remain
        if clean_dfs[name].isna().any().any():
            print(f"CRITICAL: NaNs detected in {name} after alignment!")
            # Fill remaining NaNs just in case (e.g. ffill/bfill)
            clean_dfs[name] = clean_dfs[name].fillna(method='ffill').fillna(method='bfill')

    return clean_dfs, list(clean_dfs['wind'].columns)

def normalize_data(clean_dfs):
    """
    Normalizes data to [0, 1] per station per feature.
    Each feature is normalized based on its own min/max values for each station.
    Example: Station A's pressure values are normalized using only Station A's pressure min/max.
    """
    print("Normalizing data (per-station, per-feature)...")
    norm_dfs = {}
    stats = {}

    for name, df in clean_dfs.items():
        min_val = df.min()
        max_val = df.max()
        diff = max_val - min_val
        diff[diff == 0] = 1.0

        norm_df = (df - min_val) / diff
        norm_dfs[name] = norm_df
        stats[name] = {'min': min_val, 'max': max_val, 'diff': diff}

    return norm_dfs, stats

def create_space_time_windows(norm_dfs, window_size, horizon_size, step_size):
    """
    Returns:
        X: [Samples, Channels=5, Window, Stations] - 5 channel format for Conv2D
        Y: [Samples, Horizon, Stations]
    """
    # Stack features as separate channels: [Time, Stations, Channels=5]
    # Order: wind, pressure, temperature, humidity, wind_direction
    feature_list = [norm_dfs['wind'], norm_dfs['pressure'], norm_dfs['temperature'], norm_dfs['humidity'], norm_dfs['wind_direction']]
    data_np = np.stack([df.values for df in feature_list], axis=-1)

    n_timesteps, n_stations, n_features = data_np.shape

    X_list = []
    Y_list = []

    for i in range(0, n_timesteps - window_size - horizon_size + 1, step_size):
        # [Window, Stations, Features]
        x_window = data_np[i : i + window_size]
        # [Horizon, Stations] (Target is wind speed, idx 0)
        y_window = data_np[i + window_size : i + window_size + horizon_size, :, 0]

        X_list.append(x_window)
        Y_list.append(y_window)

    X = np.array(X_list)  # [B, T, N, F]
    Y = np.array(Y_list)  # [B, H, N]

    # Transpose X to [B, F, T, N] for Conv2D (channels first)
    X = X.transpose(0, 3, 1, 2)  # [B, Channels=5, Time, Stations]

    return X, Y

def apply_top_percentile_filter(adj, percentile=75):
    """
    Filters adjacency matrix by keeping only the top (100 - percentile)% strongest
    edges per node (row-wise). Each node independently keeps its top 25% edges.

    For each node i, the threshold is the percentile-th quantile of row i.
    Edges with weight >= threshold are kept; the rest are zeroed out.

    Args:
        adj: [N, N] or [B, N, N] adjacency matrix

    Returns:
        filtered_adj: row-normalized adjacency matrix with only top-25% edges per node
    """
    is_batched = adj.dim() == 3
    if not is_batched:
        adj = adj.unsqueeze(0)  # [1, N, N]

    B, N, _ = adj.shape

    # Compute per-row threshold at the given percentile: [B, N, 1]
    threshold = torch.quantile(adj, percentile / 100.0, dim=-1, keepdim=True)

    # Keep edges >= threshold (top 25% per node)
    mask = adj >= threshold  # [B, N, N]

    # Apply mask - zero out sub-threshold edges
    filtered_adj = torch.where(mask, adj, torch.zeros_like(adj))

    # Row normalization (without softmax)
    row_sums = filtered_adj.sum(dim=-1, keepdim=True)
    filtered_adj = filtered_adj / (row_sums + 1e-8)

    if not is_batched:
        filtered_adj = filtered_adj.squeeze(0)

    return filtered_adj

def compute_dynamic_ccf_adj(x, max_lag=24):
    """
    Computes dynamic CCF adjacency matrix for a batch of windows.
    Uses top-25% percentile filtering for edge selection.
    Input: x [Batch, Time, Stations] (only wind speed usually)
    Output: adj [Batch, Stations, Stations]

    New Algorithm:
    1. Standardize per window/station
    2. Compute pairwise CCF via FFT
    3. Find peak correlation and its lag for each pair (i, j)
    4. Construct directed graph:
       - If peak_lag (i->j) < 0 (i leads j), create edge i->j
       - This translates to adj[j, i] = peak_val (target=j, source=i)
    5. Apply top-25% percentile filtering
    """
    B, T, N = x.shape
    device = x.device

    # 1. Standardize (per window, per station)
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-5)

    # 2. Compute CCF using FFT
    x_norm = x_norm.permute(0, 2, 1)
    n_fft = 2 * T
    X_f = torch.fft.rfft(x_norm, n=n_fft, dim=-1)
    C_f = X_f.unsqueeze(2) * torch.conj(X_f.unsqueeze(1))
    ccf = torch.fft.irfft(C_f, n=n_fft, dim=-1)

    indices = torch.cat((
        torch.arange(n_fft - max_lag, n_fft, device=device),
        torch.arange(0, max_lag + 1, device=device)
    ))

    ccf_selected = ccf[..., indices]
    ccf_selected = ccf_selected / T

    # 3. Peak Finding
    # ccf_selected shape: [B, N, N, 2*max_lag + 1]
    # indices mapping:
    #   0        -> lag = -max_lag
    #   max_lag  -> lag = 0
    #   2*max_lag -> lag = +max_lag

    abs_ccf = torch.abs(ccf_selected)
    peak_values, peak_indices = torch.max(abs_ccf, dim=-1)

    # 4. Construct Directed Adjacency
    # We want adj[target, source] > 0 if source leads target.
    # indices mapping: 0 -> -max_lag, max_lag -> 0, 2*max_lag -> +max_lag

    # Case 1: i leads j (negative lag, indices < max_lag)
    # Effect: j is influenced by i.
    # Set adj[j, i] = peak_value (target=j, source=i)
    mask_i_leads_j = peak_indices < max_lag

    # Case 2: j leads i (positive lag, indices > max_lag)
    # Effect: i is influenced by j.
    # Set adj[i, j] = peak_value (target=i, source=j)
    mask_j_leads_i = peak_indices > max_lag

    # Case 3: lag = 0 (bidirectional, indices == max_lag)
    # Effect: Mutual influence.
    # Set adj[i, j] and adj[j, i]
    mask_zero = peak_indices == max_lag

    adj = torch.zeros(B, N, N, device=device)

    # 1. i leads j: set adj[b, j, i]
    # mask_i_leads_j is boolean mask at [b, i, j]
    # We want to assign peak_values[b, i, j] to adj[b, j, i]
    # Safe method: use indices
    b_idx, i_idx, j_idx = torch.where(mask_i_leads_j)
    adj[b_idx, j_idx, i_idx] = peak_values[mask_i_leads_j]

    # 2. j leads i: set adj[b, i, j]
    # Direct assignment works fine
    adj[mask_j_leads_i] = peak_values[mask_j_leads_i]

    # 3. lag = 0: set both directions
    # Direct assignment for [i, j]
    adj[mask_zero] = peak_values[mask_zero]
    # Assignment for [j, i] using explicit indices
    b_idx, i_idx, j_idx = torch.where(mask_zero)
    adj[b_idx, j_idx, i_idx] = peak_values[mask_zero]

    # 5. Apply top-25% percentile filtering
    adj = apply_top_percentile_filter(adj)

    return adj

def compute_static_ccf_adj(x, max_lag=24):
    """
    Computes static CCF adjacency using profile similarity (undirected).
    Designed for long time series (e.g., full training data).
    Input: x [1, Time, Stations] - single long window
    Output: adj [N, N] (not batched, static)
    """
    B, T, N = x.shape
    device = x.device

    # 1. Standardize
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    x_norm = (x - mean) / (std + 1e-5)

    # 2. Compute CCF using FFT
    x_norm = x_norm.permute(0, 2, 1)  # [1, N, T]
    n_fft = 2 * T
    X_f = torch.fft.rfft(x_norm, n=n_fft, dim=-1)
    C_f = X_f.unsqueeze(2) * torch.conj(X_f.unsqueeze(1))  # [1, N, N, n_fft//2+1]
    ccf = torch.fft.irfft(C_f, n=n_fft, dim=-1)  # [1, N, N, n_fft]

    indices = torch.cat((
        torch.arange(n_fft - max_lag, n_fft, device=device),
        torch.arange(0, max_lag + 1, device=device)
    ))

    ccf_selected = ccf[..., indices]  # [1, N, N, 2*max_lag+1]
    ccf_selected = ccf_selected / T

    # 3. Profile Similarity (undirected)
    profiles = ccf_selected.reshape(1, N, -1)  # [1, N, N*(2*max_lag+1)]
    diff = torch.abs(profiles.unsqueeze(2) - profiles.unsqueeze(1))  # [1, N, N, ...]
    diff_matrix = diff.mean(dim=-1)  # [1, N, N]

    # 4. Normalize to [0, 1] and invert
    min_diff = diff_matrix.amin(dim=(1, 2), keepdim=True)
    max_diff = diff_matrix.amax(dim=(1, 2), keepdim=True)
    norm_diff = (diff_matrix - min_diff) / (max_diff - min_diff + 1e-6)
    ccf_matrix = 1.0 - norm_diff  # [1, N, N]

    # 5. Top-25% percentile filtering
    adj = apply_top_percentile_filter(ccf_matrix)

    return adj.squeeze(0)  # [N, N]

# --- Model Components ---

class GraphLearner(nn.Module):
    """
    Learns a directed (asymmetric) adjacency matrix using separate
    source and target embeddings for each node.
    Uses top-25% percentile filtering for edge selection.
    """
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.source_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.target_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))

    def forward(self):
        # Normalize embeddings
        source_norm = F.normalize(self.source_embeddings, p=2, dim=1)  # [N, D]
        target_norm = F.normalize(self.target_embeddings, p=2, dim=1)  # [N, D]

        # Directed adjacency: source_i -> target_j
        # adj[i,j] = source_i · target_j (asymmetric)
        adj = torch.mm(source_norm, target_norm.t())  # [N, N]

        if torch.isnan(adj).any():
            print("Warning: NaN detected in adjacency matrix inside GraphLearner")
            adj = torch.nan_to_num(adj, 0.0)

        # Apply top-25% percentile filtering
        adj = apply_top_percentile_filter(adj)

        return adj

class FeatureLSTMEncoder(nn.Module):
    """
    Returns embeddings for each feature: [B, N, 5, D_MODEL]
    """
    def __init__(self, hidden_size=D_MODEL, dropout=0.2):
        super().__init__()
        # Separate LSTM for each feature
        self.lstm_wind = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lstm_pressure = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lstm_temperature = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lstm_humidity = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lstm_wind_direction = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, 5, T, N] - 5 channels (wind, pressure, temp, humidity, wind_direction)
        Returns:
            embeddings: [B, N, 5, D_MODEL] - per-station, per-feature embeddings
        """
        B, C, T, N = x.shape

        # Extract each feature: [B, T, N] -> [B*N, T, 1]
        wind = x[:, 0].permute(0, 2, 1).reshape(B * N, T, 1)
        pressure = x[:, 1].permute(0, 2, 1).reshape(B * N, T, 1)
        temperature = x[:, 2].permute(0, 2, 1).reshape(B * N, T, 1)
        humidity = x[:, 3].permute(0, 2, 1).reshape(B * N, T, 1)
        wind_direction = x[:, 4].permute(0, 2, 1).reshape(B * N, T, 1)

        # LSTM encoding for each feature
        _, (h_wind, _) = self.lstm_wind(wind)
        _, (h_pressure, _) = self.lstm_pressure(pressure)
        _, (h_temperature, _) = self.lstm_temperature(temperature)
        _, (h_humidity, _) = self.lstm_humidity(humidity)
        _, (h_wind_direction, _) = self.lstm_wind_direction(wind_direction)

        # Each h: [1, B*N, D_MODEL] -> [B, N, D_MODEL]
        emb_wind = h_wind[-1].reshape(B, N, -1)
        emb_pressure = h_pressure[-1].reshape(B, N, -1)
        emb_temperature = h_temperature[-1].reshape(B, N, -1)
        emb_humidity = h_humidity[-1].reshape(B, N, -1)
        emb_wind_direction = h_wind_direction[-1].reshape(B, N, -1)

        # Stack: [B, N, 5, D_MODEL]
        embeddings = torch.stack([emb_wind, emb_pressure, emb_temperature, emb_humidity, emb_wind_direction], dim=2)

        return self.dropout(embeddings)

class GraphWindModel(nn.Module):
    def __init__(self, num_stations, adj_static=None, num_horizons=HORIZON_SIZE, window_size=WINDOW_SIZE, model_window=MODEL_WINDOW,
                 num_gnn_layers=NUM_GNN_LAYERS, gnn_type=GNN_TYPE, dropout=0.2):
        super().__init__()

        self.num_stations = num_stations
        self.window_size = window_size
        self.model_window = model_window
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type.upper()

        # Register Static Adjacency
        if adj_static is None:
            # Fallback if not provided (should be provided)
            print("Warning: adj_static not provided, initializing to identity")
            adj_static = torch.eye(num_stations)
        self.register_buffer('adj_static_ccf', adj_static)

        # 1. Per-Feature LSTM Encoder
        self.feature_encoder = FeatureLSTMEncoder(hidden_size=D_MODEL, dropout=dropout)

        # 2. Dual Learnable Feature Weights per Station
        # self_weights: [N, 5] - for creating self embedding (used in self-contribution)
        # influence_weights: [N, 5] - for creating influence embedding (used to affect neighbors)
        self.self_weights = nn.Parameter(torch.randn(num_stations, 5))
        self.influence_weights = nn.Parameter(torch.randn(num_stations, 5))

        # 3. Graph Learning (Adaptive View)
        self.graph_learner = GraphLearner(num_stations, D_STATION_EMB)

        # 4. GNN Layers (Triple View)
        # Layer 1: D_MODEL -> D_HIDDEN_GCN
        self.gnn_learned_1 = get_gnn_layer(self.gnn_type, D_MODEL, D_HIDDEN_GCN, dropout=dropout)
        self.gnn_dynamic_ccf_1 = get_gnn_layer(self.gnn_type, D_MODEL, D_HIDDEN_GCN, dropout=dropout)
        self.gnn_static_ccf_1 = get_gnn_layer(self.gnn_type, D_MODEL, D_HIDDEN_GCN, dropout=dropout)

        # Layer 2: D_HIDDEN_GCN -> D_HIDDEN_GCN (only if num_gnn_layers == 2)
        if self.num_gnn_layers == 2:
            self.gnn_learned_2 = get_gnn_layer(self.gnn_type, D_HIDDEN_GCN, D_HIDDEN_GCN, dropout=dropout)
            self.gnn_dynamic_ccf_2 = get_gnn_layer(self.gnn_type, D_HIDDEN_GCN, D_HIDDEN_GCN, dropout=dropout)
            self.gnn_static_ccf_2 = get_gnn_layer(self.gnn_type, D_HIDDEN_GCN, D_HIDDEN_GCN, dropout=dropout)
            # Projection for residual connection (D_MODEL -> D_HIDDEN_GCN)
            self.residual_proj = nn.Linear(D_MODEL, D_HIDDEN_GCN)

        # 5. Per-Group Attention and Prediction (separate modules per group)
        self.num_horizons = num_horizons
        self.num_horizon_groups = NUM_HORIZON_GROUPS
        self.group_size = num_horizons // NUM_HORIZON_GROUPS  # 8 horizons per group

        # Pre-compute horizon to group mapping
        self.register_buffer('horizon_to_group',
            torch.clamp(torch.arange(num_horizons) // self.group_size, max=NUM_HORIZON_GROUPS - 1))

        # Learnable view attention query per group
        # No key projection, keys are views directly
        self.group_attn_query = nn.ModuleList([
            nn.Linear(D_HIDDEN_GCN, D_HIDDEN_GCN) for _ in range(NUM_HORIZON_GROUPS)
        ])
        # Horizon Embedding
        self.horizon_emb = nn.Embedding(NUM_HORIZON_GROUPS, D_HIDDEN_GCN)

        # Separate prediction heads per group (each predicts group_size horizons)
        self.group_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D_HIDDEN_GCN, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, self.group_size)  # Each group predicts 8 horizons
            ) for _ in range(NUM_HORIZON_GROUPS)
        ])

    def forward(self, x_5ch):
        """
        Args:
            x_5ch: [B, 5, T, N] - 5-channel input
        """
        B, C, T, N = x_5ch.shape

        # 1. Encode each feature separately (Using MODEL_WINDOW steps)
        # [B, 5, T, N] -> [B, N, 5, D_MODEL]

        # Slice the input: Take last model_window steps for feature extraction
        start_t = T - self.model_window
        x_features = x_5ch[:, :, start_t:, :]

        feature_embeddings = self.feature_encoder(x_features)

        # 2. Compute dual embeddings with separate weight vectors

        self_w = F.softmax(self.self_weights, dim=1)  # [N, 5]
        influence_w = F.softmax(self.influence_weights, dim=1)  # [N, 5]

        # Expand weights: [1, N, 5, 1]
        self_w_expanded = self_w.unsqueeze(0).unsqueeze(-1)
        influence_w_expanded = influence_w.unsqueeze(0).unsqueeze(-1)

        # Weighted sum over features -> [B, N, D_MODEL]
        self_emb = (feature_embeddings * self_w_expanded).sum(dim=2)  # For self-prediction
        influence_emb = (feature_embeddings * influence_w_expanded).sum(dim=2)  # For influencing neighbors

        # --- View 1: Learned Graph ---
        adj_learned = self.graph_learner()
        z1_learned = self.gnn_learned_1(self_emb, influence_emb, adj_learned)  # [B, N, D_HIDDEN_GCN]

        # --- View 2: Dynamic CCF Graph ---
        wind = x_5ch[:, 0, :, :]  # [B, T, N]
        adj_dynamic_ccf = compute_dynamic_ccf_adj(wind)
        z1_dynamic_ccf = self.gnn_dynamic_ccf_1(self_emb, influence_emb, adj_dynamic_ccf)  # [B, N, D_HIDDEN_GCN]

        # --- View 3: Static CCF Graph ---
        # self.adj_static_ccf is [1, N, N] or [N, N]
        z1_static_ccf = self.gnn_static_ccf_1(self_emb, influence_emb, self.adj_static_ccf)  # [B, N, D_HIDDEN_GCN]

        # Apply second GNN layer with residual connection if num_gnn_layers == 2
        if self.num_gnn_layers == 2:
            # Residual projection for skip connection
            x_residual = self.residual_proj(self_emb)  # [B, N, D_HIDDEN_GCN]

            z2_learned = self.gnn_learned_2(z1_learned, z1_learned, adj_learned)
            emb_learned = z2_learned + x_residual

            z2_dynamic_ccf = self.gnn_dynamic_ccf_2(z1_dynamic_ccf, z1_dynamic_ccf, adj_dynamic_ccf)
            emb_dynamic_ccf = z2_dynamic_ccf + x_residual

            z2_static_ccf = self.gnn_static_ccf_2(z1_static_ccf, z1_static_ccf, self.adj_static_ccf)
            emb_static_ccf = z2_static_ccf + x_residual
        else:
            # Single layer, no residual
            emb_learned = z1_learned
            emb_dynamic_ccf = z1_dynamic_ccf
            emb_static_ccf = z1_static_ccf

        # --- Per-Group Attention Fusion ---
        # Stack views for weighted combination
        # Views: Learned, Dynamic CCF, Static CCF
        views = torch.stack([emb_learned, emb_dynamic_ccf, emb_static_ccf], dim=2)  # [B, N, 3, D_HIDDEN_GCN]

        # Process each group with learnable view weights
        group_preds = []

        for g in range(self.num_horizon_groups):
            # Attention Mechanism:
            # Query: transform of learned embedding [B, N, D]
            # Keys: views directly [B, N, 3, D]

            h_emb = self.horizon_emb(torch.tensor(g, device=x_5ch.device))  # [D]
            # query_g = self.group_attn_query[g](emb_learned + h_emb)  # [B, N, D]
            query_g = self.group_attn_query[g](self_emb + h_emb)
            keys_g = views  # [B, N, 3, D]

            # [B, N, 1, D] x [B, N, D, 3] -> [B, N, 1, 3] -> [B, N, 3]
            attn_scores = torch.matmul(query_g.unsqueeze(2), keys_g.transpose(-1, -2)).squeeze(2)

            # Scaled Dot-Product Attention
            attn_weights = F.softmax(attn_scores / (D_HIDDEN_GCN ** 0.15), dim=-1) # [B, N, 3]

            # Weighted sum: [B, N, 3] * [B, N, 3, D] -> [B, N, D]
            fused_g = (views * attn_weights.unsqueeze(-1)).sum(dim=2)

            # Prediction for this group: [B, N, group_size]
            pred_g = self.group_heads[g](fused_g)  # [B, N, 8]
            group_preds.append(pred_g)

        # Concatenate all group predictions: [B, N, H]
        preds = torch.cat(group_preds, dim=-1)  # [B, N, 24]

        return preds.permute(0, 2, 1)  # [B, H, N]

# --- Training & Eval ---

class WindSpaceTimeDataset(Dataset):
    def __init__(self, X, Y):
        # X is already in [B, 4, T, N] format
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def weighted_rmse_loss(preds, targets, horizon):
    weights = torch.arange(1, horizon + 1, device=preds.device).float()
    weights = weights / weights.sum()

    mse = (preds - targets) ** 2
    mse = mse.mean(dim=(0, 2))

    weighted_loss = (mse * weights).sum()
    return weighted_loss

def weighted_mae_loss(preds, targets, horizon):
    weights = torch.sqrt(torch.arange(1, horizon + 1, device=preds.device).float())
    weights = weights / weights.sum()

    mae = torch.abs(preds - targets)
    mae = mae.mean(dim=(0, 2))

    weighted_loss = (mae * weights).sum()
    return weighted_loss

def huber_loss(preds, targets, horizon, delta=1.0):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    mae = torch.mean(abs_diff)
    rmse = torch.sqrt(torch.mean(diff**2))
    loss = mae + rmse

    return loss

def evaluate_metrics(model, loader, min_t, diff_t, horizon_size=HORIZON_SIZE, device=DEVICE):
    model.eval()
    total_mae = torch.zeros(horizon_size).to(device)
    total_rmse = torch.zeros(horizon_size).to(device)
    count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)

            min_b = min_t.view(1, 1, -1)
            diff_b = diff_t.view(1, 1, -1)

            preds_denorm = preds * diff_b + min_b
            targets_denorm = y_batch * diff_b + min_b

            # MAE
            mae = torch.abs(preds_denorm - targets_denorm).mean(dim=2).sum(dim=0)
            total_mae += mae

            # RMSE
            mse = ((preds_denorm - targets_denorm) ** 2).mean(dim=2).sum(dim=0)
            total_rmse += mse

            count += x_batch.size(0)

    avg_mae = total_mae / count
    avg_rmse = torch.sqrt(total_rmse / count)
    return avg_mae, avg_rmse

def calculate_matrix_similarity(model, loader, device=DEVICE, epsilon=1e-4):
    """
    For every window (sample) in the loader, computes the dynamic CCF matrix and
    calculates its Pearson correlation and Jaccard similarity against the static CCF
    and learned adjacency matrices.

    Returns averages over all windows:
        pearson_dyn_static, pearson_dyn_learned,
        jaccard_dyn_static, jaccard_dyn_learned
    """
    model.eval()

    adj_learned = model.graph_learner().detach()        # [N, N]
    adj_static  = model.adj_static_ccf.detach()         # [N, N] or [1, N, N]
    if adj_static.dim() == 3:
        adj_static = adj_static.squeeze(0)              # -> [N, N]

    # Flatten static references once
    learned_flat = adj_learned.view(-1).cpu()           # [N*N]
    static_flat  = adj_static.view(-1).cpu()            # [N*N]

    # Binary masks (for Jaccard)
    learned_bin = (learned_flat > epsilon)
    static_bin  = (static_flat  > epsilon)

    pearson_ds_list, pearson_dl_list = [], []
    jaccard_ds_list, jaccard_dl_list = [], []

    def _pearson(a, b):
        """Pearson correlation between two 1-D tensors."""
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.norm() * b.norm())
        if denom < 1e-12:
            return 0.0
        return (a * b).sum().item() / denom.item()

    def _jaccard(bin_a, bin_b):
        intersection = (bin_a & bin_b).float().sum().item()
        union        = (bin_a | bin_b).float().sum().item()
        return intersection / union if union > 0 else 0.0

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            wind = x_batch[:, 0, :, :]                          # [B, T, N]
            adj_dyn = compute_dynamic_ccf_adj(wind).cpu()        # [B, N, N]

            for b in range(adj_dyn.size(0)):
                dyn_flat = adj_dyn[b].view(-1)                   # [N*N]
                dyn_bin  = (dyn_flat > epsilon)

                pearson_ds_list.append(_pearson(dyn_flat, static_flat))
                pearson_dl_list.append(_pearson(dyn_flat, learned_flat))
                jaccard_ds_list.append(_jaccard(dyn_bin,  static_bin))
                jaccard_dl_list.append(_jaccard(dyn_bin,  learned_bin))

    avg_p_ds = sum(pearson_ds_list) / len(pearson_ds_list)
    avg_p_dl = sum(pearson_dl_list) / len(pearson_dl_list)
    avg_j_ds = sum(jaccard_ds_list) / len(jaccard_ds_list)
    avg_j_dl = sum(jaccard_dl_list) / len(jaccard_dl_list)

    return avg_p_ds, avg_p_dl, avg_j_ds, avg_j_dl


def calculate_graph_metrics(model, loader, device=DEVICE):
    """Calculates density and mean weight for all 3 views."""
    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(loader))
        x_batch = x_batch.to(device)

        # Extract wind channel from 5-channel input
        wind = x_batch[:, 0, :, :]  # [B, T, N]

        adj_learned = model.graph_learner()
        adj_dynamic_ccf = compute_dynamic_ccf_adj(wind)  # [B, N, N]
        adj_static_ccf = model.adj_static_ccf # [N, N]

        epsilon = 1e-4

        # View 1 (Learned) - static graph, same for all samples
        dense_learned = (adj_learned > epsilon).float().mean().item()
        weight_learned = adj_learned.mean().item()

        # View 2 (Dynamic CCF) - dynamic graph, compute per-sample density then average
        dense_dynamic_ccf_per_sample = (adj_dynamic_ccf > epsilon).float().mean(dim=(1, 2))  # [B]
        dense_dynamic_ccf = dense_dynamic_ccf_per_sample.mean().item()
        weight_dynamic_ccf = adj_dynamic_ccf.mean().item()

        # View 3 (Static CCF)
        dense_static_ccf = (adj_static_ccf > epsilon).float().mean().item()
        weight_static_ccf = adj_static_ccf.mean().item()

    return dense_learned, weight_learned, dense_dynamic_ccf, weight_dynamic_ccf, dense_static_ccf, weight_static_ccf

def calculate_attention_weights(model, loader, device=DEVICE):
    """Calculates average attention weights on a batch."""
    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(loader))
        x_batch = x_batch.to(device)

        # Forward pass components to get embeddings
        B, C, T, N = x_batch.shape
        start_t = T - model.model_window
        x_features = x_batch[:, :, start_t:, :]
        feature_embeddings = model.feature_encoder(x_features)

        self_w = F.softmax(model.self_weights, dim=1)
        influence_w = F.softmax(model.influence_weights, dim=1)
        self_w_expanded = self_w.unsqueeze(0).unsqueeze(-1)
        influence_w_expanded = influence_w.unsqueeze(0).unsqueeze(-1)

        self_emb = (feature_embeddings * self_w_expanded).sum(dim=2)
        influence_emb = (feature_embeddings * influence_w_expanded).sum(dim=2)

        adj_learned = model.graph_learner()
        z1_learned = model.gnn_learned_1(self_emb, influence_emb, adj_learned)

        emb_learned = z1_learned
        if model.num_gnn_layers == 2:
            x_residual = model.residual_proj(self_emb)
            z2_learned = model.gnn_learned_2(z1_learned, z1_learned, adj_learned)
            emb_learned = z2_learned + x_residual

        wind = x_batch[:, 0, :, :]
        adj_dynamic = compute_dynamic_ccf_adj(wind)
        z1_dynamic = model.gnn_dynamic_ccf_1(self_emb, influence_emb, adj_dynamic)
        emb_dynamic = z1_dynamic
        if model.num_gnn_layers == 2:
             z2_dynamic = model.gnn_dynamic_ccf_2(z1_dynamic, z1_dynamic, adj_dynamic)
             emb_dynamic = z2_dynamic + x_residual

        z1_static = model.gnn_static_ccf_1(self_emb, influence_emb, model.adj_static_ccf)
        emb_static = z1_static
        if model.num_gnn_layers == 2:
             z2_static = model.gnn_static_ccf_2(z1_static, z1_static, model.adj_static_ccf)
             emb_static = z2_static + x_residual

        views = torch.stack([emb_learned, emb_dynamic, emb_static], dim=2)

        # Calculate attention
        total_weights = torch.zeros(3, device=device)

        for g in range(model.num_horizon_groups):
            h_emb = model.horizon_emb(torch.tensor(g, device=device))
            query_g = model.group_attn_query[g](emb_learned + h_emb)
            keys_g = views
            attn_scores = torch.matmul(query_g.unsqueeze(2), keys_g.transpose(-1, -2)).squeeze(2)
            attn_weights = F.softmax(attn_scores / (D_HIDDEN_GCN ** 0.15), dim=-1)

            # Average over Batch and Nodes
            total_weights += attn_weights.mean(dim=(0, 1))

        avg_weights = total_weights / model.num_horizon_groups

    return avg_weights[0].item(), avg_weights[1].item(), avg_weights[2].item()  # (learned, dynamic, static)

def train_model(num_gnn_layers=NUM_GNN_LAYERS, gnn_type=GNN_TYPE, save_dir=None, run_idx=0):
    clean_dfs, stations = load_and_clean_data(DATA_DIR)
    num_stations = len(stations)
    print(f"Number of stations: {num_stations}")

    norm_dfs, stats = normalize_data(clean_dfs)
    X, Y = create_space_time_windows(norm_dfs, WINDOW_SIZE, HORIZON_SIZE, STEP_SIZE)
    print(f"Data shapes: X={X.shape}, Y={Y.shape}")
    print(f"  X format: [Batch, Channels=5, Time, Stations]")

    total_samples = len(X)

    if np.isnan(X).any():
        print("CRITICAL WARNING: X contains NaNs after data processing!")
    if np.isnan(Y).any():
        print("CRITICAL WARNING: Y contains NaNs after data processing!")

    train_end = int(total_samples * TRAIN_SPLIT)
    val_end = int(total_samples * (TRAIN_SPLIT + VAL_SPLIT))

    train_dataset = WindSpaceTimeDataset(X[:train_end], Y[:train_end])
    val_dataset = WindSpaceTimeDataset(X[train_end:val_end], Y[train_end:val_end])
    test_dataset = WindSpaceTimeDataset(X[val_end:], Y[val_end:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Compute Static CCF on Training Data ---
    print("Computing static CCF on full training data...")
    # Use the normalized wind DataFrame directly, sliced training part
    # norm_dfs['wind'] is [Total_Time, N]

    # We need to be careful with timestamps. train_dataset uses indices from 0 to train_end.
    # The X array construction uses a rolling window.
    # The actual time range covered by X[:train_end] is roughly the first (train_end * step_size + window_size) steps.
    # Let's just take the first Split% of the raw data to be safe and consistent with "training data".
    train_split_idx = int(len(norm_dfs['wind']) * TRAIN_SPLIT)
    wind_train_df = norm_dfs['wind'].iloc[:train_split_idx]

    # Convert to tensor: [1, T, N]
    # Keep on CPU to save GPU memory for FFT
    wind_train_tensor = torch.tensor(wind_train_df.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        adj_static = compute_static_ccf_adj(wind_train_tensor) # [1, N, N] -> [N, N] already squeezed in function
        # adj_static is [N, N]
    print(f"Static CCF computed on CPU. Density: {(adj_static > 1e-4).float().mean().item():.4f}")

    # Stats for denormalization (wind speed only)
    min_vals = np.zeros(num_stations)
    diff_vals = np.zeros(num_stations)
    for i, station in enumerate(stations):
        min_vals[i] = stats['wind']['min'][station]
        diff_vals[i] = stats['wind']['diff'][station]
    min_t = torch.tensor(min_vals, device=DEVICE).float()
    diff_t = torch.tensor(diff_vals, device=DEVICE).float()

    # Model
    print(f"Creating model with GNN type: {gnn_type}, Layers: {num_gnn_layers}...")
    # Create model on CPU first
    model = GraphWindModel(num_stations, adj_static=adj_static.cpu(), num_gnn_layers=num_gnn_layers, gnn_type=gnn_type)
    print("Model created successfully!")

    # Weight Initialization (on CPU for faster execution)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    print("Applying weight initialization...")
    model.apply(init_weights)
    print("Weight initialization done!")

    # Move model to GPU after initialization
    print(f"Moving model to {DEVICE}...")
    model = model.to(DEVICE)
    print("Model ready on device!")

    print("Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=4)
    print("Optimizer ready!")

    print("Starting training...")
    print(f"Architecture: LSTM (5 channels) -> {gnn_type} x{num_gnn_layers} (Triple View) -> Attention Fusion -> Prediction")
    if num_gnn_layers == 2:
        print("  Note: Using residual connection for 2-layer GNN")

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    early_stop_patience = 11
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = huber_loss(preds, y_batch, HORIZON_SIZE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(x_batch)
                val_loss += huber_loss(preds, y_batch, HORIZON_SIZE).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train: {avg_train_loss:.4f} - Val: {avg_val_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            avg_mae, avg_rmse = evaluate_metrics(model, test_loader, min_t, diff_t)
            print(f"Epoch {epoch+1} - Test MAE: {avg_mae.mean().item():.4f}, Test RMSE: {avg_rmse.mean().item():.4f}")

            d_l, w_l, d_d, w_d, d_s, w_s = calculate_graph_metrics(model, val_loader)
            print(f"  Graph Metrics: Learned(D={d_l:.2f}, M={w_l:.4f}) | DynamicCCF(D={d_d:.2f}, M={w_d:.4f}) | StaticCCF(D={d_s:.2f}, M={w_s:.4f})")

            attn_l, attn_d, attn_s = calculate_attention_weights(model, val_loader)
            print(f"  View Attention: Learned={attn_l:.4f} | DynamicCCF={attn_d:.4f} | StaticCCF={attn_s:.4f}")

            p_ds, p_dl, j_ds, j_dl = calculate_matrix_similarity(model, val_loader)
            print(f"  Matrix Similarity (Dynamic vs Static):  Pearson={p_ds:.4f} | Jaccard={j_ds:.4f}")
            print(f"  Matrix Similarity (Dynamic vs Learned): Pearson={p_dl:.4f} | Jaccard={j_dl:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        print(f"Loading best model from epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_state)

    # Save model if directory provided
    if save_dir:
        model_path = os.path.join(save_dir, f"model_seed_{run_idx}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("Evaluating...")
    avg_mae, avg_rmse = evaluate_metrics(model, test_loader, min_t, diff_t)

    print("\nAverage MAE per Horizon:")
    for h in range(HORIZON_SIZE):
        print(f"Horizon {h+1}: {avg_mae[h].item():.4f}")

    print("\nAverage RMSE per Horizon:")
    for h in range(HORIZON_SIZE):
        print(f"Horizon {h+1}: {avg_rmse[h].item():.4f}")

    overall_mae = avg_mae.mean().item()
    overall_rmse = avg_rmse.mean().item()
    print(f"\nOverall Average MAE: {overall_mae:.4f}")
    print(f"Overall Average RMSE: {overall_rmse:.4f}")

    # Save results to text file
    if save_dir:
        results_path = os.path.join(save_dir, f"results_seed_{run_idx}.txt")
        with open(results_path, 'w') as f:
            f.write(f"Run {run_idx} Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Overall MAE: {overall_mae:.4f}\n")
            f.write(f"Overall RMSE: {overall_rmse:.4f}\n\n")
            f.write("MAE per Horizon:\n")
            for h in range(HORIZON_SIZE):
                f.write(f"Horizon {h+1}: {avg_mae[h].item():.4f}\n")
            f.write("\nRMSE per Horizon:\n")
            for h in range(HORIZON_SIZE):
                f.write(f"Horizon {h+1}: {avg_rmse[h].item():.4f}\n")
        print(f"Results saved to {results_path}")

    return overall_mae, overall_rmse

def run_n_times(n_runs=5, num_gnn_layers=NUM_GNN_LAYERS, gnn_type=GNN_TYPE):
    """
    Modeli N kez çalıştırır ve ortalama test MAE ile standart sapma hesaplar.

    Args:
        n_runs: Kaç kez çalıştırılacağı (default: 5)
        num_gnn_layers: GNN layer sayısı (1 veya 2)
        gnn_type: GNN tipi ('GCN', 'GAT', 'GraphSAGE')
    """

    # Create timestamped directory for this execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("./journal-results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running model {n_runs} times to calculate statistics...")
    print(f"GNN Type: {gnn_type}, Layers: {num_gnn_layers}")
    print(f"Results will be saved to: {save_dir}")
    print(f"{'='*60}\n")

    mae_results = []
    rmse_results = []

    for run_idx in range(n_runs):
        print(f"\n{'#'*60}")
        print(f"RUN {run_idx + 1}/{n_runs}")
        print(f"{'#'*60}\n")

        mae, rmse = train_model(num_gnn_layers=num_gnn_layers, gnn_type=gnn_type, save_dir=save_dir, run_idx=run_idx+1)
        mae_results.append(mae)
        rmse_results.append(rmse)

        print(f"\n>>> Run {run_idx + 1} completed. MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # İstatistikleri hesapla
    mae_array = np.array(mae_results)
    mean_mae = np.mean(mae_array)
    std_mae = np.std(mae_array)
    min_mae = np.min(mae_array)
    max_mae = np.max(mae_array)

    rmse_array = np.array(rmse_results)
    mean_rmse = np.mean(rmse_array)
    std_rmse = np.std(rmse_array)
    min_rmse = np.min(rmse_array)
    max_rmse = np.max(rmse_array)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS AFTER {n_runs} RUNS")
    print(f"{'='*60}")
    print(f"\nMAE Stats:")
    print(f"Individual MAE values: {[f'{m:.4f}' for m in mae_results]}")
    print(f"Mean MAE:   {mean_mae:.4f}")
    print(f"Std MAE:    {std_mae:.4f}")
    print(f"Min MAE:    {min_mae:.4f}")
    print(f"Max MAE:    {max_mae:.4f}")

    print(f"\nRMSE Stats:")
    print(f"Individual RMSE values: {[f'{m:.4f}' for m in rmse_results]}")
    print(f"Mean RMSE:   {mean_rmse:.4f}")
    print(f"Std RMSE:    {std_rmse:.4f}")
    print(f"Min RMSE:    {min_rmse:.4f}")
    print(f"Max RMSE:    {max_rmse:.4f}")
    print(f"{'='*60}\n")

    # Save summary stats
    summary_path = os.path.join(save_dir, "summary_stats.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Summary Statistics for {n_runs} runs\n")
        f.write(f"GNN Type: {gnn_type}, Layers: {num_gnn_layers}\n")
        f.write("=" * 40 + "\n\n")

        f.write("MAE Statistics:\n")
        f.write(f"Mean: {mean_mae:.4f}\n")
        f.write(f"Std:  {std_mae:.4f}\n")
        f.write(f"Min:  {min_mae:.4f}\n")
        f.write(f"Max:  {max_mae:.4f}\n")
        f.write(f"Values: {mae_results}\n\n")

        f.write("RMSE Statistics:\n")
        f.write(f"Mean: {mean_rmse:.4f}\n")
        f.write(f"Std:  {std_rmse:.4f}\n")
        f.write(f"Min:  {min_rmse:.4f}\n")
        f.write(f"Max:  {max_rmse:.4f}\n")
        f.write(f"Values: {rmse_results}\n")

    print(f"Summary statistics saved to {summary_path}")

    return mean_mae, std_mae, mae_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train GraphWindModel')
    parser.add_argument('-n', '--n_runs', type=int, default=1,
                        help='Number of times to run the model (default: 1)')
    parser.add_argument('--gnn_type', type=str, default=GNN_TYPE, choices=['GCN', 'GAT', 'GraphSAGE'],
                        help='GNN layer type: GCN, GAT, or GraphSAGE (default: GCN)')
    parser.add_argument('--num_gnn_layers', type=int, default=NUM_GNN_LAYERS, choices=[1, 2],
                        help='Number of GNN layers: 1 or 2 (default: 2, with residual connection)')
    args = parser.parse_args()

    if args.n_runs == 1:
        # Just use run_n_times with n=1 to get directory creation and saving logic for free
        run_n_times(1, num_gnn_layers=args.num_gnn_layers, gnn_type=args.gnn_type)
    else:
        run_n_times(args.n_runs, num_gnn_layers=args.num_gnn_layers, gnn_type=args.gnn_type)
