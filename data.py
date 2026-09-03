from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import LACGNNConfig


FEATURE_FILES = {
    "wind": "wind_speed.csv",
    "pressure": "pressure.csv",
    "temperature": "temperature.csv",
    "humidity": "humidity.csv",
    "direction": "wind_direction.csv",
}

USA_STATIONS = (
    "Portland", "Seattle", "Los Angeles", "San Diego", "Las Vegas",
    "Phoenix", "Albuquerque", "Denver", "San Antonio", "Dallas",
    "Houston", "Kansas City", "Minneapolis", "Saint Louis", "Chicago",
    "Nashville", "Indianapolis", "Atlanta", "Detroit", "Jacksonville",
    "Charlotte", "Pittsburgh", "Toronto", "Philadelphia", "Montreal",
    "Boston",
)


class WindDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.from_numpy(inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def load_clean_data(data_dir: str | Path):
    data_dir = Path(data_dir)
    frames = {}
    for feature, filename in FEATURE_FILES.items():
        frame = pd.read_csv(data_dir / filename).iloc[1:].copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frames[feature] = frame.set_index("datetime")

    common_stations = set.intersection(
        *(set(frame.columns) for frame in frames.values()))
    if data_dir.name == "hourly_data":
        stations = [name for name in USA_STATIONS if name in common_stations]
    else:
        excluded = set()
        for frame in frames.values():
            missing = frame.isna().mean()
            excluded.update(missing[missing > 0.30].index)
        stations = sorted(common_stations - excluded)

    common_index = frames["wind"].index
    for frame in frames.values():
        common_index = common_index.intersection(frame.index)
    common_index = common_index.sort_values()

    cleaned = {}
    for feature, frame in frames.items():
        aligned = frame.loc[common_index, stations]
        cleaned[feature] = aligned.interpolate(
            method="linear", limit_direction="both")
    return cleaned, stations


def normalize_data(frames, train_fraction: float):
    """Fit per-station min-max statistics on the chronological training part."""
    train_end = int(len(frames["wind"]) * train_fraction)
    normalized, statistics = {}, {}
    for feature, frame in frames.items():
        training = frame.iloc[:train_end]
        minimum = training.min()
        span = (training.max() - minimum).replace(0.0, 1.0)
        normalized[feature] = (frame - minimum) / span
        statistics[feature] = {"min": minimum, "span": span}

    radians = np.deg2rad(frames["direction"])
    normalized["direction_sin"] = np.sin(radians)
    normalized["direction_cos"] = np.cos(radians)
    return normalized, statistics


def make_windows(frames, config: LACGNNConfig):
    channels = (
        frames["wind"], frames["pressure"], frames["temperature"],
        frames["humidity"], frames["direction_sin"],
        frames["direction_cos"],
    )
    values = np.stack([frame.to_numpy() for frame in channels], axis=-1)
    inputs, targets, target_starts = [], [], []
    final_start = len(values) - config.ccf_window - config.output_length
    for start in range(0, final_start + 1, config.window_stride):
        split = start + config.ccf_window
        inputs.append(values[start:split])
        targets.append(values[split:split + config.output_length, :, 0])
        target_starts.append(split)
    x = np.asarray(inputs, dtype=np.float32).transpose(0, 3, 1, 2)
    y = np.asarray(targets, dtype=np.float32)
    return x, y, np.asarray(target_starts, dtype=np.int64)


def split_loaders(inputs, targets, target_starts, total_rows: int,
                  config: LACGNNConfig, seed: int):
    train_boundary = int(total_rows * config.train_fraction)
    validation_boundary = int(
        total_rows * (config.train_fraction + config.validation_fraction))
    target_ends = target_starts + config.output_length
    masks = (
        target_ends <= train_boundary,
        ((target_starts >= train_boundary)
         & (target_ends <= validation_boundary)),
        ((target_starts >= validation_boundary)
         & (target_ends <= total_rows)),
    )
    split_arrays = [(inputs[mask], targets[mask]) for mask in masks]
    if any(len(split_inputs) == 0 for split_inputs, _ in split_arrays):
        raise ValueError(
            "the configured timeline split leaves an empty dataset partition")

    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        WindDataset(*split_arrays[0]),
        batch_size=config.batch_size, shuffle=True, generator=generator)
    validation = DataLoader(
        WindDataset(*split_arrays[1]),
        batch_size=config.batch_size)
    test = DataLoader(
        WindDataset(*split_arrays[2]),
        batch_size=config.batch_size)
    return train, validation, test
