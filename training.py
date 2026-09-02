import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import LACGNNConfig
from .data import load_clean_data, make_windows, normalize_data, split_loaders
from .model import LACGNN, static_ccf_scores


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def composite_loss(prediction, target):
    error = prediction - target
    return error.abs().mean() + error.square().mean().sqrt()


def initialize_parameters(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)


def evaluate(model, loader, wind_minimum, wind_span, device):
    absolute = torch.zeros(model.config.output_length, device=device)
    squared = torch.zeros_like(absolute)
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            prediction = model(inputs)
            scale = wind_span.view(1, 1, -1)
            offset = wind_minimum.view(1, 1, -1)
            error = (prediction * scale + offset) - (targets * scale + offset)
            absolute += error.abs().mean(dim=2).sum(dim=0)
            squared += error.square().mean(dim=2).sum(dim=0)
            count += len(inputs)
    return absolute / count, torch.sqrt(squared / count)


def _write_log(handle, message=""):
    print(message)
    if handle is not None:
        handle.write(f"{message}\n")
        handle.flush()


def prepare_experiment(data_dir, config: LACGNNConfig):
    clean, stations = load_clean_data(data_dir)
    normalized, statistics = normalize_data(clean, config.train_fraction)
    inputs, targets = make_windows(normalized, config)
    train_rows = int(len(normalized["wind"]) * config.train_fraction)
    training_wind = torch.tensor(
        normalized["wind"].iloc[:train_rows].to_numpy(),
        dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        static = static_ccf_scores(training_wind, config.max_lag)
    minimum = torch.tensor(
        statistics["wind"]["min"].to_numpy(), dtype=torch.float32)
    span = torch.tensor(
        statistics["wind"]["span"].to_numpy(), dtype=torch.float32)
    return inputs, targets, stations, static, minimum, span


def train_once(data, config: LACGNNConfig, seed: int, device,
               checkpoint_path: str | Path | None = None, log_handle=None,
               run_index: int = 1, total_runs: int = 1,
               model_class=LACGNN):
    set_seed(seed)
    _write_log(log_handle, f"**------ Seed {run_index}/{total_runs} ------**")
    inputs, targets, stations, static, minimum, span = data
    train_loader, validation_loader, test_loader = split_loaders(
        inputs, targets, config, seed)
    model = model_class(len(stations), static, config)
    model.apply(initialize_parameters)
    model.to(device)
    minimum, span = minimum.to(device), span.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor,
        patience=config.scheduler_patience)
    best_loss, best_state, best_epoch = float("inf"), None, None
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        training_loss = 0.0
        training_count = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = composite_loss(model(batch_inputs), batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_loss += loss.item() * len(batch_inputs)
            training_count += len(batch_inputs)

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        validation_absolute = 0.0
        validation_squared = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in validation_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                prediction = model(batch_inputs)
                batch_loss = composite_loss(prediction, batch_targets)
                validation_loss += batch_loss.item() * len(batch_inputs)
                scale = span.view(1, 1, -1)
                offset = minimum.view(1, 1, -1)
                error = (
                    (prediction * scale + offset)
                    - (batch_targets * scale + offset)
                )
                validation_absolute += error.abs().mean().item() * len(batch_inputs)
                validation_squared += error.square().mean().item() * len(batch_inputs)
                validation_count += len(batch_inputs)
        training_loss /= training_count
        validation_loss /= validation_count
        validation_mae = validation_absolute / validation_count
        validation_rmse = (validation_squared / validation_count) ** 0.5
        _write_log(
            log_handle,
            f"[{epoch}/{config.max_epochs}] Train loss: {training_loss:.6f}, "
            f"Val composite loss: {validation_loss:.6f}, "
            f"Val MAE: {validation_mae:.6f}, Val RMSE: {validation_rmse:.6f}",
        )
        scheduler.step(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
    mae, rmse = evaluate(model, test_loader, minimum, span, device)
    result = {
        "seed": seed,
        "selected_epoch": best_epoch,
        "mae": mae.mean().item(),
        "rmse": rmse.mean().item(),
        "mae_by_horizon": mae.cpu().tolist(),
        "rmse_by_horizon": rmse.cpu().tolist(),
    }
    _write_log(log_handle, f"Selected epoch: {result['selected_epoch']}")
    _write_log(log_handle, f"Test MAE: {result['mae']:.6f}")
    _write_log(log_handle, f"Test RMSE: {result['rmse']:.6f}")
    _write_log(log_handle)
    return result


def train_many(data_dir, output_dir, config: LACGNNConfig, seeds, device,
               log_path: str | Path | None = None, model_class=LACGNN):
    data = prepare_experiment(data_dir, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    results = []
    log_handle = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    try:
        data_directory = Path(data_dir).name
        dataset = {
            "hourly_data": "USA",
            "ceda_data": "UK",
            "ireland_hourly_data": "IRELAND",
        }.get(data_directory, data_directory.upper())
        _write_log(log_handle, f"Dataset: {dataset}")
        _write_log(log_handle, f"Runs: {len(seeds)} seeds")
        _write_log(
            log_handle,
            "Checkpoint selection: lowest validation composite loss")
        _write_log(log_handle)
        for run_index, seed in enumerate(seeds, start=1):
            results.append(train_once(
                data, config, seed, device,
                output_dir / f"model_seed_{seed}.pt", log_handle,
                run_index, len(seeds), model_class))

        _write_log(log_handle, "**------ Final Results ------**")
        for item in results:
            _write_log(
                log_handle,
                f"Seed {item['seed']}: Test MAE: {item['mae']:.6f}, "
                f"Test RMSE: {item['rmse']:.6f}",
            )
        _write_log(
            log_handle,
            f"Overall Test MAE: {np.mean([item['mae'] for item in results]):.6f}",
        )
        _write_log(
            log_handle,
            f"Overall Test RMSE: {np.mean([item['rmse'] for item in results]):.6f}",
        )
    finally:
        if log_handle is not None:
            log_handle.close()
    summary = pd.DataFrame([
        {"seed": item["seed"], "mae": item["mae"], "rmse": item["rmse"]}
        for item in results
    ])
    summary.to_csv(output_dir / "summary.csv", index=False)
    return results
