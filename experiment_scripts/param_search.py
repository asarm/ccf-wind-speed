"""Run a grid search over graph sparsity, CCF window, and maximum lag."""

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import torch

from ..config import LACGNNConfig
from ..training import train_many


# Edit these lists; every valid Cartesian-product combination is executed.
TOP_K_VALUES = [0.15, 0.25, 0.35]
CCF_WINDOW_VALUES = [48, 72, 96]
MAX_LAG_VALUES = [8, 16, 24]
DATASETS_TO_RUN = ["usa", "uk", "ireland"]
SEEDS = [1, 2, 3, 4, 5]

DATASETS = {
    "usa": "datasets/hourly_data",
    "uk": "datasets/ceda_data",
    "ireland": "datasets/ireland_hourly_data",
}


def grid():
    for top_k, ccf_window, max_lag in product(
            TOP_K_VALUES, CCF_WINDOW_VALUES, MAX_LAG_VALUES):
        if max_lag < ccf_window:
            yield top_k, ccf_window, max_lag


def run_name(top_k, ccf_window, max_lag):
    top_k_label = f"{top_k:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"topk_{top_k_label}_ccfw_{ccf_window}_lag_{max_lag}"


def main():
    parser = argparse.ArgumentParser(description="LACGNN parameter grid search")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--device", default=(
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seeds", type=int)
    parser.add_argument(
        "--output-root", default="ccf-wind-speed/param_search_runs")
    parser.add_argument(
        "--log-root", default="ccf-wind-speed/logs/param_search")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = args.dataset or DATASETS_TO_RUN
    seeds = list(range(1, args.seeds + 1)) if args.seeds else SEEDS
    combinations = list(grid())
    if not combinations:
        parser.error("the parameter grid contains no valid combination")
    jobs = [
        (dataset, top_k, ccf_window, max_lag)
        for dataset in datasets
        for top_k, ccf_window, max_lag in combinations
    ]
    if args.dry_run:
        for dataset, top_k, ccf_window, max_lag in jobs:
            print(
                f"{dataset}: top_k={top_k}, ccf_window={ccf_window}, "
                f"max_lag={max_lag}; seeds={seeds}; device={args.device}")
        print(f"Total jobs: {len(jobs)}")
        return

    records = []
    for dataset, top_k, ccf_window, max_lag in jobs:
        name = run_name(top_k, ccf_window, max_lag)
        config = LACGNNConfig(
            top_k_fraction=top_k,
            ccf_window=ccf_window,
            max_lag=max_lag,
        )
        output = Path(args.output_root) / dataset / name
        log = Path(args.log_root) / dataset / f"{name}.txt"
        results = train_many(
            DATASETS[dataset], output, config, seeds,
            torch.device(args.device), log)
        records.append({
            "dataset": dataset,
            "top_k": top_k,
            "ccf_window": ccf_window,
            "max_lag": max_lag,
            "mae": sum(item["mae"] for item in results) / len(results),
            "rmse": sum(item["rmse"] for item in results) / len(results),
        })
    summary = Path(args.output_root) / "grid_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(summary, index=False)


if __name__ == "__main__":
    main()
