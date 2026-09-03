"""Run structural ablations reported in Table 3 of the paper."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..config import LACGNNConfig
from ..model import LACGNN, dynamic_ccf_graph
from ..training import train_many


# Edit these lists to choose the complete experiment set.
VARIANTS_TO_RUN = [
    "full",
    "no_dynamic",
    "no_static",
    "no_learned",
    "only_learned",
    "only_dynamic",
    "only_static",
    "no_dual_embedding",
    "no_view_specific_dual_embedding",
]
DATASETS_TO_RUN = ["usa", "uk", "ireland"]
SEEDS = [1, 2, 3, 4, 5]

DATASETS = {
    "usa": "datasets/hourly_data",
    "uk": "datasets/ceda_data",
    "ireland": "datasets/ireland_hourly_data",
}
ACTIVE_VIEWS = {
    "full": ("learned", "dynamic", "static"),
    "no_dynamic": ("learned", "static"),
    "no_static": ("learned", "dynamic"),
    "no_learned": ("dynamic", "static"),
    "only_learned": ("learned",),
    "only_dynamic": ("dynamic",),
    "only_static": ("static",),
    "no_dual_embedding": ("learned", "dynamic", "static"),
    "no_view_specific_dual_embedding": ("learned", "dynamic", "static"),
}


class StructuralAblationLACGNN(LACGNN):
    variant = "full"

    def __init__(self, nodes, static_adjacency, config):
        super().__init__(nodes, static_adjacency, config)
        self.active_views = ACTIVE_VIEWS[self.variant]

        if self.variant == "no_dual_embedding":
            self.self_weights = nn.ParameterDict()
            self.influence_weights = nn.ParameterDict()
            self.global_feature_weights = nn.Parameter(torch.zeros(5))
        elif self.variant == "no_view_specific_dual_embedding":
            initial_self = self.self_weights["learned"].detach().clone()
            initial_influence = self.influence_weights["learned"].detach().clone()
            self.self_weights = nn.ParameterDict()
            self.influence_weights = nn.ParameterDict()
            self.shared_self_weights = nn.Parameter(initial_self)
            self.shared_influence_weights = nn.Parameter(initial_influence)
        else:
            for view in self.VIEWS:
                if view not in self.active_views:
                    del self.self_weights[view]
                    del self.influence_weights[view]

        if "learned" not in self.active_views:
            del self.graph_learner
            del self.learned_gnn
        if "dynamic" not in self.active_views:
            del self.dynamic_gnn
        if "static" not in self.active_views:
            del self.static_gnn
            del self.static_scores

        view_count = len(self.active_views)
        if view_count != len(self.VIEWS):
            self.view_normalization = nn.LayerNorm(
                view_count * config.gnn_hidden_size)
            self.view_scorer = nn.Linear(
                view_count * config.gnn_hidden_size, view_count)

    def _mix(self, embeddings, view, role):
        if self.variant == "no_dual_embedding":
            weights = F.softmax(self.global_feature_weights, dim=0)
            return (embeddings * weights[None, None, :, None]).sum(dim=2)
        if self.variant == "no_view_specific_dual_embedding":
            weights = getattr(self, f"shared_{role}_weights")
        else:
            table = self.self_weights if role == "self" else self.influence_weights
            weights = table[view]
        return self._weighted_embedding(embeddings, weights)

    def encode_views(self, inputs):
        embeddings = self.feature_encoder(
            inputs[:, :, -self.config.input_length:, :])
        mixed = {
            (view, role): self._mix(embeddings, view, role)
            for view in self.active_views
            for role in ("self", "influence")
        }
        outputs = []
        if "learned" in self.active_views:
            adjacency = self.sparsifier(self.graph_learner())
            outputs.append(self.learned_gnn(
                mixed[("learned", "self")],
                mixed[("learned", "influence")], adjacency))
        if "dynamic" in self.active_views:
            profile, adjacency = dynamic_ccf_graph(
                inputs[:, 0], self.config.max_lag, self.sparsifier)
            outputs.append(self.dynamic_gnn(
                mixed[("dynamic", "self")],
                mixed[("dynamic", "influence")], profile, adjacency))
        if "static" in self.active_views:
            adjacency = self.sparsifier(self.static_scores)
            outputs.append(self.static_gnn(
                mixed[("static", "self")],
                mixed[("static", "influence")], adjacency))
        return torch.stack(outputs, dim=2)

    def fuse_views(self, views):
        if views.shape[2] == 1:
            weights = torch.ones(
                *views.shape[:2], 1, device=views.device, dtype=views.dtype)
            return views[:, :, 0], weights
        return super().fuse_views(views)


def model_for_variant(variant):
    return type(
        f"LACGNN_{variant}",
        (StructuralAblationLACGNN,),
        {"variant": variant},
    )


def main():
    parser = argparse.ArgumentParser(description="LACGNN structural ablations")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--variant", choices=ACTIVE_VIEWS, action="append")
    parser.add_argument("--device", default=(
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seeds", type=int)
    parser.add_argument(
        "--output-root", default="ccf-wind-speed/structural_runs")
    parser.add_argument(
        "--log-root", default="ccf-wind-speed/logs/structural")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = args.dataset or DATASETS_TO_RUN
    variants = args.variant or VARIANTS_TO_RUN
    seeds = list(range(1, args.seeds + 1)) if args.seeds else SEEDS
    jobs = [(dataset, variant) for dataset in datasets for variant in variants]
    if args.dry_run:
        for dataset, variant in jobs:
            print(f"{dataset}: {variant}; seeds={seeds}; device={args.device}")
        return

    records = []
    for dataset, variant in jobs:
        output = Path(args.output_root) / dataset / variant
        log = Path(args.log_root) / dataset / f"{variant}.txt"
        results = train_many(
            DATASETS[dataset], output, LACGNNConfig(), seeds,
            torch.device(args.device), log, model_for_variant(variant))
        records.append({
            "dataset": dataset,
            "variant": variant,
            "mae": sum(item["mae"] for item in results) / len(results),
            "rmse": sum(item["rmse"] for item in results) / len(results),
        })
    summary = Path(args.output_root) / "structural_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(summary, index=False)


if __name__ == "__main__":
    main()
