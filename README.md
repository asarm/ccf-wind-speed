# LACGNN

Official code for the paper **“Lag-Aware Cross-correlation Graph Neural Network for
Wind Speed Forecasting.”**

## Data layout

Run the commands below from the directory that contains the cloned
`ccf-wind-speed` repository. Place the three datasets beside the repository:

```text
workspace/
├── ccf-wind-speed/
└── datasets/
    ├── hourly_data/          # USA
    ├── ceda_data/            # UK
    └── ireland_hourly_data/  # Ireland
```

Each dataset directory must contain `wind_speed.csv`, `pressure.csv`,
`temperature.csv`, `humidity.csv`, and `wind_direction.csv`. Every CSV must
have a `datetime` column and one column per station.

## Run

Install the dependencies from the workspace directory:

```bash
pip install -r ccf-wind-speed/requirements.txt
```

Train the model on one dataset:

```bash
python -m ccf-wind-speed.train \
  --data-dir datasets/hourly_data \
  --output-dir ccf-wind-speed/runs/usa
```

Checkpoints, `summary.csv`, and `train.txt` are written to the output directory.

The default window stride is 1 hour. It can be changed explicitly with
`--window-stride`

To train on another directory with the same CSV schema:

```bash
python -m ccf-wind-speed.train \
  --data-dir path/to/dataset \
  --output-dir ccf-wind-speed/runs/custom
```

## Experiment scripts

`experiment_scripts/structural_ablation.py` evaluates the full model and the
structural variants used in the ablation study. Edit `VARIANTS_TO_RUN`,
`DATASETS_TO_RUN`, and `SEEDS` at the top of the file, then run:

```bash
python -m ccf-wind-speed.experiment_scripts.structural_ablation --device cuda:0
```

`experiment_scripts/param_search.py` runs every combination of the top-k, CCF
window, and maximum-lag lists defined at the top of the file:

```bash
python -m ccf-wind-speed.experiment_scripts.param_search --device cuda:0
```

These scripts expect `datasets/hourly_data`, `datasets/ceda_data`, and
`datasets/ireland_hourly_data` for USA, UK, and Ireland. Use `--dataset uk` to
run one dataset and `--dry-run` to list the jobs without training. Results are
written below `ccf-wind-speed/structural_runs/` or
`ccf-wind-speed/param_search_runs/`, with matching TXT logs below
`ccf-wind-speed/logs/`.
