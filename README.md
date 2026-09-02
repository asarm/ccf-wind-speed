# LACGNN

Official code for the paper **“Lag-Aware Cross-correlation Graph Neural Network for
Wind Speed Forecasting.”**

## Run

Install the dependencies:

```bash
pip install -r lacgnn-paper/requirements.txt
```

Place `wind_speed.csv`, `pressure.csv`, `temperature.csv`, `humidity.csv`, and
`wind_direction.csv` in the `datasets/` directory, then run:

```bash
python -m lacgnn-paper.train --output-dir runs
```

To use a different data directory:

```bash
python -m lacgnn-paper.train --data-dir path/to/dataset --output-dir runs
```

## Experiment scripts

`experiment_scripts/structural_ablation.py` evaluates the full model and the
structural variants used in the ablation study. Edit `VARIANTS_TO_RUN`,
`DATASETS_TO_RUN`, and `SEEDS` at the top of the file, then run:

```bash
python -m lacgnn-paper.experiment_scripts.structural_ablation --device cuda:0
```

`experiment_scripts/param_search.py` runs every combination of the top-k, CCF
window, and maximum-lag lists defined at the top of the file:

```bash
python -m lacgnn-paper.experiment_scripts.param_search --device cuda:0
```

These scripts expect `datasets/hourly_data`, `datasets/ceda_data`, and
`datasets/ireland_hourly_data` for USA, UK, and Ireland. Use `--dataset uk` to
run one dataset and `--dry-run` to list the jobs without training. Results are
written to `structural_runs/` or `param_search_runs/`, with matching TXT logs
under `logs/`.
