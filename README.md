# Official repository for the paper titled "Lag-Aware Cross-correlation Graph Neural Network for Wind Speed Forecasting". 
LACGNN is a multi-view graph neural network for multi-station wind speed forecasting. It combines three directed graphs: a learned adaptive graph, a dynamic cross-correlation graph capturing peak-lag propagation within each input window, and a static cross-correlation graph built from full-lag profiles to model the directional, time-delayed nature of atmospheric influence. A dual embedding mechanism with per-station learnable weights separates each station's representation into self and influence components, allowing meteorological variables to contribute differently to local prediction and spatial message passing.

## Datasets

Three hourly meteorological datasets covering 5 variables (wind speed, pressure, temperature, humidity, wind direction).

| Dataset | Stations | Period | Link |
|---|---|---|---|
| USA | 26 | 10/2012–11/2017 | [Kaggle](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data) |
| UK | 35 | 01/2019–12/2024 | [Met Office MIDAS Open (CEDA)](https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202507) |
| Ireland | 22 | 12/2007–02/2022 | [Kaggle](https://www.kaggle.com/datasets/dariasvasileva/hourly-weather-data-in-ireland-from-24-stations/data) |

## Usage

After downloaded the datasets, update `DATA_DIR` in `wind_model.py` to point to your dataset directory, then run:

```bash
python wind_model.py --n_runs 5
```

`--n_runs` sets the number of random seeds to train with. Results for each seed are written to `journal-results/<timestamp>/`.
