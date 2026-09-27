# 36120-26SP-AT2-26294153-experiments

Experimentation repository for 36120 Advanced Machine Learning, Assessment Task 2:
Machine Learning as a Service.

Student: Ratnadeep Patra

Student ID: 26294153

## Project Overview

This repository contains the modelling work for two Sydney weather intelligence
services built from Open-Meteo historical weather observations:

- Climate Comfort Index (CCI): regression target for 1, 2 and 3 days ahead.
- Weather Hazard Category (WHC): multiclass classification target exactly 7 days ahead.

The assignment brief requires an experimentation repository and a separate FastAPI
deployment repository. This repository is the experimentation side: data acquisition,
target creation, feature engineering, model development, evaluation, saved model
artifacts and metadata prepared for serving.

## Business Goal

The models support weather-aware planning for Sydney operations. The CCI model helps
estimate near-term outdoor comfort, while the WHC model provides early warning of
potentially hazardous weather. The modelling emphasis is not only predictive accuracy,
but also reproducibility, leakage-safe forecasting, explainability and deployment
readiness.

## Data

Raw observations are sourced from the Open-Meteo Historical Weather API for:

- Location: Sydney, NSW, Australia
- Latitude: -33.8688
- Longitude: 151.2093
- Historical training/evaluation window: data before 2026

Local cached and processed data lives under:

```text
data/raw/
data/raw/open_meteo_cache/
data/processed/comfort_climate/
data/processed/weather_hazard/
```

The assignment treats 2026 onwards as production data, so the notebooks avoid using it
for training, validation or test evaluation.

## Targets

### Climate Comfort Index

CCI is a continuous regression target derived from weather comfort components:
temperature, relative humidity, wind speed, cloud cover and precipitation. The final
forecast targets are:

- `cci_target_1d`
- `cci_target_2d`
- `cci_target_3d`

### Weather Hazard Category

WHC is a multiclass target derived through a Weather Hazard Index using precipitation,
wind gusts, cloud cover, snowfall and temperature. The classes are:

- `0`: Low Risk
- `1`: Moderate Risk
- `2`: High Risk
- `3`: Extreme Risk

The final forecast target is `whc_target_7d`.

## Repository Structure

```text
.
|-- adv_ml_at2/                         # Reusable package code
|   |-- dataset.py                      # Open-Meteo retrieval and local caching
|   |-- features.py                     # Target creation and feature engineering
|   `-- modeling/custom_models.py       # Custom weighting and SMOTE wrappers
|-- data/
|   |-- raw/                            # Raw and cached Open-Meteo data
|   `-- processed/                      # Train/validation/test matrices
|-- models/
|   |-- comfort_climate/                # CCI experiment artifacts
|   |-- weather_hazard/                 # WHC experiment artifacts
|   `-- model_serving/                  # Serving bundle metadata and data fetcher
|-- notebooks/
|   |-- comfort_climate/                # CCI experiments 1-3
|   `-- weather_hazard/                 # WHC experiments 1-3
|-- reports/                            # Report outputs, if generated
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Setup

This project uses Python 3.12 and `uv`.

```bash
uv sync
```

Most commands can be run without manually activating the virtual environment by using
`uv run`.

Useful commands:

```bash
uv run ruff format --check
uv run ruff check
uv run ruff check --fix
uv run ruff format
uv run jupyter notebook
```

To rebuild the raw weather cache and base datasets:

```bash
uv run python adv_ml_at2/dataset.py
```

## Running The Experiments

For a full clean reproduction, run the classification notebooks first, then the
regression notebooks. The final comfort-climate Experiment 3 notebook also builds the
shared serving helper and metadata, and it loads the saved Weather Hazard
classification artifact from `models/weather_hazard/experiment_2/experiment_2.pkl`.

Recommended execution order:

```text
notebooks/weather_hazard/36120_26SP_AT2_26294153_experiment_1.ipynb
notebooks/weather_hazard/36120_26SP_AT2_26294153_experiment_2.ipynb
notebooks/weather_hazard/36120_26SP_AT2_26294153_experiment_3.ipynb

notebooks/comfort_climate/36120_26SP_AT2_26294153_experiment_1.ipynb
notebooks/comfort_climate/36120_26SP_AT2_26294153_experiment_2.ipynb
notebooks/comfort_climate/36120_26SP_AT2_26294153_experiment_3.ipynb
```

Open the notebooks with:

```bash
uv run jupyter notebook
```

To execute a notebook non-interactively, use the same order and run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace <notebook_path>
```

Each notebook saves its processed matrices under `data/processed/<target>/experiment_n/`
and final workflow artifacts under `models/<target>/experiment_n/`.

## Experiment Summary

### Comfort Climate

- Experiment 1: linear and regularised regression baseline using engineered temporal,
  lag, rolling and seasonal predictors.
- Experiment 2: weighted Elastic Net models to improve behaviour on uncommon or more
  difficult CCI regions.
- Experiment 3: tree-based models, including XGBoost alternatives, to test whether
  nonlinear learners improve CCI forecasts.

Selected serving model: Experiment 2, weighted Elastic Net.

Held-out test metrics from `models/model_serving/model_metadata.json`:

| Target | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| `cci_target_1d` | 9.4873 | 7.1589 | 0.3063 |
| `cci_target_2d` | 9.8465 | 7.4537 | 0.2533 |
| `cci_target_3d` | 10.0501 | 7.6371 | 0.2220 |

### Weather Hazard

- Experiment 1: Elastic-Net Logistic Regression baseline with chronological splits.
- Experiment 2: proportion-aware SMOTE with Elastic-Net Logistic Regression to improve
  minority hazardous-event detection.
- Experiment 3: SMOTE with tree-based models, including Random Forest and XGBoost,
  to test nonlinear classification boundaries.

Selected serving model: Experiment 2, Elastic-Net Logistic Regression with
proportion-aware SMOTE.

Held-out test metrics from `models/model_serving/model_metadata.json`:

| Metric | Value |
| --- | ---: |
| Severity-weighted recall | 0.2077 |
| Macro F1 | 0.2397 |
| High-risk recall | 0.1351 |
| High-risk alert precision | 0.0278 |

The WHC model improves hazardous-event sensitivity compared with a simple baseline, but
the low alert precision means predictions should be treated as decision support rather
than an automated warning system.

## Saved Artifacts

Final experiment artifacts:

```text
models/comfort_climate/experiment_1/experiment_1.pkl
models/comfort_climate/experiment_2/experiment_2.pkl
models/comfort_climate/experiment_3/experiment_3.pkl
models/weather_hazard/experiment_1/experiment_1.pkl
models/weather_hazard/experiment_2/experiment_2.pkl
models/weather_hazard/experiment_3/experiment_3.pkl
```

Serving support files:

```text
models/model_serving/data_fetch.pkl
models/model_serving/model_metadata.json
```

`model_metadata.json` documents deployed-model targets, algorithms, forecast horizons,
features, feature importance and performance metrics for the API repository.

## Deployment Handoff

The assignment brief requires a separate FastAPI deployment repository with these
endpoints:

```text
GET /
GET /health
GET /predict/index/comfort_climate
GET /predict/category/weather_hazard
GET /model-metadata
```

This experimentation repository provides the trained artifacts and metadata needed by
that deployment repository. The deployment README should document the Render URL,
Docker/FastAPI startup steps, endpoint examples and error handling behaviour.

## Custom Package

Reusable project functionality is kept in the `adv_ml_at2` package. The notebooks use
this package for Open-Meteo data retrieval, target creation, feature engineering and
custom modelling helpers. If the package is changed for deployment, the assignment
brief requires the updated package to be published to TestPyPI.

## Known Limitations

- CCI forecasts retain useful signal but still compress the range of more extreme
  comfort conditions.
- WHC classes are imbalanced. High and Extreme Risk events remain difficult to detect
  reliably.
- The WHC model has low high-risk alert precision, so false alerts are expected.
- Production use should include monitoring, retraining and validation on post-2025
  data once it becomes available for operational assessment.
