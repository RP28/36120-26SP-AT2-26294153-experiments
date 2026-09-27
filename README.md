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

## FastAPI Artifact Contract

The FastAPI repository should treat the files under `models/` as the inference
contract for this experimentation repository.

The API environment must install this project's runtime dependencies, including
`adv_ml_at2`, `mlweave`, `cloudpickle`, `scikit-learn`, `imbalanced-learn`, `xgboost`
and `lightgbm`, because the pickled workflows reference those classes at load time.

Artifact map:

| File | Serialized Type | API Purpose |
| --- | --- | --- |
| `models/comfort_climate/experiment_1/experiment_1.pkl` | `dict` | Regression experiment archive, not selected for serving. |
| `models/comfort_climate/experiment_2/experiment_2.pkl` | `dict` | Selected CCI serving workflows for 1, 2 and 3 day forecasts. |
| `models/comfort_climate/experiment_3/experiment_3.pkl` | `dict` | Regression experiment archive, not selected for serving. |
| `models/weather_hazard/experiment_1/experiment_1.pkl` | `dict` | Classification experiment archive, not selected for serving. |
| `models/weather_hazard/experiment_2/experiment_2.pkl` | `dict` | Selected WHC serving workflow for the 7 day forecast. |
| `models/weather_hazard/experiment_3/experiment_3.pkl` | `dict` | Classification experiment archive, not selected for serving. |
| `models/model_serving/data_fetch.pkl` | `function` | Shared serving data builder used before workflow preprocessing. |
| `models/model_serving/model_metadata.json` | `list[dict]` | Metadata returned by `/model-metadata`. |

### Experiment Pickle Files

All experiment model files are serialized with `cloudpickle` and contain a top-level
dictionary with this structure:

```python
{
    "ml_workflow": ...,
    "model": None,
}
```

The `ml_workflow` value is the primary inference object:

- Comfort-climate artifacts store a dictionary of workflows keyed by target:
  `cci_target_1d`, `cci_target_2d` and `cci_target_3d`.
- Weather-hazard artifacts store a single workflow for `whc_target_7d`.
- Each workflow contains the fitted preprocessing object, fitted model and fitted
  feature matrix metadata required to align inference columns.

The general `model` value is `None` when there was no intention to overwrite the model
inside the fitted workflow for inference. In that case, the API must use
`artifact["ml_workflow"]` and each workflow's fitted `model_` attribute. If a future
artifact sets `artifact["model"]` to a non-null value, the API may treat it as an
explicit model override for that artifact.

Expected inference pattern:

```python
import cloudpickle

with open("models/comfort_climate/experiment_2/experiment_2.pkl", "rb") as file:
    artifact = cloudpickle.load(file)

workflow = artifact["ml_workflow"]["cci_target_1d"]
transformed = workflow.preprocessing_.transform(serving_df)
X = transformed.loc[:, workflow.X_parts_[0].columns]
prediction = workflow.model_.predict(X)
```

For weather hazard:

```python
with open("models/weather_hazard/experiment_2/experiment_2.pkl", "rb") as file:
    artifact = cloudpickle.load(file)

workflow = artifact["ml_workflow"]
transformed = workflow.preprocessing_.transform(serving_df)
X = transformed.loc[:, workflow.X_parts_[0].columns]
prediction = workflow.model_.predict(X)
```

Selected serving artifacts:

```text
models/comfort_climate/experiment_2/experiment_2.pkl
models/weather_hazard/experiment_2/experiment_2.pkl
```

### `data_fetch.pkl`

`models/model_serving/data_fetch.pkl` is a `cloudpickle`-serialized function:

```python
fetch_dataset_for_serving(start_date: str, end_date: str, classification: bool = False)
```

It calls the Open-Meteo data retrieval helpers, builds daily features, adds the required
lagged feature names, and adds placeholder target columns so the saved preprocessing
pipelines receive the schema they were fitted with.

- Use `classification=False` for CCI regression requests.
- Use `classification=True` for WHC classification requests.
- The returned DataFrame is not the final model matrix; the API still needs to pass it
  through the selected workflow's fitted `preprocessing_` object and then align columns
  with `workflow.X_parts_[0].columns`.
- The function currently calls the downstream Open-Meteo API with cache disabled and
  refresh enabled, so the deployed API should handle network errors and invalid dates
  gracefully.

### `model_metadata.json`

`models/model_serving/model_metadata.json` is a JSON list with one entry per deployed
target family. Each entry contains:

```text
target
prediction_type
algorithm
experiment
forecast_horizon
features
feature_importance_method
feature_importance_scoring
feature_importance
performance_metrics
```

The FastAPI `/model-metadata` endpoint should return this JSON, or a direct API-shaped
wrapper around it, without recalculating feature importance or performance metrics at
request time.

## Known Limitations

- CCI forecasts retain useful signal but still compress the range of more extreme
  comfort conditions.
- WHC classes are imbalanced. High and Extreme Risk events remain difficult to detect
  reliably.
- The WHC model has low high-risk alert precision, so false alerts are expected.
- Production use should include monitoring, retraining and validation on post-2025
  data once it becomes available for operational assessment.
