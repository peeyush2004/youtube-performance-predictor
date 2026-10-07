# YouTube Trending Video View Count Estimation Using Machine Learning

## Project

This project estimates observed view counts for videos appearing in dated YouTube trending snapshots. Each observation is a video in a region on a recorded trending date. The source population is limited to videos selected for trending; results should not be generalized to all YouTube uploads.

The target is `observed_views`, the count recorded in that snapshot. Model training uses `log_observed_views = log1p(observed_views)`.

## Final model

The selected final model is an XGBoost 3.2.0 `XGBRegressor` pipeline using Group B features and the `log_observed_views` target. Group B consists of Group A metadata-style features plus `days_to_trending`. That feature is snapshot-context information and requires the dated trending observation.

- Pipeline: `models/group_b_xgboost_pipeline.joblib` — the selected final trained model used by the Streamlit application; no training is required to run the app.
- Metadata: `models/group_b_xgboost_metadata.json`
- Input schema: 13 features, defined in `docs/final_model_contract.md`
- Evaluation: existing chronological, whole-video train/validation/test split

Final held-out test metrics:

| Metric | Test result |
|---|---:|
| MAE | 964,651.57 |
| RMSE | 4,093,751.92 |
| R² | 0.062105 |
| MedianAE | 130,516.76 |
| Log MAE | 1.100355 |

In the XGBoost Group A vs Group B ablation, Group A test MAE was 1,036,434.86. Group B reduced test MAE by 6.93% on that same test split. Group B is the selected feature group for this snapshot-estimation task; `days_to_trending` must not be presented as available before publication.

## Interpretation and limits

The application estimates the view count observed in an existing dated trending snapshot. It is not a pre-publication predictor, a future-view predictor, a 30-day forecast, or a universal YouTube view predictor. The test results apply to the held-out period and this selected trending-video population. The high-view tail remains difficult to estimate and is underpredicted; aggregate metrics do not imply uniform performance across view ranges.

The displayed view estimate is obtained by converting the pipeline's predicted log value with the project's established nonnegative inverse transform. The app loads the saved pipeline and does not train a model or rebuild preprocessing.

## Repository contents

- `app/`: Streamlit interface and saved-model inference/validation helpers.
- `docs/`: data provenance, methodology, model training record, application notes, and final model contract.
- `models/group_b_xgboost_pipeline.joblib` and `models/group_b_xgboost_metadata.json`: selected final model and its required metadata.
- `reports/`: experiment metrics, bucket analyses, and figures.
- `src/`: dataset construction, split, diagnostics, and historical experiment scripts. Running a training script is not required to use the application.
- `tests/`: saved-model contract and Streamlit interface tests.

The Git repository intentionally excludes local Python environments, raw dataset files, generated processed datasets, and large historical experiment model binaries. The raw and processed datasets are not included. See the project documentation for dataset provenance and reproduction information.

## Setup and Run

Clone the repository (replace `<repository-url>` with its Git URL), then open PowerShell in the cloned project directory:

```powershell
git clone <repository-url>
cd youtube-performance-predictor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

The selected final XGBoost model is included in the repository, so no model training is needed to run the application. The existing direct-interpreter commands are also available if you do not activate the environment:

```powershell
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

### Dataset Reproduction

Raw and processed datasets are intentionally excluded from the Git repository because of their size. The project documentation contains the dataset provenance and reproduction information; no download URL is provided here.

See [`docs/final_model_contract.md`](docs/final_model_contract.md) for the exact required inputs, application validation, model-level preprocessing, and prediction conversion. See [`docs/application.md`](docs/application.md) for the app's inference flow.
