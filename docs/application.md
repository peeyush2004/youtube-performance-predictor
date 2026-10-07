# Streamlit Application Architecture

## Purpose

The Streamlit app is an inference interface for the frozen Group B XGBoost model. It estimates observed view counts for a video represented in a dated YouTube trending snapshot. It does not train a model, fit preprocessing, split data, calculate feature importance, or write to model/data artifacts.

The application is not a pre-publication forecasting model and should not be interpreted as predicting future views for a newly uploaded video. `days_to_trending` is snapshot-context information and must be available for the dated snapshot being represented.

## Components

- `app/streamlit_app.py` implements the page, form, model details, methodology notice, validation messages, and displayed result.
- `app/prediction.py` loads and verifies the saved pipeline and metadata, derives supported categorical values from the fitted encoder, validates one input record, creates a one-row DataFrame, and converts the prediction to view scale.
- `models/group_b_xgboost_pipeline.joblib` contains the frozen scikit-learn Pipeline with fitted preprocessing and the XGBoost 3.2.0 regressor.
- `models/group_b_xgboost_metadata.json` records the feature list, target, model configuration, split, and package versions.
- `docs/final_model_contract.md` is the detailed source of the model schema and methodology constraints.

## Input and inference flow

1. The UI loads the model once per Streamlit process using `st.cache_resource` and `joblib.load`.
2. The loader checks that the artifact is a preprocessing-plus-XGBoost Pipeline and that metadata and fitted input columns match the 13 Group B features.
3. Region and category choices are taken from the saved OneHotEncoder's fitted categories. The publish-year validation range (2006–2018) comes from the existing training split. Weekday values use the feature-engineering convention `Monday=0` through `Sunday=6` (`publish_time.dt.dayofweek`).
4. The form collects the exact required fields: region, category, title and description length/word counts, tag count, publish year/month/day-of-week/hour, description-missing status, and `days_to_trending`.
5. Validation checks the exact schema, supported categorical values, integer/finite numeric inputs, nonnegative counts and snapshot age, training year range, month 1–12, weekday 0–6, hour 0–23, and the binary description-missing status. Invalid values yield concise user-facing messages.
6. A single-row pandas DataFrame is created in metadata column order and passed directly to `pipeline.predict`. The application does not recreate or refit imputation, scaling, or one-hot encoding.
7. The returned `predicted_log_observed_views` is converted using the established rule `expm1(max(predicted_log_observed_views, 0))`; the result is checked finite/nonnegative and displayed as a whole view count with comma separators.

Missing model files, schema mismatches, and prediction failures are logged for development and shown to users as short messages without a Python traceback. The model does not provide calibrated uncertainty estimates; the UI does not display confidence intervals or ranges.

## Run locally

Use the project `.venv` with Python 3.11.9 and the versions in `requirements.txt`, including XGBoost 3.2.0. In PowerShell, from the project root, activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then start Streamlit:

```powershell
python -m streamlit run app/streamlit_app.py
```

The project-relative artifact paths allow the app to be launched from the project directory without manually reproducing preprocessing. Keep the frozen model artifact and its metadata together in `models/`.

## Verification

`tests/test_streamlit_app.py` covers artifact loading, feature schema, a synthetic prediction from the final model contract, invalid negative `days_to_trending`, invalid month and hour, missing input fields, invalid categories, finite/nonnegative output, and the Streamlit form submission path. Tests only load and call the frozen artifact; they do not train or alter it.
