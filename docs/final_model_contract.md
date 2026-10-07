# Final Model Contract

## Status and scope

This contract defines the input, prediction, and model-version requirements for the Streamlit application using the final saved model. Application architecture and run instructions are in `docs/application.md`.

The application estimates observed view counts for videos in dated YouTube trending snapshots. It is not a pre-publication forecasting model and should not be interpreted as predicting future views for a newly uploaded video.

Group A is the metadata-style feature set. Group B is Group A plus `days_to_trending`, a snapshot-context feature. Because `days_to_trending` depends on the dated trending snapshot, Group B is suitable only when that snapshot context is available; it must not be presented as a pre-publication feature set.

## Final model version

| Property | Contract value |
|---|---|
| Model family | XGBoost `XGBRegressor` in a scikit-learn `Pipeline` |
| XGBoost version | 3.2.0 |
| Target | `log_observed_views` |
| Feature group | Group B |
| Training dataset | `data/processed/modeling_dataset.csv` |
| Split | Existing chronological, whole-video assignment; no random split |
| Train window | 2017-11-14 through 2018-04-02; 237,400 rows |
| Validation window | 2018-04-03 through 2018-05-12; 50,981 rows |
| Test window | 2018-05-13 through 2018-06-14; 51,003 rows |
| Pipeline artifact | `models/group_b_xgboost_pipeline.joblib` |
| Metadata | `models/group_b_xgboost_metadata.json` |
| Training timestamp in metadata | 2026-10-03T12:57:07.194412+00:00 |
| Python / scikit-learn / NumPy / pandas / SciPy / joblib | 3.11.9 / 1.9.1 / 2.4.6 / 3.0.6 / 1.17.1 / 1.6.0 |

The XGBoost parameters recorded for the artifact are `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `reg_lambda=1`, `objective=reg:squarederror`, `eval_metric=rmse`, `random_state=42`, `n_jobs=-1`, and `tree_method=hist`. The fitted preprocessing is part of the saved pipeline.

## Input schema

Every prediction record must supply these 13 columns. Use the names and order in the artifact metadata; do not derive or substitute other predictors.

### A. Numerical features

These are numeric inputs processed by the pipeline's training-fitted median imputer and `StandardScaler`:

- `title_length`
- `title_word_count`
- `description_length`
- `description_word_count`
- `tag_count`
- `publish_year`
- `publish_month`
- `publish_day_of_week`
- `publish_hour`

Calendar fields should use the same definitions as the training dataset. In particular, `publish_day_of_week` is the existing dataset's encoded weekday value; do not change its encoding in the application.

### B. Categorical features

These are processed by the pipeline's training-fitted constant imputer and one-hot encoder:

- `region`
- `category_name`

### C. Boolean/status feature

- `description_missing` — binary status indicating whether the description was missing in the source record. Supply the same 0/1 representation used by the dataset. The saved pipeline processes this column in its numeric branch.

### D. Snapshot-context feature

- `days_to_trending` — numeric UTC calendar-day difference between publication and the dated trending snapshot. This is required by Group B and is also processed in the numeric branch. It is not available before publication and the relevant trending snapshot exist.

### Excluded inputs

Do not pass `observed_views`, `log_observed_views`, `likes`, `dislikes`, `comment_count`, `thumbnail_link`, `video_id`, `trending_date`, `publish_time`, `channel_title`, or other post-outcome engagement fields as model inputs. `observed_views` is the evaluation outcome, not an input; `log_observed_views` is the training target, not an input.

## Preprocessing and application input validation

### Model-level preprocessing

The saved pipeline contains the fitted preprocessing used during training. The artifact's numeric branch applies a training-fitted median `SimpleImputer` followed by `StandardScaler`. Its categorical branch fills missing values with the constant `Unknown`, then applies a fitted `OneHotEncoder(handle_unknown="ignore")`. These transformations describe what the pipeline can do when it receives a correctly shaped DataFrame; they do not describe which user inputs the Streamlit application accepts.

### Application-level validation

The Streamlit application intentionally validates user input before calling the pipeline:

- All 13 required feature fields must be present, and extra fields are rejected. Missing fields or values are reported as input errors; the application does not silently fill them with invented defaults.
- Numeric fields must be finite whole numbers. Missing, non-numeric, non-finite, or fractional values are rejected before model preprocessing can impute them.
- `region` and `category_name` must be selected from the fitted encoder's supported values. Although the underlying encoder can ignore unseen categories in a direct pipeline call, application validation rejects unsupported choices.
- `publish_year` is restricted to the application's documented training-era range, 2006 through 2018. Month must be 1 through 12, weekday 0 through 6 using Monday=0, and hour 0 through 23.
- Count-like inputs and `days_to_trending` must be nonnegative. `description_missing` must be a boolean or the dataset's 0/1 representation.
- `days_to_trending` must come from the dated trending snapshot; the application does not substitute a pre-publication estimate.

The application constructs a one-row DataFrame in the exact metadata feature order and passes it to the saved pipeline. It does not recreate or refit preprocessing. The pipeline's `ColumnTransformer` selects its fitted columns and drops remainder columns, but application requests still enforce the exact input schema.
## Prediction contract

1. Pass a one-row DataFrame containing the 13 feature values to the loaded saved pipeline.
2. The pipeline returns `predicted_log_observed_views`, its estimate on the trained `log_observed_views` scale.
3. Convert to the displayed view scale with `predicted_observed_views = expm1(predicted_log_observed_views)`.
4. Match the project's established inverse-transform behavior: clip the predicted log value at zero before `expm1`, then enforce a nonnegative result. This prevents negative displayed view counts and matches the existing Group B evaluation conversion.

Example implementation:

```python
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load("models/group_b_xgboost_pipeline.joblib")
FEATURES = [
    "region", "category_name", "title_length", "title_word_count",
    "description_length", "description_word_count", "tag_count",
    "publish_year", "publish_month", "publish_day_of_week", "publish_hour",
    "description_missing", "days_to_trending",
]
record = {
    "region": "US",
    "category_name": "Science & Technology",
    "title_length": 54,
    "title_word_count": 9,
    "description_length": 420,
    "description_word_count": 68,
    "tag_count": 8,
    "publish_year": 2018,
    "publish_month": 5,
    "publish_day_of_week": 2,
    "publish_hour": 14,
    "description_missing": 0,
    "days_to_trending": 2,
}
row = pd.DataFrame([record], columns=FEATURES)
predicted_log_observed_views = float(pipeline.predict(row)[0])
predicted_observed_views = float(np.expm1(max(predicted_log_observed_views, 0.0)))
```

The application must use the saved pipeline so its fitted imputer, scaler, and encoder remain identical to training. Do not rebuild or refit preprocessing in the application.

## Verified synthetic example

The synthetic record above was passed through the saved artifact. It returned:

- `predicted_log_observed_views`: **11.9949474335**
- `predicted_observed_views`: **161,933.54**

The log prediction and converted view count were finite; the displayed count was nonnegative. These are a pipeline smoke-check example, not a model performance claim or an observed video's result.

## Artifact validation record

The artifact loaded successfully with `joblib.load`. Its top-level object is a scikit-learn `Pipeline` with `preprocess` (`ColumnTransformer`) and `regressor` (`xgboost.sklearn.XGBRegressor`) steps. The preprocessing object's fitted input columns exactly match the 13 metadata features in the same order. The metadata identifies `XGBRegressor`, XGBoost 3.2.0, target `log_observed_views`, and feature group Group B. The documented Group B definition is Group A plus only `days_to_trending`. Neither `observed_views` nor `log_observed_views` appears in the input list. No mismatch was found, and no model artifact was modified.
