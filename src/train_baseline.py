"""Train and evaluate modest baseline regressors for snapshot view estimation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_experiment_results.csv"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_regression_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "baseline_regression_metadata.json"
RANDOM_STATE = 42

GROUP_A_CATEGORICAL = ["region", "category_name"]
GROUP_A_NUMERIC = [
    "title_length",
    "title_word_count",
    "description_length",
    "description_word_count",
    "tag_count",
    "publish_year",
    "publish_month",
    "publish_day_of_week",
    "publish_hour",
    "description_missing",
]
GROUP_B = ["days_to_trending"]
FORBIDDEN_FEATURES = {
    "observed_views",
    "views",
    "log_observed_views",
    "likes",
    "dislikes",
    "comment_count",
    "thumbnail_link",
    "video_id",
    "trending_date",
    "publish_time",
    "channel_title",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-snapshot-context",
        action="store_true",
        help="Also run a separate Group A + days_to_trending snapshot-context experiment.",
    )
    return parser.parse_args()


def build_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def estimator_specs() -> list[tuple[str, Any]]:
    return [
        ("DummyRegressor", lambda: DummyRegressor(strategy="median")),
        ("LinearRegression", lambda: LinearRegression()),
        (
            "RandomForestRegressor",
            lambda: RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "HistGradientBoostingRegressor",
            lambda: HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def inverse_predictions(predictions: np.ndarray, target_name: str) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    if target_name == "raw_views":
        view_predictions = predictions
    elif target_name == "log_views":
        upper_bound = np.log(np.finfo(np.float64).max) - 1.0
        view_predictions = np.expm1(np.clip(predictions, 0.0, upper_bound))
    else:
        raise ValueError(f"Unexpected target name: {target_name}")
    return np.maximum(view_predictions, 0.0)


def calculate_metrics(actual_views: pd.Series, predicted_views: np.ndarray) -> dict[str, float]:
    actual = actual_views.to_numpy(dtype=np.float64)
    predicted = np.asarray(predicted_views, dtype=np.float64)
    actual_log = np.log1p(actual)
    predicted_log = np.log1p(predicted)
    return {
        "MAE_views": float(mean_absolute_error(actual, predicted)),
        "RMSE_views": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2_views": float(r2_score(actual, predicted)),
        "MedianAE_views": float(median_absolute_error(actual, predicted)),
        "MAE_log_views": float(mean_absolute_error(actual_log, predicted_log)),
        "RMSE_log_views": float(np.sqrt(mean_squared_error(actual_log, predicted_log))),
    }


def load_splits() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    required_files = [SPLIT_DIR / f"{name}.csv" for name in ("train", "validation", "test")]
    missing = [path for path in required_files if not path.is_file()]
    if missing or not SPLIT_METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Split files are missing. Run `python src/split_dataset.py` from the project root first."
        )
    frames = {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv", low_memory=False)
        for name in ("train", "validation", "test")
    }
    metadata = json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))
    return frames, metadata


def run_experiments(
    frames: dict[str, pd.DataFrame],
    *,
    include_snapshot_context: bool,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    group_specs: list[tuple[str, list[str], list[str]]] = [
        ("Group_A", GROUP_A_CATEGORICAL, GROUP_A_NUMERIC),
    ]
    if include_snapshot_context:
        group_specs.append(
            ("Group_A_plus_B", GROUP_A_CATEGORICAL, GROUP_A_NUMERIC + GROUP_B)
        )

    targets = {
        "raw_views": "observed_views",
        "log_views": "log_observed_views",
    }
    results: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for feature_group, categorical_features, numeric_features in group_specs:
        feature_columns = categorical_features + numeric_features
        forbidden = FORBIDDEN_FEATURES.intersection(feature_columns)
        if forbidden:
            raise ValueError(f"Forbidden predictors present in {feature_group}: {sorted(forbidden)}")
        for column in feature_columns:
            if any(column not in frame.columns for frame in frames.values()):
                raise ValueError(f"Feature {column!r} is missing from a split")

        train_features = frames["train"][feature_columns]
        validation_features = frames["validation"][feature_columns]
        test_features = frames["test"][feature_columns]

        for target_name, target_column in targets.items():
            train_target = frames["train"][target_column]
            for model_name, estimator_factory in estimator_specs():
                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocess",
                            build_preprocessor(categorical_features, numeric_features),
                        ),
                        ("regressor", estimator_factory()),
                    ]
                )
                started = time.perf_counter()
                pipeline.fit(train_features, train_target)
                training_time = time.perf_counter() - started

                row: dict[str, Any] = {
                    "model": model_name,
                    "target": target_name,
                    "feature_group": feature_group,
                    "training_time_seconds": training_time,
                }
                for split_name, features in (
                    ("validation", validation_features),
                    ("test", test_features),
                ):
                    predicted_target = pipeline.predict(features)
                    predicted_views = inverse_predictions(predicted_target, target_name)
                    metrics = calculate_metrics(
                        frames[split_name]["observed_views"], predicted_views
                    )
                    row.update(
                        {
                            f"{split_name}_{metric}": value
                            for metric, value in metrics.items()
                        }
                    )
                results.append(row)
                fitted_pipelines[f"{feature_group}|{target_name}|{model_name}"] = pipeline
                print(
                    f"Completed {feature_group} / {target_name} / {model_name} "
                    f"in {training_time:.2f}s; validation MAE={row['validation_MAE_views']:.3f}"
                )

    return pd.DataFrame(results), fitted_pipelines


def main() -> None:
    args = parse_args()
    frames, split_metadata = load_splits()

    print("Training-only preprocessing: numeric median imputer + StandardScaler; ")
    print("categorical constant 'Unknown' imputer + dense OneHotEncoder(handle_unknown='ignore').")
    print(f"Classification thresholds from training only: {split_metadata['classification_thresholds']}")
    print(f"Classification counts: { {name: details['performance_category_counts'] for name, details in split_metadata['splits'].items()} }")

    results, pipelines = run_experiments(
        frames,
        include_snapshot_context=args.include_snapshot_context,
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    primary = results[results["feature_group"] == "Group_A"].copy()
    if primary.empty:
        raise ValueError("No Group A experiments were run")
    selected_row = primary.sort_values(
        ["validation_MAE_views", "validation_RMSE_views"],
        ascending=[True, True],
    ).iloc[0]
    selected_key = (
        f"{selected_row['feature_group']}|{selected_row['target']}|{selected_row['model']}"
    )
    selected_pipeline = pipelines[selected_key]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_PATH)
    selected_metadata = {
        "selected_model": str(selected_row["model"]),
        "target": str(selected_row["target"]),
        "feature_group": str(selected_row["feature_group"]),
        "validation_selection_strategy": "Lowest validation MAE in original view-count scale; RMSE breaks ties.",
        "validation_MAE_views": float(selected_row["validation_MAE_views"]),
        "validation_RMSE_views": float(selected_row["validation_RMSE_views"]),
        "test_MAE_views": float(selected_row["test_MAE_views"]),
        "test_RMSE_views": float(selected_row["test_RMSE_views"]),
        "group_a_categorical_features": GROUP_A_CATEGORICAL,
        "group_a_numeric_features": GROUP_A_NUMERIC,
        "group_b_snapshot_context_features": GROUP_B,
        "preprocessing": {
            "numeric": "training-fold median imputation followed by StandardScaler",
            "categorical": "training-fold constant Unknown imputation followed by OneHotEncoder(handle_unknown='ignore')",
        },
        "target_inverse_transform": "For log_views, clip predicted log1p count at zero, apply expm1, and enforce non-negative view counts.",
        "random_state": RANDOM_STATE,
        "training_time_seconds": float(selected_row["training_time_seconds"]),
        "split_metadata": "data/processed/splits/split_metadata.json",
        "results_file": "data/processed/baseline_experiment_results.csv",
        "model_path": "models/baseline_regression_pipeline.joblib",
        "target_semantics": "Observed view count in the dated trending snapshot; not a future or 30-day forecast.",
        "snapshot_context_experiment_run": bool(args.include_snapshot_context),
    }
    MODEL_METADATA_PATH.write_text(
        json.dumps(selected_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nBaseline experiment results:")
    print(results.sort_values(["feature_group", "target", "validation_MAE_views"]).to_string(index=False))
    print("\nSelected Group A baseline by validation MAE (original views):")
    print(selected_row.to_string())
    print("\nHeld-out test metrics for selected baseline:")
    for metric in (
        "test_MAE_views",
        "test_RMSE_views",
        "test_R2_views",
        "test_MedianAE_views",
        "test_MAE_log_views",
        "test_RMSE_log_views",
    ):
        print(f"{metric}: {selected_row[metric]}")
    print(f"Results saved to: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Selected pipeline saved to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Model metadata saved to: {MODEL_METADATA_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()