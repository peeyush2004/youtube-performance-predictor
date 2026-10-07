"""Controlled XGBoost feature ablation: metadata-only Group A vs Group B."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from group_a_vs_group_b_experiment import inverse_log_predictions
from random_forest_histgb_xgboost_experiment import (
    CATEGORICAL as GROUP_B_CATEGORICAL,
    FEATURES as GROUP_B_FEATURES,
    NUMERIC as GROUP_B_NUMERIC,
    PARAMETERS,
    SPLITS,
    XGB_MODEL as GROUP_B_MODEL,
    actual_bucket,
)
from random_forest_vs_histgradientboosting_experiment import verify_split
from train_baseline import FORBIDDEN_FEATURES, GROUP_A_CATEGORICAL, GROUP_A_NUMERIC


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
GROUP_A_FEATURES = list(GROUP_A_CATEGORICAL) + list(GROUP_A_NUMERIC)
GROUP_A_MODEL = MODELS / "group_a_xgboost_pipeline.joblib"
GROUP_A_METADATA = MODELS / "group_a_xgboost_metadata.json"
RESULTS_PATH = REPORTS / "xgb_group_a_vs_group_b_results.csv"
BUCKET_PATH = REPORTS / "xgb_group_a_vs_group_b_bucket_metrics.csv"
IMPORTANCE_PATH = REPORTS / "xgb_group_b_feature_importance.csv"
PLOTS = ["xgb_group_a_vs_b_mae.png", "xgb_group_a_vs_b_bucket_mae.png", "xgb_group_a_vs_b_metrics.png"]
BUCKETS = ["<=100K", ">100K to <=1M", ">1M to <=10M", ">10M"]
assert GROUP_B_FEATURES == GROUP_A_FEATURES + ["days_to_trending"]
assert not FORBIDDEN_FEATURES.intersection(GROUP_A_FEATURES + GROUP_B_FEATURES)


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Same sparse, train-only preprocessing semantics as the Group B XGBoost run."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]),
                categorical,
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=1.0,
    )


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MedianAE": float(median_absolute_error(actual, predicted)),
    }


def eval_rows(model: str, split: str, actual: np.ndarray, predicted: np.ndarray) -> list[dict[str, object]]:
    log_actual = np.log1p(actual)
    log_predicted = np.log1p(predicted)
    base = {"model": model, "split": split}
    return [
        {**base, "scale": "original", **metric_values(actual, predicted)},
        {**base, "scale": "log", **metric_values(log_actual, log_predicted)},
    ]


def bucket_rows(model: str, actual: np.ndarray, predicted: np.ndarray) -> list[dict[str, object]]:
    labels = actual_bucket(actual)
    errors = predicted - actual
    absolute = np.abs(errors)
    total_absolute = absolute.sum()
    rows = []
    for bucket in BUCKETS:
        mask = labels == bucket
        a, p, e, ae = actual[mask], predicted[mask], errors[mask], absolute[mask]
        rows.append({
            "model": model,
            "bucket": bucket,
            "row_count": int(mask.sum()),
            "MAE": float(ae.mean()),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MedianAE": float(np.median(ae)),
            "log_MAE": float(np.mean(np.abs(np.log1p(p) - np.log1p(a)))),
            "mean_signed_error": float(e.mean()),
            "median_signed_error": float(np.median(e)),
            "mean_actual_views": float(a.mean()),
            "mean_predicted_views": float(p.mean()),
            "mean_predicted_to_actual_ratio": float(p.mean() / a.mean()),
            "sum_absolute_error": float(ae.sum()),
            "total_absolute_error_contribution_pct": float(ae.sum() / total_absolute * 100),
        })
    return rows


def aggregate_group_b_importance(pipeline: Pipeline) -> pd.DataFrame:
    prep = pipeline.named_steps["preprocess"]
    regressor = pipeline.named_steps["regressor"]
    transformed_importance = np.asarray(regressor.feature_importances_, dtype=np.float64)
    onehot = prep.named_transformers_["categorical"].named_steps["onehot"]
    source_names = []
    for feature, categories in zip(GROUP_B_CATEGORICAL, onehot.categories_):
        source_names.extend([feature] * len(categories))
    source_names.extend(GROUP_B_NUMERIC)
    if len(source_names) != len(transformed_importance):
        raise ValueError("Cannot align Group B transformed features to source columns")
    details = pd.DataFrame({"feature": source_names, "importance": transformed_importance})
    result = details.groupby("feature", as_index=False)["importance"].sum()
    result["importance"] /= result["importance"].sum()
    return result.sort_values("importance", ascending=False).reset_index(drop=True).assign(
        rank=lambda df: np.arange(1, len(df) + 1)
    )[["rank", "feature", "importance"]]


def build_plots(results: pd.DataFrame, buckets: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    view_mae = results[results["scale"] == "original"]
    pivot = view_mae.pivot(index="split", columns="model", values="MAE").loc[["train", "validation", "test"]]
    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set(ylabel="MAE (views)", title="XGBoost Group A vs Group B MAE")
    ax.legend(title="Feature group")
    ax.figure.tight_layout(); ax.figure.savefig(FIGURES / PLOTS[0], dpi=180); plt.close(ax.figure)

    b = buckets.pivot(index="bucket", columns="model", values="MAE").loc[BUCKETS]
    ax = b.plot(kind="bar", figsize=(9, 5), rot=0, logy=True)
    ax.set(ylabel="MAE (views, log axis)", title="XGBoost test MAE by actual-view range")
    ax.legend(title="Feature group")
    ax.figure.tight_layout(); ax.figure.savefig(FIGURES / PLOTS[1], dpi=180); plt.close(ax.figure)

    fields = [
        ("original", "MAE", "Views MAE"),
        ("original", "RMSE", "Views RMSE"),
        ("original", "R2", "Views R²"),
        ("log", "MAE", "Log MAE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (scale, metric, title) in zip(axes.flat, fields):
        subset = results[(results["scale"] == scale) & (results["split"] == "test")]
        ax.bar(["Group A", "Group B"], [subset.loc[subset["model"] == "XGBoost Group A", metric].iloc[0],
                                        subset.loc[subset["model"] == "XGBoost Group B", metric].iloc[0]])
        ax.set_title(title)
    fig.suptitle("XGBoost test metrics: Group A vs Group B")
    fig.tight_layout(); fig.savefig(FIGURES / PLOTS[2], dpi=180); plt.close(fig)


def main() -> None:
    outputs = [GROUP_A_MODEL, GROUP_A_METADATA, RESULTS_PATH, BUCKET_PATH, IMPORTANCE_PATH]
    outputs.extend(FIGURES / name for name in PLOTS)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    frames = {name: pd.read_csv(SPLITS / f"{name}.csv", low_memory=False)
              for name in ("train", "validation", "test")}
    parsed = verify_split(frames)
    split_info = {
        name: {**parsed[name], "rows": parsed[name]["rows"], "unique_videos": parsed[name]["unique_videos"]}
        for name in parsed
    }
    for name, frame in frames.items():
        required = set(GROUP_B_FEATURES + ["observed_views", "log_observed_views", "video_id", "trending_date"])
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} split lacks required fields")

    # Score the saved Group B XGBoost model and ensure it reproduces its recorded results.
    group_b = joblib.load(GROUP_B_MODEL)
    reference = pd.read_csv(REPORTS / "rf_histgb_xgb_results.csv")
    reference = reference[reference["model"] == "XGBoost Group B"]
    predictions: dict[str, dict[str, np.ndarray]] = {"XGBoost Group B": {}}
    result_rows: list[dict[str, object]] = []
    for split, frame in frames.items():
        actual = frame["observed_views"].to_numpy(dtype=np.float64)
        pred = inverse_log_predictions(group_b.predict(frame[GROUP_B_FEATURES]))
        predictions["XGBoost Group B"][split] = pred
        for result_row in eval_rows("XGBoost Group B", split, actual, pred):
            result_rows.append(result_row)
            expected = reference[(reference["split"] == split) & (reference["scale"] == result_row["scale"])]
            for metric in ("MAE", "RMSE", "R2", "MedianAE"):
                if not np.isclose(result_row[metric], float(expected.iloc[0][metric]), rtol=1e-9, atol=1e-6):
                    raise AssertionError(f"Saved Group B XGBoost {split} {result_row['scale']} {metric} did not reproduce")

    # Fit only Group A's preprocessing and XGBoost model; the target, rows, and split are unchanged.
    group_a_preprocessor = make_preprocessor(list(GROUP_A_CATEGORICAL), list(GROUP_A_NUMERIC))
    train = frames["train"]
    train_matrix = group_a_preprocessor.fit_transform(train[GROUP_A_FEATURES])
    group_a_regressor = XGBRegressor(**PARAMETERS)
    group_a_pipeline = Pipeline([("preprocess", group_a_preprocessor), ("regressor", group_a_regressor)])
    started = time.perf_counter()
    group_a_regressor.fit(train_matrix, train["log_observed_views"].to_numpy(dtype=np.float64))
    training_seconds = time.perf_counter() - started
    predictions["XGBoost Group A"] = {}
    for split, frame in frames.items():
        matrix = train_matrix if split == "train" else group_a_preprocessor.transform(frame[GROUP_A_FEATURES])
        pred = inverse_log_predictions(group_a_regressor.predict(matrix))
        predictions["XGBoost Group A"][split] = pred
        result_rows.extend(eval_rows("XGBoost Group A", split,
                                     frame["observed_views"].to_numpy(dtype=np.float64), pred))

    # Output one long table with Group B minus Group A absolute and relative change.
    results = pd.DataFrame(result_rows)
    a = results[results["model"] == "XGBoost Group A"].set_index(["split", "scale"])
    b = results[results["model"] == "XGBoost Group B"].set_index(["split", "scale"])
    for metric in ("MAE", "RMSE", "R2", "MedianAE"):
        results[f"{metric}_GroupB_minus_GroupA"] = [
            float(b.loc[(row.split, row.scale), metric] - a.loc[(row.split, row.scale), metric])
            for row in results.itertuples()
        ]
        results[f"{metric}_percent_change_GroupB_vs_GroupA"] = [
            float((b.loc[(row.split, row.scale), metric] - a.loc[(row.split, row.scale), metric])
                  / abs(a.loc[(row.split, row.scale), metric]) * 100)
            if a.loc[(row.split, row.scale), metric] != 0 else np.nan
            for row in results.itertuples()
        ]

    test_actual = frames["test"]["observed_views"].to_numpy(dtype=np.float64)
    bucket_df = pd.DataFrame([
        row for model, split_predictions in predictions.items()
        for row in bucket_rows(model, test_actual, split_predictions["test"])
    ])
    importance = aggregate_group_b_importance(group_b)

    build_plots(results, bucket_df)
    results.to_csv(RESULTS_PATH, index=False)
    bucket_df.to_csv(BUCKET_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    joblib.dump(group_a_pipeline, GROUP_A_MODEL, compress=3)
    metadata = {
        "model": "XGBRegressor",
        "xgboost_version": xgboost.__version__,
        "target": "log_observed_views",
        "target_transform": "log1p observed_views (stored target column)",
        "feature_group": "Group A",
        "feature_list": GROUP_A_FEATURES,
        "hyperparameters": PARAMETERS,
        "preprocessing": {
            "categorical": "training-only constant Unknown imputation then OneHotEncoder(handle_unknown='ignore', sparse_output=True)",
            "numeric": "training-only median imputation then StandardScaler",
            "column_transformer_sparse_threshold": 1.0,
            "matrix_format": "scipy CSR sparse",
            "same_strategy_as_group_b_xgboost": True,
        },
        "split": split_info,
        "split_method": "Existing chronological date windows with whole-video assignment; no new split created.",
        "random_state": 42,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": training_seconds,
        "versions": {
            "python": ".venv Python 3.11.9",
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "reference_group_b_model": str(GROUP_B_MODEL.relative_to(ROOT)),
        "results_file": str(RESULTS_PATH.relative_to(ROOT)),
        "bucket_metrics_file": str(BUCKET_PATH.relative_to(ROOT)),
        "group_b_feature_importance_file": str(IMPORTANCE_PATH.relative_to(ROOT)),
    }
    GROUP_A_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = results[(results["split"] == "test") & (results["scale"] == "original")]
    print("Verified chronological split:", split_info)
    print("XGBoost version:", xgboost.__version__, "Group A fit seconds:", round(training_seconds, 3))
    print("Test summary:")
    print(summary[["model", "MAE", "RMSE", "R2", "MedianAE", "MAE_GroupB_minus_GroupA",
                   "MAE_percent_change_GroupB_vs_GroupA"]].to_string(index=False))
    print("Test bucket metrics:")
    print(bucket_df.to_string(index=False))
    print("Group B feature importance:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
