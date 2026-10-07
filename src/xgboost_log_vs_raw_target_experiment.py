"""Compare the existing log-target XGBoost with one raw-target XGBoost fit."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from random_forest_histgb_xgboost_experiment import (
    FEATURES,
    PARAMETERS,
    SPLITS,
    XGB_MODEL,
    actual_bucket,
    metric_dict,
    preprocessor,
)
from random_forest_vs_histgradientboosting_experiment import verify_split
from group_a_vs_group_b_experiment import inverse_log_predictions


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RAW_MODEL = MODELS / "group_b_xgboost_raw_target_pipeline.joblib"
RAW_METADATA = MODELS / "group_b_xgboost_raw_target_metadata.json"
RESULTS = REPORTS / "xgb_log_vs_raw_results.csv"
BUCKETS_OUT = REPORTS / "xgb_log_vs_raw_bucket_metrics.csv"
TRADEOFF_OUT = REPORTS / "xgb_log_vs_raw_tradeoff.csv"
DISTRIBUTION_OUT = REPORTS / "xgb_log_vs_raw_prediction_distribution.csv"
PLOT_NAMES = [
    "xgb_log_vs_raw_actual_predicted.png",
    "xgb_log_vs_raw_bucket_mae.png",
    "xgb_log_vs_raw_prediction_distribution.png",
    "xgb_log_vs_raw_bias.png",
]
RANGES = ["<=100K", ">100K to <=1M", ">1M to <=10M", ">10M"]


def get_metrics(model_name: str, target_repr: str, split: str, truth: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return {
        "model": model_name,
        "target_representation": target_repr,
        "target": "log_observed_views" if target_repr == "log" else "observed_views",
        "split": split,
        "scale": "original",
        **metric_dict(truth, pred),
    }


def bucket_analysis(model_name: str, target_repr: str, actual: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    labels = actual_bucket(actual)
    error = pred - actual
    abs_error = np.abs(error)
    total_abs_error = float(abs_error.sum())
    rows: list[dict[str, object]] = []
    for bucket in RANGES:
        mask = labels == bucket
        a, p = actual[mask], pred[mask]
        e, ae = error[mask], abs_error[mask]
        rows.append({
            "model": model_name,
            "target_representation": target_repr,
            "bucket": bucket,
            "row_count": int(mask.sum()),
            "MAE": float(ae.mean()),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MedianAE": float(np.median(ae)),
            "mean_signed_error": float(e.mean()),
            "median_signed_error": float(np.median(e)),
            "mean_actual_views": float(a.mean()),
            "mean_predicted_views": float(p.mean()),
            "mean_predicted_to_actual_ratio": float(p.mean() / a.mean()),
            "sum_absolute_error": float(ae.sum()),
            "total_absolute_error_contribution_pct": float(ae.sum() / total_abs_error * 100),
        })
    return rows


def cumulative_metrics(model_name: str, target_repr: str, actual: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    masks = [("<=1M", actual <= 1_000_000), (">1M", actual > 1_000_000), (">10M", actual > 10_000_000)]
    abs_error = np.abs(pred - actual)
    total_abs = abs_error.sum()
    rows = []
    for name, mask in masks:
        a, p = actual[mask], pred[mask]
        e, ae = p - a, abs_error[mask]
        rows.append({
            "model": model_name,
            "target_representation": target_repr,
            "subset": name,
            "row_count": int(mask.sum()),
            "row_pct": float(mask.mean() * 100),
            "MAE": float(ae.mean()),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MedianAE": float(np.median(ae)),
            "mean_signed_error": float(e.mean()),
            "median_signed_error": float(np.median(e)),
            "mean_actual_views": float(a.mean()),
            "mean_predicted_views": float(p.mean()),
            "mean_predicted_to_actual_ratio": float(p.mean() / a.mean()),
            "sum_absolute_error": float(ae.sum()),
            "total_absolute_error_contribution_pct": float(ae.sum() / total_abs * 100),
        })
    return rows


def prediction_summary(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    def stats(name: str, values: np.ndarray) -> dict[str, object]:
        return {
            "series": name,
            "mean_views": float(np.mean(values)),
            "median_views": float(np.median(values)),
            "p95_views": float(np.quantile(values, 0.95)),
            "p99_views": float(np.quantile(values, 0.99)),
            "maximum_views": float(np.max(values)),
            "minimum_views": float(np.min(values)),
        }
    rows = [stats("Actual test views", actual)]
    rows.extend(stats(name, value) for name, value in predictions.items())
    return pd.DataFrame(rows)


def verify_log_reference(log_pipeline: Pipeline, frames: dict[str, pd.DataFrame], expected: pd.DataFrame) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for split, frame in frames.items():
        pred = inverse_log_predictions(log_pipeline.predict(frame[FEATURES]))
        predictions[split] = pred
        truth = frame["observed_views"].to_numpy(dtype=np.float64)
        reproduced = metric_dict(truth, pred)
        saved = expected[(expected["model"] == "XGBoost Group B") & (expected["split"] == split)
                         & (expected["scale"] == "original")]
        for metric in ("MAE", "RMSE", "R2", "MedianAE"):
            if not np.isclose(reproduced[metric], float(saved.iloc[0][metric]), rtol=1e-9, atol=1e-6):
                raise AssertionError(f"Saved log-target XGBoost {split} {metric} did not reproduce")
    return predictions


def save_plots(actual: np.ndarray, predictions: dict[str, np.ndarray], bucket_df: pd.DataFrame, distribution_df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sample = np.sort(np.random.default_rng(42).choice(len(actual), size=min(25_000, len(actual)), replace=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, pred in zip(axes, predictions.keys(), predictions.values()):
        ax.scatter(np.maximum(actual[sample], 1), np.maximum(pred[sample], 1), s=6, alpha=0.18)
        hi = max(float(actual.max()), float(pred.max()), 1.0)
        ax.plot([1, hi], [1, hi], "k--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlabel="Actual views (log axis)", ylabel="Predicted views (log axis; values <=0 shown at 1)", title=name)
    fig.suptitle("XGBoost target representation: actual vs predicted")
    fig.tight_layout(); fig.savefig(FIGURES / PLOT_NAMES[0], dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    indexed = bucket_df.set_index(["target_representation", "bucket"])
    x = np.arange(len(RANGES)); width = 0.36
    ax.bar(x - width / 2, [indexed.loc[("log", b), "MAE"] for b in RANGES], width, label="Log target")
    ax.bar(x + width / 2, [indexed.loc[("raw", b), "MAE"] for b in RANGES], width, label="Raw target")
    ax.set_xticks(x, RANGES); ax.set_yscale("log")
    ax.set(ylabel="MAE (views, log axis)", title="Test MAE by actual-view range")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / PLOT_NAMES[1], dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in predictions.items():
        ax.hist(np.log1p(np.maximum(values, 0)), bins=70, density=True, histtype="step", linewidth=1.8, label=name)
    ax.hist(np.log1p(actual), bins=70, density=True, histtype="step", linewidth=1.8, color="black", label="Actual test")
    ax.set(xlabel="log1p(views)", ylabel="Density", title="Test prediction distributions")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / PLOT_NAMES[2], dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for repr_name, pred, marker in (("Log target", predictions["XGBoost log target"], "o"),
                                    ("Raw target", predictions["XGBoost raw target"], "s")):
        means = []
        for bucket in RANGES:
            row = bucket_df[(bucket_df["target_representation"] == ("log" if repr_name == "Log target" else "raw"))
                            & (bucket_df["bucket"] == bucket)].iloc[0]
            means.append(row["mean_signed_error"])
        ax.plot(RANGES, means, marker=marker, label=repr_name)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("symlog", linthresh=10_000)
    ax.set(ylabel="Mean signed error (views; symlog axis)", title="Test prediction bias by actual-view range")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / PLOT_NAMES[3], dpi=180); plt.close(fig)


def main() -> None:
    outputs = [RAW_MODEL, RAW_METADATA, RESULTS, BUCKETS_OUT, TRADEOFF_OUT, DISTRIBUTION_OUT]
    outputs.extend(FIGURES / name for name in PLOT_NAMES)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    frames = {name: pd.read_csv(SPLITS / f"{name}.csv", low_memory=False)
              for name in ("train", "validation", "test")}
    split_info = verify_split(frames)
    for name, frame in frames.items():
        if not set(FEATURES + ["observed_views", "log_observed_views"]).issubset(frame.columns):
            raise ValueError(f"{name} split missing target or Group B feature")

    # Existing fitted log-target model is reference; do not refit it.
    log_pipeline = joblib.load(XGB_MODEL, mmap_mode="r")
    saved_results = pd.read_csv(REPORTS / "rf_histgb_xgb_results.csv")
    log_predictions = verify_log_reference(log_pipeline, frames, saved_results)

    transformer = preprocessor()
    train_x = transformer.fit_transform(frames["train"][FEATURES])
    raw_xgb = XGBRegressor(**PARAMETERS)
    raw_pipeline = Pipeline([("preprocess", transformer), ("regressor", raw_xgb)])
    started = time.perf_counter()
    raw_xgb.fit(train_x, frames["train"]["observed_views"].to_numpy(dtype=np.float64))
    training_seconds = time.perf_counter() - started

    predictions: dict[str, np.ndarray] = {"XGBoost log target": log_predictions["test"]}
    rows: list[dict[str, object]] = []
    # Preserve existing recorded metrics, after artifact predictions validate them.
    log_rows = saved_results[(saved_results["model"] == "XGBoost Group B")].copy()
    log_rows["target_representation"] = "log"
    log_rows["target"] = "log_observed_views"
    rows.extend(log_rows[["model", "target_representation", "target", "split", "scale", "MAE", "RMSE", "R2", "MedianAE"]].to_dict("records"))
    raw_predictions: dict[str, np.ndarray] = {}
    for split, frame in frames.items():
        matrix = train_x if split == "train" else transformer.transform(frame[FEATURES])
        pred = raw_xgb.predict(matrix)
        raw_predictions[split] = pred
        rows.append(get_metrics("XGBoost raw target", "raw", split,
                                frame["observed_views"].to_numpy(dtype=np.float64), pred))
    predictions["XGBoost raw target"] = raw_predictions["test"]
    results = pd.DataFrame(rows)

    test_actual = frames["test"]["observed_views"].to_numpy(dtype=np.float64)
    bucket_rows = []
    bucket_rows.extend(bucket_analysis("XGBoost log target", "log", test_actual, predictions["XGBoost log target"]))
    bucket_rows.extend(bucket_analysis("XGBoost raw target", "raw", test_actual, predictions["XGBoost raw target"]))
    bucket_df = pd.DataFrame(bucket_rows)
    tradeoff_rows = cumulative_metrics("XGBoost log target", "log", test_actual, predictions["XGBoost log target"])
    tradeoff_rows += cumulative_metrics("XGBoost raw target", "raw", test_actual, predictions["XGBoost raw target"])
    tradeoff_df = pd.DataFrame(tradeoff_rows)
    distribution_df = prediction_summary(test_actual, predictions)

    save_plots(test_actual, predictions, bucket_df, distribution_df)
    results.to_csv(RESULTS, index=False)
    bucket_df.to_csv(BUCKETS_OUT, index=False)
    tradeoff_df.to_csv(TRADEOFF_OUT, index=False)
    distribution_df.to_csv(DISTRIBUTION_OUT, index=False)
    joblib.dump(raw_pipeline, RAW_MODEL, compress=3)

    metadata = {
        "model": "XGBRegressor",
        "xgboost_version": xgboost.__version__,
        "target": "observed_views",
        "target_transform": "none",
        "feature_group": "Group B",
        "feature_list": FEATURES,
        "hyperparameters": PARAMETERS,
        "preprocessing": {
            "categorical": "training-only constant Unknown imputation then OneHotEncoder(handle_unknown='ignore', sparse_output=True)",
            "numeric": "training-only median imputation then StandardScaler",
            "column_transformer_sparse_threshold": 1.0,
            "matrix_format": "scipy CSR sparse; no dense conversion",
            "same_preprocessing_function_as_log_target_model": True,
        },
        "split": split_info,
        "split_method": "Existing chronological date windows with whole-video assignment; no new split created.",
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
        "reference_log_target_model": str(XGB_MODEL.relative_to(ROOT)),
        "metrics_file": str(RESULTS.relative_to(ROOT)),
        "bucket_metrics_file": str(BUCKETS_OUT.relative_to(ROOT)),
        "tradeoff_metrics_file": str(TRADEOFF_OUT.relative_to(ROOT)),
        "prediction_distribution_file": str(DISTRIBUTION_OUT.relative_to(ROOT)),
    }
    RAW_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Split verified:", split_info)
    print("XGBoost:", xgboost.__version__, "raw-target training seconds:", round(training_seconds, 3))
    print("Train/validation/test metrics:")
    print(results.to_string(index=False))
    print("Test range metrics:")
    print(bucket_df.to_string(index=False))
    print("Cumulative tail tradeoff:")
    print(tradeoff_df.to_string(index=False))
    print("Prediction distribution:")
    print(distribution_df.to_string(index=False))


if __name__ == "__main__":
    main()
