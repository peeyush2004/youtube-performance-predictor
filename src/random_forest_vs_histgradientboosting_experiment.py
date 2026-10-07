"""Compare Group B RandomForest and HistGradientBoosting on the saved split."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

from train_baseline import FORBIDDEN_FEATURES, GROUP_A_CATEGORICAL, GROUP_A_NUMERIC
from group_a_vs_group_b_experiment import (
    build_preprocessor,
    compute_log_metrics,
    compute_metrics,
    inverse_log_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RF_RESULTS_PATH = REPORTS_DIR / "group_a_vs_group_b_results.csv"
RF_BUCKET_PATH = REPORTS_DIR / "group_a_vs_group_b_bucket_metrics.csv"
RF_REGION_PATH = REPORTS_DIR / "group_a_vs_group_b_region_comparison.csv"
RF_MODEL_PATH = PROJECT_ROOT / "models" / "group_b_random_forest_pipeline.joblib"
HGB_RESULTS_PATH = REPORTS_DIR / "random_forest_vs_histgradientboosting_results.csv"
HGB_BUCKET_PATH = REPORTS_DIR / "random_forest_vs_histgradientboosting_bucket_metrics.csv"
HGB_COMPRESSION_PATH = REPORTS_DIR / "random_forest_vs_histgradientboosting_compression.csv"
HGB_REGION_PATH = REPORTS_DIR / "random_forest_vs_histgradientboosting_region_mae.csv"
HGB_MODEL_PATH = PROJECT_ROOT / "models" / "group_b_histgradientboosting_pipeline.joblib"
HGB_METADATA_PATH = PROJECT_ROOT / "models" / "group_b_histgradientboosting_metadata.json"

RANDOM_STATE = 42
GROUP_B_CATEGORICAL = list(GROUP_A_CATEGORICAL)
GROUP_B_NUMERIC = list(GROUP_A_NUMERIC) + ["days_to_trending"]
GROUP_B_FEATURES = GROUP_B_CATEGORICAL + GROUP_B_NUMERIC
BUCKETS = ["<10K", "10K-100K", "100K-1M", "1M-10M", ">10M"]
HISTGB_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": RANDOM_STATE,
    # Use every training row and avoid an internal random validation split.
    "early_stopping": False,
}


def verify_split(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    ranges: dict[str, dict[str, object]] = {}
    parsed_dates: dict[str, pd.Series] = {}
    for name, frame in frames.items():
        dates = pd.to_datetime(frame["trending_date"], format="%y.%d.%m", errors="raise")
        parsed_dates[name] = dates
        ranges[name] = {
            "rows": int(len(frame)),
            "unique_videos": int(frame["video_id"].nunique()),
            "date_range": [dates.min().date().isoformat(), dates.max().date().isoformat()],
        }

    assert parsed_dates["train"].max() < parsed_dates["validation"].min()
    assert parsed_dates["validation"].max() < parsed_dates["test"].min()
    id_sets = {name: set(frame["video_id"]) for name, frame in frames.items()}
    assert not (id_sets["train"] & id_sets["validation"])
    assert not (id_sets["train"] & id_sets["test"])
    assert not (id_sets["validation"] & id_sets["test"])
    return ranges


def feature_group_pipeline(regressor: object) -> Pipeline:
    forbidden = FORBIDDEN_FEATURES.intersection(GROUP_B_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden predictors present: {sorted(forbidden)}")
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(GROUP_B_CATEGORICAL, GROUP_B_NUMERIC),
            ),
            ("regressor", regressor),
        ]
    )


def pivot_saved_rf_metrics(saved: pd.DataFrame) -> pd.DataFrame:
    rf_long = saved[saved["feature_group"] == "Group B"].copy()
    rf_metrics = rf_long.pivot_table(
        index=["split", "scale"], columns="metric", values="value", aggfunc="first"
    ).reset_index()
    rf_metrics["model"] = "RandomForest Group B"
    return rf_metrics[["model", "split", "scale", "MAE", "RMSE", "R2", "MedianAE"]]


def metric_rows(model_name: str, predictions: dict[str, np.ndarray], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, pred in predictions.items():
        actual_views = frames[split_name]["observed_views"]
        for scale, metrics in (
            ("original", compute_metrics(actual_views, pred)),
            ("log", compute_log_metrics(actual_views, pred)),
        ):
            rows.append({"model": model_name, "split": split_name, "scale": scale, **metrics})
    return pd.DataFrame(rows)


def bucket_for_value(value: float) -> str:
    if value < 10_000:
        return "<10K"
    if value < 100_000:
        return "10K-100K"
    if value < 1_000_000:
        return "100K-1M"
    if value < 10_000_000:
        return "1M-10M"
    return ">10M"


def bucket_metrics(
    model_name: str,
    actual: pd.Series,
    predicted: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual.to_numpy(dtype=np.float64), "predicted": predicted})
    frame["bucket"] = frame["actual"].map(bucket_for_value)
    rows: list[dict[str, object]] = []
    for bucket in BUCKETS:
        subset = frame[frame["bucket"] == bucket]
        if subset.empty:
            continue
        actual_values = subset["actual"].to_numpy()
        predicted_values = subset["predicted"].to_numpy()
        rows.append({
            "model": model_name,
            "bucket": bucket,
            "row_count": int(len(subset)),
            "mean_actual_views": float(actual_values.mean()),
            "mean_predicted_views": float(predicted_values.mean()),
            "median_actual_views": float(np.median(actual_values)),
            "median_predicted_views": float(np.median(predicted_values)),
            "mean_predicted_to_actual_ratio": float(predicted_values.mean() / actual_values.mean()),
            "MAE": float(np.mean(np.abs(predicted_values - actual_values))),
            "RMSE": float(np.sqrt(np.mean((predicted_values - actual_values) ** 2))),
            "mean_signed_error": float(np.mean(predicted_values - actual_values)),
        })
    return pd.DataFrame(rows)


def region_mae(model_name: str, frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    error_frame = pd.DataFrame({
        "region": frame["region"].to_numpy(),
        "actual": frame["observed_views"].to_numpy(dtype=np.float64),
        "predicted": predictions,
    })
    error_frame["absolute_error"] = np.abs(error_frame["actual"] - error_frame["predicted"])
    return (
        error_frame.groupby("region", as_index=False)
        .agg(observations=("actual", "size"), MAE=("absolute_error", "mean"))
        .assign(model=model_name)
    )


def save_figures(
    test_frame: pd.DataFrame,
    hgb_predictions: np.ndarray,
    rf_predictions: np.ndarray,
    buckets: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    actual = test_frame["observed_views"].to_numpy(dtype=np.float64)
    actual_log = np.log1p(actual)
    predicted_log = np.log1p(hgb_predictions)
    sample_size = min(30_000, len(actual))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = np.sort(rng.choice(len(actual), size=sample_size, replace=False))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.maximum(actual[sample_idx], 1), np.maximum(hgb_predictions[sample_idx], 1), s=7, alpha=0.22)
    bounds = [1, max(float(actual.max()), float(hgb_predictions.max()))]
    ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("HistGradientBoosting: actual vs predicted test views")
    ax.set_xlabel("Actual views (log axis; zero predictions shown at 1 for display)")
    ax.set_ylabel("Predicted views (log axis; zero predictions shown at 1 for display)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "histgb_actual_vs_predicted_test.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual_log[sample_idx], predicted_log[sample_idx], s=7, alpha=0.22)
    upper = max(float(actual_log.max()), float(predicted_log.max()))
    ax.plot([0, upper], [0, upper], color="black", linestyle="--", linewidth=1)
    ax.set_title("HistGradientBoosting: actual vs predicted log1p test views")
    ax.set_xlabel("log1p(actual views)")
    ax.set_ylabel("log1p(predicted views)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "histgb_actual_vs_predicted_log_test.png", dpi=180)
    plt.close(fig)

    residual = predicted_log - actual_log
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residual, bins=70, ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("HistGradientBoosting log-scale test residuals")
    ax.set_xlabel("log1p(predicted views) - log1p(actual views)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "histgb_residual_distribution_log_test.png", dpi=180)
    plt.close(fig)

    hgb_bucket = buckets[buckets["model"] == "HistGradientBoosting Group B"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=hgb_bucket, x="bucket", y="MAE", order=BUCKETS, ax=ax, color="steelblue")
    ax.set_title("HistGradientBoosting test MAE by actual-view bucket")
    ax.set_xlabel("Actual-view bucket")
    ax.set_ylabel("MAE (views)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "histgb_mae_by_bucket.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=buckets, x="bucket", y="MAE", hue="model", order=BUCKETS, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Test MAE by actual-view bucket: RandomForest vs HistGradientBoosting")
    ax.set_xlabel("Actual-view bucket")
    ax.set_ylabel("MAE (views, log axis)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rf_vs_histgb_bucket_mae.png", dpi=180)
    plt.close(fig)

    test_original = results[(results["split"] == "test") & (results["scale"] == "original")]
    plot_metrics = test_original.melt(
        id_vars="model", value_vars=["MAE", "RMSE"], var_name="metric", value_name="views_error"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_metrics, x="metric", y="views_error", hue="model", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Overall test error: RandomForest vs HistGradientBoosting")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Error in views (log axis)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rf_vs_histgb_test_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv", low_memory=False)
        for name in ("train", "validation", "test")
    }
    split_ranges = verify_split(frames)
    print("Verified existing chronological split:", json.dumps(split_ranges, sort_keys=True))
    print("Group B features:", GROUP_B_FEATURES)

    train_X = frames["train"][GROUP_B_FEATURES]
    train_y = frames["train"]["log_observed_views"]
    if any(not frame[GROUP_B_FEATURES].columns.equals(pd.Index(GROUP_B_FEATURES)) for frame in frames.values()):
        raise ValueError("Feature order differs across split files")

    rf_reference = pd.read_csv(RF_RESULTS_PATH)
    rf_reference = rf_reference[rf_reference["feature_group"] == "Group B"]
    rf_metrics = pivot_saved_rf_metrics(rf_reference)
    if len(rf_metrics) != 6:
        raise ValueError("Saved Group B RandomForest metrics are incomplete")

    # Load the saved RandomForest to derive row-level test diagnostics without retraining it.
    rf_pipeline = joblib.load(RF_MODEL_PATH, mmap_mode="r")
    if not isinstance(rf_pipeline.named_steps["regressor"], RandomForestRegressor):
        raise TypeError("Saved Group B model is not a RandomForestRegressor pipeline")
    rf_test_X = frames["test"][GROUP_B_FEATURES]
    rf_test_pred = inverse_log_predictions(rf_pipeline.predict(rf_test_X))
    rf_test_actual = frames["test"]["observed_views"]
    rf_metric_check = compute_metrics(rf_test_actual, rf_test_pred)
    rf_log_check = compute_log_metrics(rf_test_actual, rf_test_pred)
    expected_rf = rf_reference[(rf_reference["split"] == "test") & (rf_reference["scale"] == "original")].set_index("metric")["value"]
    expected_rf_log = rf_reference[(rf_reference["split"] == "test") & (rf_reference["scale"] == "log")].set_index("metric")["value"]
    for metric, value in rf_metric_check.items():
        if not np.isclose(value, expected_rf[metric], rtol=1e-9, atol=1e-6):
            raise ValueError(f"Saved RandomForest {metric} does not match the reference results")
    for metric, value in rf_log_check.items():
        if not np.isclose(value, expected_rf_log[metric], rtol=1e-9, atol=1e-9):
            raise ValueError(f"Saved RandomForest log {metric} does not match the reference results")
    del rf_pipeline

    hgb_pipeline = feature_group_pipeline(HistGradientBoostingRegressor(**HISTGB_PARAMETERS))
    started = time.perf_counter()
    hgb_pipeline.fit(train_X, train_y)
    training_seconds = time.perf_counter() - started

    hgb_predictions: dict[str, np.ndarray] = {}
    for split_name, frame in frames.items():
        hgb_predictions[split_name] = inverse_log_predictions(
            hgb_pipeline.predict(frame[GROUP_B_FEATURES])
        )

    hgb_metrics = metric_rows("HistGradientBoosting Group B", hgb_predictions, frames)
    results = pd.concat([rf_metrics, hgb_metrics], ignore_index=True)
    results = results[["model", "split", "scale", "MAE", "RMSE", "R2", "MedianAE"]]
    results.to_csv(HGB_RESULTS_PATH, index=False)

    rf_buckets = bucket_metrics("RandomForest Group B", rf_test_actual, rf_test_pred)
    hgb_buckets = bucket_metrics(
        "HistGradientBoosting Group B", frames["test"]["observed_views"], hgb_predictions["test"]
    )
    buckets = pd.concat([rf_buckets, hgb_buckets], ignore_index=True)
    buckets.to_csv(HGB_BUCKET_PATH, index=False)

    compression_rows: list[dict[str, object]] = []
    for model_name, prediction in (
        ("RandomForest Group B", rf_test_pred),
        ("HistGradientBoosting Group B", hgb_predictions["test"]),
    ):
        actual = frames["test"]["observed_views"].to_numpy(dtype=np.float64)
        compression_rows.append({
            "model": model_name,
            "bucket": "All test rows",
            "row_count": int(len(actual)),
            "mean_actual_views": float(actual.mean()),
            "mean_predicted_views": float(prediction.mean()),
            "median_actual_views": float(np.median(actual)),
            "median_predicted_views": float(np.median(prediction)),
            "mean_predicted_to_actual_ratio": float(prediction.mean() / actual.mean()),
        })
    compression = pd.concat([pd.DataFrame(compression_rows), buckets[[
        "model", "bucket", "row_count", "mean_actual_views", "mean_predicted_views",
        "median_actual_views", "median_predicted_views", "mean_predicted_to_actual_ratio",
    ]]], ignore_index=True)
    compression.to_csv(HGB_COMPRESSION_PATH, index=False)

    rf_regions = region_mae("RandomForest Group B", frames["test"], rf_test_pred)
    hgb_regions = region_mae(
        "HistGradientBoosting Group B", frames["test"], hgb_predictions["test"]
    )
    region_results = pd.concat([rf_regions, hgb_regions], ignore_index=True)
    region_results.to_csv(HGB_REGION_PATH, index=False)

    save_figures(frames["test"], hgb_predictions["test"], rf_test_pred, buckets, results)
    joblib.dump(hgb_pipeline, HGB_MODEL_PATH)
    metadata = {
        "model": "HistGradientBoostingRegressor",
        "target": "log_observed_views",
        "feature_group": "Group B",
        "feature_list": GROUP_B_FEATURES,
        "hyperparameters": HISTGB_PARAMETERS,
        "preprocessing": {
            "categorical": "training-only constant Unknown imputation followed by dense OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "numeric": "training-only median imputation followed by StandardScaler",
            "feature_engineering": "unchanged Group B categorical and numeric feature list",
        },
        "target_inverse_transform": "Clip predicted log1p views at zero, apply expm1, and enforce non-negative predictions, matching the Group B RandomForest experiment.",
        "split": split_ranges,
        "split_method": "Existing chronological date windows with whole-video assignment; no new split created.",
        "training_date": date.today().isoformat(),
        "training_time_seconds": float(training_seconds),
        "versions": {
            "python": ".venv Python 3.11.9",
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "metrics_file": str(HGB_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "bucket_metrics_file": str(HGB_BUCKET_PATH.relative_to(PROJECT_ROOT)),
        "model_path": str(HGB_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "random_forest_reference": {
            "model_path": str(RF_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "test_MAE_views": float(rf_metric_check["MAE"]),
            "test_RMSE_views": float(rf_metric_check["RMSE"]),
            "test_log_MAE": float(rf_log_check["MAE"]),
        },
    }
    HGB_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Training time (seconds):", round(training_seconds, 3))
    print("Comparison results:")
    print(results.to_string(index=False))
    print("Bucket results:")
    print(buckets.to_string(index=False))
    print("Compression summary:")
    print(compression.to_string(index=False))
    print("Region MAE:")
    print(region_results.pivot(index="region", columns="model", values="MAE").to_string())
    print("Saved HistGradientBoosting pipeline and new comparison outputs.")


if __name__ == "__main__":
    main()
