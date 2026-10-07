"""Diagnostic analysis for the selected baseline model without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_regression_pipeline.joblib"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "baseline_feature_importance.csv"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"

from train_baseline import GROUP_A_CATEGORICAL, GROUP_A_NUMERIC  # noqa: E402


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    metrics = {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MedianAE": float(median_absolute_error(actual, predicted)),
    }
    return metrics


def log_metric_summary(actual: np.ndarray, predicted_views: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted_views = np.asarray(predicted_views, dtype=np.float64)
    actual_log = np.log1p(actual)
    predicted_log = np.log1p(predicted_views)
    return {
        "MAE": float(mean_absolute_error(actual_log, predicted_log)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_log, predicted_log))),
        "R2": float(r2_score(actual_log, predicted_log)),
        "MedianAE": float(median_absolute_error(actual_log, predicted_log)),
    }


def inverse_log_predictions(predicted_log: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(predicted_log, dtype=np.float64), a_min=0.0, a_max=np.log(np.finfo(np.float64).max) - 1.0)
    return np.maximum(np.expm1(clipped), 0.0)


def split_summary(split_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | int | str]]:
    summary: dict[str, dict[str, float | int | str]] = {}
    for name, frame in split_frames.items():
        dates = pd.to_datetime(frame["trending_date"], format="%y.%d.%m", errors="raise")
        summary[name] = {
            "date_min": dates.min().date().isoformat(),
            "date_max": dates.max().date().isoformat(),
            "rows": int(len(frame)),
            "unique_videos": int(frame["video_id"].nunique()),
        }
    return summary


def load_splits() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv", low_memory=False)
        for name in ("train", "validation", "test")
    }


def add_prediction_columns(frame: pd.DataFrame, model) -> pd.DataFrame:
    features = frame[GROUP_A_CATEGORICAL + GROUP_A_NUMERIC].copy()
    pred_log = model.predict(features)
    pred_views = inverse_log_predictions(pred_log)
    actual_views = frame["observed_views"].to_numpy(dtype=np.float64)
    frame = frame.copy()
    frame["predicted_log_views"] = pred_log
    frame["predicted_views"] = pred_views
    frame["absolute_error"] = np.abs(actual_views - pred_views)
    frame["relative_error"] = np.abs(actual_views - pred_views) / np.maximum(actual_views, 1.0)
    frame["log_absolute_error"] = np.abs(np.log1p(actual_views) - np.log1p(pred_views))
    frame["log_actual_views"] = np.log1p(actual_views)
    frame["log_predicted_views"] = np.log1p(pred_views)
    frame["residual_log"] = frame["log_predicted_views"] - frame["log_actual_views"]
    return frame


def bucket_labels() -> list[str]:
    return ["<10K", "10K-100K", "100K-1M", "1M-10M", ">10M"]


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


def bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["observed_views"].to_numpy(dtype=np.float64)
    labels = pd.Series(values, index=frame.index).map(bucket_for_value)
    rows: list[dict[str, float | int | str]] = []
    for label in bucket_labels():
        subset = frame.loc[labels == label]
        if subset.empty:
            rows.append(
                {
                    "bucket": label,
                    "count": 0,
                    "MAE": np.nan,
                    "MedianAE": np.nan,
                    "RMSE": np.nan,
                    "mean_actual_views": np.nan,
                    "mean_predicted_views": np.nan,
                }
            )
            continue
        actual = subset["observed_views"].to_numpy(dtype=np.float64)
        pred = subset["predicted_views"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "bucket": label,
                "count": int(len(subset)),
                "MAE": float(mean_absolute_error(actual, pred)),
                "MedianAE": float(median_absolute_error(actual, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                "mean_actual_views": float(actual.mean()),
                "mean_predicted_views": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def region_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[["region", "observed_views", "predicted_views", "absolute_error"]].copy()
    data["squared_error"] = data["absolute_error"] ** 2
    grouped = data.groupby("region", sort=True).agg(
        observations=("observed_views", "size"),
        MAE=("absolute_error", "mean"),
        MedianAE=("absolute_error", "median"),
        RMSE=("squared_error", lambda s: np.sqrt(s.mean())),
        mean_actual_views=("observed_views", "mean"),
        mean_predicted_views=("predicted_views", "mean"),
    )
    return grouped.reset_index()


def category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[["category_name", "observed_views", "predicted_views", "absolute_error"]].copy()
    data["squared_error"] = data["absolute_error"] ** 2
    grouped = data.groupby("category_name", sort=True).agg(
        observations=("observed_views", "size"),
        MAE=("absolute_error", "mean"),
        MedianAE=("absolute_error", "median"),
        RMSE=("squared_error", lambda s: np.sqrt(s.mean())),
        mean_actual_views=("observed_views", "mean"),
        mean_predicted_views=("predicted_views", "mean"),
    )
    return grouped.reset_index()


def monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[["trending_date", "observed_views", "predicted_views", "absolute_error"]].copy()
    data["month"] = pd.to_datetime(data["trending_date"], format="%y.%d.%m", errors="raise").dt.to_period("M").astype(str)
    data["squared_error"] = data["absolute_error"] ** 2
    grouped = data.groupby("month", sort=True).agg(
        observations=("observed_views", "size"),
        MAE=("absolute_error", "mean"),
        MedianAE=("absolute_error", "median"),
        RMSE=("squared_error", lambda s: np.sqrt(s.mean())),
        mean_actual_views=("observed_views", "mean"),
        mean_predicted_views=("predicted_views", "mean"),
    )
    return grouped.reset_index()


def target_distribution_by_split(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, frame in split_frames.items():
        actual = frame["observed_views"].to_numpy(dtype=np.float64)
        actual_log = np.log1p(actual)
        rows.append(
            {
                "split": split_name,
                "median_views": float(np.median(actual)),
                "mean_views": float(np.mean(actual)),
                "p95_views": float(np.quantile(actual, 0.95)),
                "p99_views": float(np.quantile(actual, 0.99)),
                "max_views": float(np.max(actual)),
                "median_log_views": float(np.median(actual_log)),
                "mean_log_views": float(np.mean(actual_log)),
            }
        )
    return pd.DataFrame(rows)


def distribution_shift(frame: pd.DataFrame, split_name: str, column: str) -> pd.Series:
    counts = frame[column].value_counts(normalize=True)
    return counts.rename(split_name)


def systematic_bias(frame: pd.DataFrame) -> pd.DataFrame:
    bucketed = frame.copy()
    bucketed["bucket"] = bucketed["observed_views"].map(bucket_for_value)
    grouped = bucketed.groupby("bucket", sort=False).agg(
        mean_actual=("observed_views", "mean"),
        mean_predicted=("predicted_views", "mean"),
        mean_error=("predicted_views", lambda s: np.mean(s - bucketed.loc[s.index, "observed_views"])),
        median_error=("predicted_views", lambda s: np.median(s - bucketed.loc[s.index, "observed_views"])),
    )
    # Use a direct per-row calculation to avoid the lambda mismatch on grouped series.
    bucketed["error"] = bucketed["predicted_views"] - bucketed["observed_views"]
    grouped = bucketed.groupby("bucket", sort=False).agg(
        mean_actual=("observed_views", "mean"),
        mean_predicted=("predicted_views", "mean"),
        mean_error=("error", "mean"),
        median_error=("error", "median"),
    )
    return grouped.reset_index()


def save_feature_importance(model, preprocessor) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = list(preprocessor.get_feature_names_out())
    importances = model.named_steps["regressor"].feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["source_feature"] = df["feature"].str.split("__", n=1).str[-1]
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    top = df.head(20).copy()
    top.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return top


def save_plot(path: Path, func, *args, **kwargs):
    func(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    split_frames = load_splits()
    metadata = json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))
    print("SPLIT VERIFICATION")
    print(json.dumps(split_summary(split_frames), indent=2))
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = set(split_frames[left]["video_id"]).intersection(split_frames[right]["video_id"])
        print(f"{left} vs {right} overlap count: {len(overlap)}")
    train_date_max = pd.to_datetime(split_frames["train"]["trending_date"], format="%y.%d.%m", errors="raise").max()
    validation_date_min = pd.to_datetime(split_frames["validation"]["trending_date"], format="%y.%d.%m", errors="raise").min()
    validation_date_max = pd.to_datetime(split_frames["validation"]["trending_date"], format="%y.%d.%m", errors="raise").max()
    test_date_min = pd.to_datetime(split_frames["test"]["trending_date"], format="%y.%d.%m", errors="raise").min()
    print(f"train max date: {train_date_max.date()}")
    print(f"validation min date: {validation_date_min.date()}")
    print(f"validation max date: {validation_date_max.date()}")
    print(f"test min date: {test_date_min.date()}")
    print(f"chronological order valid: {train_date_max < validation_date_min and validation_date_max < test_date_min}")

    for split_name in ("train", "validation", "test"):
        split_frames[split_name] = add_prediction_columns(split_frames[split_name], model)

    for split_name in ("train", "validation", "test"):
        actual = split_frames[split_name]["observed_views"].to_numpy(dtype=np.float64)
        predicted = split_frames[split_name]["predicted_views"].to_numpy(dtype=np.float64)
        print(f"\n{split_name.upper()} metrics (original scale)")
        print(json.dumps(metric_summary(actual, predicted), indent=2))
        print(f"\n{split_name.upper()} metrics (log scale)")
        print(json.dumps(log_metric_summary(actual, predicted), indent=2))

    valid = split_frames["validation"].copy()
    test = split_frames["test"].copy()
    val_bucket = bucket_summary(valid)
    test_bucket = bucket_summary(test)
    print("\nVALIDATION bucket metrics")
    print(val_bucket.to_string(index=False))
    print("\nTEST bucket metrics")
    print(test_bucket.to_string(index=False))

    print("\nTEST region metrics")
    print(region_summary(test).sort_values("MAE", ascending=True).to_string(index=False))
    print("\nTEST category metrics")
    print(category_summary(test).sort_values("MAE", ascending=True).head(20).to_string(index=False))
    print("\nTEST monthly metrics")
    print(monthly_summary(test).to_string(index=False))

    target_dist = target_distribution_by_split(split_frames)
    print("\nTARGET DISTRIBUTION BY SPLIT")
    print(target_dist.to_string(index=False))

    bias_summary = systematic_bias(test)
    print("\nSYSTEMATIC BIAS BY VIEW BUCKET (TEST)")
    print(bias_summary.to_string(index=False))

    top_features = save_feature_importance(model, model.named_steps["preprocess"])
    print("\nTOP FEATURE IMPORTANCE")
    print(top_features.to_string(index=False))

    # Plots
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    # Actual vs predicted original scale
    for split_name in ("validation", "test"):
        frame = split_frames[split_name]
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.scatterplot(x=frame["observed_views"], y=frame["predicted_views"], alpha=0.25, ax=ax)
        mx = np.linspace(frame["observed_views"].min(), frame["observed_views"].max(), 200)
        ax.plot(mx, mx, color="black", linestyle="--", linewidth=1.2, label="y=x")
        ax.set_title(f"Actual vs predicted views ({split_name})")
        ax.set_xlabel("Actual views")
        ax.set_ylabel("Predicted views")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"actual_vs_predicted_{split_name}.png", dpi=200)
        plt.close(fig)

    # Actual vs predicted log scale
    for split_name in ("validation", "test"):
        frame = split_frames[split_name]
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.scatterplot(x=np.log1p(frame["observed_views"]), y=np.log1p(frame["predicted_views"]), alpha=0.25, ax=ax)
        mx = np.linspace(np.log1p(frame["observed_views"]).min(), np.log1p(frame["observed_views"]).max(), 200)
        ax.plot(mx, mx, color="black", linestyle="--", linewidth=1.2, label="y=x")
        ax.set_title(f"Actual vs predicted log views ({split_name})")
        ax.set_xlabel("log1p(actual views)")
        ax.set_ylabel("log1p(predicted views)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"actual_vs_predicted_log_{split_name}.png", dpi=200)
        plt.close(fig)

    # Residual distribution on log scale
    for split_name in ("validation", "test"):
        frame = split_frames[split_name]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(frame["residual_log"], bins=40, color="steelblue", ax=ax)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"Log residual distribution ({split_name})")
        ax.set_xlabel("log(predicted) - log(actual)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"residual_distribution_log_{split_name}.png", dpi=200)
        plt.close(fig)

    # Error by bucket
    bucket_plot_data = []
    for split_name in ("validation", "test"):
        frame = split_frames[split_name]
        frame = frame.copy()
        frame["bucket"] = frame["observed_views"].map(bucket_for_value)
        for bucket in bucket_labels():
            subset = frame[frame["bucket"] == bucket]
            if subset.empty:
                continue
            bucket_plot_data.append(
                {
                    "split": split_name,
                    "bucket": bucket,
                    "MAE": mean_absolute_error(subset["observed_views"], subset["predicted_views"]),
                    "count": len(subset),
                }
            )
    bucket_df = pd.DataFrame(bucket_plot_data)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=bucket_df, x="bucket", y="MAE", hue="split", ax=ax)
    ax.set_title("MAE by actual-view bucket")
    ax.set_xlabel("Actual view bucket")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_by_bucket.png", dpi=200)
    plt.close(fig)

    # Chronological test period predictions
    test_frame = split_frames["test"].copy()
    test_frame["trending_date"] = pd.to_datetime(test_frame["trending_date"], format="%y.%d.%m", errors="raise")
    daily = test_frame.groupby("trending_date", as_index=False).agg(
        actual_views=("observed_views", "mean"),
        predicted_views=("predicted_views", "mean"),
    ).sort_values("trending_date")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily["trending_date"], daily["actual_views"], label="Actual mean views", color="tab:blue")
    ax.plot(daily["trending_date"], daily["predicted_views"], label="Predicted mean views", color="tab:orange")
    ax.set_title("Predicted vs actual views over the chronological test period")
    ax.set_xlabel("Trending date")
    ax.set_ylabel("Mean views")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual_test_period.png", dpi=200)
    plt.close(fig)

    # Feature importance bar chart
    top_features = save_feature_importance(model, model.named_steps["preprocess"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_features, x="importance", y="source_feature", orient="h", ax=ax, color="steelblue")
    ax.set_title("Top 20 feature importances (RandomForestRegressor)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_feature_importance.png", dpi=200)
    plt.close(fig)

    print("\nSaved diagnostic artifacts to:")
    print(REPORTS_DIR)
    print(FIG_DIR)


if __name__ == "__main__":
    main()
