"""Controlled Group A vs Group B feature comparison using the existing split and baseline model setup."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_PATH = REPORTS_DIR / "group_a_vs_group_b_results.csv"
GROUP_B_IMPORTANCE_PATH = REPORTS_DIR / "group_b_feature_importance.csv"
GROUP_B_MODEL_PATH = PROJECT_ROOT / "models" / "group_b_random_forest_pipeline.joblib"
GROUP_B_METADATA_PATH = PROJECT_ROOT / "models" / "group_b_random_forest_metadata.json"
RANDOM_STATE = 42

from train_baseline import GROUP_A_CATEGORICAL, GROUP_A_NUMERIC  # noqa: E402

GROUP_A_FEATURES = GROUP_A_CATEGORICAL + GROUP_A_NUMERIC
GROUP_B_FEATURES = GROUP_A_FEATURES + ["days_to_trending"]


def verify_split(frames: dict[str, pd.DataFrame]) -> None:
    train_dates = pd.to_datetime(frames["train"]["trending_date"], format="%y.%d.%m", errors="raise")
    valid_dates = pd.to_datetime(frames["validation"]["trending_date"], format="%y.%d.%m", errors="raise")
    test_dates = pd.to_datetime(frames["test"]["trending_date"], format="%y.%d.%m", errors="raise")

    train_max = train_dates.max()
    valid_min = valid_dates.min()
    valid_max = valid_dates.max()
    test_min = test_dates.min()

    assert train_max < valid_min, "Train dates overlap validation start"
    assert valid_max < test_min, "Validation dates overlap test start"

    train_ids = set(frames["train"]["video_id"])
    valid_ids = set(frames["validation"]["video_id"])
    test_ids = set(frames["test"]["video_id"])
    assert not (train_ids & valid_ids)
    assert not (train_ids & test_ids)
    assert not (valid_ids & test_ids)


def build_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def inverse_log_predictions(predicted_log: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(predicted_log, dtype=np.float64), 0.0, np.log(np.finfo(np.float64).max) - 1.0)
    view_predictions = np.expm1(clipped)
    return np.maximum(view_predictions, 0.0)


def compute_metrics(actual_views: pd.Series, predicted_views: np.ndarray) -> dict[str, float]:
    actual = actual_views.to_numpy(dtype=np.float64)
    pred = np.asarray(predicted_views, dtype=np.float64)
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
        "MedianAE": float(median_absolute_error(actual, pred)),
    }


def compute_log_metrics(actual_views: pd.Series, predicted_views: np.ndarray) -> dict[str, float]:
    actual = actual_views.to_numpy(dtype=np.float64)
    pred = np.asarray(predicted_views, dtype=np.float64)
    actual_log = np.log1p(actual)
    pred_log = np.log1p(pred)
    return {
        "MAE": float(mean_absolute_error(actual_log, pred_log)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_log, pred_log))),
        "R2": float(r2_score(actual_log, pred_log)),
        "MedianAE": float(median_absolute_error(actual_log, pred_log)),
    }


def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_feature_group(
    frames: dict[str, pd.DataFrame],
    label: str,
) -> tuple[pd.DataFrame, Pipeline]:
    if label == "Group A":
        categorical = GROUP_A_CATEGORICAL
        numeric = GROUP_A_NUMERIC
    else:
        categorical = GROUP_A_CATEGORICAL
        numeric = GROUP_A_NUMERIC + ["days_to_trending"]
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(categorical, numeric)),
            ("regressor", make_model()),
        ]
    )
    train_X = frames["train"][categorical + numeric]
    train_y = frames["train"]["log_observed_views"]
    model.fit(train_X, train_y)

    rows: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        split_X = frames[split_name][categorical + numeric]
        predicted_log = model.predict(split_X)
        predicted_views = inverse_log_predictions(predicted_log)
        original = compute_metrics(frames[split_name]["observed_views"], predicted_views)
        logged = compute_log_metrics(frames[split_name]["observed_views"], predicted_views)
        for metric_name, value in original.items():
            rows.append({
                "feature_group": label,
                "split": split_name,
                "scale": "original",
                "metric": metric_name,
                "value": float(value),
            })
        for metric_name, value in logged.items():
            rows.append({
                "feature_group": label,
                "split": split_name,
                "scale": "log",
                "metric": metric_name,
                "value": float(value),
            })
    return pd.DataFrame(rows), model


def compute_percentage_change(before: float, after: float) -> float:
    if before == 0:
        return np.nan
    return ((after - before) / abs(before)) * 100.0


def summarize_days_to_trending(frame: pd.DataFrame) -> dict[str, object]:
    s = pd.to_numeric(frame["days_to_trending"], errors="coerce")
    q = s.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    return {
        "count": int(s.count()),
        "missing": int(s.isna().sum()),
        "min": float(s.min()) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "median": float(s.median()) if s.notna().any() else np.nan,
        "std": float(s.std(ddof=1)) if s.notna().any() and s.count() > 1 else np.nan,
        "p01": float(q.get(0.01, np.nan)),
        "p05": float(q.get(0.05, np.nan)),
        "p25": float(q.get(0.25, np.nan)),
        "p75": float(q.get(0.75, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "negatives": int((s < 0).sum()),
        "zeros": int((s == 0).sum()),
        "greater_than_30": int((s > 30).sum()),
        "greater_than_365": int((s > 365).sum()),
    }


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


def bucket_analysis(frame: pd.DataFrame, predicted_col: str) -> pd.DataFrame:
    bucket_df = frame[["observed_views", predicted_col]].copy()
    bucket_df["bucket"] = bucket_df["observed_views"].map(bucket_for_value)
    rows: list[dict[str, object]] = []
    for bucket in ["<10K", "10K-100K", "100K-1M", "1M-10M", ">10M"]:
        subset = bucket_df[bucket_df["bucket"] == bucket]
        if subset.empty:
            continue
        actual = subset["observed_views"].to_numpy(dtype=np.float64)
        pred = subset[predicted_col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "bucket": bucket,
                "row_count": int(len(subset)),
                "mean_actual_views": float(actual.mean()),
                "mean_predicted_views": float(pred.mean()),
                "MAE": float(mean_absolute_error(actual, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                "mean_signed_error": float(np.mean(pred - actual)),
            }
        )
    return pd.DataFrame(rows)


def region_summary(frame: pd.DataFrame, predicted_col: str) -> pd.DataFrame:
    data = frame[["region", "observed_views", predicted_col]].copy()
    data["absolute_error"] = (data["observed_views"] - data[predicted_col]).abs()
    grouped = data.groupby("region").agg(
        observations=("observed_views", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("absolute_error", lambda s: np.sqrt((s ** 2).mean())),
        mean_actual_views=("observed_views", "mean"),
        mean_predicted_views=(predicted_col, "mean"),
    ).reset_index()
    return grouped.sort_values("MAE", ascending=True)


def save_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    feature_importance = model.named_steps["regressor"].feature_importances_
    feature_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance,
    })
    feature_df["source_feature"] = feature_df["feature"].str.split("__", n=1).str[-1]
    feature_df = feature_df.sort_values("importance", ascending=False).reset_index(drop=True)
    # Keep the complete transformed-feature table so the rank is auditable.
    feature_df.to_csv(GROUP_B_IMPORTANCE_PATH, index=False)
    return feature_df


def make_plot_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    make_plot_dir()
    dataset = pd.read_csv(DATASET_PATH, low_memory=False)
    frames = {name: pd.read_csv(SPLIT_DIR / f"{name}.csv", low_memory=False) for name in ("train", "validation", "test")}
    verify_split(frames)

    days_summary = {
        "full": summarize_days_to_trending(dataset),
        "train": summarize_days_to_trending(frames["train"]),
        "validation": summarize_days_to_trending(frames["validation"]),
        "test": summarize_days_to_trending(frames["test"]),
    }
    summary_df = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in days_summary.items()}).T.reset_index().rename(columns={"index": "split"})
    summary_df.to_csv(REPORTS_DIR / "days_to_trending_summary.csv", index=False)

    # days_to_trending distribution figure
    full_days = pd.to_numeric(dataset["days_to_trending"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(full_days, bins=60, kde=True, ax=ax)
    ax.set_title("days_to_trending distribution (full modeling dataset)")
    ax.set_xlabel("days_to_trending")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "days_to_trending_distribution.png", dpi=200)
    plt.close(fig)

    # scatter plot
    sample = dataset[["days_to_trending", "log_observed_views"]].dropna().sample(n=min(50000, len(dataset)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=sample, x="days_to_trending", y="log_observed_views", alpha=0.2, s=8, ax=ax)
    ax.set_title("days_to_trending vs log_observed_views")
    ax.set_xlabel("days_to_trending")
    ax.set_ylabel("log_observed_views")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "days_to_trending_vs_log_views.png", dpi=200)
    plt.close(fig)

    full_corr = dataset["days_to_trending"].corr(dataset["log_observed_views"])
    full_spear = dataset["days_to_trending"].corr(dataset["log_observed_views"], method="spearman")
    print("Full dataset corr (Pearson):", full_corr)
    print("Full dataset corr (Spearman):", full_spear)

    results_by_group: list[dict[str, object]] = []
    trained_models: dict[str, Pipeline] = {}
    for label in ("Group A", "Group B"):
        metrics_df, model = evaluate_feature_group(frames, label)
        results_by_group.append(metrics_df)
        trained_models[label] = model

    results_df = pd.concat(results_by_group, ignore_index=True)
    results_df = results_df.sort_values(["feature_group", "split", "scale", "metric"]).reset_index(drop=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    validation_a = results_df[(results_df["feature_group"] == "Group A") & (results_df["split"] == "validation") & (results_df["scale"] == "original")].set_index("metric")["value"]
    validation_b = results_df[(results_df["feature_group"] == "Group B") & (results_df["split"] == "validation") & (results_df["scale"] == "original")].set_index("metric")["value"]
    test_a = results_df[(results_df["feature_group"] == "Group A") & (results_df["split"] == "test") & (results_df["scale"] == "original")].set_index("metric")["value"]
    test_b = results_df[(results_df["feature_group"] == "Group B") & (results_df["split"] == "test") & (results_df["scale"] == "original")].set_index("metric")["value"]

    comparison = pd.DataFrame({
        "group_a_validation": validation_a,
        "group_b_validation": validation_b,
        "group_a_test": test_a,
        "group_b_test": test_b,
    })
    comparison["val_MAE_pct_change"] = comparison.apply(lambda row: compute_percentage_change(row["group_a_validation"], row["group_b_validation"]) if row.name == "MAE" else np.nan, axis=1)
    comparison["test_MAE_pct_change"] = comparison.apply(lambda row: compute_percentage_change(row["group_a_test"], row["group_b_test"]) if row.name == "MAE" else np.nan, axis=1)
    comparison["val_RMSE_pct_change"] = comparison.apply(lambda row: compute_percentage_change(row["group_a_validation"], row["group_b_validation"]) if row.name == "RMSE" else np.nan, axis=1)
    comparison["test_RMSE_pct_change"] = comparison.apply(lambda row: compute_percentage_change(row["group_a_test"], row["group_b_test"]) if row.name == "RMSE" else np.nan, axis=1)

    # Save a compact comparison report in addition to the long-form table.
    compact = pd.DataFrame({
        "metric": ["MAE", "RMSE", "R2", "MedianAE"],
        "group_a_validation": [validation_a[metric] for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
        "group_b_validation": [validation_b[metric] for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
        "validation_absolute_change": [validation_b[metric] - validation_a[metric] for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
        "validation_pct_change": [compute_percentage_change(validation_a[metric], validation_b[metric]) if metric in {"MAE", "RMSE"} else np.nan for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
        "group_a_test": [test_a[metric] for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
        "group_b_test": [test_b[metric] for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
        "test_absolute_change": [test_b[metric] - test_a[metric] for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
        "test_pct_change": [compute_percentage_change(test_a[metric], test_b[metric]) if metric in {"MAE", "RMSE"} else np.nan for metric in ["MAE", "RMSE", "R2", "MedianAE"]],
    })
    compact.to_csv(REPORTS_DIR / "group_a_vs_group_b_summary.csv", index=False)

    # Bucket comparison by test set
    group_a_bucket = bucket_analysis(frames["test"].copy(), "predicted_views_group_a") if False else None
    gA_pred = trained_models["Group A"].predict(frames["test"][GROUP_A_CATEGORICAL + GROUP_A_NUMERIC])
    gB_pred = trained_models["Group B"].predict(frames["test"][GROUP_A_CATEGORICAL + GROUP_A_NUMERIC + ["days_to_trending"]])
    test_a_frame = frames["test"].copy()
    test_b_frame = frames["test"].copy()
    test_a_frame["predicted_views_group_a"] = inverse_log_predictions(gA_pred)
    test_b_frame["predicted_views_group_b"] = inverse_log_predictions(gB_pred)
    bucket_a = bucket_analysis(test_a_frame, "predicted_views_group_a")
    bucket_b = bucket_analysis(test_b_frame, "predicted_views_group_b")
    bucket_a["feature_group"] = "Group A"
    bucket_b["feature_group"] = "Group B"
    bucket_compare = pd.concat([bucket_a, bucket_b], ignore_index=True)
    bucket_compare.to_csv(REPORTS_DIR / "group_a_vs_group_b_bucket_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=bucket_compare, x="bucket", y="MAE", hue="feature_group", ax=ax)
    ax.set_title("Test MAE by actual-view bucket (Group A vs Group B)")
    ax.set_xlabel("Actual view bucket")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "group_a_vs_group_b_bucket_mae.png", dpi=200)
    plt.close(fig)

    # MAE and RMSE comparison plots
    summary_original = results_df[(results_df["scale"] == "original") & (results_df["split"].isin(["validation", "test"]))]
    summary_original = summary_original[summary_original["metric"].isin(["MAE", "RMSE"])].pivot_table(
        index=["split", "metric"], columns="feature_group", values="value", aggfunc="first"
    ).reset_index()
    # Build separate MAE/RMSE plots via melted data
    plot_df = summary_original.melt(id_vars=["split", "metric"], var_name="feature_group", value_name="value")
    for metric_name in ["MAE", "RMSE"]:
        metric_df = plot_df[plot_df["metric"] == metric_name].copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=metric_df, x="split", y="value", hue="feature_group", ax=ax)
        ax.set_title(f"{metric_name} by split and feature group")
        ax.set_xlabel("Split")
        ax.set_ylabel(metric_name)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"group_a_vs_group_b_{metric_name.lower()}.png", dpi=200)
        plt.close(fig)

    # Region comparison
    region_a = region_summary(test_a_frame, "predicted_views_group_a")
    region_b = region_summary(test_b_frame, "predicted_views_group_b")
    region_compare = region_a[["region", "MAE"]].rename(columns={"MAE": "group_a_MAE"}).merge(
        region_b[["region", "MAE"]].rename(columns={"MAE": "group_b_MAE"}),
        on="region",
        how="outer",
    )
    region_compare["delta_MAE"] = region_compare["group_b_MAE"] - region_compare["group_a_MAE"]
    region_compare.to_csv(REPORTS_DIR / "group_a_vs_group_b_region_comparison.csv", index=False)

    # Feature importance for Group B and save csv
    feature_names = trained_models["Group B"].named_steps["preprocess"].get_feature_names_out()
    importance_df = save_feature_importance(trained_models["Group B"], list(feature_names))
    importance_df.to_csv(REPORTS_DIR / "group_b_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    top = importance_df.head(20).copy()
    sns.barplot(data=top, x="importance", y="source_feature", orient="h", ax=ax, color="steelblue")
    ax.set_title("Top 20 Group B feature importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "group_b_feature_importance.png", dpi=200)
    plt.close(fig)

    group_b_rank = int(importance_df[importance_df["source_feature"] == "days_to_trending"].index[0] + 1) if (importance_df["source_feature"] == "days_to_trending").any() else None
    print("days_to_trending rank in Group B importance:", group_b_rank)

    # Save model artifact and metadata
    joblib.dump(trained_models["Group B"], GROUP_B_MODEL_PATH)
    metadata = {
        "model": "RandomForestRegressor",
        "target": "log_observed_views",
        "feature_group": "Group B",
        "feature_list": GROUP_B_FEATURES,
        "split": {
            "train": {"rows": int(len(frames["train"])), "date_range": [frames["train"]["trending_date"].min(), frames["train"]["trending_date"].max()]},
            "validation": {"rows": int(len(frames["validation"])), "date_range": [frames["validation"]["trending_date"].min(), frames["validation"]["trending_date"].max()]},
            "test": {"rows": int(len(frames["test"])), "date_range": [frames["test"]["trending_date"].min(), frames["test"]["trending_date"].max()]},
        },
        "metrics": results_df[(results_df["feature_group"] == "Group B")].to_dict(orient="records"),
        "days_to_trending_rank": group_b_rank,
        "days_to_trending_pearson_correlation_with_log_views": float(full_corr),
        "days_to_trending_spearman_correlation_with_log_views": float(full_spear),
    }
    GROUP_B_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Group A vs Group B comparison saved to:", RESULTS_PATH)
    print("Group B importance saved to:", GROUP_B_IMPORTANCE_PATH)
    print("Group B model saved to:", GROUP_B_MODEL_PATH)
    print("Drop-in model metadata saved to:", GROUP_B_METADATA_PATH)


if __name__ == "__main__":
    main()
