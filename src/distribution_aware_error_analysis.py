"""Analyze saved Group B model predictions across actual-view ranges."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from group_a_vs_group_b_experiment import GROUP_B_FEATURES, inverse_log_predictions
from random_forest_vs_histgradientboosting_experiment import verify_split


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RF_MODEL = ROOT / "models" / "group_b_random_forest_pipeline.joblib"
HGB_MODEL = ROOT / "models" / "group_b_histgradientboosting_pipeline.joblib"
RF_REFERENCE = REPORTS / "group_a_vs_group_b_results.csv"
HGB_REFERENCE = REPORTS / "random_forest_vs_histgradientboosting_results.csv"

SUBSETS = [
    ("All test observations", lambda s: np.ones(len(s), dtype=bool)),
    ("<=100K", lambda s: s <= 100_000),
    (">100K to <=1M", lambda s: (s > 100_000) & (s <= 1_000_000)),
    (">1M to <=10M", lambda s: (s > 1_000_000) & (s <= 10_000_000)),
    (">10M", lambda s: s > 10_000_000),
    ("<=1M", lambda s: s <= 1_000_000),
    ("<=10M", lambda s: s <= 10_000_000),
]


def metrics_for_subset(actual: np.ndarray, predicted: np.ndarray, total_abs_error: float) -> dict[str, float | int]:
    errors = predicted - actual
    absolute = np.abs(errors)
    positive = actual > 0
    abs_log = np.abs(np.log1p(predicted) - np.log1p(actual))
    return {
        "row_count": int(len(actual)),
        "test_observation_pct": float(len(actual) / 51_003 * 100),
        "mean_actual_views": float(np.mean(actual)),
        "median_actual_views": float(np.median(actual)),
        "mean_predicted_views": float(np.mean(predicted)),
        "median_predicted_views": float(np.median(predicted)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MedianAE": float(median_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "log_MAE": float(np.mean(abs_log)),
        "mean_signed_error": float(np.mean(errors)),
        "median_signed_error": float(np.median(errors)),
        "mean_predicted_to_actual_ratio": float(np.mean(predicted) / np.mean(actual)),
        "MAPE_pct": float(np.mean(absolute[positive] / actual[positive]) * 100) if positive.any() else float("nan"),
        "median_APE_pct": float(np.median(absolute[positive] / actual[positive]) * 100) if positive.any() else float("nan"),
        "median_absolute_log_error": float(np.median(abs_log)),
        "sum_absolute_error": float(np.sum(absolute)),
        "total_absolute_error_contribution_pct": float(np.sum(absolute) / total_abs_error * 100),
    }


def verify_overall_metrics(model_name: str, predicted: np.ndarray) -> None:
    reference_path = RF_REFERENCE if model_name == "RandomForest Group B" else HGB_REFERENCE
    reference = pd.read_csv(reference_path)
    reference = reference[(reference["model" if "model" in reference.columns else "feature_group"] == model_name)
                          & (reference["split"] == "test") & (reference["scale"] == "original")]
    if reference.empty:
        # The RandomForest reference CSV uses feature_group, whose value is Group B.
        reference = pd.read_csv(reference_path)
        reference = reference[(reference["feature_group"] == "Group B") & (reference["split"] == "test")
                              & (reference["scale"] == "original")]
    saved_mae = float(reference.loc[reference["metric"] == "MAE", "value"].iloc[0]) if "metric" in reference else float(reference["MAE"].iloc[0])
    actual = pd.read_csv(SPLIT_DIR / "test.csv", usecols=["observed_views"])["observed_views"].to_numpy(dtype=np.float64)
    reproduced_mae = float(mean_absolute_error(actual, predicted))
    if not np.isclose(saved_mae, reproduced_mae, rtol=1e-10, atol=1e-6):
        raise AssertionError(f"{model_name} reproduced MAE {reproduced_mae} != saved MAE {saved_mae}")


def main() -> None:
    frames = {name: pd.read_csv(SPLIT_DIR / f"{name}.csv", low_memory=False)
              for name in ("train", "validation", "test")}
    split_info = verify_split(frames)
    test = frames["test"]
    actual = test["observed_views"].to_numpy(dtype=np.float64)
    features = test[GROUP_B_FEATURES]
    models = {
        "RandomForest Group B": joblib.load(RF_MODEL, mmap_mode="r"),
        "HistGradientBoosting Group B": joblib.load(HGB_MODEL, mmap_mode="r"),
    }
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        predictions[name] = inverse_log_predictions(model.predict(features))
        verify_overall_metrics(name, predictions[name])

    rows: list[dict[str, object]] = []
    total_abs = {name: float(np.abs(pred - actual).sum()) for name, pred in predictions.items()}
    for name, pred in predictions.items():
        for subset_name, predicate in SUBSETS + [(">10M (cumulative)", lambda s: s > 10_000_000)]:
            mask = predicate(actual)
            values = metrics_for_subset(actual[mask], pred[mask], total_abs[name])
            rows.append({"model": name, "subset": subset_name, **values})

    all_results = pd.DataFrame(rows)
    rf_results = all_results[all_results["model"] == "RandomForest Group B"].copy()
    rf_results.to_csv(REPORTS / "distribution_aware_error_analysis.csv", index=False)
    all_results.to_csv(REPORTS / "distribution_aware_model_comparison.csv", index=False)
    FIGURES.mkdir(parents=True, exist_ok=True)

    disjoint = ["<=100K", ">100K to <=1M", ">1M to <=10M", ">10M"]
    rf_ranges = rf_results.set_index("subset").loc[disjoint]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(disjoint))
    width = 0.36
    ax.bar(x - width / 2, rf_ranges["MAE"], width, label="MAE")
    ax.bar(x + width / 2, rf_ranges["MedianAE"], width, label="MedianAE")
    ax.set_xticks(x, disjoint)
    ax.set_ylabel("Absolute error (views)")
    ax.set_title("RandomForest test error by actual-view range")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "error_by_view_range_rf.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(disjoint, rf_ranges["total_absolute_error_contribution_pct"])
    ax.set_ylabel("Share of total test absolute error (%)")
    ax.set_title("RandomForest total absolute-error contribution by range")
    fig.tight_layout()
    fig.savefig(FIGURES / "error_contribution_by_view_range_rf.png", dpi=180)
    plt.close(fig)

    compare = all_results[all_results["subset"].isin(disjoint)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, grp in compare.groupby("model", sort=False):
        grp = grp.set_index("subset").loc[disjoint]
        ax.plot(disjoint, grp["MAE"], marker="o", label=model_name)
    ax.set_ylabel("MAE (views)")
    ax.set_title("Group B test MAE by actual-view range")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "rf_vs_histgb_error_by_view_range.png", dpi=180)
    plt.close(fig)

    rf_pred = predictions["RandomForest Group B"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.maximum(actual, 1), np.maximum(rf_pred, 1), s=7, alpha=0.2)
    upper = max(float(actual.max()), float(rf_pred.max()))
    ax.plot([1, upper], [1, upper], color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual views (log axis; zero displayed at 1)")
    ax.set_ylabel("Predicted views (log axis; zero displayed at 1)")
    ax.set_title("Group B RandomForest: actual vs predicted test views")
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_predicted_rf_log_scale.png", dpi=180)
    plt.close(fig)

    print("Split verified:", split_info)
    print("Saved RF metrics reproduced; artifacts loaded without fitting.")
    print(rf_results[["subset", "row_count", "test_observation_pct", "MAE", "MedianAE", "log_MAE",
                      "mean_signed_error", "mean_predicted_to_actual_ratio", "total_absolute_error_contribution_pct"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
