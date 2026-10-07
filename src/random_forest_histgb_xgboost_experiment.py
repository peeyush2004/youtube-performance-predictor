"""One controlled XGBoost baseline against saved Group B model artifacts."""

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
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from group_a_vs_group_b_experiment import GROUP_B_FEATURES, inverse_log_predictions
from random_forest_vs_histgradientboosting_experiment import verify_split
from train_baseline import FORBIDDEN_FEATURES, GROUP_A_CATEGORICAL, GROUP_A_NUMERIC


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RF_MODEL = MODELS / "group_b_random_forest_pipeline.joblib"
HGB_MODEL = MODELS / "group_b_histgradientboosting_pipeline.joblib"
XGB_MODEL = MODELS / "group_b_xgboost_pipeline.joblib"
XGB_METADATA = MODELS / "group_b_xgboost_metadata.json"
RF_SAVED_RESULTS = REPORTS / "group_a_vs_group_b_results.csv"
HGB_SAVED_RESULTS = REPORTS / "random_forest_vs_histgradientboosting_results.csv"
RESULTS_PATH = REPORTS / "rf_histgb_xgb_results.csv"
XGB_BUCKET_PATH = REPORTS / "xgb_bucket_metrics.csv"
BUCKET_COMPARISON_PATH = REPORTS / "rf_histgb_xgb_bucket_comparison.csv"

CATEGORICAL = list(GROUP_A_CATEGORICAL)
NUMERIC = list(GROUP_A_NUMERIC) + ["days_to_trending"]
FEATURES = CATEGORICAL + NUMERIC
assert FEATURES == GROUP_B_FEATURES
assert not FORBIDDEN_FEATURES.intersection(FEATURES)

PARAMETERS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 1,
}
DISJOINT_BUCKETS = ["<=100K", ">100K to <=1M", ">1M to <=10M", ">10M"]


def preprocessor() -> ColumnTransformer:
    # Sparse one-hot encoding preserves feature semantics and avoids a large dense allocation.
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                CATEGORICAL,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=1.0,
    )


def metric_dict(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
        "MedianAE": float(median_absolute_error(actual, pred)),
    }


def actual_bucket(values: np.ndarray) -> np.ndarray:
    return np.select(
        [values <= 100_000, values <= 1_000_000, values <= 10_000_000],
        DISJOINT_BUCKETS[:3],
        default=DISJOINT_BUCKETS[3],
    )


def bucket_rows(model_name: str, actual: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    bucket = actual_bucket(actual)
    errors = pred - actual
    abs_errors = np.abs(errors)
    total_abs = float(abs_errors.sum())
    rows: list[dict[str, object]] = []
    for name in DISJOINT_BUCKETS:
        mask = bucket == name
        a, p, e, ae = actual[mask], pred[mask], errors[mask], abs_errors[mask]
        rows.append({
            "model": model_name,
            "bucket": name,
            "row_count": int(mask.sum()),
            "row_pct": float(mask.mean() * 100),
            "mean_actual_views": float(a.mean()),
            "mean_predicted_views": float(p.mean()),
            "mean_predicted_to_actual_ratio": float(p.mean() / a.mean()),
            "MAE": float(ae.mean()),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MedianAE": float(np.median(ae)),
            "log_MAE": float(np.mean(np.abs(np.log1p(p) - np.log1p(a)))),
            "mean_signed_error": float(e.mean()),
            "median_signed_error": float(np.median(e)),
            "sum_absolute_error": float(ae.sum()),
            "total_absolute_error_contribution_pct": float(ae.sum() / total_abs * 100),
        })
    return rows


def load_saved_metric_results() -> pd.DataFrame:
    rf_raw = pd.read_csv(RF_SAVED_RESULTS)
    rf = rf_raw[rf_raw["feature_group"] == "Group B"].pivot_table(
        index=["split", "scale"], columns="metric", values="value", aggfunc="first"
    ).reset_index()
    rf["model"] = "RandomForest Group B"
    rf = rf[["model", "split", "scale", "MAE", "RMSE", "R2", "MedianAE"]]

    hgb = pd.read_csv(HGB_SAVED_RESULTS)
    hgb = hgb.rename(columns={"model": "model"})
    hgb = hgb[["model", "split", "scale", "MAE", "RMSE", "R2", "MedianAE"]]
    combined = pd.concat([rf, hgb], ignore_index=True)
    # The saved Group A/B result CSV can contain repeated identical records.
    # Collapse only exact duplicate model/split/scale rows and reject conflicts.
    keys = ["model", "split", "scale"]
    for _, group in combined.groupby(keys, dropna=False):
        if any(group[column].nunique(dropna=False) > 1 for column in ("MAE", "RMSE", "R2", "MedianAE")):
            raise ValueError("Conflicting duplicate metric rows in saved experiment results")
    return combined.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def verify_saved_model_metrics(model_name: str, metrics: pd.DataFrame) -> None:
    saved = load_saved_metric_results()
    reproduced = metrics[(metrics["model"] == model_name) & (metrics["split"].isin(["validation", "test"]))]
    reference = saved[(saved["model"] == model_name) & (saved["split"].isin(["validation", "test"]))]
    for _, row in reproduced.iterrows():
        baseline = reference[(reference["split"] == row["split"]) & (reference["scale"] == row["scale"])]
        for key in ("MAE", "RMSE", "R2", "MedianAE"):
            expected = float(baseline.iloc[0][key])
            if not np.isclose(float(row[key]), expected, rtol=1e-9, atol=1e-6):
                raise AssertionError(f"{model_name} {row['split']} {row['scale']} {key}: {row[key]} != {expected}")


def save_plots(test: pd.DataFrame, actual: np.ndarray, predictions: dict[str, np.ndarray], buckets: pd.DataFrame, metrics: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    xgb_pred = predictions["XGBoost Group B"]
    actual_log = np.log1p(actual)
    predicted_log = np.log1p(xgb_pred)
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(len(actual), size=min(30_000, len(actual)), replace=False))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.maximum(actual[sample], 1), np.maximum(xgb_pred[sample], 1), s=7, alpha=0.2)
    hi = max(float(actual.max()), float(xgb_pred.max()))
    ax.plot([1, hi], [1, hi], "k--", linewidth=1)
    ax.set(xscale="log", yscale="log", xlabel="Actual views (log axis)", ylabel="Predicted views (log axis)", title="XGBoost: actual vs predicted test views")
    fig.tight_layout(); fig.savefig(FIGURES / "xgb_actual_vs_predicted_test.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual_log[sample], predicted_log[sample], s=7, alpha=0.2)
    hi = max(float(actual_log.max()), float(predicted_log.max()))
    ax.plot([0, hi], [0, hi], "k--", linewidth=1)
    ax.set(xlabel="log1p(actual views)", ylabel="log1p(predicted views)", title="XGBoost: actual vs predicted log1p test views")
    fig.tight_layout(); fig.savefig(FIGURES / "xgb_actual_vs_predicted_log_test.png", dpi=180); plt.close(fig)

    residual = predicted_log - actual_log
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual, bins=70, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="log1p(predicted views) - log1p(actual views)", ylabel="Observations", title="XGBoost log residuals on test observations")
    fig.tight_layout(); fig.savefig(FIGURES / "xgb_residual_distribution_log_test.png", dpi=180); plt.close(fig)

    xgb_bucket = buckets[buckets["model"] == "XGBoost Group B"].set_index("bucket").loc[DISJOINT_BUCKETS]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(DISJOINT_BUCKETS, xgb_bucket["MAE"])
    ax.set(ylabel="MAE (views)", title="XGBoost test MAE by actual-view range")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    fig.tight_layout(); fig.savefig(FIGURES / "xgb_mae_by_view_range.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in buckets.groupby("model", sort=False):
        group = group.set_index("bucket").loc[DISJOINT_BUCKETS]
        ax.plot(DISJOINT_BUCKETS, group["MAE"], marker="o", label=model_name)
    ax.set(ylabel="MAE (views)", title="Group B test MAE by actual-view range")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6)); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "rf_vs_histgb_vs_xgb_bucket_mae.png", dpi=180); plt.close(fig)

    test_metrics = metrics[(metrics["split"] == "test") & (metrics["scale"] == "original")].set_index("model")
    plot_data = test_metrics.loc[["RandomForest Group B", "HistGradientBoosting Group B", "XGBoost Group B"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(plot_data))
    width = 0.36
    ax.bar(positions - width / 2, plot_data["MAE"], width, label="MAE")
    ax.bar(positions + width / 2, plot_data["RMSE"], width, label="RMSE")
    ax.set_xticks(positions, plot_data.index)
    ax.set(ylabel="Views", title="Group B model test MAE and RMSE comparison")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6)); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "rf_vs_histgb_vs_xgb_metrics.png", dpi=180); plt.close(fig)


def main() -> None:
    expected_outputs = (
        XGB_MODEL, XGB_METADATA, RESULTS_PATH, XGB_BUCKET_PATH, BUCKET_COMPARISON_PATH,
        FIGURES / "xgb_actual_vs_predicted_test.png",
        FIGURES / "xgb_actual_vs_predicted_log_test.png",
        FIGURES / "xgb_residual_distribution_log_test.png",
        FIGURES / "xgb_mae_by_view_range.png",
        FIGURES / "rf_vs_histgb_vs_xgb_bucket_mae.png",
        FIGURES / "rf_vs_histgb_vs_xgb_metrics.png",
    )
    for p in expected_outputs:
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {p}")

    frames = {name: pd.read_csv(SPLITS / f"{name}.csv", low_memory=False)
              for name in ("train", "validation", "test")}
    split_info = verify_split(frames)
    for name, frame in frames.items():
        if not set(FEATURES + ["log_observed_views", "observed_views"]).issubset(frame.columns):
            raise ValueError(f"{name} split is missing required columns")
        if FORBIDDEN_FEATURES.intersection(FEATURES):
            raise ValueError("Forbidden feature found in Group B")

    train, validation, test = frames["train"], frames["validation"], frames["test"]
    transformer = preprocessor()
    train_matrix = transformer.fit_transform(train[FEATURES])
    if not sparse.issparse(train_matrix):
        raise TypeError("Expected sparse transformed feature matrix; refusing dense XGBoost input")
    dense_bytes = len(train) * train_matrix.shape[1] * np.dtype(np.float64).itemsize
    sparse_bytes = train_matrix.data.nbytes + train_matrix.indices.nbytes + train_matrix.indptr.nbytes
    print(f"Transformed train shape={train_matrix.shape}; sparse bytes={sparse_bytes:,}; hypothetical float64 dense bytes={dense_bytes:,} ({dense_bytes / 1024**2:.1f} MiB)")

    model = XGBRegressor(**PARAMETERS)
    pipeline = Pipeline([("preprocess", transformer), ("regressor", model)])
    # Reuse the already-fitted train-only transformer instead of fitting it twice.
    started = time.perf_counter()
    pipeline.named_steps["regressor"].fit(train_matrix, train["log_observed_views"].to_numpy(dtype=np.float64))
    training_seconds = time.perf_counter() - started

    split_predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []
    metric_rows.extend(load_saved_metric_results().to_dict("records"))
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        transformed = train_matrix if split_name == "train" else transformer.transform(frame[FEATURES])
        if not sparse.issparse(transformed):
            raise TypeError(f"Expected sparse {split_name} matrix")
        pred_views = inverse_log_predictions(pipeline.named_steps["regressor"].predict(transformed))
        split_predictions[split_name] = pred_views
        actual = frame["observed_views"].to_numpy(dtype=np.float64)
        for scale, truth, prediction in (
            ("original", actual, pred_views),
            ("log", np.log1p(actual), np.log1p(pred_views)),
        ):
            row = {"model": "XGBoost Group B", "split": split_name, "scale": scale,
                   **metric_dict(truth, prediction)}
            metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    rf_pipeline = joblib.load(RF_MODEL, mmap_mode="r")
    hgb_pipeline = joblib.load(HGB_MODEL, mmap_mode="r")
    rf_pred = inverse_log_predictions(rf_pipeline.predict(test[FEATURES]))
    hgb_pred = inverse_log_predictions(hgb_pipeline.predict(test[FEATURES]))
    actual_test = test["observed_views"].to_numpy(dtype=np.float64)
    predictions = {
        "RandomForest Group B": rf_pred,
        "HistGradientBoosting Group B": hgb_pred,
        "XGBoost Group B": split_predictions["test"],
    }
    # Verify reference artifact inference against the already saved metrics.
    for name, pred in (("RandomForest Group B", rf_pred), ("HistGradientBoosting Group B", hgb_pred)):
        reproduced_mae = mean_absolute_error(actual_test, pred)
        saved = metrics[(metrics["model"] == name) & (metrics["split"] == "test") & (metrics["scale"] == "original")]
        if not np.isclose(reproduced_mae, float(saved.iloc[0]["MAE"]), rtol=1e-9, atol=1e-6):
            raise AssertionError(f"{name} test MAE did not reproduce saved result")

    buckets = pd.DataFrame([row for name, pred in predictions.items()
                            for row in bucket_rows(name, actual_test, pred)])
    xgb_buckets = buckets[buckets["model"] == "XGBoost Group B"].copy()
    mae_val_test = metrics[(metrics["scale"] == "original") & (metrics["MAE"].notna())].pivot(
        index="model", columns="split", values="MAE")
    metrics["test_to_validation_MAE_ratio"] = metrics["model"].map(
        (mae_val_test["test"] / mae_val_test["validation"]).to_dict())

    summary: list[dict[str, object]] = []
    for name in predictions:
        test_row = metrics[(metrics["model"] == name) & (metrics["split"] == "test") & (metrics["scale"] == "original")].iloc[0]
        log_row = metrics[(metrics["model"] == name) & (metrics["split"] == "test") & (metrics["scale"] == "log")].iloc[0]
        high = buckets[(buckets["model"] == name) & (buckets["bucket"] == ">10M")].iloc[0]
        over_1m = actual_test > 1_000_000
        over_10m = actual_test > 10_000_000
        abs_error = np.abs(predictions[name] - actual_test)
        summary.append({
            "model": name,
            "test_MAE": test_row["MAE"],
            "test_RMSE": test_row["RMSE"],
            "test_R2": test_row["R2"],
            "test_log_MAE": log_row["MAE"],
            "over_1M_total_absolute_error_contribution_pct": float(abs_error[over_1m].sum() / abs_error.sum() * 100),
            "over_10M_total_absolute_error_contribution_pct": float(abs_error[over_10m].sum() / abs_error.sum() * 100),
            "over_10M_MAE": high["MAE"],
            "over_10M_mean_signed_error": high["mean_signed_error"],
            "over_10M_mean_predicted_to_actual_ratio": high["mean_predicted_to_actual_ratio"],
            "validation_MAE": mae_val_test.loc[name, "validation"],
            "test_to_validation_MAE_ratio": mae_val_test.loc[name, "test"] / mae_val_test.loc[name, "validation"],
        })
    summary_df = pd.DataFrame(summary)

    FIGURES.mkdir(parents=True, exist_ok=True)
    save_plots(test, actual_test, predictions, buckets, metrics)
    metrics.to_csv(RESULTS_PATH, index=False)
    xgb_buckets.to_csv(XGB_BUCKET_PATH, index=False)
    buckets.to_csv(BUCKET_COMPARISON_PATH, index=False)

    joblib.dump(pipeline, XGB_MODEL, compress=3)
    metadata = {
        "model": "XGBRegressor",
        "xgboost_version": xgboost.__version__,
        "target": "log_observed_views",
        "feature_group": "Group B",
        "feature_list": FEATURES,
        "hyperparameters": PARAMETERS,
        "preprocessing": {
            "categorical": "train-only constant Unknown imputation then OneHotEncoder(handle_unknown='ignore', sparse_output=True)",
            "numeric": "train-only median imputation then StandardScaler",
            "column_transformer_sparse_threshold": 1.0,
            "feature_engineering": "unchanged Group B feature list",
            "matrix_format": "scipy CSR sparse; no dense conversion",
            "estimated_dense_train_matrix_bytes": int(dense_bytes),
            "observed_sparse_train_matrix_bytes": int(sparse_bytes),
        },
        "target_inverse_transform": "Clip predicted log1p views at zero, apply expm1, and enforce non-negative predictions, matching Group B references.",
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
        "metrics_file": "reports/rf_histgb_xgb_results.csv",
        "xgboost_bucket_metrics_file": "reports/xgb_bucket_metrics.csv",
        "all_model_bucket_comparison_file": "reports/rf_histgb_xgb_bucket_comparison.csv",
        "model_path": "models/group_b_xgboost_pipeline.joblib",
        "reference_models_loaded_read_only": [str(RF_MODEL.relative_to(ROOT)), str(HGB_MODEL.relative_to(ROOT))],
        "validation_test_summary": summary_df.to_dict("records"),
    }
    XGB_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"XGBoost version: {xgboost.__version__}")
    print("Reproduced reference metrics for both saved models; no retraining of RF or HistGradientBoosting.")
    print(summary_df.to_string(index=False))
    print("XGBoost bucket results:")
    print(xgb_buckets.to_string(index=False))


if __name__ == "__main__":
    main()
