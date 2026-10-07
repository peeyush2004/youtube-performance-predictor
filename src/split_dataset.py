"""Create chronological, video-aware splits for the trending snapshot dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
METADATA_PATH = SPLIT_DIR / "split_metadata.json"
TRAIN_END = pd.Timestamp("2018-04-02")
VALIDATION_END = pd.Timestamp("2018-05-12")
CLASS_NAMES = ("Low", "Medium", "High")


def parse_snapshot_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%y.%d.%m", errors="raise")


def target_summary(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def class_counts(values: pd.Series, low_threshold: float, high_threshold: float) -> dict[str, int]:
    labels = pd.Series(
        np.select(
            [values < low_threshold, values < high_threshold],
            ["Low", "Medium"],
            default="High",
        ),
        index=values.index,
    )
    return {name: int(labels.eq(name).sum()) for name in CLASS_NAMES}


def validate_split_order(split_frames: dict[str, pd.DataFrame]) -> None:
    train_ids = set(split_frames["train"]["video_id"])
    validation_ids = set(split_frames["validation"]["video_id"])
    test_ids = set(split_frames["test"]["video_id"])

    overlaps = {
        "train_validation": train_ids & validation_ids,
        "train_test": train_ids & test_ids,
        "validation_test": validation_ids & test_ids,
    }
    if any(overlaps.values()):
        raise ValueError(f"Video ID overlap detected: { {key: len(value) for key, value in overlaps.items()} }")

    train_max = split_frames["train"]["_snapshot_date"].max()
    validation_min = split_frames["validation"]["_snapshot_date"].min()
    validation_max = split_frames["validation"]["_snapshot_date"].max()
    test_min = split_frames["test"]["_snapshot_date"].min()
    if not train_max < validation_min:
        raise ValueError(f"Chronology violation: max train date {train_max} >= min validation date {validation_min}")
    if not validation_max < test_min:
        raise ValueError(f"Chronology violation: max validation date {validation_max} >= min test date {test_min}")


def build_splits() -> dict[str, pd.DataFrame]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Modeling dataset not found: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH, low_memory=False)
    required = {"video_id", "region", "trending_date", "observed_views", "log_observed_views"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Modeling dataset is missing required columns: {sorted(missing)}")

    dataset["_snapshot_date"] = parse_snapshot_dates(dataset["trending_date"])
    if TRAIN_END not in set(dataset["_snapshot_date"]):
        raise ValueError(f"Train boundary {TRAIN_END.date()} is absent from the snapshot dates")
    if VALIDATION_END not in set(dataset["_snapshot_date"]):
        raise ValueError(f"Validation boundary {VALIDATION_END.date()} is absent from the snapshot dates")
    if not TRAIN_END < VALIDATION_END:
        raise ValueError("Train and validation boundary dates are not ordered")

    video_spans = dataset.groupby("video_id", sort=False)["_snapshot_date"].agg(
        first_snapshot="min",
        last_snapshot="max",
    )
    assignments = pd.Series(index=video_spans.index, dtype="string")
    assignments.loc[video_spans["last_snapshot"] <= TRAIN_END] = "train"
    assignments.loc[
        (video_spans["first_snapshot"] > TRAIN_END)
        & (video_spans["last_snapshot"] <= VALIDATION_END)
    ] = "validation"
    assignments.loc[video_spans["first_snapshot"] > VALIDATION_END] = "test"

    split_names = dataset["video_id"].map(assignments)
    crossing_video_ids = assignments.index[assignments.isna()].astype(str).tolist()
    excluded_mask = split_names.isna()
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        frame = dataset.loc[split_names.eq(split_name)].copy()
        if frame.empty:
            raise ValueError(f"The {split_name} split is empty")
        split_frames[split_name] = frame

    validate_split_order(split_frames)

    training_targets = split_frames["train"]["observed_views"]
    low_threshold = float(training_targets.quantile(1 / 3))
    high_threshold = float(training_targets.quantile(2 / 3))
    if not np.isfinite([low_threshold, high_threshold]).all():
        raise ValueError("Training-only classification thresholds are non-finite")

    for frame in split_frames.values():
        frame.drop(columns="_snapshot_date", inplace=True)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, frame in split_frames.items():
        frame.to_csv(SPLIT_DIR / f"{split_name}.csv", index=False, encoding="utf-8")

    metadata = {
        "source_dataset": "data/processed/modeling_dataset.csv",
        "split_method": "chronological date windows with whole-video assignment",
        "boundary_selection": (
            "Date boundaries were selected after inspecting the observed date distribution "
            "and searching nearby date cuts to approximate 70/15/15 while retaining rows."
        ),
        "train_end_inclusive": TRAIN_END.date().isoformat(),
        "validation_end_inclusive": VALIDATION_END.date().isoformat(),
        "cross_boundary_policy": "Exclude all rows for any video_id whose snapshots span split windows.",
        "excluded_cross_boundary_video_count": len(crossing_video_ids),
        "excluded_cross_boundary_row_count": int(excluded_mask.sum()),
        "excluded_cross_boundary_video_ids": crossing_video_ids,
        "threshold_method": "Training-only 33rd and 67th percentiles of observed_views.",
        "classification_thresholds": {
            "low_medium": low_threshold,
            "medium_high": high_threshold,
            "boundaries": "Low < q33; Medium >= q33 and < q67; High >= q67.",
        },
        "splits": {},
    }

    retained_rows = sum(len(frame) for frame in split_frames.values())
    for split_name, frame in split_frames.items():
        frame_dates = parse_snapshot_dates(frame["trending_date"])
        metadata["splits"][split_name] = {
            "rows": int(len(frame)),
            "unique_videos": int(frame["video_id"].nunique()),
            "min_trending_date": frame_dates.min().date().isoformat(),
            "max_trending_date": frame_dates.max().date().isoformat(),
            "row_fraction_of_retained": float(len(frame) / retained_rows),
            "observed_views_summary": target_summary(frame["observed_views"]),
            "performance_category_counts": class_counts(
                frame["observed_views"], low_threshold, high_threshold
            ),
        }
    metadata["row_counts"] = {
        "input_rows": int(len(dataset)),
        "retained_rows": int(retained_rows),
        "retained_fraction": float(retained_rows / len(dataset)),
    }
    metadata["video_overlap"] = {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Input rows: {len(dataset)}")
    print(f"Distinct snapshot dates: {dataset['_snapshot_date'].nunique()}")
    print(f"Date range: {dataset['_snapshot_date'].min().date()} to {dataset['_snapshot_date'].max().date()}")
    print(f"Split boundaries: train through {TRAIN_END.date()}, validation through {VALIDATION_END.date()}")
    print(f"Excluded cross-boundary videos: {len(crossing_video_ids)}")
    print(f"Excluded cross-boundary rows: {int(excluded_mask.sum())}")
    print(f"Training-only classification thresholds: q33={low_threshold:.3f}, q67={high_threshold:.3f}")
    for split_name, info in metadata["splits"].items():
        print(
            f"{split_name}: rows={info['rows']}, videos={info['unique_videos']}, "
            f"dates={info['min_trending_date']}..{info['max_trending_date']}, "
            f"classes={info['performance_category_counts']}"
        )
    print("Video overlap checks: train/validation=0, train/test=0, validation/test=0")
    print(f"Split files and metadata: {SPLIT_DIR.relative_to(PROJECT_ROOT)}")
    return split_frames


if __name__ == "__main__":
    build_splits()