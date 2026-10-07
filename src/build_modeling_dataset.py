"""Build the documented snapshot-level modeling dataset from immutable raw files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from inspect_dataset import parse_date_column, read_csv_safely


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
REGIONS = {"CA", "DE", "FR", "GB", "IN", "JP", "KR", "MX", "RU", "US"}
SNAPSHOT_KEY = ["video_id", "region", "trending_date"]

GROUP_A_FEATURES = [
    "region",
    "category_name",
    "title_length",
    "title_word_count",
    "description_length",
    "description_word_count",
    "description_missing",
    "tag_count",
    "publish_year",
    "publish_month",
    "publish_day_of_week",
    "publish_hour",
]
GROUP_B_FEATURES = ["days_to_trending"]
AUDIT_ONLY_COLUMNS = [
    "comments_disabled",
    "ratings_disabled",
    "video_error_or_removed",
]
IDENTIFIER_COLUMNS = ["video_id", "trending_date", "publish_time", "category_id"]
FORBIDDEN_INPUT_COLUMNS = {
    "views",
    "observed_views",
    "log_observed_views",
    "likes",
    "dislikes",
    "comment_count",
    "thumbnail_link",
}


def region_from_filename(path: Path) -> str:
    match = re.fullmatch(r"([A-Z]{2})videos", path.stem)
    if not match:
        raise ValueError(f"Cannot identify region from source filename: {path.name}")
    region = match.group(1)
    if region not in REGIONS:
        raise ValueError(f"Unexpected region {region!r} in source filename: {path.name}")
    return region


def category_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def load_category_mappings() -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    for path in sorted(RAW_DATA_DIR.glob("*_category_id.json")):
        region = path.name[:2]
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        items = payload.get("items", []) if isinstance(payload, dict) else []
        mapping: dict[str, str] = {}
        for item in items:
            category_id = item.get("id")
            category_name = item.get("snippet", {}).get("title")
            if category_id is not None and category_name:
                mapping[str(category_id)] = str(category_name)
        mappings[region] = mapping
    return mappings


def count_tags(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.casefold() == "[none]":
        return 0
    return sum(bool(tag.strip()) for tag in text.split("|"))


def add_text_features(frame: pd.DataFrame) -> None:
    title = frame["title"].astype("string")
    description = frame["description"].astype("string")
    frame["title_length"] = title.str.len().astype("Int64")
    frame["title_word_count"] = title.str.split().str.len().astype("Int64")
    frame["description_missing"] = frame["description"].isna().astype(bool)
    frame["description_length"] = description.str.len().astype("Int64")
    frame["description_word_count"] = description.str.split().str.len().astype("Int64")
    frame["tag_count"] = pd.array(frame["tags"].map(count_tags), dtype="Int64")


def add_time_features(frame: pd.DataFrame) -> int:
    publish_time = pd.to_datetime(frame["publish_time"], errors="coerce", utc=True)
    trending_date = parse_date_column(frame["trending_date"], "trending_date")

    frame["publish_year"] = publish_time.dt.year.astype("Int64")
    frame["publish_month"] = publish_time.dt.month.astype("Int64")
    frame["publish_day_of_week"] = publish_time.dt.dayofweek.astype("Int64")
    frame["publish_hour"] = publish_time.dt.hour.astype("Int64")

    # trending_date has day precision only, so compute UTC calendar-day difference.
    day_delta = trending_date.dt.normalize() - publish_time.dt.normalize()
    frame["days_to_trending"] = day_delta.dt.days.astype("Int64")
    return int(publish_time.isna().sum() + trending_date.isna().sum())


def inspect_duration_availability(frames: list[pd.DataFrame]) -> list[str]:
    aliases = ("duration", "video_duration", "video_duration_seconds", "contentDetails.duration")
    return [column for column in aliases if any(column in frame.columns for frame in frames)]


def print_target_distribution(target: pd.Series) -> None:
    summary = target.describe(percentiles=[0.95, 0.99])
    print("Observed-view target distribution (outliers retained):")
    for label, value in (
        ("min", target.min()),
        ("median", target.median()),
        ("mean", target.mean()),
        ("95th percentile", target.quantile(0.95)),
        ("99th percentile", target.quantile(0.99)),
        ("max", target.max()),
    ):
        print(f"  {label}: {value:.3f}")


def build() -> pd.DataFrame:
    csv_paths = sorted(RAW_DATA_DIR.glob("*videos.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No regional video CSVs found in {RAW_DATA_DIR}")

    category_mappings = load_category_mappings()
    source_frames: list[pd.DataFrame] = []
    raw_rows = 0
    exact_duplicates_removed = 0
    repeated_snapshot_key_groups = 0
    conflicting_snapshot_keys = 0
    conflict_rows_excluded = 0
    identical_repeat_key_groups = 0

    for path in csv_paths:
        region = region_from_filename(path)
        frame, encoding = read_csv_safely(path)
        required = {"video_id", "trending_date", "category_id", "publish_time", "views"}
        missing_required = required.difference(frame.columns)
        if missing_required:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing_required)}")

        source_rows = len(frame)
        missing_keys = int(frame[SNAPSHOT_KEY[:1] + ["trending_date"]].isna().any(axis=1).sum())
        if missing_keys:
            raise ValueError(f"{path.name} contains {missing_keys} rows with missing snapshot keys")

        raw_rows += len(frame)
        raw_key_counts = frame.groupby(
            ["video_id", "trending_date"], dropna=False, sort=False
        ).size()
        repeated_in_file = int((raw_key_counts > 1).sum())
        repeated_snapshot_key_groups += repeated_in_file

        exact_duplicate_mask = frame.duplicated(keep="first")
        exact_duplicates_removed += int(exact_duplicate_mask.sum())
        frame = frame.loc[~exact_duplicate_mask].copy()

        remaining_key_counts = frame.groupby(
            ["video_id", "trending_date"], dropna=False, sort=False
        ).size()
        conflict_counts = remaining_key_counts[remaining_key_counts > 1]
        conflicting_snapshot_keys += int(len(conflict_counts))
        conflict_rows_excluded += int(conflict_counts.sum())
        identical_repeat_key_groups += repeated_in_file - int(len(conflict_counts))

        if len(conflict_counts):
            conflict_index = conflict_counts.index
            row_keys = pd.MultiIndex.from_frame(frame[["video_id", "trending_date"]])
            frame = frame.loc[~row_keys.isin(conflict_index)].copy()

        frame["region"] = region
        source_frames.append(frame)
        print(f"Loaded {path.name}: raw_rows={source_rows}; encoding={encoding}")

    if set(category_mappings) != REGIONS:
        missing_mappings = sorted(REGIONS.difference(category_mappings))
        raise FileNotFoundError(f"Missing regional category maps: {missing_mappings}")

    data = pd.concat(source_frames, ignore_index=True, sort=False)
    duration_columns = inspect_duration_availability(source_frames)
    if duration_columns:
        print(f"Duration source columns detected but not yet standardized: {duration_columns}")
    else:
        print("Duration feature excluded: no duration field exists in the approved raw dataset.")

    category_names: list[object] = []
    for region, category_id in zip(data["region"], data["category_id"]):
        key = category_key(category_id)
        category_names.append(category_mappings.get(region, {}).get(key, pd.NA))
    data["category_name"] = pd.array(category_names, dtype="string")

    target = pd.to_numeric(data["views"], errors="coerce")
    if target.isna().any():
        raise ValueError(f"Cannot build target: {int(target.isna().sum())} missing/non-numeric views")
    if (target < 0).any():
        raise ValueError("Cannot apply log1p: negative view counts are present")
    data["observed_views"] = target.astype("int64")
    data["log_observed_views"] = np.log1p(data["observed_views"].astype("float64"))

    add_text_features(data)
    invalid_time_rows = add_time_features(data)

    output_columns = [
        "video_id",
        "region",
        "trending_date",
        "publish_time",
        "category_id",
        "category_name",
        "title_length",
        "title_word_count",
        "description_length",
        "description_word_count",
        "description_missing",
        "tag_count",
        "publish_year",
        "publish_month",
        "publish_day_of_week",
        "publish_hour",
        "comments_disabled",
        "ratings_disabled",
        "video_error_or_removed",
        "days_to_trending",
        "observed_views",
        "log_observed_views",
    ]
    missing_output = set(output_columns).difference(data.columns)
    if missing_output:
        raise ValueError(f"Cannot build output schema; missing columns: {sorted(missing_output)}")

    modeling = data[output_columns].copy()
    group_a_columns = set(GROUP_A_FEATURES)
    group_b_columns = set(GROUP_B_FEATURES)
    if group_a_columns.intersection(FORBIDDEN_INPUT_COLUMNS) or group_b_columns.intersection(
        FORBIDDEN_INPUT_COLUMNS
    ):
        raise ValueError("A forbidden target/engagement column was added to an input feature group")

    valid_regions = set(modeling["region"].unique()).issubset(REGIONS)
    if not valid_regions:
        raise ValueError("Unexpected region values in the modeling dataset")
    if modeling["observed_views"].isna().any():
        raise ValueError("The modeling target contains missing values")
    if (modeling["observed_views"] < 0).any():
        raise ValueError("The modeling target contains negative values")
    if not np.isfinite(modeling["log_observed_views"].to_numpy(dtype="float64")).all():
        raise ValueError("log_observed_views contains non-finite values")
    if modeling.duplicated(subset=SNAPSHOT_KEY).any():
        raise ValueError("Snapshot keys are still duplicated after conflict handling")
    if not modeling["region"].isin(REGIONS).all():
        raise ValueError("Region validation failed")

    feature_ranges = {
        "title_length": (0, modeling["title_length"].max()),
        "title_word_count": (0, modeling["title_word_count"].max()),
        "description_length": (0, modeling["description_length"].max()),
        "description_word_count": (0, modeling["description_word_count"].max()),
        "tag_count": (0, modeling["tag_count"].max()),
        "publish_month": (int(modeling["publish_month"].min()), int(modeling["publish_month"].max())),
        "publish_day_of_week": (
            int(modeling["publish_day_of_week"].min()),
            int(modeling["publish_day_of_week"].max()),
        ),
        "publish_hour": (int(modeling["publish_hour"].min()), int(modeling["publish_hour"].max())),
        "days_to_trending": (
            int(modeling["days_to_trending"].min()),
            int(modeling["days_to_trending"].max()),
        ),
    }
    if feature_ranges["publish_month"][0] < 1 or feature_ranges["publish_month"][1] > 12:
        raise ValueError("publish_month is outside 1..12")
    if feature_ranges["publish_day_of_week"][0] < 0 or feature_ranges["publish_day_of_week"][1] > 6:
        raise ValueError("publish_day_of_week is outside Monday=0..Sunday=6")
    if feature_ranges["publish_hour"][0] < 0 or feature_ranges["publish_hour"][1] > 23:
        raise ValueError("publish_hour is outside 0..23")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    if not OUTPUT_PATH.is_file():
        raise FileNotFoundError(f"Output was not created: {OUTPUT_PATH}")

    saved = pd.read_csv(OUTPUT_PATH, low_memory=False)
    if len(saved) != len(modeling) or saved.columns.tolist() != output_columns:
        raise ValueError("Saved CSV dimensions/schema do not match the in-memory dataset")

    category_missing = int(saved["category_name"].isna().sum())
    print("\nCleaning and conflict summary")
    print(f"Raw CSV rows: {raw_rows}")
    print(f"Exact duplicate rows removed: {exact_duplicates_removed}")
    print(f"Repeated snapshot-key groups before exact deduplication: {repeated_snapshot_key_groups}")
    print(f"Repeated identical-key groups collapsed by exact deduplication: {identical_repeat_key_groups}")
    print(f"Conflicting (video_id, trending_date, region) keys: {conflicting_snapshot_keys}")
    print(f"Rows excluded for unresolved conflicting keys: {conflict_rows_excluded}")
    print(f"Final modeling rows: {len(modeling)}")
    print(f"Final modeling columns: {len(output_columns)}")
    print(f"Category mapping coverage: {len(modeling) - category_missing}/{len(modeling)}")
    print(f"Rows with invalid/missing publication or trending date: {invalid_time_rows}")
    print(f"Description-missing rows (preserved via description_missing): {int(modeling['description_missing'].sum())}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Output missing values: {saved.isna().sum().to_dict()}")
    print(f"Output region values: {sorted(saved['region'].unique().tolist())}")
    print(f"Output columns: {output_columns}")
    print(f"Group A metadata-only columns: {GROUP_A_FEATURES}")
    print(f"Group B snapshot-context columns: {GROUP_B_FEATURES}")
    print(f"Audit-only status columns, not default predictors: {AUDIT_ONLY_COLUMNS}")
    print(f"Traceability columns, not predictors: {IDENTIFIER_COLUMNS}")
    print(f"Leakage columns excluded from output: {sorted(FORBIDDEN_INPUT_COLUMNS - {'observed_views', 'log_observed_views'})}")
    print(f"Feature ranges: {feature_ranges}")
    print_target_distribution(modeling["observed_views"])
    return modeling


if __name__ == "__main__":
    build()