"""Inspect the downloaded YouTube trending dataset without modifying raw files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATE_COLUMNS = ("trending_date", "publishedAt", "publish_time")
FEATURE_AUDIT_COLUMNS = (
    "view_count",
    "views",
    "likes",
    "comment_count",
    "comments",
    "trending_date",
    "publish_time",
    "publishedAt",
    "video_id",
    "channel_title",
    "channelTitle",
    "category_id",
    "categoryId",
    "duration",
    "tags",
    "title",
    "description",
)


def get_region(path: Path) -> str | None:
    """Return the two-letter region prefix used by the source file names."""
    prefix = path.name[:2]
    return prefix if len(prefix) == 2 and prefix.isalpha() else None


def parse_date_column(values: pd.Series, column: str) -> pd.Series:
    """Parse source date fields without inventing missing time precision."""
    if column == "trending_date":
        return pd.to_datetime(values, format="%y.%d.%m", errors="coerce", utc=True)
    return pd.to_datetime(values, errors="coerce", utc=True)


def read_csv_safely(path: Path) -> tuple[pd.DataFrame, str]:
    """Read UTF-8 CSVs and replace malformed bytes without changing source files."""
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        frame = pd.read_csv(
            path,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
        replacement_count = sum(
            int(frame[column].astype("string").str.count("\ufffd").sum())
            for column in frame.columns
            if pd.api.types.is_string_dtype(frame[column].dtype)
        )
        return frame, f"utf-8-sig replacement fallback ({replacement_count} replacement characters)"


def print_date_ranges(frame: pd.DataFrame) -> None:
    date_columns = [column for column in DATE_COLUMNS if column in frame.columns]
    if not date_columns:
        print("Date ranges: none of the recognized date columns are present")
        return

    print("Date ranges:")
    for column in date_columns:
        parsed = parse_date_column(frame[column], column).dropna()
        if parsed.empty:
            print(f"  {column}: no parseable dates")
        else:
            print(f"  {column}: {parsed.min()} to {parsed.max()}")


def print_frame_report(
    path: Path,
    frame: pd.DataFrame,
    *,
    json_items: bool = False,
    source_encoding: str | None = None,
) -> None:
    row_label = "category entries" if json_items else "rows"
    region = get_region(path)
    print(f"\n{'=' * 80}\nFile: {path.name}")
    print(f"Region inferred from file name: {region or 'not identified'}")
    if source_encoding:
        print(f"Encoding used for inspection: {source_encoding}")
    print(f"Dimensions: {len(frame)} {row_label} x {len(frame.columns)} columns")
    print(f"Column names: {frame.columns.tolist()}")
    print("Data types:")
    print(frame.dtypes.to_string())
    print("Missing-value counts:")
    print(frame.isna().sum().to_string())
    print(f"Exact duplicate-row count: {int(frame.duplicated().sum())}")

    if json_items:
        print_date_ranges(frame)
        print("Unique video IDs: unavailable (category mapping file)")
        print("Unique channel IDs: unavailable (category mapping file)")
        item_id_column = "id" if "id" in frame.columns else None
        if item_id_column:
            print(f"Unique category IDs: {frame[item_id_column].nunique(dropna=True)}")
        return

    print_date_ranges(frame)
    video_column = "video_id" if "video_id" in frame.columns else None
    channel_column = next(
        (column for column in ("channelId", "channel_id") if column in frame.columns),
        None,
    )
    if video_column:
        video_counts = frame[video_column].dropna().astype(str).value_counts()
        unique_video_count = int(video_counts.size)
        print(f"Unique video IDs: {unique_video_count}")
        if unique_video_count:
            print(f"Average observations per video in file: {len(frame) / unique_video_count:.3f}")
            print(f"Maximum observations for one video in file: {int(video_counts.max())}")
        if "trending_date" in frame.columns:
            snapshot_key = [video_column, "trending_date"]
            key_counts = frame.groupby(snapshot_key, dropna=False, sort=False).size()
            repeated_keys = key_counts[key_counts > 1]
            print(f"Repeated (video_id, trending_date) keys within file: {len(repeated_keys)}")
            print(
                "Extra rows beyond one per repeated snapshot key: "
                f"{int((repeated_keys - 1).sum())}"
            )
            repeated_rows = frame[frame.duplicated(subset=snapshot_key, keep=False)]
            distinct_rows_per_key = repeated_rows.drop_duplicates().groupby(
                snapshot_key, dropna=False, sort=False
            ).size()
            print(
                "Repeated snapshot keys with conflicting row contents: "
                f"{int((distinct_rows_per_key > 1).sum())}"
            )
    else:
        print("Unique video IDs: unavailable (no video_id column)")
    if channel_column:
        print(f"Unique channel IDs: {frame[channel_column].nunique(dropna=True)}")
    else:
        print("Unique channel IDs: unavailable (no channel ID column)")
    channel_title_column = next(
        (column for column in ("channel_title", "channelTitle") if column in frame.columns),
        None,
    )
    if channel_title_column:
        print(
            "Unique channel_title strings (proxy, not guaranteed unique channels): "
            f"{frame[channel_title_column].nunique(dropna=True)}"
        )

    present_audit_columns = [
        column for column in FEATURE_AUDIT_COLUMNS if column in frame.columns
    ]
    missing_audit_columns = [
        column for column in FEATURE_AUDIT_COLUMNS if column not in frame.columns
    ]
    print(f"Requested audit columns present: {present_audit_columns}")
    print(f"Requested audit columns absent: {missing_audit_columns}")

    for column in ("title", "channel_title", "tags", "description", "thumbnail_link"):
        if column in frame.columns:
            values = frame[column].dropna().astype(str)
            if values.empty:
                print(f"Text summary for {column}: no non-missing values")
            else:
                lengths = values.str.len()
                print(
                    f"Text summary for {column}: non-missing={len(values)}, "
                    f"unique={values.nunique()}, chars_min={int(lengths.min())}, "
                    f"chars_median={lengths.median():.1f}, chars_max={int(lengths.max())}"
                )

    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        print("Numeric summary: no numeric columns")
    else:
        print("Numeric summary:")
        print(numeric.describe().transpose().to_string())


def inspect_csv_files(
    paths: list[Path],
) -> tuple[int, Counter[str], dict[str, set[str]], set[str]]:
    total_rows = 0
    video_observations: Counter[str] = Counter()
    video_regions: dict[str, set[str]] = defaultdict(set)
    channel_title_labels: set[str] = set()

    for path in paths:
        try:
            frame, encoding = read_csv_safely(path)
        except Exception as error:
            print(f"\nCould not read {path.name}: {type(error).__name__}: {error}")
            continue

        print_frame_report(path, frame, source_encoding=encoding)
        total_rows += len(frame)

        if "video_id" in frame.columns:
            identifiers = frame["video_id"].dropna().astype(str)
            video_observations.update(identifiers.tolist())
            region = get_region(path)
            if region:
                for video_id in identifiers.unique():
                    video_regions[video_id].add(region)

        channel_title_column = next(
            (column for column in ("channel_title", "channelTitle") if column in frame.columns),
            None,
        )
        if channel_title_column:
            channel_title_labels.update(
                frame[channel_title_column].dropna().astype(str).unique()
            )

    return total_rows, video_observations, video_regions, channel_title_labels


def inspect_json_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                items = payload.get("items", [])
                print(f"\nJSON root keys for {path.name}: {list(payload.keys())}")
            elif isinstance(payload, list):
                items = payload
                print(f"\nJSON root for {path.name}: list")
            else:
                items = []
                print(f"\nJSON root for {path.name}: {type(payload).__name__}")

            if items and isinstance(items[0], dict):
                frame = pd.json_normalize(items)
            elif items:
                frame = pd.DataFrame({"value": items})
            else:
                frame = pd.DataFrame()
            print_frame_report(path, frame, json_items=True)
        except Exception as error:
            print(f"\nCould not read {path.name}: {type(error).__name__}: {error}")


def print_repeated_video_summary(
    total_rows: int,
    video_observations: Counter[str],
    video_regions: dict[str, set[str]],
    channel_title_labels: set[str],
    csv_count: int,
) -> None:
    print(f"\n{'=' * 80}\nCross-file video snapshot summary")
    print(f"CSV files inspected: {csv_count}")
    print(f"Total CSV rows: {total_rows}")
    print(f"Unique video IDs across CSVs: {len(video_observations)}")
    print(f"Unique observations beyond first per video: {total_rows - len(video_observations)}")
    print(
        "Unique channel_title strings across CSVs (proxy, not guaranteed unique channels): "
        f"{len(channel_title_labels)}"
    )

    if video_observations:
        average = total_rows / len(video_observations)
        print(f"Average observations per video: {average:.3f}")
        print(f"Maximum observations for one video: {max(video_observations.values())}")
        print(
            "Videos appearing in multiple regions: "
            f"{sum(len(regions) > 1 for regions in video_regions.values())}"
        )
    else:
        print("Repeated-video statistics unavailable (video_id not found in CSV files)")


def main() -> None:
    if not RAW_DATA_DIR.is_dir():
        raise SystemExit(f"Raw data directory not found: {RAW_DATA_DIR}")

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    json_files = sorted(RAW_DATA_DIR.glob("*.json"))
    if not csv_files and not json_files:
        raise SystemExit(f"No CSV or JSON dataset files found in {RAW_DATA_DIR}")

    print(f"Raw dataset directory: {RAW_DATA_DIR}")
    print(f"CSV files: {len(csv_files)}; JSON files: {len(json_files)}")
    total_rows, video_observations, video_regions, channel_title_labels = inspect_csv_files(
        csv_files
    )
    inspect_json_files(json_files)
    print_repeated_video_summary(
        total_rows,
        video_observations,
        video_regions,
        channel_title_labels,
        len(csv_files),
    )

    if csv_files and len(csv_files) > 1:
        print(
            "Observation unit: inspect the CSV columns above; where video_id, "
            "trending_date, and a regional file are present, a row represents "
            "one video-region-trending-date snapshot."
        )


if __name__ == "__main__":
    main()