"""Inference helpers for the frozen Group B XGBoost pipeline."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "group_b_xgboost_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "group_b_xgboost_metadata.json"

FEATURES = [
    "region",
    "category_name",
    "title_length",
    "title_word_count",
    "description_length",
    "description_word_count",
    "tag_count",
    "publish_year",
    "publish_month",
    "publish_day_of_week",
    "publish_hour",
    "description_missing",
    "days_to_trending",
]
CATEGORICAL_FEATURES = ["region", "category_name"]
COUNT_FEATURES = [
    "title_length",
    "title_word_count",
    "description_length",
    "description_word_count",
    "tag_count",
]
INTEGER_FEATURES = COUNT_FEATURES + [
    "publish_year",
    "publish_month",
    "publish_day_of_week",
    "publish_hour",
    "days_to_trending",
]
# Verified from the existing train split; do not derive a new range at inference time.
TRAINING_YEAR_RANGE = (2006, 2018)


class InputValidationError(ValueError):
    """User-correctable input/schema error."""


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]
    regions: tuple[str, ...]
    categories: tuple[str, ...]


def load_model_bundle(
    model_path: Path | str = MODEL_PATH,
    metadata_path: Path | str = METADATA_PATH,
) -> ModelBundle:
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline) or "preprocess" not in pipeline.named_steps or "regressor" not in pipeline.named_steps:
        raise RuntimeError("Saved artifact is not the expected preprocessing and regressor pipeline")
    if not isinstance(pipeline.named_steps["regressor"], XGBRegressor):
        raise RuntimeError("Saved pipeline regressor is not XGBoost")
    if metadata.get("model") != "XGBRegressor" or metadata.get("target") != "log_observed_views":
        raise RuntimeError("Model metadata does not identify the expected XGBoost log-target model")
    if metadata.get("feature_group") != "Group B" or metadata.get("feature_list") != FEATURES:
        raise RuntimeError("Model metadata feature group/list does not match the application contract")

    fitted_columns = list(pipeline.named_steps["preprocess"].feature_names_in_)
    if fitted_columns != FEATURES:
        raise RuntimeError("Fitted pipeline columns do not match the application feature contract")
    categorical_pipeline = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["onehot"]
    regions = tuple(str(value) for value in encoder.categories_[0])
    categories = tuple(str(value) for value in encoder.categories_[1])
    return ModelBundle(pipeline, metadata, regions, categories)


def validate_record(
    record: Mapping[str, Any],
    regions: tuple[str, ...] | list[str],
    categories: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise InputValidationError("Input must be one feature record.")
    keys = set(record)
    expected = set(FEATURES)
    missing, extra = expected - keys, keys - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected fields: {', '.join(sorted(extra))}")
        raise InputValidationError("Input schema mismatch (" + "; ".join(details) + ").")

    normalized: dict[str, Any] = {}
    for feature, allowed in (("region", regions), ("category_name", categories)):
        value = record[feature]
        if not isinstance(value, str) or value not in allowed:
            raise InputValidationError(f"Choose a supported value for {feature.replace('_', ' ')}.")
        normalized[feature] = value

    for feature in INTEGER_FEATURES:
        value = record[feature]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise InputValidationError(f"{feature.replace('_', ' ').title()} must be a numeric whole number.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise InputValidationError(f"{feature.replace('_', ' ').title()} must be a finite whole number.")
        normalized[feature] = int(numeric_value)

    missing_flag = record["description_missing"]
    if isinstance(missing_flag, (bool, np.bool_)):
        normalized["description_missing"] = int(missing_flag)
    elif isinstance(missing_flag, (int, np.integer)) and int(missing_flag) in (0, 1):
        normalized["description_missing"] = int(missing_flag)
    else:
        raise InputValidationError("Description Missing must be true or false.")

    for feature in COUNT_FEATURES:
        if normalized[feature] < 0:
            raise InputValidationError(f"{feature.replace('_', ' ').title()} cannot be negative.")
    year = normalized["publish_year"]
    if not TRAINING_YEAR_RANGE[0] <= year <= TRAINING_YEAR_RANGE[1]:
        raise InputValidationError(
            f"Publish Year must be between {TRAINING_YEAR_RANGE[0]} and {TRAINING_YEAR_RANGE[1]}, the training-data range."
        )
    if not 1 <= normalized["publish_month"] <= 12:
        raise InputValidationError("Publish Month must be from 1 through 12.")
    if not 0 <= normalized["publish_day_of_week"] <= 6:
        raise InputValidationError("Publish Day of Week must use Monday=0 through Sunday=6.")
    if not 0 <= normalized["publish_hour"] <= 23:
        raise InputValidationError("Publish Hour must be from 0 through 23.")
    if normalized["days_to_trending"] < 0:
        raise InputValidationError("Days to Trending must be non-negative.")

    return normalized


def make_input_frame(record: Mapping[str, Any], bundle: ModelBundle) -> pd.DataFrame:
    normalized = validate_record(record, bundle.regions, bundle.categories)
    frame = pd.DataFrame([normalized], columns=FEATURES)
    if list(frame.columns) != bundle.metadata["feature_list"]:
        raise InputValidationError("Input columns do not match the final model contract.")
    return frame


def predict_record(record: Mapping[str, Any], bundle: ModelBundle) -> tuple[float, float]:
    input_frame = make_input_frame(record, bundle)
    predicted_log = float(bundle.pipeline.predict(input_frame)[0])
    if not math.isfinite(predicted_log):
        raise RuntimeError("Model returned a non-finite prediction")
    # Match the project's existing inverse-transform behavior: floor log prediction at 0.
    predicted_views = float(np.expm1(max(predicted_log, 0.0)))
    if not math.isfinite(predicted_views) or predicted_views < 0:
        raise RuntimeError("Model returned an invalid view-count estimate")
    return predicted_log, predicted_views


def log_prediction_error(message: str) -> None:
    logging.exception(message)
