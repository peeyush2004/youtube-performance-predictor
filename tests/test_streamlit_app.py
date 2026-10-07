"""Contract and UI smoke tests for saved-model inference (no training)."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.prediction import (
    FEATURES,
    InputValidationError,
    load_model_bundle,
    make_input_frame,
    predict_record,
)


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"
SYNTHETIC_RECORD = {
    "region": "US",
    "category_name": "Science & Technology",
    "title_length": 54,
    "title_word_count": 9,
    "description_length": 420,
    "description_word_count": 68,
    "tag_count": 8,
    "publish_year": 2018,
    "publish_month": 5,
    "publish_day_of_week": 2,
    "publish_hour": 14,
    "description_missing": 0,
    "days_to_trending": 2,
}


class StreamlitInferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_model_bundle()

    def test_model_loading(self) -> None:
        self.assertEqual(type(self.bundle.pipeline.named_steps["regressor"]).__name__, "XGBRegressor")
        self.assertIn("preprocess", self.bundle.pipeline.named_steps)
        self.assertEqual(self.bundle.metadata["xgboost_version"], "3.2.0")

    def test_input_schema_is_exact(self) -> None:
        frame = make_input_frame(SYNTHETIC_RECORD, self.bundle)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(len(frame.columns), 13)
        self.assertNotIn("observed_views", frame.columns)
        self.assertNotIn("log_observed_views", frame.columns)

    def test_valid_prediction_uses_synthetic_contract_example(self) -> None:
        predicted_log, predicted_views = predict_record(SYNTHETIC_RECORD, self.bundle)
        self.assertTrue(math.isfinite(predicted_log))
        self.assertTrue(math.isfinite(predicted_views))
        self.assertGreaterEqual(predicted_views, 0)

    def test_invalid_negative_days_to_trending(self) -> None:
        record = {**SYNTHETIC_RECORD, "days_to_trending": -1}
        with self.assertRaisesRegex(InputValidationError, "non-negative"):
            make_input_frame(record, self.bundle)

    def test_invalid_month(self) -> None:
        record = {**SYNTHETIC_RECORD, "publish_month": 13}
        with self.assertRaisesRegex(InputValidationError, "Month"):
            make_input_frame(record, self.bundle)

    def test_invalid_hour(self) -> None:
        record = {**SYNTHETIC_RECORD, "publish_hour": 24}
        with self.assertRaisesRegex(InputValidationError, "Hour"):
            make_input_frame(record, self.bundle)

    def test_missing_required_column(self) -> None:
        record = dict(SYNTHETIC_RECORD)
        del record["tag_count"]
        with self.assertRaisesRegex(InputValidationError, "missing fields"):
            make_input_frame(record, self.bundle)

    def test_invalid_category(self) -> None:
        record = {**SYNTHETIC_RECORD, "category_name": "Made-up category"}
        with self.assertRaisesRegex(InputValidationError, "supported value"):
            make_input_frame(record, self.bundle)

    def test_nonfinite_numeric_rejected(self) -> None:
        record = {**SYNTHETIC_RECORD, "title_length": float("inf")}
        with self.assertRaisesRegex(InputValidationError, "finite"):
            make_input_frame(record, self.bundle)

    def test_streamlit_form_button_runs_prediction(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertFalse(app.exception)
        button = next(item for item in app.button if item.label == "Estimate Observed Views")
        button.click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.label == "Estimated Observed Views" for item in app.metric))


if __name__ == "__main__":
    unittest.main()
