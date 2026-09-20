"""Unit tests for preprocessing and feature engineering."""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_prediction.data.preprocessing import (
    ALLOWED_VALUES,
    CLINICAL_RANGES,
    DataPreprocessor,
    deep_clean_text,
    engineer_features,
    repair_decimal_shifts,
)


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing functions and pipeline."""

    def test_deep_clean_text(self):
        # Tabs and whitespace removal
        self.assertEqual(deep_clean_text("\tno ", ALLOWED_VALUES["dm"]), "no")
        self.assertEqual(deep_clean_text(" YES\t", ALLOWED_VALUES["htn"]), "yes")
        # Disallowed value returns None
        self.assertIsNone(deep_clean_text("invalid_val", ALLOWED_VALUES["dm"]))
        # NaN returns None
        self.assertIsNone(deep_clean_text(np.nan, ALLOWED_VALUES["dm"]))

    def test_repair_decimal_shifts(self):
        df = pd.DataFrame({
            "sod": [4.5, 135.0],
            "pot": [39.0, 4.5],
        })
        repaired = repair_decimal_shifts(df)
        self.assertAlmostEqual(repaired.loc[0, "sod"], 45.0)
        self.assertAlmostEqual(repaired.loc[1, "sod"], 135.0)
        self.assertAlmostEqual(repaired.loc[0, "pot"], 3.9)
        self.assertAlmostEqual(repaired.loc[1, "pot"], 4.5)

    def test_engineer_features(self):
        df = pd.DataFrame({
            "bu": [36.0],
            "sc": [1.2],
            "hemo": [15.4],
            "pcv": [44.0],
            "bgr": [121.0],
            "su": [0.0],
            "bp": [80.0],
            "rbc": [np.nan],
            "rbcc": [5.2],
            "wbcc": [7800.0],
            "sod": [135.0],
            "pot": [4.5],
        })
        engineered = engineer_features(df)
        self.assertIn("bu_sc_ratio", engineered.columns)
        self.assertIn("hemo_x_pcv", engineered.columns)
        self.assertIn("bgr_x_su", engineered.columns)
        self.assertIn("bp_high", engineered.columns)
        self.assertIn("bgr_high", engineered.columns)
        self.assertIn("anemia", engineered.columns)
        self.assertIn("rbc_missing", engineered.columns)
        self.assertEqual(engineered.loc[0, "rbc_missing"], 1)

    def test_preprocessor_pipeline(self):
        # Create a mock training dataframe with 20 samples
        n_samples = 20
        data = {
            "age": np.random.uniform(20, 80, n_samples),
            "bp": np.random.uniform(60, 100, n_samples),
            "sg": [1.020] * n_samples,
            "al": [1.0] * n_samples,
            "su": [0.0] * n_samples,
            "bgr": np.random.uniform(70, 200, n_samples),
            "bu": np.random.uniform(20, 70, n_samples),
            "sc": np.random.uniform(0.6, 2.5, n_samples),
            "sod": np.random.uniform(130, 145, n_samples),
            "pot": np.random.uniform(3.5, 5.0, n_samples),
            "hemo": np.random.uniform(10, 16, n_samples),
            "pcv": np.random.uniform(30, 48, n_samples),
            "wbcc": np.random.uniform(4000, 11000, n_samples),
            "rbcc": np.random.uniform(3.5, 6.0, n_samples),
            "rbc": ["normal"] * n_samples,
            "pc": ["normal"] * n_samples,
            "pcc": ["notpresent"] * n_samples,
            "ba": ["notpresent"] * n_samples,
            "cad": ["no"] * n_samples,
            "appet": ["good"] * n_samples,
            "pe": ["no"] * n_samples,
            "ane": ["no"] * n_samples,
        }
        df_train = pd.DataFrame(data)

        preprocessor = DataPreprocessor(n_quantiles=15)
        preprocessor.fit(df_train)

        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(len(preprocessor.feature_columns_eng), 37)

        X_nn = preprocessor.transform_neural(df_train)
        self.assertEqual(X_nn.shape[0], n_samples)
        self.assertGreaterEqual(X_nn.shape[1], 37)


if __name__ == "__main__":
    unittest.main()
