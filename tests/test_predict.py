"""Unit tests for inference and prediction service."""

import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_prediction.predict import RiskPredictor


class TestPredict(unittest.TestCase):
    """Test suite for risk prediction service."""

    @classmethod
    def setUpClass(cls):
        model_path = PROJECT_ROOT / "models" / "multioutput_nn.keras"
        preproc_path = PROJECT_ROOT / "models" / "preprocessor.joblib"
        cls.predictor = RiskPredictor(
            model_path=model_path,
            preprocessor_path=preproc_path,
        )

    def test_valid_patient_prediction(self):
        sample_patient = {
            "age": 48.0,
            "bp": 80.0,
            "sg": 1.020,
            "al": 1.0,
            "su": 0.0,
            "rbc": "normal",
            "pc": "normal",
            "pcc": "notpresent",
            "ba": "notpresent",
            "bgr": 121.0,
            "bu": 36.0,
            "sc": 1.2,
            "sod": 138.0,
            "pot": 4.4,
            "hemo": 15.4,
            "pcv": 44.0,
            "wbcc": 7800.0,
            "rbcc": 5.2,
            "cad": "no",
            "appet": "good",
            "pe": "no",
            "ane": "no",
        }
        res = self.predictor.predict(sample_patient)

        self.assertIn("diabetes", res)
        self.assertIn("hypertension", res)

        for target in ["diabetes", "hypertension"]:
            t_res = res[target]
            self.assertIn("probability", t_res)
            self.assertIn("prediction", t_res)
            self.assertIn("risk_level", t_res)
            self.assertIn("label", t_res)
            self.assertTrue(0.0 <= t_res["probability"] <= 1.0)
            self.assertIn(t_res["prediction"], [0, 1])

    def test_input_validation(self):
        # Invalid categorical value
        invalid_cat = {"rbc": "completely_wrong_value"}
        is_valid, msg = self.predictor.validate_inputs(invalid_cat)
        self.assertFalse(is_valid)
        self.assertIn("Invalid value", msg)

        # Extreme impossible numerical value
        invalid_num = {"age": 999.0}
        is_valid, msg = self.predictor.validate_inputs(invalid_num)
        self.assertFalse(is_valid)
        self.assertIn("clinically implausible", msg)

    def test_missing_values_handled_gracefully(self):
        # Patient with only vitals provided (partial input)
        partial_patient = {
            "age": 55.0,
            "bp": 85.0,
            "bgr": 160.0,
        }
        res = self.predictor.predict(partial_patient)
        self.assertIn("diabetes", res)
        self.assertTrue(0.0 <= res["diabetes"]["probability"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
