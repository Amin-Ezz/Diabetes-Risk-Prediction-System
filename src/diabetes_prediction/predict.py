"""Inference service for Diabetes and Hypertension risk assessment."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from diabetes_prediction.data.preprocessing import (
    ALLOWED_VALUES,
    BASE_FEATURE_COLS,
    CLINICAL_RANGES,
    DataPreprocessor,
)
from diabetes_prediction.models.neural_network import load_trained_model


class RiskPredictor:
    """Production inference engine loading serialized preprocessor and models."""

    def __init__(
        self,
        model_path: str | Path = "models/multioutput_nn.keras",
        preprocessor_path: str | Path = "models/preprocessor.joblib",
        ensemble_path: Optional[str | Path] = "models/ensemble_models.joblib",
        dm_threshold: float = 0.50,
        htn_threshold: float = 0.39,
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.ensemble_path = Path(ensemble_path) if ensemble_path else None
        self.dm_threshold = dm_threshold
        self.htn_threshold = htn_threshold

        self.nn_model = None
        self.preprocessor = None
        self.ensemble_models = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model and preprocessing pipeline."""
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found at {self.preprocessor_path}. "
                "Please run 'python scripts/train.py' first."
            )
        self.preprocessor = DataPreprocessor.load(str(self.preprocessor_path))

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Neural Network model not found at {self.model_path}. "
                "Please run 'python scripts/train.py' first."
            )
        self.nn_model = load_trained_model(self.model_path)

        if self.ensemble_path and self.ensemble_path.exists():
            try:
                self.ensemble_models = joblib.load(str(self.ensemble_path))
            except Exception:
                self.ensemble_models = None

    def validate_inputs(self, patient_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate patient dictionary against clinical ranges and allowed categories."""
        for key, val in patient_dict.items():
            if val is None or pd.isna(val):
                continue

            if key in CLINICAL_RANGES:
                try:
                    num_val = float(val)
                    lo, hi = CLINICAL_RANGES[key]
                    if num_val < (lo * 0.1) or num_val > (hi * 2.0):
                        return False, f"Value for '{key}' ({num_val}) is clinically implausible (expected range: {lo}-{hi})."
                except (ValueError, TypeError):
                    return False, f"Field '{key}' must be numeric."

            if key in ALLOWED_VALUES:
                cleaned_str = str(val).replace("\t", "").replace(" ", "").strip().lower()
                if cleaned_str not in ALLOWED_VALUES[key]:
                    return False, f"Invalid value '{val}' for '{key}'. Allowed: {sorted(list(ALLOWED_VALUES[key]))}."

        return True, None

    def _categorize_risk(self, probability: float, threshold: float) -> str:
        """Categorize risk level based on calibrated probability."""
        if probability < (threshold * 0.65):
            return "Low Risk"
        elif probability < threshold:
            return "Borderline / Moderate Risk"
        elif probability < min(0.80, threshold + 0.25):
            return "Elevated Risk"
        else:
            return "High Risk"

    def predict(
        self,
        patient_data: Dict[str, Any] | pd.DataFrame,
        use_ensemble: bool = False,
    ) -> Dict[str, Any]:
        """Compute Diabetes and Hypertension risk predictions.

        Args:
            patient_data: Single patient dictionary or DataFrame of features.
            use_ensemble: If True and ensemble models are loaded, blend tree ensemble predictions.

        Returns:
            Structured risk assessment dictionary.
        """
        if isinstance(patient_data, dict):
            is_valid, err_msg = self.validate_inputs(patient_data)
            if not is_valid:
                raise ValueError(err_msg)
            X_tab, X_nn = self.preprocessor.transform_single_patient(patient_data)
        elif isinstance(patient_data, pd.DataFrame):
            X_tab = self.preprocessor.transform_tabular(patient_data)
            X_nn = self.preprocessor.transform_neural(patient_data)
        else:
            raise TypeError("patient_data must be a dict or a pandas DataFrame.")

        # Neural Network inference
        nn_preds = self.nn_model.predict(X_nn, verbose=0)
        p_dm_nn = float(np.ravel(nn_preds[0])[0])
        p_htn_nn = float(np.ravel(nn_preds[1])[0])

        final_p_dm = p_dm_nn
        final_p_htn = p_htn_nn
        model_used = "Multi-Output Deep Neural Network"

        # Tree ensemble inference if requested and available
        if use_ensemble and self.ensemble_models is not None:
            try:
                dm_xgb_preds = [m.predict_proba(X_tab)[:, 1] for m in self.ensemble_models["diabetes"]["xgb"]]
                dm_lgb_preds = [m.predict_proba(X_tab)[:, 1] for m in self.ensemble_models["diabetes"]["lgb"]]
                p_dm_xgb = np.mean(dm_xgb_preds, axis=0)[0]
                p_dm_lgb = np.mean(dm_lgb_preds, axis=0)[0]
                p_dm_blend = 0.95 * p_dm_lgb + 0.05 * p_dm_xgb

                htn_xgb_preds = [m.predict_proba(X_tab)[:, 1] for m in self.ensemble_models["hypertension"]["xgb"]]
                htn_lgb_preds = [m.predict_proba(X_tab)[:, 1] for m in self.ensemble_models["hypertension"]["lgb"]]
                p_htn_xgb = np.mean(htn_xgb_preds, axis=0)[0]
                p_htn_lgb = np.mean(htn_lgb_preds, axis=0)[0]
                p_htn_blend = 0.40 * p_htn_lgb + 0.60 * p_htn_xgb

                final_p_dm = float(p_dm_blend)
                final_p_htn = float(p_htn_blend)
                model_used = "XGBoost + LightGBM Ensemble Blend"
            except Exception:
                pass

        dm_pred = int(final_p_dm >= self.dm_threshold)
        htn_pred = int(final_p_htn >= self.htn_threshold)

        return {
            "model_used": model_used,
            "diabetes": {
                "probability": round(final_p_dm, 4),
                "threshold": round(self.dm_threshold, 2),
                "prediction": dm_pred,
                "label": "Positive (At Risk)" if dm_pred == 1 else "Negative (Normal)",
                "risk_level": self._categorize_risk(final_p_dm, self.dm_threshold),
                "nn_probability": round(p_dm_nn, 4),
            },
            "hypertension": {
                "probability": round(final_p_htn, 4),
                "threshold": round(self.htn_threshold, 2),
                "prediction": htn_pred,
                "label": "Positive (At Risk)" if htn_pred == 1 else "Negative (Normal)",
                "risk_level": self._categorize_risk(final_p_htn, self.htn_threshold),
                "nn_probability": round(p_htn_nn, 4),
            },
        }
