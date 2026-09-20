"""Data cleaning, feature engineering, and neural network transformation pipeline."""

from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

# Clinical validation dictionary for categorical features
ALLOWED_VALUES: Dict[str, set] = {
    "rbc": {"normal", "abnormal"},
    "pc": {"normal", "abnormal"},
    "pcc": {"present", "notpresent"},
    "ba": {"present", "notpresent"},
    "htn": {"yes", "no"},
    "dm": {"yes", "no"},
    "cad": {"yes", "no"},
    "appet": {"good", "poor"},
    "pe": {"yes", "no"},
    "ane": {"yes", "no"},
    "class": {"ckd", "notckd"},
}

CLINICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (1.0, 120.0),
    "bp": (40.0, 260.0),
    "bgr": (30.0, 600.0),
    "bu": (1.0, 500.0),
    "sc": (0.1, 100.0),
    "sod": (100.0, 180.0),
    "pot": (1.5, 12.0),
    "hemo": (3.0, 25.0),
    "pcv": (5.0, 70.0),
    "wbcc": (1000.0, 100000.0),
    "rbcc": (1.0, 10.0),
}

RAW_CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba", "cad", "appet", "pe", "ane"]
RAW_NUMERICAL_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc"
]
BASE_FEATURE_COLS = RAW_NUMERICAL_COLS + RAW_CATEGORICAL_COLS

HIGH_MISSING_COLS = ["rbc", "rbcc", "wbcc", "sod", "pot", "pcv", "hemo", "bgr", "bp"]


def deep_clean_text(value: Any, allowed: set) -> Optional[str]:
    """Normalize raw text values: strip tabs/spaces, lowercase, check validity."""
    if pd.isna(value):
        return None
    val_str = str(value).replace("\t", "").replace(" ", "").strip().lower()
    return val_str if val_str in allowed else None


def repair_decimal_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Repair documented clinical decimal-shift errors (sod x10, pot /10)."""
    df_fixed = df.copy()
    if "sod" in df_fixed.columns:
        mask_sod = df_fixed["sod"] < CLINICAL_RANGES["sod"][0]
        df_fixed.loc[mask_sod, "sod"] = df_fixed.loc[mask_sod, "sod"] * 10
    if "pot" in df_fixed.columns:
        mask_pot = df_fixed["pot"] > CLINICAL_RANGES["pot"][1]
        df_fixed.loc[mask_pot, "pot"] = df_fixed.loc[mask_pot, "pot"] / 10
    return df_fixed


def engineer_features(Xin: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-motivated ratios, risk flags, and missing indicators without leakage."""
    Xe = Xin.copy()

    # Ratios / interactions
    Xe["bu_sc_ratio"] = Xe["bu"] / (Xe["sc"] + 1e-3)
    Xe["hemo_x_pcv"] = Xe["hemo"] * Xe["pcv"]
    Xe["bgr_x_su"] = Xe["bgr"] * (Xe["su"] + 1)

    # Clinical risk flags (NaN-aware)
    Xe["bp_high"] = np.where(Xe["bp"].isna(), np.nan, (Xe["bp"] >= 90).astype(float))
    Xe["bgr_high"] = np.where(Xe["bgr"].isna(), np.nan, (Xe["bgr"] >= 180).astype(float))
    Xe["anemia"] = np.where(Xe["hemo"].isna(), np.nan, (Xe["hemo"] < 12).astype(float))

    # Missing indicators for high-missingness clinical columns
    for c in HIGH_MISSING_COLS:
        Xe[f"{c}_missing"] = Xe[c].isna().astype(int)

    return Xe


class DataPreprocessor:
    """End-to-end clinical preprocessing pipeline for Diabetes & Hypertension prediction."""

    def __init__(self, random_state: int = 42, n_quantiles: int = 300):
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.cat_code_map: Dict[str, List[str]] = {}
        self.feature_columns_eng: List[str] = []
        self.imputer: Optional[SimpleImputer] = None
        self.quantile_tf: Optional[QuantileTransformer] = None
        self.is_fitted: bool = False

    def clean_dataset(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Run text cleaning, decimal shift repair, and target filtering on raw dataframe."""
        df = df_raw.copy()

        # Clean categorical columns
        for col, allowed in ALLOWED_VALUES.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda v: deep_clean_text(v, allowed))

        # Drop rows missing targets
        target_cols = [c for c in ["dm", "htn"] if c in df.columns]
        if target_cols:
            df = df.dropna(subset=target_cols).reset_index(drop=True)

        # Repair decimal shifts
        df = repair_decimal_shifts(df)
        return df

    def encode_categoricals(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical columns ordinally, maintaining NaN for missing values."""
        X_out = X.copy()
        for col in RAW_CATEGORICAL_COLS:
            if col not in X_out.columns:
                continue

            if fit:
                cat = X_out[col].astype("category")
                categories = list(cat.cat.categories)
                self.cat_code_map[col] = categories
                X_out[col] = cat.cat.codes.astype(float).replace(-1, np.nan)
            else:
                categories = self.cat_code_map.get(col, [])
                cat = pd.Categorical(X_out[col], categories=categories)
                X_out[col] = pd.Series(cat.codes, index=X_out.index).astype(float).replace(-1, np.nan)

        return X_out

    def fit(self, X_train: pd.DataFrame) -> "DataPreprocessor":
        """Fit feature engineering, categorical mappings, imputer, and quantile transformer."""
        X_clean = repair_decimal_shifts(X_train)
        X_eng = engineer_features(X_clean)
        X_encoded = self.encode_categoricals(X_eng, fit=True)
        self.feature_columns_eng = X_encoded.columns.tolist()

        # Fit imputer with missing indicators
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        X_imp = self.imputer.fit_transform(X_encoded)

        # Fit quantile transformer
        n_quant = min(self.n_quantiles, len(X_encoded))
        self.quantile_tf = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quant,
            random_state=self.random_state,
        )
        self.quantile_tf.fit(X_imp)

        self.is_fitted = True
        return self

    def transform_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw/cleaned features into engineered tabular features (for tree models)."""
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform_tabular.")
        X_clean = repair_decimal_shifts(X)
        X_eng = engineer_features(X_clean)
        X_encoded = self.encode_categoricals(X_eng, fit=False)
        # Ensure exact column ordering
        return X_encoded.reindex(columns=self.feature_columns_eng)

    def transform_neural(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features into normalized array of dimension 64 for Neural Network."""
        X_tab = self.transform_tabular(X)
        X_imp = self.imputer.transform(X_tab)
        X_norm = self.quantile_tf.transform(X_imp)
        return X_norm

    def transform_single_patient(self, patient_dict: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Transform a single patient dictionary into inputs for both tree models and neural network."""
        # Clean text inputs
        cleaned_dict = {}
        for k, v in patient_dict.items():
            if k in ALLOWED_VALUES:
                cleaned_dict[k] = deep_clean_text(v, ALLOWED_VALUES[k])
            else:
                cleaned_dict[k] = v

        df_single = pd.DataFrame([cleaned_dict])
        for col in BASE_FEATURE_COLS:
            if col not in df_single.columns:
                df_single[col] = np.nan

        df_single = df_single[BASE_FEATURE_COLS]
        X_tab = self.transform_tabular(df_single)
        X_nn = self.transform_neural(df_single)
        return X_tab, X_nn

    def save(self, filepath: str) -> None:
        """Serialize preprocessor instance to disk using joblib."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """Load serialized preprocessor instance from disk."""
        return joblib.load(filepath)
