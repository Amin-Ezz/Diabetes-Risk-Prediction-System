"""Data loading utilities for the Chronic Kidney Disease / Diabetes dataset."""

from pathlib import Path
from typing import Tuple
import pandas as pd


def load_raw_dataset(
    uci_id: int = 336,
    local_cache_path: str | Path = "data/raw/chronic_kidney_disease.csv",
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch the dataset from UCI ML repo or load from local cache.

    Args:
        uci_id: The UCI repository dataset ID (default 336).
        local_cache_path: File path to save/load cached CSV.
        force_download: If True, bypass local cache and download fresh from UCI.

    Returns:
        pd.DataFrame containing features and target columns.
    """
    cache_file = Path(local_cache_path)

    if cache_file.exists() and not force_download:
        df = pd.read_csv(cache_file)
        return df

    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=uci_id)
        X_raw = dataset.data.features
        y_raw = dataset.data.targets
        df = pd.concat([X_raw, y_raw], axis=1)

        # Cache locally
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        return df

    except Exception as exc:
        if cache_file.exists():
            return pd.read_csv(cache_file)
        raise RuntimeError(
            f"Failed to fetch dataset from UCI (ID: {uci_id}) and no local cache was found at {cache_file}."
        ) from exc


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Separate features and target columns (diabetes, hypertension, and ckd class).

    Args:
        df: Cleaned dataframe containing 'dm', 'htn', and 'class'.

    Returns:
        Tuple of (X, y_diabetes, y_hypertension, y_ckd_class)
    """
    y_diabetes = (df["dm"] == "yes").astype(int)
    y_hypertension = (df["htn"] == "yes").astype(int)
    y_ckd = (df["class"] == "ckd").astype(int) if "class" in df.columns else None

    cols_to_drop = [c for c in ["dm", "htn", "class"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    return X, y_diabetes, y_hypertension, y_ckd
