"""Model training pipelines for both Deep Learning Neural Network and Gradient Boosting Ensemble."""

from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from diabetes_prediction.models.neural_network import build_multioutput_nn


def compute_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights for class imbalance."""
    counts = np.bincount(y)
    w = len(y) / (2.0 * counts)
    return w[y]


def make_xgb(params: Dict[str, Any], seed: int = 42) -> XGBClassifier:
    """Instantiate XGBClassifier with tuned hyperparameters."""
    return XGBClassifier(
        **params,
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
    )


def make_lgb(params: Dict[str, Any], seed: int = 42) -> LGBMClassifier:
    """Instantiate LGBMClassifier with tuned hyperparameters."""
    return LGBMClassifier(
        **params,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


def train_neural_network(
    X_train_nn: np.ndarray,
    y_dm_train: np.ndarray,
    y_htn_train: np.ndarray,
    config: Dict[str, Any],
    save_path: str | Path = "models/multioutput_nn.keras",
) -> Tuple[Any, Dict[str, Any]]:
    """Train the multi-output neural network with callbacks and balanced weights."""
    nn_cfg = config.get("neural_network", {})
    epochs = nn_cfg.get("epochs", 400)
    batch_size = nn_cfg.get("batch_size", 32)
    val_split = nn_cfg.get("validation_split", 0.15)
    es_patience = nn_cfg.get("early_stopping_patience", 40)
    lr_patience = nn_cfg.get("reduce_lr_patience", 12)
    min_lr = nn_cfg.get("min_lr", 1e-6)

    sw_dm = compute_balanced_weights(y_dm_train)
    sw_htn = compute_balanced_weights(y_htn_train)

    model = build_multioutput_nn(
        n_features=X_train_nn.shape[1],
        hidden_units=nn_cfg.get("hidden_units", [128, 64, 32]),
        dropout_rates=nn_cfg.get("dropout_rates", [0.35, 0.25, 0.15]),
        l2_reg=float(nn_cfg.get("l2_reg", 3e-4)),
        learning_rate=float(nn_cfg.get("learning_rate", 1e-3)),
        label_smoothing=float(nn_cfg.get("label_smoothing", 0.05)),
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(nn_cfg.get("reduce_lr_factor", 0.5)),
            patience=lr_patience,
            min_lr=float(min_lr),
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_nn,
        [y_dm_train, y_htn_train],
        sample_weight=[sw_dm, sw_htn],
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    return model, history.history


def train_tree_ensembles(
    X_train_tab: pd.DataFrame,
    y_dm_train: np.ndarray,
    y_htn_train: np.ndarray,
    config: Dict[str, Any],
    save_path: str | Path = "models/ensemble_models.joblib",
) -> Dict[str, Any]:
    """Train seed-averaged XGBoost and LightGBM models for Diabetes & Hypertension."""
    ens_cfg = config.get("ensemble", {})
    blend_seeds = ens_cfg.get("blend_seeds", [42, 7, 123])

    trained_models = {
        "diabetes": {"xgb": [], "lgb": []},
        "hypertension": {"xgb": [], "lgb": []},
        "config": ens_cfg,
    }

    for seed in blend_seeds:
        # Diabetes
        m_xgb_dm = make_xgb(ens_cfg["dm_xgb"], seed=seed)
        m_lgb_dm = make_lgb(ens_cfg["dm_lgb"], seed=seed)
        m_xgb_dm.fit(X_train_tab, y_dm_train)
        m_lgb_dm.fit(X_train_tab, y_dm_train)
        trained_models["diabetes"]["xgb"].append(m_xgb_dm)
        trained_models["diabetes"]["lgb"].append(m_lgb_dm)

        # Hypertension
        m_xgb_htn = make_xgb(ens_cfg["htn_xgb"], seed=seed)
        m_lgb_htn = make_lgb(ens_cfg["htn_lgb"], seed=seed)
        m_xgb_htn.fit(X_train_tab, y_htn_train)
        m_lgb_htn.fit(X_train_tab, y_htn_train)
        trained_models["hypertension"]["xgb"].append(m_xgb_htn)
        trained_models["hypertension"]["lgb"].append(m_lgb_htn)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models, save_path)

    return trained_models
