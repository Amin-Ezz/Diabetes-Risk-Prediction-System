"""Standalone training script for Diabetes and Hypertension prediction models."""

import json
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to python path for modular imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_prediction.data.load_data import load_raw_dataset
from diabetes_prediction.data.preprocessing import DataPreprocessor
from diabetes_prediction.evaluation.metrics import compute_binary_metrics
from diabetes_prediction.evaluation.plots import (
    plot_confusion_matrix_heatmaps,
    plot_roc_curves,
    plot_training_history,
)
from diabetes_prediction.models.train import train_neural_network, train_tree_ensembles


def main() -> None:
    print("=" * 60)
    print("Starting Model Training Pipeline")
    print("=" * 60)

    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random_state = config.get("random_state", 42)

    # 1. Load Data
    print("\n[Step 1/6] Loading raw dataset...")
    raw_csv = PROJECT_ROOT / config["dataset"]["raw_csv_path"]
    df_raw = load_raw_dataset(
        uci_id=config["dataset"]["uci_id"],
        local_cache_path=raw_csv,
    )
    print(f"Loaded raw dataset with shape: {df_raw.shape}")

    # 2. Preprocess & Clean
    print("\n[Step 2/6] Cleaning text and repairing decimal-shift anomalies...")
    preprocessor = DataPreprocessor(
        random_state=random_state,
        n_quantiles=config["preprocessing"]["quantile_transformer"]["n_quantiles"],
    )
    df_clean = preprocessor.clean_dataset(df_raw)
    print(f"Dataset shape after cleaning and label filtering: {df_clean.shape}")

    # Targets & Features
    y_dm = (df_clean["dm"] == "yes").astype(int).to_numpy()
    y_htn = (df_clean["htn"] == "yes").astype(int).to_numpy()
    X = df_clean.drop(columns=[c for c in ["dm", "htn", "class"] if c in df_clean.columns])

    # 3. Stratified Split
    print("\n[Step 3/6] Performing 80/20 stratified split...")
    joint_target = [f"{d}_{h}" for d, h in zip(y_dm, y_htn)]
    idx_train, idx_test = train_test_split(
        np.arange(len(X)),
        test_size=config["dataset"]["test_size"],
        random_state=random_state,
        stratify=joint_target,
    )

    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_dm_train, y_dm_test = y_dm[idx_train], y_dm[idx_test]
    y_htn_train, y_htn_test = y_htn[idx_train], y_htn[idx_test]

    # Save processed splits
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df["dm"] = y_dm_train
    train_df["htn"] = y_htn_train
    train_df.to_csv(processed_dir / "train.csv", index=False)

    test_df = X_test.copy()
    test_df["dm"] = y_dm_test
    test_df["htn"] = y_htn_test
    test_df.to_csv(processed_dir / "test.csv", index=False)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. Fit Preprocessor
    print("\n[Step 4/6] Fitting preprocessor (imputation + normal quantile transformation)...")
    preprocessor.fit(X_train)
    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    preprocessor.save(str(preprocessor_path))
    print(f"Saved fitted preprocessor to: {preprocessor_path}")

    X_train_nn = preprocessor.transform_neural(X_train)
    X_test_nn = preprocessor.transform_neural(X_test)
    X_train_tab = preprocessor.transform_tabular(X_train)
    X_test_tab = preprocessor.transform_tabular(X_test)
    print(f"Transformed neural network feature dimension: {X_train_nn.shape[1]}")

    # 5. Train Deep Neural Network
    print("\n[Step 5/6] Training Multi-Output Deep Neural Network...")
    model_nn_path = PROJECT_ROOT / "models" / "multioutput_nn.keras"
    nn_model, history = train_neural_network(
        X_train_nn=X_train_nn,
        y_dm_train=y_dm_train,
        y_htn_train=y_htn_train,
        config=config,
        save_path=model_nn_path,
    )
    print(f"Saved Neural Network model to: {model_nn_path}")

    # 6. Train Tree Ensembles
    print("\n[Step 6/6] Training Gradient Boosting Ensembles (XGBoost + LightGBM)...")
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_models.joblib"
    ensemble_models = train_tree_ensembles(
        X_train_tab=X_train_tab,
        y_dm_train=y_dm_train,
        y_htn_train=y_htn_train,
        config=config,
        save_path=ensemble_path,
    )
    print(f"Saved Ensemble models to: {ensemble_path}")

    # Evaluate on Test Split
    print("\n" + "=" * 60)
    print("Evaluating Models on Held-out Test Set")
    print("=" * 60)

    # NN Predictions
    p_dm_nn, p_htn_nn = nn_model.predict(X_test_nn, verbose=0)
    p_dm_nn, p_htn_nn = p_dm_nn.ravel(), p_htn_nn.ravel()

    # Ensemble Predictions
    dm_preds_xgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["diabetes"]["xgb"]]
    dm_preds_lgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["diabetes"]["lgb"]]
    p_dm_blend = 0.95 * np.mean(dm_preds_lgb, axis=0) + 0.05 * np.mean(dm_preds_xgb, axis=0)

    htn_preds_xgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["hypertension"]["xgb"]]
    htn_preds_lgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["hypertension"]["lgb"]]
    p_htn_blend = 0.40 * np.mean(htn_preds_lgb, axis=0) + 0.60 * np.mean(htn_preds_xgb, axis=0)

    dm_thr = config["targets"]["diabetes"]["threshold"]
    htn_thr = config["targets"]["hypertension"]["threshold"]

    # Metrics computation
    results = {
        "diabetes": {
            "neural_network": compute_binary_metrics(y_dm_test, p_dm_nn, 0.50, "diabetes_nn"),
            "ensemble_blend": compute_binary_metrics(y_dm_test, p_dm_blend, dm_thr, "diabetes_blend"),
        },
        "hypertension": {
            "neural_network": compute_binary_metrics(y_htn_test, p_htn_nn, 0.50, "hypertension_nn"),
            "ensemble_blend": compute_binary_metrics(y_htn_test, p_htn_blend, htn_thr, "hypertension_blend"),
        },
    }

    # Save metrics and history
    results_dir = PROJECT_ROOT / "reports" / "results"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        # Convert numpy floats in history
        cleaned_hist = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(cleaned_hist, f, indent=2)

    # Save plots
    plot_training_history(history, save_path=figures_dir / "training_history.png")

    roc_data = {
        "diabetes": {
            "Improved NN": (y_dm_test, p_dm_nn),
            "XGB+LGB Blend": (y_dm_test, p_dm_blend),
        },
        "hypertension": {
            "Improved NN": (y_htn_test, p_htn_nn),
            "XGB+LGB Blend": (y_htn_test, p_htn_blend),
        },
    }
    plot_roc_curves(roc_data, save_path=figures_dir / "roc_curves.png")

    cm_data = {
        "Diabetes (NN)": np.array([
            [results["diabetes"]["neural_network"]["confusion"]["tn"], results["diabetes"]["neural_network"]["confusion"]["fp"]],
            [results["diabetes"]["neural_network"]["confusion"]["fn"], results["diabetes"]["neural_network"]["confusion"]["tp"]],
        ]),
        "Diabetes (Blend)": np.array([
            [results["diabetes"]["ensemble_blend"]["confusion"]["tn"], results["diabetes"]["ensemble_blend"]["confusion"]["fp"]],
            [results["diabetes"]["ensemble_blend"]["confusion"]["fn"], results["diabetes"]["ensemble_blend"]["confusion"]["tp"]],
        ]),
        "Hypertension (NN)": np.array([
            [results["hypertension"]["neural_network"]["confusion"]["tn"], results["hypertension"]["neural_network"]["confusion"]["fp"]],
            [results["hypertension"]["neural_network"]["confusion"]["fn"], results["hypertension"]["neural_network"]["confusion"]["tp"]],
        ]),
        "Hypertension (Blend)": np.array([
            [results["hypertension"]["ensemble_blend"]["confusion"]["tn"], results["hypertension"]["ensemble_blend"]["confusion"]["fp"]],
            [results["hypertension"]["ensemble_blend"]["confusion"]["fn"], results["hypertension"]["ensemble_blend"]["confusion"]["tp"]],
        ]),
    }
    plot_confusion_matrix_heatmaps(cm_data, save_path=figures_dir / "confusion_matrices.png")

    print("\n--- Final Test Performance Summary ---")
    for tgt in ["diabetes", "hypertension"]:
        print(f"\nTarget: {tgt.upper()}")
        for model_k, res in results[tgt].items():
            print(
                f"  [{model_k:15s} | Thr={res['threshold']:.2f}] "
                f"Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | "
                f"Rec: {res['recall']:.4f} | F1: {res['f1']:.4f} | ROC-AUC: {res['roc_auc']:.4f} | "
                f"MCC: {res['mcc']:.4f}"
            )

    print("\n" + "=" * 60)
    print("Training Pipeline Finished Successfully!")
    print(f"Artifacts saved in {PROJECT_ROOT / 'models'} and {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
