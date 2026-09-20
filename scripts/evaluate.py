"""Model evaluation script for Diabetes and Hypertension prediction."""

import json
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_prediction.data.preprocessing import DataPreprocessor
from diabetes_prediction.evaluation.metrics import compute_binary_metrics
from diabetes_prediction.evaluation.plots import (
    plot_confusion_matrix_heatmaps,
    plot_roc_curves,
)
from diabetes_prediction.models.neural_network import load_trained_model
import joblib


def main() -> None:
    print("=" * 60)
    print("Starting Model Evaluation Pipeline")
    print("=" * 60)

    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Processed test data not found at {test_csv}. Please run 'python scripts/train.py' first."
        )

    df_test = pd.read_csv(test_csv)
    y_dm_test = df_test["dm"].to_numpy()
    y_htn_test = df_test["htn"].to_numpy()
    X_test = df_test.drop(columns=["dm", "htn"])

    # Load artifacts
    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    model_nn_path = PROJECT_ROOT / "models" / "multioutput_nn.keras"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_models.joblib"

    print("\nLoading preprocessor and models...")
    preprocessor = DataPreprocessor.load(str(preprocessor_path))
    nn_model = load_trained_model(model_nn_path)
    ensemble_models = joblib.load(str(ensemble_path)) if ensemble_path.exists() else None

    # Transform
    X_test_nn = preprocessor.transform_neural(X_test)
    X_test_tab = preprocessor.transform_tabular(X_test)

    # Predictions
    p_dm_nn, p_htn_nn = nn_model.predict(X_test_nn, verbose=0)
    p_dm_nn, p_htn_nn = p_dm_nn.ravel(), p_htn_nn.ravel()

    dm_preds_xgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["diabetes"]["xgb"]]
    dm_preds_lgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["diabetes"]["lgb"]]
    p_dm_blend = 0.95 * np.mean(dm_preds_lgb, axis=0) + 0.05 * np.mean(dm_preds_xgb, axis=0)

    htn_preds_xgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["hypertension"]["xgb"]]
    htn_preds_lgb = [m.predict_proba(X_test_tab)[:, 1] for m in ensemble_models["hypertension"]["lgb"]]
    p_htn_blend = 0.40 * np.mean(htn_preds_lgb, axis=0) + 0.60 * np.mean(htn_preds_xgb, axis=0)

    dm_thr = config["targets"]["diabetes"]["threshold"]
    htn_thr = config["targets"]["hypertension"]["threshold"]

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

    # Print markdown table
    print("\n### Evaluation Results on Held-out Test Set (N=80)\n")
    print("| Target | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Specificity | MCC |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tgt in ["diabetes", "hypertension"]:
        for m_name, res in results[tgt].items():
            disp_model = "Improved Neural Network" if m_name == "neural_network" else "XGB+LGB Blend"
            print(
                f"| {tgt.capitalize()} | {disp_model} | {res['accuracy']:.4f} | {res['precision']:.4f} | "
                f"{res['recall']:.4f} | {res['f1']:.4f} | {res['roc_auc']:.4f} | {res['specificity']:.4f} | {res['mcc']:.4f} |"
            )

    results_dir = PROJECT_ROOT / "reports" / "results"
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation metrics updated in: {results_dir / 'metrics.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
