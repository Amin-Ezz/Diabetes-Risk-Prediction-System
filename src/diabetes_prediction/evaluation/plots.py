"""Visualization utilities for model performance, training history, and feature importances."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot neural network training and validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # Loss
    if "loss" in history:
        axes[0].plot(history["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy
    dm_acc = history.get("diabetes_accuracy") or history.get("accuracy", [])
    val_dm_acc = history.get("val_diabetes_accuracy") or history.get("val_accuracy", [])
    htn_acc = history.get("hypertension_accuracy", [])
    val_htn_acc = history.get("val_hypertension_accuracy", [])

    if len(dm_acc) > 0:
        axes[1].plot(dm_acc, label="DM Train Acc", color="#2ca02c", linewidth=1.8)
    if len(val_dm_acc) > 0:
        axes[1].plot(val_dm_acc, label="DM Val Acc", color="#98df8a", linewidth=1.8, linestyle="--")
    if len(htn_acc) > 0:
        axes[1].plot(htn_acc, label="HTN Train Acc", color="#d62728", linewidth=1.8)
    if len(val_htn_acc) > 0:
        axes[1].plot(val_htn_acc, label="HTN Val Acc", color="#ff9896", linewidth=1.8, linestyle="--")

    axes[1].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(
    roc_data: Dict[str, Dict[str, Any]],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot comparative ROC curves for test evaluations."""
    fig, axes = plt.subplots(1, len(roc_data), figsize=(6 * len(roc_data), 4.8), constrained_layout=True)
    if len(roc_data) == 1:
        axes = [axes]

    for ax, (target_name, models_dict) in zip(axes, roc_data.items()):
        for model_label, (y_true, y_prob) in models_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{model_label} (AUC={auc_val:.3f})", linewidth=2)

        ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
        ax.set_title(f"ROC Curve — {target_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix_heatmaps(
    cm_dict: Dict[str, np.ndarray],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot confusion matrix heatmaps for evaluation results."""
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(5.5 * len(cm_dict), 4.2), constrained_layout=True)
    if len(cm_dict) == 1:
        axes = [axes]

    for ax, (title, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Negative (0)", "Positive (1)"],
            yticklabels=["Negative (0)", "Positive (1)"],
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
