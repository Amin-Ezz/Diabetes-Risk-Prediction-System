"""Visualization components for risk scores, metric dashboards, and clinical telemetry."""

from typing import Any, Dict
import pandas as pd
import streamlit as st


def render_risk_card(
    title: str,
    probability: float,
    threshold: float,
    prediction: int,
    risk_level: str,
    icon: str = "🩺",
) -> None:
    """Render a clinical risk summary card with probability bar and clinical indicator."""
    # Determine color scheme
    if risk_level == "Low Risk":
        badge_bg = "#ecfdf5"
        badge_color = "#047857"
        border_color = "#a7f3d0"
        bar_color = "#10b981"
    elif risk_level in ["Borderline / Moderate Risk", "Elevated Risk"]:
        badge_bg = "#fffbeb"
        badge_color = "#b45309"
        border_color = "#fde68a"
        bar_color = "#f59e0b"
    else:
        badge_bg = "#fef2f2"
        badge_color = "#b91c1c"
        border_color = "#fecaca"
        bar_color = "#ef4444"

    status_label = "POSITIVE / ELEVATED RISK" if prediction == 1 else "NEGATIVE / NORMAL"

    st.markdown(
        f"""
        <div style="background: white; border: 1px solid {border_color}; border-left: 6px solid {bar_color}; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #0f172a; font-size: 1.15rem; font-weight: 700;">{icon} {title}</h4>
                <span style="background-color: {badge_bg}; color: {badge_color}; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; border: 1px solid {border_color};">
                    {risk_level.upper()}
                </span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 2.2rem; font-weight: 800; color: #1e3a8a;">{probability * 100:.1f}%</span>
                <span style="color: #64748b; font-size: 0.9rem;">Estimated Probability (Threshold: {threshold * 100:.0f}%)</span>
            </div>
            <div style="background-color: #f1f5f9; border-radius: 6px; height: 12px; width: 100%; overflow: hidden; margin-bottom: 12px;">
                <div style="background-color: {bar_color}; width: {min(100.0, probability * 100):.1f}%; height: 100%; border-radius: 6px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #475569;">
                <span><strong>Clinical Classification:</strong> <span style="color: {badge_color}; font-weight: 700;">{status_label}</span></span>
                <span>Calibrated Cutoff: {threshold:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_table(metrics_dict: Dict[str, Any]) -> None:
    """Render a structured comparative metrics table in Streamlit."""
    rows = []
    for target in ["diabetes", "hypertension"]:
        if target in metrics_dict:
            for m_key, m_val in metrics_dict[target].items():
                m_label = "Multi-Output Neural Network" if m_key == "neural_network" else "XGBoost + LightGBM Ensemble"
                rows.append({
                    "Target": target.capitalize(),
                    "Model Architecture": m_label,
                    "Decision Threshold": f"{m_val.get('threshold', 0.5):.2f}",
                    "Accuracy": f"{m_val.get('accuracy', 0.0) * 100:.2f}%",
                    "Precision": f"{m_val.get('precision', 0.0) * 100:.2f}%",
                    "Recall (Sensitivity)": f"{m_val.get('recall', 0.0) * 100:.2f}%",
                    "Specificity": f"{m_val.get('specificity', 0.0) * 100:.2f}%",
                    "F1-Score": f"{m_val.get('f1', 0.0):.4f}",
                    "ROC-AUC": f"{m_val.get('roc_auc', 0.0):.4f}",
                    "MCC": f"{m_val.get('mcc', 0.0):.4f}",
                })

    if rows:
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
