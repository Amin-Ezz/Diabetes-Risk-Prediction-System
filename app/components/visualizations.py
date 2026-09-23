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
    """Render a structured performance metrics table in Streamlit."""
    rows = []
    for target in ["diabetes", "hypertension"]:
        if target in metrics_dict and "neural_network" in metrics_dict[target]:
            m_val = metrics_dict[target]["neural_network"]
            rows.append({
                "Target Condition": target.capitalize(),
                "Architecture": "Multi-Output Deep Neural Network",
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


def render_feature_radar(patient_data: Dict[str, Any]) -> None:
    """Render a radar chart comparing key biomarkers against normalized healthy baseline references."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Key numerical biomarkers with healthy target / reference values
    biomarkers = [
        ("Blood Glucose (bgr)", float(patient_data.get("bgr", 100)), 70.0, 140.0),
        ("Blood Pressure (bp)", float(patient_data.get("bp", 80)), 60.0, 90.0),
        ("Serum Creatinine (sc)", float(patient_data.get("sc", 1.0)), 0.6, 1.2),
        ("Blood Urea (bu)", float(patient_data.get("bu", 30)), 15.0, 45.0),
        ("Hemoglobin (hemo)", float(patient_data.get("hemo", 14)), 12.0, 17.5),
        ("PCV / Hematocrit", float(patient_data.get("pcv", 40)), 36.0, 50.0),
    ]

    labels = [b[0] for b in biomarkers]
    normalized_patient = []
    normalized_ref = []

    for name, val, norm_min, norm_max in biomarkers:
        norm_range = norm_max - norm_min if norm_max > norm_min else 1.0
        # Normalize relative to standard healthy midpoint (1.0 = mid normal)
        mid = (norm_min + norm_max) / 2.0
        rel_val = (val / mid) if mid > 0 else 1.0
        normalized_patient.append(min(max(rel_val, 0.2), 2.5))
        normalized_ref.append(1.0)

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the circular loop
    normalized_patient += normalized_patient[:1]
    normalized_ref += normalized_ref[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor="none")
    ax.set_facecolor("#f8fafc")

    # Draw healthy baseline reference polygon
    ax.plot(angles, normalized_ref, color="#10b981", linewidth=1.8, linestyle="--", label="Standard Normal Center (1.0x)")
    ax.fill(angles, normalized_ref, color="#10b981", alpha=0.12)

    # Draw patient profile polygon
    ax.plot(angles, normalized_patient, color="#1e3a8a", linewidth=2.4, marker="o", label="Patient Biomarkers")
    ax.fill(angles, normalized_patient, color="#3b82f6", alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9, fontweight="bold", color="#1e293b")
    ax.tick_params(pad=14)

    ax.set_rlabel_position(22)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5x", "1.0x (Norm)", "1.5x", "2.0x"], color="#64748b", size=8)
    ax.set_ylim(0, 2.5)

    ax.grid(color="#cbd5e1", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

