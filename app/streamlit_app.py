"""Diabetes and Hypertension Risk Prediction Web Application."""

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Setup sys.path for local modular imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.components.input_form import render_input_form
from app.components.visualizations import render_metrics_table, render_risk_card
from diabetes_prediction.predict import RiskPredictor

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: White & Navy Blue Palette
st.markdown(
    """
    <style>
        /* Base styles */
        .main {
            background-color: #f8fafc;
        }
        h1, h2, h3, h4 {
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        /* Top Hero Header */
        .hero-banner {
            background: linear-gradient(135deg, #0f2537 0%, #1e3a8a 100%);
            color: white;
            padding: 28px 32px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(15, 37, 55, 0.15);
        }
        .hero-banner h1 {
            color: white !important;
            margin: 0 0 8px 0;
            font-size: 2.2rem;
            font-weight: 800;
        }
        .hero-banner p {
            color: #cbd5e1;
            font-size: 1.05rem;
            margin: 0;
        }
        /* Disclaimer callout */
        .disclaimer-box {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 24px;
            color: #92400e;
            font-size: 0.92rem;
        }
        /* Sidebar branding */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor() -> RiskPredictor:
    """Load model predictor singleton from models directory."""
    model_path = PROJECT_ROOT / "models" / "multioutput_nn.keras"
    preproc_path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_models.joblib"

    if not model_path.exists() or not preproc_path.exists():
        st.error(
            "⚠️ Trained model artifacts not found! "
            "Please train the model first by opening a terminal and running: "
            "`python scripts/train.py`."
        )
        st.stop()

    return RiskPredictor(
        model_path=model_path,
        preprocessor_path=preproc_path,
        ensemble_path=ensemble_path if ensemble_path.exists() else None,
    )


# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 15px 0; text-align: left;">
            <h3 style="margin: 0; color: #1e3a8a; font-weight: 800;">🩺 CardioMetabolic AI</h3>
            <p style="margin: 2px 0 0 0; color: #64748b; font-size: 0.85rem;">Deep Learning Risk Analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=["Overview", "Risk Assessment", "Model Performance", "Data Insights"],
        index=1,
    )

    st.markdown("---")
    st.markdown("#### Model Engine Configuration")

    model_choice = st.selectbox(
        "Active Inference Architecture:",
        options=["Multi-Output Neural Network (Primary)", "XGBoost + LightGBM Ensemble"],
        index=0,
    )
    use_ensemble = "Ensemble" in model_choice

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748b;">
            <p><strong>Dataset:</strong> UCI ML Repo (ID: 336)</p>
            <p><strong>Framework:</strong> TensorFlow / Keras 2.21</p>
            <p><strong>Environment:</strong> Windows 10/11 x64</p>
            <p><strong>Status:</strong> <span style="color: #10b981; font-weight: 600;">● Model Loaded & Ready</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Main Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <h1>Diabetes Risk Prediction System</h1>
        <p>Production Deep Learning and Ensemble Architecture for Simultaneous Diabetes & Hypertension Risk Assessment</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Medical Disclaimer Banner (Always Visible)
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Disclaimer:</strong> This software is engineered strictly for educational, scientific, and decision-support research.
        It does not constitute formal medical advice, diagnostic confirmation, or treatment recommendation. Always consult certified healthcare professionals for clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------
if page == "Overview":
    st.header("Project Overview & Architecture")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            ### Background & Clinical Motivation
            Diabetes Mellitus and Hypertension are twin metabolic conditions that frequently co-occur and dramatically accelerate end-stage renal disease and cardiovascular mortality.

            This application operationalizes a research deep learning pipeline that transforms **22 patient clinical markers** (routine blood work, urinalysis, hemodynamic vitals, and microscopic findings) into calibrated risk scores for both **Diabetes** and **Hypertension**.

            ### Key Engineering Highlights
            * **Simultaneous Multi-Output Deep Learning:** A single neural network model trained with dual sigmoid classification heads for multi-task predictive synergy.
            * **Leakage-Free Preprocessing:** Fitted strictly on training folds with clinical decimal-shift corrections (`sod`, `pot`), robust missingness encoding, and Gaussian quantile normalization.
            * **Dual Inference Pipeline:** Seamless support for both Deep Neural Networks and Optuna-calibrated Gradient Boosting Ensembles (XGBoost + LightGBM).
            * **Zero Online Retraining:** Production-ready inference loading serialized preprocessors and weights instantaneously.
            """
        )

    with col2:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="margin-top: 0; color: #1e3a8a;">Network Architecture</h4>
                <ul style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                    <li><strong>Input Dimension:</strong> 64 Normalized Features</li>
                    <li><strong>Layer 1:</strong> Dense(128) + BatchNorm + ReLU + Dropout(0.35)</li>
                    <li><strong>Layer 2:</strong> Dense(64) + BatchNorm + ReLU + Dropout(0.25)</li>
                    <li><strong>Layer 3:</strong> Dense(32) + BatchNorm + ReLU + Dropout(0.15)</li>
                    <li><strong>Regularization:</strong> L2 (3e-4) on all kernels</li>
                    <li><strong>Loss Function:</strong> BinaryCrossentropy (Label Smoothing: 0.05)</li>
                    <li><strong>Optimizer:</strong> Adam (lr=1e-3, ReduceLROnPlateau)</li>
                    <li><strong>Output Heads:</strong> Dual Sigmoid (Diabetes, Hypertension)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Dataset Summary (UCI Machine Learning Repository #336)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cohort Records", "400 Patients")
    m2.metric("Predictor Features", "22 Clinical Tests")
    m3.metric("Transformed Features", "64 Engineered Dimensions")
    m4.metric("Test Split (Held-out)", "80 Patients (20%)")


# -------------------------------------------------------------
# PAGE 2: RISK ASSESSMENT (PREDICTION)
# -------------------------------------------------------------
elif page == "Risk Assessment":
    st.header("Patient Risk Assessment Form")
    st.write("Enter laboratory and hemodynamic parameters below to generate personalized risk probabilities.")

    predictor = get_predictor()
    form_result = render_input_form()

    if form_result["submitted"]:
        patient_data = form_result["data"]

        with st.spinner("Executing neural inference and evaluating biomarkers..."):
            try:
                prediction_result = predictor.predict(patient_data, use_ensemble=use_ensemble)
            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
                st.stop()

        st.markdown("### 📊 Assessment Findings")
        st.caption(f"Evaluated using: **{prediction_result['model_used']}**")

        rc1, rc2 = st.columns(2)
        with rc1:
            dm_info = prediction_result["diabetes"]
            render_risk_card(
                title="Diabetes Mellitus Assessment",
                probability=dm_info["probability"],
                threshold=dm_info["threshold"],
                prediction=dm_info["prediction"],
                risk_level=dm_info["risk_level"],
                icon="🩸",
            )

        with rc2:
            htn_info = prediction_result["hypertension"]
            render_risk_card(
                title="Hypertension Assessment",
                probability=htn_info["probability"],
                threshold=htn_info["threshold"],
                prediction=htn_info["prediction"],
                risk_level=htn_info["risk_level"],
                icon="🫀",
            )

        # Biomarker Summary Callouts
        st.markdown("#### Clinical Telemetry & Trigger Signals")
        triggers = []
        if patient_data.get("bgr", 0) >= 140.0:
            triggers.append(f"• **Elevated Blood Glucose:** Random glucose is {patient_data['bgr']} mg/dL (Clinical hyperglycemia threshold: >= 140 mg/dL).")
        if patient_data.get("bp", 0) >= 90.0:
            triggers.append(f"• **Elevated Diastolic BP:** Resting blood pressure is {patient_data['bp']} mm Hg (Hypertension threshold: >= 90 mm Hg).")
        if patient_data.get("su", 0) > 0:
            triggers.append(f"• **Glucosuria:** Detected urinary glucose grade is {int(patient_data['su'])}/5.")
        if patient_data.get("hemo", 15) < 12.0:
            triggers.append(f"• **Clinical Anemia:** Hemoglobin level is {patient_data['hemo']} g/dL (< 12 g/dL indicates anemia).")
        if patient_data.get("sc", 0) > 1.4:
            triggers.append(f"• **Renal Stress:** Serum creatinine is {patient_data['sc']} mg/dL (Normal: 0.6 - 1.3 mg/dL).")

        if triggers:
            for t in triggers:
                st.warning(t)
        else:
            st.success("• **All Primary Biomarkers Within Standard Baseline Limits.** No acute metabolic alert signals detected.")


# -------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# -------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Evaluation & Experimental Benchmarks")
    st.write("Rigorous quantitative metrics evaluated on the 20% held-out test split (N=80 patients).")

    metrics_file = PROJECT_ROOT / "reports" / "results" / "metrics.json"

    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        render_metrics_table(metrics_data)
    else:
        st.info("Metrics report not found. Run `python scripts/train.py` to generate evaluation results.")

    st.markdown("---")
    st.subheader("Diagnostic Curves & Validation Analytics")

    tab1, tab2, tab3 = st.tabs(["ROC Curves", "Confusion Matrices", "Training Convergence"])

    with tab1:
        roc_img_path = PROJECT_ROOT / "reports" / "figures" / "roc_curves.png"
        if roc_img_path.exists():
            st.image(str(roc_img_path), caption="Receiver Operating Characteristic (ROC) on Held-Out Test Set", use_container_width=True)
        else:
            st.warning("ROC curves figure not found.")

    with tab2:
        cm_img_path = PROJECT_ROOT / "reports" / "figures" / "confusion_matrices.png"
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="Confusion Matrices across Neural Network and Tree Ensembles", use_container_width=True)
        else:
            st.warning("Confusion matrix figure not found.")

    with tab3:
        hist_img_path = PROJECT_ROOT / "reports" / "figures" / "training_history.png"
        if hist_img_path.exists():
            st.image(str(hist_img_path), caption="Training & Validation Convergence (Early Stopping at optimal epoch)", use_container_width=True)
        else:
            st.warning("Training history figure not found.")


# -------------------------------------------------------------
# PAGE 4: DATA INSIGHTS
# -------------------------------------------------------------
elif page == "Data Insights":
    st.header("Dataset Exploration & Clinical Insights")

    raw_csv = PROJECT_ROOT / "data" / "raw" / "chronic_kidney_disease.csv"
    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"

    data_to_load = train_csv if train_csv.exists() else raw_csv
    if data_to_load.exists():
        df_display = pd.read_csv(data_to_load)

        st.markdown(f"### Cohort Sample View (`{data_to_load.name}`)")
        st.dataframe(df_display.head(10), use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### Target Prevalence in Cohort")
            if "dm" in df_display.columns and "htn" in df_display.columns:
                target_counts = pd.DataFrame({
                    "Condition": ["Diabetes Mellitus", "Hypertension"],
                    "Positive Count": [int((df_display["dm"] == 1).sum() if df_display["dm"].dtype != object else (df_display["dm"] == "yes").sum()),
                                       int((df_display["htn"] == 1).sum() if df_display["htn"].dtype != object else (df_display["htn"] == "yes").sum())],
                    "Total Samples": [len(df_display), len(df_display)],
                })
                target_counts["Prevalence (%)"] = (target_counts["Positive Count"] / target_counts["Total Samples"] * 100).round(1)
                st.dataframe(target_counts, hide_index=True, use_container_width=True)

        with col_d2:
            st.markdown("#### Glycemic & Hemodynamic Summary")
            num_cols = [c for c in ["age", "bp", "bgr", "bu", "sc", "hemo"] if c in df_display.columns]
            if num_cols:
                st.dataframe(df_display[num_cols].describe().round(2).T[["mean", "std", "min", "50%", "max"]], use_container_width=True)

        st.markdown("---")
        st.markdown("#### Clinical Feature Correlation with Diabetes")
        num_df = df_display.select_dtypes(include="number")
        if "dm" in num_df.columns:
            corr = num_df.corr()["dm"].drop(index=["dm"]).sort_values(ascending=False).head(10)
            st.bar_chart(corr)
    else:
        st.info("Dataset CSV not found. Run `python scripts/train.py` to ingest and cache the dataset.")
