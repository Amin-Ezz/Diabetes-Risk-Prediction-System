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
from app.components.visualizations import render_feature_radar, render_metrics_table, render_risk_card
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
        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        /* -------------------------------------------------------------
           PATIENT CLINICAL PARAMETERS & FORM STYLING:
           Clean, bright white & light slate palette with high contrast
           ------------------------------------------------------------- */
        /* Form container */
        [data-testid="stForm"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 24px !important;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
        }

        /* Section headers in form */
        [data-testid="stForm"] h4 {
            color: #1e3a8a !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            border-bottom: 2px solid #f1f5f9 !important;
            padding-bottom: 8px !important;
            margin-top: 20px !important;
            margin-bottom: 16px !important;
        }

        /* Input Labels in main content */
        .main [data-testid="stWidgetLabel"] p,
        .main [data-testid="stWidgetLabel"] label {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        /* Number Input Container & Fields */
        div[data-testid="stNumberInputContainer"],
        .main div[data-testid="stNumberInputContainer"],
        div[data-testid="stForm"] div[data-testid="stNumberInputContainer"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            padding: 3px 8px 3px 4px !important;
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            box-sizing: border-box !important;
            overflow: visible !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-testid="stNumberInputContainer"]:focus-within {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
        }
        div[data-testid="stNumberInputContainer"] input,
        input[data-testid="stNumberInputField"] {
            background-color: transparent !important;
            color: #0f172a !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            padding: 4px 8px !important;
            flex: 1 1 auto !important;
            min-width: 0 !important;
            width: 100% !important;
            height: 100% !important;
        }
        .main [data-testid="stTextInput"] input {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            padding: 8px 12px !important;
        }

        /* Step Buttons Wrapper Container */
        div[data-testid="stNumberInputContainer"] > div:last-child {
            display: inline-flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-end !important;
            gap: 6px !important;
            flex-shrink: 0 !important;
            margin-left: 6px !important;
            height: auto !important;
        }

        /* Distinct Square Step Buttons (+ and -) */
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"],
        div[data-testid="stNumberInputContainer"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 30px !important;
            min-width: 30px !important;
            max-width: 30px !important;
            height: 30px !important;
            min-height: 30px !important;
            max-height: 30px !important;
            background-color: #f1f5f9 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 6px !important;
            color: #1e293b !important;
            cursor: pointer !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
            transition: all 0.15s ease-in-out !important;
            opacity: 1 !important;
            flex-shrink: 0 !important;
        }
        button[data-testid="stNumberInputStepDown"]:hover:not(:disabled),
        button[data-testid="stNumberInputStepUp"]:hover:not(:disabled),
        div[data-testid="stNumberInputContainer"] button:hover:not(:disabled) {
            background-color: #e2e8f0 !important;
            border-color: #94a3b8 !important;
            color: #0f172a !important;
        }
        button[data-testid="stNumberInputStepDown"]:active:not(:disabled),
        button[data-testid="stNumberInputStepUp"]:active:not(:disabled),
        div[data-testid="stNumberInputContainer"] button:active:not(:disabled) {
            background-color: #cbd5e1 !important;
            transform: scale(0.94) !important;
        }
        button[data-testid="stNumberInputStepDown"]:disabled,
        button[data-testid="stNumberInputStepUp"]:disabled,
        div[data-testid="stNumberInputContainer"] button:disabled {
            opacity: 0.3 !important;
            cursor: not-allowed !important;
            background-color: #f8fafc !important;
            border-color: #e2e8f0 !important;
        }
        div[data-testid="stNumberInputContainer"] button svg,
        div[data-testid="stNumberInputContainer"] button span {
            color: inherit !important;
            fill: currentColor !important;
            width: 14px !important;
            height: 14px !important;
        }

        /* Selectboxes in main content (Patient Profile preset & form dropdowns) */
        .main [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            color: #0f172a !important;
        }
        .main [data-testid="stSelectbox"] [data-baseweb="select"] span,
        .main [data-testid="stSelectbox"] [data-baseweb="select"] div {
            color: #0f172a !important;
            font-weight: 500 !important;
        }
        .main [data-testid="stSelectbox"] [data-baseweb="select"] svg {
            fill: #475569 !important;
        }
        .main [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
            border-color: #94a3b8 !important;
        }

        /* Submit Button */
        [data-testid="stForm"] button[kind="primaryFormSubmit"],
        [data-testid="stForm"] button[data-testid="baseButton-secondaryFormSubmit"],
        [data-testid="stForm"] button {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
            transition: all 0.2s ease !important;
            margin-top: 16px !important;
        }
        [data-testid="stForm"] button:hover {
            background: linear-gradient(135deg, #172554 0%, #1d4ed8 100%) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
            transform: translateY(-1px);
        }
        [data-testid="stForm"] button p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* -------------------------------------------------------------
           SIDEBAR THEME: Modern Deep Navy & Slate
           Solves text invisibility and provides high contrast & sleek design
           ------------------------------------------------------------- */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarUserContent"],
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1329 0%, #0f172a 50%, #131f37 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        /* Sidebar Header and Collapse Control */
        [data-testid="stSidebarHeader"] {
            background: transparent !important;
        }
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarHeader"] button,
        [data-testid="collapsedControl"] button {
            color: #cbd5e1 !important;
            background: rgba(255, 255, 255, 0.06) !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stSidebarHeader"] button:hover,
        [data-testid="collapsedControl"] button:hover {
            color: #38bdf8 !important;
            background: rgba(56, 189, 248, 0.15) !important;
        }

        /* Sidebar Typography */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6 {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] li {
            color: #cbd5e1 !important;
        }
        [data-testid="stSidebar"] strong {
            color: #f1f5f9 !important;
        }

        /* Sidebar Horizontal Dividers */
        [data-testid="stSidebar"] hr {
            border: none !important;
            border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
            margin: 18px 0 !important;
        }

        /* Sidebar Navigation Radio Buttons */
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
            color: #94a3b8 !important;
            font-size: 0.76rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            margin-bottom: 6px !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
            gap: 6px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin-bottom: 3px !important;
            transition: all 0.2s ease-in-out !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
            background: rgba(56, 189, 248, 0.12) !important;
            border-color: rgba(56, 189, 248, 0.4) !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p,
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label span,
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label div {
            color: #f1f5f9 !important;
            font-weight: 500 !important;
            font-size: 0.92rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] input[type="radio"] {
            accent-color: #38bdf8 !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"],
        [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
            background: rgba(56, 189, 248, 0.18) !important;
            border-color: #38bdf8 !important;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.15) !important;
        }

        /* Sidebar Selectbox */
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > label,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            margin-bottom: 6px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div {
            color: #f8fafc !important;
            font-size: 0.88rem !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #94a3b8 !important;
        }

        /* Sidebar Custom Component Cards */
        .sidebar-brand-card {
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.5) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .sidebar-brand-title {
            margin: 0;
            color: #38bdf8 !important;
            font-size: 1.12rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }
        .sidebar-brand-sub {
            margin: 4px 0 0 0;
            color: #94a3b8 !important;
            font-size: 0.82rem;
        }
        .sidebar-info-box {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.82rem;
            color: #94a3b8;
            line-height: 1.6;
        }
        .sidebar-info-box p {
            margin: 4px 0 !important;
            color: #94a3b8 !important;
        }
        .sidebar-info-box strong {
            color: #cbd5e1 !important;
        }
        .sidebar-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #34d399 !important;
            font-weight: 600;
        }
        .sidebar-status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.7);
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
        ensemble_path=None,
    )


# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <h3 class="sidebar-brand-title">🩺 CardioMetabolic AI</h3>
            <p class="sidebar-brand-sub">Deep Learning Risk Analytics</p>
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
    st.markdown(
        """
        <div class="sidebar-info-box">
            <p><strong>Architecture:</strong> Multi-Output Deep Neural Network</p>
            <p><strong>Dataset:</strong> UCI ML Repo (ID: 336)</p>
            <p><strong>Framework:</strong> TensorFlow / Keras 2.21</p>
            <p><strong>Environment:</strong> Windows 10/11 x64</p>
            <p style="margin-top: 8px;"><strong>Status:</strong> <span class="sidebar-status-badge"><span class="sidebar-status-dot"></span> Model Loaded & Ready</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Main Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <h1>Diabetes Risk Prediction System</h1>
        <p>Production Deep Learning Architecture for Simultaneous Diabetes & Hypertension Risk Assessment</p>
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
            * **Calibrated Decision Boundaries:** Optimal thresholding tailored for high sensitivity and balanced clinical classification.
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
                prediction_result = predictor.predict(patient_data, use_ensemble=False)
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
                icon="❤️",
            )

        st.markdown("---")
        st.subheader("Laboratory Biomarker Reference & Risk Map")
        render_feature_radar(patient_data)


# -------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# -------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Evaluation & Diagnostics")
    st.write("Comprehensive validation metrics evaluated on an independent 20% stratified test split.")

    metrics_path = PROJECT_ROOT / "reports" / "results" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
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
            st.image(str(cm_img_path), caption="Confusion Matrices for Multi-Output Neural Network", use_container_width=True)
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
