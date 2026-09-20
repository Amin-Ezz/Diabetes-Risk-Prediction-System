# Diabetes Risk Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready Deep Learning and Ensemble Machine Learning platform for simultaneous **Diabetes Mellitus** and **Essential Hypertension** risk prediction from routine clinical and laboratory biomarkers.

---

## 1. Project Overview & Clinical Objectives

Diabetes Mellitus (DM) and Essential Hypertension (HTN) are two tightly coupled metabolic comorbidities that drastically accelerate chronic kidney disease progression and cardiovascular events.

This repository provides a modular, fully tested deep learning framework and a modern **Streamlit GUI dashboard** that allows clinicians, researchers, and students to:

- Assess a patient's simultaneous risk for Diabetes and Hypertension from **22 routine clinical predictors**.
- Leverage a **Multi-Output Deep Neural Network** trained with label smoothing, batch normalization, and stepped dropout.
- Compare predictions against an **Optuna-tuned Gradient Boosting Ensemble** (blended XGBoost and LightGBM).
- Launch the web interface effortlessly on Windows via a double-clickable `start.bat` script.

---

## 2. Key Features

- **Multi-Output Deep Learning Architecture:** Single shared-representation neural network predicting dual binary outputs with calibrated probabilities.
- **Leakage-Free Preprocessing Pipeline:** Strict training-fold isolation with clinical decimal-shift repair, median imputation with missingness indicators, and Gaussian quantile normalization.
- **Interactive Streamlit Web Dashboard:**
  - Clean White & Navy Blue clinical theme.
  - Patient parameter input form with presets (Low Risk, Borderline, High Risk).
  - Real-time risk cards, probability meters, and biomarker alert signals.
  - Model performance analytics with interactive ROC curves, confusion matrices, and loss convergence curves.
  - Cohort data insights and feature correlation charts.
- **One-Click Windows Launcher (`start.bat`):** Automated virtual environment creation, dependency installation, model verification, and browser startup.
- **Zero Online Retraining:** App loads pre-trained serialized model artifacts (`.keras` and `.joblib`) instantaneously.
- **Comprehensive Test Suite:** High-coverage unit tests for data preprocessing, neural network compilation, and inference services.

---

## 3. Dataset & Features

- **Source:** UCI Machine Learning Repository (Dataset ID: 336 - _Chronic Kidney Disease_).
- **Sample Size:** 400 patient records (398 validated cases after label normalization; 80% Train, 20% Held-out Test).
- **Predictor Features (22 total):**
  - **Vitals & Demographics:** Age, Resting Blood Pressure (`bp`).
  - **Blood & Glycemic Tests:** Random Blood Glucose (`bgr`), Blood Urea (`bu`), Serum Creatinine (`sc`), Serum Sodium (`sod`), Serum Potassium (`pot`).
  - **Hematology:** Hemoglobin (`hemo`), Packed Cell Volume (`pcv`), White Blood Cell Count (`wbcc`), Red Blood Cell Count (`rbcc`).
  - **Urinalysis:** Specific Gravity (`sg`), Albumin (`al`), Sugar (`su`).
  - **Microscopic & Clinical History:** Red Blood Cells (`rbc`), Pus Cells (`pc`), Pus Cell Clumps (`pcc`), Bacteria (`ba`), Coronary Artery Disease (`cad`), Appetite (`appet`), Pedal Edema (`pe`), Anemia (`ane`).

---

## 4. Deep Neural Network Architecture

```
Input [64 Dimensions: 37 engineered features + missing indicators]
  │
Dense(128, L2=3e-4) ──► BatchNormalization ──► ReLU ──► Dropout(0.35)
  │
Dense(64, L2=3e-4)  ──► BatchNormalization ──► ReLU ──► Dropout(0.25)
  │
Dense(32, L2=3e-4)  ──► BatchNormalization ──► ReLU ──► Dropout(0.15)
  ├──► Output 1: Dense(1, Sigmoid, name="diabetes")
  └──► Output 2: Dense(1, Sigmoid, name="hypertension")
```

- **Loss Function:** Dual `BinaryCrossentropy` with `label_smoothing=0.05` to mitigate overconfident probabilities.
- **Optimization:** Adam ($\alpha = 10^{-3}$) with `ReduceLROnPlateau(factor=0.5, patience=12)` and `EarlyStopping(patience=40, restore_best_weights=True)`.
- **Class Balancing:** Inverse frequency sample weights applied per target.

---

## 5. Quantitative Evaluation Results

Evaluated on the 20% held-out test split ($N=80$ patients):

| Target Condition | Model Architecture          | Accuracy   | Precision  | Recall     | Specificity | F1-Score   | ROC-AUC    | MCC        |
| ---------------- | --------------------------- | ---------- | ---------- | ---------- | ----------- | ---------- | ---------- | ---------- |
| **Diabetes**     | Multi-Output Neural Network | **78.75%** | **65.71%** | **82.14%** | **76.92%**  | **0.7302** | **0.8695** | **0.5679** |
| **Diabetes**     | XGBoost + LightGBM Blend    | **81.25%** | **72.41%** | **75.00%** | **84.62%**  | **0.7368** | **0.9114** | **0.5915** |
| **Hypertension** | Multi-Output Neural Network | **81.25%** | **70.27%** | **86.67%** | **78.00%**  | **0.7761** | **0.8747** | **0.6279** |
| **Hypertension** | XGBoost + LightGBM Blend    | **83.75%** | **75.76%** | **83.33%** | **84.00%**  | **0.7937** | **0.9113** | **0.6622** |

---

## 6. Project Structure

```
diabetes-risk-prediction/
│
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pinned Python package dependencies
├── pyproject.toml                 # Package metadata and build configuration
├── .gitignore                     # Git tracking exclusions
├── LICENSE                        # MIT License
├── start.bat                      # Windows one-click double-executable launcher
│
├── configs/
│   └── model_config.yaml          # Hyperparameters, architecture & thresholds
│
├── data/
│   ├── raw/                       # Cached raw CSV dataset (UCI #336)
│   ├── processed/                 # Train (N=318) and Test (N=80) splits
│   └── README.md                  # Dataset origin and feature definitions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   └── diabetes_prediction/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load_data.py       # UCI ingestion and caching
│       │   └── preprocessing.py   # Cleaning, feature engineering & scaling
│       ├── models/
│       │   ├── __init__.py
│       │   ├── neural_network.py  # Multi-output Keras model architecture
│       │   └── train.py           # Training loops and callbacks
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py         # Diagnostic classification metrics
│       │   └── plots.py           # ROC, confusion matrix & history plots
│       └── predict.py             # Production inference service
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py           # Main multi-page Streamlit dashboard
│   └── components/
│       ├── __init__.py
│       ├── input_form.py          # Clinical input form and patient presets
│       └── visualizations.py      # Risk gauge cards and metric tables
│
├── scripts/
│   ├── train.py                   # CLI training and evaluation pipeline
│   └── evaluate.py                # Standalone test set evaluation
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # Unit tests for preprocessing & cleaning
│   ├── test_model.py              # Unit tests for neural network architecture
│   └── test_predict.py            # Unit tests for prediction and inference
│
├── models/
│   ├── .gitkeep
│   ├── multioutput_nn.keras       # Trained Keras multi-output neural network
│   ├── preprocessor.joblib        # Fitted preprocessor (imputer + quantile tf)
│   └── ensemble_models.joblib     # Trained XGBoost + LightGBM ensemble
│
├── reports/
│   ├── figures/                   # Generated evaluation plots (ROC, CM, Loss)
│   └── results/                   # Evaluation metrics JSON and training log
│
└── docs/
    └── methodology.md             # In-depth clinical and mathematical methodology
```

---

## 7. Installation & Quickstart

### Method A: One-Click Launch on Windows (`start.bat`)

Simply double-click `start.bat` in the project root.

The launcher will automatically:

1. Verify that Python is installed.
2. Create an isolated virtual environment (`.venv`).
3. Install required packages from `requirements.txt`.
4. Check that trained model artifacts exist (or train them automatically if missing).
5. Launch the Streamlit application and open your default web browser at `http://localhost:8501`.

### Method B: Manual Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd diabetes-risk-prediction
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Train the models (if not already trained):**

   ```bash
   python scripts/train.py
   ```

5. **Launch the Streamlit GUI:**

   ```bash
   streamlit run app/streamlit_app.py
   ```

6. **Run Unit Tests:**
   ```bash
   python -m unittest discover -s tests -v
   ```

---

## 8. Standalone Model Training & Inference API

### Retraining Models

To run data cleaning, train the neural network, fit tree ensembles, and update test evaluation metrics:

```bash
python scripts/train.py
```

### Programmatic Python Inference

```python
from diabetes_prediction.predict import RiskPredictor

# Initialize predictor (loads weights and preprocessor)
predictor = RiskPredictor()

# Patient profile
patient = {
    "age": 52.0, "bp": 85.0, "bgr": 155.0, "bu": 42.0, "sc": 1.3,
    "sod": 136.0, "pot": 4.6, "hemo": 13.2, "pcv": 39.0, "wbcc": 8100.0,
    "rbcc": 4.4, "sg": 1.015, "al": 1.0, "su": 1.0, "rbc": "normal",
    "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "cad": "no",
    "appet": "good", "pe": "no", "ane": "no"
}

# Run inference
result = predictor.predict(patient)
print("Diabetes Risk:", result["diabetes"]["risk_level"], f"({result['diabetes']['probability']:.1%})")
print("Hypertension Risk:", result["hypertension"]["risk_level"], f"({result['hypertension']['probability']:.1%})")
```

---

## 9. Limitations & Future Roadmap

- **Cohort Scale:** The current model is trained on the UCI Chronic Kidney Disease cohort ($N=400$). While data quality and validation rigor are high, external generalization on diverse populations requires continuous validation.
- **Missing Value Imputation:** While median imputation with missingness indicators effectively models missingness bias, deep generative imputation (e.g., GAIN or MICE) could be explored.
- **Continuous Monitoring Integration:** Future iterations could connect to FHIR/HL7 electronic health record (EHR) feeds for continuous inpatient risk tracking.

---

## 10. Medical Disclaimer

> **IMPORTANT:** This software application and deep learning models are developed strictly for **educational, experimental, and scientific research purposes**. The predictions, probabilities, and risk categories generated by this system do NOT constitute medical diagnosis, clinical judgment, or treatment recommendations. Always seek the advice of qualified healthcare professionals for medical conditions and diagnostic assessments.
