"""Clinical input form component for Diabetes and Hypertension risk assessment."""

from typing import Any, Dict
import streamlit as st

# Preset patient scenarios for convenient clinical exploration
PRESETS = {
    "Normal / Low Risk": {
        "age": 35.0,
        "bp": 70.0,
        "sg": 1.025,
        "al": 0.0,
        "su": 0.0,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "bgr": 92.0,
        "bu": 24.0,
        "sc": 0.8,
        "sod": 140.0,
        "pot": 4.2,
        "hemo": 15.6,
        "pcv": 46.0,
        "wbcc": 6400.0,
        "rbcc": 5.1,
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
    },
    "Borderline / Elevated Glucose": {
        "age": 52.0,
        "bp": 85.0,
        "sg": 1.015,
        "al": 1.0,
        "su": 1.0,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "bgr": 155.0,
        "bu": 42.0,
        "sc": 1.3,
        "sod": 136.0,
        "pot": 4.6,
        "hemo": 13.2,
        "pcv": 39.0,
        "wbcc": 8100.0,
        "rbcc": 4.4,
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
    },
    "High Risk (Diabetic & Hypertensive)": {
        "age": 64.0,
        "bp": 100.0,
        "sg": 1.010,
        "al": 3.0,
        "su": 3.0,
        "rbc": "abnormal",
        "pc": "abnormal",
        "pcc": "present",
        "ba": "notpresent",
        "bgr": 240.0,
        "bu": 78.0,
        "sc": 2.8,
        "sod": 132.0,
        "pot": 5.4,
        "hemo": 9.5,
        "pcv": 29.0,
        "wbcc": 11500.0,
        "rbcc": 3.4,
        "cad": "yes",
        "appet": "poor",
        "pe": "yes",
        "ane": "yes",
    },
}


def render_input_form() -> Dict[str, Any]:
    """Render structured clinical input widgets with validation and presets."""
    st.subheader("📋 Patient Clinical Parameters")

    col_pre, _ = st.columns([2, 1])
    with col_pre:
        preset_choice = st.selectbox(
            "Load Sample Patient Profile (Optional):",
            options=["Custom Manual Entry", "Normal / Low Risk", "Borderline / Elevated Glucose", "High Risk (Diabetic & Hypertensive)"],
            index=0,
            help="Select a representative clinical profile or enter patient values manually.",
        )

    defaults = PRESETS.get(preset_choice, PRESETS["Normal / Low Risk"]) if preset_choice != "Custom Manual Entry" else PRESETS["Normal / Low Risk"]

    # Direct styling to guarantee step buttons (+ / -) render as distinct full square boxes
    st.markdown(
        """
        <style>
            div[data-testid="stNumberInputContainer"] {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                align-items: center !important;
                background-color: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 8px !important;
                padding: 3px 8px 3px 4px !important;
                height: 44px !important;
                min-height: 44px !important;
                max-height: 44px !important;
                box-sizing: border-box !important;
                overflow: visible !important;
            }
            div[data-testid="stNumberInputContainer"] input {
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
                opacity: 1 !important;
                flex-shrink: 0 !important;
            }
            button[data-testid="stNumberInputStepDown"]:hover,
            button[data-testid="stNumberInputStepUp"]:hover,
            div[data-testid="stNumberInputContainer"] button:hover {
                background-color: #e2e8f0 !important;
                border-color: #94a3b8 !important;
                color: #0f172a !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.form("clinical_prediction_form"):
        # Section 1: Demographics & Blood Pressure
        st.markdown("#### 1. Demographics & Hemodynamics")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Age (years)",
                min_value=1.0,
                max_value=110.0,
                value=float(defaults["age"]),
                step=1.0,
                help="Patient age in years (1 - 110).",
            )
        with c2:
            bp = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=40.0,
                max_value=240.0,
                value=float(defaults["bp"]),
                step=5.0,
                help="Resting diastolic blood pressure (mm Hg). Clinical threshold: >= 90 mm Hg indicates hypertension risk.",
            )

        # Section 2: Glycemic & Renal Serum Chemistry
        st.markdown("#### 2. Glycemic & Renal Metabolic Panel")
        c3, c4, c5 = st.columns(3)
        with c3:
            bgr = st.number_input(
                "Blood Glucose Random (bgr, mg/dL)",
                min_value=40.0,
                max_value=550.0,
                value=float(defaults["bgr"]),
                step=5.0,
                help="Random blood glucose in mg/dL. Primary biomarker for diabetes.",
            )
        with c4:
            bu = st.number_input(
                "Blood Urea (bu, mg/dL)",
                min_value=5.0,
                max_value=400.0,
                value=float(defaults["bu"]),
                step=2.0,
                help="Serum urea concentration (mg/dL). Reference range: 10-50 mg/dL.",
            )
        with c5:
            sc = st.number_input(
                "Serum Creatinine (sc, mg/dL)",
                min_value=0.1,
                max_value=50.0,
                value=float(defaults["sc"]),
                step=0.1,
                help="Serum creatinine (mg/dL). Reference range: 0.6-1.3 mg/dL.",
            )

        c6, c7 = st.columns(2)
        with c6:
            sod = st.number_input(
                "Serum Sodium (sod, mEq/L)",
                min_value=100.0,
                max_value=175.0,
                value=float(defaults["sod"]),
                step=1.0,
                help="Sodium ion level (mEq/L). Reference range: 135-145 mEq/L.",
            )
        with c7:
            pot = st.number_input(
                "Serum Potassium (pot, mEq/L)",
                min_value=1.5,
                max_value=10.0,
                value=float(defaults["pot"]),
                step=0.1,
                help="Potassium ion level (mEq/L). Reference range: 3.5-5.2 mEq/L.",
            )

        # Section 3: Hematology & Urinalysis
        st.markdown("#### 3. Hematology & Urinalysis")
        c8, c9, c10, c11 = st.columns(4)
        with c8:
            hemo = st.number_input(
                "Hemoglobin (hemo, g/dL)",
                min_value=3.0,
                max_value=22.0,
                value=float(defaults["hemo"]),
                step=0.2,
                help="Hemoglobin concentration. Values < 12 g/dL signal anemia.",
            )
        with c9:
            pcv = st.number_input(
                "Packed Cell Volume (pcv, %)",
                min_value=10.0,
                max_value=65.0,
                value=float(defaults["pcv"]),
                step=1.0,
                help="Hematocrit percentage.",
            )
        with c10:
            wbcc = st.number_input(
                "White Blood Cell Count (wbcc, /cumm)",
                min_value=1500.0,
                max_value=40000.0,
                value=float(defaults["wbcc"]),
                step=200.0,
                help="Leukocyte count. Normal: 4,000-11,000 cells/cumm.",
            )
        with c11:
            rbcc = st.number_input(
                "Red Blood Cell Count (rbcc, m/cumm)",
                min_value=1.5,
                max_value=8.5,
                value=float(defaults["rbcc"]),
                step=0.1,
                help="Erythrocyte count (millions/cumm).",
            )

        c12, c13, c14 = st.columns(3)
        with c12:
            sg = st.selectbox(
                "Urinary Specific Gravity (sg)",
                options=[1.005, 1.010, 1.015, 1.020, 1.025],
                index=[1.005, 1.010, 1.015, 1.020, 1.025].index(defaults["sg"]),
                help="Specific gravity of urine.",
            )
        with c13:
            al = st.selectbox(
                "Albumin in Urine (al, scale 0-5)",
                options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                index=int(defaults["al"]),
                help="Microalbuminuria / Proteinuria degree.",
            )
        with c14:
            su = st.selectbox(
                "Sugar in Urine (su, scale 0-5)",
                options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                index=int(defaults["su"]),
                help="Glucosuria degree (presence of glucose in urine).",
            )

        # Section 4: Clinical History & Microscopic Findings
        st.markdown("#### 4. Clinical History & Microscopic Findings")
        c15, c16, c17, c18 = st.columns(4)
        with c15:
            rbc = st.selectbox(
                "Red Blood Cells in Urine (rbc)",
                options=["normal", "abnormal"],
                index=0 if defaults["rbc"] == "normal" else 1,
            )
        with c16:
            pc = st.selectbox(
                "Pus Cells in Urine (pc)",
                options=["normal", "abnormal"],
                index=0 if defaults["pc"] == "normal" else 1,
            )
        with c17:
            pcc = st.selectbox(
                "Pus Cell Clumps (pcc)",
                options=["notpresent", "present"],
                index=0 if defaults["pcc"] == "notpresent" else 1,
            )
        with c18:
            ba = st.selectbox(
                "Bacteria in Urine (ba)",
                options=["notpresent", "present"],
                index=0 if defaults["ba"] == "notpresent" else 1,
            )

        c19, c20, c21, c22 = st.columns(4)
        with c19:
            cad = st.selectbox(
                "Coronary Artery Disease (cad)",
                options=["no", "yes"],
                index=0 if defaults["cad"] == "no" else 1,
            )
        with c20:
            appet = st.selectbox(
                "Patient Appetite (appet)",
                options=["good", "poor"],
                index=0 if defaults["appet"] == "good" else 1,
            )
        with c21:
            pe = st.selectbox(
                "Pedal Edema (pe)",
                options=["no", "yes"],
                index=0 if defaults["pe"] == "no" else 1,
                help="Swelling in the feet and lower extremities.",
            )
        with c22:
            ane = st.selectbox(
                "Clinical Anemia (ane)",
                options=["no", "yes"],
                index=0 if defaults["ane"] == "no" else 1,
            )

        submitted = st.form_submit_button("🔍 Run Dual Risk Assessment", use_container_width=True)

    if submitted:
        patient_data = {
            "age": float(age),
            "bp": float(bp),
            "sg": float(sg),
            "al": float(al),
            "su": float(su),
            "rbc": rbc,
            "pc": pc,
            "pcc": pcc,
            "ba": ba,
            "bgr": float(bgr),
            "bu": float(bu),
            "sc": float(sc),
            "sod": float(sod),
            "pot": float(pot),
            "hemo": float(hemo),
            "pcv": float(pcv),
            "wbcc": float(wbcc),
            "rbcc": float(rbcc),
            "cad": cad,
            "appet": appet,
            "pe": pe,
            "ane": ane,
        }
        return {"submitted": True, "data": patient_data}

    return {"submitted": False, "data": None}
