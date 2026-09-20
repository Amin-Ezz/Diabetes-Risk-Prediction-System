# Dataset Documentation: CardioMetabolic & Chronic Kidney Disease Cohort

## 1. Dataset Origin & Overview

* **Source:** UCI Machine Learning Repository (Dataset ID: 336)
* **Title:** Chronic Kidney Disease Dataset
* **Cohort Size:** 400 patient records
* **Clinical Predictors:** 22 active features used for Diabetes and Hypertension prediction
* **Target Variables:**
  * `dm` (Diabetes Mellitus): Binary outcome (0 = Normal, 1 = Diabetic)
  * `htn` (Hypertension): Binary outcome (0 = Normal, 1 = Hypertensive)
  * `class` (Chronic Kidney Disease): 1 = ckd, 0 = notckd

## 2. Accessing & Ingesting the Data

The raw dataset is automatically fetched and cached locally when executing the data loader or training pipeline:

```bash
python scripts/train.py
```

Under the hood, `src/diabetes_prediction/data/load_data.py` uses `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=336)
```

If internet connectivity is unavailable, the system automatically falls back to the local cached CSV at:
`data/raw/chronic_kidney_disease.csv`.

## 3. Directory Layout

```
data/
├── raw/
│   └── chronic_kidney_disease.csv    # Cached raw cohort from UCI ML repo
├── processed/
│   ├── train.csv                     # 80% stratified training split (N=318)
│   └── test.csv                      # 20% stratified held-out test split (N=80)
└── README.md                         # This dataset documentation
```

## 4. Clinical Features Reference

| Symbol | Clinical Parameter | Type | Reference / Allowed Values |
|---|---|---|---|
| `age` | Patient Age | Numerical | 1 – 120 years |
| `bp` | Blood Pressure | Numerical | 40 – 260 mm Hg |
| `sg` | Urinary Specific Gravity | Categorical/Discrete | 1.005, 1.010, 1.015, 1.020, 1.025 |
| `al` | Albumin (Urinalysis) | Discrete (0-5) | 0, 1, 2, 3, 4, 5 |
| `su` | Sugar (Urinalysis) | Discrete (0-5) | 0, 1, 2, 3, 4, 5 |
| `rbc` | Red Blood Cells in Urine | Categorical | normal, abnormal |
| `pc` | Pus Cells in Urine | Categorical | normal, abnormal |
| `pcc` | Pus Cell Clumps | Categorical | present, notpresent |
| `ba` | Bacteria | Categorical | present, notpresent |
| `bgr` | Blood Glucose Random | Numerical | 30 – 600 mg/dL |
| `bu` | Blood Urea | Numerical | 1 – 500 mg/dL |
| `sc` | Serum Creatinine | Numerical | 0.1 – 100 mg/dL |
| `sod` | Serum Sodium | Numerical | 100 – 180 mEq/L (decimal shifts repaired) |
| `pot` | Serum Potassium | Numerical | 1.5 – 12.0 mEq/L (decimal shifts repaired) |
| `hemo` | Hemoglobin | Numerical | 3 – 25 g/dL |
| `pcv` | Packed Cell Volume | Numerical | 5 – 70 % |
| `wbcc` | White Blood Cell Count | Numerical | 1,000 – 100,000 /cumm |
| `rbcc` | Red Blood Cell Count | Numerical | 1.0 – 10.0 m/cumm |
| `cad` | Coronary Artery Disease | Categorical | yes, no |
| `appet` | Patient Appetite | Categorical | good, poor |
| `pe` | Pedal Edema | Categorical | yes, no |
| `ane` | Anemia | Categorical | yes, no |

## 5. Ethical & Scientific Disclaimer

This dataset is utilized for retrospective algorithmic benchmarking and educational modeling. Data has been de-identified at source by original study investigators.
