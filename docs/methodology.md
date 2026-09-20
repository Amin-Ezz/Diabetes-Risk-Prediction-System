# Clinical Deep Learning Methodology & Technical Specification

## 1. Problem Formulation

Diabetes Mellitus (DM) and Essential Hypertension (HTN) represent the two most prevalent and damaging comorbidities in nephrology and internal medicine. Rather than framing them as isolated binary classification tasks, our system models them as a **joint multi-task predictive problem** utilizing 22 routine clinical predictors.

$$\hat{y}_{\text{dm}}, \hat{y}_{\text{htn}} = f_{\boldsymbol{\theta}}(\mathbf{x})$$

where $\mathbf{x} \in \mathbb{R}^{22}$ denotes patient clinical measurements and $f_{\boldsymbol{\theta}}$ is a multi-output deep neural network parameterized by weights $\boldsymbol{\theta}$.

---

## 2. Data Cleaning & Clinical Quality Assurance

Raw clinical registries frequently suffer from clerical typographical errors and whitespace corruption. The pipeline implements:

1. **Deep String Normalization:** Strips carriage returns, embedded tabs (`\t`), trailing spaces, enforces lowercase, and validates each categorical entry against clinically established enumerations (`ALLOWED_VALUES`).
2. **Decimal-Shift Error Repair:** Identified biological implausibilities resulting from transcription errors:
   * Serum Sodium (`sod`): Values $< 100 \text{ mEq/L}$ (e.g. $4.5$) were multiplied by 10 to restore the true clinical scale ($45 \to \text{median correction}$).
   * Serum Potassium (`pot`): Values $> 12 \text{ mEq/L}$ (e.g. $39.0, 47.0$) were divided by 10 ($3.9, 4.7 \text{ mEq/L}$).
3. **Unlabeled Sample Removal:** Two records missing both training target labels (`dm` and `htn`) were discarded, yielding $N=398$ validated cases.

---

## 3. Clinical Feature Engineering

To enrich tabular representation without inducing lookahead bias or data leakage, three categories of row-wise transformations are applied:

### A. Physiological Ratios & Biomarker Interactions
* **Urea-to-Creatinine Ratio (`bu_sc_ratio`):**
  $$\text{bu\_sc\_ratio} = \frac{\text{bu}}{\text{sc} + 10^{-3}}$$
  Differentiates pre-renal azotemia from intrinsic renal parenchymal injury.
* **Hematological Anemia Index (`hemo_x_pcv`):**
  $$\text{hemo\_x\_pcv} = \text{hemo} \times \text{pcv}$$
* **Glycemic Multiplier (`bgr_x_su`):**
  $$\text{bgr\_x\_su} = \text{bgr} \times (\text{su} + 1)$$
  Amplifies subtle hyperglycemic signals combining blood glucose and glucosuria.

### B. Clinical Threshold Flags (NaN-Aware)
* $\text{bp\_high} = \mathbb{I}(\text{bp} \ge 90 \text{ mm Hg})$
* $\text{bgr\_high} = \mathbb{I}(\text{bgr} \ge 180 \text{ mg/dL})$
* $\text{anemia} = \mathbb{I}(\text{hemo} < 12.0 \text{ g/dL})$

### C. Missingness Pattern Indicators
In clinical workflows, whether a lab test was ordered often correlates with disease acuity. Binary missing indicators are computed for 9 high-missing columns:
$$\mathbf{m}_c = \mathbb{I}(x_c \text{ is missing}) \quad \forall c \in \{\text{rbc}, \text{rbcc}, \text{wbcc}, \text{sod}, \text{pot}, \text{pcv}, \text{hemo}, \text{bgr}, \text{bp}\}$$

Expands tabular features from 22 to 37 dimensions.

---

## 4. Leakage-Free Preprocessing Pipeline

To eliminate any data leakage between train and test sets:
1. **Stratified Partitioning:** A joint stratification variable $\text{joint} = y_{\text{dm}} \oplus y_{\text{htn}}$ ensures equal class ratios across both targets in the 80% train ($N=318$) and 20% test ($N=80$) partitions (`random_state=42`).
2. **Median Imputation with Missing Indicators:** `SimpleImputer(strategy="median", add_indicator=True)` fits strictly on train folds and generates 64 transformed features.
3. **Gaussian Quantile Transformation:** `QuantileTransformer(output_distribution="normal", n_quantiles=300)` maps arbitrary empirical distributions to standard normal, mitigating the distortion of biological outliers without discarding genuine patient signals.

---

## 5. Neural Network Architecture & Regularization

```
Input [64 dimensions]
  │
Dense(128, L2=3e-4) ──► BatchNorm ──► ReLU ──► Dropout(0.35)
  │
Dense(64, L2=3e-4)  ──► BatchNorm ──► ReLU ──► Dropout(0.25)
  │
Dense(32, L2=3e-4)  ──► BatchNorm ──► ReLU ──► Dropout(0.15)
  ├──► Output Head 1: Dense(1, Sigmoid) ──► Diabetes Probability
  └──► Output Head 2: Dense(1, Sigmoid) ──► Hypertension Probability
```

### Regularization & Optimization Details:
* **Kernel Regularization:** $L_2 = 3 \times 10^{-4}$ across all dense layers.
* **Batch Normalization:** Applied before non-linear activations to stabilize intermediate covariate shifts.
* **Label Smoothing:** Both heads use Binary Cross-Entropy with $\epsilon = 0.05$ label smoothing to prevent overconfident probabilistic calibration.
* **Class Weighting:** Balanced inverse frequency sample weights per target.
* **Learning Rate Schedule:** Adam ($\text{lr} = 10^{-3}$) with `ReduceLROnPlateau(factor=0.5, patience=12, min_lr=1e-6)`.
* **Early Stopping:** Monitored on validation loss with `patience=40` and automatic restoration of best weights.

---

## 6. Tree Ensembles & Threshold Calibration

For tabular benchmarking, Optuna-tuned XGBoost and LightGBM models are trained across 3 random seeds (42, 7, 123) and blended.

Optimal decision thresholds are established using 5-fold Out-Of-Fold (OOF) cross-validation exclusively on the training split:
* **Diabetes Mellitus:** Blended probability with decision cutoff $T_{\text{dm}} = 0.50$.
* **Hypertension:** Blended probability with decision cutoff $T_{\text{htn}} = 0.39$.

---

## 7. Experimental Results on Held-out Test Set (N=80)

| Condition | Architecture | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| **Diabetes** | Multi-Output Neural Network | **78.75%** | **65.71%** | **82.14%** | **76.92%** | **0.7302** | **0.8695** | **0.5679** |
| **Diabetes** | XGBoost + LightGBM Blend | **81.25%** | **72.41%** | **75.00%** | **84.62%** | **0.7368** | **0.9114** | **0.5915** |
| **Hypertension** | Multi-Output Neural Network | **81.25%** | **70.27%** | **86.67%** | **78.00%** | **0.7761** | **0.8747** | **0.6279** |
| **Hypertension** | XGBoost + LightGBM Blend | **83.75%** | **75.76%** | **83.33%** | **84.00%** | **0.7937** | **0.9113** | **0.6622** |
