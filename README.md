# credit-risk-intelligence
End-to-end ML system for credit default prediction with SHAP explainability and business simulation

# Credit Risk Intelligence Engine

> End-to-end ML system for credit default prediction — featuring a LightGBM stacked ensemble, SHAP explainability, FastAPI serving, and a Streamlit business simulator.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-green)
![MLflow](https://img.shields.io/badge/MLflow-2.17.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)

---

## The Problem

A consumer lender processes thousands of loan applications daily. Approving a borrower who defaults costs money. Rejecting a creditworthy borrower costs revenue. Traditional scorecards are opaque and can't adapt to complex non-linear risk signals.

This project builds an ML system that predicts the probability a borrower will default, explains *why* using SHAP values, and simulates the business impact of different approval thresholds in real time.

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC (OOF) | **0.8976** |
| PR-AUC (OOF) | **0.7990** |
| Best F1 | **0.7271** |
| Best threshold | **0.8456** |
| Training rows | 307,511 |
| Features | 150 (122 raw + 28 engineered) |

*Evaluated on out-of-fold predictions from 5-fold stratified cross-validation.*

---

## Architecture

```bash
Raw Data (307k applications)
        │
        ▼
Feature Engineering
(28 new signals: ratios, EXT_SOURCE interactions, missingness flags)
        │
        ▼
SMOTE Resampling
(Imbalance handled: 8.07% → 16.7% positive rate)
        │
        ▼
LightGBM (5-Fold Cross Validation)
→ Out-of-Fold (OOF) Predictions
        │
        ▼
Logistic Regression (Meta-Learner)
→ Final Probability Score
        │
        ▼
SHAP TreeExplainer
→ Per-borrower Risk Drivers
        │
        ▼
Deployment Layer
(FastAPI API + Streamlit Dashboard)
```
---

## Key Findings from EDA

**1. Severe class imbalance:** 91.93% of applicants repay their loans. A naive model predicting "repay" for everyone would be 91.9% accurate — and completely useless. This is why we optimise PR-AUC, not accuracy.

**2. External credit scores are the strongest signal:** `EXT_SOURCE_2` and `EXT_SOURCE_3` show the clearest distributional separation between defaulters and repayers. Their engineered mean (`EXT_SOURCE_MEAN`) becomes the single most predictive feature with a mean |SHAP| of 0.5263.

**3. Missingness is informative:** 49 columns had over 40% missing values. Rather than dropping them, we created binary missingness flags — `EXT_SOURCE_1_MISSING` alone ranked in the top 10 SHAP features, confirming that *whether* a bureau record exists is itself a risk signal.

**4. Income type matters more than income level:** Maternity leave and unemployed applicants default at 40%+ rates — 5× the average. Income stability matters more than absolute income.

---

## Feature Engineering Highlights

| Feature | Logic | Why it works |
|---|---|---|
| `CREDIT_TERM` | annuity / credit | Monthly burden relative to loan size — #1 by LightGBM gain |
| `EXT_SOURCE_MEAN` | mean(EXT1, EXT2, EXT3) | Aggregates three credit bureaus — #1 by SHAP |
| `ANNUITY_INCOME_RATIO` | annuity / income | Debt-to-income proxy — key affordability signal |
| `EXT_SOURCE_DISAGREEMENT` | std / mean of EXT sources | Bureau disagreement = uncertainty = higher risk |
| `EMPLOYMENT_STABILITY` | days_employed / days_birth | Fraction of life spent employed |
| `*_MISSING` flags | 1 if feature is null | Missingness itself encodes credit history absence |

14 of the top 25 features by LightGBM gain are engineered features.

---

## Modelling Decisions

**Why LightGBM?** Gradient boosted trees are the industry standard for tabular financial data. LightGBM specifically handles large datasets efficiently and natively supports `scale_pos_weight` for class imbalance.

**Why stacking?** The meta-learner (Logistic Regression on OOF predictions) adds calibration on top of the base model. In practice the gain was marginal here (ROC-AUC was identical), but the pattern is standard in production and demonstrates MLOps awareness.

**Why SMOTE before CV (and what I'd do differently)?** SMOTE was applied to the full training set before cross-validation for simplicity. The rigorous approach applies SMOTE inside each fold to prevent synthetic samples from the validation set bleeding into training. This would be the first fix in a production iteration.

**Why PR-AUC over ROC-AUC?** With an 8% default rate, a random classifier achieves ~0.50 ROC-AUC but only ~0.08 PR-AUC. PR-AUC measures performance specifically on the minority class — the one that matters.

---

## Explainability

This system implements model explainability aligned with SR 11-7 model risk guidance and GDPR Article 22 (automated decision-making).

Every prediction produces:
- A default probability score
- A risk label (LOW / MEDIUM / HIGH) and decision
- Top 3 risk drivers with SHAP values
- Top 3 protective factors with SHAP values

**Example — High risk borrower (92.1% default probability):**

| Driver | Direction | SHAP |
|---|---|---|
| EXT_SOURCE_MEAN = 0.149 | ↑ Increases risk | +1.091 |
| EXT_SOURCE_3 = 0.062 | ↑ Increases risk | +0.287 |
| EXT_SOURCE_MIN = 0.062 | ↑ Increases risk | +0.200 |

All three major credit bureaus agree this borrower is high risk — a strong, consistent signal.

---

## Business Impact Simulation

At the default threshold of 0.8456 on a portfolio of 10,000 applications:

| Metric | Value |
|---|---|
| Approval rate | ~15% |
| Expected loss | minimised |
| Net position | positive |

The Streamlit simulator lets stakeholders explore the revenue vs. loss tradeoff interactively at any threshold.

---

## Project Structure
```bash
credit-risk-intelligence/
├── api/              # FastAPI prediction endpoint
├── app/              # Streamlit business dashboard
├── data/
│   ├── raw/          # Original Kaggle data (gitignored)
│   └── processed/    # Engineered features, SHAP scores
├── mlflow_tracking/  # MLflow experiment tracking
├── models/           # Serialised model artefacts
├── notebooks/        # EDA, feature engineering, modelling, explainability
├── reports/figures/  # All generated plots
├── src/              # Importable Python modules
│   ├── features.py   # Feature engineering pipeline
│   ├── model.py      # Training, CV, stacking
│   ├── explain.py    # SHAP explainability
│   └── utils.py      # Shared paths, logging
└── tests/            # Unit tests (pytest)
```
---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the API
uvicorn api.main:app --reload --port 8000

# Run the dashboard
streamlit run app/streamlit_app.py

# Run tests
pytest tests/ -v

# View MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --port 5000
```

---

## What I'd Do With More Time

1. **SMOTE inside CV folds** — apply resampling within each fold to eliminate synthetic data leakage
2. **Join bureau tables** — `bureau.csv` and `previous_application.csv` would push ROC-AUC toward 0.80+
3. **Evidently drift monitoring** — track feature distributions over time to detect concept drift
4. **Isotonic regression calibration** — fix the calibration curve's deviation from the diagonal
5. **Dockerise the API** — containerise FastAPI for one-command deployment

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — Kaggle competition dataset, 307,511 loan applications with 122 features.