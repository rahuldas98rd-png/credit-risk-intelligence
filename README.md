# Credit Risk Intelligence Engine

> An end-to-end machine learning system that predicts loan default probability, explains every prediction with SHAP values, and lets stakeholders simulate the business impact of approval thresholds in real time.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-2E7D32)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45.1-9C27B0)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.17.2-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🚀 Live Demos

| Service | URL |
|---|---|
| 📊 **Interactive Dashboard** | [credit-risk-intelligence.streamlit.app](https://credit-risk-intelligence-yp7wp3dpfhbnnjzpzrnfxs.streamlit.app) |
| 🔌 **REST API (Swagger UI)** | [credit-risk-intelligence-vspf.onrender.com/docs](https://credit-risk-intelligence-vspf.onrender.com/docs) |

> **Note:** The free Render tier spins down after 15 minutes of inactivity. The first API request after a cold start may take ~30 seconds to wake up.

---

## 📋 Table of Contents

- [The Problem](#the-problem)
- [Solution Overview](#solution-overview)
- [Headline Results](#headline-results)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Key Findings from EDA](#key-findings-from-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Explainability](#explainability)
- [Business Impact Simulator](#business-impact-simulator)
- [Getting Started](#getting-started)
- [API Usage](#api-usage)
- [Testing](#testing)
- [What I'd Do With More Time](#what-id-do-with-more-time)
- [Dataset & References](#dataset--references)
- [License](#license)

---

## The Problem

A consumer lender processes thousands of loan applications every day. Two failure modes destroy value in opposite directions:

- **Approving a borrower who defaults** — direct financial loss equal to the unpaid principal × loss-given-default rate.
- **Rejecting a creditworthy borrower** — foregone interest revenue and reputational cost in a competitive market.

Traditional rule-based scorecards are interpretable but rigid. They miss complex non-linear patterns and can't adapt as borrower behaviour shifts. A black-box ML model captures more signal but creates a compliance problem under SR 11-7 (model risk management) and GDPR Article 22 (right to explanation for automated decisions).

This project resolves the tradeoff: a high-performance gradient boosted ensemble paired with SHAP explainability and an interactive business simulator, so risk officers can see *both* the prediction *and* the reasoning before any loan decision is made.

---

## Solution Overview

A production-style three-tier system:

1. **Modelling layer** — a LightGBM stacked ensemble trained with 5-fold stratified cross-validation, tracked end-to-end in MLflow.
2. **Explainability layer** — SHAP TreeExplainer generates per-borrower risk drivers and protective factors for every prediction.
3. **Application layer** — a FastAPI serving endpoint exposes predictions as JSON; a Streamlit dashboard provides three interactive views: a business simulator, a borrower explainer, and a model insights page.

Both services are deployed and publicly accessible (links above).

---

## Headline Results

| Metric | Value | What it means |
|---|---|---|
| **ROC-AUC (OOF)** | **0.8976** | Excellent discrimination between defaulters and repayers |
| **PR-AUC (OOF)** | **0.7990** | ~10× lift over the 0.08 random baseline on this imbalanced dataset |
| **Best F1** | **0.7271** | Optimal balance of precision and recall at threshold 0.8456 |
| **Brier Score** | 0.1537 | Probability calibration quality |
| Training rows | 307,511 | Full Home Credit application table |
| Total features | 150 | 122 raw + 28 engineered |

*All metrics computed on out-of-fold predictions from 5-fold stratified cross-validation — no train-test contamination.*

---

## Architecture

```
                                ┌─────────────────────────┐
                                │  Raw data (Kaggle)      │
                                │  307,511 applications   │
                                │  122 features           │
                                └────────────┬────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────┐
                          │  Feature engineering pipeline       │
                          │  • Drop columns >60% missing        │
                          │  • Missingness indicator flags      │
                          │  • Domain ratios (8 features)       │
                          │  • EXT_SOURCE interactions (5)      │
                          │  • Anomaly correction               │
                          └────────────┬────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────────────────┐
                          │  SMOTE resampling                   │
                          │  8.07% → 16.7% positive class       │
                          └────────────┬────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────────────────┐
                          │  LightGBM × 5-fold stratified CV    │
                          │  → Out-of-fold predictions          │
                          └────────────┬────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────────────────┐
                          │  Logistic Regression meta-learner   │
                          │  → Final calibrated probability     │
                          └────────────┬────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────────────────┐
                          │  SHAP TreeExplainer                 │
                          │  → Per-borrower risk drivers        │
                          └────────────┬────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                ┌──────────────────┐    ┌──────────────────────┐
                │  FastAPI         │    │  Streamlit dashboard │
                │  /predict        │    │  • Business sim      │
                │  /health         │    │  • Borrower explain  │
                │  /features       │    │  • Model insights    │
                └──────────────────┘    └──────────────────────┘
                       Render                Streamlit Cloud
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas, numpy, pyarrow |
| Modelling | LightGBM, scikit-learn, imbalanced-learn |
| Explainability | SHAP |
| Experiment tracking | MLflow (SQLite backend) |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Visualisation | matplotlib, seaborn |
| Testing | pytest |
| Deployment | Streamlit Community Cloud, Render, Git LFS |

---

## Project Structure

```
credit-risk-intelligence/
├── api/
│   └── main.py                    # FastAPI prediction endpoint
├── app/
│   └── streamlit_app.py           # 3-page Streamlit dashboard
├── data/
│   ├── raw/                       # Kaggle dataset (gitignored)
│   └── processed/                 # Engineered features, SHAP scores
├── models/                        # Serialised model artefacts (Git LFS)
│   ├── lgbm_folds.pkl
│   ├── meta_learner.pkl
│   └── feature_names.pkl
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb          # CV + stacking + MLflow
│   └── 04_explainability.ipynb    # SHAP analysis
├── reports/
│   └── figures/                   # Generated plots
├── src/                           # Reusable Python modules
│   ├── features.py
│   ├── model.py
│   ├── explain.py
│   └── utils.py
├── tests/                         # pytest unit tests
│   ├── test_features.py
│   └── test_model.py
├── mlflow_tracking/               # MLflow experiment store
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

---

## Key Findings from EDA

**1. Severe class imbalance (8.07% default rate).** A naive classifier predicting "repaid" for everyone would score 91.93% accuracy and be commercially worthless. This is why every model decision in the project is benchmarked against PR-AUC (10× the random baseline) rather than accuracy.

**2. External credit scores carry the strongest signal.** `EXT_SOURCE_2` and `EXT_SOURCE_3` show clean distributional separation between defaulters and repayers. Their engineered mean (`EXT_SOURCE_MEAN`) becomes the single most predictive feature in the entire model with a mean |SHAP| value of 0.5263 — three times higher than the next feature.

**3. Missingness itself is informative.** 49 columns had over 40% missing values. Rather than dropping them, binary `*_MISSING` flags were created — `EXT_SOURCE_1_MISSING` alone ranks in the top 10 SHAP features, confirming that the *absence* of a credit bureau record is a meaningful risk signal in its own right.

**4. Income type matters more than income level.** Maternity-leave and unemployed applicants default at 40%+ rates — five times the dataset average. Income stability dominates absolute income amount in predictive power.

---

## Feature Engineering

28 new features were engineered from domain knowledge. The most impactful:

| Feature | Logic | Why it works |
|---|---|---|
| `CREDIT_TERM` | annuity / credit | Monthly burden vs loan size — **#1 by LightGBM gain** |
| `EXT_SOURCE_MEAN` | mean of 3 bureau scores | Aggregates external signals — **#1 by SHAP** |
| `ANNUITY_INCOME_RATIO` | annuity / income | Debt-to-income proxy |
| `EXT_SOURCE_DISAGREEMENT` | std / mean of EXT sources | Bureau disagreement encodes uncertainty |
| `EMPLOYMENT_STABILITY` | days_employed / days_birth | Fraction of life spent employed |
| `*_MISSING` flags | 1 if value is null | Missingness as signal (32 flags) |

**14 of the top 25 features by LightGBM gain are engineered features**, validating the domain-knowledge approach over raw-feature ingestion.

---

## Modeling Approach

### Why LightGBM?

Gradient boosted trees are the industry standard for tabular financial data. LightGBM specifically handles large datasets efficiently, supports `scale_pos_weight` natively for imbalance, and outperforms XGBoost on this dataset shape (300k rows, 150 features) in benchmarks.

### Why a Stacked Ensemble?

The Logistic Regression meta-learner trained on out-of-fold predictions adds calibration on top of the base model. While the metric gain was marginal here (ROC-AUC was identical), stacking is the standard production pattern and demonstrates MLOps awareness for interview discussion.

### Why PR-AUC Over Accuracy?

With an 8% default rate, ROC-AUC saturates and accuracy is meaningless. PR-AUC measures performance specifically on the minority class — the one we actually care about predicting. Random baseline PR-AUC on this dataset is the positive rate itself (≈0.08); achieving 0.799 represents a ~10× lift.

### Class Imbalance Handling

Two-pronged approach:
- `scale_pos_weight=11` in LightGBM (≈ majority/minority ratio)
- SMOTE oversampling raising positive rate from 8.07% to 16.7%

> ⚠️ **Honest disclosure:** SMOTE was applied to the full training set before cross-validation for simplicity. The rigorous approach applies SMOTE *inside each fold* to prevent synthetic samples from the validation set leaking into training. Current metrics may be modestly inflated as a result. This is documented as the first item in [What I'd Do With More Time](#what-id-do-with-more-time).

---

## Explainability

This system implements model explainability aligned with **SR 11-7 model risk guidance** and **GDPR Article 22** (right to explanation for automated decisions).

Every prediction returns:
- A default probability score (0.0 – 1.0)
- A risk label (LOW / MEDIUM / HIGH) and recommended decision
- **Top 3 risk drivers** with SHAP values and feature values
- **Top 3 protective factors** with SHAP values and feature values

### Example — High-risk borrower (92.1% default probability)

| Feature | Value | SHAP | Direction |
|---|---|---|---|
| EXT_SOURCE_MEAN | 0.149 | +1.091 | ↑ Increases risk |
| EXT_SOURCE_3 | 0.062 | +0.287 | ↑ Increases risk |
| EXT_SOURCE_MIN | 0.062 | +0.200 | ↑ Increases risk |

All three external credit bureaus produced consistently low scores — a strong, non-conflicting signal that justifies the deny decision.

---

## Business Impact Simulator

The Streamlit dashboard's first page lets stakeholders interactively explore the revenue–loss tradeoff at any approval threshold. Key inputs:

- Approval threshold (0.10 – 0.99)
- Average loan value
- Loss given default percentage

The simulator outputs in real time:
- Approval rate
- Expected loss (false approvals × loan value × LGD)
- Foregone revenue (false denials × loan value × interest)
- Net position (revenue from correctly approved loans minus expected loss)
- Threshold sweep charts showing the full revenue/loss curve

This frames every model decision as a **business decision** with explicit dollar consequences — the framing risk officers actually care about.

---

## Getting Started

### Prerequisites

- Python 3.12+
- Git + Git LFS (for downloading the model artefacts)
- A Kaggle account (for the dataset, if retraining)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-risk-intelligence.git
cd credit-risk-intelligence

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Pull model artefacts via Git LFS
git lfs pull
```

### Run the Services

```bash
# Start the FastAPI endpoint (runs on http://localhost:8000)
uvicorn api.main:app --reload --port 8000

# Start the Streamlit dashboard (runs on http://localhost:8501)
streamlit run app/streamlit_app.py

# Launch the MLflow UI for experiment tracking
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --port 5000
```

### Retrain From Scratch

```bash
# 1. Configure Kaggle credentials in ~/.kaggle/kaggle.json
# 2. Download the dataset
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw/

# 3. Run the notebooks in order
jupyter notebook notebooks/
```

---

## API Usage

### Health Check

```bash
curl https://credit-risk-intelligence-vspf.onrender.com/health
```

### Predict With Explanation

```bash
curl -X POST "https://credit-risk-intelligence-vspf.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "EXT_SOURCE_1": 0.65,
      "EXT_SOURCE_2": 0.72,
      "EXT_SOURCE_3": 0.68,
      "AMT_INCOME_TOTAL": 150000,
      "AMT_CREDIT": 300000,
      "AMT_ANNUITY": 15000,
      "DAYS_BIRTH": 14600
    },
    "explain": true
  }'
```

**Response:**

```json
{
  "default_probability": 0.0312,
  "risk_label": "LOW",
  "decision": "Approve",
  "confidence": "High",
  "top_risk_drivers": [
    {"feature": "AMT_ANNUITY", "value": 15000.0, "shap": 0.0642}
  ],
  "top_protectors": [
    {"feature": "EXT_SOURCE_MEAN", "value": 0.683, "shap": -0.8965},
    {"feature": "CREDIT_TERM", "value": 0.05, "shap": -0.8738}
  ]
}
```

Full interactive documentation available at the [Swagger UI](https://credit-risk-intelligence-vspf.onrender.com/docs).

---

## Testing

The repository includes pytest unit tests covering the feature engineering pipeline and metric computation.

```bash
pytest tests/ -v
```

Tests cover:
- High-missing column dropping logic
- Missingness flag generation and binary encoding
- Anomaly correction (`DAYS_EMPLOYED` 365243 placeholder)
- Engineered feature value ranges and constraints
- Metric computation correctness on perfect, random, and edge-case inputs

---

## What I'd Do With More Time

In rough order of priority — these are the next steps a production team would tackle:

1. **SMOTE inside CV folds.** Refactor `train_lgbm_cv` to apply resampling within each fold rather than to the full training set, eliminating synthetic data leakage and producing more honest metrics.
2. **Join bureau and previous application tables.** `bureau.csv` and `previous_application.csv` contain rich behavioural data that should push ROC-AUC further. Aggregate features (count, mean, max of past credits) typically lift performance by 2–4 points.
3. **Isotonic or Platt calibration.** The calibration plot shows the model is under-confident at high probabilities — a known SMOTE artefact. A post-hoc isotonic regression on a held-out set would correct this.
4. **Evidently AI drift monitoring.** Track feature distributions and prediction shifts over time to detect concept drift in production.
5. **Bayesian hyperparameter optimisation.** Replace the hand-tuned LightGBM config with Optuna or Hyperopt — likely worth 1–2 PR-AUC points.
6. **Containerise the API.** Wrap the FastAPI service in a Docker image for reproducible cloud deployment beyond Render's free tier.
7. **Maintain split requirements files.** `requirements-dev.txt` (full freeze for reproducibility) and `requirements.txt` (minimal for cloud) — this project learned that lesson the hard way during deployment.

---

## Dataset & References

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — Kaggle competition, 307,511 loan applications with 122 features. Used under the competition's terms.

**Key references:**
- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Ke et al. (2017) — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*
- Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*
- Federal Reserve SR 11-7 — *Guidance on Model Risk Management*

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Built by **[Your Name]** · [LinkedIn](https://linkedin.com/in/your-handle) · [GitHub](https://github.com/your-handle) · [Email](mailto:your.email@example.com)

If this project is useful to you, please ⭐ the repo on GitHub.
