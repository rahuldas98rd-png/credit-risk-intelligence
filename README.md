# Credit Risk Intelligence Engine

> An end-to-end machine learning system that predicts loan default probability, explains every prediction with SHAP values, and lets stakeholders simulate the business impact of approval thresholds in real time.

[![CI](https://github.com/rahuldas98rd-png/credit-risk-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/rahuldas98rd-png/credit-risk-intelligence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-2E7D32)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45.1-9C27B0)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.17.2-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)](Dockerfile)
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
- [Engineering Decisions](#engineering-decisions)
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
4. **Production infrastructure** — multi-stage Docker build, GitHub Actions CI with lint/test/Docker-smoke-test pipeline, `pyproject.toml`-based packaging with `uv` for deterministic installs.

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
                │  (Dockerized)    │    │  • Business sim      │
                │  /predict        │    │  • Borrower explain  │
                │  /health         │    │  • Model insights    │
                │  /features       │    │                      │
                └──────────────────┘    └──────────────────────┘
                  Render / Docker         Streamlit Cloud
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
| Packaging | `pyproject.toml` (PEP 621), `uv` for resolution + lockfile |
| Testing | pytest |
| Linting | ruff |
| Containerization | Docker (multi-stage build), docker-compose |
| CI/CD | GitHub Actions (lint → test → Docker smoke test) |
| Deployment | Streamlit Community Cloud, Render, Git LFS |

---

## Project Structure

```
credit-risk-intelligence/
├── api/
│   ├── __init__.py
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
│   ├── model.py                   # mlflow imported lazily inside training fns
│   ├── explain.py
│   ├── simulate.py
│   └── utils.py
├── tests/                         # pytest unit tests
│   ├── test_features.py
│   └── test_model.py
├── mlflow/                        # MLflow experiment store (SQLite)
├── .github/
│   └── workflows/
│       └── ci.yml                 # Lint, test, Docker build + smoke test
├── Dockerfile                     # Multi-stage build, non-root runtime
├── docker-compose.yml
├── .dockerignore
├── pyproject.toml                 # PEP 621 metadata + tool config
├── uv.lock                        # Deterministic dependency lockfile
├── requirements.txt               # Slim production deps (Streamlit Cloud)
├── render.yaml                    # Render deployment config
├── Makefile
└── README.md
```

---

## Engineering Decisions

These are the kinds of choices a senior reviewer will probe in an interview. Each one was deliberate.

### Why `pyproject.toml` over `setup.py`

Single source of truth (PEP 621) for project metadata, runtime dependencies, optional groups (`test`, `training`, `dev`), and tool configuration (ruff, pytest). One file replaces `setup.py` + standalone `ruff.toml` + scattered `[tool.*]` configs. Modern Python packaging standard since 2021.

### Why `uv` over `pip`

`uv sync` resolves and installs the dependency tree in **~5 seconds** vs `pip`'s ~45 seconds for the same operation. The `uv.lock` file gives bit-for-bit reproducible installs across machines and CI. The CI workflow time dropped from ~6 minutes to under 2 minutes (warm cache) after migrating, with no functional changes.

### Why lazy mlflow imports in `src/model.py`

Originally `model.py` imported `mlflow.lightgbm` and `mlflow.sklearn` at module level. This forced mlflow as a dependency for *every* consumer of the file — including `tests/test_model.py` (which only tests pure-numpy metric functions) and the deployed FastAPI service (which only does inference). Moving the imports inside `run_training()` decouples the training-only dependency from the inference and testing paths. Result: tests run with mlflow uninstalled, the API container ships without 100MB of experiment-tracking machinery.

### Why a multi-stage Docker build

The builder stage installs `build-essential` (~250MB) to compile LightGBM/scipy. The runtime stage only needs `libgomp1` for LightGBM's OpenMP runtime. Splitting the stages drops the final image from ~850MB to **~360MB content size**. The runtime image runs as a non-root `app` user with explicit `HEALTHCHECK` directive — both standard production hygiene that container orchestrators (compose, ECS, K8s) rely on for traffic routing.

### Why two-job CI with `needs:` ordering

The `lint-and-test` job runs in ~30 seconds. The `docker-smoke-test` job runs in ~3 minutes. Having lint+test gate the Docker build means a missed import or unused variable fails CI in seconds — no waiting on a Docker build that won't matter. GHA cache on the Docker layer cuts subsequent runs to under 90 seconds.

### Why a smoke test that hits `/predict`, not just `/health`

A `/health` smoke test only proves the server started. Hitting `/predict` with a real payload proves: (1) Pydantic validation works, (2) the model files unpickled correctly, (3) SHAP runs end-to-end, (4) the response shape matches the schema. End-to-end validation in 30 seconds. It also caught a Git LFS misconfiguration on first CI run — model files were checked in as LFS pointers and never resolved, which `/health` would have happily returned 200 for. `/predict` failed loudly on the unpickle, surfacing the bug immediately.

### Why a conservative ruff ruleset

Ruff is configured with `select = ["E", "F", "I"]` — pycodestyle errors, pyflakes (unused imports, undefined names), and isort. Style warnings (line length, naming conventions) are intentionally **off**. Starting strict on a previously-unlinted codebase produces a wall of red and forces noise edits; starting with rules that catch real bugs gets the codebase clean and lets style tightening happen incrementally.

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
- [`uv`](https://github.com/astral-sh/uv) (recommended) or pip
- Git + Git LFS (for downloading the model artefacts)
- A Kaggle account (for the dataset, if retraining)

### Local Setup with `uv` (recommended)

```bash
# Clone the repository
git clone https://github.com/rahuldas98rd-png/credit-risk-intelligence.git
cd credit-risk-intelligence

# Pull model artefacts via Git LFS
git lfs pull

# Install dependencies (creates .venv automatically)
uv sync --extra test

# Run tests to verify
uv run pytest tests/
```

### Local Setup with pip (alternative)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
git lfs pull
```

### Run the Services

```bash
# Option 1: Run with uv
uv run uvicorn api.main:app --reload --port 8000
uv run streamlit run app/streamlit_app.py

# Option 2: Run the API in Docker (preferred for production parity)
docker compose up

# MLflow tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```

### Retrain From Scratch

```bash
# 1. Configure Kaggle credentials in ~/.kaggle/kaggle.json
# 2. Download the dataset
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw/

# 3. Install training extras (includes mlflow)
uv sync --extra training

# 4. Run the notebooks in order
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

The repository includes pytest unit tests covering the feature engineering pipeline and metric computation. **CI runs them on every push and PR**, plus a Docker build and container smoke test that hits `/predict` with a real payload.

```bash
uv run pytest tests/
```

Tests cover:
- High-missing column dropping logic
- Missingness flag generation and binary encoding
- Anomaly correction (`DAYS_EMPLOYED` 365243 placeholder)
- Engineered feature value ranges and constraints
- Metric computation correctness on perfect, random, and edge-case inputs

Tests run **without** mlflow installed (mlflow is a training-only dependency). The CI lint + test job completes in ~20 seconds on a warm cache.

---

## What I'd Do With More Time

In rough order of priority — these are the next steps a production team would tackle. Items already shipped via Phase 2 work are noted ✅.

### Modelling

1. **SMOTE inside CV folds.** Refactor `train_lgbm_cv` to apply resampling within each fold rather than to the full training set, eliminating synthetic data leakage and producing more honest metrics.
2. **Join bureau and previous application tables.** `bureau.csv` and `previous_application.csv` contain rich behavioural data that should push ROC-AUC further. Aggregate features (count, mean, max of past credits) typically lift performance by 2–4 points.
3. **Isotonic or Platt calibration.** The calibration plot shows the model is under-confident at high probabilities — a known SMOTE artefact. A post-hoc isotonic regression on a held-out set would correct this.
4. **Bayesian hyperparameter optimisation.** Replace the hand-tuned LightGBM config with Optuna or Hyperopt — likely worth 1–2 PR-AUC points.

## MLflow Workflow

The trained model is registered in the MLflow Model Registry with explicit
stage promotion. Production-bound models move through:

None  →  Staging  →  Production
(registered)   (validated)    (live)

### Register the existing model

```bash
# One-time bootstrap: wraps the trained pickles in a PyFunc model,
# logs to MLflow, registers as v1, transitions to Staging
uv run python scripts/register_model.py --bootstrap

# Promote Staging → Production after manual review
uv run python scripts/register_model.py --promote

# Inspect the registry
uv run python scripts/register_model.py --list
```

### View the registry UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
# Open http://localhost:5000/#/models
```

### Serve the API from the registry

By default the API loads models from local pickle files (production-friendly,
no MLflow dependency). Set `MODEL_SOURCE` to load from the registry:

```bash
# Linux / macOS
export MODEL_SOURCE=registry:Production

# Windows PowerShell
$env:MODEL_SOURCE = "registry:Production"

uv run uvicorn api.main:app --port 8000
```

### Why the model is a PyFunc

The production model is a stacked ensemble: **5 LightGBM fold models averaged,
then their averaged probability passed through a logistic regression
meta-learner for calibration**. Standard `mlflow.lightgbm.log_model` handles
a single model; PyFunc bundles the whole pipeline (5 LGBM folds + LR + feature
ordering) into one registered artifact with custom prediction logic.
See `src/registry.py::CreditRiskStackedModel`.

### MLOps & Production

5. ✅ **Containerise the API** — multi-stage Dockerfile, runs as non-root, ~360MB content size.
6. ✅ **CI/CD pipeline** — GitHub Actions: lint → test → Docker build → container smoke test.
7. ✅ **Modern packaging** — `pyproject.toml`, `uv.lock`, optional dependency groups for `test` / `training` / `dev`.
8. ⏳ **MLflow Model Registry** — promote experiment-tracked models to Staging / Production stages; serve from registry rather than committed pickle files.
9. ⏳ **Prometheus metrics** — `/metrics` endpoint exposing latency histograms, RPS, prediction-class distribution.
10. ⏳ **Grafana dashboard** — operational view: 4 panels (RPS, p95 latency, prediction class balance, drift score).
11. ⏳ **Evidently AI drift monitoring** — daily cron generating drift reports against the training distribution.
12. ⏳ **Cloud Run / ECS Fargate deployment** — alternative to Render's free tier, with Terraform IaC.
13. ⏳ **Drift-triggered retraining loop** — drift detected → GitHub Actions cron triggers retraining → new model registered in MLflow → manual approval → deploy.

---

## Drift Detection

The model registry tells you *which* version is deployed. The Grafana dashboard tells you *whether the service is healthy*. The drift report tells you the third question: **is the model still seeing the same world it was trained on?**

### How it works
data/raw/application_train.csv (10K sample)
│
▼
data/processed/drift_baseline.parquet  ← reference distribution
│
▼
┌──────────────────────────────┐
current ──► │  Evidently DataDriftPreset   │ ──► reports/drift/<date>.html
│  + DataQualityPreset         │     reports/drift/latest_summary.json
└──────────────────────────────┘

### Capture a baseline (run once after each model retraining)

```bash
uv run python scripts/capture_baseline.py
# Writes data/processed/drift_baseline.parquet (~1 MB, committed to repo)
```

### Generate a drift report

```bash
# Synthetic comparison: no drift expected
uv run python scripts/run_drift_report.py --drift-factor 0.0

# Synthetic moderate drift (typical demo scenario)
uv run python scripts/run_drift_report.py --drift-factor 0.3

# Real production data (when you have it captured from /predict logs)
uv run python scripts/run_drift_report.py --current data/processed/last_24h_inputs.parquet
```

Output:

- `reports/drift/<timestamp>_drift_report.html` — full interactive Evidently report (per-feature drift charts, PSI scores, KS tests, distribution comparisons)
- `reports/drift/<timestamp>_summary.json` — machine-readable summary
- `reports/drift/latest.html` — symlinked to most recent report

### CI integration

`.github/workflows/drift-check.yml` runs the drift detection on demand or on a schedule. Reports are uploaded as workflow artifacts (downloadable from the Actions UI for 30 days). The cron schedule is commented out by default in this portfolio repo; production deployments would enable it.

### Why synthetic vs. production data

Real production drift detection requires logging every `/predict` input to durable storage (S3 parquet, Postgres, etc.) and reading it back daily. Render's free tier has ephemeral storage so the production-data pipeline isn't deployed here. Instead, the script can perturb the baseline with a configurable `--drift-factor` to simulate scenarios (no drift / moderate drift / severe drift) — useful for testing alerting thresholds and demonstrating the workflow. In a deployed system, swap the synthetic generator for `--current path/to/recent_logs.parquet`.

### Roadmap (not yet implemented)

- Push drift score to a Prometheus gauge so it appears as a 5th Grafana panel
- Authenticated `/admin/drift` API endpoint for scripts to update the gauge
- Alert rules for `drift_score > 0.3` (PagerDuty / Slack webhook)
- Drift-triggered retraining: drift detected → GitHub Actions cron → retrain → register new MLflow version → manual approval → promote to Production

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

Built by **Rahul Das** · [GitHub](https://github.com/rahuldas98rd-png)

If this project is useful to you, please ⭐ the repo on GitHub.
