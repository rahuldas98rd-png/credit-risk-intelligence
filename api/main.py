from contextlib import asynccontextmanager

import joblib
import os
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import time
from prometheus_fastapi_instrumentator import Instrumentator

from src.metrics import (
    prediction_class_total,
    prediction_probability,
    shap_computation_seconds,
    model_info,
    detect_model_version,
)

from src.utils import ROOT_DIR, get_logger

logger = get_logger("api")
MODELS_DIR = ROOT_DIR / "models"

# Module-level state initialized in the lifespan
fold_models = None
meta_model = None
feature_names: list[str] = []
explainer = None


def _load_from_files(models_dir):
    """Default: load joblib pickles directly from disk. Used in production
    Docker container — no MLflow dependency required."""
    return {
        "lgbm_folds": joblib.load(models_dir / "lgbm_folds.pkl"),
        "meta_learner": joblib.load(models_dir / "meta_learner.pkl"),
        "feature_names": joblib.load(models_dir / "feature_names.pkl"),
    }


def _load_from_registry(stage: str):
    """Load the registered model's artifacts from the MLflow registry.
    
    This is opt-in via MODEL_SOURCE=registry:<stage>. mlflow is imported
    lazily so the default file-based path doesn't require it.
    """
    try:
        from src.registry import download_registry_artifacts  # lazy import
    except ModuleNotFoundError as e:
        if "mlflow" in str(e).lower():
            raise RuntimeError(
                "MODEL_SOURCE=registry requires mlflow, but it isn't installed. "
                "Run `uv sync --extra training` to install it, "
                "or set MODEL_SOURCE=file to use local pickle files instead."
            ) from e
        raise
    
    artifacts_dir = download_registry_artifacts(stage=stage)
    return {
        "lgbm_folds":    joblib.load(artifacts_dir / "lgbm_folds.pkl"),
        "meta_learner":  joblib.load(artifacts_dir / "meta_learner.pkl"),
        "feature_names": joblib.load(artifacts_dir / "feature_names.pkl"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup; release on shutdown."""
    global fold_models, meta_model, feature_names, explainer
    MODEL_SOURCE = os.getenv("MODEL_SOURCE", "file")
    logger.info(f"Loading models with MODEL_SOURCE={MODEL_SOURCE}")

    if MODEL_SOURCE.startswith("registry"):
        # Format: "registry" or "registry:Staging" or "registry:Production"
        stage = MODEL_SOURCE.split(":", 1)[1] if ":" in MODEL_SOURCE else "Production"
        loaded = _load_from_registry(stage)
    else:
        loaded = _load_from_files(MODELS_DIR)

    fold_models   = loaded["lgbm_folds"]
    meta_model    = loaded["meta_learner"]
    feature_names = loaded["feature_names"]

    explainer     = shap.TreeExplainer(fold_models[0])
    logger.info(f"Models loaded — {len(feature_names)} features")

    # Set the model_info gauge so Prometheus can correlate metric shifts with deploys
    model_info.labels(
        model_name="credit-risk-lgbm-stack",
        version=detect_model_version(MODEL_SOURCE),
        source=MODEL_SOURCE,
    ).set(1)
    logger.info(f"Metrics initialized for {detect_model_version(MODEL_SOURCE)} ({MODEL_SOURCE})")

    yield
    # Shutdown hook (no cleanup needed today; could close DB pools etc. later)
    logger.info("API shutting down")


app = FastAPI(
    title="Credit Risk Intelligence API",
    description="Predict loan default probability with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-expose HTTP metrics at /metrics (request rate, latency, status codes)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: dict = Field(..., description="Feature name → value mapping")
    explain:  bool = Field(default=True, description="Include SHAP explanation")

class RiskDriver(BaseModel):
    feature: str
    value:   float
    shap:    float

class PredictionResponse(BaseModel):
    default_probability: float
    risk_label:          str
    decision:            str
    confidence:          str
    top_risk_drivers:    list[RiskDriver]
    top_protectors:      list[RiskDriver]


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.0.0", "features": len(feature_names)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Build feature vector — fill missing with 0
        row = pd.DataFrame([{f: request.features.get(f, 0.0) for f in feature_names}])

        # Ensemble prediction: average fold probabilities → meta-learner
        fold_probs = np.mean([
            m.predict_proba(row)[0, 1] for m in fold_models
        ])
        final_prob = float(meta_model.predict_proba([[fold_probs]])[0, 1])

        THRESHOLD = 0.8456
        if final_prob >= THRESHOLD:
            risk_label = "HIGH"
            decision   = "Deny"
            confidence = "High" if final_prob > 0.90 else "Medium"
        elif final_prob >= 0.5:
            risk_label = "MEDIUM"
            decision   = "Manual Review"
            confidence = "Medium"
        else:
            risk_label = "LOW"
            decision   = "Approve"
            confidence = "High" if final_prob < 0.10 else "Medium"
        
        # Record application-level metrics
        prediction_class_total.labels(risk_label=risk_label).inc()
        prediction_probability.observe(final_prob)

        # SHAP explanation
        risk_drivers, protectors = [], []
        if request.explain:
            shap_start = time.perf_counter()
            sv = explainer.shap_values(row)
            shap_computation_seconds.observe(time.perf_counter() - shap_start)
            if isinstance(sv, list):
                sv = sv[1]
            sv_flat = sv[0]
            contrib = sorted(
                zip(feature_names, row.values[0], sv_flat),
                key=lambda x: abs(x[2]), reverse=True
            )
            risk_drivers = [
                RiskDriver(feature=f, value=round(float(v), 4), shap=round(float(s), 4))
                for f, v, s in contrib if s > 0
            ][:3]
            protectors = [
                RiskDriver(feature=f, value=round(float(v), 4), shap=round(float(s), 4))
                for f, v, s in contrib if s < 0
            ][:3]

        logger.info(f"Prediction: prob={final_prob:.4f} label={risk_label}")
        return PredictionResponse(
            default_probability=round(final_prob, 4),
            risk_label=risk_label,
            decision=decision,
            confidence=confidence,
            top_risk_drivers=risk_drivers,
            top_protectors=protectors,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
def list_features():
    return {"feature_count": len(feature_names), "features": feature_names}