import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from src.utils import ROOT_DIR, get_logger

logger = get_logger("api")
MODELS_DIR = ROOT_DIR / "models"

app = FastAPI(
    title="Credit Risk Intelligence API",
    description="Predict loan default probability with SHAP explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ─────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global fold_models, meta_model, feature_names, explainer
    fold_models   = joblib.load(MODELS_DIR / "lgbm_folds.pkl")
    meta_model    = joblib.load(MODELS_DIR / "meta_learner.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    explainer     = shap.TreeExplainer(fold_models[0])
    logger.info(f"Models loaded — {len(feature_names)} features")


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

        # SHAP explanation
        risk_drivers, protectors = [], []
        if request.explain:
            sv = explainer.shap_values(row)
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