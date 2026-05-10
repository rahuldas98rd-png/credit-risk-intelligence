"""MLflow Model Registry helpers + custom PyFunc wrapper.

This module centralizes all interaction with the MLflow Model Registry
so callers don't need to know about MlflowClient internals or model URIs.

Why a custom PyFunc:
    The production model is a stacked ensemble — 5 LightGBM fold models
    averaged, then their averaged probability passed through a logistic
    regression meta-learner for calibration. Standard mlflow.lightgbm.log_model
    handles a single model; we need to bundle the whole pipeline.

Backing store:
    The registry uses the same SQLite database as experiment tracking
    (mlflow/mlflow.db). For production deployment, swap this for a Postgres
    backend store and a remote tracking server.
"""
from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc

import mlflow
from src.utils import ROOT_DIR, get_logger

logger = get_logger("registry")

# ── Configuration ────────────────────────────────────────────────────────────

# Registered model name in the registry. By convention, dash-separated.
MODEL_NAME = "credit-risk-lgbm-stack"

# Tracking URI — same SQLite store used during training (see src/model.py).
TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow' / 'mlflow.db').as_posix()}"

# Where bootstrap reads the existing pickle files from.
MODELS_DIR = ROOT_DIR / "models"


def _setup_tracking() -> None:
    """Idempotent: point mlflow at our backing store before any operation."""
    mlflow.set_tracking_uri(TRACKING_URI)


# ── PyFunc wrapper for the stacked ensemble ──────────────────────────────────

class CreditRiskStackedModel(mlflow.pyfunc.PythonModel):
    """Stacked ensemble: averaged LGBM fold predictions → LR meta-learner.

    This class is what gets serialized into the registry. When the model
    is loaded, MLflow re-instantiates this class and calls load_context()
    with the artifact paths.

    Inference contract:
        - Input: pandas.DataFrame with columns matching feature_names.pkl
                 (or a dict of {feature_name: value} for single predictions)
        - Output: numpy.ndarray of shape (n_rows,) with default probabilities
    """

    def load_context(self, context):
        """Called once when the model is loaded from the registry.

        `context.artifacts` is a dict mapping the artifact names declared
        at log_model time to their on-disk paths.
        """
        import joblib

        self.lgbm_folds = joblib.load(context.artifacts["lgbm_folds"])
        self.meta_learner = joblib.load(context.artifacts["meta_learner"])
        self.feature_names = joblib.load(context.artifacts["feature_names"])

    def predict(self, context, model_input, params=None):
        """Stacked prediction: avg(fold_probs) → meta_learner."""
        import numpy as np
        import pandas as pd

        # Accept dict for single-row predictions (convenience for API callers).
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])

        # Enforce feature ordering — the LGBM folds were trained with a
        # specific column order, mismatched columns silently corrupt predictions.
        X = model_input[self.feature_names]

        # Stage 1: average the 5 LGBM fold-model predictions.
        fold_probs = np.mean(
            [m.predict_proba(X)[:, 1] for m in self.lgbm_folds],
            axis=0,
        )

        # Stage 2: pass averaged prob through the LR meta-learner for calibration.
        final_probs = self.meta_learner.predict_proba(
            fold_probs.reshape(-1, 1)
        )[:, 1]

        return final_probs


# ── Registry operations ──────────────────────────────────────────────────────

def bootstrap_register_existing(name: str = MODEL_NAME) -> int:
    """Register the EXISTING pickle files in models/ as a new registered version.

    Use this once to onboard your current trained model to the registry without
    retraining. For training a fresh model and registering, use the train script
    that calls src.model.run_training() first, then registers from the run_id.

    Returns:
        The registered version number (int).

    Raises:
        FileNotFoundError: if any of the expected pickle files is missing.
    """
    _setup_tracking()

    artifacts = {
        "lgbm_folds": str(MODELS_DIR / "lgbm_folds.pkl"),
        "meta_learner": str(MODELS_DIR / "meta_learner.pkl"),
        "feature_names": str(MODELS_DIR / "feature_names.pkl"),
    }
    for artifact_name, path in artifacts.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Expected artifact {artifact_name!r} at {path}. "
                "Did you `git lfs pull`?"
            )

    # Use a dedicated experiment for registry bootstrap runs so they don't
    # clutter the main training experiment.
    experiment_name = "credit-risk-intelligence"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="bootstrap_existing_v1") as run:
        mlflow.log_param("source", "bootstrap_from_pickle_files")
        mlflow.log_param("model_name", name)
        mlflow.log_param("note", "Onboarded existing trained model to registry")
        mlflow.log_param("ensemble", "5x LightGBM folds + LR meta-learner")

        # Log key headline metrics from the training run for registry context.
        # These are the numbers we'd want to see in the registry UI for any
        # version, so reviewers know what they're looking at.
        mlflow.log_metric("oof_roc_auc", 0.8976)
        mlflow.log_metric("oof_pr_auc", 0.7990)
        mlflow.log_metric("best_f1", 0.7271)
        mlflow.log_metric("best_threshold", 0.8456)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CreditRiskStackedModel(),
            artifacts=artifacts,
            pip_requirements=[
                "lightgbm==4.3.0",
                "scikit-learn==1.5.0",
                "pandas==2.2.2",
                "numpy==1.26.4",
                "joblib==1.5.3",
            ],
        )

        run_id = run.info.run_id
        logger.info(f"Logged pyfunc model in run {run_id}")

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, name)
    logger.info(f"Registered {name} v{result.version} from {model_uri}")
    return int(result.version)


def transition_stage(
    name: str,
    version: int,
    stage: str,
    archive_existing: bool = True,
) -> None:
    """Move a model version to None / Staging / Production / Archived."""
    valid_stages = {"None", "Staging", "Production", "Archived"}
    if stage not in valid_stages:
        raise ValueError(
            f"Invalid stage {stage!r}. Must be one of {sorted(valid_stages)}"
        )

    _setup_tracking()
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info(f"{name} v{version} → {stage}")


def get_latest_version(name: str = MODEL_NAME, stage: str = "Production") -> int | None:
    """Return the latest version number in `stage`, or None if none exists."""
    _setup_tracking()
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(name, stages=[stage])
    if not versions:
        return None
    return int(versions[0].version)


def load_pyfunc_model(name: str = MODEL_NAME, stage: str = "Production"):
    """Load a registered pyfunc model from the given stage.

    Returns the loaded model — call .predict(dataframe) on it for inference.
    """
    _setup_tracking()
    model_uri = f"models:/{name}/{stage}"
    logger.info(f"Loading model from registry: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def download_registry_artifacts(
    name: str = MODEL_NAME,
    stage: str = "Production",
    dst: Path | None = None,
) -> Path:
    """Download the raw artifacts (pickle files) of a registered model.

    Useful when the API wants to use its own loading + SHAP logic rather
    than calling .predict() through the pyfunc wrapper. Returns the local
    directory path containing lgbm_folds.pkl, meta_learner.pkl, etc.
    """
    _setup_tracking()
    client = mlflow.MlflowClient()

    versions = client.get_latest_versions(name, stages=[stage])
    if not versions:
        raise RuntimeError(
            f"No version of {name} found in stage {stage!r}. "
            "Run scripts/register_model.py --bootstrap first."
        )
    version = versions[0]

    dst = dst or (ROOT_DIR / "models" / "registry_cache" / f"{name}_v{version.version}")
    dst.mkdir(parents=True, exist_ok=True)

    # The pyfunc model logged its artifacts under "model/artifacts/<name>".
    download_path = client.download_artifacts(
        run_id=version.run_id,
        path="model/artifacts",
        dst_path=str(dst),
    )
    logger.info(f"Downloaded {name} v{version.version} artifacts to {download_path}")
    return Path(download_path)
