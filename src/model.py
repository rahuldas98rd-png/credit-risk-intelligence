import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from src.utils import get_logger, DATA_PROC, ROOT_DIR

logger = get_logger("model")
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx  = np.argmax(f1_scores)
    return {
        "roc_auc":    round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":     round(average_precision_score(y_true, y_prob), 4),
        "brier":      round(brier_score_loss(y_true, y_prob), 4),
        "best_f1":    round(f1_scores[best_idx], 4),
        "best_threshold": round(float(thresholds[best_idx]), 4),
    }


# ── LightGBM base learner ──────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":       "binary",
    "metric":          "average_precision",
    "boosting_type":   "gbdt",
    "num_leaves":      63,
    "learning_rate":   0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "min_child_samples": 100,
    "scale_pos_weight": 11,   # ~91.93/8.07 — handles class imbalance
    "n_estimators":    1000,
    "random_state":    42,
    "n_jobs":          -1,
    "verbose":         -1,
}


def train_lgbm_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> tuple:
    """
    Stratified K-Fold cross-validation for LightGBM.
    Returns out-of-fold predictions and trained fold models.
    """
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1)
            ]
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_prob
        metrics = compute_metrics(y_val, val_prob)
        fold_metrics.append(metrics)
        fold_models.append(model)

        logger.info(
            f"Fold {fold} | ROC-AUC: {metrics['roc_auc']} | "
            f"PR-AUC: {metrics['pr_auc']} | Best F1: {metrics['best_f1']}"
        )

    oof_metrics = compute_metrics(y, oof_preds)
    logger.info(f"OOF | ROC-AUC: {oof_metrics['roc_auc']} | PR-AUC: {oof_metrics['pr_auc']}")
    return oof_preds, fold_models, oof_metrics, fold_metrics


# ── Meta-learner (stacking) ────────────────────────────────────────────────────

def train_meta_learner(oof_preds: np.ndarray, y: pd.Series) -> LogisticRegression:
    """
    Logistic Regression trained on LightGBM OOF predictions.
    Simple but effective — captures any residual calibration.
    """
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(oof_preds.reshape(-1, 1), y)
    meta_prob = meta.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
    metrics   = compute_metrics(y, meta_prob)
    logger.info(f"Meta-learner | ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']}")
    return meta, metrics


# ── Full training run with MLflow ──────────────────────────────────────────────

def run_training(X: pd.DataFrame, y: pd.Series):
    db_path = ROOT_DIR / "mlflow" / "mlflow.db"
    db_path.parent.mkdir(exist_ok=True)
    uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI: {uri}")

    # Create or fetch experiment
    experiment = mlflow.get_experiment_by_name("credit-risk-intelligence")
    if experiment is None:
        experiment_id = mlflow.create_experiment("credit-risk-intelligence")
    else:
        experiment_id = experiment.experiment_id
    logger.info(f"Experiment ID: {experiment_id}")

    with mlflow.start_run(
        run_name="lgbm_stacked_v1",
        experiment_id=experiment_id
    ) as run:
        logger.info(f"Run started: {run.info.run_id}")

        mlflow.log_params({k: str(v) for k, v in LGBM_PARAMS.items()})
        mlflow.log_param("n_cv_folds", 5)
        mlflow.log_param("meta_learner", "LogisticRegression")
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_train_rows", len(y))
        mlflow.log_param("positive_rate", round(float(y.mean()), 4))

        logger.info("Training LightGBM with 5-fold CV...")
        oof_preds, fold_models, oof_metrics, fold_metrics = train_lgbm_cv(X, y)

        logger.info("Training meta-learner on OOF predictions...")
        meta_model, meta_metrics = train_meta_learner(oof_preds, y)

        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})
        mlflow.log_metrics({f"meta_{k}": v for k, v in meta_metrics.items()})

        for i, fm in enumerate(fold_metrics, 1):
            mlflow.log_metric("fold_pr_auc", fm["pr_auc"], step=i)

        best_fold_idx = np.argmax([m["pr_auc"] for m in fold_metrics])
        best_model    = fold_models[best_fold_idx]
        fi_df = pd.DataFrame({
            "feature":    X.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)

        fi_path = DATA_PROC / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(fold_models, MODELS_DIR / "lgbm_folds.pkl")
        joblib.dump(meta_model,  MODELS_DIR / "meta_learner.pkl")
        joblib.dump(list(X.columns), MODELS_DIR / "feature_names.pkl")

        logger.info(f"Run complete: {run.info.run_id}")
        logger.info(f"OOF ROC-AUC: {oof_metrics['roc_auc']} | PR-AUC: {oof_metrics['pr_auc']}")

    return fold_models, meta_model, oof_preds, oof_metrics