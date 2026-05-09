import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import compute_metrics

# ── compute_metrics ───────────────────────────────────────────────────────────

def test_perfect_model_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"]  == 1.0

def test_random_model_roc_near_half():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.random.uniform(0, 1, 1000)
    metrics = compute_metrics(y_true, y_prob)
    assert 0.40 < metrics["roc_auc"] < 0.60

def test_metrics_keys_present():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    metrics = compute_metrics(y_true, y_prob)
    for key in ["roc_auc", "pr_auc", "brier", "best_f1", "best_threshold"]:
        assert key in metrics

def test_threshold_in_valid_range():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = compute_metrics(y_true, y_prob)
    assert 0.0 <= metrics["best_threshold"] <= 1.0

def test_brier_score_range():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.1, 0.9])
    metrics = compute_metrics(y_true, y_prob)
    assert 0.0 <= metrics["brier"] <= 1.0