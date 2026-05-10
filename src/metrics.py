"""Custom Prometheus metrics for the credit-risk API.

Two layers of observability ship with this service:

  1. HTTP-level (auto-captured by prometheus-fastapi-instrumentator):
     - http_requests_total{method, status, handler}
     - http_request_duration_seconds_bucket{...}
     - http_request_size_bytes / http_response_size_bytes

  2. Application-level (defined here):
     - credit_risk_prediction_class_total{risk_label}
     - credit_risk_prediction_probability_bucket
     - credit_risk_shap_computation_seconds_bucket
     - credit_risk_model_info{model_name, version, source}

These are what Grafana panels chart in Session 5, and what Evidently
compares against the training distribution for drift detection. Naming
follows Prometheus conventions: `_total` suffix for counters, units in
the histogram names.
"""
from prometheus_client import Counter, Gauge, Histogram

# ── Counter: predictions by risk class ───────────────────────────────────────
# Lets us monitor approval-rate drift: if the LOW/MEDIUM/HIGH ratio shifts
# significantly, either input distribution moved (data drift) or the model
# is mis-calibrated.
prediction_class_total = Counter(
    "credit_risk_prediction_class_total",
    "Total predictions issued, labeled by risk class",
    ["risk_label"],  # LOW, MEDIUM, HIGH
)

# ── Histogram: default probability distribution ──────────────────────────────
# Buckets concentrated near the THRESHOLD=0.8456 decision boundary so we
# can see how concentrated predictions are near the deny line. Predictions
# near 0.0 or 1.0 are "confident"; predictions near the threshold are the
# ones a human reviewer would want to look at.
prediction_probability = Histogram(
    "credit_risk_prediction_probability",
    "Distribution of default probability scores returned by /predict",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

# ── Histogram: SHAP computation latency ──────────────────────────────────────
# SHAP TreeExplainer is the dominant cost in /predict — LightGBM inference
# is sub-millisecond, but SHAP value computation is ~50–100ms per request.
# Tracking it separately from total request latency lets us isolate model
# explainability cost from network/serialization cost.
shap_computation_seconds = Histogram(
    "credit_risk_shap_computation_seconds",
    "SHAP TreeExplainer compute latency per prediction (seconds)",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Gauge: which model is currently loaded ───────────────────────────────────
# Set to 1 at startup with labels identifying name/version/source. When we
# do model rollouts in production, this is what tells Prometheus "the
# version label changed at timestamp T," so we can correlate metric shifts
# with deployments.
model_info = Gauge(
    "credit_risk_model_info",
    "Currently loaded model: 1 if active, 0 otherwise",
    ["model_name", "version", "source"],
)


def detect_model_version(model_source: str) -> str:
    """Resolve a stable version string for the model_info gauge.

    Returns:
        - "v{N}_{Stage}" when serving from the registry (e.g. "v1_Production")
        - "file_local" when serving from local pickle files
        - "registry_unknown" if registry lookup fails (fallback)
    """
    if not model_source.startswith("registry"):
        return "file_local"

    try:
        from src.registry import MODEL_NAME, get_latest_version

        stage = (
            model_source.split(":", 1)[1] if ":" in model_source else "Production"
        )
        version = get_latest_version(MODEL_NAME, stage=stage)
        return f"v{version}_{stage}" if version is not None else "registry_unknown"
    except Exception:
        return "registry_unknown"
