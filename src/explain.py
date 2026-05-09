
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import REPORTS, ROOT_DIR, get_logger

logger = get_logger("explain")
MODELS_DIR = ROOT_DIR / "models"


def load_artifacts() -> tuple:
    fold_models    = joblib.load(MODELS_DIR / "lgbm_folds.pkl")
    feature_names  = joblib.load(MODELS_DIR / "feature_names.pkl")
    best_model     = fold_models[0]   # fold 1 — use consistently
    return best_model, feature_names


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 2000) -> tuple:
    """
    Compute SHAP values on a stratified sample for speed.
    Returns shap_values array and the sampled X.
    """
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42).reset_index(drop=True)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary returns list [neg_class, pos_class] — take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(f"SHAP values computed — shape: {shap_values.shape}")
    return shap_values, X_sample, explainer


def plot_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, top_n: int = 20):
    """Beeswarm summary plot — global feature importance with direction."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=top_n,
        show=False,
        plot_size=None
    )
    plt.title("SHAP Summary — Top 20 Features\n(Red = increases default risk, Blue = decreases)",
              fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "09_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Saved: 09_shap_summary.png")


def plot_dependence(shap_values: np.ndarray, X_sample: pd.DataFrame,
                    feature: str, interaction_feature: str = "auto"):
    """Dependence plot — how one feature's SHAP value changes across its range."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature, shap_values, X_sample,
        interaction_index=interaction_feature,
        ax=ax, show=False
    )
    ax.set_title(f"SHAP Dependence — {feature}", fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    plt.tight_layout()
    fname = f"10_shap_dependence_{feature.lower()}.png"
    plt.savefig(REPORTS / fname, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Saved: {fname}")


def explain_single_borrower(
    model, explainer, X_sample: pd.DataFrame,
    idx: int, threshold: float = 0.8456
) -> dict:
    """
    Generate human-readable risk explanation for one borrower.
    Returns top 3 risk drivers and top 3 protective factors.
    """
    row        = X_sample.iloc[[idx]]
    prob       = model.predict_proba(row)[0, 1]
    decision   = "HIGH RISK — Deny" if prob >= threshold else "LOW RISK — Approve"

    sv         = explainer.shap_values(row)
    if isinstance(sv, list):
        sv = sv[1]
    sv_flat    = sv[0]

    # Build feature contribution table
    contrib_df = pd.DataFrame({
        "feature": X_sample.columns,
        "value":   row.values[0],
        "shap":    sv_flat
    }).sort_values("shap", key=abs, ascending=False)

    risk_drivers  = contrib_df[contrib_df["shap"] > 0].head(3)
    protectors    = contrib_df[contrib_df["shap"] < 0].head(3)

    explanation = {
        "borrower_idx":   idx,
        "default_prob":   round(float(prob), 4),
        "decision":       decision,
        "risk_drivers":   risk_drivers[["feature","value","shap"]].to_dict("records"),
        "protectors":     protectors[["feature","value","shap"]].to_dict("records"),
    }
    return explanation


def plot_borrower_explanation(explanation: dict):
    """Horizontal waterfall-style bar chart for a single borrower."""
    drivers   = explanation["risk_drivers"]
    protectors = explanation["protectors"]

    features  = [d["feature"] for d in protectors[::-1]] + \
                [d["feature"] for d in drivers]
    shap_vals = [d["shap"]    for d in protectors[::-1]] + \
                [d["shap"]    for d in drivers]
    colors    = ["#4C9BE8" if v < 0 else "#E8593C" for v in shap_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(features, shap_vals, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, shap_vals):
        x_pos = bar.get_width() + (0.001 if val >= 0 else -0.001)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=10)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (impact on default probability)")
    ax.set_title(
        f"Borrower #{explanation['borrower_idx']} — "
        f"Default prob: {explanation['default_prob']:.1%} — {explanation['decision']}",
        fontsize=11, fontweight="bold"
    )

    red_patch  = mpatches.Patch(color="#E8593C", label="Increases risk")
    blue_patch = mpatches.Patch(color="#4C9BE8", label="Decreases risk")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right")

    plt.tight_layout()
    idx = explanation["borrower_idx"]
    plt.savefig(REPORTS / f"11_borrower_{idx}_explanation.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Saved: 11_borrower_{idx}_explanation.png")