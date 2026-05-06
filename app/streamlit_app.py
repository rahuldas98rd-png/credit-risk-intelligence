import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import ROOT_DIR, DATA_PROC, get_logger

logger = get_logger("streamlit")
MODELS_DIR = ROOT_DIR / "models"

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    fold_models   = joblib.load(MODELS_DIR / "lgbm_folds.pkl")
    meta_model    = joblib.load(MODELS_DIR / "meta_learner.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    explainer     = shap.TreeExplainer(fold_models[0])
    fi_df         = pd.read_csv(DATA_PROC / "feature_importance.csv")
    shap_df       = pd.read_csv(DATA_PROC / "shap_importance.csv")
    return fold_models, meta_model, feature_names, explainer, fi_df, shap_df

fold_models, meta_model, feature_names, explainer, fi_df, shap_df = load_artifacts()

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict(features: dict) -> tuple:
    row = pd.DataFrame([{f: features.get(f, 0.0) for f in feature_names}])
    fold_probs = np.mean([m.predict_proba(row)[0, 1] for m in fold_models])
    prob = float(meta_model.predict_proba([[fold_probs]])[0, 1])
    return prob, row

def get_shap(row):
    sv = explainer.shap_values(row)
    return sv[1][0] if isinstance(sv, list) else sv[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Credit Risk Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Business Simulator", "🔍 Borrower Explainer", "📈 Model Insights"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — BUSINESS SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Business Simulator":
    st.title("📊 Approval Threshold Business Simulator")
    st.markdown("Adjust the approval threshold to see the business impact in real time.")

    col1, col2, col3 = st.columns(3)
    with col1:
        threshold   = st.slider("Approval threshold", 0.10, 0.99, 0.8456, 0.01,
                                help="Applicants above this probability are denied")
    with col2:
        avg_loan    = st.number_input("Avg loan value ($)", 10000, 500000, 100000, 5000)
    with col3:
        loss_rate   = st.slider("Loss given default (%)", 10, 100, 45, 5) / 100

    # Simulate on OOF predictions stored in session or use synthetic probs
    np.random.seed(42)
    n = 10000
    true_default_rate = 0.0807
    sim_probs = np.concatenate([
        np.random.beta(2, 18, int(n * (1 - true_default_rate))),
        np.random.beta(8, 4,  int(n * true_default_rate))
    ])
    sim_labels = np.concatenate([
        np.zeros(int(n * (1 - true_default_rate))),
        np.ones(int(n * true_default_rate))
    ])

    approved_mask  = sim_probs < threshold
    denied_mask    = ~approved_mask
    approved_total = approved_mask.sum()
    denied_total   = denied_mask.sum()

    true_pos   = (approved_mask & (sim_labels == 1)).sum()   # approved defaults
    true_neg   = (denied_mask  & (sim_labels == 1)).sum()    # correctly denied defaults
    false_neg  = (denied_mask  & (sim_labels == 0)).sum()    # good borrowers denied

    expected_loss   = true_pos  * avg_loan * loss_rate
    foregone_rev    = false_neg * avg_loan * 0.05
    net_position    = (approved_mask & (sim_labels == 0)).sum() * avg_loan * 0.05 - expected_loss
    approval_rate   = approved_total / n * 100

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Approval rate",     f"{approval_rate:.1f}%")
    m2.metric("Expected loss",     f"${expected_loss:,.0f}")
    m3.metric("Foregone revenue",  f"${foregone_rev:,.0f}")
    m4.metric("Net position",      f"${net_position:,.0f}",
              delta="positive" if net_position > 0 else "negative")

    # Threshold sweep chart
    thresholds  = np.arange(0.05, 1.0, 0.01)
    losses, revenues, approvals = [], [], []
    for t in thresholds:
        am = sim_probs < t
        tp = (am & (sim_labels == 1)).sum()
        fn = (~am & (sim_labels == 0)).sum()
        ap = (am & (sim_labels == 0)).sum()
        losses.append(tp * avg_loan * loss_rate)
        revenues.append(ap * avg_loan * 0.05)
        approvals.append(am.sum() / n * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=revenues, name="Revenue",
                             line=dict(color="#4C9BE8", width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=losses,   name="Expected Loss",
                             line=dict(color="#E8593C", width=2)))
    fig.add_vline(x=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Current: {threshold:.2f}")
    fig.update_layout(
        title="Revenue vs Expected Loss Across Thresholds",
        xaxis_title="Approval Threshold",
        yaxis_title="Dollar Amount ($)",
        height=400, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=thresholds, y=approvals, name="Approval Rate %",
                              line=dict(color="#1D9E75", width=2), fill="tozeroy",
                              fillcolor="rgba(29,158,117,0.1)"))
    fig2.add_vline(x=threshold, line_dash="dash", line_color="gray")
    fig2.update_layout(title="Approval Rate Across Thresholds",
                       xaxis_title="Threshold", yaxis_title="Approval Rate (%)",
                       height=300, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BORROWER EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Borrower Explainer":
    st.title("🔍 Individual Borrower Risk Explainer")
    st.markdown("Enter borrower details to get a real-time risk prediction with explanations.")

    with st.form("borrower_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ext1 = st.slider("EXT_SOURCE_1 (bureau score 1)", 0.0, 1.0, 0.5, 0.01)
            ext2 = st.slider("EXT_SOURCE_2 (bureau score 2)", 0.0, 1.0, 0.5, 0.01)
            ext3 = st.slider("EXT_SOURCE_3 (bureau score 3)", 0.0, 1.0, 0.5, 0.01)
        with col2:
            income       = st.number_input("Annual income ($)", 10000, 1000000, 150000, 5000)
            credit_amt   = st.number_input("Loan amount ($)",   10000, 2000000, 300000, 5000)
            annuity      = st.number_input("Monthly annuity ($)", 1000, 100000, 15000, 500)
        with col3:
            age_years    = st.slider("Age (years)", 20, 70, 40)
            days_employed = st.slider("Years employed", 0, 40, 5) * 365
            goods_price  = st.number_input("Goods price ($)", 5000, 2000000, 250000, 5000)

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    if submitted:
        features = {f: 0.0 for f in feature_names}
        features.update({
            "EXT_SOURCE_1":         ext1,
            "EXT_SOURCE_2":         ext2,
            "EXT_SOURCE_3":         ext3,
            "AMT_INCOME_TOTAL":     float(income),
            "AMT_CREDIT":           float(credit_amt),
            "AMT_ANNUITY":          float(annuity),
            "AMT_GOODS_PRICE":      float(goods_price),
            "DAYS_BIRTH":           age_years * 365,
            "DAYS_EMPLOYED":        float(days_employed),
            "EXT_SOURCE_MEAN":      (ext1 + ext2 + ext3) / 3,
            "EXT_SOURCE_STD":       float(np.std([ext1, ext2, ext3])),
            "EXT_SOURCE_MIN":       min(ext1, ext2, ext3),
            "EXT_SOURCE_PROD":      ext1 * ext2 * ext3,
            "CREDIT_INCOME_RATIO":  credit_amt / (income + 1),
            "ANNUITY_INCOME_RATIO": annuity / (income + 1),
            "CREDIT_TERM":          annuity / (credit_amt + 1),
            "GOODS_CREDIT_RATIO":   goods_price / (credit_amt + 1),
            "AGE_YEARS":            float(age_years),
        })

        prob, row = predict(features)
        sv = get_shap(row)

        # Decision banner
        THRESHOLD = 0.8456
        if prob >= THRESHOLD:
            st.error(f"🚫 HIGH RISK — DENY  |  Default probability: {prob:.1%}")
        elif prob >= 0.5:
            st.warning(f"⚠️ MEDIUM RISK — MANUAL REVIEW  |  Default probability: {prob:.1%}")
        else:
            st.success(f"✅ LOW RISK — APPROVE  |  Default probability: {prob:.1%}")

        # SHAP waterfall chart
        contrib = sorted(
            zip(feature_names, row.values[0], sv),
            key=lambda x: abs(x[2]), reverse=True
        )[:10]
        feats  = [c[0] for c in contrib]
        vals   = [c[2] for c in contrib]
        colors = ["#E8593C" if v > 0 else "#4C9BE8" for v in vals]

        fig = go.Figure(go.Bar(
            x=vals, y=feats, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in vals],
            textposition="outside"
        ))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(
            title=f"Top 10 SHAP Contributions — Default Prob: {prob:.1%}",
            xaxis_title="SHAP value (impact on default probability)",
            height=450, template="plotly_white",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Performance & Feature Insights")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC",   "0.8976")
    col2.metric("PR-AUC",    "0.7990")
    col3.metric("Best F1",   "0.7271")
    col4.metric("Threshold", "0.8456")

    tab1, tab2 = st.tabs(["Feature Importance (Gain)", "SHAP Importance"])

    with tab1:
        top_fi = fi_df.head(20)
        fig = px.bar(top_fi[::-1], x="importance", y="feature",
                     orientation="h", color="importance",
                     color_continuous_scale="Blues",
                     title="Top 20 Features by LightGBM Gain")
        fig.update_layout(height=550, template="plotly_white",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_shap = shap_df.head(20)
        fig = px.bar(top_shap[::-1], x="mean_abs_shap", y="feature",
                     orientation="h", color="mean_abs_shap",
                     color_continuous_scale="Reds",
                     title="Top 20 Features by Mean |SHAP|")
        fig.update_layout(height=550, template="plotly_white",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)