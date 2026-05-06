import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import (
    drop_high_missing, add_missingness_flags,
    fix_anomalies, engineer_features
)


@pytest.fixture
def sample_df():
    """Minimal dataframe mimicking application_train structure."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "AMT_CREDIT":        np.random.uniform(50000, 500000, n),
        "AMT_INCOME_TOTAL":  np.random.uniform(30000, 300000, n),
        "AMT_ANNUITY":       np.random.uniform(5000, 50000, n),
        "AMT_GOODS_PRICE":   np.random.uniform(40000, 400000, n),
        "CNT_FAM_MEMBERS":   np.random.randint(1, 6, n).astype(float),
        "CNT_CHILDREN":      np.random.randint(0, 4, n).astype(float),
        "DAYS_BIRTH":        np.random.randint(7000, 25000, n).astype(float),
        "DAYS_EMPLOYED":     np.random.randint(0, 10000, n).astype(float),
        "EXT_SOURCE_1":      np.random.uniform(0, 1, n),
        "EXT_SOURCE_2":      np.random.uniform(0, 1, n),
        "EXT_SOURCE_3":      np.random.uniform(0, 1, n),
        "HIGH_MISS_COL":     pd.array([np.nan] * 400 + list(np.random.uniform(0, 1, 100))),
        "MED_MISS_COL":      pd.array([np.nan] * 220 + list(np.random.uniform(0, 1, 280))),
    })


# ── drop_high_missing ─────────────────────────────────────────────────────────

def test_drop_high_missing_removes_correct_columns(sample_df):
    result, dropped = drop_high_missing(sample_df, threshold=0.60)
    assert "HIGH_MISS_COL" in dropped
    assert "HIGH_MISS_COL" not in result.columns

def test_drop_high_missing_keeps_low_missing(sample_df):
    result, dropped = drop_high_missing(sample_df, threshold=0.60)
    assert "AMT_CREDIT" not in dropped
    assert "AMT_CREDIT" in result.columns

def test_drop_high_missing_returns_list(sample_df):
    _, dropped = drop_high_missing(sample_df, threshold=0.60)
    assert isinstance(dropped, list)


# ── add_missingness_flags ─────────────────────────────────────────────────────

def test_missingness_flags_created(sample_df):
    result = add_missingness_flags(sample_df, low=0.40, high=0.60)
    assert "MED_MISS_COL_MISSING" in result.columns

def test_missingness_flags_are_binary(sample_df):
    result = add_missingness_flags(sample_df)
    flag_cols = [c for c in result.columns if c.endswith("_MISSING")]
    for col in flag_cols:
        assert set(result[col].unique()).issubset({0, 1})

def test_no_flag_for_complete_columns(sample_df):
    result = add_missingness_flags(sample_df)
    assert "AMT_CREDIT_MISSING" not in result.columns


# ── fix_anomalies ─────────────────────────────────────────────────────────────

def test_days_employed_anomaly_flag(sample_df):
    sample_df.loc[0, "DAYS_EMPLOYED"] = 365243
    result = fix_anomalies(sample_df.copy())
    assert "DAYS_EMPLOYED_ANOMALY" in result.columns
    assert result.loc[0, "DAYS_EMPLOYED_ANOMALY"] == 1

def test_days_employed_365243_replaced_with_nan(sample_df):
    sample_df.loc[0, "DAYS_EMPLOYED"] = 365243
    result = fix_anomalies(sample_df.copy())
    assert pd.isna(result.loc[0, "DAYS_EMPLOYED"])


# ── engineer_features ─────────────────────────────────────────────────────────

def test_engineered_columns_exist(sample_df):
    result = engineer_features(sample_df.copy())
    expected = [
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
        "GOODS_CREDIT_RATIO", "INCOME_PER_PERSON", "AGE_YEARS",
        "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_DISAGREEMENT"
    ]
    for col in expected:
        assert col in result.columns, f"Missing: {col}"

def test_age_years_reasonable(sample_df):
    result = engineer_features(sample_df.copy())
    assert result["AGE_YEARS"].min() >= 19
    assert result["AGE_YEARS"].max() <= 70

def test_ratios_non_negative(sample_df):
    result = engineer_features(sample_df.copy())
    for col in ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM"]:
        assert (result[col] >= 0).all(), f"Negative values in {col}"

def test_ext_source_mean_in_range(sample_df):
    result = engineer_features(sample_df.copy())
    assert result["EXT_SOURCE_MEAN"].between(0, 1).all()