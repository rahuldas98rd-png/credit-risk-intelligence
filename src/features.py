import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.utils import get_logger

logger = get_logger("features")


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.60) -> tuple[pd.DataFrame, list]:
    missing_pct = df.isnull().mean()
    cols = missing_pct[missing_pct > threshold].index.tolist()
    return df.drop(columns=cols), cols


def add_missingness_flags(df: pd.DataFrame, low: float = 0.40, high: float = 0.60) -> pd.DataFrame:
    missing_pct = df.isnull().mean()
    flag_cols = missing_pct[(missing_pct > low) & (missing_pct <= high)].index.tolist()
    for col in flag_cols:
        df[f"{col}_MISSING"] = df[col].isnull().astype(np.int8)
    return df


def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    day_cols = [c for c in df.columns if "DAYS_" in c]
    for col in day_cols:
        if df[col].min() < 0:
            df[col] = df[col].abs()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["CREDIT_INCOME_RATIO"]      = df["AMT_CREDIT"]      / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"]     = df["AMT_ANNUITY"]     / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"]              = df["AMT_ANNUITY"]     / (df["AMT_CREDIT"] + 1)
    df["GOODS_CREDIT_RATIO"]       = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)
    df["INCOME_PER_PERSON"]        = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["CHILDREN_RATIO"]           = df["CNT_CHILDREN"]    / (df["CNT_FAM_MEMBERS"] + 1)
    df["AGE_YEARS"]                = df["DAYS_BIRTH"]      / 365
    df["EMPLOYMENT_STABILITY"]     = df["DAYS_EMPLOYED"]   / (df["DAYS_BIRTH"] + 1)

    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"]          = df[ext].mean(axis=1)
    df["EXT_SOURCE_STD"]           = df[ext].std(axis=1)
    df["EXT_SOURCE_MIN"]           = df[ext].min(axis=1)
    df["EXT_SOURCE_PROD"]          = df[ext].prod(axis=1)
    df["EXT_SOURCE_DISAGREEMENT"]  = df["EXT_SOURCE_STD"] / (df["EXT_SOURCE_MEAN"] + 1e-5)
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cat_cols = df.select_dtypes("object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna("Unknown")
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def impute_numerics(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df, imputer


def build_features(train_path, test_path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    y = train.pop("TARGET")

    train["_split"] = "train"
    test["_split"]  = "test"
    combined = pd.concat([train, test], ignore_index=True)

    combined, dropped = drop_high_missing(combined)
    combined = add_missingness_flags(combined)
    combined = fix_anomalies(combined)
    combined = engineer_features(combined)

    split_col = combined.pop("_split")
    combined, _ = encode_categoricals(combined)
    combined, _ = impute_numerics(combined)
    combined["_split"] = split_col

    train_out = combined[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    test_out  = combined[combined["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

    logger.info(f"Features built — train: {train_out.shape}, test: {test_out.shape}")
    return train_out, test_out, y