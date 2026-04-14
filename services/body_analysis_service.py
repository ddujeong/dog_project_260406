import numpy as np
import pandas as pd


BODY_NUMERIC_COLUMNS = [
    "weight",
    "shoulder_height",
    "neck_size",
    "back_length",
    "chest_size",
    "bcs",
]


def to_numeric_body_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in BODY_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_body_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 0 division 방지
    shoulder = df["shoulder_height"].replace(0, np.nan)
    chest = df["chest_size"].replace(0, np.nan)

    df["weight_height_ratio"] = df["weight"] / shoulder
    df["chest_height_ratio"] = df["chest_size"] / shoulder
    df["back_height_ratio"] = df["back_length"] / shoulder
    df["neck_chest_ratio"] = df["neck_size"] / chest

    return df


def make_body_vector(row: pd.Series) -> np.ndarray:
    features = [
        row.get("weight", np.nan),
        row.get("shoulder_height", np.nan),
        row.get("neck_size", np.nan),
        row.get("back_length", np.nan),
        row.get("chest_size", np.nan),
        row.get("bcs", np.nan),
        row.get("weight_height_ratio", np.nan),
        row.get("chest_height_ratio", np.nan),
        row.get("back_height_ratio", np.nan),
        row.get("neck_chest_ratio", np.nan),
    ]

    return np.array(features, dtype=float)


def add_body_vectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body_vector"] = df.apply(make_body_vector, axis=1)
    return df


def summarize_body_stats(df: pd.DataFrame) -> dict:
    numeric_cols = [
        "weight",
        "shoulder_height",
        "neck_size",
        "back_length",
        "chest_size",
        "bcs",
        "weight_height_ratio",
        "chest_height_ratio",
        "back_height_ratio",
        "neck_chest_ratio",
    ]

    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": float(df[col].mean()) if col in df else None,
            "std": float(df[col].std()) if col in df else None,
            "min": float(df[col].min()) if col in df else None,
            "max": float(df[col].max()) if col in df else None,
        }

    return summary


def classify_body_type(row: pd.Series) -> str:
    ratio = row.get("chest_height_ratio")
    bcs = row.get("bcs")

    if pd.isna(ratio) or pd.isna(bcs):
        return "분석불가"

    if bcs >= 6:
        return "통통형"
    if bcs <= 3:
        return "마른형"

    if ratio >= 1.45:
        return "체폭발달형"
    elif ratio <= 1.1:
        return "슬림형"
    else:
        return "표준형"


def add_body_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body_type"] = df.apply(classify_body_type, axis=1)
    return df