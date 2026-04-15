import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from services.body_data_service import build_body_dataset


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["weight_height_ratio"] = df["weight"] / df["shoulder_height"]
    df["chest_height_ratio"] = df["chest_size"] / df["shoulder_height"]
    df["back_height_ratio"] = df["back_length"] / df["shoulder_height"]
    df["neck_chest_ratio"] = df["neck_size"] / df["chest_size"]

    return df


def print_basic_info(df: pd.DataFrame):
    print("\n=== BASIC INFO ===")
    print(f"total rows: {len(df)}")

    print("\n=== BCS DISTRIBUTION ===")
    print(df["bcs"].value_counts().sort_index())

    print("\n=== BREED TOP 20 ===")
    print(df["breed"].value_counts().head(20))

    print("\n=== SEX DISTRIBUTION ===")
    print(df["sex"].value_counts())


def print_numeric_summary(df: pd.DataFrame):
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

    print("\n=== NUMERIC SUMMARY ===")
    print(df[numeric_cols].describe().T)


def print_bcs_group_summary(df: pd.DataFrame):
    group_cols = [
        "weight",
        "shoulder_height",
        "neck_size",
        "back_length",
        "chest_size",
        "weight_height_ratio",
        "chest_height_ratio",
        "back_height_ratio",
        "neck_chest_ratio",
    ]

    print("\n=== GROUPED BY BCS ===")
    print(df.groupby("bcs")[group_cols].mean().round(3))


def print_quantiles(df: pd.DataFrame):
    target_cols = [
        "weight_height_ratio",
        "chest_height_ratio",
        "back_height_ratio",
        "neck_chest_ratio",
    ]

    print("\n=== FEATURE QUANTILES ===")
    for col in target_cols:
        print(f"\n[{col}]")
        print(df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3))


def save_feature_dataset(df: pd.DataFrame):
    output_path = os.path.join(PROJECT_ROOT, "data", "aihub_body", "body_feature_dataset.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nfeature dataset 저장 완료: {output_path}")


def main():
    labels_dir = os.path.join(PROJECT_ROOT, "data", "aihub_body", "labels")
    images_dir = os.path.join(PROJECT_ROOT, "data", "aihub_body", "images")

    df = build_body_dataset(
        labels_dir=labels_dir,
        images_dir=images_dir,
        drop_missing_image=True,
        drop_missing_label=False
    )

    if df.empty:
        print("데이터가 비어 있습니다.")
        return

    df = add_ratio_features(df)

    print_basic_info(df)
    print_numeric_summary(df)
    print_bcs_group_summary(df)
    print_quantiles(df)
    save_feature_dataset(df)


if __name__ == "__main__":
    main()