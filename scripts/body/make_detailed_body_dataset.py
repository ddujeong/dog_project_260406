import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from services.body_data_service import build_body_dataset
from services.body_label_service import make_detailed_body_label


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["weight_height_ratio"] = df["weight"] / df["shoulder_height"]
    df["chest_height_ratio"] = df["chest_size"] / df["shoulder_height"]
    df["back_height_ratio"] = df["back_length"] / df["shoulder_height"]
    df["neck_chest_ratio"] = df["neck_size"] / df["chest_size"]

    return df


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
    df["detailed_body_label"] = df.apply(make_detailed_body_label, axis=1)

    print("\n=== DETAILED LABEL DISTRIBUTION ===")
    print(df["detailed_body_label"].value_counts(dropna=False))

    print("\n=== BCS x DETAILED LABEL ===")
    print(pd.crosstab(df["bcs"], df["detailed_body_label"]))

    output_cols = [
        "image_path",
        "image_id",
        "breed",
        "age",
        "sex",
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
        "detailed_body_label",
    ]

    output_path = os.path.join(PROJECT_ROOT, "data", "aihub_body", "detailed_body_dataset.csv")
    df[output_cols].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()