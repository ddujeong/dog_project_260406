import os
import sys

import pandas as pd

# 프로젝트 루트 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from services.body_data_service import get_training_dataframe


def main():
    labels_dir = os.path.join(PROJECT_ROOT, "data", "aihub_body", "labels")
    images_dir = os.path.join(PROJECT_ROOT, "data", "aihub_body", "images")
    output_dir = os.path.join(PROJECT_ROOT, "data", "aihub_body")
    output_csv = os.path.join(output_dir, "body_dataset.csv")

    df = get_training_dataframe(labels_dir=labels_dir, images_dir=images_dir)

    print("\n=== body dataset build result ===")
    print(f"total rows: {len(df)}")

    if df.empty:
        print("데이터셋이 비어 있습니다.")
        return

    print("\n[columns]")
    print(df.columns.tolist())

    print("\n[label distribution]")
    print(df["body_label"].value_counts(dropna=False))

    print("\n[missing values]")
    print(df.isnull().sum())

    print("\n[sample rows]")
    print(df.head())

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\nCSV 저장 완료: {output_csv}")


if __name__ == "__main__":
    main()