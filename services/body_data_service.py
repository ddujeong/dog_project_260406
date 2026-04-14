import os
import json
import glob
from typing import Optional

import pandas as pd


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def find_json_files(labels_dir: str) -> list[str]:
    return glob.glob(os.path.join(labels_dir, "**", "*.json"), recursive=True)


def build_image_map(images_dir: str) -> dict[str, str]:
    image_map = {}

    for ext in IMAGE_EXTENSIONS:
        image_paths = glob.glob(os.path.join(images_dir, "**", ext), recursive=True)
        for path in image_paths:
            filename = os.path.basename(path)
            image_map[filename] = path

    return image_map


def safe_get(data: dict, keys: list[str], default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def make_body_label_from_bcs(bcs) -> Optional[str]:
    if pd.isna(bcs):
        return None

    try:
        bcs = float(bcs)
    except (TypeError, ValueError):
        return None

    if bcs <= 3:
        return "slim"
    elif bcs <= 5:
        return "normal"
    else:
        return "overweight"


def parse_body_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    metadata = raw.get("metadata", {})
    physical = safe_get(raw, ["metadata", "physical"], {})
    annotation = raw.get("annotations", {})

    row = {
        "json_path": json_path,
        "image_id": annotation.get("image-id"),

        "species": safe_get(metadata, ["id", "species"]),
        "mission_id": safe_get(metadata, ["id", "mission-id"]),
        "provider_code": safe_get(metadata, ["id", "provider-code"]),
        "breed": safe_get(metadata, ["id", "breed"]),
        "age": safe_get(metadata, ["id", "age"]),
        "class": safe_get(metadata, ["id", "class"]),
        "sex": safe_get(metadata, ["id", "sex"]),
        "group": safe_get(metadata, ["id", "group"]),

        "weight": physical.get("weight"),
        "shoulder_height": physical.get("shoulder-height"),
        "neck_size": physical.get("neck-size"),
        "back_length": physical.get("back-length"),
        "chest_size": physical.get("chest-size"),
        "bcs": physical.get("BCS"),

        "exercise": safe_get(metadata, ["breeding", "exercise"]),
        "food_count": safe_get(metadata, ["breeding", "food-count"]),
        "environment": safe_get(metadata, ["breeding", "environment"]),
        "defecation": safe_get(metadata, ["breeding", "defecation"]),
        "food_amount": safe_get(metadata, ["breeding", "food-amount"]),
        "snack_amount": safe_get(metadata, ["breeding", "snack-amount"]),
        "food_kind": safe_get(metadata, ["breeding", "food-kind"]),
    }

    return row


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "age",
        "weight",
        "shoulder_height",
        "neck_size",
        "back_length",
        "chest_size",
        "bcs",
        "exercise",
        "food_count",
        "environment",
        "defecation",
        "food_amount",
        "snack_amount",
        "food_kind",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_body_dataset(
    labels_dir: str = "data/aihub_body/labels",
    images_dir: str = "data/aihub_body/images",
    drop_missing_image: bool = True,
    drop_missing_label: bool = True,
) -> pd.DataFrame:
    json_files = find_json_files(labels_dir)
    image_map = build_image_map(images_dir)

    rows = []
    for json_path in json_files:
        try:
            row = parse_body_json(json_path)
            rows.append(row)
        except Exception as e:
            print(f"[JSON 파싱 실패] {json_path}: {e}")

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = convert_numeric_columns(df)

    df["image_path"] = df["image_id"].map(image_map)
    df["body_label"] = df["bcs"].apply(make_body_label_from_bcs)

    if drop_missing_image:
        df = df[df["image_path"].notna()].copy()

    if drop_missing_label:
        df = df[df["body_label"].notna()].copy()

    df = df.reset_index(drop=True)
    return df


def get_training_dataframe(
    labels_dir: str = "data/aihub_body/labels",
    images_dir: str = "data/aihub_body/images",
) -> pd.DataFrame:
    df = build_body_dataset(labels_dir=labels_dir, images_dir=images_dir)

    if df.empty:
        return df

    train_cols = [
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
        "body_label",
    ]

    existing_cols = [col for col in train_cols if col in df.columns]
    return df[existing_cols].copy()