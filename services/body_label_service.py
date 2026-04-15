import pandas as pd


def make_detailed_body_label(row: pd.Series) -> str | None:
    bcs = row.get("bcs")
    weight_height_ratio = row.get("weight_height_ratio")
    chest_height_ratio = row.get("chest_height_ratio")

    if pd.isna(bcs):
        return None

    try:
        bcs = float(bcs)
    except (TypeError, ValueError):
        return None

    # 1) 마른형
    if bcs <= 3:
        return "slim"

    # 2) 과체형
    if bcs >= 6:
        return "overweight"

    # 3) BCS 4~5 구간은 체폭/비율로 세분화
    if pd.notna(chest_height_ratio) and chest_height_ratio >= 1.63:
        return "broad"

    if pd.notna(weight_height_ratio) and weight_height_ratio >= 0.13:
        return "broad"

    # 4) 기본형
    return "standard"