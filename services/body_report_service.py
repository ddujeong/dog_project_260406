import pandas as pd


def build_body_comment(row: pd.Series) -> str:
    body_type = row.get("body_type", "분석불가")
    bcs = row.get("bcs")
    weight = row.get("weight")
    shoulder_height = row.get("shoulder_height")
    chest_size = row.get("chest_size")

    if body_type == "분석불가":
        return "체형 데이터가 충분하지 않아 분석이 어렵습니다."

    comments = []

    comments.append(f"이 개체는 전체적으로 '{body_type}' 체형으로 분류됩니다.")

    if pd.notna(bcs):
        if bcs >= 6:
            comments.append("BCS 기준으로 체지방이 다소 높은 편으로 해석할 수 있습니다.")
        elif bcs <= 3:
            comments.append("BCS 기준으로 비교적 마른 체형에 가깝습니다.")
        else:
            comments.append("BCS 기준으로는 대체로 적정 범위에 가깝습니다.")

    if pd.notna(weight) and pd.notna(shoulder_height):
        comments.append(f"체중 {weight}, 체고 {shoulder_height} 데이터를 기반으로 체형 비율을 계산했습니다.")

    if pd.notna(chest_size) and pd.notna(shoulder_height):
        comments.append("흉부 둘레 대비 체고 비율을 통해 체폭 발달 정도를 함께 반영했습니다.")

    return " ".join(comments)
def build_body_type_description(primary, secondary):
    desc_map = {
        "slim": "마른 체형으로 체지방이 낮은 편입니다.",
        "standard": "표준 체형으로 균형 잡힌 신체 비율을 보입니다.",
        "broad": "흉곽이 발달된 체형으로 체폭이 넓은 편입니다.",
        "overweight": "체지방이 많은 과체형 경향이 있습니다.",
    }

    primary_desc = desc_map.get(primary, "")
    secondary_desc = desc_map.get(secondary, "")

    return f"{primary_desc} (보조 특징: {secondary_desc})"

def build_body_report_dict(row: pd.Series) -> dict:
    return {
        "image_id": row.get("image_id"),
        "breed": row.get("breed"),
        "age": row.get("age"),
        "sex": row.get("sex"),
        "weight": row.get("weight"),
        "shoulder_height": row.get("shoulder_height"),
        "neck_size": row.get("neck_size"),
        "back_length": row.get("back_length"),
        "chest_size": row.get("chest_size"),
        "bcs": row.get("bcs"),
        "body_type": row.get("body_type"),
        "comment": build_body_comment(row),
    }