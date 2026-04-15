import json


BODY_LABEL_KO = {
    "slim": "마른형",
    "standard": "표준형",
    "broad": "체폭발달형",
    "overweight": "과체형",
}


def build_body_report_prompt(summary: dict) -> str:
    return f"""
너는 반려견 이미지 분석 결과를 보호자에게 설명하는 AI 도우미다.

규칙:
1. 단정하지 말고 '추정', '경향', '가능성' 표현을 사용한다.
2. 의료 진단처럼 말하지 않는다.
3. 한국어로 자연스럽게 작성한다.
4. 반드시 아래 분석 결과만 근거로 작성한다.
5. 출력은 아래 4개 섹션으로 작성한다.
- 한줄 요약
- 체형 설명
- 외형 분석 포인트
- 보호자 참고사항

분석 결과:
{json.dumps(summary, ensure_ascii=False, indent=2)}
""".strip()


def call_dummy_llm(summary: dict) -> str:
    breed_primary = summary["breed"]["primary"]
    breed_secondary = summary["breed"]["secondary"]

    body_primary = summary["body"]["primary"]
    body_secondary = summary["body"]["secondary"]

    body_primary_ko = BODY_LABEL_KO.get(body_primary, body_primary)
    body_secondary_ko = BODY_LABEL_KO.get(body_secondary, body_secondary)

    region = summary["visual_focus"]["region"]

    return f"""
### 한줄 요약
이 강아지는 **{body_primary_ko}** 체형으로 추정되며, **{body_secondary_ko}** 특징도 일부 함께 보입니다.

### 체형 설명
이미지 기반 분석 결과, 주요 체형은 **{body_primary_ko}**, 보조 체형은 **{body_secondary_ko}**로 나타났습니다.
즉 전반적인 체형은 {body_primary_ko}에 가깝지만, 일부 비율에서는 {body_secondary_ko} 경향도 함께 보이는 복합적인 외형으로 해석할 수 있습니다.

### 외형 분석 포인트
AI는 주로 **{region}** 부위를 중심으로 외형을 분석했습니다.
견종은 **{breed_primary}** 특징이 가장 강하게 나타났고, **{breed_secondary}** 특징도 일부 함께 관찰되었습니다.

### 보호자 참고사항
본 결과는 단일 이미지 기반 추정이므로 촬영 각도, 자세, 털 길이, 조명에 따라 실제 체형과 차이가 있을 수 있습니다.
정확한 상태 확인은 실제 체중, 체형 측정, 활동량, 식이 정보와 함께 판단하는 것이 좋습니다.
""".strip()


def generate_llm_body_report(summary: dict) -> str:
    _ = build_body_report_prompt(summary)
    return call_dummy_llm(summary)