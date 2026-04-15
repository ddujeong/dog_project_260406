import json
import os
import requests


BODY_LABEL_KO = {
    "slim": "마른형",
    "standard": "표준형",
    "broad": "체폭발달형",
    "overweight": "과체형",
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "gemma2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))


def label_ko(label: str) -> str:
    return BODY_LABEL_KO.get(label, label)


def build_body_interpretation_hint(primary: str, secondary: str) -> str:
    if primary == "standard" and secondary == "overweight":
        return "전반적으로는 표준 체형에 가깝지만, 일부 비율에서는 과체형 경향이 함께 관찰됩니다."
    if primary == "standard" and secondary == "broad":
        return "전반적으로는 표준 체형에 가깝지만, 체폭이 다소 발달해 보이는 특징이 함께 나타납니다."
    if primary == "slim" and secondary == "standard":
        return "마른형 경향이 조금 더 강하지만, 전체적인 균형은 표준 체형과도 일부 유사합니다."
    if primary == "slim" and secondary == "overweight":
        return "전반적으로는 마른형 경향이 우세하지만, 일부 비율에서는 과체형 신호가 함께 나타납니다."
    if primary == "broad" and secondary == "standard":
        return "체폭이 발달한 편으로 보이지만, 전체적인 균형은 표준 체형과도 일부 유사합니다."
    if primary == "broad" and secondary == "overweight":
        return "체폭이 발달한 편이며, 일부 비율에서는 과체형 가능성도 함께 나타납니다."
    if primary == "overweight" and secondary == "broad":
        return "과체형 가능성이 상대적으로 더 크며, 체폭이 넓어 보이는 특징도 함께 나타납니다."
    if primary == "overweight" and secondary == "standard":
        return "과체형 경향이 우세하지만, 일부 비율에서는 표준 체형과 유사한 균형도 함께 보입니다."

    primary_ko = label_ko(primary)
    secondary_ko = label_ko(secondary)
    return f"주요 체형은 {primary_ko}, 보조 체형은 {secondary_ko} 경향으로 해석할 수 있습니다."


def build_body_management_points(primary: str, secondary: str) -> list[str]:
    if primary == "standard" and secondary == "overweight":
        return [
            "전반적인 체형 균형은 유지되고 있지만, 체중 증가 방향의 변화가 있는지 함께 살펴보는 것이 좋습니다.",
            "간식량과 식사량이 조금씩 늘고 있지는 않은지 점검해 보세요.",
            "활동량이 충분한지 확인하면서 체중 변화를 주기적으로 기록해 두면 도움이 됩니다.",
        ]

    if primary == "standard":
        return [
            "현재 식사량과 활동량의 균형을 유지하면서 체형 변화를 주기적으로 확인해 보세요.",
            "간식량이 과해지지 않도록 관리하면 표준 체형 유지에 도움이 됩니다.",
            "정기적으로 체중과 몸 상태를 기록해 두면 변화 파악이 쉬워집니다.",
        ]

    if primary == "overweight":
        return [
            "식사량과 간식량을 함께 점검하면서 체중 변화를 기록해 보세요.",
            "무리가 적은 산책이나 일상 활동을 꾸준히 유지하는 것이 도움이 될 수 있습니다.",
            "체중 증가로 관절 부담이 커질 수 있어 움직임 변화를 함께 살펴보는 것이 좋습니다.",
        ]

    if primary == "slim":
        return [
            "식사량 대비 체중이 너무 빠지지 않는지 주기적으로 확인해 보세요.",
            "활동량과 체력 상태를 함께 보면서 근육량 유지도 같이 살펴보는 것이 좋습니다.",
            "몸 상태 변화가 지속되면 추가 정보를 함께 확인해 보는 것이 도움이 됩니다.",
        ]

    if primary == "broad":
        return [
            "체폭이 발달한 체형일 수 있으므로 실제 체중과 활동량을 함께 확인해 보세요.",
            "과체중으로 단정하기보다 몸통 비율과 체중 변화를 같이 살펴보는 것이 좋습니다.",
            "관절에 부담이 가지 않도록 무리 없는 운동량을 유지해 보세요.",
        ]

    return [
        "식사량과 활동량의 균형을 유지하면서 체형 변화를 주기적으로 확인해 보세요.",
        "체중 변화를 기록해 두면 상태 파악에 도움이 됩니다.",
        "생활 습관과 몸 상태를 함께 관찰해 보세요.",
    ]


def preprocess_summary_for_body_llm(summary: dict) -> dict:
    """
    체형 리포트용 summary 전처리
    - visual_focus 제거
    - body 해석 힌트 추가
    - body 관리 포인트 추가
    """
    processed = {
        "breed": dict(summary.get("breed", {})),
        "body": dict(summary.get("body", {})),
    }

    primary = processed.get("body", {}).get("primary", "")
    secondary = processed.get("body", {}).get("secondary", "")

    processed["body"]["primary_ko"] = label_ko(primary)
    processed["body"]["secondary_ko"] = label_ko(secondary)
    processed["body"]["interpretation_hint"] = build_body_interpretation_hint(primary, secondary)
    processed["body"]["management_points"] = build_body_management_points(primary, secondary)

    return processed


def build_body_report_prompt(summary: dict) -> str:
    return f"""
너는 반려견 '체형 분석 결과'를 보호자에게 설명하는 AI 도우미다.

매우 중요한 원칙:
1. 너는 이미지를 직접 보고 판단하지 않는다.
2. 반드시 아래 제공된 모델 결과만 근거로 설명한다.
3. 반드시 체형 중심으로만 작성한다.
4. 얼굴, 귀, 표정, 털, 분위기, 귀여움, 일반 외모 감상은 절대 쓰지 않는다.
5. 의료 진단처럼 단정하지 않는다.
6. '추정', '경향', '가능성', '참고용' 같은 표현을 사용한다.
7. 모델 결과에 없는 사실은 추가하지 않는다.
8. 보호자가 실제로 참고할 수 있는 관리 포인트를 포함한다.
9. breed 정보는 필요할 때만 아주 짧게 참고 수준으로만 언급한다.
10. visual_focus 관련 내용은 사용하지 않는다.

체형 해석 기준:
- slim: 마른형 경향
- standard: 표준형 경향
- broad: 체폭이 발달했거나 넓어 보이는 체형 경향
- overweight: 과체형 가능성

작성 규칙:
- body.primary를 중심으로 설명한다.
- body.secondary는 함께 보이는 보조 경향으로만 설명한다.
- body.interpretation_hint가 있으면 체형 설명의 핵심 표현으로 활용한다.
- body.management_points를 보호자 참고사항에 자연스럽게 반영한다.
- 과체형이면 체중 관리, 활동량, 관절 부담 같은 내용을 중심으로 쓴다.
- 마른형이면 식사량, 체중 유지, 근육량 관찰 같은 내용을 중심으로 쓴다.
- 표준형이면 현재 균형 유지와 주기적 관찰 중심으로 쓴다.
- 체폭발달형이면 과체중으로 단정하지 말고 체형 비율, 활동량, 관절 부담 가능성을 함께 안내한다.
- 불필요하게 장황하게 쓰지 않는다.
- 마지막에는 단일 이미지 기반 참고용 분석이라는 한계를 1문장 포함한다.
- 같은 의미를 반복하지 마라.
- 각 문장은 새로운 정보만 포함해야 한다.
- 이미 설명한 내용을 다른 표현으로 반복하지 마라.
- 각 문장은 구체적인 의미를 가져야 하며, 추상적인 표현만 반복하지 마라.
- "경향이 있다" 대신 "무엇을 확인해야 하는지"를 함께 설명하라.

출력 형식:
반드시 아래 4개 섹션 제목을 그대로 사용해서 완성된 형태로 작성하라.

### 한줄 요약
- 1~2문장

### 체형 설명
- primary/secondary 조합의 해석만 설명한다.
- "왜 이런 체형으로 해석되는지" 중심

### 외형 분석 포인트
- 실제로 관찰해야 할 포인트만 작성한다.
- 예:
  - 몸통 비율
  - 체폭 변화
  - 체중 증가 신호
- 체형 설명 문장을 반복하지 마라.

### 보호자 참고사항
- 3문장 이상
- 보호자가 실제로 참고할 수 있는 관리 포인트 포함
- 마지막 문장은 단일 이미지 기반 참고용이라는 한계 설명

분석 결과:
{json.dumps(summary, ensure_ascii=False, indent=2)}
""".strip()


def build_fallback_report(summary: dict) -> str:
    breed_primary = summary.get("breed", {}).get("primary", "확인 어려움")
    breed_secondary = summary.get("breed", {}).get("secondary", "확인 어려움")

    body = summary.get("body", {})
    body_primary = body.get("primary", "unknown")
    body_secondary = body.get("secondary", "unknown")
    body_primary_ko = body.get("primary_ko", label_ko(body_primary))
    body_secondary_ko = body.get("secondary_ko", label_ko(body_secondary))
    interpretation_hint = body.get(
        "interpretation_hint",
        f"주요 체형은 {body_primary_ko}, 보조 체형은 {body_secondary_ko} 경향으로 해석할 수 있습니다."
    )
    management_points = body.get("management_points", [])

    point_1 = management_points[0] if len(management_points) > 0 else "식사량과 활동량의 균형을 함께 살펴보는 것이 좋습니다."
    point_2 = management_points[1] if len(management_points) > 1 else "체중 변화를 주기적으로 확인하면 도움이 됩니다."
    point_3 = management_points[2] if len(management_points) > 2 else "생활 습관과 몸 상태를 함께 관찰해 보세요."

    breed_text = ""
    if breed_primary != "확인 어려움":
        breed_text = f"견종 정보는 {breed_primary}"
        if breed_secondary != "확인 어려움":
            breed_text += f", {breed_secondary}"
        breed_text += " 특징이 참고 수준으로 함께 반영되었습니다."

    return f"""
### 한줄 요약
전반적으로 **{body_primary_ko}** 경향이 중심이며, **{body_secondary_ko}** 특징도 일부 함께 관찰됩니다.

### 체형 설명
모델 분석 결과를 기준으로 보면 주요 체형은 **{body_primary_ko}**, 보조 체형은 **{body_secondary_ko}**로 해석됩니다.
{interpretation_hint}

### 외형 분석 포인트
전체적인 체형 비율과 몸통 중심의 균형을 바탕으로 체형 경향을 해석한 결과입니다.
{breed_text if breed_text else "견종 정보는 체형 설명의 보조 참고 정보로만 사용되었습니다."}

### 보호자 참고사항
{point_1}
{point_2}
{point_3}
본 결과는 단일 이미지 기반의 참고용 분석이므로, 실제 체중과 활동량, 생활 습관 정보와 함께 종합적으로 살펴보는 것이 좋습니다.
""".strip()


def is_ollama_available() -> tuple[bool, str | None]:
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        response.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)


def is_incomplete_report(text: str) -> bool:
    required_sections = [
        "### 한줄 요약",
        "### 체형 설명",
        "### 외형 분석 포인트",
        "### 보호자 참고사항",
    ]

    if len(text.strip()) < 180:
        return True

    missing_count = sum(1 for section in required_sections if section not in text)
    return missing_count >= 1

def _request_ollama(prompt: str, model_name: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 900,
        }
    }

    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    print("[OLLAMA RAW RESPONSE]", data)

    text = data.get("response", "").strip()

    if not text:
        raise ValueError("Ollama 응답이 비어 있습니다.")

    if is_incomplete_report(text):
        raise ValueError("불완전한 LLM 응답입니다.")

    return text


def call_real_llm(summary: dict) -> dict:
    ok, err = is_ollama_available()
    if not ok:
        return {
            "report": build_fallback_report(summary),
            "source": "rule_based_fallback",
            "model": None,
            "error": f"Ollama 서버 연결 실패: {err}"
        }

    prompt = build_body_report_prompt(summary)

    try:
        text = _request_ollama(prompt, OLLAMA_MODEL)
        return {
            "report": text,
            "source": "main",
            "model": OLLAMA_MODEL,
            "error": None
        }
    except Exception as e1:
        print(f"[WARN] 메인 Ollama 호출 실패: {e1}")

    try:
        text = _request_ollama(prompt, OLLAMA_FALLBACK_MODEL)
        return {
            "report": text,
            "source": "fallback_model",
            "model": OLLAMA_FALLBACK_MODEL,
            "error": None
        }
    except Exception as e2:
        print(f"[WARN] fallback Ollama 호출 실패: {e2}")
        return {
            "report": build_fallback_report(summary),
            "source": "rule_based_fallback",
            "model": None,
            "error": str(e2)
        }


def generate_llm_body_report(summary: dict) -> dict:
    processed_summary = preprocess_summary_for_body_llm(summary)
    return call_real_llm(processed_summary)