import requests


def build_chatbot_prompt(user_input, contexts):
    context_texts = []

    for i, c in enumerate(contexts, start=1):
        q_part = f"Q: {c['question']}\n" if c.get("question") else ""
        context_texts.append(
            f"[근거 {i}]\n"
            f"source_type: {c.get('source_type', '-')}\n"
            f"{q_part}"
            f"{c['content']}"
        )

    joined_context = "\n\n".join(context_texts)

    prompt = f"""
너는 반려견 건강 상담 보조 AI다.

[핵심 역할]
- 사용자의 현재 상황을 이해하고, 가장 현실적인 수준에서 설명 + 판단 + 행동 가이드를 제공한다.

[판단 규칙]
1. 항상 사용자 질문의 현재 상태를 최우선 기준으로 삼아라.
2. 증상이 경미하거나 (식욕/배변/활동 정상 등) 특별한 이상이 없는 경우:
   - 질병 가능성을 과도하게 강조하지 말고
   - 생리적/일반적 원인을 먼저 설명해라.
3. 반대로 위험 가능성이 높은 상황(아주 어린 나이, 급성 증상, 전염 가능성, 급격한 변화)이라면:
   - 일반 설명보다 병원 방문 필요성을 먼저 강조해라.
   - 왜 위험한지 짧게 설명해라.
4. 특정 증상 조합에서 현실적으로 흔하고 중요한 원인이 있으면:
   - 하나의 가능성으로 포함해라.
   - 단정하지 말고 가능성으로 표현해라.
5. 근거에 없는 정보라도 일반적인 수의학 상식 수준의 설명은 가능하다.
   - 단, 희귀 질환이나 과도한 추측은 금지한다.
6. 근거와 질문이 완전히 맞지 않으면:
   - 근거를 억지로 끼워 맞추지 말고
   - 질문 중심으로 상식 기반 설명을 우선해라.
7. 절대 금지:
   - 단정적 진단
   - 불필요하게 겁주는 표현
   - 질문과 무관한 내용 확장

[답변 구조]
1. 한줄 요약
2. 가장 가능성 높은 설명
3. 집에서 볼 체크포인트
4. 병원 가야 하는 경우

단, 위험도가 높은 상황이면 병원 관련 내용을 앞쪽에서 더 강하게 강조해도 된다.

[사용자 질문]
{user_input}

[검색된 근거]
{joined_context}

이제 위 규칙을 기반으로 현재 상황에 가장 맞는 현실적인 답변을 작성해라.
"""
    return prompt.strip()


def build_fallback_prompt(user_input):
    prompt = f"""
너는 반려견 건강 상담 보조 AI다.

현재 검색된 근거는 질문과 충분히 잘 맞지 않거나 신뢰도가 낮다.
따라서 검색 근거를 억지로 활용하지 말고, 일반적인 수의학 상식과 보호자 안내 관점에서 답변해라.

[판단 규칙]
1. 사용자 질문의 현재 상태를 최우선으로 해석해라.
2. 가장 흔하고 현실적인 원인부터 설명해라.
3. 단정적으로 진단하지 말고 가능성으로 설명해라.
4. 위험 신호가 있으면 병원 방문 필요성을 분명히 말해라.
5. 불필요하게 겁주지 말고, 실질적인 관찰 포인트를 제시해라.
6. 질문과 무관한 내용은 넣지 마라.

[답변 구조]
1. 한줄 요약
2. 가장 가능성 높은 설명
3. 집에서 볼 체크포인트
4. 병원 가야 하는 경우

[사용자 질문]
{user_input}

이제 위 규칙을 기반으로 현재 상황에 맞는 현실적인 답변을 작성해라.
"""
    return prompt.strip()


import google.generativeai as genai
import streamlit as st

# 프롬프트 빌더 함수들은 그대로 유지 (동일하게 사용 가능)
def build_chatbot_prompt(user_input, contexts):
    # ... (기존 코드와 동일) ...
    return prompt.strip()

def build_fallback_prompt(user_input):
    # ... (기존 코드와 동일) ...
    return prompt.strip()

# --- 핵심: Gemini 호출 함수 ---
def ask_gemini(prompt):
    """
    Ollama 대신 Gemini API를 호출합니다.
    Streamlit Secrets에서 GEMINI_API_KEY를 가져와야 합니다.
    """
    try:
        # API 키 설정
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 답변 생성
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        else:
            return "답변을 생성할 수 없습니다. 다시 시도해 주세요."
            
    except Exception as e:
        raise Exception(f"Gemini API 호출 중 에러 발생: {str(e)}")

# --- 서비스 인터페이스 함수 ---
def generate_chatbot_answer(user_input, contexts):
    try:
        prompt = build_chatbot_prompt(user_input, contexts)
        answer = ask_gemini(prompt)  # ask_ollama -> ask_gemini
        return {
            "answer": answer,
            "source": "Gemini-2.5-Flash",
            "error": None
        }
    except Exception as e:
        return {
            "answer": "챗봇 답변 생성 중 오류가 발생했습니다.",
            "source": "context_error",
            "error": str(e)
        }

def generate_fallback_answer(user_input):
    try:
        prompt = build_fallback_prompt(user_input)
        answer = ask_gemini(prompt)  # ask_ollama -> ask_gemini
        return {
            "answer": answer,
            "source": "fallback-Gemini",
            "error": None
        }
    except Exception as e:
        return {
            "answer": "일반 상식 답변 생성 중 오류가 발생했습니다.",
            "source": "fallback_error",
            "error": str(e)
        }