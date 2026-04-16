import streamlit as st
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.efficientnet import preprocess_input
# from services.body_data_service import load_body_dataframe, map_images_to_dataframe
from services.body_analysis_service import (
    to_numeric_body_df,
    add_body_features,
    add_body_vectors,
    add_body_type
)
from services.body_model_service import predict_body_type
from services.body_report_service import build_body_type_description
from services.analysis_summary_service import build_analysis_summary
from services.llm_report_service import generate_llm_body_report
from services.breed_service import (
    get_models,
    classes,
    ko_breed_map,
    build_breed_message,
    get_confidence,
)
from services.gradcam_service import (
    make_gradcam_heatmap,
    analyze_heatmap_region,
    region_to_text,
    get_heatmap_from_url,
    generate_gradcam_reason,
)
from services.health_service import get_dog_info
from services.abandoned_service import get_live_abandoned_data
from services.recommendation_service import get_cached_recommendations
from services.chatbot_service import ChatbotService
model, feature_model = get_models()
# --- Body 데이터 로드 ---
@st.cache_data
def load_body_data():
    base_dir = "data/aihub_body/sample"

    # df = load_body_dataframe(base_dir)
    # df = map_images_to_dataframe(df, base_dir)
    df = to_numeric_body_df(df)
    df = add_body_features(df)
    df = add_body_vectors(df)
    df = add_body_type(df)

    return df
@st.cache_resource
def get_chatbot():
    return ChatbotService()
# --- 1. 페이지 설정 및 제목 ---
st.set_page_config(page_title="Dog-nostic AI", page_icon="🐾", layout="centered")
st.title("🐾 Dog-nostic: 견종 분석 & 건강 리포트")
st.write("강아지 사진을 올리면 AI가 분석하고 건강 정보를 알려드려요!")

with st.sidebar:
    st.title("🐾 Dog-nostic")
    st.markdown("### 사진 업로드")
    uploaded_file = st.file_uploader("강아지 사진을 선택하세요...", type=["jpg", "jpeg", "png"])
    st.divider()
    st.info("AI 분석을 통해 견종 확인부터 건강 가이드, 유기견 매칭까지 한 번에 확인하세요!")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 견종 분석 리포트",
    "🧬 AI 체형 리포트",
    "🏠 닮은꼴 친구 찾기",
    "💬 건강 Q&A 챗봇"
])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_rgb = image.convert("RGB").resize((224, 224))
    img_array = preprocess_input(np.expand_dims(np.array(img_rgb).astype('float32'), axis=0))
    
    with st.spinner("AI가 사진을 정밀 분석 중입니다..."):
        predictions = model.predict(img_array)[0]
        user_feature_raw = feature_model.predict(img_array, verbose=0)[0]
        user_feature_flattened = user_feature_raw.flatten()
        user_feature = normalize([user_feature_flattened])[0]

    top1_idx = predictions.argsort()[-1]
    top1_prob = predictions[top1_idx]

    if top1_prob < 0.25:
        st.error("⚠️ 강아지 이미지가 아니거나 분석이 어려운 사진입니다.")
    else:
        # [Tab 1: 견종 분석]
        with tab1:
            col_img, col_res = st.columns([1, 1.2])
            with col_img:
                st.image(image, caption='분석 대상', use_container_width=True)
            
            with col_res:
                top2_indices = predictions.argsort()[-2:][::-1]
                p1, p2 = predictions[top2_indices[0]], predictions[top2_indices[1]]
                breed1, breed2 = classes[top2_indices[0]], classes[top2_indices[1]]
                d_name1 = ko_breed_map.get(breed1.replace('_', ' ').title(), breed1)
                d_name2 = ko_breed_map.get(breed2.replace('_', ' ').title(), breed2)

                st.subheader(f"분석 결과: {d_name1}")
                st.success(f"🐶 대표 특징: {d_name1} ({p1*100:.1f}%)")
                st.info(f"🔎 보조 특징: {d_name2} ({p2*100:.1f}%)")
                st.write(build_breed_message(d_name1, d_name2, p1, p2))
                st.write(f"📊 분석 신뢰도: **{get_confidence(p1, p2)}**")

            st.divider()
            with st.expander("👁️ AI의 시선 (Grad-CAM 시각화)", expanded=True):
                heatmap = make_gradcam_heatmap(img_array, model)
                if heatmap is not None:
                    user_heatmap = heatmap
                    img_cv = np.array(img_rgb)
                    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
                    heatmap_resized = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
                    
                    c1, c2 = st.columns(2)
                    c1.image(superimposed_img, caption='주요 분석 영역', use_container_width=True)
                    with c2:
                        region = analyze_heatmap_region(user_heatmap)
                        st.markdown("### 🧬 외형 특징 요약")
                        st.write(f"• 집중 분석 부위: **{region_to_text(region)}**")
                        st.write(f"• {d_name1}의 골격 및 패턴을 중점적으로 분석했습니다.")
        # [Tab 2: 체형 분석]
        with tab2:
            st.subheader("🧬 AI 체형 분석 리포트")

            body_result = predict_body_type(image)

            body_primary = body_result["primary"]
            body_secondary = body_result["secondary"]
            body_primary_prob = body_result["primary_prob"]
            body_secondary_prob = body_result["secondary_prob"]

            body_desc = build_body_type_description(body_primary, body_secondary)

            summary = build_analysis_summary(
                breed_primary=d_name1,
                breed_secondary=d_name2,
                breed_primary_prob=p1,
                breed_secondary_prob=p2,
                breed_confidence=get_confidence(p1, p2),
                body_primary=body_primary,
                body_secondary=body_secondary,
                body_primary_prob=body_primary_prob,
                body_secondary_prob=body_secondary_prob,
                heatmap_region_text=region_to_text(region),  # build_analysis_summary 내부에서 아직 필요하면 둬도 됨
            )

            llm_result = generate_llm_body_report(summary)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="입력 이미지", width="stretch")

            with col2:
                st.markdown("### 📊 체형 분석 결과")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.markdown(
                        f"""
                        <div style="background-color:#163b2e;padding:18px;border-radius:16px;">
                            <div style="font-size:14px;color:#8df0b8;">주요 체형</div>
                            <div style="font-size:28px;font-weight:700;color:white;">{body_primary}</div>
                            <div style="font-size:18px;color:#d8ffe7;">{body_primary_prob * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with result_col2:
                    st.markdown(
                        f"""
                        <div style="background-color:#1c3557;padding:18px;border-radius:16px;">
                            <div style="font-size:14px;color:#91c9ff;">보조 체형</div>
                            <div style="font-size:28px;font-weight:700;color:white;">{body_secondary}</div>
                            <div style="font-size:18px;color:#dceeff;">{body_secondary_prob * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("### 🧠 규칙 기반 해석")
                st.info(body_desc)

                # 체형 리포트에서는 visual_focus/heatmap 부위 설명 제거
                # st.markdown("### 👁️ AI가 주로 본 부위")
                # st.write(f"**{region_to_text(region)}** 중심으로 외형 특징을 분석했습니다.")

            st.divider()

            st.markdown("### 🤖 LLM 설명 리포트")
            st.markdown(llm_result["report"])

            if llm_result["source"] == "main":
                st.success(f"LLM 응답 경로: 메인 모델 ({llm_result['model']})")
            elif llm_result["source"] == "fallback_model":
                st.warning(f"LLM 응답 경로: fallback 모델 ({llm_result['model']})")
            else:
                st.error("LLM 응답 경로: 규칙 기반 fallback")
                if llm_result["error"]:
                    st.caption(f"오류 내용: {llm_result['error']}")

            st.caption(
                "본 리포트는 이미지 기반 AI 분석 결과를 바탕으로 생성된 참고용 설명입니다. "
                "의학적 진단이나 전문 상담을 대체하지 않습니다."
            )
        # [Tab 3: 유기견 매칭]
        with tab3:
            st.subheader(f"🏠 {d_name1} 닮은꼴 친구 찾기")
            with st.spinner("실시간 보호소 데이터를 조회 중입니다..."):
                live_items = get_live_abandoned_data(d_name1)

            if live_items:
                target_idx = -1
                for i, breed_en in enumerate(classes):
                    if ko_breed_map.get(breed_en.replace('_', ' ').title()) == d_name1:
                        target_idx = i
                        break
                if target_idx == -1: target_idx = top1_idx
                with st.spinner(f"AI가 보호소 아이들의 사진을 분석 중입니다..."):
                    status_text = st.empty()
                    my_bar = st.progress(0)
                    # sample_items = random.sample(live_items, min(50, len(live_items)))
                    # top_recommendations = recommend_dogs(sample_items, target_idx, model)
                    top_recommendations = get_cached_recommendations(live_items, user_feature, feature_model)
                    my_bar.progress(100) 
                    my_bar.empty() # 완료 후 제거
                    if top_recommendations:
                        for rec in top_recommendations:
                            item = rec['item']
                            similarity = rec['similarity']
                            
                            with st.container(border=True):
                                c1, c2 = st.columns([1, 1.5])
                                with c1:
                                    if similarity >= 80: st.success(f"🌟 강력 추천 ({similarity:.1f}%)")
                                    else: st.info(f"🐾 유사도: {similarity:.1f}%")
                                    st.image(item['popfile1'], use_container_width=True)
                                with c2:
                                    st.markdown(f"### {item['kindNm']}")
                                    st.write(f"📍 {item['careNm']} ({item['careTel']})")
                                    st.write(f"📝 {item['specialMark']}")
                                    with st.expander("AI 매칭 상세 사유"):
                                        dog_heatmap = get_heatmap_from_url(item['popfile1'], model)
                                        st.write(generate_gradcam_reason(user_heatmap, dog_heatmap, similarity))
                    else:
                        st.info("비슷한 외형의 친구를 찾지 못했습니다.")
            else:
                st.warning("보호소 데이터를 가져올 수 없습니다.")
# [Tab 4: 건강 Q&A 챗봇]
with tab4:
    st.subheader("🐶 반려견 건강 챗봇")

    chatbot = get_chatbot()

    user_input = st.text_input("궁금한 건강 관련 질문을 입력하세요:")

    if st.button("질문하기", key="ask_chatbot"):
        if user_input.strip():
            result = chatbot.answer(user_input)

            st.markdown("### 🤖 답변")
            st.write(result["answer"])

            if result["source"] == "fallback" and result["error"]:
                st.caption(f"LLM 오류: {result['error']}")

            with st.expander("📌 검색된 근거 보기"):
                for i, r in enumerate(result["contexts"], start=1):
                    st.markdown(
                        f"**{i}. [{r['source_type']}] score={r.get('final_score', r['score']):.3f}**"
                    )
                    if r["question"]:
                        st.write(f"Q: {r['question']}")
                    st.write(r["content"])
                    st.write("---")
        else:
            st.warning("질문을 입력해 주세요.")
            
    st.divider()
    st.caption("제공되는 정보는 AI 분석 결과이며, 정확한 상태 확인은 전문가와 상담하시기 바랍니다.")