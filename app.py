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
from services.recommendation_service import recommend_dogs

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
        tab1, tab2, tab3 = st.tabs(["🔍 견종 분석 리포트", "🩺 맞춤 건강 케어", "🏠 닮은꼴 친구 찾기"])

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
            st.subheader("🧬 체형 분석 리포트")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="입력 이미지", use_container_width=True)

            with col2:
                result = predict_body_type(image)

                primary = result["primary"]
                secondary = result["secondary"]

                p1 = result["primary_prob"]
                p2 = result["secondary_prob"]

                st.markdown("### 📊 분석 결과")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div style="background-color:#1f4f3f;padding:20px;border-radius:15px">
                        <h3 style="color:#6effa6">주요 체형</h3>
                        <h2>{primary.upper()}</h2>
                        <p>{p1*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="background-color:#1f3a5f;padding:20px;border-radius:15px">
                        <h3 style="color:#6ec1ff">보조 체형</h3>
                        <h2>{secondary.upper()}</h2>
                        <p>{p2*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                desc = build_body_type_description(primary, secondary)

                st.markdown("### 🧠 체형 해석")
                st.write(desc)

            st.divider()
            st.markdown("### 📈 체형 지표 분석")

            st.metric("체중/체고 비율", "0.12", "평균 수준")
            st.metric("흉곽 비율", "1.58", "높음")
            st.markdown("### 📌 분석 안내")
            st.caption(
                "본 결과는 이미지 기반 AI 추정이며, 실제 체형과 차이가 있을 수 있습니다. "
                "체형은 연속적인 특성을 가지므로 복합적인 특징이 함께 나타날 수 있습니다."
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
                    top_recommendations = recommend_dogs(
                        live_items, 
                        target_idx, 
                        model, 
                        feature_model, 
                        user_feature, 
                        my_bar, 
                        status_text
                    )
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

    st.divider()
    st.caption("제공되는 정보는 AI 분석 결과이며, 정확한 상태 확인은 전문가와 상담하시기 바랍니다.")