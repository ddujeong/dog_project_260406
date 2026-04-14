import streamlit as st
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.efficientnet import preprocess_input

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

        # [Tab 2: 건강 케어]
        with tab2:
            alpha = 1.5
            denom = (p1**alpha) + (p2**alpha)
            w1, w2 = (p1**alpha) / denom, (p2**alpha) / denom
            total_metrics = {"patella": 0, "hip": 0, "heart": 0, "skin": 0, "eye": 0, "special": 0}
            
            found_data = False
            for b_name, weight in [(breed1, w1), (breed2, w2)]:
                dog_data = get_dog_info(b_name)
                if dog_data:
                    found_data = True
                    for key in total_metrics.keys():
                        val = dog_data.get(key, 15)
                        total_metrics[key] += float(val if val is not None else 15.0) * weight

            if found_data:
                col_chart, col_guide = st.columns([1, 1])
                with col_chart:
                    st.markdown("#### 📊 유전적 리스크 분포")
                    categories = ["슬개골", "고관절", "심폐/호흡", "피부", "안구", "유전질환"]
                    scores = [int(total_metrics[k]) for k in ["patella", "hip", "heart", "skin", "eye", "special"]]
                    fig = go.Figure(go.Scatterpolar(
                        r=scores+[scores[0]], 
                        theta=categories+[categories[0]], 
                        fill='toself', 
                        line_color='#FF4B4B',
                        text=scores+[scores[0]],           
                        mode='markers+lines+text',         
                        textposition="top center"        
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col_guide:
                    st.markdown("#### 💡 케어 가이드 & 영양소")
                    all_notes, nutrients = set(), set()
                    if total_metrics["heart"] >= 60:
                        all_notes.add("🫁 **심폐/호흡기**: 리스크가 높으니 무더운 날 산책은 피해주세요.")
                        nutrients.add("💊 **코엔자임 Q10**: 심장 기능 강화")
                    if total_metrics["patella"] >= 60 or total_metrics["hip"] >= 60:
                        all_notes.add("🦴 **관절 관리**: 미끄럼 방지 매트와 체중 관리가 필수입니다.")
                        nutrients.add("💊 **글루코사민**: 관절 연골 보호")
                    if total_metrics["eye"] >= 60:
                        all_notes.add("👀 **안구 건강**: 정기적으로 눈 상태를 체크해주세요.")
                        nutrients.add("💊 **루테인**: 안구 기능 유지")
                    if total_metrics["skin"] >= 60:
                        all_notes.add("🧴 **피부/알러지**: 보습과 사료 성분에 신경 써주세요.")
                        nutrients.add("💊 **오메가-3**: 피부 염증 완화")

                    with st.container(border=True):
                        st.markdown("##### 🩺 필수 관리 포인트")
                        for note in sorted(list(all_notes)): st.write(note)
                        if nutrients:
                            st.divider()
                            st.markdown("##### 🥗 추천 영양 성분")
                            for n in sorted(list(nutrients)): st.write(n)
            else:
                st.error("데이터베이스에 해당 견종의 건강 정보가 아직 없습니다.")

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