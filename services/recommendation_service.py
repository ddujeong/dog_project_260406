import numpy as np
import requests
from PIL import Image
import io
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import streamlit as st

def extract_feature_vector(img_url, model):
    try:
        response = requests.get(img_url, timeout=3)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        vec = model.predict(img_array, verbose=0)[0]
        vec_flattened = vec.flatten() 
        return normalize([vec_flattened])[0]
    except:
        return None
@st.cache_data(show_spinner=False)
def get_cached_recommendations(items, user_feature, _feature_model):
    """
    UI 관련 인자(progress_bar 등)를 모두 제거하여 
    값(items, feature)이 같으면 즉시 결과를 반환하도록 함.
    앞에 '_'가 붙은 _feature_model은 캐시 체크 대상에서 제외됨.
    """
    recommendations = []
    u_feat = user_feature.reshape(1, -1)
    
    for item in items:
        try:
            res = requests.get(item['popfile1'], timeout=3)
            img = Image.open(io.BytesIO(res.content)).convert('RGB').resize((224, 224))
            img_arr = np.expand_dims(np.array(img).astype('float32'), axis=0)
            processed_img = preprocess_input(img_arr) 
            
            # 특징 추출
            dog_features = _feature_model.predict(processed_img, verbose=0)[0]
            d_feat = dog_features.flatten().reshape(1, -1)
            
            sim_val = cosine_similarity(u_feat, d_feat)[0][0]
            similarity = float(sim_val * 100)
            
            if similarity > 10: 
                recommendations.append({"item": item, "similarity": similarity})
        except:
            continue
            
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    return recommendations[:5]   