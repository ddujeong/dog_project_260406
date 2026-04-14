import numpy as np
import requests
from PIL import Image
import io
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

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
def recommend_dogs(items, user_target_breed_idx, model, feature_model, user_feature, progress_bar, status_text):
    recommendations = []
    total = len(items)
    
    u_feat = user_feature.reshape(1, -1)
    
    for i, item in enumerate(items):
        if progress_bar:
            progress_bar.progress((i + 1) / total)
            status_text.caption(f"🔎 전국 보호소 분석 중... ({i+1}/{total})")
            
        try:
            res = requests.get(item['popfile1'], timeout=3)
            img = Image.open(io.BytesIO(res.content)).convert('RGB').resize((224, 224))
            img_arr = np.expand_dims(np.array(img).astype('float32'), axis=0)
            
            processed_img = preprocess_input(img_arr) 
            
            # 특징 추출
            dog_features = feature_model.predict(processed_img, verbose=0)[0]
            d_feat = dog_features.flatten().reshape(1, -1)
            
            #  둘 다 똑같이 normalize를 적용한 상태에서 계산
            sim_val = cosine_similarity(u_feat, d_feat)[0][0]
            
            # 코사인 유사도는 0~1 사이 값입니다. 
            similarity = float(sim_val * 100)
            
            if similarity > 10: 
                recommendations.append({"item": item, "similarity": similarity})
                
        except Exception as e:
            continue
            
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    return recommendations[:5]    