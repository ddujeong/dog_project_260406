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
# @st.cache_data(show_spinner=False)
def get_cached_recommendations(items, user_feature, _feature_model):
    recommendations = []

    print("items count:", len(items))
    print("user_feature is None:", user_feature is None)
    print("user_feature shape:", None if user_feature is None else user_feature.shape)
    print("feature_model output shape:", getattr(_feature_model, "output_shape", None))

    if user_feature is None:
        return []

    u_feat = user_feature.reshape(1, -1)

    for idx, item in enumerate(items):
        try:
            print(f"[{idx}] url:", item.get("popfile1"))

            res = requests.get(item['popfile1'], timeout=5)
            res.raise_for_status()

            img = Image.open(io.BytesIO(res.content)).convert('RGB').resize((224, 224))
            img_arr = np.expand_dims(np.array(img).astype('float32'), axis=0)
            processed_img = preprocess_input(img_arr)

            dog_features = _feature_model.predict(processed_img, verbose=0)[0]
            d_feat = normalize([dog_features.flatten()])[0].reshape(1, -1)

            print("u_feat shape:", u_feat.shape, "d_feat shape:", d_feat.shape)

            sim_val = cosine_similarity(u_feat, d_feat)[0][0]
            similarity = float(sim_val * 100)

            print("similarity:", similarity)

            if similarity > 0:
                recommendations.append({"item": item, "similarity": similarity})

        except Exception as e:
            print("recommendation error:", item.get("popfile1"), e)
            continue

    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    print("final recommendation count:", len(recommendations))
    print("top similarities:", [r["similarity"] for r in recommendations[:5]])

    return recommendations[:5]