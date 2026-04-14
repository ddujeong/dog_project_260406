import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import io
import streamlit as st
from tensorflow.keras.applications.efficientnet import preprocess_input

def make_gradcam_heatmap(img_array, model):
    try:
        # 1. 하위 모델(EfficientNet) 추출
        try:
            base_model = model.get_layer('efficientnetb0')
        except:
            base_model = [l for l in model.layers if 'efficient' in l.name.lower()][0]

        target_layer = base_model.get_layer('top_conv')
        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[target_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_array, tf.float32)
            conv_outputs, base_preds = grad_model(img_tensor)
            
            # 3. 상위 모델 레이어(Dense 등) 통과
            x = base_preds
            for i in range(model.layers.index(base_model) + 1, len(model.layers)):
                x = model.layers[i](x)
            
            predictions = x
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        # 4. Gradient 계산
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        print("GradCAM Error:", e)
        return None
def analyze_heatmap_region(heatmap):
    h, w = heatmap.shape

    # 영역 나누기 (3x3 grid)
    regions = {
        "top_left": heatmap[:h//3, :w//3],
        "top_center": heatmap[:h//3, w//3:2*w//3],
        "top_right": heatmap[:h//3, 2*w//3:],
        "mid_left": heatmap[h//3:2*h//3, :w//3],
        "center": heatmap[h//3:2*h//3, w//3:2*w//3],
        "mid_right": heatmap[h//3:2*h//3, 2*w//3:],
        "bottom_left": heatmap[2*h//3:, :w//3],
        "bottom_center": heatmap[2*h//3:, w//3:2*w//3],
        "bottom_right": heatmap[2*h//3:, 2*w//3:]
    }

    region_scores = {k: np.mean(v) for k, v in regions.items()}
    top_region = max(region_scores, key=region_scores.get)

    return top_region
def region_to_text(region):
    mapping = {
        "top_center": "귀와 머리 부분",
        "center": "눈과 코 주변",
        "mid_left": "얼굴 윤곽",
        "mid_right": "얼굴 윤곽",
        "bottom_center": "입과 턱 부분",
        "top_left": "귀 주변",
        "top_right": "귀 주변"
    }
    return mapping.get(region, "전체적인 외형")
def get_top_regions(heatmap, k=2):
    h, w = heatmap.shape

    regions = {
        "top_left": heatmap[:h//3, :w//3],
        "top_center": heatmap[:h//3, w//3:2*w//3],
        "top_right": heatmap[:h//3, 2*w//3:],
        "center": heatmap[h//3:2*h//3, w//3:2*w//3],
        "bottom_center": heatmap[2*h//3:, w//3:2*w//3]
    }

    scores = {k: np.mean(v) for k, v in regions.items()}
    return sorted(scores, key=scores.get, reverse=True)[:k]
def generate_gradcam_reason(user_heatmap, dog_heatmap, similarity):
    
    user_regions = get_top_regions(user_heatmap)
    dog_regions = get_top_regions(dog_heatmap)

    common = set(user_regions) & set(dog_regions)

    user_mask = user_heatmap > np.percentile(user_heatmap, 75)
    dog_mask = dog_heatmap > np.percentile(dog_heatmap, 75)

    overlap_score = np.mean(user_mask & dog_mask)

    if common:
        regions_text = ", ".join([region_to_text(r) for r in common])
        focus_text = f"{regions_text} 중심 특징이 유사합니다"
    else:
        focus_text = "전체적인 외형 패턴이 유사합니다"
    
    # 🔥 추가
    if overlap_score > 0.3:
        focus_text += " (핵심 영역 집중도도 유사)"
    # 🔥 similarity 결합
    if similarity > 80:
        level_text = "외형 전반이 매우 유사한 개체입니다"
    elif similarity > 50:
        level_text = "주요 특징이 유사한 개체입니다"
    else:
        level_text = "일부 특징이 유사한 개체입니다"

    return f"{focus_text}. {level_text}."
@st.cache_data
def get_heatmap_from_url(img_url, model):
    try:
        response = requests.get(img_url, timeout=3)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = preprocess_input(img_array)

        # base_model = model.get_layer('efficientnetb0')
        heatmap = make_gradcam_heatmap(
            img_array,
            model 
        )

        return heatmap
    except:
        return None