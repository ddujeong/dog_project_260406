import streamlit as st
import pymysql
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import plotly.graph_objects as go
import requests
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from datetime import datetime
from pymysql.cursors import DictCursor

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
def get_heatmap_from_url(img_url, model, last_conv_layer):
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
# --- 모델 로드 및 빌드 ---
@st.cache_resource
def load_my_model():
    m = tf.keras.models.load_model('models/dog_breed_classifier_final.keras')
    # 명시적 빌드
    m.build((None, 224, 224, 3)) 
    dummy_input = tf.zeros((1, 224, 224, 3))
    _ = m(dummy_input)
    return m

model = load_my_model()
try:
    base_model = model.get_layer('efficientnetb0')
    # .output 대신 레이어를 직접 쌓아서 GAP를 추가
    feature_model = tf.keras.models.Model(
        inputs=base_model.input, 
        outputs=base_model.get_layer('avg_pool').output 
    )
except:
    # 만약 avg_pool 레이어가 없다면 수동으로 뭉개주기
    feature_model = tf.keras.models.Model(
        inputs=base_model.input, 
        outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.get_layer('top_activation').output)
    )
# --- 1. 페이지 설정 및 제목 ---
st.set_page_config(page_title="Dog-nostic AI", page_icon="🐾", layout="centered")
st.title("🐾 Dog-nostic: 견종 분석 & 건강 리포트")
st.write("강아지 사진을 올리면 AI가 분석하고 건강 정보를 알려드려요!")

# 이 순서가 모델 학습 시의 순서
classes = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 
           'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 
           'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 
           'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 
           'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 
           'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 
           'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 
           'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 
           'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 
           'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 
           'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 
           'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 
           'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 
           'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 
           'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 
           'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 
           'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 
           'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']
classes.sort()
# 120개 견종 한글 매핑
ko_breed_map = {
    'Affenpinscher': '아펜핀셔', 'Afghan_Hound': '아프간 하운드', 'African_Hunting_Dog': '리키온', 'Airedale': '에어데일 테리어',
    'American_Staffordshire_Terrier': '아메리칸 스태퍼드셔 테리어', 'Appenzeller': '아펜젤러 마운틴 독', 'Australian_Terrier': '오스트레일리안 테리어',
    'Basenji': '바센지', 'Basset': '바셋 하운드', 'Beagle': '비글', 'Bedlington_Terrier': '베들링턴 테리어', 'Bernese_Mountain_Dog': '버니즈 마운틴 독',
    'Black-And-Tan_Coonhound': '블랙 앤 탄 쿤하운드', 'Blenheim_Spaniel': '블레넘 스패니얼', 'Bloodhound': '블러드하운드', 'Bluetick': '블루틱 쿤하운드',
    'Border_Collie': '보더 콜리', 'Border_Terrier': '보더 테리어', 'Borzoi': '보르조이', 'Boston_Bull': '보스턴 테리어',
    'Bouvier_Des_Flandres': '부비에 데 플랑드르', 'Boxer': '복서', 'Brabancon_Griffon': '브뤼셀 그리폰', 'Briard': '브리어드',
    'Brittany_Spaniel': '브리타니 스패니얼', 'Bull_Mastiff': '불마스티프', 'Cairn': '케언 테리어', 'Cardigan': '카디건 웰시코기',
    'Chesapeake_Bay_Retriever': '체서피크 베이 리트리버', 'Chihuahua': '치와와', 'Chow': '차우차우', 'Clumber': '클럼버 스패니얼',
    'Cocker_Spaniel': '코커 스패니얼', 'Collie': '콜리', 'Curly-Coated_Retriever': '컬리 코티드 리트리버', 'Dandie_Dinmont': '단디 딘몬트 테리어',
    'Dhole': '돌(인도들개)', 'Dingo': '딩고', 'Doberman': '도베르만', 'English_Foxhound': '잉글리시 폭스하운드', 'English_Setter': '잉글리시 세터',
    'English_Springer': '잉글리시 스프링거 스패니얼', 'Entlebucher': '엔틀레부허 마운틴 독', 'Eskimo_Dog': '아메리칸 에스키모 독',
    'Flat-Coated_Retriever': '플랫 코티드 리트리버', 'French_Bulldog': '프렌치 불독', 'German_Shepherd': '저먼 셰퍼드',
    'German_Short-Haired_Pointer': '저먼 쇼트헤어드 포인터', 'Giant_Schnauzer': '자이언트 슈나우저', 'Golden_Retriever': '골든 리트리버',
    'Gordon_Setter': '고든 세터', 'Great_Dane': '그레이트 데인', 'Great_Pyrenees': '그레이트 피레니즈', 'Greater_Swiss_Mountain_Dog': '그레이터 스위스 마운틴 독',
    'Groenendael': '그로넨달', 'Ibizan_Hound': '이비잔 하운드', 'Irish_Setter': '아이리시 세터', 'Irish_Terrier': '아이리시 테리어',
    'Irish_Water_Spaniel': '아이리시 워터 스패니얼', 'Irish_Wolfhound': '아이리시 울프하운드', 'Italian_Greyhound': '이탈리안 그레이하운드',
    'Japanese_Spaniel': '제패니즈 친', 'Keeshond': '키스혼드', 'Kelpie': '켈피', 'Kerry_Blue_Terrier': '케리 블루 테리어', 'Komondor': '코몬도르',
    'Kuvasz': '쿠바스', 'Labrador_Retriever': '라브라도 리트리버', 'Lakeland_Terrier': '레이클랜드 테리어', 'Leonberg': '레온베르거',
    'Lhasa': '라사압소', 'Malamute': '알래스칸 말라뮤트', 'Malinois': '마리노아', 'Maltese_Dog': '말티즈', 'Mexican_Hairless': '멕시칸 헤어리스',
    'Miniature_Pinscher': '미니어처 핀셔', 'Miniature_Poodle': '미니어처 푸들', 'Miniature_Schnauzer': '미니어처 슈나우저', 'Newfoundland': '뉴펀들랜드',
    'Norfolk_Terrier': '노포크 테리어', 'Norwegian_Elkhound': '노르웨이지언 엘크하운드', 'Norwich_Terrier': '노리치 테리어', 'Old_English_Sheepdog': '올드 잉글리시 쉽독',
    'Otterhound': '오터하운드', 'Papillon': '파피용', 'Pekinese': '페키니즈', 'Pembroke': '웰시코기 펨브록', 'Pomeranian': '포메라니안',
    'Pug': '퍼그', 'Redbone': '레드본 쿤하운드', 'Rhodesian_Ridgeback': '로데지안 리지백', 'Rottweiler': '로트와일러', 'Saint_Bernard': '세인트 버나드',
    'Saluki': '살루키', 'Samoyed': '사모예드', 'Schipperke': '스키퍼키', 'Scotch_Terrier': '스코티시 테리어', 'Scottish_Deerhound': '스코티시 디어하운드',
    'Sealyham_Terrier': '실리함 테리어', 'Shetland_Sheepdog': '셔틀랜드 쉽독', 'Shih-Tzu': '시츄', 'Siberian_Husky': '시베리안 허스키',
    'Silky_Terrier': '실키 테리어', 'Soft-Coated_Wheaten_Terrier': '소프트 코티드 휘튼 테리어', 'Staffordshire_Bullterrier': '스태퍼드셔 불 테리어',
    'Standard_Poodle': '스탠다드 푸들', 'Standard_Schnauzer': '스탠다드 슈나우저', 'Sussex_Spaniel': '서섹스 스패니얼', 'Tibetan_Mastiff': '티베탄 마스티프',
    'Tibetan_Terrier': '티베탄 테리어', 'Toy_Poodle': '토이 푸들', 'Toy_Terrier': '토이 테리어', 'Vizsla': '비즐라', 'Walker_Hound': '트리잉 워커 쿤하운드',
    'Weimaraner': '바이마라너', 'Welsh_Springer_Spaniel': '웰시 스프링거 스패니얼', 'West_Highland_White_Terrier': '화이트 테리어',
    'Whippet': '휘핏', 'Wire-Haired_Fox_Terrier': '와이어 폭스 테리어', 'Yorkshire_Terrier': '요크셔 테리어'
}
name_map = {
    'pembroke': 'welsh_corgi',
    'maltese_dog': 'maltese',
    'german_shepherd': 'german_shepherd',
    'shih-tzu': 'shih_tzu',
    'staffordshire_bullterrier': 'staffordshire_bull_terrier',
    'boston_bull': 'boston_terrier',
    'brabancon_griffon': 'brussels_griffon',
    'pekinese': 'pekingese',
    'eskimo_dog': 'american_eskimo_dog',
    'dhole': 'africanis',
    'japanese_spaniel': 'japanese_chin',
    'cairn': 'cairn_terrier',
    'chow': 'chow_chow',
    'clumber': 'clumber_spaniel',
    'blenheim_spaniel': 'cocker_spaniel',
    'cocker_spaniel': 'cocker_spaniel',
    'flat-coated_retriever': 'flat-coated_retriever',
    'afghan_hound': 'afghan_hound'
}
@st.cache_resource
def get_connection():
    return pymysql.connect(
        host=st.secrets["DB_HOST"],
        port=int(st.secrets["DB_PORT"]),
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        db=st.secrets["DB_NAME"],
        charset="utf8mb4",
        cursorclass=DictCursor
    )

@st.cache_data(show_spinner=False)
def get_dog_info(breed_name):
    # 1. 이름 정규화
    formatted_name = breed_name.lower().replace(' ', '_')

    if formatted_name in name_map:
        formatted_name = name_map[formatted_name]

    conn = get_connection()
    conn.ping(reconnect=True)  # 연결 끊김 대비

    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM breed_full_data WHERE breed_name LIKE %s"
            cursor.execute(sql, (f"%{formatted_name}%",))
            result = cursor.fetchone()

            # 방어 로직
            if not result and len(formatted_name) > 3:
                short_name = f"%{formatted_name[:4]}%"
                cursor.execute(sql, (short_name,))
                result = cursor.fetchone()

            return result

    except Exception as e:
        st.error(f"DB 조회 오류: {e}")
        return None

    finally:
        conn.close()

@st.cache_data
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
def get_live_abandoned_data(target_breed_name):
    url = "http://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
    
    # [중요] 서비스키는 꼭 '일반 인증키(Decoding)'를 사용하세요!
    service_key = st.secrets['API_KEY']
    today_str = datetime.now().strftime('%Y%m%d')
    params = {
        'serviceKey': service_key,
        'bgnde': '20260101',    
        'endde': today_str ,  
        'upkind': '417000',   
        'state': 'protect',    
        'pageNo': '1',
        'numOfRows': '100',     
        '_type': 'json'       
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            body = data.get('response', {}).get('body', {})
            items_dict = body.get('items', {})
           
            if items_dict and 'item' in items_dict:
                item_list = items_dict['item']
                if not isinstance(item_list, list):
                    item_list = [item_list]
                    
                filtered_dogs = []
                for dog in item_list:
                    if 'popfile1' not in dog or not dog['popfile1']:
                        continue

                    # 2. 나이 정보 확인
                    age_str = dog.get('age', '')  
                    special_mark = dog.get('specialMark', '') # 특이사항에도 정보가 많음
                    
                    puppy_keywords = ['일령', '60일', '미만', '개월', '아기', '강아지']
                    
                    # 1. 나이 문자열이나 특이사항에 새끼 관련 단어가 있는지 확인
                    is_too_young = any(kw in age_str for kw in puppy_keywords) or \
                                any(kw in special_mark for kw in puppy_keywords)
                    
                    # 2. 연도 기반 필터링 보완 (올해 태어난 '추정' 개체들)
                    if "2026" in age_str:
                        is_too_young = True

                    if is_too_young:
                        continue # 너무 어린 친구들은 분석에서 제외
                        
                    filtered_dogs.append(dog)
                
                return filtered_dogs
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
    return []
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
def build_breed_message(b1, b2, p1, p2):
    gap = p1 - p2

    if p1 > 0.5 and gap > 0.15:
        return f"{b1} 특징이 매우 강하게 나타나며, {b2}의 일부 특징도 관찰됩니다."
    
    elif p1 > 0.4:
        return f"{b1}와 {b2}의 특징이 함께 나타나는 외형입니다."
    
    else:
        return f"{b1}, {b2}를 포함한 여러 견종 특징이 혼합된 형태로 보입니다."
def get_confidence(p1, p2):
    gap = p1 - p2
    if p1 > 0.6 and gap > 0.2:
        return "높음"
    elif p1 > 0.35:
        return "보통"
    else:
        return "낮음"
st.set_page_config(page_title="Dog-nostic AI", page_icon="🐾", layout="wide")

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
                                        dog_heatmap = get_heatmap_from_url(item['popfile1'], model, 'top_conv')
                                        st.write(generate_gradcam_reason(user_heatmap, dog_heatmap, similarity))
                    else:
                        st.info("비슷한 외형의 친구를 찾지 못했습니다.")
            else:
                st.warning("보호소 데이터를 가져올 수 없습니다.")

    st.divider()
    st.caption("제공되는 정보는 AI 분석 결과이며, 정확한 상태 확인은 전문가와 상담하시기 바랍니다.")