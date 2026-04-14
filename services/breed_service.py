import tensorflow as tf
import streamlit as st

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

@st.cache_resource
def load_breed_model():
    m = tf.keras.models.load_model('models/breed/dog_breed_classifier_final.keras')
    m.build((None, 224, 224, 3))
    _ = m(tf.zeros((1, 224, 224, 3)))
    return m
@st.cache_resource
def get_models():
    model = load_breed_model()

    try:
        base_model = model.get_layer('efficientnetb0')
        feature_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )
    except:
        feature_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=tf.keras.layers.GlobalAveragePooling2D()(
                base_model.get_layer('top_activation').output
            )
        )

    return model, feature_model
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