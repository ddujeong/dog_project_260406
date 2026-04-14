import pandas as pd
from services.db_service import get_connection

# 1. CSV 데이터 로드
df = pd.read_csv('data/kaggle/dog_breeds.csv')


def calculate_scores(health_text):
    t = str(health_text).lower()

    # 1. 관절 (Joint)
    is_joint = any(word in t for word in ['hip', 'dysplasia', 'elbow', 'disc'])
    hip_score = 90 if is_joint else 15
    patella_score = 95 if 'patella' in t or 'knee' in t else (75 if is_joint else 15)

    # 2. 심폐/호흡 (Cardio-Respiratory)
    cardio_score = 90 if any(word in t for word in ['heart', 'breathing', 'respiratory']) else 15

    # 3. 안구 & 피부 (Eye & Skin)
    eye_score = 85 if 'eye' in t else 15
    skin_score = 85 if 'skin' in t or 'allergies' in t else 15

    special_keywords = [
        'cancer', 'epilepsy', 'diabetes', 'pancreatitis',
        'bladder', 'obesity', 'disc', 'ear'
    ]

    # 4. 기타 특수 질환 (Special)
    special_score = 90 if any(word in t for word in special_keywords) else 15

    return {
        'patella': patella_score,
        'hip': hip_score,
        'heart': cardio_score,
        'skin': skin_score,
        'eye': eye_score,
        'special': special_score
    }


def seed_breed_data():
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS breed_full_data (
                    breed_name VARCHAR(100) PRIMARY KEY,
                    origin VARCHAR(100),
                    lifespan VARCHAR(50),
                    traits TEXT,
                    health_txt TEXT,
                    patella INT,
                    hip INT,
                    heart INT,
                    skin INT,
                    eye INT,
                    special INT
                )
            """)

            sql = """
                INSERT INTO breed_full_data
                (breed_name, origin, lifespan, traits, health_txt, patella, hip, heart, skin, eye, special)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    origin=VALUES(origin),
                    lifespan=VALUES(lifespan),
                    traits=VALUES(traits),
                    health_txt=VALUES(health_txt),
                    patella=VALUES(patella),
                    hip=VALUES(hip),
                    heart=VALUES(heart),
                    skin=VALUES(skin),
                    eye=VALUES(eye),
                    special=VALUES(special)
            """

            for _, row in df.iterrows():
                b_name = str(row['Breed']).lower().replace(' ', '_')
                h_text = row['Common Health Problems']
                scores = calculate_scores(h_text)

                cursor.execute(sql, (
                    b_name,
                    row['Country of Origin'],
                    row['Longevity (yrs)'],
                    row['Character Traits'],
                    h_text,
                    scores['patella'],
                    scores['hip'],
                    scores['heart'],
                    scores['skin'],
                    scores['eye'],
                    scores['special']
                ))

        conn.commit()
        print(f"✅ {len(df)}종 데이터 적재 완료")

    except Exception as e:
        conn.rollback()
        print(f"❌ 작업 중 에러 발생: {e}")

    finally:
        conn.close()


if __name__ == "__main__":
    seed_breed_data()