from services.db_service import get_connection
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
        return None

    finally:
        conn.close()
