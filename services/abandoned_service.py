import os
from datetime import datetime, timedelta
import streamlit as st
import dotenv
import requests

dotenv.load_dotenv()


def get_live_abandoned_data(target_breed_name):
    url = "http://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"

    service_key = st.secrets.get("API_KEY")
    today = datetime.now()
    start_day = (today - timedelta(days=90)).strftime('%Y%m%d')
    today_str = today.strftime('%Y%m%d')

    params = {
        'serviceKey': service_key,
        'bgnde': start_day,
        'endde': today_str,
        'upkind': '417000',
        'state': 'protect',
        'pageNo': '1',
        'numOfRows': '100',
        '_type': 'json'
    }

    try:
        print("API_KEY:", service_key)
        response = requests.get(url, params=params, timeout=10)
        print("status:", response.status_code)
        if response.status_code == 200:
            data = response.json()
            body = data.get('response', {}).get('body', {})
            print("items raw:", body.get('items'))

            items_dict = body.get('items', {})

            if items_dict and 'item' in items_dict:
                item_list = items_dict['item']
                if not isinstance(item_list, list):
                    item_list = [item_list]

                filtered_dogs = []
                for dog in item_list:
                    if 'popfile1' not in dog or not dog['popfile1']:
                        continue

                    age_str = dog.get('age', '')
                    special_mark = dog.get('specialMark', '')

                    puppy_keywords = ['일령', '60일', '미만', '개월', '아기', '강아지']

                    is_too_young = (
                        any(kw in age_str for kw in puppy_keywords)
                        or any(kw in special_mark for kw in puppy_keywords)
                    )

                    if str(today.year) in age_str:
                        is_too_young = True

                    if is_too_young:
                        continue

                    filtered_dogs.append(dog)
                print("filtered_dogs count:", len(filtered_dogs))
                return filtered_dogs
    
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")

    return []