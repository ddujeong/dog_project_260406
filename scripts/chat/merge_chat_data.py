import json
import os

TL_PATH = "data/aihub_chat/processed/tl_dataset.json"
TS_PATH = "data/aihub_chat/processed/ts_chunks.json"
OUTPUT_PATH = "data/aihub_chat/processed/merged_dataset.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    tl_data = load_json(TL_PATH)
    ts_data = load_json(TS_PATH)

    merged_data = tl_data + ts_data

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"완료: 총 {len(merged_data)}개 저장")
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()