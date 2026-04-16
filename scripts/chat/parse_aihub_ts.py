import json
import os

INPUT_DIR = "data/aihub_chat/raw"
OUTPUT_PATH = "data/aihub_chat/processed/ts_chunks.json"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r", "").strip()


def split_ts_text(text: str):
    text = clean_text(text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for p in paragraphs:
        if len(p) <= 500:
            chunks.append(p)
        else:
            for i in range(0, len(p), 300):
                chunk = p[i:i+300].strip()
                if chunk:
                    chunks.append(chunk)

    return chunks


def load_json_files(folder_path):
    all_data = []

    for root, _, files in os.walk(folder_path):
        # ts_ 폴더만 처리
        if "ts_" not in os.path.basename(root).lower():
            continue

        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)

                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)

                    except Exception as e:
                        print(f"[ERROR] {file_path}: {e}")

    return all_data


def transform(data):
    processed = []

    for item in data:
        try:
            department = item.get("department", "")
            title = item.get("title", "")
            source_text = item.get("disease", "")

            chunks = split_ts_text(source_text)

            for idx, chunk in enumerate(chunks):
                processed.append({
                    "source_type": "ts",
                    "department": department,
                    "lifeCycle": None,
                    "disease": title,
                    "question": None,
                    "content": chunk,
                    "chunk_index": idx
                })
        except Exception:
            continue

    return processed


def main():
    raw_data = load_json_files(INPUT_DIR)
    processed_data = transform(raw_data)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"완료: {len(processed_data)}개 저장")
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()