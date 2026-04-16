import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class UnifiedRetriever:
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # content 중심 검색
        self.contents = [item["content"] for item in self.data]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2
        )
        self.content_vectors = self.vectorizer.fit_transform(self.contents)

    def search(self, query, top_k=20):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.content_vectors)[0]

        results = []

        for idx, score in enumerate(similarities):
            item = self.data[idx]

            results.append({
                "score": float(score),
                "source_type": item.get("source_type"),
                "department": item.get("department"),
                "content": item.get("content"),
                "question": item.get("question")
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]