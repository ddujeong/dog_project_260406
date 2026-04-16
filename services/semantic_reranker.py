from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticReranker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.doc_texts = None

    # 🔥 전체 문서 embedding 1회 생성
    def build_index(self, docs):
        self.doc_texts = [d["content"] for d in docs]

        self.doc_embeddings = self.model.encode(
            self.doc_texts,
            batch_size=32,
            show_progress_bar=True
        )

        # 🔥 content → index 매핑
        self.text_to_idx = {
            text: i for i, text in enumerate(self.doc_texts)
        }

    # 🔥 query만 매번 계산
    def rerank(self, query, docs, top_k=5):
        query_vec = self.model.encode([query])

        # 🔥 미리 만들어둔 embedding 사용
        indices = [self.text_to_idx[d["content"]] for d in docs]
        doc_vecs = self.doc_embeddings[indices]

        sims = cosine_similarity(query_vec, doc_vecs)[0]

        for i, d in enumerate(docs):
            d["semantic_score"] = float(sims[i])

        # score 결합
        alpha = 0.1
        beta = 0.9

        for d in docs:
            d["final_score"] = d["semantic_score"]

        docs = sorted(docs, key=lambda x: x["final_score"], reverse=True)
        return docs[:top_k]