from services.qa_retrieval_service import UnifiedRetriever
from services.semantic_reranker import SemanticReranker
from services.chatbot_llm_service import (
    generate_chatbot_answer,
    generate_fallback_answer,
)
import os

class ChatbotService:
    def __init__(self):
        # 1. 절대 경로 기준점 잡기
        current_file_path = os.path.abspath(__file__) 
        services_dir = os.path.dirname(current_file_path) 
        base_dir = os.path.dirname(services_dir) 

        # 2. 데이터셋 경로와 임베딩 경로 각각 설정
        data_path = os.path.join(base_dir, "data", "aihub_chat", "processed", "merged_dataset.json")
        # [수정 포인트] 임베딩 파일 경로 추가
        emb_path = os.path.join(base_dir, "data", "aihub_chat", "processed", "embeddings.npy")
        
        # 파일 체크
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터셋이 없습니다: {data_path}")
        if not os.path.exists(emb_path):
            print(f"⚠️ 임베딩 파일이 경로에 없습니다: {emb_path}")

        # 3. 리트리버 초기화
        self.retriever = UnifiedRetriever(data_path)
        
        # 4. [가장 중요] 리랭커 생성 시 emb_path를 꼭 전달하세요!
        self.reranker = SemanticReranker(embedding_path=emb_path)
        
        # 5. 인덱스 빌드 (이제 reranker 내부에 이미 로드된 임베딩이 있으므로 실시간 생성을 건너뜁니다)
        self.reranker.build_index(self.retriever.data)

    def get_context(self, user_input, top_k=5):
        candidates = self.retriever.search(user_input, top_k=20)
        results = self.reranker.rerank(user_input, candidates, top_k=top_k)
        return results

    def should_use_context(self, contexts):
        if not contexts:
            return False

        top_score = contexts[0].get("final_score", 0.0)
        avg_score = sum(c.get("final_score", 0.0) for c in contexts) / len(contexts)

        # 임시 기준
        if top_score < 0.35:
            return False
        if avg_score < 0.30:
            return False

        return True

    def answer(self, user_input):
        contexts = self.get_context(user_input)
        use_context = self.should_use_context(contexts)

        if use_context:
            llm_result = generate_chatbot_answer(user_input, contexts)
        else:
            llm_result = generate_fallback_answer(user_input)

        return {
            "answer": llm_result["answer"],
            "contexts": contexts,
            "used_context": use_context,
            "source": llm_result["source"],
            "error": llm_result["error"],
        }