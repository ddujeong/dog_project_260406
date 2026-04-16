from services.qa_retrieval_service import UnifiedRetriever
from services.semantic_reranker import SemanticReranker
from services.chatbot_llm_service import (
    generate_chatbot_answer,
    generate_fallback_answer,
)
import os

class ChatbotService:
    def __init__(self):
        # 현재 파일의 절대 경로를 기준으로 루트 디렉토리를 찾습니다.
        current_file_path = os.path.abspath(__file__) # services/chatbot_service.py
        services_dir = os.path.dirname(current_file_path) # services/
        base_dir = os.path.dirname(services_dir) # 프로젝트 root/

        # 아래와 같이 절대 경로를 완성합니다.
        data_path = os.path.join(base_dir, "data", "aihub_chat", "processed", "merged_dataset.json")
        
        # [중요] 파일이 정말 있는지 확인하는 로직 추가
        if not os.path.exists(data_path):
            # 만약 못 찾는다면 현재 위치에서 가능한 모든 경로를 디버깅용으로 출력
            raise FileNotFoundError(f"실제 경로에 파일이 없습니다: {data_path}")

        self.retriever = UnifiedRetriever(data_path)
        self.reranker = SemanticReranker()
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