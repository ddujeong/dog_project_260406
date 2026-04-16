from services.qa_retrieval_service import UnifiedRetriever
from services.semantic_reranker import SemanticReranker
from services.chatbot_llm_service import (
    generate_chatbot_answer,
    generate_fallback_answer,
)


class ChatbotService:
    def __init__(self):
        self.retriever = UnifiedRetriever(
            "data/aihub_chat/processed/merged_dataset.json"
        )
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